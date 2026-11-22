"""The elbow joint: the J4310 with the link on its output shaft, in joint units.

    from exo.joint import Joint

    with Joint(motor_id=0x05, feedback_id=0x15, log_name="my_test") as joint:  # on config.channel
        joint.resume()              # reuse this power session's homing
        joint.move_to(30.0)         # degrees, 0 = mid-stroke
        joint.hold_until_enter(30.0)

Units: degrees, deg/s and Nm at the joint. The motor is started when the `with` block is entered
and always disabled when it is left, also after an error or Ctrl-C. Every update() is logged as one
row of logs/<log_name>_<time>.csv.

Writing your own control loop:
    for t in joint.loop():                  # joint.config.rate_hz times per second
        joint.command_position(target_deg)  # or command_velocity / command_torque
        joint.update()                      # send, wait for the reply, log, check limits
        if t > 2.0:
            break

Before homing, angle 0 is wherever the joint was at start-up and there are no stop limits; after
homing (or resume()) angle 0 is mid-stroke and targets beyond the stops are refused.
"""

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from damiao_motor import KD_MAX

from .calibration import Calibration, CalibrationError
from .compensation import Compensation
from .config import JointConfig
from .logs import LOG_DIR, LOGGER
from .loop import RealtimeLoop, enter_pressed
from .motor import DamiaoMotor


class JointError(RuntimeError):
    """A move or homing step could not be completed safely; the motor is disabled on leaving the block."""


@dataclass
class PushResult:
    """What push_until_moves() found."""
    moved: bool  # True: the joint started moving before the torque reached its limit
    torque_nm: float  # the torque when it started moving (or the limit, if it never moved)
    angle_deg: float  # where it was; if it never moved, the mean over the capture window
    motor_rad: float  # the same in the motor's frame (for the calibration)


class Joint:
    def __init__(self, channel: Optional[str] = None, motor_id: int = 0x05, feedback_id: int = 0x15, *,
                 log_name: str = "joint", rate_hz: Optional[float] = None,
                 config: JointConfig = JointConfig(), log_dir: Optional[Path] = None) -> None:
        """
        Args:
            channel: CAN interface of the motor (default: config.channel).
            motor_id, feedback_id: CAN IDs of the motor.
            log_name: name of this run's log files (logs/<log_name>_<time>.*).
            rate_hz: control loop rate (default: config.rate_hz).
            config: joint parameters (exo/config.py).
        """
        self.config = replace(config, rate_hz=rate_hz) if rate_hz else config
        self.motor = DamiaoMotor(channel or self.config.channel, motor_id, feedback_id, rate_hz=self.config.rate_hz,
                                 torque_trip_nm=self.config.torque_trip_nm,
                                 can_timeout_units=self.config.timeout_unhomed)
        self.log_name = log_name
        self.log_dir = Path(log_dir) if log_dir is not None else LOG_DIR
        self.log_stem: Optional[Path] = None  # logs/<log_name>_<time>, set when the block is entered
        self.calibration: Optional[Calibration] = None  # set by homing or resume()
        self.compensation: Optional[Compensation] = None  # friction / gravity feedforward
        self.zero_motor_rad = 0.0  # joint angle 0 in the motor's frame
        # What the joint is doing, for the log
        self.phase = ""
        self.setpoint_deg = math.nan
        self.setpoint_vel_deg_s = math.nan
        self.lag_deg = math.nan

    def __enter__(self) -> "Joint":
        self.log_stem = self._start_log(self.log_name)  # first, so the motor's start-up messages are in the file
        try:
            self.motor.start()
        except BaseException:
            LOGGER.close()  # __exit__ does not run when entering fails
            raise
        self.zero_motor_rad = self.motor.position  # until homed: angle 0 = where the joint started
        self._track()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is KeyboardInterrupt:
            LOGGER.error("Stopped by Ctrl-C")
        elif exc_type is not None:
            LOGGER.error(f"ABORTED: {exc_type.__name__}: {exc}")
        try:
            self.motor.stop()
        finally:
            LOGGER.close()  # flushes the per-tick CSV
            LOGGER.info(f"Log: {LOGGER.csv_path}")

    # ------------------------------------------------------------------------------------------
    # State (joint units)
    # ------------------------------------------------------------------------------------------
    @property
    def angle_deg(self) -> float:
        return math.degrees(self.config.output_sign * (self.motor.position - self.zero_motor_rad))

    @property
    def velocity_deg_s(self) -> float:
        return math.degrees(self.config.output_sign * self.motor.velocity)

    @property
    def torque_nm(self) -> float:
        """Measured torque (the firmware's estimate from the motor current)."""
        return self.config.output_sign * self.motor.torque

    @property
    def fresh(self) -> bool:
        """True if the last update() received a new reply (False: the state is a repeat)."""
        return self.motor.fresh

    @property
    def sample_time(self) -> float:
        """time.monotonic() when the command answered by the current state was sent (~ when it was measured)."""
        return self.motor.sample_time

    @property
    def is_homed(self) -> bool:
        return self.calibration is not None

    # ------------------------------------------------------------------------------------------
    # Commands (take effect at the next update())
    # ------------------------------------------------------------------------------------------
    def loop(self) -> RealtimeLoop:
        """A fixed-rate loop at the joint's rate: `for t in joint.loop(): ...`"""
        return RealtimeLoop(self.config.rate_hz)

    def update(self) -> None:
        """Send the command, wait for the motor's reply, log this tick and check the stop limits."""
        self.motor.update()
        LOGGER.update()
        if self.is_homed:
            limit = self.calibration.limit_deg + self.config.range_margin_deg
            if abs(self.angle_deg) > limit:
                raise JointError(f"joint at {self.angle_deg:+.1f} deg, beyond the stops (+/-{limit:.1f} deg): "
                                 "the homing is wrong; home again")

    def idle(self) -> None:
        """Motor off (no torque)."""
        self.motor.idle()
        self.setpoint_deg = self.setpoint_vel_deg_s = math.nan

    def command_torque(self, torque_nm: float) -> None:
        """Pure torque [Nm]. 0 Nm leaves the joint free to be moved by hand."""
        self.motor.command_torque(self.config.output_sign * torque_nm)
        self.setpoint_deg = self.setpoint_vel_deg_s = math.nan

    def command_velocity(self, velocity_deg_s: float, kd: float, torque_nm: float = 0.0) -> None:
        """Damper toward a speed: torque = torque_nm + kd * (velocity - v), kd in Nm s/rad.
        When blocked, the motor pushes torque_nm + kd * velocity and no more."""
        s = self.config.output_sign
        self.motor.command_velocity(s * math.radians(velocity_deg_s), kd, s * torque_nm)
        self.setpoint_deg, self.setpoint_vel_deg_s = math.nan, velocity_deg_s

    def command_position(self, angle_deg: float, velocity_deg_s: float = 0.0, torque_nm: float = 0.0,
                         kp: Optional[float] = None, kd: Optional[float] = None) -> None:
        """Spring-damper toward angle_deg moving at velocity_deg_s, plus a feedforward torque:
        torque = torque_nm + kp * (angle - a) + kd * (velocity - v); kp in Nm/rad, kd in Nm s/rad
        (default: config.kp / config.kd)."""
        s = self.config.output_sign
        self.motor.command_position(self.zero_motor_rad + s * math.radians(angle_deg),
                                    s * math.radians(velocity_deg_s),
                                    self.config.kp if kp is None else kp, self.config.kd if kd is None else kd,
                                    s * torque_nm)
        self.setpoint_deg, self.setpoint_vel_deg_s = angle_deg, velocity_deg_s

    def compensation_torque(self, angle_deg: float, velocity_deg_s: float, error_deg: float = 0.0) -> float:
        """Feedforward [Nm] cancelling the identified gravity and (most of the) friction; 0 before
        homing has identified them. error_deg (target - angle) sets the friction push when holding still."""
        if self.compensation is None:
            return 0.0
        c = self.config
        return self.compensation.torque(angle_deg, velocity_deg_s, error_deg, c.friction_scale,
                                        c.friction_v_band_deg_s, c.friction_err_deg)

    # ------------------------------------------------------------------------------------------
    # Moves and holds (position control)
    # ------------------------------------------------------------------------------------------
    def move_to(self, target_deg: float, speed_deg_s: Optional[float] = None, accel_deg_s2: Optional[float] = None,
                phase: str = "move", on_tick: Optional[Callable[[], None]] = None) -> dict:
        """Move to target_deg along a smooth (trapezoidal speed) profile, then return holding it.
        on_tick() runs after every update (e.g. to record samples). Returns follow()'s summary."""
        c = self.config
        if self.is_homed:
            limit = self.calibration.limit_deg - c.soft_margin_deg
            if abs(target_deg) > limit:
                raise ValueError(f"target {target_deg:+.1f} deg is outside +/-{limit:.1f} deg")
        duration, reference = trapezoid(self.angle_deg, target_deg, speed_deg_s or c.move_speed_deg_s,
                                        accel_deg_s2 or c.move_accel_deg_s2)
        return self.follow(reference, duration, phase, c.settle_s, on_tick)

    def move_by(self, delta_deg: float, speed_deg_s: Optional[float] = None, phase: str = "move") -> dict:
        return self.move_to(self.angle_deg + delta_deg, speed_deg_s, phase=phase)

    def follow(self, reference: Callable[[float], tuple[float, float]], duration: float, phase: str = "follow",
               settle_s: float = 0.0, on_tick: Optional[Callable[[], None]] = None) -> dict:
        """Track reference(t) -> (angle_deg, velocity_deg_s) in position control for duration + settle_s.

        The reference depends on time only: if the joint lags behind (friction, gravity), the
        spring-damper pulls it back onto the plan. Aborts if it lags by more than track_tol_deg
        (obstacle). Returns {max_lag_deg, peak_torque_nm}."""
        c = self.config
        lead = c.target_lead_ticks / c.rate_hz
        self.phase = phase
        max_lag, peak_torque, over = 0.0, 0.0, 0
        loop = self.loop()
        for t in loop:
            target, velocity = reference(t + lead)
            self.command_position(target, velocity, self.compensation_torque(target, velocity, target - self.angle_deg))
            self.update()
            if on_tick is not None:
                on_tick()
            peak_torque = max(peak_torque, abs(self.torque_nm))
            if self.fresh:
                # Lag behind the plan at the moment this reply was measured.
                self.lag_deg = reference(self.sample_time - loop.start_time)[0] - self.angle_deg
                max_lag = max(max_lag, abs(self.lag_deg))
                over = over + 1 if abs(self.lag_deg) > c.track_tol_deg else 0
                if over >= c.track_confirm:
                    raise JointError(f"the joint cannot follow ({self.lag_deg:+.1f} deg behind at "
                                     f"{self.angle_deg:+.1f} deg): obstacle?")
            if t >= duration + settle_s:
                break
        # LOGGER.flush_buffer()
        return {"max_lag_deg": max_lag, "peak_torque_nm": peak_torque}

    def hold(self, angle_deg: float, seconds: float, phase: str = "hold") -> None:
        """Hold angle_deg for `seconds`."""
        self.phase = phase
        for t in self.loop():
            self._hold_step(angle_deg)
            if t >= seconds:
                break
        # LOGGER.flush_buffer()

    def hold_until_enter(self, angle_deg: float, prompt: str = "", phase: str = "wait") -> None:
        """Hold angle_deg, showing angle and torque, until Enter is pressed."""
        print(prompt or f"Holding {angle_deg:+.1f} deg. Press Enter to continue (Ctrl-C to abort).")
        self.phase = phase
        for i, _ in enumerate(self.loop()):
            self._hold_step(angle_deg)
            if enter_pressed():
                break
            if i % 20 == 0:
                print(f"  angle {self.angle_deg:+7.2f} deg | torque {self.torque_nm:+.3f} Nm", end="\r")
        print()
        # LOGGER.flush_buffer()

    def measure_holding_torque(self, seconds: float = 0.5, torque_nm: float = 0.0) -> float:
        """Hold the current angle for `seconds` and return the mean measured torque over the second
        half [Nm]: the torque that keeps the joint there (gravity, plus any friction still loaded)."""
        target, torques = self.angle_deg, []
        self.phase = "measure hold"
        for t in self.loop():
            self.command_position(target, 0.0, torque_nm)
            self.update()
            if t > seconds / 2 and self.fresh:
                torques.append(self.torque_nm)
            if t >= seconds:
                break
        return statistics.fmean(torques)

    def _hold_step(self, angle_deg: float) -> None:
        self.command_position(angle_deg, 0.0, self.compensation_torque(angle_deg, 0.0, angle_deg - self.angle_deg))
        self.update()
        self.lag_deg = angle_deg - self.angle_deg

    # ------------------------------------------------------------------------------------------
    # Building blocks for finding the stops (velocity control: the push is limited by design)
    # ------------------------------------------------------------------------------------------
    def drive_until_stall(self, direction: int, speed_deg_s: float, push_nm: float, max_travel_deg: float,
                          accel_deg_s2: float = 20.0, stall_time_s: float = 0.15, min_speed_fraction: float = 0.2,
                          phase: str = "drive") -> float:
        """Drive toward `direction` (+1 / -1) at speed_deg_s until the joint stops making progress.

        Velocity control: torque = ff + kd * (v_des - v), so the motor pushes only as hard as needed
        to keep moving, and at most push_nm once blocked. The heavier the load, the slower it goes
        (at a load L it moves at about speed * (1 - L / push_nm)). Stalled = slower than
        min_speed_fraction of the commanded speed over stall_time_s, once at full speed: blocked,
        or the push is too weak for the load there. Returns the distance travelled [deg].
        Raises if it travels max_travel_deg without stalling."""
        rate = self.config.rate_hz
        kd = min(KD_MAX, push_nm / math.radians(speed_deg_s))
        ff = push_nm - kd * math.radians(speed_deg_s)  # only nonzero if kd hit its maximum
        start, v_cmd = self.angle_deg, 0.0
        recent: deque = deque(maxlen=max(2, round(stall_time_s * rate)))
        overspeed = 0
        stall_move_deg = min_speed_fraction * speed_deg_s * stall_time_s  # forward progress needed per window
        timeout = max_travel_deg / (min_speed_fraction * speed_deg_s) + 5.0
        self.phase = phase
        for t in self.loop():
            v_cmd = min(speed_deg_s, v_cmd + accel_deg_s2 / rate)  # soft start
            self.command_velocity(direction * v_cmd, kd, direction * ff)
            self.update()
            travel = direction * (self.angle_deg - start)
            if travel > max_travel_deg:
                raise JointError(f"{phase}: no stop within {max_travel_deg:.0f} deg of travel")
            if not self.fresh:
                continue
            # Gravity can pull the joint faster than commanded; stop it running away.
            overspeed = overspeed + 1 if abs(self.velocity_deg_s) > max(3 * speed_deg_s, 45.0) else 0
            if overspeed >= 5:
                raise JointError(f"{phase}: moving at {self.velocity_deg_s:+.0f} deg/s, far above the "
                                 f"{speed_deg_s:g} deg/s commanded (gravity?)")
            if v_cmd >= speed_deg_s:  # the stall check only counts once the full push is available
                recent.append(self.angle_deg)
                if len(recent) == recent.maxlen and direction * (recent[-1] - recent[0]) < stall_move_deg:
                    return travel
            if t > timeout:
                raise JointError(f"{phase}: timed out after {travel:.1f} deg")
        raise AssertionError("unreachable")

    def push_until_moves(self, direction: int, start_nm: float, max_nm: float, rate_nm_s: float = 0.25,
                         move_deg: float = 1.0, damping: float = 0.5, capture_s: float = 0.3,
                         phase: str = "push") -> PushResult:
        """Push in `direction` with a slowly rising torque until the joint moves.

        The torque starts at start_nm (signed) and rises by rate_nm_s per second toward
        direction * max_nm. It is applied through a damper (torque - damping * v, damping in
        Nm s/rad), so once the joint breaks free it cannot speed away. Ends:
            - moved: the joint moved move_deg in `direction` (more than a stop compresses when pushed
              hard). Returns the torque at that moment (the breakaway torque). The push stays on
              until the next command.
            - not moved: the torque reached max_nm. It is kept for capture_s and the angle averaged:
              the joint is against a hard stop."""
        self.phase = phase
        start = self.angle_deg
        captured_deg, captured_rad = [], []
        for t in self.loop():
            torque = start_nm + direction * rate_nm_s * t
            at_limit = direction * torque >= max_nm
            if at_limit:
                torque = direction * max_nm
            self.command_velocity(0.0, damping, torque)
            self.update()
            moved = direction * (self.angle_deg - start)
            if moved > move_deg:
                return PushResult(True, torque, self.angle_deg, self.motor.position)
            if moved < -3 * move_deg:
                raise JointError(f"{phase}: the joint moved the wrong way ({moved:+.2f} deg) at {torque:+.2f} Nm")
            if at_limit:
                if self.fresh:
                    captured_deg.append(self.angle_deg)
                    captured_rad.append(self.motor.position)
                if len(captured_deg) >= capture_s * self.config.rate_hz:
                    return PushResult(False, torque, statistics.fmean(captured_deg), statistics.fmean(captured_rad))
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------------------------------
    # Homing session
    # ------------------------------------------------------------------------------------------
    def start_session(self, calibration: Calibration) -> None:
        """Make mid-stroke angle 0, use the calibration's friction / gravity values, and mark the
        motor as homed for this power session (until it is switched off)."""
        self.zero_motor_rad = calibration.mid_motor_rad
        self.compensation = Compensation(**calibration.compensation) if calibration.compensation else None
        self.motor.set_can_timeout_units(self.config.timeout_homed)
        self.calibration = calibration

    def resume(self) -> Calibration:
        """Reuse the homing done since the motor was powered up (for test scripts after homing)."""
        cal = Calibration.load()
        if self.motor.timeout_at_start != self.config.timeout_homed:
            raise CalibrationError("the motor was not homed since it was powered up; run homing first "
                                   "(pixi run home-auto or home-manual)")
        for rid, value in cal.angle_offsets.items():
            now = self.motor.angle_offsets.get(rid)
            if now is not None and not math.isclose(now, value, abs_tol=1e-4):
                raise CalibrationError(f"motor zero offset (register {rid}) changed from {value} to {now}; home again")
        self.start_session(cal)
        if abs(self.angle_deg) > cal.limit_deg + self.config.range_margin_deg:
            raise CalibrationError(f"joint at {self.angle_deg:+.1f} deg, beyond the stops; home again")
        LOGGER.info(f"Resumed homing from {cal.created} ({cal.method}): joint at {self.angle_deg:+.2f} deg, "
                    f"stops at +/-{cal.limit_deg:.1f} deg, "
                    f"{'with' if self.compensation else 'without'} friction/gravity compensation")
        return cal

    # ------------------------------------------------------------------------------------------
    # Logging (one CSV row per update)
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _start_log(name: str) -> Path:
        """Messages and the per-tick CSV to logs/<name>_<date>_<time>.log / .csv (with _2, _3, ...
        if two runs start within the same second). Returns the path without extension."""
        LOGGER.reset()  # one shared logger: drop what an earlier run in this Python session tracked
        ts = f"{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = self.log_dir / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.set_log_path(run_dir)
        run = base = f"{name}_{ts}"
        n = 2
        while (run_dir / f"{run}.log").exists():
            run, n = f"{base}_{n}", n + 1
        LOGGER.set_file_name(run)
        return run_dir / run

    def _track(self) -> None:
        """The CSV columns, logged on every update()."""
        columns = self._log_columns()
        LOGGER.track_function(list(columns.values()), list(columns))

    def _log_columns(self) -> dict:
        m, t0 = self.motor, time.monotonic()
        return {
            "t_s": lambda: time.monotonic() - t0,
            "tick_ms": lambda: m.update_interval * 1e3,  # time between update() calls
            "rtt_ms": lambda: m.rtt * 1e3,  # command -> reply (empty: no reply in time)
            "fresh": lambda: int(m.fresh),
            "phase": lambda: self.phase,
            "mode": lambda: m.mode,
            "setpoint_deg": lambda: self.setpoint_deg,
            "setpoint_vel_deg_s": lambda: self.setpoint_vel_deg_s,
            "lag_deg": lambda: self.lag_deg,
            "angle_deg": lambda: self.angle_deg,
            "velocity_deg_s": lambda: self.velocity_deg_s,
            "torque_nm": lambda: self.torque_nm,  # measured
            "tau_ff_nm": lambda: self.config.output_sign * m.commanded_torque,  # feedforward sent
            "tau_cmd_nm": lambda: self.config.output_sign * m.commanded_total_torque,  # whole command
            "motor_rad": lambda: m.position,
            "t_mos_c": lambda: m.mos_temperature,
            "t_rotor_c": lambda: m.rotor_temperature,
            "status": lambda: m.status,
        }


def trapezoid(start_deg: float, end_deg: float, speed: float, accel: float):
    """Rest-to-rest move profile: accelerate, cruise at `speed`, decelerate (a triangle if too short
    to reach `speed`). Returns (duration, reference) with reference(t) -> (angle, velocity)."""
    d, sign = abs(end_deg - start_deg), math.copysign(1.0, end_deg - start_deg)
    t_acc = speed / accel
    if accel * t_acc ** 2 > d:
        t_acc = math.sqrt(d / accel)
    v_peak = accel * t_acc
    t_flat = (d - accel * t_acc ** 2) / v_peak if v_peak > 0 else 0.0
    duration = 2 * t_acc + t_flat

    def reference(t: float) -> tuple[float, float]:
        t = min(max(t, 0.0), duration)
        if t < t_acc:
            p, v = 0.5 * accel * t ** 2, accel * t
        elif t < t_acc + t_flat:
            p, v = 0.5 * accel * t_acc ** 2 + v_peak * (t - t_acc), v_peak
        else:
            rest = duration - t
            p, v = d - 0.5 * accel * rest ** 2, accel * rest
        return start_deg + sign * p, sign * v

    return duration, reference
