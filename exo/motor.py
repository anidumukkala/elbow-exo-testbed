"""Driver for the Damiao DM-J4310-2EC motor on CAN FD, built on the ``damiao-motor`` library.

You normally do not use this file directly: ``exo.joint.Joint`` wraps it in joint units (degrees,
zero = mid-stroke). Everything here is in the motor's own frame: rad, rad/s, Nm at the motor's
output shaft (after its internal 10:1 gearbox).

How the motor is driven:
    The firmware stays in MIT mode. Every command is one MIT frame with five numbers, and the
    firmware applies
        torque = tau_ff + kp * (p_des - p) + kd * (v_des - v)
    The three ways this script uses it:
        command_torque(tau)              kp = kd = 0: a pure torque (current) command
        command_velocity(v, kd, tau)     kp = 0: pushes toward speed v; blocked, it pushes tau + kd * |v|
        command_position(p, v, kp, kd)   spring-damper toward position p (impedance control)
    A command takes effect at the next ``update()``. Each ``update()`` sends exactly one frame and
    waits (about 1 ms) for the motor's reply; the motor only reports when it receives a command.

Safety (any of these disables the motor and latches it off until the program restarts):
    no reply for STALE_LIMIT_S, a fault reported by the motor, measured torque over ``torque_trip``,
    temperatures over the limits below, or a position too close to the frame's roll-over.
    The firmware's own CAN watchdog (register 9) also drops the motor if commands stop for ~100 ms.

Position frame:
    The firmware keeps a continuous multi-turn position while powered (register 80, p_m). At
    power-up it restarts at the output shaft's single-turn angle in (-pi, pi], so across a power
    cycle it shifts by a whole number of turns. Feedback frames carry it packed into +/-PMAX
    (12.5 rad) and roll over; ``position`` follows the roll-overs, so it always equals p_m.

CAN FD: the interface must be in FD mode (``pixi run can-up``) and the motor configured for it
(register 35, saved to flash; done once per motor).
"""

import json
import math
import subprocess
import threading
import time
from typing import Optional

import can
from damiao_motor import KD_MAX, KP_MAX, MOTOR_TYPE_PRESETS, REGISTER_TABLE, DaMiaoController
from damiao_motor.core.motor import DM_MOTOR_DISABLED, DM_MOTOR_ENABLED

from .logs import LOGGER

BUSTYPE = "socketcan"  # python-can interface type (SocketCAN on the Raspberry Pi)
MOTOR_TYPE = "4310"  # damiao-motor preset: PMAX 12.5 rad, VMAX 30 rad/s, TMAX 10 Nm
EXPECTED_GEAR_RATIO = 10.0  # the J4310's internal planetary gearbox (register 20)

MAX_MOS_TEMPERATURE_C = 80.0  # driver protection at 120 C
MAX_ROTOR_TEMPERATURE_C = 100.0  # manual: coil <= 100 C recommended

# Register IDs (names and descriptions in damiao_motor.REGISTER_TABLE)
REG_TIMEOUT = 9  # CAN watchdog, 50 us units (RAM; reverts to the flash value at power-up)
REG_CTRL_MODE = 10  # 1 = MIT
REG_SW_VER = 14
REG_GR = 20
REG_PMAX, REG_VMAX, REG_TMAX = 21, 22, 23
REG_ANGLE_OFFSETS = (54, 56)  # change when the motor zero is re-saved
REG_P_M = 80  # multi-turn position

STARTUP_REGISTERS = (REG_TIMEOUT, REG_CTRL_MODE, REG_SW_VER, REG_GR, REG_PMAX, REG_VMAX, REG_TMAX)

STALE_LIMIT_S = 0.025  # fault after this long without a reply
ENABLE_GRACE_UPDATES = 20  # replies still saying DISABLED are tolerated this long after enabling
REPLY_TIMEOUT_S = 0.003  # update() waits at most this long for the reply (and at most 60 % of a tick)
MIN_REPLY_S = 0.0001  # a frame received sooner after sending answers an earlier command
POSITION_MARGIN_RAD = 1.0  # position targets are kept this far inside +/-PMAX

ZERO_COMMAND = {"pos": 0.0, "vel": 0.0, "kp": 0.0, "kd": 0.0, "tau": 0.0}


class MotorFault(RuntimeError):
    """Motor fault, excessive torque or lost communication. The motor is disabled and latched off."""


def interface_is_fd(channel: str) -> bool:
    """True if the SocketCAN interface is configured for CAN FD."""
    out = subprocess.run(["ip", "-details", "-json", "link", "show", channel], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"no CAN interface {channel}: {out.stderr.strip()}")
    info = json.loads(out.stdout)[0].get("linkinfo", {}).get("info_data", {})
    return "FD" in info.get("ctrlmode", [])


class DamiaoMotor:
    """One DM-J4310-2EC on a CAN FD bus. Call start() first and stop() at the end (Joint does this)."""

    def __init__(self, channel: str = "can0", motor_id: int = 0x05, feedback_id: int = 0x15,
                 rate_hz: float = 200.0, torque_trip_nm: Optional[float] = None,
                 can_timeout_units: int = 2000) -> None:
        """
        Args:
            channel: CAN interface name.
            motor_id: command CAN ID (ESC_ID), 1-15.
            feedback_id: feedback CAN ID (MST_ID).
            rate_hz: how often update() will be called.
            torque_trip_nm: fault as soon as one reply reports more torque than this [Nm].
            can_timeout_units: firmware CAN watchdog written at start, 50 us units (2000 = 100 ms).
        """
        if not 1 <= motor_id <= 0x0F:
            raise ValueError(f"motor_id must be 1-15 (got {motor_id})")
        self.channel, self.motor_id, self.feedback_id = channel, motor_id, feedback_id
        self.torque_trip_nm = torque_trip_nm
        self.can_timeout_units = can_timeout_units
        self._stale_limit = max(1, round(STALE_LIMIT_S * rate_hz))
        self._reply_timeout = min(REPLY_TIMEOUT_S, 0.6 / rate_hz)
        self._reply_event = threading.Event()  # set by the receive thread after each feedback frame

        preset = MOTOR_TYPE_PRESETS[MOTOR_TYPE]
        self.pmax, self.vmax, self.tmax = preset["p_max"], preset["v_max"], preset["t_max"]
        self._controller: Optional[DaMiaoController] = None
        self._motor = None
        self.registers: dict[int, float] = {}
        self.timeout_at_start: Optional[int] = None  # CAN watchdog register as found at start

        self.mode = "idle"  # idle, torque, velocity or position
        self._cmd = dict(ZERO_COMMAND)
        self._enabled = False
        self._enable_grace = 0
        self.faulted = False
        self.fault_reason: Optional[str] = None

        # Latest state (from the reply to the last command)
        self.position = math.nan  # rad, unwrapped (= register p_m)
        self.velocity = math.nan  # rad/s
        self.torque = math.nan  # Nm, firmware estimate from the measured current
        self.mos_temperature = math.nan  # C
        self.rotor_temperature = math.nan  # C
        self.status = "UNKNOWN"
        self._status_code = -1
        self._raw_position = math.nan
        self._last_state = None
        self._stale = 0
        # Timing (time.monotonic seconds)
        self.fresh = False  # did the last update() get a new reply?
        self.sample_time = math.nan  # when the command answered by the current state was sent
        self.rtt = math.nan  # command -> reply of the last update (nan without reply)
        self.update_interval = math.nan  # between the last two update() calls
        self._last_send = math.nan
        self.n_updates = 0
        self.n_missed = 0

    # ------------------------------------------------------------------------------------------
    # Start / stop / update
    # ------------------------------------------------------------------------------------------
    def start(self) -> None:
        LOGGER.info(f"Connecting to motor 0x{self.motor_id:02X} on {self.channel} (CAN FD)")
        self._controller = DaMiaoController(channel=self.channel, bustype=BUSTYPE)
        try:
            self._use_can_fd()
            self._use_blocking_receive()
            self._controller.flush_bus()
            self._motor = self._controller.add_motor(motor_id=self.motor_id, feedback_id=self.feedback_id,
                                                     motor_type=MOTOR_TYPE)
            self._patch_feedback()
            self._motor.disable()  # known state before touching registers
            time.sleep(0.02)
            self._read_registers()
            self._ensure_mit_mode()
            self.timeout_at_start = int(self.registers[REG_TIMEOUT])
            self._motor.write_register(REG_TIMEOUT, self.can_timeout_units)
            time.sleep(0.05)  # register-write echoes arrive on the feedback ID; let them pass
            self._wait_first_feedback()
            self._anchor_to_p_m()
        except Exception:
            self._shutdown_bus()
            raise
        LOGGER.info(f"Motor ready: position {self.position:+.4f} rad, status {self.status}, "
                    f"MOS {self.mos_temperature:.0f} C, rotor {self.rotor_temperature:.0f} C")

    def stop(self) -> None:
        """Zero command, disable, close the bus. Safe to call more than once."""
        if self._motor is not None:
            try:
                if self._enabled and not self.faulted:
                    self._cmd = dict(ZERO_COMMAND)
                    self._send()
                self._motor.disable()
            except Exception as e:
                LOGGER.warning(f"disabling the motor at stop failed: {e}")
            self._enabled = False
        self._shutdown_bus()
        self.mode = "idle"
        LOGGER.info("Motor disabled")

    def update(self) -> None:
        """Send the staged command, wait for the reply, update the state and check it."""
        if self.faulted:
            raise MotorFault(f"latched fault: {self.fault_reason}")
        if self._motor is None:
            raise RuntimeError("update() called before start()")
        self._reply_event.clear()
        sent = time.monotonic()
        self._send()
        self.update_interval = sent - self._last_send
        self._last_send = sent
        self.n_updates += 1
        # Only a frame received after this command went out is its reply; an older one is a late
        # reply to an earlier command.
        state, deadline = None, sent + self._reply_timeout
        while True:
            candidate = self._motor.state
            rx = candidate.get("rx_time") if candidate else None
            if rx is not None and rx >= sent + MIN_REPLY_S and candidate is not self._last_state:
                state = candidate
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._reply_event.wait(remaining)
            self._reply_event.clear()

        if state is None:
            self.fresh, self.rtt = False, math.nan
            self._stale += 1
            self.n_missed += 1
            if self._stale >= self._stale_limit:
                self._trip(f"no reply for {self._stale} consecutive updates")
            return
        self.fresh, self.rtt, self.sample_time = True, state["rx_time"] - sent, sent
        self._stale = 0
        self._last_state = state
        self._ingest(state)
        self._check_health()

    # ------------------------------------------------------------------------------------------
    # Commands (take effect at the next update())
    # ------------------------------------------------------------------------------------------
    def idle(self) -> None:
        """Disable the motor (no torque)."""
        if self._enabled and self._motor is not None:
            self._motor.disable()
        self._enabled = False
        self.mode = "idle"
        self._cmd = dict(ZERO_COMMAND)

    def command_torque(self, torque: float) -> None:
        """Pure torque [Nm] (the firmware closes the current loop)."""
        self._activate("torque")
        self._cmd = {"pos": self._safe_position(), "vel": 0.0, "kp": 0.0, "kd": 0.0, "tau": self._limit(torque)}

    def command_velocity(self, velocity: float, kd: float, torque: float = 0.0) -> None:
        """Push toward `velocity` [rad/s] with damping kd [Nm s/rad] and feedforward torque [Nm]:
        torque = tau + kd * (velocity - v). Blocked (v = 0) it pushes tau + kd * velocity."""
        if not 0 < kd <= KD_MAX:
            raise ValueError(f"kd must be in (0, {KD_MAX}] (got {kd})")
        if not math.isfinite(velocity):
            raise ValueError(f"non-finite velocity command: {velocity}")
        self._activate("velocity")
        self._cmd = {"pos": self._safe_position(), "vel": max(-self.vmax, min(self.vmax, velocity)),
                     "kp": 0.0, "kd": kd, "tau": self._limit(torque)}

    def command_position(self, position: float, velocity: float, kp: float, kd: float, torque: float = 0.0) -> None:
        """Spring-damper toward position [rad] moving at velocity [rad/s]:
        torque = tau + kp * (position - p) + kd * (velocity - v)."""
        if not 0 <= kp <= KP_MAX or not 0 < kd <= KD_MAX:
            raise ValueError(f"kp must be in [0, {KP_MAX}] and kd in (0, {KD_MAX}]")
        limit = self.pmax - POSITION_MARGIN_RAD
        if not abs(position) <= limit:
            raise ValueError(f"position target {position:+.3f} rad is outside +/-{limit:.2f} rad")
        self._activate("position")
        self._cmd = {"pos": position, "vel": max(-self.vmax, min(self.vmax, velocity)), "kp": kp, "kd": kd,
                     "tau": self._limit(torque)}

    @property
    def commanded_torque(self) -> float:
        """Feedforward torque being sent [Nm] (0 when disabled)."""
        return self._cmd["tau"] if self._enabled else 0.0

    @property
    def commanded_total_torque(self) -> float:
        """The whole command tau + kp * (p_des - p) + kd * (v_des - v) at the last reply [Nm]."""
        if not self._enabled:
            return 0.0
        c = self._cmd
        tau = c["tau"] + c["kp"] * (c["pos"] - self.position) + c["kd"] * (c["vel"] - self.velocity)
        return max(-self.tmax, min(self.tmax, tau))

    # ------------------------------------------------------------------------------------------
    # Registers
    # ------------------------------------------------------------------------------------------
    def read_register(self, rid: int, timeout: float = 0.5) -> float:
        """Fresh register read (the library would otherwise return a cached value)."""
        with self._motor.registers_lock:
            self._motor.registers.pop(rid, None)
        return self._motor.get_register(rid, timeout=timeout)

    def set_can_timeout_units(self, units: int) -> None:
        """Write the CAN watchdog register (RAM only; reverts to the flash value at power-up)."""
        self._motor.write_register(REG_TIMEOUT, int(units))

    @property
    def angle_offsets(self) -> dict[str, float]:
        """Zero-offset registers (change if the motor zero is re-saved). Empty if unreadable."""
        return {str(r): float(self.registers[r]) for r in REG_ANGLE_OFFSETS if r in self.registers}

    # ------------------------------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------------------------------
    def _activate(self, mode: str) -> None:
        if self.faulted:
            raise MotorFault(f"latched fault: {self.fault_reason}")
        if not self._enabled:
            self._motor.enable()
            self._enabled = True
            self._enable_grace = ENABLE_GRACE_UPDATES
        self.mode = mode

    def _safe_position(self) -> float:
        # With kp = 0 the position field does nothing, but it must still fit in the frame.
        return max(-self.pmax, min(self.pmax, self.position)) if math.isfinite(self.position) else 0.0

    def _limit(self, torque: float) -> float:
        if not math.isfinite(torque):
            raise ValueError(f"non-finite torque command: {torque}")
        return max(-self.tmax, min(self.tmax, torque))

    def _send(self) -> None:
        c = self._cmd if self._enabled else ZERO_COMMAND
        self._motor.send_raw(self._motor.encode_cmd_msg(pos=c["pos"], vel=c["vel"], torq=c["tau"], kp=c["kp"], kd=c["kd"]))

    def _use_can_fd(self) -> None:
        """Reopen the library's bus with FD frames enabled and send every frame as FD with bit-rate
        switch (the library itself only knows classic frames)."""
        if BUSTYPE == "socketcan" and not interface_is_fd(self.channel):
            raise RuntimeError(f"{self.channel} is in classic CAN mode; run `pixi run can-up`")
        self._controller.bus.shutdown()
        bus = can.Bus(interface=BUSTYPE, channel=self.channel, fd=True)
        send = bus.send

        def send_fd(msg: can.Message, timeout: Optional[float] = None) -> None:
            msg.is_fd = True
            msg.bitrate_switch = True
            send(msg, timeout)

        bus.send = send_fd
        self._controller.bus = bus

    def _use_blocking_receive(self) -> None:
        """Replace the library's receive loop (drain, sleep 1 ms, repeat) with one that blocks in recv()
        and handles each frame as soon as it arrives: the 1 ms sleep added ~1 ms to every round trip."""
        controller = self._controller

        def receive_loop() -> None:
            while controller._polling_active:
                try:
                    msg = controller.bus.recv(timeout=0.02)
                except Exception as e:  # bus closed (shutdown) or failed
                    if controller._polling_active:
                        LOGGER.error(f"CAN receive thread stopped: {e!r}")
                    controller._polling_active = False
                    return
                if msg is None or len(msg.data) != 8:
                    continue
                motor = controller._motors_by_feedback.get(msg.data[0] & 0x0F)
                if motor is None:
                    continue
                try:
                    motor.process_feedback_frame(bytes(msg.data), arbitration_id=msg.arbitration_id)
                except Exception as e:
                    LOGGER.warning(f"CAN frame {bytes(msg.data).hex(' ')} not decoded: {e!r}")

        controller._polling_loop = receive_loop  # started by add_motor()

    def _patch_feedback(self) -> None:
        """Two fixes to how the library decodes received frames:
        - register write/store replies (D[2] = 0x55 / 0xAA) would be decoded as feedback (status
          DISABLED, bogus position); they update the register cache instead.
        - each feedback frame is stamped with its receive time (state["rx_time"]) and wakes update()."""
        motor = self._motor
        process, decode = motor.process_feedback_frame, motor.decode_sensor_feedback

        def process_frame(data: bytes, arbitration_id: Optional[int] = None) -> None:
            if len(data) == 8 and data[1] <= 0x0F and data[2] in (0x55, 0xAA) and data[3] in REGISTER_TABLE:
                if data[2] == 0x55:
                    motor.decode_register_reply(data, arbitration_id=arbitration_id)
                return
            process(data, arbitration_id=arbitration_id)
            self._reply_event.set()

        def decode_and_stamp(data: bytes, arbitration_id: Optional[int] = None) -> dict:
            received = time.monotonic()
            state = decode(data, arbitration_id=arbitration_id)
            state["rx_time"] = received
            return state

        motor.process_feedback_frame = process_frame
        motor.decode_sensor_feedback = decode_and_stamp

    def _read_registers(self) -> None:
        for rid in STARTUP_REGISTERS:
            try:
                self.registers[rid] = self._motor.get_register(rid, timeout=0.5)
            except TimeoutError as e:
                raise RuntimeError(
                    f"no reply reading register {rid} ({REGISTER_TABLE[rid].variable}) from motor "
                    f"0x{self.motor_id:02X} on {self.channel}: check power, wiring, IDs and that the motor "
                    "is configured for CAN FD") from e
        for rid in REG_ANGLE_OFFSETS:
            try:
                self.registers[rid] = self._motor.get_register(rid, timeout=0.2)
            except TimeoutError:
                LOGGER.warning(f"register {rid} ({REGISTER_TABLE[rid].variable}) not readable")
        # Frames must be scaled with the firmware's mapping ranges.
        pmax, vmax, tmax = (float(self.registers[r]) for r in (REG_PMAX, REG_VMAX, REG_TMAX))
        if (pmax, vmax, tmax) != (self.pmax, self.vmax, self.tmax):
            LOGGER.warning(f"firmware PMAX/VMAX/TMAX = {pmax}/{vmax}/{tmax}, expected "
                           f"{self.pmax}/{self.vmax}/{self.tmax}; using the firmware values")
            self._motor.set_p_limits(-pmax, pmax)
            self._motor.set_v_limits(-vmax, vmax)
            self._motor.set_t_limits(-tmax, tmax)
            self.pmax, self.vmax, self.tmax = pmax, vmax, tmax
        if not math.isclose(float(self.registers[REG_GR]), EXPECTED_GEAR_RATIO, rel_tol=1e-3):
            LOGGER.warning(f"gear ratio register Gr = {self.registers[REG_GR]}, expected {EXPECTED_GEAR_RATIO}")
        LOGGER.debug(f"registers: { {REGISTER_TABLE[r].variable: v for r, v in self.registers.items()} }")

    def _ensure_mit_mode(self) -> None:
        if int(self.registers[REG_CTRL_MODE]) == 1:
            return
        LOGGER.warning(f"control mode register = {self.registers[REG_CTRL_MODE]}, switching to MIT (1)")
        self._motor.write_register(REG_CTRL_MODE, 1)
        time.sleep(0.05)
        mode = int(self.read_register(REG_CTRL_MODE))
        if mode != 1:
            raise RuntimeError(f"could not switch the motor to MIT mode (register reads {mode})")
        self.registers[REG_CTRL_MODE] = mode

    def _wait_first_feedback(self, timeout: float = 0.5) -> None:
        before = self._motor.state
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._send()  # the motor is disabled, so this is a zero frame
            time.sleep(0.01)
            state = self._motor.state
            if state and state is not before:
                self._last_state = state
                self._ingest(state)
                return
        raise RuntimeError(f"no feedback frame from motor 0x{self.motor_id:02X} within {timeout} s")

    def _anchor_to_p_m(self) -> None:
        # Feedback only knows the position modulo the roll-over period; p_m knows the turn.
        p_m = float(self.read_register(REG_P_M))
        turns = round((p_m - self._raw_position) / self._wrap_period)
        self.position = self._raw_position + turns * self._wrap_period
        if abs(self.position - p_m) > 0.05:
            LOGGER.warning(f"feedback {self.position:+.4f} and p_m {p_m:+.4f} rad disagree (was the joint moving?)")

    @property
    def _wrap_period(self) -> float:
        return 2 * self.pmax * 65536 / 65535

    def _ingest(self, state: dict) -> None:
        raw = float(state["pos"])
        if math.isnan(self._raw_position):
            self.position = raw
        else:  # follow roll-overs of the packed position
            delta = raw - self._raw_position
            if delta > self._wrap_period / 2:
                delta -= self._wrap_period
            elif delta < -self._wrap_period / 2:
                delta += self._wrap_period
            self.position += delta
        self._raw_position = raw
        self.velocity = float(state["vel"])
        self.torque = float(state["torq"])
        self.mos_temperature = float(state["t_mos"])
        self.rotor_temperature = float(state["t_rotor"])
        self.status = str(state["status"])
        self._status_code = int(state["status_code"])

    def _check_health(self) -> None:
        code = self._status_code
        if code not in (DM_MOTOR_ENABLED, DM_MOTOR_DISABLED):
            self._trip(f"motor reported {self.status}")
        if self._enabled:
            if code == DM_MOTOR_DISABLED:
                if self._enable_grace > 0:
                    self._enable_grace -= 1
                else:
                    self._trip("motor reports DISABLED while it should be enabled")
            else:
                self._enable_grace = 0
        if self.torque_trip_nm is not None and abs(self.torque) > self.torque_trip_nm:
            self._trip(f"measured torque {self.torque:+.3f} Nm exceeds {self.torque_trip_nm} Nm")
        if abs(self.position) > self.pmax - POSITION_MARGIN_RAD / 2:
            self._trip(f"position {self.position:+.3f} rad too close to the +/-PMAX roll-over")
        if self.mos_temperature >= MAX_MOS_TEMPERATURE_C:
            self._trip(f"MOSFET temperature {self.mos_temperature:.0f} C >= {MAX_MOS_TEMPERATURE_C:.0f} C")
        if self.rotor_temperature >= MAX_ROTOR_TEMPERATURE_C:
            self._trip(f"rotor temperature {self.rotor_temperature:.0f} C >= {MAX_ROTOR_TEMPERATURE_C:.0f} C")

    def _trip(self, reason: str) -> None:
        self.faulted, self.fault_reason = True, reason
        self._cmd = dict(ZERO_COMMAND)
        self._enabled = False
        self.mode = "idle"
        try:
            self._motor.disable()
        except Exception as e:
            LOGGER.warning(f"disable after fault failed: {e}")
        LOGGER.critical(f"MOTOR FAULT: {reason}. Motor disabled and latched off.")
        raise MotorFault(reason)

    def _shutdown_bus(self) -> None:
        if self._controller is not None:
            try:
                self._controller.shutdown()
            except Exception as e:
                LOGGER.warning(f"CAN shutdown failed: {e}")
        self._controller = None
        self._motor = None
