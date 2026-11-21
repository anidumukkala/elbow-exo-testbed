"""Passive exoskeleton test (Experiment 2): human arm inserted, evaluating mechanical transparency
and identifying human forearm inertia + skin-cuff interface friction.

    pixi run passive-exo
    python passive_exo.py --rig qdd_10to1 --subject S01 --exo-frf logs/free_chirp_..._frf.csv

Before:
    1. Run homing since the last motor power-up (pixi run home-auto or home-manual) with NOBODY wearing the exo.
    2. Run Experiment 1 (pixi run chirp-free) with NOBODY wearing the exo to identify J_exo.
    3. Verify CAN FD is up (pixi run can-up).

Test phases (all selectable via CLI flags):
    Phase A (optional, enabled by default): Bidirectional relaxed static gravity calibration.
        With the human arm strapped in and completely relaxed, the exo slowly approaches each static
        angle (-25, 0, +25 deg) from BOTH + and - directions so averaging cancels Coulomb stiction.
    Phase B: Active human swings in zero-torque mode (Mechanical Transparency & Backdrivability).
        The motor commands 0.0 Nm (kp = 0, kd = 0, tau = 0). The human actively swings their forearm
        across slow, medium, and fast speeds while data is logged at RATE_HZ.
    Phase C: Relaxed Human Forearm Inertia + Cuff Friction System ID.
        - Default (--phase-c-mode chirp): Gentle motor-driven sinusoidal chirp centered at 0 deg
          (0.5 -> 6.0 Hz, +/-8 deg, 30 s) while the subject keeps their forearm completely relaxed.
          Works cleanly on hard-stop-limited rigs where gravity free-decay cannot oscillate.
        - Legacy (--phase-c-mode decay): Relaxed free-decay pendulum swings in zero-torque mode
          (only valid when the arm's vertical hanging equilibrium lies well inside the hard stops).

Safety guards (abort and immediately disable the motor):
    - Zero-torque mode: measured |torque| > ZERO_TORQUE_ABORT_NM for 3 consecutive fresh ticks.
    - Chirp mode: |torque - hold_nm| > CHIRP_TORQUE_ABORT_NM for 3 consecutive fresh ticks,
      and |angle| > chirp_amp + CHIRP_ENVELOPE_MARGIN_DEG.
    - Velocity limit: |velocity| > MAX_SWING_SPEED_DEG_S for 3 consecutive fresh ticks.
    - Stop limits: enforced on every update() by Joint (hard stop + range_margin_deg).
    - Thermal limit: rotor temperature rise > TEMP_RISE_ABORT_C.
    - Behind all guards: DamiaoMotor torque_trip_nm and firmware CAN watchdog (~100 ms).

DRY_RUN (--dry-run): runs the same loop with the motor disabled (no homing needed) to verify CAN timing,
logging, and terminal prompts before strapping in a human arm.

Output (logs/passive_exo_<time>*):
    .csv / .log        full Joint tick log and console messages
    _passive.csv       clean per-tick experiment dataset (exo.passive.PASSIVE_COLUMNS)
    _meta.json         run metadata (rig, subject, settings, motor IDs)
    _analysis.json     computed transparency & system-ID metrics (if phases completed)
    _summary.txt       human-readable summary report
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from exo.chirp import Chirp
from exo.config import JointConfig
from exo.joint import Joint, JointError
from exo.logs import LOG_DIR, LOGGER
from exo.loop import enter_pressed
from exo.passive import (
    ExoBaseline,
    format_summary,
    load_exo_baseline,
    run_full_analysis,
    save_analysis_outputs,
    save_passive_csv,
)

# ---- Default Motor IDs ---- (CAN interface comes from JointConfig.channel in exo/config.py)
DEFAULT_MOTOR_ID = 0x02
DEFAULT_FEEDBACK_ID = 0x12

# ---- Default Test Settings ----
DEFAULT_RATE_HZ = 500.0
DEFAULT_GRAVITY_ANGLES_DEG = (-25.0, 0.0, 25.0)
GRAVITY_APPROACH_OFFSET_DEG = 5.0  # pre-position beyond end stops so every angle is approached in + and -
GRAVITY_HOLD_S = 1.5               # seconds held at each static angle (second half is averaged)
GRAVITY_MOVE_SPEED_DEG_S = 15.0
GRAVITY_MOVE_ACCEL_DEG_S2 = 30.0
GRAVITY_HOLD_KP = 20.0             # firm enough to overcome stiction while gentle for a relaxed arm
GRAVITY_HOLD_KD = 0.4

DEFAULT_SWING_DURATION_S = 30.0

# ---- Phase C Default: Gentle Motor-Driven Relaxed Chirp ----
DEFAULT_CHIRP_F0_HZ = 0.5
DEFAULT_CHIRP_F1_HZ = 6.0
DEFAULT_CHIRP_DURATION_S = 30.0
DEFAULT_CHIRP_AMP_DEG = 8.0
DEFAULT_CHIRP_MAX_SPEED_DEG_S = 90.0
DEFAULT_CHIRP_MAX_ACCEL_DEG_S2 = 1200.0
CHIRP_KP = 20.0
CHIRP_KD = 0.4

# ---- Phase C Legacy: Free-Decay Settings ----
DEFAULT_DECAY_TRIALS = 3
DEFAULT_DECAY_DURATION_S = 8.0
DECAY_REST_SPEED_DEG_S = 2.0  # auto-end decay trial once |v| stays below this for DECAY_REST_S
DECAY_REST_S = 1.2
DECAY_MIN_S = 2.0             # minimum recording time before auto-rest detection activates

# ---- Human-in-the-Loop Safety Guards ----
ZERO_TORQUE_ABORT_NM = 1.0    # measured torque while commanding 0 Nm
CHIRP_TORQUE_ABORT_NM = 2.5   # extra torque above holding torque during Phase C relaxed chirp
CHIRP_ENVELOPE_MARGIN_DEG = 8.0
MAX_SWING_SPEED_DEG_S = 400.0 # excessive swing velocity guard
SOFT_STOP_WARN_DEG = 8.0      # warn in console when within this margin of a hard stop
TEMP_RISE_ABORT_C = 15.0
GUARD_CONFIRM_TICKS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2: Passive Exoskeleton Test")
    p.add_argument("--motor-id", type=lambda s: int(s, 0), default=DEFAULT_MOTOR_ID,
                   help=f"Motor command CAN ID (default: 0x{DEFAULT_MOTOR_ID:02X})")
    p.add_argument("--feedback-id", type=lambda s: int(s, 0), default=DEFAULT_FEEDBACK_ID,
                   help=f"Motor feedback CAN ID (default: 0x{DEFAULT_FEEDBACK_ID:02X})")
    p.add_argument("--channel", type=str, default=None, help="CAN interface override (default: exo/config.py)")
    p.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="Control/logging rate in Hz")
    p.add_argument("--rig", type=str, default="qdd_10to1",
                   help="Rig label saved in metadata (e.g., qdd_10to1 or ume_40to1)")
    p.add_argument("--subject", type=str, default="S01", help="Subject identifier saved in metadata")
    p.add_argument("--swing-duration", type=float, default=DEFAULT_SWING_DURATION_S,
                   help="Phase B active zero-torque swing duration in seconds")
    p.add_argument("--phase-c-mode", choices=("chirp", "decay"), default="chirp",
                   help="Phase C system-ID method: 'chirp' (default, works with hard stops) or 'decay'")
    p.add_argument("--chirp-duration", type=float, default=DEFAULT_CHIRP_DURATION_S,
                   help="Phase C relaxed chirp duration in seconds (default: 30)")
    p.add_argument("--chirp-amp", type=float, default=DEFAULT_CHIRP_AMP_DEG,
                   help="Phase C relaxed chirp amplitude in deg around 0 deg (default: 8.0)")
    p.add_argument("--decay-trials", type=int, default=DEFAULT_DECAY_TRIALS,
                   help="Number of Phase C legacy free-decay trials (when --phase-c-mode decay)")
    p.add_argument("--decay-duration", type=float, default=DEFAULT_DECAY_DURATION_S,
                   help="Maximum duration per Phase C free-decay trial in seconds")
    p.add_argument("--skip-gravity-cal", action="store_true",
                   help="Skip Phase A static gravity holds")
    p.add_argument("--skip-active", action="store_true", help="Skip Phase B active swing phase")
    p.add_argument("--skip-phase-c", "--skip-decay", dest="skip_phase_c", action="store_true",
                   help="Skip Phase C system-ID phase")
    p.add_argument("--exo-frf", type=Path, default=None,
                   help="Path to Experiment 1 free_chirp_*_frf.csv for J_exo baseline")
    p.add_argument("--exo-inertia", type=float, default=None,
                   help="Manual override for J_exo [kg m^2] (overrides --exo-frf)")
    p.add_argument("--dry-run", action="store_true",
                   help="Keep motor disabled (no homing required) to verify CAN timing and prompts")
    return p.parse_args()


def _find_latest_frf() -> Optional[Path]:
    candidates = sorted(LOG_DIR.rglob("free_chirp_*_frf.csv"))
    return candidates[-1] if candidates else None


def _append_row(rows: list, t: float, loop_start: float, joint: Joint, phase: str, trial: int) -> None:
    t_sample = joint.sample_time - loop_start if math.isfinite(joint.sample_time) else t
    rows.append((
        t,
        t_sample,
        int(joint.fresh),
        phase,
        trial,
        joint.angle_deg,
        joint.velocity_deg_s,
        joint.torque_nm,
        joint.config.output_sign * joint.motor.commanded_total_torque,
        joint.motor.mos_temperature,
        joint.motor.rotor_temperature,
    ))


def _check_zero_torque_guards(
    joint: Joint,
    dry_run: bool,
    rotor_at_start: float,
    over_torque: int,
    over_speed: int,
) -> tuple[int, int]:
    """Enforce zero-torque safety guards on fresh motor replies."""
    if dry_run or not joint.fresh:
        return over_torque, over_speed

    over_torque = over_torque + 1 if abs(joint.torque_nm) > ZERO_TORQUE_ABORT_NM else 0
    if over_torque >= GUARD_CONFIRM_TICKS:
        raise JointError(
            f"measured torque {joint.torque_nm:+.3f} Nm exceeded {ZERO_TORQUE_ABORT_NM:.2f} Nm "
            "while commanding 0 Nm"
        )

    over_speed = over_speed + 1 if abs(joint.velocity_deg_s) > MAX_SWING_SPEED_DEG_S else 0
    if over_speed >= GUARD_CONFIRM_TICKS:
        raise JointError(
            f"angular velocity {joint.velocity_deg_s:+.1f} deg/s exceeded +/-{MAX_SWING_SPEED_DEG_S:.0f} deg/s"
        )

    if math.isfinite(rotor_at_start) and (joint.motor.rotor_temperature - rotor_at_start) > TEMP_RISE_ABORT_C:
        raise JointError(f"rotor temperature rose by more than {TEMP_RISE_ABORT_C:g} C")

    return over_torque, over_speed


def run_zero_torque_wait(
    joint: Joint,
    prompt: str,
    dry_run: bool,
    rotor_at_start: float,
) -> None:
    """Keep the motor in safe zero-torque mode while waiting for the operator to press Enter."""
    print(f"\n{prompt}")
    joint.phase = "zero_wait"
    over_t, over_v = 0, 0
    for i, _t in enumerate(joint.loop()):
        if dry_run:
            joint.idle()
        else:
            joint.command_torque(0.0)
        joint.update()
        over_t, over_v = _check_zero_torque_guards(joint, dry_run, rotor_at_start, over_t, over_v)
        if enter_pressed():
            print()
            return
        if i % max(1, int(joint.config.rate_hz // 10)) == 0:
            print(
                f"  [0 Nm] angle {joint.angle_deg:+7.2f} deg | vel {joint.velocity_deg_s:+7.1f} deg/s | "
                f"tau {joint.torque_nm:+.3f} Nm",
                end="\r",
            )


def _hold_static_angle(
    joint: Joint,
    rows: list,
    t0_mono: float,
    target_deg: float,
    phase_name: str,
    idx: int,
) -> None:
    """Move slowly to target_deg and hold for GRAVITY_HOLD_S while logging."""
    joint.move_to(
        target_deg,
        speed_deg_s=GRAVITY_MOVE_SPEED_DEG_S,
        accel_deg_s2=GRAVITY_MOVE_ACCEL_DEG_S2,
        phase=f"move_{phase_name}",
    )
    joint.phase = phase_name
    for t in joint.loop():
        ff = joint.compensation_torque(target_deg, 0.0, target_deg - joint.angle_deg)
        joint.command_position(target_deg, 0.0, ff, kp=GRAVITY_HOLD_KP, kd=GRAVITY_HOLD_KD)
        joint.update()
        _append_row(rows, time.monotonic() - t0_mono, t0_mono, joint, phase_name, idx)
        if t >= GRAVITY_HOLD_S:
            break
    # LOGGER.flush_buffer()  # flushed safely after motor.stop()
    LOGGER.info(
        f"    {phase_name}: settled at {joint.angle_deg:+.2f} deg | holding torque {joint.torque_nm:+.3f} Nm"
    )


def run_phase_a_gravity(
    joint: Joint,
    rows: list,
    t0_mono: float,
    dry_run: bool,
) -> None:
    """Phase A: bidirectional static gravity holds to cancel Coulomb stiction."""
    LOGGER.info("Phase A: Bidirectional static gravity calibration (relaxed forearm + exo)")
    if dry_run:
        LOGGER.info("  DRY RUN: skipping active position holds in Phase A")
        return

    soft_limit = joint.calibration.limit_deg - joint.config.soft_margin_deg - 5.0
    angles = [max(-soft_limit, min(soft_limit, a)) for a in DEFAULT_GRAVITY_ANGLES_DEG]
    unique_angles: list[float] = []
    for a in angles:
        if not any(math.isclose(a, u, abs_tol=2.0) for u in unique_angles):
            unique_angles.append(a)
    unique_angles.sort()

    print(
        f"\nPhase A will slowly approach {unique_angles} deg from BOTH + and - directions "
        f"(speed {GRAVITY_MOVE_SPEED_DEG_S:g} deg/s, hold {GRAVITY_HOLD_S:g} s each) to cancel stiction."
    )
    joint.hold_until_enter(
        joint.angle_deg,
        "Relax your forearm completely in the cuff, then press Enter to start Phase A (Ctrl-C to abort).",
        phase="gravity_ready",
    )

    # Pass 1: approach each angle moving in the + direction
    low_pre = max(-soft_limit, unique_angles[0] - GRAVITY_APPROACH_OFFSET_DEG)
    joint.move_to(low_pre, speed_deg_s=GRAVITY_MOVE_SPEED_DEG_S, accel_deg_s2=GRAVITY_MOVE_ACCEL_DEG_S2,
                  phase="gravity_pre_pos")
    for idx, target_deg in enumerate(unique_angles, start=1):
        _hold_static_angle(joint, rows, t0_mono, target_deg, f"gravity_hold_{idx}_pos", idx)

    # Pass 2: approach each angle moving in the - direction
    high_pre = min(soft_limit, unique_angles[-1] + GRAVITY_APPROACH_OFFSET_DEG)
    joint.move_to(high_pre, speed_deg_s=GRAVITY_MOVE_SPEED_DEG_S, accel_deg_s2=GRAVITY_MOVE_ACCEL_DEG_S2,
                  phase="gravity_pre_neg")
    for idx in range(len(unique_angles), 0, -1):
        target_deg = unique_angles[idx - 1]
        _hold_static_angle(joint, rows, t0_mono, target_deg, f"gravity_hold_{idx}_neg", idx)

    # Smoothly release back to 0 Nm
    joint.command_torque(0.0)
    joint.update()


def run_phase_b_active_swing(
    joint: Joint,
    rows: list,
    t0_mono: float,
    duration_s: float,
    dry_run: bool,
    rotor_at_start: float,
) -> None:
    """Phase B: active human swinging in pure zero-torque mode to evaluate transparency."""
    LOGGER.info(f"Phase B: Active zero-torque swing test ({duration_s:g} s)")
    run_zero_torque_wait(
        joint,
        f"Phase B (Active Swing, {duration_s:g} s, 0 Nm): Press Enter to start recording.\n"
        "  Cue: swing smoothly through slow (first 10 s), medium (next 10 s), and fast (last 10 s) cycles.\n"
        "  Press Enter again at any time to finish Phase B early (Ctrl-C aborts).",
        dry_run,
        rotor_at_start,
    )

    joint.phase = "active_swing"
    over_t, over_v = 0, 0
    next_print = 0.0
    warn_limit = (joint.calibration.limit_deg - SOFT_STOP_WARN_DEG) if joint.is_homed else math.inf

    for t in joint.loop():
        if dry_run:
            joint.idle()
        else:
            joint.command_torque(0.0)
        joint.update()
        _append_row(rows, time.monotonic() - t0_mono, t0_mono, joint, "active_swing", 1)
        over_t, over_v = _check_zero_torque_guards(joint, dry_run, rotor_at_start, over_t, over_v)

        if enter_pressed():
            print()
            LOGGER.info(f"  Phase B stopped early by operator at t = {t:.1f} s")
            break
        if t >= next_print:
            frac = t / max(duration_s, 1e-3)
            cue = "SLOW" if frac < 0.33 else ("MEDIUM" if frac < 0.67 else "FAST")
            stop_flag = " [NEAR STOP!]" if abs(joint.angle_deg) > warn_limit else ""
            print(
                f"  [Phase B {cue:6s}] t {t:5.1f}/{duration_s:4.0f} s | "
                f"angle {joint.angle_deg:+6.1f} deg | vel {joint.velocity_deg_s:+6.1f} deg/s | "
                f"tau {joint.torque_nm:+.3f} Nm{stop_flag}",
                end="\r",
            )
            next_print = t + 0.25
        if t >= duration_s:
            print()
            break
    # LOGGER.flush_buffer()  # flushed safely after motor.stop()


def run_phase_c_relaxed_chirp(
    joint: Joint,
    rows: list,
    t0_mono: float,
    duration_s: float,
    amplitude_deg: float,
    dry_run: bool,
    rotor_at_start: float,
) -> None:
    """Phase C (default): gentle motor-driven chirp around 0 deg with relaxed human forearm."""
    chirp = Chirp(
        DEFAULT_CHIRP_F0_HZ,
        DEFAULT_CHIRP_F1_HZ,
        duration_s,
        amplitude_deg,
        DEFAULT_CHIRP_MAX_SPEED_DEG_S,
        DEFAULT_CHIRP_MAX_ACCEL_DEG_S2,
    )
    LOGGER.info(
        f"Phase C: Relaxed motor-driven chirp ({DEFAULT_CHIRP_F0_HZ:g} -> {DEFAULT_CHIRP_F1_HZ:g} Hz, "
        f"{duration_s:g} s, +/-{amplitude_deg:g} deg around 0 deg)"
    )

    hold_nm = 0.0
    if not dry_run:
        joint.move_to(0.0, speed_deg_s=GRAVITY_MOVE_SPEED_DEG_S, accel_deg_s2=GRAVITY_MOVE_ACCEL_DEG_S2,
                      phase="chirp_to_0")
        hold_nm = joint.measure_holding_torque(torque_nm=joint.compensation_torque(0.0, 0.0))
        LOGGER.info(f"  Phase C holding torque at 0 deg: {hold_nm:+.3f} Nm")
        joint.hold_until_enter(
            0.0,
            f"Phase C (Relaxed Chirp, {duration_s:g} s, +/-{amplitude_deg:g} deg around 0 deg):\n"
            "  Keep your forearm COMPLETELY RELAXED ('dead weight') while the motor gently oscillates it.\n"
            "  Press Enter to start (press Enter again or Ctrl-C at any time to stop).",
            phase="chirp_ready",
        )
    else:
        run_zero_torque_wait(
            joint,
            f"Phase C DRY RUN (Relaxed Chirp, {duration_s:g} s): Press Enter to start.",
            dry_run,
            rotor_at_start,
        )

    joint.phase = "relaxed_chirp"
    lead = 0.5 / joint.config.rate_hz
    envelope = amplitude_deg + CHIRP_ENVELOPE_MARGIN_DEG
    over_t = 0
    next_print = 0.0

    for t in joint.loop():
        target, target_vel = chirp.reference(t + lead)
        if dry_run:
            joint.idle()
        else:
            # Feedforward: exo compensation + static extra arm holding torque at 0 deg
            ff = joint.compensation_torque(target, target_vel, target - joint.angle_deg)
            joint.command_position(target, target_vel, ff, kp=CHIRP_KP, kd=CHIRP_KD)
        joint.update()
        _append_row(rows, time.monotonic() - t0_mono, t0_mono, joint, "relaxed_chirp", 1)

        if not dry_run and joint.fresh:
            over_t = over_t + 1 if abs(joint.torque_nm - hold_nm) > CHIRP_TORQUE_ABORT_NM else 0
            if over_t >= GUARD_CONFIRM_TICKS:
                raise JointError(
                    f"Phase C chirp torque {joint.torque_nm:+.3f} Nm (holding {hold_nm:+.3f}) exceeded "
                    f"+/-{CHIRP_TORQUE_ABORT_NM:g} Nm guard at {chirp.freq(t):.2f} Hz"
                )
            if abs(joint.angle_deg) > envelope:
                raise JointError(
                    f"Phase C chirp angle {joint.angle_deg:+.2f} deg exceeded +/-{envelope:.1f} deg envelope"
                )
            if math.isfinite(rotor_at_start) and (joint.motor.rotor_temperature - rotor_at_start) > TEMP_RISE_ABORT_C:
                raise JointError(f"rotor temperature rose by more than {TEMP_RISE_ABORT_C:g} C")

        if enter_pressed():
            print()
            LOGGER.info(f"  Phase C chirp stopped early by operator at t = {t:.1f} s")
            break
        if t >= next_print:
            f = chirp.freq(t)
            print(
                f"  [Phase C Chirp] t {t:5.1f}/{duration_s:4.0f} s | f {f:4.2f} Hz | "
                f"angle {joint.angle_deg:+6.2f} deg | tau {joint.torque_nm:+.3f} Nm",
                end="\r",
            )
            next_print = t + 0.5
        if t >= duration_s + 0.5:
            print()
            break

    # Gently release to 0 Nm
    if not dry_run:
        joint.command_torque(0.0)
        joint.update()
    # LOGGER.flush_buffer()  # flushed safely after motor.stop()


def run_phase_c_free_decay(
    joint: Joint,
    rows: list,
    t0_mono: float,
    n_trials: int,
    max_duration_s: float,
    dry_run: bool,
    rotor_at_start: float,
) -> None:
    """Phase C (legacy): relaxed free-decay pendulum swings in zero-torque mode."""
    LOGGER.info(f"Phase C (legacy): Relaxed free-decay trials ({n_trials} trials, max {max_duration_s:g} s each)")
    warn_limit = (joint.calibration.limit_deg - SOFT_STOP_WARN_DEG) if joint.is_homed else math.inf

    for trial in range(1, n_trials + 1):
        run_zero_torque_wait(
            joint,
            f"Phase C Trial {trial}/{n_trials} (Relaxed Free-Decay, 0 Nm):\n"
            "  1. Lift the forearm to a flexed starting angle clear of hard stops.\n"
            "  2. Press Enter to start recording, then IMMEDIATELY relax and release the forearm.\n"
            "  (Recording stops automatically when the arm settles, or press Enter to stop.)",
            dry_run,
            rotor_at_start,
        )

        phase_name = f"free_decay_{trial}"
        joint.phase = phase_name
        over_t, over_v = 0, 0
        rest_time = 0.0
        prev_t = 0.0
        next_print = 0.0

        for t in joint.loop():
            if dry_run:
                joint.idle()
            else:
                joint.command_torque(0.0)
            joint.update()
            _append_row(rows, time.monotonic() - t0_mono, t0_mono, joint, phase_name, trial)
            over_t, over_v = _check_zero_torque_guards(joint, dry_run, rotor_at_start, over_t, over_v)

            dt = t - prev_t
            prev_t = t
            if t >= DECAY_MIN_S and abs(joint.velocity_deg_s) < DECAY_REST_SPEED_DEG_S:
                rest_time += dt
            else:
                rest_time = 0.0

            if enter_pressed():
                print()
                LOGGER.info(f"  Trial {trial} ended by operator at t = {t:.2f} s")
                break
            if rest_time >= DECAY_REST_S:
                print()
                LOGGER.info(f"  Trial {trial} settled at t = {t:.2f} s (angle {joint.angle_deg:+.1f} deg)")
                break
            if t >= next_print:
                stop_flag = " [NEAR STOP!]" if abs(joint.angle_deg) > warn_limit else ""
                print(
                    f"  [Decay {trial}/{n_trials}] t {t:4.1f} s | angle {joint.angle_deg:+6.1f} deg | "
                    f"vel {joint.velocity_deg_s:+6.1f} deg/s | tau {joint.torque_nm:+.3f} Nm{stop_flag}",
                    end="\r",
                )
                next_print = t + 0.25
            if t >= max_duration_s:
                print()
                LOGGER.info(f"  Trial {trial} reached max duration ({max_duration_s:g} s)")
                break
        # LOGGER.flush_buffer()  # flushed safely after motor.stop()


def main() -> None:
    args = parse_args()
    cfg = replace(JointConfig(), rate_hz=args.rate_hz)
    frf_path = args.exo_frf or _find_latest_frf()

    print(
        f"Passive Exo Test (Experiment 2) | rig={args.rig} | subject={args.subject} | "
        f"rate={args.rate_hz:g} Hz | phaseC={args.phase_c_mode} | "
        f"{'DRY RUN (motor off)' if args.dry_run else 'LIVE'}"
    )
    if frf_path is not None:
        print(f"  Exp 1 FRF baseline: {frf_path}")

    rows: list = []
    stem: Optional[Path] = None
    stop_limit_deg: Optional[float] = None

    try:
        with Joint(
            channel=args.channel,
            motor_id=args.motor_id,
            feedback_id=args.feedback_id,
            log_name="passive_exo",
            rate_hz=args.rate_hz,
            config=cfg,
        ) as joint:
            stem = joint.log_stem
            if not args.dry_run:
                joint.resume()
                if joint.is_homed:
                    stop_limit_deg = joint.calibration.limit_deg

            rotor_at_start = joint.motor.rotor_temperature
            t0_mono = time.monotonic()

            if not args.skip_gravity_cal:
                run_phase_a_gravity(joint, rows, t0_mono, args.dry_run)

            if not args.skip_active:
                run_phase_b_active_swing(
                    joint, rows, t0_mono, args.swing_duration, args.dry_run, rotor_at_start
                )

            if not args.skip_phase_c:
                if args.phase_c_mode == "chirp":
                    run_phase_c_relaxed_chirp(
                        joint, rows, t0_mono, args.chirp_duration, args.chirp_amp,
                        args.dry_run, rotor_at_start,
                    )
                elif args.decay_trials > 0:
                    run_phase_c_free_decay(
                        joint, rows, t0_mono, args.decay_trials, args.decay_duration,
                        args.dry_run, rotor_at_start,
                    )

            LOGGER.info("Passive exo test complete. Motor releasing.")
    finally:
        if stem is not None and rows:
            try:
                csv_out = Path(f"{stem}_passive.csv")
                save_passive_csv(csv_out, rows)
                meta = {
                    "created": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "rig": args.rig,
                    "subject": args.subject,
                    "dry_run": args.dry_run,
                    "rate_hz": args.rate_hz,
                    "phase_c_mode": args.phase_c_mode,
                    "motor_id": f"0x{args.motor_id:02X}",
                    "feedback_id": f"0x{args.feedback_id:02X}",
                    "stop_limit_deg": stop_limit_deg,
                    "exo_frf": str(frf_path) if frf_path else "",
                    "exo_inertia_override": args.exo_inertia,
                    "rows": len(rows),
                }
                Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
                LOGGER.info(f"Saved data: {csv_out}")

                if not args.dry_run:
                    baseline: ExoBaseline = load_exo_baseline(
                        frf_path=frf_path,
                        inertia_override_kg_m2=args.exo_inertia,
                    )
                    report, _ = run_full_analysis(
                        csv_out, baseline, stop_limit_deg=stop_limit_deg
                    )
                    summary = format_summary(report)
                    print("\n" + summary)
                    save_analysis_outputs(stem, report, summary)
                    LOGGER.info(f"Saved analysis: {stem}_summary.txt, {stem}_analysis.json")
            except Exception as exc:
                LOGGER.error(f"Post-run save/analysis failed: {exc}")


if __name__ == "__main__":
    main()
