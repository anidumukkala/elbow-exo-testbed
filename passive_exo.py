"""Passive exoskeleton test (Experiment 2): human arm inserted, swinging to drive the exo in zero-torque mode.

    pixi run passive-exo
    python passive_exo.py --rig qdd_10to1 --subject S01 --exo-frf logs/free_chirp_..._frf.csv

Before:
    1. Run homing since the last motor power-up (pixi run home-auto or home-manual) with NOBODY wearing the exo.
    2. Optionally run Experiment 1 (pixi run chirp-free) with NOBODY wearing the exo to identify J_exo.
    3. Verify CAN FD is up (pixi run can-up).

Test phases (all selectable via CLI flags):
    Phase A (optional, enabled by default): Relaxed static gravity calibration.
        With the human arm strapped in and completely relaxed, the exo slowly moves to 3 static angles
        (GRAVITY_HOLD_ANGLES_DEG) and measures holding torque to isolate g_total(q) = g_exo(q) + g_arm(q).
        This establishes the exact Nm scale needed to identify J_human and cuff friction in Phase C.
    Phase B: Active human swings in zero-torque mode (Mechanical Transparency & Backdrivability).
        The motor commands 0.0 Nm (kp = 0, kd = 0, tau = 0). The human actively swings their forearm
        across slow, medium, and fast speeds while data is logged at RATE_HZ.
    Phase C: Relaxed free-decay swings in zero-torque mode (Human Forearm Inertia + Cuff Friction ID).
        The motor stays at 0.0 Nm. For each trial, the subject lifts their arm to a flexed starting angle,
        relaxes completely, and lets the forearm swing freely to rest under gravity and damping.

Safety guards (abort and immediately disable the motor):
    - Zero-torque mode: measured |torque| > ZERO_TORQUE_ABORT_NM for 3 consecutive fresh ticks.
    - Velocity limit: |velocity| > MAX_SWING_SPEED_DEG_S for 3 consecutive fresh ticks.
    - Stop limits: enforced on every update() by Joint (hard stop + range_margin_deg).
    - Thermal limit: rotor temperature rise > TEMP_RISE_ABORT_C.
    - Behind all guards: DamiaoMotor torque_trip_nm (1.5 Nm) and firmware CAN watchdog (~100 ms).

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
GRAVITY_HOLD_S = 2.0          # seconds held at each static angle (second half is averaged)
GRAVITY_MOVE_SPEED_DEG_S = 15.0
GRAVITY_MOVE_ACCEL_DEG_S2 = 30.0
GRAVITY_HOLD_KP = 12.0        # softer position control while human arm is strapped in
GRAVITY_HOLD_KD = 0.4

DEFAULT_SWING_DURATION_S = 30.0
DEFAULT_DECAY_TRIALS = 3
DEFAULT_DECAY_DURATION_S = 8.0
DECAY_REST_SPEED_DEG_S = 2.0  # auto-end decay trial once |v| stays below this for DECAY_REST_S
DECAY_REST_S = 1.2
DECAY_MIN_S = 2.0             # minimum recording time before auto-rest detection activates

# ---- Human-in-the-Loop Safety Guards ----
ZERO_TORQUE_ABORT_NM = 1.0   # measured torque while commanding 0 Nm
MAX_SWING_SPEED_DEG_S = 400.0 # excessive swing velocity guard
SOFT_STOP_WARN_DEG = 8.0      # warn in console when within this margin of a hard stop
TEMP_RISE_ABORT_C = 15.0
GUARD_CONFIRM_TICKS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2: Passive Exoskeleton Zero-Torque Test")
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
    p.add_argument("--decay-trials", type=int, default=DEFAULT_DECAY_TRIALS,
                   help="Number of Phase C relaxed free-decay trials")
    p.add_argument("--decay-duration", type=float, default=DEFAULT_DECAY_DURATION_S,
                   help="Maximum duration per Phase C free-decay trial in seconds")
    p.add_argument("--skip-gravity-cal", action="store_true",
                   help="Skip Phase A static gravity holds (run pure zero-torque only)")
    p.add_argument("--skip-active", action="store_true", help="Skip Phase B active swing phase")
    p.add_argument("--skip-decay", action="store_true", help="Skip Phase C relaxed free-decay trials")
    p.add_argument("--exo-frf", type=Path, default=None,
                   help="Path to Experiment 1 free_chirp_*_frf.csv for J_exo baseline")
    p.add_argument("--exo-inertia", type=float, default=None,
                   help="Manual override for J_exo [kg m^2] (overrides --exo-frf)")
    p.add_argument("--dry-run", action="store_true",
                   help="Keep motor disabled (no homing required) to verify CAN timing and prompts")
    return p.parse_args()


def _find_latest_frf() -> Optional[Path]:
    candidates = sorted(LOG_DIR.glob("free_chirp_*_frf.csv"))
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


def run_phase_a_gravity(
    joint: Joint,
    rows: list,
    t0_mono: float,
    dry_run: bool,
) -> None:
    """Phase A: gently move relaxed arm to 3 static angles and record holding torque."""
    LOGGER.info("Phase A: Static gravity calibration (relaxed forearm + exo)")
    if dry_run:
        LOGGER.info("  DRY RUN: skipping active position holds in Phase A")
        return

    soft_limit = joint.calibration.limit_deg - joint.config.soft_margin_deg - 5.0
    angles = [max(-soft_limit, min(soft_limit, a)) for a in DEFAULT_GRAVITY_ANGLES_DEG]
    # Remove duplicate clipped angles
    unique_angles = []
    for a in angles:
        if not any(math.isclose(a, u, abs_tol=2.0) for u in unique_angles):
            unique_angles.append(a)

    print(
        f"\nPhase A will slowly move your RELAXED forearm to {unique_angles} deg "
        f"(speed {GRAVITY_MOVE_SPEED_DEG_S:g} deg/s) and hold {GRAVITY_HOLD_S:g} s at each angle."
    )
    joint.hold_until_enter(
        joint.angle_deg,
        "Relax your forearm completely in the cuff, then press Enter to start Phase A (Ctrl-C to abort).",
        phase="gravity_ready",
    )

    for idx, target_deg in enumerate(unique_angles, start=1):
        LOGGER.info(f"  Phase A hold {idx}/{len(unique_angles)}: moving slowly to {target_deg:+.1f} deg")
        joint.move_to(
            target_deg,
            speed_deg_s=GRAVITY_MOVE_SPEED_DEG_S,
            accel_deg_s2=GRAVITY_MOVE_ACCEL_DEG_S2,
            phase=f"gravity_move_{idx}",
        )
        phase_name = f"gravity_hold_{idx}"
        joint.phase = phase_name
        loop = joint.loop()
        for t in loop:
            ff = joint.compensation_torque(target_deg, 0.0, target_deg - joint.angle_deg)
            joint.command_position(target_deg, 0.0, ff, kp=GRAVITY_HOLD_KP, kd=GRAVITY_HOLD_KD)
            joint.update()
            t_elapsed = time.monotonic() - t0_mono
            _append_row(rows, t_elapsed, t0_mono, joint, phase_name, idx)
            if t >= GRAVITY_HOLD_S:
                break
        # LOGGER.flush_buffer()  # flushed safely after motor.stop()
        LOGGER.info(
            f"    settled at {joint.angle_deg:+.2f} deg | holding torque {joint.torque_nm:+.3f} Nm"
        )

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
            # Pace cue across the recording window
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


def run_phase_c_free_decay(
    joint: Joint,
    rows: list,
    t0_mono: float,
    n_trials: int,
    max_duration_s: float,
    dry_run: bool,
    rotor_at_start: float,
) -> None:
    """Phase C: relaxed free-decay pendulum swings in zero-torque mode."""
    LOGGER.info(f"Phase C: Relaxed free-decay trials ({n_trials} trials, max {max_duration_s:g} s each)")
    warn_limit = (joint.calibration.limit_deg - SOFT_STOP_WARN_DEG) if joint.is_homed else math.inf

    for trial in range(1, n_trials + 1):
        run_zero_torque_wait(
            joint,
            f"Phase C Trial {trial}/{n_trials} (Relaxed Free-Decay, 0 Nm):\n"
            "  1. Lift the forearm to a flexed starting angle (e.g., +35 to +55 deg).\n"
            "  2. Press Enter to start recording, then IMMEDIATELY relax and release the forearm to swing freely.\n"
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
        f"rate={args.rate_hz:g} Hz | {'DRY RUN (motor off)' if args.dry_run else 'LIVE ZERO-TORQUE'}"
    )
    if frf_path is not None:
        print(f"  Exp 1 FRF baseline: {frf_path}")

    rows: list = []
    stem: Optional[Path] = None

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

            rotor_at_start = joint.motor.rotor_temperature
            t0_mono = time.monotonic()

            if not args.skip_gravity_cal:
                run_phase_a_gravity(joint, rows, t0_mono, args.dry_run)

            if not args.skip_active:
                run_phase_b_active_swing(
                    joint, rows, t0_mono, args.swing_duration, args.dry_run, rotor_at_start
                )

            if not args.skip_decay and args.decay_trials > 0:
                run_phase_c_free_decay(
                    joint, rows, t0_mono, args.decay_trials, args.decay_duration, args.dry_run, rotor_at_start
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
                    "motor_id": f"0x{args.motor_id:02X}",
                    "feedback_id": f"0x{args.feedback_id:02X}",
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
                    report, _ = run_full_analysis(csv_out, baseline)
                    summary = format_summary(report)
                    print("\n" + summary)
                    save_analysis_outputs(stem, report, summary)
                    LOGGER.info(f"Saved analysis: {stem}_summary.txt, {stem}_analysis.json")
            except Exception as exc:
                LOGGER.error(f"Post-run save/analysis failed: {exc}")


if __name__ == "__main__":
    main()
