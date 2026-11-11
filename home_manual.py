"""Manual homing: you move the link to each hard stop by hand while the motor applies no torque.

    pixi run home-manual

Use it for a first bring-up, or if automatic homing (home_auto.py) does not work.

Steps:
    1. The motor holds 0 Nm (the link is free; support it if gravity pulls it down).
    2. Rest the link gently on one stop and press Enter: its position is averaged over CAPTURE_S.
    3. The same at the other stop.
    4. Joint angle 0 = mid-stroke; save the calibration and mark the motor as homed.
    5. (IDENTIFY = True) Constant-speed sweeps from end to end, the joint moving by itself: fit dry
       and viscous friction and gravity, save them in the calibration.
    6. Move to 0 deg and hold until Enter.

Output: calibration/joint_cal.json and logs/home_manual_<time>.csv / .log.
"""

import math
import statistics

from exo.calibration import Calibration
from exo.compensation import run_identification
from exo.joint import Joint, JointError
from exo.loop import enter_pressed

# ---- Motor ----
CHANNEL = "can0"
MOTOR_ID = 0x02
FEEDBACK_ID = 0x12

# ---- Homing settings ----
CAPTURE_S = 0.5  # the stop position is averaged over this long after Enter
MAX_TORQUE_AT_0NM = 0.5  # at 0 Nm commanded the measured torque should stay near zero
IDENTIFY = True  # step 5: identify friction and gravity (the joint moves by itself)

# ---- Friction / gravity identification (as in home_auto.py) ----
IDENTIFY_SPEEDS_DEG_S = [10.0, 30.0]
IDENTIFY_MARGIN_DEG = 10.0
IDENTIFY_ACCEL_DEG_S2 = 200.0


def capture_by_hand(joint: Joint, prompt: str) -> float:
    """Keep the motor at 0 Nm until Enter, then average the position over CAPTURE_S.
    Returns the stop's position in the motor's frame [rad]."""
    print(f"\n{prompt}")
    joint.phase = "by hand"
    start = joint.angle_deg
    positions = []
    capture_end = None
    for i, t in enumerate(joint.loop()):
        joint.command_torque(0.0)
        joint.update()
        if abs(joint.torque_nm) > MAX_TORQUE_AT_0NM:
            raise JointError(f"measured torque {joint.torque_nm:+.3f} Nm while commanding 0 Nm")
        if capture_end is None:  # waiting for Enter
            if enter_pressed():
                capture_end = t + CAPTURE_S
                print("Capturing, hold still...")
            elif i % 20 == 0:
                print(f"  moved {joint.angle_deg - start:+7.2f} deg | torque {joint.torque_nm:+.3f} Nm", end="\r")
        else:  # capturing
            if joint.fresh:
                positions.append(joint.motor.position)
            if t >= capture_end:
                spread_deg = math.degrees(max(positions) - min(positions))
                print(f"  captured at {joint.angle_deg:+.2f} deg (from start), spread {spread_deg:.3f} deg")
                if spread_deg > 1.0:
                    print("  WARNING: the link moved during the capture")
                return statistics.fmean(positions)
    raise AssertionError("unreachable")


def main() -> None:
    with Joint(CHANNEL, MOTOR_ID, FEEDBACK_ID, log_name="home_manual") as joint:
        # Steps 1-3: the two stops, by hand
        print("Manual homing: the motor applies no torque. Ctrl-C aborts.")
        stop_1 = capture_by_hand(joint, "Move the link slowly to ONE hard stop, rest it gently there, press Enter.")
        stop_2 = capture_by_hand(joint, "Now move it to the OTHER hard stop, rest it gently there, press Enter.")

        # Step 4: angle 0 = mid-stroke
        cal = Calibration.from_stops(stop_1, stop_2, "manual", joint.motor.angle_offsets, joint.config)
        cal.save()
        joint.start_session(cal)
        print(f"Step 4: stroke {cal.stroke_deg:.2f} deg, stops at +/-{cal.limit_deg:.2f} deg; homed")

        # Step 5: friction and gravity
        if IDENTIFY:
            print("\nStep 5: friction / gravity identification. The joint will move by itself from end to end.")
            # Keep holding where it is while waiting (with the motor on, it must get a command every tick).
            joint.hold_until_enter(joint.angle_deg, "Let go of the link, then press Enter to start (Ctrl-C aborts).")
            comp = run_identification(joint, IDENTIFY_SPEEDS_DEG_S, IDENTIFY_MARGIN_DEG, IDENTIFY_ACCEL_DEG_S2)
            for line in comp.describe():
                print(f"  {line}")
            cal.compensation = comp.as_dict()
            cal.save()
            joint.start_session(cal)  # use the compensation from now on

        # Step 6: to mid-stroke
        joint.move_to(0.0, phase="to 0")
        joint.hold_until_enter(0.0, "Homed. Holding 0 deg; press Enter to release.")
        print("Released.")


if __name__ == "__main__":
    main()
