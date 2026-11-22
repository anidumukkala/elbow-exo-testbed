"""Automatic homing: the joint finds both hard stops by itself, then measures its friction and gravity.

    pixi run home-auto

Run it after every motor power-up, with the link free (nobody wearing the exo, hands clear).

Steps:
    1. Breakaway test: near the starting position, push up and then down with a slowly rising
       torque until the joint moves (if it rests on a stop, the free direction goes first and the
       joint moves CLEARANCE_DEG off the stop before the other push). Half the difference of the
       two torques is the static friction, half the sum is the gravity torque there. They set how
       hard the motor pushes in steps 2-3.
    2. Find the first stop: drive toward it in velocity control (the push is capped). When the
       joint stalls, push harder, slowly, up to STOP_TORQUE_NM: if it moves again, the stall was a
       stiff spot (friction or gravity) and the drive continues; if it does not, it is the stop,
       and its position is measured while pushing with STOP_TORQUE_NM. Then back off RELAX_DEG.
    3. Find the second stop the same way.
    4. Joint angle 0 = mid-stroke; save the calibration and mark the motor as homed.
    5. Constant-speed sweeps from end to end: fit dry and viscous friction and gravity
       (exo/compensation.py), save them in the calibration.
    6. Move to 0 deg with the new compensation and hold until Enter.

Output: calibration/joint_cal.json and logs/home_auto_<time>.csv / .log.
"""

from exo.calibration import Calibration
from exo.compensation import run_identification
from exo.joint import Joint, JointError
from exo.logs import LOGGER, ROOT

# ---- Motor ---- (the CAN interface is JointConfig.channel in exo/config.py)
MOTOR_ID = 0x02
FEEDBACK_ID = 0x12

# ---- Homing settings ----
FIRST_DIRECTION = +1  # which stop first: +1 = positive joint angle
STOP_TORQUE_NM = 1.0  # most the motor pushes against a stop (below the 1.5 Nm trip)
TORQUE_RAMP_NM_S = 0.25  # how fast the push rises while testing whether the joint moves
APPROACH_SPEED_DEG_S = 15.0  # speed toward a stop
APPROACH_MARGIN_NM = 0.10  # approach push = largest breakaway torque + this ...
MIN_APPROACH_NM = 0.15  # ... but at least this
MAX_TRAVEL_DEG = 216.0  # give up if no stop within this travel (1.2 x the design stroke)
RELAX_DEG = 3.0  # back off the stop by this much after measuring it
CLEARANCE_DEG = 5.0  # step 1: distance from a stop the joint may be resting on
MAX_FALSE_STALLS = 10  # stiff spots allowed on the way to one stop

# ---- Friction / gravity identification ----
IDENTIFY_SPEEDS_DEG_S = [10.0, 30.0]  # two speeds separate dry from viscous friction
IDENTIFY_MARGIN_DEG = 10.0  # the sweeps stay this far from the stops
IDENTIFY_ACCEL_DEG_S2 = 200.0


def breakaway(joint: Joint, direction: int):
    """Hold still, then push in `direction` with a slowly rising torque until the joint moves.
    Returns the torque at which it moved [Nm], or None if it did not move even with
    STOP_TORQUE_NM (it rests on a stop)."""
    hold_nm = joint.measure_holding_torque()
    result = joint.push_until_moves(direction, start_nm=hold_nm, max_nm=STOP_TORQUE_NM, rate_nm_s=TORQUE_RAMP_NM_S,
                                    phase="breakaway +" if direction > 0 else "breakaway -")
    return result.torque_nm if result.moved else None


def find_stop(joint: Joint, direction: int, push_nm: float) -> float:
    """Drive toward the stop in `direction` and measure it. Returns the stop's position in the
    motor's frame [rad] and leaves the joint RELAX_DEG away from it."""
    start = joint.angle_deg
    for _ in range(MAX_FALSE_STALLS):
        travelled = direction * (joint.angle_deg - start)
        joint.drive_until_stall(direction, APPROACH_SPEED_DEG_S, push_nm, MAX_TRAVEL_DEG - travelled,
                                phase="approach")
        # Stalled. Is it the stop, or a stiff spot? Push harder, slowly, and see whether it moves.
        result = joint.push_until_moves(direction, start_nm=direction * push_nm, max_nm=STOP_TORQUE_NM,
                                        rate_nm_s=TORQUE_RAMP_NM_S, phase="stall check")
        if not result.moved:
            LOGGER.info(f"  stop at {result.angle_deg:+.2f} deg (from start), pushing {STOP_TORQUE_NM:g} Nm")
            joint.move_by(-direction * RELAX_DEG, phase="relax")
            return result.motor_rad
        # A stiff spot: it needed more push than we gave, so push harder from now on. 1.5 x the
        # torque that moved it keeps the approach at a useful speed (about 1/3 of APPROACH_SPEED_DEG_S).
        push_nm = max(push_nm, 1.5 * abs(result.torque_nm), abs(result.torque_nm) + APPROACH_MARGIN_NM)
        push_nm = min(push_nm, STOP_TORQUE_NM - APPROACH_MARGIN_NM)
        LOGGER.info(f"  stiff spot at {result.angle_deg:+.1f} deg (moved at {result.torque_nm:+.2f} Nm); "
                    f"continuing with {push_nm:.2f} Nm")
    raise JointError(f"{MAX_FALSE_STALLS} stalls that were not the stop (friction too high?)")


def main() -> None:
    with Joint(motor_id=MOTOR_ID, feedback_id=FEEDBACK_ID, log_name="home_auto_10to1", log_dir=ROOT / "logs_10to1") as joint:
        print("Automatic homing: the joint will move by itself. Keep hands clear; nobody wearing the exo.")
        input("Press Enter to start (Ctrl-C to cancel)... ")

        # Step 1: breakaway test at the starting position
        LOGGER.info("Step 1: breakaway test")
        up = breakaway(joint, +1)
        if up is not None:
            joint.move_by(+CLEARANCE_DEG, phase="clear")  # off the - stop, in case it rested there
            down = breakaway(joint, -1)
        else:  # resting on the + stop: test downward first
            LOGGER.info("  resting on the + stop; testing downward first")
            down = breakaway(joint, -1)
            if down is not None:
                joint.move_by(-CLEARANCE_DEG, phase="clear")
                up = breakaway(joint, +1)
        if up is None or down is None:
            raise JointError(f"the joint did not move with {STOP_TORQUE_NM:g} Nm in either direction: jammed?")
        static_friction = (up - down) / 2
        gravity_here = (up + down) / 2
        LOGGER.info(f"  moves up at {up:+.3f} Nm, down at {down:+.3f} Nm: "
                    f"static friction {static_friction:.3f} Nm, gravity here {gravity_here:+.3f} Nm")
        push_nm = max(abs(up), abs(down)) + APPROACH_MARGIN_NM
        push_nm = min(max(push_nm, MIN_APPROACH_NM), STOP_TORQUE_NM - APPROACH_MARGIN_NM)
        LOGGER.info(f"  approach push {push_nm:.2f} Nm")

        # Steps 2 and 3: the two stops
        LOGGER.info(f"Step 2: first stop ({'+' if FIRST_DIRECTION > 0 else '-'} direction)")
        stop_1 = find_stop(joint, FIRST_DIRECTION, push_nm)
        LOGGER.info(f"Step 3: second stop ({'+' if FIRST_DIRECTION < 0 else '-'} direction)")
        stop_2 = find_stop(joint, -FIRST_DIRECTION, push_nm)

        # Step 4: angle 0 = mid-stroke
        cal = Calibration.from_stops(stop_1, stop_2, "auto", joint.motor.angle_offsets, joint.config)
        cal.breakaway = {"up_nm": up, "down_nm": down, "static_friction_nm": static_friction,
                         "gravity_at_start_nm": gravity_here}
        cal.save()
        joint.start_session(cal)
        LOGGER.info(f"Step 4: stroke {cal.stroke_deg:.2f} deg, stops at +/-{cal.limit_deg:.2f} deg; homed")

        # Step 5: friction and gravity
        LOGGER.info("Step 5: friction / gravity identification")
        comp = run_identification(joint, IDENTIFY_SPEEDS_DEG_S, IDENTIFY_MARGIN_DEG, IDENTIFY_ACCEL_DEG_S2)
        for line in comp.describe():
            LOGGER.info(f"  {line}")
        cal.compensation = comp.as_dict()
        cal.save()
        joint.start_session(cal)  # use the compensation from now on

        # Step 6: to mid-stroke
        joint.move_to(0.0, phase="to 0")
        joint.hold_until_enter(0.0, "Homed. Holding 0 deg; press Enter to release.")
        LOGGER.info("Released.")


if __name__ == "__main__":
    main()
