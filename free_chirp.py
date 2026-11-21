"""Free chirp: sweep the joint angle sinusoidally with rising frequency (log chirp), link free.

    pixi run chirp-free

The joint follows a position chirp centred at 0 deg in position control (spring-damper KP / KD,
with the friction / gravity feedforward from homing). Two frequency responses come out:
    tracking    measured angle / reference     how well the position control follows
    impedance   torque / angle                 inertia J of link + motor, and equivalent damping c
See exo/chirp.py for the reference and the analysis.

Before: home since the last motor power-up (pixi run home-auto). Nobody wearing the exo.

Steps:
    1. Reuse this power session's homing, move to 0 deg, measure the holding torque there.
    2. Hold until Enter.
    3. The sweep (DURATION_S), then back to 0 deg; hold until Enter, release.
    4. Analyse and save the results (also after an abort, as far as the sweep got).
Guards (abort, motor disabled): the torque beyond the holding torque over TORQUE_ABORT_NM for 3 replies
in a row, the angle more than ENVELOPE_MARGIN_DEG beyond the amplitude, the rotor warming by more
than TEMP_RISE_ABORT_C. Behind them: the 1.5 Nm torque trip and the stop limits.

DRY_RUN = True: the same loop with the motor off (no homing needed, nothing moves): checks loop
timing and CAN replies at RATE_HZ.

Output (logs/free_chirp_<time>*): .csv / .log (moves and holds), _chirp.csv (the sweep, per tick),
_frf.csv (responses per frequency), _summary.txt.
"""

from exo.chirp import Chirp, analyse, save_results, summarize
from exo.joint import Joint, JointError
from exo.logs import LOGGER

# ---- Motor ---- (the CAN interface is JointConfig.channel in exo/config.py)
MOTOR_ID = 0x02
FEEDBACK_ID = 0x12

# ---- Sweep settings ----
# Conservative first-run values. The link's inertia J is not known yet: once a sweep has measured
# it (summary: inertia fit), the amplitude and limits can be raised (inertia torque = J * accel).
F0_HZ = 0.5  # start frequency
F1_HZ = 10.0  # end frequency (at most RATE_HZ / 20)
DURATION_S = 60.0
AMPLITUDE_DEG = 3.0
MAX_SPEED_DEG_S = 60.0  # the amplitude is reduced at high frequency to stay under these
MAX_ACCEL_DEG_S2 = 600.0
RATE_HZ = 500.0  # control loop rate during this test
KP = 20.0  # position control stiffness [Nm/rad]
KD = 0.3  # position control damping [Nm s/rad]
POINTS = 30  # frequencies analysed
DRY_RUN = False

# ---- Guards ----
TORQUE_ABORT_NM = 0.6  # torque on top of the holding torque
ENVELOPE_MARGIN_DEG = 3.0
TEMP_RISE_ABORT_C = 15.0


def main() -> None:
    if not 0 < F0_HZ < F1_HZ <= RATE_HZ / 20:
        raise SystemExit(f"need 0 < F0_HZ < F1_HZ <= RATE_HZ / 20 ({RATE_HZ / 20:g} Hz)")
    chirp = Chirp(F0_HZ, F1_HZ, DURATION_S, AMPLITUDE_DEG, MAX_SPEED_DEG_S, MAX_ACCEL_DEG_S2)
    settings = (f"{'DRY RUN (motor off); ' if DRY_RUN else ''}chirp {F0_HZ:g} -> {F1_HZ:g} Hz in {DURATION_S:g} s "
                f"at {RATE_HZ:g} Hz; amplitude {AMPLITUDE_DEG:g} deg limited by {MAX_SPEED_DEG_S:g} deg/s and "
                f"{MAX_ACCEL_DEG_S2:g} deg/s^2 ({chirp.amp(F1_HZ):.2f} deg at {F1_HZ:g} Hz); kp {KP:g}, kd {KD:g}")
    print(settings)

    rows = []  # one row per tick of the sweep, columns as exo.chirp.COLUMNS
    stem = None
    try:
        with Joint(motor_id=MOTOR_ID, feedback_id=FEEDBACK_ID, log_name="free_chirp", rate_hz=RATE_HZ) as joint:
            stem = joint.log_stem
            LOGGER.debug(f"settings: {settings}")  # in the .log file

            # Steps 1-2: to 0 deg and wait
            hold_nm = 0.0
            if not DRY_RUN:
                joint.resume()
                joint.move_to(0.0, phase="to 0")
                hold_nm = joint.measure_holding_torque(torque_nm=joint.compensation_torque(0.0, 0.0))
                LOGGER.info(f"Holding torque at 0 deg: {hold_nm:+.3f} Nm")
                joint.hold_until_enter(0.0, f"At 0 deg. Press Enter to start the sweep (~{DURATION_S:.0f} s; "
                                            "Ctrl-C aborts).")

            # Step 3: the sweep
            joint.phase = "sweep"
            lead = 0.5 / RATE_HZ  # the firmware holds each target for one tick: aim at its middle
            envelope = AMPLITUDE_DEG + ENVELOPE_MARGIN_DEG
            rotor_at_start = joint.motor.rotor_temperature
            over, next_print = 0, 0.0
            loop = joint.loop()
            for t in loop:
                # Command: the reference half a tick ahead, plus friction / gravity feedforward
                target, target_vel = chirp.reference(t + lead)
                if not DRY_RUN:
                    ff = joint.compensation_torque(target, target_vel, target - joint.angle_deg)
                    joint.command_position(target, target_vel, ff, kp=KP, kd=KD)
                joint.update()

                # Record: when the reply was measured, and what the joint should have been at then
                t_sample = joint.sample_time - loop.start_time
                rows.append((t, t_sample, int(joint.fresh), target, chirp.position(t_sample),
                             joint.angle_deg, joint.velocity_deg_s, joint.torque_nm))

                # Guards
                if not DRY_RUN and joint.fresh:
                    over = over + 1 if abs(joint.torque_nm - hold_nm) > TORQUE_ABORT_NM else 0
                    if over >= 3:
                        raise JointError(f"torque {joint.torque_nm:+.3f} Nm (holding {hold_nm:+.3f}) at "
                                         f"{chirp.freq(t):.2f} Hz: more than {TORQUE_ABORT_NM:g} Nm extra")
                    if abs(joint.angle_deg) > envelope:
                        raise JointError(f"angle {joint.angle_deg:+.2f} deg, beyond +/-{envelope:.1f} deg")
                    if joint.motor.rotor_temperature - rotor_at_start > TEMP_RISE_ABORT_C:
                        raise JointError(f"rotor warmed by more than {TEMP_RISE_ABORT_C:g} C")

                # Progress, once a second
                if t >= next_print:
                    f = chirp.freq(t)
                    print(f"  t {t:5.1f} s  f {f:6.2f} Hz  amplitude {chirp.amp(f) * chirp.envelope(t):5.2f} deg  "
                          f"angle {joint.angle_deg:+6.2f} deg  torque {joint.torque_nm:+.3f} Nm", end="\r")
                    next_print = t + 1.0
                if t >= DURATION_S + 0.5:  # a little longer, to settle at 0 deg
                    print()
                    break
            # LOGGER.flush_buffer()  # flushed safely after motor.stop()

            # Back to 0 deg
            if not DRY_RUN:
                joint.move_to(0.0, phase="to 0")
                joint.hold_until_enter(0.0, "Sweep done. Holding 0 deg; press Enter to release.")
                LOGGER.info("Released.")
    finally:
        # Step 4: results (also after an abort)
        if rows:
            try:
                frf = [] if DRY_RUN else analyse(rows, chirp, POINTS)
                summary = f"Free chirp {stem.name}\n" + summarize(rows, frf, chirp, settings)
                print("\n" + summary)
                save_results(stem, rows, frf, summary)
                LOGGER.info(f"Results: {stem}_chirp.csv, _frf.csv, _summary.txt")
            except Exception as e:  # never hide the run's own error behind a problem here
                LOGGER.error(f"no results: {e}")


if __name__ == "__main__":
    main()
