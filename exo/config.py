"""Parameters of the joint shared by all scripts. Values marked PLACEHOLDER are first guesses for the
rigid-link elbow joint and still have to be checked on the hardware.

A script can change any of them for one run:
    from dataclasses import replace
    config = replace(JointConfig(), kp=30.0)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class JointConfig:
    # --- Mechanism ---
    # The link is bolted to the motor's output shaft (no transmission), so joint angle = motor angle.
    # +1: positive joint angle = positive motor direction. PLACEHOLDER: check which way is flexion.
    output_sign: int = +1
    design_stroke_deg: float = 150.0  # PLACEHOLDER: rough travel between the hard stops
    stroke_tol_deg: float = 45.0  # homing refuses a stroke further off the design value than this

    # --- Control loop ---
    rate_hz: float = 200.0

    # --- Safety ---
    torque_trip_nm: float = 1.5  # any measured torque above this disables the motor
    range_margin_deg: float = 5.0  # once homed: abort if the angle reads this far beyond a stop
    soft_margin_deg: float = 5.0  # once homed: move targets are kept this far inside the stops

    # --- Position control (spring-damper in the firmware) --- PLACEHOLDER
    kp: float = 20.0  # Nm/rad (0.35 Nm per degree of error)
    kd: float = 0.3  # Nm s/rad

    # --- Moves (move_to) ---
    move_speed_deg_s: float = 30.0
    move_accel_deg_s2: float = 60.0
    settle_s: float = 0.3  # move_to keeps tracking the end point this long before returning
    # Abort if the joint lags its planned position by more than this, for track_confirm replies in
    # a row (an obstacle, or a person resisting). Generous until gravity is compensated.
    track_tol_deg: float = 5.0
    track_confirm: int = 3
    # Targets are sent half a tick ahead of the plan: the firmware holds each one for a whole tick.
    target_lead_ticks: float = 0.5

    # --- Friction / gravity feedforward (uses the values identified by homing) ---
    friction_scale: float = 0.8  # fraction of the identified friction fed forward (1.0 can overshoot)
    friction_v_band_deg_s: float = 1.0  # moving: friction direction fades in over this speed
    friction_err_deg: float = 0.5  # holding still: full friction push at this position error

    # --- Homing marker ---
    # The CAN watchdog register (RAM, cleared at power-up) doubles as a "homed since power-up" flag.
    # Both values give the same ~100 ms watchdog.
    timeout_unhomed: int = 2000
    timeout_homed: int = 2001
