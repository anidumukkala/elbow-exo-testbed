"""Friction and gravity of the joint: the model, how homing identifies it, and the feedforward torque.

Model (joint side, angle theta in rad with 0 = mid-stroke, velocity v in rad/s):

    torque = coulomb * sign(v)  +  viscous * v  +  gravity_cos * cos(theta) + gravity_sin * sin(theta)
             dry friction          viscous         the link's weight (any orientation and offset)

How it is identified (run_identification, the last step of homing): the joint moves from one end
of the stroke to the other and back at constant speed, at two speeds. At constant speed there is no
inertia torque, so the measured torque is exactly the model above. Friction changes sign with the
direction and gravity does not, which separates them; the two speeds separate dry from viscous
friction. All four numbers come out of one least-squares fit.

The gravity terms combine into  m*g*l * cos(theta - theta_max): m*g*l (link mass x gravity x
distance to its centre of mass) is the largest holding torque, needed at theta_max (link horizontal).
A horizontal link gives gravity terms near zero.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from .logs import log


@dataclass
class Compensation:
    coulomb_nm: float  # dry friction
    viscous_nm_s_rad: float  # viscous friction
    gravity_cos_nm: float
    gravity_sin_nm: float
    fit_rms_nm: float = math.nan  # how far the measured torques are from the fitted model
    samples: int = 0

    def gravity(self, angle_deg: float) -> float:
        """Torque that holds the link against gravity at angle_deg [Nm]."""
        th = math.radians(angle_deg)
        return self.gravity_cos_nm * math.cos(th) + self.gravity_sin_nm * math.sin(th)

    def friction(self, velocity_deg_s: float, error_deg: float, v_band_deg_s: float, err_band_deg: float) -> float:
        """Torque that overcomes friction [Nm]. Moving: in the direction of motion, fading in over
        v_band_deg_s (so it passes smoothly through zero at a reversal). Holding still: proportional
        to the position error, full at err_band_deg, to push through stiction."""
        if velocity_deg_s != 0.0:
            direction = math.tanh(velocity_deg_s / v_band_deg_s)
        else:
            direction = max(-1.0, min(1.0, error_deg / err_band_deg))
        return direction * self.coulomb_nm + self.viscous_nm_s_rad * math.radians(velocity_deg_s)

    def torque(self, angle_deg: float, velocity_deg_s: float, error_deg: float, friction_scale: float,
               v_band_deg_s: float, err_band_deg: float) -> float:
        """Feedforward: all of gravity plus friction_scale of the friction [Nm]."""
        return self.gravity(angle_deg) + friction_scale * self.friction(velocity_deg_s, error_deg, v_band_deg_s,
                                                                         err_band_deg)

    def describe(self) -> list[str]:
        mgl = math.hypot(self.gravity_cos_nm, self.gravity_sin_nm)
        th_max = math.degrees(math.atan2(self.gravity_sin_nm, self.gravity_cos_nm))
        return [f"dry friction     {self.coulomb_nm:.3f} Nm",
                f"viscous friction {self.viscous_nm_s_rad:.4f} Nm s/rad ({self.viscous_nm_s_rad * math.radians(30):.3f} Nm at 30 deg/s)",
                f"gravity          m*g*l = {mgl:.3f} Nm, largest at {th_max:+.0f} deg (link horizontal there)",
                f"fit              {self.samples} samples, residual {self.fit_rms_nm:.3f} Nm rms"]

    def as_dict(self) -> dict:
        return asdict(self)


def fit(angle_deg, velocity_deg_s, torque_nm) -> Compensation:
    """Least-squares fit of the model to constant-speed samples (velocity = the planned speed)."""
    th = np.radians(np.asarray(angle_deg, dtype=float))
    v = np.radians(np.asarray(velocity_deg_s, dtype=float))
    tau = np.asarray(torque_nm, dtype=float)
    columns = [np.sign(v), v, np.cos(th), np.sin(th)]
    two_speeds = len(np.unique(np.round(np.abs(v), 6))) >= 2
    if not two_speeds:  # with a single speed, dry and viscous friction cannot be told apart
        columns.pop(1)
    X = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(X, tau, rcond=None)
    rms = float(np.sqrt(np.mean((tau - X @ coef) ** 2)))
    if not two_speeds:
        coef = np.insert(coef, 1, 0.0)
    return Compensation(coulomb_nm=float(coef[0]), viscous_nm_s_rad=float(coef[1]), gravity_cos_nm=float(coef[2]),
                        gravity_sin_nm=float(coef[3]), fit_rms_nm=rms, samples=len(tau))


def run_identification(joint, speeds_deg_s: list, margin_deg: float, accel_deg_s2: float,
                       settle_s: float = 0.5) -> Compensation:
    """Move end to end and back at each speed (position control), record angle, planned speed and
    measured torque while cruising, and fit the model. The joint must be homed. Ends at -end.
    The first settle_s of each cruise is skipped: the joint is still settling onto the planned speed
    then, and its acceleration torque would bias the fit."""
    end = joint.calibration.limit_deg - margin_deg
    if end < 20:
        raise ValueError(f"stroke too small for the identification sweeps (+/-{end:.0f} deg)")
    angles, speeds, torques = [], [], []
    cruise = 0.0
    cruising_ticks = 0

    def record() -> None:
        # Only while cruising (planned speed constant, settled): no inertia torque then.
        nonlocal cruising_ticks
        if not math.isclose(abs(joint.setpoint_vel_deg_s), cruise, rel_tol=1e-6):
            cruising_ticks = 0
            return
        cruising_ticks += 1
        if joint.fresh and cruising_ticks > settle_s * joint.config.rate_hz:
            angles.append(joint.angle_deg)
            speeds.append(joint.setpoint_vel_deg_s)
            torques.append(joint.torque_nm)

    joint.move_to(-end, phase="identify start")
    for cruise in speeds_deg_s:
        for target in (+end, -end):
            log.info(f"  identification sweep to {target:+.0f} deg at {cruise:g} deg/s")
            joint.move_to(target, cruise, accel_deg_s2, phase=f"identify {cruise:g} deg/s", on_tick=record)
    if len(torques) < 50:
        raise RuntimeError(f"only {len(torques)} samples while cruising; use a longer stroke or lower speeds")
    comp = fit(angles, speeds, torques)
    if comp.coulomb_nm < 0 or comp.viscous_nm_s_rad < 0:
        log.warning(f"negative friction fitted (dry {comp.coulomb_nm:+.4f} Nm, viscous {comp.viscous_nm_s_rad:+.4f} "
                    "Nm s/rad): too small to measure here; using 0 for it")
        comp.coulomb_nm, comp.viscous_nm_s_rad = max(0.0, comp.coulomb_nm), max(0.0, comp.viscous_nm_s_rad)
    if 2 * end < 60:
        log.warning("the sweeps cover less than 60 deg: the two gravity terms are poorly separated")
    return comp
