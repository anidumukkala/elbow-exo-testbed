"""Passive exoskeleton test (Experiment 2): data schema, signal processing, transparency metrics,
and system identification of human forearm inertia and skin-cuff interface friction.

Dynamics model (joint side, angle q in rad with 0 = mid-stroke, velocity qd in rad/s, accel qdd in rad/s^2):

    Exo baseline (identified in Experiment 1: homing + free_chirp):
        tau_exo(q, qd, qdd) = J_exo * qdd + tau_c_exo * tanh(qd / v_band) + b_exo * qd + g_exo(q)

    Interaction torque estimated from exo dynamics (Phase B, zero-torque mode, tau_m ~ 0):
        tau_h_hat = tau_exo(q, qd, qdd) - tau_m

    Coupled relaxed human forearm + cuff + exo system ID (Phase C):
      Mode 1 (default, recommended for hard-stop-limited rigs): Gentle motor-driven relaxed chirp
        tau_m = J_total * qdd + tau_c_total * tanh(qd / v_band) + b_total * qd + k_eff * q + tau_0
        Stays centered at 0 deg (+/-8 deg), works at any rig orientation and friction level, never
        hits mechanical hard stops, and directly measures torque scale from motor current tau_m.

      Mode 2 (legacy, requires hanging equilibrium well inside mechanical stops): Free-decay swings
        J_total * qdd + tau_c_total * tanh(qd / v_band) + b_total * qd = tau_m - g_total(q)

    In both modes:
        J_human      = J_total     - J_exo          (human forearm + hand inertia about elbow)
        tau_c_cuff   = tau_c_total - tau_c_exo      (skin-cuff interface + passive elbow dry friction)
        b_cuff       = b_total     - b_exo          (skin-cuff interface + passive elbow viscous damping)
        g_arm(q)     = g_total(q)  - g_exo(q)       (human forearm gravity torque, from Phase A holds)

Everything in this module uses only Python stdlib + NumPy so it runs directly inside the existing
Raspberry Pi `pixi` environment without additional dependencies.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .calibration import CAL_FILE, Calibration
from .logs import write_csv

# Per-tick data columns written to <stem>_passive.csv during passive_exo.py
PASSIVE_COLUMNS = [
    "t",               # loop elapsed time [s]
    "t_sample",        # motor feedback timestamp [s, same monotonic clock]
    "fresh",           # 1 if a new CAN reply arrived on this tick
    "phase",           # test phase: gravity_hold_1_pos, active_swing, relaxed_chirp, free_decay_1, ...
    "trial",           # integer trial index within phase (0 if not applicable)
    "angle_deg",       # joint angle [deg], 0 = mid-stroke
    "velocity_deg_s",  # joint angular velocity [deg/s]
    "torque_nm",       # measured motor torque [Nm] (firmware current estimate)
    "tau_cmd_nm",      # commanded total torque [Nm] (0.0 in zero-torque mode)
    "t_mos_c",         # MOSFET temperature [C]
    "t_rotor_c",       # rotor/coil temperature [C]
]

# Smooth sign(qd) transition band [deg/s] for Coulomb friction regression
DEFAULT_V_BAND_DEG_S = 2.0


@dataclass
class ExoBaseline:
    """Exoskeleton-only dynamic parameters from Experiment 1 (homing + free_chirp)."""
    inertia_kg_m2: float = 0.0
    coulomb_nm: float = 0.0
    viscous_nm_s_rad: float = 0.0
    gravity_cos_nm: float = 0.0
    gravity_sin_nm: float = 0.0
    source_cal: str = ""
    source_frf: str = ""

    def gravity(self, q_rad: np.ndarray) -> np.ndarray:
        return self.gravity_cos_nm * np.cos(q_rad) + self.gravity_sin_nm * np.sin(q_rad)

    def friction(self, qd_rad_s: np.ndarray, v_band_deg_s: float = DEFAULT_V_BAND_DEG_S) -> np.ndarray:
        v_band = math.radians(max(v_band_deg_s, 1e-3))
        return self.coulomb_nm * np.tanh(qd_rad_s / v_band) + self.viscous_nm_s_rad * qd_rad_s

    def torque(self, q_rad: np.ndarray, qd_rad_s: np.ndarray, qdd_rad_s2: np.ndarray,
               v_band_deg_s: float = DEFAULT_V_BAND_DEG_S) -> np.ndarray:
        return self.inertia_kg_m2 * qdd_rad_s2 + self.friction(qd_rad_s, v_band_deg_s) + self.gravity(q_rad)


@dataclass
class GravityFit:
    """Fitted gravity model g(q) = gravity_cos_nm * cos(q) + gravity_sin_nm * sin(q)."""
    gravity_cos_nm: float
    gravity_sin_nm: float
    mgl_nm: float
    horizontal_angle_deg: float
    fit_rms_nm: float
    n_angles: int
    hold_angles_deg: list[float] = field(default_factory=list)
    hold_torques_nm: list[float] = field(default_factory=list)
    source: str = "measured"  # "measured", "measured_bidirectional", "anthropometric", or "none"
    bidirectional: bool = False
    stiction_halfwidth_nm: float = math.nan

    def torque(self, q_rad: np.ndarray) -> np.ndarray:
        return self.gravity_cos_nm * np.cos(q_rad) + self.gravity_sin_nm * np.sin(q_rad)


@dataclass
class TransparencyMetrics:
    """Mechanical transparency & backdrivability metrics during active zero-torque swings."""
    duration_s: float
    n_samples: int
    angle_min_deg: float
    angle_max_deg: float
    peak_speed_deg_s: float
    rms_speed_deg_s: float
    peak_accel_deg_s2: float
    rms_accel_deg_s2: float
    peak_motor_torque_nm: float
    rms_motor_torque_nm: float
    peak_interaction_torque_nm: float
    rms_interaction_torque_nm: float
    rms_inertial_torque_nm: float
    rms_friction_torque_nm: float
    rms_gravity_torque_nm: float
    total_work_j: float
    mean_power_w: float
    speed_bins: list[dict] = field(default_factory=list)


@dataclass
class DecayTrialFit:
    """System-ID fit for a single relaxed Phase C trial (free-decay or chirp)."""
    trial: int
    duration_s: float
    n_samples: int
    angle_span_deg: float
    peak_speed_deg_s: float
    j_total_kg_m2: float
    coulomb_total_nm: float
    viscous_total_nm_s_rad: float
    r2: float
    rms_residual_nm: float
    cond_number: float
    natural_freq_hz: float = math.nan
    damping_ratio: float = math.nan


@dataclass
class PassiveIdResult:
    """Pooled identification result for human forearm inertia + skin-cuff interface friction."""
    j_total_kg_m2: float
    j_total_se_kg_m2: float
    coulomb_total_nm: float
    coulomb_total_se_nm: float
    viscous_total_nm_s_rad: float
    viscous_total_se_nm_s_rad: float
    j_human_kg_m2: float
    coulomb_cuff_nm: float
    viscous_cuff_nm_s_rad: float
    r2: float
    rms_residual_nm: float
    cond_number: float
    n_trials: int
    n_samples: int
    method: str = "free_decay"  # "relaxed_chirp" or "free_decay"
    k_eff_nm_rad: float = math.nan
    tau_offset_nm: float = math.nan
    per_trial: list[DecayTrialFit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class LoadcellValidation:
    """Optional cross-validation against an external cuff loadcell."""
    n_samples: int
    rmse_nm: float
    mae_nm: float
    pearson_r: float
    residual_j_kg_m2: float
    residual_coulomb_nm: float
    residual_viscous_nm_s_rad: float
    residual_r2: float


@dataclass
class PassiveAnalysisReport:
    """Complete Experiment 2 analysis output."""
    csv_path: str
    sample_rate_hz: float
    fresh_fraction: float
    exo_baseline: ExoBaseline
    gravity_total: Optional[GravityFit] = None
    gravity_arm: Optional[GravityFit] = None
    transparency: Optional[TransparencyMetrics] = None
    passive_id: Optional[PassiveIdResult] = None
    loadcell: Optional[LoadcellValidation] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------------------------
# Exo Baseline Loaders (Experiment 1 outputs)
# ----------------------------------------------------------------------------------------------
def load_exo_baseline(
    cal_path: Optional[Path] = None,
    frf_path: Optional[Path] = None,
    inertia_override_kg_m2: Optional[float] = None,
    coulomb_override_nm: Optional[float] = None,
    viscous_override_nm_s_rad: Optional[float] = None,
    min_frf_hz: float = 2.0,
) -> ExoBaseline:
    """Load exo-only dynamics from calibration/joint_cal.json and a free_chirp_*_frf.csv file."""
    base = ExoBaseline()

    # 1. Homing calibration (friction + gravity)
    cp = Path(cal_path) if cal_path is not None else CAL_FILE
    if cp.exists():
        cal = Calibration.load(cp)
        comp = cal.compensation or {}
        base.coulomb_nm = float(comp.get("coulomb_nm", 0.0))
        base.viscous_nm_s_rad = float(comp.get("viscous_nm_s_rad", 0.0))
        base.gravity_cos_nm = float(comp.get("gravity_cos_nm", 0.0))
        base.gravity_sin_nm = float(comp.get("gravity_sin_nm", 0.0))
        base.source_cal = str(cp)

    # 2. Free-chirp FRF (reflected inertia J_exo = J_a + J_l)
    if frf_path is not None and Path(frf_path).exists():
        base.inertia_kg_m2 = fit_inertia_from_frf_csv(Path(frf_path), min_hz=min_frf_hz)
        base.source_frf = str(frf_path)

    # 3. Explicit CLI overrides
    if inertia_override_kg_m2 is not None:
        base.inertia_kg_m2 = float(inertia_override_kg_m2)
        base.source_frf = base.source_frf or "cli_override"
    if coulomb_override_nm is not None:
        base.coulomb_nm = float(coulomb_override_nm)
    if viscous_override_nm_s_rad is not None:
        base.viscous_nm_s_rad = float(viscous_override_nm_s_rad)

    return base


def fit_inertia_from_frf_csv(frf_csv: Path, min_hz: float = 2.0) -> float:
    """Reproduce exo/chirp.py's high-frequency inertia fit Re(Z) = -J * w^2 + k0 from _frf.csv."""
    freqs, inertias = [], []
    with open(frf_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fz = float(row["f_hz"])
                jj = float(row["inertia_kg_m2"])
            except (KeyError, ValueError):
                continue
            if math.isfinite(fz) and math.isfinite(jj) and fz >= min_hz:
                freqs.append(fz)
                inertias.append(jj)
    if len(freqs) < 3:
        all_j = []
        with open(frf_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    jj = float(row["inertia_kg_m2"])
                    if math.isfinite(jj):
                        all_j.append(jj)
                except (KeyError, ValueError):
                    pass
        if not all_j:
            raise ValueError(f"No valid inertia rows in {frf_csv}")
        return float(np.median(all_j))

    w2 = (2.0 * math.pi * np.asarray(freqs, dtype=float)) ** 2
    re_z = -np.asarray(inertias, dtype=float) * w2
    A = np.column_stack([-w2, np.ones_like(w2)])
    (J, _k0), *_ = np.linalg.lstsq(A, re_z, rcond=None)
    return float(J)


# ----------------------------------------------------------------------------------------------
# Pure-NumPy Zero-Phase Savitzky-Golay Smoothing & Differentiation
# ----------------------------------------------------------------------------------------------
def savgol_coeffs(window_length: int, polyorder: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    """Compute Savitzky-Golay FIR filter coefficients using pure NumPy (no SciPy required)."""
    if window_length % 2 == 0 or window_length < 3:
        raise ValueError("window_length must be an odd integer >= 3")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")
    if deriv > polyorder:
        raise ValueError("deriv must be <= polyorder")

    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, N=polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    coeffs = pinv[deriv] * math.factorial(deriv) / (delta ** deriv)
    return coeffs[::-1]


def smooth_kinematics(
    t_s: np.ndarray,
    angle_deg: np.ndarray,
    velocity_deg_s: Optional[np.ndarray] = None,
    window_s: float = 0.06,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample onto a uniform time grid and compute zero-phase smoothed q, qd, qdd [rad, rad/s, rad/s^2]."""
    t = np.asarray(t_s, dtype=float)
    q_raw = np.radians(np.asarray(angle_deg, dtype=float))
    order = np.argsort(t)
    t, q_raw = t[order], q_raw[order]
    keep = np.concatenate(([True], np.diff(t) > 1e-7))
    t, q_raw = t[keep], q_raw[keep]

    dt = float(np.median(np.diff(t)))
    if dt <= 0 or len(t) < 15:
        raise ValueError("Segment too short or invalid timestamps for kinematic differentiation")

    t_u = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    q_u = np.interp(t_u, t, q_raw)

    win = max(polyorder + 2, int(round(window_s / dt)))
    if win % 2 == 0:
        win += 1
    if win >= len(t_u):
        win = len(t_u) - 1 if (len(t_u) - 1) % 2 == 1 else len(t_u) - 2
    if win <= polyorder:
        raise ValueError(f"Segment too short ({len(t_u)} samples) for Savitzky-Golay filter")

    c0 = savgol_coeffs(win, polyorder, deriv=0, delta=dt)
    c1 = savgol_coeffs(win, polyorder, deriv=1, delta=dt)
    c2 = savgol_coeffs(win, polyorder, deriv=2, delta=dt)

    q_f = np.convolve(q_u, c0, mode="valid")
    qd_f = np.convolve(q_u, c1, mode="valid")
    qdd_f = np.convolve(q_u, c2, mode="valid")

    half = win // 2
    t_trim = t_u[half:len(t_u) - half]

    if velocity_deg_s is not None:
        v_raw = np.radians(np.asarray(velocity_deg_s, dtype=float))[order][keep]
        v_u = np.interp(t_u, t, v_raw)
        v_f = np.convolve(v_u, c0, mode="valid")
    else:
        v_f = qd_f.copy()

    return t_trim, q_f, qd_f, qdd_f, v_f


# ----------------------------------------------------------------------------------------------
# CSV Loading & Saving
# ----------------------------------------------------------------------------------------------
def save_passive_csv(path: Path, rows: Sequence[Sequence]) -> None:
    """Write per-tick passive exo data to CSV."""
    write_csv(Path(path), PASSIVE_COLUMNS, list(rows))


def load_passive_csv(path: Path) -> list[dict]:
    """Read a passive_exo CSV (either _passive.csv or a compatible Joint log CSV)."""
    records = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t_val = float(row.get("t_sample") or row.get("t") or row.get("t_s") or "nan")
                t_loop = float(row.get("t") or row.get("t_s") or t_val)
                fresh = int(float(row.get("fresh", "1") or "1"))
                angle = float(row["angle_deg"])
                vel = float(row["velocity_deg_s"])
                tau = float(row["torque_nm"])
                tau_cmd = float(row.get("tau_cmd_nm") or "0.0")
                trial = int(float(row.get("trial", "0") or "0"))
                phase = (row.get("phase") or "").strip()
                loadcell = float(row["loadcell_nm"]) if row.get("loadcell_nm") not in (None, "") else math.nan
            except (KeyError, ValueError):
                continue
            if not (math.isfinite(t_val) and math.isfinite(angle) and math.isfinite(tau)):
                continue
            records.append({
                "t": t_loop,
                "t_sample": t_val,
                "fresh": fresh,
                "phase": phase,
                "trial": trial,
                "angle_deg": angle,
                "velocity_deg_s": vel,
                "torque_nm": tau,
                "tau_cmd_nm": tau_cmd,
                "loadcell_nm": loadcell,
            })
    return records


# ----------------------------------------------------------------------------------------------
# Phase A: Static Gravity Calibration (Relaxed Forearm + Exo, Bidirectional Stiction Cancellation)
# ----------------------------------------------------------------------------------------------
def fit_gravity_from_holds(
    records: list[dict],
    exo_baseline: ExoBaseline,
    settle_fraction: float = 0.5,
) -> tuple[Optional[GravityFit], Optional[GravityFit]]:
    """Fit total gravity g_total(q) = gc*cos(q) + gs*sin(q) from Phase A static holds,
    and subtract g_exo(q) to obtain human forearm gravity g_arm(q).

    When bidirectional approach pairs (`gravity_hold_<i>_pos` and `gravity_hold_<i>_neg`) are
    present, averages the + and - approach torques at each angle to cancel Coulomb stiction bias.
    """
    holds_by_phase: dict[str, list[dict]] = {}
    for r in records:
        if r["fresh"] == 1 and r["phase"].startswith("gravity_hold"):
            holds_by_phase.setdefault(r["phase"], []).append(r)

    if not holds_by_phase:
        return None, None

    # Compute settled (mean_angle_deg, mean_torque_nm) per recorded hold phase
    phase_stats: dict[str, tuple[float, float]] = {}
    for phase_name in sorted(holds_by_phase):
        seg = holds_by_phase[phase_name]
        if len(seg) < 10:
            continue
        start_idx = int(len(seg) * settle_fraction)
        settled = seg[start_idx:]
        q_deg = np.array([r["angle_deg"] for r in settled], dtype=float)
        tau = np.array([r["torque_nm"] for r in settled], dtype=float)
        phase_stats[phase_name] = (float(np.mean(q_deg)), float(np.mean(tau)))

    # Group bidirectional _pos / _neg pairs by base name (e.g. "gravity_hold_1")
    grouped: dict[str, dict[str, tuple[float, float]]] = {}
    for pname, stats in phase_stats.items():
        if pname.endswith("_pos"):
            grouped.setdefault(pname[:-4], {})["pos"] = stats
        elif pname.endswith("_neg"):
            grouped.setdefault(pname[:-4], {})["neg"] = stats
        else:
            grouped.setdefault(pname, {})["single"] = stats

    hold_angles: list[float] = []
    hold_torques: list[float] = []
    stiction_halves: list[float] = []

    for base_name in sorted(grouped):
        entry = grouped[base_name]
        if "pos" in entry and "neg" in entry:
            q_p, t_p = entry["pos"]
            q_n, t_n = entry["neg"]
            hold_angles.append(0.5 * (q_p + q_n))
            hold_torques.append(0.5 * (t_p + t_n))
            stiction_halves.append(0.5 * abs(t_p - t_n))
        elif "single" in entry:
            q_s, t_s = entry["single"]
            hold_angles.append(q_s)
            hold_torques.append(t_s)
        else:
            # Only one direction recorded for this angle
            only = next(iter(entry.values()))
            hold_angles.append(only[0])
            hold_torques.append(only[1])

    if len(hold_angles) < 2:
        return None, None

    q_rad = np.radians(np.asarray(hold_angles, dtype=float))
    tau_arr = np.asarray(hold_torques, dtype=float)
    X = np.column_stack([np.cos(q_rad), np.sin(q_rad)])
    coef, *_ = np.linalg.lstsq(X, tau_arr, rcond=None)
    gc_tot, gs_tot = float(coef[0]), float(coef[1])
    rms = float(np.sqrt(np.mean((tau_arr - X @ coef) ** 2)))
    mgl_tot = math.hypot(gc_tot, gs_tot)
    th_tot = math.degrees(math.atan2(gs_tot, gc_tot))

    bidir = len(stiction_halves) >= 2
    stiction_hw = float(np.mean(stiction_halves)) if stiction_halves else math.nan
    src = "measured_bidirectional" if bidir else "measured"

    g_total = GravityFit(
        gravity_cos_nm=gc_tot,
        gravity_sin_nm=gs_tot,
        mgl_nm=mgl_tot,
        horizontal_angle_deg=th_tot,
        fit_rms_nm=rms,
        n_angles=len(hold_angles),
        hold_angles_deg=hold_angles,
        hold_torques_nm=hold_torques,
        source=src,
        bidirectional=bidir,
        stiction_halfwidth_nm=stiction_hw,
    )

    gc_arm = gc_tot - exo_baseline.gravity_cos_nm
    gs_arm = gs_tot - exo_baseline.gravity_sin_nm
    mgl_arm = math.hypot(gc_arm, gs_arm)
    th_arm = math.degrees(math.atan2(gs_arm, gc_arm))
    g_arm = GravityFit(
        gravity_cos_nm=gc_arm,
        gravity_sin_nm=gs_arm,
        mgl_nm=mgl_arm,
        horizontal_angle_deg=th_arm,
        fit_rms_nm=rms,
        n_angles=len(hold_angles),
        hold_angles_deg=hold_angles,
        hold_torques_nm=[
            t - float(exo_baseline.gravity(np.array([math.radians(a)]))[0])
            for a, t in zip(hold_angles, hold_torques)
        ],
        source=src,
        bidirectional=bidir,
        stiction_halfwidth_nm=stiction_hw,
    )
    return g_total, g_arm


def gravity_from_anthropometrics(
    arm_mgl_nm: float,
    arm_horizontal_deg: float,
    exo_baseline: ExoBaseline,
) -> tuple[GravityFit, GravityFit]:
    """Construct total and arm gravity models from user-supplied forearm m*g*l and horizontal angle."""
    th = math.radians(arm_horizontal_deg)
    gc_arm = arm_mgl_nm * math.cos(th)
    gs_arm = arm_mgl_nm * math.sin(th)
    gc_tot = gc_arm + exo_baseline.gravity_cos_nm
    gs_tot = gs_arm + exo_baseline.gravity_sin_nm
    g_arm = GravityFit(
        gravity_cos_nm=gc_arm,
        gravity_sin_nm=gs_arm,
        mgl_nm=arm_mgl_nm,
        horizontal_angle_deg=arm_horizontal_deg,
        fit_rms_nm=math.nan,
        n_angles=0,
        source="anthropometric",
    )
    g_tot = GravityFit(
        gravity_cos_nm=gc_tot,
        gravity_sin_nm=gs_tot,
        mgl_nm=math.hypot(gc_tot, gs_tot),
        horizontal_angle_deg=math.degrees(math.atan2(gs_tot, gc_tot)),
        fit_rms_nm=math.nan,
        n_angles=0,
        source="anthropometric",
    )
    return g_tot, g_arm


# ----------------------------------------------------------------------------------------------
# Phase B: Active Human Swinging in Zero-Torque Mode (Transparency & Backdrivability)
# ----------------------------------------------------------------------------------------------
def analyze_active_swing(
    records: list[dict],
    exo_baseline: ExoBaseline,
    window_s: float = 0.06,
    v_band_deg_s: float = DEFAULT_V_BAND_DEG_S,
) -> tuple[Optional[TransparencyMetrics], Optional[dict]]:
    """Compute mechanical transparency metrics and decomposed interaction torques during active swings."""
    seg = [r for r in records if r["fresh"] == 1 and r["phase"].startswith("active_swing")]
    if len(seg) < 50:
        return None, None

    t = np.array([r["t_sample"] for r in seg], dtype=float)
    ang = np.array([r["angle_deg"] for r in seg], dtype=float)
    vel = np.array([r["velocity_deg_s"] for r in seg], dtype=float)
    tau_m_raw = np.array([r["torque_nm"] for r in seg], dtype=float)

    t_u, q, qd, qdd, _ = smooth_kinematics(t, ang, vel, window_s=window_s)
    tau_m = np.interp(t_u, t, tau_m_raw)

    tau_inertia = exo_baseline.inertia_kg_m2 * qdd
    tau_friction = exo_baseline.friction(qd, v_band_deg_s=v_band_deg_s)
    tau_gravity = exo_baseline.gravity(q)
    tau_h_hat = tau_inertia + tau_friction + tau_gravity - tau_m

    power_w = tau_h_hat * qd
    dt_u = float(np.median(np.diff(t_u)))
    total_work_j = float(np.sum(np.abs(power_w)) * dt_u)
    duration = float(t_u[-1] - t_u[0])

    abs_speed_deg_s = np.abs(np.degrees(qd))
    bins = [
        ("slow (<60 deg/s)", abs_speed_deg_s < 60.0),
        ("medium (60-150 deg/s)", (abs_speed_deg_s >= 60.0) & (abs_speed_deg_s < 150.0)),
        ("fast (>=150 deg/s)", abs_speed_deg_s >= 150.0),
    ]
    speed_bins = []
    for label, mask in bins:
        n_b = int(np.sum(mask))
        if n_b >= 10:
            speed_bins.append({
                "bin": label,
                "n_samples": n_b,
                "rms_interaction_nm": float(np.sqrt(np.mean(tau_h_hat[mask] ** 2))),
                "peak_interaction_nm": float(np.max(np.abs(tau_h_hat[mask]))),
                "rms_motor_nm": float(np.sqrt(np.mean(tau_m[mask] ** 2))),
            })

    metrics = TransparencyMetrics(
        duration_s=duration,
        n_samples=len(t_u),
        angle_min_deg=float(np.min(np.degrees(q))),
        angle_max_deg=float(np.max(np.degrees(q))),
        peak_speed_deg_s=float(np.max(abs_speed_deg_s)),
        rms_speed_deg_s=float(np.sqrt(np.mean(np.degrees(qd) ** 2))),
        peak_accel_deg_s2=float(np.max(np.abs(np.degrees(qdd)))),
        rms_accel_deg_s2=float(np.sqrt(np.mean(np.degrees(qdd) ** 2))),
        peak_motor_torque_nm=float(np.max(np.abs(tau_m))),
        rms_motor_torque_nm=float(np.sqrt(np.mean(tau_m ** 2))),
        peak_interaction_torque_nm=float(np.max(np.abs(tau_h_hat))),
        rms_interaction_torque_nm=float(np.sqrt(np.mean(tau_h_hat ** 2))),
        rms_inertial_torque_nm=float(np.sqrt(np.mean(tau_inertia ** 2))),
        rms_friction_torque_nm=float(np.sqrt(np.mean(tau_friction ** 2))),
        rms_gravity_torque_nm=float(np.sqrt(np.mean(tau_gravity ** 2))),
        total_work_j=total_work_j,
        mean_power_w=total_work_j / duration if duration > 0 else 0.0,
        speed_bins=speed_bins,
    )

    series = {
        "t": t_u,
        "q_deg": np.degrees(q),
        "qd_deg_s": np.degrees(qd),
        "qdd_deg_s2": np.degrees(qdd),
        "tau_m": tau_m,
        "tau_inertia": tau_inertia,
        "tau_friction": tau_friction,
        "tau_gravity": tau_gravity,
        "tau_h_hat": tau_h_hat,
    }
    return metrics, series


# ----------------------------------------------------------------------------------------------
# Phase C (Default): Relaxed Motor-Driven Chirp System ID (Hard-Stop & Orientation Independent)
# ----------------------------------------------------------------------------------------------
def _build_passive_warnings(
    r2: float,
    j_human: float,
    tc_cuff: float,
    b_cuff: float,
    cond: float,
    method: str,
) -> list[str]:
    warnings: list[str] = []
    if r2 < 0.85:
        if method == "relaxed_chirp":
            warnings.append(
                f"Low pooled R^2 ({r2:.3f}): subject may have tensed/guarded during the relaxed chirp "
                "or cuff strap slipped on the forearm."
            )
        else:
            warnings.append(
                f"Low pooled R^2 ({r2:.3f}): subject may not have been fully relaxed during free decay, "
                "gravity calibration has an offset, or the arm contacted a hard stop."
            )
    if j_human <= 0:
        warnings.append(
            f"Estimated human inertia is non-positive ({j_human * 1e3:+.2f} e-3 kg m^2): "
            "check exo baseline J_exo and subject relaxation."
        )
    elif not (0.015 <= j_human <= 0.12):
        warnings.append(
            f"Estimated human forearm inertia ({j_human * 1e3:.2f} e-3 kg m^2) is outside the typical "
            "adult forearm+hand range (30-80 e-3 kg m^2); verify relaxation and cuff alignment."
        )
    if tc_cuff < -0.02 or b_cuff < -0.01:
        warnings.append(
            f"Negative interface friction component (coulomb {tc_cuff:+.4f} Nm, viscous {b_cuff:+.4f} Nm s/rad): "
            "possible subtle muscle assistance or exo baseline friction mismatch (e.g. cold vs warm motor)."
        )
    if cond > 250.0:
        warnings.append(
            f"High regression condition number ({cond:.1f}): check excitation amplitude and frequency range."
        )
    return warnings


def analyze_relaxed_chirp(
    records: list[dict],
    exo_baseline: ExoBaseline,
    window_s: float = 0.04,
    v_band_deg_s: float = DEFAULT_V_BAND_DEG_S,
    min_speed_deg_s: float = 2.0,
    trim_edge_s: float = 1.0,
    stop_limit_deg: Optional[float] = None,
    stop_margin_deg: float = 5.0,
) -> tuple[Optional[PassiveIdResult], Optional[dict]]:
    """Identify total inertia, Coulomb/viscous friction, and effective stiffness from a gentle
    motor-driven sinusoidal chirp with the relaxed human arm strapped in.

    Regression model:
        tau_m(t) = J_total * qdd + tau_c_total * tanh(qd / v_band) + b_total * qd + k_eff * q + tau_0

    Because frequency sweeps across a wide band (e.g. 0.5 -> 6 Hz), qdd = -w^2 * q is cleanly
    separated from k_eff * q (gravity gradient + passive elbow stiffness), and tau_0 absorbs any
    static gravity offset at the sweep center. Works regardless of rig orientation or hard stops.
    """
    trials: dict[int, list[dict]] = {}
    for r in records:
        if r["fresh"] == 1 and r["phase"].startswith("relaxed_chirp"):
            trials.setdefault(r["trial"], []).append(r)

    if not trials:
        return None, None

    chirp_win = min(window_s, 0.04)  # preserve up to ~6-8 Hz without Savitzky-Golay attenuation
    v_band = math.radians(max(v_band_deg_s, 1e-3))
    min_v = math.radians(min_speed_deg_s)

    per_trial: list[DecayTrialFit] = []
    X_all, y_all = [], []
    plot_trials = []

    for tr_id in sorted(trials):
        seg = trials[tr_id]
        if len(seg) < 100:
            continue
        t = np.array([r["t_sample"] for r in seg], dtype=float)
        ang = np.array([r["angle_deg"] for r in seg], dtype=float)
        vel = np.array([r["velocity_deg_s"] for r in seg], dtype=float)
        tau_m_raw = np.array([r["torque_nm"] for r in seg], dtype=float)

        t_u, q, qd, qdd, _ = smooth_kinematics(t, ang, vel, window_s=chirp_win)
        tau_m = np.interp(t_u, t, tau_m_raw)

        # Trim fade-in / fade-out tails where excitation is tiny
        t_rel = t_u - t_u[0]
        dur = float(t_rel[-1])
        if dur > 2.5 * trim_edge_s:
            active = (t_rel >= trim_edge_s) & (t_rel <= dur - trim_edge_s)
        else:
            active = np.ones_like(q, dtype=bool)

        # Keep samples with meaningful motion and well clear of hard stops
        active &= (np.abs(qd) >= min_v) | (np.abs(qdd) >= math.radians(20.0))
        if stop_limit_deg is not None and math.isfinite(stop_limit_deg):
            active &= np.abs(np.degrees(q)) < (stop_limit_deg - stop_margin_deg)

        if int(np.sum(active)) < 50:
            continue

        q_c = q - float(np.mean(q[active]))
        X = np.column_stack([
            qdd[active],
            np.tanh(qd[active] / v_band),
            qd[active],
            q_c[active],
            np.ones(int(np.sum(active)), dtype=float),
        ])
        y = tau_m[active]

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        resid = y - y_hat
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
        rms = float(np.sqrt(np.mean(resid ** 2)))
        cond = float(np.linalg.cond(X))

        j_tot, tc_tot, b_tot, k_eff, tau_0 = [float(c) for c in coef]
        per_trial.append(DecayTrialFit(
            trial=tr_id,
            duration_s=dur,
            n_samples=int(X.shape[0]),
            angle_span_deg=float(np.max(np.degrees(q)) - np.min(np.degrees(q))),
            peak_speed_deg_s=float(np.max(np.abs(np.degrees(qd)))),
            j_total_kg_m2=j_tot,
            coulomb_total_nm=tc_tot,
            viscous_total_nm_s_rad=b_tot,
            r2=r2,
            rms_residual_nm=rms,
            cond_number=cond,
        ))
        X_all.append(X)
        y_all.append(y)

        y_fit_full = (
            j_tot * qdd
            + tc_tot * np.tanh(qd / v_band)
            + b_tot * qd
            + k_eff * q_c
            + tau_0
        )
        plot_trials.append({
            "trial": tr_id,
            "t": t_rel,
            "q_deg": np.degrees(q),
            "qd_deg_s": np.degrees(qd),
            "active": active,
            "y_meas": tau_m,
            "y_fit_trial": y_fit_full,
        })

    if not X_all:
        return None, None

    Xp = np.vstack(X_all)
    yp = np.concatenate(y_all)
    coef, *_ = np.linalg.lstsq(Xp, yp, rcond=None)
    resid = yp - Xp @ coef
    dof = max(1, Xp.shape[0] - Xp.shape[1])
    s2 = float(np.sum(resid ** 2)) / dof
    cov = s2 * np.linalg.pinv(Xp.T @ Xp)
    se = np.sqrt(np.maximum(0.0, np.diag(cov)))

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((yp - np.mean(yp)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
    rms = float(np.sqrt(np.mean(resid ** 2)))
    cond = float(np.linalg.cond(Xp))

    j_tot, tc_tot, b_tot, k_eff, tau_0 = [float(c) for c in coef]
    j_human = j_tot - exo_baseline.inertia_kg_m2
    tc_cuff = tc_tot - exo_baseline.coulomb_nm
    b_cuff = b_tot - exo_baseline.viscous_nm_s_rad

    warnings = _build_passive_warnings(r2, j_human, tc_cuff, b_cuff, cond, method="relaxed_chirp")

    result = PassiveIdResult(
        j_total_kg_m2=j_tot,
        j_total_se_kg_m2=float(se[0]),
        coulomb_total_nm=tc_tot,
        coulomb_total_se_nm=float(se[1]),
        viscous_total_nm_s_rad=b_tot,
        viscous_total_se_nm_s_rad=float(se[2]),
        j_human_kg_m2=j_human,
        coulomb_cuff_nm=tc_cuff,
        viscous_cuff_nm_s_rad=b_cuff,
        r2=r2,
        rms_residual_nm=rms,
        cond_number=cond,
        n_trials=len(per_trial),
        n_samples=int(Xp.shape[0]),
        method="relaxed_chirp",
        k_eff_nm_rad=k_eff,
        tau_offset_nm=tau_0,
        per_trial=per_trial,
        warnings=warnings,
    )
    return result, {"method": "relaxed_chirp", "trials": plot_trials, "pooled_coef": coef}




def analyze_multi_amp_sweep(
    records: list[dict],
    exo_baseline: ExoBaseline,
    g_total: Optional[GravityFit] = None,
    window_s: float = 0.04,
    v_band_deg_s: float = DEFAULT_V_BAND_DEG_S,
    min_speed_deg_s: float = 2.0,
    trim_edge_s: float = 0.8,
    stop_limit_deg: Optional[float] = None,
    stop_margin_deg: float = 5.0,
) -> tuple[Optional[PassiveIdResult], Optional[dict]]:
    """Identify inertia, Coulomb/viscous friction, and passive elbow stiffness from a multi-frequency
    amplitude-ramp test (each frequency step logged as phase="multi_amp_sweep", trial=1..N)."""
    trials: dict[int, list[dict]] = {}
    for r in records:
        if r["fresh"] == 1 and r["phase"].startswith("multi_amp_sweep"):
            trials.setdefault(r["trial"], []).append(r)
    if not trials:
        return None, None

    sweep_win = min(window_s, 0.04)
    v_band = math.radians(max(v_band_deg_s, 1e-3))
    min_v = math.radians(min_speed_deg_s)

    seg_cache = []
    X_all, y_all = [], []
    for tr_id in sorted(trials):
        seg = trials[tr_id]
        if len(seg) < 100:
            continue
        t = np.array([r["t_sample"] for r in seg], dtype=float)
        ang = np.array([r["angle_deg"] for r in seg], dtype=float)
        vel = np.array([r["velocity_deg_s"] for r in seg], dtype=float)
        tau_m_raw = np.array([r["torque_nm"] for r in seg], dtype=float)

        t_u, q, qd, qdd, _ = smooth_kinematics(t, ang, vel, window_s=sweep_win)
        tau_m = np.interp(t_u, t, tau_m_raw)
        t_rel = t_u - t_u[0]
        dur = float(t_rel[-1])

        active = (t_rel >= trim_edge_s) & (t_rel <= dur - trim_edge_s) if dur > 2.5 * trim_edge_s else np.ones_like(q, dtype=bool)
        active &= (np.abs(qd) >= min_v) | (np.abs(qdd) >= math.radians(15.0))
        if stop_limit_deg is not None and math.isfinite(stop_limit_deg):
            active &= np.abs(np.degrees(q)) < (stop_limit_deg - stop_margin_deg)
        if int(np.sum(active)) < 40:
            continue

        y_full = tau_m - g_total.torque(q) if g_total is not None else tau_m
        X_full = np.column_stack([qdd, np.tanh(qd / v_band), qd, q, np.ones_like(q)])
        X_all.append(X_full[active])
        y_all.append(y_full[active])
        seg_cache.append((tr_id, t_rel, dur, q, qd, qdd, y_full, active))

    if not X_all:
        return None, None

    Xp = np.vstack(X_all)
    yp = np.concatenate(y_all)
    coef, *_ = np.linalg.lstsq(Xp, yp, rcond=None)
    resid = yp - Xp @ coef
    dof = max(1, Xp.shape[0] - Xp.shape[1])
    s2 = float(np.sum(resid ** 2)) / dof
    se = np.sqrt(np.maximum(0.0, np.diag(s2 * np.linalg.pinv(Xp.T @ Xp))))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((yp - np.mean(yp)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
    rms = float(np.sqrt(np.mean(resid ** 2)))
    cond = float(np.linalg.cond(Xp))

    j_tot, tc_tot, b_tot, k_eff, tau_0 = [float(c) for c in coef]
    j_human = j_tot - exo_baseline.inertia_kg_m2
    tc_cuff = tc_tot - exo_baseline.coulomb_nm
    b_cuff = b_tot - exo_baseline.viscous_nm_s_rad

    per_trial: list[DecayTrialFit] = []
    plot_t, plot_q, plot_qd, plot_act, plot_ym, plot_yf = [], [], [], [], [], []
    t_offset = 0.0
    for tr_id, t_rel, dur, q, qd, qdd, y_full, active in seg_cache:
        q_c = q[active] - np.mean(q[active])
        f_est = float(np.sqrt(np.mean(qd[active] ** 2)) / max(np.sqrt(np.mean(q_c ** 2)), 1e-6) / (2.0 * math.pi))
        Xb = np.column_stack([qdd[active], np.tanh(qd[active] / v_band), qd[active], np.ones(int(np.sum(active)))])
        yb = y_full[active] - k_eff * q[active]
        cb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        rb = yb - Xb @ cb
        sst = float(np.sum((yb - np.mean(yb)) ** 2))
        r2b = 1.0 - float(np.sum(rb ** 2)) / sst if sst > 1e-12 else math.nan
        per_trial.append(DecayTrialFit(
            trial=tr_id, duration_s=dur, n_samples=int(np.sum(active)),
            angle_span_deg=float(np.max(np.degrees(q[active])) - np.min(np.degrees(q[active]))),
            peak_speed_deg_s=float(np.max(np.abs(np.degrees(qd[active])))),
            j_total_kg_m2=float(cb[0]), coulomb_total_nm=float(cb[1]), viscous_total_nm_s_rad=float(cb[2]),
            r2=r2b, rms_residual_nm=float(np.sqrt(np.mean(rb ** 2))), cond_number=float(np.linalg.cond(Xb)),
            natural_freq_hz=f_est,
        ))
        y_fit_seg = j_tot * qdd + tc_tot * np.tanh(qd / v_band) + b_tot * qd + k_eff * q + tau_0
        plot_t.append(t_rel + t_offset); plot_q.append(np.degrees(q)); plot_qd.append(np.degrees(qd))
        plot_act.append(active); plot_ym.append(y_full); plot_yf.append(y_fit_seg)
        t_offset += dur + 0.2

    warnings = _build_passive_warnings(r2, j_human, tc_cuff, b_cuff, cond, method="relaxed_chirp")
    result = PassiveIdResult(
        j_total_kg_m2=j_tot, j_total_se_kg_m2=float(se[0]),
        coulomb_total_nm=tc_tot, coulomb_total_se_nm=float(se[1]),
        viscous_total_nm_s_rad=b_tot, viscous_total_se_nm_s_rad=float(se[2]),
        j_human_kg_m2=j_human, coulomb_cuff_nm=tc_cuff, viscous_cuff_nm_s_rad=b_cuff,
        r2=r2, rms_residual_nm=rms, cond_number=cond, n_trials=len(per_trial), n_samples=int(Xp.shape[0]),
        method="multi_amp_sweep", k_eff_nm_rad=k_eff, tau_offset_nm=tau_0,
        per_trial=per_trial, warnings=warnings,
    )
    plot_trials = [{
        "trial": 1, "t": np.concatenate(plot_t), "q_deg": np.concatenate(plot_q),
        "qd_deg_s": np.concatenate(plot_qd), "active": np.concatenate(plot_act),
        "y_meas": np.concatenate(plot_ym), "y_fit_trial": np.concatenate(plot_yf),
    }]
    return result, {"method": "multi_amp_sweep", "trials": plot_trials, "pooled_coef": coef}

def analyze_amp_sweep(
    records: list[dict],
    exo_baseline: ExoBaseline,
    g_total: Optional[GravityFit] = None,
    window_s: float = 0.04,
    v_band_deg_s: float = DEFAULT_V_BAND_DEG_S,
    min_speed_deg_s: float = 2.0,
    trim_edge_s: float = 1.5,
    stop_limit_deg: Optional[float] = None,
    stop_margin_deg: float = 5.0,
    n_bins: int = 3,
) -> tuple[Optional[PassiveIdResult], Optional[dict]]:
    """Identify inertia and friction from a fixed-frequency amplitude-ramp perturbation."""
    seg = [r for r in records if r["fresh"] == 1 and r["phase"].startswith("amp_sweep")]
    if len(seg) < 100:
        return None, None

    sweep_win = min(window_s, 0.04)
    v_band = math.radians(max(v_band_deg_s, 1e-3))
    min_v = math.radians(min_speed_deg_s)

    t = np.array([r["t_sample"] for r in seg], dtype=float)
    ang = np.array([r["angle_deg"] for r in seg], dtype=float)
    vel = np.array([r["velocity_deg_s"] for r in seg], dtype=float)
    tau_m_raw = np.array([r["torque_nm"] for r in seg], dtype=float)

    t_u, q, qd, qdd, _ = smooth_kinematics(t, ang, vel, window_s=sweep_win)
    tau_m = np.interp(t_u, t, tau_m_raw)
    t_rel = t_u - t_u[0]
    dur = float(t_rel[-1])

    active = (t_rel >= trim_edge_s) & (t_rel <= dur - trim_edge_s) if dur > 2.5 * trim_edge_s else np.ones_like(q, dtype=bool)
    active &= (np.abs(qd) >= min_v) | (np.abs(qdd) >= math.radians(20.0))
    if stop_limit_deg is not None and math.isfinite(stop_limit_deg):
        active &= np.abs(np.degrees(q)) < (stop_limit_deg - stop_margin_deg)
    if int(np.sum(active)) < 50:
        return None, None

    y_full = tau_m - g_total.torque(q) if g_total is not None else tau_m
    X_full = np.column_stack([qdd, np.tanh(qd / v_band), qd, np.ones_like(q)])
    X, y = X_full[active], y_full[active]

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(1, X.shape[0] - X.shape[1])
    s2 = float(np.sum(resid ** 2)) / dof
    se = np.sqrt(np.maximum(0.0, np.diag(s2 * np.linalg.pinv(X.T @ X))))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
    rms = float(np.sqrt(np.mean(resid ** 2)))
    cond = float(np.linalg.cond(X))

    j_tot, tc_tot, b_tot, tau_0 = [float(c) for c in coef]
    j_human = j_tot - exo_baseline.inertia_kg_m2
    tc_cuff = tc_tot - exo_baseline.coulomb_nm
    b_cuff = b_tot - exo_baseline.viscous_nm_s_rad

    per_bin: list[DecayTrialFit] = []
    idx_active = np.where(active)[0]
    for b_idx, sl in enumerate(np.array_split(idx_active, max(1, n_bins)), start=1):
        if len(sl) < 40:
            continue
        Xb, yb = X_full[sl], y_full[sl]
        cb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        rb = yb - Xb @ cb
        sst = float(np.sum((yb - np.mean(yb)) ** 2))
        r2b = 1.0 - float(np.sum(rb ** 2)) / sst if sst > 1e-12 else math.nan
        per_bin.append(DecayTrialFit(
            trial=b_idx, duration_s=float(t_rel[sl[-1]] - t_rel[sl[0]]), n_samples=int(len(sl)),
            angle_span_deg=float(np.max(np.degrees(q[sl])) - np.min(np.degrees(q[sl]))),
            peak_speed_deg_s=float(np.max(np.abs(np.degrees(qd[sl])))),
            j_total_kg_m2=float(cb[0]), coulomb_total_nm=float(cb[1]), viscous_total_nm_s_rad=float(cb[2]),
            r2=r2b, rms_residual_nm=float(np.sqrt(np.mean(rb ** 2))), cond_number=float(np.linalg.cond(Xb)),
        ))

    warnings = _build_passive_warnings(r2, j_human, tc_cuff, b_cuff, cond, method="relaxed_chirp")
    if g_total is None:
        warnings.append("Phase A gravity was skipped during single-frequency amp_sweep: run with Phase A enabled for unbiased J_human.")

    result = PassiveIdResult(
        j_total_kg_m2=j_tot, j_total_se_kg_m2=float(se[0]),
        coulomb_total_nm=tc_tot, coulomb_total_se_nm=float(se[1]),
        viscous_total_nm_s_rad=b_tot, viscous_total_se_nm_s_rad=float(se[2]),
        j_human_kg_m2=j_human, coulomb_cuff_nm=tc_cuff, viscous_cuff_nm_s_rad=b_cuff,
        r2=r2, rms_residual_nm=rms, cond_number=cond, n_trials=len(per_bin), n_samples=int(X.shape[0]),
        method="amp_sweep", tau_offset_nm=tau_0, per_trial=per_bin, warnings=warnings,
    )
    plot_trials = [{"trial": 1, "t": t_rel, "q_deg": np.degrees(q), "qd_deg_s": np.degrees(qd),
                    "active": active, "y_meas": y_full, "y_fit_trial": X_full @ coef}]
    return result, {"method": "amp_sweep", "trials": plot_trials, "pooled_coef": coef}

# ----------------------------------------------------------------------------------------------
# Phase C (Legacy): Relaxed Free-Decay Swings (with Hard-Stop Masking)
# ----------------------------------------------------------------------------------------------
def _fit_decay_segment(
    t: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    tau_m: np.ndarray,
    g_total: GravityFit,
    v_band_deg_s: float = DEFAULT_V_BAND_DEG_S,
    min_speed_deg_s: float = 3.0,
    stop_limit_deg: Optional[float] = None,
    stop_margin_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray, int]:
    """Fit J_total * qdd + tau_c_total * tanh(qd/v_band) + b_total * qd = tau_m - g_total(q).

    Masks out near-stationary tail samples and samples near mechanical hard stops so stop impacts
    do not bias the inertia/friction regression.
    """
    v_band = math.radians(max(v_band_deg_s, 1e-3))
    min_v = math.radians(min_speed_deg_s)
    active = (np.abs(qd) >= min_v) | (np.abs(qdd) >= math.radians(25.0))

    stop_masked = 0
    if stop_limit_deg is not None and math.isfinite(stop_limit_deg):
        clear_of_stop = np.abs(np.degrees(q)) < (stop_limit_deg - stop_margin_deg)
        stop_masked = int(np.sum(active & ~clear_of_stop))
        active = active & clear_of_stop

    if int(np.sum(active)) < 30:
        active = np.ones_like(q, dtype=bool)
        if stop_limit_deg is not None and math.isfinite(stop_limit_deg):
            clear_of_stop = np.abs(np.degrees(q)) < (stop_limit_deg - stop_margin_deg)
            if int(np.sum(clear_of_stop)) >= 20:
                active = clear_of_stop

    X = np.column_stack([qdd[active], np.tanh(qd[active] / v_band), qd[active]])
    y = tau_m[active] - g_total.torque(q[active])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
    rms = float(np.sqrt(np.mean(resid ** 2)))
    cond = float(np.linalg.cond(X))
    return coef, X, y, r2, rms, cond, active, stop_masked


def analyze_free_decay(
    records: list[dict],
    exo_baseline: ExoBaseline,
    g_total: Optional[GravityFit],
    window_s: float = 0.06,
    v_band_deg_s: float = DEFAULT_V_BAND_DEG_S,
    min_speed_deg_s: float = 3.0,
    stop_limit_deg: Optional[float] = None,
    stop_margin_deg: float = 5.0,
) -> tuple[Optional[PassiveIdResult], Optional[dict]]:
    """Identify total inertia and friction from relaxed free-decay trials, then subtract exo baseline."""
    if g_total is None or g_total.mgl_nm < 0.05:
        return None, None

    trials: dict[int, list[dict]] = {}
    for r in records:
        if r["fresh"] == 1 and r["phase"].startswith("free_decay"):
            trials.setdefault(r["trial"], []).append(r)

    if not trials:
        return None, None

    per_trial: list[DecayTrialFit] = []
    X_all, y_all = [], []
    plot_trials = []
    total_stop_masked = 0

    for tr_id in sorted(trials):
        seg = trials[tr_id]
        if len(seg) < 50:
            continue
        t = np.array([r["t_sample"] for r in seg], dtype=float)
        ang = np.array([r["angle_deg"] for r in seg], dtype=float)
        vel = np.array([r["velocity_deg_s"] for r in seg], dtype=float)
        tau_m_raw = np.array([r["torque_nm"] for r in seg], dtype=float)

        t_u, q, qd, qdd, _ = smooth_kinematics(t, ang, vel, window_s=window_s)
        tau_m = np.interp(t_u, t, tau_m_raw)

        coef, X, y, r2, rms, cond, active, stop_masked = _fit_decay_segment(
            t_u, q, qd, qdd, tau_m, g_total,
            v_band_deg_s=v_band_deg_s, min_speed_deg_s=min_speed_deg_s,
            stop_limit_deg=stop_limit_deg, stop_margin_deg=stop_margin_deg,
        )
        total_stop_masked += stop_masked
        j_tot, tc_tot, b_tot = float(coef[0]), float(coef[1]), float(coef[2])
        wn = math.sqrt(g_total.mgl_nm / j_tot) if j_tot > 0 else math.nan
        fn = wn / (2.0 * math.pi) if math.isfinite(wn) else math.nan
        zeta = b_tot / (2.0 * math.sqrt(j_tot * g_total.mgl_nm)) if j_tot > 0 else math.nan

        per_trial.append(DecayTrialFit(
            trial=tr_id,
            duration_s=float(t_u[-1] - t_u[0]),
            n_samples=int(X.shape[0]),
            angle_span_deg=float(np.max(np.degrees(q)) - np.min(np.degrees(q))),
            peak_speed_deg_s=float(np.max(np.abs(np.degrees(qd)))),
            j_total_kg_m2=j_tot,
            coulomb_total_nm=tc_tot,
            viscous_total_nm_s_rad=b_tot,
            r2=r2,
            rms_residual_nm=rms,
            cond_number=cond,
            natural_freq_hz=fn,
            damping_ratio=zeta,
        ))
        X_all.append(X)
        y_all.append(y)
        plot_trials.append({
            "trial": tr_id,
            "t": t_u - t_u[0],
            "q_deg": np.degrees(q),
            "qd_deg_s": np.degrees(qd),
            "active": active,
            "y_meas": tau_m - g_total.torque(q),
            "y_fit_trial": (
                j_tot * qdd
                + tc_tot * np.tanh(qd / math.radians(max(v_band_deg_s, 1e-3)))
                + b_tot * qd
            ),
        })

    if not X_all:
        return None, None

    Xp = np.vstack(X_all)
    yp = np.concatenate(y_all)
    coef, *_ = np.linalg.lstsq(Xp, yp, rcond=None)
    resid = yp - Xp @ coef
    dof = max(1, Xp.shape[0] - Xp.shape[1])
    s2 = float(np.sum(resid ** 2)) / dof
    cov = s2 * np.linalg.pinv(Xp.T @ Xp)
    se = np.sqrt(np.maximum(0.0, np.diag(cov)))

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((yp - np.mean(yp)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
    rms = float(np.sqrt(np.mean(resid ** 2)))
    cond = float(np.linalg.cond(Xp))

    j_tot, tc_tot, b_tot = float(coef[0]), float(coef[1]), float(coef[2])
    j_human = j_tot - exo_baseline.inertia_kg_m2
    tc_cuff = tc_tot - exo_baseline.coulomb_nm
    b_cuff = b_tot - exo_baseline.viscous_nm_s_rad

    warnings = _build_passive_warnings(r2, j_human, tc_cuff, b_cuff, cond, method="free_decay")
    if total_stop_masked > 0:
        warnings.append(
            f"Masked {total_stop_masked} samples within {stop_margin_deg:.1f} deg of hard stop; "
            "use --phase-c-mode chirp on hard-stop-limited rigs for clean inertia ID."
        )
    # Check whether vertical hanging equilibrium lies outside the joint stroke
    q_eq_deg = ((g_total.horizontal_angle_deg - 90.0 + 180.0) % 360.0) - 180.0
    if stop_limit_deg is not None and abs(q_eq_deg) > (stop_limit_deg - stop_margin_deg):
        warnings.append(
            f"Gravity equilibrium ({q_eq_deg:+.1f} deg) lies outside the +/-{stop_limit_deg:.1f} deg stops: "
            "free-decay oscillation is mechanically blocked; re-run with --phase-c-mode chirp."
        )

    result = PassiveIdResult(
        j_total_kg_m2=j_tot,
        j_total_se_kg_m2=float(se[0]),
        coulomb_total_nm=tc_tot,
        coulomb_total_se_nm=float(se[1]),
        viscous_total_nm_s_rad=b_tot,
        viscous_total_se_nm_s_rad=float(se[2]),
        j_human_kg_m2=j_human,
        coulomb_cuff_nm=tc_cuff,
        viscous_cuff_nm_s_rad=b_cuff,
        r2=r2,
        rms_residual_nm=rms,
        cond_number=cond,
        n_trials=len(per_trial),
        n_samples=int(Xp.shape[0]),
        method="free_decay",
        per_trial=per_trial,
        warnings=warnings,
    )
    return result, {"method": "free_decay", "trials": plot_trials, "pooled_coef": coef}


# ----------------------------------------------------------------------------------------------
# Optional Loadcell Cross-Validation
# ----------------------------------------------------------------------------------------------
def analyze_loadcell_validation(
    active_series: Optional[dict],
    records: list[dict],
    loadcell_csv: Optional[Path] = None,
    v_band_deg_s: float = DEFAULT_V_BAND_DEG_S,
) -> Optional[LoadcellValidation]:
    """Cross-validate exo-predicted interaction torque tau_h_hat against measured cuff loadcell torque."""
    if active_series is None:
        return None

    t_u = active_series["t"]
    tau_h_hat = active_series["tau_h_hat"]
    qdd = np.radians(active_series["qdd_deg_s2"])
    qd = np.radians(active_series["qd_deg_s"])

    lc_t, lc_tau = [], []
    if loadcell_csv is not None and Path(loadcell_csv).exists():
        with open(loadcell_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    tv = float(row.get("t_sample") or row.get("t") or row.get("t_s"))
                    lv = float(row.get("loadcell_nm") or row.get("torque_nm"))
                    if math.isfinite(tv) and math.isfinite(lv):
                        lc_t.append(tv)
                        lc_tau.append(lv)
                except (TypeError, ValueError):
                    continue
    else:
        for r in records:
            if r["fresh"] == 1 and r["phase"].startswith("active_swing") and math.isfinite(r["loadcell_nm"]):
                lc_t.append(r["t_sample"])
                lc_tau.append(r["loadcell_nm"])

    if len(lc_t) < 30:
        return None

    lc_t_arr = np.asarray(lc_t, dtype=float)
    lc_tau_arr = np.asarray(lc_tau, dtype=float)
    order = np.argsort(lc_t_arr)
    lc_t_arr, lc_tau_arr = lc_t_arr[order], lc_tau_arr[order]
    tau_lc = np.interp(t_u, lc_t_arr, lc_tau_arr)

    diff = tau_lc - tau_h_hat
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    r_mat = np.corrcoef(tau_lc, tau_h_hat)
    pearson_r = float(r_mat[0, 1]) if np.isfinite(r_mat[0, 1]) else math.nan

    v_band = math.radians(max(v_band_deg_s, 1e-3))
    X = np.column_stack([qdd, np.tanh(qd / v_band), qd])
    coef, *_ = np.linalg.lstsq(X, diff, rcond=None)
    ss_res = float(np.sum((diff - X @ coef) ** 2))
    ss_tot = float(np.sum((diff - np.mean(diff)) ** 2))
    res_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan

    return LoadcellValidation(
        n_samples=len(t_u),
        rmse_nm=rmse,
        mae_nm=mae,
        pearson_r=pearson_r,
        residual_j_kg_m2=float(coef[0]),
        residual_coulomb_nm=float(coef[1]),
        residual_viscous_nm_s_rad=float(coef[2]),
        residual_r2=res_r2,
    )


# ----------------------------------------------------------------------------------------------
# Full Pipeline & Human-Readable Summary
# ----------------------------------------------------------------------------------------------
def run_full_analysis(
    csv_path: Path,
    exo_baseline: ExoBaseline,
    window_s: float = 0.06,
    v_band_deg_s: float = DEFAULT_V_BAND_DEG_S,
    min_speed_deg_s: float = 3.0,
    arm_mgl_nm: Optional[float] = None,
    arm_horizontal_deg: float = 0.0,
    loadcell_csv: Optional[Path] = None,
    stop_limit_deg: Optional[float] = None,
    stop_margin_deg: float = 5.0,
) -> tuple[PassiveAnalysisReport, dict]:
    """Run complete Experiment 2 analysis on a passive_exo CSV file."""
    records = load_passive_csv(Path(csv_path))
    if not records:
        raise ValueError(f"No valid data rows found in {csv_path}")

    t_all = np.array([r["t_sample"] for r in records if r["fresh"] == 1], dtype=float)
    dt_med = float(np.median(np.diff(np.sort(t_all)))) if len(t_all) > 2 else math.nan
    rate_hz = 1.0 / dt_med if math.isfinite(dt_med) and dt_med > 0 else math.nan
    fresh_frac = sum(1 for r in records if r["fresh"] == 1) / len(records)

    notes: list[str] = []
    if exo_baseline.inertia_kg_m2 == 0.0:
        notes.append(
            "Exo baseline inertia J_exo is 0.0 kg m^2 (no Exp 1 _frf.csv or --exo-inertia supplied). "
            "Reported J_human equals J_total until J_exo is provided."
        )

    # Phase A: Static gravity calibration (optional for relaxed_chirp, required for free_decay)
    g_total, g_arm = fit_gravity_from_holds(records, exo_baseline)
    if g_total is None and arm_mgl_nm is not None:
        g_total, g_arm = gravity_from_anthropometrics(arm_mgl_nm, arm_horizontal_deg, exo_baseline)
        notes.append(
            f"Using user-supplied anthropometric forearm gravity m*g*l = {arm_mgl_nm:.3f} Nm "
            f"(horizontal at {arm_horizontal_deg:+.1f} deg)."
        )

    # Phase B: Active human swings (transparency)
    transparency, active_series = analyze_active_swing(
        records, exo_baseline, window_s=window_s, v_band_deg_s=v_band_deg_s
    )

    # Phase C: Auto-detect relaxed_chirp (preferred) vs legacy free_decay
    has_multi = any(r["fresh"] == 1 and r["phase"].startswith("multi_amp_sweep") for r in records)
    has_amp = any(r["fresh"] == 1 and r["phase"].startswith("amp_sweep") for r in records)
    has_chirp = any(r["fresh"] == 1 and r["phase"].startswith("relaxed_chirp") for r in records)
    has_decay = any(r["fresh"] == 1 and r["phase"].startswith("free_decay") for r in records)

    passive_id: Optional[PassiveIdResult] = None
    phase_c_series: Optional[dict] = None
    if has_multi:
        passive_id, phase_c_series = analyze_multi_amp_sweep(
            records, exo_baseline, g_total=g_total, window_s=min(window_s, 0.04),
            v_band_deg_s=v_band_deg_s, min_speed_deg_s=min(min_speed_deg_s, 2.0),
            stop_limit_deg=stop_limit_deg, stop_margin_deg=stop_margin_deg,
        )
    elif has_amp:
        passive_id, phase_c_series = analyze_amp_sweep(
            records, exo_baseline, g_total=g_total, window_s=min(window_s, 0.04),
            v_band_deg_s=v_band_deg_s, min_speed_deg_s=min(min_speed_deg_s, 2.0),
            stop_limit_deg=stop_limit_deg, stop_margin_deg=stop_margin_deg,
        )
    elif has_chirp:
        passive_id, phase_c_series = analyze_relaxed_chirp(
            records, exo_baseline,
            window_s=min(window_s, 0.04),
            v_band_deg_s=v_band_deg_s,
            min_speed_deg_s=min(min_speed_deg_s, 2.0),
            stop_limit_deg=stop_limit_deg,
            stop_margin_deg=stop_margin_deg,
        )
    elif has_decay:
        if g_total is None:
            notes.append(
                "No Phase A static gravity holds found and no --arm-mgl supplied: "
                "free-decay inertia/friction identification requires a known gravity scale."
            )
        else:
            passive_id, phase_c_series = analyze_free_decay(
                records, exo_baseline, g_total,
                window_s=window_s, v_band_deg_s=v_band_deg_s,
                min_speed_deg_s=min_speed_deg_s,
                stop_limit_deg=stop_limit_deg,
                stop_margin_deg=stop_margin_deg,
            )

    # Optional loadcell cross-validation
    loadcell = analyze_loadcell_validation(
        active_series, records, loadcell_csv=loadcell_csv, v_band_deg_s=v_band_deg_s
    )

    report = PassiveAnalysisReport(
        csv_path=str(csv_path),
        sample_rate_hz=rate_hz,
        fresh_fraction=fresh_frac,
        exo_baseline=exo_baseline,
        gravity_total=g_total,
        gravity_arm=g_arm,
        transparency=transparency,
        passive_id=passive_id,
        loadcell=loadcell,
        notes=notes,
    )
    artifacts = {
        "records": records,
        "active_series": active_series,
        "decay_series": phase_c_series,
    }
    return report, artifacts


def format_summary(report: PassiveAnalysisReport) -> str:
    """Format a clean, readable text summary matching the style of exo/chirp.py."""
    b = report.exo_baseline
    lines = [
        f"Passive Exo Analysis: {Path(report.csv_path).name}",
        f"  sampling : {report.sample_rate_hz:.1f} Hz effective, {100.0 * report.fresh_fraction:.2f}% fresh replies",
        f"  exo base : J_exo = {b.inertia_kg_m2 * 1e3:.3f} e-3 kg m^2, "
        f"coulomb = {b.coulomb_nm:.4f} Nm, viscous = {b.viscous_nm_s_rad:.4f} Nm s/rad",
    ]

    if report.gravity_total is not None and report.gravity_arm is not None:
        gt, ga = report.gravity_total, report.gravity_arm
        stict_note = (
            f", stiction half-width {gt.stiction_halfwidth_nm:.3f} Nm"
            if math.isfinite(gt.stiction_halfwidth_nm)
            else ""
        )
        lines += [
            "",
            f"  [Phase A] Static Gravity ({gt.source}, {gt.n_angles} hold angles{stict_note}):",
            f"    total (exo + arm) : m*g*l = {gt.mgl_nm:.3f} Nm (horizontal at {gt.horizontal_angle_deg:+.1f} deg)",
            f"    human forearm     : m*g*l = {ga.mgl_nm:.3f} Nm (horizontal at {ga.horizontal_angle_deg:+.1f} deg)",
        ]

    if report.transparency is not None:
        tr = report.transparency
        lines += [
            "",
            f"  [Phase B] Active Zero-Torque Swing Transparency ({tr.duration_s:.1f} s, {tr.n_samples} samples):",
            f"    kinematics        : angle [{tr.angle_min_deg:+.1f}, {tr.angle_max_deg:+.1f}] deg, "
            f"peak |v| = {tr.peak_speed_deg_s:.1f} deg/s, peak |a| = {tr.peak_accel_deg_s2:.0f} deg/s^2",
            f"    motor 0-Nm ripple : RMS = {tr.rms_motor_torque_nm:.4f} Nm, peak = {tr.peak_motor_torque_nm:.4f} Nm",
            f"    interaction tau_h : RMS = {tr.rms_interaction_torque_nm:.4f} Nm, "
            f"peak = {tr.peak_interaction_torque_nm:.4f} Nm",
            f"      breakdown (RMS) : inertia {tr.rms_inertial_torque_nm:.4f} Nm | "
            f"friction {tr.rms_friction_torque_nm:.4f} Nm | gravity {tr.rms_gravity_torque_nm:.4f} Nm",
            f"    mechanical work   : {tr.total_work_j:.3f} J ({tr.mean_power_w:.3f} W mean)",
        ]
        if tr.speed_bins:
            lines.append("    speed-binned resistance:")
            for sb in tr.speed_bins:
                lines.append(
                    f"      {sb['bin']:<22} RMS tau_h = {sb['rms_interaction_nm']:.4f} Nm, "
                    f"peak = {sb['peak_interaction_nm']:.4f} Nm (n={sb['n_samples']})"
                )

    if report.passive_id is not None:
        pid = report.passive_id
        title = (
            "Multi-Frequency Amplitude-Sweep System ID (fn Hz = step freq)"
            if pid.method == "multi_amp_sweep"
            else "Fixed-Frequency Amplitude-Ramp System ID (bins 1..3 = small -> large amp)"
            if pid.method == "amp_sweep"
            else ("Relaxed Motor-Driven Chirp System ID" if pid.method == "relaxed_chirp" else "Passive Free-Decay System ID")
        )
        lines += [
            "",
            f"  [Phase C] {title} ({pid.n_trials} trial(s), {pid.n_samples} samples):",
            f"    J_total           : {pid.j_total_kg_m2 * 1e3:.3f} +/- {pid.j_total_se_kg_m2 * 1e3:.3f} e-3 kg m^2",
            f"    J_human (forearm) : {pid.j_human_kg_m2 * 1e3:.3f} e-3 kg m^2  (J_total - J_exo)",
            f"    coulomb_total     : {pid.coulomb_total_nm:.4f} +/- {pid.coulomb_total_se_nm:.4f} Nm",
            f"    coulomb_cuff      : {pid.coulomb_cuff_nm:.4f} Nm  (skin-cuff + passive elbow)",
            f"    viscous_total     : {pid.viscous_total_nm_s_rad:.4f} +/- {pid.viscous_total_se_nm_s_rad:.4f} Nm s/rad",
            f"    viscous_cuff      : {pid.viscous_cuff_nm_s_rad:.4f} Nm s/rad  (skin-cuff + passive elbow)",
        ]
        if pid.method in ("relaxed_chirp", "multi_amp_sweep") and math.isfinite(pid.k_eff_nm_rad):
            lines.append(
                f"    passive stiffness : k_eff = {pid.k_eff_nm_rad:+.3f} Nm/rad, "
                f"static offset tau_0 = {pid.tau_offset_nm:+.3f} Nm"
            )
        lines.append(
            f"    fit quality       : R^2 = {pid.r2:.4f}, residual RMS = {pid.rms_residual_nm:.4f} Nm, "
            f"cond(X) = {pid.cond_number:.1f}"
        )
        if pid.per_trial and (pid.method != "relaxed_chirp" or len(pid.per_trial) > 1):
            lines.append("    per-trial breakdown:")
            lines.append(
                f"      {'trial':>5} {'dur s':>6} {'span':>6} {'J_tot e-3':>10} {'tc Nm':>8} {'b':>8} {'R^2':>7} {'fn Hz':>7}"
            )
            for pt in pid.per_trial:
                lines.append(
                    f"      {pt.trial:5d} {pt.duration_s:6.2f} {pt.angle_span_deg:6.1f} "
                    f"{pt.j_total_kg_m2 * 1e3:10.3f} {pt.coulomb_total_nm:8.4f} "
                    f"{pt.viscous_total_nm_s_rad:8.4f} {pt.r2:7.3f} {pt.natural_freq_hz:7.2f}"
                )
        for w in pid.warnings:
            lines.append(f"    WARNING: {w}")

    if report.loadcell is not None:
        lc = report.loadcell
        lines += [
            "",
            f"  [Loadcell Cross-Validation] ({lc.n_samples} samples):",
            f"    tau_lc vs tau_h   : RMSE = {lc.rmse_nm:.4f} Nm, MAE = {lc.mae_nm:.4f} Nm, r = {lc.pearson_r:.4f}",
            f"    residual fit      : dJ = {lc.residual_j_kg_m2 * 1e3:+.3f} e-3 kg m^2, "
            f"dtc = {lc.residual_coulomb_nm:+.4f} Nm, db = {lc.residual_viscous_nm_s_rad:+.4f} Nm s/rad "
            f"(R^2 = {lc.residual_r2:.3f})",
        ]

    if report.notes:
        lines.append("")
        for n in report.notes:
            lines.append(f"  NOTE: {n}")

    return "\n".join(lines)


def save_analysis_outputs(stem: Path, report: PassiveAnalysisReport, summary: str) -> None:
    """Save <stem>_analysis.json and <stem>_summary.txt."""
    Path(f"{stem}_analysis.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    Path(f"{stem}_summary.txt").write_text(summary + "\n")
