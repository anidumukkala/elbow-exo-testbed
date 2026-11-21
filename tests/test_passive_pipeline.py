"""End-to-end synthetic verification suite for the Experiment 2 Passive Exo Test Package.

Tests BOTH:
  1. Default hard-stop-safe pipeline (bidirectional Phase A with large 0.46 Nm stiction,
     Phase B zero-torque active swings + loadcell validation, and Phase C relaxed motor-driven
     chirp around 0 deg even when gravity equilibrium lies outside the +/-60 deg hard stops).
  2. Legacy free-decay pipeline (including hard-stop margin masking).
  3. Full CLI execution and PNG plot generation via analyze_passive_exo.py.

Run:
    pixi run test-passive
    python tests/test_passive_pipeline.py
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exo.chirp import Chirp  # noqa: E402
from exo.passive import (  # noqa: E402
    ExoBaseline,
    format_summary,
    run_full_analysis,
    save_passive_csv,
)


def simulate_chirp_session(
    rng: np.random.Generator,
    exo: ExoBaseline,
    j_human: float = 0.048,
    coulomb_cuff: float = 0.065,
    viscous_cuff: float = 0.032,
    arm_mgl: float = 2.60,
    arm_horiz_deg: float = 170.0,  # hanging eq at +80 deg -> OUTSIDE +/-60 deg hard stops!
    k_passive_elbow: float = 0.15,
    rate_hz: float = 500.0,
) -> tuple[list[tuple], list[tuple]]:
    """Simulate a 500 Hz hard-stop-limited session using bidirectional Phase A + Phase B + Phase C chirp."""
    dt = 1.0 / rate_hz
    v_band = math.radians(2.0)

    th_arm = math.radians(arm_horiz_deg)
    gc_arm = arm_mgl * math.cos(th_arm)
    gs_arm = arm_mgl * math.sin(th_arm)
    gc_tot = exo.gravity_cos_nm + gc_arm
    gs_tot = exo.gravity_sin_nm + gs_arm

    j_tot = exo.inertia_kg_m2 + j_human
    tc_tot = exo.coulomb_nm + coulomb_cuff
    b_tot = exo.viscous_nm_s_rad + viscous_cuff

    rows = []
    lc_rows = []
    t_now = 0.0

    # ---- Phase A: Bidirectional static holds (-25, 0, +25 deg from + and - with full stiction bias) ----
    angles = (-25.0, 0.0, 25.0)
    for direction, suffix in ((+1.0, "pos"), (-1.0, "neg")):
        seq = list(enumerate(angles, start=1)) if direction > 0 else list(reversed(list(enumerate(angles, start=1))))
        for idx, ang_deg in seq:
            n = int(1.5 * rate_hz)
            q = math.radians(ang_deg)
            g_val = gc_tot * math.cos(q) + gs_tot * math.sin(q)
            # Approaching in + direction adds +tc_tot stiction; - direction adds -tc_tot
            tau_hold = g_val + direction * tc_tot
            for _ in range(n):
                fresh = 0 if rng.random() < 0.005 else 1
                rows.append((
                    t_now, t_now + 0.0005, fresh, f"gravity_hold_{idx}_{suffix}", idx,
                    ang_deg + rng.normal(0.0, 0.02),
                    rng.normal(0.0, 0.1),
                    tau_hold + rng.normal(0.0, 0.015),
                    tau_hold, 35.0, 38.0,
                ))
                t_now += dt

    # ---- Phase B: 12 s active human swings (slow -> medium -> fast) ----
    n_act = int(12.0 * rate_hz)
    t_b = np.arange(n_act) * dt
    q_b = (
        math.radians(30.0) * np.sin(2.0 * math.pi * 0.45 * t_b)
        + math.radians(12.0) * np.sin(2.0 * math.pi * 1.15 * t_b + 0.4)
    )
    qd_b = (
        math.radians(30.0) * (2.0 * math.pi * 0.45) * np.cos(2.0 * math.pi * 0.45 * t_b)
        + math.radians(12.0) * (2.0 * math.pi * 1.15) * np.cos(2.0 * math.pi * 1.15 * t_b + 0.4)
    )
    qdd_b = (
        -math.radians(30.0) * (2.0 * math.pi * 0.45) ** 2 * np.sin(2.0 * math.pi * 0.45 * t_b)
        - math.radians(12.0) * (2.0 * math.pi * 1.15) ** 2 * np.sin(2.0 * math.pi * 1.15 * t_b + 0.4)
    )
    for k in range(n_act):
        fresh = 0 if rng.random() < 0.005 else 1
        tau_m = rng.normal(0.0, 0.015)
        tau_exo = (
            exo.inertia_kg_m2 * qdd_b[k]
            + exo.coulomb_nm * math.tanh(qd_b[k] / v_band)
            + exo.viscous_nm_s_rad * qd_b[k]
            + exo.gravity_cos_nm * math.cos(q_b[k])
            + exo.gravity_sin_nm * math.sin(q_b[k])
        )
        tau_lc = (tau_exo - tau_m) + rng.normal(0.0, 0.01)
        rows.append((
            t_now, t_now + 0.0005, fresh, "active_swing", 1,
            math.degrees(q_b[k]) + rng.normal(0.0, 0.015),
            math.degrees(qd_b[k]) + rng.normal(0.0, 0.35),
            tau_m, 0.0, 35.5, 38.5,
        ))
        lc_rows.append((t_now + 0.0005, tau_lc))
        t_now += dt

    # ---- Phase C: 30 s gentle motor-driven relaxed chirp around 0 deg (+/-8 deg, 0.5 -> 6.0 Hz) ----
    chirp = Chirp(0.5, 6.0, 30.0, 8.0, 90.0, 1200.0)
    n_c = int(30.0 * rate_hz)
    t_c = np.arange(n_c) * dt
    q_c = np.array([math.radians(chirp.position(float(ti))) for ti in t_c], dtype=float)
    qd_c = np.gradient(q_c, dt)
    qdd_c = np.gradient(qd_c, dt)
    for k in range(n_c):
        fresh = 0 if rng.random() < 0.005 else 1
        g_val = gc_tot * math.cos(q_c[k]) + gs_tot * math.sin(q_c[k])
        tau_m_true = (
            j_tot * qdd_c[k]
            + tc_tot * math.tanh(qd_c[k] / v_band)
            + b_tot * qd_c[k]
            + k_passive_elbow * q_c[k]
            + g_val
        )
        rows.append((
            t_now, t_now + 0.0005, fresh, "relaxed_chirp", 1,
            math.degrees(q_c[k]) + rng.normal(0.0, 0.012),
            math.degrees(qd_c[k]) + rng.normal(0.0, 0.30),
            tau_m_true + rng.normal(0.0, 0.015),
            tau_m_true, 36.0, 39.0,
        ))
        t_now += dt

    return rows, lc_rows


def simulate_legacy_decay_session(
    rng: np.random.Generator,
    exo: ExoBaseline,
    j_human: float = 0.048,
    coulomb_cuff: float = 0.045,
    viscous_cuff: float = 0.032,
    arm_mgl: float = 2.80,
    arm_horiz_deg: float = 85.0,  # hanging eq near -5 deg -> INSIDE +/-60 deg stops
    rate_hz: float = 500.0,
) -> list[tuple]:
    """Simulate a legacy free-decay session where hanging equilibrium is inside the hard stops."""
    dt = 1.0 / rate_hz
    v_band = math.radians(2.0)
    th_arm = math.radians(arm_horiz_deg)
    gc_tot = exo.gravity_cos_nm + arm_mgl * math.cos(th_arm)
    gs_tot = exo.gravity_sin_nm + arm_mgl * math.sin(th_arm)
    j_tot = exo.inertia_kg_m2 + j_human
    tc_tot = exo.coulomb_nm + coulomb_cuff
    b_tot = exo.viscous_nm_s_rad + viscous_cuff

    rows = []
    t_now = 0.0

    # Phase A holds
    for idx, ang_deg in enumerate((-25.0, 0.0, 25.0), start=1):
        q = math.radians(ang_deg)
        g_val = gc_tot * math.cos(q) + gs_tot * math.sin(q)
        for _ in range(int(1.0 * rate_hz)):
            rows.append((
                t_now, t_now + 0.0005, 1, f"gravity_hold_{idx}", idx,
                ang_deg + rng.normal(0.0, 0.01), 0.0,
                g_val + rng.normal(0.0, 0.01), g_val, 35.0, 38.0,
            ))
            t_now += dt

    # Phase C free decay (RK4)
    th_max = math.atan2(gs_tot, gc_tot)
    q_eq = th_max - math.pi / 2.0

    def accel(qv: float, qdv: float, tm: float) -> float:
        gv = gc_tot * math.cos(qv) + gs_tot * math.sin(qv)
        fv = tc_tot * math.tanh(qdv / v_band) + b_tot * qdv
        return (tm - gv - fv) / j_tot

    for tr_id, offset_deg in enumerate((45.0, -40.0), start=1):
        q_cur = q_eq + math.radians(offset_deg)
        qd_cur = 0.0
        for _ in range(int(3.5 * rate_hz)):
            tm = rng.normal(0.0, 0.005)
            k1_q, k1_v = qd_cur, accel(q_cur, qd_cur, tm)
            k2_q, k2_v = qd_cur + 0.5 * dt * k1_v, accel(q_cur + 0.5 * dt * k1_q, qd_cur + 0.5 * dt * k1_v, tm)
            k3_q, k3_v = qd_cur + 0.5 * dt * k2_v, accel(q_cur + 0.5 * dt * k2_q, qd_cur + 0.5 * dt * k2_v, tm)
            k4_q, k4_v = qd_cur + dt * k3_v, accel(q_cur + dt * k3_q, qd_cur + dt * k3_v, tm)
            q_cur += (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            qd_cur += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            rows.append((
                t_now, t_now + 0.0005, 1, f"free_decay_{tr_id}", tr_id,
                math.degrees(q_cur) + rng.normal(0.0, 0.01),
                math.degrees(qd_cur) + rng.normal(0.0, 0.25),
                tm + rng.normal(0.0, 0.006), 0.0, 35.0, 38.0,
            ))
            t_now += dt

    return rows


def test_end_to_end(out_dir: Path | None = None) -> None:
    rng = np.random.default_rng(42)

    # Realistic QDD exo baseline matching the user's real hardware (high 0.46 Nm Coulomb friction!)
    exo_real = ExoBaseline(
        inertia_kg_m2=0.002135,
        coulomb_nm=0.4617,
        viscous_nm_s_rad=0.012,
        gravity_cos_nm=-0.45,
        gravity_sin_nm=0.08,
    )
    j_human_true = 0.048
    tc_cuff_true = 0.065
    b_cuff_true = 0.032
    arm_mgl_true = 2.60

    rows, lc_rows = simulate_chirp_session(
        rng, exo_real,
        j_human=j_human_true,
        coulomb_cuff=tc_cuff_true,
        viscous_cuff=b_cuff_true,
        arm_mgl=arm_mgl_true,
    )

    tmp_ctx = None
    if out_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory()
        work = Path(tmp_ctx.name)
    else:
        work = out_dir
        work.mkdir(parents=True, exist_ok=True)

    try:
        # ---- Test 1: Default Relaxed Chirp + Bidirectional Phase A (Hard-Stop Rig) ----
        csv_path = work / "passive_exo_chirp_passive.csv"
        save_passive_csv(csv_path, rows)

        lc_path = work / "loadcell_sim.csv"
        with open(lc_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_sample", "loadcell_nm"])
            w.writerows(lc_rows)

        report, _ = run_full_analysis(csv_path, exo_real, loadcell_csv=lc_path, stop_limit_deg=60.8)
        print(format_summary(report))

        assert report.gravity_total is not None and report.gravity_total.bidirectional, "Expected bidirectional Phase A"
        assert report.gravity_arm is not None
        assert abs(report.gravity_arm.mgl_nm - arm_mgl_true) < 0.05, (
            f"Bidirectional gravity fit failed to cancel stiction: got {report.gravity_arm.mgl_nm:.3f} Nm"
        )
        assert abs(report.gravity_total.stiction_halfwidth_nm - (exo_real.coulomb_nm + tc_cuff_true)) < 0.03

        assert report.transparency is not None
        assert report.transparency.rms_interaction_torque_nm > 0.30

        assert report.passive_id is not None and report.passive_id.method == "relaxed_chirp"
        pid = report.passive_id
        j_err_pct = 100.0 * abs(pid.j_human_kg_m2 - j_human_true) / j_human_true
        tc_err = abs(pid.coulomb_cuff_nm - tc_cuff_true)
        b_err = abs(pid.viscous_cuff_nm_s_rad - b_cuff_true)
        assert j_err_pct < 3.0, f"Relaxed chirp J_human error too high: {j_err_pct:.2f}%"
        assert tc_err < 0.015, f"Relaxed chirp Coulomb cuff error too high: {tc_err:.4f} Nm"
        assert b_err < 0.01, f"Relaxed chirp viscous cuff error too high: {b_err:.4f} Nm s/rad"
        assert pid.r2 > 0.98, f"Expected R^2 > 0.98, got {pid.r2:.4f}"

        assert report.loadcell is not None and report.loadcell.pearson_r > 0.99

        # ---- Test 2: Legacy Free-Decay Regression ----
        exo_low = ExoBaseline(inertia_kg_m2=0.0085, coulomb_nm=0.030, viscous_nm_s_rad=0.015,
                              gravity_cos_nm=0.05, gravity_sin_nm=0.35)
        dec_rows = simulate_legacy_decay_session(rng, exo_low)
        dec_csv = work / "passive_exo_decay_passive.csv"
        save_passive_csv(dec_csv, dec_rows)
        dec_report, _ = run_full_analysis(dec_csv, exo_low, stop_limit_deg=60.0)
        assert dec_report.passive_id is not None and dec_report.passive_id.method == "free_decay"
        assert dec_report.passive_id.r2 > 0.97
        dec_j_err = 100.0 * abs(dec_report.passive_id.j_human_kg_m2 - 0.048) / 0.048
        assert dec_j_err < 4.0, f"Legacy decay J_human error too high: {dec_j_err:.2f}%"

        # ---- Test 3: Full CLI + 4 PNG Plot Generation ----
        cmd = [
            sys.executable,
            str(ROOT / "analyze_passive_exo.py"),
            "--csv", str(csv_path),
            "--exo-inertia", str(exo_real.inertia_kg_m2),
            "--exo-coulomb", str(exo_real.coulomb_nm),
            "--exo-viscous", str(exo_real.viscous_nm_s_rad),
            "--loadcell-csv", str(lc_path),
            "--stop-limit-deg", "60.8",
            "--out-dir", str(work),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Relaxed Motor-Driven Chirp System ID" in res.stdout
        assert (work / "passive_exo_chirp_summary.txt").exists()
        assert (work / "passive_exo_chirp_analysis.json").exists()
        assert (work / "passive_exo_chirp_01_time_series.png").exists()
        assert (work / "passive_exo_chirp_02_transparency.png").exists()
        assert (work / "passive_exo_chirp_03_passive_id_fit.png").exists()
        assert (work / "passive_exo_chirp_04_gravity_cal.png").exists()

        print(
            f"\nALL CHECKS PASSED:\n"
            f"  Chirp ID  : J_human err = {j_err_pct:.2f}%, tc_cuff err = {tc_err:.4f} Nm, "
            f"b_cuff err = {b_err:.4f} Nm s/rad, R^2 = {pid.r2:.4f}\n"
            f"  Phase A   : bidirectional stiction cancelled (arm mgl = {report.gravity_arm.mgl_nm:.3f} vs {arm_mgl_true:.3f} Nm)\n"
            f"  Legacy ID : free-decay J_human err = {dec_j_err:.2f}%, R^2 = {dec_report.passive_id.r2:.4f}"
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    test_end_to_end(out)
