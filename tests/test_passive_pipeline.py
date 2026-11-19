"""End-to-end synthetic verification suite for the Experiment 2 Passive Exo Test Package.

Simulates a complete 500 Hz passive exo session (Phase A static gravity holds, Phase B active
zero-torque human swings, Phase C 3 relaxed free-decay pendulum trials, plus optional loadcell
data) with known ground-truth parameters and realistic sensor noise + dropped CAN frames, then
verifies that exo/passive.py and analyze_passive_exo.py recover the true parameters accurately.

Run:
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

from exo.passive import (  # noqa: E402
    ExoBaseline,
    PASSIVE_COLUMNS,
    format_summary,
    run_full_analysis,
    save_passive_csv,
)


def simulate_session(
    rng: np.random.Generator,
    exo: ExoBaseline,
    j_human: float = 0.048,
    coulomb_cuff: float = 0.045,
    viscous_cuff: float = 0.032,
    arm_mgl: float = 2.80,
    arm_horiz_deg: float = -12.0,
    rate_hz: float = 500.0,
) -> tuple[list[tuple], list[tuple]]:
    """Simulate a realistic 500 Hz passive_exo dataset + matching loadcell CSV."""
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

    # ---- Phase A: 3 static holds at -25, 0, +25 deg ----
    for idx, ang_deg in enumerate((-25.0, 0.0, 25.0), start=1):
        n = int(2.0 * rate_hz)
        q = math.radians(ang_deg)
        g_val = gc_tot * math.cos(q) + gs_tot * math.sin(q)
        for _ in range(n):
            fresh = 0 if rng.random() < 0.005 else 1
            rows.append((
                t_now, t_now + 0.0005, fresh, f"gravity_hold_{idx}", idx,
                ang_deg + rng.normal(0.0, 0.02),
                rng.normal(0.0, 0.1),
                g_val + rng.normal(0.0, 0.015),
                g_val, 35.0, 38.0,
            ))
            t_now += dt

    # ---- Phase B: 12 s active human swings (slow -> medium -> fast) ----
    n_act = int(12.0 * rate_hz)
    t_b = np.arange(n_act) * dt
    # Multi-frequency smooth human trajectory
    q_b = (
        math.radians(35.0) * np.sin(2.0 * math.pi * 0.45 * t_b)
        + math.radians(15.0) * np.sin(2.0 * math.pi * 1.15 * t_b + 0.4)
    )
    qd_b = (
        math.radians(35.0) * (2.0 * math.pi * 0.45) * np.cos(2.0 * math.pi * 0.45 * t_b)
        + math.radians(15.0) * (2.0 * math.pi * 1.15) * np.cos(2.0 * math.pi * 1.15 * t_b + 0.4)
    )
    qdd_b = (
        -math.radians(35.0) * (2.0 * math.pi * 0.45) ** 2 * np.sin(2.0 * math.pi * 0.45 * t_b)
        - math.radians(15.0) * (2.0 * math.pi * 1.15) ** 2 * np.sin(2.0 * math.pi * 1.15 * t_b + 0.4)
    )
    for k in range(n_act):
        fresh = 0 if rng.random() < 0.005 else 1
        tau_m = rng.normal(0.0, 0.012)  # small zero-command current ripple
        # True loadcell torque = exo dynamics - tau_m + small cuff residual noise
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

    # ---- Phase C: 3 relaxed free-decay pendulum trials (RK4 integration) ----
    # Equilibrium angle where g_total(q) = 0 (hanging downward)
    # g_tot(q) = mgl * cos(q - th_max); torque opposing gravity in free fall is -g_tot(q)
    # Wait: positive holding torque holds against gravity, so gravity accelerates q with -g_tot(q).
    # Equilibrium stable point is q_eq = th_max - pi/2 (where g_tot(q) = 0 and dg/dq > 0).
    th_max = math.atan2(gs_tot, gc_tot)
    q_eq = th_max - math.pi / 2.0

    def accel(q_val: float, qd_val: float, tau_m_val: float) -> float:
        g_val = gc_tot * math.cos(q_val) + gs_tot * math.sin(q_val)
        f_val = tc_tot * math.tanh(qd_val / v_band) + b_tot * qd_val
        return (tau_m_val - g_val - f_val) / j_tot

    for tr_id, offset_deg in enumerate((50.0, 42.0, -45.0), start=1):
        q_cur = q_eq + math.radians(offset_deg)
        qd_cur = 0.0
        n_dec = int(4.0 * rate_hz)
        for _ in range(n_dec):
            tau_m_true = rng.normal(0.0, 0.006)
            # RK4 step
            k1_q, k1_v = qd_cur, accel(q_cur, qd_cur, tau_m_true)
            k2_q, k2_v = qd_cur + 0.5 * dt * k1_v, accel(q_cur + 0.5 * dt * k1_q, qd_cur + 0.5 * dt * k1_v, tau_m_true)
            k3_q, k3_v = qd_cur + 0.5 * dt * k2_v, accel(q_cur + 0.5 * dt * k2_q, qd_cur + 0.5 * dt * k2_v, tau_m_true)
            k4_q, k4_v = qd_cur + dt * k3_v, accel(q_cur + dt * k3_q, qd_cur + dt * k3_v, tau_m_true)
            q_cur += (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            qd_cur += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

            fresh = 0 if rng.random() < 0.005 else 1
            rows.append((
                t_now, t_now + 0.0005, fresh, f"free_decay_{tr_id}", tr_id,
                math.degrees(q_cur) + rng.normal(0.0, 0.012),
                math.degrees(qd_cur) + rng.normal(0.0, 0.30),
                tau_m_true + rng.normal(0.0, 0.008),
                0.0, 36.0, 39.0,
            ))
            t_now += dt

    return rows, lc_rows


def test_end_to_end(out_dir: Path | None = None) -> None:
    rng = np.random.default_rng(42)
    exo = ExoBaseline(
        inertia_kg_m2=0.0085,
        coulomb_nm=0.030,
        viscous_nm_s_rad=0.015,
        gravity_cos_nm=0.35,
        gravity_sin_nm=-0.08,
    )
    j_human_true = 0.048
    tc_cuff_true = 0.045
    b_cuff_true = 0.032
    arm_mgl_true = 2.80

    rows, lc_rows = simulate_session(
        rng, exo,
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
        csv_path = work / "passive_exo_sim_passive.csv"
        save_passive_csv(csv_path, rows)

        lc_path = work / "loadcell_sim.csv"
        with open(lc_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_sample", "loadcell_nm"])
            w.writerows(lc_rows)

        # 1. Direct Python API test
        report, _ = run_full_analysis(csv_path, exo, loadcell_csv=lc_path)
        print(format_summary(report))

        assert report.gravity_arm is not None, "Phase A gravity fit missing"
        assert abs(report.gravity_arm.mgl_nm - arm_mgl_true) < 0.03, f"mgl error: {report.gravity_arm.mgl_nm}"

        assert report.transparency is not None, "Phase B transparency metrics missing"
        assert report.transparency.rms_interaction_torque_nm > 0.05

        assert report.passive_id is not None, "Phase C passive ID missing"
        pid = report.passive_id
        j_err_pct = 100.0 * abs(pid.j_human_kg_m2 - j_human_true) / j_human_true
        tc_err = abs(pid.coulomb_cuff_nm - tc_cuff_true)
        b_err = abs(pid.viscous_cuff_nm_s_rad - b_cuff_true)
        assert j_err_pct < 3.0, f"J_human recovery error too high: {j_err_pct:.2f}%"
        assert tc_err < 0.01, f"Coulomb cuff error too high: {tc_err:.4f} Nm"
        assert b_err < 0.005, f"Viscous cuff error too high: {b_err:.4f} Nm s/rad"
        assert pid.r2 > 0.98, f"Expected R^2 > 0.98, got {pid.r2:.4f}"

        assert report.loadcell is not None, "Loadcell cross-validation missing"
        assert report.loadcell.pearson_r > 0.99, f"Expected loadcell r > 0.99, got {report.loadcell.pearson_r}"

        # 2. Full CLI + Plot generation test
        cmd = [
            sys.executable,
            str(ROOT / "analyze_passive_exo.py"),
            "--csv", str(csv_path),
            "--exo-inertia", str(exo.inertia_kg_m2),
            "--exo-coulomb", str(exo.coulomb_nm),
            "--exo-viscous", str(exo.viscous_nm_s_rad),
            "--loadcell-csv", str(lc_path),
            "--out-dir", str(work),
        ]
        # Note: CLI without joint_cal.json has g_exo=0, which still runs cleanly and produces all 4 plots
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Passive Exo Analysis:" in res.stdout
        assert (work / "passive_exo_sim_summary.txt").exists()
        assert (work / "passive_exo_sim_analysis.json").exists()
        assert (work / "passive_exo_sim_01_time_series.png").exists()
        assert (work / "passive_exo_sim_02_transparency.png").exists()
        assert (work / "passive_exo_sim_03_free_decay_fit.png").exists()
        assert (work / "passive_exo_sim_04_gravity_cal.png").exists()

        print(
            f"\nALL CHECKS PASSED: J_human error = {j_err_pct:.2f}%, "
            f"coulomb err = {tc_err:.4f} Nm, viscous err = {b_err:.4f} Nm s/rad, R^2 = {pid.r2:.4f}"
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    test_end_to_end(out)
