"""Offline analysis and plotting for Experiment 2 (Passive Exoskeleton Test).

    pixi run analyze-passive -- logs/passive_exo_20261121_164733_passive.csv
    python analyze_passive_exo.py --csv logs/passive_exo_..._passive.csv --frf logs/free_chirp_..._frf.csv

Outputs (saved alongside the CSV or in --out-dir):
    <stem>_summary.txt               human-readable summary report
    <stem>_analysis.json             machine-readable metrics & parameter estimates
    <stem>_01_time_series.png        angle, velocity, and motor torque across phases
    <stem>_02_transparency.png       active swing interaction torque breakdown & hysteresis loop
    <stem>_03_passive_id_fit.png     Phase C system-ID model fit, residuals, and Exo vs. Human/Cuff bars
    <stem>_04_gravity_cal.png        Phase A static gravity calibration fit (if Phase A was run)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np

from exo.calibration import CAL_FILE, Calibration
from exo.logs import LOG_DIR
from exo.passive import (
    DEFAULT_V_BAND_DEG_S,
    PassiveAnalysisReport,
    format_summary,
    load_exo_baseline,
    run_full_analysis,
    save_analysis_outputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Experiment 2 Passive Exo Test data")
    p.add_argument("csv_pos", nargs="?", type=Path, default=None,
                   help="Path to passive_exo_*_passive.csv (default: latest in logs/)")
    p.add_argument("--csv", type=Path, default=None, help="Path to passive_exo_*_passive.csv")
    p.add_argument("--cal", type=Path, default=None, help="Path to calibration/joint_cal.json")
    p.add_argument("--frf", type=Path, default=None, help="Path to Experiment 1 free_chirp_*_frf.csv")
    p.add_argument("--exo-inertia", type=float, default=None,
                   help="Override J_exo [kg m^2] (e.g., 0.0021)")
    p.add_argument("--exo-coulomb", type=float, default=None, help="Override exo Coulomb friction [Nm]")
    p.add_argument("--exo-viscous", type=float, default=None, help="Override exo viscous damping [Nm s/rad]")
    p.add_argument("--arm-mgl", type=float, default=None,
                   help="Fallback human forearm gravity m*g*l [Nm] if Phase A was skipped (free-decay mode)")
    p.add_argument("--arm-horizontal-deg", type=float, default=0.0,
                   help="Joint angle [deg] where forearm is horizontal (used with --arm-mgl)")
    p.add_argument("--loadcell-csv", type=Path, default=None,
                   help="Optional external loadcell CSV (columns: t or t_sample, loadcell_nm or torque_nm)")
    p.add_argument("--window-s", type=float, default=0.06,
                   help="Savitzky-Golay smoothing window [s] (default: 0.06; chirp uses <=0.04)")
    p.add_argument("--v-band", type=float, default=DEFAULT_V_BAND_DEG_S,
                   help="Smooth Coulomb friction velocity band [deg/s] (default: 2.0)")
    p.add_argument("--min-speed", type=float, default=3.0,
                   help="Minimum speed [deg/s] included in Phase C regression (default: 3.0)")
    p.add_argument("--stop-limit-deg", type=float, default=None,
                   help="Hard-stop half-stroke [deg] for masking stop impacts (default: from joint_cal.json)")
    p.add_argument("--stop-margin-deg", type=float, default=5.0,
                   help="Margin inside hard stops excluded from Phase C regression [deg] (default: 5.0)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for report and plots (default: same folder as input CSV)")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib PNG generation")
    return p.parse_args()


def _resolve_latest(pattern: str) -> Optional[Path]:
    matches = sorted(LOG_DIR.rglob(pattern))
    return matches[-1] if matches else None


def generate_plots(
    stem: Path,
    report: PassiveAnalysisReport,
    artifacts: dict,
) -> list[Path]:
    """Generate diagnostic PNG figures using matplotlib (skipped cleanly if matplotlib unavailable)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Note: matplotlib is not installed; skipping PNG plots (install matplotlib to enable).")
        return []

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })

    created: list[Path] = []
    records = artifacts["records"]
    active = artifacts.get("active_series")
    phase_c = artifacts.get("decay_series")

    # 1. Full time-series overview
    fresh = [r for r in records if r["fresh"] == 1]
    if fresh:
        t = np.array([r["t_sample"] for r in fresh], dtype=float)
        t = t - t[0]
        ang = np.array([r["angle_deg"] for r in fresh], dtype=float)
        vel = np.array([r["velocity_deg_s"] for r in fresh], dtype=float)
        tau = np.array([r["torque_nm"] for r in fresh], dtype=float)
        phases = [r["phase"] for r in fresh]

        fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
        fig.suptitle(f"Passive Exo Test Overview — {Path(report.csv_path).name}", fontweight="bold")
        axes[0].plot(t, ang, color="#1f4e79", lw=1.2)
        axes[0].set_ylabel("Angle [deg]")
        axes[1].plot(t, vel, color="#2e7d32", lw=1.1)
        axes[1].set_ylabel("Velocity [deg/s]")
        axes[2].plot(t, tau, color="#c62828", lw=1.0)
        axes[2].set_ylabel("Motor Torque [Nm]")
        axes[2].set_xlabel("Time [s]")

        phase_colors = {
            "gravity_hold": "#fff3cd",
            "active_swing": "#d1ecf1",
            "relaxed_chirp": "#e2d9f3",
            "amp_sweep": "#d4edda",
            "multi_amp_sweep": "#cce5ff",
            "free_decay": "#f8d7da",
        }
        i = 0
        labeled = set()
        while i < len(phases):
            j = i + 1
            while j < len(phases) and phases[j] == phases[i]:
                j += 1
            prefix = next((k for k in phase_colors if phases[i].startswith(k)), None)
            if prefix is not None:
                lbl = prefix.replace("_", " ") if prefix not in labeled else None
                labeled.add(prefix)
                for ax in axes:
                    ax.axvspan(t[i], t[j - 1], color=phase_colors[prefix], alpha=0.35, label=lbl)
                    lbl = None
            i = j
        if labeled:
            axes[0].legend(loc="upper right", frameon=True, ncol=4, fontsize=8.5)
        fig.tight_layout()
        p1 = Path(f"{stem}_01_time_series.png")
        fig.savefig(p1)
        plt.close(fig)
        created.append(p1)

    # 2. Phase B Transparency Breakdown & Hysteresis Loop
    if active is not None and report.transparency is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.45, 1.0]})
        fig.suptitle("Phase B: Mechanical Transparency in Zero-Torque Mode", fontweight="bold")

        t_a = active["t"] - active["t"][0]
        ax1.plot(t_a, active["tau_h_hat"], color="#111111", lw=1.4, label=r"Total $\hat{\tau}_h$")
        ax1.plot(t_a, active["tau_inertia"], color="#1565c0", lw=1.0, alpha=0.85, label="Inertia ($J_{exo}\\ddot{q}$)")
        ax1.plot(t_a, active["tau_friction"], color="#e65100", lw=1.0, alpha=0.85, label="Friction ($\\tau_{f,exo}$)")
        ax1.plot(t_a, active["tau_gravity"], color="#2e7d32", lw=0.9, ls="--", alpha=0.8, label="Gravity ($g_{exo}$)")
        ax1.plot(t_a, -active["tau_m"], color="#8e24aa", lw=0.8, alpha=0.7, label="Motor ripple ($-\\tau_m$)")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Resistive Torque [Nm]")
        ax1.set_title(
            f"Interaction Torque Breakdown (RMS {report.transparency.rms_interaction_torque_nm:.3f} Nm)"
        )
        ax1.legend(loc="upper right", fontsize=8, ncol=2)

        ax2.plot(active["qd_deg_s"], active["tau_h_hat"], color="#1565c0", lw=0.8, alpha=0.75)
        ax2.axhline(0.0, color="#666666", lw=0.7)
        ax2.axvline(0.0, color="#666666", lw=0.7)
        ax2.set_xlabel("Angular Velocity [deg/s]")
        ax2.set_ylabel(r"Estimated Interaction Torque $\hat{\tau}_h$ [Nm]")
        ax2.set_title("Backdrivability Hysteresis Loop")

        fig.tight_layout()
        p2 = Path(f"{stem}_02_transparency.png")
        fig.savefig(p2)
        plt.close(fig)
        created.append(p2)

    # 3. Phase C System ID Fit & Parameter Decomposition (Relaxed Chirp or Free Decay)
    if phase_c is not None and report.passive_id is not None and phase_c["trials"]:
        pid = report.passive_id
        b = report.exo_baseline
        tr0 = phase_c["trials"][0]
        is_chirp = pid.method == "relaxed_chirp"

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.55, 0.72, 1.0]}
        )
        mode_label = "Multi-Freq Amplitude Sweep ID" if pid.method == "multi_amp_sweep" else "Fixed-Freq Amplitude Ramp ID" if pid.method == "amp_sweep" else ("Relaxed Chirp System ID" if is_chirp else "Relaxed Free-Decay System ID")
        fig.suptitle(
            f"Phase C: {mode_label} (Pooled $R^2$ = {pid.r2:.3f}, RMS = {pid.rms_residual_nm:.4f} Nm)",
            fontweight="bold",
        )

        meas_lbl = r"Measured $\tau_m$" if is_chirp else r"Measured $\tau_m - g_{total}(q)$"
        fit_lbl = (
            r"Fitted $J\ddot{q} + \tau_f(\dot{q}) + k_{eff}q + \tau_0$"
            if is_chirp
            else r"Fitted $J\ddot{q} + \tau_f(\dot{q})$"
        )
        ax1.plot(tr0["t"], tr0["y_meas"], color="#1f4e79", lw=1.1, label=meas_lbl)
        ax1.plot(tr0["t"], tr0["y_fit_trial"], color="#d84315", lw=1.0, ls="--", label=fit_lbl)
        ax1.plot(tr0["t"], tr0["y_meas"] - tr0["y_fit_trial"], color="#757575", lw=0.7, alpha=0.8, label="Residual")
        ax1.set_xlabel("Elapsed Time [s]")
        ax1.set_ylabel("Torque [Nm]")
        ax1.set_title("Relaxed Chirp Torque Fit" if is_chirp else f"Free-Decay Torque Balance (Trial {tr0['trial']})")
        ax1.legend(loc="upper right", fontsize=8)

        j_bars = ax2.bar(
            ["Exo\n(Exp 1)", "Human\n(Exp 2)"],
            [b.inertia_kg_m2 * 1e3, pid.j_human_kg_m2 * 1e3],
            width=0.52,
            color=["#455a64", "#00897b"],
        )
        ax2.axhline(0.0, color="#333333", lw=0.7)
        ax2.set_ylabel("Inertia [$10^{-3}$ kg m$^2$]")
        ax2.set_title("Inertia Breakdown")
        ax2.margins(y=0.18)
        for bar in j_bars:
            v = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

        f_labels = ["Coulomb\n[mNm]", "Viscous\n[mNm s/rad]"]
        exo_f = [b.coulomb_nm * 1e3, b.viscous_nm_s_rad * 1e3]
        cuff_f = [pid.coulomb_cuff_nm * 1e3, pid.viscous_cuff_nm_s_rad * 1e3]
        xf = np.arange(len(f_labels))
        wf = 0.34
        b1 = ax3.bar(xf - wf / 2, exo_f, width=wf, color="#455a64", label="Exo Baseline (Exp 1)")
        b2 = ax3.bar(xf + wf / 2, cuff_f, width=wf, color="#00897b", label="Skin-Cuff + Elbow (Exp 2)")
        ax3.set_xticks(xf)
        ax3.set_xticklabels(f_labels)
        ax3.axhline(0.0, color="#333333", lw=0.7)
        ax3.set_ylabel("mNm  or  mNm s/rad")
        ax3.set_title("Friction & Damping")
        ax3.margins(y=0.22)
        ax3.legend(loc="upper right", fontsize=8)
        for bar in list(b1) + list(b2):
            v = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)

        fig.tight_layout()
        p3 = Path(f"{stem}_03_passive_id_fit.png")
        fig.savefig(p3)
        plt.close(fig)
        created.append(p3)

    # 4. Phase A Gravity Calibration
    if report.gravity_total is not None and report.gravity_total.hold_angles_deg:
        gt = report.gravity_total
        b = report.exo_baseline
        q_grid = np.linspace(-60.0, 60.0, 200)
        q_rad = np.radians(q_grid)

        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        hold_lbl = (
            "Bidirectional avg holds (stiction-cancelled)"
            if gt.bidirectional
            else "Measured static holds (exo + arm)"
        )
        ax.scatter(gt.hold_angles_deg, gt.hold_torques_nm, color="#1f4e79", s=55, zorder=5, label=hold_lbl)
        ax.plot(q_grid, gt.torque(q_rad), color="#1f4e79", lw=1.6,
                label=f"Fitted $g_{{total}}(q)$ ($mgl$={gt.mgl_nm:.2f} Nm)")
        ax.plot(q_grid, b.gravity(q_rad), color="#757575", lw=1.3, ls="--",
                label=f"Exo-only $g_{{exo}}(q)$ ($mgl$={math.hypot(b.gravity_cos_nm, b.gravity_sin_nm):.2f} Nm)")
        if report.gravity_arm is not None:
            ax.plot(q_grid, report.gravity_arm.torque(q_rad), color="#00897b", lw=1.4, ls=":",
                    label=f"Human forearm $g_{{arm}}(q)$ ($mgl$={report.gravity_arm.mgl_nm:.2f} Nm)")
        ax.set_xlabel("Joint Angle [deg]")
        ax.set_ylabel("Static Gravity Torque [Nm]")
        title = "Phase A: Static Gravity Calibration (Relaxed Forearm)"
        if gt.bidirectional and math.isfinite(gt.stiction_halfwidth_nm):
            title += f" — Stiction +/-{gt.stiction_halfwidth_nm:.2f} Nm"
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8.5)
        fig.tight_layout()
        p4 = Path(f"{stem}_04_gravity_cal.png")
        fig.savefig(p4)
        plt.close(fig)
        created.append(p4)

    return created


def main() -> None:
    args = parse_args()
    csv_path = args.csv or args.csv_pos or _resolve_latest("passive_exo_*_passive.csv")
    if csv_path is None or not csv_path.exists():
        raise SystemExit(
            "No passive_exo CSV found. Pass a path (e.g., python analyze_passive_exo.py logs/passive_exo_..._passive.csv)."
        )

    frf_path = args.frf or _resolve_latest("free_chirp_*_frf.csv")
    baseline = load_exo_baseline(
        cal_path=args.cal,
        frf_path=frf_path,
        inertia_override_kg_m2=args.exo_inertia,
        coulomb_override_nm=args.exo_coulomb,
        viscous_override_nm_s_rad=args.exo_viscous,
    )

    stop_limit_deg = args.stop_limit_deg
    if stop_limit_deg is None:
        cp = Path(args.cal) if args.cal is not None else CAL_FILE
        if cp.exists():
            try:
                stop_limit_deg = Calibration.load(cp).limit_deg
            except Exception:
                stop_limit_deg = None

    report, artifacts = run_full_analysis(
        csv_path=csv_path,
        exo_baseline=baseline,
        window_s=args.window_s,
        v_band_deg_s=args.v_band,
        min_speed_deg_s=args.min_speed,
        arm_mgl_nm=args.arm_mgl,
        arm_horizontal_deg=args.arm_horizontal_deg,
        loadcell_csv=args.loadcell_csv,
        stop_limit_deg=stop_limit_deg,
        stop_margin_deg=args.stop_margin_deg,
    )
    summary = format_summary(report)
    print(summary)

    out_dir = args.out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = csv_path.stem.removesuffix("_passive")
    stem = out_dir / base_name
    save_analysis_outputs(stem, report, summary)
    print(f"\nSaved: {stem}_summary.txt")
    print(f"Saved: {stem}_analysis.json")

    if not args.no_plots:
        plots = generate_plots(stem, report, artifacts)
        for p in plots:
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()
