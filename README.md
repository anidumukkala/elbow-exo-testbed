# Elbow Exoskeleton Testbed — Experiment 2: Passive Exo Test Package

This package extends `elbow-exo-testbed` with a ready-to-use **Experiment 2 (Passive Exoskeleton)** data collection, system identification, and analysis suite.

## 1. Premise & Experimental Goal

To compare the **UME-Exo elbow assembly (40:1 actuator, heavier brace)** against the **lightweight QDD elbow brace (DM-J4310-2EC, 10:1 actuator)**, the test plan evaluates:

1. **Experiment 1 — Actuator + Exo Dynamics** *(already in repo: `home_auto.py` / `home_manual.py` + `free_chirp.py`)*  
   With no human arm inserted, identifies the exoskeleton's baseline reflected inertia ($J_{\text{exo}} = J_a + J_l$), Coulomb/viscous friction ($\tau_{c,\text{exo}}, b_{\text{exo}}$), and link gravity ($g_{\text{exo}}(q)$).
2. **Experiment 2 — Passive Exo** *(this package: `passive_exo.py`, `exo/passive.py`, `analyze_passive_exo.py`)*  
   With a **human arm inserted** and the motor commanded to **zero-torque mode** ($\tau_{\text{cmd}} = 0\ \text{Nm}$, $k_p = k_d = 0$):
   - **Mechanical transparency & backdrivability (Phase B):** Human actively swings their forearm to backdrive the exo; quantifies the resistive interaction torque $\hat{\tau}_h(t)$, speed-binned drag, and mechanical work dissipated.
   - **Human forearm inertia & skin-cuff interface friction (Phase A + Phase C):** Measures static gravity torque of the relaxed forearm (Phase A) to establish an absolute Nm scale, then records relaxed free-decay pendulum swings in zero-torque mode (Phase C) to identify total inertia and friction and subtract the Experiment 1 exo baseline:
     $$\hat{J}_{\text{human}} = \hat{J}_{\text{total}} - J_{\text{exo}}, \qquad \hat{\tau}_{c,\text{cuff}} = \hat{\tau}_{c,\text{total}} - \tau_{c,\text{exo}}, \qquad \hat{b}_{\text{cuff}} = \hat{b}_{\text{total}} - b_{\text{exo}}$$

---

## 2. Package Structure

New files added for Experiment 2 are marked with **[NEW]**:

| File | Role |
| :--- | :--- |
| `exo/passive.py` **[NEW]** | Core data schema (`PASSIVE_COLUMNS`), Experiment 1 baseline loader, pure-NumPy zero-phase Savitzky-Golay smoothing/differentiation, gravity calibration, transparency metrics, free-decay least-squares ID, optional loadcell cross-validation, and report formatting. Uses only Python stdlib + NumPy (no new Raspberry Pi dependencies). |
| `passive_exo.py` **[NEW]** | Real-time Experiment 2 test runner and 500 Hz data collector with zero-torque mode, guided 3-phase protocol, live terminal telemetry, dry-run mode, and human-in-the-loop safety guards. |
| `analyze_passive_exo.py` **[NEW]** | Standalone offline analysis CLI that loads `_passive.csv`, `calibration/joint_cal.json`, Experiment 1 `_frf.csv` (and optional loadcell CSV), prints/saves the full report, and generates 4 diagnostic PNG plots (when `matplotlib` is installed). |
| `tests/test_passive_pipeline.py` **[NEW]** | End-to-end synthetic 500 Hz simulation & verification suite (no hardware required). |
| `pixi.toml` *(updated)* | Adds `passive-exo`, `passive-exo-dry`, `analyze-passive`, and `test-passive` tasks. |
| `exo/{config,motor,joint,loop,logs,calibration,compensation,chirp}.py`, `home_auto.py`, `home_manual.py`, `free_chirp.py`, `setup/can_up.sh` | Original testbed codebase preserved intact. |

---

## 3. Prerequisites & Safety Checklist

Before strapping a human arm into the exoskeleton:

1. **Bring up CAN FD:**
   ```bash
   pixi run can-up
   ```
2. **Home the joint (nobody wearing the exo, hands clear):**
   ```bash
   pixi run home-auto      # or: pixi run home-manual
   ```
   This saves `calibration/joint_cal.json` with hard-stop limits, exo Coulomb/viscous friction, and exo gravity $g_{\text{exo}}(q)$.
3. **Run Experiment 1 baseline (nobody wearing the exo):**
   ```bash
   pixi run chirp-free
   ```
   This saves `logs/free_chirp_<timestamp>_frf.csv`, which `passive_exo.py` and `analyze_passive_exo.py` automatically pick up to obtain $J_{\text{exo}}$.
4. **Verify timing with a dry run (optional, motor disabled):**
   ```bash
   pixi run passive-exo-dry
   ```
5. **Confirm motor CAN IDs:** `passive_exo.py` defaults to `--motor-id 0x02 --feedback-id 0x12` (matching `free_chirp.py` and `home_manual.py`). Pass `--motor-id 0x01 --feedback-id 0x11` if using the other motor.

### Built-In Human-in-the-Loop Safety Guards
- **Zero-torque guard:** Aborts and disables the motor if measured $|\tau_m| > 0.50\ \text{Nm}$ for 3 consecutive fresh replies while commanding `0.0 Nm`.
- **Speed guard:** Aborts if $|\dot{q}| > 400\ \text{deg/s}$ for 3 consecutive fresh replies.
- **Hard-stop guard:** `Joint.update()` aborts if $|q|$ exceeds the calibrated stop limits; the console also flashes `[NEAR STOP!]` within `8 deg` of a stop.
- **Thermal & hardware guards:** Rotor temperature rise $> 15\ ^\circ\text{C}$, `DamiaoMotor` `1.5 Nm` torque trip, firmware `~100 ms` CAN watchdog, and instant `Ctrl-C` motor disable.

---

## 4. General Test Procedure

### Step 1 — Run the Test Script
```bash
pixi run passive-exo -- --rig qdd_10to1 --subject S01
```
*(Or directly: `python passive_exo.py --rig qdd_10to1 --subject S01 --exo-frf logs/free_chirp_<time>_frf.csv`)*

Strap the subject's forearm securely into the cuff so the anatomical elbow axis aligns with the motor shaft axis. Ensure the upper arm is comfortably supported on the bench without restricting forearm flexion/extension.

### Step 2 — Phase A: Static Gravity Calibration (Relaxed Arm, ~15 s)
- **Why:** In zero-torque free decay ($\tau_m \approx 0$), the homogeneous equation $J\ddot{q} + \tau_f(\dot{q}) + g(q) = 0$ requires a known torque scale ($g_{\text{total}}(q)$ in Nm) to identify absolute $J_{\text{total}}$, $\tau_{c,\text{total}}$, and $b_{\text{total}}$.
- **Procedure:** Instruct the subject to **completely relax** their forearm. Press `Enter`. The exo slowly (`15 deg/s`) moves to three static angles (`-25, 0, +25 deg`, clipped inside soft limits) and holds each for `2.0 s` to measure $g_{\text{total}}(q) = g_{\text{exo}}(q) + g_{\text{arm}}(q)$, then immediately releases to `0.0 Nm`.
- *Note:* If your rig operates purely in a horizontal plane (no gravity torque) or you want to supply an anthropometric forearm $mgl$ offline, pass `--skip-gravity-cal`.

### Step 3 — Phase B: Active Human Swings in Zero-Torque Mode (30 s)
- **Procedure:** With the motor commanding `0.0 Nm`, press `Enter` to start recording. Have the subject smoothly flex and extend their elbow through a comfortable range of motion (staying clear of the hard stops), following the terminal pace cues:
  - `0–10 s`: **SLOW** (~0.3–0.5 Hz, $<60\ \text{deg/s}$)
  - `10–20 s`: **MEDIUM** (~0.8–1.2 Hz, $60\text{–}150\ \text{deg/s}$)
  - `20–30 s`: **FAST** (~1.5–2.5 Hz, $>150\ \text{deg/s}$)
- Press `Enter` at any time to finish early, or let the 30 s timer complete.

### Step 4 — Phase C: Relaxed Free-Decay Swings (3 Trials, ~5–8 s each)
- **Procedure:** The motor stays at `0.0 Nm`. For each of the 3 trials:
  1. Lift the subject's forearm to a flexed starting angle (e.g., `+35 to +55 deg` above resting equilibrium).
  2. Press `Enter` to start recording and **immediately release** the arm while the subject keeps all elbow muscles completely relaxed (like a passive pendulum).
  3. Recording stops automatically once $|\dot{q}| < 2\ \text{deg/s}$ for `1.2 s` (or press `Enter`).
- Repeat 3 times to verify trial-to-trial repeatability.

---

## 5. Data Collection & Offline Analysis

### Recorded Files (in `logs/`)
Each run writes:
- `passive_exo_<timestamp>_passive.csv` — per-tick 500 Hz dataset with columns:  
  `t, t_sample, fresh, phase, trial, angle_deg, velocity_deg_s, torque_nm, tau_cmd_nm, t_mos_c, t_rotor_c`
- `passive_exo_<timestamp>_meta.json` — rig label, subject ID, CAN IDs, rate, baseline FRF path.
- `passive_exo_<timestamp>_summary.txt` & `_analysis.json` — auto-generated at the end of the run.
- `passive_exo_<timestamp>.csv` & `.log` — standard `Joint` tick log and diagnostic log.

### Running Offline Analysis & Generating Plots
```bash
python analyze_passive_exo.py \
  --csv logs/passive_exo_<timestamp>_passive.csv \
  --cal calibration/joint_cal.json \
  --frf logs/free_chirp_<timestamp>_frf.csv
```
Optional flags:
- `--exo-inertia 0.0085` — manually set $J_{\text{exo}}$ ($\text{kg}\cdot\text{m}^2$) if no `_frf.csv` is available.
- `--arm-mgl 2.8 --arm-horizontal-deg -10` — fallback anthropometric forearm gravity if Phase A was skipped.
- `--loadcell-csv path/to/loadcell.csv` — cross-validate $\hat{\tau}_h(t)$ against an external cuff loadcell (columns `t_sample` or `t`, and `loadcell_nm` or `torque_nm`).
- `--window-s 0.06` — Savitzky-Golay smoothing window (default `60 ms`, ~10 Hz effective bandwidth).
- `--out-dir results/qdd_S01` — write reports and the 4 PNG plots to a dedicated folder.

### Verifying the Pipeline Without Hardware
```bash
pixi run test-passive
# or: python tests/test_passive_pipeline.py
```
This simulates a noisy 500 Hz session with known ground-truth parameters and verifies accurate recovery of $J_{\text{human}}$, $\tau_{c,\text{cuff}}$, $b_{\text{cuff}}$, transparency metrics, loadcell cross-validation, and all 4 plots.

---

## 6. How to Interpret the Results

### A. Mechanical Transparency Metrics (Phase B — Active Swings)
| Metric | Physical Meaning | What Good (QDD 10:1) Looks Like vs. High-Ratio (UME 40:1) |
| :--- | :--- | :--- |
| **`interaction tau_h` (RMS & Peak, Nm)** | Total resistive torque $\hat{\tau}_h = J_{\text{exo}}\ddot{q} + \tau_{f,\text{exo}}(\dot{q}) + g_{\text{exo}}(q) - \tau_m$ the human must overcome to move the exo. | **Lower is better.** QDD 10:1 should show substantially lower RMS/peak $\hat{\tau}_h$, especially in the **medium** and **fast** speed bins where reflected inertia $J_a \propto N^2$ dominates. |
| **Breakdown: `inertia` vs. `friction` vs. `gravity`** | Shows which physical term dominates human effort. | High-ratio rigs are typically **inertia- and friction-dominated** at fast speeds; a lightweight QDD rig shifts resistance toward much smaller inertia and residual gravity. |
| **`motor 0-Nm ripple` (RMS & Peak, Nm)** | Unintended torque measured by the motor while commanding `0.0 Nm` (current-loop noise, back-EMF, cogging). | Should stay near the sensor noise floor ($< 0.03\text{–}0.05\ \text{Nm}$ RMS). Large values scaling with speed indicate back-EMF or current-loop lag. |
| **`mechanical work` (J) & `mean power` (W)** | Total energy the human expended fighting the exoskeleton during the swing trial. | Lower energy per unit swing amplitude/speed indicates superior mechanical transparency. |
| **Plot `02_transparency.png` (Hysteresis Loop)** | Plots $\hat{\tau}_h$ vs. $\dot{q}$. | A **thin, nearly flat band** near 0 Nm means high backdrivability. A **wide vertical loop** indicates high Coulomb friction + inertial phase lag (classic signature of high reflected inertia). |

### B. Human Forearm Inertia & Skin-Cuff Interface Friction (Phase C — Free Decay)
| Parameter | Physical Meaning | Typical Range & Sanity Checks |
| :--- | :--- | :--- |
| **`J_human` ($\text{kg}\cdot\text{m}^2$)** | Estimated inertia of the subject's forearm + hand about the elbow axis ($J_{\text{total}} - J_{\text{exo}}$). | **Typical adult range:** $30\text{–}80 \times 10^{-3}\ \text{kg}\cdot\text{m}^2$ ($0.03\text{–}0.08\ \text{kg}\cdot\text{m}^2$), depending on stature and hand posture. Should remain **consistent** for the same subject across both rigs (UME 40:1 vs. QDD 10:1)! |
| **`coulomb_cuff` (Nm)** | Additional dry/hysteretic friction beyond the bare exo ($\tau_{c,\text{total}} - \tau_{c,\text{exo}}$). | Captures **skin-cuff strap friction/hysteresis plus passive elbow joint friction**. Higher strap tension or soft-tissue shear increases this term. |
| **`viscous_cuff` ($\text{Nm}\cdot\text{s/rad}$)** | Additional velocity-dependent damping beyond the bare exo ($b_{\text{total}} - b_{\text{exo}}$). | Captures **skin-cuff viscoelastic damping plus passive elbow joint damping** (typically $0.01\text{–}0.15\ \text{Nm}\cdot\text{s/rad}$). |
| **Fit Quality (`R^2`, `residual RMS`, `cond(X)`)** | Goodness of fit of the passive rigid-body + friction model. | Target **$R^2 \ge 0.90$**, low `residual RMS` ($< 0.05\text{–}0.10\ \text{Nm}$), `cond(X) < 50`, and consistent per-trial $J_{\text{total}}$ (within ~10% across the 3 trials). |

### C. Troubleshooting Common Experimental Artifacts
- **$R^2 < 0.85$ or large trial-to-trial spread in $J_{\text{total}}$:** The subject was likely subtly guarding or assisting the swing with their biceps/triceps. Repeat Phase C with explicit coaching to stay completely limp ("dead weight"), and check `03_free_decay_fit.png` for non-zero residual spikes near reversals.
- **Negative `coulomb_cuff` or `viscous_cuff`:** Either the subject unconsciously assisted motion, or `calibration/joint_cal.json` was measured when the motor/gearbox was cold (higher friction) while Experiment 2 ran warm. Re-run homing or verify `J_exo` and friction baselines at the same operating temperature.
- **`J_human` much higher or lower than $30\text{–}80 \times 10^{-3}\ \text{kg}\cdot\text{m}^2$:** Check Phase A (`04_gravity_cal.png`) first—if the subject tensed their arm during the static gravity holds, $g_{\text{total}}(q)$ is biased, which directly scales $J_{\text{total}}$ proportionally! Also confirm the elbow joint center is aligned with the motor shaft (misalignment adds cuff shear and alters effective moment arm).
- **Loadcell Cross-Validation (`r` and `RMSE`):** When a cuff loadcell is attached, high correlation ($r > 0.95$) and low RMSE confirm that the exo dynamics model accurately predicts true human–exoskeleton interaction torque; any systematic residual $(\Delta J, \Delta \tau_c, \Delta b)$ quantifies unmodeled cuff compliance or baseline drift.
