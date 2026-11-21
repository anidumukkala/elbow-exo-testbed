# Elbow Exoskeleton Testbed — Experiment 2: Passive Exo Test Package

This package extends `elbow-exo-testbed` with a ready-to-use **Experiment 2 (Passive Exoskeleton)** data collection, system identification, and analysis suite.

## 1. Premise & Experimental Goal

To compare the **UME-Exo elbow assembly (40:1 actuator, heavier brace)** against the **lightweight QDD elbow brace (DM-J4310-2EC, 10:1 actuator)**, the test plan evaluates:

1. **Experiment 1 — Actuator + Exo Dynamics** *(already in repo: `home_auto.py` / `home_manual.py` + `free_chirp.py`)*  
   With no human arm inserted, identifies the exoskeleton's baseline reflected inertia ($J_{\text{exo}} = J_a + J_l$), Coulomb/viscous friction ($\tau_{c,\text{exo}}, b_{\text{exo}}$), and link gravity ($g_{\text{exo}}(q)$).
2. **Experiment 2 — Passive Exo** *(this package: `passive_exo.py`, `exo/passive.py`, `analyze_passive_exo.py`)*  
   With a **human arm inserted**:
   - **Phase A — Bidirectional Static Gravity Calibration (Relaxed Arm):** Approaches each static hold angle (`-25, 0, +25 deg`) from **both `+` and `-` directions** and averages the holding torques to cancel Coulomb stiction and isolate $g_{\text{arm}}(q) = g_{\text{total}}(q) - g_{\text{exo}}(q)$.
   - **Phase B — Mechanical Transparency & Backdrivability (Zero-Torque Mode):** Commands $\tau_{\text{cmd}} = 0\ \text{Nm}$ ($k_p = k_d = 0$) while the human actively swings their forearm; quantifies resistive interaction torque $\hat{\tau}_h(t)$, speed-binned drag, and mechanical work dissipated.
   - **Phase C — Human Forearm Inertia & Skin-Cuff Friction ID (Default: Gentle Relaxed Motor Chirp):** Because the testbed has mechanical hard stops ($\approx \pm 60^\circ$) that prevent the forearm from oscillating freely around its vertical hanging equilibrium (and dry friction can exceed gravity), Phase C defaults to a **gentle, small-amplitude motor-driven chirp** (`0.5 -> 6.0 Hz`, $\pm 8^\circ$ around `0 deg`, `30 s`) while the subject stays completely relaxed:
     $$\tau_m(t) = J_{\text{total}}\ddot{q} + \tau_{c,\text{total}}\tanh(\dot{q}/v_{\text{band}}) + b_{\text{total}}\dot{q} + k_{\text{eff}}q + \tau_0$$
     Subtracting the Experiment 1 exo baseline yields:
     $$\hat{J}_{\text{human}} = \hat{J}_{\text{total}} - J_{\text{exo}}, \qquad \hat{\tau}_{c,\text{cuff}} = \hat{\tau}_{c,\text{total}} - \tau_{c,\text{exo}}, \qquad \hat{b}_{\text{cuff}} = \hat{b}_{\text{total}} - b_{\text{exo}}$$
     *(Legacy free-decay pendulum mode remains available via `--phase-c-mode decay` for rigs whose hanging equilibrium sits well inside the hard stops.)*

---

## 2. Prerequisites & Safety Checklist

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
5. **Confirm motor CAN IDs:** `passive_exo.py` defaults to `--motor-id 0x02 --feedback-id 0x12` (matching `free_chirp.py` and `home_auto.py`). Pass `--motor-id 0x01 --feedback-id 0x11` if using the other motor.

### Built-In Human-in-the-Loop Safety Guards
- **Zero-torque guard (Phase B):** Aborts and disables the motor if measured $|\tau_m| > 1.0\ \text{Nm}$ for 3 consecutive fresh replies while commanding `0.0 Nm`.
- **Relaxed-chirp guards (Phase C):** Aborts if $|\tau_m - \tau_{\text{hold}}| > 2.5\ \text{Nm}$ for 3 consecutive fresh replies, or if $|q|$ exceeds `chirp_amp + 8 deg` ($\pm 16^\circ$ by default).
- **Speed guard:** Aborts if $|\dot{q}| > 400\ \text{deg/s}$ for 3 consecutive fresh replies.
- **Hard-stop guard:** `Joint.update()` aborts if $|q|$ exceeds the calibrated stop limits; the console flashes `[NEAR STOP!]` within `8 deg` of a stop.
- **Thermal & hardware guards:** Rotor temperature rise $> 15\ ^\circ\text{C}$, `DamiaoMotor` `torque_trip_nm` (`5.0 Nm` in `exo/config.py`), firmware `~100 ms` CAN watchdog, and instant `Enter` / `Ctrl-C` stop.

---

## 3. General Test Procedure

### Step 1 — Run the Test Script
```bash
pixi run passive-exo -- --rig qdd_10to1 --subject S01
```
*(Or directly: `python passive_exo.py --rig qdd_10to1 --subject S01 --exo-frf logs/free_chirp_<time>_frf.csv`)*

Strap the subject's forearm securely into the cuff so the anatomical elbow axis aligns with the motor shaft axis. Ensure the upper arm is comfortably supported on the bench without restricting forearm flexion/extension.

### Step 2 — Phase A: Bidirectional Static Gravity Calibration (~22 s)
- **Procedure:** Instruct the subject to **completely relax** their forearm ("dead weight"). Press `Enter`. The exo slowly (`15 deg/s`) approaches `-25, 0, +25 deg` in the `+` direction and then `+25, 0, -25 deg` in the `-` direction (`1.5 s` hold each). Averaging the two directions cancels Coulomb stiction and reports the static stiction half-width.
- *Optional:* Pass `--skip-gravity-cal` if you only want Phase B + Phase C (Phase C relaxed chirp does **not** require Phase A for $J_{\text{human}}$ or cuff friction).

### Step 3 — Phase B: Active Human Swings in Zero-Torque Mode (30 s)
- **Procedure:** With the motor commanding `0.0 Nm`, press `Enter` to start recording. Have the subject smoothly flex and extend their elbow through a comfortable range of motion (staying clear of the hard stops), following the terminal pace cues:
  - `0–10 s`: **SLOW** (~0.3–0.5 Hz, $<60\ \text{deg/s}$)
  - `10–20 s`: **MEDIUM** (~0.8–1.2 Hz, $60\text{–}150\ \text{deg/s}$)
  - `20–30 s`: **FAST** (~1.5–2.5 Hz, $>150\ \text{deg/s}$)
- Press `Enter` at any time to finish early, or let the 30 s timer complete.

### Step 4 — Phase C: Relaxed Motor-Driven Chirp (Default, 30 s)
- **Why chirp instead of free decay:** On rigs with mechanical hard stops that block the vertical hanging angle (or with substantial gearbox dry friction), a released forearm cannot oscillate freely like a pendulum. A small-amplitude chirp stays centered at `0 deg` ($\pm 8^\circ$, $>50^\circ$ clear of stops), works in any rig orientation, and directly measures torque scale from motor current.
- **Procedure:** The exo moves gently to `0 deg` and holds. Instruct the subject to keep their arm **completely relaxed** ("dead weight" — do not resist or help the motion). Press `Enter` to run the `0.5 -> 6.0 Hz` swept-sine perturbation (`30 s`). Press `Enter` or `Ctrl-C` at any time to stop immediately.
- *Tuning flags:* `--chirp-amp 6.0 --chirp-duration 25` (or `--phase-c-mode decay` for legacy free-decay swings on rigs without hard-stop obstruction).

---

## 4. Data Collection & Offline Analysis

### Recorded Files (in `logs/`)
Each run writes:
- `passive_exo_<timestamp>_passive.csv` — per-tick 500 Hz dataset with columns:  
  `t, t_sample, fresh, phase, trial, angle_deg, velocity_deg_s, torque_nm, tau_cmd_nm, t_mos_c, t_rotor_c`
- `passive_exo_<timestamp>_meta.json` — rig label, subject ID, Phase C mode, CAN IDs, stop limit, baseline FRF path.
- `passive_exo_<timestamp>_summary.txt` & `_analysis.json` — auto-generated at the end of the run.
- `passive_exo_<timestamp>.csv` & `.log` — standard `Joint` tick log and diagnostic log.

### Running Offline Analysis & Generating Plots
```bash
python analyze_passive_exo.py \
  --csv logs/passive_exo_<timestamp>_passive.csv \
  --cal calibration/joint_cal.json \
  --frf logs/free_chirp_<timestamp>_frf.csv
```
Generates 4 PNG figures (`_01_time_series.png`, `_02_transparency.png`, `_03_passive_id_fit.png`, `_04_gravity_cal.png`). Optional flags:
- `--exo-inertia 0.0021` — manually set $J_{\text{exo}}$ ($\text{kg}\cdot\text{m}^2$) if no `_frf.csv` is available.
- `--stop-limit-deg 60.8 --stop-margin-deg 5.0` — mask samples within `stop-margin-deg` of mechanical stops during Phase C analysis.
- `--loadcell-csv path/to/loadcell.csv` — cross-validate $\hat{\tau}_h(t)$ against an external cuff loadcell.
- `--out-dir results/qdd_S01` — write reports and PNG plots to a dedicated folder.

### Verifying the Pipeline Without Hardware
```bash
pixi run test-passive
# or: python tests/test_passive_pipeline.py
```
This simulates a noisy 500 Hz hard-stop-limited session (including high `0.46 Nm` exo stiction, bidirectional Phase A, Phase B active swings + loadcell, Phase C relaxed chirp, plus legacy free-decay) and verifies parameter recovery and all 4 PNG plots.

---

## 5. How to Interpret the Results

### A. Mechanical Transparency Metrics (Phase B — Active Swings)
| Metric | Physical Meaning | What Good (QDD 10:1) Looks Like vs. High-Ratio (UME 40:1) |
| :--- | :--- | :--- |
| **`interaction tau_h` (RMS & Peak, Nm)** | Total resistive torque $\hat{\tau}_h = J_{\text{exo}}\ddot{q} + \tau_{f,\text{exo}}(\dot{q}) + g_{\text{exo}}(q) - \tau_m$ the human must overcome to move the exo. | **Lower is better.** QDD 10:1 should show substantially lower RMS/peak $\hat{\tau}_h$, especially in the **medium** and **fast** speed bins where reflected inertia $J_a \propto N^2$ dominates. |
| **Breakdown: `inertia` vs. `friction` vs. `gravity`** | Shows which physical term dominates human effort. | High-ratio rigs are typically **inertia- and friction-dominated** at fast speeds; a lightweight QDD rig shifts resistance toward much smaller inertia and residual gravity. |
| **`motor 0-Nm ripple` (RMS & Peak, Nm)** | Unintended torque measured by the motor while commanding `0.0 Nm` (current-loop noise, back-EMF, cogging). | Should stay near the sensor noise floor ($< 0.03\text{–}0.05\ \text{Nm}$ RMS). |
| **`mechanical work` (J) & `mean power` (W)** | Total energy the human expended fighting the exoskeleton during the swing trial. | Lower energy per unit swing amplitude/speed indicates superior mechanical transparency. |

### B. Human Forearm Inertia & Skin-Cuff Interface Friction (Phase C — Relaxed Chirp)
| Parameter | Physical Meaning | Typical Range & Sanity Checks |
| :--- | :--- | :--- |
| **`J_human` ($\text{kg}\cdot\text{m}^2$)** | Estimated inertia of the subject's forearm + hand about the elbow axis ($J_{\text{total}} - J_{\text{exo}}$). | **Typical adult range:** $30\text{–}80 \times 10^{-3}\ \text{kg}\cdot\text{m}^2$ ($0.03\text{–}0.08\ \text{kg}\cdot\text{m}^2$). Should remain consistent for the same subject across both rigs. |
| **`coulomb_cuff` (Nm)** | Additional dry/hysteretic friction beyond the bare exo ($\tau_{c,\text{total}} - \tau_{c,\text{exo}}$). | Captures **skin-cuff strap friction/hysteresis plus passive elbow joint friction**. |
| **`viscous_cuff` ($\text{Nm}\cdot\text{s/rad}$)** | Additional velocity-dependent damping beyond the bare exo ($b_{\text{total}} - b_{\text{exo}}$). | Captures **skin-cuff viscoelastic damping plus passive elbow joint damping** (typically $0.01\text{–}0.15\ \text{Nm}\cdot\text{s/rad}$). |
| **`k_eff` ($\text{Nm/rad}$) & `tau_0` (Nm)** | Linearized gravity gradient + passive elbow stiffness and static gravity torque at `0 deg`. | Absorbs static gravity and passive joint stiffness so they do not bias $J_{\text{total}}$. |
| **Fit Quality (`R^2`, `residual RMS`, `cond(X)`)** | Goodness of fit of the relaxed rigid-body + friction model. | Target **$R^2 \ge 0.90$** and low `residual RMS` ($< 0.05\text{–}0.10\ \text{Nm}$). |

### C. Troubleshooting Common Experimental Artifacts
- **$R^2 < 0.85$ during Phase C relaxed chirp:** The subject likely tensed/guarded their biceps or triceps in response to the motor oscillation, or the cuff strap slipped on the forearm. Repeat Phase C with explicit coaching to stay completely limp ("dead weight") and tighten the cuff snugly.
- **Negative `coulomb_cuff` or `viscous_cuff`:** Either the subject unconsciously assisted the motion, or `calibration/joint_cal.json` was measured when the motor/gearbox was cold (higher friction) while Experiment 2 ran warm. Re-run `home-auto` at the same operating temperature.
- **Unexpectedly high `coulomb_exo` (e.g. $> 0.4\ \text{Nm}$ on a 10:1 QDD):** Backdrive the unpowered exo by hand to verify whether the drag is mechanical binding/misalignment or a stale homing calibration, then re-run `pixi run home-auto` and `pixi run chirp-free`.
- **Using legacy `--phase-c-mode decay`:** Only valid if your rig's vertical hanging angle sits comfortably inside the hard stops (with at least $\pm 35^\circ$ of free swing) and dry friction is small relative to gravity. Otherwise, use the default `--phase-c-mode chirp`.
