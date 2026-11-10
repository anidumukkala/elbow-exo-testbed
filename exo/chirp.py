"""Chirp test pieces: the swept-sine reference and the frequency-response analysis.

Reference: a log sweep f0 -> f1 (frequency rises exponentially with time) of the joint angle,
    angle(t) = A(f) * fade(t) * sin(phase(t))
with a smooth fade in (FADE_IN_S) and out (FADE_OUT_S). The amplitude falls at high frequency so
that the speed and acceleration stay under their limits:
    A(f) = min(amplitude, max_speed / (2 pi f), max_accel / (2 pi f)^2)

Analysis (lock-in): around each analysed frequency f, a signal y(t) is fitted over a few periods with
    y = a cos(phase) + b sin(phase) + offset + drift
using the sweep's exact phase. Y = a - j b is its phasor (amplitude |Y|, phase angle(Y)); the ratio
of two signals' phasors is their frequency response. Each sample keeps its own measurement time, so
nothing needs resampling. Results:
    tracking    measured angle / reference        how well position control follows (1 / 0 deg ideal)
    impedance   torque / angle (rad)              free motion: Re = -J w^2 gives the inertia J,
                                                  Im / w the equivalent damping c
"""

import math
from dataclasses import dataclass

import numpy as np

from .logs import write_csv

FADE_IN_S, FADE_OUT_S = 2.0, 1.0
ANALYSIS_CYCLES = 3  # lock-in window per analysed frequency, in periods

# One row per control tick of the sweep (the script appends them in this order).
COLUMNS = ["t", "t_sample", "fresh", "target_deg", "ref_deg", "angle_deg", "velocity_deg_s", "torque_nm"]
#  t           loop time [s]
#  t_sample    when the reply was measured [s, same clock]
#  fresh       1 if a new reply arrived this tick
#  target_deg  the target sent (half a tick ahead)
#  ref_deg     the reference at t_sample (what the joint should have been at when measured)


@dataclass(frozen=True)
class Chirp:
    f0_hz: float
    f1_hz: float
    duration_s: float
    amplitude_deg: float
    max_speed_deg_s: float
    max_accel_deg_s2: float

    @property
    def rate(self) -> float:  # d ln(f) / dt
        return math.log(self.f1_hz / self.f0_hz) / self.duration_s

    def freq(self, t: float) -> float:
        return self.f0_hz * math.exp(self.rate * min(max(t, 0.0), self.duration_s))

    def time_of(self, f: float) -> float:
        return math.log(f / self.f0_hz) / self.rate

    def phase(self, t):
        """Sweep phase [rad]; works on numpy arrays too."""
        return 2 * math.pi * self.f0_hz / self.rate * (np.exp(self.rate * np.asarray(t, dtype=float)) - 1.0)

    def envelope(self, t: float) -> float:
        """0 -> 1 over FADE_IN_S, 1 -> 0 over the last FADE_OUT_S (raised cosine), 0 outside the sweep."""
        if t <= 0.0 or t >= self.duration_s:
            return 0.0
        fade = min(1.0, t / FADE_IN_S, (self.duration_s - t) / FADE_OUT_S)
        return 0.5 - 0.5 * math.cos(math.pi * fade)

    def amp(self, f: float) -> float:
        """Amplitude at frequency f [deg], limited by speed and acceleration."""
        w = 2 * math.pi * f
        return min(self.amplitude_deg, self.max_speed_deg_s / w, self.max_accel_deg_s2 / w ** 2)

    def position(self, t: float) -> float:
        return self.amp(self.freq(t)) * self.envelope(t) * math.sin(float(self.phase(t)))

    def reference(self, t: float) -> tuple[float, float]:
        """(angle deg, velocity deg/s); velocity by central difference."""
        h = 1e-4
        return self.position(t), (self.position(t + h) - self.position(t - h)) / (2 * h)


def lockin(t: np.ndarray, y: np.ndarray, phase, t_c: float, half: float) -> complex:
    """Phasor of y around t_c from the samples with |t - t_c| <= half (nan if too few)."""
    keep = np.abs(t - t_c) <= half
    if keep.sum() < 20:
        return complex(np.nan, np.nan)
    tt, yy = t[keep], y[keep]
    ph = phase(tt)
    X = np.column_stack([np.cos(ph), np.sin(ph), np.ones_like(tt), tt - t_c])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return complex(coef[0], -coef[1])


def analysis_points(chirp: Chirp, points: int) -> list[tuple[float, float, float]]:
    """(f, window centre, half window) at up to `points` log-spaced frequencies whose window lies
    inside the sweep, fades excluded."""
    rows = []
    for f in np.geomspace(chirp.freq(FADE_IN_S) * 1.05, chirp.f1_hz, points * 3):
        half = ANALYSIS_CYCLES / f / 2
        t_c = chirp.time_of(f)
        if FADE_IN_S <= t_c - half and t_c + half <= chirp.duration_s - FADE_OUT_S:
            rows.append((float(f), t_c, half))
    if len(rows) > points:  # thin out, keeping both ends
        rows = [rows[i] for i in np.unique(np.linspace(0, len(rows) - 1, points).round().astype(int))]
    return rows


FRF_COLUMNS = ["f_hz", "amp_ref_deg", "track_gain", "track_phase_deg", "torque_amp_nm", "inertia_kg_m2",
               "damping_nm_s_rad"]


def analyse(rows: list, chirp: Chirp, points: int) -> list[dict]:
    """Frequency responses of the sweep (rows as in COLUMNS). Phase < 0 means lagging."""
    data = np.array(rows, dtype=float)
    data = data[data[:, COLUMNS.index("fresh")] == 1]
    col = {name: data[:, i] for i, name in enumerate(COLUMNS)}
    t = col["t_sample"]
    out = []
    for f, t_c, half in analysis_points(chirp, points):
        w = 2 * math.pi * f
        ref = lockin(t, col["ref_deg"], chirp.phase, t_c, half)
        ang = lockin(t, col["angle_deg"], chirp.phase, t_c, half)
        tau = lockin(t, col["torque_nm"], chirp.phase, t_c, half)
        if np.isnan(ref) or abs(ref) == 0:  # not enough samples there (aborted sweep)
            continue
        track = ang / ref
        z = tau / (ang * math.pi / 180)  # Nm per rad
        out.append(dict(f_hz=f, amp_ref_deg=abs(ref), track_gain=abs(track),
                        track_phase_deg=math.degrees(np.angle(track)), torque_amp_nm=abs(tau),
                        inertia_kg_m2=-z.real / w ** 2, damping_nm_s_rad=z.imag / w))
    return out


def summarize(rows: list, frf: list[dict], chirp: Chirp, settings: str) -> str:
    data = np.array(rows, dtype=float)
    t, fresh = data[:, COLUMNS.index("t")], data[:, COLUMNS.index("fresh")]
    tick_ms = np.diff(t) * 1e3
    lines = [f"  settings: {settings}",
             f"  loop: {len(t)} ticks, tick median {np.median(tick_ms):.2f} ms, max {tick_ms.max():.2f} ms; "
             f"missed replies {int((fresh == 0).sum())} ({100 * (fresh == 0).mean():.2f} %)"]
    good = data[fresh == 1]
    err = good[:, COLUMNS.index("angle_deg")] - good[:, COLUMNS.index("ref_deg")]
    lines.append(f"  peaks: |torque| {np.abs(good[:, COLUMNS.index('torque_nm')]).max():.3f} Nm, "
                 f"|angle| {np.abs(good[:, COLUMNS.index('angle_deg')]).max():.2f} deg, "
                 f"|angle - reference| {np.abs(err).max():.2f} deg")
    if not frf:
        return "\n".join(lines + ["  (no responses: dry run, or the sweep was too short to analyse)"])
    lines += ["  responses (track = measured / reference angle, phase < 0 = lags; J = inertia, c = damping):",
              f"  {'f Hz':>7}{'A deg':>8}{'track':>8}{'ph deg':>8}{'tau Nm':>8}{'J e-3':>9}{'c':>8}"]
    for r in frf:
        lines.append(f"  {r['f_hz']:7.2f}{r['amp_ref_deg']:8.3f}{r['track_gain']:8.3f}{r['track_phase_deg']:8.1f}"
                     f"{r['torque_amp_nm']:8.3f}{r['inertia_kg_m2'] * 1e3:9.3f}{r['damping_nm_s_rad']:8.4f}")
    # Inertia from the higher frequencies, where it dominates the torque: Re(Z) = -J w^2 + k0
    fast = [r for r in frf if r["f_hz"] >= 2.0]
    if len(fast) >= 3:
        w2 = np.array([(2 * math.pi * r["f_hz"]) ** 2 for r in fast])
        re_z = np.array([-r["inertia_kg_m2"] for r in fast]) * w2
        A = np.column_stack([-w2, np.ones_like(w2)])
        (J, k0), *_ = np.linalg.lstsq(A, re_z, rcond=None)
        lines.append(f"  inertia fit over {fast[0]['f_hz']:.1f}-{fast[-1]['f_hz']:.1f} Hz: J = {J * 1e3:.3f} e-3 kg m^2 "
                     f"(link + motor), k0 = {k0:+.2f} Nm/rad (near 0 if gravity is compensated)")
        if abs(k0) > 0.5:
            lines.append("  NOTE: k0 far from 0: where the amplitude is small (high frequencies), friction acts like a "
                         "spring, which biases this fit; trust the J column where the amplitude is larger, or rerun "
                         "with a larger amplitude / acceleration limit")
    return "\n".join(lines)


def save_results(stem, rows: list, frf: list[dict], summary: str) -> None:
    """<stem>_chirp.csv (the sweep, per tick), <stem>_frf.csv (responses), <stem>_summary.txt."""
    write_csv(f"{stem}_chirp.csv", COLUMNS, rows)
    if frf:
        write_csv(f"{stem}_frf.csv", FRF_COLUMNS, [[r[c] for c in FRF_COLUMNS] for r in frf])
    with open(f"{stem}_summary.txt", "w") as f:
        f.write(summary + "\n")
