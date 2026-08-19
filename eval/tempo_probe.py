"""Diagnostic: show what the tempo search actually considers, and how it scores.

Prints, for one audio file, every candidate the search generates, the tempo
beat_track actually returns for it, and the alignment score. Use it to see
whether candidate generation is doing any work or whether the tracker collapses
every seed onto the same answer.

    ../venv/bin/python tempo_probe.py ../work/<id>/source.wav [--truth 60]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from pipeline.analyze import _beat_alignment, _fold_tempo, _TEMPO_MIN, _TEMPO_MAX  # noqa: E402

HOP = 512


def probe(path: Path, truth: float | None) -> None:
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    try:
        y_perc = librosa.effects.percussive(y, margin=2.0)
        onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP)
    except Exception:
        onset_env = np.array([])
    plain_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    if onset_env.size == 0 or onset_env.mean() < plain_env.mean() * 0.2:
        onset_env = plain_env

    onset_times = librosa.onset.onset_detect(
        onset_envelope=plain_env, sr=sr, hop_length=HOP, units="time", backtrack=True)

    seeds = list(np.atleast_1d(librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, aggregate=np.median)))
    seeds += list(np.atleast_1d(librosa.feature.tempo(
        onset_envelope=plain_env, sr=sr, hop_length=HOP, aggregate=np.median)))
    print(f"file        {path}")
    print(f"duration    {librosa.get_duration(y=y, sr=sr):.1f}s   onsets={len(onset_times)}")
    print(f"seeds       {[round(float(s), 2) for s in seeds]}")

    cands = sorted({round(float(s) * f, 2) for s in seeds + [90.0, 120.0]
                    for f in (0.25, 0.5, 1.0, 2.0, 4.0)
                    if _TEMPO_MIN <= float(s) * f <= _TEMPO_MAX})
    print(f"candidates  {cands}\n")

    print(f"{'cand':>7} {'env':>6} {'tracked':>8} {'score':>7}   {'rigid grid':>10}")
    print("-" * 50)
    rows = []
    for c in cands:
        for label, env in (("perc", onset_env), ("plain", plain_env)):
            tempo, frames = librosa.beat.beat_track(
                onset_envelope=env, sr=sr, hop_length=HOP, start_bpm=c, trim=False)
            tempo = _fold_tempo(float(np.atleast_1d(tempo)[0]))
            beats = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
            score = _beat_alignment(onset_times, beats)
            rigid = best_rigid(onset_times, c, y, sr)
            rows.append((score, c, label, tempo))
            print(f"{c:>7.2f} {label:>6} {tempo:>8.2f} {score:>7.3f}   {rigid:>10.3f}")

    rows.sort(reverse=True)
    print(f"\nwinner      {rows[0][1]:.2f} seed -> {rows[0][3]:.2f} bpm  (score {rows[0][0]:.3f})")
    tracked = sorted({round(r[3], 1) for r in rows})
    print(f"distinct tracked tempos: {tracked}")
    if truth:
        print(f"\nground truth {truth:.2f}")
        for mult, name in ((0.5, "half"), (1.0, "true"), (1.5, "3:2"), (2.0, "double"), (3.0, "triple")):
            t = truth * mult
            if _TEMPO_MIN <= t <= _TEMPO_MAX:
                print(f"  {name:>6} {t:>7.2f}  rigid-grid score {best_rigid(onset_times, t, y, sr):.3f}")


def best_rigid(onset_times: np.ndarray, bpm: float, y: np.ndarray, sr: int) -> float:
    """Score a perfectly steady grid at this bpm, over all phases."""
    if len(onset_times) == 0:
        return 0.0
    period = 60.0 / bpm
    duration = librosa.get_duration(y=y, sr=sr)
    best = 0.0
    for phase in np.arange(0, period, period / 16):
        grid = np.arange(phase, duration, period)
        best = max(best, _beat_alignment(onset_times, grid))
    return best


def hierarchy(path: Path) -> None:
    """Show the metrical hierarchy: is a grid too fine, or right?

    At the notated beat, grid points sit on accented onsets and the midpoints
    between them are weaker. At twice the notated beat, every other grid point
    lands on an off-beat and the accent contrast collapses. Printing both the
    onset coverage and that contrast separates the two.
    """
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=HOP, units="time", backtrack=True)
    duration = librosa.get_duration(y=y, sr=sr)
    ioi = np.diff(onset_times)
    print(f"file      {path}")
    print(f"onsets    {len(onset_times)} over {duration:.1f}s")
    if len(ioi):
        print(f"IOI       median {np.median(ioi):.3f}s  "
              f"({60 / np.median(ioi):.1f} onsets/min)  "
              f"p10 {np.percentile(ioi, 10):.3f}  p90 {np.percentile(ioi, 90):.3f}")
    print()
    print(f"{'bpm':>7} {'F1':>6} {'strength@beat':>14} {'@midpoint':>10} {'contrast':>9}"
          f" {'odd/even':>9}")
    print("-" * 62)
    for bpm in (40, 45, 50, 54, 60, 69, 77, 80, 92, 103, 108, 120, 132, 154, 161, 172, 185):
        period = 60.0 / bpm
        best, grid = -1.0, None
        for phase in np.arange(0, period, period / 16):
            g = np.arange(phase, duration, period)
            f1 = _beat_alignment(onset_times, g)
            if f1 > best:
                best, grid = f1, g
        fr = np.clip(librosa.time_to_frames(grid, sr=sr, hop_length=HOP), 0, len(env) - 1)
        mid = np.clip(librosa.time_to_frames(grid + period / 2, sr=sr, hop_length=HOP),
                      0, len(env) - 1)
        s_beat, s_mid = env[fr].mean(), env[mid].mean()
        odd_even = env[fr][::2].mean() / (env[fr][1::2].mean() + 1e-9)
        print(f"{bpm:>7} {best:>6.3f} {s_beat:>14.3f} {s_mid:>10.3f} "
              f"{s_beat / (s_mid + 1e-9):>9.3f} {odd_even:>9.3f}")


def sweep(path: Path) -> None:
    """Score a steady grid at every tempo — what the scoring function *could* pick.

    Separates two different failures: the search never considering the right
    tempo, versus the scoring function preferring the wrong one.
    """
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    plain_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=plain_env, sr=sr, hop_length=HOP, units="time", backtrack=True)
    print(f"file      {path}")
    print(f"onsets    {len(onset_times)} over {librosa.get_duration(y=y, sr=sr):.1f}s\n")
    scored = [(best_rigid(onset_times, bpm, y, sr), bpm)
              for bpm in np.arange(_TEMPO_MIN, _TEMPO_MAX + 0.01, 0.5)]
    scored.sort(reverse=True)
    print("best steady grids:")
    for score, bpm in scored[:12]:
        print(f"  {bpm:>7.1f}  {score:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--truth", type=float, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="score steady grids across the whole tempo range")
    ap.add_argument("--hierarchy", action="store_true",
                    help="show accent contrast per tempo level")
    a = ap.parse_args()
    if a.hierarchy:
        hierarchy(Path(a.audio))
    elif a.sweep:
        sweep(Path(a.audio))
    else:
        probe(Path(a.audio), a.truth)
