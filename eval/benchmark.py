"""Measure transcription accuracy against known ground truth.

Two stages, deliberately separated:

  stage A (transcribe) — run on audio rendered from one part alone, so no
      source-separation error is involved. Deterministic, so it is the stage
      to iterate against when tuning the transcriber or the quantiser.
  stage B (end-to-end)  — run the whole pipeline on the mix. Includes Demucs,
      which is not bit-reproducible on MPS, so treat small moves as noise.

Metric is mir_eval note onset+pitch F1: a note counts as found when its onset
lands within the tolerance and its pitch is right.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import librosa  # noqa: E402
import mir_eval  # noqa: E402

from pipeline import analyze as A  # noqa: E402
from pipeline import separate as SEP  # noqa: E402
from pipeline import transcribe as T  # noqa: E402
from pipeline.quantize import BeatGrid, clean_part, quantize_notes  # noqa: E402
import cases as C  # noqa: E402

ONSET_TOLERANCE = 0.075
CACHE = Path(__file__).resolve().parent / ".cache"

# ground-truth part -> (transcription profile, reduce to a single line?)
PROFILES = {
    "melody": ("vocals", True),
    "piano": ("accompaniment", False),
    "bass": ("bass", True),
}
# ground-truth part -> which separated stem carries it in the mix
STEM_FOR = {"melody": "vocals", "piano": "accompaniment", "bass": "bass"}


def _render(score, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = path.with_suffix(".musicxml")
    score.write("musicxml", fp=str(xml))
    subprocess.run([shutil.which("mscore"), "-o", str(path), str(xml)],
                   capture_output=True, timeout=600)
    xml.unlink(missing_ok=True)
    if not path.exists():
        raise RuntimeError(f"MuseScore 無法合成 {path.name}")
    return path


def lead_in(wav: Path) -> float:
    """Silence MuseScore places before the first note; ground truth shifts by it."""
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    return float(onsets[0]) if len(onsets) else 0.0


def truth_arrays(events, bpm: float, shift: float):
    spb = 60.0 / bpm
    intervals, pitches = [], []
    for off, dur, pitch in events:
        start = off * spb + shift
        intervals.append([start, start + dur * spb])
        pitches.append(librosa.midi_to_hz(pitch))
    order = np.argsort([i[0] for i in intervals])
    return np.array(intervals)[order], np.array(pitches)[order]


def est_from_notes(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n.start, max(n.end, n.start + 1e-3)] for n in notes])
    return iv, np.array([librosa.midi_to_hz(n.pitch) for n in notes])


def est_from_grid(grid_notes, grid: BeatGrid):
    if not grid_notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv, pitches = [], []
    for n in grid_notes:
        start = grid.to_seconds(n.offset)
        end = grid.to_seconds(n.offset + max(n.duration, 0.05))
        iv.append([max(0.0, start), max(start + 1e-3, end)])
        pitches.append(librosa.midi_to_hz(n.pitch))
    return np.array(iv), np.array(pitches)


def score_notes(ref_iv, ref_p, est_iv, est_p) -> dict:
    if len(ref_iv) == 0 or len(est_iv) == 0:
        return {"p": 0.0, "r": 0.0, "f1": 0.0,
                "n_ref": len(ref_iv), "n_est": len(est_iv)}
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=ONSET_TOLERANCE, pitch_tolerance=50.0, offset_ratio=None)
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3),
            "n_ref": len(ref_iv), "n_est": len(est_iv)}


def _refine(analysis, raw_by_part):
    """Mirror the pipeline: correct the beat grid using transcribed onsets."""
    return analysis, BeatGrid(analysis)


def stage_a(case: C.Case, workdir: Path, verbose: bool) -> dict:
    """Transcribe each part from its own isolated render.

    The beat grid still comes from the full mix, because that is what the
    pipeline does: tempo is estimated once and shared by every part.
    """
    mix = _render(case.score, workdir / f"{case.name}__mix.wav")
    analysis = A.analyze(mix)

    raw_by_part, refs = {}, {}
    for part, events in case.truth.items():
        if part not in PROFILES:
            continue
        isolated = C.isolate(case, C.PART_STAVES.get(part, []))
        if not isolated.parts:
            continue
        wav = _render(isolated, workdir / f"{case.name}__{part}.wav")
        profile, monophonic = PROFILES[part]
        notes = T.transcribe_pitched(wav, profile)
        if monophonic:
            notes = T.make_monophonic(notes)
        raw_by_part[part] = notes
        refs[part] = truth_arrays(events, case.tempo_bpm, lead_in(wav))

    analysis, grid = _refine(analysis, raw_by_part)

    out = {}
    for part, notes in raw_by_part.items():
        ref_iv, ref_p = refs[part]
        monophonic = PROFILES[part][1]
        raw = score_notes(ref_iv, ref_p, *est_from_notes(notes))
        gridded = clean_part(quantize_notes(notes, grid), fix_octaves=monophonic)
        final = score_notes(ref_iv, ref_p, *est_from_grid(gridded, grid))
        out[part] = {"raw": raw, "final": final,
                     "tempo_detected": round(analysis.tempo, 1),
                     "tempo_err": round(abs(analysis.tempo - case.tempo_bpm), 1)}
        if verbose:
            print(f"    A {part:8s} raw F1={raw['f1']:.3f}  final F1={final['f1']:.3f}"
                  f"   (tempo {analysis.tempo:.0f} vs {case.tempo_bpm:.0f})")
    return out


def stage_b(case: C.Case, workdir: Path, verbose: bool) -> dict:
    """Full pipeline on the mix, separation included."""
    wav = _render(case.score, workdir / f"{case.name}__mix.wav")
    shift = lead_in(wav)

    analysis = A.analyze(wav)
    stems = SEP.separate(wav, workdir / case.name)
    accomp = SEP.make_accompaniment(stems, workdir / case.name)
    if accomp:
        stems["accompaniment"] = accomp

    raw_by_part = {}
    for part in case.truth:
        if part not in PROFILES or STEM_FOR[part] not in stems:
            continue
        profile, monophonic = PROFILES[part]
        notes = T.transcribe_pitched(stems[STEM_FOR[part]], profile)
        if monophonic:
            notes = T.make_monophonic(notes)
        raw_by_part[part] = notes

    analysis, grid = _refine(analysis, raw_by_part)

    out = {
        "tempo_true": case.tempo_bpm,
        "tempo_detected": round(analysis.tempo, 1),
        "tempo_err": round(abs(analysis.tempo - case.tempo_bpm), 1),
        "key_true": case.key_name,
        "parts": {},
    }
    all_parts = {}
    for part, notes in raw_by_part.items():
        monophonic = PROFILES[part][1]
        ref_iv, ref_p = truth_arrays(case.truth[part], case.tempo_bpm, shift)
        gridded = clean_part(quantize_notes(notes, grid), fix_octaves=monophonic)
        all_parts[part] = gridded
        final = score_notes(ref_iv, ref_p, *est_from_grid(gridded, grid))
        out["parts"][part] = final
        if verbose:
            print(f"    B {part:8s} final F1={final['f1']:.3f}")

    refined = A.estimate_key_from_notes(all_parts, analysis.beats_per_bar)
    if refined and refined[2] > analysis.key_confidence:
        analysis.tonic, analysis.mode, analysis.key_confidence = refined
    out["key_detected"] = analysis.key_name
    out["key_ok"] = analysis.key_name == case.key_name
    if verbose:
        print(f"    B key      {out['key_detected']:<10s} "
              f"(true {case.key_name}) {'OK' if out['key_ok'] else 'X'}"
              f"   tempo {analysis.tempo:.0f} vs {case.tempo_bpm:.0f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--stage", choices=["a", "b", "both"], default="both")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    results, t0 = [], time.time()

    for factory in C.ALL_CASES:
        case = factory()
        print(f"\n  {case.name}")
        entry = {"case": case.name}
        if args.stage in ("a", "both"):
            entry["stage_a"] = stage_a(case, CACHE, True)
        if args.stage in ("b", "both"):
            entry["stage_b"] = stage_b(case, CACHE, True)
        results.append(entry)

    a_raw = [p["raw"]["f1"] for r in results for p in r.get("stage_a", {}).values()]
    a_fin = [p["final"]["f1"] for r in results for p in r.get("stage_a", {}).values()]
    b_fin = [p["f1"] for r in results for p in r.get("stage_b", {}).get("parts", {}).values()]
    keys = [r["stage_b"]["key_ok"] for r in results if "stage_b" in r]
    temps = [r["stage_b"]["tempo_err"] for r in results if "stage_b" in r]

    summary = {
        "A_raw_f1": round(float(np.mean(a_raw)), 3) if a_raw else None,
        "A_final_f1": round(float(np.mean(a_fin)), 3) if a_fin else None,
        "B_final_f1": round(float(np.mean(b_fin)), 3) if b_fin else None,
        "key_accuracy": round(float(np.mean(keys)), 3) if keys else None,
        "tempo_err": round(float(np.mean(temps)), 1) if temps else None,
        "seconds": round(time.time() - t0, 1),
    }

    print("\n" + "=" * 56)
    print(f"  {args.label}")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k:14s} {v}")
    print("=" * 56)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.label}.json").write_text(
        json.dumps({"label": args.label, "summary": summary, "cases": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
