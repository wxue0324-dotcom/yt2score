"""Where do quiet and short notes disappear? Recall per stage, per note kind.

The end-to-end F1 in `benchmark.py` averages over every note, so a draft that
reproduces the loud crotchets and loses every quiet semiquaver still scores
respectably. This walks the same notes through each stage of the pipeline in
turn and reports recall split by loudness and by note value, which is the only
way to see which stage is doing the discarding and to what.

    ../venv/bin/python note_loss.py
    ../venv/bin/python note_loss.py --subdivision 4
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np  # noqa: E402

from pipeline import analyze as A  # noqa: E402
from pipeline import transcribe as T  # noqa: E402
from pipeline import quantize as Q  # noqa: E402
import benchmark as B  # noqa: E402
import cases as C  # noqa: E402


def band_recall(ref_iv, ref_p, est_iv, est_p, bands, bpm, shift):
    """Recall restricted to the ground-truth notes inside each offset band."""
    spb = 60.0 / bpm
    out = {}
    for label, ranges in bands.items():
        keep = np.zeros(len(ref_iv), dtype=bool)
        for lo, hi in ranges:
            t0, t1 = lo * spb + shift, hi * spb + shift
            keep |= (ref_iv[:, 0] >= t0 - 1e-6) & (ref_iv[:, 0] < t1 - 1e-6)
        if not keep.any():
            out[label] = None
            continue
        s = B.score_notes(ref_iv[keep], ref_p[keep], est_iv, est_p)
        out[label] = s["r"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdivision", type=int, default=Q.SUBDIVISION)
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--true-tempo", action="store_true",
                    help="build the grid from the known tempo instead of the "
                         "detected one, to measure quantisation on its own")
    args = ap.parse_args()

    case = C.soft_and_short()
    iso = C.isolate(case, C.PART_STAVES["melody"])
    wav = B._render(iso, B.CACHE / f"{case.name}__melody.wav")
    shift = B.lead_in(wav)
    ref_iv, ref_p = B.truth_arrays(case.truth["melody"], case.tempo_bpm, shift)

    analysis = A.analyze(B._render(case.score, B.CACHE / f"{case.name}__mix.wav"))
    if args.true_tempo:
        # A rigid grid at the true tempo, phased to the first note. Tempo error
        # and grid coarseness both move onsets; measuring either one means
        # holding the other still.
        spb = 60.0 / case.tempo_bpm
        analysis.tempo = case.tempo_bpm
        analysis.beats = np.arange(shift, shift + spb * 400, spb)
    grid = Q.BeatGrid(analysis)
    print(f"\n{case.name}   真值 {len(ref_iv)} 音   "
          f"速度 {analysis.tempo:.1f} vs {case.tempo_bpm:.0f}   "
          f"格線 1/{args.subdivision} 拍\n")

    raw = T.transcribe_pitched(wav, "vocals")
    stages = [("basic-pitch 原始", raw, None)]
    mono = T.make_monophonic(raw)
    stages.append(("→ 單旋律化", mono, None))

    gridded = Q.quantize_notes(mono, grid, subdivision=args.subdivision)
    stages.append(("→ 量化到格線", gridded, grid))
    merged = Q.merge_repeats(list(gridded))
    stages.append(("→ 合併重疊同音", merged, grid))
    octaved = Q.fix_octave_jumps(list(merged))
    stages.append(("→ 修八度跳", octaved, grid))
    final = Q.drop_stray_notes(list(octaved), min_duration=args.min_duration)
    stages.append(("→ 丟短又弱的音", final, grid))

    labels = list(C.SOFT_SHORT_BANDS)
    print(f"{'階段':<18}{'音數':>5}{'精確':>7}{'召回':>7}{'F1':>7}   "
          + "".join(f"{l:>9}" for l in labels))
    print("-" * (44 + 9 * len(labels)))
    for name, notes, g in stages:
        est = B.est_from_grid(notes, g) if g is not None else B.est_from_notes(notes)
        s = B.score_notes(ref_iv, ref_p, *est)
        bands = band_recall(ref_iv, ref_p, *est, C.SOFT_SHORT_BANDS,
                            case.tempo_bpm, shift)
        cells = "".join(f"{bands[l]:>9.3f}" if bands[l] is not None else f"{'--':>9}"
                        for l in labels)
        print(f"{name:<18}{len(notes):>5}{s['p']:>7.3f}{s['r']:>7.3f}{s['f1']:>7.3f}   {cells}")
    print()


if __name__ == "__main__":
    main()
