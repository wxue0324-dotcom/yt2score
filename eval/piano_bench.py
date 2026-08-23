"""Piano accuracy, broken down far enough to act on.

`benchmark.py` gives the piano one F1 per case. That number cannot distinguish
a draft that reproduces the right hand and loses the whole bass from one that
is evenly mediocre, and those call for opposite fixes — so this splits recall
by register, and reports the transcription and the hand assignment separately.

    ../venv/bin/python piano_bench.py
    ../venv/bin/python piano_bench.py --label mine --compare base
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np  # noqa: E402

from pipeline import analyze as A  # noqa: E402
from pipeline import separate as SEP  # noqa: E402
from pipeline import transcribe as T  # noqa: E402
from pipeline import quantize as Q  # noqa: E402
import benchmark as B  # noqa: E402
import cases as C  # noqa: E402
import hand_split as H  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
# Registers, by where a pianist's hands actually live rather than by even thirds.
BANDS = [("低音 <52", 0, 52), ("中音 52-71", 52, 72), ("高音 ≥72", 72, 128)]

PIANO_CASES = ("solo_piano", "full_band", "roaming_piano")


def transcribe_piano(case: C.Case, stage: str = "a"):
    """Everything the pipeline does to a piano part, up to the split.

    Stage a reads an isolated render of the piano staves; stage b reads what
    Demucs actually hands the transcriber — bass+other mixed back together.
    Stage a is deterministic and fast enough to iterate against, but it is not
    the input the pipeline sees, and separation artefacts are exactly where the
    bass register gets lost, so a change has to be confirmed on b.
    """
    mix = B._render(case.score, B.CACHE / f"{case.name}__mix.wav")
    stems, present = {}, set()
    if stage == "a":
        source = B._render(C.isolate(case, C.PART_STAVES["piano"]),
                           B.CACHE / f"{case.name}__piano.wav")
        shift = B.lead_in(source)
    else:
        stems = SEP.separate(mix, B.CACHE / case.name)
        present = SEP.present_stems(SEP.stem_levels(mix, stems))
        source = SEP.make_accompaniment(stems, B.CACHE / case.name)
        shift = B.lead_in(mix)
    analysis = A.analyze(mix)
    grid = Q.BeatGrid(analysis)
    raw = T.transcribe_pitched(source, "accompaniment", tempo=analysis.tempo)
    gridded = Q.quantize_notes(raw, grid)
    if stage == "b" and "bass" in present:
        bass = T.make_monophonic(T.transcribe_pitched(stems["bass"], "bass"))
        gridded = Q.drop_notes_shared_with(
            gridded, Q.clean_part(Q.quantize_notes(bass, grid), fix_octaves=True))
    return raw, gridded, grid, shift


def band_scores(ref_iv, ref_p, est_iv, est_p):
    """Recall inside each register, plus the overall figures."""
    out = {}
    midi = np.round(69 + 12 * np.log2(np.asarray(ref_p) / 440.0)).astype(int)
    for label, lo, hi in BANDS:
        keep = (midi >= lo) & (midi < hi)
        if not keep.any():
            out[label] = None
            continue
        out[label] = B.score_notes(ref_iv[keep], ref_p[keep], est_iv, est_p)["r"]
    return out


def run(stage: str = "a", verbose: bool = True) -> list[dict]:
    rows = []
    for factory in C.ALL_CASES:
        case = factory()
        if case.name not in PIANO_CASES or "piano" not in case.truth:
            continue
        t0 = time.time()
        raw, gridded, grid, shift = transcribe_piano(case, stage)
        # Score what the pipeline actually writes: split into hands and cleaned,
        # not the raw quantised list. The tidy-up is not free either way and
        # measuring before it means measuring something no user receives.
        rh, lh = Q.split_hands(gridded)
        final = sorted(Q.clean_part(list(rh)) + Q.clean_part(list(lh), min_duration=1.0),
                       key=lambda n: (n.offset, n.pitch))
        ref_iv, ref_p = B.truth_arrays(case.truth["piano"], case.tempo_bpm, shift)
        est = B.est_from_grid(final, grid)
        overall = B.score_notes(ref_iv, ref_p, *est)
        bands = band_scores(ref_iv, ref_p, *est)
        gridded = final
        hands = H.evaluate(case, verbose=False) if stage == "a" else None
        rows.append({
            "case": case.name, "n_true": len(ref_iv), "n_got": len(gridded),
            "p": overall["p"], "r": overall["r"], "f1": overall["f1"],
            "bands": bands,
            "hand_accuracy": hands["accuracy"] if hands else None,
            "seconds": round(time.time() - t0, 1),
        })
        if verbose:
            r = rows[-1]
            cells = "  ".join(
                f"{label.split()[0]} {r['bands'][label]:.3f}" if r['bands'][label] is not None
                else f"{label.split()[0]}   -  " for label, _, _ in BANDS)
            hand = f"{r['hand_accuracy']:.3f}" if r["hand_accuracy"] is not None else "  -  "
            print(f"  {case.name:<15} 真{r['n_true']:>3} 得{r['n_got']:>4}   "
                  f"P {r['p']:.3f}  R {r['r']:.3f}  F1 {r['f1']:.3f}   "
                  f"│ {cells} │ 分手 {hand}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--stage", choices=["a", "b"], default="a")
    args = ap.parse_args()

    print(f"\n鋼琴精準度 — {args.label} (stage {args.stage})\n")
    rows = run(args.stage)
    if not rows:
        sys.exit("沒有鋼琴案例")
    mean_f1 = float(np.mean([r["f1"] for r in rows]))
    mean_hand = float(np.mean([r["hand_accuracy"] for r in rows
                               if r["hand_accuracy"] is not None]))
    low = [r["bands"]["低音 <52"] for r in rows if r["bands"]["低音 <52"] is not None]
    print(f"\n  平均 F1 {mean_f1:.3f}   平均分手 {mean_hand:.3f}"
          f"   低音召回 {np.mean(low):.3f}" if low else "")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"piano_{args.stage}_{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "mean_f1": round(mean_f1, 3),
                               "mean_hand": round(mean_hand, 3), "cases": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → results/{out.name}")

    if args.compare:
        prior = RESULTS / f"piano_{args.stage}_{args.compare}.json"
        if not prior.exists():
            print(f"  找不到 {prior.name}")
            return
        was = {c["case"]: c for c in json.loads(prior.read_text())["cases"]}
        print(f"\n  vs {args.compare}:")
        for r in rows:
            w = was.get(r["case"])
            if not w:
                continue
            d = r["f1"] - w["f1"]
            print(f"    {r['case']:<15} F1 {w['f1']:.3f} → {r['f1']:.3f} ({d:+.3f})"
                  f"   分手 {w['hand_accuracy']} → {r['hand_accuracy']}")


if __name__ == "__main__":
    main()
