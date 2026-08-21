"""How often does each note end up in the hand that actually played it?

`benchmark.py` scores the piano as one part, so every note counts as correct
wherever it lands — the staff it was written on is invisible to the metric.
That is the one thing a pianist reads first, and the case data has always
carried the answer (`solo_piano` knows which notes were right hand and which
were left), so this measures against it directly.

Reported separately from note accuracy on purpose: a split can only be judged
on notes that were transcribed at all, and mixing the two hides which one moved.

    ../venv/bin/python hand_split.py
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

TOLERANCE = 0.12          # seconds; matching a transcribed note to a true one


def truth_hand_map(case: C.Case, shift: float):
    """(time, pitch) -> "R" / "L", from the case's own two-staff ground truth."""
    spb = 60.0 / case.tempo_bpm
    out = []
    for hand, key in (("R", "piano_rh"), ("L", "piano_lh")):
        for off, _dur, pitch in case.truth.get(key, []):
            out.append((off * spb + shift, int(pitch), hand))
    return out


def evaluate(case: C.Case, verbose: bool = True) -> dict | None:
    if "piano_rh" not in case.truth or "piano_lh" not in case.truth:
        return None
    iso = C.isolate(case, C.PART_STAVES["piano"])
    wav = B._render(iso, B.CACHE / f"{case.name}__piano.wav")
    shift = B.lead_in(wav)
    truth = truth_hand_map(case, shift)

    analysis = A.analyze(B._render(case.score, B.CACHE / f"{case.name}__mix.wav"))
    grid = Q.BeatGrid(analysis)
    notes = T.transcribe_pitched(wav, "accompaniment")
    gridded = Q.quantize_notes(notes, grid)
    right, left = Q.split_hands(gridded)

    scored = [(n, "R") for n in right] + [(n, "L") for n in left]
    hits = misses = 0
    per_hand = {"R": [0, 0], "L": [0, 0]}      # [correct, total matched]
    for note, hand in scored:
        seconds = grid.to_seconds(note.offset)
        best, best_gap = None, TOLERANCE
        for t, pitch, true_hand in truth:
            if pitch != note.pitch:
                continue
            gap = abs(t - seconds)
            if gap < best_gap:
                best, best_gap = true_hand, gap
        if best is None:
            continue                            # a note the truth does not have
        per_hand[best][1] += 1
        if best == hand:
            hits += 1
            per_hand[best][0] += 1
        else:
            misses += 1

    matched = hits + misses
    if not matched:
        return None
    result = {
        "case": case.name,
        "matched": matched,
        "accuracy": round(hits / matched, 3),
        "right_recall": round(per_hand["R"][0] / per_hand["R"][1], 3) if per_hand["R"][1] else None,
        "left_recall": round(per_hand["L"][0] / per_hand["L"][1], 3) if per_hand["L"][1] else None,
        "n_right": len(right),
        "n_left": len(left),
    }
    if verbose:
        r = result
        print(f"  {case.name:<16} 對到 {matched:>3} 音   正確率 {r['accuracy']:.3f}"
              f"   右手 {r['right_recall']}   左手 {r['left_recall']}"
              f"   （分出 右{r['n_right']} 左{r['n_left']}）")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.parse_args()
    print("\n分手正確率（只計算能對上標準答案的音）\n")
    any_run = False
    for factory in C.ALL_CASES:
        if evaluate(factory()):
            any_run = True
    if not any_run:
        print("  沒有帶左右手標準答案的案例")
    print()


if __name__ == "__main__":
    main()
