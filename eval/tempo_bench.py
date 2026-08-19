"""Measure tempo and metre detection against pieces with an exact known tempo.

Absolute BPM error is the wrong unit for this: 60 detected as 120 and 60
detected as 119 are both "59 off", but the first writes every duration at half
value and the second is a piece played a shade fast. So results are classified
by the *ratio* to the truth — 2×, ½×, 3× — and only a ratio near 1 counts as
correct.

    ../venv/bin/python tempo_bench.py --label baseline
    ../venv/bin/python tempo_bench.py --label myfix --compare baseline
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np  # noqa: E402

from pipeline import analyze as A  # noqa: E402
import tempo_cases as TC  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache" / "tempo"
RESULTS = Path(__file__).resolve().parent / "results" / "tempo"

# Ratios worth naming. A detected/true ratio within TOLERANCE of one of these
# is that error; anything else is just "off".
RATIOS = [(1.0, "ok"), (2.0, "2x"), (0.5, "half"), (3.0, "3x"),
          (1 / 3, "third"), (4.0, "4x"), (0.25, "quarter"),
          (1.5, "3:2"), (2 / 3, "2:3")]
TOLERANCE = 0.04          # ±4%: a real performance drifts more than this


def classify(detected: float, truth: float) -> tuple[str, float]:
    ratio = detected / truth
    for value, name in RATIOS:
        if abs(ratio / value - 1.0) <= TOLERANCE:
            return name, ratio
    return "off", ratio


def render(case, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = path.with_suffix(".musicxml")
    case.score.write("musicxml", fp=str(xml))
    mscore = shutil.which("mscore") or shutil.which("musescore")
    if not mscore:
        raise RuntimeError("找不到 MuseScore,無法合成測試音訊")
    subprocess.run([mscore, "-o", str(path), str(xml)], capture_output=True, timeout=600)
    xml.unlink(missing_ok=True)
    if not path.exists():
        raise RuntimeError(f"MuseScore 無法合成 {path.name}")
    return path


def run(only: str | None = None) -> list[dict]:
    rows = []
    for factory in TC.TEMPO_CASES:
        case = factory()
        if only and only not in case.name:
            continue
        wav = render(case, CACHE / f"{case.name}.wav")
        t0 = time.time()
        analysis = A.analyze(wav)
        elapsed = time.time() - t0
        verdict, ratio = classify(analysis.tempo, case.tempo_bpm)
        bpb_true = TC.BEATS_PER_BAR[case.name]
        # The reported tempo and the beat grid have to describe the same piece.
        # They are produced separately, and when they drifted apart the score
        # header said one thing while every duration was quantised against
        # another — silently, since each looked reasonable on its own.
        beats = np.asarray(analysis.beats, dtype=float)
        grid_bpm = (60.0 / float(np.median(np.diff(beats)))
                    if len(beats) > 2 else 0.0)
        grid_ok = bool(grid_bpm and abs(grid_bpm / analysis.tempo - 1) < 0.03)
        rows.append({
            "case": case.name,
            "tempo_true": case.tempo_bpm,
            "tempo_detected": round(float(analysis.tempo), 2),
            "ratio": round(ratio, 3),
            "verdict": verdict,
            "bpb_true": bpb_true,
            "bpb_detected": int(analysis.beats_per_bar),
            "bpb_ok": int(analysis.beats_per_bar) == bpb_true,
            "grid_bpm": round(grid_bpm, 2),
            "grid_ok": grid_ok,
            "seconds": round(elapsed, 1),
        })
        r = rows[-1]
        mark = "✓" if verdict == "ok" else "✗"
        print(f"  {mark} {case.name:16} {r['tempo_detected']:>7.2f} vs "
              f"{case.tempo_bpm:>6.1f}  ({verdict:>7}, ×{r['ratio']:.3f})   "
              f"metre {r['bpb_detected']}/4 vs {bpb_true}/4"
              f"{'' if r['bpb_ok'] else '  ✗'}"
              f"{'' if grid_ok else f'   grid {grid_bpm:.1f} ✗'}   {r['seconds']}s")
    return rows


def summarise(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "tempo_ok": sum(r["verdict"] == "ok" for r in rows),
        "metre_ok": sum(r["bpb_ok"] for r in rows),
        "grid_ok": sum(r["grid_ok"] for r in rows),
        "errors": sorted({r["verdict"] for r in rows if r["verdict"] != "ok"}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--only", default=None, help="substring of a case name")
    ap.add_argument("--compare", default=None, help="label of an earlier run")
    args = ap.parse_args()

    print(f"\ntempo bench — {args.label}\n")
    rows = run(args.only)
    s = summarise(rows)
    print(f"\n  tempo {s['tempo_ok']}/{s['cases']}   metre {s['metre_ok']}/{s['cases']}"
          f"   grid agrees {s['grid_ok']}/{s['cases']}"
          f"   errors: {', '.join(s['errors']) or 'none'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.label}.json").write_text(
        json.dumps({"label": args.label, "summary": s, "cases": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → results/tempo/{args.label}.json")

    if args.compare:
        prior_path = RESULTS / f"{args.compare}.json"
        if not prior_path.exists():
            print(f"\n  找不到 {args.compare}.json,略過對照")
            return
        prior = {c["case"]: c for c in json.loads(prior_path.read_text())["cases"]}
        print(f"\n  vs {args.compare}:")
        for r in rows:
            was = prior.get(r["case"])
            if not was:
                continue
            if was["verdict"] == r["verdict"] and was["bpb_ok"] == r["bpb_ok"]:
                continue
            print(f"    {r['case']:16} {was['tempo_detected']:>7.2f} ({was['verdict']})"
                  f"  →  {r['tempo_detected']:>7.2f} ({r['verdict']})")


if __name__ == "__main__":
    main()
