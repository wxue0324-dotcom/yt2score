"""Measure tempo detection against real recordings whose tempo a listener settled.

The synthetic cases in `tempo_bench.py` have an exact tempo by construction, but
they are MuseScore renderings — no rubato, no mix, no separation artefacts. This
runs the same measurement against `truth.json`, where every answer came from a
person listening to a metronome mixed into the actual recording.

Ratio, not absolute error, is the unit: 55 detected as 112 is not "57 bpm off",
it is every note length doubled. See `tempo_bench.classify`.

    ../venv/bin/python real_bench.py --label prior
    ../venv/bin/python real_bench.py --label myfix --compare prior
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np  # noqa: E402

from pipeline import analyze as A  # noqa: E402
from tempo_bench import classify  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work"
RESULTS = HERE / "results" / "tempo"
TRUTH = HERE / "truth.json"


def load_cases(tag: str | None = None) -> list[dict]:
    data = json.loads(TRUTH.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if c.get("true_tempo")]
    if tag:
        cases = [c for c in cases if tag in c.get("tags", [])]
    return cases


def run(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        wav = WORK / case["id"] / "source.wav"
        if not wav.exists():
            print(f"  – {case['title'][:34]:34} 找不到 {wav}，略過")
            continue
        t0 = time.time()
        analysis = A.analyze(wav)
        elapsed = time.time() - t0
        truth = float(case["true_tempo"])
        verdict, ratio = classify(analysis.tempo, truth)

        # Same check as tempo_bench: the header tempo and the grid every later
        # stage quantises against have to describe the same piece.
        beats = np.asarray(analysis.beats, dtype=float)
        grid_bpm = (60.0 / float(np.median(np.diff(beats)))
                    if len(beats) > 2 else 0.0)
        grid_ok = bool(grid_bpm and abs(grid_bpm / analysis.tempo - 1) < 0.03)

        rows.append({
            "id": case["id"],
            "title": case["title"],
            "tags": case.get("tags", []),
            "tempo_true": truth,
            "tempo_detected": round(float(analysis.tempo), 2),
            "ratio": round(ratio, 3),
            "verdict": verdict,
            "bpb_detected": int(analysis.beats_per_bar),
            "grid_bpm": round(grid_bpm, 2),
            "grid_ok": grid_ok,
            "seconds": round(elapsed, 1),
        })
        r = rows[-1]
        mark = "✓" if verdict == "ok" else "✗"
        print(f"  {mark} {case['title'][:34]:34} {r['tempo_detected']:>7.2f} vs "
              f"{truth:>6.1f}  ({verdict:>7}, ×{r['ratio']:.3f})"
              f"{'' if grid_ok else f'   grid {grid_bpm:.1f} ✗'}   {r['seconds']}s")
    return rows


def summarise(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "tempo_ok": sum(r["verdict"] == "ok" for r in rows),
        "grid_ok": sum(r["grid_ok"] for r in rows),
        "errors": sorted({r["verdict"] for r in rows if r["verdict"] != "ok"}),
        "failed": [r["title"] for r in rows if r["verdict"] != "ok"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="real")
    ap.add_argument("--tag", default=None, help="only cases carrying this tag")
    ap.add_argument("--compare", default=None, help="label of an earlier run")
    args = ap.parse_args()

    cases = load_cases(args.tag)
    pending = [c for c in json.loads(TRUTH.read_text(encoding="utf-8"))["cases"]
               if not c.get("true_tempo")]
    print(f"\nreal bench — {args.label}   ({len(cases)} 首已定案"
          f"{f'，{len(pending)} 首待聽' if pending else ''})\n")

    rows = run(cases)
    if not rows:
        sys.exit("沒有可測的曲目")
    s = summarise(rows)
    print(f"\n  tempo {s['tempo_ok']}/{s['cases']}"
          f"   grid agrees {s['grid_ok']}/{s['cases']}"
          f"   errors: {', '.join(s['errors']) or 'none'}")
    for title in s["failed"]:
        print(f"    ✗ {title}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"real_{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "summary": s, "cases": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → results/tempo/{out.name}")

    if args.compare:
        prior_path = RESULTS / f"real_{args.compare}.json"
        if not prior_path.exists():
            print(f"\n  找不到 {prior_path.name}，略過對照")
            return
        prior = {c["id"]: c for c in json.loads(prior_path.read_text())["cases"]}
        print(f"\n  vs {args.compare}:")
        changed = False
        for r in rows:
            was = prior.get(r["id"])
            if not was or was["verdict"] == r["verdict"]:
                continue
            changed = True
            print(f"    {r['title'][:34]:34} {was['tempo_detected']:>7.2f} "
                  f"({was['verdict']})  →  {r['tempo_detected']:>7.2f} ({r['verdict']})")
        if not changed:
            print("    沒有曲目改變判定")


if __name__ == "__main__":
    main()
