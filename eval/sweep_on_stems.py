"""Tune the vocal profile on *separated* stems — what production actually feeds it.

Stems are separated once and cached, so the sweep itself is deterministic even
though Demucs is not.
"""
import itertools, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from pipeline import separate as SEP, transcribe as T
import benchmark as B, cases as C

targets = []
for factory in C.ALL_CASES:
    case = factory()
    if "melody" not in case.truth:
        continue
    mix = B._render(case.score, B.CACHE / f"{case.name}__mix.wav")
    stems = SEP.separate(mix, B.CACHE / f"sweepsep_{case.name}")
    ref = B.truth_arrays(case.truth["melody"], case.tempo_bpm, B.lead_in(mix))
    targets.append((case.name, stems["vocals"], ref))
    print(f"  cached stems for {case.name}")

grid = {
    "onset_threshold": [0.45, 0.55, 0.65, 0.72, 0.80],
    "frame_threshold": [0.16, 0.24, 0.32],
    "minimum_note_length": [70.0, 100.0],
}
keys = list(grid)
base = dict(T.PROFILES["vocals"])
rows = []
for combo in itertools.product(*(grid[k] for k in keys)):
    T.PROFILES["vocals"] = dict(base, **dict(zip(keys, combo)))
    scores = []
    for name, wav, (ref_iv, ref_p) in targets:
        notes = T.make_monophonic(T.transcribe_pitched(wav, "vocals"))
        scores.append(B.score_notes(ref_iv, ref_p, *B.est_from_notes(notes))["f1"])
    rows.append((float(np.mean(scores)), dict(zip(keys, combo)), scores))
    print(f"  onset={combo[0]:.2f} frame={combo[1]:.2f} minlen={combo[2]:5.0f}"
          f"  ->  F1={np.mean(scores):.3f}  {[round(s,2) for s in scores]}")

rows.sort(key=lambda r: -r[0])
print("\n在分軌音軌上的最佳設定：")
for f1, params, scores in rows[:4]:
    print(f"  F1={f1:.3f}  {params}")
