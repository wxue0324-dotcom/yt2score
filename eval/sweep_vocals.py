"""Grid-search the vocal transcription profile against isolated ground truth."""
import itertools, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from pipeline import transcribe as T
import benchmark as B, cases as C

targets = []
for factory in C.ALL_CASES:
    case = factory()
    if "melody" not in case.truth:
        continue
    iso = C.isolate(case, C.PART_STAVES["melody"])
    wav = B._render(iso, B.CACHE / f"{case.name}__melody.wav")
    shift = B.lead_in(wav)
    ref = B.truth_arrays(case.truth["melody"], case.tempo_bpm, shift)
    targets.append((case.name, wav, ref))

base = dict(T.PROFILES["vocals"])
print("baseline:", base)

grid = {
    "onset_threshold": [0.65, 0.72, 0.80, 0.88],
    "frame_threshold": [0.10, 0.16, 0.22],
    "minimum_note_length": [60.0, 100.0],
}
keys = list(grid)
rows = []
for combo in itertools.product(*(grid[k] for k in keys)):
    params = dict(base, **dict(zip(keys, combo)))
    T.PROFILES["vocals"] = params
    scores = []
    for name, wav, (ref_iv, ref_p) in targets:
        notes = T.make_monophonic(T.transcribe_pitched(wav, "vocals"))
        scores.append(B.score_notes(ref_iv, ref_p, *B.est_from_notes(notes))["f1"])
    rows.append((float(np.mean(scores)), dict(zip(keys, combo)), scores))
    print(f"  onset={combo[0]:.2f} frame={combo[1]:.2f} minlen={combo[2]:5.0f}"
          f"  ->  F1={np.mean(scores):.3f}  {[round(s,2) for s in scores]}")

rows.sort(key=lambda r: -r[0])
print("\nbest 3:")
for f1, params, scores in rows[:3]:
    print(f"  F1={f1:.3f}  {params}")
