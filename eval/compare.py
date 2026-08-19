"""Before/after comparison under identical measurement conditions.

Restores the original behaviour by patching, rather than editing the pipeline,
so both configurations run through exactly the same harness.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, librosa
from pipeline import analyze as A, transcribe as T
from pipeline import quantize as Q
import benchmark as B, cases as C

ORIGINAL_VOCALS = dict(onset_threshold=0.55, frame_threshold=0.32,
                       minimum_note_length=90.0, minimum_frequency=75.0,
                       maximum_frequency=1200.0, melodia_trick=True)


def original_analyze(wav_path):
    """The first version: one librosa call, key from chroma, no refinement."""
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) < 4 or not np.isfinite(tempo) or tempo <= 0:
        tempo = 120.0
        beats = np.arange(0, librosa.get_duration(y=y, sr=sr), 0.5)
    bpb = A._estimate_beats_per_bar(y, sr, beats)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = np.clip(librosa.time_to_frames(beats, sr=sr), 0, len(onset_env) - 1)
    strengths = onset_env[frames]
    phase = max(range(bpb), key=lambda o: strengths[o::bpb].mean())
    tonic, mode, conf = A._estimate_key(y, sr)
    return A.Analysis(tempo=tempo, beats=beats, downbeats=beats[phase::bpb],
                      beats_per_bar=bpb, tonic=tonic, mode=mode,
                      key_confidence=conf)


def original_merge_repeats(notes, gap=1e-6):
    """Merged merely-adjacent same-pitch notes, flattening repeated notes."""
    by_pitch = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    merged = []
    for group in by_pitch.values():
        group.sort(key=lambda g: g.offset)
        current = group[0]
        for nxt in group[1:]:
            if nxt.offset <= current.offset + current.duration + gap:
                end = max(current.offset + current.duration,
                          nxt.offset + nxt.duration)
                current.duration = end - current.offset
            else:
                merged.append(current); current = nxt
        merged.append(current)
    return sorted(merged, key=lambda g: (g.offset, g.pitch))


def run(label):
    results = []
    for factory in C.ALL_CASES:
        case = factory()
        entry = {"case": case.name,
                 "stage_a": B.stage_a(case, B.CACHE, False),
                 "stage_b": B.stage_b(case, B.CACHE, False)}
        results.append(entry)
    a_raw = [p["raw"]["f1"] for r in results for p in r["stage_a"].values()]
    a_fin = [p["final"]["f1"] for r in results for p in r["stage_a"].values()]
    b_fin = [p["f1"] for r in results for p in r["stage_b"]["parts"].values()]
    keys = [r["stage_b"]["key_ok"] for r in results]
    temps = [r["stage_b"]["tempo_err"] for r in results]
    out = {"A_raw": np.mean(a_raw), "A_final": np.mean(a_fin),
           "B_final": np.mean(b_fin), "key": np.mean(keys),
           "tempo_err": np.mean(temps)}
    print(f"  {label:10s} A_raw={out['A_raw']:.3f}  A_final={out['A_final']:.3f}"
          f"  B_final={out['B_final']:.3f}  key={out['key']:.0%}"
          f"  tempo_err={out['tempo_err']:.1f}")
    return out


print("改動後（目前）")
after = run("after")

# Restore the original behaviour.
real_analyze = A.analyze
real_key = A.estimate_key_from_notes
real_merge = Q.merge_repeats

A.analyze = original_analyze
A.estimate_key_from_notes = lambda parts, bpb=4: None
Q.merge_repeats = original_merge_repeats
B.clean_part.__globals__["merge_repeats"] = original_merge_repeats
T.PROFILES["vocals"] = ORIGINAL_VOCALS

print("改動前（原始）")
before = run("before")

A.analyze = real_analyze
A.estimate_key_from_notes, Q.merge_repeats = real_key, real_merge

print("\n" + "=" * 62)
print(f"  {'指標':<12s} {'改動前':>10s} {'改動後':>10s} {'變化':>12s}")
for key, name in (("A_raw", "採譜 F1"), ("A_final", "量化後 F1"),
                  ("B_final", "端到端 F1"), ("key", "調性正確率"),
                  ("tempo_err", "速度誤差")):
    b, a = before[key], after[key]
    if key == "key":
        print(f"  {name:<12s} {b:>9.0%} {a:>10.0%} {(a-b)*100:>+11.0f}pt")
    elif key == "tempo_err":
        print(f"  {name:<12s} {b:>9.1f} {a:>10.1f} {a-b:>+11.1f} BPM")
    else:
        print(f"  {name:<12s} {b:>9.3f} {a:>10.3f} {(a-b)/max(b,1e-9)*100:>+11.1f}%")
print("=" * 62)
