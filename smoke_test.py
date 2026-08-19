"""Validate analyze -> transcribe -> quantize -> score -> jianpu without touching YouTube."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import numpy as np, soundfile as sf

SR = 44100
TEMPO = 120.0
SPB = 60.0 / TEMPO

def tone(freq, dur, sr=SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # A few harmonics + a decay envelope give basic-pitch something realistic.
    wave = sum(np.sin(2*np.pi*freq*h*t) / (h**1.4) for h in (1, 2, 3, 4))
    env = np.minimum(1.0, np.linspace(0, 30, len(t))) * np.exp(-2.0 * t)
    return wave * env

def midi_hz(m): return 440.0 * 2 ** ((m - 69) / 12)

# C major: a plain 8-bar melody over root-position triads.
melody = [(60,1),(62,1),(64,1),(65,1),(67,2),(65,1),(64,1),
          (62,1),(64,1),(65,1),(64,1),(62,4),
          (67,1),(65,1),(64,1),(62,1),(60,4)]
chords  = [([48,52,55],4),([53,57,60],4),([55,59,62],4),([48,52,55],4)] * 2

total = sum(d for _, d in melody) * SPB + 2
buf = np.zeros(int(SR * total))

pos = 0.0
for pitch, beats in melody:
    d = beats * SPB
    seg = tone(midi_hz(pitch), d) * 0.5
    i = int(pos * SR); buf[i:i+len(seg)] += seg[:len(buf)-i]
    pos += d

pos = 0.0
for pitches, beats in chords:
    d = beats * SPB
    for p in pitches:
        seg = tone(midi_hz(p), d) * 0.18
        i = int(pos * SR); buf[i:i+len(seg)] += seg[:len(buf)-i]
    pos += d

buf = buf / (np.max(np.abs(buf)) + 1e-9) * 0.9
work = Path("work/smoke"); work.mkdir(parents=True, exist_ok=True)
wav = work / "test.wav"
sf.write(wav, np.stack([buf, buf], axis=1), SR)
print(f"generated {wav} ({total:.1f}s)")

from pipeline import analyze as A, transcribe as T, score as S, jianpu as J
from pipeline.quantize import BeatGrid, quantize_notes, split_hands

an = A.analyze(wav)
print(f"\nANALYZE  tempo={an.tempo:.1f} key={an.key_name} conf={an.key_confidence:.2f} bpb={an.beats_per_bar}")
assert 100 < an.tempo < 140, f"tempo way off: {an.tempo}"

notes = T.transcribe_pitched(wav, "accompaniment")
print(f"TRANSCRIBE  {len(notes)} raw notes, pitch range {min(n.pitch for n in notes)}-{max(n.pitch for n in notes)}")
assert notes, "no notes transcribed"

mel = T.make_monophonic([n for n in notes if n.pitch >= 58])
print(f"MELODY  {len(mel)} monophonic notes")

grid = BeatGrid(an)
gm = quantize_notes(mel, grid)
ga = quantize_notes(notes, grid)
rh, lh = split_hands(ga)
print(f"QUANTIZE  melody={len(gm)} rh={len(rh)} lh={len(lh)}")
print(f"  first 8 melody: {[(g.offset, g.duration, g.pitch) for g in gm[:8]]}")

sc = S.build_score({"vocal": gm, "piano_rh": rh, "piano_lh": lh}, an, "Smoke Test", "yt2score")
out = S.export(sc, work / "out", "score")
print(f"\nEXPORT  {[(k, str(v)[-40:] if not isinstance(v, list) else f'{len(v)} pages') for k, v in out.items()]}")
assert "musicxml" in out and "midi" in out

jp = J.write_jianpu(gm, an, "Smoke Test", work / "out", subtitle="test")
print(f"JIANPU  {jp.get('page_count')} page(s) -> {jp.get('html')}")
print("\nALL STAGES PASSED")
