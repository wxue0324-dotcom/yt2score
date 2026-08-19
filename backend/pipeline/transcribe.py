"""Audio stems -> note events, via basic-pitch (pitched) and onset banding (drums)."""
from __future__ import annotations

import contextlib
import io
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

warnings.filterwarnings("ignore", message=".*pkg_resources.*")


@dataclass
class Note:
    start: float        # seconds
    end: float
    pitch: int          # MIDI number
    velocity: int = 80

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DrumHit:
    time: float
    kind: str           # "kick" | "snare" | "hihat"


# Per-stem transcription settings. basic-pitch is generic, so the frequency
# window and thresholds are what keep each stem from picking up its neighbours'
# bleed as spurious notes.
PROFILES = {
    # Tuned by grid search on *separated* stems (eval/sweep_on_stems.py), not
    # on clean isolated audio — the optimum differs, because separation leaves
    # artefacts that a lower onset threshold reads as extra notes. A high onset
    # threshold with a low frame threshold is what a sung line wants: demand
    # real evidence to start a note, then track it loosely so vibrato does not
    # chop it into fragments.
    #
    # 0.80 -> 0.70 once `make_monophonic` stopped discarding quiet notes. The
    # two interact: while overlap resolution dropped anything quieter than its
    # neighbour, a lower onset threshold only added notes that were then thrown
    # away, so the sweep saw no gain from lowering it. It does now.
    "vocals": dict(onset_threshold=0.70, frame_threshold=0.16,
                   minimum_note_length=100.0, minimum_frequency=75.0,
                   maximum_frequency=1200.0, melodia_trick=True),
    "bass": dict(onset_threshold=0.45, frame_threshold=0.30,
                 minimum_note_length=110.0, minimum_frequency=30.0,
                 maximum_frequency=400.0, melodia_trick=True),
    "other": dict(onset_threshold=0.55, frame_threshold=0.34,
                  minimum_note_length=70.0, minimum_frequency=55.0,
                  maximum_frequency=3000.0, melodia_trick=False),
    "accompaniment": dict(onset_threshold=0.55, frame_threshold=0.34,
                          minimum_note_length=70.0, minimum_frequency=32.0,
                          maximum_frequency=3000.0, melodia_trick=False),
}


def transcribe_pitched(wav_path: Path, profile: str) -> list[Note]:
    from basic_pitch.inference import predict

    params = PROFILES.get(profile, PROFILES["other"])
    # The CoreML backend chatters tensor shapes to stdout on every chunk.
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, note_events = predict(str(wav_path), **params)

    notes = [
        Note(start=float(s), end=float(e), pitch=int(p),
             velocity=int(np.clip(amp * 127, 30, 120)))
        for s, e, p, amp, *_ in note_events
    ]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


# Semitone distances at which a detection is most likely an overtone of a
# concurrent louder note rather than a note someone actually sang or played.
_HARMONIC_INTERVALS = (12, 19, 24, 28, 31)


def suppress_harmonics(notes: list[Note], ratio: float = 0.6) -> list[Note]:
    """Drop faint notes sitting an octave/twelfth above a much louder concurrent one.

    basic-pitch reports strong overtones as real notes on harmonically rich
    sources, and those ghosts wreck a melody line. The margin matters: a real
    melody note an octave above the bass is common, so only clearly weaker
    detections — under `ratio` of the lower note's level, and fully covered by
    it in time — are treated as artefacts.
    """
    keep: list[Note] = []
    for n in notes:
        ghost = False
        for other in notes:
            if other is n or n.velocity > other.velocity * ratio:
                continue
            if (n.pitch - other.pitch) not in _HARMONIC_INTERVALS:
                continue
            # An overtone cannot outlast the note that generates it.
            if n.start >= other.start - 0.05 and n.end <= other.end + 0.05:
                ghost = True
                break
        if not ghost:
            keep.append(n)
    return keep


# A note lying entirely inside a louder one, at less than this share of its
# velocity, is leftover harmony or an artefact rather than the next melody note.
CONTAINED_RATIO = 0.7
# Onsets closer together than this are the same attack detected twice.
SIMULTANEOUS = 0.06


def make_monophonic(notes: list[Note]) -> list[Note]:
    """Reduce to a single line by truncating held notes, not by discarding quiet ones.

    An earlier version kept the louder note wherever two overlapped, which cost
    far more than it removed: basic-pitch runs a sustained note's detected end
    well past its real one on reverberant material, so the *next* melody note
    would arrive while the previous was nominally still sounding and be dropped
    for being quieter. Measured on the benchmark's full_band case, that alone
    took melody recall down to 0.562 while precision sat at 0.900 — the notes
    were being found and then thrown away.

    Overlap on its own is therefore treated as a held note that outstayed its
    welcome: clip the earlier note and keep both. Only a note *contained* within
    a louder one is dropped, which is the shape harmony and artefacts actually
    have. (`suppress_harmonics` has already removed the overtones it can
    identify by interval; this catches the rest.)
    """
    if not notes:
        return []
    notes = suppress_harmonics(notes)
    ordered = sorted(notes, key=lambda n: (n.start, -n.velocity, n.pitch))
    result: list[Note] = []
    for note in ordered:
        if not result:
            result.append(note)
            continue
        prev = result[-1]
        if note.start >= prev.end - 0.01:             # no overlap, nothing to resolve
            result.append(note)
            continue
        if (note.end <= prev.end + 0.05
                and note.velocity < prev.velocity * CONTAINED_RATIO):
            continue                                  # sits inside a louder note
        if note.velocity > prev.velocity and note.start - prev.start < SIMULTANEOUS:
            result.pop()                              # same attack, keep the louder
            result.append(note)
            continue
        prev.end = min(prev.end, note.start)          # held too long — clip it
        if prev.duration < 0.05:
            result.pop()
        result.append(note)
    return [n for n in result if n.duration >= 0.05]


def transcribe_drums(wav_path: Path) -> list[DrumHit]:
    """Detect hits and sort them into kick / snare / hihat by band energy."""
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        return []

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, backtrack=True, units="frames",
        pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.15, wait=2,
    )
    onsets = librosa.frames_to_time(onset_frames, sr=sr)

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    low = freqs < 150
    mid = (freqs >= 150) & (freqs < 900)
    high = freqs >= 5000

    hits: list[DrumHit] = []
    for t in onsets:
        f = int(librosa.time_to_frames(t, sr=sr, hop_length=256))
        window = S[:, max(0, f): f + 4]
        if window.size == 0:
            continue
        spec = window.mean(axis=1)
        total = spec.sum() + 1e-9
        lo, md, hi = spec[low].sum() / total, spec[mid].sum() / total, spec[high].sum() / total

        if lo > 0.45:
            kind = "kick"
        elif hi > md:
            kind = "hihat"
        else:
            kind = "snare"
        hits.append(DrumHit(time=float(t), kind=kind))
    return hits
