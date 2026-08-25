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
    # Grid-searched over the three piano cases in eval/piano_bench.py, on the
    # separated stem as well as the isolated render. Raising both thresholds
    # buys precision cheaply here: a piano stem's spurious detections are short
    # and weakly-onset, while the writing itself is neither.
    "accompaniment": dict(onset_threshold=0.65, frame_threshold=0.42,
                          minimum_note_length=70.0, minimum_frequency=32.0,
                          maximum_frequency=3000.0, melodia_trick=False,
                          min_note_semiquavers=0.8),
}

# Keys that configure this module rather than basic-pitch, and must not be
# forwarded to it.
_LOCAL_KEYS = ("min_note_semiquavers",)


def _minimum_note_ms(params: dict, tempo: float | None) -> float:
    """The shortest note worth writing, in ms, for a piece at this tempo.

    A fixed millisecond floor cannot be right for two pieces at once: 150ms
    filters spurious detections cleanly at 90 BPM and deletes every semiquaver
    at 150. The grid search picked a large fixed floor because every case it
    scored was slow; expressed as a fraction of a semiquaver it scores the same
    on those and stops being a trap for fast music.
    """
    share = params.get("min_note_semiquavers")
    if not share or not tempo or tempo <= 0:
        return params["minimum_note_length"]
    semiquaver_ms = (60.0 / tempo / 4.0) * 1000.0
    return max(30.0, semiquaver_ms * share)


def transcribe_pitched(wav_path: Path, profile: str,
                       tempo: float | None = None) -> list[Note]:
    """Transcribe one stem. `tempo` lets the note-length floor scale musically."""
    from basic_pitch.inference import predict

    params = dict(PROFILES.get(profile, PROFILES["other"]))
    params["minimum_note_length"] = _minimum_note_ms(params, tempo)
    for key in _LOCAL_KEYS:
        params.pop(key, None)
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



# Floor and ceiling for calibrated velocities. The floor matters: a note quiet
# enough to round to nothing still has to be audible, or checking the
# transcription by ear silently misses it.
VELOCITY_FLOOR, VELOCITY_CEILING = 45, 108
# How much of a note's start to measure. Perceived loudness of a note is set by
# its attack, not by the tail of whatever it decays into.
ATTACK_WINDOW = 0.25


def calibrate_velocity(notes: list[Note], wav_path: Path,
                       floor: int = VELOCITY_FLOOR,
                       ceiling: int = VELOCITY_CEILING) -> list[Note]:
    """Reset each note's velocity from how loud its own stem actually is there.

    basic-pitch reports a per-note amplitude, but it is a confidence-flavoured
    number from the model rather than a measurement of the recording, so a
    clearly-detected quiet note and a marginally-detected loud one can come out
    the same. Reading the stem directly at each onset gives the dynamics a
    listener would actually describe.

    Velocity is set *proportional to measured amplitude*, anchored so the
    loudest notes sit at `ceiling`. Stretching each part's own range onto the
    full written span instead is the obvious thing to write and it is wrong: it
    manufactures dynamics that the performance does not have, and a part played
    at an even level comes back out swinging between pp and ff. Measured that
    way on YOASOBI 勇者 it pushed the rendered dynamic range to 25.3 dB against
    the recording's 15.8 dB — further off than doing nothing at all.
    """
    if not notes:
        return notes
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    if y.size == 0 or np.max(np.abs(y)) < 1e-6:
        return notes
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    levels = []
    for n in notes:
        lo = int(np.searchsorted(times, n.start))
        hi = int(np.searchsorted(times, min(n.end, n.start + ATTACK_WINDOW)))
        window = rms[lo:max(hi, lo + 1)]
        levels.append(float(window.max()) if window.size else 0.0)

    amplitude = np.asarray(levels, dtype=float)
    loud = float(np.percentile(amplitude, 95))
    if loud <= 1e-9:
        return notes
    scaled = ceiling * (amplitude / loud)
    for note, value in zip(notes, scaled):
        note.velocity = int(np.clip(value, floor, ceiling))
    return notes



# A dedicated piano model, used when the material really is a piano. Kept as a
# soft dependency: it needs a 164MB checkpoint, and everything still works
# without it.
_PIANO_MODEL = None
_PIANO_SR = 16000


def piano_model_available() -> bool:
    try:
        import piano_transcription_inference  # noqa: F401
    except Exception:
        return False
    from pathlib import Path as _P
    ckpt = _P.home() / "piano_transcription_inference_data" / \
        "note_F1=0.9677_pedal_F1=0.9186.pth"
    return ckpt.exists() and ckpt.stat().st_size > 1.6e8


def transcribe_piano(wav_path: Path) -> list[Note] | None:
    """Transcribe with a piano-specific model, or None if it is not installed.

    basic-pitch is trained across instruments and has to be right about what it
    is hearing before it can be right about the notes; this one assumes a piano
    and spends all of its capacity on the notes. On the benchmark's
    register-roaming piano the difference is large — note F1 0.910 to 0.960,
    with recall in the outer registers going 0.76 to 0.97 in the bass and 0.79
    to 0.96 in the treble.

    That assumption is also its limit, which is why the caller checks the
    material first: pointed at a band's accompaniment stem it writes the guitars
    and synths out as piano too, and precision falls from 0.88 to 0.72. It is
    also 10-40x slower, so it earns its place only where it wins.
    """
    global _PIANO_MODEL
    if not piano_model_available():
        return None
    from piano_transcription_inference import PianoTranscription

    if _PIANO_MODEL is None:
        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
        with contextlib.redirect_stdout(io.StringIO()):
            _PIANO_MODEL = PianoTranscription(device=device)

    # Its own loader calls a librosa API that no longer exists, so read the
    # audio here instead.
    audio, _ = librosa.load(str(wav_path), sr=_PIANO_SR, mono=True)
    with contextlib.redirect_stdout(io.StringIO()):
        result = _PIANO_MODEL.transcribe(audio, None)

    notes = [
        Note(start=float(e["onset_time"]), end=float(e["offset_time"]),
             pitch=int(e["midi_note"]),
             velocity=int(np.clip(e.get("velocity", 80), 1, 127)))
        for e in result.get("est_note_events", [])
    ]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


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
