"""Tempo, beat grid and key estimation — the scaffolding every later stage hangs off."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

# Krumhansl-Kessler profiles, the standard weights for key correlation.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_PITCH_NAMES = ["C", "C#", "D", "E-", "E", "F", "F#", "G", "G#", "A", "B-", "B"]


@dataclass
class Analysis:
    tempo: float
    beats: np.ndarray = field(repr=False)   # beat times in seconds
    downbeats: np.ndarray = field(repr=False)
    beats_per_bar: int
    tonic: str
    mode: str          # "major" | "minor"
    key_confidence: float

    @property
    def key_name(self) -> str:
        return f"{self.tonic} {self.mode}"

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo


def _estimate_key(y: np.ndarray, sr: int) -> tuple[str, str, float]:
    # CQT chroma tracks pitch class far more cleanly than STFT chroma on real mixes.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    if profile.sum() == 0:
        return "C", "major", 0.0
    profile = profile / profile.sum()

    best = ("C", "major", -1.0)
    for tonic in range(12):
        for name, template in (("major", _MAJOR), ("minor", _MINOR)):
            rotated = np.roll(template, tonic)
            corr = float(np.corrcoef(profile, rotated)[0, 1])
            if corr > best[2]:
                best = (_PITCH_NAMES[tonic], name, corr)
    return best


def _estimate_beats_per_bar(y: np.ndarray, sr: int, beats: np.ndarray) -> int:
    """Pick 3 or 4 by testing which grouping puts more onset energy on the downbeat."""
    if len(beats) < 12:
        return 4
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.time_to_frames(beats, sr=sr)
    frames = np.clip(frames, 0, len(onset_env) - 1)
    strengths = onset_env[frames]

    scores = {}
    for bpb in (3, 4):
        # Best phase offset for this grouping: strongest average accent on beat 1.
        scores[bpb] = max(
            strengths[offset::bpb].mean() / (strengths.mean() + 1e-9)
            for offset in range(bpb)
        )
    # 4/4 is overwhelmingly the prior; only pick 3 when it wins clearly.
    return 3 if scores[3] > scores[4] * 1.05 else 4


# Plausible range for a notated pulse. Outside this, a "tempo" is really a
# doubled or halved reading of the real one.
_TEMPO_MIN, _TEMPO_MAX = 50.0, 200.0


def _beat_alignment(onset_times: np.ndarray, beats: np.ndarray,
                    tolerance: float = 0.07) -> float:
    """F1 between a beat grid and the actual note onsets.

    A grid at half the true tempo misses every off-beat onset and loses recall,
    which is what stops this collapsing to 27 BPM for a 110 BPM piece the way
    scoring beat-position strength alone did.

    It is *not* symmetric, despite the shape of the formula. A grid at double
    the tempo only loses precision if the extra beats land on silence — so on
    music carrying a steady subdivision, where they land on notes instead, the
    doubled grid scores as well as the true one and the quadrupled grid better
    still. Measured on a piece of flowing solo piano, this score rises
    monotonically from 0.21 at 60 BPM to 0.41 at 185. That bias is why the
    caller weights this by `_tempo_prior` instead of taking the argmax.
    """
    if len(beats) < 2 or len(onset_times) == 0:
        return 0.0
    beats = np.asarray(beats, dtype=float)
    onset_times = np.asarray(onset_times, dtype=float)

    def covered(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) == 0:
            return 0.0
        idx = np.searchsorted(b, a)
        best = np.full(len(a), np.inf)
        for offset in (-1, 0):
            probe = np.clip(idx + offset, 0, len(b) - 1)
            best = np.minimum(best, np.abs(b[probe] - a))
        return float(np.mean(best <= tolerance))

    precision = covered(beats, onset_times)     # beats that land on a note
    recall = covered(onset_times, beats)        # notes that land on a beat
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _subdivide(beats: np.ndarray) -> np.ndarray:
    """Halve the beat period by adding a beat between each existing pair."""
    if len(beats) < 2:
        return beats
    mid = (beats[:-1] + beats[1:]) / 2
    return np.sort(np.concatenate([beats, mid]))


def _fold_to_range(tempo: float, beats: np.ndarray) -> tuple[float, np.ndarray]:
    """Bring a tempo into the notatable range, carrying its beat grid with it.

    Folding the number on its own is a silent disaster: `BeatGrid` counts one
    quarter note per entry in `beats`, while the score header is written from
    `tempo`. Halve the number and leave the grid alone and the two describe
    different pieces — every duration comes out doubled, and the lead-in the
    grid extrapolates from `seconds_per_beat` no longer lines up with the
    beats it is extending. Both have to move together.
    """
    beats = np.asarray(beats, dtype=float)
    if not np.isfinite(tempo) or tempo <= 0:
        return 120.0, beats
    while tempo < _TEMPO_MIN:
        tempo *= 2
        beats = _subdivide(beats)
    while tempo > _TEMPO_MAX:
        tempo /= 2
        beats = beats[::2]
    return tempo, beats


# Listeners settle on a pulse near two per second; grids far from it are
# usually a subdivision, or a bar being read as the beat. Wide on purpose —
# this breaks ties between metrical levels rather than overruling evidence, so
# a genuinely fast or slow piece can still win on alignment.
_TEMPO_CENTRE, _TEMPO_SPREAD = 120.0, 1.0


def _tempo_prior(tempo: float) -> float:
    """How plausible this tempo is as a *notated* pulse, on a log-2 scale."""
    if not np.isfinite(tempo) or tempo <= 0:
        return 0.0
    octaves = np.log2(tempo / _TEMPO_CENTRE) / _TEMPO_SPREAD
    return float(np.exp(-0.5 * octaves ** 2))


def _grid_tempo(beats: np.ndarray, fallback: float) -> float:
    """The tempo a beat grid actually runs at, from its median beat period."""
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 3:
        return fallback
    period = float(np.median(np.diff(beats)))
    return 60.0 / period if period > 0 else fallback


def _fold_tempo(tempo: float) -> float:
    """Bring a tempo into the notatable range by octaves."""
    return _fold_to_range(tempo, np.array([]))[0]


# `beat_track` runs on a 512-sample hop, so every beat it returns is pinned to a
# 23 ms lattice and the period it implies can only be a whole number of frames.
# Near 150 BPM the only readings reachable are 143.55 and 152.00 — 150 cannot be
# expressed at all. That is not cosmetic: the number lands in the score header
# and sets the speed the rendered audio plays back at, so the draft slides
# against the recording it was made from. Measured against hand-fitted ground
# truth, the median-of-frames reading was out by 0.6-1.3% on every track tested
# (Octopath 152.00 vs 150.02 true, LOFI 80.75 vs 80.17, 勇者 103.36 vs 103.99);
# refining the period against a finer envelope brought all three inside 0.06%.
_REFINE_HOP = 128
# The hop `beat_track` ran at, which is what sets the lattice being undone.
_TRACK_HOP = 512
# Hard ceiling on the search, so this can never reach another metrical level —
# the tempo-octave question is `_estimate_tempo`'s to answer, not this one.
_REFINE_SPAN_MAX = 0.04
# Roughly a performer's timing spread. Onsets inside it count as on the grid.
_REFINE_SIGMA = 0.025


def _subframe_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """Onset times measured off the frame lattice.

    Peak picking alone would just swap one lattice for a finer one, so each
    peak is located by fitting a parabola through it and its two neighbours —
    the standard sub-sample peak estimate. Without this the refinement below
    has nothing to measure that is sharper than what it is trying to correct.
    """
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_REFINE_HOP)
    if env.size < 3:
        return np.array([])
    # backtrack=False on purpose: backtracking slides each onset to the quiet
    # point *before* the attack, which is not where the beat is.
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                       hop_length=_REFINE_HOP, backtrack=False)
    step = _REFINE_HOP / sr
    times = []
    for k in peaks:
        if 0 < k < len(env) - 1:
            before, here, after = float(env[k - 1]), float(env[k]), float(env[k + 1])
            curve = before - 2 * here + after
            shift = 0.5 * (before - after) / curve if curve else 0.0
            times.append((k + float(np.clip(shift, -0.5, 0.5))) * step)
        else:
            times.append(float(k) * step)
    return np.asarray(times, dtype=float)


def onset_times(wav_path: Path) -> np.ndarray:
    """Attack times in a recording, measured off the frame lattice.

    Exposed for the quantiser, which has to decide what grid a passage is
    written on. Doing that from transcribed notes is much weaker than it looks:
    the piano model alone adds around 29 ms of onset jitter, which is most of a
    sixth of a beat at 150 BPM, and it buries the difference between a passage
    of quavers and a passage of triplets. The audio it was transcribed from
    carries the same attacks with roughly 4 ms of spread.
    """
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    return _subframe_onsets(y, sr)


def _refine_tempo(y: np.ndarray, sr: int, beats: np.ndarray, tempo: float) -> float:
    """Sharpen the tempo reading against onsets measured off the frame lattice.

    Only the number moves. The beat grid keeps following the performance, which
    is what every later stage quantises against and what holds phase through the
    small wanderings a real recording has; a rigid grid rebuilt at the refined
    period was measured to be far worse, explaining 15% of the onsets where the
    tracked grid explains 88%. So this is not the octave-correcting fold in
    `_fold_to_range`, which must move number and grid together — it is a
    sub-percent correction that cannot change any note's slot, only the speed
    the score is played and printed at.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4 or not np.isfinite(tempo) or tempo <= 0:
        return tempo
    onsets = _subframe_onsets(y, sr)
    if len(onsets) < 8:
        return tempo

    # Only ever move by as much as the lattice could have been wrong: the
    # tracked period is a whole number of frames, so the truth is at most half a
    # frame per beat away. Deriving the span rather than fixing it keeps the
    # correction provably confined to the error it exists to undo — and keeps it
    # small at slow tempos, where half a frame is a much smaller share of the
    # beat (1.5% at 80 BPM against 2.9% at 152).
    period = 60.0 / tempo
    span = min(_REFINE_SPAN_MAX, (_TRACK_HOP / sr) / 2.0 / period)
    best_score, best_period = -1.0, period
    for period in best_period * np.linspace(1 - span, 1 + span, 321):
        # Score against half-beat slots. Scoring on beats alone leaves music
        # that puts most of its attacks off the beat with almost nothing to
        # fit; finer than this and a triplet passage starts pulling the answer.
        step = period / 2.0
        angle = 2 * np.pi * (onsets % step) / step
        phase = float(np.angle(np.exp(1j * angle).mean()) / (2 * np.pi) * step)
        offset = np.abs(((onsets - phase + step / 2) % step) - step / 2)
        score = float(np.mean(np.exp(-0.5 * (offset / _REFINE_SIGMA) ** 2)))
        if score > best_score:
            best_score, best_period = score, period
    return 60.0 / best_period


def _estimate_tempo(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Pick the tempo whose beat grid best explains the onsets.

    librosa's single-shot estimate frequently locks onto half or double the
    notated pulse, so several seeds are tried and each resulting grid scored.

    Be aware of how little the seeding actually buys: `beat_track` treats
    `start_bpm` as a suggestion and its own periodicity estimate usually wins,
    so a dozen seeds commonly collapse onto one or two distinct grids. Measured
    on a 24-second lofi sample, twelve seed/envelope pairs produced exactly two
    answers. Passing `bpm=` instead would force each candidate to be a real
    alternative — but the scoring function is biased toward fine grids (see
    `_beat_alignment`), so forcing genuine candidates without a better score
    makes the result worse, not better. The collapse is currently hiding that.
    """
    hop = 512
    # The percussive component gives a cleaner pulse than the full mix, but on
    # sustained material (solo piano, strings) it can be nearly empty — so keep
    # the plain envelope as a fallback.
    try:
        y_perc = librosa.effects.percussive(y, margin=2.0)
        onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop)
    except Exception:
        onset_env = np.array([])
    plain_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if onset_env.size == 0 or onset_env.mean() < plain_env.mean() * 0.2:
        onset_env = plain_env

    onset_times = librosa.onset.onset_detect(
        onset_envelope=plain_env, sr=sr, hop_length=hop,
        units="time", backtrack=True)

    seeds = list(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
                              hop_length=hop, aggregate=np.median)))
    seeds += list(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=plain_env, sr=sr,
                              hop_length=hop, aggregate=np.median)))
    candidates: list[float] = []
    for seed in seeds + [90.0, 120.0]:
        for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
            value = float(seed) * factor
            if _TEMPO_MIN <= value <= _TEMPO_MAX:
                candidates.append(round(value, 2))
    candidates = sorted(set(candidates))

    best_score, best_tempo, best_beats = -1.0, 120.0, np.array([])
    for candidate in candidates:
        for env in (onset_env, plain_env):
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=env, sr=sr, hop_length=hop,
                start_bpm=candidate, trim=False)
            beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
            # Fold first, then score: the grid that gets scored has to be the
            # grid that gets used, or the winner is chosen on evidence that no
            # later stage ever sees.
            tempo, beats = _fold_to_range(
                _grid_tempo(beats, fallback=float(np.atleast_1d(tempo)[0])), beats)
            # Read the tempo off the folded grid, not from the scalar
            # beat_track returned. Its dynamic programming inserts and drops
            # beats, so the scalar can differ from the spacing of the beats it
            # actually produced, and folding an uneven grid does not scale that
            # spacing exactly either. The grid is what every later stage
            # quantises against, so the grid is what the header must describe.
            tempo = _grid_tempo(beats, fallback=tempo)
            score = _beat_alignment(onset_times, beats) * _tempo_prior(tempo)
            if score > best_score:
                best_score, best_tempo, best_beats = score, tempo, beats

    return best_tempo, best_beats


def analyze(wav_path: Path) -> Analysis:
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)

    tempo, beats = _estimate_tempo(y, sr)

    if len(beats) < 4 or not np.isfinite(tempo) or tempo <= 0:
        # Degenerate audio (silence, pure noise) — fall back to a plain 120bpm grid.
        tempo = 120.0
        duration = librosa.get_duration(y=y, sr=sr)
        beats = np.arange(0, duration, 0.5)
    else:
        # Only worth doing on a grid that came from real tracking; the fallback
        # grid above is already exact and has no onsets to fit against.
        tempo = _refine_tempo(y, sr, beats, tempo)

    bpb = _estimate_beats_per_bar(y, sr, beats)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = np.clip(librosa.time_to_frames(beats, sr=sr), 0, len(onset_env) - 1)
    strengths = onset_env[frames]
    phase = max(range(bpb), key=lambda o: strengths[o::bpb].mean())
    downbeats = beats[phase::bpb]

    tonic, mode, conf = _estimate_key(y, sr)

    return Analysis(
        tempo=tempo, beats=beats, downbeats=downbeats, beats_per_bar=bpb,
        tonic=tonic, mode=mode, key_confidence=conf,
    )


def estimate_key_from_notes(parts: dict, beats_per_bar: int = 4
                            ) -> tuple[str, str, float] | None:
    """Estimate key from transcribed notes rather than raw audio chroma.

    Once notes exist they are a much cleaner signal than a CQT chromagram of a
    full mix: percussion, reverb tails and inharmonic content are already gone,
    and note length tells us which pitches actually carry structural weight.

    `parts` maps a part name to a list of objects with .pitch and .duration.
    Returns (tonic, mode, confidence), or None when there is too little to go on.
    """
    weights = np.zeros(12)
    total = 0.0
    for name, notes in parts.items():
        if not notes:
            continue
        # The bass line pins the tonic; melody defines the mode. Inner voices
        # are the least reliable, so they count for less.
        emphasis = 1.4 if "bass" in name else (1.2 if "vocal" in name else 1.0)
        for note in notes:
            duration = float(getattr(note, "duration", 1.0) or 0.0)
            if duration <= 0:
                continue
            weights[int(getattr(note, "pitch", 0)) % 12] += duration * emphasis
            total += duration

    if total < beats_per_bar * 2 or not np.any(weights):
        return None
    weights = weights / weights.sum()

    best = ("C", "major", -1.0)
    for tonic in range(12):
        for mode, template in (("major", _MAJOR), ("minor", _MINOR)):
            rotated = np.roll(template, tonic)
            corr = float(np.corrcoef(weights, rotated)[0, 1])
            if corr > best[2]:
                best = (_PITCH_NAMES[tonic], mode, corr)
    return best


# NOTE: an earlier version tried to correct the tempo octave here by checking
# how many transcribed onsets fell between beats, doubling the grid when many
# did. It was removed: notes between beats are simply quavers, which is
# ordinary music, so the test fired on correctly-tracked tracks and doubled
# them. Distinguishing "the beat is a crotchet and this piece has quavers" from
# "the beat should be twice as fast" is not decidable from onset positions —
# it is a question about which pulse a listener feels.
