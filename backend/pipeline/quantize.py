"""Snap raw seconds-based notes onto the musical grid found by analyze.py."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analyze import Analysis
from .transcribe import DrumHit, Note

# Slots per beat. Transcription timing jitter is roughly a 30th note, so a
# 16th-note grid mostly encodes noise on material that has no semiquavers in
# it; 8ths give a far more readable draft. Material that *does* carry
# semiquavers loses them entirely at 8ths, so the choice is made per part by
# `choose_subdivision` rather than fixed here.
SUBDIVISION = 2
FINE_SUBDIVISION = 4
MIN_SLOTS = 1

# What share of the coarse grid's residual the finer grid has to remove before
# it is judged to be describing real semiquavers rather than jitter.
#
# Deliberately a ratio, not an absolute distance in beats. Separated stems carry
# far more onset jitter than a clean render — the same part measured after
# Demucs shows roughly double the residual against either grid — so an absolute
# threshold tuned on clean audio fires on everything once separation is in the
# path, which is exactly what happened: every stem-fed part chose 16ths and
# full_band's melody fell from 0.909 to 0.788 end to end. As a proportion the
# two conditions agree. Measured across both conditions: parts with no
# semiquavers in them score 0.00–0.43, the semiquaver case scores 0.57 clean
# and 0.57 through separation. The threshold sits between those, nearer the
# middle than either edge, because the closest non-semiquaver case (full_band's
# melody after separation, 17 notes) is the noisiest estimate in the set.
FINE_MARGIN = 0.50


@dataclass
class GridNote:
    offset: float        # in quarter-length units from the start of the score
    duration: float      # quarter-length
    pitch: int
    velocity: int = 80


class BeatGrid:
    """Maps seconds <-> quarter-length using the detected beat times.

    Real recordings drift, so we interpolate against the actual beat positions
    rather than assuming a constant tempo.
    """

    def __init__(self, analysis: Analysis):
        beats = np.asarray(analysis.beats, dtype=float)
        # Extend the grid backwards to 0 and forwards a little so notes at the
        # very start or end of the track still land somewhere sensible.
        spb = analysis.seconds_per_beat
        if len(beats) < 2:
            beats = np.arange(0.0, 600.0, spb)
        lead = np.arange(beats[0] - spb, -spb, -spb)[::-1]
        tail = beats[-1] + np.arange(spb, 60.0, spb)
        self._times = np.concatenate([lead, beats, tail])
        self._positions = np.arange(len(self._times), dtype=float) - len(lead)
        self.analysis = analysis
        # Anchor beat 0 at audio time 0 so no part ever produces a negative
        # offset, and every part is quantised against the same origin. Shifting
        # per part instead would silently desynchronise the staves.
        raw_origin = float(np.interp(0.0, self._times, self._positions))
        bar = float(analysis.beats_per_bar)
        self.origin = np.floor(raw_origin / bar) * bar

    def to_beats(self, seconds: float) -> float:
        return float(np.interp(seconds, self._times, self._positions)) - self.origin

    def to_seconds(self, beats: float) -> float:
        """Inverse of to_beats: quarter-length position back to wall-clock time."""
        return float(np.interp(beats + self.origin, self._positions, self._times))

    def snap(self, seconds: float, subdivision: int = SUBDIVISION) -> float:
        """Seconds -> quarter-length, snapped to the given subdivision."""
        return round(self.to_beats(seconds) * subdivision) / subdivision


def choose_subdivision(notes: list[Note], grid: BeatGrid,
                       margin: float = FINE_MARGIN) -> int:
    """Pick the 8th- or 16th-note grid by asking which one the onsets already fit.

    Snapping to a grid coarser than the music moves every off-grid note onto a
    neighbour, where `_merge_duplicates` and the engraver fold it into whatever
    is already there — semiquavers do not come out inaccurate, they come out
    missing. Snapping to one finer than the music is the milder failure: the
    notes survive, the draft just carries more subdivisions than a person would
    have written.

    So rather than assume, compare how far the onsets actually sit from each
    grid. Material genuinely written in quavers is no closer to the 16th grid
    than to the 8th — its residual is timing jitter, which both grids see
    equally. Material carrying semiquavers is much closer to the finer one,
    because half its onsets lie a full 16th away from every 8th.

    The comparison is proportional so that it survives separation, which
    roughly doubles the residual without changing what was written.
    """
    if len(notes) < 8:
        return SUBDIVISION
    positions = np.array([grid.to_beats(n.start) for n in notes], dtype=float)

    def residual(subdivision: int) -> float:
        scaled = positions * subdivision
        return float(np.mean(np.abs(scaled - np.round(scaled)) / subdivision))

    coarse = residual(SUBDIVISION)
    if coarse <= 1e-9:
        return SUBDIVISION
    improvement = (coarse - residual(FINE_SUBDIVISION)) / coarse
    return FINE_SUBDIVISION if improvement > margin else SUBDIVISION


def quantize_notes(notes: list[Note], grid: BeatGrid,
                   subdivision: int | None = None) -> list[GridNote]:
    """Snap notes onto the grid. `subdivision=None` picks one per part."""
    if subdivision is None:
        subdivision = choose_subdivision(notes, grid)
    out: list[GridNote] = []
    step = 1.0 / subdivision
    for n in notes:
        start = grid.snap(n.start, subdivision)
        end = grid.snap(n.end, subdivision)
        if end - start < step * MIN_SLOTS:
            end = start + step * MIN_SLOTS
        out.append(GridNote(offset=max(0.0, start), duration=end - start,
                            pitch=n.pitch, velocity=n.velocity))

    out.sort(key=lambda g: (g.offset, g.pitch))
    return _merge_duplicates(out)


def _merge_duplicates(notes: list[GridNote]) -> list[GridNote]:
    """Two notes of the same pitch landing on the same slot are one note."""
    seen: dict[tuple[float, int], GridNote] = {}
    for n in notes:
        key = (n.offset, n.pitch)
        if key in seen:
            seen[key].duration = max(seen[key].duration, n.duration)
        else:
            seen[key] = n
    return sorted(seen.values(), key=lambda g: (g.offset, g.pitch))


def quantize_drums(hits: list[DrumHit], grid: BeatGrid) -> list[tuple[float, str]]:
    return sorted({(max(0.0, grid.snap(h.time)), h.kind) for h in hits})


def split_hands(notes: list[GridNote], split_pitch: int = 60) -> tuple[list[GridNote], list[GridNote]]:
    """Divide accompaniment notes into right hand / left hand around middle C.

    A fixed split at C4 misfires on bass-heavy or treble-heavy material, so the
    point moves to the median of the actual pitch range first.
    """
    if not notes:
        return [], []
    pitches = np.array([n.pitch for n in notes])
    median = int(np.median(pitches))
    split = int(np.clip(median, 52, 67)) if abs(median - split_pitch) > 7 else split_pitch
    right = [n for n in notes if n.pitch >= split]
    left = [n for n in notes if n.pitch < split]
    return right, left


def leading_rest_shift(parts: dict, drums_key: str = "drums",
                       beats_per_bar: int = 4) -> float:
    """How many quarter-lengths of empty bars sit before the first note.

    Exposed separately from `trim_leading_rest` because callers that want to
    line the score up against the original recording need the same number: once
    the rest is trimmed, score time 0 no longer means audio time 0, and nothing
    downstream can recover the difference.
    """
    starts = []
    for key, value in parts.items():
        if not value:
            continue
        if key == drums_key:
            starts.append(min(o for o, _ in value))
        else:
            starts.append(min(n.offset for n in value))
    if not starts:
        return 0.0
    bar = float(beats_per_bar)
    return max(0.0, float(np.floor(min(starts) / bar) * bar))


def trim_leading_rest(parts: dict, drums_key: str = "drums",
                      beats_per_bar: int = 4) -> dict:
    """Remove whole empty bars at the start, shifting every part by the same amount.

    Applied per part this would desynchronise the score, so it takes the whole
    set at once and uses one shift for all of them.
    """
    shift = leading_rest_shift(parts, drums_key, beats_per_bar)
    if shift <= 0:
        return parts

    out = {}
    for key, value in parts.items():
        if not value:
            out[key] = value
        elif key == drums_key:
            out[key] = [(o - shift, k) for o, k in value]
        else:
            out[key] = [GridNote(offset=n.offset - shift, duration=n.duration,
                                 pitch=n.pitch, velocity=n.velocity) for n in value]
    return out


def merge_repeats(notes: list[GridNote], overlap: float = 1e-6) -> list[GridNote]:
    """Fuse same-pitch notes that genuinely overlap in time.

    Only real overlap counts. Merging merely *adjacent* same-pitch notes would
    silently rewrite deliberately repeated notes — a bass ostinato, a repeated
    quaver figure — as one long held note, which is wrong far more often than
    it is right.
    """
    by_pitch: dict[int, list[GridNote]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)

    merged: list[GridNote] = []
    for group in by_pitch.values():
        group.sort(key=lambda g: g.offset)
        current = group[0]
        for nxt in group[1:]:
            if nxt.offset < current.offset + current.duration - overlap:
                end = max(current.offset + current.duration,
                          nxt.offset + nxt.duration)
                current.duration = end - current.offset
                current.velocity = max(current.velocity, nxt.velocity)
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
    return sorted(merged, key=lambda g: (g.offset, g.pitch))


def fix_octave_jumps(notes: list[GridNote], window: int = 5,
                     threshold: int = 9) -> list[GridNote]:
    """Pull outlying notes back by an octave when that fits the local line.

    Pitch trackers on bass and vocals slip an octave now and then; left alone
    those notes throw the part onto far ledger lines.
    """
    if len(notes) < 3:
        return notes
    ordered = sorted(notes, key=lambda g: g.offset)
    pitches = [n.pitch for n in ordered]

    for i, n in enumerate(ordered):
        lo = max(0, i - window)
        hi = min(len(ordered), i + window + 1)
        neighbours = sorted(pitches[j] for j in range(lo, hi) if j != i)
        if not neighbours:
            continue
        local = neighbours[len(neighbours) // 2]
        if abs(n.pitch - local) <= threshold:
            continue
        best = min((n.pitch - 12, n.pitch, n.pitch + 12),
                   key=lambda cand: abs(cand - local))
        if best != n.pitch and 21 <= best <= 108:
            n.pitch = best
            pitches[i] = best
    return ordered


def drop_stray_notes(notes: list[GridNote], min_duration: float = 0.5,
                     keep_ratio: float = 0.35) -> list[GridNote]:
    """Remove the quietest very-short notes, which are almost always artefacts.

    Anything at or above `min_duration` is kept regardless — brief notes are
    real in fast passages, so only the weak *and* short ones go.
    """
    if not notes:
        return notes
    short = [n for n in notes if n.duration < min_duration]
    if not short:
        return notes
    cutoff = sorted(n.velocity for n in short)[int(len(short) * keep_ratio)]
    return [n for n in notes
            if n.duration >= min_duration or n.velocity > cutoff]


def clean_part(notes: list[GridNote], min_duration: float = 0.5,
               fix_octaves: bool = False) -> list[GridNote]:
    """The standard tidy-up applied to every transcribed part.

    Kept here rather than in the pipeline runner so that evaluation measures
    exactly what users receive.
    """
    notes = merge_repeats(notes)
    if fix_octaves:
        notes = fix_octave_jumps(notes)
    return drop_stray_notes(notes, min_duration=min_duration)
