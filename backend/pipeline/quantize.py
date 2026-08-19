"""Snap raw seconds-based notes onto the musical grid found by analyze.py."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analyze import Analysis
from .transcribe import DrumHit, Note

# Slots per beat. Transcription timing jitter is roughly a 30th note, so a
# 16th-note grid mostly encodes noise; 8ths give a far more readable draft.
SUBDIVISION = 2
FINE_SUBDIVISION = 4
MIN_SLOTS = 1


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


def quantize_notes(notes: list[Note], grid: BeatGrid,
                   subdivision: int = SUBDIVISION) -> list[GridNote]:
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


def trim_leading_rest(parts: dict, drums_key: str = "drums",
                      beats_per_bar: int = 4) -> dict:
    """Remove whole empty bars at the start, shifting every part by the same amount.

    Applied per part this would desynchronise the score, so it takes the whole
    set at once and uses one shift for all of them.
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
        return parts

    bar = float(beats_per_bar)
    shift = np.floor(min(starts) / bar) * bar
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
