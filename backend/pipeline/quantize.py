"""Snap raw seconds-based notes onto the musical grid found by analyze.py."""
from __future__ import annotations

import bisect

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
# Thirds of a beat. Without these no triplet can be written down at all: every
# one of them lands a third of a beat from the nearest slot on any grid built
# by halving, which is the largest error a grid can make, and the notes that
# collide after being pushed there are folded together by `_merge_duplicates`.
# Measured on a solo piano arrangement whose middle section is in triplets, the
# onsets sat 44 ms from the 8th-note grid and 5 ms from the 6-per-beat one.
TRIPLET_SUBDIVISION = 3
# Both halves and thirds. Needed because a passage rarely switches over
# cleanly — it carries ordinary quavers alongside the triplets, and a pure
# triplet grid cannot place those either.
MIXED_SUBDIVISION = 6
# Ordered coarse to fine; `choose_subdivision` walks it in this order.
_SUBDIVISIONS = (SUBDIVISION, TRIPLET_SUBDIVISION, FINE_SUBDIVISION,
                 MIXED_SUBDIVISION)
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
        # Extrapolate at the spacing the grid itself runs at, not at
        # `analysis.seconds_per_beat`: the header tempo is refined against the
        # onsets afterwards and is deliberately a fraction of a percent away
        # from the tracked spacing, so using it here would put a small seam
        # between the extension and the beats it is extending.
        if len(beats) >= 2:
            spb = float(np.median(np.diff(beats)))
        else:
            spb = analysis.seconds_per_beat
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
    """Pick the grid the onsets already fit, from 8ths, triplets, 16ths or sixths.

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

    This picks one grid for everything it is given. Use `subdivision_plan` for
    a part whose writing changes partway through — that is the common case on
    anything longer than a phrase, and it is what the pipeline calls.
    """
    if len(notes) < 8:
        return SUBDIVISION
    positions = np.array([grid.to_beats(n.start) for n in notes], dtype=float)
    return _pick_subdivision(positions, margin)


# A finer grid has to come within this of the best candidate's residual before
# it is preferred for being coarser.
_SUBDIVISION_TOL = 0.10


def _pick_subdivision(positions: np.ndarray, margin: float) -> int:
    """Which grid these beat positions sit on, or 8ths if none fits better."""
    if len(positions) < 8:
        return SUBDIVISION

    def residual(subdivision: int) -> float:
        scaled = positions * subdivision
        return float(np.mean(np.abs(scaled - np.round(scaled)) / subdivision))

    scores = {s: residual(s) for s in _SUBDIVISIONS}
    base = scores[SUBDIVISION]
    if base <= 1e-9:
        return SUBDIVISION
    finest = min(scores, key=scores.get)
    if (base - scores[finest]) / base <= margin:
        return SUBDIVISION
    # Then fall back to the coarsest grid that is essentially as good — but
    # only among grids the winner actually contains, so that a near-tie is
    # broken toward the simpler notation without changing what can be written.
    # 16ths and sixths are not interchangeable that way: neither contains the
    # other, and only sixths can place a triplet at all. They still score
    # within a few percent of each other on a passage of sixths measured
    # against the tracked grid (0.0300 against 0.0311 here), because the grid's
    # own error floors both — so a plain "prefer the coarser" comparison
    # between them hands triplet material to a grid that cannot hold it, and
    # every third note lands a twelfth of a beat away and merges with its
    # neighbour. Measured with that grid error removed, the same passage sits
    # 2.2 ms from sixths and 14.1 ms from 16ths.
    for candidate in _SUBDIVISIONS:
        if finest % candidate == 0 and \
                scores[candidate] <= scores[finest] * (1 + _SUBDIVISION_TOL):
            return candidate
    return finest


# How much music one subdivision decision covers, in quarter-lengths. Sized
# like SPLIT_WINDOW and for the same reason: long enough that a handful of
# stray onsets cannot flip a section, short enough to follow a piece that
# changes character partway through.
SUBDIVISION_WINDOW = 12.0


def subdivision_plan(onsets: np.ndarray, grid: BeatGrid,
                     margin: float = FINE_MARGIN,
                     window: float = SUBDIVISION_WINDOW) -> list[tuple[float, int]]:
    """Per-section subdivisions for a part, decided from raw audio onsets.

    One subdivision for a whole part is the same mistake `split_hands` already
    documents for one split point: it has to be wrong wherever the music
    changes. Measured on the arrangement that prompted this, the first 40
    seconds and the middle 40 sit exactly on quavers while three other stretches
    sit on sixths — and averaged over the part the sixths improve on quavers by
    42%, under the margin, so a single decision writes the whole piece in
    quavers and mangles every triplet in it.

    Decided from audio onsets rather than transcribed notes because the model's
    own timing jitter is comparable to the slot spacing being tested; see
    `analyze.onset_times`.

    Returns (start in quarter-lengths, subdivision) pairs, ascending.
    """
    if len(onsets) == 0:
        return [(0.0, SUBDIVISION)]
    positions = np.array([grid.to_beats(float(t)) for t in onsets], dtype=float)
    positions = positions[np.isfinite(positions)]
    if len(positions) < 8:
        return [(0.0, SUBDIVISION)]

    buckets = np.floor(positions / window).astype(int)
    first, last = int(buckets.min()), int(buckets.max())
    indices = list(range(first, last + 1))
    grouped = {i: positions[buckets == i] for i in indices}

    picks: dict[int, int] = {}
    previous = SUBDIVISION
    for i in indices:
        # A sparse window has no opinion of its own; carrying the previous
        # decision forward beats resetting a triplet section to quavers because
        # one bar of it happened to be quiet.
        picks[i] = (_pick_subdivision(grouped[i], margin)
                    if len(grouped[i]) >= 8 else previous)
        previous = picks[i]

    # A single duple window between two divided ones is almost always the same
    # passage, momentarily thinner. Bridging it first stops one quiet bar
    # splitting a section into two that then disagree with each other.
    for i in indices[1:-1]:
        if picks[i] == SUBDIVISION and SUBDIVISION not in (picks[i - 1], picks[i + 1]):
            picks[i] = picks[i - 1]

    # Then settle each run of divided windows on one grid, pooling its onsets.
    # Window by window the same passage comes out 16ths here and sixths there —
    # they score within a few percent of each other on material that mixes
    # halves and thirds — and alternating between them mid-phrase is both wrong
    # and unreadable. Pooled, the run has enough evidence to answer once.
    plan: list[tuple[float, int]] = []
    i = first
    while i <= last:
        if picks[i] == SUBDIVISION:
            i += 1
            continue
        start = i
        while i <= last and picks[i] != SUBDIVISION:
            i += 1
        pooled = np.concatenate([grouped[j] for j in range(start, i)])
        settled = _pick_subdivision(pooled, margin)
        if settled != SUBDIVISION:
            plan.extend(_beat_divisions(pooled, start * window, i * window,
                                        settled))
            if i <= last:
                plan.append((i * window, SUBDIVISION))
    plan.sort()
    if not plan or plan[0][0] > 0:
        plan.insert(0, (0.0, SUBDIVISION))
    return plan


# How far an onset may sit from a candidate division before that division is
# judged not to describe the beat. A little wider than transcription jitter.
_BEAT_FIT = 0.12


def _beat_divisions(positions: np.ndarray, start: float, end: float,
                    settled: int) -> list[tuple[float, int]]:
    """Split a divided run into one division per beat.

    A section written in sixths does not divide every beat into six. Some of
    its beats are plain quavers, some are triplets, and notation follows the
    beat: nobody writes a quaver and a triplet quaver inside the same beat, and
    the formats cannot express it either. Left mixed, music21 brackets the bar
    by guesswork and emits tuplets like 24:17 with unbalanced start/stop tags —
    MusicXML that reads as valid until MuseScore crashes rendering its audio.

    So each beat is given the coarsest division its own onsets actually fit,
    which keeps every beat internally uniform while still letting the section
    place notes on sixths where it needs them.
    """
    # Only divisions the settled grid contains, and never sixths themselves: a
    # beat divided into six is the one case that cannot be written down beside
    # anything else, because half of its slots read as quavers and the other
    # half as triplet quavers, and no format has a way to say both at once.
    # Sixths stay in `_pick_subdivision`, where they are the evidence that a
    # section is divided at all; here that section is written a beat at a time
    # in halves or in thirds, which is how the music is played and read.
    candidates = [d for d in (SUBDIVISION, TRIPLET_SUBDIVISION, FINE_SUBDIVISION)
                  if settled % d == 0]
    def deviation(offsets: np.ndarray, candidate: int) -> float:
        scaled = offsets * candidate
        return float(np.max(np.abs(scaled - np.round(scaled))) / candidate)

    out: list[tuple[float, int]] = []
    previous = None
    beat = float(np.floor(start))
    while beat < end:
        inside = positions[(positions >= beat) & (positions < beat + 1.0)]
        if len(inside):
            offsets = inside - beat
            # The coarsest division that actually describes this beat, and
            # failing that the one that comes closest. Never `settled` itself:
            # every beat has to end up on a grid the engraver can write, so a
            # beat that fits none of them is approximated rather than left on a
            # division that would poison the bar around it.
            fits = [c for c in candidates if deviation(offsets, c) <= _BEAT_FIT]
            chosen = fits[0] if fits else min(candidates,
                                              key=lambda c: deviation(offsets, c))
        else:
            chosen = previous if previous is not None else candidates[0]
        if chosen != previous:
            out.append((beat, chosen))
        previous = chosen
        beat += 1.0
    return out


def subdivision_at(plan: list[tuple[float, int]], offset: float,
                   default: int = SUBDIVISION) -> int:
    """The subdivision in force at a given quarter-length offset.

    Bisected rather than scanned: the plan carries an entry per beat wherever
    the writing changes, and this is called once per note.
    """
    if not plan:
        return default
    index = bisect.bisect_right([start for start, _ in plan], offset + 1e-9) - 1
    return plan[max(0, index)][1]


def quantize_notes(notes: list[Note], grid: BeatGrid,
                   subdivision: int | list[tuple[float, int]] | None = None,
                   ) -> list[GridNote]:
    """Snap notes onto the grid.

    `subdivision` takes one grid for the whole part, a plan from
    `subdivision_plan` to vary it by section, or None to choose one per part.
    """
    if subdivision is None:
        subdivision = choose_subdivision(notes, grid)
    plan = subdivision.copy() if isinstance(subdivision, list) else None

    out: list[GridNote] = []
    for n in notes:
        # Locate the note before snapping it, so the section it belongs to is
        # decided by where it was played rather than by where it ends up.
        local = (subdivision_at(plan, grid.to_beats(n.start)) if plan
                 else subdivision)
        step = 1.0 / local
        start = grid.snap(n.start, local)
        end = grid.snap(n.end, local)
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


# Semitones one hand covers without repositioning. A window narrower than this
# is one hand playing, not two hands close together.
HAND_SPAN = 14
# Where a split may sit. Outside this it is not dividing hands, it is slicing
# through the middle of one of them.
SPLIT_RANGE = (40, 79)
# Context used to place each split, in quarter-lengths — two bars of 4/4.
SPLIT_WINDOW = 8.0
# A gap narrower than this is spacing inside one hand's chord, not the space
# between two hands.
MIN_HAND_GAP = 3
# Each side of a candidate gap has to carry at least this share of the window's
# notes, and at least two of them.
MIN_HAND_SHARE = 0.15


def _window_split(notes: list[GridNote], default: int) -> int:
    """Where the two hands part company in one window of music.

    The hands are separated by a band of pitch nobody is playing, so the split
    goes in the widest such band — but only where there is real writing on both
    sides of it. That last condition is what makes it survive contact with a
    transcription: a couple of stray detections below the left hand open a gap
    wider than the true one, and taking the widest gap unconditionally puts the
    split there and moves an entire hand onto the wrong staff. Measured on
    `roaming_piano`, one window had its hands 17 semitones apart and the split
    landed on a false 10-semitone gap held up by two spurious notes.

    Weighting a pitch histogram by note length and splitting at its valley was
    tried instead and is worse (solo_piano 1.000 -> 0.800): held chords in the
    left hand outweigh a single-note melody in the right, so the density peak
    sits on the left hand and drags the valley up into the melody.
    """
    if len(notes) < 2:
        return default
    pitches = sorted(n.pitch for n in notes)
    if pitches[-1] - pitches[0] <= HAND_SPAN:
        # One hand reaches all of it, so everything belongs to whichever hand
        # plays in this register — the other is resting, and an empty staff is
        # the honest notation for that.
        centre = pitches[len(pitches) // 2]
        return pitches[0] if centre >= 60 else pitches[-1] + 1

    floor = max(2, int(len(pitches) * MIN_HAND_SHARE))
    distinct = sorted(set(pitches))
    best = None
    for lower, upper in zip(distinct, distinct[1:]):
        size = upper - lower
        if size < MIN_HAND_GAP:
            continue
        below = sum(1 for p in pitches if p <= lower)
        if below < floor or len(pitches) - below < floor:
            continue
        key = (size, -abs(upper - 60))
        if best is None or key > best[0]:
            best = (key, upper)
    if best is None:
        return default
    return int(np.clip(best[1], *SPLIT_RANGE))


def drop_notes_shared_with(notes: list[GridNote], other: list[GridNote],
                           ) -> list[GridNote]:
    """Remove notes another staff is already carrying, at the same slot and pitch.

    The accompaniment mix contains the bass stem, so a bass line that gets its
    own staff is transcribed twice and printed twice. Measured on the
    `full_band` case, 92% of the bass part reappeared inside the piano and took
    the piano's precision down to 0.53; removing the overlap afterwards brings
    it to 0.94 while keeping the low piano notes that the bass is *not*
    playing, which excluding the whole stem would have deleted along with it.
    """
    if not other:
        return notes
    taken = {(round(n.offset, 3), n.pitch) for n in other}
    return [n for n in notes if (round(n.offset, 3), n.pitch) not in taken]


def split_hands(notes: list[GridNote], split_pitch: int = 60,
                window: float = SPLIT_WINDOW) -> tuple[list[GridNote], list[GridNote]]:
    """Divide accompaniment notes into right hand / left hand, following the music.

    One split point chosen once for the whole piece only works while the music
    stays in one register. It does not: a figure comes back an octave up, the
    left hand walks down into the bass, and the pitch that was the left hand's
    top note in one section is the right hand's bottom note in the next. A
    global threshold has to be wrong in one of those places — measured on the
    `roaming_piano` case, it put a third of the left hand on the wrong staff.

    So the split is placed per window of music instead, from the gap that
    actually separates the hands there, and then smoothed over time so it does
    not flicker between neighbouring windows and scatter a held texture across
    both staves.
    """
    if not notes:
        return [], []
    ordered = sorted(notes, key=lambda n: n.offset)
    offsets = np.array([n.offset for n in ordered], dtype=float)
    pitches = [n.pitch for n in ordered]

    fallback = int(np.clip(int(np.median(pitches)), 52, 67))
    if abs(fallback - split_pitch) <= 7:
        fallback = split_pitch

    half = window / 2.0
    raw = []
    for offset in offsets:
        lo = int(np.searchsorted(offsets, offset - half, side="left"))
        hi = int(np.searchsorted(offsets, offset + half, side="right"))
        raw.append(_window_split(ordered[lo:hi], fallback))

    # Median-smooth so a single dense chord cannot drag the boundary for its
    # neighbours; the split should move with the writing, not with each attack.
    splits = []
    span = 5
    for i in range(len(raw)):
        lo, hi = max(0, i - span), min(len(raw), i + span + 1)
        splits.append(int(np.median(raw[lo:hi])))

    right = [n for n, split in zip(ordered, splits) if n.pitch >= split]
    left = [n for n, split in zip(ordered, splits) if n.pitch < split]
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
