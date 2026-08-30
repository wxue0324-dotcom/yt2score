"""Assemble note grids into a music21 score and engrave it with MuseScore."""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from music21 import (bar, chord, clef, instrument, key, layout, metadata,
                     meter, note, stream, tempo)

from .analyze import Analysis
from .quantize import GridNote, subdivision_at

# General MIDI percussion map, so the exported MIDI plays back as real drums.
DRUM_PITCH = {"kick": 36, "snare": 38, "hihat": 42}
DRUM_STAFF_LINE = {"kick": "F3", "snare": "C4", "hihat": "G4"}


# Durations that engrave as a single notehead (plus at most one dot). Anything
# else turns into double-dotted or tied clutter that nobody wants to read.
_CLEAN_DURATIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

# Denominator ceiling when pinning a float offset or gap back to an exact
# fraction. Covers every grid the quantiser uses (halves, thirds, quarters,
# sixths) with room to spare, without letting float noise invent a 1/97.
_DENOM_LIMIT = 48

# The same idea on a grid of thirds. Written as Fractions, not floats, so
# music21 sees an exact 1/3 and prints a 3:2 tuplet instead of trying to infer
# one from 0.3333333.
_TRIPLET_DURATIONS = (Fraction(1, 3), Fraction(2, 3), Fraction(1),
                      Fraction(4, 3), Fraction(2), Fraction(3), Fraction(4))

def _durations_for(subdivision: int) -> tuple:
    """The note values a part quantised at this subdivision can actually land on.

    Chosen per note from the section that note sits in, never pooled across a
    part. Pooling looks harmless and is not: it puts the duple 0.25 and 0.75
    within reach of a section written in sixths, so one beat ends up holding a
    dotted quaver beside a triplet quaver. That is not a rhythm anyone can
    notate, and music21 does not refuse it — it emits a 12:7 tuplet and leaves
    the tuplet brackets unbalanced, which reads as valid MusicXML right up
    until MuseScore crashes trying to play it.
    """
    if subdivision == 3:
        return _TRIPLET_DURATIONS
    return _CLEAN_DURATIONS


def _clean_duration(ql: float, available: float, durations=_CLEAN_DURATIONS,
                    step: Fraction | None = None):
    """Round to a readable note value that still fits before the next attack.

    Anything shorter than the gap leaves a rest behind it, and that rest has to
    land on the grid as well. Rounding a 5/6 gap down to a dotted quaver leaves
    a twelfth of a beat over — not a rest anyone can write, and music21 renders
    it as a 12:7 tuplet with a dangling bracket. So a shorter value is offered
    only when what it leaves over is itself on the grid; otherwise the note
    simply fills the gap.
    """
    fits = [d for d in durations if d <= available + 1e-6]
    if not fits:
        return max(available, 0.25)
    if step:
        exact = Fraction(available).limit_denominator(_DENOM_LIMIT)
        tidy = [d for d in fits
                if (exact - Fraction(d).limit_denominator(_DENOM_LIMIT)) % step == 0]
        # Where nothing leaves a tidy rest — a gap that runs from one beat's
        # division into a differently divided one — keep the ordinary choice
        # rather than inventing a value off the list. Filling the gap exactly
        # was tried and is worse: it produces durations like 5/3, which no
        # notehead spells, and music21 renders them as 6:5 tuplets.
        fits = tidy or fits
    return min(fits, key=lambda d: (abs(d - ql), -d))


def _same_division_until(plan, off: float, local: int) -> float:
    """The offset where the division in force at `off` first changes.

    Bounded by the last boundary in the plan: past that the division never
    changes again, so an unbounded walk would not terminate.
    """
    last = plan[-1][0] if plan else 0.0
    beat = math.floor(off) + 1.0
    while beat <= last and subdivision_at(plan, beat) == local:
        beat += 1.0
    return beat if beat <= last else math.inf


def _collapse(notes: list[GridNote], subdivision=2,
              ) -> list[tuple[object, object, list[int], int]]:
    """Group simultaneous notes into chords and clip each to the next attack.

    Overlapping ragged durations would force music21 into multiple voices, which
    reads badly. Clipping keeps one clean rhythmic line per staff.

    `subdivision` is either one grid for the whole part or a plan from
    `subdivision_plan`, in which case each note is written with the values its
    own section can reach.
    """
    plan = subdivision if isinstance(subdivision, list) else None
    by_offset: dict[float, dict[int, GridNote]] = {}
    for n in notes:
        slot = by_offset.setdefault(n.offset, {})
        if n.pitch not in slot or n.duration > slot[n.pitch].duration:
            slot[n.pitch] = n

    offsets = sorted(by_offset)
    out = []
    for i, off in enumerate(offsets):
        group = list(by_offset[off].values())
        dur = max(g.duration for g in group)
        available = (offsets[i + 1] - off) if i + 1 < len(offsets) else dur
        local = subdivision_at(plan, off) if plan else int(subdivision)
        if plan:
            # Stop the note at the first beat that is divided differently from
            # its own. A gap that spans both divisions has no length either one
            # can spell — the note is rounded down and the rest left over is a
            # twelfth of a beat, which music21 emits as a 12:7 tuplet. Only
            # boundaries where the division actually changes cost anything, so
            # held notes inside a uniform passage are untouched.
            available = min(available, _same_division_until(plan, off, local) - off)
        dur = _clean_duration(min(dur, available), available,
                              _durations_for(local), Fraction(1, local))
        if dur <= 0:
            continue
        vel = max(g.velocity for g in group)
        # Quantising produced this offset by dividing, so it is a whole number
        # of slots but arrives as a float — 1/3 as 0.3333333333. Pin it back to
        # the exact fraction, or music21 measures a triplet bar as fractionally
        # short and pads every one of them with a rest.
        exact = Fraction(off).limit_denominator(_DENOM_LIMIT)
        out.append((exact, dur, sorted(g.pitch for g in group), vel))
    return out


def _build_part(notes: list[GridNote], part_name: str, clef_obj,
                inst=None, max_chord: int = 4,
                subdivision=2) -> stream.Part:
    part = stream.Part(id=part_name)
    part.partName = part_name
    if inst is not None:
        part.insert(0, inst)
    part.insert(0, clef_obj)

    for off, dur, pitches, vel in _collapse(notes, subdivision):
        # Transcription noise shows up as implausibly dense chords; keep the
        # outer voices, which carry the harmony.
        if len(pitches) > max_chord:
            pitches = pitches[:2] + pitches[-(max_chord - 2):]
        if len(pitches) == 1:
            el = note.Note(pitches[0], quarterLength=dur)
        else:
            el = chord.Chord(pitches, quarterLength=dur)
        el.volume.velocity = vel
        part.insert(off, el)
    return part


def _build_drum_part(hits: list[tuple[float, str]]) -> stream.Part:
    part = stream.Part(id="Drums")
    part.partName = "Drums"
    part.insert(0, instrument.Percussion())
    part.insert(0, clef.PercussionClef())

    by_offset: dict[float, set[str]] = {}
    for off, kind in hits:
        by_offset.setdefault(off, set()).add(kind)

    offsets = sorted(by_offset)
    for i, off in enumerate(offsets):
        dur = (offsets[i + 1] - off) if i + 1 < len(offsets) else 0.5
        dur = min(max(dur, 0.25), 1.0)
        lines = [DRUM_STAFF_LINE[k] for k in sorted(by_offset[off])]
        el = note.Note(lines[0], quarterLength=dur) if len(lines) == 1 \
            else chord.Chord(lines, quarterLength=dur)
        part.insert(off, el)
    return part


def build_score(parts_data: dict, analysis: Analysis, title: str,
                composer: str = "", subdivisions: dict | None = None,
                ) -> stream.Score:
    """`subdivisions` maps a part key to the grid it was quantised on, so each
    staff is written with the note values it can actually land on. Parts missing
    from it are treated as ordinary duple material."""
    subdivisions = subdivisions or {}
    score = stream.Score()
    score.insert(0, metadata.Metadata(
        title=title,
        composer=composer or "自動採譜 · yt2score",
    ))

    ts = meter.TimeSignature(f"{analysis.beats_per_bar}/4")
    ks = key.Key(analysis.tonic, analysis.mode)
    mm = tempo.MetronomeMark(number=round(analysis.tempo))

    parts: list[stream.Part] = []

    if parts_data.get("vocal"):
        parts.append(_build_part(parts_data["vocal"], "Vocal",
                                 clef.TrebleClef(), instrument.Vocalist(),
                                 max_chord=1,
                                 subdivision=subdivisions.get("vocal", 2)))

    piano_rh = parts_data.get("piano_rh") or []
    piano_lh = parts_data.get("piano_lh") or []
    if piano_rh or piano_lh:
        # Both hands come from one transcription, so they share a grid.
        piano_sub = subdivisions.get("piano", 2)
        rh = _build_part(piano_rh, "Piano RH", clef.TrebleClef(),
                         instrument.Piano(), subdivision=piano_sub)
        lh = _build_part(piano_lh, "Piano LH", clef.BassClef(),
                         instrument.Piano(), subdivision=piano_sub)
        # A StaffGroup with a brace is what makes MuseScore draw a grand staff.
        parts.extend([rh, lh])
        score.insert(0, layout.StaffGroup([rh, lh], name="Piano",
                                          symbol="brace", barTogether=True))

    if parts_data.get("bass"):
        parts.append(_build_part(parts_data["bass"], "Bass",
                                 clef.BassClef(), instrument.ElectricBass(),
                                 max_chord=2,
                                 subdivision=subdivisions.get("bass", 2)))

    if parts_data.get("drums"):
        parts.append(_build_drum_part(parts_data["drums"]))

    # Every staff must span the same number of bars. Parts are notated
    # independently, so a part whose last note ends early would come out
    # shorter — which MuseScore 4 refuses to render at all.
    bar_length = float(analysis.beats_per_bar)
    longest = max((p.highestTime for p in parts), default=0.0)
    total = max(bar_length, np.ceil(longest / bar_length) * bar_length)

    for part in parts:
        if part.highestTime < total:
            part.insert(part.highestTime,
                        note.Rest(quarterLength=total - part.highestTime))
        part.insert(0, ts)
        part.insert(0, ks)
        part.insert(0, mm)
        part.makeRests(fillGaps=True, inPlace=True, hideRests=False)
        part.makeNotation(inPlace=True)
        for measure in part.getElementsByClass(stream.Measure):
            measure.makeRests(fillGaps=True, inPlace=True, hideRests=False,
                              timeRangeFromBarDuration=True)
        _fix_accidentals(part, ks)
        part.append(bar.Barline("final"))
        score.insert(0, part)

    return score


def _fix_accidentals(part: stream.Part, ks: key.Key) -> None:
    """Re-derive which accidentals to print, given the key signature.

    Notes built from MIDI numbers carry an explicit natural that music21 marks
    for display, so an unedited score prints a ♮ on nearly every notehead.
    Clearing them first lets makeAccidentals show only what the key requires.
    """
    for n in part.recurse().notes:
        for pitch in n.pitches:
            if pitch.accidental is not None and pitch.accidental.name == "natural":
                pitch.accidental = None
    for measure in part.getElementsByClass(stream.Measure):
        measure.makeAccidentals(useKeySignature=ks, overrideStatus=True,
                                inPlace=True)


def export(score: stream.Score, outdir: Path, basename: str = "score") -> dict[str, Path]:
    """Write MusicXML + MIDI, then engrave a PDF if MuseScore is installed."""
    outdir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    xml_path = outdir / f"{basename}.musicxml"
    score.write("musicxml", fp=str(xml_path))
    results["musicxml"] = xml_path

    midi_path = outdir / f"{basename}.mid"
    score.write("midi", fp=str(midi_path))
    results["midi"] = midi_path

    mscore = shutil.which("mscore") or shutil.which("musescore")
    if mscore:
        for fmt in ("pdf", "svg"):
            target = outdir / f"{basename}.{fmt}"
            try:
                subprocess.run(
                    [mscore, "-o", str(target), str(xml_path)],
                    capture_output=True, text=True, timeout=300,
                )
            except subprocess.TimeoutExpired:
                continue
            # MuseScore 4 writes the file correctly and *then* aborts on exit,
            # so its return code says nothing. Trust the file on disk instead.
            # Multi-page SVG is numbered score-1.svg, score-2.svg…
            # Sort by page number, not by name: plain sorted() puts page 10
            # before page 2 and the viewer shows the score out of order.
            def _page_no(path: Path) -> int:
                m = re.fullmatch(rf"{re.escape(basename)}-(\d+)", path.stem)
                return int(m.group(1)) if m else 0

            pages = sorted(
                (p for p in [target, *outdir.glob(f"{basename}-*.{fmt}")]
                 if p.exists() and p.stat().st_size > 0),
                key=_page_no,
            )
            if pages:
                results[fmt] = pages[0]
                if fmt == "svg":
                    results["svg_pages"] = pages
    return results
