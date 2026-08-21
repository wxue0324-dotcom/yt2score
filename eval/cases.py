"""Ground-truth test pieces, rendered to audio by MuseScore.

Synthetic material is cleaner than a real recording, so absolute scores here
run optimistic. What it measures reliably is *relative* change: whether a
pipeline edit helped or hurt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from music21 import (chord, clef, dynamics, instrument, key, layout, meter,
                     note, stream, tempo)


@dataclass
class Case:
    name: str
    score: stream.Score
    tempo_bpm: float
    key_name: str
    # part label -> [(onset_ql, duration_ql, midi_pitch), …]
    truth: dict[str, list[tuple[float, float, int]]] = field(default_factory=dict)


def _part(events, name, clef_obj, inst, ks, ts, mm):
    p = stream.Part(id=name)
    p.partName = name
    p.insert(0, inst)
    p.insert(0, clef_obj)
    p.insert(0, ks)
    p.insert(0, ts)
    p.insert(0, mm)
    for off, dur, pitches in events:
        el = (note.Note(pitches[0], quarterLength=dur) if len(pitches) == 1
              else chord.Chord(list(pitches), quarterLength=dur))
        p.insert(off, el)
    p.makeNotation(inPlace=True)
    return p


def _truth(events):
    return [(off, dur, pit) for off, dur, pitches in events for pit in pitches]


def solo_melody() -> Case:
    """A single sung-style line: the easiest case, and the one users care most about."""
    bpm, tonic = 96.0, "C"
    notes = [60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 67, 72, 71, 69, 67, 65, 64, 62, 60]
    durs = [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 4]
    events, off = [], 0.0
    for pitch, dur in zip(notes, durs):
        events.append((off, float(dur), (pitch,)))
        off += dur

    ks, ts = key.Key(tonic, "major"), meter.TimeSignature("4/4")
    mm = tempo.MetronomeMark(number=bpm)
    sc = stream.Score()
    sc.insert(0, _part(events, "Melody", clef.TrebleClef(),
                       instrument.Soprano(), ks, ts, mm))
    return Case("solo_melody", sc, bpm, "C major", {"melody": _truth(events)})


def solo_piano() -> Case:
    """Melody over block chords — tests polyphony and hand splitting."""
    bpm, tonic = 84.0, "F"
    rh = [(0,2,(69,)), (2,2,(72,)), (4,2,(74,)), (6,2,(72,)),
          (8,2,(69,)), (10,2,(65,)), (12,4,(67,)),
          (16,2,(69,)), (18,2,(72,)), (20,2,(77,)), (22,2,(76,)), (24,4,(74,))]
    # B-flat, not G major: a B natural here would make the piece C major and
    # the case would be testing the wrong key.
    lh = [(0,4,(53,57,60)), (4,4,(48,52,55)), (8,4,(50,53,57)), (12,4,(46,50,53)),
          (16,4,(53,57,60)), (20,4,(48,52,55)), (24,4,(53,57,60))]

    ks, ts = key.Key(tonic, "major"), meter.TimeSignature("4/4")
    mm = tempo.MetronomeMark(number=bpm)
    sc = stream.Score()
    rp = _part(rh, "Piano RH", clef.TrebleClef(), instrument.Piano(), ks, ts, mm)
    lp = _part(lh, "Piano LH", clef.BassClef(), instrument.Piano(), ks, ts, mm)
    sc.insert(0, rp); sc.insert(0, lp)
    sc.insert(0, layout.StaffGroup([rp, lp], symbol="brace", barTogether=True))
    return Case("solo_piano", sc, bpm, "F major",
                {"piano": _truth(rh) + _truth(lh),
                 "piano_rh": _truth(rh), "piano_lh": _truth(lh)})


def full_band() -> Case:
    """Voice + piano + bass + drums: the realistic, and hardest, case."""
    bpm, tonic = 110.0, "G"
    mel = [(0,1,(67,)), (1,1,(69,)), (2,2,(71,)), (4,1,(74,)), (5,1,(71,)),
           (6,2,(69,)), (8,1,(67,)), (9,1,(69,)), (10,2,(71,)), (12,4,(67,)),
           (16,1,(74,)), (17,1,(76,)), (18,2,(78,)), (20,2,(76,)), (22,2,(74,)),
           (24,4,(71,))]
    pno = [(0,4,(55,59,62)), (4,4,(60,64,67)), (8,4,(57,60,64)), (12,4,(50,54,57)),
           (16,4,(55,59,62)), (20,4,(60,64,67)), (24,4,(55,59,62))]
    bas = [(0,2,(43,)), (2,2,(43,)), (4,2,(48,)), (6,2,(48,)),
           (8,2,(45,)), (10,2,(45,)), (12,2,(38,)), (14,2,(38,)),
           (16,2,(43,)), (18,2,(43,)), (20,2,(48,)), (22,2,(48,)), (24,4,(43,))]

    ks, ts = key.Key(tonic, "major"), meter.TimeSignature("4/4")
    mm = tempo.MetronomeMark(number=bpm)
    sc = stream.Score()
    sc.insert(0, _part(mel, "Vocal", clef.TrebleClef(), instrument.Soprano(), ks, ts, mm))
    rp = _part(pno, "Piano RH", clef.TrebleClef(), instrument.Piano(), ks, ts, mm)
    sc.insert(0, rp)
    sc.insert(0, _part(bas, "Bass", clef.BassClef(),
                       instrument.ElectricBass(), ks, ts, mm))
    return Case("full_band", sc, bpm, "G major",
                {"melody": _truth(mel), "piano": _truth(pno), "bass": _truth(bas)})


def roaming_piano() -> Case:
    """Two hands whose register moves through the piece.

    `solo_piano` keeps the right hand above C4 and the left below it from start
    to finish, so one split point chosen once for the whole piece gets it right
    — which is what the pipeline does, and why the case never showed the
    problem. Real playing moves: the same passage returns an octave up, the
    left hand walks down into the bass, and a note that belonged to the left
    hand in bar 4 belongs to the right in bar 20.

    Here the left hand of the middle section (60–69) and the right hand of the
    last (60–67) occupy the same octave, so **no single threshold can score
    well on both**. Splitting has to follow the music.
    """
    bpm, tonic = 90.0, "C"
    rh_figure = [(0, 1, 72), (1, 1, 74), (2, 1, 76), (3, 1, 77), (4, 1, 76),
                 (5, 1, 74), (6, 2, 72), (8, 1, 72), (9, 1, 76), (10, 1, 79),
                 (11, 1, 77), (12, 4, 76)]
    lh_figure = [(0, 2, (48, 52, 55)), (2, 2, (48, 52, 55)), (4, 2, (50, 53, 57)),
                 (6, 2, (50, 53, 57)), (8, 2, (48, 52, 55)), (10, 2, (48, 52, 55)),
                 (12, 4, (46, 50, 53))]

    rh, lh = [], []
    for section, shift in enumerate((0, 12, -12)):      # 原位、高八度、低八度
        bar = section * 16.0
        for off, dur, pitch in rh_figure:
            rh.append((bar + off, float(dur), (pitch + shift,)))
        for off, dur, pitches in lh_figure:
            lh.append((bar + off, float(dur), tuple(p + shift for p in pitches)))

    ks, ts = key.Key(tonic, "major"), meter.TimeSignature("4/4")
    mm = tempo.MetronomeMark(number=bpm)
    sc = stream.Score()
    rp = _part(rh, "Piano RH", clef.TrebleClef(), instrument.Piano(), ks, ts, mm)
    lp = _part(lh, "Piano LH", clef.BassClef(), instrument.Piano(), ks, ts, mm)
    sc.insert(0, rp); sc.insert(0, lp)
    sc.insert(0, layout.StaffGroup([rp, lp], symbol="brace", barTogether=True))
    return Case("roaming_piano", sc, bpm, "C major",
                {"piano": _truth(rh) + _truth(lh),
                 "piano_rh": _truth(rh), "piano_lh": _truth(lh)})


def soft_and_short() -> Case:
    """Quiet notes and sixteenth notes — the two things a draft quietly loses.

    The other cases are all crotchets and minims at one dynamic, so every stage
    that discards material by *duration* or by *velocity* scores untouched on
    them. This case separates the two conditions so a change can be read:

        bars 1, 6   loud crotchets          — the control
        bars 2, 5   eighths, loud then pp   — dynamics alone
        bar  4      loud semiquaver run     — brevity alone
        bars 3, 7   pp, and pp semiquavers  — both together, the worst case

    Tempo is deliberately slow: at 80 BPM a semiquaver lasts 188ms, comfortably
    above the 100ms `minimum_note_length` the vocal profile imposes, so anything
    lost here is lost by choice rather than by that floor.
    """
    bpm, tonic = 80.0, "C"
    events = []
    #                    bar 1 — f, crotchets (control)
    for i, pitch in enumerate([60, 62, 64, 65]):
        events.append((float(i), 1.0, (pitch,)))
    #                    bar 2 — f, quavers
    for i, pitch in enumerate([67, 65, 64, 62, 60, 62, 64, 65]):
        events.append((4.0 + i * 0.5, 0.5, (pitch,)))
    #                    bar 3 — pp, crotchets
    for i, pitch in enumerate([67, 69, 71, 72]):
        events.append((8.0 + i, 1.0, (pitch,)))
    #                    bar 4 — f, semiquaver run then a minim
    for i, pitch in enumerate([72, 71, 69, 67, 65, 64, 62, 60]):
        events.append((12.0 + i * 0.25, 0.25, (pitch,)))
    events.append((14.0, 2.0, (60,)))
    #                    bar 5 — pp, quavers
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        events.append((16.0 + i * 0.5, 0.5, (pitch,)))
    #                    bar 6 — f, crotchets (control)
    for i, pitch in enumerate([72, 71, 69, 67]):
        events.append((20.0 + i, 1.0, (pitch,)))
    #                    bar 7 — pp, semiquaver run then a minim
    for i, pitch in enumerate([67, 69, 71, 72, 71, 69, 67, 65]):
        events.append((24.0 + i * 0.25, 0.25, (pitch,)))
    events.append((26.0, 2.0, (64,)))
    #                    bar 8 — closing semibreve
    events.append((28.0, 4.0, (60,)))

    ks, ts = key.Key(tonic, "major"), meter.TimeSignature("4/4")
    mm = tempo.MetronomeMark(number=bpm)
    part = _part(events, "Melody", clef.TrebleClef(), instrument.Soprano(),
                 ks, ts, mm)
    # MuseScore synthesises from these, which is what puts a real level
    # difference into the rendered audio — a velocity set on the note objects
    # would not survive the MusicXML round-trip.
    for offset, mark in ((0.0, "f"), (8.0, "pp"), (12.0, "f"),
                         (16.0, "pp"), (20.0, "f"), (24.0, "pp")):
        part.insert(offset, dynamics.Dynamic(mark))

    sc = stream.Score()
    sc.insert(0, part)
    return Case("soft_and_short", sc, bpm, "C major", {"melody": _truth(events)})


# Offset ranges within `soft_and_short`, so recall can be reported per condition
# instead of averaged into one number that hides which kind of note went missing.
SOFT_SHORT_BANDS = {
    "響·四分": [(0.0, 4.0), (20.0, 24.0)],
    "響·八分": [(4.0, 8.0)],
    "響·十六分": [(12.0, 14.0)],
    "弱·四分": [(8.0, 12.0)],
    "弱·八分": [(16.0, 20.0)],
    "弱·十六分": [(24.0, 26.0)],
}


def isolate(case: Case, part_names: list[str]) -> stream.Score:
    """A score holding only the named staves — used to render a part alone."""
    import copy
    sub = stream.Score()
    for p in case.score.parts:
        if p.partName in part_names:
            sub.insert(0, copy.deepcopy(p))
    return sub


# Ground-truth part -> the staves that play it, for isolated rendering.
PART_STAVES = {
    "melody": ["Melody", "Vocal"],
    "piano": ["Piano RH", "Piano LH"],
    "piano_rh": ["Piano RH"],
    "piano_lh": ["Piano LH"],
    "bass": ["Bass"],
}

ALL_CASES = (solo_melody, solo_piano, full_band, soft_and_short,
              roaming_piano)
