"""Ground-truth test pieces, rendered to audio by MuseScore.

Synthetic material is cleaner than a real recording, so absolute scores here
run optimistic. What it measures reliably is *relative* change: whether a
pipeline edit helped or hurt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from music21 import chord, clef, instrument, key, layout, meter, note, stream, tempo


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

ALL_CASES = (solo_melody, solo_piano, full_band)
