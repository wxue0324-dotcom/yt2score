"""Ground-truth pieces for tempo estimation, built around the octave traps.

The three cases in `cases.py` all have notes moving at roughly the beat, so any
estimator that finds *a* periodicity gets them right. Tempo fails on the
material they leave out: a slow piece carrying a fast subdivision reads as fast,
a fast piece written in long notes reads as slow. Each case here is a piece
where the note rate and the notated beat deliberately disagree, so an estimator
that reports note density instead of pulse scores visibly wrong.

Rendered by MuseScore at a fixed tempo, so the truth is exact — unlike a real
recording, where the "true" tempo is itself a judgement call.
"""
from __future__ import annotations

from music21 import clef, instrument, key, meter, stream, tempo

from cases import Case, _part


def _score(parts, ks, ts, bpm):
    sc = stream.Score()
    mm = tempo.MetronomeMark(number=bpm)
    for events, name, cl, inst in parts:
        sc.insert(0, _part(events, name, cl, inst, ks, ts, mm))
    return sc


def slow_dense() -> Case:
    """♩=60 under a continuous sixteenth-note figure — 240 onsets per minute.

    The trap: every sixteenth is a note, so a grid four times too fast lands on
    a note at every point and scores perfectly on onset coverage. Reporting the
    note rate here gives 240 (folded to 120), i.e. exactly double.
    """
    bpm = 60.0
    rh, lh = [], []
    melody = [72, 74, 76, 74, 72, 71, 69, 67]      # one note per bar
    figure = [48, 55, 60, 64]                       # arpeggio, four per beat
    for bar in range(8):
        rh.append((bar * 4.0, 4.0, (melody[bar],)))
        for beat in range(4):
            for i, pitch in enumerate(figure):
                lh.append((bar * 4.0 + beat + i * 0.25, 0.25, (pitch,)))
    sc = _score([(rh, "Piano RH", clef.TrebleClef(), instrument.Piano()),
                 (lh, "Piano LH", clef.BassClef(), instrument.Piano())],
                key.Key("C", "major"), meter.TimeSignature("4/4"), bpm)
    return Case("slow_dense", sc, bpm, "C major")


def fast_driving() -> Case:
    """♩=168 with the pulse played on every beat, melody in eighths.

    An earlier version of this case had the melody in half and whole notes and
    called the truth 168. That was unfair: nothing in the audio articulated the
    fast pulse, so hearing it at 84 was the better answer, not an error. Here a
    staccato chord marks every beat, so 168 is genuinely recoverable — and the
    eighth-note melody pushes the onset rate to ~250/min, which folds to 125.
    The failure this catches is halving a genuinely fast piece to a comfortable
    one.
    """
    bpm = 168.0
    mel, comp = [], []
    tune = [67, 69, 71, 72, 74, 72, 71, 69]
    for bar in range(8):
        base = tune[bar]
        mel += [(bar * 4.0, 1.0, (base,)),
                (bar * 4.0 + 1, 0.5, (base + 2,)),
                (bar * 4.0 + 1.5, 0.5, (base + 4,)),
                (bar * 4.0 + 2, 1.0, (base + 5,)),
                (bar * 4.0 + 3, 0.5, (base + 4,)),
                (bar * 4.0 + 3.5, 0.5, (base + 2,))]
        for beat in range(4):                       # the pulse itself
            comp.append((bar * 4.0 + beat, 0.5, (48, 55)))
    sc = _score([(mel, "Melody", clef.TrebleClef(), instrument.Soprano()),
                 (comp, "Piano LH", clef.BassClef(), instrument.Piano())],
                key.Key("G", "major"), meter.TimeSignature("4/4"), bpm)
    return Case("fast_driving", sc, bpm, "G major")


def triplet_slow() -> Case:
    """♩=76 with eighth-note triplets — 228 onsets per minute.

    Triple subdivision, so the wrong answer is 3× rather than 2× or 4×. An
    estimator that only folds by powers of two cannot even represent the error.
    """
    bpm = 76.0
    mel, off = [], 0.0
    pattern = [72, 74, 76, 77, 76, 74]
    for bar in range(8):
        for beat in range(4):
            for i in range(3):
                mel.append((off, 1 / 3, (pattern[(beat * 3 + i) % len(pattern)],)))
                off += 1 / 3
    chords = [(b * 4.0, 4.0, (48, 52, 55)) for b in range(8)]
    sc = _score([(mel, "Melody", clef.TrebleClef(), instrument.Flute()),
                 (chords, "Piano LH", clef.BassClef(), instrument.Piano())],
                key.Key("C", "major"), meter.TimeSignature("4/4"), bpm)
    return Case("triplet_slow", sc, bpm, "C major")


def waltz_fast() -> Case:
    """3/4 at ♩=150, accent firmly on beat 1 — tests the bar grouping too.

    A three-grouping is the case where a estimator that assumes duple metre
    quietly writes the right notes into the wrong bars.
    """
    bpm = 150.0
    mel, bass = [], []
    tune = [72, 74, 76, 77, 79, 77, 76, 74]
    for bar in range(8):
        mel.append((bar * 3.0, 1.0, (tune[bar],)))
        mel.append((bar * 3.0 + 1, 1.0, (tune[bar] + 4,)))
        mel.append((bar * 3.0 + 2, 1.0, (tune[bar] + 7,)))
        bass.append((bar * 3.0, 1.0, (36, 43)))      # downbeat carries the accent
    sc = _score([(mel, "Melody", clef.TrebleClef(), instrument.Violin()),
                 (bass, "Piano LH", clef.BassClef(), instrument.Piano())],
                key.Key("C", "major"), meter.TimeSignature("3/4"), bpm)
    return Case("waltz_fast", sc, bpm, "C major")


def syncopated_mid() -> Case:
    """♩=100 with the melody pushed onto the off-beats.

    Onsets sit consistently between beats, so a grid locked to the notes is
    half a beat out of phase with the notated one. Tempo should survive that;
    the beat *phase* is what suffers.
    """
    bpm = 100.0
    mel, chords = [], []
    tune = [67, 69, 71, 72, 71, 69, 67, 65]
    for bar in range(8):
        mel.append((bar * 4.0 + 0.5, 1.5, (tune[bar],)))
        mel.append((bar * 4.0 + 2.5, 1.5, (tune[bar] + 3,)))
        chords.append((bar * 4.0, 2.0, (48, 52, 55)))
        chords.append((bar * 4.0 + 2, 2.0, (47, 50, 55)))
    sc = _score([(mel, "Melody", clef.TrebleClef(), instrument.Soprano()),
                 (chords, "Piano LH", clef.BassClef(), instrument.Piano())],
                key.Key("C", "major"), meter.TimeSignature("4/4"), bpm)
    return Case("syncopated_mid", sc, bpm, "C major")


# case -> beats per bar, the other thing analyze() has to get right
BEATS_PER_BAR = {
    "slow_dense": 4, "fast_driving": 4, "triplet_slow": 4,
    "waltz_fast": 3, "syncopated_mid": 4,
}

TEMPO_CASES = (slow_dense, fast_driving, triplet_slow, waltz_fast, syncopated_mid)
