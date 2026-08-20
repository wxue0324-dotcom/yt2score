"""Render an engraved score back to audio, so you can hear what was transcribed.

MuseScore does the synthesis: it already ships the soundfont and knows each
part's instrument, which beats wiring a synthesiser into the browser and
sounds considerably better than one.
"""
from __future__ import annotations

import copy
import re
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from music21 import instrument, layout, stream

# Staves that belong on one playable instrument rather than their own track.
_GROUPED = {
    "Piano RH": "鋼琴",
    "Piano LH": "鋼琴",
}
_PART_LABEL = {
    "Vocal": "主唱旋律",
    "Bass": "貝斯",
    "Drums": "鼓組",
}

# Instruments swapped in for playback only — the engraved score keeps its own.
# MuseScore synthesises a voice as a vague sustained "aah" with no attack, which
# is close to useless for checking whether a transcribed pitch is right. A piano
# gives every note a clear onset, so errors are audible immediately.
_PLAYBACK_INSTRUMENT = {"Vocal": instrument.Piano}


def _use_playback_instruments(score: stream.Score) -> stream.Score:
    """Return a copy voiced for listening rather than for printing."""
    playable = copy.deepcopy(score)
    for part in playable.parts:
        replacement = _PLAYBACK_INSTRUMENT.get(part.partName)
        if replacement is None:
            continue
        for existing in list(part.recurse().getElementsByClass(
                instrument.Instrument)):
            try:
                part.remove(existing, recurse=True)
            except Exception:
                pass
        part.insert(0, replacement())
    return playable


def _mscore() -> str | None:
    return shutil.which("mscore") or shutil.which("musescore")


def _render(xml_path: Path, out_path: Path, timeout: int = 600) -> Path | None:
    binary = _mscore()
    if not binary:
        return None
    try:
        subprocess.run([binary, "-o", str(out_path), str(xml_path)],
                       capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    # MuseScore 4 aborts on exit after a successful write, so the return code
    # is meaningless — judge by the file.
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


def render_score_audio(score: stream.Score, outdir: Path,
                       basename: str = "score") -> Path | None:
    """Render the whole score to MP3, voiced for listening."""
    outdir.mkdir(parents=True, exist_ok=True)
    if not _mscore():
        return None
    playable = _use_playback_instruments(score)
    xml_path = outdir / f"{basename}__playback.musicxml"
    playable.write("musicxml", fp=str(xml_path))
    rendered = _render(xml_path, outdir / f"{basename}.mp3")
    xml_path.unlink(missing_ok=True)
    return rendered


def _subset(score: stream.Score, part_names: list[str]) -> stream.Score:
    """A new score holding only the named parts, keeping the piano brace."""
    subset = stream.Score()
    if score.metadata is not None:
        subset.insert(0, copy.deepcopy(score.metadata))

    kept = [copy.deepcopy(p) for p in score.parts if p.partName in part_names]
    for part in kept:
        subset.insert(0, part)
    if len(kept) == 2 and all(p.partName in _GROUPED for p in kept):
        subset.insert(0, layout.StaffGroup(kept, name="Piano",
                                           symbol="brace", barTogether=True))
    return subset


def render_part_audio(score: stream.Score, outdir: Path) -> dict[str, Path]:
    """Render one MP3 per instrument, so single lines can be heard alone.

    Piano right and left hand are rendered together — hearing half a piano
    part is not useful.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if not _mscore():
        return {}

    # Collect the staves belonging to each playable instrument.
    groups: dict[str, list[str]] = {}
    for part in score.parts:
        name = part.partName or ""
        label = _GROUPED.get(name) or _PART_LABEL.get(name)
        if label:
            groups.setdefault(label, []).append(name)

    results: dict[str, Path] = {}
    playable = _use_playback_instruments(score)
    for index, (label, part_names) in enumerate(groups.items()):
        subset = _subset(playable, part_names)
        if not subset.parts:
            continue
        stem = f"part{index}"
        xml_path = outdir / f"{stem}.musicxml"
        subset.write("musicxml", fp=str(xml_path))
        rendered = _render(xml_path, outdir / f"{stem}.mp3")
        xml_path.unlink(missing_ok=True)
        if rendered:
            results[label] = rendered
    return results


# Which separated source each rendered part should be auditioned against. The
# piano staves come from bass+other mixed together, so that is what they are
# compared with — not the full mix, which still has the voice and drums in it.
SOURCE_FOR_PART = {
    "主唱旋律": "vocals",
    "貝斯": "bass",
    "鼓組": "drums",
    "鋼琴": "accompaniment",
}
COMPARE_SR = 22050


def _first_onset(path: Path) -> float:
    """Where the sound actually starts — MuseScore pads its renders with silence."""
    y, sr = librosa.load(str(path), sr=COMPARE_SR, mono=True)
    if y.size == 0:
        return 0.0
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    return float(onsets[0]) if len(onsets) else 0.0


def _normalise(y: np.ndarray, target: float = 0.12) -> np.ndarray:
    level = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    return y * (target / level) if level > 1e-6 else y


def render_comparisons(part_audio: dict[str, Path], sources: dict[str, Path],
                       outdir: Path, anchor: float = 0.0,
                       reference: Path | None = None) -> dict[str, Path]:
    """Original in the left ear, transcription in the right, lined up in time.

    Listening to a transcribed line on its own says very little: you are asked
    to remember what the record sounded like and compare from memory, across a
    change of instrument. Side by side in one stereo image, a wrong pitch or a
    dropped note stops being something to recall and becomes something you hear
    immediately, at the moment it happens.

    Comparison is per part and against the *separated* source, because the
    playback of one line against a full band mix differs for reasons that have
    nothing to do with whether the transcription is right.

    `anchor` is the time in the recording that the score's first note refers to,
    and `reference` is the full-score render whose leading silence pins the same
    moment on the other side. Both have to be measured once for the whole score
    rather than per part: a part that enters late — a bass line resting through
    the intro — has its own first note nowhere near the start, so aligning each
    part on its own first onset shifts it by however long it waited. Measured on
    YOASOBI 勇者 that put the bass 9.6 seconds out while the other three parts
    were within a third of a second.

    The two renders still drift apart later in long tracks, because the score
    plays at one notated tempo while the recording does not.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    lead = int(_first_onset(reference) * COMPARE_SR) if reference else 0
    results: dict[str, Path] = {}
    for label, rendered in part_audio.items():
        source = sources.get(SOURCE_FOR_PART.get(label, ""))
        if source is None or not Path(source).exists():
            continue
        right, _ = librosa.load(str(rendered), sr=COMPARE_SR, mono=True)
        left, _ = librosa.load(str(source), sr=COMPARE_SR, mono=True)
        if right.size == 0 or left.size == 0:
            continue

        start = int(max(0.0, anchor) * COMPARE_SR)
        right, left = right[lead:], left[start:]
        width = min(len(right), len(left))
        if width < COMPARE_SR:
            continue

        stereo = np.stack([_normalise(left[:width]),
                           _normalise(right[:width])], axis=-1)
        peak = float(np.max(np.abs(stereo))) or 1.0
        stereo = stereo / peak * 0.95

        safe = re.sub(r"[^\w]+", "_", label).strip("_") or "part"
        wav_path = outdir / f"compare_{safe}.wav"
        sf.write(wav_path, stereo, COMPARE_SR)
        mp3_path = outdir / f"compare_{safe}.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                        "-b:a", "160k", str(mp3_path)], check=False)
        if mp3_path.exists():
            wav_path.unlink(missing_ok=True)
            results[label] = mp3_path
        else:
            results[label] = wav_path
    return results
