"""Render an engraved score back to audio, so you can hear what was transcribed.

MuseScore does the synthesis: it already ships the soundfont and knows each
part's instrument, which beats wiring a synthesiser into the browser and
sounds considerably better than one.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

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
