"""End-to-end: YouTube URL -> stems -> notes -> engraved score files."""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import analyze as analyze_mod
from . import audio as audio_mod
from . import download as download_mod
from . import jianpu as jianpu_mod
from . import score as score_mod
from . import separate as separate_mod
from . import transcribe as transcribe_mod
from .quantize import (BeatGrid, clean_part as clean, leading_rest_shift,
                       quantize_drums, quantize_notes, split_hands,
                       trim_leading_rest)

Progress = Callable[[int, str], None]

STEM_LABEL = {"vocals": "人聲", "drums": "鼓組", "bass": "貝斯", "other": "其他樂器"}


@dataclass
class Result:
    title: str
    uploader: str
    duration: float
    tempo: float
    key_name: str
    key_confidence: float
    beats_per_bar: int
    note_counts: dict
    files: dict          # label -> path relative to the job dir
    warnings: list


def _noop(pct: int, msg: str) -> None:
    pass


def run(url: str, workdir: Path, progress: Progress = _noop,
        include_drums: bool = True) -> Result:
    workdir = Path(workdir)
    outdir = workdir / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    progress(3, "下載音訊中…")
    track = download_mod.download(url, workdir,
                                  progress=lambda m: progress(5, m))
    progress(10, f"已取得：{track.title}")
    if track.is_low_quality:
        warnings.append(
            f"YouTube 只放行低音質串流（format {track.format_note}，{track.abr:.0f}kbps）。"
            "分軌與採譜的準確度會明顯下降——若在意品質，設定 cookies 後重跑可拿到高音質版本。"
        )

    progress(13, "分析速度與調性…")
    analysis = analyze_mod.analyze(track.wav_path)
    progress(18, f"{analysis.key_name} · {round(analysis.tempo)} BPM · "
                 f"{analysis.beats_per_bar}/4")

    stems = separate_mod.separate(
        track.wav_path, workdir,
        progress=lambda m: progress(25, m),
    )
    levels = separate_mod.stem_levels(track.wav_path, stems)
    present = separate_mod.present_stems(levels)
    progress(58, "分軌完成：" + "、".join(
        f"{STEM_LABEL.get(s, s)}"
        + ("" if s in present else "（幾乎無聲）")
        for s in stems
    ))

    # Every stem is transcribed before anything is quantised, so the key can be
    # re-derived from the notes themselves further down.
    raw_parts: dict = {}
    drum_hits: list = []

    if "vocals" in present:
        progress(62, "採譜：主唱旋律…")
        raw = transcribe_mod.transcribe_pitched(stems["vocals"], "vocals")
        raw_parts["vocal"] = transcribe_mod.calibrate_velocity(
            transcribe_mod.make_monophonic(raw), stems["vocals"])
        if not raw_parts["vocal"]:
            warnings.append("人聲軌沒有偵測到可用的旋律。")
    elif "vocals" in stems:
        warnings.append(
            f"人聲軌能量只有全曲的 {levels.get('vocals', 0):.1%}，"
            "判定為純演奏曲，已略過主唱聲部（避免把樂器串音寫成假旋律）。"
        )

    progress(70, "採譜：伴奏（鋼琴大譜表）…")
    accomp = separate_mod.make_accompaniment(stems, workdir)
    if accomp:
        raw_parts["accomp"] = transcribe_mod.calibrate_velocity(
            transcribe_mod.transcribe_pitched(accomp, "accompaniment"), accomp)

    if "bass" in present:
        progress(78, "採譜：貝斯…")
        raw = transcribe_mod.transcribe_pitched(stems["bass"], "bass")
        raw_parts["bass"] = transcribe_mod.calibrate_velocity(
            transcribe_mod.make_monophonic(raw), stems["bass"])

    if include_drums and "drums" in present:
        progress(83, "採譜：鼓組…")
        drum_hits = transcribe_mod.transcribe_drums(stems["drums"])

    grid = BeatGrid(analysis)
    parts_data: dict = {}
    note_counts: dict = {}

    if raw_parts.get("vocal"):
        parts_data["vocal"] = clean(
            quantize_notes(raw_parts["vocal"], grid), fix_octaves=True)
        note_counts["主唱旋律"] = len(parts_data["vocal"])

    if raw_parts.get("accomp"):
        rh, lh = split_hands(quantize_notes(raw_parts["accomp"], grid))
        parts_data["piano_rh"] = clean(rh)
        parts_data["piano_lh"] = clean(lh, min_duration=1.0)
        note_counts["鋼琴右手"] = len(parts_data["piano_rh"])
        note_counts["鋼琴左手"] = len(parts_data["piano_lh"])

    if raw_parts.get("bass"):
        parts_data["bass"] = clean(
            quantize_notes(raw_parts["bass"], grid), fix_octaves=True)
        note_counts["貝斯"] = len(parts_data["bass"])

    if drum_hits:
        parts_data["drums"] = quantize_drums(drum_hits, grid)
        note_counts["鼓組"] = len(parts_data["drums"])

    # Now that notes exist, re-derive the key from them: far more reliable
    # than the chroma estimate made from the raw mix.
    refined = analyze_mod.estimate_key_from_notes(
        {name: notes for name, notes in parts_data.items() if name != "drums"},
        analysis.beats_per_bar)
    if refined and refined[2] > analysis.key_confidence:
        old_key = analysis.key_name
        analysis.tonic, analysis.mode, analysis.key_confidence = refined
        if analysis.key_name != old_key:
            progress(86, f"依採譜音符修正調性：{old_key} → {analysis.key_name}")

    if analysis.key_confidence < 0.6:
        warnings.append(
            f"調性判斷信心偏低（{analysis.key_confidence:.2f}），"
            "簡譜的「1 = ?」可能需要人工修正。"
        )

    progress(87, "排版五線譜…")
    # Where score time 0 lands in the recording, kept before the trim discards it.
    lead_shift = leading_rest_shift(parts_data, beats_per_bar=analysis.beats_per_bar)
    first_offset = min((min(n.offset for n in notes) for name, notes in parts_data.items()
                        if name != "drums" and notes), default=lead_shift)
    score_anchor = grid.to_seconds(max(lead_shift, first_offset))
    parts_data = trim_leading_rest(parts_data, beats_per_bar=analysis.beats_per_bar)
    sc = score_mod.build_score(parts_data, analysis, track.title, track.uploader)
    files_abs = score_mod.export(sc, outdir, "score")
    if "pdf" not in files_abs:
        warnings.append("MuseScore 沒有產生 PDF，但 MusicXML 可以正常開啟。")

    progress(92, "合成試聽音檔…")
    audio_full = None
    audio_parts: dict = {}
    audio_full = audio_mod.render_score_audio(sc, outdir)
    audio_parts = audio_mod.render_part_audio(sc, outdir)
    compare_sources = dict(stems)
    if accomp:
        compare_sources["accompaniment"] = accomp
    audio_compare = audio_mod.render_comparisons(
        audio_parts, compare_sources, outdir, anchor=score_anchor,
        reference=audio_full)
    if audio_full is None:
        warnings.append("無法合成試聽音檔（MuseScore 音訊匯出失敗）。")

    progress(94, "排版簡譜…")
    melody_notes = parts_data.get("vocal") or parts_data.get("piano_rh") or []
    jp = jianpu_mod.write_jianpu(
        melody_notes, analysis, track.title, outdir,
        subtitle=f"{track.uploader} · 自動採譜草稿",
    )
    if not jp:
        warnings.append("沒有足夠的旋律音符可以產生簡譜。")

    files: dict = {}
    for label, path in files_abs.items():
        if label == "svg_pages":
            files["score_svg_pages"] = [str(p.relative_to(workdir)) for p in path]
        else:
            files[f"score_{label}"] = str(Path(path).relative_to(workdir))
    if jp:
        files["jianpu_html"] = str(jp["html"].relative_to(workdir))
        files["jianpu_svg_pages"] = [str(p.relative_to(workdir)) for p in jp["svg_pages"]]
    if audio_full:
        files["audio_full"] = str(audio_full.relative_to(workdir))
    if audio_parts:
        files["audio_parts"] = {
            label: str(path.relative_to(workdir))
            for label, path in audio_parts.items()
        }
    if audio_compare:
        files["audio_compare"] = {
            label: str(path.relative_to(workdir))
            for label, path in audio_compare.items()
        }
    for stem, path in stems.items():
        files[f"stem_{stem}"] = str(path.relative_to(workdir))

    result = Result(
        title=track.title, uploader=track.uploader, duration=track.duration,
        tempo=round(analysis.tempo, 1), key_name=analysis.key_name,
        key_confidence=round(analysis.key_confidence, 3),
        beats_per_bar=analysis.beats_per_bar,
        note_counts=note_counts, files=files, warnings=warnings,
    )
    (workdir / "result.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress(100, "完成")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m pipeline.run <youtube-url> <workdir>")
        raise SystemExit(2)
    try:
        r = run(sys.argv[1], Path(sys.argv[2]),
                progress=lambda p, m: print(f"[{p:3d}%] {m}", flush=True))
        print(json.dumps(asdict(r), ensure_ascii=False, indent=2))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
