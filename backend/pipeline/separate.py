"""Split a mix into vocals / drums / bass / other with Demucs."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("vocals", "drums", "bass", "other")
MODEL = "htdemucs"


def _torch_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def separate(wav_path: Path, workdir: Path, progress=None) -> dict[str, Path]:
    """Return {stem_name: wav path}. Falls back to CPU if the GPU path fails."""
    out_root = workdir / "stems"
    out_root.mkdir(parents=True, exist_ok=True)

    device = _torch_device()
    for attempt_device in ([device, "cpu"] if device != "cpu" else ["cpu"]):
        if progress:
            progress(f"分軌中（{MODEL} / {attempt_device}）…這是最花時間的一步")
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", MODEL, "-d", attempt_device,
            "-o", str(out_root), "--filename", "{stem}.{ext}",
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            break
        if attempt_device == "cpu":
            raise RuntimeError(f"Demucs 分軌失敗：\n{proc.stderr[-2000:]}")
        if progress:
            progress(f"{attempt_device} 失敗，改用 CPU 重試（會慢一些）")

    stem_dir = out_root / MODEL
    found = {}
    for stem in STEMS:
        path = stem_dir / f"{stem}.wav"
        if path.exists():
            found[stem] = path
    if not found:
        raise RuntimeError(f"Demucs 沒有產生任何音軌，請檢查 {stem_dir}")
    return found


def make_accompaniment(stems: dict[str, Path], workdir: Path) -> Path | None:
    """Mix bass + other into one track — the source for the piano grand staff."""
    parts = [stems[s] for s in ("bass", "other") if s in stems]
    if not parts:
        return None
    out = workdir / "accompaniment.wav"
    if len(parts) == 1:
        shutil.copy(parts[0], out)
        return out
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in parts:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", f"amix=inputs={len(parts)}:normalize=0", str(out)]
    subprocess.run(cmd, check=True)
    return out


def stem_levels(mix_path: Path, stems: dict[str, Path]) -> dict[str, float]:
    """RMS of each stem relative to the mix.

    Demucs always emits all four stems, so an instrumental track still gets a
    "vocals" file — just full of faint bleed. Transcribing that invents a vocal
    line that was never there, so callers use this to decide what is real.
    """
    def rms(path: Path) -> float:
        data, _ = sf.read(str(path), dtype="float32")
        if data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(data))))

    mix = rms(mix_path)
    if mix <= 0:
        return {name: 0.0 for name in stems}
    return {name: rms(path) / mix for name, path in stems.items()}


# Below this share of the mix's energy a stem is bleed, not a performance.
PRESENCE_THRESHOLD = 0.08


def present_stems(levels: dict[str, float],
                  threshold: float = PRESENCE_THRESHOLD) -> set[str]:
    return {name for name, level in levels.items() if level >= threshold}
