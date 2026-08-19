"""Mix a metronome click into a song at each candidate tempo, so a listener can
settle which one is right.

Tempo octave errors are not decidable from onset positions — the estimators
disagree and no amount of tuning settles it without knowing the answer. A
person can settle it in ten seconds: play the song with a click on top and hear
whether the click sits on the beat, double-times it, or drags.

Clicks come from the estimators' real beat positions, not a rigid grid, so
phase and rubato are audible too. Halved and doubled levels are derived from
the same beats by thinning and interpolating, which keeps them in phase.

    ../venv/bin/python click.py ../work/647103e8fe0a
    ../venv/bin/python click.py ../work/647103e8fe0a --start 0.4 --length 25
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

OUT = Path(__file__).resolve().parent / "clicks"
CLICK_GAIN = 0.35          # loud enough to judge, quiet enough to hear the music


def _levels(beats: np.ndarray) -> dict[str, np.ndarray]:
    """The same beat sequence read at neighbouring metrical levels."""
    out = {"x1": beats}
    if len(beats) > 3:
        out["half"] = beats[::2]
        mid = (beats[:-1] + beats[1:]) / 2
        out["x2"] = np.sort(np.concatenate([beats, mid]))
        third = np.sort(np.concatenate([
            beats[:-1] + (beats[1:] - beats[:-1]) * k / 3 for k in (0, 1, 2)]))
        out["x3"] = third
        out["third"] = beats[::3]
    return out


def _bpm(beats: np.ndarray) -> float:
    return 60.0 / float(np.median(np.diff(beats))) if len(beats) > 2 else 0.0


def beats_from_librosa(wav: Path) -> np.ndarray:
    from pipeline.analyze import _estimate_tempo
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    _, beats = _estimate_tempo(y, sr)
    return np.asarray(beats, dtype=float)


def beats_from_beat_this(wav: Path) -> np.ndarray:
    from beat_this.inference import File2Beats
    beats, _ = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(str(wav))
    return np.asarray(beats, dtype=float)


def render(wav: Path, beats: np.ndarray, name: str, outdir: Path,
           start_frac: float, length: float) -> tuple[Path, float] | None:
    if len(beats) < 3:
        return None
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    total = len(y) / sr
    t0 = min(total * start_frac, max(0.0, total - length))
    t1 = min(t0 + length, total)
    seg = y[int(t0 * sr):int(t1 * sr)]
    inside = beats[(beats >= t0) & (beats < t1)] - t0
    if len(inside) < 2:
        return None
    click = librosa.clicks(times=inside, sr=sr, length=len(seg), click_freq=1400.0)
    peak = np.max(np.abs(seg)) or 1.0
    mixed = seg / peak * 0.8 + click * CLICK_GAIN
    mixed = mixed / (np.max(np.abs(mixed)) or 1.0) * 0.95

    outdir.mkdir(parents=True, exist_ok=True)
    bpm = _bpm(beats)
    stem = f"{name}_{bpm:.0f}bpm"
    wav_out = outdir / f"{stem}.wav"
    sf.write(wav_out, mixed, sr)
    mp3 = outdir / f"{stem}.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_out),
                    "-b:a", "160k", str(mp3)], check=False)
    if mp3.exists():
        wav_out.unlink(missing_ok=True)
        return mp3, bpm
    return wav_out, bpm


def _steady(wav: Path, bpm: float) -> np.ndarray:
    """An isochronous grid at `bpm`, phased to sit on as many onsets as it can."""
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    period = 60.0 / bpm
    best, grid = -1.0, np.arange(0.0, duration, period)
    for phase in np.arange(0, period, period / 24):
        candidate = np.arange(phase, duration, period)
        if len(candidate) < 2 or len(onsets) == 0:
            continue
        idx = np.clip(np.searchsorted(onsets, candidate), 0, len(onsets) - 1)
        hit = float(np.mean(np.abs(onsets[idx] - candidate) <= 0.07))
        if hit > best:
            best, grid = hit, candidate
    return grid


def _report(made: list[tuple[float, Path]]) -> None:
    print("\n  聽這幾個檔案,哪個點擊聲一直踩在拍子上就是對的:")
    for bpm, path in sorted(made):
        print(f"    {bpm:>7.1f} bpm   {path}")
    print(f"\n  候選速度: {sorted({round(b) for b, _ in made})}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="a work/<id> directory or an audio file")
    ap.add_argument("--start", type=float, default=0.35,
                    help="where in the song to excerpt, as a fraction")
    ap.add_argument("--length", type=float, default=20.0, help="excerpt seconds")
    ap.add_argument("--levels", default="x1,half,x2",
                    help="which metrical levels to render, per estimator")
    ap.add_argument("--bpm", default=None,
                    help="comma-separated tempi to audition as a steady grid, "
                         "instead of running the estimators")
    args = ap.parse_args()

    target = Path(args.target)
    if target.is_dir():
        wav = target / "source.wav"
        rj = target / "result.json"
        title = json.loads(rj.read_text())["title"] if rj.exists() else target.name
        slug = target.name
    else:
        wav, title, slug = target, target.stem, target.stem
    if not wav.exists():
        sys.exit(f"找不到音檔 {wav}")

    outdir = OUT / slug
    print(f"\n{title}\n  {wav}")
    wanted = [s.strip() for s in args.levels.split(",")]
    made = []
    if args.bpm:
        # A steady grid at a tempo you name. Useful when the estimators have
        # all been ruled out by ear and you want to test a hunch — and, on a
        # rubato performance, to hear whether *any* steady grid can hold.
        for bpm in [float(b) for b in args.bpm.split(",")]:
            got = render(wav, _steady(wav, bpm), f"steady_{bpm:.0f}",
                         outdir, args.start, args.length)
            if got:
                made.append((got[1], got[0]))
        _report(made)
        return
    for est, fn in (("librosa", beats_from_librosa), ("beatthis", beats_from_beat_this)):
        try:
            beats = fn(wav)
        except Exception as exc:                       # one estimator missing is fine
            print(f"  {est}: 略過 ({exc})")
            continue
        print(f"  {est}: {_bpm(beats):.2f} bpm")
        for level, seq in _levels(beats).items():
            if level not in wanted:
                continue
            got = render(wav, seq, f"{est}_{level}", outdir, args.start, args.length)
            if got:
                made.append((got[1], got[0]))

    _report(made)


if __name__ == "__main__":
    main()
