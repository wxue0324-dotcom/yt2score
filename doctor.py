"""Check that everything yt2score needs is installed and working.

Run this first when a download fails: it separates "your setup is broken" from
"YouTube refused this particular video".
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

OK, WARN, FAIL = "\033[32m✓\033[0m", "\033[33m!\033[0m", "\033[31m✗\033[0m"
# A short, always-public video used only to prove the download path works.
PROBE_URL = "https://www.youtube.com/watch?v=0mAuj1rtx6k"


def check_binaries() -> bool:
    print("\n必要程式")
    all_ok = True
    for name, why, required in (
        ("ffmpeg", "音訊解碼", True),
        ("mscore", "五線譜 PDF 排版", True),
        ("deno", "yt-dlp 解 YouTube 簽章", True),
    ):
        path = shutil.which(name)
        if path:
            print(f"  {OK} {name:8s} {path}")
        else:
            print(f"  {FAIL if required else WARN} {name:8s} 找不到 — {why}")
            all_ok = all_ok and not required
    return all_ok


def check_python() -> bool:
    print("\nPython 套件")
    ok = True
    for module, label in (("torch", "PyTorch"), ("demucs", "Demucs"),
                          ("basic_pitch", "basic-pitch"), ("music21", "music21"),
                          ("librosa", "librosa"), ("yt_dlp", "yt-dlp")):
        try:
            __import__(module)
            print(f"  {OK} {label}")
        except Exception as exc:
            print(f"  {FAIL} {label} — {exc}")
            ok = False
    try:
        import torch
        device = ("MPS (Apple Silicon 加速)" if torch.backends.mps.is_available()
                  else "CPU only（分軌會慢很多）")
        print(f"  {OK if torch.backends.mps.is_available() else WARN} 運算裝置：{device}")
    except Exception:
        pass
    return ok


def check_model() -> None:
    """The separation weights are an 84MB download on first use.

    Left to the first real run, that fetch happens inside the separation step
    and looks like a hang. Doing it here — where waiting is the whole point of
    the command — is the difference between a slow doctor and a stalled song.
    """
    print("\n分軌模型")
    from pipeline import separate as sep

    if sep.model_is_cached():
        print(f"  {OK} {sep.MODEL} 已快取")
        return
    print(f"  {WARN} {sep.MODEL} 尚未下載，現在抓（約 84MB，慢的網路要幾分鐘）…")
    try:
        sep.ensure_model(progress=lambda msg: print(f"      {msg}"))
        print(f"  {OK} 下載完成")
    except Exception as exc:
        print(f"  {FAIL} 下載失敗：{exc}")
        print("      → 需要能連到 dl.fbaipublicfiles.com。")


def check_challenge_solver() -> None:
    """YouTube's "n" signature challenge must be solved or media URLs 403."""
    print("\nYouTube 簽章挑戰求解器 (EJS)")
    proc = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--remote-components", "ejs:github",
         "--simulate", "--quiet", "--print", "%(id)s", PROBE_URL],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode == 0:
        print(f"  {OK} 可正常下載求解器並解析影片")
    else:
        err = (proc.stderr or "").strip()
        print(f"  {FAIL} 失敗：{err[-300:]}")
        print("      → 需要能連到 GitHub。若在受限網路下，請改用 cookies。")


def check_cookies() -> None:
    print("\nCookies 設定")
    from pipeline import download as dl

    opts = dl.cookie_opts()
    if not opts:
        print(f"  {OK} 未設定 — 通常不需要。程式會自動輪替 client 與音訊格式，")
        print("      多數受限影片靠降級格式就能取得（音質較差但可用）。")
        print("      只有極少數影片需要： export YT2SCORE_COOKIES_FROM_BROWSER=chrome")
        return

    if "cookiefile" in opts:
        print(f"  {OK} 使用 cookies 檔：{opts['cookiefile']}")
        return

    browser = opts["cookiesfrombrowser"][0]
    print(f"  · 設定為從 {browser} 讀取，測試中…")
    proc = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--cookies-from-browser", browser,
         "--simulate", "--quiet", "--no-warnings", "--print", "%(id)s", PROBE_URL],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode == 0:
        print(f"  {OK} 成功讀取 {browser} 的 cookies")
    else:
        err = (proc.stderr or "").strip()
        print(f"  {FAIL} 讀取失敗：{err[-300:]}")
        if "keychain" in err.lower() or "decrypt" in err.lower():
            print("      → macOS 鑰匙圈拒絕。請手動執行下面這行，跳出視窗時按「一律允許」：")
            print(f"        ./venv/bin/yt-dlp --cookies-from-browser {browser} "
                  f"--simulate --print '%(title)s' '{PROBE_URL}'")


def check_download() -> None:
    print("\n下載測試（公開短片）")
    from pipeline import download as dl
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        try:
            track = dl.download(PROBE_URL, Path(tmp),
                                progress=lambda m: print(f"      {m}"))
            print(f"  {OK} 下載並轉檔成功：{track.title} ({track.duration:.0f}s)")
        except Exception as exc:
            print(f"  {FAIL} 失敗：\n      " + str(exc).replace("\n", "\n      "))


def main() -> int:
    print("yt2score 環境檢查")
    print("=" * 46)
    binaries_ok = check_binaries()
    python_ok = check_python()
    check_model()
    check_challenge_solver()
    check_cookies()
    check_download()
    print("\n" + "=" * 46)
    if binaries_ok and python_ok:
        print("核心環境正常。若特定影片仍下載失敗，那是 YouTube 對該片的限制。")
        return 0
    print("有缺少的元件，請依上面提示安裝。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
