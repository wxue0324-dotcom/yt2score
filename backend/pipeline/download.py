"""Fetch audio from a YouTube URL and normalise it to 44.1kHz stereo WAV.

YouTube gates a large share of videos behind session checks, and which client
gets through changes without notice. Rather than trusting one request shape,
this walks a ladder of strategies and only gives up once every rung has failed.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yt_dlp

# Accept the handful of YouTube URL shapes people actually paste.
_YT_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
}

# Player clients to try in order. Different clients are gated differently, and
# the one that works today is not the one that worked last month, so breadth
# beats picking a favourite. None = let yt-dlp choose.
_CLIENTS: tuple[str | None, ...] = (
    None, "tv_simply", "web_safari", "mweb", "ios", "android_vr", "tv",
)

# Format ladder. YouTube increasingly serves metadata for every format but
# then refuses the media URL for the good ones, while leaving a low-bitrate
# stream open. Walking down the ladder turns those 403s into a usable
# (if lossy) download instead of a hard failure.
_FORMATS: tuple[str, ...] = (
    "bestaudio/best",
    "140",              # 130k m4a
    "251",              # 134k opus
    "18",               # 360p mp4, audio muxed in
    "139",              # 49k m4a — often the only one left open
    "worstaudio/worst",
)

# yt-dlp needs this helper script to solve YouTube's "n" signature challenge.
# Without it YouTube throttles or refuses the media URLs outright. It is
# fetched from the yt-dlp project's own GitHub on first use.
_REMOTE_COMPONENTS = ["ejs:github"]

# Errors worth retrying verbatim: YouTube throws these intermittently and the
# same request often succeeds seconds later.
_TRANSIENT = (
    "the page needs to be reloaded",
    "unable to download video data",
    "temporary failure",
    "connection reset",
    "read timed out",
    "http error 5",
)


@dataclass
class Track:
    title: str
    uploader: str
    duration: float
    video_id: str
    wav_path: Path
    abr: float = 0.0            # audio bitrate actually obtained, kbps
    format_note: str = ""

    @property
    def is_low_quality(self) -> bool:
        return 0 < self.abr < 90


@dataclass
class Attempt:
    client: str | None
    cookies: bool
    error: str


def is_youtube_url(url: str) -> bool:
    m = re.match(r"^https?://([^/]+)/", url.strip() + "/")
    return bool(m) and m.group(1).lower() in _YT_HOSTS


def cookie_opts() -> dict:
    """Cookie settings from the environment, if the user configured any.

    Set YT2SCORE_COOKIES_FROM_BROWSER=chrome (or firefox/edge/brave), or point
    YT2SCORE_COOKIES_FILE at an exported cookies.txt.
    """
    browser = os.environ.get("YT2SCORE_COOKIES_FROM_BROWSER", "").strip()
    cookie_file = os.environ.get("YT2SCORE_COOKIES_FILE", "").strip()
    if browser:
        # Accept "chrome" or "chrome:Profile 1" for a non-default profile.
        name, _, profile = browser.partition(":")
        return {"cookiesfrombrowser": (name, profile or None, None, None)}
    if cookie_file and Path(cookie_file).is_file():
        return {"cookiefile": cookie_file}
    return {}


def cookies_configured() -> bool:
    return bool(cookie_opts())


def _is_transient(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _TRANSIENT)


def _is_fatal(message: str) -> bool:
    """Errors where trying another client cannot possibly help."""
    lowered = message.lower()
    return any(marker in lowered for marker in (
        "private video", "video unavailable", "removed by the uploader",
        "does not exist", "terminated", "copyright",
    ))


def _extract(url: str, outtmpl: str, client: str | None, cookies: dict,
             fmt: str = "bestaudio/best", retries: int = 2) -> dict:
    opts = {
        "format": fmt,
        "remote_components": _REMOTE_COMPONENTS,
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        **cookies,
    }
    if client:
        opts["extractor_args"] = {"youtube": {"player_client": [client]}}

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            last = exc
            message = str(exc)
            if _is_fatal(message) or not _is_transient(message):
                raise
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def download(url: str, workdir: Path, max_duration: float = 720.0,
             progress: Callable[[str], None] | None = None) -> Track:
    """Download bestaudio and decode to WAV. Raises ValueError on bad input."""
    if not is_youtube_url(url):
        raise ValueError("這不是有效的 YouTube 連結")

    workdir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(workdir / "source.%(ext)s")

    cookies = cookie_opts()
    # Without cookies there is nothing to fall back to; with them, try the
    # plain request first since it is faster and often enough.
    cookie_ladder = [{}, cookies] if cookies else [{}]

    attempts: list[Attempt] = []
    info = None

    for cookie_set in cookie_ladder:
        for index, client in enumerate(_CLIENTS):
            label = client or "預設"
            suffix = " + cookies" if cookie_set else ""
            # Only the first client walks the whole format ladder. Once a
            # client has failed on every format, the others rarely do better
            # than their best stream, so they get one shot each.
            formats = _FORMATS if index == 0 else ("bestaudio/best",)

            for fmt in formats:
                detail = "" if fmt == "bestaudio/best" else f" · 格式 {fmt}"
                if progress:
                    progress(f"嘗試下載（{label}{suffix}{detail}）…")
                try:
                    info = _extract(url, outtmpl, client, cookie_set, fmt=fmt)
                    break
                except yt_dlp.utils.DownloadError as exc:
                    message = str(exc)
                    attempts.append(Attempt(client, bool(cookie_set), message))
                    if _is_fatal(message):
                        raise ValueError(_explain(attempts)) from exc
                except Exception as exc:  # cookie decryption, keychain denial…
                    attempts.append(Attempt(client, bool(cookie_set), str(exc)))
                    break
            if info is not None:
                break
        if info is not None:
            break

    if info is None:
        raise ValueError(_explain(attempts))

    downloaded = Path(yt_dlp.YoutubeDL({"outtmpl": outtmpl})
                      .prepare_filename(info))
    if not downloaded.exists():
        candidates = sorted(workdir.glob("source.*"))
        if not candidates:
            raise ValueError("下載似乎成功，但找不到輸出檔案。")
        downloaded = candidates[0]

    duration = float(info.get("duration") or 0.0)
    if duration > max_duration:
        downloaded.unlink(missing_ok=True)
        raise ValueError(
            f"歌曲長度 {duration/60:.1f} 分鐘，超過上限 {max_duration/60:.0f} 分鐘。"
            "太長的曲子分軌會非常慢，請換一首或調高 max_duration。"
        )

    wav = workdir / "source.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(downloaded),
         "-ac", "2", "-ar", "44100", str(wav)],
        check=True,
    )
    downloaded.unlink(missing_ok=True)

    return Track(
        title=info.get("title") or "Unknown",
        uploader=info.get("uploader") or "",
        duration=duration,
        video_id=info.get("id") or "",
        wav_path=wav,
        abr=float(info.get("abr") or 0.0),
        format_note=str(info.get("format_id") or ""),
    )


def _is_auth_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in (
        "403", "forbidden", "requested format is not available",
        "sign in", "login required", "po token", "not a bot",
    ))


def _explain(attempts: list[Attempt]) -> str:
    """Diagnose from the whole set of attempts, not just the last one.

    Individual clients report the same refusal in wildly different words — the
    `tv` client in particular calls an ordinary auth block "DRM protected" —
    so the majority verdict is far more trustworthy than the final message.
    """
    if not attempts:
        return "下載失敗，但沒有取得任何錯誤訊息。"

    errors = [a.error for a in attempts]
    tried = "、".join(dict.fromkeys(a.client or "預設" for a in attempts))
    last = errors[-1].strip()

    def any_match(*markers: str) -> bool:
        return any(m in e.lower() for e in errors for m in markers)

    # Unambiguous, video-specific verdicts first.
    if any_match("private video"):
        return "這是私人影片，無法下載。"
    if any_match("video unavailable", "removed by the uploader", "does not exist"):
        return "影片不存在或已被移除。"
    if any_match("age", "age-restricted") and any_match("confirm your age"):
        return "這支影片有年齡限制，需要登入過的 cookies 才能下載。"

    if any_match("keychain", "could not decrypt", "operation not permitted",
                 "unable to read"):
        browser = os.environ.get("YT2SCORE_COOKIES_FROM_BROWSER", "chrome")
        return (
            "讀取瀏覽器 cookies 失敗（macOS 鑰匙圈拒絕存取）。\n"
            "請在終端機手動執行一次，跳出視窗時按「一律允許」：\n"
            f"  {_sample_command()}\n"
            "授權後重新啟動伺服器即可。"
        )

    auth_count = sum(_is_auth_error(e) for e in errors)
    drm_count = sum("drm" in e.lower() for e in errors)

    # Only believe DRM when it is the consistent story across clients.
    if drm_count and drm_count >= auth_count:
        return "這支影片有 DRM 保護，無法下載。"

    if auth_count:
        if not cookies_configured():
            return (
                f"YouTube 拒絕提供這支影片的所有音訊格式（已試過 {tried}，"
                "以及由高到低的各種格式）。\n"
                "這支影片的限制比較嚴，需要登入過的瀏覽器 session 才拿得到：\n"
                "  export YT2SCORE_COOKIES_FROM_BROWSER=chrome\n"
                "  ./start.sh\n"
                "（首次會跳出鑰匙圈授權，按「一律允許」；可先跑 "
                "python doctor.py 確認設定是否生效）"
            )
        return (
            f"帶了 cookies 仍無法下載（已試過 {tried}）。\n"
            "可能是地區限制、年齡限制，或該帳號無權觀看。請換一首試試。\n"
            f"yt-dlp 最後的訊息：{last[-200:]}"
        )

    return f"下載失敗（已試過 {tried}）：{last[-300:]}"


def _sample_command() -> str:
    browser = os.environ.get("YT2SCORE_COOKIES_FROM_BROWSER", "chrome") or "chrome"
    return (f"~/yt2score/venv/bin/yt-dlp --cookies-from-browser {browser} "
            f"--simulate --print '%(title)s' "
            f"'https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
