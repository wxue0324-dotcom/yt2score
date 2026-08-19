"""Local web service: paste a YouTube URL, get a score back."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from pipeline.download import cookies_configured
from pipeline.run import run as run_pipeline

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
WEB = ROOT / "web"

# Transcription pegs a core for minutes; one at a time keeps the machine usable
# and stops two Demucs runs fighting over the GPU.
EXECUTOR = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="yt2score")


@dataclass
class Job:
    id: str
    url: str
    include_drums: bool = True
    status: str = "queued"          # queued | running | done | error
    percent: int = 0
    log: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


JOBS: dict[str, Job] = {}


class JobRequest(BaseModel):
    url: str
    include_drums: bool = True


def _work_dir(job_id: str) -> Path:
    return WORK / job_id


def _execute(job: Job) -> None:
    job.status = "running"

    def progress(pct: int, msg: str) -> None:
        job.percent = pct
        job.log.append({"pct": pct, "msg": msg})

    try:
        result = run_pipeline(job.url, _work_dir(job.id), progress=progress,
                              include_drums=job.include_drums)
        job.result = asdict(result)
        job.status = "done"
        job.percent = 100
    except Exception as exc:  # surface the real reason in the UI
        job.status = "error"
        job.error = str(exc) or exc.__class__.__name__
        job.log.append({"pct": job.percent, "msg": f"錯誤：{job.error}"})
        traceback.print_exc()


@app.post("/api/jobs")
def create_job(req: JobRequest):
    job = Job(id=uuid.uuid4().hex[:12], url=req.url.strip(),
              include_drums=req.include_drums)
    JOBS[job.id] = job
    EXECUTOR.submit(_execute, job)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return asdict(job)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    async def stream():
        sent = 0
        while True:
            while sent < len(job.log):
                entry = job.log[sent]
                sent += 1
                yield f"data: {json.dumps({'type': 'log', **entry}, ensure_ascii=False)}\n\n"
            if job.status in ("done", "error"):
                payload = {"type": job.status, "result": job.result, "error": job.error}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(404, "job not found")
    shutil.rmtree(_work_dir(job_id), ignore_errors=True)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/files/{path:path}")
def get_file(job_id: str, path: str):
    if job_id not in JOBS or not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(404, "job not found")

    base = _work_dir(job_id).resolve()
    target = (base / path).resolve()
    # Keep a crafted path from reaching outside the job's own directory.
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(404, "file not found")

    media = {
        ".pdf": "application/pdf", ".svg": "image/svg+xml",
        ".musicxml": "application/vnd.recordare.musicxml+xml",
        ".mid": "audio/midi", ".html": "text/html", ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
    }.get(target.suffix, "application/octet-stream")
    return FileResponse(target, media_type=media, filename=target.name)


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "musescore": bool(shutil.which("mscore")),
        "cookies": cookies_configured(),
        "jobs": len(JOBS),
    }
