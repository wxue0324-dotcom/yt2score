#!/usr/bin/env bash
# Launch the local yt2score server.
set -euo pipefail
cd "$(dirname "$0")"

PORT=8420

# Free the port if a previous run of *this* server is still holding it. Editing
# the pipeline and restarting is the normal loop, and without this the new
# process exits on "address already in use" while the old one keeps serving —
# which looks like the changes did nothing. Only ever kills a uvicorn running
# this app: anything else on the port is someone else's business, so say so and
# stop rather than guess.
for pid in $(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true); do
  if ps -o command= -p "$pid" | grep -q "uvicorn app:app"; then
    echo "停掉舊的 server (pid $pid)…"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  else
    echo "連接埠 $PORT 被別的程式佔用 (pid $pid)：" >&2
    ps -o pid,command= -p "$pid" >&2
    exit 1
  fi
done

exec ./venv/bin/python -m uvicorn app:app \
  --app-dir backend --host 127.0.0.1 --port "$PORT" "$@"
