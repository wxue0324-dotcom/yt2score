#!/usr/bin/env bash
# Launch the local yt2score server.
set -euo pipefail
cd "$(dirname "$0")"
exec ./venv/bin/python -m uvicorn app:app \
  --app-dir backend --host 127.0.0.1 --port 8420 "$@"
