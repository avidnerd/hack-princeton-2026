#!/usr/bin/env bash
# preflight -> photon sidecar -> app (opens the browser; FSM homes the arm on boot)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BACKEND="${1:-xarm}"

$PY scripts/preflight.py --backend "$BACKEND"

if [ -d sidecar/node_modules ] && grep -q '^PHOTON_PROJECT_ID=.' .env 2>/dev/null; then
  (cd sidecar && node photon.js > ../data/sidecar.log 2>&1 &)
  echo "photon sidecar started (data/sidecar.log)"
else
  echo "photon sidecar skipped (npm install in sidecar/ and set PHOTON_* in .env to enable)"
fi

trap 'pkill -f "node photon.js" 2>/dev/null || true' EXIT
$PY -m spoonful --backend "$BACKEND" run --open
