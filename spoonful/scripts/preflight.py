#!/usr/bin/env python
"""Preflight: refuses the demo on any RED item and prints the fix. Exit code 0 = go.
  .venv/bin/python scripts/preflight.py [--backend xarm|raw|sim] [--no-camera]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spoonful import config as C  # noqa: E402
from spoonful.preflight import run_checks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["sim", "xarm", "raw"])
ap.add_argument("--no-camera", action="store_true")
a = ap.parse_args()
cfg = C.load()
C.load_env()
ok, lines = run_checks(cfg, a.backend or cfg.arm.backend, camera=not a.no_camera)
print("\n".join(lines))
print("\nPREFLIGHT", "GREEN: go" if ok else "RED: fix the items above")
sys.exit(0 if ok else 1)
