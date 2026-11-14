"""Phase 0 scratch: confirm camera index + MediaPipe Face Landmarker FPS. NOT product code.

  .venv/bin/python ../scratch/check_camera.py --list          # which indices open
  .venv/bin/python ../scratch/check_camera.py --index 0 --secs 10
"""
import argparse
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--secs", type=float, default=10)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--model", default="models/face_landmarker.task")
ap.add_argument("--list", action="store_true")
ap.add_argument("--api", default="auto", help="auto (DirectShow on Windows) | any | dshow | msmf")
a = ap.parse_args()
API = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(
    a.api, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)

if a.list:
    for i in range(4):
        cap = cv2.VideoCapture(i, API)
        ok, f = cap.read()
        print(i, "OK" if ok else "-", f.shape if ok else "")
        cap.release()
    raise SystemExit

cap = cv2.VideoCapture(a.index, API)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=a.model),
    running_mode=vision.RunningMode.VIDEO, num_faces=1,
    output_facial_transformation_matrixes=True))

n = faces = 0
t0 = time.time()
proc = []
while time.time() - t0 < a.secs:
    ok, frame = cap.read()
    if not ok:
        print("frame read failed")
        break
    t = time.time()
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    r = lm.detect_for_video(img, int(t * 1000))
    proc.append((time.time() - t) * 1000)
    n += 1
    if r.face_landmarks:
        faces += 1
        p = r.face_landmarks[0]
        ratio = abs(p[14].y - p[13].y) / max(1e-6, abs(p[152].y - p[10].y))
        if n % 15 == 0:
            print(f"mouth_open_ratio {ratio:.3f}")
el = time.time() - t0
print(f"\nframe {frame.shape[1]}x{frame.shape[0]}  loop FPS {n/el:.1f}  "
      f"landmarker {sum(proc)/max(1,len(proc)):.1f} ms/frame  face in {faces}/{n} frames")
