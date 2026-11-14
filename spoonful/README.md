# Spoonful

A self-feeding arm for people who cannot bring a spoon to their own mouth. A Hiwonder xArm 1S, the laptop camera and
this repo: the last few centimetres of every bite re-target live to the diner's face, the arm approaches only while
the diner looks at it with their mouth open, and a caregiver teaches a bite by moving the arm by hand.

- Safety design: [SAFETY.md](SAFETY.md) (generated from the code)
- Hardware quirks and measurements: [NOTES.md](NOTES.md)
- Pitch script: [PITCH.md](PITCH.md)

## Setup (10 min)

On Windows, use [../SETUP_WINDOWS.md](../SETUP_WINDOWS.md) instead (PowerShell commands, `.venv\Scripts\python.exe`,
`scripts\demo.ps1`). The face model is already committed in `models/`, so the curl step below is only needed if it's missing.

```bash
cd spoonful
uv venv --python 3.12 .venv            # or python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models && curl -sSfL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
cp .env.example .env                    # fill keys you have; everything has an offline fallback
.venv/bin/python -m pytest              # 66 tests, no hardware needed
```

Optional: `cd sidecar && npm install` for caregiver texts (Photon), and flash `firmware/esp32_stop_piezo` for the
STOP button and the swallow sensor. Without the ESP32 the spacebar is STOP.

## Try it without the robot

```bash
.venv/bin/python -m spoonful seed-sim                    # placeholder trajectories for the simulator
.venv/bin/python -m spoonful --backend sim run --sim-face --open
```

Press `o` for the operator drawer. On the **Sim** tab, tick "mouth open" to let the arm deliver, and move the yaw
slider to watch it back off. Use **Start session**, then **Bite**. Leave out `--sim-face` to use your real face on
the camera.

## Bring-up (Phase 0, arm connected, about 15 min) {#bring-up}

1. Power: 7.5 V 6 A adapter through the rocker switch. Rocker OFF. USB to the laptop.
2. Rocker ON. `python -m spoonful --backend xarm test-arm`. It prints the battery voltage and all six positions,
   and never moves the arm. If it can't open the device, try `--backend raw`.
3. Map the joints: `python ../scratch/bringup_arm.py watch` turns torque off and prints positions at 10 Hz. Move
   one joint at a time from end stop to end stop. Record which id moved, its raw range, and which direction is
   "forward/down" in NOTES.md.
4. Copy into `config.yaml`: `arm.servo_map`, `arm.soft_limits_deg` (5 deg inside each hard stop), `kinematics.sign`
   and `zero_deg`, link lengths measured with a ruler.
5. Check motion: `python ../scratch/bringup_arm.py jog` (dry run), then `jog --live`. It asks before each joint.
6. Camera: `python ../scratch/check_camera.py --list`, then `--index N`. The pipeline needs at least 20 FPS.
7. `python scripts/preflight.py --backend xarm`.

Set `arm.backend: xarm` in config.yaml once bring-up passes.

## Teach a bite (caregiver)

In the operator drawer (`o`) > **Teach**, with the FSM in IDLE:

| Name | Kind | What to do |
|---|---|---|
| `rest` | pose | arm folded, spoon level |
| `scoop` (`scoop_small`, `scoop_full`) | path | from rest: down into the bowl, drag, tilt level, up a little |
| `lift` | path | from the end of the scoop up toward the diner, stop about 10 cm short |
| `pre_delivery` | pose | where the spoon waits: about 10 cm in front of the mouth, outside the face zone |
| `mouth_nominal` | pose | spoon at a seated diner's lips |
| `present`, `retreat` | path | optional; `retreat` goes from pre_delivery back to rest |

**Record path** turns torque off. Move the arm, then press **DONE** (or the ESP32 button). **Save current pose**
captures a pose. Before any live replay, run **Dry run** and read the plan. The CLI does the same:
`python -m spoonful teach record scoop`, `python -m spoonful replay scoop --speed 0.8` (dry run by default; add `--live` to move).

## Calibrate adaptive delivery

`python -m spoonful calibrate` opens the drawer on **Calibrate**. With nobody in the chair:
1. Turn **Calibration mode ON**. This allows approach with no diner, and a banner says so.
2. Put the spoon at `mouth_nominal`, click **nominal**, then click the spoon tip on the camera image.
3. Repeat about 30 mm left, right, up and down (move by hand with torque off, or use the jog buttons).
4. Depth (optional, recommended): **away** = the tip moved about 30 mm straight away from the camera, so it barely
   moves in the image. **toward** is the same in the other direction. If you tape-measure lens to tip at nominal,
   type it in; otherwise depth comes from `camera.hfov_deg`, which only scales the correction, it doesn't bias it.
5. **Fit map**. The residual should be under 1 deg. Read any `warnings` in the result.
6. **Draw window** and drag the box where a diner's mouth may be.

Depth: the camera sees how far away the diner is from the iris (about 11.7 mm in every adult), learns that
person's face width from it in about a second, then tracks distance from the face width. The distance shows next
to the mouth marker. If the diner leans back, the spoon follows by up to `offsets.max_depth_correction_mm`.

## Demo

```bash
scripts/demo.sh            # preflight -> photon sidecar (if configured) -> app -> browser; arm homes on boot
```

Pick the diner profile, press **Start session**, then "Spoonful, bite" / **Bite** / `b`. After the session:
`python scripts/export_stats.py` prints the pitch numbers.

## Layout

```
spoonful/arm         backends (sim, xarm, raw HID), Arm (guards, streaming), kinematics, trajectory, teach, contact
spoonful/perception  camera, face (MediaPipe), gate (consent), offsets (adaptive delivery), food (optional VLM)
spoonful/brain       fsm, watchdog, speech, voice, pace, profiles, session
spoonful/app         FastAPI + WebSocket + MJPEG, static UI, caregiver view
spoonful/io          ESP32 serial, Photon client
sidecar/             Node Photon sender      firmware/  ESP32 STOP + piezo
scripts/             preflight, demo.sh, export_stats, make_safety_table
```

## Versions (verified 2026-11-12, Python 3.12.2, macOS arm64)

xarm 0.0.4 · hidapi 0.15.0 · mediapipe 1.2.3 · opencv-python 5.0.0.93 · fastapi 0.162.5 · uvicorn 0.54.0 ·
pydantic 2.14.3 · faster-whisper 1.2.1 · elevenlabs 2.77.0 · google-genai 3.2.0 · pyserial 3.5 ·
spectrum-ts 12.16.0 (Node 20+)
