# NOTES: hardware quirks and measurements

Add one entry per surprise. Measurements that end up in config.yaml should be recorded here too, with how they
were measured.

## Phase 0: to fill on the arm

| Joint | Servo id | Raw range (units) | Soft limits (deg) | +deg direction | Notes |
|---|---|---|---|---|---|
| base | ? | | | | |
| shoulder | ? | | | | |
| elbow | ? | | | | |
| wrist_pitch | ? | | | | |
| wrist_roll | ? | | | | |
| gripper | ? | | | | |

- Battery voltage at idle / under motion:
- Pot noise with torque on (tests/hw/test_arm_limits.py::test_noise_floor):
- Link lengths (ruler, axis to axis): base height __, upper arm __, forearm __, wrist-pitch axis to spoon tip __
- Camera index / FPS at 720p:
- ESP32 port name:

## Known so far (from source and the laptop, before the arm)

- **xarm 0.0.4 assumes 250 deg per 1000 units** (`Util._angle_to_position` maps -125..125). The LX-15D is 240 deg.
  Spoonful never uses the package's degree API; it sends integer units with `units_per_deg: 4.1667`. Check against
  a protractor in Phase 0.
- **xarm 0.0.4 `getPosition`** takes an int, or a list of `xarm.Servo` objects (not ints) that it fills in place. One
  HID round trip reads all six.
- **xarm 0.0.4 `_recv`** indexes the report without checking it is non-empty: an empty 50 ms read raises
  `IndexError`. XarmBackend retries, then falls back to single reads.
- Raw HID protocol: reply length byte = payload + 2. CMD 0x14 (20) is unload and CMD 0x15 (21) is position read,
  matching what the package uses.
- **mediapipe 1.2.3 `FaceLandmarker.__del__`** deadlocks at interpreter exit if close() was not called.
  `FaceTracker.stop()` closes it explicitly.
- MacBook built-in camera: index 0, 1280x720 at 29.8 FPS; landmarker 4.7 ms/frame (measured 2026-11-12, no face in
  view).
- **Depth from one camera** (offsets + face.py): iris diameter gives absolute distance but jitters about 5 %
  (about 25 px across). Face width is smooth but varies about 8 % between people. The iris is used to learn each
  person's face width, then face width gives the depth. Not yet checked on a real face: sit at a tape-measured
  50 cm and read the number next to the mouth marker. Expect about 3 % error.
- Nothing showed up for VID 0483 / PID 5750 on the laptop yet (arm not connected).
- Reach: with the spoon about 210 mm past the wrist pitch, a level spoon reaches about 300 mm from the base axis at
  250 mm height. The diner's mouth has to be within about 30 cm of the base. Plan the table layout around that.
