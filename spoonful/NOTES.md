# NOTES: hardware quirks and measurements

Add one entry per surprise. Measurements that end up in config.yaml should be recorded here too, with how they
were measured.

## Phase 0: to fill on the arm

| Joint | Servo id | Raw range (units) | Soft limits (deg) | +deg direction | Notes |
|---|---|---|---|---|---|
| base | 6 | -156 .. 1163 | -115..115 | +units = CCW (left), sign +1 | Mechanical travel exceeds the servo's 0..1000 |
| shoulder | 5 | 133 .. 868 | -83..83 | +units = back, sign -1 | |
| elbow | 4 | -40 .. 1063 | -115..115 | +units = back, sign -1 | Mechanical travel exceeds the servo's 0..1000 |
| wrist_pitch | 3 | 39 .. 1044 | -105..115 | +units = forward, sign +1 | |
| wrist_roll | 2 | -5 .. 953 | -115..103 | +units = CCW | |
| gripper | 1 | 164 .. 729 | -75..50 | ? | |

Measured 2026-11-13 with `scratch/bringup_arm.py watch`, torque off, joints moved by hand stop to stop.
Soft limits are servo degrees, (units - 500) / 4.1667, placed 5 deg inside the smaller of the hard stop and the
servo's 0..1000 end.

- **Straight up (zero_deg)**: arm held vertical by hand, `bringup_arm.py probe`: shoulder 527, elbow 479, wrist
  pitch 485 units, i.e. +6.5, -5.0, -3.6 servo deg.
- **Base faces the diner at units 122 (-90.7 servo deg)**, not at the servo centre (500), so `zero_deg.base` is
  -90.7. Consequence: from facing the diner the base can turn only ~24 deg to the arm's right (servo limit) but ~200
  deg to its left. Put the plate on the arm's left (the diner's right). Re-centring the arm on its base plate would
  even this out.
- **Home pose** posed by hand, probe: units base 122, shoulder 764, elbow 232, wrist pitch 865, roll 473. Upper arm
  leans back 57 deg, forearm vertical, spoon level; tip 118 mm toward the diner, 227 mm above the table.
- **GO button over Bluetooth (2026-11-13)**: one micro-USB cable, and the arm needs it, so the ESP32-C3-MINI-1 runs
  `firmware/esp32_go_ble`: it pairs with Windows as a Bluetooth keyboard named "Spoonful GO" and types Page Down
  (= GO in the app) when the arcade button (GPIO 4 to GND) is pressed. Powered by a breadboard power module on 5V.
  The Python Bluetooth route (bleak) is out: Windows Smart App Control blocks its winrt DLL. The serial ESP32 link
  stays off (`esp32.enabled: false`); no physical STOP button (`stop_switch: none`). STOP = voice, spacebar, UI,
  looking away, power switch. The keyboard types into the focused window: keep the Spoonful tab in front.
- **One arcade button, used as GO** (2026-11-13, `esp32.button_role: go`, `gate.require_go: true`). Press while
  LISTENING = request a bite; press while holding at pre_delivery = bring it in (counts for go_window_s, one press per
  approach, still needs the face gate). STOP is voice "stop", spacebar, UI STOP, or looking away. Enter = GO on the
  keyboard. Set `button_role: stop` to go back to the original STOP-button design if a second button turns up.
- **Positions past the 0 end read as 65535, 65533...** The servo reports a signed 16-bit value. Both backends and
  bringup_arm.py now decode it as signed (fixed 2026-11-13).

- Battery voltage at idle / under motion:
- Pot noise with torque on (tests/hw/test_arm_limits.py::test_noise_floor):
- Link lengths (ruler, axis to axis, 2026-11-13): base height 67.2, upper arm 94.9, forearm 90.4, wrist-pitch axis
  to gripper servo front face 91.8 (+100 CAD spoon tip ahead of that face = tool 191.8 mm). Base +units turns the
  arm to its own left (diner's right), so base sign +1.
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
- Reach (measured links, 2026-11-13): a level spoon at 250 mm height reaches at most ~297 mm from the base axis, at
  full stretch. Plan for the diner's mouth about 260-280 mm from the base axis, with the plate on the arm's left.
  Sim placeholder mouth_nominal moved from 300 to 280 mm and the sim bowl to the arm's left (+40 deg).