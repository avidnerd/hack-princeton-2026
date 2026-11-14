#!/usr/bin/env python
"""Render the FSM transition table and the layered-stop list into SAFETY.md (between the AUTOGEN markers)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spoonful import config as C  # noqa: E402
from spoonful.brain.fsm import TRANSITIONS  # noqa: E402
from spoonful.perception.gate import REASONS  # noqa: E402

cfg = C.load()
f, a, fs, g = cfg.face, cfg.arm, cfg.fsm, cfg.gate

rows = ["| From | To | Trigger |", "|---|---|---|"]
for src, dst, why in TRANSITIONS:
    rows.append(f"| {'any' if src == '*' else src.value} | {dst.value} | {why} |")

layers = f"""
### Motion pipeline (every setpoint, every {1000 / a.stream_hz:.0f} ms, in `arm/arm.py`)
1. **Soft joint limits**: taught waypoints outside the limits are rejected at load (`LimitViolation`); live
   servo targets are clamped.
2. **Speed / accel limits**: {a.max_speed_deg_s.default:.0f} deg/s, {a.max_speed_deg_s.near_face:.0f} deg/s whenever the spoon
   tip is in the face zone, {a.max_accel_deg_s2:.0f} deg/s^2, min-jerk profiles, per-tick step cap.
3. **FSM permission**: no motion in STOPPED, PAUSED or BOOT. Inside the face zone, a setpoint that brings the tip
   closer to the mouth is refused unless the state is DELIVERING.
4. **Gate permit**: in DELIVERING the last verdict must be a PERMIT, at most 200 ms old.
5. **Watchdog liveness**: setpoints stop if the FSM heartbeat is older than {a.heartbeat_timeout_s * 1000:.0f} ms.
Exemption: `park` (slow, near-face speed, toward the folded rest pose) skips layers 3 to 5 so the arm can always
move away.

### Gate (consent), `perception/gate.py`
Permit only when all hold: face present >= {f.presence_hold_s} s; |yaw| <= {f.yaw_max_deg} deg; |pitch| <= {f.pitch_max_deg} deg;
mouth in the delivery window; mouth open ratio >= {f.mouth_open_ratio_threshold} for >= {f.mouth_open_hold_s} s; frame <= {f.frame_max_age_ms:.0f} ms old.
Hysteresis: once permitted, the mouth may sag to {f.hysteresis_fraction:.0%} of the threshold.
Deny reasons: {", ".join(f"`{r}`" for r in REASONS)}.

### Layered stops (fastest first)
| Layer | Trigger | Effect |
|---|---|---|
| Power rocker switch | hand | servos lose power. Independent of all software. |
| STOP button (ESP32) / spacebar / voice "stop" / UI STOP | diner, caregiver | STOPPED: hold, torque off after {a.stop_torque_off_after_s} s, latched. Release = physical press + UI confirm within {fs.stop_release_window_s:.0f} s, then slow HOMING. |
| Gate deny while DELIVERING | diner turns away, closes mouth, leaves | hold, back off {g.backoff_mm:.0f} mm; a second loss in one bite retreats. |
| Contact (commanded vs actual > {a.contact_error_deg} deg for {a.contact_hold_s} s) | spoon meets resistance | near face: retreat; elsewhere: FAULT + park. |
| Watchdog | FSM stalled 250 ms / camera gone 500 ms mid-bite / no arm reads 500 ms | hold then park; FAULT. |
| Camera lost > {fs.camera_fault_s} s | camera | FAULT + park. |
| HID errors x3 | USB | hold, FAULT. |
| Voice "pause" / UI PAUSE | diner | PAUSED: hold; resume mid-bite retreats first. |
"""

block = "## FSM transitions\n\n" + "\n".join(rows) + "\n" + layers
path = ROOT / "SAFETY.md"
text = path.read_text() if path.exists() else "# Safety\n\n<!-- AUTOGEN START -->\n<!-- AUTOGEN END -->\n"
start, end = "<!-- AUTOGEN START -->", "<!-- AUTOGEN END -->"
pre, rest = text.split(start, 1)
_, post = rest.split(end, 1)
path.write_text(pre + start + "\n" + block + "\n" + end + post)
print(f"wrote {path} ({len(TRANSITIONS)} transitions)")
