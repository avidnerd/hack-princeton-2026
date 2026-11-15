# Safety

Generated sections are rebuilt by `scripts/make_safety_table.py` from the code (the FSM transition table in
`spoonful/brain/fsm.py` and config.yaml). Do not edit between the markers.

**Principle:** the arm moves toward a face only while the person is looking at it with their mouth open, backs off the
moment they turn away, and never moves without a permit. Safety is architecture: every setpoint passes the same five
layers, in one place (`Arm._guard`).

<!-- AUTOGEN START -->
## FSM transitions

| From | To | Trigger |
|---|---|---|
| BOOT | HOMING | arm connected, trajectories validated |
| HOMING | IDLE | home pose reached |
| IDLE | LISTENING | session start |
| LISTENING | IDLE | session end / "that's enough" |
| LISTENING | SCOOPING | bite request: voice "bite", button, UI, or pace offer (auto-pace opted in) |
| SCOOPING | LIFTING | scoop path done (outside face zone, no permit needed) |
| SCOOPING | RETREAT | no scoop: the cup was not seen or has no safe entry, the spoon touched the cup, or it felt something above the food |
| LIFTING | PRE_DELIVERY_HOLD | lift path done, spoon at pre_delivery (outside face zone) |
| PRE_DELIVERY_HOLD | DELIVERING | gate PERMIT, plus a GO press within go_window_s if gate.require_go (one press per approach) |
| PRE_DELIVERY_HOLD | RETREAT | no permit for pre_delivery_timeout_s, or retreat_after_s after a back-off |
| PRE_DELIVERY_HOLD | SCOOPING | face camera sees an empty spoon: re-scoop (at most plate.spoon_check.max_rescoops) |
| DELIVERING | PRE_DELIVERY_HOLD | permit lost: back off backoff_mm, wait again |
| DELIVERING | RETREAT | permit lost twice in one bite, or contact near the face |
| DELIVERING | MOUTH_HOLD | spoon tip within mouth_arrive_mm of the (re-targeted) mouth |
| MOUTH_HOLD | RETREAT | mouth closed mouth_closed_taken_s (taken), hold_s elapsed, or diner turned away |
| RETREAT | LISTENING | retreat done, bite logged |
| RETREAT | IDLE | retreat done after "that's enough" |
| any | PAUSED | voice "pause" / UI PAUSE (from an active state): hold position |
| PAUSED | LISTENING | resume with the spoon at rest |
| PAUSED | RETREAT | resume mid-bite: retreat first |
| any | STOPPED | STOP switch off or unreadable / STOP button / spacebar / voice "stop" / UI STOP: hold position (torque off after arm.stop_torque_off_after_s; < 0 = keeps holding) (latched) |
| STOPPED | HOMING | STOP press (button or spacebar) or STOP switch back on, THEN UI confirm within stop_release_window_s; never while the switch is off; GO never releases |
| any | FAULT | contact outside face zone, HID errors, camera lost > camera_fault_s, watchdog trip: park |
| FAULT | HOMING | operator reset in the UI |

### Motion pipeline (every setpoint, every 40 ms, in `arm/arm.py`)
1. **Soft joint limits**: taught waypoints outside the limits are rejected at load (`LimitViolation`); live
   servo targets are clamped.
2. **Speed / accel limits**: 40 deg/s, 12 deg/s whenever the spoon
   tip is in the face zone, 120 deg/s^2, min-jerk profiles, per-tick step cap.
3. **FSM permission**: no motion in STOPPED, PAUSED or BOOT. Inside the face zone, a setpoint that brings the tip
   closer to the mouth is refused unless the state is DELIVERING.
4. **Gate permit**: in DELIVERING the last verdict must be a PERMIT, at most 200 ms old.
5. **Watchdog liveness**: setpoints stop if the FSM heartbeat is older than 250 ms.
Exemption: `park` (slow, near-face speed, toward the folded rest pose) skips layers 3 to 5 so the arm can always
move away.

### Gate (consent), `perception/gate.py`
Permit only when all hold: face present >= 0.5 s; |yaw| <= 22.0 deg; |pitch| <= 20.0 deg;
mouth in the delivery window; mouth open ratio >= 0.08 for >= 0.3 s; frame <= 150 ms old.
Hysteresis: once permitted, the mouth may sag to 70% of the threshold.
Deny reasons: `stop_latched`, `paused`, `frame_stale`, `no_face`, `face_too_brief`, `looking_away`, `mouth_outside_window`, `mouth_closed`.

### Layered stops (fastest first)
| Layer | Trigger | Effect |
|---|---|---|
| Power rocker switch | hand | servos lose power. Independent of all software. |
| STOP switch (ESP32 GPIO 5: rocker at O, wire broken, or ESP32 lost) | diner, caregiver | STOPPED within one serial message (ms), held while the switch is off; flipping it back on arms the release. |
| STOP button (ESP32) / spacebar / voice "stop" / UI STOP | diner, caregiver | STOPPED: hold, torque stays on (a limp xArm folds past its servo range), latched. Release = STOP press, spacebar or switch back on + UI confirm within 30 s, then slow HOMING. A GO press never stops or releases. |
| Gate deny while DELIVERING | diner turns away, closes mouth, leaves | hold, back off 50 mm; a second loss in one bite retreats. |
| Contact (commanded vs actual > 4.8 deg for 0.3 s) | spoon meets resistance | near face: retreat; elsewhere: FAULT + park. |
| Watchdog | FSM stalled 250 ms / camera gone 500 ms mid-bite / no arm reads 1 s | hold then park; FAULT. |
| Camera lost > 2.0 s | camera | FAULT + park. |
| HID errors x3 | USB | hold, FAULT. |
| Voice "pause" / UI PAUSE | diner | PAUSED: hold; resume mid-bite retreats first. |

<!-- AUTOGEN END -->

## Known limits (be honest with judges)
- No force sensing: contact is inferred from position error, so a soft push may not register. Speeds near the face
  are capped at 12 deg/s and the spoon is silicone on a magnet breakaway cradle.
- Face-zone geometry relies on hand-measured link lengths; preflight checks that the taught poses land on the right
  side of the zone.
- The gate trusts MediaPipe. A photo of a face with an open mouth would be permitted; the operator is present.
