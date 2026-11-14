"""FSM end to end on the sim backend + simulated face. No hardware."""
import time

from conftest import wait_state

from spoonful.arm import kinematics as K
from spoonful.brain.fsm import ANY, TRANSITIONS, S, allowed


def start_listening(s):
    wait_state(s, "IDLE")
    s.fsm.post("session_start")
    wait_state(s, "LISTENING")


def bites(s):
    s.log.flush()
    return s.log.query("SELECT outcome, holds, aborts FROM bites ORDER BY id")


def test_transition_table_is_closed():
    states = {x for x in S}
    for f, t, why in TRANSITIONS:
        assert f == ANY or f in states
        assert t in states and why
    assert allowed(S.DELIVERING, S.STOPPED) and allowed(S.LISTENING, S.FAULT)
    assert not allowed(S.LISTENING, S.DELIVERING)       # no skipping the hold
    assert not allowed(S.STOPPED, S.LISTENING)          # stop only releases through HOMING


def test_full_bite_taken(sim_system):
    s = sim_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=False)
    s.fsm.post("bite", source="test")
    wait_state(s, "PRE_DELIVERY_HOLD")
    time.sleep(0.4)
    assert s.fsm.state == S.PRE_DELIVERY_HOLD          # mouth closed: never advances
    s.face.sim["mouth_open"] = True
    wait_state(s, "MOUTH_HOLD")
    s.face.sim["mouth_open"] = False
    wait_state(s, "LISTENING")
    assert bites(s)[-1]["outcome"] == "taken"


def test_no_permit_times_out_and_retreats(sim_system):
    s = sim_system
    start_listening(s)
    s.face.sim.update(present=False)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    wait_state(s, "RETREAT", timeout=6)
    wait_state(s, "LISTENING")
    assert bites(s)[-1]["outcome"] == "timeout"


def test_turn_away_twice_aborts(sim_system):
    s = sim_system
    start_listening(s)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    for _ in range(2):
        s.face.sim.update(present=True, mouth_open=True, yaw=0)
        wait_state(s, "DELIVERING")
        time.sleep(0.05)
        s.face.sim["yaw"] = 40
        time.sleep(0.15)
    wait_state(s, "LISTENING")
    b = bites(s)[-1]
    assert b["outcome"] == "aborted" and b["aborts"] == 2


def test_arm_never_approaches_without_permit(sim_system):
    """Ask the arm to servo straight at the mouth while the gate denies. It may move up to the face-zone boundary
    (no permit needed outside it) but every setpoint that would enter the zone toward the mouth is refused."""
    s = sim_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=False)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    s.arm.servo_toward(s.lib.pose("mouth_nominal"))
    for _ in range(40):
        time.sleep(0.025)
        assert not K.in_face_zone(s.cfg.kinematics, s.arm.setpoint)
    assert s.arm.last_deny == "approach_outside_delivering"
    assert s.arm.mode == "hold"


def test_stop_latched_until_button_and_confirm(sim_system):
    s = sim_system
    start_listening(s)
    s.fsm.post("bite")
    wait_state(s, "SCOOPING")
    s.fsm.post("stop", source="ui")
    wait_state(s, "STOPPED")
    time.sleep(s.cfg.arm.stop_torque_off_after_s + 0.3)
    assert not s.arm.torque_on
    s.fsm.post("reset")
    time.sleep(0.2)
    assert s.fsm.state == S.STOPPED                    # UI alone cannot release
    s.fsm.post("stop", source="ui")                    # a UI stop does not arm the release either
    s.fsm.post("reset")
    time.sleep(0.2)
    assert s.fsm.state == S.STOPPED
    s.fsm.post("button")
    s.fsm.post("reset")
    wait_state(s, "IDLE")


def test_pause_resume_mid_bite_retreats(sim_system):
    s = sim_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=False)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    s.fsm.post("pause", source="voice")
    wait_state(s, "PAUSED")
    s.fsm.post("resume")
    wait_state(s, "LISTENING")
    assert s.session.pauses == 1


def test_contact_outside_face_zone_faults_and_parks(sim_system):
    s = sim_system
    start_listening(s)
    s.fsm.post("bite")
    wait_state(s, "SCOOPING")
    for j in ("base", "shoulder", "elbow", "wrist_pitch"):
        s.backend.jam(s.cfg.arm.servo_map[j], True)
    wait_state(s, "FAULT", timeout=10)
    assert s.fsm.fault_reason == "contact"
    for j in ("base", "shoulder", "elbow", "wrist_pitch"):
        s.backend.jam(s.cfg.arm.servo_map[j], False)
    s.fsm.post("reset")
    wait_state(s, "IDLE")


def test_enough_ends_session_with_summary(sim_system):
    s = sim_system
    start_listening(s)
    s.fsm.post("enough", source="voice")
    wait_state(s, "IDLE")
    assert s.summary_outbox and "Spoonful:" in s.summary_outbox[-1]["text"]


def test_watchdog_fsm_stall(sim_system):
    s = sim_system
    start_listening(s)
    s.fsm.stop_thread()          # wedge the FSM
    time.sleep(0.4)
    assert s.watchdog.check(time.monotonic()) == "fsm_stalled"
    assert s.watchdog.trips and s.watchdog.trips[-1][1] == "fsm_stalled"
