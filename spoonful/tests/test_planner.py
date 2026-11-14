"""Open-ended requests: Gemini plans with skills; plans are verified; bites still need GO."""
import time

from conftest import wait_state
from test_fsm import start_listening
from spoonful.brain.planner import scene_text, verify

BOWLS = [{"name": "bowl1", "food": "applesauce", "amount": "full", "texture": "soft"},
         {"name": "bowl2", "food": "rice", "amount": "low", "texture": "thick"},
         {"name": "bowl3", "food": "soup", "amount": "empty", "empty": True}]


def test_verify_keeps_only_safe_known_steps():
    plan = {"plan": [{"skill": "say", "text": "Applesauce is sweet."},
                     {"skill": "select", "bowl": "bowl1"},
                     {"skill": "select", "bowl": "bowl9"},                  # invented bowl
                     {"skill": "queue", "bowls": ["bowl1", "bowl3", "bowl2"]},  # bowl3 is empty
                     {"skill": "move_arm", "to": [0, 0, 0]},                # not a skill
                     {"skill": "queue", "bowls": ["bowl1"] * 20}]}          # too many
    steps, problems = verify(plan, BOWLS)
    assert steps[0] == {"skill": "say", "text": "Applesauce is sweet."}
    assert {"skill": "select", "bowl": "bowl1"} in steps
    assert {"skill": "queue", "bowls": ["bowl1", "bowl2"]} in steps
    assert all(s["skill"] != "move_arm" for s in steps)
    assert max(len(s.get("bowls", [])) for s in steps) <= 5
    assert any("bowl9" in p for p in problems) and any("bowl3" in p for p in problems)
    assert verify("nonsense", BOWLS) == ([], ["plan is not a list"])


def test_scene_text_mentions_state():
    t = scene_text(BOWLS)
    assert "bowl2: rice" in t and "amount left: low" in t and "EMPTY" in t


def _speech(s):
    s.log.flush()
    return [r["detail"] for r in s.log.query("SELECT detail FROM events WHERE type='speech' ORDER BY rowid")]


def test_a_little_of_each_queues_one_bite_per_press(sim_system):
    s = sim_system
    s.cfg.voice.requests, s.cfg.voice.select = False, True
    s.plate.set_label("bowl1", "applesauce")
    s.plate.set_label("bowl2", "rice")
    prompts = []

    def fake_gemini(prompt):
        prompts.append(prompt)
        return {"plan": [{"skill": "say", "text": "Applesauce first, then rice. Press the button when you're ready."},
                         {"skill": "queue", "bowls": ["bowl1", "bowl2"]}]}
    s.plate.text_generate = fake_gemini
    start_listening(s)
    s.on_voice(None, "Spoonful, could I have a little bit of each one please", 0.9)
    t0 = time.time()
    while not s.fsm.bite_queue and time.time() - t0 < 3:
        time.sleep(0.02)
    assert s.fsm.bite_queue == ["bowl1", "bowl2"] and s.fsm.state.value == "LISTENING"   # nothing moved
    assert "applesauce" in prompts[0] and "rice" in prompts[0]            # it saw the scene
    assert any("Applesauce first" in d for d in _speech(s))
    s.fsm.post("go", source="button")
    wait_state(s, "SCOOPING")
    assert s.fsm.bite.bowl == "bowl1" and s.fsm.bite_queue == ["bowl2"]


def test_planner_failure_falls_back_and_never_moves_by_itself(sim_system):
    s = sim_system
    s.cfg.voice.requests, s.cfg.voice.select = False, True
    s.plate.set_label("bowl1", "applesauce")

    def broken(prompt):
        raise RuntimeError("429 quota")
    s.plate.text_generate = broken
    start_listening(s)
    s.on_voice(None, "Spoonful, I'd really love some of the applesauce right now", 0.9)
    time.sleep(0.5)
    assert s.fsm.state.value == "LISTENING" and s.fsm.last_bowl == "bowl1"   # fell back to the name match
