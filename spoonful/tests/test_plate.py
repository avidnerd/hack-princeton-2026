"""Plate awareness: which food is in which bowl, voice requests by food name, per-bowl scoops."""
import time

import numpy as np
import pytest

from conftest import wait_state
from test_fsm import start_listening
from spoonful.perception.plate import PlateWatcher, asks_whats_on_plate, describe, match_food

LABELS = {"bowl1": "applesauce", "bowl2": "mashed potatoes", "bowl3": "yogurt"}


@pytest.mark.parametrize("text,bowl", [
    ("Spoonful, a bite of the applesauce please", "bowl1"),
    ("spoonful apple sauce", "bowl1"),
    ("can I have some potatoes", "bowl2"),
    ("Spoonful, mashed potatoes!", "bowl2"),
    ("some yoghurt", "bowl3"),       # a respelling still counts
    ("Give me the apples.", "bowl1"),  # what the recogniser made of "applesauce"
    ("could I get some jello", None),  # nothing like it on this table
    ("Spoonful, yogurt", "bowl3"),
    ("the carrots please", None),
    ("", None),
])
def test_match_food(text, bowl):
    assert match_food(text, LABELS) == bowl


def test_match_ignores_unknown_bowls_and_ambiguity():
    assert match_food("rice", {"bowl1": None, "bowl2": "rice"}) == "bowl2"
    assert match_food("some rice", {"bowl1": "brown rice", "bowl2": "white rice"}) is None
    assert match_food("brown rice", {"bowl1": "brown rice", "bowl2": "white rice"}) == "bowl1"


def test_plate_questions_and_describe():
    assert asks_whats_on_plate("Spoonful, what's on my plate?")
    assert asks_whats_on_plate("spoonful what is on the plate")
    assert not asks_whats_on_plate("spoonful applesauce please")
    assert describe(["applesauce", "yogurt"]) == "I see applesauce and yogurt"
    assert describe([]) == "I don't see any food yet"


def _watcher(cfg, results):
    cfg.plate.enabled = True
    calls = []

    def vision(crops):
        calls.append([name for name, _ in crops])
        return results.pop(0) if results else []

    img = np.zeros((720, 1280, 3), np.uint8)
    return PlateWatcher(cfg, lambda: img, vision_fn=vision), calls


def test_watcher_labels_empty_and_override(fast_cfg):
    w, calls = _watcher(fast_cfg, [
        [{"bowl": "bowl1", "food": "Applesauce", "amount": "full"}, {"bowl": "bowl2", "food": "yogurt", "amount": "low"}],
        [{"bowl": "bowl1", "food": "applesauce", "amount": "empty"}, {"bowl": "bowl2", "food": "yogurt", "amount": "low"}],
        [{"bowl": "bowl1", "food": "applesauce", "amount": "empty"}, {"bowl": "bowl2", "food": "rice", "amount": "half"}],
    ])
    assert w.check_now() and calls[0] == ["bowl1", "bowl2"]
    assert w.labels() == {"bowl1": "applesauce", "bowl2": "yogurt"}
    assert w.find("spoonful applesauce") == "bowl1"
    assert w.check_now() and not w.is_empty("bowl1")         # one "empty" is not enough
    w.set_label("bowl2", "Yogurt")                           # operator override sticks...
    assert w.check_now() and w.is_empty("bowl1")
    assert w.labels()["bowl2"] == "yogurt"                   # ...even when vision says rice
    assert w.describe() == "I see yogurt"
    w.set_label("bowl2", "")
    assert w.labels()["bowl2"] is None                       # override cleared, no config label


def test_watcher_survives_vision_errors(fast_cfg):
    fast_cfg.plate.enabled = True

    def broken(crops):
        raise RuntimeError("403 blocked")

    w = PlateWatcher(fast_cfg, lambda: np.zeros((360, 640, 3), np.uint8), vision_fn=broken)
    assert not w.check_now() and "403" in w.error
    w2 = PlateWatcher(fast_cfg, lambda: None, vision_fn=broken)
    assert not w2.check_now() and "no plate camera frame" in w2.error


def test_rois_scale_to_the_frame(fast_cfg):
    fast_cfg.plate.enabled = True
    w = PlateWatcher(fast_cfg, lambda: None, vision_fn=lambda c: [])
    full = w.crops(np.zeros((720, 1280, 3), np.uint8))
    half = w.crops(np.zeros((360, 640, 3), np.uint8))
    assert [n for n, _ in full] == [n for n, _ in half] == ["bowl1", "bowl2"]


# ---------------------------------------------------------------- end to end in the sim
@pytest.fixture
def plate_system(sim_system):
    s = sim_system
    s.plate.set_label("bowl1", "applesauce")
    s.plate.set_label("bowl2", "yogurt")
    ran = []                                   # the moves each plan sent to the arm was made of (by name)
    orig = s.arm.run

    def spy(p):
        names = [seg.label for seg in p.segments if not seg.label.startswith("blend->")]
        if names and names[0] != "noop":
            ran.append(names)
        return orig(p)

    s.arm.run = spy
    return s, ran


def refusals(s):
    s.log.flush()
    return [r["detail"] for r in s.log.query("SELECT detail FROM events WHERE type='bowl_refused' ORDER BY rowid")]


def test_voice_request_scoops_that_bowl(plate_system):
    s, ran = plate_system
    start_listening(s)
    s.on_voice(None, "Spoonful, a bite of the yogurt please", 0.9)
    wait_state(s, "SCOOPING")
    assert s.fsm.bite.bowl == "bowl2" and ran[-1] == ["scoop_bowl2"]


def test_go_repeats_the_last_food(plate_system):
    s, ran = plate_system
    start_listening(s)
    s.on_voice("bite", "Spoonful, more applesauce", 0.9)
    wait_state(s, "SCOOPING")
    assert ran[-1] == ["scoop_bowl1"]
    s.fsm.post("stop", source="key")                  # abandon this bite
    wait_state(s, "STOPPED")
    s.fsm.post("stop", source="key")                  # physical half of the release
    s.fsm.post("reset")
    wait_state(s, "IDLE")
    s.fsm.post("session_start")
    wait_state(s, "LISTENING")
    s.fsm.post("go", source="button")
    wait_state(s, "SCOOPING")
    assert s.fsm.bite.bowl == "bowl1" and ran[-1] == ["scoop_bowl1"]


def test_empty_and_untaught_bowls_are_refused(plate_system):
    s, ran = plate_system
    start_listening(s)
    s.plate.bowls["bowl2"]["empty_count"] = 5
    s.plate.manual.pop("bowl2")                       # an operator label would reset the empty count
    s.on_voice(None, "Spoonful, yogurt please", 0.9)
    time.sleep(0.3)
    assert s.fsm.state.value == "LISTENING" and "empty" in refusals(s)[-1]
    s.lib.items.pop("scoop_bowl1")
    s.on_voice(None, "Spoonful, applesauce", 0.9)
    time.sleep(0.3)
    assert s.fsm.state.value == "LISTENING" and "not taught" in refusals(s)[-1]
    assert ran == []


def test_unknown_food_and_plain_bite(plate_system):
    s, ran = plate_system
    start_listening(s)
    s.on_voice(None, "Spoonful, can I have the carrots", 0.9)   # not on the plate: nothing moves
    time.sleep(0.3)
    assert s.fsm.state.value == "LISTENING"
    s.fsm.post("bite", source="ui")                           # no food named, none asked before: default scoop
    wait_state(s, "SCOOPING")
    assert s.fsm.bite.bowl is None and ran[-1] == [s.profiles.current.scoop_path]


def test_voice_stop_still_wins(plate_system):
    s, ran = plate_system
    start_listening(s)
    s.on_voice(None, "Spoonful, applesauce", 0.9)
    wait_state(s, "SCOOPING")
    s.on_voice("stop", "stop the applesauce", 0.2)
    wait_state(s, "STOPPED")


def test_gemini_falls_back_to_the_next_model(fast_cfg):
    fast_cfg.plate.enabled = True
    fast_cfg.plate.model, fast_cfg.plate.fallback_models = "busy-model", ["ok-model"]
    tried = []

    class Resp:
        text = '[{"bowl": "bowl1", "food": "applesauce", "amount": "full"}]'

    class Models:
        def generate_content(self, model, contents, config):
            tried.append(model)
            if model == "busy-model":
                raise RuntimeError("503 UNAVAILABLE high demand")
            return Resp()

    class Client:
        models = Models()

    w = PlateWatcher(fast_cfg, lambda: None, vision_fn=lambda c: [])
    assert w._gemini(Client(), [("bowl1", b"x")])[0]["food"] == "applesauce"
    assert tried == ["busy-model", "ok-model"] and w.model_used == "ok-model"
    w._gemini(Client(), [("bowl1", b"x")])
    assert tried[-1] == "ok-model"                     # the model that worked is tried first next time


def test_configured_label_is_not_overridden_by_vision(fast_cfg):
    fast_cfg.plate.bowls[0].label = "applesauce"
    w, _ = _watcher(fast_cfg, [[{"bowl": "bowl1", "food": "broth", "amount": "full", "texture": "liquid"}]])
    assert w.check_now()
    assert w.food_of("bowl1") == "applesauce" and w.texture_of("bowl1") == ("soft", "food name")


# ---------------------------------------------------------------- demo flow: voice picks the food, the button bites
def _speech(s):
    s.log.flush()
    return [r["detail"] for r in s.log.query("SELECT detail FROM events WHERE type='speech' ORDER BY rowid")]


def test_voice_selects_food_and_button_gives_that_bite(plate_system):
    s, ran = plate_system
    s.cfg.voice.requests, s.cfg.voice.select = False, True
    start_listening(s)
    s.on_voice(None, "Spoonful, give me the applesauce", 0.9)
    time.sleep(0.3)
    assert s.fsm.state.value == "LISTENING" and ran == []           # picked, not moving
    assert any("Applesauce. Ready when you are" in d for d in _speech(s))
    s.fsm.post("go", source="button")
    wait_state(s, "SCOOPING")
    assert s.fsm.bite.bowl == "bowl1" and ran[-1] == ["scoop_bowl1"]
    s.fsm.post("stop", source="key"); wait_state(s, "STOPPED")
    s.fsm.post("stop", source="key"); s.fsm.post("reset"); wait_state(s, "IDLE")
    s.on_voice(None, "Spoonful, ok now switch to yogurt", 0.9)       # switch while idle
    time.sleep(0.3)
    assert any("Yogurt. Ready when you are" in d for d in _speech(s))
    s.fsm.post("go", source="button")                                  # GO in IDLE starts the session too
    wait_state(s, "SCOOPING")
    assert s.fsm.bite.bowl == "bowl2" and ran[-1] == ["scoop_bowl2"]


def test_select_unknown_food_says_so(plate_system):
    s, ran = plate_system
    s.cfg.voice.requests, s.cfg.voice.select = False, True
    start_listening(s)
    s.on_voice(None, "Spoonful, switch to the carrots", 0.9)
    time.sleep(0.3)
    assert any("I don't see that" in d for d in _speech(s)) and ran == []


def test_unwatched_bowls_are_never_judged_from_the_webcam(fast_cfg):
    """A bowl set up from an overhead photo (watch=False) isn't in the webcam's view: the periodic check must not
    look for it there (it would see no food and call the bowl finished)."""
    for b in fast_cfg.plate.bowls:
        b.watch, b.label = False, "jelly"
    w, calls = _watcher(fast_cfg, [[{"bowl": "bowl1", "food": None, "amount": "empty"}]] * 5)
    for _ in range(4):
        assert not w.check_now()
    assert calls == [] and w.error is None and not w.is_empty("bowl1") and w.food_of("bowl1") == "jelly"



def test_food_named_without_the_wake_word_is_still_picked(plate_system):
    # The recogniser sometimes loses "Spoonful". A food name alone picks the food (nothing moves until GO).
    s, ran = plate_system
    s.cfg.voice.requests, s.cfg.voice.select = False, True
    start_listening(s)
    s.on_unaddressed_voice(" Give me the apples.", 0.5)
    time.sleep(0.3)
    assert s.fsm.last_bowl == "bowl1" and s.fsm.state.value == "LISTENING" and ran == []
    s.on_unaddressed_voice("so anyway I was telling him about the game", 0.9)   # ordinary talk: ignored
    time.sleep(0.2)
    assert s.fsm.last_bowl == "bowl1"
    assert "applesauce" in s.voice_hint() and "Spoonful" in s.voice_hint()
