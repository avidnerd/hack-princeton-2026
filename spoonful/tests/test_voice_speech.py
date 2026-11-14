import pytest

from spoonful.brain.speech import LINES, Speaker
from spoonful.brain.voice import parse_command


@pytest.mark.parametrize("text,cmd", [
    ("Spoonful, bite please", "bite"),
    ("spoon full pause", "pause"),
    ("Spoonful that's enough", "enough"),
    ("Spoonful, slower", "slower"),
    ("stop!", "stop"),                           # bare stop always works
    ("spoonful bite... no wait stop", "stop"),   # stop wins
    ("I'd like a bite", None),                   # no wake word
    ("spoonful what time is it", None),
])
def test_parse(cfg, text, cmd):
    assert parse_command(text, cfg.voice) == cmd


def test_ptt_needs_no_wake(cfg):
    assert parse_command("bite", cfg.voice, require_wake=False) == "bite"


def test_speech_policy(cfg):
    cfg.speech.backend = "none"
    sp = Speaker(cfg)
    sp.backend = "none"
    assert sp.say("ready")
    assert not sp.say("ready")          # never the same line twice in a row
    assert not sp.say("here")           # cooldown
    assert sp.say("stopped")            # safety line bypasses cooldown
    assert not sp.say("i see you")      # not in the allowed set
    assert set(LINES.values()) == {"Ready when you are", "Here you go", "I'll wait", "Paused", "Taking a break",
                                   "That's the last one", "Stopped"}
