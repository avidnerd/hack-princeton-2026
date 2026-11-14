import pytest

from spoonful.brain.pace import Pace, intervals
from spoonful.brain.profiles import Profile


def bites(gaps, sid=1, source="voice"):
    t, out = 1000.0, []
    for g in [0] + gaps:
        t += g
        out.append({"session_id": sid, "ts_request": t, "ts_done": t, "source": source})
    return out


def test_default_until_enough_bites(cfg):
    p = Pace(cfg)
    assert p.refit(bites([10, 12]), Profile("x")) == cfg.pace.default_interval_s
    assert not p.learned


def test_median_after_five(cfg):
    p = Pace(cfg)
    iv = p.refit(bites([10, 12, 11, 30, 13, 12]), Profile("x"))
    assert p.learned and iv == pytest.approx(12)


def test_pace_scale_and_minimum(cfg):
    p = Pace(cfg)
    assert p.refit(bites([10] * 6), Profile("x", pace_scale=1.25)) == pytest.approx(12.5)
    assert p.refit(bites([1] * 6), Profile("x")) == cfg.pace.min_interval_s


def test_breaks_and_auto_offers_ignored():
    assert intervals(bites([10, 500, 10])) == [10, 10]
    assert intervals(bites([10, 10], source="pace")) == []


def test_swallow_moves_next_offer(cfg):
    p = Pace(cfg)
    assert p.suggest_next_offer_time(100.0) == 100.0 + p.interval_s
    p.on_piezo(cfg.pace.swallow_threshold + 1, now=105.0)
    assert p.suggest_next_offer_time(100.0) == pytest.approx(105.0 + cfg.pace.swallow_offset_s)
    assert not p.on_piezo(cfg.pace.swallow_threshold + 5, now=105.1)   # still high: no new edge
