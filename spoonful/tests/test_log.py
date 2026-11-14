import json

from spoonful.log import Log
from spoonful.stats import compute


def test_writes_and_queries(tmp_path):
    log = Log(tmp_path / "t.db")
    sid = log.start_session("demo")
    log.event("state", "IDLE", "LISTENING", why="test")
    log.gate(True, "", 1.0, 2.0, 0.1, (640, 400), 20.0)
    log.fault("contact", joint="elbow")
    log.voice("spoonful bite", "bite", 0.8)
    log.bite(session_id=sid, profile="demo", ts_request=1.0, ts_delivered=2.0, ts_done=5.0, outcome="taken",
             holds=1, aborts=0, offset_mm=3.0, source="voice")
    log.end_session(sid, 1, 0, 1, "summary")
    log.flush()
    assert log.recent_events(5)[0]["detail"]["why"] == "test"
    assert log.bites_for("demo")[0]["outcome"] == "taken"
    assert json.loads(log.query("SELECT detail FROM faults")[0]["detail"])["joint"] == "elbow"
    assert log.sessions()[0]["summary"] == "summary"
    assert any(r["kind"] == "voice" for r in log.tail(10))
    s = compute(log)
    assert s["bites"] == 1 and s["taken"] == 1 and s["contacts"] == 1 and s["mean_bite_s"] == 4.0
    log.close()


def test_listener_notified(tmp_path):
    log = Log(tmp_path / "t.db")
    seen = []
    log.on_event(lambda k, r: seen.append(k))
    log.event("x")
    log.fault("y")
    assert seen == ["event", "fault"]
    log.close()
