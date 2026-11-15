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


def test_bite_does_not_wait_for_the_writers_open_transaction(tmp_path):
    """The FSM logs a bite on its own thread at the end of every bite (also on STOP). While the writer thread had
    rows pending, that insert used to wait about a quarter of a second for the write lock, longer than the
    watchdog allows an FSM tick."""
    import time
    log = Log(tmp_path / "t.db")
    for k in range(3):
        for i in range(20):
            log.servo(time.time(), "elbow", float(i), float(i))    # the writer now holds an open transaction
        time.sleep(0.05)
        t0 = time.perf_counter()
        rid = log.bite(session_id=1, profile="demo", ts_request=float(k), ts_delivered=None, ts_done=2.0,
                       outcome="aborted", holds=0, aborts=0, offset_mm=0.0, source="test")
        took = time.perf_counter() - t0
        assert took < 0.15, f"bite() blocked for {took * 1000:.0f} ms"
        assert rid == k + 1 and len(log.bites_for("demo")) == k + 1       # and it can be read back at once
    log.close()
    assert log.bite(session_id=1, profile="demo", ts_request=9.0, outcome="taken") == 4    # no writer: still written
