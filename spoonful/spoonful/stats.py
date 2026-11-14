"""Pitch numbers, all from the log: bites, people, contacts, holds, aborts, mean bite time, p95 gate latency."""
from __future__ import annotations

import statistics

from .log import Log


def _p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def gate_close_latencies(log: Log) -> list[float]:
    """Time from the first non-OK frame after an OK run to the arm's 'permit lost' transition (ms)."""
    rows = log.query("SELECT ts, ok FROM gate ORDER BY ts")
    out = []
    prev_ok, t_drop = False, None
    for r in rows:
        if prev_ok and not r["ok"]:
            t_drop = r["ts"]
        prev_ok = bool(r["ok"])
        if t_drop is not None:
            out.append(t_drop)
            t_drop = None
    lost = [r["ts"] for r in log.query(
        "SELECT ts FROM events WHERE type='state' AND state_from='DELIVERING' ORDER BY ts")]
    lat = []
    for t in out:
        nxt = [x for x in lost if x >= t]
        if nxt and nxt[0] - t < 2:
            lat.append((nxt[0] - t) * 1000)
    return lat


def compute(log: Log) -> dict:
    bites = log.query("SELECT * FROM bites")
    taken = [b for b in bites if b["outcome"] == "taken"]
    durations = [b["ts_done"] - b["ts_request"] for b in bites if b["ts_done"] and b["ts_request"]]
    frame_age = [r["age_ms"] for r in log.query("SELECT age_ms FROM gate WHERE age_ms >= 0 ORDER BY ts DESC LIMIT 20000")]
    contacts = log.query("SELECT COUNT(*) n FROM faults WHERE kind='contact'")[0]["n"]
    faults = log.query("SELECT kind, COUNT(*) n FROM faults GROUP BY kind")
    lat = gate_close_latencies(log)
    return {
        "bites": len(bites), "taken": len(taken),
        "people": len({b["profile"] for b in bites}),
        "sessions": log.query("SELECT COUNT(*) n FROM sessions")[0]["n"],
        "contacts": contacts,
        "holds": sum(b["holds"] or 0 for b in bites), "aborts": sum(b["aborts"] or 0 for b in bites),
        "outcomes": {o: sum(1 for b in bites if b["outcome"] == o) for o in {b["outcome"] for b in bites}},
        "mean_bite_s": round(statistics.mean(durations), 1) if durations else None,
        "p95_frame_age_ms": round(_p95(frame_age), 1) if frame_age else None,
        "p95_gate_close_ms": round(_p95(lat), 1) if lat else None,
        "faults": {f["kind"]: f["n"] for f in faults},
    }
