"""Write synthetic gate fixtures (replace/add real ones recorded from the log during Phase 2:
`python tests/make_fixtures.py --from-log data/spoonful.db --since <ts> --until <ts> --name turn_away_real`)."""
import argparse
import json
import sqlite3
from pathlib import Path

OUT = Path(__file__).parent / "fixtures"


def frame(ts, present=True, ratio=0.15, yaw=0.0, pitch=0.0, mouth=(640, 400)):
    return {"present": present, "ts": ts, "latency_ms": 5.0, "yaw_deg": yaw, "pitch_deg": pitch,
            "mouth_open_ratio": ratio, "mouth_center_px": list(mouth) if present else None,
            "face_width_px": 220.0, "frame_w": 1280, "frame_h": 720}


def synthetic():
    t = [100 + i / 30 for i in range(60)]
    return {
        "gate_open_mouth.json": {"expect_reason": "", "frames": [frame(x) for x in t]},
        "gate_turn_away.json": {"expect_reason": "looking_away",
                                "frames": [frame(x) for x in t[:40]] + [frame(x, yaw=35) for x in t[40:]]},
        "gate_mouth_closes.json": {"expect_reason": "mouth_closed",
                                   "frames": [frame(x) for x in t[:40]] + [frame(x, ratio=0.01) for x in t[40:]]},
        "gate_face_lost.json": {"expect_reason": "no_face",
                                "frames": [frame(x) for x in t[:40]] + [frame(x, present=False) for x in t[40:]]},
    }


def from_log(db, since, until):
    con = sqlite3.connect(db)
    rows = con.execute("SELECT ts, ok, reason, yaw, pitch, mouth_ratio, mouth_px FROM gate WHERE ts BETWEEN ? AND ? "
                       "ORDER BY ts", (since, until)).fetchall()
    frames = []
    for ts, ok, reason, yaw, pitch, ratio, mpx in rows:
        m = json.loads(mpx) if mpx else None
        frames.append(frame(ts, m is not None, ratio or 0.0, yaw or 0.0, pitch or 0.0, tuple(m) if m else (0, 0)))
    return {"expect_reason": rows[-1][2] if rows else "", "frames": frames}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-log")
    ap.add_argument("--since", type=float)
    ap.add_argument("--until", type=float)
    ap.add_argument("--name", default="real")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    items = {f"gate_{a.name}.json": from_log(a.from_log, a.since, a.until)} if a.from_log else synthetic()
    for name, d in items.items():
        (OUT / name).write_text(json.dumps(d))
        print("wrote", name, len(d["frames"]), "frames, expect", repr(d["expect_reason"]))
