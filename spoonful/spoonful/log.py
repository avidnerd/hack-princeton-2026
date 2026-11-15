"""SQLite log (R6). One writer thread; any thread or coroutine can call the non-blocking `write_*` helpers.

Everything downstream (dashboard, caregiver summary, pace model, pitch stats) is built from these tables.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(ts REAL, type TEXT, state_from TEXT, state_to TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS gate(ts REAL, ok INTEGER, reason TEXT, yaw REAL, pitch REAL, mouth_ratio REAL,
                                mouth_px TEXT, age_ms REAL);
CREATE TABLE IF NOT EXISTS bites(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, profile TEXT,
                                 ts_request REAL, ts_delivered REAL, ts_done REAL, outcome TEXT,
                                 holds INTEGER, aborts INTEGER, offset_mm REAL, source TEXT,
                                 depth_mm REAL);
CREATE TABLE IF NOT EXISTS servo(ts REAL, joint TEXT, cmd REAL, actual REAL);
CREATE TABLE IF NOT EXISTS faults(ts REAL, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS voice(ts REAL, text TEXT, command TEXT, confidence REAL);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, profile TEXT, started REAL, ended REAL,
                                    bites INTEGER DEFAULT 0, pauses INTEGER DEFAULT 0, faults INTEGER DEFAULT 0,
                                    summary_sent INTEGER DEFAULT 0, summary TEXT);
CREATE TABLE IF NOT EXISTS swallows(ts REAL, profile TEXT, envelope REAL);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_gate_ts ON gate(ts);
CREATE INDEX IF NOT EXISTS idx_bites_profile ON bites(profile, ts_request);
"""


class Log:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._q: "queue.Queue[tuple[str, tuple] | None]" = queue.Queue()
        self._local = threading.local()
        con = self._con()
        con.executescript(SCHEMA)
        try:                                # databases from before depth tracking
            con.execute("ALTER TABLE bites ADD COLUMN depth_mm REAL")
        except sqlite3.OperationalError:
            pass
        con.commit()
        self._listeners: list = []          # callables(kind, row_dict) for the UI timeline
        self._thread = threading.Thread(target=self._run, name="log-writer", daemon=True)
        self._thread.start()

    # -- connections: one per thread (sqlite objects are thread-bound) --
    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    def _run(self) -> None:
        con = sqlite3.connect(self.path, timeout=5)
        con.execute("PRAGMA journal_mode=WAL")
        pending = 0
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                item = "flush"
            if item is None:
                con.commit()
                con.close()
                return
            if item == "flush":
                if pending:
                    con.commit()
                    pending = 0
                continue
            sql, args, *reply = item
            rowid = None
            try:
                rowid = con.execute(sql, args).lastrowid
                pending += 1
            except sqlite3.Error as e:  # never let logging take the system down
                print(f"[log] {e}: {sql}")
            if reply:                   # a caller is waiting for the row (bite): commit now so it can read it back
                try:
                    con.commit()
                    pending = 0
                except sqlite3.Error as e:
                    print(f"[log] {e}: commit")
                reply[0].put(rowid)
            if pending >= 200:
                con.commit()
                pending = 0

    def _put(self, sql: str, args: tuple) -> None:
        self._q.put((sql, args))

    def flush(self, timeout: float = 2.0) -> None:
        self._q.put("flush")
        t0 = time.time()
        while not self._q.empty() and time.time() - t0 < timeout:
            time.sleep(0.01)
        time.sleep(0.05)

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=3)

    def on_event(self, fn) -> None:
        self._listeners.append(fn)

    def _notify(self, kind: str, row: dict) -> None:
        for fn in self._listeners:
            try:
                fn(kind, row)
            except Exception:
                pass

    # -- writers (non-blocking) --
    def event(self, type_: str, state_from: str = "", state_to: str = "", **detail: Any) -> None:
        ts = time.time()
        self._put("INSERT INTO events VALUES(?,?,?,?,?)", (ts, type_, state_from, state_to, json.dumps(detail)))
        self._notify("event", {"ts": ts, "type": type_, "from": state_from, "to": state_to, "detail": detail})

    def gate(self, ok: bool, reason: str, yaw: float, pitch: float, mouth_ratio: float,
             mouth_px: Optional[tuple], age_ms: float, ts: Optional[float] = None) -> None:
        self._put("INSERT INTO gate VALUES(?,?,?,?,?,?,?,?)",
                  (ts or time.time(), int(ok), reason, yaw, pitch, mouth_ratio, json.dumps(mouth_px), age_ms))

    def servo(self, ts: float, joint: str, cmd: float, actual: float) -> None:
        self._put("INSERT INTO servo VALUES(?,?,?,?)", (ts, joint, cmd, actual))

    def fault(self, kind: str, **detail: Any) -> None:
        ts = time.time()
        self._put("INSERT INTO faults VALUES(?,?,?)", (ts, kind, json.dumps(detail)))
        self._notify("fault", {"ts": ts, "kind": kind, "detail": detail})

    def voice(self, text: str, command: Optional[str], confidence: float) -> None:
        self._put("INSERT INTO voice VALUES(?,?,?,?)", (time.time(), text, command, confidence))

    def swallow(self, profile: str, envelope: float) -> None:
        self._put("INSERT INTO swallows VALUES(?,?,?)", (time.time(), profile, envelope))

    # -- synchronous writers that need the row id --
    def start_session(self, profile: str) -> int:
        con = self._con()
        cur = con.execute("INSERT INTO sessions(profile, started) VALUES(?,?)", (profile, time.time()))
        con.commit()
        return int(cur.lastrowid)

    def end_session(self, sid: int, bites: int, pauses: int, faults: int, summary: str) -> None:
        con = self._con()
        con.execute("UPDATE sessions SET ended=?, bites=?, pauses=?, faults=?, summary=? WHERE id=?",
                    (time.time(), bites, pauses, faults, summary, sid))
        con.commit()

    def mark_summary_sent(self, sid: int) -> None:
        con = self._con()
        con.execute("UPDATE sessions SET summary_sent=1 WHERE id=?", (sid,))
        con.commit()

    def bite(self, **row: Any) -> int:
        cols = ["session_id", "profile", "ts_request", "ts_delivered", "ts_done", "outcome", "holds",
                "aborts", "offset_mm", "source", "depth_mm"]
        sql = f"INSERT INTO bites({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        args = tuple(row.get(c) for c in cols)
        # Written by the writer thread, which holds the database's write lock while it has rows pending. Inserting
        # from the caller's own connection waited for that lock: about 260 ms on the FSM thread at the end of a
        # bite, longer than the watchdog's 250 ms, so now and then the watchdog parked the arm (even after a STOP).
        rowid, reply = None, queue.Queue()
        if self._thread.is_alive():
            self._q.put((sql, args, reply))
            try:
                rowid = reply.get(timeout=2.0)
            except queue.Empty:
                pass
        else:                               # after close(): no writer to wait behind
            con = self._con()
            rowid = con.execute(sql, args).lastrowid
            con.commit()
        self._notify("bite", row)
        return int(rowid or 0)

    # -- queries --
    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        return [dict(r) for r in self._con().execute(sql, args).fetchall()]

    def recent_events(self, n: int = 20) -> list[dict]:
        rows = self.query("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (n,))
        for r in rows:
            r["detail"] = json.loads(r["detail"] or "{}")
        return rows

    def bites_for(self, profile: str, limit: int = 200) -> list[dict]:
        return self.query("SELECT * FROM bites WHERE profile=? ORDER BY ts_request DESC LIMIT ?", (profile, limit))

    def sessions(self, limit: int = 50) -> list[dict]:
        return self.query("SELECT * FROM sessions ORDER BY started DESC LIMIT ?", (limit,))

    def tail(self, n: int = 50) -> list[dict]:
        rows = self.query(
            "SELECT ts, 'event' AS kind, type || ' ' || state_from || '->' || state_to || ' ' || detail AS text "
            "FROM events UNION ALL SELECT ts, 'fault', kind || ' ' || detail FROM faults "
            "UNION ALL SELECT ts, 'voice', text || ' => ' || IFNULL(command,'-') FROM voice "
            "ORDER BY ts DESC LIMIT ?", (n,))
        return rows
