"""Phase 0 scratch: xArm 1S bring-up. NOT product code; never copy into the repo.

Usage (from hack-princeton/spoonful):
  .venv/bin/python ../scratch/bringup_arm.py probe          # voltage + one position read per ID
  .venv/bin/python ../scratch/bringup_arm.py watch          # torque OFF, 10 Hz positions, min/max per ID; Ctrl-C ends
  .venv/bin/python ../scratch/bringup_arm.py jog            # DRY RUN: prints the +/-10 deg moves
  .venv/bin/python ../scratch/bringup_arm.py jog --live     # asks y/N before EACH joint, then moves it +/-10 deg and back
  add --raw to any command to use the raw HID protocol instead of the xarm package
"""
import argparse
import statistics
import sys
import time

IDS = [1, 2, 3, 4, 5, 6]
VID, PID = 0x0483, 0x5750
JOG_UNITS = 42  # ~10 deg at 0.24 deg/unit (240 deg servo); the xarm pkg itself assumes 250 deg (0.25/unit)


class XarmPkg:
    def __init__(self):
        import xarm
        self.xarm = xarm
        self.c = xarm.Controller("USB")

    def voltage(self):
        return self.c.getBatteryVoltage()

    def pos(self, sid):
        # xarm 0.0.4: getPosition(int) works; a list must be of xarm.Servo objects, not ints.
        # _recv() can raise IndexError when the 50 ms HID read returns nothing: retry.
        for _ in range(3):
            try:
                return self.c.getPosition(sid)
            except Exception:
                time.sleep(0.01)
        return None

    def off(self):
        self.c.servoOff()

    def move(self, pairs, ms):
        self.c.setPosition([(i, int(p)) for i, p in pairs], duration=ms, wait=False)


class RawHid:
    """Hiwonder HID: report [0, 0x55, 0x55, len(payload)+2, cmd, *payload]."""
    MOVE, VOLT, UNLOAD, POSREAD = 3, 15, 20, 21

    def __init__(self):
        import hid
        self.d = hid.device()
        self.d.open(VID, PID)
        self.d.set_nonblocking(0)

    def _send(self, cmd, payload=()):
        self.d.write([0, 0x55, 0x55, len(payload) + 2, cmd, *payload])

    def _recv(self, cmd, timeout_ms=100):
        r = self.d.read(64, timeout_ms)
        if len(r) >= 4 and r[0] == 0x55 and r[1] == 0x55 and r[3] == cmd:
            return r[4:4 + r[2]]
        return None

    def voltage(self):
        self._send(self.VOLT)
        r = self._recv(self.VOLT)
        return None if not r else (r[1] * 256 + r[0]) / 1000.0

    def pos(self, sid):
        self._send(self.POSREAD, [1, sid])
        r = self._recv(self.POSREAD)
        return None if not r else r[3] * 256 + r[2]

    def off(self):
        self._send(self.UNLOAD, [6, *IDS])

    def move(self, pairs, ms):
        payload = [len(pairs), ms & 0xFF, ms >> 8]
        for i, p in pairs:
            p = int(p)
            payload += [i, p & 0xFF, p >> 8]
        self._send(self.MOVE, payload)


def median_pos(arm, sid, n=5):
    vals = [v for v in (arm.pos(sid) for _ in range(n)) if v is not None]
    return int(statistics.median(vals)) if vals else None


def cmd_probe(arm):
    print(f"battery: {arm.voltage()} V")
    for sid in IDS:
        print(f"id {sid}: {median_pos(arm, sid)}")


def cmd_watch(arm):
    arm.off()
    print("torque OFF. Move each joint by hand, one at a time, end stop to end stop. Ctrl-C to finish.")
    lo, hi = {i: 1001 for i in IDS}, {i: -1 for i in IDS}
    try:
        while True:
            t0 = time.time()
            row = {}
            for sid in IDS:
                p = arm.pos(sid)
                row[sid] = p
                if p is not None:
                    lo[sid], hi[sid] = min(lo[sid], p), max(hi[sid], p)
            print("  ".join(f"{s}:{row[s] if row[s] is not None else '---':>4}" for s in IDS), flush=True)
            time.sleep(max(0, 0.1 - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    print("\nraw range seen (units)  -> paste into NOTES.md")
    for sid in IDS:
        print(f"  id {sid}: {lo[sid]} .. {hi[sid]}")


def cmd_jog(arm, live):
    print(f"battery: {arm.voltage()} V")
    start = {sid: median_pos(arm, sid) for sid in IDS}
    print("start pose:", start)
    for sid in [6, 5, 4, 3, 2]:  # gripper (1) skipped: it may be closed on the bite block
        p0 = start[sid]
        if p0 is None:
            print(f"id {sid}: no read, skipped")
            continue
        seq = [min(1000, p0 + JOG_UNITS), max(0, p0 - JOG_UNITS), p0]
        print(f"id {sid}: {p0} -> {seq[0]} -> {seq[1]} -> {seq[2]}  (1.5 s each)")
        if not live:
            continue
        if input(f"  move id {sid}? arm clear, hand on the power switch [y/N] ").strip().lower() != "y":
            continue
        for tgt in seq:
            arm.move([(sid, tgt)], 1500)
            time.sleep(1.7)
            print(f"    cmd {tgt}  actual {median_pos(arm, sid)}")
    if not live:
        print("DRY RUN. Nothing sent. Re-run with --live.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "watch", "jog"])
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    try:
        arm = RawHid() if a.raw else XarmPkg()
    except Exception as e:
        sys.exit(f"could not open the arm (VID 0483 PID 5750): {e!r}\n"
                 "Is USB plugged in and the 7.5 V supply switched on? Try --raw.")
    {"probe": lambda: cmd_probe(arm), "watch": lambda: cmd_watch(arm),
     "jog": lambda: cmd_jog(arm, a.live)}[a.cmd]()


if __name__ == "__main__":
    main()
