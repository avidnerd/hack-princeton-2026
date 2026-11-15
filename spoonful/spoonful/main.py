"""python -m spoonful [run|calibrate|teach|replay|demo|test-arm|seed-sim] [--backend sim|xarm|raw] ...

Motion entry points print their plan and do nothing unless --live is given (R2: --dry-run is the default).
"""
from __future__ import annotations

import argparse
import sys as _sys
import threading
import time
import webbrowser

from . import config as C


def _system(args, **kw):
    from .system import Options, System
    cfg = C.load(args.config)
    C.load_env()
    opts = Options(backend=args.backend, sim_face=getattr(args, "sim_face", False),
                   camera=not getattr(args, "no_camera", False), voice=getattr(args, "voice", None),
                   esp32=getattr(args, "esp32", None), **kw)
    return cfg, System(cfg, opts)


def cmd_run(args, page: str = "/") -> None:
    import uvicorn
    import socket
    from .app.server import create_app
    c0 = C.load(args.config)
    probe = socket.socket()
    try:                                     # a second copy would fight the first one for the arm: refuse early
        probe.bind(("0.0.0.0" if c0.app.phone_link else c0.app.host, c0.app.port))
    except OSError:
        print(f"Spoonful is already running (http://{c0.app.host}:{c0.app.port}/ is in use). Use that window, or "
              f"close it (Ctrl+C) before starting another.")
        return
    finally:
        probe.close()
    cfg, sys = _system(args)
    sys.start()
    app = create_app(sys)
    url = f"http://{cfg.app.host}:{cfg.app.port}{page}"
    print(f"Spoonful on {url}  (backend={sys.backend_kind}, sim_face={sys.face.sim_enabled})")
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host="0.0.0.0" if cfg.app.phone_link else cfg.app.host, port=cfg.app.port,
                    log_level="warning")
    finally:
        sys.stop()


def cmd_teach(args) -> None:
    from .arm.arm import Arm
    from .arm.backend import make_backend
    from .arm.teach import Recorder, library_for
    from .log import Log
    cfg = C.load(args.config)
    backend = make_backend(args.backend or cfg.arm.backend, [cfg.arm.servo_map[j] for j in C.JOINTS])
    log = Log(cfg.path(cfg.log.db))
    arm = Arm(cfg, backend, log)
    arm.start()
    lib = library_for(cfg, backend.name)
    rec = Recorder(cfg, arm, lib.dir, args.profile)
    kind = "pose" if args.pose else "path"
    input(f"Torque goes OFF on the arm joints when you press Enter. Support the arm. Recording {kind} "
          f"'{args.name}'. Enter to start... ")
    rec.start(args.name, kind)
    print("Recording. Move the arm through the motion. Press Enter when DONE.")
    stop = threading.Event()

    def show():
        while not stop.is_set():
            st = arm.read()
            print("\r  " + "  ".join(f"{j[:5]}:{st.deg.get(j, 0):6.1f}" for j in C.ARM_JOINTS)
                  + f"  n={len(rec.samples)}", end="", flush=True)
            time.sleep(0.1)
    threading.Thread(target=show, daemon=True).start()
    input()
    stop.set()
    try:
        tf = rec.stop(hold=not args.no_hold)
        print(f"\nsaved {tf.name}: {len(tf.waypoints)} waypoints ({len(tf.raw)} raw samples) in {lib.dir}")
    except Exception as e:
        print(f"\nNOT saved: {e}")
    finally:
        time.sleep(0.2)
        arm.close()
        log.close()


def cmd_replay(args) -> None:
    from .arm.arm import Arm
    from .arm.backend import make_backend
    from .arm.teach import library_for
    from .arm.trajectory import plan
    from .log import Log
    cfg = C.load(args.config)
    backend = make_backend(args.backend or cfg.arm.backend, [cfg.arm.servo_map[j] for j in C.JOINTS])
    lib = library_for(cfg, backend.name)
    if args.name not in lib.items:
        _sys.exit(f"no trajectory {args.name!r} in {lib.dir}. Have: {sorted(lib.items)}; errors: {lib.errors}")
    log = Log(cfg.path(cfg.log.db))
    arm = Arm(cfg, backend, log)
    arm.start()
    tf = lib.items[args.name]
    p = plan(cfg, arm.current(), [(tf.poses, tf.speed, tf.name)], args.speed, tf.name)
    print(p.describe(hz=args.hz))
    if not args.live:
        print("\nDRY RUN (default). Nothing was sent. Re-run with --live to move.")
    else:
        if input("Arm clear, hand on the power switch? type 'go': ").strip() != "go":
            print("aborted")
        else:
            m = arm.run(p).wait(p.duration + 5)
            print("done" if m.ok else f"motion ended: {m.error!r}")
    arm.close()
    log.close()


def cmd_test_arm(args) -> None:
    from .arm.backend import make_backend
    cfg = C.load(args.config)
    b = make_backend(args.backend or cfg.arm.backend, [cfg.arm.servo_map[j] for j in C.JOINTS])
    b.connect()
    print(f"backend {b.name}  battery {b.voltage()} V")
    for _ in range(int(args.seconds * 10)):
        r = b.read_all([cfg.arm.servo_map[j] for j in C.JOINTS])
        print("  ".join(f"{j[:5]}({cfg.arm.servo_map[j]}):{r[cfg.arm.servo_map[j]]}" for j in C.JOINTS))
        time.sleep(0.1)
    b.close()


def cmd_seed_sim(args) -> None:
    from .sim_seed import seed
    cfg = C.load(args.config)
    for line in seed(cfg):
        print(line)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="spoonful")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=["sim", "xarm", "raw"], default=None)
    sub = ap.add_subparsers(dest="cmd")

    def run_flags(p):
        p.add_argument("--sim-face", action="store_true", help="fake the diner's face from the operator drawer")
        p.add_argument("--no-camera", action="store_true")
        p.add_argument("--voice", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--esp32", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--open", action="store_true", help="open the browser")

    run_flags(sub.add_parser("run", help="full system + web app (default)"))
    run_flags(sub.add_parser("calibrate", help="web app with the operator drawer open on calibration"))
    run_flags(sub.add_parser("demo", help="preflight, then the full system"))
    t = sub.add_parser("teach", help="record a path or pose by hand")
    t.add_argument("action", choices=["record"])
    t.add_argument("name")
    t.add_argument("--pose", action="store_true")
    t.add_argument("--profile", default="demo")
    t.add_argument("--no-hold", action="store_true", help="leave torque off after recording")
    r = sub.add_parser("replay", help="replay a taught path (dry run unless --live)")
    r.add_argument("name")
    r.add_argument("--speed", type=float, default=0.8)
    r.add_argument("--dry-run", action="store_true", default=True)
    r.add_argument("--live", action="store_true")
    r.add_argument("--hz", type=float, default=5)
    ta = sub.add_parser("test-arm", help="print battery voltage and positions (no motion)")
    ta.add_argument("--seconds", type=float, default=3)
    sub.add_parser("seed-sim", help="write placeholder trajectories for the sim backend")
    args = ap.parse_args(argv)
    cmd = args.cmd or "run"
    if args.cmd is None:
        args.sim_face = args.no_camera = args.open = False
        args.voice = args.esp32 = None
    if cmd == "run":
        cmd_run(args)
    elif cmd == "calibrate":
        cmd_run(args, "/#operator=calibrate")
    elif cmd == "demo":
        from .preflight import run_checks
        cfg = C.load(args.config)
        C.load_env()
        ok, lines = run_checks(cfg, args.backend or cfg.arm.backend, camera=not args.no_camera)
        print("\n".join(lines))
        if not ok:
            _sys.exit("preflight RED: fix the items above before the demo")
        cmd_run(args)
    elif cmd == "teach":
        cmd_teach(args)
    elif cmd == "replay":
        cmd_replay(args)
    elif cmd == "test-arm":
        cmd_test_arm(args)
    elif cmd == "seed-sim":
        cmd_seed_sim(args)


if __name__ == "__main__":
    main()
