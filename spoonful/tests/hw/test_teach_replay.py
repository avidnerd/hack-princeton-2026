"""MANUAL: Phase 1 acceptance helper. 10 consecutive scoop -> lift -> pre_delivery -> retreat cycles on real food.
    SPOONFUL_HW=1 .venv/bin/python -m pytest -m hw tests/hw/test_teach_replay.py -s
"""
import os
import time

import pytest

from spoonful.arm.arm import Arm
from spoonful.arm.backend import make_backend
from spoonful.arm.teach import library_for
from spoonful.arm.trajectory import plan
from spoonful.config import JOINTS
from spoonful.log import Log

pytestmark = [pytest.mark.hw, pytest.mark.skipif(not os.environ.get("SPOONFUL_HW"), reason="set SPOONFUL_HW=1")]


def test_ten_cycles(cfg):
    kind = cfg.arm.backend if cfg.arm.backend != "sim" else "xarm"
    b = make_backend(kind, [cfg.arm.servo_map[j] for j in JOINTS])
    log = Log(cfg.path(cfg.log.db))
    arm = Arm(cfg, b, log)
    arm.start()
    lib = library_for(cfg, b.name)
    for n in ("scoop", "lift", "pre_delivery", "retreat", "rest"):
        assert lib.has(n), f"teach {n} first"
    p = plan(cfg, arm.current(), [(lib.path("scoop"), "normal", "scoop"), (lib.path("lift"), "normal", "lift"),
                                  ([lib.pose("pre_delivery")], "normal", "pre"),
                                  (lib.path("retreat"), "normal", "retreat"), ([lib.pose("rest")], "normal", "rest")],
             0.8, "cycle")
    print(p.describe(hz=2))
    assert input("Reviewed the plan above? type 'go': ").strip() == "go"
    times = []
    for i in range(10):
        t0 = time.time()
        m = arm.run(plan(cfg, arm.current(), [(s.path, s.speed, s.label) for s in p.segments], 0.8, "cycle")).wait(120)
        assert m.ok, f"cycle {i + 1}: {m.error!r}"
        times.append(time.time() - t0)
        ok = input(f"cycle {i + 1}: {times[-1]:.1f} s. Food on the spoon at pre_delivery? [y/n] ").strip()
        assert ok == "y"
        log.event("phase1_cycle", detail_n=i + 1, seconds=round(times[-1], 1))
    print("cycle times:", [round(t, 1) for t in times])
    arm.close()
    log.close()
