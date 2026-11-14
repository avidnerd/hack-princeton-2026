#!/usr/bin/env python
"""Print the pitch numbers from the log: bites, people, contacts, holds, aborts, mean bite time, p95 latencies."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spoonful import config as C  # noqa: E402
from spoonful.log import Log  # noqa: E402
from spoonful.stats import compute  # noqa: E402

cfg = C.load()
db = sys.argv[1] if len(sys.argv) > 1 else cfg.path(cfg.log.db)
log = Log(db)
s = compute(log)
log.close()
if "--json" in sys.argv:
    print(json.dumps(s, indent=1))
else:
    print(f"""
  bites delivered      {s['bites']}  ({s['taken']} taken)
  people fed           {s['people']}   sessions {s['sessions']}
  contacts             {s['contacts']}
  holds / aborts       {s['holds']} / {s['aborts']}
  mean bite time       {s['mean_bite_s']} s
  p95 frame age        {s['p95_frame_age_ms']} ms
  p95 gate->back-off   {s['p95_gate_close_ms']} ms
  outcomes             {s['outcomes']}
  faults               {s['faults']}
""")
