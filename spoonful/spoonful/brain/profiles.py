"""Per-person profiles, one JSON file each. Demo set: demo, judge_a, judge_b."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Profile:
    name: str
    approach_speed: float = 1.0        # x near_face speed cap (<= 1)
    hold_s: float = 2.5
    bite_size: str = "full"            # small | full -> scoop_small / scoop_full path, falls back to scoop
    preferred_food: str = ""
    mouth_offset_mm: list = field(default_factory=lambda: [0.0, 0.0])
    pace_scale: float = 1.0            # "slower" x1.25, "faster" x0.8
    learned_interval_s: float = 0.0    # written by the pace model
    speech: bool = True
    auto_pace: bool = False

    @property
    def scoop_path(self) -> str:
        return f"scoop_{self.bite_size}"


DEFAULTS = {"demo": {}, "judge_a": {"hold_s": 3.0}, "judge_b": {"approach_speed": 0.8, "bite_size": "small"}}


class ProfileStore:
    def __init__(self, directory: Path, default: str = "demo"):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        for name, over in DEFAULTS.items():
            if not (self.dir / f"{name}.json").exists():
                self.save(Profile(name, **over))
        self.current = self.load(default)

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def load(self, name: str) -> Profile:
        p = self.dir / f"{name}.json"
        if not p.exists():
            prof = Profile(name)
            self.save(prof)
            return prof
        d = json.loads(p.read_text())
        known = {f.name for f in fields(Profile)}
        return Profile(**{k: v for k, v in d.items() if k in known})

    def save(self, prof: Profile) -> None:
        with self._lock:
            (self.dir / f"{prof.name}.json").write_text(json.dumps(asdict(prof), indent=1))

    def select(self, name: str) -> Profile:
        self.current = self.load(name)
        return self.current

    def update(self, **kw) -> Profile:
        for k, v in kw.items():
            if hasattr(self.current, k):
                setattr(self.current, k, v)
        self.save(self.current)
        return self.current
