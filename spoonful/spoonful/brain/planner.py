"""Open-ended requests: a vision-language model plans, verified skills execute.

"Spoonful, I want something sweet" / "a little of each" / "which one has more protein?" / "finish the one that's
almost empty". None of these are programmed. The planner gives Gemini the request plus what the robot currently
perceives (each bowl's food, amount left, texture) and a short list of skills, and gets back a plan as JSON. The plan
is then VERIFIED before anything happens:

  - only known skills (say, select, queue, decline), only bowls that exist, at most MAX_QUEUE bites
  - a bowl that is empty or whose scoop isn't taught is dropped (and the robot says so)
  - no skill moves the arm by itself: "select"/"queue" only decide WHICH bowl the next GO press scoops from, and
    every bite still goes through the normal FSM (GO, face gate, contact sensing, face zone)

So the language model decides intent; it never gets direct control of motion.
"""
from __future__ import annotations

import json
import threading
from typing import Callable, Optional

SKILLS = ("say", "select", "queue", "decline")
MAX_QUEUE = 5

PROMPT = """You are the planner of Spoonful, a robot arm that feeds a person with a spoon.
The person said: "{request}"

What the robot sees on the table right now (from its camera):
{scene}

You can only use these skills. Answer only JSON: {{"plan": [ ...steps... ]}}
  {{"skill": "say", "text": "..."}}            say something short and friendly (always include one step like this)
  {{"skill": "select", "bowl": "<bowl id>"}}   the next bite comes from this bowl (the person presses a button to get it)
  {{"skill": "queue", "bowls": ["<id>", ...]}} several bites in order, one per button press (at most {max_queue})
  {{"skill": "decline", "reason": "..."}}      if the request is impossible, unsafe, or not about eating from these bowls
Rules: use only bowl ids listed above. Never invent food that isn't listed. If a bowl is empty, don't pick it. You may
answer questions about the food (taste, nutrition, which is which) with "say" alone. Keep "say" under 20 words. If
you select or queue food, end the "say" text with "Press the button when you're ready."
"""


def scene_text(bowls: list[dict]) -> str:
    lines = []
    for b in bowls:
        food = b.get("food") or "unknown food"
        extra = []
        if b.get("amount"):
            extra.append(f"amount left: {b['amount']}")
        if b.get("texture"):
            extra.append(f"texture: {b['texture']}")
        if b.get("empty"):
            extra.append("EMPTY")
        if not b.get("taught", True):
            extra.append("robot can't scoop this one yet")
        lines.append(f'- {b["name"]}: {food}' + (f' ({", ".join(extra)})' if extra else ""))
    return "\n".join(lines) or "- (no bowls seen)"


def verify(plan: object, bowls: list[dict]) -> tuple[list[dict], list[str]]:
    """Pure. Keep only well-formed steps on bowls that exist, aren't empty and can be scooped.
    Returns (steps, problems) where problems are short notes for the log."""
    ok_bowls = {b["name"] for b in bowls if not b.get("empty") and b.get("taught", True)}
    known = {b["name"] for b in bowls}
    steps, problems = [], []
    raw = plan.get("plan") if isinstance(plan, dict) else plan
    if not isinstance(raw, list):
        return [], ["plan is not a list"]
    for st in raw[:8]:
        if not isinstance(st, dict) or st.get("skill") not in SKILLS:
            problems.append(f"unknown step {st!r}"[:80])
            continue
        k = st["skill"]
        if k == "say":
            t = str(st.get("text", "")).strip()
            if t:
                steps.append({"skill": "say", "text": t[:160]})
        elif k == "decline":
            steps.append({"skill": "decline", "reason": str(st.get("reason", ""))[:160]})
        elif k == "select":
            b = st.get("bowl")
            if b in ok_bowls:
                steps.append({"skill": "select", "bowl": b})
            else:
                problems.append(f"select {b!r}: " + ("not available" if b in known else "no such bowl"))
        elif k == "queue":
            bs = [b for b in (st.get("bowls") or []) if b in ok_bowls][:MAX_QUEUE]
            dropped = [b for b in (st.get("bowls") or []) if b not in ok_bowls]
            if dropped:
                problems.append(f"queue dropped {dropped}")
            if bs:
                steps.append({"skill": "queue", "bowls": bs})
    return steps, problems


class Planner:
    def __init__(self, generate: Optional[Callable[[str], object]], log=None):
        """generate(prompt) -> parsed JSON (dict/list). None = no model available."""
        self.generate = generate
        self.log = log
        self.busy = False
        self.last: dict = {}

    @property
    def available(self) -> bool:
        return self.generate is not None

    def plan(self, request: str, bowls: list[dict]) -> tuple[list[dict], list[str]]:
        prompt = PROMPT.format(request=request.strip()[:300], scene=scene_text(bowls), max_queue=MAX_QUEUE)
        raw = self.generate(prompt)
        steps, problems = verify(raw, bowls)
        self.last = {"request": request, "raw": raw, "steps": steps, "problems": problems}
        if self.log:
            self.log.event("plan", detail_request=request[:200], steps=steps, problems=problems)
        return steps, problems

    def plan_async(self, request: str, bowls: list[dict], done: Callable[[list[dict], list[str]], None],
                   failed: Callable[[Exception], None]) -> bool:
        if self.busy or not self.available:
            return False
        self.busy = True

        def run():
            try:
                steps, problems = self.plan(request, bowls)
                done(steps, problems)
            except Exception as e:
                failed(e)
            finally:
                self.busy = False
        threading.Thread(target=run, name="planner", daemon=True).start()
        return True
