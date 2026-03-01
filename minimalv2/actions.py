# actions.py

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Action:
    kind: Literal["inspect_window"]
    t0: float
    t1: float
    stride: float
    resolution: str


def build_action_space(candidates, window_len=4.0, topk=50):
    candidates = sorted(candidates, key=lambda c: (-c.score, c.t0))[:topk]

    actions = []
    for c in candidates:
        mid = 0.5 * (c.t0 + c.t1)
        t0 = max(0.0, mid - window_len / 2)
        t1 = t0 + window_len
        actions.append(Action("inspect_window", t0, t1, 0.5, "high"))

    return actions