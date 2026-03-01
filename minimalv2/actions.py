# actions.py
from dataclasses import dataclass
from typing import Literal, Iterable, Tuple, Optional

@dataclass(frozen=True)
class Action:
    kind: Literal["inspect_window"]
    t0: float
    t1: float
    stride: float
    resolution: str


def build_action_space(
    candidates,
    window_len: float = 4.0,
    topk: int = 50,
    *,
    strides: Optional[Iterable[float]] = None,
    resolutions: Optional[Iterable[str]] = None,
) -> list[Action]:
    """
    Build inspect actions around each candidate.
    You can pass multiple strides to get multi-scale coverage.
    """
    candidates = sorted(candidates, key=lambda c: (-c.score, c.t0))[:topk]
    strides = list(strides) if strides is not None else [0.5]
    resolutions = list(resolutions) if resolutions is not None else ["high"]

    actions: list[Action] = []
    for c in candidates:
        mid = 0.5 * (c.t0 + c.t1)
        base_t0 = max(0.0, mid - window_len / 2)
        base_t1 = base_t0 + window_len

        for s in strides:
            for r in resolutions:
                actions.append(Action("inspect_window", base_t0, base_t1, s, r))

    # de-dupe exact duplicates while preserving order
    seen = set()
    uniq: list[Action] = []
    for a in actions:
        key = (a.t0, a.t1, a.stride, a.resolution)
        if key not in seen:
            uniq.append(a)
            seen.add(key)
    return uniq