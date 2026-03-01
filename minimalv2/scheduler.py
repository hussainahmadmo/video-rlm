# scheduler.py

def pick_next_action(profile, state, actions):
    inspected = state.windows

    def is_new(a):
        return (a.t0, a.t1) not in inspected

    candidates = [a for a in actions if is_new(a)]
    if not candidates:
        return None

    if profile.mode == "ordering":
        if profile.anchor_policy == "earliest":
            return min(candidates, key=lambda a: a.t0)
        if profile.anchor_policy == "latest":
            return max(candidates, key=lambda a: a.t0)

    if profile.mode == "distributed":
        return candidates[0]  # already sorted by probe score

    if profile.mode == "causal":
        return candidates[0]

    return candidates[0]


def propose_followup_window(anchor: tuple[float,float], *, direction: str, gap_s: float = 0.0, width_s: float = 4.0):
    """
    anchor = event
    width_s is the duration of the succ/pred segment evaluated for transition based reasoning
    direction - 

    """
    t0, t1 = anchor
    if direction == "after":
        return (t1 + gap_s, t1 + gap_s + width_s)
    if direction == "before":
        return (max(0.0, t0 - gap_s - width_s), max(0.0, t0 - gap_s))
    raise ValueError(direction)


from typing import List, Tuple

def propose_followup_windows(
    anchor: Tuple[float, float],
    *,
    direction: str,
    width_s: float,
    gaps_s: List[float],   # multiple offsets from the anchor boundary
) -> List[Tuple[float, float]]:
    """
    Returns multiple follow-up windows at increasing temporal offsets.

    Example (direction="after"):
        gaps_s=[0, 4, 12] with width=4 and anchor=(t0,t1) -> [(t1, t1+4), (t1+4, t1+8), (t1+12, t1+16)]
        anchor = event 
        width_s is the duration of the succ/pred segment evaluated for transition based reasoning
        direction - can be either before or after

    """
    t0, t1 = anchor
    out: List[Tuple[float, float]] = []

    for g in gaps_s:
        if direction == "after":
            a = t1 + g
            b = a + width_s
        elif direction == "before":
            b = max(0.0, t0 - g)
            a = max(0.0, b - width_s)
        else:
            raise ValueError(direction)

        if b > a:
            out.append((a, b))

    # de-dupe while preserving order
    seen = set()
    uniq = []
    for w in out:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq