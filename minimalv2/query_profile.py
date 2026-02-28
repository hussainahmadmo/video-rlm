from dataclasses import dataclass
from typing import Literal


#mode controls how many windows are selected and how man
Mode = Literal["attribute", "microevent", "causal", "distributed", "ordering"]
Anchor = Literal["best_score", "earliest", "latest", "around_event"]
temporal_markers = ["after", "afterwards", "then", "next", "following", "later", "before", "previously"]

@dataclass(frozen=True)
class QueryProfile:
    #type of reasoning required
    mode: Mode

    anchor_policy: Anchor
    coverage_target: int = 2
    require_temporal_pair: bool = False
    eps_marginal_gain: float = 0.01
    max_steps: int = 20


def classify_query(query: str) -> QueryProfile:
    q = query.lower()

    # #the first inspected window should be the earliest segment
    if "first" in q or "earliest" in q:
        return QueryProfile(mode="ordering", anchor_policy="earliest", coverage_target=2)
    #the 
    if "last" in q or "latest" in q:
        return QueryProfile(mode="ordering", anchor_policy="latest", coverage_target=2)

    # NEW: microevent / temporal-followup questions
    # dont change the if else ordering of the branches, dont want generic words in the 
    # below conditions to end up stealing temporal markers.

    if "why" in q or "cause" in q or "because" in q:
        return QueryProfile(mode="causal", anchor_policy="best_score", coverage_target=2)

    if "how many" in q or "throughout" in q or "all" in q:
        return QueryProfile(mode="distributed", anchor_policy="best_score", coverage_target=3)

    return QueryProfile(mode="attribute", anchor_policy="best_score", coverage_target=1)