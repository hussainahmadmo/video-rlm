from dataclasses import dataclass
import torch

@dataclass
class Segment:
    seg_idx: int
    t0: float
    t1: float
    clip_emb: torch.Tensor
    motion_mean: float

@dataclass
class SearchResult:
    seg_idx: int
    t0: float
    t1: float
    score: float

@dataclass
class RefineResult:
    t0: float
    t1: float
    score: float
    seg_idx: int
