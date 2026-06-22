# state.py
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class BeliefState:
    distinct_windows: int = 0
    windows: list[Tuple[float, float]] = field(default_factory=list)

    # NEW: microevent bookkeeping
    anchor_window: Optional[Tuple[float, float]] = None
    last_window: Optional[Tuple[float, float]] = None

    best_relevance_score: float = 0.0
    prev_best_score: float = 0.0
    score_improvement: float = 0.0

    dense_seconds_encoded: float = 0.0
    approx_frames_encoded: int = 0
    inspect_wallclock_s: float = 0.0
    steps: int = 0
    probe_wallclock_s: float = 0.0

    # --- NEW: probe metrics ---
    probe_wallclock_s: float = 0.0
    probe_segments_scanned: int = 0
    probe_fps: float = 0.0


    def update_from_window(self, result):
        self.steps += 1
        self.prev_best_score = self.best_relevance_score
        self.best_relevance_score = max(self.best_relevance_score, result.relevance_score)
        self.score_improvement = self.best_relevance_score - self.prev_best_score

        self.dense_seconds_encoded += result.dense_seconds
        self.approx_frames_encoded += result.frames_encoded
        self.inspect_wallclock_s += result.wallclock_s

        window = (result.t0, result.t1)

        # NEW: always update last_window
        self.last_window = window

        # NEW: set anchor_window once (first inspected window)
        if self.anchor_window is None:
            self.anchor_window = window

        if window not in self.windows:
            self.windows.append(window)
            self.distinct_windows += 1

    def update_from_probe(self, wallclock_s: float, segments_scanned: int, fps: float):
        self.probe_wallclock_s += wallclock_s
        self.probe_segments_scanned += segments_scanned
        self.probe_fps = fps