#!/usr/bin/env python3
"""Measure end-to-end priority across external video preparation and vLLM.

Background videos arrive first and urgent videos arrive later. The preparation
dispatcher can be FCFS, priority ordered, shortest-job-first, statically
reserved, adaptive to request SLO slack, max-min fair across tenants, hierarchical
tenant-fair priority, fair across tenants and slowdown-aware within a tenant,
or max-min fair with age-aware tail protection.
Prepared requests retain the same priority when submitted to one or more vLLM
replicas using ``--scheduling-policy priority``. The ``sjf``, ``max_min``, ``tenant_fair``,
``tenant_priority``, ``fair_slowdown``, ``age_aware_max_min``, and
``engine_tenant_fair`` policies
instead submit a uniform engine priority because their external dispatcher
owns the relevant admission ordering. ``engine_tenant_fair`` is a VTC-style
engine-only baseline: preparation remains FCFS and tenant fairness starts only
after a request becomes model-ready.

Unlike submitting every preparation task to ThreadPoolExecutor immediately,
this runner keeps an explicit pending heap and admits at most ``--prep-workers``
tasks.  A newly arrived urgent request can therefore overtake background work
that has not begun preparation.  Running decode work is intentionally
non-preemptive.
"""

from __future__ import annotations

import argparse
import base64
import heapq
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[4]
CODEC_PATH = (
    ROOT
    / "conductor/experiments/scripts/run/run_codec_guided_vllm_baseline.py"
)
WRITE_LOCK = threading.Lock()
CROSS_STAGE_FAIR_POLICIES = frozenset({
    "max_min", "tenant_fair", "tenant_priority", "fair_slowdown",
    "cross_stage", "age_aware_max_min",
})
PREPARATION_FAIR_POLICIES = frozenset({
    *CROSS_STAGE_FAIR_POLICIES, "prep_max_min",
})
INFERENCE_FAIR_POLICIES = frozenset({
    *CROSS_STAGE_FAIR_POLICIES, "engine_tenant_fair",
})
SCHEDULER_OWNED_POLICIES = frozenset({
    "sjf", *PREPARATION_FAIR_POLICIES, *INFERENCE_FAIR_POLICIES,
    "tenant_round_robin",
})


class OnlineStageCostProfiler:
    """Online EWMA cost profiler with progressively coarser fallbacks.

    Predictions use only metadata available before a stage is admitted.  An
    observation updates every compatible projection, so requests without
    duration, resolution, or codec metadata can still use a frame/backend or
    global estimate learned from earlier requests.
    """

    _DURATION_EDGES_S = (30.0, 120.0, 600.0, 1800.0, 3600.0)
    _PIXEL_EDGES = (640 * 480, 1280 * 720, 1920 * 1080, 3840 * 2160)

    def __init__(
        self,
        *,
        stage: str,
        mode: str,
        backend: str,
        alpha: float,
        min_samples: int = 1,
    ) -> None:
        if stage not in {"prep", "engine"}:
            raise ValueError("stage must be prep or engine")
        if mode not in {"frame_ewma", "metadata_ewma"}:
            raise ValueError("unsupported cost-profiler mode")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        self.stage = stage
        self.mode = mode
        self.backend = backend
        self.alpha = alpha
        self.min_samples = min_samples
        self._ewma: dict[tuple[Any, ...], float] = {}
        self._counts: dict[tuple[Any, ...], int] = defaultdict(int)
        self._prediction_errors_s: list[float] = []
        self._absolute_percentage_errors: list[float] = []

    @staticmethod
    def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
        for name in names:
            value = row.get(name)
            if value is not None and value != "":
                return value
        return None

    @classmethod
    def _bucket(cls, value: Any, edges: tuple[float, ...]) -> str | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        for edge in edges:
            if number <= edge:
                return f"le_{edge:g}"
        return f"gt_{edges[-1]:g}"

    @classmethod
    def _resolution_pixels(cls, row: dict[str, Any]) -> float | None:
        width = cls._first(row, ("video_width", "width"))
        height = cls._first(row, ("video_height", "height"))
        if width is not None and height is not None:
            try:
                return float(width) * float(height)
            except (TypeError, ValueError):
                return None
        resolution = cls._first(row, ("video_resolution", "resolution"))
        if isinstance(resolution, str):
            match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", resolution)
            if match:
                return float(int(match.group(1)) * int(match.group(2)))
        return None

    def _keys(
        self,
        job: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[Any, ...]]:
        row = job.get("row") or {}
        metadata = metadata or job.get("profile_metadata") or {}
        frames = int(job["frame_count"])
        output_budget = int(job.get("max_tokens", 0))
        if self.mode == "frame_ewma":
            if self.stage == "engine":
                return [
                    ("frames_output", frames, output_budget),
                    ("frames", frames),
                ]
            return [("frames", frames)]

        duration = self._first(
            metadata, ("duration_s", "video_duration_s", "duration")
        )
        if duration is None:
            duration = self._first(
                row, ("duration_s", "video_duration_s", "duration")
            )
        codec = self._first(metadata, ("codec", "video_codec", "codec_name"))
        if codec is None:
            codec = self._first(row, ("codec", "video_codec", "codec_name"))
        pixels = self._resolution_pixels(metadata)
        if pixels is None:
            pixels = self._resolution_pixels(row)

        duration_bucket = self._bucket(duration, self._DURATION_EDGES_S)
        resolution_bucket = self._bucket(pixels, self._PIXEL_EDGES)
        detailed = (
            "metadata", self.backend, frames, duration_bucket,
            resolution_bucket, str(codec).lower() if codec is not None else None,
            output_budget if self.stage == "engine" else None,
        )
        # Ordered from most to least specific. Duplicate projections are
        # removed while retaining their fallback order.
        candidates = [
            detailed,
            *(
                [("backend_frames_output", self.backend, frames, output_budget)]
                if self.stage == "engine"
                else []
            ),
            ("backend_frames", self.backend, frames),
            ("frames", frames),
            ("global",),
        ]
        return list(dict.fromkeys(candidates))

    def predict(
        self,
        job: dict[str, Any],
        *,
        default_s: float,
    ) -> float:
        if default_s <= 0:
            raise ValueError("default stage cost must be positive")
        for key in self._keys(job):
            if self._counts[key] >= self.min_samples:
                return self._ewma[key]
        return float(default_s)

    def observe(
        self,
        job: dict[str, Any],
        *,
        observed_s: float,
        predicted_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if observed_s < 0:
            raise ValueError("observed stage cost cannot be negative")
        if predicted_s is not None:
            self._prediction_errors_s.append(predicted_s - observed_s)
            if observed_s > 0:
                self._absolute_percentage_errors.append(
                    abs(predicted_s - observed_s) / observed_s
                )
        for key in self._keys(job, metadata):
            previous = self._ewma.get(key, observed_s)
            self._ewma[key] = (
                self.alpha * observed_s + (1.0 - self.alpha) * previous
            )
            self._counts[key] += 1

    def frame_estimates(self) -> dict[int, float]:
        return {
            int(key[1]): value
            for key, value in self._ewma.items()
            if len(key) == 2 and key[0] == "frames"
        }

    def snapshot(self) -> dict[str, Any]:
        absolute_errors = [abs(value) for value in self._prediction_errors_s]
        return {
            "stage": self.stage,
            "mode": self.mode,
            "backend": self.backend,
            "alpha": self.alpha,
            "min_samples": self.min_samples,
            "observations": len(self._prediction_errors_s),
            "mean_absolute_error_s": mean(absolute_errors),
            "mean_absolute_percentage_error_percent": (
                None
                if not self._absolute_percentage_errors
                else 100.0 * mean(self._absolute_percentage_errors)
            ),
            "profiles": {
                json.dumps(key, separators=(",", ":")): {
                    "samples": self._counts[key],
                    "ewma_service_s": value,
                }
                for key, value in sorted(
                    self._ewma.items(), key=lambda item: repr(item[0])
                )
            },
        }


class BatchAwareTokenCostProfile:
    """Platform-specific inference cost fitted at several batch concurrencies.

    The input is the JSON emitted by ``analyze_vllm_token_service_profile.py``.
    A prediction contains one fixed request cost plus marginal text-prefill,
    visual-input, and decode costs. Coefficients are linearly interpolated for
    the number of requests currently sharing a replica and clamped outside the
    profiled concurrency range.
    """

    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text())
        raw_fits = payload.get("fits_by_concurrency") or {}
        if not raw_fits:
            raise ValueError(f"inference profile has no fitted costs: {path}")
        self.path = path
        self.fits: dict[int, dict[str, Any]] = {
            int(concurrency): fit for concurrency, fit in raw_fits.items()
        }
        if min(self.fits) < 1:
            raise ValueError("profile concurrency must be positive")
        self.concurrencies = sorted(self.fits)
        self.visual_tokens_per_frame = self._infer_visual_tokens_per_frame(
            payload.get("aggregated_points") or []
        )

    @staticmethod
    def _infer_visual_tokens_per_frame(points: list[dict[str, Any]]) -> float:
        samples: list[float] = []
        baselines: dict[int, float] = {
            int(point["concurrency"]): float(point["mean_prompt_tokens"])
            for point in points
            if point.get("family") == "visual" and int(point.get("target", -1)) == 0
        }
        for point in points:
            if point.get("family") != "visual":
                continue
            frames = int(point.get("target", 0))
            concurrency = int(point.get("concurrency", 0))
            if frames > 0 and concurrency in baselines:
                samples.append(
                    (float(point["mean_prompt_tokens"]) - baselines[concurrency])
                    / frames
                )
        positive = sorted(value for value in samples if value > 0)
        if not positive:
            return 1.0
        middle = len(positive) // 2
        if len(positive) % 2:
            return positive[middle]
        return 0.5 * (positive[middle - 1] + positive[middle])

    @staticmethod
    def _coefficient(fit: dict[str, Any], family: str, name: str) -> float:
        value = (fit.get(f"{family}_fit") or {}).get(name)
        if value is None:
            return 0.0
        return float(value)

    def _coefficients_at(self, concurrency: int) -> dict[str, float]:
        concurrency = max(1, int(concurrency))
        lower = max((c for c in self.concurrencies if c <= concurrency), default=None)
        upper = min((c for c in self.concurrencies if c >= concurrency), default=None)
        if lower is None:
            lower = upper
        if upper is None:
            upper = lower
        assert lower is not None and upper is not None

        def coefficients(profile: dict[str, Any]) -> dict[str, float]:
            intercepts = [
                self._coefficient(profile, family, "intercept")
                for family in ("prefill", "decode", "visual")
            ]
            return {
                "fixed": max(0.0, *intercepts),
                "text": max(0.0, self._coefficient(profile, "prefill", "slope")),
                "output": max(0.0, self._coefficient(profile, "decode", "slope")),
                "visual": max(0.0, self._coefficient(profile, "visual", "slope")),
            }

        low = coefficients(self.fits[lower])
        if lower == upper:
            return low
        high = coefficients(self.fits[upper])
        fraction = (concurrency - lower) / (upper - lower)
        return {
            name: low[name] + fraction * (high[name] - low[name])
            for name in low
        }

    @staticmethod
    def estimate_text_tokens(job: dict[str, Any]) -> int:
        row = job.get("row") or {}
        for name in ("estimated_text_tokens", "text_tokens", "input_tokens"):
            if row.get(name) is not None:
                return max(0, int(row[name]))
        parts = [
            str(row.get(name) or "")
            for name in ("prompt_override", "question")
        ]
        parts.extend(str(choice) for choice in (row.get("choices") or []))
        # Four characters per token is a deliberately simple cold-start
        # estimate. API-reported token counts replace it at reconciliation.
        return max(1, (len(" ".join(parts)) + 3) // 4)

    def predict(
        self,
        job: dict[str, Any],
        *,
        concurrency: int,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> float:
        coefficients = self._coefficients_at(concurrency)
        frames = 0 if job.get("modality") == "text" else int(job["frame_count"])
        visual_tokens = frames * self.visual_tokens_per_frame
        if prompt_tokens is None:
            text_tokens = self.estimate_text_tokens(job)
        else:
            # vLLM reports visual tokens as part of prompt_tokens. Separate
            # them so visual work is charged using its profiled coefficient.
            text_tokens = max(0.0, float(prompt_tokens) - visual_tokens)
        output_tokens = (
            int(job["max_tokens"])
            if completion_tokens is None
            else max(0, int(completion_tokens))
        )
        cost = (
            coefficients["fixed"]
            + coefficients["text"] * text_tokens
            + coefficients["visual"] * visual_tokens
            + coefficients["output"] * output_tokens
        )
        return max(cost, 1e-9)

    def snapshot(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "profiled_concurrencies": self.concurrencies,
            "visual_tokens_per_frame": self.visual_tokens_per_frame,
            "accounting": "fixed + text prefill + visual input + decode",
        }


def parse_cpu_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("CPU set is empty")
    return cpus


def parse_tenant_weights(values: list[str]) -> dict[str, float]:
    """Parse repeated TENANT=WEIGHT command-line values."""
    weights: dict[str, float] = {}
    for value in values:
        tenant, separator, weight_text = value.partition("=")
        tenant = tenant.strip()
        if not separator or not tenant:
            raise ValueError(
                f"invalid tenant weight {value!r}; expected TENANT=WEIGHT"
            )
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid tenant weight {value!r}; weight must be numeric"
            ) from exc
        if weight <= 0:
            raise ValueError(
                f"invalid tenant weight {value!r}; weight must be positive"
            )
        if tenant in weights:
            raise ValueError(f"duplicate tenant weight for {tenant!r}")
        weights[tenant] = weight
    return weights


def validate_max_min_tenant_weights(weights: dict[str, float]) -> None:
    """Keep the max_min policy distinct from weighted max-min fairness."""
    non_unit = {
        tenant: weight
        for tenant, weight in weights.items()
        if abs(weight - 1.0) > 1e-12
    }
    if non_unit:
        formatted = ", ".join(
            f"{tenant}={weight:g}" for tenant, weight in sorted(non_unit.items())
        )
        raise ValueError(
            "--prep-policy max_min requires unit tenant weights; "
            f"remove or set these weights to 1: {formatted}"
        )


def predicted_slowdown(
    *, elapsed_s: float, remaining_s: float, solo_s: float,
) -> float:
    """Return completion slowdown if a queued request is dispatched now."""
    if solo_s <= 0:
        raise ValueError("solo service estimate must be positive")
    return (max(0.0, elapsed_s) + max(0.0, remaining_s)) / solo_s


def fair_slowdown_choice(
    jobs: list[dict[str, Any]],
    *,
    now_s: float,
    tenant_virtual_service: dict[str, float],
    remaining_service_s,
    solo_service_s,
) -> dict[str, Any]:
    """Choose the least-served tenant, then its highest-slowdown request."""
    if not jobs:
        raise ValueError("cannot choose from an empty job list")
    tenants = {str(job["tenant"]) for job in jobs}
    tenant = min(
        tenants,
        key=lambda item: (tenant_virtual_service.get(item, 0.0), item),
    )
    candidates = [job for job in jobs if str(job["tenant"]) == tenant]
    return max(
        candidates,
        key=lambda job: (
            predicted_slowdown(
                elapsed_s=now_s - float(job["arrival_s"]),
                remaining_s=float(remaining_service_s(job)),
                solo_s=float(solo_service_s(job)),
            ),
            -int(job["sequence"]),
        ),
    )


def fair_tenant_choice(
    jobs: list[dict[str, Any]],
    *,
    tenant_virtual_service: dict[str, float],
) -> dict[str, Any]:
    """Choose the least-served tenant, then its oldest queued request."""
    if not jobs:
        raise ValueError("cannot choose from an empty job list")
    tenants = {str(job["tenant"]) for job in jobs}
    tenant = min(
        tenants,
        key=lambda item: (tenant_virtual_service.get(item, 0.0), item),
    )
    return min(
        (job for job in jobs if str(job["tenant"]) == tenant),
        key=lambda job: (float(job["arrival_s"]), int(job["sequence"])),
    )


def age_aware_fair_tenant_choice(
    jobs: list[dict[str, Any]],
    *,
    now_s: float,
    tenant_virtual_service: dict[str, float],
    soft_threshold_s: float,
    hard_threshold_s: float,
) -> tuple[dict[str, Any], dict[str, float | str]]:
    """Apply max-min sharing with two tiers of request-age protection.

    Below the soft threshold this is ordinary per-tenant max-min selection.
    Once any request crosses the soft threshold, selection remains max-min but
    is restricted to tenants with aged requests. A request crossing the hard
    threshold is selected globally by oldest original arrival time. Using the
    same arrival timestamp at both stages prevents handoff from resetting age.
    """
    if not jobs:
        raise ValueError("cannot choose from an empty job list")
    if soft_threshold_s <= 0:
        raise ValueError("soft age threshold must be positive")
    if hard_threshold_s < soft_threshold_s:
        raise ValueError("hard age threshold must be at least the soft threshold")

    def age_s(job: dict[str, Any]) -> float:
        return max(0.0, now_s - float(job["arrival_s"]))

    hard_aged = [job for job in jobs if age_s(job) >= hard_threshold_s]
    if hard_aged:
        selected = min(
            hard_aged,
            key=lambda job: (float(job["arrival_s"]), int(job["sequence"])),
        )
        mode = "hard_oldest_request"
    else:
        soft_tenants = {
            str(job["tenant"])
            for job in jobs
            if age_s(job) >= soft_threshold_s
        }
        candidates = (
            [job for job in jobs if str(job["tenant"]) in soft_tenants]
            if soft_tenants else jobs
        )
        selected = fair_tenant_choice(
            candidates,
            tenant_virtual_service=tenant_virtual_service,
        )
        mode = "soft_aged_tenant_max_min" if soft_tenants else "max_min"

    return selected, {
        "mode": mode,
        "request_age_s": age_s(selected),
        "soft_threshold_s": soft_threshold_s,
        "hard_threshold_s": hard_threshold_s,
    }


def cross_stage_tenant_choice(
    jobs: list[dict[str, Any]],
    *,
    tenant_prep_virtual_service: dict[str, float],
    tenant_vlm_virtual_service: dict[str, float],
    active_tenants: set[str],
    tenant_pipeline_supply: dict[str, int],
    debt_threshold_s: float,
    max_boost_s: float,
    unblock_target: int,
) -> tuple[dict[str, Any], dict[str, float | int | bool]]:
    """Choose a CPU tenant using bounded, readiness-aware GPU feedback.

    The preparation counter remains the base score. A GPU-behind tenant gets
    a bounded decrease in that score only while it lacks enough work between
    preparation dispatch and GPU completion. Counting all of that released
    work prevents parallel CPU workers from producing a downstream backlog.

    This is an online coordination heuristic, not the offline fair reference.
    """
    if not jobs:
        raise ValueError("cannot choose from an empty job list")
    if debt_threshold_s < 0 or max_boost_s < 0:
        raise ValueError("cross-stage debt and boost values must be non-negative")
    if unblock_target < 1:
        raise ValueError("cross-stage unblock target must be positive")

    queued_tenants = {str(job["tenant"]) for job in jobs}
    frontier_tenants = active_tenants | queued_tenants
    gpu_frontier_s = max(
        (
            tenant_vlm_virtual_service.get(tenant, 0.0)
            for tenant in frontier_tenants
        ),
        default=0.0,
    )

    decisions: dict[str, dict[str, float | int | bool]] = {}
    for tenant in queued_tenants:
        gpu_service_s = tenant_vlm_virtual_service.get(tenant, 0.0)
        gpu_debt_s = max(0.0, gpu_frontier_s - gpu_service_s)
        pipeline_supply = tenant_pipeline_supply.get(tenant, 0)
        unblocking = (
            gpu_debt_s > debt_threshold_s
            and pipeline_supply < unblock_target
        )
        boost_s = min(gpu_debt_s, max_boost_s) if unblocking else 0.0
        prep_service_s = tenant_prep_virtual_service.get(tenant, 0.0)
        decisions[tenant] = {
            "gpu_frontier_s": gpu_frontier_s,
            "gpu_debt_s": gpu_debt_s,
            "pipeline_supply": pipeline_supply,
            "unblocking": unblocking,
            "boost_s": boost_s,
            "effective_prep_service_s": prep_service_s - boost_s,
        }

    tenant = min(
        queued_tenants,
        key=lambda item: (
            float(decisions[item]["effective_prep_service_s"]),
            tenant_prep_virtual_service.get(item, 0.0),
            tenant_vlm_virtual_service.get(item, 0.0),
            item,
        ),
    )
    selected = min(
        (job for job in jobs if str(job["tenant"]) == tenant),
        key=lambda job: (float(job["arrival_s"]), int(job["sequence"])),
    )
    return selected, decisions[tenant]


def fair_tenant_priority_choice(
    jobs: list[dict[str, Any]],
    *,
    now_s: float,
    tenant_virtual_service: dict[str, float],
    background_aging_s: float,
) -> dict[str, Any]:
    """Choose the least-served tenant, then priority order within it.

    Background work that has waited for ``background_aging_s`` is promoted to
    the best priority currently queued for its tenant. This bounds starvation
    without allowing one tenant's priority labels to consume another tenant's
    fair share.
    """
    if not jobs:
        raise ValueError("cannot choose from an empty job list")
    tenants = {str(job["tenant"]) for job in jobs}
    tenant = min(
        tenants,
        key=lambda item: (tenant_virtual_service.get(item, 0.0), item),
    )
    candidates = [job for job in jobs if str(job["tenant"]) == tenant]
    best_priority = min(int(job["priority"]) for job in candidates)

    def key(job: dict[str, Any]) -> tuple[int, float, int]:
        wait_s = max(0.0, now_s - float(job["arrival_s"]))
        aged_background = (
            job["workload"] == "background"
            and wait_s >= background_aging_s
        )
        effective_priority = (
            best_priority if aged_background else int(job["priority"])
        )
        return (
            effective_priority,
            float(job["arrival_s"]),
            int(job["sequence"]),
        )

    return min(candidates, key=key)


def shortest_job_choice(jobs: list[dict[str, Any]], service_s) -> dict[str, Any]:
    """Choose the request with the smallest predicted remaining service."""
    if not jobs:
        raise ValueError("cannot choose from an empty job list")
    return min(
        jobs,
        key=lambda job: (float(service_s(job)), int(job["sequence"])),
    )


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_media_path(value: Any) -> str | None:
    """Return a stable lookup key for local media paths."""
    if value is None or value == "":
        return None
    text = str(value)
    if "://" in text:
        return text
    return str(Path(text).expanduser().resolve(strict=False))


def load_video_metadata_index(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional pre-probed metadata keyed by a video's local path."""
    if path is None:
        return {}
    index: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        video = row.get("video") or row.get("video_path") or row.get("path")
        key = normalized_media_path(video)
        if key is None:
            raise ValueError(f"metadata row has no video path: {row}")
        index[key] = {
            name: row[name]
            for name in (
                "duration_s", "video_width", "video_height", "video_codec"
            )
            if row.get(name) is not None
        }
    return index


def attach_video_metadata(
    row: dict[str, Any],
    index: dict[str, dict[str, Any]],
) -> None:
    """Attach scheduling metadata without replacing trace-provided values."""
    video = row.get("video") or row.get("video_path") or row.get("path")
    key = normalized_media_path(video)
    if key is None:
        return
    for name, value in index.get(key, {}).items():
        row.setdefault(name, value)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK, path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def qid(row: dict[str, Any]) -> str:
    value = row.get("qid") or row.get("question_id") or row.get("id")
    if value is None:
        raise ValueError("row has no qid/question_id/id")
    return str(value)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_label(text: str | None, choice_count: int) -> str | None:
    if not text:
        return None
    valid = [chr(ord("A") + index) for index in range(choice_count)]
    normalized = text.strip().upper()
    if normalized in valid:
        return normalized
    match = re.search(r"\b(" + "|".join(valid) + r")\b", normalized)
    return match.group(1) if match else None


def make_codec_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        max_tokens=args.max_tokens,
        max_pixels=args.max_pixels,
        window_s=8.0,
        decode_backend=args.decode_backend,
        batch_decode=None,
        decode_max_side=args.decode_max_side,
        index_timeout_s=args.index_timeout_s,
        decode_timeout_s=args.decode_timeout_s,
        request_timeout_s=args.request_timeout_s,
        refine_with_clip=False,
        clip_device="cpu",
        refine_candidates=1,
        refine_regions=1,
        refine_radius_s=1.0,
        probe_max_side=args.decode_max_side,
    )


def prepare_uniform(
    job: dict[str, Any], codec: Any, codec_args: SimpleNamespace
) -> dict[str, Any]:
    """Decode uniformly sampled JPEGs outside vLLM."""
    row = job["row"]
    if job["modality"] == "text":
        started = time.perf_counter()
        return {
            "content": str(
                row.get("prompt_override")
                or row.get("question")
                or "Respond briefly."
            ),
            "duration_s": 0.0,
            "timestamps": [],
            "decode_service_s": time.perf_counter() - started,
        }
    video = Path(row["video"])
    started = time.perf_counter()
    duration_s = codec.probe_duration(video, codec_args.index_timeout_s)
    timestamps = codec.temporal_anchors(duration_s, job["frame_count"])
    if codec_args.decode_backend in (
        "batch_cpu", "batch_nvdec", "indexed_nvdec"
    ):
        jpegs = codec_args.batch_decode(
            video, timestamps, codec_args.decode_max_side,
            codec_args.decode_timeout_s,
        )
    else:
        jpegs = [
            codec.decode_jpeg(
                video, timestamp, codec_args.decode_max_side,
                codec_args.decode_timeout_s,
            )
            for timestamp in timestamps
        ]
    content: list[dict[str, Any]] = []
    for jpeg in jpegs:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(jpeg).decode("ascii")
                },
            }
        )
    content.append({"type": "text", "text": codec.make_prompt(row)})
    return {
        "content": content,
        "duration_s": duration_s,
        "timestamps": timestamps,
        "decode_service_s": time.perf_counter() - started,
    }


def prepare_with_affinity(
    job: dict[str, Any], codec: Any, codec_args: SimpleNamespace,
    cpu_sets: dict[str, set[int] | None],
) -> dict[str, Any]:
    """Prepare one request while optionally pinning its worker thread."""
    requested = cpu_sets.get(job["workload"])
    if requested is None:
        return prepare_uniform(job, codec, codec_args)
    original = os.sched_getaffinity(0)
    os.sched_setaffinity(0, requested)
    try:
        # Decoder subprocesses inherit the calling worker thread's affinity.
        return prepare_uniform(job, codec, codec_args)
    finally:
        os.sched_setaffinity(0, original)


def call_vllm(
    job: dict[str, Any],
    prepared: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
) -> dict[str, Any]:
    """Stream one prepared request so time to first token is observable."""
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=args.request_timeout_s,
        max_retries=0,
    )
    submitted_at = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    engine_priority = (
        0 if args.prep_policy in SCHEDULER_OWNED_POLICIES
        else job["priority"]
    )
    try:
        request_options: dict[str, Any] = {
            "model": args.model,
            "messages": [{"role": "user", "content": prepared["content"]}],
            "temperature": 0.0,
            "max_tokens": job["max_tokens"],
            "stream": True,
            "extra_body": {
                "priority": engine_priority,
                "mm_processor_kwargs": {"max_pixels": args.max_pixels},
                "ignore_eos": args.ignore_eos,
            },
        }
        if args.engine_token_profile is not None:
            request_options["stream_options"] = {"include_usage": True}
        stream = client.chat.completions.create(
            **request_options,
        )
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(usage.prompt_tokens)
                completion_tokens = int(usage.completion_tokens)
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(delta)
    except Exception as exc:
        error = repr(exc)
    completed_at = time.perf_counter()
    return {
        "text": "".join(pieces),
        "error": error,
        "engine_priority": engine_priority,
        "ttft_from_vllm_submit_s": (
            None if first_token_at is None else first_token_at - submitted_at
        ),
        "vlm_service_s": completed_at - submitted_at,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "first_token_offset_s": (
            None if first_token_at is None else first_token_at - submitted_at
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "prep_queue_wait_s",
        "prep_service_s",
        "prepared_queue_wait_s",
        "engine_to_first_token_s",
        "end_to_end_ttft_s",
        "end_to_end_s",
        "estimated_slowdown",
    )
    summary: dict[str, Any] = {
        "requests": len(rows),
        "errors": sum(row.get("error") is not None for row in rows),
        "correct": sum(bool(row.get("correct")) for row in rows),
        "accuracy_percent": (
            100.0 * sum(bool(row.get("correct")) for row in rows) / len(rows)
            if rows
            else None
        ),
    }
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[f"mean_{field}"] = mean(values)
        summary[f"p50_{field}"] = percentile(values, 0.50)
        summary[f"p95_{field}"] = percentile(values, 0.95)
        summary[f"max_{field}"] = max(values) if values else None
    slo_rows = [row for row in rows if row.get("ttft_slo_s") is not None]
    summary["ttft_slo_requests"] = len(slo_rows)
    summary["ttft_slo_attained"] = sum(
        bool(row.get("slo_attained")) for row in slo_rows
    )
    summary["ttft_slo_attainment_percent"] = (
        100.0 * summary["ttft_slo_attained"] / len(slo_rows)
        if slo_rows else None
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-dataset", type=Path)
    parser.add_argument("--urgent-dataset", type=Path)
    parser.add_argument("--arrival-trace", type=Path, help="Per-request JSONL arrival trace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument(
        "--ports", type=int, nargs="+",
        help="vLLM replica ports; overrides --port when provided",
    )
    parser.add_argument(
        "--replica-routing",
        choices=["round_robin", "least_inflight", "workload_isolated"],
        default="least_inflight",
    )
    parser.add_argument(
        "--background-ports", type=int, nargs="+",
        help=("vLLM ports dedicated to background requests when "
              "--replica-routing workload_isolated is selected"),
    )
    parser.add_argument(
        "--urgent-ports", type=int, nargs="+",
        help=("vLLM ports dedicated to urgent requests when "
              "--replica-routing workload_isolated is selected"),
    )
    parser.add_argument(
        "--prep-policy",
        choices=[
            "fcfs", "priority", "priority_reserved", "slo_adaptive",
            "static_isolation", "sjf", "max_min", "prep_max_min",
            "tenant_round_robin", "tenant_fair", "tenant_priority",
            "fair_slowdown", "engine_tenant_fair", "cross_stage",
            "age_aware_max_min",
        ],
        default="priority",
    )
    parser.add_argument(
        "--default-tenant", default="default",
        help=("Tenant assigned to requests without tenant/operator/tenant_id "
              "in the arrival trace"),
    )
    parser.add_argument(
        "--tenant-weight", action="append", default=[], metavar="TENANT=WEIGHT",
        help=("Fair-share weight under tenant_fair, tenant_priority, "
              "fair_slowdown, or engine_tenant_fair; may be repeated and "
              "defaults to 1 for unlisted tenants. Unweighted max-min "
              "policies require all tenant weights to equal 1"),
    )
    parser.add_argument("--background-requests", type=int, default=32)
    parser.add_argument("--urgent-requests", type=int, default=16)
    parser.add_argument("--background-arrival-s", type=float, default=0.0)
    parser.add_argument("--urgent-arrival-s", type=float, default=1.0)
    parser.add_argument("--background-frames", type=int, default=128)
    parser.add_argument("--urgent-frames", type=int, default=8)
    parser.add_argument("--background-priority", type=int, default=10)
    parser.add_argument("--urgent-priority", type=int, default=0)
    parser.add_argument("--prep-workers", type=int, default=4)
    parser.add_argument(
        "--background-prep-workers", type=int,
        help=("preparation slots dedicated to background requests under "
              "static_isolation"),
    )
    parser.add_argument(
        "--urgent-prep-workers", type=int,
        help=("preparation slots dedicated to urgent requests under "
              "static_isolation"),
    )
    parser.add_argument(
        "--background-cpu-set",
        help=("optional Linux CPU list dedicated to background preparation "
              "under static_isolation, for example 0-15,32-47"),
    )
    parser.add_argument(
        "--urgent-cpu-set",
        help=("optional Linux CPU list dedicated to urgent preparation under "
              "static_isolation, for example 16-31,48-63"),
    )
    parser.add_argument(
        "--decode-backend",
        choices=["seek_cpu", "batch_cpu", "batch_nvdec", "indexed_nvdec"],
        default="seek_cpu",
        help="External frame preparation backend",
    )
    parser.add_argument(
        "--background-prep-limit",
        type=int,
        help=(
            "Maximum concurrent background preparations under "
            "priority_reserved. Defaults to prep-workers minus one."
        ),
    )
    parser.add_argument(
        "--urgent-prep-reserve", type=int, default=1,
        help=("Slots protected while urgent work is pending under slo_adaptive; "
              "capacity is work-conserving when no urgent work waits"),
    )
    parser.add_argument("--urgent-ttft-slo-s", type=float, default=30.0)
    parser.add_argument("--background-ttft-slo-s", type=float, default=300.0)
    parser.add_argument(
        "--background-aging-s", type=float, default=120.0,
        help="Promote background work after this preparation-queue wait",
    )
    parser.add_argument("--prep-fixed-cost-s", type=float, default=0.25)
    parser.add_argument("--prep-seconds-per-frame", type=float, default=0.12)
    parser.add_argument("--prep-cost-ewma-alpha", type=float, default=0.2)
    parser.add_argument(
        "--cost-profiler",
        choices=["frame_ewma", "metadata_ewma"],
        default="frame_ewma",
        help=("Default online profiler for both stages. frame_ewma preserves "
              "the original frame-count estimator; metadata_ewma additionally "
              "conditions on backend, duration, resolution, and codec. Stage-"
              "specific options override this value."),
    )
    parser.add_argument(
        "--prep-cost-profiler",
        choices=["frame_ewma", "metadata_ewma"],
        help="Preparation-stage profiler; defaults to --cost-profiler",
    )
    parser.add_argument(
        "--engine-cost-profiler",
        choices=["frame_ewma", "metadata_ewma"],
        help="Inference-admission profiler; defaults to --cost-profiler",
    )
    parser.add_argument(
        "--engine-token-profile",
        type=Path,
        help=("JSON produced by analyze_vllm_token_service_profile.py. When "
              "provided, inference admission and reconciliation use its "
              "batch-aware text, visual, and output-token cost instead of "
              "request residence time."),
    )
    parser.add_argument(
        "--cost-profiler-min-samples", type=int, default=1,
        help="Observations required before a learned profile is used",
    )
    parser.add_argument(
        "--no-service-reconciliation", action="store_true",
        help=("Keep dispatch-time estimated virtual-service charges instead "
              "of replacing them with observed stage service. Intended for "
              "the max-min accounting ablation."),
    )
    parser.add_argument(
        "--completion-only-accounting", action="store_true",
        help=("Do not reserve predicted service at dispatch. Charge observed "
              "service only when a stage completes. This deliberately exposes "
              "the concurrency lag addressed by in-flight accounting."),
    )
    parser.add_argument(
        "--video-metadata-index", type=Path,
        help=("Optional JSONL produced by build_video_metadata_index.py. "
              "It makes duration, resolution, and codec available to the "
              "metadata profiler before preparation admission."),
    )
    parser.add_argument(
        "--vlm-fixed-cost-s", type=float, default=4.0,
        help=("Initial inference-service estimate used by fair_slowdown; "
              "replaced per frame count by online EWMA observations"),
    )
    parser.add_argument("--vlm-concurrency", type=int, default=4)
    parser.add_argument(
        "--age-soft-threshold-s", type=float, default=75.0,
        help=("Under age_aware_max_min, restrict max-min selection to tenants "
              "with requests at least this old"),
    )
    parser.add_argument(
        "--age-hard-threshold-s", type=float, default=150.0,
        help=("Under age_aware_max_min, select the globally oldest eligible "
              "request after it reaches this age"),
    )
    parser.add_argument("--prepared-queue-depth", type=int, default=32)
    parser.add_argument(
        "--cross-stage-debt-threshold-s", type=float, default=0.0,
        help=("Minimum normalized GPU-service debt required before the "
              "cross_stage policy boosts CPU preparation"),
    )
    parser.add_argument(
        "--cross-stage-max-boost-s", type=float, default=4.0,
        help=("Maximum normalized seconds subtracted from a tenant's CPU "
              "virtual-service score by cross-stage unblocking"),
    )
    parser.add_argument(
        "--cross-stage-unblock-target", type=int, default=1,
        help=("Stop cross-stage boosting after this many requests for the "
              "tenant are preparation-in-flight, GPU-ready, or GPU-in-flight"),
    )
    parser.add_argument(
        "--background-prepared-queue-depth", type=int,
        help=("handoff slots dedicated to background requests under "
              "static_isolation"),
    )
    parser.add_argument(
        "--urgent-prepared-queue-depth", type=int,
        help=("handoff slots dedicated to urgent requests under "
              "static_isolation"),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--ignore-eos", action="store_true",
        help=("Force every request to consume its complete max-token budget. "
              "Useful for controlled decode-saturation experiments."),
    )
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--decode-max-side", type=int, default=448)
    parser.add_argument("--index-timeout-s", type=float, default=120.0)
    parser.add_argument("--decode-timeout-s", type=float, default=60.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()
    prep_cost_profiler_mode = args.prep_cost_profiler or args.cost_profiler
    engine_cost_profiler_mode = args.engine_cost_profiler or args.cost_profiler

    try:
        engine_token_profile = (
            None
            if args.engine_token_profile is None
            else BatchAwareTokenCostProfile(args.engine_token_profile)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(f"cannot load --engine-token-profile: {exc}")

    if min(args.prep_workers, args.vlm_concurrency, args.prepared_queue_depth) < 1:
        parser.error("worker and queue counts must be positive")
    if not 0 <= args.urgent_prep_reserve < args.prep_workers:
        parser.error("--urgent-prep-reserve must be in [0, prep-workers)")
    if min(args.urgent_ttft_slo_s, args.background_ttft_slo_s) <= 0:
        parser.error("TTFT SLOs must be positive")
    if args.background_aging_s <= 0:
        parser.error("--background-aging-s must be positive")
    if args.prep_fixed_cost_s < 0 or args.prep_seconds_per_frame <= 0:
        parser.error("preparation cost estimates must be positive")
    if args.vlm_fixed_cost_s <= 0:
        parser.error("--vlm-fixed-cost-s must be positive")
    if args.age_soft_threshold_s <= 0:
        parser.error("--age-soft-threshold-s must be positive")
    if args.age_hard_threshold_s < args.age_soft_threshold_s:
        parser.error(
            "--age-hard-threshold-s must be at least --age-soft-threshold-s"
        )
    if args.cross_stage_debt_threshold_s < 0:
        parser.error("--cross-stage-debt-threshold-s must be non-negative")
    if args.cross_stage_max_boost_s < 0:
        parser.error("--cross-stage-max-boost-s must be non-negative")
    if args.cross_stage_unblock_target < 1:
        parser.error("--cross-stage-unblock-target must be positive")
    if not 0 < args.prep_cost_ewma_alpha <= 1:
        parser.error("--prep-cost-ewma-alpha must be in (0, 1]")
    if args.cost_profiler_min_samples < 1:
        parser.error("--cost-profiler-min-samples must be positive")
    try:
        tenant_weights = parse_tenant_weights(args.tenant_weight)
        if args.prep_policy in {
            "max_min", "prep_max_min", "age_aware_max_min",
        }:
            validate_max_min_tenant_weights(tenant_weights)
    except ValueError as exc:
        parser.error(str(exc))
    if args.no_service_reconciliation and args.completion_only_accounting:
        parser.error(
            "--no-service-reconciliation and --completion-only-accounting "
            "are mutually exclusive"
        )
    if args.background_prep_limit is None:
        args.background_prep_limit = max(1, args.prep_workers - 1)
    if not 1 <= args.background_prep_limit <= args.prep_workers:
        parser.error(
            "--background-prep-limit must be between 1 and --prep-workers"
        )
    if args.prep_policy == "static_isolation":
        if (
            args.background_prep_workers is None
            or args.urgent_prep_workers is None
        ):
            parser.error(
                "static_isolation requires --background-prep-workers and "
                "--urgent-prep-workers"
            )
        if min(args.background_prep_workers, args.urgent_prep_workers) < 1:
            parser.error("static-isolation preparation quotas must be positive")
        if (
            args.background_prep_workers + args.urgent_prep_workers
            != args.prep_workers
        ):
            parser.error(
                "static-isolation preparation quotas must sum to "
                "--prep-workers"
            )
        if (
            args.background_prepared_queue_depth is None
            or args.urgent_prepared_queue_depth is None
        ):
            parser.error(
                "static_isolation requires "
                "--background-prepared-queue-depth and "
                "--urgent-prepared-queue-depth"
            )
        if min(
            args.background_prepared_queue_depth,
            args.urgent_prepared_queue_depth,
        ) < 1:
            parser.error("static-isolation handoff quotas must be positive")
        if (
            args.background_prepared_queue_depth
            + args.urgent_prepared_queue_depth
            != args.prepared_queue_depth
        ):
            parser.error(
                "static-isolation handoff quotas must sum to "
                "--prepared-queue-depth"
            )
        try:
            background_cpu_set = parse_cpu_set(args.background_cpu_set)
            urgent_cpu_set = parse_cpu_set(args.urgent_cpu_set)
        except ValueError as exc:
            parser.error(str(exc))
        if (background_cpu_set is None) != (urgent_cpu_set is None):
            parser.error(
                "provide both --background-cpu-set and --urgent-cpu-set, "
                "or neither"
            )
        if background_cpu_set is not None:
            if background_cpu_set & urgent_cpu_set:
                parser.error("background and urgent CPU sets must be disjoint")
            allowed_cpus = os.sched_getaffinity(0)
            if not (background_cpu_set | urgent_cpu_set) <= allowed_cpus:
                parser.error(
                    "requested CPU sets include CPUs outside this process's "
                    f"allowed affinity: {sorted(allowed_cpus)}"
                )
    else:
        if args.background_cpu_set or args.urgent_cpu_set:
            parser.error("CPU-set isolation requires static_isolation")
        background_cpu_set = None
        urgent_cpu_set = None
    if args.arrival_trace is None:
        if args.background_dataset is None or args.urgent_dataset is None:
            parser.error("provide --arrival-trace, or both dataset arguments")
        if min(args.background_requests, args.urgent_requests) < 1:
            parser.error("request counts must be positive")
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")

    background_rows: list[dict[str, Any]] = []
    urgent_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    video_metadata_index = load_video_metadata_index(args.video_metadata_index)
    if args.arrival_trace is not None:
        trace_rows = load_jsonl(args.arrival_trace)
        if not trace_rows:
            raise SystemExit("arrival trace is empty")
    else:
        background_rows = load_jsonl(args.background_dataset)[: args.background_requests]
        urgent_rows = load_jsonl(args.urgent_dataset)[: args.urgent_requests]
        if len(background_rows) != args.background_requests:
            raise SystemExit("background dataset has fewer rows than requested")
        if len(urgent_rows) != args.urgent_requests:
            raise SystemExit("urgent dataset has fewer rows than requested")

    codec = import_path("mixed_priority_codec", CODEC_PATH)
    codec_args = make_codec_args(args)
    cpu_sets = {
        "background": background_cpu_set,
        "urgent": urgent_cpu_set,
    }
    if args.decode_backend in ("batch_cpu", "batch_nvdec", "indexed_nvdec"):
        module_name = {
            "batch_cpu": "batched_cpu_decode",
            "batch_nvdec": "batched_nvdec_decode",
            "indexed_nvdec": "indexed_nvdec_decode",
        }[args.decode_backend]
        batch_codec = import_path(
            module_name, CODEC_PATH.with_name(f"{module_name}.py")
        )
        codec_args.batch_decode = batch_codec.decode_jpegs_batch_cpu
    ports = list(dict.fromkeys(args.ports or [args.port]))
    background_ports = list(dict.fromkeys(args.background_ports or []))
    urgent_ports = list(dict.fromkeys(args.urgent_ports or []))
    if args.replica_routing == "workload_isolated":
        if not background_ports or not urgent_ports:
            parser.error(
                "workload_isolated requires --background-ports and "
                "--urgent-ports"
            )
        if set(background_ports) & set(urgent_ports):
            parser.error("background and urgent port sets must be disjoint")
        isolated_ports = set(background_ports) | set(urgent_ports)
        if not isolated_ports <= set(ports):
            parser.error("isolated workload ports must also appear in --ports")
    elif background_ports or urgent_ports:
        parser.error(
            "--background-ports/--urgent-ports require "
            "--replica-routing workload_isolated"
        )
    base_urls = {port: f"http://127.0.0.1:{port}/v1" for port in ports}
    for base_url in base_urls.values():
        readiness_client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=min(args.request_timeout_s, 30.0),
            max_retries=0,
        )
        readiness_client.models.list()

    args.output.mkdir(parents=True)
    results_path = args.output / "results.jsonl"
    events_path = args.output / "events.jsonl"
    started = time.perf_counter()

    arrivals: list[tuple[float, int, dict[str, Any]]] = []

    def add_job(
        row, workload, arrival_s, frames, priority, request_id, tenant,
        profiled_solo_service_s=None, max_tokens=None, modality="video",
    ):
        nonlocal_sequence = len(arrivals)
        job = {
            "request_id": request_id, "workload": workload, "row": row,
            "arrival_s": float(arrival_s), "frame_count": int(frames),
            "priority": int(priority), "sequence": nonlocal_sequence,
            "tenant": str(tenant),
            "max_tokens": int(
                args.max_tokens if max_tokens is None else max_tokens
            ),
            "modality": str(modality),
            "profiled_solo_service_s": (
                None
                if profiled_solo_service_s is None
                else float(profiled_solo_service_s)
            ),
        }
        if (
            job["arrival_s"] < 0
            or job["frame_count"] < (0 if job["modality"] == "text" else 1)
            or job["max_tokens"] < 1
            or job["modality"] not in {"text", "video"}
            or not job["tenant"]
            or (
                job["profiled_solo_service_s"] is not None
                and job["profiled_solo_service_s"] <= 0
            )
        ):
            raise SystemExit(f"invalid trace job: {request_id}")
        heapq.heappush(arrivals, (job["arrival_s"], nonlocal_sequence, job))

    if trace_rows:
        for index, trace_row in enumerate(trace_rows):
            row = dict(trace_row)
            attach_video_metadata(row, video_metadata_index)
            workload = str(row.pop("class", row.pop("workload", "background")))
            arrival_s = row.pop("arrival_s", 0.0)
            frames = row.pop("frame_count", 8)
            priority = row.pop("priority", 0)
            request_id = str(row.pop("request_id", f"{workload}-{index}"))
            profiled_solo_service_s = row.pop("solo_service_s", None)
            max_tokens = row.pop("max_tokens", args.max_tokens)
            modality = row.pop("modality", "video")
            tenant = args.default_tenant
            for field in ("tenant", "operator", "tenant_id"):
                if field in row:
                    tenant = row.pop(field)
                    break
            add_job(
                row, workload, arrival_s, frames, priority, request_id, tenant,
                profiled_solo_service_s,
                max_tokens,
                modality,
            )
    else:
        for workload, rows, arrival_s, frames, priority in (
            ("background", background_rows, args.background_arrival_s, args.background_frames, args.background_priority),
            ("urgent", urgent_rows, args.urgent_arrival_s, args.urgent_frames, args.urgent_priority),
        ):
            for index, row in enumerate(rows):
                row = dict(row)
                attach_video_metadata(row, video_metadata_index)
                add_job(
                    row, workload, arrival_s, frames, priority,
                    f"{workload}-{index}", args.default_tenant,
                )

    pending: list[tuple[int, int, dict[str, Any]]] = []
    ready: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    prep_futures: dict[Future, dict[str, Any]] = {}
    vlm_futures: dict[
        Future, tuple[dict[str, Any], dict[str, Any], int]
    ] = {}
    completed: list[dict[str, Any]] = []
    prep_cost_profiler = OnlineStageCostProfiler(
        stage="prep",
        mode=prep_cost_profiler_mode,
        backend=args.decode_backend,
        alpha=args.prep_cost_ewma_alpha,
        min_samples=args.cost_profiler_min_samples,
    )
    vlm_cost_profiler = OnlineStageCostProfiler(
        stage="engine",
        mode=engine_cost_profiler_mode,
        backend=args.decode_backend,
        alpha=args.prep_cost_ewma_alpha,
        min_samples=args.cost_profiler_min_samples,
    )
    tenant_prep_virtual_service: dict[str, float] = {}
    tenant_vlm_virtual_service: dict[str, float] = {}
    tenant_prep_dispatches: dict[str, float] = {}
    tenant_vlm_dispatches: dict[str, float] = {}
    tenant_active_requests: dict[str, int] = defaultdict(int)
    replica_inflight = {port: 0 for port in ports}
    round_robin_cursor = 0

    def elapsed() -> float:
        return time.perf_counter() - started

    def scheduling_key(job: dict[str, Any]) -> int:
        return (
            0
            if args.prep_policy in {
                "fcfs", "max_min", "prep_max_min", "tenant_round_robin",
                "engine_tenant_fair", "age_aware_max_min",
            }
            else job["priority"]
        )

    def predicted_prep_service_s(job: dict[str, Any]) -> float:
        frames = job["frame_count"]
        return prep_cost_profiler.predict(
            job,
            default_s=(
                args.prep_fixed_cost_s + args.prep_seconds_per_frame * frames
            ),
        )

    def engine_profile_concurrency(job: dict[str, Any]) -> int:
        replica_port = job.get("replica_port")
        if replica_port in replica_inflight:
            return max(1, replica_inflight[replica_port])
        candidates = replica_candidates(job)
        if not candidates:
            return 1
        return max(1, min(replica_inflight[port] for port in candidates) + 1)

    def predicted_vlm_service_s(job: dict[str, Any]) -> float:
        if engine_token_profile is not None:
            return engine_token_profile.predict(
                job, concurrency=engine_profile_concurrency(job)
            )
        return vlm_cost_profiler.predict(
            job, default_s=args.vlm_fixed_cost_s,
        )

    def predicted_solo_service_s(job: dict[str, Any]) -> float:
        if job.get("profiled_solo_service_s") is not None:
            return float(job["profiled_solo_service_s"])
        return predicted_prep_service_s(job) + predicted_vlm_service_s(job)

    def tenant_weight(job: dict[str, Any]) -> float:
        return tenant_weights.get(str(job["tenant"]), 1.0)

    def initialize_tenant_virtual_service(job: dict[str, Any]) -> None:
        """Start or reactivate a tenant at the fair-service frontier."""
        tenant = str(job["tenant"])
        for virtual_service in (
            tenant_prep_virtual_service, tenant_vlm_virtual_service,
            tenant_prep_dispatches, tenant_vlm_dispatches,
        ):
            active_values = [
                virtual_service[item]
                for item, count in tenant_active_requests.items()
                if count > 0 and item in virtual_service
            ]
            frontier = min(active_values) if active_values else 0.0
            virtual_service[tenant] = max(
                virtual_service.get(tenant, 0.0), frontier,
            )
        tenant_active_requests[tenant] += 1

    def finish_tenant_request(job: dict[str, Any]) -> None:
        tenant = str(job["tenant"])
        tenant_active_requests[tenant] -= 1
        if tenant_active_requests[tenant] < 0:
            raise RuntimeError(f"negative active-request count for {tenant}")

    def charge_virtual_service(
        virtual_service: dict[str, float],
        job: dict[str, Any],
        service_s: float,
    ) -> None:
        tenant = str(job["tenant"])
        virtual_service[tenant] = virtual_service.get(tenant, 0.0) + (
            max(0.0, service_s) / tenant_weight(job)
        )

    def reconcile_virtual_service(
        virtual_service: dict[str, float],
        job: dict[str, Any],
        *,
        estimated_s: float,
        observed_s: float,
    ) -> None:
        """Replace a dispatch-time estimate with observed resource service."""
        tenant = str(job["tenant"])
        adjustment = (observed_s - estimated_s) / tenant_weight(job)
        virtual_service[tenant] = max(
            0.0, virtual_service.get(tenant, 0.0) + adjustment,
        )

    def ttft_slo_s(job: dict[str, Any]) -> float:
        if job["workload"] == "urgent":
            return args.urgent_ttft_slo_s
        return args.background_ttft_slo_s

    def adaptive_key(job: dict[str, Any], now: float) -> tuple[int, float, int]:
        wait_s = max(0.0, now - job["arrival_s"])
        slack_s = (
            job["arrival_s"] + ttft_slo_s(job) - now
            - predicted_prep_service_s(job)
        )
        if job["workload"] == "background" and wait_s > args.background_aging_s:
            slack_s -= wait_s - args.background_aging_s
        return job["priority"], slack_s, job["sequence"]

    def remove_pending(index: int) -> dict[str, Any]:
        _, _, job = pending[index]
        pending[index] = pending[-1]
        pending.pop()
        if pending:
            heapq.heapify(pending)
        return job

    def pipeline_supply_by_tenant() -> dict[str, int]:
        """Count work already released from the pending CPU queue."""
        supply: dict[str, int] = defaultdict(int)
        for job in prep_futures.values():
            supply[str(job["tenant"])] += 1
        for _, _, job, _ in ready:
            supply[str(job["tenant"])] += 1
        for job, _, _ in vlm_futures.values():
            supply[str(job["tenant"])] += 1
        return dict(supply)

    def pop_pending_for_preparation() -> dict[str, Any] | None:
        if not pending:
            return None

        if args.prep_policy in PREPARATION_FAIR_POLICIES:
            now = elapsed()
            jobs = [job for _, _, job in pending]
            if args.prep_policy == "cross_stage":
                selected, decision = cross_stage_tenant_choice(
                    jobs,
                    tenant_prep_virtual_service=tenant_prep_virtual_service,
                    tenant_vlm_virtual_service=tenant_vlm_virtual_service,
                    active_tenants={
                        tenant
                        for tenant, count in tenant_active_requests.items()
                        if count > 0
                    },
                    tenant_pipeline_supply=pipeline_supply_by_tenant(),
                    debt_threshold_s=args.cross_stage_debt_threshold_s,
                    max_boost_s=args.cross_stage_max_boost_s,
                    unblock_target=args.cross_stage_unblock_target,
                )
                selected["cross_stage_decision"] = decision
            elif args.prep_policy == "age_aware_max_min":
                selected, decision = age_aware_fair_tenant_choice(
                    jobs,
                    now_s=now,
                    tenant_virtual_service=tenant_prep_virtual_service,
                    soft_threshold_s=args.age_soft_threshold_s,
                    hard_threshold_s=args.age_hard_threshold_s,
                )
                selected["age_aware_prep_decision"] = decision
            elif args.prep_policy == "fair_slowdown":
                selected = fair_slowdown_choice(
                    jobs,
                    now_s=now,
                    tenant_virtual_service=tenant_prep_virtual_service,
                    remaining_service_s=predicted_solo_service_s,
                    solo_service_s=predicted_solo_service_s,
                )
            elif args.prep_policy == "tenant_priority":
                selected = fair_tenant_priority_choice(
                    jobs,
                    now_s=now,
                    tenant_virtual_service=tenant_prep_virtual_service,
                    background_aging_s=args.background_aging_s,
                )
            else:
                selected = fair_tenant_choice(
                    jobs,
                    tenant_virtual_service=tenant_prep_virtual_service,
                )
            index = next(
                item
                for item, (_, _, job) in enumerate(pending)
                if job is selected
            )
            return remove_pending(index)

        if args.prep_policy == "tenant_round_robin":
            jobs = [job for _, _, job in pending]
            selected = fair_tenant_choice(
                jobs,
                tenant_virtual_service=tenant_prep_dispatches,
            )
            index = next(
                item
                for item, (_, _, job) in enumerate(pending)
                if job is selected
            )
            return remove_pending(index)

        if args.prep_policy == "sjf":
            selected = shortest_job_choice(
                [job for _, _, job in pending], predicted_solo_service_s,
            )
            index = next(
                item for item, (_, _, job) in enumerate(pending)
                if job is selected
            )
            return remove_pending(index)

        active_background = sum(
            job["workload"] == "background" for job in prep_futures.values()
        )

        if args.prep_policy == "static_isolation":
            active_by_workload = {
                workload: sum(
                    job["workload"] == workload
                    for job in prep_futures.values()
                )
                for workload in ("background", "urgent")
            }
            limits = {
                "background": args.background_prep_workers,
                "urgent": args.urgent_prep_workers,
            }
            handoff_limits = {
                "background": args.background_prepared_queue_depth,
                "urgent": args.urgent_prepared_queue_depth,
            }
            admitted_by_workload = {
                workload: (
                    sum(
                        job["workload"] == workload
                        for job in prep_futures.values()
                    )
                    + sum(
                        job["workload"] == workload
                        for _, _, job, _ in ready
                    )
                )
                for workload in ("background", "urgent")
            }
            eligible = [
                index
                for index, (_, _, job) in enumerate(pending)
                if job["workload"] in limits
                and active_by_workload[job["workload"]]
                < limits[job["workload"]]
                and admitted_by_workload[job["workload"]]
                < handoff_limits[job["workload"]]
            ]
            if not eligible:
                return None
            index = min(
                eligible,
                key=lambda item: (pending[item][0], pending[item][1]),
            )
            return remove_pending(index)

        if args.prep_policy == "slo_adaptive":
            urgent_waiting = any(
                job["workload"] == "urgent" for _, _, job in pending
            )
            background_limit = (
                args.prep_workers - args.urgent_prep_reserve
                if urgent_waiting else args.prep_workers
            )
            eligible = [
                index for index, (_, _, job) in enumerate(pending)
                if job["workload"] != "background"
                or active_background < background_limit
            ]
            if not eligible:
                return None
            now = elapsed()
            index = min(
                eligible,
                key=lambda item: adaptive_key(pending[item][2], now),
            )
            return remove_pending(index)

        if args.prep_policy != "priority_reserved":
            return heapq.heappop(pending)[2]

        if active_background < args.background_prep_limit:
            return heapq.heappop(pending)[2]

        eligible = [
            (key, sequence, index)
            for index, (key, sequence, job) in enumerate(pending)
            if job["workload"] != "background"
        ]
        if not eligible:
            return None
        return remove_pending(min(eligible)[2])

    def replica_candidates(job: dict[str, Any]) -> list[int]:
        if args.replica_routing != "workload_isolated":
            return ports
        if job["workload"] == "urgent":
            return urgent_ports
        if job["workload"] == "background":
            return background_ports
        raise RuntimeError(
            "workload_isolated supports only background and urgent requests; "
            f"received {job['workload']!r}"
        )

    def choose_replica(job: dict[str, Any]) -> int | None:
        nonlocal round_robin_cursor
        candidates = [
            port for port in replica_candidates(job)
            if replica_inflight[port] < args.vlm_concurrency
        ]
        if not candidates:
            return None
        if args.replica_routing == "round_robin":
            port = candidates[round_robin_cursor % len(candidates)]
            round_robin_cursor += 1
            return port
        return min(candidates, key=lambda port: (replica_inflight[port], port))

    def remove_ready(index: int):
        item = ready[index]
        ready[index] = ready[-1]
        ready.pop()
        if ready:
            heapq.heapify(ready)
        return item

    def pop_ready_for_vllm():
        eligible = [
            index
            for index, (_, _, job, _) in enumerate(ready)
            if any(
                replica_inflight[port] < args.vlm_concurrency
                for port in replica_candidates(job)
            )
        ]
        if not eligible:
            return None
        if args.prep_policy in INFERENCE_FAIR_POLICIES:
            jobs = [ready[index][2] for index in eligible]
            if args.prep_policy == "age_aware_max_min":
                selected, decision = age_aware_fair_tenant_choice(
                    jobs,
                    now_s=elapsed(),
                    tenant_virtual_service=tenant_vlm_virtual_service,
                    soft_threshold_s=args.age_soft_threshold_s,
                    hard_threshold_s=args.age_hard_threshold_s,
                )
                selected["age_aware_infer_decision"] = decision
            elif args.prep_policy == "fair_slowdown":
                selected = fair_slowdown_choice(
                    jobs,
                    now_s=elapsed(),
                    tenant_virtual_service=tenant_vlm_virtual_service,
                    remaining_service_s=predicted_vlm_service_s,
                    solo_service_s=predicted_solo_service_s,
                )
            elif args.prep_policy == "tenant_priority":
                selected = fair_tenant_priority_choice(
                    jobs,
                    now_s=elapsed(),
                    tenant_virtual_service=tenant_vlm_virtual_service,
                    background_aging_s=args.background_aging_s,
                )
            else:
                selected = fair_tenant_choice(
                    jobs,
                    tenant_virtual_service=tenant_vlm_virtual_service,
                )
            index = next(
                item for item in eligible if ready[item][2] is selected
            )
            return remove_ready(index)
        if args.prep_policy == "tenant_round_robin":
            jobs = [ready[index][2] for index in eligible]
            selected = fair_tenant_choice(
                jobs,
                tenant_virtual_service=tenant_vlm_dispatches,
            )
            index = next(
                item for item in eligible if ready[item][2] is selected
            )
            return remove_ready(index)
        if args.prep_policy == "sjf":
            selected = shortest_job_choice(
                [ready[index][2] for index in eligible],
                predicted_vlm_service_s,
            )
            index = next(
                item for item in eligible if ready[item][2] is selected
            )
            return remove_ready(index)
        index = min(
            eligible,
            key=lambda item: (ready[item][0], ready[item][1]),
        )
        return remove_ready(index)

    def event(name: str, job: dict[str, Any], **extra: Any) -> None:
        append_jsonl(
            events_path,
            {
                "event": name,
                "time_s": elapsed(),
                "request_id": job["request_id"],
                "workload": job["workload"],
                "tenant": job["tenant"],
                "priority": job["priority"],
                "frame_count": job["frame_count"],
                "max_tokens": job["max_tokens"],
                "modality": job["modality"],
                **extra,
            },
        )

    with ThreadPoolExecutor(max_workers=args.prep_workers) as prep_pool, ThreadPoolExecutor(
        max_workers=args.vlm_concurrency * len(ports)
    ) as vlm_pool:
        while arrivals or pending or prep_futures or ready or vlm_futures:
            now = elapsed()
            while arrivals and arrivals[0][0] <= now:
                _, _, job = heapq.heappop(arrivals)
                initialize_tenant_virtual_service(job)
                heapq.heappush(
                    pending, (scheduling_key(job), job["sequence"], job)
                )
                event("arrival", job, pending_depth=len(pending))

            while (
                pending
                and len(prep_futures) < args.prep_workers
                and (
                    args.prep_policy == "static_isolation"
                    or len(ready) + len(prep_futures)
                    < args.prepared_queue_depth
                )
            ):
                job = pop_pending_for_preparation()
                if job is None:
                    break
                job["prep_started_s"] = elapsed()
                job["predicted_prep_service_s"] = predicted_prep_service_s(job)
                job["predicted_vlm_service_s"] = predicted_vlm_service_s(job)
                job["predicted_solo_service_s"] = predicted_solo_service_s(job)
                job["ttft_slo_s"] = ttft_slo_s(job)
                if args.prep_policy in PREPARATION_FAIR_POLICIES:
                    if not args.completion_only_accounting:
                        job["prep_virtual_charge_s"] = job[
                            "predicted_prep_service_s"
                        ]
                        charge_virtual_service(
                            tenant_prep_virtual_service,
                            job,
                            job["prep_virtual_charge_s"],
                        )
                elif args.prep_policy == "tenant_round_robin":
                    tenant = str(job["tenant"])
                    tenant_prep_dispatches[tenant] = (
                        tenant_prep_dispatches.get(tenant, 0.0) + 1.0
                    )
                event(
                    "prep_start", job,
                    pending_depth=len(pending),
                    predicted_prep_service_s=job["predicted_prep_service_s"],
                    deadline_slack_s=adaptive_key(job, elapsed())[1],
                    active_background=sum(
                        item["workload"] == "background"
                        for item in prep_futures.values()
                    ),
                    predicted_solo_service_s=job[
                        "predicted_solo_service_s"
                    ],
                    predicted_slowdown=predicted_slowdown(
                        elapsed_s=elapsed() - job["arrival_s"],
                        remaining_s=job["predicted_solo_service_s"],
                        solo_s=job["predicted_solo_service_s"],
                    ),
                    tenant_virtual_service=tenant_prep_virtual_service.get(
                        job["tenant"], 0.0,
                    ),
                    cross_stage_decision=job.pop(
                        "cross_stage_decision", None,
                    ),
                    age_aware_decision=job.pop(
                        "age_aware_prep_decision", None,
                    ),
                )
                prep_futures[
                    prep_pool.submit(
                        prepare_with_affinity, job, codec, codec_args, cpu_sets
                    )
                ] = job

            while ready and len(vlm_futures) < args.vlm_concurrency * len(ports):
                ready_item = pop_ready_for_vllm()
                if ready_item is None:
                    break
                _, _, job, prepared = ready_item
                replica_port = choose_replica(job)
                if replica_port is None:
                    raise RuntimeError("eligible ready request has no replica capacity")
                replica_inflight[replica_port] += 1
                job["replica_port"] = replica_port
                job["engine_profile_concurrency"] = replica_inflight[replica_port]
                job["vlm_submit_s"] = elapsed()
                job["predicted_vlm_service_s"] = predicted_vlm_service_s(job)
                if args.prep_policy in INFERENCE_FAIR_POLICIES:
                    if not args.completion_only_accounting:
                        job["vlm_virtual_charge_s"] = job[
                            "predicted_vlm_service_s"
                        ]
                        charge_virtual_service(
                            tenant_vlm_virtual_service,
                            job,
                            job["vlm_virtual_charge_s"],
                        )
                elif args.prep_policy == "tenant_round_robin":
                    tenant = str(job["tenant"])
                    tenant_vlm_dispatches[tenant] = (
                        tenant_vlm_dispatches.get(tenant, 0.0) + 1.0
                    )
                event(
                    "vlm_submit", job,
                    ready_depth=len(ready),
                    replica_port=replica_port,
                    replica_inflight=dict(replica_inflight),
                    engine_profile_concurrency=job[
                        "engine_profile_concurrency"
                    ],
                    predicted_vlm_service_s=job[
                        "predicted_vlm_service_s"
                    ],
                    predicted_slowdown=predicted_slowdown(
                        elapsed_s=elapsed() - job["arrival_s"],
                        remaining_s=job["predicted_vlm_service_s"],
                        solo_s=job["predicted_solo_service_s"],
                    ),
                    tenant_virtual_service=tenant_vlm_virtual_service.get(
                        job["tenant"], 0.0,
                    ),
                    age_aware_decision=job.pop(
                        "age_aware_infer_decision", None,
                    ),
                )
                vlm_futures[
                    vlm_pool.submit(
                        call_vllm, job, prepared, args, base_urls[replica_port]
                    )
                ] = (job, prepared, replica_port)

            futures = list(prep_futures) + list(vlm_futures)
            if not futures:
                if arrivals:
                    time.sleep(max(0.0, min(0.05, arrivals[0][0] - elapsed())))
                continue

            done, _ = wait(
                futures, timeout=0.05, return_when=FIRST_COMPLETED
            )
            for future in done:
                if future in prep_futures:
                    job = prep_futures.pop(future)
                    job["prep_ready_s"] = elapsed()
                    if args.prep_policy in PREPARATION_FAIR_POLICIES:
                        observed_prep_service_s = (
                            job["prep_ready_s"] - job["prep_started_s"]
                        )
                        if args.completion_only_accounting:
                            charge_virtual_service(
                                tenant_prep_virtual_service,
                                job,
                                observed_prep_service_s,
                            )
                        elif not args.no_service_reconciliation:
                            reconcile_virtual_service(
                                tenant_prep_virtual_service,
                                job,
                                estimated_s=job["prep_virtual_charge_s"],
                                observed_s=observed_prep_service_s,
                            )
                    try:
                        prepared = future.result()
                        observed = float(prepared["decode_service_s"])
                        prep_cost_profiler.observe(
                            job,
                            observed_s=observed,
                            predicted_s=job["predicted_prep_service_s"],
                            metadata=prepared,
                        )
                        job["profile_metadata"] = {
                            "duration_s": prepared.get("duration_s"),
                            "video_width": prepared.get("video_width"),
                            "video_height": prepared.get("video_height"),
                            "video_codec": prepared.get("video_codec"),
                        }
                    except Exception as exc:
                        result = {
                            "request_id": job["request_id"],
                            "workload": job["workload"],
                            "tenant": job["tenant"],
                            "qid": qid(job["row"]),
                            "priority": job["priority"],
                            "engine_priority": None,
                            "frame_count": job["frame_count"],
                            "replica_port": None,
                            "ttft_slo_s": job["ttft_slo_s"],
                            "predicted_prep_service_s": job["predicted_prep_service_s"],
                            "predicted_vlm_service_s": job["predicted_vlm_service_s"],
                            "predicted_solo_service_s": job["predicted_solo_service_s"],
                            "profiled_solo_service_s": job[
                                "profiled_solo_service_s"
                            ],
                            "arrival_s": job["arrival_s"],
                            "prep_started_s": job["prep_started_s"],
                            "prep_ready_s": job["prep_ready_s"],
                            "error": repr(exc),
                            "correct": False,
                            "prep_queue_wait_s": job["prep_started_s"] - job["arrival_s"],
                            "prep_service_s": job["prep_ready_s"] - job["prep_started_s"],
                            "prepared_queue_wait_s": None,
                            "engine_to_first_token_s": None,
                            "end_to_end_ttft_s": None,
                            "end_to_end_s": job["prep_ready_s"] - job["arrival_s"],
                            "estimated_slowdown": None,
                            "slo_attained": False,
                        }
                        completed.append(result)
                        append_jsonl(results_path, result)
                        event("prep_error", job, error=repr(exc))
                        finish_tenant_request(job)
                        continue
                    event("prep_ready", job, ready_depth=len(ready) + 1)
                    heapq.heappush(
                        ready,
                        (
                            scheduling_key(job),
                            job["sequence"],
                            job,
                            prepared,
                        ),
                    )
                else:
                    job, prepared, replica_port = vlm_futures.pop(future)
                    replica_inflight[replica_port] -= 1
                    job["completion_s"] = elapsed()
                    response = future.result()
                    vlm_wall_s = float(response["vlm_service_s"])
                    if (
                        engine_token_profile is not None
                        and response["prompt_tokens"] is not None
                        and response["completion_tokens"] is not None
                    ):
                        observed_vlm_service_s = engine_token_profile.predict(
                            job,
                            concurrency=job["engine_profile_concurrency"],
                            prompt_tokens=response["prompt_tokens"],
                            completion_tokens=response["completion_tokens"],
                        )
                    elif engine_token_profile is not None:
                        # Preserve the dispatch-time reservation if a server
                        # does not return streaming usage; wall time would
                        # double-count concurrently executing requests.
                        observed_vlm_service_s = job["predicted_vlm_service_s"]
                    else:
                        observed_vlm_service_s = vlm_wall_s
                        vlm_cost_profiler.observe(
                            job,
                            observed_s=observed_vlm_service_s,
                            predicted_s=job["predicted_vlm_service_s"],
                            metadata=prepared,
                        )
                    if args.prep_policy in INFERENCE_FAIR_POLICIES:
                        if args.completion_only_accounting:
                            charge_virtual_service(
                                tenant_vlm_virtual_service,
                                job,
                                observed_vlm_service_s,
                            )
                        elif not args.no_service_reconciliation:
                            reconcile_virtual_service(
                                tenant_vlm_virtual_service,
                                job,
                                estimated_s=job["vlm_virtual_charge_s"],
                                observed_s=observed_vlm_service_s,
                            )
                    first_token_s = (
                        None
                        if response["first_token_offset_s"] is None
                        else job["vlm_submit_s"]
                        + response["first_token_offset_s"]
                    )
                    row = job["row"]
                    prediction = parse_label(
                        response["text"], len(row.get("choices") or [])
                    )
                    gold = row.get("answer_label")
                    if gold is None and row.get("answer_idx") is not None:
                        gold = chr(ord("A") + int(row["answer_idx"]))
                    result = {
                        "request_id": job["request_id"],
                        "workload": job["workload"],
                        "tenant": job["tenant"],
                        "qid": qid(row),
                        "video": row.get("video"),
                        "priority": job["priority"],
                        "engine_priority": response["engine_priority"],
                        "frame_count": job["frame_count"],
                        "max_tokens": job["max_tokens"],
                        "modality": job["modality"],
                        "replica_port": job["replica_port"],
                        "ttft_slo_s": job["ttft_slo_s"],
                        "predicted_prep_service_s": job["predicted_prep_service_s"],
                        "predicted_vlm_service_s": job["predicted_vlm_service_s"],
                        "predicted_solo_service_s": job["predicted_solo_service_s"],
                        "profiled_solo_service_s": job[
                            "profiled_solo_service_s"
                        ],
                        "arrival_s": job["arrival_s"],
                        "prep_started_s": job["prep_started_s"],
                        "prep_ready_s": job["prep_ready_s"],
                        "vlm_submit_s": job["vlm_submit_s"],
                        "first_token_s": first_token_s,
                        "completion_s": job["completion_s"],
                        "prep_queue_wait_s": job["prep_started_s"] - job["arrival_s"],
                        "prep_service_s": job["prep_ready_s"] - job["prep_started_s"],
                        "prepared_queue_wait_s": job["vlm_submit_s"] - job["prep_ready_s"],
                        "vlm_service_s": vlm_wall_s,
                        "inference_accounted_service_s": observed_vlm_service_s,
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                        "engine_profile_concurrency": job[
                            "engine_profile_concurrency"
                        ],
                        "engine_to_first_token_s": response["ttft_from_vllm_submit_s"],
                        "end_to_end_ttft_s": (
                            None
                            if first_token_s is None
                            else first_token_s - job["arrival_s"]
                        ),
                        "end_to_end_s": job["completion_s"] - job["arrival_s"],
                        "estimated_slowdown": (
                            (job["completion_s"] - job["arrival_s"])
                            / job["predicted_solo_service_s"]
                        ),
                        "slo_attained": (
                            first_token_s is not None
                            and first_token_s - job["arrival_s"] <= job["ttft_slo_s"]
                        ),
                        "duration_s": prepared["duration_s"],
                        "selected_timestamps_s": prepared["timestamps"],
                        "prediction_label": prediction,
                        "prediction_text": response["text"],
                        "answer_label": gold,
                        "correct": prediction == gold,
                        "error": response["error"],
                    }
                    completed.append(result)
                    append_jsonl(results_path, result)
                    event("completion", job, error=response["error"])
                    finish_tenant_request(job)
                    print(
                        f"[done] {job['request_id']} priority={job['priority']} "
                        f"prep_wait={result['prep_queue_wait_s']:.2f} "
                        f"ttft={result['end_to_end_ttft_s']} "
                        f"error={response['error'] is not None}",
                        flush=True,
                    )

    wall_s = elapsed()
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tenant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_workload[row["workload"]].append(row)
        by_tenant[row["tenant"]].append(row)
    summary = {
        "method": "mixed_end_to_end_video_priority",
        "model": args.model,
        "prep_policy": args.prep_policy,
        "server_requirement": "--scheduling-policy priority",
        "port": ports[0],
        "ports": ports,
        "replica_count": len(ports),
        "replica_routing": args.replica_routing,
        "wall_time_s": wall_s,
        "total_requests": len(completed),
        "errors": sum(row.get("error") is not None for row in completed),
        "throughput_qps": len(completed) / wall_s if wall_s else 0.0,
        "prep_workers": args.prep_workers,
        "decode_backend": args.decode_backend,
        "tenant_weights": tenant_weights,
        "fairness_mode": (
            "unweighted_max_min_with_age_protection"
            if args.prep_policy == "age_aware_max_min"
            else "unweighted_max_min"
            if args.prep_policy == "max_min"
            else "readiness_aware_cross_stage"
            if args.prep_policy == "cross_stage"
            else "preparation_only_unweighted_max_min"
            if args.prep_policy == "prep_max_min"
            else "tenant_round_robin"
            if args.prep_policy == "tenant_round_robin"
            else None
        ),
        "service_reconciliation": (
            not args.no_service_reconciliation
            and not args.completion_only_accounting
        ),
        "service_accounting_mode": (
            "completion_only"
            if args.completion_only_accounting
            else "estimate_only"
            if args.no_service_reconciliation
            else "predicted_then_reconciled"
        ),
        "default_tenant": args.default_tenant,
        "background_prep_limit": (
            args.background_prep_limit
            if args.prep_policy == "priority_reserved"
            else None
        ),
        "background_prep_workers": (
            args.background_prep_workers
            if args.prep_policy == "static_isolation" else None
        ),
        "urgent_prep_workers": (
            args.urgent_prep_workers
            if args.prep_policy == "static_isolation" else None
        ),
        "background_cpu_set": (
            sorted(background_cpu_set)
            if background_cpu_set is not None else None
        ),
        "urgent_cpu_set": (
            sorted(urgent_cpu_set) if urgent_cpu_set is not None else None
        ),
        "background_ports": (
            background_ports
            if args.replica_routing == "workload_isolated" else None
        ),
        "urgent_ports": (
            urgent_ports
            if args.replica_routing == "workload_isolated" else None
        ),
        "urgent_prep_reserve": (
            args.urgent_prep_reserve
            if args.prep_policy == "slo_adaptive" else None
        ),
        "vlm_concurrency_per_replica": args.vlm_concurrency,
        "vlm_concurrency_total": args.vlm_concurrency * len(ports),
        "prepared_queue_depth": args.prepared_queue_depth,
        "background_prepared_queue_depth": (
            args.background_prepared_queue_depth
            if args.prep_policy == "static_isolation" else None
        ),
        "urgent_prepared_queue_depth": (
            args.urgent_prepared_queue_depth
            if args.prep_policy == "static_isolation" else None
        ),
        "background": summarize(by_workload["background"]),
        "urgent": summarize(by_workload["urgent"]),
        "tenants": {
            tenant: summarize(rows)
            for tenant, rows in sorted(by_tenant.items())
        },
        "configuration": {
            "arrival_trace": str(args.arrival_trace) if args.arrival_trace else None,
            "trace_driven": args.arrival_trace is not None,
            "trace_requests": len(trace_rows) if trace_rows else None,
            "trace_workloads": ({
                name: sum(1 for row in trace_rows if row.get("class", row.get("workload", "background")) == name)
                for name in sorted({str(row.get("class", row.get("workload", "background"))) for row in trace_rows})
            } if trace_rows else None),
            "background_requests": args.background_requests,
            "urgent_requests": args.urgent_requests,
            "background_arrival_s": args.background_arrival_s,
            "urgent_arrival_s": args.urgent_arrival_s,
            "background_frames": args.background_frames,
            "urgent_frames": args.urgent_frames,
            "background_priority": args.background_priority,
            "urgent_priority": args.urgent_priority,
            "ignore_eos": args.ignore_eos,
            "urgent_ttft_slo_s": args.urgent_ttft_slo_s,
            "background_ttft_slo_s": args.background_ttft_slo_s,
            "background_aging_s": args.background_aging_s,
            "age_soft_threshold_s": (
                args.age_soft_threshold_s
                if args.prep_policy == "age_aware_max_min" else None
            ),
            "age_hard_threshold_s": (
                args.age_hard_threshold_s
                if args.prep_policy == "age_aware_max_min" else None
            ),
            "cross_stage_debt_threshold_s": (
                args.cross_stage_debt_threshold_s
                if args.prep_policy == "cross_stage" else None
            ),
            "cross_stage_max_boost_s": (
                args.cross_stage_max_boost_s
                if args.prep_policy == "cross_stage" else None
            ),
            "cross_stage_unblock_target": (
                args.cross_stage_unblock_target
                if args.prep_policy == "cross_stage" else None
            ),
            "prep_fixed_cost_s": args.prep_fixed_cost_s,
            "prep_seconds_per_frame": args.prep_seconds_per_frame,
            "prep_cost_ewma_alpha": args.prep_cost_ewma_alpha,
            "vlm_fixed_cost_s": args.vlm_fixed_cost_s,
            "cost_profiler": args.cost_profiler,
            "prep_cost_profiler": prep_cost_profiler_mode,
            "engine_cost_profiler": engine_cost_profiler_mode,
            "engine_token_profile": (
                None
                if engine_token_profile is None
                else engine_token_profile.snapshot()
            ),
            "cost_profiler_min_samples": args.cost_profiler_min_samples,
            "video_metadata_index": (
                str(args.video_metadata_index)
                if args.video_metadata_index else None
            ),
            "video_metadata_entries": len(video_metadata_index),
            "tenant_weights": tenant_weights,
            "learned_prep_cost_s_by_frame_count": (
                prep_cost_profiler.frame_estimates()
            ),
            "learned_vlm_cost_s_by_frame_count": (
                vlm_cost_profiler.frame_estimates()
            ),
            "prep_cost_profile": prep_cost_profiler.snapshot(),
            "engine_cost_profile": vlm_cost_profiler.snapshot(),
            "tenant_prep_virtual_service": tenant_prep_virtual_service,
            "tenant_vlm_virtual_service": tenant_vlm_virtual_service,
            "tenant_prep_dispatches": tenant_prep_dispatches,
            "tenant_vlm_dispatches": tenant_vlm_dispatches,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
