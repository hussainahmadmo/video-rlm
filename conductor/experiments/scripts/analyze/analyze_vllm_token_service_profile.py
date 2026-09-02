#!/usr/bin/env python3
"""Fit and plot VTC-style token weights from a service-profile JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def median_rows(rows: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), int(row["target"]), int(row["concurrency"]))].append(row)
    result: list[dict[str, float | int | str]] = []
    for (family, target, concurrency), items in sorted(groups.items()):
        result.append(
            {
                "family": family,
                "target": target,
                "concurrency": concurrency,
                "amortized_service_s": float(np.median(
                    [float(item["amortized_service_s"]) for item in items]
                )),
                "mean_prompt_tokens": float(np.median(
                    [float(item["mean_prompt_tokens"]) for item in items]
                )),
                "mean_completion_tokens": float(np.median(
                    [float(item["mean_completion_tokens"]) for item in items]
                )),
            }
        )
    return result


def line_fit(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    if len(xs) < 2 or max(xs) == min(xs):
        return {"intercept": None, "slope": None, "r2": None}
    design = np.column_stack([np.ones(len(xs)), np.asarray(xs, dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(ys, dtype=float), rcond=None)
    predicted = design @ coefficients
    residual = float(np.sum((np.asarray(ys) - predicted) ** 2))
    total = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "r2": None if total == 0 else 1.0 - residual / total,
    }


def fits_by_concurrency(points: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    concurrencies = sorted({int(point["concurrency"]) for point in points})
    for concurrency in concurrencies:
        selected = [point for point in points if int(point["concurrency"]) == concurrency]
        prefill = [point for point in selected if point["family"] == "prefill"]
        decode = [point for point in selected if point["family"] == "decode"]
        visual = [point for point in selected if point["family"] == "visual"]
        prefill_fit = line_fit(
            [float(point["mean_prompt_tokens"]) for point in prefill],
            [float(point["amortized_service_s"]) for point in prefill],
        )
        decode_fit = line_fit(
            [float(point["mean_completion_tokens"]) for point in decode],
            [float(point["amortized_service_s"]) for point in decode],
        )
        visual_baseline = next(
            (
                float(point["mean_prompt_tokens"])
                for point in visual
                if int(point["target"]) == 0
            ),
            min((float(point["mean_prompt_tokens"]) for point in visual), default=0.0),
        )
        visual_fit = line_fit(
            [max(0.0, float(point["mean_prompt_tokens"]) - visual_baseline) for point in visual],
            [float(point["amortized_service_s"]) for point in visual],
        )
        prefill_slope = prefill_fit["slope"]
        decode_slope = decode_fit["slope"]
        visual_slope = visual_fit["slope"]
        valid_prefill = prefill_slope is not None and prefill_slope > 0
        result[str(concurrency)] = {
            "prefill_fit": prefill_fit,
            "decode_fit": decode_fit,
            "visual_fit": visual_fit,
            "visual_prompt_token_baseline": visual_baseline,
            "weights_normalized_to_text_input": {
                "w_text": 1.0,
                "w_output": (
                    float(decode_slope) / float(prefill_slope)
                    if valid_prefill and decode_slope is not None
                    else None
                ),
                "w_visual": (
                    float(visual_slope) / float(prefill_slope)
                    if valid_prefill and visual_slope is not None
                    else None
                ),
            },
        }
    return result


def plot(points: list[dict[str, Any]], output: Path) -> None:
    families = (
        ("prefill", "Reported prompt tokens", "Text prefill"),
        ("decode", "Generated output tokens", "Autoregressive decode"),
        ("visual", "Number of images", "Visual input"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    concurrencies = sorted({int(point["concurrency"]) for point in points})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(concurrencies)))
    for axis, (family, xlabel, title) in zip(axes, families):
        for concurrency, color in zip(concurrencies, colors):
            selected = sorted(
                (
                    point for point in points
                    if point["family"] == family
                    and int(point["concurrency"]) == concurrency
                ),
                key=lambda point: int(point["target"]),
            )
            if family == "prefill":
                xs = [float(point["mean_prompt_tokens"]) for point in selected]
            elif family == "decode":
                xs = [float(point["mean_completion_tokens"]) for point in selected]
            else:
                xs = [int(point["target"]) for point in selected]
            ys = [float(point["amortized_service_s"]) for point in selected]
            axis.plot(xs, ys, marker="o", color=color, label=f"concurrency={concurrency}")
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Batch wall time / batch size (s)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Measured vLLM Service-Cost Profile — Prefill, Decode, and Visual Input",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Weights are platform- and concurrency-specific; use the measured curves and sensitivity analysis.",
        ha="center",
        fontsize=10,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    if not rows:
        parser.error("input contains no measurements")
    points = median_rows(rows)
    fits = fits_by_concurrency(points)
    summary = {
        "input": str(args.input.resolve()),
        "measurements": len(rows),
        "aggregated_points": points,
        "fits_by_concurrency": fits,
        "caution": (
            "The linear weights are descriptive local slopes, not universal GPU costs. "
            "Inspect R^2 and run weight sensitivity before using them for fairness accounting."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(points, args.output)
    print(json.dumps(summary["fits_by_concurrency"], indent=2))


if __name__ == "__main__":
    main()
