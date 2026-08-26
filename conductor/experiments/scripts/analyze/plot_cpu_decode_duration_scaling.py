#!/usr/bin/env python3
"""Plot media-preparation service time as video duration increases."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CPU_BACKENDS = ("seek_cpu", "batch_cpu")
TITLES = {
    "seek_cpu": "Per-frame CPU seeking",
    "batch_cpu": "One-pass sequential CPU decode",
    "batch_nvdec": "Batch NVDEC (preliminary)",
}
FRAME_COLORS = {
    8: "#56B4E9",
    16: "#009E73",
    32: "#F0E442",
    64: "#E69F00",
    128: "#D55E00",
}


def load_medians(root: Path, backend: str) -> dict[tuple[str, int], tuple[float, float]]:
    """Return (duration, median service) for each question/frame-count pair."""
    samples: dict[tuple[str, int], list[float]] = defaultdict(list)
    durations: dict[tuple[str, int], float] = {}
    for path in root.glob(f"*/{backend}_*/results.jsonl"):
        if not (path.parent / "summary.json").is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") or row.get("prep_service_s") is None:
                continue
            key = (str(row["qid"]), int(row["frame_count"]))
            samples[key].append(float(row["prep_service_s"]))
            durations[key] = float(row["duration_s"])
    return {
        key: (durations[key], float(np.median(values)))
        for key, values in samples.items()
    }


def fit_duration_and_frames(
    points: list[tuple[float, int, float]],
) -> tuple[np.ndarray, float]:
    """Fit log10(service) ~ log10(duration) + log10(frame count)."""
    design = np.asarray([
        [1.0, math.log10(duration), math.log10(frames)]
        for duration, frames, _ in points
    ])
    target = np.asarray([math.log10(service) for _, _, service in points])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    degrees = len(target) - design.shape[1]
    variance = float(residual @ residual) / degrees
    covariance = variance * np.linalg.inv(design.T @ design)
    duration_se = math.sqrt(float(covariance[1, 1]))
    return beta, 1.96 * duration_se


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument(
        "--nvdec-suite",
        type=Path,
        help="Optional partial NVDEC suite to add as a preliminary third panel.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    values = {backend: load_medians(args.suite, backend) for backend in CPU_BACKENDS}
    paired_keys = sorted(set(values["seek_cpu"]) & set(values["batch_cpu"]))
    if not paired_keys:
        raise SystemExit("No paired successful seek_cpu/batch_cpu results found")

    all_points: dict[str, list[tuple[float, int, float]]] = {}
    fits: dict[str, tuple[np.ndarray, float]] = {}
    for backend in CPU_BACKENDS:
        all_points[backend] = [
            (values[backend][key][0], key[1], values[backend][key][1])
            for key in paired_keys
        ]
        fits[backend] = fit_duration_and_frames(all_points[backend])

    plotted_backends = list(CPU_BACKENDS)
    if args.nvdec_suite is not None:
        nvdec_values = load_medians(args.nvdec_suite, "batch_nvdec")
        if not nvdec_values:
            raise SystemExit("No successful completed batch_nvdec results found")
        all_points["batch_nvdec"] = [
            (duration, key[1], service)
            for key, (duration, service) in sorted(nvdec_values.items())
        ]
        fits["batch_nvdec"] = fit_duration_and_frames(all_points["batch_nvdec"])
        plotted_backends.append("batch_nvdec")

    x_min = min(point[0] for points in all_points.values() for point in points)
    x_max = max(point[0] for points in all_points.values() for point in points)
    y_min = min(point[2] for points in all_points.values() for point in points)
    y_max = max(point[2] for points in all_points.values() for point in points)
    duration_grid = np.logspace(math.log10(x_min), math.log10(x_max), 200)

    fig, axes = plt.subplots(
        1,
        len(plotted_backends),
        figsize=(6.5 * len(plotted_backends), 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for ax, backend in zip(axes, plotted_backends, strict=True):
        points = all_points[backend]
        for frames, color in FRAME_COLORS.items():
            subset = [point for point in points if point[1] == frames]
            ax.scatter(
                [point[0] for point in subset],
                [point[2] for point in subset],
                s=24,
                alpha=0.48,
                color=color,
                edgecolors="none",
                label=f"{frames} frames",
            )

        beta, duration_ci = fits[backend]
        reference_frames = 32
        predicted_log = (
            beta[0]
            + beta[1] * np.log10(duration_grid)
            + beta[2] * math.log10(reference_frames)
        )
        ax.plot(
            duration_grid,
            10 ** predicted_log,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label="Fit at 32 frames",
        )
        ax.text(
            0.04,
            0.95,
            (
                f"Duration exponent: {beta[1]:.2f} ± {duration_ci:.2f}\n"
                f"Frame-count exponent: {beta[2]:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )
        if backend == "batch_nvdec":
            ax.text(
                0.04,
                0.05,
                (
                    f"{len(points)} successful cases from one completed trace\n"
                    "Timeouts and aborted runs excluded"
                ),
                transform=ax.transAxes,
                va="bottom",
                fontsize=9.5,
                color="#9C2F1B",
                bbox={"facecolor": "#FFF4EE", "alpha": 0.92, "edgecolor": "#D98265"},
            )
        ax.set_title(TITLES[backend])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_min * 0.8, x_max * 1.25)
        ax.set_ylim(y_min * 0.65, y_max * 1.5)
        ax.set_xlabel("Video duration (seconds, log scale)")
        ax.grid(which="both", alpha=0.2)

    axes[0].set_ylabel("Median preparation service time (seconds, log scale)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=6,
        frameon=True,
    )
    fig.suptitle(
        "Decode scaling depends on the media-access strategy, not just the hardware\n"
        + (
            f"CPU panels: {len(paired_keys)} paired successful cases; "
            f"NVDEC panel: {len(all_points['batch_nvdec'])} successful cases (preliminary)"
            if "batch_nvdec" in all_points
            else f"{len(paired_keys)} paired successful question/frame-count cases"
        )
        + "; service times are medians across repeats",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.75))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
