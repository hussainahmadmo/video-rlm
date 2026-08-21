#!/usr/bin/env python3
"""Plot resource-matched native-video versus prepared-frame concurrency results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def metric(summary: dict, name: str) -> float:
    value = summary.get(name)
    if value is None:
        raise KeyError(f"{name} missing from summary")
    return float(value)


def grouped_bars(ax, x, native, prepared, ylabel, title, log=False):
    width = 0.35
    ax.bar(x - width / 2, native, width, label="Native TorchCodec", color="#d95f02")
    ax.bar(x + width / 2, prepared, width, label="External prepared pipeline", color="#1b9e77")
    ax.set_xticks(x, ["C=2", "C=4", "C=8"])
    ax.set_xlabel("Concurrency / workers per stage")
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    if log:
        ax.set_yscale("log")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)


def save(fig, output: Path):
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/workspace/video-rlm/conductor/experiments/large_sweeps/concurrency_scaling_t1_20260820_055518"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/video-rlm/analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    concurrencies = [2, 4, 8]
    # C=4 was run as an earlier resource-matched pilot; C=2/8 are in the
    # subsequent concurrency sweep. All use 1 QPS, 8 frames, and one decoder
    # thread per worker.
    c4_root = Path("/workspace/video-rlm/conductor/experiments/large_sweeps/resource_matched_t1_20260820_053331")
    native_paths = {
        2: args.run_root / "native_c2" / "summary.json",
        4: c4_root / "native_torchcodec_c4_t1" / "summary.json",
        8: args.run_root / "native_c8" / "summary.json",
    }
    prepared_paths = {
        2: args.run_root / "prepared_c2" / "summary.json",
        4: c4_root / "prepared_uniform8_c4_t1" / "summary.json",
        8: args.run_root / "prepared_c8" / "summary.json",
    }
    native = [load_summary(native_paths[c]) for c in concurrencies]
    prepared = [load_summary(prepared_paths[c]) for c in concurrencies]
    x = np.arange(len(concurrencies))
    subtitle = "249 questions · 1 QPS offered load · 8 frames · one decode thread"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    grouped_bars(ax, x,
                 [metric(s, "mean_end_to_end_delay_s") for s in native],
                 [metric(s, "mean_end_to_end_delay_s") for s in prepared],
                 "Mean end-to-end delay (s)", "Mean E2E latency")
    ax.text(0.5, -0.21, subtitle, ha="center", transform=ax.transAxes, fontsize=9)
    ax.legend(frameon=False)
    save(fig, args.output_dir / "concurrency_scaling_mean_e2e.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    grouped_bars(ax, x,
                 [metric(s, "p95_end_to_end_delay_s") for s in native],
                 [metric(s, "p95_end_to_end_delay_s") for s in prepared],
                 "p95 end-to-end delay (s)", "Tail E2E latency (p95)")
    ax.text(0.5, -0.21, subtitle, ha="center", transform=ax.transAxes, fontsize=9)
    ax.legend(frameon=False)
    save(fig, args.output_dir / "concurrency_scaling_p95_e2e.png")

    # Paper-style tail comparison with directly annotated measured reductions.
    n_p95 = [metric(s, "p95_end_to_end_delay_s") for s in native]
    p_p95 = [metric(s, "p95_end_to_end_delay_s") for s in prepared]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.2), sharey=True)
    for ax, c, n_value, p_value in zip(axes, concurrencies, n_p95, p_p95):
        bars = ax.bar([0, 1], [n_value, p_value], color=["#d95f02", "#1b9e77"], width=0.62)
        ax.set_xticks([0, 1], ["Native\nTorchCodec", "External\npipeline"])
        ax.set_title(f"C={c}", weight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.1f s", padding=3, fontsize=9)
        ratio = n_value / p_value
        annotation_y = max(n_value, p_value) * 1.14
        ax.annotate("", xy=(1, p_value * 1.03), xytext=(0, n_value * 1.03),
                    arrowprops={"arrowstyle": "<->", "color": "#16803a", "lw": 1.7})
        ax.text(0.5, annotation_y, f"{ratio:.2f}× lower", color="#16803a",
                ha="center", va="bottom", fontsize=11, weight="bold")
    axes[0].set_ylabel("p95 end-to-end delay (s)")
    fig.suptitle("Tail-latency reduction from explicit video preparation", y=1.03, weight="bold")
    fig.text(0.5, -0.03, subtitle, ha="center", fontsize=9)
    save(fig, args.output_dir / "concurrency_scaling_p95_annotated.png")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, c, n, p in zip(axes, concurrencies, native, prepared):
        names = ["p50", "Mean", "p95"]
        n_values = [metric(n, "p50_end_to_end_delay_s"), metric(n, "mean_end_to_end_delay_s"), metric(n, "p95_end_to_end_delay_s")]
        p_values = [metric(p, "p50_end_to_end_delay_s"), metric(p, "mean_end_to_end_delay_s"), metric(p, "p95_end_to_end_delay_s")]
        ix = np.arange(3)
        width = 0.36
        ax.bar(ix - width / 2, n_values, width, color="#d95f02", label="Native TorchCodec")
        ax.bar(ix + width / 2, p_values, width, color="#1b9e77", label="External prepared pipeline")
        ax.set_xticks(ix, names)
        ax.set_title(f"C={c}", weight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("End-to-end delay (s)")
    axes[1].set_xlabel("Latency statistic")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
    fig.suptitle(f"E2E latency profile by concurrency\n{subtitle}", y=1.08, weight="bold")
    save(fig, args.output_dir / "concurrency_scaling_latency_profile.png")


if __name__ == "__main__":
    main()
