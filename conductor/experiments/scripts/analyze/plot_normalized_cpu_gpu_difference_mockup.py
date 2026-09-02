#!/usr/bin/env python3
"""Create a two-panel CPU/GPU normalized service-difference mockup."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("analysis/figures/normalized_cpu_gpu_service_difference_mockup")
TIME = np.arange(0, 601, 50, dtype=float)

STYLES = {
    "FCFS": dict(color="#D55E00", marker="s", linestyle="-"),
    "Tenant round-robin": dict(color="#CC79A7", marker="D", linestyle="--"),
    "Preparation-only": dict(color="#E69F00", marker="o", linestyle="-."),
    "Inference-only / VTC-style": dict(color="#56B4E9", marker="^", linestyle=":"),
    "Cross-stage max-min": dict(color="#009E73", marker="v", linestyle="-"),
}


def rising(endpoint: float, exponent: float = 1.35) -> np.ndarray:
    return endpoint * (TIME / TIME[-1]) ** exponent


def bounded(level: float, phase: float = 0.0) -> np.ndarray:
    values = level + 0.006 * np.sin(TIME / 42.0 + phase)
    values[0] = 0.0
    return values


# Illustrative expected trends, not measured data.
CPU = {
    "FCFS": rising(0.64, 1.38),
    "Tenant round-robin": rising(0.40, 1.25),
    "Preparation-only": bounded(0.030, 0.8),
    "Inference-only / VTC-style": rising(0.56, 1.32),
    "Cross-stage max-min": bounded(0.025, 0.1),
}

GPU = {
    "FCFS": rising(0.58, 1.34),
    "Tenant round-robin": rising(0.36, 1.22),
    "Preparation-only": rising(0.52, 1.30),
    "Inference-only / VTC-style": bounded(0.030, 0.7),
    "Cross-stage max-min": bounded(0.024, 0.0),
}


def plot_panel(axis, values: dict[str, np.ndarray], title: str) -> None:
    for label, series in values.items():
        axis.plot(
            TIME,
            series,
            label=label,
            linewidth=2.7,
            markersize=5.5,
            markevery=1,
            **STYLES[label],
        )
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel("Time (seconds)")
    axis.set_xlim(0, 600)
    axis.set_ylim(0, 0.70)
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.0),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.72, wspace=0.05)
    plot_panel(axes[0], CPU, "(a) CPU video-preparation stage")
    plot_panel(axes[1], GPU, "(b) GPU inference-admission stage")
    axes[0].set_ylabel(
        "Normalized cumulative tenant-service difference\n(lower is fairer)"
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
    )
    fig.suptitle(
        "MOCKUP — Stage-Specific Multimodal Fairness Over Time\n"
        "Illustrative expected trends; not measured experimental results",
        fontsize=16,
        fontweight="bold",
    )

    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
