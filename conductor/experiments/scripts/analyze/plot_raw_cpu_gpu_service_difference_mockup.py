#!/usr/bin/env python3
"""Create unnormalized CPU/GPU cumulative service-difference mockups."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("analysis/figures/raw_cpu_gpu_service_difference_mockup")
TIME = np.arange(0, 601, 50, dtype=float)

STYLES = {
    "FCFS": dict(color="#D55E00", marker="s", linestyle="-"),
    "Tenant round-robin": dict(color="#CC79A7", marker="D", linestyle="--"),
    "Preparation-only": dict(color="#E69F00", marker="o", linestyle="-."),
    "Inference-only / VTC-style": dict(color="#56B4E9", marker="^", linestyle=":"),
    "Cross-stage max-min": dict(color="#009E73", marker="v", linestyle="-"),
}


def rising(endpoint: float, exponent: float = 1.18) -> np.ndarray:
    return endpoint * (TIME / TIME[-1]) ** exponent


def bounded(level: float, amplitude: float, phase: float = 0.0) -> np.ndarray:
    values = level + amplitude * np.sin(TIME / 42.0 + phase)
    values[0] = 0.0
    return values


# Illustrative expected values, not measured experimental results.
CPU = {
    "FCFS": rising(1500, 1.22),
    "Tenant round-robin": rising(850, 1.15),
    "Preparation-only": bounded(32, 8, 0.8),
    "Inference-only / VTC-style": rising(1250, 1.20),
    "Cross-stage max-min": bounded(25, 6, 0.1),
}

GPU = {
    "FCFS": rising(350, 1.20),
    "Tenant round-robin": rising(220, 1.14),
    "Preparation-only": rising(310, 1.18),
    "Inference-only / VTC-style": bounded(11, 3, 0.7),
    "Cross-stage max-min": bounded(8, 2, 0.0),
}


def plot_panel(
    axis,
    values: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    for label, series in values.items():
        axis.plot(
            TIME,
            series,
            label=label,
            linewidth=2.7,
            markersize=5.5,
            **STYLES[label],
        )
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0, 600)
    axis.set_ylim(*ylim)
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.72, wspace=0.18)

    plot_panel(
        axes[0],
        CPU,
        "(a) CPU video-preparation stage",
        "Service gap: |Tenant A - Tenant B|\n(CPU worker-seconds; lower is fairer)",
        (0, 1650),
    )
    plot_panel(
        axes[1],
        GPU,
        "(b) GPU inference stage",
        "Service gap: |Tenant A - Tenant B|\n(GPU service-seconds; lower is fairer)",
        (0, 385),
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
        "MOCKUP — Two-Tenant Service Gap Over Time\n"
        "Tenant A and Tenant B are both backlogged; CPU and GPU units are not comparable",
        fontsize=16,
        fontweight="bold",
    )

    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
