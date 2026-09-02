#!/usr/bin/env python3
"""Create an illustrative mockup for stage-wise cumulative service difference."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    time_s = np.linspace(0, 600, 61)

    # Illustrative expectations only. These values are not measurements.
    bounded_low = 2.5 + 1.2 * np.sin(time_s / 42.0) ** 2
    bounded_mid = 5.0 + 2.0 * np.sin(time_s / 55.0) ** 2
    growing_fast = 0.065 * time_s + 0.00008 * time_s**2
    growing_mid = 0.040 * time_s + 0.00005 * time_s**2

    panels = [
        (
            "(a) Preparation service difference",
            {
                "FCFS": growing_fast,
                "Preparation-only": bounded_mid,
                "Inference-only": growing_mid,
                "Cross-stage max-min": bounded_low,
            },
            "Preparation service difference\n(worker-seconds; lower is fairer)",
        ),
        (
            "(b) Inference service difference",
            {
                "FCFS": growing_fast * 0.90,
                "Preparation-only": growing_mid * 1.05,
                "Inference-only": bounded_mid,
                "Cross-stage max-min": bounded_low,
            },
            "Inference service difference\n(service units; lower is fairer)",
        ),
    ]

    colors = {
        "FCFS": "#d55e00",
        "Preparation-only": "#e69f00",
        "Inference-only": "#56b4e9",
        "Cross-stage max-min": "#009e73",
    }
    styles = {
        "FCFS": "-",
        "Preparation-only": "--",
        "Inference-only": "-.",
        "Cross-stage max-min": "-",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    for ax, (title, curves, ylabel) in zip(axes, panels):
        for policy, values in curves.items():
            ax.plot(
                time_s,
                values,
                label=policy,
                color=colors[policy],
                linestyle=styles[policy],
                linewidth=2.5,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 600)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Expected stage-wise cumulative service difference\n"
        "Illustrative trends only; not measured results",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))

    output = Path("analysis/figures/stagewise_service_difference_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
