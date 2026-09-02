#!/usr/bin/env python3
"""Mock up unnormalized preparation and inference service differences."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bounded(time_s: np.ndarray, scale: float, offset: float = 1.0) -> np.ndarray:
    warmup = 1.0 - np.exp(-time_s / 25.0)
    return warmup * scale * (
        0.025 * offset + 0.012 * np.sin(time_s / 38.0) ** 2
    )


def growing(time_s: np.ndarray, scale: float, factor: float) -> np.ndarray:
    return factor * scale * (0.00065 * time_s + 0.0000007 * time_s**2)


def main() -> None:
    time_s = np.arange(0, 601, 10, dtype=float)
    prep = {
        "FCFS": growing(time_s, 100.0, 1.0),
        "Tenant round-robin": growing(time_s, 100.0, 0.62),
        "Preparation-only": bounded(time_s, 100.0, 1.35),
        "Inference-only / VTC": growing(time_s, 100.0, 0.88),
        "Cross-stage max-min": bounded(time_s, 100.0),
    }
    infer = {
        "FCFS": growing(time_s, 80.0, 1.0),
        "Tenant round-robin": growing(time_s, 80.0, 0.62),
        "Preparation-only": growing(time_s, 80.0, 0.88),
        "Inference-only / VTC": bounded(time_s, 80.0, 1.35),
        "Cross-stage max-min": bounded(time_s, 80.0),
    }
    styles = {
        "FCFS": ("#d55e00", "s", "-"),
        "Tenant round-robin": ("#cc79a7", "D", "--"),
        "Preparation-only": ("#e69f00", "o", "-."),
        "Inference-only / VTC": ("#56b4e9", "^", ":"),
        "Cross-stage max-min": ("#009e73", "v", "-"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharex=True)
    panels = [
        (
            axes[0],
            prep,
            "(a) Preparation service difference",
            "Absolute difference (worker-seconds)",
        ),
        (
            axes[1],
            infer,
            "(b) Inference service difference",
            "Absolute difference (inference-service units)",
        ),
    ]
    for ax, data, title, ylabel in panels:
        for policy, values in data.items():
            color, marker, linestyle = styles[policy]
            ax.plot(
                time_s,
                values,
                color=color,
                marker=marker,
                markevery=6,
                linestyle=linestyle,
                linewidth=2.2,
                markersize=4.5,
                label=policy,
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
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Unnormalized stage-wise service difference\n"
        "Both tenants backlogged; heterogeneous video costs; lower is fairer",
        y=1.03,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.91))

    output = Path("analysis/figures/unnormalized_stage_service_difference_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
