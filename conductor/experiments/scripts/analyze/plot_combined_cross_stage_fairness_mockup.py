#!/usr/bin/env python3
"""Mock up a normalized combined cross-stage fairness metric."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bounded(time_s: np.ndarray, offset: float = 1.0) -> np.ndarray:
    warmup = 1.0 - np.exp(-time_s / 25.0)
    return warmup * (0.025 * offset + 0.012 * np.sin(time_s / 38.0) ** 2)


def growing(time_s: np.ndarray, factor: float) -> np.ndarray:
    return factor * (0.00065 * time_s + 0.0000007 * time_s**2)


def main() -> None:
    time_s = np.arange(0, 601, 10, dtype=float)

    # Each array is already normalized by its stage's available service.
    prep = {
        "FCFS": growing(time_s, 1.0),
        "Tenant round-robin": growing(time_s, 0.62),
        "Preparation-only": bounded(time_s, 1.35),
        "Inference-only / VTC": growing(time_s, 0.88),
        "Cross-stage max-min": bounded(time_s),
    }
    infer = {
        "FCFS": growing(time_s, 1.0),
        "Tenant round-robin": growing(time_s, 0.62),
        "Preparation-only": growing(time_s, 0.88),
        "Inference-only / VTC": bounded(time_s, 1.35),
        "Cross-stage max-min": bounded(time_s),
    }
    combined = {
        policy: np.maximum(prep[policy], infer[policy]) for policy in prep
    }

    styles = {
        "FCFS": ("#d55e00", "s", "-"),
        "Tenant round-robin": ("#cc79a7", "D", "--"),
        "Preparation-only": ("#e69f00", "o", "-."),
        "Inference-only / VTC": ("#56b4e9", "^", ":"),
        "Cross-stage max-min": ("#009e73", "v", "-"),
    }

    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    for policy, values in combined.items():
        color, marker, linestyle = styles[policy]
        ax.plot(
            time_s,
            values,
            color=color,
            marker=marker,
            markevery=5,
            linestyle=linestyle,
            linewidth=2.4,
            markersize=5,
            label=policy,
        )

    ax.set_title(
        "MOCKUP — Combined cross-stage fairness\n"
        "Worst normalized service difference across preparation and inference",
        fontweight="bold",
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Worst-stage normalized service difference\n(lower is fairer)")
    ax.set_xlim(0, 600)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()

    output = Path("analysis/figures/combined_cross_stage_fairness_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
