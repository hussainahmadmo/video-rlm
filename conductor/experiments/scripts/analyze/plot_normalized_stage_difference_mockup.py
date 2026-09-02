#!/usr/bin/env python3
"""Plot an illustrative normalized cross-stage service-imbalance comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("analysis/figures/normalized_stage_service_difference_mockup")

POLICIES = [
    "FCFS",
    "Tenant\nround-robin",
    "Preparation-only\nmax-min",
    "Inference-only\nfairness",
    "Cross-stage\nmax-min",
]

# Illustrative expected values, not measurements. Each value is the tenant
# service gap divided by the service capacity available at that stage during
# the measurement interval.
PREPARATION = np.array([45, 30, 8, 42, 9], dtype=float)
INFERENCE = np.array([60, 35, 48, 7, 8], dtype=float)
WORST_STAGE = np.maximum(PREPARATION, INFERENCE)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(POLICIES), dtype=float)
    width = 0.24

    fig, axis = plt.subplots(figsize=(10.8, 5.3), constrained_layout=True)
    bars = [
        axis.bar(
            x - width,
            PREPARATION,
            width,
            label="Preparation difference",
            color="#56B4E9",
        ),
        axis.bar(
            x,
            INFERENCE,
            width,
            label="Inference difference",
            color="#E69F00",
        ),
        axis.bar(
            x + width,
            WORST_STAGE,
            width,
            label="Worst-stage difference",
            color="#CC79A7",
            hatch="//",
        ),
    ]

    for group in bars:
        axis.bar_label(group, fmt="%.0f%%", padding=3, fontsize=9)

    axis.set_xticks(x, POLICIES)
    axis.set_ylabel("Normalized tenant-service difference (%)\nLower is fairer")
    axis.set_ylim(0, 70)
    axis.set_title(
        "Expected Normalized Service Difference Across Both Stages\n"
        "Illustrative mockup — not measured experimental results",
        fontweight="bold",
        pad=12,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
    )

    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
