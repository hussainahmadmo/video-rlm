#!/usr/bin/env python3
"""Mock up fair service with large and small video requests."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    time_s = np.arange(0, 601, 10, dtype=float)
    blue = "#0072b2"
    orange = "#e69f00"

    # Both tenants are continuously backlogged. Cross-stage max-min provides
    # approximately equal service, with bounded request-granularity variation.
    service_large = 0.5 + 0.055 * np.sin(2 * np.pi * time_s / 120.0)
    service_small = 1.0 - service_large

    # Illustrative costs: a 128-frame request costs four times an 32-frame
    # request, so equal service yields four times as many small completions.
    completed_large = time_s / 20.0
    completed_small = time_s / 5.0

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharex=True)

    axes[0].plot(
        time_s,
        service_large,
        "o-",
        markevery=3,
        color=blue,
        linewidth=2.3,
        markersize=4.5,
        label="Tenant A: large video (128 frames)",
    )
    axes[0].plot(
        time_s,
        service_small,
        "s-",
        markevery=3,
        color=orange,
        linewidth=2.3,
        markersize=4.5,
        label="Tenant B: small video (32 frames)",
    )
    axes[0].axhline(0.5, color="#777777", linestyle=":", linewidth=1.5)
    axes[0].set_title("(a) Received preparation service rate", fontweight="bold")
    axes[0].set_ylabel("Fraction of preparation capacity\n(lower gap is fairer)")
    axes[0].set_ylim(0, 1.02)

    axes[1].plot(
        time_s,
        completed_large,
        "o-",
        markevery=3,
        color=blue,
        linewidth=2.3,
        markersize=4.5,
    )
    axes[1].plot(
        time_s,
        completed_small,
        "s-",
        markevery=3,
        color=orange,
        linewidth=2.3,
        markersize=4.5,
    )
    axes[1].set_title("(b) Completed requests", fontweight="bold")
    axes[1].set_ylabel("Cumulative completed requests")

    for ax in axes:
        ax.set_xlabel("Time (seconds)")
        ax.set_xlim(0, 600)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Heterogeneous video costs under cross-stage max-min\n"
        "Equal service shares do not imply equal request counts",
        y=1.03,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))

    output = Path("analysis/figures/heterogeneous_request_service_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
