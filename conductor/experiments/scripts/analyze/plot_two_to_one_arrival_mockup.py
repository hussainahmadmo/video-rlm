#!/usr/bin/env python3
"""Mock up two-tenant 2:1 arrivals and their expected service allocation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    rounds = np.arange(0, 13)
    time_s = rounds * 5
    submitted_a = rounds * 2
    submitted_b = rounds

    # Illustrative service values only; all requests have the same cost.
    fcfs_a = rounds * 4.0
    fcfs_b = rounds * 2.0
    maxmin_a = rounds * 3.0 + 0.6 * (rounds % 2)
    maxmin_b = rounds * 3.0 - 0.6 * (rounds % 2)

    blue = "#0072b2"
    orange = "#e69f00"
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharex=True)

    panels = [
        (
            "(a) Offered requests (2:1)",
            submitted_a,
            submitted_b,
            "Cumulative requests submitted",
        ),
        (
            "(b) FCFS",
            fcfs_a,
            fcfs_b,
            "Cumulative service received",
        ),
        (
            "(c) Cross-stage max-min",
            maxmin_a,
            maxmin_b,
            "Cumulative service received",
        ),
    ]

    for ax, (title, tenant_a, tenant_b, ylabel) in zip(axes, panels):
        ax.plot(time_s, tenant_a, "o-", color=blue, linewidth=2.4, label="Tenant A")
        ax.plot(time_s, tenant_b, "s-", color=orange, linewidth=2.4, label="Tenant B")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 60)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Two tenants, identical 32-frame requests, sustained high load\n"
        "Tenant A submits two requests for every one from Tenant B",
        y=1.04,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))

    output = Path("analysis/figures/two_to_one_arrival_service_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
