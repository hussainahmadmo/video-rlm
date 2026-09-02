#!/usr/bin/env python3
"""Mock up per-tenant cumulative service over time at both stages."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def cumulative_curves(time_s: np.ndarray, slopes: tuple[float, ...], fair: bool) -> np.ndarray:
    curves = []
    phases = np.linspace(0.0, 3.4, len(slopes))
    for slope, phase in zip(slopes, phases):
        if fair:
            # Small bounded offsets illustrate request-granularity scheduling.
            offset = 1.6 * np.sin(time_s / 45.0 + phase)
            values = slope * time_s + offset - offset[0]
        else:
            values = slope * time_s
        curves.append(np.maximum.accumulate(np.maximum(values, 0.0)))
    return np.asarray(curves)


def main() -> None:
    time_s = np.arange(0, 601, 20, dtype=float)
    policies = ["FCFS", "Preparation-only", "Inference-only", "Cross-stage max-min"]
    tenants = ["Tenant A", "Tenant B"]
    colors = ["#0072b2", "#e69f00"]

    prep = {
        "FCFS": cumulative_curves(time_s, (0.120, 0.060), False),
        "Preparation-only": cumulative_curves(time_s, (0.090, 0.090), True),
        "Inference-only": cumulative_curves(time_s, (0.115, 0.065), False),
        "Cross-stage max-min": cumulative_curves(time_s, (0.090, 0.090), True),
    }
    infer = {
        "FCFS": cumulative_curves(time_s, (0.105, 0.055), False),
        "Preparation-only": cumulative_curves(time_s, (0.100, 0.060), False),
        "Inference-only": cumulative_curves(time_s, (0.080, 0.080), True),
        "Cross-stage max-min": cumulative_curves(time_s, (0.080, 0.080), True),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.0), sharex=True, sharey="row")
    for row, (stage, data, ylabel) in enumerate(
        [
            ("Preparation", prep, "Cumulative preparation service\n(worker-seconds)"),
            ("Inference", infer, "Cumulative inference service\n(service units)"),
        ]
    ):
        for col, policy in enumerate(policies):
            ax = axes[row, col]
            for tenant_index, tenant in enumerate(tenants):
                ax.plot(
                    time_s,
                    data[policy][tenant_index],
                    marker="o",
                    markevery=3,
                    markersize=4.5,
                    linewidth=2.0,
                    color=colors[tenant_index],
                    label=tenant,
                )
            if row == 0:
                ax.set_title(policy, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Time (seconds)")
            ax.grid(alpha=0.25)
            ax.text(
                0.03,
                0.92,
                stage,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="#444444",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Per-tenant cumulative service over time\n"
        "Illustrative trends only; not measured results",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))

    output = Path("analysis/figures/tenant_service_over_time_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
