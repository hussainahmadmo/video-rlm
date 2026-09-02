#!/usr/bin/env python3
"""Create an illustrative per-tenant service dot-plot mockup."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    policies = ["FCFS", "Preparation-only", "Inference-only", "Cross-stage\nmax-min"]
    tenants = ["Tenant A", "Tenant B"]
    colors = ["#0072b2", "#e69f00"]

    # Illustrative values only; not measurements.
    preparation = np.array(
        [
            [68, 28],
            [38, 37],
            [62, 30],
            [37, 38],
        ],
        dtype=float,
    )
    inference = np.array(
        [
            [62, 28],
            [57, 31],
            [35, 36],
            [36, 37],
        ],
        dtype=float,
    )

    panels = [
        ("(a) Preparation service by tenant", preparation, "Preparation service (worker-seconds)"),
        ("(b) Inference service by tenant", inference, "Inference service (service units)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), sharex=True)
    x = np.arange(len(policies))
    offsets = [-0.08, 0.08]

    for ax, (title, values, ylabel) in zip(axes, panels):
        for policy_index in range(len(policies)):
            low = float(np.min(values[policy_index]))
            high = float(np.max(values[policy_index]))
            ax.vlines(policy_index, low, high, color="#777777", linewidth=3, alpha=0.55)
            ax.text(
                policy_index,
                high + 3.0,
                f"difference = {high - low:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )
        for tenant_index, tenant in enumerate(tenants):
            ax.scatter(
                x + offsets[tenant_index],
                values[:, tenant_index],
                s=85,
                color=colors[tenant_index],
                edgecolor="white",
                linewidth=0.8,
                label=tenant,
                zorder=3,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, policies)
        ax.set_ylim(0, 80)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Per-tenant service under matched workloads\n"
        "Illustrative values only; not measured results",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))

    output = Path("analysis/figures/tenant_service_dot_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
