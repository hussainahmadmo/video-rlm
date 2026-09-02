#!/usr/bin/env python3
"""Mock up cross-stage work conservation for two tenants."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ramp(time_s: np.ndarray, target: float, tau: float = 32.0) -> np.ndarray:
    return target * (1.0 - np.exp(-time_s / tau))


def main() -> None:
    time_s = np.arange(0, 601, 20, dtype=float)
    blue = "#0072b2"
    orange = "#e69f00"

    # Tenant A demands only 25% of each stage; overloaded Tenant B consumes
    # the remaining 75%. Values are illustrative and not measurements.
    prep_a = ramp(time_s, 0.25)
    prep_b = ramp(time_s, 0.75)
    infer_a = ramp(time_s, 0.25, tau=42.0)
    infer_b = ramp(time_s, 0.75, tau=42.0)
    ttft_a = 2.0 + 0.25 * np.sin(time_s / 55.0) ** 2
    ttft_b = 3.0 + 0.22 * time_s

    panels = [
        (
            "(a) Preparation service rate",
            prep_a,
            prep_b,
            "Fraction of preparation capacity",
            (0, 1.02),
        ),
        (
            "(b) Inference service rate",
            infer_a,
            infer_b,
            "Fraction of inference capacity",
            (0, 1.02),
        ),
        (
            "(c) End-to-end TTFT",
            ttft_a,
            ttft_b,
            "End-to-end TTFT (seconds)",
            (0, None),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4), sharex=True)
    for ax, (title, tenant_a, tenant_b, ylabel, ylim) in zip(axes, panels):
        ax.plot(
            time_s,
            tenant_a,
            "o-",
            markevery=3,
            color=blue,
            linewidth=2.3,
            label="Tenant A (underloaded)",
        )
        ax.plot(
            time_s,
            tenant_b,
            "s-",
            markevery=3,
            color=orange,
            linewidth=2.3,
            label="Tenant B (overloaded)",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 600)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)

    axes[0].axhline(0.5, color="#777777", linestyle=":", linewidth=1.4)
    axes[1].axhline(0.5, color="#777777", linestyle=":", linewidth=1.4)
    axes[0].text(590, 0.515, "equal share", ha="right", va="bottom", fontsize=8)
    axes[1].text(590, 0.515, "equal share", ha="right", va="bottom", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Cross-stage max-min is work-conserving\n"
        "Identical 32-frame requests; A demands 25%, B remains backlogged",
        y=1.04,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))

    output = Path("analysis/figures/work_conservation_two_tenant_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
