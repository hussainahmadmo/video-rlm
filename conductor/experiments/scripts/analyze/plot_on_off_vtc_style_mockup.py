#!/usr/bin/env python3
"""Create a VTC-style ON/OFF mockup for cross-stage multimodal service."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def triangular_service(time_s: np.ndarray, period_s: float, peak: float) -> np.ndarray:
    """Illustrate a rolling-window average of periodic ON/OFF service."""
    phase = (time_s % period_s) / period_s
    triangle = 1.0 - np.abs(2.0 * phase - 1.0)
    return peak * triangle


def main() -> None:
    time_s = np.arange(0, 601, 10, dtype=float)
    blue = "#0072b2"
    orange = "#e69f00"

    # A's demand is below half of capacity and periodically turns off. B is
    # continuously backlogged and consumes all remaining capacity. Curves
    # illustrate rolling-window rates rather than instantaneous allocation.
    prep_a = triangular_service(time_s, 120.0, 0.25)
    prep_b = 1.0 - prep_a
    infer_a = triangular_service(time_s, 120.0, 0.25)
    infer_b = 1.0 - infer_a
    ttft_a = 2.0 + 0.8 * np.sin(np.pi * time_s / 120.0) ** 2
    ttft_b = 4.0 + 0.22 * time_s

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True)
    panels = [
        (
            "(a) Preparation service rate",
            prep_a,
            prep_b,
            "Received preparation service rate\n(normalized; 60 s window)",
        ),
        (
            "(b) Inference service rate",
            infer_a,
            infer_b,
            "Received inference service rate\n(normalized; 60 s window)",
        ),
        (
            "(c) End-to-end response time",
            ttft_a,
            ttft_b,
            "End-to-end TTFT (seconds)",
        ),
    ]

    for ax, (title, tenant_a, tenant_b, ylabel) in zip(axes, panels):
        ax.plot(
            time_s,
            tenant_a,
            marker="v",
            markevery=2,
            color=blue,
            linewidth=2.2,
            markersize=5,
            label="Tenant A (ON/OFF, underloaded)",
        )
        ax.plot(
            time_s,
            tenant_b,
            marker="s",
            markevery=2,
            color=orange,
            linewidth=2.2,
            markersize=4.5,
            label="Tenant B (always backlogged)",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 600)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.28)

    axes[0].set_ylim(0, 1.08)
    axes[1].set_ylim(0, 1.08)
    axes[0].text(585, 1.01, "total = 1", ha="right", va="bottom", fontsize=8)
    axes[1].text(585, 1.01, "total = 1", ha="right", va="bottom", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Multimodal ON/OFF workload under cross-stage max-min\n"
        "Identical 32-frame requests; illustrative values, not measurements",
        y=1.03,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))

    output = Path("analysis/figures/on_off_vtc_style_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
