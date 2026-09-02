#!/usr/bin/env python3
"""Plot service-gap curves for preparation-only versus cross-stage fairness."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("analysis/figures/evaluation_mockups/11_prep_only_vs_cross_stage_service_gap")


def gap_curve(schedule: list[str]) -> np.ndarray:
    service_a = 0
    service_b = 0
    gaps = [0]
    for tenant in schedule:
        if tenant == "A":
            service_a += 1
        else:
            service_b += 1
        gaps.append(abs(service_a - service_b))
    return np.asarray(gaps, dtype=float)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Equal preparation cost: both policies alternate one CPU unit per tenant.
    cpu_schedule = ["A", "B"] * 8
    cpu_gap = gap_curve(cpu_schedule)

    # A request occupies four GPU units; B request occupies one GPU unit.
    # Preparation-only/native admission takes equal request counts.
    prep_only_gpu = list("AAAABAAAABAAAAB")
    # Cross-stage admission gives B four inexpensive requests per A request.
    cross_stage_gpu = list("AAAABBBBAAAABBBB")
    prep_gpu_gap = gap_curve(prep_only_gpu)
    cross_gpu_gap = gap_curve(cross_stage_gpu)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    cpu_time = np.arange(len(cpu_gap))
    axes[0].plot(
        cpu_time,
        cpu_gap,
        color="#E69F00",
        marker="o",
        linewidth=2.4,
        label="Preparation-only",
    )
    axes[0].plot(
        cpu_time,
        cpu_gap,
        color="#009E73",
        marker="v",
        linewidth=1.8,
        linestyle="--",
        label="Cross-stage max-min",
    )
    axes[0].set_title("(a) CPU preparation-service gap", fontweight="bold")
    axes[0].set_xlabel("Elapsed preparation time (illustrative units)")
    axes[0].set_ylabel("|A's CPU service - B's CPU service|\n(worker-time units; lower is fairer)")
    axes[0].set_ylim(-0.1, 4.5)

    prep_time = np.arange(len(prep_gpu_gap))
    cross_time = np.arange(len(cross_gpu_gap))
    axes[1].plot(
        prep_time,
        prep_gpu_gap,
        color="#E69F00",
        marker="o",
        linewidth=2.4,
        label="Preparation-only",
    )
    axes[1].plot(
        cross_time,
        cross_gpu_gap,
        color="#009E73",
        marker="v",
        linewidth=2.4,
        label="Cross-stage max-min",
    )
    axes[1].set_title("(b) GPU inference-service gap", fontweight="bold")
    axes[1].set_xlabel("Elapsed inference time (illustrative units)")
    axes[1].set_ylabel("|A's GPU service - B's GPU service|\n(service-time units; lower is fairer)")
    axes[1].set_ylim(-0.1, 10.5)

    for axis in axes:
        axis.set_xlim(0, 16)
        axis.set_xticks(np.arange(0, 17, 2))
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)

    fig.suptitle(
        "MOCKUP — Service Gap Between Two Continuously Backlogged Tenants\n"
        "A: GPU cost 4 units/request; B: GPU cost 1 unit/request; equal CPU cost",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=230, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
