#!/usr/bin/env python3
"""Visualize a workload that separates preparation-only from cross-stage fairness."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUTPUT = Path("analysis/figures/evaluation_mockups/10_prep_only_vs_cross_stage_workload")
COLOR_A = "#D55E00"
COLOR_B = "#0072B2"
CPU_COLOR = "#8A8A8A"


def request_box(axis, x, y, width, height, tenant, stage, color) -> None:
    axis.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="white", linewidth=1.2))
    axis.text(x + width / 2, y + height / 2, tenant, ha="center", va="center", color="white", fontweight="bold", fontsize=9)


def timeline(axis, y, jobs, label, total_width=16) -> None:
    axis.text(-0.35, y + 0.35, label, ha="right", va="center", fontweight="bold", fontsize=10)
    cursor = 0.0
    for tenant, width in jobs:
        request_box(axis, cursor, y, width, 0.7, tenant, label, COLOR_A if tenant == "A" else COLOR_B)
        cursor += width
    axis.plot([0, total_width], [y - 0.08, y - 0.08], color="black", linewidth=0.8)
    axis.text(total_width, y - 0.25, "time →", ha="right", va="top", fontsize=9)


def system_panel(axis, title, gpu_jobs, gpu_summary) -> None:
    axis.set_xlim(-2.9, 16.5)
    axis.set_ylim(-0.2, 3.2)
    axis.axis("off")
    axis.set_title(title, fontweight="bold", fontsize=13, pad=8)

    cpu_jobs = [("A", 1), ("B", 1)] * 8
    timeline(axis, 2.0, cpu_jobs, "CPU preparation")
    timeline(axis, 0.65, gpu_jobs, "GPU inference")

    axis.text(
        0,
        3.02,
        "CPU received service: A = 8 units, B = 8 units",
        fontsize=10,
        color=CPU_COLOR,
    )
    axis.text(8.0, 0.25, gpu_summary, ha="center", va="top", fontsize=10, fontweight="bold")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.5, 9.0))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.0, 2.0], hspace=0.42)

    setup = fig.add_subplot(grid[0])
    setup.axis("off")
    setup.add_patch(FancyBboxPatch((0.02, 0.13), 0.96, 0.75, boxstyle="round,pad=0.02", facecolor="#F6F6F6", edgecolor="#BBBBBB"))
    setup.text(0.05, 0.70, "Workload", fontweight="bold", fontsize=13)
    setup.text(0.05, 0.46, "Both Tenant A and Tenant B are continuously backlogged.", fontsize=11)
    setup.text(0.05, 0.24, "Every request has the same CPU preparation cost: 1 worker-time unit.", fontsize=11)
    setup.text(0.55, 0.56, "Tenant A request", color=COLOR_A, fontweight="bold", fontsize=11)
    setup.text(0.70, 0.56, "GPU cost = 4 units (long inference)", fontsize=11)
    setup.text(0.55, 0.30, "Tenant B request", color=COLOR_B, fontweight="bold", fontsize=11)
    setup.text(0.70, 0.30, "GPU cost = 1 unit (short inference)", fontsize=11)

    prep_only = fig.add_subplot(grid[1])
    # FCFS admits equal request counts. Because A's requests are four times as
    # expensive, A receives four times the inference service.
    prep_gpu = [("A", 4), ("B", 1), ("A", 4), ("B", 1), ("A", 4), ("B", 1)]
    system_panel(
        prep_only,
        "Preparation-only max-min",
        prep_gpu,
        "GPU received service: A = 12 units, B = 3 units  →  gap = 9",
    )
    prep_only.text(
        8,
        -0.08,
        "After fair preparation, native/FCFS inference admits equal request counts—not equal GPU service.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    cross = fig.add_subplot(grid[2])
    # Service-aware admission permits four inexpensive B requests for each A
    # request, equalizing occupied inference service without preemption.
    cross_gpu = [("A", 4), ("B", 1), ("B", 1), ("B", 1), ("B", 1), ("A", 4), ("B", 1), ("B", 1), ("B", 1), ("B", 1)]
    system_panel(
        cross,
        "Cross-stage max-min",
        cross_gpu,
        "GPU received service: A = 8 units, B = 8 units  →  gap = 0",
    )
    cross.text(
        8,
        -0.08,
        "Inference admission chooses the least-served tenant; B receives more requests because each is cheaper.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.suptitle(
        "MOCKUP — Why Preparation-Only and Cross-Stage Differ at the GPU\n"
        "Illustrative service units; non-preemptive request execution",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=230, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
