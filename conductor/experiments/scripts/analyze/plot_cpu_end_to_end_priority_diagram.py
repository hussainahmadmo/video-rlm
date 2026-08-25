#!/usr/bin/env python3
"""Draw the CPU-to-GPU priority gap and the end-to-end scheduling design."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BG = "#9E9E9E"
BG_EDGE = "#555555"
URGENT = "#D62728"
CPU = "#E69F00"
GPU = "#0072B2"
GREEN = "#009E73"
TEXT = "#202020"


def box(ax, x, y, w, h, face, edge, label, *, fontsize=10, lw=1.8, z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT if face != URGENT else "white",
        fontweight="bold",
        zorder=z + 1,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=TEXT, lw=1.8, style="-"):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def stage_outline(ax, x, y, w, h, color, title, subtitle):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="white",
        edgecolor=color,
        linewidth=2.4,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h + 0.16,
        title,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + w / 2,
        y - 0.12,
        subtitle,
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )


def request_stream(ax, y):
    ax.text(0.35, y + 0.55, "Arrivals", ha="center", fontsize=11, fontweight="bold")
    items = [
        (0.05, "B1", BG, BG_EDGE),
        (0.42, "B2", BG, BG_EDGE),
        (0.79, "B3", BG, BG_EDGE),
        (1.16, "U", URGENT, URGENT),
    ]
    for x, label, face, edge in items:
        box(ax, x, y - 0.05, 0.30, 0.44, face, edge, label, fontsize=9)
    ax.text(0.74, y - 0.22, "background", ha="center", fontsize=8, color="#666666")
    ax.text(1.31, y - 0.22, "urgent", ha="center", fontsize=8, color=URGENT)


def queue_boxes(ax, y, ordering):
    start = 2.18
    for index, item in enumerate(ordering):
        face = URGENT if item == "U" else BG
        edge = URGENT if item == "U" else BG_EDGE
        box(ax, start + index * 0.47, y + 0.25, 0.39, 0.52, face, edge, item, fontsize=10)


def engine_boxes(ax, y, ordering):
    start = 6.18
    for index, item in enumerate(ordering):
        face = URGENT if item == "U" else "#A6CEE3"
        edge = URGENT if item == "U" else GPU
        box(ax, start + index * 0.47, y + 0.25, 0.39, 0.52, face, edge, item, fontsize=10)


def draw_row(ax, y, *, end_to_end: bool):
    request_stream(ax, y + 0.52)
    arrow(ax, 1.54, y + 0.73, 2.02, y + 0.73)

    cpu_title = "CPU media preparation"
    cpu_policy = "Priority queue" if end_to_end else "FCFS queue"
    stage_outline(
        ax,
        2.02,
        y,
        2.50,
        1.46,
        GREEN if end_to_end else CPU,
        cpu_title,
        f"{cpu_policy}: decode → sample → resize",
    )
    ordering = ["B1", "U", "B2", "B3"] if end_to_end else ["B1", "B2", "B3", "U"]
    queue_boxes(ax, y, ordering)
    ax.text(
        3.26,
        y + 1.17,
        "Queued work reordered by priority" if end_to_end else "Urgent request blocked behind background work",
        ha="center",
        fontsize=9.2,
        color=GREEN if end_to_end else URGENT,
        fontweight="bold",
    )

    arrow(ax, 4.57, y + 0.73, 5.26, y + 0.73)
    ax.text(4.92, y + 0.88, "prepared", ha="center", fontsize=8, color="#555555")

    stage_outline(
        ax,
        5.30,
        y,
        2.18,
        1.46,
        GPU,
        "vLLM inference engine",
        "Native priority scheduling on GPU",
    )
    engine_order = ["U", "B1", "B2"] if end_to_end else ["U", "B3", "B2"]
    engine_boxes(ax, y, engine_order)
    ax.text(
        6.39,
        y + 1.17,
        "Priority preserved" if end_to_end else "Priority starts here—too late",
        ha="center",
        fontsize=9.2,
        color=GREEN if end_to_end else URGENT,
        fontweight="bold",
    )
    arrow(ax, 7.54, y + 0.73, 8.15, y + 0.73, color=GPU)
    box(ax, 8.18, y + 0.48, 0.67, 0.50, "#E8F2FA", GPU, "First\ntoken", fontsize=9)

    if end_to_end:
        ax.annotate(
            "Same application priority\npropagated across both stages",
            xy=(4.92, y + 0.73),
            xytext=(4.92, y + 1.72),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=GREEN,
            arrowprops={"arrowstyle": "-[,widthB=7.8", "lw": 1.8, "color": GREEN},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(15.5, 7.7))
    ax.set_xlim(-0.15, 9.05)
    ax.set_ylim(-0.25, 6.15)
    ax.axis("off")

    # Panel separators and labels.
    ax.add_patch(Rectangle((-0.10, 3.02), 9.05, 2.45, facecolor="#FFF8F2", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((-0.10, 0.26), 9.05, 2.45, facecolor="#F3FBF8", edgecolor="none", zorder=0))
    ax.text(
        0.02,
        5.27,
        "(a) Engine-only priority",
        fontsize=14,
        fontweight="bold",
        color="#8A3B00",
    )
    ax.text(
        0.02,
        2.51,
        "(b) End-to-end CPU–GPU priority (ours)",
        fontsize=14,
        fontweight="bold",
        color=GREEN,
    )

    draw_row(ax, 3.35, end_to_end=False)
    draw_row(ax, 0.59, end_to_end=True)

    ax.text(
        8.85,
        3.22,
        "Long TTFT",
        ha="right",
        fontsize=11,
        color=URGENT,
        fontweight="bold",
    )
    ax.text(
        8.85,
        0.46,
        "Urgent queueing removed",
        ha="right",
        fontsize=11,
        color=GREEN,
        fontweight="bold",
    )

    fig.suptitle(
        "Priority must begin at CPU media preparation, not only at the GPU engine",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = [args.output]
    if args.output.suffix.lower() == ".png":
        outputs.append(args.output.with_suffix(".pdf"))
    for output in outputs:
        fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
        print(output)


if __name__ == "__main__":
    main()
