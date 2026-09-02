#!/usr/bin/env python3
"""Draw a high-level CPU/GPU service-profiler control flow."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#17365D"
CPU = "#F4B183"
GPU = "#9DC3E6"
PROFILE = "#DFC0E5"
INPUT = "#F7F8FA"
GRAY = "#5B6573"


def box(axis, xy, width, height, text, color, *, fontsize=12, linewidth=2):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=color, edgecolor=NAVY, linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=fontsize, color="#15202B",
        linespacing=1.3,
    )


def arrow(axis, start, end, *, color=NAVY, linestyle="-", width=2,
          connectionstyle="arc3"):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=width, linestyle=linestyle, color=color,
        connectionstyle=connectionstyle, shrinkA=3, shrinkB=3,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fig, axis = plt.subplots(figsize=(13.5, 6.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    fig.suptitle("Service Profiler", fontsize=24, fontweight="bold", y=0.94)
    axis.text(
        0.5, 0.865,
        "Predict the resource cost of each request before it enters each stage",
        ha="center", fontsize=14, color=GRAY,
    )

    box(
        axis, (0.035, 0.38), 0.18, 0.23,
        "Request\n\nMedia metadata\nToken budgets",
        INPUT,
    )
    box(
        axis, (0.30, 0.29), 0.27, 0.41,
        "SERVICE PROFILER\n\nPredict CPU cost\nPredict GPU cost",
        PROFILE,
        fontsize=13,
        linewidth=2.4,
    )
    box(
        axis, (0.68, 0.57), 0.25, 0.18,
        "CPU preparation\n\nWorker time",
        CPU,
    )
    box(
        axis, (0.68, 0.23), 0.25, 0.18,
        "GPU inference\n\nTokens + batching",
        GPU,
    )

    arrow(axis, (0.215, 0.495), (0.30, 0.495))

    arrow(axis, (0.57, 0.59), (0.68, 0.66))
    arrow(axis, (0.57, 0.41), (0.68, 0.32))

    arrow(axis, (0.805, 0.57), (0.805, 0.41))

    arrow(
        axis, (0.68, 0.63), (0.56, 0.64), color=GRAY, linestyle="--",
        connectionstyle="arc3,rad=0.18",
    )
    arrow(
        axis, (0.68, 0.28), (0.56, 0.35), color=GRAY, linestyle="--",
        connectionstyle="arc3,rad=-0.18",
    )
    axis.plot([0.34, 0.39], [0.09, 0.09], color=NAVY, linewidth=2)
    axis.text(0.40, 0.09, "Predicted cost", va="center", fontsize=10,
              color=GRAY)
    axis.plot([0.56, 0.61], [0.09, 0.09], color=GRAY, linewidth=2,
              linestyle="--")
    axis.text(0.62, 0.09, "Observed-cost feedback", va="center", fontsize=10,
              color=GRAY)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
