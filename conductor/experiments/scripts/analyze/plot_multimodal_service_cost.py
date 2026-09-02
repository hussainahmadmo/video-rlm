#!/usr/bin/env python3
"""Draw the paper's multimodal service-cost motivation diagram.

The figure is intentionally schematic.  It explains why counting completed
requests is not a meaningful fairness unit when requests consume different
amounts of host-side media preparation and accelerator-side inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


PREP = "#3B82C4"
PREFILL = "#F28E2B"
DECODE = "#7A5195"
EDGE = "#263238"


def draw_request(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    prep: float,
    prefill: float,
    decode_tokens: int,
    token_width: float,
    height: float = 0.36,
) -> float:
    """Draw one request and return its total illustrated width."""
    ax.add_patch(
        Rectangle((x, y), prep, height, facecolor=PREP, edgecolor=EDGE, lw=0.8)
    )
    x += prep
    ax.add_patch(
        Rectangle(
            (x, y), prefill, height, facecolor=PREFILL, edgecolor=EDGE, lw=0.8
        )
    )
    x += prefill
    for _ in range(decode_tokens):
        ax.add_patch(
            Rectangle(
                (x, y),
                token_width,
                height,
                facecolor=DECODE,
                edgecolor=EDGE,
                lw=0.65,
            )
        )
        x += token_width
    return prep + prefill + decode_tokens * token_width


def bracket(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    y: float,
    text: str,
    text_y: float,
) -> None:
    ax.plot([x0, x0, x1, x1], [y, y + 0.09, y + 0.09, y], color="#555", lw=1)
    ax.text((x0 + x1) / 2, text_y, text, ha="center", va="bottom", fontsize=9.5)


def build_figure(output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.25))
    ax.set_xlim(0, 15.3)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # Left: two expensive requests occupy much of the illustrated interval.
    left_x = 0.65
    left_y = [3.55, 2.65]
    expensive_widths = []
    for i, y in enumerate(left_y):
        expensive_widths.append(
            draw_request(
                ax,
                x=left_x,
                y=y,
                prep=2.25 if i == 0 else 1.65,
                prefill=1.15,
                decode_tokens=4,
                token_width=0.34,
            )
        )

    bracket(
        ax,
        x0=left_x,
        x1=left_x + 2.25,
        y=4.03,
        text="CPU media\npreparation",
        text_y=4.13,
    )
    bracket(
        ax,
        x0=left_x + 2.25,
        x1=left_x + 3.40,
        y=4.03,
        text="GPU visual\nprefill",
        text_y=4.13,
    )
    ax.annotate(
        "one output token",
        xy=(left_x + 3.40 + 0.17, 3.92),
        xytext=(left_x + 4.55, 4.62),
        ha="center",
        fontsize=10.5,
        arrowprops=dict(arrowstyle="->", color="#555", lw=1),
    )
    ax.text(
        left_x + 2.8,
        1.85,
        "Higher service cost per request\n$\Rightarrow$ lower request throughput",
        ha="center",
        va="top",
        fontsize=13,
        linespacing=1.3,
    )

    # Right: the same resource interval can finish more inexpensive requests.
    right_x = 9.0
    right_y = [4.05, 3.35, 2.65, 1.95]
    for y in right_y:
        draw_request(
            ax,
            x=right_x,
            y=y,
            prep=0.75,
            prefill=0.58,
            decode_tokens=3,
            token_width=0.25,
            height=0.32,
        )
    ax.text(
        12.55,
        3.15,
        "Lower service cost per request\n$\Rightarrow$ higher request throughput",
        ha="left",
        va="center",
        fontsize=13,
        linespacing=1.3,
    )

    handles = [
        Patch(facecolor=PREP, edgecolor=EDGE, label="CPU media preparation"),
        Patch(facecolor=PREFILL, edgecolor=EDGE, label="GPU visual prefill"),
        Patch(facecolor=DECODE, edgecolor=EDGE, label="GPU output decoding"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=1.6,
    )

    ax.text(
        7.65,
        0.55,
        "Equal request counts do not imply equal service across CPU and GPU stages.",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/multimodal_service_cost"),
        help="Output path without an extension (PDF, SVG, and PNG are written).",
    )
    args = parser.parse_args()
    build_figure(args.output)


if __name__ == "__main__":
    main()
