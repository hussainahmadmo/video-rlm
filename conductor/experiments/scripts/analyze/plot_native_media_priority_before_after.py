#!/usr/bin/env python3
"""Draw native vLLM media scheduling before and after priority propagation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BACKGROUND = "#8A8A8A"
BACKGROUND_EDGE = "#4D4D4D"
URGENT = "#D55E00"
PRIORITY = "#009E73"
ENGINE = "#0072B2"
FETCH = "#E69F00"
TEXT = "#202020"
MUTED = "#666666"


def rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str = "white",
    edgecolor: str = TEXT,
    fontsize: float = 9.5,
    linewidth: float = 1.7,
    textcolor: str = TEXT,
    weight: str = "normal",
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.055",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight=weight,
        linespacing=1.12,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = TEXT,
    linewidth: float = 1.7,
    linestyle: str = "-",
):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=8,
    )
    ax.add_patch(patch)
    return patch


def request_box(ax, x: float, y: float, label: str, urgent: bool = False):
    color = URGENT if urgent else BACKGROUND
    edge = URGENT if urgent else BACKGROUND_EDGE
    return rounded_box(
        ax,
        x,
        y,
        0.38,
        0.46,
        label,
        facecolor=color,
        edgecolor=edge,
        fontsize=9,
        textcolor="white",
        weight="bold",
    )


def draw_arrivals(ax, x: float, y: float):
    ax.text(x + 0.66, y + 0.72, "Raw-video arrivals", ha="center", fontsize=10.5, fontweight="bold")
    request_box(ax, x, y, "B5")
    request_box(ax, x + 0.43, y, "B6")
    request_box(ax, x + 0.86, y, "B7")
    request_box(ax, x + 1.29, y, "U", urgent=True)
    ax.text(x + 0.61, y - 0.17, "background, p=10", ha="center", fontsize=7.7, color=MUTED)
    ax.text(x + 1.48, y - 0.34, "urgent, p=0", ha="center", fontsize=7.7, color=URGENT)


def draw_queue(ax, x: float, y: float, order: list[str], title: str, color: str):
    width = 2.18
    rounded_box(
        ax,
        x,
        y,
        width,
        1.22,
        "",
        facecolor="white",
        edgecolor=color,
        linewidth=2.2,
    )
    ax.text(x + width / 2, y + 0.96, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)
    for index, item in enumerate(order):
        request_box(ax, x + 0.20 + index * 0.47, y + 0.25, item, urgent=item == "U")
    ax.text(x + 0.12, y + 0.07, "front", ha="left", fontsize=7.7, color=MUTED)


def draw_workers(ax, x: float, y: float, after: bool):
    width = 2.18
    color = PRIORITY if after else FETCH
    rounded_box(
        ax,
        x,
        y,
        width,
        1.38,
        "",
        facecolor="white",
        edgecolor=color,
        linewidth=2.2,
    )
    label = "4 admitted media jobs" if after else "4 media-loading threads"
    ax.text(x + width / 2, y + 1.12, label, ha="center", fontsize=10.5, fontweight="bold", color=color)
    for index in range(4):
        request_box(ax, x + 0.17 + index * 0.47, y + 0.51, f"B{index + 1}")
    action = "fetch + decode + sample" if after else "decode + sample"
    ax.text(x + width / 2, y + 0.20, action, ha="center", fontsize=8.5, color=MUTED)
    ax.text(
        x + width / 2,
        y - 0.15,
        "running jobs remain non-preemptive",
        ha="center",
        fontsize=8,
        color=MUTED,
    )


def draw_engine(ax, x: float, y: float):
    rounded_box(
        ax,
        x,
        y,
        1.62,
        1.22,
        "vLLM engine\npriority heap",
        facecolor="#E6F2F8",
        edgecolor=ENGINE,
        fontsize=10.5,
        linewidth=2.2,
        weight="bold",
    )
    rounded_box(
        ax,
        x + 1.98,
        y + 0.22,
        0.78,
        0.78,
        "GPU\nfirst token",
        facecolor="#DCECF5",
        edgecolor=ENGINE,
        fontsize=9,
        linewidth=2.0,
        weight="bold",
    )
    arrow(ax, x + 1.66, y + 0.61, x + 1.94, y + 0.61, color=ENGINE)


def draw_before(ax):
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 4.55)
    ax.axis("off")
    ax.add_patch(Rectangle((0.05, 0.10), 14.35, 4.18, facecolor="#FFF8F2", edgecolor="#E8D8CA", linewidth=1.2))
    ax.text(0.32, 3.93, "(a) Before: native vLLM engine-only priority", fontsize=14, fontweight="bold", color="#8A3B00")

    draw_arrivals(ax, 0.38, 1.83)
    arrow(ax, 2.18, 2.07, 2.56, 2.07, color=FETCH)

    rounded_box(
        ax,
        2.62,
        1.47,
        1.58,
        1.22,
        "Async HTTP\nfetch",
        facecolor="#FFF1D6",
        edgecolor=FETCH,
        fontsize=10,
        linewidth=2.1,
        weight="bold",
    )
    ax.text(3.41, 1.22, "priority ignored", ha="center", fontsize=8.5, color=URGENT, fontweight="bold")
    arrow(ax, 4.24, 2.07, 4.62, 2.07, color=FETCH)

    draw_queue(ax, 4.68, 1.47, ["B5", "B6", "B7", "U"], "Unbounded FIFO SimpleQueue", FETCH)
    ax.text(5.77, 1.22, "urgent remains behind queued background", ha="center", fontsize=8.5, color=URGENT, fontweight="bold")
    arrow(ax, 6.90, 2.07, 7.27, 2.07, color=FETCH)

    draw_workers(ax, 7.33, 1.39, after=False)
    arrow(ax, 9.57, 2.07, 9.93, 2.07)
    draw_engine(ax, 9.99, 1.47)

    ax.annotate(
        "Request priority becomes visible only here",
        xy=(10.80, 2.72),
        xytext=(10.80, 3.36),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=URGENT,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": URGENT, "lw": 1.5},
    )
    ax.text(7.02, 0.47, "Upstream waiting is already part of TTFT and cannot be recovered by engine scheduling", ha="center", fontsize=10.5, color=URGENT, fontweight="bold")


def draw_after(ax):
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 4.55)
    ax.axis("off")
    ax.add_patch(Rectangle((0.05, 0.10), 14.35, 4.18, facecolor="#F2FAF7", edgecolor="#CDE6DC", linewidth=1.2))
    ax.text(0.32, 3.93, "(b) After: native end-to-end media and engine priority", fontsize=14, fontweight="bold", color=PRIORITY)

    draw_arrivals(ax, 0.38, 1.83)
    arrow(ax, 2.18, 2.07, 2.72, 2.07, color=PRIORITY)

    draw_queue(ax, 2.78, 1.47, ["U", "B5", "B6", "B7"], "Priority media admission", PRIORITY)
    ax.text(3.87, 1.22, "ordered by (priority, arrival sequence)", ha="center", fontsize=8.5, color=PRIORITY, fontweight="bold")
    arrow(ax, 5.00, 2.07, 5.55, 2.07, color=PRIORITY)

    draw_workers(ax, 5.61, 1.39, after=True)
    arrow(ax, 7.85, 2.07, 8.38, 2.07, color=PRIORITY)
    ax.text(8.12, 2.29, "prepared", ha="center", fontsize=8, color=MUTED)
    draw_engine(ax, 8.44, 1.47)

    ax.annotate(
        "Same application priority",
        xy=(9.25, 2.72),
        xytext=(9.25, 3.36),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=PRIORITY,
        fontweight="bold",
        arrowprops={"arrowstyle": "-[,widthB=10.4", "color": PRIORITY, "lw": 1.6},
    )
    ax.text(6.42, 0.47, "Urgent work starts when the next media slot becomes free; active background work is not interrupted", ha="center", fontsize=10.5, color=PRIORITY, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(16.5, 9.0))
    draw_before(axes[0])
    draw_after(axes[1])
    fig.suptitle(
        "Priority must govern native media admission before engine admission",
        fontsize=19,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.55)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = {args.output}
    if args.output.suffix.lower() == ".png":
        outputs.add(args.output.with_suffix(".pdf"))
    elif args.output.suffix.lower() == ".pdf":
        outputs.add(args.output.with_suffix(".png"))

    for output in outputs:
        if output.suffix.lower() == ".png":
            fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(output, bbox_inches="tight", facecolor="white")

    for output in sorted(outputs):
        print(output)


if __name__ == "__main__":
    main()
