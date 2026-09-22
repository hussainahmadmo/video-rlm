#!/usr/bin/env python3
"""Generate the compact Conductor system overview."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("69f58e9744973ad74e35062f/figures/system")
OUT.mkdir(parents=True, exist_ok=True)

INK = "#243746"
MUTED = "#687681"
ORANGE = "#ef7d22"
ORANGE_LIGHT = "#fff5ec"
BLUE_LIGHT = "#edf5fa"
GRAY_LIGHT = "#f6f7f8"

fig, ax = plt.subplots(figsize=(11.7, 1.65))
ax.set(xlim=(0, 15.8), ylim=(0, 2.1))
ax.axis("off")


def box(x, y, w, h, title, detail="", *, fill=GRAY_LIGHT, edge=INK, width=1.1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.07",
        facecolor=fill, edgecolor=edge, linewidth=width,
    )
    ax.add_patch(patch)
    center = x + w / 2
    ax.text(center, y + h * (0.62 if detail else 0.50), title,
            ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
    if detail:
        ax.text(center, y + h * 0.30, detail, ha="center", va="center",
                fontsize=6.7, color=MUTED, linespacing=1.08)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=1.1, color=INK,
    ))


y, h = 0.47, 1.16
box(0.25, y, 2.05, h, "Pending requests", "request metadata")
box(2.80, y, 3.45, h, "Conductor", "predict service  →  select (SJF)  →  place",
    fill=ORANGE_LIGHT, edge=ORANGE, width=1.5)
box(6.75, y, 2.70, h, "Preparation", "CPU workers  |  GPU decoder lanes",
    fill=BLUE_LIGHT)
box(9.95, y, 2.05, h, "Model-ready queue")
box(12.50, y, 2.05, h, "vLLM inference", "unmodified")

for start, end in ((2.30, 2.80), (6.25, 6.75), (9.45, 9.95), (12.00, 12.50)):
    arrow(start, y + h / 2, end, y + h / 2)
arrow(14.55, y + h / 2, 15.35, y + h / 2)
ax.text(15.40, y + h / 2, "response", ha="left", va="center",
        fontsize=7.2, color=INK)

ax.text(4.525, 0.18, "Fixed limits bound preparation capacity and the model-ready queue",
        ha="center", va="center", fontsize=6.5, color=MUTED)

fig.tight_layout(pad=0.10)
for extension in ("pdf", "svg", "png"):
    options = {"dpi": 300} if extension == "png" else {}
    fig.savefig(OUT / f"conductor_architecture.{extension}",
                bbox_inches="tight", **options)
plt.close(fig)
