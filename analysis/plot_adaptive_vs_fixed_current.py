from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


out = Path("/workspace/video-rlm/analysis")
out.mkdir(parents=True, exist_ok=True)

methods = ["Fixed-2", "Adaptive codec\nrefinement (2/8)", "Fixed-32"]
accuracy = [54.21686746987952, 59.83935742971887, 69.07630522088354]
throughput = [0.9844407358988972, 0.8318920268584389, 0.4504788178432452]
mean_e2e = [2.9631322175395267, 15.183619841360992, 106.50592928705737]
p50_e2e = [0.8144301539286971, 6.194415481761098, 72.57795103220269]
p95_e2e = [9.781931836158037, 45.968872035853565, 272.3605127730407]

colors = ["#F4A742", "#2A9D3F", "#7B4AB5"]
hatches = ["\\\\", "xx", "///"]
x = np.arange(len(methods))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.65))
fig.suptitle(
    "Adaptive Codec-Refinement Method vs Strict Fixed Budgets",
    fontsize=16,
    fontweight="bold",
    y=1.03,
)


def bars(ax, values, title, ylabel, ylim, labels, log=False):
    if log:
        ax.set_yscale("log")
    patches = ax.bar(x, values, color=colors, edgecolor="#222222", width=0.67, linewidth=0.8)
    for patch, hatch in zip(patches, hatches):
        patch.set_hatch(hatch)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, methods)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for patch, label, value in zip(patches, labels, values):
        y = value * 1.2 if log else value + (ylim[1] - ylim[0]) * 0.025
        ax.text(patch.get_x() + patch.get_width() / 2, y, label, ha="center", fontweight="bold", fontsize=10)


bars(
    axes[0],
    accuracy,
    "Answer Quality",
    "Accuracy (%)",
    (48, 73),
    [f"{v:.1f}%" for v in accuracy],
)
axes[0].annotate(
    "+5.62 points",
    xy=(1, accuracy[1]),
    xytext=(0.42, 65.0),
    arrowprops=dict(arrowstyle="->", lw=1.0, color="#2A9D3F"),
    color="#1B7F2A",
    fontsize=10,
    fontweight="bold",
)

bars(
    axes[1],
    throughput,
    "System Throughput",
    "Throughput (QPS)",
    (0, 1.10),
    [f"{v:.3f}" for v in throughput],
)

width = 0.23
axes[2].bar(x - width, mean_e2e, width, label="Mean", color="#4C78A8", edgecolor="#222222")
axes[2].bar(x, p50_e2e, width, label="P50", color="#72B7B2", edgecolor="#222222")
axes[2].bar(x + width, p95_e2e, width, label="P95", color="#F58518", edgecolor="#222222")
axes[2].set_yscale("log")
axes[2].set_ylim(0.5, 500)
axes[2].set_title("End-to-End Delay", fontweight="bold")
axes[2].set_ylabel("Seconds (log scale)")
axes[2].set_xticks(x, methods)
axes[2].grid(axis="y", which="both", color="#D8D8D8", linewidth=0.6)
axes[2].spines[["top", "right"]].set_visible(False)
axes[2].legend(frameon=False, ncol=3, loc="upper left")

fig.text(
    0.5,
    -0.015,
    "All configurations: 249 questions, 1.0 QPS offered load, concurrency 4, and zero errors. "
    "Fixed-8 is omitted until its load-matched run completes.",
    ha="center",
    fontsize=9.5,
)
fig.tight_layout(w_pad=1.6)

png = out / "adaptive_vs_strict_fixed_current.png"
svg = out / "adaptive_vs_strict_fixed_current.svg"
fig.savefig(png, dpi=240, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
print(png)
print(svg)
