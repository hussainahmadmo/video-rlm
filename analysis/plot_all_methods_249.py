from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


output_dir = Path("/workspace/video-rlm/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

methods = [
    "Native\nraw-8",
    "External\nuniform-8",
    "Fixed-8\nhybrid*",
    "Adaptive\n2/8",
    "Fixed\nbudget-2",
    "Fixed\nbudget-32",
]
accuracy = np.array([58.6345, 67.4699, 67.0683, 59.8394, 54.2169, 69.0763])
throughput = np.array([0.0948738, 0.9935435, 0.4800069, 0.8318920, 0.9844407, 0.4504788])
mean_e2e = np.array([544.8702, 2.2163, 7.4148, 15.1836, 2.9631, 106.5059])
p50_e2e = np.array([212.6170, 1.9993, 2.1627, 6.1944, 0.8144, 72.5780])
p95_e2e = np.array([2034.7125, 3.5436, 27.3163, 45.9689, 9.7819, 272.3605])
errors = [9, 0, 0, 0, 0, 0]

colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
x = np.arange(len(methods))

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
fig.suptitle(
    "Video QA Ablation — Full 249-Question Workload",
    fontsize=18,
    fontweight="bold",
    y=0.985,
)

# Accuracy
bars = axes[0].bar(x, accuracy, color=colors, width=0.72)
bars[2].set_hatch("//")
axes[0].set_title("Accuracy (higher is better)", fontweight="bold")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(45, 73)
axes[0].set_xticks(x, methods)
for bar, value, error_count in zip(bars, accuracy, errors):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.55,
        f"{value:.1f}%",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
    if error_count:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            46,
            f"{error_count} errors",
            ha="center",
            fontsize=8,
            color="#9C2F00",
        )

# Throughput
bars = axes[1].bar(x, throughput, color=colors, width=0.72)
bars[2].set_hatch("//")
axes[1].set_title("Throughput (higher is better)", fontweight="bold")
axes[1].set_ylabel("Questions per second (QPS)")
axes[1].set_ylim(0, 1.12)
axes[1].set_xticks(x, methods)
for bar, value in zip(bars, throughput):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
axes[1].annotate(
    "10.47x vs native",
    xy=(1, throughput[1]),
    xytext=(1.65, 1.065),
    arrowprops={"arrowstyle": "->", "color": "#333333"},
    ha="center",
    fontsize=10,
    fontweight="bold",
)

# Latency profile on a log scale.
width = 0.23
axes[2].bar(x - width, mean_e2e, width, label="Mean", color="#4C78A8")
axes[2].bar(x, p50_e2e, width, label="P50", color="#72B7B2")
axes[2].bar(x + width, p95_e2e, width, label="P95", color="#F58518")
axes[2].set_title("End-to-end delay (lower is better)", fontweight="bold")
axes[2].set_ylabel("Seconds (log scale)")
axes[2].set_yscale("log")
axes[2].set_ylim(0.5, 4000)
axes[2].set_xticks(x, methods)
axes[2].legend(frameon=False, ncol=3, loc="upper center")

for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=9)

fig.text(
    0.5,
    0.017,
    "* Fixed-8 hybrid was measured at 0.5 QPS offered load; all other methods at 1.0 QPS. "
    "Its accuracy is comparable, but its throughput and latency are not load-matched until the rate-1.0 rerun.",
    ha="center",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0.065, 1, 0.94))

png_path = output_dir / "all_methods_249_ablation.png"
svg_path = output_dir / "all_methods_249_ablation.svg"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
print(png_path)
print(svg_path)
