from pathlib import Path

import matplotlib.pyplot as plt


output_dir = Path("/workspace/video-rlm/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

methods = ["Native raw-8", "External uniform-8"]
throughput = [0.09487377899673455, 0.9935434501654518]
wall_time = [2624.539705628995, 250.61812843568623]
colors = ["#D55E00", "#0072B2"]

gain = throughput[1] / throughput[0]
percent_increase = (gain - 1) * 100
wall_reduction = (1 - wall_time[1] / wall_time[0]) * 100

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
fig.suptitle(
    "Full 249-Question Workload: External Decoding Throughput Gain",
    fontsize=16,
    fontweight="bold",
)

bars = axes[0].bar(methods, throughput, color=colors, width=0.58)
axes[0].set_title("Observed throughput (higher is better)")
axes[0].set_ylabel("Questions per second (QPS)")
axes[0].set_ylim(0, 1.14)
for bar, value in zip(bars, throughput):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.035,
        f"{value:.3f} QPS",
        ha="center",
        fontweight="bold",
    )
axes[0].text(
    0.5,
    0.72,
    f"{gain:.2f}x throughput\n+{percent_increase:.0f}% increase",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="#006D9C",
)

bars = axes[1].bar(methods, wall_time, color=colors, width=0.58)
axes[1].set_title("Time to complete 249 questions (lower is better)")
axes[1].set_ylabel("Wall-clock seconds")
axes[1].set_ylim(0, 2950)
for bar, value in zip(bars, wall_time):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + 70,
        f"{value:.1f}s",
        ha="center",
        fontweight="bold",
    )
axes[1].text(
    0.5,
    1500,
    f"{wall_reduction:.1f}% less\nwall time",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="#006D9C",
)

for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)

fig.text(
    0.5,
    0.018,
    "Same 249-question mixed workload at 1.0 QPS offered load and client concurrency 4. "
    "Native raw-8 recorded 9 timeouts; external uniform-8 recorded 0.",
    ha="center",
    fontsize=9.5,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.92))

png_path = output_dir / "full249_external_throughput_gain.png"
svg_path = output_dir / "full249_external_throughput_gain.svg"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
print(f"gain={gain:.6f}x")
print(f"percent_increase={percent_increase:.2f}%")
print(png_path)
print(svg_path)
