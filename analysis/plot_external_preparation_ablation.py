"""Plot the clean native-vLLM versus external-prepared-frames ablation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("/workspace/video-rlm/analysis")
OUT.mkdir(parents=True, exist_ok=True)

methods = ["Native vLLM\nvideo input", "Ours: external\nprepared frames"]
colors = ["#4C72B0", "#2A9D4B"]

# Same 249-question workload, offered load 1.0 QPS.
accuracy = np.array([100 * 146 / 249, 100 * 168 / 249])
throughput = np.array([0.09487377899673455, 0.9935434501654518])
mean_delay = np.array([544.8701747950844, 2.216342286866936])
p95_delay = np.array([2034.712491658982, 3.5436477889306843])
errors = [9, 0]

qps_gain = throughput[1] / throughput[0]
delay_gain = mean_delay[0] / mean_delay[1]
accuracy_gain = accuracy[1] - accuracy[0]

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 12,
    "axes.titleweight": "bold",
})
fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1))
fig.suptitle(
    "External Prepared-Frame Serving vs. Native vLLM Video Input",
    fontsize=17,
    fontweight="bold",
    y=1.02,
)

def bars(axis, values, title, ylabel, fmt, ylim=None, log=False):
    positions = np.arange(2)
    rendered = axis.bar(positions, values, color=colors, width=0.62,
                         edgecolor="#333333", linewidth=0.7)
    axis.set_xticks(positions, methods)
    axis.set_title(title, pad=11)
    axis.set_ylabel(ylabel)
    if log:
        axis.set_yscale("log")
    elif ylim:
        axis.set_ylim(*ylim)
    for rect, value in zip(rendered, values):
        axis.text(rect.get_x() + rect.get_width() / 2,
                  value * (1.30 if log else 1.035), fmt.format(value),
                  ha="center", va="bottom", fontsize=11, fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25)


bars(axes[0], accuracy, "Answer accuracy", "Accuracy (%)", "{:.2f}%", ylim=(0, 78))
axes[0].annotate(
    f"+{accuracy_gain:.2f} pp", xy=(1, accuracy[1]), xytext=(0.5, 74),
    ha="center", color="#167A33", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#167A33", lw=1.5),
)
axes[0].text(0.5, 8, "146 / 249 correct\n9 errors", ha="center", color="#4C72B0", fontsize=10)
axes[0].text(1.0, 8, "168 / 249 correct\n0 errors", ha="center", color="#167A33", fontsize=10)

bars(axes[1], throughput, "Completed-workload throughput", "Questions per second", "{:.3f} QPS", ylim=(0, 1.18))
axes[1].annotate(
    f"{qps_gain:.2f}× higher", xy=(1, throughput[1]), xytext=(0.5, 0.80),
    ha="center", color="#167A33", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#167A33", lw=1.5),
)

positions = np.arange(2)
width = 0.32
axes[2].bar(positions - width / 2, mean_delay, width, label="Mean E2E", color="#F4A261", edgecolor="#333333", linewidth=0.7)
axes[2].bar(positions + width / 2, p95_delay, width, label="p95 E2E", color="#E76F51", edgecolor="#333333", linewidth=0.7)
axes[2].set_yscale("log")
axes[2].set_xticks(positions, methods)
axes[2].set_ylabel("End-to-end delay (seconds; log scale)")
axes[2].set_title("Tail-latency collapse")
axes[2].legend(frameon=False, fontsize=10, loc="upper left")
axes[2].annotate(
    f"{delay_gain:.1f}× lower\nmean E2E", xy=(1 - width / 2, mean_delay[1]), xytext=(0.5, 140),
    ha="center", color="#167A33", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#167A33", lw=1.5),
)
axes[2].spines[["top", "right"]].set_visible(False)
axes[2].grid(axis="y", alpha=0.25, which="both")

fig.text(
    0.5, -0.02,
    "Full 249-question workload at 1.0 QPS offered load. Both configurations use eight uniformly sampled frames; "
    "the difference is native vLLM video ingestion versus external JPEG preparation.",
    ha="center", fontsize=10.5,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))

for suffix, options in [("png", {"dpi": 220}), ("svg", {})]:
    fig.savefig(OUT / f"external_preparation_vs_native_249.{suffix}", bbox_inches="tight", **options)

print(OUT / "external_preparation_vs_native_249.png")
print(OUT / "external_preparation_vs_native_249.svg")
