from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("/workspace/video-rlm/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

methods = ["Native raw-8", "External uniform-8"]
colors = ["#D55E00", "#0072B2"]

accuracy = [44.89795918367347, 53.06122448979592]
service = {
    "Mean": [10.09976110683412, 1.9734863122278938],
    "P50": [7.339966169092804, 1.922157343942672],
    "P95": [28.372852573171258, 2.691581543069333],
}
e2e = {
    "Mean": [930.3814141068094, 2.0839623265177467],
    "P50": [1249.000189539045, 2.035072656814009],
    "P95": [2055.216070560273, 2.754299869760871],
}

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
fig.suptitle(
    "LVBench: Native Raw Video vs External Frame Decoding (n=49)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# Accuracy
bars = axes[0].bar(methods, accuracy, color=colors, width=0.62)
axes[0].set_title("Accuracy (higher is better)")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 65)
for bar, value in zip(bars, accuracy):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.2,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold",
    )
axes[0].text(0.5, 60, "+8.16 percentage points", ha="center", color="#333333")

# Service latency
x = np.arange(len(service))
width = 0.34
native_service = [values[0] for values in service.values()]
external_service = [values[1] for values in service.values()]
axes[1].bar(x - width / 2, native_service, width, label=methods[0], color=colors[0])
axes[1].bar(x + width / 2, external_service, width, label=methods[1], color=colors[1])
axes[1].set_title("Request service latency (lower is better)")
axes[1].set_ylabel("Seconds")
axes[1].set_xticks(x, service.keys())
axes[1].legend(frameon=False)
for i, (native, external) in enumerate(zip(native_service, external_service)):
    axes[1].text(i - width / 2, native + 0.7, f"{native:.2f}", ha="center", fontsize=9)
    axes[1].text(i + width / 2, external + 0.7, f"{external:.2f}", ha="center", fontsize=9)

# E2E latency under the shared mixed workload; log scale keeps both visible.
native_e2e = [values[0] for values in e2e.values()]
external_e2e = [values[1] for values in e2e.values()]
axes[2].bar(x - width / 2, native_e2e, width, label=methods[0], color=colors[0])
axes[2].bar(x + width / 2, external_e2e, width, label=methods[1], color=colors[1])
axes[2].set_title("End-to-end delay under mixed load")
axes[2].set_ylabel("Seconds (log scale)")
axes[2].set_yscale("log")
axes[2].set_xticks(x, e2e.keys())
axes[2].legend(frameon=False)
for i, (native, external) in enumerate(zip(native_e2e, external_e2e)):
    axes[2].text(i - width / 2, native * 1.12, f"{native:.0f}", ha="center", fontsize=9)
    axes[2].text(i + width / 2, external * 1.18, f"{external:.2f}", ha="center", fontsize=9)

for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelrotation=8)

fig.text(
    0.5,
    0.015,
    "External decoding: 5.12x lower mean LVBench service latency. "
    "Overall 249-question mixed-workload throughput: 10.47x higher (not an LVBench-only throughput estimate).",
    ha="center",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.93))

png_path = OUTPUT_DIR / "lvbench_external_vs_native.png"
svg_path = OUTPUT_DIR / "lvbench_external_vs_native.svg"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
print(png_path)
print(svg_path)
