from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/workspace/video-rlm/conductor/experiments/large_sweeps")
OUT = Path("/workspace/video-rlm/analysis")
OUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "Native raw-8": ROOT / "native_raw8_rate10_20260819_163241",
    "External uniform-8": ROOT / "overnight_ablations_20260819_180956/external_uniform8",
    "Fixed-8 hybrid*": ROOT / "fixed8_rate05_20260819_152945",
    "Adaptive 2/8": ROOT / "overnight_ablations_20260819_180956/adaptive_2_8_threshold4",
    "Fixed budget-2": ROOT / "overnight_ablations_20260819_180956/fixed_budget2",
    "Fixed budget-32": ROOT / "overnight_ablations_20260819_180956/fixed_budget32",
}

STYLES = {
    "Native raw-8": dict(color="#3155D9", marker="x", hatch="///"),
    "External uniform-8": dict(color="#E53935", marker="P", hatch="--"),
    "Fixed-8 hybrid*": dict(color="#2A9D3F", marker="o", hatch="xx"),
    "Adaptive 2/8": dict(color="#111111", marker="*", hatch=".."),
    "Fixed budget-2": dict(color="#F4A742", marker="^", hatch="\\\\"),
    "Fixed budget-32": dict(color="#8E5CC2", marker="s", hatch="++"),
}

DATASETS = ["EgoSchema", "LVBench", "NExT-QA", "STAR", "VRBench"]


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


summaries = {name: read_json(path / "summary.json") for name, path in METHODS.items()}
records = {name: read_jsonl(path / "results.jsonl") for name, path in METHODS.items()}


def accuracy(rows):
    return 100 * sum(bool(row.get("correct")) for row in rows) / len(rows)


def mean_e2e(rows):
    return float(np.mean([row["end_to_end_delay_s"] for row in rows]))


def method_summary(name):
    summary = summaries[name]
    if "accuracy_percent" in summary:
        accuracy_value = summary["accuracy_percent"]
    else:
        accuracy_value = (
            100
            * summary["correct_this_invocation"]
            / summary["completed_this_invocation"]
        )
    return {
        "accuracy": accuracy_value,
        "throughput": summary["throughput_qps"],
        "mean_e2e": summary["mean_end_to_end_delay_s"],
        "p50": summary["p50_end_to_end_delay_s"],
        "p95": summary["p95_end_to_end_delay_s"],
    }


metrics = {name: method_summary(name) for name in METHODS}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

# Figure 1: paper-style quality-delay panels per dataset.
fig, axes = plt.subplots(1, 5, figsize=(17.2, 3.65), sharey=False)
handles = []
labels = []
for ax, dataset in zip(axes, DATASETS):
    subset = defaultdict(list)
    for method, rows in records.items():
        subset[method] = [row for row in rows if row.get("dataset") == dataset]

    for method in METHODS:
        rows = subset[method]
        style = STYLES[method]
        point = ax.scatter(
            mean_e2e(rows),
            accuracy(rows),
            s=58 if style["marker"] != "*" else 95,
            color=style["color"],
            marker=style["marker"],
            linewidths=1.4,
            zorder=3,
            label=method,
        )
        if ax is axes[0]:
            handles.append(point)
            labels.append(method)

    native_delay = mean_e2e(subset["Native raw-8"])
    external_delay = mean_e2e(subset["External uniform-8"])
    native_acc = accuracy(subset["Native raw-8"])
    external_acc = accuracy(subset["External uniform-8"])
    delay_gain = native_delay / external_delay
    ax.annotate(
        f"{delay_gain:.1f}x lower delay",
        xy=(external_delay, external_acc),
        xytext=(native_delay, max(native_acc, external_acc) + 3.0),
        arrowprops=dict(arrowstyle="->", color="#1B7F2A", lw=1.0),
        ha="right",
        fontsize=8,
        color="#1B7F2A",
    )
    ax.set_title(f"Dataset: {dataset}")
    ax.set_xscale("log")
    ax.grid(True, which="both", color="#D9D9D9", linewidth=0.55)
    ax.set_xlabel("Mean E2E Delay (s)")
    ymin = min(accuracy(rows) for rows in subset.values()) - 6
    ymax = max(accuracy(rows) for rows in subset.values()) + 9
    ax.set_ylim(max(0, ymin), min(100, ymax))
    if ax is axes[0]:
        ax.set_ylabel("Accuracy (%)")

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=6,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.35,
)
fig.suptitle(
    "Quality–Delay Trade-off Across the 249-Question Workload",
    y=1.19,
    fontsize=14,
    fontweight="bold",
)
fig.text(
    0.5,
    -0.04,
    "Lower-left is faster; higher is more accurate. Mean E2E delay includes queueing in the shared mixed workload. "
    "*Fixed-8 hybrid was measured at 0.5 QPS; other methods at 1.0 QPS.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(w_pad=1.05)
fig.savefig(OUT / "paper_quality_delay_by_dataset.png", dpi=240, bbox_inches="tight")
fig.savefig(OUT / "paper_quality_delay_by_dataset.svg", bbox_inches="tight")
plt.close(fig)

# Figure 2: overall quality, throughput, and delay in compact paper panels.
names = list(METHODS)
x = np.arange(len(names))
short_names = ["Native\nraw-8", "External\n8", "Hybrid\nfixed-8*", "Adaptive\n2/8", "Fixed\n2", "Fixed\n32"]
colors = [STYLES[name]["color"] for name in names]
hatches = [STYLES[name]["hatch"] for name in names]

fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.55))

def styled_bars(ax, values, title, ylabel, ylim=None, fmt="{:.2f}"):
    bars = ax.bar(x, values, color=colors, edgecolor="#222222", linewidth=0.7, width=0.70)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, short_names)
    if ylim:
        ax.set_ylim(*ylim)
    for bar, value in zip(bars, values):
        y = value * 1.035 if ax.get_yscale() == "log" else value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
        ax.text(bar.get_x() + bar.get_width() / 2, y, fmt.format(value), ha="center", fontsize=8.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return bars

styled_bars(
    axes[0],
    [metrics[name]["accuracy"] for name in names],
    "Answer Quality",
    "Accuracy (%)",
    (45, 73),
    "{:.1f}%",
)

styled_bars(
    axes[1],
    [metrics[name]["throughput"] for name in names],
    "System Throughput",
    "Throughput (QPS)",
    (0, 1.13),
    "{:.3f}",
)
axes[1].annotate(
    "10.47x",
    xy=(1, metrics["External uniform-8"]["throughput"]),
    xytext=(1.8, 1.075),
    arrowprops=dict(arrowstyle="->", lw=1.0),
    ha="center",
    fontsize=10,
    fontweight="bold",
)

axes[2].set_yscale("log")
bars = styled_bars(
    axes[2],
    [metrics[name]["mean_e2e"] for name in names],
    "Average End-to-End Delay",
    "Mean E2E delay (s, log scale)",
    (1, 1300),
    "{:.1f}",
)
native_delay = metrics["Native raw-8"]["mean_e2e"]
for index, name in enumerate(names[1:], start=1):
    gain = native_delay / metrics[name]["mean_e2e"]
    axes[2].text(index, metrics[name]["mean_e2e"] * 1.85, f"{gain:.1f}x", ha="center", fontsize=8, fontweight="bold")

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=STYLES[name]["color"], edgecolor="#222222", hatch=STYLES[name]["hatch"])
    for name in names
]
fig.legend(
    legend_handles,
    names,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.075),
    ncol=3,
    frameon=False,
    columnspacing=1.5,
)
fig.suptitle("Full 249-Question Ablation", fontsize=15, fontweight="bold", y=1.17)
fig.text(
    0.5,
    -0.025,
    "Numbers above delay bars are reductions relative to native raw-8. "
    "*Fixed-8 hybrid used 0.5 QPS offered load; all other methods used 1.0 QPS.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "paper_all_methods_overall.png", dpi=240, bbox_inches="tight")
fig.savefig(OUT / "paper_all_methods_overall.svg", bbox_inches="tight")
plt.close(fig)

print(OUT / "paper_quality_delay_by_dataset.png")
print(OUT / "paper_all_methods_overall.png")
