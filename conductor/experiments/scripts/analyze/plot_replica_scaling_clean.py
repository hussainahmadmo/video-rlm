#!/usr/bin/env python3
"""Plot a publication-focused proportional-load replica-scaling figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"fcfs": "#D55E00", "priority": "#009E73"}
LABELS = {
    "fcfs": "Engine-only priority\n(FCFS media preparation)",
    "priority": "End-to-end priority",
}


def load(root: Path, replicas: int, policy: str) -> dict:
    path = root / f"replicas{replicas}_{policy}" / "summary.json"
    if not path.is_file():
        raise SystemExit(f"missing summary: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    replicas = [1, 2, 4]
    data = {
        policy: [load(args.root, count, policy) for count in replicas]
        for policy in ("fcfs", "priority")
    }

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "DejaVu Sans",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharex=True)

    metrics = (
        ("mean_end_to_end_ttft_s", "Urgent mean end-to-end TTFT (s)",
         "(a) Urgent response latency"),
        ("mean_prep_queue_wait_s", "Urgent mean preparation wait (s)",
         "(b) Delay accumulated before vLLM"),
    )
    for ax, (field, ylabel, title) in zip(axes, metrics):
        for policy, marker in (("fcfs", "o"), ("priority", "s")):
            values = [float(row["urgent"][field]) for row in data[policy]]
            ax.plot(
                replicas,
                values,
                color=COLORS[policy],
                marker=marker,
                markersize=7,
                linewidth=2.5,
                label=LABELS[policy],
                zorder=3,
            )
            for x, value in zip(replicas, values):
                offset = 8 if policy == "fcfs" else -16
                ax.annotate(
                    f"{value:.1f}",
                    (x, value),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if offset > 0 else "top",
                    color=COLORS[policy],
                    fontsize=9.5,
                    fontweight="semibold",
                )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(replicas, ["1\n80", "2\n160", "4\n320"])
        ax.set_xlabel("vLLM replicas\nTotal offered requests")
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)

    # The speedup annotation links the outcome to the scheduling change while
    # keeping the plot free of secondary metrics and duplicated legends.
    fcfs_4 = data["fcfs"][-1]["urgent"]["mean_end_to_end_ttft_s"]
    priority_4 = data["priority"][-1]["urgent"]["mean_end_to_end_ttft_s"]
    axes[0].annotate(
        f"{fcfs_4 / priority_4:.2f}× faster",
        xy=(4, priority_4),
        xytext=(3.05, 132),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#999999"},
        fontsize=10.5,
        fontweight="bold",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "GPU scale-out does not remove a shared media-preparation bottleneck",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.955,
        "Offered load scales with replicas; the shared CPU preparation pool remains fixed at 4 workers",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    fig.text(
        0.5,
        -0.02,
        "Controlled matched workload; one run per point. All configurations use native vLLM priority.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.88), w_pad=2.6)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(args.output_prefix.with_suffix(".png"))
    print(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
