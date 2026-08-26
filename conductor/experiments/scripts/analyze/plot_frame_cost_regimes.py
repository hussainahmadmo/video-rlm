#!/usr/bin/env python3
"""Plot the controlled background/urgent frame-cost matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REGIMES = (
    (8, 8, "Light background\nLight urgent"),
    (8, 128, "Light background\nHeavy urgent"),
    (128, 8, "Heavy background\nLight urgent"),
    (128, 128, "Heavy background\nHeavy urgent"),
)
POLICIES = ("fcfs", "priority")
LABELS = {
    "fcfs": "Engine-only priority",
    "priority": "End-to-end priority",
}
COLORS = {"fcfs": "#D55E00", "priority": "#009E73"}


def read_summaries(paths: list[Path]) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(paths)]


def summaries_for(
    root: Path,
    fallback: Path | None,
    background_frames: int,
    urgent_frames: int,
    policy: str,
) -> list[dict]:
    condition = f"background{background_frames}_urgent{urgent_frames}"
    paths = list(root.glob(f"{condition}/*/{policy}/summary.json"))
    if paths:
        return read_summaries(paths)
    if background_frames == urgent_frames and fallback is not None:
        paths = list(
            fallback.glob(
                f"frames{background_frames}/*/{policy}/summary.json"
            )
        )
    return read_summaries(paths)


def mean_urgent_ttft(summaries: list[dict]) -> float:
    if not summaries:
        return float("nan")
    return float(
        np.mean(
            [
                float(summary["urgent"]["mean_end_to_end_ttft_s"])
                for summary in summaries
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--uniform-fallback",
        type=Path,
        help="Optional low-frame validation root used for the 8/8 regime",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    values: dict[tuple[int, int, str], float] = {}
    counts: dict[tuple[int, int, str], int] = {}
    for background, urgent, _ in REGIMES:
        for policy in POLICIES:
            summaries = summaries_for(
                args.root,
                args.uniform_fallback,
                background,
                urgent,
                policy,
            )
            values[(background, urgent, policy)] = mean_urgent_ttft(summaries)
            counts[(background, urgent, policy)] = len(summaries)

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    x = np.arange(len(REGIMES))
    width = 0.36
    for policy_index, policy in enumerate(POLICIES):
        heights = np.array(
            [values[(background, urgent, policy)] for background, urgent, _ in REGIMES]
        )
        positions = x + (policy_index - 0.5) * width
        present = np.isfinite(heights)
        bars = ax_bar.bar(
            positions[present],
            heights[present],
            width,
            color=COLORS[policy],
            label=LABELS[policy],
        )
        for bar, value in zip(bars, heights[present], strict=True):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.09,
                f"{value:.1f}s",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=COLORS[policy],
            )

        for position, is_present in zip(positions, present, strict=True):
            if not is_present:
                ax_bar.text(
                    position,
                    2.2,
                    "pending",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666666",
                )

    ax_bar.set_yscale("log")
    ax_bar.set_ylim(1, 1100)
    ax_bar.set_xticks(x, [label for _, _, label in REGIMES], fontsize=8)
    ax_bar.set_ylabel("Urgent mean end-to-end TTFT (seconds, log scale)")
    ax_bar.set_title("(a) Urgent latency by cost regime", fontweight="bold")
    ax_bar.grid(axis="y", which="both", alpha=0.23)
    ax_bar.legend(frameon=False, fontsize=8)

    speedups = np.full((2, 2), np.nan)
    # Rows: urgent light/heavy. Columns: background light/heavy.
    for background_index, background in enumerate((8, 128)):
        for urgent_index, urgent in enumerate((8, 128)):
            fcfs = values[(background, urgent, "fcfs")]
            priority = values[(background, urgent, "priority")]
            if np.isfinite(fcfs) and np.isfinite(priority):
                speedups[urgent_index, background_index] = fcfs / priority

    masked = np.ma.masked_invalid(speedups)
    image = ax_heat.imshow(masked, cmap="YlGnBu", vmin=1, vmax=25)
    for urgent_index in range(2):
        for background_index in range(2):
            value = speedups[urgent_index, background_index]
            if np.isfinite(value):
                text = f"{value:.2f}$\\times$"
                color = "white" if value > 13 else "black"
            else:
                text = "Pending"
                color = "#666666"
            ax_heat.text(
                background_index,
                urgent_index,
                text,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
            )
    ax_heat.set_xticks((0, 1), ("Light (8)", "Heavy (128)"))
    ax_heat.set_yticks((0, 1), ("Light (8)", "Heavy (128)"))
    ax_heat.set_xlabel("Background frame budget")
    ax_heat.set_ylabel("Urgent frame budget")
    ax_heat.set_title("(b) End-to-end-priority speedup", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    colorbar.set_label("Speedup over engine-only priority")

    fig.suptitle(
        "Priority benefit depends primarily on queued background preparation cost",
        y=1.04,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Burst workload: 64 background requests at $t=0$, 16 urgent requests at "
        "$t=10$s, 4 preparation workers. Light=8 frames; heavy=128 frames.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(args.output)
    print(pdf)
    for background, urgent, _ in REGIMES:
        fcfs = values[(background, urgent, "fcfs")]
        priority = values[(background, urgent, "priority")]
        print(
            f"background={background} urgent={urgent} "
            f"fcfs={fcfs:.3f} priority={priority:.3f} "
            f"fcfs_n={counts[(background, urgent, 'fcfs')]} "
            f"priority_n={counts[(background, urgent, 'priority')]}"
        )


if __name__ == "__main__":
    main()
