#!/usr/bin/env python3
"""Plot the pooled accuracy-latency Pareto frontier for fixed policies only."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("conductor/experiments/consistent_eval")
SOURCE = ROOT / "fixed_config_summary.csv"
OUT_CSV = ROOT / "all_videos_fixed_pareto.csv"
OUT_PNG = ROOT / "all_videos_fixed_pareto.png"
OUT_PDF = ROOT / "all_videos_fixed_pareto.pdf"


def main() -> None:
    with SOURCE.open(newline="") as handle:
        source = list(csv.DictReader(handle))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in source:
        grouped.setdefault(row["config"], []).append(row)

    points = []
    for config, rows in grouped.items():
        n = sum(int(row["n"]) for row in rows)
        accuracy = sum(float(row["accuracy_pct"]) * int(row["n"]) for row in rows) / n
        latency = sum(float(row["latency_mean"]) * int(row["n"]) for row in rows) / n
        points.append({
            "config": config,
            "accuracy_pct": accuracy,
            "latency_mean_s": latency,
            "n_questions": n,
            "datasets": len(rows),
        })

    for point in points:
        point["is_fixed_pareto"] = not any(
            other["latency_mean_s"] <= point["latency_mean_s"]
            and other["accuracy_pct"] >= point["accuracy_pct"]
            and (
                other["latency_mean_s"] < point["latency_mean_s"]
                or other["accuracy_pct"] > point["accuracy_pct"]
            )
            for other in points
        )

    points.sort(key=lambda row: row["latency_mean_s"])
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    pareto = [row for row in points if row["is_fixed_pareto"]]
    dominated = [row for row in points if not row["is_fixed_pareto"]]

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.scatter(
        [row["latency_mean_s"] for row in dominated],
        [row["accuracy_pct"] for row in dominated],
        color="#9ca3af", alpha=0.48, s=55, label="Dominated fixed config",
    )
    ax.plot(
        [row["latency_mean_s"] for row in pareto],
        [row["accuracy_pct"] for row in pareto],
        color="#2563eb", linewidth=2.2, marker="o", markersize=7,
        label="Fixed-policy Pareto frontier", zorder=4,
    )

    highlights = {"budget2": "#059669", "budget32": "#dc2626"}
    for row in points:
        color = highlights.get(row["config"], "#374151")
        if row["config"] in highlights:
            ax.scatter(
                row["latency_mean_s"], row["accuracy_pct"],
                color=color, s=115, edgecolor="black", linewidth=0.7, zorder=6,
                label=row["config"],
            )
        if row["is_fixed_pareto"] or row["config"] in {"budget2", "budget8", "budget32", "budget64"}:
            ax.annotate(
                row["config"],
                (row["latency_mean_s"], row["accuracy_pct"]),
                xytext=(6, 6), textcoords="offset points", fontsize=8,
                color=color,
                weight="bold" if row["config"] in highlights else "normal",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Mean end-to-end latency per question (s, log scale)")
    ax.set_ylabel("Question-weighted accuracy (%)")
    ax.set_title("Fixed-policy Pareto frontier across 146 videos / 177 questions")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("saved", OUT_CSV)
    print("saved", OUT_PNG)
    print("saved", OUT_PDF)
    for row in pareto:
        print(
            f"{row['config']}: accuracy={row['accuracy_pct']:.2f}% "
            f"latency={row['latency_mean_s']:.3f}s n={row['n_questions']}"
        )


if __name__ == "__main__":
    main()
