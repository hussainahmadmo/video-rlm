from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("conductor/experiments/consistent_eval")
OUT_PNG = ROOT / "junchen_oracle_tradeoff.png"
OUT_PDF = ROOT / "junchen_oracle_tradeoff.pdf"
OUT_CSV = ROOT / "junchen_oracle_tradeoff_points.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return str(value).lower() == "true"


def write_points(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_points() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fixed = read_csv(ROOT / "fixed_config_summary.csv")
    macro = read_csv(ROOT / "vimio_vs_fixed_summary.csv")
    vimio = read_csv(ROOT / "vimio_summary.csv")
    oracle = read_csv(ROOT / "oracle_summary.csv")
    calibrated = read_csv(
        ROOT / "vimio_rerun_oracle_calibrated_summary.csv"
    )
    robust = read_csv(ROOT / "vimio_rerun_robust_summary.csv")
    robust_comparison = read_csv(ROOT / "vimio_robust_comparison_summary.csv")

    macro_points = []

    for row in macro:
        if not str(row["policy"]).startswith("fixed_"):
            continue
        macro_points.append({
            "scope": "Macro",
            "method": "Fixed configs",
            "label": row["config"],
            "accuracy_pct": float(row["accuracy_pct_macro"]),
            "latency_s": float(row["latency_mean_macro"]),
            "n": int(row["n_total"]),
            "is_pareto": as_bool(row["is_macro_pareto"]),
        })

    old_row = next(row for row in macro if row["policy"] == "VIMIO")
    macro_points.append({
        "scope": "Macro",
        "method": "VIMIO old",
        "label": "VIMIO old",
        "accuracy_pct": float(old_row["accuracy_pct_macro"]),
        "latency_s": float(old_row["latency_mean_macro"]),
        "n": int(old_row["n_total"]),
        "is_pareto": False,
    })

    cal_row = next(row for row in calibrated if row["dataset"] == "macro")
    macro_points.append({
        "scope": "Macro",
        "method": "VIMIO calibrated",
        "label": "VIMIO calibrated",
        "accuracy_pct": 100 * float(cal_row["accuracy"]),
        "latency_s": float(cal_row["avg_latency_s"]),
        "n": int(cal_row["n"]),
        "is_pareto": False,
    })

    robust_macro = next(
        row
        for row in robust_comparison
        if row["method"] == "VIMIO_robust_rerun"
    )
    macro_points.append({
        "scope": "Macro",
        "method": "VIMIO robust",
        "label": "VIMIO robust",
        "accuracy_pct": float(robust_macro["accuracy_pct_macro"]),
        "latency_s": float(robust_macro["latency_mean_macro"]),
        "n": int(robust_macro["n_total"]),
        "is_pareto": False,
    })

    oracle_comparison = read_csv(ROOT / "vimio_calibrated_comparison_summary.csv")
    oracle_macro = next(
        row
        for row in oracle_comparison
        if row["method"] == "Oracle" and row["dataset"] == "macro"
    )
    macro_points.append({
        "scope": "Macro",
        "method": "Oracle",
        "label": "Oracle",
        "accuracy_pct": float(oracle_macro["accuracy_pct"]),
        "latency_s": float(oracle_macro["avg_latency_s"]),
        "n": int(oracle_macro["n"]),
        "is_pareto": True,
    })

    dataset_points = []
    for row in fixed:
        dataset_points.append({
            "scope": row["dataset"],
            "method": "Fixed configs",
            "label": row["config"],
            "accuracy_pct": float(row["accuracy_pct"]),
            "latency_s": float(row["latency_mean"]),
            "n": int(row["n"]),
            "is_pareto": as_bool(row["is_pareto"]),
        })

    for row in vimio:
        dataset_points.append({
            "scope": row["dataset"],
            "method": "VIMIO old",
            "label": "VIMIO old",
            "accuracy_pct": float(row["accuracy_pct"]),
            "latency_s": float(row["latency_mean"]),
            "n": int(row["n"]),
            "is_pareto": False,
        })

    for row in calibrated:
        if row["dataset"] in {"macro", "weighted"}:
            continue
        dataset_points.append({
            "scope": row["dataset"],
            "method": "VIMIO calibrated",
            "label": "VIMIO calibrated",
            "accuracy_pct": 100 * float(row["accuracy"]),
            "latency_s": float(row["avg_latency_s"]),
            "n": int(row["n"]),
            "is_pareto": False,
        })

    for row in robust:
        dataset_points.append({
            "scope": row["dataset"],
            "method": "VIMIO robust",
            "label": "VIMIO robust",
            "accuracy_pct": float(row["accuracy_pct"]),
            "latency_s": float(row["avg_latency_s"]),
            "n": int(row["n"]),
            "is_pareto": False,
        })

    for row in oracle:
        dataset_points.append({
            "scope": row["dataset"],
            "method": "Oracle",
            "label": "Oracle",
            "accuracy_pct": float(row["accuracy_pct"]),
            "latency_s": float(row["latency_mean"]),
            "n": int(row["n"]),
            "is_pareto": True,
        })

    return (
        macro_points,
        dataset_points,
    )


def values(rows: list[dict[str, object]], key: str) -> list[object]:
    return [row[key] for row in rows]


def draw_panel(
    ax,
    points: list[dict[str, object]],
    title: str,
) -> None:
    fixed = [row for row in points if row["method"] == "Fixed configs"]
    dominated = [row for row in fixed if not row["is_pareto"]]
    pareto = sorted(
        [row for row in fixed if row["is_pareto"]],
        key=lambda row: float(row["latency_s"]),
    )

    ax.scatter(
        values(dominated, "latency_s"),
        values(dominated, "accuracy_pct"),
        s=35,
        color="#9ca3af",
        alpha=0.35,
        label="Fixed configs",
    )

    ax.scatter(
        values(pareto, "latency_s"),
        values(pareto, "accuracy_pct"),
        s=65,
        color="#4b5563",
        alpha=0.9,
        label="Fixed Pareto",
    )

    if len(pareto) >= 2:
        ax.plot(
            values(pareto, "latency_s"),
            values(pareto, "accuracy_pct"),
            color="#4b5563",
            linewidth=1.5,
            alpha=0.8,
        )

    styles = {
        "VIMIO old": ("#2563eb", "o", 95),
        "VIMIO calibrated": ("#059669", "D", 95),
        "VIMIO robust": ("#7c3aed", "P", 115),
        "Oracle": ("#dc2626", "*", 180),
    }
    label_offsets = {
        "VIMIO old": (6, 6),
        "VIMIO calibrated": (6, -2),
        "VIMIO robust": (6, 10),
        "Oracle": (6, 6),
    }

    for method, (color, marker, size) in styles.items():
        sub = [row for row in points if row["method"] == method]
        if not sub:
            continue
        ax.scatter(
            values(sub, "latency_s"),
            values(sub, "accuracy_pct"),
            s=size,
            marker=marker,
            color=color,
            edgecolors="black" if method == "Oracle" else color,
            linewidths=0.7,
            label=method,
            zorder=5,
        )
        for row in sub:
            ax.annotate(
                row["label"],
                (row["latency_s"], row["accuracy_pct"]),
                textcoords="offset points",
                xytext=label_offsets[method],
                fontsize=8,
                color=color,
                weight="bold" if method == "Oracle" else "normal",
            )

    ax.set_xscale("log")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("Mean latency / question (s, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.25)


def main() -> None:
    macro, datasets = load_points()
    all_points = macro + datasets
    write_points(all_points, OUT_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("Macro average", macro),
        (
            "EgoSchema",
            [row for row in datasets if row["scope"] == "EgoSchema"],
        ),
        (
            "LVBench",
            [row for row in datasets if row["scope"] == "LVBench"],
        ),
        (
            "VRBench",
            [row for row in datasets if row["scope"] == "VRBench"],
        ),
    ]

    for ax, (title, points) in zip(axes.ravel(), panels):
        draw_panel(ax, points, title)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        "VIMIO vs fixed policies and oracle upper bound",
        y=1.035,
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT_PNG)
    print("saved", OUT_PDF)
    print("saved", OUT_CSV)


if __name__ == "__main__":
    main()
