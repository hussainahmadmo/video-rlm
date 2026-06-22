import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATASETS = {
    "EgoSchema": "egoschema/results/egoschema_result_summary_with_pareto.csv",
    "VRBench": "vrbench/results/vrbench_result_allconfigs_summary_with_pareto.csv",
    "LVBench": "lvb/results/lvb_knob_sensitivity_filtered_summary_with_pareto.csv",
}

plt.figure(figsize=(10, 6))
found_any = False

for dataset_name, path in DATASETS.items():
    path = Path(path)
    if not path.exists():
        print(f"Skipping missing file: {path}")
        continue

    # df = pd.read_csv(path)

    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    if df["pareto"].dtype == object:
        df["pareto"] = df["pareto"].astype(str).str.lower().eq("true")

    pareto = df[df["pareto"]].sort_values("latency_mean")

    print(f"{dataset_name}: loaded {len(df)} configs, {len(pareto)} Pareto configs")

    plt.scatter(
        df["latency_mean"],
        df["accuracy_pct"],
        alpha=0.35,
        label=f"{dataset_name} all",
    )

    plt.plot(
        pareto["latency_mean"],
        pareto["accuracy_pct"],
        marker="o",
        linewidth=2,
        label=f"{dataset_name} Pareto",
    )

    for _, row in pareto.iterrows():
        plt.annotate(
            row["config_name"],
            (row["latency_mean"], row["accuracy_pct"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    found_any = True

if not found_any:
    raise RuntimeError("No files found.")

plt.xscale("log")
plt.xlabel("Mean latency per query (s, log scale)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy–Latency Pareto Frontier Across Benchmarks")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.savefig("all_datasets_accuracy_latency_pareto.png", dpi=300, bbox_inches="tight")
print("Saved all_datasets_accuracy_latency_pareto.png")