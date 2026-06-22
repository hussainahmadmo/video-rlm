import pandas as pd

datasets = {
    "EgoSchema": "egoschema/results/egoschema_result_with_duration.jsonl",
    "LVBench": "lvb/results/lvb_test.jsonl",
    "VRBench": "vrbench/results/vrbench_result_allconfigs_with_duration.jsonl",
}

configs = [
    "budget2",
    "budget8",
    "budget32",
    "scan05",
    "baseline",
]

rows = []

for dataset, path in datasets.items():

    df = pd.read_json(path, lines=True)

    for cfg in configs:

        sub = df[df["config_name"] == cfg]

        if len(sub) == 0:
            continue

        rows.append({
            "dataset": dataset,
            "config": cfg,
            "accuracy": sub["correct"].mean() * 100,
            "latency": sub["latency_s"].mean(),
        })

out = pd.DataFrame(rows)

out.to_csv(
    "cross_dataset_fixed_configs.csv",
    index=False
)

print(out)