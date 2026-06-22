# build_average_fixed_configs.py

import pandas as pd

ego = pd.read_csv("egoschema_summary.csv")
lvb = pd.read_csv("lvb_summary.csv")
vr  = pd.read_csv("vrbench_summary.csv")

ego["dataset"] = "EgoSchema"
lvb["dataset"] = "LVBench"
vr["dataset"] = "VRBench"

df = pd.concat([ego, lvb, vr], ignore_index=True)

# configs that appear in all datasets
common = (
    df.groupby("config_name")["dataset"]
      .nunique()
)

common = common[common == 3].index

df = df[df["config_name"].isin(common)]

summary = (
    df.groupby("config_name")
      .agg(
          avg_accuracy=("accuracy", "mean"),
          avg_latency=("latency", "mean"),
          min_accuracy=("accuracy", "min"),
          max_accuracy=("accuracy", "max"),
      )
      .sort_values(
          ["avg_accuracy", "avg_latency"],
          ascending=[False, True]
      )
)

print(summary)

summary.to_csv(
    "average_fixed_configs.csv"
)

print("\nSaved average_fixed_configs.csv")