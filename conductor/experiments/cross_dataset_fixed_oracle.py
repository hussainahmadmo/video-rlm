import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame([
    ["budget2",40.83,2.03],
    ["budget8",50.27,2.42],
    ["budget32",54.42,3.54],
    ["scan05",56.02,38.14],
    ["baseline",55.12,48.16],
    ["oracle",75.15,5.78],
], columns=["config","accuracy","latency"])

plt.figure(figsize=(8,5))

for _, r in df.iterrows():
    plt.scatter(r["latency"], r["accuracy"], s=120)

    plt.annotate(
        r["config"],
        (r["latency"], r["accuracy"]),
        xytext=(5,5),
        textcoords="offset points"
    )

plt.xlabel("Average Latency (s)")
plt.ylabel("Average Accuracy (%)")
plt.title("Cross-Dataset Fixed Policies vs Oracle")
plt.grid(alpha=0.3)

plt.savefig(
    "plots/cross_dataset_fixed_vs_oracle.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()