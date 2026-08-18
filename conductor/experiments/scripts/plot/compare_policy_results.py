#!/usr/bin/env python3
"""Compare native-uniform, fixed-budget, and adaptive result JSONLs."""

import argparse
import json
import statistics
from pathlib import Path


def load_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def qid(row):
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def key(row):
    return str(row.get("dataset") or ""), qid(row)


def mappings(values, option):
    output = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} requires LABEL=VALUE: {value}")
        label, item = value.split("=", 1)
        if not label or not item or label in output:
            raise SystemExit(f"invalid or duplicate {option}: {value}")
        output[label] = item
    return output


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def metrics(rows, wall_s=None):
    valid = [row for row in rows if not row.get("error")]
    latencies = [
        float(row["latency_s"]) for row in valid
        if row.get("latency_s") is not None
    ]
    correct = sum(bool(row.get("correct")) for row in valid)
    return {
        "examples": len(rows), "valid": len(valid),
        "errors": len(rows) - len(valid), "correct": correct,
        "accuracy": correct / len(valid) if valid else None,
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "p95_latency_s": percentile(latencies, 0.95),
        "wall_time_s": wall_s,
        "throughput_qps": len(valid) / wall_s if wall_s else None,
    }


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", action="append", required=True,
        help="LABEL=/path/results.jsonl; repeat for each method",
    )
    parser.add_argument(
        "--config", action="append", default=[],
        help="LABEL=config_name filter for a multi-policy result file",
    )
    parser.add_argument(
        "--wall", action="append", default=[],
        help="LABEL=/path/wall_seconds.txt",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    paths = mappings(args.result, "--result")
    configs = mappings(args.config, "--config")
    walls = mappings(args.wall, "--wall")
    unknown = (set(configs) | set(walls)) - set(paths)
    if unknown:
        raise SystemExit(f"labels missing from --result: {sorted(unknown)}")

    indexed = {}
    source_rows = {}
    for label, path in paths.items():
        rows = load_jsonl(path)
        source_rows[label] = len(rows)
        if label in configs:
            rows = [
                row for row in rows
                if row.get("config_name") == configs[label]
            ]
        by_key = {}
        for row in rows:
            item_key = key(row)
            if item_key in by_key:
                raise SystemExit(
                    f"duplicate key for {label}; add --config: {item_key}"
                )
            by_key[item_key] = row
        indexed[label] = by_key

    common = set.intersection(*(set(rows) for rows in indexed.values()))
    if not common:
        raise SystemExit("no matched questions across methods")
    wall_values = {
        label: float(Path(path).read_text().strip())
        for label, path in walls.items()
    }

    labels = list(indexed)
    first = labels[0]
    summary = {
        "common_examples": len(common),
        "common_key": ["dataset", "qid"],
        "methods": {}, "paired_vs_first": {},
    }
    for label in labels:
        rows = [indexed[label][item] for item in sorted(common)]
        item = metrics(rows, wall_values.get(label))
        item["source_rows"] = source_rows[label]
        item["config_filter"] = configs.get(label)
        summary["methods"][label] = item

    baseline = indexed[first]
    for label in labels[1:]:
        other = indexed[label]
        counts = {
            "both_correct": 0, "baseline_only_correct": 0,
            "compared_only_correct": 0, "both_wrong": 0,
        }
        ratios = []
        for item in common:
            left = bool(baseline[item].get("correct"))
            right = bool(other[item].get("correct"))
            if left and right:
                counts["both_correct"] += 1
            elif left:
                counts["baseline_only_correct"] += 1
            elif right:
                counts["compared_only_correct"] += 1
            else:
                counts["both_wrong"] += 1
            left_s = float(baseline[item].get("latency_s") or 0)
            right_s = float(other[item].get("latency_s") or 0)
            if left_s > 0 and right_s > 0:
                ratios.append(left_s / right_s)
        left_m = summary["methods"][first]
        right_m = summary["methods"][label]
        summary["paired_vs_first"][label] = {
            "baseline": first,
            "accuracy_delta_pp": 100 * (
                right_m["accuracy"] - left_m["accuracy"]
            ),
            "throughput_gain": (
                right_m["throughput_qps"] / left_m["throughput_qps"]
                if right_m["throughput_qps"] and left_m["throughput_qps"]
                else None
            ),
            "median_per_query_latency_speedup": (
                statistics.median(ratios) if ratios else None
            ),
            **counts,
        }

    print(f"matched questions: {len(common)}")
    print("method\tn\taccuracy\tmean_s\tmedian_s\tp95_s\tqps")
    for label in labels:
        item = summary["methods"][label]
        print(
            f"{label}\t{item['valid']}\t{fmt(item['accuracy'])}\t"
            f"{fmt(item['mean_latency_s'])}\t"
            f"{fmt(item['median_latency_s'])}\t"
            f"{fmt(item['p95_latency_s'])}\t"
            f"{fmt(item['throughput_qps'])}"
        )
    print(f"\npaired relative to {first}:")
    for label, item in summary["paired_vs_first"].items():
        print(
            f"{label}: accuracy_delta_pp={fmt(item['accuracy_delta_pp'], 2)} "
            f"throughput_gain={fmt(item['throughput_gain'], 3)} "
            f"median_latency_speedup="
            f"{fmt(item['median_per_query_latency_speedup'], 3)}"
        )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
