from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conductor.profiler.resource_aware_selector import (
    Budget2GateSelector,
    train_budget2_gate,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(
    rows: list[dict[str, Any]],
    *,
    holdout_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if holdout_fraction <= 0:
        return rows, []

    train = []
    holdout = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset") or "UNKNOWN")].append(row)

    for dataset_rows in by_dataset.values():
        cutoff = max(1, int(round(len(dataset_rows) * holdout_fraction)))
        holdout.extend(dataset_rows[:cutoff])
        train.extend(dataset_rows[cutoff:])

    if not train:
        return rows, []
    return train, holdout


def evaluate(
    selector: Budget2GateSelector,
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    total = 0
    correct = 0
    predicted_ok = 0
    actual_ok = 0
    confusion = Counter()

    for row in rows:
        if "budget2_ok" not in row:
            continue
        label = bool(row["budget2_ok"])
        prob, _meta = selector.predict_probability(row)
        pred = prob >= threshold
        total += 1
        correct += int(pred == label)
        predicted_ok += int(pred)
        actual_ok += int(label)
        confusion[(label, pred)] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "threshold": threshold,
        "predicted_budget2_ok": predicted_ok,
        "actual_budget2_ok": actual_ok,
        "confusion": {
            "true_ok_pred_ok": confusion[(True, True)],
            "true_ok_pred_upgrade": confusion[(True, False)],
            "true_not_ok_pred_ok": confusion[(False, True)],
            "true_not_ok_pred_upgrade": confusion[(False, False)],
        },
    }


def print_eval(name: str, metrics: dict[str, Any]) -> None:
    print(
        f"{name}: n={metrics['total']} correct={metrics['correct']} "
        f"acc={100 * metrics['accuracy']:.2f}% "
        f"threshold={metrics['threshold']:.2f}"
    )
    print(
        "  predicted budget2_ok: "
        f"{metrics['predicted_budget2_ok']} / {metrics['total']}"
    )
    print(
        "  actual budget2_ok: "
        f"{metrics['actual_budget2_ok']} / {metrics['total']}"
    )
    print("  confusion:")
    for key, value in metrics["confusion"].items():
        print(f"    {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--metadata-out")
    args = parser.parse_args()

    rows = [
        row
        for row in load_jsonl(Path(args.labels))
        if "budget2_ok" in row
    ]
    if not rows:
        raise SystemExit("no rows contain budget2_ok")

    train_rows, holdout_rows = split_rows(
        rows,
        holdout_fraction=args.holdout_fraction,
    )
    selector = train_budget2_gate(
        train_rows,
        k_neighbors=args.k_neighbors,
    )
    selector.save(args.model_out)

    train_metrics = evaluate(
        selector,
        train_rows,
        threshold=args.threshold,
    )
    holdout_metrics = evaluate(
        selector,
        holdout_rows,
        threshold=args.threshold,
    )

    counts = Counter(bool(row["budget2_ok"]) for row in train_rows)
    print(f"labels: {args.labels}")
    print(f"model: {args.model_out}")
    print(f"train labels: ok={counts[True]} not_ok={counts[False]}")
    print_eval("train", train_metrics)
    if holdout_rows:
        print_eval("holdout", holdout_metrics)

    metadata = {
        "labels": str(args.labels),
        "model_out": str(args.model_out),
        "holdout_fraction": args.holdout_fraction,
        "k_neighbors": args.k_neighbors,
        "threshold": args.threshold,
        "training_examples": len(train_rows),
        "holdout_examples": len(holdout_rows),
        "train": train_metrics,
        "holdout": holdout_metrics,
        "model": selector.model,
    }
    if args.metadata_out:
        path = Path(args.metadata_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
