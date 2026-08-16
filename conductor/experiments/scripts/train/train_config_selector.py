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
    TIER_ORDER,
    ResourceAwareSelector,
    train_nearest_centroid,
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
    selector: ResourceAwareSelector,
    rows: list[dict[str, Any]],
    *,
    label_key: str,
) -> dict[str, Any]:
    total = 0
    correct = 0
    labels = Counter()
    predictions = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        label = str(row.get(label_key) or "")
        if label not in TIER_ORDER:
            continue
        pred, _meta = selector.predict_base_config(row)
        total += 1
        correct += int(pred == label)
        labels[label] += 1
        predictions[pred] += 1
        confusion[label][pred] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "labels": dict(labels),
        "predictions": dict(predictions),
        "confusion": {
            label: dict(counts)
            for label, counts in sorted(confusion.items())
        },
    }


def print_eval(name: str, metrics: dict[str, Any]) -> None:
    total = int(metrics["total"])
    correct = int(metrics["correct"])
    accuracy = float(metrics["accuracy"])
    print(f"{name}: n={total} correct={correct} acc={100 * accuracy:.2f}%")
    print("  labels:")
    for label in TIER_ORDER:
        count = metrics["labels"].get(label, 0)
        if count:
            print(f"    {label}: {count}")
    print("  predictions:")
    for label in TIER_ORDER:
        count = metrics["predictions"].get(label, 0)
        if count:
            print(f"    {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--label-key", default="label_config")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Optional per-dataset holdout fraction for a quick sanity check.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=7,
        help="Number of nearest labeled questions to vote over.",
    )
    parser.add_argument(
        "--metadata-out",
        help="Optional JSON summary path for training/eval metadata.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.labels))
    rows = [
        row
        for row in rows
        if row.get(args.label_key) in TIER_ORDER
    ]
    if not rows:
        raise SystemExit(
            f"no rows have {args.label_key} in {', '.join(TIER_ORDER)}"
        )

    train_rows, holdout_rows = split_rows(
        rows,
        holdout_fraction=args.holdout_fraction,
    )
    selector = train_nearest_centroid(
        train_rows,
        label_key=args.label_key,
        k_neighbors=args.k_neighbors,
    )
    selector.save(args.model_out)

    train_metrics = evaluate(
        selector,
        train_rows,
        label_key=args.label_key,
    )
    holdout_metrics = evaluate(
        selector,
        holdout_rows,
        label_key=args.label_key,
    )

    print(f"labels: {args.labels}")
    print(f"model: {args.model_out}")
    print_eval("train", train_metrics)
    if holdout_rows:
        print_eval("holdout", holdout_metrics)

    metadata = {
        "labels": str(args.labels),
        "model_out": str(args.model_out),
        "label_key": args.label_key,
        "holdout_fraction": args.holdout_fraction,
        "k_neighbors": args.k_neighbors,
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
