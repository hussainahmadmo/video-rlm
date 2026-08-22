from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def dataset_name(row: dict[str, Any]) -> str:
    return str(
        row.get("dataset")
        or row.get("topic_category")
        or row.get("source_dataset")
        or "UNKNOWN"
    )


def qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def dedupe_by_qid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        key = qid(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create dataset-balanced train/eval splits from config selector "
            "label JSONL files."
        )
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument(
        "--train-per-dataset",
        type=int,
        required=True,
        help="Number of training labels to sample from each dataset.",
    )
    parser.add_argument(
        "--eval-per-dataset",
        type=int,
        required=True,
        help="Number of evaluation labels to sample from each dataset.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help=(
            "Keep datasets with fewer than train+eval examples by taking as "
            "many as possible. Without this, short datasets are skipped."
        ),
    )
    args = parser.parse_args()

    rows = dedupe_by_qid(load_jsonl(Path(args.labels)))
    rng = random.Random(args.seed)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[dataset_name(row)].append(row)

    train = []
    eval_rows = []
    skipped = {}
    summary = {}
    required = args.train_per_dataset + args.eval_per_dataset

    for dataset, dataset_rows in sorted(by_dataset.items()):
        shuffled = list(dataset_rows)
        rng.shuffle(shuffled)

        if len(shuffled) < required and not args.allow_short:
            skipped[dataset] = {
                "available": len(shuffled),
                "required": required,
            }
            continue

        eval_count = min(args.eval_per_dataset, len(shuffled))
        remaining = len(shuffled) - eval_count
        train_count = min(args.train_per_dataset, remaining)

        eval_part = shuffled[:eval_count]
        train_part = shuffled[eval_count : eval_count + train_count]
        eval_rows.extend(eval_part)
        train.extend(train_part)
        summary[dataset] = {
            "available": len(shuffled),
            "train": len(train_part),
            "eval": len(eval_part),
        }

    rng.shuffle(train)
    rng.shuffle(eval_rows)

    write_jsonl(train, Path(args.train_output))
    write_jsonl(eval_rows, Path(args.eval_output))

    print(f"labels: {args.labels}")
    print(f"input rows: {len(rows)}")
    print(f"train output: {args.train_output}")
    print(f"eval output: {args.eval_output}")
    print(f"train rows: {len(train)}")
    print(f"eval rows: {len(eval_rows)}")
    print("input by dataset:")
    for dataset, count in Counter(dataset_name(row) for row in rows).most_common():
        print(f"  {dataset}: {count}")
    print("split by dataset:")
    for dataset, item in summary.items():
        print(
            f"  {dataset}: available={item['available']} "
            f"train={item['train']} eval={item['eval']}"
        )
    if skipped:
        print("skipped short datasets:")
        for dataset, item in skipped.items():
            print(
                f"  {dataset}: available={item['available']} "
                f"required={item['required']}"
            )


if __name__ == "__main__":
    main()
