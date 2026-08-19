#!/usr/bin/env python3
"""Build a question-disjoint contention subset from prepared VLM jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> tuple[str, str]:
    dataset = str(row.get("dataset") or "NExT-QA")
    qid = str(row.get("qid") or row.get("question_id") or row.get("id"))
    return dataset, qid


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--adaptive-jobs", type=Path, required=True)
    parser.add_argument("--baseline-jobs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    training_rows = load_jsonl(args.training_labels)
    adaptive_rows = load_jsonl(args.adaptive_jobs)
    baseline_rows = load_jsonl(args.baseline_jobs)

    training_keys = {key(row) for row in training_rows}
    adaptive_by_key = {key(row): row for row in adaptive_rows}
    baseline_by_key = {key(row): row for row in baseline_rows}
    common_keys = set(adaptive_by_key) & set(baseline_by_key)
    heldout_keys = sorted(common_keys - training_keys)

    if set(heldout_keys) & training_keys:
        raise RuntimeError("held-out split overlaps selector training data")
    if len(adaptive_by_key) != len(adaptive_rows):
        raise RuntimeError("adaptive jobs contain duplicate dataset/qid keys")
    if len(baseline_by_key) != len(baseline_rows):
        raise RuntimeError("baseline jobs contain duplicate dataset/qid keys")

    adaptive_heldout = [adaptive_by_key[item] for item in heldout_keys]
    baseline_heldout = [baseline_by_key[item] for item in heldout_keys]
    dataset_rows = [
        {
            field: adaptive_by_key[item].get(field)
            for field in (
                "dataset",
                "qid",
                "video_id",
                "video",
                "question",
                "choices",
                "answer_idx",
                "answer_label",
                "answer",
            )
        }
        for item in heldout_keys
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "heldout_dataset.jsonl", dataset_rows)
    write_jsonl(args.output_dir / "adaptive_jobs.jsonl", adaptive_heldout)
    write_jsonl(args.output_dir / "budget32_jobs.jsonl", baseline_heldout)

    per_dataset: dict[str, int] = {}
    for dataset, _qid in heldout_keys:
        per_dataset[dataset] = per_dataset.get(dataset, 0) + 1

    manifest = {
        "split_key": ["dataset", "qid"],
        "training_labels": str(args.training_labels.resolve()),
        "adaptive_jobs_source": str(args.adaptive_jobs.resolve()),
        "baseline_jobs_source": str(args.baseline_jobs.resolve()),
        "training_examples": len(training_keys),
        "source_common_examples": len(common_keys),
        "heldout_examples": len(heldout_keys),
        "training_overlap": len(set(heldout_keys) & training_keys),
        "heldout_by_dataset": per_dataset,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
