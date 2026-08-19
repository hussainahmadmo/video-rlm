#!/usr/bin/env python3
"""Build deterministic, dataset-balanced, video-disjoint JSONL splits.

Questions from a video are assigned to at most one split. If a video has more
questions than a split still needs, a deterministic subset is retained and the
remaining questions from that video are discarded to preserve exact counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATASETS = ("EgoSchema", "LVBench", "NExT-QA", "STAR", "VRBench")
ALIASES = {
    "egoschema": "EgoSchema",
    "lvbench": "LVBench",
    "nextqa": "NExT-QA",
    "next-qa": "NExT-QA",
    "nexT-qa": "NExT-QA",
    "star": "STAR",
    "vrbench": "VRBench",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            row["_pool_source"] = str(path.resolve())
            rows.append(row)
    return rows


def canonical_dataset(row: dict) -> str:
    raw = str(row.get("dataset") or row.get("source_dataset") or "").strip()
    lowered = raw.lower().replace("_", "-")
    for alias, canonical in ALIASES.items():
        if lowered == alias.lower():
            return canonical
    return raw


def question_id(row: dict) -> str:
    value = row.get("qid") or row.get("question_id") or row.get("id")
    if value is None:
        raise ValueError("row has no qid, question_id, or id")
    return str(value)


def video_id(row: dict) -> str:
    value = row.get("video_id") or row.get("video") or row.get("video_path")
    if value is None:
        raise ValueError("row has no video_id, video, or video_path")
    return str(value)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--train-per-dataset", type=int, default=250)
    parser.add_argument("--val-per-dataset", type=int, default=50)
    parser.add_argument("--test-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    requested = set(args.datasets)
    by_question: dict[tuple[str, str, str], dict] = {}
    duplicate_questions = 0
    for path in args.input:
        for row in read_jsonl(path):
            dataset = canonical_dataset(row)
            if dataset not in requested:
                continue
            row["dataset"] = dataset
            row["qid"] = question_id(row)
            row["video_id"] = video_id(row)
            key = (dataset, row["video_id"], row["qid"])
            if key in by_question:
                duplicate_questions += 1
                continue
            by_question[key] = row

    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in by_question.values():
        groups[row["dataset"]][row["video_id"]].append(row)

    targets = {
        "train": args.train_per_dataset,
        "val": args.val_per_dataset,
        "test": args.test_per_dataset,
    }
    total_needed = sum(targets.values())
    availability = {
        dataset: {
            "questions": sum(len(rows) for rows in groups[dataset].values()),
            "videos": len(groups[dataset]),
        }
        for dataset in args.datasets
    }
    insufficient = {
        dataset: counts
        for dataset, counts in availability.items()
        if counts["questions"] < total_needed
    }
    if insufficient:
        details = ", ".join(
            f"{name}={counts['questions']} questions/{counts['videos']} videos"
            for name, counts in insufficient.items()
        )
        raise SystemExit(
            f"Insufficient candidates; need {total_needed} questions per dataset: {details}"
        )

    split_rows: dict[str, list[dict]] = {name: [] for name in targets}
    # Test is allocated first so the held-out set does not receive leftovers.
    allocation_order = ("test", "val", "train")
    for dataset in args.datasets:
        rng = random.Random(f"{args.seed}:{dataset}")
        video_groups = list(groups[dataset].items())
        for _, rows in video_groups:
            rows.sort(key=lambda row: row["qid"])
            rng.shuffle(rows)
        rng.shuffle(video_groups)

        cursor = 0
        for split in allocation_order:
            remaining = targets[split]
            while remaining and cursor < len(video_groups):
                _, rows = video_groups[cursor]
                cursor += 1
                selected = rows[:remaining]
                for source_row in selected:
                    row = dict(source_row)
                    row["split"] = split
                    split_rows[split].append(row)
                remaining -= len(selected)
            if remaining:
                raise SystemExit(
                    f"Could not fill {dataset}/{split}; missing {remaining} questions after "
                    "enforcing video-disjoint allocation. Add more unique videos/questions."
                )

    qid_sets = {
        split: {(row["dataset"], row["video_id"], row["qid"]) for row in rows}
        for split, rows in split_rows.items()
    }
    video_sets = {
        split: {(row["dataset"], row["video_id"]) for row in rows}
        for split, rows in split_rows.items()
    }
    names = list(split_rows)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if qid_sets[left] & qid_sets[right]:
                raise RuntimeError(f"question leakage between {left} and {right}")
            if video_sets[left] & video_sets[right]:
                raise RuntimeError(f"video leakage between {left} and {right}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        "train": f"large_train_{len(split_rows['train'])}.jsonl",
        "val": f"large_val_{len(split_rows['val'])}.jsonl",
        "test": f"large_heldout_{len(split_rows['test'])}.jsonl",
    }
    outputs = {}
    for split, filename in output_names.items():
        rows = sorted(split_rows[split], key=lambda row: (row["dataset"], row["qid"]))
        path = args.output_dir / filename
        write_jsonl(path, rows)
        outputs[split] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "questions": len(rows),
            "videos": len(video_sets[split]),
            "by_dataset": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        }

    manifest = {
        "seed": args.seed,
        "split_unit": ["dataset", "video_id"],
        "inputs": [str(path.resolve()) for path in args.input],
        "datasets": args.datasets,
        "targets_per_dataset": targets,
        "availability_after_question_dedup": availability,
        "duplicate_questions_removed": duplicate_questions,
        "cross_split_question_overlap": 0,
        "cross_split_video_overlap": 0,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
