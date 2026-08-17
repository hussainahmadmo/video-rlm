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

from conductor.experiments.scripts.build.build_config_selection_labels import (
    config_name,
    is_correct,
    load_result_rows,
    qid,
)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def label_row(source: dict[str, Any], *, budget2_ok: bool) -> dict[str, Any]:
    return {
        "dataset": source.get("dataset"),
        "qid": qid(source),
        "video_id": source.get("video_id"),
        "duration_s": source.get("duration_s"),
        "duration_bucket": source.get("duration_bucket"),
        "question": source.get("question"),
        "choices": source.get("choices"),
        "answer_idx": source.get("answer_idx"),
        "answer_label": source.get("answer_label"),
        "answer": source.get("answer"),
        "question_category": source.get("question_category"),
        "topic_category": source.get("topic_category"),
        "vimio_profile": source.get("vimio_profile"),
        "budget2_ok": budget2_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_result_rows(Path(args.fixed_results))
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_id = qid(row)
        if question_id != "None":
            by_qid[question_id].append(row)

    labels = []
    missing_budget2 = 0
    for question_rows in by_qid.values():
        budget2_rows = [
            row
            for row in question_rows
            if config_name(row) == "budget2"
        ]
        if not budget2_rows:
            missing_budget2 += 1
            continue
        labels.append(
            label_row(
                budget2_rows[0],
                budget2_ok=any(is_correct(row) for row in budget2_rows),
            )
        )

    write_jsonl(labels, Path(args.output))

    datasets = Counter(str(row.get("dataset") or "UNKNOWN") for row in labels)
    counts = Counter(bool(row["budget2_ok"]) for row in labels)
    print(f"fixed rows: {len(rows)}")
    print(f"questions: {len(by_qid)}")
    print(f"labels written: {len(labels)}")
    print(f"missing budget2 rows: {missing_budget2}")
    print("budget2 labels:")
    print(f"  ok: {counts[True]}")
    print(f"  not_ok: {counts[False]}")
    print("datasets:")
    for dataset, count in sorted(datasets.items()):
        print(f"  {dataset}: {count}")


if __name__ == "__main__":
    main()
