from __future__ import annotations

import argparse
import json
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


def qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        required=True,
        help="Config-selection labels with oracle_correct.",
    )
    parser.add_argument(
        "--fixed-results",
        required=True,
        action="append",
        help="Original fixed-sweep results containing full dataset rows.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = load_jsonl(Path(args.labels))
    missed_qids = {
        qid(row)
        for row in labels
        if not bool(row.get("oracle_correct"))
    }

    seen = set()
    missed_rows = []
    for fixed_results in args.fixed_results:
        for row in load_jsonl(Path(fixed_results)):
            question_id = qid(row)
            if question_id not in missed_qids or question_id in seen:
                continue
            seen.add(question_id)
            missed_rows.append(
                {
                    key: row.get(key)
                    for key in (
                        "dataset",
                        "qid",
                        "id",
                        "question_id",
                        "video",
                        "video_id",
                        "question",
                        "choices",
                        "answer",
                        "answer_idx",
                        "answer_label",
                        "question_category",
                        "topic_category",
                        "vimio_profile",
                        "duration_s",
                        "duration_bucket",
                        "lvb_duration_bucket",
                    )
                    if key in row
                }
            )

    missing = missed_qids - seen
    write_jsonl(missed_rows, Path(args.output))
    print(f"missed labels: {len(missed_qids)}")
    print(f"wrote rows: {len(missed_rows)}")
    print(f"missing rows: {len(missing)}")
    if missing:
        for question_id in sorted(missing)[:20]:
            print(f"missing: {question_id}")


if __name__ == "__main__":
    main()
