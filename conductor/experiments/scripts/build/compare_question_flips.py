#!/usr/bin/env python3
"""Compare per-question result flips across sweep output folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("results.part*.jsonl"))


def question_id(row: dict) -> str:
    for key in ("qid", "question_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def prediction(row: dict) -> str:
    for key in ("prediction", "final_answer", "answer", "pred"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def gold(row: dict) -> str:
    for key in ("gold", "answer_label", "label", "target"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def is_correct(row: dict) -> bool:
    for key in ("correct", "is_correct", "final_correct"):
        if key in row:
            return bool(row[key])

    pred = prediction(row).upper()
    target = gold(row).upper()
    return bool(pred and target and pred[0] == target[0])


def latency(row: dict) -> float:
    for key in (
        "wall_latency_s",
        "execution_latency_s",
        "latency_s",
        "agent_internal_latency_s",
    ):
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def load_folder(path: Path) -> tuple[dict[str, dict], int]:
    rows: dict[str, dict] = {}
    raw_rows = 0

    for file_path in result_files(path):
        with file_path.open("r", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw_rows += 1
                row = json.loads(line)
                qid = question_id(row)
                if not qid:
                    continue

                rows[qid] = {
                    "qid": qid,
                    "dataset": row.get("dataset", ""),
                    "video_id": row.get("video_id", ""),
                    "question": row.get("question", ""),
                    "gold": gold(row),
                    "prediction": prediction(row),
                    "correct": is_correct(row),
                    "latency_s": latency(row),
                    "file": file_path.name,
                }

    return rows, raw_rows


def folder_label(path: Path) -> str:
    return path.name if path.name else str(path)


def compare_pair(
    prev_label: str,
    prev_rows: dict[str, dict],
    next_label: str,
    next_rows: dict[str, dict],
    *,
    include_unchanged: bool,
) -> list[dict]:
    rows = []
    common_qids = sorted(set(prev_rows) & set(next_rows))

    for qid in common_qids:
        before = prev_rows[qid]
        after = next_rows[qid]

        before_correct = before["correct"]
        after_correct = after["correct"]
        pred_changed = before["prediction"] != after["prediction"]

        if before_correct and not after_correct:
            flip = "right_to_wrong"
        elif not before_correct and after_correct:
            flip = "wrong_to_right"
        elif pred_changed:
            flip = "prediction_changed_same_correctness"
        else:
            flip = "unchanged"

        if flip == "unchanged" and not include_unchanged:
            continue

        rows.append(
            {
                "from": prev_label,
                "to": next_label,
                "flip": flip,
                "dataset": after["dataset"] or before["dataset"],
                "qid": qid,
                "video_id": after["video_id"] or before["video_id"],
                "gold": after["gold"] or before["gold"],
                "before_prediction": before["prediction"],
                "before_correct": before_correct,
                "before_latency_s": f"{before['latency_s']:.2f}",
                "before_file": before["file"],
                "after_prediction": after["prediction"],
                "after_correct": after_correct,
                "after_latency_s": f"{after['latency_s']:.2f}",
                "after_file": after["file"],
                "question": after["question"] or before["question"],
            }
        )

    return rows


def print_summary(rows: list[dict], prev_total: int, next_total: int) -> None:
    counts: dict[str, int] = {}
    by_dataset: dict[tuple[str, str], int] = {}

    for row in rows:
        counts[row["flip"]] = counts.get(row["flip"], 0) + 1
        key = (row["dataset"], row["flip"])
        by_dataset[key] = by_dataset.get(key, 0) + 1

    print(f"  common changed/printed questions: {len(rows)}")
    print(f"  previous unique questions: {prev_total}")
    print(f"  next unique questions: {next_total}")

    for key in (
        "wrong_to_right",
        "right_to_wrong",
        "prediction_changed_same_correctness",
        "unchanged",
    ):
        if counts.get(key, 0):
            print(f"  {key}: {counts[key]}")

    datasets = sorted({dataset for dataset, _ in by_dataset})
    for dataset in datasets:
        parts = []
        for key in (
            "wrong_to_right",
            "right_to_wrong",
            "prediction_changed_same_correctness",
            "unchanged",
        ):
            value = by_dataset.get((dataset, key), 0)
            if value:
                parts.append(f"{key}={value}")
        if parts:
            print(f"  {dataset}: " + " ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folders",
        nargs="+",
        help="Sweep output folders or result jsonl files, in comparison order.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels matching folders.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Write all flip rows to this CSV file.",
    )
    parser.add_argument(
        "--include-unchanged",
        action="store_true",
        help="Also include questions with unchanged prediction/correctness.",
    )
    parser.add_argument(
        "--print-questions",
        action="store_true",
        help="Print exact flipped questions after the summaries.",
    )
    args = parser.parse_args()

    paths = [Path(value) for value in args.folders]
    labels = args.labels or [folder_label(path) for path in paths]

    if len(labels) != len(paths):
        raise SystemExit("--labels must have the same length as folders")

    loaded = []
    for label, path in zip(labels, paths):
        rows, raw_rows = load_folder(path)
        print(
            f"{label}: raw_rows={raw_rows} unique_questions={len(rows)} "
            f"path={path}"
        )
        loaded.append((label, rows))

    all_flips = []
    for (prev_label, prev_rows), (next_label, next_rows) in zip(
        loaded,
        loaded[1:],
    ):
        print(f"\n{prev_label} -> {next_label}")
        flips = compare_pair(
            prev_label,
            prev_rows,
            next_label,
            next_rows,
            include_unchanged=args.include_unchanged,
        )
        print_summary(flips, len(prev_rows), len(next_rows))
        all_flips.extend(flips)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "from",
            "to",
            "flip",
            "dataset",
            "qid",
            "video_id",
            "gold",
            "before_prediction",
            "before_correct",
            "before_latency_s",
            "before_file",
            "after_prediction",
            "after_correct",
            "after_latency_s",
            "after_file",
            "question",
        ]
        with args.out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_flips)
        print(f"\nwrote: {args.out_csv}")

    if args.print_questions:
        print("\nQuestions")
        for index, row in enumerate(all_flips, start=1):
            print(
                f"{index}. {row['from']} -> {row['to']} "
                f"{row['flip']} {row['dataset']} qid={row['qid']}"
            )
            print(
                f"   gold={row['gold']} before={row['before_prediction']} "
                f"after={row['after_prediction']}"
            )
            print(f"   question: {row['question']}")


if __name__ == "__main__":
    main()
