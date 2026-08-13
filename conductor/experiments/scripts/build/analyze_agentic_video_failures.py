#!/usr/bin/env python3
"""Summarize agentic result correctness by video and failure type."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ACTION_KEYS = (
    "SEARCH_LOCAL",
    "GLOBAL_SCAN",
    "COUNT_EVENTS",
    "TEMPORAL_SEARCH",
    "INCREASE_DENSITY",
    "VERIFY_DETAIL",
    "IMAGE_CAPTION",
    "ZOOM_CAPTION",
    "OBJECT_DETECTION",
    "OBJECT_TRACKING",
    "OPTION_VERIFY",
)


def qid(row: dict) -> str:
    for key in ("qid", "question_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def prediction(row: dict) -> str:
    for key in ("prediction", "prediction_label", "final_answer", "answer", "pred"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    return ""


def gold(row: dict) -> str:
    for key in ("gold", "answer_label", "label", "target"):
        if row.get(key) is not None:
            return str(row[key]).strip()
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
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def classify_failure(row: dict) -> str:
    dataset = str(row.get("dataset", ""))
    question = str(row.get("question", "")).lower()
    row_qid = qid(row)

    if dataset == "STAR":
        prefix = row_qid.split("_", 1)[0]
        if prefix in {"Interaction", "Sequence", "Prediction", "Feasibility"}:
            return f"star_{prefix.lower()}"

    if "subtitle" in question:
        return "subtitle"
    if re.search(r"\b(before|after|next|then|first|sequence|order)\b", question):
        return "temporal_sequence"
    if re.search(r"\bwhy\b", question):
        return "causal_why"
    if re.search(r"\b(how many|count|number of)\b", question):
        return "counting"
    if re.search(
        r"\b(object|laptop|paper|notebook|blanket|broom|towel|food|cup|glass|bottle|bed|table|sofa|couch)\b",
        question,
    ):
        return "object_interaction"
    if re.search(r"\b(summarize|summary|overall|process|workflow|key steps|turning points)\b", question):
        return "global_workflow"
    if re.search(r"\b(color|wearing|hand|left|right|small|text|sign)\b", question):
        return "fine_detail"
    return "other"


def result_files(folder: Path) -> list[Path]:
    if folder.is_file():
        return [folder]
    return sorted(folder.glob("results.part*.jsonl"))


def iter_result_rows(folder: Path):
    for file_path in result_files(folder):
        with file_path.open("r", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_result_file"] = file_path.name
                yield row


def default_folders(root: Path) -> list[Path]:
    prefixes = (
        "agent",
        "combined_agent",
    )
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and path.name.startswith(prefixes)
        and result_files(path)
    ]


def load_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = qid(row)
            if row_id:
                rows[row_id] = row
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="conductor/experiments/large_sweeps",
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Specific result folders to analyze. Defaults to all agentic folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="conductor/experiments/large_sweeps/agentic_failure_analysis",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional full dataset manifest used to emit wrong-only JSONL.",
    )
    parser.add_argument(
        "--wrong-jsonl",
        default=None,
        help="Optional path for unique wrong questions copied from --manifest.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    folders = (
        [Path(item) for item in args.folders]
        if args.folders
        else default_folders(root)
    )
    out_dir = Path(args.out_dir)

    question_rows = []
    video_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    run_groups: dict[str, list[dict]] = defaultdict(list)
    failure_counts = Counter()
    tool_counts = Counter()
    wrong_qids = set()

    for folder in folders:
        rows_by_qid = {}
        for row in iter_result_rows(folder):
            row_id = qid(row)
            if row_id:
                rows_by_qid[row_id] = row

        for row in rows_by_qid.values():
            correct = is_correct(row)
            failure_type = "" if correct else classify_failure(row)
            run = folder.name
            dataset = str(row.get("dataset", ""))
            video_id = str(row.get("video_id") or row.get("video") or row.get("video_path") or "")
            row_id = qid(row)

            action_counts = row.get("action_counts") or {}
            for action in ACTION_KEYS:
                tool_counts[(run, action)] += int(action_counts.get(action, 0) or 0)

            if not correct:
                failure_counts[(run, dataset, failure_type)] += 1
                wrong_qids.add(row_id)

            summary_row = {
                "run": run,
                "dataset": dataset,
                "video_id": video_id,
                "qid": row_id,
                "correct": correct,
                "failure_type": failure_type,
                "prediction": prediction(row),
                "gold": gold(row),
                "latency_s": f"{latency(row):.2f}",
                "result_file": row.get("_result_file", ""),
                "question": row.get("question", ""),
            }
            question_rows.append(summary_row)
            video_groups[(run, dataset, video_id)].append(summary_row)
            run_groups[run].append(summary_row)

    video_rows = []
    for (run, dataset, video_id), rows in sorted(video_groups.items()):
        total = len(rows)
        correct = sum(1 for row in rows if row["correct"])
        failures = Counter(row["failure_type"] for row in rows if not row["correct"])
        video_rows.append(
            {
                "run": run,
                "dataset": dataset,
                "video_id": video_id,
                "n": total,
                "correct": correct,
                "wrong": total - correct,
                "accuracy": f"{100 * correct / total if total else 0:.2f}",
                "failure_types": ";".join(
                    f"{name}:{count}"
                    for name, count in failures.most_common()
                ),
            }
        )

    failure_rows = [
        {
            "run": run,
            "dataset": dataset,
            "failure_type": failure_type,
            "wrong": count,
        }
        for (run, dataset, failure_type), count in sorted(failure_counts.items())
    ]

    tool_rows = [
        {
            "run": run,
            "action": action,
            "count": count,
        }
        for (run, action), count in sorted(tool_counts.items())
    ]

    write_csv(
        out_dir / "question_outcomes.csv",
        sorted(question_rows, key=lambda row: (row["run"], row["dataset"], row["qid"])),
        [
            "run",
            "dataset",
            "video_id",
            "qid",
            "correct",
            "failure_type",
            "prediction",
            "gold",
            "latency_s",
            "result_file",
            "question",
        ],
    )
    write_csv(
        out_dir / "video_outcomes.csv",
        video_rows,
        [
            "run",
            "dataset",
            "video_id",
            "n",
            "correct",
            "wrong",
            "accuracy",
            "failure_types",
        ],
    )
    write_csv(
        out_dir / "failure_type_counts.csv",
        failure_rows,
        ["run", "dataset", "failure_type", "wrong"],
    )
    write_csv(
        out_dir / "tool_counts.csv",
        tool_rows,
        ["run", "action", "count"],
    )

    if args.manifest and args.wrong_jsonl:
        manifest_rows = load_manifest(Path(args.manifest))
        selected = [
            manifest_rows[row_id]
            for row_id in sorted(wrong_qids)
            if row_id in manifest_rows
        ]
        wrong_path = Path(args.wrong_jsonl)
        wrong_path.parent.mkdir(parents=True, exist_ok=True)
        wrong_path.write_text(
            "".join(json.dumps(row) + "\n" for row in selected)
        )
        print(f"wrote wrong-only manifest: {wrong_path} rows={len(selected)}")

    print(f"analyzed folders: {len(folders)}")
    for run, rows in sorted(run_groups.items()):
        total = len(rows)
        correct = sum(1 for row in rows if row["correct"])
        print(
            f"{run}: n={total} correct={correct} "
            f"acc={100 * correct / total if total else 0:.2f}%"
        )
    print(f"wrote: {out_dir / 'question_outcomes.csv'}")
    print(f"wrote: {out_dir / 'video_outcomes.csv'}")
    print(f"wrote: {out_dir / 'failure_type_counts.csv'}")
    print(f"wrote: {out_dir / 'tool_counts.csv'}")


if __name__ == "__main__":
    main()
