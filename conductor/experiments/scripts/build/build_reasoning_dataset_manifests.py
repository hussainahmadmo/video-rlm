#!/usr/bin/env python3
"""Build small reasoning-focused VideoQA manifests from local datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def find_video(video_root: Path, video_id: str) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        path = video_root / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


def build_activitynet_qa(
    *,
    root: Path,
    limit: int,
) -> list[dict]:
    question_path = root / "dataset" / "val_q.json"
    answer_path = root / "dataset" / "val_a.json"
    video_root = root / "subset_50" / "videos"

    questions = json.loads(question_path.read_text())
    answers = {
        row["question_id"]: row
        for row in json.loads(answer_path.read_text())
    }

    rows: list[dict] = []

    for question_row in questions:
        qid = question_row["question_id"]
        answer_row = answers.get(qid)
        if answer_row is None:
            continue

        answer = str(answer_row.get("answer", "")).strip().lower()
        if answer not in {"yes", "no"}:
            continue

        video_id = str(question_row["video_name"])
        video_path = find_video(video_root, video_id)
        if video_path is None:
            continue

        choices = [
            "yes",
            "no",
        ]
        answer_idx = choices.index(answer)

        rows.append({
            "dataset": "ActivityNet-QA",
            "qid": qid,
            "video_id": video_id,
            "video": str(video_path),
            "question": question_row["question"],
            "choices": choices,
            "answer_idx": answer_idx,
            "answer_label": LABELS[answer_idx],
            "answer": choices[answer_idx],
            "answer_type": "yes_no",
            "question_category": f"activitynet_type_{answer_row.get('type')}",
            "source_dataset": "ActivityNet-QA_val_available_yesno",
        })

        if len(rows) >= limit:
            break

    return rows


def build_star(
    *,
    root: Path,
    limit: int,
) -> list[dict]:
    annotation_path = root / "subset_50" / "annotations.json"
    video_root = root / "subset_50" / "videos"

    annotations = json.loads(annotation_path.read_text())
    rows: list[dict] = []

    for item in annotations:
        video_id = str(item["video_id"])
        video_path = find_video(video_root, video_id)
        if video_path is None:
            continue

        choices = [
            str(choice["choice"])
            for choice in item["choices"]
        ]
        answer = str(item["answer"])

        try:
            answer_idx = choices.index(answer)
        except ValueError:
            continue

        rows.append({
            "dataset": "STAR",
            "qid": str(item["question_id"]),
            "video_id": video_id,
            "video": str(video_path),
            "start_s": float(item.get("start", 0.0)),
            "end_s": float(item.get("end", 0.0)),
            "question": item["question"],
            "choices": choices,
            "answer_idx": answer_idx,
            "answer_label": LABELS[answer_idx],
            "answer": answer,
            "answer_type": "multiple_choice",
            "question_category": str(item["question_id"]).split("_")[0],
            "source_dataset": "STAR_val_subset_50_available",
        })

        if len(rows) >= limit:
            break

    return rows


def audit(name: str, rows: list[dict]) -> None:
    videos = {
        row["video"]
        for row in rows
    }
    missing = [
        path
        for path in videos
        if not Path(path).exists()
    ]

    print(
        f"{name}: questions={len(rows)} "
        f"unique_videos={len(videos)} missing_videos={len(missing)}"
    )

    if missing:
        for path in missing[:10]:
            print(f"  missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activitynet-root",
        type=Path,
        default=Path("/dataheart/hussainahmad/video-datasets/activitynet-qa"),
    )
    parser.add_argument(
        "--star-root",
        type=Path,
        default=Path("/dataheart/hussainahmad/video-datasets/STAR_Benchmark"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("conductor/experiments/diverse_eval/reasoning"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    activitynet = build_activitynet_qa(
        root=args.activitynet_root,
        limit=args.limit,
    )
    star = build_star(
        root=args.star_root,
        limit=args.limit,
    )
    combined = activitynet + star

    activitynet_path = args.out_dir / "activitynet_qa_yesno_100_available.jsonl"
    star_path = args.out_dir / "star_100_available.jsonl"
    combined_path = args.out_dir / "activitynet_star_reasoning_200_available.jsonl"

    write_jsonl(activitynet_path, activitynet)
    write_jsonl(star_path, star)
    write_jsonl(combined_path, combined)

    audit("ActivityNet-QA", activitynet)
    audit("STAR", star)
    audit("Combined", combined)

    print(f"wrote: {activitynet_path}")
    print(f"wrote: {star_path}")
    print(f"wrote: {combined_path}")

    print()
    print("Not built locally: AGQA and YouCook2")
    print("Reason: no local annotation/video roots were found under /dataheart.")
    print("Add their roots and extend this builder once the data is present.")


if __name__ == "__main__":
    main()
