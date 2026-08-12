#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path


CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def video_duration_s(path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def duration_bucket(duration_s):
    if duration_s is None:
        return "unknown"
    if duration_s < 30:
        return "short"
    if duration_s < 180:
        return "medium"
    return "long"


def normalize_nextqa_type(value):
    mapping = {
        "CH": "causal_how",
        "CW": "causal_why",
        "TN": "temporal_next",
        "TP": "temporal_previous",
        "TC": "temporal_count",
        "DL": "descriptive_location",
        "DC": "descriptive_count",
        "DO": "descriptive_object",
    }
    return mapping.get(str(value), str(value or "nextqa"))


def find_video(video_id, roots):
    names = [
        f"{video_id}.mp4",
        f"{video_id}.mkv",
        f"{video_id}.webm",
    ]
    for root in roots:
        root = Path(root)
        for name in names:
            direct = root / name
            if direct.exists():
                return direct
        for name in names:
            matches = list(root.rglob(name))
            if matches:
                return matches[0]
    return None


def build_nextqa_rows(annotation_csv, video_roots, limit):
    rows = []
    missing = []
    with open(annotation_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = str(row["video"])
            video = find_video(video_id, video_roots)
            if video is None:
                missing.append(video_id)
                continue

            choices = [
                row.get(f"a{index}", "")
                for index in range(5)
            ]

            try:
                answer_idx = int(row["answer"])
            except (TypeError, ValueError):
                continue

            if not (0 <= answer_idx < len(choices)):
                continue

            duration = video_duration_s(video)
            qid = f"{video_id}_{row.get('qid', len(rows))}"

            rows.append({
                "dataset": "NExT-QA",
                "qid": qid,
                "video_id": video_id,
                "video": str(video),
                "duration_s": duration,
                "duration_bucket": duration_bucket(duration),
                "question": str(row["question"]),
                "choices": choices,
                "answer_idx": answer_idx,
                "answer_label": CHOICE_LABELS[answer_idx],
                "answer": choices[answer_idx],
                "question_category": normalize_nextqa_type(
                    row.get("type")
                ),
                "topic_category": "nextqa",
                "vimio_profile": "agent_friendly_temporal_reasoning",
                "source_dataset": "NExT-QA",
            })

            if limit and len(rows) >= limit:
                break

    return rows, missing


def build_egoschema_rows(path, limit):
    rows = []
    for row in load_jsonl(path):
        if row.get("dataset") != "EgoSchema":
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--egoschema",
        default="conductor/experiments/diverse_eval/egoschema_diverse_available.jsonl",
    )
    parser.add_argument(
        "--nextqa_csv",
        default="/dataheart/hussainahmad/video-datasets/NExT-QA/subset_50/annotations.csv",
    )
    parser.add_argument(
        "--nextqa_video_root",
        action="append",
        default=[
            "/dataheart/hussainahmad/video-datasets/NExT-QA/subset_50/videos",
            "/dataheart/hussainahmad/video-datasets/NExT-QA/hf_nextqa/videos/NExTVideo",
            "/dataheart/hussainahmad/datasets/NExTVideo_subset",
        ],
    )
    parser.add_argument("--ego_limit", type=int, default=50)
    parser.add_argument("--nextqa_limit", type=int, default=50)
    parser.add_argument(
        "--output",
        default="conductor/experiments/diverse_eval/agent_friendly_egoschema_nextqa.jsonl",
    )
    parser.add_argument(
        "--nextqa_output",
        default="conductor/experiments/diverse_eval/nextqa_agent_friendly_available.jsonl",
    )
    args = parser.parse_args()

    ego_rows = build_egoschema_rows(
        Path(args.egoschema),
        args.ego_limit,
    )
    nextqa_rows, missing_nextqa = build_nextqa_rows(
        Path(args.nextqa_csv),
        [Path(root) for root in args.nextqa_video_root],
        args.nextqa_limit,
    )

    write_jsonl(
        Path(args.nextqa_output),
        nextqa_rows,
    )
    combined = ego_rows + nextqa_rows
    write_jsonl(
        Path(args.output),
        combined,
    )

    print("Wrote:", args.nextqa_output)
    print("Wrote:", args.output)
    print("Rows:", len(combined))
    print("By dataset:", dict(Counter(row["dataset"] for row in combined)))
    print("Missing NExT-QA videos:", len(set(missing_nextqa)))
    if missing_nextqa:
        print("First missing:", sorted(set(missing_nextqa))[:10])


if __name__ == "__main__":
    main()
