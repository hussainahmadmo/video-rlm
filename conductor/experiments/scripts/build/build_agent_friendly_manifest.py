#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path


CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm"}


def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_annotation_records(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with open(path, newline="") as f:
            yield from csv.DictReader(f)
        return

    if suffix == ".jsonl":
        yield from load_jsonl(path)
        return

    if suffix == ".json":
        with open(path, "r") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            yield from payload
            return

        for key in (
            "data",
            "annotations",
            "questions",
            "examples",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return

        raise ValueError(
            f"Unsupported JSON annotation structure: {path}"
        )

    raise ValueError(
        f"Unsupported annotation file type: {path}"
    )


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


def normalize_intentqa_type(value):
    return str(value or "intent_reasoning")


def first_present(row, names, default=""):
    for name in names:
        if name in row and row[name] not in {
            None,
            "",
        }:
            return row[name]
    return default


def parse_choices(row):
    raw = first_present(
        row,
        [
            "choices",
            "options",
            "answers",
            "candidates",
        ],
        None,
    )

    if isinstance(raw, list):
        return [
            str(choice)
            for choice in raw
        ]

    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [
                    str(choice)
                    for choice in parsed
                ]
        except json.JSONDecodeError:
            pass

        if "||" in raw:
            return [
                item.strip()
                for item in raw.split("||")
                if item.strip()
            ]

    choices = []
    for prefix in (
        "a",
        "option_",
        "choice_",
    ):
        for index in range(10):
            key = f"{prefix}{index}"
            if key in row and str(row[key]).strip():
                choices.append(
                    str(row[key])
                )
        if choices:
            return choices

    letter_choices = []
    for label in "ABCDE":
        for key in (
            label,
            label.lower(),
            f"answer_{label}",
            f"answer_{label.lower()}",
        ):
            if key in row and str(row[key]).strip():
                letter_choices.append(
                    str(row[key])
                )
                break

    return letter_choices


def parse_answer_idx(row, choices):
    raw = first_present(
        row,
        [
            "answer_idx",
            "answer_index",
            "answer",
            "label",
            "correct",
            "correct_idx",
            "correct_answer",
        ],
        None,
    )

    if raw is None:
        return None

    if isinstance(raw, int):
        return raw

    text = str(raw).strip()

    try:
        return int(text)
    except ValueError:
        pass

    if len(text) == 1 and text.upper() in CHOICE_LABELS:
        return CHOICE_LABELS.index(
            text.upper()
        )

    normalized = text.lower()
    for index, choice in enumerate(choices):
        if normalized == str(choice).strip().lower():
            return index

    return None


def build_video_index(roots):
    index = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                index.setdefault(path.stem, path)
                index.setdefault(path.name, path)
    return index


def find_video(video_id, roots, video_index=None):
    names = [
        f"{video_id}.mp4",
        f"{video_id}.mkv",
        f"{video_id}.webm",
    ]
    if video_index is not None:
        for key in [video_id, *names]:
            video = video_index.get(key)
            if video is not None:
                return video
        return None

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
    video_index = build_video_index(video_roots)
    with open(annotation_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = str(row["video"])
            video = find_video(video_id, video_roots, video_index)
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


def build_intentqa_rows(annotation_path, video_roots, limit):
    if not annotation_path or not Path(annotation_path).exists():
        return [], []

    rows = []
    missing = []
    video_index = build_video_index(video_roots)

    for raw_row in load_annotation_records(annotation_path):
        row = dict(raw_row)
        video_id = str(
            first_present(
                row,
                [
                    "video_id",
                    "video",
                    "vid",
                    "clip_id",
                ],
            )
        )

        if not video_id:
            continue

        video_path = first_present(
            row,
            [
                "video_path",
                "path",
            ],
            "",
        )

        video = (
            Path(video_path)
            if video_path
            else None
        )

        if not (
            video
            and video.exists()
        ):
            video = find_video(
                video_id,
                video_roots,
                video_index,
            )

        if video is None:
            missing.append(video_id)
            continue

        question = str(
            first_present(
                row,
                [
                    "question",
                    "query",
                    "q",
                ],
            )
        )

        choices = parse_choices(row)
        answer_idx = parse_answer_idx(
            row,
            choices,
        )

        if (
            not question
            or not choices
            or answer_idx is None
            or not (0 <= answer_idx < len(choices))
        ):
            continue

        duration = video_duration_s(video)
        qid = str(
            first_present(
                row,
                [
                    "qid",
                    "question_id",
                    "id",
                ],
                f"{video_id}_{len(rows)}",
            )
        )
        qid = f"{video_id}_{qid}_{len(rows)}"

        rows.append({
            "dataset": "IntentQA",
            "qid": qid,
            "video_id": video_id,
            "video": str(video),
            "duration_s": duration,
            "duration_bucket": duration_bucket(duration),
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "answer_label": CHOICE_LABELS[answer_idx],
            "answer": choices[answer_idx],
            "question_category": normalize_intentqa_type(
                first_present(
                    row,
                    [
                        "type",
                        "question_type",
                        "category",
                    ],
                    "intent_reasoning",
                )
            ),
            "topic_category": "intentqa",
            "vimio_profile": "agent_friendly_intent_reasoning",
            "source_dataset": "IntentQA",
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
        "--intentqa_annotations",
        default="",
        help=(
            "Optional IntentQA CSV/JSON/JSONL annotations. If omitted or "
            "missing, IntentQA is skipped."
        ),
    )
    parser.add_argument(
        "--intentqa_video_root",
        action="append",
        default=[
            "/dataheart/hussainahmad/video-datasets/IntentQA/videos",
            "/dataheart/hussainahmad/datasets/IntentQA/videos",
            "/dataheart/hussainahmad/datasets/IntentQA",
            "/dataheart/hussainahmad/video-datasets/NExT-QA",
            "/dataheart/hussainahmad/datasets/NExT-QA",
            "/dataheart/hussainahmad/datasets/NExTVideo_subset",
        ],
    )
    parser.add_argument("--intentqa_limit", type=int, default=50)
    parser.add_argument(
        "--output",
        default="conductor/experiments/diverse_eval/agent_friendly_egoschema_nextqa_intentqa.jsonl",
    )
    parser.add_argument(
        "--nextqa_output",
        default="conductor/experiments/diverse_eval/nextqa_agent_friendly_available.jsonl",
    )
    parser.add_argument(
        "--intentqa_output",
        default="conductor/experiments/diverse_eval/intentqa_agent_friendly_available.jsonl",
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
    intentqa_rows, missing_intentqa = build_intentqa_rows(
        Path(args.intentqa_annotations)
        if args.intentqa_annotations
        else None,
        [Path(root) for root in args.intentqa_video_root],
        args.intentqa_limit,
    )

    write_jsonl(
        Path(args.nextqa_output),
        nextqa_rows,
    )
    write_jsonl(
        Path(args.intentqa_output),
        intentqa_rows,
    )
    combined = ego_rows + nextqa_rows + intentqa_rows
    write_jsonl(
        Path(args.output),
        combined,
    )

    print("Wrote:", args.nextqa_output)
    print("Wrote:", args.intentqa_output)
    print("Wrote:", args.output)
    print("Rows:", len(combined))
    print("By dataset:", dict(Counter(row["dataset"] for row in combined)))
    print("Missing NExT-QA videos:", len(set(missing_nextqa)))
    if missing_nextqa:
        print("First missing:", sorted(set(missing_nextqa))[:10])
    print("Missing IntentQA videos:", len(set(missing_intentqa)))
    if missing_intentqa:
        print("First missing:", sorted(set(missing_intentqa))[:10])


if __name__ == "__main__":
    main()
