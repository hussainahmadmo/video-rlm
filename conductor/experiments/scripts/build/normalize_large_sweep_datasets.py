#!/usr/bin/env python3
"""Normalize the five large-sweep benchmarks using videos present on disk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi"}


def video_index(root: Path) -> dict[str, Path]:
    return {
        path.stem: path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_row(dataset: str, qid: str, video_id: str, video: Path, question: str,
             choices: list[str], answer_idx: int, source: str, **extra: object) -> dict:
    return {
        "dataset": dataset,
        "qid": qid,
        "video_id": video_id,
        "video": str(video),
        "question": question,
        "choices": choices,
        "answer_idx": answer_idx,
        "answer_label": LABELS[answer_idx],
        "answer": choices[answer_idx],
        "source_dataset": source,
        **extra,
    }


def normalize_egoschema(meta: Path, videos: Path) -> list[dict]:
    questions = json.loads((meta / "questions.json").read_text())
    answers = json.loads((meta / "subset_answers.json").read_text())
    available = video_index(videos)
    rows = []
    for item in questions:
        qid = str(item["q_uid"])
        if qid not in answers or qid not in available:
            continue
        choices = [str(item[f"option {index}"]) for index in range(5)]
        rows.append(base_row("EgoSchema", qid, qid, available[qid],
                             str(item["question"]), choices, int(answers[qid]),
                             "EgoSchema_public_labeled"))
    return rows


def normalize_longvideobench(annotation: Path, videos: Path) -> list[dict]:
    available = video_index(videos)
    rows = []
    for item in json.loads(annotation.read_text()):
        video_id = str(item["video_id"])
        if video_id not in available:
            continue
        choices = [str(value) for value in item["candidates"]]
        answer_idx = int(item["correct_choice"])
        rows.append(base_row(
            "LongVideoBench", str(item["id"]), video_id, available[video_id],
            str(item["question"]), choices, answer_idx, "LongVideoBench_val",
            duration_s=item.get("duration"), question_category=item.get("question_category"),
            topic_category=item.get("topic_category"), position=item.get("position")))
    return rows


def normalize_nextqa(meta: Path, videos: Path) -> list[dict]:
    available = video_index(videos)
    rows = []
    for split in ("train", "val", "test"):
        with (meta / f"{split}.csv").open(newline="") as handle:
            for item in csv.DictReader(handle):
                video_id = str(item["video"])
                if video_id not in available or item.get("answer", "") == "":
                    continue
                choices = [str(item[f"a{index}"]) for index in range(5)]
                qid = str(item["qid"])
                rows.append(base_row(
                    "NExT-QA", qid, video_id, available[video_id], str(item["question"]),
                    choices, int(item["answer"]), f"NExT-QA_{split}",
                    question_type=item.get("type"), original_split=split))
    return rows


def normalize_star(annotation: Path, videos: Path) -> list[dict]:
    available = video_index(videos)
    rows = []
    for item in json.loads(annotation.read_text()):
        video_id = str(item["video_id"])
        if video_id not in available:
            continue
        choices = [str(choice["choice"]) for choice in item["choices"]]
        try:
            answer_idx = choices.index(str(item["answer"]))
        except ValueError as exc:
            raise ValueError(f"STAR answer not found in choices: {item['question_id']}") from exc
        rows.append(base_row(
            "STAR", str(item["question_id"]), video_id, available[video_id],
            str(item["question"]), choices, answer_idx, f"STAR_{annotation.stem.removeprefix('STAR_')}",
            start=item.get("start"), end=item.get("end")))
    return rows


def normalize_vrbench(annotation: Path, videos: Path) -> list[dict]:
    available = video_index(videos)
    rows = []
    with annotation.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            video_id = str(item["video_id"])
            if video_id not in available:
                continue
            for qa_name, qa in item["mcq"].items():
                options = qa["options"]
                labels = sorted(options)
                choices = [str(options[label]) for label in labels]
                answer_idx = labels.index(str(qa["answer"]))
                rows.append(base_row(
                    "VRBench", f"{video_id}_{qa_name}", video_id, available[video_id],
                    str(qa["question"]), choices, answer_idx, "VRBench_eval",
                    reasoning_type=qa.get("reasoning_type")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path,
                        default=Path("/workspace/video-datasets/official_metadata"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.official_root
    datasets = {
        "egoschema": normalize_egoschema(root / "EgoSchema", Path("/workspace/datasets/EgoSchema")),
        "longvideobench": normalize_longvideobench(
            root / "LongVideoBench/lvb_val.json", Path("/workspace/datasets/LongVideoBench")),
        "nextqa": normalize_nextqa(
            root / "NExT-QA/dataset/nextqa", Path("/workspace/video-datasets/NExT-QA")),
        "star": (
            normalize_star(root / "STAR/STAR_train.json", Path("/workspace/video-datasets/STAR_Benchmark"))
            + normalize_star(root / "STAR/STAR_val.json", Path("/workspace/video-datasets/STAR_Benchmark"))
        ),
        "vrbench": normalize_vrbench(
            root / "VRBench/VRBench_eval.jsonl", Path("/workspace/datasets/VRBench")),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    summary = {}
    for name, rows in datasets.items():
        rows.sort(key=lambda row: (row["video_id"], row["qid"]))
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
        combined.extend(rows)
        summary[name] = {
            "questions": len(rows),
            "videos": len({row["video_id"] for row in rows}),
        }
    combined.sort(key=lambda row: (row["dataset"], row["video_id"], row["qid"]))
    write_jsonl(args.output_dir / "all_available.jsonl", combined)
    (args.output_dir / "normalization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
