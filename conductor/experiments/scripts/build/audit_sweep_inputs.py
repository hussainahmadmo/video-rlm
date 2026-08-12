import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def human_bytes(num_bytes):
    units = ["B", "K", "M", "G", "T"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024


def disk_summary(path):
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total": human_bytes(usage.total),
        "used": human_bytes(usage.used),
        "free": human_bytes(usage.free),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument(
        "--storage_roots",
        nargs="*",
        default=[
            "/dataheart/hussainahmad/datasets",
            "/dataheart/hussainahmad/video-datasets",
            "/tmp",
            "/home/hussainahmad",
        ],
    )
    parser.add_argument("--show_missing", type=int, default=20)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    config_path = Path(args.config_file)

    rows = list(load_jsonl(dataset_path))
    configs = json.loads(config_path.read_text())

    by_dataset = Counter()
    missing = []
    existing_video_paths = set()
    qids = []

    for row in rows:
        qids.append(row.get("qid"))
        by_dataset[
            row.get("dataset")
            or row.get("topic_category")
            or "unknown"
        ] += 1

        video = row.get("video")
        if not video or not Path(video).exists():
            missing.append({
                "qid": row.get("qid"),
                "dataset": row.get("dataset"),
                "video": video,
            })
        else:
            existing_video_paths.add(str(Path(video)))

    duplicates = [
        qid
        for qid, count in Counter(qids).items()
        if qid is not None and count > 1
    ]

    print("Dataset:")
    print(f"  path: {dataset_path}")
    print(f"  questions: {len(rows)}")
    print(f"  unique videos: {len(existing_video_paths)}")
    print(f"  missing videos: {len(missing)}")
    print(f"  duplicate qids: {len(duplicates)}")
    print("  by dataset:")
    for name, count in sorted(by_dataset.items()):
        print(f"    {name}: {count}")

    print("\nConfig sweep:")
    print(f"  path: {config_path}")
    print(f"  configs: {len(configs)}")
    print(f"  total runs: {len(rows) * len(configs)}")
    print("  config names:")
    for config in configs:
        print(f"    {config.get('name')}")

    print("\nStorage roots:")
    for root in args.storage_roots:
        path = Path(root)
        if path.exists():
            usage = disk_summary(path)
            print(
                f"  {usage['path']}: "
                f"free={usage['free']} "
                f"used={usage['used']} "
                f"total={usage['total']}"
            )
        else:
            print(f"  {path}: missing")

    if missing:
        print("\nMissing videos:")
        for item in missing[: args.show_missing]:
            print(
                f"  qid={item['qid']} "
                f"dataset={item['dataset']} "
                f"video={item['video']}"
            )
        if len(missing) > args.show_missing:
            print(f"  ... {len(missing) - args.show_missing} more")

    if duplicates:
        print("\nDuplicate qids:")
        for qid in duplicates[: args.show_missing]:
            print(f"  {qid}")

    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
