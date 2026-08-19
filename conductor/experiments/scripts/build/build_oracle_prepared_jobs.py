#!/usr/bin/env python3
"""Prepare a Cartesian query x policy grid for oracle evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_prepared_vlm_jobs import import_runner, prepare_clip_oneshot_job


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    runner = import_runner()
    rows = load_jsonl(args.dataset)
    configs = json.loads(args.config_file.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str]] = set()
    if args.output.exists() and not args.no_resume:
        for row in load_jsonl(args.output):
            done.add((str(row.get("qid")), str(row.get("config_name"))))
    elif args.output.exists():
        args.output.unlink()

    total = len(rows) * len(configs)
    index = 0
    for item in rows:
        for raw_config in configs:
            index += 1
            config = dict(raw_config)
            config.setdefault("evidence_type", "generic")
            config.setdefault("expand_neighbors", False)
            config.setdefault("preserve_order", True)
            config.setdefault("include_uniform_anchors", False)
            config["answer_with_confidence"] = True
            config["enable_evidence_fallback"] = False
            key = (str(item.get("qid")), str(config["name"]))
            if key in done:
                print(f"[{index}/{total}] skip qid={key[0]} config={key[1]}")
                continue

            print(f"[{index}/{total}] prepare qid={key[0]} config={key[1]}")
            try:
                job = prepare_clip_oneshot_job(runner, item, config)
                job["prepare_error"] = None
            except Exception as exc:
                job = {
                    "qid": item.get("qid"),
                    "dataset": item.get("dataset"),
                    "config_name": config.get("name"),
                    "prepare_error": repr(exc),
                }
            append_jsonl(args.output, job)

    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
