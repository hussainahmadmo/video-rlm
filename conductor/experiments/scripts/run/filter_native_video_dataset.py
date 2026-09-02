#!/usr/bin/env python3
"""Create a decoder-compatible JSONL subset using ffprobe metadata.

This keeps CPU and GPU video-backend comparisons matched: every retained
record references a file whose container/codec pair is accepted by the
requested experiment.  Compatibility filtering is reported explicitly rather
than silently counting backend crashes as latency samples.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


def video_path(row: dict) -> Path:
    value = row.get("video") or row.get("video_path") or row.get("path")
    if not value:
        raise ValueError("row has no video path")
    return Path(str(value))


def probe(path: Path) -> tuple[str, str]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=format_name:stream=codec_name",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(completed.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError("no video stream")
    return (
        str((data.get("format") or {}).get("format_name") or "unknown"),
        str(streams[0].get("codec_name") or "unknown"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exclude-qid", action="append", default=[])
    parser.add_argument(
        "--allowed-codecs", default="h264,hevc",
        help="Comma-separated codecs supported by every compared backend.",
    )
    parser.add_argument(
        "--required-extension", default=".mp4",
        help="Retain only this file suffix; use an empty value to disable.",
    )
    args = parser.parse_args()

    allowed_codecs = {
        value.strip().lower()
        for value in args.allowed_codecs.split(",")
        if value.strip()
    }
    excluded = set(args.exclude_qid)
    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    cache: dict[Path, tuple[str, str] | Exception] = {}
    kept: list[dict] = []
    reasons: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    codecs: Counter[str] = Counter()

    for row in rows:
        qid = str(row.get("qid") or row.get("question_id") or row.get("id"))
        if qid in excluded:
            reasons["excluded_qid"] += 1
            continue
        try:
            path = video_path(row)
        except ValueError:
            reasons["missing_video_path"] += 1
            continue
        if args.required_extension and path.suffix.lower() != args.required_extension:
            reasons[f"extension:{path.suffix.lower() or 'none'}"] += 1
            continue
        if path not in cache:
            try:
                cache[path] = probe(path)
            except Exception as exc:  # record failures; never hide them
                cache[path] = exc
        metadata = cache[path]
        if isinstance(metadata, Exception):
            reasons["ffprobe_failure"] += 1
            continue
        format_name, codec_name = metadata
        formats[format_name] += 1
        codecs[codec_name] += 1
        if codec_name.lower() not in allowed_codecs:
            reasons[f"codec:{codec_name.lower()}"] += 1
            continue
        kept.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in kept))
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "input_rows": len(rows),
        "retained_rows": len(kept),
        "retained_fraction": len(kept) / len(rows) if rows else 0.0,
        "unique_videos_probed": len(cache),
        "required_extension": args.required_extension,
        "allowed_codecs": sorted(allowed_codecs),
        "excluded_qids": sorted(excluded),
        "excluded_reasons": dict(sorted(reasons.items())),
        "observed_formats": dict(sorted(formats.items())),
        "observed_codecs": dict(sorted(codecs.items())),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
