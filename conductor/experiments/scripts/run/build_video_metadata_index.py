#!/usr/bin/env python3
"""Build a reusable video metadata index from one or more request traces."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def trace_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.trace)
    if args.trace_root is not None:
        paths.extend(sorted(args.trace_root.glob("*.jsonl")))
    return list(dict.fromkeys(path.resolve() for path in paths))


def videos_from_traces(paths: list[Path]) -> list[Path]:
    videos: set[Path] = set()
    for trace in paths:
        with trace.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get("video") or row.get("video_path") or row.get("path")
                if value is None:
                    raise ValueError(
                        f"{trace}:{line_number}: row has no video path"
                    )
                text = str(value)
                if "://" in text:
                    raise ValueError(
                        f"{trace}:{line_number}: remote URL cannot be ffprobed: {text}"
                    )
                videos.add(Path(text).expanduser().resolve(strict=False))
    return sorted(videos)


def probe_video(video: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height:format=duration",
        "-of", "json",
        str(video),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise ValueError("ffprobe returned no video stream")
        stream = streams[0]
        duration = (payload.get("format") or {}).get("duration")
        return {
            "video": str(video),
            "duration_s": float(duration) if duration is not None else None,
            "video_width": int(stream["width"]),
            "video_height": int(stream["height"]),
            "video_codec": stream.get("codec_name"),
            "error": None,
        }
    except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        return {
            "video": str(video),
            "duration_s": None,
            "video_width": None,
            "video_height": None,
            "video_codec": None,
            "error": detail,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", type=Path, action="append", default=[],
        help="Request-trace JSONL; may be repeated",
    )
    parser.add_argument(
        "--trace-root", type=Path,
        help="Directory whose top-level JSONL traces should be indexed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    paths = trace_paths(args)
    if not paths:
        parser.error("provide --trace or --trace-root")
    if args.workers < 1:
        parser.error("--workers must be positive")
    missing_traces = [str(path) for path in paths if not path.is_file()]
    if missing_traces:
        parser.error(f"trace files do not exist: {', '.join(missing_traces)}")

    videos = videos_from_traces(paths)
    rows: dict[Path, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(probe_video, video, args.ffprobe): video
            for video in videos
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for video in videos:
            handle.write(json.dumps(rows[video]) + "\n")
    temporary.replace(args.output)

    failures = sum(row["error"] is not None for row in rows.values())
    print(f"traces={len(paths)} videos={len(videos)} failures={failures}")
    print(args.output)


if __name__ == "__main__":
    main()
