#!/usr/bin/env python3
"""Split the Azure peak-window replay into fixed-duration trace segments."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


TENANT_MAP = {"tenant_a": "a", "tenant_b": "b", "tenant_c": "c"}


def interpolate(points: dict[str, float], frame_count: int) -> float:
    anchors = sorted((int(key), float(value)) for key, value in points.items())
    for frames, value in anchors:
        if frame_count == frames:
            return value
    if frame_count <= anchors[0][0]:
        return anchors[0][1]
    if frame_count >= anchors[-1][0]:
        return anchors[-1][1]
    for (left_frames, left_value), (right_frames, right_value) in zip(anchors, anchors[1:]):
        if left_frames < frame_count < right_frames:
            ratio = (frame_count - left_frames) / (right_frames - left_frames)
            return left_value + ratio * (right_value - left_value)
    raise AssertionError(frame_count)


def expanded_lane_profile(path: Path, frame_counts: list[int]) -> dict:
    source = json.loads(path.read_text())
    result = {"cpu": {}}
    for frame_count in frame_counts:
        result["cpu"][str(frame_count)] = interpolate(source["cpu"], frame_count)
        result[str(frame_count)] = {
            str(width): interpolate(
                {frames: values[str(width)] for frames, values in source.items() if frames != "cpu"},
                frame_count,
            )
            for width in (1, 2, 4)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--segment-s", type=float, default=100.0)
    parser.add_argument("--lane-profile", type=Path)
    args = parser.parse_args()
    if args.duration_s <= 0 or args.segment_s <= 0:
        parser.error("durations must be positive")

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("input trace is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    frame_counts = sorted({int(row["frame_count"]) for row in rows})
    segments = []
    segment_count = math.ceil(args.duration_s / args.segment_s)
    assigned = 0
    for index in range(segment_count):
        start = index * args.segment_s
        end = min(args.duration_s, start + args.segment_s)
        selected = [
            row for row in rows
            if start <= float(row["arrival_s"])
            and (float(row["arrival_s"]) < end or (index == segment_count - 1 and float(row["arrival_s"]) <= end))
        ]
        output_rows = []
        for row in selected:
            updated = dict(row)
            updated["azure_peak_arrival_s"] = float(row["arrival_s"])
            updated["arrival_s"] = float(row["arrival_s"]) - start
            updated["tenant"] = TENANT_MAP.get(str(row["tenant"]), str(row["tenant"]))
            updated["video"] = args.video
            updated["trace_segment"] = index
            output_rows.append(updated)
        output_path = args.output / f"segment{index}.jsonl"
        output_path.write_text("".join(json.dumps(row) + "\n" for row in output_rows))
        assigned += len(output_rows)
        counts = Counter(int(row["frame_count"]) for row in output_rows)
        segments.append({
            "segment": index,
            "source_start_s": start,
            "source_end_s": end,
            "requests": len(output_rows),
            "capped_frames": sum(frame * count for frame, count in counts.items()),
            "requests_at_cap": counts[128],
            "tenant_requests": dict(Counter(row["tenant"] for row in output_rows)),
            "frame_counts": dict(sorted(counts.items())),
            "path": str(output_path),
        })
    if assigned != len(rows):
        raise RuntimeError(f"assigned {assigned} of {len(rows)} requests")

    manifest = {
        "design": "azure_peak_fixed_time_segments",
        "source": str(args.input),
        "source_requests": len(rows),
        "duration_s": args.duration_s,
        "segment_s": args.segment_s,
        "segments": segments,
        "arrival_mapping": "Original relative arrivals rebased within each segment",
        "size_mapping": "Original image counts mapped to frame counts and capped at 128 upstream",
        "tenant_mapping": "Cyclic synthetic Azure tenant labels remapped to a/b/c",
        "media": "One substitute video; not original Azure media",
    }
    if args.lane_profile:
        profile = expanded_lane_profile(args.lane_profile, frame_counts)
        profile_path = args.output / "lane_profile_interpolated.json"
        profile_path.write_text(json.dumps(profile, indent=2) + "\n")
        manifest["lane_profile"] = str(profile_path)
        manifest["profile_method"] = (
            "Piecewise-linear interpolation of the measured 1/16/128-frame "
            "CPU and GPU-width calibration; no evaluation completions used"
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
