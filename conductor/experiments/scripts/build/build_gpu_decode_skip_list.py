#!/usr/bin/env python
import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def qid(row):
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def decord_ctx_arg(ctx_name):
    import decord
    from decord import cpu, gpu

    raw = str(ctx_name or "cpu").strip().lower()
    if raw.startswith("gpu"):
        parts = raw.split(":", 1)
        device_id = int(parts[1]) if len(parts) == 2 and parts[1] else 0
        return gpu(device_id)
    return cpu(0)


def probe_video(args):
    import decord
    from decord import VideoReader, bridge

    bridge.set_bridge("torch")
    ctx = decord_ctx_arg(args.ctx)
    vr = VideoReader(args.video, ctx=ctx, width=args.width, height=args.height)
    fps = float(vr.get_avg_fps())
    nframes = len(vr)
    if nframes <= 0:
        raise RuntimeError("video has zero frames")

    if args.scan_fps > 0:
        step = max(1, int(round(fps / float(args.scan_fps))))
        indices = list(range(0, nframes, step))[: args.max_probe_frames]
    else:
        indices = [0]

    if not indices:
        indices = [0]

    _ = vr.get_batch(indices)
    print(
        json.dumps(
            {
                "ok": True,
                "frames": nframes,
                "fps": fps,
                "indices": indices,
                "decord_file": decord.__file__,
            }
        ),
        flush=True,
    )


def run_probe(video, args):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-video",
        str(video),
        "--ctx",
        args.ctx,
        "--scan-fps",
        str(args.scan_fps),
        "--max-probe-frames",
        str(args.max_probe_frames),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_s,
        check=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--ctx", default="gpu:0")
    parser.add_argument("--scan-fps", type=float, default=0.00390625)
    parser.add_argument("--max-probe-frames", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--probe-video", default=None)
    args = parser.parse_args()

    if args.probe_video:
        args.video = args.probe_video
        probe_video(args)
        return

    if not args.dataset or not args.output:
        parser.error("--dataset and --output are required unless --probe-video is used")

    rows = load_jsonl(args.dataset)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    by_key = OrderedDict()
    for row in rows:
        row_qid = qid(row)
        video = row.get("video")
        if not row_qid or not video:
            continue
        by_key[(row_qid, str(video))] = row

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bad = []
    ok = 0

    with output.open("w") as handle:
        for idx, ((row_qid, video), row) in enumerate(by_key.items(), start=1):
            print(
                f"[{idx}/{len(by_key)}] probe qid={row_qid} "
                f"ctx={args.ctx} video={video}",
                flush=True,
            )
            try:
                result = run_probe(video, args)
            except subprocess.TimeoutExpired:
                print(
                    f"[bad] timeout qid={row_qid} timeout_s={args.timeout_s}",
                    flush=True,
                )
                handle.write(row_qid + "\n")
                handle.flush()
                bad.append(row_qid)
                continue

            if result.returncode == 0:
                ok += 1
                continue

            print(
                f"[bad] failed qid={row_qid} returncode={result.returncode}",
                flush=True,
            )
            if result.stderr:
                print(result.stderr[-2000:], flush=True)
            handle.write(row_qid + "\n")
            handle.flush()
            bad.append(row_qid)

    print(f"probed: {len(by_key)}")
    print(f"ok: {ok}")
    print(f"bad: {len(bad)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
