#!/usr/bin/env python3
"""Run a preparation noisy-neighbor fairness pilot on one GPU."""

import argparse
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
POLICIES = ["fcfs", "engine_tenant_fair", "prep_max_min", "max_min"]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction + .999999) - 1))]


def workload(video):
    rows = []
    for index in range(12):
        rows.append({
            "request_id": f"a-{index}", "qid": f"a-{index}",
            "tenant": "a", "modality": "video", "video": str(video),
            "frame_count": 128, "arrival_s": round(index * .05, 6),
            "max_tokens": 32, "priority": 0,
            "prompt_override": "Briefly describe the video.", "class": "background",
        })
    for tenant, offset in (("b", .75), ("c", 1.25)):
        for index in range(6):
            rows.append({
                "request_id": f"{tenant}-{index}", "qid": f"{tenant}-{index}",
                "tenant": tenant, "modality": "video", "video": str(video),
                "frame_count": 1 if index % 2 == 0 else 16,
                "arrival_s": round(offset + index * 2.0, 6),
                "max_tokens": 32, "priority": 0,
                "prompt_override": "Briefly describe the video.", "class": "background",
            })
    return sorted(rows, key=lambda row: (row["arrival_s"], row["request_id"]))


def summarize(rows):
    def group(selected):
        e2e = [row["end_to_end_s"] for row in selected]
        wait = [row["prep_queue_wait_s"] for row in selected]
        ready = [row["prep_ready_s"] - row["arrival_s"] for row in selected]
        ttft = [row["end_to_end_ttft_s"] for row in selected]
        return {
            "requests": len(selected),
            "mean_e2e_s": statistics.mean(e2e),
            "median_e2e_s": statistics.median(e2e),
            "p95_e2e_s": percentile(e2e, .95),
            "median_ttft_s": statistics.median(ttft),
            "p95_ttft_s": percentile(ttft, .95),
            "mean_prep_wait_s": statistics.mean(wait),
            "p95_prep_wait_s": percentile(wait, .95),
            "median_model_ready_s": statistics.median(ready),
            "p95_model_ready_s": percentile(ready, .95),
            "prep_service_s": sum(row["prep_service_s"] for row in selected),
        }

    all_metrics = group(rows)
    all_metrics["tenants"] = {
        tenant: group([row for row in rows if row["tenant"] == tenant])
        for tenant in ("a", "b", "c")
    }
    victims = [row for row in rows if row["tenant"] in ("b", "c")]
    all_metrics["victims"] = group(victims)
    total_service = sum(item["prep_service_s"] for item in all_metrics["tenants"].values())
    all_metrics["prep_service_shares"] = {
        tenant: item["prep_service_s"] / total_service
        for tenant, item in all_metrics["tenants"].items()
    }
    start = min(row["arrival_s"] for row in rows)
    finish = max(row["completion_s"] for row in rows)
    all_metrics["throughput_qps"] = len(rows) / (finish - start)
    return all_metrics


def write_report(output, records, order):
    lines = [
        "# Preparation noisy-neighbor fairness pilot", "",
        "Tenant A sends a burst of twelve 128-frame requests at 0--0.55 seconds. "
        "Tenants B and C each send six later requests, alternating between one and "
        "sixteen frames. All requests share four CPU preparation workers, four "
        "inference slots, and handoff capacity eight. This is a one-video, one-order "
        "pilot; it is not a significance result.", "",
        "Policy order: " + ", ".join(order) + ".", "",
        "| Policy | Mean E2E (s) | Median E2E (s) | p95 E2E (s) | Victim median (s) | Victim p95 (s) | Victim mean prep wait (s) | Throughput (req/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        metrics = record["metrics"]
        victim = metrics["victims"]
        lines.append(
            f"| {record['policy']} | {metrics['mean_e2e_s']:.2f} | "
            f"{metrics['median_e2e_s']:.2f} | {metrics['p95_e2e_s']:.2f} | "
            f"{victim['median_e2e_s']:.2f} | {victim['p95_e2e_s']:.2f} | "
            f"{victim['mean_prep_wait_s']:.2f} | {metrics['throughput_qps']:.3f} |"
        )
    lines += ["", "Preparation service shares reflect demand as well as allocation:", ""]
    for record in records:
        shares = record["metrics"]["prep_service_shares"]
        lines.append(
            f"- {record['policy']}: " +
            ", ".join(f"{tenant} {shares[tenant]:.1%}" for tenant in ("a", "b", "c"))
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    (output / "records.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("fresh output directory required")
    if not args.video.is_file() or not args.model_snapshot.is_dir():
        parser.error("video and model snapshot must exist")
    output.mkdir(parents=True)

    rows = workload(args.video)
    trace = output / "trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    order = list(POLICIES)
    random.Random(19).shuffle(order)
    manifest = {
        "policies": POLICIES,
        "policy_order": order,
        "requests": len(rows),
        "tenant_counts": {tenant: sum(row["tenant"] == tenant for row in rows)
                          for tenant in ("a", "b", "c")},
        "tenant_frames": {"a": [128], "b": [1, 16], "c": [1, 16]},
        "prep_workers": 4,
        "cpu_decoder_threads": 1,
        "inference_slots": 4,
        "handoff_capacity": 8,
        "decode_backend": "batch_cpu",
        "scope": "one-video one-order mechanism pilot",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in (Path(__file__), RUNNER, RUNNER.parent / "batched_cpu_decode.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               VIDEO_RLM_FFMPEG_THREADS="1")
    server = telemetry = None
    records = []
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2)
        except Exception:
            pass
        else:
            raise RuntimeError(f"port {args.port} already hosts a server")
        (output / "status.json").write_text(json.dumps({
            "state": "starting", "active": "model_server", "completed": 0,
            "total": len(order),
        }) + "\n")
        server_command = [
            str(Path(sys.executable).with_name("vllm")), "serve", str(args.model_snapshot),
            "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", str(args.port),
            "--api-key", "EMPTY", "--dtype", "auto", "--tensor-parallel-size", "1",
            "--max-model-len", "16384", "--gpu-memory-utilization", ".80", "--enforce-eager",
            "--max-num-seqs", "4", "--max-num-batched-tokens", "4096",
            "--limit-mm-per-prompt", '{"image":128,"video":0}',
            "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0",
            "--scheduling-policy", "priority",
        ]
        manifest["server_command"] = server_command
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(server_command, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(150):
            if server.poll() is not None:
                raise RuntimeError("server failed")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server startup timeout")
        with (output / "gpu.csv").open("w") as log:
            telemetry = subprocess.Popen([
                "nvidia-smi", f"--id={args.gpu}",
                "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv", "--loop=1",
            ], stdout=log, start_new_session=True)

        for policy in order:
            (output / "status.json").write_text(json.dumps({
                "state": "running", "active": policy,
                "completed": len(records), "total": len(order),
            }) + "\n")
            command = [
                sys.executable, str(RUNNER), "--arrival-trace", str(trace),
                "--output", str(output / policy), "--port", str(args.port),
                "--model", MODEL, "--prep-policy", policy,
                "--decode-backend", "batch_cpu", "--prep-placement", "fixed",
                "--prep-workers", "4", "--cpu-decoder-threads", "1",
                "--vlm-concurrency", "4", "--prepared-queue-depth", "8",
                "--urgent-prep-reserve", "0", "--ignore-eos",
                "--include-stream-usage", "--decode-timeout-s", "1800",
                "--request-timeout-s", "1800",
            ]
            with (output / f"{policy}.log").open("w") as log:
                subprocess.run(command, check=True, env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=3600)
            result = [json.loads(line) for line in
                      (output / policy / "results.jsonl").read_text().splitlines()
                      if line.strip()]
            if len(result) != len(rows) or any(row.get("error") for row in result):
                raise RuntimeError(f"invalid results for {policy}")
            records.append({"policy": policy, "metrics": summarize(result)})
            write_report(output, records, order)
        (output / "status.json").write_text(json.dumps({
            "state": "complete", "active": None,
            "completed": len(records), "total": len(order),
        }) + "\n")
        (output / "COMPLETE").write_text("All noisy-neighbor policies completed.\n")
    except BaseException as error:
        (output / "status.json").write_text(json.dumps({
            "state": "failed", "error": repr(error),
            "completed": len(records), "total": len(order),
        }) + "\n")
        (output / "FAILED").write_text(repr(error) + "\n")
        raise
    finally:
        if telemetry is not None and telemetry.poll() is None:
            telemetry.terminate()
            telemetry.wait()
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
