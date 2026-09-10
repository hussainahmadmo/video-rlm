#!/usr/bin/env python3
"""Plan disjoint, whole-core CPU sets for independent one-GPU experiments."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Core:
    socket: int
    core: int
    node: int
    cpus: tuple[int, ...]


def expand_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return cpus


def format_cpu_list(cpus: set[int] | tuple[int, ...] | list[int]) -> str:
    values = sorted(cpus)
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def allowed_cpus() -> set[int]:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return expand_cpu_list(line.split(":", 1)[1])
    raise RuntimeError("cannot determine the container's allowed CPU set")


def read_lscpu(text: str, allowed: set[int]) -> list[Core]:
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row in csv.reader(
        line for line in text.splitlines() if line and not line.startswith("#")
    ):
        if len(row) < 4:
            continue
        cpu, core, socket, node = map(int, row[:4])
        if cpu in allowed:
            grouped[(socket, core, node)].append(cpu)
    cores = [
        Core(socket=key[0], core=key[1], node=key[2], cpus=tuple(sorted(cpus)))
        for key, cpus in grouped.items()
    ]
    return sorted(cores, key=lambda item: (item.node, item.socket, item.core))


def discover_gpu_nodes(gpu_count: int) -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: dict[int, str] = {}
    for line in output.splitlines():
        index_text, bus_id = (part.strip() for part in line.split(",", 1))
        rows[int(index_text)] = bus_id
    if any(index not in rows for index in range(gpu_count)):
        raise RuntimeError(f"fewer than {gpu_count} GPUs are visible")

    nodes: list[int] = []
    for index in range(gpu_count):
        match = re.search(
            r"([0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7])$",
            rows[index],
        )
        if not match:
            nodes.append(-1)
            continue
        path = Path("/sys/bus/pci/devices") / match.group(1).lower() / "numa_node"
        try:
            nodes.append(int(path.read_text().strip()))
        except (FileNotFoundError, ValueError):
            nodes.append(-1)
    return nodes


def take_exact_logical_cpus(
    available: list[Core], target: int
) -> tuple[list[Core], list[Core]]:
    selected: list[Core] = []
    total = 0
    for core in available:
        if total + len(core.cpus) > target:
            continue
        selected.append(core)
        total += len(core.cpus)
        if total == target:
            selected_set = set(selected)
            return selected, [core for core in available if core not in selected_set]
    raise RuntimeError(
        f"cannot allocate exactly {target} logical CPUs without splitting a physical core"
    )


def allocate(
    cores: list[Core], gpu_nodes: list[int], prep_vcpus: int, engine_vcpus: int
) -> list[tuple[int, str, str, int]]:
    if prep_vcpus <= 0 or engine_vcpus <= 0:
        raise ValueError("stage CPU counts must be positive")
    remaining = list(cores)
    plans: list[tuple[int, str, str, int]] = []
    warned_remote = False
    for gpu, node in enumerate(gpu_nodes):
        local = [core for core in remaining if node >= 0 and core.node == node]
        required = prep_vcpus + engine_vcpus
        if sum(len(core.cpus) for core in local) >= required:
            candidates = local
        else:
            candidates = remaining
            if node >= 0 and not warned_remote:
                print(
                    "warning: insufficient GPU-local CPUs; using disjoint remote CPUs where needed",
                    file=sys.stderr,
                )
                warned_remote = True
        lane_cores, _ = take_exact_logical_cpus(candidates, required)
        lane_set = set(lane_cores)
        remaining = [core for core in remaining if core not in lane_set]
        prep_cores, engine_cores = take_exact_logical_cpus(lane_cores, prep_vcpus)
        prep = {cpu for core in prep_cores for cpu in core.cpus}
        engine = {cpu for core in engine_cores for cpu in core.cpus}
        lane_nodes = {core.node for core in lane_cores}
        effective_node = next(iter(lane_nodes)) if len(lane_nodes) == 1 else -1
        plans.append(
            (gpu, format_cpu_list(prep), format_cpu_list(engine), effective_node)
        )
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--prep-vcpus", type=int, default=8)
    parser.add_argument("--engine-vcpus", type=int, default=8)
    parser.add_argument("--base-port", type=int, default=9000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lscpu-file", type=Path)
    parser.add_argument("--gpu-numa-nodes", help="testing override, e.g. 0,0,1,1")
    args = parser.parse_args()
    if args.gpu_count <= 0:
        parser.error("--gpu-count must be positive")
    allowed = allowed_cpus()
    lscpu_text = (
        args.lscpu_file.read_text()
        if args.lscpu_file
        else subprocess.check_output(
            ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True
        )
    )
    cores = read_lscpu(lscpu_text, allowed)
    if args.gpu_numa_nodes:
        gpu_nodes = [int(value) for value in args.gpu_numa_nodes.split(",")]
        if len(gpu_nodes) != args.gpu_count:
            parser.error("--gpu-numa-nodes must contain one entry per GPU")
    else:
        gpu_nodes = discover_gpu_nodes(args.gpu_count)
    try:
        plans = allocate(cores, gpu_nodes, args.prep_vcpus, args.engine_vcpus)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"cannot create isolated CPU plan: {exc}") from exc
    lines = ["gpu\tport\tprep_cpus\tengine_cpus\tcpu_numa_node"]
    for gpu, prep, engine, node in plans:
        lines.append(f"{gpu}\t{args.base_port + gpu}\t{prep}\t{engine}\t{node}")
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
