#!/usr/bin/env python3
"""Run one or all paper fairness policies through the same serving pipeline.

This is a deliberately small public entry point.  The execution machinery
(trace replay, preparation, bounded handoff, vLLM calls, accounting, and
logging) remains in ``run_mixed_end_to_end_priority.py`` so every policy is
compared with identical infrastructure.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).with_name("run_mixed_end_to_end_priority.py")
POLICIES = (
    "fcfs",
    "tenant_round_robin",
    "engine_tenant_fair",
    "prep_max_min",
    "max_min",
    "age_aware_max_min",
    "cross_stage",
)
POLICY_HELP = {
    "fcfs": "FCFS at preparation and inference admission",
    "tenant_round_robin": "equal request turns per tenant at both stages",
    "engine_tenant_fair": "FCFS preparation and fair inference admission",
    "prep_max_min": "fair preparation and FCFS inference admission",
    "max_min": "independent fair-service accounting at both stages",
    "age_aware_max_min": "two-stage max-min with soft and hard age protection",
    "cross_stage": "two-stage accounting with bounded GPU-debt feedback",
}


def selected_policies(methods: list[str] | None) -> list[str]:
    if not methods or methods == ["all"]:
        return list(POLICIES)
    if "all" in methods:
        raise ValueError("use --method all alone, or list individual methods")
    # Preserve command-line order while avoiding accidental duplicate runs.
    return list(dict.fromkeys(methods))


def build_command(
    *,
    python: str,
    method: str,
    trace: Path,
    output_root: Path,
    runner_args: list[str],
) -> list[str]:
    forbidden = {"--prep-policy", "--arrival-trace", "--output"}
    conflicts = forbidden.intersection(runner_args)
    if conflicts:
        raise ValueError(
            "wrapper owns these arguments: " + ", ".join(sorted(conflicts))
        )
    return [
        python,
        str(RUNNER),
        "--prep-policy",
        method,
        "--arrival-trace",
        str(trace),
        "--output",
        str(output_root / method),
        *runner_args,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrival-trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        choices=[*POLICIES, "all"],
        help="policy to run; repeat the option or use 'all' (default: all)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the underlying experiment runner",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without contacting vLLM or running preparation",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="print the supported methods and exit",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="common runner options after '--', such as --ports 9000 9001",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_methods:
        for policy in POLICIES:
            print(f"{policy:22s} {POLICY_HELP[policy]}")
        return 0

    if not args.arrival_trace.is_file():
        raise SystemExit(f"arrival trace does not exist: {args.arrival_trace}")

    runner_args = list(args.runner_args)
    if runner_args[:1] == ["--"]:
        runner_args.pop(0)

    try:
        methods = selected_policies(args.method)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_root.mkdir(parents=True, exist_ok=True)
    for method in methods:
        try:
            command = build_command(
                python=args.python,
                method=method,
                trace=args.arrival_trace,
                output_root=args.output_root,
                runner_args=runner_args,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"\n[{method}] {POLICY_HELP[method]}", flush=True)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
