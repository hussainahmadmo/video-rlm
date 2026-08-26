#!/usr/bin/env python3
"""Launch the installed vLLM with an optional native-media scheduler.

Examples:

  # Stock vLLM behavior (no monkey-patching):
  python run_vllm_with_media_priority.py serve MODEL --port 9000

  # Bound native fetch/decode to four active jobs and prioritize requests:
  python run_vllm_with_media_priority.py \
      --media-preparation-policy priority \
      --media-preparation-workers 4 \
      serve MODEL --port 9000 --scheduling-policy priority
"""

from __future__ import annotations

import argparse
import sys


def _custom_parser() -> argparse.ArgumentParser:
    # Disable abbreviation so vLLM's own ``--help`` is not interpreted as
    # this launcher's ``--help-media-priority``.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--media-preparation-policy",
        choices=("native", "priority"),
        default="native",
        help="native keeps stock vLLM; priority enables bounded priority admission",
    )
    parser.add_argument(
        "--media-preparation-workers",
        type=int,
        default=4,
        help="maximum concurrent native media fetch/decode jobs (default: 4)",
    )
    parser.add_argument(
        "--media-preparation-max-pending",
        type=int,
        default=0,
        help="pending-media queue bound; 0 means unbounded (default: 0)",
    )
    parser.add_argument("--help-media-priority", action="store_true")
    return parser


def main() -> None:
    parser = _custom_parser()
    custom, vllm_args = parser.parse_known_args()

    if custom.help_media_priority:
        parser.print_help()
        return
    if custom.media_preparation_workers < 1:
        parser.error("--media-preparation-workers must be at least 1")
    if custom.media_preparation_max_pending < 0:
        parser.error("--media-preparation-max-pending cannot be negative")

    if custom.media_preparation_policy == "priority":
        from vllm_media_priority_plugin import install

        install(
            max_active_jobs=custom.media_preparation_workers,
            max_pending_jobs=custom.media_preparation_max_pending,
        )

    # Hand all remaining arguments to the unmodified installed vLLM CLI.
    sys.argv = [sys.argv[0], *vllm_args]
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
