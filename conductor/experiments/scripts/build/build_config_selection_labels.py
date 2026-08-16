from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conductor.profiler.resource_aware_selector import TIER_ORDER, tier_index


def load_jsonl_path(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return load_jsonl_path(path)

    rows = []
    files = sorted(path.glob("results*.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    for file in files:
        rows.extend(load_jsonl_path(file))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def config_name(row: dict[str, Any]) -> str:
    return str(
        row.get("config_name")
        or row.get("chosen_config")
        or row.get("config")
        or row.get("name")
        or ""
    )


def latency_s(row: dict[str, Any]) -> float:
    for key in (
        "latency_s",
        "wall_latency_s",
        "execution_latency_s",
        "agent_internal_latency_s",
    ):
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def is_correct(row: dict[str, Any]) -> bool:
    for key in ("correct", "is_correct", "final_correct"):
        if key in row:
            return bool(row[key])

    pred = str(
        row.get("prediction")
        or row.get("prediction_label")
        or row.get("final_answer")
        or row.get("answer")
        or ""
    ).strip().upper()
    gold = str(
        row.get("gold")
        or row.get("answer_label")
        or row.get("label")
        or ""
    ).strip().upper()
    return bool(pred and gold and pred[0] == gold[0])


def row_for_training(
    source: dict[str, Any],
    *,
    label_config: str,
    oracle_correct: bool,
    cheapest_correct_config: str | None,
    fastest_config: str | None,
    num_correct_configs: int,
) -> dict[str, Any]:
    return {
        "dataset": source.get("dataset"),
        "qid": qid(source),
        "video_id": source.get("video_id"),
        "duration_s": source.get("duration_s"),
        "duration_bucket": source.get("duration_bucket"),
        "question": source.get("question"),
        "choices": source.get("choices"),
        "answer_idx": source.get("answer_idx"),
        "answer_label": source.get("answer_label"),
        "answer": source.get("answer"),
        "question_category": source.get("question_category"),
        "topic_category": source.get("topic_category"),
        "vimio_profile": source.get("vimio_profile"),
        "label_config": label_config,
        "oracle_correct": oracle_correct,
        "cheapest_correct_config": cheapest_correct_config,
        "fastest_config": fastest_config,
        "num_correct_configs": num_correct_configs,
    }


def choose_label(
    rows: list[dict[str, Any]],
    *,
    fallback: str,
) -> tuple[str, bool, str | None, str | None, int]:
    tier_rows = [
        row
        for row in rows
        if config_name(row) in TIER_ORDER
    ]
    if not tier_rows:
        raise ValueError(
            "no rows use supported selector configs: "
            + ", ".join(TIER_ORDER)
        )

    correct_rows = [
        row
        for row in tier_rows
        if is_correct(row)
    ]
    fastest = min(
        tier_rows,
        key=lambda row: (latency_s(row), tier_index(config_name(row))),
    )
    fastest_name = config_name(fastest)

    if correct_rows:
        chosen = min(
            correct_rows,
            key=lambda row: (
                tier_index(config_name(row)),
                latency_s(row),
            ),
        )
        chosen_name = config_name(chosen)
        return (
            chosen_name,
            True,
            chosen_name,
            fastest_name,
            len({config_name(row) for row in correct_rows}),
        )

    if fallback == "fastest":
        return fastest_name, False, None, fastest_name, 0
    if fallback in TIER_ORDER:
        return fallback, False, None, fastest_name, 0
    raise ValueError(f"unsupported fallback: {fallback}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fallback",
        default="scan0.0039_k8_budget32",
        help=(
            "Label to use when no supported config is correct. "
            "Use a tier name or 'fastest'."
        ),
    )
    parser.add_argument(
        "--only-oracle-correct",
        action="store_true",
        help="Write only questions where at least one supported config was correct.",
    )
    args = parser.parse_args()

    rows = load_result_rows(Path(args.fixed_results))
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if qid(row) != "None":
            by_qid[qid(row)].append(row)

    labels = []
    for question_id, question_rows in sorted(by_qid.items()):
        (
            label,
            oracle_correct,
            cheapest_correct,
            fastest,
            num_correct,
        ) = choose_label(question_rows, fallback=args.fallback)
        if args.only_oracle_correct and not oracle_correct:
            continue
        labels.append(
            row_for_training(
                question_rows[0],
                label_config=label,
                oracle_correct=oracle_correct,
                cheapest_correct_config=cheapest_correct,
                fastest_config=fastest,
                num_correct_configs=num_correct,
            )
        )

    write_jsonl(labels, Path(args.output))
    print(f"fixed rows: {len(rows)}")
    print(f"questions: {len(by_qid)}")
    print(f"labels written: {len(labels)}")
    counts: dict[str, int] = defaultdict(int)
    oracle_correct_count = 0
    for row in labels:
        counts[str(row["label_config"])] += 1
        oracle_correct_count += int(bool(row["oracle_correct"]))
    print(f"oracle-correct labels: {oracle_correct_count}")
    print("label counts:")
    for name in TIER_ORDER:
        if counts.get(name):
            print(f"  {name}: {counts[name]}")


if __name__ == "__main__":
    main()
