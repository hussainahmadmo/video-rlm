from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]

    rows = []
    for file in sorted(path.glob("results*.jsonl")):
        rows.extend(load_jsonl(file))
    return rows


def qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def config_name(row: dict[str, Any]) -> str:
    return str(
        row.get("config_name")
        or row.get("chosen_config")
        or row.get("config")
        or ""
    )


def is_correct(row: dict[str, Any]) -> bool:
    if "correct" in row:
        return bool(row["correct"])

    pred = str(
        row.get("prediction")
        or row.get("prediction_label")
        or row.get("final_answer")
        or ""
    ).strip().upper()
    gold = str(
        row.get("gold")
        or row.get("answer_label")
        or row.get("label")
        or ""
    ).strip().upper()
    return bool(pred and gold and pred[0] == gold[0])


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
    stage = row.get("stage_latency_s") or {}
    if stage.get("total_s") is not None:
        return float(stage["total_s"])
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--miss-dataset", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    misses = load_jsonl(Path(args.miss_dataset))
    rows = load_jsonl(Path(args.results))
    missed_qids = {qid(row) for row in misses}

    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_id = qid(row)
        if question_id in missed_qids:
            by_qid[question_id].append(row)

    recovered = {}
    for question_id, question_rows in by_qid.items():
        correct_rows = [row for row in question_rows if is_correct(row)]
        if correct_rows:
            recovered[question_id] = min(
                correct_rows,
                key=lambda row: latency_s(row),
            )

    recovered_by_config = Counter(
        config_name(row)
        for row in recovered.values()
    )
    recovered_by_dataset = Counter()
    miss_by_qid = {qid(row): row for row in misses}
    for question_id in recovered:
        recovered_by_dataset[miss_by_qid[question_id].get("dataset")] += 1

    print(f"miss questions: {len(missed_qids)}")
    print(f"result rows: {len(rows)}")
    print(f"evaluated miss questions: {len(by_qid)}")
    print(f"recovered: {len(recovered)}")
    if missed_qids:
        print(f"recovery rate: {100 * len(recovered) / len(missed_qids):.2f}%")
    print("recovered by config:")
    for name, count in recovered_by_config.most_common():
        print(f"  {name}: {count}")
    print("recovered by dataset:")
    for name, count in recovered_by_dataset.most_common():
        print(f"  {name}: {count}")

    print("examples:")
    for question_id, row in list(recovered.items())[:20]:
        miss = miss_by_qid[question_id]
        print()
        print(f"qid: {question_id}")
        print(f"dataset: {miss.get('dataset')}")
        print(f"config: {config_name(row)}")
        print(f"latency_s: {latency_s(row):.3f}")
        print(f"question: {miss.get('question')}")
        print(f"pred: {row.get('prediction_label') or row.get('prediction')}")
        print(f"gold: {row.get('answer_label') or row.get('gold')}")


if __name__ == "__main__":
    main()
