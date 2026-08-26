from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


RUNNER = Path(__file__).with_name("run_mixed_end_to_end_priority.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("static_isolation_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeModels:
    def list(self):
        return []


class FakeCompletions:
    def create(self, **_kwargs):
        delta = SimpleNamespace(content="A")
        choice = SimpleNamespace(delta=delta)
        return [SimpleNamespace(choices=[choice])]


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.models = FakeModels()
        self.chat = SimpleNamespace(completions=FakeCompletions())


class StaticIsolationTest(unittest.TestCase):
    def test_prep_and_replica_boundaries(self) -> None:
        runner = load_runner()

        def fake_prepare(job, _codec, _codec_args):
            # Keep background tasks active long enough for the urgent request
            # to arrive and exercise its dedicated preparation slot.
            time.sleep(0.08 if job["workload"] == "background" else 0.01)
            return {
                "content": [{"type": "text", "text": "test"}],
                "duration_s": 1.0,
                "timestamps": [0.0],
                "decode_service_s": 0.01,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace = root / "trace.jsonl"
            rows = []
            for index in range(5):
                rows.append(
                    {
                        "request_id": f"background-{index}",
                        "class": "background",
                        "arrival_s": 0.0,
                        "frame_count": 8,
                        "priority": 10,
                        "qid": f"background-qid-{index}",
                        "video": "/unused/background.mp4",
                        "choices": ["one", "two"],
                        "answer_label": "A",
                    }
                )
            rows.append(
                {
                    "request_id": "urgent-0",
                    "class": "urgent",
                    "arrival_s": 0.01,
                    "frame_count": 8,
                    "priority": 0,
                    "qid": "urgent-qid-0",
                    "video": "/unused/urgent.mp4",
                    "choices": ["one", "two"],
                    "answer_label": "A",
                }
            )
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

            output = root / "output"
            argv = [
                str(RUNNER),
                "--arrival-trace", str(trace),
                "--output", str(output),
                "--ports", "9000", "9001",
                "--replica-routing", "workload_isolated",
                "--background-ports", "9000",
                "--urgent-ports", "9001",
                "--prep-policy", "static_isolation",
                "--prep-workers", "4",
                "--background-prep-workers", "3",
                "--urgent-prep-workers", "1",
                "--vlm-concurrency", "1",
                "--prepared-queue-depth", "8",
                "--background-prepared-queue-depth", "6",
                "--urgent-prepared-queue-depth", "2",
            ]

            with (
                mock.patch.object(runner, "OpenAI", FakeOpenAI),
                mock.patch.object(runner, "prepare_uniform", fake_prepare),
                mock.patch.object(runner, "import_path", return_value=object()),
                mock.patch.object(sys, "argv", argv),
            ):
                runner.main()

            results = [
                json.loads(line)
                for line in (output / "results.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(results), 6)
            self.assertTrue(
                all(
                    row["replica_port"]
                    == (9001 if row["workload"] == "urgent" else 9000)
                    for row in results
                )
            )

            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            urgent_start = next(
                row for row in events
                if row["event"] == "prep_start"
                and row["request_id"] == "urgent-0"
            )
            background_completions_before_urgent = [
                row for row in events
                if row["event"] == "prep_ready"
                and row["workload"] == "background"
                and row["time_s"] < urgent_start["time_s"]
            ]
            self.assertFalse(background_completions_before_urgent)

            background_starts_before_urgent = [
                row for row in events
                if row["event"] == "prep_start"
                and row["workload"] == "background"
                and row["time_s"] < urgent_start["time_s"]
            ]
            self.assertEqual(len(background_starts_before_urgent), 3)

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["background_prep_workers"], 3)
            self.assertEqual(summary["urgent_prep_workers"], 1)
            self.assertEqual(summary["background_ports"], [9000])
            self.assertEqual(summary["urgent_ports"], [9001])
            self.assertEqual(summary["background_prepared_queue_depth"], 6)
            self.assertEqual(summary["urgent_prepared_queue_depth"], 2)


if __name__ == "__main__":
    unittest.main()
