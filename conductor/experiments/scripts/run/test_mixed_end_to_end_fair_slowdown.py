from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


RUNNER = Path(__file__).with_name("run_mixed_end_to_end_priority.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("fair_slowdown_runner", RUNNER)
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
        return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.models = FakeModels()
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FairSlowdownTest(unittest.TestCase):
    def test_batch_aware_token_profile_interpolates_and_reconciles(self) -> None:
        runner = load_runner()
        profile = {
            "fits_by_concurrency": {
                "1": {
                    "prefill_fit": {"intercept": 0.1, "slope": 0.001},
                    "decode_fit": {"intercept": 0.05, "slope": 0.01},
                    "visual_fit": {"intercept": 0.08, "slope": 0.002},
                },
                "3": {
                    "prefill_fit": {"intercept": 0.05, "slope": 0.0005},
                    "decode_fit": {"intercept": 0.03, "slope": 0.005},
                    "visual_fit": {"intercept": 0.04, "slope": 0.001},
                },
            },
            "aggregated_points": [
                {"family": "visual", "target": 0, "concurrency": 1,
                 "mean_prompt_tokens": 10},
                {"family": "visual", "target": 1, "concurrency": 1,
                 "mean_prompt_tokens": 20},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.json"
            path.write_text(json.dumps(profile))
            model = runner.BatchAwareTokenCostProfile(path)
            job = {
                "modality": "video", "frame_count": 2, "max_tokens": 4,
                "row": {"estimated_text_tokens": 100},
            }
            # At concurrency two, every coefficient is halfway between its
            # concurrency-one and concurrency-three measurement.
            self.assertAlmostEqual(
                model.predict(job, concurrency=2), 0.21, places=8,
            )
            # API usage replaces the estimated text and output counts. The
            # prompt has 20 visual tokens, leaving 30 text tokens.
            self.assertAlmostEqual(
                model.predict(
                    job, concurrency=2, prompt_tokens=50,
                    completion_tokens=2,
                ),
                0.1425,
                places=8,
            )

    def test_frame_profiler_preserves_original_ewma_behavior(self) -> None:
        runner = load_runner()
        profiler = runner.OnlineStageCostProfiler(
            stage="prep", mode="frame_ewma", backend="seek_cpu", alpha=0.5,
        )
        job = {"frame_count": 8, "row": {}}
        self.assertEqual(profiler.predict(job, default_s=2.0), 2.0)
        profiler.observe(job, observed_s=6.0, predicted_s=2.0)
        self.assertEqual(profiler.predict(job, default_s=2.0), 6.0)
        profiler.observe(job, observed_s=2.0, predicted_s=6.0)
        self.assertEqual(profiler.predict(job, default_s=2.0), 4.0)

    def test_metadata_profiler_falls_back_when_metadata_is_missing(self) -> None:
        runner = load_runner()
        profiler = runner.OnlineStageCostProfiler(
            stage="prep", mode="metadata_ewma", backend="seek_cpu", alpha=1.0,
        )
        profiled = {
            "frame_count": 32,
            "row": {
                "duration_s": 95.0,
                "video_width": 1280,
                "video_height": 720,
                "video_codec": "h264",
            },
        }
        profiler.observe(profiled, observed_s=9.0, predicted_s=4.0)

        # A new trace without media metadata still benefits from the learned
        # backend/frame projection instead of reverting to the cold start.
        missing_metadata = {"frame_count": 32, "row": {}}
        self.assertEqual(
            profiler.predict(missing_metadata, default_s=4.0), 9.0,
        )

    def test_metadata_profiler_uses_specific_profile_when_available(self) -> None:
        runner = load_runner()
        profiler = runner.OnlineStageCostProfiler(
            stage="engine", mode="metadata_ewma", backend="indexed_nvdec",
            alpha=1.0,
        )
        short = {
            "frame_count": 8,
            "row": {"duration_s": 20.0, "resolution": "640x480"},
        }
        long = {
            "frame_count": 8,
            "row": {"duration_s": 900.0, "resolution": "1920x1080"},
        }
        profiler.observe(short, observed_s=2.0, predicted_s=4.0)
        profiler.observe(long, observed_s=8.0, predicted_s=4.0)
        self.assertEqual(profiler.predict(short, default_s=4.0), 2.0)
        self.assertEqual(profiler.predict(long, default_s=4.0), 8.0)
        snapshot = profiler.snapshot()
        self.assertEqual(snapshot["observations"], 2)
        self.assertEqual(snapshot["mean_absolute_error_s"], 3.0)
        self.assertEqual(
            snapshot["mean_absolute_percentage_error_percent"], 75.0,
        )

    def test_metadata_index_is_attached_before_scheduling(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "video.mp4"
            metadata_path = Path(temp_dir) / "metadata.jsonl"
            metadata_path.write_text(json.dumps({
                "video": str(video),
                "duration_s": 95.0,
                "video_width": 1280,
                "video_height": 720,
                "video_codec": "h264",
            }) + "\n")
            index = runner.load_video_metadata_index(metadata_path)
            row = {"video": str(video), "video_codec": "trace-override"}
            runner.attach_video_metadata(row, index)
            self.assertEqual(row["duration_s"], 95.0)
            self.assertEqual(row["video_width"], 1280)
            self.assertEqual(row["video_codec"], "trace-override")

    def test_parse_tenant_weights(self) -> None:
        runner = load_runner()
        self.assertEqual(
            runner.parse_tenant_weights(["alpha=2", "beta=0.5"]),
            {"alpha": 2.0, "beta": 0.5},
        )
        with self.assertRaises(ValueError):
            runner.parse_tenant_weights(["alpha=0"])

    def test_max_min_rejects_non_unit_tenant_weights(self) -> None:
        runner = load_runner()
        self.assertIn("max_min", runner.CROSS_STAGE_FAIR_POLICIES)
        self.assertIn("prep_max_min", runner.PREPARATION_FAIR_POLICIES)
        self.assertNotIn("prep_max_min", runner.INFERENCE_FAIR_POLICIES)
        self.assertIn("tenant_round_robin", runner.SCHEDULER_OWNED_POLICIES)
        self.assertIn("max_min", runner.INFERENCE_FAIR_POLICIES)
        self.assertIn("max_min", runner.SCHEDULER_OWNED_POLICIES)
        runner.validate_max_min_tenant_weights({"alpha": 1.0, "beta": 1.0})
        with self.assertRaisesRegex(ValueError, "requires unit tenant weights"):
            runner.validate_max_min_tenant_weights({"alpha": 2.0})

    def test_slowdown_uses_measured_wait_and_solo_estimate(self) -> None:
        runner = load_runner()
        self.assertEqual(
            runner.predicted_slowdown(
                elapsed_s=20.0, remaining_s=5.0, solo_s=5.0,
            ),
            5.0,
        )

    def test_fair_share_then_slowdown(self) -> None:
        runner = load_runner()
        jobs = []
        sequence = 0
        for tenant in ("a", "b", "c"):
            for index in range(3):
                jobs.append(
                    {
                        "tenant": tenant,
                        "arrival_s": float(index),
                        "sequence": sequence,
                        "remaining_s": 1.0,
                        "solo_s": 1.0,
                    }
                )
                sequence += 1

        virtual_service = {"a": 0.0, "b": 0.0, "c": 0.0}
        selected_tenants = []
        for _ in range(5):
            selected = runner.fair_slowdown_choice(
                jobs,
                now_s=10.0,
                tenant_virtual_service=virtual_service,
                remaining_service_s=lambda job: job["remaining_s"],
                solo_service_s=lambda job: job["solo_s"],
            )
            selected_tenants.append(selected["tenant"])
            virtual_service[selected["tenant"]] += selected["remaining_s"]
            jobs.remove(selected)

        self.assertEqual(Counter(selected_tenants), {"a": 2, "b": 2, "c": 1})
        # Within tenant a, the oldest request has the largest slowdown.
        self.assertEqual(selected_tenants[0], "a")

    def test_tenant_fair_uses_fcfs_within_tenant(self) -> None:
        runner = load_runner()
        jobs = [
            {"tenant": "a", "arrival_s": 2.0, "sequence": 2},
            {"tenant": "a", "arrival_s": 1.0, "sequence": 1},
            {"tenant": "b", "arrival_s": 0.0, "sequence": 0},
        ]
        selected = runner.fair_tenant_choice(
            jobs,
            tenant_virtual_service={"a": 0.0, "b": 1.0},
        )
        self.assertEqual(selected["tenant"], "a")
        self.assertEqual(selected["sequence"], 1)

    def test_tenant_priority_is_fair_across_tenants(self) -> None:
        runner = load_runner()
        jobs = [
            {
                "tenant": "a", "workload": "urgent", "priority": 0,
                "arrival_s": 5.0, "sequence": 1,
            },
            {
                "tenant": "b", "workload": "background", "priority": 10,
                "arrival_s": 0.0, "sequence": 0,
            },
        ]
        selected = runner.fair_tenant_priority_choice(
            jobs,
            now_s=10.0,
            tenant_virtual_service={"a": 5.0, "b": 0.0},
            background_aging_s=120.0,
        )
        self.assertEqual(selected["tenant"], "b")

    def test_tenant_priority_prefers_urgent_within_tenant(self) -> None:
        runner = load_runner()
        jobs = [
            {
                "tenant": "a", "workload": "background", "priority": 10,
                "arrival_s": 0.0, "sequence": 0,
            },
            {
                "tenant": "a", "workload": "urgent", "priority": 0,
                "arrival_s": 5.0, "sequence": 1,
            },
        ]
        selected = runner.fair_tenant_priority_choice(
            jobs,
            now_s=10.0,
            tenant_virtual_service={"a": 0.0},
            background_aging_s=120.0,
        )
        self.assertEqual(selected["workload"], "urgent")

    def test_tenant_priority_ages_background_within_tenant(self) -> None:
        runner = load_runner()
        jobs = [
            {
                "tenant": "a", "workload": "background", "priority": 10,
                "arrival_s": 0.0, "sequence": 0,
            },
            {
                "tenant": "a", "workload": "urgent", "priority": 0,
                "arrival_s": 9.0, "sequence": 1,
            },
        ]
        selected = runner.fair_tenant_priority_choice(
            jobs,
            now_s=130.0,
            tenant_virtual_service={"a": 0.0},
            background_aging_s=120.0,
        )
        self.assertEqual(selected["workload"], "background")

    def test_shortest_job_first_uses_predicted_service(self) -> None:
        runner = load_runner()
        jobs = [
            {"sequence": 0, "service_s": 8.0},
            {"sequence": 1, "service_s": 2.0},
            {"sequence": 2, "service_s": 4.0},
        ]
        selected = runner.shortest_job_choice(
            jobs, lambda job: job["service_s"],
        )
        self.assertEqual(selected["sequence"], 1)

    def test_max_min_policy_runs_end_to_end_and_records_tenants(self) -> None:
        runner = load_runner()

        def fake_prepare(job, _codec, _codec_args):
            time.sleep(0.005)
            return {
                "content": [{"type": "text", "text": "test"}],
                "duration_s": 1.0,
                "timestamps": [0.0],
                "decode_service_s": 0.005,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace = root / "trace.jsonl"
            rows = []
            for index, tenant in enumerate(("a", "a", "b", "b", "c", "c")):
                rows.append(
                    {
                        "request_id": f"request-{index}",
                        "class": "background",
                        "tenant": tenant,
                        "arrival_s": 0.0,
                        "frame_count": 8,
                        "priority": index,
                        "solo_service_s": 1.0,
                        "qid": f"qid-{index}",
                        "video": "/unused/video.mp4",
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
                "--port", "9000",
                "--prep-policy", "max_min",
                "--prep-workers", "2",
                "--vlm-concurrency", "2",
                "--prepared-queue-depth", "4",
                "--prep-cost-profiler", "metadata_ewma",
                "--engine-cost-profiler", "frame_ewma",
                "--tenant-weight", "a=1",
                "--tenant-weight", "b=1",
                "--tenant-weight", "c=1",
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
            self.assertEqual(Counter(row["tenant"] for row in results), {
                "a": 2, "b": 2, "c": 2,
            })
            self.assertTrue(
                all(row["engine_priority"] == 0 for row in results)
            )
            self.assertTrue(
                all(row["profiled_solo_service_s"] == 1.0 for row in results)
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["prep_policy"], "max_min")
            self.assertEqual(summary["fairness_mode"], "unweighted_max_min")
            self.assertEqual(set(summary["tenants"]), {"a", "b", "c"})
            self.assertEqual(
                summary["configuration"]["prep_cost_profiler"],
                "metadata_ewma",
            )
            self.assertEqual(
                summary["configuration"]["engine_cost_profiler"],
                "frame_ewma",
            )

    def test_new_fairness_policy_modes_run_end_to_end(self) -> None:
        runner = load_runner()

        def fake_prepare(job, _codec, _codec_args):
            time.sleep(0.002)
            return {
                "content": [{"type": "text", "text": "test"}],
                "duration_s": 1.0,
                "timestamps": [0.0],
                "decode_service_s": 0.002,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace = root / "trace.jsonl"
            rows = [
                {
                    "request_id": f"request-{tenant}",
                    "class": "background",
                    "tenant": tenant,
                    "arrival_s": 0.0,
                    "frame_count": 8,
                    "priority": 0,
                    "qid": f"qid-{tenant}",
                    "video": "/unused/video.mp4",
                    "choices": ["one", "two"],
                    "answer_label": "A",
                }
                for tenant in ("a", "b", "c")
            ]
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

            cases = (
                ("prep_max_min", [], "preparation_only_unweighted_max_min"),
                ("tenant_round_robin", [], "tenant_round_robin"),
                ("max_min", ["--completion-only-accounting"], "unweighted_max_min"),
                (
                    "age_aware_max_min",
                    ["--age-soft-threshold-s", "1", "--age-hard-threshold-s", "2"],
                    "unweighted_max_min_with_age_protection",
                ),
            )
            for policy, extra, fairness_mode in cases:
                output = root / policy
                argv = [
                    str(RUNNER),
                    "--arrival-trace", str(trace),
                    "--output", str(output),
                    "--port", "9000",
                    "--prep-policy", policy,
                    "--prep-workers", "2",
                    "--vlm-concurrency", "2",
                    "--prepared-queue-depth", "4",
                    *extra,
                ]
                with (
                    mock.patch.object(runner, "OpenAI", FakeOpenAI),
                    mock.patch.object(runner, "prepare_uniform", fake_prepare),
                    mock.patch.object(runner, "import_path", return_value=object()),
                    mock.patch.object(sys, "argv", argv),
                    redirect_stdout(io.StringIO()),
                ):
                    runner.main()
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["fairness_mode"], fairness_mode)
                self.assertEqual(summary["errors"], 0)
                if policy == "prep_max_min":
                    self.assertTrue(any(
                        value > 0 for value in summary["configuration"][
                            "tenant_prep_virtual_service"
                        ].values()
                    ))
                    self.assertTrue(all(
                        value == 0 for value in summary["configuration"][
                            "tenant_vlm_virtual_service"
                        ].values()
                    ))
                elif policy == "tenant_round_robin":
                    self.assertEqual(
                        set(summary["configuration"]["tenant_prep_dispatches"].values()),
                        {1.0},
                    )
                    self.assertEqual(
                        set(summary["configuration"]["tenant_vlm_dispatches"].values()),
                        {1.0},
                    )
                elif policy == "max_min":
                    self.assertEqual(
                        summary["service_accounting_mode"], "completion_only"
                    )
                else:
                    self.assertEqual(
                        summary["service_accounting_mode"],
                        "predicted_then_reconciled",
                    )
                    self.assertEqual(
                        summary["configuration"]["age_soft_threshold_s"], 1.0
                    )
                    self.assertEqual(
                        summary["configuration"]["age_hard_threshold_s"], 2.0
                    )


if __name__ == "__main__":
    unittest.main()
