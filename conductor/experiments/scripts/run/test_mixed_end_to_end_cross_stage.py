from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_mixed_end_to_end_priority.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("cross_stage_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CrossStageChoiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()
        self.jobs = [
            {"tenant": "a", "arrival_s": 0.0, "sequence": 0},
            {"tenant": "c", "arrival_s": 0.0, "sequence": 1},
        ]

    def choose(self, *, supply=None, max_boost_s=4.0):
        return self.runner.cross_stage_tenant_choice(
            self.jobs,
            tenant_prep_virtual_service={"a": 4.0, "c": 5.0},
            tenant_vlm_virtual_service={"a": 8.0, "c": 2.0},
            active_tenants={"a", "c"},
            tenant_pipeline_supply=supply or {},
            debt_threshold_s=0.0,
            max_boost_s=max_boost_s,
            unblock_target=1,
        )

    def test_gpu_behind_blocked_tenant_is_unblocked_at_cpu(self) -> None:
        selected, decision = self.choose()
        self.assertEqual(selected["tenant"], "c")
        self.assertTrue(decision["unblocking"])
        self.assertEqual(decision["gpu_debt_s"], 6.0)
        self.assertEqual(decision["boost_s"], 4.0)

    def test_existing_pipeline_supply_removes_boost(self) -> None:
        selected, decision = self.choose(supply={"c": 1})
        self.assertEqual(selected["tenant"], "a")
        self.assertFalse(decision["unblocking"])
        self.assertEqual(decision["boost_s"], 0.0)

    def test_boost_is_bounded_to_protect_cpu_fairness(self) -> None:
        selected, _ = self.choose(max_boost_s=0.5)
        self.assertEqual(selected["tenant"], "a")

    def test_cross_stage_policy_is_fair_at_both_stages(self) -> None:
        self.assertIn("cross_stage", self.runner.PREPARATION_FAIR_POLICIES)
        self.assertIn("cross_stage", self.runner.INFERENCE_FAIR_POLICIES)
        self.assertIn("cross_stage", self.runner.SCHEDULER_OWNED_POLICIES)


if __name__ == "__main__":
    unittest.main()
