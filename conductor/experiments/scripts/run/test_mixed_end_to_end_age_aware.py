from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_mixed_end_to_end_priority.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("age_aware_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AgeAwareChoiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()

    def choose(self, jobs, *, now=50.0, soft=75.0, hard=150.0, counters=None):
        return self.runner.age_aware_fair_tenant_choice(
            jobs,
            now_s=now,
            tenant_virtual_service=counters or {},
            soft_threshold_s=soft,
            hard_threshold_s=hard,
        )

    def test_below_soft_threshold_is_ordinary_max_min(self) -> None:
        jobs = [
            {"tenant": "a", "arrival_s": 0.0, "sequence": 0},
            {"tenant": "b", "arrival_s": 10.0, "sequence": 1},
        ]
        selected, decision = self.choose(
            jobs, counters={"a": 5.0, "b": 1.0},
        )
        self.assertEqual(selected["tenant"], "b")
        self.assertEqual(decision["mode"], "max_min")

    def test_soft_threshold_restricts_selection_to_aged_tenants(self) -> None:
        jobs = [
            {"tenant": "a", "arrival_s": 0.0, "sequence": 0},
            {"tenant": "b", "arrival_s": 80.0, "sequence": 1},
            {"tenant": "c", "arrival_s": 30.0, "sequence": 2},
        ]
        selected, decision = self.choose(
            jobs, now=100.0, soft=75.0, hard=150.0,
            counters={"a": 9.0, "b": 0.0, "c": 1.0},
        )
        self.assertEqual(selected["tenant"], "a")
        self.assertEqual(decision["mode"], "soft_aged_tenant_max_min")

    def test_soft_tier_remains_max_min_among_aged_tenants(self) -> None:
        jobs = [
            {"tenant": "a", "arrival_s": 0.0, "sequence": 0},
            {"tenant": "b", "arrival_s": 10.0, "sequence": 1},
        ]
        selected, _ = self.choose(
            jobs, now=100.0, soft=75.0, hard=150.0,
            counters={"a": 9.0, "b": 2.0},
        )
        self.assertEqual(selected["tenant"], "b")

    def test_hard_threshold_selects_globally_oldest_request(self) -> None:
        jobs = [
            {"tenant": "a", "arrival_s": 0.0, "sequence": 0},
            {"tenant": "b", "arrival_s": 20.0, "sequence": 1},
        ]
        selected, decision = self.choose(
            jobs, now=200.0, counters={"a": 100.0, "b": 0.0},
        )
        self.assertEqual(selected["tenant"], "a")
        self.assertEqual(decision["mode"], "hard_oldest_request")
        self.assertEqual(decision["request_age_s"], 200.0)

    def test_threshold_validation(self) -> None:
        job = [{"tenant": "a", "arrival_s": 0.0, "sequence": 0}]
        with self.assertRaises(ValueError):
            self.choose(job, soft=0.0)
        with self.assertRaises(ValueError):
            self.choose(job, soft=100.0, hard=99.0)

    def test_policy_is_fair_at_both_stages(self) -> None:
        policy = "age_aware_max_min"
        self.assertIn(policy, self.runner.PREPARATION_FAIR_POLICIES)
        self.assertIn(policy, self.runner.INFERENCE_FAIR_POLICIES)
        self.assertIn(policy, self.runner.SCHEDULER_OWNED_POLICIES)


if __name__ == "__main__":
    unittest.main()
