from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_fairness_comparison.py")


def load_script():
    spec = importlib.util.spec_from_file_location("fairness_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FairnessComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.script = load_script()

    def test_default_selects_all_six_paper_policies(self) -> None:
        self.assertEqual(
            self.script.selected_policies(None), list(self.script.POLICIES)
        )

    def test_repeated_methods_preserve_order_and_remove_duplicates(self) -> None:
        self.assertEqual(
            self.script.selected_policies(["cross_stage", "fcfs", "cross_stage"]),
            ["cross_stage", "fcfs"],
        )

    def test_command_assigns_a_separate_output_to_each_policy(self) -> None:
        command = self.script.build_command(
            python="python",
            method="cross_stage",
            trace=Path("trace.jsonl"),
            output_root=Path("results"),
            runner_args=["--ports", "9000", "9001"],
        )
        self.assertIn("--prep-policy", command)
        self.assertEqual(command[command.index("--prep-policy") + 1], "cross_stage")
        self.assertEqual(command[command.index("--output") + 1], "results/cross_stage")
        self.assertEqual(command[-3:], ["--ports", "9000", "9001"])

    def test_wrapper_owned_arguments_cannot_be_overridden(self) -> None:
        with self.assertRaises(ValueError):
            self.script.build_command(
                python="python",
                method="fcfs",
                trace=Path("trace.jsonl"),
                output_root=Path("results"),
                runner_args=["--prep-policy", "cross_stage"],
            )


if __name__ == "__main__":
    unittest.main()
