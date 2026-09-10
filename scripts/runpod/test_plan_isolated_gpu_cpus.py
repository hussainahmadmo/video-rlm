from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("plan_isolated_gpu_cpus.py")


def load_planner():
    spec = importlib.util.spec_from_file_location("cpu_planner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CpuPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = load_planner()

    def test_cpu_list_round_trip(self) -> None:
        cpus = self.planner.expand_cpu_list("0-3,8,10-11")
        self.assertEqual(cpus, {0, 1, 2, 3, 8, 10, 11})
        self.assertEqual(self.planner.format_cpu_list(cpus), "0-3,8,10-11")

    def test_allocation_keeps_smt_siblings_together(self) -> None:
        cores = []
        for node in (0, 1):
            for core in range(node * 16, node * 16 + 16):
                cores.append(
                    self.planner.Core(
                        socket=node,
                        core=core,
                        node=node,
                        cpus=(core, core + 32),
                    )
                )
        plans = self.planner.allocate(cores, [0, 0, 1, 1], 8, 8)
        self.assertEqual(len(plans), 4)
        used: set[int] = set()
        for _, prep_text, engine_text, _ in plans:
            prep = self.planner.expand_cpu_list(prep_text)
            engine = self.planner.expand_cpu_list(engine_text)
            self.assertEqual(len(prep), 8)
            self.assertEqual(len(engine), 8)
            self.assertFalse(prep & engine)
            self.assertFalse(used & (prep | engine))
            used |= prep | engine
            for cpu in prep:
                sibling = cpu + 32 if cpu < 32 else cpu - 32
                self.assertIn(sibling, prep)
            for cpu in engine:
                sibling = cpu + 32 if cpu < 32 else cpu - 32
                self.assertIn(sibling, engine)

    def test_insufficient_cpu_capacity_fails(self) -> None:
        cores = [
            self.planner.Core(0, index, 0, (index, index + 8))
            for index in range(8)
        ]
        with self.assertRaises(RuntimeError):
            self.planner.allocate(cores, [0, 0], 8, 8)


if __name__ == "__main__":
    unittest.main()
