# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import unittest

import torch

from lada.models.mioh_restorer.training_memory import (
    GIB,
    MemorySnapshot,
    MemoryThresholds,
    TrainingMemoryGuard,
    tree_to_cpu,
)


class MiohTrainingMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = TrainingMemoryGuard(
            torch.device("cpu"),
            MemoryThresholds(
                warning_mps_ratio=0.80,
                critical_mps_ratio=0.92,
                warning_system_available_gib=8.0,
                critical_system_available_gib=4.0,
            ),
        )

    @staticmethod
    def snapshot(
        *,
        mps_ratio: float = 0.5,
        mps_current_ratio: float | None = None,
        system_available_gib: float = 16.0,
    ) -> MemorySnapshot:
        recommended = 40 * GIB
        current_ratio = mps_ratio if mps_current_ratio is None else mps_current_ratio
        return MemorySnapshot(
            stage="test",
            mps_current_bytes=int(recommended * current_ratio),
            mps_driver_bytes=int(recommended * mps_ratio),
            mps_recommended_bytes=recommended,
            system_available_bytes=int(system_available_gib * GIB),
            system_total_bytes=48 * GIB,
            swap_used_bytes=0,
        )

    def test_guard_distinguishes_normal_warning_and_critical_pressure(self):
        self.assertEqual(self.guard.status(self.snapshot()), "normal")
        self.assertEqual(
            self.guard.status(self.snapshot(mps_ratio=0.85)), "warning"
        )
        self.assertEqual(
            self.guard.status(self.snapshot(mps_ratio=0.95)), "critical"
        )
        self.assertEqual(
            self.guard.status(self.snapshot(system_available_gib=6.0)), "warning"
        )
        self.assertEqual(
            self.guard.status(self.snapshot(system_available_gib=3.0)), "critical"
        )

    def test_reclaimable_mps_cache_avoids_repeated_warning_cleanup(self):
        snapshot = self.snapshot(
            mps_ratio=0.50,
            mps_current_ratio=0.01,
            system_available_gib=6.0,
        )

        self.assertGreater(snapshot.effective_system_available_gib, 8.0)
        self.assertEqual(self.guard.status(snapshot), "normal")

    def test_tree_to_cpu_detaches_all_nested_tensors(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        source = {
            "state": [parameter, (torch.zeros(1, requires_grad=True),)],
            "step": 12,
        }

        copied = tree_to_cpu(source)

        self.assertEqual(copied["step"], 12)
        for tensor in (copied["state"][0], copied["state"][1][0]):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)

    def test_cpu_capture_includes_system_memory_without_mps_values(self):
        snapshot = self.guard.capture("cpu")

        self.assertGreater(snapshot.system_total_bytes, 0)
        self.assertGreater(snapshot.system_available_bytes, 0)
        self.assertIsNone(snapshot.mps_current_bytes)
        self.assertIsNone(snapshot.mps_pressure_ratio)

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(ValueError):
            MemoryThresholds(warning_mps_ratio=0.95, critical_mps_ratio=0.90)
        with self.assertRaises(ValueError):
            MemoryThresholds(
                warning_system_available_gib=4.0,
                critical_system_available_gib=8.0,
            )


if __name__ == "__main__":
    unittest.main()
