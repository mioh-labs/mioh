"""Numerical checks for the controlled multi-frame mosaic experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from lada.models.basicvsrpp.recoverable_hf_dataset import phase_block_average_mosaic


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/training/evaluate-multiframe-mosaic-oracle.py"
)
SPEC = importlib.util.spec_from_file_location("mosaic_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class MultiframeMosaicOracleTests(unittest.TestCase):
    def test_forward_matches_training_mosaic_up_to_uint8_rounding(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        for block in (4, 8):
            for channel in range(3):
                observation = ORACLE.expanded_measurement(
                    ORACLE.block_mean(image[:, :, channel].astype(np.float64), block),
                    block,
                )
                training = phase_block_average_mosaic(
                    image, block_size=block, phase=(0, 0)
                )[:, :, channel]
                self.assertLessEqual(np.max(np.abs(observation - training)), 0.5)

    def test_objective_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(4)
        clean = rng.random((8, 8))
        candidate = rng.random((8, 8)) * 0.5 + 0.25
        shifts = [(0, 0), (1, 2), (3, 1)]
        measurements = ORACLE.observe(clean, 4, shifts)
        args = (clean.shape, 4, shifts, measurements, 0.003)
        _, gradient = ORACLE.objective_and_gradient(candidate.ravel(), *args)
        epsilon = 1e-6
        for index in (0, 7, 13, 45, 63):
            plus = candidate.ravel().copy()
            minus = candidate.ravel().copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            value_plus, _ = ORACLE.objective_and_gradient(plus, *args)
            value_minus, _ = ORACLE.objective_and_gradient(minus, *args)
            self.assertAlmostEqual(
                gradient[index], (value_plus - value_minus) / (2 * epsilon),
                delta=1e-5,
            )

    def test_static_observations_add_no_new_information(self):
        rng = np.random.default_rng(5)
        clean = rng.random((16, 16))
        shifts = ORACLE.shifts_for(4, 9, 17, "static")
        self.assertEqual(len(set(shifts)), 1)
        observations = ORACLE.observe(clean, 4, shifts)
        for observation in observations[1:]:
            np.testing.assert_array_equal(observations[0], observation)


if __name__ == "__main__":
    unittest.main()
