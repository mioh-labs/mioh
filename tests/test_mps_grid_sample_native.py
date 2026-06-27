import types
import unittest
from unittest import mock

import torch

from lada.models.basicvsrpp.mmagic import flow_warp as flow_warp_module
from lada.utils import mps_utils


class MPSGridSampleNativeTests(unittest.TestCase):
    def test_safe_mps_grid_sample_preserves_border_padding_mode(self):
        fake_input = types.SimpleNamespace(device=types.SimpleNamespace(type="mps"))
        fake_grid = object()

        with mock.patch.object(mps_utils.F, "grid_sample", return_value="native") as grid_sample:
            result = mps_utils.safe_mps_grid_sample(
                fake_input,
                fake_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

        self.assertEqual(result, "native")
        grid_sample.assert_called_once_with(
            fake_input,
            fake_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def test_safe_mps_grid_sample_does_not_fallback_to_cpu_on_mps_error(self):
        fake_input = mock.Mock()
        fake_input.device = types.SimpleNamespace(type="mps")
        fake_grid = mock.Mock()

        with mock.patch.object(mps_utils.F, "grid_sample", side_effect=RuntimeError("mps missing")):
            with mock.patch.object(mps_utils.logger, "warning"):
                with self.assertRaisesRegex(RuntimeError, "mps missing"):
                    mps_utils.safe_mps_grid_sample(fake_input, fake_grid)

        fake_input.cpu.assert_not_called()
        fake_grid.cpu.assert_not_called()

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required")
    def test_flow_warp_border_uses_safe_grid_sample_on_mps(self):
        x = torch.randn(1, 2, 4, 5, device="mps")
        flow = torch.randn(1, 4, 5, 2, device="mps") * 0.1
        sentinel = torch.empty_like(x)

        with mock.patch.object(flow_warp_module, "safe_mps_grid_sample", return_value=sentinel) as safe_grid:
            result = flow_warp_module.flow_warp(x, flow, padding_mode="border")

        self.assertIs(result, sentinel)
        self.assertEqual(safe_grid.call_args.kwargs["padding_mode"], "border")


if __name__ == "__main__":
    unittest.main()
