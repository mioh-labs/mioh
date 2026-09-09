import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from lada.models.basicvsrpp.mmagic import flow_warp as flow_warp_module
from lada.utils import mps_utils
from lada.utils import mps_grid_sample_backward


ROOT = Path(__file__).resolve().parents[1]
METAL_SOURCE = ROOT / "lada" / "utils" / "csrc" / "mps_grid_sample_backward.mm"
PROJECT_METADATA = ROOT / "pyproject.toml"


class MPSGridSampleNativeTests(unittest.TestCase):
    def test_native_backward_registers_the_missing_aten_mps_dispatch(self):
        source = METAL_SOURCE.read_text()
        metadata = PROJECT_METADATA.read_text()

        self.assertIn("TORCH_LIBRARY_IMPL(aten, MPS, m)", source)
        self.assertIn('"grid_sampler_2d_backward"', source)
        self.assertIn("at::mps::getCurrentMPSStream()", source)
        self.assertIn("device atomic_float* grad_input", source)
        self.assertIn("INTERPOLATION_BICUBIC", source)
        self.assertIn("PADDING_REFLECTION", source)
        self.assertIn("input.to(at::kFloat)", source)
        self.assertIn("'csrc/*.mm'", metadata)

    def test_safe_mps_grid_sample_loads_backward_only_for_autograd(self):
        fake_input = types.SimpleNamespace(
            device=types.SimpleNamespace(type="mps"), requires_grad=True
        )
        fake_grid = types.SimpleNamespace(requires_grad=False)

        with mock.patch.object(
            mps_grid_sample_backward,
            "enable_native_mps_grid_sample_backward",
            return_value=True,
        ) as enable_backward:
            with mock.patch.object(
                mps_utils.F, "grid_sample", return_value="native"
            ):
                result = mps_utils.safe_mps_grid_sample(fake_input, fake_grid)

        self.assertEqual(result, "native")
        enable_backward.assert_called_once_with(raise_on_error=True)

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

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required")
    def test_native_backward_matches_cpu_for_every_2d_mode(self):
        self.assertTrue(
            mps_grid_sample_backward.enable_native_mps_grid_sample_backward(
                raise_on_error=True
            )
        )
        torch.manual_seed(91)
        modes = ("bilinear", "nearest", "bicubic")
        padding_modes = ("zeros", "border", "reflection")

        for mode in modes:
            for padding_mode in padding_modes:
                for align_corners in (False, True):
                    with self.subTest(
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    ):
                        input_cpu = torch.randn(2, 3, 4, 5, requires_grad=True)
                        # Include out-of-bounds samples while avoiding exact
                        # interpolation boundaries where derivatives are not
                        # uniquely defined.
                        grid_cpu = (
                            torch.empty(2, 3, 4, 2).uniform_(-2.7, 2.7) + 0.013
                        ).requires_grad_(True)
                        output_gradient = torch.randn(2, 3, 3, 4)
                        output_cpu = torch.nn.functional.grid_sample(
                            input_cpu,
                            grid_cpu,
                            mode=mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                        )
                        output_cpu.backward(output_gradient)

                        input_mps = (
                            input_cpu.detach().to("mps").requires_grad_(True)
                        )
                        grid_mps = grid_cpu.detach().to("mps").requires_grad_(True)
                        output_mps = mps_utils.safe_mps_grid_sample(
                            input_mps,
                            grid_mps,
                            mode=mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                        )
                        output_mps.backward(output_gradient.to("mps"))
                        torch.mps.synchronize()

                        torch.testing.assert_close(
                            input_mps.grad.cpu(),
                            input_cpu.grad,
                            rtol=2e-5,
                            atol=2e-5,
                        )
                        torch.testing.assert_close(
                            grid_mps.grad.cpu(),
                            grid_cpu.grad,
                            rtol=2e-5,
                            atol=2e-5,
                        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required")
    def test_native_backward_matches_cpu_for_reduced_precision(self):
        self.assertTrue(
            mps_grid_sample_backward.enable_native_mps_grid_sample_backward(
                raise_on_error=True
            )
        )
        torch.manual_seed(1408)

        for dtype, tolerance in (
            (torch.float16, 2e-3),
            (torch.bfloat16, 2e-2),
        ):
            with self.subTest(dtype=dtype):
                source_input = torch.randn(1, 4, 13, 17)
                source_grid = (
                    torch.empty(1, 11, 15, 2).uniform_(-1.7, 1.7) + 0.017
                )
                source_gradient = torch.randn(1, 4, 11, 15)

                # Quantize the CPU reference first so both devices receive
                # identical half/bfloat16 values.
                input_cpu = source_input.to(dtype).float().requires_grad_(True)
                grid_cpu = source_grid.to(dtype).float().requires_grad_(True)
                output_gradient_cpu = source_gradient.to(dtype).float()
                output_cpu = torch.nn.functional.grid_sample(
                    input_cpu,
                    grid_cpu,
                    mode="bilinear",
                    padding_mode="reflection",
                    align_corners=False,
                )
                output_cpu.backward(output_gradient_cpu)

                input_mps = source_input.to("mps", dtype=dtype).requires_grad_(
                    True
                )
                grid_mps = source_grid.to("mps", dtype=dtype).requires_grad_(
                    True
                )
                output_mps = mps_utils.safe_mps_grid_sample(
                    input_mps,
                    grid_mps,
                    mode="bilinear",
                    padding_mode="reflection",
                    align_corners=False,
                )
                output_mps.backward(source_gradient.to("mps", dtype=dtype))
                torch.mps.synchronize()

                torch.testing.assert_close(
                    input_mps.grad.float().cpu(),
                    input_cpu.grad.to(dtype).float(),
                    rtol=tolerance,
                    atol=tolerance,
                )
                torch.testing.assert_close(
                    grid_mps.grad.float().cpu(),
                    grid_cpu.grad.to(dtype).float(),
                    rtol=tolerance,
                    atol=tolerance,
                )


if __name__ == "__main__":
    unittest.main()
