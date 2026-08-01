# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
import torchvision

from scripts.apple import basicvsrpp_coreai_kernels as kernels


class CoreAIKernelReferenceTests(unittest.TestCase):
    def test_deform_kernel_uses_threadgroup_sample_tile_and_tensorops_matmul(self):
        self.assertIn(
            "threadgroup TYPE sampled_values", kernels.DEFORM_CONV_METAL_SOURCE
        )
        self.assertIn("threadgroup_barrier", kernels.DEFORM_CONV_METAL_SOURCE)
        self.assertIn("matmul2d", kernels.DEFORM_CONV_METAL_SOURCE)
        self.assertIn("execution_simdgroups<8>", kernels.DEFORM_CONV_METAL_SOURCE)
        self.assertIn(
            "coordinate_count = tile_rows * deform_groups * kernel_size",
            kernels.DEFORM_CONV_METAL_SOURCE,
        )
        self.assertIn(
            "channel_in_group < channels_per_deform_group",
            kernels.DEFORM_CONV_METAL_SOURCE,
        )

    def test_deform_kernel_dispatches_eight_simdgroups_per_eight_output_rows(self):
        image = torch.zeros((1, 2, 4, 4), dtype=torch.float16)
        offset = torch.zeros((1, 18, 4, 4), dtype=torch.float16)
        weight = torch.zeros((3, 2, 3, 3), dtype=torch.float16)
        bias = torch.zeros(3, dtype=torch.float16)
        mask = torch.ones((1, 9, 4, 4), dtype=torch.float16)
        kernel = mock.Mock(return_value=torch.zeros((16, 3)))

        kernels.run_deform_conv_kernel(
            kernel,
            image,
            offset,
            weight,
            bias,
            mask,
        )

        self.assertEqual(kernel.call_args.kwargs["threads_per_grid"], (512, 1, 1))
        self.assertEqual(
            kernel.call_args.kwargs["threads_per_thread_group"], (256, 1, 1)
        )

    def test_grid_sample_dispatches_sixteen_by_sixteen_threadgroups(self):
        image = torch.zeros((1, 2, 4, 4), dtype=torch.float16)
        grid = torch.zeros((1, 3, 5, 2), dtype=torch.float16)
        kernel = mock.Mock(return_value=torch.zeros((1, 2, 3, 5)))

        kernels.run_grid_sample_kernel(kernel, image, grid)

        self.assertEqual(kernel.call_args.kwargs["threads_per_grid"], (5, 3, 2))
        self.assertEqual(
            kernel.call_args.kwargs["threads_per_thread_group"], (16, 16, 1)
        )

    def test_flow_warp_dispatches_sixteen_by_sixteen_threadgroups(self):
        image = torch.zeros((1, 2, 4, 5), dtype=torch.float16)
        flow = torch.zeros((1, 2, 4, 5), dtype=torch.float16)
        kernel = mock.Mock(return_value=torch.zeros((1, 2, 4, 5)))

        kernels.run_flow_warp_kernel(kernel, image, flow)

        self.assertEqual(kernel.call_args.kwargs["threads_per_grid"], (5, 4, 2))
        self.assertEqual(
            kernel.call_args.kwargs["threads_per_thread_group"], (16, 16, 1)
        )
        self.assertTrue(kernel.call_args.args[1].is_contiguous())

    def test_flow_warp_reference_keeps_coordinate_math_in_float32(self):
        image = torch.arange(40, dtype=torch.float16).reshape(1, 2, 4, 5) / 40
        flow_nhwc = torch.tensor(
            [
                [
                    [[0.0, 0.0], [0.5, 0.0], [0.0, -0.5], [1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.5], [-0.5, 0.0], [0.25, 0.25], [0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [-0.25, 0.5], [0.0, -1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
                ]
            ],
            dtype=torch.float16,
        )
        flow = flow_nhwc.permute(0, 3, 1, 2).contiguous()
        height, width = image.shape[-2:]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        grid_flow = (
            torch.stack((grid_x, grid_y), dim=-1)
            + flow.permute(0, 2, 3, 1).float()
        )
        grid = torch.stack(
            (
                2.0 * grid_flow[..., 0] / (width - 1) - 1.0,
                2.0 * grid_flow[..., 1] / (height - 1) - 1.0,
            ),
            dim=-1,
        )
        expected = F.grid_sample(
            image.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).half()

        actual = kernels.flow_warp_reference(image, flow, False)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_deform_kernel_rejects_grouped_convolution_weights(self):
        image = torch.zeros((1, 4, 4, 4), dtype=torch.float16)
        offset = torch.zeros((1, 18, 4, 4), dtype=torch.float16)
        weight = torch.zeros((4, 2, 3, 3), dtype=torch.float16)
        bias = torch.zeros(4, dtype=torch.float16)
        mask = torch.ones((1, 9, 4, 4), dtype=torch.float16)
        kernel = mock.Mock()

        with self.assertRaisesRegex(ValueError, "groups=1"):
            kernels.run_deform_conv_kernel(
                kernel,
                image,
                offset,
                weight,
                bias,
                mask,
            )

        kernel.assert_not_called()

    def test_deform_kernel_requires_channels_divisible_by_deform_groups(self):
        image = torch.zeros((1, 6, 4, 4), dtype=torch.float16)
        offset = torch.zeros((1, 4 * 2 * 9, 4, 4), dtype=torch.float16)
        weight = torch.zeros((3, 6, 3, 3), dtype=torch.float16)
        bias = torch.zeros(3, dtype=torch.float16)
        mask = torch.ones((1, 4 * 9, 4, 4), dtype=torch.float16)
        kernel = mock.Mock()

        with self.assertRaisesRegex(ValueError, "divisible by deform groups"):
            kernels.run_deform_conv_kernel(
                kernel,
                image,
                offset,
                weight,
                bias,
                mask,
            )

        kernel.assert_not_called()

    def test_deform_kernel_requires_matching_mask_channels(self):
        image = torch.zeros((1, 8, 4, 4), dtype=torch.float16)
        offset = torch.zeros((1, 4 * 2 * 9, 4, 4), dtype=torch.float16)
        weight = torch.zeros((3, 8, 3, 3), dtype=torch.float16)
        bias = torch.zeros(3, dtype=torch.float16)
        mask = torch.ones((1, 9, 4, 4), dtype=torch.float16)
        kernel = mock.Mock()

        with self.assertRaisesRegex(ValueError, "mask channels"):
            kernels.run_deform_conv_kernel(
                kernel,
                image,
                offset,
                weight,
                bias,
                mask,
            )

        kernel.assert_not_called()

    def test_reference_matches_bilinear_zeros_align_corners(self):
        image = torch.arange(12, dtype=torch.float16).reshape(1, 1, 3, 4)
        grid = torch.tensor(
            [[[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [1.5, 0.0]]]],
            dtype=torch.float16,
        )

        expected = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        actual = kernels.grid_sample_reference(image, grid, False)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


    def test_reference_matches_torchvision_deform_conv(self):
        image = torch.arange(32, dtype=torch.float16).reshape(1, 2, 4, 4) / 32
        offset = torch.zeros((1, 18, 4, 4), dtype=torch.float16)
        weight = torch.arange(54, dtype=torch.float16).reshape(3, 2, 3, 3) / 54
        bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float16)
        mask = torch.full((1, 9, 4, 4), 0.75, dtype=torch.float16)

        expected = torchvision.ops.deform_conv2d(
            image,
            offset,
            weight,
            bias,
            padding=(1, 1),
            mask=mask,
        )
        actual = kernels.deform_conv_reference(image, offset, weight, bias, mask)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        weight_matrix = weight.permute(1, 2, 3, 0).reshape(18, 3).contiguous()
        matrix = kernels.deform_conv_tensorops_reference(
            image,
            offset,
            weight_matrix,
            mask,
        )
        reconstructed = (
            matrix.add(bias).reshape(1, 4, 4, 3).permute(0, 3, 1, 2)
        )
        torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)

    def test_reference_matches_bilinear_border_align_corners(self):
        image = torch.arange(12, dtype=torch.float16).reshape(1, 1, 3, 4)
        grid = torch.tensor(
            [[[[1.5, 0.0], [-1.5, 0.5]]]],
            dtype=torch.float16,
        )

        expected = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        actual = kernels.grid_sample_reference(image, grid, True)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(
    importlib.util.find_spec("coreai_torch"),
    "coreai-torch is installed only in the isolated conversion environment",
)
class CoreAIMetalKernelTests(unittest.TestCase):
    def test_custom_kernel_exports_and_converts_to_coreai(self):
        import coreai_torch

        kernel = kernels.build_grid_sample_kernel(coreai_torch)

        class GridSampleModel(torch.nn.Module):
            def forward(self, image, grid):
                return kernels.run_grid_sample_kernel(kernel, image, grid)

        model = GridSampleModel().eval()
        image = torch.arange(12, dtype=torch.float16).reshape(1, 1, 3, 4)
        grid = torch.tensor(
            [[[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [1.5, 0.0]]]],
            dtype=torch.float16,
        )

        torch.testing.assert_close(
            model(image, grid),
            kernels.grid_sample_reference(image, grid, False),
            rtol=0,
            atol=0,
        )
        exported = torch.export.export(model, (image, grid))
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        targets = {
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        }
        self.assertIn(
            "coreai_metal_kernels.grid_sample_bilinear_align_corners.default",
            targets,
        )
        self.assertNotIn("aten.grid_sampler_2d.default", targets)

        converter = coreai_torch.TorchConverter()
        converter.register_custom_kernels([kernel])
        converter.add_exported_program(
            exported,
            input_names=["image", "grid"],
            output_names=["warped"],
        )
        program = converter.to_coreai()
        program.optimize()

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "grid-sample.aimodel"
            program.save_asset(output)
            self.assertTrue(output.exists())

    def test_flow_warp_kernel_exports_and_converts_to_coreai(self):
        import coreai_torch

        kernel = kernels.build_flow_warp_kernel(coreai_torch)

        class FlowWarpModel(torch.nn.Module):
            def forward(self, image, flow):
                return kernels.run_flow_warp_kernel(kernel, image, flow)

        model = FlowWarpModel().eval()
        image = torch.arange(40, dtype=torch.float16).reshape(1, 2, 4, 5) / 40
        flow = torch.zeros((1, 2, 4, 5), dtype=torch.float16)
        flow[:, :, 1:3, 1:4] = 0.5

        exported = torch.export.export(model, (image, flow))
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        targets = {
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        }
        self.assertIn(
            "coreai_metal_kernels.flow_warp_nchw_bilinear_align_corners.default",
            targets,
        )
        self.assertNotIn("aten.grid_sampler_2d.default", targets)

        converter = coreai_torch.TorchConverter()
        converter.register_custom_kernels([kernel])
        converter.add_exported_program(
            exported,
            input_names=["image", "flow"],
            output_names=["warped"],
        )
        program = converter.to_coreai()
        program.optimize()

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "flow-warp.aimodel"
            program.save_asset(output)
            self.assertTrue(output.exists())

    def test_deform_conv_kernel_exports_and_converts_to_coreai(self):
        import coreai_torch

        kernel = kernels.build_deform_conv_kernel(coreai_torch)

        class DeformConvModel(torch.nn.Module):
            def forward(self, image, offset, weight, bias, mask):
                return kernels.run_deform_conv_kernel(
                    kernel,
                    image,
                    offset,
                    weight,
                    bias,
                    mask,
                )

        model = DeformConvModel().eval()
        image = torch.arange(32, dtype=torch.float16).reshape(1, 2, 4, 4) / 32
        offset = torch.zeros((1, 18, 4, 4), dtype=torch.float16)
        weight = torch.arange(54, dtype=torch.float16).reshape(3, 2, 3, 3) / 54
        bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float16)
        mask = torch.full((1, 9, 4, 4), 0.75, dtype=torch.float16)

        exported = torch.export.export(
            model,
            (image, offset, weight, bias, mask),
        )
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        targets = {
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        }
        self.assertIn(
            "coreai_metal_kernels.modulated_deform_conv2d.default",
            targets,
        )
        self.assertNotIn("torchvision.deform_conv2d.default", targets)

        converter = coreai_torch.TorchConverter()
        converter.register_custom_kernels([kernel])
        converter.add_exported_program(
            exported,
            input_names=["image", "offset", "weight", "bias", "mask"],
            output_names=["aligned"],
        )
        program = converter.to_coreai()
        program.optimize()

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "deform-conv.aimodel"
            program.save_asset(output)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
