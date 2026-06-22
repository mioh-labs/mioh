import unittest

import mlx.core as mx
import numpy as np
import torch
import torchvision

from experiments.mlx_dcnv2 import deform_conv2d_forward


class MLXDCNv2ForwardTests(unittest.TestCase):
    def test_1x1_zero_offset_matches_torchvision_deform_conv2d(self):
        rng = np.random.default_rng(123)
        x_np = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
        weight_np = rng.normal(size=(3, 2, 1, 1)).astype(np.float32)
        bias_np = rng.normal(size=(3,)).astype(np.float32)
        offset_np = np.zeros((1, 2, 3, 4), dtype=np.float32)
        mask_np = np.ones((1, 1, 3, 4), dtype=np.float32)

        expected = torchvision.ops.deform_conv2d(
            torch.from_numpy(x_np),
            torch.from_numpy(offset_np),
            torch.from_numpy(weight_np),
            torch.from_numpy(bias_np),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            mask=torch.from_numpy(mask_np),
        ).numpy()

        actual = np.array(
            deform_conv2d_forward(
                mx.array(x_np),
                mx.array(offset_np),
                mx.array(weight_np),
                mx.array(bias_np),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                mask=mx.array(mask_np),
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_3x3_fractional_offset_matches_torchvision_deform_conv2d(self):
        rng = np.random.default_rng(456)
        x_np = rng.normal(size=(1, 2, 5, 6)).astype(np.float32)
        weight_np = rng.normal(size=(3, 2, 3, 3)).astype(np.float32)
        bias_np = rng.normal(size=(3,)).astype(np.float32)
        offset_np = rng.uniform(-0.35, 0.35, size=(1, 18, 5, 6)).astype(np.float32)
        mask_np = rng.uniform(0.25, 1.0, size=(1, 9, 5, 6)).astype(np.float32)

        expected = torchvision.ops.deform_conv2d(
            torch.from_numpy(x_np),
            torch.from_numpy(offset_np),
            torch.from_numpy(weight_np),
            torch.from_numpy(bias_np),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            mask=torch.from_numpy(mask_np),
        ).numpy()

        actual = np.array(
            deform_conv2d_forward(
                mx.array(x_np),
                mx.array(offset_np),
                mx.array(weight_np),
                mx.array(bias_np),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                mask=mx.array(mask_np),
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_grouped_deform_groups_without_bias_matches_torchvision_deform_conv2d(self):
        rng = np.random.default_rng(789)
        x_np = rng.normal(size=(1, 4, 4, 5)).astype(np.float32)
        weight_np = rng.normal(size=(4, 2, 3, 3)).astype(np.float32)
        offset_np = rng.uniform(-0.25, 0.25, size=(1, 36, 4, 5)).astype(np.float32)
        mask_np = rng.uniform(0.2, 1.0, size=(1, 18, 4, 5)).astype(np.float32)

        expected = torchvision.ops.deform_conv2d(
            torch.from_numpy(x_np),
            torch.from_numpy(offset_np),
            torch.from_numpy(weight_np),
            None,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            mask=torch.from_numpy(mask_np),
        ).numpy()

        actual = np.array(
            deform_conv2d_forward(
                mx.array(x_np),
                mx.array(offset_np),
                mx.array(weight_np),
                None,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                mask=mx.array(mask_np),
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
