import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from experiments.mlx_dcnv2.alignment import (
    compute_second_order_offset_mask,
    second_order_deformable_alignment_forward,
)


class MLXSecondOrderAlignmentTests(unittest.TestCase):
    def test_second_order_alignment_forward_matches_pytorch(self):
        rng = np.random.default_rng(654)
        mid_channels = 4
        deform_groups = 2
        n, h, w = 1, 4, 5
        x_np = rng.normal(size=(n, 2 * mid_channels, h, w)).astype(np.float32)
        extra_np = rng.normal(size=(n, 3 * mid_channels, h, w)).astype(np.float32)
        flow1_np = rng.normal(size=(n, 2, h, w)).astype(np.float32)
        flow2_np = rng.normal(size=(n, 2, h, w)).astype(np.float32)
        tensors = _random_alignment_tensors(rng, mid_channels, deform_groups)

        offset, mask = _torch_offset_mask(extra_np, flow1_np, flow2_np, tensors)
        expected = torchvision.ops.deform_conv2d(
            torch.from_numpy(x_np),
            torch.from_numpy(offset),
            torch.from_numpy(tensors["weight"]),
            torch.from_numpy(tensors["bias"]),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            mask=torch.from_numpy(mask),
        ).numpy()

        actual = second_order_deformable_alignment_forward(
            mx.array(x_np),
            mx.array(extra_np),
            mx.array(flow1_np),
            mx.array(flow2_np),
            {name: mx.array(value) for name, value in tensors.items()},
        )

        np.testing.assert_allclose(np.array(actual), expected, rtol=1e-4, atol=5e-4)


def _torch_offset_mask(extra_np, flow1_np, flow2_np, tensors):
    extra = torch.from_numpy(extra_np)
    flow1 = torch.from_numpy(flow1_np)
    flow2 = torch.from_numpy(flow2_np)
    x = torch.cat([extra, flow1, flow2], dim=1)
    for layer in (0, 2, 4):
        x = F.conv2d(
            x,
            torch.from_numpy(tensors[f"conv_offset.{layer}.weight"]),
            torch.from_numpy(tensors[f"conv_offset.{layer}.bias"]),
            padding=1,
        )
        x = F.leaky_relu(x, negative_slope=0.1)
    x = F.conv2d(
        x,
        torch.from_numpy(tensors["conv_offset.6.weight"]),
        torch.from_numpy(tensors["conv_offset.6.bias"]),
        padding=1,
    )
    o1, o2, mask = torch.chunk(x, 3, dim=1)
    offset = 10 * torch.tanh(torch.cat((o1, o2), dim=1))
    offset1, offset2 = torch.chunk(offset, 2, dim=1)
    offset1 = offset1 + flow1.flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
    offset2 = offset2 + flow2.flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
    return torch.cat([offset1, offset2], dim=1).numpy(), torch.sigmoid(mask).numpy()


def _random_alignment_tensors(rng, mid_channels, deform_groups):
    return {
        "weight": rng.normal(size=(mid_channels, 2 * mid_channels, 3, 3)).astype(np.float32),
        "bias": rng.normal(size=(mid_channels,)).astype(np.float32),
        "conv_offset.0.weight": rng.normal(size=(mid_channels, 3 * mid_channels + 4, 3, 3)).astype(np.float32),
        "conv_offset.0.bias": rng.normal(size=(mid_channels,)).astype(np.float32),
        "conv_offset.2.weight": rng.normal(size=(mid_channels, mid_channels, 3, 3)).astype(np.float32),
        "conv_offset.2.bias": rng.normal(size=(mid_channels,)).astype(np.float32),
        "conv_offset.4.weight": rng.normal(size=(mid_channels, mid_channels, 3, 3)).astype(np.float32),
        "conv_offset.4.bias": rng.normal(size=(mid_channels,)).astype(np.float32),
        "conv_offset.6.weight": rng.normal(size=(27 * deform_groups, mid_channels, 3, 3)).astype(np.float32),
        "conv_offset.6.bias": rng.normal(size=(27 * deform_groups,)).astype(np.float32),
    }


if __name__ == "__main__":
    unittest.main()
