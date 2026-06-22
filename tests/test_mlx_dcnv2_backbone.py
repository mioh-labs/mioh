import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from experiments.mlx_dcnv2.backbone import residual_blocks_with_input_conv_forward


class MLXBackboneTests(unittest.TestCase):
    def test_residual_blocks_with_input_conv_matches_pytorch(self):
        rng = np.random.default_rng(123)
        in_channels = 5
        mid_channels = 4
        num_blocks = 3
        x = rng.normal(size=(1, in_channels, 6, 7)).astype(np.float32)
        tensors = _random_backbone_tensors(rng, in_channels, mid_channels, num_blocks)

        expected = _torch_backbone_forward(
            torch.from_numpy(x),
            {name: torch.from_numpy(value) for name, value in tensors.items()},
            num_blocks,
        ).numpy()
        actual = np.array(
            residual_blocks_with_input_conv_forward(
                mx.array(x),
                {name: mx.array(value) for name, value in tensors.items()},
                num_blocks=num_blocks,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)


def _torch_backbone_forward(x, tensors, num_blocks):
    out = F.conv2d(x, tensors["main.0.weight"], tensors["main.0.bias"], padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    for block_index in range(num_blocks):
        identity = out
        out = F.conv2d(
            out,
            tensors[f"main.2.{block_index}.conv1.weight"],
            tensors[f"main.2.{block_index}.conv1.bias"],
            padding=1,
        )
        out = F.relu(out)
        out = F.conv2d(
            out,
            tensors[f"main.2.{block_index}.conv2.weight"],
            tensors[f"main.2.{block_index}.conv2.bias"],
            padding=1,
        )
        out = identity + out
    return out


def _random_backbone_tensors(rng, in_channels, mid_channels, num_blocks):
    tensors = {
        "main.0.weight": (rng.normal(size=(mid_channels, in_channels, 3, 3)) * 0.02).astype(np.float32),
        "main.0.bias": (rng.normal(size=(mid_channels,)) * 0.02).astype(np.float32),
    }
    for block_index in range(num_blocks):
        tensors[f"main.2.{block_index}.conv1.weight"] = (
            rng.normal(size=(mid_channels, mid_channels, 3, 3)) * 0.02
        ).astype(np.float32)
        tensors[f"main.2.{block_index}.conv1.bias"] = (rng.normal(size=(mid_channels,)) * 0.02).astype(np.float32)
        tensors[f"main.2.{block_index}.conv2.weight"] = (
            rng.normal(size=(mid_channels, mid_channels, 3, 3)) * 0.02
        ).astype(np.float32)
        tensors[f"main.2.{block_index}.conv2.bias"] = (rng.normal(size=(mid_channels,)) * 0.02).astype(np.float32)
    return tensors


if __name__ == "__main__":
    unittest.main()
