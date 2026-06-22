import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from experiments.mlx_dcnv2.feature_extract import feature_extract_forward


class MLXFeatureExtractTests(unittest.TestCase):
    def test_feature_extract_matches_pytorch(self):
        rng = np.random.default_rng(901)
        mid_channels = 4
        num_blocks = 2
        x = rng.normal(size=(1, 3, 20, 24)).astype(np.float32)
        tensors = _random_feature_extract_tensors(rng, mid_channels, num_blocks)

        expected = _torch_feature_extract_forward(
            torch.from_numpy(x),
            {name: torch.from_numpy(value) for name, value in tensors.items()},
            num_blocks,
        ).numpy()
        actual = np.array(
            feature_extract_forward(
                mx.array(x),
                {name: mx.array(value) for name, value in tensors.items()},
                num_blocks=num_blocks,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)


def _torch_feature_extract_forward(x, tensors, num_blocks):
    out = F.conv2d(x, tensors["0.weight"], tensors["0.bias"], stride=2, padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    out = F.conv2d(out, tensors["2.weight"], tensors["2.bias"], stride=2, padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    out = F.conv2d(out, tensors["4.main.0.weight"], tensors["4.main.0.bias"], padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    for block_index in range(num_blocks):
        identity = out
        out = F.conv2d(
            out,
            tensors[f"4.main.2.{block_index}.conv1.weight"],
            tensors[f"4.main.2.{block_index}.conv1.bias"],
            padding=1,
        )
        out = F.relu(out)
        out = F.conv2d(
            out,
            tensors[f"4.main.2.{block_index}.conv2.weight"],
            tensors[f"4.main.2.{block_index}.conv2.bias"],
            padding=1,
        )
        out = identity + out
    return out


def _random_feature_extract_tensors(rng, mid_channels, num_blocks):
    scale = 0.02
    tensors = {
        "0.weight": (rng.normal(size=(mid_channels, 3, 3, 3)) * scale).astype(np.float32),
        "0.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "2.weight": (rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale).astype(np.float32),
        "2.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "4.main.0.weight": (rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale).astype(np.float32),
        "4.main.0.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
    }
    for block_index in range(num_blocks):
        tensors[f"4.main.2.{block_index}.conv1.weight"] = (
            rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale
        ).astype(np.float32)
        tensors[f"4.main.2.{block_index}.conv1.bias"] = (rng.normal(size=(mid_channels,)) * scale).astype(np.float32)
        tensors[f"4.main.2.{block_index}.conv2.weight"] = (
            rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale
        ).astype(np.float32)
        tensors[f"4.main.2.{block_index}.conv2.bias"] = (rng.normal(size=(mid_channels,)) * scale).astype(np.float32)
    return tensors


if __name__ == "__main__":
    unittest.main()
