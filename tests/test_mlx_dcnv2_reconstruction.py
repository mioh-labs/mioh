import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from experiments.mlx_dcnv2.reconstruction import reconstruction_forward


class MLXReconstructionTests(unittest.TestCase):
    def test_reconstruction_matches_pytorch(self):
        rng = np.random.default_rng(1001)
        mid_channels = 4
        num_blocks = 2
        x = rng.normal(size=(1, 5 * mid_channels, 5, 6)).astype(np.float32)
        lq = rng.normal(size=(1, 3, 20, 24)).astype(np.float32)
        tensors = _random_reconstruction_tensors(rng, mid_channels, num_blocks)

        expected = _torch_reconstruction_forward(
            torch.from_numpy(x),
            torch.from_numpy(lq),
            {name: torch.from_numpy(value) for name, value in tensors.items()},
            num_blocks,
        ).numpy()
        actual = np.array(
            reconstruction_forward(
                mx.array(x),
                mx.array(lq),
                {name: mx.array(value) for name, value in tensors.items()},
                num_blocks=num_blocks,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)


def _torch_reconstruction_forward(x, lq, tensors, num_blocks):
    out = F.conv2d(x, tensors["reconstruction.main.0.weight"], tensors["reconstruction.main.0.bias"], padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    for block_index in range(num_blocks):
        identity = out
        out = F.conv2d(
            out,
            tensors[f"reconstruction.main.2.{block_index}.conv1.weight"],
            tensors[f"reconstruction.main.2.{block_index}.conv1.bias"],
            padding=1,
        )
        out = F.relu(out)
        out = F.conv2d(
            out,
            tensors[f"reconstruction.main.2.{block_index}.conv2.weight"],
            tensors[f"reconstruction.main.2.{block_index}.conv2.bias"],
            padding=1,
        )
        out = identity + out
    out = F.conv2d(out, tensors["upsample1.upsample_conv.weight"], tensors["upsample1.upsample_conv.bias"], padding=1)
    out = F.pixel_shuffle(out, 2)
    out = F.leaky_relu(out, negative_slope=0.1)
    out = F.conv2d(out, tensors["upsample2.upsample_conv.weight"], tensors["upsample2.upsample_conv.bias"], padding=1)
    out = F.pixel_shuffle(out, 2)
    out = F.leaky_relu(out, negative_slope=0.1)
    out = F.conv2d(out, tensors["conv_hr.weight"], tensors["conv_hr.bias"], padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    out = F.conv2d(out, tensors["conv_last.weight"], tensors["conv_last.bias"], padding=1)
    return out + lq


def _random_reconstruction_tensors(rng, mid_channels, num_blocks):
    scale = 0.02
    tensors = {
        "reconstruction.main.0.weight": (rng.normal(size=(mid_channels, 5 * mid_channels, 3, 3)) * scale).astype(np.float32),
        "reconstruction.main.0.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
    }
    for block_index in range(num_blocks):
        tensors[f"reconstruction.main.2.{block_index}.conv1.weight"] = (
            rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale
        ).astype(np.float32)
        tensors[f"reconstruction.main.2.{block_index}.conv1.bias"] = (rng.normal(size=(mid_channels,)) * scale).astype(np.float32)
        tensors[f"reconstruction.main.2.{block_index}.conv2.weight"] = (
            rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale
        ).astype(np.float32)
        tensors[f"reconstruction.main.2.{block_index}.conv2.bias"] = (rng.normal(size=(mid_channels,)) * scale).astype(np.float32)
    tensors.update(
        {
            "upsample1.upsample_conv.weight": (rng.normal(size=(4 * mid_channels, mid_channels, 3, 3)) * scale).astype(np.float32),
            "upsample1.upsample_conv.bias": (rng.normal(size=(4 * mid_channels,)) * scale).astype(np.float32),
            "upsample2.upsample_conv.weight": (rng.normal(size=(4 * 64, mid_channels, 3, 3)) * scale).astype(np.float32),
            "upsample2.upsample_conv.bias": (rng.normal(size=(4 * 64,)) * scale).astype(np.float32),
            "conv_hr.weight": (rng.normal(size=(64, 64, 3, 3)) * scale).astype(np.float32),
            "conv_hr.bias": (rng.normal(size=(64,)) * scale).astype(np.float32),
            "conv_last.weight": (rng.normal(size=(3, 64, 3, 3)) * scale).astype(np.float32),
            "conv_last.bias": (rng.normal(size=(3,)) * scale).astype(np.float32),
        }
    )
    return tensors


if __name__ == "__main__":
    unittest.main()
