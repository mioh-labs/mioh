import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from experiments.mlx_dcnv2.flow_warp import flow_warp


class MLXFlowWarpTests(unittest.TestCase):
    def test_flow_warp_zeros_matches_pytorch_grid_sample(self):
        rng = np.random.default_rng(321)
        x = rng.normal(size=(1, 3, 5, 6)).astype(np.float32)
        flow = rng.normal(size=(1, 5, 6, 2)).astype(np.float32) * 0.4

        expected = _torch_flow_warp(x, flow, padding_mode="zeros")
        actual = np.array(flow_warp(mx.array(x), mx.array(flow), padding_mode="zeros"))

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_flow_warp_border_matches_pytorch_grid_sample(self):
        rng = np.random.default_rng(322)
        x = rng.normal(size=(1, 2, 4, 5)).astype(np.float32)
        flow = rng.normal(size=(1, 4, 5, 2)).astype(np.float32) * 1.2

        expected = _torch_flow_warp(x, flow, padding_mode="border")
        actual = np.array(flow_warp(mx.array(x), mx.array(flow), padding_mode="border"))

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def _torch_flow_warp(x, flow, padding_mode):
    x_t = torch.from_numpy(x)
    flow_t = torch.from_numpy(flow)
    _, _, h, w = x_t.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x_t.dtype),
        torch.arange(0, w, dtype=x_t.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid_flow = grid + flow_t
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(x_t, grid_flow, mode="bilinear", padding_mode=padding_mode, align_corners=True).numpy()


if __name__ == "__main__":
    unittest.main()
