import unittest

import mlx.core as mx
import numpy as np

from experiments.mlx_dcnv2.flow_warp import flow_warp
from experiments.mlx_dcnv2.fused_propagation_warp import (
    fused_propagation_warp_cond,
    two_stage_propagation_warp_cond,
)


class MLXFusedPropagationWarpTests(unittest.TestCase):
    def test_fused_cond_matches_separate_mlx_warps(self):
        rng = np.random.default_rng(1401)
        channels = 3
        height = 5
        width = 6
        feat_current = mx.array(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        feat_prop = mx.array(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        feat_n2 = mx.array(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        flow_n1 = mx.array((rng.normal(size=(1, 2, height, width)) * 0.2).astype(np.float32))
        flow_n2 = mx.array((rng.normal(size=(1, 2, height, width)) * 0.2).astype(np.float32))

        flow_n1_grid = mx.transpose(flow_n1, (0, 2, 3, 1))
        cond_n1 = flow_warp(feat_prop, flow_n1_grid, padding_mode="zeros")
        flow_n2_total = flow_n1 + flow_warp(flow_n2, flow_n1_grid, padding_mode="zeros")
        cond_n2 = flow_warp(feat_n2, mx.transpose(flow_n2_total, (0, 2, 3, 1)), padding_mode="zeros")
        expected = mx.concatenate([cond_n1, feat_current, cond_n2], axis=1)

        actual = fused_propagation_warp_cond(feat_current, feat_prop, feat_n2, flow_n1, flow_n2)

        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-4, atol=1e-4)

    def test_two_stage_cond_matches_separate_mlx_warps(self):
        rng = np.random.default_rng(1402)
        channels = 3
        height = 5
        width = 6
        feat_current = mx.array(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        feat_prop = mx.array(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        feat_n2 = mx.array(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        flow_n1 = mx.array((rng.normal(size=(1, 2, height, width)) * 0.2).astype(np.float32))
        flow_n2 = mx.array((rng.normal(size=(1, 2, height, width)) * 0.2).astype(np.float32))

        flow_n1_grid = mx.transpose(flow_n1, (0, 2, 3, 1))
        cond_n1 = flow_warp(feat_prop, flow_n1_grid, padding_mode="zeros")
        flow_n2_total = flow_n1 + flow_warp(flow_n2, flow_n1_grid, padding_mode="zeros")
        cond_n2 = flow_warp(feat_n2, mx.transpose(flow_n2_total, (0, 2, 3, 1)), padding_mode="zeros")
        expected = mx.concatenate([cond_n1, feat_current, cond_n2], axis=1)

        actual = two_stage_propagation_warp_cond(feat_current, feat_prop, feat_n2, flow_n1, flow_n2)

        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
