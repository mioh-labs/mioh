import io
import os
import unittest
from contextlib import redirect_stderr
from unittest import mock

import numpy as np
import torch

from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import (
    _BasicVSRPPProfiler,
    _mlx_propagation_warp_bridge_enabled,
    _mlx_second_order_cond_bridge,
)
from lada.models.basicvsrpp.mmagic.flow_warp import flow_warp


class BasicVSRPPProfilerTests(unittest.TestCase):
    def test_disabled_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            profiler = _BasicVSRPPProfiler(torch.zeros(1))

        self.assertFalse(profiler.enabled)

    def test_emits_accumulated_timing_when_enabled(self):
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {"LADA_BASICVSRPP_PROFILE": "1"}):
            profiler = _BasicVSRPPProfiler(torch.zeros(1))
            with mock.patch("time.perf_counter", side_effect=[10.0, 10.25]):
                with profiler.time("feat_extract"):
                    pass

            with redirect_stderr(stderr):
                profiler.emit({"frames": 20, "height": 64, "width": 64})

        line = stderr.getvalue()
        self.assertIn("[BASICVSRPP_PROFILE]", line)
        self.assertIn("frames=20", line)
        self.assertIn("feat_extract=0.250", line)


class BasicVSRPPMLXWarpBridgeTests(unittest.TestCase):
    def test_mlx_propagation_warp_bridge_can_be_disabled(self):
        with mock.patch.dict(os.environ, {"LADA_BASICVSRPP_MLX_PROPAGATION_WARP": "0"}):
            self.assertFalse(_mlx_propagation_warp_bridge_enabled())

    def test_mlx_second_order_cond_bridge_matches_torch_warps(self):
        rng = np.random.default_rng(2027)
        channels = 3
        height = 5
        width = 6
        feat_current = torch.from_numpy(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        feat_prop = torch.from_numpy(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        feat_n2 = torch.from_numpy(rng.normal(size=(1, channels, height, width)).astype(np.float32))
        flow_n1 = torch.from_numpy((rng.normal(size=(1, 2, height, width)) * 0.2).astype(np.float32))
        flow_n2 = torch.from_numpy((rng.normal(size=(1, 2, height, width)) * 0.2).astype(np.float32))

        cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
        flow_total = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
        cond_n2 = flow_warp(feat_n2, flow_total.permute(0, 2, 3, 1))
        expected_cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)

        actual_cond, actual_flow = _mlx_second_order_cond_bridge(
            feat_current,
            feat_prop,
            feat_n2,
            flow_n1,
            flow_n2,
        )

        torch.testing.assert_close(actual_cond, expected_cond, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_flow, flow_total, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
