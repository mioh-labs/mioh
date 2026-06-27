import unittest


class MLXDCNv2CompareBenchTests(unittest.TestCase):
    def test_flow_warp_compare_returns_torch_and_mlx_timings(self):
        from experiments.mlx_dcnv2.bench_flow_warp_compare import run_flow_warp_case

        row = run_flow_warp_case(
            channels=2,
            height=5,
            width=6,
            padding_mode="zeros",
            torch_device="cpu",
            warmup=0,
            iters=1,
            seed=11,
        )

        self.assertEqual(row["case"], "flow_warp")
        self.assertEqual(row["shape"], "1x2x5x6")
        self.assertLess(row["max_abs_error"], 1e-4)
        self.assertGreater(row["torch_ms"], 0)
        self.assertGreater(row["mlx_ms"], 0)

    def test_propagation_warp_compare_runs_three_warps(self):
        from experiments.mlx_dcnv2.bench_flow_warp_compare import run_propagation_warp_case

        row = run_propagation_warp_case(
            channels=2,
            height=5,
            width=6,
            padding_mode="zeros",
            torch_device="cpu",
            warmup=0,
            iters=1,
            seed=12,
        )

        self.assertEqual(row["case"], "propagation_warp")
        self.assertEqual(row["warp_calls"], 3)
        self.assertLess(row["max_abs_error"], 1e-4)
        self.assertGreater(row["torch_ms"], 0)
        self.assertGreater(row["mlx_ms"], 0)

    def test_bridge_compare_includes_torch_to_mlx_roundtrip(self):
        from experiments.mlx_dcnv2.bench_flow_warp_compare import run_bridge_flow_warp_case

        row = run_bridge_flow_warp_case(
            channels=2,
            height=5,
            width=6,
            padding_mode="zeros",
            torch_device="cpu",
            warmup=0,
            iters=1,
            seed=13,
        )

        self.assertEqual(row["case"], "flow_warp_bridge")
        self.assertEqual(row["candidate"], "mlx_bridge")
        self.assertLess(row["max_abs_error"], 1e-4)
        self.assertGreater(row["torch_ms"], 0)
        self.assertGreater(row["mlx_ms"], 0)

    def test_fused_propagation_warp_compare_returns_timings(self):
        from experiments.mlx_dcnv2.bench_flow_warp_compare import run_fused_propagation_warp_case

        row = run_fused_propagation_warp_case(
            channels=2,
            height=5,
            width=6,
            padding_mode="zeros",
            torch_device="cpu",
            warmup=0,
            iters=1,
            seed=14,
        )

        self.assertEqual(row["case"], "propagation_warp_fused")
        self.assertEqual(row["candidate"], "mlx_fused_resident")
        self.assertEqual(row["warp_calls"], 3)
        self.assertLess(row["max_abs_error"], 1e-4)
        self.assertGreater(row["torch_ms"], 0)
        self.assertGreater(row["mlx_ms"], 0)

    def test_bridge_fused_propagation_warp_compare_returns_timings(self):
        from experiments.mlx_dcnv2.bench_flow_warp_compare import run_bridge_fused_propagation_warp_case

        row = run_bridge_fused_propagation_warp_case(
            channels=2,
            height=5,
            width=6,
            padding_mode="zeros",
            torch_device="cpu",
            warmup=0,
            iters=1,
            seed=15,
        )

        self.assertEqual(row["case"], "propagation_warp_fused_bridge")
        self.assertEqual(row["candidate"], "mlx_fused_bridge")
        self.assertLess(row["max_abs_error"], 1e-4)
        self.assertGreater(row["torch_ms"], 0)
        self.assertGreater(row["mlx_ms"], 0)

    def test_two_stage_propagation_warp_compare_returns_timings(self):
        from experiments.mlx_dcnv2.bench_flow_warp_compare import run_two_stage_propagation_warp_case

        row = run_two_stage_propagation_warp_case(
            channels=2,
            height=5,
            width=6,
            padding_mode="zeros",
            torch_device="cpu",
            warmup=0,
            iters=1,
            seed=16,
        )

        self.assertEqual(row["case"], "propagation_warp_two_stage")
        self.assertEqual(row["candidate"], "mlx_two_stage_resident")
        self.assertLess(row["max_abs_error"], 1e-4)
        self.assertGreater(row["torch_ms"], 0)
        self.assertGreater(row["mlx_ms"], 0)

    def test_bridge_two_stage_propagation_warp_compare_returns_timings(self):
        from experiments.mlx_dcnv2.bench_flow_warp_compare import run_bridge_two_stage_propagation_warp_case

        row = run_bridge_two_stage_propagation_warp_case(
            channels=2,
            height=5,
            width=6,
            padding_mode="zeros",
            torch_device="cpu",
            warmup=0,
            iters=1,
            seed=17,
        )

        self.assertEqual(row["case"], "propagation_warp_two_stage_bridge")
        self.assertEqual(row["candidate"], "mlx_two_stage_bridge")
        self.assertLess(row["max_abs_error"], 1e-4)
        self.assertGreater(row["torch_ms"], 0)
        self.assertGreater(row["mlx_ms"], 0)

    def test_window_bridge_propagation_warp_batches_steps(self):
        from experiments.mlx_dcnv2.bench_flow_warp_compare import run_window_bridge_propagation_warp_case

        row = run_window_bridge_propagation_warp_case(
            channels=2,
            height=5,
            width=6,
            steps=3,
            padding_mode="zeros",
            torch_device="cpu",
            warmup=0,
            iters=1,
            seed=18,
        )

        self.assertEqual(row["case"], "propagation_warp_window_bridge")
        self.assertEqual(row["candidate"], "mlx_two_stage_window_bridge")
        self.assertEqual(row["steps"], 3)
        self.assertLess(row["max_abs_error"], 1e-4)
        self.assertGreater(row["torch_ms"], 0)
        self.assertGreater(row["mlx_ms"], 0)


if __name__ == "__main__":
    unittest.main()
