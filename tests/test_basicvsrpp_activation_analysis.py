import unittest

import torch

from lada.models.basicvsrpp.activation_analysis import (
    AlignmentCapturePolicy,
    BasicVSRPPActivationAnalyzer,
    StreamingTensorStats,
)


class _FakeSPyNet(torch.nn.Module):
    def forward(self, value):
        return value


class _FakeAlignment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deform_groups = 2
        self.max_residue_magnitude = 10
        self.conv_offset = torch.nn.Sequential(torch.nn.Conv2d(10, 54, 1))
        torch.nn.init.zeros_(self.conv_offset[-1].weight)
        torch.nn.init.zeros_(self.conv_offset[-1].bias)

    def forward(self, value, extra, flow_1, flow_2):
        # Running conv_offset is sufficient to exercise the raw-output hook;
        # returning value verifies that analysis never changes model behavior.
        self.conv_offset(extra)
        return value + 1


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spynet = _FakeSPyNet()
        self.deform_align = torch.nn.ModuleDict({"forward_1": _FakeAlignment()})


class StreamingTensorStatsTests(unittest.TestCase):
    def test_reports_exact_moments_and_bounded_percentiles(self):
        stats = StreamingTensorStats(sample_capacity=16)
        stats.update(torch.tensor([-2.0, 0.0, 2.0]))
        report = stats.as_dict()
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["mean"], 0.0)
        self.assertAlmostEqual(report["std"], (8 / 3) ** 0.5)
        self.assertEqual(report["min"], -2.0)
        self.assertEqual(report["max"], 2.0)


class BasicVSRPPActivationAnalyzerTests(unittest.TestCase):
    def test_hooks_capture_flow_alignment_groups_and_samples_without_mutation(self):
        model = _FakeModel().eval()
        value = torch.zeros(1, 4, 3, 3)
        extra = torch.zeros(1, 10, 3, 3)
        flow_1 = torch.ones(1, 2, 3, 3)
        flow_2 = torch.full((1, 2, 3, 3), 2.0)
        expected = model.deform_align["forward_1"](value, extra, flow_1, flow_2)

        with BasicVSRPPActivationAnalyzer(model) as analyzer:
            analyzer.begin_clip(3)
            model.spynet(flow_1)
            model.spynet(flow_2)
            actual = model.deform_align["forward_1"](
                value, extra, flow_1, flow_2
            )
            report = analyzer.report(top_shifts=3)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(report["clips"], 1)
        self.assertEqual(report["frames"], 3)
        self.assertIn("flow.backward.magnitude", report["metrics"])
        self.assertIn("flow.forward.magnitude", report["metrics"])
        self.assertIn("alignment.forward_1.mask", report["metrics"])
        self.assertEqual(
            report["metrics"]["alignment.forward_1.mask"]["mean"], 0.5
        )
        self.assertEqual(set(report["alignment_groups"]["forward_1"]), {"0", "1"})
        self.assertTrue(report["activation_sample_keys"])

    def test_close_removes_all_hooks(self):
        model = _FakeModel()
        analyzer = BasicVSRPPActivationAnalyzer(model)
        analyzer.close()
        analyzer.begin_clip(2)
        model.spynet(torch.ones(1, 2, 2, 2))
        self.assertEqual(analyzer.report()["metrics"], {})

    def test_callback_is_branch_stride_and_count_bounded(self):
        model = _FakeModel().eval()
        captures = []
        policy = AlignmentCapturePolicy(
            branches=frozenset({"forward_1"}),
            call_stride=2,
            max_calls_per_branch=2,
            channels=2,
            spatial_size=2,
        )
        value = torch.zeros(1, 4, 3, 3)
        extra = torch.zeros(1, 10, 3, 3)
        flow = torch.zeros(1, 2, 3, 3)
        with BasicVSRPPActivationAnalyzer(
            model, capture_policy=policy, activation_callback=captures.append
        ) as analyzer:
            analyzer.begin_clip(5)
            for _ in range(5):
                model.deform_align["forward_1"](value, extra, flow, flow)

        self.assertEqual([item.call_index for item in captures], [0, 2])
        self.assertTrue(all(item.offset.device.type == "cpu" for item in captures))
        self.assertEqual(len(analyzer.samples), 6)

    def test_callback_only_mode_skips_statistics_and_cpu_samples(self):
        model = _FakeModel().eval()
        captures = []
        value = torch.zeros(1, 4, 3, 3)
        extra = torch.zeros(1, 10, 3, 3)
        flow = torch.zeros(1, 2, 3, 3)
        with BasicVSRPPActivationAnalyzer(
            model,
            capture_policy=AlignmentCapturePolicy(max_calls_per_branch=1),
            activation_callback=captures.append,
            collect_statistics=False,
        ) as analyzer:
            analyzer.begin_clip(3)
            for _ in range(3):
                model.deform_align["forward_1"](value, extra, flow, flow)

        self.assertEqual(len(captures), 1)
        self.assertEqual(analyzer.report()["metrics"], {})
        self.assertEqual(analyzer.samples, {})
        self.assertEqual(analyzer.report()["alignment_calls"]["forward_1"], 3)

    def test_codebook_adds_three_by_three_kernel_base_positions(self):
        model = _FakeModel().eval()
        value = torch.zeros(1, 4, 3, 3)
        extra = torch.zeros(1, 10, 3, 3)
        flow = torch.zeros(1, 2, 3, 3)
        with BasicVSRPPActivationAnalyzer(model) as analyzer:
            analyzer.begin_clip(2)
            model.deform_align["forward_1"](value, extra, flow, flow)
            report = analyzer.report(top_shifts=20)

        codebook = report["alignment_groups"]["forward_1"]["0"][
            "recommended_sampling_codebook"
        ]
        self.assertEqual(
            {(entry["x"], entry["y"]) for entry in codebook},
            {(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)},
        )


if __name__ == "__main__":
    unittest.main()
