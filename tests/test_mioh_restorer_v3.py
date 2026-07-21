# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import unittest

import torch

from lada.models.mioh_restorer import (
    HierarchicalCoreAIShiftAlignment,
    MiohRestorerV2,
    MiohRestorerV3,
)


class MiohRestorerV3Tests(unittest.TestCase):
    @staticmethod
    def make_model() -> MiohRestorerV3:
        torch.manual_seed(11)
        return MiohRestorerV3(
            window_frames=4,
            channels=8,
            num_blocks=1,
            encoder_blocks=1,
            reconstruction_blocks=1,
            alignment_radius=1,
            first_order_dilation=1,
            second_order_dilation=2,
            alignment_key_channels=4,
            alignment_groups=2,
        )

    def test_untrained_model_is_masked_identity(self):
        model = self.make_model().eval()
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.rand(1, 4, 1, 16, 16)

        restored, forward, backward = model.forward_with_directions(
            frames, masks
        )

        torch.testing.assert_close(restored, frames)
        torch.testing.assert_close(forward, frames)
        torch.testing.assert_close(backward, frames)

    def test_zero_mask_preserves_input_after_output_heads_change(self):
        model = self.make_model().eval()
        torch.nn.init.constant_(model.output_head.bias, 0.4)
        torch.nn.init.constant_(model.direction_decoder.output_head.bias, -0.3)
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.zeros(1, 4, 1, 16, 16)

        restored, forward, backward = model.forward_with_directions(
            frames, masks
        )

        torch.testing.assert_close(restored, frames)
        torch.testing.assert_close(forward, frames)
        torch.testing.assert_close(backward, frames)

    def test_four_second_order_propagation_branches_are_present(self):
        model = self.make_model()

        self.assertEqual(
            tuple(model.propagation),
            ("backward_1", "forward_1", "backward_2", "forward_2"),
        )
        self.assertEqual(
            model.propagation["forward_2"].prior_branches,
            3,
        )

    def test_v2_initialization_reuses_encoder_but_not_incompatible_head(self):
        source = MiohRestorerV2(
            window_frames=4,
            chunk_frames=2,
            channels=8,
            num_blocks=1,
            fusion_full_channels=8,
            fusion_half_channels=8,
            fusion_quarter_channels=8,
        )
        torch.nn.init.normal_(source.forward_branch.encoder[0].weight)
        target = self.make_model()

        target.initialize_from_v2(source)

        torch.testing.assert_close(
            target.encoder[0].weight,
            source.forward_branch.encoder[0].weight,
        )
        torch.testing.assert_close(
            target.direction_decoder.output_head.weight,
            torch.zeros_like(target.direction_decoder.output_head.weight),
        )

    def test_alignment_uses_independent_channel_groups(self):
        alignment = self.make_model().propagation["forward_1"].first_order_alignment

        self.assertEqual(alignment.groups, 2)
        self.assertEqual(alignment.offset_bias.shape[:3], (1, 2, 9))

    def test_hierarchical_alignment_covers_measured_p99_motion(self):
        model = MiohRestorerV3(
            window_frames=4,
            channels=8,
            num_blocks=1,
            encoder_blocks=1,
            reconstruction_blocks=1,
            alignment_key_channels=4,
            alignment_groups=2,
            hierarchical_alignment_dilations=(9, 3, 1),
            alignment_temperature=0.5,
        )
        alignment = model.propagation["forward_1"].first_order_alignment

        self.assertIsInstance(alignment, HierarchicalCoreAIShiftAlignment)
        self.assertEqual(alignment.maximum_offset, 13)
        self.assertEqual(model.architecture_revision, 3)
        self.assertEqual(
            tuple(len(stage) for stage in alignment.stage_offsets),
            (9, 9, 9),
        )

    def test_v31_initialization_reuses_v3_except_alignment(self):
        source = self.make_model()
        torch.nn.init.normal_(source.encoder[0].weight)
        target = MiohRestorerV3(
            window_frames=4,
            channels=8,
            num_blocks=1,
            encoder_blocks=1,
            reconstruction_blocks=1,
            alignment_key_channels=4,
            alignment_groups=2,
            hierarchical_alignment_dilations=(9, 3, 1),
            alignment_temperature=0.5,
        )

        copied, fresh = target.initialize_from_v3_state_dict(
            source.state_dict()
        )

        self.assertGreater(copied, 0)
        self.assertGreater(fresh, 0)
        torch.testing.assert_close(target.encoder[0].weight, source.encoder[0].weight)

    def test_distillation_forward_captures_only_requested_alignment_calls(self):
        model = self.make_model().eval()
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.ones(1, 4, 1, 16, 16)

        restored, forward, backward, diagnostics = (
            model.forward_with_distillation(
                frames,
                masks,
                capture_branch="forward_1",
                capture_calls=frozenset({0, 2}),
            )
        )

        self.assertEqual(restored.shape, frames.shape)
        self.assertEqual(forward.shape, frames.shape)
        self.assertEqual(backward.shape, frames.shape)
        self.assertEqual([item["call_index"] for item in diagnostics], [0, 2])
        self.assertIn("first_weights", diagnostics[0])
        self.assertIn("first_confidence", diagnostics[0])
        self.assertNotIn("second_weights", diagnostics[0])
        self.assertIn("second_weights", diagnostics[1])
        self.assertIn("second_confidence", diagnostics[1])

    def test_gradient_checkpointing_preserves_training_output_and_backward(self):
        baseline = self.make_model().train()
        checkpointed = self.make_model().train()
        checkpointed.load_state_dict(baseline.state_dict())
        checkpointed.set_gradient_checkpointing(True)
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.ones(1, 4, 1, 16, 16)

        expected = baseline(frames, masks)
        actual = checkpointed(frames, masks)
        actual.mean().backward()

        torch.testing.assert_close(actual, expected)
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in checkpointed.parameters()
            )
        )

    def test_export_graph_avoids_grid_sample_and_deformable_convolution(self):
        model = self.make_model().eval()
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.ones(1, 4, 1, 16, 16)

        exported = torch.export.export(model, (frames, masks))
        restored = exported.module()(frames, masks)
        targets = {
            str(node.target)
            for node in exported.graph_module.graph.nodes
            if node.op == "call_function"
        }

        self.assertEqual(restored.shape, frames.shape)
        self.assertFalse(any("grid_sampler" in target for target in targets))
        self.assertFalse(any("deform_conv" in target for target in targets))

    def test_hierarchical_export_graph_uses_only_static_alignment(self):
        model = MiohRestorerV3(
            window_frames=3,
            channels=4,
            num_blocks=0,
            encoder_blocks=0,
            reconstruction_blocks=0,
            alignment_key_channels=2,
            alignment_groups=1,
            hierarchical_alignment_dilations=(9, 3, 1),
            alignment_temperature=0.5,
        ).eval()
        frames = torch.rand(1, 3, 3, 16, 16)
        masks = torch.ones(1, 3, 1, 16, 16)

        exported = torch.export.export(model, (frames, masks))
        targets = {
            str(node.target)
            for node in exported.graph_module.graph.nodes
            if node.op == "call_function"
        }

        self.assertFalse(any("grid_sampler" in target for target in targets))
        self.assertFalse(any("deform_conv" in target for target in targets))


if __name__ == "__main__":
    unittest.main()
