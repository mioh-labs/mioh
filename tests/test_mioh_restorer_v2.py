# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import unittest

import torch

from lada.models.mioh_restorer import (
    MiohRestorerV1,
    MiohRestorerV2,
    TemporalPatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    masked_multiscale_structural_loss,
    temporal_discriminator_input,
)


class MiohRestorerV2Tests(unittest.TestCase):
    @staticmethod
    def make_model() -> MiohRestorerV2:
        torch.manual_seed(7)
        return MiohRestorerV2(
            window_frames=4,
            chunk_frames=2,
            channels=8,
            num_blocks=1,
            fusion_full_channels=8,
            fusion_half_channels=8,
            fusion_quarter_channels=8,
        )

    def test_default_model_has_target_capacity(self):
        model = MiohRestorerV2()
        parameters = sum(parameter.numel() for parameter in model.parameters())

        self.assertGreaterEqual(parameters, 6_000_000)
        self.assertLessEqual(parameters, 8_000_000)

    def test_untrained_bidirectional_model_is_identity(self):
        model = self.make_model().eval()
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.rand(1, 4, 1, 16, 16)

        restored, forward, backward = model.forward_with_directions(frames, masks)

        torch.testing.assert_close(restored, frames)
        torch.testing.assert_close(forward, frames)
        torch.testing.assert_close(backward, frames)

    def test_zero_mask_preserves_input_after_heads_change(self):
        model = self.make_model().eval()
        torch.nn.init.constant_(model.forward_branch.output_head.bias, 0.4)
        torch.nn.init.constant_(model.backward_branch.output_head.bias, -0.3)
        torch.nn.init.constant_(model.fusion.detail_head.bias, 0.5)
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.zeros(1, 4, 1, 16, 16)

        restored = model(frames, masks)

        torch.testing.assert_close(restored, frames)

    def test_v1_initialization_copies_both_temporal_branches(self):
        source = MiohRestorerV1(chunk_frames=2, channels=8, num_blocks=1)
        torch.nn.init.normal_(source.output_head.weight)
        model = self.make_model()

        model.initialize_branches_from_v1(source.state_dict())

        for name, expected in source.state_dict().items():
            torch.testing.assert_close(model.forward_branch.state_dict()[name], expected)
            torch.testing.assert_close(model.backward_branch.state_dict()[name], expected)

    def test_model_exports_with_two_inputs_and_one_output(self):
        model = self.make_model().eval()
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.ones(1, 4, 1, 16, 16)

        exported = torch.export.export(model, (frames, masks))
        restored = exported.module()(frames, masks)

        self.assertEqual(restored.shape, frames.shape)

    def test_multiscale_structural_loss_detects_masked_error(self):
        target = torch.zeros(1, 4, 3, 32, 32)
        target[..., 8:24, 8:24] = 1.0
        prediction = target.clone()
        prediction[..., 12:20, 12:20] = 0.0
        masks = torch.zeros(1, 4, 1, 32, 32)
        masks[..., 6:26, 6:26] = 1.0

        loss = masked_multiscale_structural_loss(
            prediction,
            target,
            masks,
            frame_stride=2,
        )

        self.assertGreater(float(loss), 0.0)

    def test_temporal_patch_gan_produces_generator_gradients(self):
        discriminator = TemporalPatchDiscriminator(base_channels=4)
        target = torch.rand(1, 5, 3, 32, 32)
        fake = torch.rand(1, 5, 3, 32, 32, requires_grad=True)
        masks = torch.ones(1, 5, 1, 32, 32)
        real_input = temporal_discriminator_input(
            target, target, masks, frame_stride=2, image_size=32
        )
        fake_input = temporal_discriminator_input(
            fake, target, masks, frame_stride=2, image_size=32
        )

        real_logits = discriminator(real_input)
        fake_logits = discriminator(fake_input)
        discriminator_loss = discriminator_hinge_loss(
            real_logits, fake_logits.detach()
        )
        generator_loss = generator_hinge_loss(fake_logits)
        generator_loss.backward()

        self.assertTrue(torch.isfinite(discriminator_loss))
        self.assertIsNotNone(fake.grad)
        self.assertGreater(float(fake.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
