# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lada.models.mioh_restorer.distillation_v4 import (
    HIER27_EIGHTH_COARSE,
    V4FeatureDistillationAdapter,
    compute_spynet_pair_flows,
    dense_flow_to_hier27_distributions,
    exact_motion_to_hier27_distributions,
    extract_basicvsrpp_reconstruction_features,
    load_basicvsrpp_feature_teacher,
    load_spynet_teacher,
    projected_roi_feature_loss,
    roi_shift_kl_loss,
)


class _FakeSPyNet(nn.Module):
    def forward(self, reference, support):
        difference = support[:, :2].mean(dim=(-2, -1), keepdim=True) - reference[
            :, :2
        ].mean(dim=(-2, -1), keepdim=True)
        return difference.expand(-1, -1, *reference.shape[-2:])


class _FakeFeatureExtract(nn.Module):
    def forward(self, values):
        reduced = F.interpolate(values.mean(dim=1, keepdim=True), scale_factor=0.25)
        return reduced.expand(-1, 64, -1, -1)


class _FakeReconstruction(nn.Module):
    def forward(self, values):
        chunks = values.reshape(values.shape[0], 5, 64, *values.shape[-2:])
        return chunks.sum(dim=1)


class _FakeBasicVSRPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = _FakeFeatureExtract()
        self.reconstruction = _FakeReconstruction()

    def compute_flow(self, low_resolution):
        shape = (
            low_resolution.shape[0],
            low_resolution.shape[1] - 1,
            2,
            *low_resolution.shape[-2:],
        )
        zeros = low_resolution.new_zeros(shape)
        return zeros, zeros

    def propagate(self, features, flows, branch):
        branch_number = len(features)
        features[branch] = [
            item + float(branch_number) for item in features["spatial"]
        ]
        return features


class V4DistillationTests(unittest.TestCase):
    def test_exact_motion_decomposes_full_reach_without_approximation(self):
        displacement = torch.tensor([[40.0, -40.0], [16.0, -8.0]])
        coarse, middle, fine = exact_motion_to_hier27_distributions(
            displacement, eighth_size=(3, 4)
        )
        self.assertEqual(coarse.shape, (2, 9, 3, 4))
        self.assertEqual(middle.shape, (2, 9, 3, 4))
        self.assertEqual(fine.shape, (2, 9, 6, 8))
        offsets_coarse = torch.tensor(HIER27_EIGHTH_COARSE)
        offsets_middle = torch.tensor(
            [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)]
        )
        offsets_fine = torch.tensor(
            [(y * 2, x * 2) for y in (-1, 0, 1) for x in (-1, 0, 1)]
        )
        reconstructed_eighth = (
            offsets_coarse[coarse[:, :, 0, 0].argmax(dim=1)]
            + offsets_middle[middle[:, :, 0, 0].argmax(dim=1)]
            + offsets_fine[fine[:, :, 0, 0].argmax(dim=1)] / 2
        )
        torch.testing.assert_close(
            reconstructed_eighth.float(), displacement / 8
        )

    def test_zero_flow_selects_the_center_in_all_three_banks(self):
        flow = torch.zeros(2, 2, 12, 16)
        distributions = dense_flow_to_hier27_distributions(
            flow, temperature=0.01
        )
        self.assertEqual(distributions[0].shape, (2, 9, 6, 8))
        self.assertEqual(distributions[1].shape, (2, 9, 6, 8))
        self.assertEqual(distributions[2].shape, (2, 9, 12, 16))
        for distribution in distributions:
            self.assertTrue(torch.equal(distribution.argmax(dim=1), torch.full_like(distribution[:, 0], 4, dtype=torch.long)))
            torch.testing.assert_close(
                distribution.sum(dim=1),
                torch.ones_like(distribution[:, 0]),
            )

    def test_positive_spynet_x_flow_maps_to_negative_v4_horizontal_shift(self):
        flow = torch.zeros(1, 2, 8, 8)
        flow[:, 0].fill_(6.0)
        coarse, middle, fine = dense_flow_to_hier27_distributions(
            flow, temperature=0.01
        )
        negative_horizontal = HIER27_EIGHTH_COARSE.index((0, -3))
        self.assertTrue(
            torch.equal(
                coarse.argmax(dim=1),
                torch.full_like(coarse[:, 0], negative_horizontal, dtype=torch.long),
            )
        )
        self.assertTrue(torch.equal(middle.argmax(dim=1), torch.full_like(middle[:, 0], 4, dtype=torch.long)))
        self.assertTrue(torch.equal(fine.argmax(dim=1), torch.full_like(fine[:, 0], 4, dtype=torch.long)))

    def test_roi_shift_kl_is_zero_for_identical_distributions(self):
        distribution = torch.softmax(torch.randn(2, 9, 6, 8), dim=1)
        roi = torch.ones(2, 1, 24, 32)
        same = roi_shift_kl_loss(distribution, distribution, roi)
        other = roi_shift_kl_loss(
            distribution,
            torch.roll(distribution, shifts=1, dims=1),
            roi,
        )
        self.assertLess(abs(float(same)), 1e-6)
        self.assertGreater(float(other), 0.0)

    def test_feature_adapter_and_projected_loss(self):
        adapter = V4FeatureDistillationAdapter(64, 64)
        with torch.no_grad():
            adapter.projection.weight.zero_()
            for channel in range(64):
                adapter.projection.weight[channel, channel, 0, 0] = 1.0
        student = torch.randn(2, 5, 64, 8, 8)
        teacher = student.clone()
        roi = torch.ones(2, 5, 1, 32, 32)
        projected = adapter(student)
        self.assertEqual(projected.shape, teacher.shape)
        loss = projected_roi_feature_loss(student, teacher, roi, adapter)
        self.assertLess(float(loss.detach()), 1e-8)

    def test_extracts_only_requested_basicvsrpp_reconstruction_features(self):
        teacher = _FakeBasicVSRPP().eval()
        values = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1, 1, 1)
        frames = values.expand(1, 9, 3, 256, 256) / 9.0
        result = extract_basicvsrpp_reconstruction_features(teacher, frames)
        self.assertEqual(result.shape, (1, 5, 64, 64, 64))
        # The selected frames are 2..6 and therefore remain distinguishable.
        means = result.mean(dim=(0, 2, 3, 4))
        self.assertTrue(torch.all(means[1:] > means[:-1]))

    def test_pair_flow_helper_preserves_batch_and_pair_order(self):
        frames = torch.zeros(2, 4, 3, 32, 32)
        for batch in range(2):
            for frame in range(4):
                frames[batch, frame, :2].fill_(batch * 10 + frame)
        flows = compute_spynet_pair_flows(
            _FakeSPyNet(), frames, ((0, 1), (3, 1)), chunk_size=1
        )
        self.assertEqual(flows.shape, (2, 2, 2, 8, 8))
        torch.testing.assert_close(flows[:, 0], torch.ones_like(flows[:, 0]))
        torch.testing.assert_close(flows[:, 1], torch.full_like(flows[:, 1], -2.0))

    def test_spynet_only_loader_accepts_generator_ema_prefix(self):
        from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import SPyNet

        original = SPyNet(pretrained=None).eval()
        checkpoint = {
            f"generator_ema.spynet.{key}": value
            for key, value in original.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pth"
            torch.save(checkpoint, path)
            loaded = load_spynet_teacher(path, "cpu")
        self.assertFalse(loaded.training)
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
        first_key = next(iter(original.state_dict()))
        torch.testing.assert_close(
            loaded.state_dict()[first_key], original.state_dict()[first_key]
        )

    def test_basicvsrpp_loader_accepts_stripped_nested_state_dict(self):
        from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet

        original = BasicVSRPlusPlusGanNet(
            mid_channels=64, num_blocks=15, spynet_pretrained=None
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pth"
            torch.save({"state_dict": original.state_dict()}, path)
            loaded = load_basicvsrpp_feature_teacher(path, "cpu")
        self.assertFalse(loaded.training)
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
        first_key = next(iter(original.state_dict()))
        torch.testing.assert_close(
            loaded.state_dict()[first_key], original.state_dict()[first_key]
        )


if __name__ == "__main__":
    unittest.main()
