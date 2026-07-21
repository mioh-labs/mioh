# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import unittest

import torch

from lada.models.mioh_restorer.distillation import (
    roi_alignment_kl_loss,
    roi_confidence_loss,
    roi_feature_energy_loss,
    teacher_source_confidence,
    teacher_hierarchical_shift_distributions,
    teacher_shift_distribution,
)


class MiohRestorerDistillationTests(unittest.TestCase):
    def test_teacher_offsets_are_projected_to_matching_shift(self):
        # 16 groups x 9 kernel points x 2 coordinates.
        offset = torch.zeros(1, 288, 2, 2)
        vectors = offset.reshape(1, 16, 9, 2, 2, 2)
        vectors[:, :8, :, 0].fill_(2.0)
        vectors[:, :8, :, 1].fill_(-1.0)
        mask = torch.ones(1, 144, 2, 2)
        shifts = ((0, 0), (2, -1), (-2, 1))

        distribution = teacher_shift_distribution(
            offset, mask, shifts, source=0, temperature=0.05
        )

        self.assertEqual(distribution.shape, (1, 8, 3, 2, 2))
        self.assertTrue(torch.all(distribution[:, :, 1] > 0.999))

    def test_matching_distribution_has_zero_kl(self):
        values = torch.softmax(torch.randn(1, 2, 3, 4, 4), dim=2)
        mask = torch.ones(1, 1, 8, 8)

        loss = roi_alignment_kl_loss(values, values, mask)

        self.assertLess(float(loss), 1e-6)

    def test_hierarchical_target_decomposes_large_teacher_offset(self):
        offset = torch.zeros(1, 288, 1, 1)
        vectors = offset.reshape(1, 16, 9, 2, 1, 1)
        vectors[:, :8, :, 0].fill_(10.0)
        vectors[:, :8, :, 1].fill_(-7.0)
        mask = torch.ones(1, 144, 1, 1)
        stages = (
            tuple((y, x) for y in (-9, 0, 9) for x in (-9, 0, 9)),
            tuple((y, x) for y in (-3, 0, 3) for x in (-3, 0, 3)),
            tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)),
        )

        targets = teacher_hierarchical_shift_distributions(
            offset,
            mask,
            stages,
            source=0,
            temperature=0.01,
        )

        expected = ((9, -9), (0, 3), (1, -1))
        for distribution, shifts, shift in zip(
            targets, stages, expected, strict=True
        ):
            index = shifts.index(shift)
            self.assertTrue(torch.all(distribution[:, :, index] > 0.999))

    def test_teacher_confidence_splits_the_two_temporal_sources(self):
        mask = torch.zeros(1, 144, 2, 2)
        grouped = mask.reshape(1, 16, 9, 2, 2)
        grouped[:, :8].fill_(0.2)
        grouped[:, 8:].fill_(0.7)

        first = teacher_source_confidence(mask, source=0)
        second = teacher_source_confidence(mask, source=1)

        torch.testing.assert_close(first, torch.full_like(first, 0.2))
        torch.testing.assert_close(second, torch.full_like(second, 0.7))
        self.assertAlmostEqual(
            float(roi_confidence_loss(first, first, torch.ones(1, 1, 4, 4))),
            0.001,
            places=6,
        )

    def test_feature_energy_works_across_channel_counts(self):
        student = torch.ones(1, 12, 4, 4)
        teacher = torch.ones(1, 8, 4, 4)
        mask = torch.ones(1, 1, 8, 8)

        loss = roi_feature_energy_loss(student, teacher, mask)

        self.assertAlmostEqual(float(loss), 0.001, places=6)


if __name__ == "__main__":
    unittest.main()
