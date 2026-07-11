import unittest

import numpy as np
import torch

from lada.restorationpipeline.frame_restorer import composite_restored_region
from lada.utils.mask_utils import create_inner_feather_mask, create_outward_feather_mask


class BlendMaskTests(unittest.TestCase):
    def test_outward_feather_is_zero_inside_and_fades_outside(self):
        mask = torch.zeros((64, 64, 1), dtype=torch.uint8)
        mask[20:44, 20:44] = 255

        feather = create_outward_feather_mask(mask, feather_pixels=8)

        self.assertTrue(torch.all(feather[mask.squeeze() > 0] == 0))
        self.assertGreater(feather[32, 44].item(), 0.0)
        self.assertEqual(feather[32, 54].item(), 0.0)

    def test_inner_feather_reaches_full_weight_away_from_boundary(self):
        mask = torch.zeros((64, 64, 1), dtype=torch.uint8)
        mask[20:44, 20:44] = 255

        feather = create_inner_feather_mask(mask, feather_pixels=6)

        self.assertGreater(feather[32, 20].item(), 0.0)
        self.assertLess(feather[32, 20].item(), 1.0)
        self.assertEqual(feather[32, 32].item(), 1.0)
        self.assertEqual(feather[32, 19].item(), 0.0)

    def test_outward_composite_keeps_restoration_sharp_inside_without_using_bad_context(self):
        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        restored = np.zeros_like(original)
        restored[20:44, 20:44] = 200
        mask = np.zeros((64, 64, 1), dtype=np.uint8)
        mask[20:44, 20:44] = 255

        composited = composite_restored_region(
            original,
            restored,
            mask,
            outward_feather_pixels=8,
            inner_feather_pixels=4,
        )

        self.assertTrue(np.all(composited[32, 32] == 200))
        self.assertGreater(composited[32, 20, 0], 100)
        self.assertLess(composited[32, 20, 0], 200)
        self.assertTrue(np.all(composited[32, 55] == 100))
        self.assertGreater(composited[32, 44, 0], 100)
        self.assertGreater(composited[32, 44, 0], restored[32, 44, 0])

    def test_disabled_outward_feather_is_a_hard_segmentation_composite(self):
        original = np.full((32, 32, 3), 100, dtype=np.uint8)
        restored = np.full_like(original, 200)
        mask = np.zeros((32, 32, 1), dtype=np.uint8)
        mask[8:24, 10:22] = 255

        composited = composite_restored_region(
            original,
            restored,
            mask,
            outward_feather_pixels=0,
        )

        self.assertTrue(np.all(composited[8:24, 10:22] == 200))
        self.assertTrue(np.all(composited[mask.squeeze() == 0] == 100))


if __name__ == "__main__":
    unittest.main()
