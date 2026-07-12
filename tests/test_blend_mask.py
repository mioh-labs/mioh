import unittest

import torch

from lada.utils.mask_utils import create_blend_mask


class BlendMaskTests(unittest.TestCase):
    def test_feathered_blend_extends_outside_segmentation_mask(self):
        mask = torch.zeros((320, 320, 1), dtype=torch.uint8)
        mask[80:240, 96:224] = 255

        blend = create_blend_mask(mask.float(), feather_multiplier=1.0)

        self.assertGreater(blend[160, 230].item(), 0.0)
        self.assertAlmostEqual(blend[160, 160].item(), 1.0, places=5)

    def test_small_crop_uses_full_restoration_like_upstream(self):
        mask = torch.zeros((32, 32, 1), dtype=torch.uint8)
        mask[8:24, 10:22] = 255

        blend = create_blend_mask(mask.float(), feather_multiplier=1.0)

        self.assertTrue(torch.all(blend == 1.0))

    def test_disabled_feather_uses_the_segmentation_mask(self):
        mask = torch.zeros((32, 32, 1), dtype=torch.uint8)
        mask[8:24, 10:22] = 255

        blend = create_blend_mask(mask.float(), feather_multiplier=0.0)

        self.assertTrue(torch.equal(blend, (mask.squeeze() > 0).float()))


if __name__ == "__main__":
    unittest.main()
