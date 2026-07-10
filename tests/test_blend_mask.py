import unittest

import torch

from lada.utils.mask_utils import create_blend_mask


class BlendMaskTests(unittest.TestCase):
    def test_feathered_blend_never_extends_outside_segmentation_mask(self):
        mask = torch.zeros((32, 32, 1), dtype=torch.uint8)
        mask[8:24, 10:22] = 255

        blend = create_blend_mask(mask.float(), feather_multiplier=1.0)

        self.assertTrue(torch.all(blend[mask.squeeze() == 0] == 0))
        self.assertEqual(blend[16, 16].item(), 1.0)

    def test_disabled_feather_uses_the_segmentation_mask(self):
        mask = torch.zeros((32, 32, 1), dtype=torch.uint8)
        mask[8:24, 10:22] = 255

        blend = create_blend_mask(mask.float(), feather_multiplier=0.0)

        self.assertTrue(torch.equal(blend, (mask.squeeze() > 0).float()))


if __name__ == "__main__":
    unittest.main()
