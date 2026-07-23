import unittest

import torch

from lada.utils.mask_utils import (
    create_blend_mask,
    stabilize_temporal_mask_tensor,
    stabilize_temporal_masks,
)


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

    def test_soft_mask_alpha_is_preserved_for_compositing(self):
        mask = torch.zeros((32, 32, 1), dtype=torch.float32)
        mask[8:24, 10:22] = 0.5

        blend = create_blend_mask(mask, feather_multiplier=0.0)

        self.assertAlmostEqual(blend[16, 16].item(), 0.5, places=6)

    def test_temporal_stabilizer_retains_neighbour_boundary_softly(self):
        masks = torch.zeros(3, 1, 32, 32)
        masks[0, :, 8:24, 8:24] = 1
        masks[1, :, 8:24, 8:20] = 1
        masks[2, :, 8:24, 8:24] = 1

        stable = stabilize_temporal_mask_tensor(
            masks, spatial_radius=0, feather_radius=0
        )

        self.assertEqual(stable[1, 0, 16, 16].item(), 1.0)
        self.assertGreater(stable[1, 0, 16, 22].item(), 0.8)

    def test_temporal_stabilizer_list_preserves_uint8_layout(self):
        masks = [torch.zeros(16, 16, 1, dtype=torch.uint8) for _ in range(3)]
        masks[0][4:12, 4:12] = 255

        stable = stabilize_temporal_masks(
            masks, spatial_radius=0, feather_radius=0
        )

        self.assertEqual(len(stable), 3)
        self.assertEqual(stable[0].shape, masks[0].shape)
        self.assertEqual(stable[0].dtype, torch.uint8)
        self.assertGreater(stable[1].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
