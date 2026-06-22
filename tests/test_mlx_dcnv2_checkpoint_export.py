import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.mlx_dcnv2.export_deform_alignment import export_deform_alignment_weights


class MLXDCNv2CheckpointExportTests(unittest.TestCase):
    def test_exports_deform_alignment_weights_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "fixture.pth"
            output_dir = root / "exported"
            state = {
                "generator_ema.deform_align.backward_1.weight": torch.ones(64, 128, 3, 3),
                "generator_ema.deform_align.backward_1.bias": torch.arange(64, dtype=torch.float32),
                "generator_ema.deform_align.backward_1.conv_offset.0.weight": torch.ones(64, 196, 3, 3) * 2,
                "generator_ema.deform_align.backward_1.conv_offset.0.bias": torch.zeros(64),
                "generator_ema.deform_align.backward_1.conv_offset.2.weight": torch.ones(64, 64, 3, 3) * 3,
                "generator_ema.deform_align.backward_1.conv_offset.2.bias": torch.zeros(64),
                "generator_ema.deform_align.backward_1.conv_offset.4.weight": torch.ones(64, 64, 3, 3) * 4,
                "generator_ema.deform_align.backward_1.conv_offset.4.bias": torch.zeros(64),
                "generator_ema.deform_align.backward_1.conv_offset.6.weight": torch.ones(432, 64, 3, 3) * 5,
                "generator_ema.deform_align.backward_1.conv_offset.6.bias": torch.zeros(432),
            }
            torch.save(state, checkpoint)

            manifest_path = export_deform_alignment_weights(
                checkpoint_path=checkpoint,
                output_dir=output_dir,
                prefix="generator_ema",
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["prefix"], "generator_ema")
            self.assertIn("backward_1", manifest["modules"])
            self.assertEqual(manifest["modules"]["backward_1"]["weight_shape"], [64, 128, 3, 3])
            npz_path = output_dir / manifest["modules"]["backward_1"]["npz"]
            self.assertTrue(npz_path.exists())
            exported = np.load(npz_path)
            np.testing.assert_allclose(exported["weight"], np.ones((64, 128, 3, 3), dtype=np.float32))
            np.testing.assert_allclose(exported["conv_offset.6.weight"], np.ones((432, 64, 3, 3), dtype=np.float32) * 5)

    def test_exports_backbone_weights_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "fixture.pth"
            output_dir = root / "exported"
            state = {
                "generator_ema.backbone.backward_1.main.0.weight": torch.ones(64, 128, 3, 3),
                "generator_ema.backbone.backward_1.main.0.bias": torch.arange(64, dtype=torch.float32),
            }
            for block_index in range(15):
                state[f"generator_ema.backbone.backward_1.main.2.{block_index}.conv1.weight"] = (
                    torch.ones(64, 64, 3, 3) * (block_index + 1)
                )
                state[f"generator_ema.backbone.backward_1.main.2.{block_index}.conv1.bias"] = torch.zeros(64)
                state[f"generator_ema.backbone.backward_1.main.2.{block_index}.conv2.weight"] = (
                    torch.ones(64, 64, 3, 3) * (block_index + 2)
                )
                state[f"generator_ema.backbone.backward_1.main.2.{block_index}.conv2.bias"] = torch.zeros(64)
            torch.save(state, checkpoint)

            manifest_path = export_deform_alignment_weights(
                checkpoint_path=checkpoint,
                output_dir=output_dir,
                prefix="generator_ema",
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertIn("backward_1", manifest["backbones"])
            self.assertEqual(manifest["backbones"]["backward_1"]["input_channels"], 128)
            self.assertEqual(manifest["backbones"]["backward_1"]["mid_channels"], 64)
            self.assertEqual(manifest["backbones"]["backward_1"]["num_blocks"], 15)
            npz_path = output_dir / manifest["backbones"]["backward_1"]["npz"]
            self.assertTrue(npz_path.exists())
            exported = np.load(npz_path)
            np.testing.assert_allclose(exported["main.0.weight"], np.ones((64, 128, 3, 3), dtype=np.float32))
            np.testing.assert_allclose(exported["main.2.14.conv2.weight"], np.ones((64, 64, 3, 3), dtype=np.float32) * 16)

    def test_exports_feature_extract_weights_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "fixture.pth"
            output_dir = root / "exported"
            state = {
                "generator_ema.feat_extract.0.weight": torch.ones(64, 3, 3, 3),
                "generator_ema.feat_extract.0.bias": torch.arange(64, dtype=torch.float32),
                "generator_ema.feat_extract.2.weight": torch.ones(64, 64, 3, 3) * 2,
                "generator_ema.feat_extract.2.bias": torch.zeros(64),
                "generator_ema.feat_extract.4.main.0.weight": torch.ones(64, 64, 3, 3) * 3,
                "generator_ema.feat_extract.4.main.0.bias": torch.zeros(64),
            }
            for block_index in range(5):
                state[f"generator_ema.feat_extract.4.main.2.{block_index}.conv1.weight"] = (
                    torch.ones(64, 64, 3, 3) * (block_index + 1)
                )
                state[f"generator_ema.feat_extract.4.main.2.{block_index}.conv1.bias"] = torch.zeros(64)
                state[f"generator_ema.feat_extract.4.main.2.{block_index}.conv2.weight"] = (
                    torch.ones(64, 64, 3, 3) * (block_index + 2)
                )
                state[f"generator_ema.feat_extract.4.main.2.{block_index}.conv2.bias"] = torch.zeros(64)
            torch.save(state, checkpoint)

            manifest_path = export_deform_alignment_weights(
                checkpoint_path=checkpoint,
                output_dir=output_dir,
                prefix="generator_ema",
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["feature_extract"]["input_channels"], 3)
            self.assertEqual(manifest["feature_extract"]["mid_channels"], 64)
            self.assertEqual(manifest["feature_extract"]["num_blocks"], 5)
            npz_path = output_dir / manifest["feature_extract"]["npz"]
            self.assertTrue(npz_path.exists())
            exported = np.load(npz_path)
            np.testing.assert_allclose(exported["0.weight"], np.ones((64, 3, 3, 3), dtype=np.float32))
            np.testing.assert_allclose(exported["4.main.2.4.conv2.weight"], np.ones((64, 64, 3, 3), dtype=np.float32) * 6)

    def test_exports_reconstruction_weights_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "fixture.pth"
            output_dir = root / "exported"
            state = {
                "generator_ema.reconstruction.main.0.weight": torch.ones(64, 320, 3, 3),
                "generator_ema.reconstruction.main.0.bias": torch.arange(64, dtype=torch.float32),
                "generator_ema.upsample1.upsample_conv.weight": torch.ones(256, 64, 3, 3) * 2,
                "generator_ema.upsample1.upsample_conv.bias": torch.zeros(256),
                "generator_ema.upsample2.upsample_conv.weight": torch.ones(256, 64, 3, 3) * 3,
                "generator_ema.upsample2.upsample_conv.bias": torch.zeros(256),
                "generator_ema.conv_hr.weight": torch.ones(64, 64, 3, 3) * 4,
                "generator_ema.conv_hr.bias": torch.zeros(64),
                "generator_ema.conv_last.weight": torch.ones(3, 64, 3, 3) * 5,
                "generator_ema.conv_last.bias": torch.zeros(3),
            }
            for block_index in range(5):
                state[f"generator_ema.reconstruction.main.2.{block_index}.conv1.weight"] = (
                    torch.ones(64, 64, 3, 3) * (block_index + 1)
                )
                state[f"generator_ema.reconstruction.main.2.{block_index}.conv1.bias"] = torch.zeros(64)
                state[f"generator_ema.reconstruction.main.2.{block_index}.conv2.weight"] = (
                    torch.ones(64, 64, 3, 3) * (block_index + 2)
                )
                state[f"generator_ema.reconstruction.main.2.{block_index}.conv2.bias"] = torch.zeros(64)
            torch.save(state, checkpoint)

            manifest_path = export_deform_alignment_weights(
                checkpoint_path=checkpoint,
                output_dir=output_dir,
                prefix="generator_ema",
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["reconstruction"]["input_channels"], 320)
            self.assertEqual(manifest["reconstruction"]["mid_channels"], 64)
            self.assertEqual(manifest["reconstruction"]["num_blocks"], 5)
            npz_path = output_dir / manifest["reconstruction"]["npz"]
            self.assertTrue(npz_path.exists())
            exported = np.load(npz_path)
            np.testing.assert_allclose(exported["reconstruction.main.0.weight"], np.ones((64, 320, 3, 3), dtype=np.float32))
            np.testing.assert_allclose(exported["conv_last.weight"], np.ones((3, 64, 3, 3), dtype=np.float32) * 5)

    def test_exports_spynet_weights_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "fixture.pth"
            output_dir = root / "exported"
            state = {
                "generator_ema.spynet.mean": torch.ones(1, 3, 1, 1),
                "generator_ema.spynet.std": torch.ones(1, 3, 1, 1) * 2,
            }
            channels = [8, 32, 64, 32, 16, 2]
            for pyramid_level in range(6):
                for layer in range(5):
                    state[f"generator_ema.spynet.basic_module.{pyramid_level}.basic_module.{layer}.conv.weight"] = (
                        torch.ones(channels[layer + 1], channels[layer], 7, 7) * (pyramid_level + layer + 1)
                    )
                    state[f"generator_ema.spynet.basic_module.{pyramid_level}.basic_module.{layer}.conv.bias"] = torch.zeros(channels[layer + 1])
            torch.save(state, checkpoint)

            manifest_path = export_deform_alignment_weights(
                checkpoint_path=checkpoint,
                output_dir=output_dir,
                prefix="generator_ema",
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["spynet"]["num_modules"], 6)
            npz_path = output_dir / manifest["spynet"]["npz"]
            self.assertTrue(npz_path.exists())
            exported = np.load(npz_path)
            np.testing.assert_allclose(exported["mean"], np.ones((1, 3, 1, 1), dtype=np.float32))
            np.testing.assert_allclose(
                exported["basic_module.5.basic_module.4.conv.weight"],
                np.ones((2, 16, 7, 7), dtype=np.float32) * 10,
            )


if __name__ == "__main__":
    unittest.main()
