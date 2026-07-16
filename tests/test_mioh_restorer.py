# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from lada.models.mioh_restorer import MiohRestorerV1
from lada.models.mioh_restorer.training import (
    MaskedVGG16PerceptualLoss,
    masked_high_frequency_loss,
    masked_psnr,
    restoration_loss,
    run_training_sequence,
)
from lada.cli.main import infer_restoration_model_name, is_restoration_model_path
from lada.restorationpipeline import load_restoration_model
from lada.restorationpipeline.frame_restorer import FrameRestorer, _restoration_clip_size
from lada.restorationpipeline.mioh_restorer import (
    CoreAIMiohRestorerRuntime,
    MiohMosaicRestorer,
    TorchMiohRestorerRuntime,
    infer_mioh_contract,
)


class MiohRestorerV1Tests(unittest.TestCase):
    def make_model(self) -> MiohRestorerV1:
        torch.manual_seed(0)
        return MiohRestorerV1(chunk_frames=4, channels=8, num_blocks=1).eval()

    def test_untrained_model_is_identity_with_persistent_state(self):
        model = self.make_model()
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.rand(1, 4, 1, 16, 16)
        state = model.initial_state(frames)

        restored, next_state = model(frames, masks, state)

        torch.testing.assert_close(restored, frames)
        self.assertEqual(next_state.shape, (1, 8, 4, 4))
        self.assertFalse(torch.equal(next_state, state))

    def test_zero_mask_preserves_input_even_when_head_changes(self):
        model = self.make_model()
        torch.nn.init.constant_(model.output_head.bias, 0.5)
        frames = torch.rand(1, 4, 3, 16, 16)
        masks = torch.zeros(1, 4, 1, 16, 16)

        restored, _state = model(frames, masks, model.initial_state(frames))

        torch.testing.assert_close(restored, frames)

    def test_model_exports_with_three_inputs_and_two_outputs(self):
        model = self.make_model()
        frames = torch.zeros(1, 4, 3, 16, 16)
        masks = torch.ones(1, 4, 1, 16, 16)
        state = model.initial_state(frames)

        exported = torch.export.export(model, (frames, masks, state))
        restored, next_state = exported.module()(frames, masks, state)

        self.assertEqual(restored.shape, frames.shape)
        self.assertEqual(next_state.shape, state.shape)

    def test_clip_adapter_handles_partial_chunk_and_bidirectional_mode(self):
        model = self.make_model()
        runtime = TorchMiohRestorerRuntime(model)
        runtime.image_size = 16
        restorer = MiohMosaicRestorer(runtime)
        video = [
            torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8)
            for _ in range(6)
        ]
        masks = [torch.ones(16, 16, dtype=torch.uint8) * 255 for _ in video]

        output = restorer.restore(video, masks, bidirectional=True)

        self.assertEqual(len(output), len(video))
        for actual, expected in zip(output, video, strict=True):
            torch.testing.assert_close(actual, expected)

    def test_clip_adapter_rejects_misaligned_masks(self):
        runtime = TorchMiohRestorerRuntime(self.make_model(), image_size=16)
        restorer = MiohMosaicRestorer(runtime)
        video = [torch.zeros(16, 16, 3, dtype=torch.uint8) for _ in range(2)]

        with self.assertRaisesRegex(ValueError, "mask count"):
            restorer.restore(video, [torch.zeros(16, 16, dtype=torch.uint8)])

    def test_training_unroll_and_losses_produce_gradients(self):
        model = self.make_model()
        inputs = torch.rand(2, 8, 3, 16, 16)
        targets = torch.rand_like(inputs)
        masks = torch.ones(2, 8, 1, 16, 16)

        restored = run_training_sequence(model, inputs, masks)
        loss = restoration_loss(restored, targets, masks)
        loss.total.backward()

        self.assertEqual(restored.shape, inputs.shape)
        self.assertTrue(torch.isfinite(loss.total))
        self.assertIsNotNone(model.output_head.weight.grad)
        self.assertGreater(float(model.output_head.weight.grad.abs().sum()), 0.0)

    def test_masked_psnr_ignores_pixels_outside_mask(self):
        target = torch.zeros(1, 1, 3, 2, 2)
        prediction = target.clone()
        prediction[..., 0, 0] = 0.1
        prediction[..., 1, 1] = 1.0
        mask = torch.zeros(1, 1, 1, 2, 2)
        mask[..., 0, 0] = 1.0

        actual = masked_psnr(prediction, target, mask)

        self.assertAlmostEqual(float(actual), 20.0, places=4)

    def test_high_frequency_loss_detects_masked_detail_error(self):
        target = torch.zeros(1, 2, 3, 16, 16)
        target[..., 6:10, 6:10] = 1.0
        restored = target.clone()
        restored[..., 7:9, 7:9] = 0.0
        masks = torch.zeros(1, 2, 1, 16, 16)
        masks[..., 4:12, 4:12] = 1.0

        actual = masked_high_frequency_loss(restored, target, masks)

        self.assertGreater(float(actual), 0.0)

    def test_masked_perceptual_loss_produces_restorer_gradients(self):
        # Identity layers exercise masking/sampling/gradient flow without
        # downloading pretrained VGG weights in the unit test.
        features = nn.Sequential(*(nn.Identity() for _ in range(16)))
        criterion = MaskedVGG16PerceptualLoss(
            frame_stride=2,
            image_size=32,
            features=features,
        )
        restored = torch.zeros(1, 4, 3, 32, 32, requires_grad=True)
        target = torch.ones_like(restored)
        masks = torch.zeros(1, 4, 1, 32, 32)
        masks[..., 8:24, 8:24] = 1.0

        perceptual = criterion(restored, target, masks)
        loss = restoration_loss(
            restored,
            target,
            masks,
            high_frequency_weight=0.1,
            perceptual_weight=0.05,
            perceptual=perceptual,
        )
        loss.total.backward()

        self.assertGreater(float(loss.perceptual.detach()), 0.0)
        self.assertTrue(torch.isfinite(loss.high_frequency))
        self.assertIsNotNone(restored.grad)
        self.assertGreater(float(restored.grad.abs().sum()), 0.0)


class CoreAIMiohRestorerRuntimeTests(unittest.TestCase):
    def test_contract_is_inferred_from_deployment_asset_name(self):
        self.assertEqual(
            infer_mioh_contract(
                Path("mioh-restorer-v1-t6-c48-s320-fp16.h17s.aimodelc")
            ),
            (6, 48, 320),
        )

    def test_runtime_uses_stateful_tensor_contract(self):
        captured = {}

        class FakeRuntime:
            def __init__(self, model_path, *, inputs, outputs, runner_path):
                captured["model_path"] = model_path
                captured["inputs"] = inputs
                captured["outputs"] = outputs
                captured["runner_path"] = runner_path

            def infer(self, values):
                return {
                    "restored": values["frames"].copy(),
                    "next_state": values["history"] + np.float16(1),
                }

            def close(self):
                captured["closed"] = True

        runtime = CoreAIMiohRestorerRuntime(
            Path("prototype.aimodelc"),
            chunk_frames=4,
            channels=8,
            image_size=16,
            runner_path="runner",
            runtime_factory=FakeRuntime,
        )
        frames = torch.rand(1, 4, 3, 16, 16, dtype=torch.float16)
        masks = torch.ones(1, 4, 1, 16, 16, dtype=torch.float16)
        state = torch.zeros(1, 8, 4, 4, dtype=torch.float16)

        restored, next_state = runtime(frames, masks, state)
        runtime.close()

        torch.testing.assert_close(restored, frames)
        torch.testing.assert_close(next_state, torch.ones_like(state))
        self.assertEqual(
            [spec.name for spec in captured["inputs"]],
            ["frames", "masks", "history"],
        )
        self.assertEqual(
            [spec.name for spec in captured["outputs"]],
            ["restored", "next_state"],
        )
        self.assertEqual(captured["runner_path"], "runner")
        self.assertTrue(captured["closed"])


class MiohRestorerPipelineTests(unittest.TestCase):
    def test_mioh_runtime_controls_detector_clip_size(self):
        runtime = mock.Mock(image_size=384)
        restorer = mock.Mock(runtime=runtime)

        self.assertEqual(
            _restoration_clip_size("mioh-restorer-v1", restorer),
            384,
        )
        self.assertEqual(_restoration_clip_size("basicvsrpp-v1.2", object()), 256)

    def test_explicit_prototype_assets_are_identified_without_changing_defaults(self):
        self.assertEqual(
            infer_restoration_model_name("/models/mioh-restorer-v1-t4-fp16.aimodelc"),
            "mioh-restorer-v1-prototype",
        )
        self.assertEqual(
            infer_restoration_model_name("/models/basicvsrpp-v1.2.aimodelc"),
            "basicvsrpp-coreai",
        )
        with tempfile.TemporaryDirectory(suffix=".mlmodelc") as model_dir:
            self.assertTrue(is_restoration_model_path(model_dir))

    def test_loader_selects_mioh_coreai_runtime_before_basicvsrpp(self):
        runtime = mock.Mock()
        restorer = mock.Mock()
        with (
            mock.patch(
                "lada.restorationpipeline.mioh_restorer.CoreAIMiohRestorerRuntime",
                return_value=runtime,
            ) as runtime_class,
            mock.patch(
                "lada.restorationpipeline.mioh_restorer.MiohMosaicRestorer",
                return_value=restorer,
            ) as restorer_class,
        ):
            result, pad_mode = load_restoration_model(
                torch.device("mps"),
                "mioh-restorer-v1-prototype",
                "prototype.aimodelc",
                None,
                True,
            )

        self.assertIs(result, restorer)
        self.assertEqual(pad_mode, "zero")
        runtime_class.assert_called_once_with(Path("prototype.aimodelc"))
        restorer_class.assert_called_once_with(runtime)

    def test_frame_restorer_passes_detection_masks_to_mioh_model(self):
        model = MiohRestorerV1(chunk_frames=4, channels=8, num_blocks=1)
        runtime = TorchMiohRestorerRuntime(model)
        runtime.image_size = 16
        mioh_restorer = MiohMosaicRestorer(runtime)
        restorer = object.__new__(FrameRestorer)
        restorer.mosaic_restoration_model_name = "mioh-restorer-v1-prototype"
        restorer.mosaic_restoration_model = mioh_restorer
        images = [torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8) for _ in range(3)]
        masks = [torch.ones(16, 16, dtype=torch.uint8) * 255 for _ in images]

        output = restorer._restore_clip_frames(images, masks)

        self.assertEqual(len(output), len(images))
        for actual, expected in zip(output, images, strict=True):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
