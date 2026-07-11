import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.apple import export_v4_fast_coreai as exporter


class FakeSegmentationModel(torch.nn.Module):
    def forward(self, image):
        batch = image.shape[0]
        candidates = torch.zeros((batch, 38, 8400), dtype=image.dtype)
        prototypes = torch.zeros((batch, 32, 160, 160), dtype=image.dtype)
        return ((candidates, prototypes), None)


class V4FastCoreAIExportTests(unittest.TestCase):
    def test_defaults_define_fixed_fp16_contract(self):
        args = exporter.parse_args([])

        self.assertEqual(args.model, Path("model_weights/lada_mosaic_detection_model_v4_fast.pt"))
        self.assertEqual(args.output, Path("model_weights/lada_mosaic_detection_model_v4_fast-fp16.aimodel"))
        self.assertEqual(args.imgsz, 640)

    def test_wrapper_returns_raw_candidates_and_prototypes(self):
        wrapper = exporter.RawSegmentationOutputs(FakeSegmentationModel())
        image = torch.zeros((1, 3, 640, 640), dtype=torch.float16)

        candidates, prototypes = wrapper(image)

        self.assertEqual(tuple(candidates.shape), (1, 38, 8400))
        self.assertEqual(tuple(prototypes.shape), (1, 32, 160, 160))

    def test_converter_names_raw_outputs(self):
        coreai_torch = mock.Mock()
        converter = coreai_torch.TorchConverter.return_value
        exported = mock.sentinel.exported

        result = exporter.convert_exported_program(exported, coreai_torch)

        self.assertIs(result, converter.to_coreai.return_value)
        converter.add_exported_program.assert_called_once_with(
            exported,
            input_names=["image"],
            output_names=["candidates", "prototypes"],
        )


if __name__ == "__main__":
    unittest.main()
