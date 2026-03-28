import unittest
from unittest import mock

import torch

from lada.cli.main import setup_argparser
from lada.restorationpipeline import load_models


class DetectionBackendSelectionTests(unittest.TestCase):
    def test_cli_does_not_expose_detection_backend_argument(self):
        parser = setup_argparser()
        args = parser.parse_args(["--input", "in.mp4"])
        self.assertFalse(hasattr(args, "mosaic_detection_backend"))

    def test_load_models_uses_torch_backend_by_default(self):
        with mock.patch("lada.restorationpipeline.Yolo11SegmentationModel") as torch_model:
            with mock.patch("lada.models.basicvsrpp.inference.load_model") as load_model_mock:
                with mock.patch("lada.restorationpipeline.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer") as restorer_mock:
                    load_model_mock.return_value = object()
                    restorer_mock.return_value = object()
                    load_models(
                        torch.device("cpu"),
                        "basicvsrpp-v1.2",
                        "restoration.pth",
                        None,
                        "detect.pt",
                        False,
                        False,
                    )
        torch_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
