import unittest
from unittest import mock

import torch

from lada import ModelFiles
from lada.cli.main import setup_argparser
from lada.restorationpipeline import load_models


class DetectionBackendSelectionTests(unittest.TestCase):
    def test_vr_detection_models_are_registered(self):
        vr_model = ModelFiles.get_detection_model_by_name("vr-v2-accurate")
        vr_coreml_model = ModelFiles.get_detection_model_by_name("vr-v2-accurate-coreml")

        self.assertIsNotNone(vr_model)
        self.assertIsNotNone(vr_coreml_model)
        self.assertTrue(vr_model.path.endswith("lada_mosaic_detection_model_vr_v2_accurate.pt"))
        self.assertTrue(
            vr_coreml_model.path.endswith(
                (
                    "lada_mosaic_detection_model_vr_v2_accurate.mlpackage",
                    "lada_mosaic_detection_model_vr_v2_accurate.mlmodelc",
                )
            )
        )

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

    def test_load_models_uses_coreml_backend_for_mlpackage_path(self):
        with mock.patch("lada.restorationpipeline.Yolo11SegmentationModel") as torch_model:
            with mock.patch("lada.models.yolo.yolo11_coreml_segmentation_model.Yolo11CoreMLSegmentationModel") as coreml_model:
                with mock.patch("lada.models.basicvsrpp.inference.load_model") as load_model_mock:
                    with mock.patch("lada.restorationpipeline.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer") as restorer_mock:
                        load_model_mock.return_value = object()
                        restorer_mock.return_value = object()
                        load_models(
                            torch.device("mps"),
                            "basicvsrpp-v1.2",
                            "restoration.pth",
                            None,
                            "detect.mlpackage",
                            False,
                            False,
                        )
        torch_model.assert_not_called()
        coreml_model.assert_called_once()

    def test_load_models_uses_coreml_backend_for_compiled_model_path(self):
        with mock.patch("lada.restorationpipeline.Yolo11SegmentationModel") as torch_model:
            with mock.patch("lada.models.yolo.yolo11_coreml_segmentation_model.Yolo11CoreMLSegmentationModel") as coreml_model:
                with mock.patch("lada.models.basicvsrpp.inference.load_model") as load_model_mock:
                    with mock.patch("lada.restorationpipeline.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer") as restorer_mock:
                        load_model_mock.return_value = object()
                        restorer_mock.return_value = object()
                        load_models(
                            torch.device("mps"),
                            "basicvsrpp-v1.2",
                            "restoration.pth",
                            None,
                            "detect.mlmodelc",
                            False,
                            False,
                        )
        torch_model.assert_not_called()
        coreml_model.assert_called_once()

    def test_load_models_uses_coreai_backend_for_aimodel_path(self):
        with mock.patch("lada.restorationpipeline.Yolo11SegmentationModel") as torch_model:
            with mock.patch("lada.models.yolo.yolo11_coreai_segmentation_model.Yolo11CoreAISegmentationModel") as coreai_model:
                with mock.patch("lada.models.basicvsrpp.inference.load_model") as load_model_mock:
                    with mock.patch("lada.restorationpipeline.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer") as restorer_mock:
                        load_model_mock.return_value = object()
                        restorer_mock.return_value = object()
                        load_models(
                            torch.device("mps"),
                            "basicvsrpp-v1.2",
                            "restoration.pth",
                            None,
                            "detect.aimodel",
                            False,
                            False,
                        )
        torch_model.assert_not_called()
        coreai_model.assert_called_once()

    def test_load_models_configures_jasna_v6_large_for_768px(self):
        with mock.patch(
            "lada.models.rfdetr.RFDETRCoreAISegmentationModel"
        ) as rfdetr_model:
            with mock.patch(
                "lada.models.basicvsrpp.inference.load_model"
            ) as load_model_mock:
                with mock.patch(
                    "lada.restorationpipeline.basicvsrpp_mosaic_restorer."
                    "BasicvsrppMosaicRestorer"
                ) as restorer_mock:
                    load_model_mock.return_value = object()
                    restorer_mock.return_value = object()
                    load_models(
                        torch.device("mps"),
                        "basicvsrpp-v1.2",
                        "restoration.pth",
                        None,
                        "rfdetr-v6-large-768-fp32.aimodel",
                        False,
                        False,
                    )

        rfdetr_model.assert_called_once()
        self.assertEqual(rfdetr_model.call_args.kwargs["resolution"], 768)
        self.assertEqual(rfdetr_model.call_args.kwargs["conf"], 0.40)


if __name__ == "__main__":
    unittest.main()
