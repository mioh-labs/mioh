# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import unittest
from unittest import mock

import torch

from lada.models.yolo.yolo11_coreml_segmentation_model import Yolo11CoreMLSegmentationModel


def make_model(**kwargs):
    with mock.patch("lada.models.yolo.yolo11_coreml_segmentation_model.AutoBackend") as autobackend:
        backend = autobackend.return_value
        backend.task = "segment"
        backend.names = {0: "mosaic_nsfw", 1: "mosaic_sfw_head"}
        model = Yolo11CoreMLSegmentationModel("detect.mlpackage", torch.device("mps"), **kwargs)
    return model, backend


class Yolo11CoreMLSegmentationModelTests(unittest.TestCase):
    def test_rejects_non_mlpackage_path(self):
        with self.assertRaises(ValueError):
            Yolo11CoreMLSegmentationModel("detect.pt", torch.device("cpu"))

    def test_runs_on_cpu_tensors_regardless_of_requested_device(self):
        model, _ = make_model()
        self.assertEqual(model.device.type, "cpu")
        self.assertEqual(model.dtype, torch.float32)

    def test_letterbox_pads_to_full_square(self):
        model, _ = make_model()
        self.assertFalse(model.letterbox.auto)
        self.assertEqual(list(model.imgsz), [640, 640])

    def test_kwargs_flow_into_predict_config(self):
        model, _ = make_model(conf=0.15, classes=[0])
        self.assertEqual(model.args.conf, 0.15)
        self.assertEqual(model.args.classes, [0])

    def test_loads_coreml_model_once_with_selected_compute_unit(self):
        ml_model = mock.Mock(return_value=object())
        fake_coremltools = mock.Mock()
        fake_coremltools.ComputeUnit.CPU_AND_NE = mock.sentinel.cpu_and_ne
        fake_coremltools.models.MLModel = ml_model

        def make_backend(*, model, **_kwargs):
            backend = mock.Mock()
            backend.task = "segment"
            backend.model = fake_coremltools.models.MLModel(model)
            return backend

        with (
            mock.patch.dict("sys.modules", {"coremltools": fake_coremltools}),
            mock.patch(
                "lada.models.yolo.yolo11_coreml_segmentation_model.AutoBackend",
                side_effect=make_backend,
            ),
        ):
            Yolo11CoreMLSegmentationModel("detect.mlpackage", torch.device("mps"))

        ml_model.assert_called_once_with(
            "detect.mlpackage",
            compute_units=mock.sentinel.cpu_and_ne,
        )

    def test_loads_precompiled_coreml_model_without_opening_package(self):
        compiled_model = mock.Mock()
        fake_coremltools = mock.Mock()
        fake_coremltools.ComputeUnit.CPU_AND_NE = mock.sentinel.cpu_and_ne
        fake_coremltools.models.CompiledMLModel.return_value = compiled_model

        with mock.patch.dict("sys.modules", {"coremltools": fake_coremltools}):
            model = Yolo11CoreMLSegmentationModel(
                "detect.mlmodelc",
                torch.device("mps"),
            )

        fake_coremltools.models.CompiledMLModel.assert_called_once_with(
            "detect.mlmodelc",
            compute_units=mock.sentinel.cpu_and_ne,
        )
        self.assertIs(model.model.model, compiled_model)

    def test_inference_merges_single_image_outputs(self):
        model, backend = make_model()
        det = torch.zeros(1, 38, 8400)
        proto = torch.zeros(1, 32, 160, 160)
        backend.side_effect = lambda im, **kw: [det.clone(), proto.clone()]
        preds = model.inference(torch.zeros(3, 3, 640, 640))
        self.assertEqual(backend.call_count, 3)
        det_batch, proto_batch = preds[0]
        self.assertEqual(tuple(det_batch.shape), (3, 38, 8400))
        self.assertEqual(tuple(proto_batch.shape), (3, 32, 160, 160))
        # postprocess reads protos via preds[0][-1]
        self.assertIs(preds[0][-1], proto_batch)


if __name__ == "__main__":
    unittest.main()
