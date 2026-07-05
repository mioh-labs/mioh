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
