# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import numpy as np
import torch

from lada.models.rfdetr import RFDETRCoreMLSegmentationModel


class FakeCoreMLRuntime:
    def infer_selected(self, image, *, conf, max_det):
        assert image.shape == (1, 3, 576, 576)
        assert image.dtype == np.float32
        assert conf == 0.35
        assert max_det == 16
        return (
            np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32),
            np.ones((1, 144, 144), dtype=np.float32),
        )


def test_coreml_rfdetr_reuses_validated_postprocessing():
    model = RFDETRCoreMLSegmentationModel(
        Path("rfdetr-v6-576-fp32.mlpackage"),
        runtime=FakeCoreMLRuntime(),
    )
    image = torch.zeros((120, 200, 3), dtype=torch.uint8)

    result = model.inference_and_postprocess([image], [image])[0]

    torch.testing.assert_close(
        result.boxes.xyxy[0],
        torch.tensor([50.0, 30.0, 150.0, 90.0]),
    )
    assert result.boxes.conf[0] > 0.89
    assert result.masks.data.shape == (1, 144, 144)
    assert result._lada_direct_resize_masks is True
