# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core ML backend for the LADA mosaic detection model.

Runs a `.mlpackage` exported by scripts/apple/export_v4_fast_coreml.py
through coremltools (Neural Engine / GPU as chosen by Core ML). Torch-side
tensors stay on CPU, so detection never takes the process-wide MPS
execution lock and restoration keeps the Metal queue to itself.
"""

import logging
import os

import numpy as np
import torch
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.checks import check_imgsz

from lada.utils import ImageTensor
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)


class _CoremltoolsTorchVersionWarningFilter(logging.Filter):
    """
    coremltools warns at import time that the installed torch is newer than
    the version its converter was tested with. That only concerns model
    conversion; runtime inference goes through Core ML and never touches the
    torch converter, so keep the warning out of CLI/GUI logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "has not been tested with coremltools" not in record.getMessage()


logging.getLogger("coremltools").addFilter(_CoremltoolsTorchVersionWarningFilter())


class _CompiledCoreMLBackend:
    """Small AutoBackend-compatible adapter for a compiled Core ML model."""

    task = "segment"

    def __init__(self, model_path: str, coremltools, compute_unit):
        self.model = coremltools.models.CompiledMLModel(
            model_path,
            compute_units=compute_unit,
        )

    def eval(self):
        return self

    def __call__(self, image_batch: torch.Tensor, **_kwargs):
        image = image_batch[0].permute(1, 2, 0).detach().cpu().numpy()
        pil_image = Image.fromarray((image * 255).astype("uint8"))
        outputs = list(self.model.predict({"image": pil_image}).values())
        if len(outputs) == 2 and outputs[1].ndim != 4:
            outputs.reverse()
        return [torch.from_numpy(np.asarray(output)) for output in outputs]


class Yolo11CoreMLSegmentationModel(Yolo11SegmentationModel):
    def __init__(self, model_path: str, device=None, imgsz=640, fp16=False, **kwargs):
        model_path = str(model_path)
        if not model_path.endswith((".mlpackage", ".mlmodelc")):
            raise ValueError(f"Expected a .mlpackage or .mlmodelc path, got {model_path!r}")
        if device is not None and torch.device(device).type != "cpu":
            logger.info("Core ML detection ignores torch device %s; compute unit is chosen by Core ML", device)

        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        # The exported Core ML model has a fixed square input, so pad to the
        # full square instead of the stride-aligned rectangle the torch path uses.
        self.letterbox = LetterBox(self.imgsz, auto=False, stride=self.stride)

        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "device": "cpu", "half": False}
        self.args = get_cfg(DEFAULT_CFG, {**custom, **kwargs})

        self.device = torch.device("cpu")
        self.model = self._load_backend(model_path)
        task = getattr(self.model, "task", None)
        if task != "segment":
            raise ValueError(f"Expected segment model, got {task!r}")
        self.model.eval()
        self.dtype = torch.float32

    def _load_backend(self, model_path: str):
        """
        Load the underlying MLModel once, pinned to CPU+ANE by default.

        Ultralytics does not expose Core ML compute units through AutoBackend,
        so temporarily wrap its MLModel constructor. Reloading the model after
        AutoBackend initialization can block while Core ML tears down the first
        model package on macOS.
        """
        unit_name = os.environ.get("LADA_COREML_COMPUTE_UNITS", "CPU_AND_NE").upper()
        try:
            import coremltools as ct
            unit = getattr(ct.ComputeUnit, unit_name)
        except Exception as e:
            logger.warning("Could not select Core ML compute units %s, keeping default: %s", unit_name, e)
            return AutoBackend(
                model=model_path,
                device=self.device,
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=False,
                verbose=False,
            )

        if model_path.endswith(".mlmodelc"):
            logger.info("Loading precompiled Core ML detection model with %s", unit_name)
            return _CompiledCoreMLBackend(model_path, ct, unit)

        original_mlmodel = ct.models.MLModel

        def load_mlmodel_once(*args, **kwargs):
            kwargs.setdefault("compute_units", unit)
            return original_mlmodel(*args, **kwargs)

        ct.models.MLModel = load_mlmodel_once
        try:
            backend = AutoBackend(
                model=model_path,
                device=self.device,
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=False,
                verbose=False,
            )
        finally:
            ct.models.MLModel = original_mlmodel
        logger.info("Core ML detection compute units pinned to %s", unit_name)
        return backend

    def preprocess(self, imgs: list[ImageTensor]) -> torch.Tensor:
        imgs = [img if img.device.type == "cpu" else img.cpu() for img in imgs]
        return self._preprocess_cpu(imgs)

    def inference(self, image_batch: torch.Tensor):
        # The Core ML AutoBackend path only consumes the first image of a
        # batch, so run frames one at a time and re-batch the raw outputs.
        dets = []
        protos = []
        for i in range(image_batch.shape[0]):
            outputs = self.model(image_batch[i:i + 1], augment=False, visualize=False, embed=None)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            dets.append(outputs[0])
            if len(outputs) > 1:
                protos.append(outputs[1])
        det_batch = torch.cat(dets, dim=0)
        proto_batch = torch.cat(protos, dim=0) if protos else None
        return [(det_batch, proto_batch)]
