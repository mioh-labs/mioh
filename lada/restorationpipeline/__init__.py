import logging
import re
from pathlib import Path

import torch

from lada import LOG_LEVEL, ModelFiles
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


def _coreai_frame_count(model_name: str, model_path: str) -> int:
    asset_name = Path(model_path).name.lower()
    for frame_count in (90, 36):
        suffix = f"-t{frame_count}"
        if model_name.endswith(suffix):
            return frame_count
        if re.search(rf"(?:^|[-_.])t{frame_count}(?:[-_.]|$)", asset_name):
            return frame_count
    return 18


def load_restoration_model(
    device: torch.device,
    model_name: str,
    model_path: str,
    config_path: str | None,
    fp16: bool,
):
    if model_path.endswith((".aimodel", ".aimodelc")):
        from lada.restorationpipeline.basicvsrpp_coreai_restorer import (
            CoreAIBasicvsrppMosaicRestorer,
        )

        frame_count = _coreai_frame_count(model_name, model_path)
        return CoreAIBasicvsrppMosaicRestorer(
            Path(model_path),
            frame_count=frame_count,
        ), "zero"
    if model_name.startswith("deepmosaics"):
        from lada.models.deepmosaics.models import loadmodel
        from lada.restorationpipeline.deepmosaics_mosaic_restorer import (
            DeepmosaicsMosaicRestorer,
        )

        model = loadmodel.video(device, model_path, fp16)
        return DeepmosaicsMosaicRestorer(model, device), "reflect"
    if model_name.startswith("basicvsrpp"):
        from lada.models.basicvsrpp.inference import load_model
        from lada.restorationpipeline.basicvsrpp_mosaic_restorer import (
            BasicvsrppMosaicRestorer,
        )

        model = load_model(config_path, model_path, device, fp16)
        return BasicvsrppMosaicRestorer(model, device, fp16), "zero"
    raise NotImplementedError()

def load_models(
    device: torch.device,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    mosaic_detection_model_path: str,
    fp16: bool,
    detect_face_mosaics: bool):
    mosaic_restoration_model, pad_mode = load_restoration_model(
        device,
        mosaic_restoration_model_name,
        mosaic_restoration_model_path,
        mosaic_restoration_config_path,
        fp16,
    )
    # setting classes=[0] will consider only detections of class id = 0 (nsfw mosaics) therefore filtering out sfw mosaics (heads, faces)
    if detect_face_mosaics:
        classes = [0]
        detection_model_name = ModelFiles.get_detection_model_by_path(mosaic_detection_model_path)
        if detection_model_name and detection_model_name == "v2":
            logger.info("Mosaic detection model v2 does not support detecting face mosaics. Use detection models v3 or newer. Ignoring...")
    else:
        classes = None
    if str(mosaic_detection_model_path).endswith((".aimodel", ".aimodelc")):
        from lada.models.yolo.yolo11_coreai_segmentation_model import Yolo11CoreAISegmentationModel
        mosaic_detection_model = Yolo11CoreAISegmentationModel(mosaic_detection_model_path, device, classes=classes, conf=0.15)
    elif str(mosaic_detection_model_path).endswith((".mlpackage", ".mlmodelc")):
        from lada.models.yolo.yolo11_coreml_segmentation_model import Yolo11CoreMLSegmentationModel
        mosaic_detection_model = Yolo11CoreMLSegmentationModel(mosaic_detection_model_path, device, classes=classes, conf=0.15)
    else:
        mosaic_detection_model = Yolo11SegmentationModel(mosaic_detection_model_path, device, classes=classes, conf=0.15, fp16=fp16)
    return mosaic_detection_model, mosaic_restoration_model, pad_mode
