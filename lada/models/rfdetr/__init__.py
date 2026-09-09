# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from .rfdetr_coreai_segmentation_model import RFDETRCoreAISegmentationModel
from .rfdetr_coreml_segmentation_model import RFDETRCoreMLSegmentationModel

__all__ = [
    "RFDETRCoreAISegmentationModel",
    "RFDETRCoreMLSegmentationModel",
]
