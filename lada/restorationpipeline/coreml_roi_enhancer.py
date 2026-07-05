# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core ML backend for ROI enhancer models (Real-ESRGAN, MewZoom, ...).

Runs an .mlpackage exported by scripts/apple/export_realesrgan_coreml.py
or scripts/apple/export_mewzoom_coreml.py on the Neural Engine via
coremltools. Exposes the same enhance() call shape as
realesrgan.RealESRGANer so frame_restorer can use any implementation
interchangeably. The export must carry lada.enhancer, lada.scale and
lada.imgsz metadata and use fixed-size image input/output.
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CoreMLROIEnhancer:
    # Composition can run without the MPS execution lock with this enhancer.
    uses_torch_device = False

    def __init__(self, model_path: str):
        import coremltools as ct
        from PIL import Image

        self._pil_image = Image
        unit_name = os.environ.get("LADA_COREML_COMPUTE_UNITS", "CPU_AND_NE").upper()
        unit = getattr(ct.ComputeUnit, unit_name, ct.ComputeUnit.CPU_AND_NE)
        self.model = ct.models.MLModel(str(model_path), compute_units=unit)
        metadata = dict(self.model.user_defined_metadata)
        if "lada.enhancer" not in metadata or "lada.scale" not in metadata or "lada.imgsz" not in metadata:
            raise ValueError(f"{model_path} is not a LADA ROI enhancer Core ML export")
        self.enhancer_name = metadata["lada.enhancer"]
        self.scale = int(metadata["lada.scale"])
        self.imgsz = int(metadata["lada.imgsz"])
        spec = self.model.get_spec()
        self._input_name = spec.description.input[0].name
        self._output_name = spec.description.output[0].name

    def enhance(self, img_bgr: np.ndarray, outscale: int | None = None):
        """
        Mirrors RealESRGANer.enhance: BGR uint8 in, upscaled BGR uint8 out.

        The Core ML graph has a fixed input size; the ROI is resized to it
        and the result is returned at input_size * scale. LADA resizes the
        result back onto the ROI afterwards, and restored content originates
        from LADA's own 256x256 crops, so the fixed-size hop loses nothing.
        """
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if (h, w) != (self.imgsz, self.imgsz):
            rgb = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA if max(h, w) > self.imgsz else cv2.INTER_CUBIC)
        enhanced = self.model.predict({self._input_name: self._pil_image.fromarray(rgb)})[self._output_name]
        enhanced = np.asarray(enhanced)[:, :, :3]
        target = (w * self.scale, h * self.scale)
        if enhanced.shape[:2] != (target[1], target[0]):
            enhanced = cv2.resize(enhanced, target, interpolation=cv2.INTER_AREA if enhanced.shape[0] > target[1] else cv2.INTER_CUBIC)
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR), None
