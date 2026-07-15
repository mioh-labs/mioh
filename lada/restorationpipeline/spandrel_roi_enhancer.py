# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Spandrel-backed PyTorch ROI enhancer for the command-line pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class SpandrelROIEnhancer:
    """Expose a Spandrel image model through RealESRGANer's enhance API."""

    uses_torch_device = True
    prefer_pre_resize = True

    def __init__(self, model_path: str, device=None, fp16: bool = False, tile: int = 0):
        try:
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            raise RuntimeError(
                "Spandrel ROI enhancer support requires the roi-enhancer-spandrel extra "
                "(pip install 'lada[roi-enhancer-spandrel]')"
            ) from exc

        descriptor = ModelLoader().load_from_file(model_path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise ValueError(f"{model_path} is not a Spandrel image-to-image model")

        self.device = torch.device(device or "cpu")
        use_fp16 = fp16 and self.device.type != "cpu" and descriptor.supports_half
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.model = descriptor.to(device=self.device, dtype=self.dtype).eval()
        self.scale = int(descriptor.scale)
        self.tile = int(tile)
        if self.tile < 0:
            raise ValueError("Spandrel tile size must be 0 or greater")

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        if self.tile <= 0 or (image.shape[-2] <= self.tile and image.shape[-1] <= self.tile):
            return self.model(image)

        _, channels, height, width = image.shape
        scale = self.scale
        overlap = min(16, max(0, self.tile // 4))
        output = torch.empty(
            (1, channels, height * scale, width * scale),
            device=image.device,
            dtype=image.dtype,
        )
        for top in range(0, height, self.tile):
            bottom = min(top + self.tile, height)
            for left in range(0, width, self.tile):
                right = min(left + self.tile, width)
                padded_top = max(0, top - overlap)
                padded_bottom = min(height, bottom + overlap)
                padded_left = max(0, left - overlap)
                padded_right = min(width, right + overlap)
                tile = image[..., padded_top:padded_bottom, padded_left:padded_right]
                enhanced = self.model(tile)
                crop_top = (top - padded_top) * scale
                crop_bottom = crop_top + (bottom - top) * scale
                crop_left = (left - padded_left) * scale
                crop_right = crop_left + (right - left) * scale
                output[..., top * scale:bottom * scale, left * scale:right * scale] = enhanced[
                    ..., crop_top:crop_bottom, crop_left:crop_right
                ]
        return output

    def enhance(self, img_bgr: np.ndarray, outscale: int | None = None):
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Expected a BGR image with shape HxWx3, got {img_bgr.shape}")
        height, width = img_bgr.shape[:2]
        rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device=self.device, dtype=self.dtype).div_(255.0)
        with torch.inference_mode():
            enhanced = self._run(image)
        enhanced_rgb = (
            enhanced.squeeze(0)
            .permute(1, 2, 0)
            .float()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .byte()
            .cpu()
            .numpy()
        )
        target_scale = self.scale if outscale is None else int(outscale)
        if target_scale < 1:
            raise ValueError("Spandrel output scale must be 1 or greater")
        target = (width * target_scale, height * target_scale)
        if enhanced_rgb.shape[:2] != (target[1], target[0]):
            interpolation = cv2.INTER_AREA if self.scale > target_scale else cv2.INTER_CUBIC
            enhanced_rgb = cv2.resize(enhanced_rgb, target, interpolation=interpolation)
        return cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR), None
