# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lada.utils import Box, Mask, box_utils
from lada.utils import image_utils

def get_box(mask: Mask) -> Box:
    points = cv2.findNonZero(mask)
    return box_utils.convert_from_opencv(cv2.boundingRect(points))

def morph(mask: Mask, iterations=1, operator=cv2.MORPH_DILATE) -> Mask:
    if get_mask_area(mask) < 0.01:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, operator, kernel, iterations=iterations)

def dilate_mask(mask: Mask, dilatation_size=11, iterations=2):
    if iterations == 0:
        return mask
    element = np.ones((dilatation_size, dilatation_size), np.uint8)
    mask_img = cv2.dilate(mask, element, iterations=iterations).reshape(mask.shape)
    return mask_img

def extend_mask(mask: Mask, value) -> Mask:
    # value between 0 and 3 -> higher values mean more extension of mask area. 0 does not change mask at all
    if value == 0:
        return mask

    # Dilations are slow when using huge kernels (which we would need for high-res masks). therefore we downscale mask to perform morph operations on much smaller pixel space with smaller kernels
    target_size = 256
    extended_mask = image_utils.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    extended_mask = morph(extended_mask, iterations=value, operator=cv2.MORPH_DILATE)
    extended_mask = image_utils.resize(extended_mask, mask.shape[:2], interpolation=cv2.INTER_NEAREST)
    extended_mask = extended_mask.reshape(mask.shape)
    assert mask.shape == extended_mask.shape
    return extended_mask

def clean_mask(mask: Mask, box: Box) -> tuple[Mask, Box]:
    t, l, b, r = box
    # Masks from YOLO prediction extend detection area in some cases. Let's crop
    mask[:t + 1, :, :] = 0
    mask[b:, :, :] = 0
    mask[:, :l + 1, :] = 0
    mask[:, r:, :] = 0

    # Mask from YOLO prediction can sometimes contain additional disconnected (tiny) segments. Keep only the largest
    edited_mask = np.zeros_like(mask, dtype=mask.dtype)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) != 0
    if len(contours) > 1:
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0]
    largest_contour = contours[0]
    cv_box = cv2.boundingRect(largest_contour)
    box = box_utils.convert_from_opencv(cv_box)
    cv2.drawContours(edited_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    return edited_mask, box

def get_mask_area(mask: Mask) -> float:
    pixels = cv2.countNonZero(mask)
    return pixels / (mask.shape[0] * mask.shape[1])

def smooth_mask(mask: Mask, kernel_size: int) -> Mask:
    return cv2.medianBlur(mask, kernel_size).reshape(mask.shape)


def stabilize_temporal_mask_tensor(
    masks: torch.Tensor,
    *,
    temporal_radius: int = 2,
    temporal_decay: float = 0.82,
    spatial_radius: int | None = None,
    feather_radius: int | None = None,
) -> torch.Tensor:
    """Stabilize a tracked mask sequence without resampling it.

    The direct mask remains fully active. Neighbouring masks are retained with
    a distance-dependent alpha so small detector contractions cannot switch a
    boundary abruptly between the original and restored images. A narrow
    spatial guard band covers segmentation uncertainty and a finite-support
    box filter supplies a soft compositing edge.

    Accepted layouts are ``[T,1,H,W]`` and ``[B,T,1,H,W]``. Floating input is
    returned in ``[0,1]``; integer input keeps its dtype and 0..255 range.
    """

    if masks.ndim not in (4, 5) or masks.shape[-3] != 1:
        raise ValueError("temporal masks must be [T,1,H,W] or [B,T,1,H,W]")
    if temporal_radius < 0:
        raise ValueError("temporal radius must be non-negative")
    if not 0.0 <= temporal_decay <= 1.0:
        raise ValueError("temporal decay must be in [0,1]")

    had_batch = masks.ndim == 5
    values = masks if had_batch else masks.unsqueeze(0)
    source_dtype = values.dtype
    source_is_float = source_dtype.is_floating_point
    values = values.float()
    if values.numel() and float(values.max()) > 1.0:
        values = values / 255.0
    values = values.clamp(0.0, 1.0)

    height, width = values.shape[-2:]
    scale_radius = max(1, int(round(min(height, width) / 128.0)))
    spatial_radius = scale_radius if spatial_radius is None else spatial_radius
    feather_radius = scale_radius if feather_radius is None else feather_radius
    if spatial_radius < 0 or feather_radius < 0:
        raise ValueError("mask radii must be non-negative")

    batch, frames = values.shape[:2]
    flat = values.reshape(batch * frames, 1, height, width)
    if spatial_radius:
        kernel = spatial_radius * 2 + 1
        flat = F.max_pool2d(flat, kernel, stride=1, padding=spatial_radius)
    guarded = flat.reshape(batch, frames, 1, height, width)

    stable_frames = []
    for index in range(frames):
        stable = guarded[:, index]
        first = max(0, index - temporal_radius)
        last = min(frames, index + temporal_radius + 1)
        for neighbour in range(first, last):
            distance = abs(neighbour - index)
            if distance:
                stable = torch.maximum(
                    stable,
                    guarded[:, neighbour] * (temporal_decay**distance),
                )
        stable_frames.append(stable)
    stable = torch.stack(stable_frames, dim=1)

    if feather_radius:
        flat = stable.reshape(batch * frames, 1, height, width)
        kernel = feather_radius * 2 + 1
        feathered = F.avg_pool2d(
            flat, kernel, stride=1, padding=feather_radius
        ).reshape_as(stable)
        stable = torch.maximum(stable, feathered)
    stable = stable.clamp(0.0, 1.0)

    if not source_is_float:
        stable = stable.mul(255.0).round().to(source_dtype)
    else:
        stable = stable.to(source_dtype)
    return stable if had_batch else stable.squeeze(0)


def stabilize_temporal_masks(
    masks: list[torch.Tensor],
    **kwargs,
) -> list[torch.Tensor]:
    """List adapter used by the restoration/compositing pipeline."""

    if not masks:
        return []
    shapes = {tuple(mask.shape) for mask in masks}
    if len(shapes) != 1:
        raise ValueError("all temporal masks must have the same shape")
    sample = masks[0]
    if sample.ndim == 2:
        stacked = torch.stack(masks, dim=0).unsqueeze(1)
        stabilized = stabilize_temporal_mask_tensor(stacked, **kwargs)[:, 0]
    elif sample.ndim == 3 and sample.shape[-1] == 1:
        stacked = torch.stack(masks, dim=0).permute(0, 3, 1, 2)
        stabilized = stabilize_temporal_mask_tensor(stacked, **kwargs).permute(
            0, 2, 3, 1
        )
    elif sample.ndim == 3 and sample.shape[0] == 1:
        stacked = torch.stack(masks, dim=0)
        stabilized = stabilize_temporal_mask_tensor(stacked, **kwargs)
    else:
        raise ValueError("mask list items must be HW, HW1 or 1HW tensors")
    return list(stabilized.unbind(0))


def create_blend_mask(crop_mask: torch.Tensor, feather_multiplier: float = 1.0):
    mask = crop_mask.squeeze().to(dtype=crop_mask.dtype)
    if mask.numel() and float(mask.max()) > 1.0:
        mask = mask / 255.0
    mask = mask.clamp(0.0, 1.0)
    h, w = mask.shape
    if feather_multiplier <= 0:
        return mask

    border_ratio = 0.05
    h_inner, w_inner = int(h * (1.0 - border_ratio)), int(w * (1.0 - border_ratio))
    h_outer, w_outer = h - h_inner, w - w_inner
    border_size = int(round(min(h_outer, w_outer) * feather_multiplier))
    if border_size < 5:
        return torch.ones_like(mask)
    blur_size = border_size
    if blur_size % 2 == 0:
        blur_size += 1
    inner = torch.ones((h_inner, w_inner), device=mask.device, dtype=mask.dtype)
    pad_top = h_outer // 2
    pad_bottom = h_outer - pad_top
    pad_left = w_outer // 2
    pad_right = w_outer - pad_left
    blend = F.pad(inner, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    blend = torch.maximum(mask, blend)
    kernel = torch.tensor(1.0 / (blur_size**2), device=blend.device, dtype=blend.dtype).expand(1, blur_size, blur_size)
    blend = image_utils.filter2D(blend.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    assert blend.shape == mask.shape
    return blend

def apply_random_mask_extensions(mask: Mask) -> Mask:
    value = np.random.choice([0, 0, 1, 1, 2])
    return extend_mask(mask, value)

def box_to_mask(box: Box, shape, mask_value: int):
    mask = np.zeros((shape[0], shape[1], 1), np.uint8)
    t, l, b, r = box
    mask[t:b + 1, l:r + 1] = mask_value
    return mask
