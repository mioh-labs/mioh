# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Teacher utilities for the first two MiohRestorerV4 training stages.

The deployable V4 graph never contains any of the objects in this module.
Stage 1 uses frozen SPyNet flow and exact synthetic translations to supervise
V4's static shift banks.  Stage 2 uses the final quarter-resolution feature
of a nine-frame BasicVSR++ teacher through a small trainer-owned 1x1 adapter.

There is deliberately no DCN-offset or RGB-output distillation here.  The
former is not an appropriate target for V4's correlation hierarchy and the
latter would make BasicVSR++'s image characteristics a quality ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


HIER27_EIGHTH_COARSE = tuple(
    (vertical * 3, horizontal * 3)
    for vertical in (-1, 0, 1)
    for horizontal in (-1, 0, 1)
)
HIER27_EIGHTH_MIDDLE = tuple(
    (vertical, horizontal)
    for vertical in (-1, 0, 1)
    for horizontal in (-1, 0, 1)
)
HIER27_QUARTER_FINE = tuple(
    (vertical * 2, horizontal * 2)
    for vertical in (-1, 0, 1)
    for horizontal in (-1, 0, 1)
)
V4_ALIGNMENT_PAIRS = tuple(
    (reference, target)
    for reference in range(2, 7)
    for target in range(reference - 2, reference + 3)
    if target != reference
)


class _SPyNetLike(Protocol):
    def __call__(
        self, reference: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor: ...


def _validate_flow(flow: torch.Tensor) -> None:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [N,2,H,W]")
    if flow.shape[-2] % 2 or flow.shape[-1] % 2:
        raise ValueError("quarter-resolution flow dimensions must be even")


def _shift_distribution(
    displacement_yx: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("shift temperature must be positive")
    if displacement_yx.ndim != 4 or displacement_yx.shape[1] != 2:
        raise ValueError("displacement must have shape [N,2,H,W]")
    offset_tensor = displacement_yx.new_tensor(offsets).reshape(
        1, len(offsets), 2, 1, 1
    )
    squared_distance = (
        displacement_yx.unsqueeze(1) - offset_tensor
    ).square().sum(dim=2)
    distribution = torch.softmax(-squared_distance / temperature, dim=1)
    expected = (distribution.unsqueeze(2) * offset_tensor).sum(dim=1)
    return distribution, expected


def dense_flow_to_hier27_distributions(
    flow_quarter: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project dense SPyNet flow onto V4's three hier27 shift banks.

    ``flow_quarter`` is SPyNet's ``(x, y)`` sampling displacement in
    quarter-resolution pixels.  V4's padded slice shifts move *content*, so
    their direction is the negative of the sampling displacement.  Coarse and
    middle targets live on the 1/8 grid; the final residual is supervised on
    the 1/4 grid.

    Returns tensors shaped ``[N,9,H/2,W/2]``, ``[N,9,H/2,W/2]`` and
    ``[N,9,H,W]`` in exactly the candidate order used by
    :class:`HierarchicalAlignment27`.
    """

    _validate_flow(flow_quarter)
    flow = flow_quarter.float()
    # SPyNet channels are x,y.  V4 offsets are vertical,horizontal and have
    # the opposite sign because shift2d translates content rather than sample
    # coordinates.
    displacement_quarter = torch.stack((-flow[:, 1], -flow[:, 0]), dim=1)
    eighth_size = (flow.shape[-2] // 2, flow.shape[-1] // 2)
    displacement_eighth = F.interpolate(
        displacement_quarter,
        size=eighth_size,
        mode="bilinear",
        align_corners=False,
    ) * 0.5

    coarse, expected_coarse = _shift_distribution(
        displacement_eighth,
        HIER27_EIGHTH_COARSE,
        temperature=temperature,
    )
    middle_residual = displacement_eighth - expected_coarse
    middle, expected_middle = _shift_distribution(
        middle_residual,
        HIER27_EIGHTH_MIDDLE,
        temperature=temperature,
    )

    represented_eighth = expected_coarse + expected_middle
    represented_quarter = F.interpolate(
        represented_eighth,
        size=flow.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ) * 2.0
    fine_residual = displacement_quarter - represented_quarter
    fine, _expected_fine = _shift_distribution(
        fine_residual,
        HIER27_QUARTER_FINE,
        temperature=temperature,
    )
    return (
        coarse.to(dtype=flow_quarter.dtype),
        middle.to(dtype=flow_quarter.dtype),
        fine.to(dtype=flow_quarter.dtype),
    )


def exact_motion_to_hier27_distributions(
    displacement_input_yx: torch.Tensor,
    *,
    eighth_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode exact 8-pixel-quantized translations as hard V4 bank targets.

    Args:
        displacement_input_yx: ``[N,2]`` content shifts in input pixels.
        eighth_size: Spatial size of the two 1/8 alignment banks.

    The canonical decomposition is coarse ``{-3,0,3}``, then middle
    ``{-1,0,1}`` on the 1/8 grid, followed by the remaining ``{-2,0,2}``
    shift on the 1/4 grid.  It represents every quantized displacement in
    V4's +/-40-pixel range exactly and avoids using a teacher approximation
    for the synthetic-motion branch.
    """

    if displacement_input_yx.ndim != 2 or displacement_input_yx.shape[1] != 2:
        raise ValueError("exact displacement must have shape [N,2]")
    if len(eighth_size) != 2 or min(eighth_size) <= 0:
        raise ValueError("eighth_size must contain two positive dimensions")
    rounded = displacement_input_yx.round().to(dtype=torch.int64)
    if not torch.equal(
        displacement_input_yx,
        rounded.to(dtype=displacement_input_yx.dtype),
    ):
        raise ValueError("exact displacement must contain integer pixels")
    if not torch.equal(rounded.remainder(8), torch.zeros_like(rounded)):
        raise ValueError("exact displacement must be quantized to eight pixels")
    eighth = rounded // 8
    if int(eighth.abs().max()) > 5:
        raise ValueError("exact displacement exceeds V4's +/-40-pixel reach")

    coarse = torch.where(
        eighth.abs() >= 3,
        eighth.sign() * 3,
        torch.zeros_like(eighth),
    )
    remainder = eighth - coarse
    middle = remainder.clamp(-1, 1)
    fine_quarter = (remainder - middle) * 2

    def indices(
        offsets: Sequence[tuple[int, int]], values: torch.Tensor
    ) -> torch.Tensor:
        table = values.new_tensor(offsets)
        matches = (values[:, None, :] == table[None, :, :]).all(dim=2)
        if not bool(matches.any(dim=1).all()):
            raise ValueError("exact displacement cannot be represented by hier27")
        return matches.to(dtype=torch.int64).argmax(dim=1)

    coarse_index = indices(HIER27_EIGHTH_COARSE, coarse)
    middle_index = indices(HIER27_EIGHTH_MIDDLE, middle)
    fine_index = indices(HIER27_QUARTER_FINE, fine_quarter)
    height, width = eighth_size

    def one_hot(index: torch.Tensor, out_height: int, out_width: int) -> torch.Tensor:
        result = F.one_hot(index, num_classes=9).to(
            device=displacement_input_yx.device,
            dtype=displacement_input_yx.dtype,
        )
        return result[:, :, None, None].expand(-1, -1, out_height, out_width)

    return (
        one_hot(coarse_index, height, width),
        one_hot(middle_index, height, width),
        one_hot(fine_index, height * 2, width * 2),
    )


def roi_shift_kl_loss(
    student_distribution: torch.Tensor,
    teacher_distribution: torch.Tensor,
    roi_mask: torch.Tensor,
    *,
    context_radius: int = 2,
) -> torch.Tensor:
    """KL(target || student) for V4's four-dimensional shift volumes."""

    if student_distribution.ndim != 4:
        raise ValueError("student shift distribution must be [N,K,H,W]")
    if student_distribution.shape != teacher_distribution.shape:
        raise ValueError("student and teacher shift distributions must match")
    if roi_mask.ndim != 4 or roi_mask.shape[:2] != (
        student_distribution.shape[0],
        1,
    ):
        raise ValueError("ROI mask must have shape [N,1,H,W]")
    if context_radius < 0:
        raise ValueError("context radius cannot be negative")
    roi = F.interpolate(
        roi_mask.float(),
        size=student_distribution.shape[-2:],
        mode="nearest",
    )
    if context_radius:
        kernel = context_radius * 2 + 1
        roi = F.max_pool2d(
            roi, kernel_size=kernel, stride=1, padding=context_radius
        )
    target = teacher_distribution.float().clamp_min(1e-6)
    student = student_distribution.float().clamp_min(1e-6)
    divergence = (target * (target.log() - student.log())).sum(
        dim=1, keepdim=True
    )
    return (divergence * roi).sum() / roi.sum().clamp_min(1.0)


class V4FeatureDistillationAdapter(nn.Module):
    """Trainer-only 1x1 projection from V4 fusion to teacher channels."""

    def __init__(self, student_channels: int = 96, teacher_channels: int = 64):
        super().__init__()
        if student_channels <= 0 or teacher_channels <= 0:
            raise ValueError("feature channel counts must be positive")
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.projection = nn.Conv2d(
            student_channels, teacher_channels, kernel_size=1, bias=False
        )

    def forward(self, student: torch.Tensor) -> torch.Tensor:
        if student.ndim not in (4, 5):
            raise ValueError("student feature must be NCHW or BTCHW")
        if student.shape[-3] != self.student_channels:
            raise ValueError("unexpected student feature channel count")
        if student.ndim == 4:
            return self.projection(student)
        batch, frames, channels, height, width = student.shape
        projected = self.projection(
            student.reshape(batch * frames, channels, height, width)
        )
        return projected.reshape(
            batch, frames, self.teacher_channels, height, width
        )


def projected_roi_feature_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    roi_mask: torch.Tensor,
    adapter: V4FeatureDistillationAdapter,
    *,
    context_radius: int = 2,
) -> torch.Tensor:
    """Compare normalized projected V4 features with frozen teacher features."""

    if student.ndim != 5 or teacher.ndim != 5:
        raise ValueError("feature tensors must have shape [B,T,C,H,W]")
    if student.shape[:2] != teacher.shape[:2] or student.shape[-2:] != teacher.shape[-2:]:
        raise ValueError("student and teacher feature geometry must match")
    if roi_mask.ndim != 5 or roi_mask.shape[:2] != student.shape[:2] or roi_mask.shape[2] != 1:
        raise ValueError("ROI mask must have shape [B,T,1,H,W]")
    if teacher.shape[2] != adapter.teacher_channels:
        raise ValueError("unexpected teacher feature channel count")
    if context_radius < 0:
        raise ValueError("context radius cannot be negative")

    projected = adapter(student).float()
    target = teacher.detach().float()
    projected = F.normalize(projected, p=2, dim=2, eps=1e-6)
    target = F.normalize(target, p=2, dim=2, eps=1e-6)
    batch, frames, _, height, width = projected.shape
    roi = F.interpolate(
        roi_mask.float().reshape(batch * frames, 1, *roi_mask.shape[-2:]),
        size=(height, width),
        mode="nearest",
    )
    if context_radius:
        kernel = context_radius * 2 + 1
        roi = F.max_pool2d(
            roi, kernel_size=kernel, stride=1, padding=context_radius
        )
    roi = roi.reshape(batch, frames, 1, height, width)
    difference = F.smooth_l1_loss(projected, target, reduction="none").mean(
        dim=2, keepdim=True
    )
    return (difference * roi).sum() / roi.sum().clamp_min(1.0)


@torch.no_grad()
def extract_basicvsrpp_reconstruction_features(
    teacher: nn.Module,
    frames: torch.Tensor,
    *,
    output_indices: Sequence[int] = (2, 3, 4, 5, 6),
) -> torch.Tensor:
    """Run a BasicVSR++ teacher on exactly one window, stopping before RGB heads.

    The returned ``[B,len(output_indices),64,H/4,W/4]`` tensors are the output
    of BasicVSR++'s reconstruction fusion block.  They contain all four
    propagation branches but avoid all full-resolution upsampling and RGB
    image computation.
    """

    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("teacher frames must have shape [B,T,3,H,W]")
    batch, frame_count, channels, height, width = frames.shape
    indices = tuple(int(index) for index in output_indices)
    if not indices or any(index < 0 or index >= frame_count for index in indices):
        raise ValueError("teacher output index is outside the input window")
    if height % 4 or width % 4:
        raise ValueError("teacher frame dimensions must be divisible by four")
    quarter_height, quarter_width = height // 4, width // 4
    if quarter_height < 64 or quarter_width < 64:
        raise ValueError("BasicVSR++ quarter-resolution inputs must be at least 64x64")

    flat = frames.reshape(batch * frame_count, channels, height, width)
    low_resolution = F.interpolate(
        flat, scale_factor=0.25, mode="bicubic"
    ).reshape(
        batch, frame_count, channels, quarter_height, quarter_width
    )
    spatial = teacher.feat_extract(flat)
    spatial = spatial.reshape(
        batch, frame_count, spatial.shape[1], spatial.shape[-2], spatial.shape[-1]
    )
    features: dict[str, list[torch.Tensor]] = {
        "spatial": [spatial[:, index] for index in range(frame_count)]
    }
    flows_forward, flows_backward = teacher.compute_flow(low_resolution)
    for iteration in (1, 2):
        for direction in ("backward", "forward"):
            branch = f"{direction}_{iteration}"
            features[branch] = []
            flows = flows_backward if direction == "backward" else flows_forward
            features = teacher.propagate(features, flows, branch)

    branch_names = tuple(name for name in features if name != "spatial")
    outputs: list[torch.Tensor] = []
    for index in indices:
        fused = torch.cat(
            [features["spatial"][index]]
            + [features[name][index] for name in branch_names],
            dim=1,
        )
        outputs.append(teacher.reconstruction(fused))
    return torch.stack(outputs, dim=1)


def load_spynet_teacher(
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    fp16: bool = False,
) -> nn.Module:
    """Load only SPyNet from a LADA BasicVSR++ generator checkpoint."""

    from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import SPyNet

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("unexpected BasicVSR++ checkpoint")
    prefixes = (
        "generator_ema.spynet.",
        "generator.spynet.",
        "spynet.",
    )
    state: dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        state = {
            str(key)[len(prefix) :]: value
            for key, value in checkpoint.items()
            if str(key).startswith(prefix)
        }
        if state:
            break
    if not state and isinstance(checkpoint.get("state_dict"), dict):
        nested = checkpoint["state_dict"]
        for prefix in prefixes:
            state = {
                str(key)[len(prefix) :]: value
                for key, value in nested.items()
                if str(key).startswith(prefix)
            }
            if state:
                break
    if not state:
        raise ValueError("SPyNet weights were not found in the checkpoint")
    model = SPyNet(pretrained=None)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(torch.device(device))
    if fp16:
        if torch.device(device).type == "cpu":
            raise ValueError("fp16 SPyNet requires an accelerator device")
        model.half()
    return model


def load_basicvsrpp_feature_teacher(
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    fp16: bool = False,
) -> nn.Module:
    """Load the frozen nine-frame BasicVSR++ Stage-2 feature teacher.

    Release checkpoints normally store flattened ``generator_ema.*`` keys.
    Training/export checkpoints may instead put prefixed or already stripped
    generator weights below ``state_dict``.  All of those forms are accepted;
    optimizer and discriminator payloads are never considered.
    """

    from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("unexpected BasicVSR++ checkpoint")
    candidates: list[dict] = [checkpoint]
    nested = checkpoint.get("state_dict")
    if isinstance(nested, dict):
        candidates.insert(0, nested)
    state: dict[str, torch.Tensor] = {}
    for candidate in candidates:
        for prefix in ("generator_ema.", "generator."):
            state = {
                str(key)[len(prefix) :]: value
                for key, value in candidate.items()
                if str(key).startswith(prefix)
            }
            if state:
                break
        if state:
            break
        # A stripped generator state starts with the model's own module names.
        if (
            any(str(key).startswith("spynet.") for key in candidate)
            and any(str(key).startswith("feat_extract.") for key in candidate)
            and all(isinstance(value, torch.Tensor) for value in candidate.values())
        ):
            state = {str(key): value for key, value in candidate.items()}
            break
    if not state:
        raise ValueError("BasicVSR++ generator weights were not found")

    teacher = BasicVSRPlusPlusGanNet(
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,
    )
    teacher.load_state_dict(state, strict=True)
    teacher.requires_grad_(False).eval().to(torch.device(device))
    if fp16:
        if torch.device(device).type == "cpu":
            raise ValueError("fp16 BasicVSR++ teacher requires an accelerator device")
        teacher.half()
    return teacher


@torch.no_grad()
def compute_spynet_pair_flows(
    spynet: _SPyNetLike,
    frames: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    *,
    chunk_size: int = 0,
) -> torch.Tensor:
    """Compute direct ref-to-target quarter-grid flow for arbitrary frame pairs.

    Returns ``[B,P,2,H/4,W/4]``.  Pair order is preserved, which allows the
    caller to match an output-major V4 alignment diagnostics list exactly.
    """

    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("SPyNet frames must have shape [B,T,3,H,W]")
    batch, frame_count, channels, height, width = frames.shape
    pair_list = tuple((int(reference), int(target)) for reference, target in pairs)
    if not pair_list:
        raise ValueError("at least one SPyNet frame pair is required")
    if any(
        reference < 0
        or target < 0
        or reference >= frame_count
        or target >= frame_count
        or reference == target
        for reference, target in pair_list
    ):
        raise ValueError("invalid SPyNet frame pair")
    if height % 4 or width % 4:
        raise ValueError("SPyNet frame dimensions must be divisible by four")
    low = F.interpolate(
        frames.reshape(batch * frame_count, channels, height, width),
        scale_factor=0.25,
        mode="bicubic",
    ).reshape(batch, frame_count, channels, height // 4, width // 4)
    references = torch.cat(
        [low[:, reference] for reference, _target in pair_list], dim=0
    )
    supports = torch.cat(
        [low[:, target] for _reference, target in pair_list], dim=0
    )
    total = references.shape[0]
    chunk = total if chunk_size <= 0 else chunk_size
    flows = [
        spynet(references[start : start + chunk], supports[start : start + chunk])
        for start in range(0, total, chunk)
    ]
    flow = torch.cat(flows, dim=0)
    return flow.reshape(
        len(pair_list), batch, 2, height // 4, width // 4
    ).transpose(0, 1)
