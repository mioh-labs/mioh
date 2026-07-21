#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Train the ANE-first, fixed-window MiohRestorer V4-Q model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lada.models.mioh_restorer.curriculum_v4 import (
    effective_loss_weights,
    schedule_record,
    stage_definition,
    stage_learning_rate,
)
from lada.models.mioh_restorer.distillation_v4 import (
    V4_ALIGNMENT_PAIRS,
    V4FeatureDistillationAdapter,
    compute_spynet_pair_flows,
    dense_flow_to_hier27_distributions,
    exact_motion_to_hier27_distributions,
    extract_basicvsrpp_reconstruction_features,
    load_basicvsrpp_feature_teacher,
    load_spynet_teacher,
    projected_roi_feature_loss,
    roi_shift_kl_loss,
)
from lada.models.mioh_restorer.losses_v4 import (
    MiohRestorerV4Loss,
    confidence_error_correlation,
)
from lada.models.mioh_restorer.model_v4 import MiohRestorerV4Q, parameter_count
from lada.models.mioh_restorer.synthetic_motion_v4 import (
    alignment_displacement,
    make_synthetic_motion_sequence,
    shift_without_wrap,
)
from lada.models.mioh_restorer.training import MaskedVGG16PerceptualLoss
from lada.models.mioh_restorer.training_dataset import MiohRestorationDataset
from lada.models.mioh_restorer.training_memory import (
    MemoryThresholds,
    TrainingMemoryGuard,
    release_device_memory,
    tree_to_cpu,
)


WINDOW_FRAMES = 9
DATASET_FRAMES = 9
OUTPUT_SLICE = slice(2, 7)
MEMORY_EXIT_CODE = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata-root", type=Path, nargs="+", required=True)
    parser.add_argument("--val-metadata-root", type=Path, nargs="+")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--initialize-weights", choices=("ema", "raw"), default="ema"
    )
    parser.add_argument("--stage", choices=("1", "2", "3", "4", "5"), default="1")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--teacher-fp16", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--spynet-alignment-weight", type=float)
    parser.add_argument("--exact-motion-alignment-weight", type=float)
    parser.add_argument("--exact-motion-probability", type=float)
    parser.add_argument("--feature-distillation-weight", type=float)
    parser.add_argument("--teacher-shift-temperature", type=float, default=0.5)
    parser.add_argument("--teacher-flow-chunk-size", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--alignment-variant", choices=("hier27",), default="hier27")
    parser.add_argument("--execution-mode", choices=("batch", "serial"), default="batch")
    parser.add_argument("--quarter-channels", type=int, default=64)
    parser.add_argument("--eighth-channels", type=int, default=96)
    parser.add_argument("--fusion-eighth-channels", type=int, default=192)
    parser.add_argument("--fusion-quarter-channels", type=int, default=96)
    parser.add_argument("--eighth-blocks", type=int, default=10)
    parser.add_argument("--quarter-blocks", type=int, default=4)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--stage-transition-steps", type=int, default=500)
    parser.add_argument("--candidate-weight", type=float, default=0.5)
    parser.add_argument("--high-frequency-weight", type=float, default=0.1)
    parser.add_argument("--base-weight", type=float, default=0.05)
    parser.add_argument("--confidence-weight", type=float, default=0.2)
    parser.add_argument("--confidence-regularization-weight", type=float, default=1e-3)
    parser.add_argument("--confidence-scale", type=float, default=0.05)
    parser.add_argument("--temporal-weight", type=float, default=0.2)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--structural-weight", type=float, default=0.0)
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--perceptual-image-size", type=int, default=224)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-latest-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--memory-warning-ratio", type=float, default=0.80)
    parser.add_argument("--memory-critical-ratio", type=float, default=0.92)
    parser.add_argument("--memory-warning-available-gib", type=float, default=8.0)
    parser.add_argument("--memory-critical-available-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degrade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--horizontal-flip", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    args = parser.parse_args()
    if args.steps is None:
        args.steps = stage_definition(args.stage).default_steps
    return args


def validate_args(args: argparse.Namespace) -> None:
    stage = stage_definition(args.stage)
    if args.resume and args.initialize_from:
        raise ValueError("resume and initialize-from cannot be used together")
    if stage.stage_id == 1 and args.initialize_from:
        raise ValueError("stage 1 must start from newly initialized weights")
    if stage.stage_id > 1 and not (args.resume or args.initialize_from):
        raise ValueError(
            f"stage {stage.stage_id} requires --initialize-from or --resume"
        )
    if stage.stage_id in (1, 2) and args.teacher_checkpoint is None:
        raise ValueError(f"stage {stage.stage_id} requires --teacher-checkpoint")
    if stage.stage_id >= 3 and args.teacher_checkpoint is not None:
        raise ValueError("stages 3-5 must not load a BasicVSR++ teacher")
    if args.teacher_checkpoint is not None and not args.teacher_checkpoint.is_file():
        raise ValueError(f"teacher checkpoint not found: {args.teacher_checkpoint}")
    positive = (
        "steps",
        "image_size",
        "batch_size",
        "prefetch_factor",
        "quarter_channels",
        "eighth_channels",
        "fusion_eighth_channels",
        "fusion_quarter_channels",
        "learning_rate",
        "minimum_learning_rate",
        "max_grad_norm",
        "save_every",
        "save_latest_every",
        "validate_every",
        "validation_batches",
        "log_every",
        "perceptual_image_size",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.image_size % 8:
        raise ValueError("image-size must be divisible by eight")
    if (
        args.workers < 0
        or args.validation_workers < 0
        or args.warmup_steps < 0
        or args.stage_transition_steps < 0
    ):
        raise ValueError("workers and warmup-steps must not be negative")
    if args.minimum_learning_rate > args.learning_rate:
        raise ValueError("minimum-learning-rate must not exceed learning-rate")
    if not 0 < args.ema_decay < 1:
        raise ValueError("ema-decay must be between zero and one")
    distillation_values = (
        args.spynet_alignment_weight,
        args.exact_motion_alignment_weight,
        args.feature_distillation_weight,
    )
    if any(value is not None and value < 0 for value in distillation_values):
        raise ValueError("distillation weights must not be negative")
    if (
        args.exact_motion_probability is not None
        and not 0 <= args.exact_motion_probability <= 1
    ):
        raise ValueError("exact-motion-probability must be between zero and one")
    if args.teacher_shift_temperature <= 0:
        raise ValueError("teacher-shift-temperature must be positive")
    if args.teacher_flow_chunk_size < 0:
        raise ValueError("teacher-flow-chunk-size must not be negative")
    MemoryThresholds(
        warning_mps_ratio=args.memory_warning_ratio,
        critical_mps_ratio=args.memory_critical_ratio,
        warning_system_available_gib=args.memory_warning_available_gib,
        critical_system_available_gib=args.memory_critical_available_gib,
    )


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    roots: list[Path], args: argparse.Namespace, *, training: bool
) -> DataLoader:
    workers = args.workers if training else args.validation_workers
    dataset = MiohRestorationDataset(
        roots,
        sequence_frames=DATASET_FRAMES,
        image_size=args.image_size,
        degrade=args.degrade,
        horizontal_flip=args.horizontal_flip if training else False,
        deterministic=not training,
        degradation_mix=training,
        time_reverse=training,
        limit=args.limit_train_samples if training else args.limit_val_samples,
    )
    options: dict[str, object] = {
        "batch_size": args.batch_size if training else 1,
        "shuffle": training,
        "num_workers": workers,
        "persistent_workers": training and workers > 0,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": training and len(dataset) >= args.batch_size,
    }
    if workers:
        options["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **options)


def build_model(args: argparse.Namespace) -> MiohRestorerV4Q:
    model = MiohRestorerV4Q(
        alignment_variant=args.alignment_variant,
        execution_mode=args.execution_mode,
        quarter_channels=args.quarter_channels,
        eighth_channels=args.eighth_channels,
        fusion_eighth_channels=args.fusion_eighth_channels,
        fusion_quarter_channels=args.fusion_quarter_channels,
        eighth_blocks=args.eighth_blocks,
        quarter_blocks=args.quarter_blocks,
    )
    model.enable_gradient_checkpointing(args.gradient_checkpointing)
    return model


def distillation_settings(args: argparse.Namespace) -> dict[str, float]:
    stage = stage_definition(args.stage)

    def selected(name: str, default: float) -> float:
        override = getattr(args, name)
        return default if override is None else float(override)

    return {
        "spynet_alignment": selected(
            "spynet_alignment_weight", stage.spynet_alignment_weight
        ),
        "exact_motion_alignment": selected(
            "exact_motion_alignment_weight",
            stage.exact_motion_alignment_weight,
        ),
        "exact_motion_probability": selected(
            "exact_motion_probability", stage.exact_motion_probability
        ),
        "feature": selected(
            "feature_distillation_weight", stage.feature_distillation_weight
        ),
    }


def model_config(model: MiohRestorerV4Q, args: argparse.Namespace) -> dict[str, object]:
    return {
        "version": 4,
        "architecture": "v4q-output-specific-hierarchical-correlation",
        "architecture_revision": model.ARCHITECTURE_REVISION,
        "alignment_variant": model.alignment_variant,
        "execution_mode": model.execution_mode,
        "input_frames": WINDOW_FRAMES,
        "output_frames": 5,
        "inference_stride": 4,
        "image_size": args.image_size,
        "quarter_channels": model.quarter_channels,
        "eighth_channels": model.eighth_channels,
        "fusion_eighth_channels": model.fusion_eighth_channels,
        "fusion_quarter_channels": model.fusion_quarter_channels,
        "eighth_blocks": model.eighth_blocks,
        "quarter_blocks": model.quarter_blocks,
        "gradient_checkpointing": model.gradient_checkpointing,
        "loss": {
            "candidate": args.candidate_weight,
            "high_frequency": args.high_frequency_weight,
            "base": args.base_weight,
            "confidence": args.confidence_weight,
            "confidence_regularization": args.confidence_regularization_weight,
            "confidence_scale": args.confidence_scale,
            "temporal": args.temporal_weight,
            "gradient": args.gradient_weight,
            "structural": args.structural_weight,
            "perceptual": args.perceptual_weight,
        },
        "training_plan": "independent_stages",
        "training_stage": stage_definition(args.stage).name,
        "training_stage_id": stage_definition(args.stage).stage_id,
        "training_stage_steps": args.steps,
        "stage_transition_steps": args.stage_transition_steps,
        "stages": schedule_record(),
        "distillation": {
            **distillation_settings(args),
            "teacher_checkpoint": (
                str(args.teacher_checkpoint.resolve())
                if args.teacher_checkpoint is not None
                else None
            ),
            "teacher_checkpoint_sha256": getattr(
                args, "teacher_checkpoint_sha256", None
            ),
            "teacher_fp16": args.teacher_fp16,
            "teacher_shift_temperature": args.teacher_shift_temperature,
            "teacher_flow_chunk_size": args.teacher_flow_chunk_size,
            "same_nine_frame_window": True,
            "ground_truth_is_primary": True,
            "pixel_output_distillation": False,
            "dcn_offset_distillation": False,
        },
    }


def learning_rate(args: argparse.Namespace, step: int) -> float:
    return stage_learning_rate(
        stage_definition(args.stage),
        step,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def set_module_trainable(module: torch.nn.Module, enabled: bool) -> None:
    module.requires_grad_(enabled)


def apply_training_stage(model: MiohRestorerV4Q, args: argparse.Namespace) -> str:
    """Freeze unreliable heads until the reconstruction candidate is useful."""

    stage = stage_definition(args.stage)
    set_module_trainable(model.up_half_to_full, stage.train_texture)
    set_module_trainable(model.texture_head, stage.train_texture)
    set_module_trainable(model.confidence_head, stage.train_confidence)
    return stage.name


def apply_loss_weights(
    criterion: MiohRestorerV4Loss,
    args: argparse.Namespace,
    step: int,
) -> dict[str, float]:
    weights = effective_loss_weights(
        stage_definition(args.stage),
        step,
        transition_steps=args.stage_transition_steps,
    )
    values = {
        "candidate": weights.candidate,
        "high_frequency": weights.high_frequency,
        "base": weights.base,
        "confidence": weights.confidence,
        "confidence_regularization": weights.confidence_regularization,
        "temporal": weights.temporal,
        "temporal_acceleration": weights.temporal_acceleration,
        "gradient": weights.gradient,
        "structural": weights.structural,
        "perceptual": weights.perceptual,
    }
    criterion.candidate_weight = values["candidate"]
    criterion.high_frequency_weight = values["high_frequency"]
    criterion.base_weight = values["base"]
    criterion.confidence_weight = values["confidence"]
    criterion.confidence_regularization_weight = values[
        "confidence_regularization"
    ]
    criterion.temporal_weight = values["temporal"]
    criterion.temporal_acceleration_weight = values["temporal_acceleration"]
    criterion.gradient_weight = values["gradient"]
    criterion.structural_weight = values["structural"]
    return values


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_value, value in zip(ema.parameters(), model.parameters(), strict=True):
        ema_value.lerp_(value.detach(), 1 - decay)
    for ema_value, value in zip(ema.buffers(), model.buffers(), strict=True):
        ema_value.copy_(value)


def move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def window_tensors(
    batch: dict[str, torch.Tensor | list[str]], start: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = batch["inputs"]
    targets = batch["targets"]
    masks = batch["masks"]
    if not all(isinstance(item, torch.Tensor) for item in (inputs, targets, masks)):
        raise TypeError("dataset batch does not contain tensors")
    inputs = inputs[:, start : start + WINDOW_FRAMES].to(device)
    targets = targets[:, start + 2 : start + 7].to(device)
    masks = masks[:, start : start + WINDOW_FRAMES].to(device)
    output_masks = masks[:, OUTPUT_SLICE]
    values = torch.cat((inputs, masks), dim=2)
    output_inputs = inputs[:, OUTPUT_SLICE]
    return values, targets, output_inputs, output_masks


def forward_loss(
    model: MiohRestorerV4Q,
    criterion: MiohRestorerV4Loss,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    capture_alignment: bool = False,
    capture_features: bool = False,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    values, targets, output_inputs, output_masks = tensors
    if capture_alignment or capture_features:
        restored, confidence, base, texture, diagnostics = (
            model.forward_with_distillation(
                values,
                capture_alignment=capture_alignment,
                capture_features=capture_features,
            )
        )
    else:
        restored, confidence, base, texture = model.forward_components(values)
        diagnostics = {}
    total, stats = criterion(
        restored,
        confidence,
        base,
        texture,
        targets,
        output_inputs,
        output_masks,
    )
    return total, stats, restored, confidence, output_masks, diagnostics


def alignment_pair_masks(values: torch.Tensor) -> torch.Tensor:
    """Return output-major union masks matching ``V4_ALIGNMENT_PAIRS``."""

    masks = values[:, :, 3:4]
    return torch.stack(
        [torch.maximum(masks[:, reference], masks[:, target])
         for reference, target in V4_ALIGNMENT_PAIRS],
        dim=1,
    )


def _flatten_alignment_bank(bank: torch.Tensor) -> torch.Tensor:
    if bank.ndim != 6 or bank.shape[1:4] != (5, 4, 9):
        raise ValueError("unexpected V4 alignment diagnostic shape")
    return bank.reshape(-1, 9, *bank.shape[-2:])


def alignment_kl_from_targets(
    diagnostics: dict[str, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pair_masks: torch.Tensor,
) -> torch.Tensor:
    if pair_masks.ndim != 5 or pair_masks.shape[1] != len(V4_ALIGNMENT_PAIRS):
        raise ValueError("pair masks must have shape [B,20,1,H,W]")
    flattened_masks = pair_masks.reshape(-1, 1, *pair_masks.shape[-2:])
    losses = []
    for name, target in zip(
        ("alignment_coarse", "alignment_middle", "alignment_fine"),
        targets,
        strict=True,
    ):
        student = _flatten_alignment_bank(diagnostics[name])
        teacher = target.reshape(-1, 9, *target.shape[-2:])
        losses.append(roi_shift_kl_loss(student, teacher, flattened_masks))
    return torch.stack(losses).mean()


def spynet_alignment_loss(
    diagnostics: dict[str, torch.Tensor],
    values: torch.Tensor,
    spynet: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.Tensor:
    teacher_dtype = next(spynet.parameters()).dtype
    flows = compute_spynet_pair_flows(
        spynet,
        values[:, :, :3].to(dtype=teacher_dtype),
        V4_ALIGNMENT_PAIRS,
        chunk_size=args.teacher_flow_chunk_size,
    ).to(dtype=values.dtype)
    batch, pairs = flows.shape[:2]
    targets_flat = dense_flow_to_hier27_distributions(
        flows.reshape(batch * pairs, 2, *flows.shape[-2:]),
        temperature=args.teacher_shift_temperature,
    )
    targets = tuple(
        item.reshape(batch, pairs, 9, *item.shape[-2:])
        for item in targets_flat
    )
    return alignment_kl_from_targets(
        diagnostics, targets, alignment_pair_masks(values)
    )


def _shift_batch(
    values: torch.Tensor, displacement_yx: torch.Tensor
) -> torch.Tensor:
    shifted = []
    for batch_index in range(values.shape[0]):
        vertical, horizontal = (
            int(item) for item in displacement_yx[batch_index].detach().cpu()
        )
        shifted.append(
            shift_without_wrap(
                values[batch_index : batch_index + 1], vertical, horizontal
            )
        )
    return torch.cat(shifted, dim=0)


def exact_motion_alignment_loss(
    model: MiohRestorerV4Q,
    anchor_values: torch.Tensor,
) -> torch.Tensor:
    synthetic, positions, validity = make_synthetic_motion_sequence(anchor_values)
    _restored, _confidence, _base, _texture, diagnostics = (
        model.forward_with_distillation(
            synthetic, capture_alignment=True, capture_features=False
        )
    )
    displacements = torch.stack(
        [alignment_displacement(positions, reference, target)
         for reference, target in V4_ALIGNMENT_PAIRS],
        dim=1,
    )
    batch, pairs = displacements.shape[:2]
    coarse_shape = diagnostics["alignment_coarse"].shape[-2:]
    targets_flat = exact_motion_to_hier27_distributions(
        displacements.reshape(batch * pairs, 2).to(dtype=synthetic.dtype),
        eighth_size=coarse_shape,
    )
    targets = tuple(
        item.reshape(batch, pairs, 9, *item.shape[-2:])
        for item in targets_flat
    )

    pair_masks: list[torch.Tensor] = []
    for pair_index, (reference, target) in enumerate(V4_ALIGNMENT_PAIRS):
        displacement = displacements[:, pair_index]
        aligned_target_mask = _shift_batch(
            synthetic[:, target, 3:4], displacement
        )
        aligned_target_validity = _shift_batch(
            validity[:, target], displacement
        )
        valid = validity[:, reference] * aligned_target_validity
        pair_masks.append(
            torch.maximum(synthetic[:, reference, 3:4], aligned_target_mask)
            * valid
        )
    return alignment_kl_from_targets(
        diagnostics, targets, torch.stack(pair_masks, dim=1)
    )


def feature_distillation_loss(
    diagnostics: dict[str, torch.Tensor],
    values: torch.Tensor,
    output_masks: torch.Tensor,
    teacher: torch.nn.Module,
    adapter: V4FeatureDistillationAdapter,
) -> torch.Tensor:
    teacher_dtype = next(teacher.parameters()).dtype
    teacher_features = extract_basicvsrpp_reconstruction_features(
        teacher, values[:, :, :3].to(dtype=teacher_dtype)
    ).to(dtype=diagnostics["fused_quarter"].dtype)
    return projected_roi_feature_loss(
        diagnostics["fused_quarter"],
        teacher_features,
        output_masks,
        adapter,
    )


def roi_psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.expand_as(prediction)
    mse = ((prediction.float() - target.float()).square() * weights).sum()
    mse = mse / weights.sum().clamp_min(1.0)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


@torch.no_grad()
def validate(
    model: MiohRestorerV4Q,
    loader: DataLoader,
    criterion: MiohRestorerV4Loss,
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "roi_psnr": 0.0, "confidence_error_corr": 0.0}
    count = 0
    for batch in loader:
        values, targets, output_inputs, masks = window_tensors(batch, 0, device)
        loss, _stats, restored, confidence, _, _diagnostics = forward_loss(
            model, criterion, (values, targets, output_inputs, masks)
        )
        error = (targets - (output_inputs + masks * (restored - output_inputs))).abs().mean(
            dim=2, keepdim=True
        )
        totals["loss"] += float(loss)
        totals["roi_psnr"] += float(roi_psnr(restored, targets, masks))
        totals["confidence_error_corr"] += float(
            confidence_error_correlation(confidence, error, masks)
        )
        count += 1
        if count >= batches:
            break
    model.train()
    return {key: value / max(1, count) for key, value in totals.items()}


def rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    model: MiohRestorerV4Q,
    ema: MiohRestorerV4Q,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    *,
    feature_adapter: V4FeatureDistillationAdapter | None = None,
    archive: bool,
    stop_reason: str | None = None,
) -> Path:
    stage = stage_definition(args.stage)
    lineage_base_steps = int(getattr(args, "lineage_base_steps", 0))
    payload = {
        "state_dict": tree_to_cpu(model.state_dict()),
        "ema_state_dict": tree_to_cpu(ema.state_dict()),
        "optimizer": tree_to_cpu(optimizer.state_dict()),
        "step": step,
        "trained": step > 0,
        "config": model_config(model, args),
        "stage_id": stage.stage_id,
        "stage_name": stage.name,
        "stage_step": step,
        "stage_steps": args.steps,
        "stage_complete": step >= args.steps and stop_reason is None,
        "lineage_base_steps": lineage_base_steps,
        "lineage_total_steps": lineage_base_steps + step,
        "parent_checkpoint": getattr(args, "parent_checkpoint", None),
        "parent_checkpoint_sha256": getattr(args, "parent_checkpoint_sha256", None),
        "initialized_from_weights": getattr(args, "initialized_from_weights", None),
        "rng_state": rng_state(),
    }
    if feature_adapter is not None:
        payload["feature_adapter_state_dict"] = tree_to_cpu(
            feature_adapter.state_dict()
        )
    if stop_reason:
        payload["stop_reason"] = stop_reason
    latest = args.work_dir / "mioh-restorer-v4-latest.pth"
    temporary = latest.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(latest)
    if archive:
        output = args.work_dir / f"mioh-restorer-v4-step-{step:07d}.pth"
        temporary = output.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        return output
    return latest


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.work_dir / "metrics.jsonl"
    train_loader = make_loader(args.train_metadata_root, args, training=True)
    val_loader = (
        make_loader(args.val_metadata_root, args, training=False)
        if args.val_metadata_root
        else None
    )
    model = build_model(args).to(device).train()
    stage = stage_definition(args.stage)
    feature_adapter = (
        V4FeatureDistillationAdapter(
            student_channels=model.fusion_quarter_channels,
            teacher_channels=64,
        ).to(device)
        if stage.stage_id == 2
        else None
    )
    ema = copy.deepcopy(model).eval()
    ema.enable_gradient_checkpointing(False)
    ema.requires_grad_(False)
    optimizer_parameters = list(model.parameters())
    if feature_adapter is not None:
        optimizer_parameters.extend(feature_adapter.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = MiohRestorerV4Loss(
        candidate_weight=args.candidate_weight,
        high_frequency_weight=args.high_frequency_weight,
        base_weight=args.base_weight,
        confidence_weight=args.confidence_weight,
        confidence_regularization_weight=args.confidence_regularization_weight,
        temporal_weight=args.temporal_weight,
        gradient_weight=args.gradient_weight,
        structural_weight=args.structural_weight,
        confidence_scale=args.confidence_scale,
    ).to(device)
    args.teacher_checkpoint_sha256 = (
        file_sha256(args.teacher_checkpoint)
        if args.teacher_checkpoint is not None
        else None
    )
    start_step = 0
    args.lineage_base_steps = 0
    args.parent_checkpoint = None
    args.parent_checkpoint_sha256 = None
    args.initialized_from_weights = None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if int(payload.get("config", {}).get("version", 0)) != 4:
            raise ValueError("resume checkpoint is not a V4 checkpoint")
        if int(payload.get("stage_id", 0)) != stage.stage_id:
            raise ValueError(
                "resume checkpoint belongs to a different training stage"
            )
        model.load_state_dict(payload["state_dict"], strict=True)
        ema.load_state_dict(payload["ema_state_dict"], strict=True)
        if feature_adapter is not None:
            adapter_state = payload.get("feature_adapter_state_dict")
            if not isinstance(adapter_state, dict):
                raise ValueError("stage 2 resume checkpoint has no feature adapter")
            feature_adapter.load_state_dict(adapter_state, strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        move_optimizer(optimizer, device)
        start_step = int(payload.get("stage_step", payload["step"]))
        if start_step >= args.steps:
            raise ValueError("resume checkpoint has already reached requested steps")
        args.lineage_base_steps = int(payload.get("lineage_base_steps", 0))
        args.parent_checkpoint = payload.get("parent_checkpoint")
        args.parent_checkpoint_sha256 = payload.get("parent_checkpoint_sha256")
        args.initialized_from_weights = payload.get("initialized_from_weights")
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
    elif args.initialize_from:
        payload = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        if int(payload.get("config", {}).get("version", 0)) != 4:
            raise ValueError("initialization checkpoint is not a V4 checkpoint")
        expected_parent = stage.stage_id - 1
        if int(payload.get("stage_id", 0)) != expected_parent:
            raise ValueError(
                f"stage {stage.stage_id} must initialize from completed "
                f"stage {expected_parent}"
            )
        if not bool(payload.get("stage_complete", False)):
            raise ValueError("previous-stage checkpoint is not marked complete")
        state_key = (
            "ema_state_dict"
            if args.initialize_weights == "ema"
            else "state_dict"
        )
        inherited_state = payload[state_key]
        model.load_state_dict(inherited_state, strict=True)
        # A new EMA begins from the selected parent weights.  Carrying the old
        # average would keep stale history from an objective that has ended.
        ema.load_state_dict(inherited_state, strict=True)
        args.lineage_base_steps = int(
            payload.get("lineage_total_steps", payload.get("stage_step", 0))
        )
        args.parent_checkpoint = str(args.initialize_from.resolve())
        args.parent_checkpoint_sha256 = file_sha256(args.initialize_from)
        args.initialized_from_weights = args.initialize_weights
    spynet_teacher: torch.nn.Module | None = None
    feature_teacher: torch.nn.Module | None = None
    if stage.stage_id == 1:
        spynet_teacher = load_spynet_teacher(
            args.teacher_checkpoint,
            device,
            fp16=args.teacher_fp16,
        )
    elif stage.stage_id == 2:
        feature_teacher = load_basicvsrpp_feature_teacher(
            args.teacher_checkpoint,
            device,
            fp16=args.teacher_fp16,
        )
    distillation = distillation_settings(args)
    guard = TrainingMemoryGuard(
        device,
        MemoryThresholds(
            warning_mps_ratio=args.memory_warning_ratio,
            critical_mps_ratio=args.memory_critical_ratio,
            warning_system_available_gib=args.memory_warning_available_gib,
            critical_system_available_gib=args.memory_critical_available_gib,
        ),
    )
    stop_requested = False
    perceptual_criterion: MaskedVGG16PerceptualLoss | None = None

    def request_stop(_signal_number, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        "MiohRestorerV4-Q training: "
        f"device={device}, parameters={parameter_count(model):,}, "
        f"stage={stage.stage_id}:{stage.name}, "
        f"samples={len(train_loader.dataset)}, stage_steps={start_step + 1}-{args.steps}, "
        f"frames=9->5, stride=4, reach=+/-{model.alignment.input_reach}px, "
        f"checkpointing={model.gradient_checkpointing}, "
        f"lineage_base_steps={args.lineage_base_steps}, "
        f"gt_primary=True, pixel_distillation=False, "
        f"distillation={distillation}"
    )
    active_stage_name = apply_training_stage(model, args)
    print(
        f"training stage fixed for this run: {stage.stage_id}:{active_stage_name}",
        flush=True,
    )
    iterator = iter(train_loader)
    last_step = start_step
    last_time = time.perf_counter()
    last_log_step = start_step
    for step in range(start_step + 1, args.steps + 1):
        if stop_requested:
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        lr = learning_rate(args, step)
        set_learning_rate(optimizer, lr)
        loss_weights = apply_loss_weights(criterion, args, step)
        optimizer.zero_grad(set_to_none=True)
        tensors = window_tensors(batch, 0, device)
        result = forward_loss(
            model,
            criterion,
            tensors,
            capture_alignment=stage.stage_id == 1,
            capture_features=stage.stage_id == 2,
        )
        primary_loss = result[0]
        loss = primary_loss
        spynet_loss = loss.new_zeros(())
        feature_loss = loss.new_zeros(())
        if stage.stage_id == 1:
            if spynet_teacher is None:
                raise AssertionError("Stage 1 SPyNet teacher was not loaded")
            spynet_loss = spynet_alignment_loss(
                result[5], tensors[0], spynet_teacher, args
            )
            loss = loss + distillation["spynet_alignment"] * spynet_loss
        elif stage.stage_id == 2:
            if feature_teacher is None or feature_adapter is None:
                raise AssertionError("Stage 2 feature teacher was not loaded")
            feature_loss = feature_distillation_loss(
                result[5], tensors[0], tensors[3], feature_teacher, feature_adapter
            )
            loss = loss + distillation["feature"] * feature_loss
        perceptual = loss.new_zeros(())
        perceptual_every = stage.perceptual_every
        if (
            loss_weights["perceptual"] > 0
            and perceptual_every > 0
            and step % perceptual_every == 0
        ):
            if perceptual_criterion is None:
                perceptual_criterion = MaskedVGG16PerceptualLoss(
                    frame_stride=1,
                    image_size=args.perceptual_image_size,
                ).to(device)
            # The middle output is sampled to control MPS memory.  Across
            # random clips this remains an unbiased spatial-detail teacher.
            perceptual = perceptual_criterion(
                result[2][:, 2:3], tensors[1][:, 2:3], result[4][:, 2:3]
            )
            loss = loss + loss_weights["perceptual"] * perceptual
        # GT reconstruction remains the main objective.  Teacher terms only
        # add internal alignment/feature guidance and never supply RGB targets.
        loss.backward()
        exact_motion_loss = loss.new_zeros(())
        use_exact_motion = (
            stage.stage_id == 1
            and distillation["exact_motion_alignment"] > 0
            and random.random() < distillation["exact_motion_probability"]
        )
        if use_exact_motion:
            exact_motion_loss = exact_motion_alignment_loss(
                model, tensors[0][:, 4]
            )
            (
                distillation["exact_motion_alignment"] * exact_motion_loss
            ).backward()
        total_logged_loss = (
            loss.detach()
            + distillation["exact_motion_alignment"]
            * exact_motion_loss.detach()
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters, args.max_grad_norm
        )
        optimizer.step()
        update_ema(
            ema,
            model,
            stage.ema_decay,
        )
        last_step = step
        if step % args.log_every == 0 or step == start_step + 1:
            now = time.perf_counter()
            stats = result[1]
            record = {
                "step": step,
                "training_stage": active_stage_name,
                "loss": float(total_logged_loss),
                "gt_primary_loss": float(primary_loss.detach()),
                "perceptual": float(perceptual.detach()),
                "spynet_alignment": float(spynet_loss.detach()),
                "exact_motion_alignment": float(exact_motion_loss.detach()),
                "exact_motion_applied": use_exact_motion,
                "feature_distillation": float(feature_loss.detach()),
                "distillation_weights": distillation,
                "loss_weights": loss_weights,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "steps_per_second": (step - last_log_step)
                / max(now - last_time, 1e-6),
                **{key: float(value.detach()) for key, value in stats.items()},
            }
            last_time = now
            last_log_step = step
            snapshot = guard.capture("train")
            record["memory"] = snapshot.as_record()
            append_jsonl(metrics_path, record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if guard.status(snapshot) == "critical":
                release_device_memory(device)
                if guard.status(guard.capture("after_cleanup")) == "critical":
                    save_checkpoint(
                        model,
                        ema,
                        optimizer,
                        args,
                        step,
                        feature_adapter=feature_adapter,
                        archive=True,
                        stop_reason="memory_pressure",
                    )
                    return MEMORY_EXIT_CODE
        if step % args.save_latest_every == 0:
            save_checkpoint(
                model,
                ema,
                optimizer,
                args,
                step,
                feature_adapter=feature_adapter,
                archive=False,
            )
        if step % args.save_every == 0:
            output = save_checkpoint(
                model,
                ema,
                optimizer,
                args,
                step,
                feature_adapter=feature_adapter,
                archive=True,
            )
            print(f"saved: {output}", flush=True)
        if val_loader is not None and step % args.validate_every == 0:
            validation = validate(
                ema, val_loader, criterion, device, args.validation_batches
            )
            append_jsonl(metrics_path, {"step": step, "validation": validation})
            print(f"validation: {json.dumps(validation)}", flush=True)
    reason = "signal" if stop_requested else None
    output = save_checkpoint(
        model,
        ema,
        optimizer,
        args,
        last_step,
        feature_adapter=feature_adapter,
        archive=True,
        stop_reason=reason,
    )
    print(f"training finished at step {last_step}; saved: {output}")
    if reason is None and last_step >= args.steps:
        manifest = {
            "stage_id": stage.stage_id,
            "stage_name": stage.name,
            "stage_steps": last_step,
            "lineage_total_steps": args.lineage_base_steps + last_step,
            "checkpoint": str(output.resolve()),
            "checkpoint_sha256": file_sha256(output),
            "parent_checkpoint": args.parent_checkpoint,
            "initialized_from_weights": args.initialized_from_weights,
            "status": "complete_awaiting_evaluation",
        }
        (args.work_dir / "stage-complete.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("stage complete; evaluate raw/EMA before starting the next stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
