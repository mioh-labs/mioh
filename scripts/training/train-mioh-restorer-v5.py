#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Train one independent MiohRestorer V5 stage on Apple Silicon.

V5-Q/S are greenfield fixed-operator models. V5-HQ is the quality-first hybrid:
it initializes a recurrent BasicVSR++ backbone (including SPyNet and DCNv2),
then trains the new full-resolution deformable-attention ROI refiner. Every
stage starts from the previous stage's EMA weights with a fresh optimizer.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lada.models.mioh_restorer.adversarial import (
    SpectralUNetDiscriminator,
    TemporalPatchDiscriminator,
    discriminator_feature_matching_loss,
    discriminator_hinge_loss,
    discriminator_relativistic_pair_loss,
    discriminator_roi_patch_weights,
    generator_hinge_loss,
    generator_relativistic_pair_loss,
    roi_temporal_discriminator_input,
)
from lada.models.mioh_restorer.curriculum_v5 import (
    V5_STAGES,
    previous_stage,
    stage_definition,
    stage_learning_rate,
)
from lada.models.mioh_restorer.curriculum_v5_hq import (
    V5_HQ_STAGES,
    hq_learning_rate,
    hq_stage_definition,
)
from lada.models.mioh_restorer.losses_v5 import (
    MiohRestorerV5Loss,
    high_frequency,
    masked_correlation,
    masked_local_correlation,
    masked_mean,
    masked_projection_statistics,
)
from lada.models.mioh_restorer.model_v5 import (
    MiohRestorerV5,
    MiohRestorerV5Config,
    parameter_count,
)
from lada.models.mioh_restorer.model_v5_hq import (
    MiohRestorerV5HQ,
    MiohRestorerV5HQConfig,
    feather_texture_compositor_mask,
)
from lada.models.mioh_restorer.native_dataset_v5 import (
    MiohRestorerV5NativeDataset,
    V5BucketBatchSampler,
)
from lada.models.mioh_restorer.supervision_v5 import (
    V5PerceptualLoss,
    flow_aligned_temporal_tensors,
    known_motion_alignment_loss,
    natural_alignment_losses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("q", "s", "hq"), default="q")
    parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default="off")
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--hq-fixed-learning-rate",
        type=float,
        help=(
            "diagnostic: keep the HQ learning rate fixed, primarily when "
            "continuing a completed short ablation without restarting warmup"
        ),
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--perceptual-image-size", type=int, default=224)
    parser.add_argument("--known-motion-maximum", type=float, default=40.0)
    parser.add_argument(
        "--basicvsrpp-checkpoint",
        type=Path,
        default=Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"),
        help="V5-HQ recurrent-backbone initialization; used only by variant hq",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help=(
            "load raw model/EMA and local step from --resume, but rebuild "
            "optimizers for a changed frozen-parameter policy"
        ),
    )
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--restart-stage", action="store_true")
    parser.add_argument(
        "--initialize-refiner-from",
        type=Path,
        help="V5-HQ checkpoint whose non-backbone EMA weights initialize a diagnostic run",
    )
    parser.add_argument("--hq-hf-ablation", action="store_true")
    parser.add_argument("--hq-hf-amplitude-weight", type=float, default=0.05)
    parser.add_argument("--hq-hf-correlation-weight", type=float, default=0.02)
    parser.add_argument("--hq-hf-local-correlation-weight", type=float, default=0.0)
    parser.add_argument("--hq-hf-local-correlation-patch-size", type=int, default=32)
    parser.add_argument(
        "--hq-hf-local-correlation-on-residual", action="store_true"
    )
    parser.add_argument("--hq-hf-residual-reconstruction-weight", type=float, default=0.0)
    parser.add_argument(
        "--hq-hf-correction-target-projection-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--hq-hf-correction-target-orthogonal-energy-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--hq-guard-ring-identity-weight", type=float, default=0.0)
    parser.add_argument(
        "--hq-texture-effective-mask",
        action="store_true",
        help=(
            "diagnostic: composite the HQ texture residual only over the actual "
            "mosaic footprint while retaining the stabilized mask for the backbone"
        ),
    )
    parser.add_argument(
        "--hq-texture-effective-mask-feather-radius",
        type=int,
        default=0,
    )
    parser.add_argument("--hq-freeze-base-head", action="store_true")
    parser.add_argument("--hq-freeze-confidence-head", action="store_true")
    parser.add_argument("--hq-raw-temporal-candidates", action="store_true")
    parser.add_argument("--hq-raw-temporal-encoder-channels", type=int, default=0)
    parser.add_argument("--hq-raw-temporal-nearest-warp", action="store_true")
    parser.add_argument("--hq-raw-temporal-zero-input", action="store_true")
    parser.add_argument("--hq-confidence-weight", type=float, default=0.03)
    parser.add_argument("--hq-gan-weight", type=float, default=0.0)
    parser.add_argument(
        "--hq-gan-loss",
        choices=("hinge", "rpgan"),
        default="hinge",
        help="training-only discriminator objective",
    )
    parser.add_argument("--hq-gan-generator-hinge-weight", type=float, default=1.0)
    parser.add_argument("--hq-gan-feature-matching-weight", type=float, default=0.0)
    parser.add_argument("--hq-gan-start-step", type=int, default=0)
    parser.add_argument(
        "--hq-gan-discriminator-pretrain-until-step",
        type=int,
        default=0,
        help=(
            "diagnostic: update only the discriminator through this local step; "
            "the generator and EMA remain bit-identical"
        ),
    )
    parser.add_argument("--hq-gan-warmup-steps", type=int, default=50)
    parser.add_argument("--hq-gan-learning-rate", type=float, default=1e-4)
    parser.add_argument("--hq-gan-discriminator-channels", type=int, default=16)
    parser.add_argument(
        "--hq-gan-discriminator-architecture",
        choices=("patch", "unet-sn"),
        default="patch",
        help="training-only discriminator topology",
    )
    parser.add_argument(
        "--hq-gan-freeze-discriminator",
        action="store_true",
        help=(
            "keep a pretrained discriminator fixed and use it as a learned "
            "structural loss during generator training"
        ),
    )
    parser.add_argument("--hq-gan-image-size", type=int, default=192)
    parser.add_argument("--hq-gan-frame-stride", type=int, default=1)
    parser.add_argument("--hq-gan-crop-padding", type=int, default=16)
    parser.add_argument("--hq-gan-minimum-crop-size", type=int, default=96)
    parser.add_argument("--hq-gan-temporal", action="store_true")
    parser.add_argument("--hq-gan-normalize-secondary-rms", action="store_true")
    parser.add_argument(
        "--hq-gan-candidate-primary",
        action="store_true",
        help=(
            "diagnostic: show the discriminator generated RGB rather than a "
            "shared clean-target condition in its primary channels"
        ),
    )
    parser.add_argument(
        "--hq-gan-initialize-discriminator-from",
        type=Path,
        help=(
            "initialize only the training-time discriminator from a trusted "
            "GAN checkpoint while keeping the requested generator checkpoint"
        ),
    )
    parser.add_argument("--hq-hf-early-stop", action="store_true")
    parser.add_argument("--mosaic-block-minimum", type=float)
    parser.add_argument("--mosaic-block-maximum", type=float)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.train_manifest, args.validation_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    positive = (
        "batch_size",
        "accumulate",
        "prefetch",
        "save_every",
        "validate_every",
        "validation_batches",
        "log_every",
        "perceptual_image_size",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.workers < 0 or (args.steps is not None and args.steps <= 0):
        raise ValueError("workers/steps are invalid")
    if args.hq_fixed_learning_rate is not None:
        if args.hq_fixed_learning_rate <= 0:
            raise ValueError("hq-fixed-learning-rate must be positive")
        if args.variant != "hq":
            raise ValueError("hq-fixed-learning-rate requires --variant hq")
    if not 0 < args.ema_decay < 1:
        raise ValueError("ema-decay must be between zero and one")
    if args.resume and args.initialize_from:
        raise ValueError("resume and initialize-from are mutually exclusive")
    if args.resume_model_only and args.resume is None:
        raise ValueError("resume-model-only requires an explicit --resume checkpoint")
    if args.initialize_refiner_from and (args.resume or args.initialize_from):
        raise ValueError(
            "initialize-refiner-from cannot be combined with resume/initialize-from"
        )
    if args.restart_stage and args.resume:
        raise ValueError("restart-stage and resume are mutually exclusive")
    if args.stage >= 5 and args.variant not in {"q", "hq"}:
        raise ValueError("temporal stages 5/6 require a five-output V5-Q/HQ model")
    if args.variant == "hq" and not args.basicvsrpp_checkpoint.is_file():
        raise FileNotFoundError(args.basicvsrpp_checkpoint)
    if args.device != "cuda" and args.amp != "off":
        raise ValueError("mixed precision is enabled only for CUDA; use --amp off on MPS")
    if args.variant == "hq":
        for path in (args.train_manifest, args.validation_manifest):
            validate_hq_manifest_minimum_bucket(path)
    if args.hq_hf_ablation and args.variant != "hq":
        raise ValueError("hq-hf-ablation requires --variant hq")
    if (
        args.hq_gan_weight < 0
        or args.hq_gan_generator_hinge_weight < 0
        or args.hq_gan_feature_matching_weight < 0
        or args.hq_gan_learning_rate <= 0
    ):
        raise ValueError("invalid HQ GAN weight/learning rate")
    if args.hq_gan_weight > 0 and not args.hq_hf_ablation:
        raise ValueError("HQ GAN training requires hq-hf-ablation")
    if (
        args.hq_gan_start_step < 0
        or args.hq_gan_warmup_steps < 0
        or args.hq_gan_discriminator_pretrain_until_step < 0
    ):
        raise ValueError("HQ GAN start/warmup steps cannot be negative")
    if (
        args.hq_gan_discriminator_pretrain_until_step
        and args.hq_gan_weight <= 0
    ):
        raise ValueError("discriminator pretraining requires HQ GAN training")
    if (
        args.hq_gan_freeze_discriminator
        and args.hq_gan_discriminator_pretrain_until_step
    ):
        raise ValueError("a frozen discriminator cannot be pretrained")
    if args.hq_gan_discriminator_channels <= 0:
        raise ValueError("HQ GAN discriminator channels must be positive")
    if args.hq_gan_image_size < 32 or args.hq_gan_frame_stride <= 0:
        raise ValueError("invalid HQ GAN image size/frame stride")
    if args.hq_gan_crop_padding < 0 or args.hq_gan_minimum_crop_size < 1:
        raise ValueError("invalid HQ GAN crop settings")
    if args.hq_gan_weight > 0 and args.accumulate != 1:
        raise ValueError("HQ GAN pilot currently requires --accumulate 1")
    if args.hq_gan_initialize_discriminator_from is not None:
        if args.hq_gan_weight <= 0:
            raise ValueError("discriminator initialization requires HQ GAN training")
        if not args.hq_gan_initialize_discriminator_from.is_file():
            raise FileNotFoundError(args.hq_gan_initialize_discriminator_from)
    if args.hq_freeze_base_head and not args.hq_hf_ablation:
        raise ValueError("hq-freeze-base-head requires hq-hf-ablation")
    if args.hq_freeze_confidence_head and not args.hq_hf_ablation:
        raise ValueError("hq-freeze-confidence-head requires hq-hf-ablation")
    if args.hq_texture_effective_mask and not args.hq_hf_ablation:
        raise ValueError("hq-texture-effective-mask requires hq-hf-ablation")
    if args.hq_texture_effective_mask_feather_radius < 0:
        raise ValueError(
            "hq-texture-effective-mask-feather-radius cannot be negative"
        )
    if (
        args.hq_texture_effective_mask_feather_radius
        and not args.hq_texture_effective_mask
    ):
        raise ValueError(
            "hq-texture-effective-mask-feather-radius requires "
            "hq-texture-effective-mask"
        )
    if args.hq_raw_temporal_candidates and not args.hq_hf_ablation:
        raise ValueError("hq-raw-temporal-candidates requires hq-hf-ablation")
    if args.hq_raw_temporal_candidates and args.initialize_refiner_from:
        raise ValueError(
            "raw temporal candidate pilot cannot load a legacy refiner shape"
        )
    if args.hq_raw_temporal_encoder_channels < 0:
        raise ValueError("hq-raw-temporal-encoder-channels cannot be negative")
    if (
        args.hq_raw_temporal_encoder_channels
        and not args.hq_raw_temporal_candidates
    ):
        raise ValueError(
            "hq-raw-temporal-encoder-channels requires raw temporal candidates"
        )
    if args.hq_raw_temporal_nearest_warp and not args.hq_raw_temporal_candidates:
        raise ValueError(
            "hq-raw-temporal-nearest-warp requires raw temporal candidates"
        )
    if args.hq_raw_temporal_zero_input and not args.hq_raw_temporal_candidates:
        raise ValueError(
            "hq-raw-temporal-zero-input requires raw temporal candidates"
        )
    if (
        args.hq_hf_local_correlation_on_residual
        and not args.hq_freeze_base_head
    ):
        raise ValueError(
            "hq-hf-local-correlation-on-residual requires hq-freeze-base-head"
        )
    if args.initialize_refiner_from and not args.initialize_refiner_from.is_file():
        raise FileNotFoundError(args.initialize_refiner_from)
    if min(
        args.hq_hf_amplitude_weight,
        args.hq_hf_correlation_weight,
        args.hq_hf_local_correlation_weight,
        args.hq_hf_residual_reconstruction_weight,
        args.hq_hf_correction_target_projection_weight,
        args.hq_hf_correction_target_orthogonal_energy_weight,
        args.hq_guard_ring_identity_weight,
        args.hq_confidence_weight,
    ) < 0:
        raise ValueError("HQ HF diagnostic weights must be non-negative")
    if args.hq_hf_local_correlation_patch_size <= 1:
        raise ValueError("HQ HF local correlation patch size must exceed one")
    if (
        args.hq_hf_correction_target_projection_weight > 0
        or args.hq_hf_correction_target_orthogonal_energy_weight > 0
    ) and not args.hq_freeze_base_head:
        raise ValueError(
            "HQ correction-target supervision requires hq-freeze-base-head"
        )
    if (args.mosaic_block_minimum is None) != (args.mosaic_block_maximum is None):
        raise ValueError("mosaic block minimum/maximum must be set together")
    if args.mosaic_block_minimum is not None and (
        args.mosaic_block_minimum <= 0
        or args.mosaic_block_maximum < args.mosaic_block_minimum
    ):
        raise ValueError("invalid mosaic block range")


def validate_hq_manifest_minimum_bucket(path: Path) -> None:
    """Reject crops too small for BasicVSR++'s quarter-resolution SPyNet."""

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            bucket = int(value["bucket"])
            if bucket < 256:
                raise ValueError(
                    f"V5-HQ requires native buckets >= 256; "
                    f"{path}:{line_number} uses {bucket}"
                )


def available_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    variant: str,
    stage: int,
    basicvsrpp_checkpoint: Path | None = None,
    *,
    raw_temporal_candidates: bool = False,
    raw_temporal_encoder_channels: int = 0,
    raw_temporal_nearest_warp: bool = False,
    raw_temporal_zero_input: bool = False,
) -> MiohRestorerV5 | MiohRestorerV5HQ:
    if variant == "s":
        return MiohRestorerV5.shipping()
    if variant == "hq":
        model = MiohRestorerV5HQ(
            MiohRestorerV5HQConfig(
                raw_temporal_candidates=raw_temporal_candidates,
                raw_temporal_encoder_channels=raw_temporal_encoder_channels,
                raw_temporal_nearest_warp=raw_temporal_nearest_warp,
                raw_temporal_zero_input=raw_temporal_zero_input,
            )
        )
        if basicvsrpp_checkpoint is None:
            raise ValueError("V5-HQ requires BasicVSR++ backbone initialization")
        model.load_basicvsrpp_checkpoint(basicvsrpp_checkpoint)
        return model
    return MiohRestorerV5(MiohRestorerV5Config.quality())


def configure_trainable_parameters(
    model: MiohRestorerV5 | MiohRestorerV5HQ,
    *,
    variant: str,
    stage: object,
) -> None:
    model.requires_grad_(True)
    if variant != "hq":
        return
    if not isinstance(model, MiohRestorerV5HQ):
        raise TypeError("V5-HQ variant built an incompatible model")
    train_backbone = bool(getattr(stage, "train_backbone"))
    train_spynet = bool(getattr(stage, "train_spynet"))
    model.backbone.requires_grad_(train_backbone)
    model.spynet.requires_grad_(train_spynet)
    # The quality refiner and deformable-attention branch always learn.
    for name, parameter in model.named_parameters():
        if not name.startswith("backbone."):
            parameter.requires_grad_(True)


def load_hq_refiner_checkpoint(
    model: MiohRestorerV5HQ, checkpoint: Path
) -> None:
    """Load only the V5 refiner EMA while preserving the chosen backbone."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("V5-HQ refiner checkpoint is invalid")
    raw = payload.get("ema_state_dict", payload.get("state_dict"))
    if not isinstance(raw, dict):
        raise TypeError("V5-HQ refiner checkpoint has no state dictionary")
    current = model.state_dict()
    refiner = {
        str(key): value
        for key, value in raw.items()
        if not str(key).startswith("backbone.")
    }
    missing = [key for key in current if not key.startswith("backbone.") and key not in refiner]
    unexpected = [key for key in refiner if key not in current]
    if missing or unexpected:
        raise ValueError(
            f"V5-HQ refiner state mismatch; missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    current.update(refiner)
    model.load_state_dict(current, strict=True)


def amp_configuration(
    requested: str, device: torch.device
) -> tuple[bool, torch.dtype | None, bool]:
    if requested == "off" or device.type != "cuda":
        return False, None, False
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    dtype = torch.bfloat16 if requested == "bf16" else torch.float16
    return True, dtype, dtype == torch.float16


def make_loader(
    manifest: Path,
    *,
    output_indices: tuple[int, ...],
    batch_size: int,
    workers: int,
    prefetch: int,
    training: bool,
    seed: int,
    mosaic_block_size_range: tuple[float, float] | None = None,
) -> tuple[DataLoader, V5BucketBatchSampler]:
    dataset = MiohRestorerV5NativeDataset(
        manifest,
        output_indices=output_indices,
        degrade=training,
        horizontal_flip=training,
        time_reverse=training,
        deterministic=not training,
        mosaic_block_size_range=mosaic_block_size_range,
    )
    sampler = V5BucketBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        seed=seed,
    )
    options: dict[str, object] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers:
        options["prefetch_factor"] = prefetch
    return DataLoader(dataset, **options), sampler


def move_batch(
    batch: dict[str, object], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = []
    for name in ("inputs", "targets", "masks"):
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch field {name} is not a tensor")
        tensors.append(value.to(device, non_blocking=True))
    loss_masks = batch.get("loss_masks", batch["masks"])
    if not isinstance(loss_masks, torch.Tensor):
        raise TypeError("batch field loss_masks is not a tensor")
    return (*tensors, loss_masks.to(device, non_blocking=True))  # type: ignore[return-value]


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for destination, source in zip(ema.parameters(), model.parameters(), strict=True):
        destination.lerp_(source.detach(), 1.0 - decay)
    for destination, source in zip(ema.buffers(), model.buffers(), strict=True):
        destination.copy_(source)


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def hq_gan_weight(args: argparse.Namespace, step: int) -> float:
    """Ramp adversarial pressure without a discontinuity at GAN activation."""

    if args.hq_gan_weight <= 0 or step <= args.hq_gan_start_step:
        return 0.0
    active_step = step - args.hq_gan_start_step
    if args.hq_gan_warmup_steps:
        return args.hq_gan_weight * min(
            active_step / args.hq_gan_warmup_steps,
            1.0,
        )
    return args.hq_gan_weight


def atomic_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def checkpoint_payload(
    model: MiohRestorerV5,
    ema: MiohRestorerV5,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    *,
    step: int,
    epoch: int,
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": "mioh-restorer-v5-native-v1",
        "variant": args.variant,
        "stage": args.stage,
        "config": asdict(model.config),
        "local_step": step,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "arguments": vars(args),
    }
    if discriminator is not None:
        payload["discriminator_state_dict"] = discriminator.state_dict()
    if discriminator_optimizer is not None:
        payload["discriminator_optimizer_state_dict"] = (
            discriminator_optimizer.state_dict()
        )
    return payload


def save_checkpoint(
    model: MiohRestorerV5,
    ema: MiohRestorerV5,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    work_dir: Path,
    *,
    step: int,
    epoch: int,
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    payload = checkpoint_payload(
        model,
        ema,
        optimizer,
        scaler,
        args,
        step=step,
        epoch=epoch,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
    )
    numbered = work_dir / f"mioh-v5-{args.variant}-stage{args.stage}-step-{step:06d}.pth"
    latest = work_dir / f"mioh-v5-{args.variant}-stage{args.stage}-latest.pth"
    atomic_save(payload, numbered)
    temporary = latest.with_suffix(latest.suffix + ".tmp")
    shutil.copyfile(numbered, temporary)
    temporary.replace(latest)
    return numbered


def load_checkpoint(
    path: Path,
    model: MiohRestorerV5,
    ema: MiohRestorerV5,
    *,
    expected_stage: int,
    expected_variant: str,
    resume: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    resume_optimizer: bool = True,
) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    found_stage = int(payload.get("stage", -1))
    required_stage = expected_stage if resume else expected_stage - 1
    if found_stage != required_stage:
        raise ValueError(
            f"checkpoint stage {found_stage} cannot initialize stage {expected_stage}"
        )
    if payload.get("variant") != expected_variant:
        raise ValueError("checkpoint variant does not match the requested model")
    raw = payload["state_dict"]
    ema_state = payload.get("ema_state_dict", raw)
    model.load_state_dict(raw if resume else ema_state, strict=True)
    ema.load_state_dict(ema_state, strict=True)
    if resume:
        if resume_optimizer:
            if optimizer is None or scaler is None:
                raise ValueError("resume requires optimizer and scaler")
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            if payload.get("scaler_state_dict"):
                scaler.load_state_dict(payload["scaler_state_dict"])
        discriminator_state = payload.get("discriminator_state_dict")
        if discriminator is not None and isinstance(discriminator_state, dict):
            discriminator.load_state_dict(discriminator_state, strict=True)
        discriminator_optimizer_state = payload.get(
            "discriminator_optimizer_state_dict"
        )
        if (
            resume_optimizer
            and discriminator_optimizer is not None
            and isinstance(discriminator_optimizer_state, dict)
        ):
            discriminator_optimizer.load_state_dict(
                discriminator_optimizer_state
            )
        return int(payload.get("local_step", 0)), int(payload.get("epoch", 0))
    return 0, 0


def load_discriminator_checkpoint(
    path: Path,
    discriminator: torch.nn.Module,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("discriminator_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no discriminator state: {path}")
    discriminator.load_state_dict(state, strict=True)


def append_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type == "mps":
        return {
            "mps_allocated_gib": torch.mps.current_allocated_memory() / 2**30,
            "mps_driver_gib": torch.mps.driver_allocated_memory() / 2**30,
        }
    if device.type == "cuda":
        return {
            "cuda_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "cuda_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
    return {}


def clear_accelerator_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def hq_hf_early_stop_reason(metrics: dict[str, object]) -> str | None:
    """Reject an HF gain that reduces correspondence with ground truth."""

    required = (
        "hf_amplitude_ratio",
        "hf_correlation",
        "backbone_hf_amplitude_ratio",
        "backbone_hf_correlation",
    )
    if any(name not in metrics for name in required):
        return None
    amplitude = float(metrics["hf_amplitude_ratio"])
    correlation = float(metrics["hf_correlation"])
    backbone_amplitude = float(metrics["backbone_hf_amplitude_ratio"])
    backbone_correlation = float(metrics["backbone_hf_correlation"])
    if amplitude > backbone_amplitude and correlation < backbone_correlation:
        return (
            "HF amplitude exceeded the deployed backbone while HF correlation "
            "fell below it; likely unfaithful detail"
        )
    return None


@torch.no_grad()
def validate(
    model: MiohRestorerV5,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    limit: int,
    use_loss_masks: bool = False,
    use_texture_effective_mask: bool = False,
    texture_effective_mask_feather_radius: int = 0,
    hf_local_correlation_patch_size: int = 32,
) -> dict[str, object]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    metric_counts: dict[str, int] = defaultdict(int)
    bucket_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    bucket_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    count = 0
    empty_roi_batches = 0
    for batch in loader:
        inputs, targets, masks, loss_masks = move_batch(batch, device)
        metric_masks = loss_masks if use_loss_masks else masks
        if not bool(torch.any(metric_masks > 0)):
            empty_roi_batches += 1
            continue
        bucket_value = batch["bucket"]
        bucket = int(bucket_value[0]) if isinstance(bucket_value, torch.Tensor) else int(bucket_value)  # type: ignore[arg-type]
        sources = inputs[:, list(model.config.output_indices), :3]
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            texture_mask = (
                feather_texture_compositor_mask(
                    loss_masks,
                    masks,
                    radius=texture_effective_mask_feather_radius,
                )
                if use_texture_effective_mask
                else None
            )
            restored, confidence, _, texture = model.forward_components(
                inputs,
                texture_compositor_mask=texture_mask,
            )
            backbone_restored = None
            if isinstance(model, MiohRestorerV5HQ) and use_loss_masks:
                output_indices = list(model.config.output_indices)
                backbone_raw = model.backbone(inputs[:, :, :3])[:, output_indices]
                compositor_masks = inputs[:, output_indices, 3:4].clamp(0.0, 1.0)
                backbone_restored = sources + compositor_masks * (
                    backbone_raw - sources
                )
        mse = masked_mean((restored.float() - targets).square(), metric_masks)
        mae = masked_mean((restored.float() - targets).abs(), metric_masks)
        baseline = masked_mean((sources.float() - targets).abs(), metric_masks)
        restored_hf = high_frequency(restored.float())
        target_hf = high_frequency(targets.float())
        restored_amplitude = torch.sqrt(
            masked_mean(restored_hf.square(), metric_masks).clamp_min(1e-12)
        )
        target_amplitude = torch.sqrt(
            masked_mean(target_hf.square(), metric_masks).clamp_min(1e-12)
        )
        (
            hf_projection_gain,
            hf_orthogonal_rms_ratio,
            hf_orthogonal_energy_ratio,
        ) = masked_projection_statistics(restored_hf, target_hf, metric_masks)
        metrics = {
            "roi_psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
            "roi_mae": float(mae),
            "identity_source_mae": float(baseline),
            "confidence_mean": float(masked_mean(confidence.float(), masks)),
            "hf_amplitude_ratio": float(
                restored_amplitude / target_amplitude.clamp_min(1e-6)
            ),
            "hf_correlation": float(
                masked_correlation(restored_hf, target_hf, metric_masks)
            ),
            "hf_local_correlation": float(
                masked_local_correlation(
                    restored_hf,
                    target_hf,
                    metric_masks,
                    patch_size=hf_local_correlation_patch_size,
                )
            ),
            "hf_projection_gain": float(hf_projection_gain),
            "hf_orthogonal_rms_ratio": float(hf_orthogonal_rms_ratio),
            "hf_orthogonal_energy_ratio": float(hf_orthogonal_energy_ratio),
            # Kept for comparison with the first pilots. Despite its historic
            # name, this is the total texture-head RMS, not its HF component.
            "texture_hf_amplitude_ratio": float(
                torch.sqrt(masked_mean(texture.float().square(), metric_masks).clamp_min(1e-12))
                / target_amplitude.clamp_min(1e-6)
            ),
            "texture_total_amplitude_ratio": float(
                torch.sqrt(masked_mean(texture.float().square(), metric_masks).clamp_min(1e-12))
                / target_amplitude.clamp_min(1e-6)
            ),
            "texture_high_frequency_amplitude_ratio": float(
                torch.sqrt(
                    masked_mean(
                        high_frequency(texture.float()).square(), metric_masks
                    ).clamp_min(1e-12)
                )
                / target_amplitude.clamp_min(1e-6)
            ),
        }
        if use_loss_masks:
            # Splitting the supervision mask leaves the stabilized guard ring
            # unsupervised while it is still composited at inference. Report it
            # so an HF-amplitude push cannot grow a boundary halo unnoticed.
            ring = (masks - loss_masks).clamp(0.0, 1.0)
            if bool(torch.any(ring > 0)):
                metrics["ring_mae"] = float(
                    masked_mean((restored.float() - targets).abs(), ring)
                )
                metrics["ring_hf_amplitude_ratio"] = float(
                    torch.sqrt(masked_mean(restored_hf.square(), ring).clamp_min(1e-12))
                    / torch.sqrt(
                        masked_mean(target_hf.square(), ring).clamp_min(1e-12)
                    ).clamp_min(1e-6)
                )
        if backbone_restored is not None:
            backbone_error = backbone_restored.float() - targets
            backbone_hf = high_frequency(backbone_restored.float())
            correction_hf = high_frequency(
                restored.float() - backbone_restored.float()
            )
            needed_correction_hf = high_frequency(
                targets.float() - backbone_restored.float()
            )
            needed_correction_amplitude = torch.sqrt(
                masked_mean(
                    needed_correction_hf.square(), metric_masks
                ).clamp_min(1e-12)
            )
            (
                residual_projection_gain,
                residual_orthogonal_rms_ratio,
                residual_orthogonal_energy_ratio,
            ) = masked_projection_statistics(
                correction_hf, needed_correction_hf, metric_masks
            )
            (
                correction_target_projection_gain,
                correction_target_orthogonal_rms_ratio,
                correction_target_orthogonal_energy_ratio,
            ) = masked_projection_statistics(
                correction_hf, target_hf, metric_masks
            )
            (
                backbone_projection_gain,
                backbone_orthogonal_rms_ratio,
                backbone_orthogonal_energy_ratio,
            ) = masked_projection_statistics(
                backbone_hf, target_hf, metric_masks
            )
            metrics.update(
                {
                    "backbone_roi_psnr": float(
                        -10
                        * torch.log10(
                            masked_mean(backbone_error.square(), metric_masks).clamp_min(1e-12)
                        )
                    ),
                    "backbone_hf_amplitude_ratio": float(
                        torch.sqrt(
                            masked_mean(backbone_hf.square(), metric_masks).clamp_min(1e-12)
                        )
                        / target_amplitude.clamp_min(1e-6)
                    ),
                    "backbone_hf_correlation": float(
                        masked_correlation(backbone_hf, target_hf, metric_masks)
                    ),
                    "backbone_hf_local_correlation": float(
                        masked_local_correlation(
                            backbone_hf,
                            target_hf,
                            metric_masks,
                            patch_size=hf_local_correlation_patch_size,
                        )
                    ),
                    "backbone_hf_projection_gain": float(
                        backbone_projection_gain
                    ),
                    "backbone_hf_orthogonal_rms_ratio": float(
                        backbone_orthogonal_rms_ratio
                    ),
                    "backbone_hf_orthogonal_energy_ratio": float(
                        backbone_orthogonal_energy_ratio
                    ),
                    "hf_residual_amplitude_ratio": float(
                        torch.sqrt(
                            masked_mean(
                                correction_hf.square(), metric_masks
                            ).clamp_min(1e-12)
                        )
                        / needed_correction_amplitude.clamp_min(1e-6)
                    ),
                    "hf_residual_local_correlation": float(
                        masked_local_correlation(
                            correction_hf,
                            needed_correction_hf,
                            metric_masks,
                            patch_size=hf_local_correlation_patch_size,
                        )
                    ),
                    "hf_residual_projection_gain": float(
                        residual_projection_gain
                    ),
                    "hf_residual_orthogonal_rms_ratio": float(
                        residual_orthogonal_rms_ratio
                    ),
                    "hf_residual_orthogonal_energy_ratio": float(
                        residual_orthogonal_energy_ratio
                    ),
                    "hf_correction_target_projection_gain": float(
                        correction_target_projection_gain
                    ),
                    "hf_correction_target_orthogonal_rms_ratio": float(
                        correction_target_orthogonal_rms_ratio
                    ),
                    "hf_correction_target_orthogonal_energy_ratio": float(
                        correction_target_orthogonal_energy_ratio
                    ),
                }
            )
        for name, value in metrics.items():
            totals[name] += value
            metric_counts[name] += 1
            bucket_totals[bucket][name] += value
            bucket_counts[bucket][name] += 1
        count += 1
        if count >= limit:
            break
    if not count:
        raise RuntimeError("validation produced no batches")
    # The ring metrics are absent when a crop's mask saturates its bucket, so
    # every metric averages over the batches that actually produced it.
    result: dict[str, object] = {
        name: value / max(metric_counts[name], 1) for name, value in totals.items()
    }
    result["valid_roi_batches"] = count
    result["empty_roi_batches_skipped"] = empty_roi_batches
    result["buckets"] = {
        str(bucket): {
            name: value / max(bucket_counts[bucket][name], 1)
            for name, value in values.items()
        }
        for bucket, values in sorted(bucket_totals.items())
    }
    return result


def default_parent_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.stage <= 1:
        return None
    if args.variant == "hq":
        parent = V5_HQ_STAGES[args.stage - 2]
    else:
        stage = stage_definition(args.stage)
        parent = previous_stage(stage)
        if parent is None:
            return None
    parent_dir = args.work_root / f"stage-{parent.stage_id:02d}-{parent.name}"
    return parent_dir / f"mioh-v5-{args.variant}-stage{parent.stage_id}-latest.pth"


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = available_device(args.device)
    stage = (
        hq_stage_definition(args.stage)
        if args.variant == "hq"
        else stage_definition(args.stage)
    )
    total_steps = args.steps or stage.default_steps
    work_dir = args.work_root / f"stage-{stage.stage_id:02d}-{stage.name}"
    work_dir.mkdir(parents=True, exist_ok=True)
    latest = work_dir / f"mioh-v5-{args.variant}-stage{args.stage}-latest.pth"
    metrics_path = work_dir / "metrics.jsonl"

    amp_enabled, amp_dtype, scale_gradients = amp_configuration(args.amp, device)
    model = build_model(
        args.variant,
        args.stage,
        args.basicvsrpp_checkpoint if args.variant == "hq" else None,
        raw_temporal_candidates=args.hq_raw_temporal_candidates,
        raw_temporal_encoder_channels=args.hq_raw_temporal_encoder_channels,
        raw_temporal_nearest_warp=args.hq_raw_temporal_nearest_warp,
        raw_temporal_zero_input=args.hq_raw_temporal_zero_input,
    ).to(device)
    if args.initialize_refiner_from is not None:
        if not isinstance(model, MiohRestorerV5HQ):
            raise TypeError("initialize-refiner-from requires V5-HQ")
        load_hq_refiner_checkpoint(model, args.initialize_refiner_from)
    configure_trainable_parameters(
        model, variant=args.variant, stage=stage
    )
    if args.hq_hf_ablation:
        if not isinstance(model, MiohRestorerV5HQ):
            raise TypeError("HQ HF ablation built an incompatible model")
        # Preserve the already-deployed BasicVSR++ quality floor. The pilot
        # asks only whether the V5 refiner can learn recoverable detail.
        model.backbone.requires_grad_(False)
        if args.hq_freeze_base_head:
            model.base_head.requires_grad_(False)
        if args.hq_freeze_confidence_head:
            model.confidence_head.requires_grad_(False)
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=stage.learning_rate,
    )
    gan_enabled = args.hq_gan_weight > 0
    discriminator_classes = {
        "patch": TemporalPatchDiscriminator,
        "unet-sn": SpectralUNetDiscriminator,
    }
    discriminator = (
        discriminator_classes[args.hq_gan_discriminator_architecture](
            args.hq_gan_discriminator_channels
        ).to(device)
        if gan_enabled
        else None
    )
    discriminator_optimizer = (
        torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.hq_gan_learning_rate,
            betas=(0.0, 0.99),
        )
        if discriminator is not None
        else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=scale_gradients)
    boundary_temporal_weight = float(
        getattr(stage, "boundary_temporal_weight", 0.0)
    )
    loss_function = MiohRestorerV5Loss(
        weights=stage.loss,
        boundary_temporal_weight=boundary_temporal_weight,
        base_loss_weight_override=0.0 if args.hq_hf_ablation else None,
        candidate_loss_weight_override=0.0 if args.hq_hf_ablation else None,
        confidence_loss_weight_override=(
            args.hq_confidence_weight if args.hq_hf_ablation else None
        ),
        hf_amplitude_weight=(
            args.hq_hf_amplitude_weight if args.hq_hf_ablation else 0.0
        ),
        hf_correlation_weight=(
            args.hq_hf_correlation_weight if args.hq_hf_ablation else 0.0
        ),
        hf_local_correlation_weight=(
            args.hq_hf_local_correlation_weight if args.hq_hf_ablation else 0.0
        ),
        hf_local_correlation_patch_size=args.hq_hf_local_correlation_patch_size,
        hf_local_correlation_on_residual=(
            args.hq_hf_local_correlation_on_residual
            if args.hq_hf_ablation
            else False
        ),
        hf_residual_reconstruction_weight=(
            args.hq_hf_residual_reconstruction_weight
            if args.hq_hf_ablation
            else 0.0
        ),
        hf_correction_target_projection_weight=(
            args.hq_hf_correction_target_projection_weight
            if args.hq_hf_ablation
            else 0.0
        ),
        hf_correction_target_orthogonal_energy_weight=(
            args.hq_hf_correction_target_orthogonal_energy_weight
            if args.hq_hf_ablation
            else 0.0
        ),
        guard_ring_identity_weight=(
            args.hq_guard_ring_identity_weight if args.hq_hf_ablation else 0.0
        ),
        supervise_final_high_frequency=args.hq_hf_ablation,
    ).to(device)
    perceptual = (
        V5PerceptualLoss(
            image_size=args.perceptual_image_size,
            preserve_native_scale=args.hq_hf_ablation,
        ).to(device).eval()
        if stage.loss.perceptual
        else None
    )

    step = 0
    epoch = 0
    resume_path = args.resume
    if resume_path is None and latest.is_file() and not args.restart_stage:
        resume_path = latest
    if resume_path is not None:
        step, epoch = load_checkpoint(
            resume_path,
            model,
            ema,
            expected_stage=args.stage,
            expected_variant=args.variant,
            resume=True,
            optimizer=optimizer,
            scaler=scaler,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            resume_optimizer=not args.resume_model_only,
        )
    elif args.initialize_refiner_from is not None:
        # The backbone and refiner were deliberately initialized from two
        # independent trusted checkpoints before the optimizer was created.
        pass
    elif args.hq_hf_ablation:
        # A clean diagnostic deliberately starts the V5 branch at its native
        # zero initialization on top of the frozen deployed backbone.
        pass
    elif args.stage > 1:
        parent_path = args.initialize_from or default_parent_checkpoint(args)
        if parent_path is None or not parent_path.is_file():
            raise FileNotFoundError(
                f"completed Stage {args.stage - 1} checkpoint is required: {parent_path}"
            )
        completion = parent_path.parent / "stage-complete.json"
        if not completion.is_file():
            raise RuntimeError(f"parent stage has no completion gate: {completion}")
        load_checkpoint(
            parent_path,
            model,
            ema,
            expected_stage=args.stage,
            expected_variant=args.variant,
            resume=False,
        )

    if args.hq_gan_initialize_discriminator_from is not None:
        if discriminator is None:
            raise RuntimeError("discriminator initialization requested without GAN")
        load_discriminator_checkpoint(
            args.hq_gan_initialize_discriminator_from,
            discriminator,
        )

    train_loader, train_sampler = make_loader(
        args.train_manifest,
        output_indices=model.config.output_indices,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch=args.prefetch,
        training=True,
        seed=args.seed,
        mosaic_block_size_range=(
            (args.mosaic_block_minimum, args.mosaic_block_maximum)
            if args.mosaic_block_minimum is not None
            else None
        ),
    )
    validation_loader, _ = make_loader(
        args.validation_manifest,
        output_indices=model.config.output_indices,
        batch_size=1,
        workers=min(args.workers, 1),
        prefetch=args.prefetch,
        training=False,
        seed=args.seed,
        mosaic_block_size_range=(
            (args.mosaic_block_minimum, args.mosaic_block_maximum)
            if args.mosaic_block_minimum is not None
            else None
        ),
    )
    startup = {
        "event": "start",
        "stage": args.stage,
        "stage_name": stage.name,
        "variant": args.variant,
        "device": str(device),
        "parameters": parameter_count(model),
        "training_output_indices": model.config.output_indices,
        "steps": total_steps,
        "starting_step": step,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "external_teacher": False,
        "basicvsrpp_backbone_initialization": (
            str(args.basicvsrpp_checkpoint) if args.variant == "hq" else None
        ),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "fresh_optimizer_per_stage": True,
        "hq_hf_ablation": args.hq_hf_ablation,
        "hq_freeze_base_head": args.hq_freeze_base_head,
        "hq_freeze_confidence_head": args.hq_freeze_confidence_head,
        "hq_raw_temporal_candidates": args.hq_raw_temporal_candidates,
        "hq_raw_temporal_encoder_channels": args.hq_raw_temporal_encoder_channels,
        "hq_raw_temporal_nearest_warp": args.hq_raw_temporal_nearest_warp,
        "hq_raw_temporal_zero_input": args.hq_raw_temporal_zero_input,
        "hq_texture_effective_mask": args.hq_texture_effective_mask,
        "hq_texture_effective_mask_feather_radius": (
            args.hq_texture_effective_mask_feather_radius
        ),
        "hq_gan_weight": args.hq_gan_weight,
        "hq_gan_loss": args.hq_gan_loss,
        "hq_gan_generator_hinge_weight": args.hq_gan_generator_hinge_weight,
        "hq_gan_feature_matching_weight": args.hq_gan_feature_matching_weight,
        "hq_gan_start_step": args.hq_gan_start_step,
        "hq_gan_discriminator_pretrain_until_step": (
            args.hq_gan_discriminator_pretrain_until_step
        ),
        "hq_gan_warmup_steps": args.hq_gan_warmup_steps,
        "hq_gan_learning_rate": args.hq_gan_learning_rate,
        "hq_gan_discriminator_channels": args.hq_gan_discriminator_channels,
        "hq_gan_discriminator_architecture": (
            args.hq_gan_discriminator_architecture
        ),
        "hq_gan_freeze_discriminator": args.hq_gan_freeze_discriminator,
        "hq_gan_image_size": args.hq_gan_image_size,
        "hq_gan_temporal": args.hq_gan_temporal,
        "hq_gan_normalize_secondary_rms": args.hq_gan_normalize_secondary_rms,
        "hq_gan_candidate_primary": args.hq_gan_candidate_primary,
        "hq_gan_initialize_discriminator_from": (
            str(args.hq_gan_initialize_discriminator_from)
            if args.hq_gan_initialize_discriminator_from is not None
            else None
        ),
        "initialize_refiner_from": str(args.initialize_refiner_from)
        if args.initialize_refiner_from
        else None,
        "mosaic_block_size_range": (
            [args.mosaic_block_minimum, args.mosaic_block_maximum]
            if args.mosaic_block_minimum is not None
            else None
        ),
    }
    print(json.dumps(startup, ensure_ascii=False, indent=2), flush=True)
    append_json(metrics_path, startup)
    if args.hq_hf_ablation and step == 0:
        initial = validate(
            ema,
            validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            limit=args.validation_batches,
            use_loss_masks=args.hq_hf_ablation,
            use_texture_effective_mask=args.hq_texture_effective_mask,
            texture_effective_mask_feather_radius=(
                args.hq_texture_effective_mask_feather_radius
            ),
            hf_local_correlation_patch_size=args.hq_hf_local_correlation_patch_size,
        )
        initial_record = {"event": "validation", "step": 0, **initial}
        print(json.dumps(initial_record, ensure_ascii=False, indent=2), flush=True)
        append_json(metrics_path, initial_record)
        model.train()
    optimizer.zero_grad(set_to_none=True)
    session_started = time.perf_counter()
    session_start_step = step

    try:
        while step < total_steps:
            train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            while step < total_steps:
                micro_batches = []
                for _ in range(args.accumulate):
                    try:
                        micro_batches.append(next(iterator))
                    except StopIteration:
                        break
                if not micro_batches:
                    break
                prepared_batches = []
                for batch in micro_batches:
                    inputs, targets, masks, loss_masks = move_batch(batch, device)
                    if bool(torch.any(loss_masks > 0)):
                        prepared_batches.append((inputs, targets, masks, loss_masks))
                if not prepared_batches:
                    continue
                accumulated: dict[str, float] = defaultdict(float)
                next_step = step + 1
                current_gan_weight = hq_gan_weight(args, next_step)
                discriminator_pretraining = bool(
                    current_gan_weight > 0
                    and next_step
                    <= args.hq_gan_discriminator_pretrain_until_step
                )
                for inputs, targets, masks, loss_masks in prepared_batches:
                    supervision_masks = loss_masks if args.hq_hf_ablation else masks
                    guard_ring_mask = (
                        (masks - loss_masks).clamp(0.0, 1.0)
                        if args.hq_hf_ablation
                        else None
                    )
                    sources = inputs[:, list(model.config.output_indices), :3]
                    with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                        texture_mask = (
                            feather_texture_compositor_mask(
                                loss_masks,
                                masks,
                                radius=args.hq_texture_effective_mask_feather_radius,
                            )
                            if args.hq_texture_effective_mask
                            else None
                        )
                        restored, confidence, base, texture = model.forward_components(
                            inputs,
                            texture_compositor_mask=texture_mask,
                        )
                        perceptual_value = (
                            perceptual(restored, targets, masks)
                            if perceptual is not None
                            else None
                        )
                        temporal_values = (None, None, None)
                        if stage.loss.temporal or boundary_temporal_weight:
                            temporal_values = flow_aligned_temporal_tensors(
                                restored, targets, masks
                            )
                        loss, stats = loss_function(
                            restored,
                            confidence,
                            base,
                            texture,
                            targets,
                            sources,
                            supervision_masks,
                            aligned_previous_restored=temporal_values[0],
                            aligned_previous_target=temporal_values[1],
                            temporal_valid=temporal_values[2],
                            perceptual=perceptual_value,
                            guard_ring_mask=guard_ring_mask,
                            hf_reference=(
                                sources + masks * base.detach()
                                if (
                                    args.hq_hf_local_correlation_on_residual
                                    or args.hq_hf_residual_reconstruction_weight > 0
                                    or args.hq_hf_correction_target_projection_weight > 0
                                    or args.hq_hf_correction_target_orthogonal_energy_weight > 0
                                )
                                else None
                            ),
                        )
                        exact_motion_weight = float(
                            getattr(stage, "exact_motion_weight", 0.0)
                        )
                        if exact_motion_weight:
                            exact, exact_stats = known_motion_alignment_loss(
                                model,
                                inputs,
                                maximum_translation=args.known_motion_maximum,
                            )
                            loss = loss + exact_motion_weight * exact
                            stats.update(exact_stats)
                        natural_motion_weight = float(
                            getattr(stage, "natural_motion_weight", 0.0)
                        )
                        feature_consistency_weight = float(
                            getattr(stage, "feature_consistency_weight", 0.0)
                        )
                        if natural_motion_weight or feature_consistency_weight:
                            natural, feature, natural_stats = natural_alignment_losses(
                                model, inputs
                            )
                            loss = (
                                loss
                                + natural_motion_weight * natural
                                + feature_consistency_weight * feature
                            )
                            stats.update(natural_stats)
                        discriminator_loss = restored.new_zeros(())
                        generator_adversarial = restored.new_zeros(())
                        generator_feature_matching = restored.new_zeros(())
                        generator_gan_objective = restored.new_zeros(())
                        discriminator_grad_norm = restored.new_zeros(())
                        discriminator_real_score = restored.new_zeros(())
                        discriminator_fake_score = restored.new_zeros(())
                        generator_fake_score = restored.new_zeros(())
                        if current_gan_weight > 0:
                            if (
                                discriminator is None
                                or discriminator_optimizer is None
                            ):
                                raise RuntimeError(
                                    "HQ GAN is active without a discriminator"
                                )
                            if args.hq_gan_freeze_discriminator:
                                discriminator.eval()
                                set_requires_grad(discriminator, False)
                            else:
                                discriminator.train()
                                set_requires_grad(discriminator, True)
                                discriminator_optimizer.zero_grad(set_to_none=True)
                            discriminator_context = (
                                torch.no_grad()
                                if args.hq_gan_freeze_discriminator
                                else torch.enable_grad()
                            )
                            with discriminator_context, torch.autocast(
                                device_type=device.type, enabled=False
                            ):
                                real_input = roi_temporal_discriminator_input(
                                    targets.float(),
                                    targets.float(),
                                    supervision_masks.float(),
                                    frame_stride=args.hq_gan_frame_stride,
                                    image_size=args.hq_gan_image_size,
                                    crop_padding=args.hq_gan_crop_padding,
                                    minimum_crop_size=(
                                        args.hq_gan_minimum_crop_size
                                    ),
                                    include_motion=args.hq_gan_temporal,
                                    normalize_secondary_rms=(
                                        args.hq_gan_normalize_secondary_rms
                                    ),
                                    condition_on_target_rgb=(
                                        not args.hq_gan_candidate_primary
                                    ),
                                )
                                fake_input = roi_temporal_discriminator_input(
                                    restored.detach().float(),
                                    targets.float(),
                                    supervision_masks.float(),
                                    frame_stride=args.hq_gan_frame_stride,
                                    image_size=args.hq_gan_image_size,
                                    crop_padding=args.hq_gan_crop_padding,
                                    minimum_crop_size=(
                                        args.hq_gan_minimum_crop_size
                                    ),
                                    include_motion=args.hq_gan_temporal,
                                    normalize_secondary_rms=(
                                        args.hq_gan_normalize_secondary_rms
                                    ),
                                    condition_on_target_rgb=(
                                        not args.hq_gan_candidate_primary
                                    ),
                                )
                                if not real_input.shape[0]:
                                    raise RuntimeError(
                                        "HQ GAN batch contains no ROI pairs"
                                    )
                                real_logits = discriminator(real_input)
                                fake_logits = discriminator(fake_input)
                                discriminator_weights = (
                                    discriminator_roi_patch_weights(
                                        real_input,
                                        real_logits,
                                    )
                                )
                                discriminator_weight_sum = (
                                    discriminator_weights.sum().clamp_min(1.0)
                                )
                                discriminator_real_score = (
                                    real_logits * discriminator_weights
                                ).sum() / discriminator_weight_sum
                                discriminator_fake_score = (
                                    fake_logits * discriminator_weights
                                ).sum() / discriminator_weight_sum
                                if args.hq_gan_loss == "rpgan":
                                    discriminator_loss = (
                                        discriminator_relativistic_pair_loss(
                                            real_logits,
                                            fake_logits,
                                            discriminator_weights,
                                        )
                                    )
                                else:
                                    discriminator_loss = discriminator_hinge_loss(
                                        real_logits,
                                        fake_logits,
                                        discriminator_weights,
                                    )
                            if not args.hq_gan_freeze_discriminator:
                                discriminator_loss.backward()
                                discriminator_grad_norm = (
                                    torch.nn.utils.clip_grad_norm_(
                                        discriminator.parameters(),
                                        args.gradient_clip,
                                    )
                                )
                                discriminator_optimizer.step()

                            if not discriminator_pretraining:
                                set_requires_grad(discriminator, False)
                                with torch.autocast(
                                    device_type=device.type,
                                    enabled=False,
                                ):
                                    generator_input = (
                                        roi_temporal_discriminator_input(
                                            restored.float(),
                                            targets.float(),
                                            supervision_masks.float(),
                                            frame_stride=args.hq_gan_frame_stride,
                                            image_size=args.hq_gan_image_size,
                                            crop_padding=args.hq_gan_crop_padding,
                                            minimum_crop_size=(
                                                args.hq_gan_minimum_crop_size
                                            ),
                                            include_motion=args.hq_gan_temporal,
                                            normalize_secondary_rms=(
                                                args.hq_gan_normalize_secondary_rms
                                            ),
                                            condition_on_target_rgb=(
                                                not args.hq_gan_candidate_primary
                                            ),
                                        )
                                    )
                                    if args.hq_gan_feature_matching_weight > 0:
                                        with torch.no_grad():
                                            _, real_features = (
                                                discriminator.forward_features(real_input)
                                            )
                                        generator_logits, generator_features = (
                                            discriminator.forward_features(generator_input)
                                        )
                                        generator_feature_matching = (
                                            discriminator_feature_matching_loss(
                                                real_features,
                                                generator_features,
                                                generator_input,
                                            )
                                        )
                                    else:
                                        generator_logits = discriminator(generator_input)
                                    generator_weights = (
                                        discriminator_roi_patch_weights(
                                            generator_input,
                                            generator_logits,
                                        )
                                    )
                                    generator_fake_score = (
                                        generator_logits * generator_weights
                                    ).sum() / generator_weights.sum().clamp_min(1.0)
                                    if args.hq_gan_loss == "rpgan":
                                        with torch.no_grad():
                                            generator_real_logits = discriminator(
                                                real_input
                                            )
                                        generator_adversarial = (
                                            generator_relativistic_pair_loss(
                                                generator_real_logits,
                                                generator_logits,
                                                generator_weights,
                                            )
                                        )
                                    else:
                                        generator_adversarial = generator_hinge_loss(
                                            generator_logits,
                                            generator_weights,
                                        )
                                    generator_gan_objective = (
                                        args.hq_gan_generator_hinge_weight
                                        * generator_adversarial
                                        + args.hq_gan_feature_matching_weight
                                        * generator_feature_matching
                                    )
                                loss = (
                                    loss
                                    + current_gan_weight
                                    * generator_gan_objective.to(loss.dtype)
                                )
                        stats.update(
                            {
                                "generator_adversarial": float(
                                    generator_adversarial.detach()
                                ),
                                "generator_feature_matching": float(
                                    generator_feature_matching.detach()
                                ),
                                "generator_gan_objective": float(
                                    generator_gan_objective.detach()
                                ),
                                "discriminator": float(
                                    discriminator_loss.detach()
                                ),
                                "discriminator_grad_norm": float(
                                    discriminator_grad_norm.detach()
                                ),
                                "discriminator_real_score": float(
                                    discriminator_real_score.detach()
                                ),
                                "discriminator_fake_score": float(
                                    discriminator_fake_score.detach()
                                ),
                                "generator_fake_score": float(
                                    generator_fake_score.detach()
                                ),
                                "gan_weight": current_gan_weight,
                                "discriminator_pretraining": float(
                                    discriminator_pretraining
                                ),
                            }
                        )
                        stats["total_with_stage_supervision"] = float(loss.detach())
                        scaled = loss / len(prepared_batches)
                    if not discriminator_pretraining:
                        scaler.scale(scaled).backward()
                    for name, value in stats.items():
                        accumulated[name] += float(value) / len(prepared_batches)

                warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
                if args.variant == "hq" and args.hq_fixed_learning_rate is not None:
                    learning_rate = args.hq_fixed_learning_rate
                else:
                    learning_rate = (
                        hq_learning_rate(stage, next_step, total_steps, warmup_steps)
                        if args.variant == "hq"
                        else stage_learning_rate(
                            stage,
                            next_step,
                            total_steps=total_steps,
                            warmup_steps=warmup_steps,
                        )
                    )
                for group in optimizer.param_groups:
                    group["lr"] = 0.0 if discriminator_pretraining else learning_rate
                if not discriminator_pretraining:
                    if args.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if not discriminator_pretraining:
                    update_ema(ema, model, args.ema_decay)
                step = next_step

                if step % args.log_every == 0 or step == 1:
                    elapsed = time.perf_counter() - session_started
                    record = {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "learning_rate": learning_rate,
                        "seconds_per_step": elapsed / max(step - session_start_step, 1),
                        **accumulated,
                        **memory_stats(device),
                    }
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    append_json(metrics_path, record)
                if step % args.validate_every == 0 or step == total_steps:
                    if args.hq_hf_ablation:
                        raw_result = validate(
                            model,
                            validation_loader,
                            device=device,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                            limit=args.validation_batches,
                            use_loss_masks=args.hq_hf_ablation,
                            use_texture_effective_mask=args.hq_texture_effective_mask,
                            texture_effective_mask_feather_radius=(
                                args.hq_texture_effective_mask_feather_radius
                            ),
                            hf_local_correlation_patch_size=args.hq_hf_local_correlation_patch_size,
                        )
                        raw_record = {
                            "event": "validation_raw",
                            "step": step,
                            **raw_result,
                        }
                        print(
                            json.dumps(raw_record, ensure_ascii=False, indent=2),
                            flush=True,
                        )
                        append_json(metrics_path, raw_record)
                        early_stop_reason = (
                            hq_hf_early_stop_reason(raw_result)
                            if args.hq_hf_early_stop
                            else None
                        )
                    else:
                        early_stop_reason = None
                    result = validate(
                        ema,
                        validation_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        limit=args.validation_batches,
                        use_loss_masks=args.hq_hf_ablation,
                        use_texture_effective_mask=args.hq_texture_effective_mask,
                        texture_effective_mask_feather_radius=(
                            args.hq_texture_effective_mask_feather_radius
                        ),
                        hf_local_correlation_patch_size=args.hq_hf_local_correlation_patch_size,
                    )
                    record = {"event": "validation", "step": step, **result}
                    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
                    append_json(metrics_path, record)
                    model.train()
                    clear_accelerator_cache(device)
                    if early_stop_reason is not None:
                        saved = save_checkpoint(
                            model,
                            ema,
                            optimizer,
                            scaler,
                            args,
                            work_dir,
                            step=step,
                            epoch=epoch,
                            discriminator=discriminator,
                            discriminator_optimizer=discriminator_optimizer,
                        )
                        stopped = {
                            "event": "early_stop",
                            "step": step,
                            "reason": early_stop_reason,
                            "checkpoint": str(saved),
                        }
                        print(json.dumps(stopped, ensure_ascii=False, indent=2), flush=True)
                        append_json(metrics_path, stopped)
                        return 0
                if step % args.save_every == 0 or step == total_steps:
                    saved = save_checkpoint(
                        model,
                        ema,
                        optimizer,
                        scaler,
                        args,
                        work_dir,
                        step=step,
                        epoch=epoch,
                        discriminator=discriminator,
                        discriminator_optimizer=discriminator_optimizer,
                    )
                    print(f"saved: {saved}", flush=True)
            epoch += 1
    except KeyboardInterrupt:
        saved = save_checkpoint(
            model,
            ema,
            optimizer,
            scaler,
            args,
            work_dir,
            step=step,
            epoch=epoch,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
        )
        print(f"interrupted safely; saved: {saved}", flush=True)
        return 130

    completion = {
        "event": "complete",
        "stage": args.stage,
        "stage_name": stage.name,
        "step": step,
        "elapsed_seconds": time.perf_counter() - session_started,
        "checkpoint": str(latest),
    }
    append_json(metrics_path, completion)
    (work_dir / "stage-complete.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
