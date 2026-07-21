# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Train MiohRestorerV2 or the Core AI aligned MiohRestorerV3 model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lada.models.mioh_restorer import (
    MaskedVGG16PerceptualLoss,
    MiohRestorerV2,
    MiohRestorerV3,
    TemporalPatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    masked_charbonnier_loss,
    masked_multiscale_structural_loss,
    masked_psnr,
    restoration_loss,
    temporal_discriminator_input,
)
from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet
from lada.models.basicvsrpp.activation_analysis import (
    AlignmentActivation,
    AlignmentCapturePolicy,
    BasicVSRPPActivationAnalyzer,
)
from lada.models.mioh_restorer.distillation import (
    roi_alignment_kl_loss,
    roi_confidence_loss,
    roi_feature_energy_loss,
    teacher_source_confidence,
    teacher_hierarchical_shift_distributions,
    teacher_shift_distribution,
)
from lada.models.mioh_restorer.training_dataset import MiohRestorationDataset
from lada.models.mioh_restorer.training_memory import (
    MemorySnapshot,
    MemoryThresholds,
    TrainingMemoryGuard,
    release_device_memory,
    tree_to_cpu,
)


MEMORY_EMERGENCY_EXIT_CODE = 75


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata-root", type=Path, nargs="+", required=True)
    parser.add_argument("--val-metadata-root", type=Path, nargs="+")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from-v2", type=Path)
    parser.add_argument("--initialize-from-v3", type=Path)
    parser.add_argument("--reset-optimizers", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-version", type=int, choices=(2, 3), default=2)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--window-frames", type=int, default=24)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--fusion-full-channels", type=int, default=32)
    parser.add_argument("--fusion-half-channels", type=int, default=64)
    parser.add_argument("--fusion-quarter-channels", type=int, default=96)
    parser.add_argument("--detail-scale", type=float, default=0.25)
    parser.add_argument("--encoder-blocks", type=int, default=5)
    parser.add_argument("--reconstruction-blocks", type=int, default=5)
    parser.add_argument("--alignment-radius", type=int, default=1)
    parser.add_argument("--first-order-dilation", type=int, default=2)
    parser.add_argument("--second-order-dilation", type=int, default=4)
    parser.add_argument("--alignment-key-channels", type=int, default=16)
    parser.add_argument("--alignment-groups", type=int, default=1)
    parser.add_argument(
        "--hierarchical-alignment-dilations",
        type=int,
        nargs="*",
        default=(),
    )
    parser.add_argument("--alignment-temperature", type=float, default=1.0)

    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--teacher-feature-weight", type=float, default=0.0)
    parser.add_argument("--teacher-alignment-weight", type=float, default=0.0)
    parser.add_argument("--teacher-distill-calls", type=int, default=2)
    parser.add_argument("--teacher-shift-temperature", type=float, default=1.0)
    parser.add_argument(
        "--teacher-fp16", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--gradient-weight", type=float, default=0.30)
    parser.add_argument("--temporal-weight", type=float, default=0.20)
    parser.add_argument("--high-frequency-weight", type=float, default=0.15)
    parser.add_argument("--perceptual-weight", type=float, default=0.02)
    parser.add_argument("--perceptual-frame-stride", type=int, default=4)
    parser.add_argument("--perceptual-image-size", type=int, default=224)
    parser.add_argument("--structural-weight", type=float, default=0.10)
    parser.add_argument("--structural-frame-stride", type=int, default=2)
    parser.add_argument("--directional-aux-weight", type=float, default=0.15)
    parser.add_argument("--direction-consistency-weight", type=float, default=0.02)

    parser.add_argument("--gan-weight", type=float, default=0.002)
    parser.add_argument("--gan-start-step", type=int, default=10_000)
    parser.add_argument("--gan-learning-rate", type=float, default=5e-5)
    parser.add_argument("--gan-warmup-steps", type=int, default=500)
    parser.add_argument("--gan-frame-stride", type=int, default=4)
    parser.add_argument("--gan-image-size", type=int, default=192)
    parser.add_argument("--discriminator-channels", type=int, default=32)

    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-latest-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument(
        "--validate-at-start",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--memory-log-every", type=int, default=20)
    parser.add_argument("--memory-warning-ratio", type=float, default=0.80)
    parser.add_argument("--memory-critical-ratio", type=float, default=0.92)
    parser.add_argument(
        "--memory-warning-available-gib", type=float, default=8.0
    )
    parser.add_argument(
        "--memory-critical-available-gib", type=float, default=4.0
    )
    parser.add_argument(
        "--memory-emergency-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--degrade", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--horizontal-flip", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    return parser.parse_args(argv)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "window_frames": args.window_frames,
        "chunk_frames": args.chunk_frames,
        "image_size": args.image_size,
        "channels": args.channels,
        "blocks": args.blocks,
        "fusion_full_channels": args.fusion_full_channels,
        "fusion_half_channels": args.fusion_half_channels,
        "fusion_quarter_channels": args.fusion_quarter_channels,
        "encoder_blocks": args.encoder_blocks,
        "reconstruction_blocks": args.reconstruction_blocks,
        "first_order_dilation": args.first_order_dilation,
        "second_order_dilation": args.second_order_dilation,
        "alignment_key_channels": args.alignment_key_channels,
        "alignment_groups": args.alignment_groups,
        "alignment_temperature": args.alignment_temperature,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "save_every": args.save_every,
        "save_latest_every": args.save_latest_every,
        "validate_every": args.validate_every,
        "validation_batches": args.validation_batches,
        "log_every": args.log_every,
        "memory_log_every": args.memory_log_every,
        "prefetch_factor": args.prefetch_factor,
        "perceptual_frame_stride": args.perceptual_frame_stride,
        "perceptual_image_size": args.perceptual_image_size,
        "structural_frame_stride": args.structural_frame_stride,
        "gan_learning_rate": args.gan_learning_rate,
        "gan_frame_stride": args.gan_frame_stride,
        "gan_image_size": args.gan_image_size,
        "discriminator_channels": args.discriminator_channels,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        args.workers < 0
        or args.validation_workers < 0
        or args.warmup_steps < 0
        or args.gan_warmup_steps < 0
        or args.alignment_radius < 0
    ):
        raise ValueError("workers and warmup steps must not be negative")
    if args.window_frames % args.chunk_frames:
        raise ValueError("window_frames must be divisible by chunk_frames")
    if args.image_size % 4:
        raise ValueError("image_size must be divisible by 4")
    if args.minimum_learning_rate > args.learning_rate:
        raise ValueError("minimum_learning_rate must not exceed learning_rate")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema_decay must be between zero and one")
    loss_weights = {
        "gradient_weight": args.gradient_weight,
        "temporal_weight": args.temporal_weight,
        "high_frequency_weight": args.high_frequency_weight,
        "perceptual_weight": args.perceptual_weight,
        "structural_weight": args.structural_weight,
        "directional_aux_weight": args.directional_aux_weight,
        "direction_consistency_weight": args.direction_consistency_weight,
        "gan_weight": args.gan_weight,
        "teacher_weight": args.teacher_weight,
        "teacher_feature_weight": args.teacher_feature_weight,
        "teacher_alignment_weight": args.teacher_alignment_weight,
    }
    for name, value in loss_weights.items():
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if args.gan_start_step < 0:
        raise ValueError("gan_start_step must not be negative")
    teacher_signal = (
        args.teacher_weight
        + args.teacher_feature_weight
        + args.teacher_alignment_weight
    )
    if bool(args.teacher_checkpoint) != (teacher_signal > 0):
        raise ValueError(
            "teacher-checkpoint and a positive teacher loss weight must be used together"
        )
    if args.teacher_distill_calls <= 0:
        raise ValueError("teacher-distill-calls must be positive")
    if args.teacher_shift_temperature <= 0:
        raise ValueError("teacher-shift-temperature must be positive")
    if any(item <= 0 for item in args.hierarchical_alignment_dilations):
        raise ValueError("hierarchical alignment dilations must be positive")
    if (
        args.model_version != 3
        and (args.teacher_feature_weight > 0 or args.teacher_alignment_weight > 0)
    ):
        raise ValueError("intermediate teacher distillation requires model-version 3")
    initialization_sources = sum(
        item is not None
        for item in (
            args.resume,
            args.initialize_from_v2,
            args.initialize_from_v3,
        )
    )
    if initialization_sources > 1:
        raise ValueError(
            "resume, initialize-from-v2 and initialize-from-v3 are mutually exclusive"
        )
    if args.initialize_from_v2 is not None and args.model_version != 3:
        raise ValueError("initialize-from-v2 requires model-version 3")
    if args.initialize_from_v3 is not None and args.model_version != 3:
        raise ValueError("initialize-from-v3 requires model-version 3")
    MemoryThresholds(
        warning_mps_ratio=args.memory_warning_ratio,
        critical_mps_ratio=args.memory_critical_ratio,
        warning_system_available_gib=args.memory_warning_available_gib,
        critical_system_available_gib=args.memory_critical_available_gib,
    )


def make_loader(
    roots: list[Path],
    args: argparse.Namespace,
    *,
    training: bool,
) -> DataLoader:
    workers = args.workers if training else args.validation_workers
    dataset = MiohRestorationDataset(
        roots,
        sequence_frames=args.window_frames,
        image_size=args.image_size,
        degrade=args.degrade,
        horizontal_flip=args.horizontal_flip if training else False,
        deterministic=not training,
        limit=args.limit_train_samples if training else args.limit_val_samples,
    )
    loader_options = {
        "batch_size": args.batch_size if training else 1,
        "shuffle": training,
        "num_workers": workers,
        "persistent_workers": training and workers > 0,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": training and len(dataset) >= args.batch_size,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    return DataLoader(
        dataset,
        **loader_options,
    )


def build_model(args: argparse.Namespace) -> MiohRestorerV2 | MiohRestorerV3:
    if args.model_version == 2:
        return MiohRestorerV2(
            window_frames=args.window_frames,
            chunk_frames=args.chunk_frames,
            channels=args.channels,
            num_blocks=args.blocks,
            fusion_full_channels=args.fusion_full_channels,
            fusion_half_channels=args.fusion_half_channels,
            fusion_quarter_channels=args.fusion_quarter_channels,
            detail_scale=args.detail_scale,
        )
    return MiohRestorerV3(
        window_frames=args.window_frames,
        channels=args.channels,
        num_blocks=args.blocks,
        encoder_blocks=args.encoder_blocks,
        reconstruction_blocks=args.reconstruction_blocks,
        alignment_radius=args.alignment_radius,
        first_order_dilation=args.first_order_dilation,
        second_order_dilation=args.second_order_dilation,
        alignment_key_channels=args.alignment_key_channels,
        alignment_groups=args.alignment_groups,
        hierarchical_alignment_dilations=(
            args.hierarchical_alignment_dilations
        ),
        alignment_temperature=args.alignment_temperature,
        detail_scale=args.detail_scale,
    )


def model_config(
    model: MiohRestorerV2 | MiohRestorerV3,
    args: argparse.Namespace,
) -> dict:
    config = {
        "version": args.model_version,
        "window_frames": model.window_frames,
        "channels": model.channels,
        "num_blocks": model.num_blocks,
        "image_size": args.image_size,
        "detail_scale": model.detail_scale,
        "gradient_weight": args.gradient_weight,
        "temporal_weight": args.temporal_weight,
        "high_frequency_weight": args.high_frequency_weight,
        "perceptual_weight": args.perceptual_weight,
        "structural_weight": args.structural_weight,
        "directional_aux_weight": args.directional_aux_weight,
        "direction_consistency_weight": args.direction_consistency_weight,
        "gan_weight": args.gan_weight,
        "gan_start_step": args.gan_start_step,
        "teacher_weight": args.teacher_weight,
        "teacher_feature_weight": args.teacher_feature_weight,
        "teacher_alignment_weight": args.teacher_alignment_weight,
        "teacher_distill_calls": args.teacher_distill_calls,
        "teacher_shift_temperature": args.teacher_shift_temperature,
        "teacher_checkpoint": (
            str(args.teacher_checkpoint.resolve())
            if args.teacher_checkpoint is not None
            else None
        ),
        "teacher_checkpoint_sha256": getattr(
            args, "teacher_checkpoint_sha256", None
        ),
        "training_memory": {
            "workers": args.workers,
            "validation_workers": args.validation_workers,
            "prefetch_factor": args.prefetch_factor,
            "memory_log_every": args.memory_log_every,
            "warning_mps_ratio": args.memory_warning_ratio,
            "critical_mps_ratio": args.memory_critical_ratio,
            "warning_system_available_gib": args.memory_warning_available_gib,
            "critical_system_available_gib": args.memory_critical_available_gib,
            "emergency_stop": args.memory_emergency_stop,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
    }
    if isinstance(model, MiohRestorerV2):
        config.update(
            {
                "chunk_frames": model.chunk_frames,
                "fusion_full_channels": model.fusion_full_channels,
                "fusion_half_channels": model.fusion_half_channels,
                "fusion_quarter_channels": model.fusion_quarter_channels,
            }
        )
    else:
        config.update(
            {
                "encoder_blocks": model.encoder_blocks,
                "architecture_revision": model.architecture_revision,
                "reconstruction_blocks": model.reconstruction_blocks,
                "alignment_radius": model.alignment_radius,
                "first_order_dilation": model.first_order_dilation,
                "second_order_dilation": model.second_order_dilation,
                "alignment_key_channels": model.alignment_key_channels,
                "alignment_groups": model.alignment_groups,
                "hierarchical_alignment_dilations": list(
                    model.hierarchical_alignment_dilations
                ),
                "alignment_temperature": model.alignment_temperature,
            }
        )
    return config


def checkpoint_prefix(args: argparse.Namespace) -> str:
    suffix = "1" if args.hierarchical_alignment_dilations else ""
    return f"mioh-restorer-v{args.model_version}{suffix}"


def load_v2_for_initialization(path: Path) -> MiohRestorerV2:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if int(config.get("version", 0)) != 2:
        raise ValueError(f"initialization checkpoint is not V2: {path}")
    model = MiohRestorerV2(
        window_frames=int(config["window_frames"]),
        chunk_frames=int(config["chunk_frames"]),
        channels=int(config["channels"]),
        num_blocks=int(config["num_blocks"]),
        fusion_full_channels=int(config["fusion_full_channels"]),
        fusion_half_channels=int(config["fusion_half_channels"]),
        fusion_quarter_channels=int(config["fusion_quarter_channels"]),
        detail_scale=float(config["detail_scale"]),
    )
    state = payload.get("ema_state_dict", payload["state_dict"])
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_v3_state_for_initialization(
    path: Path,
) -> tuple[dict, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if int(config.get("version", 0)) != 3:
        raise ValueError(f"initialization checkpoint is not V3: {path}")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"initialization checkpoint has no raw state: {path}")
    return config, state


def load_basicvsrpp_teacher(
    path: Path,
    device: torch.device,
    *,
    fp16: bool,
) -> BasicVSRPlusPlusGanNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unexpected BasicVSR++ checkpoint: {path}")
    prefix = (
        "generator_ema."
        if any(str(key).startswith("generator_ema.") for key in checkpoint)
        else "generator."
    )
    state = {
        str(key)[len(prefix) :]: value
        for key, value in checkpoint.items()
        if str(key).startswith(prefix)
    }
    if not state:
        raise ValueError(f"BasicVSR++ generator weights not found: {path}")
    teacher = BasicVSRPlusPlusGanNet(
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,
    )
    teacher.load_state_dict(state, strict=True)
    teacher.requires_grad_(False).eval().to(device)
    if fp16:
        if device.type == "cpu":
            raise ValueError("teacher-fp16 requires an accelerator device")
        teacher.half()
    return teacher


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rng_state_payload() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.random.get_rng_state(),
    }


def restore_rng_state(payload: dict) -> None:
    random.setstate(payload["python"])
    numpy_state = payload["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_state["state"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.random.set_rng_state(payload["torch"].cpu())


def move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def cosine_learning_rate(
    *,
    step: int,
    total_steps: int,
    base: float,
    minimum: float,
    warmup_steps: int,
) -> float:
    if warmup_steps and step <= warmup_steps:
        return base * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def gan_learning_rate(args: argparse.Namespace, step: int) -> float:
    active_step = step - args.gan_start_step + 1
    if active_step <= 0:
        return 0.0
    if args.gan_warmup_steps and active_step <= args.gan_warmup_steps:
        return args.gan_learning_rate * active_step / args.gan_warmup_steps
    return args.gan_learning_rate


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(
        ema.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def checkpoint_payload(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    discriminator: TemporalPatchDiscriminator,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    *,
    stop_reason: str | None = None,
) -> dict:
    payload = {
        "state_dict": tree_to_cpu(model.state_dict()),
        "ema_state_dict": tree_to_cpu(ema.state_dict()),
        "discriminator_state_dict": tree_to_cpu(discriminator.state_dict()),
        "optimizer": tree_to_cpu(optimizer.state_dict()),
        "discriminator_optimizer": tree_to_cpu(
            discriminator_optimizer.state_dict()
        ),
        "step": step,
        "trained": step > 0,
        "config": model_config(model, args),
        "rng_state": rng_state_payload(),
    }
    if stop_reason is not None:
        payload["stop_reason"] = stop_reason
    return payload


def save_checkpoint(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    discriminator: TemporalPatchDiscriminator,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    *,
    archive: bool,
    emergency: bool = False,
) -> Path:
    payload = checkpoint_payload(
        model,
        ema,
        discriminator,
        optimizer,
        discriminator_optimizer,
        args,
        step,
        stop_reason="memory_pressure" if emergency else None,
    )
    prefix = checkpoint_prefix(args)
    latest = args.work_dir / f"{prefix}-latest.pth"
    latest_temporary = latest.with_suffix(".tmp")
    torch.save(payload, latest_temporary)
    latest_temporary.replace(latest)
    if not archive:
        return latest
    suffix = "emergency-step" if emergency else "step"
    output = args.work_dir / f"{prefix}-{suffix}-{step:07d}.pth"
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def distillation_capture_plan(
    model: MiohRestorerV3,
    step: int,
    requested_calls: int,
) -> tuple[str, frozenset[int], int]:
    """Cycle branches and choose evenly spaced, memory-bounded calls."""

    branch = model.BRANCHES[(step - 1) % len(model.BRANCHES)][0]
    available_calls = model.window_frames - 1
    call_count = min(requested_calls, available_calls)
    stride = max(1, available_calls // call_count)
    calls = frozenset(
        list(range(0, available_calls, stride))[:call_count]
    )
    return branch, calls, stride


def calculate_intermediate_teacher_losses(
    model: MiohRestorerV3,
    diagnostics: list[dict[str, object]],
    teacher_activations: list[AlignmentActivation],
    masks: torch.Tensor,
    *,
    branch: str,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distill selected teacher alignment behavior without channel adapters."""

    zero = masks.new_zeros(())
    teacher_by_call = {
        activation.call_index: activation
        for activation in teacher_activations
        if activation.branch == branch
    }
    propagation = model.propagation[branch]
    feature_losses: list[torch.Tensor] = []
    alignment_losses: list[torch.Tensor] = []

    def append_shift_losses(
        weights: object,
        alignment: torch.nn.Module,
        activation: AlignmentActivation,
        roi_mask: torch.Tensor,
        *,
        source: int,
    ) -> None:
        if isinstance(weights, torch.Tensor):
            offsets = getattr(alignment, "offsets", None)
            if offsets is None:
                raise TypeError("single-stage alignment has no offsets")
            target = teacher_shift_distribution(
                activation.offset,
                activation.mask,
                offsets,
                source=source,
                temperature=temperature,
            )
            alignment_losses.append(
                roi_alignment_kl_loss(weights, target, roi_mask)
            )
            return
        if isinstance(weights, tuple) and all(
            isinstance(item, torch.Tensor) for item in weights
        ):
            stage_offsets = getattr(alignment, "stage_offsets", None)
            if stage_offsets is None:
                raise TypeError("hierarchical alignment has no stage offsets")
            targets = teacher_hierarchical_shift_distributions(
                activation.offset,
                activation.mask,
                stage_offsets,
                source=source,
                temperature=temperature,
            )
            if len(weights) != len(targets):
                raise ValueError("student and teacher hierarchy lengths differ")
            alignment_losses.extend(
                roi_alignment_kl_loss(student, target, roi_mask)
                for student, target in zip(weights, targets, strict=True)
            )
    for diagnostic in diagnostics:
        call_index = int(diagnostic["call_index"])
        frame_index = int(diagnostic["frame_index"])
        teacher_activation = teacher_by_call.get(call_index)
        if teacher_activation is None:
            continue
        aligned = diagnostic["aligned"]
        if not isinstance(aligned, torch.Tensor):
            continue
        if teacher_activation.aligned_output is not None:
            feature_losses.append(
                roi_feature_energy_loss(
                    aligned,
                    teacher_activation.aligned_output,
                    masks[:, frame_index],
                )
            )

        first_weights = diagnostic.get("first_weights")
        if isinstance(first_weights, (torch.Tensor, tuple)):
            append_shift_losses(
                first_weights,
                propagation.first_order_alignment,
                teacher_activation,
                masks[:, frame_index],
                source=0,
            )
            first_confidence = diagnostic.get("first_confidence")
            if isinstance(first_confidence, torch.Tensor):
                alignment_losses.append(
                    roi_confidence_loss(
                        first_confidence,
                        teacher_source_confidence(
                            teacher_activation.mask, source=0
                        ),
                        masks[:, frame_index],
                    )
                )
        second_weights = diagnostic.get("second_weights")
        if isinstance(second_weights, (torch.Tensor, tuple)):
            append_shift_losses(
                second_weights,
                propagation.second_order_alignment,
                teacher_activation,
                masks[:, frame_index],
                source=1,
            )
            second_confidence = diagnostic.get("second_confidence")
            if isinstance(second_confidence, torch.Tensor):
                alignment_losses.append(
                    roi_confidence_loss(
                        second_confidence,
                        teacher_source_confidence(
                            teacher_activation.mask, source=1
                        ),
                        masks[:, frame_index],
                    )
                )
    feature = (
        torch.stack(feature_losses).mean() if feature_losses else zero
    )
    alignment = (
        torch.stack(alignment_losses).mean() if alignment_losses else zero
    )
    return feature, alignment


def calculate_reconstruction_losses(
    restored: torch.Tensor,
    forward: torch.Tensor,
    backward: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    perceptual_criterion: MaskedVGG16PerceptualLoss,
    args: argparse.Namespace,
    teacher_targets: torch.Tensor | None = None,
    teacher_feature_loss: torch.Tensor | None = None,
    teacher_alignment_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    perceptual = perceptual_criterion(restored, targets, masks)
    structural = masked_multiscale_structural_loss(
        restored,
        targets,
        masks,
        frame_stride=args.structural_frame_stride,
    )
    base = restoration_loss(
        restored,
        targets,
        masks,
        gradient_weight=args.gradient_weight,
        temporal_weight=args.temporal_weight,
        high_frequency_weight=args.high_frequency_weight,
        perceptual_weight=args.perceptual_weight,
        perceptual=perceptual,
        structural_weight=args.structural_weight,
        structural=structural,
    )
    forward_aux = masked_charbonnier_loss(forward, targets, masks)
    backward_aux = masked_charbonnier_loss(backward, targets, masks)
    directional_aux = 0.5 * (forward_aux + backward_aux)
    direction_consistency = masked_charbonnier_loss(forward, backward, masks)
    teacher = (
        masked_charbonnier_loss(restored, teacher_targets, masks)
        if teacher_targets is not None
        else restored.new_zeros(())
    )
    teacher_feature = (
        teacher_feature_loss
        if teacher_feature_loss is not None
        else restored.new_zeros(())
    )
    teacher_alignment = (
        teacher_alignment_loss
        if teacher_alignment_loss is not None
        else restored.new_zeros(())
    )
    total = (
        base.total
        + args.directional_aux_weight * directional_aux
        + args.direction_consistency_weight * direction_consistency
        + args.teacher_weight * teacher
        + args.teacher_feature_weight * teacher_feature
        + args.teacher_alignment_weight * teacher_alignment
    )
    return total, {
        "pixel": base.pixel,
        "gradient": base.gradient,
        "temporal": base.temporal,
        "high_frequency": base.high_frequency,
        "perceptual": base.perceptual,
        "structural": base.structural,
        "directional_aux": directional_aux,
        "direction_consistency": direction_consistency,
        "teacher": teacher,
        "teacher_feature": teacher_feature,
        "teacher_alignment": teacher_alignment,
    }


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    perceptual_criterion: MaskedVGG16PerceptualLoss,
    args: argparse.Namespace,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "pixel": 0.0,
        "gradient": 0.0,
        "temporal": 0.0,
        "high_frequency": 0.0,
        "perceptual": 0.0,
        "structural": 0.0,
        "directional_aux": 0.0,
        "direction_consistency": 0.0,
        "teacher": 0.0,
        "teacher_feature": 0.0,
        "teacher_alignment": 0.0,
        "input_psnr": 0.0,
        "restored_psnr": 0.0,
        "forward_psnr": 0.0,
        "backward_psnr": 0.0,
    }
    count = 0
    model.eval()
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)
        restored, forward, backward = model.forward_with_directions(inputs, masks)
        total, components = calculate_reconstruction_losses(
            restored,
            forward,
            backward,
            targets,
            masks,
            perceptual_criterion,
            args,
        )
        totals["loss"] += float(total)
        for name, value in components.items():
            totals[name] += float(value)
        totals["input_psnr"] += float(masked_psnr(inputs, targets, masks))
        totals["restored_psnr"] += float(masked_psnr(restored, targets, masks))
        totals["forward_psnr"] += float(masked_psnr(forward, targets, masks))
        totals["backward_psnr"] += float(masked_psnr(backward, targets, masks))
        count += 1
    if not count:
        raise RuntimeError("validation loader yielded no batches")
    averages = {name: value / count for name, value in totals.items()}
    averages["psnr_gain"] = averages["restored_psnr"] - averages["input_psnr"]
    return averages


def append_metric(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(record) + "\n")


def memory_metric(
    step: int,
    snapshots: list[MemorySnapshot],
    guard: TrainingMemoryGuard,
) -> dict:
    """Build a compact JSON record while preserving stage-level diagnostics."""

    current_values = [
        snapshot.mps_current_bytes
        for snapshot in snapshots
        if snapshot.mps_current_bytes is not None
    ]
    driver_values = [
        snapshot.mps_driver_bytes
        for snapshot in snapshots
        if snapshot.mps_driver_bytes is not None
    ]
    final = snapshots[-1]
    return {
        "step": step,
        "memory": {
            "status": guard.status(final),
            "peak_mps_current_gib": (
                None
                if not current_values
                else round(max(current_values) / (1024**3), 3)
            ),
            "peak_mps_driver_gib": (
                None
                if not driver_values
                else round(max(driver_values) / (1024**3), 3)
            ),
            "stages": {
                snapshot.stage: snapshot.as_record() for snapshot in snapshots
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.teacher_checkpoint_sha256 = (
        sha256_file(args.teacher_checkpoint)
        if args.teacher_checkpoint is not None
        else None
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    memory_guard = TrainingMemoryGuard(
        device,
        MemoryThresholds(
            warning_mps_ratio=args.memory_warning_ratio,
            critical_mps_ratio=args.memory_critical_ratio,
            warning_system_available_gib=args.memory_warning_available_gib,
            critical_system_available_gib=args.memory_critical_available_gib,
        ),
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    if args.initialize_from_v2 is not None:
        if not isinstance(model, MiohRestorerV3):
            raise AssertionError("V2 initialization requires a V3 model")
        source_v2 = load_v2_for_initialization(args.initialize_from_v2)
        model.initialize_from_v2(source_v2)
        del source_v2
    if args.initialize_from_v3 is not None:
        if not isinstance(model, MiohRestorerV3):
            raise AssertionError("V3 initialization requires a V3 model")
        source_config, source_state = load_v3_state_for_initialization(
            args.initialize_from_v3
        )
        for key, expected in (
            ("window_frames", model.window_frames),
            ("channels", model.channels),
            ("num_blocks", model.num_blocks),
            ("encoder_blocks", model.encoder_blocks),
            ("reconstruction_blocks", model.reconstruction_blocks),
        ):
            if int(source_config.get(key, -1)) != expected:
                raise ValueError(
                    f"V3 initialization {key}={source_config.get(key)} "
                    f"does not match target {expected}"
                )
        copied, fresh = model.initialize_from_v3_state_dict(source_state)
        print(
            "V3 initialization: "
            f"copied={copied} tensors, fresh={fresh} tensors "
            "(alignment stages relearned)"
        )
        del source_config, source_state
    model = model.to(device)
    if isinstance(model, MiohRestorerV3):
        model.set_gradient_checkpointing(args.gradient_checkpointing)
    ema = copy.deepcopy(model).eval()
    set_requires_grad(ema, False)
    discriminator = TemporalPatchDiscriminator(args.discriminator_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99)
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=args.gan_learning_rate, betas=(0.0, 0.99)
    )
    perceptual_criterion = MaskedVGG16PerceptualLoss(
        frame_stride=args.perceptual_frame_stride,
        image_size=args.perceptual_image_size,
    ).to(device).eval()
    teacher = (
        load_basicvsrpp_teacher(
            args.teacher_checkpoint,
            device,
            fp16=args.teacher_fp16,
        )
        if args.teacher_checkpoint is not None
        else None
    )
    teacher_activations: list[AlignmentActivation] = []
    intermediate_teacher_enabled = (
        teacher is not None
        and (
            args.teacher_feature_weight > 0
            or args.teacher_alignment_weight > 0
        )
    )
    teacher_analyzer = (
        BasicVSRPPActivationAnalyzer(
            teacher,
            capture_policy=AlignmentCapturePolicy(
                branches=frozenset(), max_calls_per_branch=0
            ),
            activation_callback=teacher_activations.append,
            collect_statistics=False,
        )
        if intermediate_teacher_enabled
        else None
    )

    start_step = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        expected = model_config(model, args)
        checkpoint_config = payload["config"]
        architecture_keys = (
            (
                "version",
                "window_frames",
                "chunk_frames",
                "channels",
                "num_blocks",
                "image_size",
                "fusion_full_channels",
                "fusion_half_channels",
                "fusion_quarter_channels",
            )
            if args.model_version == 2
            else (
                "version",
                "architecture_revision",
                "window_frames",
                "channels",
                "num_blocks",
                "image_size",
                "encoder_blocks",
                "reconstruction_blocks",
                "alignment_radius",
                "first_order_dilation",
                "second_order_dilation",
                "alignment_key_channels",
                "alignment_groups",
                "hierarchical_alignment_dilations",
                "alignment_temperature",
            )
        )
        for key in architecture_keys:
            checkpoint_value = checkpoint_config.get(
                key,
                [] if key == "hierarchical_alignment_dilations" else 1.0,
            )
            if checkpoint_value != expected[key]:
                raise ValueError(
                    f"resume checkpoint {key}={checkpoint_value} "
                    f"does not match {expected[key]}"
                )
        for key in (
            "teacher_checkpoint_sha256",
            "teacher_feature_weight",
            "teacher_alignment_weight",
            "teacher_distill_calls",
            "teacher_shift_temperature",
        ):
            if checkpoint_config.get(key) != expected.get(key):
                raise ValueError(
                    f"resume checkpoint {key}={checkpoint_config.get(key)} "
                    f"does not match {expected.get(key)}"
                )
        checkpoint_gradient_checkpointing = checkpoint_config.get(
            "training_memory", {}
        ).get("gradient_checkpointing", False)
        if checkpoint_gradient_checkpointing != args.gradient_checkpointing:
            raise ValueError(
                "resume checkpoint gradient_checkpointing="
                f"{checkpoint_gradient_checkpointing} does not match "
                f"{args.gradient_checkpointing}"
            )
        model.load_state_dict(payload["state_dict"], strict=True)
        ema.load_state_dict(payload["ema_state_dict"], strict=True)
        discriminator.load_state_dict(payload["discriminator_state_dict"], strict=True)
        if not args.reset_optimizers:
            optimizer.load_state_dict(payload["optimizer"])
            discriminator_optimizer.load_state_dict(
                payload["discriminator_optimizer"]
            )
            move_optimizer_state(optimizer, device)
            move_optimizer_state(discriminator_optimizer, device)
        start_step = int(payload["step"])
        if start_step >= args.steps:
            raise ValueError("steps must be greater than the resumed step")
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        del payload
        release_device_memory(device)

    train_loader = make_loader(args.train_metadata_root, args, training=True)
    val_loader = (
        make_loader(args.val_metadata_root, args, training=False)
        if args.val_metadata_root
        else None
    )
    train_iterator = iter(train_loader)
    metrics_path = args.work_dir / "metrics.jsonl"
    print(
        f"MiohRestorerV{args.model_version} training: "
        f"device={device}, samples={len(train_loader.dataset)}, "
        f"generator_parameters={sum(p.numel() for p in model.parameters()):,}, "
        "discriminator_parameters="
        f"{sum(p.numel() for p in discriminator.parameters()):,}, "
        f"steps={start_step + 1}-{args.steps}, gan_start={args.gan_start_step}, "
        f"teacher_weight={args.teacher_weight:.3f}, "
        f"teacher_feature_weight={args.teacher_feature_weight:.3f}, "
        f"teacher_alignment_weight={args.teacher_alignment_weight:.3f}, "
        f"workers={args.workers}, prefetch={args.prefetch_factor}, "
        f"memory_warning={args.memory_warning_ratio:.2f}, "
        f"memory_critical={args.memory_critical_ratio:.2f}"
    )
    startup_memory = memory_metric(
        start_step,
        [memory_guard.capture("startup", synchronize=True)],
        memory_guard,
    )
    append_metric(metrics_path, startup_memory)
    print(json.dumps(startup_memory))
    if val_loader is not None and args.validate_at_start:
        release_device_memory(device)
        validation = validate(ema, val_loader, device, perceptual_criterion, args)
        record = {"step": start_step, "validation": validation, "model": "ema"}
        append_metric(metrics_path, record)
        print(json.dumps(record))
        release_device_memory(device)

    model.train()
    emergency_path: Path | None = None
    for step in range(start_step + 1, args.steps + 1):
        started = time.perf_counter()
        detailed_memory = step == 1 or step % args.memory_log_every == 0
        memory_snapshots: list[MemorySnapshot] = []
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)
        if detailed_memory:
            memory_snapshots.append(memory_guard.capture("batch_ready"))

        generator_lr = cosine_learning_rate(
            step=step,
            total_steps=args.steps,
            base=args.learning_rate,
            minimum=args.minimum_learning_rate,
            warmup_steps=args.warmup_steps,
        )
        set_learning_rate(optimizer, generator_lr)
        optimizer.zero_grad(set_to_none=True)
        teacher_targets = None
        capture_branch: str | None = None
        capture_calls: frozenset[int] = frozenset()
        teacher_activations.clear()
        if teacher is not None:
            teacher_dtype = next(teacher.parameters()).dtype
            if teacher_analyzer is not None:
                if not isinstance(model, MiohRestorerV3):
                    raise AssertionError(
                        "intermediate distillation requires MiohRestorerV3"
                    )
                capture_branch, capture_calls, capture_stride = (
                    distillation_capture_plan(
                        model, step, args.teacher_distill_calls
                    )
                )
                teacher_analyzer.capture_policy = AlignmentCapturePolicy(
                    branches=frozenset({capture_branch}),
                    call_stride=capture_stride,
                    max_calls_per_branch=len(capture_calls),
                )
                teacher_analyzer.begin_clip(inputs.shape[1])
            with torch.inference_mode():
                teacher_targets = teacher(inputs.to(dtype=teacher_dtype)).to(
                    dtype=inputs.dtype
                )
            if args.teacher_weight <= 0:
                teacher_targets = None
        diagnostics: list[dict[str, object]] = []
        if capture_branch is not None:
            if not isinstance(model, MiohRestorerV3):
                raise AssertionError(
                    "intermediate distillation requires MiohRestorerV3"
                )
            restored, forward, backward, diagnostics = (
                model.forward_with_distillation(
                    inputs,
                    masks,
                    capture_branch=capture_branch,
                    capture_calls=capture_calls,
                )
            )
            teacher_feature_loss, teacher_alignment_loss = (
                calculate_intermediate_teacher_losses(
                    model,
                    diagnostics,
                    teacher_activations,
                    masks,
                    branch=capture_branch,
                    temperature=args.teacher_shift_temperature,
                )
            )
        else:
            restored, forward, backward = model.forward_with_directions(
                inputs, masks
            )
            teacher_feature_loss = restored.new_zeros(())
            teacher_alignment_loss = restored.new_zeros(())
        reconstruction_total, components = calculate_reconstruction_losses(
            restored,
            forward,
            backward,
            targets,
            masks,
            perceptual_criterion,
            args,
            teacher_targets,
            teacher_feature_loss,
            teacher_alignment_loss,
        )

        discriminator_loss = restored.new_zeros(())
        generator_adversarial = restored.new_zeros(())
        discriminator_grad_norm = restored.new_zeros(())
        active_gan = args.gan_weight > 0 and step >= args.gan_start_step
        if active_gan:
            discriminator_lr = gan_learning_rate(args, step)
            set_learning_rate(discriminator_optimizer, discriminator_lr)
            set_requires_grad(discriminator, True)
            discriminator.train()
            discriminator_optimizer.zero_grad(set_to_none=True)
            real_input = temporal_discriminator_input(
                targets,
                targets,
                masks,
                frame_stride=args.gan_frame_stride,
                image_size=args.gan_image_size,
            )
            fake_input = temporal_discriminator_input(
                restored.detach(),
                targets,
                masks,
                frame_stride=args.gan_frame_stride,
                image_size=args.gan_image_size,
            )
            discriminator_loss = discriminator_hinge_loss(
                discriminator(real_input),
                discriminator(fake_input),
            )
            discriminator_loss.backward()
            discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), args.max_grad_norm
            )
            discriminator_optimizer.step()

            set_requires_grad(discriminator, False)
            generator_fake_input = temporal_discriminator_input(
                restored,
                targets,
                masks,
                frame_stride=args.gan_frame_stride,
                image_size=args.gan_image_size,
            )
            generator_adversarial = generator_hinge_loss(
                discriminator(generator_fake_input)
            )
        else:
            discriminator_lr = 0.0

        total = reconstruction_total + args.gan_weight * generator_adversarial
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"non-finite generator loss at step {step}: {total}"
            )
        if detailed_memory:
            memory_snapshots.append(memory_guard.capture("loss_graph"))
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optimizer.step()
        update_ema(ema, model, args.ema_decay)
        if active_gan:
            set_requires_grad(discriminator, True)
        if detailed_memory:
            memory_snapshots.append(memory_guard.capture("after_optimizer"))

        record = {
            "step": step,
            "loss": float(total.detach()),
            **{name: float(value.detach()) for name, value in components.items()},
            "generator_adversarial": float(generator_adversarial.detach()),
            "discriminator": float(discriminator_loss.detach()),
            "grad_norm": float(grad_norm),
            "discriminator_grad_norm": float(discriminator_grad_norm),
            "learning_rate": generator_lr,
            "discriminator_learning_rate": discriminator_lr,
            "gan_active": active_gan,
            "seconds": time.perf_counter() - started,
        }
        append_metric(metrics_path, record)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(record))

        if active_gan:
            del real_input, fake_input, generator_fake_input
        if teacher_targets is not None:
            del teacher_targets
        teacher_activations.clear()
        del (
            batch,
            inputs,
            targets,
            masks,
            restored,
            forward,
            backward,
            reconstruction_total,
            components,
            discriminator_loss,
            generator_adversarial,
            discriminator_grad_norm,
            total,
            grad_norm,
            diagnostics,
            teacher_feature_loss,
            teacher_alignment_loss,
        )
        optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)

        safe_snapshot = memory_guard.capture("after_step_cleanup")
        memory_snapshots.append(safe_snapshot)
        detected_pressure = memory_guard.status(safe_snapshot)
        if detected_pressure != "normal":
            print(
                "memory pressure detected: "
                f"step={step}, status={detected_pressure}, "
                f"available={safe_snapshot.system_available_gib:.2f}GiB"
            )
            release_device_memory(device)
            safe_snapshot = memory_guard.capture(
                "after_pressure_cleanup", synchronize=True
            )
            memory_snapshots.append(safe_snapshot)

        safe_status = memory_guard.status(safe_snapshot)
        if safe_status == "critical" and args.memory_emergency_stop:
            emergency_path = save_checkpoint(
                model,
                ema,
                discriminator,
                optimizer,
                discriminator_optimizer,
                args,
                step,
                archive=True,
                emergency=True,
            )
            release_device_memory(device)
            memory_snapshots.append(
                memory_guard.capture("after_emergency_checkpoint", synchronize=True)
            )
            emergency_record = memory_metric(
                step, memory_snapshots, memory_guard
            )
            emergency_record["memory"]["pressure_detected"] = detected_pressure
            emergency_record["memory"]["action"] = "emergency_checkpoint_and_stop"
            emergency_record["memory"]["checkpoint"] = str(emergency_path)
            append_metric(metrics_path, emergency_record)
            print(json.dumps(emergency_record))
            break

        if val_loader is not None and step % args.validate_every == 0:
            release_device_memory(device)
            if detailed_memory:
                memory_snapshots.append(
                    memory_guard.capture("before_validation", synchronize=True)
                )
            validation = validate(ema, val_loader, device, perceptual_criterion, args)
            validation_record = {
                "step": step,
                "validation": validation,
                "model": "ema",
            }
            append_metric(metrics_path, validation_record)
            print(json.dumps(validation_record))
            model.train()
            release_device_memory(device)
            if detailed_memory:
                memory_snapshots.append(
                    memory_guard.capture("after_validation", synchronize=True)
                )
        if step % args.save_latest_every == 0:
            release_device_memory(device)
            if detailed_memory:
                memory_snapshots.append(
                    memory_guard.capture("before_checkpoint", synchronize=True)
                )
            output = save_checkpoint(
                model,
                ema,
                discriminator,
                optimizer,
                discriminator_optimizer,
                args,
                step,
                archive=step % args.save_every == 0,
            )
            print(f"checkpoint: {output}")
            release_device_memory(device)
            if detailed_memory:
                memory_snapshots.append(
                    memory_guard.capture("after_checkpoint", synchronize=True)
                )

        if detailed_memory or detected_pressure != "normal":
            memory_record = memory_metric(step, memory_snapshots, memory_guard)
            if detected_pressure != "normal":
                memory_record["memory"]["pressure_detected"] = detected_pressure
            append_metric(metrics_path, memory_record)
            print(json.dumps(memory_record))

    if emergency_path is not None:
        if teacher_analyzer is not None:
            teacher_analyzer.close()
        print(
            "training stopped safely because memory remained critical after cleanup: "
            f"{emergency_path}"
        )
        print(
            "resume: "
            f"RESUME='{emergency_path}' "
            f"scripts/training/run-mioh-restorer-v{args.model_version}-allin.sh"
        )
        return MEMORY_EMERGENCY_EXIT_CODE

    release_device_memory(device)
    if args.steps % args.save_latest_every:
        final_path = save_checkpoint(
            model,
            ema,
            discriminator,
            optimizer,
            discriminator_optimizer,
            args,
            args.steps,
            archive=True,
        )
    elif args.steps % args.save_every:
        final_path = save_checkpoint(
            model,
            ema,
            discriminator,
            optimizer,
            discriminator_optimizer,
            args,
            args.steps,
            archive=True,
        )
    else:
        final_path = (
            args.work_dir
            / f"{checkpoint_prefix(args)}-step-{args.steps:07d}.pth"
        )
    release_device_memory(device)
    if teacher_analyzer is not None:
        teacher_analyzer.close()
    print(f"training complete: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
