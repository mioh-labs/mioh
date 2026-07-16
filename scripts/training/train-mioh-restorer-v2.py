# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Train the all-in bidirectional MiohRestorerV2 model."""

from __future__ import annotations

import argparse
import copy
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
    TemporalPatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    masked_charbonnier_loss,
    masked_multiscale_structural_loss,
    masked_psnr,
    restoration_loss,
    temporal_discriminator_input,
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
    parser.add_argument("--reset-optimizers", action="store_true")
    parser.add_argument("--device", default="auto")
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
    }
    for name, value in loss_weights.items():
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if args.gan_start_step < 0:
        raise ValueError("gan_start_step must not be negative")
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


def build_model(args: argparse.Namespace) -> MiohRestorerV2:
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


def model_config(model: MiohRestorerV2, args: argparse.Namespace) -> dict:
    return {
        "version": 2,
        "window_frames": model.window_frames,
        "chunk_frames": model.chunk_frames,
        "channels": model.channels,
        "num_blocks": model.num_blocks,
        "image_size": args.image_size,
        "fusion_full_channels": model.fusion_full_channels,
        "fusion_half_channels": model.fusion_half_channels,
        "fusion_quarter_channels": model.fusion_quarter_channels,
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
        },
    }


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
def update_ema(ema: MiohRestorerV2, model: MiohRestorerV2, decay: float) -> None:
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
    model: MiohRestorerV2,
    ema: MiohRestorerV2,
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
    }
    if stop_reason is not None:
        payload["stop_reason"] = stop_reason
    return payload


def save_checkpoint(
    model: MiohRestorerV2,
    ema: MiohRestorerV2,
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
    latest = args.work_dir / "mioh-restorer-v2-latest.pth"
    latest_temporary = latest.with_suffix(".tmp")
    torch.save(payload, latest_temporary)
    latest_temporary.replace(latest)
    if not archive:
        return latest
    suffix = "emergency-step" if emergency else "step"
    output = args.work_dir / f"mioh-restorer-v2-{suffix}-{step:07d}.pth"
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def calculate_reconstruction_losses(
    restored: torch.Tensor,
    forward: torch.Tensor,
    backward: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    perceptual_criterion: MaskedVGG16PerceptualLoss,
    args: argparse.Namespace,
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
    total = (
        base.total
        + args.directional_aux_weight * directional_aux
        + args.direction_consistency_weight * direction_consistency
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
    }


@torch.inference_mode()
def validate(
    model: MiohRestorerV2,
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

    model = build_model(args).to(device)
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

    start_step = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        expected = model_config(model, args)
        checkpoint_config = payload["config"]
        architecture_keys = (
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
        for key in architecture_keys:
            if checkpoint_config[key] != expected[key]:
                raise ValueError(
                    f"resume checkpoint {key}={checkpoint_config[key]} "
                    f"does not match {expected[key]}"
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
        "MiohRestorerV2 all-in training: "
        f"device={device}, samples={len(train_loader.dataset)}, "
        f"generator_parameters={sum(p.numel() for p in model.parameters()):,}, "
        "discriminator_parameters="
        f"{sum(p.numel() for p in discriminator.parameters()):,}, "
        f"steps={start_step + 1}-{args.steps}, gan_start={args.gan_start_step}, "
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
        restored, forward, backward = model.forward_with_directions(inputs, masks)
        reconstruction_total, components = calculate_reconstruction_losses(
            restored,
            forward,
            backward,
            targets,
            masks,
            perceptual_criterion,
            args,
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
        print(
            "training stopped safely because memory remained critical after cleanup: "
            f"{emergency_path}"
        )
        print(
            "resume: "
            f"RESUME='{emergency_path}' "
            "scripts/training/run-mioh-restorer-v2-allin.sh"
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
        final_path = args.work_dir / f"mioh-restorer-v2-step-{args.steps:07d}.pth"
    release_device_memory(device)
    print(f"training complete: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
