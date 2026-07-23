#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Train the paired-GT V5 quality stages on CUDA/Colab.

This runner intentionally supports stages 3 and 4 only.  V5 stages 1, 2, 5
and 6 require exact-motion, natural-flow, feature-distillation or flow-aligned
temporal teachers that are not yet part of the V5 training implementation.
Silently dropping those terms would produce a checkpoint with the wrong
quality contract, so unsupported stages fail closed.
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

from lada.models.mioh_restorer.curriculum_v5 import (
    stage_definition,
    stage_learning_rate,
)
from lada.models.mioh_restorer.losses_v5 import MiohRestorerV5Loss, masked_mean
from lada.models.mioh_restorer.model_v5 import MiohRestorerV5, parameter_count
from lada.models.mioh_restorer.native_dataset_v5 import (
    MiohRestorerV5NativeDataset,
    V5BucketBatchSampler,
)
from lada.models.mioh_restorer.training import MaskedVGG16PerceptualLoss


SUPPORTED_STAGES = (3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("q", "s"), default="q")
    parser.add_argument("--stage", type=int, choices=SUPPORTED_STAGES, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp",
        choices=("auto", "bf16", "fp16", "off"),
        default="auto",
    )
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=24)
    parser.add_argument("--perceptual-frame-stride", type=int, default=1)
    parser.add_argument("--perceptual-image-size", type=int, default=224)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps is not None and args.steps <= 0:
        raise ValueError("steps must be positive")
    for name in (
        "batch_size",
        "accumulate",
        "save_every",
        "validate_every",
        "validation_batches",
        "log_every",
        "perceptual_frame_stride",
        "perceptual_image_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.prefetch <= 0:
        raise ValueError("workers and prefetch are invalid")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema-decay must be between zero and one")
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("resume and initialize-from are mutually exclusive")
    for manifest in (args.train_manifest, args.validation_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(manifest)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(variant: str) -> MiohRestorerV5:
    return MiohRestorerV5.quality() if variant == "q" else MiohRestorerV5.shipping()


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
) -> tuple[DataLoader, V5BucketBatchSampler]:
    dataset = MiohRestorerV5NativeDataset(
        manifest,
        output_indices=output_indices,
        degrade=training,
        horizontal_flip=training,
        time_reverse=training,
        deterministic=not training,
    )
    sampler = V5BucketBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        seed=seed,
    )
    kwargs: dict[str, object] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
    return DataLoader(dataset, **kwargs), sampler


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_value, value in zip(ema.parameters(), model.parameters(), strict=True):
        ema_value.lerp_(value.detach(), 1.0 - decay)
    for ema_value, value in zip(ema.buffers(), model.buffers(), strict=True):
        ema_value.copy_(value)


def checkpoint_payload(
    *,
    model: MiohRestorerV5,
    ema: MiohRestorerV5,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    local_step: int,
    epoch: int,
) -> dict[str, object]:
    return {
        "format": "mioh-restorer-v5-colab-v1",
        "variant": args.variant,
        "config": asdict(model.config),
        "stage": args.stage,
        "local_step": local_step,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "arguments": vars(args),
    }


def atomic_torch_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def save_checkpoint(
    *,
    model: MiohRestorerV5,
    ema: MiohRestorerV5,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    local_step: int,
    epoch: int,
) -> Path:
    payload = checkpoint_payload(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        local_step=local_step,
        epoch=epoch,
    )
    numbered = args.work_dir / f"mioh-v5-{args.variant}-stage{args.stage}-step-{local_step:06d}.pth"
    latest = args.work_dir / f"mioh-v5-{args.variant}-stage{args.stage}-latest.pth"
    atomic_torch_save(payload, numbered)
    temporary_latest = latest.with_suffix(latest.suffix + ".tmp")
    shutil.copyfile(numbered, temporary_latest)
    temporary_latest.replace(latest)
    return numbered


def load_weights(
    path: Path,
    *,
    model: MiohRestorerV5,
    ema: MiohRestorerV5,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    expected_stage: int | None = None,
    initialize_from_ema: bool = False,
) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = payload["state_dict"]
    ema_state = payload.get("ema_state_dict", raw_state)
    model.load_state_dict(ema_state if initialize_from_ema else raw_state, strict=True)
    ema.load_state_dict(ema_state, strict=True)
    if expected_stage is not None and int(payload.get("stage", -1)) != expected_stage:
        raise ValueError("resume checkpoint belongs to a different V5 stage")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict"):
        scaler.load_state_dict(payload["scaler_state_dict"])
    return int(payload.get("local_step", 0)), int(payload.get("epoch", 0))


def batch_tensors(
    batch: dict[str, torch.Tensor | list[str]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["inputs"].to(device, non_blocking=True),  # type: ignore[union-attr]
        batch["targets"].to(device, non_blocking=True),  # type: ignore[union-attr]
        batch["masks"].to(device, non_blocking=True),  # type: ignore[union-attr]
    )


@torch.no_grad()
def validate(
    model: MiohRestorerV5,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    limit: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        inputs, targets, masks = batch_tensors(batch, device)
        output_indices = model.config.output_indices
        sources = inputs[:, list(output_indices), :3]
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            restored, confidence, _base, _texture = model.forward_components(inputs)
        error = masked_mean((restored.float() - targets).abs(), masks)
        mse = masked_mean((restored.float() - targets).square(), masks)
        totals["roi_mae"] += float(error)
        totals["roi_psnr"] += float(-10.0 * torch.log10(mse.clamp_min(1e-12)))
        totals["confidence_mean"] += float(masked_mean(confidence.float(), masks))
        totals["identity_source_mae"] += float(
            masked_mean((sources.float() - targets).abs(), masks)
        )
        count += 1
        if count >= limit:
            break
    if not count:
        raise RuntimeError("validation loader produced no batches")
    return {name: value / count for name, value in totals.items()}


def log_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    stage = stage_definition(args.stage)
    steps = args.steps or stage.default_steps
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU Colab runtime")
    amp_enabled, amp_dtype, scale_gradients = amp_configuration(args.amp, device)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.variant).to(device)
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=scale_gradients)
    loss_function = MiohRestorerV5Loss(stage=args.stage).to(device)
    perceptual_criterion = None
    if stage.loss.perceptual > 0:
        perceptual_criterion = MaskedVGG16PerceptualLoss(
            frame_stride=args.perceptual_frame_stride,
            image_size=args.perceptual_image_size,
        ).to(device)
        perceptual_criterion.eval()
    train_loader, train_sampler = make_loader(
        args.train_manifest,
        output_indices=model.config.output_indices,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch=args.prefetch,
        training=True,
        seed=args.seed,
    )
    validation_loader, _validation_sampler = make_loader(
        args.validation_manifest,
        output_indices=model.config.output_indices,
        batch_size=1,
        workers=max(0, min(args.workers, 2)),
        prefetch=args.prefetch,
        training=False,
        seed=args.seed,
    )

    local_step = 0
    epoch = 0
    if args.resume is not None:
        local_step, epoch = load_weights(
            args.resume,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            expected_stage=args.stage,
        )
    elif args.initialize_from is not None:
        load_weights(
            args.initialize_from,
            model=model,
            ema=ema,
            initialize_from_ema=True,
        )

    startup = {
        "event": "start",
        "variant": args.variant,
        "stage": args.stage,
        "stage_name": stage.name,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": parameter_count(model),
        "amp": str(amp_dtype).replace("torch.", "") if amp_enabled else "off",
        "batch_size": args.batch_size,
        "accumulate": args.accumulate,
        "steps": steps,
        "starting_step": local_step,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
    }
    print(json.dumps(startup, ensure_ascii=False, indent=2), flush=True)
    metrics_path = args.work_dir / "metrics.jsonl"
    log_json(metrics_path, startup)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    try:
        while local_step < steps:
            train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            while local_step < steps:
                accumulated: dict[str, float] = defaultdict(float)
                micro_batches = []
                while len(micro_batches) < args.accumulate:
                    try:
                        micro_batches.append(next(iterator))
                    except StopIteration:
                        break
                if not micro_batches:
                    break
                for micro in micro_batches:
                    inputs, targets, masks = batch_tensors(micro, device)
                    sources = inputs[:, list(model.config.output_indices), :3]
                    with torch.autocast(
                        device_type=device.type,
                        enabled=amp_enabled,
                        dtype=amp_dtype,
                    ):
                        restored, confidence, base, texture = model.forward_components(inputs)
                        perceptual = (
                            perceptual_criterion(restored, targets, masks)
                            if perceptual_criterion is not None
                            else None
                        )
                        loss, stats = loss_function(
                            restored,
                            confidence,
                            base,
                            texture,
                            targets,
                            sources,
                            masks,
                            perceptual=perceptual,
                        )
                        scaled_loss = loss / len(micro_batches)
                    scaler.scale(scaled_loss).backward()
                    for name, value in stats.items():
                        accumulated[name] += value / len(micro_batches)

                next_step = local_step + 1
                learning_rate = stage_learning_rate(
                    stage,
                    next_step,
                    total_steps=steps,
                    warmup_steps=min(args.warmup_steps, max(1, steps // 10)),
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                if args.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model, args.ema_decay)
                local_step = next_step

                if local_step % args.log_every == 0 or local_step == 1:
                    elapsed = time.perf_counter() - started
                    record = {
                        "event": "train",
                        "step": local_step,
                        "epoch": epoch,
                        "learning_rate": learning_rate,
                        "seconds_per_step": elapsed / max(local_step, 1),
                        **accumulated,
                    }
                    if device.type == "cuda":
                        record["cuda_allocated_gib"] = torch.cuda.memory_allocated(device) / 2**30
                        record["cuda_peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    log_json(metrics_path, record)

                if local_step % args.validate_every == 0 or local_step == steps:
                    result = validate(
                        ema,
                        validation_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        limit=args.validation_batches,
                    )
                    record = {"event": "validation", "step": local_step, **result}
                    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
                    log_json(metrics_path, record)
                    model.train()

                if local_step % args.save_every == 0 or local_step == steps:
                    path = save_checkpoint(
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scaler=scaler,
                        args=args,
                        local_step=local_step,
                        epoch=epoch,
                    )
                    print(f"saved: {path}", flush=True)
                if local_step >= steps:
                    break
            epoch += 1
    except KeyboardInterrupt:
        path = save_checkpoint(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            local_step=local_step,
            epoch=epoch,
        )
        print(f"interrupted safely; saved: {path}", flush=True)
        return 130

    completion = {
        "event": "complete",
        "stage": args.stage,
        "step": local_step,
        "elapsed_seconds": time.perf_counter() - started,
    }
    log_json(metrics_path, completion)
    (args.work_dir / "stage-complete.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
