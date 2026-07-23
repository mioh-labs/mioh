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
from lada.models.mioh_restorer.losses_v5 import MiohRestorerV5Loss, masked_mean
from lada.models.mioh_restorer.model_v5 import (
    MiohRestorerV5,
    MiohRestorerV5Config,
    parameter_count,
)
from lada.models.mioh_restorer.model_v5_hq import MiohRestorerV5HQ
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
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--restart-stage", action="store_true")
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
    if not 0 < args.ema_decay < 1:
        raise ValueError("ema-decay must be between zero and one")
    if args.resume and args.initialize_from:
        raise ValueError("resume and initialize-from are mutually exclusive")
    if args.restart_stage and args.resume:
        raise ValueError("restart-stage and resume are mutually exclusive")
    if args.stage >= 5 and args.variant not in {"q", "hq"}:
        raise ValueError("temporal stages 5/6 require a five-output V5-Q/HQ model")
    if args.variant == "hq" and not args.basicvsrpp_checkpoint.is_file():
        raise FileNotFoundError(args.basicvsrpp_checkpoint)
    if args.device != "cuda" and args.amp != "off":
        raise ValueError("mixed precision is enabled only for CUDA; use --amp off on MPS")


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
) -> MiohRestorerV5 | MiohRestorerV5HQ:
    if variant == "s":
        return MiohRestorerV5.shipping()
    if variant == "hq":
        model = MiohRestorerV5HQ()
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = []
    for name in ("inputs", "targets", "masks"):
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch field {name} is not a tensor")
        tensors.append(value.to(device, non_blocking=True))
    return tuple(tensors)  # type: ignore[return-value]


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for destination, source in zip(ema.parameters(), model.parameters(), strict=True):
        destination.lerp_(source.detach(), 1.0 - decay)
    for destination, source in zip(ema.buffers(), model.buffers(), strict=True):
        destination.copy_(source)


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
) -> dict[str, object]:
    return {
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
) -> Path:
    payload = checkpoint_payload(
        model, ema, optimizer, scaler, args, step=step, epoch=epoch
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
        if optimizer is None or scaler is None:
            raise ValueError("resume requires optimizer and scaler")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scaler_state_dict"):
            scaler.load_state_dict(payload["scaler_state_dict"])
        return int(payload.get("local_step", 0)), int(payload.get("epoch", 0))
    return 0, 0


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


@torch.no_grad()
def validate(
    model: MiohRestorerV5,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    limit: int,
) -> dict[str, object]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    bucket_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    count = 0
    empty_roi_batches = 0
    for batch in loader:
        inputs, targets, masks = move_batch(batch, device)
        if not bool(torch.any(masks > 0)):
            empty_roi_batches += 1
            continue
        bucket_value = batch["bucket"]
        bucket = int(bucket_value[0]) if isinstance(bucket_value, torch.Tensor) else int(bucket_value)  # type: ignore[arg-type]
        sources = inputs[:, list(model.config.output_indices), :3]
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            restored, confidence = model(inputs)
        mse = masked_mean((restored.float() - targets).square(), masks)
        mae = masked_mean((restored.float() - targets).abs(), masks)
        baseline = masked_mean((sources.float() - targets).abs(), masks)
        metrics = {
            "roi_psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
            "roi_mae": float(mae),
            "identity_source_mae": float(baseline),
            "confidence_mean": float(masked_mean(confidence.float(), masks)),
        }
        for name, value in metrics.items():
            totals[name] += value
            bucket_totals[bucket][name] += value
        bucket_totals[bucket]["count"] += 1
        count += 1
        if count >= limit:
            break
    if not count:
        raise RuntimeError("validation produced no batches")
    result: dict[str, object] = {
        name: value / count for name, value in totals.items()
    }
    result["valid_roi_batches"] = count
    result["empty_roi_batches_skipped"] = empty_roi_batches
    result["buckets"] = {
        str(bucket): {
            name: value / values["count"]
            for name, value in values.items()
            if name != "count"
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
    ).to(device)
    configure_trainable_parameters(
        model, variant=args.variant, stage=stage
    )
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=stage.learning_rate,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=scale_gradients)
    boundary_temporal_weight = float(
        getattr(stage, "boundary_temporal_weight", 0.0)
    )
    loss_function = MiohRestorerV5Loss(
        weights=stage.loss,
        boundary_temporal_weight=boundary_temporal_weight,
    ).to(device)
    perceptual = (
        V5PerceptualLoss(image_size=args.perceptual_image_size).to(device).eval()
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
        )
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

    train_loader, train_sampler = make_loader(
        args.train_manifest,
        output_indices=model.config.output_indices,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch=args.prefetch,
        training=True,
        seed=args.seed,
    )
    validation_loader, _ = make_loader(
        args.validation_manifest,
        output_indices=model.config.output_indices,
        batch_size=1,
        workers=min(args.workers, 1),
        prefetch=args.prefetch,
        training=False,
        seed=args.seed,
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
    }
    print(json.dumps(startup, ensure_ascii=False, indent=2), flush=True)
    append_json(metrics_path, startup)
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
                    inputs, targets, masks = move_batch(batch, device)
                    if bool(torch.any(masks > 0)):
                        prepared_batches.append((inputs, targets, masks))
                if not prepared_batches:
                    continue
                accumulated: dict[str, float] = defaultdict(float)
                for inputs, targets, masks in prepared_batches:
                    sources = inputs[:, list(model.config.output_indices), :3]
                    with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                        restored, confidence, base, texture = model.forward_components(inputs)
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
                            masks,
                            aligned_previous_restored=temporal_values[0],
                            aligned_previous_target=temporal_values[1],
                            temporal_valid=temporal_values[2],
                            perceptual=perceptual_value,
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
                        stats["total_with_stage_supervision"] = float(loss.detach())
                        scaled = loss / len(prepared_batches)
                    scaler.scale(scaled).backward()
                    for name, value in stats.items():
                        accumulated[name] += float(value) / len(prepared_batches)

                next_step = step + 1
                warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
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
                    group["lr"] = learning_rate
                if args.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
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
                    result = validate(
                        ema,
                        validation_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        limit=args.validation_batches,
                    )
                    record = {"event": "validation", "step": step, **result}
                    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
                    append_json(metrics_path, record)
                    model.train()
                    clear_accelerator_cache(device)
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
                    )
                    print(f"saved: {saved}", flush=True)
            epoch += 1
    except KeyboardInterrupt:
        saved = save_checkpoint(
            model, ema, optimizer, scaler, args, work_dir, step=step, epoch=epoch
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
