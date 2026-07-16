# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Train MiohRestorerV1 without the BasicVSR++/MMagic model stack."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lada.models.mioh_restorer import (
    MaskedVGG16PerceptualLoss,
    MiohRestorerV1,
    masked_psnr,
    restoration_loss,
    run_training_sequence,
)
from lada.models.mioh_restorer.training_dataset import MiohRestorationDataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mioh restoration model")
    parser.add_argument("--train-metadata-root", type=Path, nargs="+", required=True)
    parser.add_argument("--val-metadata-root", type=Path, nargs="+")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help=(
            "load model weights and step from --resume, but start with a fresh "
            "optimizer using --learning-rate"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sequence-frames", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--chunk-frames", type=int, default=MiohRestorerV1.DEFAULT_CHUNK_FRAMES
    )
    parser.add_argument(
        "--channels", type=int, default=MiohRestorerV1.DEFAULT_CHANNELS
    )
    parser.add_argument("--blocks", type=int, default=MiohRestorerV1.DEFAULT_BLOCKS)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-weight", type=float, default=0.2)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--high-frequency-weight", type=float, default=0.0)
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--perceptual-frame-stride", type=int, default=4)
    parser.add_argument("--perceptual-image-size", type=int, default=224)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--validate-every", type=int, default=2_000)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument(
        "--validate-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degrade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction, default=True)
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


def make_loader(
    roots: list[Path],
    args: argparse.Namespace,
    *,
    training: bool,
) -> DataLoader:
    dataset = MiohRestorationDataset(
        roots,
        sequence_frames=args.sequence_frames,
        image_size=args.image_size,
        degrade=args.degrade,
        horizontal_flip=args.horizontal_flip if training else False,
        deterministic=not training,
        limit=args.limit_train_samples if training else args.limit_val_samples,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if training else 1,
        shuffle=training,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=training and len(dataset) >= args.batch_size,
    )


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def checkpoint_payload(
    model: MiohRestorerV1,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
) -> dict:
    return {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "step": step,
        "trained": step > 0,
        "prototype": False,
        "config": {
            "chunk_frames": model.chunk_frames,
            "channels": model.channels,
            "num_blocks": model.num_blocks,
            "image_size": args.image_size,
            "sequence_frames": args.sequence_frames,
            "gradient_weight": args.gradient_weight,
            "temporal_weight": args.temporal_weight,
            "high_frequency_weight": args.high_frequency_weight,
            "perceptual_weight": args.perceptual_weight,
            "perceptual_frame_stride": args.perceptual_frame_stride,
            "perceptual_image_size": args.perceptual_image_size,
        },
    }


def save_checkpoint(
    model: MiohRestorerV1,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
) -> Path:
    output = args.work_dir / f"mioh-restorer-v1-step-{step:07d}.pth"
    temporary = output.with_suffix(".tmp")
    torch.save(checkpoint_payload(model, optimizer, args, step), temporary)
    temporary.replace(output)
    latest = args.work_dir / "mioh-restorer-v1-latest.pth"
    latest_tmp = latest.with_suffix(".tmp")
    torch.save(checkpoint_payload(model, optimizer, args, step), latest_tmp)
    latest_tmp.replace(latest)
    return output


@torch.inference_mode()
def validate(
    model: MiohRestorerV1,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    perceptual_criterion: MaskedVGG16PerceptualLoss | None,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "pixel": 0.0,
        "gradient": 0.0,
        "temporal": 0.0,
        "high_frequency": 0.0,
        "perceptual": 0.0,
        "input_psnr": 0.0,
        "restored_psnr": 0.0,
    }
    count = 0
    model.eval()
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)
        restored = run_training_sequence(model, inputs, masks)
        perceptual = (
            perceptual_criterion(restored, targets, masks)
            if perceptual_criterion is not None
            else None
        )
        loss = restoration_loss(
            restored,
            targets,
            masks,
            gradient_weight=args.gradient_weight,
            temporal_weight=args.temporal_weight,
            high_frequency_weight=args.high_frequency_weight,
            perceptual_weight=args.perceptual_weight,
            perceptual=perceptual,
        )
        totals["loss"] += float(loss.total)
        totals["pixel"] += float(loss.pixel)
        totals["gradient"] += float(loss.gradient)
        totals["temporal"] += float(loss.temporal)
        totals["high_frequency"] += float(loss.high_frequency)
        totals["perceptual"] += float(loss.perceptual)
        totals["input_psnr"] += float(masked_psnr(inputs, targets, masks))
        totals["restored_psnr"] += float(masked_psnr(restored, targets, masks))
        count += 1
    model.train()
    if not count:
        raise RuntimeError("validation loader yielded no batches")
    averages = {key: value / count for key, value in totals.items()}
    averages["psnr_gain"] = averages["restored_psnr"] - averages["input_psnr"]
    return averages


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "sequence_frames": args.sequence_frames,
        "image_size": args.image_size,
        "chunk_frames": args.chunk_frames,
        "channels": args.channels,
        "learning_rate": args.learning_rate,
        "save_every": args.save_every,
        "validate_every": args.validate_every,
        "validation_batches": args.validation_batches,
        "log_every": args.log_every,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.blocks < 0 or args.workers < 0:
        raise ValueError("blocks and workers must not be negative")
    if (
        args.gradient_weight < 0
        or args.temporal_weight < 0
        or args.high_frequency_weight < 0
        or args.perceptual_weight < 0
    ):
        raise ValueError("loss weights must not be negative")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if args.sequence_frames % args.chunk_frames:
        raise ValueError("sequence_frames must be divisible by chunk_frames")
    if args.image_size % MiohRestorerV1.DOWNSCALE:
        raise ValueError("image_size must be divisible by 4")
    if args.perceptual_frame_stride <= 0:
        raise ValueError("perceptual_frame_stride must be positive")
    if args.perceptual_image_size < 32:
        raise ValueError("perceptual_image_size must be at least 32")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    model = MiohRestorerV1(
        chunk_frames=args.chunk_frames,
        channels=args.channels,
        num_blocks=args.blocks,
    ).to(device)
    perceptual_criterion = None
    if args.perceptual_weight > 0:
        print(
            "Loading pretrained VGG16 features for masked perceptual loss "
            f"(stride={args.perceptual_frame_stride}, "
            f"size={args.perceptual_image_size})"
        )
        perceptual_criterion = MaskedVGG16PerceptualLoss(
            frame_stride=args.perceptual_frame_stride,
            image_size=args.perceptual_image_size,
        ).to(device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99)
    )
    start_step = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        expected = {
            "chunk_frames": args.chunk_frames,
            "channels": args.channels,
            "num_blocks": args.blocks,
            "image_size": args.image_size,
        }
        for key, value in expected.items():
            if int(payload["config"][key]) != value:
                raise ValueError(
                    f"resume checkpoint {key}={payload['config'][key]} does not match {value}"
                )
        model.load_state_dict(payload["state_dict"], strict=True)
        if not args.reset_optimizer:
            optimizer.load_state_dict(payload["optimizer"])
            move_optimizer_state(optimizer, device)
        start_step = int(payload["step"])
        if start_step >= args.steps:
            raise ValueError(
                f"steps ({args.steps}) must be greater than resumed step ({start_step})"
            )

    train_loader = make_loader(args.train_metadata_root, args, training=True)
    val_loader = (
        make_loader(args.val_metadata_root, args, training=False)
        if args.val_metadata_root
        else None
    )
    train_iterator = iter(train_loader)
    metrics_path = args.work_dir / "metrics.jsonl"
    model.train()
    print(
        f"MiohRestorer training: device={device}, samples={len(train_loader.dataset)}, "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}, "
        f"steps={start_step + 1}-{args.steps}"
    )
    if val_loader is not None and args.validate_at_start:
        baseline = validate(model, val_loader, device, args, perceptual_criterion)
        baseline_record = {"step": start_step, "validation": baseline}
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(baseline_record) + "\n")
        print(json.dumps(baseline_record))

    for step in range(start_step + 1, args.steps + 1):
        started = time.perf_counter()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)
        optimizer.zero_grad(set_to_none=True)
        restored = run_training_sequence(model, inputs, masks)
        perceptual = (
            perceptual_criterion(restored, targets, masks)
            if perceptual_criterion is not None
            else None
        )
        loss = restoration_loss(
            restored,
            targets,
            masks,
            gradient_weight=args.gradient_weight,
            temporal_weight=args.temporal_weight,
            high_frequency_weight=args.high_frequency_weight,
            perceptual_weight=args.perceptual_weight,
            perceptual=perceptual,
        )
        if not torch.isfinite(loss.total):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.total}")
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.total.detach()),
            "pixel": float(loss.pixel.detach()),
            "gradient": float(loss.gradient.detach()),
            "temporal": float(loss.temporal.detach()),
            "high_frequency": float(loss.high_frequency.detach()),
            "perceptual": float(loss.perceptual.detach()),
            "grad_norm": float(grad_norm),
            "seconds": time.perf_counter() - started,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record) + "\n")
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(record))

        if val_loader is not None and step % args.validate_every == 0:
            validation = validate(
                model,
                val_loader,
                device,
                args,
                perceptual_criterion,
            )
            validation_record = {"step": step, "validation": validation}
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(validation_record) + "\n")
            print(json.dumps(validation_record))
        if step % args.save_every == 0:
            print(f"checkpoint: {save_checkpoint(model, optimizer, args, step)}")

    if args.steps % args.save_every:
        final_path = save_checkpoint(model, optimizer, args, args.steps)
    else:
        final_path = args.work_dir / f"mioh-restorer-v1-step-{args.steps:07d}.pth"
    print(f"training complete: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
