#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Train the resolution-independent Native-HF 512 prototype.

The BasicVSR++ guide is always frozen and sees the full nine-frame window at
256px.  Only the five-frame native high-frequency refiner is optimized.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from lada.models.mioh_restorer.losses_native_hf import (
    MiohNativeHF512Loss,
    NativeHFLossWeights,
    eroded_roi_mask,
    missing_detail_oracle,
    native_detail_innovation,
)
from lada.models.mioh_restorer.losses_v5 import (
    high_frequency,
    masked_correlation,
    masked_mean,
)
from lada.models.mioh_restorer.model_native_hf import (
    FrozenBasicVSRPP256Guide,
    MiohNativeHF512,
    NATIVE_HF_INITIALIZATION_RECIPE,
    NATIVE_HF_MODEL_INITIALIZATION_SEED,
    NativeHF512Config,
    build_mioh_native_hf512,
    native_hf_parameter_count,
)
from lada.models.mioh_restorer.native_hf_dataset import MiohNativeHF512Dataset
from lada.models.mioh_restorer.supervision_native_hf import (
    NATIVE_HF_MAXIMUM_TRANSLATION,
    known_motion_alignment_loss,
)


STAGE_PREDECESSORS = {"joint": "hf-bootstrap"}
CHECKPOINT_FORMAT = "mioh-native-hf-512-v7"
DETAIL_SKIP_LEARNING_RATE_SCALE = 0.015
HF_BOOTSTRAP_MINIMUM_BLOCK_SIZE = 6
HF_BOOTSTRAP_MAXIMUM_BLOCK_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--basicvsrpp-checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="load --resume and run fixed raw/EMA validation without training",
    )
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-validation-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--stage",
        choices=("hf-bootstrap", "joint"),
        default="hf-bootstrap",
        help=(
            "hf-bootstrap starts from the frozen analytic alignment and learns "
            "the residual path on recoverable small mosaics; joint initializes "
            "from the completed hf-bootstrap EMA and calibrates the restoration "
            "and confidence heads on the deployment block-size distribution"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.train_manifest,
        args.validation_manifest,
        args.basicvsrpp_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    positive = (
        "steps",
        "batch_size",
        "accumulate",
        "learning_rate",
        "warmup_steps",
        "save_every",
        "validate_every",
        "validation_batches",
        "log_every",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        raise ValueError("Native-HF numeric arguments must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if not 0 < args.ema_decay < 1:
        raise ValueError("ema decay must be between zero and one")
    if args.resume and args.initialize_from:
        raise ValueError("resume and initialize-from are mutually exclusive")
    if args.evaluate_only and not args.resume:
        raise ValueError("evaluate-only requires --resume")
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if args.initialize_from and not args.initialize_from.is_file():
        raise FileNotFoundError(args.initialize_from)
    if args.initialize_from and args.stage not in STAGE_PREDECESSORS:
        raise ValueError("hf-bootstrap must start from the fixed initialization")
    if args.stage in STAGE_PREDECESSORS and not (args.resume or args.initialize_from):
        raise ValueError(
            f"{args.stage} requires --initialize-from from its completed "
            "predecessor (or --resume for the same stage)"
        )


def device_for(name: str) -> torch.device:
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


def central_refiner_window(
    guided_nine: torch.Tensor, model: MiohNativeHF512
) -> torch.Tensor:
    count = model.config.input_frames
    start = (guided_nine.shape[1] - count) // 2
    return guided_nine[:, start : start + count]


class EpochRandomSampler(Sampler[int]):
    """Deterministic epoch permutation with an exact resume offset."""

    def __init__(self, data_source: object, *, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        if start_index < 0 or start_index > len(self.data_source):
            raise ValueError("training sampler resume offset is invalid")
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            len(self.data_source), generator=generator
        ).tolist()
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = loader.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(dataset, MiohNativeHF512Dataset):
        dataset.set_epoch(epoch)


def make_loader(
    manifest: Path,
    *,
    stage: str,
    batch_size: int,
    workers: int,
    training: bool,
    limit: int | None,
    seed: int,
) -> DataLoader:
    bootstrap = stage == "hf-bootstrap"
    dataset = MiohNativeHF512Dataset(
        manifest,
        native_size=512,
        output_indices=(4,),
        # Learn the inverse mapping from clean native observations before
        # introducing compression/noise in the deployment-calibration stage.
        degrade=training and not bootstrap,
        time_reverse=training,
        deterministic=not training,
        minimum_block_size=(
            HF_BOOTSTRAP_MINIMUM_BLOCK_SIZE if bootstrap else 6
        ),
        maximum_block_size=(
            HF_BOOTSTRAP_MAXIMUM_BLOCK_SIZE if bootstrap else 48
        ),
        block_size_sampling="uniform" if bootstrap else "manifest",
        seed=seed,
    )
    if limit is not None:
        if training:
            indices = list(range(min(limit, len(dataset))))
        else:
            indices = source_balanced_validation_indices(dataset, limit)
        dataset = Subset(dataset, indices)
    sampler = EpochRandomSampler(dataset, seed=seed) if training else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        # Recreate workers at epoch boundaries so dataset.set_epoch() is
        # visible in worker-local copies and resume remains deterministic.
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )


def source_balanced_validation_indices(
    dataset: MiohNativeHF512Dataset, count: int
) -> list[int]:
    """Choose deterministic validation samples across sources and time.

    Manifest entries are grouped by source video, so taking the first N entries
    silently evaluates one scene.  Allocate the budget across source videos and
    spread each source's picks over its complete timeline instead.
    """

    count = min(int(count), len(dataset))
    if count <= 0:
        return []
    grouped: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, entry in enumerate(dataset.entries):
        grouped[entry.source_video_id][str(entry.target_video)].append(index)
    sources = sorted(grouped)
    selected_by_source: list[list[int]] = []
    base, remainder = divmod(count, len(sources))
    for source_index, source in enumerate(sources):
        budget = base + (1 if source_index < remainder else 0)
        clips = sorted(grouped[source].values(), key=lambda values: values[0])
        budget = min(budget, len(clips))
        if budget == 0:
            selected_by_source.append([])
            continue
        clip_positions = np.floor(
            (np.arange(budget, dtype=np.float64) + 0.5)
            * len(clips)
            / budget
        ).astype(np.int64)
        selected_by_source.append(
            [
                clips[int(position)][len(clips[int(position)]) // 2]
                for position in clip_positions
            ]
        )

    # Interleave sources so even a smaller validation-batches limit stays
    # source-balanced. Fill any shortfall from still-unused entries.
    selected: list[int] = []
    for offset in range(max(map(len, selected_by_source), default=0)):
        for values in selected_by_source:
            if offset < len(values):
                selected.append(values[offset])
    if len(selected) < count:
        used = set(selected)
        for index in range(len(dataset)):
            if index not in used:
                selected.append(index)
                if len(selected) == count:
                    break
    return selected[:count]


def to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    cpu_metadata = {"mosaic_phases", "mosaic_block_size"}
    result: dict[str, object] = {}
    for key, value in batch.items():
        result[key] = (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor) and key not in cpu_metadata
            else value
        )
    return result


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for target, source in zip(ema.parameters(), model.parameters(), strict=True):
        target.lerp_(source.detach(), 1.0 - decay)
    for target, source in zip(ema.buffers(), model.buffers(), strict=True):
        target.copy_(source)


def atomic_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def input_identities(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    return {
        "basicvsrpp_checkpoint": file_identity(args.basicvsrpp_checkpoint),
        "train_manifest": file_identity(args.train_manifest),
        "validation_manifest": file_identity(args.validation_manifest),
    }


def stage_loss_weights(stage: str) -> NativeHFLossWeights:
    """Return the versioned objective for an independent training stage."""

    if stage == "hf-bootstrap":
        # Innovation remains the only metric that can promote this stage, but
        # a controlled block-8 overfit showed that using it alone throws away
        # useful reconstruction/HF gradients.  These auxiliary terms recover
        # image fidelity while the source-only short skip and explicit span
        # penalty keep BasicVSR++ smoothing from becoming the easy solution.
        return NativeHFLossWeights(
            reconstruction=1.0,
            residual=0.0,
            missing_detail=0.0,
            non_detail_suppression=0.0,
            innovation=0.25,
            innovation_span=0.10,
            innovation_zero=0.10,
            fidelity_guard=0.10,
            high_frequency=0.50,
            gradient=0.15,
            wavelet=0.15,
            observation=0.0,
            low_frequency_drift=0.05,
            confidence=0.0,
            confidence_regularization=0.0,
        )
    if stage == "joint":
        return NativeHFLossWeights()
    raise ValueError(f"unsupported Native-HF stage: {stage}")


def configure_stage(
    model: MiohNativeHF512, stage: str, *, initialize_stage: bool
) -> NativeHFLossWeights:
    model.requires_grad_(False)
    if stage == "hf-bootstrap":
        model.decoder.requires_grad_(True)
        model.decoder.alignment.requires_grad_(False)
        model.decoder.confidence_head.requires_grad_(False)
        if initialize_stage:
            with torch.no_grad():
                model.decoder.confidence_head[-1].weight.zero_()
                # sigmoid(4) = 0.982: the residual receives nearly its full
                # reconstruction gradient while zero residual still preserves
                # the exact global-base identity at initialization.
                model.decoder.confidence_head[-1].bias.fill_(4.0)
        return stage_loss_weights(stage)
    if stage == "joint":
        # The analytic correlation path is a fixed deployment primitive.
        # Updating either its encoder features or offset logits degraded held-
        # out motion in the Stage-0 experiment, so the final stage calibrates
        # the usable residual/confidence without moving alignment itself.
        model.decoder.requires_grad_(True)
        model.decoder.alignment.requires_grad_(False)
        model.decoder.alignment.pair_gate.requires_grad_(True)
        return stage_loss_weights(stage)
    raise ValueError(f"unsupported Native-HF stage: {stage}")


def optimizer_groups(
    model: MiohNativeHF512, stage: str, learning_rate: float
) -> list[dict[str, object]]:
    """Use a bounded LR for the zero-init, high-fan-in detail skip.

    A real clean-input overfit showed that the normal 2e-4 LR overshoots this
    path on its first Adam update.  The stable 0.015 scale (3e-6 at the default
    LR) lets the nonlinear fusion path learn before the high-fan-in linear skip
    can dominate the output with uncorrelated detail.
    """

    detail_ids = {
        id(parameter)
        for parameter in model.decoder.detail_skip.parameters()
        if parameter.requires_grad
    }
    regular = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in detail_ids
    ]
    groups: list[dict[str, object]] = []
    if regular:
        groups.append(
            {
                "params": regular,
                "lr": learning_rate,
                "lr_scale": 1.0,
            }
        )
    detail = [
        parameter
        for parameter in model.decoder.detail_skip.parameters()
        if parameter.requires_grad
    ]
    if detail:
        if stage not in ("hf-bootstrap", "joint"):
            raise ValueError("detail skip is trainable outside an HF stage")
        groups.append(
            {
                "params": detail,
                "lr": learning_rate * DETAIL_SKIP_LEARNING_RATE_SCALE,
                "lr_scale": DETAIL_SKIP_LEARNING_RATE_SCALE,
                "weight_decay": 0.0,
            }
        )
    if not groups:
        raise ValueError("Native-HF stage has no trainable parameters")
    return groups


def known_motion_curriculum(
    *, step: int, steps: int, minimum: float, maximum: float
) -> float:
    """Grow analytic motion range only after easier alignment is learned."""

    if steps <= 1:
        return maximum
    fraction = min(max(step - 1, 0), steps - 1) / float(steps - 1)
    return minimum + (maximum - minimum) * fraction


def save_checkpoint(
    model: MiohNativeHF512,
    ema: MiohNativeHF512,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    step: int,
    epoch: int,
    batch_in_epoch: int,
) -> Path:
    payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "initialization_recipe": NATIVE_HF_INITIALIZATION_RECIPE,
        "model_initialization_seed": NATIVE_HF_MODEL_INITIALIZATION_SEED,
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "basicvsrpp_checkpoint": str(args.basicvsrpp_checkpoint),
        "stage": args.stage,
        "loss_weights": asdict(stage_loss_weights(args.stage)),
        "initialized_from": (
            str(args.initialize_from.resolve()) if args.initialize_from else None
        ),
        "initialized_from_identity": (
            file_identity(args.initialize_from) if args.initialize_from else None
        ),
        "inputs": input_identities(args),
        "arguments": vars(args),
    }
    numbered = args.work_dir / f"mioh-native-hf-512-step-{step:06d}.pth"
    latest = args.work_dir / "mioh-native-hf-512-latest.pth"
    atomic_save(payload, numbered)
    temporary = latest.with_suffix(latest.suffix + ".tmp")
    shutil.copyfile(numbered, temporary)
    temporary.replace(latest)
    return numbered


def compatible_checkpoint_payload(
    path: Path,
    model: MiohNativeHF512,
    expected_inputs: dict[str, dict[str, object]],
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not a Native-HF 512 checkpoint")
    if payload.get("initialization_recipe") != NATIVE_HF_INITIALIZATION_RECIPE:
        raise ValueError("checkpoint Native-HF initialization recipe does not match")
    if payload.get("model_initialization_seed") != NATIVE_HF_MODEL_INITIALIZATION_SEED:
        raise ValueError("checkpoint Native-HF initialization seed does not match")
    if payload.get("config") != asdict(model.config):
        raise ValueError("checkpoint Native-HF configuration does not match")
    if payload.get("inputs") != expected_inputs:
        raise ValueError(
            "checkpoint guide or manifest identity does not match this run"
        )
    return payload


def load_checkpoint(
    path: Path,
    model: MiohNativeHF512,
    ema: MiohNativeHF512,
    optimizer: torch.optim.Optimizer,
    expected_inputs: dict[str, dict[str, object]],
    expected_stage: str,
) -> tuple[int, int, int]:
    payload = compatible_checkpoint_payload(path, model, expected_inputs)
    if payload.get("stage") != expected_stage:
        raise ValueError("resume checkpoint Native-HF stage does not match this run")
    if payload.get("loss_weights") != asdict(stage_loss_weights(expected_stage)):
        raise ValueError("resume checkpoint Native-HF loss recipe does not match")
    model.load_state_dict(payload["state_dict"], strict=True)
    ema.load_state_dict(payload.get("ema_state_dict", payload["state_dict"]), strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    step = int(payload.get("step", 0))
    epoch = int(payload.get("epoch", 0))
    # v5 pilots created before exact cursor persistence were batch-size one,
    # accumulate one and never completed a full 92k-sample epoch.
    batch_in_epoch = int(payload.get("batch_in_epoch", step if epoch == 0 else 0))
    return step, epoch, batch_in_epoch


def initialize_from_checkpoint(
    path: Path,
    model: MiohNativeHF512,
    expected_inputs: dict[str, dict[str, object]],
    target_stage: str,
) -> str:
    payload = compatible_checkpoint_payload(path, model, expected_inputs)
    source_stage = str(payload.get("stage", ""))
    expected_source = STAGE_PREDECESSORS.get(target_stage)
    if expected_source is None or source_stage != expected_source:
        raise ValueError(
            f"Native-HF stage {source_stage!r} cannot initialize {target_stage!r}"
        )
    if payload.get("loss_weights") != asdict(stage_loss_weights(source_stage)):
        raise ValueError("parent checkpoint Native-HF loss recipe does not match")
    if "ema_state_dict" not in payload:
        raise ValueError("Native-HF parent checkpoint has no EMA state")
    model.load_state_dict(payload["ema_state_dict"], strict=True)
    return source_stage


def append_json(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def block_size_bucket(block_size: int) -> str:
    if block_size <= 8:
        return "block_small_le8"
    if block_size <= 16:
        return "block_medium_9_16"
    return "block_large_ge17"


def missing_detail_metrics(
    correction: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    baseline: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure native detail recovery without rewarding guide smoothing."""

    oracle, support = missing_detail_oracle(target, base, mask)
    weighted_correction = correction * support
    weighted_oracle = oracle * support
    dot = (weighted_correction * weighted_oracle).sum()
    correction_energy = weighted_correction.square().sum()
    oracle_energy = weighted_oracle.square().sum()
    denominator = torch.sqrt(correction_energy * oracle_energy + 1e-12)
    cosine = dot / denominator
    gain = dot / oracle_energy.clamp_min(1e-12)
    zero_error = masked_mean(torch.abs(oracle), support)
    corrected_error = masked_mean(torch.abs(oracle - correction), support)
    recovery = 100.0 * (zero_error - corrected_error) / zero_error.clamp_min(1e-12)

    detail_candidate = baseline + support * correction
    baseline_mse = masked_mean((baseline - target).square(), mask)
    detail_mse = masked_mean((detail_candidate - target).square(), mask)
    detail_psnr_delta = (
        -10.0 * torch.log10(detail_mse.clamp_min(1e-12))
        + 10.0 * torch.log10(baseline_mse.clamp_min(1e-12))
    )

    inner = eroded_roi_mask(mask, radius=2).expand_as(target)
    support_fraction = support.sum() / inner.sum().clamp_min(1.0)

    # Per-sample, per-output, per-channel oracle scalar for the known nuisance
    # family alpha*HF(base), alpha <= 0.  This is intentionally optimistic: a
    # learned model must add useful detail, not merely match cheap smoothing.
    basis = high_frequency(base)
    desired = target - baseline
    numerator = (desired * basis * inner).sum(dim=(-2, -1), keepdim=True)
    basis_energy = (basis.square() * inner).sum(
        dim=(-2, -1), keepdim=True
    )
    smoothing_alpha = (numerator / basis_energy.clamp_min(1e-12)).clamp(-2.0, 0.0)
    smoothing_correction = smoothing_alpha * basis
    smoothing_candidate = baseline + mask * smoothing_correction
    smoothing_mse = masked_mean((smoothing_candidate - target).square(), mask)
    smoothing_psnr_delta = (
        -10.0 * torch.log10(smoothing_mse.clamp_min(1e-12))
        + 10.0 * torch.log10(baseline_mse.clamp_min(1e-12))
    )
    restored = baseline + correction
    restored_mse = masked_mean((restored - target).square(), mask)
    over_smoothing = (
        -10.0 * torch.log10(restored_mse.clamp_min(1e-12))
        + 10.0 * torch.log10(smoothing_mse.clamp_min(1e-12))
    )
    return {
        "missing_detail_cosine": cosine,
        "missing_detail_gain": gain,
        "missing_detail_recovery_percent": recovery,
        "missing_detail_positive": (recovery > 0).to(recovery.dtype),
        "missing_detail_roi_psnr_delta": detail_psnr_delta,
        "missing_detail_support_fraction": support_fraction,
        "missing_detail_oracle_rms": torch.sqrt(
            oracle_energy / support.sum().clamp_min(1.0)
        ),
        "oracle_smoothing_roi_psnr_delta": smoothing_psnr_delta,
        "roi_psnr_over_oracle_smoothing": over_smoothing,
        "oracle_smoothing_alpha": smoothing_alpha.mean(),
    }


@torch.no_grad()
def validate(
    model: MiohNativeHF512,
    guide: FrozenBasicVSRPP256Guide,
    loader: DataLoader,
    *,
    device: torch.device,
    limit: int,
) -> dict[str, float]:
    model.eval()
    guide.eval()
    totals: dict[str, float] = defaultdict(float)
    metric_counts: dict[str, int] = defaultdict(int)
    bucket_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    bucket_metric_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    bucket_counts: dict[str, int] = defaultdict(int)
    count = 0
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        native = batch["native_inputs"]
        target = batch["targets"]
        mask = batch["masks"]
        if not all(isinstance(value, torch.Tensor) for value in (native, target, mask)):
            raise TypeError("Native-HF validation batch is malformed")
        guided_nine = guide(native)
        guided = central_refiner_window(guided_nine, model)
        restored, confidence, residual, base = model.forward_components(guided)
        source = guided[:, list(model.config.output_indices), :3]
        baseline = source + mask * (base - source)
        effective_correction = restored - baseline
        baseline_mse = masked_mean((baseline - target).square(), mask)
        restored_mse = masked_mean((restored - target).square(), mask)
        base_hf_error = masked_mean(
            torch.abs(high_frequency(baseline) - high_frequency(target)), mask
        )
        restored_hf_error = masked_mean(
            torch.abs(high_frequency(restored) - high_frequency(target)), mask
        )
        residual_target = high_frequency(target - base)
        residual_zero_error = masked_mean(torch.abs(residual_target), mask)
        residual_error = masked_mean(
            torch.abs(residual - residual_target), mask
        )
        residual_target_rms = torch.sqrt(
            masked_mean(residual_target.square(), mask).clamp_min(1e-12)
        )
        residual_rms = torch.sqrt(
            masked_mean(residual.square(), mask).clamp_min(0.0)
        )
        # Feathered boundary pixels are intentionally part of the ROI blend.
        # Exact preservation applies where the mask is exactly zero.
        outside = (mask <= 0).expand_as(restored)
        outside_max = torch.max(torch.abs((restored - source) * outside))
        detail_metrics = missing_detail_metrics(
            effective_correction,
            target,
            base,
            baseline,
            mask,
        )
        _innovation, _span, _zero, innovation_metrics = (
            native_detail_innovation(
                effective_correction,
                target,
                base,
                source,
                mask,
            )
        )
        sample_metrics = {
            "roi_psnr": float(
                -10 * torch.log10(restored_mse.clamp_min(1e-12))
            ),
            "baseline_roi_psnr": float(
                -10 * torch.log10(baseline_mse.clamp_min(1e-12))
            ),
            "roi_psnr_delta": float(
                -10 * torch.log10(restored_mse.clamp_min(1e-12))
                + 10 * torch.log10(baseline_mse.clamp_min(1e-12))
            ),
            "mse_reduction_percent": float(
                100
                * (baseline_mse - restored_mse)
                / baseline_mse.clamp_min(1e-12)
            ),
            "hf_error_reduction_percent": float(
                100
                * (base_hf_error - restored_hf_error)
                / base_hf_error.clamp_min(1e-12)
            ),
            "residual_target_correlation": float(
                masked_correlation(residual, residual_target, mask)
            ),
            "residual_amplitude_ratio": float(
                residual_rms / residual_target_rms
            ),
            "residual_mae_reduction_percent": float(
                100
                * (residual_zero_error - residual_error)
                / residual_zero_error.clamp_min(1e-12)
            ),
            "outside_roi_maximum": float(outside_max),
            "confidence_mean": float(masked_mean(confidence, mask)),
            **{
                name: float(value)
                for name, value in detail_metrics.items()
            },
            **{
                name: float(value)
                for name, value in innovation_metrics.items()
            },
        }
        block_size = batch.get("mosaic_block_size")
        if not isinstance(block_size, torch.Tensor) or block_size.numel() != 1:
            raise TypeError("Native-HF validation block size is malformed")
        bucket = block_size_bucket(int(block_size.item()))
        innovation_valid = sample_metrics.get("innovation_valid", 0.0) > 0
        for name, value in sample_metrics.items():
            if (
                name.startswith("innovation_")
                and name not in ("innovation_valid", "innovation_valid_patches")
                and not innovation_valid
            ):
                continue
            totals[name] += value
            metric_counts[name] += 1
            bucket_totals[bucket][name] += value
            bucket_metric_counts[bucket][name] += 1
        bucket_counts[bucket] += 1
        count += 1
        if count >= limit:
            break
    if not count:
        raise RuntimeError("Native-HF validation produced no batches")
    result = {
        name: value / metric_counts[name] for name, value in totals.items()
    }
    for bucket in (
        "block_small_le8",
        "block_medium_9_16",
        "block_large_ge17",
    ):
        sample_count = bucket_counts[bucket]
        result[f"{bucket}_samples"] = float(sample_count)
        if sample_count:
            result.update(
                {
                    f"{bucket}_{name}": value / sample_count
                    if name in ("innovation_valid", "innovation_valid_patches")
                    else value / bucket_metric_counts[bucket][name]
                    for name, value in bucket_totals[bucket].items()
                }
            )
    return result


def accelerator_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def restore_accelerator_rng_state(
    device: torch.device, state: torch.Tensor | None
) -> None:
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)


def seed_validation_motion(device: torch.device, seed: int) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


@torch.no_grad()
def validate_alignment(
    model: MiohNativeHF512,
    guide: FrozenBasicVSRPP256Guide,
    loader: DataLoader,
    *,
    device: torch.device,
    limit: int,
    maximum_translation: float,
    seed: int,
) -> dict[str, float]:
    """Evaluate exact-motion alignment without perturbing training RNG state."""

    model.eval()
    guide.eval()
    cpu_rng_state = torch.get_rng_state()
    device_rng_state = accelerator_rng_state(device)
    totals: dict[str, float] = defaultdict(float)
    endpoint_errors: list[float] = []
    count = 0
    try:
        seed_validation_motion(device, seed)
        for raw_batch in loader:
            batch = to_device(raw_batch, device)
            native = batch["native_inputs"]
            if not isinstance(native, torch.Tensor):
                raise TypeError("Native-HF alignment validation batch is malformed")
            guided_nine = guide(native)
            guided = central_refiner_window(guided_nine, model)
            _loss, stats = known_motion_alignment_loss(
                model,
                guided,
                maximum_translation=maximum_translation,
            )
            for name, value in stats.items():
                totals[name] += float(value.cpu())
            endpoint_errors.append(float(stats["known_motion_epe"].cpu()))
            count += 1
            if count >= limit:
                break
    finally:
        torch.set_rng_state(cpu_rng_state)
        restore_accelerator_rng_state(device, device_rng_state)
    if not count:
        raise RuntimeError("Native-HF alignment validation produced no batches")
    result = {name: value / count for name, value in totals.items()}
    # Validation uses batch size one, so the per-batch p95 would equal the
    # sample EPE.  Recompute it across the complete held-out sample set.
    result["known_motion_epe_p95"] = float(
        np.percentile(np.asarray(endpoint_errors, dtype=np.float64), 95)
    )
    return result


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = device_for(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.work_dir / "metrics.jsonl"

    identities = input_identities(args)
    model = build_mioh_native_hf512()
    initialized_from_stage: str | None = None
    if args.initialize_from:
        initialized_from_stage = initialize_from_checkpoint(
            args.initialize_from,
            model,
            identities,
            args.stage,
        )
    loss_weights = configure_stage(
        model, args.stage, initialize_stage=args.resume is None
    )
    model.to(device)
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    guide = FrozenBasicVSRPP256Guide.from_checkpoint(
        args.basicvsrpp_checkpoint, use_ema=True
    ).to(device)
    guide.eval()
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, args.stage, args.learning_rate),
        lr=args.learning_rate,
    )
    loss_function = (
        MiohNativeHF512Loss(weights=loss_weights).to(device)
        if loss_weights is not None
        else None
    )
    train_loader = make_loader(
        args.train_manifest,
        stage=args.stage,
        batch_size=args.batch_size,
        workers=args.workers,
        training=True,
        limit=args.limit_train_samples,
        seed=args.seed,
    )
    validation_loader = make_loader(
        args.validation_manifest,
        stage=args.stage,
        batch_size=1,
        workers=min(args.workers, 1),
        training=False,
        limit=args.limit_validation_samples or args.validation_batches,
        seed=args.seed,
    )
    step = 0
    epoch = 0
    batch_in_epoch = 0
    if args.resume:
        step, epoch, batch_in_epoch = load_checkpoint(
            args.resume, model, ema, optimizer, identities, args.stage
        )

    startup: dict[str, object] = {
        "event": "start",
        "format": CHECKPOINT_FORMAT,
        "stage": args.stage,
        "device": str(device),
        "steps": args.steps,
        "starting_step": step,
        "starting_epoch": epoch,
        "starting_batch_in_epoch": batch_in_epoch,
        "initialized_from": str(args.initialize_from) if args.initialize_from else None,
        "initialized_from_identity": (
            file_identity(args.initialize_from) if args.initialize_from else None
        ),
        "initialized_from_stage": initialized_from_stage,
        "fresh_optimizer": args.resume is None,
        "optimizer_lr_scales": [
            float(group.get("lr_scale", 1.0)) for group in optimizer.param_groups
        ],
        "initialization_recipe": NATIVE_HF_INITIALIZATION_RECIPE,
        "model_initialization_seed": NATIVE_HF_MODEL_INITIALIZATION_SEED,
        "parameters": native_hf_parameter_count(model),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "global_guide_trainable_parameters": sum(
            parameter.numel() for parameter in guide.parameters() if parameter.requires_grad
        ),
        "global_guide_size": 256,
        "native_size": 512,
        "global_frames": 9,
        "native_refiner_frames": model.config.input_frames,
        "block_size_sampling": (
            "uniform" if args.stage == "hf-bootstrap" else "manifest"
        ),
        "minimum_block_size": (
            HF_BOOTSTRAP_MINIMUM_BLOCK_SIZE
            if args.stage == "hf-bootstrap"
            else 6
        ),
        "maximum_block_size": (
            HF_BOOTSTRAP_MAXIMUM_BLOCK_SIZE
            if args.stage == "hf-bootstrap"
            else 48
        ),
        "degradation_curriculum": (
            "clean-only" if args.stage == "hf-bootstrap" else "clean-mild-full"
        ),
        "loss_weights": asdict(stage_loss_weights(args.stage)),
        "output_indices": model.config.output_indices,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "gt_primary": True,
        "analytic_motion_supervision": False,
        "gan": False,
        "perceptual_teacher": False,
        "inputs": identities,
    }
    print(json.dumps(startup, ensure_ascii=False, indent=2), flush=True)
    append_json(metrics_path, startup)

    if args.evaluate_only:
        raw_result = validate(
            model,
            guide,
            validation_loader,
            device=device,
            limit=args.validation_batches,
        )
        ema_result = validate(
            ema,
            guide,
            validation_loader,
            device=device,
            limit=args.validation_batches,
        )
        record = {
            "event": "evaluation",
            "step": step,
            **{f"raw_{name}": value for name, value in raw_result.items()},
            **{f"ema_{name}": value for name, value in ema_result.items()},
        }
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        append_json(metrics_path, record)
        return 0

    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    session_started = time.perf_counter()
    session_start_step = step
    try:
        while step < args.steps:
            set_loader_epoch(train_loader, epoch)
            sampler = train_loader.sampler
            if not isinstance(sampler, EpochRandomSampler):
                raise TypeError("Native-HF training sampler is not resumable")
            sampler.set_epoch(
                epoch, start_index=batch_in_epoch * args.batch_size
            )
            for raw_batch in train_loader:
                batch_in_epoch += 1
                batch = to_device(raw_batch, device)
                native = batch["native_inputs"]
                if not isinstance(native, torch.Tensor):
                    raise TypeError("native_inputs is not a tensor")
                targets = batch["targets"]
                masks = batch["masks"]
                observations = batch["mosaic_observations"]
                phases = batch["mosaic_phases"]
                block_size = batch["mosaic_block_size"]
                observation_weight = batch["observation_weight"]
                if not all(
                    isinstance(value, torch.Tensor)
                    for value in (
                        targets,
                        masks,
                        observations,
                        phases,
                        block_size,
                        observation_weight,
                    )
                ):
                    raise TypeError("Native-HF training metadata is malformed")
                if not torch.count_nonzero(masks):
                    continue
                guided_nine = guide(native)
                guided = central_refiner_window(guided_nine, model).detach()
                if loss_function is None:
                    raise RuntimeError("Native-HF restoration loss is unavailable")
                restored, confidence, residual, base = model.forward_components(
                    guided
                )
                source = guided[:, list(model.config.output_indices), :3]
                loss, stats = loss_function(
                    restored,
                    confidence,
                    residual,
                    base,
                    targets,
                    source,
                    masks,
                    observations,
                    phases,
                    block_size,
                    observation_weight,
                )
                (loss / args.accumulate).backward()
                accumulated += 1
                if accumulated < args.accumulate:
                    continue

                next_step = step + 1
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip
                    )
                warmup = min(args.warmup_steps, max(1, args.steps // 10))
                learning_rate = args.learning_rate * min(1.0, next_step / warmup)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model, args.ema_decay)
                accumulated = 0
                step = next_step

                if step == 1 or step % args.log_every == 0:
                    if device.type == "mps":
                        torch.mps.synchronize()
                    elapsed = time.perf_counter() - session_started
                    session_steps = max(step - session_start_step, 1)
                    record: dict[str, object] = {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "seconds_per_step": elapsed / session_steps,
                        "learning_rates": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                        **{
                            name: float(value.cpu())
                            for name, value in stats.items()
                        },
                    }
                    if device.type == "mps":
                        record.update(
                            {
                                "mps_allocated_gib": torch.mps.current_allocated_memory() / 2**30,
                                "mps_driver_gib": torch.mps.driver_allocated_memory() / 2**30,
                            }
                        )
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    append_json(metrics_path, record)
                if step % args.validate_every == 0 or step == args.steps:
                    raw_result = validate(
                        model,
                        guide,
                        validation_loader,
                        device=device,
                        limit=args.validation_batches,
                    )
                    ema_result = validate(
                        ema,
                        guide,
                        validation_loader,
                        device=device,
                        limit=args.validation_batches,
                    )
                    record = {
                        "event": "validation",
                        "step": step,
                        **{f"raw_{name}": value for name, value in raw_result.items()},
                        **{f"ema_{name}": value for name, value in ema_result.items()},
                    }
                    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
                    append_json(metrics_path, record)
                    model.train()
                    guide.eval()
                    if device.type == "mps":
                        torch.mps.empty_cache()
                if step % args.save_every == 0 or step == args.steps:
                    saved = save_checkpoint(
                        model,
                        ema,
                        optimizer,
                        args,
                        step=step,
                        epoch=epoch,
                        batch_in_epoch=batch_in_epoch,
                    )
                    print(f"saved: {saved}", flush=True)
                if step >= args.steps:
                    break
            epoch += 1
            batch_in_epoch = 0
    except KeyboardInterrupt:
        # Model/optimizer state reflects only complete accumulation groups.
        # Re-read any partially accumulated samples after resume.
        resume_batch = max(0, batch_in_epoch - accumulated)
        optimizer.zero_grad(set_to_none=True)
        saved = save_checkpoint(
            model,
            ema,
            optimizer,
            args,
            step=step,
            epoch=epoch,
            batch_in_epoch=resume_batch,
        )
        print(f"interrupted safely; saved: {saved}", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
