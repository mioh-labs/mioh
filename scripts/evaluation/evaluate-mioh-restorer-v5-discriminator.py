#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Evaluate training-only V5-HQ discriminators on one fixed validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lada.models.mioh_restorer.adversarial import (
    SpectralUNetDiscriminator,
    TemporalPatchDiscriminator,
    discriminator_hinge_loss,
    discriminator_roi_patch_weights,
    roi_temporal_discriminator_input,
)
from lada.models.mioh_restorer.model_v5_hq import (
    MiohRestorerV5HQ,
    MiohRestorerV5HQConfig,
)
from lada.models.mioh_restorer.native_dataset_v5 import (
    MiohRestorerV5NativeDataset,
    V5BucketBatchSampler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"invalid checkpoint: {path}")
    return payload


def build_discriminator(payload: dict[str, object], device: torch.device) -> torch.nn.Module:
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("checkpoint has no training arguments")
    architecture = str(arguments.get("hq_gan_discriminator_architecture", "patch"))
    channels = int(arguments.get("hq_gan_discriminator_channels", 16))
    classes = {
        "patch": TemporalPatchDiscriminator,
        "unet-sn": SpectralUNetDiscriminator,
    }
    if architecture not in classes:
        raise ValueError(f"unknown discriminator architecture: {architecture}")
    discriminator = classes[architecture](channels).to(device).eval()
    state = payload.get("discriminator_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no discriminator state")
    discriminator.load_state_dict(state, strict=True)
    return discriminator


def main() -> int:
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("batches must be positive")
    for path in (args.validation_manifest, *args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    first = load_payload(args.checkpoint[0])
    config = first.get("config")
    state = first.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint has no model config/state")
    model = MiohRestorerV5HQ(MiohRestorerV5HQConfig(**config)).to(device).eval()
    model.load_state_dict(state, strict=True)

    dataset = MiohRestorerV5NativeDataset(
        args.validation_manifest,
        output_indices=model.config.output_indices,
        degrade=False,
        horizontal_flip=False,
        time_reverse=False,
        deterministic=True,
    )
    sampler = V5BucketBatchSampler(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        seed=20260722,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    prepared: list[tuple[torch.Tensor, torch.Tensor]] = []
    first_arguments = first["arguments"]
    assert isinstance(first_arguments, dict)
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            masks = batch.get("loss_masks", batch["masks"]).to(device)
            if not bool(torch.any(masks > 0)):
                continue
            restored, _, _, _ = model.forward_components(inputs)
            options = {
                "frame_stride": int(first_arguments.get("hq_gan_frame_stride", 1)),
                "image_size": int(first_arguments.get("hq_gan_image_size", 192)),
                "crop_padding": int(first_arguments.get("hq_gan_crop_padding", 16)),
                "minimum_crop_size": int(
                    first_arguments.get("hq_gan_minimum_crop_size", 96)
                ),
                "include_motion": bool(first_arguments.get("hq_gan_temporal", False)),
                "normalize_secondary_rms": bool(
                    first_arguments.get("hq_gan_normalize_secondary_rms", False)
                ),
                "condition_on_target_rgb": not bool(
                    first_arguments.get("hq_gan_candidate_primary", False)
                ),
            }
            real = roi_temporal_discriminator_input(targets, targets, masks, **options)
            fake = roi_temporal_discriminator_input(restored, targets, masks, **options)
            prepared.append((real.cpu(), fake.cpu()))
            if len(prepared) >= args.batches:
                break
    if not prepared:
        raise RuntimeError("validation produced no discriminator inputs")

    results: list[dict[str, object]] = []
    for checkpoint in args.checkpoint:
        payload = load_payload(checkpoint)
        discriminator = build_discriminator(payload, device)
        totals = {
            "real_score": 0.0,
            "fake_score": 0.0,
            "hinge_loss": 0.0,
            "paired_preference": 0.0,
        }
        cells = 0
        batches = 0
        with torch.no_grad():
            for real_cpu, fake_cpu in prepared:
                real = real_cpu.to(device)
                fake = fake_cpu.to(device)
                real_logits = discriminator(real)
                fake_logits = discriminator(fake)
                weights = discriminator_roi_patch_weights(real, real_logits)
                weight_sum = int(weights.sum().item())
                if not weight_sum:
                    continue
                totals["real_score"] += float((real_logits * weights).sum())
                totals["fake_score"] += float((fake_logits * weights).sum())
                totals["hinge_loss"] += float(
                    discriminator_hinge_loss(real_logits, fake_logits, weights)
                )
                totals["paired_preference"] += float(
                    ((real_logits > fake_logits).to(weights.dtype) * weights).sum()
                )
                cells += weight_sum
                batches += 1
        result = {
            "checkpoint": str(checkpoint),
            "local_step": int(payload.get("local_step", -1)),
            "batches": batches,
            "roi_cells": cells,
            "real_score": totals["real_score"] / cells,
            "fake_score": totals["fake_score"] / cells,
            "paired_margin": (
                totals["real_score"] - totals["fake_score"]
            ) / cells,
            "paired_preference": totals["paired_preference"] / cells,
            "hinge_loss": totals["hinge_loss"] / max(batches, 1),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {"validation_batches": len(prepared), "results": results}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
