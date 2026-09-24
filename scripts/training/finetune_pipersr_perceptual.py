#!/usr/bin/env python3
"""Local detail-oriented PiperSR fine-tune on aligned 2K/4K frame pairs.

Uses low-frequency fidelity, MobileNet feature distance, local high-frequency
energy matching, and a small patch discriminator. GAN detail is not guaranteed
to be true to the source. Always compare on a different video before use.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from finetune_pipersr_detail import (
    evaluate,
    high_frequency,
    load_coreml_weights,
    paired_image_pairs,
    save_comparison,
    verify_coreml_parity,
)


class PatchDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 3
        for output in (32, 64, 128, 256):
            layers.extend((nn.Conv2d(channels, output, 4, stride=2, padding=1),
                           nn.LeakyReLU(0.2, inplace=True)))
            channels = output
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Judge the fine structure itself. Otherwise the very small 4K-only
        # texture difference is overwhelmed by the matching low frequencies.
        return self.layers(high_frequency(image) * 16)


class PerceptualFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self.layers = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        ).features[:7].eval()
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])
        self.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = (image - self.mean) / self.std
        outputs = []
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index in (2, 4, 6):
                outputs.append(hidden)
        return tuple(outputs)


def local_detail_energy(image: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(high_frequency(image).abs(), 8, stride=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--hr-images", type=Path, required=True)
    parser.add_argument("--lr-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--generator-lr", type=float, default=2e-6)
    parser.add_argument("--adversarial-weight", type=float, default=0.003)
    parser.add_argument("--perceptual-weight", type=float, default=0.02)
    parser.add_argument("--detail-weight", type=float, default=0.2)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--low-frequency-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-start", type=int, default=100)
    parser.add_argument("--critic-warmup", type=int, default=0)
    parser.add_argument("--detail-fraction", type=float, default=0.0)
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1:
        parser.error("--steps and --batch must be positive")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    paths = sorted(args.hr_images.glob("*.png"))
    if len(paths) < 12:
        raise ValueError("at least 12 aligned frame pairs are needed")
    rng.shuffle(paths)
    heldout = paths[: max(4, len(paths) // 5)]
    training = paths[len(heldout) :]
    train_pairs = paired_image_pairs(
        training, args.lr_images, args.patch, args.steps * args.batch, args.seed,
        detail_fraction=args.detail_fraction,
    )
    test_pairs = paired_image_pairs(
        heldout, args.lr_images, args.patch, max(96, len(heldout) * 8), args.seed + 1
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    generator = load_coreml_weights(args.package).to(device)
    parity = verify_coreml_parity(generator.cpu(), args.package)
    generator.to(device)
    baseline = evaluate(generator, test_pairs, device)
    with torch.inference_mode():
        baseline_previews = [
            generator(lr[None].to(device))[0].cpu() for lr, _ in test_pairs[:4]
        ]
    print(f"baseline: {json.dumps(baseline)}", flush=True)
    discriminator = PatchDiscriminator().to(device)
    perceptual = PerceptualFeatures().to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(0.9, 0.99))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = []

    for warmup_step in range(args.critic_warmup):
        start = (warmup_step * args.batch) % len(train_pairs)
        selected = [train_pairs[(start + offset) % len(train_pairs)] for offset in range(args.batch)]
        lr = torch.stack([pair[0] for pair in selected]).to(device)
        hr = torch.stack([pair[1] for pair in selected]).to(device)
        with torch.no_grad():
            fake = generator(lr)
        real_score = discriminator(hr)
        fake_score = discriminator(fake)
        discriminator_loss = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        discriminator_optimizer.step()
        if (warmup_step + 1) % 100 == 0:
            print(
                f"critic warmup {warmup_step + 1}/{args.critic_warmup}: "
                f"loss={discriminator_loss.item():.5f} "
                f"real={real_score.mean().item():.5f} fake={fake_score.mean().item():.5f}",
                flush=True,
            )

    for step in range(args.steps):
        selected = train_pairs[step * args.batch : (step + 1) * args.batch]
        lr = torch.stack([pair[0] for pair in selected]).to(device)
        hr = torch.stack([pair[1] for pair in selected]).to(device)
        with torch.no_grad():
            fake_detached = generator(lr)
        real_score = discriminator(hr)
        fake_score = discriminator(fake_detached)
        discriminator_loss = (
            F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
        )
        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        prediction = generator(lr)
        pixel_loss = F.l1_loss(prediction, hr)
        low_frequency_loss = F.l1_loss(
            F.avg_pool2d(prediction, 4), F.avg_pool2d(hr, 4)
        )
        detail_loss = F.l1_loss(
            local_detail_energy(prediction), local_detail_energy(hr)
        )
        with torch.no_grad():
            target_features = perceptual(hr)
        predicted_features = perceptual(prediction)
        perceptual_loss = sum(
            F.l1_loss(predicted, target)
            for predicted, target in zip(predicted_features, target_features)
        ) / len(target_features)
        adversarial_loss = -discriminator(prediction).mean()
        loss = (
            args.pixel_weight * pixel_loss
            + args.low_frequency_weight * low_frequency_loss
            + args.detail_weight * detail_loss
            + args.perceptual_weight * perceptual_loss
        )
        if step >= args.adversarial_start:
            loss = loss + args.adversarial_weight * adversarial_loss
        generator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        generator_optimizer.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(
                f"step {step + 1}/{args.steps}: pixel={pixel_loss.item():.5f} "
                f"detail={detail_loss.item():.5f} perceptual={perceptual_loss.item():.5f} "
                f"D={discriminator_loss.item():.5f} Gadv={adversarial_loss.item():.5f}",
                flush=True,
            )
        if (step + 1) % 250 == 0 or step + 1 == args.steps:
            metrics = evaluate(generator, test_pairs, device)
            print(f"validation {step + 1}: {json.dumps(metrics)}", flush=True)
            checkpoint = args.output / f"pipersr-perceptual-step{step + 1}.pt"
            torch.save(generator.cpu().state_dict(), checkpoint)
            generator.to(device)
            checkpoints.append({"step": step + 1, "checkpoint": str(checkpoint), "metrics": metrics})

    save_comparison(args.output / "comparison.png", test_pairs, baseline_previews, generator, device)
    report = {
        "source_package": str(args.package),
        "hr_images": str(args.hr_images), "lr_images": str(args.lr_images),
        "train_images": [str(path) for path in training],
        "heldout_images": [str(path) for path in heldout],
        "parameters": vars(args) | {"package": str(args.package), "hr_images": str(args.hr_images),
                                      "lr_images": str(args.lr_images), "output": str(args.output)},
        "coreml_parity": parity, "baseline": baseline,
        "checkpoints": checkpoints,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
