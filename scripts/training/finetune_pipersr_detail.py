#!/usr/bin/env python3
"""Experimental PiperSR detail fine-tune; never replaces the bundled model.

The public PiperSR release contains only a Core ML package. This script reads
its fused convolution weights through coremltools, reconstructs the same
PyTorch graph, checks Core ML parity, and fine-tunes on user-supplied HR images.
Keep dataset and checkpoints outside Git; verify quality before any export.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Conv2d(64, 64, 3, padding=1)
        self.second = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.second(F.silu(self.first(x)))


class PiperSR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, 64, 3, padding=1)
        self.blocks = nn.ModuleList(ResidualBlock() for _ in range(6))
        self.tail = nn.Conv2d(64, 12, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        head = self.head(image)
        hidden = head
        for block in self.blocks:
            hidden = block(hidden)
        return F.pixel_shuffle(self.tail(hidden + head), 2).clamp(0, 1)


def load_coreml_weights(package: Path) -> PiperSR:
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load

    coreml = ct.models.MLModel(str(package), skip_model_load=True)
    spec = coreml.get_spec()
    graph = load(
        spec,
        spec.specificationVersion,
        str(package / "Data/com.apple.CoreML/weights"),
    )
    convolutions = [
        op for op in graph.functions["main"].operations if op.op_type == "conv"
    ]
    model = PiperSR()
    targets = [model.head]
    for block in model.blocks:
        targets.extend((block.first, block.second))
    targets.append(model.tail)
    if len(convolutions) != len(targets):
        raise ValueError(f"expected {len(targets)} convolutions, found {len(convolutions)}")
    with torch.no_grad():
        for operation, layer in zip(convolutions, targets):
            weight = np.asarray(operation.inputs["weight"].val).copy()
            bias = np.asarray(operation.inputs["bias"].val).copy()
            if tuple(weight.shape) != tuple(layer.weight.shape):
                raise ValueError(f"unexpected weight shape in {operation.name}: {weight.shape}")
            layer.weight.copy_(torch.from_numpy(weight.astype(np.float32)))
            layer.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    return model


def verify_coreml_parity(model: PiperSR, package: Path) -> dict[str, float]:
    import coremltools as ct

    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, "RGB")
    tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float() / 255
    with torch.inference_mode():
        pytorch = model(tensor)[0].permute(1, 2, 0).numpy()
    coreml = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    result = coreml.predict({"input_image": image})
    output_image = next(iter(result.values())).convert("RGB")
    output = np.asarray(output_image, dtype=np.float32) / 255
    if output.shape != pytorch.shape:
        raise ValueError(f"Core ML output shape {output.shape} != {pytorch.shape}")
    difference = np.abs(output - pytorch)
    metrics = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}
    if metrics["mean_abs"] > 0.006 or metrics["max_abs"] > 0.08:
        raise ValueError(f"Core ML parity failed: {metrics}")
    return metrics


def tensor_from_image(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 255


def image_pairs(paths: list[Path], patch: int, count: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    pairs = []
    images = {}
    for path in paths:
        with Image.open(path) as loaded:
            images[path] = loaded.convert("RGB")
    for index in range(count):
        path = paths[index % len(paths)]
        image = images[path]
        if min(image.size) < patch * 2:
            continue
        x = rng.randrange(0, image.width - patch * 2 + 1)
        y = rng.randrange(0, image.height - patch * 2 + 1)
        hr = image.crop((x, y, x + patch * 2, y + patch * 2))
        # Real compressed videos are not perfectly bicubic. Keep the pilot
        # degradation mild so a small fine-tune cannot simply learn sharpening.
        if rng.random() < 0.6:
            hr_for_lr = hr.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.5)))
        else:
            hr_for_lr = hr
        lr = hr_for_lr.resize((patch, patch), Image.Resampling.BICUBIC)
        if rng.random() < 0.5:
            buffer = io.BytesIO()
            lr.save(buffer, format="JPEG", quality=rng.randrange(82, 97))
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                lr = compressed.convert("RGB")
        pairs.append((tensor_from_image(lr), tensor_from_image(hr)))
    return pairs


def paired_image_pairs(
    paths: list[Path], lr_directory: Path, patch: int, count: int, seed: int,
    detail_fraction: float = 0.0, min_detail: float = 0.0015,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Sample aligned LR/HR patches from real 2K/4K frames with shared stems."""
    rng = random.Random(seed)
    pairs = []
    frames = {}
    for hr_path in paths:
        lr_path = lr_directory / hr_path.name
        if not lr_path.is_file():
            raise FileNotFoundError(f"missing paired 2K frame: {lr_path}")
        with Image.open(hr_path) as loaded:
            hr_image = loaded.convert("RGB")
        with Image.open(lr_path) as loaded:
            lr_image = loaded.convert("RGB")
        if hr_image.size != (lr_image.width * 2, lr_image.height * 2):
            raise ValueError(f"2K/4K sizes are not 1:2: {lr_path} / {hr_path}")
        frames[hr_path] = (lr_image, hr_image)
    for index in range(count):
        detailed = rng.random() < detail_fraction
        best = None
        for attempt in range(8 if detailed else 1):
            path = rng.choice(paths) if detailed else paths[index % len(paths)]
            lr_image, hr_image = frames[path]
            if min(lr_image.size) < patch:
                continue
            x = rng.randrange(0, lr_image.width - patch + 1)
            y = rng.randrange(0, lr_image.height - patch + 1)
            if not detailed:
                best = (lr_image, hr_image, x, y)
                break
            import cv2

            crop = np.asarray(hr_image.crop((x * 2, y * 2, (x + patch) * 2, (y + patch) * 2)))
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            score = float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1)).mean() / 255)
            if best is None or score > best[0]:
                best = (score, lr_image, hr_image, x, y)
            if score >= min_detail:
                break
        if best is None:
            continue
        if detailed:
            _, lr_image, hr_image, x, y = best
        else:
            lr_image, hr_image, x, y = best
        lr = lr_image.crop((x, y, x + patch, y + patch))
        hr = hr_image.crop((x * 2, y * 2, (x + patch) * 2, (y + patch) * 2))
        pairs.append((tensor_from_image(lr), tensor_from_image(hr)))
    return pairs


def high_frequency(image: torch.Tensor) -> torch.Tensor:
    blurred = F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    return image - blurred


def evaluate(model: PiperSR, pairs: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> dict[str, float]:
    model.eval()
    accum = {"psnr": 0.0, "hf_mae": 0.0, "pred_hf": 0.0, "target_hf": 0.0,
             "hf_dot": 0.0, "pred_hf_squared": 0.0, "target_hf_squared": 0.0}
    with torch.inference_mode():
        for lr, hr in pairs:
            prediction = model(lr[None].to(device)).float()
            target = hr[None].to(device)
            mse = F.mse_loss(prediction, target).item()
            pred_hf = high_frequency(prediction)
            target_hf = high_frequency(target)
            accum["psnr"] += -10 * math.log10(max(mse, 1e-12))
            accum["hf_mae"] += F.l1_loss(pred_hf, target_hf).item()
            accum["pred_hf"] += pred_hf.abs().mean().item()
            accum["target_hf"] += target_hf.abs().mean().item()
            accum["hf_dot"] += (pred_hf * target_hf).mean().item()
            accum["pred_hf_squared"] += pred_hf.square().mean().item()
            accum["target_hf_squared"] += target_hf.square().mean().item()
    return {
        "psnr": accum["psnr"] / len(pairs),
        "hf_mae": accum["hf_mae"] / len(pairs),
        "hf_ratio": accum["pred_hf"] / max(accum["target_hf"], 1e-9),
        "hf_cosine": accum["hf_dot"] / max(
            math.sqrt(accum["pred_hf_squared"] * accum["target_hf_squared"]), 1e-9
        ),
    }


def as_image(tensor: torch.Tensor) -> Image.Image:
    pixels = (tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def save_comparison(
    path: Path,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    baseline: list[torch.Tensor],
    model: PiperSR,
    device: torch.device,
) -> None:
    samples = min(4, len(baseline))
    size = pairs[0][1].shape[-1]
    canvas = Image.new("RGB", (size * 4, size * samples + 22), "#202020")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Bicubic", "Original PiperSR", "Fine-tuned", "Ground truth")):
        draw.text((column * size + 4, 4), label, fill="white")
    model.eval()
    with torch.inference_mode():
        for row in range(samples):
            lr, hr = pairs[row]
            tuned = model(lr[None].to(device))[0]
            images = (
                as_image(lr).resize((size, size), Image.Resampling.BICUBIC),
                as_image(baseline[row]),
                as_image(tuned),
                as_image(hr),
            )
            for column, image in enumerate(images):
                canvas.paste(image, (column * size, 22 + row * size))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--lr-images", type=Path, help="Aligned real 2K PNGs with the same names as --images 4K PNGs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--detail-weight", type=float, default=0.0)
    parser.add_argument("--hf-weight", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Previously fine-tuned weights for independent evaluation")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = load_coreml_weights(args.package)
    parity = verify_coreml_parity(model, args.package)
    print(f"Core ML parity: {json.dumps(parity)}", flush=True)
    if args.verify_only:
        return
    if args.images is None or args.output is None:
        parser.error("--images and --output are required for training")
    paths = sorted(path for path in args.images.rglob("*.png"))[: args.max_images]
    if len(paths) < 6:
        raise ValueError("at least six PNG images are needed for a train/holdout split")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    holdout = paths[: max(2, len(paths) // 5)]
    train = paths[len(holdout) :]
    if args.lr_images is not None:
        make_pairs = lambda source, count, seed: paired_image_pairs(
            source, args.lr_images, args.patch, count, seed
        )
    else:
        make_pairs = lambda source, count, seed: image_pairs(
            source, args.patch, count, seed
        )
    if args.evaluate_only:
        if args.checkpoint is None:
            parser.error("--evaluate-only requires --checkpoint")
        test_pairs = make_pairs(paths, max(96, len(paths) * 12), args.seed + 1)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        baseline = evaluate(model, test_pairs, device)
        with torch.inference_mode():
            baseline_previews = [model(lr[None].to(device))[0].cpu() for lr, _ in test_pairs[:4]]
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        tuned = evaluate(model, test_pairs, device)
        args.output.mkdir(parents=True, exist_ok=True)
        save_comparison(args.output / "comparison.png", test_pairs, baseline_previews, model, device)
        report = {
            "source_package": str(args.package), "checkpoint": str(args.checkpoint),
            "lr_images": str(args.lr_images) if args.lr_images else None,
            "test_images": [str(path) for path in paths],
            "baseline": baseline, "tuned": tuned, "coreml_parity": parity,
        }
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"baseline: {json.dumps(baseline)}", flush=True)
        print(f"tuned: {json.dumps(tuned)}", flush=True)
        return
    test_pairs = make_pairs(holdout, max(48, len(holdout) * 12), args.seed + 1)
    train_pairs = make_pairs(train, max(args.steps * args.batch, 64), args.seed)
    if not train_pairs or not test_pairs:
        raise ValueError("the input images are too small for the selected patch")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    baseline = evaluate(model, test_pairs, device)
    with torch.inference_mode():
        baseline_previews = [
            model(lr[None].to(device))[0].cpu() for lr, _ in test_pairs[:4]
        ]
    print(f"baseline: {json.dumps(baseline)}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        chosen = train_pairs[step * args.batch : (step + 1) * args.batch]
        lr = torch.stack([pair[0] for pair in chosen]).to(device)
        hr = torch.stack([pair[1] for pair in chosen]).to(device)
        prediction = model(lr)
        prediction_hf = high_frequency(prediction)
        target_hf = high_frequency(hr)
        loss = F.l1_loss(prediction, hr) + args.hf_weight * F.l1_loss(prediction_hf, target_hf)
        if args.detail_weight:
            # Match local high-frequency energy in addition to its pixels.
            # This is a conservative texture objective, not a GAN hallucination.
            predicted_energy = prediction_hf.abs().mean(dim=(-2, -1))
            target_energy = target_hf.abs().mean(dim=(-2, -1))
            loss = loss + args.detail_weight * F.l1_loss(
                predicted_energy, target_energy
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0 or step == 0:
            print(f"step {step + 1}/{args.steps}: loss={loss.item():.6f}", flush=True)
    tuned = evaluate(model, test_pairs, device)
    print(f"tuned: {json.dumps(tuned)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    save_comparison(args.output / "comparison.png", test_pairs, baseline_previews, model, device)
    report = {
        "source_package": str(args.package),
        "lr_images": str(args.lr_images) if args.lr_images else None,
        "train_images": [str(path) for path in train],
        "holdout_images": [str(path) for path in holdout],
        "steps": args.steps,
        "loss": f"L1 + {args.hf_weight} * high-frequency L1 + {args.detail_weight} * frequency-energy L1",
        "baseline": baseline,
        "tuned": tuned,
        "coreml_parity": parity,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    torch.save(model.cpu().state_dict(), args.output / "pipersr-detail-experimental.pt")


if __name__ == "__main__":
    main()
