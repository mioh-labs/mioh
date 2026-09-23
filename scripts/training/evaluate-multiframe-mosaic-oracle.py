#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Probe recoverability of a fixed-grid mosaic with oracle-known motion.

One clean image is translated by *known integer shifts* before each frame is
block-averaged on a stationary image grid. The inverse solver gets the exact
shifts and block size, but never the clean pixels. This isolates measurement
diversity from the much harder problem of estimating motion through a mosaic.

This is a controlled static-scene experiment, not a mathematical upper bound
or a prediction for real video. The finite-iteration solver may stop before
convergence; its result is only one achievable point with this prior.
Occlusion, deformation, compression, phase errors and optical-flow errors are
absent here but present in real video. Results are compared with the same
solver using one frame and with the single-frame block mosaic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize


def block_mean(image: np.ndarray, block: int) -> np.ndarray:
    """One scalar per block, equivalent to the phase-zero training mosaic."""
    height, width = image.shape
    if height % block or width % block:
        raise ValueError("image dimensions must be multiples of block size")
    return image.reshape(height // block, block, width // block, block).mean(
        axis=(1, 3)
    )


def expanded_measurement(measurement: np.ndarray, block: int) -> np.ndarray:
    return np.repeat(np.repeat(measurement, block, axis=0), block, axis=1)


def shifts_for(block: int, count: int, seed: int, motion: str) -> list[tuple[int, int]]:
    if motion == "static":
        return [(0, 0)] * count
    rng = np.random.default_rng(seed + block)
    phases = [(0, 0)] + [
        (int(index % block), int(index // block))
        for index in rng.permutation(block * block)
        if index != 0
    ]
    # With more frames than phases, repeat only after all phases were used.
    return [phases[index % len(phases)] for index in range(count)]


def observe(clean: np.ndarray, block: int, shifts: list[tuple[int, int]]) -> list[np.ndarray]:
    return [
        block_mean(np.roll(clean, (dy, dx), axis=(0, 1)), block)
        for dy, dx in shifts
    ]


def objective_and_gradient(
    flat: np.ndarray,
    shape: tuple[int, int],
    block: int,
    shifts: list[tuple[int, int]],
    measurements: list[np.ndarray],
    tv_weight: float,
) -> tuple[float, np.ndarray]:
    candidate = flat.reshape(shape)
    height, width = shape
    count = len(shifts)
    gradient = np.zeros(shape, dtype=np.float64)
    data_loss = 0.0
    for (dy, dx), observation in zip(shifts, measurements, strict=True):
        residual = block_mean(np.roll(candidate, (dy, dx), axis=(0, 1)), block) - observation
        data_loss += 0.5 * float(np.mean(residual * residual)) / count
        # Adjoint of block_mean, with the mean-square normalization above.
        residual_image = expanded_measurement(residual, block) / (height * width * count)
        gradient += np.roll(residual_image, (-dy, -dx), axis=(0, 1))

    # Smooth isotropic TV, with periodic differences to match np.roll motion.
    dx = np.roll(candidate, -1, axis=1) - candidate
    dy = np.roll(candidate, -1, axis=0) - candidate
    norm = np.sqrt(dx * dx + dy * dy + 1e-6)
    tv_loss = tv_weight * float(np.mean(norm))
    flux_x = dx / norm
    flux_y = dy / norm
    gradient += tv_weight * (
        np.roll(flux_x, 1, axis=1) - flux_x
        + np.roll(flux_y, 1, axis=0) - flux_y
    ) / (height * width)
    # L-BFGS-B's absolute gradient tolerance would otherwise stop at the
    # initial mosaic on a 256px image despite a large normalized residual.
    scale = height * width
    return (data_loss + tv_loss) * scale, gradient.ravel() * scale


def reconstruct(
    shape: tuple[int, int],
    block: int,
    shifts: list[tuple[int, int]],
    measurements: list[np.ndarray],
    *,
    tv_weight: float,
    iterations: int,
) -> tuple[np.ndarray, bool, int]:
    start = expanded_measurement(measurements[0], block).astype(np.float64)
    result = minimize(
        objective_and_gradient,
        start.ravel(),
        args=(shape, block, shifts, measurements, tv_weight),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * start.size,
        options={"maxiter": iterations, "ftol": 1e-12, "gtol": 1e-8},
    )
    return result.x.reshape(shape), bool(result.success), int(result.nit)


def quality(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    error = float(np.mean((candidate - reference) ** 2))
    reference_hf = reference - cv2.GaussianBlur(reference, (0, 0), 1.2)
    candidate_hf = candidate - cv2.GaussianBlur(candidate, (0, 0), 1.2)
    ref_centered = reference_hf - reference_hf.mean()
    out_centered = candidate_hf - candidate_hf.mean()
    ref_rms = float(np.sqrt(np.mean(ref_centered**2)))
    out_rms = float(np.sqrt(np.mean(out_centered**2)))
    correlation = float(
        np.mean(ref_centered * out_centered) / max(ref_rms * out_rms, 1e-12)
    )
    amplitude = out_rms / max(ref_rms, 1e-12)
    return {
        "psnr_db": -10 * math.log10(max(error, 1e-12)),
        "hf_correlation": correlation,
        "hf_amplitude_ratio": amplitude,
        "hf_corr_times_amp": correlation * amplitude,
    }


def load_image(
    path: Path | None,
    video: Path | None,
    time: float,
    size: int,
    left: int | None,
    top: int | None,
) -> np.ndarray:
    if video is not None:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video}")
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
            okay, frame = capture.read()
        finally:
            capture.release()
        if not okay:
            raise ValueError(f"cannot decode frame at {time:.3f}s: {video}")
        source = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif path is None:
        from skimage.data import camera

        source = camera()
    else:
        source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise ValueError(f"cannot read image: {path}")
    height, width = source.shape
    if min(height, width) < size:
        raise ValueError(f"source must be at least {size}x{size}")
    top = (height - size) // 2 if top is None else top
    left = (width - size) // 2 if left is None else left
    if not 0 <= top <= height - size or not 0 <= left <= width - size:
        raise ValueError("requested crop lies outside the source frame")
    return source[top : top + size, left : left + size].astype(np.float64) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="clean image; default: skimage camera")
    source.add_argument("--video", type=Path, help="use one unmasked video crop as the clean image")
    parser.add_argument("--time", type=float, default=0.0, help="video frame time in seconds")
    parser.add_argument("--crop-left", type=int, help="left edge of the clean crop")
    parser.add_argument("--crop-top", type=int, help="top edge of the clean crop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, help="optional PNGs of clean, mosaic and reconstructions")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--blocks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 3, 9, 26])
    parser.add_argument("--motion", choices=["diverse", "static"], default="diverse")
    parser.add_argument("--tv-weight", type=float, default=0.003)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.size <= 0 or any(b <= 1 or args.size % b for b in args.blocks):
        parser.error("size must be positive and divisible by each block size > 1")
    if any(n <= 0 for n in args.frames) or args.tv_weight < 0 or args.iterations <= 0:
        parser.error("frames and iterations must be positive; TV weight must be nonnegative")
    if args.time < 0:
        parser.error("video time must be nonnegative")
    clean = load_image(
        args.image, args.video, args.time, args.size,
        args.crop_left, args.crop_top,
    )
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview_dir / "clean.png"), np.rint(clean * 255).astype(np.uint8))
    rows = []
    for block in args.blocks:
        for count in args.frames:
            shifts = shifts_for(block, count, args.seed, args.motion)
            measurements = observe(clean, block, shifts)
            recovered, converged, iterations = reconstruct(
                clean.shape, block, shifts, measurements,
                tv_weight=args.tv_weight, iterations=args.iterations,
            )
            mosaic = expanded_measurement(measurements[0], block)
            if args.preview_dir is not None:
                if count == args.frames[0]:
                    cv2.imwrite(
                        str(args.preview_dir / f"block-{block}-mosaic.png"),
                        np.rint(mosaic * 255).astype(np.uint8),
                    )
                cv2.imwrite(
                    str(args.preview_dir / f"block-{block}-frames-{count}.png"),
                    np.rint(np.clip(recovered, 0, 1) * 255).astype(np.uint8),
                )
            row = {
                "block": block,
                "frames": count,
                "distinct_phases": len(set(shifts)),
                "solver_converged": converged,
                "solver_iterations": iterations,
                "mosaic": quality(clean, mosaic),
                "reconstructed": quality(clean, recovered),
            }
            rows.append(row)
            print(
                f"block={block:2d} frames={count:2d} phases={row['distinct_phases']:2d} "
                f"PSNR={row['reconstructed']['psnr_db']:.2f} dB "
                f"HF={row['reconstructed']['hf_corr_times_amp']:.3f} "
                f"iter={iterations} converged={converged}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "design": "static scene, exact integer shifts, fixed phase-zero block average",
                "image": str(args.image) if args.image else None,
                "video": str(args.video) if args.video else None,
                "video_time": args.time if args.video else None,
                "crop_left": args.crop_left,
                "crop_top": args.crop_top,
                "size": args.size,
                "motion": args.motion,
                "tv_weight": args.tv_weight,
                "maximum_iterations": args.iterations,
                "seed": args.seed,
                "results": rows,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
