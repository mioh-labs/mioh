#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Audit extracted clips and build a source-disjoint sharpness dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import statistics
from pathlib import Path

import cv2
import numpy as np


PRODUCT_PATTERN = re.compile(r'FC2[- _]?P?PV?[- _]?(\d+)', re.IGNORECASE)
SUBDIRECTORIES = ('crop_unscaled_img', 'crop_unscaled_mask', 'crop_unscaled_meta')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--validation-product', default='4054883')
    parser.add_argument('--maximum-per-product', type=int, default=220)
    parser.add_argument('--maximum-per-source', type=int, default=100)
    parser.add_argument('--minimum-mask-bbox', type=int, default=192)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=430)
    return parser.parse_args()


def read_frame(path: Path, index: int):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def product_id(name: str) -> str:
    match = PRODUCT_PATTERN.search(name)
    return match.group(1) if match else name


def resolve_metadata_path(meta_path: Path, relative_path: str) -> Path:
    return (meta_path.parent / relative_path).resolve()


def inspect_clip(meta_path: Path):
    try:
        metadata = json.loads(meta_path.read_text())
        video_path = resolve_metadata_path(
            meta_path, metadata['relative_nsfw_video_path']
        )
        mask_path = resolve_metadata_path(
            meta_path, metadata['relative_mask_video_path']
        )
        laplacian_scores = []
        gradient_scores = []
        mask_areas = []
        brightness_scores = []
        bounding_sizes = []
        for fraction in (0.25, 0.5, 0.75):
            frame_index = min(
                metadata['frames_count'] - 1,
                int(metadata['frames_count'] * fraction),
            )
            frame = read_frame(video_path, frame_index)
            mask_frame = read_frame(mask_path, frame_index)
            if frame is None or mask_frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) > 127
            if mask.sum() < 64:
                continue
            eroded = cv2.erode(
                mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
            ).astype(bool)
            if eroded.sum() < 64:
                eroded = mask

            laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            ys, xs = np.where(mask)
            laplacian_scores.append(float(np.var(laplacian[eroded])))
            gradient_scores.append(
                float(np.mean(gradient_x[eroded] ** 2 + gradient_y[eroded] ** 2))
            )
            mask_areas.append(float(mask.mean()))
            brightness_scores.append(float(gray[mask].mean()))
            bounding_sizes.append(
                int(min(np.ptp(ys) + 1, np.ptp(xs) + 1))
            )

        if not laplacian_scores:
            raise RuntimeError('could not decode a usable masked frame')

        name = metadata['name']
        laplacian = statistics.median(laplacian_scores)
        gradient = statistics.median(gradient_scores)
        # Log scaling prevents a few extremely crisp edges dominating ranking.
        quality_score = math.log1p(laplacian) + 0.5 * math.log1p(gradient)
        return {
            'meta_path': str(meta_path.resolve()),
            'video_path': str(video_path),
            'mask_path': str(mask_path),
            'meta_name': meta_path.name,
            'video_name': video_path.name,
            'mask_name': mask_path.name,
            'source_name': name,
            'product_id': product_id(name),
            'laplacian_variance': laplacian,
            'gradient_energy': gradient,
            'mask_area': statistics.median(mask_areas),
            'brightness': statistics.median(brightness_scores),
            'mask_bbox_minimum': statistics.median(bounding_sizes),
            'quality_score': quality_score,
            'error': None,
        }
    except Exception as error:
        return {
            'meta_path': str(meta_path.resolve()),
            'error': str(error),
        }


def stratified_quality_selection(rows, count, seed):
    if len(rows) <= count:
        return list(rows)
    rows = sorted(rows, key=lambda row: row['quality_score'], reverse=True)
    top_end = max(1, int(len(rows) * 0.55))
    middle_end = max(top_end + 1, int(len(rows) * 0.85))
    pools = (rows[:top_end], rows[top_end:middle_end], rows[middle_end:])
    targets = (round(count * 0.70), round(count * 0.20))
    targets = (targets[0], targets[1], count - sum(targets))
    rng = random.Random(seed)
    selected = []
    for pool, target in zip(pools, targets):
        selected.extend(rng.sample(pool, min(target, len(pool))))

    if len(selected) < count:
        selected_paths = {row['meta_path'] for row in selected}
        remainder = [row for row in rows if row['meta_path'] not in selected_paths]
        selected.extend(remainder[:count - len(selected)])
    return selected[:count]


def link_clip(row, split_root: Path):
    paths = {
        'crop_unscaled_meta': Path(row['meta_path']),
        'crop_unscaled_img': Path(row['video_path']),
        'crop_unscaled_mask': Path(row['mask_path']),
    }
    names = {
        'crop_unscaled_meta': row['meta_name'],
        'crop_unscaled_img': row['video_name'],
        'crop_unscaled_mask': row['mask_name'],
    }
    for subdirectory in SUBDIRECTORIES:
        target = split_root / subdirectory / names[subdirectory]
        source = paths[subdirectory]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'refusing to replace {target}')
        target.symlink_to(source)


def summarize(rows):
    products = {}
    for row in rows:
        products[row['product_id']] = products.get(row['product_id'], 0) + 1
    return {
        'clips': len(rows),
        'products': products,
        'median_laplacian_variance': statistics.median(
            row['laplacian_variance'] for row in rows
        ) if rows else None,
        'median_gradient_energy': statistics.median(
            row['gradient_energy'] for row in rows
        ) if rows else None,
    }


def main():
    args = parse_args()
    metadata_paths = sorted(
        (args.dataset_root / 'crop_unscaled_meta').glob('*.json')
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        rows = list(executor.map(inspect_clip, metadata_paths))

    errors = [row for row in rows if row['error']]
    usable = [
        row for row in rows
        if not row['error']
        and row['mask_bbox_minimum'] >= args.minimum_mask_bbox
        and 20.0 <= row['brightness'] <= 235.0
    ]
    validation = [
        row for row in usable
        if row['product_id'] == args.validation_product
    ]
    train_candidates = [
        row for row in usable
        if row['product_id'] != args.validation_product
    ]

    by_source = {}
    for row in train_candidates:
        by_source.setdefault(row['source_name'], []).append(row)
    source_limited = []
    for index, source_rows in enumerate(sorted(by_source.values(), key=len)):
        source_limited.extend(stratified_quality_selection(
            source_rows,
            args.maximum_per_source,
            args.seed + index,
        ))

    by_product = {}
    for row in source_limited:
        by_product.setdefault(row['product_id'], []).append(row)
    train = []
    for index, product_rows in enumerate(sorted(by_product.values(), key=len)):
        train.extend(stratified_quality_selection(
            product_rows,
            args.maximum_per_product,
            args.seed + 1000 + index,
        ))

    if not validation:
        raise RuntimeError(
            f'no validation clips found for product {args.validation_product}'
        )

    for row in train:
        link_clip(row, args.output_root / 'train')
    for row in validation:
        link_clip(row, args.output_root / 'validation')

    report = {
        'configuration': {
            'dataset_root': str(args.dataset_root),
            'validation_product': args.validation_product,
            'maximum_per_product': args.maximum_per_product,
            'maximum_per_source': args.maximum_per_source,
            'minimum_mask_bbox': args.minimum_mask_bbox,
            'seed': args.seed,
        },
        'audited': len(rows),
        'errors': errors,
        'usable': len(usable),
        'train': summarize(train),
        'validation': summarize(validation),
        'clips': rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / 'quality-report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({
        'report': str(report_path),
        'audited': len(rows),
        'errors': len(errors),
        'usable': len(usable),
        'train': summarize(train),
        'validation': summarize(validation),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
