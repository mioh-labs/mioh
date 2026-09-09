#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Freeze representative source membership for exact known-grid training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lada.datasetcreation.restoration_dataset_metadata import (
    RestorationDatasetMetadataV2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-root',
        type=Path,
        default=Path(
            '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
            'dataset_representative'
        ),
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path(
            '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
            'known_grid_26_v1/manifests'
        ),
    )
    parser.add_argument('--num-frames', type=int, default=26)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def build_split(
    source_root: Path,
    output_root: Path,
    split: str,
    num_frames: int,
    overwrite: bool,
) -> tuple[Path, int]:
    metadata_root = source_root / split / 'crop_unscaled_meta'
    if not metadata_root.is_dir():
        raise FileNotFoundError(metadata_root)
    output_path = output_root / f'{split}-known-grid-26-v1.jsonl'
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f'{output_path} already exists; pass --overwrite to replace it'
        )

    entries: list[dict] = []
    for metadata_path in sorted(metadata_root.glob('*.json')):
        if metadata_path.name.startswith('._'):
            continue
        metadata = RestorationDatasetMetadataV2.from_json_file(metadata_path)
        if metadata.frames_count < num_frames:
            continue
        target_path = (
            metadata_path.parent / metadata.relative_nsfw_video_path
        ).resolve()
        mask_path = (
            metadata_path.parent / metadata.relative_mask_video_path
        ).resolve()
        if not target_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f'missing source for {metadata_path}: {target_path}, {mask_path}'
            )
        entries.append(
            {
                'version': 1,
                'split': split,
                'name': metadata.name,
                'metadata_path': str(metadata_path.resolve()),
                'target_video_path': str(target_path),
                'mask_video_path': str(mask_path),
                'frames_count': int(metadata.frames_count),
                'fps': float(metadata.fps),
            }
        )
    if not entries:
        raise RuntimeError(f'no eligible metadata found in {metadata_root}')

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write('\n')
    temporary_path.replace(output_path)
    return output_path, len(entries)


def main() -> None:
    args = parse_args()
    if args.num_frames < 2:
        raise SystemExit('--num-frames must be at least 2')
    for split in ('train', 'validation', 'test'):
        path, count = build_split(
            args.source_root,
            args.output_root,
            split,
            args.num_frames,
            args.overwrite,
        )
        print(f'{split}: {count} samples -> {path}')


if __name__ == '__main__':
    main()
