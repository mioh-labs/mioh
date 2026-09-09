#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Build a non-destructive BasicVSR++ curriculum view with sources excluded.

The source datasets and manifests are never edited.  A new directory tree is
made from symlinks and filtered JSONL files so every training stage can share
one auditable denylist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDED_SOURCES = (
    "hhd800.com@FC2-PPV-3031139.mp4",
    "hhd800.com@FC2-PPV-3118963.mp4",
    "ZSD-074 Yu Shinoda Uncensored Leaked-1080p.mp4",
)

DEFAULT_EXCLUDED_ASSET_NAMES = (
    "fc2-1965332-b--progressive-2997.mkv-000009-.mp4",
    "fc2-1965332-b--progressive-2997.mkv-000009-.mkv",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(
        "/Volumes/Project_HD/lada_finetune_aozora_hikari"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--additional-train-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Additional representative train view to append; repeatable. "
            "Defaults to the progressive-clean fc2_best view."
        ),
    )
    parser.add_argument(
        "--representative-root",
        type=Path,
        default=project_root / "dataset_representative",
    )
    parser.add_argument(
        "--known-grid-root",
        type=Path,
        default=project_root / "known_grid_26_v1" / "manifests",
    )
    parser.add_argument(
        "--native-hf-manifest",
        type=Path,
        default=(
            project_root
            / "mioh_native_hf/manifests/train-native-hf-512-recoverable-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--aozora-hf-manifest",
        type=Path,
        default=(
            project_root
            / "aozora_hf_4source/manifests/train-old512-plus-aozora-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--fc2-hf-manifest",
        type=Path,
        default=(
            project_root
            / "fc2_best_hf_v1/manifests/train-old560-plus-fc2-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--validation-hf-manifest",
        type=Path,
        default=(
            project_root
            / "mioh_native_hf/manifests/validation-native-hf-512-recoverable-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "basicvsrpp_clean_no_interlace_fc2_v3",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Exact metadata name/source_video_id to exclude (repeatable)",
    )
    parser.add_argument(
        "--exclude-asset-name",
        action="append",
        default=[],
        help="Exact video/mask basename to exclude from JSONL manifests",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source:
            return
        raise FileExistsError(destination)
    destination.symlink_to(source)


def _destination_for_relative(
    metadata_dir: Path,
    split_root: Path,
    relative_value: str,
) -> Path:
    destination = Path(
        os.path.normpath(str(metadata_dir / Path(relative_value)))
    )
    try:
        destination.relative_to(split_root)
    except ValueError as error:
        raise ValueError(
            f"metadata relative path escapes split root: {relative_value}"
        ) from error
    return destination


def build_representative_view(
    source_root: Path,
    output_root: Path,
    excluded: set[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    path_fields = (
        "relative_nsfw_video_path",
        "relative_mask_video_path",
        "relative_mosaic_nsfw_video_path",
        "relative_mosaic_mask_video_path",
    )
    for split in ("train", "validation", "test"):
        source_metadata_dir = source_root / split / "crop_unscaled_meta"
        if not source_metadata_dir.is_dir():
            raise FileNotFoundError(source_metadata_dir)
        destination_split = output_root / split
        destination_metadata_dir = destination_split / "crop_unscaled_meta"
        destination_metadata_dir.mkdir(parents=True, exist_ok=False)
        kept = 0
        removed = 0
        removed_by_source: dict[str, int] = {}
        for metadata_path in sorted(source_metadata_dir.glob("*.json")):
            if metadata_path.name.startswith("._"):
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_id = str(metadata.get("name", ""))
            if source_id in excluded:
                removed += 1
                removed_by_source[source_id] = (
                    removed_by_source.get(source_id, 0) + 1
                )
                continue
            _link(metadata_path, destination_metadata_dir / metadata_path.name)
            for field in path_fields:
                relative_value = metadata.get(field)
                if not relative_value:
                    continue
                source_asset = metadata_path.parent / relative_value
                destination_asset = _destination_for_relative(
                    destination_metadata_dir,
                    destination_split,
                    relative_value,
                )
                _link(source_asset, destination_asset)
            kept += 1
        report[split] = {
            "kept": kept,
            "removed": removed,
            "removed_by_source": removed_by_source,
        }
    return report


def append_representative_train_view(
    source_train_root: Path,
    destination_train_root: Path,
) -> dict[str, Any]:
    metadata_dir = source_train_root / "crop_unscaled_meta"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(metadata_dir)
    destination_metadata_dir = destination_train_root / "crop_unscaled_meta"
    path_fields = (
        "relative_nsfw_video_path",
        "relative_mask_video_path",
        "relative_mosaic_nsfw_video_path",
        "relative_mosaic_mask_video_path",
    )
    kept = 0
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        if metadata_path.name.startswith("._"):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _link(metadata_path, destination_metadata_dir / metadata_path.name)
        for field in path_fields:
            relative_value = metadata.get(field)
            if not relative_value:
                continue
            source_asset = metadata_path.parent / relative_value
            destination_asset = _destination_for_relative(
                destination_metadata_dir,
                destination_train_root,
                relative_value,
            )
            _link(source_asset, destination_asset)
        kept += 1
    return {"source": str(source_train_root), "kept": kept}


def filter_jsonl(
    source: Path,
    destination: Path,
    excluded: set[str],
    source_key: str,
    excluded_asset_names: set[str],
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept: list[dict[str, Any]] = []
    removed_by_source: dict[str, int] = {}
    removed_by_asset: dict[str, int] = {}
    for row in rows:
        source_id = str(row.get(source_key, ""))
        if source_id in excluded:
            removed_by_source[source_id] = (
                removed_by_source.get(source_id, 0) + 1
            )
            continue
        row_assets = {
            Path(value).name
            for key, value in row.items()
            if key.endswith("video") and isinstance(value, str) and value
        }
        matched_assets = row_assets & excluded_asset_names
        if matched_assets:
            for asset in matched_assets:
                removed_by_asset[asset] = removed_by_asset.get(asset, 0) + 1
            continue
        kept.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(destination)
    return {
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output": str(destination.resolve()),
        "output_sha256": _sha256(destination),
        "original": len(rows),
        "kept": len(kept),
        "removed": len(rows) - len(kept),
        "removed_by_source": removed_by_source,
        "removed_by_asset": removed_by_asset,
    }


def main() -> None:
    args = parse_args()
    project_root = Path("/Volumes/Project_HD/lada_finetune_aozora_hikari")
    additional_train_roots = args.additional_train_root or [
        project_root
        / "fc2_best_hf_v1/curated_dataset_progressive_clean_v1/train",
        project_root / "basicvsrpp_new_sources_5pct_dataset_v1",
    ]
    excluded = set(DEFAULT_EXCLUDED_SOURCES) | set(args.exclude_source)
    excluded_asset_names = set(DEFAULT_EXCLUDED_ASSET_NAMES) | set(
        args.exclude_asset_name
    )
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; refusing to overwrite: {output_root}"
        )
    output_root.mkdir(parents=True)

    report: dict[str, Any] = {
        "version": 1,
        "excluded_sources": sorted(excluded),
        "excluded_asset_names": sorted(excluded_asset_names),
        "representative": build_representative_view(
            args.representative_root.expanduser().resolve(),
            output_root / "dataset_representative",
            excluded,
        ),
        "manifests": {},
    }
    report["additional_representative_train"] = [
        append_representative_train_view(
            source_root.expanduser().resolve(strict=True),
            output_root / "dataset_representative/train",
        )
        for source_root in additional_train_roots
    ]
    manifest_root = output_root / "manifests"
    manifest_specs = (
        (
            "known_grid_train",
            args.known_grid_root / "train-known-grid-26-v1.jsonl",
            "train-known-grid-26-clean-v1.jsonl",
            "name",
        ),
        (
            "known_grid_validation",
            args.known_grid_root / "validation-known-grid-26-v1.jsonl",
            "validation-known-grid-26-clean-v1.jsonl",
            "name",
        ),
        (
            "known_grid_test",
            args.known_grid_root / "test-known-grid-26-v1.jsonl",
            "test-known-grid-26-clean-v1.jsonl",
            "name",
        ),
        (
            "native_hf_train",
            args.native_hf_manifest,
            "train-native-hf-clean-v1.jsonl",
            "source_video_id",
        ),
        (
            "aozora_hf_train",
            args.aozora_hf_manifest,
            "train-aozora-hf-clean-v1.jsonl",
            "source_video_id",
        ),
        (
            "fc2_hf_train",
            args.fc2_hf_manifest,
            "train-fc2-hf-clean-v1.jsonl",
            "source_video_id",
        ),
        (
            "hf_validation",
            args.validation_hf_manifest,
            "validation-native-hf-clean-v1.jsonl",
            "source_video_id",
        ),
    )
    for key, source, filename, source_key in manifest_specs:
        report["manifests"][key] = filter_jsonl(
            source.expanduser().resolve(),
            manifest_root / filename,
            excluded,
            source_key,
            excluded_asset_names,
        )

    _atomic_json(output_root / "dataset-report.json", report)
    print(f"Prepared clean curriculum: {output_root}")
    for split, values in report["representative"].items():
        print(
            f"{split}: {values['kept']} kept, {values['removed']} removed"
        )
    for key, values in report["manifests"].items():
        print(f"{key}: {values['kept']} kept, {values['removed']} removed")


if __name__ == "__main__":
    main()
