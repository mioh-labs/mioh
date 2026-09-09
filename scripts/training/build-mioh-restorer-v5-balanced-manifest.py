#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Build one source-balanced, native multi-bucket MiohRestorer V5 manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from json import JSONDecodeError
from pathlib import Path
import re
import time

from lada.models.mioh_restorer.native_manifest_v5 import (
    DEFAULT_BUCKET_RATIOS,
    StratifiedCandidateReservoir,
    candidates_for_metadata,
    select_balanced_source_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--source-cap", type=int, default=50)
    parser.add_argument("--context-fraction", type=float, default=0.30)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--minimum-bucket", type=int, default=192, choices=(128, 192, 256, 384, 512))
    parser.add_argument("--maximum-bucket", type=int, default=384, choices=(128, 192, 256, 384, 512))
    parser.add_argument("--reservoir-per-stratum", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--strip-sampled-timestamp",
        action="store_true",
        help="group lossless sample clips ending in -t<seconds>.mkv/mp4 as one source video",
    )
    parser.add_argument(
        "--canonicalize-fc2-source-id",
        action="store_true",
        help="group multipart FC2 filenames by their FC2-PPV numeric work ID",
    )
    parser.add_argument(
        "--exclude-source-id",
        action="append",
        default=[],
        help="canonical source ID to exclude; may be repeated to enforce split isolation",
    )
    return parser.parse_args()


def relative_or_absolute(path: Path, output: Path) -> str:
    try:
        return str(path.relative_to(output.parent))
    except ValueError:
        return str(path)


def metadata_source_id(
    path: Path,
    *,
    strip_sampled_timestamp: bool = False,
    canonicalize_fc2_source_id: bool = False,
) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    source_id = str(value["name"])
    if strip_sampled_timestamp:
        source_id = re.sub(r"-t[0-9]+\.(?:mkv|mp4)$", "", source_id, flags=re.IGNORECASE)
    if canonicalize_fc2_source_id:
        match = re.search(r"fc2[^0-9]{0,8}([0-9]{6,8})", source_id, flags=re.IGNORECASE)
        if match is not None:
            source_id = f"FC2PPV-{match.group(1)}"
    return source_id


def main() -> int:
    args = parse_args()
    for metadata_root in args.metadata_root:
        if not metadata_root.is_dir():
            raise FileNotFoundError(metadata_root)
    if args.minimum_bucket > args.maximum_bucket:
        raise ValueError("minimum-bucket must not exceed maximum-bucket")
    if args.source_cap <= 0 or args.reservoir_per_stratum <= 0:
        raise ValueError("source-cap and reservoir-per-stratum must be positive")
    metadata_paths = [
        path
        for metadata_root in args.metadata_root
        for path in sorted(metadata_root.glob("*.json"))
        if not path.name.startswith("._")
    ]
    by_source: dict[str, list[Path]] = defaultdict(list)
    excluded_source_ids = set(args.exclude_source_id)
    unreadable_metadata: list[str] = []
    for path in metadata_paths:
        try:
            source_id = metadata_source_id(
                path,
                strip_sampled_timestamp=args.strip_sampled_timestamp,
                canonicalize_fc2_source_id=args.canonicalize_fc2_source_id,
            )
            if source_id not in excluded_source_ids:
                by_source[source_id].append(path)
        except (OSError, JSONDecodeError, KeyError, TypeError, ValueError):
            unreadable_metadata.append(str(path))

    selected_all = []
    rejection_counts: Counter[str] = Counter()
    source_candidate_counts: dict[str, int] = {}
    source_selected_counts: dict[str, int] = {}
    failed_metadata: list[str] = []
    started = time.monotonic()
    for source_index, source_id in enumerate(sorted(by_source), start=1):
        reservoir = StratifiedCandidateReservoir(args.reservoir_per_stratum)
        for metadata_path in by_source[source_id]:
            try:
                for candidate in candidates_for_metadata(
                    metadata_path,
                    stride=args.stride,
                    context_fraction=args.context_fraction,
                    tile_overlap=args.tile_overlap,
                    minimum_bucket=args.minimum_bucket,
                    maximum_bucket=args.maximum_bucket,
                    seed=args.seed,
                    source_video_id=source_id,
                    rejection_counts=rejection_counts,
                ):
                    reservoir.add(candidate)
            except (OSError, JSONDecodeError, KeyError, TypeError, ValueError) as error:
                failed_metadata.append(f"{metadata_path}: {error}")
        selected = select_balanced_source_candidates(
            reservoir.candidates(),
            source_cap=args.source_cap,
            bucket_ratios={
                bucket: weight
                for bucket, weight in DEFAULT_BUCKET_RATIOS.items()
                if args.minimum_bucket <= bucket <= args.maximum_bucket
            },
        )
        selected_all.extend(selected)
        source_candidate_counts[source_id] = reservoir.seen
        source_selected_counts[source_id] = len(selected)
        if source_index == 1 or source_index % 5 == 0 or source_index == len(by_source):
            elapsed = time.monotonic() - started
            rate = source_index / max(elapsed, 1e-6)
            remaining = (len(by_source) - source_index) / max(rate, 1e-6)
            print(
                f"sources {source_index}/{len(by_source)} | candidates {sum(source_candidate_counts.values())} | "
                f"selected {len(selected_all)} | ETA {remaining / 60:.1f} min",
                flush=True,
            )

    if not selected_all:
        raise RuntimeError("no balanced native V5 windows were selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fingerprint = hashlib.sha256()
    bucket_counts: Counter[int] = Counter()
    motion_counts: Counter[str] = Counter()
    occupancy_counts: Counter[str] = Counter()
    seen_names: set[str] = set()
    with temporary.open("w", encoding="utf-8") as destination:
        for candidate in selected_all:
            entry = dict(candidate.entry)
            name = str(entry["name"])
            if name in seen_names:
                raise RuntimeError(f"duplicate selected window: {name}")
            seen_names.add(name)
            entry["target_video"] = relative_or_absolute(Path(entry["target_video"]), args.output)
            entry["mask_video"] = relative_or_absolute(Path(entry["mask_video"]), args.output)
            encoded = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            destination.write(encoded + "\n")
            fingerprint.update(encoded.encode("utf-8"))
            fingerprint.update(b"\n")
            bucket_counts[int(entry["bucket"])] += 1
            motion_counts[str(entry["motion_bucket"])] += 1
            occupancy_counts[str(entry["occupancy_bucket"])] += 1
    temporary.replace(args.output)

    report = {
        "schema": "mioh-restorer-v5-balanced-native-v1",
        "output": str(args.output),
        "sha256": fingerprint.hexdigest(),
        "windows": len(selected_all),
        "source_videos": len(source_selected_counts),
        "source_cap": args.source_cap,
        "source_selected_min": min(source_selected_counts.values()),
        "source_selected_max": max(source_selected_counts.values()),
        "buckets": dict(sorted(bucket_counts.items())),
        "motion": dict(sorted(motion_counts.items())),
        "occupancy": dict(sorted(occupancy_counts.items())),
        "candidate_windows": sum(source_candidate_counts.values()),
        "rejections": dict(sorted(rejection_counts.items())),
        "metadata_files": len(metadata_paths),
        "metadata_roots": [str(path) for path in args.metadata_root],
        "excluded_source_ids": sorted(excluded_source_ids),
        "unreadable_metadata": unreadable_metadata,
        "failed_metadata": failed_metadata,
        "resized_rgb_frames": 0,
        "source_selected_counts": source_selected_counts,
        "seed": args.seed,
        "stride": args.stride,
        "minimum_bucket": args.minimum_bucket,
        "maximum_bucket": args.maximum_bucket,
        "bucket_ratios": DEFAULT_BUCKET_RATIOS,
    }
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
