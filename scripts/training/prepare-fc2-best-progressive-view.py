#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Prepare a non-destructive progressive view of the curated FC2-best clips.

Most curated clips are linked without modification.  Clips identified by a
pixel-level interlace audit are deinterlaced together with their masks so the
clean rebuild can retain the full FC2-best curriculum without learning combs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_REPROCESS = (
    "fc2-1965332-a--progressive-2997.mkv-000001-",
    "fc2-1965332-b--progressive-2997.mkv-000009-",
)


def parse_args() -> argparse.Namespace:
    project_root = Path("/Volumes/Project_HD/lada_finetune_aozora_hikari")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "fc2_best_hf_v1/curated_dataset/train",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            project_root
            / "fc2_best_hf_v1/curated_dataset_progressive_clean_v1/train"
        ),
    )
    parser.add_argument("--reprocess", action="append", default=[])
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(strict=True))


def _probe_frame_count(path: Path) -> int:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
    ).strip()
    return int(output)


def _deinterlace_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            "bwdif=mode=send_frame:parity=auto:deint=all,setfield=prog",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def _deinterlace_mask(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            (
                "bwdif=mode=send_frame:parity=auto:deint=all,"
                "format=gray,lut=y='if(gte(val,128),255,0)',setfield=prog"
            ),
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            str(destination),
        ]
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; refusing to overwrite: {output_root}"
        )
    output_root.mkdir(parents=True)
    reprocess = set(args.reprocess or DEFAULT_REPROCESS)
    report: dict[str, Any] = {
        "version": 1,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "requested_reprocess": sorted(reprocess),
        "clips": [],
    }

    metadata_root = source_root / "crop_unscaled_meta"
    for metadata_path in sorted(metadata_root.glob("*.json")):
        if metadata_path.name.startswith("._"):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stem = metadata_path.stem
        output_metadata = output_root / "crop_unscaled_meta" / metadata_path.name
        output_metadata.parent.mkdir(parents=True, exist_ok=True)
        output_metadata.symlink_to(metadata_path.resolve(strict=True))

        input_video = (metadata_path.parent / metadata["relative_nsfw_video_path"]).resolve(strict=True)
        input_mask = (metadata_path.parent / metadata["relative_mask_video_path"]).resolve(strict=True)
        output_video = output_root / "crop_unscaled_img" / input_video.name
        output_mask = output_root / "crop_unscaled_mask" / input_mask.name
        transformed = stem in reprocess
        if transformed:
            _deinterlace_video(input_video, output_video)
            _deinterlace_mask(input_mask, output_mask)
        else:
            _link(input_video, output_video)
            _link(input_mask, output_mask)

        input_frames = _probe_frame_count(input_video)
        output_frames = _probe_frame_count(output_video)
        mask_frames = _probe_frame_count(output_mask)
        expected_frames = int(metadata["frames_count"])
        if not (
            input_frames == output_frames == mask_frames == expected_frames
        ):
            raise RuntimeError(
                f"frame mismatch for {stem}: metadata={expected_frames}, "
                f"input={input_frames}, output={output_frames}, mask={mask_frames}"
            )
        report["clips"].append(
            {
                "metadata": metadata_path.name,
                "frames": output_frames,
                "transformed": transformed,
                "video": str(output_video.resolve()),
                "mask": str(output_mask.resolve()),
            }
        )

    found = {Path(row["metadata"]).stem for row in report["clips"] if row["transformed"]}
    missing = reprocess - found
    if missing:
        raise RuntimeError(f"requested clips were not found: {sorted(missing)}")
    report["clip_count"] = len(report["clips"])
    report["transformed_count"] = sum(row["transformed"] for row in report["clips"])
    (output_root.parent / "progressive-view-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {report['clip_count']} FC2-best clips; "
        f"deinterlaced {report['transformed_count']}: {output_root}"
    )


if __name__ == "__main__":
    main()
