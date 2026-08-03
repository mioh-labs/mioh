# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import json
import runpy
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest


MODULE = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "training"
        / "preprocess-interlaced-hf-clips.py"
    )
)


def test_parse_and_aggregate_idet_uses_final_summary() -> None:
    log = """
    Multi frame detection: TFF: 0 BFF: 0 Progressive: 0 Undetermined: 0
    Multi frame detection: TFF: 299 BFF: 1 Progressive: 0 Undetermined: 1
    """
    counts = MODULE["parse_idet"](log)
    assert counts == {"tff": 299, "bff": 1, "progressive": 0, "undetermined": 1}
    aggregate = MODULE["aggregate_idet"]([{"counts": counts}])
    assert aggregate["classification"] == "interlaced"
    assert aggregate["parity"] == "tff"


def test_deinterlace_command_preserves_frame_rate_contract(tmp_path: Path) -> None:
    command = MODULE["build_deinterlace_command"](
        "/usr/bin/ffmpeg",
        source=tmp_path / "source.mp4",
        output=tmp_path / "output.mkv",
        start_frame=30,
        frame_count=240,
        parity="tff",
    )
    filter_graph = command[command.index("-vf") + 1]
    assert "bwdif=mode=send_frame:parity=tff:deint=all" in filter_graph
    assert "setpts=N*1001/(30000*TB)" in filter_graph
    assert "setparams=range=limited" in filter_graph
    assert command[command.index("-frames:v") + 1] == "240"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-c:v") + 1] == "ffv1"
    assert command[command.index("-color_range") + 1] == "tv"
    assert command[command.index("-colorspace") + 1] == "bt709"


@pytest.mark.parametrize(
    "rate",
    ("30000/1001", "30/1", "60000/1001", "60/1"),
)
def test_progressive_command_accepts_supported_rates_without_spatial_resize(
    tmp_path: Path, rate: str
) -> None:
    command = MODULE["build_progressive_command"](
        "/usr/bin/ffmpeg",
        source=tmp_path / "source.mp4",
        output=tmp_path / "output.mkv",
        start_seconds=Fraction(7, 3),
        output_frame_count=240,
        source_rate=Fraction(rate),
    )
    filter_graph = command[command.index("-vf") + 1]
    assert "fps=fps=30000/1001:start_time=0:round=near:eof_action=pass" in filter_graph
    assert "setfield=prog" in filter_graph
    assert "scale" not in filter_graph
    assert "crop" not in filter_graph
    assert command[command.index("-frames:v") + 1] == "240"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-c:v") + 1] == "ffv1"


def test_progressive_scene_contract_rejects_unsupported_or_ambiguous_rate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    scan = {
        "probe": {
            "avg_frame_rate": "25/1",
            "duration_seconds": 10.0,
        }
    }
    with pytest.raises(ValueError, match="unsupported progressive source frame rate"):
        MODULE["_progressive_scene_values"](
            {
                "source": str(source),
                "start_frame": 0,
                "output_frame_count": 30,
            },
            root=tmp_path,
            scan=scan,
        )

    scan["probe"]["avg_frame_rate"] = "60/1"
    with pytest.raises(ValueError, match="both start_frame and start_seconds"):
        MODULE["_progressive_scene_values"](
            {
                "source": str(source),
                "start_frame": 0,
                "start_seconds": 0,
                "output_frame_count": 30,
            },
            root=tmp_path,
            scan=scan,
        )


def test_progressive_scan_classification_is_mandatory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not classified as progressive"):
        MODULE["_validate_progressive_scan"](
            {
                "idet_aggregate": {"classification": "interlaced"},
                "probe": {"avg_frame_rate": "30000/1001"},
            },
            tmp_path / "source.mp4",
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required for the deinterlace integration test",
)
def test_tiny_tff_clip_is_deinterlaced_and_verified(tmp_path: Path) -> None:
    ffmpeg = str(Path(shutil.which("ffmpeg")).resolve())
    ffprobe = str(Path(shutil.which("ffprobe")).resolve())
    source = tmp_path / "interlaced.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=60000/1001:duration=4",
            "-vf",
            (
                "tinterlace=mode=interleave_top,setfield=tff,"
                "setparams=range=limited:color_primaries=bt709:"
                "color_trc=bt709:colorspace=bt709"
            ),
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30000/1001",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            str(source),
        ],
        check=True,
    )

    scan = MODULE["build_scan_report"](
        ffmpeg,
        ffprobe,
        [source],
        sample_count=1,
        sample_frames=60,
    )
    assert scan["sources"][0]["idet_aggregate"]["classification"] == "interlaced"
    assert scan["sources"][0]["idet_aggregate"]["parity"] == "tff"
    scan_path = tmp_path / "scan.json"
    MODULE["_atomic_json"](scan_path, scan)

    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text(
        json.dumps(
            {
                "name": "pilot",
                "source": str(source),
                "start_frame": 10,
                "frame_count": 30,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = MODULE["prepare_scenes"](
        ffmpeg,
        ffprobe,
        scan_report_path=scan_path,
        scenes_path=scenes,
        output_dir=tmp_path / "output",
        overwrite=False,
    )
    clip = provenance["clips"][0]
    assert clip["frame_count"] == 30
    assert clip["verification"]["passed"] is True
    assert clip["verification"]["probe"]["nb_read_frames"] == "30"
    assert clip["verification"]["probe"]["avg_frame_rate"] == "30000/1001"
    assert clip["verification"]["probe"]["field_order"] == "progressive"
    assert clip["verification"]["probe"]["color_range"] == "tv"
    assert clip["verification"]["probe"]["color_space"] == "bt709"
    assert Path(clip["output"]).is_file()
    assert len(clip["output_sha256"]) == 64


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required for the progressive preparation integration test",
)
@pytest.mark.parametrize("source_rate", ("30000/1001", "30", "60000/1001", "60"))
def test_progressive_clip_is_exact_lossless_cfr_and_not_resized(
    tmp_path: Path, source_rate: str
) -> None:
    ffmpeg = str(Path(shutil.which("ffmpeg")).resolve())
    ffprobe = str(Path(shutil.which("ffprobe")).resolve())
    safe_rate = source_rate.replace("/", "-")
    source = tmp_path / f"progressive-{safe_rate}.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x96:rate={source_rate}:duration=2",
            "-vf",
            (
                "setfield=prog,setparams=range=limited:color_primaries=bt709:"
                "color_trc=bt709:colorspace=bt709"
            ),
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            str(source),
        ],
        check=True,
    )
    probe = MODULE["probe_video"](ffprobe, source)
    scan = {
        "schema": MODULE["SCAN_SCHEMA"],
        "sources": [
            {
                "source": str(source.resolve()),
                "source_fingerprint": MODULE["sampled_file_fingerprint"](source),
                "probe": probe,
                "idet_aggregate": {
                    "classification": "progressive",
                    "parity": None,
                },
            }
        ],
    }
    scan_path = tmp_path / f"scan-{safe_rate}.json"
    MODULE["_atomic_json"](scan_path, scan)
    scenes = tmp_path / f"scenes-{safe_rate}.jsonl"
    scenes.write_text(
        json.dumps(
            {
                "name": f"pilot-{safe_rate}",
                "source": str(source),
                "start_frame": 3,
                "output_frame_count": 24,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    provenance = MODULE["prepare_progressive_scenes"](
        ffmpeg,
        ffprobe,
        scan_report_path=scan_path,
        scenes_path=scenes,
        output_dir=tmp_path / f"output-{safe_rate}",
        overwrite=False,
    )
    assert provenance["schema"] == MODULE["PROGRESSIVE_PROVENANCE_SCHEMA"]
    clip = provenance["clips"][0]
    assert clip["output_frame_count"] == 24
    assert clip["operation"]["spatial_resize"] is False
    assert clip["verification"]["passed"] is True
    output_probe = clip["verification"]["probe"]
    assert output_probe["nb_read_frames"] == "24"
    assert output_probe["avg_frame_rate"] == "30000/1001"
    assert output_probe["width"] == 160
    assert output_probe["height"] == 96
    assert output_probe["codec_name"] == "ffv1"
    assert output_probe["field_order"] == "progressive"
    assert output_probe["color_range"] == "tv"
    assert output_probe["color_space"] == "bt709"
    assert Path(clip["output"]).is_file()
    assert len(clip["output_sha256"]) == 64
    assert not any(
        path.name.endswith(".part.mkv")
        for path in Path(clip["output"]).parent.iterdir()
    )
