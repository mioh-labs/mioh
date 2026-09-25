"""Small deterministic checks for the BasicVSR++ degradation benchmark."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/training/evaluate-basicvsrpp-degradation-grid.py"
)
SPEC = importlib.util.spec_from_file_location("basicvsrpp_degradation_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grid)


@pytest.mark.parametrize("count", [1, 2, 9, 26, 48])
def test_windows_score_the_same_center_frame(count: int) -> None:
    start, center = grid.window_for(count, 48)
    assert start + center == 24
    assert 0 <= center < count


def test_mosaic_changes_only_the_roi_and_keeps_grid_fixed() -> None:
    rng = np.random.default_rng(17)
    clean = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
    roi = np.zeros((256, 256), dtype=bool)
    roi[64:192, 64:192] = True
    observed = grid.mosaic_clip(clean, 16, (3, 5), roi)
    np.testing.assert_array_equal(observed[:, ~roi], clean[:, ~roi])
    assert np.any(observed[:, roi] != clean[:, roi])


def test_perfect_prediction_has_perfect_metrics() -> None:
    rng = np.random.default_rng(9)
    target = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    roi = np.ones((256, 256), dtype=bool)
    result = grid.score(target, target, roi)
    assert result["roi_psnr_db"] == pytest.approx(120.0)
    assert result["hf_correlation"] == pytest.approx(1.0, abs=1e-5)
    assert result["hf_corr_times_amp"] == pytest.approx(1.0, abs=1e-5)


def test_crop_and_crf_arguments_accept_zero() -> None:
    assert grid.parse_crop("0,0,256,256") == (0, 0, 256, 256)
    assert grid.nonnegative_int_list("0,18") == [0, 18]
    assert grid.nonnegative_int_list("none") == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_h264_roundtrip_keeps_frame_count_and_shape() -> None:
    frames = np.zeros((3, 256, 256, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.array((32, 128, 224), dtype=np.uint8)[:, None, None]
    decoded = grid.h264_roundtrip(frames, 18, "ffmpeg")
    assert decoded.shape == frames.shape
    assert decoded.dtype == np.uint8
