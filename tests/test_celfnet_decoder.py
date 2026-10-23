"""Check centre/offset separation and temporal feedback against the Python oracle."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy import ndimage

from lada.models.celfnet.decoder import decode_objects

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if sys.platform != "darwin" or not shutil.which("swiftc"):
        pytest.skip("native Swift harness requires macOS")
    path = tmp_path_factory.mktemp("celfnet") / "decoder"
    subprocess.run([
        "swiftc", "-O", "-parse-as-library",
        str(ROOT / "packaging/macOS/standalone/CelfNetObjectDecoder.swift"),
        str(ROOT / "tests/swift/CelfNetDecoderHarness.swift"), "-o", str(path),
    ], check=True, capture_output=True)
    return path


def make_output(case):
    out = np.zeros((11, 136, 240), np.float32)
    out[0] = -10; out[8] = -10
    yy, xx = np.mgrid[:136, :240]

    def region(ys, xs, centre, logit=5):
        out[0, ys, xs] = logit
        cy, cx = centre
        out[8, cy, cx] = 5
        out[9, ys, xs] = (cx - xx[ys, xs]) / 16
        out[10, ys, xs] = (cy - yy[ys, xs]) / 16

    if case == "empty":
        return out
    if case == "touching":
        region(slice(25, 45), slice(40, 60), (35, 50))
        region(slice(25, 45), slice(60, 80), (35, 70))
    elif case == "fragmented":
        region(slice(25, 35), slice(40, 60), (30, 50))
        region(slice(37, 47), slice(40, 60), (30, 50))
    elif case == "orphan":
        region(slice(25, 45), slice(40, 60), (35, 50))
        out[0, 25:45, 60:70] = 5
        out[9, 25:45, 60:70] = (90 - xx[25:45, 60:70]) / 16
        out[10, 25:45, 60:70] = (35 - yy[25:45, 60:70]) / 16
    elif case == "weak":
        region(slice(25, 45), slice(40, 60), (35, 50), logit=-0.5)
    elif case == "padding":
        region(slice(0, 8), slice(0, 20), (4, 10))
    elif case == "tiny":
        region(slice(25, 45), slice(40, 60), (35, 50))
        out[9, 25:28, 57:60] = (70 - xx[25:28, 57:60]) / 16
        out[10, 25:28, 57:60] = (26 - yy[25:28, 57:60]) / 16
        out[8, 26, 70] = 5
    return out


def reference_prior(objects):
    labels = np.zeros((544, 960), np.int32)
    for k, obj in enumerate(objects, 1):
        full = np.repeat(np.repeat(obj["mask"], 4, axis=0), 4, axis=1)
        labels[2:542][full[2:542]] = k
    edge = np.zeros_like(labels, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        nb = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)
        edge |= (labels > 0) & (nb > 0) & (nb != labels)
    centres = np.zeros(labels.shape, np.float32)
    yi, xi = np.mgrid[:544, :960].astype(np.float32)
    for k in range(1, len(objects) + 1):
        cy, cx = ndimage.center_of_mass(labels == k)
        if not np.isfinite(cy):
            continue
        y0, y1 = max(0, int(cy - 24)), min(544, int(cy + 24) + 1)
        x0, x1 = max(0, int(cx - 24)), min(960, int(cx + 24) + 1)
        g = np.exp(-((yi[y0:y1, x0:x1] - cy) ** 2 + (xi[y0:y1, x0:x1] - cx) ** 2) / 128)
        centres[y0:y1, x0:x1] = np.maximum(centres[y0:y1, x0:x1], g)
    return np.stack([np.zeros_like(centres), ndimage.binary_dilation(edge, iterations=2), centres])


@pytest.mark.parametrize("case", ["empty", "touching", "fragmented", "orphan", "weak", "padding", "tiny"])
def test_native_decoder_and_feedback_match_python(harness, tmp_path, case):
    output = make_output(case)
    source, prior_path = tmp_path / "output.bin", tmp_path / "prior.bin"
    output.tofile(source)
    actual = json.loads(subprocess.check_output([str(harness), str(source), str(prior_path)]))
    expected = decode_objects(output)
    assert actual["zeroChannel"] and actual["paddingEmpty"]
    assert len(actual["objects"]) == len(expected)
    for a, e in zip(actual["objects"], expected):
        assert a["pixels"] == np.flatnonzero(e["mask"]).tolist()
        assert abs(a["score"] - e["score"]) < 1e-5
    prior = np.fromfile(prior_path, np.float32).reshape(3, 544, 960)
    np.testing.assert_allclose(prior, reference_prior(expected), atol=1e-6, rtol=1e-5)
