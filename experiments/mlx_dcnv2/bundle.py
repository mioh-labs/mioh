"""Load exported LADA MLX inference weight bundles."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_lada_mlx_bundle(manifest_path: str | Path) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    return {
        "feature_extract": _load_npz(root / manifest["feature_extract"]["npz"]),
        "spynet": _load_npz(root / manifest["spynet"]["npz"]),
        "alignment": {
            name: _load_npz(root / row["npz"])
            for name, row in manifest["modules"].items()
        },
        "backbones": {
            name: _load_npz(root / row["npz"])
            for name, row in manifest["backbones"].items()
        },
        "reconstruction": _load_npz(root / manifest["reconstruction"]["npz"]),
    }


def _load_npz(path: Path) -> dict[str, mx.array]:
    data = np.load(path)
    return {name: mx.array(data[name].astype(np.float32, copy=False)) for name in data.files}
