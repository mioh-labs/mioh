import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from experiments.mlx_dcnv2.bundle import load_lada_mlx_bundle


class MLXBundleTests(unittest.TestCase):
    def test_load_lada_mlx_bundle_returns_nested_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "feat.npz", **{"0.weight": np.ones((2, 3, 3, 3), dtype=np.float32)})
            np.savez(root / "spynet.npz", mean=np.ones((1, 3, 1, 1), dtype=np.float32))
            np.savez(root / "reconstruction.npz", **{"conv_last.bias": np.ones((3,), dtype=np.float32)})
            np.savez(root / "align_b1.npz", weight=np.ones((2, 4, 3, 3), dtype=np.float32))
            np.savez(root / "backbone_b1.npz", **{"main.0.bias": np.ones((2,), dtype=np.float32)})
            manifest = {
                "feature_extract": {"npz": "feat.npz"},
                "spynet": {"npz": "spynet.npz"},
                "reconstruction": {"npz": "reconstruction.npz"},
                "modules": {"backward_1": {"npz": "align_b1.npz"}},
                "backbones": {"backward_1": {"npz": "backbone_b1.npz"}},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            bundle = load_lada_mlx_bundle(manifest_path)

            self.assertIsInstance(bundle["feature_extract"]["0.weight"], mx.array)
            self.assertEqual(bundle["feature_extract"]["0.weight"].shape, (2, 3, 3, 3))
            self.assertEqual(bundle["spynet"]["mean"].shape, (1, 3, 1, 1))
            self.assertEqual(bundle["alignment"]["backward_1"]["weight"].shape, (2, 4, 3, 3))
            self.assertEqual(bundle["backbones"]["backward_1"]["main.0.bias"].shape, (2,))
            self.assertEqual(bundle["reconstruction"]["conv_last.bias"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
