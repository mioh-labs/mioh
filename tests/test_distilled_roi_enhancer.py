"""Synthetic temporal ROI regression; never reads or starts a user video."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DistilledROIIntegrationTests(unittest.TestCase):
    def test_ui_and_native_pipeline_use_the_temporal_backend(self):
        folder = ROOT / "packaging/macOS/standalone"
        app = (folder / "MiohApp.swift").read_text()
        pipeline = (folder / "NativePreviewPipeline.swift").read_text()
        build = (folder / "build_app.sh").read_text()
        self.assertIn('case "roi-distill":', app)
        self.assertIn('name: "roi-distill-x2-coreml"', app)
        self.assertIn('coreMLPrefixes = ["roi-distill-x2"]', app)
        self.assertIn('if roiEnhancer == "roi-distill" { roiEnhancerScale = 2 }', app)
        self.assertIn('MIOH_ROI_DISTILL_COREML_MODEL', build)
        self.assertIn('"$RESOURCES/models/roi-distill-x2.mlmodelc"', build)
        self.assertIn('"$PACKAGE_DIR/DistilledROIEnhancer.swift"', build)
        self.assertIn('let enhancerFrameIndices = scene.frames.map(\\.batchIndex)', pipeline)
        self.assertIn('frameIndices: enhancerFrameIndices', pipeline)
        self.assertIn('mask.contains(where: { $0 > 0.5 })', pipeline)
        self.assertIn('configuration.computeUnits = .cpuAndNeuralEngine', pipeline)
        self.assertIn('DistilledROIEnhancer(model: loaded, scale: self.scale)', pipeline)
        self.assertNotIn('temporalConfiguration.computeUnits = .cpuAndGPU', pipeline)
        backend = (folder / "DistilledROIEnhancer.swift").read_text()
        self.assertIn('static let computeUnits: MLComputeUnits = .cpuAndNeuralEngine', backend)


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("xcrun"), "native Core ML required")
class DistilledROIEnhancerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="distilled-roi-regression-")
        cls.executable = Path(cls.temp.name) / "harness"
        result = subprocess.run([
            "xcrun", "swiftc", "-O", "-parse-as-library", "-framework", "CoreML",
            str(ROOT / "packaging/macOS/standalone/DistilledROIEnhancer.swift"),
            str(ROOT / "tests/swift/DistilledROIEnhancerHarness.swift"),
            "-o", str(cls.executable)], capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_temporal_boundaries_gaps_and_invalid_indices(self):
        result = subprocess.run([str(self.executable)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("temporal window tests passed", result.stdout)

    @unittest.skipUnless(os.environ.get("MIOH_TEST_DISTILLED_ROI_MODEL") and
                         os.environ.get("MIOH_TEST_DISTILLED_ROI_CHECKPOINT"), "opt-in learned model required")
    def test_real_swift_backend_matches_pytorch(self):
        import numpy as np
        import torch
        sys.path.insert(0, str(ROOT / "notebooks"))
        import roi_enhancer_distill_runtime as runtime
        folder = Path(self.temp.name) / "outputs"
        result = subprocess.run([str(self.executable), os.environ["MIOH_TEST_DISTILLED_ROI_MODEL"], str(folder)],
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Core ML's compiler may emit a warning before the JSON report.
        reports = [json.loads(line) for line in result.stdout.splitlines()
                   if line.startswith('{')]
        self.assertEqual(len(reports), 1, result.stdout + result.stderr)
        report = reports[0]
        self.assertEqual(report["compute_units"], "CPU_AND_NEURAL_ENGINE")
        self.assertEqual(report["windows"], [[0, 0, 0, 1, 2], [0, 1, 2, 2, 2],
                                              [3, 3, 3, 4, 5], [4, 5, 6, 6, 6]])
        torch.set_num_threads(2)
        model, _ = runtime.load_checkpoint(os.environ["MIOH_TEST_DISTILLED_ROI_CHECKPOINT"])
        model.eval()
        pixels = np.arange(3 * 256 * 256) % 127
        source = np.stack([((frame * 17 + pixels) % 251).astype(np.float32) / 250 for frame in range(7)])
        source = torch.from_numpy(source.astype(np.float16).astype(np.float32))
        errors = []
        with torch.inference_mode():
            for centre, window in zip(report["tested_centres"], report["windows"]):
                expected = model(source[window].reshape(1, 15, 256, 256)).clamp(0, 1).numpy()
                actual = np.fromfile(folder / f"frame-{centre}.f32", dtype=np.float32).reshape(1, 3, 512, 512)
                self.assertTrue(np.isfinite(actual).all())
                error = float(np.max(np.abs(expected - actual)))
                self.assertLess(error, .002)
                errors.append(error)
        print(json.dumps({**report, "max_abs_errors_vs_pytorch": errors}))


if __name__ == "__main__":
    unittest.main()
