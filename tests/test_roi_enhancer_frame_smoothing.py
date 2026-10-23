"""Geometry tests use synthetic patterns only; do not start a user export."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'packaging/macOS/standalone'


class ROIEnhancerFrameSmoothingIntegrationTests(unittest.TestCase):
    def test_smoothing_does_not_emit_repeated_scene_notice(self):
        pipeline = (SOURCE / 'NativePreviewPipeline.swift').read_text()
        self.assertNotIn('residual mapped to original ROI', pipeline)
        self.assertNotIn('ROI Distill: frame smoothing', pipeline)
        self.assertIn('ROIEnhancerFrameSmoothing.mapResidualBack(', pipeline)

    def test_setting_reaches_distilled_backend_without_changing_image_models(self):
        app = (SOURCE / 'MiohApp.swift').read_text()
        pipeline = (SOURCE / 'NativePreviewPipeline.swift').read_text()
        self.assertIn('swiftVRFrameSmoothing: roiEnhancerSupportsFrameSmoothing', app)
        self.assertEqual(app.count('integerSliderField("枠のなめらかさ"'), 1)
        self.assertIn('枠のなめらかさ: \\(configuration.swiftVRFrameSmoothing ?? 0)', app)
        self.assertIn('self.frameSmoothing = max(0, min(30, frameSmoothing))', pipeline)
        self.assertIn('frameSmoothing: config.swiftVRFrameSmoothing ?? 0', pipeline)
        self.assertIn('geometries, window: roiEnhancer.frameSmoothing, frameIndices: enhancerFrameIndices', pipeline)
        self.assertIn('restored: temporalInput ?? restored', pipeline)
        self.assertIn('ROIEnhancerFrameSmoothing.mapResidualBack(', pipeline)
        self.assertIn('"$PACKAGE_DIR/ROIEnhancerFrameSmoothing.swift"', (SOURCE / 'build_app.sh').read_text())

    @unittest.skipUnless(sys.platform == 'darwin' and shutil.which('xcrun'), 'Swift compiler required')
    def test_smoothing_gaps_mapping_and_unchanged_restoration_base(self):
        with tempfile.TemporaryDirectory(prefix='roi-frame-smoothing-test-') as directory:
            executable = Path(directory) / 'harness'
            built = subprocess.run([
                'xcrun', 'swiftc', '-O', '-parse-as-library',
                str(SOURCE / 'ROIEnhancerFrameSmoothing.swift'),
                str(ROOT / 'tests/swift/ROIEnhancerFrameSmoothingHarness.swift'),
                '-o', str(executable)], capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('ROI frame smoothing tests passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
