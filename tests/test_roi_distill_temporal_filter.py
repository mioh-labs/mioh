"""Apple temporal filtering with synthetic pixels only, never user media."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'packaging/macOS/standalone'


class ROIDistillTemporalFilterIntegrationTests(unittest.TestCase):
    def test_filter_is_opt_in_for_tensor_models_and_shared_in_ui(self):
        app = (SOURCE / 'MiohApp.swift').read_text()
        pipeline = (SOURCE / 'NativePreviewPipeline.swift').read_text()
        self.assertIn('swiftVRTemporalFilter: roiEnhancer != "none"', app)
        self.assertEqual(app.count('doubleSliderField("揺らぎ低減"'), 1)
        self.assertIn('temporalFilter: config.swiftVRTemporalFilter ?? 0', pipeline)
        self.assertIn('private(set) var temporalFilter: Float = 0', pipeline)
        model_section = pipeline[pipeline.index('private final class NativeROIEnhancer'):]
        model_section = model_section[:model_section.index('func enhanceFrame(')]
        self.assertIn('!VTTemporalNoiseFilterConfiguration.isSupported', model_section)
        self.assertIn('self.temporalFilter = requestedFilter', model_section)
        self.assertNotIn('residual mapped to original ROI', pipeline)

    def test_composited_coordinates_bounded_lookahead_and_tail_flush(self):
        pipeline = (SOURCE / 'NativePreviewPipeline.swift').read_text()
        branch = pipeline[pipeline.index('if let roiEnhancer {\n        guard let originalInput'):]
        branch = branch[:branch.index('let compositionStart = Date()\n      var composed')]
        self.assertIn('ROIEnhancerTemporalWindow.contiguousRuns(', branch)
        self.assertIn('roiEnhancer.temporalFilter > 0 && filterRun.count > 1', branch)
        self.assertLess(branch.index('outputs[frameIndex] = try composite('),
                        branch.index('try filter.add(index, from: outputs[frameIndex]'))
        self.assertIn('let ready = index - lookahead', branch)
        self.assertIn('value: detected[frameIndex].frame.ptsNanoseconds', branch)
        self.assertIn('filterRun.upperBound - lookahead', branch)
        self.assertEqual(branch.count('featherFraction: 0, restrictToEnhancerMask: true'), 2)
        self.assertEqual(branch.count('exclusiveWeights: compositePlans[ready].blendMask'), 2)
        self.assertNotIn('FileHandle.standardError', branch)

    @unittest.skipUnless(sys.platform == 'darwin' and shutil.which('xcrun'), 'Apple frameworks required')
    def test_apple_filter_and_masked_writeback_on_synthetic_frames(self):
        with tempfile.TemporaryDirectory(prefix='roi-distill-temporal-test-') as directory:
            executable = Path(directory) / 'harness'
            result = subprocess.run([
                'xcrun', 'swiftc', '-O', '-parse-as-library',
                '-D', 'MIOH_NATIVE_PREVIEW_PIPELINE', '-target', 'arm64-apple-macosx27.0',
                '-framework', 'CoreGraphics', '-framework', 'CoreML',
                '-framework', 'ImageIO', '-framework', 'UniformTypeIdentifiers',
                '-framework', 'CoreMedia', '-framework', 'CoreVideo',
                '-framework', 'VideoToolbox', '-framework', 'CryptoKit',
                str(SOURCE / 'SwiftVRNativeClip.swift'),
                str(SOURCE / 'SwiftVRInlineEnhancer.swift'),
                str(ROOT / 'tests/swift/ROIDistillTemporalFilterHarness.swift'),
                '-o', str(executable)], capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('ROI temporal masked writeback tests passed', run.stdout)
            if 'Apple temporal filter unsupported' in run.stdout:
                self.skipTest('This Mac does not support the Apple temporal processor')
            self.assertIn('Apple temporal filter processed 7 synthetic frames', run.stdout)


if __name__ == '__main__':
    unittest.main()
