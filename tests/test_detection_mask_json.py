"""Synthetic mask/export tests; do not run a user video or change an app."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'packaging/macOS/standalone'


class DetectionMaskJSONTests(unittest.TestCase):
    def test_opt_in_persisted_and_export_only(self):
        app = (SOURCE / 'MiohApp.swift').read_text()
        pipeline = (SOURCE / 'NativePreviewPipeline.swift').read_text()
        self.assertIn('@Published var exportDetectionMasks = false', app)
        self.assertIn('exportDetectionMasks = snapshot.exportDetectionMasks ?? false', app)
        self.assertIn('overwriteDetectionMasks: overwrite', app)
        self.assertIn('!config.isExport || config.isWorker', pipeline)
        self.assertIn('"$PACKAGE_DIR/DetectionMaskJSON.swift"', (SOURCE / 'build_app.sh').read_text())

    def test_records_unique_output_frames_and_provenance(self):
        source = (SOURCE / 'NativePreviewPipeline.swift').read_text()
        self.assertIn('for frame in acceptedFrames { try appendMaskJSON(frame) }', source)
        self.assertIn('index >= detectedBatch.skipPrefix || maskExportFrames[pts] != nil', source)
        self.assertIn('accepted && belongsToCoreOutput(frame.ptsNanoseconds)', source)
        self.assertIn('maskJSON.frameCount == encodedFrames', source)
        self.assertIn('sampled[index] == nil ? "reused" : "model"', source)
        self.assertIn('? "model" : "empty_lookahead"', source)
        self.assertIn('defer { try? maskJSON?.finish(complete: false) }', source)
        self.assertLess(source.index('try NativeExportSupport.finishExport('), source.index('try maskJSON.finish(complete: true)'))

    def test_celf_snapshot_is_after_gap_fill_and_includes_both_crossfade_sides(self):
        source = (SOURCE / 'NativePreviewPipeline.swift').read_text()
        self.assertIn('fillLabelFieldGaps(&pending, maximumGap: 10)', source)
        self.assertIn('DetectionMaskJSONCELF.capture(item.labelField', source)
        self.assertIn('DetectionMaskJSONCELF.capture(frame.celfCrossfadePrevious', source)
        self.assertIn('previousCELF[deferredCrossfadeTail[index].ptsNanoseconds]', source)
        self.assertIn('crossfadeLaterWeight: frame.crossfadeLaterWeight', source)

    @unittest.skipUnless(sys.platform == 'darwin' and shutil.which('xcrun'), 'Swift compiler required')
    def test_projection_rle_streaming_and_interruption(self):
        with tempfile.TemporaryDirectory(prefix='mask-json-harness-') as directory:
            binary = Path(directory) / 'harness'
            compile_result = subprocess.run([
                'xcrun', 'swiftc', '-O', '-parse-as-library',
                str(SOURCE / 'DetectionMaskJSON.swift'),
                str(SOURCE / 'MosaicLabelFieldTracker.swift'),
                str(ROOT / 'tests/swift/DetectionMaskJSONHarness.swift'),
                '-o', str(binary)], capture_output=True, text=True, timeout=120)
            self.assertEqual(compile_result.returncode, 0, compile_result.stdout + compile_result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('Detection mask JSON tests passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
