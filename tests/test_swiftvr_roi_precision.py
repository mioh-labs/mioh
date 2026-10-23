"""Opt-in real-model ROI precision regression using synthetic moving patterns.

Run with MIOH_TEST_SWIFTVR_MODEL_ROOT pointing to the SwiftVR ROI model pack.
This is intentionally skipped by ordinary CI without those external weights.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = os.environ.get('MIOH_TEST_SWIFTVR_MODEL_ROOT')


@unittest.skipUnless(sys.platform == 'darwin' and shutil.which('xcrun') and MODEL_ROOT,
                     'opt-in SwiftVR model pack and native Core ML required')
class SwiftVRROIPrecisionTests(unittest.TestCase):
    def test_default_fp16_matches_widened_path_at_both_scales(self):
        with tempfile.TemporaryDirectory(prefix='swiftvr-roi-precision-') as temporary:
            executable = Path(temporary) / 'benchmark'
            compile_result = subprocess.run([
                'xcrun', 'swiftc', '-O', '-parse-as-library',
                '-D', 'MIOH_NATIVE_PREVIEW_PIPELINE', '-target', 'arm64-apple-macosx27.0',
                '-framework', 'CoreGraphics', '-framework', 'CoreML',
                '-framework', 'ImageIO', '-framework', 'UniformTypeIdentifiers',
                '-framework', 'CoreMedia', '-framework', 'CoreVideo',
                '-framework', 'VideoToolbox', '-framework', 'CryptoKit',
                str(ROOT / 'packaging/macOS/standalone/SwiftVRNativeClip.swift'),
                str(ROOT / 'packaging/macOS/standalone/SwiftVRInlineEnhancer.swift'),
                str(ROOT / 'tests/swift/SwiftVRROIPrecisionHarness.swift'),
                '-o', str(executable),
            ], text=True, capture_output=True, timeout=180)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stdout + compile_result.stderr)
            singleton = subprocess.run([
                str(executable), MODEL_ROOT, '2', 'singleton', '1',
            ], text=True, capture_output=True, timeout=30)
            self.assertEqual(singleton.returncode, 0, singleton.stdout + singleton.stderr)
            self.assertEqual(json.loads(singleton.stdout), {'skipped': True, 'frames': 1})
            self.assertNotIn('Prepared ', singleton.stdout)
            for scale in (2, 4):
                rows = []
                for mode in ('fp32', 'fp16'):
                    with self.subTest(scale=scale, mode=mode):
                        result = subprocess.run([
                            str(executable), MODEL_ROOT, str(scale), mode, '49',
                        ], text=True, capture_output=True, timeout=600)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        row = next(json.loads(line) for line in result.stdout.splitlines()
                                   if line.startswith('{'))
                        self.assertEqual(row['frames'], 49)
                        self.assertEqual(len(row['frame_sha256']), 49)
                        rows.append(row)
                self.assertEqual(rows[0]['frame_sha256'], rows[1]['frame_sha256'])
