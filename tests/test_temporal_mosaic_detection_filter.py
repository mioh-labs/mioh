"""Retired confirmation experiment and regression guard for its removal."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class UnfilteredMosaicDetectionPipelineTests(unittest.TestCase):
    def test_native_pipeline_keeps_single_frame_detections(self):
        pipeline = (ROOT / 'packaging/macOS/standalone/NativePreviewPipeline.swift').read_text()
        build = (ROOT / 'packaging/macOS/standalone/build_app.sh').read_text()
        for removed in ('TemporalMosaicDetectionFilter', 'detectionSupport', 'deliverConfirmed',
                        'isolated_detections_rejected', 'no second model observation'):
            self.assertNotIn(removed, pipeline)
        self.assertNotIn('TemporalMosaicDetectionFilter.swift', build)
        append = pipeline.split('func appendDetectedFrame(', 1)[1].split(
            'func processLookaheadWindow()', 1)[0]
        # Every detected frame is appended as is (the mask JSON adds its status and PTS).
        self.assertIn('frame: frame, detections: detections,', append)
        self.assertIn('labelField: labelField)', append)
        self.assertNotIn('.filter', append)
        self.assertNotIn('return', append)
        self.assertIn('detections: reusedDetections', pipeline)
        self.assertIn('if firstDetections.isEmpty && lastDetections.isEmpty', pipeline)

    def test_false_detection_scenes_are_not_restored(self):
        pipeline = (ROOT / 'packaging/macOS/standalone/NativePreviewPipeline.swift').read_text()
        self.assertIn('let scenes = Self.droppingFalseDetectionScenes(', pipeline)
        body = pipeline.split('private static func droppingFalseDetectionScenes(', 1)[1].split('func process(', 1)[0]
        self.assertIn('empty(index - 1) == true, empty(index + 1) == true', body)
        # a frame at a batch edge has an unknown neighbour and is kept
        self.assertIn('detected.indices.contains(index) ? detected[index].detections.isEmpty : nil', body)
        self.assertIn('(scene.frames.map(confidence).max() ?? 0) < 0.5', body)
        self.assertIn('return nearSide && nearEnd && small && still', body)

@unittest.skipUnless(shutil.which('swiftc') or shutil.which('xcrun'), 'Swift compiler required')
class TemporalMosaicDetectionFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix='mioh-detection-support-')
        cls.addClassCleanup(temporary.cleanup)
        executable = Path(temporary.name) / 'confirmation'
        compiler = ['swiftc'] if shutil.which('swiftc') else ['xcrun', 'swiftc']
        result = subprocess.run([
            *compiler, '-O', '-parse-as-library',
            str(ROOT / 'packaging/macOS/standalone/TemporalMosaicDetectionFilter.swift'),
            str(ROOT / 'tests/swift/TemporalMosaicDetectionFilterHarness.swift'),
            '-o', str(executable),
        ], text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        run = subprocess.run([str(executable)], text=True, capture_output=True, timeout=30)
        if run.returncode:
            raise AssertionError(run.stdout + run.stderr)
        cls.cases = json.loads(run.stdout)

    def test_singletons_are_removed_at_start_middle_and_eof(self):
        for name in ('empty', 'only_frame', 'first_singleton', 'middle_singleton', 'last_singleton'):
            with self.subTest(name=name):
                self.assertTrue(all(not boxes for boxes in self.cases[name]))

    def test_confirmed_tracks_and_short_missed_detections_survive(self):
        self.assertEqual(self.cases['two_frames'], [[0], [0]])
        self.assertEqual(self.cases['intermittent_track'], [[0], [], [], [0]])
        self.assertEqual(self.cases['batch_boundary'], [[], [0], [0], []])

    def test_distant_or_unrelated_candidates_do_not_confirm_each_other(self):
        for name in ('outside_horizon', 'two_boxes_one_observation', 'different_objects', 'different_classes'):
            with self.subTest(name=name):
                self.assertTrue(all(not boxes for boxes in self.cases[name]))
        self.assertEqual(self.cases['one_extra_object'], [[0], [0], [0]])

    def test_reuse_is_not_independent_model_evidence(self):
        self.assertEqual(self.cases['reused_singleton'], [[], [], [], []])
        self.assertEqual(self.cases['confirmed_reuse'], [[0], [0], [0], [0]])
