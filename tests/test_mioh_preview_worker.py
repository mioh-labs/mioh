import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import av
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "packaging" / "macOS" / "standalone" / "mioh_preview_worker.py"


def load_worker_module():
    spec = importlib.util.spec_from_file_location("mioh_preview_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreviewProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load_worker_module()

    def test_emit_event_includes_kind_and_generation(self):
        output = io.StringIO()

        self.worker.emit_event("ready", generation=3, stream=output, duration=12.5)

        event = json.loads(output.getvalue())
        self.assertEqual(event["kind"], "ready")
        self.assertEqual(event["generation"], 3)
        self.assertEqual(event["duration"], 12.5)

    def test_preview_command_parses_seek(self):
        command = self.worker.PreviewCommand.parse(
            '{"command":"seek","position_ns":42000000000}'
        )

        self.assertEqual(command.command, "seek")
        self.assertEqual(command.position_ns, 42_000_000_000)


class SegmentEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load_worker_module()

    def test_writes_independent_two_second_segments_and_final_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder = self.worker.SegmentEncoder(
                output_dir=Path(temp_dir),
                width=32,
                height=24,
                fps=Fraction(4, 1),
                generation=2,
                preferred_codec="libx264",
                segment_seconds=2.0,
            )
            events = []
            for index in range(10):
                frame = np.full((24, 32, 3), index * 10, dtype=np.uint8)
                events.extend(encoder.add_frame(frame, pts_ns=index * 250_000_000))
            events.extend(encoder.finish())

            self.assertEqual([event["sequence"] for event in events], [0, 1])
            self.assertEqual(events[0]["start_ns"], 0)
            self.assertEqual(events[0]["end_ns"], 2_000_000_000)
            self.assertEqual(events[1]["start_ns"], 2_000_000_000)
            self.assertEqual(events[1]["end_ns"], 2_500_000_000)
            for event in events:
                path = Path(event["path"])
                self.assertTrue(path.is_file())
                with av.open(path) as container:
                    self.assertGreater(sum(1 for _ in container.decode(video=0)), 0)

    def test_falls_back_to_ultrafast_libx264_when_videotoolbox_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            attempts = []

            def stream_factory(container, codec, rate):
                attempts.append(codec)
                if codec == "h264_videotoolbox":
                    raise RuntimeError("VideoToolbox unavailable")
                stream = container.add_stream(codec, rate=rate)
                return stream

            encoder = self.worker.SegmentEncoder(
                output_dir=Path(temp_dir),
                width=32,
                height=24,
                fps=Fraction(4, 1),
                generation=0,
                preferred_codec="h264_videotoolbox",
                stream_factory=stream_factory,
            )

            encoder.add_frame(np.zeros((24, 32, 3), dtype=np.uint8), 0)
            encoder.finish()

            self.assertEqual(attempts[:2], ["h264_videotoolbox", "libx264"])
            self.assertEqual(encoder.active_codec, "libx264")
            self.assertEqual(encoder.active_options.get("preset"), "ultrafast")

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe is required")
    def test_ffprobe_reads_ordered_independent_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder = self.worker.SegmentEncoder(
                output_dir=Path(temp_dir),
                width=64,
                height=48,
                fps=Fraction(12, 1),
                generation=4,
                preferred_codec="libx264",
                segment_seconds=2.0,
            )
            events = []
            for index in range(60):
                frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
                events.extend(
                    encoder.add_frame(frame, pts_ns=index * 1_000_000_000 // 12)
                )
            events.extend(encoder.finish())

            self.assertEqual([event["sequence"] for event in events], [0, 1, 2])
            self.assertEqual(
                [Path(event["path"]).name for event in events],
                [
                    "preview-g4-000000.mp4",
                    "preview-g4-000001.mp4",
                    "preview-g4-000002.mp4",
                ],
            )
            for event in events:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,nb_frames",
                        "-of",
                        "json",
                        event["path"],
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                stream = json.loads(result.stdout)["streams"][0]
                self.assertEqual(stream["codec_name"], "h264")
                self.assertGreater(int(stream["nb_frames"]), 0)


class PreviewSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load_worker_module()

    def make_config(self):
        return SimpleNamespace(
            input="input.mp4",
            device="mps",
            fp16=True,
            restoration_model="basicvsrpp-v1.2",
            detection_model="v2-coreml",
            max_clip_length=180,
            restore_max_frames=None,
            detect_face_mosaics=False,
            detection_empty_lookahead=10,
            sharpen_strength=0.0,
            detail_boost=0.0,
            blend_feather=1.0,
            texture_mix=0.0,
            smooth_strength=0.0,
            roi_enhancer="none",
            roi_enhancer_model="",
            roi_enhancer_scale=4,
            roi_enhancer_strength=0.0,
            roi_enhancer_tile=0,
            effect_upscale=1,
            segment_seconds=2.0,
            buffer_limit=8.0,
        )

    def test_seek_reuses_loaded_models_and_restarts_only_frame_restorer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loaded_models = object()
            loader = mock.Mock(return_value=loaded_models)
            restorers = []

            class FakeRestorer:
                def __init__(self):
                    self.starts = []
                    self.stopped = False

                def start(self, start_ns=0):
                    self.starts.append(start_ns)

                def stop(self):
                    self.stopped = True

            def factory(config, models):
                self.assertIs(models, loaded_models)
                restorer = FakeRestorer()
                restorers.append(restorer)
                return restorer

            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                model_loader=loader,
                restorer_factory=factory,
            )
            session.start_generation(0)
            stale_path = Path(temp_dir) / "preview-g0-000000.mp4"
            stale_path.touch()

            session.seek(42_000_000_000)

            loader.assert_called_once()
            self.assertEqual(len(restorers), 2)
            self.assertTrue(restorers[0].stopped)
            self.assertEqual(restorers[1].starts, [42_000_000_000])
            self.assertEqual(session.generation, 1)
            self.assertFalse(stale_path.exists())

    def test_buffer_capacity_is_bounded_to_four_two_second_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                model_loader=lambda _config: object(),
                restorer_factory=lambda _config, _models: mock.Mock(),
            )
            session.generation = 7
            for sequence in range(4):
                (Path(temp_dir) / f"preview-g7-{sequence:06d}.mp4").touch()

            self.assertFalse(session.has_buffer_capacity())
            (Path(temp_dir) / "preview-g7-000000.mp4").unlink()
            self.assertTrue(session.has_buffer_capacity())

    def test_buffer_capacity_excludes_active_encoder_segment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                model_loader=lambda _config: object(),
                restorer_factory=lambda _config, _models: mock.Mock(),
            )
            session.generation = 7
            for sequence in range(3):
                (Path(temp_dir) / f"preview-g7-{sequence:06d}.mp4").touch()
            active_path = Path(temp_dir) / "preview-g7-000003.mp4"
            active_path.touch()
            session.encoder = SimpleNamespace(segment_path=active_path)

            self.assertTrue(session.has_buffer_capacity())

            session.encoder.segment_path = None
            self.assertFalse(session.has_buffer_capacity())

    def test_set_buffer_limit_emits_applied_value_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                model_loader=lambda _config: object(),
                restorer_factory=lambda _config, _models: mock.Mock(),
            )
            session.generation = 9
            output = io.StringIO()

            with mock.patch.object(self.worker, "PROTOCOL_STREAM", output):
                session._apply_command(
                    self.worker.PreviewCommand.parse(
                        '{"command":"set_buffer_limit","seconds":30}'
                    )
                )

            self.assertEqual(session.config.buffer_limit, 30.0)
            self.assertEqual(
                json.loads(output.getvalue()),
                {"kind": "buffer_limit", "generation": 9, "seconds": 30.0},
            )

    def test_stop_releases_restorer_and_removes_session_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            output_dir = Path(parent) / "preview-session"
            output_dir.mkdir()
            restorer = mock.Mock()
            session = self.worker.PreviewSession(
                self.make_config(),
                output_dir,
                model_loader=lambda _config: object(),
                restorer_factory=lambda _config, _models: restorer,
            )
            session.start_generation(0)

            session.stop(remove_output=True)

            restorer.stop.assert_called_once()
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
