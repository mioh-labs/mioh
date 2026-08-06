import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
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

    def test_preview_command_rejects_seek_because_seek_is_process_scoped(self):
        with self.assertRaisesRegex(ValueError, "Unsupported preview command"):
            self.worker.PreviewCommand.parse(
                '{"command":"seek","position_ns":42000000000,"generation":17}'
            )

    def test_preview_parser_accepts_temporal_overlap(self):
        args = self.worker.build_parser().parse_args([
            "--input", "in.mp4",
            "--output-dir", "out",
            "--restoration-model", "basicvsrpp-v1.2",
            "--detection-model", "v2-coreml",
            "--generation", "12",
            "--restore-temporal-overlap", "15",
            "--disable-crossfade",
        ])

        self.assertEqual(args.generation, 12)
        self.assertEqual(args.restore_temporal_overlap, 15)
        self.assertFalse(args.restore_crossfade)

    def test_preview_parser_enables_realtime_optimization(self):
        args = self.worker.build_parser().parse_args([
            "--input", "in.mp4",
            "--output-dir", "out",
            "--restoration-model", "basicvsrpp-v1.2-coreai-variable",
            "--detection-model", "v4-accurate-coreai",
            "--realtime-optimize",
        ])

        self.assertTrue(args.realtime_optimize)

    def test_native_swift_preview_requires_variable_coreai_models(self):
        args = self.worker.build_parser().parse_args([
            "--input", "in.mp4",
            "--output-dir", "out",
            "--restoration-model", "basicvsrpp-v1.2-coreai-variable",
            "--detection-model", "v4-accurate-coreai",
        ])
        with tempfile.NamedTemporaryFile() as runner, mock.patch.dict(
            os.environ,
            {
                "LADA_NATIVE_SWIFT_PREVIEW": "1",
                "LADA_NATIVE_COREAI_PREVIEW_RUNNER": runner.name,
            },
            clear=False,
        ):
            compatible, reason = (
                self.worker._native_swift_preview_compatibility(args)
            )
        self.assertTrue(compatible, reason)

        args.restoration_model = "basicvsrpp-v1.2"
        with mock.patch.dict(
            os.environ,
            {"LADA_NATIVE_SWIFT_PREVIEW": "1"},
            clear=False,
        ):
            compatible, _ = self.worker._native_swift_preview_compatibility(args)
        self.assertFalse(compatible)

    def test_native_swift_preview_preserves_nonzero_postprocessing(self):
        args = self.worker.build_parser().parse_args([
            "--input", "in.mp4",
            "--output-dir", "out",
            "--restoration-model", "basicvsrpp-v1.2-coreai-variable",
            "--detection-model", "v4-accurate-coreai",
            "--smooth-strength", "0.15",
        ])
        with tempfile.NamedTemporaryFile() as runner, mock.patch.dict(
            os.environ,
            {
                "LADA_NATIVE_SWIFT_PREVIEW": "1",
                "LADA_NATIVE_COREAI_PREVIEW_RUNNER": runner.name,
            },
            clear=False,
        ):
            compatible, reason = (
                self.worker._native_swift_preview_compatibility(args)
            )
        self.assertFalse(compatible)
        self.assertIn("postprocessing", reason)

    def test_native_swift_preview_maps_coreai_detector_to_coreml_lane(self):
        self.assertEqual(
            self.worker._native_swift_coreml_detector_name(
                "v4-accurate-coreai"
            ),
            "v4-accurate-coreml",
        )
        self.assertEqual(
            self.worker._native_swift_coreml_detector_name("v2-coreai"),
            "v2-coreml",
        )
        self.assertIsNone(
            self.worker._native_swift_coreml_detector_name(
                "v4-accurate-coreml"
            )
        )

    def test_native_swift_preview_treats_negative_restore_max_frames_as_auto(self):
        config = SimpleNamespace(
            restore_max_frames=-1,
            max_clip_length=180,
        )

        self.assertEqual(
            self.worker._native_temporal_frame_count(config, 36),
            36,
        )

    def test_native_swift_preview_honors_positive_restore_max_frames(self):
        config = SimpleNamespace(
            restore_max_frames=12,
            max_clip_length=180,
        )

        self.assertEqual(
            self.worker._native_temporal_frame_count(config, 36),
            12,
        )

    def test_native_swift_realtime_payload_uses_t30_overlap_and_crossfade(self):
        source = WORKER_PATH.read_text()

        for contract in [
            "temporal_frame_limit = 30",
            '"temporalOverlap": temporal_overlap',
            '"crossfade": bool(config.restore_crossfade)',
        ]:
            self.assertIn(contract, source)


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
                if encoder.segment_path is not None:
                    self.assertTrue(encoder.segment_path.name.endswith(".mp4.part"))
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

    def test_segment_becomes_visible_only_after_atomic_finalization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            encoder = self.worker.SegmentEncoder(
                output_dir=output_dir,
                width=32,
                height=24,
                fps=Fraction(4, 1),
                generation=7,
                preferred_codec="libx264",
                segment_seconds=2.0,
            )
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            encoder.add_frame(frame, pts_ns=0)

            working = output_dir / "preview-g7-000000.mp4.part"
            final = output_dir / "preview-g7-000000.mp4"
            self.assertTrue(working.is_file())
            self.assertFalse(final.exists())

            event = encoder.finish()[0]
            self.assertEqual(Path(event["path"]), final)
            self.assertFalse(working.exists())
            self.assertTrue(final.is_file())
            with av.open(final) as container:
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

    def test_async_encoder_overlaps_append_and_preserves_events(self):
        started = threading.Event()
        release = threading.Event()
        expected = {
            "kind": "segment",
            "generation": 4,
            "sequence": 0,
            "start_ns": 0,
            "end_ns": 1,
            "path": "/tmp/preview.mp4",
            "codec": "h264_videotoolbox",
        }

        class SlowEncoder:
            active_codec = "h264_videotoolbox"
            active_options = {}

            def add_frame(self, frame, pts_ns):
                started.set()
                release.wait(timeout=2)
                return [expected]

            def finish(self):
                return []

            def discard(self):
                pass

        encoder = self.worker.AsyncSegmentEncoder(
            SlowEncoder(),
            max_pending_frames=2,
        )
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        begin = time.perf_counter()
        self.assertEqual(encoder.add_frame(frame, 0), [])
        self.assertLess(time.perf_counter() - begin, 0.1)
        self.assertTrue(started.wait(timeout=1))
        release.set()
        deadline = time.monotonic() + 1
        events = []
        while time.monotonic() < deadline and not events:
            events = encoder.poll_events()
            if not events:
                time.sleep(0.01)
        self.assertEqual(events, [expected])
        self.assertEqual(encoder.finish(), [])

    def test_async_encoder_retries_first_frame_with_startup_fallback(self):
        received = []

        class FailingPrimary:
            active_codec = "h264_videotoolbox"
            active_options = {}

            def add_frame(self, frame, pts_ns):
                raise RuntimeError("Cannot Encode")

            def finish(self):
                return []

            def discard(self):
                pass

        class WorkingFallback:
            active_codec = "libx264"
            active_options = {"preset": "ultrafast"}

            def add_frame(self, frame, pts_ns):
                received.append((frame.copy(), pts_ns))
                return []

            def finish(self):
                return []

            def discard(self):
                pass

        encoder = self.worker.AsyncSegmentEncoder(
            FailingPrimary(),
            startup_fallback_factory=WorkingFallback,
        )
        frame = np.full((2, 2, 3), 9, dtype=np.uint8)
        encoder.add_frame(frame, 123)
        self.assertEqual(encoder.finish(), [])
        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(received[0][0], frame)
        self.assertEqual(received[0][1], 123)
        self.assertEqual(encoder.active_codec, "libx264")

    @unittest.skipUnless(
        os.environ.get("LADA_PREVIEW_VIDEOTOOLBOX_RUNNER"),
        "Swift preview encoder runner is required",
    )
    def test_swift_pixel_buffer_pool_encoder_writes_bgr_frames(self):
        runner = os.environ["LADA_PREVIEW_VIDEOTOOLBOX_RUNNER"]
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder = self.worker.SwiftVideoToolboxSegmentEncoder(
                output_dir=Path(temp_dir),
                width=64,
                height=48,
                fps=Fraction(12, 1),
                generation=5,
                segment_seconds=2.0,
                runner_path=runner,
            )
            # Deliberately asymmetric BGR values expose channel swaps.
            frame = np.empty((48, 64, 3), dtype=np.uint8)
            frame[..., 0] = 12
            frame[..., 1] = 68
            frame[..., 2] = 210
            for index in range(12):
                self.assertEqual(
                    encoder.add_frame(
                        frame,
                        pts_ns=index * 1_000_000_000 // 12,
                    ),
                    [],
                )
            events = encoder.finish()

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["codec"], "h264_videotoolbox")
            output_path = Path(events[0]["path"])
            self.assertTrue(output_path.is_file())
            with av.open(output_path) as container:
                decoded = next(container.decode(video=0)).to_ndarray(format="bgr24")
            means = decoded.mean(axis=(0, 1))
            self.assertLess(means[0], means[1])
            self.assertLess(means[1], means[2])
            mse = np.mean(
                (
                    decoded.astype(np.float32)
                    - frame.astype(np.float32)
                ) ** 2
            )
            psnr = 10 * np.log10((255.0 ** 2) / mse)
            self.assertGreater(psnr, 40.0)
            self.assertFalse(
                (Path(temp_dir) / ".preview-frame-g5.bin").exists()
            )


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

    def test_restart_inside_one_process_reuses_loaded_models(self):
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

            session.generation = 9
            session.start_generation(42_000_000_000)

            loader.assert_called_once()
            self.assertEqual(len(restorers), 2)
            self.assertTrue(restorers[0].stopped)
            self.assertEqual(restorers[1].starts, [42_000_000_000])
            self.assertEqual(session.generation, 9)
            self.assertFalse(stale_path.exists())

    def test_command_stream_eof_stops_detached_preview_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                model_loader=lambda _config: object(),
                restorer_factory=lambda _config, _models: mock.Mock(),
            )

            session._read_commands(io.StringIO(""))

            command = session._next_command()
            self.assertIsNotNone(command)
            self.assertEqual(command.command, "stop")

    def test_video_eof_releases_worker_instead_of_waiting_for_seek(self):
        from lada.utils.threading_utils import EOF_MARKER

        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            event_condition = threading.Condition()
            restorers = []
            worker_module = self.worker

            class FakeRestorer:
                def __init__(self):
                    self.results = worker_module.queue.Queue()
                    self.stopped = False

                def start(self, start_ns=0):
                    del start_ns
                    self.results.put(EOF_MARKER)

                def stop(self):
                    self.stopped = True

                def get_frame_restoration_queue(self):
                    return self.results

            class FakeEncoder:
                segment_path = None

                def finish(self):
                    return []

                def discard(self):
                    pass

            def make_restorer(_config, _models):
                restorer = FakeRestorer()
                restorers.append(restorer)
                return restorer

            def record_event(kind, *, generation, **payload):
                with event_condition:
                    events.append(
                        {"kind": kind, "generation": generation, **payload}
                    )
                    event_condition.notify_all()

            metadata = SimpleNamespace(
                duration=120.0,
                video_fps_exact=Fraction(30, 1),
                video_width=32,
                video_height=24,
                time_base=Fraction(1, 30),
            )
            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                model_loader=lambda _config: {"metadata": metadata},
                restorer_factory=make_restorer,
                encoder_factory=lambda **_kwargs: FakeEncoder(),
            )

            def wait_for_event(kind, generation):
                deadline = time.monotonic() + 2
                with event_condition:
                    while not any(
                        event["kind"] == kind
                        and event["generation"] == generation
                        for event in events
                    ):
                        remaining = deadline - time.monotonic()
                        self.assertGreater(remaining, 0)
                        event_condition.wait(remaining)

            with mock.patch.object(self.worker, "emit_event", side_effect=record_event):
                run_thread = threading.Thread(target=session.run, daemon=True)
                run_thread.start()
                wait_for_event("ready", 0)
                wait_for_event("ended", 0)
                run_thread.join(timeout=2)

            self.assertFalse(run_thread.is_alive())
            self.assertEqual(len(restorers), 1)
            self.assertTrue(all(restorer.stopped for restorer in restorers))
            ready = next(event for event in events if event["kind"] == "ready")
            self.assertEqual(ready["segment_seconds"], 2.0)

    def test_initial_generation_stays_in_sync_with_controller(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                initial_generation=12,
                model_loader=lambda _config: object(),
                restorer_factory=lambda _config, _models: mock.Mock(),
            )

            self.assertEqual(session.generation, 12)

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

    def test_buffer_capacity_does_not_count_encoder_active_segment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                model_loader=lambda _config: object(),
                restorer_factory=lambda _config, _models: mock.Mock(),
            )
            session.generation = 7
            for sequence in range(3):
                path = Path(temp_dir) / f"preview-g7-{sequence:06d}.mp4"
                path.touch()
            active = Path(temp_dir) / "preview-g7-000003.mp4.part"
            active.touch()

            self.assertTrue(session.has_buffer_capacity())

            active.unlink()
            completed = Path(temp_dir) / "preview-g7-000003.mp4"
            completed.touch()
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

    def test_release_through_deletes_consumed_segments_and_refills_capacity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self.worker.PreviewSession(
                self.make_config(),
                Path(temp_dir),
                model_loader=lambda _config: object(),
                restorer_factory=lambda _config, _models: mock.Mock(),
            )
            session.generation = 4
            for sequence in range(4):
                (Path(temp_dir) / f"preview-g4-{sequence:06d}.mp4").touch()
            self.assertFalse(session.has_buffer_capacity())

            output = io.StringIO()
            with mock.patch.object(self.worker, "PROTOCOL_STREAM", output):
                session._apply_command(
                    self.worker.PreviewCommand.parse(
                        '{"command":"release_through","sequence":1}'
                    )
                )

            self.assertFalse((Path(temp_dir) / "preview-g4-000000.mp4").exists())
            self.assertFalse((Path(temp_dir) / "preview-g4-000001.mp4").exists())
            self.assertTrue((Path(temp_dir) / "preview-g4-000002.mp4").exists())
            self.assertTrue(session.has_buffer_capacity())
            self.assertEqual(
                json.loads(output.getvalue()),
                {"kind": "released", "generation": 4, "sequence": 1},
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
