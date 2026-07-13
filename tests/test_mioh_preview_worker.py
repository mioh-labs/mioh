import importlib.util
import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
