import io
import json
import queue
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import process_video_parallel as pvp


class InteractiveBuffer(io.StringIO):
    def isatty(self):
        return True


class ProcessVideoParallelProgressTests(unittest.TestCase):
    def worker_config(self):
        return pvp.WorkerRuntimeConfig(
            device="cpu",
            fp16=False,
            mps_memory_fraction=None,
            log_mps_memory=False,
            encoding_preset=None,
            encoder=None,
            encoder_options=None,
            optimal_encoder_options=None,
            mp4_fast_start=False,
            mosaic_restoration_model="basicvsrpp-v1.2",
            max_clip_length=180,
            mosaic_detection_model="v4-fast",
            detect_face_mosaics=False,
            lada_temp_dir=None,
            overwrite=True,
        )

    def test_parse_progress_line_returns_percentage(self):
        self.assertEqual(
            pvp.parse_progress_line(
                "Processing video:  13%|█▎|Processed: 00:51 (5606f)"
            ),
            13.0,
        )
        self.assertIsNone(pvp.parse_progress_line("ordinary log message"))

    def test_app_protocol_keeps_worker_lanes_distinct(self):
        stream = io.StringIO()
        renderer = pvp.ParallelProgressRenderer(
            stream=stream,
            app_protocol=True,
            min_interval=0,
        )

        renderer.progress({
            "kind": "progress",
            "lane": "worker-101",
            "segment": 1,
            "text": "Processing video: 10%",
            "percent": 10.0,
        })
        renderer.progress({
            "kind": "progress",
            "lane": "worker-202",
            "segment": 2,
            "text": "Processing video: 20%",
            "percent": 20.0,
        })

        events = [
            json.loads(line.removeprefix(pvp.APP_PROGRESS_PREFIX))
            for line in stream.getvalue().splitlines()
        ]
        self.assertEqual([event["lane"] for event in events], ["worker-101", "worker-202"])
        self.assertEqual(set(renderer.active_lanes), {"worker-101", "worker-202"})

    def test_repeated_lane_updates_are_throttled(self):
        stream = io.StringIO()
        renderer = pvp.ParallelProgressRenderer(
            stream=stream,
            app_protocol=True,
            min_interval=0.25,
        )
        first = {
            "kind": "progress",
            "lane": "worker-101",
            "segment": 1,
            "text": "Processing video: 10%",
            "percent": 10.0,
        }
        second = {**first, "text": "Processing video: 11%", "percent": 11.0}

        with mock.patch.object(pvp.time, "monotonic", side_effect=[1.0, 1.1]):
            renderer.progress(first)
            renderer.progress(second)

        self.assertEqual(len(stream.getvalue().splitlines()), 1)
        self.assertEqual(renderer.active_lanes["worker-101"]["percent"], 11.0)

    def test_completion_releases_lane_and_preserves_message(self):
        stream = io.StringIO()
        renderer = pvp.ParallelProgressRenderer(
            stream=stream,
            app_protocol=True,
            min_interval=0,
        )
        renderer.progress({
            "kind": "progress",
            "lane": "worker-101",
            "segment": 1,
            "text": "Processing video: 100%",
            "percent": 100.0,
        })

        renderer.complete("worker-101", "[並列処理] セグメント #1 完了")

        events = [
            json.loads(line.removeprefix(pvp.APP_PROGRESS_PREFIX))
            for line in stream.getvalue().splitlines()
        ]
        self.assertEqual(events[-1]["kind"], "complete")
        self.assertEqual(events[-1]["text"], "[並列処理] セグメント #1 完了")
        self.assertNotIn("worker-101", renderer.active_lanes)

    def test_interactive_renderer_redraws_all_active_lanes(self):
        stream = InteractiveBuffer()
        renderer = pvp.ParallelProgressRenderer(
            stream=stream,
            app_protocol=False,
            min_interval=0,
        )
        renderer.progress({
            "kind": "progress",
            "lane": "worker-101",
            "segment": 1,
            "text": "Processing video: 10%",
            "percent": 10.0,
        })
        renderer.progress({
            "kind": "progress",
            "lane": "worker-202",
            "segment": 2,
            "text": "Processing video: 20%",
            "percent": 20.0,
        })

        output = stream.getvalue()
        self.assertIn("[segment 1] Processing video: 10%", output)
        self.assertIn("[segment 2] Processing video: 20%", output)
        self.assertIn("\x1b[", output)

    def test_worker_sends_progress_to_parent_without_printing_it(self):
        progress_queue = queue.Queue()
        process = mock.Mock()
        process.stdout = iter([
            "Preparing model\n",
            "Processing video:  13%|█▎|Processed: 00:51 (5606f)\n",
        ])
        process.wait.return_value = 0
        stdout = io.StringIO()

        with mock.patch.object(pvp.subprocess, "Popen", return_value=process):
            with mock.patch.object(pvp, "aggressive_memory_cleanup_for_device"):
                with mock.patch.object(pvp.time, "sleep"):
                    with redirect_stdout(stdout):
                        result = pvp.process_segment_worker(
                            (3, Path("input.mp4"), Path("output.mp4")),
                            self.worker_config(),
                            progress_queue,
                        )

        events = []
        while not progress_queue.empty():
            events.append(progress_queue.get_nowait())
        event = next(event for event in events if event["kind"] == "progress")
        self.assertEqual(event["kind"], "progress")
        self.assertEqual(event["segment"], 3)
        self.assertEqual(event["percent"], 13.0)
        self.assertNotIn("Processing video", stdout.getvalue())
        self.assertEqual(result["status"], "success")

    def test_thread_workers_get_distinct_progress_lanes_in_one_process(self):
        lanes = []
        for segment, thread_id in [(1, 101), (2, 202)]:
            progress_queue = queue.Queue()
            process = mock.Mock()
            process.stdout = iter(["Processing video: 10%\n"])
            process.wait.return_value = 0
            with mock.patch.object(pvp.subprocess, "Popen", return_value=process):
                with mock.patch.object(pvp, "aggressive_memory_cleanup_for_device"):
                    with mock.patch.object(pvp.time, "sleep"):
                        with mock.patch.object(pvp.os, "getpid", return_value=999):
                            with mock.patch.object(
                                pvp.threading,
                                "get_ident",
                                return_value=thread_id,
                            ):
                                pvp.process_segment_worker(
                                    (
                                        segment,
                                        Path(f"input-{segment}.mp4"),
                                        Path(f"output-{segment}.mp4"),
                                    ),
                                    self.worker_config(),
                                    progress_queue,
                                )
            events = []
            while not progress_queue.empty():
                events.append(progress_queue.get_nowait())
            lanes.append(next(event["lane"] for event in events if event["kind"] == "progress"))

        self.assertNotEqual(lanes[0], lanes[1])


if __name__ == "__main__":
    unittest.main()
