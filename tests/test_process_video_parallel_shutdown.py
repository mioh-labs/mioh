import os
import unittest
from unittest import mock

import process_video_parallel as pvp


class ProcessVideoParallelShutdownTests(unittest.TestCase):
    def tearDown(self):
        pvp._shutdown_requested = False
        pvp._force_shutdown = False
        pvp._set_active_executor(None)

    def test_second_signal_exits_immediately_and_shuts_down_executor(self):
        fake_child = mock.Mock()
        fake_process = mock.Mock()
        fake_process.children.return_value = [fake_child]
        fake_executor = mock.Mock()
        pvp._set_active_executor(fake_executor)
        pvp._shutdown_requested = True

        with mock.patch.object(pvp.psutil, "Process", return_value=fake_process):
            with mock.patch.object(pvp.os, "_exit", side_effect=SystemExit(130)) as exit_mock:
                with self.assertRaises(SystemExit):
                    pvp.signal_handler(pvp.signal.SIGINT, None)

        self.assertTrue(pvp._force_shutdown)
        fake_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        fake_child.terminate.assert_called_once()
        exit_mock.assert_called_once_with(130)

    def test_first_signal_exits_immediately(self):
        fake_child = mock.Mock()
        fake_process = mock.Mock()
        fake_process.children.return_value = [fake_child]
        fake_executor = mock.Mock()

        pvp._set_active_executor(fake_executor)

        with mock.patch.object(pvp.psutil, "Process", return_value=fake_process):
            with mock.patch.object(pvp.os, "_exit", side_effect=SystemExit(130)) as exit_mock:
                with self.assertRaises(SystemExit):
                    pvp.signal_handler(pvp.signal.SIGINT, None)

        fake_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        fake_child.terminate.assert_called_once()
        exit_mock.assert_called_once_with(130)

    def test_cleanup_resources_does_not_stop_resource_tracker_directly(self):
        with mock.patch.object(pvp.gc, "collect") as gc_collect:
            with mock.patch("torch.cuda.is_available", return_value=False):
                with mock.patch("torch.backends.mps.is_available", return_value=False):
                    with mock.patch.object(pvp.resource_tracker._resource_tracker, "_stop") as stop_tracker:
                        pvp.cleanup_resources()

        gc_collect.assert_called_once()
        stop_tracker.assert_not_called()

    def test_shutdown_executor_uses_public_api_only(self):
        class FakeExecutor:
            def __init__(self):
                self.calls = []

            def shutdown(self, wait, cancel_futures=False):
                self.calls.append((wait, cancel_futures))

        executor = FakeExecutor()
        pvp._shutdown_executor(executor, wait=True)
        pvp._shutdown_executor(executor, wait=False, cancel_futures=True)

        self.assertEqual(executor.calls, [(True, False), (False, True)])

    def test_force_shutdown_kills_descendants_and_exits(self):
        fake_child = mock.Mock()
        fake_process = mock.Mock()
        fake_process.children.return_value = [fake_child]
        fake_executor = mock.Mock()

        pvp._set_active_executor(fake_executor)
        pvp._force_shutdown = True

        with mock.patch.object(pvp.psutil, "Process", return_value=fake_process):
            with mock.patch.object(pvp.os, "_exit", side_effect=SystemExit(130)) as exit_mock:
                with self.assertRaises(SystemExit):
                    pvp.signal_handler(pvp.signal.SIGINT, None)

        fake_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        fake_child.terminate.assert_called_once()
        exit_mock.assert_called_once_with(130)

    def test_build_worker_env_sets_pythonwarnings_filter(self):
        config = pvp.WorkerRuntimeConfig(
            device="mps",
            fp16=True,
            mps_memory_fraction=0.5,
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
            overwrite=False,
        )

        env = pvp.build_worker_env(config)

        self.assertIn("PYTHONWARNINGS", env)
        self.assertIn("ignore::UserWarning:multiprocessing.resource_tracker", env["PYTHONWARNINGS"])

    def test_build_worker_env_respects_existing_mps_fallback_override(self):
        config = pvp.WorkerRuntimeConfig(
            device="mps",
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
            overwrite=False,
        )

        with mock.patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            env = pvp.build_worker_env(config)

        self.assertEqual(env["PYTORCH_ENABLE_MPS_FALLBACK"], "0")


if __name__ == "__main__":
    unittest.main()
