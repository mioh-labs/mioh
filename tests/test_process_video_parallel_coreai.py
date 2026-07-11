import unittest
import tempfile
from unittest import mock

import process_video_parallel as pvp


def make_config(model_name: str) -> pvp.WorkerRuntimeConfig:
    return pvp.WorkerRuntimeConfig(
        device="mps",
        fp16=True,
        mps_memory_fraction=0.46,
        log_mps_memory=False,
        encoding_preset="hevc-apple-gpu-balanced",
        encoder=None,
        encoder_options=None,
        optimal_encoder_options=None,
        mp4_fast_start=False,
        mosaic_restoration_model=model_name,
        max_clip_length=180,
        mosaic_detection_model="v4-fast",
        detect_face_mosaics=False,
        lada_temp_dir=None,
        overwrite=False,
    )


class ProcessVideoParallelCoreAITests(unittest.TestCase):
    def test_t90_coreai_limits_parallel_workers_to_one(self):
        self.assertEqual(
            pvp.get_memory_safe_parallel_workers(
                "basicvsrpp-v1.2-coreai-t90", 4
            ),
            1,
        )

    def test_non_coreai_model_keeps_requested_parallel_workers(self):
        self.assertEqual(
            pvp.get_memory_safe_parallel_workers("basicvsrpp-v1.2", 4),
            4,
        )

    def test_t18_coreai_uses_padding_free_streaming_clip_length_by_default(self):
        self.assertEqual(
            pvp.get_effective_max_clip_length("basicvsrpp-v1.2-coreai", None),
            98,
        )

    def test_t36_coreai_uses_padding_free_streaming_clip_length_by_default(self):
        self.assertEqual(
            pvp.get_effective_max_clip_length("basicvsrpp-v1.2-coreai-t36", None),
            104,
        )

    def test_t90_coreai_uses_two_window_clip_length_by_default(self):
        self.assertEqual(
            pvp.get_effective_max_clip_length("basicvsrpp-v1.2-coreai-t90", None),
            178,
        )

    def test_non_coreai_model_keeps_legacy_default_clip_length(self):
        self.assertEqual(
            pvp.get_effective_max_clip_length("basicvsrpp-v1.2", None),
            180,
        )

    def test_explicit_clip_length_is_never_replaced(self):
        self.assertEqual(
            pvp.get_effective_max_clip_length("basicvsrpp-v1.2-coreai", 162),
            162,
        )

    def test_standard_model_remains_default_when_coreai_assets_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pvp.Path(temp_dir)
            python = root / "bin" / "python"
            python.parent.mkdir()
            python.touch()
            model = root / "basicvsrpp-t36.aimodel"
            model.mkdir()

            with (
                mock.patch.object(pvp, "COREAI_PYTHON", python),
                mock.patch.object(pvp, "COREAI_T36_MODEL_PATH", model),
            ):
                default_model = pvp.get_default_mosaic_restoration_model()

        self.assertEqual(default_model, "basicvsrpp-v1.2")

    def test_standard_model_is_default_without_t36_asset(self):
        with (
            mock.patch.object(pvp, "COREAI_PYTHON", pvp.Path("/missing/python")),
            mock.patch.object(pvp, "COREAI_T36_MODEL_PATH", pvp.Path("/missing/model.aimodel")),
        ):
            default_model = pvp.get_default_mosaic_restoration_model()

        self.assertEqual(default_model, "basicvsrpp-v1.2")

    def test_coreai_detection_is_default_when_runtime_and_asset_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pvp.Path(temp_dir)
            python = root / "bin" / "python"
            python.parent.mkdir()
            python.touch()
            model = root / "detect.aimodel"
            model.mkdir()

            with (
                mock.patch.object(pvp, "COREAI_PYTHON", python),
                mock.patch.object(pvp, "COREAI_V4_FAST_MODEL_PATH", model),
            ):
                default_model = pvp.get_default_mosaic_detection_model()

        self.assertEqual(default_model, "v4-fast-coreai")

    def test_torch_detection_is_default_without_coreai_asset(self):
        with (
            mock.patch.object(pvp, "COREAI_PYTHON", pvp.Path("/missing/python")),
            mock.patch.object(pvp, "COREAI_V4_FAST_MODEL_PATH", pvp.Path("/missing/detect.aimodel")),
        ):
            default_model = pvp.get_default_mosaic_detection_model()

        self.assertEqual(default_model, "v4-fast")

    def test_parser_uses_detected_default_restoration_model(self):
        with mock.patch.object(
            pvp,
            "get_default_mosaic_restoration_model",
            return_value="basicvsrpp-v1.2-coreai-t36",
        ):
            args = pvp.build_arg_parser().parse_args([])

        self.assertEqual(args.mosaic_restoration_model, "basicvsrpp-v1.2-coreai-t36")
        self.assertIsNone(args.max_clip_length)
        self.assertEqual(args.mosaic_detection_empty_lookahead, 10)

    def test_coreai_model_uses_coreai_virtualenv_for_worker(self):
        config = make_config("basicvsrpp-v1.2-coreai")

        with mock.patch.object(pvp, "COREAI_PYTHON", pvp.Path("/repo/.venv-coreai/bin/python")):
            cmd = pvp.build_lada_cli_command(config, pvp.Path("in.mp4"), pvp.Path("out.mp4"))

        self.assertEqual(cmd[:3], ["/repo/.venv-coreai/bin/python", "-m", "lada.cli.main"])

    def test_t36_coreai_model_uses_coreai_virtualenv_for_worker(self):
        config = make_config("basicvsrpp-v1.2-coreai-t36")

        with mock.patch.object(pvp, "COREAI_PYTHON", pvp.Path("/repo/.venv-coreai/bin/python")):
            cmd = pvp.build_lada_cli_command(config, pvp.Path("in.mp4"), pvp.Path("out.mp4"))

        self.assertEqual(cmd[:3], ["/repo/.venv-coreai/bin/python", "-m", "lada.cli.main"])

    def test_t90_coreai_model_uses_coreai_virtualenv_for_worker(self):
        config = make_config("basicvsrpp-v1.2-coreai-t90")

        with mock.patch.object(pvp, "COREAI_PYTHON", pvp.Path("/repo/.venv-coreai/bin/python")):
            cmd = pvp.build_lada_cli_command(config, pvp.Path("in.mp4"), pvp.Path("out.mp4"))

        self.assertEqual(cmd[:3], ["/repo/.venv-coreai/bin/python", "-m", "lada.cli.main"])

    def test_compiled_coreai_model_uses_coreai_virtualenv_for_worker(self):
        config = make_config("/models/basicvsrpp-t90.aimodelc")

        with mock.patch.object(pvp, "COREAI_PYTHON", pvp.Path("/repo/.venv-coreai/bin/python")):
            cmd = pvp.build_lada_cli_command(
                config,
                pvp.Path("in.mp4"),
                pvp.Path("out.mp4"),
            )

        self.assertEqual(
            cmd[:3],
            ["/repo/.venv-coreai/bin/python", "-m", "lada.cli.main"],
        )

    def test_coreai_detection_model_uses_coreai_virtualenv_for_worker(self):
        config = make_config("basicvsrpp-v1.2")
        config = pvp.WorkerRuntimeConfig(
            **{
                **config.__dict__,
                "mosaic_detection_model": "v4-fast-coreai",
            }
        )

        with mock.patch.object(pvp, "COREAI_PYTHON", pvp.Path("/repo/.venv-coreai/bin/python")):
            cmd = pvp.build_lada_cli_command(config, pvp.Path("in.mp4"), pvp.Path("out.mp4"))

        self.assertEqual(cmd[:3], ["/repo/.venv-coreai/bin/python", "-m", "lada.cli.main"])

    def test_coreai_enhancer_uses_coreai_virtualenv_for_worker(self):
        config = make_config("basicvsrpp-v1.2")
        config = pvp.WorkerRuntimeConfig(
            **{
                **config.__dict__,
                "restore_roi_enhancer": "realesrgan",
                "restore_roi_enhancer_model_path": "realesr-general-x4v3-coreai",
                "restore_roi_enhancer_strength": 1.0,
            }
        )

        with (
            mock.patch.object(
                pvp,
                "COREAI_PYTHON",
                pvp.Path("/repo/.venv-coreai/bin/python"),
            ),
            mock.patch.object(pvp.sys, "executable", "/current/python"),
        ):
            cmd = pvp.build_lada_cli_command(
                config,
                pvp.Path("in.mp4"),
                pvp.Path("out.mp4"),
            )

        self.assertEqual(
            cmd[:3],
            ["/repo/.venv-coreai/bin/python", "-m", "lada.cli.main"],
        )

    def test_x4plus_coreai_name_uses_coreai_virtualenv(self):
        with (
            mock.patch.object(
                pvp,
                "COREAI_PYTHON",
                pvp.Path("/repo/.venv-coreai/bin/python"),
            ),
            mock.patch.object(pvp.sys, "executable", "/current/python"),
        ):
            command = pvp.lada_cli_command_prefix(
                "basicvsrpp-v1.2",
                "v4-fast",
                "realesrgan-x4-coreai",
            )

        self.assertEqual(
            command,
            ["/repo/.venv-coreai/bin/python", "-m", "lada.cli.main"],
        )

    def test_non_coreai_model_keeps_current_interpreter_for_worker(self):
        config = make_config("basicvsrpp-v1.2")

        with mock.patch.object(pvp.sys, "executable", "/current/python"):
            cmd = pvp.build_lada_cli_command(config, pvp.Path("in.mp4"), pvp.Path("out.mp4"))

        self.assertEqual(cmd[:3], ["/current/python", "-m", "lada.cli.main"])


if __name__ == "__main__":
    unittest.main()
