import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import process_video_parallel as pvp


def make_worker_config() -> pvp.WorkerRuntimeConfig:
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
        mosaic_restoration_model="basicvsrpp-v1.2",
        max_clip_length=180,
        mosaic_detection_model="v4-fast",
        detect_face_mosaics=False,
        lada_temp_dir=None,
        overwrite=False,
    )


class ProcessVideoParallelEncodingTests(unittest.TestCase):
    def test_additional_options_override_base_and_keep_order(self):
        merged = pvp.merge_encoder_options(
            "-q:v 55 -pix_fmt yuv420p -realtime 0",
            "-pix_fmt yuv420p10le -profile:v main10",
        )

        self.assertEqual(
            pvp.parse_encoder_options(merged),
            [
                ("-q:v", "55"),
                ("-pix_fmt", "yuv420p10le"),
                ("-realtime", "0"),
                ("-profile:v", "main10"),
            ],
        )

    def test_parser_supports_quotes_flags_and_negative_values(self):
        self.assertEqual(
            pvp.parse_encoder_options(
                '-metadata "title=My Video" -qmin -1 -fast'
            ),
            [
                ("-metadata", "title=My Video"),
                ("-qmin", "-1"),
                ("-fast", None),
            ],
        )

    def test_empty_additional_options_preserve_base(self):
        base = "-crf 18 -preset slow"

        self.assertEqual(
            pvp.parse_encoder_options(pvp.merge_encoder_options(base, "")),
            pvp.parse_encoder_options(base),
        )

    def test_parser_rejects_unbalanced_quotes(self):
        with self.assertRaisesRegex(ValueError, "引用符"):
            pvp.parse_encoder_options('-metadata "unterminated')

    def test_parser_rejects_value_before_option_name(self):
        with self.assertRaisesRegex(ValueError, "オプション名"):
            pvp.parse_encoder_options("orphan -crf 18")

    def test_preset_options_are_overridden_by_additional_values(self):
        config = replace(
            make_worker_config(),
            encoder_options="-q:v 70 -profile:v main10",
            optimal_encoder_options="-b:v 99M",
        )

        encoder, options = pvp.resolve_worker_encoding(config)
        parsed = pvp.parse_encoder_options(options)

        self.assertEqual(encoder, "hevc_videotoolbox")
        self.assertIn(("-q:v", "70"), parsed)
        self.assertIn(("-profile:v", "main10"), parsed)
        self.assertNotIn(("-b:v", "99M"), parsed)

    def test_preset_command_uses_resolved_encoder_and_options(self):
        config = replace(make_worker_config(), encoder_options="-q:v 70")

        cmd = pvp.build_lada_cli_command(
            config,
            pvp.Path("in.mp4"),
            pvp.Path("out.mp4"),
        )

        self.assertNotIn("--encoding-preset", cmd)
        self.assertEqual(
            cmd[cmd.index("--encoder") + 1],
            "hevc_videotoolbox",
        )
        final_options = cmd[cmd.index("--encoder-options") + 1]
        self.assertIn(
            ("-q:v", "70"),
            pvp.parse_encoder_options(final_options),
        )

    def test_automatic_and_custom_modes_use_resolved_final_options(self):
        automatic = replace(
            make_worker_config(),
            encoding_preset=None,
            optimal_encoder_options="-q:v 72",
        )
        custom = replace(
            automatic,
            encoder="libx264",
            optimal_encoder_options="-crf 19",
        )

        self.assertEqual(
            pvp.resolve_worker_encoding(automatic),
            ("hevc_videotoolbox", "-q:v 72"),
        )
        self.assertEqual(
            pvp.resolve_worker_encoding(custom),
            ("libx264", "-crf 19"),
        )

    def test_pre_fps_uses_automatic_options_without_user_overrides(self):
        source = Path(pvp.__file__).read_text()
        start = source.index(
            "# pre_fps変換が有効な場合（ローカル変数を使用）"
        )
        split_block = source[
            start:source.index("self.stats['total_segments']", start)
        ]

        self.assertIn(
            "encoder_options=self.intermediate_encoder_options",
            split_block,
        )
        self.assertNotIn(
            "encoder_options=self.optimal_encoder_options",
            split_block,
        )

    def test_uses_lada_apple_preset_when_mps_has_no_custom_encoding(self):
        args = mock.Mock(
            device="mps",
            encoding_preset=None,
            encoder=None,
            encoder_options=None,
        )

        with mock.patch.object(pvp, "get_default_preset_name", return_value="hevc-apple-gpu-balanced"):
            preset = pvp.get_lada_encoding_preset(args, optimal_encoder_options="")

        self.assertEqual(preset, "hevc-apple-gpu-balanced")

    def test_keeps_custom_mps_encoder_options_instead_of_lada_default_preset(self):
        args = mock.Mock(
            device="mps",
            encoding_preset=None,
            encoder=None,
            encoder_options=None,
        )

        with mock.patch.object(pvp, "get_default_preset_name", return_value="hevc-apple-gpu-balanced"):
            preset = pvp.get_lada_encoding_preset(args, optimal_encoder_options="-q:v 55")

        self.assertIsNone(preset)

    def test_non_mps_keeps_lada_cli_default_when_not_explicitly_set(self):
        args = mock.Mock(
            device="cpu",
            encoding_preset=None,
            encoder=None,
            encoder_options=None,
        )

        preset = pvp.get_lada_encoding_preset(args, optimal_encoder_options="")

        self.assertIsNone(preset)


if __name__ == "__main__":
    unittest.main()
