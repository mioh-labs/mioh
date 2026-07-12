import unittest
from unittest import mock

import process_video_parallel as pvp


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
