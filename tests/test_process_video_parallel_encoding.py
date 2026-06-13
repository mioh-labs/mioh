import unittest
from unittest import mock

import process_video_parallel as pvp


class ProcessVideoParallelEncodingTests(unittest.TestCase):
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
