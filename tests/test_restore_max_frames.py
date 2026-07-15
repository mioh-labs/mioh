import unittest

import torch

from lada.cli.main import setup_argparser
from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from lada.restorationpipeline.frame_restorer import FrameRestorer
import process_video_parallel as pvp


class RestoreMaxFramesTests(unittest.TestCase):
    def test_lada_cli_accepts_unlimited_restore_max_frames(self):
        parser = setup_argparser()

        args = parser.parse_args([
            "--input", "in.mp4",
            "--restore-max-frames", "-1",
        ])

        self.assertEqual(args.restore_max_frames, -1)

    def test_parallel_command_passes_restore_max_frames(self):
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
            restore_max_frames=-1,
            mosaic_detection_model="v4-fast",
            detect_face_mosaics=False,
            lada_temp_dir=None,
            overwrite=False,
        )

        cmd = pvp.build_lada_cli_command(config, pvp.Path("in.mp4"), pvp.Path("out.mp4"))

        self.assertIn("--restore-max-frames", cmd)
        self.assertIn("-1", cmd)
        self.assertIn("--restore-temporal-overlap", cmd)
        self.assertIn("8", cmd)
        self.assertIn("--enable-crossfade", cmd)

    def test_lada_cli_accepts_temporal_overlap_and_crossfade_toggle(self):
        parser = setup_argparser()

        args = parser.parse_args([
            "--input", "in.mp4",
            "--restore-temporal-overlap", "15",
            "--disable-crossfade",
        ])

        self.assertEqual(args.restore_temporal_overlap, 15)
        self.assertFalse(args.restore_crossfade)

    def test_explicit_restore_max_frames_overrides_mps_adaptive_chunking(self):
        restorer = FrameRestorer.__new__(FrameRestorer)
        restorer.mosaic_restoration_model_name = "basicvsrpp-v1.2"
        restorer.device = torch.device("mps")
        restorer.restore_max_frames = -1
        restorer._adaptive_restore_chunk_frames = 16
        restorer._mps_adaptive_profile = {
            "min_chunk": 6,
            "low_mem_gb": 3.5,
            "mid_mem_gb": 5.0,
            "high_mem_gb": 8.0,
        }

        model = BasicvsrppMosaicRestorer(object(), torch.device("cpu"), fp16=False)
        seen = []

        def fake_restore(
            images,
            max_frames=-1,
            temporal_overlap=8,
            enable_crossfade=True,
        ):
            seen.append((max_frames, temporal_overlap, enable_crossfade))
            return images

        model.restore = fake_restore
        restorer.mosaic_restoration_model = model
        restorer.restore_temporal_overlap = 15
        restorer.restore_crossfade = False

        frames = [torch.zeros((2, 2, 3), dtype=torch.uint8)]
        restored = restorer._restore_clip_frames(frames)

        self.assertIs(restored, frames)
        self.assertEqual(seen, [(-1, 15, False)])


if __name__ == "__main__":
    unittest.main()
