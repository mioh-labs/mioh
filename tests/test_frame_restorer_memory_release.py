import unittest
from unittest import mock

import torch

from lada.restorationpipeline.frame_restorer import FrameRestorer


class FakeClip:
    def __init__(self, frame_count: int):
        self.frame_count = frame_count

    def __len__(self):
        return self.frame_count


class FrameRestorerMemoryReleaseTests(unittest.TestCase):
    def test_completed_clip_releases_only_unused_memory_under_pressure(self):
        restorer = object.__new__(FrameRestorer)
        restorer.device = torch.device("mps")
        completed = FakeClip(0)
        active = FakeClip(1)
        clip_buffer = [completed, active]

        with (
            mock.patch(
                "lada.restorationpipeline.frame_restorer.release_mps_memory_if_needed",
                return_value=True,
            ) as release_mps,
            mock.patch(
                "lada.restorationpipeline.frame_restorer._release_darwin_malloc_cache",
                create=True,
            ) as release_malloc,
        ):
            restorer._collect_garbage(clip_buffer)

        self.assertEqual(clip_buffer, [active])
        release_mps.assert_called_once_with()
        release_malloc.assert_called_once_with()

    def test_completed_clip_keeps_caches_without_memory_pressure(self):
        restorer = object.__new__(FrameRestorer)
        restorer.device = torch.device("mps")
        clip_buffer = [FakeClip(0)]

        with (
            mock.patch(
                "lada.restorationpipeline.frame_restorer.release_mps_memory_if_needed",
                return_value=False,
            ) as release_mps,
            mock.patch(
                "lada.restorationpipeline.frame_restorer._release_darwin_malloc_cache",
                create=True,
            ) as release_malloc,
        ):
            restorer._collect_garbage(clip_buffer)

        self.assertEqual(clip_buffer, [])
        release_mps.assert_called_once_with()
        release_malloc.assert_not_called()

    def test_active_clip_does_not_trigger_memory_release(self):
        restorer = object.__new__(FrameRestorer)
        restorer.device = torch.device("mps")
        clip_buffer = [FakeClip(1)]

        with (
            mock.patch("gc.collect") as collect,
            mock.patch(
                "lada.restorationpipeline.frame_restorer.release_mps_memory_if_needed",
            ) as release_mps,
            mock.patch(
                "lada.restorationpipeline.frame_restorer._release_darwin_malloc_cache",
                create=True,
            ) as release_malloc,
        ):
            restorer._collect_garbage(clip_buffer)

        collect.assert_not_called()
        release_mps.assert_not_called()
        release_malloc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
