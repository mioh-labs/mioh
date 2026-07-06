import unittest

import torch

from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer


class ChunkOffsetModel:
    def __init__(self, step: float = 90.0):
        self.calls = 0
        self.step = step

    def __call__(self, inputs):
        value = self.calls * self.step / 255.0
        self.calls += 1
        return torch.full_like(inputs, value)


class BasicVSRPPChunkOverlapTests(unittest.TestCase):
    def test_max_frame_chunks_are_crossfaded_at_boundaries(self):
        model = ChunkOffsetModel()
        restorer = BasicvsrppMosaicRestorer(model, torch.device("cpu"), fp16=False)
        video = [torch.zeros((4, 4, 3), dtype=torch.uint8) for _ in range(6)]

        restored = restorer.restore(video, max_frames=4)

        values = [int(frame[0, 0, 0]) for frame in restored]
        self.assertEqual(values, [0, 0, 30, 60, 90, 90])


if __name__ == "__main__":
    unittest.main()
