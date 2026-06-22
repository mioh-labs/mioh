import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from experiments.mlx_dcnv2.sequence import lada_sequence_forward
from tests.test_mlx_dcnv2_branch_propagation import _torch_lada_branches_forward
from tests.test_mlx_dcnv2_feature_extract import _torch_feature_extract_forward, _random_feature_extract_tensors
from tests.test_mlx_dcnv2_reconstruction import _torch_reconstruction_forward, _random_reconstruction_tensors
from tests.test_mlx_dcnv2_spynet import _random_spynet_tensors, _torch_spynet_forward
from tests.test_mlx_dcnv2_branch_propagation import _random_alignment_tensors, _random_backbone_tensors


class MLXSequenceTests(unittest.TestCase):
    def test_lada_sequence_forward_matches_pytorch_reference(self):
        rng = np.random.default_rng(1201)
        mid_channels = 4
        deform_groups = 2
        feature_blocks = 2
        backbone_blocks = 2
        reconstruction_blocks = 2
        frames = rng.normal(size=(1, 2, 3, 128, 128)).astype(np.float32)
        tensors = {
            "feature_extract": _random_feature_extract_tensors(rng, mid_channels, feature_blocks),
            "spynet": _random_spynet_tensors(rng),
            "alignment": {name: _random_alignment_tensors(rng, mid_channels, deform_groups) for name in _MODULE_NAMES},
            "backbones": {
                "backward_1": _random_backbone_tensors(rng, 2 * mid_channels, mid_channels, backbone_blocks),
                "forward_1": _random_backbone_tensors(rng, 3 * mid_channels, mid_channels, backbone_blocks),
                "backward_2": _random_backbone_tensors(rng, 4 * mid_channels, mid_channels, backbone_blocks),
                "forward_2": _random_backbone_tensors(rng, 5 * mid_channels, mid_channels, backbone_blocks),
            },
            "reconstruction": _random_reconstruction_tensors(rng, mid_channels, reconstruction_blocks),
        }

        expected = _torch_lada_sequence_forward(
            torch.from_numpy(frames),
            _to_torch_nested(tensors),
            feature_blocks=feature_blocks,
            backbone_blocks=backbone_blocks,
            reconstruction_blocks=reconstruction_blocks,
        ).numpy()
        actual = np.array(
            lada_sequence_forward(
                mx.array(frames),
                _to_mlx_nested(tensors),
                feature_blocks=feature_blocks,
                backbone_blocks=backbone_blocks,
                reconstruction_blocks=reconstruction_blocks,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)


def _torch_lada_sequence_forward(frames, tensors, *, feature_blocks, backbone_blocks, reconstruction_blocks):
    batch, frame_count, channels, height, width = frames.shape
    flat_frames = frames.reshape(batch * frame_count, channels, height, width)
    spatial = _torch_feature_extract_forward(flat_frames, tensors["feature_extract"], feature_blocks)
    spatial = spatial.reshape(batch, frame_count, -1, height // 4, width // 4)

    downsampled = F.interpolate(flat_frames, size=(height // 4, width // 4), mode="bilinear", align_corners=False)
    downsampled = downsampled.reshape(batch, frame_count, channels, height // 4, width // 4)
    flows_backward = []
    flows_forward = []
    for idx in range(frame_count - 1):
        flows_backward.append(_torch_spynet_forward(downsampled[:, idx], downsampled[:, idx + 1], tensors["spynet"]))
        flows_forward.append(_torch_spynet_forward(downsampled[:, idx + 1], downsampled[:, idx], tensors["spynet"]))
    flows_backward = torch.stack(flows_backward, dim=1)
    flows_forward = torch.stack(flows_forward, dim=1)

    branches = _torch_lada_branches_forward(
        spatial,
        flows_forward,
        flows_backward,
        tensors["alignment"],
        tensors["backbones"],
        backbone_blocks,
    )
    outputs = []
    for idx in range(frame_count):
        reconstruction_input = torch.cat(
            [
                spatial[:, idx],
                branches["backward_1"][:, idx],
                branches["forward_1"][:, idx],
                branches["backward_2"][:, idx],
                branches["forward_2"][:, idx],
            ],
            dim=1,
        )
        outputs.append(
            _torch_reconstruction_forward(
                reconstruction_input,
                frames[:, idx],
                tensors["reconstruction"],
                reconstruction_blocks,
            )
        )
    return torch.stack(outputs, dim=1)


def _to_torch_nested(value):
    if isinstance(value, dict):
        return {key: _to_torch_nested(item) for key, item in value.items()}
    return torch.from_numpy(value)


def _to_mlx_nested(value):
    if isinstance(value, dict):
        return {key: _to_mlx_nested(item) for key, item in value.items()}
    return mx.array(value)


_MODULE_NAMES = ["backward_1", "forward_1", "backward_2", "forward_2"]


if __name__ == "__main__":
    unittest.main()
