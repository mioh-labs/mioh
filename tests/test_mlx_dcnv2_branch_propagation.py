import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from experiments.mlx_dcnv2.propagation import (
    propagate_branch_forward,
    propagate_first_order_branch_forward,
    propagate_lada_branches_forward,
)


class MLXBranchPropagationTests(unittest.TestCase):
    def test_first_order_branch_forward_matches_pytorch(self):
        rng = np.random.default_rng(789)
        mid_channels = 4
        deform_groups = 2
        num_blocks = 2
        frame_count = 3
        spatial = rng.normal(size=(1, frame_count, mid_channels, 5, 6)).astype(np.float32)
        flows = (rng.normal(size=(1, frame_count - 1, 2, 5, 6)) * 0.2).astype(np.float32)
        alignment_tensors = _random_alignment_tensors(rng, mid_channels, deform_groups)
        backbone_tensors = _random_backbone_tensors(rng, 2 * mid_channels, mid_channels, num_blocks)

        expected = _torch_branch_forward(
            torch.from_numpy(spatial),
            torch.from_numpy(flows),
            {name: torch.from_numpy(value) for name, value in alignment_tensors.items()},
            {name: torch.from_numpy(value) for name, value in backbone_tensors.items()},
            num_blocks,
        )
        actual = propagate_first_order_branch_forward(
            [mx.array(spatial[:, i]) for i in range(frame_count)],
            mx.array(flows),
            {name: mx.array(value) for name, value in alignment_tensors.items()},
            {name: mx.array(value) for name, value in backbone_tensors.items()},
            num_backbone_blocks=num_blocks,
        )

        self.assertEqual(len(actual), frame_count)
        np.testing.assert_allclose(np.stack([np.array(x) for x in actual], axis=1), expected.numpy(), rtol=1e-4, atol=5e-4)

    def test_lada_four_branch_propagation_matches_pytorch_reference(self):
        rng = np.random.default_rng(791)
        mid_channels = 4
        deform_groups = 2
        num_blocks = 2
        frame_count = 3
        spatial = rng.normal(size=(1, frame_count, mid_channels, 5, 6)).astype(np.float32)
        flows_forward = (rng.normal(size=(1, frame_count - 1, 2, 5, 6)) * 0.2).astype(np.float32)
        flows_backward = (rng.normal(size=(1, frame_count - 1, 2, 5, 6)) * 0.2).astype(np.float32)
        alignment = {name: _random_alignment_tensors(rng, mid_channels, deform_groups) for name in _MODULE_NAMES}
        backbones = {
            "backward_1": _random_backbone_tensors(rng, 2 * mid_channels, mid_channels, num_blocks),
            "forward_1": _random_backbone_tensors(rng, 3 * mid_channels, mid_channels, num_blocks),
            "backward_2": _random_backbone_tensors(rng, 4 * mid_channels, mid_channels, num_blocks),
            "forward_2": _random_backbone_tensors(rng, 5 * mid_channels, mid_channels, num_blocks),
        }

        expected = _torch_lada_branches_forward(
            torch.from_numpy(spatial),
            torch.from_numpy(flows_forward),
            torch.from_numpy(flows_backward),
            {name: {k: torch.from_numpy(v) for k, v in tensors.items()} for name, tensors in alignment.items()},
            {name: {k: torch.from_numpy(v) for k, v in tensors.items()} for name, tensors in backbones.items()},
            num_blocks,
        )
        actual = propagate_lada_branches_forward(
            [mx.array(spatial[:, i]) for i in range(frame_count)],
            mx.array(flows_forward),
            mx.array(flows_backward),
            {name: {k: mx.array(v) for k, v in tensors.items()} for name, tensors in alignment.items()},
            {name: {k: mx.array(v) for k, v in tensors.items()} for name, tensors in backbones.items()},
            num_backbone_blocks=num_blocks,
        )

        self.assertEqual(set(actual), set(_MODULE_NAMES))
        for name in _MODULE_NAMES:
            np.testing.assert_allclose(
                np.stack([np.array(x) for x in actual[name]], axis=1),
                expected[name].numpy(),
                rtol=1e-4,
                atol=5e-4,
            )

    def test_branch_forward_with_previous_features_matches_pytorch(self):
        rng = np.random.default_rng(790)
        mid_channels = 4
        deform_groups = 2
        num_blocks = 2
        frame_count = 3
        previous_branch_count = 2
        spatial = rng.normal(size=(1, frame_count, mid_channels, 5, 6)).astype(np.float32)
        previous_branches = [
            rng.normal(size=(1, frame_count, mid_channels, 5, 6)).astype(np.float32)
            for _ in range(previous_branch_count)
        ]
        flows = (rng.normal(size=(1, frame_count - 1, 2, 5, 6)) * 0.2).astype(np.float32)
        alignment_tensors = _random_alignment_tensors(rng, mid_channels, deform_groups)
        backbone_tensors = _random_backbone_tensors(
            rng,
            (2 + previous_branch_count) * mid_channels,
            mid_channels,
            num_blocks,
        )

        expected = _torch_branch_forward(
            torch.from_numpy(spatial),
            torch.from_numpy(flows),
            {name: torch.from_numpy(value) for name, value in alignment_tensors.items()},
            {name: torch.from_numpy(value) for name, value in backbone_tensors.items()},
            num_blocks,
            previous_branch_feats=[
                [torch.from_numpy(branch[:, i]) for i in range(frame_count)]
                for branch in previous_branches
            ],
        )
        actual = propagate_branch_forward(
            [mx.array(spatial[:, i]) for i in range(frame_count)],
            mx.array(flows),
            {name: mx.array(value) for name, value in alignment_tensors.items()},
            {name: mx.array(value) for name, value in backbone_tensors.items()},
            num_backbone_blocks=num_blocks,
            previous_branch_feats=[
                [mx.array(branch[:, i]) for i in range(frame_count)]
                for branch in previous_branches
            ],
        )

        self.assertEqual(len(actual), frame_count)
        np.testing.assert_allclose(np.stack([np.array(x) for x in actual], axis=1), expected.numpy(), rtol=1e-4, atol=5e-4)


def _torch_branch_forward(
    spatial,
    flows,
    alignment_tensors,
    backbone_tensors,
    num_blocks,
    *,
    previous_branch_feats=None,
):
    previous_branch_feats = previous_branch_feats or []
    outputs = []
    feat_prop = torch.zeros_like(spatial[:, 0])
    for idx in range(spatial.shape[1]):
        feat_current = spatial[:, idx]
        if idx > 0:
            flow_n1 = flows[:, idx - 1]
            cond_n1 = _torch_flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
            feat_n2 = torch.zeros_like(feat_prop)
            flow_n2 = torch.zeros_like(flow_n1)
            cond_n2 = torch.zeros_like(cond_n1)
            if idx > 1:
                feat_n2 = outputs[-2]
                flow_n2 = flows[:, idx - 2]
                flow_n2 = flow_n1 + _torch_flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                cond_n2 = _torch_flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
            cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
            x = torch.cat([feat_prop, feat_n2], dim=1)
            offset, mask = _torch_alignment_offset_mask(cond, flow_n1, flow_n2, alignment_tensors)
            feat_prop = torchvision.ops.deform_conv2d(
                x,
                offset,
                alignment_tensors["weight"],
                alignment_tensors["bias"],
                padding=(1, 1),
                mask=mask,
            )
        backbone_input = torch.cat(
            [feat_current] + [branch[idx] for branch in previous_branch_feats] + [feat_prop],
            dim=1,
        )
        feat_prop = feat_prop + _torch_backbone_forward(backbone_input, backbone_tensors, num_blocks)
        outputs.append(feat_prop)
    return torch.stack(outputs, dim=1)


def _torch_lada_branches_forward(spatial, flows_forward, flows_backward, alignment, backbones, num_blocks):
    backward_1 = _torch_branch_forward(
        torch.flip(spatial, dims=[1]),
        torch.flip(flows_backward, dims=[1]),
        alignment["backward_1"],
        backbones["backward_1"],
        num_blocks,
    ).flip(1)
    forward_1 = _torch_branch_forward(
        spatial,
        flows_forward,
        alignment["forward_1"],
        backbones["forward_1"],
        num_blocks,
        previous_branch_feats=[_sequence_to_list(backward_1)],
    )
    backward_2 = _torch_branch_forward(
        torch.flip(spatial, dims=[1]),
        torch.flip(flows_backward, dims=[1]),
        alignment["backward_2"],
        backbones["backward_2"],
        num_blocks,
        previous_branch_feats=[_sequence_to_list(backward_1.flip(1)), _sequence_to_list(forward_1.flip(1))],
    ).flip(1)
    forward_2 = _torch_branch_forward(
        spatial,
        flows_forward,
        alignment["forward_2"],
        backbones["forward_2"],
        num_blocks,
        previous_branch_feats=[_sequence_to_list(backward_1), _sequence_to_list(forward_1), _sequence_to_list(backward_2)],
    )
    return {
        "backward_1": backward_1,
        "forward_1": forward_1,
        "backward_2": backward_2,
        "forward_2": forward_2,
    }


def _sequence_to_list(sequence):
    return [sequence[:, i] for i in range(sequence.shape[1])]


def _torch_flow_warp(x, flow):
    _, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x.dtype),
        torch.arange(0, w, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(x, grid_flow, mode="bilinear", padding_mode="zeros", align_corners=True)


def _torch_alignment_offset_mask(extra_feat, flow_1, flow_2, tensors):
    feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
    for layer in (0, 2, 4):
        feat = F.conv2d(feat, tensors[f"conv_offset.{layer}.weight"], tensors[f"conv_offset.{layer}.bias"], padding=1)
        feat = F.leaky_relu(feat, negative_slope=0.1)
    feat = F.conv2d(feat, tensors["conv_offset.6.weight"], tensors["conv_offset.6.bias"], padding=1)
    o1, o2, mask = torch.chunk(feat, 3, dim=1)
    offset = 10 * torch.tanh(torch.cat((o1, o2), dim=1))
    offset1, offset2 = torch.chunk(offset, 2, dim=1)
    offset1 = offset1 + flow_1.flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
    offset2 = offset2 + flow_2.flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
    return torch.cat([offset1, offset2], dim=1), torch.sigmoid(mask)


def _torch_backbone_forward(x, tensors, num_blocks):
    out = F.conv2d(x, tensors["main.0.weight"], tensors["main.0.bias"], padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    for block_index in range(num_blocks):
        identity = out
        out = F.conv2d(
            out,
            tensors[f"main.2.{block_index}.conv1.weight"],
            tensors[f"main.2.{block_index}.conv1.bias"],
            padding=1,
        )
        out = F.relu(out)
        out = F.conv2d(
            out,
            tensors[f"main.2.{block_index}.conv2.weight"],
            tensors[f"main.2.{block_index}.conv2.bias"],
            padding=1,
        )
        out = identity + out
    return out


def _random_alignment_tensors(rng, mid_channels, deform_groups):
    scale = 0.02
    return {
        "weight": (rng.normal(size=(mid_channels, 2 * mid_channels, 3, 3)) * scale).astype(np.float32),
        "bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "conv_offset.0.weight": (rng.normal(size=(mid_channels, 3 * mid_channels + 4, 3, 3)) * scale).astype(np.float32),
        "conv_offset.0.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "conv_offset.2.weight": (rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale).astype(np.float32),
        "conv_offset.2.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "conv_offset.4.weight": (rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale).astype(np.float32),
        "conv_offset.4.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "conv_offset.6.weight": (rng.normal(size=(27 * deform_groups, mid_channels, 3, 3)) * scale).astype(np.float32),
        "conv_offset.6.bias": (rng.normal(size=(27 * deform_groups,)) * scale).astype(np.float32),
    }


def _random_backbone_tensors(rng, in_channels, mid_channels, num_blocks):
    scale = 0.02
    tensors = {
        "main.0.weight": (rng.normal(size=(mid_channels, in_channels, 3, 3)) * scale).astype(np.float32),
        "main.0.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
    }
    for block_index in range(num_blocks):
        tensors[f"main.2.{block_index}.conv1.weight"] = (
            rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale
        ).astype(np.float32)
        tensors[f"main.2.{block_index}.conv1.bias"] = (rng.normal(size=(mid_channels,)) * scale).astype(np.float32)
        tensors[f"main.2.{block_index}.conv2.weight"] = (
            rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale
        ).astype(np.float32)
        tensors[f"main.2.{block_index}.conv2.bias"] = (rng.normal(size=(mid_channels,)) * scale).astype(np.float32)
    return tensors


_MODULE_NAMES = ["backward_1", "forward_1", "backward_2", "forward_2"]


if __name__ == "__main__":
    unittest.main()
