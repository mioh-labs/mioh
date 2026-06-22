import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from experiments.mlx_dcnv2.propagation import propagation_step_forward


class MLXPropagationTests(unittest.TestCase):
    def test_backward_1_propagation_step_matches_pytorch(self):
        rng = np.random.default_rng(456)
        mid_channels = 4
        deform_groups = 2
        num_blocks = 2
        feat_current = rng.normal(size=(1, mid_channels, 5, 6)).astype(np.float32)
        feat_prop = rng.normal(size=(1, mid_channels, 5, 6)).astype(np.float32)
        feat_n2 = np.zeros_like(feat_prop)
        flow_n1 = (rng.normal(size=(1, 2, 5, 6)) * 0.2).astype(np.float32)
        flow_n2 = np.zeros_like(flow_n1)
        alignment_tensors = _random_alignment_tensors(rng, mid_channels, deform_groups)
        backbone_tensors = _random_backbone_tensors(rng, 2 * mid_channels, mid_channels, num_blocks)

        expected = _torch_propagation_step(
            torch.from_numpy(feat_current),
            torch.from_numpy(feat_prop),
            torch.from_numpy(feat_n2),
            torch.from_numpy(flow_n1),
            torch.from_numpy(flow_n2),
            {name: torch.from_numpy(value) for name, value in alignment_tensors.items()},
            {name: torch.from_numpy(value) for name, value in backbone_tensors.items()},
            num_blocks,
        ).numpy()
        actual = np.array(
            propagation_step_forward(
                mx.array(feat_current),
                mx.array(feat_prop),
                mx.array(feat_n2),
                mx.array(flow_n1),
                mx.array(flow_n2),
                {name: mx.array(value) for name, value in alignment_tensors.items()},
                {name: mx.array(value) for name, value in backbone_tensors.items()},
                num_backbone_blocks=num_blocks,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)


def _torch_propagation_step(
    feat_current,
    feat_prop,
    feat_n2,
    flow_n1,
    flow_n2,
    alignment_tensors,
    backbone_tensors,
    num_blocks,
):
    cond_n1 = _torch_flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
    cond_n2 = _torch_flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
    x = torch.cat([feat_prop, feat_n2], dim=1)
    offset, mask = _torch_alignment_offset_mask(cond, flow_n1, flow_n2, alignment_tensors)
    aligned = torchvision.ops.deform_conv2d(
        x,
        offset,
        alignment_tensors["weight"],
        alignment_tensors["bias"],
        padding=(1, 1),
        mask=mask,
    )
    backbone_input = torch.cat([feat_current, aligned], dim=1)
    return aligned + _torch_backbone_forward(backbone_input, backbone_tensors, num_blocks)


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


if __name__ == "__main__":
    unittest.main()
