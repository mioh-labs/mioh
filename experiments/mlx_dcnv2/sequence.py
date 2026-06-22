"""Minimal MLX LADA BasicVSR++ sequence inference subset."""

from __future__ import annotations

import mlx.core as mx

from .feature_extract import feature_extract_forward
from .propagation import propagate_lada_branches_forward
from .reconstruction import reconstruction_forward
from .spynet import interpolate_bilinear_nchw_to_size, spynet_forward


def lada_sequence_forward(
    frames: mx.array,
    tensors: dict[str, object],
    *,
    feature_blocks: int = 5,
    backbone_blocks: int = 15,
    reconstruction_blocks: int = 5,
) -> mx.array:
    """Run the LADA-required BasicVSR++ inference path for a short sequence."""

    batch, frame_count, channels, height, width = frames.shape
    flat_frames = mx.reshape(frames, (batch * frame_count, channels, height, width))
    spatial = feature_extract_forward(
        flat_frames,
        tensors["feature_extract"],
        num_blocks=feature_blocks,
    )
    spatial = mx.reshape(spatial, (batch, frame_count, spatial.shape[1], height // 4, width // 4))

    downsampled = interpolate_bilinear_nchw_to_size(
        flat_frames,
        size=(height // 4, width // 4),
        align_corners=False,
    )
    downsampled = mx.reshape(downsampled, (batch, frame_count, channels, height // 4, width // 4))

    flows_forward_items: list[mx.array] = []
    flows_backward_items: list[mx.array] = []
    for idx in range(frame_count - 1):
        flows_backward_items.append(spynet_forward(downsampled[:, idx], downsampled[:, idx + 1], tensors["spynet"]))
        flows_forward_items.append(spynet_forward(downsampled[:, idx + 1], downsampled[:, idx], tensors["spynet"]))
    flows_backward = mx.stack(flows_backward_items, axis=1)
    flows_forward = mx.stack(flows_forward_items, axis=1)

    spatial_feats = [spatial[:, idx] for idx in range(frame_count)]
    branches = propagate_lada_branches_forward(
        spatial_feats,
        flows_forward,
        flows_backward,
        tensors["alignment"],
        tensors["backbones"],
        num_backbone_blocks=backbone_blocks,
    )

    outputs = []
    for idx in range(frame_count):
        reconstruction_input = mx.concatenate(
            [
                spatial[:, idx],
                branches["backward_1"][idx],
                branches["forward_1"][idx],
                branches["backward_2"][idx],
                branches["forward_2"][idx],
            ],
            axis=1,
        )
        outputs.append(
            reconstruction_forward(
                reconstruction_input,
                frames[:, idx],
                tensors["reconstruction"],
                num_blocks=reconstruction_blocks,
            )
        )
    return mx.stack(outputs, axis=1)
