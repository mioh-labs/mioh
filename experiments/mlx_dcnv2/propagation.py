"""MLX BasicVSR++ propagation building blocks."""

from __future__ import annotations

import mlx.core as mx

from .alignment import second_order_deformable_alignment_forward
from .backbone import residual_blocks_with_input_conv_forward
from .flow_warp import flow_warp


def propagation_step_forward(
    feat_current: mx.array,
    feat_prop: mx.array,
    feat_n2: mx.array,
    flow_n1: mx.array,
    flow_n2: mx.array,
    alignment_tensors: dict[str, mx.array],
    backbone_tensors: dict[str, mx.array],
    *,
    num_backbone_blocks: int,
) -> mx.array:
    """Run one BasicVSR++ propagation step after the first frame.

    This covers the `backward_1`/`forward_1` shape where the backbone input is
    `[feat_current, aligned_feat_prop]`. Later branches add previous branch
    features before `aligned_feat_prop`; those will build on this primitive.
    """

    cond_n1 = flow_warp(feat_prop, mx.transpose(flow_n1, (0, 2, 3, 1)), padding_mode="zeros")
    cond_n2 = flow_warp(feat_n2, mx.transpose(flow_n2, (0, 2, 3, 1)), padding_mode="zeros")
    cond = mx.concatenate([cond_n1, feat_current, cond_n2], axis=1)
    aligned = second_order_deformable_alignment_forward(
        mx.concatenate([feat_prop, feat_n2], axis=1),
        cond,
        flow_n1,
        flow_n2,
        alignment_tensors,
    )
    backbone_input = mx.concatenate([feat_current, aligned], axis=1)
    return aligned + residual_blocks_with_input_conv_forward(
        backbone_input,
        backbone_tensors,
        num_blocks=num_backbone_blocks,
    )


def propagate_first_order_branch_forward(
    spatial_feats: list[mx.array],
    flows: mx.array,
    alignment_tensors: dict[str, mx.array],
    backbone_tensors: dict[str, mx.array],
    *,
    num_backbone_blocks: int,
) -> list[mx.array]:
    """Propagate one BasicVSR++ first-iteration branch in forward order.

    This matches the branch shape used by `forward_1` and `backward_1` after
    the caller has already chosen frame order and matching flow order.
    """

    return propagate_branch_forward(
        spatial_feats,
        flows,
        alignment_tensors,
        backbone_tensors,
        num_backbone_blocks=num_backbone_blocks,
        previous_branch_feats=[],
    )


def propagate_branch_forward(
    spatial_feats: list[mx.array],
    flows: mx.array,
    alignment_tensors: dict[str, mx.array],
    backbone_tensors: dict[str, mx.array],
    *,
    num_backbone_blocks: int,
    previous_branch_feats: list[list[mx.array]] | None = None,
) -> list[mx.array]:
    """Propagate one BasicVSR++ branch in caller-provided temporal order."""

    previous_branch_feats = previous_branch_feats or []
    if not spatial_feats:
        return []
    feat_prop = mx.zeros_like(spatial_feats[0])
    outputs: list[mx.array] = []
    for idx, feat_current in enumerate(spatial_feats):
        if idx > 0:
            flow_n1 = flows[:, idx - 1]
            cond_n1 = flow_warp(feat_prop, mx.transpose(flow_n1, (0, 2, 3, 1)), padding_mode="zeros")
            feat_n2 = mx.zeros_like(feat_prop)
            flow_n2 = mx.zeros_like(flow_n1)
            cond_n2 = mx.zeros_like(cond_n1)
            if idx > 1:
                feat_n2 = outputs[-2]
                flow_n2 = flows[:, idx - 2]
                flow_n2 = flow_n1 + flow_warp(flow_n2, mx.transpose(flow_n1, (0, 2, 3, 1)), padding_mode="zeros")
                cond_n2 = flow_warp(feat_n2, mx.transpose(flow_n2, (0, 2, 3, 1)), padding_mode="zeros")
            cond = mx.concatenate([cond_n1, feat_current, cond_n2], axis=1)
            feat_prop = second_order_deformable_alignment_forward(
                mx.concatenate([feat_prop, feat_n2], axis=1),
                cond,
                flow_n1,
                flow_n2,
                alignment_tensors,
            )

        backbone_input = mx.concatenate(
            [feat_current] + [branch[idx] for branch in previous_branch_feats] + [feat_prop],
            axis=1,
        )
        feat_prop = feat_prop + residual_blocks_with_input_conv_forward(
            backbone_input,
            backbone_tensors,
            num_blocks=num_backbone_blocks,
        )
        outputs.append(feat_prop)
    return outputs


def propagate_lada_branches_forward(
    spatial_feats: list[mx.array],
    flows_forward: mx.array,
    flows_backward: mx.array,
    alignment_tensors: dict[str, dict[str, mx.array]],
    backbone_tensors: dict[str, dict[str, mx.array]],
    *,
    num_backbone_blocks: int,
) -> dict[str, list[mx.array]]:
    """Run the four LADA BasicVSR++ propagation branches.

    This intentionally implements only the inference branch order used by
    LADA's BasicVSR++ model: backward_1, forward_1, backward_2, forward_2.
    """

    backward_1_reversed = propagate_branch_forward(
        spatial_feats[::-1],
        _reverse_time(flows_backward),
        alignment_tensors["backward_1"],
        backbone_tensors["backward_1"],
        num_backbone_blocks=num_backbone_blocks,
    )
    backward_1 = backward_1_reversed[::-1]

    forward_1 = propagate_branch_forward(
        spatial_feats,
        flows_forward,
        alignment_tensors["forward_1"],
        backbone_tensors["forward_1"],
        num_backbone_blocks=num_backbone_blocks,
        previous_branch_feats=[backward_1],
    )

    backward_2_reversed = propagate_branch_forward(
        spatial_feats[::-1],
        _reverse_time(flows_backward),
        alignment_tensors["backward_2"],
        backbone_tensors["backward_2"],
        num_backbone_blocks=num_backbone_blocks,
        previous_branch_feats=[backward_1[::-1], forward_1[::-1]],
    )
    backward_2 = backward_2_reversed[::-1]

    forward_2 = propagate_branch_forward(
        spatial_feats,
        flows_forward,
        alignment_tensors["forward_2"],
        backbone_tensors["forward_2"],
        num_backbone_blocks=num_backbone_blocks,
        previous_branch_feats=[backward_1, forward_1, backward_2],
    )

    return {
        "backward_1": backward_1,
        "forward_1": forward_1,
        "backward_2": backward_2,
        "forward_2": forward_2,
    }


def _reverse_time(flows: mx.array) -> mx.array:
    return mx.take(flows, mx.array(list(range(flows.shape[1] - 1, -1, -1))), axis=1)
