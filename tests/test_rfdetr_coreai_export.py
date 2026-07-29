# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import torch

from scripts.apple import export_rfdetr_seg_coreai as exporter
from scripts.apple.rfdetr_coreai_kernels import (
    ATTENTION_HEADS,
    FEATURE_SIDE,
    POINTS_PER_HEAD,
    ms_deform_attn_reference,
)


def test_export_defaults_to_validated_fp32_asset():
    args = exporter.parse_args([])

    assert args.output == Path(
        "model_weights/rf-detr-seg-small-384-fp32.aimodel"
    )
    assert args.fp16 is False
    assert exporter.MODEL_CLASSES["large"] == "RFDETRSegLarge"


def test_fixed_deform_attention_samples_native_contiguous_layout():
    head_dim = 16
    value = torch.zeros(
        1,
        FEATURE_SIDE * FEATURE_SIDE,
        ATTENTION_HEADS,
        head_dim,
    )
    value[:, 0] = torch.arange(
        ATTENTION_HEADS * head_dim,
        dtype=torch.float32,
    ).reshape(ATTENTION_HEADS, head_dim)
    pixel_center = 0.5 / FEATURE_SIDE
    locations = torch.full(
        (1, 1, ATTENTION_HEADS, POINTS_PER_HEAD, 2),
        pixel_center,
    )
    weights = torch.zeros(1, 1, ATTENTION_HEADS, POINTS_PER_HEAD)
    weights[..., 0] = 1.0

    output = ms_deform_attn_reference(value, locations, weights)

    expected = value[:, 0].reshape(1, 1, ATTENTION_HEADS * head_dim)
    torch.testing.assert_close(output, expected)
