# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import torch

from scripts.apple import export_rfdetr_seg_coreml as exporter


def test_defaults_target_jasna_v6_fp32_coreml_package():
    args = exporter.parse_args([])

    assert args.variant == "medium"
    assert args.resolution == 576
    assert args.fp16 is False
    assert args.output == Path(
        "model_weights/rfdetr-v6-576-fp32.mlpackage"
    )


def test_fixed_proposal_grid_matches_expected_pixel_centers():
    memory = torch.zeros(1, 4, 2)

    output_memory, proposals = exporter._fixed_encoder_output_proposals(
        memory,
        spatial_shapes=[(2, 2)],
        unsigmoid=False,
    )

    torch.testing.assert_close(output_memory, memory)
    assert proposals.shape == (1, 4, 4)
    torch.testing.assert_close(
        proposals[0, :, :2],
        torch.tensor(
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
        ),
    )
    torch.testing.assert_close(
        proposals[0, :, 2:],
        torch.full((4, 2), 0.05),
    )


def test_error_metrics_report_exact_identity():
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3).numpy()

    assert exporter._error_metrics(value, value) == {
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "rmse": 0.0,
    }
