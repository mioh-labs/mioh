# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Quality-first curriculum for the hybrid Core AI/Metal V5-HQ model."""

from __future__ import annotations

from dataclasses import dataclass

from .curriculum_v5 import V5LossWeights


@dataclass(frozen=True)
class V5HQTrainingStage:
    stage_id: int
    name: str
    default_steps: int
    learning_rate: float
    end_learning_rate: float
    loss: V5LossWeights
    train_backbone: bool
    train_spynet: bool
    boundary_temporal_weight: float = 0.0


V5_HQ_STAGES = (
    V5HQTrainingStage(
        1,
        "refiner-bootstrap",
        5_000,
        2e-4,
        4e-5,
        V5LossWeights(1.0, 0.35, 0.12, 0.12, 0.05, 0.08, 0.0, 0.0, 0.2, 5e-4),
        False,
        False,
    ),
    V5HQTrainingStage(
        2,
        "deformable-attention",
        10_000,
        1e-4,
        2e-5,
        V5LossWeights(1.0, 0.3, 0.10, 0.28, 0.20, 0.14, 0.03, 0.0, 0.18, 2e-4),
        False,
        False,
    ),
    V5HQTrainingStage(
        3,
        "recurrent-finetune",
        20_000,
        5e-5,
        1e-5,
        V5LossWeights(1.0, 0.25, 0.08, 0.30, 0.22, 0.15, 0.03, 0.0, 0.15, 2e-4),
        True,
        False,
    ),
    V5HQTrainingStage(
        4,
        "flow-finetune",
        10_000,
        2e-5,
        5e-6,
        V5LossWeights(1.0, 0.25, 0.08, 0.28, 0.20, 0.14, 0.02, 0.0, 0.15, 2e-4),
        True,
        True,
        0.12,
    ),
    V5HQTrainingStage(
        5,
        "long-temporal",
        15_000,
        2e-5,
        4e-6,
        V5LossWeights(1.0, 0.2, 0.06, 0.26, 0.18, 0.12, 0.02, 0.2, 0.12, 1e-4),
        True,
        True,
        0.10,
    ),
    V5HQTrainingStage(
        6,
        "fidelity-polish",
        10_000,
        1e-5,
        2e-6,
        V5LossWeights(1.0, 0.2, 0.08, 0.22, 0.14, 0.10, 0.01, 0.1, 0.1, 1e-4),
        True,
        True,
        0.06,
    ),
)


def hq_stage_definition(stage_id: int) -> V5HQTrainingStage:
    if not 1 <= stage_id <= len(V5_HQ_STAGES):
        raise ValueError(f"invalid V5-HQ stage: {stage_id}")
    return V5_HQ_STAGES[stage_id - 1]


def hq_learning_rate(
    stage: V5HQTrainingStage,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if step <= warmup_steps:
        return stage.learning_rate * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return stage.learning_rate + progress * (
        stage.end_learning_rate - stage.learning_rate
    )
