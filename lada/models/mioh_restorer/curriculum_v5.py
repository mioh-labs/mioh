# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Independent quality stages for MiohRestorer V5."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V5LossWeights:
    reconstruction: float
    candidate: float
    base: float
    high_frequency: float
    wavelet: float
    gradient: float
    perceptual: float
    temporal: float
    confidence: float
    confidence_regularization: float


@dataclass(frozen=True)
class V5TrainingStage:
    stage_id: int
    name: str
    default_steps: int
    learning_rate: float
    end_learning_rate: float
    loss: V5LossWeights
    exact_motion_weight: float = 0.0
    natural_motion_weight: float = 0.0
    feature_distillation_weight: float = 0.0


V5_STAGES = (
    V5TrainingStage(
        1,
        "known-motion-alignment",
        8_000,
        2e-4,
        5e-5,
        V5LossWeights(0.25, 0.25, 0.05, 0.02, 0.0, 0.02, 0.0, 0.0, 0.1, 1e-3),
        exact_motion_weight=1.0,
    ),
    V5TrainingStage(
        2,
        "natural-motion-alignment",
        10_000,
        1.5e-4,
        4e-5,
        V5LossWeights(0.5, 0.25, 0.08, 0.04, 0.0, 0.03, 0.0, 0.0, 0.15, 1e-3),
        natural_motion_weight=0.2,
        feature_distillation_weight=0.03,
    ),
    V5TrainingStage(
        3,
        "faithful-structure",
        15_000,
        1e-4,
        3e-5,
        V5LossWeights(1.0, 0.35, 0.12, 0.08, 0.03, 0.05, 0.0, 0.0, 0.2, 5e-4),
    ),
    V5TrainingStage(
        4,
        "detail-recovery",
        20_000,
        8e-5,
        2e-5,
        V5LossWeights(1.0, 0.3, 0.08, 0.28, 0.22, 0.14, 0.03, 0.0, 0.15, 2e-4),
    ),
    V5TrainingStage(
        5,
        "flow-aligned-temporal",
        15_000,
        6e-5,
        1.5e-5,
        V5LossWeights(1.0, 0.25, 0.06, 0.25, 0.18, 0.12, 0.02, 0.2, 0.15, 2e-4),
    ),
    V5TrainingStage(
        6,
        "fidelity-polish",
        10_000,
        3e-5,
        5e-6,
        V5LossWeights(1.0, 0.2, 0.08, 0.2, 0.14, 0.1, 0.01, 0.1, 0.1, 1e-4),
    ),
)


def stage_definition(value: int | str) -> V5TrainingStage:
    for stage in V5_STAGES:
        if value == stage.stage_id or str(value).replace("_", "-") == stage.name:
            return stage
    raise ValueError(f"unknown V5 training stage: {value}")


def previous_stage(stage: V5TrainingStage) -> V5TrainingStage | None:
    return V5_STAGES[stage.stage_id - 2] if stage.stage_id > 1 else None


def stage_learning_rate(
    stage: V5TrainingStage,
    local_step: int,
    *,
    total_steps: int,
    warmup_steps: int = 500,
) -> float:
    if local_step <= 0 or total_steps <= 0:
        raise ValueError("step counts must be positive")
    if local_step <= warmup_steps:
        return stage.learning_rate * local_step / warmup_steps
    progress = min(
        max((local_step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
        1.0,
    )
    return stage.learning_rate + (stage.end_learning_rate - stage.learning_rate) * progress
