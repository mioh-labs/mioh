# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Independent quality-training stages for MiohRestorerV4-Q."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class V4LossWeights:
    candidate: float
    high_frequency: float
    base: float
    confidence: float
    confidence_regularization: float
    temporal: float
    temporal_acceleration: float
    gradient: float
    structural: float
    perceptual: float

    def interpolate(self, other: "V4LossWeights", amount: float) -> "V4LossWeights":
        amount = min(1.0, max(0.0, amount))
        return V4LossWeights(
            **{
                key: float(value + (getattr(other, key) - value) * amount)
                for key, value in asdict(self).items()
            }
        )


@dataclass(frozen=True)
class V4TrainingStage:
    stage_id: int
    name: str
    default_steps: int
    peak_learning_rate: float
    end_learning_rate: float
    loss: V4LossWeights
    train_texture: bool
    train_confidence: bool
    perceptual_every: int
    ema_decay: float
    spynet_alignment_weight: float = 0.0
    exact_motion_alignment_weight: float = 0.0
    exact_motion_probability: float = 0.0
    feature_distillation_weight: float = 0.0


QUALITY_V4_STAGES = (
    V4TrainingStage(
        stage_id=1,
        name="foundation",
        default_steps=10_000,
        peak_learning_rate=5e-5,
        end_learning_rate=1.5e-5,
        loss=V4LossWeights(
            candidate=1.0,
            high_frequency=0.0,
            base=0.15,
            confidence=0.0,
            confidence_regularization=0.0,
            temporal=0.05,
            temporal_acceleration=0.0,
            gradient=0.05,
            structural=0.0,
            perceptual=0.0,
        ),
        train_texture=False,
        train_confidence=False,
        perceptual_every=0,
        ema_decay=0.999,
        spynet_alignment_weight=0.02,
        exact_motion_alignment_weight=0.04,
        exact_motion_probability=0.35,
    ),
    V4TrainingStage(
        stage_id=2,
        name="faithful_reconstruction",
        default_steps=15_000,
        peak_learning_rate=3e-5,
        end_learning_rate=8e-6,
        loss=V4LossWeights(
            candidate=0.75,
            high_frequency=0.10,
            base=0.08,
            confidence=0.10,
            confidence_regularization=1e-4,
            temporal=0.10,
            temporal_acceleration=0.02,
            gradient=0.12,
            structural=0.05,
            perceptual=0.005,
        ),
        train_texture=True,
        train_confidence=True,
        perceptual_every=4,
        ema_decay=0.999,
        feature_distillation_weight=0.01,
    ),
    V4TrainingStage(
        stage_id=3,
        name="detail_recovery",
        default_steps=20_000,
        peak_learning_rate=2e-5,
        end_learning_rate=4e-6,
        loss=V4LossWeights(
            candidate=0.50,
            high_frequency=0.24,
            base=0.05,
            confidence=0.15,
            confidence_regularization=1e-4,
            temporal=0.15,
            temporal_acceleration=0.05,
            gradient=0.25,
            structural=0.10,
            perceptual=0.02,
        ),
        train_texture=True,
        train_confidence=True,
        perceptual_every=2,
        ema_decay=0.9995,
    ),
    V4TrainingStage(
        stage_id=4,
        name="temporal_consistency",
        default_steps=15_000,
        peak_learning_rate=1.2e-5,
        end_learning_rate=2e-6,
        loss=V4LossWeights(
            candidate=0.45,
            high_frequency=0.24,
            base=0.04,
            confidence=0.15,
            confidence_regularization=1e-4,
            temporal=0.30,
            temporal_acceleration=0.10,
            gradient=0.25,
            structural=0.10,
            perceptual=0.02,
        ),
        train_texture=True,
        train_confidence=True,
        perceptual_every=2,
        ema_decay=0.9995,
    ),
    V4TrainingStage(
        stage_id=5,
        name="fidelity_polish",
        default_steps=10_000,
        peak_learning_rate=5e-6,
        end_learning_rate=5e-7,
        loss=V4LossWeights(
            candidate=0.50,
            high_frequency=0.18,
            base=0.04,
            confidence=0.12,
            confidence_regularization=5e-5,
            temporal=0.25,
            temporal_acceleration=0.08,
            gradient=0.22,
            structural=0.08,
            perceptual=0.01,
        ),
        train_texture=True,
        train_confidence=True,
        perceptual_every=2,
        ema_decay=0.9995,
    ),
)


def stage_definition(identifier: int | str) -> V4TrainingStage:
    value = str(identifier).strip().lower().replace("-", "_")
    for stage in QUALITY_V4_STAGES:
        if value in (str(stage.stage_id), stage.name):
            return stage
    raise ValueError(f"unknown V4 training stage: {identifier}")


def previous_stage(stage: V4TrainingStage) -> V4TrainingStage | None:
    if stage.stage_id == 1:
        return None
    return QUALITY_V4_STAGES[stage.stage_id - 2]


def effective_loss_weights(
    stage: V4TrainingStage,
    local_step: int,
    *,
    transition_steps: int = 500,
) -> V4LossWeights:
    if local_step <= 0:
        raise ValueError("local training step must be positive")
    parent = previous_stage(stage)
    if parent is None or transition_steps <= 0:
        return stage.loss
    amount = local_step / transition_steps
    return parent.loss.interpolate(stage.loss, amount)


def stage_learning_rate(
    stage: V4TrainingStage,
    local_step: int,
    *,
    total_steps: int,
    warmup_steps: int = 500,
) -> float:
    """Warm up a fresh stage optimizer and cosine-decay within that stage."""

    if local_step <= 0 or total_steps <= 0:
        raise ValueError("local_step and total_steps must be positive")
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    if warmup_steps and local_step <= warmup_steps:
        start_fraction = 0.0 if stage.stage_id == 1 else 0.1
        progress = local_step / warmup_steps
        return stage.peak_learning_rate * (
            start_fraction + (1.0 - start_fraction) * progress
        )
    progress = (local_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return stage.end_learning_rate + 0.5 * (
        stage.peak_learning_rate - stage.end_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def schedule_record() -> list[dict[str, object]]:
    return [asdict(stage) for stage in QUALITY_V4_STAGES]
