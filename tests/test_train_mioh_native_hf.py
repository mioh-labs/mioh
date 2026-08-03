# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import runpy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from lada.models.mioh_restorer.model_native_hf import (
    MiohNativeHF512,
    NativeHF512Config,
    build_mioh_native_hf512,
)


TRAINING = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "training"
        / "train-mioh-native-hf.py"
    )
)


def _trainable_names(model: MiohNativeHF512) -> set[str]:
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def test_native_hf_stage_freezing_and_detail_learning_rate_contract() -> None:
    configure_stage = TRAINING["configure_stage"]
    optimizer_groups = TRAINING["optimizer_groups"]

    hf = build_mioh_native_hf512()
    weights = configure_stage(hf, "hf-bootstrap", initialize_stage=True)
    assert weights is not None
    assert weights.innovation == 0.25
    assert weights.innovation_span == 0.10
    assert weights.innovation_zero == 0.10
    assert weights.fidelity_guard == 0.10
    assert weights.reconstruction == 1.0
    assert weights.residual == 0
    assert weights.high_frequency == 0.50
    assert weights.gradient == 0.15
    assert weights.wavelet == 0.15
    assert weights.observation == 0
    hf_names = _trainable_names(hf)
    assert hf_names
    assert not any(name.startswith("encoder.") for name in hf_names)
    assert not any(name.startswith("decoder.alignment.") for name in hf_names)
    assert not any(name.startswith("decoder.confidence_head.") for name in hf_names)
    assert any(name.startswith("decoder.detail_skip.") for name in hf_names)
    assert torch.count_nonzero(hf.decoder.alignment.corr_8.offset_bias) == 0
    assert torch.count_nonzero(hf.decoder.alignment.corr_4.offset_bias) == 0
    assert torch.count_nonzero(hf.decoder.alignment.corr_2.offset_bias) == 0
    assert torch.count_nonzero(hf.decoder.alignment.phase_bias) == 0
    torch.testing.assert_close(
        hf.decoder.confidence_head[-1].bias,
        torch.full_like(hf.decoder.confidence_head[-1].bias, 4.0),
    )
    groups = optimizer_groups(hf, "hf-bootstrap", 2e-4)
    assert [group["lr_scale"] for group in groups] == [1.0, 0.015]
    assert groups[1]["lr"] == pytest.approx(3e-6)
    assert groups[1]["weight_decay"] == 0.0

    joint = MiohNativeHF512()
    configure_stage(joint, "joint", initialize_stage=False)
    joint_names = _trainable_names(joint)
    assert joint_names
    assert not any(name.startswith("encoder.") for name in joint_names)
    assert {
        name
        for name in joint_names
        if name.startswith("decoder.alignment.")
    } == {
        name
        for name, _parameter in joint.named_parameters()
        if name.startswith("decoder.alignment.pair_gate.")
    }
    assert any(name.startswith("decoder.confidence_head.") for name in joint_names)


def test_native_hf_block_size_bucket_boundaries() -> None:
    bucket = TRAINING["block_size_bucket"]
    assert bucket(8) == "block_small_le8"
    assert bucket(9) == "block_medium_9_16"
    assert bucket(16) == "block_medium_9_16"
    assert bucket(17) == "block_large_ge17"


def test_native_hf_epoch_sampler_resumes_exact_permutation_suffix() -> None:
    sampler_type = TRAINING["EpochRandomSampler"]
    sampler = sampler_type(list(range(17)), seed=20260802)
    sampler.set_epoch(3)
    complete = list(sampler)
    sampler.set_epoch(3, start_index=7)
    assert list(sampler) == complete[7:]
    assert len(sampler) == 10

    sampler.set_epoch(4)
    assert list(sampler) != complete


def test_native_hf_known_motion_curriculum_spans_easy_to_full_reach() -> None:
    curriculum = TRAINING["known_motion_curriculum"]
    assert curriculum(step=1, steps=500, minimum=4, maximum=20) == 4
    assert curriculum(step=500, steps=500, minimum=4, maximum=20) == 20
    middle = curriculum(step=250, steps=500, minimum=4, maximum=20)
    assert 11.9 < middle < 12.1


def test_native_hf_initialization_accepts_only_predecessor_ema(
    tmp_path: Path,
) -> None:
    initialize = TRAINING["initialize_from_checkpoint"]
    checkpoint_format = TRAINING["CHECKPOINT_FORMAT"]
    recipe = TRAINING["NATIVE_HF_INITIALIZATION_RECIPE"]
    model_seed = TRAINING["NATIVE_HF_MODEL_INITIALIZATION_SEED"]
    identities = {"fixture": {"path": "fixture", "size": 1, "modified_ns": 2}}
    parent = build_mioh_native_hf512(NativeHF512Config())
    with torch.no_grad():
        parent.decoder.alignment.phase_bias.fill_(0.375)
    payload = {
        "format": checkpoint_format,
        "initialization_recipe": recipe,
        "model_initialization_seed": model_seed,
        "config": asdict(parent.config),
        "inputs": identities,
        "stage": "hf-bootstrap",
        "loss_weights": asdict(TRAINING["stage_loss_weights"]("hf-bootstrap")),
        "ema_state_dict": parent.state_dict(),
    }
    path = tmp_path / "alignment.pth"
    torch.save(payload, path)

    target = build_mioh_native_hf512(NativeHF512Config())
    source_stage = initialize(path, target, identities, "joint")
    assert source_stage == "hf-bootstrap"
    torch.testing.assert_close(
        target.decoder.alignment.phase_bias,
        torch.full_like(target.decoder.alignment.phase_bias, 0.375),
    )

    with pytest.raises(ValueError, match="cannot initialize"):
        initialize(path, MiohNativeHF512(), identities, "hf-bootstrap")


def test_native_hf_fixed_model_seed_is_independent_of_training_seed() -> None:
    torch.manual_seed(1)
    first = build_mioh_native_hf512()
    torch.manual_seed(999)
    second = build_mioh_native_hf512()

    first_alignment = {
        name: value
        for name, value in first.state_dict().items()
        if name.startswith("encoder.") or name.startswith("decoder.alignment.")
    }
    second_alignment = {
        name: value
        for name, value in second.state_dict().items()
        if name.startswith("encoder.") or name.startswith("decoder.alignment.")
    }
    assert first_alignment.keys() == second_alignment.keys()
    for name in first_alignment:
        torch.testing.assert_close(first_alignment[name], second_alignment[name])


def test_native_hf_checkpoint_rejects_old_initialization_contract(
    tmp_path: Path,
) -> None:
    compatible = TRAINING["compatible_checkpoint_payload"]
    model = build_mioh_native_hf512()
    identities = {"fixture": {"path": "fixture", "size": 1, "modified_ns": 2}}
    path = tmp_path / "old.pth"
    torch.save(
        {
            "format": "mioh-native-hf-512-v4",
            "config": asdict(model.config),
            "inputs": identities,
        },
        path,
    )
    with pytest.raises(ValueError, match="not a Native-HF 512 checkpoint"):
        compatible(path, model, identities)
