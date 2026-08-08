# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch
from mmengine.config import Config

from lada.models.basicvsrpp import register_all_modules
from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet
from lada.models.basicvsrpp.recoverable_hf_dataset import (
    phase_block_average_mosaic,
)
from lada.models.basicvsrpp.mmagic.data_sample import DataSample
from lada.models.basicvsrpp.mmagic.registry import MODELS
from lada.models.basicvsrpp.mmagic.roi_loss import (
    KnownGridMosaicConsistencyLoss,
    ROIPixelLoss,
    _known_phase_block_average,
)


TAIL_MODULES = (
    'reconstruction',
    'upsample1',
    'upsample2',
    'conv_hr',
    'conv_last',
)


def test_faithful_hf_freezes_everything_except_reconstruction_tail():
    register_all_modules()
    model = BasicVSRPlusPlusGanNet(
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,
        trainable_modules=TAIL_MODULES,
    )

    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.split('.', 1)[0] in TAIL_MODULES for name in trainable)
    assert sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    ) == 887_747

    model.train()
    assert model.reconstruction.training
    assert not model.spynet.training
    assert not model.feat_extract.training
    assert not model.deform_align.training
    assert not model.backbone.training


def test_roi_pixel_loss_ignores_error_outside_mask():
    loss = ROIPixelLoss(loss_weight=1.0, mask_dilation=0)
    target = torch.zeros(1, 3, 5, 5)
    prediction = target.clone()
    prediction[..., 0, 0] = 1.0
    mask = torch.zeros(1, 1, 5, 5)
    mask[..., 2, 2] = 1.0

    assert torch.isclose(loss(prediction, target, mask), torch.tensor(1e-6))

    prediction[..., 2, 2] = 1.0
    assert loss(prediction, target, mask) > 0.99


def test_known_grid_mosaic_consistency_uses_only_complete_roi_cells():
    block_size = 4
    phase = (1, 2)
    prediction = torch.full((1, 1, 3, 10, 10), 102.0 / 255.0)
    averaged, valid = _known_phase_block_average(
        prediction[0, 0], block_size=block_size, phase=phase
    )
    observation = (torch.round(averaged * 255.0) / 255.0).view_as(prediction)
    mask = torch.ones(1, 1, 1, 10, 10)
    phases = torch.tensor([[phase]], dtype=torch.int64)
    block_sizes = torch.tensor([block_size], dtype=torch.int64)
    weights = torch.ones(1)
    loss = KnownGridMosaicConsistencyLoss(loss_weight=1.0)

    assert torch.count_nonzero(valid) == 64
    torch.testing.assert_close(
        loss(prediction, observation, mask, phases, block_sizes, weights),
        torch.tensor(0.0),
    )

    # The x=0 strip belongs to a cell crossing the crop edge and is excluded.
    edge_only = prediction.clone()
    edge_only[..., :, :, 0] += 32.0 / 255.0
    torch.testing.assert_close(
        loss(edge_only, observation, mask, phases, block_sizes, weights),
        torch.tensor(0.0),
    )

    # A single non-ROI pixel excludes its complete 4x4 measurement cell.
    partial_mask = mask.clone()
    partial_mask[..., 2, 1] = 0.0
    excluded_cell_error = prediction.clone()
    excluded_cell_error[..., 2:6, 1:5] += 8.0 / 255.0
    torch.testing.assert_close(
        loss(
            excluded_cell_error,
            observation,
            partial_mask,
            phases,
            block_sizes,
            weights,
        ),
        torch.tensor(0.0),
    )


def test_known_grid_mosaic_consistency_backpropagates_mean_violation():
    block_size = 4
    phase = (0, 0)
    baseline = torch.full((1, 1, 3, 8, 8), 102.0 / 255.0)
    observation = baseline.clone()
    prediction = (baseline + 2.0 / 255.0).requires_grad_()
    mask = torch.ones(1, 1, 1, 8, 8)
    phases = torch.tensor([[phase]], dtype=torch.int64)
    block_sizes = torch.tensor([block_size], dtype=torch.int64)
    loss = KnownGridMosaicConsistencyLoss(loss_weight=0.2)

    value = loss(
        prediction,
        observation,
        mask,
        phases,
        block_sizes,
        torch.ones(1),
    )
    assert value > 0
    value.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0

    disabled = loss(
        prediction.detach(),
        observation,
        mask,
        phases,
        block_sizes,
        torch.zeros(1),
    )
    torch.testing.assert_close(disabled, torch.tensor(0.0))


def test_known_grid_mosaic_consistency_accepts_numpy_uint8_quantization():
    rng = np.random.default_rng(20260808)
    height, width = 32, 33
    image = rng.integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )
    prediction = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).float().div(255.0).view(1, 1, 3, height, width)
    mask = torch.ones(1, 1, 1, height, width)
    loss = KnownGridMosaicConsistencyLoss(loss_weight=1.0)

    for block_size in (6, 8, 10, 12):
        phase = (block_size - 2, block_size // 2)
        _, valid = _known_phase_block_average(
            prediction[0, 0], block_size=block_size, phase=phase
        )
        assert torch.count_nonzero(valid) > 0
        observation = phase_block_average_mosaic(
            image,
            block_size=block_size,
            phase=phase,
        )
        observation = torch.from_numpy(
            observation.transpose(2, 0, 1).copy()
        ).float().div(255.0).view_as(prediction)
        value = loss(
            prediction,
            observation,
            mask,
            torch.tensor([[phase]], dtype=torch.int64),
            torch.tensor([block_size], dtype=torch.int64),
            torch.ones(1),
        )
        assert value < 1e-9


def test_unknown_trainable_module_is_rejected():
    try:
        BasicVSRPlusPlusGanNet(
            mid_channels=64,
            num_blocks=1,
            spynet_pretrained=None,
            trainable_modules=('not_a_module',),
        )
    except ValueError as error:
        assert 'unknown' in str(error)
    else:
        raise AssertionError('unknown trainable module was accepted')


def test_aozora_hf_continuation_keeps_the_faithful_tail_contract():
    config = Config.fromfile(
        'configs/basicvsrpp/mosaic_restoration_generic_stage2.9_aozora_hf.py')

    assert config.model.generator.trainable_modules == list(TAIL_MODULES)
    assert config.model.perceptual_loss is None
    assert config.model.gan_loss is None
    assert config.optim_wrapper.generator.optimizer.lr == 1e-6
    assert config.train_cfg.max_iters == 500
    assert config.train_cfg.val_interval == 250
    assert 'train-old512-plus-aozora-v1.jsonl' in config.train_manifest
    assert 'validation-native-hf-512-recoverable-v1.jsonl' in (
        config.validation_manifest)


def test_fc2_hf_continuation_keeps_the_faithful_tail_contract():
    config = Config.fromfile(
        'configs/basicvsrpp/mosaic_restoration_generic_stage2.10_fc2_hf.py')

    assert config.model.generator.trainable_modules == list(TAIL_MODULES)
    assert config.model.perceptual_loss is None
    assert config.model.gan_loss is None
    assert config.optim_wrapper.generator.optimizer.lr == 1e-6
    assert config.train_cfg.max_iters == 500
    assert config.train_cfg.val_interval == 250
    assert 'hf2500-aozora-ema-as-generator.pth' in (
        config.initialization_checkpoint)
    assert 'train-old560-plus-fc2-v1.jsonl' in config.train_manifest
    assert 'validation-native-hf-512-recoverable-v1.jsonl' in (
        config.validation_manifest)


def test_forward_consistency_continuation_is_isolated_and_deterministic():
    config = Config.fromfile(
        'configs/basicvsrpp/'
        'mosaic_restoration_generic_stage2.11_forward_consistency.py'
    )

    assert config.model.generator.trainable_modules == list(TAIL_MODULES)
    assert config.model.mosaic_forward_consistency_loss.type == (
        'KnownGridMosaicConsistencyLoss'
    )
    assert config.model.mosaic_forward_consistency_loss.loss_weight == 0.20
    metric_types = [metric.type for metric in config.val_evaluator.metrics]
    assert 'ROIMosaicConsistencyError' in metric_types
    assert config.initialization_checkpoint == (
        'model_weights/hf2500-plus-fc2-500-ema.pth'
    )
    assert config.load_from == config.initialization_checkpoint
    assert config.randomness.seed == 20260808
    assert config.randomness.deterministic is True
    assert config.train_cfg.max_iters == 500
    assert 'train-old560-plus-fc2-v1.jsonl' in config.train_manifest
    assert 'validation-native-hf-512-recoverable-v1.jsonl' in (
        config.validation_manifest
    )


def test_forward_consistency_1000_extension_uses_selected_ema_and_new_seed():
    config = Config.fromfile(
        'configs/basicvsrpp/'
        'mosaic_restoration_generic_stage2.12_forward_consistency_1000.py'
    )

    assert config.model.generator.trainable_modules == list(TAIL_MODULES)
    assert config.model.mosaic_forward_consistency_loss.type == (
        'KnownGridMosaicConsistencyLoss'
    )
    assert config.model.mosaic_forward_consistency_loss.loss_weight == 0.20
    assert config.initialization_checkpoint.endswith(
        'hf2500-fc2-500-forward-consistency-500-ema.pth'
    )
    assert config.load_from == config.initialization_checkpoint
    assert config.randomness.seed == 20260809
    assert config.train_dataloader.dataset.seed == 20260809
    assert config.val_dataloader.dataset.seed == 20260803
    assert config.train_cfg.max_iters == 500
    assert config.train_cfg.val_interval == 250


def test_forward_consistency_1500_extension_uses_selected_ema_and_new_seed():
    config = Config.fromfile(
        'configs/basicvsrpp/'
        'mosaic_restoration_generic_stage2.13_forward_consistency_1500.py'
    )

    assert config.model.generator.trainable_modules == list(TAIL_MODULES)
    assert config.model.mosaic_forward_consistency_loss.type == (
        'KnownGridMosaicConsistencyLoss'
    )
    assert config.model.mosaic_forward_consistency_loss.loss_weight == 0.20
    assert config.initialization_checkpoint.endswith(
        'hf2500-fc2-forward-consistency-1000-ema.pth'
    )
    assert config.load_from == config.initialization_checkpoint
    assert config.randomness.seed == 20260810
    assert config.train_dataloader.dataset.seed == 20260810
    assert config.val_dataloader.dataset.seed == 20260803
    assert config.train_cfg.max_iters == 500
    assert config.train_cfg.val_interval == 250


def test_sharp_gan_wires_observation_and_requires_explicit_validity_metadata():
    register_all_modules()
    config = Config.fromfile(
        'configs/basicvsrpp/'
        'mosaic_restoration_generic_stage2.11_forward_consistency.py'
    )
    model = MODELS.build(config.model)
    sequence = torch.full((1, 2, 3, 8, 8), 102.0 / 255.0)
    sample = DataSample(
        gt_img=sequence.clone(),
        mask=torch.ones(1, 2, 1, 8, 8),
    )

    try:
        model.extract_gt_data(sample)
    except ValueError as error:
        assert 'mosaic_observation_weight' in str(error)
    else:
        raise AssertionError('missing exact-operator metadata was accepted')

    sample.mosaic_phase = torch.zeros(1, 2, 2, dtype=torch.int64)
    sample.mosaic_block_size = torch.tensor([4], dtype=torch.int64)
    sample.mosaic_observation_weight = torch.ones(1)
    losses = model.g_step(
        (sequence.clone(), sequence.clone()),
        model.extract_gt_data(sample),
    )
    assert 'loss_mosaic_forward_consistency_roi' in losses
    torch.testing.assert_close(
        losses['loss_mosaic_forward_consistency_roi'], torch.tensor(0.0)
    )
