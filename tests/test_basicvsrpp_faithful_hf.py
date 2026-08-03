# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
from mmengine.config import Config

from lada.models.basicvsrpp import register_all_modules
from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet
from lada.models.basicvsrpp.mmagic.roi_loss import ROIPixelLoss


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
