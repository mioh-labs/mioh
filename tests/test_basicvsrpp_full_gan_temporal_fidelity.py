# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / 'configs/basicvsrpp/'
    'mosaic_restoration_generic_stage2.14_full_gan_temporal_fidelity.py'
)


def test_full_gan_temporal_fidelity_config_keeps_all_required_safeguards():
    config = Config.fromfile(CONFIG_PATH)
    model = config.model

    assert model.type == 'BasicVSRPlusPlusSharpGan'
    assert model.generator.type == 'BasicVSRPlusPlusGanNet'
    assert model.generator.spynet_pretrained is None
    assert model.pixel_loss.loss_weight == 0.10
    assert model.roi_pixel_loss.loss_weight == 1.0
    assert model.high_frequency_loss.loss_weight == 0.25
    assert model.temporal_loss.loss_weight == 0.03
    assert model.mosaic_forward_consistency_loss.loss_weight == 0.20
    assert model.perceptual_loss.perceptual_weight == 0.5
    assert model.gan_loss.loss_weight == 0.03
    assert model.is_use_ema is True

    train_dataset = config.train_dataloader.dataset
    assert train_dataset.type == 'AlternatingKnownGridMosaicVideoDataset'
    assert train_dataset.generic_dataset.num_frame == 26
    assert train_dataset.generic_dataset.native_roi_crop is True
    assert train_dataset.generic_dataset.return_mosaic_mask is True
    assert train_dataset.generic_dataset.time_reverse is True
    assert train_dataset.known_grid_dataset.num_frame == 26
    assert train_dataset.known_grid_dataset.time_reverse is True
    assert train_dataset.known_grid_dataset.manifest.endswith(
        'train-known-grid-26-v1.jsonl'
    )

    validation_metrics = {
        metric.type for metric in config.val_evaluator.metrics
    }
    assert {
        'ROIPSNR',
        'ROILaplacianError',
        'ROIMosaicConsistencyError',
        'PSNR',
        'SSIM',
    } <= validation_metrics
    assert config.train_cfg.max_iters == 10_000
    assert config.load_from.endswith(
        'hf2500-plus-fc2-forward-consistency-w005-500-ema.pth'
    )


def test_mosaic_video_dataset_time_reverse_reverses_all_three_streams():
    source = (ROOT / 'lada/models/basicvsrpp/mosaic_video_dataset.py').read_text()

    assert "self.time_reverse = opt.get('time_reverse', False)" in source
    reverse_block = source.index(
        'if self.time_reverse and rng_random.random() < 0.5:'
    )
    hflip_block = source.index('if self.use_hflip', reverse_block)
    block = source[reverse_block:hflip_block]
    assert 'img_gts.reverse()' in block
    assert 'img_lqs.reverse()' in block
    assert 'mask_lqs.reverse()' in block
