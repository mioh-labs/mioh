"""Full-generator GAN continuation with fidelity and temporal safeguards.

This clean 10k-step arm starts from the deployed HF2500 + FC2 +
forward-consistency EMA generator.  It keeps the useful distribution learning
from generic-v1.2-full, but restores the safeguards that were learned later in
the promoted curriculum: ROI-normalized pixel fidelity, recoverable
high-frequency supervision and matching of real consecutive-frame changes.

The discriminator and VGG see RGB images, but fake pixels outside the mosaic
ROI are replaced by GT.  Their gradients therefore cannot spend capacity on
already-correct background.  SPyNet stays frozen; every other generator module
is trainable.  Exact known-grid forward consistency is intentionally deferred
to the recoverable-HF tail because MosaicVideoDataset does not expose a valid
per-frame grid phase for all of its degradation modes.
"""

from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *


experiment_name = (
    'mosaic_restoration_generic_stage2.14_full_gan_temporal_fidelity'
)
experiment_root = (
    '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
    'basicvsrpp_full_gan_reactivation'
)
work_dir = experiment_root + '/stage-a-temporal-fidelity-v1-seed20260814'

initialization_checkpoint = (
    '/Users/okatti/Documents/lada/model_weights/'
    'hf2500-plus-fc2-forward-consistency-w005-500-ema.pth'
)
train_metadata = (
    '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
    'dataset_representative/train/crop_unscaled_meta'
)
validation_metadata = (
    '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
    'dataset_representative/validation/crop_unscaled_meta'
)
known_grid_manifest_root = experiment_root + '/../known_grid_26_v1/manifests'
known_grid_train_manifest = (
    known_grid_manifest_root + '/train-known-grid-26-v1.jsonl'
)
known_grid_validation_manifest = (
    known_grid_manifest_root + '/validation-known-grid-26-v1.jsonl'
)

model = dict(
    type='BasicVSRPlusPlusSharpGan',
    generator=dict(
        type='BasicVSRPlusPlusGanNet',
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,
    ),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True,
    ),
    # Preserve clean context weakly; ROI fidelity below is the primary anchor.
    pixel_loss=dict(
        type='CharbonnierLoss', loss_weight=0.10, reduction='mean'),
    roi_pixel_loss=dict(
        type='ROIPixelLoss', loss_weight=1.0, mask_dilation=4),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        pretrained='model_weights/3rd_party/vgg19-dcbb9e9d.pth',
        perceptual_weight=0.5,
        style_weight=0,
        norm_img=False,
    ),
    high_frequency_loss=dict(
        type='ROIHighFrequencyLoss',
        loss_weight=0.25,
        gradient_weight=1.0,
        laplacian_weight=0.5,
        mask_dilation=4,
    ),
    temporal_loss=dict(
        type='ROITemporalDifferenceLoss',
        loss_weight=0.03,
        mask_dilation=4,
    ),
    mosaic_forward_consistency_loss=dict(
        type='KnownGridMosaicConsistencyLoss',
        loss_weight=0.20,
        dead_zone=0.5 / 255.0,
    ),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.03,
        real_label_val=1.0,
        fake_label_val=0,
    ),
    roi_dilation=4,
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    ),
)

train_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='AlternatingKnownGridMosaicVideoDataset',
        generic_dataset=dict(
            type='MosaicVideoDataset',
            metadata_root_dir=train_metadata,
            num_frame=26,
            degrade=True,
            use_hflip=True,
            time_reverse=True,
            repeatable_random=False,
            random_mosaic_params=True,
            filter_watermark=False,
            filter_nudenet_nsfw=False,
            filter_video_quality=False,
            lq_size=256,
            native_roi_crop=True,
            return_mosaic_mask=True,
            rotation_probability=0.15,
        ),
        known_grid_dataset=dict(
            type='KnownGridMosaicVideoDataset',
            manifest=known_grid_train_manifest,
            num_frame=26,
            lq_size=256,
            training=True,
            use_hflip=True,
            time_reverse=True,
            rotation_probability=0.15,
            minimum_block_size=6,
            maximum_block_size=12,
            seed=20260814,
        ),
    ),
    collate_fn=dict(type='default_collate'),
)

val_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KnownGridMosaicVideoDataset',
        manifest=known_grid_validation_manifest,
        num_frame=26,
        lq_size=256,
        training=False,
        use_hflip=False,
        time_reverse=False,
        rotation_probability=0.0,
        minimum_block_size=6,
        maximum_block_size=12,
        seed=20260814,
    ),
    collate_fn=dict(type='default_collate'),
)

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='ROIPSNR'),
        dict(type='ROILaplacianError'),
        dict(type='ROIMosaicConsistencyError'),
        dict(type='PSNR'),
        dict(type='SSIM'),
    ],
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=10_000,
    val_interval=500,
)
val_cfg = dict(type='MultiValLoop')

optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-5, betas=(0.9, 0.99)),
    ),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-5, betas=(0.9, 0.99)),
    ),
)

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    name='visualizer',
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True,
)

custom_hooks = [
    dict(type='BasicVisualizationHook', interval=50),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys='generator_ema',
        interval=1,
        interp_cfg=dict(momentum=0.001),
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=250,
        max_keep_ckpts=40,
    ),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
)

randomness = dict(seed=20260814, deterministic=False)
load_from = initialization_checkpoint
resume = False
