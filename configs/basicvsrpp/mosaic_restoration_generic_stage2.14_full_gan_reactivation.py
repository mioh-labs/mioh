"""Reactivate the proven full-generator perceptual/GAN curriculum.

This experiment starts from the deployed HF2500 + FC2 + forward-consistency
EMA candidate.  It deliberately mirrors the successful generic-v1.2-full
Stage 2.6 objective: 26-frame BasicVSR++ propagation, full-RGB U-Net
discrimination, VGG19 perceptual supervision and a Charbonnier fidelity
anchor.  SPyNet remains frozen, exactly as it was for the original generic
model; every other generator module is trainable.

The run is isolated from production weights.  A later, separately gated stage
will turn perceptual/GAN supervision off and reapply the conservative HF,
FC2 and forward-consistency tail curriculum.
"""

from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *


experiment_name = 'mosaic_restoration_generic_stage2.14_full_gan_reactivation'
experiment_root = (
    '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
    'basicvsrpp_full_gan_reactivation'
)
work_dir = experiment_root + '/stage-a-v1-seed20260814'

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

model = dict(
    type='BasicVSRPlusPlusGan',
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
    pixel_loss=dict(
        type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
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
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False,
    ),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.1,
        real_label_val=1.0,
        fake_label_val=0,
    ),
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
        type='MosaicVideoDataset',
        metadata_root_dir=train_metadata,
        num_frame=26,
        degrade=True,
        use_hflip=True,
        repeatable_random=False,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256,
        rotation_probability=0.15,
    ),
    collate_fn=dict(type='default_collate'),
)

val_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=validation_metadata,
        num_frame=30,
        degrade=True,
        use_hflip=False,
        repeatable_random=True,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256,
        rotation_probability=0.0,
    ),
    collate_fn=dict(type='default_collate'),
)

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ],
)

# Start conservatively because the initial generator is already a promoted,
# high-fidelity model.  The loss ratios match generic v1.2 full, while the
# optimizer rates match its safer sharp continuation phase.
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

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=500,
    val_interval=250,
)
val_cfg = dict(type='MultiValLoop')

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
        interval=50,
        max_keep_ckpts=12,
    ),
    logger=dict(
        type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
)

# mioh's native Metal deformable-convolution backward uses atomic gradient
# accumulation.  The seed still fixes sampler and augmentation streams, but
# bit-exact GPU replay is not available.
randomness = dict(seed=20260814, deterministic=False)
load_from = initialization_checkpoint
resume = False
