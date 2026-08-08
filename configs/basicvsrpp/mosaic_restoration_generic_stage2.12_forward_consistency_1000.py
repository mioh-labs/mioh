"""Fresh 500-step extension of the selected Stage 2.11 EMA candidate.

The model-only Stage 2.11 EMA checkpoint is used as both the raw and EMA
initial state.  A distinct training seed avoids replaying the first 500-step
sample/augmentation sequence.  Local iterations 250 and 500 correspond to
cumulative continuation points 750 and 1000 respectively.
"""

from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *


experiment_name = 'mosaic_restoration_generic_stage2.12_forward_consistency_1000'
work_dir = f'./experiments/basicvsrpp/{experiment_name}'

initialization_checkpoint = (
    '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
    'forward_consistency_stage2_11/deployment/'
    'hf2500-fc2-500-forward-consistency-500-ema.pth'
)
train_manifest = (
    '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
    'fc2_best_hf_v1/manifests/train-old560-plus-fc2-v1.jsonl'
)
validation_manifest = (
    '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
    'mioh_native_hf/manifests/validation-native-hf-512-recoverable-v1.jsonl'
)

tail_modules = [
    'reconstruction',
    'upsample1',
    'upsample2',
    'conv_hr',
    'conv_last',
]

model = dict(
    type='BasicVSRPlusPlusSharpGan',
    generator=dict(
        type='BasicVSRPlusPlusGanNet',
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,
        trainable_modules=tail_modules,
    ),
    discriminator=None,
    pixel_loss=dict(
        type='CharbonnierLoss', loss_weight=0.10, reduction='mean'),
    roi_pixel_loss=dict(
        type='ROIPixelLoss', loss_weight=1.0, mask_dilation=4),
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
    perceptual_loss=None,
    gan_loss=None,
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
        type='RecoverableHFMosaicVideoDataset',
        manifest=train_manifest,
        training=True,
        use_hflip=True,
        time_reverse=True,
        minimum_block_size=6,
        maximum_block_size=12,
        seed=20260809,
    ),
    collate_fn=dict(type='default_collate'),
)

val_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RecoverableHFMosaicVideoDataset',
        manifest=validation_manifest,
        training=False,
        use_hflip=False,
        time_reverse=False,
        minimum_block_size=6,
        maximum_block_size=12,
        seed=20260803,
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
    max_iters=500,
    val_interval=250,
)
val_cfg = dict(type='MultiValLoop')

optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        modules=[
            r'generator\.reconstruction',
            r'generator\.upsample1',
            r'generator\.upsample2',
            r'generator\.conv_hr',
            r'generator\.conv_last',
        ],
        optimizer=dict(type='Adam', lr=1e-6, betas=(0.9, 0.99)),
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
    dict(type='BasicVisualizationHook', interval=250),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.001),
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=250,
        max_keep_ckpts=3,
        save_best='ROILaplacianError',
        rule='less',
    ),
    logger=dict(
        type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
)

randomness = dict(seed=20260809, deterministic=True)
load_from = initialization_checkpoint
resume = False
