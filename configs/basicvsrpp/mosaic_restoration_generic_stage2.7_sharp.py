from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *

experiment_name = 'mosaic_restoration_generic_stage2.7_sharp'
work_dir = f'./experiments/basicvsrpp/{experiment_name}'
save_dir = './experiments/basicvsrpp'

model = dict(
    type='BasicVSRPlusPlusSharpGan',
    generator=dict(
        type='BasicVSRPlusPlusGanNet',
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained='model_weights/3rd_party/spynet_20210409-c6c1bd09.pth'),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
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
        norm_img=False),
    high_frequency_loss=dict(
        type='ROIHighFrequencyLoss',
        loss_weight=0.15,
        gradient_weight=1.0,
        laplacian_weight=0.5,
        mask_dilation=4),
    temporal_loss=dict(
        type='ROITemporalDifferenceLoss',
        loss_weight=0.03,
        mask_dilation=4),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.03,
        real_label_val=1.0,
        fake_label_val=0),
    roi_dilation=4,
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

data_root = (
    '/Volumes/Project_HD/lada_finetune_aozora_hikari/'
    'basicvsrpp_sharp_fc2_best/dataset_sharp_selected'
)

train_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=data_root + '/train/crop_unscaled_meta',
        num_frame=26,
        degrade=True,
        use_hflip=True,
        repeatable_random=False,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256,
        native_roi_crop=True,
        return_mosaic_mask=True,
        rotation_probability=0.15),
    collate_fn=dict(type='default_collate'))

val_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=data_root + '/validation/crop_unscaled_meta',
        num_frame=30,
        degrade=True,
        use_hflip=False,
        repeatable_random=True,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256,
        native_roi_crop=True,
        return_mosaic_mask=True,
        rotation_probability=0.0),
    collate_fn=dict(type='default_collate'))

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=3000,
    val_interval=1000)
val_cfg = dict(type='MultiValLoop')

optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-5, betas=(0.9, 0.99)),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-5, betas=(0.9, 0.99))),
)

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    name='visualizer',
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [
    dict(type='BasicVisualizationHook', interval=50),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.001),
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        out_dir=save_dir),
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
)
