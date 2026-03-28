from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *

experiment_name = 'mosaic_restoration_generic_stage2.6'
work_dir = f'./experiments/basicvsrpp/{experiment_name}'
save_dir = './experiments/basicvsrpp'

model = dict(
    type='BasicVSRPlusPlusGan',
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
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.1,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

data_root = 'datasets/mosaic_removal_vid'

train_dataloader = dict(
    num_workers=4,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=data_root + "/train/crop_unscaled_meta",
        num_frame=26,
        degrade=True,
        use_hflip=True,
        repeatable_random=False,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256),
    collate_fn=dict(type='default_collate'))

val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=data_root + "/val/crop_unscaled_meta",
        num_frame=30,
        degrade=True,
        use_hflip=False,
        repeatable_random=True,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256),
    collate_fn=dict(type='default_collate'))

val_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100_000, val_interval=4000)
val_cfg = dict(type='MultiValLoop')

# optimizer
optim_wrapper = dict(
    #_delete_=True, # this was set to true in RealBasicVSR but I get a exception as value is not expected to be a boolean but needs to be a dict
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=5e-5, betas=(0.9, 0.99)),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})
    ),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))),
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
    dict(type='BasicVisualizationHook', interval=5),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.001),
    )
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, out_dir=save_dir),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
)
