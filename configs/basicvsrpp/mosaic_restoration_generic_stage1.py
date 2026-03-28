from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *

experiment_name = 'mosaic_restoration_generic_stage1'
work_dir = f'./experiments/basicvsrpp/{experiment_name}'
save_dir = './experiments/basicvsrpp'

model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlusNet',
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained='model_weights/3rd_party/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

data_root = 'datasets/mosaic_removal_vid'

train_dataloader = dict(
    num_workers=4,
    batch_size=2,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=data_root + "/train/crop_unscaled_meta",
        num_frame=30,
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
    type='IterBasedTrainLoop', max_iters=100_000, val_interval=5000)
val_cfg = dict(type='MultiValLoop')

optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}))


vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    name='visualizer',
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=5)]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, out_dir=save_dir),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False))
