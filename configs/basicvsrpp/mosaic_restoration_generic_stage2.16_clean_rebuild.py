"""Clean continuation of the generic v1.2 full BasicVSR++ foundation.

This reproduces the historical 26-frame, full-generator RGB GAN/perceptual
fine-tune through the selected 9,000-step checkpoint.  As in the historical
iter9000 run, it starts from ``generic_v1.2_full`` rather than random generator
weights.  Its dataset is a non-destructive filtered view that excludes sources
whose interlace combing was baked into nominally progressive clips.
"""

from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *


experiment_name = "mosaic_restoration_generic_stage2.16_clean_rebuild"
experiment_root = (
    "/Volumes/Project_HD/lada_finetune_aozora_hikari/"
    "basicvsrpp_clean_no_interlace_fc2_v3"
)
work_dir = (
    experiment_root
    + "/stage-00-generic-v1.2-full-finetune-9000-seed229883930"
)
dataset_root = experiment_root + "/dataset_representative"
initialization_checkpoint = (
    "model_weights/lada_mosaic_restoration_model_generic_v1.2_full.pth"
)

model = dict(
    type="BasicVSRPlusPlusGan",
    generator=dict(
        type="BasicVSRPlusPlusGanNet",
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=(
            "model_weights/3rd_party/spynet_20210409-c6c1bd09.pth"
        ),
    ),
    discriminator=dict(
        type="UNetDiscriminatorWithSpectralNorm",
        in_channels=3,
        mid_channels=64,
        skip_connection=True,
    ),
    pixel_loss=dict(
        type="CharbonnierLoss", loss_weight=1.0, reduction="mean"
    ),
    perceptual_loss=dict(
        type="PerceptualLoss",
        layer_weights={
            "2": 0.1,
            "7": 0.1,
            "16": 1.0,
            "25": 1.0,
            "34": 1.0,
        },
        vgg_type="vgg19",
        pretrained="model_weights/3rd_party/vgg19-dcbb9e9d.pth",
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False,
    ),
    gan_loss=dict(
        type="GANLoss",
        gan_type="vanilla",
        loss_weight=0.1,
        real_label_val=1.0,
        fake_label_val=0,
    ),
    is_use_ema=True,
    data_preprocessor=dict(
        type="DataPreprocessor",
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    ),
)

train_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="MosaicVideoDataset",
        metadata_root_dir=dataset_root + "/train/crop_unscaled_meta",
        num_frame=26,
        degrade=True,
        use_hflip=True,
        repeatable_random=False,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256,
    ),
    collate_fn=dict(type="default_collate"),
)

val_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="MosaicVideoDataset",
        metadata_root_dir=dataset_root + "/validation/crop_unscaled_meta",
        num_frame=30,
        degrade=True,
        use_hflip=False,
        repeatable_random=True,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256,
    ),
    collate_fn=dict(type="default_collate"),
)

val_evaluator = dict(
    type="Evaluator",
    metrics=[dict(type="PSNR"), dict(type="SSIM")],
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=9_000,
    val_interval=2_000,
)
val_cfg = dict(type="MultiValLoop")

optim_wrapper = dict(
    constructor="MultiOptimWrapperConstructor",
    generator=dict(
        type="OptimWrapper",
        optimizer=dict(type="Adam", lr=5e-5, betas=(0.9, 0.99)),
        paramwise_cfg=dict(custom_keys={"spynet": dict(lr_mult=0.25)}),
    ),
    discriminator=dict(
        type="OptimWrapper",
        optimizer=dict(type="Adam", lr=1e-4, betas=(0.9, 0.99)),
    ),
)

vis_backends = [dict(type="TensorboardVisBackend")]
visualizer = dict(
    name="visualizer",
    type="ConcatImageVisualizer",
    vis_backends=vis_backends,
    fn_key="gt_path",
    img_keys=["gt_img", "input", "pred_img"],
    bgr2rgb=True,
)

custom_hooks = [
    dict(type="BasicVisualizationHook", interval=5),
    dict(
        type="ExponentialMovingAverageHook",
        module_keys="generator_ema",
        interval=1,
        interp_cfg=dict(momentum=0.001),
    ),
]

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=1_000),
    logger=dict(
        type="LoggerHook", interval=10, log_metric_by_epoch=False
    ),
    param_scheduler=dict(type="ParamSchedulerHook"),
)

randomness = dict(seed=229883930, deterministic=False)
load_from = initialization_checkpoint
resume = False
