"""Conservative tail consolidation after full-trunk large-ROI training."""

from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *


experiment_name = "mosaic_restoration_generic_stage2.22_large_roi_tail_consolidation"
experiment_root = (
    "/Volumes/Project_HD/lada_finetune_aozora_hikari/"
    "basicvsrpp_clean_no_interlace_fc2_v3"
)
work_dir = experiment_root + "/stage-05-large-roi-full/phase-b-tail-consolidation"
train_manifest = experiment_root + "/manifests/train-large-roi-native-tiles-v1.jsonl"
large_roi_validation_manifest = (
    experiment_root + "/manifests/validation-large-roi-native-tiles-v1.jsonl"
)
generic_validation_manifest = (
    experiment_root + "/manifests/validation-native-hf-clean-v1.jsonl"
)

tail_modules = [
    "reconstruction",
    "upsample1",
    "upsample2",
    "conv_hr",
    "conv_last",
]

model = dict(
    type="BasicVSRPlusPlusSharpGan",
    generator=dict(
        type="BasicVSRPlusPlusGanNet",
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,
        trainable_modules=tail_modules,
    ),
    discriminator=None,
    pixel_loss=dict(type="CharbonnierLoss", loss_weight=0.10, reduction="mean"),
    roi_pixel_loss=dict(type="ROIPixelLoss", loss_weight=1.0, mask_dilation=4),
    high_frequency_loss=dict(
        type="ROIHighFrequencyLoss",
        loss_weight=0.25,
        gradient_weight=1.0,
        laplacian_weight=0.5,
        mask_dilation=4,
    ),
    high_frequency_projection_loss=dict(
        type="ROIHighFrequencyProjectionLoss",
        loss_weight=1.0,
        projection_weight=0.0035,
        orthogonal_energy_weight=0.0015,
        mask_dilation=0,
    ),
    temporal_loss=dict(
        type="ROITemporalDifferenceLoss", loss_weight=0.03, mask_dilation=4
    ),
    mosaic_forward_consistency_loss=dict(
        type="KnownGridMosaicConsistencyLoss",
        loss_weight=0.05,
        dead_zone=0.5 / 255.0,
    ),
    perceptual_loss=None,
    gan_loss=None,
    roi_dilation=4,
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
        type="RecoverableHFMosaicVideoDataset",
        manifest=train_manifest,
        training=True,
        use_hflip=True,
        time_reverse=True,
        minimum_block_size=6,
        maximum_block_size=12,
        seed=20260821,
    ),
    collate_fn=dict(type="default_collate"),
)


val_dataloader = [
    dict(
        num_workers=0,
        batch_size=1,
        persistent_workers=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="RecoverableHFMosaicVideoDataset",
            manifest=large_roi_validation_manifest,
            training=False,
            use_hflip=False,
            time_reverse=False,
            minimum_block_size=6,
            maximum_block_size=12,
            seed=20260821,
        ),
        collate_fn=dict(type="default_collate"),
    ),
    dict(
        num_workers=0,
        batch_size=1,
        persistent_workers=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="RecoverableHFMosaicVideoDataset",
            manifest=generic_validation_manifest,
            training=False,
            use_hflip=False,
            time_reverse=False,
            minimum_block_size=6,
            maximum_block_size=12,
            seed=20260822,
        ),
        collate_fn=dict(type="default_collate"),
    ),
]
val_evaluator = [
    dict(
        type="Evaluator",
        metrics=[
            dict(type="ROIPSNR", prefix="large_roi"),
            dict(type="ROILaplacianError", prefix="large_roi"),
            dict(type="ROIMosaicConsistencyError", prefix="large_roi"),
            dict(type="PSNR", prefix="large_roi"),
            dict(type="SSIM", prefix="large_roi"),
        ],
    ),
    dict(
        type="Evaluator",
        metrics=[
            dict(type="ROIPSNR", prefix="generic"),
            dict(type="ROILaplacianError", prefix="generic"),
            dict(type="ROIMosaicConsistencyError", prefix="generic"),
            dict(type="PSNR", prefix="generic"),
            dict(type="SSIM", prefix="generic"),
        ],
    ),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=10_000, val_interval=500)
val_cfg = dict(type="MultiValLoop")

optim_wrapper = dict(
    constructor="MultiOptimWrapperConstructor",
    generator=dict(
        type="OptimWrapper",
        modules=[
            r"generator\.reconstruction",
            r"generator\.upsample1",
            r"generator\.upsample2",
            r"generator\.conv_hr",
            r"generator\.conv_last",
        ],
        clip_grad=dict(max_norm=1.0, norm_type=2),
        optimizer=dict(type="Adam", lr=3e-7, betas=(0.9, 0.99)),
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
    dict(type="BasicVisualizationHook", interval=500),
    dict(
        type="ExponentialMovingAverageHook",
        module_keys="generator_ema",
        interval=1,
        interp_cfg=dict(momentum=0.001),
    ),
]
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=250,
        max_keep_ckpts=45,
        save_best="large_roi/ROILaplacianError",
        rule="less",
    ),
    logger=dict(type="LoggerHook", interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
)

randomness = dict(seed=20260821, deterministic=False)
# The runner requires an explicitly selected phase-A checkpoint.
load_from = None
resume = False
