from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
PHASE_A = (
    ROOT
    / "configs/basicvsrpp/mosaic_restoration_generic_stage2.21_large_roi_full_trunk.py"
)
PHASE_B = (
    ROOT
    / "configs/basicvsrpp/mosaic_restoration_generic_stage2.22_large_roi_tail_consolidation.py"
)
RUNNER = ROOT / "scripts/training/run-basicvsrpp-large-roi-full.sh"


def test_large_roi_training_is_isolated_from_start612():
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (PHASE_A, PHASE_B, RUNNER)
    ).lower()

    # The guard may name the excluded source only in a rejection message.
    assert "start612-source" not in combined
    assert "start612-dynamic" not in combined
    assert "train-large-roi-native-tiles-start612" not in combined
    assert "validation-large-roi-native-tiles-start612" not in combined


def test_phase_a_continues_adopted_large_roi_model_with_full_restoration_trunk():
    config = Config.fromfile(PHASE_A)

    assert "trainable_modules" not in config.model.generator
    assert config.model.generator.spynet_pretrained is None
    assert config.initialization_checkpoint.endswith(
        "/model_weights/basicvsrpp-v1.2-large-roi-native-tiles-27000-ema.pth"
    )
    assert config.train_manifest.endswith("/train-large-roi-native-tiles-v1.jsonl")
    assert config.train_cfg.max_iters == 20_000
    assert config.optim_wrapper.generator.optimizer.lr == 2e-6
    assert len(config.val_dataloader) == 2
    assert config.default_hooks.checkpoint.save_best == (
        "large_roi/ROILaplacianError"
    )


def test_phase_b_is_explicit_low_lr_tail_consolidation():
    config = Config.fromfile(PHASE_B)

    assert set(config.model.generator.trainable_modules) == {
        "reconstruction",
        "upsample1",
        "upsample2",
        "conv_hr",
        "conv_last",
    }
    assert config.load_from is None
    assert config.train_cfg.max_iters == 10_000
    assert config.optim_wrapper.generator.optimizer.lr == 3e-7


def test_runner_defaults_to_preflight_and_requires_explicit_training_consent():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'action="${ACTION:-preflight}"' in source
    assert '"${CONFIRM_TRAINING:-0}" != "1"' in source
    assert "PHASE_A_CHECKPOINT must name the accepted phase-A" in source
    assert "enable_native_mps_grid_sample_backward(raise_on_error=True)" in source
    assert "start612_references\": 0" in source
