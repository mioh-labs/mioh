import sys
import types
from pathlib import Path

import pytest

from lada.models.yolo.yolo11_coreai_segmentation_model import (
    CoreAISegmentationRuntime,
)
from lada.restorationpipeline.basicvsrpp_coreai_restorer import CoreAIModelRuntime
from lada.restorationpipeline.coreai_roi_enhancer import CoreAIEnhancerRuntime


class FakeLoadedModel:
    def load_function(self, name):
        assert name == "main"
        return object()


class SuccessfulAIModel:
    @staticmethod
    async def load(path):
        assert Path(path).suffix == ".aimodel"
        return FakeLoadedModel()


class FailingAIModel:
    @staticmethod
    async def load(path):
        raise ValueError(f"specialization failed for {Path(path).name}")


def install_fake_coreai(monkeypatch, ai_model):
    coreai = types.ModuleType("coreai")
    runtime = types.ModuleType("coreai.runtime")
    runtime.AIModel = ai_model
    coreai.runtime = runtime
    monkeypatch.setitem(sys.modules, "coreai", coreai)
    monkeypatch.setitem(sys.modules, "coreai.runtime", runtime)


@pytest.mark.parametrize(
    "filename,factory",
    [
        (
            "basicvsrpp-v1.2-t18-fp16.aimodel",
            lambda path: CoreAIModelRuntime(path, frame_count=18),
        ),
        (
            "lada_mosaic_detection_model_v4_fast-fp16.aimodel",
            CoreAISegmentationRuntime,
        ),
        (
            "realesr-general-x4v3-256-fp16.aimodel",
            CoreAIEnhancerRuntime,
        ),
    ],
)
def test_source_model_load_reports_first_use_preparation(
    tmp_path, monkeypatch, capsys, filename, factory
):
    model_path = tmp_path / filename
    model_path.mkdir()
    install_fake_coreai(monkeypatch, SuccessfulAIModel)
    runtime = factory(model_path)

    runtime._ensure_loaded()
    runtime.close()

    output = capsys.readouterr().out
    assert f"Core AIモデルを準備中（初回はこのMac向けに最適化します）: {filename}" in output
    assert f"Core AIモデルの準備完了: {filename}" in output


def test_source_model_load_wraps_error_with_model_name(tmp_path, monkeypatch):
    model_path = tmp_path / "basicvsrpp-v1.2-t18-fp16.aimodel"
    model_path.mkdir()
    install_fake_coreai(monkeypatch, FailingAIModel)
    runtime = CoreAIModelRuntime(model_path, frame_count=18)

    with pytest.raises(
        RuntimeError,
        match=r"BasicVSR\+\+.*basicvsrpp-v1.2-t18-fp16.aimodel.*specialization failed",
    ):
        runtime._ensure_loaded()
    runtime.close()
