import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    ROOT / "packaging" / "macOS" / "standalone" / "verify_coreai_models.py"
)
SPEC = importlib.util.spec_from_file_location("verify_coreai_models", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


EXPECTED_MODELS = {
    "basicvsrpp-v1.2-t18-fp16.h17s.aimodelc",
    "basicvsrpp-v1.2-t36-fp16.h17s.aimodelc",
    "basicvsrpp-v1.2-t90-fp16.h17s.aimodelc",
    "lada_mosaic_detection_model_v4_fast-fp16.h17s.aimodelc",
    "RealESRGAN_x2plus-256-fp16.h17s.aimodelc",
    "RealESRGAN_x4plus-256-fp16.h17s.aimodelc",
    "realesr-general-x4v3-256-fp16.h17s.aimodelc",
    "4xNomosWebPhoto_RealPLKSR-256-fp16.h17s.aimodelc",
    "basicvsrpp-v1.2-variable-coreai.h17s.aimodelc",
}

EXPECTED_PORTABLE_MODELS = {
    "basicvsrpp-v1.2-t18-fp16.aimodel",
    "basicvsrpp-v1.2-t36-fp16.aimodel",
    "basicvsrpp-v1.2-t90-fp16.aimodel",
    "lada_mosaic_detection_model_v4_fast-fp16.aimodel",
    "RealESRGAN_x2plus-256-fp16.aimodel",
    "RealESRGAN_x4plus-256-fp16.aimodel",
    "realesr-general-x4v3-256-fp16.aimodel",
    "4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel",
    "basicvsrpp-v1.2-variable-coreai.aimodel",
}


def test_verifier_requires_exact_specialization_set(tmp_path):
    for model in EXPECTED_MODELS | {"unexpected.h17g.aimodelc"}:
        (tmp_path / model).mkdir()

    with pytest.raises(RuntimeError, match="unexpected Core AI assets"):
        verifier.verify_asset_set(tmp_path)


def test_verifier_rejects_source_models(tmp_path):
    for model in EXPECTED_MODELS:
        (tmp_path / model).mkdir()
    (tmp_path / "basicvsrpp-v1.2-t36-b2-fp16.aimodel").mkdir()

    with pytest.raises(RuntimeError, match="source Core AI assets"):
        verifier.verify_asset_set(tmp_path)


def test_verifier_manifest_includes_variable_pipeline_collection():
    assert verifier.EXPECTED_MODEL_ASSETS == EXPECTED_MODELS
    assert set(verifier.MODEL_CONTRACTS) == {
        "basicvsrpp-v1.2-coreai",
        "basicvsrpp-v1.2-coreai-t36",
        "basicvsrpp-v1.2-coreai-t90",
        "v4-fast-coreai",
        "realesrgan-x2-coreai",
        "realesrgan-x4-coreai",
        "realesr-general-x4v3-coreai",
        "nomos-webphoto-realplksr-x4-coreai",
    }


def test_gradient_input_is_deterministic_contiguous_fp16():
    first = verifier.gradient_input((1, 3, 4, 4))
    second = verifier.gradient_input((1, 3, 4, 4))

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float16
    assert first.flags.c_contiguous
    assert first.min() == 0
    assert first.max() == 1


def test_portable_verifier_accepts_exact_source_model_set(tmp_path):
    for model in EXPECTED_PORTABLE_MODELS:
        (tmp_path / model).mkdir()

    verifier.verify_asset_set(tmp_path, distribution="portable")


def test_portable_verifier_rejects_compiled_model(tmp_path):
    for model in EXPECTED_PORTABLE_MODELS:
        (tmp_path / model).mkdir()
    (tmp_path / "basicvsrpp-v1.2-t18-fp16.h17s.aimodelc").mkdir()

    with pytest.raises(RuntimeError, match="unexpected Core AI assets"):
        verifier.verify_asset_set(tmp_path, distribution="portable")


def test_distribution_manifest_adds_variable_collection_to_dedicated_build():
    assert verifier.expected_model_assets("dedicated", "h17s") == EXPECTED_MODELS
    assert verifier.expected_model_assets("portable", "h17s") == EXPECTED_PORTABLE_MODELS


def test_distribution_manifest_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported Core AI distribution"):
        verifier.expected_model_assets("server", "h17s")


@pytest.mark.parametrize("distribution", ["dedicated", "portable"])
def test_verifier_resolves_all_seven_models_for_distribution(
    tmp_path, monkeypatch, distribution
):
    resources = tmp_path / "Resources"
    models = resources / "models"
    models.mkdir(parents=True)
    for asset in verifier.expected_model_assets(distribution, "h17s"):
        (models / asset).mkdir()

    def resolve(name, kind):
        if name == verifier.VARIABLE_MODEL_NAME:
            variable_asset = (
                "basicvsrpp-v1.2-variable-coreai.h17s.aimodelc"
                if distribution == "dedicated"
                else "basicvsrpp-v1.2-variable-coreai.aimodel"
            )
            return SimpleNamespace(
                path=str(models / variable_asset)
            )
        contract = verifier.MODEL_CONTRACTS[name]
        asset = verifier.model_asset_name(contract["asset"], distribution, "h17s")
        return SimpleNamespace(path=str(models / asset))

    monkeypatch.setattr(verifier, "_resolve_model", resolve)
    monkeypatch.setattr(verifier, "_verify_restoration", lambda *args: None)
    monkeypatch.setattr(verifier, "_verify_detection", lambda *args: None)
    monkeypatch.setattr(verifier, "_verify_enhancer", lambda *args: None)
    monkeypatch.setattr(
        verifier, "_verify_variable_restoration", lambda *args: None
    )

    verifier.verify_models(
        resources,
        distribution=distribution,
        architecture="h17s",
        smoke_names=set(),
    )


def test_portable_environment_removes_architecture_override(tmp_path, monkeypatch):
    resources = tmp_path / "Resources"
    monkeypatch.setenv("LADA_COREAI_ARCHITECTURE", "h17s")

    verifier.configure_environment(resources, "portable", "h17s")

    assert "LADA_COREAI_ARCHITECTURE" not in verifier.os.environ
    assert verifier.os.environ["LADA_MODEL_WEIGHTS_DIR"] == str(resources / "models")
    assert verifier.os.environ["LADA_VARIABLE_COREAI_SWIFT_RUNNER"].endswith(
        "lada-basicvsrpp-variable-runner"
    )
