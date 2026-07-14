import importlib.util
from pathlib import Path

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
    "RealESRGAN_x4plus-256-fp16.h17s.aimodelc",
    "realesr-general-x4v3-256-fp16.h17s.aimodelc",
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


def test_verifier_manifest_is_exactly_the_six_m5_pro_models():
    assert verifier.EXPECTED_MODEL_ASSETS == EXPECTED_MODELS
    assert set(verifier.MODEL_CONTRACTS) == {
        "basicvsrpp-v1.2-coreai",
        "basicvsrpp-v1.2-coreai-t36",
        "basicvsrpp-v1.2-coreai-t90",
        "v4-fast-coreai",
        "realesrgan-x4-coreai",
        "realesr-general-x4v3-coreai",
    }


def test_gradient_input_is_deterministic_contiguous_fp16():
    first = verifier.gradient_input((1, 3, 4, 4))
    second = verifier.gradient_input((1, 3, 4, 4))

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float16
    assert first.flags.c_contiguous
    assert first.min() == 0
    assert first.max() == 1
