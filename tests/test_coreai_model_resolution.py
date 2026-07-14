import lada


def test_coreai_model_path_prefers_selected_specialization(tmp_path, monkeypatch):
    source = tmp_path / "basicvsrpp-v1.2-t18-fp16.aimodel"
    compiled = tmp_path / "basicvsrpp-v1.2-t18-fp16.h17s.aimodelc"
    source.mkdir()
    compiled.mkdir()
    monkeypatch.setattr(lada, "MODEL_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("LADA_COREAI_ARCHITECTURE", "h17s")

    assert lada._coreai_model_path(source.name) == str(compiled)


def test_coreai_model_path_falls_back_to_source(tmp_path, monkeypatch):
    source = tmp_path / "basicvsrpp-v1.2-t18-fp16.aimodel"
    source.mkdir()
    monkeypatch.setattr(lada, "MODEL_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("LADA_COREAI_ARCHITECTURE", "h17s")

    assert lada._coreai_model_path(source.name) == str(source)

