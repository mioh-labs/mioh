import asyncio
import sys
import types

from lada.coreai import source_runtime


def test_source_model_load_retries_install_collision(tmp_path, monkeypatch):
    model_path = tmp_path / "model.aimodel"
    model_path.mkdir()
    (model_path / "main.hash").write_bytes(b"same-model")
    monkeypatch.setenv("LADA_COREAI_MODEL_LOCK_DIR", str(tmp_path / "locks"))

    calls = 0
    expected = object()

    class FakeAIModel:
        @staticmethod
        async def load(_path):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("an item with the same name already exists")
            return expected

    coreai = types.ModuleType("coreai")
    runtime = types.ModuleType("coreai.runtime")
    runtime.AIModel = FakeAIModel
    coreai.runtime = runtime
    monkeypatch.setitem(sys.modules, "coreai", coreai)
    monkeypatch.setitem(sys.modules, "coreai.runtime", runtime)

    runner = asyncio.Runner()
    try:
        result = source_runtime.load_source_model(runner, model_path, purpose="test")
    finally:
        runner.close()

    assert result is expected
    assert calls == 2
    assert len(list((tmp_path / "locks").glob("*.lock"))) == 1


def test_identical_model_content_uses_same_cross_process_lock(tmp_path, monkeypatch):
    first = tmp_path / "first.aimodel"
    second = tmp_path / "second.aimodel"
    first.mkdir()
    second.mkdir()
    (first / "main.hash").write_bytes(b"identical")
    (second / "main.hash").write_bytes(b"identical")
    monkeypatch.setenv("LADA_COREAI_MODEL_LOCK_DIR", str(tmp_path / "locks"))

    assert source_runtime._model_load_lock_path(
        first
    ) == source_runtime._model_load_lock_path(second)
