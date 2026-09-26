import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "packaging/macOS/standalone/model-tools/install-swiftvr-models.zsh"
BUILD_SCRIPT = ROOT / "packaging/macOS/standalone/build_app.sh"


def _package(path: Path) -> None:
    (path / "Data/com.apple.CoreML/weights").mkdir(parents=True)
    (path / "Manifest.json").write_text("{}")
    (path / "Data/com.apple.CoreML/model.mlmodel").write_bytes(b"model")
    (path / "Data/com.apple.CoreML/weights/weight.bin").write_bytes(b"weights")


def test_installer_is_bundled_with_universal() -> None:
    build = BUILD_SCRIPT.read_text()
    assert '"$RESOURCES/model-tools/install-swiftvr-models.zsh"' in build
    assert '"$RESOURCES/model-tools/scripts/apple/swiftvr_imports.py"' in build
    assert '"$DMG_ROOT/install-swiftvr-models.zsh"' in build
    subprocess.run(["zsh", "-n", str(INSTALLER)], check=True)


def test_verify_only_checks_all_runtime_assets() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        app = root / "mioh-universal.app"
        (app / "Contents/Resources/model-tools").mkdir(parents=True)
        helper = app / "Contents/Resources/bin/mioh-native-coreai-preview"
        helper.parent.mkdir(parents=True)
        helper.write_bytes(b"helper")
        helper.chmod(0o755)
        pack = root / "models"
        for variant in ("t6", "t7"):
            components = pack / f"native-4x-{variant}-fp16/components"
            components.mkdir(parents=True)
            for name in ("context.f32", "modulation.f32", "rope-cosine.f32", "rope-sine.f32", "components.json"):
                (components / name).write_bytes(b"data")
            for name in ("patch", "head"):
                _package(components / f"{name}.mlpackage")
        for name in ("encoder-4f", "encoder-24f", "encoder-28f", "decoder-1latent", "decoder-6latent", "decoder-7latent"):
            _package(pack / f"reae-stateful-{name}-1024-fp32.mlpackage")
        for first in range(0, 30, 6):
            _package(pack / f"native-4x-fp16-grouped/dit-group-{first:02d}-{first+5:02d}-4x-float16.mlpackage")

        command = ["zsh", str(INSTALLER), "--app", str(app), "--model-root", str(pack), "--scale", "4", "--verify-only"]
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "4x pack verified" in result.stdout

        (pack / "native-4x-t6-fp16/components/rope-sine.f32").unlink()
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode != 0
        assert "incomplete components" in result.stderr
