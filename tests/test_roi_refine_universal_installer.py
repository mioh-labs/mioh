import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "packaging/macOS/standalone/model-tools/install-roi-refine.zsh"
BUILD_SCRIPT = ROOT / "packaging/macOS/standalone/build_app.sh"
RATES = ("23.976", "24.000", "25.000", "29.970", "30.000", "50.000", "59.940", "60.000")
SHAPES = ("384", "768", "1024", "256x512", "512x256")


def test_installer_and_converters_are_bundled_with_universal() -> None:
    build = BUILD_SCRIPT.read_text()
    assert '"$RESOURCES/model-tools/install-roi-refine.zsh"' in build
    assert '"$DMG_ROOT/install-roi-refine.zsh"' in build
    for script in ("export_ltx23_dit_block_coreai.py", "export_ltx23_roi_refine_constants.py",
                   "run_ltx23_roi_refine_coreai.py"):
        assert f'"$RESOURCES/model-tools/scripts/apple/{script}"' in build
        assert (ROOT / "scripts/apple" / script).is_file()
    subprocess.run(["zsh", "-n", str(INSTALLER)], check=True)


def test_verify_only_checks_every_pack_asset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        app = root / "mioh-universal.app"
        (app / "Contents/Resources/model-tools").mkdir(parents=True)
        python = app / "Contents/Resources/runtime/bin/python3.12"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        pack = root / "pack"
        (pack / "bin/mlx-swift_Cmlx.bundle").mkdir(parents=True)
        server = pack / "bin/ltx-video"
        server.write_bytes(b"server")
        server.chmod(0o755)
        (pack / "vae.safetensors").write_bytes(b"vae")
        (pack / "roi-refine-pack.json").write_text("{}")
        for layer in range(0, 48, 4):
            (pack / f"blocks/group{layer:02d}.aimodel").mkdir(parents=True)
            (pack / f"blocks/group{layer:02d}.aimodel.json").write_text("{}")
        for rate in RATES:
            (pack / "constants").mkdir(exist_ok=True)
            (pack / f"constants/fps-{rate}.safetensors").write_bytes(b"c")
            for shape in SHAPES:
                (pack / f"constants/size-{shape}").mkdir(exist_ok=True)
                (pack / f"constants/size-{shape}/fps-{rate}.safetensors").write_bytes(b"c")

        command = ["zsh", str(INSTALLER), "--app", str(app), "--pack", str(pack), "--verify-only"]
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "pack verified" in result.stdout

        (pack / "constants/size-512x256/fps-59.940.safetensors").unlink()
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode != 0
        assert "size-512x256/fps-59.940" in result.stderr
