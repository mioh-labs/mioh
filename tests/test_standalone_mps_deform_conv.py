import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packaging" / "macOS" / "standalone"
VENDOR = PACKAGE / "vendor" / "mps-deform-conv-0.2.2"
BUILD_SCRIPT = PACKAGE / "build_app.sh"
APP_SOURCE = PACKAGE / "MiohApp.swift"
VERIFY_SCRIPT = PACKAGE / "verify_mps_deform_conv.py"


class StandaloneMPSDeformConvTests(unittest.TestCase):
    def test_vendored_source_is_pinned_and_uses_cxx20(self):
        setup = (VENDOR / "setup.py").read_text()
        runtime = (VENDOR / "mps_deform_conv" / "__init__.py").read_text()
        metadata = (VENDOR / "pyproject.toml").read_text()
        license_text = (VENDOR / "LICENSE").read_text()

        self.assertIn('version="0.2.2"', setup)
        self.assertIn('version = "0.2.2"', metadata)
        self.assertNotIn("-std=c++17", setup + runtime)
        self.assertEqual((setup + runtime).count("-std=c++20"), 2)
        self.assertIn("MIT License", license_text)
        self.assertIn("imperatormk", license_text)

    def test_build_installs_vendor_against_bundled_torch(self):
        script = BUILD_SCRIPT.read_text()
        self.assertIn(
            'VENDORED_MPS_DEFORM_CONV="$PACKAGE_DIR/vendor/mps-deform-conv-0.2.2"',
            script,
        )
        self.assertIn(
            'MPS_DEFORM_BUILD_SOURCE="$BUILD_DIR/mps-deform-conv-source"',
            script,
        )
        self.assertIn(
            'ditto "$VENDORED_MPS_DEFORM_CONV" "$MPS_DEFORM_BUILD_SOURCE"',
            script,
        )
        self.assertIn("--no-deps", script)
        self.assertIn("--no-build-isolation", script)
        self.assertIn('"$MPS_DEFORM_BUILD_SOURCE"', script)
        self.assertIn(
            'cp "$VENDORED_MPS_DEFORM_CONV/LICENSE" '
            '"$RESOURCES/LICENSES/mps-deform-conv.txt"',
            script,
        )
        self.assertIn("verify_mps_deform_conv.py", script)

        verifier = VERIFY_SCRIPT.read_text()
        self.assertIn("from mps_deform_conv import deform_conv2d", verifier)
        self.assertIn("torch.backends.mps.is_available()", verifier)
        self.assertIn("torch.mps.synchronize()", verifier)
        self.assertIn("torch.isfinite(output).all()", verifier)

    def test_gui_selects_bundled_backend_without_removing_models(self):
        source = APP_SOURCE.read_text()
        self.assertIn(
            'result["LADA_DEFORM_CONV_BACKEND"] = "mps_deform_conv"',
            source,
        )
        self.assertIn('"basicvsrpp-v1.2"', source)
        self.assertIn('"カスタム"', source)


if __name__ == "__main__":
    unittest.main()
