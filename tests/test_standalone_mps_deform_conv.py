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

    def test_native_app_does_not_bundle_torch_or_mps_deform_conv(self):
        script = BUILD_SCRIPT.read_text()
        self.assertNotIn("VENDORED_MPS_DEFORM_CONV", script)
        self.assertNotIn("mps-deform-conv.txt", script)
        self.assertNotIn("verify_mps_deform_conv.py", script)
        self.assertNotIn("$RESOURCES/runtime", script)
        self.assertNotIn("site-packages/torch", script)

    def test_gui_uses_native_models_without_a_python_backend_override(self):
        source = APP_SOURCE.read_text()
        self.assertNotIn("LADA_DEFORM_CONV_BACKEND", source)
        self.assertIn('"bin/mioh-native-coreai-preview"', source)
        self.assertIn('"basicvsrpp-v1.2"', source)
        self.assertIn('"カスタム"', source)


if __name__ == "__main__":
    unittest.main()
