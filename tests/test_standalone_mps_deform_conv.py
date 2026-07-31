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

    def test_only_the_portable_build_installs_mps_deform_conv(self):
        script = BUILD_SCRIPT.read_text()

        # torch and the deform-conv extension travel with the bundled Python
        # runtime, which only the portable/universal package carries.
        self.assertIn("VENDORED_MPS_DEFORM_CONV", script)
        self.assertIn("mps-deform-conv.txt", script)
        self.assertIn("verify_mps_deform_conv.py", script)
        self.assertIn('if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 ]]', script)
        for guarded in [
            'ditto "$VENDORED_MPS_DEFORM_CONV" "$MPS_DEFORM_BUILD_SOURCE"',
            'cp "$VENDORED_MPS_DEFORM_CONV/LICENSE"',
        ]:
            self.assertIn(guarded, script)
        # Nothing may reach the app outside that gate.
        prologue = script.split(
            'if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 ]]; then', 1
        )[0]
        self.assertNotIn("$RESOURCES/runtime", prologue)

    def test_gui_selects_the_deform_conv_backend_for_the_python_engine(self):
        source = APP_SOURCE.read_text()

        # The override only reaches the bundled interpreter's environment.
        self.assertIn(
            'result["LADA_DEFORM_CONV_BACKEND"] = "mps_deform_conv"', source
        )
        self.assertNotIn(
            'nativeEnvironment["LADA_DEFORM_CONV_BACKEND"]', source
        )
        self.assertIn('"bin/mioh-native-coreai-preview"', source)
        self.assertIn('"basicvsrpp-v1.2"', source)
        self.assertIn('"カスタム"', source)


if __name__ == "__main__":
    unittest.main()
