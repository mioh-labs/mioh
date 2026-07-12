import unittest
import plistlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "MiohApp.swift"
BUILD_SCRIPT = ROOT / "packaging" / "macOS" / "standalone" / "build_app.sh"
INFO_PLIST = ROOT / "packaging" / "macOS" / "standalone" / "Info.plist"


class StandaloneAppOptionTests(unittest.TestCase):
    def test_product_is_named_mioh(self):
        self.assertTrue(APP_SOURCE.is_file(), "MiohApp.swift must be the app entry source")
        source = APP_SOURCE.read_text()
        build_script = BUILD_SCRIPT.read_text()
        with INFO_PLIST.open("rb") as handle:
            info = plistlib.load(handle)

        self.assertIn('Text("mioh")', source)
        self.assertIn('PathSettingRow(title: "mioh一時フォルダ"', source)
        self.assertIn("struct MiohStandaloneApp: App", source)
        self.assertEqual(info["CFBundleDisplayName"], "mioh")
        self.assertEqual(info["CFBundleName"], "mioh")
        self.assertEqual(info["CFBundleExecutable"], "mioh")
        self.assertEqual(info["CFBundleIdentifier"], "com.okatti.lada.coreai")
        self.assertIn('APP="$BUILD_DIR/mioh.app"', build_script)
        self.assertIn('-o "$CONTENTS/MacOS/mioh"', build_script)
        self.assertIn('DMG="$BUILD_DIR/mioh-0.11.0-unsigned.dmg"', build_script)
        self.assertIn('--volumeName "mioh"', build_script)
        self.assertIn('ditto "$APP" "$DMG_ROOT/mioh.app"', build_script)

    def test_gui_exposes_all_processing_options(self):
        source = APP_SOURCE.read_text()
        expected_options = {
            "--input", "--output", "--temp-dir", "--ffmpeg-temp-dir",
            "--lada-temp-dir", "--parallel-workers", "--executor",
            "--segment-duration", "--segment-count", "--merge-encoder",
            "--delete-segments", "--keep-temp", "--force-split", "--device",
            "--fp16", "--no-fp16", "--encoding-preset", "--encoder",
            "--encoder-options", "--bitrate-multiplier", "--quality", "--qmin",
            "--qmax", "--fps", "--pre-fps-conversion", "--mp4-fast-start",
            "--auto-optimize", "--no-auto-optimize", "--mosaic-restoration-model",
            "--max-clip-length", "--restore-max-frames",
            "--restore-sharpen-strength", "--restore-detail-boost",
            "--restore-blend-feather", "--restore-texture-mix",
            "--restore-smooth-strength", "--restore-effect-upscale",
            "--restore-roi-enhancer", "--restore-roi-enhancer-model-path",
            "--restore-roi-enhancer-scale", "--restore-roi-enhancer-strength",
            "--restore-roi-enhancer-tile", "--mosaic-detection-model",
            "--mosaic-detection-empty-lookahead", "--detect-face-mosaics",
            "--no-detect-face-mosaics", "--memory-cleanup-interval",
            "--cleanup-trigger-gb", "--mps-memory-fraction", "--log-mps-memory",
            "--overwrite",
        }

        missing = sorted(option for option in expected_options if option not in source)
        self.assertEqual(missing, [])

    def test_app_bundles_parallel_processor(self):
        script = BUILD_SCRIPT.read_text()
        self.assertIn("process_video_parallel.py", script)

    def test_standalone_app_uses_mioh_icon(self):
        icon = ROOT / "lada" / "gui" / "icons" / "mioh-icon.png"
        script = BUILD_SCRIPT.read_text()

        self.assertTrue(icon.is_file())
        self.assertEqual(icon.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn('SOURCE_ICON="$ROOT/lada/gui/icons/mioh-icon.png"', script)

    def test_app_replaces_structured_progress_rows(self):
        source = APP_SOURCE.read_text()

        self.assertIn('result["LADA_APP_PROGRESS"] = "1"', source)
        self.assertIn("struct AppProgressEvent: Decodable", source)
        self.assertIn('"@@LADA_PROGRESS@@"', source)
        self.assertIn("activeProgress", source)
        self.assertIn("logHistory", source)
        self.assertIn("rebuildVisibleLog()", source)


if __name__ == "__main__":
    unittest.main()
