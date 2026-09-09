import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "packaging" / "macOS" / "standalone" / "MacChildProcessPipe.swift"
HARNESS = ROOT / "tests" / "swift" / "MacChildProcessPipeHarness.swift"
PIPELINE = ROOT / "packaging" / "macOS" / "standalone" / "MacHLSRealtimePipeline.swift"
REALTIME_PLAYER = ROOT / "packaging" / "macOS" / "standalone" / "RealtimePlayer.swift"
BUILD_SCRIPT = ROOT / "packaging" / "macOS" / "standalone" / "build_app.sh"
UNIVERSAL_BUILD_SCRIPT = (
    ROOT / "packaging" / "macOS" / "standalone" / "build_universal_app.sh"
)


@unittest.skipUnless(sys.platform == "darwin", "Darwin F_SETNOSIGPIPE is macOS-only")
class MacChildProcessPipeRuntimeTests(unittest.TestCase):
    def test_hls_worker_commands_use_the_scoped_no_sigpipe_helper(self):
        pipeline = PIPELINE.read_text(encoding="utf-8")
        player = REALTIME_PLAYER.read_text(encoding="utf-8")
        build_script = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("MacChildProcessPipe.prepare(inputPipe.fileHandleForWriting)", pipeline)
        self.assertIn("MacChildProcessPipe.write(line, to: handle)", pipeline)
        self.assertIn("MacChildProcessPipe.prepare(inputPipe.fileHandleForWriting)", player)
        self.assertIn("MacChildProcessPipe.write(line, to: handle)", player)
        app_compile = build_script.split('xcrun swiftc \\\n', 1)[1].split(
            '-o "$CONTENTS/MacOS/mioh"', 1
        )[0]
        self.assertIn('"$PACKAGE_DIR/MacChildProcessPipe.swift"', app_compile)

        native_preview_compile = build_script.split(
            "-D MIOH_NATIVE_PREVIEW_PIPELINE", 1
        )[1].split('-o "$RESOURCES/bin/mioh-native-coreai-preview"', 1)[0]
        self.assertIn(
            '"$PACKAGE_DIR/MacChildProcessPipe.swift"', native_preview_compile
        )

    def test_dedicated_and_universal_builds_share_the_protected_source_wiring(self):
        build_script = BUILD_SCRIPT.read_text(encoding="utf-8")
        universal_script = UNIVERSAL_BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'COREAI_DISTRIBUTION="${COREAI_DISTRIBUTION:-dedicated}"', build_script
        )
        self.assertIn('export COREAI_DISTRIBUTION="portable"', universal_script)
        self.assertIn('exec "$PACKAGE_DIR/build_app.sh"', universal_script)

    def test_closed_worker_stdin_returns_false_without_sigpipe_exit(self):
        swiftc = shutil.which("swiftc")
        if not swiftc:
            xcrun = shutil.which("xcrun")
            if xcrun:
                swiftc = subprocess.check_output(
                    [xcrun, "--find", "swiftc"], text=True
                ).strip()
        if not swiftc:
            self.skipTest("Swift compiler is required")

        with tempfile.TemporaryDirectory(prefix="mioh-child-pipe-") as directory:
            executable = Path(directory) / "pipe-harness"
            module_cache = Path(directory) / "module-cache"
            module_cache.mkdir()
            environment = os.environ.copy()
            environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
            environment["SWIFT_MODULE_CACHE_PATH"] = str(module_cache)
            build = subprocess.run(
                [
                    swiftc,
                    "-parse-as-library",
                    str(HELPER),
                    str(HARNESS),
                    "-o",
                    str(executable),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
                env=environment,
            )
            self.assertEqual(
                build.returncode,
                0,
                f"child pipe harness did not compile:\n{build.stdout}",
            )
            completed = subprocess.run(
                [str(executable)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )

        self.assertEqual(
            completed.returncode,
            0,
            "SIGPIPE killed the harness or a race assertion failed "
            f"(returncode={completed.returncode}):\n{completed.stdout}",
        )
        self.assertIn("ok mac child pipe SIGPIPE regression", completed.stdout)


if __name__ == "__main__":
    unittest.main()
