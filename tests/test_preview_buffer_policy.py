import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "packaging" / "macOS" / "standalone" / "PreviewBufferPolicy.swift"


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("xcrun"), "Swift toolchain required")
class PreviewBufferPolicyTests(unittest.TestCase):
    def test_selected_duration_and_fast_rebuffer_policy(self):
        harness = r'''
        @main
        struct PolicyTest {
          static func main() {
            precondition(PreviewBufferPolicy.requiredSeconds(
              selectedBufferLimit: 60, generationHasStarted: false, shortenRebuffer: true
            ) == 60)
            precondition(PreviewBufferPolicy.requiredSeconds(
              selectedBufferLimit: 60, generationHasStarted: true, shortenRebuffer: false
            ) == 60)
            precondition(PreviewBufferPolicy.requiredSeconds(
              selectedBufferLimit: 60, generationHasStarted: true, shortenRebuffer: true
            ) == 4)
            precondition(PreviewBufferPolicy.requiredSeconds(
              selectedBufferLimit: 2, generationHasStarted: true, shortenRebuffer: true
            ) == 2)
            precondition(!PreviewBufferPolicy.canStart(
              bufferedSeconds: 59.9, selectedBufferLimit: 60,
              generationHasStarted: false, shortenRebuffer: false,
              endOfFile: false, hasQueuedSegments: true
            ))
            precondition(PreviewBufferPolicy.canStart(
              bufferedSeconds: 60, selectedBufferLimit: 60,
              generationHasStarted: false, shortenRebuffer: false,
              endOfFile: false, hasQueuedSegments: true
            ))
            precondition(PreviewBufferPolicy.canStart(
              bufferedSeconds: 1.5, selectedBufferLimit: 60,
              generationHasStarted: false, shortenRebuffer: false,
              endOfFile: true, hasQueuedSegments: true
            ))
            precondition(!PreviewBufferPolicy.canStart(
              bufferedSeconds: 0, selectedBufferLimit: 60,
              generationHasStarted: false, shortenRebuffer: false,
              endOfFile: true, hasQueuedSegments: false
            ))
            print("ok")
          }
        }
        '''
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main = directory / "PolicyTest.swift"
            binary = directory / "policy-test"
            main.write_text(harness)
            subprocess.run(
                ["xcrun", "swiftc", str(POLICY), str(main), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            )
        self.assertEqual(result.stdout.strip(), "ok")
