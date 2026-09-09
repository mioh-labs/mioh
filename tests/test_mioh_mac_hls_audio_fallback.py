import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAYER = ROOT / "packaging" / "macOS" / "standalone" / "RealtimePlayer.swift"
HARNESS = ROOT / "tests" / "swift" / "HLSAudioFallbackStateHarness.swift"


class MacHLSAudioFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.player = PLAYER.read_text(encoding="utf-8")

    @staticmethod
    def function(source, signature):
        return source.split(signature, 1)[1].split("\n  private func ", 1)[0]

    def test_runtime_late_callback_permutations_preserve_restored_playback(self):
        if sys.platform != "darwin":
            self.skipTest("Swift concurrency harness requires macOS")
        swiftc = shutil.which("swiftc")
        if not swiftc:
            xcrun = shutil.which("xcrun")
            if xcrun:
                swiftc = subprocess.check_output(
                    [xcrun, "--find", "swiftc"], text=True
                ).strip()
        if not swiftc:
            self.skipTest("Swift compiler is required")

        with tempfile.TemporaryDirectory(prefix="mioh-hls-audio-fallback-") as directory:
            executable = Path(directory) / "fallback-harness"
            module_cache = Path(directory) / "module-cache"
            module_cache.mkdir()
            compiler_environment = os.environ.copy()
            compiler_environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
            compiler_environment["SWIFT_MODULE_CACHE_PATH"] = str(module_cache)
            build = subprocess.run(
                [
                    swiftc,
                    "-parse-as-library",
                    str(HARNESS),
                    "-o",
                    str(executable),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
                env=compiler_environment,
            )
            self.assertEqual(
                build.returncode,
                0,
                f"audio fallback harness did not compile:\n{build.stdout}",
            )
            completed = subprocess.run(
                [str(executable)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"audio fallback harness crashed or hung:\n{completed.stdout}",
        )
        self.assertIn("ok fallback permutations=120", completed.stdout)

    def test_production_fallback_is_idempotent_and_keeps_queue_producer_alive(self):
        degrade = self.function(
            self.player,
            "private func degradeHLSSourcePlayback(",
        )
        self.assertIn("!hlsRestoredClockFallbackActive", degrade)
        self.assertLess(
            degrade.index("hlsRestoredClockFallbackActive = true"),
            degrade.index("sourceItemStatusObservation?.invalidate()"),
        )
        self.assertIn("sourcePlayer.pause()", degrade)
        self.assertIn("resumeIfBuffered()", degrade)
        for forbidden in (
            "sourcePlayer.replaceCurrentItem(with: nil)",
            "hlsMediaProxy?.stop()",
            "hlsProducer?.cancel()",
            "hlsProductionTask?.cancel()",
            "clearRestoredQueue",
            "state = .failed",
            "fail(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, degrade)

    def test_every_late_source_callback_has_identity_or_idempotence_guard(self):
        update = self.function(
            self.player,
            "private func updateHLSPlaybackState(",
        )
        for contract in (
            "self.generation == generation",
            "sourcePlayer.currentItem === item",
            "!hlsRestoredClockFallbackActive",
        ):
            self.assertIn(contract, update)

        observers = self.function(
            self.player,
            "private func installHLSPlaybackObservers(",
        )
        failed_to_end = observers.split(
            ".AVPlayerItemFailedToPlayToEndTime", 1
        )[1].split("hlsNotificationTokens.append(failedToEnd)", 1)[0]
        self.assertIn("self.generation == generation", failed_to_end)
        self.assertIn("self.sourcePlayer.currentItem === item", failed_to_end)

        stalled = observers.split(
            ".AVPlayerItemPlaybackStalled", 1
        )[1].split("hlsNotificationTokens.append(stalled)", 1)[0]
        self.assertIn("self.generation == generation", stalled)
        self.assertIn("self.sourcePlayer.currentItem === item", stalled)
        self.assertIn("!self.hlsRestoredClockFallbackActive", stalled)

        # Match MiohRemote: AVAsset's early HLS track snapshot is not a valid
        # preflight. Muxed and alternate audio can appear only after media
        # selection settles, so production must not run this callback at all.
        self.assertNotIn("validateHLSSourceAudio(", self.player)
        self.assertNotIn("loadTracks(withMediaType: .audio)", self.player)

        start_hls = self.player.split("func startHLS(", 1)[1].split(
            "\n  private func ", 1
        )[0]
        proxy_callback = start_hls.split(
            "let createdProxy = IPadAuthenticatedMediaProxy", 1
        )[1].split("proxy = createdProxy", 1)[0]
        self.assertIn("self.generation == startingGeneration", proxy_callback)
        self.assertIn("degradeHLSSourcePlayback(", proxy_callback)

    def test_producer_events_continue_to_enqueue_after_audio_fallback(self):
        production = self.function(
            self.player,
            "private func handleHLSProductionEvent(",
        )
        segment = production.split("case .segment(", 1)[1].split(
            "case .progress", 1
        )[0]
        self.assertIn("enqueue(segment)", segment)
        self.assertIn("resumeIfBuffered()", segment)
        self.assertNotIn("guard !hlsRestoredClockFallbackActive", segment)

        resume = self.function(self.player, "private func resumeIfBuffered(")
        self.assertIn("canStartHLSWithRestoredClockFallback", resume)
        start = self.function(
            self.player,
            "private func startPlayersFromCurrentPosition()",
        )
        self.assertIn("shouldPreferRestoredHLSPlayback", start)
        self.assertIn("restoredPlayer.play()", start)

    def test_fallback_reason_is_source_failure_not_unproven_audio_absence(self):
        self.assertNotIn("元動画の音声を利用できないため", self.player)
        self.assertIn(
            "元動画側の再生を継続できないため、復元映像のみ再生中（音声なし）",
            self.player,
        )

        set_volume = self.player.split("func setVolume(", 1)[1].split(
            "\n  func ", 1
        )[0]
        set_muted = self.player.split("func setMuted(", 1)[1].split(
            "\n  func ", 1
        )[0]
        for setter in (set_volume, set_muted):
            self.assertIn("hlsRestoredClockFallbackActive", setter)
            self.assertIn("? 0", setter)


if __name__ == "__main__":
    unittest.main()
