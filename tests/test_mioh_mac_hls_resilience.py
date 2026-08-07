import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = (
    ROOT / "packaging" / "macOS" / "standalone" / "MacHLSRealtimePipeline.swift"
)


class MacHLSResilienceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = PIPELINE.read_text(encoding="utf-8")

    def test_signed_resource_failure_re_resolves_original_or_master_url(self):
        for contract in [
            "resolveRefreshedSource(",
            "preferOriginalURL: true",
            "source.playbackURL",
            "source.submittedURL",
            "source.requestContext?.referer",
            "requestContext: resolvedSource.requestContext ?? source.requestContext",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, self.source)

    def test_live_window_jump_flushes_then_emits_discontinuity(self):
        reset = self.source.split("private func resetForLiveWindowJump(", 1)[1]
        reset = reset.split("\n  private func ", 1)[0]
        self.assertLess(reset.index("flushWindow("), reset.index("removeMaterializedSources("))
        self.assertLess(
            reset.index("removeMaterializedSources("),
            reset.index("emit(.discontinuity(position: position))"),
        )
        self.assertIn("case discontinuity(position: Double)", self.source)

    def test_encrypted_refresh_remains_rejected(self):
        self.assertIn("if case .encryptedPlaylist = error { throw error }", self.source)
        self.assertIn("isEncryptedPlaylistFailure(error) { throw error }", self.source)

    def test_cancellation_has_awaitable_cleanup_completion(self):
        self.assertIn("nonisolated func cancelAndWait() async", self.source)
        self.assertIn("completionWaiters", self.source)
        self.assertIn("defer { completeRun() }", self.source)


if __name__ == "__main__":
    unittest.main()
