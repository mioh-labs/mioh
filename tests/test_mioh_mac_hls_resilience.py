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

    def test_vod_pipeline_prefetches_next_segment_while_restoring_current_window(self):
        for contract in [
            "private actor LocalSegmentCache",
            "static let shared = LocalSegmentCache",
            "hls-segment-cache",
            "LocalSegmentCache.shared.materialize(",
            "HLS区間\\(mediaSegment.sequence)をローカルキャッシュから再利用しました",
            "private var prefetchDownloader: IPadHLSResourceDownloader?",
            "private var prefetchDownloaders: [IPadHLSResourceDownloader] = []",
            "var vodPrefetchTasks: [Int64: Task<Void, Never>] = [:]",
            "var vodPrefetchResults: [Int64: Result<RestorationSource, Error>] = [:]",
            "var vodPrefetchGeneration = UUID()",
            "var vodPrefetchCompletionCount = 0",
            "let maximumVODPrefetchSegments = 24",
            "let maximumVODPrefetchDownloads = 3",
            "private let vodRestoreBatchCoreSegments = 6",
            "coreStartIndex + vodRestoreBatchCoreSegments + 1",
            "coreEndIndex: coreEndIndex",
            "restorationWindow.removeFirst(vodRestoreBatchCoreSegments)",
            "func discardVODPrefetchResult(",
            "func startVODPrefetchIfPossible()",
            "Set(vodPrefetchTasks.keys).union(vodPrefetchResults.keys)",
            "maximumVODPrefetchDownloads - vodPrefetchTasks.count",
            "maximumVODPrefetchSegments - scheduled.count",
            "let availableSlots = min(availableDownloadSlots, availableDepthSlots)",
            "vodPrefetchTasks.isEmpty && vodPrefetchResults.isEmpty",
            "HLS先読み開始: 区間\\(candidate.sequence)から最大\\(maximumVODPrefetchSegments)本",
            "Task.detached(priority: .userInitiated)",
            "await MainActor.run",
            "HLS先読み完了",
            "materialized.cacheHit ? \"cache\" : \"network\"",
            "\" / 在庫\\(vodPrefetchResults.count)/\\(maximumVODPrefetchSegments)\\n\"",
            "vodPrefetchCompletionCount.isMultiple(of: 10)",
            "HLS区間準備: 出力\\(nextOutputSequence)",
            "連結+検証",
            "workerOutputSequenceStart",
            "emittedSegmentCount",
            "HLS復元worker: 出力\\(workerOutputSequenceStart)",
            "firstSegmentElapsed",
            "vodPrefetchTasks[candidate.sequence] = nil",
            "vodPrefetchResults[candidate.sequence] = result",
            "startVODPrefetchIfPossible()",
            "consumeVODPrefetch(",
            "HLS先読みヒット",
            "prefetchDownloader?.cancel()",
            "for downloader in prefetchDownloaders { downloader.cancel() }",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, self.source)

        consume_position = self.source.index("consumeVODPrefetch(")
        restore_position = self.source.index("restoreWindow(", consume_position)
        self.assertLess(consume_position, restore_position)

    def test_vod_segment_cache_survives_seek_generation_but_session_files_remain_disposable(self):
        for contract in [
            "func cachedCopy(",
            "FileManager.default.linkItem(",
            "FileManager.default.copyItem(",
            "func importDownloadedSegment(",
            "func pruneIfNeeded(",
            "maximumBytes",
            "resource.url.absoluteString",
            "byteRange.offset",
            "byteRange.length",
            "discontinuitySequence",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, self.source)

        cache_type = self.source.split("private actor LocalSegmentCache", 1)[1]
        cache_type = cache_type.split("\n  private struct RestorationSource", 1)[0]
        self.assertIn("return Materialized(url: cached, cacheHit: true)", cache_type)
        self.assertIn("return Materialized(url: downloadedURL, cacheHit: false)", cache_type)
        self.assertNotIn("removeItem(at: outputURL)", cache_type)


if __name__ == "__main__":
    unittest.main()
