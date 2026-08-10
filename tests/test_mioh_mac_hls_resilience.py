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

    def test_vod_prefetch_starts_only_after_first_restored_output(self):
        prefetch = self.source.split(
            "func startVODPrefetchIfPossible()", 1
        )[1].split("\n    func consumeVODPrefetch", 1)[0]
        self.assertIn("guard hasMaterializedCurrentVODSegment else { return }", prefetch)
        self.assertIn("guard hasRestoredAnyWindow else { return }", prefetch)
        self.assertIn("guard !vodPrefetchPauseInProgress else { return }", prefetch)
        self.assertIn("guard Date() >= vodPrefetchSuspendedUntil else { return }", prefetch)

        current = self.source.split(
            "if timelineEnd <= startSeconds { continue }", 1
        )[1].split("if let previous = restorationWindow.last", 1)[0]
        materialized = current.index("guard let localURL")
        proved = current.index("hasMaterializedCurrentVODSegment = true")
        started = current.index("startVODPrefetchIfPossible()")
        self.assertLess(materialized, proved)
        self.assertLess(proved, started)
        self.assertNotIn(
            "startVODPrefetchIfPossible()",
            current[:materialized],
            "speculative requests started before the current VOD segment succeeded",
        )

        # State model for startup, first-output and rate-limit gating.
        def may_prefetch(current_succeeded, restored_output_exists, now, suspended_until):
            return (
                current_succeeded
                and restored_output_exists
                and now >= suspended_until
            )

        self.assertFalse(may_prefetch(False, False, 0.0, 0.0))
        self.assertFalse(may_prefetch(True, False, 0.0, 0.0))
        self.assertTrue(may_prefetch(True, True, 0.0, 0.0))
        self.assertFalse(may_prefetch(True, True, 4.0, 8.0))
        self.assertTrue(may_prefetch(True, True, 8.0, 8.0))

        vod_restore = self.source.split(
            "let coreStartIndex = hasRestoredAnyWindow ? 1 : 0", 1
        )[1].split("\n        }\n      }", 1)[0]
        self.assertLess(
            vod_restore.index("hasRestoredAnyWindow = true"),
            vod_restore.index("startVODPrefetchIfPossible()"),
        )

    def test_http_429_downgrades_once_then_waits_without_terminal_retry_limit(self):
        current = self.source.split(
            "var ordinaryRetryCount = 0", 1
        )[1].split("guard let localURL", 1)[0]
        rate_limit_branch = current.split("if Self.isHTTPRateLimit(error)", 1)[1]
        rate_limit_branch = rate_limit_branch.split(
            "consecutiveSegmentRateLimits = 0", 1
        )[0]

        for loop_contract in ["while true", "try checkCancellation()"]:
            with self.subTest(loop_contract=loop_contract):
                self.assertIn(loop_contract, current)

        for contract in [
            "!playlist.isLive, !didAttemptVariantFallback,",
            "!sameOriginVariantFallbackRejected",
            "sameOriginVariantFallbackRejected = true",
            "resolveNextHLSVariant(for: activeSource)",
            "Self.sharesPrimaryMediaOrigin(",
            "HLSの低いvariantも同じ区間配信元を使用するため、",
            "品質を切り替えず待機します",
            "pendingVariantFallbackSource = fallbackSource",
            "throw ProductionError.variantFallbackPrepared",
            "Self.variantDescription(activeSource)",
            "Self.variantDescription(fallbackSource)",
            "Self.rateLimitRetryDelay(",
            "await suspendVODPrefetchAfterRateLimit(seconds: retryDelay)",
            "同じ区間を再取得します（バッファ中）",
            "try await sleep(seconds: retryDelay)",
            "continue",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, rate_limit_branch)
        self.assertNotIn("guard retryIndex < 3", rate_limit_branch)
        self.assertNotIn("resolveRefreshedSource(", rate_limit_branch)

        live_branch = rate_limit_branch.split("if playlist.isLive", 1)[1]
        self.assertIn("nextMediaSequence = mediaSegment.sequence", live_branch)
        self.assertIn("lastRefresh = Date.distantPast", live_branch)
        self.assertIn("continue productionLoop", live_branch)

        ordinary = current.split(
            "consecutiveSegmentRateLimits = 0", 1
        )[1]
        self.assertIn("guard ordinaryRetryCount < 3 else { break }", ordinary)
        self.assertIn("ordinaryRetryCount += 1", ordinary)

        for contract in [
            "httpStatusCode(in error: Error)",
            "httpStatusCode(in: error) == 429",
            "shouldRefreshVODSource(after error: Error)",
            "return [401, 403, 404, 410].contains(statusCode)",
            "guard Self.shouldRefreshVODSource(after: error) else",
            "forConsecutiveFailure failureCount: Int",
            "let boundedExponent = max(0, min(5, failureCount - 1))",
            "return min(30, 1.5 * pow(2, Double(boundedExponent)))",
            "func takePendingVariantFallbackSource()",
            "private nonisolated static func sharesPrimaryMediaOrigin(",
            "private nonisolated static func primaryMediaOriginKey(",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, self.source)

    def test_webkit_relay_post_dispatch_failure_uses_polite_retry_and_stops_prefetch(self):
        current = self.source.split(
            "var browserRelayRetryCount = 0", 1
        )[1].split("guard let localURL", 1)[0]
        relay = current.split(
            "if Self.isBrowserRelayAttemptedUnavailable(error)", 1
        )[1].split("if Self.isHTTPRateLimit(error)", 1)[0]
        for contract in [
            "guard browserRelayRetryCount < 3 else { break }",
            "browserRelayRetryCount += 1",
            "Self.rateLimitRetryDelay(",
            'reason: "WebKit HLS通信の再試行待ち"',
            "同じ区間を再取得します（バッファ中）",
            "try await sleep(seconds: retryDelay)",
            "continue",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, relay)
        self.assertNotIn("resolveRefreshedSource(", relay)

        helper = self.source.split(
            "private nonisolated static func isBrowserRelayAttemptedUnavailable(", 1
        )[1].split("private nonisolated static func rateLimitRetryDelay(", 1)[0]
        self.assertIn("error as? IPadHLSResourceLoadingError", helper)
        self.assertIn("loadingError == .attemptedUnavailable", helper)

        prefetch = self.source.split(
            "let browserRelayUnavailable: Bool", 1
        )[1].split("func consumeVODPrefetch(", 1)[0]
        self.assertIn("Self.isBrowserRelayAttemptedUnavailable(error)", prefetch)
        self.assertIn("else if browserRelayUnavailable", prefetch)
        self.assertIn("Date().addingTimeInterval(8)", prefetch)
        self.assertIn('"WebKit HLS通信の再試行待ち"', prefetch)

        refresh_statuses = {401, 403, 404, 410}
        self.assertNotIn(429, refresh_statuses)
        self.assertTrue(all(status in refresh_statuses for status in (401, 403, 404, 410)))
        self.assertTrue(all(status not in refresh_statuses for status in (408, 429, 500, 503)))
        delays = [min(30.0, 1.5 * (2 ** min(5, attempt - 1))) for attempt in range(1, 8)]
        self.assertEqual(delays, [1.5, 3.0, 6.0, 12.0, 24.0, 30.0, 30.0])

    def test_refresh_candidate_loop_stops_immediately_on_http_429(self):
        refresh = self.source.split(
            "private func resolveRefreshedSource(", 1
        )[1].split("\n  private func refreshCandidates", 1)[0]
        self.assertIn(
            "IPadMediaURLResolver(\n          resourceLoader: resourceLoader\n        ).resolve(",
            refresh,
        )
        resolver_catch = refresh.split(
            "catch let error as IPadMediaURLResolverError", 1
        )[1].split("} catch {", 1)[0]
        self.assertIn("if Self.isHTTPRateLimit(error) { throw error }", resolver_catch)
        self.assertLess(
            resolver_catch.index("if Self.isHTTPRateLimit(error) { throw error }"),
            resolver_catch.index("lastFailure = error"),
        )

        # Model the candidate loop: a 429 must consume exactly one candidate,
        # while an ordinary invalid candidate can continue to the next URL.
        def attempted_candidates(statuses):
            attempts = 0
            for status in statuses:
                attempts += 1
                if status == 429:
                    break
            return attempts

        self.assertEqual(attempted_candidates([429, 200, 200]), 1)
        self.assertEqual(attempted_candidates([404, 200]), 2)

    def test_rate_limit_pauses_refill_and_awaits_speculative_task_cancellation(self):
        pause = self.source.split(
            "func suspendVODPrefetchAfterRateLimit(", 1
        )[1].split("\n    func discardStaleVODPrefetch", 1)[0]
        for contract in [
            "let hadPrefetchActivity =",
            "triggeredByPrefetch: Bool = false",
            "triggeredByPrefetch || !vodPrefetchTasks.isEmpty",
            "!vodPrefetchTasks.isEmpty",
            "vodPrefetchSuspendedUntil = max(",
            "if vodPrefetchPauseInProgress { return }",
            "vodPrefetchPauseInProgress = true",
            "await cancelVODPrefetchAndWait(",
            "resetDownloader: true",
            "discardCompletedResults: false",
            "case .failure = result",
            "vodPrefetchResults[sequence] = nil",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, pause)
        self.assertNotIn("while vodPrefetchPauseInProgress", pause)
        self.assertIn(
            "if hadPrefetchActivity,\n        vodPrefetchSuspendedUntil > previousSuspension",
            pause,
        )

        cancel = self.source.split(
            "func cancelVODPrefetchAndWait(", 1
        )[1].split("\n    func suspendVODPrefetchAfterRateLimit", 1)[0]
        self.assertLess(cancel.index("vodPrefetchGeneration = UUID()"), cancel.index("task.cancel()"))
        self.assertLess(cancel.index("task.cancel()"), cancel.index("await task.value"))
        self.assertLess(cancel.index("await task.value"), cancel.index("vodPrefetchDownloader = nil"))

        completion = self.source.split(
            "if rateLimited {", 1
        )[1].split("if rateLimited {", 1)[0]
        completion_rate_limit = completion.split("} else {", 1)[0]
        self.assertIn("vodPrefetchSuspendedUntil = max(", completion_rate_limit)
        self.assertNotIn("startVODPrefetchIfPossible()", completion_rate_limit)
        self.assertIn("let shouldSuspendPrefetch = await MainActor.run", self.source)
        self.assertIn("return false", completion)
        self.assertIn("if shouldSuspendPrefetch", completion)
        self.assertIn("triggeredByPrefetch: true", completion)

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

    def test_cancellation_awaits_prefetch_and_bounds_worker_tree_retirement(self):
        for contract in [
            "func cancelVODPrefetchAndWait(",
            "let tasks = Array(vodPrefetchTasks.values)",
            "for task in tasks { task.cancel() }",
            "for task in tasks { await task.value }",
            "await cancelVODPrefetchAndWait(resetDownloader: true)",
            "func discardStaleVODPrefetch(before sequence: Int64) async",
            "for task in staleTasks { await task.value }",
            "private var activeProcessRetirementTask: Task<Void, Never>?",
            "private func scheduleProcessRetirement(",
            "private func retireProcess(_ process: Process) async",
            "nonisolated private static func retireProcessTree(",
            "kill(-processIdentifier, SIGTERM)",
            "kill(-processIdentifier, SIGKILL)",
            "kill(processIdentifier, SIGKILL)",
            "for _ in 0..<20 where Self.processTreeIsAlive(processIdentifier)",
            "private static func processTreeIsAlive(",
            "if !(await waitForProcessExit(process))",
            "await retireProcess(process)",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, self.source)

        cancel_prefetch = self.source.split(
            "func cancelVODPrefetchAndWait(", 1
        )[1].split("\n    func ensureVODPrefetchDownloader", 1)[0]
        self.assertLess(
            cancel_prefetch.index("vodPrefetchGeneration = UUID()"),
            cancel_prefetch.index("for task in tasks { task.cancel() }"),
        )
        self.assertLess(
            cancel_prefetch.index("for task in tasks { task.cancel() }"),
            cancel_prefetch.index("for task in tasks { await task.value }"),
        )

        worker_error = self.source.split('case "segment":', 1)[1]
        worker_error = worker_error.split("await waitForProcessExit(process)", 1)[0]
        self.assertIn("await retireProcess(process)", worker_error)

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
            "private let vodInitialRestoreBatchCoreSegments = 2",
            "private let vodMinimumSteadyBatchCoreSegments = 2",
            "private let vodSteadyRestoreBatchCoreSegments = 18",
            "private let vodSteadyRestoreBatchTargetSeconds = 36.0",
            "private let vodSteadyRestoreBatchMaximumBytes: Int64",
            "await vodCoreSegmentCountIfReady(",
            "coreEndIndex: coreEndIndex",
            "let retirementCount = hasRestoredAnyWindow",
            ": max(0, coreSegmentCount - 1)",
            "restorationWindow.removeFirst(retirementCount)",
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
            ".cachesDirectory",
            ".userDomainMask",
            "struct Manifest: Codable",
            "index-v2.json",
            "timeToLive",
            "func loadIfNeeded(",
            "contentsOfDirectory(",
            "pruneExpiredAndOversized(",
            "func persistManifest(",
            "func cachedCopy(",
            "FileManager.default.linkItem(",
            "FileManager.default.copyItem(",
            "func importDownloadedSegment(",
            "allowPersistentCache: Bool",
            "guard allowPersistentCache, Self.isPubliclyCacheable(segment) else",
            "(activeSource.requestContext ?? source.requestContext) == nil",
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
        self.assertNotIn(
            'resources.appendingPathComponent(\n      "hls-segment-cache"',
            self.source,
        )

    def test_persistent_segment_cache_never_replays_browser_credentials(self):
        cache = self.source.split("private actor LocalSegmentCache", 1)[1]
        cache = cache.split("\n  private struct RestorationSource", 1)[0]
        self.assertIn("allowPersistentCache: Bool", cache)
        self.assertIn(
            "guard allowPersistentCache, Self.isPubliclyCacheable(segment) else",
            cache,
        )
        bypass = cache.split(
            "guard allowPersistentCache, Self.isPubliclyCacheable(segment) else",
            1,
        )[1]
        bypass = bypass.split("let key = Self.cacheKey", 1)[0]
        self.assertIn("downloader.materialize(", bypass)
        self.assertIn("return Materialized(url: downloadedURL, cacheHit: false)", bypass)
        self.assertNotIn("cachedCopy(", bypass)
        self.assertNotIn("importDownloadedSegment(", bypass)

        # The active source can be refreshed without carrying its context value,
        # so both the refreshed and original browser contexts must be absent.
        context_gate = (
            "(activeSource.requestContext ?? source.requestContext) == nil"
        )
        self.assertGreaterEqual(self.source.count(context_gate), 2)
        self.assertGreaterEqual(self.source.count("allowPersistentCache:"), 3)
        self.assertIn("private static func isPubliclyCacheable(", cache)
        self.assertIn("components.percentEncodedQuery?.isEmpty", cache)
        self.assertIn(
            "Browser credential contexts can make identical stable URLs return",
            cache,
        )

        # Persisted cache metadata must not double as a plaintext browsing
        # history. Version 2 hashes the composite URL/range/discontinuity
        # identity before using it as the manifest key; version 1 is not loaded.
        self.assertIn("import CryptoKit", self.source)
        self.assertIn('private let manifestName = "index-v2.json"', cache)
        self.assertIn("if manifest?.version == 2", cache)
        self.assertIn("version: 2", cache)
        self.assertNotIn('manifestName = "index-v1.json"', cache)
        cache_key = cache.split("private static func cacheKey(", 1)[1]
        cache_key = cache_key.split(
            "\n    private static func isPubliclyCacheable", 1
        )[0]
        self.assertIn("let identity =", cache_key)
        self.assertIn("SHA256.hash(data: Data(identity.utf8))", cache_key)
        self.assertNotIn("return identity", cache_key)

    def test_restored_output_credit_bounds_seconds_items_and_bytes(self):
        for contract in [
            "struct OutputBufferLimits: Sendable, Equatable",
            "seconds: 60",
            "items: 48",
            "bytes: 512 * 1_024 * 1_024",
            "func updateOutputBufferLimits(",
            "func acknowledgeOutputConsumed(through sequence: Int)",
            "retainedOutputCredits.filter { $0.key <= sequence }",
            "retainedOutputSeconds = max(0, retainedOutputSeconds - credit.seconds)",
            "retainedOutputBytes = max(0, retainedOutputBytes - credit.bytes)",
            "private func waitForOutputCredit(",
            "retainedOutputCredits.count + 1 <= outputBufferLimits.items",
            "let oneSegmentDurationSlack = max(0.001, incomingSeconds)",
            "outputBufferLimits.seconds + oneSegmentDurationSlack + 0.001",
            "retainedOutputBytes + max(0, bytes) <= outputBufferLimits.bytes",
            "HLS復元出力待機",
            "HLS復元出力再開",
            "RTF ",
            "credit待機",
            "credit ",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, self.source)

        worker_segment = self.source.split('case "segment":', 1)[1]
        worker_segment = worker_segment.split('case "progress":', 1)[0]
        wait = worker_segment.index("try await waitForOutputCredit(")
        copy = worker_segment.index("try await mediaFileWorker.copyReplacing(")
        retain = worker_segment.index("retainOutputCredit(")
        emit = worker_segment.index("emit(")
        self.assertLess(wait, copy)
        self.assertLess(copy, retain)
        self.assertLess(retain, emit)

        cancel = self.source.split("private func requestCancellation()", 1)[1]
        cancel = cancel.split("\n  private func completeRun()", 1)[0]
        complete = self.source.split("private func completeRun()", 1)[1]
        complete = complete.split("\n  private func waitForOutputCredit(", 1)[0]
        self.assertIn("resumeOutputCreditWaiters()", cancel)
        self.assertIn("resumeOutputCreditWaiters()", complete)

        # 29.97-fps timing makes three nominal 2-second segments total 6.006s.
        # A strict 6.0-second cap deadlocks before the three-item startup queue;
        # one-segment duration slack admits the third while item/byte caps remain
        # strict and the following segment is blocked once the queue is over cap.
        seconds_limit = 6.0
        segment_seconds = 2.002
        retained_seconds = 0.0
        admitted = 0
        for _ in range(4):
            allowed = (
                retained_seconds + segment_seconds
                <= seconds_limit + segment_seconds + 0.001
            )
            if not allowed:
                break
            retained_seconds += segment_seconds
            admitted += 1
        self.assertEqual(admitted, 3)
        self.assertAlmostEqual(retained_seconds, 6.006)

    def test_vod_fast_start_then_uses_hierarchical_steady_windows(self):
        loop = self.source.split("let coreStartIndex = hasRestoredAnyWindow ? 1 : 0", 1)[1]
        loop = loop.split("hasRestoredAnyWindow = true", 1)[0]
        self.assertIn("await vodCoreSegmentCountIfReady(", loop)
        self.assertIn("let retirementCount = hasRestoredAnyWindow", loop)
        self.assertIn(": max(0, coreSegmentCount - 1)", loop)

        selector = self.source.split(
            "private func vodCoreSegmentCountIfReady(", 1
        )[1].split("\n  private func restoreWindow(", 1)[0]
        for contract in [
            "availableCoreCount = sources.count - coreStartIndex - 1",
            "vodMinimumSteadyBatchCoreSegments",
            "vodInitialRestoreBatchCoreSegments",
            "vodSteadyRestoreBatchCoreSegments",
            "vodSteadyRestoreBatchTargetSeconds",
            "vodSteadyRestoreBatchMaximumBytes",
            "await mediaFileWorker.byteCount(at: source.localURL)",
            "await mediaFileWorker.byteCount(at: next.localURL)",
        ]:
            with self.subTest(selector_contract=contract):
                self.assertIn(contract, selector)

        # Ten-second source cuts must not turn an eighteen-core steady batch
        # into three minutes. The selector stops at three cores (30 seconds)
        # once the reserved right-context segment proves the next would exceed
        # the 36-second budget. Two cores remain the hard startup minimum.
        selected_duration = 0.0
        selected_count = 0
        for duration in [10.0, 10.0, 10.0, 10.0]:
            if selected_count >= 2 and selected_duration + duration > 36.0:
                break
            selected_duration += duration
            selected_count += 1
        self.assertEqual(selected_count, 3)
        self.assertEqual(selected_duration, 30.0)

        # Model the exact retained-left/right-context policy. The small first
        # batch and the following 18-core batches must emit every source once,
        # without treating the first batch's right context as already emitted.
        remaining = list(range(3))
        emitted = remaining[0:2]
        del remaining[: 2 - 1]
        self.assertEqual(emitted, [0, 1])
        self.assertEqual(remaining, [1, 2])
        remaining.extend(range(3, 21))
        emitted.extend(remaining[1 : 1 + 18])
        del remaining[:18]
        self.assertEqual(emitted, list(range(20)))
        self.assertEqual(remaining, [19, 20])

        # Hierarchical assembly must never hand more than eight files to the
        # portable assembler, and rootOffset + leafOffset must equal the flat
        # original timeline mapping for an adaptive 20-source window.
        durations = [3.5 + (index % 4) * 0.25 for index in range(20)]
        leaves = [durations[index : index + 8] for index in range(0, 20, 8)]
        self.assertTrue(all(1 <= len(leaf) <= 8 for leaf in leaves))
        self.assertLessEqual(len(leaves), 8)
        root_offsets = []
        cursor = 0.0
        for leaf in leaves:
            root_offsets.append(cursor)
            cursor += sum(leaf)
        flattened = []
        for root_offset, leaf in zip(root_offsets, leaves):
            leaf_cursor = 0.0
            for duration in leaf:
                flattened.append(root_offset + leaf_cursor)
                leaf_cursor += duration
        expected = []
        cursor = 0.0
        for duration in durations:
            expected.append(cursor)
            cursor += duration
        self.assertEqual(flattened, expected)

        hierarchy = self.source.split("private func assembleInterval(", 1)[1]
        hierarchy = hierarchy.split("\n  private func runWorker(", 1)[0]
        for contract in [
            "let leafLimit = IPadHLSIntervalAssembler.maximumInputCount",
            "inputURLs.count <= leafLimit * leafLimit",
            "offset + leafLimit",
            "leafResult.sourceOffsets.count == end - offset",
            "leafResult.sourceDurations.count == end - offset",
            "inputURLs: leafURLs",
            "root.sourceOffsets.count == leafResults.count",
            "root.sourceDurations.count == leafResults.count",
            "rootOffset + $0",
            "sourceDurations.append(contentsOf: leaf.sourceDurations)",
            "sourceOffsets.count == inputURLs.count",
            "sourceDurations.count == inputURLs.count",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, hierarchy)

        # A discontinuity must flush before either adaptive batch can span it.
        share = self.source.split("private func canShareWindow(", 1)[1]
        share = share.split("\n  private func ", 1)[0]
        self.assertIn("next.mediaSegment.discontinuitySequence", share)
        self.assertIn("previous.mediaSegment.discontinuitySequence", share)
        production = self.source.split("if let previous = restorationWindow.last", 1)[1]
        production = production.split("restorationWindow.append(restorationSource)", 1)[0]
        self.assertIn("!canShareWindow(previous, restorationSource)", production)
        self.assertIn("flushWindow(", production)

        restore = self.source.split("private func restoreWindow(\n    _ sources:", 1)[1]
        restore = restore.split("\n  /// The portable assembler", 1)[0]
        self.assertIn("if coreStartIndex < coreEndIndex", restore)
        self.assertIn("for coreIndex in coreStartIndex...coreEndIndex", restore)
        self.assertIn("coreIndex: coreIndex - lowerBound", restore)

    def test_large_assembly_copy_and_process_wait_do_not_block_main_actor(self):
        for contract in [
            "private actor IntervalAssemblyWorker",
            "private actor MediaFileWorker",
            "try await intervalAssemblyWorker.concatenate(",
            "try await intervalAssemblyWorker.validateDecodableVideo(",
            "await mediaFileWorker.byteCount(",
            "try await mediaFileWorker.copyReplacing(",
            "private struct SendableProcess: @unchecked Sendable",
            "private func waitForProcessExit(",
            "timeoutSeconds: TimeInterval = 1",
            "while sendable.process.isRunning, Date() < deadline",
            "await Task.detached(priority: .utility)",
            "await waitForProcessExit(process)",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, self.source)

        producer = self.source.split("@MainActor", 1)[1]
        self.assertNotIn("\n    process.waitUntilExit()", producer)
        self.assertNotIn(
            "FileManager.default.copyItem(at: workerURL, to: stableURL)",
            producer,
        )
        media_worker = self.source.split("private actor MediaFileWorker", 1)[1]
        media_worker = media_worker.split("\n  private struct SendableProcess", 1)[0]
        self.assertLess(
            media_worker.index("FileManager.default.linkItem("),
            media_worker.index("FileManager.default.copyItem("),
        )


if __name__ == "__main__":
    unittest.main()
