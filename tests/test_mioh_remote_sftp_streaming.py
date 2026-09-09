import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = ROOT / "apps" / "MiohRemote" / "MiohRemote"
PACKAGE_SOURCE = ROOT / "packages" / "MiohRemoteKit" / "Sources" / "MiohSFTPKit"

SFTP_BROWSER = APP_SOURCE / "IPadSFTPBrowser.swift"
STANDALONE_VIEW = APP_SOURCE / "IPadStandaloneView.swift"
STANDALONE_STORE = APP_SOURCE / "IPadStandaloneStore.swift"
RANGE_PROXY = APP_SOURCE / "IPadSFTPRangeProxy.swift"
WORKER_ENGINE = APP_SOURCE / "MiohIPadWorkerEngine.swift"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _swift_function_block(source: str, signature: re.Pattern[str]) -> str:
    """Return a small Swift function body for negative whole-download checks."""
    match = signature.search(source)
    if match is None:
        return ""
    opening = source.find("{", match.start())
    if opening < 0:
        return ""
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[match.start() : index + 1]
    return ""


class MiohRemoteSFTPStreamingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.browser = _read(SFTP_BROWSER)
        cls.standalone_view = _read(STANDALONE_VIEW)
        cls.standalone_store = _read(STANDALONE_STORE)
        cls.range_proxy = _read(RANGE_PROXY)
        cls.worker_engine = _read(WORKER_ENGINE)
        cls.kit_sources = {
            path: _read(path) for path in sorted(PACKAGE_SOURCE.glob("*.swift"))
        }
        cls.all_streaming_sources = "\n".join(
            [cls.browser, cls.standalone_view, cls.standalone_store]
            + list(cls.kit_sources.values())
        )

    def _movie_row(self) -> str:
        start = self.browser.index("private func entryRow")
        end = self.browser.index("private var transferSection", start)
        return self.browser[start:end]

    def test_movie_row_tap_keeps_the_existing_download_behavior(self):
        movie_row = self._movie_row()
        self.assertRegex(
            movie_row,
            r"else\s*\{[\s\S]*?Button\s*\{\s*downloadEntry\(entry\)",
        )
        self.assertIn("store.download(entry)", movie_row)
        self.assertIn("onDownloaded(url)", movie_row)
        self.assertIn("dismiss()", movie_row)
        self.assertIn(".disabled(store.isBusy || !isInputSelectionMode)", movie_row)

    def test_movie_row_long_press_offers_download_and_streaming_restoration(self):
        movie_row = self._movie_row()
        self.assertIn(".contextMenu", movie_row)
        self.assertIn('"ダウンロード"', movie_row)
        self.assertIn('"ストリーミングで復元再生"', movie_row)
        self.assertRegex(
            movie_row,
            r"\.contextMenu\s*\{[\s\S]*?"
            r"(?:store\.)?(?:download|startDownload)\w*\(entry\)[\s\S]*?"
            r"onStreamingRequested\(entry\)",
        )
        self.assertRegex(
            movie_row,
            r"\.contextMenu\s*\{[\s\S]*?if case \.selectInput = mode",
        )

    def test_streaming_choice_is_callback_wired_to_realtime_playback(self):
        for contract in [
            "let onStreamingRequested: (MiohSFTPEntry) -> Void",
            "onStreamingRequested: @escaping (MiohSFTPEntry) -> Void",
            "self.onStreamingRequested = onStreamingRequested",
            "onStreamingRequested(entry)",
        ]:
            self.assertIn(contract, self.browser)

        sftp_sheet_start = self.standalone_view.index("case .sftpInput:")
        sftp_sheet_end = self.standalone_view.index("case .sftpUpload:", sftp_sheet_start)
        sftp_sheet = self.standalone_view[sftp_sheet_start:sftp_sheet_end]
        self.assertIn("onStreamingRequested:", sftp_sheet)
        self.assertIn("startSFTPStreamingPlayback", sftp_sheet)

        handler = _swift_function_block(
            self.standalone_view,
            re.compile(r"(?:private\s+)?func\s+startSFTPStreamingPlayback\s*\("),
        )
        self.assertTrue(handler, "missing startSFTPStreamingPlayback callback handler")
        for contract in [
            "selectedTab = .playback",
            "realtimePlayer",
            "worker.prepareModels()",
        ]:
            self.assertIn(contract, handler)
        self.assertNotIn("selectManagedDownloadedInput", handler)

    def test_streaming_reader_uses_bounded_random_ranges_not_a_full_download(self):
        range_signature = re.compile(
            r"(?:public\s+)?func\s+(?:read|readMovieRange|readRange)\w*\s*\("
            r"[\s\S]{0,400}?offset:\s*UInt64"
            r"[\s\S]{0,240}?(?:length|count):\s*(?:Int|UInt32)",
        )
        range_block = _swift_function_block(self.all_streaming_sources, range_signature)
        self.assertTrue(
            range_block,
            "SFTP streaming must expose an offset/length random-access reader",
        )

        for contract in [
            "Task.checkCancellation()",
            "offset:",
            "length:",
            "min(",
            "readPipelinedRange(",
        ]:
            self.assertIn(contract, range_block)
        self.assertRegex(
            self.all_streaming_sources,
            r"connection\.sftp\.read\([\s\S]{0,500}?offset:\s*requestOffset",
        )
        self.assertRegex(
            self.all_streaming_sources,
            r"(?:maximum\w*Range\w*Bytes|maximumRangeReadBytes|rangeReadLimit)",
        )
        self.assertRegex(
            self.all_streaming_sources,
            r"(?:fstat|lstat)\([\s\S]{0,500}?(?:byteCount|size)",
        )
        for forbidden in [
            "downloadMovie(",
            "Data(contentsOf:",
            "readToEnd()",
            "partialURL",
            "createFile(",
            "moveItem(",
        ]:
            self.assertNotIn(forbidden, range_block)

    def test_restoration_uses_a_retained_resource_loader_not_a_remote_reader_asset(self):
        for contract in [
            "private struct MiohIPadLoopbackRangeInputSource",
            "let ranged = try MiohHTTPRangeAsset(",
            "retainedOwner: ranged",
            "cancelAction: { ranged.cancel() }",
            "isAppOwnedLoopbackRangeInput(",
            "rangeValidator: request.inputSHA256",
        ]:
            self.assertIn(contract, self.worker_engine)
        self.assertIn('"ETag: \\"\\(entityTag)\\""', self.range_proxy)

    def test_cancelled_seeks_cannot_create_unbounded_detached_page_reads(self):
        for contract in [
            "private actor IPadSFTPRangePageCapacity",
            "maximumActivePages = 4",
            "try await capacity.acquire()",
            "await capacity.release()",
            "private static let maximumConnections = 12",
            "var waiters: [UUID: CheckedContinuation<Data, Error>]",
            "private func cancelPageWaiter(",
        ]:
            self.assertIn(contract, self.range_proxy)
        self.assertNotIn("IPadSFTPRangeTaskWaiter", self.range_proxy)
        self.assertIn("Task.checkCancellation()", self.range_proxy)
        self.assertNotIn("IPadSFTPRangeReadGate", self.range_proxy)

    def test_seek_reuses_bounded_orphaned_pages_and_range_reads_are_pipelined(self):
        for contract in [
            "private static let pageBytes = 512 * 1_024",
            "Keep a waiter-less page in the bounded four-page set",
            "inFlight[offset] = load",
            "awaitDiscardableNetwork(",
            "MiohSFTPDiscardableFutureWaiter",
            "let attributesFuture = connection.sftp.fstat(file: opened.handle)",
            "var pendingReads: [PendingRead] = []",
            "readRequestBytes = 64 * 1_024",
            "while slots.contains(where:",
        ]:
            self.assertIn(contract, self.range_proxy + self.all_streaming_sources)

        cancel_waiter = _swift_function_block(
            self.range_proxy,
            re.compile(r"private\s+func\s+cancelPageWaiter\s*\("),
        )
        self.assertNotIn("load.task.cancel()", cancel_waiter)
        self.assertNotIn("inFlight.removeValue", cancel_waiter)

        self.assertIn(
            "SFTP request IDs remain independently multiplexed",
            self.all_streaming_sources,
        )

        for playback_contract in [
            "startupSegments: 1",
            "activeStartupSegmentCount",
            "isLoopbackRangeInput",
            "rangeTolerance",
            "requiresStreamingClockSynchronization",
            "beginStreamingClockSynchronization()",
            "streamingDriftToleranceSeconds = 0.080",
            "streamingRestoredHeldForSourceCatchup",
        ]:
            self.assertIn(playback_contract, self.standalone_store)

    def test_streaming_pipeline_exposes_transport_and_restoration_metrics(self):
        for contract in [
            "struct IPadSFTPStreamingMetrics",
            "let bitsPerSecond: Double",
            "let activeRangeReads: Int",
            "let lastRangeLatencySeconds: Double",
            "func metrics() async -> IPadSFTPStreamingMetrics",
            "@Published private(set) var sftpBitsPerSecond",
            "@Published private(set) var restorationRealtimeFactor",
            "@Published private(set) var processingFramesPerSecond",
            "@Published private(set) var restorationFramesPerSecond",
            "recordRestorationPerformance(",
            "pipelineStageLabel",
            'format: "全体処理 %.2f倍速 / RTF %.2f"',
            'format: "実復元 %.2f fps"',
            "sftpStreamingBitRate(",
        ]:
            self.assertIn(
                contract,
                self.range_proxy + self.standalone_store + self.standalone_view,
            )

    def test_repeated_seek_retires_the_previous_loopback_generation(self):
        for proxy_contract in [
            "func prepareForSeek()",
            "proxy.cancelActiveRequestsForSeek()",
            "func cancelActiveRequestsForSeek()",
            "let activeConnections = Array(connections.values)",
            "let activeTasks = Array(requestTasks.values)",
            "connections.removeAll()",
            "requestTasks.removeAll()",
            "for task in activeTasks { task.cancel() }",
        ]:
            self.assertIn(proxy_contract, self.range_proxy)

        for playback_contract in [
            "let prepareInputForSeek: (@Sendable () -> Void)?",
            "prepareInputForSeek: sftpStreamingInput.map",
            "configuration.prepareInputForSeek?()",
        ]:
            self.assertIn(playback_contract, self.standalone_store)

    def test_background_can_cancel_a_stream_during_metadata_selection(self):
        for contract in [
            "private var pendingSFTPStreamingInput",
            "pendingSFTPStreamingInput = input",
            "discardPendingSFTPStreamingInput()",
            "pendingSFTPStreamingInput === input",
        ]:
            self.assertIn(contract, self.standalone_store)
        handler = _swift_function_block(
            self.standalone_view,
            re.compile(r"(?:private\s+)?func\s+startSFTPStreamingPlayback\s*\("),
        )
        self.assertGreaterEqual(handler.count("scenePhase == .active"), 2)
        self.assertIn("input.stop()", handler)


if __name__ == "__main__":
    unittest.main()
