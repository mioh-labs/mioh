import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packaging" / "macOS" / "standalone"
APP_SOURCE = PACKAGE / "MiohApp.swift"
PLAYER_SOURCE = PACKAGE / "RealtimePlayer.swift"
BUILD_SCRIPT = PACKAGE / "build_app.sh"
REMOTE_APP_SOURCE = PACKAGE
INTERACTIVE_BROWSER_SOURCE = REMOTE_APP_SOURCE / "IPadInteractiveMediaBrowser.swift"
RESOLVER_SOURCE = REMOTE_APP_SOURCE / "IPadMediaURLResolver.swift"
RELAY_PROBE_HARNESS = ROOT / "tests" / "swift" / "MacBrowserHLSRelayProbeHarness.swift"


class MacHLSBrowserContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = APP_SOURCE.read_text(encoding="utf-8")
        cls.player = PLAYER_SOURCE.read_text(encoding="utf-8")
        cls.build = BUILD_SCRIPT.read_text(encoding="utf-8")
        cls.interactive_browser = INTERACTIVE_BROWSER_SOURCE.read_text(
            encoding="utf-8"
        )
        cls.swift_sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in PACKAGE.glob("*.swift")
        }

    @classmethod
    def source_containing(cls, symbol):
        for name, source in cls.swift_sources.items():
            if symbol in source:
                return name, source
        raise AssertionError(f"macOS standalone source is missing {symbol}")

    def assert_contracts(self, source, contracts):
        for contract in contracts:
            with self.subTest(contract=contract):
                self.assertIn(contract, source)

    def test_build_links_webkit_and_compiles_browser_and_portable_hls_stack(self):
        browser_name, _ = self.source_containing("MacMediaBrowserController")
        producer_name, _ = self.source_containing("MacHLSRealtimeProducer")
        capture_name, _ = self.source_containing("MacHLSAVFoundationCapture")

        self.assertIn("-framework WebKit", self.build)
        for source_name in [
            browser_name,
            producer_name,
            capture_name,
            "IPadMediaURLResolver.swift",
            "IPadMPEGTSRemuxer.swift",
            "IPadAuthenticatedMediaProxy.swift",
        ]:
            with self.subTest(source=source_name):
                self.assertIn(source_name, self.build)

    def test_avfoundation_asset_keeps_parent_master_and_external_audio(self):
        self.assert_contracts(
            self.player,
            [
                "avFoundationCapture = MacHLSAVFoundationCapture(",
                "url: source.playbackURL",
                "sourceItem = avFoundationCapture.makePlaybackItem()",
                "avFoundationCapture: avFoundationCapture",
            ],
        )

    def test_hls_transport_picker_defaults_to_fast_and_is_saved(self):
        self.assert_contracts(
            self.app,
            [
                "var previewUseSafariCompatibleHLS: Bool?",
                "@Published var previewUseSafariCompatibleHLS = false",
                "previewUseSafariCompatibleHLS: false",
                "previewUseSafariCompatibleHLS: previewUseSafariCompatibleHLS",
                "snapshot.previewUseSafariCompatibleHLS ?? false",
                'Section("HLS再生")',
                'Picker(\n          "HLS通信",',
                'Text("高速（区間先読み）").tag(false)',
                'Text("Safari互換（429回避）").tag(true)',
                "変更は次回のHLS復元再生から適用されます。",
            ],
        )
        self.assert_contracts(
            self.player,
            [
                "let useSafariCompatibleHLS = runner.previewUseSafariCompatibleHLS",
                "if useSafariCompatibleHLS {",
                "HLS通信: 高速な区間先読み方式を使用します",
            ],
        )
        playback_view = self.player.split(
            "struct RealtimePlayerView: View", 1
        )[1]
        self.assertNotIn('Picker(\n              "HLS通信",', playback_view)

    def test_content_view_has_browser_and_playback_navigation_targets(self):
        self.assert_contracts(
            self.app,
            [
                "enum WorkspaceTab",
                "case browser",
                "case playback",
                "@State private var selectedTab: WorkspaceTab",
                "TabView(selection: $selectedTab)",
                "MacMediaBrowserView(",
                'Label("ブラウザ", systemImage: "globe")',
                ".tag(WorkspaceTab.browser)",
                ".tag(WorkspaceTab.playback)",
            ],
        )

    def test_browser_monitors_dynamic_hls_and_preserves_webkit_request_context(self):
        _, browser = self.source_containing("MacMediaBrowserController")

        self.assert_contracts(
            browser,
            [
                "import WebKit",
                "IPadInteractiveMediaBrowser",
            ],
        )
        self.assert_contracts(
            self.interactive_browser,
            [
                "WKWebView",
                "WKUserScript",
                ".atDocumentStart",
                "WKScriptMessageHandler",
                "fetch",
                "XMLHttpRequest",
                "PerformanceObserver",
                ".m3u8",
                "getAllCookies",
                "IPadMediaRequestContext",
            ],
        )
        # A blob: video is only a WebKit/MSE presentation URL. The monitor must
        # keep resource-observer candidates and reject the blob itself.
        self.assertIn(
            "rawCurrentSource.startsWith('blob:')",
            self.interactive_browser,
        )

    def test_browser_split_view_expands_to_full_tab_height(self):
        _, browser = self.source_containing("MacMediaBrowserView")

        self.assert_contracts(
            browser,
            [
                "HSplitView {",
                ".frame(minWidth: 500, minHeight: 430, maxHeight: .infinity)",
                "maxHeight: .infinity",
                ".frame(maxHeight: .infinity)",
                ".frame(maxHeight: .infinity, alignment: .top)",
            ],
        )

    def test_browser_adopts_popup_webview_and_can_return_to_its_opener(self):
        _, browser = self.source_containing("MacMediaBrowserView")

        self.assert_contracts(
            browser,
            [
                "IPadInteractiveBrowserWebView(browser: browser)",
                ".id(browser.webViewGeneration)",
                "browser.canGoBack",
                "browser.canReturnToOpeningPage",
                "browser.returnToOpeningPage()",
                'Label("元のページ", systemImage: "arrowshape.turn.up.backward")',
            ],
        )
        self.assertLess(
            browser.index("IPadInteractiveBrowserWebView(browser: browser)"),
            browser.index(".id(browser.webViewGeneration)"),
        )
        self.assertIn(
            ".disabled(!browser.canGoBack && !browser.canReturnToOpeningPage)",
            browser,
        )

    def test_browser_candidate_is_resolved_before_hls_restoration_starts(self):
        _, browser = self.source_containing("MacMediaBrowserController")

        self.assert_contracts(
            browser,
            [
                "IPadMediaURLResolver",
                "IPadResolvedMediaSource",
                ".resolve(",
                "let selection = try await Self.preferredHLS(",
                "let source = selection.source",
                "player.startHLS(",
                "source:",
                "runner:",
            ],
        )
        resolve_position = browser.index(
            "let selection = try await Self.preferredHLS("
        )
        start_position = browser.index("player.startHLS(")
        self.assertLess(resolve_position, start_position)

    def test_browser_uses_paired_webkit_handoff_lease_and_webkit_download(self):
        _, browser = self.source_containing("MacMediaBrowserController")

        self.assert_contracts(
            self.interactive_browser,
            [
                "func acquireMediaPlaybackHandoffLease() async throws",
                "mediaWebView.setAllMediaPlaybackSuspended(suspended)",
                "await withCheckedContinuation",
                "if let transientPopupWebView",
                "mediaWebViews.append(transientPopupWebView)",
                "final class IPadBrowserMediaHandoffLease",
                "func resourceLoader(",
                "func end() async",
                "private actor IPadBrowserHLSResourceLoader",
                "IPadBrowserWebKitDownloadOperation",
                "webView.startDownload(using: browserRequest)",
                'case live = "default"',
                'case videoOnDemand = "force-cache"',
            ],
        )
        self.assert_contracts(
            browser,
            [
                "let handoffLease = try await self.browser.acquireMediaPlaybackHandoffLease()",
                "pendingHandoffLease = handoffLease",
                "handoffLease.resourceLoader(",
                "isLive: source.hlsPlaylist?.isLive == true",
                "retainingResolvedResourceURL: source.hlsPlaylist?.url",
                "try Task.checkCancellation()",
                "self.resolutionIsCurrent(",
                "resourceLoader: resourceLoader",
                "browserHandoffLease: handoffLease",
                "pendingHandoffLease = nil",
            ],
        )
        snapshot_position = browser.index("let candidates = await self.browser.snapshotCandidates()")
        lease_position = browser.index(
            "let handoffLease = try await self.browser.acquireMediaPlaybackHandoffLease()"
        )
        resolve_position = browser.index(
            "let selection = try await Self.preferredHLS("
        )
        start_position = browser.index("player.startHLS(")
        self.assertLess(snapshot_position, lease_position)
        self.assertLess(lease_position, resolve_position)
        self.assertLess(resolve_position, start_position)

        between = browser[lease_position:resolve_position]
        self.assertIn("try Task.checkCancellation()", between)
        self.assertIn("self.resolutionIsCurrent(", between)
        self.assertNotIn("pauseMediaPlaybackForNativeHandoff", browser)
        finish = browser.split("private func finishResolution(", 1)[1].split(
            "private func resolutionIsCurrent(", 1
        )[0]
        self.assertIn("await pendingHandoffLease.end()", finish)
        self.assertLess(
            finish.index("await pendingHandoffLease.end()"),
            finish.index("isResolving = false"),
        )
        resolution = browser.split("resolutionTask = Task", 1)[1].split(
            "private func finishResolution(", 1
        )[0]
        self.assertNotIn("Task { @MainActor in\n            await pendingHandoffLease.end()", resolution)

        self.assert_contracts(
            self.player,
            [
                "private var hlsBrowserHandoffLease: IPadBrowserMediaHandoffLease?",
                "browserHandoffLease: IPadBrowserMediaHandoffLease? = nil",
                "releaseHLSBrowserHandoffLease()",
                "releaseHLSBrowserHandoffAfterTerminalEnd()",
                "await lease.end()",
            ],
        )
        release = self.player.split(
            "private func releaseHLSBrowserHandoffLease()", 1
        )[1].split("private func releaseHLSBrowserHandoffAfterTerminalEnd", 1)[0]
        self.assertIn("lease.beginEnding()", release)
        self.assertLess(release.index("lease.beginEnding()"), release.index("Task { @MainActor"))
        terminal_release = self.player.split(
            "private func releaseHLSBrowserHandoffAfterTerminalEnd()", 1
        )[1].split("private func cleanupSourceCompatibility()", 1)[0]
        self.assertIn("guard state == .ended, hlsSource != nil", terminal_release)
        self.assertIn("hlsResourceLoader = nil", terminal_release)
        self.assertIn("releaseHLSBrowserHandoffLease()", terminal_release)
        self.assertGreaterEqual(
            self.player.count("releaseHLSBrowserHandoffAfterTerminalEnd()"),
            4,
        )

    def test_browser_hls_transport_does_not_depend_on_relay_candidate_flags(self):
        _, browser = self.source_containing("MacMediaBrowserController")
        self.assert_contracts(browser, [
            "browser.activateInspection()",
            "try await Task.sleep(nanoseconds: 350_000_000)",
            "let candidates = await self.browser.snapshotCandidates()",
            "let selection = try await Self.preferredHLS(",
            "handoffLease.resourceLoader(",
            "for: selection.candidate",
        ])
        self.assert_contracts(self.interactive_browser, [
            "private func isInCurrentOpaquePlaybackActivationWindow",
            "state.sourceActivatedAt.addingTimeInterval(12)",
            "existingInActivationWindow != candidateInActivationWindow",
            "? candidateInActivationWindow",
            "let selectedRelayEvidence = preferredHLSRelayEvidence(",
            "?? nativeRelayEvidence",
            "?? opaqueRelayEvidence",
            "browserDocumentToken:",
            "selectedRelayEvidence?.documentToken ?? candidate.documentToken",
            "browserRelayEligible: selectedRelayEvidence != nil",
        ])
        loader_factory = self.interactive_browser.split(
            "func resourceLoader(", 1
        )[1].split("func beginEnding()", 1)[0]
        self.assertNotIn("candidate.browserRelayEligible", loader_factory)
        self.assertNotIn("candidate.browserDocumentToken", loader_factory)
        self.assertIn("canDownloadHLSWithWebKit(", loader_factory)

    def test_browser_hls_always_uses_webkit_download_without_probe_or_native_fallback(self):
        _, browser = self.source_containing("MacMediaBrowserController")
        self.assert_contracts(self.interactive_browser, [
            "let isGeometricallyVisible: Bool",
            'body["isGeometricallyVisible"] as? Bool',
            "isGeometricallyVisible = Boolean(intersection && localTreeVisible",
            "isVisible = visibilityAttested && isGeometricallyVisible",
            "private func nativePlaybackHLSRelayEvidence(",
            "isMediaDocumentCurrent(documentToken)",
            'sourceKind: "active-current-source"',
            "state.isPlaying, !state.isEnded",
            "state.isGeometricallyVisible, state.renderedArea >= 4_096",
            "!state.isCompactFloatingOverlay",
            "now.timeIntervalSince(state.lastObservedAt) <= 5",
            "relayIncludesCredentials: false",
            "nativeRelayEvidenceByURL",
        ])
        association = self.interactive_browser.split(
            "private func opaquePlaybackAssociations()", 1
        )[1].split("private static func isOpaquePlaybackHLSResponse", 1)[0]
        self.assertIn("$0.state.isGeometricallyVisible", association)
        self.assertNotIn("$0.state.visibilityAttested", association)

        credential_authorization = self.interactive_browser.split(
            "let credentialAuthorizedURLKeys", 1
        )[1].split("let supersededURLKeys", 1)[0]
        self.assert_contracts(credential_authorization, [
            "state.isPlaying, state.isVisible, state.visibilityAttested",
            "opaqueAssociations.compactMap",
            "guard state.isVisible, state.visibilityAttested",
        ])
        selected = self.interactive_browser.split(
            "let selectedRelayEvidence =", 1
        )[1].split("let relevantURLs", 1)[0]
        self.assertLess(
            selected.index("preferredHLSRelayEvidence("),
            selected.index("?? nativeRelayEvidence"),
        )
        self.assertLess(
            selected.index("?? nativeRelayEvidence"),
            selected.index("?? opaqueRelayEvidence"),
        )
        snapshot_candidate = self.interactive_browser.split(
            "return IPadWebMediaCandidate(", 1
        )[1].split("private static func browserMediaEvidence", 1)[0]
        self.assertIn(
            "selectedRelayEvidence?.relayIncludesCredentials ?? false",
            snapshot_candidate,
        )
        self.assertIn(
            "selectedRelayEvidence.map { !$0.relayEligible } ?? false",
            snapshot_candidate,
        )

        self.assert_contracts(browser, [
            "guard let resourceLoader = await handoffLease.resourceLoader(",
            "HLS通信: Safari/WebKitのダウンロード通信を使用します",
            "resourceLoader: resourceLoader",
            "player.startHLS(",
        ])
        self.assertNotIn("MacBrowserHLSRelayProbe", browser)
        self.assertNotIn("browserRelayRequiresProbe", browser)
        self.assertNotIn("標準通信へ切り替えます", browser)
        loader_position = browser.index(
            "guard let resourceLoader = await handoffLease.resourceLoader("
        )
        start_position = browser.index("player.startHLS(")
        self.assertLess(loader_position, start_position)
        resolution = browser.split("resolutionTask = Task", 1)[1].split(
            "private func finishResolution(", 1
        )[0]
        self.assertIn("catch is CancellationError", resolution)
        finish = browser.split("private func finishResolution(", 1)[1].split(
            "private func resolutionIsCurrent(", 1
        )[0]
        self.assertIn("await pendingHandoffLease.end()", finish)

    def test_mac_browser_hls_relay_probe_runtime(self):
        self.skipTest("preflight relay probe was replaced by direct WKDownload")
        if sys.platform != "darwin":
            self.skipTest("Mac relay probe requires macOS")
        swiftc = shutil.which("swiftc")
        if swiftc is None:
            xcrun = shutil.which("xcrun")
            if xcrun is not None:
                swiftc = subprocess.check_output(
                    [xcrun, "--find", "swiftc"], text=True
                ).strip()
        if not swiftc:
            self.skipTest("Swift compiler is required for the Mac relay probe")

        _, browser = self.source_containing("MacMediaBrowserController")
        probe_source = (
            "import Foundation\n\n"
            + "enum MacBrowserHLSRelayProbeOutcome"
            + browser.split("enum MacBrowserHLSRelayProbeOutcome", 1)[1]
            .split("@MainActor", 1)[0]
        )
        with tempfile.TemporaryDirectory(prefix="mioh-mac-relay-probe-") as directory:
            directory_path = Path(directory)
            probe_file = directory_path / "MacBrowserHLSRelayProbe.swift"
            probe_file.write_text(probe_source, encoding="utf-8")
            executable = directory_path / "mac-relay-probe"
            build = subprocess.run(
                [
                    swiftc,
                    "-module-cache-path",
                    str(directory_path / "module-cache"),
                    "-D",
                    "MIOH_TESTING",
                    "-parse-as-library",
                    str(RESOLVER_SOURCE),
                    str(probe_file),
                    str(RELAY_PROBE_HARNESS),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                build.returncode,
                0,
                f"Mac relay probe did not compile:\n{build.stdout}{build.stderr}",
            )
            completed = subprocess.run(
                [str(executable)],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertIn("Mac browser HLS relay probe passed", completed.stdout)

    def test_browser_resolution_status_is_visible_and_stale_navigation_is_cancelled(self):
        _, browser = self.source_containing("MacMediaBrowserController")

        self.assert_contracts(
            browser,
            [
                "private var resolutionID: UUID?",
                "let navigationGeneration = browser.navigationGeneration",
                "private func resolutionIsCurrent(",
                "self.resolutionID == id",
                "browser.navigationGeneration == navigationGeneration",
                "func cancelResolutionForPageChange()",
                ".onChange(of: browser.navigationGeneration)",
                "controller.cancelResolutionForPageChange()",
                "controller.isResolving",
                "? controller.statusMessage",
                ": (browser.statusMessage ?? controller.statusMessage)",
            ],
        )
        self.assertNotIn(
            "Text(browser.statusMessage ?? controller.statusMessage)",
            browser,
        )

    def test_player_owns_avfoundation_hls_capture_and_rolling_producer(self):
        _, producer = self.source_containing("MacHLSRealtimeProducer")
        _, capture = self.source_containing("final class MacHLSAVFoundationCapture")
        combined = self.player + "\n" + producer + "\n" + capture

        self.assert_contracts(
            self.player,
            [
                "func startHLS(",
                "source: IPadResolvedMediaSource",
                "private var hlsSource: IPadResolvedMediaSource?",
                "MacHLSAVFoundationCapture",
                "MacHLSRealtimeProducer",
                "MacHLSProductionEvent",
            ],
        )
        self.assert_contracts(
            combined,
            [
                "AVPlayerItemVideoOutput",
                "AVAssetWriter",
                "IPadHLSIntervalAssembler.concatenate(",
                "case segment",
                "case ended",
            ],
        )

    def test_avfoundation_capture_never_fetches_hls_resources_directly(self):
        _, capture = self.source_containing("final class MacHLSAVFoundationCapture")
        _, producer = self.source_containing("MacHLSRealtimeProducer")

        self.assert_contracts(
            capture,
            [
                "asset = AVURLAsset(url: url)",
                "AVPlayerItem(asset: asset)",
                "AVPlayerItemVideoOutput",
                "player.play()",
                "AVAssetWriter",
                "CapturedSegment(",
            ],
        )
        capture_code = "\n".join(
            line for line in capture.splitlines()
            if not line.lstrip().startswith("//")
        )
        for forbidden in [
            "URLSession",
            "WKDownload",
            "IPadHLSResourceDownloader",
        ]:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, capture_code)

        run = producer.split("func run(emit:", 1)[1]
        self.assertLess(
            run.index("if let avFoundationCapture"),
            run.index("var downloader = makeDownloader"),
        )
        capture_branch = run.split("if let avFoundationCapture", 1)[1].split(
            "let localSegmentCacheDirectory", 1
        )[0]
        self.assertIn("runAVFoundationCapture(", capture_branch)
        self.assertIn("return", capture_branch)

    def test_hls_clock_and_capture_share_one_avurlasset(self):
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]

        # Safari-compatible mode keeps its look-ahead decoder and audible
        # clock on the capture object's single AVURLAsset. Fast mode keeps the
        # previous proxy and interval-prefetch path as a separate branch.
        self.assert_contracts(
            start_hls,
            [
                "url: source.playbackURL",
                "sourceItem = avFoundationCapture.makePlaybackItem()",
                "avFoundationCapture: avFoundationCapture",
                "hlsMediaProxy = nil",
                "let createdProxy = IPadAuthenticatedMediaProxy(",
                "resourceLoader: selectedResourceLoader",
                "try await proxy.start()",
                "proxy.localURL(",
            ],
        )
        safari_branch = start_hls.split("if useSafariCompatibleHLS {", 1)[1].split(
            "} else {", 1
        )[0]
        self.assertNotIn("proxy.localURL(", safari_branch)
        self.assertNotIn("IPadAuthenticatedMediaProxy(", safari_branch)

    def test_vod_hls_attaches_source_player_only_at_first_restored_segment(self):
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]
        before_run, event_sink = start_hls.split(
            "try await producer.run", 1
        )

        # Construct the audible item from the shared AVURLAsset early, but a
        # VOD item is attached only after the look-ahead AVPlayer has produced
        # the first restored output. Live keeps eager attachment because its
        # seekable window can slide while Core AI warms up.
        self.assert_contracts(
            before_run,
            [
                "avFoundationCapture = MacHLSAVFoundationCapture(",
                "sourceItem = avFoundationCapture.makePlaybackItem()",
                "var sourceItemInstalled = false",
                "if playlist.isLive {",
                "installPreparedHLSSourceItem(",
            ],
        )
        self.assertNotIn("sourcePlayer.replaceCurrentItem", before_run)

        first_event = event_sink.split("} catch is CancellationError", 1)[0]
        self.assert_contracts(
            first_event,
            [
                "self.generation == startingGeneration",
                "!Task.isCancelled",
                "if !sourceItemInstalled",
                "case .segment = event",
                "sourceItemInstalled = self.installPreparedHLSSourceItem(",
                "self.handleHLSProductionEvent(event",
            ],
        )
        self.assertLess(
            first_event.index("sourceItemInstalled = self.installPreparedHLSSourceItem("),
            first_event.index("self.handleHLSProductionEvent(event"),
        )

        installer = self.player.split(
            "private func installPreparedHLSSourceItem(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            installer,
            [
                "generation == expectedGeneration",
                "hlsSource != nil",
                "sourcePlayer.replaceCurrentItem(with: item)",
                "installTimeObserver()",
                "installHLSPlaybackObservers(",
            ],
        )
        self.assertNotIn("sourcePlayer.play()", installer)

    def test_vod_hls_source_buffer_is_capped_without_reducing_restored_queue_limit(self):
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]
        self.assertIn(
            "min(6, max(2, runner.previewBufferLimit))",
            start_hls,
        )

        setter = self.player.split("func setBufferLimit(", 1)[1]
        setter = setter.split("\n  private func ", 1)[0]
        self.assert_contracts(
            setter,
            [
                "let sourceBufferSeconds = isLiveHLSInput",
                "min(6, max(2, seconds))",
                "preferredForwardBufferDuration = sourceBufferSeconds",
                "hlsProducer?.updateOutputBufferLimits(hlsOutputBufferLimits(for: seconds))",
            ],
        )
        self.assertNotIn(
            "hlsOutputBufferLimits(for: sourceBufferSeconds)",
            setter,
        )

    def test_terminal_variant_fallback_restarts_after_current_producer_task(self):
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]
        fallback = start_hls.split(
            "if let fallback = producer.takePendingVariantFallbackSource()", 1
        )[1].split("producer.cancel()", 2)[1]
        self.assert_contracts(
            fallback,
            [
                "let fallbackPosition = self.position.isFinite",
                "let fallbackAutoPlay = self.shouldPlay",
                "self.hlsProductionTask = nil",
                "self.scheduleHLSVariantFallbackRestart(",
                "position: fallbackPosition",
                "autoPlay: fallbackAutoPlay",
                "generation: startingGeneration",
            ],
        )

        restart = self.player.split(
            "private func scheduleHLSVariantFallbackRestart(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            restart,
            [
                "let currentHost = Self.hlsSourceHost(currentSource)",
                "let fallbackHost = Self.hlsSourceHost(fallbackSource)",
                "let currentQuality = Self.hlsSourceQuality(currentSource)",
                "let fallbackQuality = Self.hlsSourceQuality(fallbackSource)",
                "配信元 \\(currentHost) → \\(fallbackHost)",
                "品質 \\(currentQuality) → \\(fallbackQuality)",
                "await Task.yield()",
                "self.generation == expectedGeneration",
                "self.startHLS(",
                "source: fallbackSource",
                "at: position",
                "autoPlay: autoPlay",
            ],
        )

    def test_hls_uses_restored_queue_buffer_not_source_loaded_ranges_for_ui_lead(self):
        self.assertIn("private var sourceBufferedSeconds = 0.0", self.player)

        update_source = self.player.split(
            "private func updateSourceBufferedDuration()", 1
        )[1].split("\n  private func ", 1)[0]
        self.assertIn("sourceBufferedSeconds = max(0, furthestEnd - position)", update_source)
        self.assertIn("if sourceOnlyPlayback", update_source)
        self.assertIn("bufferedSeconds = sourceBufferedSeconds", update_source)

        update_restored = self.player.split(
            "private func updateBufferedDuration()", 1
        )[1].split("\n  private func ", 1)[0]
        self.assertIn("bufferedSeconds = max(0, last.endSeconds - position)", update_restored)

    def test_hls_source_item_observes_status_time_control_and_stalls(self):
        installer = self.player.split(
            "private func installPreparedHLSSourceItem(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            installer,
            [
                "installHLSPlaybackObservers(",
                "item: item",
                "generation: expectedGeneration",
            ],
        )
        observers = self.player.split(
            "func installHLSPlaybackObservers(", 1
        )[1]
        observers = observers.split("\n  private func ", 1)[0]

        self.assert_contracts(
            observers,
            [
                "sourceItemStatusObservation = item.observe(",
                "\\.status",
                "sourcePlayer.observe(",
                "\\.timeControlStatus",
                ".AVPlayerItemPlaybackStalled",
                "updateHLSPlaybackState(item:",
            ],
        )
        self.assertIn("func updateHLSPlaybackState(", self.player)

    def test_hls_audio_sync_pauses_restored_video_when_source_waits_like_remote(self):
        self.assert_contracts(
            self.player,
            [
                "private var hlsRestoredClockFallbackActive = false",
                "func tickRestored(seconds restoredLocalSeconds: Double)",
                "private var canStartHLSWithRestoredClockFallback: Bool",
                "private var hlsRestoredHeldForSourceCatchup = false",
                "private var hlsDriftCorrectionInFlight = false",
                "let hlsDriftCorrectionGraceSeconds = 0.120",
                "let hlsDriftToleranceSeconds = 0.120",
                "let hlsDriftResumeToleranceSeconds = 0.050",
                "let hlsDriftSeekToleranceSeconds = 0.100",
                "let hlsClockObservationIntervalSeconds = 0.080",
            ],
        )
        self.assertIn("private var restoredTimeObserver: Any?", self.player)
        self.assertIn("restoredPlayer.addPeriodicTimeObserver(", self.player)
        tick = self.player.split("private func tick(sourceSeconds:", 1)[1]
        tick = tick.split("\n  private var ", 1)[0]
        self.assertNotIn("sourcePlayer.timeControlStatus != .playing", self.player)
        stalled_observer = self.player.split(
            ".AVPlayerItemPlaybackStalled", 1
        )[1].split("hlsNotificationTokens.append(stalled)", 1)[0]
        self.assertIn("self.restoredPlayer.pause()", stalled_observer)
        self.assertNotIn("absorbHLSSourceWaitWithRestoredBuffer", stalled_observer)

        update_state = self.player.split(
            "private func updateHLSPlaybackState(", 1
        )[1].split("\n  private func ", 1)[0]
        for contract in [
            "case .waitingToPlayAtSpecifiedRate:",
            "restoredPlayer.pause()",
            "case .paused:",
            "if !hlsRestoredHeldForSourceCatchup",
        ]:
            self.assertIn(contract, update_state)
        self.assertNotIn("reanchorHLSClockToRestoredPlaybackIfNeeded", self.player)
        self.assertNotIn("hlsDriftSeekCooldownSeconds", self.player)

        resume = self.player.split("private func resumeIfBuffered(", 1)[1]
        resume = resume.split("\n  private func ", 1)[0]
        self.assertIn("if hlsSource != nil {", resume)
        self.assertIn("canStartHLSWithRestoredClockFallback", resume)

        start_players = self.player.split(
            "private func startPlayersFromCurrentPosition()", 1
        )[1].split("\n  private var ", 1)[0]
        self.assertIn("shouldPreferRestoredHLSPlayback", start_players)
        self.assertNotIn("hlsRestoredClockFallbackActive = true", start_players)
        self.assertIn("beginSynchronizedHLSStart()", start_players)

        synchronized_start = self.player.split(
            "private func beginSynchronizedHLSStart()", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            synchronized_start,
            [
                "sourcePlayer.preroll(atRate: 1.0)",
                "restoredPlayer.preroll(atRate: 1.0)",
                "hlsSynchronizedStartRevision",
                "sourcePlayer.currentItem === sourceItem",
                "restoredPlayer.currentItem === restoredItem",
                "self.sourcePlayer.play()",
                "self.restoredPlayer.play()",
            ],
        )
        self.assertLess(
            synchronized_start.index("self.sourcePlayer.play()"),
            synchronized_start.index("self.restoredPlayer.play()"),
        )

    def test_terminal_hls_source_failure_keeps_the_restored_queue_playing(self):
        observers = self.player.split(
            "func installHLSPlaybackObservers(", 1
        )[1].split("\n  private func ", 1)[0]
        failed_to_end = observers.split(
            ".AVPlayerItemFailedToPlayToEndTime", 1
        )[1].split("hlsNotificationTokens.append(failedToEnd)", 1)[0]
        self.assertIn("degradeHLSSourcePlayback(", failed_to_end)
        self.assertNotIn("self.fail(", failed_to_end)

        update_state = self.player.split(
            "private func updateHLSPlaybackState(", 1
        )[1].split("\n  private func ", 1)[0]
        failed_status = update_state.split("case .failed:", 1)[1].split(
            "case .unknown:", 1
        )[0]
        self.assertIn("degradeHLSSourcePlayback(", failed_status)
        self.assertNotIn("fail(", failed_status)

        degrade = self.player.split(
            "private func degradeHLSSourcePlayback(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            degrade,
            [
                "hlsRestoredClockFallbackActive = true",
                "!hlsRestoredClockFallbackActive",
                "showOriginal = false",
                "sourcePlayer.pause()",
                "sourcePlayer.volume = 0",
                "restoredPlayer.play()",
                "復元済み映像へ切り替えて継続します（音声なし）",
            ],
        )
        self.assertNotIn("generationHasStarted = true", degrade)
        self.assertNotIn("fail(", degrade)
        # Degradation must not destroy the healthy producer or local queue.
        self.assertNotIn("hlsProducer?.cancel()", degrade)
        self.assertNotIn("clearRestoredQueue", degrade)
        self.assertNotIn("state = .failed", degrade)
        # AVFoundation may still be unwinding a resource callback. Keep its
        # item and loopback server alive until the normal stop boundary.
        self.assertNotIn("sourcePlayer.replaceCurrentItem(with: nil)", degrade)
        self.assertNotIn("hlsMediaProxy?.stop()", degrade)

        restored_clock = self.player.split(
            "private var hlsShouldUseRestoredClock:", 1
        )[1].split("\n  private func ", 1)[0]
        self.assertIn("hlsRestoredClockFallbackActive", restored_clock)
        self.assertIn("hlsSourceReachedEnd", restored_clock)

        update_state = self.player.split(
            "private func updateHLSPlaybackState(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assertIn("!hlsRestoredClockFallbackActive", update_state)

        discontinuity = self.player.split("case .discontinuity", 1)[1].split(
            "case .segment", 1
        )[0]
        self.assert_contracts(
            discontinuity,
            [
                "let sourcePlaybackUnavailable = hlsRestoredClockFallbackActive",
                "let sourceRemainsReady = refreshedSourceItem?.status == .readyToPlay",
                "hlsSourceReady = sourceRemainsReady",
                "hlsRestoredClockFallbackActive = sourcePlaybackUnavailable",
                "if let item = refreshedSourceItem",
            ],
        )

        tick_restored = self.player.split(
            "private func tickRestored(seconds restoredLocalSeconds: Double)", 1
        )[1].split("\n  ///", 1)[0]
        self.assertIn("itemSegments[ObjectIdentifier(currentItem)]", tick_restored)

        self.assert_contracts(
            self.player,
            [
                "var canShowOriginal: Bool",
                ".disabled(!controller.canShowOriginal)",
                "controller.showOriginal = $0 && controller.canShowOriginal",
                "!hlsRestoredClockFallbackActive && !hlsSourceReachedEnd",
            ],
        )

    def test_vod_hls_waits_for_source_clock_before_playing_to_preserve_audio_sync(self):
        resume = self.player.split("private func resumeIfBuffered(", 1)[1]
        resume = resume.split("\n  private func ", 1)[0]
        self.assertIn("if hlsSource != nil {", resume)
        self.assertIn("guard hlsSourceClockIsReadyForSynchronizedPlayback", resume)
        self.assertIn("if state == .paused, generationHasStarted,", resume)
        self.assertIn("restoredPlayer.currentItem != nil", resume)

        preference = self.player.split(
            "private var shouldPreferRestoredHLSPlayback:", 1
        )[1].split("\n  private var ", 1)[0]
        self.assertIn("canStartHLSWithRestoredClockFallback", preference)

        update_state = self.player.split(
            "private func updateHLSPlaybackState(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assertNotIn("if shouldPreferRestoredHLSPlayback", update_state)
        self.assertNotIn("復元済みキューの再生を続行します", update_state)

    def test_hls_initial_seek_waits_for_ready_item_and_successful_completion(self):
        self.assert_contracts(
            self.player,
            [
                "func seekHLSClockWhenReady(",
                "hlsSourceReady",
                "hlsSourceSeekCompleted",
                "hlsSourceTimeOffset",
            ],
        )
        seek_clock = self.player.split("func seekHLSClockWhenReady(", 1)[1]
        seek_clock = seek_clock.split("\n  private func ", 1)[0]
        self.assert_contracts(
            seek_clock,
            [
                "item.status == .readyToPlay",
                "sourcePlayer.seek(",
                "finished in",
                "guard finished",
                "hlsInitialSeekCompleted",
                "hlsSourceSeekCompleted",
                "retryOrDegradeHLSInitialSeek(",
            ],
        )
        retry = self.player.split(
            "private func retryOrDegradeHLSInitialSeek(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assertIn("hlsMaximumInitialSeekAttempts", retry)
        self.assertIn("degradeHLSSourcePlayback(", retry)
        self.assertNotIn("self.fail(", seek_clock)

        resume = self.player.split("private func resumeIfBuffered(", 1)[1]
        resume = resume.split("\n  private func ", 1)[0]
        self.assertIn("hlsSourceClockIsReadyForSynchronizedPlayback", resume)
        clock_ready = self.player.split(
            "private var hlsSourceClockIsReadyForSynchronizedPlayback:", 1
        )[1].split("\n  private var ", 1)[0]
        self.assertIn("hlsSourceReady", clock_ready)
        self.assertIn("hlsSourceSeekCompleted", clock_ready)

    def test_hls_seek_and_synchronized_preroll_have_bounded_watchdogs(self):
        self.assert_contracts(
            self.player,
            [
                "let hlsOperationWatchdogSeconds = 2.0",
                "let hlsMaximumSynchronizedStartAttempts = 3",
                "private var hlsInitialSeekWatchdogTask: Task<Void, Never>?",
                "private var hlsSynchronizedStartWatchdogTask: Task<Void, Never>?",
                "scheduleHLSInitialSeekWatchdog(",
                "scheduleSynchronizedHLSStartWatchdog(",
                "sourcePlayer.currentItem?.cancelPendingSeeks()",
                "sourcePlayer.cancelPendingPrerolls()",
                "restoredPlayer.cancelPendingPrerolls()",
            ],
        )

        initial_watchdog = self.player.split(
            "private func scheduleHLSInitialSeekWatchdog(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            initial_watchdog,
            [
                "Task.sleep(nanoseconds: timeoutNanoseconds)",
                "self.generation == expectedGeneration",
                "self.hlsSeekRevision == revision",
                "self.hlsSeekInFlight",
                "self.sourcePlayer.currentItem === item",
                "retryOrDegradeHLSInitialSeek(",
            ],
        )

        seek_retry = self.player.split(
            "private func retryOrDegradeHLSInitialSeek(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assertLess(
            seek_retry.index("hlsSeekRevision &+= 1"),
            seek_retry.index("item.cancelPendingSeeks()"),
        )
        self.assertIn("self.hlsSeekRevision == retryRevision", seek_retry)

        synchronized_watchdog = self.player.split(
            "private func scheduleSynchronizedHLSStartWatchdog(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            synchronized_watchdog,
            [
                "Task.sleep(nanoseconds: timeoutNanoseconds)",
                "self.generation == expectedGeneration",
                "self.hlsSynchronizedStartRevision == revision",
                "self.hlsSynchronizedStartInFlight",
                "self.sourcePlayer.currentItem === sourceItem",
                "self.restoredPlayer.currentItem === restoredItem",
                "finishSynchronizedHLSStartRetry(",
            ],
        )

        synchronized_retry = self.player.split(
            "private func finishSynchronizedHLSStartRetry(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            synchronized_retry,
            [
                "hlsSynchronizedStartWatchdogTask?.cancel()",
                "hlsSynchronizedStartRevision &+= 1",
                "hlsMaximumSynchronizedStartAttempts",
                "degradeHLSSourcePlayback(",
                "self.hlsSynchronizedStartRevision == retryRevision",
            ],
        )
        self.assertLess(
            synchronized_retry.index("hlsSynchronizedStartRevision &+= 1"),
            synchronized_retry.index("sourcePlayer.cancelPendingPrerolls()"),
        )

    def test_live_hls_clock_is_mapped_to_remote_style_source_time_offset(self):
        self.assertIn("hlsSourceTimeOffset", self.player)
        tick = self.player.split("private func tick(sourceSeconds:", 1)[1]
        tick = tick.split("\n  private func ", 1)[0]
        self.assertIn("sourceSeconds - hlsSourceTimeOffset", tick)
        self.assertIn("hlsDriftCorrectionGraceSeconds", tick)
        self.assertIn("let drift = restoredAbsolute - playbackTimelineSeconds", tick)
        self.assertIn("hlsRestoredHeldForSourceCatchup = true", tick)
        self.assertIn("restoredPlayer.pause()", tick)
        self.assertIn("hlsDriftCorrectionInFlight = true", tick)
        self.assertNotIn("hlsSourceTimeOffset = sourceSeconds - restoredAbsolute", tick)
        self.assertIn("currentRestoredItemIdentifier", tick)
        self.assertIn("currentRestoredItemStartedAt", tick)
        self.assertIn("itemSegments[ObjectIdentifier(currentItem)]", tick)

        seek_clock = self.player.split("func seekHLSClockWhenReady(", 1)[1]
        seek_clock = seek_clock.split("\n  private func ", 1)[0]
        self.assert_contracts(
            seek_clock,
            [
                "isLiveHLSInput",
                "actualSourceTime - syntheticTarget",
                "hlsSourceTimeOffset",
                "requestedStartSeconds",
            ],
        )

    def test_hls_forward_drift_skips_to_the_matching_queued_item_without_rewind(self):
        tick = self.player.split("private func tick(sourceSeconds:", 1)[1]
        tick = tick.split("\n  private var ", 1)[0]
        self.assert_contracts(
            tick,
            [
                "let availableItems = restoredPlayer.items()",
                "playbackTimelineSeconds < segment.endSeconds",
                "restoredPlayer.advanceToNextItem()",
                "releaseConsumedSegments(through: targetSegment.sequence - 1)",
                "sourcePlayer.pause()",
                "HLS音声と復元映像を同期中",
                "self.sourcePlayer.play()",
                "self.restoredPlayer.play()",
                "復元映像がHLS音声へ追いつくのを待っています",
                "hlsDriftResumeToleranceSeconds",
            ],
        )
        self.assertNotIn("hlsSourceTimeOffset = sourceSeconds - restoredAbsolute", tick)

    def test_hls_has_no_unreliable_audio_preflight_and_eof_degrades_nonfatally(self):
        self.assertNotIn("validateHLSSourceAudio(", self.player)
        self.assertNotIn("loadTracks(withMediaType: .audio)", self.player)
        self.assertNotIn("元動画の音声を利用できないため", self.player)
        self.assertIn(
            "元動画側の再生を継続できないため、復元映像のみ再生中（音声なし）",
            self.player,
        )

        eof = self.player.split(
            "private func handleHLSSourceDidReachEnd(", 1
        )[1].split("\n  private func ", 1)[0]
        self.assert_contracts(
            eof,
            [
                "if isLiveHLSInput || expectedEnd - sourceTimeline > tolerance",
                "degradeHLSSourcePlayback(",
                "hlsSourceReachedEnd = true",
                "showOriginal = false",
            ],
        )

    def test_hls_output_credit_tracks_the_playback_queue_and_user_buffer_limit(self):
        self.assert_contracts(
            self.player,
            [
                "let hlsVODStartupSegmentCount = 3",
                "let hlsVODRebufferSegmentCount = 2",
                "private var itemEndNotificationTokens:",
                "producer.updateOutputBufferLimits(",
                "hlsOutputBufferLimits(for: runner.previewBufferLimit)",
            ],
        )

        set_limit = self.player.split("func setBufferLimit(", 1)[1]
        set_limit = set_limit.split("\n  private func ", 1)[0]
        self.assert_contracts(
            set_limit,
            [
                "if hlsSource != nil",
                "preferredForwardBufferDuration = sourceBufferSeconds",
                "min(6, max(2, seconds))",
                "hlsProducer?.updateOutputBufferLimits(",
            ],
        )

        limits = self.player.split("private func hlsOutputBufferLimits(", 1)[1]
        limits = limits.split("\n  func ", 1)[0]
        self.assert_contracts(
            limits,
            [
                "MacHLSRealtimeProducer.OutputBufferLimits",
                "Double(hlsVODStartupSegmentCount + 1) * previewSegmentSeconds",
                "OutputBufferLimits.playbackDefault",
                "seconds: seconds",
                "items: items",
                "bytes: defaultLimits.bytes",
            ],
        )

        release = self.player.split(
            "private func releaseConsumedSegments(through sequence: Int)", 1
        )[1].split("\n  @discardableResult", 1)[0]
        self.assert_contracts(
            release,
            [
                "hlsProducer?.acknowledgeOutputConsumed(through: sequence)",
                "itemEndNotificationTokens.removeValue(forKey: identifier)",
                "NotificationCenter.default.removeObserver(token)",
            ],
        )

        discontinuity = self.player.split("case .discontinuity", 1)[1].split(
            "case .segment", 1
        )[0]
        self.assert_contracts(
            discontinuity,
            [
                "let lastOutputSequence = queuedSegments.map(\\.sequence).max()",
                "hlsProducer?.acknowledgeOutputConsumed(through: lastOutputSequence)",
                "clearRestoredQueue(deleteFiles: true)",
            ],
        )

        clear_queue = self.player.split("private func clearRestoredQueue(", 1)[1]
        clear_queue = clear_queue.split("\n  ///", 1)[0]
        self.assert_contracts(
            clear_queue,
            [
                "for token in itemEndNotificationTokens.values",
                "itemEndNotificationTokens.removeAll()",
            ],
        )

    def test_hls_worker_launch_carries_overlap_core_range(self):
        _, producer = self.source_containing("MacHLSRealtimeProducer")

        self.assert_contracts(
            self.app,
            [
                "outputCoreStartNanoseconds: Int64?",
                "outputCoreEndNanoseconds: Int64?",
                "outputCoreStartNanoseconds: outputCoreStartNanoseconds",
                "outputCoreEndNanoseconds: outputCoreEndNanoseconds",
            ],
        )
        self.assert_contracts(
            producer,
            [
                "outputCoreStartNanoseconds:",
                "outputCoreEndNanoseconds:",
                "nativePreviewInvocation(",
            ],
        )

    def test_live_hls_seek_is_rejected_but_vod_hls_remains_seekable(self):
        self.assertIn("func seek(", self.player)
        seek = self.player.split("func seek(", 1)[1]
        seek = seek.split("\n  func ", 1)[0]
        self.assertIn("if let hlsSource", seek)
        self.assertIn(
            "guard hlsSource.hlsPlaylist?.isLive != true else { return }",
            seek,
        )
        self.assertIn("startHLS(", seek)
        self.assertIn("source: hlsSource", seek)

    def test_user_stop_preserves_hls_selection_for_replay(self):
        stop = self.player.split("func stop(", 1)[1]
        stop = stop.split("\n  private func ", 1)[0]
        self.assert_contracts(
            stop,
            [
                "preserveHLSSelection: Bool = true",
                "if !preserveHLSSelection",
                "hlsSource = nil",
                "hlsResourceLoader = nil",
                "releaseHLSBrowserHandoffLease()",
            ],
        )
        selection_clear = stop.index("if !preserveHLSSelection")
        selection_clear_end = stop.index("}", selection_clear)
        self.assertGreater(
            stop.index("releaseHLSBrowserHandoffLease()"),
            selection_clear_end,
            "terminal Stop must release the browser lease even when the HLS selection is retained",
        )

        # Starting a replacement HLS source or selecting a local file must
        # clear the old selection explicitly, while the user-facing stop()
        # keeps it available to startSelectedInput()/remotePlay().
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]
        self.assertIn("stop(preserveHLSSelection: false)", start_hls)

        select_input = self.player.split("func selectPreviewInput(", 1)[1]
        select_input = select_input.split("\n  func ", 1)[0]
        self.assertIn("stop(preserveHLSSelection: false)", select_input)


if __name__ == "__main__":
    unittest.main()
