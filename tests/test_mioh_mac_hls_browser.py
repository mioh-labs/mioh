import functools
import http.server
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packaging" / "macOS" / "standalone"
APP_SOURCE = PACKAGE / "MiohApp.swift"
PLAYER_SOURCE = PACKAGE / "RealtimePlayer.swift"
BUILD_SCRIPT = PACKAGE / "build_app.sh"
REMOTE_APP_SOURCE = ROOT / "apps" / "MiohRemote" / "MiohRemote"
INTERACTIVE_BROWSER_SOURCE = REMOTE_APP_SOURCE / "IPadInteractiveMediaBrowser.swift"
RESOLVER_SOURCE = REMOTE_APP_SOURCE / "IPadMediaURLResolver.swift"
RELAY_PROBE_HARNESS = ROOT / "tests" / "swift" / "MacBrowserHLSRelayProbeHarness.swift"
CAPTURE_RATE_HARNESS = (
    ROOT / "tests" / "swift" / "MacHLSCaptureRatePolicyHarness.swift"
)
ACCELERATED_CAPTURE_HARNESS = (
    ROOT / "tests" / "swift" / "MacHLSAVFoundationCaptureHarness.swift"
)


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

    def test_avfoundation_asset_uses_selected_quality_and_browser_proxy_when_available(self):
        self.assert_contracts(
            self.player,
            [
                "let safariCompatiblePlaybackURL = requestedHLSQuality == .automatic",
                "? source.playbackURL",
                ": source.mediaURL",
                "let makeAVFoundationCapture: (URL) -> MacHLSAVFoundationCapture",
                "url: url",
                "capture = makeAVFoundationCapture(safariCompatiblePlaybackURL)",
                "if useSafariCompatibleHLS, selectedResourceLoader == nil",
                "resourceLoader: selectedResourceLoader",
                "capture = makeAVFoundationCapture(localPlaybackURL)",
                "avFoundationCapture: capture",
            ],
        )

    def test_hls_transport_picker_defaults_to_fast_and_is_saved(self):
        self.assert_contracts(
            self.app,
            [
                "var previewUseSafariCompatibleHLS: Bool?",
                "@Published var previewUseSafariCompatibleHLS = false",
                "var previewHLSQuality: String?",
                "@Published var previewHLSQuality = PreviewHLSQuality.automatic.rawValue",
                "previewUseSafariCompatibleHLS: false",
                "previewHLSQuality: PreviewHLSQuality.automatic.rawValue",
                "previewUseSafariCompatibleHLS: previewUseSafariCompatibleHLS",
                "previewHLSQuality: previewHLSQuality",
                "snapshot.previewUseSafariCompatibleHLS ?? false",
                "snapshot.previewHLSQuality ?? \"\"",
                'Section("HLS再生")',
                'Picker(\n          "HLS通信",',
                'Text("高速（区間先読み）").tag(false)',
                'Text("Safari互換（429回避）").tag(true)',
                'Picker("HLS画質", selection: $runner.previewHLSQuality)',
                "ForEach(PreviewHLSQuality.allCases)",
                "指定画質以下で最も高いvariantを固定使用",
                "変更は次回のHLS復元再生から適用されます。",
            ],
        )
        self.assert_contracts(
            self.player,
            [
                "runner.previewUseSafariCompatibleHLS || hasSeparateAudio",
                "let requestedHLSQuality = PreviewHLSQuality(",
                "allowsVariantFallback: requestedHLSQuality == .automatic",
                "if useSafariCompatibleHLS {",
                "HLS通信: 高速な区間先読み方式を使用します",
            ],
        )
        playback_view = self.player.split(
            "struct RealtimePlayerView: View", 1
        )[1]
        self.assertNotIn('Picker(\n              "HLS通信",', playback_view)

    def test_requested_hls_quality_is_resolved_before_both_playback_modes(self):
        _, browser = self.source_containing("MacMediaBrowserController")
        _, producer = self.source_containing("MacHLSRealtimeProducer")

        self.assert_contracts(
            browser,
            [
                "let requestedQuality = PreviewHLSQuality(",
                "constrainedTo: requestedQuality",
                "while let currentHeight = selected.hlsPlaylist?.masterMetadata?.height",
                "currentHeight > targetHeight",
                "resolveNextHLSVariant(for: selected)",
                '"HLS画質: \\(requestedQuality.label) / "',
                "player.startHLS(\n          source: source,",
            ],
        )
        self.assert_contracts(
            producer,
            [
                "private let allowsVariantFallback: Bool",
                "allowsVariantFallback: Bool = true",
                "if allowsVariantFallback, !playlist.isLive,",
            ],
        )

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
                "let source = try await Self.source(",
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
                "func acquireMediaPlaybackHandoffLease(",
                "replacingActive: Bool = false",
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
            3,
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

        compatible_path = producer.split(
            "private func runAVFoundationCapture(", 1
        )[1].split("/// May be called", 1)[0]
        self.assertIn("avFoundationCoreSegmentCountIfReady(", compatible_path)
        self.assertIn("coreStartIndex: coreStartIndex", compatible_path)
        self.assertIn("coreEndIndex: coreEndIndex", compatible_path)
        self.assertIn("avFoundationSteadyRestoreBatchCoreSegments = 4", producer)
        self.assertNotIn("coreIndex: 1", compatible_path)

    def test_avfoundation_capture_fills_at_two_x_until_target(self):
        _, capture = self.source_containing("final class MacHLSAVFoundationCapture")
        _, producer = self.source_containing("MacHLSRealtimeProducer")
        self.assert_contracts(
            capture,
            [
                "struct MacHLSCaptureRatePolicy",
                "acceleratedRate: Float = 2",
                "ratePolicy.finishWarmup()",
                "setRestoredBufferLead(",
                "player.rate = desired",
            ],
        )
        self.assertNotIn("beginFrameGapRecovery", capture)
        self.assertNotIn("seekCapturePlayer(toSourceSeconds", capture)
        self.assertNotIn("reportRestorationRealtimeFactor", capture)
        self.assertNotIn("reportRestorationRealtimeFactor", producer)
        update_restored = self.player.split(
            "private func updateBufferedDuration()", 1
        )[1].split("\n  private func ", 1)[0]
        self.assertIn(
            "hlsAVFoundationCapture?.setRestoredBufferLead(bufferedSeconds)",
            update_restored,
        )

    def test_avfoundation_capture_rate_policy_runtime(self):
        if sys.platform != "darwin":
            self.skipTest("AVFoundation capture policy requires macOS")
        swiftc = shutil.which("swiftc")
        if swiftc is None:
            xcrun = shutil.which("xcrun")
            if xcrun is not None:
                swiftc = subprocess.check_output(
                    [xcrun, "--find", "swiftc"], text=True
                ).strip()
        if not swiftc:
            self.skipTest("Swift compiler is required")

        capture_name, _ = self.source_containing(
            "final class MacHLSAVFoundationCapture"
        )
        with tempfile.TemporaryDirectory(
            prefix="mioh-hls-capture-rate-"
        ) as directory:
            directory_path = Path(directory)
            executable = directory_path / "capture-rate-policy"
            build = subprocess.run(
                [
                    swiftc,
                    "-module-cache-path",
                    str(directory_path / "module-cache"),
                    "-parse-as-library",
                    str(PACKAGE / capture_name),
                    str(PACKAGE / "MacHLSAudio.swift"),
                    str(ROOT / "apps" / "MiohRemote" / "MiohRemote" / "IPadMPEGTSRemuxer.swift"),
                    str(CAPTURE_RATE_HARNESS),
                    "-framework",
                    "AVFoundation",
                    "-framework",
                    "VideoToolbox",
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
                f"Capture rate policy did not compile:\n"
                f"{build.stdout}{build.stderr}",
            )
            completed = subprocess.run(
                [str(executable)],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertIn("Mac HLS capture rate policy passed", completed.stdout)

    def test_avfoundation_accelerated_capture_preserves_fixture_frames(self):
        if sys.platform != "darwin":
            self.skipTest("AVFoundation capture requires macOS")
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        swiftc = shutil.which("swiftc")
        if not ffmpeg or not ffprobe or not swiftc:
            self.skipTest("ffmpeg, ffprobe and swiftc are required")

        capture_name, _ = self.source_containing(
            "final class MacHLSAVFoundationCapture"
        )
        with tempfile.TemporaryDirectory(
            prefix="mioh-hls-accelerated-capture-"
        ) as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "captured"
            output.mkdir()
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=30:duration=8",
                    "-f",
                    "lavfi",
                    "-i",
                    "anoisesrc=color=pink:sample_rate=48000:duration=8",
                    "-c:a",
                    "aac",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    "60",
                    "-y",
                    str(source),
                ],
                check=True,
                timeout=60,
            )
            executable = root / "accelerated-capture"
            build = subprocess.run(
                [
                    swiftc,
                    "-module-cache-path",
                    str(root / "module-cache"),
                    "-parse-as-library",
                    str(PACKAGE / capture_name),
                    str(PACKAGE / "MacHLSAudio.swift"),
                    str(ROOT / "apps" / "MiohRemote" / "MiohRemote" / "IPadMPEGTSRemuxer.swift"),
                    str(ACCELERATED_CAPTURE_HARNESS),
                    "-framework",
                    "AVFoundation",
                    "-framework",
                    "VideoToolbox",
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
                f"Accelerated capture harness did not compile:\n"
                f"{build.stdout}{build.stderr}",
            )
            # AVPlayerItemSampleBufferOutput delivers audio for HLS items, which
            # is all the Safari-compatible path ever captures, but not for a
            # plain file asset. Serve the fixture as HLS like production.
            hls = root / "hls"
            hls.mkdir()
            subprocess.run(
                [
                    ffmpeg, "-v", "error", "-i", str(source), "-c", "copy",
                    "-f", "hls", "-hls_time", "2", "-hls_playlist_type", "vod",
                    "-hls_segment_filename", str(hls / "seg%03d.ts"),
                    str(hls / "index.m3u8"),
                ],
                check=True,
                timeout=60,
            )
            handler = functools.partial(
                http.server.SimpleHTTPRequestHandler, directory=str(hls)
            )
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                completed = subprocess.run(
                    [
                        str(executable),
                        f"http://127.0.0.1:{server.server_address[1]}/index.m3u8",
                        str(output),
                        "8",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            finally:
                server.shutdown()
            self.assertEqual(
                completed.returncode,
                0,
                f"Accelerated capture failed:\n"
                f"{completed.stdout}{completed.stderr}",
            )
            frame_count = 0
            for segment in sorted(output.glob("*.mp4")):
                probe = subprocess.run(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-count_frames",
                        "-show_entries",
                        "stream=nb_read_frames",
                        "-of",
                        "default=nokey=1:noprint_wrappers=1",
                        str(segment),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                frame_count += int(probe.stdout.strip())

        self.assertIn(
            "Mac HLS AVFoundation accelerated capture passed",
            completed.stdout,
        )
        self.assertIn("2.00倍で先読みを増やします", completed.stdout)
        audio = dict(
            item.split("=", 1)
            for line in completed.stdout.splitlines()
            if line.startswith("AUDIO\t")
            for item in line.split("\t")[1:]
        )
        self.assertGreaterEqual(float(audio["end"]), 7.5, completed.stdout)
        self.assertLess(
            int(audio["silent_frames"]),
            int(audio["frames"]) * 0.01,
            f"captured audio has gaps: {audio}",
        )
        self.assertGreaterEqual(
            frame_count,
            236,
            f"accelerated capture retained only {frame_count}/240 frames",
        )

    def test_hls_original_preview_and_capture_share_one_avurlasset(self):
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]

        # Safari-compatible mode decodes video and audio with the capture
        # object's single AVURLAsset; the silent before/after comparison reuses
        # that asset. Browser-only CDNs feed it through the WebKit-backed
        # loopback proxy; direct/public URLs keep the shortest path.
        self.assert_contracts(
            start_hls,
            [
                "capture = makeAVFoundationCapture(safariCompatiblePlaybackURL)",
                "self.hlsOriginalAsset = capture.asset",
                "avFoundationCapture: capture",
                "hlsMediaProxy = proxy",
                "proxy = IPadAuthenticatedMediaProxy(",
                "resourceLoader: selectedResourceLoader",
                "try await proxy.start()",
                "self.localHLSPlaybackURL(",
                "capture = makeAVFoundationCapture(localPlaybackURL)",
                "HLS通信: Safari/WebKit通信をローカル再生へ接続しました",
            ],
        )
        self.assertLess(
            start_hls.index("try await proxy.start()"),
            start_hls.index("capture = makeAVFoundationCapture(localPlaybackURL)"),
        )

    def test_hls_buffer_limit_drives_capture_and_output_credit(self):
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]
        self.assert_contracts(
            start_hls,
            [
                "forwardBufferSeconds: runner.previewBufferLimit",
                "self.hlsAVFoundationCapture = capture",
            ],
        )

        setter = self.player.split("func setBufferLimit(", 1)[1]
        setter = setter.split("\n  private func ", 1)[0]
        self.assert_contracts(
            setter,
            [
                "hlsAVFoundationCapture?.setForwardBufferDuration(seconds)",
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
            "let fallback = producer.takePendingVariantFallbackSource()", 1
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
        self.assertIn(
            "bufferedSeconds = max(0, last.endSeconds - position)",
            update_restored,
        )

    def test_hls_has_no_unreliable_audio_preflight(self):
        self.assertNotIn("validateHLSSourceAudio(", self.player)
        self.assertNotIn("loadTracks(withMediaType: .audio)", self.player)
        self.assertNotIn("元動画の音声を利用できないため", self.player)

    def test_hls_output_credit_tracks_the_playback_queue_and_user_buffer_limit(self):
        self.assert_contracts(
            self.player,
            [
                "let hlsVODStartupSegmentCount = 3",
                "let hlsVODRebufferSegmentCount = 2",
                "private var itemEndNotificationTokens:",
                "createdProducer.updateOutputBufferLimits(",
                "hlsOutputBufferLimits(for: runner.previewBufferLimit)",
            ],
        )

        set_limit = self.player.split("func setBufferLimit(", 1)[1]
        set_limit = set_limit.split("\n  private func ", 1)[0]
        self.assert_contracts(
            set_limit,
            [
                "if hlsSource != nil",
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
