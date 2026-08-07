import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packaging" / "macOS" / "standalone"
APP_SOURCE = PACKAGE / "MiohApp.swift"
PLAYER_SOURCE = PACKAGE / "RealtimePlayer.swift"
BUILD_SCRIPT = PACKAGE / "build_app.sh"
REMOTE_APP_SOURCE = ROOT / "apps" / "MiohRemote" / "MiohRemote"
INTERACTIVE_BROWSER_SOURCE = REMOTE_APP_SOURCE / "IPadInteractiveMediaBrowser.swift"


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

        self.assertIn("-framework WebKit", self.build)
        for source_name in [
            browser_name,
            producer_name,
            "IPadMediaURLResolver.swift",
            "IPadMPEGTSRemuxer.swift",
            "IPadAuthenticatedMediaProxy.swift",
        ]:
            with self.subTest(source=source_name):
                self.assertIn(source_name, self.build)

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

    def test_browser_candidate_is_resolved_before_hls_restoration_starts(self):
        _, browser = self.source_containing("MacMediaBrowserController")

        self.assert_contracts(
            browser,
            [
                "IPadMediaURLResolver",
                "IPadResolvedMediaSource",
                ".resolve(",
                "let source = try await Self.preferredHLS(",
                "player.startHLS(",
                "source:",
                "runner:",
            ],
        )
        resolve_position = browser.index("let source = try await Self.preferredHLS(")
        start_position = browser.index("player.startHLS(")
        self.assertLess(resolve_position, start_position)

    def test_player_owns_authenticated_hls_source_proxy_and_rolling_producer(self):
        _, producer = self.source_containing("MacHLSRealtimeProducer")
        combined = self.player + "\n" + producer

        self.assert_contracts(
            self.player,
            [
                "func startHLS(",
                "source: IPadResolvedMediaSource",
                "private var hlsSource: IPadResolvedMediaSource?",
                "IPadAuthenticatedMediaProxy",
                "MacHLSRealtimeProducer",
                "MacHLSProductionEvent",
            ],
        )
        self.assert_contracts(
            combined,
            [
                "IPadHLSResourceDownloader",
                "IPadHLSIntervalAssembler.concatenate(",
                "requestContext:",
                "case segment",
                "case ended",
            ],
        )

    def test_hls_clock_proxy_prefers_resolved_playback_url_with_playlist_fallback(self):
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]

        # The resolver may select a media playlist for restoration while its
        # playbackURL still points at the authenticated master playlist used
        # by AVPlayer for audio and clock. Proxy that resolved playback URL
        # first, retaining the selected playlist only as a validated fallback.
        self.assertIn("for: source.playbackURL", start_hls)
        self.assertIn("for: playlist.url", start_hls)
        self.assertLess(
            start_hls.index("for: source.playbackURL"),
            start_hls.index("for: playlist.url"),
        )
        self.assert_contracts(
            start_hls,
            [
                "context: source.requestContext",
                "resolutionPolicy: source.resolutionPolicy",
            ],
        )

    def test_hls_source_item_observes_status_time_control_and_stalls(self):
        start_hls = self.player.split("func startHLS(", 1)[1]
        start_hls = start_hls.split("\n  private func ", 1)[0]
        self.assert_contracts(
            start_hls,
            [
                "installHLSPlaybackObservers(",
                "item: item",
                "generation: startingGeneration",
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

    def test_hls_initial_seek_waits_for_ready_item_and_successful_completion(self):
        self.assert_contracts(
            self.player,
            [
                "func seekHLSClockWhenReady(",
                "hlsSourceIsReady",
                "hlsInitialSeekCompleted",
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
            ],
        )

        resume = self.player.split("private func resumeIfBuffered(", 1)[1]
        resume = resume.split("\n  private func ", 1)[0]
        self.assertIn("hlsSourceIsReady", resume)
        self.assertIn("hlsInitialSeekCompleted", resume)

    def test_live_hls_clock_is_mapped_to_restored_timeline_with_anchor(self):
        self.assertIn("hlsTimelineAnchorOffset", self.player)
        tick = self.player.split("private func tick(sourceSeconds:", 1)[1]
        tick = tick.split("\n  private func ", 1)[0]
        self.assertIn("sourceSeconds - hlsTimelineAnchorOffset", tick)

        seek_clock = self.player.split("func seekHLSClockWhenReady(", 1)[1]
        seek_clock = seek_clock.split("\n  private func ", 1)[0]
        self.assert_contracts(
            seek_clock,
            [
                "isLiveHLSInput",
                "hlsTimelineAnchorOffset =",
                "requestedStartSeconds",
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
            ],
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
