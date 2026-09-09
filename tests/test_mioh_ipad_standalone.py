import json
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "MiohRemote" / "MiohRemote"


class MiohIPadStandaloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = (APP / "IPadStandaloneStore.swift").read_text()
        cls.view = (APP / "IPadStandaloneView.swift").read_text()
        cls.persistent_surface = (
            APP / "IPadPersistentPlayerSurface.swift"
        ).read_text()
        cls.engine = (APP / "MiohIPadWorkerEngine.swift").read_text()
        cls.resolver = (APP / "IPadMediaURLResolver.swift").read_text()
        cls.hls_assembler = (APP / "IPadMPEGTSRemuxer.swift").read_text()
        cls.hls_avfoundation_capture = (
            APP / "IPadHLSAVFoundationCapture.swift"
        ).read_text()
        cls.content = (APP / "ContentView.swift").read_text()
        cls.project = (
            ROOT
            / "apps"
            / "MiohRemote"
            / "MiohRemote.xcodeproj"
            / "project.pbxproj"
        ).read_text()

    def test_ipad_workspace_does_not_require_mac_connection(self):
        self.assertIn("IPadStandaloneView()", self.content)
        standalone = self.content.split("IPadStandaloneView()", 1)[0]
        self.assertNotIn("store.connected", standalone[-300:])
        self.assertIn("UIDevice.current.userInterfaceIdiom == .pad", self.content)
        self.assertIn("WorkspaceTab.allCases", self.view)

    def test_workspace_matches_the_mac_tabs_in_a_fixed_ipad_tab_bar(self):
        for tab in [
            'case .basic: "基本"',
            'case .browser: "ブラウザ"',
            'case .split: "分割"',
            'case .restoration: "復元"',
            'case .detection: "検出"',
            'case .output: "出力"',
            'case .memory: "メモリ"',
            'case .settings: "設定"',
            'case .playback: "再生"',
            'case .log: "ログ"',
        ]:
            self.assertIn(tab, self.view)
        self.assertIn("NavigationStack", self.view)
        self.assertNotIn("TabView(selection: $selectedTab)", self.view)
        self.assertNotIn(".tabViewStyle(.tabBarOnly)", self.view)
        workspace = self.view.split("private var workspace: some View", 1)[1]
        workspace = workspace.split("private var brandHeader", 1)[0]
        self.assertLess(workspace.index("brandHeader"), workspace.index("tabStrip"))
        self.assertIn("workspaceContent(for: selectedTab)", workspace)

        tab_strip = self.view.split("private var tabStrip: some View", 1)[1]
        tab_strip = tab_strip.split("private func workspaceContent", 1)[0]
        self.assertNotIn("ScrollView", tab_strip)
        self.assertIn("HStack(spacing: 0)", tab_strip)
        self.assertIn("ForEach(WorkspaceTab.allCases)", tab_strip)
        self.assertIn(".frame(maxWidth: .infinity)", tab_strip)
        self.assertNotIn(".frame(minWidth:", tab_strip)

    def test_ipados_window_controls_stay_above_the_macos_style_brand_header(self):
        self.assertIn("brandHeader", self.view)
        self.assertNotIn("ToolbarItem(placement: .topBarLeading)", self.view)
        self.assertNotIn("ToolbarItem(placement: .topBarPinnedTrailing)", self.view)
        self.assertIn("windowControlsLeadingInset: CGFloat = 72", self.view)
        self.assertIn('.frame(width: 34, height: 34)', self.view)
        self.assertIn('.font(.title2.weight(.semibold))', self.view)
        self.assertIn('Text("Motion-Informed Optical Healing")', self.view)
        self.assertIn('.font(.subheadline)', self.view)
        self.assertIn('.font(.caption)', self.view)
        self.assertIn(
            ".padding(.leading, Self.windowControlsLeadingInset)", self.view
        )
        self.assertGreaterEqual(
            self.view.count(
                ".padding(.leading, Self.windowControlsLeadingInset)"
            ),
            2,
        )
        self.assertIn('.padding(.trailing, 20)', self.view)
        self.assertIn('.frame(height: 66)', self.view)
        self.assertIn('.background(Color(uiColor: .systemBackground))', self.view)
        self.assertIn('Text(headerStatusText)', self.view)

    def test_ipad_uses_the_mac_mioh_artwork_for_app_and_header_icons(self):
        asset_catalog = APP / "Assets.xcassets"
        app_icon = asset_catalog / "AppIcon.appiconset" / "AppIcon-1024.png"
        logo = asset_catalog / "MiohLogo.imageset" / "MiohLogo@2x.png"
        app_icon_contents = json.loads(
            (asset_catalog / "AppIcon.appiconset" / "Contents.json").read_text()
        )

        self.assertTrue(app_icon.is_file())
        self.assertTrue(logo.is_file())
        width, height = struct.unpack(">II", app_icon.read_bytes()[16:24])
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(app_icon_contents["images"][0]["idiom"], "universal")
        self.assertEqual(app_icon_contents["images"][0]["platform"], "ios")
        self.assertEqual(app_icon_contents["images"][0]["size"], "1024x1024")
        self.assertIn("Assets.xcassets in Resources", self.project)
        self.assertEqual(
            self.project.count("ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;"),
            2,
        )
        self.assertIn('Image("MiohLogo")', self.view)
        self.assertNotIn('Image(systemName: "sparkles.tv.fill")', self.view)

    def test_local_run_reuses_verified_native_engine(self):
        for contract in [
            "prepared.executeLocal(",
            "MiohClusterJobRequest(",
            "restorationAssetSHA256ByIdentifier[restorationModelIdentifier]",
            "detectorAssetSHA256ByIdentifier[detectorModelIdentifier]",
            "MiohClusterMediaRange(",
            "progress: { [weak self] value in",
        ]:
            self.assertIn(contract, self.store)
        self.assertIn("preparedLocalExecutor", self.engine)
        self.assertIn("progress?(0.95)", self.engine)

    def test_full_restoration_supports_three_bounded_native_lanes(self):
        for contract in [
            "@Published var parallelRestorationLanes = 3",
            'Picker("復元runner数", selection: $store.parallelRestorationLanes)',
            'Text("1 — 標準").tag(1)',
            'Text("2 — 高負荷").tag(2)',
            'Text("3 — 最大").tag(3)',
            "let cores = (0..<3).map { _ in makeCore() }",
            "let sharedRestorerCache = MiohIPadSharedRestorerCache(",
            "private let preparedLocalExecutors: [LocalExecutor]",
            "lane: Int = 0",
            "withThrowingTaskGroup(",
            "completed.sorted { $0.index < $1.index }",
            "restoredVideoURLs: results.map(\\.outputURL)",
            "at: videoDuration",
            "releaseSupplementalRestorationLanes()",
        ]:
            self.assertIn(contract, self.store + self.view + self.engine)

        # Frame-accurate ranges retain one prior recurrent stride plus the
        # right overlap, matching the desktop cluster boundary contract.
        for contract in [
            "AVAssetReaderTrackOutput(track: track, outputSettings: nil)",
            "let stride = options.restorationClipLength - options.temporalOverlap",
            "coreStart == 0 ? 0 : max(0, coreStart - stride)",
            "coreEnd + options.temporalOverlap",
            "leadingOverlapFrames: coreStart - decodeStart",
            "trailingOverlapFrames: decodeEnd - coreEnd",
        ]:
            self.assertIn(contract, self.store)

        # Mac-coordinated Worker and the decoded-frame streaming path keep the
        # primary executor. Local-file realtime preview uses the independent
        # local executors through its bounded ordered pipeline.
        self.assertIn("maximumConcurrentJobs: 1", self.engine)
        self.assertIn("primaryCore.makeRealtimeFrameSession(", self.engine)

    def test_local_realtime_uses_a_bounded_ordered_three_lane_pipeline(self):
        for contract in [
            "let parallelRestorationLanes: Int",
            "produceParallelLocalFileSegments(",
            "nextSequenceToSchedule - nextSequenceToCommit < laneCount",
            "ready[result.job.sequence] = result",
            "ready.removeValue(",
            "forKey: nextSequenceToCommit",
            "lane: job.lane",
            "releaseSupplementalRestorationLanes()",
            "concurrentSegments: laneCount",
            "parallelRestorationLanes\": restorationParallelLanes",
            'Text("\\(realtimePlayer.restorationParallelLanes) runner")',
        ]:
            self.assertIn(contract, self.store + self.view)

        # SFTP/HTTP random-access input remains single-lane so three decoders
        # cannot turn one remote seek into unbounded network contention.
        self.assertIn(
            "let localLaneCount =\n        usesUnlimitedLocalCache(configuration)",
            self.store,
        )
        self.assertIn("restorationParallelLanes = localLaneCount", self.store)

    def test_all_ipad_restoration_and_detection_models_are_selectable(self):
        for model in [
            "basicvsrpp-v1.2-coreai-variable",
            "basicvsrpp-v1.2-coreai",
            "basicvsrpp-v1.2-coreai-t36",
            "basicvsrpp-v1.2-coreai-t90",
            "v2-coreml",
            "v3.1-fast-coreml",
            "v3.1-accurate-coreml",
            "v4-fast-coreml",
            "v4-accurate-coreml",
            "vr-v2-accurate-coreml",
        ]:
            self.assertIn(model, self.view)

    def test_output_restores_original_audio_and_can_be_shared(self):
        for contract in [
            "loadTracks(withMediaType: .audio)",
            "AVMutableComposition()",
            "AVAssetExportPresetPassthrough",
            "compositionAudio.insertTimeRange(",
            "ActivityView(items: [outputURL])",
            "VideoPlayer(player: player)",
        ]:
            self.assertIn(contract, self.store + self.view)

    def test_output_supports_exact_rational_fps_down_conversion(self):
        for contract in [
            '@Published var useFPS = false',
            '@Published var targetFPSNumerator = 30',
            '@Published var targetFPSDenominator = 1',
            'numerator: 30_000, denominator: 1_001',
            'numerator: 60_000, denominator: 1_001',
            'Toggle("FPS変換", isOn: $store.useFPS)',
            'Picker("フレームレート", selection: $store.selectedFrameRate)',
            '復元前に\\(store.targetFPSLabel)fpsへ変換し、検出・復元するフレーム数を減らします。',
            'targetFPSNumerator: useFPS ? targetFPSNumerator : nil',
            'targetFPSDenominator: useFPS ? targetFPSDenominator : nil',
        ]:
            self.assertIn(contract, self.store + self.view)

    def test_live_preview_uses_only_restored_written_frames(self):
        for contract in [
            "struct MiohIPadPreviewFrame",
            "preview: (@Sendable (MiohIPadPreviewFrame) async -> Void)?",
            "processedFrames == 0",
            ": frame.ptsNanoseconds - coreStartNanoseconds",
            "makePreviewFrame(frame)",
            "previewContext.createCGImage",
            "try await Task.sleep(nanoseconds:",
            "await preview(previewFrame)",
            "livePreviewImage",
            "self.livePreviewImage = frame.image",
            "復元ライブ",
        ]:
            self.assertIn(contract, self.engine + self.store + self.view)

    def test_realtime_playback_uses_finalized_segments_and_queue_player(self):
        for contract in [
            "final class IPadRealtimePreviewController",
            "let restoredPlayer = AVQueuePlayer()",
            "private let segmentSeconds = 2.0",
            "private let startupSegmentCount = 3",
            "private let rebufferSegmentCount = 2",
            "restoredPlayer.insert(item, after: nil)",
            "AVPlayerItemDidPlayToEndTime",
            "bufferedSeconds >= configuration.bufferLimitSeconds",
            "mioh-ipad-preview-",
        ]:
            self.assertIn(contract, self.store)

    def test_local_realtime_cache_runs_to_eof_with_disk_safety_and_bounded_player_items(self):
        for contract in [
            "usesUnlimitedLocalCache(configuration)",
            "hasLocalCacheStorageHeadroom(",
            ".volumeTotalCapacityKey",
            "reserve + inFlightMargin",
            "maximumResidentRestoredSegments = 32",
            "fillResidentRestoredQueue()",
            "cachesLocalInputToEnd",
            "ディスクキャッシュ",
            "末尾まで",
        ]:
            self.assertIn(contract, self.store + self.view)

        # Network-backed random access remains bounded; only a real local file
        # is allowed to continue producing until EOF.
        self.assertIn("configuration.inputURL.isFileURL", self.store)
        self.assertIn("configuration.inputRangeValidator == nil", self.store)
        self.assertIn(
            "while bufferedSeconds >= configuration.bufferLimitSeconds",
            self.store,
        )

    def test_realtime_restoration_capacity_is_measured_over_a_rolling_window(self):
        for contract in [
            "restorationPerformanceSamples.count > 12",
            "@Published private(set) var processingFramesPerSecond",
            "@Published private(set) var restorationFramesPerSecond",
            "@Published private(set) var recentRestoredFrameCount",
            "processedFrames: committedFrames",
            "restoredFrames: committedRestoredFrames",
            "restorationSeconds: committedRestorationSeconds",
            "timeIntervalSince(performanceCheckpoint)",
            "measuredWall / measuredMedia",
            "Double(measuredFrames) / frameWall",
            'format: "全体処理 %.2f倍速 / RTF %.2f"',
            'format: "全体 %.2f fps"',
            'format: "実復元 %.2f fps"',
            'Text("実復元なし（素通し）")',
            '"MiohRemoteDiagnostics"',
            '"realtime-latest.json"',
            '"processedFramesPerSecond"',
            '"actualRestorationFramesPerSecond"',
            '"actualRestoredFrameCount"',
            '"processingSpeed"',
        ]:
            self.assertIn(contract, self.store + self.view)

    def test_realtime_seek_restarts_a_new_generation_at_target(self):
        for contract in [
            "generation += 1",
            "requestedStartSeconds = target",
            "func seek(to seconds: Double)",
            "at: seconds",
            "coreStartNanoseconds: cursorNanoseconds",
            "coreEndNanoseconds: endNanoseconds",
            "editingRealtimePosition",
            "realtimePlayer.seek(to: target)",
        ]:
            self.assertIn(contract, self.store + self.view)

    def test_realtime_playback_keeps_audio_clock_and_corrects_drift(self):
        for contract in [
            "let sourcePlayer = AVPlayer()",
            "restoredPlayer.automaticallyWaitsToMinimizeStalling = false",
            "restoredPlayer.isMuted = true",
            "private let driftToleranceSeconds = 0.400",
            "private let driftCorrectionGraceSeconds = 0.350",
            "private let driftSeekToleranceSeconds = 0.100",
            "private let hlsDriftToleranceSeconds = 0.120",
            "private let hlsDriftSeekToleranceSeconds = 0.050",
            "private let streamingDriftToleranceSeconds = 0.080",
            "private let streamingDriftResumeToleranceSeconds = 0.035",
            "sourcePlayer.addPeriodicTimeObserver(",
            "item.observe(\\.status",
            "\\.timeControlStatus",
            "itemIdentifier != currentRestoredItemIdentifier",
            "systemUptime - currentRestoredItemStartedAt >= correctionGrace",
            "abs(drift) > activeDriftTolerance",
            "usesVODHLSClock ? hlsDriftToleranceSeconds : driftToleranceSeconds",
            "latestClockDriftSeconds = drift",
            '"sourceSeekErrorSeconds": sourceSeekErrorSeconds',
            '"restoredClockDriftSeconds": latestClockDriftSeconds',
            "toleranceBefore: tolerance",
            "sourcePlayer.volume = muted ? 0 : Float(",
            "private func beginStreamingClockSynchronization()",
            "streamingRestoredHeldForSourceCatchup = true",
            "sourcePlayer.pause()",
            "restoredPlayer.pause()",
            "self.sourcePlayer.playImmediately(atRate: 1)",
            "self.restoredPlayer.playImmediately(atRate: 1)",
            "clockObservationIntervalSeconds = 0.080",
        ]:
            self.assertIn(contract, self.store)
        self.assertNotIn(
            "state == .buffering && restoredPlayer.currentItem == nil",
            self.store,
            "rebuffering must retain the last restored image instead of exposing a blank source layer",
        )

    def test_vod_hls_uses_an_exact_audible_source_seek(self):
        for contract in [
            "? (isLiveHLS ? segmentTolerance : .zero)",
            "self.sourceSeekErrorSeconds = actual - absoluteTarget",
            "isLiveHLS ? actual - self.requestedStartSeconds : 0",
        ]:
            self.assertIn(contract, self.store)

    def test_realtime_segment_handoff_uses_a_persistent_display_surface(self):
        for contract in [
            "AVPlayerItemVideoOutput(pixelBufferAttributes:",
            "kCVPixelBufferIOSurfacePropertiesKey as String: [:]",
            "videoOutput.suppressesPlayerRendering = true",
            "restoredVideoOutput(",
            "latestRestoredPixelBuffer = pixelBuffer",
            "hasPresentedRestoredFrame = false",
            "IPadPersistentPlayerSurface(",
            "AVSampleBufferDisplayLayer.self",
            "CADisplayLink(",
            "sampleBufferDisplayLayer.isReadyForMoreMediaData",
            "sampleBufferDisplayLayer.enqueue(sampleBuffer)",
            "Intentionally do not flush sampleBufferDisplayLayer here",
        ]:
            self.assertIn(
                contract,
                self.store + self.view + self.persistent_surface,
            )
        self.assertNotIn("VTCreateCGImageFromCVPixelBuffer(", self.store)

    def test_realtime_segment_handoff_holds_a_boundary_frame_in_the_uiview(self):
        for contract in [
            "private let boundarySnapshotLayer = CALayer()",
            "boundarySnapshotLeadSeconds = 0.20",
            "layer.addSublayer(boundarySnapshotLayer)",
            "VTCreateCGImageFromCVPixelBuffer(",
            "lastSubmittedItemIdentifier == previousItemIdentifier",
            "boundaryReleaseDisplayTick = displayTick &+ 2",
            "releaseBoundarySnapshotIfReady()",
            "CATransaction.setDisableActions(true)",
            "boundarySnapshotLayer.isHidden = false",
            "boundarySnapshotLayer.isHidden = true",
        ]:
            self.assertIn(contract, self.persistent_surface)
        self.assertEqual(
            self.persistent_surface.count("copyPixelBuffer("),
            1,
            "the display surface must remain the sole video-output consumer",
        )

    def test_realtime_preview_handles_eof_cleanup_and_missed_end_events(self):
        for contract in [
            'detail.contains("core range contains no decoded frame")',
            "private func retireSegmentsBeforeCurrentItem()",
            "releaseSegments(through: active.sequence - 1)",
            "retiringTask?.cancel()",
            "if let retiringTask { await retiringTask.value }",
            "startAccessingSecurityScopedResource()",
            "stopAccessingSecurityScopedResource()",
            "UIApplication.shared.isIdleTimerDisabled = true",
            "endPreventingSleep()",
        ]:
            self.assertIn(contract, self.store)

    def test_completed_output_has_explicit_seek_controls(self):
        for contract in [
            "playbackPosition",
            "playbackDuration",
            "editingPlaybackPosition",
            "onReceive(playbackTimer)",
            "seekPlayback(to: playbackPosition)",
            "CMTime(seconds: target, preferredTimescale: 600)",
        ]:
            self.assertIn(contract, self.view)

    def test_playback_surface_can_open_fullscreen_without_replacing_players(self):
        for contract in [
            "@State private var showingFullscreenPlayback = false",
            ".fullScreenCover(isPresented: $showingFullscreenPlayback)",
            "private var playbackVideoSurface: some View",
            "private var fullscreenPlaybackView: some View",
            "private var fullscreenPlaybackControls: some View",
            "VideoPlayer(player: realtimePlayer.sourcePlayer)",
            "IPadPersistentPlayerSurface(",
            "VideoPlayer(player: player)",
            'accessibilityLabel("フルスクリーン")',
            'accessibilityLabel("フルスクリーンを閉じる")',
            "if showingFullscreenPlayback",
        ]:
            self.assertIn(contract, self.view)

    def test_fullscreen_playback_controls_auto_hide_and_return_on_tap(self):
        fullscreen = self.view.split(
            "private var fullscreenPlaybackView: some View", 1
        )[1].split("private var fullscreenPlaybackControls: some View", 1)[0]
        helpers = self.view.split(
            "private func revealFullscreenControls()", 1
        )[1].split("private var startBlocker: String?", 1)[0]

        for contract in [
            "@State private var fullscreenControlsVisible = true",
            "@State private var fullscreenControlsHideTask: Task<Void, Never>?",
            "if fullscreenControlsVisible",
            "toggleFullscreenControls()",
            ".transition(.opacity)",
            ".onAppear",
            "revealFullscreenControls()",
            ".onDisappear",
            "resetFullscreenControls()",
        ]:
            self.assertIn(contract, self.view)

        self.assertIn("Color.clear", fullscreen)
        self.assertIn(".contentShape(Rectangle())", fullscreen)
        self.assertIn('"再生コントロールを表示"', fullscreen)
        self.assertIn("try await Task.sleep(nanoseconds: 3_000_000_000)", helpers)
        self.assertIn("beginFullscreenControlInteraction()", self.view)
        self.assertIn("finishFullscreenControlInteraction()", self.view)
        self.assertIn("fullscreenControlsHideTask?.cancel()", helpers)

    def test_realtime_segments_reuse_only_the_selected_models(self):
        for contract in [
            "private actor MiohIPadSharedRestorerCache",
            "private var cached:",
            "cached.identifier == identifier",
            "cached.clipLength == clipLength",
            "private var cachedDetector:",
            "cached.identifier == detectorIdentifier",
        ]:
            self.assertIn(contract, self.engine)

    def test_realtime_uses_t48_without_changing_full_restoration_settings(self):
        realtime = self.store.split(
            "func realtimePreviewConfiguration(", 1
        )[1].split("func saveSettingsAsDefaults()", 1)[0]
        full_run = self.store.split(
            "private func run(", 1
        )[1].split("private func ", 1)[0]

        for contract in [
            'static let sharedRootIdentifier = "ipad-realtime-preview"',
            "static let realtimeTemporalFrames = 48",
            "static let minimumTemporalOverlap = 6",
            "let realtimeClipLength = min(",
            "IPadRealtimePreviewConfiguration.realtimeTemporalFrames",
            "restorationClipLength: realtimeClipLength",
            "temporalOverlap: realtimeTemporalOverlap",
        ]:
            self.assertIn(contract, self.store)
        self.assertIn("restorationClipLength: clipLength", full_run)
        self.assertIn("temporalOverlap: temporalOverlap", full_run)
        self.assertIn("crossfade: crossfade", full_run)

    def test_realtime_t48_has_a_separate_1080p_memory_budget(self):
        for contract in [
            "static let referenceClipLength = 18",
            "static let realtimeReferenceClipLength = 48",
            "static let maximumRealtimePixelFrames =",
            "pixelFrameBudget: Int = maximumPixelFrames",
            "pixelFrames <= pixelFrameBudget",
        ]:
            self.assertIn(contract, self.resolver)
        for contract in [
            "request.sharedRootIdentifier",
            "IPadRealtimePreviewConfiguration.sharedRootIdentifier",
            "IPadRestorationMediaLimits.maximumRealtimePixelFrames",
            "pixelFrameBudget: pixelFrameBudget",
            "1080p×\\(referenceClipLength)相当以下",
        ]:
            self.assertIn(contract, self.engine)

    def test_hls_realtime_restoration_uses_a_rolling_concatenated_interval(self):
        for contract in [
            "private struct HLSRestorationSource",
            "var restorationWindow: [HLSRestorationSource] = []",
            "canShareHLSRestorationWindow(previous, restorationSource)",
            "restorationWindow.count == 2",
            "restorationWindow.count == 3",
            "coreIndex: 0",
            "coreIndex: 1",
            "flushHLSRestorationWindow(",
            "IPadHLSIntervalAssembler.concatenate(",
            "sources.map(\\.localURL)",
            "restoreHLSContinuousInterval(",
            "coreStartNanoseconds: cursorNanoseconds",
            "mediaStartNanoseconds: mediaStartNanoseconds",
            "mediaEndNanoseconds: mediaEndNanoseconds",
        ]:
            self.assertIn(contract, self.store)
        self.assertNotIn("restoreHLSMediaSegment(", self.store)

    def test_hls_vod_restores_one_core_segment_as_soon_as_ready(self):
        vod = self.store.split("private func produceHLSStreamSegments(", 1)[1].split(
            "private func produceLiveHLSStreamSegments(", 1
        )[0]
        for contract in [
            "hlsInitialRestoreBatchCoreSegments = 1",
            "hlsSteadyRestoreBatchCoreSegments = 1",
            "var hasRestoredAnyWindow = false",
            "hlsCoreSegmentCountIfReady(",
            "coreStartIndex: coreStartIndex",
            "coreEndIndex: coreEndIndex",
            "hasLeftContext: hasRestoredAnyWindow",
            "let retirementCount =\n            hasRestoredAnyWindow",
        ]:
            self.assertIn(contract, self.store if "hls" in contract else vod)
        self.assertNotIn("restorationWindow.count == 2", vod)
        self.assertNotIn("restorationWindow.count == 3", vod)

    def test_browser_hls_transport_reaches_remote_proxy_downloader_and_live_refresh(self):
        for contract in [
            "let hlsResourceLoader: (any IPadHLSResourceLoading)?",
            "resourceLoader: configuration.hlsResourceLoader",
            "resourceLoader: downloader.resourceLoader",
            "private var browserHandoffLease: IPadBrowserMediaHandoffLease?",
            "releaseBrowserHandoffLease()",
        ]:
            self.assertIn(contract, self.store)

    def test_remote_hls_quality_preference_is_persisted_and_applied(self):
        for contract in [
            "enum IPadHLSQualityPreference",
            "case p1080",
            "case p720",
            "case p480",
            "hlsQualityPreference.targetHeight",
            "resolveNextHLSVariant(for: source)",
            "hlsQualityPreference: hlsQualityPreference.rawValue",
        ]:
            self.assertIn(contract, self.store)

    def test_remote_exposes_fast_and_safari_compatible_hls_modes(self):
        for contract in [
            "enum IPadHLSStreamingMode",
            "case fast",
            "case safariCompatible",
            'Picker("通信方式", selection: $store.hlsStreamingMode)',
            "hlsStreamingMode: hlsStreamingMode.rawValue",
            "hlsStreamingMode.allowsAES128HLS",
            "allowsAES128HLS: allowsAES128HLS",
        ]:
            self.assertIn(contract, self.store + self.view)
        self.assertIn(
            "IPadHLSAVFoundationCapture.swift in Sources",
            self.project,
        )

    def test_safari_mode_uses_unpaced_vod_and_avfoundation_only_for_live(self):
        safari_path = self.store.split(
            "private func produceHLSAVFoundationSegments(", 1
        )[1].split("private func produceHLSStreamSegments(", 1)[0]
        for contract in [
            "IPadHLSAVFoundationCapture(",
            "capture.makePlaybackItem()",
            "if playlist.isLive",
            "for try await captured in stream",
            "restoreHLSRestorationWindow(",
            "prepared.executeLocal(",
            "configuration.hlsStreamingMode == .safariCompatible",
            'hlsTransportLabel = "Safari互換・全速区間復元"',
            "AVPlayerItem(asset: captureAsset)",
            "await self.produceHLSStreamSegments(",
            "VOD must not be paced by an AVPlayer playback clock",
        ]:
            self.assertIn(contract, self.store)
        self.assertNotIn("IPadHLSResourceDownloader(", safari_path)
        self.assertNotIn("Process(", safari_path)

    def test_safari_mode_keeps_cloudflare_hls_inside_the_webkit_session(self):
        for contract in [
            "if store.hlsStreamingMode == .safariCompatible",
            ".acquireMediaPlaybackHandoffLease(replacingActive: true)",
            "resolvingResourceLoader = loader",
            "hlsResourceLoader: resolvingResourceLoader",
            "browserHLSResourceLoader = resourceLoader",
        ]:
            self.assertIn(contract, self.view)
        for contract in [
            "hlsResourceLoader: (any IPadHLSResourceLoading)? = nil",
            "resourceLoader: hlsResourceLoader",
            "configuration.hlsResourceLoader != nil",
            "sourceRequiresMediaProxy = true",
            "proxiedPlaybackURL.map(AVURLAsset.init(url:))",
            '"Safari互換（WebKit通信＋AVFoundation）"',
        ]:
            self.assertIn(contract, self.store)

    def test_safari_capture_accelerates_until_target_and_recovers_after_seek(self):
        for contract in [
            "static let acceleratedRate: Float = 2",
            "isFillingTarget = false",
            "buffered <= max(2, target * 0.70)",
            "return isFillingTarget ? Self.acceleratedRate : 1",
            "player.rate = desired",
            "player.playImmediately(atRate: desired)",
            "recoverCapturePlaybackIfNeeded(output: output)",
            "requestedCaptureRate > 0.05, frameSilence > 1.5",
            "setRestoredBufferLead",
            "setForwardBufferDuration",
        ]:
            self.assertIn(contract, self.hls_avfoundation_capture)
        for contract in [
            "retiringCapture?.cancel()",
            "at: seconds",
            "hlsAVFoundationCapture?.setRestoredBufferLead(bufferedSeconds)",
            "sourceSeekAttemptID",
            "sourceSeekTimeoutTask",
            "performSourceSeek(",
            "sourcePlayer.currentItem?.cancelPendingSeeks()",
        ]:
            self.assertIn(contract, self.store)
        for contract in [
            "performBoundedSeek(",
            "toleranceBefore: segmentTolerance",
            "toleranceAfter: segmentTolerance",
            "timeoutSeconds: 12",
            "toleranceBefore: .positiveInfinity",
            "player.currentItem?.cancelPendingSeeks()",
            "開始位置への移動がタイムアウトしました",
        ]:
            self.assertIn(contract, self.hls_avfoundation_capture)
        self.assertNotIn(
            "toleranceBefore: .zero,\n        toleranceAfter: .zero",
            self.hls_avfoundation_capture,
        )

    def test_vod_hls_defers_source_clock_until_restoration_has_started(self):
        for contract in [
            "private struct DeferredSourcePlayerItem",
            "private var deferredSourcePlayerItem:",
            "deferUntilFirstRestoredSegment: true",
            "installDeferredSourcePlayerItemIfNeeded()",
            "deferredSourcePlayerItem = nil",
            "sourcePlayer.automaticallyWaitsToMinimizeStalling = false",
            "sourceItem.preferredForwardBufferDuration = segmentSeconds",
            "sourcePlayer.playImmediately(atRate: 1)",
        ]:
            self.assertIn(contract, self.store)

        source_item_setup = self.store.split(
            "private func installSourcePlayerItem(", 1
        )[1].split("private func prepareSourcePlayerItem(", 1)[0]
        self.assertNotIn(
            "configuration?.bufferLimitSeconds",
            source_item_setup,
            "the audible source clock must not inherit the long restoration buffer",
        )

        reset = self.store.split("private func resetPlayersAndQueue()", 1)[1].split(
            "private func clearRestoredQueue", 1
        )[0]
        self.assertLess(
            reset.index("sourcePlayer.replaceCurrentItem(with: nil)"),
            reset.index("deferredSourcePlayerItem = nil"),
            "a seek must detach the stale source item before staging its replacement",
        )

        enqueue = self.store.split("private func enqueue(", 1)[1].split(
            "func restoredVideoOutput(", 1
        )[0]
        self.assertLess(
            enqueue.index("try fillResidentRestoredQueue()"),
            enqueue.index("installDeferredSourcePlayerItemIfNeeded()"),
        )
        self.assertIn("restoredPlayer.insert(item, after: nil)", enqueue)

        startup = self.store.split("productionTask = Task", 1)[1].split(
            "private func installSourcePlayerItem(", 1
        )[0]
        self.assertIn("if let proxiedPlaybackURL, !usesSafariCompatibleHLS", startup)
        self.assertEqual(
            startup.count("capture.makePlaybackItem()"),
            1,
            "Safari-compatible HLS must stage only one audible source item",
        )

    def test_safari_capture_uses_direct_frames_and_two_independent_watermarks(self):
        direct_path = self.store.split(
            "private func produceHLSAVFoundationSegments(", 1
        )[1].split("private func produceHLSStreamSegments(", 1)[0]
        for contract in [
            "let stream = try capture.frames()",
            "prepared.makeRealtimeFrameSession(",
            "MiohIPadRealtimeInputFrame(",
            "capture.setRawFramesConsumed(through:",
            "IPadRealtimeRestoredSegmentWriter(",
            'hlsTransportLabel = "Safari互換・直接フレーム復元"',
        ]:
            self.assertIn(contract, direct_path)
        self.assertNotIn("restoreHLSRestorationWindow(", direct_path)
        self.assertNotIn("prepared.executeLocal(", direct_path)
        for contract in [
            "rawLeadTargetSeconds = max(1, min(3, forwardBufferSeconds * 0.25))",
            "isRawQueuePaused = true",
            "rawLead <= rawTarget * 0.55",
            "if isRawQueuePaused { return 0 }",
            "setRestoredBufferLead",
        ]:
            self.assertIn(contract, self.hls_avfoundation_capture)

    def test_realtime_stall_guard_uses_24fps_without_changing_playback_speed(self):
        for contract in [
            "struct IPadRealtimeEmergencyFPSPolicy",
            "rtf >= 0.98 && lowBuffer",
            "slowWindows >= 2",
            "rtf <= 0.82",
            "healthyWindows >= 4",
            "setEmergency24FPSEnabled(enabled)",
            "24fps途切れ防止中",
        ]:
            self.assertIn(contract, self.store)
        policy = self.store.split(
            "struct IPadRealtimeEmergencyFPSPolicy", 1
        )[1].split("struct IPadRealtimePreviewConfiguration", 1)[0]
        self.assertNotIn("player.rate", policy)
        self.assertNotIn("playbackRate", policy)

    def test_realtime_failure_is_persisted_for_device_diagnostics(self):
        for contract in [
            "persistRealtimeDiagnostics(failureMessage: String? = nil)",
            'payload["lastFailure"] = persistedFailure',
            "lastRealtimeFailureMessage = failureMessage",
            "persistRealtimeDiagnostics(failureMessage: message)",
        ]:
            self.assertIn(contract, self.store)

    def test_realtime_mask_reuse_and_frame_rate_modes_are_user_selectable(self):
        for contract in [
            "enum IPadRealtimeFrameRateMode",
            'case .source: "元フレームレート維持（通常29.97fps）"',
            'case .fps24: "24fps固定"',
            'case .automatic: "自動（処理が遅いとき24fps）"',
            "@Published var detectionMaskReuseSkipFrames = 1",
            "@Published var limitHighFrameRateBeforeRestoration = true",
            "@Published var realtimeFrameRateMode",
            "detectionMaskReuseSkipFrames: detectionMaskReuseSkipFrames",
            "realtimeFrameRateMode: realtimeFrameRateMode",
            "detectionMaskReuseSkipFrames:",
            "realtimeFrameRateMode == .fps24",
            "setDetectionMaskReuseSkipFrames(",
            "setMaximumFrameRate(",
            "MiohIPadSourceFrameRate.maximumRate(",
            "configuration.realtimeFrameRateMode == .automatic",
        ]:
            self.assertIn(contract, self.store)
        for contract in [
            '"検出後にスキップ \\(store.detectionMaskReuseSkipFrames)フレーム"',
            "in: 0...12",
            'Picker("復元フレームレート"',
            "ForEach(IPadRealtimeFrameRateMode.allCases)",
            '"再生を最大29.97/30fpsにする"',
            '"59.94fpsは29.97fps、60fpsは30fpsへ、検出に入る前に間引きます。30fps以下は変更しません。"',
            "通常復元とリアルタイム復元の両方に適用されます。",
        ]:
            self.assertIn(contract, self.view)

    def test_high_frame_rate_limit_runs_before_detection_and_restoration(self):
        session = self.engine.split(
            "actor MiohIPadRealtimeFrameSession", 1
        )[1]
        append = session.split(
            "func append(_ frame: MiohIPadRealtimeInputFrame)", 1
        )[1].split("func flush()", 1)[0]
        self.assertLess(
            append.index("guard acceptsFrame(frame.ptsNanoseconds)"),
            append.index("pending.append(frame)"),
        )
        detector = session.index("batchDetector.detectBatch(")
        self.assertLess(
            session.index("guard acceptsFrame(frame.ptsNanoseconds)"),
            detector,
        )
        file_decode = self.engine.split(
            "private func decodeAndProcess(", 1
        )[1]
        self.assertLess(
            file_decode.index("gate.accepts(ptsNanoseconds)"),
            file_decode.index("detectLookaheadWindow("),
        )

    def test_direct_output_uses_short_startup_then_long_steady_segments(self):
        for contract in [
            "if sequence == 0 { return 2_000_000_000 }",
            "if sequence == 1 { return 4_000_000_000 }",
            "return 6_000_000_000",
            "Only restored output is compressed",
        ]:
            self.assertIn(
                contract,
                self.store + self.hls_avfoundation_capture,
            )

    def test_direct_output_encoding_is_ordered_with_core_ai_output(self):
        for contract in [
            "IPadRealtimeRestoredOutputPipeline",
            "for frame in frames",
            "try await writer.append(",
            "try await outputPipeline.submit(",
            "try await outputPipeline.finish()",
            "takePerformanceSamples()",
            "sample.processingSeconds",
            "preferShortSegments: bufferedSeconds",
            "configuration.bufferLimitSeconds * 0.90",
            "if preferShortSegment { return 2_000_000_000 }",
        ]:
            self.assertIn(
                contract,
                self.store + self.hls_avfoundation_capture,
            )
        self.assertNotIn(
            "private let maximumInFlightJobs = 2",
            self.hls_avfoundation_capture,
        )
        self.assertNotIn(
            "Task.detached(priority: .userInitiated)",
            self.hls_avfoundation_capture,
        )

    def test_safari_mode_downloads_and_decrypts_aes128_vod_segments(self):
        for contract in [
            'case .safariCompatible: "Safari互換（429・暗号化HLS対応）"',
            "var allowsAES128HLS: Bool { self == .safariCompatible }",
            "IPadHLSSegmentEncryption",
            'case "AES-128":',
            "cachedEncryptionKey(",
            "decryptAES128CBC(",
            "CCCrypt(",
            "kCCOptionPKCS7Padding",
            "encryption.resolvedInitializationVector(",
        ]:
            self.assertIn(
                contract,
                self.store + self.resolver,
            )

    def test_hls_assembler_never_leaks_raw_avfoundation_errors(self):
        for contract in [
            'detail: "単一区間"',
            "単一区間のコピー:",
            "の映像情報: \\(diagnostic(error))",
            "の時間情報: \\(diagnostic(error))",
            "の表示変換: \\(diagnostic(error))",
            "の時間範囲挿入: \\(diagnostic(error))",
            "passthrough export: \\(diagnostic(error))",
        ]:
            self.assertIn(contract, self.hls_assembler)

    def test_hls_restoration_window_splits_at_unsafe_boundaries(self):
        for contract in [
            "next.mediaSegment.sequence == previous.mediaSegment.sequence + 1",
            "next.mediaSegment.discontinuitySequence",
            "previous.mediaSegment.discontinuitySequence",
            "next.mediaSegment.initializationResource",
            "previous.mediaSegment.initializationResource",
            "restorationWindow.removeAll(keepingCapacity: true)",
        ]:
            self.assertIn(contract, self.store)
        self.assertIn("#EXT-X-DISCONTINUITY-SEQUENCE:", self.resolver)
        self.assertIn('uppercased == "#EXT-X-DISCONTINUITY"', self.resolver)
        self.assertIn(
            "discontinuitySequence: discontinuitySequence",
            self.resolver,
        )

    def test_workspace_has_shared_defaults_and_processing_log(self):
        for contract in [
            "saveSettingsAsDefaults()",
            "loadSavedSettings()",
            "resetSettings()",
            "IPadStandaloneLogEntry",
            "store.logs",
            "ログを消去しました",
        ]:
            self.assertIn(contract, self.store + self.view)

    def test_runtime_supports_cancel_and_security_scoped_input(self):
        for contract in [
            "runTask?.cancel()",
            "try Task.checkCancellation()",
            "startAccessingSecurityScopedResource()",
            "stopAccessingSecurityScopedResource()",
        ]:
            self.assertIn(contract, self.store)


if __name__ == "__main__":
    unittest.main()
