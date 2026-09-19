import unittest
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "MiohApp.swift"
PLAYER_SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "RealtimePlayer.swift"
BUILD_SCRIPT = ROOT / "packaging" / "macOS" / "standalone" / "build_app.sh"
BATCH_SOURCE = (
    ROOT / "packaging" / "macOS" / "standalone" / "MacNativeExportBatch.swift"
)
INPUT_PANEL_CACHE_SOURCE = (
    ROOT
    / "packaging"
    / "macOS"
    / "standalone"
    / "InputPanelThumbnailCache.swift"
)
BATCH_HARNESS = ROOT / "tests" / "swift" / "MacNativeExportBatchHarness.swift"
UNIVERSAL_BUILD_SCRIPT = (
    ROOT / "packaging" / "macOS" / "standalone" / "build_universal_app.sh"
)
INFO_PLIST = ROOT / "packaging" / "macOS" / "standalone" / "Info.plist"
COREAI_RUNNER_SOURCE = (
    ROOT / "packaging" / "macOS" / "standalone" / "CoreAIRunner.swift"
)
NATIVE_PIPELINE_SOURCE = (
    ROOT / "packaging" / "macOS" / "standalone" / "NativePreviewPipeline.swift"
)
PREVIEW_ENCODER_SOURCE = (
    ROOT / "packaging" / "macOS" / "standalone" / "PreviewVideoToolboxEncoder.swift"
)
EXPECTED_COREAI_SOURCES = (
    "basicvsrpp-v1.2-t18-fp16.aimodel",
    "basicvsrpp-v1.2-t36-fp16.aimodel",
    "basicvsrpp-v1.2-t90-fp16.aimodel",
    "lada_mosaic_detection_model_v2-fp16.aimodel",
    "lada_mosaic_detection_model_v3.1_fast-fp16.aimodel",
    "lada_mosaic_detection_model_v3.1_accurate-fp16.aimodel",
    "lada_mosaic_detection_model_v4_fast-fp16.aimodel",
    "lada_mosaic_detection_model_v4_accurate-fp16.aimodel",
    "lada_mosaic_detection_model_vr_v2_accurate-fp16.aimodel",
    "RealESRGAN_x2plus-256-fp16.aimodel",
    "RealESRGAN_x4plus-256-fp16.aimodel",
    "realesr-general-x4v3-256-fp16.aimodel",
    "4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel",
)


class StandaloneAppOptionTests(unittest.TestCase):
    def test_native_swift_pipeline_supports_complete_file_export(self):
        app = APP_SOURCE.read_text()
        batch = BATCH_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()
        encoder = PREVIEW_ENCODER_SOURCE.read_text()

        for contract in [
            'let mode = "export"',
            "makeNativeExportTask(",
            "resolvedOutputFile(",
            "mioh-swift-export-",
            "restorationFrameCount:",
            '"lada-coreai-runner"',
            '"lada-basicvsrpp-variable-runner"',
            "Swiftネイティブ書き出し",
            "出力: \\(output.path)",
            '"進捗: %3d%%',
            "書き出し完了",
            "ffmpegTemporaryDirectory:",
            "miohTemporaryDirectory:",
            'nativeEnvironment["TMPDIR"]',
        ]:
            self.assertIn(contract, app)
        self.assertIn('appendingPathComponent("\\(stem)-UC")', batch)
        for contract in [
            "var isExport: Bool",
            "struct DetectedBatch",
            "temporalOverlap",
            "prepareInput(",
            "finishExport(",
            'arguments(audio: ["-c:a", "copy"])',
            'audio: ["-c:a", "aac", "-b:a", "192k"]',
            '"export_progress"',
            '"duration_seconds"',
            '"eta_seconds"',
            "temporaryDirectory: directory",
            "workingDirectory: ffmpegTemporaryDirectory",
            "FixedRestorerBridge",
            "VariableRestorerBridge",
            "PTSFrameRateGate",
            "source_fps",
            "fps_conversion_stage",
            "outputFPSNumerator",
            'case "none":',
            'case "count":',
            "requestedSegmentSeconds",
            "writerSegmentSeconds",
            "detectionEmptyLookahead + 1",
            "allDetections.filter { $0.classIndex == 0 }",
        ]:
            self.assertIn(contract, pipeline)
        self.assertIn("let requestedAverageBitRate: Int?", encoder)
        self.assertIn("codec == .hevc", encoder)
        self.assertIn(
            'Toggle("復元前にFPS変換", isOn: $runner.preFPSConversion)',
            app,
        )
        self.assertIn(
            ".disabled(!runner.useFPS)",
            app,
        )

    def test_native_directory_batch_keeps_single_file_and_skips_completed_outputs(self):
        app = APP_SOURCE.read_text()
        batch = BATCH_SOURCE.read_text()
        build = BUILD_SCRIPT.read_text()
        input_panel_cache = INPUT_PANEL_CACHE_SOURCE.read_text()
        for contract in [
            "MacNativeExportBatchPlanner.plan(",
            "launchNativeExportPlan(",
            "launchNextNativeExportBatchItem()",
            "finishNativeExportBatchItem(",
            "private func chooseInput()",
            'panel.title = "入力ファイルまたはフォルダを選択"',
            "panel.canChooseFiles = true",
            "panel.canChooseDirectories = true",
            "panel.allowedContentTypes = []",
            "thumbnailCache.pause()",
            "thumbnailCache.prepare(initialURL: url)",
            'actionLabel: "入力を選択…"',
            "runner.inputURL = url.standardizedFileURL",
            'sourceBatchSummary = "バッチ入力: 直下の対応動画 \\(count)本"',
            "nativeExportBatchPending.removeFirst()",
            "未処理の動画はありません。",
            "スキップ（出力済み）",
        ]:
            self.assertIn(contract, app)
        for contract in [
            "final class InputPanelThumbnailCache",
            "static let shared = InputPanelThumbnailCache()",
            "func pause()",
            "QuickLookThumbnailing",
            ".lowQualityThumbnail, .thumbnail",
            "Task.detached(priority: .utility)",
            "includingPropertiesForKeys: nil",
        ]:
            self.assertIn(contract, input_panel_cache)
        self.assertNotIn("panelSelectionDidChange", input_panel_cache)
        self.assertNotIn("panel.delegate = thumbnailCache", app)
        self.assertIn("-framework QuickLookThumbnailing", build)
        self.assertIn('"$PACKAGE_DIR/InputPanelThumbnailCache.swift"', build)
        for contract in [
            "inputValues.isRegularFile == true",
            "inputValues.isDirectory == true",
            ".skipsHiddenFiles",
            "values.isSymbolicLink != true",
            "videoExtensions.contains(candidate.pathExtension.lowercased())",
            "fileManager.fileExists(atPath: output.path)",
            "resolvedOutputFile(input: input, selectedOutput: selectedOutput)",
        ]:
            self.assertIn(contract, batch)
        self.assertIn("MacNativeExportBatch.swift", build)

    def test_native_directory_batch_planner_runtime(self):
        if sys.platform != "darwin":
            self.skipTest("Swift batch planner requires macOS")
        swiftc = shutil.which("swiftc")
        if not swiftc:
            self.skipTest("Swift compiler is required")
        with tempfile.TemporaryDirectory(
            prefix="mioh-native-export-batch-"
        ) as directory:
            root = Path(directory)
            executable = root / "batch-planner"
            compiled = subprocess.run(
                [
                    swiftc,
                    "-module-cache-path",
                    str(root / "module-cache"),
                    "-parse-as-library",
                    str(BATCH_SOURCE),
                    str(BATCH_HARNESS),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                compiled.returncode,
                0,
                f"Batch planner did not compile:\n"
                f"{compiled.stdout}{compiled.stderr}",
            )
            completed = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"Batch planner failed:\n"
                f"{completed.stdout}{completed.stderr}",
            )
        self.assertIn("Mac native export batch harness passed", completed.stdout)

    def test_coreai_runner_is_descriptor_driven(self):
        source = COREAI_RUNNER_SOURCE.read_text()

        for contract in [
            "struct TensorDescriptor: Decodable",
            "struct RunnerDescriptor: Decodable",
            "descriptor.slotCount",
            "descriptor.slotStride",
            "for input in descriptor.inputs",
            "for output in descriptor.outputs",
            'model.loadFunction(named: descriptor.function)',
            "CommandLine.arguments.count == 4",
        ]:
            self.assertIn(contract, source)
        self.assertNotIn('missingOutput("restored")', source)

    def test_native_realtime_player_has_buffered_audio_synced_controls(self):
        self.assertTrue(PLAYER_SOURCE.is_file())
        player = PLAYER_SOURCE.read_text()
        app = APP_SOURCE.read_text()

        expected_player_contracts = [
            "import AVFoundation",
            "import AVKit",
            "enum RealtimePlayerState",
            "struct PreviewWorkerEvent: Decodable",
            "final class RealtimePlayerController: ObservableObject",
            "let sourcePlayer = AVPlayer()",
            "let restoredPlayer = AVQueuePlayer()",
            "event.generation == generation",
            "startupSegmentCount = 3",
            "rebufferSegmentCount = 2",
            "hlsVODStartupSegmentCount = 3",
            "hlsVODRebufferSegmentCount = 2",
            "generationReachedEOF",
            "driftToleranceSeconds = 0.080",
            "A seek is a generation boundary",
            "kill(-processIdentifier, SIGTERM)",
            "generation: startingGeneration",
            '"native-preview-configuration.json"',
            "showOriginal",
            "sourceOnlyPlayback",
            "struct RealtimePlayerView: View",
            'Label("再生"',
            'Label("一時停止"',
            "Text(controller.processingOverlayLabel)",
        ]
        for contract in expected_player_contracts:
            self.assertIn(contract, player)
        self.assertIn('RealtimePlayerView(controller: player, runner: runner)', app)
        self.assertIn('.tabItem { Label("再生", systemImage: "play.rectangle") }', app)

    def test_realtime_video_can_move_to_one_synchronized_independent_window(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "private struct RealtimeVideoSurface: View",
            "final class RealtimeDetachedVideoWindowController",
            "NSWindowDelegate",
            "@Published private(set) var isPresented = false",
            "NSHostingView(rootView: detachedVideo)",
            "styleMask: [.titled, .closable, .miniaturizable, .resizable]",
            "window.collectionBehavior.insert(.fullScreenPrimary)",
            "func windowWillClose(_ notification: Notification)",
            "@StateObject private var detachedVideoWindow",
            'Label("独立ウインドウで表示", systemImage: "macwindow.on.rectangle")',
            'Text("動画は独立ウインドウに表示中です")',
            "detachedVideoWindow.bringToFront()",
            "detachedVideoWindow.dismiss()",
        ]:
            self.assertIn(contract, player)

        surface = player.split(
            "private struct RealtimeVideoSurface: View", 1
        )[1].split(
            "@MainActor\nprivate final class RealtimeDetachedVideoWindowController", 1
        )[0]
        self.assertIn("RealtimePlayerLayerView(player: controller.sourcePlayer)", surface)
        self.assertIn("RealtimePlayerLayerView(player: controller.restoredPlayer)", surface)
        self.assertIn("VRPreviewSceneView(", surface)
        self.assertIn("view.controlsStyle = .none", player)
        self.assertIn("showsSystemControls: false", player)

        detached = player.split(
            "private struct RealtimeDetachedVideoView: View", 1
        )[1].split(
            "@MainActor\nprivate final class RealtimeDetachedVideoWindowController", 1
        )[0]
        for control in [
            "@State private var controlsVisible = true",
            "@State private var hideControlsTask: Task<Void, Never>?",
            "controller.seek(to: target)",
            "controller.togglePlayback()",
            "controller.setMuted(!controller.muted)",
            "Binding(get: { controller.volume }, set: controller.setVolume)",
            'Toggle(\n              "処理前"',
            ".onContinuousHover { phase in",
            "Task.sleep(nanoseconds: 2_500_000_000)",
        ]:
            self.assertIn(control, detached)

        embedded = player.split("struct RealtimePlayerView: View", 1)[1]
        self.assertIn("if detachedVideoWindow.isPresented", embedded)
        self.assertIn(
            "RealtimeVideoSurface(controller: controller, runner: runner)", embedded
        )

        for localization in ["en.lproj", "zh-Hant.lproj"]:
            strings = (
                ROOT
                / "packaging/macOS/standalone/Localizations"
                / localization
                / "Localizable.strings"
            ).read_text()
            for label in [
                "独立ウインドウで表示",
                "独立ウインドウを前面に",
                "再生タブに戻す",
                "動画は独立ウインドウに表示中です",
            ]:
                self.assertIn(f'"{label}" = ', strings)

    def test_player_starts_each_generation_once_and_resumes_without_seeking(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "private var generationHasStarted = false",
            "private var generationStartPending = false",
            "guard state != .playing, !generationStartPending else { return }",
            "Double(requiredSegmentCountForCurrentGeneration())",
            "private func requiredSegmentCountForCurrentGeneration() -> Int",
            "hlsSource != nil, !isLiveHLSInput",
            "hlsVODRebufferSegmentCount",
            "hlsVODStartupSegmentCount",
            "bufferedSeconds + 0.1 >= required",
            "private func startPlayersFromCurrentPosition()",
            "let startingGeneration = generation",
            "guard self.generation == startingGeneration else { return }",
        ]:
            self.assertIn(contract, player)

    def test_repeated_seek_serializes_worker_retirement_before_restart(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "private var workerRetirementTask: Task<Void, Never>?",
            "let retirement = workerRetirementTask",
            "await retirement.value",
            "while retiringWorker.isRunning",
            "A new generation must not load Core AI assets until the old process",
        ]:
            self.assertIn(contract, player)
        self.assertNotIn('sendCommand(["command": "seek"', player)

    def test_realtime_preview_has_single_controller_owner_and_h264_default(self):
        player = PLAYER_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        for contract in [
            "private static weak var activeRestorationController",
            "previousController.stop()",
            "Self.activeRestorationController = self",
        ]:
            self.assertIn(contract, player)
        self.assertIn(
            "codec = config.isExport ? .hevc : .h264",
            pipeline,
        )

    def test_worker_reports_the_actual_segment_duration(self):
        player = PLAYER_SOURCE.read_text()

        self.assertIn("let segmentSeconds: Double?", player)
        self.assertIn(
            "previewSegmentSeconds = max(0.1, event.segmentSeconds ?? 2.0)",
            player,
        )

    def test_full_worker_capacity_unblocks_timestamp_shortfall(self):
        player = PLAYER_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        self.assertIn('case "buffer_full":', player)
        self.assertIn("resumeIfBuffered(bufferIsFull: true)", player)
        self.assertIn("bufferIsFull && !queuedSegments.isEmpty", player)
        self.assertIn("func waitForCapacity(nextSequence: Int, segmentSeconds: Double)", pipeline)
        self.assertIn("let retained = nextSequence - releasedThrough - 1", pipeline)

    def test_rolling_buffer_release_does_not_depend_only_on_end_notification(self):
        player = PLAYER_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        for contract in [
            "restoredPlayer.actionAtItemEnd = .advance",
            "retireSegmentsBeforeCurrentItem()",
            '\"command\": \"release_through\"',
            "releaseConsumedSegments(through: activeSegment.sequence - 1)",
        ]:
            self.assertIn(contract, player)
        self.assertIn('command == "release_through"', pipeline)
        self.assertIn("releasedThrough =", pipeline)

    def test_seek_starts_before_the_configured_buffer_is_full(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "Double(requiredSegmentCountForCurrentGeneration())",
            "Start as soon as a short playable lead is available",
            "min(runner?.previewBufferLimit ?? 8, previewSegmentSeconds)",
        ]:
            self.assertIn(contract, player)
        self.assertNotIn("requireConfiguredBuffer", player)
        self.assertNotIn("generationNeedsSeekBuffer", player)

    def test_seek_slider_uses_the_dragged_position_until_commit(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "@State private var isScrubbing = false",
            "get: { isScrubbing ? seekPosition : controller.position }",
            "isScrubbing = true",
            "let target = seekPosition",
            "controller.seek(to: target)",
        ]:
            self.assertIn(contract, player)

    def test_seek_keeps_the_source_frame_and_bar_visible_while_refilling(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "preserveCurrentSource: Bool = false",
            "preserveSourceItem: canReuseCurrentSource",
            "preserveHLSSelection: false",
            "preserveCurrentSource: true",
            "sourcePlayer.seek(",
            "self.position = startSeconds",
            "showsSourceFrameWhilePreparingRestoration",
            "position = requestedStartSeconds",
        ]:
            self.assertIn(contract, player)
        self.assertIn("if !preserveSourceItem", player)

    def test_hls_buffering_keeps_the_restored_frame_visible_like_remote(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "var showsRestoredFrameWhileHLSBuffers: Bool",
            "hlsSource != nil",
            "!sourceOnlyPlayback",
            "!showOriginal",
            "generationHasStarted",
            "restoredPlayer.currentItem != nil",
            "state == .loading || state == .buffering || state == .seeking",
            "var prefersSourceVideoLayer: Bool",
            "showsSourceFrameWhilePreparingRestoration && !showsRestoredFrameWhileHLSBuffers",
            "controller.prefersSourceVideoLayer",
            "if controller.prefersSourceVideoLayer",
            "RealtimePlayerLayerView(player: controller.sourcePlayer)",
            "RealtimePlayerLayerView(player: controller.restoredPlayer)",
            "if showsRestoredFrameWhileHLSBuffers { return false }",
        ]:
            self.assertIn(contract, player)
        self.assertNotIn("controller.prefersSourceVideoLayer\n                ? 1 : 0.001", player)
        self.assertNotIn("controller.prefersSourceVideoLayer\n                ? 0.001 : 1", player)

    def test_playback_input_is_independent_from_export_input(self):
        player = PLAYER_SOURCE.read_text()
        app = APP_SOURCE.read_text()

        for contract in [
            "@Published var previewInputURL: URL?",
            "func choosePreviewInput(runner: RestorationRunner)",
            "guard let input = previewInputURL",
            'title: "再生動画"',
            "controller.previewInputURL == nil",
        ]:
            self.assertIn(contract, player)
        self.assertNotIn("guard let input = runner.inputURL", player)
        self.assertIn(
            "func nativePreviewInvocation(",
            app,
        )
        self.assertIn(
            "input: input.path",
            app,
        )

    def test_app_precompiles_coreml_detection_models(self):
        build_script = BUILD_SCRIPT.read_text()

        self.assertIn("xcrun coremlcompiler compile", build_script)
        self.assertIn('"$RESOURCES/models/$compiled_name"', build_script)
        self.assertIn("lada_mosaic_detection_model_vr_v2_accurate.mlpackage", build_script)
        regular_assets = build_script.split("MODEL_ASSETS=(", 1)[1].split(")", 1)[0]
        self.assertNotIn("lada_mosaic_detection_model_v2.pt", regular_assets)
        self.assertNotIn("lada_mosaic_detection_model_v4_fast.pt", regular_assets)
        self.assertNotIn("lada_mosaic_detection_model_vr_v2_accurate.pt", regular_assets)
        self.assertNotIn("lada_mosaic_detection_model_v4_fast.mlpackage", regular_assets)

    def test_runner_exposes_current_settings_to_native_preview(self):
        app = APP_SOURCE.read_text()

        self.assertIn("func nativePreviewInvocation(", app)
        for contract in [
            "let selectedPreviewModel = previewRestorationModel",
            "let previewModel = selectedPreviewModel",
            "let selectedPreviewDetectionModel = previewDetectionModel",
            "restorationModels: restoration.url.path",
            "detectionModel: detection.url.path",
            "bufferLimitSeconds: previewBufferLimit",
            "temporalBatchFrames: temporalFrames",
            "temporalOverlap: previewOverlap",
            "blendFeather: Float(blendFeather)",
            "detectionEmptyLookahead: max(0, detectionEmptyLookahead)",
            "detectFaceMosaics: detectFaceMosaics",
            "crossfade: restoreCrossfade",
            'environment["TMPDIR"] = miohTemporary',
        ]:
            self.assertIn(contract, app)

    def test_native_preview_uses_fast_v4_detection(self):
        source = APP_SOURCE.read_text()

        self.assertIn(
            'supportsCoreAI ? "v4-fast-coreai" : "v4-fast-coreml"',
            source,
        )

    def test_realtime_preview_can_cap_high_frame_rate_before_restoration(self):
        app = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        for contract in [
            "var previewLimitHighFrameRate: Bool?",
            "@Published var previewLimitHighFrameRate = false",
            "previewLimitHighFrameRate: previewLimitHighFrameRate",
            "previewLimitHighFrameRate = snapshot.previewLimitHighFrameRate ?? false",
            "maximumFPS: previewLimitHighFrameRate ? 30 : nil",
            "preFPSConversion: true",
        ]:
            self.assertIn(contract, app)

        for contract in [
            "let maximumFPS: Int?",
            "let maximumTargetRate: (numerator: Int, denominator: Int)?",
            "config.maximumFPS.flatMap { maximum in",
            "sourceFPS > maximumValue + 0.01 ? rate : nil",
            "maximumTargetRate ?? config.targetFPS.map",
        ]:
            self.assertIn(contract, pipeline)

        for contract in [
            "func setPreviewHighFrameRateLimit(",
            '"limitHighFrameRate=\\(runner.previewLimitHighFrameRate)"',
            '"最大30fps"',
            "controller.setPreviewHighFrameRateLimit($0, runner: runner)",
            ".disabled(controller.isVRVideo)",
        ]:
            self.assertIn(contract, player)

        # The switch is present in both the playback tab and detached window.
        self.assertEqual(
            player.count("controller.setPreviewHighFrameRateLimit($0, runner: runner)"),
            2,
        )
        self.assertNotIn(
            'Text("59.94fpsは29.97fps、60fpsは30fpsへ復元前に間引きます。30fps以下は変更しません")',
            player,
        )
        self.assertNotIn(
            'Text("復元は維持し、再生中は合成エフェクトとROIエンハンサーをバイパスします")',
            player,
        )

    def test_app_has_user_default_settings_panel(self):
        source = APP_SOURCE.read_text()

        for contract in [
            "struct MiohUserDefaultsSnapshot: Codable",
            'private let defaultsKey = "mioh.userProcessingDefaults.v1"',
            "func saveCurrentDefaults()",
            "func loadSavedDefaults()",
            "func resetDefaultsToFactory()",
            "private func loadSavedDefaultsOnLaunch()",
            "func currentDefaultsSnapshot() -> MiohUserDefaultsSnapshot",
            "func apply(defaults snapshot: MiohUserDefaultsSnapshot)",
            "UserDefaults.standard.set(data, forKey: defaultsKey)",
            "UserDefaults.standard.data(forKey: defaultsKey)",
            "UserDefaults.standard.removeObject(forKey: defaultsKey)",
            'settingsTab.tabItem { Label("設定", systemImage: "gearshape") }',
            'Section("ユーザーデフォルト")',
            'Label("現在の設定をデフォルトに保存", systemImage: "square.and.arrow.down")',
            'Label("保存済みデフォルトを読み込み", systemImage: "arrow.clockwise")',
            'Label("初期値に戻す", systemImage: "trash")',
            "入力/出力、一時フォルダ、分割、復元、検出、出力、メモリ、再生バッファ、HLS通信方式・画質まで保存します",
        ]:
            self.assertIn(contract, source)

        self.assertIn("var previewBufferLimit: Double", source)
        self.assertIn("var previewHLSQuality: String?", source)
        self.assertIn("var previewProjectionMode: String?", source)
        self.assertIn("var previewVideoLayout: String?", source)
        self.assertIn("var previewEye: String?", source)
        self.assertIn("var previewCameraFOV: Double?", source)
        self.assertIn("var encoderOptions: String", source)
        self.assertIn("var roiEnhancerStrength: Double", source)
        self.assertIn("var mpsMemoryFraction: Double", source)
        self.assertIn("var restoreTemporalOverlap: Int?", source)
        self.assertIn("var restoreCrossfade: Bool?", source)
        self.assertIn("restoreTemporalOverlap = min(max(snapshot.restoreTemporalOverlap ?? 8, 0), 120)", source)
        self.assertIn("restoreCrossfade = snapshot.restoreCrossfade ?? true", source)
        self.assertIn('previewProjectionMode = ["通常", "VR180", "360"].contains(snapshot.previewProjectionMode ?? "")', source)
        self.assertIn('previewVideoLayout = ["Mono", "SBS 左右", "上下"].contains(snapshot.previewVideoLayout ?? "")', source)
        self.assertIn('previewEye = ["左目", "右目"].contains(snapshot.previewEye ?? "")', source)
        self.assertIn("previewCameraFOV = min(max(snapshot.previewCameraFOV ?? 60, 45), 105)", source)
        self.assertNotIn("var log: String", source.split("struct MiohUserDefaultsSnapshot: Codable", 1)[1].split("@MainActor", 1)[0])
        self.assertNotIn("var progress: Double", source.split("struct MiohUserDefaultsSnapshot: Codable", 1)[1].split("@MainActor", 1)[0])

    def test_realtime_player_can_use_vrviewer_projection_controls(self):
        player = PLAYER_SOURCE.read_text()
        app = APP_SOURCE.read_text()

        for contract in [
            "import SceneKit",
            "import Metal",
            "import CoreVideo",
            "enum PreviewProjectionMode: String, CaseIterable, Identifiable",
            'case vr180 = "VR180"',
            'case sphere360 = "360"',
            "enum PreviewVideoLayout: String, CaseIterable, Identifiable",
            'case sbs = "SBS 左右"',
            'case topBottom = "上下"',
            "enum PreviewEye: String, CaseIterable, Identifiable",
            "PreviewProjectionGeometry.makeSphere",
            "static func uvWindow(layout: PreviewVideoLayout, eye: PreviewEye) -> CGRect",
            "uv.maxX - uv.width * u",
            "struct VRPreviewSceneView: NSViewRepresentable",
            "final class Coordinator: NSObject",
            "pixelBufferAttributes: CVPixelBufferAttributes(",
            "videoOutput.pixelBufferAndDisplayTime(forItemTime: itemTime)",
            "CVMetalTextureCacheCreateTextureFromImage",
            "CVMetalTextureGetTexture(videoTexture)",
            "videoNode.geometry?.firstMaterial?.diffuse.contents = metalTexture",
            "final class Coordinator: NSObject, SCNSceneRendererDelegate",
            "NSPanGestureRecognizer",
            "NSMagnificationGestureRecognizer",
            'runner.previewProjectionMode == "通常"',
            "VRPreviewSceneView(",
            "private func prepareSourcePlayerItem(",
            ".load(.isPlayable)",
            "import Network",
            "private final class HEV1LoopbackServer",
            "private static func findHEV1Offsets(in url: URL, fileSize: UInt64) throws -> [UInt64]",
            'URL(string: "http://127.0.0.1:',
            'header += "Accept-Ranges: bytes\\r\\n"',
            'header += "Content-Range: bytes \\(start)-\\(end)/\\(fileSize)\\r\\n"',
            "private func parseRangeHeader(_ line: String) -> ClosedRange<UInt64>?",
            "content: patch(sourceData, startingAt: cursor)",
            "HEV1LoopbackServer(sourceURL: input)",
            "AVPlayerItem(asset: compatibleAsset)",
            "AVFoundation互換MP4へremux中",
            '"-c:v", "copy"',
            '"-tag:v", "hvc1"',
            '"-c:v", "hevc_videotoolbox"',
            "processingInputURL: compatibleURL",
            "private func startSourceOnlyPlayback(",
            'sourceOnlyPlayback = runner.previewProjectionMode != "通常"',
            '"VR再生: 復元モデルを読み込まず、元動画を直接再生します\\n"',
            '"VR再生: 全編remuxを行わず、AVFoundation互換の仮想コンテナを使用します\\n"',
            "item.preferredForwardBufferDuration",
            "private func installSourcePlaybackObservers(item: AVPlayerItem, generation: Int)",
            "private func updateSourcePlaybackState(item: AVPlayerItem, generation: Int)",
            "guard sourceOnlyPlayback, sourceSeekNeedsBuffer, shouldPlay, state == .buffering else",
            "updateSourceBufferedDuration()\n      // AVPlayer does not guarantee another loadedTimeRanges",
            "resumeSourceAfterSeekIfBuffered()\n      return",
            "sourcePlayer.reasonForWaitingToPlay",
            "controller.sourceOnlyPlayback || controller.showOriginal",
            "var shouldShowProcessingOverlay: Bool",
            "var processingOverlayLabel: String",
            "var statusLabel: String",
            "Text(controller.processingOverlayLabel)",
            'Picker("表示", selection: $runner.previewProjectionMode)',
            'Picker("形式", selection: $runner.previewVideoLayout)',
            'Picker("目", selection: $runner.previewEye)',
            'Text("視野角")',
            "private enum PreviewVRDetector",
            "static func detect(url: URL) async -> PreviewVRDetection",
            '"vr180", "vr_180", "vr-180", "180vr", "180_vr", "180-vr"',
            'of: #"(^|[^a-z0-9])mdvr[-_ ]?[0-9]+"#',
            '"gspherical", "spherical=true", "sv3d", "equirectangular"',
            '"st3d", "stereo_mode=sbs", "stereo_mode=top-bottom"',
            'private static func containsMP4Box(_ data: Data, type: String) -> Bool',
            "@Published var isVRVideo = false",
            "@Published var isDetectingVR = false",
            "func choosePreviewInput(runner: RestorationRunner)",
            "if controller.isVRVideo",
            "ForEach(PreviewProjectionMode.allCases.filter { $0 != .normal })",
            "controller.previewInputURL == nil || controller.isDetectingVR",
        ]:
            self.assertIn(contract, player)

        for removed_blocking_remux_contract in [
            "prepareSourcePlaybackURL",
            "shouldRemuxHEVCForAVFoundation",
            "waitUntilExit()",
            "AVMutableComposition",
            "guard try await compatibleAsset.load(.isPlayable)",
            "AVAssetResourceLoaderDelegate",
            "requestsAllDataToEndOfResource",
            'string: "mioh-hev1://',
        ]:
            self.assertNotIn(removed_blocking_remux_contract, player)

        for contract in [
            '@Published var previewProjectionMode = "通常"',
            '@Published var previewVideoLayout = "SBS 左右"',
            '@Published var previewEye = "左目"',
            "@Published var previewCameraFOV = 60.0",
            'previewProjectionMode: "通常"',
            'previewVideoLayout: "SBS 左右"',
            'previewEye: "左目"',
            "previewCameraFOV: 60",
        ]:
            self.assertIn(contract, app)

    def test_realtime_player_build_links_scenekit(self):
        script = BUILD_SCRIPT.read_text()
        player = PLAYER_SOURCE.read_text()

        for framework in ["SceneKit", "Metal"]:
            self.assertIn(f"-framework {framework}", script)
        self.assertNotIn("-framework SpriteKit", script)
        self.assertNotIn("import SpriteKit", player)
        self.assertNotIn("SKVideoNode", player)

    def test_preview_buffer_slider_supports_one_minute_and_live_updates(self):
        app = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "@Published var previewBufferLimit = 8.0",
            "bufferLimitSeconds: previewBufferLimit",
        ]:
            self.assertIn(contract, app)
        for contract in [
            "func setBufferLimit(_ seconds: Double)",
            '["command": "set_buffer_limit", "seconds": seconds]',
            "let seconds: Double?",
            'case "buffer_limit":',
            'runner?.appendExternalLog("プレビューバッファ上限を適用: \\(Int(seconds))秒\\n")',
            'Text("バッファ上限")',
            "in: 1...60",
            "step: 1",
            "controller.setBufferLimit(value)",
            'Text("\\(Int(runner.previewBufferLimit))秒")',
        ]:
            self.assertIn(contract, player)

    def test_app_supports_processing_without_segment_copy(self):
        source = APP_SOURCE.read_text()

        self.assertIn("@Published var noSplit = false", source)
        self.assertIn('Toggle("分割しない", isOn: $runner.noSplit)', source)
        self.assertIn('let splitMode = noSplit ? "none"', source)
        self.assertIn(".disabled(runner.noSplit)", source)

    def test_native_preview_honors_selected_models(self):
        source = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        self.assertIn("var previewRestorationModel: String?", source)
        self.assertIn("@Published var previewRealtimeOptimization = true", source)
        self.assertIn(
            '"basicvsrpp-v1.2-coreai-variable"',
            source,
        )
        self.assertIn(
            "let selectedPreviewModel = previewRestorationModel",
            source,
        )
        self.assertIn(
            "let previewModel = selectedPreviewModel",
            source,
        )
        self.assertIn(
            "let selectedPreviewDetectionModel = previewDetectionModel",
            source,
        )
        self.assertIn("model: previewModel", source)
        self.assertIn("model: selectedPreviewDetectionModel", source)
        self.assertIn('previewRestorationModel != "カスタム"', source)
        self.assertIn('previewDetectionModel != "カスタム"', source)
        self.assertIn("@Published var previewRestorationModel: String", source)
        self.assertIn("@Published var previewDetectionModel: String", source)
        self.assertIn('Picker("復元モデル"', player)
        self.assertIn(
            "ForEach(runner.previewRestorationModels, id: \\.self)",
            player,
        )
        self.assertNotIn("usesPythonEngine", player)
        self.assertIn('Picker("再生用検出モデル"', player)
        self.assertIn(
            'Toggle("リアルタイム最適化", isOn: $runner.previewRealtimeOptimization)',
            player,
        )
        self.assertIn("var preservesRealtimeCompositeParameters: Bool", source)
        self.assertIn("#if MIOH_PORTABLE_COREAI", source)
        self.assertIn("let skipsCompositeParameters = previewRealtimeOptimization", source)
        self.assertNotIn(
            "let skipsCompositeParameters = previewRealtimeOptimization\n"
            "      && !preservesRealtimeCompositeParameters",
            source,
        )
        self.assertNotIn("previewArguments(", source)
        self.assertIn(
            "let effectiveUpscale = skipsCompositeParameters ? 1 : effectUpscale",
            source,
        )
        for contract in [
            "let effectiveBlendFeather = blendFeather",
            "let effectiveSharpenStrength = skipsCompositeParameters ? 0 : sharpenStrength",
            "let effectiveDetailBoost = skipsCompositeParameters ? 0 : detailBoost",
            "let effectiveTextureMix = skipsCompositeParameters ? 0 : textureMix",
            "let effectiveSmoothStrength = skipsCompositeParameters ? 0 : smoothStrength",
            "blendFeather: Float(effectiveBlendFeather)",
            "sharpenStrength: Float(effectiveSharpenStrength)",
            "detailBoost: Float(effectiveDetailBoost)",
            "textureMix: Float(effectiveTextureMix)",
            "smoothStrength: Float(effectiveSmoothStrength)",
        ]:
            self.assertIn(contract, source)
        self.assertIn(
            "sharpen > 0 || detail > 0 || texture > 0 || smoothing > 0 || upscale > 1",
            NATIVE_PIPELINE_SOURCE.read_text(),
        )
        self.assertIn("private var activePreviewSettingsSignature: String?", player)
        self.assertIn(
            "activePreviewSettingsSignature = previewSettingsSignature(for: runner)",
            player,
        )
        self.assertIn("private func shouldRestartPreviewForCurrentSettings()", player)
        self.assertIn("private func previewSettingsSignature(for runner: RestorationRunner) -> String", player)
        self.assertIn(
            "if shouldRestartPreviewForCurrentSettings(),\n"
            "        let runner\n"
            "      {\n"
            "        shouldPlay = true\n"
            "        restartWithCurrentSettings(runner: runner)\n"
            "        return\n"
            "      }",
            player,
        )
        self.assertIn('if !controller.isVRVideo', player)
        for contract in [
            "let usesVariableTemporalModel = restoration.fixedFrameCount == nil",
            "let temporalLimit = usesVariableTemporalModel",
            "? 48",
        ]:
            self.assertIn(contract, source)
        self.assertIn(
            '"basicvsrpp-v1.2-coreai-t90"',
            source,
        )

    def test_main_app_targets_macos_26_without_linking_coreai(self):
        source = APP_SOURCE.read_text()
        build_script = BUILD_SCRIPT.read_text()
        with INFO_PLIST.open("rb") as handle:
            info = plistlib.load(handle)

        self.assertEqual(info["LSMinimumSystemVersion"], "26.0")
        self.assertNotIn("import CoreAI", source)
        self.assertEqual(build_script.count("-target arm64-apple-macosx26.0"), 1)
        self.assertEqual(build_script.count("-target arm64-apple-macosx27.0"), 4)
        self.assertEqual(build_script.count("-framework CoreAI"), 4)

    def test_model_choices_follow_coreai_os_availability(self):
        source = APP_SOURCE.read_text()

        self.assertIn("struct PlatformCapabilities", source)
        self.assertIn("operatingSystemVersion.majorVersion >= 27", source)
        self.assertIn(
            '"basicvsrpp-v1.2-coreai-t90"',
            source,
        )
        self.assertIn(
            "var restorationModels: [String] {\n    coreAIRestorationModels",
            source,
        )
        for name in (
            "v2-coreai",
            "v3.1-fast-coreai",
            "v3.1-accurate-coreai",
            "v4-fast-coreai",
            "v4-accurate-coreai",
            "vr-v2-accurate-coreai",
        ):
            self.assertIn(f'"{name}"', source)
        self.assertIn('"vr-v2-accurate-coreml"', source)
        self.assertIn('#if !MIOH_PORTABLE_COREAI', source)
        self.assertIn('models.append("jasna-v6-coreai")', source)
        self.assertIn('models.append("jasna-v6-large-coreai")', source)
        base_models = source.split("let baseDetectionModels = [", 1)[1].split("]", 1)[0]
        self.assertNotIn('"v2"', base_models)
        self.assertNotIn('"v4-fast"', base_models)
        self.assertNotIn('"vr-v2-accurate"', base_models)
        self.assertIn("normalizeModelSelections()", source)

    def test_start612_and_large_roi_runtime_are_excluded(self):
        source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()
        excluded_model = "basicvsrpp-v1.2-start612-coreai-variable"
        excluded_hq_model = "basicvsrpp-v1.2-coreai-variable-hq"

        self.assertNotIn(excluded_model, source)
        self.assertNotIn(excluded_hq_model, source)
        self.assertNotIn("nativeLargeROITiles", source + pipeline)
        self.assertNotIn("restoreNativeTiles", pipeline)
        self.assertNotIn("NativeTile", pipeline)
        self.assertNotIn("大ROI最適化", source)

        english = (
            ROOT
            / "packaging/macOS/standalone/Localizations/en.lproj/Localizable.strings"
        ).read_text()
        traditional_chinese = (
            ROOT
            / "packaging/macOS/standalone/Localizations/zh-Hant.lproj/Localizable.strings"
        ).read_text()
        self.assertNotIn("START-612", english)
        self.assertNotIn("START-612", traditional_chinese)

    def test_rfdetr_coreai_stays_dedicated_while_coreml_is_portable(self):
        source = APP_SOURCE.read_text()
        build_script = BUILD_SCRIPT.read_text()
        restorer = (
            ROOT / "lada" / "restorationpipeline" / "frame_restorer.py"
        ).read_text()
        detector = (
            ROOT / "lada" / "restorationpipeline" / "mosaic_detector.py"
        ).read_text()

        self.assertIn(
            '#if !MIOH_PORTABLE_COREAI\n'
            '    models.append("jasna-v6-coreml")\n'
            '    models.append("jasna-v6-large-coreml")\n'
            '    models.append("jasna-v6-coreai")\n'
            '    models.append("jasna-v6-large-coreai")',
            source,
        )
        self.assertIn(
            'if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]',
            build_script,
        )
        self.assertIn("rfdetr-v6-576-fp32.aimodel", build_script)
        self.assertIn("rfdetr-v6-large-768-fp32.aimodel", build_script)
        self.assertIn("rfdetr-v6-576-fp32.mlpackage", build_script)
        self.assertIn("rfdetr-v6-large-768-fp32.mlpackage", build_script)
        self.assertIn(
            'if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then\n'
            '  COREML_DETECTION_ASSETS+=(',
            build_script,
        )
        self.assertIn("-iname '*rfdetr*' -delete", build_script)
        self.assertNotIn("$RESOURCES/runtime", build_script)
        self.assertNotIn("calculate_frame_detection_queue_size", restorer)
        self.assertNotIn("pipeline_queue_depth", detector)
        self.assertIn("maxsize=8", detector)

    def test_rfdetr_uses_native_swift_coreai_and_coreml_contract(self):
        app = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        # Both backends resolve to the same fixed-shape RF-DETR contract. The
        # Core ML medium model is also available to Universal and preview.
        self.assertIn('if base == "jasna-v6" || base == "jasna-v6-large"', app)
        self.assertIn('"rfdetr-v6-576-fp32"', app)
        self.assertIn('"rfdetr-v6-large-768-fp32"', app)
        self.assertIn('large ? 768 : 576', app)
        self.assertIn('large ? 0.40 : 0.35', app)
        self.assertIn('$0.hasSuffix("-coreml")', app)
        self.assertIn('"cpuAndGPU"', app)

        # RF-DETR has different preprocessing and output semantics from YOLO:
        # direct square resize, ImageNet-normalized FP32 NCHW input, per-query
        # logits/masks, and direct mask projection without letterboxing.
        self.assertIn(
            'private final class RFDETRDetector: NativeDetecting',
            pipeline,
        )
        self.assertIn('config.detectionBackend == "rfdetr"', pipeline)
        self.assertIn('MLFeatureValue(multiArray: input)', pipeline)
        self.assertIn('shape: [1, 3, resolution, resolution]', pipeline)
        self.assertIn('scalarType: .float32', pipeline)
        self.assertIn('expectedShape: [1, queries, 4]', pipeline)
        self.assertIn(
            'expectedShape: [1, queries, logitClasses]', pipeline
        )
        self.assertIn(
            'expectedShape: [1, queries, maskSize, maskSize]', pipeline
        )
        self.assertIn('maskProjection: .directResize', pipeline)
        self.assertIn('maskThreshold: 0', pipeline)

    def test_coreai_helper_environment_is_only_exported_when_supported(self):
        source = APP_SOURCE.read_text()

        self.assertIn("guard capabilities.supportsCoreAI else", source)
        self.assertIn('"bin/mioh-native-coreai-preview"', source)
        self.assertIn(
            "try rejectUnsupportedCoreAIModel(selectedPreviewModel)", source
        )
        self.assertNotIn("PYTHONHOME", source)
        self.assertNotIn("runtime/bin/python", source)

    def test_app_exports_m5_pro_coreai_architecture(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn('COREAI_ARCHITECTURE="${COREAI_ARCHITECTURE:-h17s}"', script)
        source = APP_SOURCE.read_text()
        self.assertNotIn("LADA_COREAI_ARCHITECTURE", source)
        self.assertNotIn(
            'nativeEnvironment["LADA_COREAI_ARCHITECTURE"]', source
        )

    def test_build_targets_only_m5_pro_coreai_specialization(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn(
            'COREAI_ARCHITECTURE="${COREAI_ARCHITECTURE:-h17s}"', script
        )
        self.assertIn(
            'DEDICATED_PREBUILT_MODELS="${DEDICATED_PREBUILT_MODELS:-'
            '$ROOT/model_weights/mioh-dedicated-$COREAI_ARCHITECTURE}"',
            script,
        )
        self.assertIn(
            'basicvsrpp-v1.2-t90-fp16.$COREAI_ARCHITECTURE.aimodelc',
            script,
        )
        self.assertNotIn("basicvsrpp-v1.2-t36-b2-fp16.aimodel", script)
        for source in EXPECTED_COREAI_SOURCES:
            self.assertIn(source, script)

    def test_build_runs_all_coreai_smoke_tests_before_signing(self):
        script = BUILD_SCRIPT.read_text()

        verifier = script.index('"$RESOURCES/bin/mioh-dedicated-model-verifier"')
        signing = script.index('codesign --force --deep --sign - "$APP"')
        self.assertLess(verifier, signing)
        self.assertIn('"$RESOURCES/models"', script)
        self.assertIn('"$COREAI_ARCHITECTURE"', script)
        self.assertIn('DedicatedModelVerifier.swift', script)

    def test_build_supports_dedicated_and_portable_coreai_distributions(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn('COREAI_DISTRIBUTION="${COREAI_DISTRIBUTION:-dedicated}"', script)
        self.assertIn('dedicated|portable)', script)
        self.assertIn('if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]', script)
        self.assertIn('ditto "$source_model" "$RESOURCES/models/$asset"', script)
        self.assertIn('--distribution "$COREAI_DISTRIBUTION"', script)
        self.assertIn('--smoke-model basicvsrpp-v1.2-coreai', script)

    def test_build_keeps_dedicated_and_portable_checkpoints_separate(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn(
            '$ROOT/model_weights/mioh-dedicated-$COREAI_ARCHITECTURE',
            script,
        )
        self.assertIn(
            '$ROOT/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth',
            script,
        )
        self.assertIn('Missing portable variable restoration checkpoint:', script)
        self.assertNotIn(
            'basicvsrpp-v1.2-large-roi-native-tiles-27000-ema.pth',
            script,
        )

    def test_build_records_variable_restoration_checkpoint_provenance(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn(
            'basicvsrpp-v1.2-variable-coreai.provenance.json', script
        )
        self.assertIn(
            'source_metadata="$DEDICATED_PREBUILT_MODELS/'
            'basicvsrpp-v1.2-standard-variable-coreai.provenance.json"',
            script,
        )
        self.assertIn(
            '"$RESOURCES/models/basicvsrpp-v1.2-variable-coreai.provenance.json"',
            script,
        )

    def test_build_packages_pre_large_roi_standard_baseline_only_for_dedicated_coreai(self):
        script = BUILD_SCRIPT.read_text()
        self.assertIn(
            "basicvsrpp-v1.2-standard-variable-coreai.$COREAI_ARCHITECTURE.aimodelc",
            script,
        )
        self.assertIn(
            "basicvsrpp-v1.2-standard-variable-coreai.provenance.json",
            script,
        )
        self.assertIn(
            'ditto "$source_standard_variable" '
            '"$RESOURCES/models/$active_variable_asset"',
            script,
        )
        self.assertNotIn("MIOH_DEDICATED_LARGE_ROI", script)
        self.assertNotIn("start612", script.lower())
        self.assertNotIn("variable-hq", script.lower())
        self.assertNotIn("VariableBasicVSRPPRunner.swift", script)

    def test_no_distribution_bundles_a_python_runtime(self):
        script = BUILD_SCRIPT.read_text()

        self.assertNotIn("MIOH_BUNDLE_PYTHON_RUNTIME", script)
        self.assertNotIn("$RESOURCES/runtime", script)
        self.assertNotIn("process_video_parallel.py", script)
        self.assertNotIn("mioh_preview_worker.py", script)

    def test_build_time_python_is_required_only_for_universal_models(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn(
            'if [[ "$COREAI_DISTRIBUTION" == "portable"',
            script,
        )
        self.assertIn("Missing build-time Python:", script)
        self.assertNotIn("Missing interpreter to bundle:", script)

    def test_dedicated_build_uses_only_prebuilt_native_models_and_swift_verification(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn("DEDICATED_PREBUILT_MODELS", script)
        self.assertIn("DEDICATED_COREAI_ASSETS=(", script)
        self.assertIn("DedicatedModelVerifier.swift", script)
        self.assertIn("mioh-dedicated-model-verifier", script)
        self.assertIn("Dedicated app unexpectedly contains Python files:", script)
        self.assertIn("Dedicated app unexpectedly contains Python checkpoints:", script)
        dedicated_model_branch = script.split(
            'if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then\n'
            '  DEDICATED_COREAI_ASSETS=(',
            1,
        )[1].split("\nelse\n", 1)[0]
        self.assertNotIn("python", dedicated_model_branch.lower())
        self.assertNotIn(".pth", dedicated_model_branch)

    def test_mioh_keeps_only_one_mewzoom_coreml_asset(self):
        script = BUILD_SCRIPT.read_text()
        regular_assets = script.split("MODEL_ASSETS=(", 1)[1].split(")", 1)[0]

        self.assertIn("MewZoom-V1-4X-Unet_256.mlpackage", regular_assets)
        self.assertNotIn("MewZoom-V1-4X-Unet_512.mlpackage", regular_assets)

    def test_build_uses_python_only_for_model_generation(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn("LADA_STANDALONE_PYTHON_ENV", script)
        self.assertIn("$ROOT/.venv-coreai", script)
        self.assertIn('if [[ "$COREAI_DISTRIBUTION" == "portable"', script)
        self.assertNotIn('if [[ -d "$ROOT/.venv" ]]', script)
        self.assertNotIn('$ROOT/.venv/lib/python3.12/site-packages', script)

    def test_universal_build_wrapper_uses_isolated_artifact_paths(self):
        self.assertTrue(UNIVERSAL_BUILD_SCRIPT.is_file())
        script = UNIVERSAL_BUILD_SCRIPT.read_text()

        self.assertIn('ROOT="${PACKAGE_DIR:h:h:h}"', script)
        self.assertIn('COREAI_DISTRIBUTION="portable"', script)
        self.assertIn('build/macos-standalone-universal', script)
        self.assertIn('APP_BASENAME="mioh-universal"', script)
        self.assertIn('DMG_BASENAME="mioh-universal-0.14.3-unsigned"', script)
        self.assertIn('exec "$PACKAGE_DIR/build_app.sh"', script)

    def test_portable_swift_build_omits_architecture_override(self):
        source = APP_SOURCE.read_text()
        script = BUILD_SCRIPT.read_text()

        self.assertIn("#if !MIOH_PORTABLE_COREAI", source)
        self.assertNotIn("LADA_COREAI_ARCHITECTURE", source)
        self.assertIn("-D MIOH_PORTABLE_COREAI", script)

    def test_target_frame_rate_inherits_the_source_timebase(self):
        source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        # The picked value is the exact rate, carried as a rational end to end.
        self.assertNotIn("ntscFPS", source)
        self.assertIn("@Published var fpsDenominator = 1", source)
        self.assertIn("targetFPS: useFPS ? max(1, fps) : nil", source)
        self.assertIn(
            "targetFPSDenominator: useFPS ? max(1, fpsDenominator) : nil",
            source,
        )
        self.assertIn("let targetFPSDenominator: Int?", pipeline)
        self.assertIn(
            "return (max(1, requested), max(1, denominator))", pipeline
        )

        # A configuration without a denominator still resolves against the
        # source timebase rather than assuming a whole rate.
        self.assertIn("enum NTSCFrameRate {", pipeline)
        self.assertIn(
            "static func isNTSC(numerator: Int, denominator: Int) -> Bool",
            pipeline,
        )
        self.assertIn("return (whole * 1000, 1001)", pipeline)
        self.assertIn("return (whole, 1)", pipeline)
        self.assertIn("NTSCFrameRate.target(", pipeline)
        self.assertIn("sourceNumerator: video.fpsNumerator", pipeline)

        # minFrameDuration reports the shortest observed gap, which reads far
        # above the real rate on VFR sources. Never use it as the source rate.
        self.assertNotIn("load(.minFrameDuration)", pipeline)
        self.assertIn("let frameRate = try await track.load(.nominalFrameRate)", pipeline)

        # The gate uses absolute rational time slots, so cluster shards do not
        # restart the conversion phase at their own boundaries.
        self.assertIn("init(numerator: Int, denominator: Int) {", pipeline)
        self.assertIn("private var lastSlot: Int64?", pipeline)
        self.assertIn(
            "Double(max(0, ptsNanoseconds)) * numerator / denominatorNanoseconds",
            pipeline,
        )
        self.assertIn("+ 1e-8", pipeline)
        self.assertIn("guard slot != lastSlot else { return false }", pipeline)
        self.assertIn(
            "let outputFPSNumerator = targetRate?.numerator ?? video.fpsNumerator",
            pipeline,
        )
        self.assertIn(
            "let outputFPSDenominator = targetRate?.denominator "
            "?? video.fpsDenominator",
            pipeline,
        )
        # Selecting the source's existing rational rate is a no-op. Sending
        # jittered HLS timestamps through the down-conversion gate used to
        # drop frames and compact the video ahead of its original audio.
        self.assertIn("static func matches(", pipeline)
        self.assertIn("let requestedTargetRate:", pipeline)
        self.assertIn(
            "SourceFrameRate.matches(requested, sourceRate) ? nil : requested",
            pipeline,
        )

        # Full exports must not be reported as successful when muxing exposes
        # a new video/audio duration skew.
        self.assertIn("validateExportSynchronization(", pipeline)
        self.assertIn("let addedDrift = abs(outputDifference - sourceDifference)", pipeline)
        self.assertIn("guard addedDrift <= 0.5", pipeline)
        self.assertIn("音声同期の検証に失敗しました", pipeline)

        # VideoToolbox must retain that exact rational in the encoded track.
        # Its default 19,200 time scale quantizes 29.97fps into periodic
        # 35-millisecond frames and adds visible cadence jitter.
        encoder = PREVIEW_ENCODER_SOURCE.read_text()
        self.assertIn("let exactTimeScale = CMTimeScale(fpsNumerator)", encoder)
        self.assertIn("writer.movieTimeScale = exactTimeScale", encoder)
        self.assertIn("input.mediaTimeScale = exactTimeScale", encoder)

    def test_native_decoder_recovers_malformed_h264_display_order(self):
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        # Some H.264 files in the field contain B-slices while advertising
        # PTS == DTS. AVFoundation then yields decoded pixels in coding order.
        # The native lane must inspect a separate compressed sidecar and put
        # those zero-copy pixel buffers back into display order.
        self.assertIn("track.load(\n      .requiresFrameReordering", pipeline)
        self.assertIn("AVAssetReaderTrackOutput(\n        track: track,\n        outputSettings: nil", pipeline)
        self.assertIn("CMSampleBufferGetSampleAttachmentsArray", pipeline)
        self.assertIn("kCMSampleAttachmentKey_DoNotDisplay", pipeline)
        self.assertIn("H264SequenceParameters.maximumReorderFrames", pipeline)
        self.assertIn("H264SampleOrder.classification", pipeline)
        self.assertIn("MalformedH264FrameReorderBuffer", pipeline)
        self.assertIn("timelinePTS.append(frame.ptsNanoseconds)", pipeline)
        self.assertIn("reorderBuffer.finish", pipeline)
        self.assertIn('"H.264 reference-B display recovery is unsupported"', pipeline)
        self.assertIn('"h264_display_order_recovery":', pipeline)

        # AVAssetReader seek preroll can contain empty or non-display samples;
        # consuming either as a visible frame desynchronizes the two readers.
        self.assertIn("if sampleCount == 0", pipeline)
        self.assertIn("guard sampleCount == 1", pipeline)

        # Recovery is deliberately limited to the progressive, one-frame
        # reorder structure that the fixed buffer can reproduce exactly.
        self.assertIn("guard frameOnly == 1", pipeline)
        self.assertIn("maximumReorderFrames == 1", pipeline)
        self.assertIn(
            '"H.264 display-order readers lost sample synchronization"',
            pipeline,
        )
        self.assertIn(
            '"H.264 display-order sidecar has an extra visible sample"',
            pipeline,
        )

    def test_basic_tab_reports_the_input_media_details(self):
        source = APP_SOURCE.read_text()

        self.assertIn("struct SourceMediaInfo: Equatable, Sendable", source)
        self.assertIn("enum SourceMediaProbe {", source)
        self.assertIn("struct SourceInfoRow: View {", source)
        self.assertIn("SourceInfoRow(", source)
        # Shown right under the input path it describes.
        self.assertIn(
            'PathRow(\n          title: "入力",\n'
            '          icon: "film",\n'
            "          url: runner.inputURL,\n"
            "          action: chooseInput,\n"
            '          actionLabel: "入力を選択…"\n'
            "        )\n        if runner.inputURL != nil {",
            source,
        )
        for field in ["解像度", "フレームレート", "長さ", "コーデック", "ビットレート", "音声"]:
            self.assertIn(f'field("{field}"', source)
        # The probe reports; it must not feed the pipeline's own decisions.
        self.assertIn("nominalFrameRate", source)
        self.assertNotIn("sourceInfo.frameRate", source)
        self.assertNotIn("targetFPS: sourceInfo", source)
        # A slow probe must not land on a newer selection.
        self.assertIn("guard let self, self.inputURL == url else { return }", source)

    def test_frame_rate_is_picked_as_an_exact_rate(self):
        source = APP_SOURCE.read_text()

        # A picker over real rates, not an integer stepper reinterpreted later.
        self.assertIn("struct FrameRateOption: Identifiable, Hashable", source)
        self.assertIn("FrameRateOption(numerator: 30000, denominator: 1001)", source)
        self.assertIn("FrameRateOption(numerator: 60000, denominator: 1001)", source)
        self.assertIn("Picker(\"\", selection: $runner.selectedFrameRate)", source)
        self.assertIn('Text("\\(option.label)fps").tag(option.key)', source)
        self.assertNotIn("Stepper(value: $runner.fps", source)
        self.assertIn('return String(format: "%.3f", value)', source)

        # The table is fixed. Probing the input and filtering by it only hid
        # the rate the user wanted whenever the probe was wrong.
        self.assertNotIn("sourceFrameRate", source)
        self.assertNotIn("minFrameDuration", source)
        for label in ["23.976", "29.970", "59.940", "119.880"]:
            numerator = int(round(float(label) * 1.001)) * 1000
            self.assertIn(
                f"FrameRateOption(numerator: {numerator}, denominator: 1001)",
                source,
            )
        for whole in [24, 25, 30, 48, 50, 60, 100, 120]:
            self.assertIn(
                f"FrameRateOption(numerator: {whole}, denominator: 1)", source
            )

        self.assertNotIn("pythonTargetFPS", source)
        self.assertNotIn('add(&args, "--fps"', source)

    def test_native_export_uses_internal_stage_concurrency(self):
        source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        self.assertNotIn("guard parallelWorkers == 1", source)
        self.assertIn('parallelWorkers = 1', source)
        self.assertIn("@Published var nativeParallelWorkers = 1", source)
        self.assertIn(
            'Picker("ネイティブ並列数", selection: '
            "$runner.nativeParallelWorkers)",
            source,
        )
        for lane in [1, 2, 3]:
            self.assertIn(f'.tag({lane})', source)
        self.assertIn(
            "nativeParallelWorkers: min(max(nativeParallelWorkers, 1), 3)",
            source,
        )
        self.assertIn("nativeParallelWorkers: 1,", source)
        self.assertIn('executor = "process"', source)
        self.assertIn('mergeEncoder = "copy"', source)
        self.assertIn("Swiftネイティブ（段階並列）", source)
        self.assertIn("let nativeParallelWorkers: Int?", pipeline)
        self.assertIn(
            "config.isExport && !config.isWorker", pipeline
        )
        self.assertIn(
            "min(max(config.nativeParallelWorkers ?? 1, 1), 3)",
            pipeline,
        )
        self.assertIn(
            "processors.reserveCapacity(nativeParallelWorkers)", pipeline
        )
        self.assertNotIn("nativeTileParallelism", pipeline)
        self.assertNotIn("NativeTileRestoration", pipeline)
        self.assertNotIn("空間タイル", source)
        self.assertIn(
            "value: nativeParallelWorkers", pipeline
        )
        self.assertIn("func launchProcessing(", pipeline)
        self.assertIn("func drainNextProcessing() async throws", pipeline)
        self.assertIn("func flushProcessing() async throws", pipeline)
        self.assertIn(
            "let pending = pendingProcessing.removeFirst()",
            pipeline,
        )
        self.assertIn(
            "try await encodeProcessedBatch(pending, result: result)",
            pipeline,
        )
        self.assertIn(
            "if pendingProcessing.count == nativeParallelWorkers {\n"
            "            try await drainNextProcessing()",
            pipeline,
        )
        self.assertIn(
            "while !pendingProcessing.isEmpty {\n"
            "          try await drainNextProcessing()",
            pipeline,
        )
        self.assertNotIn(
            "if pendingProcessing.count == nativeParallelWorkers {\n"
            "            try await flushProcessing()",
            pipeline,
        )
        self.assertIn(
            '"native_parallel_workers": nativeParallelWorkers', pipeline
        )

    def test_native_large_roi_tile_compositor_is_removed(self):
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        for forbidden in [
            "restoreNativeTiles",
            "NativeTile",
            "nativeTile",
            "restorationTileStride",
            "native_large_roi_tiles",
        ]:
            self.assertNotIn(forbidden, pipeline)

    def test_native_detection_mask_reuse_applies_to_export_and_realtime(self):
        source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        for contract in [
            "@Published var detectionMaskReuseSkipFrames = 0",
            "var detectionMaskReuseSkipFrames: Int?",
            'LabeledContent("検出後のマスク再利用")',
            "value: $runner.detectionMaskReuseSkipFrames",
            "in: 0...8",
            "detectionMaskReuseSkipFrames: min(",
            "max(detectionMaskReuseSkipFrames, 0)",
            "detectionMaskReuseSkipFrames: detectionMaskReuseSkipFrames",
            "snapshot.detectionMaskReuseSkipFrames ?? 0",
            "検出後スキップ: \\(configuration.detectionMaskReuseSkipFrames)フレーム",
        ]:
            self.assertIn(contract, source)
        self.assertGreaterEqual(
            source.count("detectionMaskReuseSkipFrames: min("),
            3,
        )

        for contract in [
            "let detectionMaskReuseSkipFrames: Int?",
            "max(config.detectionMaskReuseSkipFrames ?? 0, 0)",
            "if detectionMaskReuseSkipFrames > 0",
            "let lastDetections = try await inferDetections(",
            "if firstDetections.isEmpty && lastDetections.isEmpty",
            "let sampleStride = detectionMaskReuseSkipFrames + 1",
            "0: firstDetections",
            "lastIndex: lastDetections",
            "detections: reusedDetections",
        ]:
            self.assertIn(contract, pipeline)
        reuse_branch = pipeline.split(
            "if detectionMaskReuseSkipFrames > 0", 1
        )[1].split("if !firstDetections.isEmpty", 1)[0]
        self.assertLess(
            reuse_branch.index("lastDetections = try await inferDetections"),
            reuse_branch.index(
                "if firstDetections.isEmpty && lastDetections.isEmpty"
            ),
        )
        self.assertNotIn(
            ".disabled(runner.detectionMaskReuseSkipFrames > 0)",
            source,
        )

    def test_native_engine_is_the_only_runtime_engine(self):
        source = APP_SOURCE.read_text()

        self.assertIn('device = "mps"', source)
        self.assertIn("fp16 = true", source)
        self.assertIn("autoOptimize = true", source)
        self.assertIn("Swiftネイティブ / Core AI", source)
        self.assertIn('restorationEngine = "native"', source)
        self.assertNotIn("usesPythonEngine", source)
        self.assertNotIn("supportsPythonEngine", source)

    def test_export_and_preview_have_no_python_runtime_path(self):
        source = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        for forbidden in [
            "bundlesPythonRuntime",
            "supportsPythonEngine",
            "usesPythonEngine",
            "runtime/bin/python",
            "mioh_preview_worker.py",
            "process_video_parallel.py",
        ]:
            self.assertNotIn(forbidden, source + player)
        self.assertIn("runner.nativePreviewInvocation(", player)
        self.assertIn("launchNativeExportPlan(", source)

    def test_product_is_named_mioh(self):
        self.assertTrue(APP_SOURCE.is_file(), "MiohApp.swift must be the app entry source")
        source = APP_SOURCE.read_text()
        build_script = BUILD_SCRIPT.read_text()
        with INFO_PLIST.open("rb") as handle:
            info = plistlib.load(handle)

        self.assertIn('Text("mioh")', source)
        self.assertIn('PathSettingRow(title: "mioh一時フォルダ"', source)
        self.assertIn("struct MiohStandaloneApp: App", source)
        self.assertEqual(info["CFBundleDisplayName"], "mioh")
        self.assertEqual(info["CFBundleName"], "mioh")
        self.assertEqual(info["CFBundleExecutable"], "mioh")
        self.assertEqual(info["CFBundleIdentifier"], "com.okatti.lada.coreai")
        self.assertIn('APP_BASENAME="${APP_BASENAME:-mioh}"', build_script)
        self.assertIn('APP="$BUILD_DIR/$APP_BASENAME.app"', build_script)
        self.assertIn('-o "$CONTENTS/MacOS/mioh"', build_script)
        self.assertIn(
            'DMG_BASENAME="${DMG_BASENAME:-mioh-0.14.3-unsigned}"',
            build_script,
        )
        self.assertIn('DMG="$BUILD_DIR/$DMG_BASENAME.dmg"', build_script)
        self.assertIn('--volumeName "$APP_BASENAME"', build_script)
        self.assertIn('ditto "$APP" "$DMG_ROOT/$APP_BASENAME.app"', build_script)

    def test_gui_exposes_all_processing_options(self):
        source = APP_SOURCE.read_text()
        expected_properties = {
            "inputURL", "outputURL", "tempDirectory", "ffmpegTempDirectory",
            "ladaTempDirectory", "parallelWorkers", "nativeParallelWorkers",
            "executor",
            "segmentDuration", "segmentCount", "mergeEncoder",
            "deleteSegments", "keepTemp", "forceSplit", "noSplit", "device",
            "fp16", "encodingPreset", "encoder", "encoderOptions",
            "bitrateMultiplier", "quality", "qmin", "qmax", "fps",
            "preFPSConversion", "mp4FastStart", "autoOptimize",
            "restorationModel", "maxClipLength", "restoreMaxFrames",
            "restoreTemporalOverlap", "restoreCrossfade", "sharpenStrength",
            "detailBoost", "blendFeather", "textureMix", "smoothStrength",
            "effectUpscale", "roiEnhancer", "roiEnhancerModel",
            "roiEnhancerScale", "roiEnhancerStrength", "roiEnhancerTile",
            "detectionModel", "detectionEmptyLookahead",
            "detectionMaskReuseSkipFrames", "detectFaceMosaics",
            "memoryCleanupInterval", "cleanupTriggerGB", "mpsMemoryFraction",
            "logMPSMemory", "overwrite",
        }

        missing = sorted(
            name for name in expected_properties
            if f"var {name}" not in source
        )
        self.assertEqual(missing, [])

    def test_gui_exposes_temporal_overlap_controls(self):
        source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        for contract in [
            "@Published var restoreTemporalOverlap = 8",
            "@Published var restoreCrossfade = true",
            'LabeledContent("Temporal overlap")',
            'Stepper(value: $runner.restoreTemporalOverlap, in: 0...120)',
            'Toggle("クロスフェードを有効化", isOn: $runner.restoreCrossfade)',
            "temporalOverlap: overlap",
            "crossfade: restoreCrossfade",
            "let temporalOverlap: Int",
            "let crossfade: Bool",
            "let previewOverlap = min(",
            "usesVariableTemporalModel ? 6 : restoreTemporalOverlap",
            "temporalOverlap: previewOverlap",
        ]:
            self.assertIn(contract, source)

        self.assertIn("let temporalOverlap: Int?", pipeline)
        self.assertIn("let crossfade: Bool?", pipeline)
        self.assertIn("max(0, config.temporalOverlap ?? 0)", pipeline)
        self.assertIn("(config.crossfade ?? false), overlap > 0", pipeline)
        self.assertNotIn("let overlap = config.isExport", pipeline)

    def test_app_defaults_to_native_swift_export(self):
        script = BUILD_SCRIPT.read_text()
        source = APP_SOURCE.read_text()

        self.assertIn('"$PACKAGE_DIR/NativePreviewPipeline.swift"', script)
        self.assertIn('@Published var restorationEngine = "native"', source)
        self.assertIn('restorationEngine: "native"', source)

    def test_app_bundles_realtime_player_and_native_preview_worker(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn('"$PACKAGE_DIR/RealtimePlayer.swift"', script)
        self.assertIn("-framework AVFoundation", script)
        self.assertIn("-framework AVKit", script)
        self.assertIn("-framework Network", script)
        self.assertIn('"$PACKAGE_DIR/NativePreviewPipeline.swift"', script)
        self.assertIn(
            '"$RESOURCES/bin/mioh-native-coreai-preview"',
            script,
        )
        source = APP_SOURCE.read_text()
        self.assertIn("func nativePreviewInvocation(", source)
        self.assertIn('"bin/mioh-native-coreai-preview"', source)

    def test_coreai_runners_expose_macos27_2_specialization_controls(self):
        runner = COREAI_RUNNER_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()
        variable = (
            ROOT
            / "packaging"
            / "macOS"
            / "standalone"
            / "VariableBasicVSRPPChunk6Runner.swift"
        ).read_text()

        for source in (runner, pipeline, variable):
            self.assertIn("MiohCoreAIModelLoader", source)
            self.assertIn('modelURL.pathExtension.lowercased() == "aimodel"', source)
            self.assertIn("AIModel.specialize(", source)
            self.assertIn("AIModel(contentsOf:", source)
            self.assertIn("options: options", source)
            self.assertIn('"MIOH_COREAI_CACHE_POLICY"', source)
            self.assertIn("AIModelCache.Policy.persistent", source)
            self.assertIn("AIModelCache.Policy(purgeConditions: [.storagePressure])", source)
            self.assertIn(
                "AIModelCache.Policy(purgeConditions: [.sourceAssetChangedOrDeleted])",
                source,
            )
            self.assertIn('"MIOH_COREAI_PREFERRED_COMPUTE"', source)
            self.assertIn('"MIOH_COREAI_EXPECT_FREQUENT_RESHAPES"', source)
            self.assertIn("result.expectFrequentReshapes = true", source)

    def test_app_allows_loopback_video_streaming(self):
        info = INFO_PLIST.read_text()
        self.assertIn("<key>NSAppTransportSecurity</key>", info)
        self.assertIn("<key>NSAllowsLocalNetworking</key>", info)

    def test_standalone_app_uses_mioh_icon(self):
        icon = ROOT / "lada" / "gui" / "icons" / "mioh-icon.png"
        script = BUILD_SCRIPT.read_text()

        self.assertTrue(icon.is_file())
        self.assertEqual(icon.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn('SOURCE_ICON="$ROOT/lada/gui/icons/mioh-icon.png"', script)

    def test_app_replaces_structured_progress_rows(self):
        source = APP_SOURCE.read_text()

        self.assertIn("struct AppProgressEvent: Decodable", source)
        self.assertIn('case "export_progress":', source)
        self.assertIn("activeProgress", source)
        self.assertIn("logHistory", source)
        self.assertIn("rebuildVisibleLog()", source)

    def test_native_stats_log_exposes_stage_timings(self):
        source = APP_SOURCE.read_text()

        for contract in [
            'payload["detection_seconds"]',
            'payload["preparation_seconds"]',
            'payload["restoration_seconds"]',
            'payload["composition_seconds"]',
            "処理内訳（工程は並行するため割合の合計は100%%になりません）",
        ]:
            self.assertIn(contract, source)

    def test_gui_has_always_visible_multiline_ffmpeg_options(self):
        source = APP_SOURCE.read_text()

        self.assertIn('Section("FFmpeg詳細設定")', source)
        self.assertIn('Text("追加FFmpegオプション")', source)
        self.assertIn('TextEditor(text: $runner.encoderOptions)', source)
        self.assertIn(
            "encoderOptions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty",
            source,
        )

    def test_restoration_effects_use_slider_number_rows(self):
        source = APP_SOURCE.read_text()
        expected_slider_rows = [
            'doubleSliderField("シャープ", value: $runner.sharpenStrength, range: 0...5, step: 0.05)',
            'doubleSliderField("ディテール", value: $runner.detailBoost, range: 0...1, step: 0.05)',
            'doubleSliderField("境界フェザー", value: $runner.blendFeather, range: 0...3, step: 0.05)',
            'doubleSliderField("テクスチャ", value: $runner.textureMix, range: 0...1, step: 0.01)',
            'doubleSliderField("スムージング", value: $runner.smoothStrength, range: 0...1, step: 0.05)',
            'doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05)',
            'integerSliderField("タイル", value: $runner.roiEnhancerTile, range: 0...1024, step: 32)',
        ]

        self.assertIn("private func doubleSliderField", source)
        self.assertIn("private func integerSliderField", source)
        for row in expected_slider_rows:
            self.assertIn(row, source)
        self.assertIn(
            'LabeledContent("エフェクト倍率") { Stepper(value: $runner.effectUpscale, in: 1...4)',
            source,
        )
        self.assertIn(
            'LabeledContent("倍率") { Stepper(value: $runner.roiEnhancerScale, in: 1...8)',
            source,
        )
        self.assertIn(
            'doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05).disabled(runner.roiEnhancer == "none")',
            source,
        )
        self.assertIn(
            'integerSliderField("タイル", value: $runner.roiEnhancerTile, range: 0...1024, step: 32).disabled(runner.roiEnhancer == "none")',
            source,
        )

    def test_native_pipeline_applies_restoration_effects_inside_roi(self):
        app_source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        for field in [
            "sharpenStrength: Float",
            "detailBoost: Float",
            "textureMix: Float",
            "smoothStrength: Float",
            "effectUpscale: Int",
        ]:
            self.assertIn(field, app_source)
        self.assertNotIn("abs(sharpenStrength) < 1e-9", app_source)
        self.assertNotIn("abs(detailBoost) < 1e-9", app_source)
        self.assertNotIn("abs(textureMix) < 1e-9", app_source)
        self.assertNotIn("abs(smoothStrength) < 1e-9", app_source)
        self.assertIn("private struct NativeRestoreEffects", pipeline)
        self.assertIn("applyRestoreEffects(", pipeline)
        self.assertIn("maskedGaussianPlanar(", pipeline)
        self.assertIn("adaptiveLumaContrast(", pipeline)
        self.assertIn("downsamplePlanarArea(", pipeline)
        self.assertIn("processed = maskedMix(", pipeline)

    def test_native_output_pool_has_swap_safe_backpressure(self):
        app = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        self.assertIn(
            "CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(",
            pipeline,
        )
        self.assertIn("kCVPixelBufferPoolAllocationThresholdKey", pipeline)
        self.assertIn("kCVReturnWouldExceedAllocationThreshold", pipeline)
        self.assertIn(
            "CVPixelBufferPoolFlush(outputPool, .excessBuffers)",
            pipeline,
        )
        self.assertIn("config.temporalBatchFrames * 3 + overlap * 2 + 16", pipeline)
        self.assertNotIn(
            "nativeParallelWorkers == 1\n      ? max(64, "
            "config.temporalBatchFrames * 3",
            pipeline,
        )
        self.assertIn("private let miohMaximumClipFrames = 180", app)
        self.assertIn("in: 1...miohMaximumClipFrames", app)
        self.assertIn("private let maximumTemporalBatchFrames = 180", pipeline)
        self.assertIn(
            "config.temporalBatchFrames <= maximumTemporalBatchFrames",
            pipeline,
        )
        self.assertIn("maximumInternalExportSegmentSeconds = 60.0", pipeline)
        self.assertIn(
            "min(maximumInternalExportSegmentSeconds, requestedSegmentSeconds)",
            pipeline,
        )
        self.assertIn('"internal_segment_seconds": writerSegmentSeconds', pipeline)
        self.assertIn("pendingEncoding = nil", pipeline)
        self.assertIn(
            'CIFilter(name: "CIDissolveTransition")',
            pipeline,
        )
        self.assertIn("try autoreleasepool {", pipeline)
        self.assertNotIn("vImagePremultipliedConstAlphaBlend_ARGB8888(", pipeline)

    def test_native_roi_preparation_reuses_sampling_and_composite_plans(self):
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        for contract in [
            "private struct NativeSamplingAxis",
            "private struct NativeCompositePlan",
            "private var maskSamplingCache:",
            "Self.makeModelInputAxes(",
            "Self.writeModelInput(",
            "Self.makeCompositeAxes(geometry: geometry)",
            "plan: compositePlans[index]",
            "let alpha = plan.blendMask[",
        ]:
            self.assertIn(contract, pipeline)
        self.assertNotIn("private static func makeModelInput(\n", pipeline)

    def test_native_pipeline_supports_bounded_roi_enhancement(self):
        source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        self.assertIn("nativeROIEnhancerAsset(", source)
        self.assertIn("roiEnhancerModel: nativeEnhancer?.url.path", source)
        self.assertIn("private final class NativeROIEnhancer", pipeline)
        self.assertIn("try await roiEnhancer.enhanceFrame", pipeline)
        self.assertIn("enhancedFrame: enhancedFrame", pipeline)
        self.assertIn("lowResolution: try Self.makeLowResolution", pipeline)
        self.assertIn("enhancer legacy 256px output", pipeline)
        self.assertIn("base + (highEnhanced - lowEnhanced) * enhancerAmount", pipeline)
        self.assertIn("createEnhancerBlendMask", pipeline)
        self.assertIn("if roiEnhancer == nil, effects.isEnabled", pipeline)
        self.assertIn("only one high-resolution frame", pipeline)
        self.assertNotIn("makeDetailResidual", pipeline)
        self.assertNotIn("private var reducedPool", pipeline)
        self.assertNotIn('label: "enhancer reduced output"', pipeline)
        self.assertIn("CVPixelBufferPoolFlush(sourcePool", pipeline)
        self.assertNotIn('guard roiEnhancer == "none"', source)

    def test_universal_model_tools_keep_all_recovered_fixes(self):
        tools = ROOT / "packaging" / "macOS" / "standalone" / "model-tools"
        download = (tools / "download-mioh-models.zsh").read_text()
        convert = (tools / "convert-mioh-models.zsh").read_text()

        self.assertIn("releases/download/$MIOH_RELEASE_TAG", download)
        self.assertIn("91fe7a48b0e9edf51361918c8a30f752", download)
        self.assertIn("v0.2.5.0/realesr-general-x4v3.pth", download)
        self.assertIn("network_swinir.py", download)
        self.assertIn("--retry-all-errors", download)
        self.assertNotIn("--continue-at", download)
        self.assertIn("The following downloads failed", download)

        self.assertIn("OS_MAJOR=", convert)
        self.assertIn("export_realesrgan_coreml.py", convert)
        self.assertIn("export_srvgg_coreml.py", convert)
        self.assertIn("export_swinir_coreml.py", convert)
        self.assertIn("export_spandrel_coreml.py", convert)
        self.assertIn("expected 11 assets", convert)
        self.assertIn("basicvsrpp-v1.2-variable-coreai.$ARCHITECTURE.aimodelc", convert)

    def test_native_model_resolution_accepts_portable_conversion_outputs(self):
        source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        self.assertIn('for suffix in [".aimodelc", ".aimodel"]', source)
        self.assertIn('for suffix in [".mlmodelc", ".mlpackage"]', source)
        self.assertIn('modelExtension == "mlmodelc" || modelExtension == "mlpackage"', pipeline)
        self.assertIn("MLModel.compileModel(at: modelURL)", pipeline)

    def test_spandrel_realplksr_coreai_is_available_in_mioh(self):
        source = APP_SOURCE.read_text()
        script = BUILD_SCRIPT.read_text()
        verifier = (
            ROOT / "packaging" / "macOS" / "standalone" / "DedicatedModelVerifier.swift"
        ).read_text()

        self.assertIn('let enhancerModels = ["none", "realesrgan", "mewzoom", "swinir", "spandrel"]', source)
        self.assertIn('"nomos-webphoto-realplksr-x4-coreai"', source)
        self.assertIn('"nomos-webphoto-realplksr-x4-coreml"', source)
        self.assertIn("4xNomosWebPhoto_RealPLKSR_256.mlpackage", script)
        self.assertIn("4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel", script)
        self.assertIn('"4xNomosWebPhoto_RealPLKSR-256-fp16"', verifier)

    def test_roi_enhancer_model_picker_is_filtered_by_method(self):
        source = APP_SOURCE.read_text()
        script = BUILD_SCRIPT.read_text()
        verifier = (
            ROOT / "packaging" / "macOS" / "standalone" / "DedicatedModelVerifier.swift"
        ).read_text()

        self.assertIn("var roiEnhancerModelOptions: [ROIEnhancerModelOption]", source)
        self.assertIn('case "realesrgan":', source)
        self.assertIn('"realesrgan-x2-coreai"', source)
        self.assertIn("ForEach(runner.roiEnhancerModelOptions)", source)
        self.assertIn("runner.selectROIEnhancerModel($0)", source)
        self.assertIn("RealESRGAN_x2plus-256-fp16.aimodel", script)
        self.assertIn('"RealESRGAN_x2plus-256-fp16"', verifier)


if __name__ == "__main__":
    unittest.main()
