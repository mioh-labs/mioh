import unittest
import plistlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "MiohApp.swift"
PLAYER_SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "RealtimePlayer.swift"
BUILD_SCRIPT = ROOT / "packaging" / "macOS" / "standalone" / "build_app.sh"
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
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()
        encoder = PREVIEW_ENCODER_SOURCE.read_text()

        for contract in [
            'let mode = "export"',
            "makeNativeExportTask(",
            "resolvedOutputFile(",
            'appendingPathComponent("\\(stem)-UC")',
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
        for contract in [
            "var isExport: Bool",
            "struct DetectedBatch",
            "temporalOverlap",
            "prepareInput(",
            "finishExport(",
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
            "effectiveSegmentSeconds",
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
            ".disabled(!runner.useFPS || (runner.usesPythonEngine && runner.noSplit))",
            app,
        )

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

    def test_player_starts_each_generation_once_and_resumes_without_seeking(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "private var generationHasStarted = false",
            "private var generationStartPending = false",
            "guard state != .playing, !generationStartPending else { return }",
            "Double(generationHasStarted ? rebufferSegmentCount : startupSegmentCount)",
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
            "Double(generationHasStarted ? rebufferSegmentCount : startupSegmentCount)",
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
            "stop(preserveSourceItem: canReuseCurrentSource)",
            "preserveCurrentSource: true",
            "sourcePlayer.seek(",
            "self.position = startSeconds",
            "showsSourceFrameWhilePreparingRestoration",
            "position = requestedStartSeconds",
        ]:
            self.assertIn(contract, player)
        self.assertIn("if !preserveSourceItem", player)

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
            "let previewModel = capabilities.previewRestorationModel",
            "let previewDetectionModel = capabilities.previewDetectionModel",
            "restorationModels: restoration.url.path",
            "detectionModel: detection.url.path",
            "bufferLimitSeconds: previewBufferLimit",
            "temporalBatchFrames: temporalFrames",
            "blendFeather: Float(blendFeather)",
            "detectionEmptyLookahead: max(0, detectionEmptyLookahead)",
            "detectFaceMosaics: detectFaceMosaics",
            'environment["TMPDIR"] = miohTemporary',
        ]:
            self.assertIn(contract, app)

    def test_app_has_user_default_settings_panel(self):
        source = APP_SOURCE.read_text()

        for contract in [
            "struct MiohUserDefaultsSnapshot: Codable",
            'private let defaultsKey = "mioh.userProcessingDefaults.v1"',
            "func saveCurrentDefaults()",
            "func loadSavedDefaults()",
            "func resetDefaultsToFactory()",
            "private func loadSavedDefaultsOnLaunch()",
            "private func currentDefaultsSnapshot() -> MiohUserDefaultsSnapshot",
            "private func apply(defaults snapshot: MiohUserDefaultsSnapshot)",
            "UserDefaults.standard.set(data, forKey: defaultsKey)",
            "UserDefaults.standard.data(forKey: defaultsKey)",
            "UserDefaults.standard.removeObject(forKey: defaultsKey)",
            'settingsTab.tabItem { Label("設定", systemImage: "gearshape") }',
            'Section("ユーザーデフォルト")',
            'Label("現在の設定をデフォルトに保存", systemImage: "square.and.arrow.down")',
            'Label("保存済みデフォルトを読み込み", systemImage: "arrow.clockwise")',
            'Label("初期値に戻す", systemImage: "trash")',
            "入力/出力、一時フォルダ、分割、復元、検出、出力、メモリ、再生バッファまで保存します",
        ]:
            self.assertIn(contract, source)

        self.assertIn("var previewBufferLimit: Double", source)
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
            "AVPlayerItemVideoOutput(pixelBufferAttributes: attributes)",
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

    def test_preview_uses_fixed_models_without_model_pickers(self):
        source = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        self.assertIn("var previewRestorationModel: String?", source)
        self.assertIn("@Published var previewRealtimeOptimization = true", source)
        self.assertIn(
            'supportsCoreAI ? "basicvsrpp-v1.2-coreai-variable" : "basicvsrpp-v1.2"',
            source,
        )
        self.assertIn(
            "let previewModel = capabilities.previewRestorationModel",
            source,
        )
        self.assertIn(
            "let previewDetectionModel = capabilities.previewDetectionModel",
            source,
        )
        self.assertIn("model: previewModel", source)
        self.assertIn("model: previewDetectionModel", source)
        # The pickers exist again for the bundled Python worker, but the
        # native preview still resolves its assets from capabilities alone.
        self.assertIn("@Published var previewRestorationModel: String", source)
        self.assertIn("@Published var previewDetectionModel: String", source)
        self.assertIn(
            'if runner.usesPythonEngine {\n            HStack(spacing: 12) {\n'
            '              Picker("復元モデル"',
            player,
        )
        self.assertIn('Picker("再生用検出モデル"', player)
        self.assertIn(
            'Toggle("リアルタイム最適化", isOn: $runner.previewRealtimeOptimization)',
            player,
        )
        self.assertIn('if !controller.isVRVideo', player)
        self.assertIn(
            "let temporalLimit = detection.computeUnits == nil ? 18 : 36",
            source,
        )
        self.assertIn(
            'supportsCoreAI ? "basicvsrpp-v1.2-coreai-t90" : "basicvsrpp-v1.2"',
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
            'supportsCoreAI ? "basicvsrpp-v1.2-coreai-t90" : "basicvsrpp-v1.2"',
            source,
        )
        self.assertIn(
            'supportsCoreAI ? coreAIRestorationModels + baseRestorationModels : baseRestorationModels',
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

    def test_rfdetr_is_dedicated_only_and_does_not_change_shared_memory_policy(self):
        source = APP_SOURCE.read_text()
        build_script = BUILD_SCRIPT.read_text()
        restorer = (
            ROOT / "lada" / "restorationpipeline" / "frame_restorer.py"
        ).read_text()
        detector = (
            ROOT / "lada" / "restorationpipeline" / "mosaic_detector.py"
        ).read_text()

        self.assertIn(
            '#if !MIOH_PORTABLE_COREAI\n    models.append("jasna-v6-coreai")',
            source,
        )
        self.assertIn(
            'if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]',
            build_script,
        )
        self.assertIn("rfdetr-v6-576-fp32.aimodel", build_script)
        self.assertIn("rfdetr-v6-large-768-fp32.aimodel", build_script)
        self.assertIn("-iname '*rfdetr*' -delete", build_script)
        # The bundled runtime inherits the build venv, so RF-DETR has to be
        # stripped back out of it rather than merely never copied.
        self.assertIn(
            '"$RESOURCES/runtime/lib/python3.12/site-packages/rfdetr" \\',
            build_script,
        )
        self.assertNotIn('site-packages/lada/models/rfdetr"', build_script)
        self.assertNotIn("calculate_frame_detection_queue_size", restorer)
        self.assertNotIn("pipeline_queue_depth", detector)
        self.assertIn("maxsize=8", detector)

    def test_coreai_helper_environment_is_only_exported_when_supported(self):
        source = APP_SOURCE.read_text()

        self.assertIn("guard capabilities.supportsCoreAI else", source)
        # The Python engine environment only advertises the Core AI helpers on
        # a machine that can actually load them, and strips them otherwise.
        self.assertIn("if capabilities.supportsCoreAI {", source)
        self.assertIn(
            'result.removeValue(forKey: "LADA_COREAI_PYTHON")',
            source,
        )
        self.assertIn(
            'result.removeValue(forKey: "LADA_COREAI_SWIFT_RUNNER")',
            source,
        )
        self.assertIn('"bin/mioh-native-coreai-preview"', source)
        self.assertIn("try rejectUnsupportedCoreAIModel(previewModel)", source)

    def test_app_exports_m5_pro_coreai_architecture(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn('COREAI_ARCHITECTURE="${COREAI_ARCHITECTURE:-h17s}"', script)
        # The app only names the architecture inside the Python engine's
        # environment, and the portable build strips it there.
        source = APP_SOURCE.read_text()
        self.assertIn(
            '#if MIOH_PORTABLE_COREAI\n      result.removeValue('
            'forKey: "LADA_COREAI_ARCHITECTURE")\n#else\n'
            '      result["LADA_COREAI_ARCHITECTURE"] = "h17s"',
            source,
        )
        self.assertNotIn(
            'nativeEnvironment["LADA_COREAI_ARCHITECTURE"]', source
        )

    def test_build_targets_only_m5_pro_coreai_specialization(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn(
            'COREAI_ARCHITECTURE="${COREAI_ARCHITECTURE:-h17s}"', script
        )
        self.assertIn('--architecture "$COREAI_ARCHITECTURE"', script)
        self.assertIn('! -name "*.$COREAI_ARCHITECTURE.aimodelc"', script)
        self.assertNotIn('for model in "$COMPILED_MODELS"/*.aimodelc', script)
        self.assertNotIn("basicvsrpp-v1.2-t36-b2-fp16.aimodel", script)
        regular_assets = script.split("MODEL_ASSETS=(", 1)[1].split(")", 1)[0]
        for source in EXPECTED_COREAI_SOURCES:
            self.assertIn(source, script)
            self.assertNotIn(source, regular_assets)

    def test_build_runs_all_coreai_smoke_tests_before_signing(self):
        script = BUILD_SCRIPT.read_text()

        verifier = script.index('"$PACKAGE_DIR/verify_coreai_models.py"')
        signing = script.index('codesign --force --deep --sign - "$APP"')
        self.assertLess(verifier, signing)
        self.assertIn('LADA_COREAI_ARCHITECTURE="$COREAI_ARCHITECTURE"', script)
        self.assertIn('LADA_MODEL_WEIGHTS_DIR="$RESOURCES/models"', script)
        self.assertIn('LADA_COREAI_SWIFT_RUNNER="$RESOURCES/bin/lada-coreai-runner"', script)

    def test_build_supports_dedicated_and_portable_coreai_distributions(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn('COREAI_DISTRIBUTION="${COREAI_DISTRIBUTION:-dedicated}"', script)
        self.assertIn('dedicated|portable)', script)
        self.assertIn('if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]', script)
        self.assertIn('ditto "$source_model" "$RESOURCES/models/$asset"', script)
        self.assertIn('--distribution "$COREAI_DISTRIBUTION"', script)
        self.assertIn('--smoke-model basicvsrpp-v1.2-coreai', script)

    def test_only_the_portable_build_bundles_a_python_runtime(self):
        script = BUILD_SCRIPT.read_text()

        # The dedicated package is Core AI only; the portable/universal one
        # still ships the interpreter that drives the Python fallback.
        self.assertIn(
            'if [[ "$COREAI_DISTRIBUTION" == "portable" ]]; then\n'
            '  MIOH_BUNDLE_PYTHON_RUNTIME="${MIOH_BUNDLE_PYTHON_RUNTIME:-1}"\n'
            "else\n"
            '  MIOH_BUNDLE_PYTHON_RUNTIME="${MIOH_BUNDLE_PYTHON_RUNTIME:-0}"',
            script,
        )
        for guarded in [
            'ditto "$PYTHON_SOURCE" "$RESOURCES/runtime"',
            'cp "$ROOT/process_video_parallel.py" \\',
            'cp "$PACKAGE_DIR/mioh_preview_worker.py" \\',
            'chmod +x "$RESOURCES/runtime/bin/python3.12"',
        ]:
            self.assertIn(guarded, script)
        # Every runtime block must sit behind the bundling flag.
        for block in script.split('if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 ]]; then')[
            :1
        ]:
            self.assertNotIn('ditto "$PYTHON_SOURCE"', block)

    def test_build_time_python_is_required_for_models_or_bundling(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn(
            'if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 '
            '|| "$MIOH_MODELESS_DISTRIBUTION" != 1 ]]',
            script,
        )
        self.assertIn("Missing build-time Python:", script)
        self.assertIn("Missing interpreter to bundle:", script)

    def test_mioh_keeps_only_one_mewzoom_coreml_asset(self):
        script = BUILD_SCRIPT.read_text()
        regular_assets = script.split("MODEL_ASSETS=(", 1)[1].split(")", 1)[0]

        self.assertIn("MewZoom-V1-4X-Unet_256.mlpackage", regular_assets)
        self.assertNotIn("MewZoom-V1-4X-Unet_512.mlpackage", regular_assets)

    def test_build_uses_python_only_for_model_generation(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn("LADA_STANDALONE_PYTHON_ENV", script)
        self.assertIn("$ROOT/.venv-coreai", script)
        self.assertIn('if [[ "$MIOH_MODELESS_DISTRIBUTION" != 1 ]]', script)
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
        self.assertIn(
            '#if MIOH_PORTABLE_COREAI\n      result.removeValue('
            'forKey: "LADA_COREAI_ARCHITECTURE")',
            source,
        )
        self.assertIn("-D MIOH_PORTABLE_COREAI", script)

    def test_native_export_supports_ntsc_fractional_frame_rates(self):
        source = APP_SOURCE.read_text()
        pipeline = NATIVE_PIPELINE_SOURCE.read_text()

        # The rate travels as a rational so 29.970 stays 30000/1001.
        self.assertIn("@Published var ntscFPS = false", source)
        self.assertIn(
            "var targetFPSNumerator: Int { usesNTSCFPS ? fps * 1000 : fps }",
            source,
        )
        self.assertIn(
            "var targetFPSDenominator: Int { usesNTSCFPS ? 1001 : 1 }", source
        )
        self.assertIn("targetFPS: useFPS ? max(1, targetFPSNumerator) : nil", source)
        self.assertIn(
            "targetFPSDenominator: useFPS ? targetFPSDenominator : nil", source
        )
        # Pull-down is native-only; the Python CLI takes an integer rate.
        self.assertIn(
            "var usesNTSCFPS: Bool { ntscFPS && !usesPythonEngine }", source
        )
        # The stepper shows the effective rate, not the integer it derives from.
        self.assertIn("Text(runner.targetFPSDescription)", source)
        self.assertIn('String(format: "%.3f", targetFPSValue)', source)
        self.assertIn("Toggle(\"NTSC（1000/1001）\", isOn: $runner.ntscFPS)", source)

        self.assertIn("let targetFPSDenominator: Int?", pipeline)
        self.assertIn("init(numerator: Int, denominator: Int) {", pipeline)
        self.assertIn("intervalRemainder = step % frames", pipeline)
        self.assertIn(
            "let outputFPSNumerator = targetRate?.numerator ?? video.fpsNumerator",
            pipeline,
        )
        self.assertIn(
            "let outputFPSDenominator = targetRate?.denominator "
            "?? video.fpsDenominator",
            pipeline,
        )

    def test_native_export_uses_internal_stage_concurrency(self):
        source = APP_SOURCE.read_text()

        self.assertNotIn("guard parallelWorkers == 1", source)
        self.assertIn('parallelWorkers = 1', source)
        self.assertIn('executor = "process"', source)
        self.assertIn('mergeEncoder = "copy"', source)
        self.assertIn("Swiftネイティブ（自動段階並列）", source)
        self.assertIn("デコード・検出・復元・エンコードを1プロセス内で並行実行します", source)

    def test_native_engine_normalizes_python_only_options(self):
        source = APP_SOURCE.read_text()

        # The device/precision controls exist again, but only the bundled
        # Python engine may show them; the native engine pins its contract.
        self.assertIn("if runner.usesPythonEngine {\n          Picker(\"デバイス\"", source)
        self.assertIn('Toggle("FP16", isOn: $runner.fp16)', source)
        self.assertIn('Toggle("自動最適化", isOn: $runner.autoOptimize)', source)
        self.assertIn('device = "mps"', source)
        self.assertIn("fp16 = true", source)
        self.assertIn("autoOptimize = true", source)
        self.assertIn("Swiftネイティブ / Core AI", source)
        self.assertIn(
            "var usesPythonEngine: Bool { supportsPythonEngine "
            '&& restorationEngine == "python" }',
            source,
        )

    def test_python_engine_is_portable_only(self):
        source = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        self.assertIn("var bundlesPythonRuntime: Bool {\n#if MIOH_PORTABLE_COREAI", source)
        self.assertIn(
            "var supportsPythonEngine: Bool { capabilities.bundlesPythonRuntime }",
            source,
        )
        # Both the export and the preview honour the same switch.
        self.assertIn("if usesPythonEngine {\n        let pythonTask =", source)
        self.assertIn("if runner.usesPythonEngine {", player)
        self.assertIn(
            '"runtime/lib/python3.12/site-packages/mioh_preview_worker.py"',
            player,
        )
        self.assertIn(
            '"runtime/lib/python3.12/site-packages/process_video_parallel.py"',
            source,
        )

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
            "ladaTempDirectory", "parallelWorkers", "executor",
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
            "detectionModel", "detectionEmptyLookahead", "detectFaceMosaics",
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

        for contract in [
            "@Published var restoreTemporalOverlap = 8",
            "@Published var restoreCrossfade = true",
            'LabeledContent("Temporal overlap")',
            'Stepper(value: $runner.restoreTemporalOverlap, in: 0...120)',
            'Toggle("クロスフェードを有効化", isOn: $runner.restoreCrossfade)',
            "temporalOverlap: overlap",
            "crossfade: restoreCrossfade",
        ]:
            self.assertIn(contract, source)

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
            'doubleSliderField("シャープ", value: $runner.sharpenStrength, range: 0...2, step: 0.05)',
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

    def test_native_pipeline_keeps_roi_enhancer_disabled(self):
        source = APP_SOURCE.read_text()

        self.assertIn('guard roiEnhancer == "none"', source)
        self.assertIn("abs(roiEnhancerStrength) < 1e-9", source)
        self.assertIn("ROIエンハンサー: 無効", source)

    def test_spandrel_realplksr_coreai_is_available_in_mioh(self):
        source = APP_SOURCE.read_text()
        script = BUILD_SCRIPT.read_text()
        verifier = (
            ROOT / "packaging" / "macOS" / "standalone" / "verify_coreai_models.py"
        ).read_text()

        self.assertIn('let enhancerModels = ["none", "realesrgan", "mewzoom", "swinir", "spandrel"]', source)
        self.assertIn('"nomos-webphoto-realplksr-x4-coreai"', source)
        self.assertIn('"nomos-webphoto-realplksr-x4-coreml"', source)
        self.assertIn("4xNomosWebPhoto_RealPLKSR_256.mlpackage", script)
        self.assertIn("4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel", script)
        self.assertIn('"nomos-webphoto-realplksr-x4-coreai"', verifier)
        self.assertIn('"4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel"', verifier)

    def test_roi_enhancer_model_picker_is_filtered_by_method(self):
        source = APP_SOURCE.read_text()
        script = BUILD_SCRIPT.read_text()
        verifier = (
            ROOT / "packaging" / "macOS" / "standalone" / "verify_coreai_models.py"
        ).read_text()

        self.assertIn("var roiEnhancerModelOptions: [ROIEnhancerModelOption]", source)
        self.assertIn('case "realesrgan":', source)
        self.assertIn('"realesrgan-x2-coreai"', source)
        self.assertIn("ForEach(runner.roiEnhancerModelOptions)", source)
        self.assertIn("runner.selectROIEnhancerModel($0)", source)
        self.assertIn("RealESRGAN_x2plus-256-fp16.aimodel", script)
        self.assertIn('"realesrgan-x2-coreai"', verifier)


if __name__ == "__main__":
    unittest.main()
