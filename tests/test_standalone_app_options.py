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
EXPECTED_SEVEN_COREAI_SOURCES = (
    "basicvsrpp-v1.2-t18-fp16.aimodel",
    "basicvsrpp-v1.2-t36-fp16.aimodel",
    "basicvsrpp-v1.2-t90-fp16.aimodel",
    "lada_mosaic_detection_model_v4_fast-fp16.aimodel",
    "RealESRGAN_x4plus-256-fp16.aimodel",
    "realesr-general-x4v3-256-fp16.aimodel",
    "4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel",
)


class StandaloneAppOptionTests(unittest.TestCase):
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
            "driftToleranceSeconds = 0.080",
            'sendCommand(["command": "seek"',
            '"--generation", String(startingGeneration)',
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
            "generationHasStarted ? rebufferSegmentCount : startupSegmentCount",
            "private func startPlayersFromCurrentPosition()",
            "let startingGeneration = generation",
            "guard self.generation == startingGeneration else { return }",
        ]:
            self.assertIn(contract, player)

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
            "func previewArguments(resources: URL, outputDirectory: URL, input: URL)",
            app,
        )
        self.assertIn(
            'var args = ["--input", input.path, "--output-dir", outputDirectory.path]',
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

    def test_runner_exposes_current_settings_to_preview_worker(self):
        app = APP_SOURCE.read_text()

        self.assertIn(
            "func previewArguments(resources: URL, outputDirectory: URL, input: URL)",
            app,
        )
        for option in [
            "--restoration-model", "--detection-model", "--max-clip-length",
            "--restore-max-frames", "--restore-temporal-overlap",
            "--enable-crossfade", "--disable-crossfade",
            "--sharpen-strength", "--detail-boost",
            "--blend-feather", "--texture-mix", "--smooth-strength",
            "--roi-enhancer", "--roi-enhancer-model", "--roi-enhancer-scale",
            "--roi-enhancer-strength", "--roi-enhancer-tile",
            "--effect-upscale", "--buffer-limit",
        ]:
            self.assertIn(option, app)

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
            "private func prepareSourcePlayerItem(input: URL) async throws -> PreparedSourcePlayerItem",
            "try await source.load(.isPlayable)",
            "import Network",
            "private final class HEV1LoopbackServer",
            "private static func findHEV1Offsets(in url: URL, fileSize: UInt64) throws -> [UInt64]",
            'URL(string: "http://127.0.0.1:',
            'header += "Accept-Ranges: bytes\\r\\n"',
            'header += "Content-Range: bytes \\(start)-\\(end)/\\(fileSize)\\r\\n"',
            "private func parseRangeHeader(_ line: String) -> ClosedRange<UInt64>?",
            "content: patch(sourceData, startingAt: cursor)",
            "let resourceLoader = try HEV1LoopbackServer(sourceURL: input)",
            "AVPlayerItem(asset: compatibleAsset)",
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
            '"-tag:v", "hvc1"',
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
            'add(&args, "--buffer-limit", previewBufferLimit)',
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

    def test_preview_has_independent_selectable_restoration_model(self):
        source = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        self.assertIn("@Published var previewRestorationModel: String", source)
        self.assertIn("var previewRestorationModel: String?", source)
        self.assertIn(
            'supportsCoreAI ? "basicvsrpp-v1.2-coreai" : "basicvsrpp-v1.2"',
            source,
        )
        self.assertIn("let previewModel = try resolvedPreviewRestorationModel", source)
        self.assertIn('add(&args, "--restoration-model", previewModel)', source)
        self.assertIn("switch previewModel", source)
        self.assertIn(
            'Picker("復元モデル", selection: $runner.previewRestorationModel)',
            player,
        )
        self.assertIn('if !controller.isVRVideo', player)
        self.assertIn(
            'case "basicvsrpp-v1.2-coreai": automaticClipLength = 98',
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
        self.assertEqual(build_script.count("-target arm64-apple-macosx27.0"), 2)
        self.assertEqual(build_script.count("-framework CoreAI"), 2)

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
        self.assertIn(
            'supportsCoreAI ? baseDetectionModels + ["v4-fast-coreai"] : baseDetectionModels',
            source,
        )
        self.assertIn('"vr-v2-accurate-coreml"', source)
        base_models = source.split("let baseDetectionModels = [", 1)[1].split("]", 1)[0]
        self.assertNotIn('"v2"', base_models)
        self.assertNotIn('"v4-fast"', base_models)
        self.assertNotIn('"vr-v2-accurate"', base_models)
        self.assertIn("normalizeModelSelections()", source)

    def test_coreai_helper_environment_is_only_exported_when_supported(self):
        source = APP_SOURCE.read_text()

        self.assertIn("if capabilities.supportsCoreAI", source)
        self.assertNotIn('result["LADA_COREAI_PYTHON"]', source)
        self.assertIn('result["LADA_COREAI_SWIFT_RUNNER"]', source)
        self.assertIn(
            "try rejectUnsupportedCoreAIModel(roiEnhancerModel)",
            source,
        )

    def test_app_exports_m5_pro_coreai_architecture(self):
        source = APP_SOURCE.read_text()

        self.assertIn('result["LADA_COREAI_ARCHITECTURE"] = "h17s"', source)
        self.assertIn(
            'result.removeValue(forKey: "LADA_COREAI_ARCHITECTURE")', source
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
        for source in EXPECTED_SEVEN_COREAI_SOURCES:
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

    def test_build_prunes_packaging_only_runtime_files(self):
        script = BUILD_SCRIPT.read_text()

        for contract in [
            '"$RESOURCES/runtime/bin/pip"',
            '"$RESOURCES/runtime/lib/python3.12/site-packages/pip"',
            '"$RESOURCES/runtime/lib/python3.12/site-packages/setuptools"',
            '"$RESOURCES/runtime/lib/python3.12/site-packages/tests"',
            '"$RESOURCES/runtime/lib/python3.12/site-packages/yapftests"',
        ]:
            self.assertIn(contract, script)

    def test_mioh_keeps_only_one_mewzoom_coreml_asset(self):
        script = BUILD_SCRIPT.read_text()
        regular_assets = script.split("MODEL_ASSETS=(", 1)[1].split(")", 1)[0]

        self.assertIn("MewZoom-V1-4X-Unet_256.mlpackage", regular_assets)
        self.assertNotIn("MewZoom-V1-4X-Unet_512.mlpackage", regular_assets)

    def test_build_uses_single_standalone_python_environment(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn("LADA_STANDALONE_PYTHON_ENV", script)
        self.assertIn("$ROOT/.venv-coreai", script)
        self.assertNotIn('if [[ -d "$ROOT/.venv" ]]', script)
        self.assertNotIn('$ROOT/.venv/lib/python3.12/site-packages', script)

    def test_universal_build_wrapper_uses_isolated_artifact_paths(self):
        self.assertTrue(UNIVERSAL_BUILD_SCRIPT.is_file())
        script = UNIVERSAL_BUILD_SCRIPT.read_text()

        self.assertIn('ROOT="${PACKAGE_DIR:h:h:h}"', script)
        self.assertIn('COREAI_DISTRIBUTION="portable"', script)
        self.assertIn('build/macos-standalone-universal', script)
        self.assertIn('APP_BASENAME="mioh-universal"', script)
        self.assertIn('DMG_BASENAME="mioh-universal-0.11.0-unsigned"', script)
        self.assertIn('exec "$PACKAGE_DIR/build_app.sh"', script)

    def test_portable_swift_build_omits_architecture_override(self):
        source = APP_SOURCE.read_text()
        script = BUILD_SCRIPT.read_text()

        self.assertIn("#if MIOH_PORTABLE_COREAI", source)
        self.assertIn('result.removeValue(forKey: "LADA_COREAI_ARCHITECTURE")', source)
        self.assertIn("-D MIOH_PORTABLE_COREAI", script)

    def test_parallel_worker_selection_is_forwarded_without_platform_limit(self):
        source = APP_SOURCE.read_text()

        self.assertIn('add(&args, "--parallel-workers", parallelWorkers)', source)
        self.assertNotIn("min(parallelWorkers", source)
        self.assertNotIn("maxParallelWorkers", source)

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
            'DMG_BASENAME="${DMG_BASENAME:-mioh-0.11.0-unsigned}"',
            build_script,
        )
        self.assertIn('DMG="$BUILD_DIR/$DMG_BASENAME.dmg"', build_script)
        self.assertIn('--volumeName "$APP_BASENAME"', build_script)
        self.assertIn('ditto "$APP" "$DMG_ROOT/$APP_BASENAME.app"', build_script)

    def test_gui_exposes_all_processing_options(self):
        source = APP_SOURCE.read_text()
        expected_options = {
            "--input", "--output", "--temp-dir", "--ffmpeg-temp-dir",
            "--lada-temp-dir", "--parallel-workers", "--executor",
            "--segment-duration", "--segment-count", "--merge-encoder",
            "--delete-segments", "--keep-temp", "--force-split", "--device",
            "--fp16", "--no-fp16", "--encoding-preset", "--encoder",
            "--encoder-options", "--bitrate-multiplier", "--quality", "--qmin",
            "--qmax", "--fps", "--pre-fps-conversion", "--mp4-fast-start",
            "--auto-optimize", "--no-auto-optimize", "--mosaic-restoration-model",
            "--max-clip-length", "--restore-max-frames",
            "--restore-temporal-overlap", "--enable-crossfade", "--disable-crossfade",
            "--restore-sharpen-strength", "--restore-detail-boost",
            "--restore-blend-feather", "--restore-texture-mix",
            "--restore-smooth-strength", "--restore-effect-upscale",
            "--restore-roi-enhancer", "--restore-roi-enhancer-model-path",
            "--restore-roi-enhancer-scale", "--restore-roi-enhancer-strength",
            "--restore-roi-enhancer-tile", "--mosaic-detection-model",
            "--mosaic-detection-empty-lookahead", "--detect-face-mosaics",
            "--no-detect-face-mosaics", "--memory-cleanup-interval",
            "--cleanup-trigger-gb", "--mps-memory-fraction", "--log-mps-memory",
            "--overwrite",
        }

        missing = sorted(option for option in expected_options if option not in source)
        self.assertEqual(missing, [])

    def test_gui_exposes_temporal_overlap_controls(self):
        source = APP_SOURCE.read_text()

        for contract in [
            "@Published var restoreTemporalOverlap = 8",
            "@Published var restoreCrossfade = true",
            'LabeledContent("Temporal overlap")',
            'Stepper(value: $runner.restoreTemporalOverlap, in: 0...120)',
            'Toggle("クロスフェードを有効化", isOn: $runner.restoreCrossfade)',
            'add(&args, "--restore-temporal-overlap", restoreTemporalOverlap)',
            'args.append(restoreCrossfade ? "--enable-crossfade" : "--disable-crossfade")',
        ]:
            self.assertIn(contract, source)

    def test_app_bundles_parallel_processor(self):
        script = BUILD_SCRIPT.read_text()
        self.assertIn("process_video_parallel.py", script)

    def test_app_bundles_realtime_player_and_preview_worker(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn('"$PACKAGE_DIR/RealtimePlayer.swift"', script)
        self.assertIn("-framework AVFoundation", script)
        self.assertIn("-framework AVKit", script)
        self.assertIn("-framework Network", script)
        self.assertIn(
            '"$RESOURCES/runtime/lib/python3.12/site-packages/mioh_preview_worker.py"',
            script,
        )

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

        self.assertIn('result["LADA_APP_PROGRESS"] = "1"', source)
        self.assertIn("struct AppProgressEvent: Decodable", source)
        self.assertIn('"@@LADA_PROGRESS@@"', source)
        self.assertIn("activeProgress", source)
        self.assertIn("logHistory", source)
        self.assertIn("rebuildVisibleLog()", source)

    def test_gui_has_always_visible_multiline_ffmpeg_options(self):
        source = APP_SOURCE.read_text()

        self.assertIn('Section("FFmpeg詳細設定")', source)
        self.assertIn('Text("追加FFmpegオプション")', source)
        self.assertIn('TextEditor(text: $runner.encoderOptions)', source)
        self.assertEqual(
            source.count(
                'addOptional(&args, "--encoder-options", encoderOptions)'
            ),
            1,
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


if __name__ == "__main__":
    unittest.main()
