import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "packaging" / "macOS" / "standalone"
UPSCALER = ROOT / "packaging" / "macOS" / "upscaler"
MIOH_APP = STANDALONE / "MiohApp.swift"
MIOH_BUILD = STANDALONE / "build_app.sh"
UPSCALER_APP = UPSCALER / "UpscalerApp.swift"
UPSCALER_BUILD = UPSCALER / "build_app.sh"
UPSCALER_INFO = UPSCALER / "Info.plist"
UPSCALER_ICON = UPSCALER / "AppIcon-1024.png"
VIDEO_PREVIEW = UPSCALER / "UpscalerVideoPreview.swift"
MODEL_SETUP = UPSCALER / "UpscalerModelSetup.swift"
MODEL_SETUP_SCRIPT = UPSCALER / "model-tools" / "setup-upscaler-models.zsh"
CONTROLLER = UPSCALER / "VideoUpscaleController.swift"
ADCSR_PIPELINE = UPSCALER / "AdcSRNativePipeline.swift"
ADCSR_RUNNER = UPSCALER / "AdcSRNativeVideoRunner.swift"
H3_VIEW = UPSCALER / "MiniMaxH3VideoGenerationView.swift"
H3_FACE_REFERENCES = UPSCALER / "MiniMaxH3FaceReferences.swift"
H3_RUNNER = UPSCALER / "MiniMaxH3NativeRunner.swift"
H3_MEDIA = UPSCALER / "MiniMaxH3NativeMedia.swift"
H3_MODELS = UPSCALER / "MiniMaxH3NativeModels.swift"
H3_DENOISER = UPSCALER / "TenErosMaxH3DenoiserComposite.swift"
VENDORED_FLASHVSR_RUNNER = (
    UPSCALER / "vendor" / "flashvsr" / "deployment" / "coreai"
    / "FlashVSRNativeVideoRunner.swift"
)
FLASHVSR_RUNNER = (
    VENDORED_FLASHVSR_RUNNER
    if VENDORED_FLASHVSR_RUNNER.exists()
    else ROOT.parent / "FlashVSR_plus" / "deployment" / "coreai"
    / "FlashVSRNativeVideoRunner.swift"
)


class MiohUpscalerSeparationTests(unittest.TestCase):
    def test_mioh_no_longer_exposes_or_bundles_upscaling(self):
        app = MIOH_APP.read_text()
        build = MIOH_BUILD.read_text()
        for removed in (
            "case upscale",
            "WorkspaceTab.upscale",
            "@StateObject private var upscaler",
            'Label("アップスケール"',
            "private var upscaleTab",
            "VideoUpscaleController.swift",
            "flashvsr-coreai-video",
            "adcsr-coreai-video",
            "MIOH_BUNDLE_FLASHVSR",
            "MIOH_BUNDLE_ADCSR",
            "qualityGeneration",
            "MiniMaxH3",
            "mioh-minimax-h3-native",
        ):
            self.assertNotIn(removed, app + "\n" + build)

    def test_independent_app_has_complete_upscaler_workflow(self):
        source = UPSCALER_APP.read_text()
        for contract in (
            "struct MiohUpscalerApp: App",
            "@StateObject private var upscaler = VideoUpscaleController()",
            "@StateObject private var h3Generation = MiniMaxH3Controller()",
            "@StateObject private var modelSetup = UpscalerModelSetupController()",
            'Label("動画生成", systemImage: "sparkles.rectangle.stack")',
            'Section("アップスケール範囲")',
            'Text("FlashVSR Tiny（動画・時間整合）").tag("flashvsr")',
            'Text("AdcSR（軽量な1-step拡散）").tag("adcsr")',
            'Text("2倍").tag(2)',
            'Text("4倍").tag(4)',
            "AdcSRは内部では常に4倍推論",
            "let safeMaximum = max(minimum + 0.01, requestedMaximum)",
            "in: minimum...safeMaximum",
            'Toggle("ログ", isOn: $showLog)',
            "upscaler.estimatedRemainingText",
            "UpscalerVideoPreview(",
            '"範囲時間"',
            "upscaler.setSelectedDurationSeconds($0)",
            'Text("数値入力")',
            '"秒数", value: safeValue',
            ".textFieldStyle(.roundedBorder)",
            'accessibilityLabel("\\(title)を秒で直接入力")',
            'Label("アップスケール開始"',
            'Button("モデルを自動設定…", action: presentModelSetup)',
            ".sheet(isPresented: $showingModelSetup)",
        ):
            self.assertIn(contract, source)

    def test_independent_build_owns_runners_but_keeps_models_external(self):
        source = UPSCALER_BUILD.read_text()
        for contract in (
            'APP="$BUILD_DIR/mioh upscaler.app"',
            "UpscalerMediaProbe.swift",
            "VideoUpscaleController.swift",
            "UpscalerVideoPreview.swift",
            "UpscalerModelSetup.swift",
            "UpscalerApp.swift",
            "MiniMaxH3VideoGenerationView.swift",
            "MiniMaxH3FaceReferences.swift",
            "MiniMaxH3NativeCore.swift",
            "MiniMaxH3NativeRunner.swift",
            '"$RESOURCES/bin/mioh-minimax-h3-native"',
            "-framework AVKit",
            "-framework Vision",
            "FlashVSRNativePipeline.swift",
            "FlashVSRNativeVideoRunner.swift",
            '"$RESOURCES/bin/flashvsr-coreai-video"',
            "AdcSRNativePipeline.swift",
            "AdcSRNativeVideoRunner.swift",
            '"$RESOURCES/bin/adcsr-coreai-video"',
            'SOURCE_ICON="$UPSCALER_DIR/AppIcon-1024.png"',
            'ln -s /Applications "$DMG_ROOT/Applications"',
            'iconutil -c icns "$ICONSET" -o "$RESOURCES/AppIcon.icns"',
            "codesign --force --deep",
            "diskutil image create from",
        ):
            self.assertIn(contract, source)
        for bundled_model_contract in (
            '"$RESOURCES/models"',
            "MIOH_BUNDLE_FLASHVSR",
            "MIOH_BUNDLE_ADCSR",
            "FLASHVSR_COREAI_MODELS_DIR",
            "ADCSR_COREAI_MODEL",
            "Bundled FlashVSR",
            "Bundled AdcSR",
        ):
            self.assertNotIn(bundled_model_contract, source)

        controller = CONTROLLER.read_text()
        self.assertIn(
            'appendingPathComponent("Documents/lada/model_weights"',
            controller,
        )
        self.assertIn(
            'root.appendingPathComponent(Self.nativeDirectoryName',
            controller,
        )
        self.assertNotIn('resources.appendingPathComponent("models/', controller)

    def test_native_upscaler_runners_use_macos_27_streaming_io(self):
        for path in (ADCSR_RUNNER, FLASHVSR_RUNNER, H3_MEDIA):
            source = path.read_text()
            self.assertIn("outputProvider(for:", source, path)
            self.assertIn("try reader.start()", source, path)
            self.assertNotIn("copyNextSampleBuffer()", source, path)
            self.assertNotIn("startReading()", source, path)
            self.assertNotIn("alwaysCopiesSampleData", source, path)

        for path in (ADCSR_RUNNER, FLASHVSR_RUNNER):
            source = path.read_text()
            self.assertIn("inputPixelBufferReceiver(", source, path)
            self.assertIn("try await receiver.append(", source, path)
            self.assertNotIn("AVAssetWriterInputPixelBufferAdaptor", source, path)
            self.assertNotIn("isReadyForMoreMediaData", source, path)

    def test_adcsr_tiles_use_low_frequency_anchored_cosine_blending(self):
        runner = ADCSR_RUNNER.read_text()
        controller = CONTROLLER.read_text()
        view = UPSCALER_APP.read_text()

        for contract in (
            "private let adcSRTileOverlap = 16",
            "private let adcSRLowFrequencyAnchorStrength: Float = 1",
            "intervalCount = max(1, Int(ceil(",
            "blendLeft:",
            "blendRight:",
            "blendTop:",
            "blendBottom:",
            "kernel void downsample_adcsr",
            "inline float raised_cosine",
            "lowFrequencyAnchorStrength * (sourceLow - modelLow)",
            "try metal.add(output: output, input: input, tile: tile, canvas: canvas)",
        ):
            self.assertIn(contract, runner)
        self.assertIn(
            "let overlap = selectedUpscaler == .adcSR ? 16",
            controller,
        )
        self.assertIn("低周波を入力へ固定したcosine blend", view)

    def test_upscaler_preserves_original_audio_without_intermediate_reencode(self):
        controller = CONTROLLER.read_text()
        trim = controller.split("let task = Process()", 1)[1].split(
            "try launch(task, phase: .trim)", 1
        )[0]
        self.assertIn('"-ss", Self.number(start)', trim)
        self.assertLess(
            trim.index('"-ss", Self.number(start)'),
            trim.index('"-i", inputURL.path'),
        )
        self.assertIn('"-an"', trim)
        self.assertNotIn('"asetpts=PTS-STARTPTS"', trim)

        final_mux = controller.split("private func startFinalMux(", 1)[1]
        final_mux = final_mux.split("private func finishOutput(", 1)[0]
        for contract in (
            '"-ss", Self.number(runStartSeconds)',
            '"-t", Self.number(runDurationSeconds)',
            '"-i", runInputURL.path',
            'arguments += ["-c:a", "copy"]',
            'arguments += ["-c:a", "aac", "-b:a", "192k"]',
            'case muxAudioFallback',
        ):
            self.assertIn(contract, controller)
        self.assertIn('"-movflags", "+faststart"', final_mux)

    def test_flashvsr_tiles_are_evenly_spaced_and_cosine_blended(self):
        runner = FLASHVSR_RUNNER.read_text()
        for contract in (
            "intervalCount = max(1, Int(ceil(",
            "inline float raised_cosine",
            "raised_cosine(float(gid.x) + 0.5f, overlapLeft)",
            "raised_cosine(float(gid.y) + 0.5f, overlapTop)",
        ):
            self.assertIn(contract, runner)

    def test_video_generation_owns_minimax_workflow_and_external_models(self):
        view = H3_VIEW.read_text()
        face_references = H3_FACE_REFERENCES.read_text()
        runner = H3_RUNNER.read_text()
        media = H3_MEDIA.read_text()
        build = UPSCALER_BUILD.read_text()

        for contract in (
            'Section("動画生成（MiniMax H3）")',
            "panel.allowedContentTypes = [.movie, .image]",
            "panel.allowsMultipleSelection = true",
            '"--input-images-json"',
            "参照素材は縦横比を保って中央に収め",
            'Text("プロンプトのみ").tag(true)',
            '"manifest-fl2va.json"',
            'return mode == "fl2va"',
            '"bin/mioh-minimax-h3-native"',
            "com.okatti.mioh.upscaler.10erosMaxH3ManifestPath",
            "外部のMiniMax H3 manifest.json",
            "resolvePipelineManifestPath",
            '.appendingPathComponent("manifest.json")',
            "selectImageReferenceScope",
            "groupSelectedFacesAsOneSubject",
            "faceReferencePrompt",
            "selectedFaceReferences.map(\\.cropURL)",
        ):
            self.assertIn(contract, view)
        for contract in (
            "VNDetectFaceRectanglesRequest",
            "expandedFaceCrop",
            "face.width * 1.75",
            "face.height * 1.95",
            "UTType.png.identifier",
            "maximumReferences = 8",
            'case .faceOnly: "顔のみ"',
        ):
            self.assertIn(contract, face_references)
        self.assertIn('options["input-images-json"]', runner)
        self.assertIn("qwen-presentation-v7-continuous-edge-extend", runner)
        self.assertIn("width: H3Geometry.qwenVisionWidth", runner)
        self.assertIn("height: H3Geometry.qwenVisionHeight", runner)
        self.assertIn("case .fl2va:", runner)
        self.assertIn("referenceVideo: nil", runner)
        self.assertIn("prepareTextToVideo", H3_DENOISER.read_text())
        self.assertIn("prepareImages", H3_DENOISER.read_text())
        self.assertIn("decodeReferenceImages", media)
        self.assertIn("decodeReferenceImage", media)
        self.assertIn("decodeIdentityReferenceImages", media)
        self.assertIn("referenceImageLatents", runner)
        self.assertIn("firstVideoLatentFrame", runner)
        self.assertNotIn("H3NativeMedia.silentAudio", runner)
        self.assertEqual(
            runner.count("sampled.audio = sampled.audio.map { $0 / scale }"),
            1,
        )
        self.assertNotIn("let unscaled = values.map { $0 / audioScale }", runner)
        self.assertIn('resolutionProfileID = "864x480"', view)
        self.assertIn('Text("24fps固定")', view)
        self.assertIn("1024, height: 576", view)
        self.assertIn("768, height: 768", view)
        self.assertIn("latentPatchCells <= 576", runner)
        self.assertIn("variable-resolution manifest", runner)
        self.assertNotIn("10秒へ等間隔配置", view)
        self.assertNotIn("H3_NATIVE_ASSETS", build)
        self.assertNotIn("H3_BUNDLE_ASSETS", build)
        self.assertNotIn('models/10eros-max-h3/manifest.json', view)

    def test_minimax_h3_preserves_reference_aspect_ratio(self):
        media = H3_MEDIA.read_text()
        runner = H3_RUNNER.read_text()
        self.assertIn("let uniformScale = min(", media)
        self.assertIn("scaleX: uniformScale, y: uniformScale", media)
        self.assertIn("foreground.clampedToExtent().cropped", media)
        self.assertIn("foreground.composited(over: background)", media)
        self.assertIn('"native-image-reference-v4-continuous-edge-extend"', runner)
        self.assertIn('"media-v7-continuous-edge-extend:', runner)
        self.assertIn('"qwen-presentation-v7-continuous-edge-extend"', runner)
        self.assertIn("Data(Self.referenceMediaPreprocessingVersion.utf8)", runner)
        self.assertNotIn(
            "scaleX: CGFloat(width) / extent.width,\n"
            "        y: CGFloat(height) / extent.height",
            media,
        )

    def test_minimax_h3_decoder_matches_tile_exposure_and_cosine_blends(self):
        decoder = (UPSCALER / "MiniMaxH3NativeVideoVAE.swift").read_text()
        blender = (UPSCALER / "MiniMaxH3SpatialTileBlender.swift").read_text()
        runner = H3_RUNNER.read_text()
        build = UPSCALER_BUILD.read_text()

        self.assertIn("H3SpatialTileBlender", decoder)
        self.assertNotIn("blendSpatialTail", decoder)
        self.assertIn("private static let spatialDecodeOverlap = 128", decoder)
        self.assertIn("minimumOverlap: spatialDecodeOverlap", decoder)
        for contract in (
            "estimateCorrection(",
            "maximumGainDelta",
            "maximumOffset",
            "sin(.pi * 0.5",
            "cos(.pi * 0.5",
            "sums[destinationIndex] /= weight",
        ):
            self.assertIn(contract, blender)
        self.assertIn(
            'Data("spatial-half-tile-affine-cosine-blend-v3".utf8)', runner
        )
        self.assertIn('"$UPSCALER_DIR/MiniMaxH3SpatialTileBlender.swift"', build)

    def test_first_launch_model_setup_downloads_converts_and_configures(self):
        setup = MODEL_SETUP.read_text()
        script = MODEL_SETUP_SCRIPT.read_text()
        controller = CONTROLLER.read_text()
        build = UPSCALER_BUILD.read_text()

        for contract in (
            '"初回モデル自動設定"',
            'panel.title = "モデルの配置フォルダを選択"',
            "panel.canCreateDirectories = true",
            '"model-tools/setup-upscaler-models.zsh"',
            'arguments = ["--destination", destination.path]',
            'Text("FlashVSR-v1.1")',
            'Text("AdcSR ×4")',
        ):
            self.assertIn(contract, setup)
        for contract in (
            "JunhaoZhuang/FlashVSR-v1.1/resolve/main",
            "mlboydaisuke/AdcSR-CoreAI/resolve/main",
            "coreai-torch==0.4.2",
            "deployment.coreai.export_native",
            'DESTINATION/.mioh-upscaler-setup',
            "--dry-run",
            "FlashVSR setup needs at least 18 GiB free",
        ):
            self.assertIn(contract, script)
        self.assertIn("func stop()", setup)
        self.assertIn("func applyModelSetupDestination(_ path: String)", controller)
        self.assertIn('MODEL_TOOLS="$RESOURCES/model-tools"', build)
        self.assertIn("mioh-upscaler-0.14.3-unsigned.dmg", build)

        info = UPSCALER_INFO.read_text()
        self.assertIn("<string>0.14.3</string>", info)

    def test_minimax_dit_uses_bounded_buffers_and_single_model_residency(self):
        models = H3_MODELS.read_text()
        denoiser = H3_DENOISER.read_text()
        media = H3_MEDIA.read_text()

        for contract in (
            "static let maximumResidentModels = 1",
            "h3SpecializationOptions(",
            "allowedComputeUnitKinds",
            "intersection([.cpu, .gpu])",
            '"Core AI failed to restrict MiniMax H3 to CPU and GPU"',
            "cachePolicy: .default",
            'manifest.inputs["graphSalt"]',
            'semantic == "graphSalt"',
            "var scratch = NDArray(",
            "outputViews.insert(&output, for: entry.outputName)",
            "_ = try await function.run(",
            "swap(&hidden, &scratch)",
            "await Task.yield()",
        ):
            self.assertIn(contract, models)
        self.assertNotIn("cachePolicy: .persistent", models)
        self.assertNotIn("outputs.names.contains(entry.outputName)", models)
        self.assertIn("maximumResidentAuxiliaryModels = 1", denoiser)
        self.assertIn("private func predictOnce(", denoiser)
        self.assertIn("private static func runAuxiliaryStage(", denoiser)
        self.assertNotIn("private let textRefiner: H3StageRunner", denoiser)
        self.assertNotIn("private let videoProjection: H3StageRunner", denoiser)
        self.assertNotIn("private let finalVideo: H3StageRunner", denoiser)
        self.assertNotIn("kCVPixelBufferIOSurfacePropertiesKey", media)
        self.assertNotIn("kCVPixelBufferMetalCompatibilityKey", media)

        view = H3_VIEW.read_text()
        for contract in (
            "maximumVisibleLogCharacters = 24_000",
            "consumeStandardError(data)",
            "isInternalCoreAIWarning",
            "guard !isInternalCoreAIWarning(line) else { continue }",
            '"#aicode."',
            '"aicode.serialization"',
            '"full compile with ane as preferred device failed"',
            '"mioh-minimax-h3-native["',
            'normalized == "error:"',
            "diagnosticPunctuation",
            '"…以前のログを省略…\\n"',
        ):
            self.assertIn(contract, view)

    def test_independent_app_has_a_distinct_green_icon(self):
        self.assertTrue(UPSCALER_ICON.is_file())
        self.assertIn("<key>CFBundleIconFile</key>", UPSCALER_INFO.read_text())
        self.assertIn("<string>AppIcon</string>", UPSCALER_INFO.read_text())

    def test_input_video_preview_seeks_and_sets_range_boundaries(self):
        source = VIDEO_PREVIEW.read_text()
        for contract in (
            "VideoPlayer(player: preview.player)",
            "Slider(",
            "toleranceBefore: .zero",
            "toleranceAfter: .zero",
            'Button("ここを開始に")',
            'Button("ここを終了に")',
            'Button("開始位置へ")',
            'Button("終了位置へ")',
            "setStart(preview.currentSeconds)",
            "setEnd(preview.currentSeconds)",
            "let seekRequest: UpscalerVideoSeekRequest?",
            ".onChange(of: seekRequest)",
            "preview.seek(to: request.seconds, duration: duration)",
        ):
            self.assertIn(contract, source)

        app = UPSCALER_APP.read_text()
        for contract in (
            "@State private var previewSeekRequest: UpscalerVideoSeekRequest?",
            "seekRequest: previewSeekRequest",
            "@FocusState private var focusedTimeInput: TimeInputField?",
            ".focused($focusedTimeInput, equals: field)",
            ".onSubmit {",
            "if oldValue == field, newValue != field",
            "requestPreviewSeek(to: upscaler.normalizedStartSeconds)",
            "requestPreviewSeek(to: upscaler.normalizedEndSeconds)",
        ):
            self.assertIn(contract, app)

    def test_controller_reports_eta_and_runs_only_selected_range(self):
        source = CONTROLLER.read_text()
        for contract in (
            "UpscalerMediaInfo",
            "UpscalerMediaProbe.read",
            "estimatedRemainingText",
            "upscaleStartedAt = Date()",
            "durationLabel(remaining)",
            '"-ss", Self.number(start)',
            '"-t", Self.number(end - start)',
            '"--output-width", String(requestedOutputWidth)',
            '"--output-height", String(requestedOutputHeight)',
            '"--temporal-strength"',
            '"--scale", String(inferenceScale)',
            "setSelectedDurationSeconds",
            "durationSeconds - normalizedStartSeconds",
        ):
            self.assertIn(contract, source)

    def test_adcsr_pipeline_and_temporal_tiling_contract(self):
        pipeline = ADCSR_PIPELINE.read_text()
        runner = ADCSR_RUNNER.read_text()
        for contract in (
            "static let inputSide = 128",
            "static let outputSide = 512",
            'inputDescriptor(of: "lr")',
            'outputDescriptor(of: "sr")',
            "input.scalarType == .float32 || input.scalarType == .float16",
            "usesFloat16",
        ):
            self.assertIn(contract, pipeline)
        for contract in (
            "VNGenerateOpticalFlowRequest",
            "kernel void accumulate_adcsr",
            "flowCurrentToPrevious",
            "previousResidual - currentResidual",
            "guard frameIndex == metadata.frameCount",
            "guard writer.frameCount == metadata.frameCount",
        ):
            self.assertIn(contract, runner)

    def test_flashvsr_shared_decode_contract_is_preserved(self):
        source = FLASHVSR_RUNNER.read_text()
        for contract in (
            "private final class NativeDecodedSegment",
            "private final class NativeMappedFrames",
            "private final class NativeMetalCompositor",
            "NativeSegmentCompositor",
            "processingFrameCount + nativeTemporalLookaheadFrames",
            "acceptedRange: warmupFrameCount..<(warmupFrameCount + frameCount)",
        ):
            self.assertIn(contract, source)


if __name__ == "__main__":
    unittest.main()
