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
H3_CORE = UPSCALER / "MiniMaxH3NativeCore.swift"
H3_RUNNER = UPSCALER / "MiniMaxH3NativeRunner.swift"
H3_MEDIA = UPSCALER / "MiniMaxH3NativeMedia.swift"
H3_QWEN_COMPOSITE = UPSCALER / "MiniMaxH3NativeQwenComposite.swift"
H3_MODELS = UPSCALER / "MiniMaxH3NativeModels.swift"
H3_DENOISER = UPSCALER / "TenErosMaxH3DenoiserComposite.swift"
MCP_SERVER = UPSCALER / "MiohUpscalerMCPServer.swift"
H3_DIT_EXPORT = ROOT / "scripts" / "apple" / "export_10eros_max_h3_dit_block.py"
H3_DIT_DRIVER = ROOT / "scripts" / "apple" / "export_10eros_max_h3_dit_coreai.py"
H3_LORA_COMBINER = ROOT / "scripts" / "apple" / "combine_minimax_h3_loras.py"
H3_MANIFEST_BUILDER = ROOT / "scripts" / "apple" / "build_10eros_max_h3_manifest.py"
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
            "MiohUpscalerMCPServer.swift",
            '"$RESOURCES/bin/mioh-upscaler-mcp"',
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
            'FFMPEG_VERSION="8.1.2"',
            'FFMPEG_SHA256="c57c509ffc3c5456fb9a37101ec25468f4bfe20d2f68394b9f307066422642d0"',
            "TAS-FFMPEG/releases/download",
            "ditto \"$FFMPEG_PACKAGE/lib\" \"$RESOURCES/lib\"",
            "ditto \"$FFMPEG_PACKAGE/licenses\"",
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

        runner = H3_RUNNER.read_text()
        self.assertNotIn('"-frames:v", String(count), "-vsync", "0"', runner)
        for contract in (
            '"Reading source duration and the selected audio range"',
            '"Decoded \\(analysisAudio.shape[2]) analysis samples',
            '"Interval %d/%d · %.3f–%.3fs',
        ):
            self.assertIn(contract, runner)

        h3_view = H3_VIEW.read_text()
        self.assertIn('@Published private(set) var musicAnalysisSummary', h3_view)
        self.assertIn('LabeledContent("音源解析")', h3_view)
        self.assertIn('[musicAnalysis] queued:', h3_view)
        self.assertIn("private static func availableOutputURL(", h3_view)
        self.assertIn('"\\(stem) (\\(sequence))"', h3_view)
        self.assertIn("let availableOutput = Self.availableOutputURL(", h3_view)
        self.assertIn("生成完了後も中間キャッシュを残す", h3_view)
        self.assertIn("Generation cache removed:", h3_view)
        self.assertNotIn("Core AIキャッシュ保存先を選択", h3_view)
        self.assertIn(
            'environment.removeValue(forKey: "MIOH_H3_COREAI_CACHE_ROOT")',
            h3_view,
        )
        self.assertNotIn("coreAICacheRoot", h3_view)

        mcp = MCP_SERVER.read_text()
        for contract in (
            'case "mioh_start_video_generation"',
            'case "mioh_start_upscale"',
            'case "mioh_get_job_status"',
            'case "mioh_stop_job"',
            '"--prompt", runtimePrompt',
            '"prompt_passthrough": "exact"',
            '"audio_modes": ["background_music", "lip_sync"]',
        ):
            self.assertIn(contract, mcp)

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
            "isStructuredH3Prompt(originalPrompt)",
            "augmentStructuredH3Prompt(",
            "facial identity comes from",
        ):
            self.assertIn(contract, face_references)
        self.assertIn('options["input-images-json"]', runner)
        self.assertIn("qwen-presentation-v8-variable-duration", runner)
        self.assertIn("width: H3Geometry.qwenVisionWidth", runner)
        self.assertIn("height: H3Geometry.qwenVisionHeight", runner)
        self.assertIn("case .fl2va:", runner)
        self.assertIn("composite.prepareKeyframes(", runner)
        self.assertIn("prepareTextToVideo", H3_DENOISER.read_text())
        self.assertIn("prepareImages", H3_DENOISER.read_text())
        self.assertIn("prepareKeyframes", H3_DENOISER.read_text())
        self.assertIn("packedKeyframePositions", H3_DENOISER.read_text())
        self.assertNotIn(
            "continuationVideoLatent: H3Tensor?", H3_DENOISER.read_text()
        )
        self.assertIn("time: shape[2]", H3_DENOISER.read_text())
        denoiser = H3_DENOISER.read_text()
        self.assertIn("cursor += 1", denoiser)
        self.assertNotIn("cursor += videoSpan(shape[2])", denoiser)
        self.assertIn("decodeReferenceImages", media)
        self.assertIn("decodeReferenceImage", media)
        self.assertIn("decodeIdentityReferenceImages", media)
        self.assertIn("decodeReferenceImageSequence", media)
        self.assertIn("referenceImageLatents", runner)
        self.assertIn("denoiserImageReferenceCount", runner)
        self.assertIn("denoiserIdentityReferenceCount", runner)
        self.assertIn("usesDirectCutIdentity", runner)
        self.assertIn("denoiserImageReferenceCountOverride", runner)
        self.assertIn(
            "planned.denoiserIdentityReferenceCount", runner
        )
        self.assertIn("referenceImageLatents != nil", runner)
        self.assertIn("distillsVisionContext: true", runner)
        self.assertIn("distillsVisionContext: Bool = false", H3_DENOISER.read_text())
        self.assertIn(
            "Using semantic identity context without temporal image references",
            runner,
        )
        self.assertIn(
            "music-video-flat-v15-continuum-extend-hybrid-cut-qwen-storyboard-face-ref2va-layout-v3",
            runner,
        )
        self.assertIn(
            "cutImages.count - (storyboardFrame == nil ? 0 : 1)",
            runner,
        )
        self.assertIn(
            "ref2va-image-rope-v2-single-integer-slot",
            runner,
        )
        self.assertIn(
            "ref2va-semantic-identity-without-vision-rows-v1",
            runner,
        )
        mcp = MCP_SERVER.read_text()
        self.assertIn('reference_scope must be whole_image or face_only', mcp)
        self.assertIn('prepareAutomationReferences', mcp)
        self.assertIn('"face_only"', mcp)
        faces = H3_FACE_REFERENCES.read_text()
        self.assertIn("VNGeneratePersonSegmentationRequest", faces)
        self.assertIn("backgroundSoftenedIdentityImage", faces)
        self.assertIn('"CIBlendWithMask"', faces)
        self.assertIn("paddedSquareCrop", faces)
        self.assertIn("sqrt(width * height * 0.5)", faces)
        self.assertIn("kCIInputSaturationKey: 0.04", faces)
        self.assertIn('"CIMorphologyMaximum"', faces)
        self.assertIn('"CIMorphologyMinimum"', faces)
        self.assertIn("featherRadius * 2", faces)
        self.assertIn('"CIDissolveTransition"', faces)
        self.assertIn('"inputTime": 0.7', faces)
        self.assertIn("width - proposed.width", faces)
        self.assertIn(".clampedToExtent()", faces)
        self.assertIn("H3MusicVideoBoundary.cutPreRollFrames", runner)
        self.assertIn("H3MusicVideoBoundary.blendFrames", runner)
        self.assertIn("payloads = [first, entryFrames - first]", runner)
        self.assertIn("intervalPromptEntryIndices?[interval.index]", runner)
        self.assertIn("xfade=transition=fade", runner)
        self.assertIn('"-c:v", "hevc_videotoolbox"', runner)
        self.assertIn("sourceImageDigests[index]", runner)
        self.assertIn("H3QwenVisionFeatureMemoryCache()", runner)
        self.assertIn(
            "reusableVisionBlockCount: cutImageURLs.count", runner
        )
        self.assertIn("musicVideoStoryboardDirectory", runner)
        self.assertIn('String(format: "entry-%04d", entryIndex)', runner)
        self.assertIn("? cutImages.count", runner)
        self.assertIn('"storyboard_directory"', mcp)
        self.assertIn("continuationState", runner)
        self.assertIn("prepareHybridContinuation", H3_DENOISER.read_text())
        self.assertIn("packedHybridContinuationPositions", H3_DENOISER.read_text())
        self.assertNotIn("extractContinuationFrames(", runner)
        self.assertIn(
            "music-video-flat-v15-continuum-extend-hybrid",
            runner,
        )
        self.assertIn("conditioningModeOverride: conditioningMode", runner)
        self.assertIn("inputImages: [previousFrameURL]", runner)
        self.assertIn("inputImages: [previousFrameURL, lastFrameURL]", runner)
        self.assertIn("inputImages: [previousFrameURL, draftLastURL]", runner)
        self.assertIn("case .firstAndProvidedLast:", runner)
        self.assertIn("case .firstAndGeneratedLast:", runner)
        self.assertIn("continuationFrameOutputPath:", runner)
        self.assertIn("H3NativeMedia.writeReferenceImage(", runner)
        self.assertIn("H3FlatTimelinePrompt.parse(", runner)
        self.assertIn("flatPromptPlan.compiledPrompt(", runner)
        self.assertIn("promptPrefix", H3_CORE.read_text())
        self.assertIn("H3ChainPromptDocument", H3_CORE.read_text())
        self.assertIn("context_length must be", H3_CORE.read_text())
        self.assertIn("seam_taper_frames", H3_CORE.read_text())
        self.assertIn("continuationBlendFramesOverride", runner)
        self.assertIn('format: "interval-%04d.mp4"', runner)
        self.assertIn("entryFrameRanges", runner)
        self.assertIn("bestDelta", runner)
        self.assertIn("payloads.insert(payloadFrames - splitPayload", runner)
        self.assertNotIn('format: "shot-%04d-part-%02d.mp4"', runner)
        self.assertNotIn("continuationAnchor", runner)
        self.assertNotIn("let unscaled = values.map { $0 / audioScale }", runner)
        self.assertEqual(
            runner.count("sampled.audio = sampled.audio.map { $0 / scale }"),
            1,
        )
        self.assertNotIn("let unscaled = values.map { $0 / audioScale }", runner)
        self.assertIn('resolutionProfileID = "864x480"', view)
        self.assertIn('Text("24fps固定")', view)
        self.assertIn("in: 2...controller.maximumShotDuration", view)
        self.assertIn("durationSeconds: job.durationSeconds", runner)
        self.assertIn("maximumVisionBlocks:", runner)
        self.assertIn("1024, height: 576", view)
        self.assertIn("1344, height: 768", view)
        self.assertIn("width: 1920", view)
        self.assertIn("height: 1088", view)
        self.assertIn("outputHeight: 1080", view)
        self.assertIn("fixedDuration: 6", view)
        self.assertIn('"--output-width", String(outputWidth)', view)
        self.assertIn('"--output-height", String(outputHeight)', view)
        self.assertIn("768, height: 1344", view)
        self.assertIn("768, height: 768", view)
        self.assertIn("isOfficial1080pProfile", runner)
        self.assertIn("isOfficial1080pProfile ? 2_040 : 1_008", runner)
        self.assertIn("official 1920x1080 / 6-second profile", runner)
        self.assertIn("outputWidth: job.resolvedOutputWidth", runner)
        self.assertIn("outputHeight: job.resolvedOutputHeight", runner)
        self.assertIn("let cropY = (sourceHeight - outputHeight) / 2", media)
        for audio_contract in (
            'LabeledContent("音源")',
            'LabeledContent("音源の使い方")',
            'DisclosureGroup("歌詞・曲の意味")',
            "chooseLyricsFile",
            "openLyricsSearch",
            "lyricsConditionedPrompt",
            'Section("AIプロンプト生成")',
            "AIへの指示",
            "AI生成プロンプト",
            "MiniMaxプロンプトへ反映",
            "MiniMaxH3AIPromptProvider",
            "generateAIPrompt",
            "applyGeneratedAIPrompt",
            "API URL",
            "target_total_seconds",
            "selected_generation_duration_seconds",
            "shot_duration_limit_seconds",
            "The selected generation length is authoritative",
            "AUDIO ANALYSIS",
            "readAIPromptAnalysisAudio",
            "analyzedAIPromptIntervals",
            "SUBJECT REFERENCES",
            "aiSubjectReferenceSummary",
            "Do not replace <Subject 1> with generic names",
            "Do not invent extra <Subject N> labels",
            "Other performers, friends, crowds, dancers, reflections, posters",
            "face identity source only",
            "Do not write prompts that reproduce the reference photo itself",
            "Avoid close-up face shots of unreferenced people",
            "Default to live-action, photorealistic",
            "unless the user explicitly asks for an animated reinterpretation",
            "Suggested generation intervals",
            "Plain markers like [0.000-3.000] are invalid",
            "Avoid \"[start-end cut] body text\"",
            "Do not stop at",
            "[start-end cut] or [start-end continue]",
            "URLSession.shared.data",
            "LYRICS / SONG MEANING:",
            'case backgroundMusic = "background-music"',
            'case lipSync = "lip-sync"',
            '"--audio-input", audioInputURL.path',
            '"--audio-conditioning-mode", audioConditioningMode.rawValue',
            '"--music-video-cuts-json"',
            'musicVideoMode ? "music-video" : "run"',
            'Toggle("長尺Music Videoとして音源の最後まで連続生成"',
            'GroupBox("構図変更ポイント")',
            'MiniMaxH3MusicCutTimeline(',
            "元音源を音声latentとして生成中も固定",
            "口パクしない",
        ):
            self.assertIn(audio_contract, view)
        generation_settings = view.index('Section("生成設定")')
        ai_prompt_settings = view.index('Section("AIプロンプト生成")')
        self.assertLess(generation_settings, ai_prompt_settings)
        for audio_contract in (
            'command == "music-video"',
            "audio-conditioning-mode must be background-music or lip-sync",
            "resolvedAudioConditioningMode",
            "do not generate lip-sync",
            "H3AudioConditioning.samplerState(",
            "targetAudioLatent: targetAudioLatent",
            'appendingPathExtension("mioh-h3-work")',
            '"-c:v", "copy", "-c:a", "aac"',
            "musicVideoIntervalDirective(",
            "H3MusicVideoAnalyzer.timeline(",
            "cutPoints: manualCutPoints",
            "H3MusicVideoAnalyzer.generationIntervals(",
            "Show exactly one visible instance",
            "Never superimpose, overlap, ghost, double-expose",
            '"music-video-flat-v15-continuum-extend-hybrid"',
            "H3MusicVideoSeed.value(",
            "CONTINUE FORWARD.",
            "SINGLE UNINTERRUPTED TAKE.",
            "Generate only what physically follows",
            "defines intent, mood, setting, constraints, and allowed next developments",
            "it is not a restart pose, repeated opening action",
            "Treat the supplied continuation state as the real current body pose",
            "composite.prepareKeyframes(",
            "exact preceding physical state",
            'appendingPathExtension("signature")',
            'appendingPathExtension("latent-prefix.plist")',
            "continuationLatentPath: continuationLatentURL?.path",
            "temporalLatentOutputPath: temporalLatentOutputURL?.path",
            "continuationStateOverride: continuationLatent",
            "media-v14-fixed-encoder-audio-plus-exact-output",
            "audio-encoder-input:10s@32000",
            'media["outputAudio"]',
            "audio-grid:\\(plan.audioLatentShape[3])",
            "exactAudioGridSampleFrames(",
        ):
            self.assertIn(audio_contract, runner)
        for continuation_contract in (
            'case hybridAV = "hybrid-av"',
            'case latentPrefix = "latent-prefix"',
            'case firstFrame = "first"',
            'case firstAndProvidedLast = "first-last-provided"',
            'case firstAndGeneratedLast = "first-last-generated"',
        ):
            self.assertIn(continuation_contract, H3_CORE.read_text())
        for continuation_contract in (
            'Text("Hybrid AV（推奨）")',
            'Text("latent-prefix（従来方式）")',
            'Text("Firstのみ（高速）")',
            'Text("First＋Codex指定Last（1パス）")',
            'Text("First＋mioh生成Last（全自動）")',
            '"--music-video-continuation", musicVideoContinuationMode.rawValue',
            '"--music-video-last-frame-directory", musicVideoLastFrameDirectory',
        ):
            self.assertIn(continuation_contract, view)
        mcp = MCP_SERVER.read_text()
        for continuation_contract in (
            '"hybrid-av"',
            '"latent-prefix"',
            '"first-last-provided"',
            '"first-last-generated"',
            '"music_video_continuation_modes"',
            '"audio_mode"',
            '"lyrics_text"',
            '"lyrics_file"',
            "resolvedLyricsText(",
            "promptWithLyrics(",
            '"--audio-conditioning-mode"',
            '"--music-video-continuation", continuationMode',
            '"--music-video-last-frame-directory"',
        ):
            self.assertIn(continuation_contract, mcp)
        self.assertNotIn("not additional identity subjects", runner)
        self.assertNotIn("final four frames", runner)
        self.assertIn("static func fitAudioLatent(", media)
        self.assertIn("static func exactAudioGridSampleFrames(", media)
        self.assertIn("exactSampleFrames:", media)
        self.assertIn("variable-resolution manifest", runner)
        self.assertNotIn("10秒へ等間隔配置", view)
        self.assertNotIn("H3_NATIVE_ASSETS", build)
        self.assertNotIn("H3_BUNDLE_ASSETS", build)
        self.assertNotIn('models/10eros-max-h3/manifest.json', view)
        self.assertIn(
            '"$UPSCALER_DIR/MiniMaxH3MusicAnalysis.swift"', build
        )

        qwen = H3_QWEN_COMPOSITE.read_text()
        for contract in (
            "logicalBatch <= manifest.visionBlockBatch",
            "Qwen vision patch embedding (\\(missing.count)/\\(layout.blockCount) blocks)",
            "Reused all Qwen image-reference features",
            "selectingBatchBlocks(activePatches, indices: missing)",
        ):
            self.assertIn(contract, qwen)
        self.assertNotIn("paddedVisionPatches", qwen)

    def test_minimax_h3_preserves_reference_aspect_ratio(self):
        media = H3_MEDIA.read_text()
        runner = H3_RUNNER.read_text()
        self.assertIn("let uniformScale = min(", media)
        self.assertIn("scaleX: uniformScale, y: uniformScale", media)
        self.assertIn("foreground.clampedToExtent().cropped", media)
        self.assertIn("foreground.composited(over: background)", media)
        self.assertIn('"native-image-reference-v4-continuous-edge-extend"', runner)
        self.assertIn('"media-v14-fixed-encoder-audio-plus-exact-output:', runner)
        self.assertIn('"outputAudio": outputAudio', runner)
        self.assertIn('"qwen-presentation-v8-variable-duration"', runner)
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
            "let cachePolicy = try cachePolicy()",
            "cachePolicy: cachePolicy",
            '"MIOH_H3_COREAI_CACHE_POLICY"',
            "AIModelCache.Policy.persistent",
            "AIModelCache.Policy(purgeConditions: [.storagePressure])",
            "AIModelCache.Policy(purgeConditions: [.sourceAssetChangedOrDeleted])",
            "applyingRuntimeSpecializationOverrides(to: options)",
            '"MIOH_H3_COREAI_EXPECT_FREQUENT_RESHAPES"',
            "result.expectFrequentReshapes = true",
            'assetName.contains("dit-blocks")',
            'preferredCompute?.lowercased() == "gpu"',
            "SpecializationOptions(preferredComputeUnitKind: .gpu)",
            'manifest.inputs["graphSalt"]',
            'semantic == "graphSalt"',
            "var scratch = NDArray(",
            "outputViews.insert(&output, for: entry.outputName)",
            "_ = try await function.run(",
            "swap(&hidden, &scratch)",
            "await Task.yield()",
        ):
            self.assertIn(contract, models)
        self.assertNotIn("cachePolicy: AIModelCache.Policy.persistent", models)
        self.assertNotIn("outputs.names.contains(entry.outputName)", models)
        direct_dit_load = models.split(
            'if preferredCompute?.lowercased() == "gpu",', 1
        )[1].split("let options = try specializationOptions", 1)[0]
        self.assertIn('assetName.contains("dit-blocks")', direct_dit_load)
        self.assertIn("return try await AIModel(", direct_dit_load)
        self.assertNotIn("AIModel.specialize", direct_dit_load)
        runner = H3_RUNNER.read_text()
        runner_main = runner.split("static func main() async {", 1)[1].split(
            "private static func execute", 1
        )[0]
        self.assertIn("MPSGRAPH_DISABLE_ANEC_MODULE_VALIDATION", runner_main)
        self.assertIn('== "prepare-part"', runner_main)
        self.assertLess(runner_main.index("setenv("), runner_main.index("do {"))
        self.assertLess(runner_main.index("unsetenv("), runner_main.index("do {"))
        self.assertEqual(
            runner.count(
                'setenv("MPSGRAPH_DISABLE_ANEC_MODULE_VALIDATION", "1", 1)'
            ),
            1,
        )
        self.assertEqual(
            runner.count('unsetenv("MPSGRAPH_DISABLE_ANEC_MODULE_VALIDATION")'),
            1,
        )
        self.assertIn("private static func runSingleVideo(", runner)
        single_run = runner.split("private static func runSingleVideo(", 1)[1].split(
            "private static func runPreparePart", 1
        )[0]
        self.assertIn("conditioningOnly: true", single_run)
        self.assertIn("conditioningPipeline.preparationDescriptor()", single_run)
        self.assertIn("H3DenoisePrepareWorker(", single_run)
        self.assertIn("try await prepareWorker.prepare()", single_run)
        self.assertIn("let renderPipeline = try await H3NativePipeline(", single_run)
        self.assertLess(
            single_run.index("try await conditioningPipeline.run()"),
            single_run.index("try await prepareWorker.prepare()"),
        )
        self.assertLess(
            single_run.index("try await prepareWorker.prepare()"),
            single_run.index("try await renderPipeline.run()"),
        )
        self.assertIn(
            'environment["MPSGRAPH_DISABLE_ANEC_MODULE_VALIDATION"] = "1"',
            runner,
        )
        run_interval = runner.split("func runInterval(", 1)[1].split(
            "func continuationLatentURL", 1
        )[0]
        self.assertIn("let conditioningPipeline = try await H3NativePipeline(", run_interval)
        self.assertIn("conditioningModeOverride: conditioningMode", run_interval)
        self.assertIn("continuationStateOverride: continuationLatent", run_interval)
        self.assertIn("conditioningOnly: true", run_interval)
        self.assertLess(
            run_interval.index("try await conditioningPipeline.run()"),
            run_interval.index("let prepareWorker = H3DenoisePrepareWorker("),
        )
        self.assertIn('"prepare-part", "--manifest"', runner)
        self.assertIn("H3DenoisePrepareWorker(", runner)
        self.assertIn("H3PrepareWorkerDiagnosticFilter", runner)
        for contract in (
            "let decoderTicks = decoderShape[3]",
            "let samplesPerTick = outputShape[2] / decoderTicks",
            "let overlapTicks = min(16",
            "audioLatentWindow(",
            "decodeAudioChunk(",
            'Data("fixed-window-crossfade-v1:',
            "weights[globalSample] += weight",
        ):
            self.assertIn(contract, runner)
        self.assertIn('"incompatible element type for ane"', runner)
        self.assertIn('"#aicode."', runner)
        self.assertEqual(runner.count('"-nostdin"'), 2)
        self.assertIn('"mpsgraph_disable_anec_module_validation"', runner)
        self.assertIn("process.standardOutput = diagnosticPipe", runner)
        self.assertIn("process.standardError = diagnosticPipe", runner)
        self.assertIn("conditioningOnly: true", runner)
        self.assertIn("H3PipelineControl.conditioningPrepared", runner)
        self.assertIn("pipeline.run(decodeOutput: false)", runner)
        self.assertIn("guard decodeOutput else", runner)
        self.assertNotIn("configureMPSGraph", runner)
        exporter = H3_DIT_EXPORT.read_text()
        driver = H3_DIT_DRIVER.read_text()
        for contract in (
            '"--graph-identity"',
            "hidden_states = hidden_states + graph_identity_salt.sum()",
            "entrypoint_name=entrypoint_name",
            'f"w{identity_digest[:16]}_"',
            '"--lora"',
            '"--lora-strength"',
            "class LoRALinear",
            'f"diffusion_model.{prefix}"',
            "strength *= alpha / rank",
        ):
            self.assertIn(contract, exporter)
        combiner = H3_LORA_COMBINER.read_text()
        for contract in (
            '"--component"',
            "user_strength * alpha / tensor.shape[0]",
            '"scale_baked_into_lora_B": "true"',
        ):
            self.assertIn(contract, combiner)
        core = H3_CORE.read_text()
        self.assertIn('["res_multistep", "er_sde", "euler"]', core)
        self.assertIn("enum H3Euler", core)
        self.assertIn('case "euler":', runner)
        manifest_builder = H3_MANIFEST_BUILDER.read_text()
        for contract in (
            '"--sampler", choices=("res_multistep", "er_sde", "euler")',
            'parser.add_argument("--steps", type=int)',
            'simple_flow_sigmas(steps=steps, shift=args.video_shift)',
            'if sampler == "er_sde":',
        ):
            self.assertIn(contract, manifest_builder)
        for contract in (
            "checkpointSHA256",
            'f"main_{graph_identity}"',
            '"graphSalt": graph_salt_name',
        ):
            self.assertIn(contract, driver)
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
