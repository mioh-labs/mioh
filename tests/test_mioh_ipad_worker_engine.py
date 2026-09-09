import hashlib
import json
import unicodedata
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "MiohRemote" / "MiohRemote"
ENGINE = APP / "MiohIPadWorkerEngine.swift"
PROCESSOR = APP / "MiohIPadFrameProcessor.swift"
WRITER = APP / "MiohIPadVideoWriter.swift"
REMUXER = APP / "IPadMPEGTSRemuxer.swift"
STORE = APP / "IPadStandaloneStore.swift"
VARIABLE_RESTORER = APP / "MiohIPadVariableRestorer.swift"
IDENTITY_MANIFEST = APP / "mioh-cluster-model-identities-v1.json"
VARIABLE_SOURCE_MODELS = (
    ROOT / "build" / "macos-standalone" / "variable-basicvsrpp-source"
)
PROJECT = ROOT / "apps" / "MiohRemote" / "MiohRemote.xcodeproj" / "project.pbxproj"
M5_COMPILED_MODELS = ROOT / "build" / "ios-coreai-h17g"
WORKER_SERVICE = (
    ROOT
    / "packages"
    / "MiohRemoteKit"
    / "Sources"
    / "MiohRemoteKit"
    / "MiohClusterWorkerService.swift"
)
RANGE_ASSET = (
    ROOT
    / "packages"
    / "MiohRemoteKit"
    / "Sources"
    / "MiohRemoteKit"
    / "MiohHTTPRangeAsset.swift"
)


class MiohIPadWorkerEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ENGINE.read_text()
        cls.processor = PROCESSOR.read_text()
        cls.writer = WRITER.read_text()
        cls.remuxer = REMUXER.read_text()
        cls.store = STORE.read_text()
        cls.variable_restorer = VARIABLE_RESTORER.read_text()
        cls.project = PROJECT.read_text()
        cls.worker_service = WORKER_SERVICE.read_text()
        cls.range_asset = RANGE_ASSET.read_text()
        cls.native_sources = "\n".join(
            [cls.engine, cls.processor, cls.writer, cls.variable_restorer]
        )

    def test_startup_validation_is_digest_only_and_runtime_loading_is_lazy(self):
        prepare = self.engine.split(
            "static func prepare(modelRoot: URL)", 1
        )[1].split("private static func validateRestorationContract", 1)[0]
        for contract in [
            "MiohPortableModelIdentityManifest.load(from: modelRoot)",
            "manifest.validateCoreAIModelDigest(",
            "identifier: spec.identifier",
            "manifest.validateCoreAIModelCollectionDigest(",
            "identifier: variableRestorationModelIdentifier",
            "MiohIPadExecutionCore(",
            "return MiohIPadPreparedWorker(",
        ]:
            self.assertIn(contract, prepare)

        self.assertNotIn("AIModel(contentsOf:", prepare)
        self.assertNotIn("MiohIPadVariableRestorer(", prepare)
        self.assertIn("private func loadSelectedModels(", self.engine)
        self.assertIn("let model = try await AIModel(", self.engine)
        self.assertIn("maximumFrames: clipLength", self.engine)
        self.assertIn("!ProcessInfo.processInfo.isiOSAppOnMac", prepare)

    def test_fixed_and_variable_restorer_and_detector_contracts_are_exact(self):
        for contract in [
            'RestorationSpec(identifier: "basicvsrpp-v1.2-coreai", frameCount: 18)',
            'RestorationSpec(identifier: "basicvsrpp-v1.2-coreai-t36", frameCount: 36)',
            'RestorationSpec(identifier: "basicvsrpp-v1.2-coreai-t90", frameCount: 90)',
            'DetectorSpec(identifier: "v2-coreai", candidateChannels: 37)',
            'DetectorSpec(identifier: "v4-fast-coreai", candidateChannels: 38)',
            '"basicvsrpp-v1.2-coreai-variable"',
            'of: "frames"',
            'of: "restored"',
            "input.shape == shape, output.shape == shape",
            "let shape = [1, frameCount, 3, 256, 256]",
            'of: "image"',
            'of: "candidates"',
            'of: "prototypes"',
            "input.shape == [1, 3, 640, 640]",
            "candidates.shape == [1, candidateChannels, 8400]",
            "prototypes.shape == [1, 32, 160, 160]",
            "input.scalarType == .float16",
        ]:
            self.assertIn(contract, self.engine)

    def test_coreml_detectors_use_neural_engine_without_coreai_aliasing(self):
        for identifier in [
            "v2-coreml",
            "v3.1-fast-coreml",
            "v3.1-accurate-coreml",
            "v4-fast-coreml",
            "v4-accurate-coreml",
            "vr-v2-accurate-coreml",
        ]:
            self.assertIn(f'DetectorSpec(identifier: "{identifier}"', self.engine)
        for contract in [
            "import CoreML",
            'detectorIdentifier.hasSuffix("-coreml")',
            "MLModel.compileModel(",
            "configuration.computeUnits = .cpuAndNeuralEngine",
            "MLFeatureValue(pixelBuffer: letterboxed)",
            "private func readMultiArray(",
        ]:
            self.assertIn(contract, self.engine)
        for package in [
            "lada_mosaic_detection_model_v2.mlpackage",
            "lada_mosaic_detection_model_v3.1_fast.mlpackage",
            "lada_mosaic_detection_model_v3.1_accurate.mlpackage",
            "lada_mosaic_detection_model_v4_fast.mlpackage",
            "lada_mosaic_detection_model_v4_accurate.mlpackage",
            "lada_mosaic_detection_model_vr_v2_accurate.mlpackage",
        ]:
            self.assertIn(package, self.project)

    def test_realtime_mask_reuse_has_a_bounded_configurable_skip_count(self):
        for contract in [
            "protocol MiohIPadRealtimeFrameSessioning: Sendable",
            "actor MiohIPadRealtimeFrameSession:",
            "setDetectionMaskReuseSkipFrames",
            "!frameIndex.isMultiple(of: skipFrames + 1)",
            "detections = reused",
            "CoreAI still",
            "floor(Double(max(0, ptsNanoseconds)) * 24",
        ]:
            self.assertIn(contract, self.engine)

    def test_realtime_uses_configured_empty_frame_lookahead(self):
        for contract in [
            "detectionEmptyLookahead: options.detectionEmptyLookahead",
            "emptyLookahead: detectionEmptyLookahead",
            "self.emptyLookahead = min(120, max(1, emptyLookahead))",
            "windowStart + emptyLookahead",
            "private func detectWindow(",
            "if first.isEmpty, last.isEmpty",
            "return frames.map { detectedFrame($0, detections: []) }",
        ]:
            self.assertIn(contract, self.engine)
        for contract in [
            "request.options.detectionMaskReuseSkipFrames ?? 0",
            "if skipFrames > 0 {",
            "if index.isMultiple(of: skipFrames + 1)",
            "detections = reused",
            "if first.isEmpty, last.isEmpty",
        ]:
            self.assertIn(contract, self.engine)

    def test_realtime_path_precomputes_sampling_and_compositing_plans(self):
        for contract in [
            "CVPixelBufferLockBaseAddress(source, .readOnly)",
            "private func prepare(_ scene:",
            "private static func writeModelInput(",
            "private static func makeModelInputAxes(",
            "private static func makeCompositeAxes(",
            "MiohIPadCompositePlan(",
            "private func composite(",
            "func crossfade(",
        ]:
            self.assertIn(contract, self.processor)
        self.assertNotIn("MiohIPadFrameMetalProcessor", self.processor)
        self.assertNotIn("crossfadeBatch(", self.processor)

    def test_realtime_pipeline_processes_one_batch_at_a_time(self):
        for contract in [
            "private func processBatch(",
            "let detected = try await batchDetector.detectBatch(",
            "let restored = try await processor.process(detected)",
            "MiohIPadRealtimeBatchDetector",
            "MiohIPadRealtimePerformanceSample",
            "let restoredFrames: Int",
            "let modelRestorationSeconds: Double",
            "processor.lastRestoredFrameCount",
            "processor.lastRestorationSeconds",
            "takePerformanceSamples()",
            "try processor.crossfade(",
        ]:
            self.assertIn(contract, self.engine)
        for contract in [
            "private let maximumPipelineBatches = 2",
            "startDetectionIfPossible()",
            "startRestorationIfPossible()",
            "crossfadeBatch(",
        ]:
            self.assertNotIn(contract, self.engine)

    def test_variable_restorer_keeps_bounded_per_chunk_core_ai_stages(self):
        for contract in [
            "chunkFrameBuffer",
            "restoredChunkBuffer",
            "MiohIPadRestoredFrames(",
            "metalBuffer: workspace.restoredBuffer",
            "await workspace.spatialStream.currentWorkCompleted()",
            "await workspace.flowStream.currentWorkCompleted()",
            "await workspace.reconstructionStream.currentWorkCompleted()",
            "byteOffset: 0",
        ]:
            self.assertIn(contract, self.variable_restorer)
        self.assertNotIn("roundedMaximumFrames", self.variable_restorer)
        self.assertNotIn("spatialStageBuffer", self.variable_restorer)

    def test_parallel_restoration_gates_only_model_and_detaches_output(self):
        for contract in [
            "struct MiohIPadRestoredFrames: @unchecked Sendable",
            "case metalBuffer(MTLBuffer)",
            "private actor MiohIPadSharedRestorerCache",
            "sharedRestorerCache: sharedRestorerCache",
            "let restorer = try await sharedRestorerCache.restorer(",
            "MiohIPadRestorationMemoryGate(limit: 1)",
            "await MiohIPadRestorationMemoryGate.shared.acquire()",
            "await MiohIPadRestorationMemoryGate.shared.release()",
            "restored = shared.copiedValues()",
            "restored.withUnsafeBufferPointer",
        ]:
            self.assertIn(contract, self.native_sources)
        process_body = self.processor.split(
            "func process(_ detected:", 1
        )[1].split("func crossfade(", 1)[0]
        self.assertLess(
            process_body.index("let prepared = try prepare(scene)"),
            process_body.index("MiohIPadRestorationMemoryGate.shared.acquire()"),
        )
        self.assertLess(
            process_body.index("MiohIPadRestorationMemoryGate.shared.acquire()"),
            process_body.index("let shared = try await restorer.restore("),
        )
        self.assertLess(
            process_body.index("MiohIPadRestorationMemoryGate.shared.release()"),
            process_body.index(
                "outputs[frameIndex] = try restored.withUnsafeBufferPointer"
            ),
        )
        variable_restore = self.variable_restorer.split(
            "func restore(_ frames:", 1
        )[1].split("private func infer", 1)[0]
        self.assertNotIn("return Array(", variable_restore)

    def test_native_failures_keep_the_exact_pipeline_stage(self):
        for contract in [
            "case internalFailure(String)",
            "Worker実行中の未分類エラー:",
            "video track geometry metadata is unavailable:",
            "[\\(value.domain):\\(value.code)]",
        ]:
            self.assertIn(contract, self.engine)
        for contract in [
            "static func runtimeAssetURL(for sourceURL: URL)",
            "AIModel.deviceArchitectureName",
            '"\\(sourceBaseName).\\(architecture).aimodelc"',
            "FileManager.default.fileExists(atPath: compiledURL.path)",
            "モデル初期化 \\(runtimeURL.lastPathComponent)",
            "main関数読込み \\(runtimeURL.lastPathComponent)",
            "spatial encode chunk",
            "flow encode chunk",
            "propagation encode \\(functionName)",
            "reconstruction encode chunk",
            "private static func stageFailure(",
            "可変長BasicVSR++ \\(stage)",
        ]:
            self.assertIn(contract, self.variable_restorer)
        self.assertGreaterEqual(
            self.engine.count("MiohIPadVariableRestorer.runtimeAssetURL("), 2
        )

    def test_m5_runtime_specializations_are_bundled_for_all_variable_stages_and_detectors(self):
        model_names = [
            "basicvsrpp-variable-spatial6.h17g.aimodelc",
            "basicvsrpp-variable-flow6.h17g.aimodelc",
            "basicvsrpp-variable-backward_1_start6.h17g.aimodelc",
            "basicvsrpp-variable-backward_1_continue6.h17g.aimodelc",
            "basicvsrpp-variable-forward_1_start6.h17g.aimodelc",
            "basicvsrpp-variable-forward_1_continue6.h17g.aimodelc",
            "basicvsrpp-variable-backward_2_start6.h17g.aimodelc",
            "basicvsrpp-variable-backward_2_continue6.h17g.aimodelc",
            "basicvsrpp-variable-forward_2_start6.h17g.aimodelc",
            "basicvsrpp-variable-forward_2_continue6.h17g.aimodelc",
            "basicvsrpp-variable-reconstruction6.h17g.aimodelc",
            "lada_mosaic_detection_model_v2-fp16.h17g.aimodelc",
            "lada_mosaic_detection_model_v3.1_fast-fp16.h17g.aimodelc",
            "lada_mosaic_detection_model_v3.1_accurate-fp16.h17g.aimodelc",
            "lada_mosaic_detection_model_v4_fast-fp16.h17g.aimodelc",
            "lada_mosaic_detection_model_v4_accurate-fp16.h17g.aimodelc",
            "lada_mosaic_detection_model_vr_v2_accurate-fp16.h17g.aimodelc",
        ]
        for model_name in model_names:
            self.assertTrue((M5_COMPILED_MODELS / model_name).is_dir(), model_name)
            self.assertIn(model_name, self.project)

        self.assertIn('runtimeURL.pathExtension == "aimodelc"', self.engine)
        self.assertIn("cachedDetector = nil", self.engine)

    def test_worker_exposes_only_the_options_it_really_executes(self):
        for contract in [
            "transferMode: .coordinatorHTTPV1",
            "? [.coordinatorHTTPV1]",
            ": [.coordinatorHTTPV1, .sharedRootV1]",
            "supportedTransferModes: transferModes",
            "maximumConcurrentJobs: 1",
            "maximumRestorationClipLength: maximumRestorationClipLength",
            "supportsROIEnhancer: false",
            "supportsRestorationEffects: false",
            "supportsFPSConversion: true",
            'supportedInputExtensions: ["mp4", "mov", "m4v"]',
            "restorationAssetSHA256ByIdentifier:",
            "detectorAssetSHA256ByIdentifier:",
        ]:
            self.assertIn(contract, self.engine)

        for rejected in [
            "request.options.restorationClipLength <= 90",
            "request.options.roiEnhancerModelIdentifier == nil",
            "request.options.roiEnhancerAssetSHA256 == nil",
            "request.options.roiEnhancerStrength == 0",
            "request.options.sharpenStrength == 0",
            "request.options.detailBoost == 0",
            "request.options.textureMix == 0",
            "request.options.smoothStrength == 0",
            "request.options.effectUpscale == 1",
        ]:
            self.assertIn(rejected, self.engine)

        for supported in [
            "MiohIPadSourceFrameRate.outputRate(",
            "requestedNumerator: request.options.targetFPSNumerator",
            "fpsNumerator: outputRate.numerator",
            "MiohIPadPTSFrameRateGate",
        ]:
            self.assertIn(supported, self.engine)

    def test_fps_conversion_uses_absolute_gate_and_source_relative_output_pts(self):
        for contract in [
            "first source frame in each absolute target-rate time slot",
            "Double(max(0, ptsNanoseconds)) * numerator / denominatorNanoseconds",
            "+ 1e-8",
            "guard slot != lastSlot else { return false }",
            "inputFrameRateGate = gate",
            ": frame.ptsNanoseconds - coreStartNanoseconds",
            "Count-based compaction assumes",
            "FPS変換はダウン変換のみ対応しています",
        ]:
            self.assertIn(contract, self.engine)

        decoder = self.engine.split("private func decodeAndProcess(", 1)[1].split(
            "/// Matches the macOS native endpoint lookahead contract", 1
        )[0]
        emitter = self.engine.split("private final class MiohIPadShardEmitter", 1)[1].split(
            "actor MiohIPadRealtimeFrameSession", 1
        )[0]
        self.assertIn("var inputFrameRateGate: MiohIPadPTSFrameRateGate?", decoder)
        self.assertLess(
            decoder.index("let accepted = gate.accepts(ptsNanoseconds)"),
            decoder.index("let oriented = try orienter.orient(image)"),
        )
        self.assertLess(
            decoder.index("let accepted = gate.accepts(ptsNanoseconds)"),
            decoder.index("detectLookaheadWindow("),
        )
        self.assertNotIn(
            "Double(processedFrames) * Double(targetFPSDenominator)",
            emitter,
        )
        self.assertNotIn("frameRateGate", emitter)

    def test_http_only_worker_advertises_an_explicit_empty_shared_root(self):
        txt_record = self.worker_service.split(
            "let txt: [String: String] = [", 1
        )[1].split("listener.service =", 1)[0]
        self.assertIn('"root": capabilities.sharedRootIdentifier', txt_record)
        self.assertNotIn('txt["root"] =', txt_record)

    def test_execution_is_native_avfoundation_and_core_ai_only(self):
        for contract in [
            "private protocol MiohIPadVideoInputSource: Sendable",
            "func makeAsset(attemptID: UUID) async throws -> MiohIPadVideoAssetHandle",
            "MiohIPadVideoAssetHandle(asset: AVURLAsset(url: url))",
            'mediaPathExtension == "ts"',
            'AVURLAssetOverrideMIMETypeKey: "video/mp2t"',
            "AVAssetReader(asset: asset)",
            "AVAssetReaderTrackOutput(track: track",
            "AVAssetWriter(outputURL: url, fileType: .mp4)",
            "AVAssetWriterInput(mediaType: .video",
            'function.run(inputs: ["frames": input])',
            'function.run(inputs: ["image": input])',
        ]:
            self.assertIn(contract, self.native_sources)

        for forbidden in [
            "Foundation.Process",
            "Process()",
            "NSTask",
            "ffmpeg",
            "python3",
            "PythonKit",
        ]:
            self.assertNotIn(forbidden, self.native_sources)

    def test_http_transfer_uses_range_asset_but_ledger_owns_output_upload(self):
        for contract in [
            "if let transfer = request.httpTransfer",
            "MiohIPadCoordinatorHTTPInputSource(",
            "descriptorInputURL == inputURL",
            "expectedByteCount: request.inputByteCount",
            "expectedSHA256: request.inputSHA256",
            "let ranged = try MiohHTTPRangeAsset(",
            "remoteURL: url",
            "expectedByteCount: expectedByteCount",
            "expectedSHA256: expectedSHA256",
            "if await Self.probeVideoTracks(ranged.asset)",
            "retainedOwner: ranged",
            "cancelAction: { ranged.cancel() }",
            "ranged.cancel()",
            "let localURL = try await MiohIPadHTTPInputCache.shared.localFile(",
            "return MiohIPadVideoAssetHandle(asset: AVURLAsset(url: localURL))",
            "let assetHandle = try await io.input.makeAsset(attemptID: request.attemptID)",
            "withTaskCancellationHandler(",
            "defer { assetHandle.cancel() }",
            "assetHandle.cancel()",
            "MiohIPadLocalOutputSink(",
            "MiohIPadSharedRootInputSource(",
            "FileManager.default.moveItem(at: localFile, to: targetURL)",
        ]:
            self.assertIn(contract, self.engine)
        for contract in [
            "private enum MiohIPadHTTPRangeProbeError",
            "private static func probeVideoTracks(_ asset: AVURLAsset) async -> Bool",
            "try await Task.sleep(nanoseconds: 8_000_000_000)",
            "asset.loadTracks(withMediaType: .video)",
            "return !tracks.isEmpty",
        ]:
            self.assertIn(contract, self.engine)

        # The engine must leave the verified candidate for the ledger. It does
        # not own HTTP publication or a second private output file.
        for forbidden in [
            "MiohIPadCoordinatorHTTPOutputSink",
            "session.upload(",
            'request.httpMethod = "PUT"',
            "AVURLAssetHTTPHeaderFieldsKey",
            'forHTTPHeaderField: "Authorization"',
            "Data(contentsOf: localFile)",
            "URLSession.download",
            "downloadTask",
        ]:
            self.assertNotIn(forbidden, self.engine)

        range_source = self.range_asset
        for contract in [
            "AVAssetResourceLoaderDelegate",
            "public final class MiohHTTPRangeAsset",
            "asset.resourceLoader.setDelegate(self, queue: queue)",
            'request.setValue("bytes=\\(start)-\\(end)", forHTTPHeaderField: "Range")',
            "http.statusCode == 206",
            'http.value(forHTTPHeaderField: "Content-Range") == expectedContentRange',
            'http.value(forHTTPHeaderField: "ETag") == self.expectedETag',
            "configuration.timeoutIntervalForRequest = Self.transferTimeout",
            "configuration.timeoutIntervalForResource = 24 * 60 * 60",
            "session.invalidateAndCancel()",
        ]:
            self.assertIn(contract, range_source)

        for forbidden in [
            "Data(contentsOf:",
            "downloadTask",
            "download(for:",
        ]:
            self.assertNotIn(forbidden, self.engine)

        http_run = self.worker_service.split(
            "private func runHTTPAccepted(", 1
        )[1].split("private func runSharedRootAccepted(", 1)[0]
        for contract in [
            "let metrics = try await launcher(request, inputURL, candidate)",
            "let actualBytes = try validateOutput(candidate, metrics: metrics)",
            "actualBytes <= transfer.maximumOutputBytes",
            "try await httpUploader(transfer, candidate, actualBytes)",
            "try complete(request.attemptID, metrics: metrics)",
        ]:
            self.assertIn(contract, http_run)
        self.assertLess(http_run.index("launcher("), http_run.index("validateOutput("))
        self.assertLess(http_run.index("validateOutput("), http_run.index("httpUploader("))
        self.assertLess(http_run.index("httpUploader("), http_run.index("complete("))

        for contract in [
            'request.httpMethod = "PUT"',
            'forHTTPHeaderField: "Content-Length"',
            "session.upload(for: request, fromFile: localFile)",
            "(200..<300).contains(http.statusCode)",
        ]:
            self.assertIn(contract, self.worker_service)
        self.assertNotIn("Date() < transfer.expiresAt", self.worker_service)

    def test_decode_processes_padding_but_emits_only_owned_core_frames(self):
        for contract in [
            "request.mediaRange.decodeStartNanoseconds",
            "request.mediaRange.decodeEndNanoseconds",
            "request.mediaRange.coreStartNanoseconds",
            "request.mediaRange.coreEndNanoseconds",
            "request.options.temporalOverlap",
            "request.options.crossfade",
            "try await processor.process(batch)",
            "frame.ptsNanoseconds >= coreStartNanoseconds",
            "frame.ptsNanoseconds < coreEndNanoseconds",
            "writtenFrames == decodedCoreFrames",
            "writtenFrames == emitter.processedFrames",
        ]:
            self.assertIn(contract, self.engine)

    def test_detector_restoration_and_roi_composite_are_connected(self):
        for contract in [
            "nonMaximumSuppression(decoded, threshold: 0.7)",
            "makeDetectorMask(",
            "detectLookaheadWindow(",
            "restorer.restore(",
            "createBlendMask(",
            "composite(",
            "cropToBox(",
        ]:
            self.assertIn(contract, self.native_sources)

    def test_hard_detector_masks_are_bit_packed_for_parallel_mosaic_regions(self):
        for contract in [
            "struct MiohIPadBinaryMask: Sendable",
            "private(set) var words: [UInt64]",
            "let detectorMask: MiohIPadBinaryMask",
            "mask.set(y * detectorSize + x)",
            "combinedDetectorMask.formUnion(detection.detectorMask)",
            "bilinearBinaryScalar(",
        ]:
            self.assertIn(contract, self.native_sources)
        self.assertNotIn("let detectorMask: [Float]", self.processor)

    def test_mosaic_composite_reuses_the_decoded_frame_in_place(self):
        composite = self.processor.split("private func composite(", 1)[1].split(
            "private static func createBlendMask", 1
        )[0]
        for contract in [
            "let output = source",
            "CVPixelBufferLockBaseAddress(output, [])",
            "CVPixelBufferGetBaseAddress(output)",
        ]:
            self.assertIn(contract, composite)
        self.assertNotIn("allocateOutput()", composite)
        self.assertNotIn("memcpy(", composite)

    def test_mp4_is_video_only_and_published_after_successful_finish(self):
        self.assertNotIn("mediaType: .audio", self.writer)
        for contract in [
            "writer.shouldOptimizeForNetworkUse = fastStart",
            "inputPixelBufferReceiver(",
            "try await receiver.append(",
            "receiver.finish()",
            "await writer.finishWriting()",
            "try await io.output.publish(",
            "outputByteCount: Int64(byteCount)",
        ]:
            self.assertIn(contract, self.native_sources)
        self.assertIn("if #available(iOS 26.0, *)", self.writer)
        self.assertIn("legacyAdaptor", self.writer)

        self.assertLess(
            self.engine.index("let writtenFrames = try await writer.finish("),
            self.engine.index("try await io.output.publish("),
        )

    def test_variable_runner_accepts_native_recurrent_state(self):
        for contract in [
            "nativeStateN1Buffer",
            "nativeStateN2Buffer",
            "nativePreviousFlowBuffer",
            "Set(function.descriptor.stateNames)",
            '"state_n1", "state_n2", "flow_previous"',
            "states.insert(&stateN1, for: \"state_n1\")",
            "states.insert(&stateN2, for: \"state_n2\")",
            "states.insert(&previousFlow, for: \"flow_previous\")",
            "states: states",
        ]:
            self.assertIn(contract, self.variable_restorer)

    def test_remote_media_io_uses_ios_27_streaming_interfaces(self):
        for source in (self.engine, self.remuxer, self.store):
            self.assertIn("outputProvider(for: output)", source)
            self.assertIn("try reader.start()", source)
        self.assertIn("inputReceiver(for: input)", self.remuxer)
        self.assertNotIn("copyNextSampleBuffer()", self.engine)
        self.assertNotIn("startReading()", self.engine)
        self.assertNotIn("alwaysCopiesSampleData", self.engine)

        # Controller-only devices still support iOS 16, so code shared with
        # that role keeps a guarded legacy branch. Worker execution on iPadOS
        # 27 always takes the provider/receiver branch above.
        for source in (self.remuxer, self.store):
            self.assertIn("if #available(iOS 26.0, *)", source)

    def test_variable_source_identity_matches_the_bundled_manifest(self):
        if not VARIABLE_SOURCE_MODELS.is_dir():
            self.skipTest("portable variable Core AI sources are not present")
        manifest = json.loads(IDENTITY_MANIFEST.read_text())
        entry = manifest["models"]["basicvsrpp-v1.2-coreai-variable"]
        digest = hashlib.sha256()
        files = []
        for source_name in entry["source_assets"]:
            source_root = VARIABLE_SOURCE_MODELS / source_name
            self.assertTrue(source_root.is_dir(), source_root)
            for candidate in source_root.rglob("*"):
                if candidate.is_file():
                    relative = unicodedata.normalize(
                        "NFC",
                        f"{source_name}/{candidate.relative_to(source_root).as_posix()}",
                    )
                    files.append((relative, candidate))
        files.sort(key=lambda item: item[0].encode("utf-8"))
        for relative, candidate in files:
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        self.assertEqual(entry["sha256"], digest.hexdigest())

    def test_processing_budget_and_vfr_timestamps_are_hard_bounded(self):
        for contract in [
            "IPadRestorationMediaLimits.accepts(",
            "clipLength: request.options.restorationClipLength",
            "await MiohIPadSourceFrameRate.resolve(track: track)",
            "processedFrames == 0",
            ": frame.ptsNanoseconds - coreStartNanoseconds",
            "durationNanoseconds: request.mediaRange.coreEndNanoseconds",
            "track.load(.minFrameDuration)",
            "return (30, 1)",
        ]:
            self.assertIn(contract, self.native_sources)

        self.assertNotIn("max(1, Double(nominalFrameRate))", self.engine)

    def test_encoded_pixel_dimensions_ignore_odd_sample_aspect_display_width(self):
        execution = self.engine.split(
            'guard let track = tracks.first else {', 1
        )[1].split(
            "IPadRestorationMediaLimits.accepts(", 1
        )[0]

        # H.264 can carry an even coded width with a non-square sample aspect
        # ratio.  For example, 854x480 with SAR 1280:1281 is displayed as
        # 853.333x480, so AVAssetTrack.naturalSize rounds to the odd width 853
        # even though AVAssetReader decodes 854-pixel buffers.  Restoration and
        # VideoToolbox must follow the encoded pixel dimensions.
        for contract in [
            "track.load(.formatDescriptions)",
            "CMVideoFormatDescriptionGetDimensions(description)",
            "let sourcePixelSize = encodedSize ?? naturalSize",
            "CGRect(origin: .zero, size: sourcePixelSize)",
        ]:
            self.assertIn(contract, execution)

        self.assertNotIn(
            "CGRect(origin: .zero, size: naturalSize)",
            execution,
        )
        self.assertLess(
            execution.index("track.load(.formatDescriptions)"),
            execution.index("CGRect(origin: .zero, size: sourcePixelSize)"),
        )


if __name__ == "__main__":
    unittest.main()
