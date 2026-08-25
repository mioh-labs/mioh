import Darwin
import Foundation

private struct H3ProgressEvent: Codable {
  let stage: String
  let state: String
  let progress: Double
  let message: String
}

private func decodeH3PipelineManifest(_ url: URL) throws -> H3PipelineManifest {
  do {
    return try JSONDecoder().decode(
      H3PipelineManifest.self,
      from: Data(contentsOf: url)
    )
  } catch DecodingError.keyNotFound(let key, _) {
    throw H3NativeError.invalidManifest(
      "\(url.lastPathComponent) is missing required key '\(key.stringValue)'. "
        + "Select the top-level MiniMax H3 manifest.json, not a component manifest."
    )
  } catch DecodingError.typeMismatch(_, let context) {
    throw H3NativeError.invalidManifest(
      "\(url.lastPathComponent) has an invalid value at "
        + context.codingPath.map(\.stringValue).joined(separator: ".")
    )
  } catch DecodingError.valueNotFound(_, let context) {
    throw H3NativeError.invalidManifest(
      "\(url.lastPathComponent) is missing a value at "
        + context.codingPath.map(\.stringValue).joined(separator: ".")
    )
  } catch DecodingError.dataCorrupted(let context) {
    throw H3NativeError.invalidManifest(
      "\(url.lastPathComponent) contains invalid JSON: \(context.debugDescription)"
    )
  }
}

private final class H3ProgressReporter: @unchecked Sendable {
  private let lock = NSLock()
  private let encoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return encoder
  }()

  func emit(_ stage: String, _ state: String, _ progress: Double, _ message: String) {
    let event = H3ProgressEvent(
      stage: stage,
      state: state,
      progress: min(1, max(0, progress)),
      message: message
    )
    lock.withLock {
      if let data = try? encoder.encode(event),
        let line = String(data: data, encoding: .utf8)
      {
        print(line)
        fflush(stdout)
      }
    }
  }
}

private struct H3ExecutionPlan: Codable {
  let input: String
  let inputImages: [String]?
  let output: String
  let backend: String
  let sourceDuration: Double
  let sourceWidth: Int
  let sourceHeight: Int
  let generationWidth: Int
  let generationHeight: Int
  let generationFrames: Int
  let referenceFrames: Int
  let videoLatentShape: [Int]
  let audioLatentShape: [Int]
  let qwenFrames: Int
  let sigmas: [Float]
  let cacheDirectory: String
}

@available(macOS 27.0, *)
private final class H3NativePipeline {
  private static let referenceMediaPreprocessingVersion =
    "native-image-reference-v4-continuous-edge-extend"
  private let manifest: H3PipelineManifest
  private let conditioningMode: H3ConditioningMode
  private let stages: [String: H3StageManifest]
  private let manifestDirectory: URL
  private let job: H3NativeJob
  private let cache: H3StageCache
  private let reporter = H3ProgressReporter()
  private let sourceURL: URL?
  private let sourceImageURLs: [URL]
  private let outputURL: URL
  private let sourceDigest: Data
  private let effectivePrompt: String

  init(manifestURL: URL, job: H3NativeJob) async throws {
    manifestDirectory = manifestURL.deletingLastPathComponent()
    let decodedManifest = try decodeH3PipelineManifest(manifestURL)
    manifest = decodedManifest
    conditioningMode = decodedManifest.resolvedConditioningMode
    try decodedManifest.validate(relativeTo: manifestDirectory)
    try job.validate(conditioningMode: conditioningMode)
    if decodedManifest.qwenComposite != nil {
      guard abs(job.durationSeconds - 10) < 0.001 else {
        throw H3NativeError.invalidJob(
          "native MiniMax H3 uses a fixed 10.0-second, 24fps profile"
        )
      }
      let variableResolution = decodedManifest.denoiserComposite?.blocks
        .allSatisfy { stage in
          stage.inputConstraints?["hiddenStates"]?.shape?.first == -1
        } == true
      if variableResolution {
        let latentPatchCells = (job.width / 32) * (job.height / 32)
        guard job.width >= 256, job.height >= 256, latentPatchCells <= 576 else {
          throw H3NativeError.invalidJob(
            "native MiniMax H3 resolution must be at least 256x256 and no larger "
              + "than the tested 1024x576-equivalent area"
          )
        }
      } else {
        guard job.width == 864, job.height == 480 else {
          throw H3NativeError.invalidJob(
            "the selected MiniMax H3 manifest is fixed to 864x480; select the "
              + "variable-resolution manifest"
          )
        }
      }
    }
    stages = try decodedManifest.resolvedStages(backend: job.backend)
    self.job = job
    effectivePrompt = decodedManifest.fixedPrompt ?? job.prompt
    if let input = job.input, !input.isEmpty {
      sourceURL = URL(fileURLWithPath: input).standardizedFileURL
      sourceImageURLs = []
    } else {
      sourceURL = nil
      sourceImageURLs = (job.inputImages ?? []).map {
        URL(fileURLWithPath: $0).standardizedFileURL
      }
    }
    outputURL = URL(fileURLWithPath: job.output).standardizedFileURL
    cache = try H3StageCache(
      directory: URL(fileURLWithPath: job.cacheDirectory).standardizedFileURL
    )
    if let sourceURL {
      sourceDigest = try H3StageCache.fileDigest(sourceURL)
    } else {
      sourceDigest = Data(
        H3StageCache.key(
          parts: try sourceImageURLs.map(H3StageCache.fileDigest)
        ).utf8
      )
    }
  }

  func plan() async throws -> H3ExecutionPlan {
    let source: (duration: Double, width: Int, height: Int, hasAudio: Bool)
    if let sourceURL {
      source = try await H3NativeMedia.probe(sourceURL)
    } else if !sourceImageURLs.isEmpty {
      let imageSource = try H3NativeMedia.probeImages(sourceImageURLs)
      source = (
        job.durationSeconds,
        imageSource.width,
        imageSource.height,
        false
      )
    } else {
      source = (job.durationSeconds, job.width, job.height, false)
    }
    let generationFrames = H3Geometry.alignedGenerationFrameCount(
      durationSeconds: job.durationSeconds
    )
    let referenceFrames: Int
    let qwenFrames: Int
    if conditioningMode == .fl2va, sourceURL == nil, sourceImageURLs.isEmpty {
      referenceFrames = 0
      qwenFrames = 0
    } else if !sourceImageURLs.isEmpty {
      // A still image is a one-frame DiT reference and one paired Qwen vision
      // block. It must not inherit the ten-second video-reference geometry.
      referenceFrames = 1
      qwenFrames = sourceImageURLs.count * 2
    } else {
      let available = max(
        5,
        Int((min(source.duration, job.durationSeconds) * 24).rounded(.down))
      )
      referenceFrames = try H3Geometry.referenceFrameCount(
        available: available,
        output: generationFrames
      )
      let qwenSampleCount = H3Geometry.qwenVideoSampleIndices(
        frameCount: referenceFrames
      ).count
      qwenFrames = qwenSampleCount + qwenSampleCount % 2
    }
    return H3ExecutionPlan(
      input: sourceURL?.path ?? sourceImageURLs.first?.path ?? "",
      inputImages: sourceImageURLs.isEmpty ? nil : sourceImageURLs.map(\.path),
      output: outputURL.path,
      backend: job.backend?.rawValue
        ?? stages.values.first?.backend.rawValue
        ?? "unknown",
      sourceDuration: source.duration,
      sourceWidth: source.width,
      sourceHeight: source.height,
      generationWidth: job.width,
      generationHeight: job.height,
      generationFrames: generationFrames,
      referenceFrames: referenceFrames,
      videoLatentShape: [
        1, 24, H3Geometry.videoLatentFrames(pixelFrames: generationFrames),
        job.height / 16, job.width / 16,
      ],
      audioLatentShape: [
        1, 32, 2, H3Geometry.audioLatentFrames(pixelFrames: generationFrames),
      ],
      qwenFrames: qwenFrames,
      sigmas: manifest.sigmas,
      cacheDirectory: cache.directory.path
    )
  }

  func run() async throws {
    let plan = try await plan()
    reporter.emit("prepare", "started", 0.01, "Swift H3 pipeline started")
    let denoised: H3AVLatent
    switch conditioningMode {
    case .ref2va:
      let media = try await decodedMedia(plan: plan)
      let text = try await textCondition(
        referenceVideo: media["visionVideo"] ?? media["video"],
        identityReferenceCount: sourceImageURLs.isEmpty
          ? nil : sourceImageURLs.count,
        sourceKey: sourceDigest,
        progress: 0.43
      )
      guard let context = text["context"], let tokenTags = text["tokenTags"] else {
        throw H3NativeError.missingTensor("textEncoder context/tokenTags")
      }
      if sourceImageURLs.isEmpty {
        guard let referenceVideo = media["video"],
          let referenceAudio = media["audio"]
        else {
          throw H3NativeError.missingTensor("decoded reference video/audio")
        }
        let videoKey = try stageKey(
          "videoEncoder",
          upstream: [
            sourceDigest,
            Data(Self.referenceMediaPreprocessingVersion.utf8),
            Data("\(plan.referenceFrames)x\(job.width)x\(job.height)".utf8),
          ]
        )
        let referenceVideoLatent = try await encodeReferenceVideo(
          referenceVideo,
          key: videoKey
        )
        let audioKey = try stageKey(
          "audioEncoder",
          upstream: [sourceDigest, Data("\(job.durationSeconds)@32000".utf8)]
        )
        let audioCondition = try await cachedStage(
          "audioEncoder",
          key: audioKey,
          inputs: ["audio": referenceAudio],
          progress: 0.34
        )
        guard let referenceAudioLatent = audioCondition["referenceAudioLatent"] else {
          throw H3NativeError.missingTensor("audioEncoder.referenceAudioLatent")
        }
        denoised = try await denoise(
          plan: plan,
          context: context,
          tokenTags: tokenTags,
          referenceVideoLatent: referenceVideoLatent,
          referenceAudioLatent: referenceAudioLatent,
          referenceImageLatents: nil,
          upstreamKeys: [videoKey, audioKey]
        )
      } else {
        var imageLatents: [H3Tensor] = []
        var imageKeys: [String] = []
        imageLatents.reserveCapacity(sourceImageURLs.count)
        imageKeys.reserveCapacity(sourceImageURLs.count)
        for index in sourceImageURLs.indices {
          guard let image = media["image\(index)"] else {
            throw H3NativeError.missingTensor("decoded reference image \(index + 1)")
          }
          let imageKey = try stageKey(
            "videoEncoder",
            upstream: [
              sourceDigest,
              Data(Self.referenceMediaPreprocessingVersion.utf8),
              Data("still-image:\(index):\(job.width)x\(job.height)".utf8),
            ]
          )
          imageLatents.append(
            try await encodeReferenceImage(
              image,
              key: imageKey,
              index: index,
              total: sourceImageURLs.count
            )
          )
          imageKeys.append(imageKey)
        }
        denoised = try await denoise(
          plan: plan,
          context: context,
          tokenTags: tokenTags,
          referenceVideoLatent: nil,
          referenceAudioLatent: nil,
          referenceImageLatents: imageLatents,
          upstreamKeys: imageKeys
        )
      }
    case .fl2va:
      let text = try await textCondition(
        referenceVideo: nil,
        identityReferenceCount: nil,
        sourceKey: sourceDigest,
        progress: 0.43
      )
      guard let context = text["context"], let tokenTags = text["tokenTags"] else {
        throw H3NativeError.missingTensor("textEncoder context/tokenTags")
      }
      denoised = try await denoise(
        plan: plan,
        context: context,
        tokenTags: tokenTags,
        referenceVideoLatent: nil,
        referenceAudioLatent: nil,
        referenceImageLatents: nil,
        upstreamKeys: []
      )
    }

    let decodedVideo = try await decodeVideo(denoised.video, shape: denoised.videoShape)
    let decodedAudio = try await decodeAudio(denoised.audio, shape: denoised.audioShape)
    reporter.emit("write", "started", 0.96, "Writing HEVC/AAC movie with AVFoundation")
    try await H3NativeMedia.writeMovie(
      video: decodedVideo,
      audio: decodedAudio,
      outputURL: outputURL
    )
    reporter.emit("complete", "completed", 1.0, outputURL.path)
  }

  private func decodedMedia(plan: H3ExecutionPlan) async throws
    -> [String: H3Tensor]
  {
    let key = H3StageCache.key(parts: [
      sourceDigest,
      Data(
        "media-v7-continuous-edge-extend:\(plan.referenceFrames):\(job.width):\(job.height):\(job.durationSeconds)"
          .utf8
      ),
    ])
    if let hit = try cache.load(stage: "media", key: key) {
      reporter.emit("media", "cached", 0.12, "Reused decoded 24fps video/audio")
      return hit
    }
    reporter.emit("media", "started", 0.06, "Decoding at 24fps with AVFoundation")
    let result: [String: H3Tensor]
    if let sourceURL {
      async let video = H3NativeMedia.decodeReferenceVideo(
        url: sourceURL,
        width: job.width,
        height: job.height,
        frameCount: plan.referenceFrames
      )
      async let visionVideo = H3NativeMedia.decodeReferenceVideo(
        url: sourceURL,
        width: H3Geometry.qwenVisionWidth,
        height: H3Geometry.qwenVisionHeight,
        frameCount: plan.referenceFrames
      )
      async let audio = H3NativeMedia.decodeReferenceAudio(
        url: sourceURL,
        durationSeconds: job.durationSeconds
      )
      result = try await [
        "video": video,
        "visionVideo": visionVideo,
        "audio": audio,
      ]
    } else {
      async let visionVideo = H3NativeMedia.decodeIdentityReferenceImages(
        urls: sourceImageURLs,
        width: H3Geometry.qwenVisionWidth,
        height: H3Geometry.qwenVisionHeight
      )
      var images: [String: H3Tensor] = ["visionVideo": try await visionVideo]
      for index in sourceImageURLs.indices {
        images["image\(index)"] = try H3NativeMedia.decodeReferenceImage(
          url: sourceImageURLs[index],
          width: job.width,
          height: job.height
        )
      }
      result = images
    }
    try cache.store(stage: "media", key: key, tensors: result)
    reporter.emit("media", "completed", 0.16, "Prepared native reference tensors")
    return result
  }

  private func textCondition(
    referenceVideo: H3Tensor?,
    identityReferenceCount: Int?,
    sourceKey: Data,
    progress: Double
  ) async throws -> [String: H3Tensor] {
    guard let tokenizerPath = manifest.tokenizerDirectory else {
      throw H3NativeError.invalidManifest("tokenizerDirectory is required")
    }
    let tokenizerURL = URL(
      fileURLWithPath: tokenizerPath,
      relativeTo: manifestDirectory
    ).standardizedFileURL
    let tokenizerFingerprint = try H3StageCache.assetFingerprint(tokenizerURL)
    let upstream = [
      sourceKey,
      Data(effectivePrompt.utf8),
      tokenizerFingerprint,
      Data("qwen-presentation-v7-continuous-edge-extend".utf8),
    ]
    let key: String
    if let composite = manifest.qwenComposite {
      key = H3StageCache.key(parts: [
        try JSONEncoder.h3Stable.encode(composite),
        try composite.assetFingerprint(relativeTo: manifestDirectory),
      ] + upstream)
    } else {
      key = try stageKey("textEncoder", upstream: upstream)
    }
    if let hit = try cache.load(stage: "textEncoder", key: key) {
      reporter.emit("textEncoder", "cached", progress, "Reused Qwen condition")
      return hit
    }
    reporter.emit(
      "textEncoder",
      "started",
      progress - 0.05,
      referenceVideo == nil
        ? "Tokenizing text-only prompt in Swift"
        : "Tokenizing prompt and packing 2fps vision blocks in Swift"
    )
    let tokenizer = try H3QwenBPETokenizer(directory: tokenizerURL)
    let presentation: H3QwenPresentation
    if let referenceVideo {
      presentation = try H3QwenPresentation.makeReferenceVideo(
        prompt: effectivePrompt,
        video: referenceVideo,
        tokenizer: tokenizer,
        fixedSequenceLength: manifest.qwenComposite?.sequenceLength,
        identityReferenceCount: identityReferenceCount
      )
    } else {
      presentation = try H3QwenPresentation.makeTextOnly(
        prompt: effectivePrompt,
        tokenizer: tokenizer,
        fixedSequenceLength: manifest.qwenComposite?.sequenceLength
      )
    }
    if presentation.promptTokenCount > presentation.usedPromptTokenCount {
      reporter.emit(
        "textEncoder",
        "running",
        progress - 0.04,
        "Prompt truncated from \(presentation.promptTokenCount) to \(presentation.usedPromptTokenCount) Qwen tokens"
      )
    }
    var outputs: [String: H3Tensor]
    if let composite = manifest.qwenComposite {
      let encoder = H3QwenCompositeEncoder(
        manifest: composite,
        baseDirectory: manifestDirectory,
        onProgress: { fraction, message in
          self.reporter.emit(
            "textEncoder",
            "running",
            progress - 0.05 + 0.05 * fraction,
            message
          )
        }
      )
      outputs = try await encoder.encode(presentation)
    } else {
      let runner = try await makeRunner("textEncoder")
      outputs = try await runner.predict(presentation.stageInputs)
      outputs["tokenTags"] = outputs["tokenTags"] ?? presentation.tokenTags
    }
    try cache.store(stage: "textEncoder", key: key, tensors: outputs)
    reporter.emit("textEncoder", "completed", progress, "Qwen condition completed")
    return outputs
  }

  private func encodeReferenceVideo(_ video: H3Tensor, key: String) async throws
    -> H3Tensor
  {
    if let hit = try cache.load(stage: "videoEncoder", key: key),
      let latent = hit["referenceVideoLatent"]
    {
      reporter.emit("videoEncoder", "cached", 0.25, "Reused tiled video latent")
      return latent
    }
    reporter.emit(
      "videoEncoder",
      "started",
      0.17,
      "Encoding 17-frame / 256px video VAE tiles"
    )
    let runner = try await makeRunner("videoEncoder")
    let latent = try await H3VideoVAEEncoder.encode(
      video: video,
      runner: runner,
      progress: { completed, total in
        let fraction = Double(completed) / Double(max(1, total))
        self.reporter.emit(
          "videoEncoder",
          "running",
          0.17 + 0.08 * fraction,
          "video VAE tile \(completed)/\(total)"
        )
      }
    )
    try cache.store(
      stage: "videoEncoder",
      key: key,
      tensors: ["referenceVideoLatent": latent]
    )
    reporter.emit("videoEncoder", "completed", 0.25, "Tiled video latent completed")
    return latent
  }

  private func encodeReferenceImage(
    _ image: H3Tensor,
    key: String,
    index: Int,
    total: Int
  ) async throws -> H3Tensor {
    if let hit = try cache.load(stage: "videoEncoder", key: key),
      let latent = hit["referenceImageLatent"]
    {
      reporter.emit(
        "videoEncoder", "cached", 0.25,
        "Reused image reference latent \(index + 1)/\(total)"
      )
      return latent
    }
    reporter.emit(
      "videoEncoder", "running", 0.17,
      "Encoding image reference \(index + 1)/\(total)"
    )
    let runner = try await makeRunner("videoEncoder")
    let encoded = try await H3VideoVAEEncoder.encode(video: image, runner: runner)
    let latent = try Self.firstVideoLatentFrame(encoded)
    try cache.store(
      stage: "videoEncoder",
      key: key,
      tensors: ["referenceImageLatent": latent]
    )
    reporter.emit(
      "videoEncoder", "running",
      0.17 + 0.08 * Double(index + 1) / Double(max(1, total)),
      "Image reference \(index + 1)/\(total) completed"
    )
    return latent
  }

  private static func firstVideoLatentFrame(_ latent: H3Tensor) throws
    -> H3Tensor
  {
    guard latent.shape.count == 5, latent.shape[0] == 1,
      latent.shape[1] == 24, latent.shape[2] > 0
    else {
      throw H3NativeError.invalidTensor(
        "image reference latent must be [1,24,T,H,W], got \(latent.shape)"
      )
    }
    let source = try latent.float16Values()
    let time = latent.shape[2]
    let plane = latent.shape[3] * latent.shape[4]
    var output = [Float16](repeating: 0, count: 24 * plane)
    for channel in 0..<24 {
      let sourceStart = channel * time * plane
      let destinationStart = channel * plane
      output.replaceSubrange(
        destinationStart..<(destinationStart + plane),
        with: source[sourceStart..<(sourceStart + plane)]
      )
    }
    return try H3Tensor(
      float16: output,
      shape: [1, 24, 1, latent.shape[3], latent.shape[4]]
    )
  }

  private func denoise(
    plan: H3ExecutionPlan,
    context: H3Tensor,
    tokenTags: H3Tensor,
    referenceVideoLatent: H3Tensor?,
    referenceAudioLatent: H3Tensor?,
    referenceImageLatents: [H3Tensor]?,
    upstreamKeys: [String]
  ) async throws -> H3AVLatent {
    let keyParts = upstreamKeys.map { Data($0.utf8) } + [
        Data(conditioningMode.rawValue.utf8),
        Data(effectivePrompt.utf8),
        context.bytes,
        tokenTags.bytes,
        Data(
          "\(job.seed):\(plan.videoLatentShape):\(plan.audioLatentShape):"
            .appending("\(manifest.sampler ?? "res_multistep"):")
            .appending("\(manifest.samplerNoise ?? 1):")
            .appending("\(manifest.samplerMaxStage ?? 3):\(manifest.sigmas)")
            .utf8
        ),
      ]
    let key: String
    if let composite = manifest.denoiserComposite {
      key = H3StageCache.key(parts: [
        try JSONEncoder.h3Stable.encode(composite),
        try composite.assetFingerprint(relativeTo: manifestDirectory),
      ] + keyParts)
    } else {
      key = try stageKey("denoiser", upstream: keyParts)
    }
    if let hit = try cache.load(stage: "denoiser", key: key),
      let video = hit["finalVideoLatent"], let audio = hit["finalAudioLatent"]
    {
      reporter.emit("denoiser", "cached", 0.82, "Reused final AV latent")
      return try H3AVLatent(
        video: video.floatValues(),
        videoShape: video.shape,
        audio: audio.floatValues(),
        audioShape: audio.shape
      )
    }
    reporter.emit("denoiser", "started", 0.48, "Loading MiniMax H3 diffusion graph")
    var random = H3SplitMix64(seed: job.seed)
    let initial = try H3AVLatent(
      video: random.normal(count: plan.videoLatentShape.reduce(1, *)),
      videoShape: plan.videoLatentShape,
      audio: random.normal(count: plan.audioLatentShape.reduce(1, *)),
      audioShape: plan.audioLatentShape
    )
    let denoise: H3ResMultistep.Denoiser
    let usesNativeComposite = manifest.denoiserComposite != nil
    if let compositeManifest = manifest.denoiserComposite {
      let composite = try await TenErosMaxH3DenoiserComposite(
        manifest: compositeManifest,
        baseDirectory: manifestDirectory,
        onLoad: { completed, total in
          self.reporter.emit(
            "denoiser", "loading", 0.48,
            "Loading MiniMax H3 DiT block \(completed)/\(total)"
          )
        }
      )
      let prepared: TenErosMaxH3DenoiserComposite.Prepared
      switch conditioningMode {
      case .ref2va:
        if let referenceImageLatents, !referenceImageLatents.isEmpty {
          prepared = try await composite.prepareImages(
            context: context,
            tokenTags: tokenTags,
            referenceImageLatents: referenceImageLatents,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape,
            seed: job.seed,
            visualConditionNoiseAug: manifest.visualConditionNoiseAug ?? 0.999
          )
        } else {
          guard let referenceVideoLatent, let referenceAudioLatent else {
            throw H3NativeError.missingTensor("Ref2VA reference latents")
          }
          prepared = try await composite.prepare(
            context: context,
            tokenTags: tokenTags,
            referenceVideoLatent: referenceVideoLatent,
            referenceAudioLatent: referenceAudioLatent,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape,
            seed: job.seed,
            visualConditionNoiseAug: manifest.visualConditionNoiseAug ?? 0.999,
            audioConditionNoiseAug: manifest.audioConditionNoiseAug ?? 1.0
          )
        }
      case .fl2va:
        prepared = try await composite.prepareTextToVideo(
          context: context,
          tokenTags: tokenTags,
          targetVideoShape: plan.videoLatentShape,
          targetAudioShape: plan.audioLatentShape
        )
      }
      denoise = { latent, sigma, _ in
        try await composite.denoise(
          latent,
          sigma: sigma,
          prepared: prepared,
          videoShift: self.manifest.videoShift,
          audioShift: self.manifest.audioShift
        )
      }
    } else {
      guard referenceImageLatents?.isEmpty != false,
        let referenceVideoLatent, let referenceAudioLatent
      else {
        throw H3NativeError.unsupported(
          "still-image Ref2VA and prompt-only FL2VA require the native denoiser composite"
        )
      }
      let runner = try await makeRunner("denoiser")
      let uncoercedStaticInputs: [String: H3Tensor] = [
        "context": context,
        "tokenTags": tokenTags,
        "referenceVideoLatent": referenceVideoLatent,
        "referenceAudioLatent": referenceAudioLatent,
        "referenceVideoShape": try H3Tensor(
          int32: referenceVideoLatent.shape.map(Int32.init),
          shape: [referenceVideoLatent.shape.count]
        ),
        "referenceAudioShape": try H3Tensor(
          int32: referenceAudioLatent.shape.map(Int32.init),
          shape: [referenceAudioLatent.shape.count]
        ),
        "seed": try H3Tensor(
          int32: [Int32(truncatingIfNeeded: job.seed)], shape: [1]
        ),
        "videoShift": try H3Tensor(float32: [manifest.videoShift], shape: [1]),
        "audioShift": try H3Tensor(float32: [manifest.audioShift], shape: [1]),
        "visualConditionNoiseAug": try H3Tensor(
          float32: [manifest.visualConditionNoiseAug ?? 0.999], shape: [1]
        ),
        "audioConditionNoiseAug": try H3Tensor(
          float32: [manifest.audioConditionNoiseAug ?? 1.0], shape: [1]
        ),
      ]
      let staticInputs = try coerce(uncoercedStaticInputs, for: runner)
      denoise = { latent, sigma, _ in
        var inputs = staticInputs
        inputs["latentVideo"] = try H3Tensor(
          float32: latent.video, shape: latent.videoShape
        )
        inputs["latentAudio"] = try H3Tensor(
          float32: latent.audio, shape: latent.audioShape
        )
        inputs["sigma"] = try H3Tensor(float32: [sigma], shape: [1])
        inputs = try self.coerce(inputs, for: runner)
        let outputs = try await runner.predict(inputs)
        guard let video = outputs["denoisedVideo"],
          let audio = outputs["denoisedAudio"]
        else {
          throw H3NativeError.missingTensor(
            "denoiser denoisedVideo/denoisedAudio"
          )
        }
        return try H3AVLatent(
          video: video.floatValues(),
          videoShape: video.shape,
          audio: audio.floatValues(),
          audioShape: audio.shape
        )
      }
    }
    let reportStep: @Sendable (Int, Float) -> Void = { step, sigma in
      let fraction = Double(step) / Double(self.manifest.sigmas.count - 1)
      self.reporter.emit(
        "denoiser",
        "running",
        0.50 + 0.30 * fraction,
        "step \(step)/\(self.manifest.sigmas.count - 1), sigma \(sigma)"
      )
    }
    var sampled: H3AVLatent
    switch manifest.sampler ?? "res_multistep" {
    case "er_sde":
      sampled = try await H3ERSDE.sample(
        initial: initial,
        sigmas: manifest.sigmas,
        flowShift: manifest.videoShift,
        seed: job.seed,
        sNoise: manifest.samplerNoise ?? 1,
        maxStage: manifest.samplerMaxStage ?? 3,
        denoise: denoise,
        onStep: reportStep
      )
    default:
      sampled = try await H3ResMultistep.sample(
        initial: initial,
        sigmas: manifest.sigmas,
        denoise: denoise,
        onStep: reportStep
      )
    }
    if usesNativeComposite {
      // ModelSamplingAV carries the clean audio target at videoShift/audioShift.
      let scale = manifest.videoShift / manifest.audioShift
      sampled.audio = sampled.audio.map { $0 / scale }
    }
    let cached: [String: H3Tensor] = [
      "finalVideoLatent": try H3Tensor(
        float16: sampled.video.map(Float16.init),
        shape: sampled.videoShape
      ),
      "finalAudioLatent": try H3Tensor(
        float16: sampled.audio.map(Float16.init),
        shape: sampled.audioShape
      ),
    ]
    try cache.store(stage: "denoiser", key: key, tensors: cached)
    return sampled
  }

  private func decodeVideo(_ values: [Float], shape: [Int]) async throws
    -> H3Tensor
  {
    let input = try H3Tensor(float16: values.map(Float16.init), shape: shape)
    let key = try stageKey(
      "videoDecoder",
      upstream: [
        input.bytes,
        Data(shape.description.utf8),
        Data("spatial-half-tile-affine-cosine-blend-v3".utf8),
      ]
    )
    if let hit = try cache.load(stage: "videoDecoder", key: key),
      let video = hit["video"]
    {
      reporter.emit("videoDecoder", "cached", 0.88, "Reused tiled decoded video")
      return video
    }
    reporter.emit(
      "videoDecoder",
      "started",
      0.80,
      "Decoding seven-token / 256px video VAE tiles"
    )
    let runner = try await makeRunner("videoDecoder")
    let video = try await H3VideoVAEDecoder.decode(
      latent: input,
      runner: runner,
      progress: { completed, total in
        let fraction = Double(completed) / Double(max(1, total))
        self.reporter.emit(
          "videoDecoder",
          "running",
          0.80 + 0.08 * fraction,
          "video VAE decode tile \(completed)/\(total)"
        )
      }
    )
    try cache.store(stage: "videoDecoder", key: key, tensors: ["video": video])
    reporter.emit("videoDecoder", "completed", 0.88, "Tiled decoded video completed")
    return video
  }

  private func decodeAudio(_ values: [Float], shape: [Int]) async throws
    -> H3Tensor
  {
    // Native-composite sampling already performs ModelSamplingAV's single
    // process_latent_out rescale before caching. Applying the shift ratio here
    // again attenuated generated audio by another 4x (about 12 dB).
    let input = try H3Tensor(float32: values, shape: shape)
    let key = try stageKey(
      "audioDecoder",
      upstream: [input.bytes, Data(shape.description.utf8)]
    )
    let outputs = try await cachedStage(
      "audioDecoder",
      key: key,
      inputs: ["audioLatent": input],
      progress: 0.93
    )
    guard let audio = outputs["audio"] else {
      throw H3NativeError.missingTensor("audioDecoder.audio")
    }
    return audio
  }

  private func cachedStage(
    _ name: String,
    key: String,
    inputs: [String: H3Tensor],
    progress: Double
  ) async throws -> [String: H3Tensor] {
    if let hit = try cache.load(stage: name, key: key) {
      reporter.emit(name, "cached", progress, "Reused \(name) output")
      return hit
    }
    reporter.emit(name, "started", max(0, progress - 0.04), "Loading \(name)")
    let runner = try await makeRunner(name)
    let outputs = try await runner.predict(try coerce(inputs, for: runner))
    try cache.store(stage: name, key: key, tensors: outputs)
    reporter.emit(name, "completed", progress, "\(name) completed")
    return outputs
  }

  private func makeRunner(_ name: String) async throws -> H3StageRunner {
    guard let stage = stages[name] else { throw H3NativeError.missingStage(name) }
    return try await H3StageRunner(
      name: name,
      manifest: stage,
      baseDirectory: manifestDirectory
    )
  }

  private func coerce(
    _ inputs: [String: H3Tensor],
    for runner: H3StageRunner
  ) throws -> [String: H3Tensor] {
    var result = inputs
    for semantic in runner.manifest.inputs.keys {
      guard let tensor = result[semantic] else { continue }
      if let expected = runner.manifest.inputConstraints?[semantic],
        tensor.scalarType != expected.scalarType
      {
        result[semantic] = try tensor.converted(to: expected.scalarType)
      }
    }
    return result
  }

  private func stageKey(_ name: String, upstream: [Data]) throws -> String {
    guard let stage = stages[name] else { throw H3NativeError.missingStage(name) }
    let stageData = try JSONEncoder.h3Stable.encode(stage)
    let assetURL = URL(
      fileURLWithPath: stage.asset,
      relativeTo: manifestDirectory
    ).standardizedFileURL
    let asset = try H3StageCache.assetFingerprint(assetURL)
    return H3StageCache.key(parts: [stageData, asset] + upstream)
  }
}

@available(macOS 27.0, *)
@main
struct MiniMaxH3NativeRunner {
  static func main() async {
    do {
      try await execute()
    } catch {
      let message = "mioh-minimax-h3-native: \(error.localizedDescription)\n"
      FileHandle.standardError.write(Data(message.utf8))
      exit(EXIT_FAILURE)
    }
  }

  private static func execute() async throws {
    guard CommandLine.arguments.count >= 4 else { throw usage() }
    let command = CommandLine.arguments[1]
    let options = try parseOptions(Array(CommandLine.arguments.dropFirst(2)))
    guard let manifestPath = options["manifest"] else { throw usage() }
    let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
    if command == "validate" {
      let manifest = try decodeH3PipelineManifest(manifestURL)
      try manifest.validate(relativeTo: manifestURL.deletingLastPathComponent())
      print("valid: \(manifest.modelIdentifier)")
      return
    }
    guard command == "plan" || command == "run" else { throw usage() }
    let job = try loadJob(options)
    let pipeline = try await H3NativePipeline(manifestURL: manifestURL, job: job)
    if command == "plan" {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
      print(String(data: try encoder.encode(await pipeline.plan()), encoding: .utf8)!)
    } else {
      try await pipeline.run()
    }
  }

  private static func loadJob(_ options: [String: String]) throws -> H3NativeJob {
    if let jobPath = options["job"] {
      return try JSONDecoder().decode(
        H3NativeJob.self,
        from: Data(contentsOf: URL(fileURLWithPath: jobPath))
      )
    }
    guard let output = options["output"],
      let prompt = options["prompt"],
      let cache = options["cache"]
    else { throw usage() }
    let inputImages: [String]?
    if let encoded = options["input-images-json"] {
      inputImages = try JSONDecoder().decode(
        [String].self,
        from: Data(encoded.utf8)
      )
    } else {
      inputImages = nil
    }
    return H3NativeJob(
      input: options["input"],
      inputImages: inputImages,
      output: output,
      prompt: prompt,
      cacheDirectory: cache,
      width: Int(options["width"] ?? "864") ?? 864,
      height: Int(options["height"] ?? "480") ?? 480,
      durationSeconds: Double(options["duration"] ?? "10") ?? 10,
      seed: UInt64(options["seed"] ?? "261662374822964") ?? 261662374822964,
      backend: options["backend"].flatMap(H3BackendKind.init(rawValue:)),
      preserveSourceAudioWhenDecoderIsUnavailable: false
    )
  }

  private static func parseOptions(_ arguments: [String]) throws
    -> [String: String]
  {
    var result: [String: String] = [:]
    var index = 0
    while index < arguments.count {
      let key = arguments[index]
      guard key.hasPrefix("--"), index + 1 < arguments.count else {
        throw usage()
      }
      result[String(key.dropFirst(2))] = arguments[index + 1]
      index += 2
    }
    return result
  }

  private static func usage() -> H3NativeError {
    .invalidArguments(
      "usage: mioh-minimax-h3-native <validate|plan|run> --manifest <manifest.json> "
        + "[--job <job.json> | (--input <video> | --input-images-json <json>) "
        + "--output <mp4> --prompt <text> "
        + "--cache <dir> --backend <coreai|coreml> "
        + "--width <multiple-of-32> --height <multiple-of-32> "
        + "--duration 10 --seed N]"
    )
  }
}
