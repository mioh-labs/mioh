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
  private let progressBase: Double
  private let progressScale: Double
  private let messagePrefix: String
  private let encoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return encoder
  }()

  init(
    progressBase: Double = 0,
    progressScale: Double = 1,
    messagePrefix: String = ""
  ) {
    self.progressBase = progressBase
    self.progressScale = progressScale
    self.messagePrefix = messagePrefix
  }

  func emit(_ stage: String, _ state: String, _ progress: Double, _ message: String) {
    let event = H3ProgressEvent(
      stage: stage,
      state: state,
      progress: min(1, max(0, progressBase + progressScale * progress)),
      message: messagePrefix + message
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
  let orderedReferences: [H3ExecutionReference]
  let output: String
  let backend: String
  let sourceDuration: Double
  let sourceWidth: Int
  let sourceHeight: Int
  let generationWidth: Int
  let generationHeight: Int
  let outputWidth: Int
  let outputHeight: Int
  let generationFrames: Int
  let referenceFrames: Int
  let videoLatentShape: [Int]
  let audioLatentShape: [Int]
  let qwenFrames: Int
  let sigmas: [Float]
  let cacheDirectory: String
}

private struct H3ExecutionReference: Codable {
  let tag: String
  let kind: String
  let path: String
  let subjectTag: String?
}

private struct H3PartPreparationDescriptor: Codable {
  let job: H3NativeJob
  let conditioningMode: H3ConditioningMode
  let visualSourceDigest: Data
  let sourceDigest: Data
  let conditioningSourceDigest: Data
  let sourceImageDigests: [Data]
  let continuationLatentPath: String?
  let temporalLatentOutputPath: String?
  let continuationFrameOutputPath: String?
  let reusableVisionBlockCount: Int
  let denoiserImageReferenceCount: Int
  let progressBase: Double
  let progressScale: Double
  let messagePrefix: String
}

private enum H3PipelineControl: Error {
  case conditioningPrepared
}

private struct H3MusicVideoIntervalPlan {
  let interval: H3MusicVideoInterval
  let outputURL: URL
  let continuationFrameURL: URL
  let continuationLatentURL: URL
  let previousFrameURL: URL?
  let previousLatentURL: URL?
  let providedLastFrameURL: URL?
  let suppliesContinuation: Bool
  let denoiserIdentityReferenceCount: Int
  let audioStart: Double
  let job: H3NativeJob
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
  private let reporter: H3ProgressReporter
  private let sourceURL: URL?
  private let sourceImageURLs: [URL]
  private let sourceImageSubjectIndices: [Int?]
  private let continuationStateOverride: H3TemporalContinuationState?
  private let temporalLatentOutputURL: URL?
  private let continuationFrameOutputURL: URL?
  private let audioSourceURL: URL?
  private let orderedReferences: [H3ExecutionReference]
  private let outputURL: URL
  private let visualSourceDigest: Data
  private let sourceImageDigests: [Data]
  private let sourceDigest: Data
  private let conditioningSourceDigest: Data
  private let effectivePrompt: String
  private let visionFeatureMemoryCache: H3QwenVisionFeatureMemoryCache?
  private let reusableVisionBlockCount: Int
  private let denoiserImageReferenceCount: Int
  private let conditioningOnly: Bool

  init(
    manifestURL: URL,
    job: H3NativeJob,
    reporter: H3ProgressReporter = H3ProgressReporter(),
    conditioningModeOverride: H3ConditioningMode? = nil,
    visualSourceDigestOverride: Data? = nil,
    sourceDigestOverride: Data? = nil,
    conditioningSourceDigestOverride: Data? = nil,
    sourceImageDigestOverrides: [Data]? = nil,
    continuationStateOverride: H3TemporalContinuationState? = nil,
    temporalLatentOutputURL: URL? = nil,
    continuationFrameOutputURL: URL? = nil,
    visionFeatureMemoryCache: H3QwenVisionFeatureMemoryCache? = nil,
    reusableVisionBlockCount: Int? = nil,
    denoiserImageReferenceCount: Int? = nil,
    conditioningOnly: Bool = false
  ) async throws {
    manifestDirectory = manifestURL.deletingLastPathComponent()
    let decodedManifest = try decodeH3PipelineManifest(manifestURL)
    manifest = decodedManifest
    conditioningMode = conditioningModeOverride
      ?? decodedManifest.resolvedConditioningMode
    try decodedManifest.validate(relativeTo: manifestDirectory)
    let allowsLatentOnlyContinuation = conditioningMode == .ref2va
      && (continuationStateOverride != nil || conditioningOnly)
      && job.input?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false
      && (job.inputImages ?? []).allSatisfy {
        $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      }
    try job.validate(
      conditioningMode: conditioningMode,
      allowsLatentOnlyContinuation: allowsLatentOnlyContinuation
    )
    if decodedManifest.qwenComposite != nil {
      let variableResolution = decodedManifest.denoiserComposite?.blocks
        .allSatisfy { stage in
          stage.inputConstraints?["hiddenStates"]?.shape?.first == -1
      } == true
      if variableResolution {
        let latentPatchCells = (job.width / 32) * (job.height / 32)
        let isOfficial1080pProfile = job.width == 1920 && job.height == 1088
          && job.resolvedOutputWidth == 1920
          && job.resolvedOutputHeight == 1080
          && abs(job.durationSeconds - 6) < 0.001
        let maximumLatentPatchCells = isOfficial1080pProfile ? 2_040 : 1_008
        guard job.width >= 256, job.height >= 256,
          latentPatchCells <= maximumLatentPatchCells
        else {
          throw H3NativeError.invalidJob(
            "native MiniMax H3 resolution must be at least 256x256 and no larger "
              + "than the H3-Base 1344x768-equivalent area, except for the "
              + "official 1920x1080 / 6-second profile"
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
    self.reporter = reporter
    if let input = job.input, !input.isEmpty {
      sourceURL = URL(fileURLWithPath: input).standardizedFileURL
    } else {
      sourceURL = nil
    }
    sourceImageURLs = (job.inputImages ?? []).map {
      URL(fileURLWithPath: $0).standardizedFileURL
    }
    if let subjects = job.inputImageSubjects {
      sourceImageSubjectIndices = subjects.map(Optional.some)
    } else {
      sourceImageSubjectIndices = Array(repeating: nil, count: sourceImageURLs.count)
    }
    effectivePrompt = Self.promptWithSubjectBindings(
      decodedManifest.fixedPrompt ?? job.prompt,
      subjectIndices: sourceImageSubjectIndices
    )
    self.continuationStateOverride = continuationStateOverride
    self.temporalLatentOutputURL = temporalLatentOutputURL?.standardizedFileURL
    self.continuationFrameOutputURL = continuationFrameOutputURL.map {
      $0.standardizedFileURL
    }
    if let sourceImageDigestOverrides {
      guard sourceImageDigestOverrides.count == sourceImageURLs.count else {
        throw H3NativeError.invalidJob(
          "source image digest count does not match the image references"
        )
      }
      sourceImageDigests = sourceImageDigestOverrides
    } else {
      sourceImageDigests = try sourceImageURLs.map(H3StageCache.fileDigest)
    }
    self.visionFeatureMemoryCache = visionFeatureMemoryCache
    self.reusableVisionBlockCount = min(
      max(0, reusableVisionBlockCount ?? sourceImageURLs.count),
      sourceImageURLs.count
    )
    self.denoiserImageReferenceCount = min(
      max(0, denoiserImageReferenceCount ?? sourceImageURLs.count),
      sourceImageURLs.count
    )
    self.conditioningOnly = conditioningOnly
    audioSourceURL = job.audioInput.flatMap { path in
      path.isEmpty ? nil : URL(fileURLWithPath: path).standardizedFileURL
    }
    if let audioSourceURL {
      let audioDuration = try await H3NativeMedia.probeDuration(audioSourceURL)
      guard job.resolvedAudioStartSeconds < audioDuration else {
        throw H3NativeError.invalidJob(
          "audioStartSeconds is beyond the end of the selected audio"
        )
      }
    }
    let subjectIndices = sourceImageSubjectIndices
    var ordered: [H3ExecutionReference] = sourceImageURLs.enumerated().map {
      index, url in
      H3ExecutionReference(
        tag: "Picture \(index + 1)",
        kind: "image",
        path: url.path,
        subjectTag: subjectIndices[index].map { "Subject \($0)" }
      )
    }
    if let sourceURL {
      ordered.append(
        H3ExecutionReference(
          tag: "Video 1",
          kind: "video",
          path: sourceURL.path,
          subjectTag: nil
        )
      )
    }
    if let audioSourceURL {
      ordered.append(
        H3ExecutionReference(
          tag: "Audio 1",
          kind: "audio",
          path: audioSourceURL.path,
          subjectTag: nil
        )
      )
    }
    orderedReferences = ordered
    outputURL = URL(fileURLWithPath: job.output).standardizedFileURL
    cache = try H3StageCache(
      directory: URL(fileURLWithPath: job.cacheDirectory).standardizedFileURL
    )
    if let visualSourceDigestOverride {
      visualSourceDigest = visualSourceDigestOverride
    } else {
      let visualParts: [Data]
      if let sourceURL {
        visualParts = sourceImageDigests + [try H3StageCache.fileDigest(sourceURL)]
      } else {
        visualParts = sourceImageDigests
      }
      visualSourceDigest = Data(H3StageCache.key(parts: visualParts).utf8)
    }
    if let sourceDigestOverride {
      sourceDigest = sourceDigestOverride
    } else {
      var sourceParts = [visualSourceDigest]
      if let audioSourceURL {
        sourceParts.append(try H3StageCache.fileDigest(audioSourceURL))
        sourceParts.append(Data("audio-start:\(job.resolvedAudioStartSeconds)".utf8))
      }
      sourceDigest = Data(H3StageCache.key(parts: sourceParts).utf8)
    }
    conditioningSourceDigest = conditioningSourceDigestOverride ?? sourceDigest
  }

  /// Make an explicit image-to-Subject contract available to Qwen/DiT when
  /// callers provide whole-image references (face-only mode already emits its
  /// own richer subject_definitions block).  The labels are descriptive only;
  /// they do not alter the user's shot direction.
  private static func promptWithSubjectBindings(
    _ prompt: String,
    subjectIndices: [Int?]
  ) -> String {
    let bindings = Dictionary(grouping: subjectIndices.enumerated().compactMap {
      index, subject -> (Int, Int)? in
      guard let subject else { return nil }
      return (subject, index + 1)
    }, by: { $0.0 })
      .sorted { $0.key < $1.key }
      .map { subject, entries in
        let pictures = entries.map { "<Picture \($0.1)>" }.joined(separator: ", ")
        return "<Subject \(subject)> is visually defined by \(pictures). Preserve this subject's identity when referenced."
      }
      .joined(separator: "\n")
    guard !bindings.isEmpty else { return prompt }
    let block = "reference_bindings:\n\(bindings)"
    if prompt.range(of: "reference_bindings:", options: .caseInsensitive) != nil
      || prompt.range(of: "subject_definitions:", options: .caseInsensitive) != nil {
      return prompt + "\n\n" + block
    }
    return block + "\n\n" + prompt
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
        audioSourceURL != nil
      )
    } else {
      source = (job.durationSeconds, job.width, job.height, false)
    }
    let generationFrames = H3Geometry.alignedGenerationFrameCount(
      durationSeconds: job.durationSeconds
    )
    let referenceFrames: Int
    let qwenFrames: Int
    if sourceURL == nil, sourceImageURLs.isEmpty,
      (conditioningMode == .fl2va || continuationStateOverride != nil
        || conditioningOnly)
    {
      referenceFrames = 0
      qwenFrames = 0
    } else if sourceURL != nil, !sourceImageURLs.isEmpty {
      let available = max(
        5,
        Int((min(source.duration, job.durationSeconds) * 24).rounded(.down))
      )
      referenceFrames = try H3Geometry.referenceFrameCount(
        available: available,
        output: generationFrames
      )
      let maximumBlocks = manifest.qwenComposite?.visionBlockBatch ?? 10
      let videoBlocks = max(1, maximumBlocks - sourceImageURLs.count)
      qwenFrames = sourceImageURLs.count * 2
        + H3Geometry.qwenVideoSampleIndices(
          frameCount: referenceFrames,
          maximumBlocks: videoBlocks
        ).count
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
        frameCount: referenceFrames,
        maximumBlocks: manifest.qwenComposite?.visionBlockBatch ?? 10
      ).count
      qwenFrames = qwenSampleCount
    }
    return H3ExecutionPlan(
      input: sourceURL?.path ?? sourceImageURLs.first?.path ?? "",
      inputImages: sourceImageURLs.isEmpty ? nil : sourceImageURLs.map(\.path),
      orderedReferences: orderedReferences,
      output: outputURL.path,
      backend: job.backend?.rawValue
        ?? stages.values.first?.backend.rawValue
        ?? "unknown",
      sourceDuration: source.duration,
      sourceWidth: source.width,
      sourceHeight: source.height,
      generationWidth: job.width,
      generationHeight: job.height,
      outputWidth: job.resolvedOutputWidth,
      outputHeight: job.resolvedOutputHeight,
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

  func run(decodeOutput: Bool = true) async throws {
    let plan = try await plan()
    if temporalLatentOutputURL != nil {
      let audioConditioningCapacity = Int(
        H3Geometry.audioConditioningSeconds
          * Double(H3Geometry.audioLatentFramesPerSecond)
      )
      guard plan.audioLatentShape[3] <= audioConditioningCapacity else {
        throw H3NativeError.invalidTensor(
          "a Part that supplies temporal continuation needs \(plan.audioLatentShape[3]) audio latent frames, exceeding the \(audioConditioningCapacity)-frame conditioning window"
        )
      }
    }
    reporter.emit("prepare", "started", 0.01, "Swift H3 pipeline started")
    let denoised: H3AVLatent
    var exactOutputAudio: H3Tensor?
    do {
      switch conditioningMode {
    case .ref2va:
      let media = try await decodedMedia(plan: plan)
      let isLatentOnlyContinuation = (continuationStateOverride != nil
        || conditioningOnly)
        && sourceURL == nil && sourceImageURLs.isEmpty
      let text = try await textCondition(
        referenceVideo: media["visionVideo"] ?? media["video"],
        identityReferenceCount: sourceImageURLs.isEmpty
          ? nil : sourceImageURLs.count,
        sourceKey: visualSourceDigest,
        progress: 0.43
      )
      guard let context = text["context"], let tokenTags = text["tokenTags"] else {
        throw H3NativeError.missingTensor("textEncoder context/tokenTags")
      }
      if isLatentOnlyContinuation {
        guard let exactAudio = media["audio"] else {
          throw H3NativeError.missingTensor(
            "latent-only continuation audio condition"
          )
        }
        let audioKey = try stageKey(
          "audioEncoder",
          upstream: [
            conditioningSourceDigest,
            Data("audio-driven:\(job.resolvedAudioStartSeconds)".utf8),
          ]
        )
        let audioCondition = try await cachedStage(
          "audioEncoder",
          key: audioKey,
          inputs: ["audio": exactAudio],
          progress: 0.34
        )
        guard let encoded = audioCondition["referenceAudioLatent"] else {
          throw H3NativeError.missingTensor("audioEncoder.referenceAudioLatent")
        }
        let targetAudioLatent = try H3NativeMedia.fitAudioLatent(
          encoded,
          to: plan.audioLatentShape
        )
        exactOutputAudio = exactAudio
        let continuationState = continuationStateOverride
        var upstreamKeys = [audioKey]
        if let continuationState {
          let continuationKey = H3StageCache.key(parts: [
            continuationState.video.bytes,
            continuationState.audio?.bytes ?? Data(),
          ])
          upstreamKeys.insert(continuationKey, at: 0)
          reporter.emit(
            "videoEncoder", "cached", 0.25,
            "Loaded preceding-Part AV continuation state"
          )
        }
        denoised = try await denoise(
          plan: plan,
          context: context,
          tokenTags: tokenTags,
          referenceVideoLatent: nil,
          referenceAudioLatent: nil,
          referenceImageLatents: nil,
          continuationState: continuationState,
          targetAudioLatent: targetAudioLatent,
          upstreamKeys: upstreamKeys
        )
      } else if sourceImageURLs.isEmpty {
        guard let referenceVideo = media["video"],
          let referenceAudio = media["audio"]
        else {
          throw H3NativeError.missingTensor("decoded reference video/audio")
        }
        let videoKey = try stageKey(
          "videoEncoder",
          upstream: [
            visualSourceDigest,
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
          upstream: [
            conditioningSourceDigest,
            Data("\(job.durationSeconds)@32000".utf8),
          ]
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
          continuationState: nil,
          targetAudioLatent: nil,
          upstreamKeys: [videoKey, audioKey]
        )
      } else {
        var imageLatents: [H3Tensor] = []
        var imageKeys: [String] = []
        let denoiserIdentityCount = denoiserImageReferenceCount
        imageLatents.reserveCapacity(denoiserIdentityCount)
        imageKeys.reserveCapacity(denoiserIdentityCount + 1)
        for index in 0..<denoiserIdentityCount {
          guard let image = media["image\(index)"] else {
            throw H3NativeError.missingTensor("decoded reference image \(index + 1)")
          }
          let imageKey = try stageKey(
            "videoEncoder",
            upstream: [
              sourceImageDigests[index],
              Data(Self.referenceMediaPreprocessingVersion.utf8),
              Data("still-image-v2:\(job.width)x\(job.height)".utf8),
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
        var referenceVideoLatent: H3Tensor?
        var referenceAudioLatent: H3Tensor?
        var mixedReferenceKeys = imageKeys
        if sourceURL != nil {
          guard let referenceVideo = media["video"],
            let referenceAudio = media["audio"]
          else {
            throw H3NativeError.missingTensor("decoded reference video/audio")
          }
          let videoKey = try stageKey(
            "videoEncoder",
            upstream: [
              visualSourceDigest,
              Data(Self.referenceMediaPreprocessingVersion.utf8),
              Data("\(plan.referenceFrames)x\(job.width)x\(job.height)".utf8),
            ]
          )
          referenceVideoLatent = try await encodeReferenceVideo(
            referenceVideo,
            key: videoKey
          )
          let audioKey = try stageKey(
            "audioEncoder",
            upstream: [
              conditioningSourceDigest,
              Data("\(job.durationSeconds)@32000".utf8),
            ]
          )
          let audioCondition = try await cachedStage(
            "audioEncoder",
            key: audioKey,
            inputs: ["audio": referenceAudio],
            progress: 0.34
          )
          guard let encoded = audioCondition["referenceAudioLatent"] else {
            throw H3NativeError.missingTensor("audioEncoder.referenceAudioLatent")
          }
          referenceAudioLatent = encoded
          mixedReferenceKeys += [videoKey, audioKey]
        }
        let continuationState = continuationStateOverride
        if let continuationStateOverride {
          mixedReferenceKeys.append(
            H3StageCache.key(parts: [
              continuationStateOverride.video.bytes,
              continuationStateOverride.audio?.bytes ?? Data(),
            ])
          )
          reporter.emit(
            "videoEncoder", "cached", 0.25,
            "Loaded preceding-Part AV continuation state"
          )
        }
        var targetAudioLatent: H3Tensor?
        var imageKeysWithAudio = mixedReferenceKeys
        if let exactAudio = media["audio"], sourceURL == nil || audioSourceURL != nil {
          let audioKey = try stageKey(
            "audioEncoder",
            upstream: [
              conditioningSourceDigest,
              Data("audio-driven:\(job.resolvedAudioStartSeconds)".utf8),
            ]
          )
          let audioCondition = try await cachedStage(
            "audioEncoder",
            key: audioKey,
            inputs: ["audio": exactAudio],
            progress: 0.34
          )
          guard let encoded = audioCondition["referenceAudioLatent"] else {
            throw H3NativeError.missingTensor("audioEncoder.referenceAudioLatent")
          }
          targetAudioLatent = try H3NativeMedia.fitAudioLatent(
            encoded,
            to: plan.audioLatentShape
          )
          exactOutputAudio = exactAudio
          imageKeysWithAudio.append(audioKey)
        }
        denoised = try await denoise(
          plan: plan,
          context: context,
          tokenTags: tokenTags,
          referenceVideoLatent: referenceVideoLatent,
          referenceAudioLatent: referenceAudioLatent,
          referenceImageLatents: imageLatents,
          continuationState: continuationState,
          targetAudioLatent: targetAudioLatent,
          upstreamKeys: imageKeysWithAudio
        )
      }
    case .fl2va:
      let media = try await decodedMedia(plan: plan)
      let text = try await textCondition(
        referenceVideo: media["visionVideo"],
        identityReferenceCount: sourceImageURLs.isEmpty
          ? nil : sourceImageURLs.count,
        sourceKey: visualSourceDigest,
        progress: 0.43
      )
      guard let context = text["context"], let tokenTags = text["tokenTags"] else {
        throw H3NativeError.missingTensor("textEncoder context/tokenTags")
      }
      var keyframeLatents: [H3Tensor] = []
      var upstreamKeys: [String] = []
      for index in sourceImageURLs.indices {
        guard let image = media["image\(index)"] else {
          throw H3NativeError.missingTensor("decoded FL2VA keyframe \(index + 1)")
        }
        let imageKey = try stageKey(
          "videoEncoder",
          upstream: [
            sourceImageDigests[index],
            Data(Self.referenceMediaPreprocessingVersion.utf8),
            Data("fl2va-keyframe-v1:\(job.width)x\(job.height)".utf8),
          ]
        )
        keyframeLatents.append(
          try await encodeReferenceImage(
            image,
            key: imageKey,
            index: index,
            total: sourceImageURLs.count
          )
        )
        upstreamKeys.append(imageKey)
      }
      var targetAudioLatent: H3Tensor?
      if let exactAudio = media["audio"] {
        let audioKey = try stageKey(
          "audioEncoder",
          upstream: [
            conditioningSourceDigest,
            Data("audio-driven:\(job.resolvedAudioStartSeconds)".utf8),
          ]
        )
        let audioCondition = try await cachedStage(
          "audioEncoder",
          key: audioKey,
          inputs: ["audio": exactAudio],
          progress: 0.34
        )
        guard let encoded = audioCondition["referenceAudioLatent"] else {
          throw H3NativeError.missingTensor("audioEncoder.referenceAudioLatent")
        }
        targetAudioLatent = try H3NativeMedia.fitAudioLatent(
          encoded,
          to: plan.audioLatentShape
        )
        exactOutputAudio = exactAudio
        upstreamKeys.append(audioKey)
      }
      denoised = try await denoise(
        plan: plan,
        context: context,
        tokenTags: tokenTags,
        referenceVideoLatent: nil,
        referenceAudioLatent: nil,
        referenceImageLatents: keyframeLatents,
        continuationState: nil,
        targetAudioLatent: targetAudioLatent,
        upstreamKeys: upstreamKeys
      )
      }
    } catch H3PipelineControl.conditioningPrepared {
      reporter.emit(
        "prepare", "completed", 0.47,
        "Prepared Qwen, reference, and audio conditions for the DiT worker"
      )
      return
    }

    if let temporalLatentOutputURL {
      let completeVideo = try H3Tensor(
        float16: denoised.video.map(Float16.init),
        shape: denoised.videoShape
      )
      let completeAudio = try H3Tensor(
        float16: denoised.audio.map(Float16.init),
        shape: denoised.audioShape
      )
      let visibleEndFrame = Int((
        (
          job.resolvedOutputTrimStartSeconds
            + job.resolvedOutputDurationSeconds
        ) * Double(H3Geometry.framesPerSecond)
      ).rounded())
      guard H3Geometry.isAlignedFrameCount(visibleEndFrame) else {
        throw H3NativeError.media(
          "a Part that supplies temporal continuation must write 5+17n frames; got \(visibleEndFrame)"
        )
      }
      let isHybrid = job.resolvedMusicVideoContinuationMode == .hybridAV
      let videoTail = try H3VideoConditioning.tail(
        completeVideo,
        maximumTokens: isHybrid
          ? H3VideoConditioning.hybridStoredTokens
          : H3VideoConditioning.partContinuationTokens,
        endingAtPixelFrameCount: visibleEndFrame
      )
      let audioTail = isHybrid
        ? try H3AudioConditioning.tail(
          completeAudio,
          maximumFrames: H3AudioConditioning.hybridStoredLatentFrames,
          endingAtPixelFrameCount: visibleEndFrame
        )
        : nil
      let state = H3TemporalContinuationState(
        video: videoTail,
        audio: audioTail
      )
      try H3TemporalLatentStore.write(state, to: temporalLatentOutputURL)
      reporter.emit(
        "denoiser", "completed", 0.83,
        isHybrid
          ? "Saved hybrid AV state (\(videoTail.shape[2]) video tokens / \(audioTail?.shape[3] ?? 0) audio ticks) for the next Part"
          : "Saved \(videoTail.shape[2]) exact temporal latent tokens for the next Part"
      )
    }

    guard decodeOutput else {
      reporter.emit(
        "denoiser", "completed", 0.83,
        "Prepared final AV latent for VAE rendering"
      )
      return
    }

    let decodedVideo = try await decodeVideo(denoised.video, shape: denoised.videoShape)
    if let continuationFrameOutputURL {
      let firstVisibleFrame = Int((
        job.resolvedOutputTrimStartSeconds
          * Double(H3Geometry.framesPerSecond)
      ).rounded())
      let visibleFrameCount = max(
        1,
        Int((
          job.resolvedOutputDurationSeconds
            * Double(H3Geometry.framesPerSecond)
        ).rounded())
      )
      try H3NativeMedia.writeReferenceImage(
        video: decodedVideo,
        frame: firstVisibleFrame + visibleFrameCount - 1,
        outputURL: continuationFrameOutputURL
      )
      reporter.emit(
        "videoDecoder", "completed", 0.955,
        "Saved the clean final visible frame for the next FL2VA Part"
      )
    }
    let decodedAudio: H3Tensor?
    if let exactOutputAudio {
      decodedAudio = exactOutputAudio
    } else {
      decodedAudio = try await decodeAudio(
        denoised.audio,
        shape: denoised.audioShape
      )
    }
    reporter.emit("write", "started", 0.96, "Writing HEVC/AAC movie with AVFoundation")
    try await H3NativeMedia.writeMovie(
      video: decodedVideo,
      audio: decodedAudio,
      outputURL: outputURL,
      durationSeconds: job.resolvedOutputDurationSeconds,
      trimStartSeconds: job.resolvedOutputTrimStartSeconds,
      outputWidth: job.resolvedOutputWidth,
      outputHeight: job.resolvedOutputHeight
    )
    reporter.emit("complete", "completed", 1.0, outputURL.path)
  }

  func denoiserPreparationDescriptor(
    progressBase: Double = 0,
    progressScale: Double = 1,
    messagePrefix: String = ""
  ) -> H3PartPreparationDescriptor {
    H3PartPreparationDescriptor(
      job: job,
      conditioningMode: conditioningMode,
      visualSourceDigest: visualSourceDigest,
      sourceDigest: sourceDigest,
      conditioningSourceDigest: conditioningSourceDigest,
      sourceImageDigests: sourceImageDigests,
      continuationLatentPath: nil,
      temporalLatentOutputPath: temporalLatentOutputURL?.path,
      continuationFrameOutputPath: continuationFrameOutputURL?.path,
      reusableVisionBlockCount: reusableVisionBlockCount,
      denoiserImageReferenceCount: denoiserImageReferenceCount,
      progressBase: progressBase,
      progressScale: progressScale,
      messagePrefix: messagePrefix
    )
  }

  private func decodedMedia(plan: H3ExecutionPlan) async throws
    -> [String: H3Tensor]
  {
    let key = H3StageCache.key(parts: [
      conditioningSourceDigest,
      Data(
        "media-v12-av-continuation:\(continuationStateOverride == nil ? "none" : "exact"):\(plan.referenceFrames):\(job.width):\(job.height):\(job.durationSeconds):\(job.resolvedAudioStartSeconds):physical-mask-\(job.physicalReferenceMask == true ? "on" : "off"):\(job.referenceEditMode?.rawValue ?? "none"):\(job.referenceEditTargetDescription ?? ""):\(job.referenceEditTargetIndex.map(String.init) ?? "none")"
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
      let visualReferenceURL = try await physicalMaskedSourceURLIfNeeded(
        originalURL: sourceURL
      )
      async let video = H3NativeMedia.decodeReferenceVideo(
        url: visualReferenceURL,
        width: job.width,
        height: job.height,
        frameCount: plan.referenceFrames
      )
      async let visionVideo = H3NativeMedia.decodeReferenceVideo(
        url: visualReferenceURL,
        width: H3Geometry.qwenVisionWidth,
        height: H3Geometry.qwenVisionHeight,
        frameCount: plan.referenceFrames
      )
      async let audio = H3NativeMedia.decodeReferenceAudio(
        url: sourceURL,
        durationSeconds: job.durationSeconds
      )
      let decodedVideo = try await video
      let decodedVisionVideo = try await visionVideo
      let decodedAudio = try await audio
      var media = [
        "video": decodedVideo,
        "visionVideo": decodedVisionVideo,
        "audio": decodedAudio,
      ]
      if !sourceImageURLs.isEmpty {
        let visionImages = try H3NativeMedia.decodeIdentityReferenceImages(
          urls: sourceImageURLs,
          width: H3Geometry.qwenVisionWidth,
          height: H3Geometry.qwenVisionHeight
        )
        media["visionVideo"] = try Self.concatenateNCTHWTime([
          visionImages,
          decodedVisionVideo,
        ])
        for index in sourceImageURLs.indices {
          media["image\(index)"] = try H3NativeMedia.decodeReferenceImage(
            url: sourceImageURLs[index],
            width: job.width,
            height: job.height
          )
        }
      }
      result = media
    } else {
      var images: [String: H3Tensor] = [:]
      if !sourceImageURLs.isEmpty {
        images["visionVideo"] = try H3NativeMedia
          .decodeIdentityReferenceImages(
            urls: sourceImageURLs,
            width: H3Geometry.qwenVisionWidth,
            height: H3Geometry.qwenVisionHeight
          )
        for index in sourceImageURLs.indices {
          images["image\(index)"] = try H3NativeMedia.decodeReferenceImage(
            url: sourceImageURLs[index],
            width: job.width,
            height: job.height
          )
        }
      }
      if let audioSourceURL {
        images["audio"] = try await H3NativeMedia.decodeReferenceAudio(
          url: audioSourceURL,
          durationSeconds: H3Geometry.audioConditioningSeconds,
          startSeconds: job.resolvedAudioStartSeconds
        )
      }
      result = images
    }
    try cache.store(stage: "media", key: key, tensors: result)
    reporter.emit("media", "completed", 0.16, "Prepared native reference tensors")
    return result
  }

  private func physicalMaskedSourceURLIfNeeded(originalURL: URL) async throws
    -> URL
  {
    guard job.physicalReferenceMask == true,
      (job.referenceEditMode ?? .none) != .none
    else { return originalURL }
    let target = job.referenceEditTargetDescription?
      .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    reporter.emit(
      "media",
      "running",
      0.07,
      target.isEmpty
        ? "Preparing physical person mask for reference video"
        : "Preparing physical person mask: \(target)"
    )
    let key = H3StageCache.key(parts: [
      visualSourceDigest,
      Data("physical-reference-mask-v1".utf8),
      Data((job.referenceEditMode?.rawValue ?? "none").utf8),
      Data(target.utf8),
      Data((job.referenceEditTargetIndex.map(String.init) ?? "none").utf8),
      Data("\(job.durationSeconds)".utf8),
    ])
    let directory = URL(fileURLWithPath: job.cacheDirectory)
      .standardizedFileURL
      .appendingPathComponent("reference-video-masks", isDirectory: true)
    let output = directory.appendingPathComponent("\(key).mp4")
    if FileManager.default.fileExists(atPath: output.path) {
      return output
    }
    return try await MiniMaxH3ReferenceVideoMaskProcessor
      .createMaskedReferenceVideo(
        sourceURL: originalURL,
        targetDescription: target,
        targetIndex: job.referenceEditTargetIndex,
        durationSeconds: job.durationSeconds,
        outputURL: output
      )
  }

  private static func concatenateNCTHWTime(_ tensors: [H3Tensor]) throws
    -> H3Tensor
  {
    guard let first = tensors.first else {
      throw H3NativeError.invalidTensor("cannot concatenate an empty tensor list")
    }
    guard first.shape.count == 5, first.shape[0] == 1 else {
      throw H3NativeError.invalidTensor(
        "Qwen visual tensors must be NCTHW, got \(first.shape)"
      )
    }
    let channels = first.shape[1]
    let height = first.shape[3]
    let width = first.shape[4]
    var totalFrames = 0
    let values = try tensors.map { tensor -> [Float] in
      guard tensor.shape.count == 5,
        tensor.shape[0] == 1,
        tensor.shape[1] == channels,
        tensor.shape[3] == height,
        tensor.shape[4] == width
      else {
        throw H3NativeError.invalidTensor(
          "cannot concatenate mismatched NCTHW tensors: \(tensors.map(\.shape))"
        )
      }
      totalFrames += tensor.shape[2]
      return try tensor.floatValues()
    }
    let plane = height * width
    var output: [Float] = []
    output.reserveCapacity(channels * totalFrames * plane)
    for channel in 0..<channels {
      for (tensor, tensorValues) in zip(tensors, values) {
        let frames = tensor.shape[2]
        let start = channel * frames * plane
        output.append(
          contentsOf: tensorValues[start..<(start + frames * plane)]
        )
      }
    }
    return try H3Tensor(
      float32: output,
      shape: [1, channels, totalFrames, height, width]
    )
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
      Data("qwen-presentation-v8-variable-duration".utf8),
    ]
    let key: String
    var compositeManifestData: Data?
    var compositeAssetFingerprint: Data?
    if let composite = manifest.qwenComposite {
      let manifestData = try JSONEncoder.h3Stable.encode(composite)
      let assetFingerprint = try composite.assetFingerprint(
        relativeTo: manifestDirectory
      )
      compositeManifestData = manifestData
      compositeAssetFingerprint = assetFingerprint
      key = H3StageCache.key(
        parts: [manifestData, assetFingerprint] + upstream
      )
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
        : "Tokenizing prompt and packing duration-aware vision blocks in Swift"
    )
    let tokenizer = try H3QwenBPETokenizer(directory: tokenizerURL)
    let presentation: H3QwenPresentation
    if let referenceVideo {
      presentation = try H3QwenPresentation.makeReferenceVideo(
        prompt: effectivePrompt,
        video: referenceVideo,
        tokenizer: tokenizer,
        fixedSequenceLength: manifest.qwenComposite?.sequenceLength,
        identityReferenceCount: identityReferenceCount,
        maximumVisionBlocks: manifest.qwenComposite?.visionBlockBatch ?? 10
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
      let visualBlockCount = try encoder.visualBlockCount(in: presentation)
      var reused: [Int: [String: H3Tensor]] = [:]
      var reusableKeys: [Int: String] = [:]
      if visualBlockCount > 0,
        let memoryCache = visionFeatureMemoryCache,
        let compositeManifestData,
        let compositeAssetFingerprint
      {
        let reusableCount = min(
          visualBlockCount,
          reusableVisionBlockCount,
          sourceImageDigests.count
        )
        for index in 0..<reusableCount {
          let featureKey = H3StageCache.key(parts: [
            Data("qwen-vision-block-v1-active-only".utf8),
            compositeManifestData,
            compositeAssetFingerprint,
            sourceImageDigests[index],
            Data(Self.referenceMediaPreprocessingVersion.utf8),
            Data("\(H3Geometry.qwenVisionWidth)x\(H3Geometry.qwenVisionHeight)".utf8),
          ])
          reusableKeys[index] = featureKey
          if let features = memoryCache.features(for: featureKey) {
            reused[index] = features
          }
        }
        if !reused.isEmpty {
          reporter.emit(
            "textEncoder", "cached", progress - 0.045,
            "Reused \(reused.count)/\(visualBlockCount) fixed image-reference features"
          )
        }
      }
      let visionFeatures: [[String: H3Tensor]]?
      if visualBlockCount > 0 {
        let features = try await encoder.makeVisionBlockFeatures(
          for: presentation,
          reusing: reused
        )
        if let memoryCache = visionFeatureMemoryCache {
          for (index, featureKey) in reusableKeys {
            memoryCache.insert(features[index], for: featureKey)
          }
        }
        visionFeatures = features
      } else {
        visionFeatures = nil
      }
      outputs = try await encoder.encode(
        presentation,
        visionBlockFeatures: visionFeatures
      )
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
    continuationState: H3TemporalContinuationState?,
    targetAudioLatent: H3Tensor?,
    upstreamKeys: [String]
  ) async throws -> H3AVLatent {
    if conditioningOnly {
      throw H3PipelineControl.conditioningPrepared
    }
    let usesSemanticCutIdentity = conditioningMode == .ref2va
      && continuationState == nil
      && referenceVideoLatent == nil
      && referenceAudioLatent == nil
      && referenceImageLatents?.isEmpty == true
    let usesImageReferenceRows = conditioningMode == .ref2va
      && referenceImageLatents?.isEmpty == false
    let keyParts = upstreamKeys.map { Data($0.utf8) } + [
        Data(conditioningMode.rawValue.utf8),
        Data(
          (continuationState == nil
            ? "part-video-context-none-v1"
            : job.resolvedMusicVideoContinuationMode == .hybridAV
              ? "part-av-context-v5-continuum-extend-hybrid"
              : "part-video-context-v4-fixed-target-prefix").utf8
        ),
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
      ] + (usesSemanticCutIdentity
        ? [Data("ref2va-semantic-identity-without-vision-rows-v1".utf8)]
        : []) + (usesImageReferenceRows
          ? [Data("ref2va-image-rope-v2-single-integer-slot".utf8)]
          : [])
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
    let isHybridContinuation = continuationState != nil
      && job.resolvedMusicVideoContinuationMode == .hybridAV
    let continuationPrefixVideo: H3Tensor?
    let hybridHistoryVideo: H3Tensor?
    let hybridHistoryAudio: H3Tensor?
    let hybridPrefixAudio: H3Tensor?
    if let continuationState, isHybridContinuation {
      guard continuationState.video.shape[2]
        == H3VideoConditioning.hybridStoredTokens,
        let storedAudio = continuationState.audio,
        storedAudio.shape[3] == H3AudioConditioning.hybridStoredLatentFrames
      else {
        throw H3NativeError.invalidTensor(
          "hybrid continuation requires \(H3VideoConditioning.hybridStoredTokens) video tokens and \(H3AudioConditioning.hybridStoredLatentFrames) audio ticks"
        )
      }
      hybridHistoryVideo = try H3VideoConditioning.slice(
        continuationState.video,
        tokens: 0..<H3VideoConditioning.hybridHistoryTokens
      )
      continuationPrefixVideo = try H3VideoConditioning.slice(
        continuationState.video,
        tokens: H3VideoConditioning.hybridHistoryTokens..<H3VideoConditioning.hybridStoredTokens
      )
      hybridHistoryAudio = try H3AudioConditioning.slice(
        storedAudio,
        frames: 0..<H3AudioConditioning.hybridHistoryLatentFrames
      )
      hybridPrefixAudio = try H3AudioConditioning.slice(
        storedAudio,
        frames: H3AudioConditioning.hybridHistoryLatentFrames..<H3AudioConditioning.hybridStoredLatentFrames
      )
    } else {
      continuationPrefixVideo = continuationState?.video
      hybridHistoryVideo = nil
      hybridHistoryAudio = nil
      hybridPrefixAudio = nil
    }
    let effectiveTargetAudioLatent: H3Tensor?
    if let targetAudioLatent, let hybridPrefixAudio {
      effectiveTargetAudioLatent = try H3AudioConditioning.replacingPrefix(
        in: targetAudioLatent,
        with: hybridPrefixAudio
      )
    } else {
      effectiveTargetAudioLatent = targetAudioLatent
    }
    let continuationPrefixShape = continuationPrefixVideo?.shape
    let continuationPrefixValues = try continuationPrefixVideo?.floatValues()
    let continuationPrefixNoise: [Float]?
    if let continuationPrefixShape {
      continuationPrefixNoise = try H3VideoConditioning.prefixValues(
        from: initial.video,
        targetShape: initial.videoShape,
        prefixShape: continuationPrefixShape
      )
    } else {
      continuationPrefixNoise = nil
    }
    if let continuationPrefixVideo {
      reporter.emit(
        "denoiser", "running", 0.48,
        isHybridContinuation
          ? "Hybrid continuation: protecting \(continuationPrefixVideo.shape[2]) AV-prefix tokens and attending to \(H3VideoConditioning.hybridHistoryTokens) older H3-Extend tokens"
          : "Clamping \(continuationPrefixVideo.shape[2]) preceding-Part tokens as the fixed target prefix"
      )
    }
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
        if isHybridContinuation,
          let hybridHistoryVideo,
          let hybridHistoryAudio,
          referenceImageLatents?.isEmpty != false,
          referenceVideoLatent == nil,
          referenceAudioLatent == nil
        {
          prepared = try await composite.prepareHybridContinuation(
            context: context,
            tokenTags: tokenTags,
            historyVideoLatent: hybridHistoryVideo,
            historyAudioLatent: hybridHistoryAudio,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape
          )
        } else if continuationState != nil,
          referenceImageLatents?.isEmpty != false,
          referenceVideoLatent == nil,
          referenceAudioLatent == nil
        {
          // Music-video continuation already carries the exact preceding
          // target history. Do not add decoded still frames as a second visual
          // condition: they duplicate the same history and force another Qwen
          // vision pass plus VAE reference encoding for every Part.
          prepared = try await composite.prepareTextToVideo(
            context: context,
            tokenTags: tokenTags,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape
          )
        } else if let referenceVideoLatent,
          let referenceAudioLatent,
          let referenceImageLatents,
          !referenceImageLatents.isEmpty
        {
          prepared = try await composite.prepareVideoWithImages(
            context: context,
            tokenTags: tokenTags,
            referenceVideoLatent: referenceVideoLatent,
            referenceAudioLatent: referenceAudioLatent,
            referenceImageLatents: referenceImageLatents,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape,
            seed: job.seed,
            visualConditionNoiseAug: manifest.visualConditionNoiseAug ?? 0.999,
            audioConditionNoiseAug: manifest.audioConditionNoiseAug ?? 1.0
          )
        } else if let referenceImageLatents, !referenceImageLatents.isEmpty {
          prepared = try await composite.prepareImages(
            context: context,
            tokenTags: tokenTags,
            referenceImageLatents: referenceImageLatents,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape,
            seed: job.seed,
            visualConditionNoiseAug: manifest.visualConditionNoiseAug ?? 0.999
          )
        } else if referenceImageLatents != nil {
          // Qwen has already propagated the selected identities into the
          // following prompt rows. Remove its raw vision rows, as well as the
          // image latents, before DiT so a hard Cut cannot replay a reference
          // image as the target clip's opening frames.
          reporter.emit(
            "videoEncoder", "cached", 0.25,
            "Using semantic identity context without temporal image references"
          )
          prepared = try await composite.prepareTextToVideo(
            context: context,
            tokenTags: tokenTags,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape,
            distillsVisionContext: true
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
        if let referenceImageLatents, !referenceImageLatents.isEmpty {
          let indices = referenceImageLatents.count == 1
            ? [0]
            : [0, plan.generationFrames - 1]
          prepared = try await composite.prepareKeyframes(
            context: context,
            tokenTags: tokenTags,
            keyframeLatents: referenceImageLatents,
            pixelFrameIndices: indices,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape,
            seed: job.seed,
            visualConditionNoiseAug: manifest.visualConditionNoiseAug ?? 0.999
          )
        } else {
          prepared = try await composite.prepareTextToVideo(
            context: context,
            tokenTags: tokenTags,
            targetVideoShape: plan.videoLatentShape,
            targetAudioShape: plan.audioLatentShape
          )
        }
      }
      let targetAudioValues = try effectiveTargetAudioLatent?.floatValues()
      let audioConditionNoise = initial.audio
      denoise = { latent, sigma, _ in
        var conditioned = latent
        if let continuationPrefixShape,
          let continuationPrefixValues,
          let continuationPrefixNoise
        {
          conditioned.video = try H3VideoConditioning.clampTargetPrefix(
            target: conditioned.video,
            targetShape: conditioned.videoShape,
            cleanPrefix: continuationPrefixValues,
            noisePrefix: continuationPrefixNoise,
            prefixShape: continuationPrefixShape,
            sigma: sigma
          )
        }
        if let targetAudioValues {
          conditioned.audio = H3AudioConditioning.samplerState(
            clean: targetAudioValues,
            noise: audioConditionNoise,
            videoSigma: sigma,
            videoShift: self.manifest.videoShift,
            audioShift: self.manifest.audioShift
          )
        }
        return try await composite.denoise(
          conditioned,
          sigma: sigma,
          prepared: prepared,
          videoShift: self.manifest.videoShift,
          audioShift: self.manifest.audioShift
        )
      }
    } else {
      guard referenceImageLatents?.isEmpty != false,
        continuationState == nil,
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
    if let continuationPrefixShape,
      let continuationPrefixValues,
      let continuationPrefixNoise
    {
      sampled.video = try H3VideoConditioning.clampTargetPrefix(
        target: sampled.video,
        targetShape: sampled.videoShape,
        cleanPrefix: continuationPrefixValues,
        noisePrefix: continuationPrefixNoise,
        prefixShape: continuationPrefixShape,
        sigma: 0
      )
    }
    if let effectiveTargetAudioLatent {
      sampled.audio = try effectiveTargetAudioLatent.floatValues()
    } else if usesNativeComposite {
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
    let generated = try H3Tensor(float32: values, shape: shape)
    let input: H3Tensor
    if let expectedShape = stages["audioDecoder"]?
      .inputConstraints?["audioLatent"]?.shape,
      expectedShape.count == shape.count,
      expectedShape.allSatisfy({ $0 > 0 }),
      expectedShape != shape
    {
      // The H3 audio decoder is exported with a fixed latent input window
      // (currently 405 ticks), while shorter prompt/FL2VA clips generate only
      // the visible-duration latent (for example 8s -> 320 ticks). Pad/crop at
      // the decoder boundary; movie writing still trims to the requested
      // output duration.
      input = try H3NativeMedia.fitAudioLatent(generated, to: expectedShape)
    } else {
      input = generated
    }
    let key = try stageKey(
      "audioDecoder",
      upstream: [input.bytes, Data(input.shape.description.utf8)]
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
private final class H3PrepareWorkerDiagnosticFilter: @unchecked Sendable {
  private let lock = NSLock()
  private var pending = Data()

  func consume(_ data: Data) {
    guard !data.isEmpty else {
      finish()
      return
    }
    lock.withLock {
      pending.append(data)
      forwardCompleteLines()
    }
  }

  func finish() {
    lock.withLock {
      forwardCompleteLines()
      guard !pending.isEmpty else { return }
      forward(pending)
      pending.removeAll(keepingCapacity: false)
    }
  }

  private func forwardCompleteLines() {
    while let newline = pending.firstIndex(of: 0x0a) {
      let line = Data(pending[..<newline])
      pending.removeSubrange(...newline)
      forward(line)
    }
  }

  private func forward(_ line: Data) {
    let text = String(decoding: line, as: UTF8.self)
    let normalized = text.lowercased()
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    let internalDiagnosticFragments = [
      "#aicode.",
      "aicode.serialization",
      "ane_validation_message",
      "anecompiler",
      "aneccompile(",
      "mlir mps to anec",
      "failed: ane i/o op",
      "incompatible element type for ane",
      "ane compilation failed",
      "full compile with ane as preferred device failed",
      "no ane hash for architecture",
      "gpu-only model or wrong target",
      "mpsgraph_disable_anec_module_validation",
    ]
    guard !internalDiagnosticFragments.contains(where: {
      normalized.contains($0)
    }) else {
      return
    }
    if normalized.contains("mioh-minimax-h3-native[")
      || normalized == "error:"
      || normalized == "warning:"
    {
      return
    }
    let diagnosticPunctuation = CharacterSet(charactersIn: "(){}[]<>,:#=)")
    if !trimmed.isEmpty,
      trimmed.unicodeScalars.allSatisfy({ diagnosticPunctuation.contains($0) })
    {
      return
    }
    var output = line
    output.append(0x0a)
    FileHandle.standardError.write(output)
  }
}

@available(macOS 27.0, *)
private final class H3DenoisePrepareWorker {
  private let process: Process
  private let diagnosticPipe = Pipe()
  private let diagnosticFilter = H3PrepareWorkerDiagnosticFilter()

  init(manifestURL: URL, descriptorURL: URL) {
    let process = Process()
    process.executableURL = URL(
      fileURLWithPath: CommandLine.arguments[0]
    ).standardizedFileURL
    process.arguments = [
      "prepare-part", "--manifest", manifestURL.standardizedFileURL.path,
      "--descriptor", descriptorURL.standardizedFileURL.path,
    ]
    var environment = ProcessInfo.processInfo.environment
    // This child consumes conditions cached by the parent and executes only
    // the BF16 DiT. Keep the private ANE-validation bypass out of the parent,
    // where Qwen and the VAE need normal specialization handling.
    environment["MPSGRAPH_DISABLE_ANEC_MODULE_VALIDATION"] = "1"
    process.environment = environment
    // MPSGraph emits this diagnostic on stdout on macOS 27 beta 8 (it used
    // stderr on earlier seeds). Route both streams through the same line
    // filter while preserving every other progress/error line.
    process.standardOutput = diagnosticPipe
    process.standardError = diagnosticPipe
    self.process = process
  }

  deinit {
    stop()
  }

  func prepare() async throws {
    diagnosticPipe.fileHandleForReading.readabilityHandler = {
      [diagnosticFilter] handle in
      diagnosticFilter.consume(handle.availableData)
    }
    let status = try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        process.terminationHandler = { completed in
          continuation.resume(returning: completed.terminationStatus)
        }
        do {
          try process.run()
        } catch {
          process.terminationHandler = nil
          continuation.resume(throwing: error)
        }
      }
    } onCancel: {
      if process.isRunning {
        process.terminate()
      }
    }
    diagnosticPipe.fileHandleForReading.readabilityHandler = nil
    diagnosticFilter.consume(
      diagnosticPipe.fileHandleForReading.readDataToEndOfFile()
    )
    diagnosticFilter.finish()
    guard status == 0 else {
      throw H3NativeError.inference(
        "DiT preparation worker exited with status \(status)"
      )
    }
  }

  func stop() {
    if process.isRunning {
      process.terminate()
    }
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
    if command == "prepare-part" {
      guard let descriptorPath = options["descriptor"] else { throw usage() }
      try await runPreparePart(
        manifestURL: manifestURL,
        descriptorURL: URL(fileURLWithPath: descriptorPath).standardizedFileURL
      )
      return
    }
    guard command == "plan" || command == "run" || command == "music-video"
    else { throw usage() }
    let job = try loadJob(options)
    if command == "music-video" {
      try await runMusicVideo(manifestURL: manifestURL, baseJob: job)
      return
    }
    let pipeline = try await H3NativePipeline(manifestURL: manifestURL, job: job)
    if command == "plan" {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
      print(String(data: try encoder.encode(await pipeline.plan()), encoding: .utf8)!)
    } else {
      try await runSingleVideo(manifestURL: manifestURL, job: job)
    }
  }

  private static func runPreparePart(
    manifestURL: URL,
    descriptorURL: URL
  ) async throws {
    let descriptor = try JSONDecoder().decode(
      H3PartPreparationDescriptor.self,
      from: Data(contentsOf: descriptorURL)
    )
    let continuationLatent: H3TemporalContinuationState?
    if let path = descriptor.continuationLatentPath {
      continuationLatent = try H3TemporalLatentStore.load(
        from: URL(fileURLWithPath: path).standardizedFileURL
      )
    } else {
      continuationLatent = nil
    }
    let reporter = H3ProgressReporter(
      progressBase: descriptor.progressBase,
      progressScale: descriptor.progressScale,
      messagePrefix: descriptor.messagePrefix
    )
    let pipeline = try await H3NativePipeline(
      manifestURL: manifestURL,
      job: descriptor.job,
      reporter: reporter,
      conditioningModeOverride: descriptor.conditioningMode,
      visualSourceDigestOverride: descriptor.visualSourceDigest,
      sourceDigestOverride: descriptor.sourceDigest,
      conditioningSourceDigestOverride: descriptor.conditioningSourceDigest,
      sourceImageDigestOverrides: descriptor.sourceImageDigests,
      continuationStateOverride: continuationLatent,
      temporalLatentOutputURL: descriptor.temporalLatentOutputPath.map {
        URL(fileURLWithPath: $0).standardizedFileURL
      },
      continuationFrameOutputURL: descriptor.continuationFrameOutputPath.map {
        URL(fileURLWithPath: $0).standardizedFileURL
      },
      reusableVisionBlockCount: descriptor.reusableVisionBlockCount,
      denoiserImageReferenceCount: descriptor.denoiserImageReferenceCount
    )
    try await pipeline.run(decodeOutput: false)
  }

  private static func runSingleVideo(
    manifestURL: URL,
    job: H3NativeJob
  ) async throws {
    let outputURL = URL(fileURLWithPath: job.output).standardizedFileURL
    let workURL = outputURL
      .deletingPathExtension()
      .appendingPathExtension("mioh-h3-work")
    try FileManager.default.createDirectory(
      at: workURL,
      withIntermediateDirectories: true
    )

    let conditioningReporter = H3ProgressReporter(
      progressBase: 0,
      progressScale: 0.48
    )
    let conditioningPipeline = try await H3NativePipeline(
      manifestURL: manifestURL,
      job: job,
      reporter: conditioningReporter,
      conditioningOnly: true
    )
    try await conditioningPipeline.run()

    let descriptor = conditioningPipeline.denoiserPreparationDescriptor(
      progressBase: 0,
      progressScale: 0.82
    )
    let descriptorURL = workURL.appendingPathComponent(".prepare-single-run.json")
    let descriptorEncoder = JSONEncoder()
    descriptorEncoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    try descriptorEncoder.encode(descriptor).write(
      to: descriptorURL,
      options: .atomic
    )
    let prepareWorker = H3DenoisePrepareWorker(
      manifestURL: manifestURL,
      descriptorURL: descriptorURL
    )
    try await prepareWorker.prepare()

    let renderReporter = H3ProgressReporter()
    let renderPipeline = try await H3NativePipeline(
      manifestURL: manifestURL,
      job: job,
      reporter: renderReporter
    )
    try await renderPipeline.run()
  }

  private static func runMusicVideo(
    manifestURL: URL,
    baseJob: H3NativeJob
  ) async throws {
    guard let audioPath = baseJob.audioInput, !audioPath.isEmpty,
      baseJob.input == nil,
      let imagePaths = baseJob.inputImages, !imagePaths.isEmpty
    else {
      throw H3NativeError.invalidJob(
        "music-video mode requires reference images and an external audio file"
      )
    }
    try baseJob.validate(conditioningMode: .ref2va)
    let audioURL = URL(fileURLWithPath: audioPath).standardizedFileURL
    let analysisReporter = H3ProgressReporter()
    analysisReporter.emit(
      "musicAnalysis", "started", 0,
      "Reading source duration and the selected audio range"
    )
    let sourceDuration = try await H3NativeMedia.probeDuration(audioURL)
    let availableDuration = sourceDuration - baseJob.resolvedAudioStartSeconds
    guard availableDuration >= 2 else {
      throw H3NativeError.invalidJob(
        "less than two seconds of music remain after audioStartSeconds"
      )
    }
    let isOfficial1080pProfile = baseJob.resolvedOutputWidth == 1920
      && baseJob.resolvedOutputHeight == 1080
    let flatPromptPlan = try H3FlatTimelinePrompt.parse(baseJob.prompt)
    let timeline: [H3MusicVideoSegment]
    analysisReporter.emit(
      "musicAnalysis", "running", 0,
      String(
        format: "Source %.3fs · analyzing %.3fs from %.3fs",
        locale: Locale(identifier: "en_US_POSIX"),
        sourceDuration,
        availableDuration,
        baseJob.resolvedAudioStartSeconds
      )
    )
    if let flatPromptPlan {
      let entries = flatPromptPlan.entries
      guard let finalEnd = entries.last?.endSeconds,
        abs(finalEnd - availableDuration) <= 1.0 / 24.0 + 1e-6
      else {
        throw H3NativeError.invalidJob(
          "flat timeline must end at the selected audio duration \(availableDuration)"
        )
      }
      timeline = entries.map {
        H3MusicVideoSegment(
          startSeconds: $0.startSeconds,
          durationSeconds: $0.endSeconds - $0.startSeconds,
          boundaryStrength: 0,
          normalizedEnergy: 0.5
        )
      }
      analysisReporter.emit(
        "musicAnalysis", "completed", 0,
        "Prepared \(entries.count) explicit flat timeline intervals"
      )
    } else if let manualCutPoints = baseJob.musicVideoCutPointsSeconds {
      analysisReporter.emit(
        "musicAnalysis", "started", 0,
        "Reading user-defined composition points and measuring shot energy"
      )
      let analysisSampleRate = 8_000
      let analysisAudio = try await H3NativeMedia.decodeReferenceAudio(
        url: audioURL,
        durationSeconds: availableDuration,
        startSeconds: baseJob.resolvedAudioStartSeconds,
        sampleRate: analysisSampleRate
      )
      timeline = try H3MusicVideoAnalyzer.timeline(
        audio: analysisAudio,
        sampleRate: analysisSampleRate,
        totalDuration: availableDuration,
        cutPoints: manualCutPoints,
        minimumDuration: 2
      )
      let cuts = timeline.dropLast().map {
        posixNumber($0.startSeconds + $0.durationSeconds)
      }.joined(separator: ", ")
      let cutDescription = cuts.isEmpty ? "no internal cuts" : cuts
      analysisReporter.emit(
        "musicAnalysis", "completed", 0,
        "Prepared \(timeline.count) user-defined compositions at: \(cutDescription)"
      )
    } else if isOfficial1080pProfile {
      timeline = H3MusicVideoAnalyzer.fixedTimeline(
        totalDuration: availableDuration,
        duration: 6
      )
      analysisReporter.emit(
        "musicAnalysis", "completed", 0,
        "Official 1080p uses fixed 6-second intervals; prepared \(timeline.count) intervals"
      )
    } else {
      analysisReporter.emit(
        "musicAnalysis", "started", 0,
        "Analyzing music energy, onsets, timbre changes, and phrase breaks"
      )
      let analysisSampleRate = 8_000
      let analysisAudio = try await H3NativeMedia.decodeReferenceAudio(
        url: audioURL,
        durationSeconds: availableDuration,
        startSeconds: baseJob.resolvedAudioStartSeconds,
        sampleRate: analysisSampleRate
      )
      analysisReporter.emit(
        "musicAnalysis", "running", 0,
        "Decoded \(analysisAudio.shape[2]) analysis samples at \(analysisSampleRate)Hz; measuring energy, onset, timbre, and quiet-break novelty"
      )
      timeline = try H3MusicVideoAnalyzer.timeline(
        audio: analysisAudio,
        sampleRate: analysisSampleRate,
        totalDuration: availableDuration,
        maximumDuration: 30,
        minimumDuration: 4,
        preferredDuration: 14
      )
      let cuts = timeline.dropLast().map {
        posixNumber($0.startSeconds + $0.durationSeconds)
      }.joined(separator: ", ")
      analysisReporter.emit(
        "musicAnalysis", "completed", 0,
        "Prepared \(timeline.count) composition intervals at musical cuts: \(cuts)"
      )
    }
    let continuationMode = baseJob.resolvedMusicVideoContinuationMode
    let continuationPreRollFrames: Int
    switch continuationMode {
    case .hybridAV, .latentPrefix:
      continuationPreRollFrames = H3VideoConditioning.partContinuationPixelFrames
    case .firstFrame, .firstAndProvidedLast, .firstAndGeneratedLast:
      // FL2VA reproduces its fixed First keyframe as target frame zero. Trim
      // that one duplicated boundary frame rather than an entire latent tile.
      continuationPreRollFrames = 1
    }
    let maximumGenerationDuration = isOfficial1080pProfile
      ? 6 : H3Geometry.audioConditioningSeconds
    let maximumGenerationFrames = Int(
      (maximumGenerationDuration * Double(H3Geometry.framesPerSecond))
        .rounded(.down)
    )
    let intervals: [H3MusicVideoInterval]
    let intervalPromptEntryIndices: [Int]?
    if let flatPromptPlan {
      var plannedIntervals: [H3MusicVideoInterval] = []
      var promptEntryIndices: [Int] = []
      for (entryIndex, entry) in flatPromptPlan.entries.enumerated() {
        let entryFrames = Int(
          ((entry.endSeconds - entry.startSeconds)
            * Double(H3Geometry.framesPerSecond)).rounded()
        )
        var payloads = [entryFrames]
        if entryIndex > 0, entry.transition == .cut,
          entryFrames + H3MusicVideoBoundary.cutPreRollFrames
            > maximumGenerationFrames,
          entryFrames >= 4 * H3Geometry.framesPerSecond
        {
          let minimumPayload = 2 * H3Geometry.framesPerSecond
          var first = min(
            maximumGenerationFrames
              - H3MusicVideoBoundary.cutPreRollFrames,
            entryFrames - minimumPayload
          )
          while first >= minimumPayload,
            (first + H3MusicVideoBoundary.cutPreRollFrames) % 17 != 5
          {
            first -= 1
          }
          while entryFrames - first < minimumPayload,
            first - 17 >= minimumPayload
          {
            first -= 17
          }
          if first >= minimumPayload,
            entryFrames - first >= minimumPayload,
            entryFrames - first + continuationPreRollFrames
              <= maximumGenerationFrames
          {
            payloads = [first, entryFrames - first]
          }
        }
        var relativeStartFrames = 0
        for partIndex in payloads.indices {
          let payloadFrames = payloads[partIndex]
          let transition = partIndex == 0 ? entry.transition : .continue
          let continuesToNext = partIndex + 1 < payloads.count
            || (entryIndex + 1 < flatPromptPlan.entries.count
              && flatPromptPlan.entries[entryIndex + 1].transition == .continue)
          let intervalIndex = plannedIntervals.count
          let preRollFrames = transition == .continue
            ? continuationPreRollFrames
            : H3MusicVideoBoundary.cutPreRollFrames(
              intervalIndex: intervalIndex,
              payloadFrames: payloadFrames,
              maximumFrames: maximumGenerationFrames,
              suppliesContinuation: continuesToNext
            )
          let generationFrames = payloadFrames + preRollFrames
          guard payloadFrames >= 2 * H3Geometry.framesPerSecond,
            Double(generationFrames) / Double(H3Geometry.framesPerSecond)
              <= maximumGenerationDuration + 1e-6
          else {
            throw H3NativeError.invalidJob(
              "flat timeline interval \(entryIndex + 1) is outside the supported generation duration"
            )
          }
          if continuesToNext,
            !H3Geometry.isAlignedFrameCount(generationFrames)
          {
            throw H3NativeError.invalidJob(
              "flat timeline interval \(entryIndex + 1) must end on a 5+17n frame boundary before continue"
            )
          }
          plannedIntervals.append(
            H3MusicVideoInterval(
              index: intervalIndex,
              transition: transition,
              continuesToNext: continuesToNext,
              startSeconds: entry.startSeconds
                + Double(relativeStartFrames)
                  / Double(H3Geometry.framesPerSecond),
              durationSeconds: Double(payloadFrames)
                / Double(H3Geometry.framesPerSecond),
              preRollSeconds: Double(preRollFrames)
                / Double(H3Geometry.framesPerSecond),
              boundaryStrength: 0,
              normalizedEnergy: 0.5
            )
          )
          promptEntryIndices.append(entryIndex)
          relativeStartFrames += payloadFrames
        }
      }
      intervals = plannedIntervals
      intervalPromptEntryIndices = promptEntryIndices
    } else {
      intervals = try H3MusicVideoAnalyzer.generationIntervals(
        timeline,
        maximumDuration: maximumGenerationDuration,
        continuationPreRollSeconds: Double(continuationPreRollFrames)
          / Double(H3Geometry.framesPerSecond),
        cutPreRollSeconds: Double(H3MusicVideoBoundary.cutPreRollFrames)
          / Double(H3Geometry.framesPerSecond)
      )
      intervalPromptEntryIndices = nil
    }
    let intervalCount = intervals.count
    for interval in intervals {
      analysisReporter.emit(
        "musicAnalysis", "running", 0,
        String(
          format: "Interval %d/%d · %.3f–%.3fs · %@",
          locale: Locale(identifier: "en_US_POSIX"),
          interval.index + 1,
          intervalCount,
          interval.startSeconds,
          interval.startSeconds + interval.durationSeconds,
          interval.transition.rawValue
        )
      )
    }
    analysisReporter.emit(
      "musicAnalysis", "completed", 0,
      "\(intervalCount) flat intervals are ready"
    )
    let outputURL = URL(fileURLWithPath: baseJob.output).standardizedFileURL
    let workURL = outputURL.deletingPathExtension()
      .appendingPathExtension("mioh-h3-work")
    let markerURL = workURL.appendingPathComponent(
      ".mioh-h3-music-video-work-v1"
    )
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: workURL.path) {
      let values = try workURL.resourceValues(forKeys: [
        .isDirectoryKey, .isSymbolicLinkKey,
      ])
      guard values.isDirectory == true, values.isSymbolicLink != true,
        isOwnedMusicVideoWorkDirectory(workURL, markerURL: markerURL)
      else {
        throw H3NativeError.invalidJob(
          "refusing to reuse an unowned music-video work directory: \(workURL.path)"
        )
      }
    } else {
      try fileManager.createDirectory(
        at: workURL,
        withIntermediateDirectories: true
      )
      try Data("mioh-h3-music-video-work-v1\n".utf8).write(
        to: markerURL,
        options: .atomic
      )
    }

    let identityImageURLs = imagePaths.map {
      URL(fileURLWithPath: $0).standardizedFileURL
    }
    let audioDigest = try H3StageCache.fileDigest(audioURL)
    // Qwen and reference/audio conditions run in this unmodified parent
    // process. Fixed identity-image vision features therefore remain reusable
    // in memory across cut intervals. Only the BF16 DiT runs in the short-lived worker
    // that carries the private ANE-validation switch.
    let visionFeatureMemoryCache = H3QwenVisionFeatureMemoryCache()
    let stableIdentityURLs = Array(
      identityImageURLs.prefix(H3Geometry.identityVisionBlocks)
    )
    let providedLastFrameDirectory: URL?
    if continuationMode == .firstAndProvidedLast {
      guard let path = baseJob.musicVideoLastFrameDirectory,
        !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      else {
        throw H3NativeError.invalidJob(
          "Codex Last mode requires musicVideoLastFrameDirectory"
        )
      }
      let directory = URL(fileURLWithPath: path).standardizedFileURL
      let values = try directory.resourceValues(forKeys: [
        .isDirectoryKey, .isSymbolicLinkKey,
      ])
      guard values.isDirectory == true, values.isSymbolicLink != true else {
        throw H3NativeError.invalidJob(
          "musicVideoLastFrameDirectory must be a real directory"
        )
      }
      providedLastFrameDirectory = directory
    } else {
      providedLastFrameDirectory = nil
    }
    let storyboardDirectory: URL?
    if let path = baseJob.musicVideoStoryboardDirectory,
      !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    {
      guard flatPromptPlan != nil else {
        throw H3NativeError.invalidJob(
          "storyboard_directory requires an explicit flat timeline prompt"
        )
      }
      let directory = URL(fileURLWithPath: path).standardizedFileURL
      let values = try directory.resourceValues(forKeys: [
        .isDirectoryKey, .isSymbolicLinkKey,
      ])
      guard values.isDirectory == true, values.isSymbolicLink != true else {
        throw H3NativeError.invalidJob(
          "musicVideoStoryboardDirectory must be a real directory"
        )
      }
      storyboardDirectory = directory
    } else {
      storyboardDirectory = nil
    }

    func storyboardFrameURL(entryIndex: Int) throws -> URL? {
      guard let storyboardDirectory else { return nil }
      let stem = String(format: "entry-%04d", entryIndex)
      for fileExtension in ["png", "jpg", "jpeg"] {
        let candidate = storyboardDirectory
          .appendingPathComponent(stem)
          .appendingPathExtension(fileExtension)
        if fileManager.fileExists(atPath: candidate.path) {
          return candidate
        }
      }
      throw H3NativeError.missingAsset(
        storyboardDirectory.appendingPathComponent(stem + ".png").path
      )
    }
    func makeIntervalJob(
      _ interval: H3MusicVideoInterval,
      inputImages: [URL],
      outputURL: URL,
      storyboardAnchored: Bool = false
    ) throws -> H3NativeJob {
      let frameRate = Double(H3Geometry.framesPerSecond)
      let preRollFrames = Int((interval.preRollSeconds * frameRate).rounded())
      let payloadFrames = Int((interval.durationSeconds * frameRate).rounded())
      let blendFrames = H3MusicVideoBoundary.blendFrames(
        transition: interval.transition,
        preRollFrames: preRollFrames
      )
      let movieTrimFrames = preRollFrames - blendFrames
      let movieFrameCount = payloadFrames + blendFrames
      let audioStart = baseJob.resolvedAudioStartSeconds
        + interval.startSeconds - interval.preRollSeconds
      let requestedDuration: Double
      if isOfficial1080pProfile {
        requestedDuration = 6
      } else {
        requestedDuration = max(
          2,
          interval.durationSeconds + interval.preRollSeconds
        )
      }
      let directive = musicVideoIntervalDirective(
        interval,
        audioStartSeconds: audioStart,
        durationSeconds: requestedDuration,
        continuationMode: continuationMode,
        storyboardAnchored: storyboardAnchored
      )
      let intervalPrompt: String
      if let flatPromptPlan {
        let entryIndex = intervalPromptEntryIndices?[interval.index]
          ?? interval.index
        intervalPrompt = try flatPromptPlan.compiledPrompt(
          entryIndex: entryIndex,
          directive: directive,
          entryBodyOverride: interval.transition == .cut
            ? H3FlatTimelinePrompt.singleTakeBody(
              flatPromptPlan.entries[entryIndex].body
            )
            : nil
        )
      } else {
        intervalPrompt = musicVideoIntervalPrompt(
          baseJob.prompt,
          directive: directive
        )
      }
    return H3NativeJob(
      input: nil,
      inputImages: inputImages.map(\.path),
      inputImageSubjects: baseJob.inputImageSubjects,
      referenceEditMode: nil,
      referenceEditTargetDescription: nil,
      referenceEditTargetIndex: nil,
      physicalReferenceMask: nil,
      output: outputURL.path,
        prompt: intervalPrompt,
        cacheDirectory: baseJob.cacheDirectory,
        width: baseJob.width,
        height: baseJob.height,
        outputWidth: baseJob.outputWidth,
        outputHeight: baseJob.outputHeight,
        audioInput: audioPath,
        audioStartSeconds: audioStart,
        durationSeconds: requestedDuration,
        seed: H3MusicVideoSeed.value(
          base: baseJob.seed,
          intervalIndex: interval.index
        ),
        backend: baseJob.backend,
        outputTrimStartSeconds: Double(movieTrimFrames) / frameRate,
        outputDurationSeconds: Double(movieFrameCount) / frameRate,
        preserveSourceAudioWhenDecoderIsUnavailable:
          baseJob.preserveSourceAudioWhenDecoderIsUnavailable,
        musicVideoContinuationMode: continuationMode
      )
    }
    let intervalPlans: [H3MusicVideoIntervalPlan] = try intervals.map { interval in
      let intervalURL = workURL.appendingPathComponent(
        String(format: "interval-%04d.mp4", interval.index)
      )
      let continuationFrameURL = intervalURL.deletingPathExtension()
        .appendingPathExtension("last.png")
      let continuationLatentURL = intervalURL.deletingPathExtension()
        .appendingPathExtension("latent-prefix.plist")
      let audioStart = baseJob.resolvedAudioStartSeconds
        + interval.startSeconds - interval.preRollSeconds
      let previousFrameURL: URL?
      if interval.transition == .continue, interval.index > 0 {
        previousFrameURL = workURL.appendingPathComponent(
          String(format: "interval-%04d.last.png", interval.index - 1)
        )
      } else {
        previousFrameURL = nil
      }
      let previousLatentURL: URL?
      if interval.transition == .continue, interval.index > 0 {
        previousLatentURL = workURL.appendingPathComponent(
          String(format: "interval-%04d.latent-prefix.plist", interval.index - 1)
        )
      } else {
        previousLatentURL = nil
      }
      let providedLastFrameURL = providedLastFrameDirectory.map {
        $0.appendingPathComponent(
          String(format: "interval-%04d-last.png", interval.index)
        )
      }
      let promptEntryIndex = intervalPromptEntryIndices?[interval.index]
        ?? interval.index
      let storyboardFrame = interval.transition == .cut
        ? try storyboardFrameURL(entryIndex: promptEntryIndex)
        : nil
      var cutImages = stableIdentityURLs
      if let storyboardFrame {
        // Keep face-only automation labels stable: Picture 1...N remain the
        // identity crops, and the interval composition anchor is the final
        // visual block. Reserve one of the eight Qwen vision slots for it.
        cutImages = Array(
          stableIdentityURLs.prefix(H3Geometry.identityVisionBlocks - 1)
        )
        cutImages.append(storyboardFrame)
      }
      let initialImages = interval.transition == .cut
        ? cutImages
        : [previousFrameURL!]
      return H3MusicVideoIntervalPlan(
        interval: interval,
        outputURL: intervalURL,
        continuationFrameURL: continuationFrameURL,
        continuationLatentURL: continuationLatentURL,
        previousFrameURL: previousFrameURL,
        previousLatentURL: previousLatentURL,
        providedLastFrameURL: providedLastFrameURL,
        suppliesContinuation: interval.continuesToNext,
        // The interval storyboard remains visible to Qwen, but is not an
        // omni-reference for Ref2VA's DiT. Passing it to the DiT makes the
        // model progressively replay the storyboard pixels near the end of
        // the interval instead of merely following its composition. The
        // preceding entries are the face-identity references.
        denoiserIdentityReferenceCount: interval.transition == .cut
          ? cutImages.count - (storyboardFrame == nil ? 0 : 1)
          : 0,
        audioStart: audioStart,
        job: try makeIntervalJob(
          interval,
          inputImages: initialImages,
          outputURL: intervalURL,
          storyboardAnchored: storyboardFrame != nil
        )
      )
    }
    if continuationMode == .firstAndProvidedLast {
      let missingLastFrames = intervalPlans.compactMap { planned -> String? in
        guard planned.interval.transition == .continue,
          let url = planned.providedLastFrameURL,
          !FileManager.default.fileExists(atPath: url.path)
        else {
          return nil
        }
        return url.path
      }
      if !missingLastFrames.isEmpty {
        throw H3NativeError.missingAsset(
          "Codex Last frames:\n" + missingLastFrames.joined(separator: "\n")
        )
      }
    }
    let conditioningProgressShare = 0.12
    let cutPlans = intervalPlans.filter { $0.interval.transition == .cut }
    analysisReporter.emit(
      "conditioning", "started", 0,
      storyboardDirectory == nil
        ? "Preparing shared identity conditions for \(cutPlans.count) cut intervals"
        : "Preparing storyboard and identity conditions for \(cutPlans.count) cut intervals"
    )
    for (index, planned) in cutPlans.enumerated() {
      try Task.checkCancellation()
      let cutImageURLs = (planned.job.inputImages ?? []).map {
        URL(fileURLWithPath: $0).standardizedFileURL
      }
      let cutImageDigests = try cutImageURLs.map(H3StageCache.fileDigest)
      let visualDigest = Data(
        H3StageCache.key(parts: cutImageDigests).utf8
      )
      let conditioningSourceDigest = Data(
        H3StageCache.key(parts: [
          visualDigest,
          audioDigest,
          Data("audio-start:\(planned.audioStart)".utf8),
        ]).utf8
      )
      let reporter = H3ProgressReporter(
        progressBase: conditioningProgressShare * Double(index)
          / Double(max(1, cutPlans.count)),
        progressScale: conditioningProgressShare
          / Double(max(1, cutPlans.count)),
        messagePrefix: "Cut condition \(index + 1)/\(cutPlans.count) · "
      )
      let pipeline = try await H3NativePipeline(
        manifestURL: manifestURL,
        job: planned.job,
        reporter: reporter,
        conditioningModeOverride: .ref2va,
        visualSourceDigestOverride: visualDigest,
        sourceDigestOverride: conditioningSourceDigest,
        conditioningSourceDigestOverride: conditioningSourceDigest,
        sourceImageDigestOverrides: cutImageDigests,
        visionFeatureMemoryCache: visionFeatureMemoryCache,
        reusableVisionBlockCount: cutImageURLs.count,
        denoiserImageReferenceCount:
          planned.denoiserIdentityReferenceCount,
        conditioningOnly: true
      )
      try await pipeline.run()
    }
    analysisReporter.emit(
      "conditioning", "completed", conditioningProgressShare,
      continuationMode == .hybridAV
        ? "Cut conditions are cached; Continuum AV prefixes and H3-Extend history carry continue intervals"
        : continuationMode == .latentPrefix
          ? "Cut conditions are cached; exact latent prefixes carry continue intervals"
          : "Cut conditions are cached; FL2VA conditions will be prepared for continue intervals"
    )
    let generationProgressBase = conditioningProgressShare
    let generationProgressScale = 0.96 - conditioningProgressShare
    var intervalURLs: [URL] = []
    intervalURLs.reserveCapacity(intervalCount)
    func runInterval(
      job: H3NativeJob,
      conditioningMode: H3ConditioningMode,
      continuationLatentURL: URL? = nil,
      temporalLatentOutputURL: URL? = nil,
      continuationFrameOutputURL: URL?,
      globalIndex: Int,
      progressFraction: ClosedRange<Double>,
      labelSuffix: String = "",
      denoiserImageReferenceCountOverride: Int? = nil
    ) async throws {
      let imageURLs = (job.inputImages ?? []).map {
        URL(fileURLWithPath: $0).standardizedFileURL
      }
      let imageDigests = try imageURLs.map(H3StageCache.fileDigest)
      let continuationLatent: H3TemporalContinuationState?
      if let continuationLatentURL {
        guard FileManager.default.fileExists(atPath: continuationLatentURL.path)
        else {
          throw H3NativeError.missingAsset(continuationLatentURL.path)
        }
        continuationLatent = try H3TemporalLatentStore.load(
          from: continuationLatentURL
        )
      } else {
        continuationLatent = nil
      }
      let visualDigest = Data(
        H3StageCache.key(parts: imageDigests).utf8
      )
      let audioStart = job.resolvedAudioStartSeconds
      let conditioningSourceDigest = Data(
        H3StageCache.key(parts: [
          visualDigest,
          audioDigest,
          Data("audio-start:\(audioStart)".utf8),
        ]).utf8
      )
      let continuationDigestParts = continuationLatent.map {
        [$0.video.bytes, $0.audio?.bytes ?? Data()]
      } ?? []
      let sourceDigest = Data(
        H3StageCache.key(
          parts: [conditioningSourceDigest] + continuationDigestParts
        ).utf8
      )
      let interval = intervals[globalIndex]
      let denoiserImageReferenceCount = min(
        max(
          0,
          denoiserImageReferenceCountOverride ?? imageURLs.count
        ),
        imageURLs.count
      )
      let usesDirectCutIdentity = conditioningMode == .ref2va
        && interval.transition == .cut
        && denoiserImageReferenceCount > 0
      let intervalURL = URL(fileURLWithPath: job.output).standardizedFileURL
      let signature = H3StageCache.key(parts: [
        Data(
          (usesDirectCutIdentity
            ? "music-video-flat-v15-continuum-extend-hybrid-cut-qwen-storyboard-face-ref2va-layout-v3"
            : "music-video-flat-v15-continuum-extend-hybrid").utf8
        ),
        Data(conditioningMode.rawValue.utf8),
        try JSONEncoder.h3Stable.encode(job),
        sourceDigest,
      ])
      let hasRequiredFrame = continuationFrameOutputURL.map {
        FileManager.default.fileExists(atPath: $0.path)
      } ?? true
      let hasRequiredLatent = temporalLatentOutputURL.map {
        FileManager.default.fileExists(atPath: $0.path)
      } ?? true
      if hasRequiredFrame, hasRequiredLatent,
        isUsableCompletedShot(intervalURL, expectedSignature: signature)
      {
        return
      }
      let baseProgress = generationProgressBase
        + generationProgressScale * progressFraction.lowerBound
      let scale = generationProgressScale
        * (progressFraction.upperBound - progressFraction.lowerBound)
      let messagePrefix = "Interval \(interval.index + 1)/\(intervalCount) · \(interval.transition.rawValue)\(labelSuffix) · "
      // Qwen/reference/audio conditioning always runs in the parent. The
      // prepare-part child can therefore load tensor caches and execute only
      // DiT, keeping its private MPSGraph switch away from Qwen and the VAE.
      let conditioningReporter = H3ProgressReporter(
        progressBase: baseProgress,
        progressScale: scale * 0.15,
        messagePrefix: messagePrefix
      )
      let conditioningPipeline = try await H3NativePipeline(
        manifestURL: manifestURL,
        job: job,
        reporter: conditioningReporter,
        conditioningModeOverride: conditioningMode,
        visualSourceDigestOverride: visualDigest,
        sourceDigestOverride: sourceDigest,
        conditioningSourceDigestOverride: conditioningSourceDigest,
        sourceImageDigestOverrides: imageDigests,
        continuationStateOverride: continuationLatent,
        visionFeatureMemoryCache: visionFeatureMemoryCache,
        reusableVisionBlockCount: conditioningMode == .ref2va
          ? imageURLs.count : 0,
        denoiserImageReferenceCount: denoiserImageReferenceCount,
        conditioningOnly: true
      )
      try await conditioningPipeline.run()
      let descriptor = H3PartPreparationDescriptor(
        job: job,
        conditioningMode: conditioningMode,
        visualSourceDigest: visualDigest,
        sourceDigest: sourceDigest,
        conditioningSourceDigest: conditioningSourceDigest,
        sourceImageDigests: imageDigests,
        continuationLatentPath: continuationLatentURL?.path,
        temporalLatentOutputPath: temporalLatentOutputURL?.path,
        continuationFrameOutputPath: continuationFrameOutputURL?.path,
        reusableVisionBlockCount: conditioningMode == .ref2va
          ? imageURLs.count : 0,
        denoiserImageReferenceCount: denoiserImageReferenceCount,
        progressBase: baseProgress,
        progressScale: scale,
        messagePrefix: messagePrefix
      )
      let descriptorURL = workURL.appendingPathComponent(
        ".prepare-\(intervalURL.deletingPathExtension().lastPathComponent).json"
      )
      let descriptorEncoder = JSONEncoder()
      descriptorEncoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
      try descriptorEncoder.encode(descriptor).write(
        to: descriptorURL,
        options: .atomic
      )
      let prepareWorker = H3DenoisePrepareWorker(
        manifestURL: manifestURL,
        descriptorURL: descriptorURL
      )
      try await prepareWorker.prepare()
      let reporter = H3ProgressReporter(
        progressBase: baseProgress,
        progressScale: scale,
        messagePrefix: messagePrefix
      )
      let pipeline = try await H3NativePipeline(
        manifestURL: manifestURL,
        job: job,
        reporter: reporter,
        conditioningModeOverride: conditioningMode,
        visualSourceDigestOverride: visualDigest,
        sourceDigestOverride: sourceDigest,
        conditioningSourceDigestOverride: conditioningSourceDigest,
        sourceImageDigestOverrides: imageDigests,
        continuationStateOverride: continuationLatent,
        continuationFrameOutputURL: continuationFrameOutputURL,
        reusableVisionBlockCount: conditioningMode == .ref2va
          ? imageURLs.count : 0,
        denoiserImageReferenceCount: denoiserImageReferenceCount
      )
      try await pipeline.run()
      try Data((signature + "\n").utf8).write(
        to: shotSignatureURL(intervalURL),
        options: .atomic
      )
    }
    for globalIndex in intervals.indices {
      try Task.checkCancellation()
      let planned = intervalPlans[globalIndex]
      let interval = planned.interval
      intervalURLs.append(planned.outputURL)
      let progressStart = Double(globalIndex) / Double(intervalCount)
      let progressEnd = Double(globalIndex + 1) / Double(intervalCount)
      let outputFrameURL = planned.suppliesContinuation
        && continuationMode != .latentPrefix
        ? planned.continuationFrameURL : nil
      let outputLatentURL = planned.suppliesContinuation
        && (continuationMode == .hybridAV
          || continuationMode == .latentPrefix)
        ? planned.continuationLatentURL : nil
      if interval.transition == .cut {
        try await runInterval(
          job: planned.job,
          conditioningMode: .ref2va,
          temporalLatentOutputURL: outputLatentURL,
          continuationFrameOutputURL: outputFrameURL,
          globalIndex: globalIndex,
          progressFraction: progressStart...progressEnd,
          denoiserImageReferenceCountOverride:
            planned.denoiserIdentityReferenceCount
        )
      } else {
        switch continuationMode {
        case .hybridAV, .latentPrefix:
          guard let previousLatentURL = planned.previousLatentURL else {
            throw H3NativeError.missingAsset(
              "preceding interval latent prefix"
            )
          }
          let job = try makeIntervalJob(
            interval,
            inputImages: [],
            outputURL: planned.outputURL
          )
          try await runInterval(
            job: job,
            conditioningMode: .ref2va,
            continuationLatentURL: previousLatentURL,
            temporalLatentOutputURL: outputLatentURL,
            continuationFrameOutputURL: outputFrameURL,
            globalIndex: globalIndex,
            progressFraction: progressStart...progressEnd
          )
        case .firstFrame:
          guard let previousFrameURL = planned.previousFrameURL,
            FileManager.default.fileExists(atPath: previousFrameURL.path)
          else {
            throw H3NativeError.missingAsset(
              planned.previousFrameURL?.path ?? "preceding interval final frame"
            )
          }
          let job = try makeIntervalJob(
            interval,
            inputImages: [previousFrameURL],
            outputURL: planned.outputURL
          )
          try await runInterval(
            job: job,
            conditioningMode: .fl2va,
            continuationFrameOutputURL: outputFrameURL,
            globalIndex: globalIndex,
            progressFraction: progressStart...progressEnd
          )
        case .firstAndProvidedLast:
          guard let previousFrameURL = planned.previousFrameURL,
            FileManager.default.fileExists(atPath: previousFrameURL.path)
          else {
            throw H3NativeError.missingAsset(
              planned.previousFrameURL?.path ?? "preceding interval final frame"
            )
          }
          guard let lastFrameURL = planned.providedLastFrameURL,
            FileManager.default.fileExists(atPath: lastFrameURL.path)
          else {
            throw H3NativeError.missingAsset(
              planned.providedLastFrameURL?.path ?? "Codex Last frame"
            )
          }
          let job = try makeIntervalJob(
            interval,
            inputImages: [previousFrameURL, lastFrameURL],
            outputURL: planned.outputURL
          )
          try await runInterval(
            job: job,
            conditioningMode: .fl2va,
            continuationFrameOutputURL: outputFrameURL,
            globalIndex: globalIndex,
            progressFraction: progressStart...progressEnd
          )
        case .firstAndGeneratedLast:
          guard let previousFrameURL = planned.previousFrameURL,
            FileManager.default.fileExists(atPath: previousFrameURL.path)
          else {
            throw H3NativeError.missingAsset(
              planned.previousFrameURL?.path ?? "preceding interval final frame"
            )
          }
          let draftURL = planned.outputURL.deletingPathExtension()
            .appendingPathExtension("draft.mp4")
          let draftLastURL = planned.outputURL.deletingPathExtension()
            .appendingPathExtension("draft-last.png")
          let draftJob = try makeIntervalJob(
            interval,
            inputImages: [previousFrameURL],
            outputURL: draftURL
          )
          let middle = progressStart + (progressEnd - progressStart) * 0.5
          try await runInterval(
            job: draftJob,
            conditioningMode: .fl2va,
            continuationFrameOutputURL: draftLastURL,
            globalIndex: globalIndex,
            progressFraction: progressStart...middle,
            labelSuffix: " · Last作成"
          )
          let finalJob = try makeIntervalJob(
            interval,
            inputImages: [previousFrameURL, draftLastURL],
            outputURL: planned.outputURL
          )
          try await runInterval(
            job: finalJob,
            conditioningMode: .fl2va,
            continuationFrameOutputURL: outputFrameURL,
            globalIndex: globalIndex,
            progressFraction: middle...progressEnd,
            labelSuffix: " · First＋Last"
          )
        }
      }
    }
    let reporter = H3ProgressReporter(progressBase: 0.96, progressScale: 0.04)
    reporter.emit(
      "musicVideo", "started", 0,
      "Assembling \(intervalCount) flat intervals with the master music track"
    )
    try assembleMusicVideo(
      intervalURLs: intervalURLs,
      intervals: intervals,
      audioURL: audioURL,
      audioStartSeconds: baseJob.resolvedAudioStartSeconds,
      durationSeconds: availableDuration,
      outputWidth: baseJob.resolvedOutputWidth,
      outputHeight: baseJob.resolvedOutputHeight,
      outputURL: outputURL,
      workURL: workURL
    )
    reporter.emit("musicVideo", "completed", 1, outputURL.path)
    // Keep the owned work directory after completion. Its interval movies,
    // exported boundary frames, and latent prefixes make resume and boundary
    // diagnosis possible without rebuilding the complete music video.
  }

  private static func isUsableCompletedShot(
    _ url: URL,
    expectedSignature: String
  ) -> Bool {
    guard let values = try? url.resourceValues(forKeys: [
      .isRegularFileKey, .fileSizeKey,
    ]),
      values.isRegularFile == true,
      (values.fileSize ?? 0) > 1_024,
      let signature = try? String(
        contentsOf: shotSignatureURL(url),
        encoding: .utf8
      )
    else { return false }
    return signature == expectedSignature + "\n"
  }

  private static func shotSignatureURL(_ shotURL: URL) -> URL {
    shotURL.deletingPathExtension().appendingPathExtension("signature")
  }

  private static func bundledFFmpegURL() throws -> URL {
    let executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
      .standardizedFileURL
    let ffmpegURL = executableURL.deletingLastPathComponent()
      .appendingPathComponent("ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpegURL.path) else {
      throw H3NativeError.missingAsset(ffmpegURL.path)
    }
    return ffmpegURL
  }

  private static func isOwnedMusicVideoWorkDirectory(
    _ workURL: URL,
    markerURL: URL
  ) -> Bool {
    guard workURL.lastPathComponent.hasSuffix(".mioh-h3-work"),
      let marker = try? String(contentsOf: markerURL, encoding: .utf8)
    else { return false }
    return marker == "mioh-h3-music-video-work-v1\n"
  }

  private static func assembleMusicVideo(
    intervalURLs: [URL],
    intervals: [H3MusicVideoInterval],
    audioURL: URL,
    audioStartSeconds: Double,
    durationSeconds: Double,
    outputWidth: Int,
    outputHeight: Int,
    outputURL: URL,
    workURL: URL
  ) throws {
    guard !intervalURLs.isEmpty, intervalURLs.count == intervals.count else {
      throw H3NativeError.media("music video has no completed intervals")
    }
    let listURL = workURL.appendingPathComponent("intervals.concat.txt")
    let list = intervalURLs.map { url -> String in
      let escaped = url.path.replacingOccurrences(of: "'", with: "'\\''")
      return "file '\(escaped)'"
    }.joined(separator: "\n") + "\n"
    try Data(list.utf8).write(to: listURL, options: .atomic)
    let ffmpegURL = try bundledFFmpegURL()
    try FileManager.default.createDirectory(
      at: outputURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let process = Process()
    process.executableURL = ffmpegURL
    let frameRate = H3Geometry.framesPerSecond
    let overlapFrames = intervals.map { interval in
      H3MusicVideoBoundary.blendFrames(
        transition: interval.transition,
        preRollFrames: Int(
          (interval.preRollSeconds * Double(frameRate)).rounded()
        )
      )
    }
    if overlapFrames.allSatisfy({ $0 == 0 }) {
      process.arguments = [
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", listURL.path,
        "-ss", Self.posixNumber(audioStartSeconds), "-i", audioURL.path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-t", Self.posixNumber(durationSeconds),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
        "-movflags", "+faststart", outputURL.path,
      ]
    } else {
      var arguments = [
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
      ]
      for url in intervalURLs { arguments += ["-i", url.path] }
      arguments += [
        "-ss", Self.posixNumber(audioStartSeconds), "-i", audioURL.path,
      ]
      var filters: [String] = intervalURLs.indices.map { index in
        "[\(index):v:0]fps=\(frameRate),format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[v\(index)]"
      }
      var chain = "v0"
      var timelineFrames = Int(
        (intervals[0].durationSeconds * Double(frameRate)).rounded()
      )
      for index in 1..<intervals.count {
        let next = "assembled\(index)"
        let blendFrames = overlapFrames[index]
        if blendFrames > 0 {
          let duration = Double(blendFrames) / Double(frameRate)
          let offset = Double(timelineFrames - blendFrames)
            / Double(frameRate)
          filters.append(
            "[\(chain)][v\(index)]xfade=transition=fade:duration=\(posixNumber(duration)):offset=\(posixNumber(offset))[\(next)]"
          )
        } else {
          filters.append(
            "[\(chain)][v\(index)]concat=n=2:v=1:a=0[\(next)]"
          )
        }
        chain = next
        timelineFrames += Int(
          (intervals[index].durationSeconds * Double(frameRate)).rounded()
        )
      }
      filters.append(
        "[\(chain)]trim=duration=\(posixNumber(durationSeconds)),setpts=PTS-STARTPTS[vout]"
      )
      let audioInputIndex = intervalURLs.count
      let averageBitRate = max(9_000_000, outputWidth * outputHeight * 12)
      arguments += [
        "-filter_complex", filters.joined(separator: ";"),
        "-map", "[vout]", "-map", "\(audioInputIndex):a:0?",
        "-t", posixNumber(durationSeconds),
        "-r", String(frameRate), "-fps_mode", "cfr",
        "-c:v", "hevc_videotoolbox", "-profile:v", "main",
        "-b:v", String(averageBitRate),
        "-maxrate", String(averageBitRate * 3 / 2),
        "-bufsize", String(averageBitRate * 2),
        "-tag:v", "hvc1",
        "-c:a", "aac", "-b:a", "256k",
        "-movflags", "+faststart", outputURL.path,
      ]
      process.arguments = arguments
    }
    let errorPipe = Pipe()
    process.standardOutput = FileHandle.nullDevice
    process.standardError = errorPipe
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
      let message = String(
        decoding: errorPipe.fileHandleForReading.readDataToEndOfFile(),
        as: UTF8.self
      ).trimmingCharacters(in: .whitespacesAndNewlines)
      throw H3NativeError.media(
        message.isEmpty ? "ffmpeg music-video assembly failed" : message
      )
    }
  }

  private static func posixNumber(_ value: Double) -> String {
    String(
      format: "%.6f",
      locale: Locale(identifier: "en_US_POSIX"),
      value
    )
  }

  private static func musicVideoIntervalDirective(
    _ interval: H3MusicVideoInterval,
    audioStartSeconds: Double,
    durationSeconds: Double,
    continuationMode: H3MusicVideoContinuationMode,
    storyboardAnchored: Bool = false
  ) -> String {
    let start = posixNumber(audioStartSeconds)
    let end = posixNumber(audioStartSeconds + durationSeconds)
    let transitionDirective: String
    if interval.transition == .continue {
      let carriedState: String
      switch continuationMode {
      case .hybridAV:
        carriedState = "A protected audiovisual prefix and its immediately older motion history are supplied as the exact preceding physical state."
      case .latentPrefix:
        carriedState = "A fixed latent prefix is supplied as the exact preceding physical state."
      case .firstFrame:
        carriedState = "A fixed first keyframe is supplied as the exact preceding physical state."
      case .firstAndProvidedLast, .firstAndGeneratedLast:
        carriedState = "Fixed first and last keyframes define the exact boundary states and one physically coherent motion path between them."
      }
      transitionDirective = """
        CONTINUE FORWARD. \(carriedState) Treat it only as history immediately before this time range. Begin with the next motion phase: preserve pose velocity, gaze, facial emotion, fabric motion, camera velocity, lens, lighting, and background geometry. Do not restart, replay, re-establish, freeze, dissolve, or return to an earlier position. Generate only what physically follows.
        """
    } else {
      transitionDirective = """
        SINGLE UNINTERRUPTED TAKE. The first output frame already occupies the location, composition, action phase, and fixed lighting described below. Maintain one continuous camera and one continuous physical setting for this entire interval; do not reset, re-establish, or transform into another composition or environment. No image or motion from an earlier time range is authoritative.
        """
    }
    let storyboardDirective = storyboardAnchored
      ? "The final supplied visual reference is the interval-specific photoreal storyboard anchor. Its camera height, lens, framing, subject blocking, studio geometry, color palette, lighting, and opening action are authoritative. Begin directly from that photographed composition; do not reproduce a contact sheet, border, number, timecode, drawing, paper texture, or storyboard annotation."
      : ""
    return """
      PRIORITY ABSOLUTE-TIME DIRECTIVE: Render only source-audio time \(start)-\(end) seconds. \(transitionDirective) \(storyboardDirective) The concrete scene description below is authoritative for this time range. A time-of-day phrase defines stable lighting, not a time lapse. Show exactly one visible instance of each referenced subject. Never superimpose, overlap, ghost, double-expose, split-screen, or duplicate the subject. Preserve precise vocal timing from the supplied audio, and leave a clean moving boundary when forward continuation follows.
      """
  }

  private static func musicVideoIntervalPrompt(
    _ prompt: String,
    directive: String
  ) -> String {
    var lines = prompt.components(separatedBy: .newlines)
    if let detailsIndex = lines.firstIndex(where: {
      $0.trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased() == "detailed_description:"
    }) {
      lines.insert(directive, at: detailsIndex + 1)
      return lines.joined(separator: "\n")
    }
    return directive + "\n\n" + prompt
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
    let inputImageSubjects: [Int]?
    if let encoded = options["input-image-subjects-json"] {
      inputImageSubjects = try JSONDecoder().decode(
        [Int].self,
        from: Data(encoded.utf8)
      )
    } else {
      inputImageSubjects = nil
    }
    let musicVideoCutPoints: [Double]?
    if let encoded = options["music-video-cuts-json"] {
      musicVideoCutPoints = try JSONDecoder().decode(
        [Double].self,
        from: Data(encoded.utf8)
      )
    } else {
      musicVideoCutPoints = nil
    }
    let continuationMode: H3MusicVideoContinuationMode?
    if let rawMode = options["music-video-continuation"] {
      let parsed = rawMode == "first-last"
        ? H3MusicVideoContinuationMode.firstAndProvidedLast
        : H3MusicVideoContinuationMode(rawValue: rawMode)
      guard let parsed else {
        throw H3NativeError.invalidArguments(
          "music-video-continuation must be hybrid-av, latent-prefix, first, first-last-provided, or first-last-generated"
        )
      }
      continuationMode = parsed
    } else {
      continuationMode = nil
    }
    let referenceEditMode: H3ReferenceEditMode?
    if let rawMode = options["reference-edit-mode"] {
      referenceEditMode = H3ReferenceEditMode(rawValue: rawMode)
      guard referenceEditMode != nil else {
        throw H3NativeError.invalidArguments(
          "reference-edit-mode must be none, face-swap, or body-swap"
        )
      }
    } else {
      referenceEditMode = nil
    }
    return H3NativeJob(
      input: options["input"],
      inputImages: inputImages,
      inputImageSubjects: inputImageSubjects,
      referenceEditMode: referenceEditMode,
      referenceEditTargetDescription: options["reference-edit-target"],
      referenceEditTargetIndex: options["reference-edit-target-index"]
        .flatMap(Int.init),
      physicalReferenceMask:
        options["physical-reference-mask"].map { $0 != "0" && $0 != "false" },
      output: output,
      prompt: prompt,
      cacheDirectory: cache,
      width: Int(options["width"] ?? "864") ?? 864,
      height: Int(options["height"] ?? "480") ?? 480,
      outputWidth: options["output-width"].flatMap(Int.init),
      outputHeight: options["output-height"].flatMap(Int.init),
      audioInput: options["audio-input"],
      audioStartSeconds: options["audio-start"].flatMap(Double.init),
      durationSeconds: Double(options["duration"] ?? "10") ?? 10,
      seed: UInt64(options["seed"] ?? "261662374822964") ?? 261662374822964,
      backend: options["backend"].flatMap(H3BackendKind.init(rawValue:)),
      outputTrimStartSeconds: options["output-trim-start"].flatMap(Double.init),
      outputDurationSeconds: options["output-duration"].flatMap(Double.init),
      preserveSourceAudioWhenDecoderIsUnavailable: false,
      musicVideoCutPointsSeconds: musicVideoCutPoints,
      musicVideoContinuationMode: continuationMode,
      musicVideoLastFrameDirectory:
        options["music-video-last-frame-directory"],
      musicVideoStoryboardDirectory:
        options["music-video-storyboard-directory"]
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
      "usage: mioh-minimax-h3-native <validate|plan|run|music-video> --manifest <manifest.json> "
        + "[--job <job.json> | (--input <video>)? (--input-images-json <json>)? "
        + "[--input-image-subjects-json <json>] "
        + "[--reference-edit-mode <none|face-swap|body-swap>] "
        + "[--reference-edit-target <description>] "
        + "[--reference-edit-target-index <index>] "
        + "[--physical-reference-mask <0|1>] "
        + "--output <mp4> --prompt <text> "
        + "--cache <dir> --backend <coreai|coreml> "
        + "--width <multiple-of-32> --height <multiple-of-32> "
        + "[--output-width <pixels> --output-height <pixels>] "
        + "[--audio-input <music> --audio-start <seconds>] "
        + "[--music-video-cuts-json <seconds-json>] "
        + "[--music-video-continuation <hybrid-av|latent-prefix|first|first-last-provided|first-last-generated>] "
        + "[--music-video-last-frame-directory <directory>] "
        + "[--music-video-storyboard-directory <directory>] "
        + "--duration <2...15> --seed N]"
    )
  }
}
