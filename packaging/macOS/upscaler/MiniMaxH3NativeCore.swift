import CryptoKit
import Foundation

enum H3NativeError: LocalizedError {
  case invalidArguments(String)
  case invalidManifest(String)
  case invalidJob(String)
  case invalidTensor(String)
  case missingAsset(String)
  case missingStage(String)
  case missingTensor(String)
  case unsupported(String)
  case inference(String)
  case media(String)
  case cache(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments(let message): return message
    case .invalidManifest(let message): return "invalid H3 manifest: \(message)"
    case .invalidJob(let message): return "invalid H3 job: \(message)"
    case .invalidTensor(let message): return "invalid H3 tensor: \(message)"
    case .missingAsset(let message): return "missing H3 asset: \(message)"
    case .missingStage(let message): return "missing H3 stage: \(message)"
    case .missingTensor(let message): return "missing H3 tensor: \(message)"
    case .unsupported(let message): return "unsupported H3 operation: \(message)"
    case .inference(let message): return "H3 inference failed: \(message)"
    case .media(let message): return "H3 media failed: \(message)"
    case .cache(let message): return "H3 cache failed: \(message)"
    }
  }
}

enum H3BackendKind: String, Codable, Sendable {
  case coreAI = "coreai"
  case coreML = "coreml"
}

enum H3AudioConditioningMode: String, Codable, Sendable {
  case backgroundMusic = "background-music"
  case lipSync = "lip-sync"
}

enum H3ScalarType: String, Codable, Sendable {
  case bfloat16
  case float16
  case float32
  case int32
  case int64

  var byteCount: Int {
    switch self {
    case .bfloat16: return MemoryLayout<UInt16>.stride
    case .float16: return MemoryLayout<Float16>.stride
    case .float32: return MemoryLayout<Float>.stride
    case .int32: return MemoryLayout<Int32>.stride
    case .int64: return MemoryLayout<Int64>.stride
    }
  }
}

struct H3Tensor: Sendable {
  let shape: [Int]
  let scalarType: H3ScalarType
  var bytes: Data

  init(shape: [Int], scalarType: H3ScalarType, bytes: Data) throws {
    guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
      throw H3NativeError.invalidTensor("shape must contain positive dimensions")
    }
    let count = try Self.checkedElementCount(shape)
    let (expectedBytes, overflow) = count.multipliedReportingOverflow(
      by: scalarType.byteCount
    )
    guard !overflow, bytes.count == expectedBytes else {
      throw H3NativeError.invalidTensor(
        "\(shape) \(scalarType.rawValue) needs \(expectedBytes) bytes, got \(bytes.count)"
      )
    }
    self.shape = shape
    self.scalarType = scalarType
    self.bytes = bytes
  }

  init(float32 values: [Float], shape: [Int]) throws {
    let count = try Self.checkedElementCount(shape)
    guard count == values.count else {
      throw H3NativeError.invalidTensor(
        "float32 value count \(values.count) does not match \(shape)"
      )
    }
    self.shape = shape
    scalarType = .float32
    bytes = values.withUnsafeBytes { Data($0) }
  }

  init(float16 values: [Float16], shape: [Int]) throws {
    let count = try Self.checkedElementCount(shape)
    guard count == values.count else {
      throw H3NativeError.invalidTensor(
        "float16 value count \(values.count) does not match \(shape)"
      )
    }
    self.shape = shape
    scalarType = .float16
    bytes = values.withUnsafeBytes { Data($0) }
  }

  init(bfloat16Raw values: [UInt16], shape: [Int]) throws {
    let count = try Self.checkedElementCount(shape)
    guard count == values.count else {
      throw H3NativeError.invalidTensor(
        "bfloat16 value count \(values.count) does not match \(shape)"
      )
    }
    self.shape = shape
    scalarType = .bfloat16
    bytes = values.withUnsafeBytes { Data($0) }
  }

  init(int32 values: [Int32], shape: [Int]) throws {
    let count = try Self.checkedElementCount(shape)
    guard count == values.count else {
      throw H3NativeError.invalidTensor(
        "int32 value count \(values.count) does not match \(shape)"
      )
    }
    self.shape = shape
    scalarType = .int32
    bytes = values.withUnsafeBytes { Data($0) }
  }

  init(int64 values: [Int64], shape: [Int]) throws {
    let count = try Self.checkedElementCount(shape)
    guard count == values.count else {
      throw H3NativeError.invalidTensor(
        "int64 value count \(values.count) does not match \(shape)"
      )
    }
    self.shape = shape
    scalarType = .int64
    bytes = values.withUnsafeBytes { Data($0) }
  }

  var elementCount: Int { shape.reduce(1, *) }

  func floatValues() throws -> [Float] {
    switch scalarType {
    case .bfloat16:
      return bytes.withUnsafeBytes { raw in
        raw.bindMemory(to: UInt16.self).map {
          Float(bitPattern: UInt32($0) << 16)
        }
      }
    case .float32:
      return bytes.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: Float.self))
      }
    case .float16:
      return bytes.withUnsafeBytes { raw in
        raw.bindMemory(to: Float16.self).map(Float.init)
      }
    default:
      throw H3NativeError.invalidTensor(
        "\(scalarType.rawValue) cannot be read as floating point"
      )
    }
  }

  func int32Values() throws -> [Int32] {
    guard scalarType == .int32 else {
      throw H3NativeError.invalidTensor(
        "\(scalarType.rawValue) cannot be read as int32"
      )
    }
    return bytes.withUnsafeBytes { raw in
      Array(raw.bindMemory(to: Int32.self))
    }
  }

  func reshaped(_ newShape: [Int]) throws -> H3Tensor {
    guard try Self.checkedElementCount(newShape) == elementCount else {
      throw H3NativeError.invalidTensor(
        "cannot reshape \(shape) to \(newShape)"
      )
    }
    return try H3Tensor(shape: newShape, scalarType: scalarType, bytes: bytes)
  }

  func converted(to type: H3ScalarType) throws -> H3Tensor {
    guard scalarType != type else { return self }
    let values = try floatValues()
    switch type {
    case .bfloat16:
      let words = values.map { value -> UInt16 in
        let bits = value.bitPattern
        let roundingBias = UInt32(0x7FFF) + ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: (bits &+ roundingBias) >> 16)
      }
      return try H3Tensor(bfloat16Raw: words, shape: shape)
    case .float16:
      return try H3Tensor(float16: values.map(Float16.init), shape: shape)
    case .float32:
      return try H3Tensor(float32: values, shape: shape)
    default:
      throw H3NativeError.unsupported(
        "floating point tensor conversion to \(type.rawValue)"
      )
    }
  }

  private static func checkedElementCount(_ shape: [Int]) throws -> Int {
    var result = 1
    for dimension in shape {
      let next = result.multipliedReportingOverflow(by: dimension)
      guard !next.overflow else {
        throw H3NativeError.invalidTensor("element count overflow")
      }
      result = next.partialValue
    }
    return result
  }
}

struct H3TensorConstraint: Codable, Sendable {
  let scalarType: H3ScalarType
  let shape: [Int]?

  func validate(_ tensor: H3Tensor, semantic: String) throws {
    guard tensor.scalarType == scalarType else {
      throw H3NativeError.invalidTensor(
        "\(semantic) is \(tensor.scalarType.rawValue), expected \(scalarType.rawValue)"
      )
    }
    if let shape {
      guard shape.count == tensor.shape.count else {
        throw H3NativeError.invalidTensor(
          "\(semantic) rank \(tensor.shape.count), expected \(shape.count)"
        )
      }
      for index in shape.indices where shape[index] > 0 {
        guard shape[index] == tensor.shape[index] else {
          throw H3NativeError.invalidTensor(
            "\(semantic) shape \(tensor.shape), expected \(shape); -1 means dynamic"
          )
        }
      }
    }
  }
}

struct H3StageManifest: Codable, Sendable {
  let backend: H3BackendKind
  let asset: String
  let function: String?
  let computeUnits: String?
  /// Number of original network layers fused into this executable stage.
  /// Ordinary stages and legacy manifests omit it and therefore count as one.
  let logicalLayerCount: Int?
  let inputs: [String: String]
  let outputs: [String: String]
  let inputConstraints: [String: H3TensorConstraint]?
  let outputConstraints: [String: H3TensorConstraint]?

  init(
    backend: H3BackendKind,
    asset: String,
    function: String?,
    computeUnits: String?,
    logicalLayerCount: Int? = nil,
    inputs: [String: String],
    outputs: [String: String],
    inputConstraints: [String: H3TensorConstraint]?,
    outputConstraints: [String: H3TensorConstraint]?
  ) {
    self.backend = backend
    self.asset = asset
    self.function = function
    self.computeUnits = computeUnits
    self.logicalLayerCount = logicalLayerCount
    self.inputs = inputs
    self.outputs = outputs
    self.inputConstraints = inputConstraints
    self.outputConstraints = outputConstraints
  }
}

struct H3QwenCompositeManifest: Codable, Sendable {
  let sequenceLength: Int
  let visionBlockBatch: Int
  let visionPatchesPerBlock: Int
  let visualTokensPerBlock: Int
  let tokenEmbedding: H3StageManifest
  let visionPatch: H3StageManifest
  let visionBlocks: [H3StageManifest]
  let visionDeepstackMergers: [H3StageManifest]
  let visionFinalMerger: H3StageManifest
  let languageLayers: [H3StageManifest]
  let deepstackVisionBlockIndices: [Int]
  let deepstackLanguageLayerIndices: [Int]

  var allStages: [H3StageManifest] {
    [tokenEmbedding, visionPatch]
      + visionBlocks
      + visionDeepstackMergers
      + [visionFinalMerger]
      + languageLayers
  }

  func validate(relativeTo directory: URL) throws {
    guard sequenceLength > 0,
      visionBlockBatch > 0,
      visionPatchesPerBlock > 0,
      visualTokensPerBlock > 0,
      visionPatchesPerBlock == visualTokensPerBlock * 4
    else {
      throw H3NativeError.invalidManifest("invalid qwenComposite geometry")
    }
    guard visionBlocks.count == 27, languageLayers.count == 50 else {
      throw H3NativeError.invalidManifest(
        "qwenComposite needs 27 vision blocks and 50 language layers"
      )
    }
    guard visionDeepstackMergers.count == 3,
      deepstackVisionBlockIndices == [8, 16, 24],
      deepstackLanguageLayerIndices == [0, 1, 2]
    else {
      throw H3NativeError.invalidManifest(
        "qwenComposite DeepStack must map vision 8/16/24 to language 0/1/2"
      )
    }
    for (index, stage) in allStages.enumerated() {
      guard stage.backend == .coreAI else {
        throw H3NativeError.invalidManifest(
          "qwenComposite stage \(index) must use Core AI"
        )
      }
      guard !stage.asset.isEmpty, !stage.inputs.isEmpty, !stage.outputs.isEmpty else {
        throw H3NativeError.invalidManifest(
          "qwenComposite stage \(index) has incomplete bindings"
        )
      }
      let asset = URL(fileURLWithPath: stage.asset, relativeTo: directory)
        .standardizedFileURL
      guard FileManager.default.fileExists(atPath: asset.path) else {
        throw H3NativeError.missingAsset(asset.path)
      }
    }
  }

  func assetFingerprint(relativeTo directory: URL) throws -> Data {
    var parts: [Data] = []
    parts.reserveCapacity(allStages.count)
    for stage in allStages {
      let asset = URL(fileURLWithPath: stage.asset, relativeTo: directory)
        .standardizedFileURL
      parts.append(try H3StageCache.assetFingerprint(asset))
    }
    return Data(H3StageCache.key(parts: parts).utf8)
  }
}

/// Native Swift orchestration for 10Eros-Max H3's dynamic-token DiT.  The
/// individual Core AI programs deliberately stay small enough to compile and
/// load reliably; Swift owns the ref2va packing, RoPE, time-curve lookup and
/// the 50-block execution order.
struct H3DenoiserCompositeManifest: Codable, Sendable {
  let textRefiner: H3StageManifest
  let videoProjection: H3StageManifest
  let audioProjection: H3StageManifest
  let blocks: [H3StageManifest]
  let finalVideo: H3StageManifest
  let finalAudio: H3StageManifest
  let adalnTableAsset: String
  let ropeInverseFrequencyAsset: String
  let dynamicMaximumTokens: Int

  var allStages: [H3StageManifest] {
    [textRefiner, videoProjection, audioProjection]
      + blocks
      + [finalVideo, finalAudio]
  }

  func validate(relativeTo directory: URL) throws {
    let logicalBlockCount = blocks.reduce(0) {
      $0 + max(1, $1.logicalLayerCount ?? 1)
    }
    guard logicalBlockCount == 50 else {
      throw H3NativeError.invalidManifest(
        "denoiserComposite needs exactly 50 logical DiT blocks, got \(logicalBlockCount)"
      )
    }
    guard dynamicMaximumTokens > 0 else {
      throw H3NativeError.invalidManifest(
        "denoiserComposite.dynamicMaximumTokens must be positive"
      )
    }
    for (index, stage) in allStages.enumerated() {
      guard stage.backend == .coreAI else {
        throw H3NativeError.invalidManifest(
          "denoiserComposite stage \(index) must use Core AI"
        )
      }
      guard !stage.asset.isEmpty, !stage.inputs.isEmpty, !stage.outputs.isEmpty else {
        throw H3NativeError.invalidManifest(
          "denoiserComposite stage \(index) has incomplete bindings"
        )
      }
      let asset = URL(fileURLWithPath: stage.asset, relativeTo: directory)
        .standardizedFileURL
      guard FileManager.default.fileExists(atPath: asset.path) else {
        throw H3NativeError.missingAsset(asset.path)
      }
    }
    for tablePath in [adalnTableAsset, ropeInverseFrequencyAsset] {
      let asset = URL(fileURLWithPath: tablePath, relativeTo: directory)
        .standardizedFileURL
      guard FileManager.default.fileExists(atPath: asset.path) else {
        throw H3NativeError.missingAsset(asset.path)
      }
    }
  }

  func assetFingerprint(relativeTo directory: URL) throws -> Data {
    var parts = try allStages.map { stage -> Data in
      let asset = URL(fileURLWithPath: stage.asset, relativeTo: directory)
        .standardizedFileURL
      return try H3StageCache.assetFingerprint(asset)
    }
    for tablePath in [adalnTableAsset, ropeInverseFrequencyAsset] {
      let asset = URL(fileURLWithPath: tablePath, relativeTo: directory)
        .standardizedFileURL
      parts.append(try H3StageCache.assetFingerprint(asset))
    }
    return Data(H3StageCache.key(parts: parts).utf8)
  }
}

enum H3ConditioningMode: String, Codable, Sendable {
  case ref2va
  case fl2va
}

enum H3MusicVideoContinuationMode: String, Codable, Sendable, CaseIterable {
  /// Continuum's exact 22-frame AV target prefix plus the two immediately
  /// preceding H3-Extend context tokens. This is the recommended long-form
  /// route: the overlap is bit-exact while older motion remains visible to
  /// attention without duplicating the protected prefix.
  case hybridAV = "hybrid-av"
  case latentPrefix = "latent-prefix"
  case firstFrame = "first"
  case firstAndProvidedLast = "first-last-provided"
  case firstAndGeneratedLast = "first-last-generated"
}

struct H3PipelineManifest: Codable, Sendable {
  let schemaVersion: Int
  let modelIdentifier: String
  let conditioningMode: H3ConditioningMode?
  let fixedPrompt: String?
  let tokenizerDirectory: String?
  let qwenComposite: H3QwenCompositeManifest?
  let denoiserComposite: H3DenoiserCompositeManifest?
  let stages: [String: H3StageManifest]
  let backendStages: [String: [String: H3StageManifest]]?
  let sampler: String?
  let samplerNoise: Float?
  let samplerMaxStage: Int?
  let sigmas: [Float]
  let videoShift: Float
  let audioShift: Float
  let visualConditionNoiseAug: Float?
  let audioConditionNoiseAug: Float?

  var resolvedConditioningMode: H3ConditioningMode {
    if let conditioningMode { return conditioningMode }
    return modelIdentifier.lowercased().contains("fl2va") ? .fl2va : .ref2va
  }

  static let baseRequiredStages = [
    "videoEncoder", "audioEncoder", "videoDecoder", "audioDecoder",
  ]

  func validate(relativeTo directory: URL) throws {
    guard schemaVersion == 1 else {
      throw H3NativeError.invalidManifest(
        "schemaVersion \(schemaVersion) is not supported"
      )
    }
    guard !modelIdentifier.isEmpty else {
      throw H3NativeError.invalidManifest("modelIdentifier is empty")
    }
    if let fixedPrompt, fixedPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      throw H3NativeError.invalidManifest("fixedPrompt is empty")
    }
    try validateStages(stages, profile: "default", relativeTo: directory)
    for (profile, profileStages) in backendStages ?? [:] {
      guard H3BackendKind(rawValue: profile) != nil else {
        throw H3NativeError.invalidManifest("unknown backend profile \(profile)")
      }
      try validateStages(profileStages, profile: profile, relativeTo: directory)
    }
    if let qwenComposite {
      try qwenComposite.validate(relativeTo: directory)
    } else {
      for (profile, profileStages) in [("default", stages)]
        + (backendStages ?? [:]).map({ ($0.key, $0.value) })
      {
        guard let stage = profileStages["textEncoder"] else {
          throw H3NativeError.missingStage("\(profile).textEncoder")
        }
        try validateStage(
          stage, semantic: "\(profile).textEncoder", relativeTo: directory
        )
      }
    }
    if let denoiserComposite {
      try denoiserComposite.validate(relativeTo: directory)
    } else {
      for (profile, profileStages) in [("default", stages)]
        + (backendStages ?? [:]).map({ ($0.key, $0.value) })
      {
        guard let stage = profileStages["denoiser"] else {
          throw H3NativeError.missingStage("\(profile).denoiser")
        }
        try validateStage(
          stage, semantic: "\(profile).denoiser", relativeTo: directory
        )
      }
    }
    guard sigmas.count >= 2, sigmas.last == 0,
      zip(sigmas, sigmas.dropFirst()).allSatisfy({ $0 > $1 })
    else {
      throw H3NativeError.invalidManifest(
        "sigmas must be strictly descending and end at zero"
      )
    }
    let samplerName = sampler ?? "res_multistep"
    guard ["res_multistep", "er_sde"].contains(samplerName) else {
      throw H3NativeError.invalidManifest("unsupported sampler \(samplerName)")
    }
    if let samplerNoise, samplerNoise < 0 {
      throw H3NativeError.invalidManifest("samplerNoise must be nonnegative")
    }
    if let samplerMaxStage, !(1...3).contains(samplerMaxStage) {
      throw H3NativeError.invalidManifest("samplerMaxStage must be between 1 and 3")
    }
    guard videoShift > 0, audioShift > 0 else {
      throw H3NativeError.invalidManifest("flow shifts must be positive")
    }
  }

  func resolvedStages(backend: H3BackendKind?) throws -> [String: H3StageManifest] {
    guard let backend else { return stages }
    if let profile = backendStages?[backend.rawValue] { return profile }
    guard stages.values.allSatisfy({ $0.backend == backend }) else {
      throw H3NativeError.invalidManifest(
        "manifest has no complete \(backend.rawValue) backend profile"
      )
    }
    return stages
  }

  private func validateStages(
    _ stages: [String: H3StageManifest],
    profile: String,
    relativeTo directory: URL
  ) throws {
    for name in Self.baseRequiredStages {
      guard let stage = stages[name] else {
        throw H3NativeError.missingStage("\(profile).\(name)")
      }
      try validateStage(
        stage, semantic: "\(profile).\(name)", relativeTo: directory
      )
    }
  }

  private func validateStage(
    _ stage: H3StageManifest,
    semantic: String,
    relativeTo directory: URL
  ) throws {
    guard !stage.asset.isEmpty else {
      throw H3NativeError.invalidManifest("\(semantic).asset is empty")
    }
    let asset = URL(fileURLWithPath: stage.asset, relativeTo: directory)
      .standardizedFileURL
    guard FileManager.default.fileExists(atPath: asset.path) else {
      throw H3NativeError.missingAsset(asset.path)
    }
    guard !stage.inputs.isEmpty, !stage.outputs.isEmpty else {
      throw H3NativeError.invalidManifest(
        "\(semantic) needs semantic input and output bindings"
      )
    }
  }
}

struct H3NativeJob: Codable, Sendable {
  let input: String?
  let inputImages: [String]?
  let output: String
  let prompt: String
  let cacheDirectory: String
  let width: Int
  let height: Int
  let outputWidth: Int?
  let outputHeight: Int?
  let audioInput: String?
  let audioStartSeconds: Double?
  let durationSeconds: Double
  let seed: UInt64
  let backend: H3BackendKind?
  let outputTrimStartSeconds: Double?
  let outputDurationSeconds: Double?
  let preserveSourceAudioWhenDecoderIsUnavailable: Bool?
  var audioConditioningMode: H3AudioConditioningMode? = nil
  var musicVideoCutPointsSeconds: [Double]? = nil
  var musicVideoContinuationMode: H3MusicVideoContinuationMode? = nil
  var musicVideoLastFrameDirectory: String? = nil
  /// Optional interval-specific composition anchors for a flat MV timeline.
  /// Files are named `entry-0000.png`, `entry-0001.png`, ... by the flat
  /// prompt entry index. Only entries whose transition is `cut` are read.
  var musicVideoStoryboardDirectory: String? = nil

  var resolvedMusicVideoContinuationMode: H3MusicVideoContinuationMode {
    musicVideoContinuationMode ?? .hybridAV
  }

  var resolvedOutputWidth: Int { outputWidth ?? width }
  var resolvedOutputHeight: Int { outputHeight ?? height }
  var resolvedAudioStartSeconds: Double { audioStartSeconds ?? 0 }
  var resolvedAudioConditioningMode: H3AudioConditioningMode {
    audioConditioningMode ?? .backgroundMusic
  }
  var resolvedOutputTrimStartSeconds: Double { outputTrimStartSeconds ?? 0 }
  var resolvedOutputDurationSeconds: Double {
    outputDurationSeconds ?? durationSeconds
  }

  func validate(
    conditioningMode: H3ConditioningMode = .ref2va,
    allowsLatentOnlyContinuation: Bool = false
  ) throws {
    let video = input?.trimmingCharacters(in: .whitespacesAndNewlines)
    let images = inputImages?.filter {
      !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    } ?? []
    switch conditioningMode {
    case .ref2va:
      if allowsLatentOnlyContinuation {
        guard video?.isEmpty != false, images.isEmpty else {
          throw H3NativeError.invalidJob(
            "latent-only continuation cannot also use reference media"
          )
        }
      } else {
        guard (video?.isEmpty == false) != !images.isEmpty else {
          throw H3NativeError.invalidJob(
            "select exactly one input video or one or more input images"
          )
        }
        if let video, !video.isEmpty {
          guard FileManager.default.fileExists(atPath: video) else {
            throw H3NativeError.missingAsset(video)
          }
        } else {
          guard images.count <= H3Geometry.identityVisionBlocks else {
            throw H3NativeError.invalidJob(
              "at most \(H3Geometry.identityVisionBlocks) identity images are supported"
            )
          }
          for image in images {
            guard FileManager.default.fileExists(atPath: image) else {
              throw H3NativeError.missingAsset(image)
            }
          }
        }
      }
    case .fl2va:
      guard video?.isEmpty != false, images.count <= 2 else {
        throw H3NativeError.invalidJob(
          "native FL2VA accepts prompt-only generation, a first frame, or first and last frames"
        )
      }
      for image in images {
        guard FileManager.default.fileExists(atPath: image) else {
          throw H3NativeError.missingAsset(image)
        }
      }
    }
    let audio = audioInput?.trimmingCharacters(in: .whitespacesAndNewlines)
    if let audio, !audio.isEmpty {
      let acceptsExternalAudio = conditioningMode == .ref2va
        ? ((!images.isEmpty && video?.isEmpty != false)
          || allowsLatentOnlyContinuation)
        : ((1...2).contains(images.count) && video?.isEmpty != false)
      guard acceptsExternalAudio
      else {
        throw H3NativeError.invalidJob(
          "external audio conditioning requires Ref2VA images, an exact continuation latent, or an FL2VA first frame"
        )
      }
      guard FileManager.default.fileExists(atPath: audio) else {
        throw H3NativeError.missingAsset(audio)
      }
      guard resolvedAudioStartSeconds.isFinite, resolvedAudioStartSeconds >= 0 else {
        throw H3NativeError.invalidJob("audioStartSeconds must be non-negative")
      }
      guard durationSeconds <= H3Geometry.audioConditioningSeconds else {
        throw H3NativeError.invalidJob(
          "external audio conditioning supports shots up to 10 seconds"
        )
      }
    }
    guard !output.isEmpty, !cacheDirectory.isEmpty else {
      throw H3NativeError.invalidJob("output and cacheDirectory are required")
    }
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw H3NativeError.invalidJob("prompt is empty")
    }
    guard resolvedOutputTrimStartSeconds.isFinite,
      resolvedOutputTrimStartSeconds >= 0,
      resolvedOutputDurationSeconds.isFinite,
      resolvedOutputDurationSeconds > 0,
      resolvedOutputTrimStartSeconds + resolvedOutputDurationSeconds
        <= durationSeconds + 1e-6
    else {
      throw H3NativeError.invalidJob(
        "output trim and duration must fit inside generated duration"
      )
    }
    guard width >= 32, height >= 32, width % 32 == 0, height % 32 == 0 else {
      throw H3NativeError.invalidJob(
        "width and height must be positive multiples of 32"
      )
    }
    guard (outputWidth == nil) == (outputHeight == nil) else {
      throw H3NativeError.invalidJob(
        "outputWidth and outputHeight must be specified together"
      )
    }
    guard resolvedOutputWidth > 0, resolvedOutputHeight > 0,
      resolvedOutputWidth <= width, resolvedOutputHeight <= height,
      resolvedOutputWidth % 2 == 0, resolvedOutputHeight % 2 == 0,
      (width - resolvedOutputWidth) % 2 == 0,
      (height - resolvedOutputHeight) % 2 == 0
    else {
      throw H3NativeError.invalidJob(
        "output dimensions must be even and form a centered crop of the generation canvas"
      )
    }
    guard durationSeconds >= 2, durationSeconds <= 15 else {
      throw H3NativeError.invalidJob("reference duration must be 2...15 seconds")
    }
    if resolvedOutputWidth == 1920, resolvedOutputHeight == 1080 {
      guard width == 1920, height == 1088,
        abs(durationSeconds - 6) < 0.001
      else {
        throw H3NativeError.invalidJob(
          "the 1920x1080 profile requires a 1920x1088 canvas and exactly 6 seconds"
        )
      }
    }
    if let musicVideoCutPointsSeconds {
      guard musicVideoCutPointsSeconds.allSatisfy({
        $0.isFinite && $0 > 0
      }) else {
        throw H3NativeError.invalidJob(
          "music-video cut points must contain positive finite seconds"
        )
      }
    }
  }
}

enum H3Geometry {
  static let framesPerSecond = 24
  static let audioLatentFramesPerSecond = 40
  // The converted audio VAE encoder has a fixed ten-second input contract.
  // Shorter music-video shots are zero-padded before encoding and their latent
  // is cropped to the target temporal shape.
  static let audioConditioningSeconds = 10.0
  // Keep two of the compiled ten Qwen vision slots available for prompt text.
  // Eight 405-token image blocks still leave roughly 800 tokens for motion,
  // soundscape and music instructions; ten blocks leave only 16.
  static let identityVisionBlocks = 8
  // The exported Qwen vision tower has a fixed 10 x 864x480 visual contract,
  // independently of the variable-resolution DiT/VAE output canvas.
  static let qwenVisionWidth = 864
  static let qwenVisionHeight = 480

  static func identityImageIndex(
    slot: Int,
    slotCount: Int = identityVisionBlocks,
    imageCount: Int
  ) -> Int {
    guard imageCount > 1, slotCount > 1 else { return 0 }
    return min(
      imageCount - 1,
      Int(
        (Double(slot) * Double(imageCount - 1)
          / Double(slotCount - 1)).rounded()
      )
    )
  }

  static func alignedGenerationFrameCount(durationSeconds: Double) -> Int {
    alignFrameCount(max(5, Int((durationSeconds * 24).rounded())))
  }

  static func alignFrameCount(_ proposed: Int) -> Int {
    var count = max(5, proposed)
    while count % 17 != 5 { count += 1 }
    return count
  }

  /// Returns the latest complete H3 causal frame boundary at or before the
  /// requested frame. This is used when a fixed generation profile decodes
  /// beyond the duration that is actually written to the movie.
  static func alignedFrameCount(notAfter proposed: Int) -> Int {
    guard proposed >= 5 else { return 0 }
    return proposed - (proposed - 5) % 17
  }

  static func isAlignedFrameCount(_ count: Int) -> Bool {
    count >= 5 && count % 17 == 5
  }

  static func referenceFrameCount(available: Int, output: Int) throws -> Int {
    var count = min(available, output)
    guard count >= 5 else {
      throw H3NativeError.media("reference video needs at least five frames")
    }
    while count % 17 != 5 { count -= 1 }
    return count
  }

  static func videoLatentFrames(pixelFrames: Int) -> Int {
    let aligned = alignFrameCount(pixelFrames)
    return aligned <= 5 ? 2 : ((aligned - 5) / 17) * 5 + 2
  }

  static func audioLatentFrames(pixelFrames: Int) -> Int {
    Int((Double(pixelFrames) / 24.0 * 40.0).rounded())
  }

  static func qwenVideoSampleIndices(
    frameCount: Int,
    maximumBlocks: Int = 10
  ) -> [Int] {
    guard frameCount > 0, maximumBlocks > 0 else { return [] }
    let candidates = Array(stride(from: 0, to: frameCount, by: 12))
    let maximumFrames = maximumBlocks * 2
    var selected: [Int]
    if candidates.count <= maximumFrames {
      selected = candidates
    } else {
      // The compiled Qwen graph has a fixed number of paired vision blocks.
      // Preserve the complete time range for long references by uniformly
      // selecting from the normal 2 fps candidates instead of truncating the
      // tail of the clip.
      selected = (0..<maximumFrames).map { index in
        let candidateIndex = Int(
          (Double(index) * Double(candidates.count - 1)
            / Double(maximumFrames - 1)).rounded()
        )
        return candidates[candidateIndex]
      }
    }
    if selected.count % 2 == 1, let last = selected.last {
      selected.append(last)
    }
    return selected
  }

  static func adaptCanvas(width: Int, height: Int) -> (width: Int, height: Int) {
    let baseShortEdge = 768.0
    let maximumPixels = 768.0 * 1344.0
    let ratio = Double(width) / Double(height)
    var nominalWidth: Double
    var nominalHeight: Double
    if ratio >= 1 {
      nominalWidth = baseShortEdge * ratio
      nominalHeight = baseShortEdge
    } else {
      nominalWidth = baseShortEdge
      nominalHeight = baseShortEdge / ratio
    }
    if nominalWidth * nominalHeight > maximumPixels {
      let scale = sqrt(maximumPixels / (nominalWidth * nominalHeight))
      nominalWidth *= scale
      nominalHeight *= scale
    }
    return (
      max(32, Int((nominalWidth / 32).rounded()) * 32),
      max(32, Int((nominalHeight / 32).rounded()) * 32)
    )
  }
}

enum H3ShotPromptSelector {
  private static let structuredSectionNames: Set<String> = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
  ]
  private static let continuationSectionNames: Set<String> = [
    "subject_definitions",
    "retention_analysis",
    "overall_soundscape",
    "non_diegetic_music",
  ]

  static func contains(_ prompt: String, shotIndex: Int) -> Bool {
    guard shotIndex >= 0,
      let matches = try? shotMatches(in: prompt)
    else { return false }
    let requestedNumber = shotIndex + 1
    return matches.contains { match in
      guard let numberRange = Range(match.range(at: 1), in: prompt) else {
        return false
      }
      return Int(prompt[numberRange]) == requestedNumber
    }
  }

  /// Keeps global Ref2VA sections plus only the requested `[Shot N]` block.
  /// This prevents a long-form prompt from being interpreted as a montage
  /// inside every independently generated logical Shot.
  static func select(_ prompt: String, shotIndex: Int) throws -> String {
    guard shotIndex >= 0 else {
      throw H3NativeError.invalidJob("logical shot index must be non-negative")
    }
    let fullRange = NSRange(prompt.startIndex..<prompt.endIndex, in: prompt)
    let shotMatches = try shotMatches(in: prompt)
    guard let firstShot = shotMatches.first else { return prompt }

    let suffixExpression = try NSRegularExpression(
      pattern: #"^[\t ]*(overall_soundscape|non_diegetic_music):"#,
      options: [.anchorsMatchLines, .caseInsensitive]
    )
    let suffixSearchRange = NSRange(
      location: firstShot.range.location,
      length: fullRange.length - firstShot.range.location
    )
    let suffixStart = suffixExpression.firstMatch(
      in: prompt,
      range: suffixSearchRange
    )?.range.location ?? fullRange.length

    let requestedNumber = shotIndex + 1
    let requestedMatchIndex = shotMatches.firstIndex { match in
      guard let numberRange = Range(match.range(at: 1), in: prompt) else {
        return false
      }
      return Int(prompt[numberRange]) == requestedNumber
    }

    let prefix = substring(
      prompt,
      range: NSRange(location: 0, length: firstShot.range.location)
    )
    let selectedBlock: String
    if let requestedMatchIndex {
      let start = shotMatches[requestedMatchIndex].range.location
      let nextStart = requestedMatchIndex + 1 < shotMatches.count
        ? shotMatches[requestedMatchIndex + 1].range.location
        : fullRange.length
      selectedBlock = substring(
        prompt,
        range: NSRange(
          location: start,
          length: max(0, min(nextStart, suffixStart) - start)
        )
      )
    } else {
      // The runtime may create more musical Shots than the prompt names.
      // Keep global guidance, but never fall back to all named scenes.
      selectedBlock = ""
    }
    let suffix = suffixStart < fullRange.length
      ? substring(
        prompt,
        range: NSRange(
          location: suffixStart,
          length: fullRange.length - suffixStart
        )
      )
      : ""

    var orderedParts: [String]
    let detailsExpression = try NSRegularExpression(
      pattern: #"^[\t ]*detailed_description:[^\r\n]*(?:\r?\n|$)"#,
      options: [.anchorsMatchLines, .caseInsensitive]
    )
    let prefixRange = NSRange(prefix.startIndex..<prefix.endIndex, in: prefix)
    if !selectedBlock.isEmpty,
      let details = detailsExpression.firstMatch(
        in: prefix,
        range: prefixRange
      )
    {
      let headerEnd = NSMaxRange(details.range)
      let prefixThroughHeader = substring(
        prefix,
        range: NSRange(location: 0, length: headerEnd)
      )
      let detailedPreamble = substring(
        prefix,
        range: NSRange(
          location: headerEnd,
          length: prefixRange.length - headerEnd
        )
      )
      // Put the current Shot's actual location immediately after the section
      // header. The runner inserts its priority directive at the same point,
      // so both survive Qwen's finite prompt-token budget ahead of the longer
      // global visual guidance.
      orderedParts = [
        prefixThroughHeader, selectedBlock, detailedPreamble, suffix,
      ]
    } else {
      orderedParts = [prefix, selectedBlock, suffix]
    }

    return orderedParts
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
      .joined(separator: "\n\n")
  }

  /// Keeps identity and audio invariants but removes every scene/action block.
  /// A continuation Part already receives the preceding target latent as its
  /// fixed history; repeating the Shot prose makes the model restart the same
  /// action instead of advancing beyond that history.
  static func continuationContext(_ prompt: String) -> String {
    var activeSection: String?
    var keptLines: [String] = []
    for line in prompt.components(separatedBy: .newlines) {
      let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
      let lowercased = trimmed.lowercased()
      if lowercased.hasSuffix(":"), !lowercased.contains(" ") {
        let candidate = String(lowercased.dropLast())
        if structuredSectionNames.contains(candidate) {
          activeSection = candidate
          if continuationSectionNames.contains(candidate) {
            keptLines.append(line)
          }
          continue
        }
      }
      if let activeSection, continuationSectionNames.contains(activeSection) {
        keptLines.append(line)
      }
    }
    return keptLines
      .joined(separator: "\n")
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }

  private static func substring(_ text: String, range: NSRange) -> String {
    guard let swiftRange = Range(range, in: text) else { return "" }
    return String(text[swiftRange])
  }

  private static func shotMatches(in prompt: String) throws
    -> [NSTextCheckingResult]
  {
    let expression = try NSRegularExpression(
      pattern: #"^[\t ]*\[Shot[\t ]+(\d+)\][\t ]*"#,
      options: [.anchorsMatchLines, .caseInsensitive]
    )
    return expression.matches(
      in: prompt,
      range: NSRange(prompt.startIndex..<prompt.endIndex, in: prompt)
    )
  }
}

struct H3FlatTimelinePromptEntry: Sendable, Equatable {
  let startSeconds: Double
  let endSeconds: Double
  let transition: H3MusicVideoTransition
  let body: String
}

struct H3FlatTimelinePromptPlan: Sendable, Equatable {
  let prefixThroughDetailedDescription: String
  let promptPrefix: String
  let detailedPreamble: String
  let suffix: String
  let continuationBlendFrames: Int?
  let entries: [H3FlatTimelinePromptEntry]

  func compiledPrompt(
    entryIndex: Int,
    directive: String,
    entryBodyOverride: String? = nil
  ) throws -> String {
    guard entries.indices.contains(entryIndex) else {
      throw H3NativeError.invalidJob(
        "flat timeline prompt index \(entryIndex) is out of range"
      )
    }
    return [
      prefixThroughDetailedDescription,
      directive,
      promptPrefix,
      detailedPreamble,
      entryBodyOverride ?? entries[entryIndex].body,
      suffix,
    ]
    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    .filter { !$0.isEmpty }
    .joined(separator: "\n\n")
  }
}

private struct H3ChainPromptShot: Decodable {
  let id: String?
  let prompt: String
  let length: Int?
  let frames: Int?
  let start: Double?
  let end: Double?
  let transition: String?
  let anchorMode: String?
  let contextLength: Int?

  private enum CodingKeys: String, CodingKey {
    case id
    case prompt
    case length
    case frames
    case start
    case end
    case transition
    case anchorMode = "anchor_mode"
    case contextLength = "context_length"
  }
}

private struct H3ChainPromptDocument: Decodable {
  let promptPrefix: String?
  let globalContinuity: String?
  let shots: [H3ChainPromptShot]
  let fps: Double?
  let seamTaperFrames: Int?
  let overlapBlendFrames: Int?

  private enum CodingKeys: String, CodingKey {
    case promptPrefix = "prompt_prefix"
    case globalContinuity = "global_continuity"
    case shots
    case fps
    case seamTaperFrames = "seam_taper_frames"
    case overlapBlendFrames = "overlap_blend_frames"
  }
}

/// Parses an app-side, flat long-form timeline embedded inside
/// `detailed_description`. Markers such as `[0.000-8.000 cut]` are never sent
/// to H3; each invocation receives only its concrete interval body plus the
/// normal six-section Ref2VA context.
enum H3FlatTimelinePrompt {
  /// Flat-timeline transitions are implemented by the app, outside each H3
  /// invocation.  A leading editorial cue such as "Hard cut to ..." must not
  /// be sent to a single generated interval: H3 may perform that edit midway
  /// through its own clip instead of starting in the requested composition.
  static func singleTakeBody(_ body: String) -> String {
    guard let expression = try? NSRegularExpression(
      pattern: #"^\s*(?:(?:final\s+)?hard\s+cut|cut)\s+to\s+"#,
      options: [.caseInsensitive]
    ) else {
      return body
    }
    let range = NSRange(body.startIndex..<body.endIndex, in: body)
    return expression.stringByReplacingMatches(
      in: body,
      range: range,
      withTemplate: "This single uninterrupted take begins with "
    )
  }

  static func parse(_ prompt: String) throws -> H3FlatTimelinePromptPlan? {
    if let chain = try parseChainJSON(prompt) { return chain }
    let fullRange = NSRange(prompt.startIndex..<prompt.endIndex, in: prompt)
    let detailsExpression = try NSRegularExpression(
      pattern: #"^[\t ]*detailed_description:[^\r\n]*(?:\r?\n|$)"#,
      options: [.anchorsMatchLines, .caseInsensitive]
    )
    guard let details = detailsExpression.firstMatch(
      in: prompt,
      range: fullRange
    ) else { return nil }
    let suffixExpression = try NSRegularExpression(
      pattern: #"^[\t ]*overall_soundscape:"#,
      options: [.anchorsMatchLines, .caseInsensitive]
    )
    let suffixSearch = NSRange(
      location: NSMaxRange(details.range),
      length: fullRange.length - NSMaxRange(details.range)
    )
    let suffixStart = suffixExpression.firstMatch(
      in: prompt,
      range: suffixSearch
    )?.range.location ?? fullRange.length
    let bodyRange = NSRange(
      location: NSMaxRange(details.range),
      length: max(0, suffixStart - NSMaxRange(details.range))
    )
    let body = substring(prompt, range: bodyRange)
    let markerExpression = try NSRegularExpression(
      pattern: #"^[\t ]*\[([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)s?\s+(cut|continue)\][\t ]*(.*)$"#,
      options: [.anchorsMatchLines, .caseInsensitive]
    )
    let bodyFullRange = NSRange(body.startIndex..<body.endIndex, in: body)
    let matches = markerExpression.matches(in: body, range: bodyFullRange)
    guard !matches.isEmpty else { return nil }

    let prefix = substring(
      prompt,
      range: NSRange(location: 0, length: NSMaxRange(details.range))
    )
    let rawPreamble = substring(
      body,
      range: NSRange(location: 0, length: matches[0].range.location)
    ).trimmingCharacters(in: .whitespacesAndNewlines)
    let splitPreamble = splitPromptPrefix(rawPreamble)
    let suffix = suffixStart < fullRange.length
      ? substring(
        prompt,
        range: NSRange(
          location: suffixStart,
          length: fullRange.length - suffixStart
        )
      )
      : ""
    var entries: [H3FlatTimelinePromptEntry] = []
    entries.reserveCapacity(matches.count)
    for index in matches.indices {
      let match = matches[index]
      guard let startRange = Range(match.range(at: 1), in: body),
        let endRange = Range(match.range(at: 2), in: body),
        let transitionRange = Range(match.range(at: 3), in: body),
        let start = Double(body[startRange]),
        let end = Double(body[endRange]),
        let transition = H3MusicVideoTransition(
          rawValue: body[transitionRange].lowercased()
        )
      else {
        throw H3NativeError.invalidJob("invalid flat timeline marker")
      }
      let inlineContent = match.range(at: 4).location == NSNotFound
        ? ""
        : substring(body, range: match.range(at: 4))
          .trimmingCharacters(in: .whitespacesAndNewlines)
      let contentStart = NSMaxRange(match.range)
      let contentEnd = index + 1 < matches.count
        ? matches[index + 1].range.location
        : bodyFullRange.length
      let followingContent = substring(
        body,
        range: NSRange(
          location: contentStart,
          length: max(0, contentEnd - contentStart)
        )
      ).trimmingCharacters(in: .whitespacesAndNewlines)
      let content = [inlineContent, followingContent]
        .filter { !$0.isEmpty }
        .joined(separator: "\n")
      guard start >= 0, end > start, !content.isEmpty else {
        throw H3NativeError.invalidJob(
          "flat timeline entries need increasing times and non-empty text"
        )
      }
      if let previous = entries.last {
        guard abs(previous.endSeconds - start) <= 1.0 / 24.0 + 1e-6 else {
          throw H3NativeError.invalidJob(
            "flat timeline must be contiguous at \(start) seconds"
          )
        }
      } else if start > 1.0 / 24.0 + 1e-6 {
        throw H3NativeError.invalidJob("flat timeline must start at zero")
      }
      entries.append(
        H3FlatTimelinePromptEntry(
          startSeconds: start,
          endSeconds: end,
          transition: transition,
          body: content
        )
      )
    }
    guard entries.first?.transition == .cut else {
      throw H3NativeError.invalidJob(
        "the first flat timeline entry must use cut"
      )
    }
    return H3FlatTimelinePromptPlan(
      prefixThroughDetailedDescription: prefix,
      promptPrefix: splitPreamble.promptPrefix,
      detailedPreamble: splitPreamble.remainingPreamble,
      suffix: suffix,
      continuationBlendFrames: nil,
      entries: entries
    )
  }

  private static func parseChainJSON(_ prompt: String) throws
    -> H3FlatTimelinePromptPlan?
  {
    let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.first == "{" else { return nil }
    let data = Data(trimmed.utf8)
    let decoder = JSONDecoder()
    let document: H3ChainPromptDocument
    do {
      document = try decoder.decode(H3ChainPromptDocument.self, from: data)
    } catch {
      return nil
    }
    guard !document.shots.isEmpty else {
      throw H3NativeError.invalidJob("H3 chain JSON needs at least one shot")
    }
    let frameRate = document.fps ?? Double(H3Geometry.framesPerSecond)
    guard frameRate > 0 else {
      throw H3NativeError.invalidJob("H3 chain JSON fps must be positive")
    }
    let promptPrefix = (document.promptPrefix ?? document.globalContinuity ?? "")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    let requestedBlendFrames = document.seamTaperFrames
      ?? document.overlapBlendFrames
    if let requestedBlendFrames,
      requestedBlendFrames < 0
        || requestedBlendFrames > H3VideoConditioning.partContinuationPixelFrames
    {
      throw H3NativeError.invalidJob(
        "H3 chain JSON seam taper frames must be 0...\(H3VideoConditioning.partContinuationPixelFrames)"
      )
    }
    var entries: [H3FlatTimelinePromptEntry] = []
    entries.reserveCapacity(document.shots.count)
    var cursor = 0.0
    for (index, shot) in document.shots.enumerated() {
      let transition = H3MusicVideoTransition(
        rawValue: (shot.transition ?? (index == 0 ? "cut" : "continue"))
          .lowercased()
      ) ?? (index == 0 ? .cut : .continue)
      if let anchorMode = shot.anchorMode,
        anchorMode.lowercased() != "head"
      {
        throw H3NativeError.invalidJob(
          "H3 chain JSON only supports head-anchored continuation in mioh"
        )
      }
      if let contextLength = shot.contextLength,
        contextLength != H3VideoConditioning.partContinuationPixelFrames
      {
        throw H3NativeError.invalidJob(
          "H3 chain JSON context_length must be \(H3VideoConditioning.partContinuationPixelFrames) for mioh latent continuation"
        )
      }
      let start = shot.start ?? cursor
      let end: Double
      if let explicitEnd = shot.end {
        end = explicitEnd
      } else if let frames = shot.frames ?? shot.length {
        end = start + Double(frames) / frameRate
      } else {
        throw H3NativeError.invalidJob(
          "H3 chain JSON shot \(index + 1) needs length, frames, or end"
        )
      }
      let body = shot.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
      guard start >= 0, end > start, !body.isEmpty else {
        throw H3NativeError.invalidJob(
          "H3 chain JSON shots need increasing times and non-empty prompts"
        )
      }
      if let previous = entries.last,
        abs(previous.endSeconds - start) > 1.0 / frameRate + 1e-6
      {
        throw H3NativeError.invalidJob(
          "H3 chain JSON must be contiguous at shot \(index + 1)"
        )
      }
      entries.append(
        H3FlatTimelinePromptEntry(
          startSeconds: start,
          endSeconds: end,
          transition: transition,
          body: body
        )
      )
      cursor = end
    }
    guard entries.first?.transition == .cut else {
      throw H3NativeError.invalidJob(
        "the first H3 chain JSON shot must use cut"
      )
    }
    return H3FlatTimelinePromptPlan(
      prefixThroughDetailedDescription: "detailed_description:",
      promptPrefix: promptPrefix,
      detailedPreamble: "",
      suffix: "",
      continuationBlendFrames: requestedBlendFrames,
      entries: entries
    )
  }

  private static func splitPromptPrefix(_ preamble: String)
    -> (promptPrefix: String, remainingPreamble: String)
  {
    let trimmed = preamble.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return ("", "") }
    guard let expression = try? NSRegularExpression(
      pattern: #"(?im)^[\t ]*(?:GLOBAL CONTINUITY|PROMPT_PREFIX|prompt_prefix)\s*:"#,
      options: []
    ) else {
      return (trimmed, "")
    }
    let range = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
    guard let match = expression.firstMatch(in: trimmed, range: range) else {
      return (trimmed, "")
    }
    let before = substring(
      trimmed,
      range: NSRange(location: 0, length: match.range.location)
    ).trimmingCharacters(in: .whitespacesAndNewlines)
    let prefix = substring(
      trimmed,
      range: NSRange(
        location: match.range.location,
        length: range.length - match.range.location
      )
    ).trimmingCharacters(in: .whitespacesAndNewlines)
    return (prefix, before)
  }

  private static func substring(_ text: String, range: NSRange) -> String {
    guard let swiftRange = Range(range, in: text) else { return "" }
    return String(text[swiftRange])
  }
}

enum H3MusicVideoSeed {
  static func value(base: UInt64, intervalIndex: Int) -> UInt64 {
    precondition(intervalIndex >= 0)
    return base
      &+ UInt64(intervalIndex) &* 0x9E37_79B9_7F4A_7C15
  }
}

enum H3AudioConditioning {
  static let partContinuationLatentFrames = H3Geometry.audioLatentFrames(
    pixelFrames: H3VideoConditioning.partContinuationPixelFrames
  )
  // H3-Extend's two tokens immediately before a phase-zero seven-token tile
  // are phases three and four. Together they span eight pixel frames, or
  // 13.333 audio-latent ticks at 40 Hz; the native implementation rounds this
  // context window to thirteen ticks.
  static let hybridHistoryLatentFrames = 13
  static let hybridStoredLatentFrames =
    partContinuationLatentFrames + hybridHistoryLatentFrames

  static func samplerState(
    clean: [Float],
    noise: [Float],
    videoSigma: Float,
    videoShift: Float,
    audioShift: Float
  ) -> [Float] {
    precondition(clean.count == noise.count)
    let sigmaV = max(videoSigma, 1e-6)
    let base = sigmaV / (videoShift + sigmaV * (1 - videoShift))
    let sigmaA = max(
      1e-12,
      audioShift * base / (1 + (audioShift - 1) * base)
    )
    let carryInverse = sigmaV / sigmaA
    return clean.indices.map { index in
      ((1 - sigmaA) * clean[index] + sigmaA * noise[index]) * carryInverse
    }
  }

  static func tail(
    _ tensor: H3Tensor,
    maximumFrames: Int,
    endingAtPixelFrameCount: Int? = nil
  ) throws -> H3Tensor {
    guard tensor.shape.count == 4, tensor.shape[0] == 1,
      tensor.shape[1] == 32, tensor.shape[2] == 2,
      tensor.shape[3] > 0, maximumFrames > 0
    else {
      throw H3NativeError.invalidTensor(
        "generated audio latent must be [1,32,2,T], got \(tensor.shape)"
      )
    }
    let sourceEnd: Int
    if let endingAtPixelFrameCount {
      sourceEnd = H3Geometry.audioLatentFrames(
        pixelFrames: endingAtPixelFrameCount
      )
      guard sourceEnd <= tensor.shape[3] else {
        throw H3NativeError.invalidTensor(
          "visible continuation audio end \(sourceEnd) exceeds source time \(tensor.shape[3])"
        )
      }
    } else {
      sourceEnd = tensor.shape[3]
    }
    guard sourceEnd >= maximumFrames else {
      throw H3NativeError.invalidTensor(
        "continuation needs \(maximumFrames) audio latent frames, found \(sourceEnd)"
      )
    }
    return try slice(
      tensor,
      frames: (sourceEnd - maximumFrames)..<sourceEnd
    )
  }

  static func slice(_ tensor: H3Tensor, frames: Range<Int>) throws
    -> H3Tensor
  {
    guard tensor.shape.count == 4, tensor.shape[0] == 1,
      tensor.shape[1] == 32, tensor.shape[2] == 2,
      frames.lowerBound >= 0, !frames.isEmpty,
      frames.upperBound <= tensor.shape[3]
    else {
      throw H3NativeError.invalidTensor(
        "audio continuation slice \(frames) does not fit \(tensor.shape)"
      )
    }
    let source = try tensor.floatValues().map(Float16.init)
    let sourceTime = tensor.shape[3]
    let outputTime = frames.count
    var output = [Float16](
      repeating: 0,
      count: tensor.shape[1] * tensor.shape[2] * outputTime
    )
    for channel in 0..<tensor.shape[1] {
      for side in 0..<tensor.shape[2] {
        let sourceOffset = (channel * tensor.shape[2] + side) * sourceTime
          + frames.lowerBound
        let targetOffset = (channel * tensor.shape[2] + side) * outputTime
        output.replaceSubrange(
          targetOffset..<(targetOffset + outputTime),
          with: source[sourceOffset..<(sourceOffset + outputTime)]
        )
      }
    }
    return try H3Tensor(
      float16: output,
      shape: [1, tensor.shape[1], tensor.shape[2], outputTime]
    )
  }

  static func replacingPrefix(
    in target: H3Tensor,
    with prefix: H3Tensor
  ) throws -> H3Tensor {
    guard target.shape.count == 4, prefix.shape.count == 4,
      target.shape[0...2].elementsEqual(prefix.shape[0...2]),
      prefix.shape[3] < target.shape[3]
    else {
      throw H3NativeError.invalidTensor(
        "audio prefix \(prefix.shape) does not fit target \(target.shape)"
      )
    }
    var values = try target.floatValues()
    let prefixValues = try prefix.floatValues()
    let targetTime = target.shape[3]
    let prefixTime = prefix.shape[3]
    for channel in 0..<target.shape[1] {
      for side in 0..<target.shape[2] {
        let targetOffset = (channel * target.shape[2] + side) * targetTime
        let prefixOffset = (channel * target.shape[2] + side) * prefixTime
        values.replaceSubrange(
          targetOffset..<(targetOffset + prefixTime),
          with: prefixValues[prefixOffset..<(prefixOffset + prefixTime)]
        )
      }
    }
    return try H3Tensor(float32: values, shape: target.shape)
      .converted(to: target.scalarType)
  }
}

enum H3VideoConditioning {
  // One decoder tile consumes seven latent tokens. Keeping the complete tile
  // also preserves H3's 5-token causal phase: generated clips are 5n+2 tokens,
  // so their final seven-token suffix starts at phase zero. A continuation
  // Part places this suffix at the beginning of its own target latent and
  // clamps it throughout sampling. Seven tokens decode to 22 frames; those
  // duplicate frames are trimmed from the completed Part.
  static let partContinuationTokens = 7
  static let partContinuationPixelFrames = 22
  static let hybridHistoryTokens = 2
  static let hybridStoredTokens = partContinuationTokens + hybridHistoryTokens

  static func prefixValues(
    from target: [Float],
    targetShape: [Int],
    prefixShape: [Int]
  ) throws -> [Float] {
    try validateTargetPrefix(
      target: target,
      targetShape: targetShape,
      prefixValues: nil,
      prefixShape: prefixShape
    )
    let channels = targetShape[1]
    let targetTime = targetShape[2]
    let prefixTime = prefixShape[2]
    let plane = targetShape[3] * targetShape[4]
    var result = [Float](
      repeating: 0,
      count: channels * prefixTime * plane
    )
    for channel in 0..<channels {
      for token in 0..<prefixTime {
        let sourceOffset = (channel * targetTime + token) * plane
        let destinationOffset = (channel * prefixTime + token) * plane
        result.replaceSubrange(
          destinationOffset..<(destinationOffset + plane),
          with: target[sourceOffset..<(sourceOffset + plane)]
        )
      }
    }
    return result
  }

  static func clampTargetPrefix(
    target: [Float],
    targetShape: [Int],
    cleanPrefix: [Float],
    noisePrefix: [Float],
    prefixShape: [Int],
    sigma: Float
  ) throws -> [Float] {
    try validateTargetPrefix(
      target: target,
      targetShape: targetShape,
      prefixValues: cleanPrefix,
      prefixShape: prefixShape
    )
    guard noisePrefix.count == cleanPrefix.count else {
      throw H3NativeError.invalidTensor(
        "continuation prefix noise count does not match the clean prefix"
      )
    }
    let channels = targetShape[1]
    let targetTime = targetShape[2]
    let prefixTime = prefixShape[2]
    let plane = targetShape[3] * targetShape[4]
    let amount = max(0, min(1, sigma))
    let cleanAmount = 1 - amount
    var result = target
    for channel in 0..<channels {
      for token in 0..<prefixTime {
        let destinationOffset = (channel * targetTime + token) * plane
        let sourceOffset = (channel * prefixTime + token) * plane
        for element in 0..<plane {
          let sourceIndex = sourceOffset + element
          result[destinationOffset + element] =
            cleanAmount * cleanPrefix[sourceIndex]
            + amount * noisePrefix[sourceIndex]
        }
      }
    }
    return result
  }

  private static func validateTargetPrefix(
    target: [Float],
    targetShape: [Int],
    prefixValues: [Float]?,
    prefixShape: [Int]
  ) throws {
    guard targetShape.count == 5, prefixShape.count == 5,
      targetShape[0] == 1, prefixShape[0] == 1,
      targetShape[1] == 24, prefixShape[1] == 24,
      prefixShape[2] == partContinuationTokens,
      prefixShape[2] <= targetShape[2],
      prefixShape[3] == targetShape[3],
      prefixShape[4] == targetShape[4],
      target.count == targetShape.reduce(1, *)
    else {
      throw H3NativeError.invalidTensor(
        "continuation prefix \(prefixShape) does not fit target \(targetShape)"
      )
    }
    if let prefixValues,
      prefixValues.count != prefixShape.reduce(1, *)
    {
      throw H3NativeError.invalidTensor(
        "continuation prefix values do not match \(prefixShape)"
      )
    }
  }

  static func tail(
    _ tensor: H3Tensor,
    maximumTokens: Int = partContinuationTokens,
    endingAtPixelFrameCount: Int? = nil
  ) throws -> H3Tensor {
    guard tensor.shape.count == 5, tensor.shape[0] == 1,
      tensor.shape[1] == 24, tensor.shape[2] > 0, maximumTokens > 0
    else {
      throw H3NativeError.invalidTensor(
        "generated video latent must be [1,24,T,H,W], got \(tensor.shape)"
      )
    }
    let sourceTime = tensor.shape[2]
    let sourceEnd: Int
    if let endingAtPixelFrameCount {
      guard H3Geometry.isAlignedFrameCount(endingAtPixelFrameCount) else {
        throw H3NativeError.invalidTensor(
          "visible continuation end \(endingAtPixelFrameCount) is not a 5+17n H3 frame boundary"
        )
      }
      let alignedFrames = H3Geometry.alignedFrameCount(
        notAfter: endingAtPixelFrameCount
      )
      guard alignedFrames >= 5 else {
        throw H3NativeError.invalidTensor(
          "visible continuation needs at least five pixel frames"
        )
      }
      sourceEnd = H3Geometry.videoLatentFrames(pixelFrames: alignedFrames)
      guard sourceEnd <= sourceTime else {
        throw H3NativeError.invalidTensor(
          "visible continuation latent end \(sourceEnd) exceeds source time \(sourceTime)"
        )
      }
    } else {
      sourceEnd = sourceTime
    }
    let tokenCount = min(maximumTokens, sourceEnd)
    guard tokenCount == maximumTokens else {
      throw H3NativeError.invalidTensor(
        "continuation needs \(maximumTokens) latent tokens, found \(sourceEnd)"
      )
    }
    let source = try tensor.floatValues().map(Float16.init)
    let plane = tensor.shape[3] * tensor.shape[4]
    var output = [Float16](
      repeating: 0,
      count: tensor.shape[0] * tensor.shape[1] * tokenCount * plane
    )
    for channel in 0..<tensor.shape[1] {
      for token in 0..<tokenCount {
        let sourceOffset = (
          (channel * sourceTime + sourceEnd - tokenCount + token) * plane
        )
        let targetOffset = (channel * tokenCount + token) * plane
        output.replaceSubrange(
          targetOffset..<(targetOffset + plane),
          with: source[sourceOffset..<(sourceOffset + plane)]
        )
      }
    }
    return try H3Tensor(
      float16: output,
      shape: [1, tensor.shape[1], tokenCount, tensor.shape[3], tensor.shape[4]]
    )
  }

  static func slice(_ tensor: H3Tensor, tokens: Range<Int>) throws
    -> H3Tensor
  {
    guard tensor.shape.count == 5, tensor.shape[0] == 1,
      tensor.shape[1] == 24, tokens.lowerBound >= 0, !tokens.isEmpty,
      tokens.upperBound <= tensor.shape[2]
    else {
      throw H3NativeError.invalidTensor(
        "video continuation slice \(tokens) does not fit \(tensor.shape)"
      )
    }
    let source = try tensor.floatValues().map(Float16.init)
    let sourceTime = tensor.shape[2]
    let outputTime = tokens.count
    let plane = tensor.shape[3] * tensor.shape[4]
    var output = [Float16](
      repeating: 0,
      count: tensor.shape[1] * outputTime * plane
    )
    for channel in 0..<tensor.shape[1] {
      for token in 0..<outputTime {
        let sourceOffset = (
          (channel * sourceTime + tokens.lowerBound + token) * plane
        )
        let targetOffset = (channel * outputTime + token) * plane
        output.replaceSubrange(
          targetOffset..<(targetOffset + plane),
          with: source[sourceOffset..<(sourceOffset + plane)]
        )
      }
    }
    return try H3Tensor(
      float16: output,
      shape: [1, tensor.shape[1], outputTime, tensor.shape[3], tensor.shape[4]]
    )
  }
}

struct H3TemporalContinuationState: Sendable {
  let video: H3Tensor
  let audio: H3Tensor?
}

private struct H3StoredTemporalTensor: Codable {
  let shape: [Int]
  let scalarType: H3ScalarType
  let bytes: Data
}

private struct H3TemporalLatentFileV1: Codable {
  let schemaVersion: Int
  let shape: [Int]
  let scalarType: H3ScalarType
  let bytes: Data
}

private struct H3TemporalLatentFileV2: Codable {
  let schemaVersion: Int
  let video: H3StoredTemporalTensor
  let audio: H3StoredTemporalTensor?
}

enum H3TemporalLatentStore {
  static func write(_ state: H3TemporalContinuationState, to url: URL) throws {
    let video = try state.video.converted(to: .float16)
    let audio = try state.audio?.converted(to: .float16)
    let payload = H3TemporalLatentFileV2(
      schemaVersion: 2,
      video: H3StoredTemporalTensor(
        shape: video.shape,
        scalarType: video.scalarType,
        bytes: video.bytes
      ),
      audio: audio.map {
        H3StoredTemporalTensor(
          shape: $0.shape,
          scalarType: $0.scalarType,
          bytes: $0.bytes
        )
      }
    )
    let encoder = PropertyListEncoder()
    encoder.outputFormat = .binary
    try encoder.encode(payload).write(to: url, options: .atomic)
  }

  static func load(from url: URL) throws -> H3TemporalContinuationState {
    let data = try Data(contentsOf: url)
    let propertyList = try PropertyListSerialization.propertyList(
      from: data,
      options: [],
      format: nil
    )
    guard let dictionary = propertyList as? [String: Any],
      let schemaVersion = dictionary["schemaVersion"] as? Int
    else {
      throw H3NativeError.cache(
        "invalid temporal latent state in \(url.path)"
      )
    }
    switch schemaVersion {
    case 1:
      let payload = try PropertyListDecoder().decode(
        H3TemporalLatentFileV1.self,
        from: data
      )
      return H3TemporalContinuationState(
        video: try H3Tensor(
          shape: payload.shape,
          scalarType: payload.scalarType,
          bytes: payload.bytes
        ),
        audio: nil
      )
    case 2:
      let payload = try PropertyListDecoder().decode(
        H3TemporalLatentFileV2.self,
        from: data
      )
      return H3TemporalContinuationState(
        video: try H3Tensor(
          shape: payload.video.shape,
          scalarType: payload.video.scalarType,
          bytes: payload.video.bytes
        ),
        audio: try payload.audio.map {
          try H3Tensor(
            shape: $0.shape,
            scalarType: $0.scalarType,
            bytes: $0.bytes
          )
        }
      )
    default:
      throw H3NativeError.cache(
        "unsupported temporal latent schema \(schemaVersion) in \(url.path)"
      )
    }
  }
}

struct H3AVLatent: Sendable {
  var video: [Float]
  let videoShape: [Int]
  var audio: [Float]
  let audioShape: [Int]

  init(video: [Float], videoShape: [Int], audio: [Float], audioShape: [Int]) throws {
    guard videoShape.reduce(1, *) == video.count,
      audioShape.reduce(1, *) == audio.count
    else {
      throw H3NativeError.invalidTensor("AV latent shape does not match its data")
    }
    self.video = video
    self.videoShape = videoShape
    self.audio = audio
    self.audioShape = audioShape
  }

  func tensors(type: H3ScalarType = .float16) throws -> [String: H3Tensor] {
    let videoTensor = try H3Tensor(float32: video, shape: videoShape).converted(to: type)
    let audioTensor = try H3Tensor(float32: audio, shape: audioShape).converted(to: type)
    return ["video": videoTensor, "audio": audioTensor]
  }
}

enum H3ResMultistep {
  typealias Denoiser = @Sendable (
    _ latent: H3AVLatent, _ sigma: Float, _ step: Int
  ) async throws -> H3AVLatent

  static func sample(
    initial: H3AVLatent,
    sigmas: [Float],
    denoise: Denoiser,
    onStep: (@Sendable (Int, Float) -> Void)? = nil
  ) async throws -> H3AVLatent {
    guard sigmas.count >= 2, sigmas.last == 0 else {
      throw H3NativeError.invalidManifest("res_multistep needs sigmas ending in zero")
    }
    var current = initial
    var oldDenoised: H3AVLatent?
    var oldSigmaDown: Float?
    for index in 0..<(sigmas.count - 1) {
      let sigma = sigmas[index]
      let sigmaDown = sigmas[index + 1]
      let denoised = try await denoise(current, sigma, index)
      try requireMatchingShapes(current, denoised)
      if sigmaDown == 0 || oldDenoised == nil {
        current.video = euler(
          x: current.video,
          denoised: denoised.video,
          sigma: sigma,
          sigmaDown: sigmaDown
        )
        current.audio = euler(
          x: current.audio,
          denoised: denoised.audio,
          sigma: sigma,
          sigmaDown: sigmaDown
        )
      } else if let previous = oldDenoised, let previousSigma = oldSigmaDown {
        let coefficients = secondOrderCoefficients(
          sigma: sigma,
          oldSigmaDown: previousSigma,
          sigmaDown: sigmaDown,
          previousSigma: sigmas[index - 1]
        )
        current.video = secondOrder(
          x: current.video,
          denoised: denoised.video,
          oldDenoised: previous.video,
          coefficients: coefficients
        )
        current.audio = secondOrder(
          x: current.audio,
          denoised: denoised.audio,
          oldDenoised: previous.audio,
          coefficients: coefficients
        )
      }
      oldDenoised = denoised
      oldSigmaDown = sigmaDown
      onStep?(index + 1, sigmaDown)
    }
    return current
  }

  static func euler(
    x: [Float], denoised: [Float], sigma: Float, sigmaDown: Float
  ) -> [Float] {
    guard sigma != 0 else { return denoised }
    let ratio = (sigmaDown - sigma) / sigma
    return zip(x, denoised).map { value, clean in
      value + (value - clean) * ratio
    }
  }

  struct Coefficients: Sendable {
    let expNegativeH: Float
    let h: Float
    let b1: Float
    let b2: Float
  }

  static func secondOrderCoefficients(
    sigma: Float,
    oldSigmaDown: Float,
    sigmaDown: Float,
    previousSigma: Float
  ) -> Coefficients {
    let t = -log(sigma)
    let tOld = -log(oldSigmaDown)
    let tNext = -log(sigmaDown)
    let tPrevious = -log(previousSigma)
    let h = tNext - t
    let c2 = (tPrevious - tOld) / h
    let phi1 = expm1(-h) / (-h)
    let phi2 = (phi1 - 1) / (-h)
    let rawB1 = phi1 - phi2 / c2
    let rawB2 = phi2 / c2
    return Coefficients(
      expNegativeH: exp(-h),
      h: h,
      b1: rawB1.isFinite ? rawB1 : 0,
      b2: rawB2.isFinite ? rawB2 : 0
    )
  }

  static func secondOrder(
    x: [Float],
    denoised: [Float],
    oldDenoised: [Float],
    coefficients: Coefficients
  ) -> [Float] {
    let c = coefficients
    return x.indices.map { index in
      c.expNegativeH * x[index]
        + c.h * (c.b1 * denoised[index] + c.b2 * oldDenoised[index])
    }
  }

  private static func requireMatchingShapes(
    _ left: H3AVLatent, _ right: H3AVLatent
  ) throws {
    guard left.videoShape == right.videoShape,
      left.audioShape == right.audioShape
    else {
      throw H3NativeError.invalidTensor("denoiser changed the AV latent shape")
    }
  }
}

/// Swift implementation of ComfyUI's VP ER-SDE-Solver-3 for H3's CONST flow
/// schedule. Sigma selection is supplied by the external model manifest.
enum H3ERSDE {
  typealias Denoiser = H3ResMultistep.Denoiser

  static func sample(
    initial: H3AVLatent,
    sigmas sourceSigmas: [Float],
    flowShift: Float,
    seed: UInt64,
    sNoise: Float = 1,
    maxStage: Int = 3,
    denoise: Denoiser,
    onStep: (@Sendable (Int, Float) -> Void)? = nil
  ) async throws -> H3AVLatent {
    guard sourceSigmas.count >= 2, sourceSigmas.last == 0 else {
      throw H3NativeError.invalidManifest("er_sde needs sigmas ending in zero")
    }
    guard flowShift > 0, sNoise >= 0, (1...3).contains(maxStage) else {
      throw H3NativeError.invalidManifest("invalid er_sde configuration")
    }

    var sigmas = sourceSigmas
    if sigmas[0] >= 1 {
      // Matches offset_first_sigma_for_snr(..., percent_offset: 1e-4)
      // for ModelSamplingDiscreteFlow/CONST.
      let base = Float(1 - 1e-4)
      sigmas[0] = flowShift * base / (1 + (flowShift - 1) * base)
    }
    let lambdas = sigmas.map { sigma -> Double in
      guard sigma > 0 else { return 0 }
      return Double(sigma) / Double(1 - sigma)
    }

    var current = initial
    var oldDenoised: H3AVLatent?
    var oldDerivative: H3AVLatent?
    // ComfyUI uses the same numeric seed for a fresh device-side generator,
    // while prepare_noise is generated on CPU. Those are independent random
    // streams. SplitMix64 is used for both paths here, so domain-separate the
    // sampler stream instead of accidentally replaying the initial noise.
    var random = H3SplitMix64(seed: seed &+ 1)

    for index in 0..<(sigmas.count - 1) {
      let sigma = sigmas[index]
      let sigmaNext = sigmas[index + 1]
      let denoised = try await denoise(current, sigma, index)
      try requireMatchingShapes(current, denoised)
      let stage = min(maxStage, index + 1)

      if sigmaNext == 0 {
        current = denoised
      } else {
        let lambdaS = lambdas[index]
        let lambdaT = lambdas[index + 1]
        let alphaS = Double(1 - sigma)
        let alphaT = Double(1 - sigmaNext)
        let phiS = noiseScale(lambdaS)
        let phiT = noiseScale(lambdaT)
        let rAlpha = alphaT / alphaS
        let r = phiT / phiS
        current.video = affine(
          current.video, denoised.video,
          left: rAlpha * r, right: alphaT * (1 - r)
        )
        current.audio = affine(
          current.audio, denoised.audio,
          left: rAlpha * r, right: alphaT * (1 - r)
        )

        if stage >= 2, let previous = oldDenoised {
          let dt = lambdaT - lambdaS
          let integrationStep = -dt / 200
          var reciprocalSum = 0.0
          var weightedSum = 0.0
          for point in 0..<200 {
            let position = lambdaT + Double(point) * integrationStep
            let inverse = 1 / noiseScale(position)
            reciprocalSum += inverse
            weightedSum += (position - lambdaS) * inverse
          }
          let integral = reciprocalSum * integrationStep
          let derivative = try difference(
            denoised, previous,
            divisor: lambdaS - lambdas[index - 1]
          )
          let stage2 = alphaT * (dt + integral * phiT)
          addScaled(&current.video, derivative.video, scale: stage2)
          addScaled(&current.audio, derivative.audio, scale: stage2)

          if stage >= 3, let previousDerivative = oldDerivative {
            let secondDerivative = try difference(
              derivative, previousDerivative,
              divisor: (lambdaS - lambdas[index - 2]) / 2
            )
            let weightedIntegral = weightedSum * integrationStep
            let stage3 = alphaT * (dt * dt / 2 + weightedIntegral * phiT)
            addScaled(&current.video, secondDerivative.video, scale: stage3)
            addScaled(&current.audio, secondDerivative.audio, scale: stage3)
          }
          oldDerivative = derivative
        }

        if sNoise > 0 {
          let variance = max(0, lambdaT * lambdaT - lambdaS * lambdaS * r * r)
          let coefficient = alphaT * Double(sNoise) * sqrt(variance)
          addScaled(
            &current.video,
            random.normal(count: current.video.count),
            scale: coefficient
          )
          addScaled(
            &current.audio,
            random.normal(count: current.audio.count),
            scale: coefficient
          )
        }
      }
      oldDenoised = denoised
      onStep?(index + 1, sigmaNext)
    }
    return current
  }

  private static func noiseScale(_ value: Double) -> Double {
    value * (exp(pow(value, 0.3)) + 10)
  }

  private static func affine(
    _ left: [Float], _ right: [Float], left leftScale: Double,
    right rightScale: Double
  ) -> [Float] {
    left.indices.map {
      Float(leftScale * Double(left[$0]) + rightScale * Double(right[$0]))
    }
  }

  private static func addScaled(
    _ destination: inout [Float], _ source: [Float], scale: Double
  ) {
    for index in destination.indices {
      destination[index] += Float(scale * Double(source[index]))
    }
  }

  private static func difference(
    _ left: H3AVLatent, _ right: H3AVLatent, divisor: Double
  ) throws -> H3AVLatent {
    guard divisor != 0 else {
      throw H3NativeError.invalidManifest("er_sde encountered a zero sigma interval")
    }
    try requireMatchingShapes(left, right)
    return try H3AVLatent(
      video: left.video.indices.map {
        Float((Double(left.video[$0]) - Double(right.video[$0])) / divisor)
      },
      videoShape: left.videoShape,
      audio: left.audio.indices.map {
        Float((Double(left.audio[$0]) - Double(right.audio[$0])) / divisor)
      },
      audioShape: left.audioShape
    )
  }

  private static func requireMatchingShapes(
    _ left: H3AVLatent, _ right: H3AVLatent
  ) throws {
    guard left.videoShape == right.videoShape,
      left.audioShape == right.audioShape
    else {
      throw H3NativeError.invalidTensor("denoiser changed the AV latent shape")
    }
  }
}

struct H3SplitMix64: RandomNumberGenerator, Sendable {
  private var state: UInt64

  init(seed: UInt64) { state = seed }

  mutating func next() -> UInt64 {
    state &+= 0x9E3779B97F4A7C15
    var value = state
    value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
    value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
    return value ^ (value >> 31)
  }

  mutating func normal(count: Int) -> [Float] {
    var result = [Float]()
    result.reserveCapacity(count)
    while result.count < count {
      let u1 = max(Double(next()) / Double(UInt64.max), Double.leastNonzeroMagnitude)
      let u2 = Double(next()) / Double(UInt64.max)
      let radius = sqrt(-2 * log(u1))
      let theta = 2 * Double.pi * u2
      result.append(Float(radius * cos(theta)))
      if result.count < count { result.append(Float(radius * sin(theta))) }
    }
    return result
  }
}

private struct H3CachedTensorMetadata: Codable {
  let semantic: String
  let scalarType: H3ScalarType
  let shape: [Int]
  let file: String
}

private struct H3CacheMetadata: Codable {
  let schemaVersion: Int
  let key: String
  let tensors: [H3CachedTensorMetadata]
}

final class H3StageCache: @unchecked Sendable {
  let directory: URL
  private let fileManager = FileManager.default

  init(directory: URL) throws {
    self.directory = directory
    try fileManager.createDirectory(
      at: directory,
      withIntermediateDirectories: true
    )
  }

  func load(stage: String, key: String) throws -> [String: H3Tensor]? {
    let stageDirectory = directory.appendingPathComponent(safe(stage), isDirectory: true)
      .appendingPathComponent(key, isDirectory: true)
    let metadataURL = stageDirectory.appendingPathComponent("metadata.json")
    guard fileManager.fileExists(atPath: metadataURL.path) else { return nil }
    let metadata = try JSONDecoder().decode(
      H3CacheMetadata.self,
      from: Data(contentsOf: metadataURL)
    )
    guard metadata.schemaVersion == 1, metadata.key == key else {
      throw H3NativeError.cache("metadata mismatch in \(stageDirectory.path)")
    }
    var result: [String: H3Tensor] = [:]
    for item in metadata.tensors {
      let data = try Data(contentsOf: stageDirectory.appendingPathComponent(item.file))
      result[item.semantic] = try H3Tensor(
        shape: item.shape,
        scalarType: item.scalarType,
        bytes: data
      )
    }
    return result
  }

  func store(stage: String, key: String, tensors: [String: H3Tensor]) throws {
    let parent = directory.appendingPathComponent(safe(stage), isDirectory: true)
    try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
    let target = parent.appendingPathComponent(key, isDirectory: true)
    if fileManager.fileExists(atPath: target.path) { return }
    let temporary = parent.appendingPathComponent(".\(key).\(UUID().uuidString)")
    try fileManager.createDirectory(at: temporary, withIntermediateDirectories: true)
    do {
      var items: [H3CachedTensorMetadata] = []
      for (index, pair) in tensors.sorted(by: { $0.key < $1.key }).enumerated() {
        let file = String(format: "%03d-%@.bin", index, safe(pair.key))
        try pair.value.bytes.write(
          to: temporary.appendingPathComponent(file),
          options: .atomic
        )
        items.append(
          H3CachedTensorMetadata(
            semantic: pair.key,
            scalarType: pair.value.scalarType,
            shape: pair.value.shape,
            file: file
          )
        )
      }
      let metadata = H3CacheMetadata(schemaVersion: 1, key: key, tensors: items)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(metadata).write(
        to: temporary.appendingPathComponent("metadata.json"),
        options: .atomic
      )
      try fileManager.moveItem(at: temporary, to: target)
    } catch {
      try? fileManager.removeItem(at: temporary)
      throw error
    }
  }

  static func key(parts: [Data]) -> String {
    var hasher = SHA256()
    for part in parts {
      var length = UInt64(part.count).littleEndian
      withUnsafeBytes(of: &length) { hasher.update(data: Data($0)) }
      hasher.update(data: part)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  static func fileDigest(_ url: URL) throws -> Data {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty {
      hasher.update(data: chunk)
    }
    return Data(hasher.finalize())
  }

  /// Fast cache invalidation for very large compiled model bundles. The
  /// converter writes immutable bundle files; relative path, size and mtime
  /// therefore identify a local build without re-hashing tens of gigabytes.
  static func assetFingerprint(_ url: URL) throws -> Data {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(
      atPath: url.path,
      isDirectory: &isDirectory
    ) else {
      throw H3NativeError.missingAsset(url.path)
    }
    let urls: [URL]
    if isDirectory.boolValue {
      let enumerator = FileManager.default.enumerator(
        at: url,
        includingPropertiesForKeys: [
          .isRegularFileKey, .fileSizeKey, .contentModificationDateKey,
        ],
        options: [.skipsHiddenFiles]
      )
      urls = (enumerator?.allObjects as? [URL] ?? []).sorted {
        $0.path < $1.path
      }
    } else {
      urls = [url]
    }
    var hasher = SHA256()
    for file in urls {
      let values = try file.resourceValues(forKeys: [
        .isRegularFileKey, .fileSizeKey, .contentModificationDateKey,
      ])
      guard values.isRegularFile == true else { continue }
      let relative = file.path.replacingOccurrences(of: url.path, with: "")
      hasher.update(data: Data(relative.utf8))
      var size = UInt64(values.fileSize ?? 0).littleEndian
      withUnsafeBytes(of: &size) { hasher.update(data: Data($0)) }
      let seconds = values.contentModificationDate?.timeIntervalSince1970 ?? 0
      var timeBits = seconds.bitPattern.littleEndian
      withUnsafeBytes(of: &timeBits) { hasher.update(data: Data($0)) }
    }
    return Data(hasher.finalize())
  }

  private func safe(_ value: String) -> String {
    String(value.map { character in
      character.isLetter || character.isNumber || character == "-" || character == "_"
        ? character : "_"
    })
  }
}

extension JSONEncoder {
  static var h3Stable: JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return encoder
  }
}
