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

struct H3PipelineManifest: Codable, Sendable {
  let schemaVersion: Int
  let modelIdentifier: String
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
  let durationSeconds: Double
  let seed: UInt64
  let backend: H3BackendKind?
  let preserveSourceAudioWhenDecoderIsUnavailable: Bool?

  func validate() throws {
    let video = input?.trimmingCharacters(in: .whitespacesAndNewlines)
    let images = inputImages?.filter {
      !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    } ?? []
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
    guard !output.isEmpty, !cacheDirectory.isEmpty else {
      throw H3NativeError.invalidJob("output and cacheDirectory are required")
    }
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw H3NativeError.invalidJob("prompt is empty")
    }
    guard width >= 32, height >= 32, width % 32 == 0, height % 32 == 0 else {
      throw H3NativeError.invalidJob(
        "width and height must be positive multiples of 32"
      )
    }
    guard durationSeconds >= 2, durationSeconds <= 15 else {
      throw H3NativeError.invalidJob("reference duration must be 2...15 seconds")
    }
  }
}

enum H3Geometry {
  static let framesPerSecond = 24
  static let audioLatentFramesPerSecond = 40
  static let identityVisionBlocks = 10

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

  static func qwenVideoSampleIndices(frameCount: Int) -> [Int] {
    guard frameCount > 0 else { return [] }
    return Array(stride(from: 0, to: frameCount, by: 12))
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
/// schedule. The model author's preferred 10Eros-Max TURBO recipe uses this
/// solver with a hand-tuned seven-step sigma schedule.
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
