import Foundation

/// Swift-side ref2va graph for the 10Eros-Max H3 TURBO checkpoint.
///
/// Core AI executes the learned projections, 50 DiT blocks and final heads;
/// this type performs the model's deterministic token packing, position/RoPE
/// construction and separate video/audio flow schedules.
@available(macOS 27.0, *)
final class TenErosMaxH3DenoiserComposite: @unchecked Sendable {
  /// All auxiliary projections/heads are invoked serially and scoped to one
  /// prediction so only one specialized auxiliary program is resident.
  static let maximumResidentAuxiliaryModels = 1

  struct Prepared: Sendable {
    let textHidden: H3Tensor
    let referenceAudioHidden: H3Tensor
    let referenceVideoHidden: H3Tensor
    let tokenTags: [Int32]
    let ropeCosine: H3Tensor
    let ropeSine: H3Tensor
    let textRows: Int
    let referenceAudioRows: Int
    let referenceVideoRows: Int
    let targetAudioRows: Int
    let targetVideoRows: Int
    let targetVideoShape: [Int]
    let targetAudioShape: [Int]

    var totalRows: Int {
      textRows + referenceAudioRows + referenceVideoRows
        + targetAudioRows + targetVideoRows
    }
  }

  private let manifest: H3DenoiserCompositeManifest
  private let blocks: H3CoreAIBlockSequence
  private let adalnTable: [Float]
  private let ropeInverseFrequency: [Float]
  private let baseDirectory: URL

  init(
    manifest: H3DenoiserCompositeManifest,
    baseDirectory: URL,
    onLoad: (@Sendable (Int, Int) -> Void)? = nil
  ) async throws {
    self.manifest = manifest
    self.baseDirectory = baseDirectory
    blocks = try await H3CoreAIBlockSequence(
      manifests: manifest.blocks,
      baseDirectory: baseDirectory,
      onLoad: onLoad
    )

    adalnTable = try Self.readFloat32Asset(
      manifest.adalnTableAsset, relativeTo: baseDirectory
    )
    ropeInverseFrequency = try Self.readFloat32Asset(
      manifest.ropeInverseFrequencyAsset, relativeTo: baseDirectory
    )
    guard adalnTable.count == 1025 * 8 else {
      throw H3NativeError.invalidManifest(
        "10Eros adaln table has \(adalnTable.count) values, expected 8200"
      )
    }
    guard ropeInverseFrequency.count == 16 else {
      throw H3NativeError.invalidManifest(
        "10Eros RoPE table has \(ropeInverseFrequency.count) values, expected 16"
      )
    }
  }

  func prepare(
    context: H3Tensor,
    tokenTags: H3Tensor,
    referenceVideoLatent: H3Tensor,
    referenceAudioLatent: H3Tensor,
    targetVideoShape: [Int],
    targetAudioShape: [Int],
    seed: UInt64,
    visualConditionNoiseAug: Float,
    audioConditionNoiseAug: Float
  ) async throws -> Prepared {
    try Self.validateVideoShape(targetVideoShape, semantic: "target video")
    try Self.validateAudioShape(targetAudioShape, semantic: "target audio")
    try Self.validateVideoShape(referenceVideoLatent.shape, semantic: "reference video")
    try Self.validateAudioShape(referenceAudioLatent.shape, semantic: "reference audio")

    guard context.shape.last == 5120 else {
      throw H3NativeError.invalidTensor(
        "Qwen context must end in 5120, got \(context.shape)"
      )
    }
    let textRows = context.elementCount / 5120
    let tags = try Self.tokenTagValues(tokenTags)
    guard tags.count == textRows, tags.allSatisfy({ $0 == 0 || $0 == 1 }) else {
      throw H3NativeError.invalidTensor(
        "Qwen token tags must contain one vision/text tag per context row"
      )
    }
    let flatContext = try H3Tensor(
      float32: context.floatValues(), shape: [textRows, 5120]
    ).converted(to: .bfloat16)
    guard let refined = try await predictOnce(
      name: "10erosTextRefiner",
      stage: manifest.textRefiner,
      inputs: ["context": flatContext]
    )["textHidden"] else {
      throw H3NativeError.missingTensor("10Eros textHidden")
    }

    var referenceVideoRows = try Self.patchifyVideo(referenceVideoLatent)
    if visualConditionNoiseAug < 1 {
      var random = H3SplitMix64(seed: seed)
      let noise = random.normal(count: referenceVideoRows.count)
      let keep = max(0, visualConditionNoiseAug)
      for index in referenceVideoRows.indices {
        referenceVideoRows[index] = keep * referenceVideoRows[index]
          + (1 - keep) * noise[index]
      }
    }
    var referenceAudioRows = try Self.packAudio(referenceAudioLatent)
    if audioConditionNoiseAug < 1 {
      var random = H3SplitMix64(seed: seed &+ 1)
      let noise = random.normal(count: referenceAudioRows.count)
      let keep = max(0, audioConditionNoiseAug)
      for index in referenceAudioRows.indices {
        referenceAudioRows[index] = keep * referenceAudioRows[index]
          + (1 - keep) * noise[index]
      }
    }
    let refVideoRowCount = referenceVideoRows.count / 96
    let refAudioRowCount = referenceAudioRows.count / 32
    let projectedReferenceVideo = try await projectVideo(
      referenceVideoRows, rows: refVideoRowCount
    )
    let projectedReferenceAudio = try await projectAudio(
      referenceAudioRows, rows: refAudioRowCount
    )

    let targetVideoRows = Self.videoRowCount(targetVideoShape)
    let targetAudioRows = Self.audioRowCount(targetAudioShape)
    let totalRows = textRows + refAudioRowCount + refVideoRowCount
      + targetAudioRows + targetVideoRows
    guard totalRows <= manifest.dynamicMaximumTokens else {
      throw H3NativeError.invalidTensor(
        "10Eros packed sequence \(totalRows) exceeds \(manifest.dynamicMaximumTokens)"
      )
    }
    let positions = try Self.packedPositions(
      textRows: textRows,
      referenceVideoShape: referenceVideoLatent.shape,
      referenceAudioShape: referenceAudioLatent.shape,
      targetVideoShape: targetVideoShape,
      targetAudioShape: targetAudioShape
    )
    guard positions.count == totalRows * 3 else {
      throw H3NativeError.invalidTensor("10Eros packed-position length mismatch")
    }
    let rope = try makeRoPE(positions: positions, rows: totalRows)
    return Prepared(
      textHidden: refined,
      referenceAudioHidden: projectedReferenceAudio,
      referenceVideoHidden: projectedReferenceVideo,
      tokenTags: tags,
      ropeCosine: rope.cosine,
      ropeSine: rope.sine,
      textRows: textRows,
      referenceAudioRows: refAudioRowCount,
      referenceVideoRows: refVideoRowCount,
      targetAudioRows: targetAudioRows,
      targetVideoRows: targetVideoRows,
      targetVideoShape: targetVideoShape,
      targetAudioShape: targetAudioShape
    )
  }

  /// Returns x0 in the sampler's packed video-sigma coordinate system.
  func denoise(
    _ latent: H3AVLatent,
    sigma videoSigma: Float,
    prepared: Prepared,
    videoShift: Float,
    audioShift: Float
  ) async throws -> H3AVLatent {
    guard latent.videoShape == prepared.targetVideoShape,
      latent.audioShape == prepared.targetAudioShape
    else {
      throw H3NativeError.invalidTensor("10Eros latent shape changed during sampling")
    }
    let sigmaV = max(videoSigma, 1e-6)
    let sigmaA = Self.timeShiftSigma(
      sigmaV, from: videoShift, to: audioShift
    )
    let carry = sigmaA / sigmaV
    let carriedAudio = latent.audio.map { $0 * carry }
    let targetVideoRows = try Self.patchifyVideo(
      values: latent.video, shape: latent.videoShape
    )
    let targetAudioRows = try Self.packAudio(
      values: carriedAudio, shape: latent.audioShape
    )
    let projectedTargetAudio = try await projectAudio(
      targetAudioRows, rows: prepared.targetAudioRows
    )
    let projectedTargetVideo = try await projectVideo(
      targetVideoRows, rows: prepared.targetVideoRows
    )
    var hidden = try Self.concatenateRows([
      prepared.textHidden,
      prepared.referenceAudioHidden,
      prepared.referenceVideoHidden,
      projectedTargetAudio,
      projectedTargetVideo,
    ], width: 5376)

    let time = try makeTimeInputs(
      sigmaV: sigmaV,
      sigmaA: sigmaA,
      prepared: prepared
    )
    hidden = try await blocks.predict(
      hiddenStates: hidden,
      timestepCoordinates: time.coordinates,
      modulationWeights: time.weights,
      ropeCosine: prepared.ropeCosine,
      ropeSine: prepared.ropeSine
    )

    let targetAudioStart = prepared.textRows + prepared.referenceAudioRows
      + prepared.referenceVideoRows
    let targetVideoStart = targetAudioStart + prepared.targetAudioRows
    let targetAudioHidden = try Self.sliceRows(
      hidden, start: targetAudioStart, count: prepared.targetAudioRows,
      width: 5376
    )
    let targetVideoHidden = try Self.sliceRows(
      hidden, start: targetVideoStart, count: prepared.targetVideoRows,
      width: 5376
    )
    guard let videoRows = try await predictOnce(
      name: "10erosFinalVideo",
      stage: manifest.finalVideo,
      inputs: [
        "hiddenStates": targetVideoHidden,
        "timestepCoordinate": time.videoCoordinate,
      ]
    )["videoRows"],
      let audioRows = try await predictOnce(
        name: "10erosFinalAudio",
        stage: manifest.finalAudio,
        inputs: [
          "hiddenStates": targetAudioHidden,
          "timestepCoordinate": time.audioCoordinate,
        ]
      )["audioRows"]
    else {
      throw H3NativeError.missingTensor("10Eros final video/audio rows")
    }
    let videoVelocity = try Self.unpatchifyVideo(
      videoRows.floatValues(), shape: latent.videoShape
    )
    let audioVelocity = try Self.unpackAudio(
      audioRows.floatValues(), shape: latent.audioShape
    )

    // _forward returns the negative heads.  ModelSamplingAV then differentiates
    // the carried audio coordinate with respect to video sigma.
    let audioScale = videoShift / audioShift
    let audioVelocityScale = 1 + (audioScale - 1) * sigmaA
    var denoisedVideo = latent.video
    var denoisedAudio = latent.audio
    for index in denoisedVideo.indices {
      denoisedVideo[index] += sigmaV * videoVelocity[index]
    }
    for index in denoisedAudio.indices {
      let wrappedModelOutput = (1 - audioScale) * carriedAudio[index]
        - audioVelocityScale * audioVelocity[index]
      denoisedAudio[index] -= sigmaV * wrappedModelOutput
    }
    return try H3AVLatent(
      video: denoisedVideo,
      videoShape: latent.videoShape,
      audio: denoisedAudio,
      audioShape: latent.audioShape
    )
  }

  private struct TimeInputs {
    let coordinates: H3Tensor
    let weights: H3Tensor
    let videoCoordinate: H3Tensor
    let audioCoordinate: H3Tensor
  }

  private func makeTimeInputs(
    sigmaV: Float,
    sigmaA: Float,
    prepared: Prepared
  ) throws -> TimeInputs {
    let tv = 1 - sigmaV
    let ta = 1 - sigmaA
    let referenceVideoT = max(tv, 0.999)
    let referenceAudioT = max(ta, 1.0)
    var unique = Array(Set([tv, ta, referenceVideoT, referenceAudioT])).sorted()
    guard unique.count <= 4 else {
      throw H3NativeError.invalidTensor("10Eros produced more than four timestep rows")
    }
    while unique.count < 4 { unique.append(unique.last ?? 1) }
    let coordinateValues = unique.flatMap(interpolatedTimeCoordinate)
    let coordinates = try H3Tensor(
      float32: coordinateValues, shape: [4, 8]
    ).converted(to: .bfloat16)
    func row(_ value: Float) -> Int {
      unique.firstIndex(where: { abs($0 - value) <= 1e-7 }) ?? 0
    }
    var indices: [Int] = []
    indices.reserveCapacity(prepared.totalRows)
    let videoRow = row(tv)
    let audioRow = row(ta)
    for tag in prepared.tokenTags {
      indices.append(videoRow * 3 + Int(tag))
    }
    indices.append(
      contentsOf: repeatElement(
        row(referenceAudioT) * 3 + 2,
        count: prepared.referenceAudioRows
      )
    )
    indices.append(
      contentsOf: repeatElement(
        row(referenceVideoT) * 3,
        count: prepared.referenceVideoRows
      )
    )
    indices.append(
      contentsOf: repeatElement(
        audioRow * 3 + 2,
        count: prepared.targetAudioRows
      )
    )
    indices.append(
      contentsOf: repeatElement(videoRow * 3, count: prepared.targetVideoRows)
    )
    var oneHot = [UInt16](repeating: 0, count: indices.count * 12)
    let one = Self.bfloat16Bits(1)
    for (token, index) in indices.enumerated() {
      oneHot[token * 12 + index] = one
    }
    return try TimeInputs(
      coordinates: coordinates,
      weights: H3Tensor(
        bfloat16Raw: oneHot, shape: [prepared.totalRows, 12]
      ),
      videoCoordinate: H3Tensor(
        float32: interpolatedTimeCoordinate(tv), shape: [1, 8]
      ).converted(to: .bfloat16),
      audioCoordinate: H3Tensor(
        float32: interpolatedTimeCoordinate(ta), shape: [1, 8]
      ).converted(to: .bfloat16)
    )
  }

  private func interpolatedTimeCoordinate(_ value: Float) -> [Float] {
    let position = min(1, max(0, value)) * 1024
    let lower = min(1023, Int(floor(position)))
    let fraction = position - Float(lower)
    return (0..<8).map { column in
      let left = adalnTable[lower * 8 + column]
      let right = adalnTable[(lower + 1) * 8 + column]
      return left + (right - left) * fraction
    }
  }

  private func projectVideo(_ values: [Float], rows: Int) async throws -> H3Tensor {
    guard let result = try await predictOnce(
      name: "10erosVideoProjection",
      stage: manifest.videoProjection,
      inputs: [
        "videoRows": try H3Tensor(float32: values, shape: [rows, 96])
      ]
    )["videoHidden"] else {
      throw H3NativeError.missingTensor("10Eros videoHidden")
    }
    return result
  }

  private func projectAudio(_ values: [Float], rows: Int) async throws -> H3Tensor {
    guard let result = try await predictOnce(
      name: "10erosAudioProjection",
      stage: manifest.audioProjection,
      inputs: [
        "audioRows": try H3Tensor(float32: values, shape: [rows, 32])
      ]
    )["audioHidden"] else {
      throw H3NativeError.missingTensor("10Eros audioHidden")
    }
    return result
  }

  /// Auxiliary Core AI programs are intentionally scoped to one prediction.
  /// Keeping the five runners as properties retained their specialized GPU
  /// programs while the much larger DiT sequence was running. The returned
  /// H3Tensor owns copied bytes, so the runner can be released before the next
  /// auxiliary model or DiT asset is loaded.
  private func predictOnce(
    name: String,
    stage: H3StageManifest,
    inputs: [String: H3Tensor]
  ) async throws -> [String: H3Tensor] {
    try Task.checkCancellation()
    let output = try await Self.runAuxiliaryStage(
      name: name,
      stage: stage,
      inputs: inputs,
      baseDirectory: baseDirectory
    )
    await Task.yield()
    return output
  }

  /// A separate async frame makes the runner's lifetime end before the caller
  /// yields or loads the next model.
  private static func runAuxiliaryStage(
    name: String,
    stage: H3StageManifest,
    inputs: [String: H3Tensor],
    baseDirectory: URL
  ) async throws -> [String: H3Tensor] {
    let runner = try await H3StageRunner(
      name: name,
      manifest: stage,
      baseDirectory: baseDirectory
    )
    return try await runner.predict(inputs)
  }

  private func makeRoPE(positions: [Float], rows: Int) throws
    -> (cosine: H3Tensor, sine: H3Tensor)
  {
    var cosine = [Float]()
    var sine = [Float]()
    cosine.reserveCapacity(rows * 48)
    sine.reserveCapacity(rows * 48)
    for token in 0..<rows {
      for axis in 0..<3 {
        let position = positions[token * 3 + axis]
        for frequency in ropeInverseFrequency {
          let angle = position * frequency
          cosine.append(cos(angle))
          sine.append(sin(angle))
        }
      }
    }
    return try (
      H3Tensor(float32: cosine, shape: [rows, 48]).converted(to: .bfloat16),
      H3Tensor(float32: sine, shape: [rows, 48]).converted(to: .bfloat16)
    )
  }

  private static func packedPositions(
    textRows: Int,
    referenceVideoShape: [Int],
    referenceAudioShape: [Int],
    targetVideoShape: [Int],
    targetAudioShape: [Int]
  ) throws -> [Float] {
    let refT = referenceVideoShape[2]
    let refH = referenceVideoShape[3]
    let refW = referenceVideoShape[4]
    let refAudioT = referenceAudioShape[3]
    let targetT = targetVideoShape[2]
    let targetH = targetVideoShape[3]
    let targetW = targetVideoShape[4]
    let targetAudioT = targetAudioShape[3]
    var result: [Float] = []
    result.reserveCapacity(
      (textRows + videoRowCount(referenceVideoShape)
        + audioRowCount(referenceAudioShape) + videoRowCount(targetVideoShape)
        + audioRowCount(targetAudioShape)) * 3
    )
    for row in 0..<textRows {
      result.append(Float(row)); result.append(0); result.append(0)
    }

    var cursor = Float(textRows)
    let refFrame = frameGrid(height: refH, width: refW)
    appendAudioGrid(
      to: &result, cursor: cursor, time: refAudioT,
      lowW: refFrame.w.first ?? 0, highW: refFrame.w.last ?? 0
    )
    appendVideoGrid(to: &result, cursor: cursor, time: refT, frame: refFrame.rows)
    cursor += max(Float(refAudioT), videoSpan(refT))

    let targetFrame = frameGrid(height: targetH, width: targetW)
    appendAudioGrid(
      to: &result, cursor: cursor, time: targetAudioT,
      lowW: targetFrame.w.first ?? 0, highW: targetFrame.w.last ?? 0
    )
    appendVideoGrid(
      to: &result, cursor: cursor, time: targetT, frame: targetFrame.rows
    )
    return result
  }

  private static func frameGrid(height: Int, width: Int)
    -> (rows: [(Float, Float)], w: [Float])
  {
    let area = sqrt(Float(height * width))
    let h = axis(dim: height, patch: 2, squareRootArea: area)
    let w = axis(dim: width, patch: 2, squareRootArea: area)
    var rows: [(Float, Float)] = []
    rows.reserveCapacity(h.count * w.count)
    for y in h { for x in w { rows.append((y, x)) } }
    return (rows, w)
  }

  private static func axis(
    dim: Int, patch: Int, squareRootArea: Float
  ) -> [Float] {
    let ratio = Float(dim) / squareRootArea
    let count = dim / patch
    return (0..<count).map {
      (Float($0) * ratio / Float(count) + (1 - ratio) / 2) * 32
    }
  }

  private static func appendAudioGrid(
    to result: inout [Float], cursor: Float, time: Int,
    lowW: Float, highW: Float
  ) {
    for channel in 0..<2 {
      for index in 0..<time {
        result.append(cursor + Float(index))
        result.append(0)
        result.append(channel == 0 ? lowW : highW)
      }
    }
  }

  private static func appendVideoGrid(
    to result: inout [Float], cursor: Float, time: Int,
    frame: [(Float, Float)]
  ) {
    var elapsed: Float = 0
    for index in 0..<time {
      let position = cursor + elapsed
      for (h, w) in frame {
        result.append(position); result.append(h); result.append(w)
      }
      elapsed += Float([1, 4, 4, 4, 4][index % 5]) * 5 / 3
    }
  }

  private static func videoSpan(_ time: Int) -> Float {
    (0..<time).reduce(0) {
      $0 + Float([1, 4, 4, 4, 4][$1 % 5]) * 5 / 3
    }
  }

  private static func patchifyVideo(_ tensor: H3Tensor) throws -> [Float] {
    try patchifyVideo(values: tensor.floatValues(), shape: tensor.shape)
  }

  private static func patchifyVideo(values: [Float], shape: [Int]) throws
    -> [Float]
  {
    try validateVideoShape(shape, semantic: "video")
    let channels = shape[1], time = shape[2], height = shape[3], width = shape[4]
    var rows = [Float]()
    rows.reserveCapacity(videoRowCount(shape) * 96)
    for t in 0..<time {
      for h in stride(from: 0, to: height, by: 2) {
        for w in stride(from: 0, to: width, by: 2) {
          for channel in 0..<channels {
            for ph in 0..<2 { for pw in 0..<2 {
              let index = (((channel * time + t) * height + h + ph) * width + w + pw)
              rows.append(values[index])
            }}
          }
        }
      }
    }
    return rows
  }

  private static func unpatchifyVideo(_ rows: [Float], shape: [Int]) throws
    -> [Float]
  {
    try validateVideoShape(shape, semantic: "video output")
    let channels = shape[1], time = shape[2], height = shape[3], width = shape[4]
    guard rows.count == videoRowCount(shape) * 96 else {
      throw H3NativeError.invalidTensor("10Eros video-head row count mismatch")
    }
    var values = [Float](repeating: 0, count: shape.reduce(1, *))
    var offset = 0
    for t in 0..<time {
      for h in stride(from: 0, to: height, by: 2) {
        for w in stride(from: 0, to: width, by: 2) {
          for channel in 0..<channels {
            for ph in 0..<2 { for pw in 0..<2 {
              let index = (((channel * time + t) * height + h + ph) * width + w + pw)
              values[index] = rows[offset]
              offset += 1
            }}
          }
        }
      }
    }
    return values
  }

  private static func packAudio(_ tensor: H3Tensor) throws -> [Float] {
    try packAudio(values: tensor.floatValues(), shape: tensor.shape)
  }

  private static func packAudio(values: [Float], shape: [Int]) throws -> [Float] {
    try validateAudioShape(shape, semantic: "audio")
    let channels = shape[1], stereo = shape[2], time = shape[3]
    var rows = [Float]()
    rows.reserveCapacity(stereo * time * channels)
    for side in 0..<stereo { for t in 0..<time { for channel in 0..<channels {
      rows.append(values[(channel * stereo + side) * time + t])
    }}}
    return rows
  }

  private static func unpackAudio(_ rows: [Float], shape: [Int]) throws
    -> [Float]
  {
    try validateAudioShape(shape, semantic: "audio output")
    let channels = shape[1], stereo = shape[2], time = shape[3]
    guard rows.count == stereo * time * channels else {
      throw H3NativeError.invalidTensor("10Eros audio-head row count mismatch")
    }
    var values = [Float](repeating: 0, count: shape.reduce(1, *))
    var offset = 0
    for side in 0..<stereo { for t in 0..<time { for channel in 0..<channels {
      values[(channel * stereo + side) * time + t] = rows[offset]
      offset += 1
    }}}
    return values
  }

  private static func concatenateRows(_ tensors: [H3Tensor], width: Int) throws
    -> H3Tensor
  {
    guard tensors.allSatisfy({
      $0.scalarType == .bfloat16 && $0.shape.count == 2 && $0.shape[1] == width
    }) else {
      throw H3NativeError.invalidTensor("10Eros hidden tensors are incompatible")
    }
    var data = Data()
    data.reserveCapacity(tensors.reduce(0) { $0 + $1.bytes.count })
    for tensor in tensors { data.append(tensor.bytes) }
    return try H3Tensor(
      shape: [tensors.reduce(0) { $0 + $1.shape[0] }, width],
      scalarType: .bfloat16,
      bytes: data
    )
  }

  private static func sliceRows(
    _ tensor: H3Tensor, start: Int, count: Int, width: Int
  ) throws -> H3Tensor {
    guard tensor.scalarType == .bfloat16,
      tensor.shape == [tensor.shape[0], width],
      start >= 0, count > 0, start + count <= tensor.shape[0]
    else {
      throw H3NativeError.invalidTensor("invalid 10Eros hidden-state slice")
    }
    let byteStart = start * width * H3ScalarType.bfloat16.byteCount
    let byteCount = count * width * H3ScalarType.bfloat16.byteCount
    return try H3Tensor(
      shape: [count, width], scalarType: .bfloat16,
      bytes: tensor.bytes.subdata(in: byteStart..<(byteStart + byteCount))
    )
  }

  private static func tokenTagValues(_ tensor: H3Tensor) throws -> [Int32] {
    switch tensor.scalarType {
    case .int32: return try tensor.int32Values()
    case .int64:
      return tensor.bytes.withUnsafeBytes {
        $0.bindMemory(to: Int64.self).map(Int32.init(truncatingIfNeeded:))
      }
    default:
      throw H3NativeError.invalidTensor("tokenTags must be int32 or int64")
    }
  }

  private static func validateVideoShape(_ shape: [Int], semantic: String) throws {
    guard shape.count == 5, shape[0] == 1, shape[1] == 24,
      shape[3] % 2 == 0, shape[4] % 2 == 0
    else {
      throw H3NativeError.invalidTensor(
        "\(semantic) must be [1,24,T,even H,even W], got \(shape)"
      )
    }
  }

  private static func validateAudioShape(_ shape: [Int], semantic: String) throws {
    guard shape.count == 4, shape[0] == 1, shape[1] == 32, shape[2] == 2 else {
      throw H3NativeError.invalidTensor(
        "\(semantic) must be [1,32,2,T], got \(shape)"
      )
    }
  }

  private static func videoRowCount(_ shape: [Int]) -> Int {
    shape[2] * (shape[3] / 2) * (shape[4] / 2)
  }

  private static func audioRowCount(_ shape: [Int]) -> Int { shape[2] * shape[3] }

  private static func timeShiftSigma(
    _ sigma: Float, from fromShift: Float, to toShift: Float
  ) -> Float {
    let base = sigma / (fromShift + sigma * (1 - fromShift))
    return toShift * base / (1 + (toShift - 1) * base)
  }

  private static func bfloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let bias = UInt32(0x7FFF) + ((bits >> 16) & 1)
    return UInt16(truncatingIfNeeded: (bits &+ bias) >> 16)
  }

  private static func readFloat32Asset(_ path: String, relativeTo directory: URL) throws
    -> [Float]
  {
    let url = URL(fileURLWithPath: path, relativeTo: directory).standardizedFileURL
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw H3NativeError.missingAsset(url.path)
    }
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    guard data.count % MemoryLayout<Float>.stride == 0 else {
      throw H3NativeError.invalidTensor("invalid float32 asset \(url.path)")
    }
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  }
}
