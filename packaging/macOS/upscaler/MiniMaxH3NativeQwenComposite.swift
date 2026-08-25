import Foundation

@available(macOS 27.0, *)
final class H3QwenCompositeEncoder: @unchecked Sendable {
  private static let hiddenSize = 5120
  private static let headDimension = 128
  private static let ropeTheta: Float = 5_000_000

  private let manifest: H3QwenCompositeManifest
  private let baseDirectory: URL
  private let onProgress: @Sendable (Double, String) -> Void

  init(
    manifest: H3QwenCompositeManifest,
    baseDirectory: URL,
    onProgress: @escaping @Sendable (Double, String) -> Void
  ) {
    self.manifest = manifest
    self.baseDirectory = baseDirectory
    self.onProgress = onProgress
  }

  func encode(_ presentation: H3QwenPresentation) async throws
    -> [String: H3Tensor]
  {
    let sequenceLength = presentation.inputIDs.shape[1]
    guard sequenceLength == manifest.sequenceLength else {
      throw H3NativeError.invalidTensor(
        "Qwen sequence is \(sequenceLength), compiled model needs \(manifest.sequenceLength)"
      )
    }
    let tokenIDs = try presentation.inputIDs.int32Values()
    let visualPositions = tokenIDs.indices.filter {
      tokenIDs[$0] == H3QwenPresentation.imagePad
    }
    let hasVision = !visualPositions.isEmpty
    let actualVisionBlocks = hasVision
      ? visualPositions.count / manifest.visualTokensPerBlock : 0
    guard !hasVision
      || (visualPositions.count % manifest.visualTokensPerBlock == 0
        && actualVisionBlocks > 0
        && actualVisionBlocks <= manifest.visionBlockBatch)
    else {
      throw H3NativeError.invalidTensor(
        "Qwen presentation has \(visualPositions.count) visual tokens; expected 1...\(manifest.visionBlockBatch) complete blocks"
      )
    }

    onProgress(0.02, "Qwen token embedding")
    let tokenOutput = try await predict(
      name: "qwen.tokenEmbedding",
      stage: manifest.tokenEmbedding,
      inputs: ["inputIDs": presentation.inputIDs]
    )
    guard var hiddenStates = tokenOutput["tokenEmbeddings"] else {
      throw H3NativeError.missingTensor("qwen.tokenEmbeddings")
    }

    var deepstack: [H3Tensor] = []
    if hasVision {
      let patchShape = [
        manifest.visionBlockBatch,
        manifest.visionPatchesPerBlock,
        1536,
      ]
      let patches = try paddedVisionPatches(
        presentation.pixelValues,
        actualBlocks: actualVisionBlocks,
        shape: patchShape
      )
      onProgress(0.04, "Qwen vision patch embedding")
      let patchOutput = try await predictVisionBatchStage(
        name: "qwen.visionPatch",
        stage: manifest.visionPatch,
        inputSemantic: "pixelPatches",
        outputSemantic: "visionHidden",
        input: patches
      )
      guard var visionHidden = patchOutput["visionHidden"] else {
        throw H3NativeError.missingTensor("qwen.visionHidden")
      }

      for (index, stage) in manifest.visionBlocks.enumerated() {
        onProgress(
          0.05 + Double(index + 1) / Double(manifest.visionBlocks.count) * 0.23,
          "Qwen vision layer \(index + 1)/\(manifest.visionBlocks.count)"
        )
        let output = try await predictVisionBatchStage(
          name: "qwen.visionBlock\(index)",
          stage: stage,
          inputSemantic: "visionHidden",
          outputSemantic: "visionHiddenOut",
          input: visionHidden
        )
        guard let next = output["visionHiddenOut"] else {
          throw H3NativeError.missingTensor("qwen.visionBlock\(index).output")
        }
        visionHidden = next
        if let deepstackIndex = manifest.deepstackVisionBlockIndices.firstIndex(
          of: index
        ) {
          let merged = try await predictVisionBatchStage(
            name: "qwen.deepstack\(deepstackIndex)",
            stage: manifest.visionDeepstackMergers[deepstackIndex],
            inputSemantic: "visionHidden",
            outputSemantic: "deepstack",
            input: visionHidden
          )
          guard let tensor = merged["deepstack"] else {
            throw H3NativeError.missingTensor("qwen.deepstack\(deepstackIndex)")
          }
          deepstack.append(
            try prefixVisualRows(tensor, count: visualPositions.count)
          )
        }
      }
      guard deepstack.count == 3 else {
        throw H3NativeError.missingTensor("Qwen DeepStack outputs")
      }

      let visionOutput = try await predictVisionBatchStage(
        name: "qwen.visionFinalMerger",
        stage: manifest.visionFinalMerger,
        inputSemantic: "visionHidden",
        outputSemantic: "visionMerged",
        input: visionHidden
      )
      guard let mergedVision = visionOutput["visionMerged"] else {
        throw H3NativeError.missingTensor("qwen.visionMerged")
      }
      hiddenStates = try replacingVisualTokens(
        in: hiddenStates,
        with: prefixVisualRows(mergedVision, count: visualPositions.count),
        at: visualPositions
      )
    }

    let rope = try makeLanguageRoPE(positionIDs: presentation.positionIDs)
    for (index, stage) in manifest.languageLayers.enumerated() {
      onProgress(
        0.30 + Double(index + 1) / Double(manifest.languageLayers.count) * 0.68,
        "Qwen language layer \(index + 1)/\(manifest.languageLayers.count)"
      )
      let output = try await predict(
        name: "qwen.languageLayer\(index)",
        stage: stage,
        inputs: [
          "hiddenStates": hiddenStates,
          "ropeCosine": rope.cosine,
          "ropeSine": rope.sine,
        ]
      )
      guard let next = output["hiddenStatesOut"] else {
        throw H3NativeError.missingTensor("qwen.languageLayer\(index).output")
      }
      hiddenStates = next
      if hasVision,
        let deepstackIndex = manifest.deepstackLanguageLayerIndices.firstIndex(
        of: index
      ) {
        hiddenStates = try addingVisualTokens(
          to: hiddenStates,
          from: deepstack[deepstackIndex],
          at: visualPositions
        )
      }
    }
    let effectiveLength = presentation.effectiveSequenceLength
    guard effectiveLength > 0, effectiveLength <= manifest.sequenceLength else {
      throw H3NativeError.invalidTensor(
        "Qwen effective sequence \(effectiveLength) is outside 1...\(manifest.sequenceLength)"
      )
    }
    let contextBytes = effectiveLength * Self.hiddenSize
      * hiddenStates.scalarType.byteCount
    let tagBytes = effectiveLength * presentation.tokenTags.scalarType.byteCount
    let context = try H3Tensor(
      shape: [1, effectiveLength, Self.hiddenSize],
      scalarType: hiddenStates.scalarType,
      bytes: Data(hiddenStates.bytes.prefix(contextBytes))
    )
    let tokenTags = try H3Tensor(
      shape: [1, effectiveLength],
      scalarType: presentation.tokenTags.scalarType,
      bytes: Data(presentation.tokenTags.bytes.prefix(tagBytes))
    )
    onProgress(1.0, "Qwen multimodal condition complete")
    return ["context": context, "tokenTags": tokenTags]
  }

  private func predict(
    name: String,
    stage: H3StageManifest,
    inputs: [String: H3Tensor]
  ) async throws -> [String: H3Tensor] {
    let runner = try await H3StageRunner(
      name: name,
      manifest: stage,
      baseDirectory: baseDirectory
    )
    return try await runner.predict(inputs)
  }

  private func paddedVisionPatches(
    _ input: H3Tensor,
    actualBlocks: Int,
    shape: [Int]
  ) throws -> H3Tensor {
    let converted = try input.converted(to: .float16)
    let expectedRows = actualBlocks * manifest.visionPatchesPerBlock
    guard converted.shape == [expectedRows, 1536] else {
      throw H3NativeError.invalidTensor(
        "Qwen vision patches are \(converted.shape), expected [\(expectedRows),1536]"
      )
    }
    let targetElements = shape.reduce(1, *)
    guard converted.elementCount <= targetElements else {
      throw H3NativeError.invalidTensor("Qwen vision patch batch exceeds its model")
    }
    var bytes = converted.bytes
    bytes.append(
      Data(
        count: (targetElements - converted.elementCount)
          * H3ScalarType.float16.byteCount
      )
    )
    return try H3Tensor(
      shape: shape,
      scalarType: .float16,
      bytes: bytes
    )
  }

  private func prefixVisualRows(_ input: H3Tensor, count: Int) throws
    -> H3Tensor
  {
    guard count > 0,
      input.elementCount >= count * Self.hiddenSize
    else {
      throw H3NativeError.invalidTensor(
        "Qwen vision output cannot provide \(count) rows"
      )
    }
    let byteCount = count * Self.hiddenSize * input.scalarType.byteCount
    return try H3Tensor(
      shape: [count, Self.hiddenSize],
      scalarType: input.scalarType,
      bytes: Data(input.bytes.prefix(byteCount))
    )
  }

  private func predictVisionBatchStage(
    name: String,
    stage: H3StageManifest,
    inputSemantic: String,
    outputSemantic: String,
    input: H3Tensor
  ) async throws -> [String: H3Tensor] {
    guard let requiredShape = stage.inputConstraints?[inputSemantic]?.shape,
      let requiredBatch = requiredShape.first,
      let logicalBatch = input.shape.first,
      logicalBatch == manifest.visionBlockBatch
    else {
      throw H3NativeError.invalidTensor("\(name) has no valid batch constraint")
    }
    if requiredBatch == logicalBatch {
      return try await predict(
        name: name,
        stage: stage,
        inputs: [inputSemantic: input]
      )
    }
    guard requiredBatch == 1 else {
      throw H3NativeError.invalidTensor(
        "\(name) needs batch \(requiredBatch), presentation has \(logicalBatch)"
      )
    }

    let runner = try await H3StageRunner(
      name: name,
      manifest: stage,
      baseDirectory: baseDirectory
    )
    let inputBytesPerBlock = input.bytes.count / logicalBatch
    guard inputBytesPerBlock * logicalBatch == input.bytes.count else {
      throw H3NativeError.invalidTensor("\(name) input is not batch-contiguous")
    }
    var combinedBytes = Data()
    var outputShape: [Int]?
    var outputType: H3ScalarType?
    for block in 0..<logicalBatch {
      let start = block * inputBytesPerBlock
      let end = start + inputBytesPerBlock
      let blockInput = try H3Tensor(
        shape: [1] + Array(input.shape.dropFirst()),
        scalarType: input.scalarType,
        bytes: input.bytes.subdata(in: start..<end)
      )
      let result = try await runner.predict([inputSemantic: blockInput])
      guard let blockOutput = result[outputSemantic],
        blockOutput.shape.first == 1
      else {
        throw H3NativeError.missingTensor("\(name).\(outputSemantic)")
      }
      if let outputShape {
        guard Array(blockOutput.shape.dropFirst()) == Array(outputShape.dropFirst()),
          blockOutput.scalarType == outputType
        else {
          throw H3NativeError.invalidTensor("\(name) output shape changed by block")
        }
      } else {
        outputShape = blockOutput.shape
        outputType = blockOutput.scalarType
        combinedBytes.reserveCapacity(blockOutput.bytes.count * logicalBatch)
      }
      combinedBytes.append(blockOutput.bytes)
    }
    guard let outputShape, let outputType else {
      throw H3NativeError.missingTensor("\(name).\(outputSemantic)")
    }
    let combined = try H3Tensor(
      shape: [logicalBatch] + Array(outputShape.dropFirst()),
      scalarType: outputType,
      bytes: combinedBytes
    )
    return [outputSemantic: combined]
  }

  private func replacingVisualTokens(
    in hiddenStates: H3Tensor,
    with visual: H3Tensor,
    at positions: [Int]
  ) throws -> H3Tensor {
    let hidden = try hiddenStates.converted(to: .float16)
    let vision = try visual.converted(to: .float16)
    guard hidden.shape == [1, manifest.sequenceLength, Self.hiddenSize],
      vision.elementCount == positions.count * Self.hiddenSize
    else {
      throw H3NativeError.invalidTensor("Qwen visual replacement shape mismatch")
    }
    var destination = hidden.bytes.withUnsafeBytes {
      Array($0.bindMemory(to: Float16.self))
    }
    let source = vision.bytes.withUnsafeBytes {
      Array($0.bindMemory(to: Float16.self))
    }
    for (visualIndex, tokenIndex) in positions.enumerated() {
      let sourceStart = visualIndex * Self.hiddenSize
      let destinationStart = tokenIndex * Self.hiddenSize
      destination.replaceSubrange(
        destinationStart..<(destinationStart + Self.hiddenSize),
        with: source[sourceStart..<(sourceStart + Self.hiddenSize)]
      )
    }
    return try H3Tensor(float16: destination, shape: hidden.shape)
  }

  private func addingVisualTokens(
    to hiddenStates: H3Tensor,
    from visual: H3Tensor,
    at positions: [Int]
  ) throws -> H3Tensor {
    let hidden = try hiddenStates.converted(to: .float16)
    let vision = try visual.converted(to: .float16)
    guard hidden.shape == [1, manifest.sequenceLength, Self.hiddenSize],
      vision.elementCount == positions.count * Self.hiddenSize
    else {
      throw H3NativeError.invalidTensor("Qwen DeepStack shape mismatch")
    }
    var destination = hidden.bytes.withUnsafeBytes {
      Array($0.bindMemory(to: Float16.self))
    }
    let source = vision.bytes.withUnsafeBytes {
      Array($0.bindMemory(to: Float16.self))
    }
    for (visualIndex, tokenIndex) in positions.enumerated() {
      let sourceStart = visualIndex * Self.hiddenSize
      let destinationStart = tokenIndex * Self.hiddenSize
      for channel in 0..<Self.hiddenSize {
        destination[destinationStart + channel] = Float16(
          Float(destination[destinationStart + channel])
            + Float(source[sourceStart + channel])
        )
      }
    }
    return try H3Tensor(float16: destination, shape: hidden.shape)
  }

  private func makeLanguageRoPE(positionIDs: H3Tensor) throws
    -> (cosine: H3Tensor, sine: H3Tensor)
  {
    let sequence = manifest.sequenceLength
    guard positionIDs.shape == [3, sequence] else {
      throw H3NativeError.invalidTensor(
        "Qwen position IDs are \(positionIDs.shape), expected [3, \(sequence)]"
      )
    }
    let positions = try positionIDs.int32Values()
    var cosine = [Float16](
      repeating: 0,
      count: sequence * Self.headDimension
    )
    var sine = cosine
    for frequencyIndex in 0..<(Self.headDimension / 2) {
      var axis = 0
      if frequencyIndex < 60 {
        if frequencyIndex % 3 == 1 { axis = 1 }
        if frequencyIndex % 3 == 2 { axis = 2 }
      }
      let inverseFrequency = powf(
        Self.ropeTheta,
        -Float(frequencyIndex) / Float(Self.headDimension / 2)
      )
      for token in 0..<sequence {
        let angle = Float(positions[axis * sequence + token]) * inverseFrequency
        let cosValue = Float16(cosf(angle))
        let sinValue = Float16(sinf(angle))
        cosine[token * Self.headDimension + frequencyIndex] = cosValue
        cosine[
          token * Self.headDimension + frequencyIndex + Self.headDimension / 2
        ] = cosValue
        sine[token * Self.headDimension + frequencyIndex] = sinValue
        sine[
          token * Self.headDimension + frequencyIndex + Self.headDimension / 2
        ] = sinValue
      }
    }
    let shape = [1, 1, sequence, Self.headDimension]
    return (
      try H3Tensor(float16: cosine, shape: shape),
      try H3Tensor(float16: sine, shape: shape)
    )
  }
}
