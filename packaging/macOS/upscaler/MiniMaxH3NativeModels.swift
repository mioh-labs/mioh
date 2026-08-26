import CoreAI
import CoreML
import Foundation

// macOS 27 beta 6 exports this Core AI initializer but omits it from the
// generated Swift interface. `preferredComputeUnitKind: .gpu` still leaves
// ANE in Core AI's allowed set, so use the full initializer to constrain the
// outer Core AI specialization to CPU/GPU. The MPSGraph delegate can still
// perform its own ANE placement probe; there is no public switch for that.
@available(macOS 27.0, *)
@_silgen_name("$s15CoreAIDelegates21SpecializationOptionsV23allowedComputeUnitKinds09preferredfG4KindACShyAA0fgJ0OG_AGSgtcfC")
private func h3SpecializationOptions(
  _ allowedComputeUnitKinds: __owned Set<ComputeUnitKind>,
  _ preferredComputeUnitKind: ComputeUnitKind?,
  _ type: SpecializationOptions.Type
) -> SpecializationOptions

protocol H3InferenceStage: AnyObject, Sendable {
  func predict(_ inputs: [String: H3Tensor]) async throws -> [String: H3Tensor]
}

final class H3StageRunner: @unchecked Sendable {
  let name: String
  let manifest: H3StageManifest
  private let implementation: H3InferenceStage

  init(name: String, manifest: H3StageManifest, baseDirectory: URL) async throws {
    self.name = name
    self.manifest = manifest
    let assetURL = URL(
      fileURLWithPath: manifest.asset,
      relativeTo: baseDirectory
    ).standardizedFileURL
    guard FileManager.default.fileExists(atPath: assetURL.path) else {
      throw H3NativeError.missingAsset(assetURL.path)
    }
    switch manifest.backend {
    case .coreAI:
      implementation = try await H3CoreAIStage(
        assetURL: assetURL,
        functionName: manifest.function ?? "main",
        outputNames: Set(manifest.outputs.values),
        preferredCompute: manifest.computeUnits
      )
    case .coreML:
      implementation = try await H3CoreMLStage(
        assetURL: assetURL,
        computeUnits: manifest.computeUnits
      )
    }
  }

  func predict(_ semanticInputs: [String: H3Tensor]) async throws
    -> [String: H3Tensor]
  {
    var modelInputs: [String: H3Tensor] = [:]
    for (semantic, modelName) in manifest.inputs {
      guard let value = semanticInputs[semantic] else {
        throw H3NativeError.missingTensor("\(name).\(semantic)")
      }
      try manifest.inputConstraints?[semantic]?.validate(value, semantic: semantic)
      modelInputs[modelName] = value
    }
    let modelOutputs = try await implementation.predict(modelInputs)
    var semanticOutputs: [String: H3Tensor] = [:]
    for (semantic, modelName) in manifest.outputs {
      guard let value = modelOutputs[modelName] else {
        throw H3NativeError.missingTensor("\(name) output \(modelName)")
      }
      try manifest.outputConstraints?[semantic]?.validate(value, semantic: semantic)
      semanticOutputs[semantic] = value
    }
    return semanticOutputs
  }
}

@available(macOS 27.0, *)
final class H3CoreAIStage: H3InferenceStage, @unchecked Sendable {
  private let function: InferenceFunction
  private let outputNames: Set<String>

  init(
    assetURL: URL,
    functionName: String,
    outputNames: Set<String>,
    preferredCompute: String? = nil
  ) async throws {
    let model = try await H3CoreAIModelLoader.load(
      assetURL: assetURL,
      preferredCompute: preferredCompute
    )
    guard let loaded = try model.loadFunction(named: functionName) else {
      throw H3NativeError.inference(
        "Core AI function \(functionName) is missing from \(assetURL.path)"
      )
    }
    function = loaded
    self.outputNames = outputNames
  }

  func predict(_ inputs: [String: H3Tensor]) async throws -> [String: H3Tensor] {
    var nativeInputs: [String: NDArray] = [:]
    for (name, tensor) in inputs {
      nativeInputs[name] = try Self.makeNDArray(tensor)
    }
    var nativeOutputs = try await function.run(inputs: nativeInputs)
    var outputs: [String: H3Tensor] = [:]
    for name in outputNames.sorted() {
      guard let array = nativeOutputs.remove(name)?.ndArray else {
        throw H3NativeError.missingTensor("Core AI output \(name)")
      }
      outputs[name] = try Self.makeTensor(array)
    }
    return outputs
  }

  static func makeNDArray(_ tensor: H3Tensor) throws -> NDArray {
    switch tensor.scalarType {
    case .bfloat16:
      var array = NDArray(shape: tensor.shape, scalarType: .bfloat16)
      try array.mutableRawView().withUnsafeMutableBytes {
        pointer, shape, strides in
        let dimensions = (0..<shape.count).map { shape[$0] }
        let actualStrides = (0..<strides.count).map { strides[$0] }
        guard actualStrides == contiguousStrides(dimensions) else {
          throw H3NativeError.inference(
            "Core AI allocated non-contiguous bfloat16 input"
          )
        }
        _ = tensor.bytes.withUnsafeBytes { source in
          memcpy(pointer, source.baseAddress!, tensor.bytes.count)
        }
      }
      return array
    case .float16:
      let values = tensor.bytes.withUnsafeBytes {
        Array($0.bindMemory(to: Float16.self))
      }
      return NDArray(scalars: values, shape: tensor.shape)
    case .float32:
      let values = tensor.bytes.withUnsafeBytes {
        Array($0.bindMemory(to: Float.self))
      }
      return NDArray(scalars: values, shape: tensor.shape)
    case .int32:
      let values = tensor.bytes.withUnsafeBytes {
        Array($0.bindMemory(to: Int32.self))
      }
      return NDArray(scalars: values, shape: tensor.shape)
    case .int64:
      let values = tensor.bytes.withUnsafeBytes {
        Array($0.bindMemory(to: Int64.self))
      }
      return NDArray(scalars: values, shape: tensor.shape)
    }
  }

  static func makeTensor(_ array: NDArray) throws -> H3Tensor {
    switch array.scalarType {
    case .bfloat16:
      let raw = array.rawView()
      return try raw.withUnsafeBytes { pointer, shape, strides in
        let dimensions = (0..<shape.count).map { shape[$0] }
        let actualStrides = (0..<strides.count).map { strides[$0] }
        guard actualStrides == contiguousStrides(dimensions) else {
          throw H3NativeError.inference(
            "Core AI bfloat16 output is not contiguous"
          )
        }
        let data = Data(
          bytes: pointer,
          count: dimensions.reduce(1, *) * MemoryLayout<UInt16>.stride
        )
        return try H3Tensor(
          shape: dimensions,
          scalarType: .bfloat16,
          bytes: data
        )
      }
    case .float16:
      let view = array.view(as: Float16.self)
      guard view.isContiguous else {
        throw H3NativeError.inference("Core AI float16 output is not contiguous")
      }
      return try view.withUnsafePointer { pointer, shape, _ in
        let dimensions = (0..<shape.count).map { shape[$0] }
        let data = Data(
          bytes: pointer,
          count: dimensions.reduce(1, *) * MemoryLayout<Float16>.stride
        )
        return try H3Tensor(shape: dimensions, scalarType: .float16, bytes: data)
      }
    case .float32:
      let view = array.view(as: Float.self)
      guard view.isContiguous else {
        throw H3NativeError.inference("Core AI float32 output is not contiguous")
      }
      return try view.withUnsafePointer { pointer, shape, _ in
        let dimensions = (0..<shape.count).map { shape[$0] }
        let data = Data(
          bytes: pointer,
          count: dimensions.reduce(1, *) * MemoryLayout<Float>.stride
        )
        return try H3Tensor(shape: dimensions, scalarType: .float32, bytes: data)
      }
    case .int32:
      let view = array.view(as: Int32.self)
      guard view.isContiguous else {
        throw H3NativeError.inference("Core AI int32 output is not contiguous")
      }
      return try view.withUnsafePointer { pointer, shape, _ in
        let dimensions = (0..<shape.count).map { shape[$0] }
        let data = Data(
          bytes: pointer,
          count: dimensions.reduce(1, *) * MemoryLayout<Int32>.stride
        )
        return try H3Tensor(shape: dimensions, scalarType: .int32, bytes: data)
      }
    case .int64:
      let view = array.view(as: Int64.self)
      guard view.isContiguous else {
        throw H3NativeError.inference("Core AI int64 output is not contiguous")
      }
      return try view.withUnsafePointer { pointer, shape, _ in
        let dimensions = (0..<shape.count).map { shape[$0] }
        let data = Data(
          bytes: pointer,
          count: dimensions.reduce(1, *) * MemoryLayout<Int64>.stride
        )
        return try H3Tensor(shape: dimensions, scalarType: .int64, bytes: data)
      }
    default:
      throw H3NativeError.unsupported(
        "Core AI output scalar type \(array.scalarType)"
      )
    }
  }

  private static func contiguousStrides(_ shape: [Int]) -> [Int] {
    var strides = [Int](repeating: 1, count: shape.count)
    guard shape.count > 1 else { return strides }
    for index in stride(from: shape.count - 2, through: 0, by: -1) {
      strides[index] = strides[index + 1] * shape[index + 1]
    }
    return strides
  }

}

/// Loads the already compiled asset with an explicit compute preference.
///
/// Every repeated Qwen and DiT asset carries a layer-specific structural salt.
/// This is required on macOS 27 beta 6 because weight-distinct assets with the
/// same program hash can alias. Compiled h17s assets load directly. Exact-shape
/// BF16 source assets cannot be compiled by coreai-build on macOS 27 beta 6.
/// The fixed graph avoids dynamic reshaping; MPSGraph's internal ANE placement
/// probe remains outside this loader's control and falls back to the GPU.
@available(macOS 27.0, *)
private enum H3CoreAIModelLoader {
  static func load(
    assetURL: URL,
    preferredCompute: String?
  ) async throws -> AIModel {
    let options = try specializationOptions(preferredCompute)
    // Reuse Core AI's outer CPU/GPU specialization across block unloads, but
    // leave it purgeable under storage pressure. `.persistent` disables all
    // purge conditions and allowed the H3 cache to grow to tens of GiB.
    // This does not cache or suppress MPSGraph's separate internal ANE probe.
    return try await AIModel.specialize(
      contentsOf: assetURL,
      options: options,
      cache: .default,
      cachePolicy: .default
    )
  }

  private static func specializationOptions(
    _ preferredCompute: String?
  ) throws -> SpecializationOptions {
    switch preferredCompute?.lowercased() {
    case nil, "", "default", "all":
      return .default
    case "gpu":
      let allowed = ComputeUnitKind.availableKinds.intersection([.cpu, .gpu])
      guard allowed.contains(.gpu) else {
        throw H3NativeError.inference("Core AI GPU is unavailable")
      }
      let options = h3SpecializationOptions(
        allowed,
        .gpu,
        SpecializationOptions.self
      )
      guard options.allowedComputeUnitKinds == allowed,
        options.preferredComputeUnitKind == .gpu
      else {
        throw H3NativeError.inference(
          "Core AI failed to restrict MiniMax H3 to CPU and GPU"
        )
      }
      return options
    case "ane", "neuralengine", "neural_engine":
      return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    case "cpu":
      return .cpuOnly
    default:
      throw H3NativeError.invalidManifest(
        "unsupported Core AI preferred compute \(preferredCompute ?? "")"
      )
    }
  }
}

/// Keeps the 62000x5376 BF16 hidden state inside Core AI between DiT blocks.
/// A tensor round-trip through Swift Data at every block would copy hundreds
/// of MB per block and dominate the multi-step sampler.
///
/// The learned functions are deliberately loaded one at a time. Two caller-
/// owned NDArrays are alternated as input/output buffers, which prevents each
/// result from retaining the preceding Core AI execution graph and avoids an
/// IOSurface-sized intermediate allocation at every block boundary.
@available(macOS 27.0, *)
final class H3CoreAIBlockSequence: @unchecked Sendable {
  /// The block loop is intentionally serial. Increasing this would retain more
  /// than one multi-GB specialized DiT program in unified memory.
  static let maximumResidentModels = 1

  private struct Entry: @unchecked Sendable {
    let assetURL: URL
    let functionName: String
    let inputNames: [String: String]
    let outputName: String
    let logicalLayerCount: Int
    let preferredCompute: String
    let graphSalt: NDArray?
  }

  private let entries: [Entry]
  private let onLoad: (@Sendable (Int, Int) -> Void)?

  init(
    manifests: [H3StageManifest],
    baseDirectory: URL,
    onLoad: (@Sendable (Int, Int) -> Void)? = nil
  ) async throws {
    var validated: [Entry] = []
    validated.reserveCapacity(manifests.count)
    for (index, manifest) in manifests.enumerated() {
      guard manifest.backend == .coreAI,
        let outputName = manifest.outputs["hiddenStates"]
      else {
        throw H3NativeError.invalidManifest(
          "10Eros DiT block sequence requires Core AI hiddenStates bindings"
        )
      }
      let assetURL = URL(
        fileURLWithPath: manifest.asset,
        relativeTo: baseDirectory
      ).standardizedFileURL
      guard FileManager.default.fileExists(atPath: assetURL.path) else {
        throw H3NativeError.missingAsset(assetURL.path)
      }
      let graphSalt: NDArray?
      if manifest.inputs["graphSalt"] != nil {
        guard let constraint = manifest.inputConstraints?["graphSalt"],
          let shape = constraint.shape,
          !shape.isEmpty,
          shape.allSatisfy({ $0 > 0 })
        else {
          throw H3NativeError.invalidManifest(
            "10Eros graphSalt requires a fixed positive shape"
          )
        }
        let count = shape.reduce(1, *)
        let tensor = try H3Tensor(
          shape: shape,
          scalarType: constraint.scalarType,
          bytes: Data(count: count * constraint.scalarType.byteCount)
        )
        graphSalt = try H3CoreAIStage.makeNDArray(tensor)
      } else {
        graphSalt = nil
      }
      validated.append(
        Entry(
          assetURL: assetURL,
          functionName: manifest.function ?? "main",
          inputNames: manifest.inputs,
          outputName: outputName,
          logicalLayerCount: max(1, manifest.logicalLayerCount ?? 1),
          preferredCompute: manifest.computeUnits ?? "gpu",
          graphSalt: graphSalt
        )
      )
      _ = index
    }
    entries = validated
    self.onLoad = onLoad
  }

  func predict(
    hiddenStates: H3Tensor,
    timestepCoordinates: H3Tensor,
    modulationWeights: H3Tensor,
    ropeCosine: H3Tensor,
    ropeSine: H3Tensor
  ) async throws -> H3Tensor {
    var hidden = try H3CoreAIStage.makeNDArray(hiddenStates)
    var scratch = NDArray(
      shape: hidden.shape,
      scalarType: hidden.scalarType,
      strides: hidden.strides
    )
    let shared: [String: NDArray] = [
      "timestepCoordinates": try H3CoreAIStage.makeNDArray(timestepCoordinates),
      "modulationWeights": try H3CoreAIStage.makeNDArray(modulationWeights),
      "ropeCosine": try H3CoreAIStage.makeNDArray(ropeCosine),
      "ropeSine": try H3CoreAIStage.makeNDArray(ropeSine),
    ]
    let totalLogicalLayers = entries.reduce(0) { $0 + $1.logicalLayerCount }
    var completedLogicalLayers = 0
    for entry in entries {
      try Task.checkCancellation()
      completedLogicalLayers += entry.logicalLayerCount
      onLoad?(completedLogicalLayers, totalLogicalLayers)
      try await Self.run(
        entry: entry,
        hidden: hidden,
        shared: shared,
        output: &scratch
      )
      swap(&hidden, &scratch)

      // Give the app and WindowServer a scheduling point after the just-used
      // function and its temporary command buffers leave the helper scope.
      await Task.yield()
    }
    return try H3CoreAIStage.makeTensor(hidden)
  }

  /// The helper's async frame owns the sole resident model/function. Core AI is
  /// told to write directly into `output`, so the next hidden state does not
  /// keep this frame's transient output allocation or execution graph alive.
  private static func run(
    entry: Entry,
    hidden: NDArray,
    shared: [String: NDArray],
    output: inout NDArray
  ) async throws {
    let model = try await H3CoreAIModelLoader.load(
      assetURL: entry.assetURL,
      preferredCompute: entry.preferredCompute
    )
    guard let function = try model.loadFunction(named: entry.functionName) else {
      throw H3NativeError.inference(
        "Core AI function \(entry.functionName) is missing from \(entry.assetURL.path)"
      )
    }
    var inputs: [String: NDArray] = [:]
    for (semantic, modelName) in entry.inputNames {
      if semantic == "hiddenStates" {
        inputs[modelName] = hidden
      } else if semantic == "graphSalt", let graphSalt = entry.graphSalt {
        inputs[modelName] = graphSalt
      } else if let value = shared[semantic] {
        inputs[modelName] = value
      } else {
        throw H3NativeError.missingTensor("10Eros block input \(semantic)")
      }
    }
    var outputViews = InferenceFunction.MutableViews()
    outputViews.insert(&output, for: entry.outputName)
    _ = try await function.run(
      inputs: inputs,
      outputViews: outputViews
    )
  }
}

private final class H3CoreMLStage: H3InferenceStage, @unchecked Sendable {
  private let model: MLModel
  private var temporaryCompiledURL: URL?

  init(
    assetURL: URL,
    computeUnits: String?
  ) async throws {
    let configuration = MLModelConfiguration()
    switch (computeUnits ?? "all").lowercased() {
    case "all": configuration.computeUnits = .all
    case "cpuonly": configuration.computeUnits = .cpuOnly
    case "cpuandgpu": configuration.computeUnits = .cpuAndGPU
    case "cpuandneuralengine", "cpuandane":
      configuration.computeUnits = .cpuAndNeuralEngine
    default:
      throw H3NativeError.invalidManifest(
        "unsupported Core ML computeUnits \(computeUnits ?? "")"
      )
    }
    let loadURL: URL
    if assetURL.pathExtension.lowercased() == "mlpackage" {
      loadURL = try await MLModel.compileModel(at: assetURL)
      temporaryCompiledURL = loadURL
    } else {
      loadURL = assetURL
    }
    model = try MLModel(contentsOf: loadURL, configuration: configuration)
  }

  deinit {
    if let temporaryCompiledURL {
      try? FileManager.default.removeItem(at: temporaryCompiledURL)
    }
  }

  func predict(_ inputs: [String: H3Tensor]) async throws -> [String: H3Tensor] {
    var features: [String: MLFeatureValue] = [:]
    for (name, tensor) in inputs {
      features[name] = MLFeatureValue(multiArray: try Self.makeMultiArray(tensor))
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: features)
    let prediction = try await model.prediction(from: provider)
    var outputs: [String: H3Tensor] = [:]
    for name in prediction.featureNames {
      guard let array = prediction.featureValue(for: name)?.multiArrayValue else {
        continue
      }
      outputs[name] = try Self.makeTensor(array)
    }
    return outputs
  }

  private static func makeMultiArray(_ tensor: H3Tensor) throws -> MLMultiArray {
    let dataType: MLMultiArrayDataType
    switch tensor.scalarType {
    case .bfloat16:
      throw H3NativeError.unsupported(
        "Core ML MLMultiArray has no bfloat16 storage"
      )
    case .float16: dataType = .float16
    case .float32: dataType = .float32
    case .int32: dataType = .int32
    case .int64:
      throw H3NativeError.unsupported(
        "Core ML MLMultiArray has no int64 storage; export token inputs as int32"
      )
    }
    let array = try MLMultiArray(
      shape: tensor.shape.map(NSNumber.init(value:)),
      dataType: dataType
    )
    try array.withUnsafeMutableBytes { destination, strides in
      let expected = contiguousStrides(tensor.shape)
      guard strides == expected else {
        throw H3NativeError.inference(
          "Core ML allocated non-contiguous input \(strides), expected \(expected)"
        )
      }
      guard destination.count >= tensor.bytes.count else {
        throw H3NativeError.invalidTensor("Core ML input buffer is too small")
      }
      _ = tensor.bytes.withUnsafeBytes { source in
        memcpy(destination.baseAddress!, source.baseAddress!, tensor.bytes.count)
      }
    }
    return array
  }

  private static func makeTensor(
    _ array: MLMultiArray
  ) throws -> H3Tensor {
    let scalarType: H3ScalarType
    switch array.dataType {
    case .float16: scalarType = .float16
    case .float32: scalarType = .float32
    case .int32: scalarType = .int32
    default:
      throw H3NativeError.unsupported(
        "Core ML output data type \(array.dataType.rawValue)"
      )
    }
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    let contiguous = contiguousStrides(shape)
    guard strides == contiguous else {
      throw H3NativeError.inference(
        "Core ML output is non-contiguous: \(strides), expected \(contiguous)"
      )
    }
    let count = shape.reduce(1, *) * scalarType.byteCount
    let data = array.withUnsafeBytes { raw in
      Data(bytes: raw.baseAddress!, count: count)
    }
    return try H3Tensor(shape: shape, scalarType: scalarType, bytes: data)
  }

  private static func contiguousStrides(_ shape: [Int]) -> [Int] {
    shape.indices.map { index in
      shape[(index + 1)...].reduce(1, *)
    }
  }
}
