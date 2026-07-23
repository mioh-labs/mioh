import CoreAI
import Darwin
import Foundation
import Metal

private let imageSize = 256
private let featureSize = 64
private let featureChannels = 64
private let fusedChannels = 320
private let frameElements = 3 * imageSize * imageSize
private let featureElements = featureChannels * featureSize * featureSize

struct VariableRunnerDescriptor: Decodable {
  let maximumFrames: Int
  let inputOffset: Int
  let outputOffset: Int
  let byteCount: Int
}

enum VariableRunnerError: LocalizedError {
  case invalidArguments
  case invalidDescriptor(String)
  case mappingFailed(String)
  case missingAsset(String)
  case missingFunction(String)
  case unexpectedEndOfInput

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: lada-basicvsrpp-variable-runner <models-dir> <descriptor.json> <shared-file>"
    case .invalidDescriptor(let message):
      return "invalid variable BasicVSR++ descriptor: \(message)"
    case .mappingFailed(let path):
      return "unable to map shared-memory file: \(path)"
    case .missingAsset(let name):
      return "compiled Core AI asset not found: \(name)"
    case .missingFunction(let name):
      return "Core AI function not found: \(name)"
    case .unexpectedEndOfInput:
      return "unexpected end of input"
    }
  }
}

private struct Branch {
  let name: String
  let backward: Bool
}

private let branches = [
  Branch(name: "backward_1", backward: true),
  Branch(name: "forward_1", backward: false),
  Branch(name: "backward_2", backward: true),
  Branch(name: "forward_2", backward: false),
]

private final class VariableCoreAIWorkspace {
  let frameBuffer: MTLBuffer
  let featureBuffer: MTLBuffer
  let backwardFlowBuffer: MTLBuffer
  let forwardFlowBuffer: MTLBuffer
  let restoredBuffer: MTLBuffer
  let spatialStream = ComputeStream()
  let backwardFlowStream = ComputeStream()
  let forwardFlowStream = ComputeStream()
  let propagationStream = ComputeStream()
  let reconstructionStream = ComputeStream()

  init(maximumFrames: Int) throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw VariableRunnerError.invalidDescriptor("Metal device unavailable")
    }
    let frameBytes = frameElements * MemoryLayout<Float16>.stride
    let featureBytes = fusedChannels * featureSize * featureSize
      * MemoryLayout<Float16>.stride
    let flowBytes = 2 * featureSize * featureSize * MemoryLayout<Float16>.stride
    guard
      let frameBuffer = device.makeBuffer(
        length: maximumFrames * frameBytes, options: .storageModeShared),
      let featureBuffer = device.makeBuffer(
        length: maximumFrames * featureBytes, options: .storageModeShared),
      let backwardFlowBuffer = device.makeBuffer(
        length: max(maximumFrames - 1, 1) * flowBytes, options: .storageModeShared),
      let forwardFlowBuffer = device.makeBuffer(
        length: max(maximumFrames - 1, 1) * flowBytes, options: .storageModeShared),
      let restoredBuffer = device.makeBuffer(
        length: maximumFrames * frameBytes, options: .storageModeShared)
    else {
      throw VariableRunnerError.invalidDescriptor("unable to allocate Metal buffers")
    }
    self.frameBuffer = frameBuffer
    self.featureBuffer = featureBuffer
    self.backwardFlowBuffer = backwardFlowBuffer
    self.forwardFlowBuffer = forwardFlowBuffer
    self.restoredBuffer = restoredBuffer
  }
}

@main
struct VariableBasicVSRPPRunner {
  static func main() async {
    do {
      try await run()
    } catch {
      let message = "lada-basicvsrpp-variable-runner: \(error.localizedDescription)\n"
      FileHandle.standardError.write(Data(message.utf8))
      exit(EXIT_FAILURE)
    }
  }

  private static func run() async throws {
    guard CommandLine.arguments.count == 4 else {
      throw VariableRunnerError.invalidArguments
    }
    let modelsDirectory = URL(fileURLWithPath: CommandLine.arguments[1])
    let descriptorURL = URL(fileURLWithPath: CommandLine.arguments[2])
    let sharedPath = CommandLine.arguments[3]
    let descriptor = try JSONDecoder().decode(
      VariableRunnerDescriptor.self,
      from: Data(contentsOf: descriptorURL)
    )
    try validate(descriptor)
    let functions = try await loadFunctions(from: modelsDirectory)
    let useExplicitMetalBuffers = ProcessInfo.processInfo.environment[
      "LADA_VARIABLE_COREAI_EXPLICIT_METAL_BUFFERS"
    ] == "1"
    let workspace = useExplicitMetalBuffers
      ? try VariableCoreAIWorkspace(maximumFrames: descriptor.maximumFrames)
      : nil

    let fileDescriptor = open(sharedPath, O_RDWR)
    guard fileDescriptor >= 0 else {
      throw VariableRunnerError.mappingFailed(sharedPath)
    }
    defer { close(fileDescriptor) }
    let mapping = mmap(
      nil,
      descriptor.byteCount,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      fileDescriptor,
      0
    )
    guard mapping != MAP_FAILED, let mapping else {
      throw VariableRunnerError.mappingFailed(sharedPath)
    }
    defer { munmap(mapping, descriptor.byteCount) }

    let input = FileHandle.standardInput
    while true {
      let command = try readExactly(2, from: input)
      let frameCount = command.withUnsafeBytes { bytes in
        Int(UInt16(littleEndian: bytes.loadUnaligned(as: UInt16.self)))
      }
      if frameCount == Int(UInt16.max) {
        break
      }
      guard frameCount > 0, frameCount <= descriptor.maximumFrames else {
        throw VariableRunnerError.invalidDescriptor("invalid frame count \(frameCount)")
      }
      if let workspace {
        try await inferEncoded(
          functions: functions,
          workspace: workspace,
          frameCount: frameCount,
          inputPointer: mapping.advanced(by: descriptor.inputOffset),
          outputPointer: mapping.advanced(by: descriptor.outputOffset)
        )
      } else {
        try await infer(
          functions: functions,
          frameCount: frameCount,
          inputPointer: mapping.advanced(by: descriptor.inputOffset),
          outputPointer: mapping.advanced(by: descriptor.outputOffset)
        )
      }
      try FileHandle.standardOutput.write(contentsOf: Data([0]))
    }
  }

  private static func validate(_ descriptor: VariableRunnerDescriptor) throws {
    guard descriptor.maximumFrames > 0 else {
      throw VariableRunnerError.invalidDescriptor("maximumFrames must be positive")
    }
    let frameBytes = frameElements * MemoryLayout<Float16>.stride
    let sequenceBytes = descriptor.maximumFrames * frameBytes
    guard descriptor.inputOffset >= 0, descriptor.outputOffset >= 0,
      descriptor.inputOffset + sequenceBytes <= descriptor.byteCount,
      descriptor.outputOffset + sequenceBytes <= descriptor.byteCount
    else {
      throw VariableRunnerError.invalidDescriptor("input/output ranges exceed shared file")
    }
  }

  private static func loadFunctions(
    from directory: URL
  ) async throws -> [String: InferenceFunction] {
    var names = ["spatial", "flow", "reconstruction"]
    for branch in branches {
      names.append("\(branch.name)_init")
      names.append("\(branch.name)_first")
      names.append("\(branch.name)_later")
    }
    var functions: [String: InferenceFunction] = [:]
    for name in names {
      let compiledFilename = "basicvsrpp-variable-\(name).h17s.aimodelc"
      let sourceFilename = "basicvsrpp-variable-\(name).aimodel"
      let compiledURL = directory.appendingPathComponent(
        compiledFilename, isDirectory: true)
      let sourceURL = directory.appendingPathComponent(
        sourceFilename, isDirectory: true)
      let url: URL
      if FileManager.default.fileExists(atPath: compiledURL.path) {
        url = compiledURL
      } else if FileManager.default.fileExists(atPath: sourceURL.path) {
        url = sourceURL
      } else {
        throw VariableRunnerError.missingAsset(
          "\(compiledFilename) or \(sourceFilename)")
      }
      let model = try await AIModel(contentsOf: url)
      guard let function = try model.loadFunction(named: "main") else {
        throw VariableRunnerError.missingFunction(name)
      }
      functions[name] = function
    }
    return functions
  }

  private static func inferEncoded(
    functions: [String: InferenceFunction],
    workspace: VariableCoreAIWorkspace,
    frameCount: Int,
    inputPointer: UnsafeMutableRawPointer,
    outputPointer: UnsafeMutableRawPointer
  ) async throws {
    let frameByteCount = frameElements * MemoryLayout<Float16>.stride
    let perFrameFeatureBytes = fusedChannels * featureSize * featureSize
      * MemoryLayout<Float16>.stride
    let flowByteCount = 2 * featureSize * featureSize * MemoryLayout<Float16>.stride
    let frameBuffer = workspace.frameBuffer
    let featureBuffer = workspace.featureBuffer
    let backwardFlowBuffer = workspace.backwardFlowBuffer
    let forwardFlowBuffer = workspace.forwardFlowBuffer
    let restoredBuffer = workspace.restoredBuffer
    memcpy(frameBuffer.contents(), inputPointer, frameCount * frameByteCount)
    let spatialStream = workspace.spatialStream
    let backwardFlowStream = workspace.backwardFlowStream
    let forwardFlowStream = workspace.forwardFlowStream
    let propagationStream = workspace.propagationStream
    let reconstructionStream = workspace.reconstructionStream
    guard let spatial = functions["spatial"], let flow = functions["flow"],
      let reconstruction = functions["reconstruction"]
    else {
      throw VariableRunnerError.missingFunction("spatial/flow/reconstruction")
    }

    for index in 0..<frameCount {
      let frame = InferenceFunction.AsyncValue(
        unsafeBuffer: frameBuffer,
        byteOffset: index * frameByteCount,
        scalarType: .float16,
        shape: [1, 3, imageSize, imageSize]
      )
      var destination = InferenceFunction.AsyncMutableValue(
        unsafeBuffer: featureBuffer,
        byteOffset: index * perFrameFeatureBytes,
        scalarType: .float16,
        shape: [1, featureChannels, featureSize, featureSize]
      )
      var outputs = InferenceFunction.AsyncMutableViews()
      outputs.insert(&destination, for: "feature")
      _ = try spatial.encode(
        inputs: ["frame": frame], outputViews: outputs, to: spatialStream)
    }

    if frameCount > 1 {
      for index in 0..<(frameCount - 1) {
        let firstFrame = InferenceFunction.AsyncValue(
          unsafeBuffer: frameBuffer,
          byteOffset: index * frameByteCount,
          scalarType: .float16,
          shape: [1, 3, imageSize, imageSize]
        )
        let secondFrame = InferenceFunction.AsyncValue(
          unsafeBuffer: frameBuffer,
          byteOffset: (index + 1) * frameByteCount,
          scalarType: .float16,
          shape: [1, 3, imageSize, imageSize]
        )
        var backwardDestination = InferenceFunction.AsyncMutableValue(
          unsafeBuffer: backwardFlowBuffer,
          byteOffset: index * flowByteCount,
          scalarType: .float16,
          shape: [1, 2, featureSize, featureSize]
        )
        var backwardOutputs = InferenceFunction.AsyncMutableViews()
        backwardOutputs.insert(&backwardDestination, for: "flow")
        _ = try flow.encode(
          inputs: ["ref": firstFrame, "supp": secondFrame],
          outputViews: backwardOutputs,
          to: backwardFlowStream
        )

        let reverseFirst = InferenceFunction.AsyncValue(
          unsafeBuffer: frameBuffer,
          byteOffset: (index + 1) * frameByteCount,
          scalarType: .float16,
          shape: [1, 3, imageSize, imageSize]
        )
        let reverseSecond = InferenceFunction.AsyncValue(
          unsafeBuffer: frameBuffer,
          byteOffset: index * frameByteCount,
          scalarType: .float16,
          shape: [1, 3, imageSize, imageSize]
        )
        var forwardDestination = InferenceFunction.AsyncMutableValue(
          unsafeBuffer: forwardFlowBuffer,
          byteOffset: index * flowByteCount,
          scalarType: .float16,
          shape: [1, 2, featureSize, featureSize]
        )
        var forwardOutputs = InferenceFunction.AsyncMutableViews()
        forwardOutputs.insert(&forwardDestination, for: "flow")
        _ = try flow.encode(
          inputs: ["ref": reverseFirst, "supp": reverseSecond],
          outputViews: forwardOutputs,
          to: forwardFlowStream
        )
      }
    }
    async let spatialCompleted: Void = spatialStream.currentWorkCompleted()
    async let backwardFlowCompleted: Void = backwardFlowStream.currentWorkCompleted()
    async let forwardFlowCompleted: Void = forwardFlowStream.currentWorkCompleted()
    _ = await (spatialCompleted, backwardFlowCompleted, forwardFlowCompleted)

    for (branchIndex, branch) in branches.enumerated() {
      let indices = branch.backward
        ? Array((0..<frameCount).reversed())
        : Array(0..<frameCount)
      let flowBuffer = branch.backward ? backwardFlowBuffer : forwardFlowBuffer
      let contextChannels = featureChannels * (branchIndex + 1)
      let outputChannel = featureChannels * (branchIndex + 1)
      for (order, frameIndex) in indices.enumerated() {
        let phase = order == 0 ? "init" : (order == 1 ? "first" : "later")
        let functionName = "\(branch.name)_\(phase)"
        guard let function = functions[functionName] else {
          throw VariableRunnerError.missingFunction(functionName)
        }
        var inputs: [String: InferenceFunction.AsyncValue] = [
          "context": InferenceFunction.AsyncValue(
            unsafeBuffer: featureBuffer,
            byteOffset: frameIndex * perFrameFeatureBytes,
            scalarType: .float16,
            shape: [1, contextChannels, featureSize, featureSize]
          )
        ]
        if order > 0 {
          let previousFrameIndex = indices[order - 1]
          let flowIndex = branch.backward ? frameIndex : frameIndex - 1
          inputs["state_n1"] = InferenceFunction.AsyncValue(
            unsafeBuffer: featureBuffer,
            byteOffset: previousFrameIndex * perFrameFeatureBytes
              + outputChannel * featureSize * featureSize * MemoryLayout<Float16>.stride,
            scalarType: .float16,
            shape: [1, featureChannels, featureSize, featureSize]
          )
          inputs["flow_n1"] = InferenceFunction.AsyncValue(
            unsafeBuffer: flowBuffer,
            byteOffset: flowIndex * flowByteCount,
            scalarType: .float16,
            shape: [1, 2, featureSize, featureSize]
          )
        }
        if order > 1 {
          let secondPreviousFrameIndex = indices[order - 2]
          let previousFlowIndex = branch.backward ? indices[order - 1] : frameIndex - 2
          inputs["state_n2"] = InferenceFunction.AsyncValue(
            unsafeBuffer: featureBuffer,
            byteOffset: secondPreviousFrameIndex * perFrameFeatureBytes
              + outputChannel * featureSize * featureSize * MemoryLayout<Float16>.stride,
            scalarType: .float16,
            shape: [1, featureChannels, featureSize, featureSize]
          )
          inputs["flow_previous"] = InferenceFunction.AsyncValue(
            unsafeBuffer: flowBuffer,
            byteOffset: previousFlowIndex * flowByteCount,
            scalarType: .float16,
            shape: [1, 2, featureSize, featureSize]
          )
        }
        var destination = InferenceFunction.AsyncMutableValue(
          unsafeBuffer: featureBuffer,
          byteOffset: frameIndex * perFrameFeatureBytes
            + outputChannel * featureSize * featureSize * MemoryLayout<Float16>.stride,
          scalarType: .float16,
          shape: [1, featureChannels, featureSize, featureSize]
        )
        var outputs = InferenceFunction.AsyncMutableViews()
        outputs.insert(&destination, for: "feature")
        _ = try function.encode(
          inputs: inputs, outputViews: outputs, to: propagationStream)
      }
      await propagationStream.currentWorkCompleted()
    }

    for index in 0..<frameCount {
      let frame = InferenceFunction.AsyncValue(
        unsafeBuffer: frameBuffer,
        byteOffset: index * frameByteCount,
        scalarType: .float16,
        shape: [1, 3, imageSize, imageSize]
      )
      let features = InferenceFunction.AsyncValue(
        unsafeBuffer: featureBuffer,
        byteOffset: index * perFrameFeatureBytes,
        scalarType: .float16,
        shape: [1, fusedChannels, featureSize, featureSize]
      )
      var destination = InferenceFunction.AsyncMutableValue(
        unsafeBuffer: restoredBuffer,
        byteOffset: index * frameByteCount,
        scalarType: .float16,
        shape: [1, 3, imageSize, imageSize]
      )
      var outputs = InferenceFunction.AsyncMutableViews()
      outputs.insert(&destination, for: "restored")
      _ = try reconstruction.encode(
        inputs: ["frame": frame, "features": features],
        outputViews: outputs,
        to: reconstructionStream
      )
    }
    await reconstructionStream.currentWorkCompleted()
    memcpy(outputPointer, restoredBuffer.contents(), frameCount * frameByteCount)
  }

  private static func infer(
    functions: [String: InferenceFunction],
    frameCount: Int,
    inputPointer: UnsafeMutableRawPointer,
    outputPointer: UnsafeMutableRawPointer
  ) async throws {
    let frameByteCount = frameElements * MemoryLayout<Float16>.stride
    var frames: [NDArray] = []
    frames.reserveCapacity(frameCount)
    for index in 0..<frameCount {
      var frame = NDArray(shape: [1, 3, imageSize, imageSize], scalarType: .float16)
      var view = frame.mutableView(as: Float16.self)
      _ = view.withUnsafeMutablePointer { pointer, _, _ in
        memcpy(pointer, inputPointer.advanced(by: index * frameByteCount), frameByteCount)
      }
      frames.append(frame)
    }

    var features = (0..<frameCount).map { _ in
      NDArray(shape: [1, fusedChannels, featureSize, featureSize], scalarType: .float16)
    }
    var backwardFlows = (0..<max(frameCount - 1, 0)).map { _ in
      NDArray(shape: [1, 2, featureSize, featureSize], scalarType: .float16)
    }
    var forwardFlows = (0..<max(frameCount - 1, 0)).map { _ in
      NDArray(shape: [1, 2, featureSize, featureSize], scalarType: .float16)
    }

    guard let spatial = functions["spatial"], let flow = functions["flow"],
      let reconstruction = functions["reconstruction"]
    else {
      throw VariableRunnerError.missingFunction("spatial/flow/reconstruction")
    }

    for index in 0..<frameCount {
      try await runSpatial(spatial, frame: frames[index], destination: &features[index])
    }
    if frameCount > 1 {
      for index in 0..<(frameCount - 1) {
        try await runFlow(
          flow,
          reference: frames[index],
          support: frames[index + 1],
          destination: &backwardFlows[index]
        )
        try await runFlow(
          flow,
          reference: frames[index + 1],
          support: frames[index],
          destination: &forwardFlows[index]
        )
      }
    }

    for (branchIndex, branch) in branches.enumerated() {
      let indices = branch.backward
        ? Array((0..<frameCount).reversed())
        : Array(0..<frameCount)
      let directionalFlows = branch.backward ? backwardFlows : forwardFlows
      for (order, frameIndex) in indices.enumerated() {
        let phase = order == 0 ? "init" : (order == 1 ? "first" : "later")
        let functionName = "\(branch.name)_\(phase)"
        guard let function = functions[functionName] else {
          throw VariableRunnerError.missingFunction(functionName)
        }
        let flowIndex: Int?
        let previousFlowIndex: Int?
        if order == 0 {
          flowIndex = nil
          previousFlowIndex = nil
        } else if branch.backward {
          flowIndex = frameIndex
          previousFlowIndex = order > 1 ? indices[order - 1] : nil
        } else {
          flowIndex = frameIndex - 1
          previousFlowIndex = order > 1 ? frameIndex - 2 : nil
        }
        try await runPropagation(
          function,
          branchIndex: branchIndex,
          order: order,
          frameIndex: frameIndex,
          previousFrameIndex: order > 0 ? indices[order - 1] : nil,
          secondPreviousFrameIndex: order > 1 ? indices[order - 2] : nil,
          flowIndex: flowIndex,
          previousFlowIndex: previousFlowIndex,
          flows: directionalFlows,
          features: &features
        )
      }
    }

    for index in 0..<frameCount {
      var restored = NDArray(shape: [1, 3, imageSize, imageSize], scalarType: .float16)
      try await runReconstruction(
        reconstruction,
        frame: frames[index],
        features: features[index],
        destination: &restored
      )
      let view = restored.view(as: Float16.self)
      try view.withUnsafePointer { pointer, shape, _ in
        let expectedShape = [1, 3, imageSize, imageSize]
        let matchesShape = shape.count == expectedShape.count
          && (0..<shape.count).allSatisfy { shape[$0] == expectedShape[$0] }
        guard matchesShape else {
          throw VariableRunnerError.invalidDescriptor("unexpected restored output shape")
        }
        memcpy(outputPointer.advanced(by: index * frameByteCount), pointer, frameByteCount)
      }
    }
  }

  private static func runSpatial(
    _ function: InferenceFunction,
    frame: NDArray,
    destination: inout NDArray
  ) async throws {
    var inputs = InferenceFunction.Inputs()
    inputs.insert(frame, for: "frame")
    var fullDestinationView = destination.mutableView(as: Float16.self)
    let destinationView = fullDestinationView
      .mutatingSlice(at: [.all, 0..<featureChannels, .all, .all])
    var outputs = InferenceFunction.MutableViews()
    outputs.insert(destinationView, for: "feature")
    _ = try await function.run(inputs: inputs, outputViews: outputs)
  }

  private static func runFlow(
    _ function: InferenceFunction,
    reference: NDArray,
    support: NDArray,
    destination: inout NDArray
  ) async throws {
    var inputs = InferenceFunction.Inputs()
    inputs.insert(reference, for: "ref")
    inputs.insert(support, for: "supp")
    let destinationView = destination.mutableView(as: Float16.self)
    var outputs = InferenceFunction.MutableViews()
    outputs.insert(destinationView, for: "flow")
    _ = try await function.run(inputs: inputs, outputViews: outputs)
  }

  private static func runPropagation(
    _ function: InferenceFunction,
    branchIndex: Int,
    order: Int,
    frameIndex: Int,
    previousFrameIndex: Int?,
    secondPreviousFrameIndex: Int?,
    flowIndex: Int?,
    previousFlowIndex: Int?,
    flows: [NDArray],
    features: inout [NDArray]
  ) async throws {
    let contextChannels = featureChannels * (branchIndex + 1)
    let outputStart = featureChannels * (branchIndex + 1)
    let outputEnd = outputStart + featureChannels
    if order == 0 {
      try await runPropagationInit(
        function,
        contextChannels: contextChannels,
        outputRange: outputStart..<outputEnd,
        frameIndex: frameIndex,
        features: &features
      )
      return
    }
    guard let previousFrameIndex, let flowIndex else {
      throw VariableRunnerError.invalidDescriptor("missing first-order propagation state")
    }
    if order == 1 {
      try await runPropagationFirst(
        function,
        contextChannels: contextChannels,
        outputRange: outputStart..<outputEnd,
        frameIndex: frameIndex,
        previousFrameIndex: previousFrameIndex,
        flow: flows[flowIndex],
        features: &features
      )
      return
    }
    guard let secondPreviousFrameIndex, let previousFlowIndex else {
      throw VariableRunnerError.invalidDescriptor("missing second-order propagation state")
    }
    try await runPropagationLater(
      function,
      contextChannels: contextChannels,
      outputRange: outputStart..<outputEnd,
      frameIndex: frameIndex,
      previousFrameIndex: previousFrameIndex,
      secondPreviousFrameIndex: secondPreviousFrameIndex,
      flow: flows[flowIndex],
      previousFlow: flows[previousFlowIndex],
      features: &features
    )
  }

  private static func runPropagationInit(
    _ function: InferenceFunction,
    contextChannels: Int,
    outputRange: Range<Int>,
    frameIndex: Int,
    features: inout [NDArray]
  ) async throws {
    let sourceStorage = features[frameIndex]
    var destinationStorage = features[frameIndex]
    var inputs = InferenceFunction.Inputs()
    let context = sourceStorage.view(as: Float16.self)
      .slice(at: [.all, 0..<contextChannels, .all, .all])
    inputs.insert(context, for: "context")
    var fullDestination = destinationStorage.mutableView(as: Float16.self)
    let destination = fullDestination
      .mutatingSlice(at: [.all, outputRange, .all, .all])
    var outputs = InferenceFunction.MutableViews()
    outputs.insert(destination, for: "feature")
    _ = try await function.run(inputs: inputs, outputViews: outputs)
    features[frameIndex] = destinationStorage
  }

  private static func runPropagationFirst(
    _ function: InferenceFunction,
    contextChannels: Int,
    outputRange: Range<Int>,
    frameIndex: Int,
    previousFrameIndex: Int,
    flow: NDArray,
    features: inout [NDArray]
  ) async throws {
    let sourceStorage = features[frameIndex]
    let previousStorage = features[previousFrameIndex]
    var destinationStorage = features[frameIndex]
    var inputs = InferenceFunction.Inputs()
    let context = sourceStorage.view(as: Float16.self)
      .slice(at: [.all, 0..<contextChannels, .all, .all])
    let stateN1 = previousStorage.view(as: Float16.self)
      .slice(at: [.all, outputRange, .all, .all])
    inputs.insert(context, for: "context")
    inputs.insert(stateN1, for: "state_n1")
    inputs.insert(flow, for: "flow_n1")
    var fullDestination = destinationStorage.mutableView(as: Float16.self)
    let destination = fullDestination
      .mutatingSlice(at: [.all, outputRange, .all, .all])
    var outputs = InferenceFunction.MutableViews()
    outputs.insert(destination, for: "feature")
    _ = try await function.run(inputs: inputs, outputViews: outputs)
    features[frameIndex] = destinationStorage
  }

  private static func runPropagationLater(
    _ function: InferenceFunction,
    contextChannels: Int,
    outputRange: Range<Int>,
    frameIndex: Int,
    previousFrameIndex: Int,
    secondPreviousFrameIndex: Int,
    flow: NDArray,
    previousFlow: NDArray,
    features: inout [NDArray]
  ) async throws {
    let sourceStorage = features[frameIndex]
    let previousStorage = features[previousFrameIndex]
    let secondPreviousStorage = features[secondPreviousFrameIndex]
    var destinationStorage = features[frameIndex]
    var inputs = InferenceFunction.Inputs()
    let context = sourceStorage.view(as: Float16.self)
      .slice(at: [.all, 0..<contextChannels, .all, .all])
    let stateN1 = previousStorage.view(as: Float16.self)
      .slice(at: [.all, outputRange, .all, .all])
    let stateN2 = secondPreviousStorage.view(as: Float16.self)
      .slice(at: [.all, outputRange, .all, .all])
    inputs.insert(context, for: "context")
    inputs.insert(stateN1, for: "state_n1")
    inputs.insert(stateN2, for: "state_n2")
    inputs.insert(flow, for: "flow_n1")
    inputs.insert(previousFlow, for: "flow_previous")
    var fullDestination = destinationStorage.mutableView(as: Float16.self)
    let destination = fullDestination
      .mutatingSlice(at: [.all, outputRange, .all, .all])
    var outputs = InferenceFunction.MutableViews()
    outputs.insert(destination, for: "feature")
    _ = try await function.run(inputs: inputs, outputViews: outputs)
    features[frameIndex] = destinationStorage
  }

  private static func runReconstruction(
    _ function: InferenceFunction,
    frame: NDArray,
    features: NDArray,
    destination: inout NDArray
  ) async throws {
    var inputs = InferenceFunction.Inputs()
    inputs.insert(frame, for: "frame")
    inputs.insert(features, for: "features")
    let destinationView = destination.mutableView(as: Float16.self)
    var outputs = InferenceFunction.MutableViews()
    outputs.insert(destinationView, for: "restored")
    _ = try await function.run(inputs: inputs, outputViews: outputs)
  }

  private static func readExactly(
    _ byteCount: Int,
    from handle: FileHandle
  ) throws -> Data {
    var data = Data()
    while data.count < byteCount {
      guard let chunk = try handle.read(upToCount: byteCount - data.count),
        !chunk.isEmpty
      else {
        throw VariableRunnerError.unexpectedEndOfInput
      }
      data.append(chunk)
    }
    return data
  }
}
