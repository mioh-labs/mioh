// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import CoreAI
import Darwin
import Foundation
import Metal

private let imageSize = 256
private let featureSize = 64
private let featureChannels = 64
private let fusedChannels = 320
private let chunkSize = 6
private let frameElements = 3 * imageSize * imageSize
private let featurePlaneElements = featureSize * featureSize
private let featureElements = featureChannels * featurePlaneElements
private let flowElements = 2 * featurePlaneElements

struct Descriptor: Decodable {
  let maximumFrames: Int
  let inputOffset: Int
  let outputOffset: Int
  let byteCount: Int
}

enum RunnerError: LocalizedError {
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
    case .invalidDescriptor(let message): return "invalid descriptor: \(message)"
    case .mappingFailed(let path): return "unable to map shared file: \(path)"
    case .missingAsset(let name): return "missing Core AI asset: \(name)"
    case .missingFunction(let name): return "missing Core AI function: \(name)"
    case .unexpectedEndOfInput: return "unexpected end of input"
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

private final class Workspace {
  let frameBuffer: MTLBuffer
  let featureBuffer: MTLBuffer
  let backwardFlowBuffer: MTLBuffer
  let forwardFlowBuffer: MTLBuffer
  let restoredBuffer: MTLBuffer
  let chunkFrameBuffer: MTLBuffer
  let flowFrameBuffer: MTLBuffer
  let contextBuffer: MTLBuffer
  let flowChunkBuffer: MTLBuffer
  let featureChunkBuffer: MTLBuffer
  let restoredChunkBuffer: MTLBuffer
  // Native-state continuation assets are supported for controlled A/B tests.
  // Shipping explicit-I/O assets leave these buffers unused.
  let nativeStateN1Buffer: MTLBuffer
  let nativeStateN2Buffer: MTLBuffer
  let nativePreviousFlowBuffer: MTLBuffer
  let spatialStream = ComputeStream()
  let flowStream = ComputeStream()
  let propagationStream = ComputeStream()
  let reconstructionStream = ComputeStream()

  init(maximumFrames: Int) throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw RunnerError.invalidDescriptor("Metal device unavailable")
    }
    let half = MemoryLayout<Float16>.stride
    let frameBytes = frameElements * half
    let perFrameFeatureBytes = fusedChannels * featurePlaneElements * half
    let flowBytes = flowElements * half
    func buffer(_ length: Int) throws -> MTLBuffer {
      guard let value = device.makeBuffer(length: length, options: .storageModeShared) else {
        throw RunnerError.invalidDescriptor("unable to allocate Metal buffer")
      }
      memset(value.contents(), 0, length)
      return value
    }
    frameBuffer = try buffer(maximumFrames * frameBytes)
    featureBuffer = try buffer(maximumFrames * perFrameFeatureBytes)
    backwardFlowBuffer = try buffer((maximumFrames + chunkSize) * flowBytes)
    forwardFlowBuffer = try buffer((maximumFrames + chunkSize) * flowBytes)
    restoredBuffer = try buffer(maximumFrames * frameBytes)
    chunkFrameBuffer = try buffer(chunkSize * frameBytes)
    flowFrameBuffer = try buffer((chunkSize + 1) * frameBytes)
    contextBuffer = try buffer(chunkSize * perFrameFeatureBytes)
    flowChunkBuffer = try buffer(chunkSize * flowBytes)
    featureChunkBuffer = try buffer(chunkSize * featureElements * half)
    restoredChunkBuffer = try buffer(chunkSize * frameBytes)
    nativeStateN1Buffer = try buffer(featureElements * half)
    nativeStateN2Buffer = try buffer(featureElements * half)
    nativePreviousFlowBuffer = try buffer(flowElements * half)
  }
}

@main
struct Chunk6Runner {
  static func main() async {
    do { try await run() }
    catch {
      FileHandle.standardError.write(
        Data("lada-basicvsrpp-variable-runner: \(error.localizedDescription)\n".utf8))
      exit(EXIT_FAILURE)
    }
  }

  private static func run() async throws {
    guard CommandLine.arguments.count == 4 else { throw RunnerError.invalidArguments }
    let models = URL(fileURLWithPath: CommandLine.arguments[1])
    let descriptor = try JSONDecoder().decode(
      Descriptor.self,
      from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])))
    let sharedPath = CommandLine.arguments[3]
    guard descriptor.maximumFrames > 0 else {
      throw RunnerError.invalidDescriptor("maximumFrames must be positive")
    }
    let frameBytes = frameElements * MemoryLayout<Float16>.stride
    let sequenceBytes = descriptor.maximumFrames * frameBytes
    guard descriptor.inputOffset >= 0, descriptor.outputOffset >= 0,
      descriptor.inputOffset + sequenceBytes <= descriptor.byteCount,
      descriptor.outputOffset + sequenceBytes <= descriptor.byteCount
    else { throw RunnerError.invalidDescriptor("mapping range is invalid") }

    let functions = try await loadFunctions(from: models)
    let workspace = try Workspace(maximumFrames: descriptor.maximumFrames)
    let fd = open(sharedPath, O_RDWR)
    guard fd >= 0 else { throw RunnerError.mappingFailed(sharedPath) }
    defer { close(fd) }
    let mapping = mmap(nil, descriptor.byteCount, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
    guard mapping != MAP_FAILED, let mapping else { throw RunnerError.mappingFailed(sharedPath) }
    defer { munmap(mapping, descriptor.byteCount) }

    while true {
      let command = try readExactly(2, from: .standardInput)
      let count = command.withUnsafeBytes {
        Int(UInt16(littleEndian: $0.loadUnaligned(as: UInt16.self)))
      }
      if count == Int(UInt16.max) { break }
      guard count > 0, count <= descriptor.maximumFrames else {
        throw RunnerError.invalidDescriptor("invalid frame count \(count)")
      }
      try await infer(
        functions: functions,
        workspace: workspace,
        frameCount: count,
        inputPointer: mapping.advanced(by: descriptor.inputOffset),
        outputPointer: mapping.advanced(by: descriptor.outputOffset))
      try FileHandle.standardOutput.write(contentsOf: Data([0]))
    }
  }

  private static func loadFunctions(from directory: URL) async throws -> [String: InferenceFunction] {
    var names = ["spatial6", "flow6", "reconstruction6"]
    for branch in branches {
      names.append("\(branch.name)_start6")
      names.append("\(branch.name)_continue6")
    }
    var result: [String: InferenceFunction] = [:]
    let directoryEntries = try FileManager.default.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    )
    for name in names {
      let source = directory.appendingPathComponent(
        "basicvsrpp-variable-\(name).aimodel", isDirectory: true)
      let url: URL
      if let compiled = directoryEntries.first(where: {
        let filename = $0.lastPathComponent
        return filename.hasPrefix("basicvsrpp-variable-\(name).")
          && filename.hasSuffix(".aimodelc")
      }) { url = compiled }
      else if FileManager.default.fileExists(atPath: source.path) { url = source }
      else { throw RunnerError.missingAsset(name) }
      let model = try await AIModel(contentsOf: url)
      guard let function = try model.loadFunction(named: "main") else {
        throw RunnerError.missingFunction(name)
      }
      result[name] = function
    }
    return result
  }

  private static func infer(
    functions: [String: InferenceFunction], workspace: Workspace,
    frameCount: Int, inputPointer: UnsafeMutableRawPointer,
    outputPointer: UnsafeMutableRawPointer
  ) async throws {
    let half = MemoryLayout<Float16>.stride
    let frameBytes = frameElements * half
    let featureBytes = featureElements * half
    let perFrameFeatureBytes = fusedChannels * featurePlaneElements * half
    let flowBytes = flowElements * half
    memcpy(workspace.frameBuffer.contents(), inputPointer, frameCount * frameBytes)
    memset(workspace.featureBuffer.contents(), 0, frameCount * perFrameFeatureBytes)

    guard let spatial = functions["spatial6"], let flow = functions["flow6"],
      let reconstruction = functions["reconstruction6"]
    else { throw RunnerError.missingFunction("spatial6/flow6/reconstruction6") }

    for start in stride(from: 0, to: frameCount, by: chunkSize) {
      let valid = min(chunkSize, frameCount - start)
      packFrames(
        from: workspace.frameBuffer, frameCount: frameCount,
        indices: (0..<chunkSize).map { min(start + $0, frameCount - 1) },
        into: workspace.chunkFrameBuffer)
      let input = InferenceFunction.AsyncValue(
        unsafeBuffer: workspace.chunkFrameBuffer, byteOffset: 0,
        scalarType: .float16, shape: [chunkSize, 3, imageSize, imageSize])
      var destination = InferenceFunction.AsyncMutableValue(
        unsafeBuffer: workspace.featureChunkBuffer, byteOffset: 0,
        scalarType: .float16, shape: [chunkSize, featureChannels, featureSize, featureSize])
      var outputs = InferenceFunction.AsyncMutableViews()
      outputs.insert(&destination, for: "features")
      _ = try spatial.encode(inputs: ["frames": input], outputViews: outputs, to: workspace.spatialStream)
      await workspace.spatialStream.currentWorkCompleted()
      for offset in 0..<valid {
        memcpy(
          workspace.featureBuffer.contents().advanced(by: (start + offset) * perFrameFeatureBytes),
          workspace.featureChunkBuffer.contents().advanced(by: offset * featureBytes),
          featureBytes)
      }
    }

    if frameCount > 1 {
      for start in stride(from: 0, to: frameCount - 1, by: chunkSize) {
        let valid = min(chunkSize, frameCount - 1 - start)
        packFrames(
          from: workspace.frameBuffer, frameCount: frameCount,
          indices: (0...chunkSize).map { min(start + $0, frameCount - 1) },
          into: workspace.flowFrameBuffer)
        let input = InferenceFunction.AsyncValue(
          unsafeBuffer: workspace.flowFrameBuffer, byteOffset: 0,
          scalarType: .float16, shape: [chunkSize + 1, 3, imageSize, imageSize])
        var backward = InferenceFunction.AsyncMutableValue(
          unsafeBuffer: workspace.backwardFlowBuffer, byteOffset: start * flowBytes,
          scalarType: .float16, shape: [chunkSize, 2, featureSize, featureSize])
        var forward = InferenceFunction.AsyncMutableValue(
          unsafeBuffer: workspace.forwardFlowBuffer, byteOffset: start * flowBytes,
          scalarType: .float16, shape: [chunkSize, 2, featureSize, featureSize])
        var outputs = InferenceFunction.AsyncMutableViews()
        outputs.insert(&backward, for: "backward")
        outputs.insert(&forward, for: "forward")
        _ = try flow.encode(inputs: ["frames": input], outputViews: outputs, to: workspace.flowStream)
        await workspace.flowStream.currentWorkCompleted()
        _ = valid
      }
    }

    for (branchIndex, branch) in branches.enumerated() {
      let indices = branch.backward ? Array((0..<frameCount).reversed()) : Array(0..<frameCount)
      let directional = branch.backward ? workspace.backwardFlowBuffer : workspace.forwardFlowBuffer
      let contextChannels = featureChannels * (branchIndex + 1)
      let contextBytes = contextChannels * featurePlaneElements * half
      let outputChannel = featureChannels * (branchIndex + 1)
      for chunkStart in stride(from: 0, to: frameCount, by: chunkSize) {
        let valid = min(chunkSize, frameCount - chunkStart)
        let chunkIndices = (0..<chunkSize).map {
          indices[min(chunkStart + $0, frameCount - 1)]
        }
        for offset in 0..<chunkSize {
          memcpy(
            workspace.contextBuffer.contents().advanced(by: offset * contextBytes),
            workspace.featureBuffer.contents().advanced(by: chunkIndices[offset] * perFrameFeatureBytes),
            contextBytes)
        }
        let context = InferenceFunction.AsyncValue(
          unsafeBuffer: workspace.contextBuffer, byteOffset: 0,
          scalarType: .float16,
          shape: [chunkSize, contextChannels, featureSize, featureSize])
        var inputs: [String: InferenceFunction.AsyncValue] = ["contexts": context]
        let functionName: String
        if chunkStart == 0 {
          functionName = "\(branch.name)_start6"
          for position in 1..<chunkSize {
            if position < frameCount {
              let frameIndex = indices[position]
              let flowIndex = branch.backward ? frameIndex : frameIndex - 1
              memcpy(
                workspace.flowChunkBuffer.contents().advanced(by: (position - 1) * flowBytes),
                directional.contents().advanced(by: flowIndex * flowBytes), flowBytes)
            } else {
              memset(workspace.flowChunkBuffer.contents().advanced(by: (position - 1) * flowBytes), 0, flowBytes)
            }
          }
          inputs["flows"] = InferenceFunction.AsyncValue(
            unsafeBuffer: workspace.flowChunkBuffer, byteOffset: 0,
            scalarType: .float16, shape: [chunkSize - 1, 2, featureSize, featureSize])
        } else {
          functionName = "\(branch.name)_continue6"
          var lastFlowIndex = 0
          for offset in 0..<chunkSize {
            let position = chunkStart + offset
            if position < frameCount {
              let frameIndex = indices[position]
              lastFlowIndex = branch.backward ? frameIndex : frameIndex - 1
            }
            memcpy(
              workspace.flowChunkBuffer.contents().advanced(by: offset * flowBytes),
              directional.contents().advanced(by: lastFlowIndex * flowBytes), flowBytes)
          }
          let previousFrame = indices[chunkStart - 1]
          let olderFrame = indices[chunkStart - 2]
          let previousFlowIndex = branch.backward ? previousFrame : previousFrame - 1
          inputs["state_n1"] = InferenceFunction.AsyncValue(
            unsafeBuffer: workspace.featureBuffer,
            byteOffset: previousFrame * perFrameFeatureBytes + outputChannel * featurePlaneElements * half,
            scalarType: .float16, shape: [1, featureChannels, featureSize, featureSize])
          inputs["state_n2"] = InferenceFunction.AsyncValue(
            unsafeBuffer: workspace.featureBuffer,
            byteOffset: olderFrame * perFrameFeatureBytes + outputChannel * featurePlaneElements * half,
            scalarType: .float16, shape: [1, featureChannels, featureSize, featureSize])
          inputs["flows"] = InferenceFunction.AsyncValue(
            unsafeBuffer: workspace.flowChunkBuffer, byteOffset: 0,
            scalarType: .float16, shape: [chunkSize, 2, featureSize, featureSize])
          inputs["flow_previous"] = InferenceFunction.AsyncValue(
            unsafeBuffer: directional, byteOffset: previousFlowIndex * flowBytes,
            scalarType: .float16, shape: [1, 2, featureSize, featureSize])
        }
        guard let function = functions[functionName] else { throw RunnerError.missingFunction(functionName) }
        var destination = InferenceFunction.AsyncMutableValue(
          unsafeBuffer: workspace.featureChunkBuffer, byteOffset: 0,
          scalarType: .float16, shape: [chunkSize, featureChannels, featureSize, featureSize])
        var outputs = InferenceFunction.AsyncMutableViews()
        outputs.insert(&destination, for: "features")
        let nativeStateNames = Set(function.descriptor.stateNames)
        if nativeStateNames.isEmpty {
          _ = try function.encode(
            inputs: inputs,
            outputViews: outputs,
            to: workspace.propagationStream)
        } else {
          let expectedStateNames = Set([
            "state_n1", "state_n2", "flow_previous",
          ])
          guard chunkStart > 0, nativeStateNames == expectedStateNames,
            let flows = inputs["flows"]
          else {
            throw RunnerError.invalidDescriptor(
              "unsupported native state contract for \(functionName): \(nativeStateNames.sorted())")
          }
          // Seed each branch once from the preceding start chunk. Later
          // continuation chunks reuse the state mutated by Core AI directly.
          if chunkStart == chunkSize {
            let previousFrame = indices[chunkStart - 1]
            let olderFrame = indices[chunkStart - 2]
            let previousFlowIndex = branch.backward ? previousFrame : previousFrame - 1
            memcpy(
              workspace.nativeStateN1Buffer.contents(),
              workspace.featureBuffer.contents().advanced(
                by: previousFrame * perFrameFeatureBytes
                  + outputChannel * featurePlaneElements * half),
              featureBytes)
            memcpy(
              workspace.nativeStateN2Buffer.contents(),
              workspace.featureBuffer.contents().advanced(
                by: olderFrame * perFrameFeatureBytes
                  + outputChannel * featurePlaneElements * half),
              featureBytes)
            memcpy(
              workspace.nativePreviousFlowBuffer.contents(),
              directional.contents().advanced(by: previousFlowIndex * flowBytes),
              flowBytes)
          }
          var stateN1 = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: workspace.nativeStateN1Buffer,
            byteOffset: 0,
            scalarType: .float16,
            shape: [1, featureChannels, featureSize, featureSize])
          var stateN2 = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: workspace.nativeStateN2Buffer,
            byteOffset: 0,
            scalarType: .float16,
            shape: [1, featureChannels, featureSize, featureSize])
          var previousFlow = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: workspace.nativePreviousFlowBuffer,
            byteOffset: 0,
            scalarType: .float16,
            shape: [1, 2, featureSize, featureSize])
          var states = InferenceFunction.AsyncMutableViews()
          states.insert(&stateN1, for: "state_n1")
          states.insert(&stateN2, for: "state_n2")
          states.insert(&previousFlow, for: "flow_previous")
          _ = try function.encode(
            inputs: ["contexts": context, "flows": flows],
            states: states,
            outputViews: outputs,
            to: workspace.propagationStream)
        }
        await workspace.propagationStream.currentWorkCompleted()
        for offset in 0..<valid {
          memcpy(
            workspace.featureBuffer.contents().advanced(
              by: chunkIndices[offset] * perFrameFeatureBytes + outputChannel * featurePlaneElements * half),
            workspace.featureChunkBuffer.contents().advanced(by: offset * featureBytes), featureBytes)
        }
      }
    }

    for start in stride(from: 0, to: frameCount, by: chunkSize) {
      let valid = min(chunkSize, frameCount - start)
      let indices = (0..<chunkSize).map { min(start + $0, frameCount - 1) }
      packFrames(from: workspace.frameBuffer, frameCount: frameCount, indices: indices, into: workspace.chunkFrameBuffer)
      for offset in 0..<chunkSize {
        memcpy(
          workspace.contextBuffer.contents().advanced(by: offset * perFrameFeatureBytes),
          workspace.featureBuffer.contents().advanced(by: indices[offset] * perFrameFeatureBytes),
          perFrameFeatureBytes)
      }
      let frames = InferenceFunction.AsyncValue(
        unsafeBuffer: workspace.chunkFrameBuffer, byteOffset: 0,
        scalarType: .float16, shape: [chunkSize, 3, imageSize, imageSize])
      let features = InferenceFunction.AsyncValue(
        unsafeBuffer: workspace.contextBuffer, byteOffset: 0,
        scalarType: .float16, shape: [chunkSize, fusedChannels, featureSize, featureSize])
      var destination = InferenceFunction.AsyncMutableValue(
        unsafeBuffer: workspace.restoredChunkBuffer, byteOffset: 0,
        scalarType: .float16, shape: [chunkSize, 3, imageSize, imageSize])
      var outputs = InferenceFunction.AsyncMutableViews()
      outputs.insert(&destination, for: "restored")
      _ = try reconstruction.encode(
        inputs: ["frames": frames, "features": features],
        outputViews: outputs, to: workspace.reconstructionStream)
      await workspace.reconstructionStream.currentWorkCompleted()
      memcpy(
        workspace.restoredBuffer.contents().advanced(by: start * frameBytes),
        workspace.restoredChunkBuffer.contents(), valid * frameBytes)
    }
    memcpy(outputPointer, workspace.restoredBuffer.contents(), frameCount * frameBytes)
  }

  private static func packFrames(
    from source: MTLBuffer, frameCount: Int, indices: [Int], into destination: MTLBuffer
  ) {
    let frameBytes = frameElements * MemoryLayout<Float16>.stride
    for (offset, index) in indices.enumerated() {
      precondition(index >= 0 && index < frameCount)
      memcpy(
        destination.contents().advanced(by: offset * frameBytes),
        source.contents().advanced(by: index * frameBytes), frameBytes)
    }
  }

  private static func readExactly(_ count: Int, from handle: FileHandle) throws -> Data {
    var result = Data()
    while result.count < count {
      guard let chunk = try handle.read(upToCount: count - result.count), !chunk.isEmpty
      else { throw RunnerError.unexpectedEndOfInput }
      result.append(chunk)
    }
    return result
  }
}
