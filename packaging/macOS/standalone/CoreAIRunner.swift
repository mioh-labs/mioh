import CoreAI
import Darwin
import Foundation

struct TensorDescriptor: Decodable, Sendable {
  let name: String
  let shape: [Int]
  let offset: Int
  let byteCount: Int
}

struct RunnerDescriptor: Decodable, Sendable {
  let function: String
  let slotCount: Int
  let slotStride: Int
  let inputs: [TensorDescriptor]
  let outputs: [TensorDescriptor]
}

struct InferenceWorkspace {
  var inputs: [String: NDArray]

  init(descriptor: RunnerDescriptor) {
    inputs = Dictionary(
      uniqueKeysWithValues: descriptor.inputs.map {
        ($0.name, NDArray(shape: $0.shape, scalarType: .float16))
      }
    )
  }
}

enum RunnerError: LocalizedError {
  case invalidArguments
  case invalidDescriptor(String)
  case invalidSlot(Int)
  case mappingFailed(String)
  case missingFunction(String)
  case missingOutput(String)
  case invalidOutput(String)
  case unexpectedEndOfInput

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: lada-coreai-runner <model.aimodelc> <descriptor.json> <shared-file>"
    case .invalidDescriptor(let message):
      return "invalid Core AI runner descriptor: \(message)"
    case .invalidSlot(let value):
      return "invalid shared-memory slot: \(value)"
    case .mappingFailed(let path):
      return "unable to map shared-memory file: \(path)"
    case .missingFunction(let name):
      return "model function not found: \(name)"
    case .missingOutput(let name):
      return "model output not found: \(name)"
    case .invalidOutput(let message):
      return "invalid model output: \(message)"
    case .unexpectedEndOfInput:
      return "unexpected end of input"
    }
  }
}

final class ResponseWriter: @unchecked Sendable {
  private let lock = NSLock()

  func complete(slot: UInt8) {
    lock.withLock {
      try? FileHandle.standardOutput.write(contentsOf: Data([slot]))
    }
  }

  func fail(_ error: Error) {
    lock.withLock {
      let message = "lada-coreai-runner: \(error.localizedDescription)\n"
      FileHandle.standardError.write(Data(message.utf8))
      try? FileHandle.standardOutput.write(contentsOf: Data([254]))
    }
  }
}

@main
struct CoreAIRunner {
  static func main() async {
    do {
      try await run()
    } catch {
      let message = "lada-coreai-runner: \(error.localizedDescription)\n"
      FileHandle.standardError.write(Data(message.utf8))
      exit(EXIT_FAILURE)
    }
  }

  private static func run() async throws {
    guard CommandLine.arguments.count == 4 else {
      throw RunnerError.invalidArguments
    }
    let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let descriptorURL = URL(fileURLWithPath: CommandLine.arguments[2])
    let sharedPath = CommandLine.arguments[3]
    let descriptor = try JSONDecoder().decode(
      RunnerDescriptor.self,
      from: Data(contentsOf: descriptorURL)
    )
    let mappingSize = try validate(descriptor)

    let model = try await AIModel(contentsOf: modelURL)
    guard let function = try model.loadFunction(named: descriptor.function) else {
      throw RunnerError.missingFunction(descriptor.function)
    }

    let fileDescriptor = open(sharedPath, O_RDWR)
    guard fileDescriptor >= 0 else {
      throw RunnerError.mappingFailed(sharedPath)
    }
    defer { close(fileDescriptor) }
    let mapping = mmap(
      nil,
      mappingSize,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      fileDescriptor,
      0
    )
    guard mapping != MAP_FAILED, let mapping else {
      throw RunnerError.mappingFailed(sharedPath)
    }
    defer { munmap(mapping, mappingSize) }

    let input = FileHandle.standardInput
    let responseWriter = ResponseWriter()
    var workspaces = (0..<descriptor.slotCount).map { _ in
      InferenceWorkspace(descriptor: descriptor)
    }
    while true {
      let slot = Int(try readExactly(1, from: input)[0])
      if slot == 255 {
        break
      }
      guard slot < descriptor.slotCount else {
        throw RunnerError.invalidSlot(slot)
      }
      do {
        try await infer(
          function: function,
          descriptor: descriptor,
          slot: slot,
          mapping: mapping,
          workspace: &workspaces[slot]
        )
        responseWriter.complete(slot: UInt8(slot))
      } catch {
        responseWriter.fail(error)
      }
    }
  }

  private static func validate(_ descriptor: RunnerDescriptor) throws -> Int {
    guard !descriptor.function.isEmpty else {
      throw RunnerError.invalidDescriptor("function is empty")
    }
    guard descriptor.slotCount > 0, descriptor.slotCount < 254 else {
      throw RunnerError.invalidDescriptor("slotCount must be 1...253")
    }
    guard descriptor.slotStride > 0 else {
      throw RunnerError.invalidDescriptor("slotStride must be positive")
    }
    guard !descriptor.inputs.isEmpty, !descriptor.outputs.isEmpty else {
      throw RunnerError.invalidDescriptor("inputs and outputs are required")
    }
    var names = Set<String>()
    for tensor in descriptor.inputs + descriptor.outputs {
      guard !tensor.name.isEmpty, names.insert(tensor.name).inserted else {
        throw RunnerError.invalidDescriptor("tensor names must be nonempty and unique")
      }
      guard !tensor.shape.isEmpty, tensor.shape.allSatisfy({ $0 > 0 }) else {
        throw RunnerError.invalidDescriptor("invalid shape for \(tensor.name)")
      }
      var scalarCount = 1
      var scalarOverflow = false
      for dimension in tensor.shape {
        let result = scalarCount.multipliedReportingOverflow(by: dimension)
        scalarCount = result.partialValue
        scalarOverflow = scalarOverflow || result.overflow
      }
      let (expectedBytes, byteOverflow) = scalarCount.multipliedReportingOverflow(
        by: MemoryLayout<Float16>.stride
      )
      guard !scalarOverflow, !byteOverflow, expectedBytes == tensor.byteCount else {
        throw RunnerError.invalidDescriptor("invalid byteCount for \(tensor.name)")
      }
      let (end, endOverflow) = tensor.offset.addingReportingOverflow(tensor.byteCount)
      guard tensor.offset >= 0, !endOverflow, end <= descriptor.slotStride else {
        throw RunnerError.invalidDescriptor("invalid range for \(tensor.name)")
      }
    }
    let (mappingSize, overflow) = descriptor.slotStride.multipliedReportingOverflow(
      by: descriptor.slotCount
    )
    guard !overflow, mappingSize > 0 else {
      throw RunnerError.invalidDescriptor("mapping size overflow")
    }
    return mappingSize
  }

  private static func infer(
    function: InferenceFunction,
    descriptor: RunnerDescriptor,
    slot: Int,
    mapping: UnsafeMutableRawPointer,
    workspace: inout InferenceWorkspace
  ) async throws {
    let slotPointer = mapping.advanced(by: slot * descriptor.slotStride)
    for input in descriptor.inputs {
      guard var array = workspace.inputs[input.name] else {
        throw RunnerError.invalidDescriptor("missing input workspace for \(input.name)")
      }
      let view = array.mutableView(as: Float16.self)
      _ = view.withUnsafeMutablePointer { pointer, _, _ in
        memcpy(pointer, slotPointer.advanced(by: input.offset), input.byteCount)
      }
      workspace.inputs[input.name] = array
    }

    // Core AI's dynamic output dictionary owns its result NDArrays. Inputs
    // remain persistent across calls; each completed output is copied exactly
    // once into the shared mapping before the result dictionary is released.
    var resultValues = try await function.run(inputs: workspace.inputs)
    for output in descriptor.outputs {
      guard let array = resultValues.remove(output.name)?.ndArray else {
        throw RunnerError.missingOutput(output.name)
      }
      let view = array.view(as: Float16.self)
      guard view.isContiguous else {
        throw RunnerError.invalidOutput("\(output.name) is not contiguous")
      }
      try view.withUnsafePointer { pointer, shape, _ in
        let matchesShape = shape.count == output.shape.count
          && (0..<shape.count).allSatisfy { shape[$0] == output.shape[$0] }
        guard matchesShape else {
          throw RunnerError.invalidOutput(
            "\(output.name) does not match expected shape \(output.shape)"
          )
        }
        memcpy(slotPointer.advanced(by: output.offset), pointer, output.byteCount)
      }
    }
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
        throw RunnerError.unexpectedEndOfInput
      }
      data.append(chunk)
    }
    return data
  }
}
