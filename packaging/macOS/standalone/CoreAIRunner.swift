import CoreAI
import Darwin
import Foundation

enum RunnerError: LocalizedError {
  case invalidArguments
  case invalidFrameCount(String)
  case invalidSlotCount(String)
  case invalidSlot(Int)
  case mappingFailed(String)
  case missingFunction(String)
  case missingOutput(String)
  case unexpectedEndOfInput

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: lada-coreai-runner <model.aimodelc> <frame-count> <shared-file> <slots>"
    case .invalidFrameCount(let value):
      return "invalid frame count: \(value)"
    case .invalidSlotCount(let value):
      return "invalid slot count: \(value)"
    case .invalidSlot(let value):
      return "invalid shared-memory slot: \(value)"
    case .mappingFailed(let path):
      return "unable to map shared-memory file: \(path)"
    case .missingFunction(let name):
      return "model function not found: \(name)"
    case .missingOutput(let name):
      return "model output not found: \(name)"
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
    guard CommandLine.arguments.count == 5 else {
      throw RunnerError.invalidArguments
    }
    guard let frameCount = Int(CommandLine.arguments[2]), frameCount > 0 else {
      throw RunnerError.invalidFrameCount(CommandLine.arguments[2])
    }
    let sharedPath = CommandLine.arguments[3]
    guard let slotCount = Int(CommandLine.arguments[4]),
      slotCount > 0, slotCount < 254
    else {
      throw RunnerError.invalidSlotCount(CommandLine.arguments[4])
    }

    let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let model = try await AIModel(contentsOf: modelURL)
    guard let function = try model.loadFunction(named: "main") else {
      throw RunnerError.missingFunction("main")
    }

    let shape = [1, frameCount, 3, 256, 256]
    let scalarCount = shape.reduce(1, *)
    let payloadSize = scalarCount * MemoryLayout<Float16>.stride
    let mappingSize = payloadSize * slotCount
    let descriptor = open(sharedPath, O_RDWR)
    guard descriptor >= 0 else {
      throw RunnerError.mappingFailed(sharedPath)
    }
    defer { close(descriptor) }
    let mapping = mmap(
      nil,
      mappingSize,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      descriptor,
      0
    )
    guard mapping != MAP_FAILED, let mapping else {
      throw RunnerError.mappingFailed(sharedPath)
    }
    defer { munmap(mapping, mappingSize) }

    let input = FileHandle.standardInput
    let responseWriter = ResponseWriter()

    try await withThrowingTaskGroup(of: Void.self) { group in
      while true {
        let slot = Int(try readExactly(1, from: input)[0])
        if slot == 255 {
          break
        }
        guard slot < slotCount else {
          throw RunnerError.invalidSlot(slot)
        }
        group.addTask {
          do {
            try await infer(
              function: function,
              shape: shape,
              payloadSize: payloadSize,
              slot: slot,
              mapping: mapping
            )
            responseWriter.complete(slot: UInt8(slot))
          } catch {
            responseWriter.fail(error)
          }
        }
      }
      try await group.waitForAll()
    }
  }

  private static func infer(
    function: InferenceFunction,
    shape: [Int],
    payloadSize: Int,
    slot: Int,
    mapping: UnsafeMutableRawPointer
  ) async throws {
    let slotPointer = mapping.advanced(by: slot * payloadSize)
    var frames = NDArray(shape: shape, scalarType: .float16)
    var framesView = frames.mutableView(as: Float16.self)
    _ = framesView.withUnsafeMutablePointer { pointer, _, _ in
      memcpy(pointer, slotPointer, payloadSize)
    }

    var outputs = try await function.run(inputs: ["frames": frames])
    guard let restored = outputs.remove("restored")?.ndArray else {
      throw RunnerError.missingOutput("restored")
    }
    let restoredView = restored.view(as: Float16.self)
    guard restoredView.isContiguous else {
      throw RunnerError.missingOutput("contiguous restored")
    }
    _ = restoredView.withUnsafePointer { pointer, _, _ in
      memcpy(slotPointer, pointer, payloadSize)
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
