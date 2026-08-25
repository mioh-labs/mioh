// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import CoreAI
import Foundation

enum StateCanaryError: LocalizedError {
  case invalidArguments
  case missingFunction
  case missingDescriptor(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: coreai-native-state-canary <model.aimodel|model.aimodelc>"
    case .missingFunction:
      return "Core AI function main is missing"
    case .missingDescriptor(let name):
      return "Core AI tensor descriptor is missing: \(name)"
    }
  }
}

@main
struct CoreAINativeStateCanary {
  static func main() async throws {
    guard CommandLine.arguments.count == 2 else {
      throw StateCanaryError.invalidArguments
    }
    let model = try await AIModel(
      contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    guard let function = try model.loadFunction(named: "main") else {
      throw StateCanaryError.missingFunction
    }

    guard case .ndArray(let stateDescriptor)? =
      function.descriptor.stateDescriptor(of: "acc")
    else { throw StateCanaryError.missingDescriptor("state acc") }
    guard case .ndArray(let inputDescriptor)? =
      function.descriptor.inputDescriptor(of: "x")
    else { throw StateCanaryError.missingDescriptor("input x") }
    guard case .ndArray(let outputDescriptor)? =
      function.descriptor.outputDescriptor(of: "result")
    else { throw StateCanaryError.missingDescriptor("output result") }

    var state = NDArray(descriptor: stateDescriptor)
    var input = NDArray(descriptor: inputDescriptor)
    var output = NDArray(descriptor: outputDescriptor)
    fill(&state, with: 0)
    fill(&input, with: 1)

    var calls: [[Float]] = []
    for _ in 0..<2 {
      var states = InferenceFunction.MutableViews()
      states.insert(&state, for: "acc")
      var outputViews = InferenceFunction.MutableViews()
      outputViews.insert(&output, for: "result")
      _ = try await function.run(
        inputs: ["x": input], states: states, outputViews: outputViews)
      calls.append(values(output))
    }

    let document: [String: Any] = [
      "stateNames": function.descriptor.stateNames,
      "calls": calls,
      "stateAfter": values(state),
    ]
    let data = try JSONSerialization.data(
      withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
  }

  private static func fill(_ array: inout NDArray, with value: Float) {
    let view = array.mutableView(as: Float.self)
    view.withUnsafeMutablePointer { pointer, shape, _ in
      for index in 0..<elementCount(shape) {
        pointer[index] = value
      }
    }
  }

  private static func values(_ array: NDArray) -> [Float] {
    let view = array.view(as: Float.self)
    return view.withUnsafePointer { pointer, shape, _ in
      Array(UnsafeBufferPointer(start: pointer, count: elementCount(shape)))
    }
  }

  private static func elementCount(_ shape: Span<Int>) -> Int {
    var count = 1
    for dimension in shape {
      count *= dimension
    }
    return count
  }
}
