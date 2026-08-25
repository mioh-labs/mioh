// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import CoreAI
import Foundation

enum WeightCacheCanaryError: LocalizedError {
  case invalidArguments
  case missingFunction(String)
  case missingDescriptor(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: coreai-weight-cache-canary <model-a> <model-b> [ABAB|BABA]"
    case .missingFunction(let path):
      return "Core AI function main is missing: \(path)"
    case .missingDescriptor(let name):
      return "Core AI tensor descriptor is missing: \(name)"
    }
  }
}

@main
struct CoreAIWeightCacheCanary {
  static func main() async throws {
    guard CommandLine.arguments.count == 3 || CommandLine.arguments.count == 4
    else { throw WeightCacheCanaryError.invalidArguments }
    let paths = Array(CommandLine.arguments[1...2])
    let order = CommandLine.arguments.count == 4 ? CommandLine.arguments[3] : "ABAB"
    guard order.allSatisfy({ $0 == "A" || $0 == "B" }) else {
      throw WeightCacheCanaryError.invalidArguments
    }

    var functions: [Character: InferenceFunction] = [:]
    for (label, path) in zip([Character("A"), Character("B")], paths) {
      let model = try await AIModel(contentsOf: URL(fileURLWithPath: path))
      guard let function = try model.loadFunction(named: "main") else {
        throw WeightCacheCanaryError.missingFunction(path)
      }
      functions[label] = function
    }

    var calls: [[String: Any]] = []
    for label in order {
      guard let function = functions[label] else {
        throw WeightCacheCanaryError.missingFunction(String(label))
      }
      let value = try await run(function)
      calls.append(["model": String(label), "value": value])
    }
    let document: [String: Any] = ["order": order, "calls": calls]
    let data = try JSONSerialization.data(
      withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
  }

  private static func run(_ function: InferenceFunction) async throws -> Float {
    guard case .ndArray(let inputDescriptor)? =
      function.descriptor.inputDescriptor(of: "x")
    else { throw WeightCacheCanaryError.missingDescriptor("input x") }
    guard case .ndArray(let outputDescriptor)? =
      function.descriptor.outputDescriptor(of: "y")
    else { throw WeightCacheCanaryError.missingDescriptor("output y") }
    var input = NDArray(descriptor: inputDescriptor)
    var output = NDArray(descriptor: outputDescriptor)
    input.mutableView(as: Float.self).withUnsafeMutablePointer { pointer, _, _ in
      pointer[0] = 3
    }
    var outputs = InferenceFunction.MutableViews()
    outputs.insert(&output, for: "y")
    _ = try await function.run(inputs: ["x": input], outputViews: outputs)
    return output.view(as: Float.self).withUnsafePointer { pointer, _, _ in
      pointer[0]
    }
  }
}
