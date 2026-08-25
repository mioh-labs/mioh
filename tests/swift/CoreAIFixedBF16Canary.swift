// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import CoreAI
import Foundation

enum FixedBF16CanaryError: LocalizedError {
  case invalidArguments
  case missingFunction
  case missingDescriptor(String)
  case unexpectedScalarType(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: coreai-fixed-bf16-canary <model.aimodel|model.aimodelc>"
    case .missingFunction:
      return "Core AI function main is missing"
    case .missingDescriptor(let name):
      return "Core AI tensor descriptor is missing: \(name)"
    case .unexpectedScalarType(let message):
      return "unexpected Core AI scalar type: \(message)"
    }
  }
}

@main
struct CoreAIFixedBF16Canary {
  static func main() async throws {
    guard CommandLine.arguments.count == 2 else {
      throw FixedBF16CanaryError.invalidArguments
    }
    let model = try await AIModel(
      contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    guard let function = try model.loadFunction(named: "main") else {
      throw FixedBF16CanaryError.missingFunction
    }
    guard case .ndArray(let inputDescriptor)? =
      function.descriptor.inputDescriptor(of: "x")
    else { throw FixedBF16CanaryError.missingDescriptor("input x") }
    guard case .ndArray(let outputDescriptor)? =
      function.descriptor.outputDescriptor(of: "y")
    else { throw FixedBF16CanaryError.missingDescriptor("output y") }

    var input = NDArray(descriptor: inputDescriptor)
    var output = NDArray(descriptor: outputDescriptor)
    guard input.scalarType == .bfloat16, output.scalarType == .bfloat16 else {
      throw FixedBF16CanaryError.unexpectedScalarType(
        "input=\(input.scalarType), output=\(output.scalarType)")
    }

    let inputValues: [Float] = [1, -2, 0.5, 4]
    try writeBF16(inputValues, to: &input)
    var outputs = InferenceFunction.MutableViews()
    outputs.insert(&output, for: "y")
    _ = try await function.run(inputs: ["x": input], outputViews: outputs)

    let document: [String: Any] = [
      "inputScalarType": String(describing: input.scalarType),
      "outputScalarType": String(describing: output.scalarType),
      "values": try readBF16(output),
    ]
    let data = try JSONSerialization.data(
      withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
  }

  private static func writeBF16(_ values: [Float], to array: inout NDArray) throws {
    try array.mutableRawView().withUnsafeMutableBytes { pointer, shape, strides in
      let dimensions = (0..<shape.count).map { shape[$0] }
      let actualStrides = (0..<strides.count).map { strides[$0] }
      guard dimensions.reduce(1, *) == values.count,
        actualStrides == contiguousStrides(dimensions)
      else {
        throw FixedBF16CanaryError.unexpectedScalarType(
          "non-contiguous BF16 input shape=\(dimensions), strides=\(actualStrides)")
      }
      let words = pointer.bindMemory(to: UInt16.self, capacity: values.count)
      for (index, value) in values.enumerated() {
        words[index] = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
      }
    }
  }

  private static func readBF16(_ array: NDArray) throws -> [Float] {
    try array.rawView().withUnsafeBytes { pointer, shape, strides in
      let dimensions = (0..<shape.count).map { shape[$0] }
      let actualStrides = (0..<strides.count).map { strides[$0] }
      guard actualStrides == contiguousStrides(dimensions) else {
        throw FixedBF16CanaryError.unexpectedScalarType(
          "non-contiguous BF16 output shape=\(dimensions), strides=\(actualStrides)")
      }
      let count = dimensions.reduce(1, *)
      let words = pointer.bindMemory(to: UInt16.self, capacity: count)
      return (0..<count).map {
        Float(bitPattern: UInt32(words[$0]) << 16)
      }
    }
  }

  private static func contiguousStrides(_ shape: [Int]) -> [Int] {
    var strides = [Int](repeating: 1, count: shape.count)
    if shape.count > 1 {
      for index in stride(from: shape.count - 2, through: 0, by: -1) {
        strides[index] = strides[index + 1] * shape[index + 1]
      }
    }
    return strides
  }
}
