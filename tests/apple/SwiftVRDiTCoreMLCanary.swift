// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import CoreML
import Foundation

private struct FixtureShapes: Decodable {
  let hidden: [Int]
  let context: [Int]
  let modulation: [Int]
  let cosine: [Int]
  let sine: [Int]
  let expected: [Int]
}

private enum CanaryError: Error, CustomStringConvertible {
  case invalid(String)

  var description: String {
    switch self {
    case .invalid(let message): message
    }
  }
}

@main
private enum SwiftVRDiTCoreMLCanary {
  static func main() throws {
    guard CommandLine.arguments.count == 3 else {
      throw CanaryError.invalid(
        "usage: swiftvr-dit-canary <block.mlpackage> <fixture-directory>"
      )
    }
    let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let fixture = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
    let shapes = try JSONDecoder().decode(
      FixtureShapes.self,
      from: Data(contentsOf: fixture.appendingPathComponent("shapes.json"))
    )
    let sourceURL = try MLModel.compileModel(at: modelURL)
    defer { try? FileManager.default.removeItem(at: sourceURL) }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    let model = try MLModel(contentsOf: sourceURL, configuration: configuration)

    let dimensions: [(String, [Int])] = [
      ("hidden", shapes.hidden),
      ("context", shapes.context),
      ("modulation", shapes.modulation),
      ("cosine", shapes.cosine),
      ("sine", shapes.sine),
    ]
    var features: [String: Any] = [:]
    for (name, shape) in dimensions {
      let data = try Data(contentsOf: fixture.appendingPathComponent("\(name).f32"))
      let count = shape.reduce(1, *)
      guard data.count == count * MemoryLayout<Float>.size else {
        throw CanaryError.invalid("\(name) byte count does not match shape")
      }
      let array = try MLMultiArray(
        shape: shape.map(NSNumber.init(value:)),
        dataType: .float32
      )
      data.withUnsafeBytes { source in
        guard let base = source.baseAddress else { return }
        array.dataPointer.copyMemory(from: base, byteCount: data.count)
      }
      features[name] = array
    }
    let input = try MLDictionaryFeatureProvider(dictionary: features)
    let output = try model.prediction(from: input)
    guard let result = output.featureValue(for: "output")?.multiArrayValue else {
      throw CanaryError.invalid("Core ML output is missing")
    }
    guard result.shape.map(\.intValue) == shapes.expected else {
      throw CanaryError.invalid("Core ML output shape does not match fixture")
    }
    let expectedData = try Data(
      contentsOf: fixture.appendingPathComponent("expected.f32")
    )
    let count = shapes.expected.reduce(1, *)
    guard expectedData.count == count * MemoryLayout<Float>.size else {
      throw CanaryError.invalid("Expected output byte count does not match shape")
    }
    let strides = result.strides.map(\.intValue)
    let dimensionsCount = shapes.expected.count
    let readValue: (Int) throws -> Double
    switch result.dataType {
    case .float16:
      let values = result.dataPointer.assumingMemoryBound(to: Float16.self)
      readValue = { Double(values[$0]) }
    case .float32:
      let values = result.dataPointer.assumingMemoryBound(to: Float.self)
      readValue = { Double(values[$0]) }
    case .double:
      let values = result.dataPointer.assumingMemoryBound(to: Double.self)
      readValue = { values[$0] }
    default:
      throw CanaryError.invalid("Unsupported Core ML output scalar type")
    }
    var maximum = 0.0
    var total = 0.0
    try expectedData.withUnsafeBytes { raw in
      guard let expected = raw.baseAddress?.assumingMemoryBound(to: Float.self)
      else { throw CanaryError.invalid("Expected output is empty") }
      for linear in 0..<count {
        var remaining = linear
        var offset = 0
        for axis in stride(from: dimensionsCount - 1, through: 0, by: -1) {
          let coordinate = remaining % shapes.expected[axis]
          remaining /= shapes.expected[axis]
          offset += coordinate * strides[axis]
        }
        let delta = abs(try readValue(offset) - Double(expected[linear]))
        maximum = max(maximum, delta)
        total += delta
      }
    }
    FileHandle.standardError.write(Data(
      "Swift Core ML DiT parity: max=\(maximum), mean=\(total / Double(count))\n".utf8
    ))
    guard total / Double(count) < 0.02, maximum < 1 else {
      throw CanaryError.invalid("Native DiT parity threshold exceeded")
    }
  }
}
