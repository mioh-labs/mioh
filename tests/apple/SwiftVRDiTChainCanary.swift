// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

// Standalone native-runtime validation only; this is not the mioh ROI enhancer.
import CoreML
import Foundation

private struct Shapes: Decodable {
  let hidden: [Int]
  let context: [Int]
  let modulation: [Int]
  let cosine: [Int]
  let sine: [Int]
}

private enum ChainError: Error, CustomStringConvertible {
  case invalid(String)

  var description: String {
    switch self {
    case .invalid(let message): message
    }
  }
}

private func floatArray(
  at url: URL, shape: [Int]
) throws -> MLMultiArray {
  let data = try Data(contentsOf: url)
  let count = shape.reduce(1, *)
  guard data.count == count * MemoryLayout<Float>.size else {
    throw ChainError.invalid("Unexpected size for \(url.lastPathComponent)")
  }
  let result = try MLMultiArray(
    shape: shape.map(NSNumber.init(value:)), dataType: .float32
  )
  data.withUnsafeBytes { raw in
    if let source = raw.baseAddress {
      result.dataPointer.copyMemory(from: source, byteCount: data.count)
    }
  }
  return result
}

private func contiguousFloat32(
  _ input: MLMultiArray, shape: [Int]
) throws -> MLMultiArray {
  guard input.shape.map(\.intValue) == shape else {
    throw ChainError.invalid("DiT block changed the hidden shape")
  }
  let result = try MLMultiArray(
    shape: shape.map(NSNumber.init(value:)), dataType: .float32
  )
  let strides = input.strides.map(\.intValue)
  let target = result.dataPointer.assumingMemoryBound(to: Float.self)
  let count = shape.reduce(1, *)
  switch input.dataType {
  case .float16:
    let source = input.dataPointer.assumingMemoryBound(to: Float16.self)
    for linear in 0..<count {
      var remaining = linear
      var sourceOffset = 0
      for axis in stride(from: shape.count - 1, through: 0, by: -1) {
        let index = remaining % shape[axis]
        remaining /= shape[axis]
        sourceOffset += index * strides[axis]
      }
      target[linear] = Float(source[sourceOffset])
    }
  case .float32:
    let source = input.dataPointer.assumingMemoryBound(to: Float.self)
    for linear in 0..<count {
      var remaining = linear
      var sourceOffset = 0
      for axis in stride(from: shape.count - 1, through: 0, by: -1) {
        let index = remaining % shape[axis]
        remaining /= shape[axis]
        sourceOffset += index * strides[axis]
      }
      target[linear] = source[sourceOffset]
    }
  default:
    throw ChainError.invalid("Unsupported output type \(input.dataType)")
  }
  return result
}

@main
private enum SwiftVRDiTChainCanary {
  static func main() throws {
    guard CommandLine.arguments.count == 5 else {
      throw ChainError.invalid(
        "usage: swiftvr-dit-chain <model-directory> <fixture-directory> <latent-frames:6|7> <scale:2|4>"
      )
    }
    let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let fixture = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
    guard let frames = Int(CommandLine.arguments[3]), [6, 7].contains(frames),
      let scale = Int(CommandLine.arguments[4]), [2, 4].contains(scale)
    else { throw ChainError.invalid("Unsupported frame count or scale") }
    let shapes = try JSONDecoder().decode(
      Shapes.self, from: Data(contentsOf: fixture.appendingPathComponent("shapes.json"))
    )
    let hiddenShape = shapes.hidden
    var hidden = try floatArray(
      at: fixture.appendingPathComponent("hidden.f32"), shape: hiddenShape
    )
    let constants: [(String, [Int])] = [
      ("context", shapes.context),
      ("modulation", shapes.modulation),
      ("cosine", shapes.cosine),
      ("sine", shapes.sine),
    ]
    var features: [String: Any] = [:]
    for (name, shape) in constants {
      features[name] = try floatArray(
        at: fixture.appendingPathComponent("\(name).f32"), shape: shape
      )
    }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    let started = Date()
    for layer in 0..<30 {
      let asset = String(
        format: "dit-block-%02d-t%d-%dx-float16.mlpackage", layer, frames, scale
      )
      let modelURL = root.appendingPathComponent(asset)
      guard FileManager.default.fileExists(atPath: modelURL.path) else {
        throw ChainError.invalid("Missing \(modelURL.path)")
      }
      // One resident block bounds model memory. Compilation is deliberately
      // separate from shipping a compiled cache; timings here include loads.
      let compiled = try MLModel.compileModel(at: modelURL)
      defer { try? FileManager.default.removeItem(at: compiled) }
      let model = try MLModel(contentsOf: compiled, configuration: configuration)
      features["hidden"] = hidden
      let result = try model.prediction(
        from: MLDictionaryFeatureProvider(dictionary: features)
      )
      guard let output = result.featureValue(for: "output")?.multiArrayValue else {
        throw ChainError.invalid("Block \(layer) has no output")
      }
      hidden = try contiguousFloat32(output, shape: hiddenShape)
      let values = hidden.dataPointer.assumingMemoryBound(to: Float.self)
      let count = hiddenShape.reduce(1, *)
      var absoluteSum: Double = 0
      var largest: Float = 0
      for index in 0..<count {
        let value = values[index]
        guard value.isFinite else {
          throw ChainError.invalid("Block \(layer) produced a nonfinite value")
        }
        let absolute = abs(value)
        absoluteSum += Double(absolute)
        largest = max(largest, absolute)
      }
      print(
        "block \(layer): mean-abs=\(absoluteSum / Double(count)) max-abs=\(largest) elapsed=\(Date().timeIntervalSince(started))s"
      )
    }
    let outputURL = fixture.appendingPathComponent("native-chain-output.f32")
    let count = hiddenShape.reduce(1, *)
    let bytes = Data(bytes: hidden.dataPointer, count: count * MemoryLayout<Float>.size)
    try bytes.write(to: outputURL, options: .atomic)
    let referenceURL = fixture.appendingPathComponent("expected-chain.f32")
    if FileManager.default.fileExists(atPath: referenceURL.path) {
      let expected = try Data(contentsOf: referenceURL)
      guard expected.count == bytes.count else {
        throw ChainError.invalid("Official reference size does not match native output")
      }
      var maximum = 0.0
      var total = 0.0
      var referenceTotal = 0.0
      try expected.withUnsafeBytes { raw in
        guard let reference = raw.baseAddress?.assumingMemoryBound(to: Float.self)
        else { throw ChainError.invalid("Official reference is empty") }
        let native = hidden.dataPointer.assumingMemoryBound(to: Float.self)
        for index in 0..<count {
          let delta = abs(Double(native[index]) - Double(reference[index]))
          maximum = max(maximum, delta)
          total += delta
          referenceTotal += abs(Double(reference[index]))
        }
      }
      print(
        "Native/official 30-block parity: mean-error=\(total / Double(count)) max-error=\(maximum) relative-mean=\(total / max(referenceTotal, 1))"
      )
    }
    print("Wrote \(outputURL.path)")
  }
}
