// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import CoreAI
import Foundation

@available(macOS 27.0, *)
private enum ProbeError: LocalizedError {
  case argument(String)
  case nonFinite(String)

  var errorDescription: String? {
    switch self {
    case .argument(let message), .nonFinite(let message): return message
    }
  }
}

@available(macOS 27.0, *)
private struct ProbeArguments {
  let model: URL
  let outputDirectory: URL
  let iterations: Int
  let inputTiles: [URL]

  static func parse() throws -> ProbeArguments {
    let values = Array(CommandLine.arguments.dropFirst())
    guard values.count >= 2 else {
      throw ProbeError.argument(
        "usage: adcsr-model-probe MODEL.aimodel OUTPUT_DIR [ITERATIONS]"
      )
    }
    let iterations = values.count >= 3 ? Int(values[2]) ?? 3 : 3
    guard iterations > 0 else { throw ProbeError.argument("ITERATIONS must be positive") }
    return ProbeArguments(
      model: URL(fileURLWithPath: values[0]),
      outputDirectory: URL(fileURLWithPath: values[1]),
      iterations: iterations,
      inputTiles: values.dropFirst(3).map(URL.init(fileURLWithPath:))
    )
  }
}

@available(macOS 27.0, *)
private func probeInputs(extraTiles: [URL]) throws -> [(String, [Float])] {
  let count = 3 * AdcSRNativePipeline.inputSide * AdcSRNativePipeline.inputSide
  var lowVariance = [Float](repeating: -0.2, count: count)
  lowVariance[64 * AdcSRNativePipeline.inputSide + 64] += 1 / 255
  var structured = [Float](repeating: 0, count: count)
  let plane = AdcSRNativePipeline.inputSide * AdcSRNativePipeline.inputSide
  for channel in 0..<3 {
    for y in stride(from: 0, to: AdcSRNativePipeline.inputSide, by: 4) {
      for x in stride(from: 0, to: AdcSRNativePipeline.inputSide, by: 4) {
        structured[channel * plane + y * AdcSRNativePipeline.inputSide + x] = 0.5
      }
    }
  }
  var probes = [
    ("flat-zero", [Float](repeating: 0, count: count)),
    ("flat-gray", [Float](repeating: 0.1, count: count)),
    ("low-variance", lowVariance),
    ("structured", structured),
  ]
  for tile in extraTiles {
    let data = try Data(contentsOf: tile)
    guard data.count == count * MemoryLayout<Float>.stride else {
      throw ProbeError.argument("invalid Float32 tile size: \(tile.path)")
    }
    let values = data.withUnsafeBytes { raw in
      Array(raw.bindMemory(to: Float.self))
    }
    probes.append((tile.deletingPathExtension().lastPathComponent, values))
  }
  return probes
}

@available(macOS 27.0, *)
private func floats(from output: NDArray) throws -> [Float] {
  guard output.scalarType == .float32 else {
    throw ProbeError.argument("pipeline must return Float32, got \(output.scalarType)")
  }
  let count = output.shape.reduce(1, *)
  var values = [Float](repeating: 0, count: count)
  let view = output.view(as: Float.self)
  view.withUnsafePointer { source, _, _ in
    values.withUnsafeMutableBufferPointer { destination in
      destination.baseAddress?.update(from: source, count: count)
    }
  }
  return values
}

@available(macOS 27.0, *)
private func write(_ values: [Float], to url: URL) throws {
  let data = values.withUnsafeBytes { Data($0) }
  try data.write(to: url, options: .atomic)
}

@available(macOS 27.0, *)
private func stats(_ values: [Float]) -> (minimum: Float, maximum: Float, mean: Double, rms: Double) {
  var minimum = Float.infinity
  var maximum = -Float.infinity
  var sum = 0.0
  var squared = 0.0
  for value in values {
    minimum = min(minimum, value)
    maximum = max(maximum, value)
    sum += Double(value)
    squared += Double(value) * Double(value)
  }
  return (minimum, maximum, sum / Double(values.count), sqrt(squared / Double(values.count)))
}

@available(macOS 27.0, *)
private func run() async throws {
  let arguments = try ProbeArguments.parse()
  try FileManager.default.createDirectory(
    at: arguments.outputDirectory, withIntermediateDirectories: true
  )
  let pipeline = try await AdcSRNativePipeline(
    modelLocation: arguments.model, computePolicy: .gpu
  )
  print("MODEL \(arguments.model.path)")
  print("COMPUTE \(pipeline.computeSummary)")

  var structuredInput: [Float] = []
  for (name, input) in try probeInputs(extraTiles: arguments.inputTiles) {
    if name == "structured" { structuredInput = input }
    let started = ContinuousClock.now
    let output = try await pipeline.upscale(tile: input)
    let elapsed = started.duration(to: .now)
    let values = try floats(from: output)
    guard values.allSatisfy(\.isFinite) else {
      throw ProbeError.nonFinite("NON_FINITE \(name)")
    }
    let summary = stats(values)
    try write(values, to: arguments.outputDirectory.appendingPathComponent("\(name).f32"))
    print(
      "PROBE \(name) seconds=\(elapsed.components.seconds).\(elapsed.components.attoseconds) "
        + "min=\(summary.minimum) max=\(summary.maximum) "
        + "mean=\(summary.mean) rms=\(summary.rms)"
    )
  }

  var samples: [Double] = []
  for _ in 0..<arguments.iterations {
    let started = ContinuousClock.now
    _ = try await pipeline.upscale(tile: structuredInput)
    let elapsed = started.duration(to: .now)
    samples.append(
      Double(elapsed.components.seconds)
        + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000_000
    )
  }
  let mean = samples.reduce(0, +) / Double(samples.count)
  print("TIMING iterations=\(samples.count) mean_seconds=\(mean) samples=\(samples)")
}

@main
struct AdcSRModelProbe {
  static func main() async {
    guard #available(macOS 27.0, *) else {
      FileHandle.standardError.write(Data("macOS 27 is required\n".utf8))
      Foundation.exit(2)
    }
    do {
      try await run()
    } catch {
      FileHandle.standardError.write(Data("ERROR \(error.localizedDescription)\n".utf8))
      Foundation.exit(1)
    }
  }
}
