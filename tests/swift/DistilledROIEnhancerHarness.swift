// SPDX-License-Identifier: AGPL-3.0
import CoreML
import Foundation

@main
struct DistilledROIEnhancerHarness {
  static func main() throws {
    func expect(_ centre: Int, _ indices: [Int], _ wanted: [Int]) throws {
      let actual = try ROIEnhancerTemporalWindow.indices(centre: centre, frameIndices: indices)
      precondition(actual == wanted)
    }
    let frames = [0, 1, 2, 3, 4]
    try expect(0, frames, [0, 0, 0, 1, 2])
    try expect(2, frames, [0, 1, 2, 3, 4])
    try expect(4, frames, [2, 3, 4, 4, 4])
    let gaps = [0, 1, 4, 5, 6]
    let gapRuns = try ROIEnhancerTemporalWindow.contiguousRuns(frameIndices: gaps)
    let singletonRuns = try ROIEnhancerTemporalWindow.contiguousRuns(frameIndices: [2, 4, 5, 9])
    let emptyRuns = try ROIEnhancerTemporalWindow.contiguousRuns(frameIndices: [])
    let oneRun = try ROIEnhancerTemporalWindow.contiguousRuns(frameIndices: [20])
    precondition(gapRuns == [0..<2, 2..<5])
    precondition(singletonRuns == [0..<1, 1..<3, 3..<4])
    precondition(emptyRuns == [])
    precondition(oneRun == [0..<1])
    try expect(1, gaps, [0, 0, 1, 1, 1])
    try expect(2, gaps, [2, 2, 2, 3, 4])
    try expect(0, [20], [0, 0, 0, 0, 0])
    for invalid in [[], [0, 0], [1, 0], [-1, 0]] {
      do {
        _ = try ROIEnhancerTemporalWindow.indices(centre: 0, frameIndices: invalid)
        preconditionFailure("accepted invalid indices")
      } catch is DistilledROIEnhancerError {}
      if !invalid.isEmpty {
        do {
          _ = try ROIEnhancerTemporalWindow.contiguousRuns(frameIndices: invalid)
          preconditionFailure("accepted invalid run indices")
        } catch is DistilledROIEnhancerError {}
      }
    }
    if CommandLine.arguments.count == 1 {
      print("temporal window tests passed")
      return
    }
    let source = URL(fileURLWithPath: CommandLine.arguments[1])
    let compiled = source.pathExtension == "mlpackage" ? try MLModel.compileModel(at: source) : source
    defer { if compiled != source { try? FileManager.default.removeItem(at: compiled) } }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = DistilledROIEnhancer.computeUnits
    precondition(configuration.computeUnits == .cpuAndNeuralEngine)
    let model = try MLModel(contentsOf: compiled, configuration: configuration)
    do {
      _ = try DistilledROIEnhancer(model: model, scale: 4)
      preconditionFailure("accepted incorrect scale")
    } catch is DistilledROIEnhancerError {}
    let enhancer = try DistilledROIEnhancer(model: model, scale: 2)
    let indices = [0, 1, 2, 10, 11, 12, 13]
    let elements = 3 * 256 * 256
    var input = [Float16](repeating: 0, count: indices.count * elements)
    for frame in indices.indices {
      for pixel in 0..<elements {
        input[frame * elements + pixel] = Float16(Float((frame * 17 + pixel % 127) % 251) / 250)
      }
    }
    do {
      _ = try enhancer.enhance(restored: Array(input.dropLast()), centre: 0, frameIndices: indices)
      preconditionFailure("accepted incomplete clip")
    } catch is DistilledROIEnhancerError {}
    var bad = input
    bad[0] = .nan
    do {
      _ = try enhancer.enhance(restored: bad, centre: 0, frameIndices: indices)
      preconditionFailure("accepted nonfinite input")
    } catch is DistilledROIEnhancerError {}
    let destination = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    var timings: [Double] = []
    var windows: [[Int]] = []
    _ = try enhancer.enhance(restored: input, centre: 0, frameIndices: indices)
    for centre in [0, 2, 3, 6] {
      let start = Date()
      let output = try enhancer.enhance(restored: input, centre: centre, frameIndices: indices)
      timings.append(Date().timeIntervalSince(start) * 1000)
      precondition(output.count == 3 * 512 * 512 && output.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
      let values = output.map(Float.init)
      try values.withUnsafeBytes { try Data($0).write(to: destination.appendingPathComponent("frame-\(centre).f32")) }
      windows.append(try ROIEnhancerTemporalWindow.indices(centre: centre, frameIndices: indices))
    }
    let report: [String: Any] = ["compute_units": "CPU_AND_NEURAL_ENGINE", "tested_centres": [0, 2, 3, 6], "windows": windows,
                               "milliseconds_with_tensor_io": timings, "shape": [1, 3, 512, 512]]
    print(String(data: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), encoding: .utf8)!)
  }
}
