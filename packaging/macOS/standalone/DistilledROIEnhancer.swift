// SPDX-License-Identifier: AGPL-3.0
import CoreML
import Foundation

enum DistilledROIEnhancerError: LocalizedError {
  case invalidInput(String)
  var errorDescription: String? {
    switch self {
    case .invalidInput(let message): return "ROI Distill: \(message)"
    }
  }
}

/// A window belongs to one restored ROI clip, never to another object's ROI.
/// Repeat the nearest available edge at clip boundaries and detection gaps.
enum ROIEnhancerTemporalWindow {
  /// Frame-processor sessions must not bridge missing observations. Return
  /// clip-local ranges, retaining singleton runs for the normal compositor.
  static func contiguousRuns(frameIndices: [Int]) throws -> [Range<Int>] {
    guard frameIndices.allSatisfy({ $0 >= 0 }),
      zip(frameIndices, frameIndices.dropFirst()).allSatisfy({ $0 < $1 })
    else { throw DistilledROIEnhancerError.invalidInput("invalid temporal frame indices") }
    guard !frameIndices.isEmpty else { return [] }
    var runs: [Range<Int>] = []
    var start = 0
    for index in frameIndices.indices.dropFirst() {
      if frameIndices[index] - frameIndices[index - 1] != 1 {
        runs.append(start..<index)
        start = index
      }
    }
    runs.append(start..<frameIndices.count)
    return runs
  }

  static func indices(centre: Int, frameIndices: [Int]) throws -> [Int] {
    guard frameIndices.indices.contains(centre),
      frameIndices.allSatisfy({ $0 >= 0 }),
      zip(frameIndices, frameIndices.dropFirst()).allSatisfy({ $0 < $1 })
    else { throw DistilledROIEnhancerError.invalidInput("invalid temporal frame indices") }
    var first = centre
    var last = centre
    while first > 0, centre - first < 2,
      frameIndices[first] - frameIndices[first - 1] == 1 { first -= 1 }
    while last + 1 < frameIndices.count, last - centre < 2,
      frameIndices[last + 1] - frameIndices[last] == 1 { last += 1 }
    return (-2...2).map { min(max(centre + $0, first), last) }
  }
}

/// Explicit five-frame tensor backend. Existing image-only enhancers and the
/// original SwiftVR worker are not involved in this experimental model path.
final class DistilledROIEnhancer {
  /// Permit Neural Engine and CPU fallback; do not compete with restoration on GPU.
  static let computeUnits: MLComputeUnits = .cpuAndNeuralEngine
  static let side = 256
  static let outputSide = 512
  private let model: MLModel
  private let input: MLMultiArray

  init(model: MLModel, scale: Int) throws {
    let description = model.modelDescription
    guard scale == 2,
      description.inputDescriptionsByName.count == 1,
      description.outputDescriptionsByName.count == 1,
      let source = description.inputDescriptionsByName["frames"]?.multiArrayConstraint,
      let target = description.outputDescriptionsByName["enhanced"]?.multiArrayConstraint,
      source.shape.map(\.intValue) == [1, 15, 256, 256],
      target.shape.map(\.intValue) == [1, 3, 512, 512],
      source.dataType == .float32, target.dataType == .float32
    else {
      throw DistilledROIEnhancerError.invalidInput(
        "requires FP32 frames [1,15,256,256] → enhanced [1,3,512,512], scale 2")
    }
    self.model = model
    input = try MLMultiArray(shape: [1, 15, 256, 256], dataType: .float32)
    guard input.strides.map(\.intValue) == [983040, 65536, 256, 1] else {
      throw DistilledROIEnhancerError.invalidInput("non-contiguous input tensor")
    }
  }

  func enhance(restored: [Float16], centre: Int, frameIndices: [Int]) throws -> [Float16] {
    let elements = Self.side * Self.side * 3
    guard restored.count.isMultiple(of: elements),
      restored.count / elements == frameIndices.count
    else { throw DistilledROIEnhancerError.invalidInput("restored clip dimensions do not match") }
    let window = try ROIEnhancerTemporalWindow.indices(centre: centre, frameIndices: frameIndices)
    let destination = input.dataPointer.assumingMemoryBound(to: Float.self)
    try restored.withUnsafeBufferPointer { source in
      for (slot, frame) in window.enumerated() {
        for pixel in 0..<elements {
          let value = Float(source[frame * elements + pixel])
          guard value.isFinite else {
            throw DistilledROIEnhancerError.invalidInput("non-finite restored input")
          }
          destination[slot * elements + pixel] = min(max(value, 0), 1)
        }
      }
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: ["frames": MLFeatureValue(multiArray: input)])
    let prediction = try model.prediction(from: provider)
    guard let output = prediction.featureValue(for: "enhanced")?.multiArrayValue,
      output.dataType == .float32,
      output.shape.map(\.intValue) == [1, 3, 512, 512],
      output.strides.map(\.intValue) == [786432, 262144, 512, 1]
    else { throw DistilledROIEnhancerError.invalidInput("unexpected output tensor layout") }
    let values = output.dataPointer.assumingMemoryBound(to: Float.self)
    var pixels = [Float16](repeating: 0, count: output.count)
    for index in pixels.indices {
      let value = values[index]
      guard value.isFinite else {
        throw DistilledROIEnhancerError.invalidInput("non-finite model output")
      }
      pixels[index] = Float16(min(max(value, 0), 1))
    }
    return pixels
  }
}
