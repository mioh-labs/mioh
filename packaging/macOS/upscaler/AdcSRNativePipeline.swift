// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import CoreAI
import Foundation

@available(macOS 27.0, *)
enum AdcSRComputePolicy: String {
  case hybrid
  case automatic
  case gpu
  case neuralEngine
}

@available(macOS 27.0, *)
private enum AdcSRPipelineError: LocalizedError {
  case missingModel(String)
  case missingFunction(String)
  case invalidContract(String)
  case missingOutput(String)

  var errorDescription: String? {
    switch self {
    case .missingModel(let path):
      return "AdcSR Core AIモデルが見つかりません: \(path)"
    case .missingFunction(let name):
      return "AdcSRモデルに関数 \(name) がありません"
    case .invalidContract(let message):
      return "AdcSRモデルの入出力仕様が不正です: \(message)"
    case .missingOutput(let name):
      return "AdcSRモデルの出力 \(name) がありません"
    }
  }
}

/// Stateless one-step AdcSR graph for the standalone upscaler.
///
/// Contract: `lr` [1,3,128,128] Float32/Float16 in [-1,1] ->
/// `sr` [1,3,512,512] with the matching scalar type. Color matching and tiling
/// are kept outside the graph so they can be applied once to the complete frame.
@available(macOS 27.0, *)
final class AdcSRNativePipeline {
  static let inputSide = 128
  static let outputSide = 512
  static let scale = 4

  private let model: AIModel
  private let function: InferenceFunction
  private let usesFloat16: Bool
  let computeSummary: String

  init(modelLocation: URL, computePolicy: AdcSRComputePolicy) async throws {
    let modelURL = try Self.resolveModel(at: modelLocation)
    let available = ComputeUnitKind.availableKinds
    func preferred(_ kind: ComputeUnitKind) -> SpecializationOptions {
      available.contains(kind)
        ? SpecializationOptions(preferredComputeUnitKind: kind)
        : .default
    }
    let options: SpecializationOptions
    switch computePolicy {
    case .hybrid, .gpu:
      // The released FP32 graph is GPU-oriented. `preferred` still allows
      // per-operation fallback to the other available compute units.
      options = preferred(.gpu)
    case .neuralEngine:
      options = preferred(.neuralEngine)
    case .automatic:
      options = .default
    }

    let loaded = try await AIModel(contentsOf: modelURL, options: options)
    guard let descriptor = loaded.functionDescriptor(for: "main") else {
      throw AdcSRPipelineError.missingFunction("main")
    }
    guard descriptor.stateNames.isEmpty else {
      throw AdcSRPipelineError.invalidContract("状態を持つグラフは使用できません")
    }
    guard case .ndArray(let input) = descriptor.inputDescriptor(of: "lr"),
          case .ndArray(let output) = descriptor.outputDescriptor(of: "sr"),
          input.shape == [1, 3, Self.inputSide, Self.inputSide],
          output.shape == [1, 3, Self.outputSide, Self.outputSide]
    else {
      throw AdcSRPipelineError.invalidContract(
        "lr [1,3,128,128] -> sr [1,3,512,512] が必要です"
      )
    }
    guard input.scalarType == output.scalarType,
          input.scalarType == .float32 || input.scalarType == .float16 else {
      throw AdcSRPipelineError.invalidContract("FP32またはFP16の同型入出力が必要です")
    }
    guard let loadedFunction = try loaded.loadFunction(named: "main") else {
      throw AdcSRPipelineError.missingFunction("main")
    }
    model = loaded
    function = loadedFunction
    usesFloat16 = input.scalarType == .float16
    let precision = input.scalarType == .float16 ? "FP16" : "FP32"
    computeSummary = "\(computePolicy.rawValue) · \(precision) Core AI · available=\(available)"
  }

  func upscale(tile: [Float]) async throws -> NDArray {
    let expected = 3 * Self.inputSide * Self.inputSide
    guard tile.count == expected else {
      throw AdcSRPipelineError.invalidContract(
        "入力要素数 \(tile.count)、期待値 \(expected)"
      )
    }
    var input = NDArray(
      shape: [1, 3, Self.inputSide, Self.inputSide],
      scalarType: usesFloat16 ? .float16 : .float32
    )
    if usesFloat16 {
      var view = input.mutableView(as: Float16.self)
      view.copyElements(fromContentsOf: tile.map(Float16.init))
    } else {
      var view = input.mutableView(as: Float.self)
      view.copyElements(fromContentsOf: tile)
    }
    var outputs = try await function.run(inputs: ["lr": input])
    guard let output = outputs.remove("sr")?.ndArray else {
      throw AdcSRPipelineError.missingOutput("sr")
    }
    guard output.shape == [1, 3, Self.outputSide, Self.outputSide],
          output.scalarType == (usesFloat16 ? .float16 : .float32) else {
      throw AdcSRPipelineError.invalidContract(
        "実行時出力 \(output.shape) / \(output.scalarType)"
      )
    }
    guard output.scalarType == .float16 else { return output }
    var converted = NDArray(shape: output.shape, scalarType: .float32)
    let destination = converted.mutableView(as: Float.self)
    let source = output.view(as: Float16.self)
    var containsNonFinite = false
    destination.withUnsafeMutablePointer { target, _, _ in
      source.withUnsafePointer { values, _, _ in
        for index in 0..<(3 * Self.outputSide * Self.outputSide) {
          let value = Float(values[index])
          if !value.isFinite { containsNonFinite = true }
          target[index] = value
        }
      }
    }
    guard !containsNonFinite else {
      throw AdcSRPipelineError.invalidContract("FP16出力にNaN/Infが含まれています")
    }
    return converted
  }

  private static func resolveModel(at location: URL) throws -> URL {
    let standardized = location.standardizedFileURL
    if standardized.pathExtension == "aimodel",
       FileManager.default.fileExists(atPath: standardized.path) {
      return standardized
    }
    let candidates = [
      standardized.appendingPathComponent("adcsr_x4_float32.aimodel", isDirectory: true),
      standardized.appendingPathComponent(
        "model_weights/adcsr_x4_float32.aimodel", isDirectory: true
      ),
      standardized.appendingPathComponent(
        "AdcSR-CoreAI/adcsr_x4_float32.aimodel", isDirectory: true
      ),
    ]
    if let candidate = candidates.first(where: {
      FileManager.default.fileExists(atPath: $0.path)
    }) {
      return candidate
    }
    throw AdcSRPipelineError.missingModel(standardized.path)
  }
}
