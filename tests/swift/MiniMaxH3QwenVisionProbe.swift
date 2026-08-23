import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3QwenVisionProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 4 else {
      throw H3NativeError.invalidArguments(
        "usage: MiniMaxH3QwenVisionProbe <model.aimodelc> <reference-directory> <actual.f32>"
      )
    }
    let referenceDirectory = URL(fileURLWithPath: CommandLine.arguments[2])
    let metadataData = try Data(
      contentsOf: referenceDirectory.appendingPathComponent("metadata.json")
    )
    guard
      let metadata = try JSONSerialization.jsonObject(with: metadataData)
        as? [String: Any],
      let inputName = metadata["inputName"] as? String,
      let outputName = metadata["outputName"] as? String,
      let inputShape = metadata["inputShape"] as? [Int],
      let outputShape = metadata["outputShape"] as? [Int]
    else {
      throw H3NativeError.invalidArguments("invalid Qwen vision metadata")
    }
    let inputData = try Data(
      contentsOf: referenceDirectory.appendingPathComponent("input.f32")
    )
    let input = inputData.withUnsafeBytes { raw in
      Array(raw.bindMemory(to: Float.self))
    }
    let semanticInput = "input"
    let semanticOutput = "output"
    let manifest = H3StageManifest(
      backend: .coreAI,
      asset: CommandLine.arguments[1],
      function: "main",
      computeUnits: nil,
      inputs: [semanticInput: inputName],
      outputs: [semanticOutput: outputName],
      inputConstraints: [
        semanticInput: H3TensorConstraint(scalarType: .float16, shape: inputShape)
      ],
      outputConstraints: [
        semanticOutput: H3TensorConstraint(scalarType: .float16, shape: outputShape)
      ]
    )
    let runner = try await H3StageRunner(
      name: "qwenVisionPilot",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    let result = try await runner.predict([
      semanticInput: try H3Tensor(float16: input.map(Float16.init), shape: inputShape)
    ])
    guard let output = result[semanticOutput] else {
      throw H3NativeError.missingTensor("qwenVisionPilot.output")
    }
    let actual = try output.floatValues()
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])
    try actual.withUnsafeBufferPointer { buffer in
      try Data(buffer: buffer).write(to: outputURL, options: .atomic)
    }
    let mean = actual.reduce(0, +) / Float(actual.count)
    let rms = sqrt(actual.reduce(0) { $0 + $1 * $1 } / Float(actual.count))
    print("shape=\(output.shape) mean=\(mean) rms=\(rms)")
    print(outputURL.path)
  }
}
