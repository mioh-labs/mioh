import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3QwenLinearProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 4 else {
      throw H3NativeError.invalidArguments(
        "usage: MiniMaxH3QwenLinearProbe <model.aimodelc> <reference-directory> <actual.f32>"
      )
    }
    let modelPath = CommandLine.arguments[1]
    let referenceDirectory = URL(fileURLWithPath: CommandLine.arguments[2])
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])
    let metadataData = try Data(
      contentsOf: referenceDirectory.appendingPathComponent("metadata.json")
    )
    let metadata = try JSONSerialization.jsonObject(with: metadataData)
    guard
      let dictionary = metadata as? [String: Any],
      let inputShape = dictionary["inputShape"] as? [Int],
      let outputShape = dictionary["outputShape"] as? [Int]
    else {
      throw H3NativeError.invalidArguments("invalid pilot metadata")
    }
    let inputData = try Data(
      contentsOf: referenceDirectory.appendingPathComponent("input.f32")
    )
    let inputFloats = inputData.withUnsafeBytes { rawBuffer -> [Float] in
      Array(rawBuffer.bindMemory(to: Float.self))
    }
    guard inputFloats.count == inputShape.reduce(1, *) else {
      throw H3NativeError.invalidArguments("pilot input byte count does not match metadata")
    }

    let manifest = H3StageManifest(
      backend: .coreAI,
      asset: modelPath,
      function: "main",
      computeUnits: nil,
      inputs: ["hiddenStates": "hidden_states"],
      outputs: ["projected": "projected"],
      inputConstraints: [
        "hiddenStates": H3TensorConstraint(scalarType: .float16, shape: inputShape)
      ],
      outputConstraints: [
        "projected": H3TensorConstraint(scalarType: .float16, shape: outputShape)
      ]
    )
    let runner = try await H3StageRunner(
      name: "qwenNVFP4Pilot",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    let result = try await runner.predict([
      "hiddenStates": try H3Tensor(
        float16: inputFloats.map(Float16.init),
        shape: inputShape
      )
    ])
    guard let projected = result["projected"] else {
      throw H3NativeError.missingTensor("qwenNVFP4Pilot.projected")
    }
    let actual = try projected.floatValues()
    guard actual.count == outputShape.reduce(1, *) else {
      throw H3NativeError.invalidTensor("unexpected pilot output shape")
    }
    let outputData = actual.withUnsafeBufferPointer { Data(buffer: $0) }
    try outputData.write(to: outputURL, options: .atomic)
    let mean = actual.reduce(0, +) / Float(actual.count)
    let rms = sqrt(actual.reduce(0) { $0 + $1 * $1 } / Float(actual.count))
    print("shape=\(projected.shape) mean=\(mean) rms=\(rms)")
    print(outputURL.path)
  }
}
