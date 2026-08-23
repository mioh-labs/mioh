import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3QwenLayerProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 4 else {
      throw H3NativeError.invalidArguments(
        "usage: MiniMaxH3QwenLayerProbe <model.aimodelc> <reference-directory> <actual.f32>"
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
      let hiddenShape = dictionary["hiddenShape"] as? [Int],
      let ropeShape = dictionary["ropeShape"] as? [Int],
      let outputShape = dictionary["outputShape"] as? [Int]
    else {
      throw H3NativeError.invalidArguments("invalid Qwen layer pilot metadata")
    }

    func readTensor(_ name: String, shape: [Int]) throws -> H3Tensor {
      let data = try Data(
        contentsOf: referenceDirectory.appendingPathComponent("\(name).f32")
      )
      let values = data.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: Float.self))
      }
      guard values.count == shape.reduce(1, *) else {
        throw H3NativeError.invalidArguments("\(name) does not match metadata")
      }
      return try H3Tensor(float16: values.map(Float16.init), shape: shape)
    }

    let manifest = H3StageManifest(
      backend: .coreAI,
      asset: modelPath,
      function: "main",
      computeUnits: nil,
      inputs: [
        "hiddenStates": "hidden_states",
        "ropeCosine": "rope_cosine",
        "ropeSine": "rope_sine",
      ],
      outputs: ["hiddenStatesOut": "hidden_states_out"],
      inputConstraints: [
        "hiddenStates": H3TensorConstraint(scalarType: .float16, shape: hiddenShape),
        "ropeCosine": H3TensorConstraint(scalarType: .float16, shape: ropeShape),
        "ropeSine": H3TensorConstraint(scalarType: .float16, shape: ropeShape),
      ],
      outputConstraints: [
        "hiddenStatesOut": H3TensorConstraint(
          scalarType: .float16,
          shape: outputShape
        )
      ]
    )
    let runner = try await H3StageRunner(
      name: "qwenLanguageLayerPilot",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    let result = try await runner.predict([
      "hiddenStates": try readTensor("hidden_states", shape: hiddenShape),
      "ropeCosine": try readTensor("rope_cosine", shape: ropeShape),
      "ropeSine": try readTensor("rope_sine", shape: ropeShape),
    ])
    guard let output = result["hiddenStatesOut"] else {
      throw H3NativeError.missingTensor("qwenLanguageLayerPilot.hiddenStatesOut")
    }
    let actual = try output.floatValues()
    try actual.withUnsafeBufferPointer { buffer in
      try Data(buffer: buffer).write(to: outputURL, options: .atomic)
    }
    let mean = actual.reduce(0, +) / Float(actual.count)
    let rms = sqrt(actual.reduce(0) { $0 + $1 * $1 } / Float(actual.count))
    print("shape=\(output.shape) mean=\(mean) rms=\(rms)")
    print(outputURL.path)
  }
}
