import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3QwenEmbeddingProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 3 else {
      throw H3NativeError.invalidArguments(
        "usage: MiniMaxH3QwenEmbeddingProbe <model.aimodelc> <actual.f32>"
      )
    }
    let sequence = 4152
    let inputShape = [1, sequence]
    let outputShape = [1, sequence, 5120]
    let manifest = H3StageManifest(
      backend: .coreAI,
      asset: CommandLine.arguments[1],
      function: "main",
      computeUnits: nil,
      inputs: ["inputIDs": "input_ids"],
      outputs: ["tokenEmbeddings": "token_embeddings"],
      inputConstraints: [
        "inputIDs": H3TensorConstraint(scalarType: .int32, shape: inputShape)
      ],
      outputConstraints: [
        "tokenEmbeddings": H3TensorConstraint(
          scalarType: .float16,
          shape: outputShape
        )
      ]
    )
    let runner = try await H3StageRunner(
      name: "qwenEmbeddingPilot",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    let ids = (0..<sequence).map { Int32(($0 * 37) % 151936) }
    let result = try await runner.predict([
      "inputIDs": try H3Tensor(int32: ids, shape: inputShape)
    ])
    guard let embedding = result["tokenEmbeddings"] else {
      throw H3NativeError.missingTensor("qwenEmbeddingPilot.tokenEmbeddings")
    }
    let actual = try embedding.floatValues()
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
    try actual.withUnsafeBufferPointer { buffer in
      try Data(buffer: buffer).write(to: outputURL, options: .atomic)
    }
    print("shape=\(embedding.shape) first=\(actual.prefix(4))")
    print(outputURL.path)
  }
}
