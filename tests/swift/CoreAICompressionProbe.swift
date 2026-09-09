import Foundation

@available(macOS 27.0, *)
@main
struct CoreAICompressionProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 2 else {
      throw H3NativeError.invalidArguments(
        "usage: CoreAICompressionProbe <model.aimodelc>"
      )
    }
    let shape = [1, 1024]
    let manifest = H3StageManifest(
      backend: .coreAI,
      asset: CommandLine.arguments[1],
      function: "main",
      computeUnits: nil,
      inputs: ["input": "input"],
      outputs: ["output": "output"],
      inputConstraints: [
        "input": H3TensorConstraint(scalarType: .float16, shape: shape)
      ],
      outputConstraints: [
        "output": H3TensorConstraint(scalarType: .float16, shape: shape)
      ]
    )
    let runner = try await H3StageRunner(
      name: "compressionProbe",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    let values = (0..<1024).map { Float16(sin(Float($0) * 0.03125)) }
    let result = try await runner.predict([
      "input": try H3Tensor(float16: values, shape: shape)
    ])
    guard let output = result["output"] else {
      throw H3NativeError.missingTensor("compressionProbe.output")
    }
    let floats = try output.floatValues()
    let mean = floats.reduce(0, +) / Float(floats.count)
    let rms = sqrt(floats.reduce(0) { $0 + $1 * $1 } / Float(floats.count))
    print("mean=\(mean) rms=\(rms) first=\(floats.prefix(4))")
  }
}
