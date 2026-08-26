import Foundation

@available(macOS 27.0, *)
@main
struct TenErosMaxH3QwenVisionBatchProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 2 else {
      throw H3NativeError.invalidArguments(
        "usage: TenErosMaxH3QwenVisionBatchProbe <qwen-vision-patch.aimodelc>"
      )
    }
    let batch = CommandLine.arguments[1].contains("-b1.") ? 1 : 10
    let isPatch = CommandLine.arguments[1].contains("vision-patch")
    let isDeepstack = CommandLine.arguments[1].contains("deepstack")
    let isMerger = CommandLine.arguments[1].contains("vision-merger")
    let inputShape = [batch, 1620, isPatch ? 1536 : 1152]
    let outputShape = (isDeepstack || isMerger)
      ? [batch, 405, 5120]
      : [batch, 1620, 1152]
    let semanticInput = isPatch ? "pixelPatches" : "visionHidden"
    let physicalInput = isPatch ? "pixel_patches" : "vision_hidden"
    let semanticOutput = isDeepstack
      ? "deepstack"
      : (isMerger ? "visionMerged" : (isPatch ? "visionHidden" : "visionHiddenOut"))
    let physicalOutput = isDeepstack
      ? "deepstack_0"
      : (isMerger ? "vision_merged" : (isPatch ? "vision_hidden" : "vision_hidden_out"))
    let manifest = H3StageManifest(
      backend: .coreAI,
      asset: CommandLine.arguments[1],
      function: "main",
      computeUnits: nil,
      inputs: [semanticInput: physicalInput],
      outputs: [semanticOutput: physicalOutput],
      inputConstraints: [
        semanticInput: H3TensorConstraint(
          scalarType: .float16,
          shape: inputShape
        )
      ],
      outputConstraints: [
        semanticOutput: H3TensorConstraint(
          scalarType: .float16,
          shape: outputShape
        )
      ]
    )
    let runner = try await H3StageRunner(
      name: "qwenVisionPatchBatch",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    let input = try H3Tensor(
      float16: [Float16](repeating: 0, count: inputShape.reduce(1, *)),
      shape: inputShape
    )
    let output = try await runner.predict([semanticInput: input])
    guard let hidden = output[semanticOutput] else {
      throw H3NativeError.missingTensor(semanticOutput)
    }
    let values = try hidden.floatValues()
    let rms = sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(values.count))
    print("shape=\(hidden.shape) rms=\(rms)")
  }
}
