import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3AudioEncoderProbe {
  static func main() async throws {
    guard (3...4).contains(CommandLine.arguments.count),
      let backend = H3BackendKind(rawValue: CommandLine.arguments[1])
    else {
      throw H3NativeError.invalidArguments(
        "usage: MiniMaxH3AudioEncoderProbe <coreai|coreml> <model> [encoder|decoder|encoder10|decoder10|video32|video256|video256f32|videoDecoder]"
      )
    }
    let modelURL = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
    let mode = CommandLine.arguments.count == 4 ? CommandLine.arguments[3] : "encoder"
    guard ["encoder", "decoder", "encoder10", "decoder10", "video32", "video256", "video256f32", "videoDecoder"]
      .contains(mode)
    else {
      throw H3NativeError.invalidArguments("unknown audio stage probe mode")
    }
    let isVideoDecoder = mode == "videoDecoder"
    let isVideo = mode.hasPrefix("video")
    let isEncoder = mode.hasPrefix("encoder")
    let production = mode.hasSuffix("10")
    let videoSize = mode.hasPrefix("video256") ? 256 : 32
    let semanticInput = isVideo
      ? (isVideoDecoder ? "videoLatentTile" : "videoTile")
      : (isEncoder ? "audio" : "audioLatent")
    let featureInput = isVideo
      ? (isVideoDecoder ? "video_latent_tile" : "video_tile")
      : (isEncoder ? "audio" : "audio_latent")
    let semanticOutput = isVideo
      ? (isVideoDecoder ? "videoRawTile" : "videoLatentTile")
      : (isEncoder ? "referenceAudioLatent" : "audio")
    let featureOutput = isVideo
      ? (isVideoDecoder ? "video_raw_tile" : "video_latent_tile")
      : (isEncoder ? "reference_audio_latent" : "audio")
    let inputShape = isVideoDecoder
      ? [1, 24, 7, 16, 16]
      : isVideo
      ? [1, 3, 17, videoSize, videoSize]
      : (isEncoder
        ? [1, 2, production ? 320_000 : 3_200]
        : [1, 32, 2, production ? 405 : 4])
    let outputShape = isVideoDecoder
      ? [1, 3, 28, 256, 256]
      : isVideo
      ? [1, 24, 5, videoSize / 16, videoSize / 16]
      : (isEncoder
        ? [1, 32, 2, production ? 400 : 4]
        : [1, 2, production ? 324_000 : 3_200])
    let inputScalarType: H3ScalarType = isVideo && !mode.hasSuffix("f32")
      && !(isVideoDecoder && backend == .coreML) ? .float16 : .float32
    let outputScalarType: H3ScalarType = isVideoDecoder
      ? .float32 : inputScalarType
    let manifest = H3StageManifest(
      backend: backend,
      asset: modelURL.path,
      function: "main",
      computeUnits: backend == .coreML && isVideo ? "cpuOnly" : "all",
      inputs: [semanticInput: featureInput],
      outputs: [semanticOutput: featureOutput],
      inputConstraints: [
        semanticInput: H3TensorConstraint(scalarType: inputScalarType, shape: inputShape)
      ],
      outputConstraints: [
        semanticOutput: H3TensorConstraint(
          scalarType: outputScalarType, shape: outputShape
        )
      ]
    )
    let runner = try await H3StageRunner(
      name: "audioEncoder",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    let input = inputScalarType == .float16
      ? try H3Tensor(
        float16: Array(repeating: Float16(0), count: inputShape.reduce(1, *)),
        shape: inputShape
      )
      : try H3Tensor(
        float32: Array(repeating: 0, count: inputShape.reduce(1, *)),
        shape: inputShape
      )
    let outputs = try await runner.predict([semanticInput: input])
    guard let output = outputs[semanticOutput] else {
      throw H3NativeError.missingTensor(semanticOutput)
    }
    let values = try output.floatValues()
    let mean = values.reduce(0, +) / Float(values.count)
    let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
      / Float(values.count)
    print(
      "backend=\(backend.rawValue) shape=\(output.shape) "
        + "mean=\(mean) std=\(sqrt(variance))"
    )
  }
}
