import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3VideoVAETilerProbe {
  static func main() async throws {
    guard (3...4).contains(CommandLine.arguments.count),
      let backend = H3BackendKind(rawValue: CommandLine.arguments[1])
    else {
      throw H3NativeError.invalidArguments(
        "usage: MiniMaxH3VideoVAETilerProbe <coreai|coreml> <model> [encoder|decoder]"
      )
    }
    let mode = CommandLine.arguments.count == 4 ? CommandLine.arguments[3] : "encoder"
    guard mode == "encoder" || mode == "decoder" else {
      throw H3NativeError.invalidArguments("probe mode must be encoder or decoder")
    }
    let isDecoder = mode == "decoder"
    let scalarType: H3ScalarType = backend == .coreAI ? .float16 : .float32
    let manifest = H3StageManifest(
      backend: backend,
      asset: CommandLine.arguments[2],
      function: "main",
      computeUnits: backend == .coreML ? "cpuOnly" : nil,
      inputs: isDecoder
        ? ["videoLatentTile": "video_latent_tile"]
        : ["videoTile": "video_tile"],
      outputs: isDecoder
        ? ["videoRawTile": "video_raw_tile"]
        : ["videoLatentTile": "video_latent_tile"],
      inputConstraints: [
        isDecoder ? "videoLatentTile" : "videoTile": H3TensorConstraint(
          scalarType: scalarType,
          shape: isDecoder ? [1, 24, 7, 16, 16] : [1, 3, 17, 256, 256]
        )
      ],
      outputConstraints: [
        isDecoder ? "videoRawTile" : "videoLatentTile": H3TensorConstraint(
          scalarType: isDecoder ? .float32 : scalarType,
          shape: isDecoder ? [1, 3, 28, 256, 256] : [1, 24, 5, 16, 16]
        )
      ]
    )
    let runner = try await H3StageRunner(
      name: isDecoder ? "videoDecoder" : "videoEncoder",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    let output: H3Tensor
    if isDecoder {
      let latent = try H3Tensor(
        float16: Array(repeating: 0, count: 1 * 24 * 2 * 16 * 16),
        shape: [1, 24, 2, 16, 16]
      )
      output = try await H3VideoVAEDecoder.decode(latent: latent, runner: runner)
      guard output.shape == [1, 3, 5, 256, 256] else {
        throw H3NativeError.invalidTensor("unexpected decoded shape \(output.shape)")
      }
    } else {
      let video = try H3Tensor(
        float16: Array(repeating: 0, count: 1 * 3 * 17 * 256 * 256),
        shape: [1, 3, 17, 256, 256]
      )
      output = try await H3VideoVAEEncoder.encode(video: video, runner: runner)
      guard output.shape == [1, 24, 2, 16, 16] else {
        throw H3NativeError.invalidTensor("unexpected stitched shape \(output.shape)")
      }
    }
    let values = try output.floatValues()
    let mean = values.reduce(0, +) / Float(values.count)
    print("backend=\(backend.rawValue) mode=\(mode) shape=\(output.shape) mean=\(mean)")
  }
}
