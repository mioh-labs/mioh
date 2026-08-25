import CoreAI
import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3CachedLatentDecodeProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 4 else {
      throw H3NativeError.invalidArguments(
        "usage: MiniMaxH3CachedLatentDecodeProbe <manifest.json> <latent.bin> <output.bin>"
      )
    }
    let manifestURL = URL(fileURLWithPath: CommandLine.arguments[1])
      .standardizedFileURL
    let latentURL = URL(fileURLWithPath: CommandLine.arguments[2])
      .standardizedFileURL
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])
      .standardizedFileURL
    let manifest = try JSONDecoder().decode(
      H3PipelineManifest.self,
      from: Data(contentsOf: manifestURL)
    )
    guard let stage = manifest.stages["videoDecoder"] else {
      throw H3NativeError.missingStage("videoDecoder")
    }
    let runner = try await H3StageRunner(
      name: "videoDecoder",
      manifest: stage,
      baseDirectory: manifestURL.deletingLastPathComponent()
    )
    let latent = try H3Tensor(
      shape: [1, 24, 72, 30, 54],
      scalarType: .float16,
      bytes: Data(contentsOf: latentURL)
    )
    let decoded = try await H3VideoVAEDecoder.decode(
      latent: latent,
      runner: runner,
      progress: { completed, total in
        print("tile \(completed)/\(total)")
      }
    )
    try decoded.bytes.write(to: outputURL, options: .atomic)
    let metadata: [String: Any] = [
      "shape": decoded.shape,
      "scalarType": decoded.scalarType.rawValue,
    ]
    try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted])
      .write(to: outputURL.appendingPathExtension("json"), options: .atomic)
    print("output=\(outputURL.path) shape=\(decoded.shape)")
  }
}
