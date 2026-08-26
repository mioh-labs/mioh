import Foundation

/// Runs the complete DiT block sequence declared by a denoiser composite
/// manifest against the deterministic tensors emitted by the Python exporter.
/// This catches weight-cache collisions that are invisible when only one or
/// two identically-shaped block assets are loaded.
@available(macOS 27.0, *)
@main
struct TenErosMaxH3DiTManifestProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 4 else {
      throw H3NativeError.invalidArguments(
        "usage: TenErosMaxH3DiTManifestProbe "
          + "<denoiser-composite-manifest.json> <reference-directory> <actual.f32>"
      )
    }
    let manifestURL = URL(fileURLWithPath: CommandLine.arguments[1])
      .standardizedFileURL
    let reference = URL(fileURLWithPath: CommandLine.arguments[2])
      .standardizedFileURL
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])
      .standardizedFileURL
    let manifest = try JSONDecoder().decode(
      H3DenoiserCompositeManifest.self,
      from: Data(contentsOf: manifestURL)
    )
    let metadata = try JSONSerialization.jsonObject(
      with: Data(contentsOf: reference.appendingPathComponent("metadata.json"))
    ) as! [String: Any]
    let shapes = metadata["inputShapes"] as! [[Int]]
    let scalarType = H3ScalarType(
      rawValue: metadata["scalarType"] as? String ?? "bfloat16"
    ) ?? .bfloat16
    let files = [
      "hidden_states.f32", "timestep_coordinates.f32",
      "modulation_weights.f32", "rope_cosine.f32", "rope_sine.f32",
    ]

    func tensor(_ index: Int) throws -> H3Tensor {
      let data = try Data(contentsOf: reference.appendingPathComponent(files[index]))
      let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
      return try H3Tensor(float32: values, shape: shapes[index]).converted(to: scalarType)
    }

    let sequence = try await H3CoreAIBlockSequence(
      manifests: manifest.blocks,
      baseDirectory: manifestURL.deletingLastPathComponent(),
      onLoad: { completed, total in
        print("loaded \(completed)/\(total)")
      }
    )
    let output = try await sequence.predict(
      hiddenStates: tensor(0),
      timestepCoordinates: tensor(1),
      modulationWeights: tensor(2),
      ropeCosine: tensor(3),
      ropeSine: tensor(4)
    )
    let values = try output.floatValues()
    try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(
      to: outputURL, options: .atomic
    )
    let rms = sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(values.count))
    print("shape=\(output.shape) rms=\(rms)")
  }
}
