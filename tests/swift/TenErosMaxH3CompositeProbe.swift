import Foundation

@available(macOS 27.0, *)
@main
struct TenErosMaxH3CompositeProbe {
  static func main() async throws {
    guard (2...3).contains(CommandLine.arguments.count) else {
      throw H3NativeError.invalidArguments(
        "usage: TenErosMaxH3CompositeProbe <denoiser-composite-manifest.json> "
          + "[block-source-manifest.json]"
      )
    }
    let manifestURL = URL(fileURLWithPath: CommandLine.arguments[1])
      .standardizedFileURL
    var manifestData = try Data(contentsOf: manifestURL)
    if CommandLine.arguments.count == 3 {
      let blocksURL = URL(fileURLWithPath: CommandLine.arguments[2])
        .standardizedFileURL
      var base = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
      let blockSource = try JSONSerialization.jsonObject(
        with: Data(contentsOf: blocksURL)
      ) as! [String: Any]
      base["blocks"] = blockSource["blocks"]
      manifestData = try JSONSerialization.data(withJSONObject: base)
    }
    let manifest = try JSONDecoder().decode(
      H3DenoiserCompositeManifest.self, from: manifestData
    )
    let root = manifestURL.deletingLastPathComponent()
    try manifest.validate(relativeTo: root)
    let denoiser = try await TenErosMaxH3DenoiserComposite(
      manifest: manifest,
      baseDirectory: root,
      onLoad: { completed, total in
        if completed == 1 || completed % 10 == 0 || completed == total {
          print("loaded \(completed)/\(total)")
        }
      }
    )

    // Keep the probe close to the 256-token representative shape used when
    // exporting dynamic DiT assets.  The previous 13-token toy sequence made
    // late residual blocks numerically ill-conditioned and exaggerated BF16
    // drift that does not occur at the exported operating point.
    let videoShape = [1, 24, 2, 10, 16]
    let audioShape = [1, 32, 2, 23]
    var random = H3SplitMix64(seed: 7)
    let referenceVideo = try H3Tensor(
      float32: random.normal(count: videoShape.reduce(1, *)),
      shape: videoShape
    ).converted(to: .float16)
    let referenceAudio = try H3Tensor(
      float32: random.normal(count: audioShape.reduce(1, *)),
      shape: audioShape
    )
    let prepared = try await denoiser.prepare(
      context: H3Tensor(
        float16: [Float16](repeating: 0, count: 5120),
        shape: [1, 1, 5120]
      ),
      tokenTags: H3Tensor(int32: [1], shape: [1, 1]),
      referenceVideoLatent: referenceVideo,
      referenceAudioLatent: referenceAudio,
      targetVideoShape: videoShape,
      targetAudioShape: audioShape,
      seed: 7,
      visualConditionNoiseAug: 0.999,
      audioConditionNoiseAug: 1
    )
    let latent = try H3AVLatent(
      video: random.normal(count: videoShape.reduce(1, *)),
      videoShape: videoShape,
      audio: random.normal(count: audioShape.reduce(1, *)),
      audioShape: audioShape
    )
    if let path = ProcessInfo.processInfo.environment["H3_BLOCK_INPUT_DIRECTORY"] {
      let directory = URL(fileURLWithPath: path).standardizedFileURL
      try FileManager.default.createDirectory(
        at: directory, withIntermediateDirectories: true
      )
      let input = try await denoiser.debugBlockInput(
        latent, sigma: 0.55, prepared: prepared,
        videoShift: 12, audioShift: 3
      )
      let tensors: [(String, H3Tensor)] = [
        ("hidden_states.f32", input.hidden),
        ("timestep_coordinates.f32", input.timestepCoordinates),
        ("modulation_weights.f32", input.modulationWeights),
        ("rope_cosine.f32", input.ropeCosine),
        ("rope_sine.f32", input.ropeSine),
      ]
      for (name, tensor) in tensors {
        let values = try tensor.floatValues()
        try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(
          to: directory.appendingPathComponent(name), options: .atomic
        )
      }
      let metadata: [String: Any] = [
        "inputShapes": tensors.map { $0.1.shape },
        "outputShape": input.hidden.shape,
        "scalarType": "bfloat16",
      ]
      try JSONSerialization.data(
        withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]
      ).write(to: directory.appendingPathComponent("metadata.json"), options: .atomic)
      if ProcessInfo.processInfo.environment["H3_BLOCK_INPUT_ONLY"] == "1" {
        print("wrote block input to \(directory.path)")
        return
      }
    }
    let output = try await denoiser.denoise(
      latent,
      sigma: 0.55,
      prepared: prepared,
      videoShift: 12,
      audioShift: 3
    )
    func rms(_ values: [Float]) -> Float {
      sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(values.count))
    }
    guard output.video.allSatisfy(\.isFinite),
      output.audio.allSatisfy(\.isFinite)
    else {
      throw H3NativeError.inference("10Eros composite produced non-finite output")
    }
    print("tokens=\(prepared.totalRows)")
    print("videoRMS=\(rms(output.video))")
    print("audioRMS=\(rms(output.audio))")
  }
}
