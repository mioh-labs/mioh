import Foundation

@main
struct MiniMaxH3ERSDEProbe {
  static func main() async throws {
    let initial = try H3AVLatent(
      video: [0.25, -0.5, 1.25, -1.5],
      videoShape: [1, 1, 1, 2, 2],
      audio: [0.75, -0.25],
      audioShape: [1, 1, 1, 2]
    )
    let result = try await H3ERSDE.sample(
      initial: initial,
      sigmas: [1, 0.94, 0.83, 0.72, 0.55, 0.30, 0.10, 0],
      flowShift: 12,
      seed: 123,
      sNoise: 0,
      maxStage: 3,
      denoise: { latent, sigma, _ in
        try H3AVLatent(
          video: latent.video.map { 0.4 * $0 + 0.2 * sigma },
          videoShape: latent.videoShape,
          audio: latent.audio.map { 0.4 * $0 - 0.1 * sigma },
          audioShape: latent.audioShape
        )
      }
    )
    print((result.video + result.audio).map { String(format: "%.9f", $0) }.joined(separator: " "))
  }
}
