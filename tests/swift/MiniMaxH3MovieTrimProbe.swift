import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3MovieTrimProbe {
  static func main() async throws {
    let width = 32
    let height = 32
    let frameCount = 36
    let plane = width * height
    var pixels = [Float](repeating: 0, count: 3 * frameCount * plane)
    for channel in 0..<3 {
      for frame in 0..<frameCount {
        let value = Float(frame) / Float(frameCount - 1)
        let start = (channel * frameCount + frame) * plane
        pixels.replaceSubrange(
          start..<(start + plane),
          with: repeatElement(value, count: plane)
        )
      }
    }
    let video = try H3Tensor(
      float32: pixels,
      shape: [1, 3, frameCount, height, width]
    )
    let output = URL(fileURLWithPath: "/tmp/mioh-h3-movie-trim-probe.mp4")
    try await H3NativeMedia.writeMovie(
      video: video,
      audio: nil,
      outputURL: output,
      durationSeconds: 1,
      trimStartSeconds: 0.5,
      outputWidth: width,
      outputHeight: height
    )
    let probe = try await H3NativeMedia.probe(output)
    guard abs(probe.duration - 1) < 0.05 else {
      fatalError("trimmed movie duration is \(probe.duration)")
    }
    let decoded = try await H3NativeMedia.decodeReferenceVideo(
      url: output,
      width: width,
      height: height,
      frameCount: 24
    )
    let values = try decoded.floatValues()
    guard values[0] > 0.2 else {
      fatalError("movie writer did not discard the leading frames: \(values[0])")
    }
    print("ok: duration=\(probe.duration), firstPixel=\(values[0])")
  }
}
