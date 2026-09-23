import Foundation

@main
struct MacHLSAVFoundationCaptureHarness {
  @MainActor
  static func main() async throws {
    guard (4...5).contains(CommandLine.arguments.count),
      let duration = Double(CommandLine.arguments[3]),
      let startSeconds = CommandLine.arguments.count == 5
        ? Double(CommandLine.arguments[4]) : 0
    else {
      throw NSError(
        domain: "MacHLSAVFoundationCaptureHarness",
        code: 2,
        userInfo: [NSLocalizedDescriptionKey: "input output duration"]
      )
    }
    let argument = CommandLine.arguments[1]
    let input = argument.contains("://")
      ? URL(string: argument)!
      : URL(fileURLWithPath: argument)
    let output = URL(
      fileURLWithPath: CommandLine.arguments[2],
      isDirectory: true
    )
    let capture = MacHLSAVFoundationCapture(
      url: input,
      outputDirectory: output,
      startSeconds: startSeconds,
      duration: duration,
      isLive: false,
      generation: 1,
      segmentSeconds: 2,
      forwardBufferSeconds: 60,
      log: { text in
        print("LOG\t" + text.trimmingCharacters(in: .whitespacesAndNewlines))
      }
    )
    defer {
      capture.cancel()
    }

    let stream = try capture.segments()
    var count = 0
    var lastEnd = 0.0
    for try await segment in stream {
      count += 1
      lastEnd = max(lastEnd, segment.endSeconds)
      print("SEGMENT\t\(segment.url.path)")
    }
    guard count >= 3, lastEnd >= duration - 0.20 else {
      throw NSError(
        domain: "MacHLSAVFoundationCaptureHarness",
        code: 3,
        userInfo: [
          NSLocalizedDescriptionKey:
            "capture ended early: segments=\(count), end=\(lastEnd)"
        ]
      )
    }
    // Audio comes from the same player, continuously, on the HLS timeline.
    let covered = capture.audio.pcm(from: startSeconds + 0.5, duration: duration - startSeconds - 1)
    let silentFrames = stride(from: 0, to: covered.samples.count, by: MacHLSAudio.channels)
      .filter { covered.samples[$0] == 0 && covered.samples[$0 + 1] == 0 }.count
    print("AUDIO\tend=\(capture.audio.endSeconds)\tsilent_frames=\(silentFrames)\tframes=\(covered.samples.count / MacHLSAudio.channels)")
    print("Mac HLS AVFoundation accelerated capture passed")
  }
}
