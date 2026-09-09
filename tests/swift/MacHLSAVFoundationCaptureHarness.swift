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
    let input = URL(fileURLWithPath: CommandLine.arguments[1])
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
    print("Mac HLS AVFoundation accelerated capture passed")
  }
}
