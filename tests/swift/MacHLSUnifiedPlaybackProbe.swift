import AVFoundation
import CoreMedia
import Foundation

@main struct MacHLSUnifiedPlaybackProbe {
  @MainActor static func main() async throws {
    guard #available(macOS 27.0, *) else { return }
    let source = URL(string: CommandLine.arguments[1])!
    let video = URL(fileURLWithPath: CommandLine.arguments[2])
    let destination = URL(fileURLWithPath: CommandLine.arguments[3], isDirectory: true)
    let audio = MacHLSUnifiedPlayback(item: AVPlayerItem(url: source), start: 1.125, duration: 8, isLive: false)
    defer { audio.cancel() }
    let boundaries = [1.125, 3.127, 5.129, 7.131]
    for index in 0..<boundaries.count - 1 {
      let output = destination.appendingPathComponent("combined-\(index)-\(UUID().uuidString).mov")
      let started = Date()
      try await audio.movie(videoURL: video, start: boundaries[index], end: boundaries[index + 1], outputURL: output)
      let asset = AVURLAsset(url: output)
      let tracks = try await asset.load(.tracks)
      guard tracks.filter({ $0.mediaType == .video }).count == 1,
        tracks.filter({ $0.mediaType == .audio }).count == 1
      else { fatalError("Output must have video and audio in one item") }
      let actualDuration = try await asset.load(.duration).seconds
      guard abs(actualDuration - 2.002) < 0.035 else { fatalError("Wrong duration: \(actualDuration)") }
      print("combined", index, "duration", actualDuration, "seconds", Date().timeIntervalSince(started))
    }
    print("Mac HLS unified playback passed")
  }
}
