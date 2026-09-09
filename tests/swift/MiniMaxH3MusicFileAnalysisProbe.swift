import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3MusicFileAnalysisProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 2 else {
      fatalError("usage: MiniMaxH3MusicFileAnalysisProbe AUDIO")
    }
    let url = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
    let totalDuration = try await H3NativeMedia.probeDuration(url)
    let sampleRate = 8_000
    let audio = try await H3NativeMedia.decodeReferenceAudio(
      url: url,
      durationSeconds: totalDuration,
      sampleRate: sampleRate
    )
    let timeline = try H3MusicVideoAnalyzer.timeline(
      audio: audio,
      sampleRate: sampleRate,
      totalDuration: totalDuration,
      maximumDuration: 30,
      minimumDuration: 4,
      preferredDuration: 14
    )
    let chunks = try H3MusicVideoAnalyzer.generationChunks(timeline)
    print("logicalShots=\(timeline.count), generationChunks=\(chunks.count)")
    for (index, segment) in timeline.enumerated() {
      print(
        String(
          format: "%02d  %7.3f–%7.3f  %5.3fs  boundary %.2f  energy %.2f",
          index + 1,
          segment.startSeconds,
          segment.startSeconds + segment.durationSeconds,
          segment.durationSeconds,
          segment.boundaryStrength,
          segment.normalizedEnergy
        )
      )
    }
  }
}
