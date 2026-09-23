import Foundation

// Mirrors MacHLSRealtimeProducer.restoreWindow: assemble the video, place each
// segment's audio by its first video frame, then cut consecutive outputs.
// Usage: harness <ffmpeg> <work-dir> <segment.ts>...
@main
struct MacHLSAudioHarness {
  static func main() async throws {
    let arguments = CommandLine.arguments
    let ffmpeg = URL(fileURLWithPath: arguments[1])
    let work = URL(fileURLWithPath: arguments[2], isDirectory: true)
    let segments = arguments.dropFirst(3).map { URL(fileURLWithPath: $0) }

    let assembledURL = work.appendingPathComponent("assembled.mp4")
    let assembled = try await IPadHLSIntervalAssembler.concatenate(
      inputURLs: segments,
      outputURL: assembledURL,
      temporaryDirectory: work
    )
    let decoded = try await MacHLSAudio.decode(urls: segments, ffmpeg: ffmpeg)
    let placements = decoded.indices.map {
      MacHLSAudio.Placement(
        pcm: decoded[$0],
        videoStart: assembled.sourceOffsets[$0],
        videoDuration: assembled.sourceDurations[$0]
      )
    }

    let first = assembled.sourceOffsets[0]
    let last = assembled.sourceOffsets[segments.count - 1]
      + assembled.sourceDurations[segments.count - 1]
    var joined: [Int16] = []
    var start = first
    while start < last - 0.01 {
      let end = min(last, start + 2.002)
      joined += MacHLSAudio.samples(from: start, to: end, placements: placements)
      start = end
    }
    try joined.withUnsafeBytes { Data($0) }
      .write(to: work.appendingPathComponent("joined.pcm"))

    let movieURL = work.appendingPathComponent("output.mov")
    try await MacHLSAudio.writeMovie(
      videoURL: assembledURL,
      samples: MacHLSAudio.samples(from: first, to: first + 2.002, placements: placements),
      duration: 2.002,
      outputURL: movieURL,
      ffmpeg: ffmpeg
    )
    let result: [String: Any] = [
      "first_video_seconds": first,
      "source_offsets": assembled.sourceOffsets,
      "movie": movieURL.path,
    ]
    let data = try JSONSerialization.data(withJSONObject: result)
    print(String(data: data, encoding: .utf8)!)
  }
}
