import AVFoundation
import CoreMedia
import Foundation

/// Source audio for restored HLS outputs.
///
/// Audio always comes from media the producer already holds: the downloaded
/// segment files, or the PCM of the Safari-compatible capture player. It is
/// placed on the worker input's timeline by each source's first video frame,
/// so every restored output carries exactly the audio of its own interval and
/// consecutive outputs join without gaps or repeats.
enum MacHLSAudio {
  static let sampleRate = 48_000
  static let channels = 2

  /// Interleaved 16-bit PCM whose first frame lies `startOffset` seconds after
  /// the first video frame of the source it belongs to.
  struct SourcePCM: Sendable {
    var samples: [Int16]
    var startOffset: Double

    static let silent = SourcePCM(samples: [], startOffset: 0)
  }

  /// A source on a worker input's timeline. Only its own video span is used,
  /// so audio that a segment carries past its last frame is never duplicated.
  struct Placement: Sendable {
    let pcm: SourcePCM
    let videoStart: Double
    let videoDuration: Double
  }

  enum Failure: LocalizedError {
    case media(String)

    var errorDescription: String? {
      switch self {
      case .media(let detail): return "HLS音声を処理できません: \(detail)"
      }
    }
  }

  /// Decodes the audio of consecutive downloaded segments. MPEG-TS segments
  /// are decoded as one continuous stream so the AAC decoder never restarts at
  /// a segment boundary; each result still refers to its own first video frame.
  static func decode(urls: [URL], ffmpeg: URL) async throws -> [SourcePCM] {
    guard !urls.isEmpty else { return [] }
    let ffprobe = ffmpeg.deletingLastPathComponent().appendingPathComponent("ffprobe")
    var videoStarts: [Double] = []
    for url in urls {
      let starts = try await streamStarts(of: url.path, ffprobe: ffprobe)
      guard let video = starts.video else {
        throw Failure.media("\(url.lastPathComponent)に映像がありません")
      }
      videoStarts.append(video)
    }

    let isTransportStream = urls.allSatisfy {
      $0.pathExtension.lowercased() == "ts"
        || IPadMPEGTSRemuxer.appearsToBeTransportStream($0)
    }
    if isTransportStream {
      let input = "concat:" + urls.map(\.path).joined(separator: "|")
      guard let audioStart = try await streamStarts(of: input, ffprobe: ffprobe).audio
      else { return Array(repeating: .silent, count: urls.count) }
      let samples = try await decodePCM(input: input, ffmpeg: ffmpeg)
      return videoStarts.map {
        SourcePCM(samples: samples, startOffset: audioStart - $0)
      }
    }

    var results: [SourcePCM] = []
    for (url, videoStart) in zip(urls, videoStarts) {
      guard let audioStart = try await streamStarts(of: url.path, ffprobe: ffprobe).audio
      else {
        results.append(.silent)
        continue
      }
      let samples = try await decodePCM(input: url.path, ffmpeg: ffmpeg)
      results.append(SourcePCM(samples: samples, startOffset: audioStart - videoStart))
    }
    return results
  }

  /// PCM for `[start, end)` of a worker input's timeline. Sample positions are
  /// derived from absolute times, so adjacent ranges tile exactly.
  static func samples(from start: Double, to end: Double, placements: [Placement]) -> [Int16] {
    let first = frameIndex(start)
    let frameCount = max(0, frameIndex(end) - first)
    var output = [Int16](repeating: 0, count: frameCount * channels)
    for placement in placements {
      let pcmFrames = placement.pcm.samples.count / channels
      let pcmFirst = frameIndex(placement.videoStart + placement.pcm.startOffset)
      let lower = max(first, frameIndex(placement.videoStart), pcmFirst)
      let upper = min(
        first + frameCount,
        frameIndex(placement.videoStart + placement.videoDuration),
        pcmFirst + pcmFrames
      )
      guard lower < upper else { continue }
      placement.pcm.samples.withUnsafeBufferPointer { source in
        output.withUnsafeMutableBufferPointer { destination in
          let sourceStart = (lower - pcmFirst) * channels
          let destinationStart = (lower - first) * channels
          let count = (upper - lower) * channels
          for index in 0..<count {
            destination[destinationStart + index] = source[sourceStart + index]
          }
        }
      }
    }
    return output
  }

  /// Copies the restored video and adds the PCM as an LPCM track. LPCM in MOV
  /// avoids AAC priming at every two-second queue item boundary.
  static func writeMovie(
    videoURL: URL,
    samples: [Int16],
    duration: Double,
    outputURL: URL,
    ffmpeg: URL
  ) async throws {
    guard duration.isFinite, duration > 0 else {
      throw Failure.media("出力区間が不正です")
    }
    let pcmURL = outputURL.deletingPathExtension().appendingPathExtension("pcm")
    defer { try? FileManager.default.removeItem(at: pcmURL) }
    let data = samples.withUnsafeBytes { Data($0) }
    try data.write(to: pcmURL, options: .atomic)
    try? FileManager.default.removeItem(at: outputURL)
    _ = try await run(ffmpeg, [
      "-nostdin", "-v", "error", "-y",
      "-i", videoURL.path,
      "-f", "s16le", "-ar", "\(sampleRate)", "-ac", "\(channels)", "-i", pcmURL.path,
      "-map", "0:v:0", "-map", "1:a:0",
      "-c:v", "copy", "-c:a", "pcm_s16le",
      "-f", "mov", outputURL.path,
    ])
  }

  private static func frameIndex(_ seconds: Double) -> Int {
    Int((seconds * Double(sampleRate)).rounded())
  }

  private static func streamStarts(
    of input: String,
    ffprobe: URL
  ) async throws -> (video: Double?, audio: Double?) {
    let output = try await run(ffprobe, [
      "-v", "error", "-show_entries", "stream=codec_type,start_time",
      "-of", "json", input,
    ])
    struct Probe: Decodable {
      struct Stream: Decodable {
        let codec_type: String?
        let start_time: String?
      }
      let streams: [Stream]
    }
    let probe = try JSONDecoder().decode(Probe.self, from: output)
    func start(_ type: String) -> Double? {
      probe.streams.first { $0.codec_type == type }
        .flatMap { $0.start_time.flatMap(Double.init) }
        .flatMap { $0.isFinite ? $0 : nil }
    }
    return (start("video"), start("audio"))
  }

  private static func decodePCM(input: String, ffmpeg: URL) async throws -> [Int16] {
    let data = try await run(ffmpeg, [
      "-nostdin", "-v", "error", "-i", input,
      "-map", "0:a:0", "-vn", "-ac", "\(channels)", "-ar", "\(sampleRate)",
      "-f", "s16le", "-",
    ])
    return data.withUnsafeBytes { Array($0.bindMemory(to: Int16.self)) }
  }

  private static func run(_ executable: URL, _ arguments: [String]) async throws -> Data {
    try await Task.detached(priority: .userInitiated) {
      let process = Process()
      let output = Pipe()
      let error = Pipe()
      process.executableURL = executable
      process.arguments = arguments
      process.standardInput = FileHandle.nullDevice
      process.standardOutput = output
      process.standardError = error
      try process.run()
      let errorTask = Task.detached { error.fileHandleForReading.readDataToEndOfFile() }
      let data = output.fileHandleForReading.readDataToEndOfFile()
      process.waitUntilExit()
      let message = String(data: await errorTask.value, encoding: .utf8) ?? ""
      guard process.terminationStatus == 0 else {
        throw Failure.media(
          "\(executable.lastPathComponent)が終了コード\(process.terminationStatus)で失敗しました: "
            + message.trimmingCharacters(in: .whitespacesAndNewlines)
        )
      }
      return data
    }.value
  }
}

/// PCM of the Safari-compatible capture player on the HLS timeline. The
/// player decodes audio once and continuously, so it is never re-seeked.
@MainActor
final class MacHLSPCMTimeline {
  private var firstFrame = 0
  private var samples: [Int16] = []

  var endSeconds: Double {
    samples.isEmpty
      ? -.infinity
      : Double(firstFrame + samples.count / MacHLSAudio.channels)
        / Double(MacHLSAudio.sampleRate)
  }

  func append(startSeconds: Double, interleaved: UnsafeBufferPointer<Int16>) {
    guard startSeconds.isFinite, startSeconds >= 0 else { return }
    let channels = MacHLSAudio.channels
    var start = Int((startSeconds * Double(MacHLSAudio.sampleRate)).rounded())
    var incoming = interleaved[...]
    if samples.isEmpty { firstFrame = start }
    let end = firstFrame + samples.count / channels
    if start < end {
      // Drop what is already stored; audio is never rewritten.
      let overlap = min(incoming.count / channels, end - start)
      incoming = incoming.dropFirst(overlap * channels)
      start += overlap
    } else if start > end {
      samples.append(contentsOf: repeatElement(0, count: (start - end) * channels))
    }
    samples.append(contentsOf: incoming)
  }

  func pcm(from start: Double, duration: Double) -> MacHLSAudio.SourcePCM {
    let channels = MacHLSAudio.channels
    let rate = Double(MacHLSAudio.sampleRate)
    let first = Int((start * rate).rounded())
    let count = max(0, Int(((start + duration) * rate).rounded()) - first)
    var output = [Int16](repeating: 0, count: count * channels)
    let lower = max(first, firstFrame)
    let upper = min(first + count, firstFrame + samples.count / channels)
    if lower < upper {
      output.replaceSubrange(
        (lower - first) * channels..<(upper - first) * channels,
        with: samples[(lower - firstFrame) * channels..<(upper - firstFrame) * channels]
      )
    }
    return MacHLSAudio.SourcePCM(samples: output, startOffset: 0)
  }

  func discard(before seconds: Double) {
    let frame = Int((seconds * Double(MacHLSAudio.sampleRate)).rounded())
    let removable = min(samples.count / MacHLSAudio.channels, frame - firstFrame)
    guard removable > 0 else { return }
    samples.removeFirst(removable * MacHLSAudio.channels)
    firstFrame += removable
  }
}
