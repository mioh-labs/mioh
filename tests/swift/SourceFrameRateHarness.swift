import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

@main
struct SourceFrameRateHarness {
  static func main() async throws {
    if CommandLine.arguments.count == 2 {
      let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
      let track = try await asset.loadTracks(withMediaType: .video)[0]
      let average = Double(try await track.load(.nominalFrameRate))
      let rate = try await SourceFrameRate.probe(asset: asset, track: track, averageFPS: average)
      print("average=\(average) cadence=\(rate.numerator)/\(rate.denominator)")
      return
    }
    let normal = CMTime(value: 3003, timescale: 90000)
    let samples = Array(repeating: normal, count: 120)
    let average = 29.935535430908203
    var cases: [String: Any] = [:]
    func record(_ name: String, average: Double, durations: [CMTime]) {
      let rate = SourceFrameRate.measured(averageFPS: average, durations: durations)
      cases[name] = ["numerator": rate.numerator, "denominator": rate.denominator,
                     "matches_ntsc": SourceFrameRate.matches(rate, (30000, 1001))]
    }
    record("gaps", average: average, durations: samples
      + Array(repeating: CMTime(value: 186186, timescale: 90000), count: 5))
    record("short_first_sample", average: average,
      durations: [CMTime(value: 2913, timescale: 90000)] + samples)
    record("equivalent_timebases", average: average, durations:
      Array(repeating: normal, count: 60)
        + Array(repeating: CMTime(value: 1001, timescale: 30000), count: 60))
    record("true_29936", average: 29.936,
      durations: Array(repeating: CMTime(value: 125, timescale: 3742), count: 120))
    record("variable", average: 20, durations:
      Array(repeating: normal, count: 60)
        + Array(repeating: CMTime(value: 6006, timescale: 90000), count: 60))
    record("short_clip", average: average, durations: Array(samples.prefix(10)))
    record("distant_cadence", average: 24, durations: samples)
    record("invalid", average: 25, durations: [.invalid, .indefinite, .zero])

    // Exercise the production writer with a long hold at an internal segment
    // boundary. Closing and reopening the writer must not compact that hold.
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let rate = SourceFrameRate.measured(averageFPS: average, durations: samples)
    let writer = try SegmentWriter(outputDirectory: folder, width: 64, height: 64,
      fpsNumerator: rate.numerator, fpsDenominator: rate.denominator,
      generation: 0, segmentSeconds: 1, realTime: false)
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, 64, 64,
      kCVPixelFormatType_32BGRA,
      [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
    guard status == kCVReturnSuccess, let buffer else { fatalError("pixel buffer creation failed") }
    CVPixelBufferLockBaseAddress(buffer, [])
    memset(CVPixelBufferGetBaseAddress(buffer), 0,
      CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer))
    CVPixelBufferUnlockBaseAddress(buffer, [])
    let ticks: [Int64] = [90, 3093, 6096, 192282, 195285, 198288]
    var segments: [SegmentEvent] = []
    for tick in ticks {
      let pts = Int64((Double(tick) / 90000 * 1_000_000_000).rounded())
      if let segment = try await writer.append(pixelBuffer: buffer, ptsNanoseconds: pts) {
        segments.append(segment)
      }
    }
    if let segment = try await writer.finish() { segments.append(segment) }
    var localTimes: [[Double]] = []
    var lengths: [Double] = []
    for segment in segments {
      let asset = AVURLAsset(url: URL(fileURLWithPath: segment.path))
      let track = try await asset.loadTracks(withMediaType: .video)[0]
      lengths.append(try await asset.load(.duration).seconds)
      let reader = try AVAssetReader(asset: asset)
      let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
      let provider = reader.outputProvider(for: output)
      try reader.start()
      var times: [Double] = []
      while let sample = try await provider.next() {
        let pts = sample.withUnsafeSampleBuffer {
          CMSampleBufferGetNumSamples($0) == 1 ? CMSampleBufferGetPresentationTimeStamp($0) : .invalid
        }
        if pts.isNumeric { times.append(pts.seconds) }
      }
      localTimes.append(times)
    }
    cases["writer"] = ["times": localTimes, "durations": lengths,
                       "starts": segments.map { Double($0.startNs) / 1_000_000_000 },
                       "ends": segments.map { Double($0.endNs) / 1_000_000_000 }]
    print(String(decoding: try JSONSerialization.data(withJSONObject: cases, options: [.sortedKeys]),
                 as: UTF8.self))
  }
}
