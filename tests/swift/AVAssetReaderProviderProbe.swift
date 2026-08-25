import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

@main
struct AVAssetReaderProviderProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 2 else {
      throw NSError(
        domain: "AVAssetReaderProviderProbe",
        code: 2,
        userInfo: [NSLocalizedDescriptionKey: "usage: probe VIDEO"]
      )
    }
    let url = URL(fileURLWithPath: CommandLine.arguments[1])
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw NSError(
        domain: "AVAssetReaderProviderProbe",
        code: 3,
        userInfo: [NSLocalizedDescriptionKey: "video track is missing"]
      )
    }
    let reader = try AVAssetReader(asset: asset)
    reader.timeRange = CMTimeRange(
      start: .zero,
      duration: CMTime(value: 10_000_000_000, timescale: 1_000_000_000)
    )
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
    )
    let provider = reader.outputProvider(for: output)
    try reader.start()
    let decoded = try await Task.detached {
      var frameCount = 0
      var firstPTS: CMTime?
      var lastPTS: CMTime?
      var dimensions: (Int, Int)?
      while let dynamicSample = try await provider.next() {
        guard
          let pixelSample = CMReadySampleBuffer<CVReadOnlyPixelBuffer>(
            dynamicSample
          )
        else {
          continue
        }
        if firstPTS == nil {
          firstPTS = pixelSample.presentationTimeStamp
        }
        lastPTS = pixelSample.presentationTimeStamp
        pixelSample.content.withUnsafeBuffer { pixelBuffer in
          dimensions = (
            CVPixelBufferGetWidth(pixelBuffer),
            CVPixelBufferGetHeight(pixelBuffer)
          )
        }
        frameCount += 1
      }
      return (frameCount, firstPTS, lastPTS, dimensions)
    }.value
    let (frameCount, firstPTS, lastPTS, dimensions) = decoded
    guard reader.status == .completed else {
      throw reader.error ?? NSError(
        domain: "AVAssetReaderProviderProbe",
        code: 4,
        userInfo: [NSLocalizedDescriptionKey: "reader did not complete"]
      )
    }
    let result: [String: Any] = [
      "frames": frameCount,
      "firstPTS": firstPTS?.seconds ?? -1,
      "lastPTS": lastPTS?.seconds ?? -1,
      "width": dimensions?.0 ?? 0,
      "height": dimensions?.1 ?? 0,
    ]
    let data = try JSONSerialization.data(
      withJSONObject: result,
      options: [.sortedKeys]
    )
    print(String(decoding: data, as: UTF8.self))
  }
}
