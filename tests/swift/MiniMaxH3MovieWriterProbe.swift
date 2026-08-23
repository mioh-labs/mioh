import AVFoundation
import CoreVideo
import Foundation

@main
enum MiniMaxH3MovieWriterProbe {
  static func main() async throws {
    let width = 864
    let height = 480
    let output = URL(fileURLWithPath: "/tmp/mioh-h3-writer-probe.mp4")
    try? FileManager.default.removeItem(at: output)

    let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.hevc,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: 6_000_000,
          AVVideoExpectedSourceFrameRateKey: 24,
          AVVideoMaxKeyFrameIntervalKey: 48,
        ],
      ]
    )
    var attributes = CVPixelBufferCreationAttributes(
      pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA),
      size: CVImageSize(width: width, height: height)
    )
    attributes.backing = .ioSurface
    let receiver = writer.inputPixelBufferReceiver(
      for: input,
      pixelBufferAttributes: attributes
    )
    try writer.start()
    writer.startSession(atSourceTime: .zero)
    guard let pool = receiver.pixelBufferPool else { fatalError("missing pool") }
    var buffer = try pool.makeMutablePixelBuffer()
    buffer.accessUnsafeMutableRawPlaneBytes { planes in
      for plane in planes {
        if let base = plane.bytes.baseAddress {
          memset(base, 0, plane.bytes.count)
        }
      }
    }
    try await receiver.append(CVReadOnlyPixelBuffer(buffer), with: .zero)
    receiver.finish()
    await writer.finishWriting()
    print(
      "finished=\(writer.status.rawValue) "
        + "bytes=\((try? output.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? -1) "
        + "error=\(writer.error?.localizedDescription ?? "nil")"
    )
  }
}
