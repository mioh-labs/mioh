import AVFoundation
import CoreVideo
import Foundation
import VideoToolbox

/// A single, video-only MP4 writer used by an iPad cluster shard.  The file
/// passed to this type is private staging; publication into the ledger-owned
/// local candidate is performed only after `finish()` completes successfully.
final class MiohIPadVideoWriter {
  enum WriterError: LocalizedError {
    case invalidDimensions
    case invalidFrameRate
    case cannotStart(String)
    case appendFailed(String)
    case finishFailed(String)
    case empty

    var errorDescription: String? {
      switch self {
      case .invalidDimensions:
        "動画の幅と高さは正の偶数である必要があります。"
      case .invalidFrameRate:
        "動画のフレームレートが不正です。"
      case .cannotStart(let detail):
        "MP4 writerを開始できません: \(detail)"
      case .appendFailed(let detail):
        "復元フレームを書き込めません: \(detail)"
      case .finishFailed(let detail):
        "MP4 writerを確定できません: \(detail)"
      case .empty:
        "出力対象の動画フレームがありません。"
      }
    }
  }

  private let url: URL
  private let writer: AVAssetWriter
  // Keep the iOS 26 receiver type behind availability so the controller-only
  // portion of Mioh Remote can still deploy to iOS 16. Worker devices use the
  // receiver path; older devices retain the pre-iOS 26 compatibility path.
  private let receiverStorage: Any?
  private let legacyInput: AVAssetWriterInput?
  private let legacyAdaptor: AVAssetWriterInputPixelBufferAdaptor?
  private var frameCount = 0
  private var lastPresentationTimeNanoseconds: Int64?
  private var finished = false

  init(
    url: URL,
    width: Int,
    height: Int,
    fpsNumerator: Int,
    fpsDenominator: Int,
    codec: AVVideoCodecType,
    sourceBitRate: Double,
    bitrateMultiplier: Double,
    fastStart: Bool
  ) throws {
    guard width > 0, height > 0, width.isMultiple(of: 2),
      height.isMultiple(of: 2)
    else { throw WriterError.invalidDimensions }
    guard fpsNumerator > 0, fpsDenominator > 0,
      fpsNumerator <= Int(Int32.max)
    else { throw WriterError.invalidFrameRate }

    self.url = url
    try? FileManager.default.removeItem(at: url)

    let writer: AVAssetWriter
    do {
      writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    } catch {
      let value = error as NSError
      throw WriterError.cannotStart(
        "writer作成: \(error.localizedDescription) [\(value.domain):\(value.code)]"
      )
    }
    writer.shouldOptimizeForNetworkUse = fastStart
    writer.movieTimeScale = 60_000
    let frameRate = Double(fpsNumerator) / Double(fpsDenominator)
    let pixelRate = Double(width * height) * frameRate
    let fallbackRate = min(35_000_000, max(4_000_000, pixelRate * 0.10))
    let requestedRate =
      sourceBitRate > 0
      ? sourceBitRate * bitrateMultiplier
      : fallbackRate
    let averageBitRate = Int(
      min(120_000_000, max(2_000_000, requestedRate))
    )
    let profile: String =
      codec == .hevc
      ? (kVTProfileLevel_HEVC_Main_AutoLevel as String)
      : AVVideoProfileLevelH264HighAutoLevel
    let settings: [String: Any] = [
      AVVideoCodecKey: codec,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoColorPropertiesKey: [
        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
      ],
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: averageBitRate,
        AVVideoExpectedSourceFrameRateKey: frameRate,
        AVVideoAllowFrameReorderingKey: false,
        AVVideoMaxKeyFrameIntervalKey: max(1, Int(frameRate * 2)),
        AVVideoProfileLevelKey: profile,
      ],
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.mediaTimeScale = 60_000
    if #available(iOS 26.0, *) {
      var attributes = CVPixelBufferCreationAttributes(
        pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA),
        size: CVImageSize(width: width, height: height)
      )
      attributes.backing = .ioSurface
      receiverStorage = writer.inputPixelBufferReceiver(
        for: input,
        pixelBufferAttributes: attributes
      )
      legacyInput = nil
      legacyAdaptor = nil
      do {
        try writer.start()
      } catch {
        throw WriterError.cannotStart(error.localizedDescription)
      }
    } else {
      let attributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
      let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: attributes
      )
      guard writer.canAdd(input) else {
        throw WriterError.cannotStart("video inputを追加できません")
      }
      writer.add(input)
      guard writer.startWriting() else {
        throw WriterError.cannotStart(
          writer.error?.localizedDescription ?? "startWriting failed"
        )
      }
      receiverStorage = nil
      legacyInput = input
      legacyAdaptor = adaptor
    }
    writer.startSession(atSourceTime: .zero)
    self.writer = writer
  }

  deinit {
    if !finished {
      writer.cancelWriting()
      try? FileManager.default.removeItem(at: url)
    }
  }

  func append(
    _ pixelBuffer: CVPixelBuffer,
    presentationTimeNanoseconds: Int64
  ) async throws {
    try Task.checkCancellation()
    guard presentationTimeNanoseconds >= 0,
      lastPresentationTimeNanoseconds.map({ presentationTimeNanoseconds > $0 }) ?? true
    else {
      throw WriterError.appendFailed("source presentation timestamps are not increasing")
    }
    let presentationTime = CMTime(
      value: presentationTimeNanoseconds,
      timescale: 1_000_000_000
    )
    if #available(iOS 26.0, *),
      let receiver = receiverStorage as? AVAssetWriterInput.PixelBufferReceiver
    {
      do {
        try await receiver.append(
          CVReadOnlyPixelBuffer(unsafeBuffer: pixelBuffer),
          with: presentationTime
        )
      } catch {
        throw WriterError.appendFailed(error.localizedDescription)
      }
    } else {
      guard let input = legacyInput, let adaptor = legacyAdaptor else {
        throw WriterError.appendFailed("writer backend is unavailable")
      }
      while !input.isReadyForMoreMediaData {
        try Task.checkCancellation()
        if writer.status == .failed || writer.status == .cancelled {
          throw WriterError.appendFailed(
            writer.error?.localizedDescription ?? "writer stopped"
          )
        }
        try await Task.sleep(nanoseconds: 250_000)
      }
      guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime)
      else {
        throw WriterError.appendFailed(
          writer.error?.localizedDescription ?? "append returned false"
        )
      }
    }
    lastPresentationTimeNanoseconds = presentationTimeNanoseconds
    frameCount += 1
  }

  func finish(durationNanoseconds: Int64) async throws -> Int {
    guard frameCount > 0 else { throw WriterError.empty }
    guard durationNanoseconds > 0,
      lastPresentationTimeNanoseconds.map({ durationNanoseconds > $0 }) ?? false
    else {
      throw WriterError.finishFailed("output duration does not contain the final frame")
    }
    try Task.checkCancellation()
    writer.endSession(
      atSourceTime: CMTime(
        value: durationNanoseconds,
        timescale: 1_000_000_000
      )
    )
    if #available(iOS 26.0, *),
      let receiver = receiverStorage as? AVAssetWriterInput.PixelBufferReceiver
    {
      receiver.finish()
      await writer.finishWriting()
    } else {
      legacyInput?.markAsFinished()
      await withCheckedContinuation { continuation in
        writer.finishWriting { continuation.resume() }
      }
    }
    guard writer.status == .completed else {
      throw WriterError.finishFailed(
        writer.error?.localizedDescription ?? "finishWriting failed"
      )
    }
    finished = true
    return frameCount
  }

  func cancel() {
    guard !finished else { return }
    writer.cancelWriting()
    try? FileManager.default.removeItem(at: url)
  }
}
