import AVFoundation
import CoreVideo
import Foundation

private enum IPadMPEGTSRemuxerHarnessError: LocalizedError {
  case invalidArguments
  case outputMissing
  case outputEmpty
  case videoTrackMissing
  case invalidDuration(Double)
  case outputNotPlayable
  case cannotAddVideoReaderOutput
  case videoReaderDidNotStart
  case firstVideoFrameMissing
  case rangedVideoReaderDidNotStart
  case rangedVideoReaderDidNotComplete(String)
  case rangedVideoFramesMissing
  case invalidIntervalMetadata(String)
  case concatenatedDurationMismatch(expected: Double, actual: Double)
  case concatenatedTimelineNotMonotonic(previous: Double, current: Double)
  case concatenatedBoundaryFramesMissing(before: Int, after: Int)
  case boundaryVideoReaderDidNotStart
  case boundaryVideoReaderDidNotComplete(String)
  case boundaryVideoFramesMissing
  case unexpectedNaturalDimensions(Int, Int)
  case encodedDimensionsMissing
  case unexpectedEncodedDimensions(Int32, Int32)
  case unexpectedDecodedDimensions(Int, Int)

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: IPadMPEGTSRemuxerHarness <input.ts> <output.mp4> | concatenate <first.ts> <second.ts> <output.mp4> <temporary-directory>"
    case .outputMissing:
      return "remuxer did not create the requested output"
    case .outputEmpty:
      return "remuxer created an empty output"
    case .videoTrackMissing:
      return "remuxed output does not contain a video track"
    case .invalidDuration(let duration):
      return "remuxed output has an invalid duration: \(duration)"
    case .outputNotPlayable:
      return "AVFoundation does not consider the remuxed output playable"
    case .cannotAddVideoReaderOutput:
      return "AVFoundation cannot add a decoded video reader output"
    case .videoReaderDidNotStart:
      return "AVFoundation could not start decoding the remuxed video"
    case .firstVideoFrameMissing:
      return "AVFoundation could not decode the first remuxed video frame"
    case .rangedVideoReaderDidNotStart:
      return "AVFoundation could not start decoding from a non-zero time"
    case .rangedVideoReaderDidNotComplete(let detail):
      return "AVFoundation could not finish decoding from a non-zero time: \(detail)"
    case .rangedVideoFramesMissing:
      return "AVFoundation decoded no frames from a non-zero time"
    case .invalidIntervalMetadata(let detail):
      return "HLS interval assembler returned invalid metadata: \(detail)"
    case .concatenatedDurationMismatch(let expected, let actual):
      return "concatenated duration mismatch: expected \(expected), got \(actual)"
    case .concatenatedTimelineNotMonotonic(let previous, let current):
      return "concatenated presentation timestamps are not monotonic: \(previous) then \(current)"
    case .concatenatedBoundaryFramesMissing(let before, let after):
      return "concatenated decode did not cover both sides of the boundary: before=\(before), after=\(after)"
    case .boundaryVideoReaderDidNotStart:
      return "AVFoundation could not start decoding immediately after the HLS boundary"
    case .boundaryVideoReaderDidNotComplete(let detail):
      return "AVFoundation could not finish decoding immediately after the HLS boundary: \(detail)"
    case .boundaryVideoFramesMissing:
      return "AVFoundation decoded no frames immediately after the HLS boundary"
    case .unexpectedNaturalDimensions(let width, let height):
      return "expected the SAR-adjusted display size to round to 853x480, got \(width)x\(height)"
    case .encodedDimensionsMissing:
      return "remuxed output has no encoded video dimensions"
    case .unexpectedEncodedDimensions(let width, let height):
      return "expected encoded dimensions 854x480, got \(width)x\(height)"
    case .unexpectedDecodedDimensions(let width, let height):
      return "expected decoded dimensions 854x480, got \(width)x\(height)"
    }
  }
}

@main
private struct IPadMPEGTSRemuxerHarness {
  static func main() async throws {
    if CommandLine.arguments.count == 6,
      CommandLine.arguments[1] == "concatenate"
    {
      try await verifyConcatenatedInterval(
        firstInputURL: URL(fileURLWithPath: CommandLine.arguments[2]),
        secondInputURL: URL(fileURLWithPath: CommandLine.arguments[3]),
        outputURL: URL(fileURLWithPath: CommandLine.arguments[4]),
        temporaryDirectory: URL(fileURLWithPath: CommandLine.arguments[5])
      )
      return
    }
    guard CommandLine.arguments.count == 3 else {
      throw IPadMPEGTSRemuxerHarnessError.invalidArguments
    }

    let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
    try await IPadMPEGTSRemuxer.remux(
      inputURL: inputURL,
      outputURL: outputURL
    )

    guard FileManager.default.fileExists(atPath: outputURL.path) else {
      throw IPadMPEGTSRemuxerHarnessError.outputMissing
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
    guard let byteCount = attributes[.size] as? NSNumber, byteCount.intValue > 0 else {
      throw IPadMPEGTSRemuxerHarnessError.outputEmpty
    }

    let asset = AVURLAsset(url: outputURL)
    let videoTracks = try await asset.loadTracks(withMediaType: .video)
    guard !videoTracks.isEmpty else {
      throw IPadMPEGTSRemuxerHarnessError.videoTrackMissing
    }
    let track = videoTracks[0]
    let duration = try await asset.load(.duration).seconds
    guard duration.isFinite, duration > 0 else {
      throw IPadMPEGTSRemuxerHarnessError.invalidDuration(duration)
    }
    guard try await asset.load(.isPlayable) else {
      throw IPadMPEGTSRemuxerHarnessError.outputNotPlayable
    }

    let naturalSize = try await track.load(.naturalSize)
    let naturalWidth = Int(abs(naturalSize.width).rounded())
    let naturalHeight = Int(abs(naturalSize.height).rounded())
    guard naturalWidth == 853, naturalHeight == 480 else {
      throw IPadMPEGTSRemuxerHarnessError.unexpectedNaturalDimensions(
        naturalWidth,
        naturalHeight
      )
    }
    let formatDescriptions = try await track.load(.formatDescriptions)
    guard
      let encodedDimensions = formatDescriptions.lazy
        .map(CMVideoFormatDescriptionGetDimensions)
        .first(where: { $0.width > 0 && $0.height > 0 })
    else {
      throw IPadMPEGTSRemuxerHarnessError.encodedDimensionsMissing
    }
    guard encodedDimensions.width == 854, encodedDimensions.height == 480 else {
      throw IPadMPEGTSRemuxerHarnessError.unexpectedEncodedDimensions(
        encodedDimensions.width,
        encodedDimensions.height
      )
    }

    let reader = try AVAssetReader(asset: asset)
    let readerOutput = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
    )
    guard reader.canAdd(readerOutput) else {
      throw IPadMPEGTSRemuxerHarnessError.cannotAddVideoReaderOutput
    }
    reader.add(readerOutput)
    guard reader.startReading() else {
      throw IPadMPEGTSRemuxerHarnessError.videoReaderDidNotStart
    }
    guard
      let sample = readerOutput.copyNextSampleBuffer(),
      let pixelBuffer = CMSampleBufferGetImageBuffer(sample)
    else {
      throw IPadMPEGTSRemuxerHarnessError.firstVideoFrameMissing
    }
    let decodedWidth = CVPixelBufferGetWidth(pixelBuffer)
    let decodedHeight = CVPixelBufferGetHeight(pixelBuffer)
    guard decodedWidth == 854, decodedHeight == 480 else {
      throw IPadMPEGTSRemuxerHarnessError.unexpectedDecodedDimensions(
        decodedWidth,
        decodedHeight
      )
    }
    reader.cancelReading()

    // A passthrough MP4 must preserve the H.264 sync-sample table. Without
    // kCMSampleAttachmentKey_NotSync on dependent samples, AVAssetWriter omits
    // `stss`, and AVAssetReader incorrectly seeks directly to a P/B frame.
    let rangedReader = try AVAssetReader(asset: asset)
    rangedReader.timeRange = CMTimeRange(
      start: CMTime(seconds: 0.5, preferredTimescale: 90_000),
      duration: CMTime(seconds: 0.5, preferredTimescale: 90_000)
    )
    let rangedReaderOutput = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
    )
    guard rangedReader.canAdd(rangedReaderOutput) else {
      throw IPadMPEGTSRemuxerHarnessError.cannotAddVideoReaderOutput
    }
    rangedReader.add(rangedReaderOutput)
    guard rangedReader.startReading() else {
      throw IPadMPEGTSRemuxerHarnessError.rangedVideoReaderDidNotStart
    }
    var rangedDecodedFrameCount = 0
    while let sample = rangedReaderOutput.copyNextSampleBuffer() {
      if CMSampleBufferGetImageBuffer(sample) != nil {
        rangedDecodedFrameCount += 1
      }
    }
    guard rangedReader.status == .completed else {
      throw IPadMPEGTSRemuxerHarnessError.rangedVideoReaderDidNotComplete(
        rangedReader.error?.localizedDescription
          ?? "reader status \(rangedReader.status.rawValue)"
      )
    }
    guard rangedDecodedFrameCount > 0 else {
      throw IPadMPEGTSRemuxerHarnessError.rangedVideoFramesMissing
    }

    print(
      "iPad MPEG-TS remux probe passed "
        + "bytes=\(byteCount.intValue) duration=\(duration) "
        + "natural=\(naturalWidth)x\(naturalHeight) "
        + "encoded=\(encodedDimensions.width)x\(encodedDimensions.height) "
        + "decoded=\(decodedWidth)x\(decodedHeight) "
        + "rangedDecoded=\(rangedDecodedFrameCount)"
    )
  }

  private static func verifyConcatenatedInterval(
    firstInputURL: URL,
    secondInputURL: URL,
    outputURL: URL,
    temporaryDirectory: URL
  ) async throws {
    let result = try await IPadHLSIntervalAssembler.concatenate(
      inputURLs: [firstInputURL, secondInputURL],
      outputURL: outputURL,
      temporaryDirectory: temporaryDirectory
    )
    guard result.sourceOffsets.count == 2, result.sourceDurations.count == 2 else {
      throw IPadMPEGTSRemuxerHarnessError.invalidIntervalMetadata(
        "expected two source entries"
      )
    }
    let boundary = result.sourceOffsets[1]
    let firstDuration = result.sourceDurations[0]
    let secondDuration = result.sourceDurations[1]
    guard
      result.sourceOffsets[0] >= 0,
      firstDuration > 0.5,
      secondDuration > 0.5,
      abs(boundary - (result.sourceOffsets[0] + firstDuration)) < 0.02,
      abs(result.duration - (boundary + secondDuration)) < 0.02
    else {
      throw IPadMPEGTSRemuxerHarnessError.invalidIntervalMetadata(
        "offsets=\(result.sourceOffsets) durations=\(result.sourceDurations) total=\(result.duration)"
      )
    }

    guard FileManager.default.fileExists(atPath: outputURL.path) else {
      throw IPadMPEGTSRemuxerHarnessError.outputMissing
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
    guard let byteCount = attributes[.size] as? NSNumber, byteCount.intValue > 0 else {
      throw IPadMPEGTSRemuxerHarnessError.outputEmpty
    }

    let asset = AVURLAsset(url: outputURL)
    guard try await asset.load(.isPlayable) else {
      throw IPadMPEGTSRemuxerHarnessError.outputNotPlayable
    }
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw IPadMPEGTSRemuxerHarnessError.videoTrackMissing
    }
    let outputDuration = try await asset.load(.duration).seconds
    guard outputDuration.isFinite, outputDuration > 0 else {
      throw IPadMPEGTSRemuxerHarnessError.invalidDuration(outputDuration)
    }
    guard abs(outputDuration - result.duration) < 0.08 else {
      throw IPadMPEGTSRemuxerHarnessError.concatenatedDurationMismatch(
        expected: result.duration,
        actual: outputDuration
      )
    }

    // Decode the complete composition and prove that reset timestamps in the
    // second transport stream became one monotonic movie timeline.
    let reader = try AVAssetReader(asset: asset)
    let readerOutput = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
    )
    guard reader.canAdd(readerOutput) else {
      throw IPadMPEGTSRemuxerHarnessError.cannotAddVideoReaderOutput
    }
    reader.add(readerOutput)
    guard reader.startReading() else {
      throw IPadMPEGTSRemuxerHarnessError.videoReaderDidNotStart
    }
    var previousTimestamp = -Double.infinity
    var framesBeforeBoundary = 0
    var framesAfterBoundary = 0
    while let sample = readerOutput.copyNextSampleBuffer() {
      guard CMSampleBufferGetImageBuffer(sample) != nil else { continue }
      let timestamp = CMSampleBufferGetPresentationTimeStamp(sample).seconds
      guard timestamp.isFinite else { continue }
      if timestamp + 0.000_001 < previousTimestamp {
        throw IPadMPEGTSRemuxerHarnessError.concatenatedTimelineNotMonotonic(
          previous: previousTimestamp,
          current: timestamp
        )
      }
      previousTimestamp = timestamp
      if timestamp < boundary {
        framesBeforeBoundary += 1
      } else {
        framesAfterBoundary += 1
      }
    }
    guard reader.status == .completed else {
      throw IPadMPEGTSRemuxerHarnessError.rangedVideoReaderDidNotComplete(
        reader.error?.localizedDescription ?? "reader status \(reader.status.rawValue)"
      )
    }
    guard framesBeforeBoundary > 0, framesAfterBoundary > 0 else {
      throw IPadMPEGTSRemuxerHarnessError.concatenatedBoundaryFramesMissing(
        before: framesBeforeBoundary,
        after: framesAfterBoundary
      )
    }

    // Realtime restoration starts subsequent work just after the carried
    // segment boundary. A fresh non-zero reader must therefore decode there.
    let boundaryReader = try AVAssetReader(asset: asset)
    boundaryReader.timeRange = CMTimeRange(
      start: CMTime(seconds: boundary + 0.05, preferredTimescale: 90_000),
      duration: CMTime(
        seconds: min(0.5, max(0.1, secondDuration - 0.05)),
        preferredTimescale: 90_000
      )
    )
    let boundaryOutput = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
    )
    guard boundaryReader.canAdd(boundaryOutput) else {
      throw IPadMPEGTSRemuxerHarnessError.cannotAddVideoReaderOutput
    }
    boundaryReader.add(boundaryOutput)
    guard boundaryReader.startReading() else {
      throw IPadMPEGTSRemuxerHarnessError.boundaryVideoReaderDidNotStart
    }
    var boundaryFrameCount = 0
    while let sample = boundaryOutput.copyNextSampleBuffer() {
      if CMSampleBufferGetImageBuffer(sample) != nil {
        boundaryFrameCount += 1
      }
    }
    guard boundaryReader.status == .completed else {
      throw IPadMPEGTSRemuxerHarnessError.boundaryVideoReaderDidNotComplete(
        boundaryReader.error?.localizedDescription
          ?? "reader status \(boundaryReader.status.rawValue)"
      )
    }
    guard boundaryFrameCount > 0 else {
      throw IPadMPEGTSRemuxerHarnessError.boundaryVideoFramesMissing
    }

    print(
      "iPad HLS interval concatenation probe passed "
        + "bytes=\(byteCount.intValue) duration=\(outputDuration) "
        + "sourceDurations=\(result.sourceDurations) boundary=\(boundary) "
        + "frames=\(framesBeforeBoundary)+\(framesAfterBoundary) "
        + "boundaryDecoded=\(boundaryFrameCount)"
    )
  }
}
