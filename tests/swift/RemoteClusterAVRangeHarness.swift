import AVFoundation
import CoreVideo
import CryptoKit
import Foundation

private enum AVRangeHarnessError: Error {
  case writer(String)
  case invalidAsset(String)
  case noTailFrame(String)
}

private func writeTestMovie(_ url: URL, fastStart: Bool) async throws {
  let width = 96
  let height = 64
  let frameCount = 90
  let fps: Int32 = 30
  let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
  writer.shouldOptimizeForNetworkUse = fastStart
  let input = AVAssetWriterInput(
    mediaType: .video,
    outputSettings: [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: 250_000,
        AVVideoExpectedSourceFrameRateKey: 30,
        AVVideoMaxKeyFrameIntervalKey: 30,
      ],
    ]
  )
  let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: input,
    sourcePixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
  )
  guard writer.canAdd(input) else {
    throw AVRangeHarnessError.writer("cannot add input")
  }
  writer.add(input)
  guard writer.startWriting() else {
    throw AVRangeHarnessError.writer(
      writer.error?.localizedDescription ?? "start failed"
    )
  }
  writer.startSession(atSourceTime: .zero)
  guard let pool = adaptor.pixelBufferPool else {
    throw AVRangeHarnessError.writer("missing pixel buffer pool")
  }
  for index in 0..<frameCount {
    while !input.isReadyForMoreMediaData {
      try await Task.sleep(nanoseconds: 250_000)
    }
    var optionalBuffer: CVPixelBuffer?
    guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optionalBuffer) == kCVReturnSuccess,
      let buffer = optionalBuffer
    else { throw AVRangeHarnessError.writer("pixel buffer allocation failed") }
    CVPixelBufferLockBaseAddress(buffer, [])
    if let base = CVPixelBufferGetBaseAddress(buffer) {
      let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
      for y in 0..<height {
        let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
        for x in 0..<width {
          let offset = x * 4
          row[offset] = UInt8((index * 5 + x) & 255)
          row[offset + 1] = UInt8((index * 3 + y) & 255)
          row[offset + 2] = UInt8((index * 7) & 255)
          row[offset + 3] = 255
        }
      }
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    guard adaptor.append(
      buffer,
      withPresentationTime: CMTime(value: Int64(index), timescale: fps)
    ) else {
      throw AVRangeHarnessError.writer(
        writer.error?.localizedDescription ?? "append failed"
      )
    }
  }
  input.markAsFinished()
  await withCheckedContinuation { continuation in
    writer.finishWriting { continuation.resume() }
  }
  guard writer.status == .completed else {
    throw AVRangeHarnessError.writer(
      writer.error?.localizedDescription ?? "finish failed"
    )
  }
}

private func makeRequest(
  bytes: Int64,
  sha256: String,
  attemptID: UUID
) throws -> RemoteClusterJobRequest {
  let now = Date()
  return RemoteClusterJobRequest(
    jobID: UUID(),
    attemptID: attemptID,
    leaseID: UUID(),
    coordinatorNodeID: UUID(),
    sharedRootIdentifier: "",
    inputByteCount: bytes,
    inputSHA256: sha256,
    inputRelativePath: try RemoteClusterRelativePath(validating: "input/source.mp4"),
    outputRelativePath: try RemoteClusterRelativePath(validating: "output/shard.mp4"),
    mediaRange: RemoteClusterMediaRange(
      decodeStartNanoseconds: 0,
      decodeEndNanoseconds: 3_000_000_000,
      coreStartNanoseconds: 0,
      coreEndNanoseconds: 2_900_000_000,
      leadingOverlapFrames: 0,
      trailingOverlapFrames: 3
    ),
    options: RemoteClusterRestorationOptions(
      restorationModelIdentifier: "restorer",
      restorationAssetSHA256: String(repeating: "a", count: 64),
      detectorModelIdentifier: "detector",
      detectorAssetSHA256: String(repeating: "b", count: 64),
      restorationClipLength: 18,
      temporalOverlap: 2,
      crossfade: true,
      detectionEmptyLookahead: 1,
      detectFaceMosaics: false,
      blendFeather: 1,
      sharpenStrength: 0,
      detailBoost: 0,
      textureMix: 0,
      smoothStrength: 0,
      effectUpscale: 1,
      roiEnhancerModelIdentifier: nil,
      roiEnhancerAssetSHA256: nil,
      roiEnhancerStrength: 0,
      roiEnhancerScale: 1,
      videoCodec: "h264",
      bitrateMultiplier: 1,
      mp4FastStart: false,
      targetFPSNumerator: nil,
      targetFPSDenominator: nil
    ),
    createdAt: now,
    leaseExpiresAt: now.addingTimeInterval(10 * 60)
  )
}

private func decodeTail(
  descriptor: RemoteClusterHTTPTransferDescriptor,
  byteCount: Int64,
  sha256: String,
  label: String
) async throws {
  guard let endpoint = URL(string: descriptor.inputURL) else {
    throw AVRangeHarnessError.invalidAsset(label)
  }
  let source = try MiohHTTPRangeAsset(
    remoteURL: endpoint,
    expectedByteCount: byteCount,
    expectedSHA256: sha256
  )
  defer { source.cancel() }
  let asset = source.asset
  let playable = try await asset.load(.isPlayable)
  let tracks = try await asset.loadTracks(withMediaType: .video)
  guard playable, let track = tracks.first else {
    throw AVRangeHarnessError.invalidAsset(label)
  }
  let reader = try AVAssetReader(asset: asset)
  reader.timeRange = CMTimeRange(
    start: CMTime(seconds: 2.4, preferredTimescale: 600),
    duration: CMTime(seconds: 0.5, preferredTimescale: 600)
  )
  let output = AVAssetReaderTrackOutput(
    track: track,
    outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
    ]
  )
  guard reader.canAdd(output) else {
    throw AVRangeHarnessError.invalidAsset(label + " reader")
  }
  reader.add(output)
  guard reader.startReading(), let sample = output.copyNextSampleBuffer(),
    CMSampleBufferGetImageBuffer(sample) != nil
  else {
    throw AVRangeHarnessError.noTailFrame(
      label + ": " + (reader.error?.localizedDescription ?? "none")
    )
  }
  let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
  guard pts >= 2.3 else {
    throw AVRangeHarnessError.noTailFrame(label + ": pts=\(pts)")
  }
}

@main
struct RemoteClusterAVRangeHarness {
  static func main() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-cluster-av-range-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let server = RemoteClusterHTTPTransferServer(advertisedHost: "127.0.0.1")
    try await server.start()
    defer { server.stop() }
    for fastStart in [true, false] {
      let label = fastStart ? "fast-start" : "moov-tail"
      let sourceURL = root.appendingPathComponent("\(label).mp4")
      try await writeTestMovie(sourceURL, fastStart: fastStart)
      let data = try Data(contentsOf: sourceURL)
      let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
      let attemptID = UUID()
      _ = try await server.pinSource(sourceURL)
      let descriptor = try server.register(
        request: makeRequest(
          bytes: Int64(data.count),
          sha256: sha,
          attemptID: attemptID
        ),
        inputURL: sourceURL,
        outputURL: root.appendingPathComponent("\(label)-output.mp4"),
        maximumOutputBytes: 32 * 1_024 * 1_024
      )
      try await decodeTail(
        descriptor: descriptor,
        byteCount: Int64(data.count),
        sha256: sha,
        label: label
      )
      server.unregister(attemptID: attemptID)
    }
    print("remote-cluster AVFoundation range asset: ok")
  }
}
