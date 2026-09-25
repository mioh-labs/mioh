import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

struct MacHLSCaptureRatePolicy {
  static let acceleratedRate: Float = 2

  private var isWarmingUp = true
  private var isFillingTarget = true

  mutating func finishWarmup() -> Bool {
    guard isWarmingUp else { return false }
    isWarmingUp = false
    return true
  }

  mutating func desiredRate(
    bufferedSeconds: Double,
    targetSeconds: Double,
    isLive: Bool
  ) -> Float {
    guard !isLive, !isWarmingUp else { return 1 }
    let target = max(2, targetSeconds.isFinite ? targetSeconds : 8)
    let buffered = max(0, bufferedSeconds.isFinite ? bufferedSeconds : 0)
    if buffered + 0.1 >= target {
      isFillingTarget = false
    } else if buffered <= max(2, target * 0.70) {
      isFillingTarget = true
    }
    return isFillingTarget ? Self.acceleratedRate : 1
  }

}

/// Captures HLS through AVFoundation's media stack instead of issuing raw
/// playlist/segment requests. Some CDNs accept Safari/AVPlayer HLS playback but
/// reject URLSession or WKDownload requests for the same segment with HTTP 429.
/// One player decodes both video frames and audio PCM, so no second player
/// ever fetches the same stream.
@MainActor
final class MacHLSAVFoundationCapture {
  struct CapturedSegment: Sendable {
    let sequence: Int
    let startSeconds: Double
    let endSeconds: Double
    let url: URL
  }

  enum CaptureError: LocalizedError {
    case alreadyStarted
    case source(String)
    case encoder(String)

    var errorDescription: String? {
      switch self {
      case .alreadyStarted:
        "AVFoundation HLS取込はすでに開始しています"
      case .source(let detail):
        "AVFoundationでHLS映像を取得できません: \(detail)"
      case .encoder(let detail):
        "AVFoundation HLS映像を一時保存できません: \(detail)"
      }
    }
  }

  let asset: AVURLAsset
  let audio = MacHLSPCMTimeline()

  private let outputDirectory: URL
  private let requestedStartSeconds: Double
  private let knownDuration: Double
  private let isLive: Bool
  private let generation: Int
  private let segmentSeconds: Double
  private let log: @MainActor (String) -> Void
  private var forwardBufferSeconds: Double
  private let player = AVPlayer()
  private var captureTask: Task<Void, Never>?
  private var captureItem: AVPlayerItem?
  private var videoOutput: AVPlayerItemVideoOutput?
  private var audioOutput: AnyObject?
  private var endObserver: NSObjectProtocol?
  private var didReachEnd = false
  private var restoredBufferLeadSeconds = 0.0
  private var ratePolicy = MacHLSCaptureRatePolicy()
  private var requestedCaptureRate: Float = 1
  private var capturePlaybackStarted = false

  init(
    url: URL,
    outputDirectory: URL,
    startSeconds: Double,
    duration: Double,
    isLive: Bool,
    generation: Int,
    segmentSeconds: Double,
    forwardBufferSeconds: Double,
    log: @escaping @MainActor (String) -> Void = { _ in }
  ) {
    asset = AVURLAsset(url: url)
    self.outputDirectory = outputDirectory
    requestedStartSeconds = max(0, startSeconds.isFinite ? startSeconds : 0)
    knownDuration = max(0, duration.isFinite ? duration : 0)
    self.isLive = isLive
    self.generation = generation
    self.segmentSeconds = max(0.5, segmentSeconds)
    self.forwardBufferSeconds = max(2, forwardBufferSeconds)
    self.log = log
    player.isMuted = true
    player.automaticallyWaitsToMinimizeStalling = true
    player.preventsDisplaySleepDuringVideoPlayback = false
    player.actionAtItemEnd = .pause
  }

  func setForwardBufferDuration(_ seconds: Double) {
    forwardBufferSeconds = max(2, seconds.isFinite ? seconds : 8)
    captureItem?.preferredForwardBufferDuration = forwardBufferSeconds
    updateCaptureRate()
  }

  func setRestoredBufferLead(_ seconds: Double) {
    restoredBufferLeadSeconds = max(0, seconds.isFinite ? seconds : 0)
    updateCaptureRate()
  }

  func segments() throws -> AsyncThrowingStream<CapturedSegment, Error> {
    guard captureTask == nil else { throw CaptureError.alreadyStarted }
    return AsyncThrowingStream { continuation in
      let task = Task { @MainActor [weak self] in
        guard let self else {
          continuation.finish(throwing: CancellationError())
          return
        }
        do {
          try await self.capture(into: continuation)
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      captureTask = task
      continuation.onTermination = { @Sendable [weak self] _ in
        Task { @MainActor in self?.cancel() }
      }
    }
  }

  func cancel() {
    captureTask?.cancel()
    captureTask = nil
    player.pause()
    player.replaceCurrentItem(with: nil)
    if let endObserver {
      NotificationCenter.default.removeObserver(endObserver)
      self.endObserver = nil
    }
    if let captureItem, let videoOutput {
      captureItem.remove(videoOutput)
    }
    if #available(macOS 27.0, *),
      let captureItem,
      let audioOutput = audioOutput as? AVPlayerItemSampleBufferOutput
    {
      captureItem.remove(audioOutput)
    }
    captureItem = nil
    videoOutput = nil
    audioOutput = nil
    capturePlaybackStarted = false
    requestedCaptureRate = 1
  }

  private func capture(
    into continuation: AsyncThrowingStream<CapturedSegment, Error>.Continuation
  ) async throws {
    try FileManager.default.createDirectory(
      at: outputDirectory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )

    let item = AVPlayerItem(asset: asset)
    item.preferredMaximumResolution = CGSize(width: 1_920, height: 1_080)
    item.preferredForwardBufferDuration = max(
      forwardBufferSeconds,
      segmentSeconds * 4
    )
    let output = AVPlayerItemVideoOutput(
      pixelBufferAttributes: CVPixelBufferAttributes(
        pixelFormatTypes: [
          CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA)
        ]
      )
    )
    output.suppressesPlayerRendering = true
    item.add(output)
    if #available(macOS 27.0, *) {
      let configuration = AVPlayerItemSampleBufferOutputAudioConfiguration()
      configuration.requestedAudioFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: Double(MacHLSAudio.sampleRate),
        channels: AVAudioChannelCount(MacHLSAudio.channels),
        interleaved: true
      )?.formatDescription
      let audioOutput = AVPlayerItemSampleBufferOutput(configuration: configuration)
      audioOutput.suppressesPlayerRendering = true
      item.add(audioOutput)
      self.audioOutput = audioOutput
    }
    captureItem = item
    videoOutput = output
    didReachEnd = false
    endObserver = NotificationCenter.default.addObserver(
      forName: AVPlayerItem.didPlayToEndTimeNotification,
      object: item,
      queue: .main
    ) { [weak self, weak item] _ in
      Task { @MainActor in
        guard let self, self.captureItem === item else { return }
        self.didReachEnd = true
      }
    }
    player.replaceCurrentItem(with: item)

    try await waitUntilReady(item)
    let actualStart = try await seekCapturePlayer(item)
    let timelineOffset = actualStart - requestedStartSeconds
    output.requestNotificationOfMediaDataChange(withAdvanceInterval: 1.0 / 120.0)
    player.play()
    capturePlaybackStarted = true
    requestedCaptureRate = 1
    updateCaptureRate()

    var pendingFrames: [(CVPixelBuffer, Int64)] = []
    var writer: MacHLSCaptureSegmentWriter?
    var writerDimensions: (width: Int, height: Int)?
    var nextSequence = 0
    var lastPTS: Int64?
    var lastFrameAt = Date()

    func yield(_ encoded: MacHLSCaptureSegmentWriter.Output?) {
      guard let encoded else { return }
      continuation.yield(
        CapturedSegment(
          sequence: nextSequence,
          startSeconds: Double(encoded.startNanoseconds) / 1_000_000_000,
          endSeconds: Double(encoded.endNanoseconds) / 1_000_000_000,
          url: encoded.url
        )
      )
      nextSequence += 1
      if ratePolicy.finishWarmup() {
        updateCaptureRate()
      }
    }

    do {
      captureLoop: while true {
        try Task.checkCancellation()
        if item.status == .failed {
          throw CaptureError.source(
            item.error?.localizedDescription ?? "AVPlayerItemが失敗しました"
          )
        }
        if didReachEnd { break captureLoop }

        let itemTime = output.itemTime(forHostTime: CACurrentMediaTime())
        if output.hasNewPixelBuffer(forItemTime: itemTime) {
          let frame = output.pixelBufferAndDisplayTime(forItemTime: itemTime)
          if let readOnlyPixelBuffer = frame.pixelBuffer {
            let pixelBuffer: CVPixelBuffer = readOnlyPixelBuffer.withUnsafeBuffer {
              $0
            }
            let displayTime = frame.itemTimeForDisplay
            let rawSeconds = CMTimeGetSeconds(
              displayTime.isValid ? displayTime : itemTime
            )
            let timelineSeconds = rawSeconds - timelineOffset
            if timelineSeconds.isFinite,
              timelineSeconds + 0.001 >= requestedStartSeconds
            {
              let pts = Int64((timelineSeconds * 1_000_000_000).rounded())
              if lastPTS == nil || pts > lastPTS! {
                lastPTS = pts
                lastFrameAt = Date()
                let width = CVPixelBufferGetWidth(pixelBuffer)
                let height = CVPixelBufferGetHeight(pixelBuffer)

                if let dimensions = writerDimensions,
                  dimensions.width != width || dimensions.height != height
                {
                  yield(try await writer?.finish())
                  writer = nil
                  writerDimensions = nil
                  pendingFrames.removeAll(keepingCapacity: true)
                }

                if writer == nil {
                  pendingFrames.append((pixelBuffer, pts))
                  if pendingFrames.count >= 8 {
                    let rate = Self.estimatedFrameRate(
                      from: pendingFrames.map(\.1)
                    )
                    let created = try MacHLSCaptureSegmentWriter(
                      outputDirectory: outputDirectory,
                      width: width,
                      height: height,
                      fpsNumerator: rate.numerator,
                      fpsDenominator: rate.denominator,
                      generation: generation,
                      segmentSeconds: segmentSeconds
                    )
                    writer = created
                    writerDimensions = (width, height)
                    for (buffer, bufferedPTS) in pendingFrames {
                      yield(
                        try await created.append(
                          pixelBuffer: buffer,
                          ptsNanoseconds: bufferedPTS
                        )
                      )
                    }
                    pendingFrames.removeAll(keepingCapacity: true)
                  }
                } else if let writer {
                  yield(
                    try await writer.append(
                      pixelBuffer: pixelBuffer,
                      ptsNanoseconds: pts
                    )
                  )
                }
              }
            }
          }
        }

        if let lastPTS {
          // Keep audio a little ahead of the captured video, not the whole
          // stream the output would otherwise decode in advance.
          drainAudio(
            through: Double(lastPTS) / 1_000_000_000 + 4,
            timelineOffset: timelineOffset
          )
        }
        if !isLive, knownDuration > 0 {
          let sourceSeconds = player.currentTime().seconds - timelineOffset
          if sourceSeconds.isFinite,
            sourceSeconds >= knownDuration - 0.02
          {
            break captureLoop
          }
        }
        if Date().timeIntervalSince(lastFrameAt) > 45,
          player.timeControlStatus != .waitingToPlayAtSpecifiedRate
        {
          throw CaptureError.source("45秒間映像フレームを取得できませんでした")
        }
        try await Task.sleep(nanoseconds: 5_000_000)
      }

      if writer == nil, !pendingFrames.isEmpty,
        let first = pendingFrames.first
      {
        let width = CVPixelBufferGetWidth(first.0)
        let height = CVPixelBufferGetHeight(first.0)
        let rate = Self.estimatedFrameRate(from: pendingFrames.map(\.1))
        let created = try MacHLSCaptureSegmentWriter(
          outputDirectory: outputDirectory,
          width: width,
          height: height,
          fpsNumerator: rate.numerator,
          fpsDenominator: rate.denominator,
          generation: generation,
          segmentSeconds: segmentSeconds
        )
        writer = created
        for (buffer, bufferedPTS) in pendingFrames {
          yield(
            try await created.append(
              pixelBuffer: buffer,
              ptsNanoseconds: bufferedPTS
            )
          )
        }
      }
      drainAudio(through: .infinity, timelineOffset: timelineOffset)
      yield(try await writer?.finish())
    } catch {
      writer?.discard()
      throw error
    }
    player.pause()
  }

  /// Moves decoded PCM onto the HLS timeline. Buffers decoded before the
  /// start seek belong to another position and are ignored.
  private func drainAudio(through limit: Double, timelineOffset: Double) {
    guard #available(macOS 27.0, *),
      let output = audioOutput as? AVPlayerItemSampleBufferOutput
    else { return }
    while audio.endSeconds < limit, let next = output.nextAvailableSampleBuffer() {
      next.sampleBuffer.withUnsafeSampleBuffer { sample in
        let start = CMSampleBufferGetOutputPresentationTimeStamp(sample).seconds
          - timelineOffset
        guard start.isFinite, start + 0.5 >= requestedStartSeconds else { return }
        var blockBuffer: CMBlockBuffer?
        var list = AudioBufferList()
        guard CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
          sample,
          bufferListSizeNeededOut: nil,
          bufferListOut: &list,
          bufferListSize: MemoryLayout<AudioBufferList>.size,
          blockBufferAllocator: nil,
          blockBufferMemoryAllocator: nil,
          flags: 0,
          blockBufferOut: &blockBuffer
        ) == noErr, let data = list.mBuffers.mData else { return }
        audio.append(
          startSeconds: start,
          interleaved: UnsafeBufferPointer(
            start: data.assumingMemoryBound(to: Int16.self),
            count: Int(list.mBuffers.mDataByteSize) / MemoryLayout<Int16>.size
          )
        )
      }
    }
  }

  private func updateCaptureRate() {
    let desired = ratePolicy.desiredRate(
      bufferedSeconds: restoredBufferLeadSeconds,
      targetSeconds: forwardBufferSeconds,
      isLive: isLive
    )
    guard abs(desired - requestedCaptureRate) >= 0.05 else { return }
    let previous = requestedCaptureRate
    requestedCaptureRate = desired
    guard capturePlaybackStarted, captureItem != nil else { return }
    player.rate = desired
    if desired > 1.05 {
      log(
        "HLS取込: 復元速度に余裕があるため"
          + "\(String(format: "%.2f", desired))倍で先読みを増やします\n"
      )
    } else if previous > 1.05 {
      log(
        "HLS取込: 復元バッファ"
          + "\(String(format: "%.1f", restoredBufferLeadSeconds))秒で"
          + "1倍速へ戻します\n"
      )
    }
  }

  private func waitUntilReady(_ item: AVPlayerItem) async throws {
    let deadline = Date().addingTimeInterval(45)
    while item.status == .unknown, Date() < deadline {
      try Task.checkCancellation()
      try await Task.sleep(nanoseconds: 50_000_000)
    }
    guard item.status == .readyToPlay else {
      throw CaptureError.source(
        item.error?.localizedDescription ?? "HLSの読込がタイムアウトしました"
      )
    }
  }

  private func seekCapturePlayer(_ item: AVPlayerItem) async throws -> Double {
    let target: Double
    if isLive,
      let seekable = item.seekableTimeRanges.last?.timeRangeValue
    {
      let lower = CMTimeGetSeconds(seekable.start)
      let upper = CMTimeGetSeconds(CMTimeRangeGetEnd(seekable))
      let distanceFromEdge = max(0, knownDuration - requestedStartSeconds)
      target = min(upper, max(lower, upper - distanceFromEdge))
    } else {
      target = requestedStartSeconds
    }
    let completed = await withCheckedContinuation {
      (continuation: CheckedContinuation<Bool, Never>) in
      player.seek(
        to: CMTime(seconds: target, preferredTimescale: 600),
        toleranceBefore: .zero,
        toleranceAfter: .zero
      ) { finished in continuation.resume(returning: finished) }
    }
    try Task.checkCancellation()
    guard completed else {
      throw CaptureError.source("開始位置へ移動できませんでした")
    }
    let actual = player.currentTime().seconds
    guard actual.isFinite else {
      throw CaptureError.source("開始位置の時間情報を取得できませんでした")
    }
    return actual
  }

  private static func estimatedFrameRate(
    from presentationTimes: [Int64]
  ) -> (numerator: Int, denominator: Int) {
    let deltas = zip(presentationTimes, presentationTimes.dropFirst())
      .map { $1 - $0 }
      .filter { $0 >= 8_000_000 && $0 <= 100_000_000 }
      .sorted()
    guard let median = deltas.isEmpty ? nil : deltas[deltas.count / 2] else {
      return (30, 1)
    }
    let observed = 1_000_000_000 / Double(median)
    let candidates = [
      (24_000, 1_001), (24, 1), (25, 1),
      (30_000, 1_001), (30, 1), (50, 1),
      (60_000, 1_001), (60, 1),
    ]
    return candidates.min {
      abs(Double($0.0) / Double($0.1) - observed)
        < abs(Double($1.0) / Double($1.1) - observed)
    }.map { ($0.0, $0.1) } ?? (30, 1)
  }
}

@MainActor
private final class MacHLSCaptureSegmentWriter {
  struct Output {
    let startNanoseconds: Int64
    let endNanoseconds: Int64
    let url: URL
  }

  private let outputDirectory: URL
  private let width: Int
  private let height: Int
  private let fpsNumerator: Int
  private let fpsDenominator: Int
  private let generation: Int
  private let segmentNanoseconds: Int64
  private let frameDurationNanoseconds: Int64
  private let identity = UUID().uuidString.lowercased()
  private var fileSequence = 0
  private var segmentStartNanoseconds: Int64?
  private var lastPTS: Int64?
  private var writer: AVAssetWriter?
  private var pixelBufferReceiver: AVAssetWriterInput.PixelBufferReceiver?
  private var workingURL: URL?
  private var finalURL: URL?

  init(
    outputDirectory: URL,
    width: Int,
    height: Int,
    fpsNumerator: Int,
    fpsDenominator: Int,
    generation: Int,
    segmentSeconds: Double
  ) throws {
    guard width > 0, height > 0, fpsNumerator > 0, fpsDenominator > 0 else {
      throw MacHLSAVFoundationCapture.CaptureError.encoder(
        "映像サイズまたはフレームレートが不正です"
      )
    }
    self.outputDirectory = outputDirectory
    self.width = width
    self.height = height
    self.fpsNumerator = fpsNumerator
    self.fpsDenominator = fpsDenominator
    self.generation = generation
    segmentNanoseconds = Int64(segmentSeconds * 1_000_000_000)
    frameDurationNanoseconds = Int64(
      Double(1_000_000_000 * fpsDenominator) / Double(fpsNumerator)
    )
  }

  func append(
    pixelBuffer: CVPixelBuffer,
    ptsNanoseconds: Int64
  ) async throws -> Output? {
    var completed: Output?
    if let start = segmentStartNanoseconds,
      ptsNanoseconds >= start + segmentNanoseconds
    {
      completed = try await close(endNanoseconds: ptsNanoseconds)
    }
    if writer == nil { try open(startNanoseconds: ptsNanoseconds) }
    guard writer != nil, let receiver = pixelBufferReceiver else {
      throw MacHLSAVFoundationCapture.CaptureError.encoder(
        "一時映像のwriterを開始できません"
      )
    }
    try Task.checkCancellation()
    guard let segmentStart = segmentStartNanoseconds else {
      throw MacHLSAVFoundationCapture.CaptureError.encoder(
        "一時映像の開始時刻がありません"
      )
    }
    // Keep each frame at its captured time. Numbering frames at a nominal rate
    // shortened a segment by every frame the capture loop missed, while its
    // audio and timeline stayed two seconds long: audio was cut at every
    // segment boundary and motion ran fast.
    let presentationTime = CMTime(
      value: ptsNanoseconds - segmentStart,
      timescale: 1_000_000_000
    )
    do {
      try await receiver.append(
        CVReadOnlyPixelBuffer(unsafeBuffer: pixelBuffer),
        with: presentationTime
      )
    } catch {
      throw MacHLSAVFoundationCapture.CaptureError.encoder(
        "フレームを書き込めません: \(error.localizedDescription)"
      )
    }
    lastPTS = ptsNanoseconds
    return completed
  }

  func finish() async throws -> Output? {
    guard let lastPTS else { return nil }
    return try await close(
      endNanoseconds: lastPTS + frameDurationNanoseconds
    )
  }

  func discard() {
    writer?.cancelWriting()
    if let workingURL { try? FileManager.default.removeItem(at: workingURL) }
    reset()
  }

  private func open(startNanoseconds: Int64) throws {
    let base = "hls-avfoundation-g\(generation)-\(identity)-\(fileSequence)"
    let working = outputDirectory.appendingPathComponent("\(base).mp4.part")
    let final = outputDirectory.appendingPathComponent("\(base).mp4")
    try? FileManager.default.removeItem(at: working)
    try? FileManager.default.removeItem(at: final)
    let writer = try AVAssetWriter(outputURL: working, fileType: .mp4)
    let frameRate = Double(fpsNumerator) / Double(fpsDenominator)
    let bitRate = Int(
      min(35_000_000, max(4_000_000, Double(width * height) * frameRate * 0.10))
    )
    let settings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoEncoderSpecificationKey: [
        kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder as String: true,
        kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String: false,
      ],
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: bitRate,
        AVVideoExpectedSourceFrameRateKey: frameRate,
        AVVideoAllowFrameReorderingKey: false,
        AVVideoMaxKeyFrameIntervalKey: max(1, Int(frameRate * 2)),
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      ],
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    let exactTimeScale = CMTimeScale(fpsNumerator)
    writer.movieTimeScale = exactTimeScale
    input.mediaTimeScale = exactTimeScale
    var attributes = CVPixelBufferCreationAttributes(
      pixelFormatType: CVPixelFormatType(
        rawValue: kCVPixelFormatType_32BGRA
      ),
      size: CVImageSize(width: width, height: height)
    )
    attributes.backing = .ioSurface
    let receiver = writer.inputPixelBufferReceiver(
      for: input,
      pixelBufferAttributes: attributes
    )
    do {
      try writer.start()
    } catch {
      throw MacHLSAVFoundationCapture.CaptureError.encoder(
        "writerを開始できません: \(error.localizedDescription)"
      )
    }
    writer.startSession(atSourceTime: .zero)
    self.writer = writer
    pixelBufferReceiver = receiver
    workingURL = working
    finalURL = final
    segmentStartNanoseconds = startNanoseconds
  }

  private func close(endNanoseconds: Int64) async throws -> Output {
    guard let writer, let receiver = pixelBufferReceiver, let workingURL,
      let finalURL,
      let start = segmentStartNanoseconds
    else {
      throw MacHLSAVFoundationCapture.CaptureError.encoder(
        "開始していない区間を終了できません"
      )
    }
    receiver.finish()
    // The last frame lasts until the segment's end, so the file spans exactly
    // [start, end) like the timeline it is placed on.
    writer.endSession(atSourceTime: CMTime(
      value: max(1, endNanoseconds - start),
      timescale: 1_000_000_000
    ))
    await writer.finishWriting()
    guard writer.status == .completed else {
      throw MacHLSAVFoundationCapture.CaptureError.encoder(
        writer.error?.localizedDescription ?? "一時映像を確定できません"
      )
    }
    try FileManager.default.moveItem(at: workingURL, to: finalURL)
    let output = Output(
      startNanoseconds: start,
      endNanoseconds: max(start + 1, endNanoseconds),
      url: finalURL
    )
    fileSequence += 1
    reset()
    return output
  }

  private func reset() {
    writer = nil
    pixelBufferReceiver = nil
    workingURL = nil
    finalURL = nil
    segmentStartNanoseconds = nil
    lastPTS = nil
  }
}
