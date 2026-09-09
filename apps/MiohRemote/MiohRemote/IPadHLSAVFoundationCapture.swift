import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

struct IPadHLSCaptureRatePolicy {
  static let acceleratedRate: Float = 2

  private var isWarmingUp = true
  private var isFillingTarget = true
  private var isRawQueuePaused = false

  mutating func finishWarmup() -> Bool {
    guard isWarmingUp else { return false }
    isWarmingUp = false
    return true
  }

  mutating func desiredRate(
    bufferedSeconds: Double,
    targetSeconds: Double,
    isLive: Bool,
    rawLeadSeconds: Double = 0,
    rawTargetSeconds: Double = .infinity
  ) -> Float {
    let rawLead = max(0, rawLeadSeconds.isFinite ? rawLeadSeconds : 0)
    let rawTarget = max(0.5, rawTargetSeconds.isFinite ? rawTargetSeconds : 2)
    if rawLead + 0.01 >= rawTarget {
      isRawQueuePaused = true
    } else if rawLead <= rawTarget * 0.55 {
      isRawQueuePaused = false
    }
    if isRawQueuePaused { return 0 }
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

/// Safari-compatible HLS capture for iPadOS. AVFoundation owns playlist,
/// segment, AES-128 HLS key and cookie handling; mioh receives only decoded
/// video frames.
/// A separate muted player can run at 2x without changing the audible 1x clock.
@MainActor
final class IPadHLSAVFoundationCapture {
  private final class SeekCompletionState: @unchecked Sendable {
    var result: Bool?
  }

  struct CapturedFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let ptsNanoseconds: Int64
    let timelineSeconds: Double
  }

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
  private var endObserver: NSObjectProtocol?
  private var didReachEnd = false
  private var restoredBufferLeadSeconds = 0.0
  private var ratePolicy = IPadHLSCaptureRatePolicy()
  private var requestedCaptureRate: Float = 1
  private var capturePlaybackStarted = false
  private var usesRawFrameOutput = false
  private var latestCapturedTimelineSeconds = 0.0
  private var consumedTimelineSeconds = 0.0
  private var rawLeadTargetSeconds: Double
  private var capturedFrameCount = 0

  init(
    asset: AVURLAsset,
    outputDirectory: URL,
    startSeconds: Double,
    duration: Double,
    isLive: Bool,
    generation: Int,
    segmentSeconds: Double,
    forwardBufferSeconds: Double,
    log: @escaping @MainActor (String) -> Void = { _ in }
  ) {
    self.asset = asset
    self.outputDirectory = outputDirectory
    requestedStartSeconds = max(0, startSeconds.isFinite ? startSeconds : 0)
    knownDuration = max(0, duration.isFinite ? duration : 0)
    self.isLive = isLive
    self.generation = generation
    self.segmentSeconds = max(0.5, segmentSeconds)
    self.forwardBufferSeconds = max(2, forwardBufferSeconds)
    rawLeadTargetSeconds = max(1, min(3, forwardBufferSeconds * 0.25))
    latestCapturedTimelineSeconds = requestedStartSeconds
    consumedTimelineSeconds = requestedStartSeconds
    self.log = log
    player.isMuted = true
    player.automaticallyWaitsToMinimizeStalling = true
    player.preventsDisplaySleepDuringVideoPlayback = false
    player.actionAtItemEnd = .pause
  }

  func makePlaybackItem() -> AVPlayerItem {
    AVPlayerItem(asset: asset)
  }

  func setForwardBufferDuration(_ seconds: Double) {
    forwardBufferSeconds = max(2, seconds.isFinite ? seconds : 8)
    rawLeadTargetSeconds = max(1, min(3, forwardBufferSeconds * 0.25))
    captureItem?.preferredForwardBufferDuration = forwardBufferSeconds
    updateCaptureRate()
  }

  func setRestoredBufferLead(_ seconds: Double) {
    restoredBufferLeadSeconds = max(0, seconds.isFinite ? seconds : 0)
    updateCaptureRate()
  }

  func setRawFramesConsumed(through timelineSeconds: Double) {
    guard timelineSeconds.isFinite else { return }
    consumedTimelineSeconds = max(consumedTimelineSeconds, timelineSeconds)
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
          try await self.capture(
            segmentContinuation: continuation,
            frameContinuation: nil
          )
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

  func frames() throws -> AsyncThrowingStream<CapturedFrame, Error> {
    guard captureTask == nil else { throw CaptureError.alreadyStarted }
    usesRawFrameOutput = true
    return AsyncThrowingStream { continuation in
      let task = Task { @MainActor [weak self] in
        guard let self else {
          continuation.finish(throwing: CancellationError())
          return
        }
        do {
          try await self.capture(
            segmentContinuation: nil,
            frameContinuation: continuation
          )
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
    captureItem = nil
    videoOutput = nil
    capturePlaybackStarted = false
    requestedCaptureRate = 1
    ratePolicy = IPadHLSCaptureRatePolicy()
    usesRawFrameOutput = false
    latestCapturedTimelineSeconds = requestedStartSeconds
    consumedTimelineSeconds = requestedStartSeconds
    capturedFrameCount = 0
  }

  private func capture(
    segmentContinuation:
      AsyncThrowingStream<CapturedSegment, Error>.Continuation?,
    frameContinuation:
      AsyncThrowingStream<CapturedFrame, Error>.Continuation?
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
    let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferIOSurfacePropertiesKey as String: [String: String](),
    ])
    output.suppressesPlayerRendering = true
    item.add(output)
    captureItem = item
    videoOutput = output
    didReachEnd = false
    endObserver = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemDidPlayToEndTime,
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
    var writer: IPadHLSCaptureSegmentWriter?
    var writerDimensions: (width: Int, height: Int)?
    var nextSequence = 0
    var lastPTS: Int64?
    var lastFrameAt = Date()
    var lastPlaybackRecoveryAt = Date.distantPast

    func yield(_ encoded: IPadHLSCaptureSegmentWriter.Output?) {
      guard let encoded else { return }
      segmentContinuation?.yield(
        CapturedSegment(
          sequence: nextSequence,
          startSeconds: Double(encoded.startNanoseconds) / 1_000_000_000,
          endSeconds: Double(encoded.endNanoseconds) / 1_000_000_000,
          url: encoded.url
        )
      )
      nextSequence += 1
      if ratePolicy.finishWarmup() { updateCaptureRate() }
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
          var displayTime = CMTime.invalid
          if let pixelBuffer = output.copyPixelBuffer(
            forItemTime: itemTime,
            itemTimeForDisplay: &displayTime
          ) {
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

                if let frameContinuation {
                  latestCapturedTimelineSeconds = timelineSeconds
                  capturedFrameCount += 1
                  frameContinuation.yield(
                    CapturedFrame(
                      pixelBuffer: pixelBuffer,
                      ptsNanoseconds: pts,
                      timelineSeconds: timelineSeconds
                    )
                  )
                  if capturedFrameCount >= 8 { _ = ratePolicy.finishWarmup() }
                  updateCaptureRate()
                } else if writer == nil {
                  pendingFrames.append((pixelBuffer, pts))
                  if pendingFrames.count >= 8 {
                    let rate = Self.estimatedFrameRate(
                      from: pendingFrames.map(\.1)
                    )
                    let created = try IPadHLSCaptureSegmentWriter(
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

        if !isLive, knownDuration > 0 {
          let sourceSeconds = player.currentTime().seconds - timelineOffset
          if sourceSeconds.isFinite,
            sourceSeconds >= knownDuration - 0.02
          {
            break captureLoop
          }
        }
        let frameSilence = Date().timeIntervalSince(lastFrameAt)
        if requestedCaptureRate > 0.05, frameSilence > 1.5,
          Date().timeIntervalSince(lastPlaybackRecoveryAt) > 1.5
        {
          lastPlaybackRecoveryAt = Date()
          recoverCapturePlaybackIfNeeded(output: output)
        }
        if requestedCaptureRate > 0.05, frameSilence > 45 {
          throw CaptureError.source("45秒間映像フレームを取得できませんでした")
        }
        try await Task.sleep(nanoseconds: 5_000_000)
      }

      if frameContinuation == nil, writer == nil, !pendingFrames.isEmpty,
        let first = pendingFrames.first
      {
        let rate = Self.estimatedFrameRate(from: pendingFrames.map(\.1))
        let created = try IPadHLSCaptureSegmentWriter(
          outputDirectory: outputDirectory,
          width: CVPixelBufferGetWidth(first.0),
          height: CVPixelBufferGetHeight(first.0),
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
      yield(try await writer?.finish())
    } catch {
      writer?.discard()
      throw error
    }
    player.pause()
  }

  private func updateCaptureRate() {
    let desired = ratePolicy.desiredRate(
      bufferedSeconds: restoredBufferLeadSeconds,
      targetSeconds: forwardBufferSeconds,
      isLive: isLive,
      rawLeadSeconds: usesRawFrameOutput
        ? max(0, latestCapturedTimelineSeconds - consumedTimelineSeconds) : 0,
      rawTargetSeconds: usesRawFrameOutput
        ? rawLeadTargetSeconds : .infinity
    )
    guard abs(desired - requestedCaptureRate) >= 0.05 else { return }
    let previous = requestedCaptureRate
    requestedCaptureRate = desired
    guard capturePlaybackStarted, captureItem != nil else { return }
    if desired < 0.05 {
      player.pause()
    } else if previous < 0.05
      || player.timeControlStatus != .playing
    {
      videoOutput?.requestNotificationOfMediaDataChange(
        withAdvanceInterval: 1.0 / 120.0
      )
      player.playImmediately(atRate: desired)
    } else {
      player.rate = desired
    }
    if desired < 0.05 {
      log(
        "HLS取込: 未復元フレームが"
          + String(
            format: "%.1f",
            max(0, latestCapturedTimelineSeconds - consumedTimelineSeconds)
          )
          + "秒あるため取込を一時停止します"
      )
    } else if previous < 0.05 {
      log("HLS取込: 未復元フレームを消費したため取込を再開します")
    } else if desired > 1.05 {
      log("HLS取込: \(String(format: "%.2f", desired))倍で先読みを増やします")
    } else if previous > 1.05 {
      log(
        "HLS取込: 復元バッファ"
          + "\(String(format: "%.1f", restoredBufferLeadSeconds))秒で1倍速へ戻します"
      )
    }
  }

  private func recoverCapturePlaybackIfNeeded(
    output: AVPlayerItemVideoOutput
  ) {
    guard capturePlaybackStarted, captureItem != nil,
      requestedCaptureRate > 0.05
    else { return }
    output.requestNotificationOfMediaDataChange(
      withAdvanceInterval: 1.0 / 120.0
    )
    if player.timeControlStatus != .playing
      || abs(player.rate - requestedCaptureRate) >= 0.05
    {
      player.playImmediately(atRate: requestedCaptureRate)
      log("HLS取込: シーク後の停止を検出したため取込を再開します")
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
    if isLive, let seekable = item.seekableTimeRanges.last?.timeRangeValue {
      let lower = CMTimeGetSeconds(seekable.start)
      let upper = CMTimeGetSeconds(CMTimeRangeGetEnd(seekable))
      let distanceFromEdge = max(0, knownDuration - requestedStartSeconds)
      target = min(upper, max(lower, upper - distanceFromEdge))
    } else {
      target = requestedStartSeconds
    }
    // Safari seeks HLS to a nearby independently decodable boundary. Requiring
    // a sample-exact seek can leave AVPlayer waiting indefinitely on segmented
    // or encrypted VODs (notably Jable) after a user scrub. Try the nearest
    // segment/keyframe first, then use AVPlayer's fully relaxed seek as a
    // bounded fallback. The actual start is mapped back to the requested
    // timeline below, so restoration timestamps remain continuous.
    let segmentTolerance = CMTime(
      seconds: max(0.5, segmentSeconds),
      preferredTimescale: 600
    )
    var completed = try await performBoundedSeek(
      to: target,
      toleranceBefore: segmentTolerance,
      toleranceAfter: segmentTolerance,
      timeoutSeconds: 12
    )
    if !completed {
      player.currentItem?.cancelPendingSeeks()
      completed = try await performBoundedSeek(
        to: target,
        toleranceBefore: .positiveInfinity,
        toleranceAfter: .positiveInfinity,
        timeoutSeconds: 12
      )
    }
    guard completed else {
      player.currentItem?.cancelPendingSeeks()
      throw CaptureError.source("開始位置への移動がタイムアウトしました")
    }
    let actual = player.currentTime().seconds
    guard actual.isFinite else {
      throw CaptureError.source("開始位置の時間情報を取得できませんでした")
    }
    return actual
  }

  private func performBoundedSeek(
    to seconds: Double,
    toleranceBefore: CMTime,
    toleranceAfter: CMTime,
    timeoutSeconds: Double
  ) async throws -> Bool {
    let completion = SeekCompletionState()
    player.seek(
      to: CMTime(seconds: seconds, preferredTimescale: 600),
      toleranceBefore: toleranceBefore,
      toleranceAfter: toleranceAfter
    ) { [completion] finished in
      Task { @MainActor in
        guard completion.result == nil else { return }
        completion.result = finished
      }
    }
    let deadline = Date().addingTimeInterval(max(1, timeoutSeconds))
    do {
      while completion.result == nil, Date() < deadline {
        try Task.checkCancellation()
        try await Task.sleep(nanoseconds: 50_000_000)
      }
      try Task.checkCancellation()
    } catch {
      player.currentItem?.cancelPendingSeeks()
      throw error
    }
    return completion.result == true
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

final class IPadRealtimeRestoredSegmentWriter: @unchecked Sendable {
  private let outputDirectory: URL
  private let generation: Int
  private let width: Int
  private let height: Int
  private let codec: AVVideoCodecType
  private let bitrateMultiplier: Double
  private let identity = UUID().uuidString.lowercased()
  private var sequence = 0
  private var writer: MiohIPadVideoWriter?
  private var outputURL: URL?
  private var startNanoseconds: Int64?
  private var lastNanoseconds: Int64?

  init(
    outputDirectory: URL,
    generation: Int,
    width: Int,
    height: Int,
    videoCodec: String,
    bitrateMultiplier: Double
  ) {
    self.outputDirectory = outputDirectory
    self.generation = generation
    self.width = width
    self.height = height
    codec = videoCodec == "h264" ? .h264 : .hevc
    self.bitrateMultiplier = bitrateMultiplier
  }

  func append(
    _ frame: MiohIPadRealtimeOutputFrame,
    preferShortSegment: Bool = false
  ) async throws
    -> IPadHLSAVFoundationCapture.CapturedSegment?
  {
    var completed: IPadHLSAVFoundationCapture.CapturedSegment?
    if let startNanoseconds,
      frame.ptsNanoseconds >= startNanoseconds
        + targetNanoseconds(preferShortSegment: preferShortSegment)
    {
      completed = try await close(endNanoseconds: frame.ptsNanoseconds)
    }
    if writer == nil { try open(at: frame.ptsNanoseconds) }
    guard let writer, let startNanoseconds else {
      throw IPadHLSAVFoundationCapture.CaptureError.encoder(
        "復元済み映像writerを開始できません"
      )
    }
    try await writer.append(
      frame.pixelBuffer,
      presentationTimeNanoseconds: max(
        0,
        frame.ptsNanoseconds - startNanoseconds
      )
    )
    lastNanoseconds = frame.ptsNanoseconds
    return completed
  }

  func finish() async throws -> IPadHLSAVFoundationCapture.CapturedSegment? {
    guard let lastNanoseconds else { return nil }
    return try await close(
      endNanoseconds: lastNanoseconds + 1_000_000_000 / 24
    )
  }

  func cancel() {
    writer?.cancel()
    if let outputURL { try? FileManager.default.removeItem(at: outputURL) }
    reset()
  }

  private func targetNanoseconds(preferShortSegment: Bool) -> Int64 {
    // While playback inventory is below its target, publish every two seconds
    // like the macOS realtime path. Once the queue is healthy, longer files
    // retain the lower AVQueuePlayer/VideoToolbox overhead.
    if preferShortSegment { return 2_000_000_000 }
    // Preserve the two-second first-frame path and reach the existing six-
    // second startup threshold with the second item. Steady state then
    // amortizes queue-item and VideoToolbox setup over six seconds.
    if sequence == 0 { return 2_000_000_000 }
    if sequence == 1 { return 4_000_000_000 }
    return 6_000_000_000
  }

  private func open(at ptsNanoseconds: Int64) throws {
    let name = "hls-restored-g\(generation)-\(identity)-\(sequence).mp4"
    let url = outputDirectory.appendingPathComponent(name)
    try? FileManager.default.removeItem(at: url)
    writer = try MiohIPadVideoWriter(
      url: url,
      width: width,
      height: height,
      fpsNumerator: 30,
      fpsDenominator: 1,
      codec: codec,
      sourceBitRate: 0,
      bitrateMultiplier: bitrateMultiplier,
      fastStart: true
    )
    outputURL = url
    startNanoseconds = ptsNanoseconds
  }

  private func close(endNanoseconds: Int64) async throws
    -> IPadHLSAVFoundationCapture.CapturedSegment
  {
    guard let writer, let outputURL, let startNanoseconds else {
      throw IPadHLSAVFoundationCapture.CaptureError.encoder(
        "開始していない復元済み区間を終了できません"
      )
    }
    let duration = max(1, endNanoseconds - startNanoseconds)
    _ = try await writer.finish(durationNanoseconds: duration)
    let output = IPadHLSAVFoundationCapture.CapturedSegment(
      sequence: sequence,
      startSeconds: Double(startNanoseconds) / 1_000_000_000,
      endSeconds: Double(startNanoseconds + duration) / 1_000_000_000,
      url: outputURL
    )
    sequence += 1
    reset()
    return output
  }

  private func reset() {
    writer = nil
    outputURL = nil
    startNanoseconds = nil
    lastNanoseconds = nil
  }
}

/// Writes restored frames in the same order they are produced. Keeping this
/// path single-stage avoids retaining a second batch of large pixel buffers
/// while Core AI is processing the next batch.
@MainActor
final class IPadRealtimeRestoredOutputPipeline {
  typealias CapturedSegment = IPadHLSAVFoundationCapture.CapturedSegment

  private let writer: IPadRealtimeRestoredSegmentWriter
  private let didComplete:
    @MainActor @Sendable (CapturedSegment) throws -> Void
  init(
    writer: IPadRealtimeRestoredSegmentWriter,
    didComplete:
      @escaping @MainActor @Sendable (CapturedSegment) throws -> Void
  ) {
    self.writer = writer
    self.didComplete = didComplete
  }

  func submit(
    _ frames: [MiohIPadRealtimeOutputFrame],
    preferShortSegments: Bool
  ) async throws {
    guard !frames.isEmpty else { return }
    for frame in frames {
      try Task.checkCancellation()
      if let completed = try await writer.append(
        frame,
        preferShortSegment: preferShortSegments
      ) {
        try didComplete(completed)
      }
    }
  }

  func finish() async throws -> CapturedSegment? {
    let completed = try await writer.finish()
    if let completed { try didComplete(completed) }
    return completed
  }

  func cancel() async {
    writer.cancel()
  }
}

@MainActor
private final class IPadHLSCaptureSegmentWriter {
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
  private var writer: MiohIPadVideoWriter?
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
      throw IPadHLSAVFoundationCapture.CaptureError.encoder(
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
    guard let writer, let start = segmentStartNanoseconds else {
      throw IPadHLSAVFoundationCapture.CaptureError.encoder(
        "一時映像writerを開始できません"
      )
    }
    try await writer.append(
      pixelBuffer,
      presentationTimeNanoseconds: max(0, ptsNanoseconds - start)
    )
    lastPTS = ptsNanoseconds
    return completed
  }

  func finish() async throws -> Output? {
    guard let lastPTS else { return nil }
    return try await close(endNanoseconds: lastPTS + frameDurationNanoseconds)
  }

  func discard() {
    writer?.cancel()
    if let finalURL { try? FileManager.default.removeItem(at: finalURL) }
    reset()
  }

  private func open(startNanoseconds: Int64) throws {
    let base = "hls-avfoundation-g\(generation)-\(identity)-\(fileSequence)"
    let url = outputDirectory.appendingPathComponent("\(base).mp4")
    try? FileManager.default.removeItem(at: url)
    writer = try MiohIPadVideoWriter(
      url: url,
      width: width,
      height: height,
      fpsNumerator: fpsNumerator,
      fpsDenominator: fpsDenominator,
      codec: .h264,
      sourceBitRate: 0,
      bitrateMultiplier: 1,
      fastStart: true
    )
    finalURL = url
    segmentStartNanoseconds = startNanoseconds
  }

  private func close(endNanoseconds: Int64) async throws -> Output {
    guard let writer, let finalURL, let start = segmentStartNanoseconds else {
      throw IPadHLSAVFoundationCapture.CaptureError.encoder(
        "開始していない区間を終了できません"
      )
    }
    let duration = max(1, endNanoseconds - start)
    _ = try await writer.finish(durationNanoseconds: duration)
    let output = Output(
      startNanoseconds: start,
      endNanoseconds: start + duration,
      url: finalURL
    )
    fileSequence += 1
    reset()
    return output
  }

  private func reset() {
    writer = nil
    finalURL = nil
    segmentStartNanoseconds = nil
    lastPTS = nil
  }
}
