import AVFoundation
import CoreMedia
import Foundation

/// The producer awaits this before publishing a file. Audio never runs on an
/// independent audible player: the returned movie owns both tracks and clock.
@MainActor
protocol MacHLSPlaybackPreparing: AnyObject {
  var sourceTimeOffset: Double { get }
  func prepare() async throws
  func movie(videoURL: URL, start: Double, end: Double, outputURL: URL) async throws
  func cancel()
}

@available(macOS 27.0, *)
@MainActor
final class MacHLSUnifiedPlayback: MacHLSPlaybackPreparing {
  enum Failure: LocalizedError {
    case media(String)
    var errorDescription: String? {
      switch self {
      case .media(let detail): return "HLS音声・映像の統合に失敗しました: \(detail)"
      }
    }
  }

  private let item: AVPlayerItem
  private let player = AVPlayer()
  private let output: AVPlayerItemSampleBufferOutput
  private let start: Double
  private let duration: Double
  private let isLive: Bool
  private var prepared = false
  private var cancelled = false
  private var hasAudio = true
  private var timeOffset = 0.0
  var sourceTimeOffset: Double { timeOffset }
  private var pending: CMSampleBuffer?
  private var lastReadEnd = -Double.infinity
  private var audioEnd = Double.infinity

  init(item: AVPlayerItem, start: Double, duration: Double, isLive: Bool) {
    self.item = item
    self.start = start
    self.duration = duration
    self.isLive = isLive
    let configuration = AVPlayerItemSampleBufferOutputAudioConfiguration()
    configuration.requestedAudioFormat = AVAudioFormat(
      standardFormatWithSampleRate: 48_000, channels: 2
    )!.formatDescription
    output = AVPlayerItemSampleBufferOutput(configuration: configuration)
    output.suppressesPlayerRendering = true
    item.add(output)
    item.preferredForwardBufferDuration = 8
    player.isMuted = true
    player.preventsDisplaySleepDuringVideoPlayback = false
  }

  func prepare() async throws {
    try checkCancellation()
    guard !prepared else { return }
    player.replaceCurrentItem(with: item)
    let deadline = Date().addingTimeInterval(30)
    while item.status == .unknown, Date() < deadline {
      try checkCancellation()
      try await Task.sleep(for: .milliseconds(25))
    }
    guard item.status == .readyToPlay else {
      throw Failure.media(item.error?.localizedDescription ?? "音声の読込がタイムアウトしました")
    }
    // HLS master/media assets may report no AVAsset audio tracks until their
    // variant is actually decoded. The macOS 27 sample-buffer output is the
    // authoritative audio capability check, so keep the pull path enabled.
    hasAudio = true
    let target: Double
    if isLive {
      guard let range = item.seekableTimeRanges.last?.timeRangeValue,
        range.start.seconds.isFinite, range.duration.seconds.isFinite
      else { throw Failure.media("ライブ音声の再生可能範囲がありません") }
      target = max(range.start.seconds, CMTimeRangeGetEnd(range).seconds - max(0, duration - start))
    } else {
      target = start
    }
    // The output can pull PCM ahead while paused. It does not need a second
    // real-time player, playback-rate stretching, or a second network decoder.
    let seek = await player.seek(
      to: CMTime(seconds: target, preferredTimescale: 48_000),
      toleranceBefore: .zero, toleranceAfter: .zero
    )
    try checkCancellation()
    guard seek, player.currentTime().seconds.isFinite else {
      throw Failure.media("音声の開始位置へ移動できませんでした")
    }
    // The player is started only while a requested interval is being pulled.
    // Leaving it running here would advance the HLS clock during the (often
    // much slower) video restoration, so the next interval could no longer
    // seek to its requested source time.
    timeOffset = player.currentTime().seconds - start
    if !isLive, let track = try? await item.asset.loadTracks(withMediaType: .audio).first {
      let range = try await track.load(.timeRange)
      let end = CMTimeRangeGetEnd(range).seconds - timeOffset
      if end.isFinite, end > start { audioEnd = end }
    }
    prepared = true
  }

  func cancel() {
    cancelled = true
    player.pause()
    item.cancelPendingSeeks()
    item.remove(output)
    player.replaceCurrentItem(with: nil)
    pending = nil
  }

  private func checkCancellation() throws {
    try Task.checkCancellation()
    if cancelled { throw CancellationError() }
  }

  private func audio(start: Double, end: Double) async throws -> [CMSampleBuffer] {
    guard hasAudio else { return [] }
    // Restore generation can take minutes per interval. Seek the muted source
    // player for every request instead of relying on wall-clock progression.
    // This keeps the decoded PCM on the same media timeline as the output
    // video even when intervals are generated out of real time.
    let target = start + timeOffset
    let sought = await player.seek(
      to: CMTime(seconds: target, preferredTimescale: 48_000),
      toleranceBefore: .zero, toleranceAfter: .zero
    )
    guard sought else { throw Failure.media("音声区間の開始位置へ移動できませんでした") }
    pending = nil
    lastReadEnd = start
    player.play()
    defer { player.pause() }
    var samples: [CMSampleBuffer] = []
    var deadline = Date().addingTimeInterval(30)
    while true {
      try checkCancellation()
      if pending == nil, let next = output.nextAvailableSampleBuffer() {
        let copied: CMSampleBuffer? = next.sampleBuffer.withUnsafeSampleBuffer { sample in
          guard CMSampleBufferGetNumSamples(sample) > 0 else { return nil }
          var copy: CMSampleBuffer?
          guard CMSampleBufferCreateCopy(
            allocator: kCFAllocatorDefault, sampleBuffer: sample, sampleBufferOut: &copy
          ) == noErr else { return nil }
          return copy
        }
        guard let copied else { continue }
        pending = copied
      }
      guard let sample = pending else {
        if lastReadEnd >= min(end, audioEnd) - 0.000_1 { break }
        guard item.status != .failed, Date() < deadline else {
          throw Failure.media(item.error?.localizedDescription ?? "音声区間 \(String(format: "%.3f–%.3f", start, end)) の取得がタイムアウトしました")
        }
        try await Task.sleep(for: .milliseconds(10))
        continue
      }
      let pts = CMSampleBufferGetOutputPresentationTimeStamp(sample).seconds - timeOffset
      let sampleEnd = pts + CMSampleBufferGetDuration(sample).seconds
      guard pts.isFinite, sampleEnd.isFinite, sampleEnd > pts else {
        throw Failure.media("PCMの時間範囲が不正です")
      }
      if pts >= end { break }
      lastReadEnd = sampleEnd
      // Some HLS PCM buffers reject CMSampleBufferCopySampleBufferForRange.
      // Assign a crossing buffer to exactly one interval by its midpoint; this
      // avoids duplicate audio at queue boundaries while limiting the boundary
      // error to one decoded PCM buffer (normally < 180 ms).
      let midpoint = (pts + sampleEnd) * 0.5
      if midpoint >= start && midpoint < end {
        var count = 0
        CMSampleBufferGetSampleTimingInfoArray(
          sample, entryCount: 0, arrayToFill: nil, entriesNeededOut: &count
        )
        var timings = [CMSampleTimingInfo](repeating: CMSampleTimingInfo(), count: count)
        CMSampleBufferGetSampleTimingInfoArray(
          sample, entryCount: count, arrayToFill: &timings, entriesNeededOut: &count
        )
        for index in timings.indices {
          timings[index].presentationTimeStamp = CMTime(
            seconds: max(0, timings[index].presentationTimeStamp.seconds - start),
            preferredTimescale: 48_000
          )
          if timings[index].decodeTimeStamp.isNumeric {
            timings[index].decodeTimeStamp = CMTime(
              seconds: max(0, timings[index].decodeTimeStamp.seconds - start),
              preferredTimescale: 48_000
            )
          }
        }
        var retimed: CMSampleBuffer?
        guard CMSampleBufferCreateCopyWithNewTiming(
          allocator: kCFAllocatorDefault, sampleBuffer: sample,
          sampleTimingEntryCount: count, sampleTimingArray: &timings,
          sampleBufferOut: &retimed
        ) == noErr, let retimed else {
          throw Failure.media("PCM時刻を設定できません")
        }
        samples.append(retimed)
      }
      if sampleEnd >= end || pts >= end { break } // keep crossing sample for next interval
      pending = nil
      deadline = Date().addingTimeInterval(30)
    }
    return samples
  }

  func movie(videoURL: URL, start: Double, end: Double, outputURL: URL) async throws {
    guard start.isFinite, end.isFinite, end > start else {
      throw Failure.media("出力区間が不正です")
    }
    try await prepare()
    try checkCancellation()
    // `audio` works in the public HLS timeline (the sample timestamps are
    // normalized by `timeOffset` internally). Do not apply the offset twice.
    let samples = try await audio(start: start, end: end)
    try await Self.writeMovie(videoURL: videoURL, audio: samples, duration: end - start, outputURL: outputURL)
    try checkCancellation()
  }

  /// Compressed video is copied. LPCM in MOV avoids AAC encoder priming at
  /// every two-second boundary; both tracks have the same zero-based timeline.
  static func writeMovie(videoURL: URL, audio: [CMSampleBuffer], duration: Double, outputURL: URL) async throws {
    let asset = AVURLAsset(url: videoURL)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw Failure.media("復元映像トラックがありません")
    }
    let range = try await track.load(.timeRange)
    let formats = try await track.load(.formatDescriptions)
    let reader = try AVAssetReader(asset: asset)
    let videoOutput = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
    reader.add(videoOutput)
    try? FileManager.default.removeItem(at: outputURL)
    let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
    let videoInput = AVAssetWriterInput(
      mediaType: .video, outputSettings: nil, sourceFormatHint: formats.first
    )
    videoInput.transform = try await track.load(.preferredTransform)
    let videoReceiver = writer.inputReceiver(for: videoInput)
    let audioReceiver: AVAssetWriterInput.SampleBufferReceiver?
    if let first = audio.first {
      let input = AVAssetWriterInput(
        mediaType: .audio,
        outputSettings: [
          AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48_000,
          AVNumberOfChannelsKey: 2, AVLinearPCMBitDepthKey: 16,
          AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
          AVLinearPCMIsNonInterleaved: false,
        ],
        sourceFormatHint: CMSampleBufferGetFormatDescription(first)
      )
      audioReceiver = writer.inputReceiver(for: input)
    } else {
      audioReceiver = nil
    }
    func appendImmediately(
      _ receiver: AVAssetWriterInput.SampleBufferReceiver,
      _ sample: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>
    ) async throws {
      let deadline = Date().addingTimeInterval(10)
      while try !receiver.appendImmediately(sample) {
        try Task.checkCancellation()
        guard writer.status == .writing, Date() < deadline else {
          throw Failure.media(writer.error?.localizedDescription ?? "AVAssetWriter入力が詰まりました")
        }
        try await Task.sleep(for: .milliseconds(2))
      }
    }
    try writer.start()
    writer.startSession(atSourceTime: .zero)
    guard reader.startReading() else {
      throw Failure.media(reader.error?.localizedDescription ?? "映像を読めません")
    }
    while let sample = videoOutput.copyNextSampleBuffer() {
      try Task.checkCancellation()
      let videoPTS = CMSampleBufferGetPresentationTimeStamp(sample).seconds - range.start.seconds
      guard videoPTS < duration else { continue }
      var count = 0
      CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: 0, arrayToFill: nil, entriesNeededOut: &count)
      var timings = [CMSampleTimingInfo](repeating: CMSampleTimingInfo(), count: count)
      CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: count, arrayToFill: &timings, entriesNeededOut: &count)
      for index in timings.indices {
        timings[index].presentationTimeStamp = CMTime(seconds: max(0, timings[index].presentationTimeStamp.seconds - range.start.seconds), preferredTimescale: 48_000)
        if timings[index].decodeTimeStamp.isNumeric {
          timings[index].decodeTimeStamp = CMTime(seconds: max(0, timings[index].decodeTimeStamp.seconds - range.start.seconds), preferredTimescale: 48_000)
        }
      }
      var adjusted: CMSampleBuffer?
      guard CMSampleBufferCreateCopyWithNewTiming(allocator: kCFAllocatorDefault, sampleBuffer: sample, sampleTimingEntryCount: count, sampleTimingArray: &timings, sampleBufferOut: &adjusted) == noErr, let adjusted else {
        throw Failure.media("映像の時刻を設定できません")
      }
      try await appendImmediately(videoReceiver, CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: adjusted))
    }
    if let audioReceiver {
      for sample in audio {
        try await appendImmediately(audioReceiver, CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: sample))
      }
    }
    guard reader.status == .completed else {
      throw Failure.media(reader.error?.localizedDescription ?? "映像の読込が中断されました")
    }
    videoReceiver.finish()
    audioReceiver?.finish()
    writer.endSession(atSourceTime: CMTime(seconds: duration, preferredTimescale: 48_000))
    await writer.finishWriting()
    guard writer.status == .completed else {
      throw Failure.media(writer.error?.localizedDescription ?? "統合映像を確定できません")
    }
  }
}
