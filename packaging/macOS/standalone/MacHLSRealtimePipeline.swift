import Foundation

enum MacHLSProductionEvent: Sendable {
  case ready(duration: Double, isLive: Bool)
  case segment(
    sequence: Int,
    startSeconds: Double,
    endSeconds: Double,
    url: URL,
    codec: String
  )
  /// The media timeline advanced beyond the locally buffered live window.
  /// Consumers must discard queued restored media before accepting segments
  /// at or after `position`.
  case discontinuity(position: Double)
  case progress(position: Double, duration: Double)
  case ended(duration: Double)
}

/// Converts an authenticated HLS media playlist into the same rolling,
/// restored MP4 segments consumed by the normal macOS realtime player.
///
/// Each HLS segment is restored with temporal context from its neighbours:
/// after two downloads the first segment is emitted, after three downloads
/// the centre segment is emitted and the oldest input is retired. This keeps
/// latency bounded while avoiding an artificial BasicVSR++ boundary at every
/// HLS cut. The final segment is flushed when a VOD playlist is exhausted.
@MainActor
final class MacHLSRealtimeProducer {
  typealias Logger = (String) -> Void
  typealias EventSink = (MacHLSProductionEvent) -> Void

  private struct RestorationSource {
    let mediaSegment: IPadHLSMediaSegment
    let timelineStart: Double
    let localURL: URL

    var timelineEnd: Double {
      timelineStart + mediaSegment.duration
    }
  }

  private struct WorkerEvent: Decodable {
    let kind: String
    let generation: Int
    let sequence: Int?
    let startNs: Int64?
    let endNs: Int64?
    let path: String?
    let positionNs: Int64?
    let codec: String?
    let message: String?
    let detail: String?
  }

  private enum ProductionError: LocalizedError {
    case invalidSource(String)
    case invalidTimeline(String)
    case worker(String)
    case missingOutput(String)

    var errorDescription: String? {
      switch self {
      case .invalidSource(let detail):
        return "HLS配信を復元できません: \(detail)"
      case .invalidTimeline(let detail):
        return "HLSの時間情報が不正です: \(detail)"
      case .worker(let detail):
        return "HLS区間のBasicVSR++復元に失敗しました: \(detail)"
      case .missingOutput(let detail):
        return "HLS復元区間を保存できません: \(detail)"
      }
    }
  }

  private let source: IPadResolvedMediaSource
  private let runner: RestorationRunner
  private let resources: URL
  private let sessionDirectory: URL
  private let startSeconds: Double
  private let generation: Int
  private let log: Logger

  private var downloader: IPadHLSResourceDownloader?
  private var activeProcess: Process?
  private var activeWorkerInput: Pipe?
  private var cancellationRequested = false
  private var nextOutputSequence = 0
  private var isRunActive = false
  private var completionWaiters: [CheckedContinuation<Void, Never>] = []

  init(
    source: IPadResolvedMediaSource,
    runner: RestorationRunner,
    resources: URL,
    sessionDirectory: URL,
    startSeconds: Double,
    generation: Int,
    log: @escaping Logger
  ) {
    self.source = source
    self.runner = runner
    self.resources = resources
    self.sessionDirectory = sessionDirectory
    self.startSeconds = max(0, startSeconds.isFinite ? startSeconds : 0)
    self.generation = generation
    self.log = log
  }

  func run(emit: @escaping EventSink) async throws {
    guard !isRunActive else {
      throw ProductionError.invalidSource("同じHLS復元処理を同時に開始できません")
    }
    isRunActive = true
    defer { completeRun() }
    guard source.kind == .hls, var playlist = source.hlsPlaylist,
      !playlist.segments.isEmpty
    else {
      throw ProductionError.invalidSource("メディアプレイリストがありません")
    }
    try checkCancellation()
    try FileManager.default.createDirectory(
      at: sessionDirectory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    let restoredDirectory = sessionDirectory.appendingPathComponent(
      "restored-hls",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: restoredDirectory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )

    var activeSource = source
    var downloader = makeDownloader(for: activeSource)
    self.downloader = downloader
    defer {
      downloader.cancel()
      if self.downloader === downloader { self.downloader = nil }
      activeWorkerInput = nil
      activeProcess = nil
    }

    var timelineStarts = Dictionary(
      playlist.segments.map { ($0.sequence, $0.startSeconds) },
      uniquingKeysWith: { first, _ in first }
    )
    var duration = maximumTimelineEnd(
      playlist: playlist,
      timelineStarts: timelineStarts
    )
    if !playlist.isLive {
      duration = max(duration, playlist.duration)
    }
    emit(.ready(duration: duration, isLive: playlist.isLive))
    log(
      "HLSリアルタイム復元を開始: \(playlist.url.host ?? playlist.url.absoluteString)"
        + (playlist.isLive ? " / ライブ\n" : " / \(formatDuration(duration))\n")
    )

    var nextMediaSequence = startingSequence(
      playlist: playlist,
      requestedStartSeconds: startSeconds
    )
    var lastRefresh = Date.distantPast
    var consecutiveRefreshFailures = 0
    var restorationWindow: [RestorationSource] = []
    defer {
      for source in restorationWindow {
        try? FileManager.default.removeItem(at: source.localURL)
      }
    }

    productionLoop: while true {
      try checkCancellation()
      let refreshInterval = max(
        0.5,
        min(2, (playlist.targetDuration ?? 2) / 2)
      )
      let availableBeforeRefresh = playlist.segments.contains {
        $0.sequence >= nextMediaSequence
      }
      if playlist.isLive,
        !availableBeforeRefresh
          || Date().timeIntervalSince(lastRefresh) >= refreshInterval
      {
        if !availableBeforeRefresh {
          try await sleep(seconds: 0.5)
        }
        let refreshedSource: IPadResolvedMediaSource
        do {
          refreshedSource = try await resolveRefreshedSource(
            activeSource: activeSource,
            currentPlaylist: playlist,
            preferOriginalURL: false
          )
          consecutiveRefreshFailures = 0
        } catch is CancellationError {
          throw CancellationError()
        } catch {
          if isEncryptedPlaylistFailure(error) { throw error }
          lastRefresh = Date()
          consecutiveRefreshFailures += 1
          if availableBeforeRefresh {
            log(
              "HLSライブ一覧の更新を一時保留し、取得済み区間を続行します: "
                + "\(error.localizedDescription)\n"
            )
            // A transient playlist failure must not interrupt media already
            // present in the current live window.
            continue
          }
          if consecutiveRefreshFailures <= 6 {
            let delay = min(5, 0.5 * pow(2, Double(consecutiveRefreshFailures - 1)))
            log(
              "HLSライブ一覧を再取得できません。\(String(format: "%.1f", delay))秒後に再試行します: "
                + "\(error.localizedDescription)\n"
            )
            try await sleep(seconds: delay)
            continue
          }
          throw ProductionError.invalidSource(
            "ライブプレイリストを更新できません: \(error.localizedDescription)"
          )
        }
        guard let refreshed = refreshedSource.hlsPlaylist else {
          throw ProductionError.invalidSource("ライブプレイリストを更新できません")
        }
        try mergeTimelineStarts(
          from: playlist,
          refreshed: refreshed,
          into: &timelineStarts,
          currentDuration: duration
        )
        activeSource = refreshedSource
        playlist = refreshed
        downloader.cancel()
        downloader = makeDownloader(for: activeSource)
        self.downloader = downloader
        lastRefresh = Date()
        duration = max(
          duration,
          maximumTimelineEnd(
            playlist: refreshed,
            timelineStarts: timelineStarts
          )
        )
        if let firstSequence = refreshed.segments.first?.sequence,
          nextMediaSequence < firstSequence
        {
          let jumpPosition = timelineStarts[firstSequence]
            ?? refreshed.segments[0].startSeconds
          try await resetForLiveWindowJump(
            to: jumpPosition,
            restorationWindow: &restorationWindow,
            restoredDirectory: restoredDirectory,
            duration: duration,
            emit: emit
          )
          log(
            "HLSライブ窓が先へ進んだため、復元キューを "
              + "\(formatDuration(jumpPosition)) へ追従しました\n"
          )
          nextMediaSequence = firstSequence
        }
      }

      guard
        let selectedMediaSegment = playlist.segments
          .filter({ $0.sequence >= nextMediaSequence })
          .min(by: { $0.sequence < $1.sequence })
      else {
        if playlist.isLive {
          try await sleep(seconds: min(refreshInterval, 0.5))
          continue
        }
        break
      }
      guard selectedMediaSegment.sequence < Int64.max else {
        throw ProductionError.invalidTimeline("segment sequenceが上限を超えています")
      }
      nextMediaSequence = selectedMediaSegment.sequence + 1
      var mediaSegment = selectedMediaSegment
      var timelineStart = timelineStarts[mediaSegment.sequence]
        ?? mediaSegment.startSeconds
      var timelineEnd = timelineStart + mediaSegment.duration
      guard timelineStart.isFinite, timelineEnd.isFinite,
        timelineStart >= 0, timelineEnd > timelineStart
      else {
        throw ProductionError.invalidTimeline("区間\(mediaSegment.sequence)")
      }
      duration = max(duration, timelineEnd)
      if timelineEnd <= startSeconds { continue }

      var localURL: URL?
      var downloadFailure: Error?
      for retryIndex in 0...3 {
        do {
          localURL = try await downloader.materialize(
            segment: mediaSegment,
            in: sessionDirectory
          )
          downloadFailure = nil
          break
        } catch is CancellationError {
          throw CancellationError()
        } catch {
          downloadFailure = error
          guard retryIndex < 3 else { break }
          log(
            "HLS区間\(mediaSegment.sequence)のURLを更新して再試行します: "
              + "\(error.localizedDescription)\n"
          )
          let refreshedSource: IPadResolvedMediaSource
          do {
            refreshedSource = try await resolveRefreshedSource(
              activeSource: activeSource,
              currentPlaylist: playlist,
              preferOriginalURL: true
            )
          } catch is CancellationError {
            throw CancellationError()
          } catch {
            if isEncryptedPlaylistFailure(error) { throw error }
            downloadFailure = error
            try await sleep(seconds: min(2, 0.5 * Double(retryIndex + 1)))
            continue
          }
          guard let refreshed = refreshedSource.hlsPlaylist else {
            downloadFailure = ProductionError.invalidSource(
              "HLS区間の更新後にメディアプレイリストがありません"
            )
            continue
          }
          if playlist.isLive {
            try mergeTimelineStarts(
              from: playlist,
              refreshed: refreshed,
              into: &timelineStarts,
              currentDuration: duration
            )
          } else {
            for segment in refreshed.segments {
              timelineStarts[segment.sequence] = segment.startSeconds
            }
          }
          activeSource = refreshedSource
          playlist = refreshed
          downloader.cancel()
          downloader = makeDownloader(for: activeSource)
          self.downloader = downloader

          if let replacement = replacementSegment(
            for: mediaSegment,
            in: refreshed
          ) {
            mediaSegment = replacement
            timelineStart = timelineStarts[replacement.sequence]
              ?? replacement.startSeconds
            timelineEnd = timelineStart + replacement.duration
            nextMediaSequence = replacement.sequence + 1
            continue
          }
          if refreshed.isLive,
            let firstSequence = refreshed.segments.first?.sequence,
            firstSequence > mediaSegment.sequence
          {
            let jumpPosition = timelineStarts[firstSequence]
              ?? refreshed.segments[0].startSeconds
            try await resetForLiveWindowJump(
              to: jumpPosition,
              restorationWindow: &restorationWindow,
              restoredDirectory: restoredDirectory,
              duration: duration,
              emit: emit
            )
            nextMediaSequence = firstSequence
            log(
              "期限切れ区間がライブ窓から外れたため、復元キューを "
                + "\(formatDuration(jumpPosition)) へ追従しました\n"
            )
            continue productionLoop
          }
          downloadFailure = ProductionError.invalidSource(
            "更新後のプレイリストに区間\(mediaSegment.sequence)がありません"
          )
          try await sleep(seconds: min(2, 0.5 * Double(retryIndex + 1)))
        }
      }
      guard let localURL else {
        throw ProductionError.invalidSource(
          "HLS区間\(mediaSegment.sequence)を取得できません: "
            + "\(downloadFailure?.localizedDescription ?? "不明なエラー")"
        )
      }
      try checkCancellation()
      let restorationSource = RestorationSource(
        mediaSegment: mediaSegment,
        timelineStart: timelineStart,
        localURL: localURL
      )

      if let previous = restorationWindow.last,
        !canShareWindow(previous, restorationSource)
      {
        try await flushWindow(
          restorationWindow,
          restoredDirectory: restoredDirectory,
          requestedStartSeconds: startSeconds,
          duration: duration,
          emit: emit
        )
        removeMaterializedSources(restorationWindow)
        restorationWindow.removeAll(keepingCapacity: true)
      }

      restorationWindow.append(restorationSource)
      if restorationWindow.count == 2 {
        try await restoreWindow(
          restorationWindow,
          coreIndex: 0,
          restoredDirectory: restoredDirectory,
          requestedStartSeconds: startSeconds,
          duration: duration,
          emit: emit
        )
      } else if restorationWindow.count == 3 {
        try await restoreWindow(
          restorationWindow,
          coreIndex: 1,
          restoredDirectory: restoredDirectory,
          requestedStartSeconds: startSeconds,
          duration: duration,
          emit: emit
        )
        let expired = restorationWindow.removeFirst()
        try? FileManager.default.removeItem(at: expired.localURL)
      }
    }

    try await flushWindow(
      restorationWindow,
      restoredDirectory: restoredDirectory,
      requestedStartSeconds: startSeconds,
      duration: duration,
      emit: emit
    )
    removeMaterializedSources(restorationWindow)
    restorationWindow.removeAll(keepingCapacity: false)
    try checkCancellation()
    emit(.progress(position: duration, duration: duration))
    emit(.ended(duration: duration))
    log("HLSリアルタイム復元が配信末尾へ到達しました\n")
  }

  /// May be called from a controller's nonisolated deinitializer. The actual
  /// Process and downloader mutation remains serialized on the main actor.
  nonisolated func cancel() {
    Task { @MainActor [self] in
      requestCancellation()
    }
  }

  /// Cancels outstanding network/worker work and returns only after `run`
  /// has unwound all of its temporary-file and process cleanup scopes.
  /// Existing synchronous callers may keep using `cancel()`; lifecycle owners
  /// that must remove the session directory can await this method first.
  nonisolated func cancelAndWait() async {
    await withCheckedContinuation { continuation in
      Task { @MainActor [self] in
        requestCancellation()
        if isRunActive {
          completionWaiters.append(continuation)
        } else {
          continuation.resume()
        }
      }
    }
  }

  private func requestCancellation() {
    guard !cancellationRequested else { return }
    cancellationRequested = true
    downloader?.cancel()
    sendWorkerCommand(["command": "stop"])
    activeProcess?.terminate()
  }

  private func completeRun() {
    isRunActive = false
    let waiters = completionWaiters
    completionWaiters.removeAll(keepingCapacity: false)
    for waiter in waiters { waiter.resume() }
  }

  private func makeDownloader(
    for resolvedSource: IPadResolvedMediaSource
  ) -> IPadHLSResourceDownloader {
    IPadHLSResourceDownloader(
      maximumResourceBytes: 256 * 1_024 * 1_024,
      maximumRedirectCount: 6,
      requestTimeout: 30,
      resolutionPolicy: resolvedSource.resolutionPolicy,
      requestContext: resolvedSource.requestContext ?? source.requestContext
    )
  }

  /// Re-resolves a media playlist while retaining the visible browser's
  /// cookies, Referer, Origin and User-Agent. A routine live refresh starts at
  /// the current media playlist. Recovery after a signed resource failure
  /// starts at the original/master URL so a new token can be issued.
  private func resolveRefreshedSource(
    activeSource: IPadResolvedMediaSource,
    currentPlaylist: IPadHLSMediaPlaylist,
    preferOriginalURL: Bool
  ) async throws -> IPadResolvedMediaSource {
    let candidates = refreshCandidates(
      activeSource: activeSource,
      currentPlaylist: currentPlaylist,
      preferOriginalURL: preferOriginalURL
    )
    var lastFailure: Error?
    for candidate in candidates {
      try checkCancellation()
      do {
        let resolved = try await IPadMediaURLResolver().resolve(
          candidate.absoluteString,
          policy: source.resolutionPolicy,
          context: activeSource.requestContext ?? source.requestContext
        )
        guard resolved.kind == .hls,
          let playlist = resolved.hlsPlaylist,
          !playlist.segments.isEmpty,
          playlist.isLive == currentPlaylist.isLive,
          isCredibleRefresh(playlist, replacing: currentPlaylist),
          !IPadBrowserMediaSourceSelector
            .isHighConfidenceAdvertisementSource(resolved)
        else {
          lastFailure = ProductionError.invalidSource(
            "更新URLが同じHLS配信を指していません"
          )
          continue
        }
        return resolved
      } catch is CancellationError {
        throw CancellationError()
      } catch let error as IPadMediaURLResolverError {
        // Never turn an encrypted/DRM rejection into a generic retry loop.
        if case .encryptedPlaylist = error { throw error }
        lastFailure = error
      } catch {
        lastFailure = error
      }
    }
    throw lastFailure
      ?? ProductionError.invalidSource("プレイリストの更新URLがありません")
  }

  private func refreshCandidates(
    activeSource: IPadResolvedMediaSource,
    currentPlaylist: IPadHLSMediaPlaylist,
    preferOriginalURL: Bool
  ) -> [URL] {
    var values: [URL] = []
    var seen: Set<String> = []
    func append(_ url: URL?) {
      guard let url,
        var components = URLComponents(url: url, resolvingAgainstBaseURL: true),
        let scheme = components.scheme?.lowercased(),
        scheme == "http" || scheme == "https",
        components.user == nil, components.password == nil,
        components.host?.isEmpty == false
      else { return }
      components.scheme = scheme
      components.fragment = nil
      guard let safeURL = components.url,
        seen.insert(safeURL.absoluteString).inserted
      else { return }
      values.append(safeURL)
    }

    let originalURLs = [
      source.playbackURL,
      source.submittedURL,
      activeSource.playbackURL,
      activeSource.submittedURL,
      source.requestContext?.referer,
      activeSource.requestContext?.referer,
    ]
    let currentURLs = [
      currentPlaylist.url,
      activeSource.mediaURL,
      source.mediaURL,
    ]
    if preferOriginalURL {
      originalURLs.forEach { append($0) }
      currentURLs.forEach { append($0) }
    } else {
      currentURLs.forEach { append($0) }
      originalURLs.forEach { append($0) }
    }
    return values
  }

  private func isCredibleRefresh(
    _ refreshed: IPadHLSMediaPlaylist,
    replacing current: IPadHLSMediaPlaylist
  ) -> Bool {
    guard !refreshed.segments.isEmpty,
      refreshed.isLive == current.isLive
    else { return false }
    if refreshed.isLive { return true }
    let currentDuration = current.duration
    let refreshedDuration = refreshed.duration
    guard currentDuration.isFinite, currentDuration > 0,
      refreshedDuration.isFinite, refreshedDuration > 0
    else { return false }
    let ratio = refreshedDuration / currentDuration
    return ratio >= 0.8 && ratio <= 1.25
  }

  private func isEncryptedPlaylistFailure(_ error: Error) -> Bool {
    guard let resolverError = error as? IPadMediaURLResolverError else {
      return false
    }
    if case .encryptedPlaylist = resolverError { return true }
    return false
  }

  private func replacementSegment(
    for previous: IPadHLSMediaSegment,
    in refreshed: IPadHLSMediaPlaylist
  ) -> IPadHLSMediaSegment? {
    if let exact = refreshed.segments.first(where: {
      $0.sequence == previous.sequence
    }) {
      return exact
    }
    guard !refreshed.isLive else { return nil }
    let tolerance = max(0.25, min(previous.duration, 2))
    return refreshed.segments.min(by: {
      abs($0.startSeconds - previous.startSeconds)
        < abs($1.startSeconds - previous.startSeconds)
    }).flatMap {
      abs($0.startSeconds - previous.startSeconds) <= tolerance ? $0 : nil
    }
  }

  private func resetForLiveWindowJump(
    to position: Double,
    restorationWindow: inout [RestorationSource],
    restoredDirectory: URL,
    duration: Double,
    emit: @escaping EventSink
  ) async throws {
    // The first retained source has already been emitted whenever the rolling
    // window contains two items. flushWindow restores only the last pending
    // core, so this does not duplicate an earlier output segment.
    try await flushWindow(
      restorationWindow,
      restoredDirectory: restoredDirectory,
      requestedStartSeconds: startSeconds,
      duration: duration,
      emit: emit
    )
    removeMaterializedSources(restorationWindow)
    restorationWindow.removeAll(keepingCapacity: true)
    guard position.isFinite, position >= 0 else {
      throw ProductionError.invalidTimeline("ライブ窓の移動位置が不正です")
    }
    emit(.discontinuity(position: position))
    emit(.progress(position: position, duration: max(duration, position)))
  }

  private func flushWindow(
    _ sources: [RestorationSource],
    restoredDirectory: URL,
    requestedStartSeconds: Double,
    duration: Double,
    emit: @escaping EventSink
  ) async throws {
    guard !sources.isEmpty else { return }
    try await restoreWindow(
      sources,
      coreIndex: sources.index(before: sources.endIndex),
      restoredDirectory: restoredDirectory,
      requestedStartSeconds: requestedStartSeconds,
      duration: duration,
      emit: emit
    )
  }

  private func restoreWindow(
    _ sources: [RestorationSource],
    coreIndex: Int,
    restoredDirectory: URL,
    requestedStartSeconds: Double,
    duration: Double,
    emit: @escaping EventSink
  ) async throws {
    try checkCancellation()
    guard sources.indices.contains(coreIndex), !sources.isEmpty else {
      throw ProductionError.invalidSource("連結区間の復元対象が不正です")
    }
    let coreSource = sources[coreIndex]
    let requestedTimelineStart = max(
      coreSource.timelineStart,
      requestedStartSeconds
    )
    guard requestedTimelineStart < coreSource.timelineEnd else { return }

    let assemblyURL = sessionDirectory.appendingPathComponent(
      "hls-window-\(nextOutputSequence)-\(UUID().uuidString.lowercased()).mp4",
      isDirectory: false
    )
    defer { try? FileManager.default.removeItem(at: assemblyURL) }

    var assembled: IPadHLSIntervalAssembler.Result
    var assembledCoreIndex: Int
    do {
      assembled = try await IPadHLSIntervalAssembler.concatenate(
        inputURLs: sources.map(\.localURL),
        outputURL: assemblyURL,
        temporaryDirectory: sessionDirectory
      )
      assembledCoreIndex = coreIndex
      guard assembled.sourceOffsets.indices.contains(assembledCoreIndex) else {
        throw ProductionError.invalidTimeline("連結区間の対応を取得できません")
      }
      try await IPadHLSIntervalAssembler.validateDecodableVideo(
        at: assemblyURL,
        near: assembled.sourceOffsets[assembledCoreIndex]
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      guard sources.count > 1 else { throw error }
      log(
        "HLS隣接区間の符号化条件が変化したため、対象区間だけで復元を続行します: "
          + "\(error.localizedDescription)\n"
      )
      try? FileManager.default.removeItem(at: assemblyURL)
      assembled = try await IPadHLSIntervalAssembler.concatenate(
        inputURLs: [coreSource.localURL],
        outputURL: assemblyURL,
        temporaryDirectory: sessionDirectory
      )
      assembledCoreIndex = 0
      guard let coreOffset = assembled.sourceOffsets.first else {
        throw ProductionError.invalidTimeline("単一区間の対応を取得できません")
      }
      try await IPadHLSIntervalAssembler.validateDecodableVideo(
        at: assemblyURL,
        near: coreOffset
      )
    }

    guard assembled.sourceOffsets.indices.contains(assembledCoreIndex),
      assembled.sourceDurations.indices.contains(assembledCoreIndex)
    else {
      throw ProductionError.invalidTimeline("連結区間の時間対応を取得できません")
    }
    let coreMediaStart = assembled.sourceOffsets[assembledCoreIndex]
    let coreMediaEnd = coreMediaStart
      + assembled.sourceDurations[assembledCoreIndex]
    let requestedOffset = max(
      0,
      requestedTimelineStart - coreSource.timelineStart
    )
    let effectiveCoreMediaStart = min(
      coreMediaEnd,
      coreMediaStart + requestedOffset
    )
    guard effectiveCoreMediaStart < coreMediaEnd else { return }

    let decodeStart = assembled.sourceOffsets.first ?? 0
    let decodeEnd = zip(
      assembled.sourceOffsets,
      assembled.sourceDurations
    ).map(+).max() ?? assembled.duration
    guard decodeStart.isFinite, decodeEnd.isFinite,
      coreMediaStart.isFinite, coreMediaEnd.isFinite,
      decodeStart >= 0, decodeEnd > decodeStart,
      coreMediaStart >= decodeStart, coreMediaEnd <= decodeEnd + 0.001
    else {
      throw ProductionError.invalidTimeline("連結区間の復元範囲が不正です")
    }

    try await runWorker(
      inputURL: assemblyURL,
      decodeStartSeconds: decodeStart,
      decodeEndSeconds: decodeEnd,
      coreMediaStartSeconds: effectiveCoreMediaStart,
      coreMediaEndSeconds: coreMediaEnd,
      coreTimelineStartSeconds: coreSource.timelineStart + requestedOffset,
      coreTimelineEndSeconds: coreSource.timelineEnd,
      restoredDirectory: restoredDirectory,
      duration: duration,
      emit: emit
    )
  }

  private func runWorker(
    inputURL: URL,
    decodeStartSeconds: Double,
    decodeEndSeconds: Double,
    coreMediaStartSeconds: Double,
    coreMediaEndSeconds: Double,
    coreTimelineStartSeconds: Double,
    coreTimelineEndSeconds: Double,
    restoredDirectory: URL,
    duration: Double,
    emit: @escaping EventSink
  ) async throws {
    try checkCancellation()
    let workerDirectory = sessionDirectory.appendingPathComponent(
      "worker-\(nextOutputSequence)-\(UUID().uuidString.lowercased())",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: workerDirectory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    defer { try? FileManager.default.removeItem(at: workerDirectory) }

    let decodeStartNanoseconds = try nanoseconds(decodeStartSeconds)
    let decodeEndNanoseconds = try nanoseconds(decodeEndSeconds)
    let coreStartNanoseconds = try nanoseconds(coreMediaStartSeconds)
    let coreEndNanoseconds = try nanoseconds(coreMediaEndSeconds)
    let invocation = try runner.nativePreviewInvocation(
      resources: resources,
      outputDirectory: workerDirectory,
      input: inputURL,
      startNanoseconds: decodeStartNanoseconds,
      generation: generation,
      decodeEndNanoseconds: decodeEndNanoseconds,
      outputCoreStartNanoseconds: coreStartNanoseconds,
      outputCoreEndNanoseconds: coreEndNanoseconds
    )
    let configurationURL = workerDirectory.appendingPathComponent(
      "native-preview-configuration.json",
      isDirectory: false
    )
    try invocation.configuration.write(to: configurationURL, options: .atomic)

    let process = Process()
    let inputPipe = Pipe()
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    process.executableURL = invocation.executable
    process.arguments = [configurationURL.path]
    process.environment = invocation.environment
    process.standardInput = inputPipe
    process.standardOutput = outputPipe
    process.standardError = errorPipe
    activeProcess = process
    activeWorkerInput = inputPipe

    do {
      try process.run()
    } catch {
      activeProcess = nil
      activeWorkerInput = nil
      throw ProductionError.worker(error.localizedDescription)
    }
    let errorTask = Task.detached(priority: .utility) {
      errorPipe.fileHandleForReading.readDataToEndOfFile()
    }

    var workerFailure: String?
    do {
      for try await line in outputPipe.fileHandleForReading.bytes.lines {
        try checkCancellation()
        guard let data = line.data(using: .utf8) else { continue }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let event: WorkerEvent
        do {
          event = try decoder.decode(WorkerEvent.self, from: data)
        } catch {
          log("HLS preview workerの応答を解析できません: \(line)\n")
          continue
        }
        guard event.generation == generation else { continue }
        switch event.kind {
        case "segment":
          guard let workerSequence = event.sequence,
            let startNs = event.startNs,
            let endNs = event.endNs,
            let path = event.path,
            endNs > startNs
          else {
            workerFailure = "復元区間の応答が不完全です"
            continue
          }
          let workerURL = URL(fileURLWithPath: path)
          let stableURL = restoredDirectory.appendingPathComponent(
            String(format: "hls-restored-%06d.mp4", nextOutputSequence),
            isDirectory: false
          )
          try? FileManager.default.removeItem(at: stableURL)
          do {
            try FileManager.default.copyItem(at: workerURL, to: stableURL)
          } catch {
            throw ProductionError.missingOutput(
              "\(workerURL.lastPathComponent): \(error.localizedDescription)"
            )
          }
          // Copy first, then release immediately. The native worker deletes
          // its rolling file on this acknowledgement and can otherwise block
          // once its output buffer reaches the configured limit.
          sendWorkerCommand([
            "command": "release_through",
            "sequence": workerSequence,
          ])

          let localStart = Double(startNs) / 1_000_000_000
          let localEnd = Double(endNs) / 1_000_000_000
          let mappedStart = max(
            coreTimelineStartSeconds,
            coreTimelineStartSeconds + localStart - coreMediaStartSeconds
          )
          let mappedEnd = min(
            coreTimelineEndSeconds,
            coreTimelineStartSeconds + localEnd - coreMediaStartSeconds
          )
          guard mappedEnd > mappedStart else {
            try? FileManager.default.removeItem(at: stableURL)
            continue
          }
          let outputSequence = nextOutputSequence
          nextOutputSequence += 1
          emit(
            .segment(
              sequence: outputSequence,
              startSeconds: mappedStart,
              endSeconds: mappedEnd,
              url: stableURL,
              codec: event.codec ?? "h264_videotoolbox"
            )
          )
          emit(.progress(position: mappedEnd, duration: max(duration, mappedEnd)))
        case "progress":
          guard let positionNs = event.positionNs else { continue }
          let localPosition = Double(positionNs) / 1_000_000_000
          let mapped = min(
            coreTimelineEndSeconds,
            max(
              coreTimelineStartSeconds,
              coreTimelineStartSeconds
                + localPosition - coreMediaStartSeconds
            )
          )
          emit(.progress(position: mapped, duration: max(duration, mapped)))
        case "error":
          workerFailure = [event.message, event.detail]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
            .joined(separator: ": ")
        default:
          break
        }
      }
    } catch {
      sendWorkerCommand(["command": "stop"])
      if process.isRunning { process.terminate() }
      process.waitUntilExit()
      activeProcess = nil
      activeWorkerInput = nil
      _ = await errorTask.value
      throw error
    }

    process.waitUntilExit()
    activeProcess = nil
    activeWorkerInput = nil
    let errorData = await errorTask.value
    let standardError = String(data: errorData, encoding: .utf8)?
      .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    try checkCancellation()
    if let workerFailure, !workerFailure.isEmpty {
      throw ProductionError.worker(workerFailure)
    }
    guard process.terminationStatus == 0 else {
      throw ProductionError.worker(
        standardError.isEmpty
          ? "workerが終了コード\(process.terminationStatus)を返しました"
          : standardError
      )
    }
    if !standardError.isEmpty {
      log(standardError + "\n")
    }
  }

  private func canShareWindow(
    _ previous: RestorationSource,
    _ next: RestorationSource
  ) -> Bool {
    guard previous.mediaSegment.sequence < Int64.max,
      next.mediaSegment.sequence == previous.mediaSegment.sequence + 1,
      next.mediaSegment.discontinuitySequence
        == previous.mediaSegment.discontinuitySequence,
      next.mediaSegment.initializationResource
        == previous.mediaSegment.initializationResource
    else { return false }
    return normalizedContainerExtension(previous.localURL)
      == normalizedContainerExtension(next.localURL)
  }

  private func normalizedContainerExtension(_ url: URL) -> String {
    let value = url.pathExtension.lowercased()
    return value == "m4s" ? "mp4" : value
  }

  private func startingSequence(
    playlist: IPadHLSMediaPlaylist,
    requestedStartSeconds: Double
  ) -> Int64 {
    let requested = playlist.segments.first {
      $0.startSeconds + $0.duration > requestedStartSeconds
    }
    guard playlist.isLive else {
      return requested?.sequence ?? playlist.segments[0].sequence
    }
    // Begin close enough to the live edge to build a three-segment temporal
    // window without first downloading the stale DVR portion.
    let liveEdge = playlist.segments.suffix(3).first ?? playlist.segments[0]
    return max(requested?.sequence ?? liveEdge.sequence, liveEdge.sequence)
  }

  private func mergeTimelineStarts(
    from previous: IPadHLSMediaPlaylist,
    refreshed: IPadHLSMediaPlaylist,
    into starts: inout [Int64: Double],
    currentDuration: Double
  ) throws {
    if let overlap = refreshed.segments.first(where: {
      starts[$0.sequence] != nil
    }), let knownStart = starts[overlap.sequence] {
      let adjustment = knownStart - overlap.startSeconds
      for segment in refreshed.segments {
        starts[segment.sequence] = segment.startSeconds + adjustment
      }
      return
    }

    let previousEnd = previous.segments.compactMap { segment in
      starts[segment.sequence].map { $0 + segment.duration }
    }.max() ?? currentDuration
    let previousLastSequence = previous.segments.map(\.sequence).max()
    let refreshedFirstSequence = refreshed.segments.map(\.sequence).min()
    let missingCount: Double
    if let previousLastSequence, let refreshedFirstSequence,
      refreshedFirstSequence > previousLastSequence
    {
      missingCount = max(
        0,
        Double(refreshedFirstSequence) - Double(previousLastSequence) - 1
      )
    } else {
      missingCount = 0
    }
    let knownDurations = previous.segments.map(\.duration)
      + refreshed.segments.map(\.duration)
    let estimatedDuration = previous.targetDuration
      ?? refreshed.targetDuration
      ?? (knownDurations.reduce(0, +) / Double(max(1, knownDurations.count)))
    let skippedDuration = missingCount * estimatedDuration
    guard previousEnd.isFinite, estimatedDuration.isFinite,
      skippedDuration.isFinite
    else {
      throw ProductionError.invalidTimeline("ライブ配信の時間軸が上限を超えています")
    }
    let base = previousEnd + skippedDuration
    guard let firstStart = refreshed.segments.first?.startSeconds else { return }
    for segment in refreshed.segments {
      starts[segment.sequence] = base + segment.startSeconds - firstStart
    }
  }

  private func maximumTimelineEnd(
    playlist: IPadHLSMediaPlaylist,
    timelineStarts: [Int64: Double]
  ) -> Double {
    playlist.segments.compactMap { segment in
      timelineStarts[segment.sequence].map { $0 + segment.duration }
    }.max() ?? max(0, playlist.duration)
  }

  private func removeMaterializedSources(_ sources: [RestorationSource]) {
    for source in sources {
      try? FileManager.default.removeItem(at: source.localURL)
    }
  }

  private func sendWorkerCommand(_ payload: [String: Any]) {
    guard let handle = activeWorkerInput?.fileHandleForWriting,
      let data = try? JSONSerialization.data(withJSONObject: payload)
    else { return }
    var line = data
    line.append(0x0A)
    try? handle.write(contentsOf: line)
  }

  private func checkCancellation() throws {
    try Task.checkCancellation()
    if cancellationRequested { throw CancellationError() }
  }

  private func sleep(seconds: Double) async throws {
    let nanoseconds = UInt64(max(0, seconds) * 1_000_000_000)
    try await Task.sleep(nanoseconds: nanoseconds)
    try checkCancellation()
  }

  private func nanoseconds(_ seconds: Double) throws -> Int64 {
    let maximumSeconds = Double(Int64.max) / 1_000_000_000
    guard seconds.isFinite, seconds >= 0, seconds < maximumSeconds else {
      throw ProductionError.invalidTimeline("秒数をnanosecondへ変換できません")
    }
    return Int64((seconds * 1_000_000_000).rounded())
  }

  private func formatDuration(_ seconds: Double) -> String {
    let whole = max(0, Int(seconds.rounded()))
    return String(format: "%d:%02d:%02d", whole / 3_600, (whole / 60) % 60, whole % 60)
  }
}
