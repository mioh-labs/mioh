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
/// HLS segments are restored with temporal context from their neighbours. Live
/// streams keep the old low-latency rolling 3-segment window; VOD streams batch
/// several core segments per worker invocation so playback behaves closer to a
/// local MP4 and avoids paying the worker/decode warm-up cost at every HLS cut.
@MainActor
final class MacHLSRealtimeProducer {
  typealias Logger = @Sendable (String) -> Void
  typealias EventSink = (MacHLSProductionEvent) -> Void

  private actor LocalSegmentCache {
    struct Materialized: Sendable {
      let url: URL
      let cacheHit: Bool
    }

    static let shared = LocalSegmentCache(maximumBytes: 1_024 * 1_024 * 1_024)

    private struct Entry {
      let url: URL
      let bytes: Int64
      var lastAccess: Date
      let pathExtension: String
    }

    private let maximumBytes: Int64
    private var entries: [String: Entry] = [:]
    private var totalBytes: Int64 = 0

    init(maximumBytes: Int64) {
      self.maximumBytes = max(64 * 1_024 * 1_024, maximumBytes)
    }

    func materialize(
      segment: IPadHLSMediaSegment,
      using downloader: IPadHLSResourceDownloader,
      in sessionDirectory: URL,
      cacheDirectory: URL
    ) async throws -> Materialized {
      let key = Self.cacheKey(for: segment)
      if let cached = try cachedCopy(
        for: key,
        sequence: segment.sequence,
        into: sessionDirectory
      ) {
        return Materialized(url: cached, cacheHit: true)
      }

      let downloadedURL = try await downloader.materialize(
        segment: segment,
        in: sessionDirectory
      )
      try importDownloadedSegment(
        downloadedURL,
        key: key,
        cacheDirectory: cacheDirectory
      )
      return Materialized(url: downloadedURL, cacheHit: false)
    }

    private func cachedCopy(
      for key: String,
      sequence: Int64,
      into sessionDirectory: URL
    ) throws -> URL? {
      guard var entry = entries[key] else { return nil }
      guard FileManager.default.fileExists(atPath: entry.url.path) else {
        totalBytes = max(0, totalBytes - entry.bytes)
        entries[key] = nil
        return nil
      }
      entry.lastAccess = Date()
      entries[key] = entry
      try FileManager.default.createDirectory(
        at: sessionDirectory,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
      )
      let outputURL = sessionDirectory.appendingPathComponent(
        "mioh-hls-cache-\(sequence)-\(UUID().uuidString.lowercased()).\(entry.pathExtension)",
        isDirectory: false
      )
      do {
        try FileManager.default.linkItem(at: entry.url, to: outputURL)
      } catch {
        try FileManager.default.copyItem(at: entry.url, to: outputURL)
      }
      return outputURL
    }

    private func importDownloadedSegment(
      _ downloadedURL: URL,
      key: String,
      cacheDirectory: URL
    ) throws {
      let values = try downloadedURL.resourceValues(forKeys: [.fileSizeKey])
      let bytes = Int64(max(0, values.fileSize ?? 0))
      guard bytes > 0, bytes <= maximumBytes else { return }
      try FileManager.default.createDirectory(
        at: cacheDirectory,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
      )
      if let existing = entries[key] {
        totalBytes = max(0, totalBytes - existing.bytes)
        try? FileManager.default.removeItem(at: existing.url)
      }
      let pathExtension = normalizedPathExtension(downloadedURL.pathExtension)
      let cacheURL = cacheDirectory.appendingPathComponent(
        "\(UUID().uuidString.lowercased()).\(pathExtension)",
        isDirectory: false
      )
      try FileManager.default.copyItem(at: downloadedURL, to: cacheURL)
      entries[key] = Entry(
        url: cacheURL,
        bytes: bytes,
        lastAccess: Date(),
        pathExtension: pathExtension
      )
      totalBytes += bytes
      try pruneIfNeeded(protecting: key)
    }

    private func pruneIfNeeded(protecting protectedKey: String) throws {
      guard totalBytes > maximumBytes else { return }
      let victims = entries
        .filter { $0.key != protectedKey }
        .sorted { $0.value.lastAccess < $1.value.lastAccess }
      for (key, entry) in victims {
        try? FileManager.default.removeItem(at: entry.url)
        entries[key] = nil
        totalBytes = max(0, totalBytes - entry.bytes)
        if totalBytes <= maximumBytes { break }
      }
    }

    private func normalizedPathExtension(_ value: String) -> String {
      let sanitized = value.lowercased().filter {
        $0.isLetter || $0.isNumber
      }
      return sanitized.isEmpty ? "mp4" : sanitized
    }

    private static func cacheKey(for segment: IPadHLSMediaSegment) -> String {
      [
        resourceKey(segment.resource),
        resourceKey(segment.initializationResource),
        "disc=\(segment.discontinuitySequence)",
      ].joined(separator: "|")
    }

    private static func resourceKey(_ resource: IPadHLSResource?) -> String {
      guard let resource else { return "nil" }
      let range: String
      if let byteRange = resource.byteRange {
        range = "\(byteRange.offset)-\(byteRange.length)"
      } else {
        range = "full"
      }
      return "\(resource.url.absoluteString)#\(range)"
    }
  }

  private struct RestorationSource: Sendable {
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
  private var prefetchDownloader: IPadHLSResourceDownloader?
  private var prefetchDownloaders: [IPadHLSResourceDownloader] = []
  private var activeProcess: Process?
  private var activeWorkerInput: Pipe?
  private var cancellationRequested = false
  private var nextOutputSequence = 0
  private var isRunActive = false
  private var completionWaiters: [CheckedContinuation<Void, Never>] = []
  private let vodRestoreBatchCoreSegments = 6

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
    let localSegmentCacheDirectory = resources.appendingPathComponent(
      "hls-segment-cache",
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
    var hasRestoredAnyWindow = false
    var vodPrefetchTasks: [Int64: Task<Void, Never>] = [:]
    var vodPrefetchResults: [Int64: Result<RestorationSource, Error>] = [:]
    var vodPrefetchGeneration = UUID()
    var vodPrefetchCompletionCount = 0
    var vodPrefetchDownloader: IPadHLSResourceDownloader?
    // Keep VOD prefetch independent from the restore window. Running downloads
    // and completed inventory are intentionally tracked separately; otherwise
    // completed segments keep occupying task slots until restoration consumes
    // them, which makes prefetch refill appear coupled to BasicVSR++ throughput.
    // Keep a bounded VOD inventory without hammering rate-limited hosts. The
    // restore side consumes multi-segment batches, but the network side should
    // stay polite; aggressive 64x6 bursts can trigger HTTP 429 on some HLS
    // origins after a seek.
    let maximumVODPrefetchSegments = 24
    let maximumVODPrefetchDownloads = 3
    func discardVODPrefetchResult(_ result: Result<RestorationSource, Error>) {
      if case let .success(source) = result {
        try? FileManager.default.removeItem(at: source.localURL)
      }
    }
    func cancelVODPrefetch(resetDownloader: Bool = false) {
      for task in vodPrefetchTasks.values { task.cancel() }
      vodPrefetchTasks.removeAll(keepingCapacity: true)
      for result in vodPrefetchResults.values {
        discardVODPrefetchResult(result)
      }
      vodPrefetchResults.removeAll(keepingCapacity: true)
      vodPrefetchGeneration = UUID()
      vodPrefetchCompletionCount = 0
      if resetDownloader {
        vodPrefetchDownloader?.cancel()
        if self.prefetchDownloader === vodPrefetchDownloader {
          self.prefetchDownloader = nil
        }
        vodPrefetchDownloader = nil
        self.prefetchDownloaders.removeAll(keepingCapacity: true)
      }
    }
    func ensureVODPrefetchDownloader() -> IPadHLSResourceDownloader {
      if let vodPrefetchDownloader { return vodPrefetchDownloader }
      let created = makeDownloader(for: activeSource)
      vodPrefetchDownloader = created
      self.prefetchDownloader = created
      self.prefetchDownloaders.append(created)
      return created
    }
    func startVODPrefetchIfPossible() {
      guard !playlist.isLive else { return }
      let scheduled = Set(vodPrefetchTasks.keys).union(vodPrefetchResults.keys)
      let availableDownloadSlots = max(
        0,
        maximumVODPrefetchDownloads - vodPrefetchTasks.count
      )
      let availableDepthSlots = max(
        0,
        maximumVODPrefetchSegments - scheduled.count
      )
      let availableSlots = min(availableDownloadSlots, availableDepthSlots)
      guard availableSlots > 0 else { return }
      let candidates = playlist.segments
        .filter {
          $0.sequence >= nextMediaSequence
            && $0.sequence < Int64.max
            && !scheduled.contains($0.sequence)
        }
        .sorted { $0.sequence < $1.sequence }
        .prefix(availableSlots)
      guard !candidates.isEmpty else { return }
      for candidate in candidates {
        let candidateStart = timelineStarts[candidate.sequence]
          ?? candidate.startSeconds
        let candidateEnd = candidateStart + candidate.duration
        guard candidateStart.isFinite, candidateEnd.isFinite,
          candidateEnd > startSeconds
        else { continue }
        let prefetchDownloader = ensureVODPrefetchDownloader()
        let prefetchDirectory = sessionDirectory
        let prefetchCacheDirectory = localSegmentCacheDirectory
        let generation = vodPrefetchGeneration
        if vodPrefetchTasks.isEmpty && vodPrefetchResults.isEmpty {
          log(
            "HLS先読み開始: 区間\(candidate.sequence)から最大\(maximumVODPrefetchSegments)本"
              + "（同時\(maximumVODPrefetchDownloads)本）\n"
          )
        }
        vodPrefetchTasks[candidate.sequence] = Task.detached(priority: .userInitiated) {
          let startedAt = Date()
          var completionLine: String?
          let result: Result<RestorationSource, Error>
          do {
            let materialized = try await LocalSegmentCache.shared.materialize(
              segment: candidate,
              using: prefetchDownloader,
              in: prefetchDirectory,
              cacheDirectory: prefetchCacheDirectory
            )
            let elapsed = Date().timeIntervalSince(startedAt)
            completionLine =
              "HLS先読み完了: 区間\(candidate.sequence) / "
                + (materialized.cacheHit ? "cache" : "network")
                + " / \(String(format: "%.2f", elapsed))秒"
            result = .success(
              RestorationSource(
                mediaSegment: candidate,
                timelineStart: candidateStart,
                localURL: materialized.url
              )
            )
          } catch {
            result = .failure(error)
          }

          let lineToEmit = completionLine
          await MainActor.run {
            guard vodPrefetchGeneration == generation,
              !Task.isCancelled
            else {
              discardVODPrefetchResult(result)
              return
            }
            vodPrefetchTasks[candidate.sequence] = nil
            vodPrefetchResults[candidate.sequence] = result
            vodPrefetchCompletionCount += 1
            if let lineToEmit,
              vodPrefetchCompletionCount <= maximumVODPrefetchDownloads
                || vodPrefetchCompletionCount.isMultiple(of: 10)
            {
              self.log(
                lineToEmit
                  + " / 在庫\(vodPrefetchResults.count)/\(maximumVODPrefetchSegments)\n"
              )
            }
            startVODPrefetchIfPossible()
          }
        }
      }
    }
    func consumeVODPrefetch(
      for sequence: Int64
    ) async throws -> RestorationSource? {
      guard !playlist.isLive else { return nil }
      if let prefetchTask = vodPrefetchTasks[sequence] {
        await prefetchTask.value
      }
      guard let result = vodPrefetchResults.removeValue(forKey: sequence)
      else { return nil }
      switch result {
      case let .success(prefetchedSource):
        guard prefetchedSource.mediaSegment.sequence == sequence else {
          try? FileManager.default.removeItem(at: prefetchedSource.localURL)
          return nil
        }
        log(
          "HLS先読みヒット: 区間\(sequence) / 復元待ち中に取得済み\n"
        )
        return prefetchedSource
      case let .failure(error):
        if error is CancellationError { throw CancellationError() }
        log(
          "HLS区間\(sequence)の先読みを使用できません。通常取得へ戻します: "
            + "\(error.localizedDescription)\n"
        )
        return nil
      }
    }
    defer {
      cancelVODPrefetch(resetDownloader: true)
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
        cancelVODPrefetch(resetDownloader: true)
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
            hasLeftContext: hasRestoredAnyWindow,
            restoredDirectory: restoredDirectory,
            duration: duration,
            emit: emit
          )
          hasRestoredAnyWindow = false
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
      if !playlist.isLive {
        for staleSequence in vodPrefetchTasks.keys where staleSequence < mediaSegment.sequence {
          vodPrefetchTasks.removeValue(forKey: staleSequence)?.cancel()
        }
        for staleSequence in vodPrefetchResults.keys where staleSequence < mediaSegment.sequence {
          if let staleResult = vodPrefetchResults.removeValue(forKey: staleSequence) {
            discardVODPrefetchResult(staleResult)
          }
        }
      }
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
      startVODPrefetchIfPossible()

      var localURL: URL?
      var downloadFailure: Error?
      let prefetchedSource = try await consumeVODPrefetch(
        for: mediaSegment.sequence
      )
      if let prefetchedSource {
        mediaSegment = prefetchedSource.mediaSegment
        timelineStart = prefetchedSource.timelineStart
        timelineEnd = prefetchedSource.timelineEnd
        localURL = prefetchedSource.localURL
      } else {
        for retryIndex in 0...3 {
          do {
            let materialized = try await LocalSegmentCache.shared.materialize(
              segment: mediaSegment,
              using: downloader,
              in: sessionDirectory,
              cacheDirectory: localSegmentCacheDirectory
            )
            localURL = materialized.url
            if materialized.cacheHit {
              log("HLS区間\(mediaSegment.sequence)をローカルキャッシュから再利用しました\n")
            }
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
            cancelVODPrefetch(resetDownloader: true)
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
                hasLeftContext: hasRestoredAnyWindow,
                restoredDirectory: restoredDirectory,
                duration: duration,
                emit: emit
              )
              hasRestoredAnyWindow = false
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
      startVODPrefetchIfPossible()

      if let previous = restorationWindow.last,
        !canShareWindow(previous, restorationSource)
      {
        try await flushWindow(
          restorationWindow,
          hasLeftContext: hasRestoredAnyWindow,
          restoredDirectory: restoredDirectory,
          requestedStartSeconds: startSeconds,
          duration: duration,
          emit: emit
        )
        removeMaterializedSources(restorationWindow)
        restorationWindow.removeAll(keepingCapacity: true)
        hasRestoredAnyWindow = false
      }

      restorationWindow.append(restorationSource)
      if playlist.isLive {
        if restorationWindow.count == 2 {
          try await restoreWindow(
            restorationWindow,
            coreIndex: 0,
            restoredDirectory: restoredDirectory,
            requestedStartSeconds: startSeconds,
            duration: duration,
            emit: emit
          )
          hasRestoredAnyWindow = true
        } else if restorationWindow.count == 3 {
          try await restoreWindow(
            restorationWindow,
            coreIndex: 1,
            restoredDirectory: restoredDirectory,
            requestedStartSeconds: startSeconds,
            duration: duration,
            emit: emit
          )
          hasRestoredAnyWindow = true
          let expired = restorationWindow.removeFirst()
          try? FileManager.default.removeItem(at: expired.localURL)
        }
      } else {
        while true {
          let coreStartIndex = hasRestoredAnyWindow ? 1 : 0
          let requiredCount = coreStartIndex + vodRestoreBatchCoreSegments + 1
          guard restorationWindow.count >= requiredCount else { break }
          let coreEndIndex = coreStartIndex + vodRestoreBatchCoreSegments - 1
          try await restoreWindow(
            restorationWindow,
            coreStartIndex: coreStartIndex,
            coreEndIndex: coreEndIndex,
            restoredDirectory: restoredDirectory,
            requestedStartSeconds: startSeconds,
            duration: duration,
            emit: emit
          )
          let retired = Array(restorationWindow.prefix(vodRestoreBatchCoreSegments))
          removeMaterializedSources(retired)
          restorationWindow.removeFirst(vodRestoreBatchCoreSegments)
          hasRestoredAnyWindow = true
        }
      }
    }

    try await flushWindow(
      restorationWindow,
      hasLeftContext: hasRestoredAnyWindow,
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
    prefetchDownloader?.cancel()
    for downloader in prefetchDownloaders { downloader.cancel() }
    prefetchDownloaders.removeAll(keepingCapacity: true)
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
    hasLeftContext: Bool,
    restoredDirectory: URL,
    duration: Double,
    emit: @escaping EventSink
  ) async throws {
    // When a retained left-context source was already emitted, flush only the
    // not-yet-emitted cores before discarding the live window.
    try await flushWindow(
      restorationWindow,
      hasLeftContext: hasLeftContext,
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
    hasLeftContext: Bool,
    restoredDirectory: URL,
    requestedStartSeconds: Double,
    duration: Double,
    emit: @escaping EventSink
  ) async throws {
    guard !sources.isEmpty else { return }
    let coreStartIndex = hasLeftContext && sources.count > 1 ? 1 : 0
    guard sources.indices.contains(coreStartIndex) else { return }
    try await restoreWindow(
      sources,
      coreStartIndex: coreStartIndex,
      coreEndIndex: sources.index(before: sources.endIndex),
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
    try await restoreWindow(
      sources,
      coreStartIndex: coreIndex,
      coreEndIndex: coreIndex,
      restoredDirectory: restoredDirectory,
      requestedStartSeconds: requestedStartSeconds,
      duration: duration,
      emit: emit
    )
  }

  private func restoreWindow(
    _ sources: [RestorationSource],
    coreStartIndex: Int,
    coreEndIndex: Int,
    restoredDirectory: URL,
    requestedStartSeconds: Double,
    duration: Double,
    emit: @escaping EventSink
  ) async throws {
    try checkCancellation()
    guard sources.indices.contains(coreStartIndex),
      sources.indices.contains(coreEndIndex),
      coreStartIndex <= coreEndIndex,
      !sources.isEmpty
    else {
      throw ProductionError.invalidSource("連結区間の復元対象が不正です")
    }
    let coreSource = sources[coreStartIndex]
    let coreEndSource = sources[coreEndIndex]
    let requestedTimelineStart = max(
      coreSource.timelineStart,
      requestedStartSeconds
    )
    guard requestedTimelineStart < coreEndSource.timelineEnd else { return }

    let assemblyURL = sessionDirectory.appendingPathComponent(
      "hls-window-\(nextOutputSequence)-\(UUID().uuidString.lowercased()).mp4",
      isDirectory: false
    )
    defer { try? FileManager.default.removeItem(at: assemblyURL) }

    let assemblyStartedAt = Date()
    var assembled: IPadHLSIntervalAssembler.Result
    var assembledCoreStartIndex: Int
    var assembledCoreEndIndex: Int
    do {
      assembled = try await IPadHLSIntervalAssembler.concatenate(
        inputURLs: sources.map(\.localURL),
        outputURL: assemblyURL,
        temporaryDirectory: sessionDirectory
      )
      assembledCoreStartIndex = coreStartIndex
      assembledCoreEndIndex = coreEndIndex
      guard assembled.sourceOffsets.indices.contains(assembledCoreStartIndex),
        assembled.sourceOffsets.indices.contains(assembledCoreEndIndex),
        assembled.sourceDurations.indices.contains(assembledCoreEndIndex)
      else {
        throw ProductionError.invalidTimeline("連結区間の対応を取得できません")
      }
      try await IPadHLSIntervalAssembler.validateDecodableVideo(
        at: assemblyURL,
        near: assembled.sourceOffsets[assembledCoreStartIndex]
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      guard coreStartIndex == coreEndIndex else { throw error }
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
      assembledCoreStartIndex = 0
      assembledCoreEndIndex = 0
      guard let coreOffset = assembled.sourceOffsets.first else {
        throw ProductionError.invalidTimeline("単一区間の対応を取得できません")
      }
      try await IPadHLSIntervalAssembler.validateDecodableVideo(
        at: assemblyURL,
        near: coreOffset
      )
    }
    let assemblyElapsed = Date().timeIntervalSince(assemblyStartedAt)

    guard assembled.sourceOffsets.indices.contains(assembledCoreStartIndex),
      assembled.sourceOffsets.indices.contains(assembledCoreEndIndex),
      assembled.sourceDurations.indices.contains(assembledCoreEndIndex)
    else {
      throw ProductionError.invalidTimeline("連結区間の時間対応を取得できません")
    }
    let coreMediaStart = assembled.sourceOffsets[assembledCoreStartIndex]
    let coreMediaEnd = assembled.sourceOffsets[assembledCoreEndIndex]
      + assembled.sourceDurations[assembledCoreEndIndex]
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
    if nextOutputSequence < 6 || nextOutputSequence.isMultiple(of: 10) {
      log(
        "HLS区間準備: 出力\(nextOutputSequence) / 入力\(sources.count)本 / "
          + "連結+検証\(String(format: "%.2f", assemblyElapsed))秒 / "
          + "復元対象\(String(format: "%.2f", coreMediaEnd - effectiveCoreMediaStart))秒 / "
          + "デコード窓\(String(format: "%.2f", decodeEnd - decodeStart))秒\n"
      )
    }

    try await runWorker(
      inputURL: assemblyURL,
      decodeStartSeconds: decodeStart,
      decodeEndSeconds: decodeEnd,
      coreMediaStartSeconds: effectiveCoreMediaStart,
      coreMediaEndSeconds: coreMediaEnd,
      coreTimelineStartSeconds: coreSource.timelineStart + requestedOffset,
      coreTimelineEndSeconds: coreEndSource.timelineEnd,
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
    let workerOutputSequenceStart = nextOutputSequence
    let workerDirectory = sessionDirectory.appendingPathComponent(
      "worker-\(workerOutputSequenceStart)-\(UUID().uuidString.lowercased())",
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

    let workerStartedAt = Date()
    var firstSegmentElapsed: TimeInterval?
    var emittedSegmentCount = 0
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
          if firstSegmentElapsed == nil {
            firstSegmentElapsed = Date().timeIntervalSince(workerStartedAt)
          }
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
          emittedSegmentCount += 1
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
    let workerElapsed = Date().timeIntervalSince(workerStartedAt)
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
    if workerOutputSequenceStart < 6
      || workerOutputSequenceStart.isMultiple(of: 10)
      || emittedSegmentCount > 1
    {
      log(
        "HLS復元worker: 出力\(workerOutputSequenceStart)"
          + "...\(max(workerOutputSequenceStart, nextOutputSequence - 1))"
          + " / \(emittedSegmentCount)本 / 初回出力"
          + "\(String(format: "%.2f", firstSegmentElapsed ?? workerElapsed))秒"
          + " / 全体\(String(format: "%.2f", workerElapsed))秒\n"
      )
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
