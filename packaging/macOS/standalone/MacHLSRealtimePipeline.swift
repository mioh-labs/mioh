import Darwin
import CryptoKit
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

  /// Bounds files already emitted to the playback queue but not yet consumed.
  /// `MacHLSProductionEvent.segment.sequence` is the credit token; the consumer
  /// must return it with `acknowledgeOutputConsumed(through:)` after AVPlayer has
  /// advanced past that segment.
  struct OutputBufferLimits: Sendable, Equatable {
    let seconds: Double
    let items: Int
    let bytes: Int64

    static let playbackDefault = OutputBufferLimits(
      seconds: 60,
      items: 48,
      bytes: 512 * 1_024 * 1_024
    )

    init(seconds: Double, items: Int, bytes: Int64) {
      self.seconds = max(2, seconds.isFinite ? seconds : 60)
      self.items = max(2, items)
      self.bytes = max(16 * 1_024 * 1_024, bytes)
    }
  }

  private struct RetainedOutputCredit {
    let seconds: Double
    let bytes: Int64
  }

  /// MPEG-TS parsing and AVMutableComposition export can run for hundreds of
  /// milliseconds. Isolate them from this producer's MainActor state so browser
  /// and playback controls remain responsive while a large VOD window is built.
  private actor IntervalAssemblyWorker {
    func concatenate(
      inputURLs: [URL],
      outputURL: URL,
      temporaryDirectory: URL
    ) async throws -> IPadHLSIntervalAssembler.Result {
      try await IPadHLSIntervalAssembler.concatenate(
        inputURLs: inputURLs,
        outputURL: outputURL,
        temporaryDirectory: temporaryDirectory
      )
    }

    func validateDecodableVideo(at url: URL, near seconds: TimeInterval) async throws {
      try await IPadHLSIntervalAssembler.validateDecodableVideo(at: url, near: seconds)
    }
  }

  private actor MediaFileWorker {
    func byteCount(at url: URL) throws -> Int64 {
      let values = try url.resourceValues(forKeys: [
        .isRegularFileKey,
        .fileSizeKey,
      ])
      guard values.isRegularFile == true, let fileSize = values.fileSize,
        fileSize > 0
      else {
        throw CocoaError(.fileReadUnknown)
      }
      return Int64(fileSize)
    }

    func copyReplacing(from source: URL, to destination: URL) throws {
      try? FileManager.default.removeItem(at: destination)
      do {
        // Worker and playback directories normally share the session volume.
        // A hard link lets the worker release its name without rewriting the
        // finalized MP4; copy remains the cross-volume/filesystem fallback.
        try FileManager.default.linkItem(at: source, to: destination)
      } catch {
        try FileManager.default.copyItem(at: source, to: destination)
      }
    }
  }

  private struct SendableProcess: @unchecked Sendable {
    let process: Process
  }

  private actor LocalSegmentCache {
    struct Materialized: Sendable {
      let url: URL
      let cacheHit: Bool
    }

    static let shared = LocalSegmentCache(
      maximumBytes: 1_024 * 1_024 * 1_024,
      timeToLive: 7 * 24 * 60 * 60
    )

    private struct Manifest: Codable {
      let version: Int
      let entries: [ManifestEntry]
    }

    private struct ManifestEntry: Codable {
      let key: String
      let fileName: String
      let bytes: Int64
      let lastAccess: Date
      let pathExtension: String
    }

    private struct Entry {
      let url: URL
      let bytes: Int64
      var lastAccess: Date
      let pathExtension: String
    }

    private let maximumBytes: Int64
    private let timeToLive: TimeInterval
    private var entries: [String: Entry] = [:]
    private var totalBytes: Int64 = 0
    private var loadedDirectory: URL?

    private let manifestName = "index-v2.json"

    init(maximumBytes: Int64, timeToLive: TimeInterval) {
      self.maximumBytes = max(64 * 1_024 * 1_024, maximumBytes)
      self.timeToLive = max(60 * 60, timeToLive)
    }

    func materialize(
      segment: IPadHLSMediaSegment,
      using downloader: IPadHLSResourceDownloader,
      in sessionDirectory: URL,
      cacheDirectory: URL,
      allowPersistentCache: Bool,
      priority: IPadSharedHTTPTransportOptions.Priority = .normal
    ) async throws -> Materialized {
      // Browser credential contexts can make identical stable URLs return
      // account-, cookie-, Referer-, or token-specific bytes. Never persist or
      // replay those bytes across sessions. The bounded in-generation prefetch
      // inventory still avoids duplicate downloads; persistent reuse remains
      // available for genuinely public context-free HLS resources.
      guard allowPersistentCache, Self.isPubliclyCacheable(segment) else {
        let downloadedURL = try await downloader.materialize(
          segment: segment,
          in: sessionDirectory,
          priority: priority
        )
        return Materialized(url: downloadedURL, cacheHit: false)
      }
      let key = Self.cacheKey(for: segment)
      var cacheIsAvailable = false
      do {
        try loadIfNeeded(from: cacheDirectory)
        cacheIsAvailable = true
        if let cached = try cachedCopy(
          for: key,
          sequence: segment.sequence,
          into: sessionDirectory,
          cacheDirectory: cacheDirectory
        ) {
          return Materialized(url: cached, cacheHit: true)
        }
      } catch {
        // Cache metadata or filesystem permissions must never make an otherwise
        // playable network segment fail. A later producer generation can retry.
        resetLoadedState()
      }

      let downloadedURL = try await downloader.materialize(
        segment: segment,
        in: sessionDirectory,
        priority: priority
      )
      if cacheIsAvailable {
        // The cache is an optimization. Disk pressure or a damaged index must
        // not poison the active HLS session after the network request succeeded.
        try? importDownloadedSegment(
          downloadedURL,
          key: key,
          cacheDirectory: cacheDirectory
        )
      }
      return Materialized(url: downloadedURL, cacheHit: false)
    }

    private func cachedCopy(
      for key: String,
      sequence: Int64,
      into sessionDirectory: URL,
      cacheDirectory: URL
    ) throws -> URL? {
      guard var entry = entries[key] else { return nil }
      let now = Date()
      guard now.timeIntervalSince(entry.lastAccess) <= timeToLive else {
        totalBytes = max(0, totalBytes - entry.bytes)
        entries[key] = nil
        try? FileManager.default.removeItem(at: entry.url)
        try? persistManifest(in: cacheDirectory)
        return nil
      }
      guard FileManager.default.fileExists(atPath: entry.url.path) else {
        totalBytes = max(0, totalBytes - entry.bytes)
        entries[key] = nil
        try? persistManifest(in: cacheDirectory)
        return nil
      }
      entry.lastAccess = now
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
      // The linked/copied media is already usable. Losing only the access-time
      // update must not turn a cache hit into a duplicate network request.
      try? persistManifest(in: cacheDirectory)
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
      let pathExtension = normalizedPathExtension(downloadedURL.pathExtension)
      let cacheURL = cacheDirectory.appendingPathComponent(
        "\(UUID().uuidString.lowercased()).\(pathExtension)",
        isDirectory: false
      )
      do {
        // Session files and the user cache normally share a volume. A hard link
        // avoids rewriting every HLS byte; copy remains the cross-volume fallback.
        try FileManager.default.linkItem(at: downloadedURL, to: cacheURL)
      } catch {
        try FileManager.default.copyItem(at: downloadedURL, to: cacheURL)
      }
      if let existing = entries[key] {
        totalBytes = max(0, totalBytes - existing.bytes)
        try? FileManager.default.removeItem(at: existing.url)
      }
      entries[key] = Entry(
        url: cacheURL,
        bytes: bytes,
        lastAccess: Date(),
        pathExtension: pathExtension
      )
      totalBytes += bytes
      pruneExpiredAndOversized(protecting: key, now: Date())
      try persistManifest(in: cacheDirectory)
    }

    private func pruneExpiredAndOversized(
      protecting protectedKey: String? = nil,
      now: Date
    ) {
      let expired = entries.filter { key, entry in
        key != protectedKey && now.timeIntervalSince(entry.lastAccess) > timeToLive
      }
      for (key, entry) in expired {
        try? FileManager.default.removeItem(at: entry.url)
        entries[key] = nil
        totalBytes = max(0, totalBytes - entry.bytes)
      }
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

    private func loadIfNeeded(from cacheDirectory: URL) throws {
      let standardized = cacheDirectory.standardizedFileURL
      if loadedDirectory == standardized { return }
      resetLoadedState()
      try FileManager.default.createDirectory(
        at: standardized,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
      )

      let manifestURL = standardized.appendingPathComponent(manifestName)
      let manifest = (try? Data(contentsOf: manifestURL)).flatMap {
        try? JSONDecoder().decode(Manifest.self, from: $0)
      }
      let now = Date()
      if manifest?.version == 2 {
        for stored in manifest?.entries ?? [] {
          guard Self.isSafeCacheFileName(stored.fileName) else { continue }
          let url = standardized.appendingPathComponent(stored.fileName)
          guard let values = try? url.resourceValues(
            forKeys: [.isRegularFileKey, .fileSizeKey]
          ), values.isRegularFile == true,
            let fileSize = values.fileSize,
            fileSize > 0,
            Int64(fileSize) == stored.bytes,
            now.timeIntervalSince(stored.lastAccess) <= timeToLive
          else {
            try? FileManager.default.removeItem(at: url)
            continue
          }
          // A corrupt manifest must not double-count two files with the same
          // logical resource key and thereby evict healthy cache entries.
          guard entries[stored.key] == nil else {
            continue
          }
          entries[stored.key] = Entry(
            url: url,
            bytes: stored.bytes,
            lastAccess: stored.lastAccess,
            pathExtension: normalizedPathExtension(stored.pathExtension)
          )
          totalBytes += stored.bytes
        }
      }

      // Scan every launch. Files absent from a valid manifest are incomplete
      // imports or leftovers from older builds and cannot be addressed safely.
      let referenced = Set(entries.values.map { $0.url.lastPathComponent })
      let contents = try FileManager.default.contentsOfDirectory(
        at: standardized,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
      )
      for url in contents where url.lastPathComponent != manifestName {
        guard !referenced.contains(url.lastPathComponent) else { continue }
        try? FileManager.default.removeItem(at: url)
      }
      loadedDirectory = standardized
      pruneExpiredAndOversized(now: now)
      try persistManifest(in: standardized)
    }

    private func persistManifest(in cacheDirectory: URL) throws {
      let manifest = Manifest(
        version: 2,
        entries: entries.map { key, entry in
          ManifestEntry(
            key: key,
            fileName: entry.url.lastPathComponent,
            bytes: entry.bytes,
            lastAccess: entry.lastAccess,
            pathExtension: entry.pathExtension
          )
        }
      )
      let data = try JSONEncoder().encode(manifest)
      try data.write(
        to: cacheDirectory.appendingPathComponent(manifestName),
        options: .atomic
      )
    }

    private func resetLoadedState() {
      entries.removeAll(keepingCapacity: false)
      totalBytes = 0
      loadedDirectory = nil
    }

    private static func isSafeCacheFileName(_ value: String) -> Bool {
      !value.isEmpty
        && value != "."
        && value != ".."
        && (value as NSString).lastPathComponent == value
    }

    private func normalizedPathExtension(_ value: String) -> String {
      let sanitized = value.lowercased().filter {
        $0.isLetter || $0.isNumber
      }
      return sanitized.isEmpty ? "mp4" : sanitized
    }

    private static func cacheKey(for segment: IPadHLSMediaSegment) -> String {
      let identity = [
        resourceKey(segment.resource),
        resourceKey(segment.initializationResource),
        "disc=\(segment.discontinuitySequence)",
      ].joined(separator: "|")
      // The manifest is user-private, but it should still not become a browsed
      // media history containing origin paths. Persist only a deterministic
      // digest; raw resource URLs remain in memory for the active request.
      return SHA256.hash(data: Data(identity.utf8))
        .map { String(format: "%02x", $0) }
        .joined()
    }

    private static func isPubliclyCacheable(
      _ segment: IPadHLSMediaSegment
    ) -> Bool {
      // Query-bearing media and init URLs commonly contain expiring bearer
      // signatures. Treat them as credentials even without a WebKit context;
      // replaying their bytes after token expiry would cross an authorization
      // boundary and would also leave the signed URL in the disk manifest.
      [segment.resource, segment.initializationResource]
        .compactMap { $0 }
        .allSatisfy { resource in
          guard let components = URLComponents(
            url: resource.url,
            resolvingAgainstBaseURL: true
          ) else { return false }
          return components.user == nil
            && components.password == nil
            && (components.percentEncodedQuery?.isEmpty ?? true)
        }
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
    case variantFallbackPrepared

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
      case .variantFallbackPrepared:
        return "HTTP 429を回避するため、低いHLS variantへの切り替えを準備しました。"
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
  private let resourceLoader: (any IPadHLSResourceLoading)?
  private let avFoundationCapture: MacHLSAVFoundationCapture?
  private let allowsVariantFallback: Bool

  private var downloader: IPadHLSResourceDownloader?
  private var prefetchDownloader: IPadHLSResourceDownloader?
  private var prefetchDownloaders: [IPadHLSResourceDownloader] = []
  private var activeProcess: Process?
  private var activeWorkerInput: Pipe?
  private var activeWorkerOutput: Pipe?
  private var activeWorkerError: Pipe?
  private var activeProcessRetirementTask: Task<Void, Never>?
  private var cancellationRequested = false
  private var nextOutputSequence = 0
  private var isRunActive = false
  private var completionWaiters: [CheckedContinuation<Void, Never>] = []
  private var outputBufferLimits = OutputBufferLimits.playbackDefault
  private var retainedOutputCredits: [Int: RetainedOutputCredit] = [:]
  private var retainedOutputSeconds = 0.0
  private var retainedOutputBytes: Int64 = 0
  private var outputCreditWaiters: [CheckedContinuation<Void, Never>] = []
  private var pendingVariantFallbackSource: IPadResolvedMediaSource?
  private var sameOriginVariantFallbackRejected = false
  private let intervalAssemblyWorker = IntervalAssemblyWorker()
  private let mediaFileWorker = MediaFileWorker()
  private let vodInitialRestoreBatchCoreSegments = 2
  private let vodMinimumSteadyBatchCoreSegments = 2
  private let vodSteadyRestoreBatchCoreSegments = 18
  private let vodSteadyRestoreBatchTargetSeconds = 36.0
  private let vodSteadyRestoreBatchMaximumBytes: Int64 = 384 * 1_024 * 1_024
  // AVFoundation capture arrives sequentially at 2x, unlike the fast path's
  // already-prefetched VOD inventory. Four steady cores amortize model/process
  // startup without waiting so long that the initial restored queue drains.
  private let avFoundationSteadyRestoreBatchCoreSegments = 4

  init(
    source: IPadResolvedMediaSource,
    runner: RestorationRunner,
    resources: URL,
    sessionDirectory: URL,
    startSeconds: Double,
    generation: Int,
    resourceLoader: (any IPadHLSResourceLoading)? = nil,
    avFoundationCapture: MacHLSAVFoundationCapture? = nil,
    allowsVariantFallback: Bool = true,
    log: @escaping Logger
  ) {
    self.source = source
    self.runner = runner
    self.resources = resources
    self.sessionDirectory = sessionDirectory
    self.startSeconds = max(0, startSeconds.isFinite ? startSeconds : 0)
    self.generation = generation
    self.resourceLoader = resourceLoader
    self.avFoundationCapture = avFoundationCapture
    self.allowsVariantFallback = allowsVariantFallback
    self.log = log
  }

  func updateOutputBufferLimits(_ limits: OutputBufferLimits) {
    outputBufferLimits = limits
    resumeOutputCreditWaiters()
  }

  /// Returns every output credit up to the last segment AVPlayer consumed.
  /// File deletion remains the controller's responsibility because it owns the
  /// AVPlayer item lifetime; this acknowledgement only resumes production.
  func acknowledgeOutputConsumed(through sequence: Int) {
    guard sequence >= 0 else { return }
    let released = retainedOutputCredits.filter { $0.key <= sequence }
    guard !released.isEmpty else { return }
    for (key, credit) in released {
      retainedOutputCredits[key] = nil
      retainedOutputSeconds = max(0, retainedOutputSeconds - credit.seconds)
      retainedOutputBytes = max(0, retainedOutputBytes - credit.bytes)
    }
    resumeOutputCreditWaiters()
  }

  /// Returns a lower-bandwidth rendition prepared after the active VOD origin
  /// rejected its current segment with HTTP 429. Taking consumes the handoff so
  /// a stale producer cannot restart the same fallback more than once.
  func takePendingVariantFallbackSource() -> IPadResolvedMediaSource? {
    let pending = pendingVariantFallbackSource
    pendingVariantFallbackSource = nil
    return pending
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

    if let avFoundationCapture {
      let timelineStarts = Dictionary(
        playlist.segments.map { ($0.sequence, $0.startSeconds) },
        uniquingKeysWith: { first, _ in first }
      )
      var duration = maximumTimelineEnd(
        playlist: playlist,
        timelineStarts: timelineStarts
      )
      if !playlist.isLive { duration = max(duration, playlist.duration) }
      try await runAVFoundationCapture(
        avFoundationCapture,
        playlist: playlist,
        restoredDirectory: restoredDirectory,
        duration: duration,
        emit: emit
      )
      return
    }

    let localSegmentCacheDirectory = try Self.userSegmentCacheDirectory()

    var activeSource = source
    var downloader = makeDownloader(for: activeSource)
    self.downloader = downloader
    defer {
      downloader.cancel()
      if self.downloader === downloader { self.downloader = nil }
      activeWorkerInput = nil
      activeWorkerOutput = nil
      activeWorkerError = nil
      activeProcess = nil
      activeProcessRetirementTask = nil
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
    var consecutiveSegmentRateLimits = 0
    var restorationWindow: [RestorationSource] = []
    var hasRestoredAnyWindow = false
    var vodPrefetchTasks: [Int64: Task<Void, Never>] = [:]
    var vodPrefetchResults: [Int64: Result<RestorationSource, Error>] = [:]
    var vodPrefetchGeneration = UUID()
    var vodPrefetchCompletionCount = 0
    var vodPrefetchDownloader: IPadHLSResourceDownloader?
    var hasMaterializedCurrentVODSegment = false
    var vodPrefetchSuspendedUntil = Date.distantPast
    var vodPrefetchPauseInProgress = false
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
    func cancelVODPrefetchAndWait(
      resetDownloader: Bool = false,
      discardCompletedResults: Bool = true
    ) async {
      // Invalidate callbacks before cancellation. Each detached task checks this
      // generation on MainActor before publishing its file into the inventory.
      vodPrefetchGeneration = UUID()
      let tasks = Array(vodPrefetchTasks.values)
      vodPrefetchTasks.removeAll(keepingCapacity: true)
      for task in tasks { task.cancel() }
      if discardCompletedResults {
        for result in vodPrefetchResults.values {
          discardVODPrefetchResult(result)
        }
        vodPrefetchResults.removeAll(keepingCapacity: true)
      }
      vodPrefetchCompletionCount = 0
      if resetDownloader {
        vodPrefetchDownloader?.cancel()
        for downloader in self.prefetchDownloaders { downloader.cancel() }
      }
      // Do not let a retired seek generation recreate files below its session
      // directory after its owner removes that tree. Await every detached task
      // while the MainActor remains suspended so invalidated callbacks can run.
      for task in tasks { await task.value }
      if resetDownloader {
        if self.prefetchDownloader === vodPrefetchDownloader {
          self.prefetchDownloader = nil
        }
        vodPrefetchDownloader = nil
        self.prefetchDownloaders.removeAll(keepingCapacity: true)
      }
    }

    func suspendVODPrefetchAfterRateLimit(
      seconds: TimeInterval,
      triggeredByPrefetch: Bool = false,
      reason: String = "HTTP 429"
    ) async {
      let hadPrefetchActivity =
        triggeredByPrefetch || !vodPrefetchTasks.isEmpty
      let boundedPause = max(1, min(12, seconds))
      let previousSuspension = vodPrefetchSuspendedUntil
      vodPrefetchSuspendedUntil = max(
        vodPrefetchSuspendedUntil,
        Date().addingTimeInterval(boundedPause)
      )
      // Another 429 callback may belong to a task the first suspension is
      // already cancelling and awaiting. It must return immediately here;
      // waiting for the first suspension would make both tasks await each other.
      if vodPrefetchPauseInProgress { return }
      vodPrefetchPauseInProgress = true
      defer { vodPrefetchPauseInProgress = false }

      // A rate limit applies to every speculative request for this origin, not
      // only the task that observed it. Invalidate callbacks before cancelling
      // the shared prefetch downloader, then await every detached task. Keep
      // already materialized successes so the pause does not create more traffic.
      await cancelVODPrefetchAndWait(
        resetDownloader: true,
        discardCompletedResults: false
      )
      for sequence in Array(vodPrefetchResults.keys) {
        guard let result = vodPrefetchResults[sequence],
          case .failure = result
        else { continue }
        vodPrefetchResults[sequence] = nil
      }
      if hadPrefetchActivity,
        vodPrefetchSuspendedUntil > previousSuspension
      {
        log(
          "HLS先読みを\(reason)のため"
            + "\(String(format: "%.1f", boundedPause))秒停止します\n"
        )
      }
    }

    func discardStaleVODPrefetch(before sequence: Int64) async {
      let staleSequences = vodPrefetchTasks.keys.filter { $0 < sequence }
      let staleTasks = staleSequences.compactMap {
        vodPrefetchTasks.removeValue(forKey: $0)
      }
      for task in staleTasks { task.cancel() }
      for task in staleTasks { await task.value }
      for staleSequence in vodPrefetchResults.keys where staleSequence < sequence {
        if let staleResult = vodPrefetchResults.removeValue(forKey: staleSequence) {
          discardVODPrefetchResult(staleResult)
        }
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
      // Fetch the current segment normally before starting a speculative burst.
      // This proves that the selected rendition, credentials and signed URL are
      // usable, and prevents a stale seek target from issuing 24 requests first.
      guard hasMaterializedCurrentVODSegment else { return }
      // Do not compete with AVPlayer's source/audio requests before the first
      // restored output exists. Restrictive origins commonly reject that
      // startup burst even though steady-state prefetch remains usable later.
      guard hasRestoredAnyWindow else { return }
      guard !vodPrefetchPauseInProgress else { return }
      guard Date() >= vodPrefetchSuspendedUntil else { return }
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
        let allowPersistentCache =
          (activeSource.requestContext ?? source.requestContext) == nil
        let generation = vodPrefetchGeneration
        if vodPrefetchTasks.isEmpty && vodPrefetchResults.isEmpty {
          log(
            "HLS先読み開始: 区間\(candidate.sequence)から最大\(maximumVODPrefetchSegments)本"
              + "（待機を含む最大\(maximumVODPrefetchDownloads)本、"
              + "通信数は配信元ごとに自動調整）\n"
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
              cacheDirectory: prefetchCacheDirectory,
              allowPersistentCache: allowPersistentCache,
              priority: .speculative
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
          let rateLimited: Bool
          let browserRelayUnavailable: Bool
          if case let .failure(error) = result {
            rateLimited = Self.isHTTPRateLimit(error)
            browserRelayUnavailable = Self.isBrowserRelayAttemptedUnavailable(error)
          } else {
            rateLimited = false
            browserRelayUnavailable = false
          }
          let shouldSuspendPrefetch = await MainActor.run { () -> Bool in
            guard vodPrefetchGeneration == generation,
              !Task.isCancelled
            else {
              discardVODPrefetchResult(result)
              return false
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
            if rateLimited {
              // Close the refill gate before another fast completion callback
              // can occupy the slot vacated by this 429 response.
              vodPrefetchSuspendedUntil = max(
                vodPrefetchSuspendedUntil,
                Date().addingTimeInterval(8)
              )
              return true
            } else if browserRelayUnavailable {
              // A CORS-hidden response or WebKit bridge failure may already
              // have reached the CDN. Stop every speculative sibling before
              // retrying so three prefetch slots cannot multiply that request.
              vodPrefetchSuspendedUntil = max(
                vodPrefetchSuspendedUntil,
                Date().addingTimeInterval(8)
              )
              return true
            } else {
              startVODPrefetchIfPossible()
              return false
            }
          }
          if shouldSuspendPrefetch {
            await suspendVODPrefetchAfterRateLimit(
              seconds: 8,
              triggeredByPrefetch: true,
              reason: rateLimited
                ? "HTTP 429" : "WebKit HLS通信の再試行待ち"
            )
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
    do {
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
        await cancelVODPrefetchAndWait(resetDownloader: true)
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
        await discardStaleVODPrefetch(before: mediaSegment.sequence)
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
        var ordinaryRetryCount = 0
        var browserRelayRetryCount = 0
        var didAttemptVariantFallback = false
        while true {
          try checkCancellation()
          do {
            let materialized = try await LocalSegmentCache.shared.materialize(
              segment: mediaSegment,
              using: downloader,
              in: sessionDirectory,
              cacheDirectory: localSegmentCacheDirectory,
              allowPersistentCache:
                (activeSource.requestContext ?? source.requestContext) == nil
            )
            localURL = materialized.url
            if materialized.cacheHit {
              log("HLS区間\(mediaSegment.sequence)をローカルキャッシュから再利用しました\n")
            }
            consecutiveSegmentRateLimits = 0
            downloadFailure = nil
            break
          } catch is CancellationError {
            throw CancellationError()
          } catch {
            downloadFailure = error
            if Self.isBrowserRelayAttemptedUnavailable(error) {
              guard browserRelayRetryCount < 3 else { break }
              browserRelayRetryCount += 1
              let retryDelay = Self.rateLimitRetryDelay(
                forConsecutiveFailure: browserRelayRetryCount
              )
              await suspendVODPrefetchAfterRateLimit(
                seconds: retryDelay,
                reason: "WebKit HLS通信の再試行待ち"
              )
              log(
                "HLS区間\(mediaSegment.sequence)のWebKit通信が完了しなかったため、"
                  + "\(String(format: "%.1f", retryDelay))秒待機して"
                  + "同じ区間を再取得します（バッファ中）\n"
              )
              try await sleep(seconds: retryDelay)
              continue
            }
            browserRelayRetryCount = 0
            if Self.isHTTPRateLimit(error) {
              if allowsVariantFallback, !playlist.isLive,
                !didAttemptVariantFallback,
                !sameOriginVariantFallbackRejected
              {
                didAttemptVariantFallback = true
                let fallbackSource: IPadResolvedMediaSource?
                do {
                  fallbackSource = try await IPadMediaURLResolver(
                    resourceLoader: resourceLoader
                  )
                    .resolveNextHLSVariant(for: activeSource)
                } catch is CancellationError {
                  throw CancellationError()
                } catch {
                  // Variant discovery uses the same shared origin cooldown.
                  // A missing/limited alternative must not turn a recoverable
                  // current-segment 429 into a terminal producer error.
                  fallbackSource = nil
                }
                if let fallbackSource {
                  if Self.sharesPrimaryMediaOrigin(
                    activeSource,
                    fallbackSource
                  ) {
                    sameOriginVariantFallbackRejected = true
                    // The master/variant hosts can differ while every media URI
                    // is ultimately served by one CDN. Restarting the player at
                    // a lower rendition then spends another playlist request
                    // but immediately hits the same origin cooldown again.
                    log(
                      "HLSの低いvariantも同じ区間配信元を使用するため、"
                        + "品質を切り替えず待機します\n"
                    )
                  } else {
                    pendingVariantFallbackSource = fallbackSource
                    log(
                      "HLS variantをHTTP 429のため切り替えます: "
                        + "\(Self.variantDescription(activeSource)) → "
                        + "\(Self.variantDescription(fallbackSource))\n"
                    )
                    throw ProductionError.variantFallbackPrepared
                  }
                }
              }

              consecutiveSegmentRateLimits = min(
                6,
                consecutiveSegmentRateLimits + 1
              )
              let retryDelay = Self.rateLimitRetryDelay(
                forConsecutiveFailure: consecutiveSegmentRateLimits
              )
              await suspendVODPrefetchAfterRateLimit(seconds: retryDelay)
              log(
                "HLS区間\(mediaSegment.sequence)がHTTP 429のため、"
                  + "\(String(format: "%.1f", retryDelay))秒待機して"
                  + "同じ区間を再取得します（バッファ中）\n"
              )
              try await sleep(seconds: retryDelay)
              if playlist.isLive {
                // A live segment can leave the sliding window while its origin
                // is limited. Return to the existing playlist refresh path
                // instead of waiting forever on a stale sequence.
                nextMediaSequence = mediaSegment.sequence
                lastRefresh = Date.distantPast
                continue productionLoop
              }
              continue
            }

            consecutiveSegmentRateLimits = 0
            guard ordinaryRetryCount < 3 else { break }
            ordinaryRetryCount += 1
            let retryDelay = min(2, 0.5 * Double(ordinaryRetryCount))
            guard Self.shouldRefreshVODSource(after: error) else {
              log(
                "HLS区間\(mediaSegment.sequence)を"
                  + "\(String(format: "%.1f", retryDelay))秒後に同じURLで再試行します: "
                  + "\(error.localizedDescription)\n"
              )
              try await sleep(seconds: retryDelay)
              continue
            }
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
              try await sleep(seconds: retryDelay)
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
            hasMaterializedCurrentVODSegment = false
            await cancelVODPrefetchAndWait(resetDownloader: true)
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
            try await sleep(seconds: retryDelay)
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
      if !playlist.isLive {
        hasMaterializedCurrentVODSegment = true
      }
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
          // Produce a small first window so startup/seek reaches AVPlayer
          // quickly, then amortize Core AI model/process startup over a much
          // larger steady-state window.
          guard let coreSegmentCount = await vodCoreSegmentCountIfReady(
            restorationWindow,
            coreStartIndex: coreStartIndex,
            hasLeftContext: hasRestoredAnyWindow
          ) else { break }
          let coreEndIndex = coreStartIndex + coreSegmentCount - 1
          try await restoreWindow(
            restorationWindow,
            coreStartIndex: coreStartIndex,
            coreEndIndex: coreEndIndex,
            restoredDirectory: restoredDirectory,
            requestedStartSeconds: startSeconds,
            duration: duration,
            emit: emit
          )
          // Keep the last emitted core as temporal left context. The very first
          // batch has no leading context, so retiring `coreSegmentCount` here
          // would leave the un-emitted right context at index zero and then skip
          // it when the steady batch starts at index one.
          let retirementCount = hasRestoredAnyWindow
            ? coreSegmentCount
            : max(0, coreSegmentCount - 1)
          let retired = Array(restorationWindow.prefix(retirementCount))
          removeMaterializedSources(retired)
          restorationWindow.removeFirst(retirementCount)
          hasRestoredAnyWindow = true
          startVODPrefetchIfPossible()
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
    await cancelVODPrefetchAndWait(resetDownloader: true)
    try checkCancellation()
    emit(.progress(position: duration, duration: duration))
    emit(.ended(duration: duration))
    log("HLSリアルタイム復元が配信末尾へ到達しました\n")
    } catch {
      await cancelVODPrefetchAndWait(resetDownloader: true)
      removeMaterializedSources(restorationWindow)
      restorationWindow.removeAll(keepingCapacity: false)
      throw error
    }
  }

  /// Feeds the restoration worker with short MP4s encoded from AVPlayer's
  /// decoded frames. No HLS playlist, init segment, or media segment is fetched
  /// by URLSession/WKDownload on this path; AVFoundation owns all origin I/O.
  private func runAVFoundationCapture(
    _ capture: MacHLSAVFoundationCapture,
    playlist: IPadHLSMediaPlaylist,
    restoredDirectory: URL,
    duration: Double,
    emit: @escaping EventSink
  ) async throws {
    emit(.ready(duration: duration, isLive: playlist.isLive))
    log(
      "HLSリアルタイム復元を開始: "
        + "Safariと同じAVFoundation映像経路"
        + (playlist.isLive ? " / ライブ\n" : " / \(formatDuration(duration))\n")
    )
    log(
      "HLS通信: 区間URLを再取得せず、AVPlayerのデコード済み映像を復元します\n"
    )

    var restorationWindow: [RestorationSource] = []
    var hasRestoredAnyWindow = false
    var lastTimelineEnd = startSeconds
    let stream = try capture.segments()
    defer { capture.cancel() }

    do {
      for try await captured in stream {
        try checkCancellation()
        guard captured.endSeconds > captured.startSeconds else {
          try? FileManager.default.removeItem(at: captured.url)
          continue
        }
        let resource = IPadHLSResource(url: captured.url, byteRange: nil)
        let mediaSegment = IPadHLSMediaSegment(
          sequence: Int64(captured.sequence),
          duration: captured.endSeconds - captured.startSeconds,
          resource: resource,
          initializationResource: nil,
          startSeconds: captured.startSeconds
        )
        let source = RestorationSource(
          mediaSegment: mediaSegment,
          timelineStart: captured.startSeconds,
          localURL: captured.url
        )
        lastTimelineEnd = max(lastTimelineEnd, captured.endSeconds)

        if let previous = restorationWindow.last,
          captured.startSeconds > previous.timelineEnd + 0.5
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
          emit(.discontinuity(position: captured.startSeconds))
        }

        restorationWindow.append(source)
        while true {
          let coreStartIndex = hasRestoredAnyWindow ? 1 : 0
          guard let coreSegmentCount = avFoundationCoreSegmentCountIfReady(
            restorationWindow,
            coreStartIndex: coreStartIndex,
            hasLeftContext: hasRestoredAnyWindow
          ) else { break }
          let coreEndIndex = coreStartIndex + coreSegmentCount - 1
          try await restoreWindow(
            restorationWindow,
            coreStartIndex: coreStartIndex,
            coreEndIndex: coreEndIndex,
            restoredDirectory: restoredDirectory,
            requestedStartSeconds: startSeconds,
            duration: duration,
            emit: emit
          )
          // Retain the final emitted core as left context and the un-emitted
          // look-ahead source as right context, matching the fast VOD path.
          let retirementCount = hasRestoredAnyWindow
            ? coreSegmentCount
            : max(0, coreSegmentCount - 1)
          let retired = Array(restorationWindow.prefix(retirementCount))
          removeMaterializedSources(retired)
          restorationWindow.removeFirst(retirementCount)
          hasRestoredAnyWindow = true
        }
      }

      try await flushWindow(
        restorationWindow,
        hasLeftContext: hasRestoredAnyWindow,
        restoredDirectory: restoredDirectory,
        requestedStartSeconds: startSeconds,
        duration: max(duration, lastTimelineEnd),
        emit: emit
      )
      removeMaterializedSources(restorationWindow)
      restorationWindow.removeAll(keepingCapacity: false)
      try checkCancellation()
      let finalDuration = max(duration, lastTimelineEnd)
      emit(.progress(position: finalDuration, duration: finalDuration))
      emit(.ended(duration: finalDuration))
      log("AVFoundation HLS復元が配信末尾へ到達しました\n")
    } catch {
      removeMaterializedSources(restorationWindow)
      restorationWindow.removeAll(keepingCapacity: false)
      throw error
    }
  }

  /// The Safari-compatible source cannot materialize eighteen intervals in a
  /// burst like the fast downloader. Start with two cores for low latency, then
  /// restore four cores per worker so model/process startup is paid once for
  /// eight seconds of output while the 2x capture keeps filling independently.
  private func avFoundationCoreSegmentCountIfReady(
    _ sources: [RestorationSource],
    coreStartIndex: Int,
    hasLeftContext: Bool
  ) -> Int? {
    guard coreStartIndex >= 0, coreStartIndex < sources.count else { return nil }
    let availableCoreCount = sources.count - coreStartIndex - 1
    let requiredCount = hasLeftContext
      ? avFoundationSteadyRestoreBatchCoreSegments
      : vodInitialRestoreBatchCoreSegments
    guard availableCoreCount >= requiredCount else { return nil }
    return requiredCount
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
    resumeOutputCreditWaiters()
    avFoundationCapture?.cancel()
    downloader?.cancel()
    prefetchDownloader?.cancel()
    for downloader in prefetchDownloaders { downloader.cancel() }
    prefetchDownloaders.removeAll(keepingCapacity: true)
    _ = sendWorkerCommand(["command": "stop"])
    let retiringInput = activeWorkerInput
    activeWorkerInput = nil
    try? retiringInput?.fileHandleForWriting.close()
    try? activeWorkerOutput?.fileHandleForReading.close()
    try? activeWorkerError?.fileHandleForReading.close()
    if let activeProcess {
      scheduleProcessRetirement(activeProcess)
    }
  }

  private func completeRun() {
    isRunActive = false
    resumeOutputCreditWaiters()
    let waiters = completionWaiters
    completionWaiters.removeAll(keepingCapacity: false)
    for waiter in waiters { waiter.resume() }
  }

  private func waitForOutputCredit(
    sequence: Int,
    seconds: Double,
    bytes: Int64
  ) async throws -> TimeInterval {
    let startedAt = Date()
    var reportedWait = false
    while !hasOutputCredit(seconds: seconds, bytes: bytes) {
      try checkCancellation()
      if !reportedWait {
        reportedWait = true
        log(
          "HLS復元出力待機: 出力\(sequence) / 在庫"
            + "\(retainedOutputCredits.count)本・"
            + "\(String(format: "%.1f", retainedOutputSeconds))秒・"
            + "\(formatBytes(retainedOutputBytes))\n"
        )
      }
      await withCheckedContinuation { continuation in
        outputCreditWaiters.append(continuation)
      }
    }
    try checkCancellation()
    if reportedWait {
      log(
        "HLS復元出力再開: 出力\(sequence) / 在庫"
          + "\(retainedOutputCredits.count)本・"
          + "\(String(format: "%.1f", retainedOutputSeconds))秒・"
          + "\(formatBytes(retainedOutputBytes))\n"
      )
    }
    return Date().timeIntervalSince(startedAt)
  }

  private func hasOutputCredit(seconds: Double, bytes: Int64) -> Bool {
    // Always admit one segment so an unusually large but valid file cannot
    // deadlock an empty queue forever.
    guard !retainedOutputCredits.isEmpty else { return true }
    let incomingSeconds = max(0, seconds)
    // A nominal 2-second HLS output is commonly 2.002 seconds at 29.97 fps.
    // Keep items and bytes strict, but allow one incoming segment of duration
    // slack. Otherwise a 6-second startup target can reject its third item at
    // 6.006 seconds and wait forever for an ACK before playback has started.
    let oneSegmentDurationSlack = max(0.001, incomingSeconds)
    return retainedOutputCredits.count + 1 <= outputBufferLimits.items
      && retainedOutputSeconds + incomingSeconds
        <= outputBufferLimits.seconds + oneSegmentDurationSlack + 0.001
      && retainedOutputBytes + max(0, bytes) <= outputBufferLimits.bytes
  }

  private func retainOutputCredit(sequence: Int, seconds: Double, bytes: Int64) {
    guard retainedOutputCredits[sequence] == nil else { return }
    let credit = RetainedOutputCredit(
      seconds: max(0, seconds),
      bytes: max(0, bytes)
    )
    retainedOutputCredits[sequence] = credit
    retainedOutputSeconds += credit.seconds
    retainedOutputBytes += credit.bytes
  }

  private func resumeOutputCreditWaiters() {
    let waiters = outputCreditWaiters
    outputCreditWaiters.removeAll(keepingCapacity: true)
    for waiter in waiters { waiter.resume() }
  }

  private func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: max(0, bytes), countStyle: .file)
  }

  private func waitForProcessExit(
    _ process: Process,
    timeoutSeconds: TimeInterval = 1
  ) async -> Bool {
    let sendable = SendableProcess(process: process)
    return await Task.detached(priority: .utility) {
      let deadline = Date().addingTimeInterval(max(0.05, timeoutSeconds))
      while sendable.process.isRunning, Date() < deadline {
        usleep(20_000)
      }
      return !sendable.process.isRunning
    }.value
  }

  /// A Core AI launcher can keep descendants alive after `Process.terminate()`.
  /// Retire the whole process group with a bounded grace period so cancellation
  /// and seek generation replacement cannot wait forever on stdout EOF.
  @discardableResult
  private func scheduleProcessRetirement(
    _ process: Process
  ) -> Task<Void, Never> {
    if activeProcess === process, let activeProcessRetirementTask {
      return activeProcessRetirementTask
    }
    let sendable = SendableProcess(process: process)
    let task = Task.detached(priority: .utility) {
      await Self.retireProcessTree(sendable)
    }
    if activeProcess === process {
      activeProcessRetirementTask = task
    }
    return task
  }

  private func retireProcess(_ process: Process) async {
    let task = scheduleProcessRetirement(process)
    await task.value
    if activeProcess === process {
      activeProcessRetirementTask = nil
    }
  }

  nonisolated private static func retireProcessTree(
    _ sendable: SendableProcess
  ) async {
    let process = sendable.process
    let processIdentifier = process.processIdentifier
    guard processIdentifier > 0 else {
      if process.isRunning { process.terminate() }
      return
    }

    if process.isRunning {
      if kill(-processIdentifier, SIGTERM) != 0 {
        process.terminate()
      }
    }
    for _ in 0..<20 where Self.processTreeIsAlive(processIdentifier) {
      try? await Task.sleep(nanoseconds: 50_000_000)
    }
    guard Self.processTreeIsAlive(processIdentifier) else { return }
    // Send to both the launcher's group and PID: the negative-PID call retires
    // Core AI descendants when the launcher owns a process group, while the
    // positive PID remains a safe fallback if no such group exists.
    _ = kill(-processIdentifier, SIGKILL)
    _ = kill(processIdentifier, SIGKILL)
    for _ in 0..<20 where Self.processTreeIsAlive(processIdentifier) {
      try? await Task.sleep(nanoseconds: 50_000_000)
    }
  }

  nonisolated private static func processIsAlive(
    _ processIdentifier: pid_t
  ) -> Bool {
    kill(processIdentifier, 0) == 0 || errno == EPERM
  }

  nonisolated private static func processTreeIsAlive(
    _ processIdentifier: pid_t
  ) -> Bool {
    if processIsAlive(processIdentifier) { return true }
    return kill(-processIdentifier, 0) == 0 || errno == EPERM
  }

  private func makeDownloader(
    for resolvedSource: IPadResolvedMediaSource
  ) -> IPadHLSResourceDownloader {
    IPadHLSResourceDownloader(
      maximumResourceBytes: 256 * 1_024 * 1_024,
      maximumRedirectCount: 6,
      requestTimeout: 30,
      resolutionPolicy: resolvedSource.resolutionPolicy,
      requestContext: resolvedSource.requestContext ?? source.requestContext,
      resourceLoader: resourceLoader
    )
  }

  private static func userSegmentCacheDirectory() throws -> URL {
    guard let userCaches = FileManager.default.urls(
      for: .cachesDirectory,
      in: .userDomainMask
    ).first else {
      throw ProductionError.invalidSource("ユーザーキャッシュフォルダを取得できません")
    }
    let bundleComponent = (Bundle.main.bundleIdentifier ?? "com.okatti.lada.coreai")
      .filter { $0.isLetter || $0.isNumber || $0 == "." || $0 == "-" }
    return userCaches
      .appendingPathComponent(
        bundleComponent.isEmpty ? "com.okatti.lada.coreai" : bundleComponent,
        isDirectory: true
      )
      .appendingPathComponent("mioh", isDirectory: true)
      .appendingPathComponent("hls-segment-cache", isDirectory: true)
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
        let resolved = try await IPadMediaURLResolver(
          resourceLoader: resourceLoader
        ).resolve(
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
        // A shared origin cooldown is already active after 429. Trying the
        // submitted URL, referer and media URL in sequence would turn one rate
        // limit into several extra HEAD/GET playlist requests.
        if Self.isHTTPRateLimit(error) { throw error }
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

  private nonisolated static func httpStatusCode(in error: Error) -> Int? {
    guard let resolverError = error as? IPadMediaURLResolverError,
      case let .invalidHTTPStatus(statusCode) = resolverError
    else { return nil }
    return statusCode
  }

  private nonisolated static func isHTTPRateLimit(_ error: Error) -> Bool {
    httpStatusCode(in: error) == 429
  }

  private nonisolated static func isBrowserRelayAttemptedUnavailable(
    _ error: Error
  ) -> Bool {
    guard let loadingError = error as? IPadHLSResourceLoadingError else {
      return false
    }
    return loadingError == .attemptedUnavailable
  }

  private nonisolated static func rateLimitRetryDelay(
    forConsecutiveFailure failureCount: Int
  ) -> TimeInterval {
    let boundedExponent = max(0, min(5, failureCount - 1))
    return min(30, 1.5 * pow(2, Double(boundedExponent)))
  }

  private nonisolated static func variantDescription(
    _ source: IPadResolvedMediaSource
  ) -> String {
    let playlist = source.hlsPlaylist
    let metadata = playlist?.masterMetadata
    let host = playlist?.url.host
      ?? source.mediaURL.host
      ?? source.playbackURL.host
      ?? "host不明"
    let resolution: String
    if let width = metadata?.width, let height = metadata?.height,
      width > 0, height > 0
    {
      resolution = "\(width)x\(height)"
    } else {
      resolution = "解像度不明"
    }
    return "\(host) / \(resolution)"
  }

  private nonisolated static func sharesPrimaryMediaOrigin(
    _ lhs: IPadResolvedMediaSource,
    _ rhs: IPadResolvedMediaSource
  ) -> Bool {
    guard let left = primaryMediaOriginKey(lhs),
      let right = primaryMediaOriginKey(rhs)
    else { return false }
    return left == right
  }

  private nonisolated static func primaryMediaOriginKey(
    _ source: IPadResolvedMediaSource
  ) -> String? {
    guard let url = source.hlsPlaylist?.segments.first?.resource.url,
      let scheme = url.scheme?.lowercased(),
      let host = url.host?.lowercased()
    else { return nil }
    let port = url.port ?? (scheme == "https" ? 443 : 80)
    return "\(scheme)://\(host):\(port)"
  }

  /// Only responses that plausibly indicate an expired or revoked signed media
  /// URL justify re-resolving the master playlist. Rate limits, server errors
  /// and transport failures retry the exact resource instead, avoiding a burst
  /// of HEAD/master/media-playlist requests while an origin is already unhappy.
  private nonisolated static func shouldRefreshVODSource(after error: Error) -> Bool {
    guard let statusCode = httpStatusCode(in: error) else { return false }
    return [401, 403, 404, 410].contains(statusCode)
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

  /// Selects a VOD worker batch while reserving one source on the right for
  /// temporal context. Fast-start always uses two core segments. Steady state
  /// grows to at most eighteen, but a variable-duration playlist is cut near a
  /// 36-second/384-MiB budget instead of accidentally creating a multi-minute
  /// worker invocation from eighteen unusually long HLS segments.
  private func vodCoreSegmentCountIfReady(
    _ sources: [RestorationSource],
    coreStartIndex: Int,
    hasLeftContext: Bool
  ) async -> Int? {
    guard coreStartIndex >= 0, coreStartIndex < sources.count else { return nil }
    let availableCoreCount = sources.count - coreStartIndex - 1
    let minimumCount = hasLeftContext
      ? vodMinimumSteadyBatchCoreSegments
      : vodInitialRestoreBatchCoreSegments
    guard availableCoreCount >= minimumCount else { return nil }
    if !hasLeftContext {
      return vodInitialRestoreBatchCoreSegments
    }

    let maximumCount = min(
      availableCoreCount,
      vodSteadyRestoreBatchCoreSegments
    )
    var selectedCount = 0
    var selectedDuration = 0.0
    var selectedBytes: Int64 = 0

    for relativeIndex in 0..<maximumCount {
      let source = sources[coreStartIndex + relativeIndex]
      let sourceDuration = max(0, source.timelineEnd - source.timelineStart)
      let sourceBytes = (try? await mediaFileWorker.byteCount(at: source.localURL)) ?? 0
      let exceedsDuration = selectedCount >= minimumCount
        && selectedDuration + sourceDuration > vodSteadyRestoreBatchTargetSeconds
      let exceedsBytes = selectedCount >= minimumCount
        && sourceBytes > 0
        && selectedBytes + sourceBytes > vodSteadyRestoreBatchMaximumBytes
      if exceedsDuration || exceedsBytes { return selectedCount }

      selectedCount += 1
      selectedDuration += sourceDuration
      selectedBytes += sourceBytes
      if selectedCount >= minimumCount,
        selectedCount == vodSteadyRestoreBatchCoreSegments
          || selectedDuration >= vodSteadyRestoreBatchTargetSeconds
          || selectedBytes >= vodSteadyRestoreBatchMaximumBytes
      {
        return selectedCount
      }
    }

    // The final source currently reserved as right context is also the next
    // potential core. If adding it would exceed a budget, the boundary is known
    // and this batch can start now; otherwise wait for more prefetch inventory.
    let nextIndex = coreStartIndex + selectedCount
    guard selectedCount >= minimumCount, sources.indices.contains(nextIndex)
    else { return nil }
    let next = sources[nextIndex]
    let nextDuration = max(0, next.timelineEnd - next.timelineStart)
    let nextBytes = (try? await mediaFileWorker.byteCount(at: next.localURL)) ?? 0
    if selectedDuration + nextDuration > vodSteadyRestoreBatchTargetSeconds
      || (nextBytes > 0
        && selectedBytes + nextBytes > vodSteadyRestoreBatchMaximumBytes)
    {
      return selectedCount
    }
    return nil
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
      assembled = try await assembleInterval(
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
      try await intervalAssemblyWorker.validateDecodableVideo(
        at: assemblyURL,
        near: assembled.sourceOffsets[assembledCoreStartIndex]
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      if coreStartIndex < coreEndIndex {
        // A provider can change codec parameters without an HLS discontinuity
        // marker. Do not make the larger steady-state optimization fatal: retry
        // each core with only its immediate temporal neighbours, which is the
        // previously proven low-latency path and still preserves output order.
        log(
          "HLS適応連結を区間単位へ縮小して続行します: "
            + "\(error.localizedDescription)\n"
        )
        try? FileManager.default.removeItem(at: assemblyURL)
        for coreIndex in coreStartIndex...coreEndIndex {
          let lowerBound = max(sources.startIndex, coreIndex - 1)
          let upperBound = min(sources.index(before: sources.endIndex), coreIndex + 1)
          let localSources = Array(sources[lowerBound...upperBound])
          try await restoreWindow(
            localSources,
            coreIndex: coreIndex - lowerBound,
            restoredDirectory: restoredDirectory,
            requestedStartSeconds: requestedStartSeconds,
            duration: duration,
            emit: emit
          )
        }
        return
      }
      guard coreStartIndex == coreEndIndex else { throw error }
      guard sources.count > 1 else { throw error }
      log(
        "HLS隣接区間の符号化条件が変化したため、対象区間だけで復元を続行します: "
          + "\(error.localizedDescription)\n"
      )
      try? FileManager.default.removeItem(at: assemblyURL)
      assembled = try await intervalAssemblyWorker.concatenate(
        inputURLs: [coreSource.localURL],
        outputURL: assemblyURL,
        temporaryDirectory: sessionDirectory
      )
      assembledCoreStartIndex = 0
      assembledCoreEndIndex = 0
      guard let coreOffset = assembled.sourceOffsets.first else {
        throw ProductionError.invalidTimeline("単一区間の対応を取得できません")
      }
      try await intervalAssemblyWorker.validateDecodableVideo(
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

  /// The portable assembler deliberately caps one operation at eight inputs to
  /// bound MPEG-TS parsing memory. Build larger VOD windows hierarchically: each
  /// leaf normalizes at most eight original segments and the root joins at most
  /// eight normalized movies. This keeps the portable safety limit intact while
  /// allowing a steady-state worker to amortize startup over 12-24 core inputs.
  private func assembleInterval(
    inputURLs: [URL],
    outputURL: URL,
    temporaryDirectory: URL
  ) async throws -> IPadHLSIntervalAssembler.Result {
    let leafLimit = IPadHLSIntervalAssembler.maximumInputCount
    guard !inputURLs.isEmpty, inputURLs.count <= leafLimit * leafLimit else {
      throw ProductionError.invalidSource(
        "階層HLS連結は1〜\(leafLimit * leafLimit)区間に対応しています"
      )
    }
    if inputURLs.count <= leafLimit {
      return try await intervalAssemblyWorker.concatenate(
        inputURLs: inputURLs,
        outputURL: outputURL,
        temporaryDirectory: temporaryDirectory
      )
    }

    var leafURLs: [URL] = []
    var leafResults: [IPadHLSIntervalAssembler.Result] = []
    defer {
      for url in leafURLs { try? FileManager.default.removeItem(at: url) }
    }
    var offset = 0
    while offset < inputURLs.count {
      try checkCancellation()
      let end = min(inputURLs.count, offset + leafLimit)
      let leafURL = temporaryDirectory.appendingPathComponent(
        "hls-leaf-\(nextOutputSequence)-\(offset)-\(UUID().uuidString.lowercased()).mp4",
        isDirectory: false
      )
      let leafResult = try await intervalAssemblyWorker.concatenate(
        inputURLs: Array(inputURLs[offset..<end]),
        outputURL: leafURL,
        temporaryDirectory: temporaryDirectory
      )
      guard leafResult.sourceOffsets.count == end - offset,
        leafResult.sourceDurations.count == end - offset
      else {
        throw ProductionError.invalidTimeline("階層HLS連結の葉区間数が一致しません")
      }
      leafURLs.append(leafURL)
      leafResults.append(leafResult)
      offset = end
    }

    let root = try await intervalAssemblyWorker.concatenate(
      inputURLs: leafURLs,
      outputURL: outputURL,
      temporaryDirectory: temporaryDirectory
    )
    guard root.sourceOffsets.count == leafResults.count,
      root.sourceDurations.count == leafResults.count
    else {
      throw ProductionError.invalidTimeline("階層HLS連結の時間対応を取得できません")
    }
    var sourceOffsets: [TimeInterval] = []
    var sourceDurations: [TimeInterval] = []
    for (leafIndex, leaf) in leafResults.enumerated() {
      let rootOffset = root.sourceOffsets[leafIndex]
      sourceOffsets.append(contentsOf: leaf.sourceOffsets.map { rootOffset + $0 })
      sourceDurations.append(contentsOf: leaf.sourceDurations)
    }
    guard sourceOffsets.count == inputURLs.count,
      sourceDurations.count == inputURLs.count
    else {
      throw ProductionError.invalidTimeline("階層HLS連結の区間数が一致しません")
    }
    return IPadHLSIntervalAssembler.Result(
      sourceOffsets: sourceOffsets,
      sourceDurations: sourceDurations,
      duration: root.duration
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
    activeWorkerOutput = outputPipe
    activeWorkerError = errorPipe
    activeProcessRetirementTask = nil
    guard MacChildProcessPipe.prepare(inputPipe.fileHandleForWriting) else {
      activeProcess = nil
      activeWorkerInput = nil
      activeWorkerOutput = nil
      activeWorkerError = nil
      throw ProductionError.worker("worker stdinを安全に準備できません")
    }

    let workerStartedAt = Date()
    var firstSegmentElapsed: TimeInterval?
    var emittedSegmentCount = 0
    var outputCreditWaitElapsed: TimeInterval = 0
    do {
      try process.run()
    } catch {
      activeProcess = nil
      activeWorkerInput = nil
      activeWorkerOutput = nil
      activeWorkerError = nil
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
            _ = sendWorkerCommand([
              "command": "release_through",
              "sequence": workerSequence,
            ])
            continue
          }
          let outputSequence = nextOutputSequence
          let workerURL = URL(fileURLWithPath: path)
          let outputBytes: Int64
          do {
            outputBytes = try await mediaFileWorker.byteCount(at: workerURL)
          } catch {
            throw ProductionError.missingOutput(
              "\(workerURL.lastPathComponent): \(error.localizedDescription)"
            )
          }
          outputCreditWaitElapsed += try await waitForOutputCredit(
            sequence: outputSequence,
            seconds: mappedEnd - mappedStart,
            bytes: outputBytes
          )
          let stableURL = restoredDirectory.appendingPathComponent(
            String(format: "hls-restored-%06d.mp4", outputSequence),
            isDirectory: false
          )
          do {
            try await mediaFileWorker.copyReplacing(
              from: workerURL,
              to: stableURL
            )
          } catch {
            throw ProductionError.missingOutput(
              "\(workerURL.lastPathComponent): \(error.localizedDescription)"
            )
          }
          // The native worker may now discard its rolling file. The separate
          // playback credit remains held until the controller acknowledges the
          // matching MacHLSProductionEvent.segment sequence.
          _ = sendWorkerCommand([
            "command": "release_through",
            "sequence": workerSequence,
          ])
          retainOutputCredit(
            sequence: outputSequence,
            seconds: mappedEnd - mappedStart,
            bytes: outputBytes
          )
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
      _ = sendWorkerCommand(["command": "stop"])
      if activeWorkerInput === inputPipe {
        activeWorkerInput = nil
      }
      try? inputPipe.fileHandleForWriting.close()
      await retireProcess(process)
      try? outputPipe.fileHandleForReading.close()
      try? errorPipe.fileHandleForReading.close()
      activeProcess = nil
      activeWorkerInput = nil
      activeWorkerOutput = nil
      activeWorkerError = nil
      _ = await errorTask.value
      throw error
    }

    if activeWorkerInput === inputPipe {
      activeWorkerInput = nil
    }
    try? inputPipe.fileHandleForWriting.close()
    if !(await waitForProcessExit(process)) {
      await retireProcess(process)
    }
    let workerElapsed = Date().timeIntervalSince(workerStartedAt)
    let restoredCoreDuration = max(
      0.001,
      coreTimelineEndSeconds - coreTimelineStartSeconds
    )
    // A full playback queue deliberately stalls this stdout consumer. Exclude
    // that backpressure time so RTF describes restoration/encode throughput,
    // while logging the wait independently for queue tuning.
    let workerActiveElapsed = max(0.001, workerElapsed - outputCreditWaitElapsed)
    let realtimeFactor = workerActiveElapsed / restoredCoreDuration
    activeProcess = nil
    activeWorkerInput = nil
    activeWorkerOutput = nil
    activeWorkerError = nil
    activeProcessRetirementTask = nil
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
          + " / 全体\(String(format: "%.2f", workerElapsed))秒"
          + " / RTF \(String(format: "%.3f", realtimeFactor))"
          + " / credit待機\(String(format: "%.2f", outputCreditWaitElapsed))秒"
          + " / credit \(retainedOutputCredits.count)本・"
          + "\(String(format: "%.1f", retainedOutputSeconds))秒・"
          + "\(formatBytes(retainedOutputBytes))\n"
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

  @discardableResult
  private func sendWorkerCommand(_ payload: [String: Any]) -> Bool {
    guard let process = activeProcess,
      process.isRunning,
      let inputPipe = activeWorkerInput,
      let data = try? JSONSerialization.data(withJSONObject: payload)
    else { return false }
    let handle = inputPipe.fileHandleForWriting
    var line = data
    line.append(0x0A)
    guard MacChildProcessPipe.write(line, to: handle) else {
      if activeWorkerInput === inputPipe {
        activeWorkerInput = nil
      }
      try? handle.close()
      return false
    }
    return true
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
