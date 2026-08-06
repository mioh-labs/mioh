import Combine
import Foundation
import Security

/// Converts finalized, video-only preview MP4 files into a small rolling HLS
/// window.  The source segment is isolated immediately with a hard link (or an
/// asynchronous copy when links are unavailable), so AVQueuePlayer remains
/// free to release its own rolling files without racing the LAN stream.
///
/// Remuxing is serialized on an asynchronous task and never feeds capacity
/// back into NativePreviewPipeline.  Slow streaming therefore drops old HLS
/// media rather than stalling detection or restoration.
@MainActor
final class RemoteStreamingCoordinator: ObservableObject, RemoteStreamingCoordinating {
  private static let maximumSessions = 2
  private static let maximumSegments = 12
  private static let maximumBytes: Int64 = 256 * 1024 * 1024
  private static let sessionLifetime: TimeInterval = 15 * 60

  private struct TicketRecord {
    let expiresAt: Date
  }

  private struct PublishedSegment {
    let sequence: Int
    let duration: Double
    let url: URL
    let byteCount: Int64
  }

  private struct PendingSegment {
    let identifier: UUID
    let generation: Int
    let sequence: Int
    let startSeconds: Double
    let endSeconds: Double
    let sourceURL: URL
    let ffmpegURL: URL
    let isolatedURL: URL
    let finalURL: URL
    let isolation: Task<URL, Error>
  }

  private struct GenerationState {
    var source: RealtimeStreamingSource
    var directory: URL?
    var segments: [PublishedSegment] = []
    var publishedBytes: Int64 = 0
    var pending: [PendingSegment] = []
    var ended = false
    var errorMessage: String?
  }

  @Published private(set) var active = false
  @Published private(set) var publishedSegmentCount = 0
  @Published private(set) var lastError = ""

  private var generation: GenerationState?
  private var tickets: [String: TicketRecord] = [:]
  private var activeTask: Task<Void, Never>?
  private var activeTaskID: UUID?
  private var activePending: PendingSegment?
  private var serverEnabled = false

  deinit {
    activeTask?.cancel()
  }

  /// The server can install this closure directly on RealtimePlayerController.
  /// Reattaching triggers the controller's current-queue snapshot contract.
  func eventConsumer() -> RealtimeStreamingEventConsumer {
    { [weak self] event in
      self?.consume(event)
    }
  }

  func consume(_ event: RealtimeStreamingEvent) {
    switch event {
    case .reset(let source):
      if generation?.source.generation == source.generation {
        generation?.source = source
        return
      }
      reset(to: source)

    case .segment(let segment):
      enqueue(segment)

    case .ended(let eventGeneration):
      guard generation?.source.generation == eventGeneration else { return }
      generation?.ended = true

    case .stopped(let eventGeneration):
      guard generation?.source.generation == eventGeneration else { return }
      stop()
    }
  }

  func setServerEnabled(_ enabled: Bool) {
    guard serverEnabled != enabled else { return }
    serverEnabled = enabled
    if !enabled {
      tickets.removeAll(keepingCapacity: false)
      discardPublishedMediaKeepingGeneration()
    }
  }

  func issueSession() -> RemoteStreamSessionIssue {
    pruneExpiredTickets()
    guard serverEnabled, generation != nil, active else { return .unavailable }
    guard tickets.count < Self.maximumSessions else { return .capacityReached }
    let ticket = Self.generateTicket()
    let expiresAt = Date().addingTimeInterval(Self.sessionLifetime)
    tickets[ticket] = TicketRecord(expiresAt: expiresAt)
    return .issued(RemoteStreamSession(ticket: ticket, expiresAt: expiresAt))
  }

  func revokeSession(ticket: String) {
    tickets.removeValue(forKey: ticket)
    if tickets.isEmpty { discardPublishedMediaKeepingGeneration() }
  }

  func revokeAllSessions() {
    tickets.removeAll(keepingCapacity: false)
    discardPublishedMediaKeepingGeneration()
  }

  func statusJSON() -> [String: Any] {
    pruneExpiredTickets()
    let segments = generation?.segments ?? []
    return [
      "active": active,
      "serving": serverEnabled && !tickets.isEmpty,
      "generation": generation?.source.generation as Any,
      "segmentCount": segments.count,
      "mediaSequence": segments.first?.sequence as Any,
      "latestSequence": segments.last?.sequence as Any,
      "bufferedSeconds": segments.reduce(0) { $0 + $1.duration },
      "ended": generation?.ended ?? false,
      "sessions": tickets.count,
      "hasError": !(generation?.errorMessage ?? "").isEmpty,
    ]
  }

  func playlistData(ticket: String) -> Data? {
    guard validTicket(ticket), let generation else { return nil }
    let targetDuration = max(
      1,
      Int(ceil(generation.segments.map(\.duration).max() ?? generation.source.segmentSeconds))
    )
    let mediaSequence = generation.segments.first?.sequence ?? 0
    var lines = [
      "#EXTM3U",
      "#EXT-X-VERSION:3",
      "#EXT-X-TARGETDURATION:\(targetDuration)",
      "#EXT-X-MEDIA-SEQUENCE:\(mediaSequence)",
      "#EXT-X-INDEPENDENT-SEGMENTS",
    ]
    for segment in generation.segments {
      lines.append(
        String(format: "#EXTINF:%.6f,", locale: Locale(identifier: "en_US_POSIX"), segment.duration)
      )
      lines.append("segment/\(segment.sequence).ts")
    }
    if generation.ended && generation.pending.isEmpty && activeTask == nil {
      lines.append("#EXT-X-ENDLIST")
    }
    lines.append("")
    return Data(lines.joined(separator: "\n").utf8)
  }

  func segmentURL(ticket: String, sequence: Int) -> URL? {
    guard validTicket(ticket),
      let segment = generation?.segments.first(where: {
        $0.sequence == sequence
      })
    else { return nil }
    return segment.url
  }

  func stop() {
    cancelWorkAndRemoveMedia(revokeTickets: true)
    generation = nil
    active = false
    publishedSegmentCount = 0
    lastError = ""
  }

  private func reset(to source: RealtimeStreamingSource) {
    cancelWorkAndRemoveMedia(revokeTickets: true)
    generation = GenerationState(source: source)
    active = true
    publishedSegmentCount = 0
    lastError = ""
  }

  private func enqueue(_ segment: RealtimeStreamingSegment) {
    pruneExpiredTickets()
    let codec = segment.codec.lowercased()
    guard var state = generation,
      serverEnabled,
      !tickets.isEmpty,
      state.source.generation == segment.generation,
      segment.endSeconds > segment.startSeconds,
      codec.contains("h264") || codec.contains("x264") || codec == "avc1"
    else { return }
    guard !state.segments.contains(where: { $0.sequence == segment.sequence }),
      activePending?.sequence != segment.sequence,
      !state.pending.contains(where: { $0.sequence == segment.sequence })
    else { return }

    do {
      let directory: URL
      if let existing = state.directory {
        directory = existing
      } else {
        // Keeping the streaming directory beside the preview segments makes
        // the normal path a metadata-only hard link, even when the user puts
        // mioh's temporary directory on an external volume.
        directory = segment.url.deletingLastPathComponent().appendingPathComponent(
          "remote-hls-g\(segment.generation)-\(UUID().uuidString)",
          isDirectory: true
        )
        try FileManager.default.createDirectory(
          at: directory,
          withIntermediateDirectories: true
        )
        state.directory = directory
      }

      let isolated = directory.appendingPathComponent(
        String(format: "source-%06d.mp4", segment.sequence)
      )
      let final = directory.appendingPathComponent(
        String(format: "segment-%06d.ts", segment.sequence)
      )
      try? FileManager.default.removeItem(at: isolated)
      try? FileManager.default.removeItem(at: final)

      let isolation: Task<URL, Error>
      do {
        try FileManager.default.linkItem(at: segment.url, to: isolated)
        isolation = Task { isolated }
      } catch {
        // Cross-volume or link-hostile filesystems take the slower path, but
        // copying starts immediately rather than waiting behind older remuxes.
        let original = segment.url
        isolation = Task.detached(priority: .utility) {
          try Task.checkCancellation()
          try FileManager.default.copyItem(at: original, to: isolated)
          try Task.checkCancellation()
          return isolated
        }
      }

      let pending = PendingSegment(
        identifier: UUID(),
        generation: segment.generation,
        sequence: segment.sequence,
        startSeconds: segment.startSeconds,
        endSeconds: segment.endSeconds,
        sourceURL: state.source.inputURL,
        ffmpegURL: state.source.ffmpegURL,
        isolatedURL: isolated,
        finalURL: final,
        isolation: isolation
      )
      state.pending.append(pending)
      // A pathological remux slowdown must never backpressure restoration.
      // Retain only the newest rolling work and let the playlist advance.
      while state.pending.count > Self.maximumSegments {
        let dropped = state.pending.removeFirst()
        dropped.isolation.cancel()
        try? FileManager.default.removeItem(at: dropped.isolatedURL)
      }
      generation = state
      scheduleNextIfNeeded()
    } catch {
      record(error, generation: segment.generation)
    }
  }

  private func scheduleNextIfNeeded() {
    guard activeTask == nil, var state = generation, !state.pending.isEmpty else { return }
    let pending = state.pending.removeFirst()
    state.errorMessage = nil
    generation = state
    let taskID = UUID()
    activeTaskID = taskID
    activePending = pending
    activeTask = Task { [weak self] in
      do {
        let isolated = try await pending.isolation.value
        try Task.checkCancellation()
        try await Self.mux(pending, isolatedURL: isolated)
        try Task.checkCancellation()
        let attributes = try FileManager.default.attributesOfItem(atPath: pending.finalURL.path)
        let byteCount = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard byteCount > 0 else { throw RemoteStreamingProcessError.emptyOutput }
        self?.complete(
          pending,
          taskID: taskID,
          result: .success(byteCount)
        )
      } catch {
        self?.complete(pending, taskID: taskID, result: .failure(error))
      }
    }
  }

  private func complete(
    _ pending: PendingSegment,
    taskID: UUID,
    result: Result<Int64, Error>
  ) {
    try? FileManager.default.removeItem(at: pending.isolatedURL)
    guard activeTaskID == taskID else {
      try? FileManager.default.removeItem(at: pending.finalURL)
      return
    }
    activeTask = nil
    activeTaskID = nil
    activePending = nil
    guard var state = generation, state.source.generation == pending.generation else {
      try? FileManager.default.removeItem(at: pending.finalURL)
      return
    }
    switch result {
    case .success(let byteCount):
      let published = PublishedSegment(
        sequence: pending.sequence,
        duration: pending.endSeconds - pending.startSeconds,
        url: pending.finalURL,
        byteCount: byteCount
      )
      state.segments.append(published)
      state.segments.sort { $0.sequence < $1.sequence }
      state.publishedBytes += byteCount
      trim(&state)
      state.errorMessage = nil
      lastError = ""
    case .failure(let error):
      try? FileManager.default.removeItem(at: pending.finalURL)
      if !(error is CancellationError) {
        state.errorMessage = error.localizedDescription
        lastError = error.localizedDescription
      }
    }
    publishedSegmentCount = state.segments.count
    generation = state
    scheduleNextIfNeeded()
  }

  private func trim(_ state: inout GenerationState) {
    while state.segments.count > 1,
      state.segments.count > Self.maximumSegments
        || state.publishedBytes > Self.maximumBytes
    {
      let removed = state.segments.removeFirst()
      state.publishedBytes = max(0, state.publishedBytes - removed.byteCount)
      try? FileManager.default.removeItem(at: removed.url)
    }
  }

  private func record(_ error: Error, generation expectedGeneration: Int) {
    guard generation?.source.generation == expectedGeneration else { return }
    generation?.errorMessage = error.localizedDescription
    lastError = error.localizedDescription
  }

  private func validTicket(_ ticket: String) -> Bool {
    pruneExpiredTickets()
    guard let record = tickets[ticket], record.expiresAt > Date() else { return false }
    // Active HLS polling keeps a session alive; abandoned browser tabs expire
    // after the inactivity window and stop all remux work automatically.
    tickets[ticket] = TicketRecord(
      expiresAt: Date().addingTimeInterval(Self.sessionLifetime)
    )
    return serverEnabled && generation != nil && active
  }

  private func pruneExpiredTickets() {
    let now = Date()
    tickets = tickets.filter { $0.value.expiresAt > now }
    if tickets.isEmpty { discardPublishedMediaKeepingGeneration() }
  }

  private static func generateTicket() -> String {
    var bytes = [UInt8](repeating: 0, count: 32)
    guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
      return (UUID().uuidString + UUID().uuidString)
        .replacingOccurrences(of: "-", with: "")
        .lowercased()
    }
    return bytes.map { String(format: "%02x", $0) }.joined()
  }

  /// Stops remuxing and removes the HLS window without discarding the current
  /// source/generation metadata. A future authorized session starts at the
  /// next finalized preview segment and never makes a remote client part of
  /// restoration backpressure.
  private func discardPublishedMediaKeepingGeneration() {
    activeTask?.cancel()
    activeTask = nil
    activeTaskID = nil
    activePending?.isolation.cancel()
    activePending = nil
    guard var state = generation else {
      publishedSegmentCount = 0
      return
    }
    for work in state.pending { work.isolation.cancel() }
    if let directory = state.directory {
      try? FileManager.default.removeItem(at: directory)
    }
    state.directory = nil
    state.segments.removeAll(keepingCapacity: false)
    state.publishedBytes = 0
    state.pending.removeAll(keepingCapacity: false)
    state.errorMessage = nil
    generation = state
    publishedSegmentCount = 0
    lastError = ""
  }

  private func cancelWorkAndRemoveMedia(revokeTickets: Bool) {
    activeTask?.cancel()
    activeTask = nil
    activeTaskID = nil
    activePending?.isolation.cancel()
    activePending = nil
    if let pending = generation?.pending {
      for work in pending { work.isolation.cancel() }
    }
    if let directory = generation?.directory {
      try? FileManager.default.removeItem(at: directory)
    }
    if revokeTickets { tickets.removeAll(keepingCapacity: false) }
  }

  private nonisolated static func mux(
    _ pending: PendingSegment,
    isolatedURL: URL
  ) async throws {
    let part = pending.finalURL.appendingPathExtension("part")
    try? FileManager.default.removeItem(at: part)
    try? FileManager.default.removeItem(at: pending.finalURL)
    let start = String(
      format: "%.9f", locale: Locale(identifier: "en_US_POSIX"), pending.startSeconds)
    let duration = String(
      format: "%.9f",
      locale: Locale(identifier: "en_US_POSIX"),
      max(0.001, pending.endSeconds - pending.startSeconds)
    )
    let process = RemoteStreamingFFmpegProcess(
      executable: pending.ffmpegURL,
      arguments: [
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", isolatedURL.path,
        "-ss", start, "-t", duration, "-i", pending.sourceURL.path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-bsf:v", "h264_mp4toannexb",
        "-c:a", "aac", "-b:a", "192k",
        // Each two-second source slice is encoded independently. Compensate
        // the one-frame AAC priming delay so adjacent HLS segments keep their
        // audio timestamps aligned with the copied restored video timeline.
        "-af", "asetpts=PTS-STARTPTS+1024/SR/TB",
        "-shortest", "-output_ts_offset", start,
        "-mpegts_flags", "+resend_headers",
        "-muxdelay", "0", "-muxpreload", "0",
        "-f", "mpegts", part.path,
      ]
    )
    do {
      try await process.run()
      try Task.checkCancellation()
      try FileManager.default.moveItem(at: part, to: pending.finalURL)
    } catch {
      try? FileManager.default.removeItem(at: part)
      throw error
    }
  }
}

private enum RemoteStreamingProcessError: LocalizedError {
  case launch(String)
  case failed(Int32, String)
  case emptyOutput

  var errorDescription: String? {
    switch self {
    case .launch(let message):
      return "HLS muxer could not start: \(message)"
    case .failed(let status, let message):
      return "HLS muxer failed (\(status)): \(message)"
    case .emptyOutput:
      return "HLS muxer produced an empty segment"
    }
  }
}

/// Cancellation-aware Process wrapper used by one serialized remux at a time.
private final class RemoteStreamingFFmpegProcess: @unchecked Sendable {
  private let process = Process()
  private let output = Pipe()
  private let lock = NSLock()
  private var continuation: CheckedContinuation<Void, Error>?
  private var cancelled = false
  private var finished = false

  init(executable: URL, arguments: [String]) {
    process.executableURL = executable
    process.arguments = arguments
    process.standardOutput = output
    process.standardError = output
  }

  func run() async throws {
    try await withTaskCancellationHandler(
      operation: {
        try await withCheckedThrowingContinuation {
          (continuation: CheckedContinuation<Void, Error>) in
          lock.lock()
          if cancelled {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return
          }
          self.continuation = continuation
          process.terminationHandler = { [weak self] completed in
            let data = self?.output.fileHandleForReading.readDataToEndOfFile() ?? Data()
            let text =
              String(data: data, encoding: .utf8)?
              .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            self?.complete(
              completed.terminationStatus == 0
                ? .success(())
                : .failure(
                  RemoteStreamingProcessError.failed(
                    completed.terminationStatus,
                    text
                  )
                )
            )
          }
          do {
            try process.run()
            lock.unlock()
          } catch {
            lock.unlock()
            complete(.failure(RemoteStreamingProcessError.launch(error.localizedDescription)))
          }
        }
      },
      onCancel: { self.cancel() }
    )
  }

  func cancel() {
    lock.lock()
    cancelled = true
    let running = process.isRunning
    let shouldComplete = !running && continuation != nil && !finished
    lock.unlock()
    if running { process.terminate() }
    if shouldComplete { complete(.failure(CancellationError())) }
  }

  private func complete(_ result: Result<Void, Error>) {
    lock.lock()
    guard !finished, let continuation else {
      lock.unlock()
      return
    }
    finished = true
    self.continuation = nil
    let wasCancelled = cancelled
    lock.unlock()
    if wasCancelled {
      continuation.resume(throwing: CancellationError())
    } else {
      continuation.resume(with: result)
    }
  }
}
