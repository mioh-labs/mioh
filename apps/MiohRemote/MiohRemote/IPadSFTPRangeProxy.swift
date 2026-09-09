import Foundation
import MiohSFTPKit
import Network

enum IPadSFTPRangeProxyError: LocalizedError {
  case invalidFile
  case listenerFailed
  case stopped

  var errorDescription: String? {
    switch self {
    case .invalidFile:
      "SFTP上の動画をストリーミング用に開けませんでした。"
    case .listenerFailed:
      "端末内のSFTPストリーミング接続を開始できませんでした。"
    case .stopped:
      "SFTPストリーミングは停止済みです。"
    }
  }
}

struct IPadSFTPStreamingMetrics: Sendable {
  let bitsPerSecond: Double
  let activeRangeReads: Int
  let cachedBytes: Int
  let completedRangeReads: UInt64
  let lastRangeLatencySeconds: Double
}

/// Bounds detached single-flight page work even if AVFoundation rapidly seeks
/// and cancels its HTTP requests. Waiting consumers are cancellation-aware.
private actor IPadSFTPRangePageCapacity {
  private static let maximumActivePages = 4
  private var activePages = 0
  private var waiterOrder: [UUID] = []
  private var waiters: [UUID: CheckedContinuation<Void, Error>] = [:]
  private var isStopped = false

  nonisolated func acquire() async throws {
    let id = UUID()
    try Task.checkCancellation()
    try await withTaskCancellationHandler {
      try await enqueue(id)
    } onCancel: {
      Task { await self.cancel(id) }
    }
  }

  private func enqueue(_ id: UUID) async throws {
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Void, Error>) in
      if isStopped || Task.isCancelled {
        continuation.resume(throwing: CancellationError())
      } else if activePages < Self.maximumActivePages {
        activePages += 1
        continuation.resume()
      } else {
        waiterOrder.append(id)
        waiters[id] = continuation
      }
    }
  }

  func release() {
    while let id = waiterOrder.first {
      waiterOrder.removeFirst()
      guard let continuation = waiters.removeValue(forKey: id) else { continue }
      continuation.resume()
      return
    }
    activePages = max(0, activePages - 1)
  }

  func stop() {
    isStopped = true
    let continuations = Array(waiters.values)
    waiters.removeAll()
    waiterOrder.removeAll()
    for continuation in continuations {
      continuation.resume(throwing: CancellationError())
    }
  }

  private func cancel(_ id: UUID) {
    guard let continuation = waiters.removeValue(forKey: id) else { return }
    waiterOrder.removeAll { $0 == id }
    continuation.resume(throwing: CancellationError())
  }
}

/// Shared 32 MiB LRU for the audible source player and the repeated two-second
/// restoration readers. Half-MiB pages keep enough data in flight for LAN
/// throughput while limiting the stale work a newly requested seek can inherit.
private actor IPadSFTPRangePageSource {
  private static let pageBytes = 512 * 1_024
  private static let maximumCacheBytes = 32 * 1_024 * 1_024

  private struct CacheEntry {
    let data: Data
    var lastAccess: UInt64
  }

  private struct InFlightPage {
    let id: UUID
    let startedAt: Double
    let task: Task<Data, Error>
    var waiters: [UUID: CheckedContinuation<Data, Error>]
  }

  private struct TransferSample {
    let startedAt: Double
    let endedAt: Double
    let byteCount: Int
  }

  private let session: MiohSFTPSession
  private let file: MiohSFTPStreamingFile
  private let capacity = IPadSFTPRangePageCapacity()
  private var cache: [UInt64: CacheEntry] = [:]
  private var inFlight: [UInt64: InFlightPage] = [:]
  private var cacheBytes = 0
  private var accessCounter: UInt64 = 0
  private var transferSamples: [TransferSample] = []
  private var completedRangeReads: UInt64 = 0
  private var lastRangeLatencySeconds = 0.0
  private var isStopped = false

  init(session: MiohSFTPSession, file: MiohSFTPStreamingFile) {
    self.session = session
    self.file = file
  }

  func read(offset: UInt64, length: Int) async throws -> Data {
    try Task.checkCancellation()
    guard !isStopped, length > 0,
      length <= MiohSFTPSession.maximumRangeReadBytes,
      offset < UInt64(file.byteCount),
      UInt64(length) <= UInt64(file.byteCount) - offset
    else { throw IPadSFTPRangeProxyError.invalidFile }

    var result = Data()
    result.reserveCapacity(length)
    let end = offset + UInt64(length)
    var cursor = offset
    while cursor < end {
      try Task.checkCancellation()
      let pageOffset = cursor / UInt64(Self.pageBytes) * UInt64(Self.pageBytes)
      let page = try await page(at: pageOffset)
      try Task.checkCancellation()
      let startInPage = Int(cursor - pageOffset)
      guard startInPage < page.count else {
        throw IPadSFTPRangeProxyError.invalidFile
      }
      let copyCount = min(page.count - startInPage, Int(end - cursor))
      result.append(page[startInPage..<(startInPage + copyCount)])
      cursor += UInt64(copyCount)
    }
    guard result.count == length else {
      throw IPadSFTPRangeProxyError.invalidFile
    }
    return result
  }

  func stop() async {
    guard !isStopped else { return }
    isStopped = true
    let loads = Array(inFlight.values)
    let tasks = loads.map(\.task)
    let waiters = loads.flatMap { $0.waiters.values }
    inFlight.removeAll()
    cache.removeAll()
    cacheBytes = 0
    await capacity.stop()
    for waiter in waiters { waiter.resume(throwing: CancellationError()) }
    for task in tasks { task.cancel() }
    for task in tasks { _ = try? await task.value }
  }

  private func page(at offset: UInt64) async throws -> Data {
    let waiterID = UUID()
    let data = try await withTaskCancellationHandler {
      try await subscribeToPage(at: offset, waiterID: waiterID)
    } onCancel: {
      Task { await self.cancelPageWaiter(
        at: offset,
        waiterID: waiterID
      ) }
    }
    guard data.count == min(
      Self.pageBytes,
      Int(UInt64(file.byteCount) - offset)
    ) else { throw IPadSFTPRangeProxyError.invalidFile }
    try Task.checkCancellation()
    return data
  }

  /// Cache lookup, load creation and waiter registration are one actor
  /// operation. That removes the old create-then-subscribe gap, so the last
  /// cancelled seek waiter can safely retire queued work immediately.
  private func subscribeToPage(
    at offset: UInt64,
    waiterID: UUID
  ) async throws -> Data {
    try Task.checkCancellation()
    accessCounter &+= 1
    if var entry = cache[offset] {
      entry.lastAccess = accessCounter
      cache[offset] = entry
      return entry.data
    }
    if var existing = inFlight[offset] {
      return try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Data, Error>) in
        existing.waiters[waiterID] = continuation
        inFlight[offset] = existing
      }
    }

    try await capacity.acquire()
    var mustReleaseCapacityOnError = true
    do {
      try Task.checkCancellation()
      guard !isStopped else { throw CancellationError() }
      if var entry = cache[offset] {
        entry.lastAccess = accessCounter
        cache[offset] = entry
        await capacity.release()
        mustReleaseCapacityOnError = false
        return entry.data
      }
      if inFlight[offset] != nil {
        await capacity.release()
        mustReleaseCapacityOnError = false
        return try await subscribeToPage(at: offset, waiterID: waiterID)
      }
      let remaining = UInt64(file.byteCount) - offset
      let length = Int(min(UInt64(Self.pageBytes), remaining))
      let session = session
      let file = file
      let startedAt = ProcessInfo.processInfo.systemUptime
      let task = Task {
        try await session.readMovieRange(
          file: file,
          offset: offset,
          length: length
        )
      }
      let loadID = UUID()
      mustReleaseCapacityOnError = false
      return try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Data, Error>) in
        let load = InFlightPage(
          id: loadID,
          startedAt: startedAt,
          task: task,
          waiters: [waiterID: continuation]
        )
        inFlight[offset] = load
        Task { [weak self] in
          let result = await task.result
          await self?.finishPage(offset: offset, id: loadID, result: result)
        }
      }
    } catch {
      if mustReleaseCapacityOnError { await capacity.release() }
      throw error
    }
  }

  private func cancelPageWaiter(
    at offset: UInt64,
    waiterID: UUID
  ) {
    guard var load = inFlight[offset],
      let waiter = load.waiters.removeValue(forKey: waiterID)
    else { return }
    // Keep a waiter-less page in the bounded four-page set. SFTP has no
    // request-level abort, and cancelling the Swift task while its READ replies
    // are still in flight used to make a later seek queue behind abandoned
    // work. Let this one page finish and become reusable cache instead; all four
    // allowed pages execute concurrently through SFTP request IDs.
    inFlight[offset] = load
    waiter.resume(throwing: CancellationError())
  }

  private func finishPage(
    offset: UInt64,
    id: UUID,
    result: Result<Data, Error>
  ) async {
    guard let load = inFlight[offset], load.id == id else { return }
    inFlight.removeValue(forKey: offset)
    let expectedCount = min(
      Self.pageBytes,
      Int(UInt64(file.byteCount) - offset)
    )
    let delivered: Result<Data, Error>
    if !isStopped, case .success(let data) = result,
      data.count == expectedCount
    {
      let endedAt = ProcessInfo.processInfo.systemUptime
      let latency = max(0.001, endedAt - load.startedAt)
      transferSamples.append(
        TransferSample(
          startedAt: load.startedAt,
          endedAt: endedAt,
          byteCount: data.count
        )
      )
      completedRangeReads &+= 1
      lastRangeLatencySeconds = latency
      trimTransferSamples(now: endedAt)
      accessCounter &+= 1
      if cache[offset] == nil {
        cache[offset] = CacheEntry(data: data, lastAccess: accessCounter)
        cacheBytes += data.count
        evictIfNeeded()
      }
      delivered = .success(data)
    } else if isStopped {
      delivered = .failure(CancellationError())
    } else if case .failure(let error) = result {
      delivered = .failure(error)
    } else {
      delivered = .failure(IPadSFTPRangeProxyError.invalidFile)
    }
    for waiter in load.waiters.values { waiter.resume(with: delivered) }
    await capacity.release()
  }

  private func evictIfNeeded() {
    while cacheBytes > Self.maximumCacheBytes,
      let oldest = cache.min(by: { $0.value.lastAccess < $1.value.lastAccess })
    {
      cache.removeValue(forKey: oldest.key)
      cacheBytes -= oldest.value.data.count
    }
  }

  func metrics() -> IPadSFTPStreamingMetrics {
    let now = ProcessInfo.processInfo.systemUptime
    trimTransferSamples(now: now)
    let totalBytes = transferSamples.reduce(0) { $0 + $1.byteCount }
    let firstStart = transferSamples.map(\.startedAt).min() ?? now
    let lastEnd = transferSamples.map(\.endedAt).max() ?? now
    let wallSeconds = max(0.001, lastEnd - firstStart)
    let bitsPerSecond = transferSamples.isEmpty
      ? 0 : Double(totalBytes) * 8 / wallSeconds
    return IPadSFTPStreamingMetrics(
      bitsPerSecond: bitsPerSecond,
      activeRangeReads: inFlight.count,
      cachedBytes: cacheBytes,
      completedRangeReads: completedRangeReads,
      lastRangeLatencySeconds: lastRangeLatencySeconds
    )
  }

  private func trimTransferSamples(now: Double) {
    let cutoff = now - 5
    transferSamples.removeAll { $0.endedAt < cutoff }
    if transferSamples.count > 64 {
      transferSamples.removeFirst(transferSamples.count - 64)
    }
  }
}

/// Retains the pinned SFTP connection and its loopback-only Range bridge for
/// exactly as long as the selected streaming input remains active.
final class IPadSFTPStreamingInput: @unchecked Sendable {
  let localURL: URL
  let displayName: String
  let byteCount: Int64
  let rangeValidator: String

  private let proxy: IPadSFTPRangeProxy

  private init(
    localURL: URL,
    displayName: String,
    byteCount: Int64,
    rangeValidator: String,
    proxy: IPadSFTPRangeProxy
  ) {
    self.localURL = localURL
    self.displayName = displayName
    self.byteCount = byteCount
    self.rangeValidator = rangeValidator
    self.proxy = proxy
  }

  static func start(
    configuration: MiohSFTPConfiguration,
    trustedHostKey: String,
    entry: MiohSFTPEntry
  ) async throws -> IPadSFTPStreamingInput {
    let proxy = IPadSFTPRangeProxy(
      configuration: configuration,
      trustedHostKey: trustedHostKey,
      entry: entry
    )
    let source = try await proxy.start()
    return IPadSFTPStreamingInput(
      localURL: source.url,
      displayName: entry.name,
      byteCount: source.byteCount,
      rangeValidator: source.rangeValidator,
      proxy: proxy
    )
  }

  func stop() {
    proxy.stop()
  }

  /// A playback seek starts a new AVFoundation generation while retaining the
  /// authenticated SFTP session and its bounded page cache. Drop every
  /// loopback request owned by the previous generation before the new player
  /// and reader issue ranges, so a second seek cannot queue behind stale HTTP
  /// bodies that AVFoundation has not torn down yet.
  func prepareForSeek() {
    proxy.cancelActiveRequestsForSeek()
  }

  func metrics() async -> IPadSFTPStreamingMetrics {
    await proxy.metrics()
  }

  deinit {
    proxy.stop()
  }
}

/// Serves one already-authorized regular SFTP movie over an opaque IPv4
/// loopback URL. Bodies are sent in 256 KiB pages; neither a complete response
/// nor a complete movie is buffered or written to a temporary file.
private final class IPadSFTPRangeProxy: @unchecked Sendable {
  private static let maximumHeaderBytes = 32 * 1_024
  private static let maximumConnections = 12
  private static let responseChunkBytes = 256 * 1_024

  private struct Request {
    let method: String
    let rangeHeader: String?
  }

  private struct ResponseRange {
    let lowerBound: UInt64
    let upperBound: UInt64
    let isPartial: Bool

    var byteCount: UInt64 { upperBound - lowerBound + 1 }
  }

  private var initialConfiguration: MiohSFTPConfiguration?
  private let trustedHostKey: String
  private let entry: MiohSFTPEntry
  private let token = UUID().uuidString.lowercased().replacingOccurrences(
    of: "-",
    with: ""
  )
  private let rangeValidator = (
    UUID().uuidString + UUID().uuidString
  ).lowercased().replacingOccurrences(of: "-", with: "")
  private let session = MiohSFTPSession()
  private let queue = DispatchQueue(
    label: "com.mioh.sftp-range-proxy",
    qos: .userInitiated
  )
  private let lock = NSLock()
  private var listener: NWListener?
  private var listenerPort: UInt16?
  private var startContinuations: [CheckedContinuation<Void, Error>] = []
  private var connections: [ObjectIdentifier: NWConnection] = [:]
  private var requestTasks: [ObjectIdentifier: Task<Void, Never>] = [:]
  private var headerTimeouts: [ObjectIdentifier: DispatchWorkItem] = [:]
  private var streamingFile: MiohSFTPStreamingFile?
  private var pageSource: IPadSFTPRangePageSource?
  private var isStopped = false

  init(
    configuration: MiohSFTPConfiguration,
    trustedHostKey: String,
    entry: MiohSFTPEntry
  ) {
    initialConfiguration = configuration
    self.trustedHostKey = trustedHostKey
    self.entry = entry
  }

  deinit {
    stop()
  }

  func start() async throws -> (
    url: URL,
    byteCount: Int64,
    rangeValidator: String
  ) {
    guard entry.kind == .movie else { throw IPadSFTPRangeProxyError.invalidFile }
    try Task.checkCancellation()
    do {
      guard let configuration = takeInitialConfiguration() else {
        throw IPadSFTPRangeProxyError.stopped
      }
      _ = try await session.connect(
        configuration: configuration,
        trustedHostKey: trustedHostKey
      )
      let file = try await session.openMovieForStreaming(
        remotePath: entry.path,
        expectedByteCount: entry.byteCount
      )
      guard file.byteCount > 0,
        file.byteCount <= MiohSFTPTransferPolicy.hardMaximumBytes
      else { throw IPadSFTPRangeProxyError.invalidFile }
      setStreamingFile(file)
      try await startListener()
      try Task.checkCancellation()

      let (port, stopped) = listenerSnapshot()
      guard !stopped, let port,
        let url = URL(
          string: "http://127.0.0.1:\(port)/v1/\(token)/input.\(file.pathExtension)"
        )
      else { throw IPadSFTPRangeProxyError.listenerFailed }
      return (
        url: url,
        byteCount: file.byteCount,
        rangeValidator: rangeValidator
      )
    } catch {
      stop()
      throw error
    }
  }

  func stop() {
    lock.lock()
    guard !isStopped else {
      lock.unlock()
      return
    }
    isStopped = true
    initialConfiguration = nil
    let listener = listener
    self.listener = nil
    listenerPort = nil
    let connections = Array(connections.values)
    self.connections.removeAll()
    let tasks = Array(requestTasks.values)
    requestTasks.removeAll()
    let timeouts = Array(headerTimeouts.values)
    headerTimeouts.removeAll()
    let waiters = startContinuations
    startContinuations.removeAll()
    streamingFile = nil
    let pageSource = pageSource
    self.pageSource = nil
    lock.unlock()

    listener?.cancel()
    for connection in connections { connection.cancel() }
    for timeout in timeouts { timeout.cancel() }
    for task in tasks { task.cancel() }
    for waiter in waiters {
      waiter.resume(throwing: IPadSFTPRangeProxyError.stopped)
    }

    let session = session
    // This session is dedicated to playback. Schedule its channel abort before
    // any joins so background/lock cannot wait on a stalled 30-second read.
    let disconnectTask = Task.detached(priority: .userInitiated) {
      await session.disconnect()
    }
    Task.detached(priority: .userInitiated) {
      if let pageSource { await pageSource.stop() }
      for task in tasks { await task.value }
      await disconnectTask.value
    }
  }

  /// Synchronously detaches all currently registered loopback requests. New
  /// connections accepted after this method returns belong to the new seek
  /// generation and are left alone. Cancelling the server tasks also removes
  /// their page waiters, which gives the next requested range priority without
  /// closing the pinned SSH connection or discarding useful cached pages.
  func cancelActiveRequestsForSeek() {
    lock.lock()
    guard !isStopped else {
      lock.unlock()
      return
    }
    let activeConnections = Array(connections.values)
    connections.removeAll()
    let activeTasks = Array(requestTasks.values)
    requestTasks.removeAll()
    let activeTimeouts = Array(headerTimeouts.values)
    headerTimeouts.removeAll()
    lock.unlock()

    for timeout in activeTimeouts { timeout.cancel() }
    for task in activeTasks { task.cancel() }
    for connection in activeConnections { connection.cancel() }
  }

  func metrics() async -> IPadSFTPStreamingMetrics {
    guard let source = pageSourceSnapshot() else {
      return IPadSFTPStreamingMetrics(
        bitsPerSecond: 0,
        activeRangeReads: 0,
        cachedBytes: 0,
        completedRangeReads: 0,
        lastRangeLatencySeconds: 0
      )
    }
    return await source.metrics()
  }

  private func pageSourceSnapshot() -> IPadSFTPRangePageSource? {
    lock.lock()
    let source = pageSource
    lock.unlock()
    return source
  }

  private func takeInitialConfiguration() -> MiohSFTPConfiguration? {
    lock.lock()
    let configuration = initialConfiguration
    initialConfiguration = nil
    lock.unlock()
    return configuration
  }

  private func startListener() async throws {
    try await withTaskCancellationHandler(
      operation: {
        try await withCheckedThrowingContinuation {
          (continuation: CheckedContinuation<Void, Error>) in
          self.beginListenerStart(continuation)
        }
      },
      onCancel: { self.stop() }
    )
  }

  private func setStreamingFile(_ file: MiohSFTPStreamingFile) {
    lock.lock()
    streamingFile = file
    pageSource = IPadSFTPRangePageSource(session: session, file: file)
    lock.unlock()
  }

  private func listenerSnapshot() -> (UInt16?, Bool) {
    lock.lock()
    let snapshot = (listenerPort, isStopped)
    lock.unlock()
    return snapshot
  }

  private func beginListenerStart(
    _ continuation: CheckedContinuation<Void, Error>
  ) {
    lock.lock()
    if isStopped {
      lock.unlock()
      continuation.resume(throwing: IPadSFTPRangeProxyError.stopped)
      return
    }
    if listenerPort != nil {
      lock.unlock()
      continuation.resume()
      return
    }
    startContinuations.append(continuation)
    guard listener == nil else {
      lock.unlock()
      return
    }
    do {
      let parameters = NWParameters.tcp
      parameters.allowLocalEndpointReuse = true
      parameters.requiredLocalEndpoint = .hostPort(
        host: "127.0.0.1",
        port: .any
      )
      let newListener = try NWListener(using: parameters, on: .any)
      listener = newListener
      lock.unlock()
      newListener.stateUpdateHandler = { [weak self, weak newListener] state in
        self?.listenerChanged(state, expected: newListener)
      }
      newListener.newConnectionHandler = { [weak self] connection in
        self?.accept(connection)
      }
      newListener.start(queue: queue)
    } catch {
      let waiters = startContinuations
      startContinuations.removeAll()
      listener = nil
      listenerPort = nil
      lock.unlock()
      for waiter in waiters {
        waiter.resume(throwing: IPadSFTPRangeProxyError.listenerFailed)
      }
    }
  }

  private func listenerChanged(
    _ state: NWListener.State,
    expected: NWListener?
  ) {
    switch state {
    case .ready:
      lock.lock()
      guard !isStopped, listener === expected, let port = listener?.port else {
        lock.unlock()
        return
      }
      listenerPort = port.rawValue
      let waiters = startContinuations
      startContinuations.removeAll()
      lock.unlock()
      for waiter in waiters { waiter.resume() }
    case .failed, .cancelled:
      let waiters = takeStartWaiters(resetListener: true)
      for waiter in waiters {
        waiter.resume(throwing: IPadSFTPRangeProxyError.listenerFailed)
      }
    default:
      break
    }
  }

  private func takeStartWaiters(
    resetListener: Bool
  ) -> [CheckedContinuation<Void, Error>] {
    lock.lock()
    let waiters = startContinuations
    startContinuations.removeAll()
    if resetListener {
      listener?.cancel()
      listener = nil
      listenerPort = nil
    }
    lock.unlock()
    return waiters
  }

  private func accept(_ connection: NWConnection) {
    let id = ObjectIdentifier(connection)
    lock.lock()
    guard !isStopped, connections.count < Self.maximumConnections else {
      lock.unlock()
      connection.cancel()
      return
    }
    connections[id] = connection
    let timeout = DispatchWorkItem { [weak self, weak connection] in
      guard let self, let connection else { return }
      self.expireHeader(on: connection)
    }
    headerTimeouts[id] = timeout
    lock.unlock()

    connection.stateUpdateHandler = { [weak self, weak connection] state in
      guard let self, let connection else { return }
      if case .failed = state { self.cancelConnection(connection) }
      if case .cancelled = state { self.cancelConnection(connection) }
    }
    connection.start(queue: queue)
    queue.asyncAfter(deadline: .now() + 5, execute: timeout)
    receiveHeader(on: connection, accumulated: Data())
  }

  private func receiveHeader(on connection: NWConnection, accumulated: Data) {
    let remaining = Self.maximumHeaderBytes - accumulated.count
    guard remaining > 0 else {
      rejectPendingHeader(431, on: connection)
      return
    }
    connection.receive(
      minimumIncompleteLength: 1,
      maximumLength: min(4_096, remaining)
    ) { [weak self, weak connection] data, _, complete, error in
      guard let self, let connection else { return }
      if error != nil {
        self.cancelConnection(connection)
        return
      }
      var header = accumulated
      if let data { header.append(data) }
      if let range = header.range(of: Data("\r\n\r\n".utf8)) {
        self.processHeader(Data(header[..<range.upperBound]), on: connection)
      } else if complete {
        self.rejectPendingHeader(400, on: connection)
      } else {
        self.receiveHeader(on: connection, accumulated: header)
      }
    }
  }

  private func processHeader(_ data: Data, on connection: NWConnection) {
    let id = ObjectIdentifier(connection)
    lock.lock()
    let timeout = headerTimeouts.removeValue(forKey: id)
    let stopped = isStopped
    guard let timeout, !stopped else {
      lock.unlock()
      if stopped { connection.cancel() }
      return
    }
    timeout.cancel()
    let file = streamingFile
    let pageSource = pageSource
    lock.unlock()
    guard let request = parseRequest(data), let file, let pageSource else {
      sendError(400, on: connection)
      return
    }

    lock.lock()
    if isStopped || connections[id] !== connection {
      lock.unlock()
      connection.cancel()
    } else {
      // Create while holding the registry lock. A very short HEAD response can
      // otherwise finish and remove itself before it has been inserted.
      let task = Task { [weak self, weak connection] in
        guard let self, let connection else { return }
        defer { self.completeConnection(connection) }
        do {
          try await self.serve(
            request,
            file: file,
            pageSource: pageSource,
            on: connection
          )
        } catch is CancellationError {
          connection.cancel()
        } catch {
          // A 200/206 header may already be on the wire. Appending an HTTP
          // error would turn it into apparently valid media bytes.
          connection.cancel()
        }
      }
      requestTasks[id] = task
      lock.unlock()
    }
  }

  private func serve(
    _ request: Request,
    file: MiohSFTPStreamingFile,
    pageSource: IPadSFTPRangePageSource,
    on connection: NWConnection
  ) async throws {
    try Task.checkCancellation()
    let total = UInt64(file.byteCount)
    guard let range = Self.responseRange(
      request.rangeHeader,
      totalBytes: total
    ) else {
      try await send(
        Self.responseHeader(
          status: 416,
          contentType: contentType(for: file.pathExtension),
          contentLength: 0,
          contentRange: "bytes */\(total)",
          entityTag: rangeValidator
        ),
        on: connection
      )
      return
    }
    let status = range.isPartial ? 206 : 200
    let contentRange = range.isPartial
      ? "bytes \(range.lowerBound)-\(range.upperBound)/\(total)" : nil
    try await send(
      Self.responseHeader(
        status: status,
        contentType: contentType(for: file.pathExtension),
        contentLength: range.byteCount,
        contentRange: contentRange,
        entityTag: rangeValidator
      ),
      on: connection
    )
    guard request.method == "GET" else { return }

    var offset = range.lowerBound
    while offset <= range.upperBound {
      try Task.checkCancellation()
      let remaining = range.upperBound - offset + 1
      let count = Int(min(UInt64(Self.responseChunkBytes), remaining))
      let body = try await pageSource.read(offset: offset, length: count)
      guard body.count == count else {
        throw IPadSFTPRangeProxyError.invalidFile
      }
      try await send(body, on: connection)
      offset += UInt64(body.count)
    }
  }

  private func parseRequest(_ data: Data) -> Request? {
    guard let header = String(data: data, encoding: .utf8),
      header.rangeOfCharacter(
        from: .controlCharacters.subtracting(
          CharacterSet(charactersIn: "\r\n\t")
        )
      ) == nil
    else { return nil }
    let lines = header.components(separatedBy: "\r\n")
    guard let requestLine = lines.first else { return nil }
    let parts = requestLine.split(separator: " ", omittingEmptySubsequences: true)
    guard parts.count == 3,
      parts[0] == "GET" || parts[0] == "HEAD",
      parts[2] == "HTTP/1.1"
    else { return nil }

    lock.lock()
    let port = listenerPort
    let file = streamingFile
    lock.unlock()
    guard let port, let file,
      parts[1] == "/v1/\(token)/input.\(file.pathExtension)"
    else { return nil }

    var hostCount = 0
    var rangeHeader: String?
    for line in lines.dropFirst() where !line.isEmpty {
      guard let colon = line.firstIndex(of: ":") else { return nil }
      let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
      let value = line[line.index(after: colon)...]
        .trimmingCharacters(in: .whitespaces)
      if name == "host" {
        hostCount += 1
        guard value.lowercased() == "127.0.0.1:\(port)" else { return nil }
      } else if name == "range" {
        guard rangeHeader == nil, Self.isSafeRangeHeader(value) else { return nil }
        rangeHeader = value
      }
    }
    guard hostCount == 1 else { return nil }
    return Request(method: String(parts[0]), rangeHeader: rangeHeader)
  }

  private static func isSafeRangeHeader(_ value: String) -> Bool {
    guard value.utf8.count <= 128, value.hasPrefix("bytes="),
      !value.contains(",")
    else { return false }
    let bounds = value.dropFirst(6).split(
      separator: "-",
      maxSplits: 1,
      omittingEmptySubsequences: false
    )
    guard bounds.count == 2, !bounds[0].isEmpty || !bounds[1].isEmpty else {
      return false
    }
    return bounds.allSatisfy { $0.allSatisfy(\.isNumber) }
  }

  private static func responseRange(
    _ header: String?,
    totalBytes: UInt64
  ) -> ResponseRange? {
    guard totalBytes > 0 else { return nil }
    guard let header else {
      return ResponseRange(
        lowerBound: 0,
        upperBound: totalBytes - 1,
        isPartial: false
      )
    }
    guard isSafeRangeHeader(header) else { return nil }
    let bounds = header.dropFirst(6).split(
      separator: "-",
      maxSplits: 1,
      omittingEmptySubsequences: false
    )
    guard bounds.count == 2 else { return nil }
    if bounds[0].isEmpty {
      guard let suffix = UInt64(bounds[1]), suffix > 0 else { return nil }
      let count = min(suffix, totalBytes)
      return ResponseRange(
        lowerBound: totalBytes - count,
        upperBound: totalBytes - 1,
        isPartial: true
      )
    }
    guard let lower = UInt64(bounds[0]), lower < totalBytes else { return nil }
    let upper: UInt64
    if bounds[1].isEmpty {
      upper = totalBytes - 1
    } else {
      guard let requestedUpper = UInt64(bounds[1]), requestedUpper >= lower else {
        return nil
      }
      upper = min(requestedUpper, totalBytes - 1)
    }
    return ResponseRange(
      lowerBound: lower,
      upperBound: upper,
      isPartial: true
    )
  }

  private static func responseHeader(
    status: Int,
    contentType: String,
    contentLength: UInt64,
    contentRange: String?,
    entityTag: String
  ) -> Data {
    let reason: String
    switch status {
    case 200: reason = "OK"
    case 206: reason = "Partial Content"
    case 416: reason = "Range Not Satisfiable"
    default: reason = "Response"
    }
    var lines = [
      "HTTP/1.1 \(status) \(reason)",
      "Content-Type: \(contentType)",
      "Content-Length: \(contentLength)",
      "Accept-Ranges: bytes",
      "ETag: \"\(entityTag)\"",
      "Cache-Control: no-store",
      "Connection: close",
    ]
    if let contentRange { lines.append("Content-Range: \(contentRange)") }
    return Data((lines.joined(separator: "\r\n") + "\r\n\r\n").utf8)
  }

  private func contentType(for pathExtension: String) -> String {
    pathExtension == "mov" ? "video/quicktime" : "video/mp4"
  }

  private func send(_ data: Data, on connection: NWConnection) async throws {
    try Task.checkCancellation()
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, Error>) in
        connection.send(content: data, completion: .contentProcessed { error in
          if error == nil {
            continuation.resume()
          } else {
            continuation.resume(throwing: IPadSFTPRangeProxyError.listenerFailed)
          }
        })
      }
    } onCancel: {
      connection.cancel()
    }
  }

  private func sendError(_ status: Int, on connection: NWConnection) {
    let data = Self.errorResponse(status: status)
    connection.send(content: data, completion: .contentProcessed { _ in
      connection.cancel()
    })
  }

  private static func errorResponse(status: Int) -> Data {
    let reason: String
    switch status {
    case 400: reason = "Bad Request"
    case 408: reason = "Request Timeout"
    case 431: reason = "Request Header Fields Too Large"
    default: reason = "Error"
    }
    let data = Data(
      "HTTP/1.1 \(status) \(reason)\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".utf8
    )
    return data
  }

  private func expireHeader(on connection: NWConnection) {
    rejectPendingHeader(408, on: connection)
  }

  private func rejectPendingHeader(_ status: Int, on connection: NWConnection) {
    let id = ObjectIdentifier(connection)
    lock.lock()
    guard let timeout = headerTimeouts.removeValue(forKey: id),
      connections.removeValue(forKey: id) != nil
    else {
      lock.unlock()
      return
    }
    timeout.cancel()
    lock.unlock()
    sendError(status, on: connection)
  }

  private func cancelConnection(_ connection: NWConnection) {
    let id = ObjectIdentifier(connection)
    lock.lock()
    connections.removeValue(forKey: id)
    headerTimeouts.removeValue(forKey: id)?.cancel()
    let task = requestTasks.removeValue(forKey: id)
    lock.unlock()
    task?.cancel()
    connection.cancel()
  }

  private func completeConnection(_ connection: NWConnection) {
    let id = ObjectIdentifier(connection)
    lock.lock()
    connections.removeValue(forKey: id)
    requestTasks.removeValue(forKey: id)
    headerTimeouts.removeValue(forKey: id)?.cancel()
    lock.unlock()
    connection.cancel()
  }
}
