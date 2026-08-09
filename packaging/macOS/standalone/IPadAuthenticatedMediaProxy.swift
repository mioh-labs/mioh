import Foundation
import Network

enum IPadAuthenticatedMediaProxyError: Error, LocalizedError, Equatable {
  case alreadyStopped
  case invalidConfiguration
  case listenerFailed
  case notStarted
  case unsafeTarget
  case targetLimitExceeded

  var errorDescription: String? {
    switch self {
    case .alreadyStopped:
      "認証メディアプロキシは停止済みです。"
    case .invalidConfiguration:
      "認証メディアプロキシの上限設定が不正です。"
    case .listenerFailed:
      "端末内の再生接続を開始できませんでした。"
    case .notStarted:
      "認証メディアプロキシが開始されていません。"
    case .unsafeTarget:
      "安全な公開HTTPSメディアとして確認できませんでした。"
    case .targetLimitExceeded:
      "認証メディアの安全上限を超えました。"
    }
  }
}

/// Keeps loopback clients queued instead of returning a transient 503 before
/// the shared origin transport gets a chance to apply its host congestion
/// policy.  A permit is transferred directly to the oldest waiter on release,
/// so the configured number of origin responses remains the hard upper bound.
private actor IPadAuthenticatedMediaProxyOriginGate {
  private let maximumActiveRequests: Int
  private var activeRequests = 0
  private var waiterOrder: [UUID] = []
  private var waiters: [UUID: CheckedContinuation<Void, Error>] = [:]
  private var stopped = false

  init(maximumActiveRequests: Int) {
    self.maximumActiveRequests = max(1, maximumActiveRequests)
  }

  nonisolated func acquire() async throws {
    let waiterID = UUID()
    try Task.checkCancellation()
    try await withTaskCancellationHandler(operation: {
      try Task.checkCancellation()
      try await self.enqueue(waiterID: waiterID)
    }, onCancel: {
      Task { await self.cancel(waiterID: waiterID) }
    })
  }

  private func enqueue(waiterID: UUID) async throws {
    try Task.checkCancellation()
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Void, Error>) in
      if stopped {
        continuation.resume(throwing: CancellationError())
      } else if activeRequests < maximumActiveRequests {
        activeRequests += 1
        continuation.resume()
      } else {
        waiterOrder.append(waiterID)
        waiters[waiterID] = continuation
      }
    }
  }

  func release() {
    while let waiterID = waiterOrder.first {
      waiterOrder.removeFirst()
      guard let continuation = waiters.removeValue(forKey: waiterID) else {
        continue
      }
      continuation.resume()
      return
    }
    activeRequests = max(0, activeRequests - 1)
  }

  func stop() {
    stopped = true
    let continuations = Array(waiters.values)
    waiters.removeAll()
    waiterOrder.removeAll()
    for continuation in continuations {
      continuation.resume(throwing: CancellationError())
    }
  }

  private func cancel(waiterID: UUID) {
    guard let continuation = waiters.removeValue(forKey: waiterID) else {
      return
    }
    waiterOrder.removeAll { $0 == waiterID }
    continuation.resume(throwing: CancellationError())
  }
}

/// A short-lived, loopback-only bridge between AVPlayer and authenticated
/// public HTTPS media. AVPlayer sees an opaque local URL; origin URLs and their
/// credentials remain only in this process's in-memory target table.
///
/// This proxy intentionally uses only public APIs. Call `stop()` as soon as the
/// player no longer needs the stream.
final class IPadAuthenticatedMediaProxy: @unchecked Sendable {
  private static let maximumRateLimitRetryCount = 2

  struct Configuration: Sendable, Equatable {
    var maximumConcurrentRequests = 4
    var maximumMappedTargets = 32_768
    var maximumRequestHeaderBytes = 32 * 1_024
    var maximumResponseBytes = 64 * 1_024 * 1_024
    var maximumPlaylistBytes = 2 * 1_024 * 1_024
    var maximumRedirectCount = 6
    var requestTimeout: TimeInterval = 30

    fileprivate var isValid: Bool {
      (1...16).contains(maximumConcurrentRequests)
        && (16...32_768).contains(maximumMappedTargets)
        && (1_024...128 * 1_024).contains(maximumRequestHeaderBytes)
        && (1_024...256 * 1_024 * 1_024).contains(maximumResponseBytes)
        && (1_024...8 * 1_024 * 1_024).contains(maximumPlaylistBytes)
        && maximumPlaylistBytes <= maximumResponseBytes
        && (0...12).contains(maximumRedirectCount)
        && requestTimeout >= 1 && requestTimeout <= 120
    }
  }

  private struct TargetEntry {
    let url: URL
    let context: IPadMediaRequestContext?
    let isPlaylist: Bool
    let resolutionPolicy: IPadMediaURLResolutionPolicy
    let syntheticPlaylist: Data?
  }

  private struct LocalRequest {
    let method: String
    let token: String
    let range: String?
    let hlsDeliveryDirectives: String?
  }

  private struct LocalResponse {
    let statusCode: Int
    let headers: [(String, String)]
    let body: Data
  }

  private enum DiagnosticFailure: String {
    case headerTimeout = "header_timeout"
    case receive = "receive"
    case badRequest = "bad_request"
    case targetMissing = "target_missing"
    case interaction = "interaction"
    case fetchTimeout = "fetch_timeout"
    case fetchNetwork = "fetch_network"
    case fetchRejected = "fetch_rejected"
    case send = "send"
  }

  private struct DiagnosticState {
    var listenerReady = false
    var localURLIssued = false
    var acceptedConnections = 0
    var parsedRequests = 0
    var lastMethod: String?
    var fetchesStarted = 0
    var lastOriginStatus: Int?
    var playlistsDetected = 0
    var playlistsRewritten = 0
    var repliesStarted = 0
    var repliesCompleted = 0
    var lastLocalStatus: Int?
    var lastFailure: DiagnosticFailure?
  }

  private let configuration: Configuration
  private let onInteractionRequired: @Sendable (URL?) -> Void
  private let queue = DispatchQueue(
    label: "com.mioh.authenticated-media-proxy",
    qos: .userInitiated
  )
  private let lock = NSLock()
  private var listener: NWListener?
  private var listenerPort: UInt16?
  private var startContinuations: [CheckedContinuation<Void, Error>] = []
  private var connections: [ObjectIdentifier: NWConnection] = [:]
  private var headerTimeouts: [ObjectIdentifier: DispatchWorkItem] = [:]
  private var requestTasks: [ObjectIdentifier: Task<Void, Never>] = [:]
  private var targets: [String: TargetEntry] = [:]
  private var tokensByURL: [URL: [String]] = [:]
  private let originRequestGate: IPadAuthenticatedMediaProxyOriginGate
  private var didReportInteractionRequired = false
  private var stopped = false
  private var diagnostics = DiagnosticState()

  init(
    configuration: Configuration = Configuration(),
    onInteractionRequired: @escaping @Sendable (URL?) -> Void = { _ in }
  ) {
    self.configuration = configuration
    self.onInteractionRequired = onInteractionRequired
    originRequestGate = IPadAuthenticatedMediaProxyOriginGate(
      maximumActiveRequests: configuration.maximumConcurrentRequests
    )
  }

  deinit {
    stop()
  }

  /// Privacy-safe aggregate state for user-visible playback diagnostics. It
  /// never contains an origin URL, host, token, credential, header, or body.
  func diagnosticSummary() -> String {
    lock.lock()
    let state = diagnostics
    lock.unlock()
    return [
      "待受=\(state.listenerReady ? "可" : "未")",
      "URL=\(state.localURLIssued ? "発行" : "未発行")",
      "接続=\(state.acceptedConnections)",
      "要求=\(state.parsedRequests)(\(state.lastMethod ?? "-"))",
      "取得=\(state.fetchesStarted)",
      "元HTTP=\(state.lastOriginStatus.map(String.init) ?? "-")",
      "書換=\(state.playlistsRewritten)/\(state.playlistsDetected)",
      "応答=\(state.repliesCompleted)/\(state.repliesStarted)"
        + "(\(state.lastLocalStatus.map(String.init) ?? "-"))",
      "原因=\(state.lastFailure?.rawValue ?? "-")",
    ].joined(separator: " ")
  }

  /// Starts an IPv4 loopback listener on an ephemeral port. Concurrent callers
  /// share the same start operation.
  func start() async throws {
    guard configuration.isValid else {
      throw IPadAuthenticatedMediaProxyError.invalidConfiguration
    }
    try Task.checkCancellation()

    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        lock.lock()
        if listenerPort != nil {
          lock.unlock()
          continuation.resume()
          return
        }
        if stopped {
          lock.unlock()
          continuation.resume(
            throwing: IPadAuthenticatedMediaProxyError.alreadyStopped
          )
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
            self?.listenerStateChanged(state, listener: newListener)
          }
          newListener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
          }
          newListener.start(queue: queue)
        } catch {
          let waiters = takeStartContinuationsAndResetListener()
          for waiter in waiters {
            waiter.resume(throwing: IPadAuthenticatedMediaProxyError.listenerFailed)
          }
        }
      }
    } onCancel: {
      self.stop()
    }
  }

  /// Registers an already-approved public HTTPS target and returns an opaque
  /// loopback URL suitable for AVPlayer. No origin URL components are exposed
  /// in the local path or query.
  func localURL(
    for approvedURL: URL,
    context: IPadMediaRequestContext? = nil,
    isPlaylist: Bool = false,
    resolutionPolicy: IPadMediaURLResolutionPolicy = .publicDiscovered
  ) throws -> URL {
    guard
      let safeURL = Self.sanitizedHTTPSURL(
        approvedURL,
        resolutionPolicy: resolutionPolicy
      )
    else {
      throw IPadAuthenticatedMediaProxyError.unsafeTarget
    }

    lock.lock()
    defer { lock.unlock() }
    guard !stopped, let listenerPort else {
      throw IPadAuthenticatedMediaProxyError.notStarted
    }
    let token = try registerTargetLocked(
      url: safeURL,
      context: context,
      isPlaylist: isPlaylist,
      resolutionPolicy: resolutionPolicy
    )
    diagnostics.localURLIssued = true
    let resourceName = isPlaylist ? "index.m3u8" : "resource"
    guard
      let localURL = URL(
        string: "http://127.0.0.1:\(listenerPort)/v1/\(token)/\(resourceName)"
      )
    else {
      throw IPadAuthenticatedMediaProxyError.listenerFailed
    }
    return localURL
  }

  /// Produces a loopback-only one-variant master that pins AVPlayer to the
  /// exact media playlist selected for restoration while retaining that
  /// variant's separate AUDIO rendition group.
  func localURL(
    forSelectedHLSMaster metadata: IPadHLSMasterMetadata,
    context: IPadMediaRequestContext? = nil,
    resolutionPolicy: IPadMediaURLResolutionPolicy = .publicDiscovered
  ) throws -> URL {
    guard
      let safeMasterURL = Self.sanitizedHTTPSURL(
        metadata.masterURL,
        resolutionPolicy: resolutionPolicy
      ),
      let safeVideoURL = Self.sanitizedHTTPSURL(
        metadata.selectedVideoPlaylistURL,
        resolutionPolicy: resolutionPolicy
      )
    else { throw IPadAuthenticatedMediaProxyError.unsafeTarget }

    lock.lock()
    guard !stopped, let listenerPort else {
      lock.unlock()
      throw IPadAuthenticatedMediaProxyError.notStarted
    }
    let existingTokens = Set(targets.keys)
    do {
      let videoToken = try registerTargetLocked(
        url: safeVideoURL,
        context: context,
        isPlaylist: true,
        resolutionPolicy: resolutionPolicy
      )
      guard let localVideoURL = Self.localURL(
        port: listenerPort,
        token: videoToken,
        isPlaylist: true
      ) else { throw IPadAuthenticatedMediaProxyError.listenerFailed }

      var localAudioURLs: [URL: URL] = [:]
      for rendition in metadata.audioRenditions {
        guard let originAudioURL = rendition.url else { continue }
        guard let safeAudioURL = Self.sanitizedHTTPSURL(
          originAudioURL,
          resolutionPolicy: resolutionPolicy
        ) else { throw IPadAuthenticatedMediaProxyError.unsafeTarget }
        let token = try registerTargetLocked(
          url: safeAudioURL,
          context: context,
          isPlaylist: true,
          resolutionPolicy: resolutionPolicy
        )
        guard let localAudioURL = Self.localURL(
          port: listenerPort,
          token: token,
          isPlaylist: true
        ) else { throw IPadAuthenticatedMediaProxyError.listenerFailed }
        localAudioURLs[originAudioURL] = localAudioURL
      }

      let playlist = metadata.syntheticPlaylist(
        videoPlaylistURL: localVideoURL,
        audioPlaylistURLs: localAudioURLs
      )
      let playlistData = Data(playlist.utf8)
      guard playlistData.count <= configuration.maximumPlaylistBytes else {
        throw IPadAuthenticatedMediaProxyError.targetLimitExceeded
      }
      let masterToken = try registerTargetLocked(
        url: safeMasterURL,
        context: context,
        isPlaylist: true,
        resolutionPolicy: resolutionPolicy,
        syntheticPlaylist: playlistData
      )
      diagnostics.localURLIssued = true
      guard let localMasterURL = Self.localURL(
        port: listenerPort,
        token: masterToken,
        isPlaylist: true
      ) else { throw IPadAuthenticatedMediaProxyError.listenerFailed }
      lock.unlock()
      return localMasterURL
    } catch {
      let insertedTokens = targets.keys.filter { !existingTokens.contains($0) }
      for token in insertedTokens { removeTargetLocked(token) }
      lock.unlock()
      throw error
    }
  }

  /// Stops the listener, cancels in-flight work, closes local connections, and
  /// removes every in-memory origin URL and credential context.
  func stop() {
    lock.lock()
    guard !stopped else {
      lock.unlock()
      return
    }
    stopped = true
    let listener = listener
    self.listener = nil
    listenerPort = nil
    let waiters = startContinuations
    startContinuations.removeAll()
    let connections = Array(connections.values)
    self.connections.removeAll()
    let headerTimeouts = Array(headerTimeouts.values)
    self.headerTimeouts.removeAll()
    let tasks = Array(requestTasks.values)
    requestTasks.removeAll()
    targets.removeAll()
    tokensByURL.removeAll()
    lock.unlock()

    listener?.cancel()
    let gate = originRequestGate
    Task { await gate.stop() }
    for connection in connections { connection.cancel() }
    for headerTimeout in headerTimeouts { headerTimeout.cancel() }
    for task in tasks { task.cancel() }
    for waiter in waiters {
      waiter.resume(throwing: IPadAuthenticatedMediaProxyError.alreadyStopped)
    }
  }

  private func reportInteractionRequired(_ url: URL?) {
    lock.lock()
    guard !stopped, !didReportInteractionRequired else {
      lock.unlock()
      return
    }
    didReportInteractionRequired = true
    let handler = onInteractionRequired
    lock.unlock()
    handler(url)
  }

  private func listenerStateChanged(
    _ state: NWListener.State,
    listener expectedListener: NWListener?
  ) {
    switch state {
    case .ready:
      lock.lock()
      guard !stopped, listener === expectedListener, let port = listener?.port else {
        lock.unlock()
        return
      }
      listenerPort = port.rawValue
      diagnostics.listenerReady = true
      let waiters = startContinuations
      startContinuations.removeAll()
      lock.unlock()
      for waiter in waiters { waiter.resume() }
    case .failed:
      let waiters = takeStartContinuationsAndResetListener()
      for waiter in waiters {
        waiter.resume(throwing: IPadAuthenticatedMediaProxyError.listenerFailed)
      }
    case .cancelled:
      lock.lock()
      let wasStarting = listenerPort == nil
      let waiters = wasStarting ? startContinuations : []
      if wasStarting { startContinuations.removeAll() }
      lock.unlock()
      for waiter in waiters {
        waiter.resume(throwing: IPadAuthenticatedMediaProxyError.listenerFailed)
      }
    default:
      break
    }
  }

  private func takeStartContinuationsAndResetListener()
    -> [CheckedContinuation<Void, Error>]
  {
    lock.lock()
    let waiters = startContinuations
    startContinuations.removeAll()
    listener?.cancel()
    listener = nil
    listenerPort = nil
    lock.unlock()
    return waiters
  }

  private func accept(_ connection: NWConnection) {
    let identifier = ObjectIdentifier(connection)
    lock.lock()
    let maximumPendingConnections = max(
      16,
      configuration.maximumConcurrentRequests * 4
    )
    guard !stopped, connections.count < maximumPendingConnections
    else {
      lock.unlock()
      connection.cancel()
      return
    }
    connections[identifier] = connection
    diagnostics.acceptedConnections += 1
    let headerTimeout = DispatchWorkItem { [weak self, weak connection] in
      guard let self, let connection else { return }
      self.recordFailure(.headerTimeout)
      self.finishConnection(connection)
    }
    headerTimeouts[identifier] = headerTimeout
    lock.unlock()

    connection.stateUpdateHandler = { [weak self, weak connection] state in
      guard let self, let connection else { return }
      if case .failed = state { self.finishConnection(connection) }
      if case .cancelled = state { self.finishConnection(connection) }
    }
    connection.start(queue: queue)
    queue.asyncAfter(
      deadline: .now() + min(5, configuration.requestTimeout),
      execute: headerTimeout
    )
    receiveRequestHeader(on: connection, accumulated: Data())
  }

  private func receiveRequestHeader(
    on connection: NWConnection,
    accumulated: Data
  ) {
    let remaining = configuration.maximumRequestHeaderBytes - accumulated.count
    guard remaining > 0 else {
      sendError(statusCode: 431, on: connection)
      return
    }
    connection.receive(
      minimumIncompleteLength: 1,
      maximumLength: min(4_096, remaining)
    ) { [weak self, weak connection] content, _, isComplete, error in
      guard let self, let connection else { return }
      if error != nil {
        self.recordFailure(.receive)
        self.finishConnection(connection)
        return
      }
      var data = accumulated
      if let content { data.append(content) }
      if let headerRange = data.range(of: Data("\r\n\r\n".utf8)) {
        let header = data[..<headerRange.upperBound]
        self.processRequestHeader(Data(header), on: connection)
      } else if isComplete {
        self.sendError(statusCode: 400, on: connection)
      } else {
        self.receiveRequestHeader(on: connection, accumulated: data)
      }
    }
  }

  private func processRequestHeader(_ data: Data, on connection: NWConnection) {
    cancelHeaderTimeout(for: ObjectIdentifier(connection))
    guard let request = parseLocalRequest(data) else {
      recordFailure(.badRequest)
      sendError(statusCode: 400, on: connection)
      return
    }
    recordParsedRequest(method: request.method)
    guard let entry = target(for: request.token) else {
      recordFailure(.targetMissing)
      sendError(statusCode: 404, on: connection)
      return
    }
    guard request.hlsDeliveryDirectives == nil || entry.isPlaylist else {
      recordFailure(.badRequest)
      sendError(statusCode: 400, on: connection)
      return
    }
    let identifier = ObjectIdentifier(connection)
    let task = Task { [weak self, weak connection] in
      guard let self, let connection else { return }
      defer { self.removeTask(for: identifier) }
      do {
        let response = try await self.fetchWithOriginPermit(
          request: request,
          entry: entry
        )
        try Task.checkCancellation()
        self.send(response, on: connection)
      } catch is CancellationError {
        self.finishConnection(connection)
      } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
        self.recordFailure(.interaction)
        self.reportInteractionRequired(challengedURL)
        self.sendError(statusCode: 502, on: connection)
      } catch let error as URLError {
        self.recordFailure(error.code == .timedOut ? .fetchTimeout : .fetchNetwork)
        self.sendError(statusCode: 502, on: connection)
      } catch {
        self.recordFailure(.fetchRejected)
        self.sendError(statusCode: 502, on: connection)
      }
    }
    lock.lock()
    if stopped {
      lock.unlock()
      task.cancel()
      finishConnection(connection)
      return
    }
    requestTasks[identifier] = task
    lock.unlock()
  }

  private func fetch(
    request localRequest: LocalRequest,
    entry: TargetEntry
  ) async throws -> LocalResponse {
    try Task.checkCancellation()
    if let syntheticPlaylist = entry.syntheticPlaylist {
      let headers = [
        ("Content-Type", "application/vnd.apple.mpegurl"),
        ("Content-Length", String(syntheticPlaylist.count)),
        ("Cache-Control", "no-store"),
      ]
      return LocalResponse(
        statusCode: 200,
        headers: headers,
        body: localRequest.method == "HEAD" ? Data() : syntheticPlaylist
      )
    }
    guard
      let originURL = Self.originURL(
        for: entry.url,
        appendingHLSDeliveryDirectives: localRequest.hlsDeliveryDirectives,
        resolutionPolicy: entry.resolutionPolicy
      )
    else {
      throw IPadAuthenticatedMediaProxyError.unsafeTarget
    }

    // Some media origins reject HEAD for playlists, while AVPlayer commonly
    // probes its URL before loading it. Fetch the complete playlist for both
    // HEAD and Range probes so every URI can be rewritten to this proxy.
    let originMethod =
      entry.isPlaylist && localRequest.method == "HEAD"
      ? "GET"
      : localRequest.method
    let originRange = entry.isPlaylist ? nil : localRequest.range

    var request = URLRequest(url: originURL)
    request.httpMethod = originMethod
    request.timeoutInterval = configuration.requestTimeout
    request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    request.httpShouldHandleCookies = false
    request.setValue(
      "application/vnd.apple.mpegurl, application/x-mpegurl, video/*, audio/*, */*;q=0.5",
      forHTTPHeaderField: "Accept"
    )
    if let range = originRange {
      request.setValue(range, forHTTPHeaderField: "Range")
    }
    entry.context?.applying(to: &request)

    let operation = IPadAuthenticatedMediaOriginRequest(
      maximumResponseBytes: configuration.maximumResponseBytes,
      maximumRedirectCount: configuration.maximumRedirectCount,
      timeout: configuration.requestTimeout,
      rangeHeader: originRange,
      method: originMethod,
      requestContext: entry.context,
      hlsDeliveryDirectives: localRequest.hlsDeliveryDirectives,
      resolutionPolicy: entry.resolutionPolicy,
      onResponseStatus: { [weak self] statusCode in
        self?.recordOriginStatus(statusCode)
      }
    )
    let payload = try await operation.start(request)
    if let error = IPadMediaURLResolver.interactionChallengeError(
      response: payload.response
    ) {
      throw error
    }
    await entry.context?.updateCookies(from: payload.response)
    try Task.checkCancellation()
    guard
      let finalURL = Self.sanitizedHTTPSURL(
        payload.response.url,
        resolutionPolicy: entry.resolutionPolicy
      )
    else {
      throw IPadAuthenticatedMediaProxyError.unsafeTarget
    }

    var transformedData = originMethod == "HEAD" ? Data() : payload.data
    var isRewrittenPlaylist = false
    if originMethod == "GET", payload.response.statusCode == 200,
      Self.isHLSPlaylist(transformedData, response: payload.response)
    {
      recordPlaylistDetected()
      guard transformedData.count <= configuration.maximumPlaylistBytes else {
        throw IPadAuthenticatedMediaProxyError.targetLimitExceeded
      }
      transformedData = try rewritePlaylist(
        transformedData,
        relativeTo: finalURL,
        context: entry.context,
        resolutionPolicy: entry.resolutionPolicy
      )
      guard transformedData.count <= configuration.maximumPlaylistBytes else {
        throw IPadAuthenticatedMediaProxyError.targetLimitExceeded
      }
      isRewrittenPlaylist = true
      recordPlaylistRewritten()
    }

    let responseData = localRequest.method == "HEAD" ? Data() : transformedData

    let headers = Self.responseHeaders(
      from: payload.response,
      bodyCount: transformedData.count,
      isHead: localRequest.method == "HEAD",
      isPlaylist: entry.isPlaylist || isRewrittenPlaylist
    )
    return LocalResponse(
      statusCode: payload.response.statusCode,
      headers: headers,
      body: responseData
    )
  }

  private func fetchWithOriginPermit(
    request: LocalRequest,
    entry: TargetEntry
  ) async throws -> LocalResponse {
    try await originRequestGate.acquire()
    do {
      var rateLimitRetryCount = 0
      while true {
        try Task.checkCancellation()
        recordFetchStarted()
        let response = try await fetch(request: request, entry: entry)
        guard response.statusCode == 429,
          request.method == "GET" || request.method == "HEAD",
          rateLimitRetryCount < Self.maximumRateLimitRetryCount
        else {
          await originRequestGate.release()
          return response
        }

        // Shared transport records Retry-After/the fallback host cooldown before
        // completing `fetch`. Re-entering it here therefore waits behind that
        // same cooldown and also keeps restoration prefetch from racing AVPlayer.
        rateLimitRetryCount += 1
      }
    } catch {
      await originRequestGate.release()
      throw error
    }
  }

  private func rewritePlaylist(
    _ data: Data,
    relativeTo baseURL: URL,
    context: IPadMediaRequestContext?,
    resolutionPolicy: IPadMediaURLResolutionPolicy
  ) throws -> Data {
    guard var text = String(data: data, encoding: .utf8) else {
      throw IPadAuthenticatedMediaProxyError.unsafeTarget
    }
    if text.hasPrefix("\u{feff}") { text.removeFirst() }
    let usesCRLF = text.contains("\r\n")
    let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
      .replacingOccurrences(of: "\r", with: "\n")
    let lines = normalized.split(separator: "\n", omittingEmptySubsequences: false)
    var rewrittenLines: [String] = []
    rewrittenLines.reserveCapacity(lines.count)
    var nextURIIsPlaylist = false

    // Register every rewritten URI as one transaction.  A long media or
    // separate-audio playlist must never evict a token already present in the
    // same response; if the hard target bound is exceeded, roll back all new
    // mappings and fail without publishing a partly-invalid playlist.
    lock.lock()
    guard !stopped, listenerPort != nil else {
      lock.unlock()
      throw IPadAuthenticatedMediaProxyError.notStarted
    }
    let existingTokens = Set(targets.keys)
    do {
      for rawLine in lines {
        try Task.checkCancellation()
        let line = String(rawLine)
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
          rewrittenLines.append(line)
        } else if trimmed.hasPrefix("#") {
          let uppercased = trimmed.uppercased()
          let attributeURIIsPlaylist =
            uppercased.hasPrefix("#EXT-X-MEDIA:")
            || uppercased.hasPrefix("#EXT-X-I-FRAME-STREAM-INF:")
            || uppercased.hasPrefix("#EXT-X-RENDITION-REPORT:")
          rewrittenLines.append(
            try rewriteURIAttributes(
              in: line,
              relativeTo: baseURL,
              context: context,
              isPlaylist: attributeURIIsPlaylist,
              resolutionPolicy: resolutionPolicy
            )
          )
          if uppercased.hasPrefix("#EXT-X-STREAM-INF:") {
            nextURIIsPlaylist = true
          }
        } else {
          let leading = String(line.prefix { $0 == " " || $0 == "\t" })
          let trailing = String(
            line.reversed().prefix { $0 == " " || $0 == "\t" }.reversed()
          )
          let localURL = try mappedLocalURL(
            for: trimmed,
            relativeTo: baseURL,
            context: context,
            isPlaylist: nextURIIsPlaylist,
            resolutionPolicy: resolutionPolicy
          )
          rewrittenLines.append(leading + localURL.absoluteString + trailing)
          nextURIIsPlaylist = false
        }
      }

      let separator = usesCRLF ? "\r\n" : "\n"
      guard
        let result = rewrittenLines.joined(separator: separator).data(using: .utf8)
      else { throw IPadAuthenticatedMediaProxyError.unsafeTarget }
      lock.unlock()
      return result
    } catch {
      let insertedTokens = targets.keys.filter { !existingTokens.contains($0) }
      for token in insertedTokens { removeTargetLocked(token) }
      lock.unlock()
      throw error
    }
  }

  private func rewriteURIAttributes(
    in line: String,
    relativeTo baseURL: URL,
    context: IPadMediaRequestContext?,
    isPlaylist: Bool,
    resolutionPolicy: IPadMediaURLResolutionPolicy
  ) throws -> String {
    let pattern = #"URI\s*=\s*\"([^\"]+)\""#
    guard let expression = try? NSRegularExpression(pattern: pattern) else {
      return line
    }
    var result = line
    let matches = expression.matches(
      in: line,
      range: NSRange(line.startIndex..., in: line)
    )
    for match in matches.reversed() {
      guard match.numberOfRanges == 2,
        let valueRange = Range(match.range(at: 1), in: result)
      else { continue }
      let original = String(result[valueRange])
      let localURL = try mappedLocalURL(
        for: original,
        relativeTo: baseURL,
        context: context,
        isPlaylist: isPlaylist,
        resolutionPolicy: resolutionPolicy
      )
      result.replaceSubrange(valueRange, with: localURL.absoluteString)
    }
    return result
  }

  private func mappedLocalURL(
    for reference: String,
    relativeTo baseURL: URL,
    context: IPadMediaRequestContext?,
    isPlaylist: Bool,
    resolutionPolicy: IPadMediaURLResolutionPolicy
  ) throws -> URL {
    guard reference.utf8.count <= 8_192,
      reference.rangeOfCharacter(from: .controlCharacters) == nil,
      let resolved = URL(string: reference, relativeTo: baseURL)?.absoluteURL,
      let safeURL = Self.sanitizedHTTPSURL(
        resolved,
        resolutionPolicy: resolutionPolicy
      )
    else {
      throw IPadAuthenticatedMediaProxyError.unsafeTarget
    }
    let pathExtension = safeURL.pathExtension.lowercased()
    guard let listenerPort else {
      throw IPadAuthenticatedMediaProxyError.notStarted
    }
    let token = try registerTargetLocked(
      url: safeURL,
      context: context,
      isPlaylist: isPlaylist || pathExtension == "m3u8" || pathExtension == "m3u",
      resolutionPolicy: resolutionPolicy
    )
    guard
      let localURL = Self.localURL(
        port: listenerPort,
        token: token,
        isPlaylist: isPlaylist || pathExtension == "m3u8" || pathExtension == "m3u"
      )
    else { throw IPadAuthenticatedMediaProxyError.listenerFailed }
    return localURL
  }

  private func parseLocalRequest(_ data: Data) -> LocalRequest? {
    guard let header = String(data: data, encoding: .utf8),
      header.rangeOfCharacter(
        from: .controlCharacters.subtracting(
          CharacterSet(charactersIn: "\r\n\t")
        )) == nil
    else { return nil }
    let lines = header.components(separatedBy: "\r\n")
    guard let requestLine = lines.first else { return nil }
    let parts = requestLine.split(separator: " ", omittingEmptySubsequences: true)
    guard parts.count == 3,
      parts[0] == "GET" || parts[0] == "HEAD",
      parts[2] == "HTTP/1.1",
      parts[1].hasPrefix("/v1/")
    else { return nil }
    let requestTarget = parts[1].split(
      separator: "?",
      maxSplits: 1,
      omittingEmptySubsequences: false
    )
    guard requestTarget.count == 1 || requestTarget.count == 2 else { return nil }
    let pathComponents = requestTarget[0].split(
      separator: "/",
      omittingEmptySubsequences: true
    )
    guard pathComponents.count == 3, pathComponents[0] == "v1",
      pathComponents[2] == "index.m3u8" || pathComponents[2] == "resource"
    else { return nil }
    let token = String(pathComponents[1])
    guard token.count == 32,
      token.allSatisfy({ $0.isHexDigit && !$0.isUppercase })
    else { return nil }
    let hlsDeliveryDirectives: String?
    if requestTarget.count == 2 {
      guard
        let normalized = Self.normalizedHLSDeliveryDirectives(
          String(requestTarget[1])
        )
      else { return nil }
      hlsDeliveryDirectives = normalized
    } else {
      hlsDeliveryDirectives = nil
    }

    var range: String?
    var hostCount = 0
    for line in lines.dropFirst() where !line.isEmpty {
      guard let colon = line.firstIndex(of: ":") else { return nil }
      let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
      let value = line[line.index(after: colon)...]
        .trimmingCharacters(in: .whitespaces)
      if name == "host" {
        hostCount += 1
        guard isExpectedHost(value) else { return nil }
      } else if name == "range" {
        guard range == nil, Self.isSafeSingleByteRange(value) else { return nil }
        range = value
      }
    }
    guard hostCount == 1 else { return nil }
    return LocalRequest(
      method: String(parts[0]),
      token: token,
      range: range,
      hlsDeliveryDirectives: hlsDeliveryDirectives
    )
  }

  private func isExpectedHost(_ value: String) -> Bool {
    lock.lock()
    let port = listenerPort
    lock.unlock()
    guard let port else { return false }
    return value.lowercased() == "127.0.0.1:\(port)"
  }

  private static func isSafeSingleByteRange(_ value: String) -> Bool {
    guard value.utf8.count <= 128, value.hasPrefix("bytes="), !value.contains(",") else {
      return false
    }
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

  private static func normalizedHLSDeliveryDirectives(_ rawQuery: String) -> String? {
    guard !rawQuery.isEmpty, rawQuery.utf8.count <= 256,
      rawQuery.rangeOfCharacter(from: .controlCharacters) == nil,
      !rawQuery.contains("#"), !rawQuery.contains("%")
    else { return nil }

    let pairs = rawQuery.split(
      separator: "&",
      omittingEmptySubsequences: false
    )
    guard (1...3).contains(pairs.count) else { return nil }
    var values: [String: String] = [:]
    for pair in pairs {
      let components = pair.split(
        separator: "=",
        maxSplits: 1,
        omittingEmptySubsequences: false
      )
      guard components.count == 2 else { return nil }
      let name = String(components[0])
      let value = String(components[1])
      guard values[name] == nil else { return nil }
      switch name {
      case "_HLS_msn", "_HLS_part":
        guard !value.isEmpty, value.count <= 20,
          value.allSatisfy(\.isNumber), UInt64(value) != nil
        else { return nil }
      case "_HLS_skip":
        guard value == "YES" || value == "v2" else { return nil }
      default:
        return nil
      }
      values[name] = value
    }
    guard values["_HLS_part"] == nil || values["_HLS_msn"] != nil else {
      return nil
    }
    return ["_HLS_msn", "_HLS_part", "_HLS_skip"].compactMap { name in
      values[name].map { "\(name)=\($0)" }
    }.joined(separator: "&")
  }

  private func target(for token: String) -> TargetEntry? {
    lock.lock()
    defer { lock.unlock() }
    guard !stopped else { return nil }
    return targets[token]
  }

  private func registerTargetLocked(
    url: URL,
    context: IPadMediaRequestContext?,
    isPlaylist: Bool,
    resolutionPolicy: IPadMediaURLResolutionPolicy,
    syntheticPlaylist: Data? = nil
  ) throws -> String {
    if let existingTokens = tokensByURL[url] {
      for token in existingTokens {
        if let entry = targets[token], entry.context == context,
          entry.isPlaylist == isPlaylist,
          entry.resolutionPolicy == resolutionPolicy,
          entry.syntheticPlaylist == syntheticPlaylist
        {
          return token
        }
      }
    }

    guard targets.count < configuration.maximumMappedTargets else {
      throw IPadAuthenticatedMediaProxyError.targetLimitExceeded
    }

    var token: String
    repeat {
      token = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    } while targets[token] != nil
    targets[token] = TargetEntry(
      url: url,
      context: context,
      isPlaylist: isPlaylist,
      resolutionPolicy: resolutionPolicy,
      syntheticPlaylist: syntheticPlaylist
    )
    tokensByURL[url, default: []].append(token)
    return token
  }

  private func removeTargetLocked(_ token: String) {
    guard let removed = targets.removeValue(forKey: token) else { return }
    guard var urlTokens = tokensByURL[removed.url] else { return }
    urlTokens.removeAll { $0 == token }
    if urlTokens.isEmpty {
      tokensByURL.removeValue(forKey: removed.url)
    } else {
      tokensByURL[removed.url] = urlTokens
    }
  }

  private static func localURL(
    port: UInt16,
    token: String,
    isPlaylist: Bool
  ) -> URL? {
    let resourceName = isPlaylist ? "index.m3u8" : "resource"
    return URL(
      string: "http://127.0.0.1:\(port)/v1/\(token)/\(resourceName)"
    )
  }

  private func removeTask(for identifier: ObjectIdentifier) {
    lock.lock()
    requestTasks.removeValue(forKey: identifier)
    lock.unlock()
  }

  private func cancelHeaderTimeout(for identifier: ObjectIdentifier) {
    lock.lock()
    let timeout = headerTimeouts.removeValue(forKey: identifier)
    lock.unlock()
    timeout?.cancel()
  }

  private func recordParsedRequest(method: String) {
    lock.lock()
    diagnostics.parsedRequests += 1
    diagnostics.lastMethod = method
    lock.unlock()
  }

  private func recordFetchStarted() {
    lock.lock()
    diagnostics.fetchesStarted += 1
    lock.unlock()
  }

  private func recordOriginStatus(_ statusCode: Int) {
    lock.lock()
    diagnostics.lastOriginStatus = statusCode
    lock.unlock()
  }

  private func recordPlaylistDetected() {
    lock.lock()
    diagnostics.playlistsDetected += 1
    lock.unlock()
  }

  private func recordPlaylistRewritten() {
    lock.lock()
    diagnostics.playlistsRewritten += 1
    lock.unlock()
  }

  private func recordReplyStarted(statusCode: Int) {
    lock.lock()
    diagnostics.repliesStarted += 1
    diagnostics.lastLocalStatus = statusCode
    lock.unlock()
  }

  private func recordReplyCompleted() {
    lock.lock()
    diagnostics.repliesCompleted += 1
    lock.unlock()
  }

  private func recordFailure(_ failure: DiagnosticFailure) {
    lock.lock()
    diagnostics.lastFailure = failure
    lock.unlock()
  }

  private func send(_ response: LocalResponse, on connection: NWConnection) {
    recordReplyStarted(statusCode: response.statusCode)
    let reason = Self.reasonPhrase(for: response.statusCode)
    var header = "HTTP/1.1 \(response.statusCode) \(reason)\r\n"
    for (name, value) in response.headers
    where Self.isSafeResponseHeader(name: name, value: value) {
      header += "\(name): \(value)\r\n"
    }
    header += "Connection: close\r\n\r\n"
    let headerData = Data(header.utf8)
    guard !response.body.isEmpty else {
      connection.send(
        content: headerData,
        contentContext: .finalMessage,
        isComplete: true,
        completion: .contentProcessed { [weak self] error in
          if error == nil {
            self?.recordReplyCompleted()
          } else {
            self?.recordFailure(.send)
          }
          self?.finishConnection(connection)
        }
      )
      return
    }

    // TCP will not process a new content context until the preceding context
    // is complete. Queue both pieces immediately on the same final context;
    // waiting for an incomplete header's callback before queueing the body
    // deadlocks the response. Keeping the two Data values separate also avoids
    // another full-size copy of a media segment.
    let streamContext = NWConnection.ContentContext.finalMessage
    connection.batch {
      connection.send(
        content: headerData,
        contentContext: streamContext,
        isComplete: false,
        completion: .contentProcessed { [weak self] error in
          if error != nil { self?.recordFailure(.send) }
        }
      )
      connection.send(
        content: response.body,
        contentContext: streamContext,
        isComplete: true,
        completion: .contentProcessed { [weak self] error in
          if error == nil {
            self?.recordReplyCompleted()
          } else {
            self?.recordFailure(.send)
          }
          self?.finishConnection(connection)
        }
      )
    }
  }

  private func sendError(statusCode: Int, on connection: NWConnection) {
    let body = Data("media proxy error\n".utf8)
    send(
      LocalResponse(
        statusCode: statusCode,
        headers: [
          ("Content-Type", "text/plain; charset=utf-8"),
          ("Content-Length", String(body.count)),
          ("Cache-Control", "no-store"),
        ],
        body: body
      ),
      on: connection
    )
  }

  private func finishConnection(_ connection: NWConnection) {
    let identifier = ObjectIdentifier(connection)
    lock.lock()
    connections.removeValue(forKey: identifier)
    let headerTimeout = headerTimeouts.removeValue(forKey: identifier)
    let task = requestTasks.removeValue(forKey: identifier)
    lock.unlock()
    headerTimeout?.cancel()
    task?.cancel()
    connection.cancel()
  }

  fileprivate static func originURL(
    for approvedURL: URL,
    appendingHLSDeliveryDirectives directives: String?,
    resolutionPolicy: IPadMediaURLResolutionPolicy = .publicDiscovered
  ) -> URL? {
    guard
      let safeURL = sanitizedHTTPSURL(
        approvedURL,
        resolutionPolicy: resolutionPolicy
      )
    else { return nil }
    guard let directives else { return safeURL }

    var directiveItems: [String: String] = [:]
    for pair in directives.split(separator: "&") {
      let components = pair.split(separator: "=", maxSplits: 1)
      guard components.count == 2 else { return nil }
      directiveItems[String(components[0])] = String(components[1])
    }
    var existingDirectiveItems: [String: String] = [:]
    for item in URLComponents(
      url: safeURL,
      resolvingAgainstBaseURL: true
    )?.queryItems ?? [] where directiveItems[item.name] != nil {
      guard existingDirectiveItems[item.name] == nil, let value = item.value else {
        return nil
      }
      existingDirectiveItems[item.name] = value
    }
    if !existingDirectiveItems.isEmpty {
      guard existingDirectiveItems == directiveItems else { return nil }
      return safeURL
    }

    let base = safeURL.absoluteString
    let separator = safeURL.query == nil ? "?" : "&"
    guard let appendedURL = URL(string: base + separator + directives),
      sanitizedHTTPSURL(
        appendedURL,
        resolutionPolicy: resolutionPolicy
      ) != nil
    else { return nil }
    return appendedURL
  }

  fileprivate static func sanitizedPublicHTTPSURL(_ rawURL: URL?) -> URL? {
    sanitizedHTTPSURL(rawURL, resolutionPolicy: .publicDiscovered)
  }

  fileprivate static func sanitizedHTTPSURL(
    _ rawURL: URL?,
    resolutionPolicy: IPadMediaURLResolutionPolicy
  ) -> URL? {
    guard let rawURL,
      var components = URLComponents(url: rawURL, resolvingAgainstBaseURL: true),
      components.scheme?.lowercased() == "https",
      components.user == nil, components.password == nil,
      components.host?.isEmpty == false
    else { return nil }
    components.scheme = "https"
    components.fragment = nil
    guard let safeURL = components.url,
      IPadMediaURLResolver.isURL(safeURL, allowedBy: resolutionPolicy)
    else { return nil }
    return safeURL
  }

  private static func isHLSPlaylist(
    _ data: Data,
    response: HTTPURLResponse
  ) -> Bool {
    let mimeType = response.mimeType?.lowercased() ?? ""
    if mimeType.contains("mpegurl") { return true }
    let prefix = String(decoding: data.prefix(128), as: UTF8.self)
      .replacingOccurrences(of: "\u{feff}", with: "")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    return prefix.hasPrefix("#EXTM3U")
  }

  private static func responseHeaders(
    from response: HTTPURLResponse,
    bodyCount: Int,
    isHead: Bool,
    isPlaylist: Bool
  ) -> [(String, String)] {
    var headers: [(String, String)] = []
    let contentType =
      isPlaylist
      ? "application/vnd.apple.mpegurl"
      : response.value(forHTTPHeaderField: "Content-Type")
    if let contentType { headers.append(("Content-Type", contentType)) }

    if isHead, !isPlaylist {
      if let originalLength = response.value(forHTTPHeaderField: "Content-Length") {
        headers.append(("Content-Length", originalLength))
      }
    } else {
      headers.append(("Content-Length", String(bodyCount)))
    }

    if !isPlaylist {
      for name in [
        "Content-Range", "Accept-Ranges", "Cache-Control", "ETag", "Last-Modified",
      ] {
        if let value = response.value(forHTTPHeaderField: name) {
          headers.append((name, value))
        }
      }
    } else {
      headers.append(("Cache-Control", "no-store"))
    }
    if let age = response.value(forHTTPHeaderField: "Age"), UInt64(age) != nil {
      headers.append(("Age", age))
    }
    if let retryAfter = response.value(forHTTPHeaderField: "Retry-After"),
      UInt64(retryAfter) != nil
    {
      headers.append(("Retry-After", retryAfter))
    }
    return headers
  }

  private static func isSafeResponseHeader(name: String, value: String) -> Bool {
    let allowed = [
      "Content-Type", "Content-Length", "Content-Range", "Accept-Ranges",
      "Cache-Control", "ETag", "Last-Modified", "Age", "Retry-After",
    ]
    return allowed.contains(name) && value.utf8.count <= 8_192
      && value.rangeOfCharacter(from: .controlCharacters) == nil
  }

  private static func reasonPhrase(for statusCode: Int) -> String {
    switch statusCode {
    case 200: return "OK"
    case 206: return "Partial Content"
    case 400: return "Bad Request"
    case 401: return "Unauthorized"
    case 403: return "Forbidden"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 416: return "Range Not Satisfiable"
    case 429: return "Too Many Requests"
    case 431: return "Request Header Fields Too Large"
    case 500: return "Internal Server Error"
    case 502: return "Bad Gateway"
    case 503: return "Service Unavailable"
    case 504: return "Gateway Timeout"
    default: return "Response"
    }
  }
}

private struct IPadAuthenticatedMediaOriginPayload {
  let data: Data
  let response: HTTPURLResponse
}

private final class IPadAuthenticatedMediaOriginRequest: @unchecked Sendable {
  private let maximumResponseBytes: Int
  private let maximumRedirectCount: Int
  private let timeout: TimeInterval
  private let rangeHeader: String?
  private let method: String
  private let requestContext: IPadMediaRequestContext?
  private let hlsDeliveryDirectives: String?
  private let resolutionPolicy: IPadMediaURLResolutionPolicy
  private let onResponseStatus: @Sendable (Int) -> Void
  private let lock = NSLock()
  private var cancellation: IPadSharedHTTPRequestCancellation?
  private var cancellationRequested = false

  init(
    maximumResponseBytes: Int,
    maximumRedirectCount: Int,
    timeout: TimeInterval,
    rangeHeader: String?,
    method: String,
    requestContext: IPadMediaRequestContext?,
    hlsDeliveryDirectives: String?,
    resolutionPolicy: IPadMediaURLResolutionPolicy,
    onResponseStatus: @escaping @Sendable (Int) -> Void
  ) {
    self.maximumResponseBytes = maximumResponseBytes
    self.maximumRedirectCount = maximumRedirectCount
    self.timeout = timeout
    self.rangeHeader = rangeHeader
    self.method = method
    self.requestContext = requestContext
    self.hlsDeliveryDirectives = hlsDeliveryDirectives
    self.resolutionPolicy = resolutionPolicy
    self.onResponseStatus = onResponseStatus
  }

  func start(_ request: URLRequest) async throws -> IPadAuthenticatedMediaOriginPayload {
    try Task.checkCancellation()
    let cancellation = IPadSharedHTTPRequestCancellation()
    guard install(cancellation: cancellation) else {
      throw CancellationError()
    }
    defer {
      cancellation.clear()
      clear(cancellation: cancellation)
    }
    do {
      let payload = try await IPadSharedHTTPTransport.shared.data(
        for: request,
        options: IPadSharedHTTPTransportOptions(
          maximumResponseBytes: maximumResponseBytes,
          maximumRedirectCount: maximumRedirectCount,
          timeout: timeout,
          resolutionPolicy: resolutionPolicy,
          requestContext: requestContext,
          requiresHTTPS: true,
          hlsDeliveryDirectives: hlsDeliveryDirectives,
          // Proxy requests are demanded by AVPlayer (playlist, separate
          // audio, or the current media range), so they outrank speculative
          // restoration prefetch on the same origin.
          priority: .critical
        ),
        cancellation: cancellation
      )
      onResponseStatus(payload.response.statusCode)
      return IPadAuthenticatedMediaOriginPayload(
        data: payload.data,
        response: payload.response
      )
    } catch let error as IPadSharedHTTPTransportError {
      switch error {
      case .unsafeURL, .insecureRedirect:
        throw IPadAuthenticatedMediaProxyError.unsafeTarget
      case .tooManyRedirects, .responseTooLarge:
        throw IPadAuthenticatedMediaProxyError.targetLimitExceeded
      case .missingResponse:
        throw IPadAuthenticatedMediaProxyError.listenerFailed
      }
    }
  }

  private func install(
    cancellation newCancellation: IPadSharedHTTPRequestCancellation
  ) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard !cancellationRequested else { return false }
    cancellation = newCancellation
    return true
  }

  private func clear(
    cancellation completedCancellation: IPadSharedHTTPRequestCancellation
  ) {
    lock.lock()
    if cancellation === completedCancellation { cancellation = nil }
    lock.unlock()
  }

  private func cancel() {
    lock.lock()
    cancellationRequested = true
    let cancellation = cancellation
    lock.unlock()
    cancellation?.cancel()
  }
}
