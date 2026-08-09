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

/// A short-lived, loopback-only bridge between AVPlayer and authenticated
/// public HTTPS media. AVPlayer sees an opaque local URL; origin URLs and their
/// credentials remain only in this process's in-memory target table.
///
/// This proxy intentionally uses only public APIs. Call `stop()` as soon as the
/// player no longer needs the stream.
final class IPadAuthenticatedMediaProxy: @unchecked Sendable {
  struct Configuration: Sendable, Equatable {
    var maximumConcurrentRequests = 4
    var maximumMappedTargets = 4_096
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
    case busy = "busy"
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
  private var targetInsertionOrder: [String] = []
  private var activeRequestCount = 0
  private var didReportInteractionRequired = false
  private var stopped = false
  private var diagnostics = DiagnosticState()

  init(
    configuration: Configuration = Configuration(),
    onInteractionRequired: @escaping @Sendable (URL?) -> Void = { _ in }
  ) {
    self.configuration = configuration
    self.onInteractionRequired = onInteractionRequired
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
    targetInsertionOrder.removeAll()
    activeRequestCount = 0
    lock.unlock()

    listener?.cancel()
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
    guard !stopped,
      connections.count < configuration.maximumConcurrentRequests * 2
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
    guard beginOriginRequest() else {
      recordFailure(.busy)
      sendError(statusCode: 503, on: connection)
      return
    }
    recordFetchStarted()

    let identifier = ObjectIdentifier(connection)
    let task = Task { [weak self, weak connection] in
      guard let self, let connection else { return }
      defer {
        self.endOriginRequest()
        self.removeTask(for: identifier)
      }
      do {
        let response = try await self.fetch(request: request, entry: entry)
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
      endOriginRequest()
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
    guard let result = rewrittenLines.joined(separator: separator).data(using: .utf8) else {
      throw IPadAuthenticatedMediaProxyError.unsafeTarget
    }
    return result
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
    return try localURL(
      for: safeURL,
      context: context,
      isPlaylist: isPlaylist || pathExtension == "m3u8" || pathExtension == "m3u",
      resolutionPolicy: resolutionPolicy
    )
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
    guard !stopped, let entry = targets[token] else { return nil }
    touchTargetLocked(token)
    return entry
  }

  private func registerTargetLocked(
    url: URL,
    context: IPadMediaRequestContext?,
    isPlaylist: Bool,
    resolutionPolicy: IPadMediaURLResolutionPolicy
  ) throws -> String {
    if let existingTokens = tokensByURL[url] {
      for token in existingTokens {
        if let entry = targets[token], entry.context == context,
          entry.isPlaylist == isPlaylist,
          entry.resolutionPolicy == resolutionPolicy
        {
          touchTargetLocked(token)
          return token
        }
      }
    }

    while targets.count >= configuration.maximumMappedTargets,
      let oldestToken = targetInsertionOrder.first
    {
      targetInsertionOrder.removeFirst()
      removeTargetLocked(oldestToken)
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
      resolutionPolicy: resolutionPolicy
    )
    tokensByURL[url, default: []].append(token)
    targetInsertionOrder.append(token)
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

  private func touchTargetLocked(_ token: String) {
    targetInsertionOrder.removeAll { $0 == token }
    targetInsertionOrder.append(token)
  }

  private func beginOriginRequest() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard !stopped, activeRequestCount < configuration.maximumConcurrentRequests else {
      return false
    }
    activeRequestCount += 1
    return true
  }

  private func endOriginRequest() {
    lock.lock()
    activeRequestCount = max(0, activeRequestCount - 1)
    lock.unlock()
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

private final class IPadAuthenticatedMediaOriginRequest: NSObject, @unchecked Sendable {
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
  private var continuation: CheckedContinuation<IPadAuthenticatedMediaOriginPayload, Error>?
  private var session: URLSession?
  private var task: URLSessionDataTask?
  private var response: HTTPURLResponse?
  private var receivedData = Data()
  private var redirectCount = 0
  private var didFinish = false
  private var cancellationRequested = false
  private var expectsBody = true

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
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpShouldSetCookies = false
        configuration.urlCredentialStorage = nil
        configuration.waitsForConnectivity = false
        configuration.httpMaximumConnectionsPerHost = 2

        let delegateQueue = OperationQueue()
        delegateQueue.maxConcurrentOperationCount = 1
        delegateQueue.qualityOfService = .userInitiated
        let session = URLSession(
          configuration: configuration,
          delegate: self,
          delegateQueue: delegateQueue
        )
        let task = session.dataTask(with: request)

        lock.lock()
        if cancellationRequested {
          didFinish = true
          lock.unlock()
          session.invalidateAndCancel()
          continuation.resume(throwing: CancellationError())
          return
        }
        self.continuation = continuation
        self.session = session
        self.task = task
        expectsBody = request.httpMethod?.uppercased() != "HEAD"
        lock.unlock()
        task.resume()
      }
    } onCancel: {
      self.cancel()
    }
  }

  private func cancel() {
    lock.lock()
    cancellationRequested = true
    guard !didFinish, let continuation else {
      let task = task
      lock.unlock()
      task?.cancel()
      return
    }
    didFinish = true
    let task = task
    let session = session
    self.continuation = nil
    self.task = nil
    self.session = nil
    lock.unlock()
    task?.cancel()
    session?.invalidateAndCancel()
    continuation.resume(throwing: CancellationError())
  }

  private func finish(
    _ result: Result<IPadAuthenticatedMediaOriginPayload, Error>
  ) {
    lock.lock()
    guard !didFinish, let continuation else {
      lock.unlock()
      return
    }
    didFinish = true
    let task = task
    let session = session
    self.continuation = nil
    self.task = nil
    self.session = nil
    lock.unlock()
    task?.cancel()
    session?.invalidateAndCancel()
    continuation.resume(with: result)
  }
}

extension IPadAuthenticatedMediaOriginRequest: URLSessionDataDelegate {
  func urlSession(
    _: URLSession,
    dataTask _: URLSessionDataTask,
    didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    guard let httpResponse = response as? HTTPURLResponse,
      IPadAuthenticatedMediaProxy.sanitizedHTTPSURL(
        httpResponse.url,
        resolutionPolicy: resolutionPolicy
      ) != nil
    else {
      completionHandler(.cancel)
      finish(.failure(IPadAuthenticatedMediaProxyError.unsafeTarget))
      return
    }
    onResponseStatus(httpResponse.statusCode)
    if let error = IPadMediaURLResolver.interactionChallengeError(
      response: httpResponse
    ) {
      completionHandler(.cancel)
      finish(.failure(error))
      return
    }
    lock.lock()
    let shouldReadBody = expectsBody
    lock.unlock()
    if shouldReadBody,
      httpResponse.expectedContentLength > Int64(maximumResponseBytes)
    {
      completionHandler(.cancel)
      finish(.failure(IPadAuthenticatedMediaProxyError.targetLimitExceeded))
      return
    }
    lock.lock()
    self.response = httpResponse
    lock.unlock()
    completionHandler(.allow)
  }

  func urlSession(_: URLSession, dataTask _: URLSessionDataTask, didReceive data: Data) {
    lock.lock()
    let wouldOverflow = !didFinish && data.count > maximumResponseBytes - receivedData.count
    if !didFinish, !wouldOverflow { receivedData.append(data) }
    lock.unlock()
    if wouldOverflow {
      finish(.failure(IPadAuthenticatedMediaProxyError.targetLimitExceeded))
    }
  }

  func urlSession(
    _: URLSession,
    task _: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    guard
      let redirectURL = IPadAuthenticatedMediaProxy.sanitizedHTTPSURL(
        request.url,
        resolutionPolicy: resolutionPolicy
      ),
      let destination = IPadAuthenticatedMediaProxy.originURL(
        for: redirectURL,
        appendingHLSDeliveryDirectives: hlsDeliveryDirectives,
        resolutionPolicy: resolutionPolicy
      )
    else {
      completionHandler(nil)
      finish(.failure(IPadAuthenticatedMediaProxyError.unsafeTarget))
      return
    }
    if let error = IPadMediaURLResolver.interactionChallengeError(
      response: response,
      destinationURL: destination
    ) {
      completionHandler(nil)
      finish(.failure(error))
      return
    }
    lock.lock()
    redirectCount += 1
    let tooManyRedirects = redirectCount > maximumRedirectCount
    lock.unlock()
    guard !tooManyRedirects else {
      completionHandler(nil)
      finish(.failure(IPadAuthenticatedMediaProxyError.targetLimitExceeded))
      return
    }
    Task {
      await requestContext?.updateCookies(from: response)
      var redirectedRequest = request
      redirectedRequest.url = destination
      redirectedRequest.httpMethod = method
      redirectedRequest.timeoutInterval = timeout
      if let rangeHeader {
        redirectedRequest.setValue(rangeHeader, forHTTPHeaderField: "Range")
      } else {
        redirectedRequest.setValue(nil, forHTTPHeaderField: "Range")
      }
      requestContext?.applying(to: &redirectedRequest)
      completionHandler(redirectedRequest)
    }
  }

  func urlSession(_: URLSession, task _: URLSessionTask, didCompleteWithError error: Error?) {
    if let error {
      finish(.failure(error))
      return
    }
    lock.lock()
    let response = response
    let data = receivedData
    lock.unlock()
    guard let response else {
      finish(.failure(IPadAuthenticatedMediaProxyError.listenerFailed))
      return
    }
    finish(.success(IPadAuthenticatedMediaOriginPayload(data: data, response: response)))
  }
}
