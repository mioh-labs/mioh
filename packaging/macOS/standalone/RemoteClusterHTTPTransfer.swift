import CryptoKit
import Darwin
import Foundation
import Network
import Security

enum RemoteClusterHTTPTransferError: LocalizedError {
  case listenerFailed(String)
  case invalidSource
  case invalidContract
  case uploadFailed(String)

  var errorDescription: String? {
    switch self {
    case .listenerFailed(let detail):
      return "クラスタHTTP転送listenerを開始できません: \(detail)"
    case .invalidSource:
      return "クラスタHTTP転送元が通常ファイルではありません"
    case .invalidContract:
      return "クラスタHTTP転送契約が不正です"
    case .uploadFailed(let detail):
      return "クラスタ成果物のHTTP転送に失敗しました: \(detail)"
    }
  }
}

struct RemoteClusterHTTPSourceIdentity: Sendable {
  let byteCount: Int64
  let sha256: String
}

/// Coordinator-owned, RemoteControl-independent HTTP/1.1 transfer endpoint.
/// It deliberately implements only the small surface AVURLAsset and the
/// Worker uploader need: HEAD, one byte Range per GET, and bounded PUT.
final class RemoteClusterHTTPTransferServer: @unchecked Sendable {
  private static let uploadIdleTimeout: TimeInterval = 5 * 60

  private struct SourceSignature: Equatable, Sendable {
    let device: UInt64
    let inode: UInt64
    let bytes: Int64
    let modificationSeconds: Int64
    let modificationNanoseconds: Int64

    static func read(fileDescriptor: Int32) throws -> Self {
      var value = stat()
      guard fstat(fileDescriptor, &value) == 0,
        (value.st_mode & S_IFMT) == S_IFREG
      else { throw RemoteClusterHTTPTransferError.invalidSource }
      return Self(
        device: UInt64(value.st_dev),
        inode: UInt64(value.st_ino),
        bytes: Int64(value.st_size),
        modificationSeconds: Int64(value.st_mtimespec.tv_sec),
        modificationNanoseconds: Int64(value.st_mtimespec.tv_nsec)
      )
    }
  }

  /// One immutable identity is established from one open descriptor per
  /// Coordinator export. Every attempt shares that pinned descriptor, so a
  /// path replacement cannot change bytes between hashing and a later Range.
  private final class PinnedSource: @unchecked Sendable {
    let url: URL
    let fileDescriptor: Int32
    let signature: SourceSignature
    let sha256: String
    private var closed = false

    private init(
      url: URL,
      fileDescriptor: Int32,
      signature: SourceSignature,
      sha256: String
    ) {
      self.url = url
      self.fileDescriptor = fileDescriptor
      self.signature = signature
      self.sha256 = sha256
    }

    static func openAndVerify(_ url: URL) throws -> PinnedSource {
      let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
      guard descriptor >= 0 else { throw RemoteClusterHTTPTransferError.invalidSource }
      do {
        let signature = try SourceSignature.read(fileDescriptor: descriptor)
        guard signature.bytes > 0 else {
          throw RemoteClusterHTTPTransferError.invalidSource
        }
        var hasher = SHA256()
        var offset: Int64 = 0
        var buffer = [UInt8](repeating: 0, count: 4 * 1_024 * 1_024)
        while offset < signature.bytes {
          let wanted = Int(min(Int64(buffer.count), signature.bytes - offset))
          let count = pread(descriptor, &buffer, wanted, off_t(offset))
          guard count > 0 else { throw RemoteClusterHTTPTransferError.invalidSource }
          hasher.update(data: Data(buffer[0..<count]))
          offset += Int64(count)
        }
        // Detect in-place mutation while the digest was being calculated.
        guard try SourceSignature.read(fileDescriptor: descriptor) == signature else {
          throw RemoteClusterHTTPTransferError.invalidSource
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return PinnedSource(
          url: url,
          fileDescriptor: descriptor,
          signature: signature,
          sha256: digest
        )
      } catch {
        Darwin.close(descriptor)
        throw error
      }
    }

    func close() {
      guard !closed else { return }
      closed = true
      Darwin.close(fileDescriptor)
    }

    deinit { close() }
  }

  private final class Binding {
    let ticket: String
    let attemptID: UUID
    let leaseID: UUID
    let source: PinnedSource
    let outputURL: URL
    let mediaRange: RemoteClusterMediaRange
    var expiresAt: Date
    let maximumOutputBytes: Int64
    var activeConnections: [UUID: NWConnection] = [:]
    var uploadInProgress = false
    var currentUploadReservation: Int64 = 0
    var publishedByteCount: Int64?
    var publishedSHA256: String?

    init(
      ticket: String,
      request: RemoteClusterJobRequest,
      source: PinnedSource,
      outputURL: URL,
      expiresAt: Date,
      maximumOutputBytes: Int64
    ) {
      self.ticket = ticket
      attemptID = request.attemptID
      leaseID = request.leaseID
      self.source = source
      self.outputURL = outputURL
      mediaRange = request.mediaRange
      self.expiresAt = expiresAt
      self.maximumOutputBytes = maximumOutputBytes
    }
  }

  private final class UploadState {
    let binding: Binding
    let connectionID: UUID
    let expected: Int64
    let handle: FileHandle
    let part: URL
    var received: Int64
    var hasher: SHA256
    var idleTimeout: DispatchWorkItem?
    var closed = false

    init(
      binding: Binding,
      connectionID: UUID,
      expected: Int64,
      handle: FileHandle,
      part: URL,
      received: Int64,
      hasher: SHA256
    ) {
      self.binding = binding
      self.connectionID = connectionID
      self.expected = expected
      self.handle = handle
      self.part = part
      self.received = received
      self.hasher = hasher
    }

    func close() {
      guard !closed else { return }
      closed = true
      idleTimeout?.cancel()
      idleTimeout = nil
      try? handle.close()
    }

    deinit { close() }
  }

  private struct HTTPRequestHead {
    let method: String
    let path: String
    let headers: [String: String]
    let initialBody: Data
  }

  private let queue = DispatchQueue(label: "mioh.remote-cluster.http-transfer")
  private let advertisedHost: String
  private var listener: NWListener?
  private var boundPort: UInt16?
  private var bindings: [String: Binding] = [:]
  private var attemptTickets: [UUID: String] = [:]
  private var pinnedSources: [String: PinnedSource] = [:]
  private var acceptedConnections: [UUID: NWConnection] = [:]
  private var connectionTickets: [UUID: String] = [:]
  private var headTimeouts: [UUID: DispatchWorkItem] = [:]
  private var uploads: [UUID: UploadState] = [:]
  private var maximumAggregateOutputBytes: Int64 = 64 * 1_073_741_824
  private var aggregateOutputBytes: Int64 = 0
  private static let maximumAcceptedConnections = 128

  init(advertisedHost: String? = nil) {
    self.advertisedHost = advertisedHost ?? Self.defaultAdvertisedHost
  }

  func start() async throws {
    if queue.sync(execute: { boundPort != nil }) { return }
    let listener = try NWListener(using: .tcp, on: .any)
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Void, Error>) in
      let completion = RemoteClusterTransferOnce()
      queue.async { self.listener = listener }
      listener.newConnectionHandler = { [weak self] connection in
        self?.accept(connection)
      }
      listener.stateUpdateHandler = { [weak self, weak listener] state in
        guard let self, let listener else { return }
        switch state {
        case .ready:
          self.queue.async {
            self.boundPort = listener.port?.rawValue
            if completion.claim() { continuation.resume() }
          }
        case .failed(let error):
          self.queue.async {
            if completion.claim() {
              continuation.resume(
                throwing: RemoteClusterHTTPTransferError.listenerFailed(
                  error.localizedDescription
                )
              )
            }
          }
        case .cancelled:
          if completion.claim() {
            continuation.resume(
              throwing: RemoteClusterHTTPTransferError.listenerFailed("cancelled")
            )
          }
        default:
          break
        }
      }
      listener.start(queue: queue)
    }
  }

  func stop() {
    queue.sync {
      listener?.cancel()
      listener = nil
      boundPort = nil
      let all = Array(bindings.values)
      let allSources = Array(pinnedSources.values)
      let allConnections = Array(acceptedConnections.values)
      bindings.removeAll()
      attemptTickets.removeAll()
      pinnedSources.removeAll()
      acceptedConnections.removeAll()
      connectionTickets.removeAll()
      headTimeouts.values.forEach { $0.cancel() }
      headTimeouts.removeAll()
      let allUploads = Array(uploads.values)
      uploads.removeAll()
      aggregateOutputBytes = 0
      allConnections.forEach { $0.cancel() }
      for binding in all {
        removePartFile(binding)
      }
      allUploads.forEach {
        $0.close()
        try? FileManager.default.removeItem(at: $0.part)
      }
      allSources.forEach { $0.close() }
    }
  }

  func configureMaximumAggregateOutputBytes(_ byteCount: Int64) throws {
    guard byteCount > 0 else { throw RemoteClusterHTTPTransferError.invalidContract }
    try queue.sync {
      guard bindings.isEmpty, aggregateOutputBytes == 0 else {
        throw RemoteClusterHTTPTransferError.invalidContract
      }
      maximumAggregateOutputBytes = byteCount
    }
  }

  /// Opens and hashes the source exactly once for this export. Hashing happens
  /// off the UI actor; subsequent attempt registration is O(1).
  func pinSource(_ inputURL: URL) async throws -> RemoteClusterHTTPSourceIdentity {
    let canonicalURL = inputURL.standardizedFileURL
    let key = canonicalURL.path
    if let existing = queue.sync(execute: { pinnedSources[key] }) {
      guard (try? SourceSignature.read(fileDescriptor: existing.fileDescriptor))
        == existing.signature
      else { throw RemoteClusterHTTPTransferError.invalidSource }
      return RemoteClusterHTTPSourceIdentity(
        byteCount: existing.signature.bytes,
        sha256: existing.sha256
      )
    }
    let source = try await Task.detached(priority: .utility) {
      try PinnedSource.openAndVerify(canonicalURL)
    }.value
    return try queue.sync {
      guard boundPort != nil else {
        source.close()
        throw RemoteClusterHTTPTransferError.listenerFailed("not_ready")
      }
      if let existing = pinnedSources[key] {
        source.close()
        guard existing.signature == source.signature,
          existing.sha256 == source.sha256
        else { throw RemoteClusterHTTPTransferError.invalidSource }
        return RemoteClusterHTTPSourceIdentity(
          byteCount: existing.signature.bytes,
          sha256: existing.sha256
        )
      }
      pinnedSources[key] = source
      return RemoteClusterHTTPSourceIdentity(
        byteCount: source.signature.bytes,
        sha256: source.sha256
      )
    }
  }

  /// Registers one exact attempt. The source was already SHA-256 checked by
  /// the Coordinator; a stable inode/size/mtime signature prevents it changing
  /// underneath later Range requests without hashing the whole file again.
  func register(
    request: RemoteClusterJobRequest,
    inputURL: URL,
    outputURL: URL,
    maximumOutputBytes: Int64
  ) throws -> RemoteClusterHTTPTransferDescriptor {
    let key = inputURL.standardizedFileURL.path
    let source = queue.sync { pinnedSources[key] }
    guard let source,
      source.signature.bytes == request.inputByteCount,
      source.sha256 == request.inputSHA256,
      (try? SourceSignature.read(fileDescriptor: source.fileDescriptor)) == source.signature,
      request.httpTransfer == nil,
      maximumOutputBytes > 0,
      request.leaseExpiresAt > Date()
    else { throw RemoteClusterHTTPTransferError.invalidContract }
    let ticket = Self.randomTicket()
    let expiration = min(
      request.leaseExpiresAt,
      Date().addingTimeInterval(15 * 60)
    )
    let port = queue.sync { boundPort }
    guard let port, port > 0 else {
      throw RemoteClusterHTTPTransferError.listenerFailed("not_ready")
    }
    let host = advertisedHost
    let inputExtension = source.url.pathExtension.lowercased()
    guard let inputEndpoint = Self.endpoint(
      host: host,
      port: port,
      ticket: ticket,
      leaf: inputExtension.isEmpty ? "input.mp4" : "input.\(inputExtension)"
    ), let outputEndpoint = Self.endpoint(
      host: host, port: port, ticket: ticket, leaf: "output.mp4"
    ) else { throw RemoteClusterHTTPTransferError.invalidContract }
    let binding = Binding(
      ticket: ticket,
      request: request,
      source: source,
      outputURL: outputURL,
      expiresAt: expiration,
      maximumOutputBytes: maximumOutputBytes
    )
    queue.sync {
      if let old = attemptTickets[request.attemptID], let prior = bindings.removeValue(forKey: old) {
        for connectionID in prior.activeConnections.keys {
          abortUpload(connectionID: connectionID)
        }
        prior.activeConnections.values.forEach { $0.cancel() }
        releaseUploadReservation(prior)
        removePartFile(prior)
      }
      bindings[ticket] = binding
      attemptTickets[request.attemptID] = ticket
    }
    return RemoteClusterHTTPTransferDescriptor(
      inputURL: inputEndpoint.absoluteString,
      outputURL: outputEndpoint.absoluteString,
      expiresAt: expiration,
      maximumOutputBytes: maximumOutputBytes
    )
  }

  func unregister(attemptID: UUID) {
    queue.async {
      guard let ticket = self.attemptTickets.removeValue(forKey: attemptID),
      let binding = self.bindings.removeValue(forKey: ticket)
      else { return }
      for connectionID in binding.activeConnections.keys {
        self.abortUpload(connectionID: connectionID)
      }
      binding.activeConnections.values.forEach { $0.cancel() }
      self.releaseUploadReservation(binding)
      self.removePartFile(binding)
    }
  }

  func renew(attemptID: UUID, leaseID: UUID, until expiration: Date) -> Bool {
    queue.sync {
      guard let ticket = attemptTickets[attemptID], let binding = bindings[ticket],
        binding.leaseID == leaseID, expiration > Date(),
        expiration.timeIntervalSinceNow <= 15 * 60
      else { return false }
      binding.expiresAt = max(binding.expiresAt, expiration)
      return true
    }
  }

  private func accept(_ connection: NWConnection) {
    guard acceptedConnections.count < Self.maximumAcceptedConnections else {
      connection.cancel()
      return
    }
    let connectionID = UUID()
    acceptedConnections[connectionID] = connection
    let timeout = DispatchWorkItem { [weak self, weak connection] in
      guard let self, let connection,
        self.acceptedConnections[connectionID] != nil,
        self.headTimeouts[connectionID] != nil
      else { return }
      self.sendStatus(408, "Request Timeout", on: connection)
    }
    headTimeouts[connectionID] = timeout
    queue.asyncAfter(deadline: .now() + 15, execute: timeout)
    connection.stateUpdateHandler = { [weak self, weak connection] state in
      guard let self else { return }
      switch state {
      case .failed, .cancelled:
        self.queue.async { self.releaseConnection(connectionID) }
      default:
        _ = connection
      }
    }
    connection.start(queue: queue)
    receiveHead(connection, connectionID: connectionID, accumulated: Data())
  }

  private func receiveHead(
    _ connection: NWConnection,
    connectionID: UUID,
    accumulated: Data
  ) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1_024) {
      [weak self, weak connection] data, _, complete, error in
      guard let self, let connection else { return }
      var bytes = accumulated
      if let data { bytes.append(data) }
      let headerMarker = Data("\r\n\r\n".utf8)
      if let headerRange = bytes.range(of: headerMarker) {
        guard headerRange.lowerBound <= 32 * 1_024 else {
          self.sendStatus(431, "Request Header Fields Too Large", on: connection)
          return
        }
      } else if bytes.count > 32 * 1_024 {
        self.sendStatus(431, "Request Header Fields Too Large", on: connection)
        return
      }
      if let parsed = self.parseHead(bytes) {
        self.headTimeouts.removeValue(forKey: connectionID)?.cancel()
        self.route(parsed, connectionID: connectionID, on: connection)
      } else if complete || error != nil {
        self.sendStatus(400, "Bad Request", on: connection)
      } else {
        self.receiveHead(connection, connectionID: connectionID, accumulated: bytes)
      }
    }
  }

  private func parseHead(_ data: Data) -> HTTPRequestHead? {
    let marker = Data("\r\n\r\n".utf8)
    guard let range = data.range(of: marker),
      let text = String(data: data[..<range.lowerBound], encoding: .utf8)
    else { return nil }
    let lines = text.components(separatedBy: "\r\n")
    guard let requestLine = lines.first else { return nil }
    let fields = requestLine.split(separator: " ", omittingEmptySubsequences: true)
    guard fields.count == 3, fields[2].hasPrefix("HTTP/1.") else { return nil }
    var headers: [String: String] = [:]
    for line in lines.dropFirst() {
      guard let separator = line.firstIndex(of: ":") else { return nil }
      let key = line[..<separator].lowercased()
      let value = line[line.index(after: separator)...]
        .trimmingCharacters(in: .whitespaces)
      if headers[key] != nil { return nil }
      headers[key] = value
    }
    let bodyStart = range.upperBound
    return HTTPRequestHead(
      method: String(fields[0]).uppercased(),
      path: String(fields[1]).components(separatedBy: "?").first ?? "",
      headers: headers,
      initialBody: bodyStart < data.endIndex ? Data(data[bodyStart...]) : Data()
    )
  }

  private func route(
    _ request: HTTPRequestHead,
    connectionID: UUID,
    on connection: NWConnection
  ) {
    let pieces = request.path.split(separator: "/", omittingEmptySubsequences: true)
    guard pieces.count == 4,
      pieces[0] == "mioh-cluster", pieces[1] == "v1",
      let binding = bindings[String(pieces[2])],
      binding.expiresAt > Date(),
      attemptTickets[binding.attemptID] == binding.ticket
    else {
      sendStatus(404, "Not Found", on: connection)
      return
    }
    binding.activeConnections[connectionID] = connection
    connectionTickets[connectionID] = binding.ticket
    enforceLease(binding, connectionID: connectionID)
    let leaf = String(pieces[3])
    switch (request.method, leaf) {
    case ("HEAD", let value) where value.hasPrefix("input."):
      serveInput(binding, rangeHeader: request.headers["range"], headOnly: true, on: connection)
    case ("GET", let value) where value.hasPrefix("input."):
      serveInput(binding, rangeHeader: request.headers["range"], headOnly: false, on: connection)
    case ("PUT", "output.mp4"):
      receiveOutput(request, binding: binding, connectionID: connectionID, on: connection)
    default:
      sendStatus(405, "Method Not Allowed", on: connection)
    }
  }

  private func releaseConnection(_ connectionID: UUID) {
    headTimeouts.removeValue(forKey: connectionID)?.cancel()
    abortUpload(connectionID: connectionID)
    acceptedConnections.removeValue(forKey: connectionID)
    if let ticket = connectionTickets.removeValue(forKey: connectionID) {
      bindings[ticket]?.activeConnections.removeValue(forKey: connectionID)
    }
  }

  private func enforceLease(_ binding: Binding, connectionID: UUID) {
    guard acceptedConnections[connectionID] != nil,
      bindings[binding.ticket] === binding,
      binding.activeConnections[connectionID] != nil
    else { return }
    if binding.expiresAt <= Date() {
      expireBinding(binding)
      return
    }
    let interval = min(5, max(0.1, binding.expiresAt.timeIntervalSinceNow))
    queue.asyncAfter(deadline: .now() + interval) { [weak self] in
      self?.enforceLease(binding, connectionID: connectionID)
    }
  }

  private func expireBinding(_ binding: Binding) {
    guard bindings[binding.ticket] === binding else { return }
    bindings.removeValue(forKey: binding.ticket)
    if attemptTickets[binding.attemptID] == binding.ticket {
      attemptTickets.removeValue(forKey: binding.attemptID)
    }
    let connectionIDs = Array(binding.activeConnections.keys)
    connectionIDs.forEach { abortUpload(connectionID: $0) }
    binding.activeConnections.values.forEach { $0.cancel() }
    binding.activeConnections.removeAll()
    connectionIDs.forEach { connectionTickets.removeValue(forKey: $0) }
    releaseUploadReservation(binding)
    removePartFile(binding)
  }

  private func serveInput(
    _ binding: Binding,
    rangeHeader: String?,
    headOnly: Bool,
    on connection: NWConnection
  ) {
    guard (try? SourceSignature.read(fileDescriptor: binding.source.fileDescriptor))
      == binding.source.signature
    else {
      sendStatus(409, "Conflict", on: connection)
      return
    }
    let size = binding.source.signature.bytes
    guard let byteRange = parseRange(rangeHeader, size: size) else {
      sendStatus(416, "Range Not Satisfiable", headers: [
        "Content-Range": "bytes */\(size)"
      ], on: connection)
      return
    }
    let isPartial = rangeHeader != nil
    let contentLength = byteRange.upperBound - byteRange.lowerBound + 1
    var headers = [
      "Accept-Ranges": "bytes",
      "Content-Type": Self.mimeType(binding.source.url.pathExtension),
      "Content-Length": String(contentLength),
      "ETag": "\"\(binding.source.sha256)\"",
      "Cache-Control": "private, max-age=30",
    ]
    if isPartial {
      headers["Content-Range"] = "bytes \(byteRange.lowerBound)-\(byteRange.upperBound)/\(size)"
    }
    let status = isPartial ? (206, "Partial Content") : (200, "OK")
    if headOnly {
      sendResponse(status.0, status.1, headers: headers, body: nil, on: connection)
      return
    }
    sendHeaders(status.0, status.1, headers: headers, complete: false, on: connection) {
      [weak self] error in
      guard let self else { return }
      if error != nil {
        connection.cancel()
        return
      }
      self.queue.async {
        self.sendFile(
          binding: binding,
          offset: byteRange.lowerBound,
          remaining: contentLength,
          on: connection
        )
      }
    }
  }

  private func sendFile(
    binding: Binding,
    offset: Int64,
    remaining: Int64,
    on connection: NWConnection
  ) {
    guard bindings[binding.ticket] === binding else {
      connection.cancel()
      return
    }
    guard remaining > 0 else {
      connection.send(content: nil, isComplete: true, completion: .contentProcessed { _ in
        connection.cancel()
      })
      return
    }
    let requested = Int(min(1_048_576, remaining))
    var data = Data(count: requested)
    let count = data.withUnsafeMutableBytes { bytes -> Int in
      guard let base = bytes.baseAddress else { return -1 }
      return pread(binding.source.fileDescriptor, base, requested, off_t(offset))
    }
    guard count > 0 else {
      connection.cancel()
      return
    }
    if count < data.count { data.removeSubrange(count..<data.count) }
    connection.send(content: data, isComplete: false, completion: .contentProcessed {
      [weak self] error in
      guard let self else { return }
      if error != nil {
        connection.cancel()
        return
      }
      self.queue.async {
        self.sendFile(
          binding: binding,
          offset: offset + Int64(count),
          remaining: remaining - Int64(count),
          on: connection
        )
      }
    })
  }

  private func receiveOutput(
    _ request: HTTPRequestHead,
    binding: Binding,
    connectionID: UUID,
    on connection: NWConnection
  ) {
    guard let value = request.headers["content-length"],
      let expected = Int64(value), expected > 0,
      expected <= binding.maximumOutputBytes,
      request.initialBody.count <= expected
    else {
      sendStatus(413, "Content Too Large", on: connection)
      return
    }
    guard !binding.uploadInProgress else {
      // A transport retry can arrive before Network.framework reports the
      // previous connection closed. Tell the client this is transient without
      // weakening real 409 digest/output conflicts.
      sendStatus(425, "Too Early", on: connection)
      return
    }
    guard expected <= maximumAggregateOutputBytes - aggregateOutputBytes else {
      sendStatus(507, "Insufficient Storage", on: connection)
      return
    }
    aggregateOutputBytes += expected
    binding.currentUploadReservation = expected
    binding.uploadInProgress = true
    let part = partURL(binding)
    do {
      try FileManager.default.createDirectory(
        at: binding.outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
      )
      try? FileManager.default.removeItem(at: part)
      guard FileManager.default.createFile(atPath: part.path, contents: nil) else {
        throw CocoaError(.fileWriteUnknown)
      }
      let handle = try FileHandle(forWritingTo: part)
      var hasher = SHA256()
      if !request.initialBody.isEmpty {
        try handle.write(contentsOf: request.initialBody)
        hasher.update(data: request.initialBody)
      }
      let state = UploadState(
        binding: binding,
        connectionID: connectionID,
        expected: expected,
        handle: handle,
        part: part,
        received: Int64(request.initialBody.count),
        hasher: hasher
      )
      uploads[connectionID] = state
      receiveOutputBody(state, on: connection)
    } catch {
      binding.uploadInProgress = false
      releaseUploadReservation(binding)
      try? FileManager.default.removeItem(at: part)
      sendStatus(500, "Internal Server Error", on: connection)
    }
  }

  private func receiveOutputBody(_ state: UploadState, on connection: NWConnection) {
    guard uploads[state.connectionID] === state else { return }
    if state.received == state.expected {
      finalizeOutput(state, on: connection)
      return
    }
    scheduleUploadIdleTimeout(state, on: connection)
    connection.receive(minimumIncompleteLength: 1, maximumLength: 1_048_576) {
      [weak self] data, _, complete, error in
      guard let self else { return }
      guard self.uploads[state.connectionID] === state else { return }
      guard error == nil, let data, !data.isEmpty,
        state.received + Int64(data.count) <= state.expected
      else {
        self.abortUpload(connectionID: state.connectionID)
        self.sendStatus(400, "Bad Request", on: connection)
        return
      }
      do {
        try state.handle.write(contentsOf: data)
        state.hasher.update(data: data)
        state.received += Int64(data.count)
        self.receiveOutputBody(state, on: connection)
      } catch {
        self.abortUpload(connectionID: state.connectionID)
        self.sendStatus(500, "Internal Server Error", on: connection)
      }
      _ = complete
    }
  }

  private func scheduleUploadIdleTimeout(_ state: UploadState, on connection: NWConnection) {
    state.idleTimeout?.cancel()
    let timeout = DispatchWorkItem { [weak self, weak connection] in
      guard let self, let connection, self.uploads[state.connectionID] === state else { return }
      self.abortUpload(connectionID: state.connectionID)
      self.sendStatus(408, "Request Timeout", on: connection)
    }
    state.idleTimeout = timeout
    queue.asyncAfter(deadline: .now() + Self.uploadIdleTimeout, execute: timeout)
  }

  private func abortUpload(connectionID: UUID) {
    guard let state = uploads.removeValue(forKey: connectionID) else { return }
    state.close()
    try? FileManager.default.removeItem(at: state.part)
    state.binding.uploadInProgress = false
    releaseUploadReservation(state.binding)
  }

  private func finalizeOutput(_ state: UploadState, on connection: NWConnection) {
    guard uploads.removeValue(forKey: state.connectionID) === state else { return }
    let binding = state.binding
    let expected = state.expected
    let part = state.part
    do {
      state.idleTimeout?.cancel()
      try state.handle.synchronize()
      try state.handle.close()
      state.closed = true
      let digest = state.hasher.finalize().map { String(format: "%02x", $0) }.joined()
      if let publishedBytes = binding.publishedByteCount,
        let publishedSHA = binding.publishedSHA256
      {
        try? FileManager.default.removeItem(at: part)
        guard publishedBytes == expected, publishedSHA == digest else {
          throw RemoteClusterHTTPTransferError.invalidContract
        }
        // The published file was counted by the first upload; a retry's
        // temporary reservation is released after byte-for-byte confirmation.
        releaseUploadReservation(binding)
      } else {
        guard !FileManager.default.fileExists(atPath: binding.outputURL.path) else {
          throw RemoteClusterHTTPTransferError.invalidContract
        }
        try FileManager.default.moveItem(at: part, to: binding.outputURL)
        binding.publishedByteCount = expected
        binding.publishedSHA256 = digest
        // Convert the in-progress reservation into committed session bytes.
        binding.currentUploadReservation = 0
      }
      binding.uploadInProgress = false
      sendResponse(201, "Created", headers: [
        "X-Mioh-Output-SHA256": digest
      ], body: nil, on: connection)
    } catch {
      binding.uploadInProgress = false
      releaseUploadReservation(binding)
      try? FileManager.default.removeItem(at: part)
      sendStatus(409, "Conflict", on: connection)
    }
  }

  private func releaseUploadReservation(_ binding: Binding) {
    guard binding.currentUploadReservation > 0 else { return }
    aggregateOutputBytes = max(0, aggregateOutputBytes - binding.currentUploadReservation)
    binding.currentUploadReservation = 0
  }

  private func parseRange(_ value: String?, size: Int64) -> ClosedRange<Int64>? {
    guard size > 0 else { return nil }
    guard let value else { return 0...(size - 1) }
    guard value.hasPrefix("bytes="), !value.contains(",") else { return nil }
    let pieces = value.dropFirst(6).split(separator: "-", omittingEmptySubsequences: false)
    guard pieces.count == 2 else { return nil }
    if pieces[0].isEmpty {
      guard let suffix = Int64(pieces[1]), suffix > 0 else { return nil }
      let length = min(size, suffix)
      return (size - length)...(size - 1)
    }
    guard let start = Int64(pieces[0]), start >= 0, start < size else { return nil }
    let end: Int64
    if pieces[1].isEmpty {
      end = size - 1
    } else {
      guard let parsed = Int64(pieces[1]), parsed >= start else { return nil }
      end = min(size - 1, parsed)
    }
    return start...end
  }

  private func sendStatus(
    _ code: Int,
    _ reason: String,
    headers: [String: String] = [:],
    on connection: NWConnection
  ) {
    sendResponse(code, reason, headers: headers, body: Data(), on: connection)
  }

  private func sendResponse(
    _ code: Int,
    _ reason: String,
    headers: [String: String],
    body: Data?,
    on connection: NWConnection
  ) {
    var allHeaders = headers
    if allHeaders["Content-Length"] == nil {
      allHeaders["Content-Length"] = String(body?.count ?? 0)
    }
    sendHeaders(
      code,
      reason,
      headers: allHeaders,
      complete: body == nil || body?.isEmpty == true,
      on: connection
    ) {
      error in
      guard error == nil, let body, !body.isEmpty else {
        connection.cancel()
        return
      }
      connection.send(content: body, isComplete: true, completion: .contentProcessed { _ in
        connection.cancel()
      })
    }
  }

  private func sendHeaders(
    _ code: Int,
    _ reason: String,
    headers: [String: String],
    complete: Bool,
    on connection: NWConnection,
    completion: @escaping (Error?) -> Void
  ) {
    var text = "HTTP/1.1 \(code) \(reason)\r\n"
    for (key, value) in headers.sorted(by: { $0.key < $1.key }) {
      text += "\(key): \(value)\r\n"
    }
    text += "Connection: close\r\n\r\n"
    connection.send(
      content: Data(text.utf8),
      isComplete: complete,
      completion: .contentProcessed(completion)
    )
  }

  private func partURL(_ binding: Binding) -> URL {
    binding.outputURL.deletingLastPathComponent().appendingPathComponent(
      ".mioh-http-\(binding.attemptID.uuidString.lowercased()).part"
    )
  }

  private func removePartFile(_ binding: Binding) {
    try? FileManager.default.removeItem(at: partURL(binding))
  }

  private static var defaultAdvertisedHost: String {
    let raw = ProcessInfo.processInfo.hostName.trimmingCharacters(in: .whitespacesAndNewlines)
    if raw.isEmpty { return "localhost" }
    return raw.contains(".") ? raw : raw + ".local"
  }

  private static func endpoint(
    host: String,
    port: UInt16,
    ticket: String,
    leaf: String
  ) -> URL? {
    var components = URLComponents()
    components.scheme = "http"
    components.host = host
    components.port = Int(port)
    components.path = "/mioh-cluster/v1/\(ticket)/\(leaf)"
    return components.url
  }

  private static func randomTicket() -> String {
    var bytes = [UInt8](repeating: 0, count: 32)
    let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
    precondition(status == errSecSuccess)
    return Data(bytes).base64EncodedString()
      .replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "=", with: "")
  }

  private static func mimeType(_ pathExtension: String) -> String {
    switch pathExtension.lowercased() {
    case "mov": return "video/quicktime"
    case "m4v": return "video/x-m4v"
    default: return "video/mp4"
    }
  }
}

private final class RemoteClusterTransferOnce: @unchecked Sendable {
  private let lock = NSLock()
  private var claimed = false

  func claim() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard !claimed else { return false }
    claimed = true
    return true
  }
}

enum RemoteClusterHTTPTransferClient {
  private static let transferTimeout: TimeInterval = 30 * 60

  /// URLSession's file upload streams from disk and does not materialize the
  /// encoded shard as one Data allocation.
  static func upload(
    file: URL,
    descriptor: RemoteClusterHTTPTransferDescriptor
  ) async throws {
    guard descriptor.isValid, let destination = URL(string: descriptor.outputURL) else {
      throw RemoteClusterHTTPTransferError.invalidContract
    }
    let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true, let size = values.fileSize,
      size > 0, Int64(size) <= descriptor.maximumOutputBytes
    else { throw RemoteClusterHTTPTransferError.invalidContract }
    var request = URLRequest(url: destination)
    request.httpMethod = "PUT"
    request.setValue(String(size), forHTTPHeaderField: "Content-Length")
    request.setValue("video/mp4", forHTTPHeaderField: "Content-Type")
    // `expiresAt` is the initial wire snapshot. The Coordinator can renew the
    // server-side lease while a long job runs, so deriving this timeout from a
    // now-stale descriptor would incorrectly cap a valid upload at 60 seconds.
    request.timeoutInterval = transferTimeout
    request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    let configuration = URLSessionConfiguration.ephemeral
    configuration.timeoutIntervalForRequest = transferTimeout
    configuration.timeoutIntervalForResource = 24 * 60 * 60
    configuration.urlCache = nil
    configuration.httpCookieStorage = nil
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let retryableStatus = Set([408, 425, 429])
    for attempt in 0..<3 {
      try Task.checkCancellation()
      do {
        let (_, response) = try await session.upload(for: request, fromFile: file)
        guard let http = response as? HTTPURLResponse else {
          throw RemoteClusterHTTPTransferError.uploadFailed("non_http_response")
        }
        if (200...299).contains(http.statusCode) { return }
        let canRetry = retryableStatus.contains(http.statusCode)
          || (500...599).contains(http.statusCode)
        guard canRetry, attempt < 2 else {
          throw RemoteClusterHTTPTransferError.uploadFailed("HTTP \(http.statusCode)")
        }
      } catch is CancellationError {
        throw CancellationError()
      } catch let error as RemoteClusterHTTPTransferError {
        throw error
      } catch {
        guard attempt < 2 else {
          throw RemoteClusterHTTPTransferError.uploadFailed(error.localizedDescription)
        }
      }
      let delayNanoseconds = UInt64(250_000_000) << UInt64(attempt)
      try await Task.sleep(nanoseconds: delayNanoseconds)
    }
    throw RemoteClusterHTTPTransferError.uploadFailed("retry_exhausted")
  }
}
