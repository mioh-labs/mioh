import Foundation
import NIOCore
import NIOPosix
import NIOSFTP
import NIOSSH

private struct MiohSFTPProbeComplete: Error {}

/// Bridges one NIO future to Swift cancellation without closing the shared SSH
/// channel. SFTP request IDs remain independently multiplexed, so a cancelled
/// AVFoundation range consumer can discard its response while newer seek
/// ranges continue on the same pinned connection.
private final class MiohSFTPDiscardableFutureWaiter<Value>: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: CheckedContinuation<Value, Error>?
  private var result: Result<Value, Error>?
  private var cancelled = false

  func wait() async throws -> Value {
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Value, Error>) in
      lock.lock()
      if cancelled {
        lock.unlock()
        continuation.resume(throwing: CancellationError())
      } else if let result {
        lock.unlock()
        continuation.resume(with: result)
      } else {
        self.continuation = continuation
        lock.unlock()
      }
    }
  }

  func complete(_ result: Result<Value, Error>) {
    lock.lock()
    guard !cancelled else {
      lock.unlock()
      return
    }
    if let continuation {
      self.continuation = nil
      lock.unlock()
      continuation.resume(with: result)
    } else {
      self.result = result
      lock.unlock()
    }
  }

  func cancel() {
    lock.lock()
    guard !cancelled else {
      lock.unlock()
      return
    }
    cancelled = true
    if let continuation {
      self.continuation = nil
      lock.unlock()
      continuation.resume(throwing: CancellationError())
    } else {
      lock.unlock()
    }
  }
}

struct MiohSFTPDownloadResumeMetadata: Codable, Equatable, Sendable {
  static let currentVersion = 1

  let version: Int
  let remotePath: String
  let expectedSize: UInt64
  let expectedModificationTime: UInt32?

  init(
    remotePath: String,
    expectedSize: UInt64,
    expectedModificationTime: UInt32?
  ) {
    version = Self.currentVersion
    self.remotePath = remotePath
    self.expectedSize = expectedSize
    self.expectedModificationTime = expectedModificationTime
  }
}

private final class MiohSFTPRejectingAuthDelegate: NIOSSHClientUserAuthenticationDelegate,
  @unchecked Sendable
{
  func nextAuthenticationType(
    availableMethods: NIOSSHAvailableUserAuthenticationMethods,
    nextChallengePromise: EventLoopPromise<NIOSSHUserAuthenticationOffer?>
  ) {
    nextChallengePromise.succeed(nil)
  }
}

private final class MiohSFTPPasswordAuthDelegate: NIOSSHClientUserAuthenticationDelegate,
  @unchecked Sendable
{
  private let username: String
  private var password: String
  private let lock = NSLock()
  private var attempted = false

  init(username: String, password: String) {
    self.username = username
    self.password = password
  }

  func nextAuthenticationType(
    availableMethods: NIOSSHAvailableUserAuthenticationMethods,
    nextChallengePromise: EventLoopPromise<NIOSSHUserAuthenticationOffer?>
  ) {
    lock.lock()
    let shouldAttempt = !attempted && availableMethods.contains(.password)
    attempted = true
    let password = self.password
    self.password = ""
    lock.unlock()

    guard shouldAttempt else {
      nextChallengePromise.succeed(nil)
      return
    }
    nextChallengePromise.succeed(
      NIOSSHUserAuthenticationOffer(
        username: username,
        serviceName: "ssh-connection",
        offer: .password(.init(password: password))
      )
    )
  }
}

private final class MiohSFTPHostKeyCollector: @unchecked Sendable {
  private let lock = NSLock()
  private let promise: EventLoopPromise<NIOSSHPublicKey>
  private var completed = false

  init(promise: EventLoopPromise<NIOSSHPublicKey>) {
    self.promise = promise
  }

  func capture(_ key: NIOSSHPublicKey) {
    lock.lock()
    guard !completed else {
      lock.unlock()
      return
    }
    completed = true
    lock.unlock()
    promise.succeed(key)
  }

  func fail(_ error: Error) {
    lock.lock()
    guard !completed else {
      lock.unlock()
      return
    }
    completed = true
    lock.unlock()
    promise.fail(error)
  }
}

private final class MiohSFTPCapturingHostKeyDelegate: NIOSSHClientServerAuthenticationDelegate,
  @unchecked Sendable
{
  private let collector: MiohSFTPHostKeyCollector

  init(collector: MiohSFTPHostKeyCollector) {
    self.collector = collector
  }

  func validateHostKey(
    hostKey: NIOSSHPublicKey,
    validationCompletePromise: EventLoopPromise<Void>
  ) {
    collector.capture(hostKey)
    validationCompletePromise.fail(MiohSFTPProbeComplete())
  }
}

private final class MiohSFTPPinnedHostKeyDelegate: NIOSSHClientServerAuthenticationDelegate,
  @unchecked Sendable
{
  private let expected: NIOSSHPublicKey
  private let expectedIdentity: MiohSFTPHostIdentity

  init(expected: NIOSSHPublicKey, expectedIdentity: MiohSFTPHostIdentity) {
    self.expected = expected
    self.expectedIdentity = expectedIdentity
  }

  func validateHostKey(
    hostKey: NIOSSHPublicKey,
    validationCompletePromise: EventLoopPromise<Void>
  ) {
    guard hostKey == expected else {
      let actual = try? MiohSFTPHostIdentity(
        shortHandKey: String(openSSHPublicKey: hostKey)
      )
      validationCompletePromise.fail(
        MiohSFTPError.hostKeyChanged(
          expected: expectedIdentity.fingerprint,
          actual: actual?.fingerprint ?? "不明"
        )
      )
      return
    }
    validationCompletePromise.succeed(())
  }
}

public actor MiohSFTPSession {
  public static let maximumDirectoryEntries = 5_000
  public static let transferBufferBytes = 256 * 1_024
  public static let maximumRangeReadBytes = 1 * 1_024 * 1_024

  private static let operationTimeout: TimeInterval = 30
  // OpenSSH and many NAS SFTP servers cap each READ reply at 32--64 KiB even
  // when a larger length was requested. Keep a four-MiB download window made
  // of server-friendly requests, and refill every short slot as one parallel
  // round so a single short reply can never collapse the transfer to one RTT
  // at a time.
  private static let readRequestBytes = 64 * 1_024
  private static let downloadReadPipelineDepth = 64
  private static let downloadReadWindowBytes =
    readRequestBytes * downloadReadPipelineDepth
  private static let progressIntervalBytes: Int64 = 4 * 1_024 * 1_024
  private static let maximumResumeMetadataBytes = 16 * 1_024
  private static let fileTypeMask: UInt32 = 0o170000
  private static let regularFileType: UInt32 = 0o100000
  private static let directoryFileType: UInt32 = 0o040000
  private static let symbolicLinkFileType: UInt32 = 0o120000

  private struct Connection {
    let channel: Channel
    let sftp: SFTPClient
  }

  private struct OpenStreamingFile {
    let descriptor: MiohSFTPStreamingFile
    let handle: SFTPFileHandle
    let expectedSize: UInt64
    let expectedModificationTime: UInt32
  }

  private struct PendingRead {
    let slotIndex: Int
    let offset: UInt64
    let length: UInt32
    let future: EventLoopFuture<ByteBuffer?>
  }

  private struct ReadSlot {
    let offset: UInt64
    let length: Int
    var data = Data()
  }

  private enum RemoteObjectKind {
    case regularFile
    case directory
    case symbolicLink
    case unsupported
  }

  private var connection: Connection?
  private var openStreamingFile: OpenStreamingFile?

  public init() {}

  public static func fetchHostIdentity(
    configuration: MiohSFTPConfiguration,
    timeout: TimeInterval = 10
  ) async throws -> MiohSFTPHostIdentity {
    let configuration = try configuration.validated()
    let eventLoop = MultiThreadedEventLoopGroup.singleton.next()
    let hostKeyPromise = eventLoop.makePromise(of: NIOSSHPublicKey.self)
    let collector = MiohSFTPHostKeyCollector(promise: hostKeyPromise)
    let timeoutAmount = nioTimeout(timeout)
    let bootstrap = ClientBootstrap(group: eventLoop)
      .connectTimeout(timeoutAmount)
      .channelInitializer { channel in
        channel.eventLoop.makeCompletedFuture {
          try channel.pipeline.syncOperations.addHandler(
            NIOSSHHandler(
              role: .client(
                SSHClientConfiguration(
                  userAuthDelegate: MiohSFTPRejectingAuthDelegate(),
                  serverAuthDelegate: MiohSFTPCapturingHostKeyDelegate(
                    collector: collector
                  )
                )
              ),
              allocator: channel.allocator,
              inboundChildChannelInitializer: nil
            )
          )
        }
      }

    let channel = try await bootstrap.connect(
      host: configuration.host,
      port: configuration.port
    ).get()
    let timeoutTask = eventLoop.scheduleTask(in: timeoutAmount) {
      collector.fail(ChannelError.connectTimeout(timeoutAmount))
      channel.close(promise: nil)
    }
    defer {
      timeoutTask.cancel()
      channel.close(promise: nil)
    }

    let key = try await withTaskCancellationHandler {
      try await hostKeyPromise.futureResult.get()
    } onCancel: {
      collector.fail(CancellationError())
      channel.close(promise: nil)
    }
    return try MiohSFTPHostIdentity(
      shortHandKey: String(openSSHPublicKey: key)
    )
  }

  @discardableResult
  public func connect(
    configuration: MiohSFTPConfiguration,
    trustedHostKey: String,
    timeout: TimeInterval = 15
  ) async throws -> String {
    let configuration = try configuration.validated()
    let identity = try MiohSFTPHostIdentity(shortHandKey: trustedHostKey)
    let expectedKey: NIOSSHPublicKey
    do {
      expectedKey = try NIOSSHPublicKey(openSSHPublicKey: identity.key)
    } catch {
      throw MiohSFTPError.invalidHostKey
    }
    await disconnect()

    let eventLoop = MultiThreadedEventLoopGroup.singleton.next()
    let timeoutAmount = Self.nioTimeout(timeout)
    let bootstrap = ClientBootstrap(group: eventLoop)
      .connectTimeout(timeoutAmount)
      .channelInitializer { channel in
        channel.eventLoop.makeCompletedFuture {
          try channel.pipeline.syncOperations.addHandler(
            NIOSSHHandler(
              role: .client(
                SSHClientConfiguration(
                  userAuthDelegate: MiohSFTPPasswordAuthDelegate(
                    username: configuration.username,
                    password: configuration.password
                  ),
                  serverAuthDelegate: MiohSFTPPinnedHostKeyDelegate(
                    expected: expectedKey,
                    expectedIdentity: identity
                  )
                )
              ),
              allocator: channel.allocator,
              inboundChildChannelInitializer: nil
            )
          )
        }
      }

    let channel = try await bootstrap.connect(
      host: configuration.host,
      port: configuration.port
    ).get()
    do {
      let sshHandler = try await Self.awaitNetwork(
        channel.pipeline.handler(type: NIOSSHHandler.self),
        channel: channel,
        timeout: timeout
      )
      let sftp = try await Self.awaitNetwork(
        SFTPClient.openChannel(with: sshHandler, on: channel),
        channel: channel,
        timeout: timeout
      )
      let newConnection = Connection(channel: channel, sftp: sftp)
      connection = newConnection
      do {
        let home = try await Self.awaitNetwork(
          sftp.realpath("."),
          channel: channel
        )
        return try MiohSFTPPath.normalize(home)
      } catch {
        connection = nil
        sftp.channel.close(promise: nil)
        channel.close(promise: nil)
        throw error
      }
    } catch {
      channel.close(promise: nil)
      throw error
    }
  }

  public func disconnect() async {
    guard let connection else { return }
    self.connection = nil
    openStreamingFile = nil
    connection.sftp.channel.close(promise: nil)
    connection.channel.close(promise: nil)
    _ = try? await connection.channel.closeFuture.get()
  }

  /// Opens one regular movie for bounded random access. Call this only on a
  /// session dedicated to streaming; browser listing and download sessions
  /// retain their independent cancellation lifetime.
  public func openMovieForStreaming(
    remotePath: String,
    expectedByteCount: Int64? = nil
  ) async throws -> MiohSFTPStreamingFile {
    let connection = try connected()
    let cleanPath = try MiohSFTPPath.normalize(remotePath)
    guard MiohSFTPPath.isSupportedMovie(cleanPath) else {
      throw MiohSFTPError.unsupportedFile
    }
    let pathAttributes = try await validatePathComponents(
      cleanPath,
      connection: connection
    )
    guard Self.kind(of: pathAttributes) == .regularFile,
      let pathSize = pathAttributes.size,
      let pathModificationTime = pathAttributes.modificationTime,
      pathSize > 0,
      pathSize <= UInt64(Int64.max)
    else { throw MiohSFTPError.unsupportedFile }
    try MiohSFTPTransferPolicy.validate(
      byteCount: pathSize,
      maximumBytes: MiohSFTPTransferPolicy.hardMaximumBytes
    )
    if let expectedByteCount {
      guard expectedByteCount > 0, UInt64(expectedByteCount) == pathSize else {
        throw MiohSFTPError.incompleteTransfer
      }
    }

    if let previous = openStreamingFile {
      openStreamingFile = nil
      _ = try? await Self.awaitCleanup(
        connection.sftp.closeFile(previous.handle),
        channel: connection.channel
      )
    }

    let handle = try await Self.awaitNetwork(
      connection.sftp.openFile(path: cleanPath, flags: [.read]),
      channel: connection.channel
    )
    do {
      let openedAttributes = try await Self.awaitNetwork(
        connection.sftp.fstat(file: handle),
        channel: connection.channel
      )
      guard Self.kind(of: openedAttributes) == .regularFile,
        openedAttributes.size == pathSize,
        openedAttributes.modificationTime == pathModificationTime
      else { throw MiohSFTPError.incompleteTransfer }
      let descriptor = MiohSFTPStreamingFile(
        id: UUID(),
        byteCount: Int64(pathSize),
        pathExtension: URL(fileURLWithPath: cleanPath).pathExtension.lowercased()
      )
      openStreamingFile = OpenStreamingFile(
        descriptor: descriptor,
        handle: handle,
        expectedSize: pathSize,
        expectedModificationTime: pathModificationTime
      )
      return descriptor
    } catch {
      _ = try? await Self.awaitCleanup(
        connection.sftp.closeFile(handle),
        channel: connection.channel
      )
      throw error
    }
  }

  /// Reads one bounded byte range without creating a local file. Short SFTP
  /// replies are refilled as parallel rounds and a changed/truncated remote
  /// file fails closed so AVFoundation never combines bytes from different
  /// revisions.
  public func readMovieRange(
    file: MiohSFTPStreamingFile,
    offset: UInt64,
    length: Int
  ) async throws -> Data {
    try Task.checkCancellation()
    let connection = try connected()
    guard let opened = openStreamingFile,
      opened.descriptor.id == file.id,
      opened.descriptor == file,
      length > 0,
      length <= Self.maximumRangeReadBytes,
      offset < opened.expectedSize
    else { throw MiohSFTPError.invalidRemotePath }

    let requested = min(
      UInt64(length),
      opened.expectedSize - offset
    )
    // FSTAT and the first READ round are sent without an intervening await, so
    // revision validation does not add another network round trip.
    let attributesFuture = connection.sftp.fstat(file: opened.handle)
    let attributesTask = Task {
      try await Self.awaitDiscardableNetwork(
        attributesFuture,
        channel: connection.channel
      )
    }
    defer { attributesTask.cancel() }
    let chunks = try await Self.readPipelinedRange(
      file: opened.handle,
      offset: offset,
      length: Int(requested),
      maximumSlots: Int(
        (requested + UInt64(Self.readRequestBytes) - 1)
          / UInt64(Self.readRequestBytes)
      ),
      connection: connection,
      interruptOnCancellation: false
    )
    var result = Data()
    result.reserveCapacity(Int(requested))
    for chunk in chunks { result.append(chunk) }
    guard UInt64(result.count) == requested else {
      throw MiohSFTPError.incompleteTransfer
    }
    // Validate after the read so a write that races this exact page is
    // rejected before any bytes cross the loopback response boundary.
    let attributes = try await attributesTask.value
    guard Self.kind(of: attributes) == .regularFile,
      attributes.size == opened.expectedSize,
      attributes.modificationTime == opened.expectedModificationTime
    else { throw MiohSFTPError.incompleteTransfer }
    return result
  }

  public func closeMovieStream(_ file: MiohSFTPStreamingFile) async {
    guard let connection, let opened = openStreamingFile,
      opened.descriptor.id == file.id
    else { return }
    openStreamingFile = nil
    _ = try? await Self.awaitCleanup(
      connection.sftp.closeFile(opened.handle),
      channel: connection.channel
    )
  }

  public func listDirectory(
    _ path: String,
    maximumEntries: Int = maximumDirectoryEntries
  ) async throws -> [MiohSFTPEntry] {
    let connection = try connected()
    let cleanPath = try MiohSFTPPath.normalize(path)
    let directoryAttributes = try await validatePathComponents(
      cleanPath,
      connection: connection
    )
    guard Self.kind(of: directoryAttributes) == .directory else {
      throw MiohSFTPError.invalidRemotePath
    }

    let handle = try await Self.awaitNetwork(
      connection.sftp.openDirectory(path: cleanPath),
      channel: connection.channel
    )
    let entryLimit = max(0, min(maximumEntries, Self.maximumDirectoryEntries))
    var receivedEntryCount = 0
    var entries: [MiohSFTPEntry] = []
    entries.reserveCapacity(min(entryLimit, 256))
    do {
      while let batch = try await Self.awaitNetwork(
        connection.sftp.readDirectoryBatch(handle),
        channel: connection.channel
      ) {
        guard !batch.isEmpty else {
          throw MiohSFTPError.incompleteTransfer
        }
        guard batch.count <= entryLimit - receivedEntryCount else {
          throw MiohSFTPError.tooManyEntries(maximum: entryLimit)
        }
        receivedEntryCount += batch.count

        // Convert and release each raw NAME packet immediately. ByteBuffer
        // slices in longname/extended attributes may retain the whole frame;
        // accumulating raw batches would otherwise multiply the frame limit.
        for item in batch {
          guard item.filename.utf8.count <= 255,
            let itemPath = try? MiohSFTPPath.appending(
              item.filename,
              to: cleanPath
            )
          else { continue }
          let attributes: SFTPAttributes
          if Self.kind(of: item.attributes) == .unsupported {
            attributes = try await Self.awaitNetwork(
              connection.sftp.lstat(path: itemPath),
              channel: connection.channel
            )
          } else {
            attributes = item.attributes
          }
          let modifiedAt = attributes.modificationTime.map {
            Date(timeIntervalSince1970: TimeInterval($0))
          }
          switch Self.kind(of: attributes) {
          case .directory:
            entries.append(
              MiohSFTPEntry(
                name: item.filename,
                path: itemPath,
                kind: .directory,
                modifiedAt: modifiedAt
              )
            )
          case .regularFile where MiohSFTPPath.isSupportedMovie(item.filename):
            let byteCount = attributes.size.flatMap {
              $0 <= UInt64(Int64.max) ? Int64($0) : nil
            }
            entries.append(
              MiohSFTPEntry(
                name: item.filename,
                path: itemPath,
                kind: .movie,
                byteCount: byteCount,
                modifiedAt: modifiedAt
              )
            )
          case .symbolicLink, .regularFile, .unsupported:
            continue
          }
        }
      }
      try await Self.awaitNetwork(
        connection.sftp.closeDirectory(handle),
        channel: connection.channel
      )
    } catch {
      _ = try? await Self.awaitNetwork(
        connection.sftp.closeDirectory(handle),
        channel: connection.channel
      )
      throw error
    }

    return entries.sorted { left, right in
      if left.isDirectory != right.isDirectory { return left.isDirectory }
      return left.name.localizedCaseInsensitiveCompare(right.name) == .orderedAscending
    }
  }

  @discardableResult
  public func downloadMovie(
    remotePath: String,
    to destinationURL: URL,
    maximumBytes: Int64,
    resumeExistingPartial: Bool = false,
    progress: @escaping @Sendable (Int64, Int64) -> Void
  ) async throws -> URL {
    let connection = try connected()
    let cleanPath = try MiohSFTPPath.normalize(remotePath)
    guard MiohSFTPPath.isSupportedMovie(cleanPath) else {
      throw MiohSFTPError.unsupportedFile
    }
    let pathAttributes = try await validatePathComponents(
      cleanPath,
      connection: connection
    )
    guard Self.kind(of: pathAttributes) == .regularFile,
      let expectedSize = pathAttributes.size
    else { throw MiohSFTPError.unsupportedFile }
    try MiohSFTPTransferPolicy.validate(
      byteCount: expectedSize,
      maximumBytes: maximumBytes
    )
    guard destinationURL.isFileURL else {
      throw MiohSFTPError.localFile("保存先がローカルファイルではありません。")
    }

    let fileManager = FileManager.default
    let partialURL = destinationURL.appendingPathExtension("part")
    let metadataURL = destinationURL.appendingPathExtension("resume")
    guard !fileManager.fileExists(atPath: destinationURL.path) else {
      throw MiohSFTPError.localFile("保存先に同名ファイルがあります。")
    }
    let metadata = MiohSFTPDownloadResumeMetadata(
      remotePath: cleanPath,
      expectedSize: expectedSize,
      expectedModificationTime: pathAttributes.modificationTime
    )
    let initialOffset = try Self.prepareLocalDownload(
      partialURL: partialURL,
      metadataURL: metadataURL,
      metadata: metadata,
      allowsResume: resumeExistingPartial,
      fileManager: fileManager
    )

    var remoteHandle: SFTPFileHandle?
    var localHandle: FileHandle?
    do {
      let opened = try await Self.awaitNetwork(
        connection.sftp.openFile(path: cleanPath, flags: [.read]),
        channel: connection.channel
      )
      remoteHandle = opened
      let openedAttributes = try await Self.awaitNetwork(
        connection.sftp.fstat(file: opened),
        channel: connection.channel
      )
      guard Self.kind(of: openedAttributes) == .regularFile,
        openedAttributes.size == expectedSize,
        openedAttributes.modificationTime == pathAttributes.modificationTime
      else { throw MiohSFTPError.incompleteTransfer }

      let output = try FileHandle(forWritingTo: partialURL)
      localHandle = output
      let actualOffset = try output.seekToEnd()
      guard actualOffset == initialOffset else {
        throw MiohSFTPError.incompleteTransfer
      }
      var offset = initialOffset
      var lastReported = Int64(offset)
      progress(Int64(offset), Int64(expectedSize))
      while offset < expectedSize {
        try Task.checkCancellation()
        let windowLength = Int(
          min(
            UInt64(Self.downloadReadWindowBytes),
            expectedSize - offset
          )
        )
        let chunks = try await Self.readPipelinedRange(
          file: opened,
          offset: offset,
          length: windowLength,
          maximumSlots: Self.downloadReadPipelineDepth,
          connection: connection,
          interruptOnCancellation: true
        )
        for data in chunks {
          let nextOffset = offset + UInt64(data.count)
          guard nextOffset <= expectedSize,
            nextOffset <= UInt64(max(0, maximumBytes))
          else {
            throw MiohSFTPError.fileTooLarge(maximumBytes: maximumBytes)
          }
          try output.write(contentsOf: data)
          offset = nextOffset
        }
        let completed = Int64(offset)
        if completed == Int64(expectedSize)
          || completed - lastReported >= Self.progressIntervalBytes
        {
          progress(completed, Int64(expectedSize))
          lastReported = completed
        }
      }

      let trailingFuture = connection.sftp.read(
        file: opened,
        offset: expectedSize,
        length: 1
      )
      let finalAttributesFuture = connection.sftp.fstat(file: opened)
      let trailingData = try await Self.awaitNetwork(
        trailingFuture,
        channel: connection.channel
      )
      guard trailingData?.readableBytes ?? 0 == 0 else {
        throw MiohSFTPError.incompleteTransfer
      }
      let finalAttributes = try await Self.awaitNetwork(
        finalAttributesFuture,
        channel: connection.channel
      )
      guard finalAttributes.size == expectedSize,
        finalAttributes.modificationTime == pathAttributes.modificationTime
      else {
        throw MiohSFTPError.incompleteTransfer
      }
      try await Self.awaitNetwork(
        connection.sftp.closeFile(opened),
        channel: connection.channel
      )
      remoteHandle = nil
      try output.synchronize()
      try output.close()
      localHandle = nil

      // `URL.resourceValues` may retain the size observed before an append.
      // Recreate the URL so resume validation always observes the closed file.
      let completedURL = URL(fileURLWithPath: partialURL.path)
      let completedValues = try completedURL.resourceValues(forKeys: [
        .isRegularFileKey,
        .isSymbolicLinkKey,
        .fileSizeKey,
      ])
      guard completedValues.isRegularFile == true,
        completedValues.isSymbolicLink != true,
        completedValues.fileSize.map(Int64.init) == Int64(expectedSize)
      else {
        throw MiohSFTPError.incompleteTransfer
      }
      try fileManager.moveItem(at: partialURL, to: destinationURL)
      try? fileManager.removeItem(at: metadataURL)
      return destinationURL
    } catch {
      try? localHandle?.close()
      if let remoteHandle {
        _ = try? await Self.awaitNetwork(
          connection.sftp.closeFile(remoteHandle),
          channel: connection.channel
        )
      }
      if !resumeExistingPartial || !Self.shouldPreservePartial(after: error) {
        try? fileManager.removeItem(at: partialURL)
        try? fileManager.removeItem(at: metadataURL)
      }
      throw error
    }
  }

  @discardableResult
  public func uploadMovie(
    localURL: URL,
    to remotePath: String,
    maximumBytes: Int64 = MiohSFTPTransferPolicy.hardMaximumBytes,
    progress: @escaping @Sendable (Int64, Int64) -> Void
  ) async throws -> String {
    let connection = try connected()
    guard localURL.isFileURL, MiohSFTPPath.isSupportedMovie(localURL.lastPathComponent)
    else { throw MiohSFTPError.unsupportedFile }
    let values = try localURL.resourceValues(forKeys: [
      .isRegularFileKey,
      .isSymbolicLinkKey,
      .fileSizeKey,
    ])
    guard values.isRegularFile == true,
      values.isSymbolicLink != true,
      let localSize = values.fileSize
    else { throw MiohSFTPError.localFile("通常ファイルではありません。") }
    try MiohSFTPTransferPolicy.validate(
      byteCount: UInt64(localSize),
      maximumBytes: maximumBytes
    )

    let cleanPath = try MiohSFTPPath.normalize(remotePath)
    guard cleanPath != "/", MiohSFTPPath.isSupportedMovie(cleanPath) else {
      throw MiohSFTPError.invalidRemotePath
    }
    let directory = try MiohSFTPPath.parent(of: cleanPath)
    let directoryAttributes = try await validatePathComponents(
      directory,
      connection: connection
    )
    guard Self.kind(of: directoryAttributes) == .directory else {
      throw MiohSFTPError.invalidRemotePath
    }
    guard try await !pathExists(cleanPath, connection: connection) else {
      throw MiohSFTPError.remoteFileAlreadyExists
    }

    let temporaryPath = try MiohSFTPPath.appending(
      ".mioh-uploading-\(UUID().uuidString.lowercased()).part",
      to: directory
    )
    var remoteHandle: SFTPFileHandle?
    var localHandle: FileHandle?
    do {
      let opened = try await Self.awaitNetwork(
        connection.sftp.openFile(
          path: temporaryPath,
          flags: [.write, .create, .exclusive]
        ),
        channel: connection.channel,
        interruptOnCancellation: false
      )
      remoteHandle = opened
      let input = try FileHandle(forReadingFrom: localURL)
      localHandle = input
      var offset: UInt64 = 0
      var lastReported: Int64 = 0
      progress(0, Int64(localSize))
      while offset < UInt64(localSize) {
        try Task.checkCancellation()
        let remaining = UInt64(localSize) - offset
        let count = min(Self.transferBufferBytes, Int(remaining))
        guard let data = try input.read(upToCount: count), !data.isEmpty else {
          throw MiohSFTPError.incompleteTransfer
        }
        var buffer = ByteBufferAllocator().buffer(capacity: data.count)
        buffer.writeBytes(data)
        try await Self.awaitNetwork(
          connection.sftp.write(file: opened, offset: offset, data: buffer),
          channel: connection.channel,
          interruptOnCancellation: false
        )
        offset += UInt64(data.count)
        guard offset <= UInt64(localSize), offset <= UInt64(max(0, maximumBytes)) else {
          throw MiohSFTPError.fileTooLarge(maximumBytes: maximumBytes)
        }
        try Task.checkCancellation()
        let completed = Int64(offset)
        if completed == Int64(localSize)
          || completed - lastReported >= Self.progressIntervalBytes
        {
          progress(completed, Int64(localSize))
          lastReported = completed
        }
      }
      guard try input.read(upToCount: 1)?.isEmpty != false else {
        throw MiohSFTPError.incompleteTransfer
      }
      try Task.checkCancellation()
      try input.close()
      localHandle = nil
      try await Self.awaitNetwork(
        connection.sftp.closeFile(opened),
        channel: connection.channel,
        interruptOnCancellation: false
      )
      remoteHandle = nil
      try Task.checkCancellation()

      let uploaded = try await Self.awaitNetwork(
        connection.sftp.lstat(path: temporaryPath),
        channel: connection.channel,
        interruptOnCancellation: false
      )
      guard Self.kind(of: uploaded) == .regularFile,
        uploaded.size == UInt64(localSize)
      else { throw MiohSFTPError.incompleteTransfer }
      try Task.checkCancellation()
      guard try await !pathExists(
        cleanPath,
        connection: connection,
        interruptOnCancellation: false
      ) else {
        throw MiohSFTPError.remoteFileAlreadyExists
      }
      try Task.checkCancellation()

      if connection.sftp.supportsExtension(.hardlink) {
        try await Self.awaitNetwork(
          connection.sftp.hardlink(from: temporaryPath, to: cleanPath),
          channel: connection.channel,
          interruptOnCancellation: false
        )
        _ = try? await Self.awaitCleanup(
          connection.sftp.remove(path: temporaryPath),
          channel: connection.channel
        )
      } else {
        try await Self.awaitNetwork(
          connection.sftp.rename(from: temporaryPath, to: cleanPath),
          channel: connection.channel,
          interruptOnCancellation: false
        )
      }
      return cleanPath
    } catch {
      try? localHandle?.close()
      if let remoteHandle {
        _ = try? await Self.awaitCleanup(
          connection.sftp.closeFile(remoteHandle),
          channel: connection.channel
        )
      }
      _ = try? await Self.awaitCleanup(
        connection.sftp.remove(path: temporaryPath),
        channel: connection.channel
      )
      throw error
    }
  }

  /// Reads a bounded window as ordered slots. Every incomplete slot is refilled
  /// before awaiting the round, so a server that returns only 32 KiB for a
  /// 64-KiB request still keeps the whole transfer window busy. The returned
  /// chunks remain ordered and bounded; callers can write them without holding
  /// an entire movie in memory.
  private static func readPipelinedRange(
    file: SFTPFileHandle,
    offset: UInt64,
    length: Int,
    maximumSlots: Int,
    connection: Connection,
    interruptOnCancellation: Bool
  ) async throws -> [Data] {
    guard length > 0, maximumSlots > 0,
      UInt64(length) <= UInt64.max - offset
    else { throw MiohSFTPError.incompleteTransfer }

    var slots: [ReadSlot] = []
    slots.reserveCapacity(maximumSlots)
    var scheduledBytes = 0
    while scheduledBytes < length {
      guard slots.count < maximumSlots else {
        throw MiohSFTPError.incompleteTransfer
      }
      let slotLength = min(Self.readRequestBytes, length - scheduledBytes)
      slots.append(
        ReadSlot(
          offset: offset + UInt64(scheduledBytes),
          length: slotLength
        )
      )
      scheduledBytes += slotLength
    }

    while slots.contains(where: { $0.data.count < $0.length }) {
      try Task.checkCancellation()
      var pendingReads: [PendingRead] = []
      pendingReads.reserveCapacity(slots.count)
      for index in slots.indices where slots[index].data.count < slots[index].length {
        let completed = slots[index].data.count
        let remaining = slots[index].length - completed
        let requestLength = UInt32(min(Self.readRequestBytes, remaining))
        let requestOffset = slots[index].offset + UInt64(completed)
        pendingReads.append(
          PendingRead(
            slotIndex: index,
            offset: requestOffset,
            length: requestLength,
            future: connection.sftp.read(
              file: file,
              offset: requestOffset,
              length: requestLength
            )
          )
        )
      }
      guard !pendingReads.isEmpty else {
        throw MiohSFTPError.incompleteTransfer
      }

      for pending in pendingReads {
        guard slots.indices.contains(pending.slotIndex) else {
          throw MiohSFTPError.incompleteTransfer
        }
        let slotIndex = pending.slotIndex
        let expectedOffset = slots[slotIndex].offset
          + UInt64(slots[slotIndex].data.count)
        guard pending.offset == expectedOffset else {
          throw MiohSFTPError.incompleteTransfer
        }
        let response: ByteBuffer?
        if interruptOnCancellation {
          response = try await awaitNetwork(
            pending.future,
            channel: connection.channel,
            interruptOnCancellation: true
          )
        } else {
          response = try await awaitDiscardableNetwork(
            pending.future,
            channel: connection.channel
          )
        }
        guard let buffer = response, buffer.readableBytes > 0,
          buffer.readableBytes <= Int(pending.length)
        else { throw MiohSFTPError.incompleteTransfer }
        slots[slotIndex].data.append(
          contentsOf: buffer.readableBytesView
        )
        guard slots[slotIndex].data.count <= slots[slotIndex].length
        else { throw MiohSFTPError.incompleteTransfer }
        try Task.checkCancellation()
      }
    }

    guard slots.allSatisfy({ $0.data.count == $0.length }) else {
      throw MiohSFTPError.incompleteTransfer
    }
    return slots.map(\.data)
  }

  static func prepareLocalDownload(
    partialURL: URL,
    metadataURL: URL,
    metadata: MiohSFTPDownloadResumeMetadata,
    allowsResume: Bool,
    fileManager: FileManager
  ) throws -> UInt64 {
    let partialExists = fileManager.fileExists(atPath: partialURL.path)
      || (try? fileManager.destinationOfSymbolicLink(atPath: partialURL.path)) != nil
    let metadataExists = fileManager.fileExists(atPath: metadataURL.path)
      || (try? fileManager.destinationOfSymbolicLink(atPath: metadataURL.path)) != nil

    if !allowsResume {
      guard !partialExists, !metadataExists else {
        throw MiohSFTPError.localFile("保存先に同名ファイルがあります。")
      }
      guard fileManager.createFile(atPath: partialURL.path, contents: nil) else {
        throw MiohSFTPError.localFile("一時ファイルを作成できませんでした。")
      }
      return 0
    }

    if partialExists, metadataExists,
      let partialValues = try? partialURL.resourceValues(forKeys: [
        .isRegularFileKey,
        .isSymbolicLinkKey,
        .fileSizeKey,
      ]),
      partialValues.isRegularFile == true,
      partialValues.isSymbolicLink != true,
      let partialSize = partialValues.fileSize,
      partialSize >= 0,
      UInt64(partialSize) <= metadata.expectedSize,
      let metadataValues = try? metadataURL.resourceValues(forKeys: [
        .isRegularFileKey,
        .isSymbolicLinkKey,
        .fileSizeKey,
      ]),
      metadataValues.isRegularFile == true,
      metadataValues.isSymbolicLink != true,
      let metadataSize = metadataValues.fileSize,
      metadataSize > 0,
      metadataSize <= maximumResumeMetadataBytes,
      let encoded = try? Data(
        contentsOf: metadataURL,
        options: [.mappedIfSafe]
      ),
      encoded.count <= maximumResumeMetadataBytes,
      let saved = try? JSONDecoder().decode(
        MiohSFTPDownloadResumeMetadata.self,
        from: encoded
      ),
      saved == metadata,
      saved.version == MiohSFTPDownloadResumeMetadata.currentVersion
    {
      return UInt64(partialSize)
    }

    if partialExists { try? fileManager.removeItem(at: partialURL) }
    if metadataExists { try? fileManager.removeItem(at: metadataURL) }
    let partialStillExists = fileManager.fileExists(atPath: partialURL.path)
      || (try? fileManager.destinationOfSymbolicLink(atPath: partialURL.path)) != nil
    let metadataStillExists = fileManager.fileExists(atPath: metadataURL.path)
      || (try? fileManager.destinationOfSymbolicLink(atPath: metadataURL.path)) != nil
    guard !partialStillExists, !metadataStillExists
    else {
      throw MiohSFTPError.localFile("安全でない再開ファイルを置き換えられませんでした。")
    }

    let encoded = try JSONEncoder().encode(metadata)
    guard encoded.count <= maximumResumeMetadataBytes else {
      throw MiohSFTPError.localFile("再開情報が大きすぎます。")
    }
    try encoded.write(to: metadataURL, options: [.atomic])
    guard fileManager.createFile(atPath: partialURL.path, contents: nil) else {
      try? fileManager.removeItem(at: metadataURL)
      throw MiohSFTPError.localFile("一時ファイルを作成できませんでした。")
    }
    return 0
  }

  private static func shouldPreservePartial(after error: Error) -> Bool {
    if error is CancellationError { return true }
    guard let error = error as? MiohSFTPError else {
      // Transport failures are the primary resume case. Keep the bounded,
      // metadata-bound partial and validate it again on the next connection.
      return true
    }
    switch error {
    case .notConnected:
      return true
    case .invalidHost, .invalidPort, .missingUsername, .missingPassword,
      .invalidHostKey, .hostKeyChanged, .invalidRemotePath, .unsupportedFile,
      .tooManyEntries, .fileTooLarge, .insufficientStorage,
      .remoteFileAlreadyExists, .incompleteTransfer, .localFile:
      return false
    }
  }

  private func connected() throws -> Connection {
    guard let connection, connection.channel.isActive else {
      throw MiohSFTPError.notConnected
    }
    return connection
  }

  private func validatePathComponents(
    _ path: String,
    connection: Connection
  ) async throws -> SFTPAttributes {
    let cleanPath = try MiohSFTPPath.normalize(path)
    if cleanPath == "/" {
      let attributes = try await Self.awaitNetwork(
        connection.sftp.lstat(path: "/"),
        channel: connection.channel
      )
      guard Self.kind(of: attributes) != .symbolicLink else {
        throw MiohSFTPError.invalidRemotePath
      }
      return attributes
    }

    var currentPath = ""
    var finalAttributes = SFTPAttributes()
    let components = cleanPath.split(separator: "/", omittingEmptySubsequences: true)
    for (index, component) in components.enumerated() {
      currentPath += "/\(component)"
      let attributes = try await Self.awaitNetwork(
        connection.sftp.lstat(path: currentPath),
        channel: connection.channel
      )
      let kind = Self.kind(of: attributes)
      guard kind != .symbolicLink,
        index == components.index(before: components.endIndex) || kind == .directory
      else { throw MiohSFTPError.invalidRemotePath }
      finalAttributes = attributes
    }
    return finalAttributes
  }

  private func pathExists(
    _ path: String,
    connection: Connection,
    interruptOnCancellation: Bool = true
  ) async throws -> Bool {
    do {
      _ = try await Self.awaitNetwork(
        connection.sftp.lstat(path: path),
        channel: connection.channel,
        interruptOnCancellation: interruptOnCancellation
      )
      return true
    } catch let error as SFTPError {
      guard case .status(let status) = error else { throw error }
      if status.code == .noSuchFile { return false }
      let message = status.message.lowercased()
      if status.code == .failure,
        message.contains("no such file") || message.contains("not found")
      {
        return false
      }
      throw error
    }
  }

  private static func kind(of attributes: SFTPAttributes) -> RemoteObjectKind {
    guard let permissions = attributes.permissions else { return .unsupported }
    switch permissions & fileTypeMask {
    case regularFileType:
      return .regularFile
    case directoryFileType:
      return .directory
    case symbolicLinkFileType:
      return .symbolicLink
    default:
      return .unsupported
    }
  }

  private static func nioTimeout(_ seconds: TimeInterval) -> TimeAmount {
    let clamped = min(300, max(1, seconds))
    return .milliseconds(Int64(clamped * 1_000))
  }

  private static func awaitNetwork<Value>(
    _ future: EventLoopFuture<Value>,
    channel: Channel,
    timeout: TimeInterval = operationTimeout,
    interruptOnCancellation: Bool = true
  ) async throws -> Value {
    let timeoutTask = channel.eventLoop.scheduleTask(in: nioTimeout(timeout)) {
      channel.close(promise: nil)
    }
    defer { timeoutTask.cancel() }
    if interruptOnCancellation {
      return try await withTaskCancellationHandler {
        try await future.get()
      } onCancel: {
        channel.close(promise: nil)
      }
    }
    return try await future.get()
  }

  /// Stops waiting immediately when an AVFoundation range request is
  /// cancelled, but deliberately leaves the underlying SFTP request alive so
  /// its eventual response can be matched and discarded by request ID. The
  /// shared channel is closed only if the server fails to answer by the normal
  /// operation deadline.
  private static func awaitDiscardableNetwork<Value>(
    _ future: EventLoopFuture<Value>,
    channel: Channel,
    timeout: TimeInterval = operationTimeout
  ) async throws -> Value {
    let waiter = MiohSFTPDiscardableFutureWaiter<Value>()
    let timeoutTask = channel.eventLoop.scheduleTask(in: nioTimeout(timeout)) {
      channel.close(promise: nil)
    }
    future.whenComplete { result in
      timeoutTask.cancel()
      waiter.complete(result)
    }
    return try await withTaskCancellationHandler {
      try Task.checkCancellation()
      return try await waiter.wait()
    } onCancel: {
      waiter.cancel()
    }
  }

  private static func awaitCleanup<Value>(
    _ future: EventLoopFuture<Value>,
    channel: Channel
  ) async throws -> Value {
    let timeoutTask = channel.eventLoop.scheduleTask(in: .seconds(5)) {
      channel.close(promise: nil)
    }
    defer { timeoutTask.cancel() }
    return try await future.get()
  }
}
