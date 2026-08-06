import Combine
import CryptoKit
import Darwin
import Foundation
import Network

public enum MiohClusterWorkerState: Equatable, Sendable {
  case stopped
  case starting
  case ready(port: UInt16)
  case waiting(String)
  case failed(String)
}

public enum MiohClusterWorkerError: LocalizedError, Equatable, Sendable {
  case invalidCapabilities
  case missingModels
  case missingSharedRoot
  case unsupportedJob
  case inputMissing
  case inputNotRegular
  case inputChangedDuringHash
  case inputByteCountMismatch
  case inputSHA256Mismatch
  case outputExists
  case outputReservationConflict
  case outputMissing
  case outputNotRegular
  case outputByteCountMismatch
  case transferExpired
  case uploadFailed
  case uploadRejected

  public var errorDescription: String? {
    switch self {
    case .invalidCapabilities: "Worker能力情報が不正です。"
    case .missingModels: "実行可能と検証済みの復元・検出モデルが必要です。"
    case .missingSharedRoot: "共有フォルダへアクセスできません。"
    case .unsupportedJob: "このiPadで実行できないジョブです。"
    case .inputMissing: "入力ファイルが見つかりません。"
    case .inputNotRegular: "入力は通常ファイルである必要があります。"
    case .inputChangedDuringHash: "入力ファイルが検証中に変更されました。"
    case .inputByteCountMismatch: "入力ファイルのサイズが一致しません。"
    case .inputSHA256Mismatch: "入力ファイルのSHA-256が一致しません。"
    case .outputExists: "出力ファイルがすでに存在します。"
    case .outputReservationConflict: "別のWorkerが同じ出力を使用中です。"
    case .outputMissing: "Workerが出力ファイルを作成しませんでした。"
    case .outputNotRegular: "Worker出力が通常ファイルではありません。"
    case .outputByteCountMismatch: "Worker出力のサイズを検証できませんでした。"
    case .transferExpired: "Coordinator転送チケットの有効期限が切れました。"
    case .uploadFailed: "復元shardをCoordinatorへ送信できませんでした。"
    case .uploadRejected: "Coordinatorが復元shardを受理しませんでした。"
    }
  }

  var failureCode: String {
    switch self {
    case .invalidCapabilities: "invalid_capabilities"
    case .missingModels: "missing_models"
    case .missingSharedRoot: "missing_shared_root"
    case .unsupportedJob: "unsupported_job"
    case .inputMissing: "input_missing"
    case .inputNotRegular: "input_not_regular"
    case .inputChangedDuringHash: "input_changed_during_hash"
    case .inputByteCountMismatch: "input_byte_count_mismatch"
    case .inputSHA256Mismatch: "input_sha256_mismatch"
    case .outputExists: "output_exists"
    case .outputReservationConflict: "output_reservation_conflict"
    case .outputMissing: "output_missing"
    case .outputNotRegular: "output_not_regular"
    case .outputByteCountMismatch: "output_byte_count_mismatch"
    case .transferExpired: "transfer_expired"
    case .uploadFailed: "upload_failed"
    case .uploadRejected: "upload_rejected"
    }
  }
}

public typealias MiohClusterHTTPUploader = @Sendable (
  _ transfer: MiohClusterHTTPTransferDescriptor,
  _ localFile: URL,
  _ byteCount: Int64
) async throws -> Void

enum WorkerHTTPUploader {
  typealias RetrySleeper = (UInt64) async throws -> Void

  static func sessionConfiguration() -> URLSessionConfiguration {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    configuration.urlCache = nil
    configuration.httpCookieStorage = nil
    configuration.httpMaximumConnectionsPerHost = 1
    // Request timeout is the idle timeout. Large shards may legitimately take
    // much longer end-to-end on Wi-Fi, so retain a separate 24-hour resource cap.
    configuration.timeoutIntervalForRequest = 5 * 60
    configuration.timeoutIntervalForResource = 24 * 60 * 60
    return configuration
  }

  static func upload(
    transfer: MiohClusterHTTPTransferDescriptor,
    localFile: URL,
    byteCount: Int64,
    session injectedSession: URLSession? = nil,
    retrySleeper: RetrySleeper = { delay in
      try await Task.sleep(nanoseconds: delay)
    }
  ) async throws {
    guard transfer.hasValidStructure(),
      byteCount > 0, byteCount <= transfer.maximumOutputBytes,
      let outputURL = URL(string: transfer.outputURL)
    else { throw MiohClusterWorkerError.outputByteCountMismatch }
    var request = URLRequest(url: outputURL)
    request.httpMethod = "PUT"
    request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    // Admission and job start validate the descriptor expiry. A long-running
    // shard may legitimately finish after that original timestamp when its
    // lease was renewed by the coordinator, so publication must not reject it
    // using stale descriptor time. The capability endpoint remains the final
    // authority and can reject an invalid or revoked ticket.
    request.timeoutInterval = 5 * 60
    request.setValue("video/mp4", forHTTPHeaderField: "Content-Type")
    request.setValue(String(byteCount), forHTTPHeaderField: "Content-Length")
    let ownsSession = injectedSession == nil
    let session = injectedSession ?? URLSession(configuration: sessionConfiguration())
    defer {
      if ownsSession { session.invalidateAndCancel() }
    }

    // Retrying the exact same ticket and immutable file is intentional. The
    // Coordinator accepts an exact replay after a successful publication whose
    // 2xx response was lost, while rejecting a different body for that ticket.
    // 425 means an exact replay reached the Coordinator while the first PUT's
    // atomic publication cleanup is still completing. It is transient. A 409,
    // by contrast, is a digest/ownership conflict and must never be retried.
    let retryableStatus = Set([408, 425, 429])
    for attempt in 0..<3 {
      try Task.checkCancellation()
      do {
        let (_, response) = try await session.upload(for: request, fromFile: localFile)
        guard let http = response as? HTTPURLResponse else {
          throw MiohClusterWorkerError.uploadFailed
        }
        if (200..<300).contains(http.statusCode) { return }
        let mayRetry = retryableStatus.contains(http.statusCode)
          || (500...599).contains(http.statusCode)
        guard mayRetry, attempt < 2 else {
          throw MiohClusterWorkerError.uploadRejected
        }
      } catch is CancellationError {
        throw CancellationError()
      } catch let error as URLError where error.code == .cancelled {
        throw CancellationError()
      } catch let error as MiohClusterWorkerError {
        throw error
      } catch {
        try Task.checkCancellation()
        guard attempt < 2 else {
          // Capability URLs are bearer credentials. Do not propagate an
          // underlying URLSession error that can include the secret URL.
          throw MiohClusterWorkerError.uploadFailed
        }
      }
      try Task.checkCancellation()
      let delayNanoseconds = UInt64(250_000_000) << UInt64(attempt)
      try await retrySleeper(delayNanoseconds)
    }
    throw MiohClusterWorkerError.uploadFailed
  }
}

private struct WorkerFileSignature: Hashable, Sendable {
  let device: UInt64
  let inode: UInt64
  let byteCount: Int64
  let modificationSeconds: Int64
  let modificationNanoseconds: Int64
  let changeSeconds: Int64
  let changeNanoseconds: Int64
}

private struct WorkerFileIntegrity: Sendable {
  let signature: WorkerFileSignature
  let sha256: String

  static func signature(of url: URL) throws -> WorkerFileSignature {
    var value = stat()
    guard lstat(url.path, &value) == 0 else {
      if errno == ENOENT { throw MiohClusterWorkerError.inputMissing }
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    guard (value.st_mode & S_IFMT) == S_IFREG else {
      throw MiohClusterWorkerError.inputNotRegular
    }
    return makeSignature(value)
  }

  static func inspect(_ url: URL, expected: WorkerFileSignature) throws -> Self {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var opened = stat()
    guard fstat(handle.fileDescriptor, &opened) == 0 else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    guard (opened.st_mode & S_IFMT) == S_IFREG, makeSignature(opened) == expected else {
      throw MiohClusterWorkerError.inputChangedDuringHash
    }
    var hasher = SHA256()
    var bytesRead: Int64 = 0
    while true {
      try Task.checkCancellation()
      guard let chunk = try handle.read(upToCount: 4 * 1_024 * 1_024), !chunk.isEmpty else {
        break
      }
      hasher.update(data: chunk)
      bytesRead += Int64(chunk.count)
    }
    var finished = stat()
    guard fstat(handle.fileDescriptor, &finished) == 0,
      makeSignature(finished) == expected,
      bytesRead == expected.byteCount
    else { throw MiohClusterWorkerError.inputChangedDuringHash }
    return Self(
      signature: expected,
      sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined()
    )
  }

  private static func makeSignature(_ value: stat) -> WorkerFileSignature {
    WorkerFileSignature(
      device: UInt64(value.st_dev),
      inode: UInt64(value.st_ino),
      byteCount: Int64(value.st_size),
      modificationSeconds: Int64(value.st_mtimespec.tv_sec),
      modificationNanoseconds: Int64(value.st_mtimespec.tv_nsec),
      changeSeconds: Int64(value.st_ctimespec.tv_sec),
      changeNanoseconds: Int64(value.st_ctimespec.tv_nsec)
    )
  }
}

/// Multiple shards of one source movie await one hash operation and reuse its
/// result while the inode signature remains unchanged.
private actor WorkerInputIntegrityCache {
  private let maximumEntries = 16
  private var completed: [WorkerFileSignature: WorkerFileIntegrity] = [:]
  private var completionOrder: [WorkerFileSignature] = []
  private var inFlight: [WorkerFileSignature: Task<WorkerFileIntegrity, Error>] = [:]

  func inspect(_ url: URL) async throws -> WorkerFileIntegrity {
    let signature = try WorkerFileIntegrity.signature(of: url)
    if let cached = completed[signature] { return cached }
    if let task = inFlight[signature] { return try await task.value }
    let task = Task.detached(priority: .utility) {
      try WorkerFileIntegrity.inspect(url, expected: signature)
    }
    inFlight[signature] = task
    do {
      let result = try await task.value
      inFlight.removeValue(forKey: signature)
      completed[signature] = result
      completionOrder.removeAll { $0 == signature }
      completionOrder.append(signature)
      while completionOrder.count > maximumEntries {
        completed.removeValue(forKey: completionOrder.removeFirst())
      }
      return result
    } catch {
      inFlight.removeValue(forKey: signature)
      throw error
    }
  }
}

/// O_EXCL on a sibling lock file prevents two devices sharing one SMB root
/// from publishing the same final output path concurrently.
private struct WorkerOutputReservation {
  let lockURL: URL

  static func acquire(outputURL: URL, attemptID: UUID) throws -> Self {
    if FileManager.default.fileExists(atPath: outputURL.path) {
      throw MiohClusterWorkerError.outputExists
    }
    let lockURL = outputURL.appendingPathExtension("mioh-cluster.lock")
    let descriptor = open(
      lockURL.path,
      O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC,
      S_IRUSR | S_IWUSR
    )
    guard descriptor >= 0 else {
      if errno == EEXIST { throw MiohClusterWorkerError.outputReservationConflict }
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    let payload = Data((attemptID.uuidString.lowercased() + "\n").utf8)
    payload.withUnsafeBytes { rawBuffer in
      if let baseAddress = rawBuffer.baseAddress {
        _ = Darwin.write(descriptor, baseAddress, rawBuffer.count)
      }
    }
    Darwin.close(descriptor)
    return Self(lockURL: lockURL)
  }

  func release() { try? FileManager.default.removeItem(at: lockURL) }
}

@MainActor
public final class MiohClusterWorkerLedger: ObservableObject {
  @Published public private(set) var attempts: [MiohClusterAttemptRecord] = []

  public var inFlightExecutionCount: Int { tasks.count }

  private let maximumLeaseSeconds: TimeInterval
  private let maximumRetainedAttempts: Int
  private let inputIntegrityCache = WorkerInputIntegrityCache()
  private let httpUploader: MiohClusterHTTPUploader
  private var requests: [UUID: MiohClusterJobRequest] = [:]
  private var tasks: [UUID: Task<Void, Never>] = [:]
  private var leaseWatchdogs: [UUID: Task<Void, Never>] = [:]

  public init(
    maximumLeaseSeconds: TimeInterval = 15 * 60,
    maximumRetainedAttempts: Int = 256
  ) {
    self.maximumLeaseSeconds = max(30, maximumLeaseSeconds)
    self.maximumRetainedAttempts = max(32, maximumRetainedAttempts)
    self.httpUploader = {
      try await WorkerHTTPUploader.upload(
        transfer: $0,
        localFile: $1,
        byteCount: $2
      )
    }
  }

  init(
    maximumLeaseSeconds: TimeInterval = 15 * 60,
    maximumRetainedAttempts: Int = 256,
    httpUploader: @escaping MiohClusterHTTPUploader
  ) {
    self.maximumLeaseSeconds = max(30, maximumLeaseSeconds)
    self.maximumRetainedAttempts = max(32, maximumRetainedAttempts)
    self.httpUploader = httpUploader
  }

  func submit(
    _ request: MiohClusterJobRequest,
    sharedRoot: URL?,
    capabilities: MiohClusterCapabilities,
    launcher: @escaping MiohClusterJobLauncher,
    now: Date = Date()
  ) -> MiohClusterJobAdmission {
    let admission = admit(request, capabilities: capabilities, now: now)
    guard admission.disposition == .accepted else { return admission }
    tasks[request.attemptID] = Task { @MainActor [weak self] in
      guard let self else { return }
      await self.runAccepted(request, sharedRoot: sharedRoot, launcher: launcher)
    }
    return admission
  }

  private func admit(
    _ request: MiohClusterJobRequest,
    capabilities: MiohClusterCapabilities,
    now: Date
  ) -> MiohClusterJobAdmission {
    if let existing = record(request.attemptID) {
      let exact = requests[request.attemptID] == request
      return MiohClusterJobAdmission(
        disposition: exact ? .duplicate : .conflict,
        record: existing,
        reason: exact ? nil : "attempt_id_reused"
      )
    }
    let requestTransferMode: MiohClusterTransferMode = request.httpTransfer == nil
      ? .sharedRootV1 : .coordinatorHTTPV1
    let supportedTransferModes = capabilities.supportedTransferModes
      ?? [capabilities.transferMode]
    guard request.protocolVersion == MiohClusterJobRequest.protocolVersion,
      supportedTransferModes.contains(requestTransferMode),
      (requestTransferMode != .sharedRootV1
        || request.sharedRootIdentifier == capabilities.sharedRootIdentifier),
      (requestTransferMode != .coordinatorHTTPV1
        || request.httpTransfer?.isValid(now: now) == true),
      (requestTransferMode != .coordinatorHTTPV1
        || (request.httpTransfer?.expiresAt ?? .distantPast) >= request.leaseExpiresAt),
      request.inputByteCount > 0,
      Self.isSHA256(request.inputSHA256),
      request.inputRelativePath != request.outputRelativePath,
      request.mediaRange.isValid,
      request.options.isValid,
      request.leaseExpiresAt > now,
      request.leaseExpiresAt.timeIntervalSince(now) <= maximumLeaseSeconds,
      capabilities.restorationModelIdentifiers.contains(
        request.options.restorationModelIdentifier
      ),
      capabilities.detectorModelIdentifiers.contains(
        request.options.detectorModelIdentifier
      ),
      (capabilities.maximumRestorationClipLength.map {
        request.options.restorationClipLength <= $0
      } ?? true),
      capabilities.supportsROIEnhancer != false || (
        request.options.roiEnhancerModelIdentifier == nil
          && request.options.roiEnhancerAssetSHA256 == nil
          && request.options.roiEnhancerStrength == 0
      ),
      capabilities.supportsRestorationEffects != false || (
        request.options.sharpenStrength == 0
          && request.options.detailBoost == 0
          && request.options.textureMix == 0
          && request.options.smoothStrength == 0
          && request.options.effectUpscale == 1
      ),
      (capabilities.supportedInputExtensions.map {
        $0.contains(Self.pathExtension(request.inputRelativePath.rawValue))
      } ?? true),
      Self.identityMatches(
        capabilities.restorationAssetSHA256ByIdentifier,
        identifier: request.options.restorationModelIdentifier,
        digest: request.options.restorationAssetSHA256
      ),
      Self.identityMatches(
        capabilities.detectorAssetSHA256ByIdentifier,
        identifier: request.options.detectorModelIdentifier,
        digest: request.options.detectorAssetSHA256
      )
    else {
      return MiohClusterJobAdmission(
        disposition: .rejected,
        record: nil,
        reason: "invalid_or_unsupported_job_contract"
      )
    }
    if let active = attempts.first(where: {
      $0.jobID == request.jobID && !$0.state.isTerminal
    }) {
      return MiohClusterJobAdmission(
        disposition: .conflict,
        record: active,
        reason: "job_attempt_already_active"
      )
    }
    if let conflict = attempts.first(where: {
      !$0.state.isTerminal && $0.outputRelativePath == request.outputRelativePath
    }) {
      return MiohClusterJobAdmission(
        disposition: .conflict,
        record: conflict,
        reason: "output_path_already_active"
      )
    }
    let occupied = max(tasks.count, attempts.filter { !$0.state.isTerminal }.count)
    guard occupied < capabilities.maximumConcurrentJobs else {
      return MiohClusterJobAdmission(
        disposition: .rejected,
        record: nil,
        reason: "worker_capacity_reached"
      )
    }
    pruneRetainedAttempts()
    guard attempts.count < maximumRetainedAttempts else {
      return MiohClusterJobAdmission(
        disposition: .rejected,
        record: nil,
        reason: "attempt_ledger_full"
      )
    }
    let record = MiohClusterAttemptRecord(
      jobID: request.jobID,
      attemptID: request.attemptID,
      leaseID: request.leaseID,
      coordinatorNodeID: request.coordinatorNodeID,
      inputRelativePath: request.inputRelativePath,
      outputRelativePath: request.outputRelativePath,
      state: .accepted,
      leaseExpiresAt: request.leaseExpiresAt,
      updatedAt: now
    )
    requests[request.attemptID] = request
    attempts.append(record)
    scheduleLeaseWatchdog(attemptID: request.attemptID)
    return MiohClusterJobAdmission(disposition: .accepted, record: record, reason: nil)
  }

  public func record(_ attemptID: UUID) -> MiohClusterAttemptRecord? {
    attempts.first { $0.attemptID == attemptID }
  }

  func renew(_ request: MiohClusterRenewLeaseRequest, now: Date = Date()) -> Bool {
    guard let recordIndex = index(request.attemptID),
      attempts[recordIndex].leaseID == request.leaseID,
      !attempts[recordIndex].state.isTerminal,
      request.newExpiration > now,
      request.newExpiration > attempts[recordIndex].leaseExpiresAt,
      request.newExpiration.timeIntervalSince(now) <= maximumLeaseSeconds
    else { return false }
    attempts[recordIndex].leaseExpiresAt = request.newExpiration
    attempts[recordIndex].updatedAt = now
    return true
  }

  func cancel(_ request: MiohClusterCancelRequest, now: Date = Date()) -> Bool {
    guard let recordIndex = index(request.attemptID),
      attempts[recordIndex].leaseID == request.leaseID,
      !attempts[recordIndex].state.isTerminal
    else { return false }
    attempts[recordIndex].state = .cancelled
    attempts[recordIndex].updatedAt = now
    tasks[request.attemptID]?.cancel()
    stopLeaseWatchdog(attemptID: request.attemptID)
    return true
  }

  public func cancelAll() {
    for recordIndex in attempts.indices where !attempts[recordIndex].state.isTerminal {
      attempts[recordIndex].state = .cancelled
      attempts[recordIndex].updatedAt = Date()
    }
    tasks.values.forEach { $0.cancel() }
    leaseWatchdogs.values.forEach { $0.cancel() }
    leaseWatchdogs.removeAll()
  }

  public func cancelAllAndWait() async {
    cancelAll()
    let inFlight = Array(tasks.values)
    for task in inFlight {
      task.cancel()
      await task.value
    }
  }

  func expireLeases(now: Date = Date()) {
    for recordIndex in attempts.indices
    where !attempts[recordIndex].state.isTerminal
      && attempts[recordIndex].leaseExpiresAt <= now
    {
      attempts[recordIndex].state = .expired
      attempts[recordIndex].updatedAt = now
      let attemptID = attempts[recordIndex].attemptID
      tasks[attemptID]?.cancel()
      stopLeaseWatchdog(attemptID: attemptID)
    }
  }

  private func runAccepted(
    _ request: MiohClusterJobRequest,
    sharedRoot: URL?,
    launcher: @escaping MiohClusterJobLauncher
  ) async {
    guard let acceptedIndex = index(request.attemptID),
      attempts[acceptedIndex].state == .accepted
    else { return }
    defer {
      tasks.removeValue(forKey: request.attemptID)
      stopLeaseWatchdog(attemptID: request.attemptID)
    }
    do {
      if let transfer = request.httpTransfer {
        try await runHTTPAccepted(
          request,
          transfer: transfer,
          launcher: launcher,
          acceptedIndex: acceptedIndex
        )
      } else {
        guard let sharedRoot else { throw MiohClusterWorkerError.missingSharedRoot }
        try await runSharedRootAccepted(
          request,
          sharedRoot: sharedRoot,
          launcher: launcher,
          acceptedIndex: acceptedIndex
        )
      }
    } catch {
      if let failedIndex = index(request.attemptID),
        !attempts[failedIndex].state.isTerminal
      {
        attempts[failedIndex].state = .failed
        attempts[failedIndex].failureCode =
          (error as? MiohClusterWorkerError)?.failureCode ?? "launcher_failed"
        attempts[failedIndex].updatedAt = Date()
      }
    }
  }

  private func runHTTPAccepted(
    _ request: MiohClusterJobRequest,
    transfer: MiohClusterHTTPTransferDescriptor,
    launcher: @escaping MiohClusterJobLauncher,
    acceptedIndex: Int
  ) async throws {
    guard transfer.hasValidStructure(), let inputURL = URL(string: transfer.inputURL) else {
      throw MiohClusterWorkerError.unsupportedJob
    }
    let workRoot = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-ipad-worker",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: workRoot,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    let candidate = workRoot.appendingPathComponent(
      "\(request.attemptID.uuidString.lowercased()).mp4"
    )
    try? FileManager.default.removeItem(at: candidate)
    defer { try? FileManager.default.removeItem(at: candidate) }
    try Task.checkCancellation()
    attempts[acceptedIndex].state = .running
    attempts[acceptedIndex].updatedAt = Date()
    let metrics = try await launcher(request, inputURL, candidate)
    try Task.checkCancellation()
    let actualBytes = try validateOutput(candidate, metrics: metrics)
    guard actualBytes <= transfer.maximumOutputBytes else {
      throw MiohClusterWorkerError.outputByteCountMismatch
    }
    try await httpUploader(transfer, candidate, actualBytes)
    try Task.checkCancellation()
    try complete(request.attemptID, metrics: metrics)
  }

  private func runSharedRootAccepted(
    _ request: MiohClusterJobRequest,
    sharedRoot: URL,
    launcher: @escaping MiohClusterJobLauncher,
    acceptedIndex: Int
  ) async throws {
    var reservation: WorkerOutputReservation?
    var stagingURL: URL?
    defer {
      reservation?.release()
      if let stagingURL { try? FileManager.default.removeItem(at: stagingURL) }
    }
    try Task.checkCancellation()
    let inputURL = try request.inputRelativePath.resolve(beneath: sharedRoot)
    var outputURL = try request.outputRelativePath.resolve(beneath: sharedRoot)
    let integrity = try await inputIntegrityCache.inspect(inputURL)
    try Task.checkCancellation()
    guard integrity.signature.byteCount == request.inputByteCount else {
      throw MiohClusterWorkerError.inputByteCountMismatch
    }
    guard integrity.sha256 == request.inputSHA256 else {
      throw MiohClusterWorkerError.inputSHA256Mismatch
    }
    let parent = outputURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(
      at: parent,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    outputURL = try request.outputRelativePath.resolve(beneath: sharedRoot)
    guard !FileManager.default.fileExists(atPath: outputURL.path) else {
      throw MiohClusterWorkerError.outputExists
    }
    reservation = try WorkerOutputReservation.acquire(
      outputURL: outputURL,
      attemptID: request.attemptID
    )
    let candidate = parent.appendingPathComponent(
      ".mioh-cluster-\(request.attemptID.uuidString.lowercased()).part"
    )
    guard !FileManager.default.fileExists(atPath: candidate.path) else {
      throw MiohClusterWorkerError.outputReservationConflict
    }
    stagingURL = candidate
    attempts[acceptedIndex].state = .running
    attempts[acceptedIndex].updatedAt = Date()
    let metrics = try await launcher(request, inputURL, candidate)
    try Task.checkCancellation()
    _ = try validateOutput(candidate, metrics: metrics)
    guard !FileManager.default.fileExists(atPath: outputURL.path) else {
      throw MiohClusterWorkerError.outputExists
    }
    try FileManager.default.moveItem(at: candidate, to: outputURL)
    stagingURL = nil
    try complete(request.attemptID, metrics: metrics)
  }

  private func validateOutput(
    _ candidate: URL,
    metrics: MiohClusterJobMetrics
  ) throws -> Int64 {
    var outputStat = stat()
    guard lstat(candidate.path, &outputStat) == 0 else {
      throw MiohClusterWorkerError.outputMissing
    }
    guard (outputStat.st_mode & S_IFMT) == S_IFREG else {
      throw MiohClusterWorkerError.outputNotRegular
    }
    let actualBytes = Int64(outputStat.st_size)
    try validateMetrics(metrics)
    guard metrics.outputByteCount == actualBytes else {
      throw MiohClusterWorkerError.outputByteCountMismatch
    }
    return actualBytes
  }

  private func validateMetrics(_ metrics: MiohClusterJobMetrics) throws {
    guard metrics.outputByteCount > 0,
      metrics.processedFrames > 0,
      metrics.wallSeconds.isFinite,
      metrics.wallSeconds >= 0
    else { throw MiohClusterWorkerError.outputByteCountMismatch }
  }

  private func complete(
    _ attemptID: UUID,
    metrics: MiohClusterJobMetrics
  ) throws {
    guard let completedIndex = index(attemptID),
      !attempts[completedIndex].state.isTerminal
    else { throw CancellationError() }
    attempts[completedIndex].state = .completed
    attempts[completedIndex].metrics = metrics
    attempts[completedIndex].updatedAt = Date()
  }

  private func index(_ attemptID: UUID) -> Int? {
    attempts.firstIndex { $0.attemptID == attemptID }
  }

  private func scheduleLeaseWatchdog(attemptID: UUID) {
    stopLeaseWatchdog(attemptID: attemptID)
    leaseWatchdogs[attemptID] = Task { @MainActor [weak self] in
      guard let self else { return }
      defer { self.leaseWatchdogs.removeValue(forKey: attemptID) }
      while let record = self.record(attemptID), !record.state.isTerminal {
        let seconds = max(0, record.leaseExpiresAt.timeIntervalSinceNow)
        if seconds > 0 {
          do {
            try await Task.sleep(
              nanoseconds: UInt64(max(1_000_000, min(seconds, 86_400) * 1_000_000_000))
            )
          } catch { return }
        }
        guard !Task.isCancelled else { return }
        self.expireLeases()
      }
    }
  }

  private func stopLeaseWatchdog(attemptID: UUID) {
    leaseWatchdogs.removeValue(forKey: attemptID)?.cancel()
  }

  private func pruneRetainedAttempts() {
    guard attempts.count >= maximumRetainedAttempts else { return }
    let removable = attempts
      .filter { $0.state.isTerminal && tasks[$0.attemptID] == nil }
      .sorted { $0.updatedAt < $1.updatedAt }
    let count = min(removable.count, attempts.count - maximumRetainedAttempts + 1)
    let removed = Set(removable.prefix(count).map(\.attemptID))
    guard !removed.isEmpty else { return }
    attempts.removeAll { removed.contains($0.attemptID) }
    for attemptID in removed {
      requests.removeValue(forKey: attemptID)
      stopLeaseWatchdog(attemptID: attemptID)
    }
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy {
      (48...57).contains($0) || (97...102).contains($0)
    }
  }

  private static func pathExtension(_ path: String) -> String {
    URL(fileURLWithPath: path).pathExtension.lowercased()
  }

  private static func identityMatches(
    _ identities: [String: String]?,
    identifier: String,
    digest: String
  ) -> Bool {
    guard let identities else { return true }
    return identities[identifier] == digest
  }
}

@MainActor
public final class MiohClusterWorkerService: ObservableObject {
  @Published public private(set) var state: MiohClusterWorkerState = .stopped
  @Published public private(set) var capabilities: MiohClusterCapabilities?
  public let ledger: MiohClusterWorkerLedger

  private let queue = DispatchQueue(label: "mioh.remote.cluster.ios-worker")
  private var listener: NWListener?
  private var transport: WorkerTransport?
  private var generation = UUID()

  public init(ledger: MiohClusterWorkerLedger? = nil) {
    self.ledger = ledger ?? MiohClusterWorkerLedger()
  }

  public func start(
    sharedRoot: URL? = nil,
    capabilities: MiohClusterCapabilities,
    launcher: @escaping MiohClusterJobLauncher
  ) throws {
    stop()
    guard capabilities.protocolVersion == MiohClusterCapabilities.protocolVersion,
      capabilities.role == .worker,
      (capabilities.supportedTransferModes ?? [capabilities.transferMode])
        .contains(capabilities.transferMode),
      capabilities.maximumConcurrentJobs == 1,
      (capabilities.maximumRestorationClipLength.map { $0 > 0 } ?? true),
      (capabilities.supportedInputExtensions.map { !$0.isEmpty } ?? true),
      Self.validIdentityMap(
        capabilities.restorationAssetSHA256ByIdentifier,
        advertised: capabilities.restorationModelIdentifiers
      ),
      Self.validIdentityMap(
        capabilities.detectorAssetSHA256ByIdentifier,
        advertised: capabilities.detectorModelIdentifiers
      )
    else { throw MiohClusterWorkerError.invalidCapabilities }
    // A worker with no complete model path must not look usable to a Mac.
    guard !capabilities.restorationModelIdentifiers.isEmpty,
      !capabilities.detectorModelIdentifiers.isEmpty
    else { throw MiohClusterWorkerError.missingModels }
    let supportedModes = capabilities.supportedTransferModes
      ?? [capabilities.transferMode]
    var standardizedRoot: URL?
    if supportedModes.contains(.sharedRootV1) {
      guard let sharedRoot,
        !capabilities.sharedRootIdentifier.isEmpty
      else { throw MiohClusterWorkerError.missingSharedRoot }
      var isDirectory: ObjCBool = false
      guard FileManager.default.fileExists(
        atPath: sharedRoot.path,
        isDirectory: &isDirectory
      ), isDirectory.boolValue
      else { throw MiohClusterWorkerError.missingSharedRoot }
      standardizedRoot = sharedRoot.standardizedFileURL.resolvingSymlinksInPath()
    }
    let listener = try NWListener(using: .tcp, on: .any)
    let transport = WorkerTransport(
      capabilities: capabilities,
      sharedRoot: standardizedRoot,
      ledger: ledger,
      launcher: launcher
    )
    let txt: [String: String] = [
      "v": String(capabilities.protocolVersion),
      "node": capabilities.nodeID.uuidString.lowercased(),
      "name": capabilities.displayName,
      "role": capabilities.role.rawValue,
      "transfer": capabilities.transferMode.rawValue,
      "transfers": supportedModes.map(\.rawValue).sorted().joined(separator: ","),
      // Keep the key present even for HTTP-only Workers. The empty value is
      // the canonical wire representation of "no shared-root fallback".
      "root": capabilities.sharedRootIdentifier,
      "jobs": String(capabilities.maximumConcurrentJobs),
    ]
    listener.service = NWListener.Service(
      name: capabilities.displayName,
      type: "_mioh-worker._tcp",
      domain: "local.",
      txtRecord: NWTXTRecord(txt)
    )
    generation = UUID()
    let currentGeneration = generation
    self.listener = listener
    self.transport = transport
    self.capabilities = capabilities
    state = .starting
    listener.newConnectionHandler = { [weak transport] connection in
      Task { @MainActor [weak transport] in
        guard let transport else { connection.cancel(); return }
        transport.accept(connection)
      }
    }
    listener.stateUpdateHandler = { [weak self, weak listener] newState in
      Task { @MainActor [weak self, weak listener] in
        guard let self, self.generation == currentGeneration,
          self.listener === listener
        else { return }
        switch newState {
        case .setup: self.state = .starting
        case .ready: self.state = .ready(port: listener?.port?.rawValue ?? 0)
        case .waiting(let error): self.state = .waiting(error.localizedDescription)
        case .failed(let error): self.state = .failed(error.localizedDescription)
        case .cancelled: self.state = .stopped
        @unknown default: self.state = .failed("unknown_listener_state")
        }
      }
    }
    listener.start(queue: queue)
  }

  public func stop() {
    generation = UUID()
    listener?.cancel()
    listener = nil
    transport?.stop()
    transport = nil
    ledger.cancelAll()
    capabilities = nil
    state = .stopped
  }

  public func stopAndWait() async {
    stop()
    await ledger.cancelAllAndWait()
  }

  private static func validIdentityMap(
    _ map: [String: String]?,
    advertised: [String]
  ) -> Bool {
    guard let map else { return true }
    return advertised.allSatisfy { identifier in
      guard let digest = map[identifier] else { return false }
      return digest.utf8.count == 64 && digest.utf8.allSatisfy {
        (48...57).contains($0) || (97...102).contains($0)
      }
    }
  }
}

private enum WorkerFraming {
  static let maximumJSONBytes = 64 * 1024

  static func encoded<T: Encodable>(_ value: T) throws -> Data {
    let payload = try JSONEncoder().encode(value)
    guard payload.count <= maximumJSONBytes else {
      throw URLError(.dataLengthExceedsMaximum)
    }
    var length = UInt32(payload.count).bigEndian
    var result = Data(bytes: &length, count: 4)
    result.append(payload)
    return result
  }

  static func receive(
    _ connection: NWConnection,
    completion: @escaping @Sendable (Result<Data, Error>) -> Void
  ) {
    receiveExactly(connection, count: 4) { header in
      do {
        let bytes = try header.get()
        let length = bytes.withUnsafeBytes {
          UInt32(bigEndian: $0.loadUnaligned(as: UInt32.self))
        }
        guard length > 0, length <= maximumJSONBytes else {
          throw URLError(.dataLengthExceedsMaximum)
        }
        receiveExactly(connection, count: Int(length), completion: completion)
      } catch { completion(.failure(error)) }
    }
  }

  private static func receiveExactly(
    _ connection: NWConnection,
    count: Int,
    accumulated: Data = Data(),
    completion: @escaping @Sendable (Result<Data, Error>) -> Void
  ) {
    guard accumulated.count < count else { completion(.success(accumulated)); return }
    connection.receive(minimumIncompleteLength: 1, maximumLength: count - accumulated.count) {
      data, _, complete, error in
      if let error { completion(.failure(error)); return }
      var next = accumulated
      if let data { next.append(data) }
      if next.count == count {
        completion(.success(next))
      } else if complete || data?.isEmpty != false {
        completion(.failure(URLError(.cannotParseResponse)))
      } else {
        receiveExactly(connection, count: count, accumulated: next, completion: completion)
      }
    }
  }
}

private final class WorkerOnceFlag: @unchecked Sendable {
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

@MainActor
private final class WorkerTransport {
  private let capabilities: MiohClusterCapabilities
  private let sharedRoot: URL?
  private let ledger: MiohClusterWorkerLedger
  private let launcher: MiohClusterJobLauncher
  private let queue = DispatchQueue(label: "mioh.remote.cluster.ios-worker-rpc")
  private var connections: [UUID: NWConnection] = [:]

  init(
    capabilities: MiohClusterCapabilities,
    sharedRoot: URL?,
    ledger: MiohClusterWorkerLedger,
    launcher: @escaping MiohClusterJobLauncher
  ) {
    self.capabilities = capabilities
    self.sharedRoot = sharedRoot
    self.ledger = ledger
    self.launcher = launcher
  }

  func accept(_ connection: NWConnection) {
    guard connections.count < 16 else { connection.cancel(); return }
    let id = UUID()
    let receiveOnce = WorkerOnceFlag()
    connections[id] = connection
    connection.stateUpdateHandler = { [weak self, weak connection] state in
      guard let connection else { return }
      switch state {
      case .ready where receiveOnce.claim():
        guard let owner = self else { connection.cancel(); return }
        WorkerFraming.receive(connection) { [connection] result in
          Task { @MainActor in
            await owner.handle(result, id: id, connection: connection)
          }
        }
      case .failed, .cancelled:
        Task { @MainActor [weak self] in self?.finish(id, connection) }
      default:
        break
      }
    }
    connection.start(queue: queue)
    queue.asyncAfter(deadline: .now() + 15) { [weak self, weak connection] in
      guard let connection else { return }
      Task { @MainActor [weak self] in
        guard let self, self.connections[id] != nil else { return }
        self.finish(id, connection)
      }
    }
  }

  func stop() {
    for connection in connections.values { connection.cancel() }
    connections.removeAll()
  }

  private func handle(
    _ result: Result<Data, Error>,
    id: UUID,
    connection: NWConnection
  ) async {
    let request: MiohClusterRPCRequest
    do {
      request = try JSONDecoder().decode(MiohClusterRPCRequest.self, from: result.get())
    } catch {
      reply(.error(requestID: UUID(), "malformed_request"), id: id, connection: connection)
      return
    }
    guard request.protocolVersion == MiohClusterCapabilities.protocolVersion else {
      reply(.error(requestID: request.requestID, "unsupported_protocol"), id: id, connection: connection)
      return
    }
    // Cluster traffic is intentionally unauthenticated. The service is
    // advertised and accepted only on the trusted local LAN.
    ledger.expireLeases()
    let response: MiohClusterRPCResponse
    switch request.action {
    case .capabilities:
      response = MiohClusterRPCResponse(
        requestID: request.requestID,
        ok: true,
        capabilities: capabilities,
        admission: nil,
        nodeStatus: nil,
        attempt: nil,
        errorCode: nil
      )
    case .submit(let job):
      guard job.coordinatorNodeID == request.coordinatorNodeID else {
        response = .error(requestID: request.requestID, "coordinator_mismatch")
        break
      }
      let admission = ledger.submit(
        job,
        sharedRoot: sharedRoot,
        capabilities: capabilities,
        launcher: launcher
      )
      response = MiohClusterRPCResponse(
        requestID: request.requestID,
        ok: admission.disposition == .accepted || admission.disposition == .duplicate,
        capabilities: nil,
        admission: admission,
        nodeStatus: nil,
        attempt: admission.record,
        errorCode: admission.reason
      )
    case .status(let query):
      if let attemptID = query.attemptID {
        let found = ledger.record(attemptID)
        let visible = found?.coordinatorNodeID == request.coordinatorNodeID ? found : nil
        response = MiohClusterRPCResponse(
          requestID: request.requestID,
          ok: visible != nil,
          capabilities: nil,
          admission: nil,
          nodeStatus: nil,
          attempt: visible,
          errorCode: visible == nil ? "attempt_not_found" : nil
        )
      } else {
        let active = ledger.attempts.filter { !$0.state.isTerminal }
        let occupied = max(ledger.inFlightExecutionCount, active.count)
        response = MiohClusterRPCResponse(
          requestID: request.requestID,
          ok: true,
          capabilities: nil,
          admission: nil,
          nodeStatus: MiohClusterNodeStatus(
            nodeID: capabilities.nodeID,
            observedAt: Date(),
            capabilities: capabilities,
            activeAttempts: active.filter {
              $0.coordinatorNodeID == request.coordinatorNodeID
            },
            acceptingJobs: occupied < capabilities.maximumConcurrentJobs
          ),
          attempt: nil,
          errorCode: nil
        )
      }
    case .renewLease(let renewal):
      let found = ledger.record(renewal.attemptID)
      let authorized = found?.coordinatorNodeID == request.coordinatorNodeID
        && found?.leaseID == renewal.leaseID
      let terminal = authorized && found?.state.isTerminal == true
      let renewed = authorized && !terminal && ledger.renew(renewal)
      let accepted = renewed || terminal
      response = MiohClusterRPCResponse(
        requestID: request.requestID,
        ok: accepted,
        capabilities: nil,
        admission: nil,
        nodeStatus: nil,
        attempt: accepted ? ledger.record(renewal.attemptID) : nil,
        errorCode: accepted ? nil : "lease_renewal_rejected"
      )
    case .cancel(let cancellation):
      let found = ledger.record(cancellation.attemptID)
      let cancelled = found?.coordinatorNodeID == request.coordinatorNodeID
        && ledger.cancel(cancellation)
      response = MiohClusterRPCResponse(
        requestID: request.requestID,
        ok: cancelled,
        capabilities: nil,
        admission: nil,
        nodeStatus: nil,
        attempt: cancelled ? ledger.record(cancellation.attemptID) : nil,
        errorCode: cancelled ? nil : "cancel_rejected"
      )
    }
    reply(response, id: id, connection: connection)
  }

  private func reply(
    _ response: MiohClusterRPCResponse,
    id: UUID,
    connection: NWConnection
  ) {
    do {
      connection.send(
        content: try WorkerFraming.encoded(response),
        completion: .contentProcessed { [weak self, weak connection] _ in
          guard let connection else { return }
          Task { @MainActor [weak self] in self?.finish(id, connection) }
        }
      )
    } catch { finish(id, connection) }
  }

  private func finish(_ id: UUID, _ connection: NWConnection) {
    connections.removeValue(forKey: id)
    connection.stateUpdateHandler = nil
    connection.cancel()
  }

}
