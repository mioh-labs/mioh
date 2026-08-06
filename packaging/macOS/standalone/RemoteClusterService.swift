import Combine
import CryptoKit
import Darwin
import Foundation
import Network

// MARK: - Cluster wire contracts

/// The optional HTTP mode extends the v1 envelope without changing its wire
/// version. Older Workers decode the absent optional fields and continue to
/// use only `shared-root-v1`.
enum RemoteClusterTransferMode: String, Codable, Hashable, Sendable {
  case sharedRootV1 = "shared-root-v1"
  case coordinatorHTTPV1 = "coordinator-http-v1"
}

/// A short-lived capability URL pair issued for one exact attempt. The random
/// ticket is carried in the unlogged URL path rather than a private
/// AVFoundation HTTP-header option, so iOS/macOS can use AVURLAsset's public
/// HTTP Range implementation directly.
struct RemoteClusterHTTPTransferDescriptor: Codable, Hashable, Sendable {
  let inputURL: String
  let outputURL: String
  let expiresAt: Date
  let maximumOutputBytes: Int64

  var isValid: Bool {
    guard expiresAt.timeIntervalSince1970.isFinite,
      expiresAt.timeIntervalSince1970 > 0, maximumOutputBytes > 0,
      let input = URL(string: inputURL), let output = URL(string: outputURL),
      let inputComponents = URLComponents(url: input, resolvingAgainstBaseURL: false),
      let outputComponents = URLComponents(url: output, resolvingAgainstBaseURL: false),
      input.scheme?.lowercased() == "http",
      output.scheme?.lowercased() == "http",
      input.host?.lowercased() == output.host?.lowercased(),
      (input.port ?? 80) == (output.port ?? 80),
      input.user == nil, input.password == nil,
      output.user == nil, output.password == nil,
      inputComponents.query == nil, inputComponents.fragment == nil,
      outputComponents.query == nil, outputComponents.fragment == nil,
      !inputComponents.percentEncodedPath.contains("%"),
      !outputComponents.percentEncodedPath.contains("%")
    else { return false }
    let inputPath = input.path.split(separator: "/", omittingEmptySubsequences: true)
    let outputPath = output.path.split(separator: "/", omittingEmptySubsequences: true)
    guard inputPath.count == 4, outputPath.count == 4,
      inputPath[0] == "mioh-cluster", outputPath[0] == "mioh-cluster",
      inputPath[1] == "v1", outputPath[1] == "v1",
      inputPath[2] == outputPath[2], inputPath[2].utf8.count == 43,
      inputPath[2].utf8.allSatisfy({
        (48...57).contains($0) || (65...90).contains($0)
          || (97...122).contains($0) || $0 == 45 || $0 == 95
      }),
      Set(["input.mp4", "input.mov", "input.m4v"]).contains(String(inputPath[3])),
      outputPath[3] == "output.mp4"
    else { return false }
    return true
  }
}

enum RemoteClusterRole: String, Codable, Hashable, Sendable {
  case coordinator
  case worker
}

struct RemoteClusterCapabilities: Codable, Hashable, Sendable {
  static let protocolVersion = 2

  let protocolVersion: Int
  let nodeID: UUID
  let displayName: String
  let role: RemoteClusterRole
  let transferMode: RemoteClusterTransferMode
  let sharedRootIdentifier: String
  let architecture: String
  let operatingSystem: String
  let maximumConcurrentJobs: Int
  let restorationModelIdentifiers: [String]
  let detectorModelIdentifiers: [String]
  /// Optional v1 extensions. Missing fields preserve compatibility with the
  /// first Mac Worker build: nil means unrestricted/supported/unknown.
  let maximumRestorationClipLength: Int?
  let supportsROIEnhancer: Bool?
  let supportsRestorationEffects: Bool?
  let supportedInputExtensions: [String]?
  let restorationAssetSHA256ByIdentifier: [String: String]?
  let detectorAssetSHA256ByIdentifier: [String: String]?
  /// Nil means the original Worker which supports only `transferMode`.
  let supportedTransferModes: [RemoteClusterTransferMode]?

  init(
    protocolVersion: Int = Self.protocolVersion,
    nodeID: UUID,
    displayName: String,
    role: RemoteClusterRole,
    sharedRootIdentifier: String,
    architecture: String,
    operatingSystem: String,
    maximumConcurrentJobs: Int,
    restorationModelIdentifiers: [String],
    detectorModelIdentifiers: [String],
    maximumRestorationClipLength: Int? = nil,
    supportsROIEnhancer: Bool? = nil,
    supportsRestorationEffects: Bool? = nil,
    supportedInputExtensions: [String]? = nil,
    restorationAssetSHA256ByIdentifier: [String: String]? = nil,
    detectorAssetSHA256ByIdentifier: [String: String]? = nil,
    transferMode: RemoteClusterTransferMode = .sharedRootV1,
    supportedTransferModes: [RemoteClusterTransferMode]? = nil
  ) {
    self.protocolVersion = protocolVersion
    self.nodeID = nodeID
    self.displayName = displayName
    self.role = role
    self.transferMode = transferMode
    self.sharedRootIdentifier = sharedRootIdentifier
    self.architecture = architecture
    self.operatingSystem = operatingSystem
    self.maximumConcurrentJobs = max(1, maximumConcurrentJobs)
    self.restorationModelIdentifiers = restorationModelIdentifiers.sorted()
    self.detectorModelIdentifiers = detectorModelIdentifiers.sorted()
    self.maximumRestorationClipLength = maximumRestorationClipLength.map { max(1, $0) }
    self.supportsROIEnhancer = supportsROIEnhancer
    self.supportsRestorationEffects = supportsRestorationEffects
    self.supportedInputExtensions = supportedInputExtensions.map {
      Array(Set($0.map { $0.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".")) }))
        .filter { !$0.isEmpty }
        .sorted()
    }
    self.restorationAssetSHA256ByIdentifier = restorationAssetSHA256ByIdentifier
    self.detectorAssetSHA256ByIdentifier = detectorAssetSHA256ByIdentifier
    self.supportedTransferModes = supportedTransferModes.map {
      Array(Set($0)).sorted { $0.rawValue < $1.rawValue }
    }
  }

  var effectiveTransferModes: Set<RemoteClusterTransferMode> {
    Set(supportedTransferModes ?? [transferMode])
  }
}

struct RemoteClusterMediaRange: Codable, Hashable, Sendable {
  /// Decode range includes the temporal halo; core range is the only part the
  /// worker may emit. Nanoseconds keep adjacent shards on one exact timeline.
  let decodeStartNanoseconds: Int64
  let decodeEndNanoseconds: Int64
  let coreStartNanoseconds: Int64
  let coreEndNanoseconds: Int64
  let leadingOverlapFrames: Int
  let trailingOverlapFrames: Int

  var isValid: Bool {
    decodeStartNanoseconds >= 0
      && decodeEndNanoseconds > decodeStartNanoseconds
      && coreStartNanoseconds >= decodeStartNanoseconds
      && coreEndNanoseconds > coreStartNanoseconds
      && coreEndNanoseconds <= decodeEndNanoseconds
      && leadingOverlapFrames >= 0 && trailingOverlapFrames >= 0
  }
}

struct RemoteClusterRestorationOptions: Codable, Hashable, Sendable {
  let restorationModelIdentifier: String
  let restorationAssetSHA256: String
  let detectorModelIdentifier: String
  let detectorAssetSHA256: String
  let restorationClipLength: Int
  let temporalOverlap: Int
  let crossfade: Bool
  let detectionEmptyLookahead: Int
  let detectFaceMosaics: Bool
  let blendFeather: Float
  let sharpenStrength: Float
  let detailBoost: Float
  let textureMix: Float
  let smoothStrength: Float
  let effectUpscale: Int
  let roiEnhancerModelIdentifier: String?
  let roiEnhancerAssetSHA256: String?
  let roiEnhancerStrength: Float
  let roiEnhancerScale: Int
  let videoCodec: String
  let bitrateMultiplier: Double
  let mp4FastStart: Bool
  /// Worker v1 intentionally rejects frame-rate conversion because each
  /// process would otherwise restart the sampling phase at its own boundary.
  let targetFPSNumerator: Int?
  let targetFPSDenominator: Int?

  var isValid: Bool {
    !restorationModelIdentifier.isEmpty && !detectorModelIdentifier.isEmpty
      && Self.isSHA256(restorationAssetSHA256)
      && Self.isSHA256(detectorAssetSHA256)
      && restorationClipLength > 0 && temporalOverlap >= 0
      && temporalOverlap < restorationClipLength
      && detectionEmptyLookahead >= 1
      && blendFeather.isFinite && blendFeather >= 0
      && sharpenStrength.isFinite && detailBoost.isFinite
      && textureMix.isFinite && smoothStrength.isFinite
      && effectUpscale >= 1 && roiEnhancerStrength.isFinite
      && roiEnhancerStrength >= 0 && roiEnhancerScale >= 1
      && ((roiEnhancerModelIdentifier == nil && roiEnhancerAssetSHA256 == nil)
        || (roiEnhancerModelIdentifier?.isEmpty == false
          && roiEnhancerAssetSHA256.map(Self.isSHA256) == true))
      && (videoCodec == "h264" || videoCodec == "hevc")
      && bitrateMultiplier.isFinite && bitrateMultiplier > 0
      && targetFPSNumerator == nil && targetFPSDenominator == nil
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy {
      (48...57).contains($0) || (97...102).contains($0)
    }
  }
}

struct RemoteClusterJobRequest: Codable, Hashable, Sendable {
  static let protocolVersion = 2

  let protocolVersion: Int
  let jobID: UUID
  let attemptID: UUID
  let leaseID: UUID
  let coordinatorNodeID: UUID
  let sharedRootIdentifier: String
  let inputByteCount: Int64
  let inputSHA256: String
  let inputRelativePath: RemoteClusterRelativePath
  let outputRelativePath: RemoteClusterRelativePath
  let mediaRange: RemoteClusterMediaRange
  let options: RemoteClusterRestorationOptions
  let createdAt: Date
  let leaseExpiresAt: Date
  /// Present only when this attempt reads and uploads through the
  /// Coordinator's dedicated transfer listener.
  let httpTransfer: RemoteClusterHTTPTransferDescriptor?

  init(
    protocolVersion: Int = Self.protocolVersion,
    jobID: UUID,
    attemptID: UUID,
    leaseID: UUID,
    coordinatorNodeID: UUID,
    sharedRootIdentifier: String,
    inputByteCount: Int64,
    inputSHA256: String,
    inputRelativePath: RemoteClusterRelativePath,
    outputRelativePath: RemoteClusterRelativePath,
    mediaRange: RemoteClusterMediaRange,
    options: RemoteClusterRestorationOptions,
    createdAt: Date,
    leaseExpiresAt: Date,
    httpTransfer: RemoteClusterHTTPTransferDescriptor? = nil
  ) {
    self.protocolVersion = protocolVersion
    self.jobID = jobID
    self.attemptID = attemptID
    self.leaseID = leaseID
    self.coordinatorNodeID = coordinatorNodeID
    self.sharedRootIdentifier = sharedRootIdentifier
    self.inputByteCount = inputByteCount
    self.inputSHA256 = inputSHA256
    self.inputRelativePath = inputRelativePath
    self.outputRelativePath = outputRelativePath
    self.mediaRange = mediaRange
    self.options = options
    self.createdAt = createdAt
    self.leaseExpiresAt = leaseExpiresAt
    self.httpTransfer = httpTransfer
  }


  func withHTTPTransfer(_ descriptor: RemoteClusterHTTPTransferDescriptor?) -> Self {
    Self(
      protocolVersion: protocolVersion,
      jobID: jobID,
      attemptID: attemptID,
      leaseID: leaseID,
      coordinatorNodeID: coordinatorNodeID,
      sharedRootIdentifier: sharedRootIdentifier,
      inputByteCount: inputByteCount,
      inputSHA256: inputSHA256,
      inputRelativePath: inputRelativePath,
      outputRelativePath: outputRelativePath,
      mediaRange: mediaRange,
      options: options,
      createdAt: createdAt,
      leaseExpiresAt: leaseExpiresAt,
      httpTransfer: descriptor
    )
  }

  /// Planned jobs may wait in the Coordinator queue much longer than one
  /// lease. Refresh only the not-yet-submitted deadline while preserving the
  /// exact attempt, lease, job and creation identities used for idempotency.
  func withLeaseExpiration(_ expiration: Date) -> Self {
    Self(
      protocolVersion: protocolVersion,
      jobID: jobID,
      attemptID: attemptID,
      leaseID: leaseID,
      coordinatorNodeID: coordinatorNodeID,
      sharedRootIdentifier: sharedRootIdentifier,
      inputByteCount: inputByteCount,
      inputSHA256: inputSHA256,
      inputRelativePath: inputRelativePath,
      outputRelativePath: outputRelativePath,
      mediaRange: mediaRange,
      options: options,
      createdAt: createdAt,
      leaseExpiresAt: expiration,
      httpTransfer: httpTransfer
    )
  }
}

enum RemoteClusterAttemptState: String, Codable, Hashable, Sendable {
  case accepted
  case running
  case completed
  case failed
  case cancelled
  case expired

  var isTerminal: Bool {
    switch self {
    case .completed, .failed, .cancelled, .expired: true
    case .accepted, .running: false
    }
  }
}

struct RemoteClusterJobMetrics: Codable, Hashable, Sendable {
  let processedFrames: Int
  let wallSeconds: Double
  let outputByteCount: Int64
}

struct RemoteClusterAttemptRecord: Codable, Hashable, Identifiable, Sendable {
  var id: UUID { attemptID }

  let jobID: UUID
  let attemptID: UUID
  let leaseID: UUID
  let coordinatorNodeID: UUID
  let inputRelativePath: RemoteClusterRelativePath
  let outputRelativePath: RemoteClusterRelativePath
  var state: RemoteClusterAttemptState
  var leaseExpiresAt: Date
  var updatedAt: Date
  var metrics: RemoteClusterJobMetrics?
  var failureCode: String?
}

enum RemoteClusterAdmissionDisposition: String, Codable, Hashable, Sendable {
  case accepted
  case duplicate
  case conflict
  case rejected
}

struct RemoteClusterJobAdmission: Codable, Hashable, Sendable {
  let disposition: RemoteClusterAdmissionDisposition
  let record: RemoteClusterAttemptRecord?
  let reason: String?
}

struct RemoteClusterNodeStatus: Codable, Hashable, Sendable {
  let protocolVersion: Int
  let nodeID: UUID
  let observedAt: Date
  let capabilities: RemoteClusterCapabilities
  let activeAttempts: [RemoteClusterAttemptRecord]
  let acceptingJobs: Bool
}

struct RemoteClusterStatusQuery: Codable, Hashable, Sendable {
  /// Nil requests node-wide status; a value requests one exact attempt.
  let attemptID: UUID?
}

struct RemoteClusterCancelRequest: Codable, Hashable, Sendable {
  let attemptID: UUID
  let leaseID: UUID
}

struct RemoteClusterRenewLeaseRequest: Codable, Hashable, Sendable {
  let attemptID: UUID
  let leaseID: UUID
  let newExpiration: Date
}

enum RemoteClusterRPCAction: Codable, Hashable, Sendable {
  case capabilities
  case submit(RemoteClusterJobRequest)
  case status(RemoteClusterStatusQuery)
  case renewLease(RemoteClusterRenewLeaseRequest)
  case cancel(RemoteClusterCancelRequest)
}

struct RemoteClusterRPCRequest: Codable, Hashable, Sendable {
  let protocolVersion: Int
  let requestID: UUID
  let coordinatorNodeID: UUID
  let action: RemoteClusterRPCAction
}

struct RemoteClusterRPCResponse: Codable, Hashable, Sendable {
  let protocolVersion: Int
  let requestID: UUID
  let ok: Bool
  let capabilities: RemoteClusterCapabilities?
  let admission: RemoteClusterJobAdmission?
  let nodeStatus: RemoteClusterNodeStatus?
  let attempt: RemoteClusterAttemptRecord?
  let errorCode: String?

  static func error(requestID: UUID, _ code: String) -> Self {
    Self(
      protocolVersion: RemoteClusterCapabilities.protocolVersion,
      requestID: requestID,
      ok: false,
      capabilities: nil,
      admission: nil,
      nodeStatus: nil,
      attempt: nil,
      errorCode: code
    )
  }
}

// MARK: - Shared-file integrity

private enum RemoteClusterExecutionFailure: Error {
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

  var code: String {
    switch self {
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
    }
  }
}

private struct RemoteClusterFileSignature: Hashable, Sendable {
  let device: UInt64
  let inode: UInt64
  let byteCount: Int64
  let modificationSeconds: Int64
  let modificationNanoseconds: Int64
  let changeSeconds: Int64
  let changeNanoseconds: Int64
}

private struct RemoteClusterFileIntegrity: Sendable {
  let signature: RemoteClusterFileSignature
  let sha256: String

  static func signature(of url: URL) throws -> RemoteClusterFileSignature {
    var value = stat()
    guard lstat(url.path, &value) == 0 else {
      if errno == ENOENT { throw RemoteClusterExecutionFailure.inputMissing }
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    guard (value.st_mode & S_IFMT) == S_IFREG else {
      throw RemoteClusterExecutionFailure.inputNotRegular
    }
    return makeSignature(value)
  }

  static func inspect(_ url: URL, expected: RemoteClusterFileSignature) throws -> Self {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    var opened = stat()
    guard fstat(handle.fileDescriptor, &opened) == 0 else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    guard (opened.st_mode & S_IFMT) == S_IFREG, makeSignature(opened) == expected else {
      throw RemoteClusterExecutionFailure.inputChangedDuringHash
    }

    var hasher = SHA256()
    var bytesRead: Int64 = 0
    let chunkBytes = 4 * 1024 * 1024
    while true {
      try Task.checkCancellation()
      guard let data = try handle.read(upToCount: chunkBytes), !data.isEmpty else { break }
      hasher.update(data: data)
      bytesRead += Int64(data.count)
    }

    var finished = stat()
    guard fstat(handle.fileDescriptor, &finished) == 0 else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    guard makeSignature(finished) == expected, bytesRead == expected.byteCount else {
      throw RemoteClusterExecutionFailure.inputChangedDuringHash
    }
    let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
    return Self(signature: expected, sha256: digest)
  }

  private static func makeSignature(_ value: stat) -> RemoteClusterFileSignature {
    RemoteClusterFileSignature(
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

/// Shards of one movie share this cache, so a multi-gigabyte source is hashed
/// once per unchanged inode rather than once per job. Concurrent first users
/// await the same in-flight task.
private actor RemoteClusterInputIntegrityCache {
  private let maximumEntries = 16
  private var completed: [RemoteClusterFileSignature: RemoteClusterFileIntegrity] = [:]
  private var completionOrder: [RemoteClusterFileSignature] = []
  private var inFlight: [RemoteClusterFileSignature: Task<RemoteClusterFileIntegrity, Error>] = [:]

  func inspect(_ url: URL) async throws -> RemoteClusterFileIntegrity {
    let signature = try RemoteClusterFileIntegrity.signature(of: url)
    if let cached = completed[signature] { return cached }
    if let task = inFlight[signature] { return try await task.value }

    let task = Task.detached(priority: .utility) {
      try RemoteClusterFileIntegrity.inspect(url, expected: signature)
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

/// Cross-process O_EXCL reservation prevents two workers sharing a filesystem
/// from simultaneously publishing the same output path.
private struct RemoteClusterOutputReservation {
  let lockURL: URL

  static func acquire(outputURL: URL, attemptID: UUID) throws -> Self {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: outputURL.path) {
      throw RemoteClusterExecutionFailure.outputExists
    }
    let lockURL = outputURL.appendingPathExtension("mioh-cluster.lock")
    let descriptor = open(
      lockURL.path,
      O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC,
      S_IRUSR | S_IWUSR
    )
    guard descriptor >= 0 else {
      if errno == EEXIST { throw RemoteClusterExecutionFailure.outputReservationConflict }
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

  func release() {
    try? FileManager.default.removeItem(at: lockURL)
  }
}

// MARK: - Shared-root path safety

enum RemoteClusterPathError: Error, Equatable {
  case empty
  case absolute
  case traversal
  case invalidCharacter
  case symbolicLink
  case outsideSharedRoot
}

/// A validated, platform-neutral path relative to the configured shared root.
/// Backslashes are rejected rather than interpreted differently by another
/// platform. Existing symlink components are also rejected when resolving.
struct RemoteClusterRelativePath: RawRepresentable, Codable, Hashable, Sendable {
  let rawValue: String

  init?(rawValue: String) {
    guard let value = try? Self.validate(rawValue) else { return nil }
    self.rawValue = value
  }

  init(validating value: String) throws {
    rawValue = try Self.validate(value)
  }

  init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    rawValue = try Self.validate(container.decode(String.self))
  }

  func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(rawValue)
  }

  private static func validate(_ value: String) throws -> String {
    guard !value.isEmpty else { throw RemoteClusterPathError.empty }
    guard !value.hasPrefix("/"), !value.hasPrefix("~") else {
      throw RemoteClusterPathError.absolute
    }
    guard !value.contains("\0"), !value.contains("\\") else {
      throw RemoteClusterPathError.invalidCharacter
    }
    let components = value.split(separator: "/", omittingEmptySubsequences: false)
    guard !components.isEmpty else { throw RemoteClusterPathError.empty }
    guard components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
      throw RemoteClusterPathError.traversal
    }
    return components.map(String.init).joined(separator: "/")
  }

  func resolve(
    beneath sharedRoot: URL,
    fileManager: FileManager = .default
  ) throws -> URL {
    let root = sharedRoot.standardizedFileURL.resolvingSymlinksInPath()
    var cursor = root
    for component in rawValue.split(separator: "/").map(String.init) {
      cursor.appendPathComponent(component, isDirectory: false)
      if fileManager.fileExists(atPath: cursor.path) {
        let values = try cursor.resourceValues(forKeys: [.isSymbolicLinkKey])
        if values.isSymbolicLink == true { throw RemoteClusterPathError.symbolicLink }
      }
      let standardized = cursor.standardizedFileURL
      guard Self.isDescendant(standardized, of: root) else {
        throw RemoteClusterPathError.outsideSharedRoot
      }
      cursor = standardized
    }
    return cursor
  }

  private static func isDescendant(_ candidate: URL, of root: URL) -> Bool {
    let rootComponents = root.pathComponents
    let candidateComponents = candidate.pathComponents
    guard candidateComponents.count >= rootComponents.count else { return false }
    return Array(candidateComponents.prefix(rootComponents.count)) == rootComponents
  }
}

// MARK: - Bonjour metadata

enum RemoteClusterBonjourError: Error, Equatable {
  case sensitiveField
  case unsupportedVersion
  case malformed
}

struct RemoteClusterBonjourMetadata: Codable, Hashable, Sendable {
  /// `_mioh._tcp` remains the user-facing control/iOS service. Compute
  /// workers use a distinct type so an iPhone never appears as a worker.
  static let serviceType = "_mioh-worker._tcp"
  static let serviceDomain = "local."
  private static let sensitiveKeys: Set<String> = [
    "token", "auth", "authorization", "bearer", "credential", "secret", "password",
  ]

  let protocolVersion: Int
  let nodeID: UUID
  let displayName: String
  let role: RemoteClusterRole
  let transferMode: RemoteClusterTransferMode
  let sharedRootIdentifier: String
  let maximumConcurrentJobs: Int

  init(capabilities: RemoteClusterCapabilities) {
    protocolVersion = capabilities.protocolVersion
    nodeID = capabilities.nodeID
    displayName = capabilities.displayName
    role = capabilities.role
    transferMode = capabilities.transferMode
    sharedRootIdentifier = capabilities.sharedRootIdentifier
    maximumConcurrentJobs = capabilities.maximumConcurrentJobs
  }

  init(txtRecord: NWTXTRecord) throws {
    try self.init(dictionary: txtRecord.dictionary)
  }

  init(txtRecordData: Data) throws {
    try self.init(dictionary: NetService.dictionary(fromTXTRecord: txtRecordData).compactMapValues {
      String(data: $0, encoding: .utf8)
    })
  }

  init(dictionary: [String: String]) throws {
    let normalized = Dictionary(uniqueKeysWithValues: dictionary.map { ($0.key.lowercased(), $0.value) })
    if !Self.sensitiveKeys.isDisjoint(with: normalized.keys) {
      throw RemoteClusterBonjourError.sensitiveField
    }
    guard let versionText = normalized["v"], let version = Int(versionText) else {
      throw RemoteClusterBonjourError.malformed
    }
    guard version == RemoteClusterCapabilities.protocolVersion else {
      throw RemoteClusterBonjourError.unsupportedVersion
    }
    // HTTP-only Workers do not use a shared root. Early iPad builds omitted
    // the optional TXT key entirely, so treat its absence as the empty value
    // while still requiring a non-empty root for shared-root transport.
    let root = normalized["root"] ?? ""
    guard let nodeText = normalized["node"], let parsedNodeID = UUID(uuidString: nodeText),
      let name = normalized["name"], !name.isEmpty, name.utf8.count <= 128,
      let roleText = normalized["role"], let parsedRole = RemoteClusterRole(rawValue: roleText),
      let transferText = normalized["transfer"],
      let parsedTransfer = RemoteClusterTransferMode(rawValue: transferText),
      root.utf8.count <= 128,
      (parsedTransfer == .coordinatorHTTPV1 || !root.isEmpty),
      let jobsText = normalized["jobs"], let jobs = Int(jobsText), (1...64).contains(jobs)
    else {
      throw RemoteClusterBonjourError.malformed
    }
    protocolVersion = version
    nodeID = parsedNodeID
    displayName = name
    role = parsedRole
    transferMode = parsedTransfer
    sharedRootIdentifier = root
    maximumConcurrentJobs = jobs
  }

  var txtRecord: NWTXTRecord {
    NWTXTRecord([
      "v": String(protocolVersion),
      "node": nodeID.uuidString.lowercased(),
      "name": displayName,
      "role": role.rawValue,
      "transfer": transferMode.rawValue,
      "root": sharedRootIdentifier,
      "jobs": String(maximumConcurrentJobs),
    ])
  }

  var txtRecordData: Data { txtRecord.data }
}

// MARK: - Worker attempt ledger

typealias RemoteClusterJobLauncher = @Sendable (
  _ request: RemoteClusterJobRequest,
  _ inputURL: URL,
  _ outputURL: URL
) async throws -> RemoteClusterJobMetrics

@MainActor
final class RemoteClusterWorkerJobLedger: ObservableObject {
  @Published private(set) var attemptRecords: [RemoteClusterAttemptRecord] = []

  private let maximumLeaseSeconds: TimeInterval
  private let maximumRetainedAttempts: Int
  private let inputIntegrityCache = RemoteClusterInputIntegrityCache()
  private var requests: [UUID: RemoteClusterJobRequest] = [:]
  private var executionTasks: [UUID: Task<Void, Never>] = [:]
  private var leaseWatchdogs: [UUID: Task<Void, Never>] = [:]

  /// Counts launchers which have not actually returned yet. A cancelled or
  /// expired attempt remains in flight until `runAccepted` completes its
  /// teardown, so capacity cannot be reused while the old native process is
  /// still releasing Core AI and IOSurface resources.
  var inFlightExecutionCount: Int { executionTasks.count }

  init(maximumLeaseSeconds: TimeInterval = 15 * 60, maximumRetainedAttempts: Int = 1_024) {
    self.maximumLeaseSeconds = max(30, maximumLeaseSeconds)
    self.maximumRetainedAttempts = max(32, maximumRetainedAttempts)
  }

  func admit(
    _ request: RemoteClusterJobRequest,
    sharedRootIdentifier: String,
    now: Date = Date()
  ) -> RemoteClusterJobAdmission {
    if let existing = record(attemptID: request.attemptID) {
      let isExactRetry = requests[request.attemptID] == request
      return RemoteClusterJobAdmission(
        disposition: isExactRetry ? .duplicate : .conflict,
        record: existing,
        reason: isExactRetry ? nil : "attempt_id_reused"
      )
    }
    if let stopping = attemptRecords.first(where: {
      $0.jobID == request.jobID && executionTasks[$0.attemptID] != nil
    }) {
      return RemoteClusterJobAdmission(
        disposition: .conflict,
        record: stopping,
        reason: stopping.state.isTerminal
          ? "previous_attempt_stopping" : "job_attempt_already_active"
      )
    }
    guard request.protocolVersion == RemoteClusterJobRequest.protocolVersion,
      (request.httpTransfer != nil
        || request.sharedRootIdentifier == sharedRootIdentifier),
      request.inputByteCount > 0,
      request.inputSHA256.utf8.count == 64,
      request.inputSHA256.utf8.allSatisfy({
        (48...57).contains($0) || (97...102).contains($0)
      }),
      request.mediaRange.isValid, request.options.isValid,
      request.inputRelativePath != request.outputRelativePath,
      request.httpTransfer?.isValid != false,
      request.httpTransfer.map({ $0.expiresAt >= request.leaseExpiresAt }) != false,
      request.leaseExpiresAt > now,
      request.leaseExpiresAt.timeIntervalSince(now) <= maximumLeaseSeconds
    else {
      return RemoteClusterJobAdmission(
        disposition: .rejected,
        record: nil,
        reason: "invalid_job_contract"
      )
    }

    if let activeIndex = attemptRecords.firstIndex(where: {
      $0.jobID == request.jobID && !$0.state.isTerminal
    }) {
      if attemptRecords[activeIndex].leaseExpiresAt > now {
        return RemoteClusterJobAdmission(
          disposition: .conflict,
          record: attemptRecords[activeIndex],
          reason: "job_attempt_already_active"
        )
      }
      attemptRecords[activeIndex].state = .expired
      attemptRecords[activeIndex].updatedAt = now
      let expiredAttemptID = attemptRecords[activeIndex].attemptID
      executionTasks[expiredAttemptID]?.cancel()
      stopLeaseWatchdog(attemptID: expiredAttemptID)
      if executionTasks[expiredAttemptID] != nil {
        return RemoteClusterJobAdmission(
          disposition: .conflict,
          record: attemptRecords[activeIndex],
          reason: "previous_attempt_stopping"
        )
      }
    }

    if let outputConflict = attemptRecords.first(where: {
      !$0.state.isTerminal && $0.outputRelativePath == request.outputRelativePath
    }) {
      return RemoteClusterJobAdmission(
        disposition: .conflict,
        record: outputConflict,
        reason: "output_path_already_active"
      )
    }
    pruneRetainedAttempts()
    guard attemptRecords.count < maximumRetainedAttempts else {
      return RemoteClusterJobAdmission(
        disposition: .rejected,
        record: nil,
        reason: "attempt_ledger_full"
      )
    }

    let record = RemoteClusterAttemptRecord(
      jobID: request.jobID,
      attemptID: request.attemptID,
      leaseID: request.leaseID,
      coordinatorNodeID: request.coordinatorNodeID,
      inputRelativePath: request.inputRelativePath,
      outputRelativePath: request.outputRelativePath,
      state: .accepted,
      leaseExpiresAt: request.leaseExpiresAt,
      updatedAt: now,
      metrics: nil,
      failureCode: nil
    )
    requests[request.attemptID] = request
    attemptRecords.append(record)
    scheduleLeaseWatchdog(attemptID: request.attemptID)
    return RemoteClusterJobAdmission(disposition: .accepted, record: record, reason: nil)
  }

  func renewLease(
    attemptID: UUID,
    leaseID: UUID,
    until newExpiration: Date,
    now: Date = Date()
  ) -> Bool {
    guard let index = index(attemptID: attemptID),
      attemptRecords[index].leaseID == leaseID,
      !attemptRecords[index].state.isTerminal,
      newExpiration > now,
      newExpiration > attemptRecords[index].leaseExpiresAt,
      newExpiration.timeIntervalSince(now) <= maximumLeaseSeconds
    else { return false }
    attemptRecords[index].leaseExpiresAt = newExpiration
    attemptRecords[index].updatedAt = now
    return true
  }

  func cancel(attemptID: UUID, leaseID: UUID, now: Date = Date()) -> Bool {
    guard let index = index(attemptID: attemptID),
      attemptRecords[index].leaseID == leaseID,
      !attemptRecords[index].state.isTerminal
    else { return false }
    attemptRecords[index].state = .cancelled
    attemptRecords[index].updatedAt = now
    executionTasks[attemptID]?.cancel()
    stopLeaseWatchdog(attemptID: attemptID)
    return true
  }

  /// Lease-scoped cancellation which does not return until the launcher has
  /// really exited. This is the coordinator's one-local-lane barrier.
  func cancelAndWait(attemptID: UUID, leaseID: UUID, now: Date = Date()) async -> Bool {
    guard let existing = record(attemptID: attemptID), existing.leaseID == leaseID else {
      return false
    }
    let task = executionTasks[attemptID]
    _ = cancel(attemptID: attemptID, leaseID: leaseID, now: now)
    task?.cancel()
    if let task { await task.value }
    return true
  }

  /// Stops every in-flight launcher when the local Worker service is disabled.
  /// Coordinator-issued cancellation remains lease-scoped; this is the local
  /// lifecycle boundary and therefore intentionally needs no remote lease.
  func cancelAllActive(now: Date = Date()) {
    let activeIDs = attemptRecords.indices.compactMap { index -> UUID? in
      guard !attemptRecords[index].state.isTerminal else { return nil }
      attemptRecords[index].state = .cancelled
      attemptRecords[index].updatedAt = now
      return attemptRecords[index].attemptID
    }
    for attemptID in activeIDs {
      executionTasks[attemptID]?.cancel()
      stopLeaseWatchdog(attemptID: attemptID)
    }
    // A record becomes terminal before its launcher has necessarily returned.
    // Cancel those teardown-phase tasks as well without dropping their handles.
    executionTasks.values.forEach { $0.cancel() }
  }

  /// Local lifecycle barrier used before a coordinator lane is reused.
  func cancelAllActiveAndWait(now: Date = Date()) async {
    cancelAllActive(now: now)
    let tasks = Array(executionTasks.values)
    for task in tasks {
      task.cancel()
      await task.value
    }
  }

  /// This is the sole future runner integration point. Exact-attempt retries
  /// return their existing record; they never launch the process twice.
  func submit(
    _ request: RemoteClusterJobRequest,
    sharedRoot: URL,
    sharedRootIdentifier: String,
    launcher: @escaping RemoteClusterJobLauncher,
    localInputURL: URL? = nil,
    localOutputURL: URL? = nil,
    now: Date = Date()
  ) -> RemoteClusterJobAdmission {
    let admission = admit(request, sharedRootIdentifier: sharedRootIdentifier, now: now)
    guard admission.disposition == .accepted else { return admission }
    let task = Task { @MainActor [weak self] in
      guard let self else { return }
      await self.runAccepted(
        request,
        sharedRoot: sharedRoot,
        launcher: launcher,
        localInputURL: localInputURL,
        localOutputURL: localOutputURL
      )
    }
    executionTasks[request.attemptID] = task
    return admission
  }

  func execute(
    _ request: RemoteClusterJobRequest,
    sharedRoot: URL,
    sharedRootIdentifier: String,
    launcher: RemoteClusterJobLauncher,
    localInputURL: URL? = nil,
    localOutputURL: URL? = nil,
    now: Date = Date()
  ) async -> RemoteClusterAttemptRecord? {
    let admission = admit(request, sharedRootIdentifier: sharedRootIdentifier, now: now)
    guard admission.disposition == .accepted else { return admission.record }
    await runAccepted(
      request,
      sharedRoot: sharedRoot,
      launcher: launcher,
      localInputURL: localInputURL,
      localOutputURL: localOutputURL
    )
    return record(attemptID: request.attemptID)
  }

  private func runAccepted(
    _ request: RemoteClusterJobRequest,
    sharedRoot: URL,
    launcher: RemoteClusterJobLauncher,
    localInputURL: URL? = nil,
    localOutputURL: URL? = nil
  ) async {
    guard let acceptedIndex = index(attemptID: request.attemptID),
      attemptRecords[acceptedIndex].state == .accepted
    else { return }
    defer {
      executionTasks.removeValue(forKey: request.attemptID)
      stopLeaseWatchdog(attemptID: request.attemptID)
    }
    var reservation: RemoteClusterOutputReservation?
    var stagingURL: URL?
    defer {
      reservation?.release()
      if let stagingURL { try? FileManager.default.removeItem(at: stagingURL) }
    }
    do {
      try Task.checkCancellation()
      let inputURL: URL
      let outputURL: URL?
      let candidateStagingURL: URL
      if let transfer = request.httpTransfer {
        guard let remoteInput = URL(string: transfer.inputURL) else {
          throw RemoteClusterHTTPTransferError.invalidContract
        }
        inputURL = remoteInput
        outputURL = nil
        candidateStagingURL = FileManager.default.temporaryDirectory
          .appendingPathComponent(
            "mioh-cluster-\(request.attemptID.uuidString.lowercased()).part.mp4"
          )
      } else {
        if let localInputURL {
          inputURL = localInputURL
        } else {
          inputURL = try request.inputRelativePath.resolve(beneath: sharedRoot)
        }
        var resolvedOutput: URL
        if let localOutputURL {
          resolvedOutput = localOutputURL
        } else {
          resolvedOutput = try request.outputRelativePath.resolve(beneath: sharedRoot)
        }
        let inputIntegrity = try await inputIntegrityCache.inspect(inputURL)
        try Task.checkCancellation()
        guard inputIntegrity.signature.byteCount == request.inputByteCount else {
          throw RemoteClusterExecutionFailure.inputByteCountMismatch
        }
        guard inputIntegrity.sha256 == request.inputSHA256 else {
          throw RemoteClusterExecutionFailure.inputSHA256Mismatch
        }
        let outputParent = resolvedOutput.deletingLastPathComponent()
        try FileManager.default.createDirectory(
          at: outputParent,
          withIntermediateDirectories: true,
          attributes: [.posixPermissions: 0o700]
        )
        // Shared-root attempts must be re-resolved after directory creation;
        // explicit local URLs were already chosen by this Coordinator.
        if localOutputURL == nil {
          resolvedOutput = try request.outputRelativePath.resolve(beneath: sharedRoot)
        }
        if FileManager.default.fileExists(atPath: resolvedOutput.path) {
          throw RemoteClusterExecutionFailure.outputExists
        }
        reservation = try RemoteClusterOutputReservation.acquire(
          outputURL: resolvedOutput,
          attemptID: request.attemptID
        )
        outputURL = resolvedOutput
        candidateStagingURL = resolvedOutput.deletingLastPathComponent()
          .appendingPathComponent(
            ".mioh-cluster-\(request.attemptID.uuidString.lowercased()).part"
          )
      }
      guard !FileManager.default.fileExists(atPath: candidateStagingURL.path) else {
        throw RemoteClusterExecutionFailure.outputReservationConflict
      }
      stagingURL = candidateStagingURL

      attemptRecords[acceptedIndex].state = .running
      attemptRecords[acceptedIndex].updatedAt = Date()
      let metrics = try await launcher(request, inputURL, candidateStagingURL)
      try Task.checkCancellation()
      guard let completedIndex = index(attemptID: request.attemptID),
        !attemptRecords[completedIndex].state.isTerminal
      else { return }

      var outputStat = stat()
      guard lstat(candidateStagingURL.path, &outputStat) == 0 else {
        throw RemoteClusterExecutionFailure.outputMissing
      }
      guard (outputStat.st_mode & S_IFMT) == S_IFREG else {
        throw RemoteClusterExecutionFailure.outputNotRegular
      }
      let actualOutputBytes = Int64(outputStat.st_size)
      guard actualOutputBytes > 0, metrics.outputByteCount == actualOutputBytes else {
        throw RemoteClusterExecutionFailure.outputByteCountMismatch
      }
      if let transfer = request.httpTransfer {
        try await RemoteClusterHTTPTransferClient.upload(
          file: candidateStagingURL,
          descriptor: transfer
        )
        try? FileManager.default.removeItem(at: candidateStagingURL)
        stagingURL = nil
      } else if let outputURL {
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
          throw RemoteClusterExecutionFailure.outputExists
        }
        try FileManager.default.moveItem(at: candidateStagingURL, to: outputURL)
        stagingURL = nil
      }
      attemptRecords[completedIndex].state = .completed
      attemptRecords[completedIndex].metrics = metrics
      attemptRecords[completedIndex].updatedAt = Date()
    } catch {
      if let failedIndex = index(attemptID: request.attemptID),
        !attemptRecords[failedIndex].state.isTerminal
      {
        attemptRecords[failedIndex].state = .failed
        attemptRecords[failedIndex].failureCode =
          (error as? RemoteClusterExecutionFailure)?.code ?? "launcher_failed"
        attemptRecords[failedIndex].updatedAt = Date()
      }
    }
  }

  func expireLeases(now: Date = Date()) {
    for index in attemptRecords.indices
    where !attemptRecords[index].state.isTerminal && attemptRecords[index].leaseExpiresAt <= now {
      attemptRecords[index].state = .expired
      attemptRecords[index].updatedAt = now
      let attemptID = attemptRecords[index].attemptID
      executionTasks[attemptID]?.cancel()
      stopLeaseWatchdog(attemptID: attemptID)
    }
  }

  func record(attemptID: UUID) -> RemoteClusterAttemptRecord? {
    attemptRecords.first { $0.attemptID == attemptID }
  }

  private func index(attemptID: UUID) -> Int? {
    attemptRecords.firstIndex { $0.attemptID == attemptID }
  }

  /// Lease expiry must not depend on another RPC arriving. Each accepted
  /// attempt owns one bounded watchdog which follows lease renewals and
  /// cancels its launcher as soon as the current lease actually expires.
  private func scheduleLeaseWatchdog(attemptID: UUID) {
    stopLeaseWatchdog(attemptID: attemptID)
    leaseWatchdogs[attemptID] = Task { @MainActor [weak self] in
      guard let self else { return }
      defer { self.leaseWatchdogs.removeValue(forKey: attemptID) }
      while let record = self.record(attemptID: attemptID), !record.state.isTerminal {
        let seconds = max(0, record.leaseExpiresAt.timeIntervalSinceNow)
        if seconds > 0 {
          let nanoseconds = UInt64(min(seconds, 24 * 60 * 60) * 1_000_000_000)
          do {
            try await Task.sleep(nanoseconds: max(1_000_000, nanoseconds))
          } catch {
            return
          }
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
    guard attemptRecords.count >= maximumRetainedAttempts else { return }
    let removable = attemptRecords
      .filter { $0.state.isTerminal && executionTasks[$0.attemptID] == nil }
      .sorted { $0.updatedAt < $1.updatedAt }
    let removeCount = min(
      removable.count,
      attemptRecords.count - maximumRetainedAttempts + 1
    )
    let removedIDs = Set(removable.prefix(removeCount).map(\.attemptID))
    guard !removedIDs.isEmpty else { return }
    attemptRecords.removeAll { removedIDs.contains($0.attemptID) }
    for attemptID in removedIDs {
      requests.removeValue(forKey: attemptID)
      stopLeaseWatchdog(attemptID: attemptID)
    }
  }
}

// MARK: - Trusted-LAN one-request transport

enum RemoteClusterTransportError: LocalizedError, Equatable {
  case frameTooLarge
  case incompleteFrame
  case encodingFailed
  case decodingFailed
  case connectionFailed(String)

  var errorDescription: String? {
    switch self {
    case .frameTooLarge: "RPCフレームが上限を超えています"
    case .incompleteFrame: "WorkerからのRPC応答が途中で終了しました"
    case .encodingFailed: "WorkerへのRPC要求を作成できませんでした"
    case .decodingFailed: "WorkerからのRPC応答を解析できませんでした"
    case .connectionFailed(let reason): "Workerへ接続できませんでした: \(reason)"
    }
  }
}

private enum RemoteClusterFraming {
  static let maximumJSONBytes = 64 * 1024

  static func encodedFrame<T: Encodable>(_ value: T) throws -> Data {
    let payload: Data
    do {
      payload = try JSONEncoder().encode(value)
    } catch {
      throw RemoteClusterTransportError.encodingFailed
    }
    guard payload.count <= maximumJSONBytes else {
      throw RemoteClusterTransportError.frameTooLarge
    }
    var length = UInt32(payload.count).bigEndian
    var frame = Data(bytes: &length, count: MemoryLayout<UInt32>.size)
    frame.append(payload)
    return frame
  }

  static func receiveFrame(
    on connection: NWConnection,
    completion: @escaping (Result<Data, Error>) -> Void
  ) {
    receiveExactly(on: connection, count: 4) { headerResult in
      switch headerResult {
      case .failure(let error):
        completion(.failure(error))
      case .success(let header):
        guard header.count == 4 else {
          completion(.failure(RemoteClusterTransportError.incompleteFrame))
          return
        }
        let length = header.withUnsafeBytes {
          UInt32(bigEndian: $0.loadUnaligned(as: UInt32.self))
        }
        guard length > 0, length <= UInt32(maximumJSONBytes) else {
          completion(.failure(RemoteClusterTransportError.frameTooLarge))
          return
        }
        receiveExactly(on: connection, count: Int(length), completion: completion)
      }
    }
  }

  static func sendFrame<T: Encodable>(
    _ value: T,
    on connection: NWConnection,
    completion: @escaping (Error?) -> Void
  ) {
    do {
      let frame = try encodedFrame(value)
      connection.send(content: frame, completion: .contentProcessed(completion))
    } catch {
      completion(error)
    }
  }

  private static func receiveExactly(
    on connection: NWConnection,
    count: Int,
    accumulated: Data = Data(),
    completion: @escaping (Result<Data, Error>) -> Void
  ) {
    let remaining = count - accumulated.count
    guard remaining > 0 else {
      completion(.success(accumulated))
      return
    }
    connection.receive(minimumIncompleteLength: 1, maximumLength: remaining) {
      data, _, isComplete, error in
      if let error {
        completion(.failure(error))
        return
      }
      var next = accumulated
      if let data { next.append(data) }
      if next.count == count {
        completion(.success(next))
      } else if isComplete || data?.isEmpty != false {
        completion(.failure(RemoteClusterTransportError.incompleteFrame))
      } else {
        receiveExactly(
          on: connection,
          count: count,
          accumulated: next,
          completion: completion
        )
      }
    }
  }
}

private final class RemoteClusterOnceFlag: @unchecked Sendable {
  private let lock = NSLock()
  private var value = false

  func claim() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard !value else { return false }
    value = true
    return true
  }
}

/// Bridges structured-concurrency cancellation to a Network.framework
/// connection. Without this bridge, cancelling one worker lane could leave a
/// task group waiting for the RPC timeout before it could fail fast.
private final class RemoteClusterConnectionCancellation: @unchecked Sendable {
  private let lock = NSLock()
  private var connection: NWConnection?
  private var cancelled = false

  func install(_ connection: NWConnection) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard !cancelled else { return false }
    self.connection = connection
    return true
  }

  func clear(_ connection: NWConnection) {
    lock.lock()
    defer { lock.unlock() }
    if self.connection === connection { self.connection = nil }
  }

  func cancel() {
    lock.lock()
    cancelled = true
    let connection = self.connection
    self.connection = nil
    lock.unlock()
    connection?.cancel()
  }
}

/// Trusted-LAN v1 transport: one length-prefixed JSON request and one reply
/// per TCP connection. Access is intentionally limited by network scope rather
/// than an application-level pairing credential.
@MainActor
private final class RemoteClusterWorkerTransport {
  private let capabilities: RemoteClusterCapabilities
  private let sharedRoot: URL
  private let ledger: RemoteClusterWorkerJobLedger
  private let launcher: RemoteClusterJobLauncher
  private let queue = DispatchQueue(label: "mioh.remote-cluster.worker-rpc")
  private let maximumConnections = 16
  private let requestTimeoutSeconds: TimeInterval = 15
  private var activeConnections: [UUID: NWConnection] = [:]

  init(
    capabilities: RemoteClusterCapabilities,
    sharedRoot: URL,
    ledger: RemoteClusterWorkerJobLedger,
    launcher: @escaping RemoteClusterJobLauncher
  ) {
    self.capabilities = capabilities
    self.sharedRoot = sharedRoot
    self.ledger = ledger
    self.launcher = launcher
  }

  func accept(_ connection: NWConnection) {
    guard activeConnections.count < maximumConnections else {
      connection.cancel()
      return
    }
    let connectionID = UUID()
    let receiveOnce = RemoteClusterOnceFlag()
    activeConnections[connectionID] = connection
    connection.stateUpdateHandler = { [weak self, weak connection] state in
      guard let connection else { return }
      switch state {
      case .ready where receiveOnce.claim():
        RemoteClusterFraming.receiveFrame(on: connection) { [connection] result in
          Task { @MainActor [weak self] in
            guard let self else {
              connection.cancel()
              return
            }
            await self.handle(result, connectionID: connectionID, connection: connection)
          }
        }
      case .failed, .cancelled:
        Task { @MainActor [weak self] in
          self?.finish(connectionID: connectionID, connection: connection)
        }
      default:
        break
      }
    }
    connection.start(queue: queue)
    queue.asyncAfter(deadline: .now() + requestTimeoutSeconds) { [weak self, weak connection] in
      guard let connection else { return }
      Task { @MainActor [weak self] in
        guard let self, self.activeConnections[connectionID] != nil else { return }
        self.finish(connectionID: connectionID, connection: connection)
      }
    }
  }

  private func handle(
    _ frameResult: Result<Data, Error>,
    connectionID: UUID,
    connection: NWConnection
  ) async {
    let request: RemoteClusterRPCRequest
    do {
      let data = try frameResult.get()
      request = try JSONDecoder().decode(RemoteClusterRPCRequest.self, from: data)
    } catch {
      reply(.error(requestID: UUID(), "malformed_request"), connectionID: connectionID, on: connection)
      return
    }
    guard request.protocolVersion == RemoteClusterCapabilities.protocolVersion else {
      reply(.error(requestID: request.requestID, "unsupported_protocol"), connectionID: connectionID, on: connection)
      return
    }
    // No pairing code is required for trusted-LAN cluster requests.

    ledger.expireLeases()
    let response: RemoteClusterRPCResponse
    switch request.action {
    case .capabilities:
      response = RemoteClusterRPCResponse(
        protocolVersion: RemoteClusterCapabilities.protocolVersion,
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
      let activeCount = max(
        ledger.inFlightExecutionCount,
        ledger.attemptRecords.filter { !$0.state.isTerminal }.count
      )
      let admission: RemoteClusterJobAdmission
      if ledger.record(attemptID: job.attemptID) == nil,
        activeCount >= capabilities.maximumConcurrentJobs
      {
        admission = RemoteClusterJobAdmission(
          disposition: .rejected,
          record: nil,
          reason: "worker_capacity_reached"
        )
      } else {
        admission = ledger.submit(
          job,
          sharedRoot: sharedRoot,
          sharedRootIdentifier: capabilities.sharedRootIdentifier,
          launcher: launcher
        )
      }
      response = RemoteClusterRPCResponse(
        protocolVersion: RemoteClusterCapabilities.protocolVersion,
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
        let attempt = ledger.record(attemptID: attemptID)
        let authorizedAttempt = attempt?.coordinatorNodeID == request.coordinatorNodeID
          ? attempt
          : nil
        response = RemoteClusterRPCResponse(
          protocolVersion: RemoteClusterCapabilities.protocolVersion,
          requestID: request.requestID,
          ok: authorizedAttempt != nil,
          capabilities: nil,
          admission: nil,
          nodeStatus: nil,
          attempt: authorizedAttempt,
          errorCode: authorizedAttempt == nil ? "attempt_not_found" : nil
        )
      } else {
        let allActive = ledger.attemptRecords.filter { !$0.state.isTerminal }
        let occupiedSlots = max(ledger.inFlightExecutionCount, allActive.count)
        let visibleActive = allActive.filter {
          !$0.state.isTerminal && $0.coordinatorNodeID == request.coordinatorNodeID
        }
        response = RemoteClusterRPCResponse(
          protocolVersion: RemoteClusterCapabilities.protocolVersion,
          requestID: request.requestID,
          ok: true,
          capabilities: nil,
          admission: nil,
          nodeStatus: RemoteClusterNodeStatus(
            protocolVersion: RemoteClusterCapabilities.protocolVersion,
            nodeID: capabilities.nodeID,
            observedAt: Date(),
            capabilities: capabilities,
            activeAttempts: visibleActive,
            acceptingJobs: occupiedSlots < capabilities.maximumConcurrentJobs
          ),
          attempt: nil,
          errorCode: nil
        )
      }
    case .renewLease(let renewal):
      let existing = ledger.record(attemptID: renewal.attemptID)
      let authorized = existing?.coordinatorNodeID == request.coordinatorNodeID
        && existing?.leaseID == renewal.leaseID
      let alreadyTerminal = authorized && existing?.state.isTerminal == true
      let renewed = authorized && !alreadyTerminal && ledger.renewLease(
        attemptID: renewal.attemptID,
        leaseID: renewal.leaseID,
        until: renewal.newExpiration
      )
      // A job may finish between the coordinator's renewal deadline check and
      // this RPC. Treat an authorized terminal record as an idempotent success;
      // the following status poll will report completed/failed/cancelled.
      let renewalAccepted = renewed || alreadyTerminal
      response = RemoteClusterRPCResponse(
        protocolVersion: RemoteClusterCapabilities.protocolVersion,
        requestID: request.requestID,
        ok: renewalAccepted,
        capabilities: nil,
        admission: nil,
        nodeStatus: nil,
        attempt: renewalAccepted ? ledger.record(attemptID: renewal.attemptID) : nil,
        errorCode: renewalAccepted ? nil : "lease_renewal_rejected"
      )
    case .cancel(let cancellation):
      let existing = ledger.record(attemptID: cancellation.attemptID)
      let cancelled = existing?.coordinatorNodeID == request.coordinatorNodeID
        && ledger.cancel(
          attemptID: cancellation.attemptID,
          leaseID: cancellation.leaseID
        )
      response = RemoteClusterRPCResponse(
        protocolVersion: RemoteClusterCapabilities.protocolVersion,
        requestID: request.requestID,
        ok: cancelled,
        capabilities: nil,
        admission: nil,
        nodeStatus: nil,
        attempt: cancelled ? ledger.record(attemptID: cancellation.attemptID) : nil,
        errorCode: cancelled ? nil : "cancel_rejected"
      )
    }
    reply(response, connectionID: connectionID, on: connection)
  }

  private func reply(
    _ response: RemoteClusterRPCResponse,
    connectionID: UUID,
    on connection: NWConnection
  ) {
    RemoteClusterFraming.sendFrame(response, on: connection) { [weak self, weak connection] _ in
      guard let connection else { return }
      Task { @MainActor [weak self] in
        self?.finish(connectionID: connectionID, connection: connection)
      }
    }
  }

  private func finish(connectionID: UUID, connection: NWConnection) {
    activeConnections.removeValue(forKey: connectionID)
    connection.stateUpdateHandler = nil
    connection.cancel()
  }
}

@MainActor
final class RemoteClusterClient {
  private let localNodeID: UUID
  private let queue = DispatchQueue(label: "mioh.remote-cluster.client-rpc")

  init(localNodeID: UUID) {
    self.localNodeID = localNodeID
  }

  func call(
    _ action: RemoteClusterRPCAction,
    node: RemoteClusterDiscoveredNode
  ) async throws -> RemoteClusterRPCResponse {
    let request = RemoteClusterRPCRequest(
      protocolVersion: RemoteClusterCapabilities.protocolVersion,
      requestID: UUID(),
      coordinatorNodeID: localNodeID,
      action: action
    )
    let frame = try RemoteClusterFraming.encodedFrame(request)
    do {
      return try await send(
        frame: frame,
        requestID: request.requestID,
        endpoint: node.endpoint
      )
    } catch let error as RemoteClusterTransportError {
      // A Bonjour result may be tied to a transient interface path even while
      // the same service remains reachable on the LAN. Retry only connection
      // failures through an interface-neutral service endpoint; RPC-level
      // responses (including unauthorized) are never retried here.
      guard case .connectionFailed = error,
        case .service(let name, let type, let domain, let interface) = node.endpoint,
        interface != nil
      else { throw error }
      return try await send(
        frame: frame,
        requestID: request.requestID,
        endpoint: .service(name: name, type: type, domain: domain, interface: nil)
      )
    }
  }

  private func send(
    frame: Data,
    requestID: UUID,
    endpoint: NWEndpoint
  ) async throws -> RemoteClusterRPCResponse {
    let cancellation = RemoteClusterConnectionCancellation()
    return try await withTaskCancellationHandler {
      try Task.checkCancellation()
      return try await withCheckedThrowingContinuation { continuation in
        let connection = NWConnection(to: endpoint, using: .tcp)
        let startOnce = RemoteClusterOnceFlag()
        let finishOnce = RemoteClusterOnceFlag()
        let finish: @Sendable (Result<RemoteClusterRPCResponse, Error>) -> Void = { result in
          guard finishOnce.claim() else { return }
          cancellation.clear(connection)
          connection.stateUpdateHandler = nil
          connection.cancel()
          continuation.resume(with: result)
        }
        guard cancellation.install(connection) else {
          finish(.failure(CancellationError()))
          return
        }
        connection.stateUpdateHandler = { state in
          switch state {
          case .ready where startOnce.claim():
            connection.send(content: frame, completion: .contentProcessed { error in
              if let error {
                finish(.failure(error))
                return
              }
              RemoteClusterFraming.receiveFrame(on: connection) { result in
                do {
                  let response = try JSONDecoder().decode(
                    RemoteClusterRPCResponse.self,
                    from: result.get()
                  )
                  guard response.requestID == requestID else {
                    throw RemoteClusterTransportError.decodingFailed
                  }
                  finish(.success(response))
                } catch {
                  finish(.failure(error))
                }
              }
            })
          case .failed(let error):
            finish(.failure(RemoteClusterTransportError.connectionFailed(error.localizedDescription)))
          case .cancelled:
            finish(.failure(CancellationError()))
          default:
            break
          }
        }
        connection.start(queue: queue)
        queue.asyncAfter(deadline: .now() + 15) {
          finish(.failure(RemoteClusterTransportError.connectionFailed("request_timeout")))
        }
      }
    } onCancel: {
      cancellation.cancel()
    }
  }
}

// MARK: - Discovery and worker service lifecycle

struct RemoteClusterDiscoveredNode: Identifiable, Hashable, Sendable {
  var id: UUID { metadata.nodeID }

  let metadata: RemoteClusterBonjourMetadata
  let endpoint: NWEndpoint
  let serviceName: String
  let serviceDomain: String
  let interfaceName: String?
  let lastSeenAt: Date
}

enum RemoteClusterDiscoveryState: Equatable, Sendable {
  case stopped
  case starting
  case ready
  case waiting(String)
  case failed(String)
}

enum RemoteClusterWorkerState: Equatable, Sendable {
  case stopped
  case starting
  case ready(port: UInt16)
  case waiting(String)
  case failed(String)
}

@MainActor
final class RemoteClusterService: ObservableObject {
  @Published private(set) var mode: RemoteClusterRole?
  @Published private(set) var discoveryState: RemoteClusterDiscoveryState = .stopped
  @Published private(set) var workerState: RemoteClusterWorkerState = .stopped
  @Published private(set) var discoveredNodes: [RemoteClusterDiscoveredNode] = []
  @Published private(set) var workerSharedRoot: URL?
  @Published private(set) var localCapabilities: RemoteClusterCapabilities?

  let jobLedger: RemoteClusterWorkerJobLedger

  private let localNodeID: UUID
  private let networkQueue = DispatchQueue(label: "mioh.remote-cluster.network")
  private var browser: NWBrowser?
  private var listener: NWListener?
  private var workerTransport: RemoteClusterWorkerTransport?
  private var generation = UUID()

  init(
    localNodeID: UUID,
    jobLedger: RemoteClusterWorkerJobLedger? = nil
  ) {
    self.localNodeID = localNodeID
    self.jobLedger = jobLedger ?? RemoteClusterWorkerJobLedger()
  }

  func startCoordinatorDiscovery() {
    stopWorker()
    stopBrowser()
    generation = UUID()
    mode = .coordinator
    discoveryState = .starting
    let browser = NWBrowser(
      for: .bonjourWithTXTRecord(
        type: RemoteClusterBonjourMetadata.serviceType,
        domain: RemoteClusterBonjourMetadata.serviceDomain
      ),
      using: .tcp
    )
    let currentGeneration = generation
    self.browser = browser
    browser.stateUpdateHandler = { [weak self] state in
      Task { @MainActor [weak self] in
        guard let self, self.generation == currentGeneration else { return }
        switch state {
        case .setup:
          self.discoveryState = .starting
        case .ready:
          self.discoveryState = .ready
        case .waiting(let error):
          self.discoveryState = .waiting(error.localizedDescription)
        case .failed(let error):
          self.discoveryState = .failed(error.localizedDescription)
        case .cancelled:
          self.discoveryState = .stopped
        @unknown default:
          self.discoveryState = .failed("unknown_browser_state")
        }
      }
    }
    browser.browseResultsChangedHandler = { [weak self] results, _ in
      Task { @MainActor [weak self] in
        guard let self, self.generation == currentGeneration else { return }
        self.updateDiscoveredNodes(from: results)
      }
    }
    browser.start(queue: networkQueue)
  }

  /// Starts a Bonjour-advertised worker endpoint. The networking foundation
  /// owns discovery and lifecycle only; `connectionHandler` is where a later
  /// authenticated job protocol adapter must be installed.
  func startWorker(
    sharedRoot: URL?,
    capabilities: RemoteClusterCapabilities,
    launcher: @escaping RemoteClusterJobLauncher
  ) throws {
    stopBrowser()
    stopWorker()
    generation = UUID()
    guard capabilities.nodeID == localNodeID, capabilities.role == .worker,
      capabilities.protocolVersion == RemoteClusterCapabilities.protocolVersion,
      capabilities.effectiveTransferModes.contains(capabilities.transferMode)
    else { throw RemoteClusterBonjourError.malformed }

    let supportsSharedRoot = capabilities.effectiveTransferModes.contains(.sharedRootV1)
    if supportsSharedRoot {
      var isDirectory: ObjCBool = false
      guard let sharedRoot,
        FileManager.default.fileExists(atPath: sharedRoot.path, isDirectory: &isDirectory),
        isDirectory.boolValue,
        !capabilities.sharedRootIdentifier.isEmpty
      else { throw CocoaError(.fileNoSuchFile) }
    }
    let workerRoot: URL
    if let sharedRoot {
      workerRoot = sharedRoot.standardizedFileURL.resolvingSymlinksInPath()
    } else {
      workerRoot = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mioh-cluster-worker-v1",
        isDirectory: true
      )
      try FileManager.default.createDirectory(
        at: workerRoot,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
      )
    }

    mode = .worker
    workerSharedRoot = sharedRoot?.standardizedFileURL.resolvingSymlinksInPath()
    localCapabilities = capabilities
    workerState = .starting

    let listener = try NWListener(using: .tcp, on: .any)
    let metadata = RemoteClusterBonjourMetadata(capabilities: capabilities)
    listener.service = NWListener.Service(
      name: capabilities.displayName,
      type: RemoteClusterBonjourMetadata.serviceType,
      domain: RemoteClusterBonjourMetadata.serviceDomain,
      txtRecord: metadata.txtRecord
    )
    let currentGeneration = generation
    let transport = RemoteClusterWorkerTransport(
      capabilities: capabilities,
      sharedRoot: workerRoot,
      ledger: jobLedger,
      launcher: launcher
    )
    workerTransport = transport
    self.listener = listener
    listener.newConnectionHandler = { [weak transport] connection in
      Task { @MainActor [weak transport] in
        guard let transport else {
          connection.cancel()
          return
        }
        transport.accept(connection)
      }
    }
    listener.stateUpdateHandler = { [weak self, weak listener] state in
      Task { @MainActor [weak self, weak listener] in
        guard let self, self.generation == currentGeneration, self.listener === listener else {
          return
        }
        switch state {
        case .setup:
          self.workerState = .starting
        case .waiting(let error):
          self.workerState = .waiting(error.localizedDescription)
        case .ready:
          self.workerState = .ready(port: listener?.port?.rawValue ?? 0)
        case .failed(let error):
          self.workerState = .failed(error.localizedDescription)
        case .cancelled:
          self.workerState = .stopped
        @unknown default:
          self.workerState = .failed("unknown_listener_state")
        }
      }
    }
    listener.start(queue: networkQueue)
  }

  func stop() {
    generation = UUID()
    stopBrowser()
    stopWorker()
    mode = nil
  }

  private func stopBrowser() {
    browser?.cancel()
    browser = nil
    discoveredNodes = []
    discoveryState = .stopped
  }

  private func stopWorker() {
    listener?.cancel()
    listener = nil
    workerTransport = nil
    jobLedger.cancelAllActive()
    workerSharedRoot = nil
    localCapabilities = nil
    workerState = .stopped
  }

  private func updateDiscoveredNodes(from results: Set<NWBrowser.Result>) {
    var nodes: [RemoteClusterDiscoveredNode] = []
    let now = Date()
    for result in results {
      guard case .bonjour(let txtRecord) = result.metadata,
        let metadata = try? RemoteClusterBonjourMetadata(txtRecord: txtRecord),
        metadata.nodeID != localNodeID,
        metadata.role == .worker,
        case .service(let name, _, let domain, let interface) = result.endpoint
      else { continue }
      nodes.append(
        RemoteClusterDiscoveredNode(
          metadata: metadata,
          endpoint: result.endpoint,
          serviceName: name,
          serviceDomain: domain,
          interfaceName: interface?.name,
          lastSeenAt: now
        )
      )
    }
    discoveredNodes = nodes.sorted {
      let comparison = $0.metadata.displayName.localizedStandardCompare($1.metadata.displayName)
      return comparison == .orderedSame
        ? $0.metadata.nodeID.uuidString < $1.metadata.nodeID.uuidString
        : comparison == .orderedAscending
    }
  }

}
