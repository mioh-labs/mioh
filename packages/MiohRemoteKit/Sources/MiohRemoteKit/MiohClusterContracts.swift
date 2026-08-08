import Foundation

// These types intentionally mirror packaging/macOS/standalone/
// RemoteClusterService.swift. Keep their Codable field and enum-case names
// stable: Mac and iPad use Swift's synthesized Codable representation on the
// wire.

public enum MiohClusterTransferMode: String, Codable, Hashable, Sendable {
  case sharedRootV1 = "shared-root-v1"
  case coordinatorHTTPV1 = "coordinator-http-v1"
}

public enum MiohClusterRole: String, Codable, Hashable, Sendable {
  case coordinator
  case worker
}

public struct MiohClusterCapabilities: Codable, Hashable, Sendable {
  public static let protocolVersion = 2

  public let protocolVersion: Int
  public let nodeID: UUID
  public let displayName: String
  public let role: MiohClusterRole
  public let transferMode: MiohClusterTransferMode
  public let sharedRootIdentifier: String
  public let architecture: String
  public let operatingSystem: String
  public let maximumConcurrentJobs: Int
  public let restorationModelIdentifiers: [String]
  public let detectorModelIdentifiers: [String]
  /// Optional v1 extensions. Older workers omit them; coordinators must then
  /// retain the original conservative behavior.
  public let maximumRestorationClipLength: Int?
  public let supportsROIEnhancer: Bool?
  public let supportsRestorationEffects: Bool?
  public let supportsFPSConversion: Bool?
  public let supportedInputExtensions: [String]?
  public let restorationAssetSHA256ByIdentifier: [String: String]?
  public let detectorAssetSHA256ByIdentifier: [String: String]?
  public let supportedTransferModes: [MiohClusterTransferMode]?

  public init(
    protocolVersion: Int = Self.protocolVersion,
    nodeID: UUID,
    displayName: String,
    role: MiohClusterRole = .worker,
    transferMode: MiohClusterTransferMode = .sharedRootV1,
    sharedRootIdentifier: String,
    architecture: String,
    operatingSystem: String,
    maximumConcurrentJobs: Int = 1,
    restorationModelIdentifiers: [String],
    detectorModelIdentifiers: [String],
    maximumRestorationClipLength: Int? = nil,
    supportsROIEnhancer: Bool? = nil,
    supportsRestorationEffects: Bool? = nil,
    supportsFPSConversion: Bool? = nil,
    supportedInputExtensions: [String]? = nil,
    restorationAssetSHA256ByIdentifier: [String: String]? = nil,
    detectorAssetSHA256ByIdentifier: [String: String]? = nil,
    supportedTransferModes: [MiohClusterTransferMode]? = nil
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
    self.maximumRestorationClipLength = maximumRestorationClipLength
    self.supportsROIEnhancer = supportsROIEnhancer
    self.supportsRestorationEffects = supportsRestorationEffects
    self.supportsFPSConversion = supportsFPSConversion
    self.supportedInputExtensions = supportedInputExtensions?.map {
      $0.lowercased()
    }.sorted()
    self.restorationAssetSHA256ByIdentifier = restorationAssetSHA256ByIdentifier
    self.detectorAssetSHA256ByIdentifier = detectorAssetSHA256ByIdentifier
    self.supportedTransferModes = supportedTransferModes.map {
      Array(Set($0)).sorted { $0.rawValue < $1.rawValue }
    }
  }
}

/// Short-lived coordinator-owned media endpoints. Each opaque capability URL
/// is bound to the attempt/lease/source by the coordinator and must never be
/// logged or exposed in Bonjour metadata.
public struct MiohClusterHTTPTransferDescriptor: Codable, Hashable, Sendable {
  public let inputURL: String
  public let outputURL: String
  public let expiresAt: Date
  public let maximumOutputBytes: Int64

  public init(
    inputURL: String,
    outputURL: String,
    expiresAt: Date,
    maximumOutputBytes: Int64
  ) {
    self.inputURL = inputURL
    self.outputURL = outputURL
    self.expiresAt = expiresAt
    self.maximumOutputBytes = maximumOutputBytes
  }

  /// Validates the immutable shape of the transfer capability. Use this after
  /// admission as the descriptor remains the original wire snapshot while the
  /// coordinator renews the live job lease and its server-side ticket.
  public func hasValidStructure() -> Bool {
    guard expiresAt.timeIntervalSince1970.isFinite,
      expiresAt.timeIntervalSince1970 > 0,
      maximumOutputBytes > 0,
      let input = URLComponents(string: inputURL),
      let output = URLComponents(string: outputURL),
      Self.isPlainHTTP(input),
      Self.isPlainHTTP(output),
      input.host?.lowercased() == output.host?.lowercased(),
      input.port == output.port,
      let inputEndpoint = Self.endpoint(in: input, role: .input),
      let outputEndpoint = Self.endpoint(in: output, role: .output),
      inputEndpoint.ticket == outputEndpoint.ticket
    else { return false }
    return true
  }

  /// New attempts must present both a valid capability shape and an unexpired
  /// initial ticket. Once admitted, the renewable job lease is authoritative.
  public func isValid(now: Date = Date()) -> Bool {
    hasValidStructure() && expiresAt > now
  }

  private enum EndpointRole {
    case input
    case output
  }

  private static func isPlainHTTP(_ components: URLComponents) -> Bool {
    guard components.scheme?.lowercased() == "http",
      let host = components.host, !host.isEmpty,
      components.user == nil, components.password == nil,
      components.query == nil, components.fragment == nil
    else { return false }
    return true
  }

  private static func endpoint(
    in components: URLComponents,
    role: EndpointRole
  ) -> (ticket: String, leaf: String)? {
    // Capability paths are deliberately ASCII-only. Reject percent-encoded
    // aliases so URL normalization cannot turn a different path into a valid
    // ticket endpoint after this check.
    let path = components.percentEncodedPath
    let prefix = "/mioh-cluster/v1/"
    guard !path.contains("%"), path.hasPrefix(prefix) else { return nil }
    let tail = path.dropFirst(prefix.count)
    guard let separator = tail.firstIndex(of: "/") else { return nil }
    let ticket = String(tail[..<separator])
    let leaf = String(tail[tail.index(after: separator)...])
    guard !ticket.isEmpty, ticket.count <= 128,
      ticket.unicodeScalars.allSatisfy({ scalar in
        CharacterSet.alphanumerics.contains(scalar) || scalar == "-" || scalar == "_"
      })
    else { return nil }

    switch role {
    case .input:
      guard ["input.mp4", "input.mov", "input.m4v"].contains(leaf) else { return nil }
    case .output:
      guard leaf == "output.mp4" else { return nil }
    }
    return (ticket, leaf)
  }
}

public struct MiohClusterMediaRange: Codable, Hashable, Sendable {
  public let decodeStartNanoseconds: Int64
  public let decodeEndNanoseconds: Int64
  public let coreStartNanoseconds: Int64
  public let coreEndNanoseconds: Int64
  public let leadingOverlapFrames: Int
  public let trailingOverlapFrames: Int

  public init(
    decodeStartNanoseconds: Int64,
    decodeEndNanoseconds: Int64,
    coreStartNanoseconds: Int64,
    coreEndNanoseconds: Int64,
    leadingOverlapFrames: Int,
    trailingOverlapFrames: Int
  ) {
    self.decodeStartNanoseconds = decodeStartNanoseconds
    self.decodeEndNanoseconds = decodeEndNanoseconds
    self.coreStartNanoseconds = coreStartNanoseconds
    self.coreEndNanoseconds = coreEndNanoseconds
    self.leadingOverlapFrames = leadingOverlapFrames
    self.trailingOverlapFrames = trailingOverlapFrames
  }

  public var isValid: Bool {
    decodeStartNanoseconds >= 0
      && decodeEndNanoseconds > decodeStartNanoseconds
      && coreStartNanoseconds >= decodeStartNanoseconds
      && coreEndNanoseconds > coreStartNanoseconds
      && coreEndNanoseconds <= decodeEndNanoseconds
      && leadingOverlapFrames >= 0 && trailingOverlapFrames >= 0
  }
}

public struct MiohClusterRestorationOptions: Codable, Hashable, Sendable {
  public let restorationModelIdentifier: String
  public let restorationAssetSHA256: String
  public let detectorModelIdentifier: String
  public let detectorAssetSHA256: String
  public let restorationClipLength: Int
  public let temporalOverlap: Int
  public let crossfade: Bool
  public let detectionEmptyLookahead: Int
  public let detectFaceMosaics: Bool
  public let blendFeather: Float
  public let sharpenStrength: Float
  public let detailBoost: Float
  public let textureMix: Float
  public let smoothStrength: Float
  public let effectUpscale: Int
  public let roiEnhancerModelIdentifier: String?
  public let roiEnhancerAssetSHA256: String?
  public let roiEnhancerStrength: Float
  public let roiEnhancerScale: Int
  public let videoCodec: String
  public let bitrateMultiplier: Double
  public let mp4FastStart: Bool
  public let targetFPSNumerator: Int?
  public let targetFPSDenominator: Int?

  public init(
    restorationModelIdentifier: String,
    restorationAssetSHA256: String,
    detectorModelIdentifier: String,
    detectorAssetSHA256: String,
    restorationClipLength: Int,
    temporalOverlap: Int,
    crossfade: Bool,
    detectionEmptyLookahead: Int,
    detectFaceMosaics: Bool,
    blendFeather: Float,
    sharpenStrength: Float,
    detailBoost: Float,
    textureMix: Float,
    smoothStrength: Float,
    effectUpscale: Int,
    roiEnhancerModelIdentifier: String? = nil,
    roiEnhancerAssetSHA256: String? = nil,
    roiEnhancerStrength: Float = 0,
    roiEnhancerScale: Int = 1,
    videoCodec: String,
    bitrateMultiplier: Double,
    mp4FastStart: Bool,
    targetFPSNumerator: Int? = nil,
    targetFPSDenominator: Int? = nil
  ) {
    self.restorationModelIdentifier = restorationModelIdentifier
    self.restorationAssetSHA256 = restorationAssetSHA256
    self.detectorModelIdentifier = detectorModelIdentifier
    self.detectorAssetSHA256 = detectorAssetSHA256
    self.restorationClipLength = restorationClipLength
    self.temporalOverlap = temporalOverlap
    self.crossfade = crossfade
    self.detectionEmptyLookahead = detectionEmptyLookahead
    self.detectFaceMosaics = detectFaceMosaics
    self.blendFeather = blendFeather
    self.sharpenStrength = sharpenStrength
    self.detailBoost = detailBoost
    self.textureMix = textureMix
    self.smoothStrength = smoothStrength
    self.effectUpscale = effectUpscale
    self.roiEnhancerModelIdentifier = roiEnhancerModelIdentifier
    self.roiEnhancerAssetSHA256 = roiEnhancerAssetSHA256
    self.roiEnhancerStrength = roiEnhancerStrength
    self.roiEnhancerScale = roiEnhancerScale
    self.videoCodec = videoCodec
    self.bitrateMultiplier = bitrateMultiplier
    self.mp4FastStart = mp4FastStart
    self.targetFPSNumerator = targetFPSNumerator
    self.targetFPSDenominator = targetFPSDenominator
  }

  public var isValid: Bool {
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
      && ((targetFPSNumerator == nil && targetFPSDenominator == nil)
        || (targetFPSNumerator.map { (1...120_000).contains($0) } == true
          && targetFPSDenominator.map { (1...1_001).contains($0) } == true))
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy {
      (48...57).contains($0) || (97...102).contains($0)
    }
  }
}

public enum MiohClusterPathError: Error, Equatable, Sendable {
  case empty
  case absolute
  case traversal
  case invalidCharacter
  case symbolicLink
  case outsideSharedRoot
}

public struct MiohClusterRelativePath: RawRepresentable, Codable, Hashable, Sendable {
  public let rawValue: String

  public init?(rawValue: String) {
    guard let value = try? Self.validate(rawValue) else { return nil }
    self.rawValue = value
  }

  public init(validating value: String) throws {
    rawValue = try Self.validate(value)
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    rawValue = try Self.validate(container.decode(String.self))
  }

  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(rawValue)
  }

  public func resolve(beneath sharedRoot: URL) throws -> URL {
    let root = sharedRoot.standardizedFileURL.resolvingSymlinksInPath()
    var cursor = root
    for component in rawValue.split(separator: "/").map(String.init) {
      cursor.appendPathComponent(component, isDirectory: false)
      if FileManager.default.fileExists(atPath: cursor.path) {
        let values = try cursor.resourceValues(forKeys: [.isSymbolicLinkKey])
        if values.isSymbolicLink == true { throw MiohClusterPathError.symbolicLink }
      }
      let candidate = cursor.standardizedFileURL
      guard Array(candidate.pathComponents.prefix(root.pathComponents.count))
        == root.pathComponents
      else { throw MiohClusterPathError.outsideSharedRoot }
      cursor = candidate
    }
    return cursor
  }

  private static func validate(_ value: String) throws -> String {
    guard !value.isEmpty else { throw MiohClusterPathError.empty }
    guard !value.hasPrefix("/"), !value.hasPrefix("~") else {
      throw MiohClusterPathError.absolute
    }
    guard !value.contains("\0"), !value.contains("\\") else {
      throw MiohClusterPathError.invalidCharacter
    }
    let components = value.split(separator: "/", omittingEmptySubsequences: false)
    guard components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
      throw MiohClusterPathError.traversal
    }
    return components.map(String.init).joined(separator: "/")
  }
}

public struct MiohClusterJobRequest: Codable, Hashable, Sendable {
  public static let protocolVersion = 2

  public let protocolVersion: Int
  public let jobID: UUID
  public let attemptID: UUID
  public let leaseID: UUID
  public let coordinatorNodeID: UUID
  public let sharedRootIdentifier: String
  public let inputByteCount: Int64
  public let inputSHA256: String
  public let inputRelativePath: MiohClusterRelativePath
  public let outputRelativePath: MiohClusterRelativePath
  public let mediaRange: MiohClusterMediaRange
  public let options: MiohClusterRestorationOptions
  public let createdAt: Date
  public let leaseExpiresAt: Date
  public let httpTransfer: MiohClusterHTTPTransferDescriptor?

  public init(
    protocolVersion: Int = Self.protocolVersion,
    jobID: UUID,
    attemptID: UUID,
    leaseID: UUID,
    coordinatorNodeID: UUID,
    sharedRootIdentifier: String,
    inputByteCount: Int64,
    inputSHA256: String,
    inputRelativePath: MiohClusterRelativePath,
    outputRelativePath: MiohClusterRelativePath,
    mediaRange: MiohClusterMediaRange,
    options: MiohClusterRestorationOptions,
    createdAt: Date,
    leaseExpiresAt: Date,
    httpTransfer: MiohClusterHTTPTransferDescriptor? = nil
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
}

public enum MiohClusterAttemptState: String, Codable, Hashable, Sendable {
  case accepted, running, completed, failed, cancelled, expired

  public var isTerminal: Bool {
    switch self {
    case .completed, .failed, .cancelled, .expired: true
    case .accepted, .running: false
    }
  }
}

public struct MiohClusterJobMetrics: Codable, Hashable, Sendable {
  public let processedFrames: Int
  public let wallSeconds: Double
  public let outputByteCount: Int64

  public init(processedFrames: Int, wallSeconds: Double, outputByteCount: Int64) {
    self.processedFrames = processedFrames
    self.wallSeconds = wallSeconds
    self.outputByteCount = outputByteCount
  }
}

public struct MiohClusterAttemptRecord: Codable, Hashable, Identifiable, Sendable {
  public var id: UUID { attemptID }
  public let jobID: UUID
  public let attemptID: UUID
  public let leaseID: UUID
  public let coordinatorNodeID: UUID
  public let inputRelativePath: MiohClusterRelativePath
  public let outputRelativePath: MiohClusterRelativePath
  public var state: MiohClusterAttemptState
  public var leaseExpiresAt: Date
  public var updatedAt: Date
  public var metrics: MiohClusterJobMetrics?
  public var failureCode: String?

  public init(
    jobID: UUID,
    attemptID: UUID,
    leaseID: UUID,
    coordinatorNodeID: UUID,
    inputRelativePath: MiohClusterRelativePath,
    outputRelativePath: MiohClusterRelativePath,
    state: MiohClusterAttemptState,
    leaseExpiresAt: Date,
    updatedAt: Date,
    metrics: MiohClusterJobMetrics? = nil,
    failureCode: String? = nil
  ) {
    self.jobID = jobID
    self.attemptID = attemptID
    self.leaseID = leaseID
    self.coordinatorNodeID = coordinatorNodeID
    self.inputRelativePath = inputRelativePath
    self.outputRelativePath = outputRelativePath
    self.state = state
    self.leaseExpiresAt = leaseExpiresAt
    self.updatedAt = updatedAt
    self.metrics = metrics
    self.failureCode = failureCode
  }
}

public enum MiohClusterAdmissionDisposition: String, Codable, Hashable, Sendable {
  case accepted, duplicate, conflict, rejected
}

public struct MiohClusterJobAdmission: Codable, Hashable, Sendable {
  public let disposition: MiohClusterAdmissionDisposition
  public let record: MiohClusterAttemptRecord?
  public let reason: String?

  public init(
    disposition: MiohClusterAdmissionDisposition,
    record: MiohClusterAttemptRecord?,
    reason: String?
  ) {
    self.disposition = disposition
    self.record = record
    self.reason = reason
  }
}

public struct MiohClusterNodeStatus: Codable, Hashable, Sendable {
  public let protocolVersion: Int
  public let nodeID: UUID
  public let observedAt: Date
  public let capabilities: MiohClusterCapabilities
  public let activeAttempts: [MiohClusterAttemptRecord]
  public let acceptingJobs: Bool

  public init(
    protocolVersion: Int = MiohClusterCapabilities.protocolVersion,
    nodeID: UUID,
    observedAt: Date,
    capabilities: MiohClusterCapabilities,
    activeAttempts: [MiohClusterAttemptRecord],
    acceptingJobs: Bool
  ) {
    self.protocolVersion = protocolVersion
    self.nodeID = nodeID
    self.observedAt = observedAt
    self.capabilities = capabilities
    self.activeAttempts = activeAttempts
    self.acceptingJobs = acceptingJobs
  }
}

public struct MiohClusterStatusQuery: Codable, Hashable, Sendable {
  public let attemptID: UUID?
  public init(attemptID: UUID?) { self.attemptID = attemptID }
}

public struct MiohClusterCancelRequest: Codable, Hashable, Sendable {
  public let attemptID: UUID
  public let leaseID: UUID
  public init(attemptID: UUID, leaseID: UUID) {
    self.attemptID = attemptID
    self.leaseID = leaseID
  }
}

public struct MiohClusterRenewLeaseRequest: Codable, Hashable, Sendable {
  public let attemptID: UUID
  public let leaseID: UUID
  public let newExpiration: Date
  public init(attemptID: UUID, leaseID: UUID, newExpiration: Date) {
    self.attemptID = attemptID
    self.leaseID = leaseID
    self.newExpiration = newExpiration
  }
}

public enum MiohClusterRPCAction: Codable, Hashable, Sendable {
  case capabilities
  case submit(MiohClusterJobRequest)
  case status(MiohClusterStatusQuery)
  case renewLease(MiohClusterRenewLeaseRequest)
  case cancel(MiohClusterCancelRequest)
}

public struct MiohClusterRPCRequest: Codable, Hashable, Sendable {
  public let protocolVersion: Int
  public let requestID: UUID
  public let coordinatorNodeID: UUID
  public let action: MiohClusterRPCAction

  public init(
    protocolVersion: Int = MiohClusterCapabilities.protocolVersion,
    requestID: UUID,
    coordinatorNodeID: UUID,
    action: MiohClusterRPCAction
  ) {
    self.protocolVersion = protocolVersion
    self.requestID = requestID
    self.coordinatorNodeID = coordinatorNodeID
    self.action = action
  }
}

public struct MiohClusterRPCResponse: Codable, Hashable, Sendable {
  public let protocolVersion: Int
  public let requestID: UUID
  public let ok: Bool
  public let capabilities: MiohClusterCapabilities?
  public let admission: MiohClusterJobAdmission?
  public let nodeStatus: MiohClusterNodeStatus?
  public let attempt: MiohClusterAttemptRecord?
  public let errorCode: String?

  public init(
    protocolVersion: Int = MiohClusterCapabilities.protocolVersion,
    requestID: UUID,
    ok: Bool,
    capabilities: MiohClusterCapabilities?,
    admission: MiohClusterJobAdmission?,
    nodeStatus: MiohClusterNodeStatus?,
    attempt: MiohClusterAttemptRecord?,
    errorCode: String?
  ) {
    self.protocolVersion = protocolVersion
    self.requestID = requestID
    self.ok = ok
    self.capabilities = capabilities
    self.admission = admission
    self.nodeStatus = nodeStatus
    self.attempt = attempt
    self.errorCode = errorCode
  }

  public static func error(requestID: UUID, _ code: String) -> Self {
    Self(
      protocolVersion: MiohClusterCapabilities.protocolVersion,
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

public typealias MiohClusterJobLauncher = @Sendable (
  _ request: MiohClusterJobRequest,
  _ inputURL: URL,
  _ outputURL: URL
) async throws -> MiohClusterJobMetrics
