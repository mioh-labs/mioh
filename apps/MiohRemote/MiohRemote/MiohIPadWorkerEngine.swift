import AVFoundation
import Accelerate
import CoreImage
import CoreML
import CoreVideo
import CryptoKit
import Foundation
import MiohRemoteKit
import UIKit

#if canImport(CoreAI)
  import CoreAI
#endif

enum MiohIPadWorkerEngineError: LocalizedError {
  case requiresIPadOS27
  case missingModelRoot
  case unsupportedJob(String)
  case unsupportedMedia(String)
  case decoder(String)
  case detector(String)
  case restorer(String)
  case output(String)
  case internalFailure(String)

  var errorDescription: String? {
    switch self {
    case .requiresIPadOS27:
      "実行Workerには実機のiPadとiPadOS 27以降が必要です。"
    case .missingModelRoot:
      "同梱Core AIモデルと識別マニフェストが見つかりません。"
    case .unsupportedJob(let detail):
      "このiPad Workerが対応していないジョブです: \(detail)"
    case .unsupportedMedia(let detail):
      "この動画をiPad Workerで処理できません: \(detail)"
    case .decoder(let detail):
      "AVFoundationの動画デコードに失敗しました: \(detail)"
    case .detector(let detail):
      "v4-fast検出に失敗しました: \(detail)"
    case .restorer(let detail):
      "BasicVSR++復元に失敗しました: \(detail)"
    case .output(let detail):
      "MP4 shardを作成できません: \(detail)"
    case .internalFailure(let detail):
      "iPad処理に失敗しました: \(detail)"
    }
  }
}

/// Keeps local/shared-root and HTTP Range transport separate from the
/// fixed-T18 decode/inference core.
private protocol MiohIPadVideoInputSource: Sendable {
  var mediaPathExtension: String { get }
  var securityScopedURL: URL? { get }
  func makeAsset(attemptID: UUID) async throws -> MiohIPadVideoAssetHandle
}

/// Keeps any custom resource loader and its URLSession alive for the entire
/// AVAssetReader lifetime. Cancellation is idempotent and is also invoked from
/// the job task's cancellation handler so lease expiry stops active Range I/O.
private final class MiohIPadVideoAssetHandle: @unchecked Sendable {
  let asset: AVURLAsset
  private let retainedOwner: AnyObject?
  private let cancelAction: @Sendable () -> Void

  init(
    asset: AVURLAsset,
    retainedOwner: AnyObject? = nil,
    cancelAction: @escaping @Sendable () -> Void = {}
  ) {
    self.asset = asset
    self.retainedOwner = retainedOwner
    self.cancelAction = cancelAction
  }

  func cancel() { cancelAction() }
}

private actor MiohIPadHTTPInputCache {
  static let shared = MiohIPadHTTPInputCache()
  private static let transferTimeout: TimeInterval = 30 * 60

  func localFile(
    for url: URL,
    mediaPathExtension: String,
    expectedByteCount: Int64,
    expectedSHA256: String
  ) async throws -> URL {
    let sha = expectedSHA256.lowercased()
    let cacheRoot = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-ipad-worker-input-cache", isDirectory: true)
    try FileManager.default.createDirectory(
      at: cacheRoot,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    let localURL = cacheRoot.appendingPathComponent(
      "\(sha).\(mediaPathExtension)"
    )
    if let existingBytes = try? byteCount(of: localURL),
      existingBytes == expectedByteCount
    {
      return localURL
    }

    let partialURL = cacheRoot.appendingPathComponent(
      ".\(sha).\(UUID().uuidString.lowercased()).part"
    )
    try? FileManager.default.removeItem(at: partialURL)
    FileManager.default.createFile(atPath: partialURL.path, contents: nil)
    let handle = try FileHandle(forWritingTo: partialURL)
    var hasher = SHA256()
    var offset: Int64 = 0
    let chunkBytes: Int64 = 1 * 1_024 * 1_024
    let configuration = URLSessionConfiguration.ephemeral
    configuration.timeoutIntervalForRequest = Self.transferTimeout
    configuration.timeoutIntervalForResource = 24 * 60 * 60
    configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    configuration.urlCache = nil
    configuration.httpCookieStorage = nil
    let session = URLSession(configuration: configuration)
    defer {
      try? handle.close()
      session.invalidateAndCancel()
    }
    do {
      while offset < expectedByteCount {
        try Task.checkCancellation()
        let count = min(chunkBytes, expectedByteCount - offset)
        let start = offset
        let end = offset + count - 1
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = Self.transferTimeout
        request.setValue("bytes=\(start)-\(end)", forHTTPHeaderField: "Range")
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let (data, response) = try await session.data(for: request)
        let expectedContentRange = "bytes \(start)-\(end)/\(expectedByteCount)"
        guard let http = response as? HTTPURLResponse,
          http.statusCode == 206,
          http.url?.absoluteString == url.absoluteString,
          http.value(forHTTPHeaderField: "Content-Range") == expectedContentRange,
          http.value(forHTTPHeaderField: "Content-Length") == String(count),
          http.value(forHTTPHeaderField: "ETag") == "\"\(sha)\"",
          data.count == Int(count)
        else {
          throw MiohIPadWorkerEngineError.decoder("HTTP入力のRange応答が不正です")
        }
        try handle.write(contentsOf: data)
        hasher.update(data: data)
        offset += count
      }
      try handle.close()
      let actualSHA256 = hasher.finalize().map {
        String(format: "%02x", $0)
      }.joined()
      guard actualSHA256 == sha else {
        throw MiohIPadWorkerEngineError.decoder("HTTP入力のSHA-256が一致しません")
      }
      guard (try? byteCount(of: partialURL)) == expectedByteCount else {
        throw MiohIPadWorkerEngineError.decoder("HTTP入力のサイズが一致しません")
      }
      try? FileManager.default.removeItem(at: localURL)
      try FileManager.default.moveItem(at: partialURL, to: localURL)
      return localURL
    } catch {
      try? FileManager.default.removeItem(at: partialURL)
      throw error
    }
  }

  private func byteCount(of url: URL) throws -> Int64 {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    return (attributes[.size] as? NSNumber)?.int64Value ?? -1
  }
}

/// The execution core always publishes a completed MP4 into the local
/// candidate supplied by the worker ledger. The ledger alone owns any later
/// HTTP upload or shared-root final publication.
private protocol MiohIPadVideoOutputSink: Sendable {
  var mediaPathExtension: String { get }
  var securityScopedURL: URL? { get }
  func makeLocalStagingURL(attemptID: UUID) throws -> URL
  func publish(localFile: URL, byteCount: Int64) async throws
}

private struct MiohIPadSharedRootInputSource: MiohIPadVideoInputSource {
  let url: URL
  let mediaPathExtension: String

  var securityScopedURL: URL? { url }

  func makeAsset(attemptID _: UUID) async throws -> MiohIPadVideoAssetHandle {
    if mediaPathExtension == "ts" {
      if #available(iOS 17.0, *) {
        return MiohIPadVideoAssetHandle(
          asset: AVURLAsset(
            url: url,
            options: [AVURLAssetOverrideMIMETypeKey: "video/mp2t"]
          )
        )
      }
    }
    return MiohIPadVideoAssetHandle(asset: AVURLAsset(url: url))
  }
}

/// Bridges the app-owned loopback Range endpoint into a custom-scheme asset.
/// AVAssetReader rejects ordinary HTTP AVURLAssets even when AVPlayer accepts
/// them, so each restoration attempt must retain its own resource loader.
private struct MiohIPadLoopbackRangeInputSource: MiohIPadVideoInputSource {
  let url: URL
  let mediaPathExtension: String
  let expectedByteCount: Int64
  let rangeValidator: String

  var securityScopedURL: URL? { nil }

  func makeAsset(attemptID _: UUID) async throws -> MiohIPadVideoAssetHandle {
    // The validator is an attempt-opaque 256-bit entity tag. The SFTP session
    // separately checks the open handle's size and modification time per page.
    let ranged = try MiohHTTPRangeAsset(
      remoteURL: url,
      expectedByteCount: expectedByteCount,
      expectedSHA256: rangeValidator,
      pageBytes: 256 * 1_024
    )
    return MiohIPadVideoAssetHandle(
      asset: ranged.asset,
      retainedOwner: ranged,
      cancelAction: { ranged.cancel() }
    )
  }
}

private enum MiohIPadHTTPRangeProbeError: Error {
  case timeout
}

private struct MiohIPadCoordinatorHTTPInputSource: MiohIPadVideoInputSource {
  let url: URL
  let mediaPathExtension: String
  let expectedByteCount: Int64
  let expectedSHA256: String

  var securityScopedURL: URL? { nil }

  func makeAsset(attemptID: UUID) async throws -> MiohIPadVideoAssetHandle {
    _ = attemptID
    let ranged = try MiohHTTPRangeAsset(
      remoteURL: url,
      expectedByteCount: expectedByteCount,
      expectedSHA256: expectedSHA256
    )
    if await Self.probeVideoTracks(ranged.asset) {
      return MiohIPadVideoAssetHandle(
        asset: ranged.asset,
        retainedOwner: ranged,
        cancelAction: { ranged.cancel() }
      )
    }
    ranged.cancel()
    let localURL = try await MiohIPadHTTPInputCache.shared.localFile(
      for: url,
      mediaPathExtension: mediaPathExtension,
      expectedByteCount: expectedByteCount,
      expectedSHA256: expectedSHA256
    )
    return MiohIPadVideoAssetHandle(asset: AVURLAsset(url: localURL))
  }

  private static func probeVideoTracks(_ asset: AVURLAsset) async -> Bool {
    do {
      let tracks = try await withThrowingTaskGroup(
        of: [AVAssetTrack].self
      ) { group -> [AVAssetTrack] in
        group.addTask {
          try await asset.loadTracks(withMediaType: .video)
        }
        group.addTask {
          try await Task.sleep(nanoseconds: 8_000_000_000)
          throw MiohIPadHTTPRangeProbeError.timeout
        }
        guard let result = try await group.next() else {
          throw MiohIPadHTTPRangeProbeError.timeout
        }
        group.cancelAll()
        return result
      }
      return !tracks.isEmpty
    } catch {
      return false
    }
  }
}

private struct MiohIPadLocalOutputSink: MiohIPadVideoOutputSink {
  let targetURL: URL
  let mediaPathExtension: String
  let needsSecurityScope: Bool

  var securityScopedURL: URL? {
    needsSecurityScope ? targetURL.deletingLastPathComponent() : nil
  }

  func makeLocalStagingURL(attemptID: UUID) throws -> URL {
    let parent = targetURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(
      at: parent,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    return parent.appendingPathComponent(
      ".mioh-ipad-\(attemptID.uuidString.lowercased()).mp4"
    )
  }

  func publish(localFile: URL, byteCount _: Int64) async throws {
    try Task.checkCancellation()
    guard !FileManager.default.fileExists(atPath: targetURL.path) else {
      throw MiohIPadWorkerEngineError.output("staging output already exists")
    }
    try FileManager.default.moveItem(at: localFile, to: targetURL)
  }
}

private struct MiohIPadJobIO: Sendable {
  let input: any MiohIPadVideoInputSource
  let output: any MiohIPadVideoOutputSink

  static func resolve(
    request: MiohClusterJobRequest,
    inputURL: URL,
    outputURL: URL
  ) throws -> Self {
    let inputExtension = URL(
      fileURLWithPath: request.inputRelativePath.rawValue
    ).pathExtension.lowercased()
    let outputExtension = URL(
      fileURLWithPath: request.outputRelativePath.rawValue
    ).pathExtension.lowercased()
    if let transfer = request.httpTransfer {
      guard transfer.hasValidStructure(),
        let descriptorInputURL = URL(string: transfer.inputURL),
        descriptorInputURL == inputURL
      else {
        throw MiohIPadWorkerEngineError.unsupportedJob(
          "invalid coordinator HTTP transfer descriptor"
        )
      }
      return Self(
        input: MiohIPadCoordinatorHTTPInputSource(
          url: inputURL,
          mediaPathExtension: inputExtension,
          expectedByteCount: request.inputByteCount,
          expectedSHA256: request.inputSHA256
        ),
        output: MiohIPadLocalOutputSink(
          targetURL: outputURL,
          mediaPathExtension: outputExtension,
          needsSecurityScope: false
        )
      )
    }
    if isAppOwnedLoopbackRangeInput(
      request: request,
      inputURL: inputURL,
      inputExtension: inputExtension
    ) {
      return Self(
        input: MiohIPadLoopbackRangeInputSource(
          url: inputURL,
          mediaPathExtension: inputExtension,
          expectedByteCount: request.inputByteCount,
          rangeValidator: request.inputSHA256
        ),
        output: MiohIPadLocalOutputSink(
          targetURL: outputURL,
          mediaPathExtension: outputExtension,
          needsSecurityScope: false
        )
      )
    }
    guard
      inputURL.standardizedFileURL
        != outputURL.standardizedFileURL
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "input and output paths must differ"
      )
    }
    return Self(
      input: MiohIPadSharedRootInputSource(
        url: inputURL,
        mediaPathExtension: inputExtension
      ),
      output: MiohIPadLocalOutputSink(
        targetURL: outputURL,
        mediaPathExtension: outputExtension,
        needsSecurityScope: true
      )
    )
  }

  private static func isAppOwnedLoopbackRangeInput(
    request: MiohClusterJobRequest,
    inputURL: URL,
    inputExtension: String
  ) -> Bool {
    let validator = request.inputSHA256.lowercased()
    let pathComponents = inputURL.path.split(separator: "/")
    guard
      request.sharedRootIdentifier
        == IPadRealtimePreviewConfiguration.sharedRootIdentifier,
      inputURL.scheme?.lowercased() == "http",
      inputURL.host == "127.0.0.1",
      inputURL.port.map({ (1...65_535).contains($0) }) == true,
      inputURL.user == nil, inputURL.password == nil,
      inputURL.query == nil, inputURL.fragment == nil,
      request.inputByteCount > 0,
      request.inputByteCount <= 20 * 1_024 * 1_024 * 1_024,
      validator != String(repeating: "0", count: 64),
      validator.utf8.count == 64,
      validator.utf8.allSatisfy({
        (48...57).contains($0) || (97...102).contains($0)
      }),
      pathComponents.count == 3,
      pathComponents[0] == "v1",
      pathComponents[1].utf8.count == 32,
      pathComponents[1].utf8.allSatisfy({
        (48...57).contains($0) || (97...102).contains($0)
      }),
      pathComponents[2] == "input.\(inputExtension)",
      ["mp4", "mov", "m4v"].contains(inputExtension)
    else { return false }
    return true
  }
}

struct MiohIPadPreviewFrame: @unchecked Sendable {
  let image: CGImage
  let ptsNanoseconds: Int64
}

struct MiohIPadRealtimeInputFrame: @unchecked Sendable {
  let pixelBuffer: CVPixelBuffer
  let ptsNanoseconds: Int64
}

struct MiohIPadRealtimeOutputFrame: @unchecked Sendable {
  let pixelBuffer: CVPixelBuffer
  let ptsNanoseconds: Int64
}

struct MiohIPadRealtimePerformanceSample: Sendable {
  let detectionSeconds: Double
  let restorationSeconds: Double
  let restorationPreparationSeconds: Double
  let restorationCompositingSeconds: Double
  let mediaSeconds: Double
  let processedFrames: Int
  let restoredFrames: Int
  let modelRestorationSeconds: Double

  var processingSeconds: Double {
    detectionSeconds + restorationSeconds
  }
}

protocol MiohIPadRealtimeFrameSessioning: Sendable {
  func setDetectionMaskReuseSkipFrames(_ count: Int) async
  func setMaximumFrameRate(_ wholeFPS: Int?) async
  func setEmergency24FPSEnabled(_ enabled: Bool) async
  func append(_ frame: MiohIPadRealtimeInputFrame) async throws
    -> [MiohIPadRealtimeOutputFrame]
  func flush() async throws -> [MiohIPadRealtimeOutputFrame]
  func takePerformanceSamples() async -> [MiohIPadRealtimePerformanceSample]
}

struct MiohIPadPreparedWorker: Sendable {
  typealias LocalExecutor =
    @Sendable (
      MiohClusterJobRequest,
      URL,
      URL,
      @escaping @Sendable (Double) -> Void,
      (@Sendable (MiohIPadPreviewFrame) async -> Void)?
    ) async throws -> MiohClusterJobMetrics

  let restorationAssetSHA256ByIdentifier: [String: String]
  let detectorAssetSHA256ByIdentifier: [String: String]
  let maximumRestorationClipLength: Int
  private let preparedLauncher: MiohClusterJobLauncher
  private let preparedLocalExecutors: [LocalExecutor]
  private let preparedRealtimeSessionFactory:
    @Sendable (
      MiohClusterRestorationOptions,
      Int,
      Int
    ) async throws -> any MiohIPadRealtimeFrameSessioning
  private let releaseSupplementalExecutors: @Sendable () async -> Void

  init(
    restorationAssetSHA256ByIdentifier: [String: String],
    detectorAssetSHA256ByIdentifier: [String: String],
    maximumRestorationClipLength: Int,
    launcher: @escaping MiohClusterJobLauncher,
    localExecutors: [LocalExecutor],
    realtimeSessionFactory:
      @escaping @Sendable (
        MiohClusterRestorationOptions,
        Int,
        Int
      ) async throws -> any MiohIPadRealtimeFrameSessioning,
    releaseSupplementalExecutors:
      @escaping @Sendable () async -> Void
  ) {
    self.restorationAssetSHA256ByIdentifier =
      restorationAssetSHA256ByIdentifier
    self.detectorAssetSHA256ByIdentifier = detectorAssetSHA256ByIdentifier
    self.maximumRestorationClipLength = maximumRestorationClipLength
    preparedLauncher = launcher
    preparedLocalExecutors = localExecutors
    preparedRealtimeSessionFactory = realtimeSessionFactory
    self.releaseSupplementalExecutors = releaseSupplementalExecutors
  }

  var restorationModelIdentifiers: [String] {
    restorationAssetSHA256ByIdentifier.keys.sorted()
  }

  var detectorModelIdentifiers: [String] {
    detectorAssetSHA256ByIdentifier.keys.sorted()
  }

  var launcher: MiohClusterJobLauncher { preparedLauncher }

  var maximumParallelRestorationLanes: Int {
    max(1, preparedLocalExecutors.count)
  }

  func executeLocal(
    request: MiohClusterJobRequest,
    inputURL: URL,
    outputURL: URL,
    lane: Int = 0,
    progress: @escaping @Sendable (Double) -> Void,
    preview: (@Sendable (MiohIPadPreviewFrame) async -> Void)? = nil
  ) async throws -> MiohClusterJobMetrics {
    do {
      guard preparedLocalExecutors.indices.contains(lane) else {
        throw MiohIPadWorkerEngineError.unsupportedJob(
          "復元lane \(lane + 1)は準備されていません"
        )
      }
      return try await preparedLocalExecutors[lane](
        request,
        inputURL,
        outputURL,
        progress,
        preview
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch let error as MiohIPadWorkerEngineError {
      throw error
    } catch {
      let value = error as NSError
      throw MiohIPadWorkerEngineError.internalFailure(
        "Worker実行中の未分類エラー: \(error.localizedDescription) "
          + "[\(value.domain):\(value.code)]"
      )
    }
  }

  func releaseSupplementalRestorationLanes() async {
    await releaseSupplementalExecutors()
  }

  func makeRealtimeFrameSession(
    options: MiohClusterRestorationOptions,
    width: Int,
    height: Int
  ) async throws -> any MiohIPadRealtimeFrameSessioning {
    try await preparedRealtimeSessionFactory(options, width, height)
  }

  @MainActor
  func makeCapabilities(
    nodeID: UUID,
    displayName: String,
    sharedRootIdentifier: String?
  ) -> MiohClusterCapabilities {
    let rootIdentifier =
      sharedRootIdentifier?
      .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let transferModes: [MiohClusterTransferMode] =
      rootIdentifier.isEmpty
      ? [.coordinatorHTTPV1]
      : [.coordinatorHTTPV1, .sharedRootV1]
    return MiohClusterCapabilities(
      nodeID: nodeID,
      displayName: displayName,
      role: .worker,
      transferMode: .coordinatorHTTPV1,
      sharedRootIdentifier: rootIdentifier,
      architecture: Self.architecture,
      operatingSystem: "iPadOS \(UIDevice.current.systemVersion)",
      maximumConcurrentJobs: 1,
      restorationModelIdentifiers: restorationModelIdentifiers,
      detectorModelIdentifiers: detectorModelIdentifiers,
      maximumRestorationClipLength: maximumRestorationClipLength,
      supportsROIEnhancer: false,
      supportsRestorationEffects: false,
      supportsFPSConversion: true,
      supportedInputExtensions: ["mp4", "mov", "m4v"],
      restorationAssetSHA256ByIdentifier: restorationAssetSHA256ByIdentifier,
      detectorAssetSHA256ByIdentifier: detectorAssetSHA256ByIdentifier,
      supportedTransferModes: transferModes
    )
  }

  private static var architecture: String {
    #if arch(arm64)
      "arm64"
    #elseif arch(x86_64)
      "x86_64"
    #else
      "unknown"
    #endif
  }
}

enum MiohIPadWorkerEngine {
  private static let variableRestorationModelIdentifier =
    "basicvsrpp-v1.2-coreai-variable"
  private struct RestorationSpec {
    let identifier: String
    let frameCount: Int
  }

  private struct DetectorSpec {
    let identifier: String
    let candidateChannels: Int
  }

  private static let restorationSpecs = [
    RestorationSpec(identifier: "basicvsrpp-v1.2-coreai", frameCount: 18),
    RestorationSpec(identifier: "basicvsrpp-v1.2-coreai-t36", frameCount: 36),
    RestorationSpec(identifier: "basicvsrpp-v1.2-coreai-t90", frameCount: 90),
  ]
  private static let detectorSpecs = [
    DetectorSpec(identifier: "v2-coreai", candidateChannels: 37),
    DetectorSpec(identifier: "v2-coreml", candidateChannels: 37),
    DetectorSpec(identifier: "v3.1-fast-coreai", candidateChannels: 38),
    DetectorSpec(identifier: "v3.1-fast-coreml", candidateChannels: 38),
    DetectorSpec(identifier: "v3.1-accurate-coreai", candidateChannels: 38),
    DetectorSpec(identifier: "v3.1-accurate-coreml", candidateChannels: 38),
    DetectorSpec(identifier: "v4-fast-coreai", candidateChannels: 38),
    DetectorSpec(identifier: "v4-fast-coreml", candidateChannels: 38),
    DetectorSpec(identifier: "v4-accurate-coreai", candidateChannels: 38),
    DetectorSpec(identifier: "v4-accurate-coreml", candidateChannels: 38),
    DetectorSpec(identifier: "vr-v2-accurate-coreai", candidateChannels: 38),
    DetectorSpec(identifier: "vr-v2-accurate-coreml", candidateChannels: 38),
  ]

  /// Resolves `models/` when the app bundles it as a folder reference, while
  /// also accepting a flat resource root for development builds.
  static func prepare(bundle: Bundle = .main) async throws
    -> MiohIPadPreparedWorker
  {
    guard let resources = bundle.resourceURL else {
      throw MiohIPadWorkerEngineError.missingModelRoot
    }
    let nested = resources.appendingPathComponent("models", isDirectory: true)
    let nestedManifest = nested.appendingPathComponent(
      MiohPortableModelIdentityManifest.fileName
    )
    let modelRoot =
      FileManager.default.fileExists(atPath: nestedManifest.path)
      ? nested : resources
    return try await prepare(modelRoot: modelRoot)
  }

  /// Injectable model root keeps asset verification testable. The returned
  /// Startup deliberately performs digest-only validation. Core AI runtimes
  /// are loaded one selected pair at a time when a job starts; instantiating
  /// every large graph here can abort inside Metal before Swift can catch it.
  static func prepare(modelRoot: URL) async throws -> MiohIPadPreparedWorker {
    let isPad = await MainActor.run {
      UIDevice.current.userInterfaceIdiom == .pad
        && !ProcessInfo.processInfo.isiOSAppOnMac
    }
    guard isPad else { throw MiohIPadWorkerEngineError.requiresIPadOS27 }
    #if targetEnvironment(simulator)
      throw MiohIPadWorkerEngineError.requiresIPadOS27
    #elseif canImport(CoreAI)
      guard #available(iOS 27.0, *) else {
        throw MiohIPadWorkerEngineError.requiresIPadOS27
      }
      let manifest = try MiohPortableModelIdentityManifest.load(from: modelRoot)
      var restorationIdentities: [String: MiohValidatedPortableModelIdentity] = [:]
      var detectorIdentities: [String: MiohValidatedPortableModelIdentity] = [:]
      var restorationDigests: [String: String] = [:]
      var detectorDigests: [String: String] = [:]
      for spec in restorationSpecs {
        let identity = try manifest.validateCoreAIModelDigest(
          identifier: spec.identifier,
          beneath: modelRoot
        )
        restorationIdentities[spec.identifier] = identity
        restorationDigests[spec.identifier] = identity.canonicalSHA256
      }
      let variableIdentity = try manifest.validateCoreAIModelCollectionDigest(
        identifier: variableRestorationModelIdentifier,
        beneath: modelRoot
      )
      for spec in detectorSpecs {
        let identity = try manifest.validateCoreAIModelDigest(
          identifier: spec.identifier,
          beneath: modelRoot
        )
        detectorIdentities[spec.identifier] = identity
        detectorDigests[spec.identifier] = identity.canonicalSHA256
      }
      restorationDigests[variableIdentity.identifier] =
        variableIdentity.canonicalSHA256
      let restorationFrameCounts = Dictionary(
        uniqueKeysWithValues: restorationSpecs.map {
          ($0.identifier, $0.frameCount)
        }
      )
      let detectorCandidateChannels = Dictionary(
        uniqueKeysWithValues: detectorSpecs.map {
          ($0.identifier, $0.candidateChannels)
        }
      )
      let sharedRestorerCache = MiohIPadSharedRestorerCache(
        fixedRestorationIdentities: restorationIdentities,
        variableRestorationIdentity: variableIdentity,
        restorationFrameCounts: restorationFrameCounts
      )
      func makeCore() -> MiohIPadExecutionCore {
        MiohIPadExecutionCore(
          sharedRestorerCache: sharedRestorerCache,
          detectorIdentities: detectorIdentities,
          restorationFrameCounts: restorationFrameCounts,
          detectorCandidateChannels: detectorCandidateChannels,
          restorationDigests: restorationDigests,
          detectorDigests: detectorDigests
        )
      }
      // Decode, detection and output remain independent across three cores.
      // BasicVSR++ itself uses the shared cache above because three T90
      // workspaces exceed the measured iPad process memory budget.
      let cores = (0..<3).map { _ in makeCore() }
      let primaryCore = cores[0]
      return MiohIPadPreparedWorker(
        restorationAssetSHA256ByIdentifier: restorationDigests,
        detectorAssetSHA256ByIdentifier: detectorDigests,
        maximumRestorationClipLength: restorationSpecs.map(\.frameCount).max() ?? 0,
        launcher: { request, inputURL, outputURL in
          try await primaryCore.execute(
            request,
            inputURL: inputURL,
            outputURL: outputURL
          )
        },
        localExecutors: cores.map { core in
          { request, inputURL, outputURL, progress, preview in
            try await core.execute(
              request,
              inputURL: inputURL,
              outputURL: outputURL,
              progress: progress,
              preview: preview
            )
          }
        },
        realtimeSessionFactory: { options, width, height in
          try await primaryCore.makeRealtimeFrameSession(
            options: options,
            width: width,
            height: height
          )
        },
        releaseSupplementalExecutors: {
          for core in cores.dropFirst() {
            await core.releaseCachedModels()
          }
        }
      )
    #else
      throw MiohIPadWorkerEngineError.requiresIPadOS27
    #endif
  }

  #if canImport(CoreAI)
    @available(iOS 27.0, *)
    fileprivate static func validateRestorationContract(
      _ function: InferenceFunction,
      frameCount: Int
    ) throws {
      let shape = [1, frameCount, 3, 256, 256]
      guard
        case .ndArray(let input)? = function.descriptor.inputDescriptor(
          of: "frames"
        ),
        case .ndArray(let output)? = function.descriptor.outputDescriptor(
          of: "restored"
        ), input.shape == shape, output.shape == shape,
        input.scalarType == .float16, output.scalarType == .float16
      else {
        throw MiohIPadWorkerEngineError.restorer(
          "T\(frameCount) FP16 tensor contract does not match"
        )
      }
    }

    @available(iOS 27.0, *)
    fileprivate static func validateDetectorContract(
      _ function: InferenceFunction,
      candidateChannels: Int
    ) throws {
      guard
        case .ndArray(let input)? = function.descriptor.inputDescriptor(
          of: "image"
        ),
        case .ndArray(let candidates)? = function.descriptor.outputDescriptor(
          of: "candidates"
        ),
        case .ndArray(let prototypes)? = function.descriptor.outputDescriptor(
          of: "prototypes"
        ), input.shape == [1, 3, 640, 640],
        candidates.shape == [1, candidateChannels, 8400],
        prototypes.shape == [1, 32, 160, 160],
        input.scalarType == .float16,
        candidates.scalarType == .float16,
        prototypes.scalarType == .float16
      else {
        throw MiohIPadWorkerEngineError.detector(
          "detector FP16 tensor contract does not match"
        )
      }
    }
  #endif
}

#if canImport(CoreAI)
  @available(iOS 27.0, *)
  private final class MiohIPadCoreAIRestorer: MiohIPadRestoring,
    @unchecked Sendable
  {
    private let function: InferenceFunction
    private let fixedFrameCount: Int
    private let frameElements = 3 * 256 * 256

    init(function: InferenceFunction, fixedFrameCount: Int) {
      self.function = function
      self.fixedFrameCount = fixedFrameCount
    }

    func restore(_ frames: [Float16], frameCount: Int) async throws
      -> MiohIPadRestoredFrames
    {
      guard frameCount > 0, frameCount <= fixedFrameCount,
        frames.count == frameCount * frameElements
      else {
        throw MiohIPadWorkerEngineError.restorer(
          "invalid T\(fixedFrameCount) input"
        )
      }
      var input = NDArray(
        shape: [1, fixedFrameCount, 3, 256, 256],
        scalarType: .float16
      )
      let view = input.mutableView(as: Float16.self)
      try view.withUnsafeMutablePointer { destination, shape, _ in
        let actualShape = (0..<shape.count).map { shape[$0] }
        guard actualShape == [1, fixedFrameCount, 3, 256, 256] else {
          throw MiohIPadWorkerEngineError.restorer("input shape changed")
        }
        frames.withUnsafeBufferPointer { source in
          destination.update(from: source.baseAddress!, count: source.count)
        }
        let last = destination.advanced(by: (frameCount - 1) * frameElements)
        if frameCount < fixedFrameCount {
          for index in frameCount..<fixedFrameCount {
            destination.advanced(by: index * frameElements).update(
              from: last,
              count: frameElements
            )
          }
        }
      }
      var outputs = try await function.run(inputs: ["frames": input])
      guard let value = outputs.remove("restored")?.ndArray else {
        throw MiohIPadWorkerEngineError.restorer("restored output is missing")
      }
      let output = value.view(as: Float16.self)
      guard output.isContiguous else {
        throw MiohIPadWorkerEngineError.restorer("output is not contiguous")
      }
      return try output.withUnsafePointer { pointer, shape, _ in
        let actualShape = (0..<shape.count).map { shape[$0] }
        guard actualShape == [1, fixedFrameCount, 3, 256, 256] else {
          throw MiohIPadWorkerEngineError.restorer("output shape changed")
        }
        return MiohIPadRestoredFrames(
          Array(
            UnsafeBufferPointer(
              start: pointer,
              count: frameCount * frameElements
            )
          )
        )
      }
    }
  }

  @available(iOS 27.0, *)
  private actor MiohIPadSharedRestorerCache {
    private let fixedRestorationIdentities: [String: MiohValidatedPortableModelIdentity]
    private let variableRestorationIdentity: MiohValidatedPortableModelCollection
    private let restorationFrameCounts: [String: Int]
    private var cached:
      (
        identifier: String,
        clipLength: Int,
        instance: any MiohIPadRestoring
      )?

    init(
      fixedRestorationIdentities:
        [String: MiohValidatedPortableModelIdentity],
      variableRestorationIdentity: MiohValidatedPortableModelCollection,
      restorationFrameCounts: [String: Int]
    ) {
      self.fixedRestorationIdentities = fixedRestorationIdentities
      self.variableRestorationIdentity = variableRestorationIdentity
      self.restorationFrameCounts = restorationFrameCounts
    }

    func restorer(
      identifier: String,
      clipLength: Int
    ) async throws -> any MiohIPadRestoring {
      if let cached,
        cached.identifier == identifier,
        cached.clipLength == clipLength
      {
        return cached.instance
      }

      // A variable-length BasicVSR++ workspace is roughly 300 MiB at T90.
      // The local preview has three decode/detect lanes, but those lanes must
      // share one restoration graph and workspace; retaining one per lane can
      // exceed the iOS process budget before inference even begins.
      cached = nil
      let loaded: any MiohIPadRestoring
      let fixedRuntimeURL = fixedRestorationIdentities[identifier].map {
        MiohIPadVariableRestorer.runtimeAssetURL(for: $0.assetURL)
      }
      if identifier == "basicvsrpp-v1.2-coreai",
        let identity = fixedRestorationIdentities[identifier],
        let frameCount = restorationFrameCounts[identifier],
        let runtimeURL = fixedRuntimeURL,
        runtimeURL.pathExtension == "aimodelc"
      {
        let model = try await AIModel(contentsOf: runtimeURL)
        guard let function = try model.loadFunction(named: identity.functionName)
        else { throw MiohClusterAssetError.missingMainFunction }
        try MiohIPadWorkerEngine.validateRestorationContract(
          function,
          frameCount: frameCount
        )
        loaded = MiohIPadCoreAIRestorer(
          function: function,
          fixedFrameCount: frameCount
        )
      } else {
        // Fixed T18/T36/T90 graphs are hundreds of MB once specialized and
        // can exceed a single Metal heap on iPad. Use the equivalent chunked
        // variable runtime unless a supported fixed AOT leaf is bundled.
        loaded = try await MiohIPadVariableRestorer(
          assetURLsBySourceName:
            variableRestorationIdentity.assetURLsBySourceName,
          maximumFrames: clipLength
        )
      }
      cached = (identifier, clipLength, loaded)
      return loaded
    }
  }

  @available(iOS 27.0, *)
  private actor MiohIPadExecutionCore {
    private let sharedRestorerCache: MiohIPadSharedRestorerCache
    private let detectorIdentities: [String: MiohValidatedPortableModelIdentity]
    private let restorationFrameCounts: [String: Int]
    private let detectorCandidateChannels: [String: Int]
    private let restorationDigests: [String: String]
    private let detectorDigests: [String: String]
    private var cachedDetector:
      (
        identifier: String,
        instance: MiohIPadDetector
      )?

    init(
      sharedRestorerCache: MiohIPadSharedRestorerCache,
      detectorIdentities: [String: MiohValidatedPortableModelIdentity],
      restorationFrameCounts: [String: Int],
      detectorCandidateChannels: [String: Int],
      restorationDigests: [String: String],
      detectorDigests: [String: String]
    ) {
      self.sharedRestorerCache = sharedRestorerCache
      self.detectorIdentities = detectorIdentities
      self.restorationFrameCounts = restorationFrameCounts
      self.detectorCandidateChannels = detectorCandidateChannels
      self.restorationDigests = restorationDigests
      self.detectorDigests = detectorDigests
    }

    func releaseCachedModels() {
      cachedDetector = nil
    }

    func execute(
      _ request: MiohClusterJobRequest,
      inputURL: URL,
      outputURL: URL,
      progress: (@Sendable (Double) -> Void)? = nil,
      preview: (@Sendable (MiohIPadPreviewFrame) async -> Void)? = nil
    ) async throws -> MiohClusterJobMetrics {
      progress?(0)
      let io = try MiohIPadJobIO.resolve(
        request: request,
        inputURL: inputURL,
        outputURL: outputURL
      )
      let identifiers = try validate(
        request,
        inputExtension: io.input.mediaPathExtension,
        outputExtension: io.output.mediaPathExtension
      )
      let startedAt = Date()
      let inputSecurityURL = io.input.securityScopedURL
      let outputSecurityURL = io.output.securityScopedURL
      let inputScope =
        inputSecurityURL?.startAccessingSecurityScopedResource()
        ?? false
      let outputScope =
        outputSecurityURL?.startAccessingSecurityScopedResource()
        ?? false
      defer {
        if outputScope { outputSecurityURL?.stopAccessingSecurityScopedResource() }
        if inputScope { inputSecurityURL?.stopAccessingSecurityScopedResource() }
      }
      let workingURL = try io.output.makeLocalStagingURL(
        attemptID: request.attemptID
      )
      try? FileManager.default.removeItem(at: workingURL)
      defer { try? FileManager.default.removeItem(at: workingURL) }

      let assetHandle = try await io.input.makeAsset(attemptID: request.attemptID)
      return try await withTaskCancellationHandler(
        operation: {
          // The handle retains MiohHTTPRangeAsset because AVAssetResourceLoader
          // holds its delegate weakly. Keep it through metadata load, reader
          // setup and every asynchronous provider read.
          defer { assetHandle.cancel() }
          let asset = assetHandle.asset
          let tracks: [AVAssetTrack]
          do {
            tracks = try await asset.loadTracks(withMediaType: .video)
          } catch is CancellationError {
            throw CancellationError()
          } catch {
            // Remote input URLs are capability credentials and must not escape in
            // error text or logs.
            throw MiohIPadWorkerEngineError.decoder(
              "video track metadata is unavailable"
            )
          }
          guard let track = tracks.first else {
            throw MiohIPadWorkerEngineError.unsupportedMedia("video track is missing")
          }
          let naturalSize: CGSize
          let preferredTransform: CGAffineTransform
          let formatDescriptions: [CMFormatDescription]
          do {
            naturalSize = try await track.load(.naturalSize)
            preferredTransform = try await track.load(.preferredTransform)
            formatDescriptions = try await track.load(.formatDescriptions)
          } catch is CancellationError {
            throw CancellationError()
          } catch {
            let value = error as NSError
            throw MiohIPadWorkerEngineError.decoder(
              "video track geometry metadata is unavailable: "
                + "\(error.localizedDescription) [\(value.domain):\(value.code)]"
            )
          }
          let encodedSize: CGSize? = formatDescriptions.lazy.compactMap {
            description -> CGSize? in
            let dimensions = CMVideoFormatDescriptionGetDimensions(description)
            guard dimensions.width > 0, dimensions.height > 0 else { return nil }
            return CGSize(
              width: Int(dimensions.width),
              height: Int(dimensions.height)
            )
          }.first
          // AVAssetTrack.naturalSize includes the H.264 sample-aspect ratio.
          // A 854x480 stream with SAR 1280:1281 is therefore reported as
          // 853.333x480 even though AVAssetReader produces 854x480 buffers.
          // Restoration and the output encoder must use the decoded buffer's
          // encoded pixel dimensions, with preferredTransform applied only for
          // orientation.
          let sourcePixelSize = encodedSize ?? naturalSize
          let transformedBounds = CGRect(origin: .zero, size: sourcePixelSize)
            .applying(preferredTransform).standardized
          let width = max(1, Int(abs(transformedBounds.width).rounded()))
          let height = max(1, Int(abs(transformedBounds.height).rounded()))
          guard width.isMultiple(of: 2), height.isMultiple(of: 2) else {
            throw MiohIPadWorkerEngineError.unsupportedMedia(
              "encoded dimensions must be even (\(width)x\(height))"
            )
          }
          let isRealtimePreview =
            request.sharedRootIdentifier
            == IPadRealtimePreviewConfiguration.sharedRootIdentifier
          let pixelFrameBudget =
            isRealtimePreview
            ? IPadRestorationMediaLimits.maximumRealtimePixelFrames
            : IPadRestorationMediaLimits.maximumPixelFrames
          let referenceClipLength =
            isRealtimePreview
            ? IPadRestorationMediaLimits.realtimeReferenceClipLength
            : IPadRestorationMediaLimits.referenceClipLength
          guard
            IPadRestorationMediaLimits.accepts(
              width: width,
              height: height,
              clipLength: request.options.restorationClipLength,
              pixelFrameBudget: pixelFrameBudget
            )
          else {
            throw MiohIPadWorkerEngineError.unsupportedMedia(
              "復元メモリ上限を超えます（\(width)x\(height)、clip \(request.options.restorationClipLength)）。長辺1920・短辺1080、かつ1080p×\(referenceClipLength)相当以下にしてください"
            )
          }
          let selected = try await loadSelectedModels(
            restorationIdentifier: identifiers.restoration,
            detectorIdentifier: identifiers.detector,
            clipLength: request.options.restorationClipLength
          )
          let rate = await MiohIPadSourceFrameRate.resolve(track: track)
          let outputRate = try MiohIPadSourceFrameRate.outputRate(
            requestedNumerator: request.options.targetFPSNumerator,
            requestedDenominator: request.options.targetFPSDenominator,
            sourceNumerator: rate.numerator,
            sourceDenominator: rate.denominator
          )
          // This value only tunes output bitrate. Some fragmented HLS assets
          // do not expose it on iOS, so metadata failure must not abort the
          // restoration pipeline.
          let estimatedDataRate =
            (try? await track.load(.estimatedDataRate)).map(Double.init) ?? 0
          let codec: AVVideoCodecType =
            request.options.videoCodec == "h264"
            ? .h264 : .hevc
          let writer = try MiohIPadVideoWriter(
            url: workingURL,
            width: width,
            height: height,
            fpsNumerator: outputRate.numerator,
            fpsDenominator: outputRate.denominator,
            codec: codec,
            sourceBitRate: estimatedDataRate,
            bitrateMultiplier: request.options.bitrateMultiplier,
            fastStart: request.options.mp4FastStart
          )
          let processor = try MiohIPadFrameProcessor(
            width: width,
            height: height,
            restorer: selected.restorer,
            blendFeather: request.options.blendFeather,
            detectionEmptyLookahead: request.options.detectionEmptyLookahead
          )
          let emitter = MiohIPadShardEmitter(
            processor: processor,
            writer: writer,
            coreStartNanoseconds: request.mediaRange.coreStartNanoseconds,
            coreEndNanoseconds: request.mediaRange.coreEndNanoseconds,
            overlap: request.options.temporalOverlap,
            crossfade: request.options.crossfade,
            preview: preview
          )
          do {
            let decodedCoreFrames = try await decodeAndProcess(
              asset: asset,
              track: track,
              preferredTransform: preferredTransform,
              outputWidth: width,
              outputHeight: height,
              request: request,
              processor: processor,
              detector: selected.detector,
              progress: progress,
              emitter: emitter
            )
            try await emitter.flush()
            progress?(0.95)
            let writtenFrames = try await writer.finish(
              durationNanoseconds: request.mediaRange.coreEndNanoseconds
                - request.mediaRange.coreStartNanoseconds
            )
            guard writtenFrames == emitter.processedFrames,
              request.options.targetFPSNumerator != nil
                || writtenFrames == decodedCoreFrames
            else {
              throw MiohIPadWorkerEngineError.output(
                "frame count mismatch decoded=\(decodedCoreFrames), written=\(writtenFrames)"
              )
            }
            try Task.checkCancellation()
            let values = try workingURL.resourceValues(forKeys: [
              .isRegularFileKey, .fileSizeKey,
            ])
            guard values.isRegularFile == true, let byteCount = values.fileSize,
              byteCount > 0
            else { throw MiohIPadWorkerEngineError.output("empty shard") }
            try await io.output.publish(
              localFile: workingURL,
              byteCount: Int64(byteCount)
            )
            progress?(1)
            return MiohClusterJobMetrics(
              processedFrames: emitter.processedFrames,
              wallSeconds: max(0.001, Date().timeIntervalSince(startedAt)),
              outputByteCount: Int64(byteCount),
              restoredFrames: emitter.restoredFrames,
              restorationSeconds: emitter.restorationSeconds,
              restorationPreparationSeconds:
                emitter.restorationPreparationSeconds,
              restorationCompositingSeconds:
                emitter.restorationCompositingSeconds
            )
          } catch {
            writer.cancel()
            throw error
          }
        },
        onCancel: {
          assetHandle.cancel()
        }
      )
    }

    private func validate(
      _ request: MiohClusterJobRequest,
      inputExtension: String,
      outputExtension: String
    ) throws -> (
      restoration: String,
      detector: String
    ) {
      guard request.options.isValid else {
        throw MiohIPadWorkerEngineError.unsupportedJob(
          "invalid restoration options"
        )
      }
      let restorationIdentifier = request.options.restorationModelIdentifier
      let detectorIdentifier = request.options.detectorModelIdentifier
      guard let restorationDigest = restorationDigests[restorationIdentifier],
        request.options.restorationAssetSHA256.lowercased()
          == restorationDigest,
        let detectorDigest = detectorDigests[detectorIdentifier],
        request.options.detectorAssetSHA256.lowercased()
          == detectorDigest
      else {
        throw MiohIPadWorkerEngineError.unsupportedJob(
          "portable model identity does not match"
        )
      }
      guard request.options.restorationClipLength > 0,
        request.options.restorationClipLength <= 90,
        request.options.temporalOverlap
          < request.options.restorationClipLength,
        (1...120).contains(request.options.detectionEmptyLookahead)
      else {
        throw MiohIPadWorkerEngineError.unsupportedJob(
          "Core AI worker requires clipLength 1...90 and lookahead 1...120"
        )
      }
      guard request.options.roiEnhancerModelIdentifier == nil,
        request.options.roiEnhancerAssetSHA256 == nil,
        request.options.roiEnhancerStrength == 0
      else {
        throw MiohIPadWorkerEngineError.unsupportedJob(
          "ROI enhancer is not supported"
        )
      }
      guard request.options.sharpenStrength == 0,
        request.options.detailBoost == 0,
        request.options.textureMix == 0,
        request.options.smoothStrength == 0,
        request.options.effectUpscale == 1
      else {
        throw MiohIPadWorkerEngineError.unsupportedJob(
          "post-restoration effects are not supported"
        )
      }
      guard request.mediaRange.isValid else {
        throw MiohIPadWorkerEngineError.unsupportedJob("invalid media range")
      }
      let supportedInputs = Set(["mp4", "mov", "m4v", "ts"])
      guard supportedInputs.contains(inputExtension),
        outputExtension == "mp4"
      else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "input must be mp4/mov/m4v/ts and output must be MP4"
        )
      }
      return (restorationIdentifier, detectorIdentifier)
    }

    private func loadSelectedModels(
      restorationIdentifier: String,
      detectorIdentifier: String,
      clipLength: Int
    ) async throws -> (
      restorer: any MiohIPadRestoring,
      detector: MiohIPadDetector
    ) {
      let restorer = try await sharedRestorerCache.restorer(
        identifier: restorationIdentifier,
        clipLength: clipLength
      )

      let detector: MiohIPadDetector
      if let cached = cachedDetector,
        cached.identifier == detectorIdentifier
      {
        detector = cached.instance
      } else {
        // Keep only the selected detector resident alongside the selected
        // restoration graph.
        cachedDetector = nil
        guard let detectorIdentity = detectorIdentities[detectorIdentifier],
          let candidateChannels = detectorCandidateChannels[detectorIdentifier]
        else {
          throw MiohIPadWorkerEngineError.unsupportedJob(
            "detector model is not available"
          )
        }
        let loaded: MiohIPadDetector
        if detectorIdentifier.hasSuffix("-coreml") {
          let compiledURL = try await MLModel.compileModel(
            at: detectorIdentity.assetURL
          )
          let configuration = MLModelConfiguration()
          // Keep detection on CPU/Neural Engine. Running it through Core AI
          // competes with the BasicVSR++ restoration graph and reduced the
          // measured restoration throughput.
          configuration.computeUnits = .cpuAndNeuralEngine
          let model = try MLModel(
            contentsOf: compiledURL,
            configuration: configuration
          )
          loaded = try MiohIPadDetector(
            coreMLModel: model,
            compiledModelURL: compiledURL,
            candidateChannels: candidateChannels
          )
        } else {
          let detectorModel = try await AIModel(
            contentsOf: MiohIPadVariableRestorer.runtimeAssetURL(
              for: detectorIdentity.assetURL
            )
          )
          guard
            let detectorFunction = try detectorModel.loadFunction(
              named: detectorIdentity.functionName
            )
          else { throw MiohClusterAssetError.missingMainFunction }
          try MiohIPadWorkerEngine.validateDetectorContract(
            detectorFunction,
            candidateChannels: candidateChannels
          )
          loaded = MiohIPadDetector(
            function: detectorFunction,
            candidateChannels: candidateChannels
          )
        }
        cachedDetector = (detectorIdentifier, loaded)
        detector = loaded
      }
      return (restorer, detector)
    }

    func makeRealtimeFrameSession(
      options: MiohClusterRestorationOptions,
      width: Int,
      height: Int
    ) async throws -> MiohIPadRealtimeFrameSession {
      guard width > 0, height > 0,
        width.isMultiple(of: 2), height.isMultiple(of: 2),
        options.isValid,
        restorationDigests[options.restorationModelIdentifier]
          == options.restorationAssetSHA256.lowercased(),
        detectorDigests[options.detectorModelIdentifier]
          == options.detectorAssetSHA256.lowercased()
      else {
        throw MiohIPadWorkerEngineError.unsupportedJob(
          "リアルタイムフレーム処理のモデルまたは映像条件が不正です"
        )
      }
      let selected = try await loadSelectedModels(
        restorationIdentifier: options.restorationModelIdentifier,
        detectorIdentifier: options.detectorModelIdentifier,
        clipLength: options.restorationClipLength
      )
      let processor = try MiohIPadFrameProcessor(
        width: width,
        height: height,
        restorer: selected.restorer,
        blendFeather: options.blendFeather,
        detectionEmptyLookahead: options.detectionEmptyLookahead
      )
      let fixedLimit = restorationFrameCounts[
        options.restorationModelIdentifier
      ]
      return MiohIPadRealtimeFrameSession(
        processor: processor,
        detector: selected.detector,
        clipLength: min(
          options.restorationClipLength,
          fixedLimit ?? options.restorationClipLength
        ),
        overlap: options.temporalOverlap,
        crossfade: options.crossfade,
        faceOnly: options.detectFaceMosaics,
        detectionEmptyLookahead: options.detectionEmptyLookahead
      )
    }

    private func decodeAndProcess(
      asset: AVURLAsset,
      track: AVAssetTrack,
      preferredTransform: CGAffineTransform,
      outputWidth: Int,
      outputHeight: Int,
      request: MiohClusterJobRequest,
      processor: MiohIPadFrameProcessor,
      detector: MiohIPadDetector,
      progress: (@Sendable (Double) -> Void)?,
      emitter: MiohIPadShardEmitter
    ) async throws -> Int {
      let reader: AVAssetReader
      do {
        reader = try AVAssetReader(asset: asset)
      } catch {
        let value = error as NSError
        throw MiohIPadWorkerEngineError.decoder(
          "reader作成: \(error.localizedDescription) [\(value.domain):\(value.code)]"
        )
      }
      let start = CMTime(
        value: request.mediaRange.decodeStartNanoseconds,
        timescale: 1_000_000_000
      )
      let end = CMTime(
        value: request.mediaRange.decodeEndNanoseconds,
        timescale: 1_000_000_000
      )
      reader.timeRange = CMTimeRange(start: start, duration: end - start)
      let settings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
      let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
      let provider = reader.outputProvider(for: output)
      do {
        try reader.start()
      } catch {
        throw MiohIPadWorkerEngineError.decoder(
          "reader start failed: \(error.localizedDescription)"
        )
      }
      defer { reader.cancelReading() }
      let orienter = try MiohIPadFrameOrienter(
        transform: preferredTransform,
        width: outputWidth,
        height: outputHeight
      )
      let clipLength = request.options.restorationClipLength
      let overlap = request.options.temporalOverlap
      var pending: [MiohIPadDetectedFrame] = []
      pending.reserveCapacity(clipLength)
      var batchIndex = 0
      var newFramesSinceBatch = 0
      var decodedCoreFrames = 0
      var lookaheadWindow: [MiohIPadDecodedFrame] = []
      let lookahead = request.options.detectionEmptyLookahead
      lookaheadWindow.reserveCapacity(lookahead)
      var inputFrameRateGate: MiohIPadPTSFrameRateGate?
      if let targetFPSNumerator = request.options.targetFPSNumerator,
        let targetFPSDenominator = request.options.targetFPSDenominator
      {
        inputFrameRateGate = MiohIPadPTSFrameRateGate(
          numerator: targetFPSNumerator,
          denominator: targetFPSDenominator
        )
      }

      func consumeDetected(_ frames: [MiohIPadDetectedFrame]) async throws {
        for detected in frames {
          pending.append(detected)
          newFramesSinceBatch += 1
          if pending.count == clipLength {
            try await emitter.consume(
              pending,
              skipPrefix: batchIndex == 0 ? 0 : overlap
            )
            batchIndex += 1
            pending = overlap > 0 ? Array(pending.suffix(overlap)) : []
            newFramesSinceBatch = 0
          }
        }
      }

      while let sample = try await provider.next() {
        try Task.checkCancellation()
        guard let pixelSample = CMReadySampleBuffer<CVReadOnlyPixelBuffer>(sample)
        else { continue }
        let image = pixelSample.content.withUnsafeBuffer { $0 }
        let pts = pixelSample.presentationTimeStamp
        let ptsNanoseconds = Int64((pts.seconds * 1_000_000_000).rounded())
        guard ptsNanoseconds < request.mediaRange.decodeEndNanoseconds else {
          break
        }
        let range = max(
          1,
          request.mediaRange.decodeEndNanoseconds
            - request.mediaRange.decodeStartNanoseconds
        )
        progress?(
          min(
            0.9,
            max(
              0,
              Double(ptsNanoseconds - request.mediaRange.decodeStartNanoseconds)
                / Double(range) * 0.9
            )
          )
        )
        if var gate = inputFrameRateGate {
          let accepted = gate.accepts(ptsNanoseconds)
          inputFrameRateGate = gate
          guard accepted else { continue }
        }
        let oriented = try orienter.orient(image)
        lookaheadWindow.append(
          MiohIPadDecodedFrame(
            pixelBuffer: oriented,
            ptsNanoseconds: ptsNanoseconds
          )
        )
        if ptsNanoseconds >= request.mediaRange.coreStartNanoseconds,
          ptsNanoseconds < request.mediaRange.coreEndNanoseconds
        {
          decodedCoreFrames += 1
        }
        if lookaheadWindow.count == lookahead {
          try await consumeDetected(
            detectLookaheadWindow(
              lookaheadWindow,
              faceOnly: request.options.detectFaceMosaics,
              reuseSkipFrames:
                request.options.detectionMaskReuseSkipFrames ?? 0,
              detector: detector
            )
          )
          lookaheadWindow.removeAll(keepingCapacity: true)
        }
      }
      if reader.status == .failed {
        throw MiohIPadWorkerEngineError.decoder(
          reader.error?.localizedDescription ?? "reading failed"
        )
      }
      if !lookaheadWindow.isEmpty {
        try await consumeDetected(
          detectLookaheadWindow(
            lookaheadWindow,
            faceOnly: request.options.detectFaceMosaics,
            reuseSkipFrames:
              request.options.detectionMaskReuseSkipFrames ?? 0,
            detector: detector
          )
        )
      }
      if newFramesSinceBatch > 0 {
        try await emitter.consume(
          pending,
          skipPrefix: batchIndex == 0 ? 0 : overlap
        )
      }
      guard decodedCoreFrames > 0 else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "core range contains no decoded frame"
        )
      }
      return decodedCoreFrames
    }

    /// Matches the macOS native endpoint lookahead contract. A range whose
    /// first and last frames are both empty can bypass all middle inference.
    /// Non-empty windows either detect every middle frame or use the configured
    /// sampling stride; the cheap empty-window bypass always wins first.
    private func detectLookaheadWindow(
      _ frames: [MiohIPadDecodedFrame],
      faceOnly: Bool,
      reuseSkipFrames: Int,
      detector: MiohIPadDetector
    ) async throws -> [MiohIPadDetectedFrame] {
      guard !frames.isEmpty else { return [] }
      func filtered(_ detections: [MiohIPadDetection]) -> [MiohIPadDetection] {
        faceOnly ? detections.filter { $0.classIndex == 0 } : detections
      }
      if frames.count == 1 {
        return [
          MiohIPadDetectedFrame(
            frame: frames[0],
            detections: filtered(try await detector.detect(frames[0].pixelBuffer))
          )
        ]
      }
      let first = filtered(try await detector.detect(frames[0].pixelBuffer))
      let lastIndex = frames.count - 1
      let last = filtered(
        try await detector.detect(frames[lastIndex].pixelBuffer)
      )
      if first.isEmpty, last.isEmpty {
        return frames.map { MiohIPadDetectedFrame(frame: $0, detections: []) }
      }
      let skipFrames = min(12, max(0, reuseSkipFrames))
      if skipFrames > 0 {
        var result: [MiohIPadDetectedFrame] = []
        result.reserveCapacity(frames.count)
        var reused = first
        for index in frames.indices {
          let detections: [MiohIPadDetection]
          if index == 0 {
            detections = first
          } else if index == lastIndex {
            reused = last
            detections = last
          } else if index.isMultiple(of: skipFrames + 1) {
            reused = filtered(
              try await detector.detect(frames[index].pixelBuffer)
            )
            detections = reused
          } else {
            detections = reused
          }
          result.append(
            MiohIPadDetectedFrame(frame: frames[index], detections: detections)
          )
        }
        return result
      }
      var result: [MiohIPadDetectedFrame] = []
      result.reserveCapacity(frames.count)
      for index in frames.indices {
        let detections: [MiohIPadDetection]
        if index == 0 {
          detections = first
        } else if index == lastIndex {
          detections = last
        } else {
          detections = filtered(
            try await detector.detect(frames[index].pixelBuffer)
          )
        }
        result.append(
          MiohIPadDetectedFrame(frame: frames[index], detections: detections)
        )
      }
      return result
    }
  }
#endif

enum MiohIPadSourceFrameRate {
  static func resolve(track: AVAssetTrack) async -> (numerator: Int, denominator: Int) {
    if let nominal = try? await track.load(.nominalFrameRate) {
      let value = Double(nominal)
      if value.isFinite, (1...240).contains(value) {
        return rational(value)
      }
    }
    if let minimumDuration = try? await track.load(.minFrameDuration),
      minimumDuration.isNumeric,
      minimumDuration.seconds.isFinite,
      minimumDuration.seconds > 0
    {
      let derived = 1 / minimumDuration.seconds
      if (1...240).contains(derived) { return rational(derived) }
    }
    // This value is only an encoder hint. The writer receives every decoded
    // source PTS, so TS and VFR duration never falls back to a synthetic 1 fps.
    return (30, 1)
  }

  static func rational(_ fps: Double) -> (numerator: Int, denominator: Int) {
    guard fps.isFinite, fps > 0 else { return (30, 1) }
    let ntscWhole = max(1, Int((fps * 1.001).rounded()))
    let ntsc = Double(ntscWhole) * 1000 / 1001
    if abs(fps - ntsc) < 0.001 { return (ntscWhole * 1000, 1001) }
    let whole = max(1, Int(fps.rounded()))
    if abs(fps - Double(whole)) < 0.001 { return (whole, 1) }
    let denominator = 1000
    let numerator = max(1, Int((fps * Double(denominator)).rounded()))
    let divisor = gcd(numerator, denominator)
    return (numerator / divisor, denominator / divisor)
  }

  static func outputRate(
    requestedNumerator: Int?,
    requestedDenominator: Int?,
    sourceNumerator: Int,
    sourceDenominator: Int
  ) throws -> (numerator: Int, denominator: Int) {
    guard let requestedNumerator, let requestedDenominator else {
      return (sourceNumerator, sourceDenominator)
    }
    guard requestedNumerator > 0, requestedDenominator > 0 else {
      throw MiohIPadWorkerEngineError.unsupportedJob(
        "FPS変換の分子・分母が不正です"
      )
    }
    let source = Double(sourceNumerator) / Double(sourceDenominator)
    let target = Double(requestedNumerator) / Double(requestedDenominator)
    guard target <= source + 0.01 else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        String(
          format: "FPS変換はダウン変換のみ対応しています（入力 %.3ffps、指定 %.3ffps）",
          source,
          target
        )
      )
    }
    return (requestedNumerator, requestedDenominator)
  }

  /// Returns the exact rate used by a maximum-FPS limit. NTSC sources stay
  /// in the NTSC family, so 59.94fps becomes 30000/1001 rather than 30/1.
  static func maximumRate(
    wholeFPS: Int,
    sourceNumerator: Int,
    sourceDenominator: Int
  ) -> (numerator: Int, denominator: Int)? {
    guard wholeFPS > 0, sourceNumerator > 0, sourceDenominator > 0 else {
      return nil
    }
    let pulled = Double(sourceNumerator) * 1.001 / Double(sourceDenominator)
    let isNTSC =
      abs(pulled - pulled.rounded()) < 0.001
      && pulled.rounded() >= 1
    let target =
      isNTSC
      ? (numerator: wholeFPS * 1_000, denominator: 1_001)
      : (numerator: wholeFPS, denominator: 1)
    let sourceValue = Double(sourceNumerator) / Double(sourceDenominator)
    let targetValue = Double(target.numerator) / Double(target.denominator)
    return sourceValue > targetValue + 0.01 ? target : nil
  }

  private static func gcd(_ lhs: Int, _ rhs: Int) -> Int {
    var x = abs(lhs)
    var y = abs(rhs)
    while y != 0 { (x, y) = (y, x % y) }
    return max(1, x)
  }
}

private struct MiohIPadOutputFrame: @unchecked Sendable {
  let pixelBuffer: CVPixelBuffer
  let ptsNanoseconds: Int64
}

/// Selects the first source frame in each absolute target-rate time slot.
/// Because the slot is anchored at PTS zero, every Worker makes the same
/// decision for overlap frames and conversion phase does not restart per shard.
private struct MiohIPadPTSFrameRateGate: Sendable {
  private let numerator: Double
  private let denominatorNanoseconds: Double
  private var lastSlot: Int64?

  init(numerator: Int, denominator: Int) {
    self.numerator = Double(max(1, numerator))
    denominatorNanoseconds = Double(max(1, denominator)) * 1_000_000_000
  }

  mutating func accepts(_ ptsNanoseconds: Int64) -> Bool {
    let slot = Int64(
      floor(
        Double(max(0, ptsNanoseconds)) * numerator / denominatorNanoseconds
          + 1e-8
      )
    )
    guard slot != lastSlot else { return false }
    lastSlot = slot
    return true
  }
}

/// Resolves a direct-frame HLS cadence from its first positive PTS delta,
/// then applies the same exact NTSC-aware maximum-rate gate as file input.
/// The first frame is always retained and a 29.97/30fps source is untouched.
private struct MiohIPadAdaptiveMaximumFrameRateGate: Sendable {
  private var maximumWholeFPS: Int?
  private var previousPTS: Int64?
  private var resolvedGate: MiohIPadPTSFrameRateGate?
  private var sourceIsWithinLimit = false

  mutating func configure(wholeFPS: Int?) {
    maximumWholeFPS = wholeFPS.flatMap { $0 > 0 ? $0 : nil }
    previousPTS = nil
    resolvedGate = nil
    sourceIsWithinLimit = false
  }

  mutating func accepts(_ ptsNanoseconds: Int64) -> Bool {
    guard let maximumWholeFPS else { return true }
    if sourceIsWithinLimit { return true }
    if var gate = resolvedGate {
      let accepted = gate.accepts(ptsNanoseconds)
      resolvedGate = gate
      return accepted
    }
    guard let previousPTS else {
      self.previousPTS = ptsNanoseconds
      return true
    }
    self.previousPTS = ptsNanoseconds
    let delta = ptsNanoseconds - previousPTS
    guard delta > 0 else { return false }
    let measuredFPS = 1_000_000_000 / Double(delta)
    let sourceRate = MiohIPadSourceFrameRate.rational(measuredFPS)
    guard
      let target = MiohIPadSourceFrameRate.maximumRate(
        wholeFPS: maximumWholeFPS,
        sourceNumerator: sourceRate.numerator,
        sourceDenominator: sourceRate.denominator
      )
    else {
      sourceIsWithinLimit = true
      return true
    }
    var gate = MiohIPadPTSFrameRateGate(
      numerator: target.numerator,
      denominator: target.denominator
    )
    _ = gate.accepts(previousPTS)
    let accepted = gate.accepts(ptsNanoseconds)
    resolvedGate = gate
    return accepted
  }
}

@available(iOS 27.0, *)
private final class MiohIPadShardEmitter {
  private let processor: MiohIPadFrameProcessor
  private let writer: MiohIPadVideoWriter
  private let coreStartNanoseconds: Int64
  private let coreEndNanoseconds: Int64
  private let overlap: Int
  private let crossfade: Bool
  private let preview: (@Sendable (MiohIPadPreviewFrame) async -> Void)?
  private lazy var previewContext = CIContext(options: [
    .workingColorSpace: NSNull(),
    .outputColorSpace: NSNull(),
    .cacheIntermediates: false,
  ])
  private var deferredTail: [MiohIPadOutputFrame] = []
  private var previousPreviewPTS: Int64?
  private(set) var processedFrames = 0
  private(set) var restoredFrames = 0
  private(set) var restorationSeconds = 0.0
  private(set) var restorationPreparationSeconds = 0.0
  private(set) var restorationCompositingSeconds = 0.0

  init(
    processor: MiohIPadFrameProcessor,
    writer: MiohIPadVideoWriter,
    coreStartNanoseconds: Int64,
    coreEndNanoseconds: Int64,
    overlap: Int,
    crossfade: Bool,
    preview: (@Sendable (MiohIPadPreviewFrame) async -> Void)?
  ) {
    self.processor = processor
    self.writer = writer
    self.coreStartNanoseconds = coreStartNanoseconds
    self.coreEndNanoseconds = coreEndNanoseconds
    self.overlap = overlap
    self.crossfade = crossfade
    self.preview = preview
  }

  func consume(_ batch: [MiohIPadDetectedFrame], skipPrefix: Int) async throws {
    let outputs = try await processor.process(batch)
    restoredFrames += processor.lastRestoredFrameCount
    restorationSeconds += processor.lastRestorationSeconds
    restorationPreparationSeconds += processor.lastPreparationSeconds
    restorationCompositingSeconds += processor.lastCompositingSeconds
    guard outputs.count == batch.count else {
      throw MiohIPadWorkerEngineError.output("processor frame count changed")
    }
    var ready: [MiohIPadOutputFrame] = []
    if crossfade && overlap > 0 {
      let prefixCount = min(skipPrefix, outputs.count)
      let matchedCount = min(prefixCount, deferredTail.count)
      for index in 0..<matchedCount {
        ready.append(
          MiohIPadOutputFrame(
            pixelBuffer: try processor.crossfade(
              earlier: deferredTail[index].pixelBuffer,
              later: outputs[index],
              laterWeight: Float(index + 1) / Float(matchedCount + 1)
            ),
            ptsNanoseconds: deferredTail[index].ptsNanoseconds
          )
        )
      }
      if deferredTail.count > matchedCount {
        ready.append(contentsOf: deferredTail[matchedCount...])
      }
      if prefixCount > matchedCount {
        for index in matchedCount..<prefixCount {
          ready.append(frame(outputs[index], batch[index]))
        }
      }
      let uniqueCount = max(0, outputs.count - prefixCount)
      let tailCount = min(overlap, uniqueCount)
      let bodyEnd = outputs.count - tailCount
      if prefixCount < bodyEnd {
        for index in prefixCount..<bodyEnd {
          ready.append(frame(outputs[index], batch[index]))
        }
      }
      deferredTail.removeAll(keepingCapacity: true)
      if tailCount > 0 {
        for index in bodyEnd..<outputs.count {
          deferredTail.append(frame(outputs[index], batch[index]))
        }
      }
    } else {
      for index in min(skipPrefix, outputs.count)..<outputs.count {
        ready.append(frame(outputs[index], batch[index]))
      }
    }
    for frame in ready { try await appendOwned(frame) }
  }

  func flush() async throws {
    for frame in deferredTail { try await appendOwned(frame) }
    deferredTail.removeAll()
  }

  private func appendOwned(_ frame: MiohIPadOutputFrame) async throws {
    guard frame.ptsNanoseconds >= coreStartNanoseconds,
      frame.ptsNanoseconds < coreEndNanoseconds
    else { return }
    // A fixed shard boundary rarely lands exactly on the source frame grid.
    // Starting an MP4 at the resulting positive timestamp creates an empty
    // edit before its first picture, which AVQueuePlayer can expose as a black
    // frame while advancing between restored shards.  Hold the first owned
    // picture from time zero; keep every later timestamp unchanged so the
    // shard still has the requested duration and stays aligned to source audio.
    // Keep the accepted source PTS inside the shard instead of compacting
    // frames to `processedFrames / targetFPS`. Count-based compaction assumes
    // the owned output count exactly matches the shard-duration rate grid;
    // HLS window and overlap boundaries do not guarantee that invariant. A
    // synthetic final PTS can therefore reach or cross the MP4 end time even
    // though every owned source PTS is inside the half-open core range.
    // Source-relative PTS preserves every accepted frame and audio alignment.
    let presentationTimeNanoseconds =
      processedFrames == 0
      ? 0
      : frame.ptsNanoseconds - coreStartNanoseconds
    try await writer.append(
      frame.pixelBuffer,
      presentationTimeNanoseconds: presentationTimeNanoseconds
    )
    processedFrames += 1
    if let preview, let previewFrame = makePreviewFrame(frame) {
      if let previousPreviewPTS {
        let sourceInterval = max(
          0,
          frame.ptsNanoseconds - previousPreviewPTS
        )
        let displayInterval = min(sourceInterval, 100_000_000)
        if displayInterval > 0 {
          try await Task.sleep(nanoseconds: UInt64(displayInterval))
        }
      }
      try Task.checkCancellation()
      await preview(previewFrame)
      previousPreviewPTS = frame.ptsNanoseconds
    }
  }

  private func makePreviewFrame(
    _ frame: MiohIPadOutputFrame
  ) -> MiohIPadPreviewFrame? {
    let source = CIImage(cvPixelBuffer: frame.pixelBuffer)
    let largestEdge = max(source.extent.width, source.extent.height)
    guard largestEdge > 0 else { return nil }
    let scale = min(1, 960 / largestEdge)
    let previewImage =
      scale < 1
      ? source.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
      : source
    let extent = previewImage.extent.integral
    guard !extent.isEmpty,
      let image = previewContext.createCGImage(previewImage, from: extent)
    else { return nil }
    return MiohIPadPreviewFrame(
      image: image,
      ptsNanoseconds: frame.ptsNanoseconds
    )
  }

  private func frame(
    _ pixelBuffer: CVPixelBuffer,
    _ source: MiohIPadDetectedFrame
  ) -> MiohIPadOutputFrame {
    MiohIPadOutputFrame(
      pixelBuffer: pixelBuffer,
      ptsNanoseconds: source.frame.ptsNanoseconds
    )
  }
}

/// Long-lived in-process realtime path. It accepts AVFoundation-owned pixel
/// buffers directly, so Safari-compatible playback no longer encodes an input
/// MP4 only for AVAssetReader to decode it again. Core AI remains a single
/// inference lane; detector sampling and output encoding can run around it.
@available(iOS 27.0, *)
actor MiohIPadRealtimeFrameSession: MiohIPadRealtimeFrameSessioning {
  private let processor: MiohIPadFrameProcessor
  private let batchDetector: MiohIPadRealtimeBatchDetector
  private let clipLength: Int
  private let overlap: Int
  private let crossfade: Bool
  private var pending: [MiohIPadRealtimeInputFrame] = []
  private var deferredTail: [MiohIPadRealtimeOutputFrame] = []
  private var performanceSamples: [MiohIPadRealtimePerformanceSample] = []
  private var isFirstBatch = true
  private var detectionMaskReuseSkipFrames = 0
  private var maximumFrameRateGate = MiohIPadAdaptiveMaximumFrameRateGate()
  private var emergency24FPS = false
  private var lastAccepted24FPSSlot: Int64?

  fileprivate init(
    processor: MiohIPadFrameProcessor,
    detector: MiohIPadDetector,
    clipLength: Int,
    overlap: Int,
    crossfade: Bool,
    faceOnly: Bool,
    detectionEmptyLookahead: Int
  ) {
    self.processor = processor
    batchDetector = MiohIPadRealtimeBatchDetector(
      detector: detector,
      faceOnly: faceOnly,
      emptyLookahead: detectionEmptyLookahead
    )
    self.clipLength = max(2, clipLength)
    self.overlap = min(max(0, overlap), max(0, clipLength - 1))
    self.crossfade = crossfade
    pending.reserveCapacity(max(2, clipLength))
  }

  func setEmergency24FPSEnabled(_ enabled: Bool) async {
    guard emergency24FPS != enabled else { return }
    emergency24FPS = enabled
    lastAccepted24FPSSlot = nil
  }

  func setMaximumFrameRate(_ wholeFPS: Int?) async {
    maximumFrameRateGate.configure(wholeFPS: wholeFPS)
  }

  func setDetectionMaskReuseSkipFrames(_ count: Int) async {
    detectionMaskReuseSkipFrames = min(12, max(0, count))
  }

  func append(_ frame: MiohIPadRealtimeInputFrame) async throws
    -> [MiohIPadRealtimeOutputFrame]
  {
    try Task.checkCancellation()
    guard acceptsFrame(frame.ptsNanoseconds) else { return [] }
    if let last = pending.last, frame.ptsNanoseconds <= last.ptsNanoseconds {
      return []
    }
    pending.append(frame)
    guard pending.count >= clipLength else { return [] }
    let batchFrames = Array(pending.prefix(clipLength))
    pending = overlap > 0 ? Array(batchFrames.suffix(overlap)) : []
    return try await processBatch(batchFrames, isFinal: false)
  }

  func flush() async throws -> [MiohIPadRealtimeOutputFrame] {
    try Task.checkCancellation()
    if pending.isEmpty {
      defer { deferredTail.removeAll(keepingCapacity: false) }
      return deferredTail
    }
    if !isFirstBatch, pending.count <= overlap {
      pending.removeAll(keepingCapacity: false)
      defer { deferredTail.removeAll(keepingCapacity: false) }
      return deferredTail
    }
    let batchFrames = pending
    pending.removeAll(keepingCapacity: false)
    return try await processBatch(batchFrames, isFinal: true)
  }

  func takePerformanceSamples() async -> [MiohIPadRealtimePerformanceSample] {
    defer { performanceSamples.removeAll(keepingCapacity: true) }
    return performanceSamples
  }

  private func processBatch(
    _ input: [MiohIPadRealtimeInputFrame],
    isFinal: Bool
  ) async throws -> [MiohIPadRealtimeOutputFrame] {
    try Task.checkCancellation()
    let detectionStarted = ContinuousClock.now
    let detected = try await batchDetector.detectBatch(
      input,
      reuseSkipFrames: detectionMaskReuseSkipFrames
    )
    let detectionElapsed = detectionStarted.duration(to: .now)

    try Task.checkCancellation()
    let restorationStarted = ContinuousClock.now
    let restored = try await processor.process(detected)
    let restorationElapsed = restorationStarted.duration(to: .now)
    guard restored.count == detected.count else {
      throw MiohIPadWorkerEngineError.output(
        "リアルタイム復元でフレーム数が変化しました"
      )
    }
    performanceSamples.append(
      MiohIPadRealtimePerformanceSample(
        detectionSeconds: Self.seconds(detectionElapsed),
        restorationSeconds: Self.seconds(restorationElapsed),
        restorationPreparationSeconds: processor.lastPreparationSeconds,
        restorationCompositingSeconds: processor.lastCompositingSeconds,
        mediaSeconds: Self.mediaSeconds(input),
        processedFrames: input.count,
        restoredFrames: processor.lastRestoredFrameCount,
        modelRestorationSeconds: processor.lastRestorationSeconds
      )
    )
    return try emit(
      restored: restored,
      detected: detected,
      isFinal: isFinal
    )
  }

  private func acceptsFrame(_ ptsNanoseconds: Int64) -> Bool {
    guard maximumFrameRateGate.accepts(ptsNanoseconds) else { return false }
    guard emergency24FPS else { return true }
    let slot = Int64(
      floor(Double(max(0, ptsNanoseconds)) * 24 / 1_000_000_000 + 1e-8)
    )
    guard slot != lastAccepted24FPSSlot else { return false }
    lastAccepted24FPSSlot = slot
    return true
  }

  private func emit(
    restored: [CVPixelBuffer],
    detected: [MiohIPadDetectedFrame],
    isFinal: Bool
  ) throws -> [MiohIPadRealtimeOutputFrame] {
    let outputs = zip(restored, detected).map { pixelBuffer, source in
      MiohIPadRealtimeOutputFrame(
        pixelBuffer: pixelBuffer,
        ptsNanoseconds: source.frame.ptsNanoseconds
      )
    }
    let prefixCount = isFirstBatch ? 0 : min(overlap, outputs.count)
    var ready: [MiohIPadRealtimeOutputFrame] = []
    if !isFirstBatch {
      let matched = min(prefixCount, deferredTail.count)
      if matched > 0 {
        for index in 0..<matched {
          let pixelBuffer =
            crossfade
            ? try processor.crossfade(
              earlier: deferredTail[index].pixelBuffer,
              later: outputs[index].pixelBuffer,
              laterWeight: Float(index + 1) / Float(matched + 1)
            )
            : outputs[index].pixelBuffer
          ready.append(
            MiohIPadRealtimeOutputFrame(
              pixelBuffer: pixelBuffer,
              ptsNanoseconds: outputs[index].ptsNanoseconds
            )
          )
        }
      }
      if deferredTail.count > matched {
        ready.append(contentsOf: deferredTail[matched...])
      }
      if prefixCount > matched {
        ready.append(contentsOf: outputs[matched..<prefixCount])
      }
    }

    let bodyEnd =
      isFinal
      ? outputs.count : max(prefixCount, outputs.count - overlap)
    if prefixCount < bodyEnd {
      ready.append(contentsOf: outputs[prefixCount..<bodyEnd])
    }
    deferredTail =
      isFinal || overlap == 0
      ? [] : Array(outputs.suffix(min(overlap, outputs.count - prefixCount)))
    isFirstBatch = false
    return ready
  }

  private static func mediaSeconds(
    _ frames: [MiohIPadRealtimeInputFrame]
  ) -> Double {
    guard let first = frames.first, let last = frames.last else { return 0 }
    if frames.count == 1 { return 1.0 / 30.0 }
    let span = max(
      0,
      Double(last.ptsNanoseconds - first.ptsNanoseconds) / 1_000_000_000
    )
    return span + span / Double(frames.count - 1)
  }

  private static func seconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds)
      + Double(components.attoseconds) / 1_000_000_000_000_000_000
  }
}

@available(iOS 27.0, *)
private final class MiohIPadRealtimeBatchDetector: @unchecked Sendable {
  private let detector: MiohIPadDetector
  private let faceOnly: Bool
  private let emptyLookahead: Int

  init(
    detector: MiohIPadDetector,
    faceOnly: Bool,
    emptyLookahead: Int
  ) {
    self.detector = detector
    self.faceOnly = faceOnly
    self.emptyLookahead = min(120, max(1, emptyLookahead))
  }

  func detectBatch(
    _ frames: [MiohIPadRealtimeInputFrame],
    reuseSkipFrames: Int
  ) async throws -> [MiohIPadDetectedFrame] {
    guard !frames.isEmpty else { return [] }
    let skipFrames = min(12, max(0, reuseSkipFrames))
    var result: [MiohIPadDetectedFrame] = []
    result.reserveCapacity(frames.count)
    var windowStart = 0
    while windowStart < frames.count {
      try Task.checkCancellation()
      let windowEnd = min(frames.count, windowStart + emptyLookahead)
      result.append(
        contentsOf: try await detectWindow(
          Array(frames[windowStart..<windowEnd]),
          skipFrames: skipFrames
        )
      )
      windowStart = windowEnd
    }
    return result
  }

  /// Applies the same empty-window shortcut as the file restoration path.
  /// When both endpoints are empty, no detector or restorer work is needed for
  /// the middle frames. A non-empty endpoint falls back to the configured mask
  /// reuse stride so mosaic motion remains tracked inside the window.
  private func detectWindow(
    _ frames: [MiohIPadRealtimeInputFrame],
    skipFrames: Int
  ) async throws -> [MiohIPadDetectedFrame] {
    guard !frames.isEmpty else { return [] }
    if frames.count == 1 {
      return [
        detectedFrame(
          frames[0],
          detections: filtered(
            try await detector.detect(frames[0].pixelBuffer)
          )
        )
      ]
    }
    let first = filtered(try await detector.detect(frames[0].pixelBuffer))
    let lastIndex = frames.count - 1
    let last = filtered(
      try await detector.detect(frames[lastIndex].pixelBuffer)
    )
    if first.isEmpty, last.isEmpty {
      return frames.map { detectedFrame($0, detections: []) }
    }

    var result: [MiohIPadDetectedFrame] = []
    result.reserveCapacity(frames.count)
    var reused = first
    for frameIndex in frames.indices {
      let detections: [MiohIPadDetection]
      if frameIndex == 0 {
        detections = first
      } else if frameIndex == lastIndex {
        reused = last
        detections = last
      } else if skipFrames > 0,
        !frameIndex.isMultiple(of: skipFrames + 1)
      {
        // Keep the detector completely off on the skipped frame. Reuse the
        // preceding frame's box and binary mask as requested; CoreAI still
        // restores and composites the current video frame itself.
        detections = reused
      } else {
        reused = filtered(
          try await detector.detect(frames[frameIndex].pixelBuffer)
        )
        detections = reused
      }
      result.append(detectedFrame(frames[frameIndex], detections: detections))
    }
    return result
  }

  private func detectedFrame(
    _ frame: MiohIPadRealtimeInputFrame,
    detections: [MiohIPadDetection]
  ) -> MiohIPadDetectedFrame {
    MiohIPadDetectedFrame(
      frame: MiohIPadDecodedFrame(
        pixelBuffer: frame.pixelBuffer,
        ptsNanoseconds: frame.ptsNanoseconds
      ),
      detections: detections
    )
  }

  private func filtered(_ detections: [MiohIPadDetection])
    -> [MiohIPadDetection]
  {
    faceOnly ? detections.filter { $0.classIndex == 0 } : detections
  }

  private func interpolate(
    _ first: [MiohIPadDetection],
    _ last: [MiohIPadDetection],
    amount: Float
  ) -> [MiohIPadDetection]? {
    if first.isEmpty, last.isEmpty { return [] }
    guard first.count == last.count else { return nil }
    let firstSorted = first.sorted {
      ($0.classIndex, $0.left) < ($1.classIndex, $1.left)
    }
    let lastSorted = last.sorted {
      ($0.classIndex, $0.left) < ($1.classIndex, $1.left)
    }
    guard
      zip(firstSorted, lastSorted).allSatisfy({ lhs, rhs in
        lhs.classIndex == rhs.classIndex && boxesOverlap(lhs, rhs)
      })
    else { return nil }
    let t = max(0, min(1, amount))
    return zip(firstSorted, lastSorted).map { lhs, rhs in
      func value(_ a: Int, _ b: Int) -> Int {
        Int((Float(a) * (1 - t) + Float(b) * t).rounded())
      }
      return MiohIPadDetection(
        left: value(lhs.left, rhs.left),
        top: value(lhs.top, rhs.top),
        right: value(lhs.right, rhs.right),
        bottom: value(lhs.bottom, rhs.bottom),
        confidence: lhs.confidence * (1 - t) + rhs.confidence * t,
        classIndex: lhs.classIndex,
        detectorMask: t < 0.5 ? lhs.detectorMask : rhs.detectorMask
      )
    }
  }

  private func boxesOverlap(
    _ lhs: MiohIPadDetection,
    _ rhs: MiohIPadDetection
  ) -> Bool {
    lhs.left < rhs.right && rhs.left < lhs.right
      && lhs.top < rhs.bottom && rhs.top < lhs.bottom
  }
}

@available(iOS 27.0, *)
private final class MiohIPadFrameOrienter {
  private let transform: CGAffineTransform
  private let context = CIContext(options: [
    .workingColorSpace: NSNull(),
    .outputColorSpace: NSNull(),
    .cacheIntermediates: false,
  ])
  private let pool: CVPixelBufferPool?

  init(transform: CGAffineTransform, width: Int, height: Int) throws {
    self.transform = transform
    if transform == .identity {
      pool = nil
      return
    }
    let attributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferMetalCompatibilityKey as String: true,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    var created: CVPixelBufferPool?
    let status = CVPixelBufferPoolCreate(
      kCFAllocatorDefault,
      nil,
      attributes as CFDictionary,
      &created
    )
    guard status == kCVReturnSuccess, let created else {
      throw MiohIPadWorkerEngineError.decoder(
        "orientation pool creation returned \(status)"
      )
    }
    pool = created
  }

  func orient(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
    guard let pool else { return source }
    var output: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pool,
      &output
    )
    guard status == kCVReturnSuccess, let output else {
      throw MiohIPadWorkerEngineError.decoder(
        "orientation allocation returned \(status)"
      )
    }
    let transformed = CIImage(cvPixelBuffer: source).transformed(by: transform)
    let bounds = transformed.extent.standardized
    let normalized = transformed.transformed(
      by: CGAffineTransform(
        translationX: -bounds.minX,
        y: -bounds.minY
      )
    )
    context.render(normalized, to: output)
    CVBufferPropagateAttachments(source, output)
    return output
  }
}

#if canImport(CoreAI)
  @available(iOS 27.0, *)
  private final class MiohIPadDetector {
    private enum Backend {
      case coreAI(InferenceFunction)
      case coreML(
        model: MLModel,
        candidateOutput: String,
        prototypeOutput: String,
        compiledModelURL: URL
      )
    }

    private struct Candidate {
      let x: Float
      let y: Float
      let width: Float
      let height: Float
      let confidence: Float
      let classIndex: Int
      let coefficients: [Float]
    }

    private let backend: Backend
    private let candidateChannels: Int
    private let classCount: Int
    private let context = CIContext(options: [
      .workingColorSpace: NSNull(),
      .outputColorSpace: NSNull(),
      .cacheIntermediates: false,
    ])
    private let pool: CVPixelBufferPool
    private var normalizationScratch = [Float](
      repeating: 0,
      count: 640 * 640 * 4
    )

    init(function: InferenceFunction, candidateChannels: Int) {
      precondition(candidateChannels == 37 || candidateChannels == 38)
      backend = .coreAI(function)
      self.candidateChannels = candidateChannels
      classCount = candidateChannels - 4 - 32
      pool = Self.makePixelBufferPool()
    }

    init(
      coreMLModel: MLModel,
      compiledModelURL: URL,
      candidateChannels: Int
    ) throws {
      precondition(candidateChannels == 37 || candidateChannels == 38)
      var candidateOutput: String?
      var prototypeOutput: String?
      for (name, description) in coreMLModel.modelDescription.outputDescriptionsByName {
        guard let shape = description.multiArrayConstraint?.shape.map(\.intValue)
        else { continue }
        if shape == [1, candidateChannels, 8400] {
          candidateOutput = name
        } else if shape == [1, 32, 160, 160] {
          prototypeOutput = name
        }
      }
      guard let candidateOutput, let prototypeOutput else {
        throw MiohIPadWorkerEngineError.detector(
          "Core ML raw output contract changed"
        )
      }
      backend = .coreML(
        model: coreMLModel,
        candidateOutput: candidateOutput,
        prototypeOutput: prototypeOutput,
        compiledModelURL: compiledModelURL
      )
      self.candidateChannels = candidateChannels
      classCount = candidateChannels - 4 - 32
      pool = Self.makePixelBufferPool()
    }

    private static func makePixelBufferPool() -> CVPixelBufferPool {
      var created: CVPixelBufferPool?
      let attributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferWidthKey as String: 640,
        kCVPixelBufferHeightKey as String: 640,
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
      let status = CVPixelBufferPoolCreate(
        kCFAllocatorDefault,
        nil,
        attributes as CFDictionary,
        &created
      )
      precondition(status == kCVReturnSuccess && created != nil)
      return created!
    }

    func detect(_ source: CVPixelBuffer) async throws -> [MiohIPadDetection] {
      let (letterboxed, scale, padX, padY) = try letterbox(source)
      let candidates: [Float]
      let prototypes: [Float]
      switch backend {
      case .coreAI(let function):
        let input = try normalizedNCHW(letterboxed)
        var outputs = try await function.run(inputs: ["image": input])
        guard let candidateArray = outputs.remove("candidates")?.ndArray,
          let prototypeArray = outputs.remove("prototypes")?.ndArray
        else {
          throw MiohIPadWorkerEngineError.detector("raw outputs are missing")
        }
        candidates = try readFloat16(
          candidateArray,
          expectedShape: [1, candidateChannels, 8400]
        )
        prototypes = try readFloat16(
          prototypeArray,
          expectedShape: [1, 32, 160, 160]
        )
      case .coreML(
        let model,
        let candidateOutput,
        let prototypeOutput,
        _
      ):
        let input = try MLDictionaryFeatureProvider(dictionary: [
          "image": MLFeatureValue(pixelBuffer: letterboxed)
        ])
        let outputs = try await model.prediction(from: input)
        guard
          let candidateArray = outputs.featureValue(
            for: candidateOutput
          )?.multiArrayValue,
          let prototypeArray = outputs.featureValue(
            for: prototypeOutput
          )?.multiArrayValue
        else {
          throw MiohIPadWorkerEngineError.detector(
            "Core ML raw outputs are missing"
          )
        }
        candidates = try readMultiArray(
          candidateArray,
          expectedShape: [1, candidateChannels, 8400]
        )
        prototypes = try readMultiArray(
          prototypeArray,
          expectedShape: [1, 32, 160, 160]
        )
      }
      var decoded: [Candidate] = []
      decoded.reserveCapacity(64)
      for index in 0..<8400 {
        var classIndex = 0
        var confidence = -Float.infinity
        for candidateClass in 0..<classCount {
          let value = candidates[(4 + candidateClass) * 8400 + index]
          if value > confidence {
            confidence = value
            classIndex = candidateClass
          }
        }
        guard confidence >= 0.25 else { continue }
        var coefficients: [Float] = []
        coefficients.reserveCapacity(32)
        for channel in 0..<32 {
          coefficients.append(
            candidates[(4 + classCount + channel) * 8400 + index]
          )
        }
        decoded.append(
          Candidate(
            x: candidates[index],
            y: candidates[8400 + index],
            width: candidates[2 * 8400 + index],
            height: candidates[3 * 8400 + index],
            confidence: confidence,
            classIndex: classIndex,
            coefficients: coefficients
          )
        )
      }
      let sourceWidth = Float(CVPixelBufferGetWidth(source))
      let sourceHeight = Float(CVPixelBufferGetHeight(source))
      let width = Int(sourceWidth)
      let height = Int(sourceHeight)
      return nonMaximumSuppression(decoded, threshold: 0.7).compactMap {
        chosen in
        let detectorLeft = chosen.x - chosen.width * 0.5
        let detectorTop = chosen.y - chosen.height * 0.5
        let detectorRight = chosen.x + chosen.width * 0.5
        let detectorBottom = chosen.y + chosen.height * 0.5
        let left = Int(max(0, min(sourceWidth, (detectorLeft - padX) / scale)))
        let top = Int(max(0, min(sourceHeight, (detectorTop - padY) / scale)))
        let right = Int(max(0, min(sourceWidth, (detectorRight - padX) / scale)))
        let bottom = Int(max(0, min(sourceHeight, (detectorBottom - padY) / scale)))
        guard right > left, bottom > top else { return nil }
        let mask = Self.makeDetectorMask(
          coefficients: chosen.coefficients,
          prototypes: prototypes,
          box: (detectorLeft, detectorTop, detectorRight, detectorBottom)
        )
        guard mask.containsSetPixel else { return nil }
        return MiohIPadDetection(
          left: min(width - 1, left),
          top: min(height - 1, top),
          right: min(width - 1, right),
          bottom: min(height - 1, bottom),
          confidence: chosen.confidence,
          classIndex: chosen.classIndex,
          detectorMask: mask
        )
      }
    }

    private func letterbox(_ source: CVPixelBuffer) throws
      -> (CVPixelBuffer, Float, Float, Float)
    {
      let sourceWidth = Float(CVPixelBufferGetWidth(source))
      let sourceHeight = Float(CVPixelBufferGetHeight(source))
      let scale = min(640 / sourceWidth, 640 / sourceHeight)
      let renderedWidth = sourceWidth * scale
      let renderedHeight = sourceHeight * scale
      let padX = (640 - renderedWidth) * 0.5
      let padY = (640 - renderedHeight) * 0.5
      var output: CVPixelBuffer?
      let status = CVPixelBufferPoolCreatePixelBuffer(
        kCFAllocatorDefault,
        pool,
        &output
      )
      guard status == kCVReturnSuccess, let output else {
        throw MiohIPadWorkerEngineError.detector(
          "letterbox allocation returned \(status)"
        )
      }
      let background = CIImage(
        color: CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255)
      ).cropped(to: CGRect(x: 0, y: 0, width: 640, height: 640))
      let image = CIImage(cvPixelBuffer: source)
        .transformed(
          by: CGAffineTransform(scaleX: CGFloat(scale), y: CGFloat(scale))
        )
        .transformed(
          by: CGAffineTransform(
            translationX: CGFloat(padX),
            y: CGFloat(padY)
          )
        )
      context.render(image.composited(over: background), to: output)
      return (output, scale, padX, padY)
    }

    private func normalizedNCHW(_ pixelBuffer: CVPixelBuffer) throws -> NDArray {
      CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
      defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
      guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        throw MiohIPadWorkerEngineError.detector("input base address")
      }
      let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
      var array = NDArray(shape: [1, 3, 640, 640], scalarType: .float16)
      let view = array.mutableView(as: Float16.self)
      try view.withUnsafeMutablePointer { destination, _, _ in
        try normalizationScratch.withUnsafeMutableBytes { scratch in
          guard let scratchBase = scratch.baseAddress else {
            throw MiohIPadWorkerEngineError.detector("normalization scratch")
          }
          let plane = 640 * 640
          let planeBytes = plane * MemoryLayout<Float>.stride
          var source = vImage_Buffer(
            data: base,
            height: 640,
            width: 640,
            rowBytes: rowBytes
          )
          var blue = vImage_Buffer(
            data: scratchBase,
            height: 640,
            width: 640,
            rowBytes: 640 * MemoryLayout<Float>.stride
          )
          var green = vImage_Buffer(
            data: scratchBase.advanced(by: planeBytes),
            height: 640,
            width: 640,
            rowBytes: 640 * MemoryLayout<Float>.stride
          )
          var red = vImage_Buffer(
            data: scratchBase.advanced(by: 2 * planeBytes),
            height: 640,
            width: 640,
            rowBytes: 640 * MemoryLayout<Float>.stride
          )
          var alpha = vImage_Buffer(
            data: scratchBase.advanced(by: 3 * planeBytes),
            height: 640,
            width: 640,
            rowBytes: 640 * MemoryLayout<Float>.stride
          )
          var maximum: [Float] = [1, 1, 1, 1]
          var minimum: [Float] = [0, 0, 0, 0]
          let conversion = vImageConvert_ARGB8888toPlanarF(
            &source,
            &blue,
            &green,
            &red,
            &alpha,
            &maximum,
            &minimum,
            vImage_Flags(kvImageNoFlags)
          )
          guard conversion == kvImageNoError else {
            throw MiohIPadWorkerEngineError.detector(
              "BGRA normalization returned \(conversion)"
            )
          }
          for (channel, sourcePlaneValue) in [red, green, blue].enumerated() {
            var sourcePlane = sourcePlaneValue
            var destinationPlane = vImage_Buffer(
              data: destination.advanced(by: channel * plane),
              height: 640,
              width: 640,
              rowBytes: 640 * MemoryLayout<Float16>.stride
            )
            let halfConversion = vImageConvert_PlanarFtoPlanar16F(
              &sourcePlane,
              &destinationPlane,
              vImage_Flags(kvImageNoFlags)
            )
            guard halfConversion == kvImageNoError else {
              throw MiohIPadWorkerEngineError.detector(
                "FP16 normalization returned \(halfConversion)"
              )
            }
          }
        }
      }
      return array
    }

    private func readFloat16(_ array: NDArray, expectedShape: [Int]) throws
      -> [Float]
    {
      let view = array.view(as: Float16.self)
      guard view.isContiguous else {
        throw MiohIPadWorkerEngineError.detector("output is not contiguous")
      }
      return try view.withUnsafePointer { pointer, shape, _ in
        let actualShape = (0..<shape.count).map { shape[$0] }
        guard actualShape == expectedShape else {
          throw MiohIPadWorkerEngineError.detector(
            "output shape \(actualShape), expected \(expectedShape)"
          )
        }
        let count = expectedShape.reduce(1, *)
        var result = [Float](repeating: 0, count: count)
        let conversion = result.withUnsafeMutableBytes { destinationBytes in
          var source = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: pointer),
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float16>.stride
          )
          var destination = vImage_Buffer(
            data: destinationBytes.baseAddress!,
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.stride
          )
          return vImageConvert_Planar16FtoPlanarF(
            &source,
            &destination,
            vImage_Flags(kvImageNoFlags)
          )
        }
        guard conversion == kvImageNoError else {
          throw MiohIPadWorkerEngineError.detector(
            "output conversion returned \(conversion)"
          )
        }
        return result
      }
    }

    private func readMultiArray(
      _ array: MLMultiArray,
      expectedShape: [Int]
    ) throws -> [Float] {
      let actualShape = array.shape.map(\.intValue)
      guard actualShape == expectedShape else {
        throw MiohIPadWorkerEngineError.detector(
          "Core ML output shape \(actualShape), expected \(expectedShape)"
        )
      }
      var expectedStride = 1
      for index in expectedShape.indices.reversed() {
        guard array.strides[index].intValue == expectedStride else {
          throw MiohIPadWorkerEngineError.detector(
            "Core ML output is not contiguous"
          )
        }
        expectedStride *= expectedShape[index]
      }
      let count = expectedShape.reduce(1, *)
      switch array.dataType {
      case .float32:
        let source = array.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: source, count: count))
      case .float16:
        let source = array.dataPointer.assumingMemoryBound(to: Float16.self)
        var result = [Float](repeating: 0, count: count)
        for index in 0..<count { result[index] = Float(source[index]) }
        return result
      case .double:
        let source = array.dataPointer.assumingMemoryBound(to: Double.self)
        var result = [Float](repeating: 0, count: count)
        for index in 0..<count { result[index] = Float(source[index]) }
        return result
      default:
        throw MiohIPadWorkerEngineError.detector(
          "unsupported Core ML output type \(array.dataType.rawValue)"
        )
      }
    }

    private func nonMaximumSuppression(
      _ candidates: [Candidate],
      threshold: Float
    ) -> [Candidate] {
      let sorted = candidates.sorted { $0.confidence > $1.confidence }
      var kept: [Candidate] = []
      for candidate in sorted {
        if kept.contains(where: {
          $0.classIndex == candidate.classIndex
            && Self.iou($0, candidate) > threshold
        }) {
          continue
        }
        kept.append(candidate)
        if kept.count >= 100 { break }
      }
      return kept
    }

    private static func iou(_ first: Candidate, _ second: Candidate) -> Float {
      let firstLeft = first.x - first.width * 0.5
      let firstTop = first.y - first.height * 0.5
      let secondLeft = second.x - second.width * 0.5
      let secondTop = second.y - second.height * 0.5
      let intersectionWidth = max(
        0,
        min(firstLeft + first.width, secondLeft + second.width)
          - max(firstLeft, secondLeft)
      )
      let intersectionHeight = max(
        0,
        min(firstTop + first.height, secondTop + second.height)
          - max(firstTop, secondTop)
      )
      let intersection = intersectionWidth * intersectionHeight
      return intersection
        / max(
          first.width * first.height + second.width * second.height - intersection,
          0.000001
        )
    }

    private static func makeDetectorMask(
      coefficients: [Float],
      prototypes: [Float],
      box: (Float, Float, Float, Float)
    ) -> MiohIPadBinaryMask {
      let prototypeSize = 160
      let detectorSize = 640
      let plane = prototypeSize * prototypeSize
      var logits = [Float](repeating: 0, count: plane)
      guard coefficients.count == 32, prototypes.count == 32 * plane else {
        return MiohIPadBinaryMask(
          pixelCount: detectorSize * detectorSize
        )
      }
      coefficients.withUnsafeBufferPointer { coefficientsBuffer in
        prototypes.withUnsafeBufferPointer { prototypesBuffer in
          logits.withUnsafeMutableBufferPointer { logitsBuffer in
            vDSP_mmul(
              coefficientsBuffer.baseAddress!, 1,
              prototypesBuffer.baseAddress!, 1,
              logitsBuffer.baseAddress!, 1,
              1, vDSP_Length(plane), 32
            )
          }
        }
      }
      var mask = MiohIPadBinaryMask(
        pixelCount: detectorSize * detectorSize
      )
      let left = max(0, Int(floor(box.0)))
      let top = max(0, Int(floor(box.1)))
      let right = min(detectorSize, Int(ceil(box.2)))
      let bottom = min(detectorSize, Int(ceil(box.3)))
      guard left < right, top < bottom else { return mask }
      for y in top..<bottom {
        let sourceY = (Float(y) + 0.5) / 4 - 0.5
        let y0 = max(0, min(prototypeSize - 1, Int(floor(sourceY))))
        let y1 = min(prototypeSize - 1, y0 + 1)
        let fy = max(0, min(1, sourceY - Float(y0)))
        for x in left..<right {
          let sourceX = (Float(x) + 0.5) / 4 - 0.5
          let x0 = max(0, min(prototypeSize - 1, Int(floor(sourceX))))
          let x1 = min(prototypeSize - 1, x0 + 1)
          let fx = max(0, min(1, sourceX - Float(x0)))
          let upper =
            logits[y0 * prototypeSize + x0] * (1 - fx)
            + logits[y0 * prototypeSize + x1] * fx
          let lower =
            logits[y1 * prototypeSize + x0] * (1 - fx)
            + logits[y1 * prototypeSize + x1] * fx
          if upper * (1 - fy) + lower * fy > 0 {
            mask.set(y * detectorSize + x)
          }
        }
      }
      return mask
    }
  }
#endif
