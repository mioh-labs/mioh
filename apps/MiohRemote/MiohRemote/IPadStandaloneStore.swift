@preconcurrency import AVFoundation
import CoreVideo
import Foundation
import MiohRemoteKit
import QuartzCore
import UIKit
import VideoToolbox

enum IPadHLSQualityPreference: String, CaseIterable, Identifiable {
  case automatic
  case p1080
  case p720
  case p480

  var id: String { rawValue }

  var label: String {
    switch self {
    case .automatic: "自動（最高画質）"
    case .p1080: "1080p固定"
    case .p720: "720p固定"
    case .p480: "480p固定"
    }
  }

  var targetHeight: Int? {
    switch self {
    case .automatic: nil
    case .p1080: 1_080
    case .p720: 720
    case .p480: 480
    }
  }
}

enum IPadHLSStreamingMode: String, CaseIterable, Identifiable {
  case fast
  case safariCompatible

  var id: String { rawValue }

  var label: String {
    switch self {
    case .fast: "高速"
    case .safariCompatible: "Safari互換（429・暗号化HLS対応）"
    }
  }

  var allowsAES128HLS: Bool { self == .safariCompatible }
}

enum IPadRealtimeFrameRateMode: String, CaseIterable, Identifiable {
  case source
  case fps24
  case automatic

  var id: String { rawValue }

  var label: String {
    switch self {
    case .source: "元フレームレート維持（通常29.97fps）"
    case .fps24: "24fps固定"
    case .automatic: "自動（処理が遅いとき24fps）"
    }
  }
}

struct IPadRealtimeEmergencyFPSPolicy {
  private(set) var isEnabled = false
  private var slowWindows = 0
  private var healthyWindows = 0

  mutating func update(
    wallSeconds: Double,
    mediaSeconds: Double,
    bufferedSeconds: Double,
    targetSeconds: Double,
    thermalState: ProcessInfo.ThermalState
  ) -> Bool? {
    guard wallSeconds.isFinite, mediaSeconds.isFinite, mediaSeconds > 0 else {
      return nil
    }
    let rtf = wallSeconds / mediaSeconds
    let lowBuffer = bufferedSeconds < max(6, targetSeconds * 0.60)
    let thermallyConstrained = thermalState == .serious || thermalState == .critical
    if !isEnabled {
      slowWindows =
        rtf >= 0.98 && lowBuffer || thermallyConstrained
        ? slowWindows + 1 : 0
      if slowWindows >= 2 {
        isEnabled = true
        slowWindows = 0
        healthyWindows = 0
        return true
      }
    } else {
      let recovered =
        rtf <= 0.82
        && bufferedSeconds >= max(8, targetSeconds * 0.75)
        && !thermallyConstrained
      healthyWindows = recovered ? healthyWindows + 1 : 0
      if healthyWindows >= 4 {
        isEnabled = false
        healthyWindows = 0
        slowWindows = 0
        return false
      }
    }
    return nil
  }
}

struct IPadRealtimePreviewConfiguration: Sendable {
  static let sharedRootIdentifier = "ipad-realtime-preview"
  static let realtimeTemporalFrames = 48
  static let minimumTemporalOverlap = 6

  let inputURL: URL
  let resolvedMediaSource: IPadResolvedMediaSource?
  let hlsResourceLoader: (any IPadHLSResourceLoading)?
  let hlsStreamingMode: IPadHLSStreamingMode
  let hlsQualityPreference: IPadHLSQualityPreference
  let detectionMaskReuseSkipFrames: Int
  let realtimeFrameRateMode: IPadRealtimeFrameRateMode
  let maximumFrameRate: Int?
  let parallelRestorationLanes: Int
  let durationNanoseconds: Int64
  let inputByteCount: Int64
  let inputExtension: String
  let inputRangeValidator: String?
  let prepareInputForSeek: (@Sendable () -> Void)?
  let streamingMetrics: (@Sendable () async -> IPadSFTPStreamingMetrics)?
  let options: MiohClusterRestorationOptions
  let bufferLimitSeconds: Double
  let keepScreenAwake: Bool

  var durationSeconds: Double {
    Double(durationNanoseconds) / 1_000_000_000
  }

  func request(
    coreStartNanoseconds: Int64,
    coreEndNanoseconds: Int64,
    contextNanoseconds: Int64? = nil,
    mediaStartNanoseconds: Int64 = 0,
    mediaEndNanoseconds: Int64? = nil,
    inputByteCount: Int64? = nil,
    inputExtension: String? = nil,
    sourceFPSNumerator: Int? = nil,
    sourceFPSDenominator: Int? = nil
  ) throws -> MiohClusterJobRequest {
    let mediaStart = max(0, mediaStartNanoseconds)
    let mediaEnd = mediaEndNanoseconds ?? durationNanoseconds
    guard mediaEnd > mediaStart else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "入力区間の時間情報が不正です"
      )
    }
    let start = min(max(mediaStart, coreStartNanoseconds), mediaEnd - 1)
    let end = min(max(start + 1, coreEndNanoseconds), mediaEnd)
    let requestOptions: MiohClusterRestorationOptions
    if let maximumFrameRate, let sourceFPSNumerator, let sourceFPSDenominator {
      requestOptions = replacingTargetFrameRate(
        MiohIPadSourceFrameRate.maximumRate(
          wholeFPS: maximumFrameRate,
          sourceNumerator: sourceFPSNumerator,
          sourceDenominator: sourceFPSDenominator
        )
      )
    } else {
      requestOptions = options
    }
    let resolvedContext =
      contextNanoseconds
      ?? {
        guard requestOptions.temporalOverlap > 0 else { return 0 }
        let numerator =
          requestOptions.targetFPSNumerator ?? sourceFPSNumerator ?? 24
        let denominator =
          requestOptions.targetFPSDenominator ?? sourceFPSDenominator ?? 1
        guard numerator > 0, denominator > 0 else { return 0 }
        // Convert overlap frames with the actual post-gate frame rate. The old
        // fixed /24 conversion decoded 25% too much context for 29.97fps input.
        let nanoseconds = Int64(
          ceil(
            Double(requestOptions.temporalOverlap)
              * 1_000_000_000 * Double(denominator)
              / Double(numerator)
          )
        )
        return min(1_000_000_000, max(1, nanoseconds))
      }()
    let now = Date()
    return MiohClusterJobRequest(
      jobID: UUID(),
      attemptID: UUID(),
      leaseID: UUID(),
      coordinatorNodeID: UUID(),
      sharedRootIdentifier: Self.sharedRootIdentifier,
      inputByteCount: inputByteCount ?? self.inputByteCount,
      inputSHA256: inputRangeValidator ?? String(repeating: "0", count: 64),
      inputRelativePath: try MiohClusterRelativePath(
        validating: "input.\(inputExtension ?? self.inputExtension)"
      ),
      outputRelativePath: try MiohClusterRelativePath(
        validating: "preview.mp4"
      ),
      mediaRange: MiohClusterMediaRange(
        decodeStartNanoseconds: max(mediaStart, start - resolvedContext),
        decodeEndNanoseconds: min(
          mediaEnd,
          end + resolvedContext
        ),
        coreStartNanoseconds: start,
        coreEndNanoseconds: end,
        leadingOverlapFrames: 0,
        trailingOverlapFrames: 0
      ),
      options: requestOptions,
      createdAt: now,
      leaseExpiresAt: now.addingTimeInterval(24 * 60 * 60)
    )
  }

  private func replacingTargetFrameRate(
    _ rate: (numerator: Int, denominator: Int)?
  ) -> MiohClusterRestorationOptions {
    MiohClusterRestorationOptions(
      restorationModelIdentifier: options.restorationModelIdentifier,
      restorationAssetSHA256: options.restorationAssetSHA256,
      detectorModelIdentifier: options.detectorModelIdentifier,
      detectorAssetSHA256: options.detectorAssetSHA256,
      restorationClipLength: options.restorationClipLength,
      temporalOverlap: options.temporalOverlap,
      crossfade: options.crossfade,
      detectionEmptyLookahead: options.detectionEmptyLookahead,
      detectFaceMosaics: options.detectFaceMosaics,
      detectionMaskReuseSkipFrames: options.detectionMaskReuseSkipFrames,
      blendFeather: options.blendFeather,
      sharpenStrength: options.sharpenStrength,
      detailBoost: options.detailBoost,
      textureMix: options.textureMix,
      smoothStrength: options.smoothStrength,
      effectUpscale: options.effectUpscale,
      roiEnhancerModelIdentifier: options.roiEnhancerModelIdentifier,
      roiEnhancerAssetSHA256: options.roiEnhancerAssetSHA256,
      roiEnhancerStrength: options.roiEnhancerStrength,
      roiEnhancerScale: options.roiEnhancerScale,
      videoCodec: options.videoCodec,
      bitrateMultiplier: options.bitrateMultiplier,
      mp4FastStart: options.mp4FastStart,
      targetFPSNumerator: rate?.numerator,
      targetFPSDenominator: rate?.denominator
    )
  }
}

private func makeIPadMediaAsset(
  url: URL,
  source: IPadResolvedMediaSource?
) -> AVURLAsset {
  guard let context = source?.requestContext else {
    return AVURLAsset(url: url)
  }
  var options: [String: Any] = [:]
  let cookies = context.httpCookies(
    relevantTo: (source?.requestURLs ?? []) + [url]
  )
  if !cookies.isEmpty {
    options[AVURLAssetHTTPCookiesKey] = cookies
  }
  if let userAgent = context.userAgent {
    options[AVURLAssetHTTPUserAgentKey] = userAgent
  }
  return AVURLAsset(url: url, options: options.isEmpty ? nil : options)
}

private final class SendableExportSession: @unchecked Sendable {
  let value: AVAssetExportSession

  init(_ value: AVAssetExportSession) {
    self.value = value
  }
}

private enum IPadProgressiveDownloadError: LocalizedError {
  case unsafeURL
  case insecureRedirect
  case tooManyRedirects
  case invalidResponse
  case invalidHTTPStatus(Int)
  case fileTooLarge
  case insufficientStorage
  case emptyResponse
  case writeFailed(String)

  var errorDescription: String? {
    switch self {
    case .unsafeURL:
      "動画URLが安全なHTTP/HTTPS URLではありません。"
    case .insecureRedirect:
      "HTTPSからHTTPへの安全でないリダイレクトは使用できません。"
    case .tooManyRedirects:
      "動画URLのリダイレクト回数が上限を超えました。"
    case .invalidResponse:
      "動画URLから有効なHTTP応答を取得できませんでした。"
    case .invalidHTTPStatus(let status):
      "動画サーバーがHTTP \(status)を返しました。"
    case .fileTooLarge:
      "動画URLのファイルが大きすぎます。"
    case .insufficientStorage:
      "動画URLを保存する空き容量が不足しています。"
    case .emptyResponse:
      "動画URLから空のファイルが返されました。"
    case .writeFailed(let detail):
      "動画URLの一時ファイルを保存できませんでした: \(detail)"
    }
  }
}

private final class IPadBoundedProgressiveDownloader: NSObject, @unchecked Sendable {
  private let destinationURL: URL
  private let maximumBytes: Int64
  private let maximumRedirectCount: Int
  private let timeout: TimeInterval
  private let originalURL: URL
  private let resolutionPolicy: IPadMediaURLResolutionPolicy
  private let requestContext: IPadMediaRequestContext?
  private let lock = NSLock()
  private var continuation: CheckedContinuation<URL, Error>?
  private var session: URLSession?
  private var task: URLSessionDataTask?
  private var fileHandle: FileHandle?
  private var receivedBytes: Int64 = 0
  private var redirectCount = 0
  private var acceptedResponse = false
  private var cancellationRequested = false
  private var didFinish = false

  init(
    destinationURL: URL,
    maximumBytes: Int64,
    maximumRedirectCount: Int,
    timeout: TimeInterval,
    originalURL: URL,
    resolutionPolicy: IPadMediaURLResolutionPolicy,
    requestContext: IPadMediaRequestContext?
  ) {
    self.destinationURL = destinationURL
    self.maximumBytes = maximumBytes
    self.maximumRedirectCount = maximumRedirectCount
    self.timeout = timeout
    self.originalURL = originalURL
    self.resolutionPolicy = resolutionPolicy
    self.requestContext = requestContext
  }

  func start(_ request: URLRequest) async throws -> URL {
    try Task.checkCancellation()
    guard let requestURL = request.url, Self.isSafeHTTPURL(requestURL) else {
      throw IPadProgressiveDownloadError.unsafeURL
    }
    if !IPadMediaURLResolver.isURL(requestURL, allowedBy: resolutionPolicy) {
      throw IPadProgressiveDownloadError.unsafeURL
    }

    guard
      FileManager.default.createFile(
        atPath: destinationURL.path,
        contents: nil
      )
    else {
      throw IPadProgressiveDownloadError.writeFailed("一時ファイルを作成できません")
    }

    let handle: FileHandle
    do {
      handle = try FileHandle(forWritingTo: destinationURL)
    } catch {
      try? FileManager.default.removeItem(at: destinationURL)
      throw IPadProgressiveDownloadError.writeFailed(error.localizedDescription)
    }

    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = 24 * 60 * 60
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpShouldSetCookies = false
        configuration.urlCredentialStorage = nil
        configuration.waitsForConnectivity = false

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInitiated
        let session = URLSession(
          configuration: configuration,
          delegate: self,
          delegateQueue: queue
        )
        let task = session.dataTask(with: request)

        lock.lock()
        if cancellationRequested {
          didFinish = true
          lock.unlock()
          try? handle.close()
          try? FileManager.default.removeItem(at: destinationURL)
          session.invalidateAndCancel()
          continuation.resume(throwing: CancellationError())
          return
        }
        self.continuation = continuation
        self.session = session
        self.task = task
        fileHandle = handle
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
    let handle = fileHandle
    self.continuation = nil
    self.task = nil
    self.session = nil
    fileHandle = nil
    lock.unlock()

    task?.cancel()
    session?.invalidateAndCancel()
    try? handle?.close()
    try? FileManager.default.removeItem(at: destinationURL)
    continuation.resume(throwing: CancellationError())
  }

  private func finish(_ result: Result<URL, Error>) {
    lock.lock()
    guard !didFinish, let continuation else {
      lock.unlock()
      return
    }
    didFinish = true
    let task = task
    let session = session
    let handle = fileHandle
    self.continuation = nil
    self.task = nil
    self.session = nil
    fileHandle = nil
    lock.unlock()

    task?.cancel()
    session?.invalidateAndCancel()

    var finalResult = result
    do {
      try handle?.close()
      if case .success = result {
        let fileSize = Int64(
          (try destinationURL.resourceValues(forKeys: [.fileSizeKey]).fileSize)
            ?? 0
        )
        guard fileSize > 0 else {
          throw IPadProgressiveDownloadError.emptyResponse
        }
        guard fileSize <= maximumBytes else {
          throw IPadProgressiveDownloadError.fileTooLarge
        }
      }
    } catch {
      finalResult = .failure(error)
    }

    if case .failure = finalResult {
      try? FileManager.default.removeItem(at: destinationURL)
    }
    continuation.resume(with: finalResult)
  }

  private static func isSafeHTTPURL(_ url: URL) -> Bool {
    guard let components = URLComponents(url: url, resolvingAgainstBaseURL: true),
      let scheme = components.scheme?.lowercased(),
      scheme == "http" || scheme == "https",
      components.user == nil,
      components.password == nil,
      let host = components.host,
      !host.isEmpty
    else { return false }
    return true
  }

  private func validate(_ response: HTTPURLResponse) -> Error? {
    if let error = IPadMediaURLResolver.interactionChallengeError(
      response: response
    ) {
      return error
    }
    guard (200...299).contains(response.statusCode) else {
      return IPadProgressiveDownloadError.invalidHTTPStatus(response.statusCode)
    }
    guard let finalURL = response.url, Self.isSafeHTTPURL(finalURL) else {
      return IPadProgressiveDownloadError.unsafeURL
    }
    if !IPadMediaURLResolver.isURL(finalURL, allowedBy: resolutionPolicy) {
      return IPadProgressiveDownloadError.unsafeURL
    }
    if originalURL.scheme?.lowercased() == "https",
      finalURL.scheme?.lowercased() == "http"
    {
      return IPadProgressiveDownloadError.insecureRedirect
    }
    if response.expectedContentLength > maximumBytes {
      return IPadProgressiveDownloadError.fileTooLarge
    }
    if let rawContentLength = response.value(forHTTPHeaderField: "Content-Length"),
      let contentLength = Int64(rawContentLength.trimmingCharacters(in: .whitespaces)),
      contentLength > maximumBytes
    {
      return IPadProgressiveDownloadError.fileTooLarge
    }
    return nil
  }

  private func markAcceptedResponse() {
    lock.lock()
    acceptedResponse = true
    lock.unlock()
  }
}

extension IPadBoundedProgressiveDownloader: URLSessionDataDelegate {
  func urlSession(
    _: URLSession,
    dataTask _: URLSessionDataTask,
    didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    guard let httpResponse = response as? HTTPURLResponse else {
      completionHandler(.cancel)
      finish(.failure(IPadProgressiveDownloadError.invalidResponse))
      return
    }
    Task {
      if let error = validate(httpResponse) {
        completionHandler(.cancel)
        finish(.failure(error))
        return
      }
      await requestContext?.updateCookies(from: httpResponse)
      markAcceptedResponse()
      completionHandler(.allow)
    }
  }

  func urlSession(_: URLSession, dataTask _: URLSessionDataTask, didReceive data: Data) {
    lock.lock()
    let byteCount = Int64(data.count)
    let wouldOverflow =
      !didFinish
      && (receivedBytes > maximumBytes || byteCount > maximumBytes - receivedBytes)
    guard !didFinish, !wouldOverflow, let fileHandle else {
      lock.unlock()
      if wouldOverflow {
        finish(.failure(IPadProgressiveDownloadError.fileTooLarge))
      }
      return
    }
    do {
      try fileHandle.write(contentsOf: data)
      receivedBytes += byteCount
      lock.unlock()
    } catch {
      lock.unlock()
      finish(.failure(IPadProgressiveDownloadError.writeFailed(error.localizedDescription)))
    }
  }

  func urlSession(
    _: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    guard let destination = request.url, Self.isSafeHTTPURL(destination) else {
      completionHandler(nil)
      finish(.failure(IPadProgressiveDownloadError.unsafeURL))
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
    if !IPadMediaURLResolver.isURL(destination, allowedBy: resolutionPolicy) {
      completionHandler(nil)
      finish(.failure(IPadProgressiveDownloadError.unsafeURL))
      return
    }
    if let sourceURL = task.currentRequest?.url,
      IPadMediaURLResolver.isDisallowedDiscoveredLocalURL(
        destination,
        pageURL: sourceURL
      )
    {
      completionHandler(nil)
      finish(.failure(IPadProgressiveDownloadError.unsafeURL))
      return
    }
    if task.currentRequest?.url?.scheme?.lowercased() == "https",
      destination.scheme?.lowercased() == "http"
    {
      completionHandler(nil)
      finish(.failure(IPadProgressiveDownloadError.insecureRedirect))
      return
    }

    lock.lock()
    redirectCount += 1
    let tooManyRedirects = redirectCount > maximumRedirectCount
    lock.unlock()
    if tooManyRedirects {
      completionHandler(nil)
      finish(.failure(IPadProgressiveDownloadError.tooManyRedirects))
      return
    }

    Task {
      await requestContext?.updateCookies(from: response)
      var sanitizedRequest = request
      sanitizedRequest.httpShouldHandleCookies = false
      sanitizedRequest.setValue(nil, forHTTPHeaderField: "Authorization")
      sanitizedRequest.setValue(nil, forHTTPHeaderField: "Proxy-Authorization")
      sanitizedRequest.setValue(nil, forHTTPHeaderField: "Cookie")
      requestContext?.applying(to: &sanitizedRequest)
      completionHandler(sanitizedRequest)
    }
  }

  func urlSession(
    _: URLSession,
    task _: URLSessionTask,
    didReceive challenge: URLAuthenticationChallenge,
    completionHandler:
      @escaping (
        URLSession.AuthChallengeDisposition,
        URLCredential?
      ) -> Void
  ) {
    if challenge.protectionSpace.authenticationMethod
      == NSURLAuthenticationMethodServerTrust
    {
      completionHandler(.performDefaultHandling, nil)
    } else {
      completionHandler(.cancelAuthenticationChallenge, nil)
    }
  }

  func urlSession(_: URLSession, task _: URLSessionTask, didCompleteWithError error: Error?) {
    if let error {
      lock.lock()
      let wasCancelled = cancellationRequested
      lock.unlock()
      finish(.failure(wasCancelled ? CancellationError() : error))
      return
    }
    lock.lock()
    let acceptedResponse = acceptedResponse
    let receivedBytes = receivedBytes
    lock.unlock()
    guard acceptedResponse else {
      finish(.failure(IPadProgressiveDownloadError.invalidResponse))
      return
    }
    guard receivedBytes > 0 else {
      finish(.failure(IPadProgressiveDownloadError.emptyResponse))
      return
    }
    finish(.success(destinationURL))
  }
}

struct IPadStandaloneLogEntry: Identifiable, Equatable {
  enum Level: String {
    case info = "情報"
    case success = "完了"
    case warning = "注意"
    case error = "エラー"
  }

  let id: UUID
  let timestamp: Date
  let level: Level
  let message: String

  init(
    id: UUID = UUID(),
    timestamp: Date = Date(),
    level: Level = .info,
    message: String
  ) {
    self.id = id
    self.timestamp = timestamp
    self.level = level
    self.message = message
  }
}

struct IPadFrameRateOption: Identifiable, Hashable {
  let numerator: Int
  let denominator: Int

  var id: String { "\(numerator)/\(denominator)" }
  var value: Double { Double(numerator) / Double(denominator) }
  var label: String {
    denominator == 1
      ? "\(numerator)"
      : String(format: "%.3f", value)
  }
}

private actor IPadStandaloneParallelProgress {
  private let weights: [Double]
  private var values: [Double]

  init(weights: [Int]) {
    let bounded = weights.map { Double(max(1, $0)) }
    let total = max(1, bounded.reduce(0, +))
    self.weights = bounded.map { $0 / total }
    values = Array(repeating: 0, count: bounded.count)
  }

  func update(lane: Int, value: Double) -> Double {
    guard values.indices.contains(lane) else { return 0 }
    values[lane] = min(1, max(0, value))
    return zip(values, weights).reduce(0) { result, pair in
      result + pair.0 * pair.1
    }
  }
}

@MainActor
final class IPadStandaloneStore: ObservableObject {
  enum RunState: Equatable {
    case idle
    case preparing
    case restoring
    case muxingAudio
    case completed
    case failed(String)

    var label: String {
      switch self {
      case .idle: "待機中"
      case .preparing: "動画を準備中"
      case .restoring: "復元中"
      case .muxingAudio: "音声を結合中"
      case .completed: "完了"
      case .failed: "失敗"
      }
    }
  }

  @Published private(set) var inputURL: URL?
  @Published private(set) var inputDuration: Double?
  @Published private(set) var resolvedMediaSource: IPadResolvedMediaSource?
  @Published private(set) var isResolvingURL = false
  @Published private(set) var urlInputStatus: String?
  @Published private(set) var urlInputRequiresInteraction = false
  @Published private(set) var urlInteractionURL: URL?
  @Published private(set) var outputURL: URL?
  @Published private(set) var state: RunState = .idle
  @Published private(set) var progress = 0.0
  @Published private(set) var metrics: MiohClusterJobMetrics?
  @Published private(set) var livePreviewImage: CGImage?
  @Published private(set) var livePreviewPosition: Double?
  @Published private(set) var logs: [IPadStandaloneLogEntry] = []
  @Published private(set) var hasSavedSettings = false

  private var inputFPSNumerator: Int?
  private var inputFPSDenominator: Int?

  @Published var restorationModelIdentifier =
    "basicvsrpp-v1.2-coreai-variable"
  @Published var detectorModelIdentifier = "v4-fast-coreml"
  @Published var clipLength = 18
  @Published var temporalOverlap = 2
  @Published var parallelRestorationLanes = 3
  @Published var crossfade = true
  @Published var detectFaceMosaics = false
  @Published var detectionEmptyLookahead = 8
  @Published var blendFeather = Float(0.08)
  @Published var videoCodec = "hevc"
  @Published var bitrateMultiplier = 1.0
  @Published var mp4FastStart = true
  @Published var useFPS = false
  @Published var targetFPSNumerator = 30
  @Published var targetFPSDenominator = 1
  @Published var livePreviewEnabled = true
  @Published var previewBufferLimit = 8.0
  @Published var keepScreenAwake = true
  @Published var hlsQualityPreference = IPadHLSQualityPreference.automatic
  @Published var hlsStreamingMode = IPadHLSStreamingMode.fast
  @Published var detectionMaskReuseSkipFrames = 1
  @Published var limitHighFrameRateBeforeRestoration = true
  @Published var realtimeFrameRateMode = IPadRealtimeFrameRateMode.automatic

  private struct StoredSettings: Codable {
    var restorationModelIdentifier: String
    var detectorModelIdentifier: String
    var clipLength: Int
    var temporalOverlap: Int
    var parallelRestorationLanes: Int?
    var crossfade: Bool
    var detectFaceMosaics: Bool
    var detectionEmptyLookahead: Int
    var blendFeather: Float
    var videoCodec: String
    var bitrateMultiplier: Double
    var mp4FastStart: Bool
    var useFPS: Bool?
    var targetFPSNumerator: Int?
    var targetFPSDenominator: Int?
    var livePreviewEnabled: Bool
    var previewBufferLimit: Double?
    var keepScreenAwake: Bool
    var hlsQualityPreference: String?
    var hlsStreamingMode: String?
    // Kept only to migrate the one-release Boolean setting.
    var reuseDetectionMaskEveryOtherFrame: Bool?
    var detectionMaskReuseSkipFrames: Int?
    var limitHighFrameRateBeforeRestoration: Bool?
    var realtimeFrameRateMode: String?

    static let standard = StoredSettings(
      restorationModelIdentifier: "basicvsrpp-v1.2-coreai-variable",
      detectorModelIdentifier: "v4-fast-coreml",
      clipLength: 18,
      temporalOverlap: 2,
      parallelRestorationLanes: 3,
      crossfade: true,
      detectFaceMosaics: false,
      detectionEmptyLookahead: 8,
      blendFeather: 0.08,
      videoCodec: "hevc",
      bitrateMultiplier: 1,
      mp4FastStart: true,
      useFPS: false,
      targetFPSNumerator: 30,
      targetFPSDenominator: 1,
      livePreviewEnabled: true,
      previewBufferLimit: 8,
      keepScreenAwake: true,
      hlsQualityPreference: IPadHLSQualityPreference.automatic.rawValue,
      hlsStreamingMode: IPadHLSStreamingMode.fast.rawValue,
      reuseDetectionMaskEveryOtherFrame: nil,
      detectionMaskReuseSkipFrames: 1,
      limitHighFrameRateBeforeRestoration: true,
      realtimeFrameRateMode: IPadRealtimeFrameRateMode.automatic.rawValue
    )
  }

  private enum Keys {
    static let savedSettings = "mioh.ipad.standalone.defaults.v1"
  }

  private let defaults: UserDefaults
  private var runTask: Task<Void, Never>?
  private var previewGeneration = UUID()
  private var previousIdleTimerDisabled: Bool?
  private var downloadedRemoteInputURL: URL?
  private var sftpStreamingInput: IPadSFTPStreamingInput?
  private var pendingSFTPStreamingInput: IPadSFTPStreamingInput?
  private var inputByteCountOverride: Int64?
  private var inputDisplayNameOverride: String?
  private var inputRangeValidatorOverride: String?
  private var activeURLResolutionID: UUID?
  private var resolvedSelectionOwnerID: UUID?
  private var inputSelectionGeneration: UInt64 = 0

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    if let data = defaults.data(forKey: Keys.savedSettings),
      let saved = try? JSONDecoder().decode(StoredSettings.self, from: data)
    {
      apply(saved)
      hasSavedSettings = true
      logs.append(
        IPadStandaloneLogEntry(message: "保存済みのiPad設定を適用しました。")
      )
    } else {
      logs.append(
        IPadStandaloneLogEntry(message: "iPad単体復元を準備しました。")
      )
    }
    normalizeClipSettings()
  }

  var isRunning: Bool {
    switch state {
    case .preparing, .restoring, .muxingAudio: true
    case .idle, .completed, .failed: false
    }
  }

  var maximumClipLength: Int {
    switch restorationModelIdentifier {
    case "basicvsrpp-v1.2-coreai": 18
    case "basicvsrpp-v1.2-coreai-t36": 36
    default: 90
    }
  }

  var outputLocationLabel: String {
    "このiPad内 ＞ mioh Remote"
  }

  static let frameRateOptions: [IPadFrameRateOption] = [
    .init(numerator: 24_000, denominator: 1_001),
    .init(numerator: 24, denominator: 1),
    .init(numerator: 25, denominator: 1),
    .init(numerator: 30_000, denominator: 1_001),
    .init(numerator: 30, denominator: 1),
    .init(numerator: 48, denominator: 1),
    .init(numerator: 50, denominator: 1),
    .init(numerator: 60_000, denominator: 1_001),
    .init(numerator: 60, denominator: 1),
    .init(numerator: 100, denominator: 1),
    .init(numerator: 120_000, denominator: 1_001),
    .init(numerator: 120, denominator: 1),
  ]

  var selectedFrameRate: String {
    get { "\(targetFPSNumerator)/\(targetFPSDenominator)" }
    set {
      guard let option = Self.frameRateOptions.first(where: { $0.id == newValue })
      else { return }
      targetFPSNumerator = option.numerator
      targetFPSDenominator = option.denominator
    }
  }

  var targetFPSLabel: String {
    IPadFrameRateOption(
      numerator: targetFPSNumerator,
      denominator: targetFPSDenominator
    ).label
  }

  var isHLSInput: Bool {
    resolvedMediaSource?.kind == .hls
  }

  var isLiveHLSInput: Bool {
    resolvedMediaSource?.hlsPlaylist?.isLive == true
  }

  var inputDisplayName: String? {
    guard let inputURL else { return nil }
    if let inputDisplayNameOverride { return inputDisplayNameOverride }
    if let source = resolvedMediaSource {
      let host = source.submittedURL.host ?? source.mediaURL.host
      return host.map { "URL: \($0)" } ?? "URLストリーム"
    }
    return inputURL.lastPathComponent
  }

  var canRunFullRestoration: Bool {
    inputURL != nil && !isHLSInput && !isSFTPStreamingInput
  }

  var isSFTPStreamingInput: Bool { sftpStreamingInput != nil }

  func configure(with prepared: MiohIPadPreparedWorker) {
    detectorModelIdentifier = Self.preferredDetectorIdentifier(
      detectorModelIdentifier
    )
    if !prepared.restorationModelIdentifiers.contains(
      restorationModelIdentifier
    ) {
      restorationModelIdentifier =
        prepared.restorationModelIdentifiers.first ?? ""
    }
    if !prepared.detectorModelIdentifiers.contains(detectorModelIdentifier) {
      detectorModelIdentifier = prepared.detectorModelIdentifiers.first ?? ""
    }
    parallelRestorationLanes = min(
      parallelRestorationLanes,
      prepared.maximumParallelRestorationLanes
    )
    normalizeClipSettings()
  }

  func selectInput(
    _ url: URL,
    resolutionOperationID: UUID? = nil
  ) async {
    guard !isRunning else { return }
    if let resolutionOperationID,
      activeURLResolutionID != resolutionOperationID
    {
      return
    }
    inputSelectionGeneration &+= 1
    let selectionGeneration = inputSelectionGeneration
    urlInputRequiresInteraction = false
    urlInteractionURL = nil
    resolvedSelectionOwnerID = nil
    if pendingSFTPStreamingInput?.localURL != url {
      discardPendingSFTPStreamingInput()
    }
    discardDownloadedRemoteInput()
    inputURL = url
    inputDuration = nil
    inputFPSNumerator = nil
    inputFPSDenominator = nil
    resolvedMediaSource = nil
    urlInputStatus = nil
    outputURL = nil
    metrics = nil
    progress = 0
    resetLivePreview()
    state = .preparing
    appendLog("入力動画を確認中: \(url.lastPathComponent)")
    let access = url.startAccessingSecurityScopedResource()
    defer { if access { url.stopAccessingSecurityScopedResource() } }
    do {
      try checkURLResolution(resolutionOperationID)
      let asset = AVURLAsset(url: url)
      let duration = try await asset.load(.duration)
      guard inputSelectionGeneration == selectionGeneration else { return }
      try checkURLResolution(resolutionOperationID)
      guard duration.isNumeric, duration.seconds.isFinite,
        duration.seconds > 0, duration.seconds <= 9_000_000_000
      else {
        throw MiohIPadWorkerEngineError.unsupportedMedia("動画時間が不正です")
      }
      let videoTracks = try await asset.loadTracks(withMediaType: .video)
      guard inputSelectionGeneration == selectionGeneration else { return }
      guard let videoTrack = videoTracks.first else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "動画トラックがありません"
        )
      }
      let frameRate = await MiohIPadSourceFrameRate.resolve(track: videoTrack)
      try checkURLResolution(resolutionOperationID)
      inputDuration = duration.seconds
      inputFPSNumerator = frameRate.numerator
      inputFPSDenominator = frameRate.denominator
      state = .idle
      appendLog(
        String(format: "入力を選択しました（%.1f秒）。", duration.seconds)
      )
    } catch {
      guard inputSelectionGeneration == selectionGeneration else { return }
      if let resolutionOperationID,
        activeURLResolutionID != resolutionOperationID
      {
        if inputURL == url { inputURL = nil }
        inputDuration = nil
        inputFPSNumerator = nil
        inputFPSDenominator = nil
        return
      }
      if inputURL == url { inputURL = nil }
      inputDuration = nil
      inputFPSNumerator = nil
      inputFPSDenominator = nil
      state = .failed(error.localizedDescription)
      appendLog(error.localizedDescription, level: .error)
    }
  }

  func selectManagedDownloadedInput(_ url: URL) async {
    guard !isRunning else {
      try? FileManager.default.removeItem(at: url)
      return
    }
    await selectInput(url)
    guard inputURL == url else {
      try? FileManager.default.removeItem(at: url)
      return
    }
    downloadedRemoteInputURL = url
    urlInputStatus = "SFTPから動画を取得しました。復元を開始できます。"
    appendLog("SFTP入力の準備が完了しました。", level: .success)
  }

  /// Selects a user-owned SFTP download from Documents. Unlike progressive
  /// URL staging files, this file remains in Files when the input is cleared.
  func selectPersistentDownloadedInput(_ url: URL) async {
    guard !isRunning else { return }
    await selectInput(url)
    guard inputURL == url else { return }
    urlInputStatus =
      "SFTPからFilesへ保存しました。入力を解除してもファイルは残ります。"
    appendLog(
      "SFTP動画をFilesへ保存し、復元入力として選択しました。",
      level: .success
    )
  }

  @discardableResult
  func selectSFTPStreamingInput(_ input: IPadSFTPStreamingInput) async -> Bool {
    guard !isRunning else {
      input.stop()
      return false
    }
    discardPendingSFTPStreamingInput()
    pendingSFTPStreamingInput = input
    await selectInput(input.localURL)
    guard pendingSFTPStreamingInput === input, inputURL == input.localURL else {
      if pendingSFTPStreamingInput === input {
        pendingSFTPStreamingInput = nil
      }
      input.stop()
      return false
    }
    pendingSFTPStreamingInput = nil
    sftpStreamingInput = input
    inputByteCountOverride = input.byteCount
    inputDisplayNameOverride = "SFTP: \(input.displayName)"
    inputRangeValidatorOverride = input.rangeValidator
    urlInputStatus = "SFTPから必要な範囲だけ取得し、復元しながら再生します。"
    appendLog("SFTPストリーミング入力の準備が完了しました。", level: .success)
    return true
  }

  @discardableResult
  func selectURLInput(
    _ rawValue: String,
    selectionOwnerID: UUID? = nil
  ) async -> Bool {
    guard !isRunning, !isResolvingURL else { return false }
    let operationID = UUID()
    activeURLResolutionID = operationID
    isResolvingURL = true
    urlInputRequiresInteraction = false
    urlInteractionURL = nil
    urlInputStatus = "URLを解析中…"
    state = .preparing
    appendLog("入力URLを解析しています。")
    defer {
      if activeURLResolutionID == operationID {
        activeURLResolutionID = nil
        isResolvingURL = false
      }
    }

    do {
      let source = try await resolveURLInput(
        rawValue,
        operationID: operationID
      )
      try checkURLResolution(operationID)
      return try await acceptResolvedURLSource(
        source,
        operationID: operationID,
        selectionOwnerID: selectionOwnerID
      )
    } catch is CancellationError {
      guard activeURLResolutionID == operationID else { return false }
      state = .idle
      urlInputStatus = "URL解析を中止しました。"
      appendLog("URL解析を中止しました。", level: .warning)
      return false
    } catch {
      guard activeURLResolutionID == operationID else { return false }
      let requiresInteraction = Self.isAnyInteractionRequired(error)
      urlInputRequiresInteraction = requiresInteraction
      if requiresInteraction {
        urlInteractionURL = Self.interactionURL(from: error)
      }
      state = requiresInteraction ? .idle : .failed(error.localizedDescription)
      urlInputStatus = error.localizedDescription
      appendLog(
        error.localizedDescription,
        level: requiresInteraction ? .warning : .error
      )
      return false
    }
  }

  /// Accepts only candidates captured by the visible browser. When a browser
  /// handoff loader is supplied, playlist resolution stays inside the same
  /// WebKit session so Cloudflare-protected HLS can retain its browser state.
  @discardableResult
  func selectBrowserCandidates(_ candidates: [IPadWebMediaCandidate]) async
    -> Bool
  {
    await selectBrowserCandidates(candidates, selectionOwnerID: nil)
  }

  @discardableResult
  func selectBrowserCandidates(
    _ candidates: [IPadWebMediaCandidate],
    selectionOwnerID: UUID?,
    hlsResourceLoader: (any IPadHLSResourceLoading)? = nil
  ) async -> Bool {
    guard !isRunning, !isResolvingURL, !candidates.isEmpty else { return false }
    let operationID = UUID()
    activeURLResolutionID = operationID
    isResolvingURL = true
    urlInputRequiresInteraction = false
    urlInteractionURL = nil
    urlInputStatus = "ブラウザで検出した配信を解析中…"
    state = .preparing
    appendLog("ブラウザで検出した配信候補を確認しています。")
    if hlsResourceLoader != nil {
      appendLog("Safari互換: HLS解析もWebKitのブラウザ通信を使用します。")
    }
    defer {
      if activeURLResolutionID == operationID {
        activeURLResolutionID = nil
        isResolvingURL = false
      }
    }

    do {
      var lastError: Error = IPadWebMediaDiscoveryError.noCandidates
      var firstUsefulError: Error?
      var pendingInteractionError: Error?
      var resolvedChoices: [IPadResolvedMediaSource] = []
      var resolvedEvidence: [IPadBrowserMediaEvidence?] = []
      var resolvedIndexes: [String: Int] = [:]

      func acceptCollectedChoices() async throws -> Bool {
        while let selectedIndex = IPadBrowserMediaSourceSelector.preferredIndex(
          in: resolvedChoices,
          evidence: resolvedEvidence
        ) {
          let selectedSource = resolvedChoices.remove(at: selectedIndex)
          resolvedEvidence.remove(at: selectedIndex)
          if resolvedChoices.count > 0 {
            appendLog(
              "再生可能な配信候補を比較し、本編向けの候補を選択しました。"
            )
          }
          do {
            if try await acceptResolvedURLSource(
              selectedSource,
              operationID: operationID,
              selectionOwnerID: selectionOwnerID,
              hlsResourceLoader: hlsResourceLoader
            ) {
              return true
            }
            if case .failed(let message) = state {
              firstUsefulError =
                firstUsefulError
                ?? MiohIPadWorkerEngineError.unsupportedMedia(message)
            }
          } catch is CancellationError {
            throw CancellationError()
          } catch {
            firstUsefulError = firstUsefulError ?? error
          }
        }
        return false
      }

      func discoveryRank(_ role: IPadWebMediaCandidate.DiscoveryRole) -> Int {
        switch role {
        case .activePlayback: 0
        case .verifiedMediaResponse: 1
        case .unverifiedMediaResponse: 2
        case .directHint: 3
        case .pageLead: 4
        }
      }
      let primaryCandidates = candidates.filter {
        $0.selectionState != .supersededCurrentSource
      }
      let orderedPrimaryCandidates = primaryCandidates.enumerated().sorted {
        lhs, rhs in
        let leftRank = discoveryRank(lhs.element.discoveryRole)
        let rightRank = discoveryRank(rhs.element.discoveryRole)
        return leftRank == rightRank ? lhs.offset < rhs.offset : leftRank < rightRank
      }.map(\.element)
      let supersededFallbackCandidates = candidates.filter {
        $0.selectionState == .supersededCurrentSource
      }
      var remainingAttempts = IPadBrowserMediaSourceSelector.maximumPlayableChoices
      for pool in [orderedPrimaryCandidates, supersededFallbackCandidates] {
        for candidate in pool {
          try checkURLResolution(operationID)
          guard remainingAttempts > 0 else { break }
          remainingAttempts -= 1
          do {
            let source = try await Self.resolveBrowserCandidate(
              candidate,
              allowsAES128HLS: hlsStreamingMode.allowsAES128HLS,
              hlsResourceLoader: hlsResourceLoader
            )
            try checkURLResolution(operationID)
            if let playlist = source.hlsPlaylist {
              let host =
                source.submittedURL.host ?? source.mediaURL.host
                ?? candidate.url.host ?? "不明なホスト"
              appendLog(
                "HLS候補を確認: \(host) / \(Self.mediaDurationLabel(playlist.duration))"
              )
            }
            if IPadBrowserMediaSourceSelector.isHighConfidenceAdvertisementSource(
              source
            ) {
              appendLog(
                "右下の広告プレイヤーが発行したHLS候補を除外しました。"
              )
              continue
            }
            if source.kind == .hls,
              source.hlsPlaylist.map({
                $0.segments.isEmpty || $0.duration <= 0
              }) != false
            {
              lastError = MiohIPadWorkerEngineError.unsupportedMedia(
                "再生可能なHLSメディア区間がありません"
              )
              continue
            }
            if IPadBrowserMediaSourceSelector.shouldAcceptImmediately(
              source,
              evidence: candidate.mediaEvidence
            ) {
              appendLog(
                "長時間のHLS本編候補を確認したため、期限切れ前に選択しました。"
              )
              if try await acceptResolvedURLSource(
                source,
                operationID: operationID,
                selectionOwnerID: selectionOwnerID,
                hlsResourceLoader: hlsResourceLoader
              ) {
                return true
              }
              if case .failed(let message) = state {
                firstUsefulError =
                  firstUsefulError
                  ?? MiohIPadWorkerEngineError.unsupportedMedia(message)
              }
              continue
            }
            let key = IPadBrowserMediaSourceSelector.deduplicationKey(for: source)
            if let existingIndex = resolvedIndexes[key] {
              if resolvedEvidence[existingIndex] == nil,
                candidate.mediaEvidence != nil
              {
                resolvedChoices[existingIndex] = source
                resolvedEvidence[existingIndex] = candidate.mediaEvidence
              }
            } else {
              resolvedIndexes[key] = resolvedChoices.count
              resolvedChoices.append(source)
              resolvedEvidence.append(candidate.mediaEvidence)
            }
            // Resolve the complete bounded browser set. The main programme is
            // often requested after several pre-roll and tracking playlists.
          } catch is CancellationError {
            throw CancellationError()
          } catch IPadMediaURLResolverError.unsafeInitialURL {
            // Initial safety rejection performs no request. Refund the network
            // attempt so a later main-programme candidate is still examined.
            remainingAttempts += 1
            continue
          } catch IPadMediaURLResolverError.interactionRequired {
            // A Cloudflare subresource/redirect URL is not a standalone page:
            // opening it as the main document loses the player frame and its
            // verification context. Return to the visible page that emitted
            // the candidate instead.
            let targetURL =
              candidate.interactionPageURL
              ?? candidate.requestContext.referer
              ?? candidate.url
            urlInteractionURL = urlInteractionURL ?? targetURL
            let interactionError = IPadMediaURLResolverError.interactionRequired(
              targetURL
            )
            pendingInteractionError = pendingInteractionError ?? interactionError
            lastError = interactionError
          } catch {
            // A visible page may advertise unusable, local or malformed URLs
            // alongside its real stream. Those failures belong to that one
            // candidate and must not become the result for the whole page.
            if !Self.isSkippableBrowserCandidateError(error) {
              firstUsefulError = firstUsefulError ?? error
            }
          }
        }
        if try await acceptCollectedChoices() { return true }
        // Every collected source has now been attempted. Clear the dedupe map
        // before optionally resolving superseded sources as a fallback.
        resolvedIndexes.removeAll(keepingCapacity: true)
        if remainingAttempts == 0 { break }
      }
      throw pendingInteractionError
        ?? firstUsefulError
        ?? lastError
    } catch is CancellationError {
      guard activeURLResolutionID == operationID else { return false }
      state = .idle
      urlInputStatus = "ブラウザ解析を中止しました。"
      appendLog("ブラウザ解析を中止しました。", level: .warning)
      return false
    } catch {
      guard activeURLResolutionID == operationID else { return false }
      let requiresInteraction = Self.isAnyInteractionRequired(error)
      urlInputRequiresInteraction = requiresInteraction
      // Browser candidates are a live observation, not a final input. A
      // rejected poster, pre-roll, or stale playlist must leave the workspace
      // ready to accept the next request emitted by the still-visible player.
      state = .idle
      urlInputStatus = error.localizedDescription
      appendLog(
        error.localizedDescription,
        level: .warning
      )
      return false
    }
  }

  /// Resolver startup performs DNS safety checks. Keeping the browser-candidate
  /// path nonisolated prevents those synchronous system lookups from blocking
  /// SwiftUI while the bounded candidates are examined.
  private nonisolated static func resolveBrowserCandidate(
    _ candidate: IPadWebMediaCandidate,
    allowsAES128HLS: Bool,
    hlsResourceLoader: (any IPadHLSResourceLoading)? = nil
  ) async throws -> IPadResolvedMediaSource {
    let policy = browserResolutionPolicy(for: candidate)
    return try await IPadMediaURLResolver(
      resourceLoader: hlsResourceLoader,
      allowsAES128HLS: allowsAES128HLS
    ).resolve(
      candidate.url.absoluteString,
      policy: policy,
      context: candidate.requestContext
    )
  }

  /// Packet-tunnel VPNs may synthesize 198.18/15 DNS answers. Permit that
  /// compatibility only for a source WebKit has actually decoded and is
  /// visibly playing. DOM attributes, iframe URLs, stale sources and hidden
  /// media remain on the strict public-network policy.
  private nonisolated static func browserResolutionPolicy(
    for candidate: IPadWebMediaCandidate
  ) -> IPadMediaURLResolutionPolicy {
    guard candidate.selectionState == .activeCurrentSource,
      let evidence = candidate.mediaEvidence,
      evidence.isPlaying,
      evidence.isVisible,
      evidence.visibilityAttested,
      evidence.renderedArea >= 4_096,
      let approvedOrigin = browserOriginURL(for: candidate.url)
    else { return .publicDiscovered }
    return .visibleBrowserDiscovered(approvedOrigin)
  }

  private nonisolated static func mediaDurationLabel(
    _ duration: TimeInterval
  ) -> String {
    guard duration.isFinite, duration >= 0 else { return "長さ不明" }
    let totalSeconds = Int(duration.rounded())
    return String(
      format: "%d:%02d:%02d",
      totalSeconds / 3_600,
      (totalSeconds % 3_600) / 60,
      totalSeconds % 60
    )
  }

  private nonisolated static func browserOriginURL(for url: URL) -> URL? {
    guard
      var components = URLComponents(url: url, resolvingAgainstBaseURL: true),
      components.scheme?.lowercased() == "https",
      components.user == nil, components.password == nil,
      components.host?.isEmpty == false
    else { return nil }
    components.scheme = "https"
    components.path = ""
    components.query = nil
    components.fragment = nil
    return components.url
  }

  private func acceptResolvedURLSource(
    _ resolvedSource: IPadResolvedMediaSource,
    operationID: UUID,
    selectionOwnerID: UUID?,
    hlsResourceLoader: (any IPadHLSResourceLoading)? = nil
  ) async throws -> Bool {
    try checkURLResolution(operationID)
    var source = resolvedSource
    if source.kind == .hls,
      let targetHeight = hlsQualityPreference.targetHeight
    {
      let resolver = IPadMediaURLResolver(
        resourceLoader: hlsResourceLoader,
        allowsAES128HLS: hlsStreamingMode.allowsAES128HLS
      )
      while let selectedHeight = source.hlsPlaylist?.masterMetadata?.height,
        selectedHeight > targetHeight,
        let lower = try await resolver.resolveNextHLSVariant(for: source)
      {
        try checkURLResolution(operationID)
        source = lower
      }
    }
    switch source.kind {
    case .hls:
      guard let playlist = source.hlsPlaylist,
        !playlist.segments.isEmpty, playlist.duration > 0
      else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "再生可能なHLSメディア区間がありません"
        )
      }
      discardDownloadedRemoteInput()
      inputURL = source.playbackURL
      inputDuration = playlist.duration
      if let frameRate = playlist.masterMetadata?.frameRate {
        let rational = MiohIPadSourceFrameRate.rational(frameRate)
        inputFPSNumerator = rational.numerator
        inputFPSDenominator = rational.denominator
      } else {
        inputFPSNumerator = nil
        inputFPSDenominator = nil
      }
      resolvedMediaSource = source
      resolvedSelectionOwnerID = selectionOwnerID
      outputURL = nil
      metrics = nil
      progress = 0
      resetLivePreview()
      state = .idle
      urlInputStatus =
        playlist.isLive
        ? "ライブHLSを検出しました。現在の配信区間から復元します。"
        : "HLSを検出しました。区間を取得しながら復元します。"
      appendLog(
        playlist.isLive
          ? "ライブHLSを検出しました: \(source.submittedURL.host ?? source.mediaURL.host ?? "不明なホスト")"
          : "HLSストリームを検出しました: \(source.submittedURL.host ?? source.mediaURL.host ?? "不明なホスト") / \(Self.mediaDurationLabel(playlist.duration))",
        level: .success
      )
      return true
    case .progressive:
      let localURL = try await Self.downloadProgressiveMedia(source)
      do {
        try checkURLResolution(operationID)
      } catch {
        try? FileManager.default.removeItem(at: localURL)
        throw error
      }
      state = .idle
      await selectInput(
        localURL,
        resolutionOperationID: operationID
      )
      do {
        try checkURLResolution(operationID)
      } catch {
        if inputURL == localURL { inputURL = nil }
        try? FileManager.default.removeItem(at: localURL)
        throw error
      }
      if case .failed = state {
        try? FileManager.default.removeItem(at: localURL)
        return false
      }
      downloadedRemoteInputURL = localURL
      resolvedMediaSource = source
      resolvedSelectionOwnerID = selectionOwnerID
      urlInputStatus = "動画URLを取得しました。復元再生を開始できます。"
      appendLog("動画URLの取得が完了しました。", level: .success)
      return true
    }
  }

  private func resolveURLInput(
    _ rawValue: String,
    operationID: UUID
  ) async throws
    -> IPadResolvedMediaSource
  {
    try checkURLResolution(operationID)
    let resolver = IPadMediaURLResolver(
      allowsAES128HLS: hlsStreamingMode.allowsAES128HLS
    )
    do {
      return try await resolver.resolve(rawValue)
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      if Self.isInteractionRequired(error) { throw error }
      guard Self.shouldOpenVisibleBrowser(after: error) else { throw error }
    }

    try checkURLResolution(operationID)
    urlInputStatus = "動的ページはブラウザタブで確認してください。"
    appendLog(
      "静的解析では配信URLを取得できないため、表示中のブラウザタブへ切り替えます。",
      level: .warning
    )
    throw IPadWebMediaDiscoveryError.interactionRequired
  }

  private static func isInteractionRequired(_ error: Error) -> Bool {
    guard let resolverError = error as? IPadMediaURLResolverError else {
      return false
    }
    if case .interactionRequired = resolverError { return true }
    return false
  }

  private static func isSkippableBrowserCandidateError(_ error: Error) -> Bool {
    guard let resolverError = error as? IPadMediaURLResolverError else {
      return false
    }
    switch resolverError {
    case .invalidURL, .unsafeInitialURL, .unsafeURL, .insecureRedirect,
      .unsupportedContent:
      return true
    case .requestFailed, .invalidHTTPStatus, .responseTooLarge,
      .tooManyRedirects, .invalidPlaylist, .encryptedPlaylist,
      .invalidByteRange, .resolutionLimitExceeded, .interactionRequired:
      return false
    }
  }

  private static func interactionURL(from error: Error) -> URL? {
    guard let resolverError = error as? IPadMediaURLResolverError,
      case .interactionRequired(let url) = resolverError
    else { return nil }
    return url
  }

  private static func isAnyInteractionRequired(_ error: Error) -> Bool {
    if isInteractionRequired(error) { return true }
    guard let discoveryError = error as? IPadWebMediaDiscoveryError else {
      return false
    }
    if case .interactionRequired = discoveryError { return true }
    return false
  }

  private static func shouldOpenVisibleBrowser(after error: Error) -> Bool {
    guard let resolverError = error as? IPadMediaURLResolverError else {
      return true
    }
    switch resolverError {
    case .invalidURL, .unsafeInitialURL, .unsafeURL, .insecureRedirect,
      .encryptedPlaylist:
      return false
    case .requestFailed, .invalidHTTPStatus, .responseTooLarge,
      .tooManyRedirects, .unsupportedContent, .invalidPlaylist,
      .invalidByteRange, .resolutionLimitExceeded, .interactionRequired:
      return true
    }
  }

  func cancelURLResolution() {
    guard isResolvingURL || activeURLResolutionID != nil else { return }
    activeURLResolutionID = nil
    isResolvingURL = false
    urlInputRequiresInteraction = false
    urlInteractionURL = nil
    if state == .preparing { state = .idle }
    urlInputStatus = "URL解析を中止しました。"
    appendLog("URL解析を中止しました。", level: .warning)
  }

  private func checkURLResolution(_ operationID: UUID?) throws {
    try Task.checkCancellation()
    if let operationID, activeURLResolutionID != operationID {
      throw CancellationError()
    }
  }

  func clearResolvedURLInput(ownedBy selectionOwnerID: UUID) {
    guard !isRunning, resolvedSelectionOwnerID == selectionOwnerID else {
      return
    }
    clearInput()
  }

  func clearInput() {
    guard !isRunning else { return }
    inputSelectionGeneration &+= 1
    urlInputRequiresInteraction = false
    urlInteractionURL = nil
    resolvedSelectionOwnerID = nil
    discardPendingSFTPStreamingInput()
    discardDownloadedRemoteInput()
    inputURL = nil
    inputDuration = nil
    inputFPSNumerator = nil
    inputFPSDenominator = nil
    resolvedMediaSource = nil
    urlInputStatus = nil
    outputURL = nil
    metrics = nil
    progress = 0
    resetLivePreview()
    state = .idle
    appendLog("入力を解除しました。")
  }

  func clearSFTPStreamingInput() {
    if let pendingURL = pendingSFTPStreamingInput?.localURL {
      inputSelectionGeneration &+= 1
      discardPendingSFTPStreamingInput()
      if inputURL == pendingURL {
        inputURL = nil
        inputDuration = nil
        inputFPSNumerator = nil
        inputFPSDenominator = nil
        resolvedMediaSource = nil
        urlInputStatus = nil
        if case .preparing = state { state = .idle }
        resetLivePreview()
      }
    }
    guard isSFTPStreamingInput else { return }
    clearInput()
  }

  func start(prepared: MiohIPadPreparedWorker) {
    guard !isRunning, canRunFullRestoration, let inputURL else { return }
    configure(with: prepared)
    outputURL = nil
    metrics = nil
    progress = 0
    resetLivePreview()
    let generation = previewGeneration
    appendLog(
      "復元を開始します: \(restorationModelIdentifier) / \(detectorModelIdentifier)"
    )
    beginPreventingSleepIfNeeded()
    runTask = Task { [weak self] in
      guard let self else { return }
      do {
        try await self.run(
          inputURL: inputURL,
          prepared: prepared,
          generation: generation
        )
      } catch is CancellationError {
        self.state = .idle
        self.progress = 0
        self.resetLivePreview()
        self.appendLog("復元を中止しました。", level: .warning)
      } catch {
        self.state = .failed(error.localizedDescription)
        self.appendLog(error.localizedDescription, level: .error)
      }
      self.endPreventingSleepIfNeeded()
      self.runTask = nil
    }
  }

  func cancel() {
    guard isRunning else { return }
    appendLog("中止を要求しました。", level: .warning)
    resetLivePreview()
    runTask?.cancel()
  }

  func clearOutput() {
    guard !isRunning else { return }
    outputURL = nil
    metrics = nil
    progress = 0
    resetLivePreview()
    state = .idle
    appendLog("結果表示を閉じました。ファイルはFilesに残っています。")
  }

  func normalizeClipSettings() {
    clipLength = min(max(1, clipLength), maximumClipLength)
    temporalOverlap = min(max(0, temporalOverlap), max(0, clipLength - 1))
    parallelRestorationLanes = min(max(1, parallelRestorationLanes), 3)
    previewBufferLimit = min(max(1, previewBufferLimit), 60)
  }

  func realtimePreviewConfiguration(
    prepared: MiohIPadPreparedWorker,
    hlsResourceLoader: (any IPadHLSResourceLoading)? = nil
  ) throws -> IPadRealtimePreviewConfiguration {
    let maximumDurationSeconds = 9_000_000_000.0
    guard let inputURL, let duration = inputDuration,
      duration.isFinite, duration > 0,
      duration <= maximumDurationSeconds
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "再生する入力動画を選択してください"
      )
    }
    guard
      let restorationDigest =
        prepared.restorationAssetSHA256ByIdentifier[restorationModelIdentifier],
      let detectorDigest =
        prepared.detectorAssetSHA256ByIdentifier[detectorModelIdentifier]
    else {
      throw MiohIPadWorkerEngineError.unsupportedJob(
        "選択したモデルが検証済み一覧にありません"
      )
    }
    let inputExtension =
      isHLSInput
      ? "mp4" : inputURL.pathExtension.lowercased()
    guard isHLSInput || ["mp4", "mov", "m4v"].contains(inputExtension) else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "MP4/MOV/M4Vを選択してください"
      )
    }
    let realtimeClipLength = min(
      maximumClipLength,
      IPadRealtimePreviewConfiguration.realtimeTemporalFrames
    )
    let realtimeTemporalOverlap = min(
      max(
        IPadRealtimePreviewConfiguration.minimumTemporalOverlap,
        temporalOverlap
      ),
      max(0, realtimeClipLength - 1)
    )
    let maximumFrameRate =
      limitHighFrameRateBeforeRestoration && realtimeFrameRateMode != .fps24
      ? 30 : nil
    let limitedRate: (numerator: Int, denominator: Int)?
    if let maximumFrameRate, let inputFPSNumerator, let inputFPSDenominator {
      limitedRate = MiohIPadSourceFrameRate.maximumRate(
        wholeFPS: maximumFrameRate,
        sourceNumerator: inputFPSNumerator,
        sourceDenominator: inputFPSDenominator
      )
    } else {
      limitedRate = nil
    }
    let targetRate: (numerator: Int, denominator: Int)? =
      realtimeFrameRateMode == .fps24
      ? (numerator: 24, denominator: 1)
      : limitedRate
    let options = MiohClusterRestorationOptions(
      restorationModelIdentifier: restorationModelIdentifier,
      restorationAssetSHA256: restorationDigest,
      detectorModelIdentifier: detectorModelIdentifier,
      detectorAssetSHA256: detectorDigest,
      restorationClipLength: realtimeClipLength,
      temporalOverlap: realtimeTemporalOverlap,
      crossfade: crossfade,
      detectionEmptyLookahead: detectionEmptyLookahead,
      detectFaceMosaics: detectFaceMosaics,
      detectionMaskReuseSkipFrames: detectionMaskReuseSkipFrames,
      blendFeather: blendFeather,
      sharpenStrength: 0,
      detailBoost: 0,
      textureMix: 0,
      smoothStrength: 0,
      effectUpscale: 1,
      videoCodec: videoCodec,
      bitrateMultiplier: bitrateMultiplier,
      mp4FastStart: mp4FastStart,
      targetFPSNumerator: targetRate?.numerator,
      targetFPSDenominator: targetRate?.denominator
    )
    return IPadRealtimePreviewConfiguration(
      inputURL: inputURL,
      resolvedMediaSource: resolvedMediaSource,
      hlsResourceLoader: hlsResourceLoader,
      hlsStreamingMode: hlsStreamingMode,
      hlsQualityPreference: hlsQualityPreference,
      detectionMaskReuseSkipFrames: detectionMaskReuseSkipFrames,
      realtimeFrameRateMode: realtimeFrameRateMode,
      maximumFrameRate: maximumFrameRate,
      parallelRestorationLanes: min(
        3,
        max(1, min(parallelRestorationLanes, prepared.maximumParallelRestorationLanes))
      ),
      durationNanoseconds: max(
        1,
        Int64((duration * 1_000_000_000).rounded())
      ),
      inputByteCount: inputByteCountOverride
        ?? Int64(
          (try? inputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        ),
      inputExtension: inputExtension,
      inputRangeValidator: inputRangeValidatorOverride,
      prepareInputForSeek: sftpStreamingInput.map { input in
        { @Sendable in input.prepareForSeek() }
      },
      streamingMetrics: sftpStreamingInput.map { input in
        { @Sendable in await input.metrics() }
      },
      options: options,
      bufferLimitSeconds: previewBufferLimit,
      keepScreenAwake: keepScreenAwake
    )
  }

  func saveSettingsAsDefaults() {
    guard !isRunning else { return }
    do {
      defaults.set(
        try JSONEncoder().encode(currentSettings),
        forKey: Keys.savedSettings
      )
      hasSavedSettings = true
      appendLog("現在の設定をiPadデフォルトとして保存しました。", level: .success)
    } catch {
      appendLog("設定を保存できませんでした: \(error.localizedDescription)", level: .error)
    }
  }

  func loadSavedSettings() {
    guard !isRunning,
      let data = defaults.data(forKey: Keys.savedSettings),
      let saved = try? JSONDecoder().decode(StoredSettings.self, from: data)
    else { return }
    apply(saved)
    normalizeClipSettings()
    appendLog("保存済みのiPad設定を読み込みました。")
  }

  func resetSettings() {
    guard !isRunning else { return }
    defaults.removeObject(forKey: Keys.savedSettings)
    hasSavedSettings = false
    apply(.standard)
    normalizeClipSettings()
    appendLog("iPad設定を初期値へ戻しました。", level: .warning)
  }

  func clearLogs() {
    guard !isRunning else { return }
    logs.removeAll()
    appendLog("ログを消去しました。")
  }

  private var currentSettings: StoredSettings {
    StoredSettings(
      restorationModelIdentifier: restorationModelIdentifier,
      detectorModelIdentifier: detectorModelIdentifier,
      clipLength: clipLength,
      temporalOverlap: temporalOverlap,
      parallelRestorationLanes: parallelRestorationLanes,
      crossfade: crossfade,
      detectFaceMosaics: detectFaceMosaics,
      detectionEmptyLookahead: detectionEmptyLookahead,
      blendFeather: blendFeather,
      videoCodec: videoCodec,
      bitrateMultiplier: bitrateMultiplier,
      mp4FastStart: mp4FastStart,
      useFPS: useFPS,
      targetFPSNumerator: targetFPSNumerator,
      targetFPSDenominator: targetFPSDenominator,
      livePreviewEnabled: livePreviewEnabled,
      previewBufferLimit: previewBufferLimit,
      keepScreenAwake: keepScreenAwake,
      hlsQualityPreference: hlsQualityPreference.rawValue,
      hlsStreamingMode: hlsStreamingMode.rawValue,
      reuseDetectionMaskEveryOtherFrame: nil,
      detectionMaskReuseSkipFrames: detectionMaskReuseSkipFrames,
      limitHighFrameRateBeforeRestoration:
        limitHighFrameRateBeforeRestoration,
      realtimeFrameRateMode: realtimeFrameRateMode.rawValue
    )
  }

  private func apply(_ settings: StoredSettings) {
    restorationModelIdentifier = settings.restorationModelIdentifier
    detectorModelIdentifier = Self.preferredDetectorIdentifier(
      settings.detectorModelIdentifier
    )
    clipLength = settings.clipLength
    temporalOverlap = settings.temporalOverlap
    parallelRestorationLanes = min(
      3,
      max(1, settings.parallelRestorationLanes ?? 3)
    )
    crossfade = settings.crossfade
    detectFaceMosaics = settings.detectFaceMosaics
    detectionEmptyLookahead = settings.detectionEmptyLookahead
    blendFeather = settings.blendFeather
    videoCodec = settings.videoCodec
    bitrateMultiplier = settings.bitrateMultiplier
    mp4FastStart = settings.mp4FastStart
    useFPS = settings.useFPS ?? false
    targetFPSNumerator = max(1, settings.targetFPSNumerator ?? 30)
    targetFPSDenominator = max(1, settings.targetFPSDenominator ?? 1)
    livePreviewEnabled = settings.livePreviewEnabled
    previewBufferLimit = settings.previewBufferLimit ?? 8
    keepScreenAwake = settings.keepScreenAwake
    hlsQualityPreference =
      IPadHLSQualityPreference(
        rawValue: settings.hlsQualityPreference ?? ""
      ) ?? .automatic
    hlsStreamingMode =
      IPadHLSStreamingMode(
        rawValue: settings.hlsStreamingMode ?? ""
      ) ?? .fast
    detectionMaskReuseSkipFrames = min(
      12,
      max(
        0,
        settings.detectionMaskReuseSkipFrames
          ?? ((settings.reuseDetectionMaskEveryOtherFrame ?? true) ? 1 : 0)
      )
    )
    limitHighFrameRateBeforeRestoration =
      settings.limitHighFrameRateBeforeRestoration ?? true
    realtimeFrameRateMode =
      IPadRealtimeFrameRateMode(
        rawValue: settings.realtimeFrameRateMode ?? ""
      ) ?? .automatic
  }

  private static func preferredDetectorIdentifier(_ identifier: String)
    -> String
  {
    guard identifier.hasSuffix("-coreai") else { return identifier }
    return String(identifier.dropLast("-coreai".count)) + "-coreml"
  }

  private func appendLog(
    _ message: String,
    level: IPadStandaloneLogEntry.Level = .info
  ) {
    logs.append(IPadStandaloneLogEntry(level: level, message: message))
    if logs.count > 300 { logs.removeFirst(logs.count - 300) }
  }

  private func discardDownloadedRemoteInput() {
    let streamingInput = sftpStreamingInput
    sftpStreamingInput = nil
    inputByteCountOverride = nil
    inputDisplayNameOverride = nil
    inputRangeValidatorOverride = nil
    streamingInput?.stop()
    guard let downloadedRemoteInputURL else { return }
    self.downloadedRemoteInputURL = nil
    try? FileManager.default.removeItem(at: downloadedRemoteInputURL)
  }

  private func discardPendingSFTPStreamingInput() {
    let pendingInput = pendingSFTPStreamingInput
    pendingSFTPStreamingInput = nil
    pendingInput?.stop()
  }

  private static func downloadProgressiveMedia(
    _ source: IPadResolvedMediaSource
  ) async throws -> URL {
    let hardMaximumBytes: Int64 = 20 * 1_024 * 1_024 * 1_024
    let temporaryDirectory = FileManager.default.temporaryDirectory
    let capacityValues = try temporaryDirectory.resourceValues(
      forKeys: [
        .volumeAvailableCapacityForImportantUsageKey,
        .volumeAvailableCapacityKey,
      ]
    )
    let availableBytes =
      capacityValues.volumeAvailableCapacityForImportantUsage
      ?? capacityValues.volumeAvailableCapacity.map { Int64($0) }
      ?? 0
    let reserveBytes: Int64 = max(
      1 * 1_024 * 1_024 * 1_024,
      availableBytes / 10
    )
    guard availableBytes > reserveBytes + 16 * 1_024 * 1_024 else {
      throw IPadProgressiveDownloadError.insufficientStorage
    }
    let maximumBytes = min(hardMaximumBytes, availableBytes - reserveBytes)
    let pathExtension: String
    switch source.mediaURL.pathExtension.lowercased() {
    case "mov": pathExtension = "mov"
    case "m4v": pathExtension = "m4v"
    default: pathExtension = "mp4"
    }
    let destination =
      temporaryDirectory
      .appendingPathComponent(
        "mioh-url-input-\(UUID().uuidString.lowercased()).\(pathExtension)"
      )
    let downloader = IPadBoundedProgressiveDownloader(
      destinationURL: destination,
      maximumBytes: maximumBytes,
      maximumRedirectCount: 6,
      timeout: 60,
      originalURL: source.mediaURL,
      resolutionPolicy: source.resolutionPolicy,
      requestContext: source.requestContext
    )
    var request = URLRequest(
      url: source.mediaURL,
      cachePolicy: .reloadIgnoringLocalAndRemoteCacheData,
      timeoutInterval: 60
    )
    request.httpShouldHandleCookies = false
    request.setValue("video/*", forHTTPHeaderField: "Accept")
    source.requestContext?.applying(to: &request)

    do {
      let localURL = try await downloader.start(request)
      try Task.checkCancellation()
      return localURL
    } catch {
      try? FileManager.default.removeItem(at: destination)
      throw error
    }
  }

  private func resetLivePreview() {
    previewGeneration = UUID()
    livePreviewImage = nil
    livePreviewPosition = nil
  }

  private func beginPreventingSleepIfNeeded() {
    guard keepScreenAwake else { return }
    previousIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
    UIApplication.shared.isIdleTimerDisabled = true
  }

  private func endPreventingSleepIfNeeded() {
    guard let previousIdleTimerDisabled else { return }
    UIApplication.shared.isIdleTimerDisabled = previousIdleTimerDisabled
    self.previousIdleTimerDisabled = nil
  }

  private struct ParallelRestorationShard: Sendable {
    let index: Int
    let lane: Int
    let coreFrameCount: Int
    let request: MiohClusterJobRequest
    let outputURL: URL
  }

  private struct ParallelRestorationResult: Sendable {
    let index: Int
    let outputURL: URL
    let metrics: MiohClusterJobMetrics
  }

  private struct VideoFrameIndex: Sendable {
    let presentationTimestamps: [Int64]
    let endNanoseconds: Int64
  }

  private static func videoFrameIndex(
    inputURL: URL,
    durationNanoseconds: Int64
  ) async throws -> VideoFrameIndex {
    try await Task.detached(priority: .userInitiated) {
      let asset = AVURLAsset(url: inputURL)
      guard let track = try await asset.loadTracks(withMediaType: .video).first
      else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "並列復元用の動画トラックを確認できません"
        )
      }
      var timestamps: [Int64]
      if #available(iOS 26.0, *) {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        let provider = reader.outputProvider(for: output)
        do {
          try reader.start()
        } catch {
          throw MiohIPadWorkerEngineError.decoder(
            "並列復元用のフレーム走査を開始できません: "
              + error.localizedDescription
          )
        }
        defer { reader.cancelReading() }
        var collected: [Int64] = []
        collected.reserveCapacity(32_768)
        while let sample = try await provider.next() {
          try Task.checkCancellation()
          let pts = sample.presentationTimeStamp
          guard pts.isValid, !pts.isIndefinite else { continue }
          let seconds = pts.seconds
          guard seconds.isFinite, seconds >= 0 else { continue }
          collected.append(Int64((seconds * 1_000_000_000).rounded()))
        }
        guard reader.status == .completed else {
          throw reader.error
            ?? MiohIPadWorkerEngineError.decoder(
              "並列復元用のフレーム走査に失敗しました"
            )
        }
        timestamps = collected
      } else {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
          throw MiohIPadWorkerEngineError.decoder(
            "並列復元用のフレーム境界を読み取れません"
          )
        }
        reader.add(output)
        guard reader.startReading() else {
          throw reader.error
            ?? MiohIPadWorkerEngineError.decoder(
              "並列復元用のフレーム走査を開始できません"
            )
        }
        defer { reader.cancelReading() }
        var collected: [Int64] = []
        collected.reserveCapacity(32_768)
        while let sample = output.copyNextSampleBuffer() {
          try Task.checkCancellation()
          let pts = CMSampleBufferGetPresentationTimeStamp(sample)
          guard pts.isValid, !pts.isIndefinite else { continue }
          let seconds = pts.seconds
          guard seconds.isFinite, seconds >= 0 else { continue }
          collected.append(Int64((seconds * 1_000_000_000).rounded()))
        }
        guard reader.status == .completed else {
          throw reader.error
            ?? MiohIPadWorkerEngineError.decoder(
              "並列復元用のフレーム走査に失敗しました"
            )
        }
        timestamps = collected
      }
      timestamps.sort()
      var unique: [Int64] = []
      unique.reserveCapacity(timestamps.count)
      for timestamp in timestamps where unique.last != timestamp {
        unique.append(timestamp)
      }
      guard !unique.isEmpty else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "並列復元できる映像フレームがありません"
        )
      }
      return VideoFrameIndex(
        presentationTimestamps: unique,
        endNanoseconds: max(durationNanoseconds, (unique.last ?? 0) + 1)
      )
    }.value
  }

  private static func makeRestorationShards(
    inputURL: URL,
    inputExtension: String,
    inputByteCount: Int64,
    durationNanoseconds: Int64,
    options: MiohClusterRestorationOptions,
    requestedLanes: Int,
    maximumLanes: Int,
    directory: URL
  ) async throws -> [ParallelRestorationShard] {
    let laneLimit = min(3, max(1, min(requestedLanes, maximumLanes)))
    let inputPath = try MiohClusterRelativePath(
      validating: "input.\(inputExtension)"
    )

    func request(
      index: Int,
      range: MiohClusterMediaRange
    ) throws -> MiohClusterJobRequest {
      let now = Date()
      return MiohClusterJobRequest(
        jobID: UUID(),
        attemptID: UUID(),
        leaseID: UUID(),
        coordinatorNodeID: UUID(),
        sharedRootIdentifier: "ipad-standalone-parallel",
        inputByteCount: inputByteCount,
        inputSHA256: String(repeating: "0", count: 64),
        inputRelativePath: inputPath,
        outputRelativePath: try MiohClusterRelativePath(
          validating: String(format: "restored-%02d.mp4", index)
        ),
        mediaRange: range,
        options: options,
        createdAt: now,
        leaseExpiresAt: now.addingTimeInterval(24 * 60 * 60)
      )
    }

    if laneLimit == 1 {
      let range = MiohClusterMediaRange(
        decodeStartNanoseconds: 0,
        decodeEndNanoseconds: durationNanoseconds,
        coreStartNanoseconds: 0,
        coreEndNanoseconds: durationNanoseconds,
        leadingOverlapFrames: 0,
        trailingOverlapFrames: 0
      )
      return [
        ParallelRestorationShard(
          index: 0,
          lane: 0,
          coreFrameCount: 1,
          request: try request(index: 0, range: range),
          outputURL: directory.appendingPathComponent("restored-00.mp4")
        )
      ]
    }

    let media = try await videoFrameIndex(
      inputURL: inputURL,
      durationNanoseconds: durationNanoseconds
    )
    let timestamps = media.presentationTimestamps
    let totalFrames = timestamps.count
    let stride = options.restorationClipLength - options.temporalOverlap
    guard stride > 0 else {
      throw MiohIPadWorkerEngineError.unsupportedJob(
        "並列復元のクリップstrideが不正です"
      )
    }
    let totalStrides = max(1, (totalFrames + stride - 1) / stride)
    let effectiveLanes = min(laneLimit, totalStrides)

    func timestamp(_ frame: Int) -> Int64 {
      if frame <= 0 { return 0 }
      if frame >= totalFrames { return media.endNanoseconds }
      return timestamps[frame]
    }

    var shards: [ParallelRestorationShard] = []
    shards.reserveCapacity(effectiveLanes)
    var coreStart = 0
    var remainingStrides = totalStrides
    for lane in 0..<effectiveLanes {
      let remainingLanes = effectiveLanes - lane
      let assignedStrides = max(
        1,
        (remainingStrides + remainingLanes - 1) / remainingLanes
      )
      let coreEnd = min(totalFrames, coreStart + assignedStrides * stride)
      let decodeStart = coreStart == 0 ? 0 : max(0, coreStart - stride)
      let decodeEnd = min(totalFrames, coreEnd + options.temporalOverlap)
      let range = MiohClusterMediaRange(
        decodeStartNanoseconds: timestamp(decodeStart),
        decodeEndNanoseconds: timestamp(decodeEnd),
        coreStartNanoseconds: timestamp(coreStart),
        coreEndNanoseconds: timestamp(coreEnd),
        leadingOverlapFrames: coreStart - decodeStart,
        trailingOverlapFrames: decodeEnd - coreEnd
      )
      guard range.isValid, coreEnd > coreStart else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "並列復元のフレーム区間を作成できません"
        )
      }
      shards.append(
        ParallelRestorationShard(
          index: lane,
          lane: lane,
          coreFrameCount: coreEnd - coreStart,
          request: try request(index: lane, range: range),
          outputURL: directory.appendingPathComponent(
            String(format: "restored-%02d.mp4", lane)
          )
        )
      )
      coreStart = coreEnd
      remainingStrides -= assignedStrides
    }
    guard coreStart == totalFrames, !shards.isEmpty else {
      throw MiohIPadWorkerEngineError.internalFailure(
        "並列復元区間が動画全体を覆っていません"
      )
    }
    return shards
  }

  private func run(
    inputURL: URL,
    prepared: MiohIPadPreparedWorker,
    generation: UUID
  ) async throws {
    state = .preparing
    normalizeClipSettings()
    guard
      let restorationDigest =
        prepared.restorationAssetSHA256ByIdentifier[restorationModelIdentifier],
      let detectorDigest =
        prepared.detectorAssetSHA256ByIdentifier[detectorModelIdentifier]
    else {
      throw MiohIPadWorkerEngineError.unsupportedJob(
        "選択したモデルが検証済み一覧にありません"
      )
    }

    let access = inputURL.startAccessingSecurityScopedResource()
    defer { if access { inputURL.stopAccessingSecurityScopedResource() } }
    let asset = AVURLAsset(url: inputURL)
    let duration = try await asset.load(.duration)
    guard duration.isNumeric, duration.seconds.isFinite,
      duration.seconds > 0, duration.seconds <= 9_000_000_000
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia("動画時間が不正です")
    }
    let durationNanoseconds = max(
      1,
      Int64((duration.seconds * 1_000_000_000).rounded())
    )
    let inputExtension = inputURL.pathExtension.lowercased()
    guard ["mp4", "mov", "m4v"].contains(inputExtension) else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "MP4/MOV/M4Vを選択してください"
      )
    }
    let options = MiohClusterRestorationOptions(
      restorationModelIdentifier: restorationModelIdentifier,
      restorationAssetSHA256: restorationDigest,
      detectorModelIdentifier: detectorModelIdentifier,
      detectorAssetSHA256: detectorDigest,
      restorationClipLength: clipLength,
      temporalOverlap: temporalOverlap,
      crossfade: crossfade,
      detectionEmptyLookahead: detectionEmptyLookahead,
      detectFaceMosaics: detectFaceMosaics,
      blendFeather: blendFeather,
      sharpenStrength: 0,
      detailBoost: 0,
      textureMix: 0,
      smoothStrength: 0,
      effectUpscale: 1,
      videoCodec: videoCodec,
      bitrateMultiplier: bitrateMultiplier,
      mp4FastStart: mp4FastStart,
      targetFPSNumerator: useFPS ? targetFPSNumerator : nil,
      targetFPSDenominator: useFPS ? targetFPSDenominator : nil
    )
    let runID = UUID().uuidString.lowercased()
    let temporaryDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "mioh-restored-parallel-\(runID)",
        isDirectory: true
      )
    let finalURL = try Self.outputURL(inputURL: inputURL, runID: runID)
    try FileManager.default.createDirectory(
      at: temporaryDirectory,
      withIntermediateDirectories: false
    )
    try? FileManager.default.removeItem(at: finalURL)
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

    let inputByteCount = Int64(
      (try? inputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    )
    let shards = try await Self.makeRestorationShards(
      inputURL: inputURL,
      inputExtension: inputExtension,
      inputByteCount: inputByteCount,
      durationNanoseconds: durationNanoseconds,
      options: options,
      requestedLanes: parallelRestorationLanes,
      maximumLanes: prepared.maximumParallelRestorationLanes,
      directory: temporaryDirectory
    )
    try Task.checkCancellation()

    state = .restoring
    appendLog(
      shards.count == 1
        ? "選択したCore AIモデルを読み込み、1laneで復元を開始しました。"
        : "独立したCore AI runnerを\(shards.count)つ読み込み、並列復元を開始しました。"
    )
    let previewHandler: (@Sendable (MiohIPadPreviewFrame) async -> Void)?
    if livePreviewEnabled {
      previewHandler = { [weak self] frame in
        await MainActor.run { [weak self] in
          guard let self, self.previewGeneration == generation, self.isRunning
          else { return }
          self.livePreviewImage = frame.image
          self.livePreviewPosition =
            max(0, Double(frame.ptsNanoseconds) / 1_000_000_000)
        }
        await Task.yield()
      }
    } else {
      previewHandler = nil
    }
    let restorationStartedAt = Date()
    let progressTracker = IPadStandaloneParallelProgress(
      weights: shards.map(\.coreFrameCount)
    )
    let results: [ParallelRestorationResult]
    do {
      results = try await withThrowingTaskGroup(
        of: ParallelRestorationResult.self,
        returning: [ParallelRestorationResult].self
      ) { group in
        for shard in shards {
          group.addTask {
            let lanePreview = shard.index == 0 ? previewHandler : nil
            let metrics = try await prepared.executeLocal(
              request: shard.request,
              inputURL: inputURL,
              outputURL: shard.outputURL,
              lane: shard.lane,
              progress: { [weak self] value in
                Task {
                  let aggregate = await progressTracker.update(
                    lane: shard.index,
                    value: value
                  )
                  await MainActor.run { [weak self] in
                    guard let self,
                      self.previewGeneration == generation
                    else { return }
                    self.progress = min(0.9, max(0, aggregate * 0.9))
                  }
                }
              },
              preview: lanePreview
            )
            return ParallelRestorationResult(
              index: shard.index,
              outputURL: shard.outputURL,
              metrics: metrics
            )
          }
        }
        var completed: [ParallelRestorationResult] = []
        completed.reserveCapacity(shards.count)
        for try await result in group { completed.append(result) }
        return completed.sorted { $0.index < $1.index }
      }
    } catch {
      if shards.count > 1 {
        await prepared.releaseSupplementalRestorationLanes()
      }
      throw error
    }
    if shards.count > 1 {
      await prepared.releaseSupplementalRestorationLanes()
    }
    try Task.checkCancellation()
    appendLog(
      shards.count == 1
        ? "映像復元が完了しました。元動画の音声を結合します。"
        : "\(shards.count)個の並列映像を元の順序で結合し、音声を戻します。"
    )
    state = .muxingAudio
    progress = 0.92
    try await Self.muxOriginalAudio(
      restoredVideoURLs: results.map(\.outputURL),
      originalURL: inputURL,
      outputURL: finalURL
    )
    try Task.checkCancellation()
    let finalByteCount = Int64(
      (try? finalURL.resourceValues(forKeys: [.fileSizeKey]).fileSize)
        ?? Int(results.reduce(Int64(0)) { $0 + $1.metrics.outputByteCount })
    )
    let processedFrames = results.reduce(0) { $0 + $1.metrics.processedFrames }
    let restoredFrames = results.reduce(0) {
      $0 + ($1.metrics.restoredFrames ?? 0)
    }
    let restorationSeconds = results.reduce(0.0) {
      $0 + ($1.metrics.restorationSeconds ?? 0)
    }
    let restorationPreparationSeconds = results.reduce(0.0) {
      $0 + ($1.metrics.restorationPreparationSeconds ?? 0)
    }
    let restorationCompositingSeconds = results.reduce(0.0) {
      $0 + ($1.metrics.restorationCompositingSeconds ?? 0)
    }
    let wallSeconds = max(
      0.001,
      Date().timeIntervalSince(restorationStartedAt)
    )
    metrics = MiohClusterJobMetrics(
      processedFrames: processedFrames,
      wallSeconds: wallSeconds,
      outputByteCount: finalByteCount,
      restoredFrames: restoredFrames,
      restorationSeconds: restorationSeconds,
      restorationPreparationSeconds: restorationPreparationSeconds,
      restorationCompositingSeconds: restorationCompositingSeconds
    )
    outputURL = finalURL
    progress = 1
    state = .completed
    appendLog(
      String(
        format: "復元完了: %dフレーム / %.1f秒 / %@",
        processedFrames,
        wallSeconds,
        finalURL.lastPathComponent
      ),
      level: .success
    )
  }

  private static func outputURL(inputURL: URL, runID: String) throws -> URL {
    let documents = try FileManager.default.url(
      for: .documentDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true
    )
    let rawBase = inputURL.deletingPathExtension().lastPathComponent
    let base = rawBase.isEmpty ? "mioh" : rawBase
    return documents.appendingPathComponent(
      "\(base)-mioh-restored-\(runID.prefix(8)).mp4"
    )
  }

  private static func muxOriginalAudio(
    restoredVideoURLs: [URL],
    originalURL: URL,
    outputURL: URL
  ) async throws {
    guard !restoredVideoURLs.isEmpty else {
      throw MiohIPadWorkerEngineError.output("復元映像がありません")
    }
    let restoredAssets = restoredVideoURLs.map { AVURLAsset(url: $0) }
    let original = AVURLAsset(url: originalURL)
    let audioTracks = try await original.loadTracks(withMediaType: .audio)
    if audioTracks.isEmpty, restoredVideoURLs.count == 1 {
      try FileManager.default.moveItem(
        at: restoredVideoURLs[0],
        to: outputURL
      )
      return
    }
    let composition = AVMutableComposition()
    guard
      let compositionVideo = composition.addMutableTrack(
        withMediaType: .video,
        preferredTrackID: kCMPersistentTrackID_Invalid
      )
    else {
      throw MiohIPadWorkerEngineError.output("映像結合トラックを作成できません")
    }
    var videoDuration = CMTime.zero
    for restored in restoredAssets {
      try Task.checkCancellation()
      let videoTracks = try await restored.loadTracks(withMediaType: .video)
      guard let videoTrack = videoTracks.first else {
        throw MiohIPadWorkerEngineError.output(
          "復元動画トラックがありません"
        )
      }
      let shardDuration = try await restored.load(.duration)
      guard shardDuration.isNumeric, shardDuration > .zero else {
        throw MiohIPadWorkerEngineError.output(
          "復元動画の区間時間が不正です"
        )
      }
      try compositionVideo.insertTimeRange(
        CMTimeRange(start: .zero, duration: shardDuration),
        of: videoTrack,
        at: videoDuration
      )
      videoDuration = videoDuration + shardDuration
    }
    for audioTrack in audioTracks {
      guard
        let compositionAudio = composition.addMutableTrack(
          withMediaType: .audio,
          preferredTrackID: kCMPersistentTrackID_Invalid
        )
      else { continue }
      let audioRange = try await audioTrack.load(.timeRange)
      let copyDuration = CMTimeMinimum(videoDuration, audioRange.duration)
      if copyDuration > .zero {
        try compositionAudio.insertTimeRange(
          CMTimeRange(start: audioRange.start, duration: copyDuration),
          of: audioTrack,
          at: .zero
        )
      }
    }
    guard
      let exporter = AVAssetExportSession(
        asset: composition,
        presetName: AVAssetExportPresetPassthrough
      )
    else {
      throw MiohIPadWorkerEngineError.output("音声結合を開始できません")
    }
    exporter.outputURL = outputURL
    exporter.outputFileType = .mp4
    exporter.shouldOptimizeForNetworkUse = true
    let sendableExporter = SendableExportSession(exporter)
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Void, Error>) in
      exporter.exportAsynchronously {
        let finishedExporter = sendableExporter.value
        switch finishedExporter.status {
        case .completed:
          continuation.resume()
        case .cancelled:
          continuation.resume(throwing: CancellationError())
        default:
          continuation.resume(
            throwing: finishedExporter.error
              ?? MiohIPadWorkerEngineError.output("音声結合に失敗しました")
          )
        }
      }
    }
  }
}

private final class IPadSecurityScopedLease: @unchecked Sendable {
  let url: URL
  private let active: Bool

  init(url: URL) {
    self.url = url
    active = url.startAccessingSecurityScopedResource()
  }

  deinit {
    if active { url.stopAccessingSecurityScopedResource() }
  }
}

private actor IPadLiveHLSPrefetchBuffer {
  struct BufferedSegment: Sendable {
    let mediaSegment: IPadHLSMediaSegment
    let timelineStart: Double
    let localURL: URL
    let byteCount: Int64
  }

  struct Rebase: Sendable {
    let targetSequence: Int64
    let targetTimelineStart: Double
    let windowTimelineStart: Double
    let latestTimelineEnd: Double
  }

  enum Delivery: Sendable {
    case waiting
    case segment(BufferedSegment)
    case rebase(Rebase)
    case ended(Double)
    case interactionRequired(URL?)
    case failed(String)
  }

  private let mediaURL: URL
  private let resolutionPolicy: IPadMediaURLResolutionPolicy
  private let requestContext: IPadMediaRequestContext?
  private let directory: URL
  private let startupSegmentCount: Int
  private let maximumBufferedSegments: Int
  private let maximumBufferedBytes: Int64
  private let downloader: IPadHLSResourceDownloader
  private var playlist: IPadHLSMediaPlaylist
  private var timelineStarts: [Int64: Double]
  private var nextSequence: Int64
  private var buffered: [BufferedSegment] = []
  private var bufferedByteCount: Int64 = 0
  private var pendingRebase: Rebase?
  private var latestTimelineEnd: Double
  private var interactionRequiredURL: URL?
  private var requiresInteraction = false
  private var failureMessage: String?
  private var reachedEnd = false
  private var stopped = false
  private var producerTask: Task<Void, Never>?
  private var lastRefresh = Date()

  init(
    mediaURL: URL,
    resolutionPolicy: IPadMediaURLResolutionPolicy,
    requestContext: IPadMediaRequestContext?,
    playlist: IPadHLSMediaPlaylist,
    directory: URL,
    requestedStartSeconds: Double,
    startupSegmentCount: Int,
    maximumBufferedSegments: Int,
    maximumBufferedBytes: Int64,
    resourceLoader: (any IPadHLSResourceLoading)? = nil
  ) {
    self.mediaURL = mediaURL
    self.resolutionPolicy = resolutionPolicy
    self.requestContext = requestContext
    self.playlist = playlist
    self.directory = directory
    self.startupSegmentCount = max(1, startupSegmentCount)
    self.maximumBufferedSegments = max(1, maximumBufferedSegments)
    self.maximumBufferedBytes = max(64 * 1_024 * 1_024, maximumBufferedBytes)
    downloader = IPadHLSResourceDownloader(
      maximumResourceBytes: 64 * 1_024 * 1_024,
      resolutionPolicy: resolutionPolicy,
      requestContext: requestContext,
      resourceLoader: resourceLoader
    )

    let initialStarts = Dictionary(
      playlist.segments.map { ($0.sequence, $0.startSeconds) },
      uniquingKeysWith: { first, _ in first }
    )
    timelineStarts = initialStarts
    let liveEdgeSegments = playlist.segments.suffix(max(1, startupSegmentCount))
    let liveEdgeStart = liveEdgeSegments.first ?? playlist.segments[0]
    let requestedSegment =
      playlist.segments.first {
        $0.startSeconds + $0.duration > requestedStartSeconds
      } ?? liveEdgeStart
    nextSequence = max(liveEdgeStart.sequence, requestedSegment.sequence)
    latestTimelineEnd =
      playlist.segments.compactMap { segment in
        initialStarts[segment.sequence].map { $0 + segment.duration }
      }.max() ?? playlist.duration
  }

  func start() {
    guard producerTask == nil, !stopped else { return }
    producerTask = Task { [weak self] in
      await self?.produce()
    }
  }

  func nextDelivery() -> Delivery {
    if let pendingRebase {
      self.pendingRebase = nil
      return .rebase(pendingRebase)
    }
    if !buffered.isEmpty {
      let next = buffered.removeFirst()
      bufferedByteCount = max(0, bufferedByteCount - next.byteCount)
      return .segment(next)
    }
    if requiresInteraction {
      return .interactionRequired(interactionRequiredURL)
    }
    if let failureMessage { return .failed(failureMessage) }
    if reachedEnd { return .ended(latestTimelineEnd) }
    return .waiting
  }

  func stop() async {
    guard !stopped else { return }
    stopped = true
    let task = producerTask
    producerTask = nil
    task?.cancel()
    downloader.cancel()
    await task?.value
    discardBufferedSegments()
  }

  private func produce() async {
    var consecutiveRefreshFailures = 0
    var consecutiveDownloadFailures = 0
    do {
      while !stopped {
        try Task.checkCancellation()
        let refreshInterval = max(
          0.5,
          min(2, (playlist.targetDuration ?? 2) / 2)
        )
        let hasNextSegment = playlist.segments.contains {
          $0.sequence >= nextSequence
        }
        if !hasNextSegment
          || Date().timeIntervalSince(lastRefresh) >= refreshInterval
        {
          do {
            try await refreshPlaylist()
            consecutiveRefreshFailures = 0
          } catch is CancellationError {
            throw CancellationError()
          } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
            throw IPadMediaURLResolverError.interactionRequired(challengedURL)
          } catch {
            consecutiveRefreshFailures += 1
            if consecutiveRefreshFailures >= 3 { throw error }
            try await sleep(seconds: min(refreshInterval, 1))
            continue
          }
        }

        guard
          let mediaSegment = playlist.segments
            .filter({ $0.sequence >= nextSequence })
            .min(by: { $0.sequence < $1.sequence })
        else {
          if playlist.isLive {
            try await sleep(seconds: min(refreshInterval, 0.5))
            continue
          }
          reachedEnd = true
          return
        }

        let timelineStart =
          timelineStarts[mediaSegment.sequence]
          ?? mediaSegment.startSeconds
        guard mediaSegment.sequence < Int64.max else {
          throw MiohIPadWorkerEngineError.unsupportedMedia(
            "HLSのsegment sequenceが上限を超えています"
          )
        }
        do {
          let localURL = try await downloader.materialize(
            segment: mediaSegment,
            in: directory
          )
          try Task.checkCancellation()
          guard !stopped else {
            try? FileManager.default.removeItem(at: localURL)
            throw CancellationError()
          }
          let byteCount = Int64(
            (try? localURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
          )
          append(
            BufferedSegment(
              mediaSegment: mediaSegment,
              timelineStart: timelineStart,
              localURL: localURL,
              byteCount: byteCount
            )
          )
          nextSequence = mediaSegment.sequence + 1
          consecutiveDownloadFailures = 0
        } catch is CancellationError {
          throw CancellationError()
        } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
          throw IPadMediaURLResolverError.interactionRequired(challengedURL)
        } catch {
          consecutiveDownloadFailures += 1
          if consecutiveDownloadFailures >= 3 { throw error }
          lastRefresh = .distantPast
          try await sleep(seconds: 0.25)
        }
      }
    } catch is CancellationError {
      return
    } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
      guard !stopped else { return }
      interactionRequiredURL = challengedURL
      requiresInteraction = true
    } catch {
      guard !stopped else { return }
      failureMessage = error.localizedDescription
    }
  }

  private func refreshPlaylist() async throws {
    let refreshedSource = try await IPadMediaURLResolver(
      resourceLoader: downloader.resourceLoader
    ).resolve(
      mediaURL.absoluteString,
      policy: resolutionPolicy,
      context: requestContext
    )
    guard refreshedSource.kind == .hls,
      let refreshed = refreshedSource.hlsPlaylist,
      !refreshed.segments.isEmpty
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLSプレイリストを更新できません"
      )
    }
    try mergeTimelineStarts(from: playlist, refreshed: refreshed)
    playlist = refreshed
    lastRefresh = Date()
    latestTimelineEnd = max(
      latestTimelineEnd,
      refreshed.segments.compactMap { segment in
        timelineStarts[segment.sequence].map { $0 + segment.duration }
      }.max() ?? latestTimelineEnd
    )

    if let firstSequence = refreshed.segments.first?.sequence,
      nextSequence < firstSequence
    {
      rebaseToLiveEdge()
    }
    pruneTimelineStarts()
  }

  private func mergeTimelineStarts(
    from previous: IPadHLSMediaPlaylist,
    refreshed: IPadHLSMediaPlaylist
  ) throws {
    if let overlap = refreshed.segments.first(where: {
      timelineStarts[$0.sequence] != nil
    }), let knownStart = timelineStarts[overlap.sequence] {
      let adjustment = knownStart - overlap.startSeconds
      for segment in refreshed.segments {
        timelineStarts[segment.sequence] = segment.startSeconds + adjustment
      }
      return
    }

    let previousEnd =
      previous.segments.compactMap { segment -> Double? in
        guard let start = timelineStarts[segment.sequence] else { return nil }
        return start + segment.duration
      }.max() ?? latestTimelineEnd
    let previousLastSequence = previous.segments.map(\.sequence).max()
    let refreshedFirstSequence = refreshed.segments.map(\.sequence).min()
    var missingCount = 0.0
    if let previousLastSequence, let refreshedFirstSequence {
      let (sequenceDelta, overflow) =
        refreshedFirstSequence
        .subtractingReportingOverflow(previousLastSequence)
      if !overflow, sequenceDelta > 1 {
        missingCount = Double(sequenceDelta - 1)
      }
    }
    let knownDurations =
      previous.segments.map(\.duration)
      + refreshed.segments.map(\.duration)
    let estimatedDuration =
      previous.targetDuration ?? refreshed.targetDuration
      ?? (knownDurations.reduce(0, +) / Double(max(1, knownDurations.count)))
    let skippedDuration = missingCount * estimatedDuration
    guard previousEnd.isFinite, estimatedDuration.isFinite,
      estimatedDuration > 0, skippedDuration.isFinite
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLSライブ配信の時間軸が上限を超えています"
      )
    }
    var cursor = previousEnd + skippedDuration
    for segment in refreshed.segments.sorted(by: {
      $0.sequence < $1.sequence
    }) {
      guard cursor.isFinite else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "HLSライブ配信の時間軸が上限を超えています"
        )
      }
      timelineStarts[segment.sequence] = cursor
      cursor += segment.duration
    }
  }

  private func append(_ segment: BufferedSegment) {
    buffered.append(segment)
    bufferedByteCount += segment.byteCount
    latestTimelineEnd = max(
      latestTimelineEnd,
      segment.timelineStart + segment.mediaSegment.duration
    )
    guard
      buffered.count > maximumBufferedSegments
        || bufferedByteCount > maximumBufferedBytes
    else { return }

    let desiredCount = min(startupSegmentCount, buffered.count)
    discardFirst(buffered.count - desiredCount)
    while bufferedByteCount > maximumBufferedBytes, buffered.count > 1 {
      discardFirst(1)
    }
    guard let target = buffered.first else { return }
    pendingRebase = makeRebase(target: target)
  }

  private func rebaseToLiveEdge() {
    discardBufferedSegments()
    let edge = playlist.segments.suffix(startupSegmentCount)
    guard let target = edge.first else { return }
    nextSequence = target.sequence
    let targetStart = timelineStarts[target.sequence] ?? target.startSeconds
    pendingRebase = Rebase(
      targetSequence: target.sequence,
      targetTimelineStart: targetStart,
      windowTimelineStart: currentWindowTimelineStart(fallback: targetStart),
      latestTimelineEnd: latestTimelineEnd
    )
  }

  private func makeRebase(target: BufferedSegment) -> Rebase {
    Rebase(
      targetSequence: target.mediaSegment.sequence,
      targetTimelineStart: target.timelineStart,
      windowTimelineStart: currentWindowTimelineStart(
        fallback: target.timelineStart
      ),
      latestTimelineEnd: latestTimelineEnd
    )
  }

  private func currentWindowTimelineStart(fallback: Double) -> Double {
    guard let first = playlist.segments.first else { return fallback }
    return timelineStarts[first.sequence] ?? fallback
  }

  private func discardFirst(_ count: Int) {
    guard count > 0 else { return }
    let removed = buffered.prefix(min(count, buffered.count))
    for segment in removed {
      bufferedByteCount = max(0, bufferedByteCount - segment.byteCount)
      try? FileManager.default.removeItem(at: segment.localURL)
    }
    buffered.removeFirst(min(count, buffered.count))
  }

  private func discardBufferedSegments() {
    for segment in buffered {
      try? FileManager.default.removeItem(at: segment.localURL)
    }
    buffered.removeAll(keepingCapacity: false)
    bufferedByteCount = 0
  }

  private func pruneTimelineStarts() {
    let retainedSequences = Set(
      playlist.segments.map(\.sequence)
        + buffered.map { $0.mediaSegment.sequence }
    )
    timelineStarts = timelineStarts.filter {
      retainedSequences.contains($0.key)
    }
  }

  private func sleep(seconds: Double) async throws {
    try await Task.sleep(
      nanoseconds: UInt64(max(0.01, seconds) * 1_000_000_000)
    )
  }
}

@MainActor
final class IPadRealtimePreviewController: ObservableObject {
  enum State: Equatable {
    case idle
    case loading
    case buffering
    case followingLiveEdge
    case playing
    case paused
    case ended
    case failed(String)

    var label: String {
      switch self {
      case .idle: "待機中"
      case .loading: "復元を準備中"
      case .buffering: "復元済み映像を先読み中"
      case .followingLiveEdge: "ライブ端へ追従中"
      case .playing: "復元しながら再生中"
      case .paused: "一時停止"
      case .ended: "再生終了"
      case .failed: "再生失敗"
      }
    }
  }

  private struct Segment {
    let sequence: Int
    let startSeconds: Double
    let endSeconds: Double
    let url: URL
  }

  private struct FileSegmentJob: Sendable {
    let sequence: Int
    let lane: Int
    let startNanoseconds: Int64
    let endNanoseconds: Int64
    let request: MiohClusterJobRequest
    let outputURL: URL

    var startSeconds: Double {
      Double(startNanoseconds) / 1_000_000_000
    }

    var endSeconds: Double {
      Double(endNanoseconds) / 1_000_000_000
    }
  }

  private struct FileSegmentResult: Sendable {
    let job: FileSegmentJob
    let metrics: MiohClusterJobMetrics?

    var isEmptyFinalSliver: Bool { metrics == nil }
  }

  private struct HLSRestorationSource {
    let mediaSegment: IPadHLSMediaSegment
    let timelineStart: Double
    let localURL: URL

    var timelineEnd: Double {
      timelineStart + mediaSegment.duration
    }
  }

  private struct DeferredSourcePlayerItem {
    let item: AVPlayerItem
    let source: IPadResolvedMediaSource?
    let generation: Int
  }

  @Published private(set) var state: State = .idle
  @Published private(set) var position = 0.0
  @Published private(set) var duration = 0.0
  @Published private(set) var bufferedSeconds = 0.0
  @Published private(set) var processingPosition = 0.0
  @Published private(set) var interactionRequiredURL: URL?
  @Published private(set) var hasPresentedRestoredFrame = false
  @Published private(set) var hlsTransportLabel: String?
  @Published private(set) var sftpBitsPerSecond = 0.0
  @Published private(set) var sftpActiveRangeReads = 0
  @Published private(set) var sftpRangeLatencySeconds = 0.0
  @Published private(set) var sftpCachedBytes = 0
  @Published private(set) var restorationRealtimeFactor = 0.0
  @Published private(set) var processingFramesPerSecond = 0.0
  @Published private(set) var restorationFramesPerSecond = 0.0
  @Published private(set) var recentRestoredFrameCount = 0
  @Published private(set) var restorationParallelLanes = 1
  @Published private(set) var cacheStorageConstrained = false
  @Published var showOriginal = false
  @Published var volume = 1.0 {
    didSet { applyVolume() }
  }
  @Published var muted = false {
    didSet { applyVolume() }
  }

  let sourcePlayer = AVPlayer()
  let restoredPlayer = AVQueuePlayer()

  private let segmentSeconds = 2.0
  private let startupSegmentCount = 3
  // Unlimited local look-ahead is stored as finalized MP4 segments on disk.
  // Keep only a bounded window materialized as AVPlayerItems so a long movie
  // does not turn its entire disk cache into an unbounded AVFoundation graph.
  private let maximumResidentRestoredSegments = 32
  // Restore one core segment at a time. Larger batches held back the next
  // playable segment until the whole batch completed on iPad.
  private let hlsInitialRestoreBatchCoreSegments = 1
  private let hlsSteadyRestoreBatchCoreSegments = 1
  private let rebufferSegmentCount = 2
  private let driftToleranceSeconds = 0.400
  private let driftCorrectionGraceSeconds = 0.350
  private let driftSeekToleranceSeconds = 0.100
  private let hlsDriftToleranceSeconds = 0.120
  private let hlsDriftSeekToleranceSeconds = 0.050
  private let streamingDriftToleranceSeconds = 0.080
  private let streamingDriftResumeToleranceSeconds = 0.035
  private let streamingDriftCorrectionGraceSeconds = 0.120
  private let streamingDriftSeekToleranceSeconds = 0.050
  private let clockObservationIntervalSeconds = 0.080
  private var preparedWorker: MiohIPadPreparedWorker?
  private var configuration: IPadRealtimePreviewConfiguration?
  private var productionTask: Task<Void, Never>?
  private var streamingMetricsTask: Task<Void, Never>?
  private var generation = 0
  private var requestedStartSeconds = 0.0
  private var shouldPlay = true
  private var generationHasStarted = false
  private var generationReachedEnd = false
  private var activeStartupSegmentCount = 3
  private var queuedSegments: [Segment] = []
  private struct RestorationPerformanceSample {
    let wallSeconds: Double
    let mediaSeconds: Double
    let processedFrames: Int?
    let restoredFrames: Int?
    let restorationSeconds: Double?
    let restorationPreparationSeconds: Double?
    let restorationCompositingSeconds: Double?
  }
  private var restorationPerformanceSamples: [RestorationPerformanceSample] = []
  private var lastRealtimeFailureMessage: String?
  private var itemSegments: [ObjectIdentifier: Segment] = [:]
  private var itemVideoOutputs: [ObjectIdentifier: AVPlayerItemVideoOutput] = [:]
  private var notificationTokens: [ObjectIdentifier: NSObjectProtocol] = [:]
  private var timeObserver: Any?
  private var sourceStatusObservation: NSKeyValueObservation?
  private var sourceTimeControlObservation: NSKeyValueObservation?
  private var sourceReady = false
  private var sourceSeekCompleted = false
  private var sourceTimeOffset = 0.0
  private var sourceSeekErrorSeconds = 0.0
  private var latestClockDriftSeconds = 0.0
  private var liveSourceWindowStartSeconds = 0.0
  private var sourceSeekRevision = 0
  private var sourceSeekAttemptID: UUID?
  private var sourceSeekTimeoutTask: Task<Void, Never>?
  private var deferredSourcePlayerItem: DeferredSourcePlayerItem?
  private var currentRestoredItemIdentifier: ObjectIdentifier?
  private var currentRestoredItemStartedAt = 0.0
  private var streamingRestoredHeldForSourceCatchup = false
  private var streamingClockCorrectionInFlight = false
  private var streamingClockCorrectionRevision = 0
  private(set) var latestRestoredPixelBuffer: CVPixelBuffer?
  private var sessionDirectory: URL?
  private var securityLease: IPadSecurityScopedLease?
  private var authenticatedMediaProxy: IPadAuthenticatedMediaProxy?
  private var hlsAVFoundationCapture: IPadHLSAVFoundationCapture?
  private var browserHandoffLease: IPadBrowserMediaHandoffLease?
  private var previousIdleTimerDisabled: Bool?

  init() {
    // The restored queue owns the long look-ahead policy. The source player is
    // only the audible 1x clock, so letting AVPlayer wait for its own large
    // buffer can hold the restored video indefinitely (for example when the
    // restoration look-ahead is configured to 60 seconds).
    sourcePlayer.automaticallyWaitsToMinimizeStalling = false
    // Every restored queue item is already a finalized local MP4. Letting this
    // player independently wait for a larger forward buffer makes it start
    // later than the SFTP source/audio clock even when both play calls are
    // adjacent.
    restoredPlayer.automaticallyWaitsToMinimizeStalling = false
    restoredPlayer.isMuted = true
    restoredPlayer.actionAtItemEnd = .advance
    installTimeObserver()
  }

  deinit {
    let retiringCapture = hlsAVFoundationCapture
    if let browserHandoffLease {
      Task { @MainActor in
        retiringCapture?.cancel()
        browserHandoffLease.beginEnding()
        await browserHandoffLease.end()
      }
    } else if let retiringCapture {
      Task { @MainActor in retiringCapture.cancel() }
    }
    authenticatedMediaProxy?.stop()
    if let timeObserver { sourcePlayer.removeTimeObserver(timeObserver) }
    sourceStatusObservation?.invalidate()
    sourceTimeControlObservation?.invalidate()
    for token in notificationTokens.values {
      NotificationCenter.default.removeObserver(token)
    }
    productionTask?.cancel()
    streamingMetricsTask?.cancel()
  }

  var isActive: Bool {
    switch state {
    case .loading, .buffering, .followingLiveEdge, .playing, .paused: true
    case .idle, .ended, .failed: false
    }
  }

  var hasPreview: Bool {
    generationHasStarted || !queuedSegments.isEmpty
  }

  var showsSourceFrame: Bool {
    // After the first restored frame, keep the persistent surface visible even
    // if production briefly leaves the queue empty. It holds the last image;
    // revealing AVPlayer's source layer here can expose its empty backing store.
    showOriginal || !generationHasStarted || !hasPresentedRestoredFrame
      || state == .followingLiveEdge
  }

  var isPlaybackRequested: Bool { shouldPlay }

  var isSeekable: Bool {
    configuration?.resolvedMediaSource?.hlsPlaylist?.isLive != true
  }

  var cachesLocalInputToEnd: Bool {
    configuration.map(usesUnlimitedLocalCache) ?? false
  }

  var pipelineStageLabel: String {
    if case .failed = state { return "停止（エラー）" }
    if state == .paused { return "一時停止" }
    if cacheStorageConstrained { return "空き容量の安全域待ち" }
    guard configuration?.inputRangeValidator != nil else {
      switch state {
      case .loading: return "入力準備"
      case .buffering: return "復元キュー待ち"
      case .followingLiveEdge: return "ライブ追従"
      case .playing:
        return restorationParallelLanes > 1
          ? "\(restorationParallelLanes)runner並列処理中"
          : "連続処理中"
      case .ended: return "完了"
      case .idle: return "待機"
      case .paused, .failed: return state.label
      }
    }
    if !sourceReady { return "SFTPメタデータ待ち" }
    if !sourceSeekCompleted { return "SFTPシークRange待ち" }
    if state == .buffering {
      if sftpActiveRangeReads > 0 { return "SFTP Range取得中" }
      if restorationRealtimeFactor > 1 { return "Core AI復元待ち" }
      return "復元キュー待ち"
    }
    if state == .playing { return "SFTP取得・復元・再生中" }
    if state == .ended { return "完了" }
    return state.label
  }

  func start(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    at requestedSeconds: Double = 0,
    autoPlay: Bool = true,
    startupSegments: Int? = nil,
    browserHandoffLease: IPadBrowserMediaHandoffLease? = nil
  ) {
    interactionRequiredURL = nil
    let continuingLiveDuration =
      configuration.resolvedMediaSource?.hlsPlaylist?.isLive == true
        && self.configuration?.inputURL == configuration.inputURL
      ? duration : 0
    let availableDuration = max(
      configuration.durationSeconds,
      continuingLiveDuration
    )
    let requestedTarget = max(
      requestedSeconds.isFinite ? requestedSeconds : 0,
      0
    )
    let liveDefaultTarget: Double
    if requestedTarget == 0,
      let playlist = configuration.resolvedMediaSource?.hlsPlaylist,
      playlist.isLive
    {
      liveDefaultTarget =
        playlist.segments.suffix(startupSegmentCount).first?.startSeconds
        ?? max(0, playlist.duration - configuration.bufferLimitSeconds)
    } else {
      liveDefaultTarget = requestedTarget
    }
    let target = min(
      liveDefaultTarget,
      max(availableDuration - 0.001, 0)
    )
    let retiringTask = productionTask
    let retiringDirectory = sessionDirectory
    let retiringLease = securityLease
    let retiringProxy = authenticatedMediaProxy
    let retiringCapture = hlsAVFoundationCapture
    let retiringBrowserHandoffLease: IPadBrowserMediaHandoffLease?
    if let current = self.browserHandoffLease,
      let browserHandoffLease, current === browserHandoffLease
    {
      retiringBrowserHandoffLease = nil
    } else {
      retiringBrowserHandoffLease = self.browserHandoffLease
      retiringBrowserHandoffLease?.beginEnding()
      self.browserHandoffLease = browserHandoffLease
    }
    retiringTask?.cancel()
    streamingMetricsTask?.cancel()
    streamingMetricsTask = nil
    productionTask = nil
    sessionDirectory = nil
    securityLease = nil
    authenticatedMediaProxy = nil
    hlsAVFoundationCapture = nil
    retiringProxy?.stop()
    retiringCapture?.cancel()

    generation += 1
    let startingGeneration = generation
    resetPlayersAndQueue()
    // The old AVPlayer item has now been detached, so it cannot reopen a
    // loopback connection after the reset. Retire the previous SFTP HTTP
    // generation before installing either the new source item or its worker
    // reader; this ordering matters when seeking repeatedly.
    configuration.prepareInputForSeek?()
    preparedWorker = prepared
    self.configuration = configuration
    requestedStartSeconds = target
    liveSourceWindowStartSeconds =
      configuration.resolvedMediaSource?.hlsPlaylist?.isLive == true
      ? (configuration.resolvedMediaSource?.hlsPlaylist?.segments.first?.startSeconds ?? 0)
      : 0
    position = target
    duration = availableDuration
    processingPosition = target
    sftpBitsPerSecond = 0
    sftpActiveRangeReads = 0
    sftpRangeLatencySeconds = 0
    sftpCachedBytes = 0
    restorationRealtimeFactor = 0
    processingFramesPerSecond = 0
    restorationFramesPerSecond = 0
    recentRestoredFrameCount = 0
    restorationParallelLanes = 1
    restorationPerformanceSamples.removeAll(keepingCapacity: true)
    lastRealtimeFailureMessage = nil
    cacheStorageConstrained = false
    shouldPlay = autoPlay
    generationHasStarted = false
    generationReachedEnd = false
    activeStartupSegmentCount = min(
      startupSegmentCount,
      max(1, startupSegments ?? startupSegmentCount)
    )
    hlsTransportLabel =
      configuration.resolvedMediaSource?.kind == .hls
      ? (configuration.hlsStreamingMode == .safariCompatible
        ? (configuration.hlsResourceLoader == nil
          ? "Safari互換（AVFoundation・暗号化HLS対応）"
          : "Safari互換（WebKit通信＋AVFoundation）")
        : "高速")
      : nil
    state = autoPlay ? .loading : .paused
    beginPreventingSleep(if: configuration.keepScreenAwake)
    beginStreamingMetricsPolling(
      configuration: configuration,
      generation: startingGeneration
    )

    let lease = IPadSecurityScopedLease(url: configuration.inputURL)
    securityLease = lease
    let session = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-ipad-preview-\(UUID().uuidString)",
      isDirectory: true
    )
    do {
      try FileManager.default.createDirectory(
        at: session,
        withIntermediateDirectories: true
      )
    } catch {
      fail(error.localizedDescription)
      Task {
        if let retiringTask { await retiringTask.value }
        _ = retiringLease
        if let retiringDirectory {
          try? FileManager.default.removeItem(at: retiringDirectory)
        }
      }
      return
    }
    sessionDirectory = session
    let mediaProxy: IPadAuthenticatedMediaProxy?
    let sourceRequiresMediaProxy: Bool
    let usesSafariCompatibleHLS =
      configuration.resolvedMediaSource?.kind == .hls
      && configuration.hlsStreamingMode == .safariCompatible
    if let source = configuration.resolvedMediaSource, source.kind == .hls,
      !usesSafariCompatibleHLS || configuration.hlsResourceLoader != nil
    {
      if configuration.hlsResourceLoader != nil {
        sourceRequiresMediaProxy = true
      } else {
        switch source.resolutionPolicy {
        case .publicDiscovered, .visibleBrowserDiscovered:
          sourceRequiresMediaProxy = true
        case .userSubmitted, .submittedPageSameOrigin:
          sourceRequiresMediaProxy = source.requestContext != nil
        }
      }
    } else {
      sourceRequiresMediaProxy = false
    }
    if sourceRequiresMediaProxy {
      let proxy = IPadAuthenticatedMediaProxy(
        resourceLoader: configuration.hlsResourceLoader
      ) {
        [weak self] challengedURL in
        Task { @MainActor [weak self] in
          self?.requireBrowserInteraction(
            challengedURL,
            source: configuration.resolvedMediaSource,
            generation: startingGeneration
          )
        }
      }
      authenticatedMediaProxy = proxy
      mediaProxy = proxy
    } else if !usesSafariCompatibleHLS {
      mediaProxy = nil
      prepareSourcePlayerItem(
        url: configuration.inputURL,
        source: configuration.resolvedMediaSource,
        generation: startingGeneration,
        appliesOriginContext: true,
        deferUntilFirstRestoredSegment:
          configuration.resolvedMediaSource?.kind == .hls
          && configuration.resolvedMediaSource?.hlsPlaylist?.isLive != true
      )
    } else {
      mediaProxy = nil
    }

    productionTask = Task { [weak self] in
      let proxiedPlaybackURL: URL?
      if let mediaProxy, let source = configuration.resolvedMediaSource {
        do {
          try await mediaProxy.start()
          let validatedPlaylistURL =
            source.hlsPlaylist?.url
            ?? source.mediaURL
          if let masterMetadata = source.hlsPlaylist?.masterMetadata,
            masterMetadata.hasSeparateAudio
          {
            proxiedPlaybackURL = try mediaProxy.localURL(
              forSelectedHLSMaster: masterMetadata,
              context: source.requestContext,
              resolutionPolicy: source.resolutionPolicy
            )
          } else {
            proxiedPlaybackURL = try mediaProxy.localURL(
              for: validatedPlaylistURL,
              context: source.requestContext,
              isPlaylist: true,
              resolutionPolicy: source.resolutionPolicy
            )
          }
        } catch is CancellationError {
          mediaProxy.stop()
          try? FileManager.default.removeItem(at: session)
          return
        } catch {
          mediaProxy.stop()
          guard let self, self.generation == startingGeneration else {
            try? FileManager.default.removeItem(at: session)
            return
          }
          self.authenticatedMediaProxy = nil
          self.fail("認証付きHLSの再生準備に失敗しました: \(error.localizedDescription)")
          return
        }
      } else {
        proxiedPlaybackURL = nil
      }

      if let retiringTask { await retiringTask.value }
      if let retiringBrowserHandoffLease {
        await retiringBrowserHandoffLease.end()
      }
      _ = retiringLease
      if let retiringDirectory {
        try? FileManager.default.removeItem(at: retiringDirectory)
      }
      guard let self, self.generation == startingGeneration else {
        try? FileManager.default.removeItem(at: session)
        return
      }
      if let proxiedPlaybackURL, !usesSafariCompatibleHLS {
        self.prepareSourcePlayerItem(
          url: proxiedPlaybackURL,
          source: configuration.resolvedMediaSource,
          generation: startingGeneration,
          appliesOriginContext: false,
          deferUntilFirstRestoredSegment:
            configuration.resolvedMediaSource?.hlsPlaylist?.isLive != true
        )
      }
      if usesSafariCompatibleHLS,
        let source = configuration.resolvedMediaSource,
        let playlist = source.hlsPlaylist
      {
        let safariPlaybackURL =
          configuration.hlsQualityPreference == .automatic
          ? source.playbackURL : source.mediaURL
        let captureAsset =
          proxiedPlaybackURL.map(AVURLAsset.init(url:))
          ?? makeIPadMediaAsset(
            url: safariPlaybackURL,
            source: source
          )
        if playlist.isLive {
          // A live edge has no future inventory to pull. Keep AVFoundation as
          // the clock-owning decoder and process frames at the available rate.
          let capture = IPadHLSAVFoundationCapture(
            asset: captureAsset,
            outputDirectory: session.appendingPathComponent(
              "avfoundation-capture",
              isDirectory: true
            ),
            startSeconds: target,
            duration: availableDuration,
            isLive: true,
            generation: startingGeneration,
            segmentSeconds: self.segmentSeconds,
            forwardBufferSeconds: configuration.bufferLimitSeconds
          )
          self.hlsAVFoundationCapture = capture
          self.prepareSourcePlayerItem(
            capture.makePlaybackItem(),
            source: source,
            generation: startingGeneration,
            deferUntilFirstRestoredSegment: false
          )
          await self.produceHLSAVFoundationSegments(
            capture: capture,
            prepared: prepared,
            configuration: configuration,
            playlist: playlist,
            session: session,
            startSeconds: target,
            generation: startingGeneration
          )
        } else {
          // VOD must not be paced by an AVPlayer playback clock. The browser
          // resource loader retains Safari cookies/headers while complete HLS
          // segments (including AES-128 segments) are downloaded, decrypted
          // and decoded as quickly as transport and restoration allow.
          self.hlsTransportLabel = "Safari互換・全速区間復元"
          self.prepareSourcePlayerItem(
            AVPlayerItem(asset: captureAsset),
            source: source,
            generation: startingGeneration,
            deferUntilFirstRestoredSegment: true
          )
          await self.produceHLSStreamSegments(
            prepared: prepared,
            configuration: configuration,
            session: session,
            startSeconds: target,
            generation: startingGeneration
          )
        }
      } else {
        await self.produceSegments(
          prepared: prepared,
          configuration: configuration,
          session: session,
          startSeconds: target,
          generation: startingGeneration
        )
      }
    }
  }

  private func installSourcePlayerItem(
    _ sourceItem: AVPlayerItem,
    source: IPadResolvedMediaSource?,
    generation: Int
  ) {
    if source?.kind == .hls {
      sourceItem.preferredMaximumResolution = CGSize(
        width: IPadRestorationMediaLimits.maximumLongEdge,
        height: IPadRestorationMediaLimits.maximumLongEdge
      )
      // Do not copy the restoration look-ahead (which may be tens of seconds)
      // onto the audible source clock. One restored segment is enough for the
      // source player to start while restoration continues ahead of it.
      sourceItem.preferredForwardBufferDuration = segmentSeconds
    }
    sourcePlayer.replaceCurrentItem(with: sourceItem)
    installSourceObservers(item: sourceItem, generation: generation)
    applyVolume()
  }

  private func prepareSourcePlayerItem(
    _ sourceItem: AVPlayerItem,
    source: IPadResolvedMediaSource?,
    generation: Int,
    deferUntilFirstRestoredSegment: Bool
  ) {
    if deferUntilFirstRestoredSegment {
      deferredSourcePlayerItem = DeferredSourcePlayerItem(
        item: sourceItem,
        source: source,
        generation: generation
      )
      return
    }
    deferredSourcePlayerItem = nil
    installSourcePlayerItem(
      sourceItem,
      source: source,
      generation: generation
    )
  }

  private func prepareSourcePlayerItem(
    url: URL,
    source: IPadResolvedMediaSource?,
    generation: Int,
    appliesOriginContext: Bool,
    deferUntilFirstRestoredSegment: Bool
  ) {
    let sourceAsset =
      appliesOriginContext
      ? makeIPadMediaAsset(url: url, source: source)
      : AVURLAsset(url: url)
    let sourceItem = AVPlayerItem(asset: sourceAsset)
    prepareSourcePlayerItem(
      sourceItem,
      source: source,
      generation: generation,
      deferUntilFirstRestoredSegment: deferUntilFirstRestoredSegment
    )
  }

  private func installDeferredSourcePlayerItemIfNeeded() {
    guard let deferred = deferredSourcePlayerItem else { return }
    guard deferred.generation == generation else {
      deferredSourcePlayerItem = nil
      return
    }
    guard sourcePlayer.currentItem == nil else {
      deferredSourcePlayerItem = nil
      return
    }
    deferredSourcePlayerItem = nil
    installSourcePlayerItem(
      deferred.item,
      source: deferred.source,
      generation: deferred.generation
    )
  }

  func togglePlayback() {
    switch state {
    case .playing:
      shouldPlay = false
      sourcePlayer.pause()
      restoredPlayer.pause()
      state = .paused
    case .loading, .buffering, .followingLiveEdge:
      if shouldPlay {
        shouldPlay = false
        sourcePlayer.pause()
        restoredPlayer.pause()
        state = .paused
      } else {
        shouldPlay = true
        state = .buffering
        resumeIfBuffered()
      }
    case .paused:
      shouldPlay = true
      state = .buffering
      resumeIfBuffered()
    case .ended, .failed, .idle:
      guard let preparedWorker, let configuration else { return }
      start(
        prepared: preparedWorker,
        configuration: configuration,
        at: state == .ended ? 0 : position,
        browserHandoffLease: browserHandoffLease
      )
    }
  }

  func seek(to seconds: Double) {
    guard let preparedWorker, let configuration else { return }
    guard isSeekable else { return }
    let resumeAfterSeek = state != .paused
    start(
      prepared: preparedWorker,
      configuration: configuration,
      at: seconds,
      autoPlay: resumeAfterSeek,
      startupSegments: 1,
      browserHandoffLease: browserHandoffLease
    )
  }

  func stop() {
    generation += 1
    let retiringTask = productionTask
    let retiringDirectory = sessionDirectory
    let retiringLease = securityLease
    let retiringProxy = authenticatedMediaProxy
    let retiringCapture = hlsAVFoundationCapture
    let retiringBrowserHandoffLease = browserHandoffLease
    retiringBrowserHandoffLease?.beginEnding()
    productionTask?.cancel()
    streamingMetricsTask?.cancel()
    streamingMetricsTask = nil
    productionTask = nil
    sessionDirectory = nil
    securityLease = nil
    authenticatedMediaProxy = nil
    hlsAVFoundationCapture = nil
    browserHandoffLease = nil
    retiringProxy?.stop()
    retiringCapture?.cancel()
    sourcePlayer.pause()
    restoredPlayer.pause()
    sourcePlayer.replaceCurrentItem(with: nil)
    resetPlayersAndQueue()
    interactionRequiredURL = nil
    state = .idle
    position = 0
    duration = configuration?.durationSeconds ?? 0
    processingPosition = 0
    hlsTransportLabel = nil
    endPreventingSleep()
    Task {
      if let retiringTask { await retiringTask.value }
      if let retiringBrowserHandoffLease {
        await retiringBrowserHandoffLease.end()
      }
      _ = retiringLease
      if let retiringDirectory {
        try? FileManager.default.removeItem(at: retiringDirectory)
      }
    }
  }

  func reportFailure(_ message: String) {
    fail(message)
  }

  func clearInteractionRequirement() {
    interactionRequiredURL = nil
  }

  private func produceSegments(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    session: URL,
    startSeconds: Double,
    generation: Int
  ) async {
    switch configuration.resolvedMediaSource?.kind {
    case .hls:
      await produceHLSStreamSegments(
        prepared: prepared,
        configuration: configuration,
        session: session,
        startSeconds: startSeconds,
        generation: generation
      )
    case .progressive, nil:
      await produceFileSegments(
        prepared: prepared,
        configuration: configuration,
        session: session,
        startSeconds: startSeconds,
        generation: generation
      )
    }
  }

  private func produceFileSegments(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    session: URL,
    startSeconds: Double,
    generation: Int
  ) async {
    do {
      let localLaneCount =
        usesUnlimitedLocalCache(configuration)
        ? min(
          3,
          max(
            1,
            min(
              configuration.parallelRestorationLanes,
              prepared.maximumParallelRestorationLanes
            )
          )
        )
        : 1
      restorationParallelLanes = localLaneCount
      if localLaneCount > 1 {
        do {
          try await produceParallelLocalFileSegments(
            prepared: prepared,
            configuration: configuration,
            session: session,
            startSeconds: startSeconds,
            generation: generation,
            laneCount: localLaneCount
          )
        } catch {
          await prepared.releaseSupplementalRestorationLanes()
          throw error
        }
        await prepared.releaseSupplementalRestorationLanes()
      } else {
        try await produceSerialFileSegments(
          prepared: prepared,
          configuration: configuration,
          session: session,
          startSeconds: startSeconds,
          generation: generation
        )
      }
      guard self.generation == generation else { return }
      generationReachedEnd = true
      if queuedSegments.isEmpty {
        shouldPlay = false
        position = duration
        state = .ended
        productionTask = nil
        releaseFinishedSession()
        return
      }
      resumeIfBuffered(endOfFile: true)
      productionTask = nil
    } catch is CancellationError {
      return
    } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
      guard self.generation == generation else { return }
      requireBrowserInteraction(
        challengedURL,
        source: configuration.resolvedMediaSource,
        generation: generation
      )
    } catch {
      guard self.generation == generation else { return }
      fail(hlsPlaybackFailureMessage(stage: "HLS区間の復元準備", error: error))
    }
  }

  private func produceSerialFileSegments(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    session: URL,
    startSeconds: Double,
    generation: Int
  ) async throws {
    var cursorNanoseconds = Int64(
      (startSeconds * 1_000_000_000).rounded()
    )
    var sequence = 0
    while cursorNanoseconds < configuration.durationNanoseconds {
      try Task.checkCancellation()
      if usesUnlimitedLocalCache(configuration) {
        try await waitForLocalCacheStorage(
          at: session,
          concurrentSegments: 1,
          generation: generation
        )
      } else {
        while bufferedSeconds >= configuration.bufferLimitSeconds {
          try Task.checkCancellation()
          try await Task.sleep(nanoseconds: 100_000_000)
        }
      }
      guard self.generation == generation else {
        throw CancellationError()
      }
      let job = try makeFileSegmentJob(
        configuration: configuration,
        session: session,
        cursorNanoseconds: cursorNanoseconds,
        sequence: sequence,
        lane: 0
      )
      let result = try await executeFileSegmentJob(
        prepared: prepared,
        configuration: configuration,
        job: job,
        generation: generation
      )
      try Task.checkCancellation()
      guard self.generation == generation else {
        throw CancellationError()
      }
      if result.isEmptyFinalSliver {
        processingPosition = configuration.durationSeconds
        return
      }
      guard let metrics = result.metrics else { return }
      recordRestorationPerformance(
        wallSeconds: metrics.wallSeconds,
        mediaSeconds: job.endSeconds - job.startSeconds,
        processedFrames: metrics.processedFrames,
        restoredFrames: metrics.restoredFrames,
        restorationSeconds: metrics.restorationSeconds,
        restorationPreparationSeconds:
          metrics.restorationPreparationSeconds,
        restorationCompositingSeconds:
          metrics.restorationCompositingSeconds
      )
      try commitFileSegment(job)
      cursorNanoseconds = job.endNanoseconds
      sequence += 1
    }
  }

  private func produceParallelLocalFileSegments(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    session: URL,
    startSeconds: Double,
    generation: Int,
    laneCount: Int
  ) async throws {
    var cursorNanoseconds = Int64(
      (startSeconds * 1_000_000_000).rounded()
    )
    var nextSequenceToSchedule = 0
    var nextSequenceToCommit = 0
    var availableLanes = Array(0..<laneCount)
    var ready: [Int: FileSegmentResult] = [:]
    var inFlight = 0
    var performanceCheckpoint = Date()

    try await withThrowingTaskGroup(of: FileSegmentResult.self) { group in
      while cursorNanoseconds < configuration.durationNanoseconds,
        !availableLanes.isEmpty,
        nextSequenceToSchedule - nextSequenceToCommit < laneCount
      {
        try await waitForLocalCacheStorage(
          at: session,
          concurrentSegments: laneCount,
          generation: generation
        )
        let lane = availableLanes.removeFirst()
        let job = try makeFileSegmentJob(
          configuration: configuration,
          session: session,
          cursorNanoseconds: cursorNanoseconds,
          sequence: nextSequenceToSchedule,
          lane: lane
        )
        cursorNanoseconds = job.endNanoseconds
        nextSequenceToSchedule += 1
        inFlight += 1
        group.addTask(priority: .userInitiated) { [weak self] in
          guard let self else { throw CancellationError() }
          return try await self.executeFileSegmentJob(
            prepared: prepared,
            configuration: configuration,
            job: job,
            generation: generation
          )
        }
      }

      while inFlight > 0 {
        try Task.checkCancellation()
        guard self.generation == generation else {
          throw CancellationError()
        }
        guard let result = try await group.next() else { break }
        inFlight -= 1
        availableLanes.append(result.job.lane)
        ready[result.job.sequence] = result

        var committedMediaSeconds = 0.0
        var committedFrames = 0
        var committedRestoredFrames = 0
        var committedRestorationSeconds = 0.0
        var committedPreparationSeconds = 0.0
        var committedCompositingSeconds = 0.0
        while let committed = ready.removeValue(
          forKey: nextSequenceToCommit
        ) {
          nextSequenceToCommit += 1
          if committed.isEmptyFinalSliver {
            processingPosition = configuration.durationSeconds
            continue
          }
          guard let metrics = committed.metrics else { continue }
          try commitFileSegment(committed.job)
          committedMediaSeconds +=
            committed.job.endSeconds - committed.job.startSeconds
          committedFrames += metrics.processedFrames
          committedRestoredFrames += metrics.restoredFrames ?? 0
          committedRestorationSeconds += metrics.restorationSeconds ?? 0
          committedPreparationSeconds +=
            metrics.restorationPreparationSeconds ?? 0
          committedCompositingSeconds +=
            metrics.restorationCompositingSeconds ?? 0
        }
        if committedMediaSeconds > 0 {
          let now = Date()
          recordRestorationPerformance(
            wallSeconds: max(0.001, now.timeIntervalSince(performanceCheckpoint)),
            mediaSeconds: committedMediaSeconds,
            processedFrames: committedFrames,
            restoredFrames: committedRestoredFrames,
            restorationSeconds: committedRestorationSeconds,
            restorationPreparationSeconds: committedPreparationSeconds,
            restorationCompositingSeconds: committedCompositingSeconds
          )
          performanceCheckpoint = now
        }

        while cursorNanoseconds < configuration.durationNanoseconds,
          !availableLanes.isEmpty,
          nextSequenceToSchedule - nextSequenceToCommit < laneCount
        {
          try await waitForLocalCacheStorage(
            at: session,
            concurrentSegments: laneCount,
            generation: generation
          )
          let lane = availableLanes.removeFirst()
          let job = try makeFileSegmentJob(
            configuration: configuration,
            session: session,
            cursorNanoseconds: cursorNanoseconds,
            sequence: nextSequenceToSchedule,
            lane: lane
          )
          cursorNanoseconds = job.endNanoseconds
          nextSequenceToSchedule += 1
          inFlight += 1
          group.addTask(priority: .userInitiated) { [weak self] in
            guard let self else { throw CancellationError() }
            return try await self.executeFileSegmentJob(
              prepared: prepared,
              configuration: configuration,
              job: job,
              generation: generation
            )
          }
        }
      }
    }
  }

  private func makeFileSegmentJob(
    configuration: IPadRealtimePreviewConfiguration,
    session: URL,
    cursorNanoseconds: Int64,
    sequence: Int,
    lane: Int
  ) throws -> FileSegmentJob {
    let endNanoseconds = min(
      configuration.durationNanoseconds,
      cursorNanoseconds
        + Int64((segmentSeconds * 1_000_000_000).rounded())
    )
    return FileSegmentJob(
      sequence: sequence,
      lane: lane,
      startNanoseconds: cursorNanoseconds,
      endNanoseconds: endNanoseconds,
      request: try configuration.request(
        coreStartNanoseconds: cursorNanoseconds,
        coreEndNanoseconds: endNanoseconds
      ),
      outputURL: session.appendingPathComponent(
        String(format: "segment-%06d.mp4", sequence)
      )
    )
  }

  private func executeFileSegmentJob(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    job: FileSegmentJob,
    generation: Int
  ) async throws -> FileSegmentResult {
    try? FileManager.default.removeItem(at: job.outputURL)
    do {
      let metrics = try await prepared.executeLocal(
        request: job.request,
        inputURL: configuration.inputURL,
        outputURL: job.outputURL,
        lane: job.lane,
        progress: { [weak self] value in
          Task { @MainActor in
            guard let self, self.generation == generation else { return }
            let lanePosition =
              job.startSeconds
              + (job.endSeconds - job.startSeconds)
              * max(0, min(1, value))
            self.processingPosition = max(
              self.processingPosition,
              min(job.endSeconds, lanePosition)
            )
          }
        }
      )
      return FileSegmentResult(job: job, metrics: metrics)
    } catch MiohIPadWorkerEngineError.unsupportedMedia(let detail)
      where job.endNanoseconds == configuration.durationNanoseconds
      && detail.contains("core range contains no decoded frame")
    {
      // Container duration can extend slightly beyond the final sample.
      // A frame-less final sliver is normal EOF, not a preview failure.
      try? FileManager.default.removeItem(at: job.outputURL)
      return FileSegmentResult(job: job, metrics: nil)
    }
  }

  private func commitFileSegment(_ job: FileSegmentJob) throws {
    try enqueue(
      Segment(
        sequence: job.sequence,
        startSeconds: job.startSeconds,
        endSeconds: job.endSeconds,
        url: job.outputURL
      )
    )
    processingPosition = max(processingPosition, job.endSeconds)
    resumeIfBuffered()
  }

  private func waitForLocalCacheStorage(
    at session: URL,
    concurrentSegments: Int,
    generation: Int
  ) async throws {
    while !Self.hasLocalCacheStorageHeadroom(
      at: session,
      concurrentSegments: concurrentSegments
    ) {
      try Task.checkCancellation()
      guard self.generation == generation else {
        throw CancellationError()
      }
      cacheStorageConstrained = true
      try await Task.sleep(nanoseconds: 500_000_000)
    }
    cacheStorageConstrained = false
  }

  /// Safari-compatible mode keeps playlist, key and cookie ownership in
  /// AVFoundation, then hands decoded IOSurface-backed frames directly to the
  /// long-lived worker. Only restored output is compressed for the bounded
  /// AVQueuePlayer buffer; there is no input MP4 encode/decode round trip.
  private func produceHLSAVFoundationSegments(
    capture: IPadHLSAVFoundationCapture,
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    playlist: IPadHLSMediaPlaylist,
    session: URL,
    startSeconds: Double,
    generation: Int
  ) async {
    var frameSession: (any MiohIPadRealtimeFrameSessioning)?
    var outputPipeline: IPadRealtimeRestoredOutputPipeline?
    var emergencyPolicy = IPadRealtimeEmergencyFPSPolicy()
    var lastTimelineEnd = startSeconds
    defer {
      capture.cancel()
    }

    do {
      let stream = try capture.frames()
      hlsTransportLabel = "Safari互換・直接フレーム復元"
      for try await captured in stream {
        try Task.checkCancellation()
        guard self.generation == generation else {
          throw CancellationError()
        }
        if frameSession == nil {
          let width = CVPixelBufferGetWidth(captured.pixelBuffer)
          let height = CVPixelBufferGetHeight(captured.pixelBuffer)
          let createdFrameSession = try await prepared.makeRealtimeFrameSession(
            options: configuration.options,
            width: width,
            height: height
          )
          await createdFrameSession.setDetectionMaskReuseSkipFrames(
            configuration.detectionMaskReuseSkipFrames
          )
          await createdFrameSession.setMaximumFrameRate(
            configuration.maximumFrameRate
          )
          switch configuration.realtimeFrameRateMode {
          case .source:
            await createdFrameSession.setEmergency24FPSEnabled(false)
            hlsTransportLabel =
              configuration.maximumFrameRate == nil
              ? "Safari互換・直接フレーム復元（入力fps維持）"
              : "Safari互換・直接フレーム復元（最大30fps）"
          case .fps24:
            await createdFrameSession.setEmergency24FPSEnabled(true)
            hlsTransportLabel = "Safari互換・直接フレーム復元（24fps固定）"
          case .automatic:
            await createdFrameSession.setEmergency24FPSEnabled(false)
            hlsTransportLabel =
              configuration.maximumFrameRate == nil
              ? "Safari互換・直接フレーム復元"
              : "Safari互換・直接フレーム復元（最大30fps）"
          }
          frameSession = createdFrameSession
          let writer = IPadRealtimeRestoredSegmentWriter(
            outputDirectory: session,
            generation: generation,
            width: width,
            height: height,
            videoCodec: configuration.options.videoCodec,
            bitrateMultiplier: configuration.options.bitrateMultiplier
          )
          outputPipeline = IPadRealtimeRestoredOutputPipeline(
            writer: writer
          ) { [weak self] completed in
            guard let self, self.generation == generation else {
              throw CancellationError()
            }
            try self.enqueue(
              Segment(
                sequence: completed.sequence,
                startSeconds: completed.startSeconds,
                endSeconds: completed.endSeconds,
                url: completed.url
              )
            )
            self.duration = max(self.duration, completed.endSeconds)
            self.resumeIfBuffered()
          }
        }
        guard let frameSession, let outputPipeline else {
          throw MiohIPadWorkerEngineError.internalFailure(
            "直接フレーム復元セッションを作成できません"
          )
        }
        let outputs = try await frameSession.append(
          MiohIPadRealtimeInputFrame(
            pixelBuffer: captured.pixelBuffer,
            ptsNanoseconds: captured.ptsNanoseconds
          )
        )
        capture.setRawFramesConsumed(through: captured.timelineSeconds)
        for sample in await frameSession.takePerformanceSamples() {
          recordRestorationPerformance(
            wallSeconds: sample.processingSeconds,
            mediaSeconds: sample.mediaSeconds,
            processedFrames: sample.processedFrames,
            restoredFrames: sample.restoredFrames,
            restorationSeconds: sample.modelRestorationSeconds,
            restorationPreparationSeconds:
              sample.restorationPreparationSeconds,
            restorationCompositingSeconds:
              sample.restorationCompositingSeconds
          )
          if configuration.realtimeFrameRateMode == .automatic,
            let enabled = emergencyPolicy.update(
              wallSeconds: sample.processingSeconds,
              mediaSeconds: sample.mediaSeconds,
              bufferedSeconds: bufferedSeconds,
              targetSeconds: configuration.bufferLimitSeconds,
              thermalState: ProcessInfo.processInfo.thermalState
            )
          {
            await frameSession.setEmergency24FPSEnabled(enabled)
            hlsTransportLabel =
              enabled
              ? "Safari互換・直接フレーム復元（24fps途切れ防止中）"
              : "Safari互換・直接フレーム復元"
          }
        }
        if !outputs.isEmpty {
          try await outputPipeline.submit(
            outputs,
            preferShortSegments: bufferedSeconds
              < max(6, configuration.bufferLimitSeconds * 0.90)
          )
          if let last = outputs.last {
            processingPosition = Double(last.ptsNanoseconds) / 1_000_000_000
            lastTimelineEnd = max(lastTimelineEnd, processingPosition)
          }
        }
      }

      if let frameSession, let outputPipeline {
        let finalOutputs = try await frameSession.flush()
        for sample in await frameSession.takePerformanceSamples() {
          recordRestorationPerformance(
            wallSeconds: sample.processingSeconds,
            mediaSeconds: sample.mediaSeconds,
            processedFrames: sample.processedFrames,
            restoredFrames: sample.restoredFrames,
            restorationSeconds: sample.modelRestorationSeconds,
            restorationPreparationSeconds:
              sample.restorationPreparationSeconds,
            restorationCompositingSeconds:
              sample.restorationCompositingSeconds
          )
          if configuration.realtimeFrameRateMode == .automatic {
            _ = emergencyPolicy.update(
              wallSeconds: sample.processingSeconds,
              mediaSeconds: sample.mediaSeconds,
              bufferedSeconds: bufferedSeconds,
              targetSeconds: configuration.bufferLimitSeconds,
              thermalState: ProcessInfo.processInfo.thermalState
            )
          }
        }
        try await outputPipeline.submit(
          finalOutputs,
          preferShortSegments: true
        )
        if let last = finalOutputs.last {
          processingPosition = Double(last.ptsNanoseconds) / 1_000_000_000
          lastTimelineEnd = max(lastTimelineEnd, processingPosition)
        }
        if let completed = try await outputPipeline.finish() {
          lastTimelineEnd = max(lastTimelineEnd, completed.endSeconds)
        }
      }

      guard self.generation == generation else { return }
      duration = max(duration, lastTimelineEnd)
      generationReachedEnd = !playlist.isLive
      if generationReachedEnd {
        if queuedSegments.isEmpty {
          shouldPlay = false
          position = duration
          state = .ended
          productionTask = nil
          releaseFinishedSession()
          return
        }
        resumeIfBuffered(endOfFile: true)
        productionTask = nil
      }
    } catch is CancellationError {
      await outputPipeline?.cancel()
      return
    } catch {
      await outputPipeline?.cancel()
      guard self.generation == generation else { return }
      fail(
        hlsPlaybackFailureMessage(
          stage: "Safari互換HLSのAVFoundation取込",
          error: error
        )
      )
    }
  }

  private func produceHLSStreamSegments(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    session: URL,
    startSeconds: Double,
    generation: Int
  ) async {
    guard let source = configuration.resolvedMediaSource,
      source.kind == .hls, var playlist = source.hlsPlaylist,
      !playlist.segments.isEmpty
    else {
      fail("HLSメディアプレイリストがありません")
      return
    }

    if playlist.isLive {
      await produceLiveHLSStreamSegments(
        prepared: prepared,
        configuration: configuration,
        source: source,
        playlist: playlist,
        session: session,
        startSeconds: startSeconds,
        generation: generation
      )
      return
    }

    let downloader = IPadHLSResourceDownloader(
      maximumResourceBytes: 64 * 1_024 * 1_024,
      resolutionPolicy: source.resolutionPolicy,
      requestContext: source.requestContext,
      resourceLoader: configuration.hlsResourceLoader
    )
    var timelineStarts = Dictionary(
      playlist.segments.map { ($0.sequence, $0.startSeconds) },
      uniquingKeysWith: { first, _ in first }
    )
    let startingMediaSegment =
      playlist.segments.first {
        $0.startSeconds + $0.duration > startSeconds
      } ?? playlist.segments[playlist.segments.index(before: playlist.segments.endIndex)]
    var nextMediaSequence = startingMediaSegment.sequence
    var outputSequence = 0
    var lastRefresh = Date.distantPast
    var restorationWindow: [HLSRestorationSource] = []
    var hasRestoredAnyWindow = false
    defer {
      for source in restorationWindow {
        try? FileManager.default.removeItem(at: source.localURL)
      }
    }

    do {
      while true {
        try Task.checkCancellation()
        guard self.generation == generation else {
          throw CancellationError()
        }

        let refreshInterval = max(
          0.5,
          min(2, (playlist.targetDuration ?? 2) / 2)
        )
        let availableBeforeRefresh = playlist.segments.contains {
          $0.sequence >= nextMediaSequence
        }
        if playlist.isLive,
          !availableBeforeRefresh
            || Date().timeIntervalSince(lastRefresh) >= refreshInterval
        {
          if !availableBeforeRefresh {
            try await Task.sleep(nanoseconds: 500_000_000)
          }
          let refreshedSource = try await IPadMediaURLResolver(
            resourceLoader: configuration.hlsResourceLoader,
            allowsAES128HLS: configuration.hlsStreamingMode.allowsAES128HLS
          ).resolve(
            source.mediaURL.absoluteString,
            policy: source.resolutionPolicy,
            context: source.requestContext
          )
          guard refreshedSource.kind == .hls,
            let refreshed = refreshedSource.hlsPlaylist
          else {
            throw MiohIPadWorkerEngineError.unsupportedMedia(
              "HLSプレイリストを更新できません"
            )
          }
          try mergeTimelineStarts(
            from: playlist,
            refreshed: refreshed,
            into: &timelineStarts
          )
          playlist = refreshed
          lastRefresh = Date()
          if let last = refreshed.segments.last,
            let lastStart = timelineStarts[last.sequence]
          {
            duration = max(duration, lastStart + last.duration)
          }
          if let firstSequence = refreshed.segments.first?.sequence,
            nextMediaSequence < firstSequence
          {
            nextMediaSequence = firstSequence
          }
        }

        guard
          let mediaSegment = playlist.segments
            .filter({ $0.sequence >= nextMediaSequence })
            .min(by: { $0.sequence < $1.sequence })
        else {
          if playlist.isLive { continue }
          break
        }
        try Task.checkCancellation()
        guard self.generation == generation else {
          throw CancellationError()
        }
        let timelineStart =
          timelineStarts[mediaSegment.sequence]
          ?? mediaSegment.startSeconds
        let timelineEnd = timelineStart + mediaSegment.duration
        guard mediaSegment.sequence < Int64.max else {
          throw MiohIPadWorkerEngineError.unsupportedMedia(
            "HLSのsegment sequenceが上限を超えています"
          )
        }
        nextMediaSequence = mediaSegment.sequence + 1
        if timelineEnd <= startSeconds { continue }

        let localURL = try await materializeHLSVODSegment(
          segment: mediaSegment,
          downloader: downloader,
          session: session,
          generation: generation
        )
        let restorationSource = HLSRestorationSource(
          mediaSegment: mediaSegment,
          timelineStart: timelineStart,
          localURL: localURL
        )
        if let previous = restorationWindow.last,
          !canShareHLSRestorationWindow(previous, restorationSource)
        {
          outputSequence = try await flushHLSRestorationWindow(
            restorationWindow,
            hasLeftContext: hasRestoredAnyWindow,
            prepared: prepared,
            configuration: configuration,
            requestedStartSeconds: startSeconds,
            outputSequence: outputSequence,
            session: session,
            generation: generation
          )
          for source in restorationWindow {
            try? FileManager.default.removeItem(at: source.localURL)
          }
          restorationWindow.removeAll(keepingCapacity: true)
        }
        restorationWindow.append(restorationSource)
        while true {
          let coreStartIndex = hasRestoredAnyWindow ? 1 : 0
          guard
            let coreSegmentCount = hlsCoreSegmentCountIfReady(
              restorationWindow,
              coreStartIndex: coreStartIndex,
              hasLeftContext: hasRestoredAnyWindow
            )
          else { break }
          let coreEndIndex = coreStartIndex + coreSegmentCount - 1
          outputSequence = try await restoreHLSRestorationWindow(
            restorationWindow,
            coreStartIndex: coreStartIndex,
            coreEndIndex: coreEndIndex,
            prepared: prepared,
            configuration: configuration,
            requestedStartSeconds: startSeconds,
            outputSequence: outputSequence,
            session: session,
            generation: generation
          )
          let retirementCount =
            hasRestoredAnyWindow
            ? coreSegmentCount
            : max(0, coreSegmentCount - 1)
          let retired = restorationWindow.prefix(retirementCount)
          for source in retired {
            try? FileManager.default.removeItem(at: source.localURL)
          }
          restorationWindow.removeFirst(retirementCount)
          hasRestoredAnyWindow = true
        }
      }

      outputSequence = try await flushHLSRestorationWindow(
        restorationWindow,
        hasLeftContext: hasRestoredAnyWindow,
        prepared: prepared,
        configuration: configuration,
        requestedStartSeconds: startSeconds,
        outputSequence: outputSequence,
        session: session,
        generation: generation
      )
      for source in restorationWindow {
        try? FileManager.default.removeItem(at: source.localURL)
      }
      restorationWindow.removeAll(keepingCapacity: false)

      guard self.generation == generation else { return }
      generationReachedEnd = !playlist.isLive
      if generationReachedEnd {
        if queuedSegments.isEmpty {
          shouldPlay = false
          position = duration
          state = .ended
          productionTask = nil
          releaseFinishedSession()
          return
        }
        resumeIfBuffered(endOfFile: true)
        productionTask = nil
      }
    } catch is CancellationError {
      return
    } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
      guard self.generation == generation else { return }
      requireBrowserInteraction(
        challengedURL,
        source: source,
        generation: generation
      )
    } catch {
      guard self.generation == generation else { return }
      fail(hlsPlaybackFailureMessage(stage: "HLS区間の復元準備", error: error))
    }
  }

  private func produceLiveHLSStreamSegments(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    source: IPadResolvedMediaSource,
    playlist: IPadHLSMediaPlaylist,
    session: URL,
    startSeconds: Double,
    generation: Int
  ) async {
    let nominalSegmentDuration = max(0.5, playlist.targetDuration ?? 2)
    let desiredPrefetchCount =
      Int(ceil(configuration.bufferLimitSeconds / nominalSegmentDuration))
      + startupSegmentCount
    let maximumPrefetchCount = min(
      6,
      max(startupSegmentCount, desiredPrefetchCount)
    )
    let prefetch = IPadLiveHLSPrefetchBuffer(
      mediaURL: source.mediaURL,
      resolutionPolicy: source.resolutionPolicy,
      requestContext: source.requestContext,
      playlist: playlist,
      directory: session,
      requestedStartSeconds: startSeconds,
      startupSegmentCount: startupSegmentCount,
      maximumBufferedSegments: maximumPrefetchCount,
      maximumBufferedBytes: 192 * 1_024 * 1_024,
      resourceLoader: configuration.hlsResourceLoader
    )
    await prefetch.start()

    var currentStartSeconds = startSeconds
    var outputSequence = 0
    var restorationWindow: [HLSRestorationSource] = []
    defer {
      for source in restorationWindow {
        try? FileManager.default.removeItem(at: source.localURL)
      }
    }
    do {
      while true {
        try Task.checkCancellation()
        guard self.generation == generation else {
          throw CancellationError()
        }
        switch await prefetch.nextDelivery() {
        case .waiting:
          try await Task.sleep(nanoseconds: 100_000_000)
        case .rebase(let rebase):
          guard self.generation == generation else {
            throw CancellationError()
          }
          for source in restorationWindow {
            try? FileManager.default.removeItem(at: source.localURL)
          }
          restorationWindow.removeAll(keepingCapacity: true)
          currentStartSeconds = rebase.targetTimelineStart
          rebaseLivePlayback(
            to: rebase,
            generation: generation
          )
        case .segment(let bufferedSegment):
          let timelineEnd =
            bufferedSegment.timelineStart
            + bufferedSegment.mediaSegment.duration
          if timelineEnd <= currentStartSeconds {
            try? FileManager.default.removeItem(
              at: bufferedSegment.localURL
            )
            continue
          }
          let restorationSource = HLSRestorationSource(
            mediaSegment: bufferedSegment.mediaSegment,
            timelineStart: bufferedSegment.timelineStart,
            localURL: bufferedSegment.localURL
          )
          if let previous = restorationWindow.last,
            !canShareHLSRestorationWindow(previous, restorationSource)
          {
            outputSequence = try await flushHLSRestorationWindow(
              restorationWindow,
              prepared: prepared,
              configuration: configuration,
              requestedStartSeconds: currentStartSeconds,
              outputSequence: outputSequence,
              session: session,
              generation: generation
            )
            for source in restorationWindow {
              try? FileManager.default.removeItem(at: source.localURL)
            }
            restorationWindow.removeAll(keepingCapacity: true)
          }
          restorationWindow.append(restorationSource)
          if restorationWindow.count == 2 {
            outputSequence = try await restoreHLSRestorationWindow(
              restorationWindow,
              coreIndex: 0,
              prepared: prepared,
              configuration: configuration,
              requestedStartSeconds: currentStartSeconds,
              outputSequence: outputSequence,
              session: session,
              generation: generation
            )
          } else if restorationWindow.count == 3 {
            outputSequence = try await restoreHLSRestorationWindow(
              restorationWindow,
              coreIndex: 1,
              prepared: prepared,
              configuration: configuration,
              requestedStartSeconds: currentStartSeconds,
              outputSequence: outputSequence,
              session: session,
              generation: generation
            )
            let expired = restorationWindow.removeFirst()
            try? FileManager.default.removeItem(at: expired.localURL)
          }
        case .ended(let timelineEnd):
          outputSequence = try await flushHLSRestorationWindow(
            restorationWindow,
            prepared: prepared,
            configuration: configuration,
            requestedStartSeconds: currentStartSeconds,
            outputSequence: outputSequence,
            session: session,
            generation: generation
          )
          for source in restorationWindow {
            try? FileManager.default.removeItem(at: source.localURL)
          }
          restorationWindow.removeAll(keepingCapacity: false)
          await prefetch.stop()
          guard self.generation == generation else { return }
          duration = max(duration, timelineEnd)
          generationReachedEnd = true
          if queuedSegments.isEmpty {
            shouldPlay = false
            position = duration
            state = .ended
            productionTask = nil
            releaseFinishedSession()
            return
          }
          resumeIfBuffered(endOfFile: true)
          productionTask = nil
          return
        case .interactionRequired(let challengedURL):
          await prefetch.stop()
          guard self.generation == generation else { return }
          requireBrowserInteraction(
            challengedURL,
            source: source,
            generation: generation
          )
          return
        case .failed(let message):
          throw MiohIPadWorkerEngineError.output(message)
        }
      }
    } catch is CancellationError {
      await prefetch.stop()
      return
    } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
      await prefetch.stop()
      guard self.generation == generation else { return }
      requireBrowserInteraction(
        challengedURL,
        source: source,
        generation: generation
      )
    } catch {
      await prefetch.stop()
      guard self.generation == generation else { return }
      fail(hlsPlaybackFailureMessage(stage: "ライブHLS区間の復元準備", error: error))
    }
  }

  private func rebaseLivePlayback(
    to rebase: IPadLiveHLSPrefetchBuffer.Rebase,
    generation: Int
  ) {
    guard self.generation == generation,
      configuration?.resolvedMediaSource?.hlsPlaylist?.isLive == true
    else { return }

    sourcePlayer.pause()
    restoredPlayer.pause()
    clearRestoredQueue(removingFiles: true)
    requestedStartSeconds = rebase.targetTimelineStart
    liveSourceWindowStartSeconds = rebase.windowTimelineStart
    position = rebase.targetTimelineStart
    processingPosition = rebase.targetTimelineStart
    duration = max(duration, rebase.latestTimelineEnd)
    generationHasStarted = false
    generationReachedEnd = false
    sourceSeekCompleted = false
    state = shouldPlay ? .followingLiveEdge : .paused
    guard sourceReady, let item = sourcePlayer.currentItem else { return }
    prepareSourceSeek(item: item, generation: generation)
  }

  private func canShareHLSRestorationWindow(
    _ previous: HLSRestorationSource,
    _ next: HLSRestorationSource
  ) -> Bool {
    guard previous.mediaSegment.sequence < Int64.max,
      next.mediaSegment.sequence == previous.mediaSegment.sequence + 1,
      next.mediaSegment.discontinuitySequence
        == previous.mediaSegment.discontinuitySequence,
      next.mediaSegment.initializationResource
        == previous.mediaSegment.initializationResource
    else { return false }
    return normalizedHLSContainerExtension(previous.localURL)
      == normalizedHLSContainerExtension(next.localURL)
  }

  private func normalizedHLSContainerExtension(_ url: URL) -> String {
    let value = url.pathExtension.lowercased()
    return value == "m4s" ? "mp4" : value
  }

  private func materializeHLSVODSegment(
    segment: IPadHLSMediaSegment,
    downloader: IPadHLSResourceDownloader,
    session: URL,
    generation: Int
  ) async throws -> URL {
    var consecutiveRateLimits = 0
    while true {
      try Task.checkCancellation()
      guard self.generation == generation else { throw CancellationError() }
      do {
        return try await downloader.materialize(segment: segment, in: session)
      } catch is CancellationError {
        throw CancellationError()
      } catch {
        let isRateLimited: Bool
        if let resolverError = error as? IPadMediaURLResolverError,
          case .invalidHTTPStatus(429) = resolverError
        {
          isRateLimited = true
        } else if let loadingError = error as? IPadHLSResourceLoadingError,
          loadingError == .attemptedUnavailable
        {
          isRateLimited = true
        } else {
          isRateLimited = false
        }
        guard isRateLimited else { throw error }
        consecutiveRateLimits = min(6, consecutiveRateLimits + 1)
        let delay = min(
          30,
          1.5 * pow(2, Double(consecutiveRateLimits - 1))
        )
        try await Task.sleep(
          nanoseconds: UInt64((delay * 1_000_000_000).rounded())
        )
      }
    }
  }

  private func flushHLSRestorationWindow(
    _ sources: [HLSRestorationSource],
    hasLeftContext: Bool = false,
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    requestedStartSeconds: Double,
    outputSequence: Int,
    session: URL,
    generation: Int
  ) async throws -> Int {
    guard !sources.isEmpty else { return outputSequence }
    let coreStartIndex = hasLeftContext && sources.count > 1 ? 1 : 0
    guard sources.indices.contains(coreStartIndex) else { return outputSequence }
    return try await restoreHLSRestorationWindow(
      sources,
      coreStartIndex: coreStartIndex,
      coreEndIndex: sources.index(before: sources.endIndex),
      prepared: prepared,
      configuration: configuration,
      requestedStartSeconds: requestedStartSeconds,
      outputSequence: outputSequence,
      session: session,
      generation: generation
    )
  }

  /// Restores one core segment as soon as its right temporal context arrives.
  private func hlsCoreSegmentCountIfReady(
    _ sources: [HLSRestorationSource],
    coreStartIndex: Int,
    hasLeftContext: Bool
  ) -> Int? {
    guard coreStartIndex >= 0, coreStartIndex < sources.count else { return nil }
    let availableCoreCount = sources.count - coreStartIndex - 1
    let requiredCount =
      hasLeftContext
      ? hlsSteadyRestoreBatchCoreSegments
      : hlsInitialRestoreBatchCoreSegments
    guard availableCoreCount >= requiredCount else { return nil }
    return requiredCount
  }

  private func restoreHLSRestorationWindow(
    _ sources: [HLSRestorationSource],
    coreIndex: Int,
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    requestedStartSeconds: Double,
    outputSequence: Int,
    session: URL,
    generation: Int
  ) async throws -> Int {
    try await restoreHLSRestorationWindow(
      sources,
      coreStartIndex: coreIndex,
      coreEndIndex: coreIndex,
      prepared: prepared,
      configuration: configuration,
      requestedStartSeconds: requestedStartSeconds,
      outputSequence: outputSequence,
      session: session,
      generation: generation
    )
  }

  private func restoreHLSRestorationWindow(
    _ sources: [HLSRestorationSource],
    coreStartIndex: Int,
    coreEndIndex: Int,
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    requestedStartSeconds: Double,
    outputSequence: Int,
    session: URL,
    generation: Int
  ) async throws -> Int {
    guard sources.indices.contains(coreStartIndex),
      sources.indices.contains(coreEndIndex),
      coreStartIndex <= coreEndIndex,
      !sources.isEmpty
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLS連結区間の復元対象が不正です"
      )
    }
    let coreSource = sources[coreStartIndex]
    let coreEndSource = sources[coreEndIndex]
    let requestedTimelineStart = max(
      coreSource.timelineStart,
      requestedStartSeconds
    )
    guard requestedTimelineStart < coreEndSource.timelineEnd else {
      return outputSequence
    }
    let restorationInputURL = session.appendingPathComponent(
      "mioh-hls-interval-\(outputSequence)-\(UUID().uuidString.lowercased()).mp4",
      isDirectory: false
    )
    defer {
      try? FileManager.default.removeItem(at: restorationInputURL)
    }
    var assembled: IPadHLSIntervalAssembler.Result
    var assembledCoreStartIndex: Int
    var assembledCoreEndIndex: Int
    do {
      assembled = try await IPadHLSIntervalAssembler.concatenate(
        inputURLs: sources.map(\.localURL),
        outputURL: restorationInputURL,
        temporaryDirectory: session
      )
      assembledCoreStartIndex = coreStartIndex
      assembledCoreEndIndex = coreEndIndex
      guard assembled.sourceOffsets.indices.contains(assembledCoreStartIndex),
        assembled.sourceOffsets.indices.contains(assembledCoreEndIndex),
        assembled.sourceDurations.indices.contains(assembledCoreEndIndex)
      else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "HLS連結区間の時間対応を取得できません"
        )
      }
      try await IPadHLSIntervalAssembler.validateDecodableVideo(
        at: restorationInputURL,
        near: assembled.sourceOffsets[assembledCoreStartIndex]
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      if coreStartIndex < coreEndIndex {
        var nextOutputSequence = outputSequence
        for index in coreStartIndex...coreEndIndex {
          let lowerBound = max(sources.startIndex, index - 1)
          let upperBound = min(
            sources.index(before: sources.endIndex),
            index + 1
          )
          nextOutputSequence = try await restoreHLSRestorationWindow(
            Array(sources[lowerBound...upperBound]),
            coreIndex: index - lowerBound,
            prepared: prepared,
            configuration: configuration,
            requestedStartSeconds: requestedStartSeconds,
            outputSequence: nextOutputSequence,
            session: session,
            generation: generation
          )
        }
        return nextOutputSequence
      }
      guard sources.count > 1 else { throw error }
      // A provider can change SPS/PPS without an HLS discontinuity tag. Keep
      // realtime playback alive by retrying only the core source; the next
      // compatible window will automatically regain temporal context.
      try? FileManager.default.removeItem(at: restorationInputURL)
      assembled = try await IPadHLSIntervalAssembler.concatenate(
        inputURLs: [coreSource.localURL],
        outputURL: restorationInputURL,
        temporaryDirectory: session
      )
      assembledCoreStartIndex = 0
      assembledCoreEndIndex = 0
      guard let coreOffset = assembled.sourceOffsets.first else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "HLS単一区間の時間対応を取得できません"
        )
      }
      try await IPadHLSIntervalAssembler.validateDecodableVideo(
        at: restorationInputURL,
        near: coreOffset
      )
    }
    guard assembled.sourceOffsets.indices.contains(assembledCoreStartIndex),
      assembled.sourceOffsets.indices.contains(assembledCoreEndIndex),
      assembled.sourceDurations.indices.contains(assembledCoreEndIndex)
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLS連結区間の時間対応を取得できません"
      )
    }
    let coreMediaStartSeconds = assembled.sourceOffsets[assembledCoreStartIndex]
    let coreMediaEndSeconds =
      assembled.sourceOffsets[assembledCoreEndIndex]
      + assembled.sourceDurations[assembledCoreEndIndex]
    return try await restoreHLSContinuousInterval(
      prepared: prepared,
      configuration: configuration,
      inputURL: restorationInputURL,
      coreMediaStartSeconds: coreMediaStartSeconds,
      coreMediaEndSeconds: coreMediaEndSeconds,
      coreTimelineStartSeconds: coreSource.timelineStart,
      coreTimelineEndSeconds: coreEndSource.timelineEnd,
      requestedTimelineStartSeconds: requestedTimelineStart,
      outputSequence: outputSequence,
      session: session,
      generation: generation
    )
  }

  private func restoreHLSContinuousInterval(
    prepared: MiohIPadPreparedWorker,
    configuration: IPadRealtimePreviewConfiguration,
    inputURL: URL,
    coreMediaStartSeconds: Double,
    coreMediaEndSeconds: Double,
    coreTimelineStartSeconds: Double,
    coreTimelineEndSeconds: Double,
    requestedTimelineStartSeconds: Double,
    outputSequence: Int,
    session: URL,
    generation: Int
  ) async throws -> Int {
    let inputExtension = inputURL.pathExtension.lowercased()
    let asset = AVURLAsset(url: inputURL)
    let videoTrack: AVAssetTrack
    do {
      guard let first = try await asset.loadTracks(withMediaType: .video).first else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "HLS区間に動画トラックがありません"
        )
      }
      videoTrack = first
    } catch is CancellationError {
      throw CancellationError()
    } catch let error as MiohIPadWorkerEngineError {
      throw error
    } catch {
      let value = error as NSError
      throw MiohIPadWorkerEngineError.decoder(
        "HLS連結動画の映像情報: \(error.localizedDescription) [\(value.domain):\(value.code)]"
      )
    }
    let sourceFrameRate = await MiohIPadSourceFrameRate.resolve(track: videoTrack)
    let timeRange: CMTimeRange
    do {
      timeRange = try await videoTrack.load(.timeRange)
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let value = error as NSError
      throw MiohIPadWorkerEngineError.decoder(
        "HLS連結動画の時間情報: \(error.localizedDescription) [\(value.domain):\(value.code)]"
      )
    }
    guard timeRange.start.isNumeric, timeRange.duration.isNumeric else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLS区間の時間情報が不正です"
      )
    }
    let maximumTimestampSeconds = 9_000_000_000.0
    let trackStartSeconds = timeRange.start.seconds
    let trackDurationSeconds = timeRange.duration.seconds
    guard timeRange.start.isNumeric, timeRange.duration.isNumeric,
      trackStartSeconds.isFinite, trackDurationSeconds.isFinite,
      trackStartSeconds >= 0, trackDurationSeconds > 0,
      trackStartSeconds <= maximumTimestampSeconds,
      trackDurationSeconds <= maximumTimestampSeconds - trackStartSeconds
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLS区間の時間情報が不正です"
      )
    }
    let mediaStartNanoseconds = max(
      0,
      Int64((trackStartSeconds * 1_000_000_000).rounded())
    )
    let mediaEndNanoseconds = max(
      mediaStartNanoseconds + 1,
      mediaStartNanoseconds
        + Int64((trackDurationSeconds * 1_000_000_000).rounded())
    )
    guard coreMediaStartSeconds.isFinite, coreMediaEndSeconds.isFinite,
      coreMediaStartSeconds >= 0,
      coreMediaEndSeconds > coreMediaStartSeconds,
      coreMediaEndSeconds <= maximumTimestampSeconds
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLS連結区間の復元範囲が不正です"
      )
    }
    let coreStartNanoseconds = min(
      mediaEndNanoseconds - 1,
      max(
        mediaStartNanoseconds,
        Int64((coreMediaStartSeconds * 1_000_000_000).rounded())
      )
    )
    let coreEndNanoseconds = min(
      mediaEndNanoseconds,
      max(
        mediaStartNanoseconds + 1,
        Int64((coreMediaEndSeconds * 1_000_000_000).rounded())
      )
    )
    guard coreEndNanoseconds > coreStartNanoseconds else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLS連結区間に復元可能な映像がありません"
      )
    }
    let requestedOffset = max(
      0,
      requestedTimelineStartSeconds - coreTimelineStartSeconds
    )
    let requestedCursorNanoseconds =
      coreStartNanoseconds
      + Int64((requestedOffset * 1_000_000_000).rounded())
    guard requestedCursorNanoseconds < coreEndNanoseconds else {
      return outputSequence
    }
    var cursorNanoseconds = max(
      coreStartNanoseconds,
      requestedCursorNanoseconds
    )
    var sequence = outputSequence
    let byteCount = Int64(
      (try? inputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize)
        ?? 0
    )
    while cursorNanoseconds < coreEndNanoseconds {
      try Task.checkCancellation()
      while bufferedSeconds >= configuration.bufferLimitSeconds {
        try Task.checkCancellation()
        try await Task.sleep(nanoseconds: 100_000_000)
      }
      guard self.generation == generation else {
        throw CancellationError()
      }
      let endNanoseconds = min(
        coreEndNanoseconds,
        cursorNanoseconds
          + Int64((segmentSeconds * 1_000_000_000).rounded())
      )
      let request = try configuration.request(
        coreStartNanoseconds: cursorNanoseconds,
        coreEndNanoseconds: endNanoseconds,
        mediaStartNanoseconds: mediaStartNanoseconds,
        mediaEndNanoseconds: mediaEndNanoseconds,
        inputByteCount: byteCount,
        inputExtension: inputExtension,
        sourceFPSNumerator: sourceFrameRate.numerator,
        sourceFPSDenominator: sourceFrameRate.denominator
      )
      let outputURL = session.appendingPathComponent(
        String(format: "segment-%06d.mp4", sequence)
      )
      try? FileManager.default.removeItem(at: outputURL)
      let relativeStart =
        Double(cursorNanoseconds - coreStartNanoseconds) / 1_000_000_000
      let relativeEnd =
        Double(endNanoseconds - coreStartNanoseconds) / 1_000_000_000
      let segmentStart = coreTimelineStartSeconds + relativeStart
      let segmentEnd = min(
        coreTimelineEndSeconds,
        coreTimelineStartSeconds + relativeEnd
      )
      do {
        _ = try await prepared.executeLocal(
          request: request,
          inputURL: inputURL,
          outputURL: outputURL,
          progress: { [weak self] value in
            Task { @MainActor in
              guard let self, self.generation == generation else { return }
              self.processingPosition = min(
                segmentEnd,
                segmentStart
                  + (segmentEnd - segmentStart) * max(0, min(1, value))
              )
            }
          }
        )
      } catch MiohIPadWorkerEngineError.unsupportedMedia(let detail)
        where endNanoseconds == coreEndNanoseconds
        && detail.contains("core range contains no decoded frame")
      {
        try? FileManager.default.removeItem(at: outputURL)
        break
      }
      try Task.checkCancellation()
      guard self.generation == generation else {
        throw CancellationError()
      }
      try enqueue(
        Segment(
          sequence: sequence,
          startSeconds: segmentStart,
          endSeconds: segmentEnd,
          url: outputURL
        )
      )
      cursorNanoseconds = endNanoseconds
      sequence += 1
      processingPosition = segmentEnd
      duration = max(duration, segmentEnd)
      resumeIfBuffered()
    }
    return sequence
  }

  private func mergeTimelineStarts(
    from previous: IPadHLSMediaPlaylist,
    refreshed: IPadHLSMediaPlaylist,
    into starts: inout [Int64: Double]
  ) throws {
    if let overlap = refreshed.segments.first(where: {
      starts[$0.sequence] != nil
    }), let knownStart = starts[overlap.sequence] {
      let adjustment = knownStart - overlap.startSeconds
      for segment in refreshed.segments {
        starts[segment.sequence] = segment.startSeconds + adjustment
      }
      return
    }

    let previousEnd =
      previous.segments.compactMap { segment -> Double? in
        guard let start = starts[segment.sequence] else { return nil }
        return start + segment.duration
      }.max() ?? duration
    let previousLastSequence = previous.segments.map(\.sequence).max()
    let refreshedFirstSequence = refreshed.segments.map(\.sequence).min()
    let missingCount: Double
    if let previousLastSequence, let refreshedFirstSequence,
      refreshedFirstSequence > previousLastSequence
    {
      missingCount = max(
        0,
        Double(refreshedFirstSequence) - Double(previousLastSequence) - 1
      )
    } else {
      missingCount = 0
    }
    let knownDurations =
      previous.segments.map(\.duration)
      + refreshed.segments.map(\.duration)
    let estimatedDuration =
      previous.targetDuration ?? refreshed.targetDuration
      ?? (knownDurations.reduce(0, +) / Double(max(1, knownDurations.count)))
    let skippedDuration = missingCount * estimatedDuration
    guard previousEnd.isFinite, estimatedDuration.isFinite,
      skippedDuration.isFinite
    else {
      throw MiohIPadWorkerEngineError.unsupportedMedia(
        "HLSライブ配信の時間軸が上限を超えています"
      )
    }
    var cursor = previousEnd + skippedDuration
    for segment in refreshed.segments.sorted(by: {
      $0.sequence < $1.sequence
    }) {
      guard cursor.isFinite else {
        throw MiohIPadWorkerEngineError.unsupportedMedia(
          "HLSライブ配信の時間軸が上限を超えています"
        )
      }
      starts[segment.sequence] = cursor
      cursor += segment.duration
    }
  }

  private func enqueue(_ segment: Segment) throws {
    queuedSegments.append(segment)
    try fillResidentRestoredQueue()
    updateBufferedDuration()
    // VOD HLS source items are attached only after restoration has produced
    // its first playable segment. This prevents a paused audible AVPlayer from
    // racing the look-ahead capture for MissAV-style signed startup URLs.
    installDeferredSourcePlayerItemIfNeeded()
  }

  private func fillResidentRestoredQueue() throws {
    let residentSequences = Set(itemSegments.values.map(\.sequence))
    var residentCount = residentSequences.count
    guard residentCount < maximumResidentRestoredSegments else { return }
    for segment in queuedSegments where !residentSequences.contains(segment.sequence) {
      try insertResidentRestoredSegment(segment)
      residentCount += 1
      if residentCount >= maximumResidentRestoredSegments { break }
    }
  }

  private func insertResidentRestoredSegment(_ segment: Segment) throws {
    let item = AVPlayerItem(url: segment.url)
    let videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ])
    videoOutput.suppressesPlayerRendering = true
    item.add(videoOutput)
    guard restoredPlayer.canInsert(item, after: nil) else {
      throw MiohIPadWorkerEngineError.output(
        "復元済みセグメントを再生キューへ追加できません"
      )
    }
    let identifier = ObjectIdentifier(item)
    itemSegments[identifier] = segment
    itemVideoOutputs[identifier] = videoOutput
    restoredPlayer.insert(item, after: nil)
    let token = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemDidPlayToEndTime,
      object: item,
      queue: .main
    ) { [weak self, weak item] _ in
      guard let item else { return }
      Task { @MainActor in self?.finished(item: item) }
    }
    notificationTokens[ObjectIdentifier(item)] = token
  }

  func restoredVideoOutput(
    for item: AVPlayerItem
  ) -> AVPlayerItemVideoOutput? {
    itemVideoOutputs[ObjectIdentifier(item)]
  }

  func didDisplayRestoredFrame(
    from item: AVPlayerItem,
    pixelBuffer: CVPixelBuffer
  ) {
    guard itemSegments[ObjectIdentifier(item)] != nil else { return }
    latestRestoredPixelBuffer = pixelBuffer
    if !hasPresentedRestoredFrame {
      hasPresentedRestoredFrame = true
    }
  }

  private func finished(item: AVPlayerItem) {
    let identifier = ObjectIdentifier(item)
    guard let segment = itemSegments[identifier] else { return }
    releaseSegments(through: segment.sequence)
    do {
      try fillResidentRestoredQueue()
    } catch {
      fail(error.localizedDescription)
      return
    }
    reconcileEmptyQueue()
  }

  private func releaseSegments(through sequence: Int) {
    let consumed = queuedSegments.filter { $0.sequence <= sequence }
    guard !consumed.isEmpty else { return }
    let consumedSequences = Set(consumed.map(\.sequence))
    let consumedIdentifiers = itemSegments.compactMap {
      identifier, segment in
      consumedSequences.contains(segment.sequence) ? identifier : nil
    }
    for identifier in consumedIdentifiers {
      itemSegments.removeValue(forKey: identifier)
      itemVideoOutputs.removeValue(forKey: identifier)
      if let token = notificationTokens.removeValue(forKey: identifier) {
        NotificationCenter.default.removeObserver(token)
      }
    }
    queuedSegments.removeAll { consumedSequences.contains($0.sequence) }
    for segment in consumed {
      try? FileManager.default.removeItem(at: segment.url)
    }
    updateBufferedDuration()
  }

  private func retireSegmentsBeforeCurrentItem() {
    guard let item = restoredPlayer.currentItem,
      let active = itemSegments[ObjectIdentifier(item)]
    else { return }
    releaseSegments(through: active.sequence - 1)
    do {
      try fillResidentRestoredQueue()
    } catch {
      fail(error.localizedDescription)
    }
  }

  private func reconcileEmptyQueue() {
    if queuedSegments.isEmpty {
      sourcePlayer.pause()
      restoredPlayer.pause()
      if generationReachedEnd {
        shouldPlay = false
        position = duration
        state = .ended
        releaseFinishedSession()
      } else if shouldPlay {
        state = .buffering
      }
    }
  }

  private func resumeIfBuffered(endOfFile: Bool = false) {
    guard shouldPlay, sourceReady, sourceSeekCompleted,
      state != .playing, !queuedSegments.isEmpty
    else {
      return
    }
    let remaining = max(0, duration - position)
    let segmentCount =
      generationHasStarted
      ? rebufferSegmentCount : activeStartupSegmentCount
    let nominalRequired = Double(segmentCount) * segmentSeconds
    let required = min(
      remaining,
      min(nominalRequired, max(segmentSeconds, configuration?.bufferLimitSeconds ?? 8))
    )
    guard bufferedSeconds + 0.1 >= required || endOfFile else {
      if state != .paused { state = .buffering }
      return
    }
    generationHasStarted = true
    if requiresStreamingClockSynchronization {
      beginStreamingClockSynchronization()
      return
    }
    sourcePlayer.playImmediately(atRate: 1)
    if sourcePlayer.timeControlStatus == .playing {
      restoredPlayer.play()
      state = .playing
    } else {
      restoredPlayer.pause()
      state = .buffering
    }
  }

  private func installSourceObservers(
    item: AVPlayerItem,
    generation: Int
  ) {
    sourceStatusObservation = item.observe(\.status, options: [.initial, .new]) {
      [weak self, weak item] _, _ in
      Task { @MainActor in
        guard let self, let item, self.generation == generation else { return }
        switch item.status {
        case .readyToPlay:
          self.sourceReady = true
          self.prepareSourceSeek(item: item, generation: generation)
        case .failed:
          self.fail(
            self.hlsPlaybackFailureMessage(
              stage: "元動画のAVPlayer読込み",
              error: item.error
            ))
        case .unknown:
          break
        @unknown default:
          break
        }
      }
    }
    sourceTimeControlObservation = sourcePlayer.observe(
      \.timeControlStatus,
      options: [.new]
    ) { [weak self] player, _ in
      Task { @MainActor in
        guard let self, self.generation == generation, self.shouldPlay else {
          return
        }
        switch player.timeControlStatus {
        case .playing:
          guard self.generationHasStarted else { return }
          if self.requiresStreamingClockSynchronization {
            if self.streamingClockCorrectionInFlight { return }
            if self.streamingRestoredHeldForSourceCatchup {
              self.restoredPlayer.pause()
              self.state = .buffering
            } else if self.restoredPlayer.rate == 0 {
              self.beginStreamingClockSynchronization()
            } else {
              self.state = .playing
            }
            return
          }
          self.restoredPlayer.play()
          self.state = .playing
        case .waitingToPlayAtSpecifiedRate:
          guard self.generationHasStarted else { return }
          self.restoredPlayer.pause()
          self.state = .buffering
        case .paused:
          break
        @unknown default:
          break
        }
      }
    }
  }

  private func prepareSourceSeek(
    item: AVPlayerItem,
    generation: Int,
    attempt: Int = 0,
    revision: Int? = nil
  ) {
    guard self.generation == generation else { return }
    let activeRevision: Int
    if let revision {
      guard revision == sourceSeekRevision else { return }
      activeRevision = revision
    } else {
      sourceSeekRevision += 1
      activeRevision = sourceSeekRevision
    }
    let isLiveHLS = configuration?.resolvedMediaSource?.hlsPlaylist?.isLive == true
    let seekable = item.seekableTimeRanges.first?.timeRangeValue
    if isLiveHLS, seekable == nil, attempt < 50 {
      Task { [weak self, weak item] in
        try? await Task.sleep(nanoseconds: 100_000_000)
        guard let self, let item else { return }
        self.prepareSourceSeek(
          item: item,
          generation: generation,
          attempt: attempt + 1,
          revision: activeRevision
        )
      }
      return
    }

    let rangeStart = isLiveHLS ? (seekable?.start.seconds ?? 0) : 0
    let safeRangeStart = rangeStart.isFinite ? max(0, rangeStart) : 0
    let maximumRelative =
      seekable.map { max(0, $0.duration.seconds) }
      ?? duration
    let requestedRelative =
      isLiveHLS
      ? requestedStartSeconds - liveSourceWindowStartSeconds
      : requestedStartSeconds
    let relativeTarget = min(
      max(0, requestedRelative),
      max(0, maximumRelative - 0.001)
    )
    let absoluteTarget = safeRangeStart + relativeTarget
    sourceTimeOffset =
      isLiveHLS
      ? absoluteTarget - requestedStartSeconds
      : 0
    performSourceSeek(
      item: item,
      generation: generation,
      revision: activeRevision,
      absoluteTarget: absoluteTarget,
      relaxed: false
    )
  }

  private func performSourceSeek(
    item: AVPlayerItem,
    generation: Int,
    revision: Int,
    absoluteTarget: Double,
    relaxed: Bool
  ) {
    guard self.generation == generation,
      sourceSeekRevision == revision,
      sourcePlayer.currentItem === item
    else { return }
    sourceSeekTimeoutTask?.cancel()
    let attemptID = UUID()
    sourceSeekAttemptID = attemptID
    let isHLS = configuration?.resolvedMediaSource?.kind == .hls
    let isLiveHLS =
      configuration?.resolvedMediaSource?.hlsPlaylist?.isLive == true
    let isLoopbackRangeInput = configuration?.inputRangeValidator != nil
    let segmentTolerance = CMTime(
      seconds: segmentSeconds,
      preferredTimescale: 600
    )
    let rangeTolerance = CMTime(
      seconds: 0.250,
      preferredTimescale: 600
    )
    let initialTolerance: CMTime =
      isHLS
      ? (isLiveHLS ? segmentTolerance : .zero)
      : (isLoopbackRangeInput ? rangeTolerance : .zero)
    let toleranceBefore: CMTime =
      relaxed
      ? .positiveInfinity : initialTolerance
    let toleranceAfter: CMTime =
      relaxed
      ? .positiveInfinity : initialTolerance
    sourceSeekTimeoutTask = Task { [weak self, weak item] in
      try? await Task.sleep(nanoseconds: 12_000_000_000)
      guard !Task.isCancelled, let self, let item,
        self.generation == generation,
        self.sourceSeekRevision == revision,
        self.sourceSeekAttemptID == attemptID,
        self.sourcePlayer.currentItem === item
      else { return }
      self.sourcePlayer.currentItem?.cancelPendingSeeks()
      if relaxed {
        self.fail("元動画のシークがタイムアウトしました")
      } else {
        self.performSourceSeek(
          item: item,
          generation: generation,
          revision: revision,
          absoluteTarget: absoluteTarget,
          relaxed: true
        )
      }
    }
    sourcePlayer.seek(
      to: CMTime(seconds: absoluteTarget, preferredTimescale: 600),
      toleranceBefore: toleranceBefore,
      toleranceAfter: toleranceAfter
    ) { [weak self, weak item] finished in
      Task { @MainActor in
        guard let self, let item,
          self.generation == generation,
          self.sourceSeekRevision == revision,
          self.sourceSeekAttemptID == attemptID,
          self.sourcePlayer.currentItem === item
        else { return }
        self.sourceSeekTimeoutTask?.cancel()
        self.sourceSeekTimeoutTask = nil
        if !finished, !relaxed {
          self.sourcePlayer.currentItem?.cancelPendingSeeks()
          self.performSourceSeek(
            item: item,
            generation: generation,
            revision: revision,
            absoluteTarget: absoluteTarget,
            relaxed: true
          )
          return
        }
        guard finished else {
          self.fail("元動画のシークを完了できませんでした")
          return
        }
        let actual = self.sourcePlayer.currentTime().seconds
        if actual.isFinite {
          self.sourceSeekErrorSeconds = actual - absoluteTarget
          // Only a live HLS window needs a presentation-time origin offset.
          // VOD HLS is restored on its zero-based media timeline, so hiding an
          // imprecise AVPlayer seek behind an offset makes the clocks appear
          // aligned while the audible content remains on the wrong frame.
          self.sourceTimeOffset =
            isLiveHLS ? actual - self.requestedStartSeconds : 0
        }
        self.sourceSeekCompleted = true
        self.resumeIfBuffered()
      }
    }
  }

  private func beginStreamingMetricsPolling(
    configuration: IPadRealtimePreviewConfiguration,
    generation: Int
  ) {
    streamingMetricsTask?.cancel()
    guard let metricsProvider = configuration.streamingMetrics else { return }
    streamingMetricsTask = Task { [weak self] in
      while !Task.isCancelled {
        let metrics = await metricsProvider()
        guard !Task.isCancelled, let self, self.generation == generation else {
          return
        }
        self.sftpBitsPerSecond = max(0, metrics.bitsPerSecond)
        self.sftpActiveRangeReads = max(0, metrics.activeRangeReads)
        self.sftpRangeLatencySeconds = max(
          0,
          metrics.lastRangeLatencySeconds
        )
        self.sftpCachedBytes = max(0, metrics.cachedBytes)
        try? await Task.sleep(nanoseconds: 500_000_000)
      }
    }
  }

  private func recordRestorationPerformance(
    wallSeconds: Double,
    mediaSeconds: Double,
    processedFrames: Int? = nil,
    restoredFrames: Int? = nil,
    restorationSeconds: Double? = nil,
    restorationPreparationSeconds: Double? = nil,
    restorationCompositingSeconds: Double? = nil
  ) {
    guard wallSeconds.isFinite, mediaSeconds.isFinite, mediaSeconds > 0 else {
      return
    }
    restorationPerformanceSamples.append(
      RestorationPerformanceSample(
        wallSeconds: max(0.001, wallSeconds),
        mediaSeconds: mediaSeconds,
        processedFrames: processedFrames,
        restoredFrames: restoredFrames,
        restorationSeconds: restorationSeconds,
        restorationPreparationSeconds: restorationPreparationSeconds,
        restorationCompositingSeconds: restorationCompositingSeconds
      )
    )
    if restorationPerformanceSamples.count > 12 {
      restorationPerformanceSamples.removeFirst(
        restorationPerformanceSamples.count - 12
      )
    }
    let measuredWall = restorationPerformanceSamples.reduce(0) {
      $0 + $1.wallSeconds
    }
    let measuredMedia = restorationPerformanceSamples.reduce(0) {
      $0 + $1.mediaSeconds
    }
    restorationRealtimeFactor =
      measuredMedia > 0
      ? measuredWall / measuredMedia : 0
    let processingFrameSamples = restorationPerformanceSamples.compactMap {
      sample in
      sample.processedFrames.map { (frames: $0, wall: sample.wallSeconds) }
    }
    let processingFrames = processingFrameSamples.reduce(0) {
      $0 + $1.frames
    }
    let processingFrameWall = processingFrameSamples.reduce(0) {
      $0 + $1.wall
    }
    processingFramesPerSecond =
      processingFrameWall > 0
      ? Double(processingFrames) / processingFrameWall : 0
    let frameSamples: [(frames: Int, wall: Double)] =
      restorationPerformanceSamples.compactMap { sample in
        guard let frames = sample.restoredFrames,
          let seconds = sample.restorationSeconds,
          frames > 0, seconds > 0
        else { return nil }
        return (frames: frames, wall: seconds)
      }
    let measuredFrames = frameSamples.reduce(0) { $0 + $1.frames }
    let frameWall = frameSamples.reduce(0) { $0 + $1.wall }
    recentRestoredFrameCount = measuredFrames
    restorationFramesPerSecond =
      frameWall > 0
      ? Double(measuredFrames) / frameWall : 0
    persistRealtimeDiagnostics()
  }

  private func persistRealtimeDiagnostics(failureMessage: String? = nil) {
    if let failureMessage {
      lastRealtimeFailureMessage = failureMessage
    }
    let persistedFailure = failureMessage ?? lastRealtimeFailureMessage
    guard restorationRealtimeFactor > 0 || persistedFailure != nil else { return }
    let sourceKind: String
    if configuration?.inputRangeValidator != nil {
      sourceKind = "sftp-stream"
    } else if configuration?.resolvedMediaSource?.kind == .hls {
      sourceKind = "hls"
    } else {
      sourceKind = "local-file"
    }
    let recentProcessedFrames = restorationPerformanceSamples.reduce(0) {
      $0 + ($1.processedFrames ?? 0)
    }
    let recentModelSeconds = restorationPerformanceSamples.reduce(0.0) {
      $0 + ($1.restorationSeconds ?? 0)
    }
    let recentPreparationSeconds = restorationPerformanceSamples.reduce(0.0) {
      $0 + ($1.restorationPreparationSeconds ?? 0)
    }
    let recentCompositingSeconds = restorationPerformanceSamples.reduce(0.0) {
      $0 + ($1.restorationCompositingSeconds ?? 0)
    }
    let recentWallSeconds = restorationPerformanceSamples.reduce(0.0) {
      $0 + $1.wallSeconds
    }
    var payload: [String: Any] = [
      "schemaVersion": 1,
      "sampledAtUnixSeconds": Date().timeIntervalSince1970,
      "sourceKind": sourceKind,
      "rollingSegmentCount": restorationPerformanceSamples.count,
      "processingSpeed": 1 / restorationRealtimeFactor,
      "realtimeFactor": restorationRealtimeFactor,
      "processedFramesPerSecond": processingFramesPerSecond,
      "actualRestorationFramesPerSecond": restorationFramesPerSecond,
      "actualRestoredFrameCount": recentRestoredFrameCount,
      "recentProcessedFrameCount": recentProcessedFrames,
      "restorationWorkloadPerOutputFrame": recentProcessedFrames > 0
        ? Double(recentRestoredFrameCount) / Double(recentProcessedFrames) : 0,
      "recentWallSeconds": recentWallSeconds,
      "recentModelSeconds": recentModelSeconds,
      "recentPreparationSeconds": recentPreparationSeconds,
      "recentCompositingSeconds": recentCompositingSeconds,
      "parallelRestorationLanes": restorationParallelLanes,
      "bufferedSeconds": bufferedSeconds,
      "processingPositionSeconds": processingPosition,
      "playbackPositionSeconds": position,
      "sftpBitsPerSecond": sftpBitsPerSecond,
      "sftpActiveRangeReads": sftpActiveRangeReads,
      "sourceSeekErrorSeconds": sourceSeekErrorSeconds,
      "restoredClockDriftSeconds": latestClockDriftSeconds,
      "hlsHasSeparateAudio":
        configuration?.resolvedMediaSource?.hlsPlaylist?.masterMetadata?
          .hasSeparateAudio == true,
    ]
    if let persistedFailure {
      payload["lastFailure"] = persistedFailure
    }
    guard JSONSerialization.isValidJSONObject(payload),
      let data = try? JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys]
      ),
      let applicationSupport = try? FileManager.default.url(
        for: .applicationSupportDirectory,
        in: .userDomainMask,
        appropriateFor: nil,
        create: true
      )
    else { return }
    let directory = applicationSupport.appendingPathComponent(
      "MiohRemoteDiagnostics",
      isDirectory: true
    )
    do {
      try FileManager.default.createDirectory(
        at: directory,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
      )
      try data.write(
        to: directory.appendingPathComponent("realtime-latest.json"),
        options: .atomic
      )
    } catch {
      // Diagnostics must never interrupt restoration playback.
    }
  }

  private func usesUnlimitedLocalCache(
    _ configuration: IPadRealtimePreviewConfiguration
  ) -> Bool {
    configuration.inputURL.isFileURL
      && configuration.inputRangeValidator == nil
      && configuration.resolvedMediaSource?.kind != .hls
  }

  private static func hasLocalCacheStorageHeadroom(
    at directory: URL,
    concurrentSegments: Int = 1
  ) -> Bool {
    guard
      let values = try? directory.resourceValues(forKeys: [
        .volumeAvailableCapacityForImportantUsageKey,
        .volumeAvailableCapacityKey,
        .volumeTotalCapacityKey,
      ])
    else { return false }
    let available =
      values.volumeAvailableCapacityForImportantUsage
      ?? values.volumeAvailableCapacity.map(Int64.init) ?? 0
    let total = Int64(values.volumeTotalCapacity ?? 0)
    let reserve = max(Int64(1 * 1_024 * 1_024 * 1_024), total / 10)
    // Every parallel two-second output can still be in flight after this
    // check, so reserve a conservative 256 MiB margin per active runner.
    let inFlightMargin =
      Int64(max(1, concurrentSegments))
      * 256 * 1_024 * 1_024
    return available > reserve + inFlightMargin
  }

  private func installTimeObserver() {
    timeObserver = sourcePlayer.addPeriodicTimeObserver(
      forInterval: CMTime(
        seconds: clockObservationIntervalSeconds,
        preferredTimescale: 600
      ),
      queue: .main
    ) { [weak self] time in
      Task { @MainActor in self?.tick(sourceSeconds: time.seconds) }
    }
  }

  private func tick(sourceSeconds absoluteSourceSeconds: Double) {
    guard absoluteSourceSeconds.isFinite else { return }
    let sourceSeconds = max(0, absoluteSourceSeconds - sourceTimeOffset)
    if generationHasStarted {
      position = min(duration, sourceSeconds)
    }
    retireSegmentsBeforeCurrentItem()
    if generationReachedEnd, sourceSeconds + 0.05 >= duration,
      restoredPlayer.currentItem == nil, let last = queuedSegments.last
    {
      releaseSegments(through: last.sequence)
      reconcileEmptyQueue()
    }
    updateBufferedDuration()
    guard state == .playing || streamingRestoredHeldForSourceCatchup,
      let item = restoredPlayer.currentItem,
      let segment = itemSegments[ObjectIdentifier(item)]
    else { return }
    let itemIdentifier = ObjectIdentifier(item)
    let systemUptime = ProcessInfo.processInfo.systemUptime
    if itemIdentifier != currentRestoredItemIdentifier {
      currentRestoredItemIdentifier = itemIdentifier
      currentRestoredItemStartedAt = systemUptime
      return
    }
    let local = restoredPlayer.currentTime().seconds
    guard local.isFinite else { return }
    let restoredAbsolute = segment.startSeconds + local
    let correctionGrace =
      requiresStreamingClockSynchronization
      ? streamingDriftCorrectionGraceSeconds : driftCorrectionGraceSeconds
    guard systemUptime - currentRestoredItemStartedAt >= correctionGrace
    else { return }
    let drift = restoredAbsolute - sourceSeconds
    latestClockDriftSeconds = drift
    if requiresStreamingClockSynchronization {
      if streamingRestoredHeldForSourceCatchup {
        if drift <= streamingDriftResumeToleranceSeconds {
          streamingRestoredHeldForSourceCatchup = false
          beginStreamingClockSynchronization()
        } else {
          restoredPlayer.pause()
          state = .buffering
        }
        return
      }
      guard abs(drift) > streamingDriftToleranceSeconds else { return }
      if drift > 0 {
        // Restored video is ahead. Keep its last frame visible while the
        // audible source clock catches up; rewinding the restored queue causes
        // oscillation at independently encoded segment boundaries.
        streamingRestoredHeldForSourceCatchup = true
        restoredPlayer.pause()
        state = .buffering
      } else {
        // Restored video is behind. Freeze the authoritative audio clock while
        // moving the restored queue, then restart both players together.
        beginStreamingClockSynchronization()
      }
      return
    }
    let usesVODHLSClock =
      configuration?.resolvedMediaSource?.kind == .hls
      && configuration?.resolvedMediaSource?.hlsPlaylist?.isLive != true
    let activeDriftTolerance =
      usesVODHLSClock ? hlsDriftToleranceSeconds : driftToleranceSeconds
    if abs(drift) > activeDriftTolerance {
      let maximumLocalTime = max(
        0,
        segment.endSeconds - segment.startSeconds - 0.001
      )
      let target = min(
        max(0, sourceSeconds - segment.startSeconds),
        maximumLocalTime
      )
      let tolerance = CMTime(
        seconds: usesVODHLSClock
          ? hlsDriftSeekToleranceSeconds : driftSeekToleranceSeconds,
        preferredTimescale: 600
      )
      restoredPlayer.seek(
        to: CMTime(
          seconds: target,
          preferredTimescale: 600
        ),
        toleranceBefore: tolerance,
        toleranceAfter: tolerance
      )
    }
  }

  private var requiresStreamingClockSynchronization: Bool {
    configuration?.inputRangeValidator != nil
  }

  /// Aligns the local restored queue to the SFTP source player's authoritative
  /// audio clock. Both players stay paused during the queue seek and are
  /// restarted in adjacent main-actor calls, preventing the source from
  /// advancing while an asynchronous correction is still in flight.
  private func beginStreamingClockSynchronization() {
    guard requiresStreamingClockSynchronization,
      shouldPlay, generationHasStarted, sourceReady, sourceSeekCompleted,
      !streamingClockCorrectionInFlight
    else { return }
    let absoluteSourceSeconds = sourcePlayer.currentTime().seconds
    guard absoluteSourceSeconds.isFinite else { return }
    let sourceSeconds = max(0, absoluteSourceSeconds - sourceTimeOffset)
    let availableItems = restoredPlayer.items()
    guard
      let targetItem = availableItems.first(where: { item in
        guard let segment = itemSegments[ObjectIdentifier(item)] else {
          return false
        }
        return sourceSeconds + 0.001 >= segment.startSeconds
          && sourceSeconds < segment.endSeconds
      }), let targetSegment = itemSegments[ObjectIdentifier(targetItem)]
    else {
      sourcePlayer.pause()
      restoredPlayer.pause()
      state = .buffering
      return
    }

    streamingClockCorrectionInFlight = true
    streamingClockCorrectionRevision &+= 1
    let revision = streamingClockCorrectionRevision
    let expectedGeneration = generation
    sourcePlayer.pause()
    restoredPlayer.pause()
    state = .buffering

    if restoredPlayer.currentItem !== targetItem {
      while let current = restoredPlayer.currentItem, current !== targetItem {
        restoredPlayer.advanceToNextItem()
      }
      guard restoredPlayer.currentItem === targetItem else {
        streamingClockCorrectionInFlight = false
        return
      }
      releaseSegments(through: targetSegment.sequence - 1)
      currentRestoredItemIdentifier = ObjectIdentifier(targetItem)
      currentRestoredItemStartedAt = ProcessInfo.processInfo.systemUptime
    }

    let maximumLocalTime = max(
      0,
      targetSegment.endSeconds - targetSegment.startSeconds - 0.001
    )
    let localTarget = min(
      max(0, sourceSeconds - targetSegment.startSeconds),
      maximumLocalTime
    )
    let tolerance = CMTime(
      seconds: streamingDriftSeekToleranceSeconds,
      preferredTimescale: 600
    )
    restoredPlayer.currentItem?.cancelPendingSeeks()
    restoredPlayer.seek(
      to: CMTime(seconds: localTarget, preferredTimescale: 600),
      toleranceBefore: tolerance,
      toleranceAfter: tolerance
    ) { [weak self, weak targetItem] finished in
      Task { @MainActor in
        guard let self, let targetItem,
          self.generation == expectedGeneration,
          self.streamingClockCorrectionRevision == revision,
          self.restoredPlayer.currentItem === targetItem
        else { return }
        self.streamingClockCorrectionInFlight = false
        guard finished, self.shouldPlay else {
          if !self.shouldPlay { self.state = .paused }
          return
        }
        self.streamingRestoredHeldForSourceCatchup = false
        self.currentRestoredItemIdentifier = ObjectIdentifier(targetItem)
        self.currentRestoredItemStartedAt = ProcessInfo.processInfo.systemUptime
        // Keep these calls adjacent. Waiting for source timeControlStatus before
        // starting restored video lets audio gain a main-thread scheduling turn.
        self.sourcePlayer.playImmediately(atRate: 1)
        self.restoredPlayer.playImmediately(atRate: 1)
        self.state = .playing
      }
    }
  }

  private func updateBufferedDuration() {
    guard let last = queuedSegments.last else {
      bufferedSeconds = 0
      hlsAVFoundationCapture?.setRestoredBufferLead(0)
      return
    }
    bufferedSeconds = max(0, last.endSeconds - position)
    hlsAVFoundationCapture?.setRestoredBufferLead(bufferedSeconds)
  }

  private func applyVolume() {
    sourcePlayer.volume = muted ? 0 : Float(min(max(volume, 0), 1))
  }

  private func beginPreventingSleep(if enabled: Bool) {
    guard enabled, previousIdleTimerDisabled == nil else { return }
    previousIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
    UIApplication.shared.isIdleTimerDisabled = true
  }

  private func endPreventingSleep() {
    guard let previousIdleTimerDisabled else { return }
    UIApplication.shared.isIdleTimerDisabled = previousIdleTimerDisabled
    self.previousIdleTimerDisabled = nil
  }

  private func resetPlayersAndQueue() {
    sourcePlayer.pause()
    sourcePlayer.currentItem?.cancelPendingSeeks()
    // A seek starts a new playback generation. Detach the old source item so
    // the first restored segment can install the deferred item for the new
    // generation instead of mistaking the stale one for an active source.
    sourcePlayer.replaceCurrentItem(with: nil)
    sourceSeekTimeoutTask?.cancel()
    sourceSeekTimeoutTask = nil
    sourceSeekAttemptID = nil
    deferredSourcePlayerItem = nil
    sourceStatusObservation?.invalidate()
    sourceStatusObservation = nil
    sourceTimeControlObservation?.invalidate()
    sourceTimeControlObservation = nil
    sourceReady = false
    sourceSeekCompleted = false
    sourceTimeOffset = 0
    sourceSeekErrorSeconds = 0
    latestClockDriftSeconds = 0
    liveSourceWindowStartSeconds = 0
    sourceSeekRevision += 1
    streamingClockCorrectionRevision &+= 1
    streamingClockCorrectionInFlight = false
    streamingRestoredHeldForSourceCatchup = false
    clearRestoredQueue(removingFiles: false)
  }

  private func clearRestoredQueue(removingFiles: Bool) {
    restoredPlayer.pause()
    restoredPlayer.removeAllItems()
    currentRestoredItemIdentifier = nil
    currentRestoredItemStartedAt = 0
    latestRestoredPixelBuffer = nil
    hasPresentedRestoredFrame = false
    for token in notificationTokens.values {
      NotificationCenter.default.removeObserver(token)
    }
    notificationTokens.removeAll(keepingCapacity: true)
    itemSegments.removeAll(keepingCapacity: true)
    itemVideoOutputs.removeAll(keepingCapacity: true)
    if removingFiles {
      for segment in queuedSegments {
        try? FileManager.default.removeItem(at: segment.url)
      }
    }
    queuedSegments.removeAll(keepingCapacity: true)
    bufferedSeconds = 0
  }

  private func releaseFinishedSession() {
    let directory = sessionDirectory
    let lease = securityLease
    let proxy = authenticatedMediaProxy
    let capture = hlsAVFoundationCapture
    streamingMetricsTask?.cancel()
    streamingMetricsTask = nil
    sessionDirectory = nil
    securityLease = nil
    authenticatedMediaProxy = nil
    hlsAVFoundationCapture = nil
    proxy?.stop()
    capture?.cancel()
    sourcePlayer.replaceCurrentItem(with: nil)
    sourceStatusObservation?.invalidate()
    sourceStatusObservation = nil
    sourceTimeControlObservation?.invalidate()
    sourceTimeControlObservation = nil
    sourceReady = false
    sourceSeekCompleted = false
    releaseBrowserHandoffLease()
    endPreventingSleep()
    _ = lease
    if let directory {
      try? FileManager.default.removeItem(at: directory)
    }
  }

  private func fail(_ message: String) {
    generation += 1
    let retiringTask = productionTask
    let directory = sessionDirectory
    let lease = securityLease
    let proxy = authenticatedMediaProxy
    let capture = hlsAVFoundationCapture
    retiringTask?.cancel()
    streamingMetricsTask?.cancel()
    streamingMetricsTask = nil
    productionTask = nil
    authenticatedMediaProxy = nil
    hlsAVFoundationCapture = nil
    proxy?.stop()
    capture?.cancel()
    resetPlayersAndQueue()
    sourcePlayer.replaceCurrentItem(with: nil)
    sessionDirectory = nil
    securityLease = nil
    sourceStatusObservation?.invalidate()
    sourceStatusObservation = nil
    sourceTimeControlObservation?.invalidate()
    sourceTimeControlObservation = nil
    releaseBrowserHandoffLease()
    endPreventingSleep()
    shouldPlay = false
    state = .failed(message)
    persistRealtimeDiagnostics(failureMessage: message)
    Task {
      if let retiringTask { await retiringTask.value }
      _ = lease
      if let directory {
        try? FileManager.default.removeItem(at: directory)
      }
    }
  }

  private func releaseBrowserHandoffLease() {
    guard let lease = browserHandoffLease else { return }
    browserHandoffLease = nil
    lease.beginEnding()
    Task { @MainActor in await lease.end() }
  }

  private func hlsPlaybackFailureMessage(
    stage: String,
    error: Error?
  ) -> String {
    let nsError = error.map { $0 as NSError }
    let description = error?.localizedDescription ?? "不明な再生エラー"
    let errorCode = nsError.map { " [\($0.domain):\($0.code)]" } ?? ""
    let proxyDetail =
      authenticatedMediaProxy.map {
        "\nHLS診断: \($0.diagnosticSummary())"
      } ?? ""
    return "\(stage)に失敗しました: \(description)\(errorCode)\(proxyDetail)"
  }

  private func requireBrowserInteraction(
    _ challengedURL: URL?,
    source: IPadResolvedMediaSource?,
    generation: Int
  ) {
    guard self.generation == generation else { return }
    interactionRequiredURL =
      source?.requestContext?.referer
      ?? source?.submittedURL
      ?? challengedURL
    fail("配信側の確認が再び必要です。ブラウザで確認を完了してください。")
  }
}
