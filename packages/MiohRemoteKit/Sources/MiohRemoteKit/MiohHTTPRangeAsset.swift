import AVFoundation
import Foundation

public enum MiohHTTPRangeAssetError: LocalizedError, Sendable {
  case invalidContract
  case invalidRange
  case invalidResponse
  case cancelled

  public var errorDescription: String? {
    switch self {
    case .invalidContract:
      return "The HTTP media range contract is invalid."
    case .invalidRange:
      return "AVFoundation requested an invalid media byte range."
    case .invalidResponse:
      return "The HTTP media range response did not match the source contract."
    case .cancelled:
      return "The HTTP media range request was cancelled."
    }
  }
}

private final class MiohNoRedirectSessionDelegate: NSObject, URLSessionTaskDelegate {
  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    // Capability URLs must never be forwarded to another origin.
    completionHandler(nil)
  }
}

/// Owns a custom-scheme AVURLAsset and services its byte requests from one
/// attempt-scoped Coordinator HTTP capability. AVAssetReader does not support
/// ordinary remote AVURLAssets on Apple platforms; the resource-loader bridge
/// preserves random access without downloading the complete source.
public final class MiohHTTPRangeAsset: NSObject, AVAssetResourceLoaderDelegate,
  @unchecked Sendable
{
  public static let defaultPageBytes = 64 * 1_024

  public let asset: AVURLAsset

  private struct ActiveRequest {
    var task: URLSessionDataTask?
    // AVAssetResourceLoader does not retain the request after the delegate
    // returns. Keeping it strong until completion is required; weak capture
    // causes a silent metadata-load hang.
    let loadingRequest: AVAssetResourceLoadingRequest
  }

  private let remoteURL: URL
  private let expectedByteCount: Int64
  private let expectedETag: String
  private let contentType: String
  private let pageBytes: Int64
  private let queue = DispatchQueue(label: "mioh.http-range-asset")
  private let redirectDelegate: MiohNoRedirectSessionDelegate
  private let session: URLSession
  private static let transferTimeout: TimeInterval = 30 * 60
  private static func makeSessionConfiguration() -> URLSessionConfiguration {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    configuration.urlCache = nil
    configuration.httpCookieStorage = nil
    configuration.httpMaximumConnectionsPerHost = 4
    configuration.timeoutIntervalForRequest = Self.transferTimeout
    configuration.timeoutIntervalForResource = 24 * 60 * 60
    return configuration
  }
  private var requests: [ObjectIdentifier: ActiveRequest] = [:]
  private var isCancelled = false

  public init(
    remoteURL: URL,
    expectedByteCount: Int64,
    expectedSHA256: String,
    pageBytes: Int = MiohHTTPRangeAsset.defaultPageBytes
  ) throws {
    let sha = expectedSHA256.lowercased()
    guard let scheme = remoteURL.scheme?.lowercased(),
      scheme == "http" || scheme == "https",
      remoteURL.user == nil,
      remoteURL.password == nil,
      remoteURL.query == nil,
      remoteURL.fragment == nil,
      expectedByteCount > 0,
      sha.utf8.count == 64,
      sha.utf8.allSatisfy({
        (48...57).contains($0) || (97...102).contains($0)
      }),
      pageBytes > 0,
      pageBytes <= 1_048_576
    else { throw MiohHTTPRangeAssetError.invalidContract }

    let pathExtension = remoteURL.pathExtension.lowercased()
    switch pathExtension {
    case "mp4", "m4v":
      contentType = AVFileType.mp4.rawValue
    case "mov":
      contentType = AVFileType.mov.rawValue
    default:
      throw MiohHTTPRangeAssetError.invalidContract
    }
    var custom = URLComponents()
    custom.scheme = "mioh-range"
    custom.host = "asset"
    custom.path = "/\(UUID().uuidString.lowercased())/input.\(pathExtension)"
    guard let customURL = custom.url else {
      throw MiohHTTPRangeAssetError.invalidContract
    }
    self.remoteURL = remoteURL
    self.expectedByteCount = expectedByteCount
    expectedETag = "\"\(sha)\""
    self.pageBytes = Int64(pageBytes)
    let redirectDelegate = MiohNoRedirectSessionDelegate()
    self.redirectDelegate = redirectDelegate
    session = URLSession(
      configuration: Self.makeSessionConfiguration(),
      delegate: redirectDelegate,
      delegateQueue: nil
    )
    asset = AVURLAsset(url: customURL)
    super.init()
    asset.resourceLoader.setDelegate(self, queue: queue)
  }

  public func cancel() {
    queue.async { [weak self] in
      guard let self, !self.isCancelled else { return }
      self.isCancelled = true
      let active = Array(self.requests.values)
      self.requests.removeAll()
      active.forEach {
        $0.task?.cancel()
        $0.loadingRequest.finishLoading(with: MiohHTTPRangeAssetError.cancelled)
      }
      self.session.invalidateAndCancel()
    }
  }

  deinit {
    session.invalidateAndCancel()
  }

  public func resourceLoader(
    _ resourceLoader: AVAssetResourceLoader,
    shouldWaitForLoadingOfRequestedResource loadingRequest: AVAssetResourceLoadingRequest
  ) -> Bool {
    guard !isCancelled else {
      loadingRequest.finishLoading(with: MiohHTTPRangeAssetError.cancelled)
      return true
    }
    if let information = loadingRequest.contentInformationRequest {
      information.contentType = contentType
      information.contentLength = expectedByteCount
      information.isByteRangeAccessSupported = true
    }
    guard let dataRequest = loadingRequest.dataRequest else {
      loadingRequest.finishLoading()
      return true
    }
    let start = max(dataRequest.requestedOffset, dataRequest.currentOffset)
    guard start >= 0, start < expectedByteCount else {
      loadingRequest.finishLoading(with: MiohHTTPRangeAssetError.invalidRange)
      return true
    }
    let requestedBytes: Int64
    if dataRequest.requestsAllDataToEndOfResource {
      requestedBytes = expectedByteCount - start
    } else {
      requestedBytes = Int64(max(1, dataRequest.requestedLength))
    }
    let totalBytes = min(requestedBytes, expectedByteCount - start)
    guard totalBytes > 0 else {
      loadingRequest.finishLoading(with: MiohHTTPRangeAssetError.invalidRange)
      return true
    }
    let identifier = ObjectIdentifier(loadingRequest)
    requests[identifier] = ActiveRequest(task: nil, loadingRequest: loadingRequest)
    fetchRangePage(
      identifier: identifier,
      loadingRequest: loadingRequest,
      dataRequest: dataRequest,
      offset: start,
      remainingBytes: totalBytes
    )
    return true
  }

  private func fetchRangePage(
    identifier: ObjectIdentifier,
    loadingRequest: AVAssetResourceLoadingRequest,
    dataRequest: AVAssetResourceLoadingDataRequest,
    offset: Int64,
    remainingBytes: Int64
  ) {
    guard let active = requests[identifier],
      active.loadingRequest === loadingRequest,
      !isCancelled
    else { return }
    let count = min(pageBytes, min(remainingBytes, expectedByteCount - offset))
    guard count > 0 else {
      requests.removeValue(forKey: identifier)
      loadingRequest.finishLoading()
      return
    }
    let start = offset
    let end = offset + count - 1
    var request = URLRequest(url: remoteURL)
    request.httpMethod = "GET"
    request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    request.timeoutInterval = Self.transferTimeout
    request.setValue("bytes=\(start)-\(end)", forHTTPHeaderField: "Range")
    request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

    let task = session.dataTask(with: request) {
      [weak self, loadingRequest] data, response, error in
      guard let self else { return }
      self.queue.async {
        guard let active = self.requests[identifier],
          active.loadingRequest === loadingRequest,
          !self.isCancelled
        else { return }
        if error != nil {
          // Foundation errors can include the capability URL in userInfo.
          // Normalize them before they cross the worker/logging boundary.
          loadingRequest.finishLoading(with: MiohHTTPRangeAssetError.invalidResponse)
          return
        }
        let expectedContentRange = "bytes \(start)-\(end)/\(self.expectedByteCount)"
        guard let http = response as? HTTPURLResponse,
          http.statusCode == 206,
          http.url?.absoluteString == self.remoteURL.absoluteString,
          http.value(forHTTPHeaderField: "Content-Range") == expectedContentRange,
          http.value(forHTTPHeaderField: "Content-Length") == String(count),
          http.value(forHTTPHeaderField: "ETag") == self.expectedETag,
          let data,
          data.count == Int(count)
        else {
          loadingRequest.finishLoading(with: MiohHTTPRangeAssetError.invalidResponse)
          return
        }
        dataRequest.respond(with: data)
        let nextRemaining = remainingBytes - count
        if nextRemaining > 0 {
          self.fetchRangePage(
            identifier: identifier,
            loadingRequest: loadingRequest,
            dataRequest: dataRequest,
            offset: offset + count,
            remainingBytes: nextRemaining
          )
        } else {
          self.requests.removeValue(forKey: identifier)
          loadingRequest.finishLoading()
        }
      }
    }
    requests[identifier]?.task = task
    task.resume()
  }

  public func resourceLoader(
    _ resourceLoader: AVAssetResourceLoader,
    didCancel loadingRequest: AVAssetResourceLoadingRequest
  ) {
    requests.removeValue(forKey: ObjectIdentifier(loadingRequest))?.task?.cancel()
  }
}
