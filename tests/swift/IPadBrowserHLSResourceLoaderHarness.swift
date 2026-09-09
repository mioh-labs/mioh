import Foundation
import Dispatch

private enum BrowserRelayHarnessFailure: Error, CustomStringConvertible {
  case assertion(String)

  var description: String {
    switch self {
    case .assertion(let message): message
    }
  }
}

private func requireBrowserRelay(
  _ condition: @autoclosure () -> Bool,
  _ message: String
) throws {
  guard condition() else {
    throw BrowserRelayHarnessFailure.assertion(message)
  }
}

enum IPadBrowserHLSFetchCacheMode: String, Sendable {
  case live = "default"
  case videoOnDemand = "force-cache"
}

actor RelayInvocationProbe {
  private var requests: [URLRequest] = []
  private var activeCount = 0
  private var maximumActiveCount = 0
  private let invalidContentRange: Bool

  init(invalidContentRange: Bool = false) {
    self.invalidContentRange = invalidContentRange
  }

  func relay(_ request: URLRequest) async throws -> IPadHLSResourceLoadResult {
    requests.append(request)
    activeCount += 1
    maximumActiveCount = max(maximumActiveCount, activeCount)
    defer { activeCount -= 1 }
    try await Task.sleep(nanoseconds: 80_000_000)
    let requestedRange = request.value(forHTTPHeaderField: "Range")
    let isRange = requestedRange != nil
    let data: Data
    let validContentRange: String?
    switch requestedRange {
    case "bytes=2-5":
      data = Data([2, 3, 4, 5])
      validContentRange = "bytes 2-5/10"
    case "bytes=2-":
      data = Data((2..<10).map(UInt8.init))
      validContentRange = "bytes 2-9/10"
    case "bytes=-4":
      data = Data((6..<10).map(UInt8.init))
      validContentRange = "bytes 6-9/10"
    default:
      data = Data((0..<10).map(UInt8.init))
      validContentRange = nil
    }
    var headers = ["Content-Type": "video/mp4"]
    if isRange {
      headers["Content-Range"] = invalidContentRange
        ? "bytes 3-6/10" : (validContentRange ?? "bytes 0-9/10")
      headers["Content-Length"] = String(data.count)
      headers["Accept-Ranges"] = "bytes"
    }
    let response = try HTTPURLResponse(
      url: request.url!,
      statusCode: isRange ? 206 : 200,
      httpVersion: "HTTP/1.1",
      headerFields: headers
    ).unwrapBrowserRelay(
      or: BrowserRelayHarnessFailure.assertion("relay response is invalid")
    )
    return IPadHLSResourceLoadResult(
      data: data,
      response: response
    )
  }

  func snapshot() -> [URLRequest] { requests }
  func maximumActive() -> Int { maximumActiveCount }
}

final class RelayLease: @unchecked Sendable {
  let probe: RelayInvocationProbe
  let cancellationDrainNanoseconds: UInt64

  init(
    probe: RelayInvocationProbe,
    cancellationDrainNanoseconds: UInt64 = 0
  ) {
    self.probe = probe
    self.cancellationDrainNanoseconds = cancellationDrainNanoseconds
  }

  func download(
    _ request: URLRequest,
    maximumResponseBytes: Int,
    resolutionPolicy: IPadMediaURLResolutionPolicy
  ) async throws -> IPadHLSResourceLoadResult {
    do {
      return try await probe.relay(request)
    } catch is CancellationError {
      if cancellationDrainNanoseconds > 0 {
        await withCheckedContinuation {
          (continuation: CheckedContinuation<Void, Never>) in
          DispatchQueue.global().asyncAfter(
            deadline: .now()
              + .nanoseconds(Int(cancellationDrainNanoseconds))
          ) {
            continuation.resume()
          }
        }
      }
      throw CancellationError()
    }
  }
}

@main
private struct IPadBrowserHLSResourceLoaderHarness {
  static func main() async throws {
    let url = try URL(string: "https://1.1.1.1/media/segment.mp4")
      .unwrapBrowserRelay(
        or: BrowserRelayHarnessFailure.assertion("fixture URL is invalid")
      )
    let probe = RelayInvocationProbe()
    let lease = RelayLease(probe: probe)
    let loader = IPadBrowserHLSResourceLoader(
      lease: lease,
      cacheMode: .live
    )

    var fullRequest = URLRequest(url: url)
    fullRequest.httpMethod = "GET"
    fullRequest.timeoutInterval = 10
    var rangeRequest = fullRequest
    rangeRequest.setValue("bytes=2-5", forHTTPHeaderField: "Range")

    async let fullResult = loader.load(
      fullRequest,
      maximumResponseBytes: 32,
      resolutionPolicy: .userSubmitted
    )
    async let rangeResult = loader.load(
      rangeRequest,
      maximumResponseBytes: 4,
      resolutionPolicy: .userSubmitted
    )
    async let duplicateRangeResult = loader.load(
      rangeRequest,
      maximumResponseBytes: 4,
      resolutionPolicy: .userSubmitted
    )
    let (loadedFull, loadedRange, loadedDuplicateRange) = try await (
      fullResult, rangeResult, duplicateRangeResult
    )
    let full = try loadedFull.unwrapBrowserRelay(
      or: BrowserRelayHarnessFailure.assertion("full relay was declined")
    )
    let range = try loadedRange.unwrapBrowserRelay(
      or: BrowserRelayHarnessFailure.assertion("range relay was declined")
    )
    let duplicateRange = try loadedDuplicateRange.unwrapBrowserRelay(
      or: BrowserRelayHarnessFailure.assertion("duplicate range relay was declined")
    )

    let requests = await probe.snapshot()
    try requireBrowserRelay(
      requests.count == 2,
      "range-specific coalescing issued \(requests.count) downloads"
    )
    try requireBrowserRelay(
      requests.contains {
        $0.httpMethod == "GET"
          && $0.value(forHTTPHeaderField: "Range") == nil
      } && requests.contains {
        $0.httpMethod == "GET"
          && $0.value(forHTTPHeaderField: "Range") == "bytes=2-5"
      },
      "WKDownload did not preserve the exact Range request"
    )
    try requireBrowserRelay(
      full.response.statusCode == 200
        && full.data == Data((0..<10).map(UInt8.init)),
      "full waiter received the wrong representation"
    )
    try requireBrowserRelay(
      range.response.statusCode == 206 && range.data == Data([2, 3, 4, 5]),
      "range waiter did not receive the native partial response"
    )
    try requireBrowserRelay(
      range.response.value(forHTTPHeaderField: "Content-Range")
        == "bytes 2-5/10",
      "range waiter is missing synthesized Content-Range"
    )
    try requireBrowserRelay(
      duplicateRange.data == range.data
        && duplicateRange.response.statusCode == range.response.statusCode
        && duplicateRange.response.value(forHTTPHeaderField: "Content-Range")
          == range.response.value(forHTTPHeaderField: "Content-Range"),
      "identical range waiters did not share one WKDownload result"
    )

    var openRangeRequest = fullRequest
    openRangeRequest.setValue("bytes=2-", forHTTPHeaderField: "Range")
    let openRange = try await loader.load(
      openRangeRequest,
      maximumResponseBytes: 8,
      resolutionPolicy: .userSubmitted
    ).unwrapBrowserRelay(
      or: BrowserRelayHarnessFailure.assertion("open range was declined")
    )
    var suffixRangeRequest = fullRequest
    suffixRangeRequest.setValue("bytes=-4", forHTTPHeaderField: "Range")
    let suffixRange = try await loader.load(
      suffixRangeRequest,
      maximumResponseBytes: 4,
      resolutionPolicy: .userSubmitted
    ).unwrapBrowserRelay(
      or: BrowserRelayHarnessFailure.assertion("suffix range was declined")
    )
    try requireBrowserRelay(
      openRange.data == Data((2..<10).map(UInt8.init))
        && openRange.response.value(forHTTPHeaderField: "Content-Range")
          == "bytes 2-9/10"
        && suffixRange.data == Data((6..<10).map(UInt8.init))
        && suffixRange.response.value(forHTTPHeaderField: "Content-Range")
          == "bytes 6-9/10",
      "open or suffix Range did not preserve its native 206 response"
    )

    let invalidProbe = RelayInvocationProbe(invalidContentRange: true)
    let invalidLease = RelayLease(probe: invalidProbe)
    let invalidLoader = IPadBrowserHLSResourceLoader(
      lease: invalidLease,
      cacheMode: .live
    )
    var invalidContentRangeWasRejected = false
    do {
      _ = try await invalidLoader.load(
        rangeRequest,
        maximumResponseBytes: 4,
        resolutionPolicy: .userSubmitted
      )
    } catch IPadMediaURLResolverError.invalidByteRange {
      invalidContentRangeWasRejected = true
    }
    try requireBrowserRelay(
      invalidContentRangeWasRejected,
      "mismatched Content-Range was accepted"
    )

    let vodProbe = RelayInvocationProbe()
    let vodLease = RelayLease(probe: vodProbe)
    let vodLoader = IPadBrowserHLSResourceLoader(
      lease: vodLease,
      cacheMode: .videoOnDemand
    )
    _ = try await vodLoader.load(
      fullRequest,
      maximumResponseBytes: 32,
      resolutionPolicy: .userSubmitted
    )
    _ = try await vodLoader.load(
      fullRequest,
      maximumResponseBytes: 32,
      resolutionPolicy: .userSubmitted
    )
    let vodRequests = await vodProbe.snapshot()
    try requireBrowserRelay(
      vodRequests.count == 1,
      "sequential VOD probe and producer load missed the warmed cache"
    )

    let vodRangeProbe = RelayInvocationProbe()
    let vodRangeLease = RelayLease(probe: vodRangeProbe)
    let vodRangeLoader = IPadBrowserHLSResourceLoader(
      lease: vodRangeLease,
      cacheMode: .videoOnDemand
    )
    _ = try await vodRangeLoader.load(
      rangeRequest,
      maximumResponseBytes: 4,
      resolutionPolicy: .userSubmitted
    )
    let cachedRange = try await vodRangeLoader.load(
      rangeRequest,
      maximumResponseBytes: 4,
      resolutionPolicy: .userSubmitted
    ).unwrapBrowserRelay(
      or: BrowserRelayHarnessFailure.assertion("cached range was declined")
    )
    let vodRangeDownloadCount = await vodRangeProbe.snapshot().count
    try requireBrowserRelay(
      vodRangeDownloadCount == 1
        && cachedRange.response.statusCode == 206
        && cachedRange.response.value(forHTTPHeaderField: "Content-Range")
          == "bytes 2-5/10"
        && cachedRange.data == Data([2, 3, 4, 5]),
      "VOD range cache lost the validated partial response"
    )

    let liveHandoffProbe = RelayInvocationProbe()
    let liveHandoffLease = RelayLease(probe: liveHandoffProbe)
    let liveHandoffLoader = IPadBrowserHLSResourceLoader(
      lease: liveHandoffLease,
      cacheMode: .videoOnDemand
    )
    _ = try await liveHandoffLoader.load(
      fullRequest,
      maximumResponseBytes: 32,
      resolutionPolicy: .userSubmitted
    )
    await liveHandoffLoader.updateCacheMode(
      .live,
      retainingResolvedResourceURL: url
    )
    _ = try await liveHandoffLoader.load(
      fullRequest,
      maximumResponseBytes: 32,
      resolutionPolicy: .userSubmitted
    )
    let startupDownloadCount = await liveHandoffProbe.snapshot().count
    try requireBrowserRelay(
      startupDownloadCount == 1,
      "live startup did not consume the resolved playlist exactly once"
    )
    _ = try await liveHandoffLoader.load(
      fullRequest,
      maximumResponseBytes: 32,
      resolutionPolicy: .userSubmitted
    )
    let refreshDownloadCount = await liveHandoffProbe.snapshot().count
    try requireBrowserRelay(
      refreshDownloadCount == 2,
      "live refresh reused the one-shot startup playlist"
    )
    let maximumLiveDownloads = await liveHandoffProbe.maximumActive()
    try requireBrowserRelay(
      maximumLiveDownloads == 1,
      "live resolve and refresh overlapped browser downloads"
    )

    let drainingProbe = RelayInvocationProbe()
    let drainingLease = RelayLease(
      probe: drainingProbe,
      cancellationDrainNanoseconds: 100_000_000
    )
    let drainingLoader = IPadBrowserHLSResourceLoader(
      lease: drainingLease,
      cacheMode: .live
    )
    let drainingLoad = Task {
      try await drainingLoader.load(
        fullRequest,
        maximumResponseBytes: 32,
        resolutionPolicy: .userSubmitted
      )
    }
    while await drainingProbe.snapshot().isEmpty {
      try await Task.sleep(nanoseconds: 5_000_000)
    }
    let cancellationStartedAt = Date()
    await drainingLoader.cancel()
    let cancellationElapsed = Date().timeIntervalSince(cancellationStartedAt)
    do {
      _ = try await drainingLoad.value
    } catch is CancellationError {}
    try requireBrowserRelay(
      cancellationElapsed >= 0.08,
      "loader cancellation returned before its shared download drained"
    )

    var expiredLease: RelayLease? = RelayLease(probe: RelayInvocationProbe())
    let expiredLoader = IPadBrowserHLSResourceLoader(
      lease: expiredLease!,
      cacheMode: .live
    )
    expiredLease = nil
    var expiredLeaseWasCancelled = false
    do {
      _ = try await expiredLoader.load(
        fullRequest,
        maximumResponseBytes: 32,
        resolutionPolicy: .userSubmitted
      )
    } catch is CancellationError {
      expiredLeaseWasCancelled = true
    }
    try requireBrowserRelay(
      expiredLeaseWasCancelled,
      "an expired browser lease declined instead of cancelling"
    )
    print("iPad browser HLS resource loader probe passed")
  }
}

private extension Optional {
  func unwrapBrowserRelay(
    or error: @autoclosure () -> Error
  ) throws -> Wrapped {
    guard let self else { throw error() }
    return self
  }
}
