import Foundation
import ObjectiveC.runtime

private enum ProxyHarnessFailure: Error, CustomStringConvertible {
  case assertion(String)

  var description: String {
    switch self {
    case .assertion(let message): message
    }
  }
}

private struct OriginRequestRecord: Sendable {
  let method: String
  let url: String
  let cookie: String?
  let referer: String?
  let origin: String?
  let userAgent: String?
  let accept: String?
  let range: String?
}

private final class ProxyOriginURLProtocol: URLProtocol, @unchecked Sendable {
  private static let lock = NSLock()
  private static var records: [OriginRequestRecord] = []
  private static var transientRateLimitRequestCount = 0

  static func reset() {
    lock.lock()
    records.removeAll()
    transientRateLimitRequestCount = 0
    lock.unlock()
  }

  static func snapshot() -> [OriginRequestRecord] {
    lock.lock()
    let snapshot = records
    lock.unlock()
    return snapshot
  }

  override class func canInit(with request: URLRequest) -> Bool {
    request.url?.scheme?.lowercased() == "https"
      && ["1.1.1.1", "1.0.0.1"].contains(request.url?.host ?? "")
  }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest {
    request
  }

  override func startLoading() {
    guard let url = request.url else {
      client?.urlProtocol(self, didFailWithError: URLError(.badURL))
      return
    }
    let transientRateLimitAttempt: Int
    Self.lock.lock()
    Self.records.append(
      OriginRequestRecord(
        method: request.httpMethod ?? "GET",
        url: url.absoluteString,
        cookie: request.value(forHTTPHeaderField: "Cookie"),
        referer: request.value(forHTTPHeaderField: "Referer"),
        origin: request.value(forHTTPHeaderField: "Origin"),
        userAgent: request.value(forHTTPHeaderField: "User-Agent"),
        accept: request.value(forHTTPHeaderField: "Accept"),
        range: request.value(forHTTPHeaderField: "Range")
      )
    )
    if url.path == "/transient-rate-limit.ts" {
      Self.transientRateLimitRequestCount += 1
      transientRateLimitAttempt = Self.transientRateLimitRequestCount
    } else {
      transientRateLimitAttempt = 0
    }
    Self.lock.unlock()

    switch url.path {
    case "/origin/master.m3u8":
      guard
        let destination = URL(
          string:
            "https://1.1.1.1/redirected/master.m3u8?master_token=a%2Bb&keep=2"
        ),
        let response = HTTPURLResponse(
          url: url,
          statusCode: 302,
          httpVersion: "HTTP/1.1",
          headerFields: [
            "Location": destination.absoluteString,
            "Set-Cookie":
              "redirect_cookie=redirected; Path=/; Secure; SameSite=None",
          ]
        )
      else {
        client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
        return
      }
      client?.urlProtocol(
        self,
        wasRedirectedTo: URLRequest(url: destination),
        redirectResponse: response
      )

    case "/redirected/master.m3u8":
      respond(
        url: url,
        contentType: "application/vnd.apple.mpegurl",
        headers: [
          "Set-Cookie": "master_cookie=master; Path=/redirected; Secure; SameSite=None"
        ],
        body: """
          #EXTM3U
          #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="Japanese",URI="audio/audio.m3u8?lang=ja%2Ben"
          #EXT-X-STREAM-INF:BANDWIDTH=4000000,AUDIO="audio"
          variants/main.m3u8?variant=high%2Bmain
          """
      )

    case "/redirected/audio/audio.m3u8":
      respond(
        url: url,
        contentType: "application/vnd.apple.mpegurl",
        body: """
          #EXTM3U
          #EXTINF:4,
          track.aac?part=1%2B2
          #EXT-X-ENDLIST
          """
      )

    case "/redirected/variants/main.m3u8":
      respond(
        url: url,
        contentType: "application/vnd.apple.mpegurl",
        headers: [
          "Set-Cookie": "variant_cookie=variant; Path=/redirected; Secure; SameSite=None"
        ],
        body: """
          #EXTM3U
          #EXT-X-KEY:METHOD=AES-128,URI="https://1.0.0.1/keys/key.bin?key=a%2Bb"
          #EXT-X-MAP:URI="../init/init.mp4?map=one%2Btwo"
          #EXTINF:4,
          ../segments/chunk.ts?segment=1%2B2
          #EXT-X-ENDLIST
          """
      )

    case "/keys/key.bin":
      respond(
        url: url,
        contentType: "application/octet-stream",
        bodyData: Data(repeating: 7, count: 16)
      )

    case "/redirected/init/init.mp4":
      respond(
        url: url,
        contentType: "video/mp4",
        bodyData: Data("init-fragment".utf8)
      )

    case "/redirected/segments/chunk.ts":
      guard
        let destination = URL(
          string:
            "https://1.1.1.1/redirected/segments/final-chunk.ts?segment=1%2B2"
        ),
        let response = HTTPURLResponse(
          url: url,
          statusCode: 302,
          httpVersion: "HTTP/1.1",
          headerFields: ["Location": destination.absoluteString]
        )
      else {
        client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
        return
      }
      client?.urlProtocol(
        self,
        wasRedirectedTo: URLRequest(url: destination),
        redirectResponse: response
      )

    case "/redirected/segments/final-chunk.ts":
      let fullBody = Data("segment-payload".utf8)
      if request.value(forHTTPHeaderField: "Range") == "bytes=1-3" {
        respond(
          url: url,
          statusCode: 206,
          contentType: "video/mp2t",
          headers: [
            "Accept-Ranges": "bytes",
            "Content-Range": "bytes 1-3/\(fullBody.count)",
          ],
          bodyData: fullBody.subdata(in: 1..<4)
        )
      } else {
        respond(url: url, contentType: "video/mp2t", bodyData: fullBody)
      }

    case "/redirected/audio/track.aac":
      respond(
        url: url,
        contentType: "audio/aac",
        bodyData: Data("audio".utf8)
      )

    case let path where path.hasPrefix("/concurrency/"):
      Thread.sleep(forTimeInterval: 0.15)
      respond(
        url: url,
        contentType: "video/mp2t",
        bodyData: Data("queued-media".utf8)
      )

    case "/transient-rate-limit.ts":
      if transientRateLimitAttempt == 1 {
        respond(
          url: url,
          statusCode: 429,
          contentType: "text/plain",
          headers: ["Retry-After": "1"],
          bodyData: Data("slow-down".utf8)
        )
      } else {
        respond(
          url: url,
          contentType: "video/mp2t",
          bodyData: Data("recovered-after-cooldown".utf8)
        )
      }

    default:
      respond(
        url: url,
        statusCode: 404,
        contentType: "text/plain",
        bodyData: Data("missing".utf8)
      )
    }
  }

  override func stopLoading() {}

  private func respond(
    url: URL,
    statusCode: Int = 200,
    contentType: String,
    headers: [String: String] = [:],
    body: String
  ) {
    respond(
      url: url,
      statusCode: statusCode,
      contentType: contentType,
      headers: headers,
      bodyData: Data(body.utf8)
    )
  }

  private func respond(
    url: URL,
    statusCode: Int = 200,
    contentType: String,
    headers: [String: String] = [:],
    bodyData: Data
  ) {
    var responseHeaders = headers
    responseHeaders["Content-Type"] = contentType
    responseHeaders["Content-Length"] = String(bodyData.count)
    guard
      let response = HTTPURLResponse(
        url: url,
        statusCode: statusCode,
        httpVersion: "HTTP/1.1",
        headerFields: responseHeaders
      )
    else {
      client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
      return
    }
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    if request.httpMethod?.uppercased() != "HEAD" {
      client?.urlProtocol(self, didLoad: bodyData)
    }
    client?.urlProtocolDidFinishLoading(self)
  }
}

private actor ProxyHarnessBrowserResourceLoader: IPadHLSResourceLoading {
  private var requestCount = 0
  private var rateLimitRequestCount = 0
  private var attemptedRequestCount = 0
  private var playlistRequestCount = 0

  func load(
    _ request: URLRequest,
    maximumResponseBytes: Int,
    resolutionPolicy: IPadMediaURLResolutionPolicy
  ) async throws -> IPadHLSResourceLoadResult? {
    guard let url = request.url else { return nil }
    if [
      "/browser-master.m3u8",
      "/browser-audio.m3u8",
      "/browser-media.m3u8",
    ].contains(url.path) {
      playlistRequestCount += 1
      let text: String
      switch url.path {
      case "/browser-master.m3u8":
        text = """
          #EXTM3U
          #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="Main",URI="browser-audio.m3u8"
          #EXT-X-STREAM-INF:BANDWIDTH=1000000,AUDIO="audio"
          browser-media.m3u8
          """
      case "/browser-audio.m3u8":
        text = """
          #EXTM3U
          #EXTINF:4,
          browser-audio.aac
          #EXT-X-ENDLIST
          """
      default:
        text = """
          #EXTM3U
          #EXTINF:4,
          browser-segment.ts
          #EXT-X-ENDLIST
          """
      }
      let body = Data(text.utf8)
      guard body.count <= maximumResponseBytes,
        let response = HTTPURLResponse(
          url: url,
          statusCode: 200,
          httpVersion: "HTTP/3",
          headerFields: [
            "Content-Type": "application/vnd.apple.mpegurl",
            "Content-Length": String(body.count),
          ]
        )
      else {
        throw ProxyHarnessFailure.assertion(
          "could not construct browser playlist response"
        )
      }
      return IPadHLSResourceLoadResult(data: body, response: response)
    }
    guard
      url.path == "/browser-relay.ts" || url.path == "/browser-rate-limit.ts"
        || url.path == "/browser-attempted.ts"
    else { return nil }
    requestCount += 1
    let isRateLimitPath = url.path == "/browser-rate-limit.ts"
    if isRateLimitPath { rateLimitRequestCount += 1 }
    let isAttemptedPath = url.path == "/browser-attempted.ts"
    if isAttemptedPath {
      attemptedRequestCount += 1
      if attemptedRequestCount <= 2 {
        throw IPadHLSResourceLoadingError.attemptedUnavailable
      }
    }
    let isRateLimited = isRateLimitPath && rateLimitRequestCount == 1
    let body = Data(
      (isRateLimited
        ? "browser-rate-limited"
        : isRateLimitPath
          ? "browser-recovered"
          : isAttemptedPath
            ? "browser-recovered-after-attempt"
            : "browser-relayed").utf8
    )
    var headers = [
      "Content-Type": "video/mp2t",
      "Content-Length": String(body.count),
    ]
    if isRateLimited { headers["Retry-After"] = "1" }
    guard body.count <= maximumResponseBytes,
      let response = HTTPURLResponse(
        url: url,
        statusCode: isRateLimited ? 429 : 200,
        httpVersion: "HTTP/3",
        headerFields: headers
      )
    else {
      throw ProxyHarnessFailure.assertion("could not construct browser relay response")
    }
    return IPadHLSResourceLoadResult(data: body, response: response)
  }

  func count() -> Int { requestCount }
  func playlistCount() -> Int { playlistRequestCount }
}

extension URLSessionConfiguration {
  @objc fileprivate dynamic func miohProxyHarnessProtocolClasses() -> [AnyClass]? {
    var classes = self.miohProxyHarnessProtocolClasses() ?? []
    if !classes.contains(where: { $0 === ProxyOriginURLProtocol.self }) {
      classes.insert(ProxyOriginURLProtocol.self, at: 0)
    }
    return classes
  }
}

@main
struct IPadAuthenticatedMediaProxyHarness {
  static func main() async {
    do {
      try await run()
      print("iPad authenticated media proxy probe passed")
    } catch {
      fputs("proxy harness failed: \(error)\n", stderr)
      exit(1)
    }
  }

  private static func run() async throws {
    ProxyOriginURLProtocol.reset()
    try installOriginProtocolInjection()

    guard
      let cookie = IPadMediaRequestCookie(
        name: "bootstrap_cookie",
        value: "bootstrap",
        domain: "1.1.1.1",
        path: "/",
        isSecure: true,
        expiresAt: nil,
        includesSubdomains: false,
        sameSitePolicy: .none
      )
    else {
      throw ProxyHarnessFailure.assertion("fixture cookie was rejected")
    }
    guard
      let keyCookie = IPadMediaRequestCookie(
        name: "key_cookie",
        value: "key",
        domain: "1.0.0.1",
        path: "/",
        isSecure: true,
        expiresAt: nil,
        includesSubdomains: false,
        sameSitePolicy: .none
      )
    else {
      throw ProxyHarnessFailure.assertion("fixture key cookie was rejected")
    }
    let pageURL = requiredURL("https://1.1.1.1/watch/page")
    let context = IPadMediaRequestContext(
      cookies: [cookie, keyCookie],
      userAgent: "MiohProxyHarness/1.0",
      referer: pageURL,
      origin: requiredURL("https://1.1.1.1"),
      cookieSourceURL: pageURL,
      allowsCredentialReplay: true,
      allowsCrossSiteCredentialReplay: true
    )
    let proxy = IPadAuthenticatedMediaProxy(
      configuration: .init(
        maximumConcurrentRequests: 4,
        maximumMappedTargets: 128,
        maximumRequestHeaderBytes: 16 * 1_024,
        maximumResponseBytes: 2 * 1_024 * 1_024,
        maximumPlaylistBytes: 256 * 1_024,
        maximumRedirectCount: 4,
        requestTimeout: 10
      )
    )
    defer { proxy.stop() }
    try await proxy.start()
    let masterURL = try proxy.localURL(
      for: requiredURL("https://1.1.1.1/origin/master.m3u8?entry=one%2Btwo"),
      context: context,
      isPlaylist: true,
      resolutionPolicy: .publicDiscovered
    )

    let head = try await fetch(masterURL, method: "HEAD")
    try require(
      head.response.statusCode == 200,
      "master HEAD did not succeed: \(head.response.statusCode); \(proxy.diagnosticSummary())"
    )
    try require(head.data.isEmpty, "master HEAD returned a body")
    try require(
      head.response.mimeType == "application/vnd.apple.mpegurl",
      "master HEAD lost the HLS content type"
    )

    let master = try await fetch(masterURL)
    try require(master.response.statusCode == 200, "master GET did not succeed")
    let masterText = String(decoding: master.data, as: UTF8.self)
    try require(masterText.hasPrefix("#EXTM3U"), "master body was not returned")
    let childPlaylists = localURLs(in: masterText)
    try require(childPlaylists.count == 2, "master child URIs were not both rewritten")
    try require(
      childPlaylists.allSatisfy { $0.host == "127.0.0.1" && $0.path.hasSuffix("index.m3u8") },
      "master exposed a non-local child URI"
    )

    var mainVariantURL: URL?
    for childURL in childPlaylists {
      let child = try await fetch(childURL)
      let childText = String(decoding: child.data, as: UTF8.self)
      if childText.contains("#EXT-X-KEY:") { mainVariantURL = childURL }
    }
    guard let mainVariantURL else {
      throw ProxyHarnessFailure.assertion("main media playlist was not reachable")
    }

    let variant = try await fetch(mainVariantURL)
    let variantText = String(decoding: variant.data, as: UTF8.self)
    let resources = localURLs(in: variantText)
    try require(resources.count == 3, "key/map/segment were not all rewritten")
    try require(
      resources.allSatisfy { $0.host == "127.0.0.1" && $0.path.hasSuffix("resource") },
      "media playlist exposed a non-local resource URI"
    )

    let key = try await fetch(resources[0])
    try require(key.data.count == 16, "AES key response was not relayed")
    let map = try await fetch(resources[1])
    try require(!map.data.isEmpty, "initialization map was not relayed")
    let segment = try await fetch(resources[2], range: "bytes=1-3")
    try require(segment.response.statusCode == 206, "segment Range status was lost")
    try require(segment.data.count == 3, "segment Range body was lost")
    try require(
      segment.response.value(forHTTPHeaderField: "Content-Range") != nil,
      "segment Content-Range was lost"
    )

    let llHLSURL = requiredURL(
      mainVariantURL.absoluteString + "?_HLS_msn=7&_HLS_part=2"
    )
    let llHLS = try await fetch(llHLSURL)
    try require(llHLS.response.statusCode == 200, "LL-HLS directive request failed")

    let selectedVideoURL = requiredURL(
      "https://1.1.1.1/redirected/variants/main.m3u8?variant=high%2Bmain"
    )
    let selectedAudioURL = requiredURL(
      "https://1.1.1.1/redirected/audio/audio.m3u8?lang=ja%2Ben"
    )
    let selectedMasterURL = try proxy.localURL(
      forSelectedHLSMaster: IPadHLSMasterMetadata(
        masterURL: requiredURL("https://1.1.1.1/origin/master.m3u8"),
        selectedVideoPlaylistURL: selectedVideoURL,
        bandwidth: 4_000_000,
        averageBandwidth: 3_500_000,
        width: 1_920,
        height: 1_080,
        codecs: "avc1.640028,mp4a.40.2",
        frameRate: 29.97,
        audioGroupID: "audio",
        audioRenditions: [
          IPadHLSAudioRendition(
            groupID: "audio",
            name: "Japanese",
            language: "ja",
            url: selectedAudioURL,
            isDefault: true,
            autoSelect: true,
            channels: "2"
          )
        ]
      ),
      context: context,
      resolutionPolicy: .publicDiscovered
    )
    let selectedMaster = try await fetch(selectedMasterURL)
    let selectedMasterText = String(decoding: selectedMaster.data, as: UTF8.self)
    let selectedChildren = localURLs(in: selectedMasterText)
    try require(
      selectedChildren.count == 2,
      "synthetic selected master did not retain video + separate audio"
    )
    try require(
      selectedMasterText.contains("#EXT-X-MEDIA:TYPE=AUDIO")
        && selectedMasterText.contains("AUDIO=\"audio\"")
        && !selectedMasterText.contains("https://1.1.1.1"),
      "synthetic selected master exposed origin URLs or lost AUDIO metadata"
    )

    let queuedURLs = try (0..<6).map { index in
      try proxy.localURL(
        for: requiredURL("https://1.1.1.1/concurrency/\(index).ts"),
        context: context,
        isPlaylist: false,
        resolutionPolicy: .publicDiscovered
      )
    }
    let queuedStatuses = try await withThrowingTaskGroup(of: Int.self) { group in
      for url in queuedURLs {
        group.addTask { try await fetch(url).response.statusCode }
      }
      var statuses: [Int] = []
      for try await status in group { statuses.append(status) }
      return statuses
    }
    try require(
      queuedStatuses.count == 6 && queuedStatuses.allSatisfy { $0 == 200 },
      "requests above the four-origin limit were rejected instead of queued"
    )

    let transientRateLimitURL = try proxy.localURL(
      for: requiredURL("https://1.1.1.1/transient-rate-limit.ts"),
      context: context,
      isPlaylist: false,
      resolutionPolicy: .publicDiscovered
    )
    let rateLimitStartedAt = Date()
    let recoveredRateLimitResponse = try await fetch(transientRateLimitURL)
    let rateLimitElapsed = Date().timeIntervalSince(rateLimitStartedAt)
    try require(
      recoveredRateLimitResponse.response.statusCode == 200
        && recoveredRateLimitResponse.data == Data("recovered-after-cooldown".utf8),
      "transient origin 429 leaked through the loopback proxy"
    )
    let rateLimitOriginRequests = ProxyOriginURLProtocol.snapshot().filter {
      URL(string: $0.url)?.path == "/transient-rate-limit.ts"
    }
    try require(
      rateLimitOriginRequests.count == 2,
      "transient origin 429 was not retried exactly once"
    )
    try require(
      rateLimitElapsed >= 0.75,
      "transient origin 429 retry bypassed the shared Retry-After cooldown"
    )

    let browserResourceLoader = ProxyHarnessBrowserResourceLoader()
    let relayProxy = IPadAuthenticatedMediaProxy(
      resourceLoader: browserResourceLoader
    )
    defer { relayProxy.stop() }
    try await relayProxy.start()
    let browserRelayURL = try relayProxy.localURL(
      for: requiredURL("https://1.1.1.1/browser-relay.ts"),
      context: context,
      isPlaylist: false,
      resolutionPolicy: .publicDiscovered
    )
    let browserRelay = try await fetch(browserRelayURL)
    try require(
      browserRelay.response.statusCode == 200
        && browserRelay.data == Data("browser-relayed".utf8),
      "browser resource relay was not used by the loopback proxy"
    )
    let browserRateLimitURL = try relayProxy.localURL(
      for: requiredURL("https://1.1.1.1/browser-rate-limit.ts"),
      context: context,
      isPlaylist: false,
      resolutionPolicy: .publicDiscovered
    )
    let browserRateLimitStartedAt = Date()
    let browserRateLimit = try await fetch(browserRateLimitURL)
    let browserRateLimitElapsed = Date().timeIntervalSince(
      browserRateLimitStartedAt
    )
    try require(
      browserRateLimit.response.statusCode == 200
        && browserRateLimit.data == Data("browser-recovered".utf8),
      "transient browser relay 429 leaked through the loopback proxy"
    )
    let browserRelayRequestCount = await browserResourceLoader.count()
    try require(
      browserRelayRequestCount == 3,
      "browser relay 429 was not retried exactly once"
    )
    try require(
      browserRateLimitElapsed >= 0.75,
      "browser relay 429 retry bypassed its bounded Retry-After delay"
    )
    let browserAttemptedURL = try relayProxy.localURL(
      for: requiredURL("https://1.1.1.1/browser-attempted.ts"),
      context: context,
      isPlaylist: false,
      resolutionPolicy: .publicDiscovered
    )
    let browserAttemptedStartedAt = Date()
    let browserAttempted = try await fetch(browserAttemptedURL)
    let browserAttemptedElapsed = Date().timeIntervalSince(
      browserAttemptedStartedAt
    )
    try require(
      browserAttempted.response.statusCode == 200
        && browserAttempted.data
          == Data("browser-recovered-after-attempt".utf8),
      "post-dispatch browser relay failure leaked through the loopback proxy"
    )
    let browserAttemptedRequestCount = await browserResourceLoader.count()
    try require(
      browserAttemptedRequestCount == 6,
      "post-dispatch browser relay failure did not use two bounded retries"
    )
    try require(
      browserAttemptedElapsed >= 2.0,
      "post-dispatch browser relay retry bypassed its bounded delays"
    )
    try require(
      !ProxyOriginURLProtocol.snapshot().contains {
        let path = URL(string: $0.url)?.path
        return path == "/browser-relay.ts" || path == "/browser-rate-limit.ts"
          || path == "/browser-attempted.ts"
      },
      "browser-relayed resource leaked into the native URLSession transport"
    )

    let browserMasterURL = try relayProxy.localURL(
      for: requiredURL("https://1.1.1.1/browser-master.m3u8"),
      context: context,
      isPlaylist: true,
      resolutionPolicy: .publicDiscovered
    )
    let browserMaster = try await fetch(browserMasterURL)
    let browserMasterText = String(decoding: browserMaster.data, as: UTF8.self)
    let browserChildPlaylists = localURLs(in: browserMasterText)
    try require(
      browserMaster.response.statusCode == 200
        && browserChildPlaylists.count == 2,
      "browser master playlist was not relayed and rewritten"
    )
    for childURL in browserChildPlaylists {
      let child = try await fetch(childURL)
      let childText = String(decoding: child.data, as: UTF8.self)
      try require(
        child.response.statusCode == 200
          && localURLs(in: childText).count == 1,
        "browser audio/media playlist was not relayed and rewritten"
      )
    }
    let browserPlaylistRequestCount = await browserResourceLoader.playlistCount()
    try require(
      browserPlaylistRequestCount == 3,
      "playlist or alternate-audio request bypassed the browser loader"
    )
    try require(
      !ProxyOriginURLProtocol.snapshot().contains {
        [
          "/browser-master.m3u8",
          "/browser-audio.m3u8",
          "/browser-media.m3u8",
        ].contains(URL(string: $0.url)?.path ?? "")
      },
      "browser playlist leaked into the native URLSession transport"
    )

    let largeProxy = IPadAuthenticatedMediaProxy()
    defer { largeProxy.stop() }
    try await largeProxy.start()
    let largeRenditions = (0..<4_100).map { index in
      IPadHLSAudioRendition(
        groupID: "large-audio",
        name: "Audio \(index)",
        language: nil,
        url: requiredURL("https://1.1.1.1/large/audio-\(index).m3u8"),
        isDefault: index == 0,
        autoSelect: true,
        channels: "2"
      )
    }
    let largeMasterURL = try largeProxy.localURL(
      forSelectedHLSMaster: IPadHLSMasterMetadata(
        masterURL: requiredURL("https://1.1.1.1/large/master.m3u8"),
        selectedVideoPlaylistURL: requiredURL(
          "https://1.1.1.1/large/video.m3u8"
        ),
        bandwidth: 4_000_000,
        averageBandwidth: nil,
        width: 1_920,
        height: 1_080,
        codecs: "avc1.640028,mp4a.40.2",
        frameRate: 29.97,
        audioGroupID: "large-audio",
        audioRenditions: largeRenditions
      ),
      resolutionPolicy: .publicDiscovered
    )
    let largeMaster = try await fetch(largeMasterURL)
    let largeLocalURLs = localURLs(in: String(decoding: largeMaster.data, as: UTF8.self))
    try require(
      largeLocalURLs.count == 4_101
        && Set(largeLocalURLs.map(\.absoluteString)).count == 4_101,
      "synthetic master with more than 4096 references lost or evicted tokens"
    )

    let records = ProxyOriginURLProtocol.snapshot()
    try verifyOriginRequests(records)
  }

  private static func verifyOriginRequests(
    _ records: [OriginRequestRecord]
  ) throws {
    let requiredPaths = [
      "/origin/master.m3u8",
      "/redirected/master.m3u8",
      "/redirected/audio/audio.m3u8",
      "/redirected/variants/main.m3u8",
      "/keys/key.bin",
      "/redirected/init/init.mp4",
      "/redirected/segments/chunk.ts",
      "/redirected/segments/final-chunk.ts",
    ]
    for path in requiredPaths {
      try require(
        records.contains { URL(string: $0.url)?.path == path },
        "origin path was not requested: \(path)"
      )
    }
    let playlistPaths: Set<String> = [
      "/origin/master.m3u8",
      "/redirected/master.m3u8",
      "/redirected/audio/audio.m3u8",
      "/redirected/variants/main.m3u8",
    ]
    for record in records {
      try require(
        record.userAgent == "MiohProxyHarness/1.0",
        "User-Agent was lost for \(record.url)"
      )
      let isKeyHost = URL(string: record.url)?.host == "1.0.0.1"
      let expectedReferer =
        isKeyHost
        ? "https://1.1.1.1/" : "https://1.1.1.1/watch/page"
      try require(record.referer == expectedReferer, "Referer was lost for \(record.url)")
      try require(
        record.origin == "https://1.1.1.1",
        "Origin was lost for \(record.url)"
      )
      let expectedCookie =
        isKeyHost
        ? "key_cookie=key" : "bootstrap_cookie=bootstrap"
      try require(
        record.cookie?.contains(expectedCookie) == true,
        "Cookie was lost for \(record.url)"
      )
      let path = URL(string: record.url)?.path ?? ""
      if playlistPaths.contains(path) {
        try require(
          record.accept?.lowercased().contains("mpegurl") == true,
          "playlist Accept was lost for \(record.url)"
        )
      } else {
        try require(
          record.accept?.lowercased().contains("mpegurl") != true,
          "playlist Accept leaked to a binary resource: \(record.url)"
        )
      }
    }

    try require(
      records.contains {
        $0.url
          == "https://1.1.1.1/origin/master.m3u8?entry=one%2Btwo"
      },
      "initial master query was not preserved"
    )
    try require(
      records.contains {
        $0.url
          == "https://1.1.1.1/redirected/master.m3u8?master_token=a%2Bb&keep=2"
      },
      "redirect destination query was not preserved"
    )
    try require(
      records.contains {
        $0.url.contains("/redirected/audio/audio.m3u8?lang=ja%2Ben")
      },
      "relative audio query was not preserved"
    )
    try require(
      records.contains {
        $0.url.contains("/redirected/variants/main.m3u8?variant=high%2Bmain")
      },
      "relative variant query was not preserved"
    )
    try require(
      records.contains {
        $0.url == "https://1.0.0.1/keys/key.bin?key=a%2Bb"
      },
      "key query was not preserved"
    )
    try require(
      records.contains {
        $0.url.contains("/redirected/init/init.mp4?map=one%2Btwo")
      },
      "map query was not preserved"
    )
    try require(
      records.contains {
        $0.url.contains("/redirected/segments/chunk.ts?segment=1%2B2")
          && $0.range == "bytes=1-3"
      },
      "segment query or Range was not preserved"
    )
    try require(
      records.contains {
        $0.url.contains("/redirected/segments/final-chunk.ts?segment=1%2B2")
          && $0.range == "bytes=1-3"
          && $0.accept?.lowercased().contains("mpegurl") != true
      },
      "redirected segment inherited playlist Accept or lost Range"
    )
    try require(
      records.contains {
        $0.url.contains("variant=high%2Bmain")
          && $0.url.contains("_HLS_msn=7&_HLS_part=2")
      },
      "LL-HLS directives did not preserve the playlist query"
    )
    try require(
      records.contains {
        URL(string: $0.url)?.path == "/redirected/master.m3u8"
          && $0.cookie?.contains("redirect_cookie=redirected") == true
      },
      "redirect Cookie did not rotate into the redirected master"
    )
    try require(
      records.contains {
        URL(string: $0.url)?.path == "/redirected/variants/main.m3u8"
          && $0.cookie?.contains("master_cookie=master") == true
      },
      "response Cookie did not rotate into the child playlist"
    )
    try require(
      records.contains {
        URL(string: $0.url)?.path == "/redirected/segments/chunk.ts"
          && $0.cookie?.contains("variant_cookie=variant") == true
      },
      "variant Cookie did not rotate into the segment request"
    )
  }

  private static func installOriginProtocolInjection() throws {
    let originalSelector = #selector(getter: URLSessionConfiguration.protocolClasses)
    let replacementSelector = #selector(
      URLSessionConfiguration.miohProxyHarnessProtocolClasses
    )
    guard
      let original = class_getInstanceMethod(
        URLSessionConfiguration.self,
        originalSelector
      ),
      let replacement = class_getInstanceMethod(
        URLSessionConfiguration.self,
        replacementSelector
      )
    else {
      throw ProxyHarnessFailure.assertion("could not install fixture URL protocol")
    }
    method_exchangeImplementations(original, replacement)
  }

  private static func fetch(
    _ url: URL,
    method: String = "GET",
    range: String? = nil
  ) async throws -> (data: Data, response: HTTPURLResponse) {
    var request = URLRequest(url: url)
    request.httpMethod = method
    request.timeoutInterval = 10
    if let range { request.setValue(range, forHTTPHeaderField: "Range") }
    let configuration = URLSessionConfiguration.ephemeral
    configuration.timeoutIntervalForRequest = 10
    configuration.timeoutIntervalForResource = 10
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let (data, response) = try await session.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw ProxyHarnessFailure.assertion("local response was not HTTP")
    }
    return (data, httpResponse)
  }

  private static func localURLs(in text: String) -> [URL] {
    let pattern = #"http://127\.0\.0\.1:\d+/v1/[a-f0-9]{32}/(?:index\.m3u8|resource)"#
    guard let expression = try? NSRegularExpression(pattern: pattern) else {
      return []
    }
    return expression.matches(
      in: text,
      range: NSRange(text.startIndex..., in: text)
    ).compactMap { match in
      Range(match.range, in: text).flatMap { URL(string: String(text[$0])) }
    }
  }

  private static func requiredURL(_ value: String) -> URL {
    guard let url = URL(string: value) else { fatalError("invalid fixture URL") }
    return url
  }

  private static func require(
    _ condition: @autoclosure () -> Bool,
    _ message: String
  ) throws {
    guard condition() else { throw ProxyHarnessFailure.assertion(message) }
  }
}
