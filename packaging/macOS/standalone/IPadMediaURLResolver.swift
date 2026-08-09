import Darwin
import Foundation

enum IPadMediaCookieSameSitePolicy: String, Sendable, Equatable {
  case strict
  case lax
  case none
}

struct IPadMediaRequestCookie: Sendable, Equatable {
  let name: String
  let value: String
  let domain: String
  let path: String
  let isSecure: Bool
  let expiresAt: Date?
  let includesSubdomains: Bool
  let sameSitePolicy: IPadMediaCookieSameSitePolicy

  init?(_ cookie: HTTPCookie) {
    let rawDomain = cookie.domain.lowercased()
      .trimmingCharacters(in: .whitespacesAndNewlines)
    let includesSubdomains = rawDomain.hasPrefix(".")
    let domain = rawDomain.trimmingCharacters(
      in: CharacterSet(charactersIn: ".[]")
    )
    let path = cookie.path.isEmpty ? "/" : cookie.path
    guard !cookie.name.isEmpty, !domain.isEmpty, path.hasPrefix("/"),
      Self.isSafeCookieName(cookie.name),
      Self.isSafeCookieValue(cookie.value), Self.isSafeCookiePath(path),
      Self.isSafeCookieDomain(domain)
    else { return nil }
    self.name = cookie.name
    self.value = cookie.value
    self.domain = domain
    self.path = path
    isSecure = cookie.isSecure
    expiresAt = cookie.expiresDate
    self.includesSubdomains = includesSubdomains
    sameSitePolicy = Self.sameSitePolicy(from: cookie)
  }

  init?(
    name: String,
    value: String,
    domain rawDomain: String,
    path: String = "/",
    isSecure: Bool = false,
    expiresAt: Date? = nil,
    includesSubdomains: Bool = false,
    sameSitePolicy: IPadMediaCookieSameSitePolicy = .lax
  ) {
    let domain = rawDomain.lowercased().trimmingCharacters(
      in: CharacterSet(charactersIn: ".[] \t\r\n")
    )
    guard !name.isEmpty, !domain.isEmpty, path.hasPrefix("/"),
      Self.isSafeCookieName(name), Self.isSafeCookieValue(value),
      Self.isSafeCookiePath(path),
      Self.isSafeCookieDomain(domain)
    else { return nil }
    self.name = name
    self.value = value
    self.domain = domain
    self.path = path
    self.isSecure = isSecure
    self.expiresAt = expiresAt
    self.includesSubdomains = includesSubdomains
    self.sameSitePolicy = sameSitePolicy
  }

  func matches(_ url: URL, now: Date = Date()) -> Bool {
    guard let scheme = url.scheme?.lowercased(),
      scheme == "http" || scheme == "https",
      let host = url.host?.lowercased()
        .trimmingCharacters(in: CharacterSet(charactersIn: ".[]")),
      expiresAt.map({ $0 > now }) ?? true,
      !isSecure || scheme == "https"
    else { return false }

    let domainMatches =
      host == domain
      || (includesSubdomains && host.hasSuffix(".\(domain)"))
    guard domainMatches else { return false }

    let requestPath = url.path.isEmpty ? "/" : url.path
    if requestPath == path { return true }
    guard requestPath.hasPrefix(path) else { return false }
    return path.hasSuffix("/")
      || requestPath.dropFirst(path.count).first == "/"
  }

  func makeHTTPCookie() -> HTTPCookie? {
    var properties: [HTTPCookiePropertyKey: Any] = [
      .name: name,
      .value: value,
      .domain: includesSubdomains ? ".\(domain)" : domain,
      .path: path,
    ]
    if isSecure { properties[.secure] = "TRUE" }
    if let expiresAt { properties[.expires] = expiresAt }
    properties[.sameSitePolicy] = sameSitePolicy.rawValue.capitalized
    return HTTPCookie(properties: properties)
  }

  func canReplay(
    to requestURL: URL,
    from sourceURL: URL?,
    credentialsAllowed: Bool,
    crossSiteCredentialsAllowed: Bool
  ) -> Bool {
    guard credentialsAllowed, matches(requestURL) else { return false }
    let sameSite =
      sourceURL.map { Self.isSameSite($0, requestURL, cookie: self) }
      ?? false
    if sameSitePolicy == .none {
      guard isSecure && requestURL.scheme?.lowercased() == "https" else {
        return false
      }
      return sameSite || crossSiteCredentialsAllowed
    }
    return sameSite
  }

  private static func sameSitePolicy(
    from cookie: HTTPCookie
  ) -> IPadMediaCookieSameSitePolicy {
    let rawValue = (cookie.properties?[.sameSitePolicy] as? String)?
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
    switch rawValue {
    case "strict": return .strict
    case "none": return .none
    default: return .lax
    }
  }

  private static func isSameOrigin(_ lhs: URL, _ rhs: URL) -> Bool {
    guard let leftScheme = lhs.scheme?.lowercased(),
      let rightScheme = rhs.scheme?.lowercased(),
      let leftHost = lhs.host?.lowercased(),
      let rightHost = rhs.host?.lowercased()
    else { return false }
    let leftPort = lhs.port ?? (leftScheme == "https" ? 443 : 80)
    let rightPort = rhs.port ?? (rightScheme == "https" ? 443 : 80)
    return leftScheme == rightScheme && leftHost == rightHost
      && leftPort == rightPort
  }

  private static func isSameSite(
    _ sourceURL: URL,
    _ requestURL: URL,
    cookie: IPadMediaRequestCookie
  ) -> Bool {
    guard sourceURL.scheme?.lowercased() == requestURL.scheme?.lowercased(),
      let sourceHost = sourceURL.host?.lowercased()
        .trimmingCharacters(in: CharacterSet(charactersIn: ".[]")),
      let requestHost = requestURL.host?.lowercased()
        .trimmingCharacters(in: CharacterSet(charactersIn: ".[]"))
    else { return false }
    if cookie.includesSubdomains {
      let sourceMatches =
        sourceHost == cookie.domain
        || sourceHost.hasSuffix(".\(cookie.domain)")
      let requestMatches =
        requestHost == cookie.domain
        || requestHost.hasSuffix(".\(cookie.domain)")
      return sourceMatches && requestMatches
    }
    return isSameOrigin(sourceURL, requestURL)
  }

  private static func isSafeCookieName(_ value: String) -> Bool {
    let separators = CharacterSet(charactersIn: "()<>@,;:\\\"/[]?={} \t")
    return !value.isEmpty && value.utf8.count <= 256
      && value.rangeOfCharacter(from: .controlCharacters) == nil
      && value.rangeOfCharacter(from: separators) == nil
  }

  private static func isSafeCookieValue(_ value: String) -> Bool {
    value.utf8.count <= 4_096
      && value.rangeOfCharacter(from: .controlCharacters) == nil
      && !value.contains(";")
  }

  private static func isSafeCookiePath(_ value: String) -> Bool {
    value.utf8.count <= 4_096
      && value.rangeOfCharacter(from: .controlCharacters) == nil
  }

  private static func isSafeCookieDomain(_ domain: String) -> Bool {
    domain.rangeOfCharacter(from: .controlCharacters) == nil
      && !domain.contains("/") && !domain.contains(":")
      && !domain.contains("@") && !domain.contains(" ")
  }
}

private final class IPadMediaCookieSnapshot: @unchecked Sendable {
  private let lock = NSLock()
  private var value: [IPadMediaRequestCookie]

  init(_ value: [IPadMediaRequestCookie]) {
    self.value = value
  }

  func read() -> [IPadMediaRequestCookie] {
    lock.lock()
    defer { lock.unlock() }
    return value
  }

  func replace(with value: [IPadMediaRequestCookie]) {
    lock.lock()
    self.value = value
    lock.unlock()
  }
}

private actor IPadMediaCookieJar {
  private struct Key: Hashable {
    let name: String
    let domain: String
    let path: String
  }

  private struct StoredCookie {
    let cookie: IPadMediaRequestCookie
    let sequence: Int64
  }

  private let maximumCookieCount = 128
  private let snapshot: IPadMediaCookieSnapshot
  private var storage: [Key: StoredCookie] = [:]
  private var nextSequence: Int64 = 0

  init(
    initialCookies: [IPadMediaRequestCookie],
    snapshot: IPadMediaCookieSnapshot
  ) {
    self.snapshot = snapshot
    for cookie in initialCookies.prefix(maximumCookieCount) {
      let key = Key(name: cookie.name, domain: cookie.domain, path: cookie.path)
      storage[key] = StoredCookie(cookie: cookie, sequence: nextSequence)
      nextSequence += 1
    }
  }

  func update(from response: HTTPURLResponse) {
    guard let responseURL = response.url,
      let responseHost = Self.normalizedHost(responseURL),
      let fields = Self.responseHeaderFields(response)
    else { return }

    let existingBroadDomains = Set(
      storage.values.compactMap { stored in
        stored.cookie.includesSubdomains ? stored.cookie.domain : nil
      }
    )
    let parsed = HTTPCookie.cookies(
      withResponseHeaderFields: fields,
      for: responseURL
    )
    let now = Date()
    for rawCookie in parsed {
      guard let cookie = IPadMediaRequestCookie(rawCookie),
        Self.originCanSet(
          cookie,
          responseHost: responseHost,
          existingBroadDomains: existingBroadDomains
        )
      else { continue }
      let key = Key(name: cookie.name, domain: cookie.domain, path: cookie.path)
      if cookie.expiresAt.map({ $0 <= now }) == true {
        storage.removeValue(forKey: key)
        continue
      }
      storage[key] = StoredCookie(cookie: cookie, sequence: nextSequence)
      if nextSequence < Int64.max { nextSequence += 1 }
    }
    publish(now: now)
  }

  private func publish(now: Date) {
    storage = storage.filter { $0.value.cookie.expiresAt.map({ $0 > now }) ?? true }
    if storage.count > maximumCookieCount {
      let overflow = storage.count - maximumCookieCount
      for key in storage.sorted(by: { $0.value.sequence < $1.value.sequence })
        .prefix(overflow).map(\.key)
      {
        storage.removeValue(forKey: key)
      }
    }
    snapshot.replace(
      with: storage.values.sorted { lhs, rhs in
        if lhs.cookie.path.count != rhs.cookie.path.count {
          return lhs.cookie.path.count > rhs.cookie.path.count
        }
        return lhs.sequence < rhs.sequence
      }.map(\.cookie)
    )
  }

  private static func normalizedHost(_ url: URL) -> String? {
    url.host?.lowercased().trimmingCharacters(
      in: CharacterSet(charactersIn: ".[]")
    )
  }

  private static func responseHeaderFields(
    _ response: HTTPURLResponse
  ) -> [String: String]? {
    var fields: [String: String] = [:]
    for (rawName, rawValue) in response.allHeaderFields {
      guard let name = rawName as? String else { continue }
      fields[name] = String(describing: rawValue)
    }
    return fields.isEmpty ? nil : fields
  }

  private static func originCanSet(
    _ cookie: IPadMediaRequestCookie,
    responseHost: String,
    existingBroadDomains: Set<String>
  ) -> Bool {
    if cookie.domain == responseHost { return true }
    // A subdomain may update a parent-domain cookie only when WebKit already
    // accepted that broad scope. This avoids inventing new public-suffix or
    // sibling-host authority from an untrusted media response.
    return cookie.includesSubdomains
      && existingBroadDomains.contains(cookie.domain)
      && responseHost.hasSuffix(".\(cookie.domain)")
  }
}

struct IPadMediaRequestContext: Sendable, Equatable {
  private let cookieSnapshot: IPadMediaCookieSnapshot
  private let cookieJar: IPadMediaCookieJar
  let userAgent: String?
  let referer: URL?
  let origin: URL?
  let cookieSourceURL: URL?
  let allowsCredentialReplay: Bool
  let allowsCrossSiteCredentialReplay: Bool

  var cookies: [IPadMediaRequestCookie] { cookieSnapshot.read() }

  init(
    cookies: [IPadMediaRequestCookie] = [],
    userAgent: String? = nil,
    referer: URL? = nil,
    origin: URL? = nil,
    cookieSourceURL: URL? = nil,
    allowsCredentialReplay: Bool = true,
    allowsCrossSiteCredentialReplay: Bool = false
  ) {
    let initialCookies = Array(cookies.prefix(128))
    let cookieSnapshot = IPadMediaCookieSnapshot(initialCookies)
    self.cookieSnapshot = cookieSnapshot
    cookieJar = IPadMediaCookieJar(
      initialCookies: initialCookies,
      snapshot: cookieSnapshot
    )
    self.userAgent = Self.sanitizedUserAgent(userAgent)
    self.referer = Self.sanitizedReferer(referer)
    self.origin = Self.sanitizedOrigin(origin)
    self.cookieSourceURL = Self.sanitizedReferer(cookieSourceURL ?? referer)
    self.allowsCredentialReplay = allowsCredentialReplay
    self.allowsCrossSiteCredentialReplay = allowsCrossSiteCredentialReplay
  }

  static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.cookies == rhs.cookies
      && lhs.userAgent == rhs.userAgent
      && lhs.referer == rhs.referer
      && lhs.origin == rhs.origin
      && lhs.cookieSourceURL == rhs.cookieSourceURL
      && lhs.allowsCredentialReplay == rhs.allowsCredentialReplay
      && lhs.allowsCrossSiteCredentialReplay == rhs.allowsCrossSiteCredentialReplay
  }

  func updateCookies(from response: HTTPURLResponse) async {
    await cookieJar.update(from: response)
  }

  func applying(to request: inout URLRequest) {
    request.httpShouldHandleCookies = false
    request.setValue(nil, forHTTPHeaderField: "Cookie")
    request.setValue(nil, forHTTPHeaderField: "User-Agent")
    request.setValue(nil, forHTTPHeaderField: "Referer")
    request.setValue(nil, forHTTPHeaderField: "Origin")
    guard let url = request.url else { return }
    if let userAgent {
      request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
    }
    if let referer = referer(for: url) {
      request.setValue(referer.absoluteString, forHTTPHeaderField: "Referer")
    }
    if let origin {
      request.setValue(origin.absoluteString, forHTTPHeaderField: "Origin")
    }
    let header = cookieHeader(for: url)
    if !header.isEmpty {
      request.setValue(header, forHTTPHeaderField: "Cookie")
    }
  }

  func httpCookies(relevantTo urls: [URL]) -> [HTTPCookie] {
    cookies.compactMap { cookie in
      guard
        urls.contains(where: {
          cookie.canReplay(
            to: $0,
            from: cookieSourceURL,
            credentialsAllowed: allowsCredentialReplay,
            crossSiteCredentialsAllowed: allowsCrossSiteCredentialReplay
          )
        })
      else { return nil }
      return cookie.makeHTTPCookie()
    }
  }

  private func cookieHeader(for url: URL) -> String {
    var byteCount = 0
    return cookies.compactMap { cookie -> String? in
      guard
        cookie.canReplay(
          to: url,
          from: cookieSourceURL,
          credentialsAllowed: allowsCredentialReplay,
          crossSiteCredentialsAllowed: allowsCrossSiteCredentialReplay
        )
      else { return nil }
      let value = "\(cookie.name)=\(cookie.value)"
      let addedBytes = value.utf8.count + (byteCount == 0 ? 0 : 2)
      guard addedBytes <= 16_384 - byteCount else { return nil }
      byteCount += addedBytes
      return value
    }.joined(separator: "; ")
  }

  /// Mirrors strict-origin-when-cross-origin for every request, including
  /// redirects. A path is sent only back to the Referer's own origin.
  private func referer(for requestURL: URL) -> URL? {
    guard let referer,
      var components = URLComponents(
        url: referer,
        resolvingAgainstBaseURL: true
      ),
      let refererScheme = components.scheme?.lowercased(),
      let refererHost = components.host?.lowercased(),
      let requestScheme = requestURL.scheme?.lowercased(),
      let requestHost = requestURL.host?.lowercased()
    else { return nil }

    let refererPort = components.port ?? (refererScheme == "https" ? 443 : 80)
    let requestPort = requestURL.port ?? (requestScheme == "https" ? 443 : 80)
    if refererScheme != requestScheme || refererHost != requestHost
      || refererPort != requestPort
    {
      components.path = "/"
      components.query = nil
      components.fragment = nil
    }
    return components.url
  }

  private static func sanitizedUserAgent(_ rawValue: String?) -> String? {
    guard let value = rawValue?.trimmingCharacters(in: .whitespacesAndNewlines),
      !value.isEmpty, value.utf8.count <= 512,
      value.rangeOfCharacter(from: .controlCharacters) == nil
    else { return nil }
    return value
  }

  private static func sanitizedReferer(_ rawURL: URL?) -> URL? {
    guard let rawURL,
      var components = URLComponents(url: rawURL, resolvingAgainstBaseURL: true),
      let scheme = components.scheme?.lowercased(),
      scheme == "http" || scheme == "https",
      components.user == nil, components.password == nil,
      components.host?.isEmpty == false
    else { return nil }
    components.scheme = scheme
    components.query = nil
    components.fragment = nil
    return components.url
  }

  private static func sanitizedOrigin(_ rawURL: URL?) -> URL? {
    guard let rawURL,
      var components = URLComponents(url: rawURL, resolvingAgainstBaseURL: true),
      let scheme = components.scheme?.lowercased(),
      scheme == "http" || scheme == "https",
      components.user == nil, components.password == nil,
      components.host?.isEmpty == false
    else { return nil }
    components.scheme = scheme
    components.path = ""
    components.query = nil
    components.fragment = nil
    return components.url
  }
}

enum IPadRestorationMediaLimits {
  static let maximumLongEdge = 1_920
  static let maximumShortEdge = 1_080
  static let maximumFramePixels = maximumLongEdge * maximumShortEdge
  static let referenceClipLength = 18
  static let maximumPixelFrames = maximumFramePixels * referenceClipLength
  static let realtimeReferenceClipLength = 30
  static let maximumRealtimePixelFrames =
    maximumFramePixels * realtimeReferenceClipLength

  static func accepts(width: Int, height: Int) -> Bool {
    guard width > 0, height > 0 else { return false }
    let (pixels, overflow) = width.multipliedReportingOverflow(by: height)
    return !overflow
      && max(width, height) <= maximumLongEdge
      && min(width, height) <= maximumShortEdge
      && pixels <= maximumFramePixels
  }

  static func accepts(
    width: Int,
    height: Int,
    clipLength: Int,
    pixelFrameBudget: Int = maximumPixelFrames
  ) -> Bool {
    guard accepts(width: width, height: height), clipLength > 0 else {
      return false
    }
    let (pixels, pixelOverflow) = width.multipliedReportingOverflow(by: height)
    let (pixelFrames, clipOverflow) = pixels.multipliedReportingOverflow(by: clipLength)
    return !pixelOverflow && !clipOverflow && pixelFrames <= pixelFrameBudget
  }
}

struct IPadResolvedMediaSource: Sendable, Equatable {
  enum Kind: Sendable, Equatable {
    case hls
    case progressive
  }

  let kind: Kind
  let submittedURL: URL
  let playbackURL: URL
  let mediaURL: URL
  let contentType: String?
  let hlsPlaylist: IPadHLSMediaPlaylist?
  let resolutionPolicy: IPadMediaURLResolutionPolicy
  let requestContext: IPadMediaRequestContext?

  var requestURLs: [URL] {
    var urls = [playbackURL, mediaURL]
    if let hlsPlaylist {
      urls.append(hlsPlaylist.url)
      for segment in hlsPlaylist.segments {
        urls.append(segment.resource.url)
        if let initializationResource = segment.initializationResource {
          urls.append(initializationResource.url)
        }
      }
    }
    return urls
  }
}

struct IPadBrowserMediaEvidence: Sendable, Equatable {
  let observedDuration: TimeInterval?
  let isPlaying: Bool
  let isVisible: Bool
  /// True only after WebKit has supplied both viewport-intersection and
  /// IntersectionObserver v2 visual-visibility proof, so a transparent or
  /// covered video cannot unlock the VPN fake-IP compatibility path.
  let visibilityAttested: Bool
  let renderedArea: Int
  let sourceGeneration: Int
  let activationOrder: Int
}

/// Chooses among media URLs captured from a visible browser without trusting
/// any one DOM or network observer as authoritative. In particular, a
/// parseable pre-roll must not win merely because it became `currentSrc`
/// before the main programme was requested.
enum IPadBrowserMediaSourceSelector {
  // A player page commonly exposes several pre-roll, tracking and rendition
  // URLs before its programme playlist. Keep the scan bounded, but large
  // enough that a page with more than twelve observations does not hide the
  // later main source.
  static let maximumPlayableChoices = 40

  static func isHighConfidenceAdvertisementSource(
    _ source: IPadResolvedMediaSource
  ) -> Bool {
    let urls = [source.playbackURL, source.mediaURL, source.hlsPlaylist?.url].compactMap {
      $0
    }
    return urls.contains { url in
      guard let host = url.host?.lowercased() else { return false }
      return host == "saawsedge.com" || host.hasSuffix(".saawsedge.com")
    }
  }

  static func deduplicationKey(for source: IPadResolvedMediaSource) -> String {
    let kind = source.kind == .hls ? "hls" : "progressive"
    let canonicalURL = source.hlsPlaylist?.url ?? source.mediaURL
    return kind + "\n" + canonicalURL.absoluteString
  }

  static func preferredIndex(
    in sources: [IPadResolvedMediaSource],
    evidence: [IPadBrowserMediaEvidence?] = []
  ) -> Int? {
    let usable = sources.indices.filter { isUsable(sources[$0]) }
    guard !usable.isEmpty else { return nil }

    // Candidate order carries current-player evidence from the visible
    // browser. Keep that baseline unless a later parsed playlist provides
    // substantially stronger programme evidence. This avoids guessing from
    // host/path words, which are commonly opaque or misleading.
    var preferred = usable[0]
    for candidate in usable.dropFirst() {
      if shouldReplace(
        sources[preferred],
        evidence: evidence.indices.contains(preferred) ? evidence[preferred] : nil,
        with: sources[candidate],
        candidateEvidence: evidence.indices.contains(candidate) ? evidence[candidate] : nil
      ) {
        preferred = candidate
      }
    }
    return preferred
  }

  /// A long, finite HLS programme is sufficiently distinct from a normal
  /// pre-roll to accept immediately. This avoids spending minutes resolving
  /// every stale/advertising hint while a signed programme URL expires.
  static func shouldAcceptImmediately(
    _ source: IPadResolvedMediaSource,
    evidence: IPadBrowserMediaEvidence?
  ) -> Bool {
    guard source.kind == .hls, isUsable(source),
      let duration = programmeDuration(for: source, evidence: evidence)
    else { return false }
    if duration >= 10 * 60 { return true }
    return duration >= 90
      && evidence?.isPlaying == true
      && evidence?.isVisible == true
      && evidence?.visibilityAttested == true
      && (evidence?.renderedArea ?? 0) >= 4_096
  }

  private static func isUsable(_ source: IPadResolvedMediaSource) -> Bool {
    switch source.kind {
    case .hls:
      guard let playlist = source.hlsPlaylist else { return false }
      return !playlist.segments.isEmpty && playlist.duration > 0
    case .progressive:
      return true
    }
  }

  private static func shouldReplace(
    _ current: IPadResolvedMediaSource,
    evidence currentEvidence: IPadBrowserMediaEvidence?,
    with candidate: IPadResolvedMediaSource,
    candidateEvidence: IPadBrowserMediaEvidence?
  ) -> Bool {
    let currentDuration = programmeDuration(
      for: current,
      evidence: currentEvidence
    )
    let candidateDuration = programmeDuration(
      for: candidate,
      evidence: candidateEvidence
    )
    if let currentDuration, let candidateDuration {
      return candidateDuration
        >= max(currentDuration + 10, currentDuration * 1.5)
    }
    if currentDuration == nil, let candidateDuration {
      return candidateDuration >= 90
    }
    return false
  }

  private static func programmeDuration(
    for source: IPadResolvedMediaSource,
    evidence: IPadBrowserMediaEvidence?
  ) -> TimeInterval? {
    if let observed = evidence?.observedDuration,
      observed.isFinite, observed > 0
    {
      return observed
    }
    guard let playlist = source.hlsPlaylist, !playlist.isLive,
      playlist.duration.isFinite, playlist.duration > 0
    else { return nil }
    return playlist.duration
  }
}

enum IPadMediaURLResolutionPolicy: Sendable, Equatable {
  /// A URL entered explicitly by the user may refer to a trusted local server.
  case userSubmitted
  /// A URL found inside an untrusted public page must remain public HTTPS at
  /// every redirect and playlist hop.
  case publicDiscovered
  /// A source visibly playing in WebKit may use 198.18.0.0/15 as a hostname-only
  /// synthetic address supplied by a packet-tunnel VPN. Its public HTTPS HLS
  /// children may use the same translation because master playlists commonly
  /// move variants and segments onto a CDN. Literal benchmark addresses and
  /// every private/link-local range remain forbidden.
  case visibleBrowserDiscovered(URL)
  /// A descendant of an explicitly submitted local page may remain on that
  /// exact origin, but may not redirect or branch to another local service.
  case submittedPageSameOrigin(URL)
}

struct IPadHLSByteRange: Sendable, Hashable {
  let offset: Int64
  let length: Int64

  var endOffset: Int64 { offset + length }
}

struct IPadHLSResource: Sendable, Hashable {
  let url: URL
  let byteRange: IPadHLSByteRange?
}

struct IPadHLSMediaSegment: Sendable, Equatable {
  let sequence: Int64
  let duration: TimeInterval
  let resource: IPadHLSResource
  let initializationResource: IPadHLSResource?
  let startSeconds: TimeInterval
  let discontinuitySequence: Int64

  init(
    sequence: Int64,
    duration: TimeInterval,
    resource: IPadHLSResource,
    initializationResource: IPadHLSResource?,
    startSeconds: TimeInterval,
    discontinuitySequence: Int64 = 0
  ) {
    self.sequence = sequence
    self.duration = duration
    self.resource = resource
    self.initializationResource = initializationResource
    self.startSeconds = startSeconds
    self.discontinuitySequence = discontinuitySequence
  }
}

struct IPadHLSMediaPlaylist: Sendable, Equatable {
  let url: URL
  let segments: [IPadHLSMediaSegment]
  let isLive: Bool
  let duration: TimeInterval
  let targetDuration: TimeInterval?
}

enum IPadMediaURLResolverError: LocalizedError, Sendable {
  case invalidURL
  case unsafeInitialURL
  case unsafeURL
  case requestFailed(String)
  case invalidHTTPStatus(Int)
  case responseTooLarge(Int)
  case tooManyRedirects
  case insecureRedirect
  case unsupportedContent
  case invalidPlaylist(String)
  case encryptedPlaylist
  case invalidByteRange
  case resolutionLimitExceeded
  case interactionRequired(URL?)

  var errorDescription: String? {
    switch self {
    case .invalidURL:
      "URLを確認してください。httpまたはhttpsのURLを入力できます。"
    case .unsafeInitialURL:
      "検出候補を安全な公開HTTPS配信として確認できませんでした。"
    case .unsafeURL:
      "URLの接続先を安全なHTTP/HTTPS配信として確認できませんでした。"
    case .requestFailed(let detail):
      "URLを読み込めませんでした: \(detail)"
    case .invalidHTTPStatus(let status):
      "配信サーバーがHTTP \(status)を返しました。"
    case .responseTooLarge(let maximumBytes):
      "URLの解析データが上限（\(maximumBytes) bytes）を超えました。"
    case .tooManyRedirects:
      "URLのリダイレクト回数が上限を超えました。"
    case .insecureRedirect:
      "HTTPSからHTTPへの安全でないリダイレクトは使用できません。"
    case .unsupportedContent:
      "このURLからHLSまたは対応動画を見つけられませんでした。"
    case .invalidPlaylist(let detail):
      "HLSプレイリストを解析できません: \(detail)"
    case .encryptedPlaylist:
      "暗号化またはDRM付きHLSには対応していません。"
    case .invalidByteRange:
      "HLSのByte Range指定が不正です。"
    case .resolutionLimitExceeded:
      "URLの解析回数、受信量、または解析時間が上限を超えました。"
    case .interactionRequired:
      "対話式のブラウザ確認が必要なため、このページは自動解析できません。"
    }
  }
}

struct IPadMediaURLResolver: Sendable {
  private static let maximumPageDepth = 3
  private static let maximumNestedPagesPerPage = 8
  private static let maximumHLSCandidatesPerPage = 8

  let maximumResponseBytes: Int
  let maximumRedirectCount: Int
  let requestTimeout: TimeInterval
  let maximumRequestCount: Int
  let maximumCumulativeResponseBytes: Int
  let resolutionTimeout: TimeInterval

  init(
    maximumResponseBytes: Int = 2 * 1_024 * 1_024,
    maximumRedirectCount: Int = 6,
    requestTimeout: TimeInterval = 20,
    maximumRequestCount: Int = 24,
    maximumCumulativeResponseBytes: Int = 8 * 1_024 * 1_024,
    resolutionTimeout: TimeInterval = 60
  ) {
    self.maximumResponseBytes = max(1_024, maximumResponseBytes)
    self.maximumRedirectCount = max(0, maximumRedirectCount)
    self.requestTimeout = max(1, requestTimeout)
    self.maximumRequestCount = max(2, maximumRequestCount)
    self.maximumCumulativeResponseBytes = max(
      self.maximumResponseBytes,
      maximumCumulativeResponseBytes
    )
    self.resolutionTimeout = max(2, resolutionTimeout)
  }

  func resolve(
    _ rawValue: String,
    policy: IPadMediaURLResolutionPolicy = .userSubmitted,
    context: IPadMediaRequestContext? = nil
  ) async throws -> IPadResolvedMediaSource {
    try Task.checkCancellation()
    guard let submittedURL = Self.normalizedHTTPURL(rawValue) else {
      throw IPadMediaURLResolverError.invalidURL
    }
    if !Self.isURL(submittedURL, allowedBy: policy) {
      switch policy {
      case .publicDiscovered, .visibleBrowserDiscovered:
        throw IPadMediaURLResolverError.unsafeInitialURL
      case .userSubmitted, .submittedPageSameOrigin:
        throw IPadMediaURLResolverError.unsafeURL
      }
    }
    let effectivePolicy = try Self.initialRequestPolicy(
      for: submittedURL,
      parentPolicy: policy
    )

    let budget = IPadMediaResolutionBudget(
      maximumRequests: maximumRequestCount,
      maximumResponseBytes: maximumCumulativeResponseBytes,
      timeout: resolutionTimeout
    )
    let headPayload: IPadHTTPPayload?
    do {
      headPayload = try await request(
        submittedURL,
        method: "HEAD",
        budget: budget,
        policy: effectivePolicy,
        context: context
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch is IPadMediaURLResolverError {
      // A significant number of media origins do not implement HEAD. The
      // bounded GET below remains safe and provides both sniffing and HTML
      // discovery in that case. Some WAFs challenge HEAD while allowing the
      // browser-equivalent GET, so only the GET result is authoritative.
      headPayload = nil
    } catch {
      // A significant number of media origins do not implement HEAD. The
      // bounded GET below remains safe and provides both sniffing and HTML
      // discovery in that case.
      headPayload = nil
    }

    if let headPayload,
      (200...299).contains(headPayload.response.statusCode)
    {
      try Self.validateSuccessfulResponse(headPayload.response)
      let finalURL = try validatedResponseURL(
        headPayload.response,
        submittedURL: submittedURL,
        policy: effectivePolicy
      )
      let contentType = Self.normalizedContentType(headPayload.response)
      switch Self.mediaKind(url: finalURL, contentType: contentType, prefix: nil) {
      case .progressive:
        return IPadResolvedMediaSource(
          kind: .progressive,
          submittedURL: submittedURL,
          playbackURL: finalURL,
          mediaURL: finalURL,
          contentType: contentType,
          hlsPlaylist: nil,
          resolutionPolicy: effectivePolicy,
          requestContext: context
        )
      case .hls:
        let resolved = try await resolveHLS(
          at: finalURL,
          depth: 0,
          visited: [],
          budget: budget,
          policy: effectivePolicy,
          context: context
        )
        return IPadResolvedMediaSource(
          kind: .hls,
          submittedURL: submittedURL,
          playbackURL: finalURL,
          mediaURL: resolved.playlist.url,
          contentType: resolved.contentType ?? contentType,
          hlsPlaylist: resolved.playlist,
          resolutionPolicy: effectivePolicy,
          requestContext: context
        )
      case nil:
        break
      }
    }

    let payload = try await request(
      submittedURL,
      method: "GET",
      budget: budget,
      policy: effectivePolicy,
      context: context
    )
    try Self.validateSuccessfulResponse(payload.response)
    let finalURL = try validatedResponseURL(
      payload.response,
      submittedURL: submittedURL,
      policy: effectivePolicy
    )
    let contentType = Self.normalizedContentType(payload.response)
    let prefix = String(decoding: payload.data.prefix(512), as: UTF8.self)

    switch Self.mediaKind(url: finalURL, contentType: contentType, prefix: prefix) {
    case .hls:
      let resolved: IPadResolvedHLS
      if Self.isHLSPlaylist(payload.data) {
        resolved = try await resolveHLSPayload(
          payload,
          at: finalURL,
          depth: 0,
          visited: [],
          budget: budget,
          policy: effectivePolicy,
          context: context
        )
      } else {
        resolved = try await resolveHLS(
          at: finalURL,
          depth: 0,
          visited: [],
          budget: budget,
          policy: effectivePolicy,
          context: context
        )
      }
      return IPadResolvedMediaSource(
        kind: .hls,
        submittedURL: submittedURL,
        playbackURL: finalURL,
        mediaURL: resolved.playlist.url,
        contentType: resolved.contentType ?? contentType,
        hlsPlaylist: resolved.playlist,
        resolutionPolicy: effectivePolicy,
        requestContext: context
      )
    case .progressive:
      return IPadResolvedMediaSource(
        kind: .progressive,
        submittedURL: submittedURL,
        playbackURL: finalURL,
        mediaURL: finalURL,
        contentType: contentType,
        hlsPlaylist: nil,
        resolutionPolicy: effectivePolicy,
        requestContext: context
      )
    case nil:
      break
    }

    guard Self.isHTMLDocument(payload.data, contentType: contentType) else {
      throw IPadMediaURLResolverError.unsupportedContent
    }
    return try await resolveStaticMediaPages(
      initialPayload: payload,
      at: finalURL,
      submittedURL: submittedURL,
      budget: budget,
      policy: effectivePolicy,
      context: context
    )
  }

  /// Resolves static embed chains breadth-first. A page-level progressive URL
  /// is retained only as a fallback so an HLS stream exposed by a nested
  /// iframe/player wins even when the outer page advertises an MP4 preview.
  private func resolveStaticMediaPages(
    initialPayload: IPadHTTPPayload,
    at initialURL: URL,
    submittedURL: URL,
    budget: IPadMediaResolutionBudget,
    policy: IPadMediaURLResolutionPolicy,
    context: IPadMediaRequestContext?
  ) async throws -> IPadResolvedMediaSource {
    let initialPagePolicy = Self.initialDiscoveredPolicy(
      pageURL: initialURL,
      parentPolicy: policy
    )
    var pages = [
      IPadHTMLPage(
        payload: initialPayload,
        url: initialURL,
        depth: 0,
        policy: initialPagePolicy
      )
    ]
    var queuedPages: Set<URL> = [initialURL]
    var visitedPages = Set<URL>()
    var progressiveFallback: (url: URL, policy: IPadMediaURLResolutionPolicy)?
    var resolvedHLSChoices: [IPadResolvedMediaSource] = []
    var resolvedHLSKeys = Set<String>()
    var pendingInteractionURL: URL?
    var pageIndex = 0

    func preferredCollectedChoice() -> IPadResolvedMediaSource? {
      if let index = IPadBrowserMediaSourceSelector.preferredIndex(
        in: resolvedHLSChoices
      ) {
        return resolvedHLSChoices[index]
      }
      guard let progressiveFallback else { return nil }
      return Self.progressiveSource(
        submittedURL: submittedURL,
        mediaURL: progressiveFallback.url,
        policy: progressiveFallback.policy,
        context: context
      )
    }

    while pageIndex < pages.count {
      try Task.checkCancellation()
      let page = pages[pageIndex]
      pageIndex += 1
      guard visitedPages.insert(page.url).inserted else { continue }

      let html = String(decoding: page.payload.data, as: UTF8.self)
      let candidates = Self.discoverMediaCandidates(in: html, relativeTo: page.url)
      if progressiveFallback == nil {
        progressiveFallback =
          candidates.progressiveCandidates.lazy.compactMap {
            candidate -> (url: URL, policy: IPadMediaURLResolutionPolicy)? in
            let candidatePolicy = Self.discoveredPolicy(for: candidate, on: page)
            guard Self.isURL(candidate, allowedBy: candidatePolicy) else {
              return nil
            }
            return (candidate, candidatePolicy)
          }.first
      }

      for candidate in candidates.hlsCandidates.prefix(Self.maximumHLSCandidatesPerPage) {
        try Task.checkCancellation()
        let candidatePolicy = Self.discoveredPolicy(for: candidate, on: page)
        do {
          let resolved = try await resolveHLS(
            at: candidate,
            depth: 0,
            visited: [],
            budget: budget,
            policy: candidatePolicy,
            context: context
          )
          guard
            !Self.isDisallowedDiscoveredLocalURL(
              resolved.playlist.url,
              pageURL: page.url
            )
          else { continue }
          let source = IPadResolvedMediaSource(
            kind: .hls,
            submittedURL: submittedURL,
            playbackURL: candidate,
            mediaURL: resolved.playlist.url,
            contentType: resolved.contentType,
            hlsPlaylist: resolved.playlist,
            resolutionPolicy: candidatePolicy,
            requestContext: context
          )
          let key = IPadBrowserMediaSourceSelector.deduplicationKey(for: source)
          if resolvedHLSKeys.insert(key).inserted {
            resolvedHLSChoices.append(source)
          }
        } catch is CancellationError {
          throw CancellationError()
        } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
          pendingInteractionURL = pendingInteractionURL ?? challengedURL ?? candidate
          continue
        } catch IPadMediaURLResolverError.resolutionLimitExceeded {
          if let preferred = preferredCollectedChoice() { return preferred }
          if let pendingInteractionURL {
            throw IPadMediaURLResolverError.interactionRequired(pendingInteractionURL)
          }
          throw IPadMediaURLResolverError.resolutionLimitExceeded
        } catch {
          continue
        }
      }

      guard page.depth < Self.maximumPageDepth else { continue }
      for candidate in candidates.pageCandidates.prefix(Self.maximumNestedPagesPerPage) {
        try Task.checkCancellation()
        let candidatePolicy = Self.discoveredPolicy(for: candidate, on: page)
        guard !visitedPages.contains(candidate), queuedPages.insert(candidate).inserted else {
          continue
        }

        do {
          let nestedPayload = try await request(
            candidate,
            method: "GET",
            budget: budget,
            policy: candidatePolicy,
            context: context
          )
          try Self.validateSuccessfulResponse(nestedPayload.response)
          let finalURL = try validatedResponseURL(
            nestedPayload.response,
            submittedURL: candidate,
            policy: candidatePolicy
          )
          guard !Self.isDisallowedDiscoveredLocalURL(finalURL, pageURL: page.url) else {
            continue
          }

          let contentType = Self.normalizedContentType(nestedPayload.response)
          let prefix = String(decoding: nestedPayload.data.prefix(512), as: UTF8.self)
          switch Self.mediaKind(url: finalURL, contentType: contentType, prefix: prefix) {
          case .hls:
            let resolved: IPadResolvedHLS
            if Self.isHLSPlaylist(nestedPayload.data) {
              resolved = try await resolveHLSPayload(
                nestedPayload,
                at: finalURL,
                depth: 0,
                visited: [],
                budget: budget,
                policy: candidatePolicy,
                context: context
              )
            } else {
              resolved = try await resolveHLS(
                at: finalURL,
                depth: 0,
                visited: [],
                budget: budget,
                policy: candidatePolicy,
                context: context
              )
            }
            guard
              !Self.isDisallowedDiscoveredLocalURL(
                resolved.playlist.url,
                pageURL: page.url
              )
            else { continue }
            let source = IPadResolvedMediaSource(
              kind: .hls,
              submittedURL: submittedURL,
              playbackURL: finalURL,
              mediaURL: resolved.playlist.url,
              contentType: resolved.contentType ?? contentType,
              hlsPlaylist: resolved.playlist,
              resolutionPolicy: candidatePolicy,
              requestContext: context
            )
            let key = IPadBrowserMediaSourceSelector.deduplicationKey(for: source)
            if resolvedHLSKeys.insert(key).inserted {
              resolvedHLSChoices.append(source)
            }
          case .progressive:
            if progressiveFallback == nil {
              progressiveFallback = (finalURL, candidatePolicy)
            }
          case nil:
            guard Self.isHTMLDocument(nestedPayload.data, contentType: contentType) else {
              continue
            }
            queuedPages.insert(finalURL)
            pages.append(
              IPadHTMLPage(
                payload: nestedPayload,
                url: finalURL,
                depth: page.depth + 1,
                policy: candidatePolicy
              )
            )
          }
        } catch is CancellationError {
          throw CancellationError()
        } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
          pendingInteractionURL = pendingInteractionURL ?? challengedURL ?? candidate
          continue
        } catch IPadMediaURLResolverError.resolutionLimitExceeded {
          if let preferred = preferredCollectedChoice() { return preferred }
          if let pendingInteractionURL {
            throw IPadMediaURLResolverError.interactionRequired(pendingInteractionURL)
          }
          throw IPadMediaURLResolverError.resolutionLimitExceeded
        } catch {
          continue
        }
      }
    }

    if let preferred = preferredCollectedChoice() { return preferred }
    if let pendingInteractionURL {
      throw IPadMediaURLResolverError.interactionRequired(pendingInteractionURL)
    }
    throw IPadMediaURLResolverError.unsupportedContent
  }

  private static func progressiveSource(
    submittedURL: URL,
    mediaURL: URL,
    policy: IPadMediaURLResolutionPolicy,
    context: IPadMediaRequestContext?
  ) -> IPadResolvedMediaSource {
    IPadResolvedMediaSource(
      kind: .progressive,
      submittedURL: submittedURL,
      playbackURL: mediaURL,
      mediaURL: mediaURL,
      contentType: nil,
      hlsPlaylist: nil,
      resolutionPolicy: policy,
      requestContext: context
    )
  }

  private static func initialDiscoveredPolicy(
    pageURL: URL,
    parentPolicy: IPadMediaURLResolutionPolicy
  ) -> IPadMediaURLResolutionPolicy {
    switch parentPolicy {
    case .publicDiscovered:
      return .publicDiscovered
    case .visibleBrowserDiscovered(let approvedOrigin):
      return isSameOrigin(pageURL, approvedOrigin)
        ? .visibleBrowserDiscovered(approvedOrigin) : .publicDiscovered
    case .submittedPageSameOrigin(let origin):
      return .submittedPageSameOrigin(origin)
    case .userSubmitted:
      guard let host = pageURL.host, isPrivateOrLocalHost(host),
        let origin = originURL(for: pageURL)
      else { return .publicDiscovered }
      return .submittedPageSameOrigin(origin)
    }
  }

  private static func initialRequestPolicy(
    for submittedURL: URL,
    parentPolicy: IPadMediaURLResolutionPolicy
  ) throws -> IPadMediaURLResolutionPolicy {
    switch parentPolicy {
    case .publicDiscovered, .visibleBrowserDiscovered,
      .submittedPageSameOrigin:
      return parentPolicy
    case .userSubmitted:
      if isPublicHTTPSURL(submittedURL) {
        return .publicDiscovered
      }
      guard let host = submittedURL.host?.lowercased(),
        let origin = originURL(for: submittedURL)
      else { throw IPadMediaURLResolverError.unsafeURL }
      if isPrivateOrLocalHost(host) {
        return .submittedPageSameOrigin(origin)
      }
      throw IPadMediaURLResolverError.unsafeURL
    }
  }

  private static func discoveredPolicy(
    for candidate: URL,
    on page: IPadHTMLPage
  ) -> IPadMediaURLResolutionPolicy {
    switch page.policy {
    case .publicDiscovered:
      return .publicDiscovered
    case .visibleBrowserDiscovered(let approvedOrigin):
      return isSameOrigin(candidate, approvedOrigin)
        ? .visibleBrowserDiscovered(approvedOrigin) : .publicDiscovered
    case .submittedPageSameOrigin(let origin):
      return isSameOrigin(candidate, origin)
        ? .submittedPageSameOrigin(origin) : .publicDiscovered
    case .userSubmitted:
      return .publicDiscovered
    }
  }

  static func isURL(
    _ candidate: URL,
    allowedBy policy: IPadMediaURLResolutionPolicy
  ) -> Bool {
    switch policy {
    case .userSubmitted:
      return true
    case .publicDiscovered:
      return isPublicHTTPSURL(candidate)
    case .visibleBrowserDiscovered(let approvedOrigin):
      return isVisibleBrowserHTTPSURL(
        candidate,
        approvedOrigin: approvedOrigin
      )
    case .submittedPageSameOrigin(let origin):
      return isSameOrigin(candidate, origin)
    }
  }

  private static func originURL(for url: URL) -> URL? {
    guard var components = URLComponents(url: url, resolvingAgainstBaseURL: true),
      let scheme = components.scheme?.lowercased(),
      scheme == "http" || scheme == "https",
      components.host?.isEmpty == false
    else { return nil }
    components.scheme = scheme
    components.user = nil
    components.password = nil
    components.path = "/"
    components.query = nil
    components.fragment = nil
    return components.url
  }

  static func normalizedHTTPURL(_ raw: String) -> URL? {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
      trimmed.rangeOfCharacter(from: .controlCharacters) == nil
    else { return nil }

    let value: String
    if trimmed.range(of: "://") == nil {
      value = "https://\(trimmed)"
    } else {
      value = trimmed
    }

    guard var components = URLComponents(string: value),
      let rawScheme = components.scheme?.lowercased(),
      rawScheme == "http" || rawScheme == "https",
      components.user == nil,
      components.password == nil,
      let host = components.host,
      !host.isEmpty
    else { return nil }

    components.scheme = rawScheme
    components.fragment = nil
    return components.url
  }

  private func request(
    _ url: URL,
    method: String,
    budget: IPadMediaResolutionBudget,
    policy: IPadMediaURLResolutionPolicy,
    context: IPadMediaRequestContext?
  ) async throws -> IPadHTTPPayload {
    try Task.checkCancellation()
    if !Self.isURL(url, allowedBy: policy) {
      throw IPadMediaURLResolverError.unsafeURL
    }
    let remainingTimeout = try budget.consumeRequest()
    var request = URLRequest(url: url)
    request.httpMethod = method
    request.timeoutInterval = min(requestTimeout, remainingTimeout)
    request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    request.setValue(
      "application/vnd.apple.mpegurl, application/x-mpegurl, video/*, text/html;q=0.8, */*;q=0.2",
      forHTTPHeaderField: "Accept"
    )
    context?.applying(to: &request)
    let operation = IPadBoundedHTTPRequest(
      maximumResponseBytes: maximumResponseBytes,
      maximumRedirectCount: maximumRedirectCount,
      timeout: min(requestTimeout, remainingTimeout),
      returnsAfterVideoResponse: method == "GET",
      resolutionPolicy: policy,
      requestContext: context
    )
    do {
      let payload = try await operation.start(request)
      try Self.validateNoInteractionChallenge(payload.response)
      await context?.updateCookies(from: payload.response)
      try budget.consumeResponseBytes(payload.data.count)
      return payload
    } catch let error as IPadMediaURLResolverError {
      throw error
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      throw IPadMediaURLResolverError.requestFailed(error.localizedDescription)
    }
  }

  private func resolveHLS(
    at url: URL,
    depth: Int,
    visited: Set<URL>,
    budget: IPadMediaResolutionBudget,
    policy: IPadMediaURLResolutionPolicy,
    context: IPadMediaRequestContext?
  ) async throws -> IPadResolvedHLS {
    let payload = try await request(
      url,
      method: "GET",
      budget: budget,
      policy: policy,
      context: context
    )
    try Self.validateSuccessfulResponse(payload.response)
    let finalURL = try validatedResponseURL(
      payload.response,
      submittedURL: url,
      policy: policy
    )
    guard Self.isHLSPlaylist(payload.data) else {
      throw IPadMediaURLResolverError.invalidPlaylist("#EXTM3Uがありません")
    }
    return try await resolveHLSPayload(
      payload,
      at: finalURL,
      depth: depth,
      visited: visited,
      budget: budget,
      policy: policy,
      context: context
    )
  }

  private func resolveHLSPayload(
    _ payload: IPadHTTPPayload,
    at finalURL: URL,
    depth: Int,
    visited: Set<URL>,
    budget: IPadMediaResolutionBudget,
    policy: IPadMediaURLResolutionPolicy,
    context: IPadMediaRequestContext?
  ) async throws -> IPadResolvedHLS {
    try Task.checkCancellation()
    guard depth <= 4, !visited.contains(finalURL) else {
      throw IPadMediaURLResolverError.invalidPlaylist("master playlistが循環しています")
    }
    guard let text = String(data: payload.data, encoding: .utf8) else {
      throw IPadMediaURLResolverError.invalidPlaylist("UTF-8ではありません")
    }
    try Self.rejectProtectedPlaylist(text)

    let variants = try Self.parseMasterVariants(text, relativeTo: finalURL)
    if !variants.isEmpty {
      let nextVisited = visited.union([finalURL])
      var lastError: Error?
      var pendingInteractionURL: URL?
      for variant in try Self.restorationVariantOrder(variants).prefix(8) {
        do {
          return try await resolveHLS(
            at: variant.url,
            depth: depth + 1,
            visited: nextVisited,
            budget: budget,
            policy: policy,
            context: context
          )
        } catch is CancellationError {
          throw CancellationError()
        } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
          // One CDN rendition can be challenged while another public
          // rendition remains playable. Try the bounded variant set first.
          pendingInteractionURL =
            pendingInteractionURL ?? challengedURL ?? variant.url
          lastError = IPadMediaURLResolverError.interactionRequired(
            pendingInteractionURL
          )
        } catch {
          lastError = error
        }
      }
      if let pendingInteractionURL {
        throw IPadMediaURLResolverError.interactionRequired(
          pendingInteractionURL
        )
      }
      throw lastError
        ?? IPadMediaURLResolverError.invalidPlaylist("再生可能なvariantがありません")
    }

    let referencedPlaylists = try Self.parseUnlabeledPlaylistReferences(
      text,
      relativeTo: finalURL
    )
    if !referencedPlaylists.isEmpty {
      let nextVisited = visited.union([finalURL])
      var lastError: Error?
      var pendingInteractionURL: URL?
      for referencedPlaylist in referencedPlaylists.prefix(8) {
        do {
          return try await resolveHLS(
            at: referencedPlaylist,
            depth: depth + 1,
            visited: nextVisited,
            budget: budget,
            policy: policy,
            context: context
          )
        } catch is CancellationError {
          throw CancellationError()
        } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
          pendingInteractionURL =
            pendingInteractionURL ?? challengedURL ?? referencedPlaylist
          lastError = IPadMediaURLResolverError.interactionRequired(
            pendingInteractionURL
          )
        } catch {
          lastError = error
        }
      }
      if let pendingInteractionURL {
        throw IPadMediaURLResolverError.interactionRequired(
          pendingInteractionURL
        )
      }
      throw lastError
        ?? IPadMediaURLResolverError.invalidPlaylist("参照先playlistを再生できません")
    }

    let playlist = try Self.parseMediaPlaylist(text, url: finalURL)
    return IPadResolvedHLS(
      playlist: playlist,
      contentType: Self.normalizedContentType(payload.response)
    )
  }

  private func validatedResponseURL(
    _ response: HTTPURLResponse,
    submittedURL: URL,
    policy: IPadMediaURLResolutionPolicy
  ) throws -> URL {
    guard let finalURL = response.url,
      let safeURL = Self.sanitizedAbsoluteHTTPURL(finalURL)
    else { throw IPadMediaURLResolverError.unsafeURL }
    if submittedURL.scheme?.lowercased() == "https",
      safeURL.scheme?.lowercased() == "http"
    {
      throw IPadMediaURLResolverError.insecureRedirect
    }
    if !Self.isURL(safeURL, allowedBy: policy) {
      throw IPadMediaURLResolverError.unsafeURL
    }
    return safeURL
  }

  private static func mediaKind(
    url: URL,
    contentType: String?,
    prefix: String?
  ) -> IPadResolvedMediaSource.Kind? {
    let pathExtension = url.pathExtension.lowercased()
    let hlsContentTypes = [
      "application/vnd.apple.mpegurl",
      "application/x-mpegurl",
      "application/mpegurl",
      "audio/mpegurl",
      "audio/x-mpegurl",
    ]
    if pathExtension == "m3u8"
      || hlsContentTypes.contains(contentType ?? "")
      || prefix?.replacingOccurrences(of: "\u{feff}", with: "")
        .trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("#EXTM3U") == true
    {
      return .hls
    }
    if ["mp4", "mov", "m4v"].contains(pathExtension)
      || contentType?.hasPrefix("video/") == true
    {
      return .progressive
    }
    return nil
  }

  private static func normalizedContentType(_ response: HTTPURLResponse) -> String? {
    guard let mimeType = response.mimeType?.lowercased(), !mimeType.isEmpty else {
      return nil
    }
    return mimeType
  }

  static func isInteractionChallengeURL(_ url: URL?) -> Bool {
    guard let url else { return false }
    let host = url.host?.lowercased() ?? ""
    return host == "challenges.cloudflare.com"
      || host.hasSuffix(".challenges.cloudflare.com")
      || url.path.lowercased().contains("/cdn-cgi/challenge-platform/")
  }

  static func interactionChallengeError(
    response: HTTPURLResponse,
    destinationURL: URL? = nil
  ) -> IPadMediaURLResolverError? {
    if response.value(forHTTPHeaderField: "cf-mitigated")?.lowercased()
      == "challenge"
      || isInteractionChallengeURL(response.url)
      || isInteractionChallengeURL(destinationURL)
    {
      return .interactionRequired(destinationURL ?? response.url)
    }
    return nil
  }

  fileprivate static func validateNoInteractionChallenge(
    _ response: HTTPURLResponse
  ) throws {
    if let error = interactionChallengeError(response: response) {
      throw error
    }
  }

  fileprivate static func validateSuccessfulResponse(_ response: HTTPURLResponse) throws {
    try validateNoInteractionChallenge(response)
    guard (200...299).contains(response.statusCode) else {
      throw IPadMediaURLResolverError.invalidHTTPStatus(response.statusCode)
    }
  }

  private static func isHLSPlaylist(_ data: Data) -> Bool {
    let prefix = String(decoding: data.prefix(1_024), as: UTF8.self)
      .replacingOccurrences(of: "\u{feff}", with: "")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    return prefix.hasPrefix("#EXTM3U")
  }

  private static func isHTMLDocument(_ data: Data, contentType: String?) -> Bool {
    if contentType == "text/html" || contentType == "application/xhtml+xml" {
      return true
    }
    guard contentType == nil || contentType == "application/octet-stream" else {
      return false
    }
    let prefix = String(decoding: data.prefix(1_024), as: UTF8.self)
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
    return prefix.hasPrefix("<!doctype html") || prefix.hasPrefix("<html")
  }

  private static func discoverMediaCandidates(
    in html: String,
    relativeTo baseURL: URL
  ) -> IPadHTMLMediaCandidates {
    var rawMediaCandidates: [String] = []
    var rawPageCandidates: [String] = []

    for tag in regexMatches("<meta\\b[^>]*>", in: html) {
      let lowercased = tag.lowercased()
      if lowercased.contains("og:video") {
        let values = attributeValues(named: "content", in: tag)
        rawMediaCandidates.append(contentsOf: values)
        rawPageCandidates.append(contentsOf: values)
      }
    }
    for mediaTag in ["video", "source"] {
      for tag in regexMatches("<\(mediaTag)\\b[^>]*>", in: html) {
        let values = attributeValues(named: "src", in: tag)
        rawMediaCandidates.append(contentsOf: values)
        rawPageCandidates.append(contentsOf: values)
      }
    }
    for embedTag in ["iframe", "embed"] {
      for tag in regexMatches("<\(embedTag)\\b[^>]*>", in: html) {
        for attribute in ["src", "data-src", "data-lazy-src", "data-url"] {
          let values = attributeValues(named: attribute, in: tag)
          rawMediaCandidates.append(contentsOf: values)
          rawPageCandidates.append(contentsOf: values)
        }
      }
    }
    for tag in regexMatches("<object\\b[^>]*>", in: html) {
      for attribute in ["data", "data-src", "data-url"] {
        let values = attributeValues(named: attribute, in: tag)
        rawMediaCandidates.append(contentsOf: values)
        rawPageCandidates.append(contentsOf: values)
      }
    }
    for tag in regexMatches("<a\\b[^>]*>", in: html) {
      let lowercased = tag.lowercased()
      guard ["player", "embed", "video"].contains(where: lowercased.contains) else {
        continue
      }
      rawPageCandidates.append(contentsOf: attributeValues(named: "href", in: tag))
    }
    rawMediaCandidates.append(
      contentsOf: regexCaptureMatches(
        "[\\\"']([^\\\"'<>\\s]+\\.(?:m3u8|mp4|mov|m4v)(?:\\?[^\\\"'<>\\s]*)?)[\\\"']",
        in: html
      )
    )
    rawMediaCandidates.append(
      contentsOf: regexCaptureMatches(
        "((?:(?:https?:)?//|/|\\.\\.?/)[^\\\"'<>\\s]+\\.(?:m3u8|mp4|mov|m4v)(?:[^\\\"'<>\\s]*)?)",
        in: html
      )
    )
    let playerValues = regexCaptureMatches(
      "[\\\"']?(?:iframe|embed|player)(?:[_-]?(?:url|src))?[\\\"']?\\s*[:=]\\s*[\\\"']([^\\\"'<>\\s]+)[\\\"']",
      in: html
    )
    rawMediaCandidates.append(contentsOf: playerValues)
    rawPageCandidates.append(contentsOf: playerValues)

    var hlsCandidates: [URL] = []
    var progressiveCandidates: [URL] = []
    var pageCandidates: [URL] = []
    var seenMedia = Set<URL>()
    for rawCandidate in rawMediaCandidates {
      guard let safeURL = discoveredHTTPURL(rawCandidate, relativeTo: baseURL),
        seenMedia.insert(safeURL).inserted
      else { continue }
      switch mediaKind(url: safeURL, contentType: nil, prefix: nil) {
      case .hls:
        hlsCandidates.append(safeURL)
      case .progressive:
        progressiveCandidates.append(safeURL)
      case nil:
        break
      }
    }
    var seenPages = Set<URL>()
    for rawCandidate in rawPageCandidates {
      guard let safeURL = discoveredHTTPURL(rawCandidate, relativeTo: baseURL),
        mediaKind(url: safeURL, contentType: nil, prefix: nil) == nil,
        seenPages.insert(safeURL).inserted
      else { continue }
      pageCandidates.append(safeURL)
    }
    return IPadHTMLMediaCandidates(
      hlsCandidates: hlsCandidates,
      progressiveCandidates: progressiveCandidates,
      pageCandidates: pageCandidates
    )
  }

  private static func discoveredHTTPURL(
    _ reference: String,
    relativeTo baseURL: URL
  ) -> URL? {
    guard let safeURL = resolvedHTTPURL(reference, relativeTo: baseURL),
      !isDisallowedDiscoveredLocalURL(safeURL, pageURL: baseURL)
    else { return nil }
    return safeURL
  }

  private static func decodeHTMLURL(_ raw: String) -> String {
    // JSON embedded in HTML commonly escapes a slash as "\/".
    raw.replacingOccurrences(of: "&amp;", with: "&")
      .replacingOccurrences(of: "&#38;", with: "&")
      .replacingOccurrences(of: "&#x26;", with: "&")
      .replacingOccurrences(of: "\\/", with: "/")
      .replacingOccurrences(of: "\\u002F", with: "/")
      .replacingOccurrences(of: "\\u002f", with: "/")
      .replacingOccurrences(of: "\\u003A", with: ":")
      .replacingOccurrences(of: "\\u003a", with: ":")
      .replacingOccurrences(of: "\\u0026", with: "&")
      .replacingOccurrences(of: "&quot;", with: "\"")
  }

  static func isDisallowedDiscoveredLocalURL(
    _ candidate: URL,
    pageURL: URL
  ) -> Bool {
    guard !isSameOrigin(candidate, pageURL) else { return false }
    guard let host = candidate.host?.lowercased() else { return true }
    return isPrivateOrLocalHost(host)
  }

  static func isPublicHTTPSURL(_ candidate: URL) -> Bool {
    guard
      let components = URLComponents(
        url: candidate,
        resolvingAgainstBaseURL: true
      ),
      components.scheme?.lowercased() == "https",
      components.user == nil, components.password == nil,
      let host = components.host?.lowercased(),
      !host.isEmpty
    else { return false }
    return !isPrivateOrLocalHost(host) && resolvedAddressesArePublic(host)
  }

  private static func isVisibleBrowserHTTPSURL(
    _ candidate: URL,
    approvedOrigin: URL
  ) -> Bool {
    guard isCanonicalHTTPSOrigin(approvedOrigin) else { return false }
    guard
      let components = URLComponents(
        url: candidate,
        resolvingAgainstBaseURL: true
      ),
      components.scheme?.lowercased() == "https",
      components.user == nil, components.password == nil,
      let host = components.host?.lowercased(), !host.isEmpty,
      !isPrivateOrLocalHost(host)
    else { return false }
    // The policy is issued only after WebKit proves a visible media source.
    // Permit hostname-only VPN fake-IP translation for its HTTPS CDN children
    // as well as the original host; private answers and IP literals still fail.
    return resolvedAddressesArePublic(
      host,
      allowsHostnameVPNBenchmarkTranslation: true
    )
  }

  private static func isCanonicalHTTPSOrigin(_ candidate: URL) -> Bool {
    guard
      let components = URLComponents(
        url: candidate,
        resolvingAgainstBaseURL: true
      ),
      components.scheme?.lowercased() == "https",
      components.user == nil, components.password == nil,
      components.host?.isEmpty == false,
      components.query == nil, components.fragment == nil,
      components.path.isEmpty || components.path == "/"
    else { return false }
    return true
  }

  private static func isPrivateOrLocalHost(_ rawHost: String) -> Bool {
    let host = rawHost.lowercased().trimmingCharacters(
      in: CharacterSet(charactersIn: ".[]")
    )
    guard !host.isEmpty else { return true }
    if host == "localhost" || host.hasSuffix(".localhost")
      || host.hasSuffix(".local")
    {
      return true
    }
    if host.allSatisfy({ $0.isNumber || $0 == "." })
      || host.hasPrefix("0x")
    {
      // Block legacy single-integer/hex IPv4 forms whose textual value can be
      // normalized to loopback or another private address by the network stack.
      let components = host.split(separator: ".", omittingEmptySubsequences: false)
      guard components.count == 4,
        components.allSatisfy({
          !$0.isEmpty && $0.allSatisfy(\.isNumber)
            && ($0.count == 1 || !$0.hasPrefix("0"))
        })
      else { return true }
    }

    let rawIPv4Parts = host.split(separator: ".", omittingEmptySubsequences: false)
    if rawIPv4Parts.count == 4,
      rawIPv4Parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) })
    {
      guard
        rawIPv4Parts.allSatisfy({
          $0.count == 1 || !$0.hasPrefix("0")
        })
      else { return true }
      let ipv4Parts = rawIPv4Parts.compactMap { UInt8($0) }
      guard ipv4Parts.count == 4 else { return true }
      let first = ipv4Parts[0]
      let second = ipv4Parts[1]
      return first == 0 || first == 10 || first == 127
        || (first == 169 && second == 254)
        || (first == 172 && (16...31).contains(second))
        || (first == 192 && second == 168)
    }

    let normalizedIPv6 = host.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    return normalizedIPv6 == "::1" || normalizedIPv6 == "::"
      || normalizedIPv6.hasPrefix("::ffff:")
      || normalizedIPv6.hasPrefix("fc") || normalizedIPv6.hasPrefix("fd")
      || normalizedIPv6.hasPrefix("fe8") || normalizedIPv6.hasPrefix("fe9")
      || normalizedIPv6.hasPrefix("fea") || normalizedIPv6.hasPrefix("feb")
  }

  /// Resolves every address immediately before a public-discovered request is
  /// started. Requiring every result to be globally routable closes the common
  /// hostname-to-LAN and legacy IPv4 SSRF paths; redirects are checked again by
  /// their URLSession delegates before they are followed.
  private static func resolvedAddressesArePublic(
    _ rawHost: String,
    allowsHostnameVPNBenchmarkTranslation: Bool = false
  ) -> Bool {
    let host = rawHost.trimmingCharacters(
      in: CharacterSet(charactersIn: ".[]")
    )
    guard !host.isEmpty else { return false }
    let allowsVPNBenchmarkTranslation =
      allowsHostnameVPNBenchmarkTranslation && !isIPAddressLiteral(host)

    var hints = addrinfo()
    hints.ai_flags = AI_ADDRCONFIG
    hints.ai_family = AF_UNSPEC
    hints.ai_socktype = SOCK_STREAM
    hints.ai_protocol = IPPROTO_TCP

    var result: UnsafeMutablePointer<addrinfo>?
    guard getaddrinfo(host, nil, &hints, &result) == 0, let first = result else {
      return false
    }
    defer { freeaddrinfo(first) }

    var foundAddress = false
    var cursor: UnsafeMutablePointer<addrinfo>? = first
    while let current = cursor {
      defer { cursor = current.pointee.ai_next }
      guard let address = current.pointee.ai_addr else { continue }
      switch Int32(current.pointee.ai_family) {
      case AF_INET:
        let ipv4 = address.withMemoryRebound(
          to: sockaddr_in.self,
          capacity: 1
        ) { UInt32(bigEndian: $0.pointee.sin_addr.s_addr) }
        foundAddress = true
        guard
          isPublicIPv4(ipv4)
            || (allowsVPNBenchmarkTranslation && isVPNBenchmarkIPv4(ipv4))
        else { return false }
      case AF_INET6:
        var ipv6 = address.withMemoryRebound(
          to: sockaddr_in6.self,
          capacity: 1
        ) { $0.pointee.sin6_addr }
        let bytes = withUnsafeBytes(of: &ipv6) { Array($0) }
        foundAddress = true
        guard isPublicIPv6(bytes) else { return false }
      default:
        continue
      }
    }
    return foundAddress
  }

  private static func isIPAddressLiteral(_ host: String) -> Bool {
    var ipv4 = in_addr()
    if host.withCString({ inet_pton(AF_INET, $0, &ipv4) }) == 1 {
      return true
    }
    var ipv6 = in6_addr()
    return host.withCString({ inet_pton(AF_INET6, $0, &ipv6) }) == 1
  }

  private static func isVPNBenchmarkIPv4(_ address: UInt32) -> Bool {
    let first = UInt8((address >> 24) & 0xff)
    let second = UInt8((address >> 16) & 0xff)
    return first == 198 && (second == 18 || second == 19)
  }

  private static func isPublicIPv4(_ address: UInt32) -> Bool {
    let first = UInt8((address >> 24) & 0xff)
    let second = UInt8((address >> 16) & 0xff)
    let third = UInt8((address >> 8) & 0xff)
    if first == 0 || first == 10 || first == 127 || first >= 224 { return false }
    if first == 100, (64...127).contains(second) { return false }
    if first == 169, second == 254 { return false }
    if first == 172, (16...31).contains(second) { return false }
    if first == 192 {
      if second == 0, third == 0 || third == 2 { return false }
      if second == 88, third == 99 { return false }
      if second == 168 { return false }
    }
    if first == 198, second == 18 || second == 19 { return false }
    if first == 198, second == 51, third == 100 { return false }
    if first == 203, second == 0, third == 113 { return false }
    return true
  }

  private static func isPublicIPv6(_ bytes: [UInt8]) -> Bool {
    guard bytes.count == 16 else { return false }

    let isIPv4Mapped =
      bytes.prefix(10).allSatisfy { $0 == 0 }
      && bytes[10] == 0xff && bytes[11] == 0xff
    let isNAT64 =
      bytes[0] == 0x00 && bytes[1] == 0x64
      && bytes[2] == 0xff && bytes[3] == 0x9b
      && bytes[4..<12].allSatisfy { $0 == 0 }
    if isIPv4Mapped || isNAT64 {
      let ipv4 =
        UInt32(bytes[12]) << 24 | UInt32(bytes[13]) << 16
        | UInt32(bytes[14]) << 8 | UInt32(bytes[15])
      return isPublicIPv4(ipv4)
    }

    // Public global-unicast IPv6 is 2000::/3. Exclude documentation,
    // benchmarking and transition ranges that are not direct public targets.
    guard bytes[0] & 0xe0 == 0x20 else { return false }
    if bytes[0] == 0x20, bytes[1] == 0x01 {
      if bytes[2] == 0x0d, bytes[3] == 0xb8 { return false }
      if bytes[2] == 0x00, bytes[3] == 0x00 { return false }
      if bytes[2] == 0x00, bytes[3] == 0x02 { return false }
    }
    return true
  }

  private static func isSameOrigin(_ lhs: URL, _ rhs: URL) -> Bool {
    guard lhs.scheme?.lowercased() == rhs.scheme?.lowercased(),
      lhs.host?.lowercased() == rhs.host?.lowercased()
    else { return false }
    return effectivePort(lhs) == effectivePort(rhs)
  }

  private static func effectivePort(_ url: URL) -> Int? {
    if let port = url.port { return port }
    switch url.scheme?.lowercased() {
    case "http": return 80
    case "https": return 443
    default: return nil
    }
  }

  private static func attributeValues(named name: String, in tag: String) -> [String] {
    regexCaptureMatches(
      "\\b\(NSRegularExpression.escapedPattern(for: name))\\s*=\\s*[\\\"']([^\\\"']+)[\\\"']",
      in: tag
    )
  }

  private static func regexMatches(_ pattern: String, in source: String) -> [String] {
    guard
      let expression = try? NSRegularExpression(
        pattern: pattern,
        options: [.caseInsensitive]
      )
    else { return [] }
    let range = NSRange(source.startIndex..<source.endIndex, in: source)
    return expression.matches(in: source, range: range).compactMap { match in
      guard let swiftRange = Range(match.range, in: source) else { return nil }
      return String(source[swiftRange])
    }
  }

  private static func regexCaptureMatches(_ pattern: String, in source: String) -> [String] {
    guard
      let expression = try? NSRegularExpression(
        pattern: pattern,
        options: [.caseInsensitive]
      )
    else { return [] }
    let range = NSRange(source.startIndex..<source.endIndex, in: source)
    return expression.matches(in: source, range: range).compactMap { match in
      guard match.numberOfRanges > 1,
        let swiftRange = Range(match.range(at: 1), in: source)
      else { return nil }
      return String(source[swiftRange])
    }
  }

  private static func rejectProtectedPlaylist(_ text: String) throws {
    for rawLine in text.components(separatedBy: .newlines) {
      let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
      let uppercased = line.uppercased()
      if uppercased.hasPrefix("#EXT-X-SESSION-KEY:") {
        throw IPadMediaURLResolverError.encryptedPlaylist
      }
      if uppercased.hasPrefix("#EXT-X-KEY:") {
        let attributes = parseAttributeList(String(line.dropFirst("#EXT-X-KEY:".count)))
        if attributes["METHOD"]?.uppercased() != "NONE" {
          throw IPadMediaURLResolverError.encryptedPlaylist
        }
      }
      if uppercased.contains("SAMPLE-AES") || uppercased.contains("KEYFORMAT=") {
        throw IPadMediaURLResolverError.encryptedPlaylist
      }
    }
  }

  private static func parseMasterVariants(
    _ text: String,
    relativeTo baseURL: URL
  ) throws -> [IPadHLSVariant] {
    let lines = normalizedPlaylistLines(text)
    var variants: [IPadHLSVariant] = []
    var pendingVariant: (bandwidth: Int64, resolution: IPadHLSResolution?)?

    for line in lines {
      if line.uppercased().hasPrefix("#EXT-X-STREAM-INF:") {
        let attributes = parseAttributeList(
          String(line.dropFirst("#EXT-X-STREAM-INF:".count))
        )
        let bandwidth =
          Int64(attributes["AVERAGE-BANDWIDTH"] ?? "")
          ?? Int64(attributes["BANDWIDTH"] ?? "")
          ?? 0
        pendingVariant = (
          bandwidth: max(0, bandwidth),
          resolution: parseResolution(attributes["RESOLUTION"])
        )
        continue
      }
      if let pending = pendingVariant, !line.hasPrefix("#") {
        guard let url = resolvedHTTPURL(line, relativeTo: baseURL) else {
          throw IPadMediaURLResolverError.invalidPlaylist("variant URLが不正です")
        }
        variants.append(
          IPadHLSVariant(
            url: url,
            bandwidth: pending.bandwidth,
            resolution: pending.resolution
          )
        )
        pendingVariant = nil
      }
    }
    return variants
  }

  private static func parseUnlabeledPlaylistReferences(
    _ text: String,
    relativeTo baseURL: URL
  ) throws -> [URL] {
    let lines = normalizedPlaylistLines(text)
    guard lines.first?.hasPrefix("#EXTM3U") == true,
      !lines.contains(where: { $0.uppercased().hasPrefix("#EXTINF:") })
    else { return [] }
    let references = lines.filter { !$0.hasPrefix("#") }
    guard !references.isEmpty,
      references.allSatisfy({
        $0.lowercased().contains(".m3u8")
          || $0.lowercased().hasSuffix(".m3u")
      })
    else { return [] }

    var urls: [URL] = []
    var seen = Set<URL>()
    for reference in references {
      guard let url = resolvedHTTPURL(reference, relativeTo: baseURL) else {
        throw IPadMediaURLResolverError.invalidPlaylist(
          "参照先playlist URLが不正です"
        )
      }
      if seen.insert(url).inserted { urls.append(url) }
    }
    return urls
  }

  private static func parseResolution(_ source: String?) -> IPadHLSResolution? {
    guard let source else { return nil }
    let components = source.lowercased().split(
      separator: "x",
      omittingEmptySubsequences: false
    )
    guard components.count == 2,
      let width = Int(components[0]),
      let height = Int(components[1]),
      width > 0, height > 0
    else { return nil }
    return IPadHLSResolution(width: width, height: height)
  }

  /// BasicVSR++ retains a full clip of decoded frames. Prefer the best known
  /// rendition within the iPad processing budget, never a declared 4K stream.
  /// A master with no RESOLUTION metadata falls back from its lowest bitrate;
  /// the decoded-dimension guard in the worker remains authoritative.
  private static func restorationVariantOrder(
    _ variants: [IPadHLSVariant]
  ) throws -> [IPadHLSVariant] {
    let bounded = variants.filter { variant in
      guard let resolution = variant.resolution else { return false }
      return IPadRestorationMediaLimits.accepts(
        width: resolution.width,
        height: resolution.height
      )
    }.sorted { lhs, rhs in
      guard let left = lhs.resolution, let right = rhs.resolution else {
        return lhs.bandwidth > rhs.bandwidth
      }
      if left.pixelCount != right.pixelCount {
        return left.pixelCount > right.pixelCount
      }
      return lhs.bandwidth > rhs.bandwidth
    }
    let unknown = variants.filter { $0.resolution == nil }.sorted {
      $0.bandwidth < $1.bandwidth
    }
    let ordered = bounded + unknown
    guard !ordered.isEmpty else {
      throw IPadMediaURLResolverError.invalidPlaylist(
        "iPad復元の上限（長辺1920・短辺1080）以下のvariantがありません"
      )
    }
    return ordered
  }

  private static func parseMediaPlaylist(
    _ text: String,
    url: URL
  ) throws -> IPadHLSMediaPlaylist {
    let maximumTimelineSeconds = 9_000_000_000.0
    let lines = normalizedPlaylistLines(text)
    guard lines.first?.hasPrefix("#EXTM3U") == true else {
      throw IPadMediaURLResolverError.invalidPlaylist("先頭タグがありません")
    }

    var nextSequence: Int64 = 0
    var pendingDuration: TimeInterval?
    var pendingByteRange: IPadHLSUnresolvedByteRange?
    var initializationResource: IPadHLSResource?
    var previousRangeEndByURL: [URL: Int64] = [:]
    var segments: [IPadHLSMediaSegment] = []
    var startSeconds: TimeInterval = 0
    var hasEndList = false
    var targetDuration: TimeInterval?
    var didReadMediaSequence = false
    var didReadDiscontinuitySequence = false
    var didReadSegment = false
    var discontinuitySequence: Int64 = 0

    for line in lines {
      try Task.checkCancellation()
      let uppercased = line.uppercased()
      if uppercased.hasPrefix("#EXT-X-MEDIA-SEQUENCE:") {
        guard !didReadMediaSequence, !didReadSegment,
          pendingDuration == nil, pendingByteRange == nil
        else {
          throw IPadMediaURLResolverError.invalidPlaylist(
            "MEDIA-SEQUENCEが重複またはsegment後に指定されています"
          )
        }
        guard let sequence = Int64(line.dropFirst("#EXT-X-MEDIA-SEQUENCE:".count)),
          sequence >= 0
        else {
          throw IPadMediaURLResolverError.invalidPlaylist("MEDIA-SEQUENCEが不正です")
        }
        nextSequence = sequence
        didReadMediaSequence = true
      } else if uppercased.hasPrefix("#EXT-X-DISCONTINUITY-SEQUENCE:") {
        guard !didReadDiscontinuitySequence, !didReadSegment,
          pendingDuration == nil, pendingByteRange == nil
        else {
          throw IPadMediaURLResolverError.invalidPlaylist(
            "DISCONTINUITY-SEQUENCEが重複またはsegment後に指定されています"
          )
        }
        guard
          let sequence = Int64(
            line.dropFirst("#EXT-X-DISCONTINUITY-SEQUENCE:".count)
          ),
          sequence >= 0
        else {
          throw IPadMediaURLResolverError.invalidPlaylist(
            "DISCONTINUITY-SEQUENCEが不正です"
          )
        }
        discontinuitySequence = sequence
        didReadDiscontinuitySequence = true
      } else if uppercased == "#EXT-X-DISCONTINUITY" {
        guard pendingDuration == nil, pendingByteRange == nil,
          discontinuitySequence < Int64.max
        else {
          throw IPadMediaURLResolverError.invalidPlaylist(
            "DISCONTINUITYの位置またはsequenceが不正です"
          )
        }
        discontinuitySequence += 1
      } else if uppercased.hasPrefix("#EXT-X-TARGETDURATION:") {
        guard
          let duration = TimeInterval(
            line.dropFirst("#EXT-X-TARGETDURATION:".count)
          ),
          duration.isFinite, duration > 0,
          duration <= maximumTimelineSeconds
        else {
          throw IPadMediaURLResolverError.invalidPlaylist(
            "TARGETDURATIONが不正です"
          )
        }
        targetDuration = duration
      } else if uppercased.hasPrefix("#EXTINF:") {
        guard
          let rawDuration = line.dropFirst("#EXTINF:".count)
            .split(separator: ",", maxSplits: 1).first
        else {
          throw IPadMediaURLResolverError.invalidPlaylist("EXTINFが不正です")
        }
        guard let duration = TimeInterval(rawDuration),
          duration.isFinite, duration > 0,
          duration <= maximumTimelineSeconds
        else {
          throw IPadMediaURLResolverError.invalidPlaylist("EXTINFが不正です")
        }
        pendingDuration = duration
      } else if uppercased.hasPrefix("#EXT-X-BYTERANGE:") {
        pendingByteRange = try parseByteRange(
          String(line.dropFirst("#EXT-X-BYTERANGE:".count))
        )
      } else if uppercased.hasPrefix("#EXT-X-MAP:") {
        let attributes = parseAttributeList(String(line.dropFirst("#EXT-X-MAP:".count)))
        guard let uri = attributes["URI"],
          let resourceURL = resolvedHTTPURL(uri, relativeTo: url)
        else {
          throw IPadMediaURLResolverError.invalidPlaylist("EXT-X-MAP URIが不正です")
        }
        let unresolvedRange = try attributes["BYTERANGE"].map(parseByteRange)
        let range = try resolvedByteRange(
          unresolvedRange,
          resourceURL: resourceURL,
          previousEnds: &previousRangeEndByURL
        )
        initializationResource = IPadHLSResource(url: resourceURL, byteRange: range)
      } else if uppercased == "#EXT-X-ENDLIST" {
        hasEndList = true
      } else if !line.hasPrefix("#") {
        guard let duration = pendingDuration,
          let resourceURL = resolvedHTTPURL(line, relativeTo: url)
        else {
          throw IPadMediaURLResolverError.invalidPlaylist("segment情報が不正です")
        }
        let range = try resolvedByteRange(
          pendingByteRange,
          resourceURL: resourceURL,
          previousEnds: &previousRangeEndByURL
        )
        guard nextSequence < Int64.max,
          startSeconds.isFinite,
          duration <= maximumTimelineSeconds - startSeconds
        else {
          throw IPadMediaURLResolverError.invalidPlaylist(
            "segmentの時間またはsequenceが上限を超えています"
          )
        }
        segments.append(
          IPadHLSMediaSegment(
            sequence: nextSequence,
            duration: duration,
            resource: IPadHLSResource(url: resourceURL, byteRange: range),
            initializationResource: initializationResource,
            startSeconds: startSeconds,
            discontinuitySequence: discontinuitySequence
          )
        )
        nextSequence += 1
        startSeconds += duration
        didReadSegment = true
        pendingDuration = nil
        pendingByteRange = nil
      }
    }

    guard !segments.isEmpty else {
      throw IPadMediaURLResolverError.invalidPlaylist("segmentがありません")
    }
    return IPadHLSMediaPlaylist(
      url: url,
      segments: segments,
      isLive: !hasEndList,
      duration: startSeconds,
      targetDuration: targetDuration
    )
  }

  private static func normalizedPlaylistLines(_ text: String) -> [String] {
    text.replacingOccurrences(of: "\u{feff}", with: "")
      .components(separatedBy: .newlines)
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
  }

  private static func parseAttributeList(_ source: String) -> [String: String] {
    var fields: [String] = []
    var current = ""
    var insideQuotes = false
    for character in source {
      if character == "\"" { insideQuotes.toggle() }
      if character == ",", !insideQuotes {
        fields.append(current)
        current = ""
      } else {
        current.append(character)
      }
    }
    fields.append(current)

    var attributes: [String: String] = [:]
    for field in fields {
      let parts = field.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
      guard parts.count == 2 else { continue }
      let key = parts[0].trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
      var value = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
      if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
        value.removeFirst()
        value.removeLast()
      }
      attributes[key] = value
    }
    return attributes
  }

  private static func parseByteRange(_ source: String) throws -> IPadHLSUnresolvedByteRange {
    let parts = source.trimmingCharacters(in: .whitespacesAndNewlines)
      .split(separator: "@", maxSplits: 1)
    guard let rawLength = parts.first,
      let length = Int64(rawLength),
      length > 0
    else {
      throw IPadMediaURLResolverError.invalidByteRange
    }
    let offset: Int64?
    if parts.count == 2 {
      guard let parsedOffset = Int64(parts[1]), parsedOffset >= 0 else {
        throw IPadMediaURLResolverError.invalidByteRange
      }
      offset = parsedOffset
    } else {
      offset = nil
    }
    return IPadHLSUnresolvedByteRange(length: length, offset: offset)
  }

  private static func resolvedByteRange(
    _ unresolved: IPadHLSUnresolvedByteRange?,
    resourceURL: URL,
    previousEnds: inout [URL: Int64]
  ) throws -> IPadHLSByteRange? {
    guard let unresolved else { return nil }
    guard let offset = unresolved.offset ?? previousEnds[resourceURL],
      offset >= 0,
      unresolved.length <= Int64.max - offset
    else { throw IPadMediaURLResolverError.invalidByteRange }
    let result = IPadHLSByteRange(offset: offset, length: unresolved.length)
    previousEnds[resourceURL] = result.endOffset
    return result
  }

  private static func resolvedHTTPURL(_ reference: String, relativeTo baseURL: URL) -> URL? {
    let decoded = decodeHTMLURL(reference.trimmingCharacters(in: .whitespacesAndNewlines))
    guard let absoluteURL = URL(string: decoded, relativeTo: baseURL)?.absoluteURL,
      let safeURL = sanitizedAbsoluteHTTPURL(absoluteURL),
      !isDisallowedDiscoveredLocalURL(safeURL, pageURL: baseURL)
    else { return nil }
    if baseURL.scheme?.lowercased() == "https", safeURL.scheme?.lowercased() == "http" {
      return nil
    }
    return safeURL
  }

  fileprivate static func sanitizedAbsoluteHTTPURL(_ url: URL) -> URL? {
    guard var components = URLComponents(url: url, resolvingAgainstBaseURL: true),
      let scheme = components.scheme?.lowercased(),
      scheme == "http" || scheme == "https",
      components.user == nil,
      components.password == nil,
      let host = components.host,
      !host.isEmpty
    else { return nil }
    components.scheme = scheme
    components.fragment = nil
    return components.url
  }
}

final class IPadHLSResourceDownloader: @unchecked Sendable {
  let maximumResourceBytes: Int
  let maximumRedirectCount: Int
  let requestTimeout: TimeInterval
  let resolutionPolicy: IPadMediaURLResolutionPolicy
  let requestContext: IPadMediaRequestContext?
  private let lock = NSLock()
  private var activeOperations: [UUID: IPadBoundedHTTPRequest] = [:]
  private var initializationCache: [IPadHLSResource: Data] = [:]
  private var isCancelled = false

  init(
    maximumResourceBytes: Int = 256 * 1_024 * 1_024,
    maximumRedirectCount: Int = 6,
    requestTimeout: TimeInterval = 30,
    resolutionPolicy: IPadMediaURLResolutionPolicy = .userSubmitted,
    requestContext: IPadMediaRequestContext? = nil
  ) {
    self.maximumResourceBytes = max(1_024, maximumResourceBytes)
    self.maximumRedirectCount = max(0, maximumRedirectCount)
    self.requestTimeout = max(1, requestTimeout)
    self.resolutionPolicy = resolutionPolicy
    self.requestContext = requestContext
  }

  func data(for resource: IPadHLSResource) async throws -> Data {
    try Task.checkCancellation()
    guard let safeURL = IPadMediaURLResolver.sanitizedAbsoluteHTTPURL(resource.url) else {
      throw IPadMediaURLResolverError.unsafeURL
    }
    if !IPadMediaURLResolver.isURL(safeURL, allowedBy: resolutionPolicy) {
      throw IPadMediaURLResolverError.unsafeURL
    }
    var request = URLRequest(url: safeURL)
    request.httpMethod = "GET"
    request.timeoutInterval = requestTimeout
    request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    requestContext?.applying(to: &request)
    if let byteRange = resource.byteRange {
      guard byteRange.offset >= 0, byteRange.length > 0,
        byteRange.length <= Int64(maximumResourceBytes),
        byteRange.length <= Int64.max - byteRange.offset
      else { throw IPadMediaURLResolverError.invalidByteRange }
      let end = byteRange.offset + byteRange.length - 1
      request.setValue(
        "bytes=\(byteRange.offset)-\(end)",
        forHTTPHeaderField: "Range"
      )
    }

    let operation = IPadBoundedHTTPRequest(
      maximumResponseBytes: maximumResourceBytes,
      maximumRedirectCount: maximumRedirectCount,
      timeout: requestTimeout,
      resolutionPolicy: resolutionPolicy,
      requestContext: requestContext
    )
    let operationID = UUID()
    guard register(operation, id: operationID) else { throw CancellationError() }
    defer { unregister(id: operationID) }

    let payload = try await operation.start(request)
    try IPadMediaURLResolver.validateNoInteractionChallenge(payload.response)
    await requestContext?.updateCookies(from: payload.response)
    try Task.checkCancellation()

    if let byteRange = resource.byteRange {
      guard payload.response.statusCode == 206,
        payload.data.count == Int(byteRange.length),
        Self.validContentRange(payload.response, expected: byteRange)
      else {
        throw IPadMediaURLResolverError.invalidByteRange
      }
    } else {
      try IPadMediaURLResolver.validateSuccessfulResponse(payload.response)
    }
    return payload.data
  }

  func materialize(
    segment: IPadHLSMediaSegment,
    in directory: URL
  ) async throws -> URL {
    try Task.checkCancellation()
    let initializationData: Data?
    if let initializationResource = segment.initializationResource {
      initializationData = try await cachedInitializationData(
        for: initializationResource
      )
    } else {
      initializationData = nil
    }
    let downloadedSegmentData = try await data(for: segment.resource)
    try Task.checkCancellation()
    try checkNotCancelled()
    guard !downloadedSegmentData.isEmpty else {
      throw IPadMediaURLResolverError.requestFailed(
        "HLS区間の応答が空でした"
      )
    }
    guard !Self.looksLikeHTML(downloadedSegmentData) else {
      throw IPadMediaURLResolverError.requestFailed(
        "HLS区間の応答が動画ではなくHTMLでした。ブラウザで再生を開始してから、配信を再解析してください。"
      )
    }
    let segmentData = try Self.normalizedMediaSegmentData(
      downloadedSegmentData,
      hasInitializationData: initializationData != nil
    )

    // EXT-X-MAP normally accompanies an fMP4 fragment, but some providers
    // attach it to transport streams or return a self-contained MP4 segment.
    // In both cases prepending the map corrupts an otherwise usable payload:
    // TS becomes "init + TS", while a complete MP4 gains duplicate ftyp/moov
    // boxes. Trust the downloaded media bytes before the playlist hint.
    let effectiveInitializationData: Data?
    if Self.transportStreamOffset(in: segmentData) == 0
      || Self.containsISOBaseMediaInitialization(segmentData)
    {
      effectiveInitializationData = nil
    } else {
      effectiveInitializationData = initializationData
    }

    guard (effectiveInitializationData?.count ?? 0)
      <= maximumResourceBytes - segmentData.count
    else {
      throw IPadMediaURLResolverError.responseTooLarge(maximumResourceBytes)
    }

    var output = Data()
    output.reserveCapacity(
      (effectiveInitializationData?.count ?? 0) + segmentData.count
    )
    if let effectiveInitializationData {
      output.append(effectiveInitializationData)
    }
    output.append(segmentData)

    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    let fileExtension = Self.materializedFileExtension(
      initializationData: effectiveInitializationData,
      segmentData: segmentData,
      resourceURL: segment.resource.url
    )
    let outputURL = directory.appendingPathComponent(
      "mioh-hls-\(segment.sequence)-\(UUID().uuidString.lowercased()).\(fileExtension)",
      isDirectory: false
    )
    try output.write(to: outputURL, options: [.atomic, .completeFileProtectionUnlessOpen])
    return outputURL
  }

  func cancel() {
    lock.lock()
    guard !isCancelled else {
      lock.unlock()
      return
    }
    isCancelled = true
    let operations = Array(activeOperations.values)
    activeOperations.removeAll()
    initializationCache.removeAll()
    lock.unlock()
    for operation in operations { operation.cancel() }
  }

  private func register(_ operation: IPadBoundedHTTPRequest, id: UUID) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard !isCancelled else { return false }
    activeOperations[id] = operation
    return true
  }

  private func unregister(id: UUID) {
    lock.lock()
    activeOperations[id] = nil
    lock.unlock()
  }

  private func checkNotCancelled() throws {
    lock.lock()
    let cancelled = isCancelled
    lock.unlock()
    if cancelled { throw CancellationError() }
  }

  private func cachedInitializationData(
    for resource: IPadHLSResource
  ) async throws -> Data {
    if let cached = cachedInitializationValue(for: resource) { return cached }
    let downloaded = try await data(for: resource)
    try checkNotCancelled()
    guard !downloaded.isEmpty else {
      throw IPadMediaURLResolverError.requestFailed(
        "HLS初期化データの応答が空でした"
      )
    }
    guard !Self.looksLikeHTML(downloaded) else {
      throw IPadMediaURLResolverError.requestFailed(
        "HLS初期化データの応答が動画ではなくHTMLでした"
      )
    }
    storeInitializationValue(downloaded, for: resource)
    return downloaded
  }

  private func cachedInitializationValue(
    for resource: IPadHLSResource
  ) -> Data? {
    lock.lock()
    let value = initializationCache[resource]
    lock.unlock()
    return value
  }

  private func storeInitializationValue(
    _ data: Data,
    for resource: IPadHLSResource
  ) {
    lock.lock()
    if !isCancelled { initializationCache[resource] = data }
    lock.unlock()
  }

  private static func validContentRange(
    _ response: HTTPURLResponse,
    expected: IPadHLSByteRange
  ) -> Bool {
    guard let rawValue = response.value(forHTTPHeaderField: "Content-Range")?.lowercased() else {
      return false
    }
    let expectedEnd = expected.offset + expected.length - 1
    return rawValue.hasPrefix("bytes \(expected.offset)-\(expectedEnd)/")
  }

  private static func looksLikeHTML(_ data: Data) -> Bool {
    let prefix = String(decoding: data.prefix(512), as: UTF8.self)
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
    return prefix.hasPrefix("<!doctype html")
      || prefix.hasPrefix("<html")
      || prefix.hasPrefix("<head")
      || prefix.hasPrefix("<body")
  }

  /// Some HLS CDNs disguise MPEG-TS segments as a tiny PNG followed by
  /// padding and the real transport stream. AVPlayer accepts the playlist,
  /// but a materialized segment still begins with PNG bytes and AVURLAsset
  /// fails with `Cannot Open`. Remove only a bounded prefix after finding a
  /// strong MPEG-TS sync pattern; ordinary TS and ISO-BMFF segments are left
  /// byte-for-byte unchanged.
  private static func normalizedMediaSegmentData(
    _ data: Data,
    hasInitializationData: Bool
  ) throws -> Data {
    guard !looksLikeISOBaseMedia(data) else {
      return data
    }
    if let offset = transportStreamOffset(in: data) {
      return offset == 0 ? data : Data(data[offset...])
    }
    // An initialization map is only a playlist hint. If the payload is not a
    // recognizable transport stream, preserve it here and let the fMP4 path
    // prepend and validate the map below.
    if hasInitializationData { return data }
    let pngSignature: [UInt8] = [
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    ]
    if data.starts(with: pngSignature) {
      throw IPadMediaURLResolverError.requestFailed(
        "HLS区間の画像前置データからMPEG-TSを検出できませんでした"
      )
    }
    return data
  }

  private static func transportStreamOffset(in data: Data) -> Int? {
    let packetSize = 188
    let requiredPackets = 5
    let requiredSpan = packetSize * (requiredPackets - 1)
    guard data.count > requiredSpan else {
      return data.first == 0x47 ? 0 : nil
    }
    let maximumLeadingBytes = 64 * 1_024
    let lastCandidate = min(
      maximumLeadingBytes,
      data.count - requiredSpan - 1
    )
    guard lastCandidate >= 0 else { return nil }
    for candidate in 0...lastCandidate where data[candidate] == 0x47 {
      var matched = true
      for packet in 1..<requiredPackets
      where data[candidate + packet * packetSize] != 0x47 {
        matched = false
        break
      }
      if matched { return candidate }
    }
    return nil
  }

  private static func looksLikeISOBaseMedia(_ data: Data) -> Bool {
    let bytes = [UInt8](data.prefix(16))
    guard bytes.count >= 8 else { return false }
    let boxType = String(bytes: bytes[4..<8], encoding: .ascii)?.lowercased()
    return ["ftyp", "styp", "moof", "moov"].contains(boxType)
  }

  /// Returns true only when the media response already carries its own `moov`
  /// initialization box. A bare fragment usually starts with `styp`/`moof`
  /// and must still receive EXT-X-MAP bytes.
  private static func containsISOBaseMediaInitialization(_ data: Data) -> Bool {
    let bytes = [UInt8](data.prefix(4 * 1_024 * 1_024))
    var offset = 0
    var inspectedBoxes = 0
    while offset <= bytes.count - 8, inspectedBoxes < 64 {
      let size32 = UInt64(bytes[offset]) << 24
        | UInt64(bytes[offset + 1]) << 16
        | UInt64(bytes[offset + 2]) << 8
        | UInt64(bytes[offset + 3])
      let type = String(
        bytes: bytes[(offset + 4)..<(offset + 8)],
        encoding: .ascii
      )?.lowercased()
      if type == "moov" { return true }
      if type == "moof" || type == "mdat" { return false }

      let headerSize: UInt64
      let boxSize: UInt64
      if size32 == 1 {
        guard offset <= bytes.count - 16 else { return false }
        headerSize = 16
        boxSize = bytes[(offset + 8)..<(offset + 16)].reduce(0) {
          ($0 << 8) | UInt64($1)
        }
      } else if size32 == 0 {
        boxSize = UInt64(bytes.count - offset)
        headerSize = 8
      } else {
        boxSize = size32
        headerSize = 8
      }
      guard boxSize >= headerSize, boxSize <= UInt64(bytes.count - offset),
        boxSize <= UInt64(Int.max)
      else { return false }
      offset += Int(boxSize)
      inspectedBoxes += 1
    }
    return false
  }

  private static func materializedFileExtension(
    initializationData: Data?,
    segmentData: Data,
    resourceURL: URL
  ) -> String {
    // The response body is more authoritative than the URL suffix. Some HLS
    // providers serve a real MPEG-TS payload from a path ending in `.mp4` (or
    // declare `video/mp4`). normalizedMediaSegmentData has already removed any
    // bounded image prefix, so a detected sync train here must stay `.ts` and
    // use the timestamp-normalizing TS remux path.
    if transportStreamOffset(in: segmentData) == 0 { return "ts" }
    if initializationData != nil { return "mp4" }
    if looksLikeISOBaseMedia(segmentData) { return "mp4" }
    if ["mp4", "m4s", "m4v", "mov"].contains(
      resourceURL.pathExtension.lowercased()
    ) {
      return "mp4"
    }
    return "ts"
  }
}

private struct IPadHTMLMediaCandidates {
  let hlsCandidates: [URL]
  let progressiveCandidates: [URL]
  let pageCandidates: [URL]
}

private struct IPadHTMLPage {
  let payload: IPadHTTPPayload
  let url: URL
  let depth: Int
  let policy: IPadMediaURLResolutionPolicy
}

private struct IPadHLSVariant {
  let url: URL
  let bandwidth: Int64
  let resolution: IPadHLSResolution?
}

private struct IPadHLSResolution {
  let width: Int
  let height: Int

  var pixelCount: Int { width * height }
}

private struct IPadHLSUnresolvedByteRange {
  let length: Int64
  let offset: Int64?
}

private struct IPadResolvedHLS {
  let playlist: IPadHLSMediaPlaylist
  let contentType: String?
}

private final class IPadMediaResolutionBudget: @unchecked Sendable {
  private let lock = NSLock()
  private var remainingRequests: Int
  private var remainingResponseBytes: Int
  private let deadline: Date

  init(maximumRequests: Int, maximumResponseBytes: Int, timeout: TimeInterval) {
    remainingRequests = maximumRequests
    remainingResponseBytes = maximumResponseBytes
    deadline = Date().addingTimeInterval(timeout)
  }

  func consumeRequest() throws -> TimeInterval {
    lock.lock()
    defer { lock.unlock() }
    let remainingTime = deadline.timeIntervalSinceNow
    guard remainingRequests > 0, remainingTime > 0 else {
      throw IPadMediaURLResolverError.resolutionLimitExceeded
    }
    remainingRequests -= 1
    return remainingTime
  }

  func consumeResponseBytes(_ count: Int) throws {
    lock.lock()
    defer { lock.unlock() }
    guard count >= 0, count <= remainingResponseBytes else {
      throw IPadMediaURLResolverError.resolutionLimitExceeded
    }
    remainingResponseBytes -= count
  }
}

private struct IPadHTTPPayload {
  let data: Data
  let response: HTTPURLResponse
}

private final class IPadBoundedHTTPRequest: NSObject, @unchecked Sendable {
  private let maximumResponseBytes: Int
  private let maximumRedirectCount: Int
  private let timeout: TimeInterval
  private let returnsAfterVideoResponse: Bool
  private let resolutionPolicy: IPadMediaURLResolutionPolicy
  private let requestContext: IPadMediaRequestContext?
  private let lock = NSLock()
  private var continuation: CheckedContinuation<IPadHTTPPayload, Error>?
  private var session: URLSession?
  private var task: URLSessionDataTask?
  private var response: HTTPURLResponse?
  private var receivedData = Data()
  private var redirectCount = 0
  private var cancellationRequested = false
  private var didFinish = false
  private var expectsResponseBody = true

  init(
    maximumResponseBytes: Int,
    maximumRedirectCount: Int,
    timeout: TimeInterval,
    returnsAfterVideoResponse: Bool = false,
    resolutionPolicy: IPadMediaURLResolutionPolicy = .userSubmitted,
    requestContext: IPadMediaRequestContext? = nil
  ) {
    self.maximumResponseBytes = maximumResponseBytes
    self.maximumRedirectCount = maximumRedirectCount
    self.timeout = timeout
    self.returnsAfterVideoResponse = returnsAfterVideoResponse
    self.resolutionPolicy = resolutionPolicy
    self.requestContext = requestContext
  }

  func start(_ request: URLRequest) async throws -> IPadHTTPPayload {
    try Task.checkCancellation()
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.urlCredentialStorage = nil
        configuration.waitsForConnectivity = false

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInitiated
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
        let task = session.dataTask(with: request)

        lock.lock()
        if cancellationRequested {
          didFinish = true
          lock.unlock()
          session.invalidateAndCancel()
          continuation.resume(throwing: CancellationError())
          return
        }
        self.continuation = continuation
        self.session = session
        self.task = task
        expectsResponseBody = request.httpMethod?.uppercased() != "HEAD"
        lock.unlock()
        task.resume()
      }
    } onCancel: {
      self.cancel()
    }
  }

  fileprivate func cancel() {
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
    self.continuation = nil
    self.task = nil
    self.session = nil
    lock.unlock()
    task?.cancel()
    session?.invalidateAndCancel()
    continuation.resume(throwing: CancellationError())
  }

  private func finish(_ result: Result<IPadHTTPPayload, Error>) {
    lock.lock()
    guard !didFinish, let continuation else {
      lock.unlock()
      return
    }
    didFinish = true
    let task = task
    let session = session
    self.continuation = nil
    self.task = nil
    self.session = nil
    lock.unlock()
    task?.cancel()
    session?.invalidateAndCancel()
    continuation.resume(with: result)
  }
}

extension IPadBoundedHTTPRequest: URLSessionDataDelegate {
  func urlSession(
    _: URLSession,
    dataTask _: URLSessionDataTask,
    didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    guard let httpResponse = response as? HTTPURLResponse else {
      completionHandler(.cancel)
      finish(.failure(IPadMediaURLResolverError.requestFailed("HTTP応答ではありません")))
      return
    }
    if let error = IPadMediaURLResolver.interactionChallengeError(
      response: httpResponse
    ) {
      completionHandler(.cancel)
      finish(.failure(error))
      return
    }
    if returnsAfterVideoResponse,
      httpResponse.mimeType?.lowercased().hasPrefix("video/") == true
    {
      completionHandler(.cancel)
      finish(.success(IPadHTTPPayload(data: Data(), response: httpResponse)))
      return
    }
    lock.lock()
    let responseBodyExpected = expectsResponseBody
    lock.unlock()
    if responseBodyExpected,
      httpResponse.expectedContentLength > Int64(maximumResponseBytes)
    {
      completionHandler(.cancel)
      finish(.failure(IPadMediaURLResolverError.responseTooLarge(maximumResponseBytes)))
      return
    }
    lock.lock()
    self.response = httpResponse
    lock.unlock()
    completionHandler(.allow)
  }

  func urlSession(_: URLSession, dataTask _: URLSessionDataTask, didReceive data: Data) {
    lock.lock()
    let wouldOverflow = !didFinish && data.count > maximumResponseBytes - receivedData.count
    if !didFinish, !wouldOverflow { receivedData.append(data) }
    lock.unlock()
    if wouldOverflow {
      finish(.failure(IPadMediaURLResolverError.responseTooLarge(maximumResponseBytes)))
    }
  }

  func urlSession(
    _: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    guard let destination = request.url,
      IPadMediaURLResolver.sanitizedAbsoluteHTTPURL(destination) != nil
    else {
      completionHandler(nil)
      finish(.failure(IPadMediaURLResolverError.unsafeURL))
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
      finish(.failure(IPadMediaURLResolverError.unsafeURL))
      return
    }
    if let sourceURL = task.currentRequest?.url,
      IPadMediaURLResolver.isDisallowedDiscoveredLocalURL(
        destination,
        pageURL: sourceURL
      )
    {
      completionHandler(nil)
      finish(.failure(IPadMediaURLResolverError.unsafeURL))
      return
    }
    let sourceScheme = task.currentRequest?.url?.scheme?.lowercased()
    let destinationScheme = destination.scheme?.lowercased()
    if sourceScheme == "https", destinationScheme == "http" {
      completionHandler(nil)
      finish(.failure(IPadMediaURLResolverError.insecureRedirect))
      return
    }
    lock.lock()
    redirectCount += 1
    let tooManyRedirects = redirectCount > maximumRedirectCount
    lock.unlock()
    if tooManyRedirects {
      completionHandler(nil)
      finish(.failure(IPadMediaURLResolverError.tooManyRedirects))
    } else {
      Task {
        await requestContext?.updateCookies(from: response)
        var redirectedRequest = request
        requestContext?.applying(to: &redirectedRequest)
        completionHandler(redirectedRequest)
      }
    }
  }

  func urlSession(_: URLSession, task _: URLSessionTask, didCompleteWithError error: Error?) {
    if let error {
      finish(.failure(error))
      return
    }
    lock.lock()
    let response = response
    let data = receivedData
    lock.unlock()
    guard let response else {
      finish(.failure(IPadMediaURLResolverError.requestFailed("応答がありません")))
      return
    }
    finish(.success(IPadHTTPPayload(data: data, response: response)))
  }
}
