import Foundation

enum IPadWebMediaDiscoveryError: LocalizedError {
  case invalidURL
  case noCandidates
  case pageLoadFailed(String)
  case interactionRequired

  var errorDescription: String? {
    switch self {
    case .invalidURL:
      "ブラウザ解析には安全なHTTPSページを指定してください。"
    case .noCandidates:
      "配信URLをまだ確認できません。ページ内の動画を一度再生し、本編映像が表示されてから、もう一度解析してください。"
    case .pageLoadFailed(let detail):
      "ブラウザでページを読み込めませんでした: \(detail)"
    case .interactionRequired:
      "動的ページは、表示中のブラウザタブで確認してから解析してください。"
    }
  }
}

struct IPadWebMediaCandidate: Sendable, Equatable {
  enum DiscoveryRole: Sendable, Equatable {
    case activePlayback
    case verifiedMediaResponse
    case unverifiedMediaResponse
    case directHint
    case pageLead
  }

  enum SelectionState: Sendable, Equatable {
    case activeCurrentSource
    case ordinary
    case supersededCurrentSource
  }

  let url: URL
  let requestContext: IPadMediaRequestContext
  /// The visible top-level page that owns this candidate. Interaction
  /// challenges must return here instead of opening a Cloudflare subresource
  /// or media CDN challenge URL as a new main document.
  let interactionPageURL: URL?
  let selectionState: SelectionState
  let discoveryRole: DiscoveryRole
  let mediaEvidence: IPadBrowserMediaEvidence?
  /// Exact isolated-world document identity captured by the visible browser.
  /// A browser relay may use this token to return to the same WKFrameInfo;
  /// native/disconnected candidates deliberately leave it nil.
  let browserDocumentToken: String?
  /// True only when the page's successful Fetch/XHR observation proved that
  /// replaying bytes through the same WebKit frame is available. Provisional
  /// native/opaque observations may also set this after exact-frame binding.
  let browserRelayEligible: Bool
  /// True when relay eligibility came only from native currentSrc or a passive
  /// performance/script association and must be proven on a VOD segment before
  /// native playback. Fulfilled Fetch/XHR observations leave this false.
  let browserRelayRequiresProbe: Bool
  /// Mirrors only an explicitly observed Fetch `credentials: include` or XHR
  /// `withCredentials = true`; unknown and ordinary media URLs stay false.
  let browserRelayIncludesCredentials: Bool

  init(
    url: URL,
    requestContext: IPadMediaRequestContext,
    interactionPageURL: URL? = nil,
    selectionState: SelectionState = .ordinary,
    discoveryRole: DiscoveryRole = .directHint,
    mediaEvidence: IPadBrowserMediaEvidence? = nil,
    browserDocumentToken: String? = nil,
    browserRelayEligible: Bool = false,
    browserRelayRequiresProbe: Bool = false,
    browserRelayIncludesCredentials: Bool = false
  ) {
    self.url = url
    self.requestContext = requestContext
    self.interactionPageURL = interactionPageURL
    self.selectionState = selectionState
    self.discoveryRole = discoveryRole
    self.mediaEvidence = mediaEvidence
    self.browserDocumentToken = browserDocumentToken
    self.browserRelayEligible = browserRelayEligible
    self.browserRelayRequiresProbe = browserRelayRequiresProbe
    self.browserRelayIncludesCredentials = browserRelayIncludesCredentials
  }
}

struct IPadWebMediaDiscoveryResult: Sendable, Equatable {
  let candidates: [IPadWebMediaCandidate]
  let encounteredInteractionRequired: Bool
}

/// Offscreen page execution is intentionally disabled. Dynamic pages and
/// interactive verification are handled only by the visible Browser tab.
@MainActor
final class IPadWebMediaDiscovery {
  func discoverCandidates(from rawValue: String) async throws
    -> IPadWebMediaDiscoveryResult
  {
    try Task.checkCancellation()
    guard Self.isValidVisibleBrowserURL(rawValue) else {
      throw IPadWebMediaDiscoveryError.invalidURL
    }
    throw IPadWebMediaDiscoveryError.interactionRequired
  }

  private static func isValidVisibleBrowserURL(_ rawValue: String) -> Bool {
    let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
      trimmed.rangeOfCharacter(from: .controlCharacters) == nil
    else { return false }
    let value = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
    guard let components = URLComponents(string: value),
      components.scheme?.lowercased() == "https",
      components.user == nil, components.password == nil,
      components.host?.isEmpty == false
    else { return false }
    return true
  }
}
