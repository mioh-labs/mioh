import Combine
import Foundation
import SwiftUI
import WebKit

private final class IPadWeakInteractiveScriptMessageHandler: NSObject,
  WKScriptMessageHandler
{
  weak var delegate: WKScriptMessageHandler?

  func userContentController(
    _ userContentController: WKUserContentController,
    didReceive message: WKScriptMessage
  ) {
    delegate?.userContentController(userContentController, didReceive: message)
  }
}

/// Gives a user-created JavaScript window a real `WindowProxy` before it is
/// adopted as the visible page. Strict mode bounds that temporary window;
/// compatibility mode lets WebKit complete ordinary multi-stage navigation.
@MainActor
private final class IPadTransientPopupCoordinator: NSObject,
  WKNavigationDelegate, WKUIDelegate
{
  private let relaxedWebCompatibility: Bool
  private let allowsNavigation: (URL?) -> Bool
  private let closeHandler: (WKWebView) -> Void
  private let readyHandler: (WKWebView) -> Void
  private var remainingNavigationCount: Int

  init(
    maximumNavigationCount: Int,
    relaxedWebCompatibility: Bool = false,
    allowsNavigation: @escaping (URL?) -> Bool,
    readyHandler: @escaping (WKWebView) -> Void,
    closeHandler: @escaping (WKWebView) -> Void
  ) {
    remainingNavigationCount = maximumNavigationCount
    self.relaxedWebCompatibility = relaxedWebCompatibility
    self.allowsNavigation = allowsNavigation
    self.readyHandler = readyHandler
    self.closeHandler = closeHandler
  }

  func webView(
    _ webView: WKWebView,
    decidePolicyFor navigationAction: WKNavigationAction,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) {
    if relaxedWebCompatibility {
      guard !navigationAction.shouldPerformDownload,
        allowsNavigation(navigationAction.request.url)
      else {
        decisionHandler(.cancel)
        closeHandler(webView)
        return
      }
      decisionHandler(.allow)
      return
    }
    guard !navigationAction.shouldPerformDownload,
      remainingNavigationCount > 0,
      allowsNavigation(navigationAction.request.url)
    else {
      decisionHandler(.cancel)
      return
    }
    remainingNavigationCount -= 1
    decisionHandler(.allow)
  }

  func webView(
    _ webView: WKWebView,
    decidePolicyFor navigationResponse: WKNavigationResponse,
    decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
  ) {
    if relaxedWebCompatibility {
      guard allowsNavigation(navigationResponse.response.url) else {
        decisionHandler(.cancel)
        closeHandler(webView)
        return
      }
      decisionHandler(.allow)
      return
    }
    let response = navigationResponse.response as? HTTPURLResponse
    let disposition = response?.value(
      forHTTPHeaderField: "Content-Disposition"
    )?.lowercased()
    guard allowsNavigation(navigationResponse.response.url),
      navigationResponse.canShowMIMEType,
      disposition?.contains("attachment") != true
    else {
      decisionHandler(.cancel)
      return
    }
    decisionHandler(.allow)
  }

  func webView(
    _ webView: WKWebView,
    createWebViewWith configuration: WKWebViewConfiguration,
    for navigationAction: WKNavigationAction,
    windowFeatures: WKWindowFeatures
  ) -> WKWebView? {
    nil
  }

  func webViewDidClose(_ webView: WKWebView) {
    closeHandler(webView)
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    guard allowsNavigation(webView.url), webView.url?.scheme?.lowercased() == "https" else {
      return
    }
    readyHandler(webView)
  }
}

@MainActor
final class IPadInteractiveMediaBrowser: NSObject, ObservableObject {
  struct CandidateSummary: Identifiable, Equatable {
    let id: String
    let url: String
    let sourceLabel: String
    let verificationLabel: String
    let frameDepth: Int
    let isVerified: Bool
  }

  struct SuccessfulPageVisit: Equatable {
    let url: URL
    let title: String
    let revision: Int
  }

  @Published var addressText = ""
  @Published private(set) var pageTitle = ""
  @Published private(set) var statusMessage: String?
  @Published private(set) var isLoading = false
  @Published private(set) var canGoBack = false
  @Published private(set) var canGoForward = false
  @Published private(set) var challengeActive = false
  @Published private(set) var candidateCount = 0
  @Published private(set) var readyCandidateGeneration = 0
  @Published private(set) var mediaSourceRevision = 0
  @Published private(set) var candidateRevision = 0
  @Published private(set) var navigationGeneration = 0
  @Published private(set) var successfulPageVisit: SuccessfulPageVisit?
  @Published private(set) var webViewGeneration = 0
  @Published private(set) var challengeCompatibilityTimedOut = false
  @Published private(set) var candidateSummaries: [CandidateSummary] = []
  @Published private(set) var hasOpenedPage = false
  @Published private(set) var canReturnToOpeningPage = false

  private(set) var webView: WKWebView

  private struct Candidate {
    enum Provenance {
      case script
      case currentPage
      case mainNavigationResponse
      case subframeNavigationResponse
    }

    let url: URL
    let sourceKind: String
    let frameDepth: Int
    let discoveryOrder: Int
    let frameURL: URL?
    let provenance: Provenance
    let activeRank: Int?
    /// Isolated-world document identity. Native navigation candidates do not
    /// have one and therefore can never inherit opaque MSE playback evidence.
    let documentToken: String?
    let observedAt: Date

    init(
      url: URL,
      sourceKind: String,
      frameDepth: Int,
      discoveryOrder: Int,
      frameURL: URL?,
      provenance: Provenance,
      activeRank: Int? = nil,
      documentToken: String? = nil,
      observedAt: Date = Date()
    ) {
      self.url = url
      self.sourceKind = sourceKind
      self.frameDepth = frameDepth
      self.discoveryOrder = discoveryOrder
      self.frameURL = frameURL
      self.provenance = provenance
      self.activeRank = activeRank
      self.documentToken = documentToken
      self.observedAt = observedAt
    }
  }

  private struct MediaSlotKey: Hashable {
    let documentToken: String
    let slotToken: String
  }

  private struct MediaSlotState {
    let generation: Int
    let currentURL: URL?
    let frameDepth: Int
    let activationOrder: Int
    let frameURL: URL?
    let duration: TimeInterval?
    let currentTime: TimeInterval
    let isPlaying: Bool
    let isEnded: Bool
    let isVisible: Bool
    let visibilityAttested: Bool
    let renderedArea: Int
    /// True only when the actual video element is inside a compact,
    /// lower-right fixed overlay. This is exclusion-only evidence: page
    /// instrumentation may suppress a real source, but can never authorize
    /// one or relax the resolver's network policy.
    let isCompactFloatingOverlay: Bool
    /// `blob:`/MSE playback has no replayable URL, but it is still useful as
    /// same-document evidence for a separately observed HLS response.
    let hasOpaqueSource: Bool
    let sourceActivatedAt: Date
    let lastObservedAt: Date
  }

  private struct OpaquePlaybackAssociation {
    let candidate: Candidate
    let state: MediaSlotState
    let includesObservedDuration: Bool
  }

  struct PreRollWait: Equatable {
    fileprivate let documentToken: String
    fileprivate let slotToken: String
    fileprivate let sourceGeneration: Int
    let remainingSeconds: TimeInterval
  }

  private enum SubframeNavigationAuthority {
    case pageLoad
    case inspection(routeToken: String)
    case challenge(chainID: String)
  }

  private struct SubframeNavigationChain {
    var authority: SubframeNavigationAuthority
    var sourceDocumentEpoch: Int
    var remainingHopCount: Int
    let mainNavigationGeneration: Int
  }

  private struct SubframeNavigationAuthorization {
    let id: String
    let initiatorDocumentToken: String
    let initiatorDocumentEpoch: Int?
    let initiatorFrameURL: String
    /// The old document in the target frame. This is `nil` for the first
    /// navigation of a newly-created child and must never alias its parent.
    let targetPriorDocumentToken: String?
    let nativeMainChallengeGeneration: Int?
    let isInitialChildChallengeFallback: Bool
    let isInitialUserActivatedFallback: Bool
    var destinationURL: String
    var chain: SubframeNavigationChain
    var retiresSubframeChallengeOnCommit = false
    let initialUserNavigationTypeRawValue: Int?
    let initialUserRedirectDeadline: Date?
    var visitedDestinationURLs: Set<String>
  }

  private static let messageHandlerName = "miohInteractiveMediaBrowser"
  /// Compatibility baseline: let WebKit navigate like an ordinary browser.
  /// Media URLs remain untrusted and still pass through the native resolver.
  /// Reintroduce navigation restrictions only after each rule is verified
  /// against real multi-stage players.
  private static let relaxedWebCompatibilityEnabled = true
  private static let instrumentationContentWorld = WKContentWorld.world(
    name: "com.mioh-labs.MiohRemote.instrumentation"
  )
  private static let maximumCandidateCount = 128
  private static let maximumCandidateURLLength = 8_192
  private static let maximumFrameDepth = 8
  private static let maximumKnownFrameCount = 64
  private static let maximumScriptTokenLength = 128
  private static let maximumMainNavigationRedirectCount = 8
  private static let maximumSubframeNavigationHopCount = 8
  private static let maximumChallengeLocalFrameNavigationCount = 32
  private static let maximumNativeChallengeFrameNavigationCount = 64
  private static let maximumInitialUserActivatedChildNavigationCount = 8
  private static let initialUserRedirectLifetime: TimeInterval = 10
  private static let maximumTransientPopupNavigationCount = 8
  private static let relaxedTransientPopupNavigationCount = 64
  private static let maximumTransientPopupCreationCount = 4
  private static let maximumRelaxedTransientPopupCreationCount = 8
  private static let transientPopupLifetimeNanoseconds: UInt64 = 30_000_000_000
  private static let maximumMediaSlotCount = 512
  private static let maximumMediaSlotTokenLength = 32
  private static let maximumCurrentSourceHistoryCount = 256

  private let websiteDataStore: WKWebsiteDataStore
  private let messageHandlerProxy: IPadWeakInteractiveScriptMessageHandler
  private let pageNetworkBridgeEventName =
    "mioh-hls-\(UUID().uuidString.lowercased())"
  private var pageNetworkObservationEpoch = UUID().uuidString.lowercased()
  private var candidates: [String: Candidate] = [:]
  private var nextDiscoveryOrder = 0
  private var mediaSlots: [MediaSlotKey: MediaSlotState] = [:]
  private var currentSourceHistory = Set<String>()
  private var reportedFloatingAdvertisementURLKeys = Set<String>()
  private var floatingAdvertisementURLKeys: Set<String> {
    reportedFloatingAdvertisementURLKeys.union(
      mediaSlots.values.compactMap { state in
        state.isCompactFloatingOverlay
          ? state.currentURL?.absoluteString : nil
      }
    )
  }
  private var nextMediaActivationOrder = 0
  private var frameHeartbeatDates: [String: Date] = [:]
  private var quietTask: Task<Void, Never>?
  private var challengeCompatibilityTask: Task<Void, Never>?
  private var mainFrameChallengeResponse = false
  private var mainFrameChallengeResponseGeneration: Int?
  private var inspectionRequested = false
  private var isClosingPage = false
  private var knownFrames: [String: WKFrameInfo] = [:]
  private var mainDocumentToken: String?
  private var committedMainDocumentURL: URL?
  private var pendingSuccessfulMainResponseURL: URL?
  private var authorizedRouteToken: String?
  private var authorizedRouteURL: String?
  private var authorizedMainResponseURLs: [String: Int] = [:]
  private var subframeDocumentEpochs: [String: Int] = [:]
  private var subframeInspectionRouteTokens: [String: String] = [:]
  private var subframeNavigationChains: [String: SubframeNavigationChain] = [:]
  private var pendingSubframeAuthorizations: [SubframeNavigationAuthorization] = []
  private var completedSubframeAuthorizations: [SubframeNavigationAuthorization] = []
  private var activeSubframeChallengeChainIDs: Set<String> = []
  private var mainFrameScriptChallengeActive = false
  private var pendingMainScriptChallengeEpoch: String?
  private var mainNavigationRedirectCount = 0
  private var activeNavigation: WKNavigation?
  private var navigationRefererURL: URL?
  private var lastPageLocationToken: String?
  private var sameDocumentHardReloadUsed = false
  private var sameDocumentHardReloadInFlight = false
  private var pendingSameDocumentPageURL: String?
  private var pendingChallengePageURL: String?
  private var acceptingScriptCandidates = false
  private var initialChildChallengeDecisionID: String?
  private var initialChildChallengeFallbackGeneration: Int?
  private var pendingInitialUserActivatedChildDecisions = 0
  private var initialUserActivatedChildNavigationCount = 0
  private var challengeLocalFrameNavigationCount = 0
  private var nativeChallengeFrameNavigationCount = 0
  private var transientPopupWebView: WKWebView?
  private var transientPopupCoordinator: IPadTransientPopupCoordinator?
  private var transientPopupRetirementTask: Task<Void, Never>?
  private var transientPopupCreationCount = 0
  private var openingPageWebView: WKWebView?
  /// Keeps intermediate popup openers alive while a multi-stage player moves
  /// through more than one real `WindowProxy`. The original page remains the
  /// single user-visible return target.
  private var retainedPopupOpenerWebViews: [WKWebView] = []
  private var webContentProcessTerminated = false

  override init() {
    // Cloudflare's WebView flow relies on ordinary DOM storage and cookies.
    // Keep browser state in WebKit's standard store, just like Safari-backed
    // in-app browsing, while still replacing the visible WebView on Close.
    let websiteDataStore = WKWebsiteDataStore.default()
    let messageHandlerProxy = IPadWeakInteractiveScriptMessageHandler()
    self.websiteDataStore = websiteDataStore
    self.messageHandlerProxy = messageHandlerProxy
    webView = Self.makeWebView(
      websiteDataStore: websiteDataStore,
      messageHandlerProxy: messageHandlerProxy
    )
    super.init()

    messageHandlerProxy.delegate = self
    attachDelegates(to: webView)
  }

  private static func makeWebView(
    websiteDataStore: WKWebsiteDataStore,
    messageHandlerProxy: WKScriptMessageHandler
  ) -> WKWebView {
    let contentController = WKUserContentController()
    contentController.add(
      messageHandlerProxy,
      contentWorld: instrumentationContentWorld,
      name: messageHandlerName
    )
    contentController.addUserScript(
      WKUserScript(
        source: passiveInstrumentationScript,
        injectionTime: .atDocumentStart,
        forMainFrameOnly: false,
        in: instrumentationContentWorld
      )
    )
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = websiteDataStore
    configuration.userContentController = contentController
    configuration.preferences.javaScriptCanOpenWindowsAutomatically = true
    configuration.defaultWebpagePreferences.allowsContentJavaScript = true
    configuration.allowsAirPlayForMediaPlayback = false
    #if os(iOS)
      configuration.allowsInlineMediaPlayback = true
      configuration.allowsPictureInPictureMediaPlayback = false
    #endif
    configuration.mediaTypesRequiringUserActionForPlayback = []
    return WKWebView(frame: .zero, configuration: configuration)
  }

  private func attachDelegates(to webView: WKWebView) {
    webView.navigationDelegate = self
    webView.uiDelegate = self
    webView.allowsBackForwardNavigationGestures = true
  }

  private static func isAllowedTransientPopupURL(_ url: URL?) -> Bool {
    guard let url else { return true }
    if url.absoluteString.lowercased() == "about:blank" { return true }
    return sanitizedPublicHTTPSURL(url) != nil
  }

  /// Blocks only high-confidence advertising landings. The embedded player,
  /// its cross-origin frames, and their HLS requests continue to use ordinary
  /// WebKit navigation in compatibility mode.
  private static func isHighConfidenceAdvertisementNavigationURL(_ url: URL?) -> Bool {
    guard let url,
      let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
      let host = components.host?.lowercased()
    else { return false }

    let advertisementHostSuffixes = [
      "turnhub.net",
      "tsyndicate.com",
      "javhd-trk.com",
      "nettrck.store",
      "doubleclick.net",
      "googlesyndication.com",
      "popads.net",
      "popcash.net",
    ]
    if advertisementHostSuffixes.contains(where: {
      host == $0 || host.hasSuffix(".\($0)")
    }) {
      return true
    }
    let path = components.percentEncodedPath.lowercased()
    let queryNames = Set((components.queryItems ?? []).map { $0.name.lowercased() })
    if path.contains("/api/click"),
      queryNames.contains("url")
        || queryNames.contains("u")
        || queryNames.contains("target")
    {
      return true
    }
    if queryNames.contains("clickid"),
      queryNames.contains("affid") || queryNames.contains("aff_id")
    {
      return true
    }
    return false
  }

  /// Media requests from this CDN were observed only in the small floating
  /// advert player, not in the page's programme player.
  private static func isHighConfidenceAdvertisementMediaURL(_ url: URL?) -> Bool {
    guard let host = url?.host?.lowercased() else { return false }
    return host == "saawsedge.com" || host.hasSuffix(".saawsedge.com")
  }

  private func retireTransientPopup(_ expectedWebView: WKWebView? = nil) {
    if let expectedWebView, transientPopupWebView !== expectedWebView { return }
    transientPopupRetirementTask?.cancel()
    transientPopupRetirementTask = nil
    transientPopupWebView?.stopLoading()
    transientPopupWebView?.navigationDelegate = nil
    transientPopupWebView?.uiDelegate = nil
    transientPopupWebView = nil
    transientPopupCoordinator = nil
    hasOpenedPage = false
  }

  private func retainTransientPopup(
    _ popupWebView: WKWebView,
    coordinator: IPadTransientPopupCoordinator,
    expires: Bool = true
  ) {
    retireTransientPopup()
    hasOpenedPage = false
    transientPopupWebView = popupWebView
    transientPopupCoordinator = coordinator
    if expires {
      scheduleTransientPopupRetirement(for: popupWebView)
    }
  }

  private func scheduleTransientPopupRetirement(for popupWebView: WKWebView) {
    transientPopupRetirementTask?.cancel()
    transientPopupRetirementTask = Task { @MainActor [weak self, weak popupWebView] in
      do {
        try await Task.sleep(
          nanoseconds: Self.transientPopupLifetimeNanoseconds
        )
      } catch {
        return
      }
      guard let self, let popupWebView,
        self.transientPopupWebView === popupWebView
      else { return }
      self.retireTransientPopup(popupWebView)
    }
  }

  private func markTransientPopupReady(_ popupWebView: WKWebView) {
    guard transientPopupWebView === popupWebView,
      let popupURL = Self.sanitizedPublicHTTPSURL(popupWebView.url)
    else { return }
    if Self.isHighConfidenceAdvertisementNavigationURL(popupURL) {
      retireTransientPopup(popupWebView)
      statusMessage =
        "広告ページへの移動を停止しました。元のページで再生操作を続けてください。"
      return
    }
    if Self.relaxedWebCompatibilityEnabled {
      hasOpenedPage = true
      showOpenedPage()
      return
    }
    scheduleTransientPopupRetirement(for: popupWebView)
    hasOpenedPage = true
    let host = popupWebView.url?.host ?? "別ページ"
    statusMessage =
      "別ページ（\(host)）が開きました。元ページが進まない場合は「開いたページ」を表示してください。"
  }

  func showOpenedPage() {
    guard let openedWebView = transientPopupWebView,
      !openedWebView.isLoading,
      Self.sanitizedPublicHTTPSURL(openedWebView.url) != nil
    else {
      statusMessage = "開いたページの読み込み完了を待っています…"
      return
    }
    transientPopupRetirementTask?.cancel()
    transientPopupRetirementTask = nil
    transientPopupWebView = nil
    transientPopupCoordinator = nil
    hasOpenedPage = false

    let opener = webView
    if openingPageWebView == nil {
      openingPageWebView = opener
    } else {
      // A second-stage player may open another real window. Retain its direct
      // opener so WebKit's WindowProxy relationship remains valid, while the
      // return button still points to the user's original page.
      retainedPopupOpenerWebViews.append(opener)
    }
    // Once hidden, this WebView must not keep sending delegate callbacks to
    // the controller for the newly visible page. With no delegate, WebKit's
    // normal navigation remains available until the opener is shown again.
    opener.navigationDelegate = nil
    opener.uiDelegate = nil
    canReturnToOpeningPage = true
    adoptSettledWebView(openedWebView, status: "開いたページを表示しました。次の再生ボタンを押してください。")
  }

  func returnToOpeningPage() {
    retireTransientPopup()
    guard let opener = openingPageWebView,
      Self.sanitizedPublicHTTPSURL(opener.url) != nil
    else { return }
    if opener.isLoading {
      opener.stopLoading()
    }
    let openedWebView = webView
    openedWebView.evaluateJavaScript(
      "document.querySelectorAll('video,audio').forEach(element => { try { element.pause(); } catch (_) {} });"
    )
    openedWebView.stopLoading()
    openedWebView.navigationDelegate = nil
    openedWebView.uiDelegate = nil
    for intermediateOpener in retainedPopupOpenerWebViews {
      intermediateOpener.evaluateJavaScript(
        "document.querySelectorAll('video,audio').forEach(element => { try { element.pause(); } catch (_) {} });"
      )
      intermediateOpener.stopLoading()
      intermediateOpener.navigationDelegate = nil
      intermediateOpener.uiDelegate = nil
    }
    retainedPopupOpenerWebViews.removeAll()
    openingPageWebView = nil
    canReturnToOpeningPage = false
    adoptSettledWebView(opener, status: "元のページへ戻りました。")
  }

  private func adoptSettledWebView(_ settledWebView: WKWebView, status: String) {
    quietTask?.cancel()
    quietTask = nil
    challengeCompatibilityTask?.cancel()
    challengeCompatibilityTask = nil
    challengeCompatibilityTimedOut = false
    pendingSuccessfulMainResponseURL = nil
    activeNavigation = nil
    resetSameDocumentNavigationState()
    acceptingScriptCandidates = false
    authorizedRouteToken = nil
    authorizedRouteURL = nil
    authorizedMainResponseURLs.removeAll()
    clearSubframeNavigationState()
    mainDocumentToken = nil
    navigationRefererURL = nil
    lastPageLocationToken = settledWebView.url?.absoluteString
    mainNavigationRedirectCount = 0
    navigationGeneration = nextGeneration(after: navigationGeneration)
    initialChildChallengeFallbackGeneration = nil
    pendingInitialUserActivatedChildDecisions = 0
    initialUserActivatedChildNavigationCount = 0
    challengeLocalFrameNavigationCount = 0
    nativeChallengeFrameNavigationCount = 0
    pendingChallengePageURL = nil
    inspectionRequested = true
    mainFrameChallengeResponse = false
    mainFrameChallengeResponseGeneration = nil
    knownFrames.removeAll()
    candidates.removeAll()
    candidateRevision = 0
    clearMediaSourceState()
    nextDiscoveryOrder = 0
    candidateCount = 0
    readyCandidateGeneration = 0
    challengeActive = false
    isLoading = false
    isClosingPage = false

    webView = settledWebView
    attachDelegates(to: settledWebView)
    if let safeURL = Self.sanitizedPublicHTTPSURL(settledWebView.url) {
      addressText = safeURL.absoluteString
      committedMainDocumentURL = safeURL
    } else {
      committedMainDocumentURL = nil
    }
    pageTitle = settledWebView.title ?? ""
    webViewGeneration = nextGeneration(after: webViewGeneration)
    updateNavigationState()
    noteSuccessfulPublicPageVisit()
    beginInspectionForCurrentRoute()
    statusMessage = status
  }

  func navigate(_ raw: String) {
    guard let url = Self.normalizedPublicHTTPSURL(raw) else {
      statusMessage = "安全な公開HTTPSページを指定してください。"
      return
    }

    navigationRefererURL = nil
    lastPageLocationToken = nil
    resetSameDocumentNavigationState()
    pendingChallengePageURL = nil
    prepareForMainNavigation()
    addressText = url.absoluteString
    statusMessage = "ページを読み込んでいます…"
    var request = URLRequest(
      url: url,
      cachePolicy: .useProtocolCachePolicy,
      timeoutInterval: 60
    )
    request.httpShouldHandleCookies = true
    isLoading = true
    guard let navigation = webView.load(request) else {
      isLoading = false
      statusMessage = "ページ要求を開始できませんでした。"
      return
    }
    activeNavigation = navigation
    updateNavigationState()
  }

  func goBack() {
    guard canGoBack else { return }
    navigationRefererURL = Self.sanitizedPublicHTTPSURL(webView.url)
    prepareForMainNavigation()
    isLoading = true
    activeNavigation = webView.goBack()
    statusMessage = "前のページを読み込んでいます…"
    updateNavigationState()
  }

  func goForward() {
    guard webView.canGoForward else { return }
    navigationRefererURL = Self.sanitizedPublicHTTPSURL(webView.url)
    prepareForMainNavigation()
    isLoading = true
    activeNavigation = webView.goForward()
    statusMessage = "次のページを読み込んでいます…"
    updateNavigationState()
  }

  func reload() {
    guard webView.url != nil else {
      navigate(addressText)
      return
    }
    navigationRefererURL = Self.sanitizedPublicHTTPSURL(webView.url)
    prepareForMainNavigation()
    isLoading = true
    activeNavigation = webView.reload()
    statusMessage = "ページを再読み込みしています…"
    updateNavigationState()
  }

  /// Stops the current navigation without closing or clearing the page.
  func stop() {
    let stoppedDuringChallenge = challengeActive || mainFrameChallengeResponse
    quietTask?.cancel()
    quietTask = nil
    challengeCompatibilityTask?.cancel()
    challengeCompatibilityTask = nil
    challengeCompatibilityTimedOut = false
    resetSameDocumentNavigationState()
    retireTransientPopup()
    webView.stopLoading()
    activeNavigation = nil
    acceptingScriptCandidates = false
    authorizedRouteToken = nil
    authorizedRouteURL = nil
    navigationGeneration = nextGeneration(after: navigationGeneration)
    authorizedMainResponseURLs.removeAll()
    subframeInspectionRouteTokens.removeAll()
    subframeNavigationChains.removeAll()
    pendingSubframeAuthorizations.removeAll()
    completedSubframeAuthorizations.removeAll()
    activeSubframeChallengeChainIDs.removeAll()
    mainFrameScriptChallengeActive = false
    pendingMainScriptChallengeEpoch = nil
    initialChildChallengeDecisionID = nil
    initialChildChallengeFallbackGeneration = nil
    pendingInitialUserActivatedChildDecisions = 0
    initialUserActivatedChildNavigationCount = 0
    transientPopupCreationCount = 0
    challengeLocalFrameNavigationCount = 0
    nativeChallengeFrameNavigationCount = 0
    mainFrameChallengeResponse = false
    mainFrameChallengeResponseGeneration = nil
    pendingSuccessfulMainResponseURL = nil
    isLoading = false
    if stoppedDuringChallenge {
      challengeActive = true
      acceptingScriptCandidates = false
      statusMessage = "確認を停止しました。再読み込みしてください。"
    } else {
      inspectionRequested = true
      beginInspectionForCurrentRoute()
      statusMessage = "読み込みを停止しました。"
    }
    updateNavigationState()
  }

  /// Keeps the live WKWebView, its back/forward list and DOM owned by this
  /// controller while iOS suspends the app. Media is paused, but the page is
  /// deliberately not blanked or replaced.
  func suspendForBackground() {
    webView.evaluateJavaScript(
      "document.querySelectorAll('video,audio').forEach(element => { try { element.pause(); } catch (_) {} });"
    )
    inspectionRequested = currentPublicPageURL != nil
    if currentPublicPageURL != nil {
      statusMessage = "ブラウザのページを保持しています。"
    }
  }

  /// Resumes the retained browser. If iOS reclaimed only WebKit's content
  /// process, reload the same page with the persistent website data store. If
  /// SwiftUI/the app process recreated the controller, the owning view restores
  /// the last safe page after this in-memory resume step.
  func resumeAfterBackground() {
    if webContentProcessTerminated {
      webContentProcessTerminated = false
      if currentPublicPageURL != nil || webView.url != nil {
        reload()
        return
      }
    }

    updateNavigationState()
    guard currentPublicPageURL != nil, !isLoading, !challengeActive else {
      return
    }
    inspectionRequested = true
    beginInspectionForCurrentRoute()
    statusMessage = candidates.isEmpty
      ? "ページを保持したまま配信監視を再開しました。"
      : "ページと配信候補を保持しています。"
  }

  /// Closes the visible page while retaining this controller's standard
  /// website data store. A later `navigate` reuses its cookies in a fresh
  /// WebView, so closed-page history and page cache cannot be reopened.
  func closePage() {
    quietTask?.cancel()
    quietTask = nil
    challengeCompatibilityTask?.cancel()
    challengeCompatibilityTask = nil
    challengeCompatibilityTimedOut = false
    retireTransientPopup()
    let hiddenOpeningWebView = openingPageWebView
    let hiddenIntermediateOpeners = retainedPopupOpenerWebViews
    retainedPopupOpenerWebViews.removeAll()
    openingPageWebView = nil
    canReturnToOpeningPage = false
    inspectionRequested = false
    mainFrameChallengeResponse = false
    mainFrameChallengeResponseGeneration = nil
    knownFrames.removeAll()
    mainDocumentToken = nil
    committedMainDocumentURL = nil
    pendingSuccessfulMainResponseURL = nil
    authorizedRouteToken = nil
    authorizedRouteURL = nil
    authorizedMainResponseURLs.removeAll()
    clearSubframeNavigationState()
    initialChildChallengeFallbackGeneration = nil
    initialUserActivatedChildNavigationCount = 0
    transientPopupCreationCount = 0
    mainNavigationRedirectCount = 0
    candidates.removeAll()
    candidateRevision = 0
    clearMediaSourceState()
    acceptingScriptCandidates = false
    nextDiscoveryOrder = 0
    candidateCount = 0
    readyCandidateGeneration = 0
    challengeActive = false
    isLoading = false
    canGoBack = false
    canGoForward = false
    addressText = ""
    pageTitle = ""
    statusMessage = nil
    isClosingPage = true

    let closingWebView = webView
    closingWebView.evaluateJavaScript(
      "document.querySelectorAll('video,audio').forEach(element => { try { element.pause(); } catch (_) {} });"
    )
    closingWebView.stopLoading()
    hiddenOpeningWebView?.evaluateJavaScript(
      "document.querySelectorAll('video,audio').forEach(element => { try { element.pause(); } catch (_) {} });"
    )
    hiddenOpeningWebView?.stopLoading()
    hiddenOpeningWebView?.navigationDelegate = nil
    hiddenOpeningWebView?.uiDelegate = nil
    for hiddenOpener in hiddenIntermediateOpeners {
      hiddenOpener.evaluateJavaScript(
        "document.querySelectorAll('video,audio').forEach(element => { try { element.pause(); } catch (_) {} });"
      )
      hiddenOpener.stopLoading()
      hiddenOpener.navigationDelegate = nil
      hiddenOpener.uiDelegate = nil
    }
    activeNavigation = nil
    navigationRefererURL = nil
    lastPageLocationToken = nil
    resetSameDocumentNavigationState()
    pendingChallengePageURL = nil
    closingWebView.navigationDelegate = nil
    closingWebView.uiDelegate = nil
    closingWebView.configuration.userContentController
      .removeScriptMessageHandler(
        forName: Self.messageHandlerName,
        contentWorld: Self.instrumentationContentWorld
      )
    closingWebView.configuration.userContentController.removeAllUserScripts()
    closingWebView.loadHTMLString(Self.blankPageHTML, baseURL: nil)

    let replacement = Self.makeWebView(
      websiteDataStore: websiteDataStore,
      messageHandlerProxy: messageHandlerProxy
    )
    attachDelegates(to: replacement)
    webView = replacement
    webContentProcessTerminated = false
    isClosingPage = false
    navigationGeneration = nextGeneration(after: navigationGeneration)
    initialChildChallengeFallbackGeneration = nil
    challengeLocalFrameNavigationCount = 0
    nativeChallengeFrameNavigationCount = 0
    webViewGeneration = nextGeneration(after: webViewGeneration)
  }

  func activateInspection() {
    inspectionRequested = true
    guard !challengeActive, !isClosingPage, !isLoading else {
      statusMessage = "ブラウザ確認の完了後に配信を解析します。"
      return
    }
    beginInspectionForCurrentRoute()
    statusMessage = "ページ内プレイヤーを解析しています…"
  }

  var currentPublicPageURL: URL? {
    Self.sanitizedPublicHTTPSURL(webView.url)
  }

  var hasVerifiedMediaCandidate: Bool {
    candidateSummaries.contains(where: \.isVerified)
  }

  /// A short, currently playing finite source is commonly a pre-roll. Keep
  /// the visible browser alive until its video element changes source instead
  /// of immediately handing that temporary URL to the restoration player.
  var likelyPreRollWait: PreRollWait? {
    let eligible = mediaSlots.compactMap {
      key, state -> (key: MediaSlotKey, state: MediaSlotState)? in
      guard isMediaDocumentCurrent(key.documentToken),
        state.currentURL != nil || state.hasOpaqueSource,
        !state.isEnded,
        !state.isCompactFloatingOverlay,
        state.isPlaying,
        state.isVisible,
        state.visibilityAttested,
        state.renderedArea >= 4_096,
        let duration = state.duration,
        duration >= 3, duration <= 90,
        state.currentTime < duration - 0.25
      else { return nil }
      return (key, state)
    }
    let mostRelevant = eligible.max { lhs, rhs in
      if lhs.state.isVisible != rhs.state.isVisible {
        return !lhs.state.isVisible
      }
      if lhs.state.renderedArea != rhs.state.renderedArea {
        return lhs.state.renderedArea < rhs.state.renderedArea
      }
      return lhs.state.activationOrder < rhs.state.activationOrder
    }
    guard let mostRelevant, let duration = mostRelevant.state.duration else {
      return nil
    }
    return PreRollWait(
      documentToken: mostRelevant.key.documentToken,
      slotToken: mostRelevant.key.slotToken,
      sourceGeneration: mostRelevant.state.generation,
      remainingSeconds: max(0, duration - mostRelevant.state.currentTime)
    )
  }

  func isPreRollWaitCurrent(_ wait: PreRollWait) -> Bool {
    let key = MediaSlotKey(
      documentToken: wait.documentToken,
      slotToken: wait.slotToken
    )
    guard isMediaDocumentCurrent(key.documentToken),
      let state = mediaSlots[key]
    else { return false }
    return state.generation == wait.sourceGeneration
      && (state.currentURL != nil || state.hasOpaqueSource)
      && !state.isEnded && !state.isCompactFloatingOverlay
      && state.isVisible && state.visibilityAttested
      && state.renderedArea >= 4_096
  }

  /// MSE exposes only `blob:` as video.currentSrc. Associate that opaque,
  /// visibly playing element with recent HLS responses from the exact same
  /// isolated-world document. The association is deliberately narrow: no DOM
  /// hint, sibling frame, stale observation or compact overlay can inherit the
  /// visible-browser resolver policy. If more than one response could belong
  /// to the player, none inherits its duration; the resolver must inspect the
  /// playlists and choose the long-form stream itself.
  private func opaquePlaybackAssociations() -> [OpaquePlaybackAssociation] {
    let now = Date()
    let currentSlots = mediaSlots.compactMap {
      key, state -> (documentToken: String, state: MediaSlotState)? in
      guard isMediaDocumentCurrent(key.documentToken), !state.isEnded,
        state.hasOpaqueSource || state.currentURL != nil,
        now.timeIntervalSince(state.lastObservedAt) <= 5
      else { return nil }
      return (key.documentToken, state)
    }
    let compactDocuments = Set(
      currentSlots.compactMap {
        $0.state.isCompactFloatingOverlay ? $0.documentToken : nil
      }
    )
    let eligibleSlots = currentSlots.filter {
      $0.state.hasOpaqueSource && !$0.state.isCompactFloatingOverlay
        && $0.state.isPlaying && $0.state.isVisible
        && $0.state.visibilityAttested && $0.state.renderedArea >= 4_096
    }
    let slotsByDocument = Dictionary(grouping: eligibleSlots, by: \.documentToken)

    var associatedURLKeys = Set<String>()
    var result: [OpaquePlaybackAssociation] = []
    for (documentToken, slots) in slotsByDocument {
      guard
        let strongestSlot = slots.max(by: {
          $0.state.renderedArea < $1.state.renderedArea
        })
      else { continue }
      let earliestObservation =
        slots.map {
          $0.state.sourceActivatedAt.addingTimeInterval(-5)
        }.min() ?? now
      let latestObservation =
        slots.map {
          $0.state.sourceActivatedAt.addingTimeInterval(12)
        }.max() ?? now
      let matches = candidates.values.filter { candidate in
        candidate.documentToken == documentToken
          && Self.isOpaquePlaybackHLSResponse(candidate)
          && candidate.observedAt >= earliestObservation
          && candidate.observedAt <= latestObservation
          && !floatingAdvertisementURLKeys.contains(candidate.url.absoluteString)
      }.sorted { lhs, rhs in
        if lhs.observedAt != rhs.observedAt {
          return lhs.observedAt > rhs.observedAt
        }
        return lhs.discoveryOrder > rhs.discoveryOrder
      }.prefix(8)
      let includesObservedDuration =
        slots.count == 1 && matches.count == 1
        && !compactDocuments.contains(documentToken)
      for match in matches
      where associatedURLKeys.insert(
        match.url.absoluteString
      ).inserted {
        result.append(
          OpaquePlaybackAssociation(
            candidate: match,
            state: strongestSlot.state,
            includesObservedDuration: includesObservedDuration
          )
        )
      }
    }
    return result
  }

  private static func isOpaquePlaybackHLSResponse(_ candidate: Candidate) -> Bool {
    guard candidate.provenance == .script,
      candidate.documentToken != nil
    else { return false }
    switch candidate.sourceKind.lowercased() {
    case "page-fetch-hls-response", "page-xhr-hls-response",
      "fetch-media-response", "xhr-media-response":
      return true
    default:
      return false
    }
  }

  func snapshotCandidates() async -> [IPadWebMediaCandidate] {
    guard !isClosingPage, !challengeActive else { return [] }
    if let authorizedRouteURL,
      webView.url?.absoluteString != authorizedRouteURL
    {
      if !challengeActive, !isLoading,
        let currentURL = webView.url,
        Self.sanitizedPublicHTTPSURL(currentURL) != nil
      {
        processSameDocumentPageLocation(currentURL.absoluteString)
      }
      return []
    }
    let snapshotWebView = webView
    let snapshotNavigationGeneration = navigationGeneration
    let snapshotRouteURL = authorizedRouteURL
    let snapshotPageURL = webView.url?.absoluteString

    let userAgent: String?
    do {
      userAgent = try await webView.evaluateJavaScript("navigator.userAgent") as? String
    } catch {
      userAgent = webView.customUserAgent
    }
    let requestCookies = await allRequestCookies()

    guard webView === snapshotWebView,
      !isClosingPage, !challengeActive,
      navigationGeneration == snapshotNavigationGeneration,
      authorizedRouteURL == snapshotRouteURL,
      webView.url?.absoluteString == snapshotPageURL
    else {
      if !isClosingPage, !challengeActive, !isLoading,
        let currentURL = webView.url,
        Self.sanitizedPublicHTTPSURL(currentURL) != nil
      {
        processSameDocumentPageLocation(currentURL.absoluteString)
      }
      return []
    }

    let activeStates = mediaSlots.compactMap { key, state -> MediaSlotState? in
      guard isMediaDocumentCurrent(key.documentToken), state.currentURL != nil,
        !state.isEnded, !state.isCompactFloatingOverlay
      else {
        return nil
      }
      return state
    }.sorted { lhs, rhs in
      if lhs.isVisible != rhs.isVisible { return lhs.isVisible }
      if lhs.isPlaying != rhs.isPlaying { return lhs.isPlaying }
      if lhs.renderedArea != rhs.renderedArea {
        return lhs.renderedArea > rhs.renderedArea
      }
      return lhs.activationOrder > rhs.activationOrder
    }
    let opaqueAssociations = opaquePlaybackAssociations()
    var activeURLKeys = Set(
      activeStates.compactMap { $0.currentURL?.absoluteString }
    )
    activeURLKeys.formUnion(
      opaqueAssociations.map { $0.candidate.url.absoluteString }
    )
    let credentialAuthorizedURLKeys = Set(
      activeStates.compactMap { state -> String? in
        guard state.isPlaying, state.isVisible, state.visibilityAttested,
          state.renderedArea >= 4_096
        else { return nil }
        return state.currentURL?.absoluteString
      }
    ).union(
      opaqueAssociations.map { $0.candidate.url.absoluteString }
    )
    let supersededURLKeys = currentSourceHistory.subtracting(activeURLKeys)

    var snapshot = Array(candidates.values)
    var rankedActiveURLs = Set<String>()
    var mediaEvidenceByURL: [String: IPadBrowserMediaEvidence] = [:]
    for association in opaqueAssociations {
      mediaEvidenceByURL[association.candidate.url.absoluteString] =
        Self.browserMediaEvidence(
          for: association.state,
          includesObservedDuration: association.includesObservedDuration
        )
    }
    for (rank, state) in activeStates.enumerated() {
      guard let activeURL = state.currentURL,
        rankedActiveURLs.insert(activeURL.absoluteString).inserted
      else { continue }
      mediaEvidenceByURL[activeURL.absoluteString] = Self.browserMediaEvidence(
        for: state,
        includesObservedDuration: true
      )
      let observed = Candidate(
        url: activeURL,
        sourceKind: "active-current-source",
        frameDepth: state.frameDepth,
        discoveryOrder: state.activationOrder,
        frameURL: state.frameURL,
        provenance: .script,
        activeRank: rank
      )
      if let index = snapshot.firstIndex(where: { $0.url == activeURL }) {
        let existing = snapshot[index]
        let keepNativeProvenance = existing.provenance != .script
        snapshot[index] = Candidate(
          url: activeURL,
          sourceKind: observed.sourceKind,
          frameDepth: observed.frameDepth,
          discoveryOrder: observed.discoveryOrder,
          frameURL: keepNativeProvenance ? existing.frameURL : observed.frameURL,
          provenance: keepNativeProvenance ? existing.provenance : observed.provenance,
          activeRank: rank
        )
      } else if snapshot.count < Self.maximumCandidateCount {
        snapshot.append(observed)
      } else if let worstIndex = Self.worstCandidateIndex(in: snapshot) {
        snapshot[worstIndex] = observed
      }
    }
    if !challengeActive, let currentURL = Self.sanitizedPublicHTTPSURL(webView.url),
      !Self.isInteractionChallenge(currentURL)
    {
      let currentCandidate = Candidate(
        url: currentURL,
        sourceKind: "page",
        frameDepth: 0,
        discoveryOrder: nextDiscoveryOrder,
        frameURL: currentURL,
        provenance: .currentPage
      )
      if let index = snapshot.firstIndex(where: { $0.url == currentURL }) {
        if Self.isPreferred(currentCandidate, over: snapshot[index]) {
          snapshot[index] = currentCandidate
        }
      } else if snapshot.count < Self.maximumCandidateCount {
        snapshot.append(currentCandidate)
      } else if let worstIndex = Self.worstCandidateIndex(in: snapshot) {
        snapshot[worstIndex] = currentCandidate
      }
    }

    let fallbackFrameURL = Self.sanitizedPublicHTTPSURL(webView.url)
    return Self.resolutionOrder(
      snapshot,
      activeURLKeys: activeURLKeys,
      supersededURLKeys: supersededURLKeys,
      excludedURLKeys: floatingAdvertisementURLKeys
    ).map { candidate in
      let relevantURLs = [candidate.url, candidate.frameURL, fallbackFrameURL].compactMap {
        $0
      }
      let prioritizedCookies = Self.prioritizedCookies(
        requestCookies,
        for: relevantURLs
      )
      let remainingCookies = requestCookies.filter {
        !prioritizedCookies.contains($0)
      }
      let refererFrameURL =
        candidate.provenance == .mainNavigationResponse
          || candidate.provenance == .subframeNavigationResponse
        ? candidate.frameURL : (candidate.frameURL ?? fallbackFrameURL)
      let cookieSourceURL: URL? =
        candidate.provenance == .mainNavigationResponse
          || candidate.provenance == .currentPage
        ? candidate.url : refererFrameURL
      let allowsCrossSiteCredentialReplay =
        candidate.provenance != .script
        || (Self.relaxedWebCompatibilityEnabled
          && (credentialAuthorizedURLKeys.contains(candidate.url.absoluteString)
            || Self.isVerifiedPageHLSObservation(candidate.sourceKind)))
      return IPadWebMediaCandidate(
        url: candidate.url,
        requestContext: IPadMediaRequestContext(
          cookies: Array(
            (prioritizedCookies + remainingCookies).prefix(128)
          ),
          userAgent: userAgent,
          referer: Self.referer(
            for: candidate.url,
            frameURL: refererFrameURL
          ),
          origin: Self.requiresBrowserOriginHeader(candidate.sourceKind)
            ? refererFrameURL : nil,
          cookieSourceURL: cookieSourceURL,
          allowsCredentialReplay: true,
          allowsCrossSiteCredentialReplay: allowsCrossSiteCredentialReplay
        ),
        interactionPageURL: fallbackFrameURL,
        selectionState:
          activeURLKeys.contains(candidate.url.absoluteString)
          ? .activeCurrentSource
          : supersededURLKeys.contains(candidate.url.absoluteString)
            ? .supersededCurrentSource : .ordinary,
        discoveryRole:
          activeURLKeys.contains(candidate.url.absoluteString)
          ? .activePlayback
          : Self.isReadyCandidate(candidate)
            ? .verifiedMediaResponse
            : Self.isUnverifiedMediaResponseHint(candidate)
              ? .unverifiedMediaResponse
              : Self.isDirectMediaCandidate(candidate.url)
                ? .directHint : .pageLead,
        mediaEvidence: mediaEvidenceByURL[candidate.url.absoluteString]
      )
    }
  }

  private static func browserMediaEvidence(
    for state: MediaSlotState,
    includesObservedDuration: Bool
  ) -> IPadBrowserMediaEvidence {
    IPadBrowserMediaEvidence(
      observedDuration: includesObservedDuration ? state.duration : nil,
      isPlaying: state.isPlaying,
      isVisible: state.isVisible,
      visibilityAttested: state.visibilityAttested,
      renderedArea: state.renderedArea,
      sourceGeneration: state.generation,
      activationOrder: state.activationOrder
    )
  }

  /// Keeps the on-screen count and list identical. Static iframe/page URLs
  /// remain visible as leads, while sources proved by WebKit are clearly
  /// distinguished from those unverified hints.
  private func refreshCandidateSummaries() {
    let activeStates = mediaSlots.compactMap {
      key, state -> MediaSlotState? in
      guard isMediaDocumentCurrent(key.documentToken), state.currentURL != nil,
        !state.isEnded, !state.isCompactFloatingOverlay
      else {
        return nil
      }
      return state
    }.sorted { lhs, rhs in
      if lhs.isVisible != rhs.isVisible { return lhs.isVisible }
      if lhs.isPlaying != rhs.isPlaying { return lhs.isPlaying }
      if lhs.renderedArea != rhs.renderedArea {
        return lhs.renderedArea > rhs.renderedArea
      }
      return lhs.activationOrder > rhs.activationOrder
    }
    let opaqueAssociations = opaquePlaybackAssociations()
    var activeURLKeys = Set(
      activeStates.compactMap { $0.currentURL?.absoluteString }
    )
    activeURLKeys.formUnion(
      opaqueAssociations.map { $0.candidate.url.absoluteString }
    )
    let supersededURLKeys = currentSourceHistory.subtracting(activeURLKeys)
    var displayed = Array(candidates.values)
    for (rank, state) in activeStates.enumerated() {
      guard let activeURL = state.currentURL else { continue }
      let active = Candidate(
        url: activeURL,
        sourceKind: "active-current-source",
        frameDepth: state.frameDepth,
        discoveryOrder: state.activationOrder,
        frameURL: state.frameURL,
        provenance: .script,
        activeRank: rank
      )
      if let index = displayed.firstIndex(where: { $0.url == activeURL }) {
        displayed[index] = active
      } else {
        displayed.append(active)
      }
    }
    candidateSummaries = Self.resolutionOrder(
      displayed,
      activeURLKeys: activeURLKeys,
      supersededURLKeys: supersededURLKeys,
      excludedURLKeys: floatingAdvertisementURLKeys
    ).map { candidate in
      let key = candidate.url.absoluteString
      let isActive = activeURLKeys.contains(key)
      let isSuperseded = supersededURLKeys.contains(key)
      let verified = isActive || Self.isReadyCandidate(candidate)
      return CandidateSummary(
        id: key,
        url: key,
        sourceLabel: Self.candidateSourceLabel(candidate.sourceKind),
        verificationLabel:
          isActive
          ? "再生ソース"
          : isSuperseded
            ? "以前のソース"
            : verified ? "応答確認済み" : "未確認候補",
        frameDepth: candidate.frameDepth,
        isVerified: verified
      )
    }
    candidateCount = candidateSummaries.count
  }

  private func prepareForMainNavigation(
    preservingSameDocumentReloadState: Bool = false
  ) {
    retireTransientPopup()
    isClosingPage = false
    lastPageLocationToken = nil
    if !preservingSameDocumentReloadState {
      resetSameDocumentNavigationState()
    }
    pendingSameDocumentPageURL = nil
    pendingChallengePageURL = nil
    acceptingScriptCandidates = false
    authorizedRouteToken = nil
    authorizedRouteURL = nil
    authorizedMainResponseURLs.removeAll()
    clearSubframeNavigationState()
    mainDocumentToken = nil
    committedMainDocumentURL = nil
    pendingSuccessfulMainResponseURL = nil
    mainNavigationRedirectCount = 0
    navigationGeneration = nextGeneration(after: navigationGeneration)
    initialChildChallengeFallbackGeneration = nil
    pendingInitialUserActivatedChildDecisions = 0
    initialUserActivatedChildNavigationCount = 0
    transientPopupCreationCount = 0
    challengeLocalFrameNavigationCount = 0
    nativeChallengeFrameNavigationCount = 0
    quietTask?.cancel()
    quietTask = nil
    challengeCompatibilityTask?.cancel()
    challengeCompatibilityTask = nil
    challengeCompatibilityTimedOut = false
    inspectionRequested = false
    mainFrameChallengeResponse = false
    mainFrameChallengeResponseGeneration = nil
    knownFrames.removeAll()
    candidates.removeAll()
    candidateRevision = 0
    clearMediaSourceState()
    nextDiscoveryOrder = 0
    candidateCount = 0
    readyCandidateGeneration = 0
    challengeActive = false
  }

  private func reloadForSameDocumentNavigation() {
    guard !sameDocumentHardReloadUsed else {
      acceptSameDocumentRouteInPlace()
      return
    }
    sameDocumentHardReloadUsed = true
    sameDocumentHardReloadInFlight = true
    pendingSameDocumentPageURL = nil
    navigationRefererURL = Self.sanitizedPublicHTTPSURL(webView.url)
    prepareForMainNavigation(preservingSameDocumentReloadState: true)
    isLoading = true
    statusMessage = "ページの切り替えを通常読み込みで確定しています…"
    guard let navigation = webView.reload() else {
      sameDocumentHardReloadInFlight = false
      isLoading = false
      statusMessage = "切り替え後のページを読み込めませんでした。"
      return
    }
    activeNavigation = navigation
  }

  private func processSameDocumentPageLocation(_ rawPageURL: String) {
    lastPageLocationToken = rawPageURL
    quietTask?.cancel()
    quietTask = nil
    if isLoading {
      pendingSameDocumentPageURL = rawPageURL
      return
    }
    // A history/hash route change does not replace the document. Keep the
    // document generation and any in-flight child-frame authorization alive;
    // player sites commonly update history and install their real iframe in
    // the same click. Invalidating either here would cancel that second stage.
    acceptSameDocumentRouteInPlace()
  }

  private func acceptSameDocumentRouteInPlace() {
    sameDocumentHardReloadInFlight = false
    pendingSameDocumentPageURL = nil
    quietTask?.cancel()
    quietTask = nil
    if let currentPageURL = webView.url,
      let safeURL = Self.sanitizedPublicHTTPSURL(currentPageURL)
    {
      addressText = safeURL.absoluteString
      // Keep the exact visible route (including its fragment) for the route
      // equality guards below. `sanitizedPublicHTTPSURL` deliberately removes
      // fragments and is therefore suitable for display, but not as the route
      // identity after a hash-only handoff.
      authorizedRouteURL = currentPageURL.absoluteString
    }
    noteSuccessfulPublicPageVisit()
    inspectionRequested = true
    acceptingScriptCandidates = authorizedRouteToken != nil
    candidateRevision = nextGeneration(after: candidateRevision)
    statusMessage = "ページの切り替えを検出しました。配信を再解析しています…"
    if authorizedRouteToken == nil {
      beginInspectionForCurrentRoute()
    } else {
      activateInspectionInKnownFrames()
    }
  }

  private func resetSameDocumentNavigationState() {
    sameDocumentHardReloadUsed = false
    sameDocumentHardReloadInFlight = false
    pendingSameDocumentPageURL = nil
  }

  private func clearMediaSourceState() {
    mediaSlots.removeAll()
    currentSourceHistory.removeAll()
    reportedFloatingAdvertisementURLKeys.removeAll()
    nextMediaActivationOrder = 0
    frameHeartbeatDates.removeAll()
    mediaSourceRevision = 0
    refreshCandidateSummaries()
  }

  private func isMediaDocumentCurrent(_ documentToken: String) -> Bool {
    guard let frame = knownFrames[documentToken] else { return false }
    if frame.isMainFrame { return documentToken == mainDocumentToken }
    guard let heartbeat = frameHeartbeatDates[documentToken] else { return false }
    return Date().timeIntervalSince(heartbeat) <= 5
  }

  private func retireMediaSlots(forDocumentToken documentToken: String) {
    let keys = mediaSlots.keys.filter { $0.documentToken == documentToken }
    guard !keys.isEmpty else { return }
    for key in keys {
      mediaSlots.removeValue(forKey: key)
    }
    frameHeartbeatDates.removeValue(forKey: documentToken)
    mediaSourceRevision = nextGeneration(after: mediaSourceRevision)
    refreshCandidateSummaries()
  }

  private func rememberCurrentSourceURL(_ url: URL) {
    let value = url.absoluteString
    guard !currentSourceHistory.contains(value) else { return }
    if currentSourceHistory.count >= Self.maximumCurrentSourceHistoryCount,
      let oldestAvailable = currentSourceHistory.first
    {
      currentSourceHistory.remove(oldestAvailable)
    }
    currentSourceHistory.insert(value)
  }

  private func nextGeneration(after generation: Int) -> Int {
    generation == Int.max ? 1 : generation + 1
  }

  private func clearSubframeNavigationState() {
    subframeDocumentEpochs.removeAll()
    subframeInspectionRouteTokens.removeAll()
    subframeNavigationChains.removeAll()
    pendingSubframeAuthorizations.removeAll()
    completedSubframeAuthorizations.removeAll()
    activeSubframeChallengeChainIDs.removeAll()
    mainFrameScriptChallengeActive = false
    pendingMainScriptChallengeEpoch = nil
    initialChildChallengeDecisionID = nil
    pendingInitialUserActivatedChildDecisions = 0
  }

  private var hasCurrentNativeMainChallenge: Bool {
    mainFrameChallengeResponse
      && mainFrameChallengeResponseGeneration == navigationGeneration
      && committedMainDocumentURL != nil
  }

  private func beginNativeChallengeCompatibilityWindow() {
    challengeCompatibilityTask?.cancel()
    challengeCompatibilityTimedOut = false
    let generation = navigationGeneration
    challengeCompatibilityTask = Task { @MainActor [weak self] in
      do {
        try await Task.sleep(nanoseconds: 15_000_000_000)
      } catch {
        return
      }
      guard let self,
        generation == self.navigationGeneration,
        self.hasCurrentNativeMainChallenge,
        self.challengeActive,
        !self.isClosingPage
      else { return }
      self.challengeCompatibilityTimedOut = true
      self.statusMessage =
        "Cloudflare確認を完了できません。このサイトはアプリ内ブラウザ非対応の可能性があります（HTTPS \(self.nativeChallengeFrameNavigationCount)件・local \(self.challengeLocalFrameNavigationCount)件）。"
    }
  }

  private func setChallengeWaitingStatus() {
    guard !challengeCompatibilityTimedOut else { return }
    statusMessage =
      "Cloudflareの確認を待っています。チェックボックスは必要な場合だけ表示されます。"
  }

  private func completeChallengeIfPossible() {
    guard challengeActive, !mainFrameChallengeResponse,
      !mainFrameScriptChallengeActive,
      activeSubframeChallengeChainIDs.isEmpty
    else { return }
    challengeCompatibilityTask?.cancel()
    challengeCompatibilityTask = nil
    challengeCompatibilityTimedOut = false
    challengeActive = false
    inspectionRequested = true
    acceptingScriptCandidates = false
    statusMessage = "確認が完了しました。ページ内プレイヤーを解析しています…"
    if let pendingPageURL = pendingChallengePageURL {
      pendingChallengePageURL = nil
      processSameDocumentPageLocation(pendingPageURL)
      return
    }
    beginInspectionForCurrentRoute()
    if candidates.values.contains(where: Self.isReadyCandidate) {
      scheduleReadyCandidate()
    }
  }

  private func receiveScriptMessage(_ message: WKScriptMessage) {
    guard !isClosingPage else { return }
    guard let body = message.body as? [String: Any],
      let messageType = body["type"] as? String,
      let documentToken = body["documentToken"] as? String,
      !documentToken.isEmpty,
      documentToken.utf8.count <= Self.maximumScriptTokenLength
    else { return }

    if messageType == "frame-ready" {
      verifyFrameReady(
        documentToken,
        frameInfo: message.frameInfo,
        reportedRouteToken: body["routeToken"] as? String
      )
      return
    }
    if messageType == "frame-heartbeat" {
      guard let knownFrame = knownFrames[documentToken],
        body["routeToken"] as? String == authorizedRouteToken,
        knownFrame.isMainFrame == message.frameInfo.isMainFrame,
        knownFrame.request.url?.absoluteString
          == message.frameInfo.request.url?.absoluteString
      else { return }
      frameHeartbeatDates[documentToken] = Date()
      return
    }
    guard knownFrames[documentToken] != nil else { return }
    if message.frameInfo.isMainFrame, documentToken != mainDocumentToken {
      return
    }
    if Self.relaxedWebCompatibilityEnabled,
      messageType == "challenge-hint" || messageType == "challenge-cleared"
    {
      return
    }
    if messageType == "challenge-hint", message.frameInfo.isMainFrame {
      let challengeEpoch = body["challengeEpoch"] as? String
      pendingMainScriptChallengeEpoch =
        challengeEpoch.flatMap { epoch in
          !epoch.isEmpty && epoch.utf8.count <= Self.maximumScriptTokenLength
            ? epoch : nil
        }
      mainFrameScriptChallengeActive = true
      challengeActive = true
      acceptingScriptCandidates = false
      quietTask?.cancel()
      quietTask = nil
      setChallengeWaitingStatus()
      return
    }
    if messageType == "challenge-cleared", message.frameInfo.isMainFrame,
      body["completed"] as? Bool == true,
      let challengeEpoch = body["challengeEpoch"] as? String,
      challengeEpoch == pendingMainScriptChallengeEpoch
    {
      pendingMainScriptChallengeEpoch = nil
      mainFrameScriptChallengeActive = false
      completeChallengeIfPossible()
      return
    }
    if messageType == "page-location", message.frameInfo.isMainFrame,
      let rawPageURL = body["url"] as? String,
      rawPageURL.utf8.count <= Self.maximumCandidateURLLength,
      let reportedURL = URL(string: rawPageURL),
      let currentWebViewURL = webView.url,
      let safeReportedURL = Self.sanitizedPublicHTTPSURL(reportedURL),
      let currentURL = Self.sanitizedPublicHTTPSURL(currentWebViewURL),
      Self.isSameOrigin(safeReportedURL, currentURL),
      safeReportedURL.absoluteString == currentURL.absoluteString,
      reportedURL.absoluteString == currentWebViewURL.absoluteString,
      rawPageURL != lastPageLocationToken
    {
      if challengeActive || mainFrameChallengeResponse {
        pendingChallengePageURL = rawPageURL
        return
      }
      processSameDocumentPageLocation(rawPageURL)
      return
    }
    guard acceptingScriptCandidates,
      let routeToken = body["routeToken"] as? String,
      routeToken == authorizedRouteToken,
      let authorizedRouteURL,
      authorizedRouteURL == webView.url?.absoluteString
    else { return }

    if messageType == "media-source" {
      receiveMediaSourceMessage(
        body,
        documentToken: documentToken,
        frameInfo: message.frameInfo
      )
      return
    }

    guard messageType == "candidate",
      let rawURL = body["url"] as? String,
      rawURL.utf8.count <= Self.maximumCandidateURLLength,
      let url = URL(string: rawURL).flatMap(Self.sanitizedPublicHTTPSURL),
      !Self.isInteractionChallenge(url)
    else { return }

    let sourceKind = (body["kind"] as? String) ?? "unknown"
    if (body["isCompactFloatingOverlay"] as? Bool) == true {
      let key = url.absoluteString
      if reportedFloatingAdvertisementURLKeys.count
        >= Self.maximumCurrentSourceHistoryCount,
        let oldest = reportedFloatingAdvertisementURLKeys.first
      {
        reportedFloatingAdvertisementURLKeys.remove(oldest)
      }
      let inserted = reportedFloatingAdvertisementURLKeys.insert(key).inserted
      let removed = candidates.removeValue(forKey: key) != nil
      if inserted || removed {
        candidateRevision = nextGeneration(after: candidateRevision)
        refreshCandidateSummaries()
      }
      return
    }
    let frameDepth = min(
      max(0, (body["frameDepth"] as? Int) ?? (message.frameInfo.isMainFrame ? 0 : 1)),
      Self.maximumFrameDepth
    )
    let trustedFrameURL = Self.sanitizedPublicHTTPSURL(message.frameInfo.request.url)
    insertCandidate(
      Candidate(
        url: url,
        sourceKind: sourceKind,
        frameDepth: frameDepth,
        discoveryOrder: nextDiscoveryOrder,
        frameURL: trustedFrameURL,
        provenance: .script,
        documentToken: documentToken
      )
    )
  }

  private func receiveMediaSourceMessage(
    _ body: [String: Any],
    documentToken: String,
    frameInfo: WKFrameInfo
  ) {
    guard let stateName = body["state"] as? String else { return }
    let isKnownSourceState =
      stateName == "current" || stateName == "cleared"
    guard isKnownSourceState || stateName == "opaque",
      let slotToken = body["slotToken"] as? String,
      !slotToken.isEmpty,
      slotToken.utf8.count <= Self.maximumMediaSlotTokenLength,
      slotToken.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }),
      let generationNumber = body["sourceGeneration"] as? NSNumber,
      generationNumber.doubleValue.rounded() == generationNumber.doubleValue,
      generationNumber.doubleValue >= 1,
      generationNumber.doubleValue <= 1_000_000
    else { return }

    let generation = generationNumber.intValue
    let key = MediaSlotKey(
      documentToken: documentToken,
      slotToken: slotToken
    )
    let existing = mediaSlots[key]
    if existing == nil, mediaSlots.count >= Self.maximumMediaSlotCount { return }
    if let existing, generation < existing.generation { return }

    let currentURL: URL?
    if stateName == "current" {
      guard let rawURL = body["url"] as? String,
        rawURL.utf8.count <= Self.maximumCandidateURLLength,
        let safeURL = URL(string: rawURL).flatMap(Self.sanitizedPublicHTTPSURL),
        !Self.isInteractionChallenge(safeURL),
        !Self.isHighConfidenceAdvertisementMediaURL(safeURL)
      else { return }
      currentURL = safeURL
    } else {
      currentURL = nil
    }
    let hasOpaqueSource = stateName == "opaque"
    if let existing, generation == existing.generation,
      existing.currentURL != currentURL
    {
      return
    }

    func finiteNumber(
      _ value: Any?,
      minimum: Double,
      maximum: Double
    ) -> Double? {
      guard let number = value as? NSNumber else { return nil }
      let result = number.doubleValue
      guard result.isFinite, result >= minimum, result <= maximum else {
        return nil
      }
      return result
    }

    let frameDepth = min(
      max(
        0,
        (body["frameDepth"] as? NSNumber)?.intValue
          ?? (frameInfo.isMainFrame ? 0 : 1)),
      Self.maximumFrameDepth
    )
    let duration = finiteNumber(
      body["duration"],
      minimum: 0.001,
      maximum: 7 * 24 * 60 * 60
    )
    let currentTime =
      finiteNumber(
        body["currentTime"],
        minimum: 0,
        maximum: 7 * 24 * 60 * 60
      ) ?? 0
    let renderedArea = Int(
      finiteNumber(
        body["renderedArea"],
        minimum: 0,
        maximum: 100_000_000
      ) ?? 0
    )
    let isPlaying = (body["isPlaying"] as? Bool) ?? false
    let isEnded = (body["isEnded"] as? Bool) ?? false
    let isCompactFloatingOverlay =
      (body["isCompactFloatingOverlay"] as? Bool) ?? false
    let sourceChanged = existing?.generation != generation
    let floatingClassificationChanged =
      existing?.isCompactFloatingOverlay != isCompactFloatingOverlay
    let isStrongVisiblePlayback =
      (currentURL != nil || hasOpaqueSource) && isPlaying && !isEnded
      && !isCompactFloatingOverlay
      && (body["isVisible"] as? Bool) == true
      && (body["visibilityAttested"] as? Bool) == true
      && renderedArea >= 4_096
    let wasStrongVisiblePlayback =
      existing.map {
        ($0.currentURL != nil || $0.hasOpaqueSource) && $0.isPlaying && !$0.isEnded
          && !$0.isCompactFloatingOverlay && $0.isVisible
          && $0.visibilityAttested && $0.renderedArea >= 4_096
      } ?? false
    if sourceChanged {
      if let oldURL = existing?.currentURL {
        rememberCurrentSourceURL(oldURL)
      }
      if let currentURL {
        rememberCurrentSourceURL(currentURL)
      }
      nextMediaActivationOrder = nextGeneration(after: nextMediaActivationOrder)
      mediaSourceRevision = nextGeneration(after: mediaSourceRevision)
    } else if existing?.isEnded != isEnded {
      // A pre-roll may retain currentSrc after ending. Treat that terminal
      // transition as a source revision so an in-flight selection cannot
      // finalize the expired advert.
      mediaSourceRevision = nextGeneration(after: mediaSourceRevision)
    } else if floatingClassificationChanged {
      // Re-rank immediately when a player is docked into, or removed from, a
      // compact floating overlay without changing currentSrc.
      mediaSourceRevision = nextGeneration(after: mediaSourceRevision)
    } else if isStrongVisiblePlayback != wasStrongVisiblePlayback {
      // MSE commonly assigns blob: before `playing` and visual visibility are
      // established. Wake the analyzer at that eligibility edge rather than
      // waiting for its periodic safety retry.
      mediaSourceRevision = nextGeneration(after: mediaSourceRevision)
    }
    mediaSlots[key] = MediaSlotState(
      generation: generation,
      currentURL: currentURL,
      frameDepth: frameDepth,
      activationOrder: sourceChanged
        ? nextMediaActivationOrder : (existing?.activationOrder ?? 0),
      frameURL: Self.sanitizedPublicHTTPSURL(frameInfo.request.url),
      duration: sourceChanged ? duration : (duration ?? existing?.duration),
      currentTime: currentTime,
      isPlaying: isPlaying,
      isEnded: isEnded,
      isVisible: (body["isVisible"] as? Bool) ?? false,
      visibilityAttested: (body["visibilityAttested"] as? Bool) ?? false,
      renderedArea: renderedArea,
      isCompactFloatingOverlay: isCompactFloatingOverlay,
      hasOpaqueSource: hasOpaqueSource,
      sourceActivatedAt:
        sourceChanged ? Date() : (existing?.sourceActivatedAt ?? Date()),
      lastObservedAt: Date()
    )
    refreshCandidateSummaries()
    if sourceChanged, currentURL != nil || hasOpaqueSource {
      scheduleReadyCandidate()
    }
  }

  private func verifyFrameReady(
    _ documentToken: String,
    frameInfo: WKFrameInfo,
    reportedRouteToken: String?
  ) {
    let generation = navigationGeneration
    let documentEpoch: Int?
    let completedAuthorizationID: String?
    if frameInfo.isMainFrame {
      documentEpoch = nil
      completedAuthorizationID = nil
    } else {
      documentEpoch = subframeDocumentEpochs[documentToken]
      let frameURL = Self.sanitizedPublicHTTPSURL(frameInfo.request.url)?.absoluteString
      completedAuthorizationID =
        completedSubframeAuthorizations.first {
          $0.destinationURL == frameURL
            && $0.chain.mainNavigationGeneration == generation
        }?.id
    }
    webView.callAsyncJavaScript(
      "return window.__miohInteractiveDocumentToken ?? null;",
      arguments: [:],
      in: frameInfo,
      in: Self.instrumentationContentWorld
    ) { [weak self] result in
      Task { @MainActor [weak self] in
        guard let self, generation == self.navigationGeneration,
          case .success(let value) = result,
          value as? String == documentToken,
          self.knownFrames[documentToken] != nil
            || self.knownFrames.count < Self.maximumKnownFrameCount
        else { return }
        if Self.relaxedWebCompatibilityEnabled {
          if frameInfo.isMainFrame {
            guard let currentURL = Self.sanitizedPublicHTTPSURL(self.webView.url),
              Self.sanitizedPublicHTTPSURL(frameInfo.request.url)?.absoluteString
                == currentURL.absoluteString
            else { return }
            self.mainDocumentToken = documentToken
          } else {
            guard frameInfo.request.url != nil else { return }
          }
          self.knownFrames[documentToken] = frameInfo
          self.frameHeartbeatDates[documentToken] = Date()
          let frameAlreadyAuthorized =
            self.authorizedRouteToken != nil
            && reportedRouteToken == self.authorizedRouteToken
          if self.inspectionRequested, !self.challengeActive {
            if frameAlreadyAuthorized, let routeToken = self.authorizedRouteToken {
              // The first candidate burst may have arrived while the
              // document-token proof was still in flight (notably after an
              // already-loaded popup is adopted). Replay the bounded DOM and
              // resource cache once, without reinstalling any network hook.
              self.webView.callAsyncJavaScript(
                "return window.__miohInteractiveRescanRegisteredFrame?.(routeToken) === true;",
                arguments: ["routeToken": routeToken],
                in: frameInfo,
                in: Self.instrumentationContentWorld
              ) { _ in }
            } else {
              self.activateInspection(in: frameInfo)
            }
          }
          return
        }
        if frameInfo.isMainFrame {
          guard let committedURL = self.committedMainDocumentURL,
            Self.sanitizedPublicHTTPSURL(frameInfo.request.url)?.absoluteString
              == committedURL.absoluteString
          else { return }
        } else {
          let currentDocumentEpoch = self.subframeDocumentEpochs[documentToken]
          if let documentEpoch {
            guard currentDocumentEpoch == documentEpoch else { return }
          } else {
            // A target-frame navigation installs/increments this epoch before
            // WebKit is allowed to proceed. Therefore an epoch appearing while
            // this proof was in flight makes the callback stale.
            guard currentDocumentEpoch == nil else { return }
          }
          guard self.mainDocumentToken != nil,
            let frameURL = Self.sanitizedPublicHTTPSURL(frameInfo.request.url)
          else { return }
          if let knownFrame = self.knownFrames[documentToken] {
            guard
              Self.sanitizedPublicHTTPSURL(knownFrame.request.url)?.absoluteString
                == frameURL.absoluteString
            else { return }
          } else {
            guard let completedAuthorizationID,
              documentEpoch != nil
                || self.subframeDocumentEpochs.count < Self.maximumKnownFrameCount,
              let authorizationIndex = self.completedSubframeAuthorizations.firstIndex(where: {
                $0.id == completedAuthorizationID
                  && $0.destinationURL == frameURL.absoluteString
                  && $0.chain.mainNavigationGeneration == generation
              })
            else { return }
            let authorization = self.completedSubframeAuthorizations[authorizationIndex]
            guard self.authorizationInitiatorIsCurrent(authorization) else {
              let staleAuthorization = self.completedSubframeAuthorizations.remove(
                at: authorizationIndex
              )
              self.revokeSubframeAuthorization(staleAuthorization)
              return
            }
            self.completedSubframeAuthorizations.remove(at: authorizationIndex)
            let verifiedDocumentEpoch =
              documentEpoch ?? authorization.chain.sourceDocumentEpoch
            if documentEpoch == nil {
              self.subframeDocumentEpochs[documentToken] = verifiedDocumentEpoch
            }
            if let targetPriorDocumentToken = authorization.targetPriorDocumentToken {
              self.retireMediaSlots(forDocumentToken: targetPriorDocumentToken)
              self.subframeNavigationChains.removeValue(
                forKey: targetPriorDocumentToken
              )
            }
            switch authorization.chain.authority {
            case .challenge(let chainID):
              if authorization.retiresSubframeChallengeOnCommit {
                self.activeSubframeChallengeChainIDs.remove(chainID)
              } else {
                var transferredChain = authorization.chain
                transferredChain.sourceDocumentEpoch = verifiedDocumentEpoch
                self.subframeNavigationChains[documentToken] = transferredChain
              }
            case .inspection:
              var transferredChain = authorization.chain
              transferredChain.sourceDocumentEpoch = verifiedDocumentEpoch
              self.subframeNavigationChains[documentToken] = transferredChain
            case .pageLoad:
              // Once the main document settles, this frame is re-authorized
              // with the current inspection route rather than carrying a
              // page-load-only authority into later user interaction.
              self.subframeNavigationChains.removeValue(forKey: documentToken)
            }
          }
        }
        self.knownFrames[documentToken] = frameInfo
        self.frameHeartbeatDates[documentToken] = Date()
        if frameInfo.isMainFrame {
          self.mainDocumentToken = documentToken
        }
        let frameAlreadyAuthorized =
          self.authorizedRouteToken != nil
          && reportedRouteToken == self.authorizedRouteToken
        if self.inspectionRequested, !self.challengeActive {
          if !frameInfo.isMainFrame,
            let authorizedRouteToken = self.authorizedRouteToken
          {
            self.subframeInspectionRouteTokens[documentToken] = authorizedRouteToken
          }
          if !frameAlreadyAuthorized {
            self.activateInspection(in: frameInfo)
          }
        }
        self.completeChallengeIfPossible()
      }
    }
  }

  private func authorizeSubframeNavigationAction(
    to destinationURL: URL,
    sourceFrame: WKFrameInfo,
    targetFrame: WKFrameInfo,
    navigationTypeRawValue: Int,
    isChallengeNavigation: Bool,
    allowsInitialChildChallengeFallback: Bool,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) {
    pruneExpiredInitialUserAuthorizations()
    let generation = navigationGeneration
    let authorizingWebView = webView
    // Probe the previous target document first. A JavaScript execution failure
    // is never treated as an empty initial frame. The narrow source fallback
    // accepts only a successful nil or a proved, otherwise-unregistered token
    // from the exact about:blank child document.
    authorizingWebView.callAsyncJavaScript(
      "return window.__miohInteractiveDocumentToken ?? null;",
      arguments: [:],
      in: targetFrame,
      in: Self.instrumentationContentWorld
    ) { [weak self, weak authorizingWebView] result in
      Task { @MainActor [weak self, weak authorizingWebView] in
        guard let self, let authorizingWebView,
          self.webView === authorizingWebView,
          !self.isClosingPage,
          generation == self.navigationGeneration
        else {
          decisionHandler(.cancel)
          return
        }
        guard case .success(let value) = result else {
          decisionHandler(.cancel)
          return
        }
        guard let documentToken = value as? String else {
          guard value is NSNull else {
            decisionHandler(.cancel)
            return
          }
          if self.beginInitialChildChallengeFallback(
            to: destinationURL,
            sourceFrame: sourceFrame,
            targetFrame: targetFrame,
            targetPriorDocumentToken: nil,
            generation: generation,
            isChallengeNavigation: isChallengeNavigation,
            allowsFallback: allowsInitialChildChallengeFallback,
            decisionHandler: decisionHandler
          )
            || self.beginInitialUserActivatedChildFallback(
              to: destinationURL,
              sourceFrame: sourceFrame,
              targetFrame: targetFrame,
              targetPriorDocumentToken: nil,
              generation: generation,
              navigationTypeRawValue: navigationTypeRawValue,
              isChallengeNavigation: isChallengeNavigation,
              allowsFallback: allowsInitialChildChallengeFallback,
              decisionHandler: decisionHandler
            )
          {
            return
          }
          decisionHandler(.cancel)
          return
        }
        guard !documentToken.isEmpty,
          documentToken.utf8.count <= Self.maximumScriptTokenLength
        else {
          decisionHandler(.cancel)
          return
        }
        if let redirectPolicy = await self.continuePendingInitialUserChildRedirect(
          to: destinationURL,
          sourceFrame: sourceFrame,
          targetDocumentToken: documentToken,
          generation: generation,
          navigationTypeRawValue: navigationTypeRawValue,
          allowsFallback: allowsInitialChildChallengeFallback,
          webView: authorizingWebView
        ) {
          decisionHandler(redirectPolicy)
          return
        }
        if self.isUnregisteredInitialTargetDocumentToken(documentToken) {
          if self.beginInitialChildChallengeFallback(
            to: destinationURL,
            sourceFrame: sourceFrame,
            targetFrame: targetFrame,
            targetPriorDocumentToken: documentToken,
            generation: generation,
            isChallengeNavigation: isChallengeNavigation,
            allowsFallback: allowsInitialChildChallengeFallback,
            decisionHandler: decisionHandler
          )
            || self.beginInitialUserActivatedChildFallback(
              to: destinationURL,
              sourceFrame: sourceFrame,
              targetFrame: targetFrame,
              targetPriorDocumentToken: documentToken,
              generation: generation,
              navigationTypeRawValue: navigationTypeRawValue,
              isChallengeNavigation: isChallengeNavigation,
              allowsFallback: allowsInitialChildChallengeFallback,
              decisionHandler: decisionHandler
            )
          {
            return
          }
        }
        if self.subframeDocumentEpochs[documentToken] == nil {
          guard self.isLoading,
            self.subframeDocumentEpochs.count < Self.maximumKnownFrameCount
          else {
            decisionHandler(.cancel)
            return
          }
          self.subframeDocumentEpochs[documentToken] = 1
        }
        guard
          self.knownFrames[documentToken] != nil
            || self.subframeNavigationChains[documentToken] != nil
            || self.isLoading,
          let authorization = self.makeSubframeNavigationAuthorization(
            targetPriorDocumentToken: documentToken,
            destinationURL: destinationURL,
            isChallengeNavigation: isChallengeNavigation
          )
        else {
          decisionHandler(.cancel)
          return
        }
        self.pendingSubframeAuthorizations.append(authorization)
        if case .challenge = authorization.chain.authority {
          self.challengeActive = true
          self.acceptingScriptCandidates = false
          self.quietTask?.cancel()
          self.quietTask = nil
          self.setChallengeWaitingStatus()
        }
        decisionHandler(.allow)
      }
    }
  }

  private func beginInitialChildChallengeFallback(
    to destinationURL: URL,
    sourceFrame: WKFrameInfo,
    targetFrame: WKFrameInfo,
    targetPriorDocumentToken: String?,
    generation: Int,
    isChallengeNavigation: Bool,
    allowsFallback: Bool,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) -> Bool {
    guard allowsFallback,
      isChallengeNavigation,
      isAllowedNativeChallengeFrameURL(destinationURL),
      sourceFrame.isMainFrame,
      sourceFrame !== targetFrame,
      targetFrame.request.url?.absoluteString == "about:blank",
      initialChildChallengeDecisionID == nil,
      !hasOutstandingInitialChildChallengeAuthorization
    else { return false }
    let decisionID = UUID().uuidString.lowercased()
    initialChildChallengeDecisionID = decisionID
    authorizeInitialChildChallengeNavigation(
      to: destinationURL,
      sourceFrame: sourceFrame,
      targetPriorDocumentToken: targetPriorDocumentToken,
      generation: generation,
      decisionID: decisionID,
      decisionHandler: decisionHandler
    )
    return true
  }

  private func beginInitialUserActivatedChildFallback(
    to destinationURL: URL,
    sourceFrame: WKFrameInfo,
    targetFrame: WKFrameInfo,
    targetPriorDocumentToken: String?,
    generation: Int,
    navigationTypeRawValue: Int,
    isChallengeNavigation: Bool,
    allowsFallback: Bool,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) -> Bool {
    guard allowsFallback,
      !isChallengeNavigation,
      sourceFrame !== targetFrame,
      targetFrame.request.url?.absoluteString.lowercased() == "about:blank",
      acceptingScriptCandidates,
      authorizedRouteToken != nil,
      authorizedRouteURL == webView.url?.absoluteString,
      pendingInitialUserActivatedChildDecisions
        + initialUserActivatedChildNavigationCount
        < Self.maximumInitialUserActivatedChildNavigationCount
    else { return false }

    pendingInitialUserActivatedChildDecisions += 1
    authorizeInitialUserActivatedChildNavigation(
      to: destinationURL,
      sourceFrame: sourceFrame,
      targetFrame: targetFrame,
      targetPriorDocumentToken: targetPriorDocumentToken,
      generation: generation,
      navigationTypeRawValue: navigationTypeRawValue,
      decisionHandler: decisionHandler
    )
    return true
  }

  private func authorizeInitialUserActivatedChildNavigation(
    to destinationURL: URL,
    sourceFrame: WKFrameInfo,
    targetFrame: WKFrameInfo,
    targetPriorDocumentToken: String?,
    generation: Int,
    navigationTypeRawValue: Int,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) {
    let authorizingWebView = webView
    resolveInitialTargetDocumentToken(
      in: targetFrame,
      knownToken: targetPriorDocumentToken,
      webView: authorizingWebView,
      generation: generation,
      remainingAttempts: 8
    ) { [weak self, weak authorizingWebView] resolvedTargetToken in
      guard let self, let authorizingWebView,
        self.webView === authorizingWebView,
        !self.isClosingPage,
        generation == self.navigationGeneration,
        let resolvedTargetToken
      else {
        if let self {
          self.pendingInitialUserActivatedChildDecisions = max(
            0,
            self.pendingInitialUserActivatedChildDecisions - 1
          )
        }
        decisionHandler(.cancel)
        return
      }
      authorizingWebView.callAsyncJavaScript(
        """
        const token = window.__miohInteractiveDocumentToken ?? null;
        const activated =
          window.__miohInteractiveConsumeTrustedActivation?.() === true;
        return {documentToken: token, activated};
        """,
        arguments: [:],
        in: sourceFrame,
        in: Self.instrumentationContentWorld
      ) { [weak self, weak authorizingWebView] result in
        Task { @MainActor [weak self, weak authorizingWebView] in
          guard let self else {
            decisionHandler(.cancel)
            return
          }
          self.pendingInitialUserActivatedChildDecisions = max(
            0,
            self.pendingInitialUserActivatedChildDecisions - 1
          )
          guard let authorizingWebView,
            self.webView === authorizingWebView,
            !self.isClosingPage,
            generation == self.navigationGeneration,
            case .success(let value) = result,
            let proof = value as? [String: Any],
            proof["activated"] as? Bool == true,
            let initiatorDocumentToken = proof["documentToken"] as? String,
            !initiatorDocumentToken.isEmpty,
            initiatorDocumentToken.utf8.count <= Self.maximumScriptTokenLength,
            let authorization = self.makeInitialUserActivatedChildAuthorization(
              initiatorDocumentToken: initiatorDocumentToken,
              initiatorFrame: sourceFrame,
              targetPriorDocumentToken: resolvedTargetToken,
              destinationURL: destinationURL,
              generation: generation,
              navigationTypeRawValue: navigationTypeRawValue
            )
          else {
            decisionHandler(.cancel)
            return
          }
          self.initialUserActivatedChildNavigationCount += 1
          self.pendingSubframeAuthorizations.append(authorization)
          decisionHandler(.allow)
        }
      }
    }
  }

  private func resolveInitialTargetDocumentToken(
    in targetFrame: WKFrameInfo,
    knownToken: String?,
    webView authorizingWebView: WKWebView,
    generation: Int,
    remainingAttempts: Int,
    completion: @escaping (String?) -> Void
  ) {
    if let knownToken, !knownToken.isEmpty,
      knownToken.utf8.count <= Self.maximumScriptTokenLength
    {
      completion(knownToken)
      return
    }
    guard remainingAttempts > 0 else {
      completion(nil)
      return
    }
    authorizingWebView.callAsyncJavaScript(
      "return window.__miohInteractiveDocumentToken ?? null;",
      arguments: [:],
      in: targetFrame,
      in: Self.instrumentationContentWorld
    ) { [weak self, weak authorizingWebView] result in
      Task { @MainActor [weak self, weak authorizingWebView] in
        guard let self, let authorizingWebView,
          self.webView === authorizingWebView,
          !self.isClosingPage,
          generation == self.navigationGeneration
        else {
          completion(nil)
          return
        }
        if case .success(let value) = result,
          let token = value as? String,
          !token.isEmpty,
          token.utf8.count <= Self.maximumScriptTokenLength
        {
          completion(token)
          return
        }
        guard remainingAttempts > 1 else {
          completion(nil)
          return
        }
        try? await Task.sleep(nanoseconds: 50_000_000)
        self.resolveInitialTargetDocumentToken(
          in: targetFrame,
          knownToken: nil,
          webView: authorizingWebView,
          generation: generation,
          remainingAttempts: remainingAttempts - 1,
          completion: completion
        )
      }
    }
  }

  private func documentToken(
    in frame: WKFrameInfo,
    webView authorizingWebView: WKWebView
  ) async -> String? {
    await withCheckedContinuation { continuation in
      authorizingWebView.callAsyncJavaScript(
        "return window.__miohInteractiveDocumentToken ?? null;",
        arguments: [:],
        in: frame,
        in: Self.instrumentationContentWorld
      ) { result in
        guard case .success(let value) = result,
          let token = value as? String,
          !token.isEmpty,
          token.utf8.count <= Self.maximumScriptTokenLength
        else {
          continuation.resume(returning: nil)
          return
        }
        continuation.resume(returning: token)
      }
    }
  }

  private func continuePendingInitialUserChildRedirect(
    to destinationURL: URL,
    sourceFrame: WKFrameInfo,
    targetDocumentToken: String,
    generation: Int,
    navigationTypeRawValue: Int,
    allowsFallback: Bool,
    webView authorizingWebView: WKWebView
  ) async -> WKNavigationActionPolicy? {
    let matches = pendingSubframeAuthorizations.filter {
      $0.isInitialUserActivatedFallback
        && $0.targetPriorDocumentToken == targetDocumentToken
        && $0.chain.mainNavigationGeneration == generation
    }
    guard !matches.isEmpty else { return nil }
    guard matches.count == 1 else {
      let ambiguousIDs = Set(matches.map(\.id))
      let ambiguous = pendingSubframeAuthorizations.filter {
        ambiguousIDs.contains($0.id)
      }
      pendingSubframeAuthorizations.removeAll {
        ambiguousIDs.contains($0.id)
      }
      for authorization in ambiguous {
        revokeSubframeAuthorization(authorization)
      }
      return .cancel
    }

    let authorizationID = matches[0].id
    guard allowsFallback,
      let sourceDocumentToken = await documentToken(
        in: sourceFrame,
        webView: authorizingWebView
      ),
      sourceDocumentToken == targetDocumentToken,
      webView === authorizingWebView,
      !isClosingPage,
      generation == navigationGeneration,
      let index = pendingSubframeAuthorizations.firstIndex(where: {
        $0.id == authorizationID
      })
    else {
      if let index = pendingSubframeAuthorizations.firstIndex(where: {
        $0.id == authorizationID
      }) {
        let rejected = pendingSubframeAuthorizations.remove(at: index)
        revokeSubframeAuthorization(rejected)
      }
      return .cancel
    }

    var authorization = pendingSubframeAuthorizations[index]
    let navigationTypeMatches =
      authorization.initialUserNavigationTypeRawValue == navigationTypeRawValue
      || navigationTypeRawValue == WKNavigationType.other.rawValue
    guard navigationTypeMatches,
      let deadline = authorization.initialUserRedirectDeadline,
      Date() <= deadline,
      authorization.chain.remainingHopCount > 0,
      !authorization.visitedDestinationURLs.contains(destinationURL.absoluteString),
      authorizationInitiatorIsCurrent(authorization),
      !pendingSubframeAuthorizations.contains(where: {
        $0.id != authorization.id
          && $0.destinationURL == destinationURL.absoluteString
          && $0.chain.mainNavigationGeneration == generation
      }),
      !completedSubframeAuthorizations.contains(where: {
        $0.destinationURL == destinationURL.absoluteString
          && $0.chain.mainNavigationGeneration == generation
      })
    else {
      let rejected = pendingSubframeAuthorizations.remove(at: index)
      revokeSubframeAuthorization(rejected)
      return .cancel
    }

    authorization.chain.remainingHopCount -= 1
    authorization.destinationURL = destinationURL.absoluteString
    authorization.visitedDestinationURLs.insert(destinationURL.absoluteString)
    pendingSubframeAuthorizations[index] = authorization
    return .allow
  }

  private func isUnregisteredInitialTargetDocumentToken(_ documentToken: String) -> Bool {
    documentToken != mainDocumentToken
      && knownFrames[documentToken] == nil
      && subframeDocumentEpochs[documentToken] == nil
      && subframeInspectionRouteTokens[documentToken] == nil
      && subframeNavigationChains[documentToken] == nil
      && !pendingSubframeAuthorizations.contains {
        $0.initiatorDocumentToken == documentToken
          || $0.targetPriorDocumentToken == documentToken
      }
      && !completedSubframeAuthorizations.contains {
        $0.initiatorDocumentToken == documentToken
          || $0.targetPriorDocumentToken == documentToken
      }
  }

  private func authorizeInitialChildChallengeNavigation(
    to destinationURL: URL,
    sourceFrame: WKFrameInfo,
    targetPriorDocumentToken: String?,
    generation: Int,
    decisionID: String,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) {
    let authorizingWebView = webView
    authorizingWebView.callAsyncJavaScript(
      "return window.__miohInteractiveDocumentToken ?? null;",
      arguments: [:],
      in: sourceFrame,
      in: Self.instrumentationContentWorld
    ) { [weak self, weak authorizingWebView] result in
      Task { @MainActor [weak self, weak authorizingWebView] in
        guard let self, let authorizingWebView,
          self.webView === authorizingWebView,
          !self.isClosingPage,
          generation == self.navigationGeneration,
          self.initialChildChallengeDecisionID == decisionID,
          case .success(let value) = result,
          let initiatorDocumentToken = value as? String,
          !initiatorDocumentToken.isEmpty,
          initiatorDocumentToken.utf8.count <= Self.maximumScriptTokenLength
        else {
          self?.clearInitialChildChallengeDecision(decisionID)
          decisionHandler(.cancel)
          return
        }

        // Frame-ready verification is asynchronous. Hold this single decision
        // briefly so a document-start token can become the registered main
        // token; never lend authority from a process-global challenge flag.
        for attempt in 0..<8 {
          if let authorization = self.makeInitialChildChallengeAuthorization(
            initiatorDocumentToken: initiatorDocumentToken,
            initiatorFrame: sourceFrame,
            targetPriorDocumentToken: targetPriorDocumentToken,
            destinationURL: destinationURL,
            generation: generation
          ) {
            self.clearInitialChildChallengeDecision(decisionID)
            self.pendingSubframeAuthorizations.append(authorization)
            self.challengeActive = true
            self.acceptingScriptCandidates = false
            self.quietTask?.cancel()
            self.quietTask = nil
            self.setChallengeWaitingStatus()
            decisionHandler(.allow)
            return
          }
          guard attempt < 7, generation == self.navigationGeneration,
            !self.isClosingPage,
            self.initialChildChallengeDecisionID == decisionID
          else { break }
          try? await Task.sleep(nanoseconds: 50_000_000)
        }
        self.clearInitialChildChallengeDecision(decisionID)
        decisionHandler(.cancel)
      }
    }
  }

  private func clearInitialChildChallengeDecision(_ decisionID: String) {
    if initialChildChallengeDecisionID == decisionID {
      initialChildChallengeDecisionID = nil
    }
  }

  private func makeInitialChildChallengeAuthorization(
    initiatorDocumentToken: String,
    initiatorFrame: WKFrameInfo,
    targetPriorDocumentToken: String?,
    destinationURL: URL,
    generation: Int
  ) -> SubframeNavigationAuthorization? {
    guard generation == navigationGeneration,
      initialChildChallengeFallbackGeneration != generation,
      pendingSubframeAuthorizations.count < Self.maximumKnownFrameCount,
      !hasOutstandingInitialChildChallengeAuthorization,
      !hasOutstandingSubframeAuthorization(for: destinationURL),
      let initiatorURL = Self.sanitizedPublicHTTPSURL(initiatorFrame.request.url)
    else { return nil }

    if let targetPriorDocumentToken,
      !isUnregisteredInitialTargetDocumentToken(targetPriorDocumentToken)
    {
      return nil
    }

    guard initiatorFrame.isMainFrame,
      mainDocumentToken == initiatorDocumentToken,
      knownFrames[initiatorDocumentToken]?.isMainFrame == true,
      committedMainDocumentURL?.absoluteString == initiatorURL.absoluteString,
      mainFrameChallengeResponse,
      mainFrameChallengeResponseGeneration == generation,
      activeSubframeChallengeChainIDs.isEmpty
    else { return nil }
    let chainID = UUID().uuidString.lowercased()
    activeSubframeChallengeChainIDs.insert(chainID)
    var chain = SubframeNavigationChain(
      authority: .challenge(chainID: chainID),
      sourceDocumentEpoch: 1,
      remainingHopCount: Self.maximumSubframeNavigationHopCount,
      mainNavigationGeneration: generation
    )
    guard chain.remainingHopCount > 0 else { return nil }
    chain.remainingHopCount -= 1
    initialChildChallengeFallbackGeneration = generation
    // Epoch 1 belongs to the not-yet-committed target. The initiator remains
    // current and is revalidated at response and frame-ready commit.
    chain.sourceDocumentEpoch = 1
    return SubframeNavigationAuthorization(
      id: UUID().uuidString.lowercased(),
      initiatorDocumentToken: initiatorDocumentToken,
      initiatorDocumentEpoch: nil,
      initiatorFrameURL: initiatorURL.absoluteString,
      targetPriorDocumentToken: targetPriorDocumentToken,
      nativeMainChallengeGeneration: generation,
      isInitialChildChallengeFallback: true,
      isInitialUserActivatedFallback: false,
      destinationURL: destinationURL.absoluteString,
      chain: chain,
      initialUserNavigationTypeRawValue: nil,
      initialUserRedirectDeadline: nil,
      visitedDestinationURLs: []
    )
  }

  private func makeInitialUserActivatedChildAuthorization(
    initiatorDocumentToken: String,
    initiatorFrame: WKFrameInfo,
    targetPriorDocumentToken: String,
    destinationURL: URL,
    generation: Int,
    navigationTypeRawValue: Int
  ) -> SubframeNavigationAuthorization? {
    guard generation == navigationGeneration,
      initialUserActivatedChildNavigationCount
        < Self.maximumInitialUserActivatedChildNavigationCount,
      pendingSubframeAuthorizations.count < Self.maximumKnownFrameCount,
      !hasOutstandingSubframeAuthorization(for: destinationURL),
      acceptingScriptCandidates,
      let routeToken = authorizedRouteToken,
      authorizedRouteURL == webView.url?.absoluteString,
      let initiatorURL = Self.sanitizedPublicHTTPSURL(
        initiatorFrame.request.url
      )
    else { return nil }

    if !isUnregisteredInitialTargetDocumentToken(targetPriorDocumentToken) {
      return nil
    }

    let initiatorEpoch: Int?
    let authority: SubframeNavigationAuthority
    if initiatorFrame.isMainFrame {
      guard mainDocumentToken == initiatorDocumentToken,
        knownFrames[initiatorDocumentToken]?.isMainFrame == true,
        let committedMainDocumentURL,
        Self.isSameOrigin(initiatorURL, committedMainDocumentURL),
        !challengeActive,
        !mainFrameChallengeResponse
      else { return nil }
      initiatorEpoch = nil
      authority = .inspection(routeToken: routeToken)
    } else {
      guard knownFrames[initiatorDocumentToken] != nil,
        let currentEpoch = subframeDocumentEpochs[initiatorDocumentToken],
        subframeInspectionRouteTokens[initiatorDocumentToken] == routeToken
      else { return nil }
      initiatorEpoch = currentEpoch
      if let existingChain = subframeNavigationChains[initiatorDocumentToken],
        existingChain.sourceDocumentEpoch == currentEpoch,
        existingChain.mainNavigationGeneration == generation
      {
        guard case .inspection(let existingRouteToken) = existingChain.authority,
          existingRouteToken == routeToken
        else { return nil }
      }
      authority = .inspection(routeToken: routeToken)
    }

    var chain = SubframeNavigationChain(
      authority: authority,
      sourceDocumentEpoch: 1,
      remainingHopCount: Self.maximumSubframeNavigationHopCount,
      mainNavigationGeneration: generation
    )
    guard chain.remainingHopCount > 0 else { return nil }
    chain.remainingHopCount -= 1
    return SubframeNavigationAuthorization(
      id: UUID().uuidString.lowercased(),
      initiatorDocumentToken: initiatorDocumentToken,
      initiatorDocumentEpoch: initiatorEpoch,
      initiatorFrameURL: initiatorURL.absoluteString,
      targetPriorDocumentToken: targetPriorDocumentToken,
      nativeMainChallengeGeneration: nil,
      isInitialChildChallengeFallback: false,
      isInitialUserActivatedFallback: true,
      destinationURL: destinationURL.absoluteString,
      chain: chain,
      initialUserNavigationTypeRawValue: navigationTypeRawValue,
      initialUserRedirectDeadline: Date().addingTimeInterval(
        Self.initialUserRedirectLifetime
      ),
      visitedDestinationURLs: [destinationURL.absoluteString]
    )
  }

  private func makeSubframeNavigationAuthorization(
    targetPriorDocumentToken: String,
    destinationURL: URL,
    isChallengeNavigation: Bool
  ) -> SubframeNavigationAuthorization? {
    guard let currentEpoch = subframeDocumentEpochs[targetPriorDocumentToken],
      pendingSubframeAuthorizations.count < Self.maximumKnownFrameCount,
      !hasOutstandingSubframeAuthorization(for: destinationURL)
    else { return nil }

    var chain: SubframeNavigationChain
    if let existingChain = subframeNavigationChains[targetPriorDocumentToken],
      existingChain.sourceDocumentEpoch == currentEpoch,
      existingChain.mainNavigationGeneration == navigationGeneration
    {
      chain = existingChain
      switch chain.authority {
      case .pageLoad:
        guard isLoading else { return nil }
      case .inspection(let routeToken):
        guard routeToken == authorizedRouteToken,
          acceptingScriptCandidates || isChallengeNavigation
        else {
          return nil
        }
      case .challenge:
        break
      }
    } else if isLoading {
      chain = SubframeNavigationChain(
        authority: .pageLoad,
        sourceDocumentEpoch: currentEpoch,
        remainingHopCount: Self.maximumSubframeNavigationHopCount,
        mainNavigationGeneration: navigationGeneration
      )
    } else {
      guard knownFrames[targetPriorDocumentToken] != nil,
        let frameRouteToken = subframeInspectionRouteTokens[targetPriorDocumentToken],
        frameRouteToken == authorizedRouteToken,
        acceptingScriptCandidates || isChallengeNavigation
      else { return nil }
      chain = SubframeNavigationChain(
        authority: .inspection(routeToken: frameRouteToken),
        sourceDocumentEpoch: currentEpoch,
        remainingHopCount: Self.maximumSubframeNavigationHopCount,
        mainNavigationGeneration: navigationGeneration
      )
    }
    guard chain.remainingHopCount > 0 else { return nil }
    chain.remainingHopCount -= 1

    if isChallengeNavigation {
      switch chain.authority {
      case .challenge(let chainID):
        guard activeSubframeChallengeChainIDs.count == 1,
          activeSubframeChallengeChainIDs.contains(chainID)
        else { return nil }
      case .pageLoad, .inspection:
        guard activeSubframeChallengeChainIDs.isEmpty else { return nil }
        let chainID = UUID().uuidString.lowercased()
        chain.authority = .challenge(chainID: chainID)
        activeSubframeChallengeChainIDs.insert(chainID)
      }
    }

    let nextEpoch = nextGeneration(after: currentEpoch)
    chain.sourceDocumentEpoch = nextEpoch
    let initiatorFrameURL =
      Self.sanitizedPublicHTTPSURL(
        knownFrames[targetPriorDocumentToken]?.request.url
      )?.absoluteString ?? destinationURL.absoluteString
    knownFrames.removeValue(forKey: targetPriorDocumentToken)
    subframeInspectionRouteTokens.removeValue(forKey: targetPriorDocumentToken)
    subframeDocumentEpochs[targetPriorDocumentToken] = nextEpoch
    pendingSubframeAuthorizations.removeAll {
      $0.targetPriorDocumentToken == targetPriorDocumentToken
    }
    completedSubframeAuthorizations.removeAll {
      $0.targetPriorDocumentToken == targetPriorDocumentToken
    }
    subframeNavigationChains[targetPriorDocumentToken] = chain
    return SubframeNavigationAuthorization(
      id: UUID().uuidString.lowercased(),
      initiatorDocumentToken: targetPriorDocumentToken,
      initiatorDocumentEpoch: nextEpoch,
      initiatorFrameURL: initiatorFrameURL,
      targetPriorDocumentToken: targetPriorDocumentToken,
      nativeMainChallengeGeneration: nil,
      isInitialChildChallengeFallback: false,
      isInitialUserActivatedFallback: false,
      destinationURL: destinationURL.absoluteString,
      chain: chain,
      initialUserNavigationTypeRawValue: nil,
      initialUserRedirectDeadline: nil,
      visitedDestinationURLs: []
    )
  }

  private func consumeSubframeAuthorization(
    for responseURL: URL
  ) -> SubframeNavigationAuthorization? {
    pendingSubframeAuthorizations.removeAll {
      $0.chain.mainNavigationGeneration != navigationGeneration
    }
    let matchingIndices = pendingSubframeAuthorizations.indices.filter {
      let authorization = pendingSubframeAuthorizations[$0]
      return
        authorization.destinationURL == responseURL.absoluteString
        && authorization.chain.mainNavigationGeneration == navigationGeneration
    }
    guard matchingIndices.count == 1, let index = matchingIndices.first else {
      if matchingIndices.count > 1 {
        let ambiguousIDs = Set(
          matchingIndices.map { pendingSubframeAuthorizations[$0].id }
        )
        let ambiguous = pendingSubframeAuthorizations.filter {
          ambiguousIDs.contains($0.id)
        }
        pendingSubframeAuthorizations.removeAll {
          ambiguousIDs.contains($0.id)
        }
        for authorization in ambiguous {
          revokeSubframeAuthorization(authorization)
        }
      }
      return nil
    }
    return pendingSubframeAuthorizations.remove(at: index)
  }

  private func pruneExpiredInitialUserAuthorizations(now: Date = Date()) {
    let expiredIDs: Set<String> = Set(
      pendingSubframeAuthorizations.compactMap { authorization in
        guard authorization.isInitialUserActivatedFallback,
          let deadline = authorization.initialUserRedirectDeadline,
          deadline < now
        else { return nil }
        return authorization.id
      }
    )
    guard !expiredIDs.isEmpty else { return }
    let expired = pendingSubframeAuthorizations.filter {
      expiredIDs.contains($0.id)
    }
    pendingSubframeAuthorizations.removeAll {
      expiredIDs.contains($0.id)
    }
    for authorization in expired {
      revokeSubframeAuthorization(authorization)
    }
  }

  private func hasOutstandingSubframeAuthorization(
    for destinationURL: URL
  ) -> Bool {
    let destination = destinationURL.absoluteString
    return pendingSubframeAuthorizations.contains {
      $0.destinationURL == destination
        && $0.chain.mainNavigationGeneration == navigationGeneration
    }
      || completedSubframeAuthorizations.contains {
        $0.destinationURL == destination
          && $0.chain.mainNavigationGeneration == navigationGeneration
      }
  }

  private var hasOutstandingInitialChildChallengeAuthorization: Bool {
    pendingSubframeAuthorizations.contains {
      $0.isInitialChildChallengeFallback
        && $0.chain.mainNavigationGeneration == navigationGeneration
    }
      || completedSubframeAuthorizations.contains {
        $0.isInitialChildChallengeFallback
          && $0.chain.mainNavigationGeneration == navigationGeneration
      }
  }

  private func authorizationInitiatorIsCurrent(
    _ authorization: SubframeNavigationAuthorization
  ) -> Bool {
    guard authorization.chain.mainNavigationGeneration == navigationGeneration else {
      return false
    }
    if authorization.isInitialUserActivatedFallback {
      guard let routeToken = authorizedRouteToken,
        authorizedRouteURL == webView.url?.absoluteString,
        let frame = knownFrames[authorization.initiatorDocumentToken],
        Self.sanitizedPublicHTTPSURL(frame.request.url)?.absoluteString
          == authorization.initiatorFrameURL
      else { return false }

      let isChallengeAuthority: Bool
      switch authorization.chain.authority {
      case .inspection(let authorityRouteToken):
        guard acceptingScriptCandidates,
          authorityRouteToken == routeToken
        else { return false }
        isChallengeAuthority = false
      case .challenge(let chainID):
        // A verified initial player child may itself receive a Cloudflare
        // response. The response delegate intentionally suspends script
        // candidate intake before frame-ready commits, so revalidate the
        // promoted, bounded child chain instead of requiring inspection mode.
        guard challengeActive, !mainFrameChallengeResponse,
          activeSubframeChallengeChainIDs.count == 1,
          activeSubframeChallengeChainIDs.contains(chainID)
        else { return false }
        isChallengeAuthority = true
      case .pageLoad:
        return false
      }
      if frame.isMainFrame {
        guard authorization.initiatorDocumentEpoch == nil,
          mainDocumentToken == authorization.initiatorDocumentToken
        else { return false }
        return isChallengeAuthority
          ? !mainFrameChallengeResponse
          : (!challengeActive && !mainFrameChallengeResponse)
      }
      guard let expectedEpoch = authorization.initiatorDocumentEpoch else {
        return false
      }
      return subframeDocumentEpochs[authorization.initiatorDocumentToken]
        == expectedEpoch
        && subframeInspectionRouteTokens[authorization.initiatorDocumentToken]
          == routeToken
    }
    if let nativeGeneration = authorization.nativeMainChallengeGeneration {
      guard nativeGeneration == navigationGeneration,
        mainFrameChallengeResponse,
        mainFrameChallengeResponseGeneration == nativeGeneration,
        mainDocumentToken == authorization.initiatorDocumentToken,
        let frame = knownFrames[authorization.initiatorDocumentToken],
        frame.isMainFrame,
        Self.sanitizedPublicHTTPSURL(frame.request.url)?.absoluteString
          == authorization.initiatorFrameURL,
        committedMainDocumentURL?.absoluteString == authorization.initiatorFrameURL
      else { return false }
      return true
    }

    guard let expectedEpoch = authorization.initiatorDocumentEpoch,
      subframeDocumentEpochs[authorization.initiatorDocumentToken] == expectedEpoch,
      let currentChain = subframeNavigationChains[authorization.initiatorDocumentToken],
      currentChain.sourceDocumentEpoch == expectedEpoch,
      currentChain.mainNavigationGeneration == navigationGeneration,
      Self.sameAuthority(currentChain.authority, authorization.chain.authority)
    else { return false }

    if authorization.targetPriorDocumentToken == nil {
      guard let frame = knownFrames[authorization.initiatorDocumentToken],
        Self.sanitizedPublicHTTPSURL(frame.request.url)?.absoluteString
          == authorization.initiatorFrameURL,
        case .challenge(let chainID) = currentChain.authority,
        activeSubframeChallengeChainIDs.count == 1,
        activeSubframeChallengeChainIDs.contains(chainID)
      else { return false }
    }
    return true
  }

  private static func sameAuthority(
    _ lhs: SubframeNavigationAuthority,
    _ rhs: SubframeNavigationAuthority
  ) -> Bool {
    switch (lhs, rhs) {
    case (.pageLoad, .pageLoad):
      return true
    case (.inspection(let left), .inspection(let right)):
      return left == right
    case (.challenge(let left), .challenge(let right)):
      return left == right
    default:
      return false
    }
  }

  private func promoteSubframeAuthorizationToChallenge(
    _ authorization: SubframeNavigationAuthorization
  ) -> SubframeNavigationAuthorization? {
    var promoted = authorization
    switch promoted.chain.authority {
    case .challenge(let chainID):
      guard activeSubframeChallengeChainIDs.count == 1,
        activeSubframeChallengeChainIDs.contains(chainID)
      else { return nil }
    case .pageLoad, .inspection:
      guard activeSubframeChallengeChainIDs.isEmpty else { return nil }
      let chainID = UUID().uuidString.lowercased()
      promoted.chain.authority = .challenge(chainID: chainID)
      activeSubframeChallengeChainIDs.insert(chainID)
    }
    if let targetPriorDocumentToken = promoted.targetPriorDocumentToken,
      subframeDocumentEpochs[targetPriorDocumentToken]
        == promoted.chain.sourceDocumentEpoch
    {
      subframeNavigationChains[targetPriorDocumentToken] = promoted.chain
    }
    return promoted
  }

  private func revokeSubframeAuthorization(
    _ authorization: SubframeNavigationAuthorization
  ) {
    if let targetPriorDocumentToken = authorization.targetPriorDocumentToken,
      subframeDocumentEpochs[targetPriorDocumentToken]
        == authorization.chain.sourceDocumentEpoch
    {
      subframeNavigationChains.removeValue(
        forKey: targetPriorDocumentToken
      )
    }
    if case .challenge(let chainID) = authorization.chain.authority {
      activeSubframeChallengeChainIDs.remove(chainID)
      completeChallengeIfPossible()
    }
  }

  private func insertCandidate(_ candidate: Candidate) {
    guard !Self.isHighConfidenceAdvertisementMediaURL(candidate.url) else {
      return
    }
    let key = candidate.url.absoluteString
    if let existing = candidates[key] {
      if existing.provenance != .script, candidate.provenance == .script {
        return
      }
      let refreshedSameFrameObservation =
        existing.provenance == .script && candidate.provenance == .script
        && existing.documentToken != nil
        && existing.documentToken == candidate.documentToken
        && existing.sourceKind == candidate.sourceKind
        && candidate.observedAt > existing.observedAt
      let credentialUpgrade =
        candidate.provenance != .script && existing.provenance == .script
      let preferredUpgrade =
        credentialUpgrade || Self.isPreferred(candidate, over: existing)
      if refreshedSameFrameObservation || preferredUpgrade {
        candidates[key] = candidate
        candidateRevision = nextGeneration(after: candidateRevision)
        refreshCandidateSummaries()
        if Self.isReadyCandidate(candidate) {
          scheduleReadyCandidate()
        }
      }
      return
    }

    nextDiscoveryOrder = nextDiscoveryOrder == Int.max ? 0 : nextDiscoveryOrder + 1
    if candidates.count < Self.maximumCandidateCount {
      candidates[key] = candidate
    } else if let worst = candidates.values.max(by: {
      Self.isPreferred($0, over: $1)
    }),
      Self.isPreferred(candidate, over: worst)
    {
      candidates.removeValue(forKey: worst.url.absoluteString)
      candidates[key] = candidate
    } else {
      return
    }
    candidateRevision = nextGeneration(after: candidateRevision)
    refreshCandidateSummaries()

    if Self.isReadyCandidate(candidate) {
      scheduleReadyCandidate()
    }
  }

  private static func isReadyCandidate(_ candidate: Candidate) -> Bool {
    switch candidate.provenance {
    case .mainNavigationResponse, .subframeNavigationResponse:
      return true
    case .script, .currentPage:
      break
    }
    switch candidate.sourceKind.lowercased() {
    case "active-current-source", "fetch-media-response", "xhr-media-response":
      return true
    default:
      return false
    }
  }

  private static func isUnverifiedMediaResponseHint(_ candidate: Candidate) -> Bool {
    guard candidate.provenance == .script else { return false }
    switch candidate.sourceKind.lowercased() {
    case "page-fetch-hls-response", "page-xhr-hls-response": return true
    default: return false
    }
  }

  private static func candidateSourceLabel(_ sourceKind: String) -> String {
    switch sourceKind.lowercased() {
    case "active-current-source": "動画の再生ソース"
    case "navigation-response": "動画HTTP応答"
    case "fetch-media-response": "Fetch動画応答"
    case "xhr-media-response": "XHR動画応答"
    case "page-fetch-hls-response": "ページFetch HLS候補"
    case "page-xhr-hls-response": "ページXHR HLS候補"
    case "page-fetch-hls-request": "ページFetch HLS要求"
    case "page-xhr-hls-request": "ページXHR HLS要求"
    case "script-text": "script内URL"
    case "fetch-response": "Fetch応答候補"
    case "xhr-response": "XHR応答候補"
    case "fetch": "Fetch要求"
    case "xhr": "XHR要求"
    case "performance": "通信履歴"
    case "video", "currentsrc": "video要素"
    case "source": "source要素"
    case "iframe": "iframeリンク"
    case "frame": "埋め込みページ"
    case "popup": "ポップアップリンク"
    case "page": "表示ページ"
    case "setattribute": "動的属性"
    default: sourceKind.isEmpty ? "その他" : sourceKind
    }
  }

  private func scheduleReadyCandidate() {
    quietTask?.cancel()
    quietTask = Task { @MainActor [weak self] in
      do {
        try await Task.sleep(nanoseconds: 2_000_000_000)
      } catch {
        return
      }
      guard let self, !self.challengeActive, !self.isClosingPage,
        !self.isLoading,
        self.hasVerifiedMediaCandidate
      else { return }
      self.readyCandidateGeneration =
        self.readyCandidateGeneration == Int.max
        ? 1 : self.readyCandidateGeneration + 1
      self.statusMessage = "再生可能性を確認した配信URLがあります。「配信を解析」を押してください。"
    }
  }

  private func activateInspectionInKnownFrames() {
    if knownFrames.isEmpty {
      activateInspection(in: nil)
    } else {
      for (documentToken, frame) in knownFrames {
        if !frame.isMainFrame, let authorizedRouteToken {
          subframeInspectionRouteTokens[documentToken] = authorizedRouteToken
        }
        activateInspection(in: frame)
      }
    }
  }

  private func beginInspectionForCurrentRoute() {
    guard !challengeActive, !mainFrameChallengeResponse, !isClosingPage,
      !isLoading,
      let currentWebViewURL = webView.url,
      Self.sanitizedPublicHTTPSURL(currentWebViewURL) != nil
    else {
      acceptingScriptCandidates = false
      return
    }
    let currentRouteURL = currentWebViewURL.absoluteString
    if let lastPageLocationToken,
      lastPageLocationToken != currentRouteURL
    {
      processSameDocumentPageLocation(currentRouteURL)
      return
    }
    lastPageLocationToken = currentRouteURL
    authorizedRouteToken = UUID().uuidString.lowercased()
    pageNetworkObservationEpoch = UUID().uuidString.lowercased()
    authorizedRouteURL = currentRouteURL
    pendingSubframeAuthorizations.removeAll()
    completedSubframeAuthorizations.removeAll()
    subframeNavigationChains.removeAll()
    subframeInspectionRouteTokens.removeAll()
    activeSubframeChallengeChainIDs.removeAll()
    acceptingScriptCandidates = true
    activateInspectionInKnownFrames()
  }

  private func activateInspection(in frame: WKFrameInfo?) {
    guard let authorizedRouteToken else { return }
    let generation = navigationGeneration
    let bridgeEventName = pageNetworkBridgeEventName
    let observationEpoch = pageNetworkObservationEpoch
    let inspectingWebView = webView
    inspectingWebView.callAsyncJavaScript(
      "window.__miohInteractiveAuthorizeRoute?.(routeToken);",
      arguments: ["routeToken": authorizedRouteToken],
      in: frame,
      in: Self.instrumentationContentWorld
    ) { _ in }
    inspectingWebView.callAsyncJavaScript(
      Self.activeInstrumentationScript,
      arguments: [:],
      in: frame,
      in: Self.instrumentationContentWorld
    ) { _ in }
    inspectingWebView.callAsyncJavaScript(
      """
      window.__miohInteractiveAuthorizeRoute?.(routeToken);
      return window.__miohInteractiveInstallPageNetworkBridge?.(
        bridgeEventName,
        observationEpoch
      ) === true;
      """,
      arguments: [
        "routeToken": authorizedRouteToken,
        "bridgeEventName": bridgeEventName,
        "observationEpoch": observationEpoch,
      ],
      in: frame,
      in: Self.instrumentationContentWorld
    ) { [weak self, weak inspectingWebView] result in
      Task { @MainActor [weak self, weak inspectingWebView] in
        guard let self, let inspectingWebView,
          self.webView === inspectingWebView,
          generation == self.navigationGeneration,
          self.authorizedRouteToken == authorizedRouteToken,
          self.acceptingScriptCandidates,
          !self.challengeActive,
          case .success(let installed) = result,
          installed as? Bool == true
        else { return }
        inspectingWebView.callAsyncJavaScript(
          Self.pageNetworkObservationScript,
          arguments: [
            "bridgeEventName": bridgeEventName,
            "observationEpoch": observationEpoch,
          ],
          in: frame,
          in: WKContentWorld.page
        ) { _ in }
      }
    }
  }

  private func allRequestCookies() async -> [IPadMediaRequestCookie] {
    let cookies = await withCheckedContinuation { continuation in
      websiteDataStore.httpCookieStore.getAllCookies { cookies in
        continuation.resume(returning: cookies)
      }
    }
    return cookies.compactMap(IPadMediaRequestCookie.init)
  }

  private func updateNavigationState() {
    if Self.relaxedWebCompatibilityEnabled {
      canGoBack = webView.canGoBack
      canGoForward = webView.canGoForward
      return
    }
    canGoBack =
      webView.backForwardList.backItem.flatMap {
        Self.sanitizedPublicHTTPSURL($0.url)
      } != nil
    canGoForward = webView.canGoForward
  }

  private func publishSuccessfulPageVisitIfReady() {
    defer { pendingSuccessfulMainResponseURL = nil }
    guard let expectedURL = pendingSuccessfulMainResponseURL,
      Self.sanitizedPublicHTTPSURL(webView.url)?.absoluteString
        == expectedURL.absoluteString
    else { return }
    noteSuccessfulPublicPageVisit(url: expectedURL)
  }

  private func noteSuccessfulPublicPageVisit(url: URL? = nil) {
    guard !isClosingPage, !challengeActive,
      let safeURL = Self.sanitizedPublicHTTPSURL(url ?? webView.url),
      !Self.isInteractionChallenge(safeURL)
    else { return }
    let revision = nextGeneration(after: successfulPageVisit?.revision ?? 0)
    successfulPageVisit = SuccessfulPageVisit(
      url: safeURL,
      title: webView.title ?? pageTitle,
      revision: revision
    )
  }

  private static func successfulVisibleMainResponseURL(
    _ navigationResponse: WKNavigationResponse,
    response: HTTPURLResponse?
  ) -> URL? {
    guard navigationResponse.isForMainFrame,
      navigationResponse.canShowMIMEType,
      let response,
      (200..<400).contains(response.statusCode),
      response.value(forHTTPHeaderField: "Content-Disposition")?
        .lowercased().contains("attachment") != true,
      response.value(forHTTPHeaderField: "cf-mitigated")?.lowercased()
        != "challenge",
      let safeURL = sanitizedPublicHTTPSURL(navigationResponse.response.url),
      !isInteractionChallenge(safeURL)
    else { return nil }
    return safeURL
  }

  private static func normalizedPublicHTTPSURL(_ raw: String) -> URL? {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
      trimmed.rangeOfCharacter(from: .controlCharacters) == nil
    else { return nil }
    let value = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
    return URL(string: value).flatMap(sanitizedPublicHTTPSURL)
  }

  private static func sanitizedPublicHTTPSURL(_ rawURL: URL?) -> URL? {
    guard let rawURL,
      rawURL.absoluteString.utf8.count <= maximumCandidateURLLength,
      var components = URLComponents(url: rawURL, resolvingAgainstBaseURL: true),
      components.scheme?.lowercased() == "https",
      components.user == nil, components.password == nil,
      components.host?.isEmpty == false
    else { return nil }
    components.scheme = "https"
    components.fragment = nil
    guard let host = components.host, Self.isPublicHostSyntax(host),
      let safeURL = components.url
    else { return nil }
    return safeURL
  }

  /// Returns a stable page URL suitable for a small session snapshot. Expiring
  /// credentials and signed media parameters are never returned; WebKit itself
  /// retains ordinary cookies and website storage.
  private static func persistableSessionURL(_ rawURL: URL?) -> URL? {
    guard let safeURL = sanitizedPublicHTTPSURL(rawURL),
      var components = URLComponents(
        url: safeURL,
        resolvingAgainstBaseURL: false
      )
    else { return nil }
    if let queryItems = components.queryItems {
      let stableItems = queryItems.filter {
        !isSensitiveSessionQueryName($0.name)
      }
      components.queryItems = stableItems.isEmpty ? nil : stableItems
    }
    components.fragment = nil
    return components.url.flatMap(sanitizedPublicHTTPSURL)
  }

  static func persistableSessionAddress(_ raw: String) -> String? {
    normalizedPublicHTTPSURL(raw).flatMap(persistableSessionURL)?.absoluteString
  }

  private static func isSensitiveSessionQueryName(_ rawName: String) -> Bool {
    let name = rawName.lowercased()
    if name.hasPrefix("x-amz-") || name.hasPrefix("x-goog-") { return true }
    return [
      "access_token", "auth", "authorization", "code", "credential",
      "credentials", "expires", "hdnts", "jwt", "key", "passwd",
      "password", "policy", "session", "sessionid", "sig", "signature",
      "token",
    ].contains(name)
  }

  /// Performs only nonblocking syntax checks suitable for WebKit delegate calls.
  /// URLs emitted by the page remain untrusted; the public-discovered resolver
  /// performs DNS-aware validation before the app fetches any candidate.
  private static func isPublicHostSyntax(_ rawHost: String) -> Bool {
    let host = rawHost.lowercased().trimmingCharacters(
      in: CharacterSet(charactersIn: ".[]")
    )
    guard !host.isEmpty, host != "localhost",
      !host.hasSuffix(".localhost"), !host.hasSuffix(".local"),
      !host.hasPrefix("0x")
    else { return false }

    let rawIPv4Parts = host.split(separator: ".", omittingEmptySubsequences: false)
    if rawIPv4Parts.count == 4,
      rawIPv4Parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) })
    {
      guard rawIPv4Parts.allSatisfy({ $0.count == 1 || !$0.hasPrefix("0") }) else {
        return false
      }
      let parts = rawIPv4Parts.compactMap { UInt8($0) }
      guard parts.count == 4 else { return false }
      let first = parts[0]
      let second = parts[1]
      let third = parts[2]
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
    if host.allSatisfy({ $0.isNumber || $0 == "." }) { return false }

    if host.contains(":") {
      return host != "::" && host != "::1" && !host.hasPrefix("::ffff:")
        && !host.hasPrefix("fc") && !host.hasPrefix("fd")
        && !host.hasPrefix("fe8") && !host.hasPrefix("fe9")
        && !host.hasPrefix("fea") && !host.hasPrefix("feb")
    }
    return true
  }

  private static func isInteractionChallenge(_ url: URL) -> Bool {
    let host = url.host?.lowercased() ?? ""
    return host == "challenges.cloudflare.com"
      || host.hasSuffix(".challenges.cloudflare.com")
      || url.path.lowercased().contains("/cdn-cgi/challenge-platform/")
  }

  private static func isChallengeLocalFrameURL(_ url: URL?) -> Bool {
    guard let value = url?.absoluteString.lowercased() else { return false }
    return value == "about:blank" || value == "about:srcdoc"
  }

  private func isCurrentMainOrCloudflareChallengeURL(_ url: URL) -> Bool {
    let host = url.host?.lowercased() ?? ""
    let isCloudflareChallengeHost =
      host == "challenges.cloudflare.com"
      || host.hasSuffix(".challenges.cloudflare.com")
    return isCloudflareChallengeHost
      || committedMainDocumentURL.map { Self.isSameOrigin(url, $0) } == true
  }

  private func isAllowedNativeChallengeFrameURL(_ url: URL) -> Bool {
    hasCurrentNativeMainChallenge
      && isCurrentMainOrCloudflareChallengeURL(url)
  }

  private func isContextualChallengeNavigationURL(_ url: URL) -> Bool {
    isCurrentMainOrCloudflareChallengeURL(url)
      && Self.isInteractionChallenge(url)
  }

  private func isEligibleNativeChallengeSource(_ sourceFrame: WKFrameInfo) -> Bool {
    if sourceFrame.isMainFrame {
      return Self.sanitizedPublicHTTPSURL(sourceFrame.request.url)?.absoluteString
        == committedMainDocumentURL?.absoluteString
    }
    guard let sourceURL = Self.sanitizedPublicHTTPSURL(sourceFrame.request.url) else {
      return false
    }
    return isAllowedNativeChallengeFrameURL(sourceURL)
  }

  private static func isDirectMediaCandidate(_ url: URL) -> Bool {
    let value = url.absoluteString.lowercased()
    let pathExtension = url.pathExtension.lowercased()
    return pathExtension == "m3u8" || pathExtension == "mp4"
      || pathExtension == "mov" || pathExtension == "m4v"
      || value.contains(".m3u8?") || value.contains(".mp4?")
  }

  private static func isDirectMediaResponse(
    _ url: URL,
    response: HTTPURLResponse?
  ) -> Bool {
    let mimeType = response?.mimeType?.lowercased() ?? ""
    if mimeType.hasPrefix("text/")
      || mimeType == "application/xhtml+xml"
      || mimeType == "application/json"
      || mimeType == "application/javascript"
    {
      return false
    }
    if mimeType.hasPrefix("video/")
      || [
        "application/vnd.apple.mpegurl",
        "application/x-mpegurl",
        "application/mpegurl",
        "audio/mpegurl",
        "audio/x-mpegurl",
      ].contains(mimeType)
    {
      return true
    }
    return isDirectMediaCandidate(url)
  }

  private static func isCancelledNavigationError(_ error: Error) -> Bool {
    let nsError = error as NSError
    return nsError.domain == NSURLErrorDomain
      && nsError.code == NSURLErrorCancelled
  }

  private static func isPreferred(_ lhs: Candidate, over rhs: Candidate) -> Bool {
    let left = priorityComponents(lhs)
    let right = priorityComponents(rhs)
    if left.source != right.source { return left.source < right.source }
    let leftActiveRank = lhs.activeRank ?? Int.max
    let rightActiveRank = rhs.activeRank ?? Int.max
    if leftActiveRank != rightActiveRank {
      return leftActiveRank < rightActiveRank
    }
    if left.penalty != right.penalty { return left.penalty < right.penalty }
    if left.media != right.media { return left.media < right.media }
    if left.hlsRole != right.hlsRole { return left.hlsRole < right.hlsRole }
    if left.depth != right.depth { return left.depth < right.depth }
    return left.order < right.order
  }

  /// Keeps player-attached sources first while reserving room in the Store's
  /// bounded attempts for direct HLS discovered by lower-tier observers.
  private static func resolutionOrder(
    _ candidates: [Candidate],
    activeURLKeys: Set<String>,
    supersededURLKeys: Set<String>,
    excludedURLKeys: Set<String>
  ) -> [Candidate] {
    let candidates = candidates.filter {
      !isHighConfidenceAdvertisementMediaURL($0.url)
        && !excludedURLKeys.contains($0.url.absoluteString)
    }
    func selectionTier(_ candidate: Candidate) -> Int {
      let key = candidate.url.absoluteString
      if activeURLKeys.contains(key) { return 0 }
      if supersededURLKeys.contains(key) { return 2 }
      return 1
    }
    let sorted = candidates.sorted { lhs, rhs in
      let leftTier = selectionTier(lhs)
      let rightTier = selectionTier(rhs)
      if leftTier != rightTier { return leftTier < rightTier }
      return isPreferred(lhs, over: rhs)
    }
    var leading = Array(sorted.prefix(8))
    var included = Set(leading.map { $0.url.absoluteString })
    for candidate in sorted where priorityComponents(candidate).media == 0 {
      guard
        leading.count < IPadBrowserMediaSourceSelector.maximumPlayableChoices
      else { break }
      if included.insert(candidate.url.absoluteString).inserted {
        leading.append(candidate)
      }
    }
    leading.append(
      contentsOf: sorted.filter {
        included.insert($0.url.absoluteString).inserted
      })
    return leading
  }

  private static func priorityComponents(
    _ candidate: Candidate
  ) -> (
    media: Int, source: Int, penalty: Int, hlsRole: Int, depth: Int, order: Int
  ) {
    let value = candidate.url.absoluteString.lowercased()
    let mediaPriority: Int
    let hlsRolePriority: Int
    if candidate.url.pathExtension.lowercased() == "m3u8"
      || value.contains(".m3u8?")
    {
      mediaPriority = 0
      let hasMasterPlaylistHint =
        value.contains("master.m3u8") || value.contains("manifest")
        || value.contains("playlist")
      hlsRolePriority = hasMasterPlaylistHint ? 0 : 1
    } else if ["mp4", "mov", "m4v"].contains(candidate.url.pathExtension.lowercased()) {
      mediaPriority = 1
      hlsRolePriority = 2
    } else {
      mediaPriority = 2
      hlsRolePriority = 2
    }

    let sourcePriority: Int
    switch candidate.sourceKind.lowercased() {
    case "active-current-source": sourcePriority = 0
    case "currentsrc": sourcePriority = 1
    case "video", "source": sourcePriority = 2
    case "navigation-response", "fetch-media-response", "xhr-media-response":
      sourcePriority = 3
    case "page-fetch-hls-response", "page-xhr-hls-response":
      sourcePriority = 4
    case "fetch-response", "xhr-response": sourcePriority = 5
    case "script-text": sourcePriority = 6
    case "fetch", "xhr", "page-fetch-hls-request", "page-xhr-hls-request":
      sourcePriority = 6
    case "setattribute": sourcePriority = 7
    case "performance": sourcePriority = 8
    case "frame", "iframe", "popup": sourcePriority = 9
    case "page": sourcePriority = 11
    default: sourcePriority = 10
    }
    return (
      media: mediaPriority,
      source: sourcePriority,
      penalty: disfavoredAdvertisementURL(candidate.url) ? 1 : 0,
      hlsRole: hlsRolePriority,
      depth: candidate.frameDepth,
      order: candidate.discoveryOrder
    )
  }

  private static func disfavoredAdvertisementURL(_ url: URL) -> Bool {
    if isHighConfidenceAdvertisementNavigationURL(url) { return true }
    let value = url.absoluteString.lowercased()
    let host = url.host?.lowercased() ?? ""
    if host.contains("doubleclick.") || host.contains("googlesyndication.")
      || host.contains("adservice.") || host == "imasdk.googleapis.com"
    {
      return true
    }
    return [
      "/ads/", "/ad/", "/preroll", "/pre-roll", "/vast", "/vmap",
      "advert", "imaad", "adtag",
    ].contains { value.contains($0) }
  }

  private static func worstCandidateIndex(in candidates: [Candidate]) -> Int? {
    candidates.indices.max { lhs, rhs in
      isPreferred(candidates[lhs], over: candidates[rhs])
    }
  }

  private static func referer(for candidateURL: URL, frameURL: URL?) -> URL? {
    guard let frameURL,
      var frame = URLComponents(url: frameURL, resolvingAgainstBaseURL: true),
      let candidate = URLComponents(url: candidateURL, resolvingAgainstBaseURL: true),
      let frameScheme = frame.scheme?.lowercased(), frameScheme == "https",
      let frameHost = frame.host?.lowercased(), !frameHost.isEmpty,
      let candidateScheme = candidate.scheme?.lowercased(),
      let candidateHost = candidate.host?.lowercased()
    else { return nil }

    frame.query = nil
    frame.fragment = nil
    frame.user = nil
    frame.password = nil
    let sameOrigin =
      frameScheme == candidateScheme
      && frameHost == candidateHost
      && effectivePort(frame) == effectivePort(candidate)
    if !sameOrigin { frame.path = "/" }
    return frame.url
  }

  private static func effectivePort(_ components: URLComponents) -> Int {
    if let port = components.port { return port }
    return components.scheme?.lowercased() == "https" ? 443 : 80
  }

  private static func isSameOrigin(_ lhs: URL, _ rhs: URL) -> Bool {
    guard let left = URLComponents(url: lhs, resolvingAgainstBaseURL: true),
      let right = URLComponents(url: rhs, resolvingAgainstBaseURL: true)
    else { return false }
    return left.scheme?.lowercased() == right.scheme?.lowercased()
      && left.host?.lowercased() == right.host?.lowercased()
      && effectivePort(left) == effectivePort(right)
  }

  private static func prioritizedCookies(
    _ cookies: [IPadMediaRequestCookie],
    for relevantURLs: [URL]
  ) -> [IPadMediaRequestCookie] {
    let relatedClearance = cookies.filter { cookie in
      cookie.name.lowercased() == "cf_clearance"
        && relevantURLs.contains { cookie.matches($0) }
    }
    let related = cookies.filter { cookie in
      cookie.name.lowercased() != "cf_clearance"
        && relevantURLs.contains { cookie.matches($0) }
    }
    return relatedClearance + related
  }

  private static func isVerifiedPageHLSObservation(_ sourceKind: String) -> Bool {
    switch sourceKind.lowercased() {
    case "navigation-response", "fetch-media-response", "xhr-media-response":
      return true
    default:
      return false
    }
  }

  private static func requiresBrowserOriginHeader(_ sourceKind: String) -> Bool {
    let normalized = sourceKind.lowercased()
    return normalized.contains("fetch") || normalized.contains("xhr")
  }

  private static let blankPageHTML = """
    <!doctype html><html><head><meta name="viewport" content="width=device-width"></head>
    <body style="background: transparent"></body></html>
    """

  private static let passiveInstrumentationScript = """
    (() => {
      'use strict';
      if (window.__miohInteractivePassiveInstalled) return;
      window.__miohInteractivePassiveInstalled = true;

      var frameDepth = 0;
      try {
        var cursor = window;
        while (cursor !== cursor.top && frameDepth < 16) {
          cursor = cursor.parent;
          frameDepth += 1;
        }
      } catch (_) {
        frameDepth = \(maximumFrameDepth);
      }
      if (frameDepth > \(maximumFrameDepth)) return;

      const documentTokenBytes = new Uint32Array(4);
      try { crypto.getRandomValues(documentTokenBytes); }
      catch (_) {
        for (let index = 0; index < documentTokenBytes.length; index += 1) {
          documentTokenBytes[index] = Math.floor(Math.random() * 0xffffffff);
        }
      }
      const documentToken = Array.from(documentTokenBytes)
        .map(value => value.toString(16).padStart(8, '0')).join('');
      try {
        Object.defineProperty(window, '__miohInteractiveDocumentToken', {
          value: documentToken,
          writable: false,
          configurable: false
        });
      } catch (_) {}
      let authorizedRouteToken = null;
      const seen = new Map();
      const mediaSlotStates = new WeakMap();
      const mediaIntersectionStates = new WeakMap();
      const trackedMediaSlots = new Map();
      const maximumMediaSlots = 32;
      const availableMediaSlotTokens = [];
      const mediaSlotGenerations = new Map();
      let nextMediaSlot = 0;
      const mediaPattern = /(?:\\.m3u8|\\.mp4|\\.mov|\\.m4v)(?:$|[?#])/i;
      const interestingPattern = /(?:manifest|playlist|master\\.m3u8)/i;
      const sourceAttributes = [
        'src', 'href', 'data-src', 'data-url', 'data-hls', 'data-stream',
        'data-file', 'data-video', 'data-lazy-src', 'data-original-src'
      ];
      const mediaElementSelector = 'video,source,iframe';
      const maximumElementsPerScan = 256;
      const maximumScriptElementsPerScan = 64;
      const maximumScriptTextCharactersPerScan = 262144;
      const maximumTextMediaMatchesPerScan = 64;
      const maximumResourceEntriesPerBatch = 256;
      const advertisementOverlayMarker = 'data-mioh-hidden-advertisement';
      const advertisementNavigationHostSuffixes = [
        'turnhub.net', 'tsyndicate.com', 'javhd-trk.com', 'nettrck.store',
        'doubleclick.net', 'googlesyndication.com', 'popads.net', 'popcash.net'
      ];
      const advertisementMediaHostSuffixes = ['saawsedge.com'];
      const maximumAdvertisementElementsPerScan = 64;
      const challengeObservationEnabled = \(relaxedWebCompatibilityEnabled ? "false" : "true");
      const heartbeatIntervalMilliseconds = \(relaxedWebCompatibilityEnabled ? "2000" : "1000");
      let challengeVisible = false;
      let challengeEpochCounter = 0;
      let activeChallengeEpoch = null;
      const acknowledgedChallengeTokens = new Set();

      const advertisementHost = raw => {
        if (!raw || typeof raw !== 'string') return false;
        try {
          const resolved = new URL(raw, document.baseURI);
          if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') {
            return null;
          }
          return resolved.hostname.toLowerCase().replace(/\\.$/, '');
        } catch (_) {
          return null;
        }
      };
      const hostMatchesAdvertisementSuffix = (host, suffixes) => Boolean(host)
        && suffixes.some(suffix => host === suffix || host.endsWith(`.${suffix}`));
      const highConfidenceAdvertisementEvidence = element => {
        if (!element?.getAttribute) return null;
        const tag = (element.tagName || '').toLowerCase();
        let raw = null;
        let allowsAdvertisementMedia = false;
        if (tag === 'a') {
          raw = element.getAttribute('href');
        } else if (tag === 'iframe') {
          raw = element.getAttribute('src');
        } else if (tag === 'video') {
          raw = element.currentSrc || element.getAttribute('src');
          allowsAdvertisementMedia = true;
        } else if (tag === 'source') {
          raw = element.getAttribute('src');
          allowsAdvertisementMedia = true;
        } else {
          return null;
        }
        const host = advertisementHost(raw);
        const navigationAdvertisement = hostMatchesAdvertisementSuffix(
          host, advertisementNavigationHostSuffixes
        );
        const mediaAdvertisement = hostMatchesAdvertisementSuffix(
          host, advertisementMediaHostSuffixes
        );
        if (!navigationAdvertisement
          && !(allowsAdvertisementMedia && mediaAdvertisement)
          && !(tag === 'iframe' && mediaAdvertisement)) return null;
        return {allowsAdvertisementMedia};
      };
      const isCompactLowerRightOverlay = element => {
        if (!element?.getBoundingClientRect) return false;
        try {
          const rect = element.getBoundingClientRect();
          const viewportWidth = Math.max(1, Number(window.innerWidth) || 0);
          const viewportHeight = Math.max(1, Number(window.innerHeight) || 0);
          const width = Math.max(0, Number(rect.width) || 0);
          const height = Math.max(0, Number(rect.height) || 0);
          if (width < 16 || height < 16
            || width > viewportWidth * 0.65
            || height > viewportHeight * 0.9
            || width * height > viewportWidth * viewportHeight * 0.45) {
            return false;
          }
          return rect.right >= viewportWidth * 0.7
            && rect.bottom >= viewportHeight * 0.7
            && rect.left >= viewportWidth * 0.25
            && rect.top >= viewportHeight * 0.05;
        } catch (_) {
          return false;
        }
      };
      const advertisementOverlayRoot = (source, evidence) => {
        let cursor = source;
        let outermostCandidate = null;
        for (let depth = 0; cursor && depth < 6; depth += 1) {
          const tag = (cursor.tagName || '').toLowerCase();
          if (tag === 'body' || tag === 'html') break;
          if (!evidence.allowsAdvertisementMedia
            && (tag === 'video' || tag === 'audio'
            || cursor.querySelector?.('video,audio'))) break;
          try {
            const position = getComputedStyle(cursor).position;
            if ((position === 'fixed' || position === 'absolute')
              && isCompactLowerRightOverlay(cursor)) {
              outermostCandidate = cursor;
            }
          } catch (_) {}
          cursor = cursor.parentElement;
        }
        return outermostCandidate;
      };
      const suppressAdvertisementElement = source => {
        const evidence = highConfidenceAdvertisementEvidence(source);
        if (!evidence) return;
        const root = advertisementOverlayRoot(source, evidence);
        if (!root) return;
        try {
          source.setAttribute(advertisementOverlayMarker, 'true');
          root.setAttribute(advertisementOverlayMarker, 'true');
          root.style.setProperty('display', 'none', 'important');
          root.style.setProperty('visibility', 'hidden', 'important');
          root.style.setProperty('pointer-events', 'none', 'important');
          root.setAttribute('aria-hidden', 'true');
        } catch (_) {}
      };
      const suppressHighConfidenceAdvertisementOverlays = root => {
        const selector = 'a[href],iframe[src],video[src],source[src]';
        let inspected = 0;
        if (root?.matches?.(selector)) {
          suppressAdvertisementElement(root);
          inspected += 1;
        }
        for (const element of root?.querySelectorAll?.(selector) || []) {
          if (inspected >= maximumAdvertisementElementsPerScan) break;
          suppressAdvertisementElement(element);
          inspected += 1;
        }
      };
      const isCompactFloatingMediaOverlay = element => {
        let cursor = element;
        for (let depth = 0; cursor && depth < 7; depth += 1) {
          const tag = (cursor.tagName || '').toLowerCase();
          if (tag === 'body' || tag === 'html') break;
          try {
            const position = getComputedStyle(cursor).position;
            if ((position === 'fixed' || position === 'sticky')
              && isCompactLowerRightOverlay(cursor)) return true;
          } catch (_) {}
          cursor = cursor.parentElement;
        }
        return false;
      };

      let mediaIntersectionObserver = null;
      let intersectionVisualVisibilityEnabled = false;
      const intersectionCallback = entries => {
        for (const entry of entries || []) {
          const rect = entry.intersectionRect;
          const width = Math.max(0, Math.min(Number(rect?.width) || 0, 10000));
          const height = Math.max(0, Math.min(Number(rect?.height) || 0, 10000));
          mediaIntersectionStates.set(entry.target, {
            isIntersecting: entry.isIntersecting === true,
            renderedArea: Math.min(100000000, Math.round(width * height)),
            isVisuallyVisible: intersectionVisualVisibilityEnabled
              && entry.isVisible === true
          });
          reportMediaSource(entry.target, true, false);
        }
      };
      if (typeof IntersectionObserver === 'function') {
        const thresholds = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1];
        const supportsVisualVisibility =
          typeof IntersectionObserverEntry !== 'undefined'
          && 'isVisible' in IntersectionObserverEntry.prototype;
        if (supportsVisualVisibility) {
          try {
            mediaIntersectionObserver = new IntersectionObserver(
              intersectionCallback,
              {threshold: thresholds, trackVisibility: true, delay: 100}
            );
            intersectionVisualVisibilityEnabled = true;
          } catch (_) {}
        }
        if (!mediaIntersectionObserver) {
          try {
            mediaIntersectionObserver = new IntersectionObserver(
              intersectionCallback,
              {threshold: thresholds}
            );
          } catch (_) {}
        }
      }

      const post = body => {
        const message = Object.assign({}, body, {documentToken});
        if (authorizedRouteToken) message.routeToken = authorizedRouteToken;
        try { window.webkit.messageHandlers.miohInteractiveMediaBrowser.postMessage(message); }
        catch (_) {}
      };
      let trustedActivationSerial = 0;
      let consumedActivationSerial = 0;
      let trustedActivationTime = -Infinity;
      const recordTrustedActivation = event => {
        if (event?.isTrusted !== true) return;
        trustedActivationSerial += 1;
        trustedActivationTime = performance.now();
      };
      document.addEventListener('click', recordTrustedActivation, true);
      document.addEventListener('keydown', event => {
        if (event?.key === 'Enter' || event?.key === ' ') {
          recordTrustedActivation(event);
        }
      }, true);
      const consumeTrustedActivation = () => {
        const age = performance.now() - trustedActivationTime;
        if (trustedActivationSerial <= consumedActivationSerial
          || !Number.isFinite(age) || age < 0 || age > 3500) {
          return false;
        }
        consumedActivationSerial = trustedActivationSerial;
        return true;
      };
      try {
        Object.defineProperty(window, '__miohInteractiveConsumeTrustedActivation', {
          value: consumeTrustedActivation,
          writable: false,
          configurable: false
        });
      } catch (_) {}
      const normalizedHTTPSURL = raw => {
        if (!raw || typeof raw !== 'string') return null;
        try {
          const resolved = new URL(raw, document.baseURI);
          if (resolved.protocol !== 'https:') return null;
          resolved.hash = '';
          if (resolved.href.length > \(maximumCandidateURLLength)) return null;
          return resolved.href;
        } catch (_) {
          return null;
        }
      };
      const reportMediaSource = (element, force, forceCleared) => {
        if (!element || (element.tagName || '').toLowerCase() !== 'video') return;
        let slot = mediaSlotStates.get(element);
        if (!slot) {
          let token = availableMediaSlotTokens.pop();
          if (!token && nextMediaSlot >= maximumMediaSlots) return;
          if (!token) nextMediaSlot += 1;
          const selectedToken = token || `v${nextMediaSlot.toString(36)}`;
          slot = {
            token: selectedToken,
            generation: mediaSlotGenerations.get(selectedToken) || 0,
            currentURL: null,
            sourceIdentity: null,
            signature: null
          };
          mediaSlotStates.set(element, slot);
          trackedMediaSlots.set(slot.token, element);
          try { mediaIntersectionObserver?.observe(element); } catch (_) {}
        }
        if (!forceCleared && element.isConnected) {
          trackedMediaSlots.set(slot.token, element);
        }
        const rawCurrentSource = forceCleared
          ? '' : String(element.currentSrc || element.src || '');
        const currentURL = forceCleared
          ? null : normalizedHTTPSURL(rawCurrentSource);
        const opaqueSourceToken = !currentURL && rawCurrentSource.startsWith('blob:')
          ? rawCurrentSource : null;
        const sourceIdentity = currentURL || opaqueSourceToken;
        if (sourceIdentity !== slot.sourceIdentity) {
          slot.generation += 1;
          mediaSlotGenerations.set(slot.token, slot.generation);
          slot.sourceIdentity = sourceIdentity;
          slot.currentURL = currentURL;
          slot.signature = null;
        }
        if (slot.generation < 1) return;

        const rawDuration = Number(element.duration);
        const duration = Number.isFinite(rawDuration) && rawDuration > 0
          ? Math.min(rawDuration, 604800) : null;
        const rawCurrentTime = Number(element.currentTime);
        const currentTime = Number.isFinite(rawCurrentTime) && rawCurrentTime >= 0
          ? Math.min(Math.floor(rawCurrentTime / 2) * 2, 604800) : 0;
        let renderedArea = 0;
        let isVisible = false;
        let visibilityAttested = false;
        try {
          const intersection = mediaIntersectionStates.get(element);
          renderedArea = intersection?.renderedArea || 0;
          let localTreeVisible = element.isConnected;
          let accumulatedOpacity = 1;
          let cursor = element;
          const visited = new Set();
          while (localTreeVisible && cursor && !visited.has(cursor)) {
            visited.add(cursor);
            const style = getComputedStyle(cursor);
            const opacity = Number(style.opacity);
            if (style.display === 'none' || style.visibility === 'hidden'
              || style.contentVisibility === 'hidden') {
              localTreeVisible = false;
              break;
            }
            accumulatedOpacity *= Number.isFinite(opacity) ? opacity : 1;
            if (accumulatedOpacity < 0.05) {
              localTreeVisible = false;
              break;
            }
            cursor = cursor.parentElement || cursor.getRootNode?.()?.host || null;
          }
          // Geometry alone does not prove that another element is not
          // covering the video. Require WebKit's visual-visibility proof in
          // every frame; older engines safely remain on the strict public-IP
          // resolver policy.
          const visualProof = intersectionVisualVisibilityEnabled
            && intersection?.isVisuallyVisible === true;
          visibilityAttested = Boolean(intersection && visualProof);
          isVisible = visibilityAttested && localTreeVisible
            && intersection.isIntersecting === true && renderedArea >= 4;
        } catch (_) {}
        const isPlaying = !element.paused && !element.ended && element.readyState >= 2;
        const isEnded = element.ended === true;
        const isCompactFloatingOverlay = isCompactFloatingMediaOverlay(element);
        const state = currentURL ? 'current'
          : opaqueSourceToken ? 'opaque' : 'cleared';
        const signature = [
          slot.generation, state, currentURL || '', duration || 0, currentTime,
          isPlaying ? 1 : 0, isEnded ? 1 : 0, isVisible ? 1 : 0,
          visibilityAttested ? 1 : 0, Math.floor(renderedArea / 256),
          isCompactFloatingOverlay ? 1 : 0
        ].join('|');
        if (!force && signature === slot.signature) return;
        slot.signature = signature;
        post({
          type: 'media-source',
          slotToken: slot.token,
          sourceGeneration: slot.generation,
          state,
          url: currentURL || '',
          frameDepth,
          duration,
          currentTime,
          isPlaying,
          isEnded,
          isVisible,
          visibilityAttested,
          renderedArea,
          isCompactFloatingOverlay
        });
      };
      const emit = (raw, kind, isCompactFloatingOverlay = false) => {
        if (!raw || typeof raw !== 'string') return;
        let resolved;
        try { resolved = new URL(raw, document.baseURI); } catch (_) { return; }
        if (resolved.protocol !== 'https:') return;
        resolved.hash = '';
        const value = resolved.href;
        const sourceKind = String(kind || 'unknown').toLowerCase();
        const evidenceRank = (() => {
          if (sourceKind === 'video' || sourceKind === 'source'
            || sourceKind === 'currentsrc') return 0;
          if (sourceKind === 'fetch-media-response'
            || sourceKind === 'xhr-media-response') return 1;
          if (sourceKind === 'page-fetch-hls-response'
            || sourceKind === 'page-xhr-hls-response') return 2;
          if (sourceKind === 'fetch-response'
            || sourceKind === 'xhr-response') return 3;
          if (sourceKind === 'fetch' || sourceKind === 'xhr'
            || sourceKind === 'page-fetch-hls-request'
            || sourceKind === 'page-xhr-hls-request') return 4;
          if (sourceKind === 'script-text') return 5;
          if (sourceKind === 'setattribute') return 6;
          if (sourceKind === 'performance') return 7;
          if (sourceKind === 'frame' || sourceKind === 'iframe'
            || sourceKind === 'popup') return 8;
          return 9;
        })();
        const priorRank = seen.get(value);
        if (Number.isInteger(priorRank) && priorRank <= evidenceRank) return;
        seen.set(value, evidenceRank);
        post({
          type: 'candidate', url: value, kind: sourceKind, frameDepth,
          isCompactFloatingOverlay: isCompactFloatingOverlay === true
        });
      };
      const inspectValue = (value, kind, isCompactFloatingOverlay = false) => {
        if (typeof value !== 'string') return;
        if (mediaPattern.test(value) || interestingPattern.test(value)) {
          emit(value, kind, isCompactFloatingOverlay);
        }
      };
      const inspectTextForMediaURLs = (text, kind) => {
        if (typeof text !== 'string' || !text) return;
        const decoded = text.slice(0, maximumScriptTextCharactersPerScan)
          .replace(/\\\\\\//g, '/')
          .replace(/\\u002[fF]/g, '/')
          .replace(/\\u003[aA]/g, ':')
          .replace(/\\u0026/g, '&');
        const pattern = /(?:(?:https?:)?\\/\\/|\\/|\\.\\.?\\/)[^"'<>\\s]+?\\.(?:m3u8|mp4|mov|m4v)(?:[^"'<>\\s]*)?/ig;
        let matchCount = 0;
        let match;
        while ((match = pattern.exec(decoded))
          && matchCount < maximumTextMediaMatchesPerScan) {
          matchCount += 1;
          emit(match[0], kind);
        }
      };
      const installedPageNetworkBridges = new Map();
      const pageNetworkBridgeSeen = new Set();
      const maximumPageNetworkBridgeEvents = 128;
      const maximumPageNetworkBridgeLifetimeEvents = 512;
      const maximumPendingPageNetworkBridgeEvents = 32;
      let pageNetworkBridgeEventCount = 0;
      let pageNetworkBridgeLifetimeEventCount = 0;
      const pendingPageNetworkBridgePayloads = [];
      const pendingPageNetworkBridgeKeys = new Set();
      const allowedPageNetworkKinds = new Set([
        'page-fetch-hls-request', 'page-fetch-hls-response',
        'page-xhr-hls-request', 'page-xhr-hls-response'
      ]);
      const acceptPageNetworkBridgePayload = (payload, countsTowardLifetime) => {
        if (!authorizedRouteToken
          || pageNetworkBridgeEventCount >= maximumPageNetworkBridgeEvents
          || (countsTowardLifetime
            && pageNetworkBridgeLifetimeEventCount
              >= maximumPageNetworkBridgeLifetimeEvents)) {
          return;
        }
        const currentPageURL = String(location.href);
        if (payload.pageURL !== currentPageURL) return;
        const key = `${payload.kind}\n${payload.value}`;
        if (pageNetworkBridgeSeen.has(key)) return;
        pageNetworkBridgeSeen.add(key);
        pageNetworkBridgeEventCount += 1;
        if (countsTowardLifetime) pageNetworkBridgeLifetimeEventCount += 1;
        if (payload.hlsResponse === true
          && (payload.kind === 'page-fetch-hls-response'
            || payload.kind === 'page-xhr-hls-response')) {
          // A page-world response can be forged by the page. It is emitted
          // only as a low-trust URL hint and is always fetched and parsed
          // again by the native public-network resolver.
          emit(payload.value, payload.kind);
        } else {
          inspectValue(payload.value, payload.kind);
        }
      };
      const installPageNetworkBridge = (rawEventName, rawObservationEpoch) => {
        const eventName = String(rawEventName || '');
        const observationEpoch = String(rawObservationEpoch || '');
        if (!/^mioh-hls-[a-f0-9-]{36}$/.test(eventName)
          || !/^[a-f0-9-]{36}$/.test(observationEpoch)) return false;
        if (installedPageNetworkBridges.has(eventName)) {
          const epochs = installedPageNetworkBridges.get(eventName);
          if (epochs.current !== observationEpoch) {
            epochs.previous = epochs.current;
            epochs.current = observationEpoch;
            // A user-initiated analysis refresh starts another bounded
            // observation window for long, multi-stage players.
            pageNetworkBridgeLifetimeEventCount = 0;
          }
          return true;
        }
        if (installedPageNetworkBridges.size >= 1) return false;
        document.addEventListener(eventName, event => {
          if (typeof event?.detail !== 'string'
            || event.detail.length > \(maximumCandidateURLLength + 512)) {
            return;
          }
          let payload;
          try { payload = JSON.parse(event.detail); } catch (_) { return; }
          const kind = String(payload?.kind || '');
          const value = String(payload?.url || '');
          const pageURL = String(payload?.pageURL || '');
          const payloadEpoch = String(payload?.observationEpoch || '');
          const acceptedEpochs = installedPageNetworkBridges.get(eventName);
          if (!allowedPageNetworkKinds.has(kind)
            || !value || value.length > \(maximumCandidateURLLength)
            || !pageURL || pageURL.length > \(maximumCandidateURLLength)
            || (payloadEpoch !== acceptedEpochs?.current
              && payloadEpoch !== acceptedEpochs?.previous)) {
            return;
          }
          const normalizedPayload = {
            kind,
            value,
            pageURL,
            hlsResponse: payload?.hlsResponse === true
          };
          const key = `${kind}\n${pageURL}\n${value}`;
          // A page-world HLS response can be the first observable signal of
          // a pushState/replaceState player handoff. Notify native before the
          // candidate is accepted so the payload waits behind the new route
          // authorization instead of being rejected with the old route.
          window.__miohInteractiveNotifyPageChange?.();
          if (!authorizedRouteToken) {
            if (pendingPageNetworkBridgePayloads.length
                >= maximumPendingPageNetworkBridgeEvents
              || pageNetworkBridgeLifetimeEventCount
                >= maximumPageNetworkBridgeLifetimeEvents
              || pendingPageNetworkBridgeKeys.has(key)) return;
            pendingPageNetworkBridgeKeys.add(key);
            pendingPageNetworkBridgePayloads.push(normalizedPayload);
            pageNetworkBridgeLifetimeEventCount += 1;
            return;
          }
          acceptPageNetworkBridgePayload(normalizedPayload, true);
        }, true);
        installedPageNetworkBridges.set(eventName, {
          current: observationEpoch,
          previous: null
        });
        return true;
      };
      try {
        Object.defineProperty(window, '__miohInteractiveInstallPageNetworkBridge', {
          value: installPageNetworkBridge,
          writable: false,
          configurable: false
        });
      } catch (_) {}
      const inspectElement = element => {
        if (!element || !element.getAttribute) return;
        const tag = (element.tagName || '').toLowerCase();
        const floatingMedia = tag === 'video'
          ? isCompactFloatingMediaOverlay(element)
          : tag === 'source'
            ? isCompactFloatingMediaOverlay(element.parentElement) : false;
        for (const attribute of sourceAttributes) {
          const value = element.getAttribute(attribute);
          if (!value) continue;
          if (tag === 'iframe') emit(value, 'iframe');
          else inspectValue(value, tag || attribute, floatingMedia);
        }
        if (tag === 'video') {
          inspectValue(element.src, 'video', floatingMedia);
          reportMediaSource(element, false, false);
        } else if (tag === 'source') {
          inspectValue(element.src, 'source', floatingMedia);
        } else if (tag === 'script') {
          inspectTextForMediaURLs(element.textContent, 'script-text');
        }
      };
      const reportChallengeState = () => {
        if (!challengeObservationEnabled || window !== window.top) return;
        const challengeResponseValues = Array.from(document.querySelectorAll(
          'input[name="cf-turnstile-response"], textarea[name="cf-turnstile-response"]'
        )).map(element => String(element.value || '').trim());
        const nonemptyChallengeTokens = challengeResponseValues.filter(Boolean);
        const allResponseFieldsComplete = challengeResponseValues.length > 0
          && challengeResponseValues.every(value => value.length > 0);
        const hardChallenge = Boolean(document.querySelector(
          '#challenge-running, #challenge-stage, form#challenge-form'
        ));
        const embeddedChallenge = Boolean(document.querySelector(
          '.cf-turnstile, iframe[src*="challenges.cloudflare.com"], iframe[src*="/cdn-cgi/challenge-platform/"]'
        ));
        const present = hardChallenge || (embeddedChallenge && !allResponseFieldsComplete);
        if (present && !challengeVisible) {
          challengeVisible = true;
          challengeEpochCounter += 1;
          activeChallengeEpoch = `${documentToken}:${challengeEpochCounter.toString(36)}`;
          for (const token of nonemptyChallengeTokens) {
            acknowledgedChallengeTokens.add(token);
          }
        }
        if (present && activeChallengeEpoch) {
          post({type: 'challenge-hint', challengeEpoch: activeChallengeEpoch});
          return;
        }
        if (activeChallengeEpoch && !present && allResponseFieldsComplete) {
          const freshChallengeTokens = new Set(
            nonemptyChallengeTokens.filter(
              token => !acknowledgedChallengeTokens.has(token)
            )
          );
          if (freshChallengeTokens.size > 0) {
            for (const token of nonemptyChallengeTokens) {
              acknowledgedChallengeTokens.add(token);
            }
            const completedEpoch = activeChallengeEpoch;
            activeChallengeEpoch = null;
            challengeVisible = false;
            post({
              type: 'challenge-cleared',
              completed: true,
              challengeEpoch: completedEpoch
            });
            return;
          }
        }
        if (!present) {
          challengeVisible = false;
          if (!activeChallengeEpoch) {
            for (const token of nonemptyChallengeTokens) {
              acknowledgedChallengeTokens.add(token);
            }
          }
          return;
        }
      };
      const recentResourceEntries = [];
      const inspectResourceEntries = entries => {
        let entryCount = 0;
        let pageURL = null;
        try {
          entryCount = Math.max(0, Number(entries?.length) || 0);
          pageURL = String(location.href);
        } catch (_) { return; }
        const start = Math.max(0, entryCount - maximumResourceEntriesPerBatch);
        for (let index = start; index < entryCount; index += 1) {
          const name = entries[index]?.name;
          if (typeof name !== 'string') continue;
          recentResourceEntries.push({name, pageURL});
          inspectValue(name, 'performance');
        }
        if (recentResourceEntries.length > maximumResourceEntriesPerBatch) {
          recentResourceEntries.splice(
            0,
            recentResourceEntries.length - maximumResourceEntriesPerBatch
          );
        }
      };
      const replayRecentResourceEntries = () => {
        let pageURL;
        try { pageURL = String(location.href); } catch (_) { return; }
        for (const entry of recentResourceEntries) {
          if (entry.pageURL === pageURL) inspectValue(entry.name, 'performance');
        }
      };
      const inspectCurrentPerformanceResources = () => {
        try {
          if (typeof performance === 'undefined'
            || typeof performance.getEntriesByType !== 'function') return;
          inspectResourceEntries(performance.getEntriesByType('resource'));
        } catch (_) {}
      };
      const retireDisconnectedMediaSlots = () => {
        for (const [slotToken, element] of trackedMediaSlots) {
          if (element.isConnected) continue;
          reportMediaSource(element, true, true);
          try { mediaIntersectionObserver?.unobserve(element); } catch (_) {}
          mediaSlotStates.delete(element);
          trackedMediaSlots.delete(slotToken);
          if (availableMediaSlotTokens.length < maximumMediaSlots) {
            availableMediaSlotTokens.push(slotToken);
          }
        }
      };
      const scan = () => {
        reportChallengeState();
        suppressHighConfidenceAdvertisementOverlays(document);
        let inspectedElementCount = 0;
        for (const element of document.querySelectorAll(mediaElementSelector)) {
          if (inspectedElementCount >= maximumElementsPerScan) break;
          inspectElement(element);
          inspectedElementCount += 1;
        }
        let inspectedScriptCount = 0;
        let remainingScriptTextBudget = maximumScriptTextCharactersPerScan;
        for (const script of document.scripts || []) {
          if (inspectedScriptCount >= maximumScriptElementsPerScan
            || remainingScriptTextBudget <= 0) break;
          inspectValue(script.src, 'script');
          const text = String(script.textContent || '');
          if (text) {
            inspectTextForMediaURLs(
              text.slice(0, remainingScriptTextBudget),
              'script-text'
            );
            remainingScriptTextBudget -= text.length;
          }
          inspectedScriptCount += 1;
        }
        retireDisconnectedMediaSlots();
      };
      let scanScheduled = false;
      const scheduleScan = () => {
        if (scanScheduled) return;
        scanScheduled = true;
        const run = () => {
          scanScheduled = false;
          scan();
        };
        try {
          if (typeof requestIdleCallback === 'function') {
            requestIdleCallback(run, {timeout: 500});
          } else {
            setTimeout(run, 100);
          }
        } catch (_) {
          setTimeout(run, 100);
        }
      };

      window.__miohInteractiveEmit = emit;
      window.__miohInteractiveInspectValue = inspectValue;
      window.__miohInteractiveScan = scan;
      let observedPageURL = String(location.href);
      window.__miohInteractiveNotifyPageChange = () => {
        const currentPageURL = String(location.href);
        if (currentPageURL === observedPageURL) return;
        observedPageURL = currentPageURL;
        authorizedRouteToken = null;
        seen.clear();
        for (const element of trackedMediaSlots.values()) {
          const slot = mediaSlotStates.get(element);
          if (slot) slot.signature = null;
        }
        post({
          type: 'page-location',
          url: currentPageURL.slice(0, \(maximumCandidateURLLength))
        });
      };
      window.addEventListener(
        'popstate',
        () => window.__miohInteractiveNotifyPageChange?.()
      );
      window.addEventListener(
        'hashchange',
        () => window.__miohInteractiveNotifyPageChange?.()
      );
      try {
        window.navigation?.addEventListener?.(
          'currententrychange',
          () => window.__miohInteractiveNotifyPageChange?.()
        );
      } catch (_) {}
      const authorizeRoute = rawToken => {
        const token = String(rawToken || '');
        if (!token || token.length > \(maximumScriptTokenLength)) return;
        const routeChanged = authorizedRouteToken !== token;
        authorizedRouteToken = token;
        if (routeChanged) {
          seen.clear();
          pageNetworkBridgeSeen.clear();
          pageNetworkBridgeEventCount = 0;
          for (const element of trackedMediaSlots.values()) {
            const slot = mediaSlotStates.get(element);
            if (slot) slot.signature = null;
          }
          const pending = pendingPageNetworkBridgePayloads.splice(0);
          pendingPageNetworkBridgeKeys.clear();
          for (const payload of pending) {
            acceptPageNetworkBridgePayload(payload, false);
          }
          inspectCurrentPerformanceResources();
          // The visible WebView can adopt an already-loaded popup after its
          // document-start notification was intentionally ignored. Re-emit
          // exactly once for the new route token so native can register that
          // document without creating a ready/activate feedback loop.
          post({type: 'frame-ready', frameDepth});
        }
        if (window !== window.top) emit(location.href, 'frame');
        if (routeChanged) scan();
      };
      try {
        Object.defineProperty(window, '__miohInteractiveAuthorizeRoute', {
          value: authorizeRoute,
          writable: false,
          configurable: false
        });
      } catch (_) {
        window.__miohInteractiveAuthorizeRoute = authorizeRoute;
      }
      const rescanAfterFrameRegistration = rawToken => {
        const token = String(rawToken || '');
        if (!token || token !== authorizedRouteToken) return false;
        seen.clear();
        for (const element of trackedMediaSlots.values()) {
          const slot = mediaSlotStates.get(element);
          if (slot) slot.signature = null;
        }
        inspectCurrentPerformanceResources();
        replayRecentResourceEntries();
        scan();
        return true;
      };
      try {
        Object.defineProperty(window, '__miohInteractiveRescanRegisteredFrame', {
          value: rescanAfterFrameRegistration,
          writable: false,
          configurable: false
        });
      } catch (_) {
        window.__miohInteractiveRescanRegisteredFrame = rescanAfterFrameRegistration;
      }

      try {
        const observer = new MutationObserver(mutations => {
          let remainingElementBudget = maximumElementsPerScan;
          let exceededElementBudget = false;
          for (const mutation of mutations) {
            if (remainingElementBudget <= 0) {
              exceededElementBudget = true;
              break;
            }
            if (mutation.type === 'attributes') {
              if (sourceAttributes.includes(mutation.attributeName)) {
                inspectElement(mutation.target);
              }
              suppressHighConfidenceAdvertisementOverlays(mutation.target);
              remainingElementBudget -= 1;
            }
            for (const node of mutation.addedNodes || []) {
              if (remainingElementBudget <= 0) {
                exceededElementBudget = true;
                break;
              }
              if (node.nodeType !== Node.ELEMENT_NODE) continue;
              inspectElement(node);
              suppressHighConfidenceAdvertisementOverlays(node);
              remainingElementBudget -= 1;
              for (const nested of node.querySelectorAll?.(mediaElementSelector) || []) {
                if (remainingElementBudget <= 0) {
                  exceededElementBudget = true;
                  break;
                }
                inspectElement(nested);
                remainingElementBudget -= 1;
              }
            }
          }
          if (exceededElementBudget) scheduleScan();
        });
        observer.observe(document, {
          subtree: true,
          childList: true,
          attributes: true,
          attributeFilter: [...sourceAttributes, 'class', 'style']
        });
      } catch (_) {}
      for (const eventName of [
        'loadstart', 'loadedmetadata', 'durationchange', 'canplay', 'playing',
        'pause', 'ended', 'emptied', 'abort', 'resize'
      ]) {
        document.addEventListener(eventName, event => {
          const element = event.target;
          if ((element?.tagName || '').toLowerCase() === 'video') {
            reportMediaSource(element, false, eventName === 'emptied');
          }
        }, true);
      }
      try {
        const performanceObserver = new PerformanceObserver(list => {
          inspectResourceEntries(list.getEntries());
        });
        performanceObserver.observe({type: 'resource', buffered: true});
      } catch (_) {}
      inspectCurrentPerformanceResources();

      post({type: 'frame-ready', frameDepth});
      if (window !== window.top) emit(location.href, 'frame');
      document.addEventListener('DOMContentLoaded', scan, {once: true});
      window.addEventListener('load', scan, {once: true});
      setInterval(() => {
        window.__miohInteractiveNotifyPageChange?.();
        inspectCurrentPerformanceResources();
        reportChallengeState();
        for (const element of trackedMediaSlots.values()) {
          reportMediaSource(element, false, false);
        }
        retireDisconnectedMediaSlots();
        post({type: 'frame-heartbeat', frameDepth});
      }, heartbeatIntervalMilliseconds);
      scan();
    })();
    """

  /// Chrome HLS extensions observe the browser network stack. WKWebView does
  /// not expose an equivalent API for HTTPS subresources, so this narrow
  /// page-world hook observes only GET/HEAD fetch and XHR HLS request/response
  /// hints. The page can forge these events; native code therefore keeps them
  /// unverified and sends every selected URL through the public-network
  /// resolver again.
  private static let pageNetworkObservationScript = """
    (() => {
      'use strict';
      const eventName = String(bridgeEventName || '');
      const initialObservationEpoch = String(observationEpoch || '');
      if (!/^mioh-hls-[a-f0-9-]{36}$/.test(eventName)
        || !/^[a-f0-9-]{36}$/.test(initialObservationEpoch)) return false;

      const installationKey = `__mioh_hls_${eventName
        .slice('mioh-hls-'.length).replace(/-/g, '_')}`;
      let installations;
      try {
        installations = window[installationKey];
        if (!(installations instanceof Map)) {
          installations = new Map();
          Object.defineProperty(window, installationKey, {
            value: installations,
            writable: false,
            configurable: false
          });
        }
      } catch (_) {
        return false;
      }
      if (installations.has(eventName)) {
        const existing = installations.get(eventName);
        if (existing.observationEpoch !== initialObservationEpoch) {
          existing.seen.clear();
          existing.routeEventCount = 0;
          existing.lifetimeEventCount = 0;
          existing.routePageURL = null;
          existing.observationEpoch = initialObservationEpoch;
        }
        return true;
      }
      if (installations.size >= 1) return false;

      const maximumRouteEvents = 128;
      const maximumLifetimeEvents = 512;
      const maximumURLLength = \(maximumCandidateURLLength);
      const observationState = {
        seen: new Set(),
        routeEventCount: 0,
        lifetimeEventCount: 0,
        routePageURL: null,
        observationEpoch: initialObservationEpoch
      };
      const hlsURLPattern = /(?:\\.m3u8)(?:$|[?#])/i;
      const hlsHintPattern = /(?:master\\.m3u8|manifest|playlist)/i;
      const isHLSContentType = raw => {
        const value = String(raw || '').split(';', 1)[0].trim().toLowerCase();
        return [
          'application/vnd.apple.mpegurl',
          'application/x-mpegurl',
          'application/mpegurl',
          'audio/mpegurl',
          'audio/x-mpegurl'
        ].includes(value);
      };
      const isReadOnlyRequest = rawMethod => {
        if (typeof rawMethod !== 'string') return false;
        const method = rawMethod.trim().toUpperCase();
        return method === 'GET' || method === 'HEAD';
      };
      const fetchUsesReadOnlyMethod = argumentsList => {
        const input = argumentsList[0];
        const init = argumentsList.length > 1 ? argumentsList[1] : undefined;
        if (init != null && init.method !== undefined) {
          return isReadOnlyRequest(init.method);
        }
        if (input != null && typeof input === 'object'
          && input.method !== undefined) {
          return isReadOnlyRequest(input.method);
        }
        // URL/string fetch without an explicit method follows Fetch's GET
        // default. Explicit null/non-string methods are not treated as GET.
        return init == null || init.method === undefined;
      };
      const normalizedURL = raw => {
        if (raw == null) return null;
        let value;
        try {
          value = typeof raw === 'string' ? raw : raw.url || String(raw);
        } catch (_) {
          return null;
        }
        if (!value || value.length > maximumURLLength) return null;
        try {
          const resolved = new URL(value, document.baseURI);
          if (resolved.protocol !== 'https:') return null;
          resolved.hash = '';
          return resolved.href.length <= maximumURLLength ? resolved.href : null;
        } catch (_) {
          return null;
        }
      };
      const currentPageURL = () => {
        const value = String(location.href || '');
        if (!value || value.length > maximumURLLength) return null;
        try {
          const resolved = new URL(value);
          return resolved.protocol === 'https:' ? resolved.href : null;
        } catch (_) {
          return null;
        }
      };
      const synchronizeRoute = () => {
        const pageURL = currentPageURL();
        if (!pageURL) return null;
        if (observationState.routePageURL !== pageURL) {
          observationState.seen.clear();
          observationState.routeEventCount = 0;
          observationState.routePageURL = pageURL;
        }
        return pageURL;
      };
      const requestContext = () => ({
        observationEpoch: observationState.observationEpoch,
        pageURL: synchronizeRoute()
      });
      const dispatchHint = (rawURL, kind, hlsResponse, context) => {
        const activePageURL = synchronizeRoute();
        if (!activePageURL || context?.pageURL !== activePageURL
          || observationState.routeEventCount >= maximumRouteEvents
          || observationState.lifetimeEventCount >= maximumLifetimeEvents
          || !context?.pageURL || !context?.observationEpoch) return;
        const url = normalizedURL(rawURL);
        if (!url) return;
        if (hlsResponse !== true
          && !hlsURLPattern.test(url) && !hlsHintPattern.test(url)) {
          return;
        }
        const key = `${context.observationEpoch}\n${context.pageURL}\n${kind}\n${
          hlsResponse === true ? 1 : 0
        }\n${url}`;
        if (observationState.seen.has(key)) return;
        observationState.seen.add(key);
        observationState.routeEventCount += 1;
        observationState.lifetimeEventCount += 1;
        try {
          document.dispatchEvent(new CustomEvent(eventName, {
            detail: JSON.stringify({
              url,
              kind,
              hlsResponse: hlsResponse === true,
              pageURL: context.pageURL,
              observationEpoch: context.observationEpoch
            })
          }));
        } catch (_) {}
      };
      const dispatchResponseHint = (rawURL, kind, hlsResponse, context) => {
        let value;
        try { value = String(rawURL || ''); } catch (_) { return; }
        if (hlsResponse !== true
          && !hlsURLPattern.test(value) && !hlsHintPattern.test(value)) return;
        dispatchHint(value, kind, hlsResponse, context);
      };

      let installedHookCount = 0;
      const reflectApply = Reflect.apply;
      try {
        if (typeof window.fetch === 'function') {
          const originalFetch = window.fetch;
          const observedFetch = new Proxy(originalFetch, {
            apply(target, thisArgument, argumentsList) {
              // Invoke the page's real fetch before any observer work, and
              // return its exact Promise. Observation failures can therefore
              // never prevent, delay, or replace the request.
              const result = reflectApply(target, thisArgument, argumentsList);
              try {
                const readOnly = fetchUsesReadOnlyMethod(argumentsList);
                if (readOnly) {
                  const context = requestContext();
                  dispatchHint(
                    argumentsList[0],
                    'page-fetch-hls-request',
                    false,
                    context
                  );
                  result.then(response => {
                    try {
                      const isHLS = isHLSContentType(
                        response.headers?.get?.('content-type')
                      );
                      dispatchResponseHint(
                        response.url,
                        'page-fetch-hls-response',
                        isHLS,
                        context
                      );
                    } catch (_) {}
                  }, () => {});
                }
              } catch (_) {}
              return result;
            }
          });
          window.fetch = observedFetch;
          if (window.fetch === observedFetch) installedHookCount += 1;
        }
      } catch (_) {}

      try {
        if (typeof XMLHttpRequest === 'function') {
          const requestState = new WeakMap();
          const observedRequests = new WeakSet();
          const originalOpen = XMLHttpRequest.prototype.open;
          const observedOpen = new Proxy(originalOpen, {
            apply(target, thisArgument, argumentsList) {
              // Preserve the native return value and synchronous exception.
              const result = reflectApply(target, thisArgument, argumentsList);
              try {
                const readOnly = isReadOnlyRequest(argumentsList[0]);
                const context = readOnly ? requestContext() : null;
                requestState.set(thisArgument, {readOnly, context});
                if (readOnly) {
                  dispatchHint(
                    argumentsList[1],
                    'page-xhr-hls-request',
                    false,
                    context
                  );
                }
                if (!observedRequests.has(thisArgument)) {
                  thisArgument.addEventListener('load', () => {
                    try {
                      const state = requestState.get(thisArgument);
                      if (state?.readOnly === true) {
                        const isHLS = isHLSContentType(
                          thisArgument.getResponseHeader('content-type')
                        );
                        dispatchResponseHint(
                          thisArgument.responseURL,
                          'page-xhr-hls-response',
                          isHLS,
                          state.context
                        );
                      }
                    } catch (_) {}
                  });
                  observedRequests.add(thisArgument);
                }
              } catch (_) {}
              return result;
            }
          });
          XMLHttpRequest.prototype.open = observedOpen;
          if (XMLHttpRequest.prototype.open === observedOpen) {
            installedHookCount += 1;
          }
        }
      } catch (_) {}

      if (installedHookCount < 1) return false;
      installations.set(eventName, observationState);
      return true;
    })();
    """

  private static let activeInstrumentationScript = """
    (() => {
      'use strict';
      if (window.__miohInteractiveActiveInstalled) return;
      if (!window.__miohInteractiveScan) return;
      window.__miohInteractiveActiveInstalled = true;
      // Compatibility mode deliberately avoids modifying page APIs or DOM
      // prototypes. Dynamic DOM changes are covered by MutationObserver and
      // HLS networking is observed by the narrow page-world hook below.
      window.__miohInteractiveScan?.();
    })();
    """
}

extension IPadInteractiveMediaBrowser: WKScriptMessageHandler {
  func userContentController(
    _ userContentController: WKUserContentController,
    didReceive message: WKScriptMessage
  ) {
    guard message.name == Self.messageHandlerName,
      message.webView === webView,
      message.world === Self.instrumentationContentWorld
    else { return }
    receiveScriptMessage(message)
  }
}

extension IPadInteractiveMediaBrowser: WKNavigationDelegate {
  func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
    guard webView === self.webView, !isClosingPage else { return }
    quietTask?.cancel()
    quietTask = nil
    challengeCompatibilityTask?.cancel()
    challengeCompatibilityTask = nil
    activeNavigation = nil
    acceptingScriptCandidates = false
    isLoading = false
    webContentProcessTerminated = true
    statusMessage = "ブラウザ処理を再開しています…"
    Task { @MainActor [weak self, weak webView] in
      await Task.yield()
      guard let self, let webView, self.webView === webView,
        self.webContentProcessTerminated
      else { return }
      self.resumeAfterBackground()
    }
  }

  func webView(
    _ webView: WKWebView,
    decidePolicyFor navigationAction: WKNavigationAction,
    decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
  ) {
    guard webView === self.webView else {
      decisionHandler(.cancel)
      return
    }
    if Self.relaxedWebCompatibilityEnabled {
      if navigationAction.targetFrame?.isMainFrame == true,
        Self.isHighConfidenceAdvertisementNavigationURL(
          navigationAction.request.url
        )
      {
        decisionHandler(.cancel)
        statusMessage =
          "広告ページへの移動を停止しました。元のページで再生操作を続けてください。"
        return
      }
      // Compatibility baseline: WebKit owns main-frame, iframe, redirect,
      // challenge, and popup navigation. HLS observation is read-only and
      // media playback still revalidates every discovered URL natively.
      if navigationAction.targetFrame?.isMainFrame == true {
        if !isLoading {
          navigationRefererURL = Self.sanitizedPublicHTTPSURL(
            navigationAction.sourceFrame.request.url
          )
          prepareForMainNavigation(
            preservingSameDocumentReloadState: sameDocumentHardReloadInFlight
          )
        }
        if let destinationURL = navigationAction.request.url {
          addressText = destinationURL.absoluteString
        }
        isLoading = true
        statusMessage = "ページを読み込んでいます…"
      }
      decisionHandler(.allow)
      return
    }
    if navigationAction.targetFrame == nil {
      guard !navigationAction.shouldPerformDownload,
        openingPageWebView == nil,
        transientPopupCreationCount < Self.maximumTransientPopupCreationCount,
        Self.isAllowedTransientPopupURL(navigationAction.request.url)
      else {
        statusMessage = "安全でないポップアップを停止しました。"
        decisionHandler(.cancel)
        return
      }
      // WKUIDelegate will create a bounded child WebView. Do not treat a nil
      // target frame as a main-frame navigation: doing so clears the opener
      // before WebKit can return its WindowProxy to the click handler.
      decisionHandler(.allow)
      return
    }
    if isClosingPage, navigationAction.request.url?.scheme?.lowercased() == "about" {
      decisionHandler(.allow)
      return
    }
    if Self.isChallengeLocalFrameURL(navigationAction.request.url) {
      let sourceURL = navigationAction.sourceFrame.request.url
      let sourceIsCurrentMain =
        navigationAction.sourceFrame.isMainFrame
        && Self.sanitizedPublicHTTPSURL(sourceURL)?.absoluteString
          == committedMainDocumentURL?.absoluteString
      let sourceIsEligibleChild =
        !navigationAction.sourceFrame.isMainFrame
        && Self.sanitizedPublicHTTPSURL(sourceURL) != nil
      guard !navigationAction.shouldPerformDownload,
        navigationAction.targetFrame?.isMainFrame == false,
        (navigationAction.request.httpMethod ?? "GET").uppercased() == "GET",
        hasCurrentNativeMainChallenge,
        sourceIsCurrentMain || sourceIsEligibleChild,
        challengeLocalFrameNavigationCount
          < Self.maximumChallengeLocalFrameNavigationCount
      else {
        decisionHandler(.cancel)
        return
      }
      // These are local WebKit documents required by Turnstile. They are
      // deliberately not registered as media candidates or as network
      // navigation authority; every later HTTPS request is still checked by
      // the normal subframe authorization path below.
      challengeLocalFrameNavigationCount += 1
      decisionHandler(.allow)
      return
    }
    if navigationAction.targetFrame?.isMainFrame == false,
      !navigationAction.shouldPerformDownload,
      let safeChallengeURL = Self.sanitizedPublicHTTPSURL(
        navigationAction.request.url
      ),
      isAllowedNativeChallengeFrameURL(safeChallengeURL),
      isEligibleNativeChallengeSource(navigationAction.sourceFrame),
      ["GET", "POST"].contains(
        (navigationAction.request.httpMethod ?? "GET").uppercased()
      ),
      nativeChallengeFrameNavigationCount
        < Self.maximumNativeChallengeFrameNavigationCount
    {
      // While a native `cf-mitigated: challenge` response is current, let
      // WebKit handle Cloudflare's same-origin/challenges.cloudflare.com
      // frame tree like an ordinary browser. This path never creates media
      // candidates and ends with the main-document navigation generation.
      nativeChallengeFrameNavigationCount += 1
      decisionHandler(.allow)
      return
    }
    guard !navigationAction.shouldPerformDownload,
      let safeURL = Self.sanitizedPublicHTTPSURL(navigationAction.request.url)
    else {
      if navigationAction.targetFrame?.isMainFrame == true {
        activeNavigation = nil
        isLoading = false
        acceptingScriptCandidates = false
        authorizedRouteToken = nil
        authorizedRouteURL = nil
        authorizedMainResponseURLs.removeAll()
        clearSubframeNavigationState()
        statusMessage = "安全でないナビゲーションを停止しました。"
        updateNavigationState()
      }
      decisionHandler(.cancel)
      return
    }
    let isChallengeNavigation =
      navigationAction.targetFrame?.isMainFrame == true
      ? Self.isInteractionChallenge(safeURL)
      : isContextualChallengeNavigationURL(safeURL)
    if navigationAction.targetFrame?.isMainFrame == true {
      activeNavigation = nil
      if !isLoading {
        navigationRefererURL = Self.sanitizedPublicHTTPSURL(
          navigationAction.sourceFrame.request.url
        )
      }
      prepareForMainNavigation(
        preservingSameDocumentReloadState: sameDocumentHardReloadInFlight
      )
      addressText = safeURL.absoluteString
      isLoading = true
      statusMessage = "ページを読み込んでいます…"
      authorizedMainResponseURLs[safeURL.absoluteString] = navigationGeneration
    } else {
      guard let targetFrame = navigationAction.targetFrame else {
        decisionHandler(.cancel)
        return
      }
      authorizeSubframeNavigationAction(
        to: safeURL,
        sourceFrame: navigationAction.sourceFrame,
        targetFrame: targetFrame,
        navigationTypeRawValue: navigationAction.navigationType.rawValue,
        isChallengeNavigation: isChallengeNavigation,
        allowsInitialChildChallengeFallback: (navigationAction.request.httpMethod ?? "GET")
          .uppercased() == "GET",
        decisionHandler: decisionHandler
      )
      return
    }
    if isChallengeNavigation {
      challengeActive = true
      acceptingScriptCandidates = false
      quietTask?.cancel()
      quietTask = nil
      setChallengeWaitingStatus()
    }
    decisionHandler(.allow)
  }

  func webView(
    _ webView: WKWebView,
    decidePolicyFor navigationResponse: WKNavigationResponse,
    decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
  ) {
    guard webView === self.webView else {
      decisionHandler(.cancel)
      return
    }
    if isClosingPage {
      decisionHandler(.allow)
      return
    }
    if Self.relaxedWebCompatibilityEnabled {
      let response = navigationResponse.response as? HTTPURLResponse
      let responseURL = Self.sanitizedPublicHTTPSURL(
        navigationResponse.response.url
      )
      if navigationResponse.isForMainFrame {
        committedMainDocumentURL = responseURL
        pendingSuccessfulMainResponseURL = Self.successfulVisibleMainResponseURL(
          navigationResponse,
          response: response
        )
        // In the compatibility baseline a challenge document is an ordinary
        // WebKit page. Do not leave the native inspection gate latched if the
        // challenge swaps in the real player within the same document.
        mainFrameChallengeResponse = false
        mainFrameChallengeResponseGeneration = nil
      }
      if let responseURL,
        Self.isDirectMediaResponse(responseURL, response: response)
      {
        insertCandidate(
          Candidate(
            url: responseURL,
            sourceKind: "navigation-response",
            frameDepth: navigationResponse.isForMainFrame ? 0 : 1,
            discoveryOrder: nextDiscoveryOrder,
            frameURL: Self.sanitizedPublicHTTPSURL(webView.url),
            provenance: .script
          )
        )
      }
      decisionHandler(.allow)
      return
    }
    if !navigationResponse.isForMainFrame,
      Self.isChallengeLocalFrameURL(navigationResponse.response.url)
    {
      decisionHandler(hasCurrentNativeMainChallenge ? .allow : .cancel)
      return
    }
    let response = navigationResponse.response as? HTTPURLResponse
    let disposition = response?.value(forHTTPHeaderField: "Content-Disposition")?.lowercased()
    let safeResponseURL = Self.sanitizedPublicHTTPSURL(
      navigationResponse.response.url
    )
    let hasChallengeResponseHeader =
      response?.value(forHTTPHeaderField: "cf-mitigated")?.lowercased()
      == "challenge"
    let isChallenge: Bool
    if navigationResponse.isForMainFrame {
      isChallenge =
        hasChallengeResponseHeader
        || safeResponseURL.map(Self.isInteractionChallenge) == true
    } else {
      isChallenge =
        safeResponseURL.map {
          (hasChallengeResponseHeader && isCurrentMainOrCloudflareChallengeURL($0))
            || isContextualChallengeNavigationURL($0)
        } ?? false
    }
    if !navigationResponse.isForMainFrame,
      let safeResponseURL,
      isAllowedNativeChallengeFrameURL(safeResponseURL)
    {
      guard navigationResponse.canShowMIMEType,
        disposition?.contains("attachment") != true
      else {
        decisionHandler(.cancel)
        return
      }
      decisionHandler(.allow)
      return
    }
    var subframeAuthorization: SubframeNavigationAuthorization?
    if !navigationResponse.isForMainFrame, let safeResponseURL {
      pruneExpiredInitialUserAuthorizations()
      subframeAuthorization = consumeSubframeAuthorization(for: safeResponseURL)
    }
    let responseAuthorizationValid: Bool
    if navigationResponse.isForMainFrame {
      responseAuthorizationValid =
        safeResponseURL.map {
          authorizedMainResponseURLs[$0.absoluteString] == navigationGeneration
            && isLoading
        } ?? false
    } else {
      responseAuthorizationValid = subframeAuthorization != nil
    }
    guard responseAuthorizationValid else {
      if navigationResponse.isForMainFrame {
        isLoading = false
        acceptingScriptCandidates = false
        statusMessage = "追跡できない応答を安全のため停止しました。"
        updateNavigationState()
      }
      decisionHandler(.cancel)
      return
    }
    if var authorization = subframeAuthorization {
      guard authorizationInitiatorIsCurrent(authorization) else {
        revokeSubframeAuthorization(authorization)
        decisionHandler(.cancel)
        return
      }
      if isChallenge {
        guard
          let promotedAuthorization = promoteSubframeAuthorizationToChallenge(
            authorization
          )
        else {
          revokeSubframeAuthorization(authorization)
          decisionHandler(.cancel)
          return
        }
        authorization = promotedAuthorization
      } else if case .challenge = authorization.chain.authority {
        authorization.retiresSubframeChallengeOnCommit = true
      }
      subframeAuthorization = authorization
    }
    if isChallenge {
      challengeActive = true
      acceptingScriptCandidates = false
      quietTask?.cancel()
      quietTask = nil
      setChallengeWaitingStatus()
    }
    if navigationResponse.isForMainFrame, responseAuthorizationValid {
      committedMainDocumentURL = safeResponseURL
      mainFrameChallengeResponse = isChallenge
      mainFrameChallengeResponseGeneration = isChallenge ? navigationGeneration : nil
      if isChallenge {
        beginNativeChallengeCompatibilityWindow()
      } else {
        challengeCompatibilityTask?.cancel()
        challengeCompatibilityTask = nil
        challengeCompatibilityTimedOut = false
      }
    }
    let isDirectMediaResponse =
      safeResponseURL.map {
        Self.isDirectMediaResponse($0, response: response)
      } ?? false
    if let safeResponseURL, isDirectMediaResponse, !isChallenge,
      !challengeActive,
      responseAuthorizationValid
    {
      insertCandidate(
        Candidate(
          url: safeResponseURL,
          sourceKind: "navigation-response",
          frameDepth: navigationResponse.isForMainFrame ? 0 : 1,
          discoveryOrder: nextDiscoveryOrder,
          frameURL: navigationResponse.isForMainFrame
            ? navigationRefererURL : Self.sanitizedPublicHTTPSURL(webView.url),
          provenance: navigationResponse.isForMainFrame
            ? .mainNavigationResponse : .subframeNavigationResponse
        )
      )
    }
    guard navigationResponse.canShowMIMEType,
      disposition?.contains("attachment") != true,
      safeResponseURL != nil
    else {
      if let subframeAuthorization {
        revokeSubframeAuthorization(subframeAuthorization)
      }
      if navigationResponse.isForMainFrame, responseAuthorizationValid {
        isLoading = false
        statusMessage =
          isDirectMediaResponse
          ? "再生可能性を確認した配信URLがあります。「配信を解析」を押してください。"
          : "表示できない応答またはダウンロードを停止しました。"
        updateNavigationState()
      }
      decisionHandler(.cancel)
      return
    }
    if navigationResponse.isForMainFrame, responseAuthorizationValid, !isChallenge {
      pendingSuccessfulMainResponseURL = Self.successfulVisibleMainResponseURL(
        navigationResponse,
        response: response
      )
    }
    if let subframeAuthorization {
      guard completedSubframeAuthorizations.count < Self.maximumKnownFrameCount else {
        revokeSubframeAuthorization(subframeAuthorization)
        decisionHandler(.cancel)
        return
      }
      completedSubframeAuthorizations.append(subframeAuthorization)
    }
    decisionHandler(.allow)
  }

  func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
    guard webView === self.webView, !isClosingPage else { return }
    activeNavigation = navigation
    quietTask?.cancel()
    quietTask = nil
    inspectionRequested = false
    mainFrameChallengeResponse = false
    mainFrameChallengeResponseGeneration = nil
    knownFrames.removeAll()
    mainDocumentToken = nil
    committedMainDocumentURL = nil
    pendingSuccessfulMainResponseURL = nil
    authorizedRouteToken = nil
    authorizedRouteURL = nil
    clearSubframeNavigationState()
    candidates.removeAll()
    candidateRevision = 0
    clearMediaSourceState()
    nextDiscoveryOrder = 0
    candidateCount = 0
    readyCandidateGeneration = 0
    acceptingScriptCandidates = false
    isLoading = true
    statusMessage = "ページを読み込んでいます…"
    updateNavigationState()
  }

  func webView(
    _ webView: WKWebView,
    didReceiveServerRedirectForProvisionalNavigation navigation: WKNavigation!
  ) {
    guard webView === self.webView, !isClosingPage,
      navigation === activeNavigation
    else { return }
    if Self.relaxedWebCompatibilityEnabled { return }
    guard mainNavigationRedirectCount < Self.maximumMainNavigationRedirectCount,
      let redirectURL = Self.sanitizedPublicHTTPSURL(webView.url)
    else {
      webView.stopLoading()
      isLoading = false
      statusMessage = "安全に追跡できるリダイレクト上限を超えました。"
      return
    }
    mainNavigationRedirectCount += 1
    authorizedMainResponseURLs[redirectURL.absoluteString] = navigationGeneration
  }

  func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
    guard webView === self.webView, !isClosingPage,
      navigation === activeNavigation
    else { return }
    if Self.relaxedWebCompatibilityEnabled, let url = webView.url {
      addressText = url.absoluteString
      committedMainDocumentURL = Self.sanitizedPublicHTTPSURL(url)
    } else if let url = Self.sanitizedPublicHTTPSURL(webView.url) {
      addressText = url.absoluteString
      committedMainDocumentURL = url
    }
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    guard webView === self.webView, !isClosingPage,
      navigation === activeNavigation
    else { return }
    isLoading = webView.isLoading
    pageTitle = webView.title ?? ""
    if Self.relaxedWebCompatibilityEnabled, let url = webView.url {
      addressText = url.absoluteString
    } else if let url = Self.sanitizedPublicHTTPSURL(webView.url) {
      addressText = url.absoluteString
    }
    lastPageLocationToken = webView.url?.absoluteString
    updateNavigationState()
    guard !webView.isLoading else { return }

    let finishedInChallenge =
      !Self.relaxedWebCompatibilityEnabled
      && (mainFrameChallengeResponse
        || challengeActive
        || Self.isInteractionChallenge(webView.url ?? URL(fileURLWithPath: "/")))
    sameDocumentHardReloadInFlight = false

    if finishedInChallenge {
      challengeActive = true
      acceptingScriptCandidates = false
      setChallengeWaitingStatus()
      return
    }
    if pendingSameDocumentPageURL == webView.url?.absoluteString {
      pendingSameDocumentPageURL = nil
      acceptSameDocumentRouteInPlace()
      return
    }
    pendingSameDocumentPageURL = nil
    challengeActive = false
    publishSuccessfulPageVisitIfReady()
    if let currentURL = Self.sanitizedPublicHTTPSURL(webView.url),
      !Self.isInteractionChallenge(currentURL), Self.isDirectMediaCandidate(currentURL)
    {
      insertCandidate(
        Candidate(
          url: currentURL,
          sourceKind: "page",
          frameDepth: 0,
          discoveryOrder: nextDiscoveryOrder,
          frameURL: currentURL,
          provenance: .currentPage
        )
      )
    }
    inspectionRequested = true
    beginInspectionForCurrentRoute()
    statusMessage =
      candidates.isEmpty
      ? "ページ内プレイヤーを解析しています…" : "配信候補を確認しています…"
    if candidates.values.contains(where: Self.isReadyCandidate) {
      scheduleReadyCandidate()
    }
  }

  func webView(
    _ webView: WKWebView,
    didFailProvisionalNavigation navigation: WKNavigation!,
    withError error: Error
  ) {
    guard webView === self.webView, !isClosingPage,
      navigation === activeNavigation,
      !Self.isCancelledNavigationError(error)
    else { return }
    isLoading = false
    challengeActive = false
    mainFrameChallengeResponse = false
    mainFrameChallengeResponseGeneration = nil
    pendingSuccessfulMainResponseURL = nil
    activeNavigation = nil
    inspectionRequested = false
    acceptingScriptCandidates = false
    authorizedRouteToken = nil
    authorizedRouteURL = nil
    authorizedMainResponseURLs.removeAll()
    clearSubframeNavigationState()
    initialChildChallengeFallbackGeneration = nil
    resetSameDocumentNavigationState()
    statusMessage = "ページを読み込めませんでした: \(error.localizedDescription)"
    updateNavigationState()
  }

  func webView(
    _ webView: WKWebView,
    didFail navigation: WKNavigation!,
    withError error: Error
  ) {
    guard webView === self.webView, !isClosingPage,
      navigation === activeNavigation,
      !Self.isCancelledNavigationError(error)
    else { return }
    isLoading = false
    challengeActive = false
    mainFrameChallengeResponse = false
    mainFrameChallengeResponseGeneration = nil
    pendingSuccessfulMainResponseURL = nil
    activeNavigation = nil
    inspectionRequested = false
    acceptingScriptCandidates = false
    authorizedRouteToken = nil
    authorizedRouteURL = nil
    authorizedMainResponseURLs.removeAll()
    clearSubframeNavigationState()
    initialChildChallengeFallbackGeneration = nil
    resetSameDocumentNavigationState()
    statusMessage = "ページの読み込みを完了できませんでした: \(error.localizedDescription)"
    updateNavigationState()
  }

  func webView(
    _ webView: WKWebView,
    didReceive challenge: URLAuthenticationChallenge,
    completionHandler:
      @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
  ) {
    guard webView === self.webView else {
      completionHandler(.cancelAuthenticationChallenge, nil)
      return
    }
    if Self.relaxedWebCompatibilityEnabled {
      completionHandler(.performDefaultHandling, nil)
      return
    }
    if challenge.protectionSpace.authenticationMethod
      == NSURLAuthenticationMethodServerTrust
    {
      completionHandler(.performDefaultHandling, nil)
    } else {
      completionHandler(.cancelAuthenticationChallenge, nil)
    }
  }
}

extension IPadInteractiveMediaBrowser: WKUIDelegate {
  func webView(
    _ webView: WKWebView,
    createWebViewWith configuration: WKWebViewConfiguration,
    for navigationAction: WKNavigationAction,
    windowFeatures: WKWindowFeatures
  ) -> WKWebView? {
    if Self.relaxedWebCompatibilityEnabled {
      guard webView === self.webView,
        navigationAction.targetFrame == nil,
        transientPopupCreationCount
          < Self.maximumRelaxedTransientPopupCreationCount
      else {
        statusMessage = "このページで開ける別ウインドウの上限に達しました。"
        return nil
      }
      configuration.preferences.javaScriptCanOpenWindowsAutomatically = true
      configuration.mediaTypesRequiringUserActionForPlayback = []
      let coordinator = IPadTransientPopupCoordinator(
        maximumNavigationCount: Self.relaxedTransientPopupNavigationCount,
        relaxedWebCompatibility: true,
        allowsNavigation: { [weak self] url in
          guard !Self.isHighConfidenceAdvertisementNavigationURL(url) else {
            self?.statusMessage =
              "広告ページへの移動を停止しました。元のページで再生操作を続けてください。"
            return false
          }
          return true
        },
        readyHandler: { [weak self] popupWebView in
          self?.markTransientPopupReady(popupWebView)
        },
        closeHandler: { [weak self] popupWebView in
          self?.retireTransientPopup(popupWebView)
        }
      )
      let popupFrame =
        webView.bounds.isEmpty
        ? CGRect(x: 0, y: 0, width: 1_024, height: 768)
        : webView.bounds
      let popupWebView = WKWebView(
        frame: popupFrame,
        configuration: configuration
      )
      transientPopupCreationCount += 1
      popupWebView.navigationDelegate = coordinator
      popupWebView.uiDelegate = coordinator
      retainTransientPopup(
        popupWebView,
        coordinator: coordinator,
        expires: false
      )
      statusMessage = "開いたページを読み込んでいます…"
      return popupWebView
    }
    guard webView === self.webView,
      openingPageWebView == nil,
      navigationAction.targetFrame == nil,
      !navigationAction.shouldPerformDownload,
      transientPopupCreationCount < Self.maximumTransientPopupCreationCount,
      Self.isAllowedTransientPopupURL(navigationAction.request.url)
    else {
      statusMessage = "安全でないポップアップを停止しました。"
      return nil
    }

    configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
    configuration.allowsAirPlayForMediaPlayback = false
    #if os(iOS)
      configuration.allowsInlineMediaPlayback = true
      configuration.allowsPictureInPictureMediaPlayback = false
    #endif
    configuration.mediaTypesRequiringUserActionForPlayback = .all

    let coordinator = IPadTransientPopupCoordinator(
      maximumNavigationCount: Self.maximumTransientPopupNavigationCount,
      allowsNavigation: { Self.isAllowedTransientPopupURL($0) },
      readyHandler: { [weak self] popupWebView in
        self?.markTransientPopupReady(popupWebView)
      },
      closeHandler: { [weak self] popupWebView in
        self?.retireTransientPopup(popupWebView)
      }
    )
    transientPopupCreationCount += 1
    let popupFrame =
      webView.bounds.isEmpty
      ? CGRect(x: 0, y: 0, width: 1_024, height: 768)
      : webView.bounds
    let popupWebView = WKWebView(frame: popupFrame, configuration: configuration)
    popupWebView.navigationDelegate = coordinator
    popupWebView.uiDelegate = coordinator
    retainTransientPopup(popupWebView, coordinator: coordinator)
    statusMessage =
      "1段目のリンクを確認しました。ページ内の次の再生ボタンを待っています…"
    return popupWebView
  }
}

#if os(iOS)
@MainActor
struct IPadInteractiveBrowserWebView: UIViewRepresentable {
  @ObservedObject var browser: IPadInteractiveMediaBrowser

  func makeUIView(context: Context) -> WKWebView {
    browser.webView
  }

  func updateUIView(_ uiView: WKWebView, context: Context) {}
}
#elseif os(macOS)
@MainActor
struct IPadInteractiveBrowserWebView: NSViewRepresentable {
  @ObservedObject var browser: IPadInteractiveMediaBrowser

  func makeNSView(context: Context) -> WKWebView {
    browser.webView
  }

  func updateNSView(_ nsView: WKWebView, context: Context) {}
}
#endif
