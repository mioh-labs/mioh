import json
import http.server
from pathlib import Path
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "MiohRemote" / "MiohRemote"
BROWSER = APP / "IPadInteractiveMediaBrowser.swift"
RESOLVER = APP / "IPadMediaURLResolver.swift"
LIBRARY = APP / "IPadBrowserLibraryStore.swift"
VIEW = APP / "IPadStandaloneView.swift"
STORE = APP / "IPadStandaloneStore.swift"
PROJECT = ROOT / "apps" / "MiohRemote" / "MiohRemote.xcodeproj" / "project.pbxproj"
CONTENT_WORLD_HARNESS = ROOT / "tests" / "swift" / "WKContentWorldBridgeHarness.swift"
BROWSER_LIBRARY_HARNESS = ROOT / "tests" / "swift" / "IPadBrowserLibraryStoreHarness.swift"
RESOURCE_LOADER_HARNESS = (
    ROOT / "tests" / "swift" / "IPadBrowserHLSResourceLoaderHarness.swift"
)
WEBKIT_DOWNLOAD_HARNESS = (
    ROOT / "tests" / "swift" / "IPadWebKitDownloadHarness.swift"
)


class IPadInteractiveBrowserContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.browser = BROWSER.read_text(encoding="utf-8") if BROWSER.exists() else ""
        cls.library = LIBRARY.read_text(encoding="utf-8") if LIBRARY.exists() else ""
        cls.view = VIEW.read_text(encoding="utf-8")
        cls.store = STORE.read_text(encoding="utf-8")
        cls.project = PROJECT.read_text(encoding="utf-8")
        cls.navigation_delegate = cls.browser.split(
            "extension IPadInteractiveMediaBrowser: WKNavigationDelegate", 1
        )[1].split("extension IPadInteractiveMediaBrowser: WKUIDelegate", 1)[0]

    def assert_contracts(self, source, contracts):
        for contract in contracts:
            self.assertIn(contract, source, f"missing source contract: {contract}")

    def page_network_observer_runtime_source(self):
        source = self.browser.split(
            'private static let pageNetworkObservationScript = """', 1
        )[1].split('\n    """', 1)[0]
        source = "\n".join(
            line[4:] if line.startswith("    ") else line
            for line in source.splitlines()
        )
        return source.replace(
            r"\(maximumCandidateURLLength)", "8192"
        ).replace(r"\\", "\\")

    def hls_relay_runtime_source(self):
        return self.browser.split(
            'private static let hlsResourceRelayScript = """', 1
        )[1].split('\n    """', 1)[0]

    def test_browser_tab_and_controller_are_in_the_ios_target(self):
        self.assertTrue(BROWSER.exists())
        self.assert_contracts(self.view, [
            "case browser",
            'case .browser: "ブラウザ"',
            'case .browser: "globe"',
            "case .browser:",
            "browserView",
            "IPadInteractiveBrowserWebView(browser: interactiveBrowser)",
        ])
        self.assertIn("IPadInteractiveMediaBrowser.swift in Sources", self.project)

    def test_browser_library_store_is_persistent_bounded_and_in_the_ios_target(self):
        self.assertTrue(LIBRARY.exists())
        self.assertIn("IPadBrowserLibraryStore.swift in Sources", self.project)
        self.assert_contracts(self.library, [
            "final class IPadBrowserLibraryStore: ObservableObject",
            "struct Entry: Codable, Equatable, Identifiable",
            "let url: String",
            "let title: String",
            "let date: Date",
            "@Published private(set) var history: [Entry]",
            "@Published private(set) var bookmarks: [Entry]",
            '"mioh.ipad.browser.history.v1"',
            '"mioh.ipad.browser.bookmarks.v1"',
            "maximumStoredDataSize",
            "maximumURLLength",
            "maximumTitleLength",
            "maximumHistoryCount: Int = 200",
            "maximumBookmarkCount: Int =",
            "func recordVisit(url: URL, title: String, date: Date = Date())",
            "func isBookmarked(_ url: URL?) -> Bool",
            "func toggleBookmark(url: URL, title: String, date: Date = Date()) -> Bool",
            "func removeHistory(id: Entry.ID)",
            "func removeBookmark(id: Entry.ID)",
            "func clearHistory()",
            "func clearBookmarks()",
            "JSONEncoder().encode(entries)",
            "JSONDecoder().decode([Entry].self, from: data)",
        ])

        record_visit = self.library.split("func recordVisit", 1)[1].split(
            "func isBookmarked", 1
        )[0]
        self.assert_contracts(record_visit, [
            "sanitizedPublicHTTPSURL(url)",
            "history.removeAll { $0.url == entry.url }",
            "history.insert(entry, at: 0)",
            "history.count > maximumHistoryCount",
            "persist(history",
        ])
        toggle = self.library.split("func toggleBookmark", 1)[1].split(
            "func removeHistory", 1
        )[0]
        self.assert_contracts(toggle, [
            "sanitizedPublicHTTPSURL(url)",
            "bookmarks.firstIndex",
            "bookmarks.remove(at: index)",
            "bookmarks.insert(",
            "bookmarks.count > maximumBookmarkCount",
            "persist(bookmarks",
        ])

    def test_browser_library_sanitizes_loaded_and_new_entries(self):
        self.assert_contracts(self.library, [
            "data.count <= maximumStoredDataSize",
            "entries.count < maximumCount",
            "var seen = Set<String>()",
            "seen.insert(safeURL.absoluteString).inserted",
            "sanitizedTitle(title, fallbackURL: url)",
            "components.scheme?.lowercased() == \"https\"",
            "components.user == nil, components.password == nil",
            "components.fragment = nil",
            "if components.port == 443 { components.port = nil }",
            "isSensitiveQueryName",
            'name.hasPrefix("x-amz-") || name.hasPrefix("x-goog-")',
            '"access_token", "auth", "authorization", "code", "credential"',
            '"sig", "signature", "token"',
            '!host.hasSuffix(".local")',
            '!host.hasPrefix("::ffff:")',
        ])
        self.assertNotIn("URLSession", self.library)
        self.assertNotIn("WKWebsiteDataStore", self.library)

    def test_successful_visible_pages_are_the_only_history_publication_source(self):
        self.assert_contracts(self.browser, [
            "struct SuccessfulPageVisit: Equatable",
            "@Published private(set) var successfulPageVisit: SuccessfulPageVisit?",
            "private var pendingSuccessfulMainResponseURL: URL?",
            "private static func successfulVisibleMainResponseURL(",
            "private func noteSuccessfulPublicPageVisit(url: URL? = nil)",
            "private func publishSuccessfulPageVisitIfReady()",
        ])

        successful_response = self.browser.split(
            "private static func successfulVisibleMainResponseURL", 1
        )[1].split("private static func normalizedPublicHTTPSURL", 1)[0]
        self.assert_contracts(successful_response, [
            "navigationResponse.isForMainFrame",
            "navigationResponse.canShowMIMEType",
            "(200..<400).contains(response.statusCode)",
            'contains("attachment") != true',
            'value(forHTTPHeaderField: "cf-mitigated")',
            '!= "challenge"',
            "sanitizedPublicHTTPSURL(navigationResponse.response.url)",
            "!isInteractionChallenge(safeURL)",
        ])

        publication = self.browser.split(
            "private func publishSuccessfulPageVisitIfReady()", 1
        )[1].split("private static func successfulVisibleMainResponseURL", 1)[0]
        self.assert_contracts(publication, [
            "defer { pendingSuccessfulMainResponseURL = nil }",
            "Self.sanitizedPublicHTTPSURL(webView.url)?.absoluteString",
            "== expectedURL.absoluteString",
            "!isClosingPage, !challengeActive",
            "!Self.isInteractionChallenge(safeURL)",
            "successfulPageVisit = SuccessfulPageVisit(",
        ])
        self.assertIn("publishSuccessfulPageVisitIfReady()", self.navigation_delegate)
        self.assertGreaterEqual(
            self.navigation_delegate.count("pendingSuccessfulMainResponseURL = nil"),
            2,
        )

    def test_browser_library_ui_can_add_open_remove_and_clear_entries(self):
        self.assert_contracts(self.view, [
            "private enum BrowserLibrarySection",
            "private struct BrowserLibrarySheet: View",
            "@StateObject private var browserLibrary = IPadBrowserLibraryStore()",
            "@State private var showingBrowserLibrary: BrowserLibrarySection?",
            ".sheet(item: $showingBrowserLibrary)",
            ".onChange(of: interactiveBrowser.successfulPageVisit)",
            "browserLibrary.recordVisit(url: visit.url, title: visit.title)",
            "private func toggleCurrentBrowserBookmark()",
            "browserLibrary.toggleBookmark(",
            "private func openBrowserLibraryEntry(",
            "showingBrowserLibrary = nil",
            "openBrowserAddress()",
            "store.removeHistory(id: entry.id)",
            "store.removeBookmark(id: entry.id)",
            "store.clearHistory()",
            "store.clearBookmarks()",
            'Label("履歴", systemImage: "clock")',
            'Label("ブックマーク", systemImage: "star")',
            '.accessibilityLabel("履歴とブックマーク")',
            '"ブックマークを解除" : "ブックマークに追加"',
            'Button("すべて削除", role: .destructive)',
            'Label("削除", systemImage: "trash")',
        ])
        open_entry = self.view.split(
            "private func openBrowserLibraryEntry", 1
        )[1].split("private func openBrowserAddress", 1)[0]
        self.assertNotIn("URLSession", open_entry)
        self.assertNotIn("webView.load", open_entry)

    def test_browser_library_store_runtime_behavior(self):
        xcrun = shutil.which("xcrun")
        if sys.platform != "darwin" or xcrun is None:
            self.skipTest("browser library store probe requires macOS and Xcode")
        with tempfile.TemporaryDirectory(prefix="mioh-browser-library-") as directory:
            executable = Path(directory) / "browser-library-probe"
            subprocess.run(
                [
                    xcrun,
                    "swiftc",
                    "-parse-as-library",
                    "-framework",
                    "Combine",
                    str(LIBRARY),
                    str(BROWSER_LIBRARY_HARNESS),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [str(executable)],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertIn("iPad browser library store probe passed", completed.stdout)

    def test_one_visible_webview_keeps_a_standard_persistent_session(self):
        self.assert_contracts(self.browser, [
            "final class IPadInteractiveMediaBrowser",
            "private(set) var webView: WKWebView",
            "WKWebsiteDataStore.default()",
            "webViewGeneration",
            "Self.makeWebView(",
            "struct IPadInteractiveBrowserWebView: UIViewRepresentable",
            "browser.webView",
            "func closePage()",
        ])
        make_body = self.browser.split("func makeUIView", 1)[1].split(
            "func updateUIView", 1
        )[0]
        self.assertNotIn("WKWebView(", make_body)
        self.assertEqual(
            self.view.count(
                "IPadInteractiveBrowserWebView(browser: interactiveBrowser)"
            ),
            1,
        )

    def test_cloudflare_requires_visible_user_confirmation(self):
        self.assert_contracts(self.browser, [
            'value(forHTTPHeaderField: "cf-mitigated")',
            '== "challenge"',
            'host == "challenges.cloudflare.com"',
            'contains("/cdn-cgi/challenge-platform/")',
            "challengeActive = true",
            "Cloudflareの確認を待っています。チェックボックスは必要な場合だけ表示されます。",
            "mainFrameChallengeResponse",
            "challengeActive, !mainFrameChallengeResponse",
            ".cf-turnstile",
            "iframe[src*=\"challenges.cloudflare.com\"]",
        ])
        self.assertNotIn("cf_clearance=", self.browser)

    def test_challenge_page_gets_only_isolated_passive_observation(self):
        self.assert_contracts(self.browser, [
            "passiveInstrumentationScript",
            "activeInstrumentationScript",
            "pageNetworkObservationScript",
            "guard !challengeActive",
            "callAsyncJavaScript(",
            "PerformanceObserver",
            "MutationObserver",
            "private static let instrumentationContentWorld",
            "contentWorld: instrumentationContentWorld",
            "in: instrumentationContentWorld",
            "in: Self.instrumentationContentWorld",
            "message.webView === webView",
            "message.world === Self.instrumentationContentWorld",
            "self.acceptingScriptCandidates,",
            "!self.challengeActive,",
            "in: WKContentWorld.page",
        ])
        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        self.assertNotIn("Element.prototype.setAttribute =", passive)
        self.assertNotIn("window.fetch =", passive)
        self.assertNotIn("XMLHttpRequest.prototype.open =", passive)

        active = self.browser.split(
            "private static let activeInstrumentationScript", 1
        )[1].split("extension IPadInteractiveMediaBrowser: WKScriptMessageHandler", 1)[0]
        self.assertIn("window.__miohInteractiveScheduleScan?.();", active)
        self.assertNotIn("Element.prototype.setAttribute =", active)
        self.assertNotIn("window.fetch =", active)
        self.assertNotIn("XMLHttpRequest.prototype.open =", active)
        self.assertNotIn("window.open =", active)

    def test_page_world_hls_observer_is_bounded_and_unverified(self):
        page_observer = self.browser.split(
            "private static let pageNetworkObservationScript", 1
        )[1].split("private static let activeInstrumentationScript", 1)[0]
        self.assert_contracts(page_observer, [
            "const maximumRouteEvents = 128;",
            "const maximumLifetimeEvents = 512;",
            "const maximumURLLength = \\(maximumCandidateURLLength);",
            "method === 'GET' || method === 'HEAD'",
            "const observedFetch = new Proxy(originalFetch",
            "const observedOpen = new Proxy(originalOpen",
            "const reflectApply = Reflect.apply;",
            "const result = reflectApply(target, thisArgument, argumentsList);",
            "const fetchUsesReadOnlyMethod = argumentsList =>",
            "dispatchHint(",
            "'page-fetch-hls-request'",
            "'page-xhr-hls-request'",
            "if (typeof rawMethod !== 'string') return false;",
            "return result;",
            "response.headers?.get?.('content-type')",
            "thisArgument.getResponseHeader('content-type')",
            "application/vnd.apple.mpegurl",
            "application/x-mpegurl",
            "document.dispatchEvent(new CustomEvent(eventName",
            "let installedHookCount = 0;",
            "if (installedHookCount < 1) return false;",
            "const observedRequests = new WeakSet();",
            "routePageURL: null",
            "const synchronizeRoute = () =>",
            "context?.pageURL !== activePageURL",
            "observationEpoch: context.observationEpoch",
            "pageURL: context.pageURL",
        ])
        self.assertNotIn("response.text()", page_observer)
        self.assertNotIn("response.arrayBuffer()", page_observer)
        self.assertNotIn("response.clone()", page_observer)
        fetch_apply = page_observer.split(
            "const observedFetch = new Proxy(originalFetch", 1
        )[1].split("window.fetch = observedFetch", 1)[0]
        self.assertLess(
            fetch_apply.index(
                "const result = reflectApply(target, thisArgument, argumentsList);"
            ),
            fetch_apply.index("const readOnly = fetchUsesReadOnlyMethod(argumentsList);"),
        )

        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        self.assert_contracts(passive, [
            "maximumPageNetworkBridgeEvents = 128",
            "maximumPageNetworkBridgeLifetimeEvents = 512",
            "maximumPendingPageNetworkBridgeEvents = 32",
            "allowedPageNetworkKinds",
            "pageNetworkBridgeEventCount >= maximumPageNetworkBridgeEvents",
            "pendingPageNetworkBridgePayloads.push(normalizedPayload)",
            "const routeChanged = authorizedRouteToken !== token;",
            "if (routeChanged)",
            "pageNetworkBridgeSeen.has(key)",
            "pageNetworkBridgeEventCount += 1",
            "page-fetch-hls-response",
            "page-xhr-hls-response",
            "native public-network resolver",
            "const installPageNetworkBridge = (rawEventName, rawObservationEpoch)",
            "epochs.previous = epochs.current",
            "payloadEpoch !== acceptedEpochs?.current",
            "payloadEpoch !== acceptedEpochs?.previous",
            "payload.pageURL !== currentPageURL",
            "window.__miohInteractiveNotifyPageChange?.();",
        ])
        self.assert_contracts(self.browser, [
            '"observationEpoch": observationEpoch',
            "bridgeEventName,",
            "observationEpoch",
        ])
        ready = self.browser.split(
            "private static func isReadyCandidate", 1
        )[1].split("private static func isUnverifiedMediaResponseHint", 1)[0]
        self.assertNotIn("page-fetch-hls-response", ready)
        self.assertNotIn("page-xhr-hls-response", ready)

    def test_passive_dom_observation_is_bounded_and_debounced(self):
        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        self.assert_contracts(passive, [
            "const mediaElementSelector = 'video,source,iframe';",
            "const maximumElementsPerScan = 256;",
            "const maximumScriptElementsPerScan = 64;",
            "const maximumScriptTextCharactersPerScan = 262144;",
            "const maximumTextMediaMatchesPerScan = 64;",
            "const maximumResourceEntriesPerBatch = 256;",
            'const challengeObservationEnabled = \\(relaxedWebCompatibilityEnabled ? "false" : "true");',
            'const heartbeatIntervalMilliseconds = \\(relaxedWebCompatibilityEnabled ? "2000" : "1000");',
            "if (!challengeObservationEnabled || window !== window.top) return;",
            "const recentResourceEntries = [];",
            "const inspectTextForMediaURLs =",
            "document.scripts || []",
            "'script-text'",
            "const inspectResourceEntries = entries =>",
            "entryCount - maximumResourceEntriesPerBatch",
            "recentResourceEntries.length - maximumResourceEntriesPerBatch",
            "const replayRecentResourceEntries = () =>",
            "const inspectCurrentPerformanceResources = () =>",
            "performance.getEntriesByType('resource')",
            "inspectResourceEntries(list.getEntries());",
            "const retireDisconnectedMediaSlots = () =>",
            "let scanScheduled = false;",
            "const scheduleScan = () =>",
            "requestIdleCallback(run, {timeout: 500});",
            "let inspectedMutationCount = 0;",
            "let shouldScan = false;",
            "node.matches?.(mediaElementSelector)",
            "attributeFilter: sourceAttributes",
            "if (sourceAttributes.includes(mutation.attributeName))",
            "reportChallengeState();",
            "retireDisconnectedMediaSlots();",
        ])
        scan = passive.split("const scan = () =>", 1)[1].split(
            "let scanScheduled = false", 1
        )[0]
        self.assertNotIn("'a[href]'", scan)
        self.assertNotIn("Array.from(entries", passive)
        interval = passive.split("setInterval(() =>", 1)[1].split(
            "}, heartbeatIntervalMilliseconds);", 1
        )[0]
        self.assertNotIn("scan();", interval)
        self.assertNotIn("inspectCurrentPerformanceResources();", interval)
        mutations = passive.split(
            "const observer = new MutationObserver(mutations =>", 1
        )[1].split("observer.observe(document", 1)[0]
        self.assertIn("if (shouldScan) scheduleScan();", mutations)
        self.assertEqual(mutations.count("scheduleScan();"), 1)
        self.assertNotIn("suppressHighConfidenceAdvertisementOverlays(", mutations)
        self.assertNotIn("querySelectorAll", mutations)
        self.assertNotIn("'video,source,iframe,a[href]'", mutations)

    def test_idle_inspection_settles_to_passive_monitoring_without_stopping_observers(self):
        self.assert_contracts(self.browser, [
            "inspectionStatusSettleNanoseconds",
            "private var inspectionStatusTask: Task<Void, Never>?",
            "private func scheduleInspectionStatusSettlement()",
            "generation == self.navigationGeneration",
            "routeToken == self.authorizedRouteToken",
            'self.statusMessage?.contains("解析しています") == true',
            "HLS通信を待っています。ページ内の動画を再生してください。",
            "配信候補を監視しています。ページ内の動画を再生してください。",
            "activateInspectionInKnownFrames()",
            "scheduleInspectionStatusSettlement()",
        ])
        settlement = self.browser.split(
            "private func scheduleInspectionStatusSettlement()", 1
        )[1].split("private func activateInspectionInKnownFrames()", 1)[0]
        self.assertIn("self.inspectionRequested", settlement)
        self.assertNotIn("self.inspectionRequested = false", settlement)
        self.assertGreaterEqual(
            self.browser.count("inspectionStatusTask?.cancel()"),
            5,
        )

    def test_frame_heartbeat_does_not_reactivate_an_authorized_route(self):
        self.assert_contracts(self.browser, [
            'if messageType == "frame-heartbeat"',
            "frameHeartbeatDates[documentToken] = Date()",
            "reportedRouteToken: body[\"routeToken\"] as? String",
            "reportedRouteToken: String?",
            "let frameAlreadyAuthorized =",
            "reportedRouteToken == self.authorizedRouteToken",
            "if frameAlreadyAuthorized",
            "post({type: 'frame-heartbeat', frameDepth});",
            "__miohInteractiveRescanRegisteredFrame",
        ])
        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        authorize = passive.split("const authorizeRoute = rawToken =>", 1)[1].split(
            "Object.defineProperty(window, '__miohInteractiveAuthorizeRoute'", 1
        )[0]
        route_changed = authorize.split("if (routeChanged) {", 1)[1].split(
            "\n        if (window !== window.top)", 1
        )[0]
        self.assertEqual(route_changed.count("post({type: 'frame-ready'"), 1)
        self.assertEqual(authorize.count("post({type: 'frame-ready'"), 1)
        rescan = passive.split(
            "const rescanAfterFrameRegistration = rawToken =>", 1
        )[1].split(
            "Object.defineProperty(window, '__miohInteractiveRescanRegisteredFrame'", 1
        )[0]
        self.assert_contracts(rescan, [
            "token !== authorizedRouteToken",
            "seen.clear();",
            "inspectCurrentPerformanceResources();",
            "replayRecentResourceEntries();",
            "scheduleScan();",
            "return true;",
        ])
        self.assertNotIn("post({type: 'frame-ready'", rescan)

    def test_relaxed_web_compatibility_is_the_default_browser_baseline(self):
        self.assert_contracts(self.browser, [
            "private static let relaxedWebCompatibilityEnabled = true",
            "configuration.preferences.javaScriptCanOpenWindowsAutomatically = true",
            "configuration.mediaTypesRequiringUserActionForPlayback = []",
            "if Self.relaxedWebCompatibilityEnabled { return }",
            "provenance: .script",
        ])
        action = self.navigation_delegate.split(
            "decidePolicyFor navigationAction: WKNavigationAction", 1
        )[1].split("decidePolicyFor navigationResponse: WKNavigationResponse", 1)[0]
        self.assertLess(
            action.index("if Self.relaxedWebCompatibilityEnabled"),
            action.index("if navigationAction.targetFrame == nil"),
        )
        relaxed_action = action.split(
            "if Self.relaxedWebCompatibilityEnabled", 1
        )[1].split("if navigationAction.targetFrame == nil", 1)[0]
        self.assert_contracts(relaxed_action, [
            "Self.isHighConfidenceAdvertisementNavigationURL(",
            "decisionHandler(.cancel)",
            "navigationAction.targetFrame?.isMainFrame == true",
            "prepareForMainNavigation(",
            "decisionHandler(.allow)",
            "return",
        ])

        response = self.navigation_delegate.split(
            "decidePolicyFor navigationResponse: WKNavigationResponse", 1
        )[1].split("didStartProvisionalNavigation", 1)[0]
        self.assertLess(
            response.index("if Self.relaxedWebCompatibilityEnabled"),
            response.index("responseAuthorizationValid"),
        )
        relaxed_response = response.split(
            "if Self.relaxedWebCompatibilityEnabled", 1
        )[1].split("if !navigationResponse.isForMainFrame", 1)[0]
        self.assert_contracts(relaxed_response, [
            "Self.isDirectMediaResponse(responseURL, response: response)",
            'sourceKind: "navigation-response"',
            "provenance: .script",
            "mainFrameChallengeResponse = false",
            "decisionHandler(.allow)",
        ])

        verification = self.browser.split("private func verifyFrameReady(", 1)[1].split(
            "private func authorizeSubframeNavigationAction", 1
        )[0]
        relaxed_frame = verification.split(
            "if Self.relaxedWebCompatibilityEnabled", 1
        )[1].split(
            "if frameInfo.isMainFrame {\n          guard let committedURL", 1
        )[0]
        self.assert_contracts(relaxed_frame, [
            "self.knownFrames[documentToken] = frameInfo",
            "self.activateInspection(in: frameInfo)",
            "guard frameInfo.request.url != nil else { return }",
            "self.webView.callAsyncJavaScript(",
            "__miohInteractiveRescanRegisteredFrame",
        ])
        self.assertLess(
            relaxed_frame.index("self.knownFrames[documentToken] = frameInfo"),
            relaxed_frame.index("self.activateInspection(in: frameInfo)"),
        )

        ui_delegate = self.browser.split(
            "extension IPadInteractiveMediaBrowser: WKUIDelegate", 1
        )[1].split("struct IPadInteractiveBrowserWebView", 1)[0]
        relaxed_popup = ui_delegate.split(
            "if Self.relaxedWebCompatibilityEnabled", 1
        )[1].split(
            "guard webView === self.webView,\n      openingPageWebView == nil", 1
        )[0]
        self.assert_contracts(relaxed_popup, [
            "maximumRelaxedTransientPopupCreationCount",
            "relaxedWebCompatibility: true",
            "allowsNavigation: { [weak self] url in",
            "Self.isHighConfidenceAdvertisementNavigationURL(url)",
            "expires: false",
            "let popupWebView = WKWebView(",
            "transientPopupCreationCount += 1",
            "retainTransientPopup(",
            "return popupWebView",
        ])
        self.assertLess(
            relaxed_popup.index("let popupWebView = WKWebView("),
            relaxed_popup.index("return popupWebView"),
        )

        show_opened = self.browser.split("func showOpenedPage()", 1)[1].split(
            "func returnToOpeningPage()", 1
        )[0]
        self.assertNotIn("guard openingPageWebView == nil", show_opened)
        self.assert_contracts(show_opened, [
            "if openingPageWebView == nil",
            "openingPageWebView = opener",
            "retainedPopupOpenerWebViews.append(opener)",
            "opener.navigationDelegate = nil",
            "opener.uiDelegate = nil",
            "adoptSettledWebView(openedWebView",
        ])
        self.assertLess(
            show_opened.index("retainedPopupOpenerWebViews.append(opener)"),
            show_opened.index("adoptSettledWebView(openedWebView"),
        )
        mark_ready = self.browser.split(
            "private func markTransientPopupReady", 1
        )[1].split("func showOpenedPage()", 1)[0]
        relaxed_mark_ready = mark_ready.split(
            "if Self.relaxedWebCompatibilityEnabled", 1
        )[1].split("scheduleTransientPopupRetirement", 1)[0]
        self.assert_contracts(relaxed_mark_ready, [
            "relaxedPopupAdoptionDelayNanoseconds",
            "let openerWebView = webView",
            "let openerGeneration = navigationGeneration",
            "let openerURL = webView.url?.absoluteString",
            "self.webView === openerWebView",
            "self.navigationGeneration == openerGeneration",
            "self.webView.url?.absoluteString == openerURL",
            "!self.isLoading",
            "self.retireTransientPopup(popupWebView)",
            "self.showOpenedPage()",
        ])
        self.assertLess(
            relaxed_mark_ready.index("Task.sleep"),
            relaxed_mark_ready.index("self.showOpenedPage()"),
        )
        adopt = self.browser.split("private func adoptSettledWebView", 1)[1].split(
            "func navigate(", 1
        )[0]
        self.assert_contracts(adopt, [
            "knownFrames.removeAll()",
            "webView = settledWebView",
            "beginInspectionForCurrentRoute()",
        ])
        self.assert_contracts(self.browser, [
            "if Self.relaxedWebCompatibilityEnabled,",
            'messageType == "challenge-hint" || messageType == "challenge-cleared"',
            "!Self.relaxedWebCompatibilityEnabled",
            "completionHandler(.performDefaultHandling, nil)",
        ])

    def test_advertisement_popup_is_rejected_without_disabling_player_navigation(self):
        classifier = self.browser.split(
            "private static func isHighConfidenceAdvertisementNavigationURL", 1
        )[1].split("private func retireTransientPopup", 1)[0]
        self.assert_contracts(classifier, [
            '"turnhub.net"',
            '"tsyndicate.com"',
            '"javhd-trk.com"',
            '"nettrck.store"',
            '"qpon"',
            '"snapptrckr.fun"',
            '"bluetrafficstream.com"',
            '"mnaspm.com"',
            '"mayzaent.com"',
            '"eix304.com"',
            "host == $0",
            'host.hasSuffix(".\\($0)")',
            'path.contains("/api/click")',
            'queryNames.contains("url")',
            'queryNames.contains("clickid")',
            'queryNames.contains("affid")',
        ])

        coordinator_action = self.browser.split(
            "private final class IPadTransientPopupCoordinator", 1
        )[1].split("decidePolicyFor navigationResponse", 1)[0]
        relaxed_action = coordinator_action.split("if relaxedWebCompatibility", 1)[1]
        self.assertIn("allowsNavigation(navigationAction.request.url)", relaxed_action)
        self.assertIn("closeHandler(webView)", relaxed_action)
        self.assertLess(
            relaxed_action.index("allowsNavigation(navigationAction.request.url)"),
            relaxed_action.index("decisionHandler(.allow)"),
        )

        mark_ready = self.browser.split(
            "private func markTransientPopupReady", 1
        )[1].split("func showOpenedPage()", 1)[0]
        self.assert_contracts(mark_ready, [
            "Self.isHighConfidenceAdvertisementNavigationURL(popupURL)",
            "retireTransientPopup(popupWebView)",
            "showOpenedPage()",
        ])
        self.assertLess(
            mark_ready.index("Self.isHighConfidenceAdvertisementNavigationURL"),
            mark_ready.index("showOpenedPage()"),
        )

    def test_confirmed_floating_ad_hls_is_never_a_browser_media_candidate(self):
        classifier = self.browser.split(
            "private static func isHighConfidenceAdvertisementMediaURL", 1
        )[1].split("private func retireTransientPopup", 1)[0]
        self.assert_contracts(classifier, [
            'host == "saawsedge.com"',
            'host.hasSuffix(".saawsedge.com")',
        ])

        insertion = self.browser.split("private func insertCandidate", 1)[1].split(
            "private static func isReadyCandidate", 1
        )[0]
        self.assertLess(
            insertion.index("Self.isHighConfidenceAdvertisementMediaURL"),
            insertion.index("let key = candidate.url.absoluteString"),
        )

        media_source = self.browser.split("private func receiveMediaSourceMessage", 1)[1].split(
            "private func verifyFrameReady", 1
        )[0]
        self.assertIn(
            "!Self.isHighConfidenceAdvertisementMediaURL(safeURL)",
            media_source,
        )

        ordering = self.browser.split("private static func resolutionOrder", 1)[1].split(
            "private static func priorityComponents", 1
        )[0]
        self.assertLess(
            ordering.index("!isHighConfidenceAdvertisementMediaURL($0.url)"),
            ordering.index("func selectionTier"),
        )

    def test_only_high_confidence_compact_ad_overlays_are_hidden(self):
        script = self.browser.split(
            'private static let passiveInstrumentationScript = """', 1
        )[1].split('private static let pageNetworkObservationScript = """', 1)[0]
        self.assert_contracts(script, [
            "const advertisementNavigationHostSuffixes = [",
            "const advertisementMediaHostSuffixes = ['saawsedge.com']",
            "'turnhub.net'",
            "'tsyndicate.com'",
            "const highConfidenceAdvertisementEvidence",
            "element.currentSrc || element.getAttribute('src')",
            "const isCompactLowerRightOverlay",
            "width > viewportWidth * 0.65",
            "height > viewportHeight * 0.9",
            "width * height > viewportWidth * viewportHeight * 0.45",
            "const advertisementOverlayRoot",
            "cursor.querySelector?.('video,audio')",
            "position === 'fixed'",
            "position === 'absolute'",
            "return outermostCandidate",
            "const suppressHighConfidenceAdvertisementOverlays",
            "const selector = 'a[href],iframe[src],video[src],source[src]'",
            "source.getAttribute?.(advertisementOverlayMarker) === 'true'",
            "root.getAttribute?.(advertisementOverlayMarker) === 'true'",
            "source.setAttribute(advertisementOverlayMarker, 'true')",
            "root.style.setProperty('display', 'none', 'important')",
            "root.style.setProperty('visibility', 'hidden', 'important')",
            "root.style.setProperty('pointer-events', 'none', 'important')",
            "suppressHighConfidenceAdvertisementOverlays(document)",
            "attributeFilter: sourceAttributes",
        ])
        self.assertNotIn("root.remove()", script)
        self.assertNotIn("preventDefault()", script)
        overlay_root = script.split("const advertisementOverlayRoot", 1)[1].split(
            "const suppressAdvertisementElement", 1
        )[0]
        self.assertNotIn("sticky", overlay_root)
        self.assertNotIn("return source", overlay_root)
        self.assertLess(
            script.index("cursor.querySelector?.('video,audio')"),
            script.index("isCompactLowerRightOverlay(cursor)"),
        )

    def test_page_world_custom_event_crosses_into_isolated_content_world(self):
        xcrun = shutil.which("xcrun")
        if sys.platform != "darwin" or xcrun is None:
            self.skipTest("WKContentWorld behavior probe requires macOS and Xcode")
        with tempfile.TemporaryDirectory(prefix="mioh-wkcontentworld-") as directory:
            executable = Path(directory) / "bridge-probe"
            subprocess.run(
                [
                    xcrun,
                    "swiftc",
                    "-parse-as-library",
                    "-framework",
                    "AppKit",
                    "-framework",
                    "WebKit",
                    str(CONTENT_WORLD_HARNESS),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [str(executable)],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertIn("WKContentWorld bridge probe passed", completed.stdout)

    def test_page_world_observer_captures_extensionless_hls_mime(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = self.page_network_observer_runtime_source()
        harness = f"""
const bridgeEventName = 'mioh-hls-00000000-0000-4000-8000-000000000000';
let observationEpoch = '11111111-1111-4111-8111-111111111111';
const events = [];
global.CustomEvent = class CustomEvent {{
  constructor(type, options) {{ this.type = type; this.detail = options.detail; }}
}};
global.document = {{
  baseURI: 'https://page.example/watch',
  dispatchEvent(event) {{ events.push(JSON.parse(event.detail)); return true; }}
}};
global.location = {{href: 'https://page.example/watch'}};
global.window = {{}};
const responseObject = {{
  url: 'https://cdn.example/session/opaque-token',
  headers: {{ get(name) {{
    return String(name).toLowerCase() === 'content-type'
      ? 'application/vnd.apple.mpegurl; charset=utf-8' : null;
  }} }}
}};
let originalPromise;
window.fetch = function() {{
  originalPromise = Promise.resolve(responseObject);
  return originalPromise;
}};
class MockXHR {{
  constructor() {{
    this.responseURL = 'https://cdn.example/session/xhr-token';
    this.listeners = {{}};
  }}
  addEventListener(name, callback) {{ this.listeners[name] = callback; }}
  getResponseHeader(name) {{
    return String(name).toLowerCase() === 'content-type'
      ? 'application/x-mpegurl' : null;
  }}
  open() {{ return 'opened'; }}
}}
global.XMLHttpRequest = MockXHR;
const observerSource = {json.dumps(script)};
const installed = eval(observerSource);
(async () => {{
  const returnedPromise = window.fetch('https://cdn.example/session/opaque-token');
  const samePromise = returnedPromise === originalPromise;
  const response = await returnedPromise;
  const sameEpochReinstalled = eval(observerSource);
  await window.fetch('https://cdn.example/session/opaque-token');
  const afterSameEpoch = events.length;
  location.href = 'https://page.example/watch/second';
  document.baseURI = location.href;
  const reinstalled = eval(observerSource);
  await window.fetch('https://cdn.example/session/opaque-token');
  const afterReplay = events.length;
  await window.fetch('https://cdn.example/session/post-token', {{method: 'POST'}});
  const xhr = new XMLHttpRequest();
  const openResult = xhr.open('GET', 'https://cdn.example/session/xhr-token');
  xhr.listeners.load();
  process.stdout.write(JSON.stringify({{
    installed,
    sameEpochReinstalled,
    afterSameEpoch,
    reinstalled,
    sameResponse: response === responseObject,
    samePromise,
    postAddedEvents: events.length - afterReplay - 1,
    openResult,
    events
  }}));
}})().catch(error => {{ console.error(error); process.exit(1); }});
"""
        completed = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["installed"])
        self.assertTrue(result["sameEpochReinstalled"])
        self.assertEqual(result["afterSameEpoch"], 1)
        self.assertTrue(result["reinstalled"])
        self.assertTrue(result["sameResponse"])
        self.assertTrue(result["samePromise"])
        self.assertEqual(result["postAddedEvents"], 0)
        self.assertEqual(result["openResult"], "opened")
        self.assertEqual(
            [event["kind"] for event in result["events"]],
            [
                "page-fetch-hls-response",
                "page-fetch-hls-response",
                "page-xhr-hls-response",
            ],
        )
        self.assertTrue(all(event["hlsResponse"] for event in result["events"]))
        self.assertEqual(
            [event["pageURL"] for event in result["events"]],
            [
                "https://page.example/watch",
                "https://page.example/watch/second",
                "https://page.example/watch/second",
            ],
        )
        self.assertEqual(
            [event["observationEpoch"] for event in result["events"]],
            [
                "11111111-1111-4111-8111-111111111111",
                "11111111-1111-4111-8111-111111111111",
                "11111111-1111-4111-8111-111111111111",
            ],
        )

    def test_page_world_observer_cannot_block_fetch_when_page_globals_throw(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = self.page_network_observer_runtime_source()
        harness = f"""
const bridgeEventName = 'mioh-hls-12121212-1212-4212-8212-121212121212';
const observationEpoch = '34343434-3434-4434-8434-343434343434';
const events = [];
global.CustomEvent = class CustomEvent {{
  constructor(type, options) {{ this.type = type; this.detail = options.detail; }}
}};
global.location = {{href: 'https://page.example/exception'}};
global.document = {{
  baseURI: location.href,
  dispatchEvent(event) {{ events.push(JSON.parse(event.detail)); return true; }}
}};
let originalPromise;
let fetchCalls = 0;
global.window = {{
  fetch() {{
    fetchCalls += 1;
    originalPromise = Promise.resolve({{
      url: 'https://cdn.example/exception/stream',
      headers: {{get() {{ return 'application/vnd.apple.mpegurl'; }}}}
    }});
    return originalPromise;
  }}
}};
global.XMLHttpRequest = undefined;
eval({json.dumps(script)});
const originalString = String;
const originalReflectApply = Reflect.apply;
global.String = () => {{ throw new Error('observer String failure'); }};
Reflect.apply = () => {{ throw new Error('late Reflect mutation'); }};
const returnedPromise = window.fetch('https://cdn.example/exception/stream');
const samePromise = returnedPromise === originalPromise;
global.String = originalString;
Reflect.apply = originalReflectApply;
returnedPromise.then(() => process.stdout.write(JSON.stringify({{
  samePromise,
  fetchCalls,
  events
}})));
"""
        completed = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["samePromise"])
        self.assertEqual(result["fetchCalls"], 1)
        self.assertEqual(result["events"], [])

    def test_page_world_observer_preserves_only_explicit_credential_modes(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = self.page_network_observer_runtime_source()
        harness = f"""
const bridgeEventName = 'mioh-hls-90909090-9090-4090-8090-909090909090';
const observationEpoch = '91919191-9191-4191-8191-919191919191';
const events = [];
global.CustomEvent = class CustomEvent {{
  constructor(type, options) {{ this.type = type; this.detail = options.detail; }}
}};
global.location = {{href: 'https://page.example/credentials'}};
global.document = {{
  baseURI: location.href,
  dispatchEvent(event) {{ events.push(JSON.parse(event.detail)); return true; }}
}};
global.window = {{
  fetch(input) {{
    const url = typeof input === 'string' ? input : input.url;
    return Promise.resolve({{
      url,
      headers: {{get() {{ return 'application/vnd.apple.mpegurl'; }}}}
    }});
  }}
}};
class MockXHR {{
  constructor() {{
    this.listeners = {{}};
    this.responseURL = '';
    this.withCredentials = false;
  }}
  addEventListener(name, callback) {{ this.listeners[name] = callback; }}
  getResponseHeader() {{ return 'application/vnd.apple.mpegurl'; }}
  open(method, url) {{ this.responseURL = String(url); return method; }}
  send() {{ return 'sent'; }}
}}
global.XMLHttpRequest = MockXHR;
eval({json.dumps(script)});
(async () => {{
  await window.fetch(
    'https://cdn.example/credentials/init.m3u8',
    {{credentials: 'include'}}
  );
  await window.fetch({{
    url: 'https://cdn.example/credentials/request.m3u8',
    method: 'GET',
    credentials: 'include'
  }});
  await window.fetch({{
    url: 'https://cdn.example/credentials/override.m3u8',
    method: 'GET',
    credentials: 'include'
  }}, {{credentials: 'same-origin'}});
  const xhr = new XMLHttpRequest();
  xhr.open('GET', 'https://cdn.example/credentials/xhr.m3u8');
  xhr.withCredentials = true;
  const sendResult = xhr.send();
  xhr.listeners.load();
  process.stdout.write(JSON.stringify({{events, sendResult}}));
}})().catch(error => {{ console.error(error); process.exit(1); }});
"""
        completed = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        responses = {
            event["url"]: event["includesCredentials"]
            for event in result["events"]
            if event["kind"].endswith("-response")
        }
        self.assertEqual(
            responses,
            {
                "https://cdn.example/credentials/init.m3u8": True,
                "https://cdn.example/credentials/request.m3u8": True,
                "https://cdn.example/credentials/override.m3u8": False,
                "https://cdn.example/credentials/xhr.m3u8": True,
            },
        )
        self.assertEqual(result["sendResult"], "sent")

    def test_page_world_observer_accepts_only_get_and_head_methods(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = self.page_network_observer_runtime_source()
        harness = f"""
const bridgeEventName = 'mioh-hls-56565656-5656-4656-8656-565656565656';
const observationEpoch = '78787878-7878-4878-8878-787878787878';
const events = [];
global.CustomEvent = class CustomEvent {{
  constructor(type, options) {{ this.type = type; this.detail = options.detail; }}
}};
global.location = {{href: 'https://page.example/methods'}};
global.document = {{
  baseURI: location.href,
  dispatchEvent(event) {{ events.push(JSON.parse(event.detail)); return true; }}
}};
global.window = {{
  fetch(input) {{
    return Promise.resolve({{
      url: String(input),
      headers: {{get() {{ return 'application/vnd.apple.mpegurl'; }}}}
    }});
  }}
}};
class MockXHR {{
  constructor() {{ this.listeners = {{}}; this.responseURL = ''; }}
  addEventListener(name, callback) {{ this.listeners[name] = callback; }}
  getResponseHeader() {{ return 'application/vnd.apple.mpegurl'; }}
  open(method, url) {{ this.responseURL = String(url); return method; }}
}}
global.XMLHttpRequest = MockXHR;
eval({json.dumps(script)});
(async () => {{
  await window.fetch('https://cdn.example/methods/fetch-default');
  await window.fetch('https://cdn.example/methods/fetch-head', {{method: 'HEAD'}});
  await window.fetch('https://cdn.example/methods/fetch-post', {{method: 'POST'}});
  await window.fetch('https://cdn.example/methods/fetch-null', {{method: null}});
  for (const [method, suffix] of [
    ['GET', 'xhr-get'], ['HEAD', 'xhr-head'], ['POST', 'xhr-post'], [null, 'xhr-null']
  ]) {{
    const xhr = new XMLHttpRequest();
    xhr.open(method, `https://cdn.example/methods/${{suffix}}`);
    xhr.listeners.load();
  }}
  process.stdout.write(JSON.stringify(events));
}})().catch(error => {{ console.error(error); process.exit(1); }});
"""
        completed = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        events = json.loads(completed.stdout)
        self.assertEqual(
            [event["url"] for event in events],
            [
                "https://cdn.example/methods/fetch-default",
                "https://cdn.example/methods/fetch-head",
                "https://cdn.example/methods/xhr-get",
                "https://cdn.example/methods/xhr-head",
            ],
        )

    def test_page_world_route_cap_resets_on_same_epoch_url_change(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = self.page_network_observer_runtime_source()
        harness = f"""
const bridgeEventName = 'mioh-hls-88888888-8888-4888-8888-888888888888';
const observationEpoch = '99999999-9999-4999-8999-999999999999';
const events = [];
global.CustomEvent = class CustomEvent {{
  constructor(type, options) {{ this.type = type; this.detail = options.detail; }}
}};
global.location = {{href: 'https://page.example/route-a'}};
global.document = {{
  baseURI: location.href,
  dispatchEvent(event) {{ events.push(JSON.parse(event.detail)); return true; }}
}};
global.window = {{
  fetch(input) {{
    return Promise.resolve({{
      url: String(input),
      headers: {{get() {{ return 'application/vnd.apple.mpegurl'; }}}}
    }});
  }}
}};
global.XMLHttpRequest = undefined;
const observerSource = {json.dumps(script)};
eval(observerSource);
(async () => {{
  for (let index = 0; index < 128; index += 1) {{
    await window.fetch(`https://cdn.example/route-a/${{index}}`);
  }}
  const beforeRouteChange = events.length;
  location.href = 'https://page.example/route-b';
  document.baseURI = location.href;
  const sameEpochReinstalled = eval(observerSource);
  await window.fetch('https://cdn.example/route-b/main');
  process.stdout.write(JSON.stringify({{
    beforeRouteChange,
    afterRouteChange: events.length,
    sameEpochReinstalled,
    lastEvent: events.at(-1)
  }}));
}})().catch(error => {{ console.error(error); process.exit(1); }});
"""
        completed = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["beforeRouteChange"], 128)
        self.assertEqual(result["afterRouteChange"], 129)
        self.assertTrue(result["sameEpochReinstalled"])
        self.assertEqual(
            result["lastEvent"]["pageURL"],
            "https://page.example/route-b",
        )

    def test_page_world_ignores_a_response_after_its_route_was_replaced(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = self.page_network_observer_runtime_source()
        harness = f"""
const bridgeEventName = 'mioh-hls-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const observationEpoch = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const events = [];
let resolveFetch;
global.CustomEvent = class CustomEvent {{
  constructor(type, options) {{ this.type = type; this.detail = options.detail; }}
}};
global.location = {{href: 'https://page.example/route-a'}};
global.document = {{
  baseURI: location.href,
  dispatchEvent(event) {{ events.push(JSON.parse(event.detail)); return true; }}
}};
global.window = {{
  fetch() {{ return new Promise(resolve => {{ resolveFetch = resolve; }}); }}
}};
global.XMLHttpRequest = undefined;
eval({json.dumps(script)});
const pending = window.fetch('https://cdn.example/route-a/late');
location.href = 'https://page.example/route-b';
document.baseURI = location.href;
resolveFetch({{
  url: 'https://cdn.example/route-a/late',
  headers: {{get() {{ return 'application/vnd.apple.mpegurl'; }}}}
}});
pending.then(() => process.stdout.write(JSON.stringify(events)));
"""
        completed = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout), [])

    def test_page_world_observer_keeps_xhr_when_fetch_is_frozen(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = self.page_network_observer_runtime_source()
        harness = f"""
const bridgeEventName = 'mioh-hls-33333333-3333-4333-8333-333333333333';
const observationEpoch = '44444444-4444-4444-8444-444444444444';
const events = [];
global.CustomEvent = class CustomEvent {{
  constructor(type, options) {{ this.type = type; this.detail = options.detail; }}
}};
global.location = {{href: 'https://page.example/frozen-fetch'}};
global.document = {{
  baseURI: location.href,
  dispatchEvent(event) {{ events.push(JSON.parse(event.detail)); return true; }}
}};
global.window = {{}};
Object.defineProperty(window, 'fetch', {{
  value() {{ return Promise.reject(new Error('must not be called')); }},
  writable: false,
  configurable: false
}});
class MockXHR {{
  constructor() {{
    this.responseURL = 'https://cdn.example/opaque-xhr';
    this.listeners = {{}};
  }}
  addEventListener(name, callback) {{ this.listeners[name] = callback; }}
  getResponseHeader() {{ return 'application/vnd.apple.mpegurl'; }}
  open() {{ return 'xhr-opened'; }}
}}
global.XMLHttpRequest = MockXHR;
const installed = eval({json.dumps(script)});
const xhr = new XMLHttpRequest();
const openResult = xhr.open('GET', xhr.responseURL);
xhr.listeners.load();
process.stdout.write(JSON.stringify({{installed, openResult, events}}));
"""
        completed = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["installed"])
        self.assertEqual(result["openResult"], "xhr-opened")
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["kind"], "page-xhr-hls-response")

    def test_route_gap_response_keeps_its_start_page_and_previous_epoch(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = self.page_network_observer_runtime_source()
        harness = f"""
const bridgeEventName = 'mioh-hls-55555555-5555-4555-8555-555555555555';
let observationEpoch = '66666666-6666-4666-8666-666666666666';
const events = [];
let resolveFetch;
global.CustomEvent = class CustomEvent {{
  constructor(type, options) {{ this.type = type; this.detail = options.detail; }}
}};
global.location = {{href: 'https://page.example/route-a'}};
global.document = {{
  baseURI: location.href,
  dispatchEvent(event) {{ events.push(JSON.parse(event.detail)); return true; }}
}};
global.window = {{
  fetch() {{ return new Promise(resolve => {{ resolveFetch = resolve; }}); }}
}};
global.XMLHttpRequest = undefined;
const observerSource = {json.dumps(script)};
eval(observerSource);
location.href = 'https://page.example/route-b';
document.baseURI = location.href;
const pending = window.fetch('https://cdn.example/opaque-route-b');
observationEpoch = '77777777-7777-4777-8777-777777777777';
eval(observerSource);
resolveFetch({{
  url: 'https://cdn.example/opaque-route-b',
  headers: {{get() {{ return 'application/vnd.apple.mpegurl'; }}}}
}});
pending.then(() => process.stdout.write(JSON.stringify(events)));
"""
        completed = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        events = json.loads(completed.stdout)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["pageURL"], "https://page.example/route-b")
        self.assertEqual(
            events[0]["observationEpoch"],
            "66666666-6666-4666-8666-666666666666",
        )

    def test_later_hls_response_can_upgrade_an_earlier_url_hint(self):
        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        self.assert_contracts(passive, [
            "const seen = new Map();",
            "const evidenceRank = (() =>",
            "const priorRank = seen.get(value);",
            "priorRank <= evidenceRank",
            "seen.set(value, evidenceRank);",
        ])
        priorities = self.browser.split(
            "private static func priorityComponents", 1
        )[1].split("private static func worstCandidateIndex", 1)[0]
        self.assert_contracts(priorities, [
            'case "navigation-response", "fetch-media-response", "xhr-media-response":',
            'case "page-fetch-hls-response", "page-xhr-hls-response":',
            'case "script-text":',
            'case "performance": sourcePriority = 6',
        ])
        self.assertLess(
            priorities.index('case "performance": sourcePriority = 6'),
            priorities.index('case "script-text": sourcePriority = 7'),
        )
        insertion = self.browser.split(
            "private func insertCandidate", 1
        )[1].split("private static func isReadyCandidate", 1)[0]
        self.assert_contracts(insertion, [
            "let relayEvidenceChanged = recordHLSRelayEvidence(candidate)",
            "nativeProvenanceUpgrade || Self.isPreferred(candidate, over: existing)",
            "if Self.isReadyCandidate(replacement ?? existing)",
            "scheduleReadyCandidate()",
        ])
        self.assert_contracts(self.store, [
            "case .unverifiedMediaResponse: 2",
            "case .directHint: 3",
            "case .pageLead: 4",
        ])
        self.assert_contracts(self.browser, [
            "private static func isUnverifiedMediaResponseHint",
            'case "script-text": "script内URL"',
            "? .unverifiedMediaResponse",
        ])

    def test_candidates_use_trusted_frame_cookie_ua_and_safe_referer(self):
        self.assert_contracts(self.browser, [
            "message.frameInfo.request.url",
            "getAllCookies",
            'evaluateJavaScript("navigator.userAgent")',
            'cookie.name.lowercased() == "cf_clearance"',
            "IPadMediaRequestCookie.init",
            "IPadMediaRequestContext(",
            "origin: Self.requiresBrowserOriginHeader(candidate.sourceKind)",
            "prioritizedCookies + remainingCookies",
            "isVerifiedPageHLSObservation",
            "frame.query = nil",
            "frame.fragment = nil",
            'frame.path = "/"',
        ])
        self.assertNotIn('body["frameURL"]', self.browser)
        self.assertNotIn("document.cookie", self.browser)

    def test_candidate_collection_is_bounded_and_quiet_debounced(self):
        self.assert_contracts(self.browser, [
            "maximumCandidateCount = 128",
            "maximumCandidateURLLength = 8_192",
            "worstCandidate",
            "quietTask?.cancel()",
            "2_000_000_000",
            "readyCandidateGeneration",
            "candidateRevision = nextGeneration(after: candidateRevision)",
        ])

    def test_visiting_a_page_never_auto_starts_resolution_or_download(self):
        self.assert_contracts(self.view, [
            'Label("配信を解析", systemImage: "magnifyingglass")',
            "private func analyzeBrowserCandidates()",
        ])
        self.assertNotIn(
            ".onChange(of: interactiveBrowser.readyCandidateGeneration)",
            self.view,
        )

    def test_direct_navigation_media_is_captured_before_webkit_cancels_it(self):
        self.assert_contracts(self.browser, [
            "isDirectMediaResponse",
            'sourceKind: "navigation-response"',
            "navigationResponse.canShowMIMEType",
            "isLoading = false",
            'statusMessage = "再生可能性を確認した配信URLがあります。「配信を解析」を押してください。"',
        ])

    def test_candidate_count_opens_a_detailed_candidate_list(self):
        self.assert_contracts(self.browser, [
            "struct CandidateSummary: Identifiable, Equatable",
            "@Published private(set) var candidateSummaries",
            "private func refreshCandidateSummaries()",
            "candidateCount = candidateSummaries.count",
            '? "再生ソース"',
            '? "応答確認済み" : "未確認候補"',
        ])
        self.assert_contracts(self.view, [
            "showingBrowserCandidates = true",
            'systemImage: "list.bullet.rectangle"',
            "private var browserCandidateList: some View",
            "ForEach(interactiveBrowser.candidateSummaries)",
            '.navigationTitle("配信候補 \\(interactiveBrowser.candidateCount)件")',
            'Button("閉じる")',
        ])

    def test_navigation_and_analysis_are_generation_guarded(self):
        self.assert_contracts(self.browser, [
            "navigationGeneration",
            "activeNavigation",
            "navigation === activeNavigation",
            "isCancelledNavigationError",
        ])
        self.assert_contracts(self.view, [
            ".onChange(of: interactiveBrowser.navigationGeneration)",
            "browserAnalysisGeneration",
            "browserAnalysisIsCurrent(generation, navigationGeneration)",
            "urlAnalysisGeneration",
            "urlAnalysisIsCurrent(generation)",
            "scenePhase == .active",
            "!interactiveBrowser.challengeActive",
            "!interactiveBrowser.isLoading",
            ".onChange(of: interactiveBrowser.challengeActive)",
            "invalidateBrowserAnalysis(closePage: true)",
        ])

    def test_close_replaces_webview_but_keeps_standard_store(self):
        self.assert_contracts(self.browser, [
            "let closingWebView = webView",
            "removeScriptMessageHandler",
            "contentWorld: Self.instrumentationContentWorld",
            "let replacement = Self.makeWebView(",
            "websiteDataStore: websiteDataStore",
            "webView = replacement",
            "webViewGeneration = nextGeneration",
        ])

    def test_cookie_snapshot_excludes_unrelated_origins(self):
        prioritized = self.browser.split(
            "private static func prioritizedCookies", 1
        )[1].split("private static let blankPageHTML", 1)[0]
        self.assertIn("return relatedClearance + related", prioritized)
        self.assertNotIn("unrelatedClearance", prioritized)
        self.assertNotIn("+ remaining", prioritized)

    def test_player_source_tier_precedes_media_extension(self):
        preferred = self.browser.split(
            "private static func isPreferred", 1
        )[1].split("private static func priorityComponents", 1)[0]
        self.assertLess(
            preferred.index("left.source"), preferred.index("left.media")
        )
        self.assert_contracts(self.browser, [
            "private static func resolutionOrder",
            "Array(sorted.prefix(8))",
            "priorityComponents(candidate).media == 0",
            "leading.count < IPadBrowserMediaSourceSelector.maximumPlayableChoices",
        ])

    def test_spa_page_changes_invalidate_old_candidates(self):
        self.assert_contracts(self.browser, [
            'messageType == "page-location"',
            "lastPageLocationToken",
            "pendingSameDocumentPageURL",
            "acceptSameDocumentRouteInPlace",
            "resetSameDocumentNavigationState",
            "processSameDocumentPageLocation",
            "pendingChallengePageURL = rawPageURL",
            "acceptingScriptCandidates = false",
            "safeReportedURL.absoluteString == currentURL.absoluteString",
            "reportedURL.absoluteString == currentWebViewURL.absoluteString",
            "guard acceptingScriptCandidates,",
            'if messageType == "media-source"',
            'guard messageType == "candidate"',
            "documentToken",
            "authorizedRouteToken",
            "authorizedRouteURL",
            "verifyFrameReady",
            "__miohInteractiveDocumentToken",
            "__miohInteractiveAuthorizeRoute",
            "routeToken == authorizedRouteToken",
            "didReceiveServerRedirectForProvisionalNavigation",
            "maximumMainNavigationRedirectCount",
            "guard webView === self.webView, !isClosingPage,",
            "navigation === activeNavigation",
            "if (currentPageURL === observedPageURL) return;",
            "currentPageURL.slice(0, \\(maximumCandidateURLLength))",
            "window.__miohInteractiveNotifyPageChange?.();",
            "post({type: 'frame-ready', frameDepth});",
            "'currententrychange',",
            "'popstate',",
            "'hashchange',",
        ])
        route_change = self.browser.split(
            "private func processSameDocumentPageLocation", 1
        )[1].split("private func acceptSameDocumentRouteInPlace", 1)[0]
        self.assertIn("acceptSameDocumentRouteInPlace()", route_change)
        self.assertNotIn("reloadForSameDocumentNavigation()", route_change)
        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        self.assertNotIn("history.pushState", passive)
        self.assertNotIn("history.replaceState", passive)
        self.assertNotIn("history[method] =", passive)
        receive = self.browser.split("private func receiveScriptMessage(", 1)[1].split(
            "private func verifyFrameReady(", 1
        )[0]
        self.assert_contracts(receive, [
            "let authorizedRouteURL",
            "authorizedRouteURL == webView.url?.absoluteString",
        ])
        inspection = self.browser.split(
            "private func beginInspectionForCurrentRoute()", 1
        )[1].split("private func activateInspection(in frame", 1)[0]
        self.assert_contracts(inspection, [
            "let currentWebViewURL = webView.url",
            "lastPageLocationToken != currentRouteURL",
            "processSameDocumentPageLocation(currentRouteURL)",
            "authorizedRouteURL = currentRouteURL",
        ])
        snapshot = self.browser.split("func snapshotCandidates()", 1)[1].split(
            "private func prepareForMainNavigation", 1
        )[0]
        self.assert_contracts(snapshot, [
            "if let authorizedRouteURL",
            "webView.url?.absoluteString != authorizedRouteURL",
            "processSameDocumentPageLocation(currentURL.absoluteString)",
            "let snapshotWebView = webView",
            "let snapshotNavigationGeneration = navigationGeneration",
            "let snapshotRouteURL = authorizedRouteURL",
            "let snapshotPageURL = webView.url?.absoluteString",
            "let requestCookies = await allRequestCookies()",
            "guard webView === snapshotWebView",
            "navigationGeneration == snapshotNavigationGeneration",
            "authorizedRouteURL == snapshotRouteURL",
            "webView.url?.absoluteString == snapshotPageURL",
            "return []",
        ])
        self.assertLess(
            snapshot.index("let requestCookies = await allRequestCookies()"),
            snapshot.index("guard webView === snapshotWebView"),
        )
        self.assertLess(
            snapshot.index("guard webView === snapshotWebView"),
            snapshot.index("var snapshot = Array(candidates.values)"),
        )

    def test_response_and_cookie_provenance_are_native_authorized(self):
        self.assert_contracts(self.browser, [
            "enum Provenance",
            "case script",
            "case currentPage",
            "case mainNavigationResponse",
            "case subframeNavigationResponse",
            "responseAuthorizationValid",
            "guard responseAuthorizationValid else",
            "decisionHandler(.cancel)",
            "cookieSourceURL:",
            "allowsCrossSiteCredentialReplay:",
        ])

    def test_subframe_navigation_is_target_epoch_and_one_shot_guarded(self):
        self.assert_contracts(self.browser, [
            "SubframeNavigationAuthorization",
            "subframeDocumentEpochs",
            "pendingSubframeAuthorizations",
            "completedSubframeAuthorizations",
            "authorizeSubframeNavigationAction(",
            '"return window.__miohInteractiveDocumentToken ?? null;"',
            "in: targetFrame",
            "guard currentDocumentEpoch == documentEpoch else { return }",
            "knownFrames.removeValue(forKey: targetPriorDocumentToken)",
            "consumeSubframeAuthorization(",
            "pendingSubframeAuthorizations.remove(at: index)",
            "responseAuthorizationValid = subframeAuthorization != nil",
        ])
        self.assertNotIn("knownFrames = knownFrames.filter", self.browser)
        self.assertNotIn("authorizedSubframeResponseURLs", self.browser)
        self.assertNotIn("provisionalSubframeURLs", self.browser)

    def test_initial_cloudflare_child_uses_a_bounded_source_frame_fallback(self):
        authorization = self.browser.split(
            "private func authorizeSubframeNavigationAction(", 1
        )[1].split("private func makeSubframeNavigationAuthorization(", 1)[0]
        self.assert_contracts(authorization, [
            "sourceFrame: WKFrameInfo",
            "targetFrame: WKFrameInfo",
            "guard case .success(let value) = result",
            "guard value is NSNull",
            "isUnregisteredInitialTargetDocumentToken(documentToken)",
            "allowsInitialChildChallengeFallback",
            "isChallengeNavigation",
            "isAllowedNativeChallengeFrameURL(destinationURL)",
            "sourceFrame.isMainFrame",
            'targetFrame.request.url?.absoluteString == "about:blank"',
            "initialChildChallengeDecisionID",
            "initialChildChallengeFallbackGeneration",
            "hasOutstandingInitialChildChallengeAuthorization",
            "authorizeInitialChildChallengeNavigation(",
            "beginInitialChildChallengeFallback(",
            "in: sourceFrame",
            "for attempt in 0..<8",
            "50_000_000",
            "clearInitialChildChallengeDecision(decisionID)",
        ])
        self.assertNotIn("case .failure", authorization)

        initial = self.browser.split(
            "private func makeInitialChildChallengeAuthorization(", 1
        )[1].split(
            "private func makeInitialUserActivatedChildAuthorization(", 1
        )[0]
        self.assert_contracts(initial, [
            "initiatorDocumentToken",
            "initiatorDocumentEpoch",
            "targetPriorDocumentToken: String?",
            "targetPriorDocumentToken: targetPriorDocumentToken",
            "isInitialChildChallengeFallback: true",
            "mainFrameChallengeResponseGeneration == generation",
            "knownFrames[initiatorDocumentToken]?.isMainFrame == true",
            "committedMainDocumentURL?.absoluteString == initiatorURL.absoluteString",
            "chain.remainingHopCount -= 1",
            "!hasOutstandingInitialChildChallengeAuthorization",
            "initiatorDocumentEpoch: nil",
            "nativeMainChallengeGeneration: generation",
            "initialChildChallengeFallbackGeneration != generation",
            "initialChildChallengeFallbackGeneration = generation",
        ])
        self.assertNotIn(
            "knownFrames.removeValue(forKey: initiatorDocumentToken)", initial
        )
        self.assertNotIn("subframeNavigationChains[initiatorDocumentToken]", initial)
        clear_state = self.browser.split(
            "private func clearSubframeNavigationState()", 1
        )[1].split("private func completeChallengeIfPossible()", 1)[0]
        self.assertNotIn("initialChildChallengeFallbackGeneration = nil", clear_state)
        outstanding = self.browser.split(
            "private var hasOutstandingInitialChildChallengeAuthorization", 1
        )[1].split("private func authorizationInitiatorIsCurrent(", 1)[0]
        self.assert_contracts(outstanding, [
            "pendingSubframeAuthorizations.contains",
            "completedSubframeAuthorizations.contains",
            "$0.isInitialChildChallengeFallback",
            "$0.chain.mainNavigationGeneration == navigationGeneration",
        ])

        unregistered = self.browser.split(
            "private func isUnregisteredInitialTargetDocumentToken(", 1
        )[1].split("private func authorizeInitialChildChallengeNavigation(", 1)[0]
        self.assert_contracts(unregistered, [
            "documentToken != mainDocumentToken",
            "knownFrames[documentToken] == nil",
            "subframeDocumentEpochs[documentToken] == nil",
            "subframeInspectionRouteTokens[documentToken] == nil",
            "subframeNavigationChains[documentToken] == nil",
            "pendingSubframeAuthorizations.contains",
            "completedSubframeAuthorizations.contains",
        ])

        string_guard = authorization.split(
            "guard !documentToken.isEmpty", 1
        )[1].split("if self.subframeDocumentEpochs[documentToken] == nil", 1)[0]
        self.assertIn(
            "isUnregisteredInitialTargetDocumentToken(documentToken)", string_guard
        )
        self.assertIn("beginInitialChildChallengeFallback(", string_guard)
        self.assertNotIn("subframeDocumentEpochs[documentToken] =", string_guard)

    def test_user_activated_dynamic_player_child_is_bounded_and_one_shot(self):
        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        self.assert_contracts(passive, [
            "event?.isTrusted !== true",
            "trustedActivationSerial",
            "consumedActivationSerial",
            "age > 3500",
            "__miohInteractiveConsumeTrustedActivation",
        ])
        fallback = self.browser.split(
            "private func beginInitialUserActivatedChildFallback(", 1
        )[1].split(
            "private func isUnregisteredInitialTargetDocumentToken", 1
        )[0]
        self.assert_contracts(fallback, [
            '!isChallengeNavigation',
            'targetFrame.request.url?.absoluteString.lowercased() == "about:blank"',
            "acceptingScriptCandidates",
            "authorizedRouteURL == webView.url?.absoluteString",
            "maximumInitialUserActivatedChildNavigationCount",
            "__miohInteractiveConsumeTrustedActivation?.() === true",
            'proof["activated"] as? Bool == true',
            "makeInitialUserActivatedChildAuthorization(",
            "pendingSubframeAuthorizations.append(authorization)",
        ])
        initial = self.browser.split(
            "private func makeInitialUserActivatedChildAuthorization(", 1
        )[1].split("private func makeSubframeNavigationAuthorization(", 1)[0]
        self.assert_contracts(initial, [
            "isInitialUserActivatedFallback: true",
            "initiatorDocumentEpoch: initiatorEpoch",
            "authority = .inspection(routeToken: routeToken)",
            "chain.remainingHopCount -= 1",
            "pendingSubframeAuthorizations.count < Self.maximumKnownFrameCount",
            "!hasOutstandingSubframeAuthorization(for: destinationURL)",
        ])
        self.assert_contracts(fallback, [
            "resolveInitialTargetDocumentToken(",
            "remainingAttempts: 8",
            "50_000_000",
            "targetPriorDocumentToken: resolvedTargetToken",
        ])
        validation = self.browser.split(
            "private func authorizationInitiatorIsCurrent(", 1
        )[1].split("private static func sameAuthority", 1)[0]
        self.assert_contracts(validation, [
            "if authorization.isInitialUserActivatedFallback",
            "authorizedRouteURL == webView.url?.absoluteString",
            "case .inspection(let authorityRouteToken)",
            "case .challenge(let chainID)",
            "activeSubframeChallengeChainIDs.count == 1",
            "activeSubframeChallengeChainIDs.contains(chainID)",
            "mainDocumentToken == authorization.initiatorDocumentToken",
            "subframeDocumentEpochs[authorization.initiatorDocumentToken]",
        ])
        initial_user_validation = validation.split(
            "if authorization.isInitialUserActivatedFallback", 1
        )[1].split("if let nativeGeneration", 1)[0]
        before_authority_switch, authority_switch = initial_user_validation.split(
            "switch authorization.chain.authority", 1
        )
        inspection_branch, challenge_branch = authority_switch.split(
            "case .challenge(let chainID)", 1
        )
        challenge_branch = challenge_branch.split("case .pageLoad", 1)[0]
        self.assertNotIn("acceptingScriptCandidates", before_authority_switch)
        self.assertIn("acceptingScriptCandidates", inspection_branch)
        self.assertNotIn("acceptingScriptCandidates", challenge_branch)

    def test_user_activated_player_redirect_chain_is_target_bound(self):
        authorization_model = self.browser.split(
            "private struct SubframeNavigationAuthorization", 1
        )[1].split("private static let messageHandlerName", 1)[0]
        self.assert_contracts(authorization_model, [
            "var destinationURL: String",
            "initialUserNavigationTypeRawValue: Int?",
            "initialUserRedirectDeadline: Date?",
            "var visitedDestinationURLs: Set<String>",
        ])

        action_authorization = self.browser.split(
            "private func authorizeSubframeNavigationAction(", 1
        )[1].split("private func beginInitialChildChallengeFallback(", 1)[0]
        self.assertLess(
            action_authorization.index("continuePendingInitialUserChildRedirect("),
            action_authorization.index(
                "if self.isUnregisteredInitialTargetDocumentToken(documentToken)"
            ),
        )
        self.assert_contracts(action_authorization, [
            "navigationTypeRawValue: navigationTypeRawValue",
            "allowsFallback: allowsInitialChildChallengeFallback",
            "webView: authorizingWebView",
        ])

        redirect = self.browser.split(
            "private func continuePendingInitialUserChildRedirect(", 1
        )[1].split("private func isUnregisteredInitialTargetDocumentToken", 1)[0]
        self.assert_contracts(redirect, [
            "$0.targetPriorDocumentToken == targetDocumentToken",
            "matches.count == 1",
            "sourceDocumentToken == targetDocumentToken",
            "authorization.initialUserNavigationTypeRawValue == navigationTypeRawValue",
            "navigationTypeRawValue == WKNavigationType.other.rawValue",
            "Date() <= deadline",
            "authorization.chain.remainingHopCount > 0",
            "!authorization.visitedDestinationURLs.contains",
            "authorizationInitiatorIsCurrent(authorization)",
            "authorization.chain.remainingHopCount -= 1",
            "authorization.destinationURL = destinationURL.absoluteString",
            "authorization.visitedDestinationURLs.insert",
        ])
        self.assert_contracts(self.browser, [
            "private func pruneExpiredInitialUserAuthorizations",
            "deadline < now",
            "revokeSubframeAuthorization(authorization)",
        ])

        consume = self.browser.split(
            "private func consumeSubframeAuthorization(", 1
        )[1].split("private func hasOutstandingSubframeAuthorization", 1)[0]
        self.assert_contracts(consume, [
            "let matchingIndices",
            "matchingIndices.count == 1",
            "pendingSubframeAuthorizations.remove(at: index)",
        ])

    def test_popup_window_keeps_the_visible_page_and_original_request(self):
        action = self.navigation_delegate.split(
            "decidePolicyFor navigationAction: WKNavigationAction", 1
        )[1].split("decidePolicyFor navigationResponse: WKNavigationResponse", 1)[0]
        popup_branch = action.split("if navigationAction.targetFrame == nil", 1)[1].split(
            "if isClosingPage", 1
        )[0]
        self.assert_contracts(popup_branch, [
            "openingPageWebView == nil",
            "Self.isAllowedTransientPopupURL(navigationAction.request.url)",
            "decisionHandler(.allow)",
        ])
        self.assertNotIn("prepareForMainNavigation", popup_branch)
        self.assertNotIn("isLoading = true", popup_branch)

        ui_delegate = self.browser.split(
            "extension IPadInteractiveMediaBrowser: WKUIDelegate", 1
        )[1]
        self.assert_contracts(ui_delegate, [
            "navigationAction.targetFrame == nil",
            "let popupFrame =",
            "webView.bounds.isEmpty",
            "CGRect(x: 0, y: 0, width: 1_024, height: 768)",
            "WKWebView(frame: popupFrame, configuration: configuration)",
            "IPadTransientPopupCoordinator(",
            "maximumTransientPopupNavigationCount",
            "maximumTransientPopupCreationCount",
            "transientPopupCreationCount += 1",
            "readyHandler:",
            "markTransientPopupReady(popupWebView)",
            "retainTransientPopup(popupWebView, coordinator: coordinator)",
            "return popupWebView",
        ])
        self.assertNotIn("navigate(safeURL.absoluteString)", ui_delegate)
        self.assert_contracts(self.browser, [
            'url.absoluteString.lowercased() == "about:blank"',
            "transientPopupLifetimeNanoseconds",
            "retireTransientPopup()",
            "remainingNavigationCount -= 1",
            "func showOpenedPage()",
            "func returnToOpeningPage()",
            "private func adoptSettledWebView(",
            "openingPageWebView = opener",
            "retainedPopupOpenerWebViews.append(opener)",
            "retireTransientPopup()",
            "scheduleTransientPopupRetirement(for: popupWebView)",
            "webView = settledWebView",
            "webViewGeneration = nextGeneration(after: webViewGeneration)",
        ])
        self.assert_contracts(self.view, [
            "interactiveBrowser.hasOpenedPage",
            'Label("開いたページ", systemImage: "rectangle.on.rectangle")',
            "interactiveBrowser.showOpenedPage()",
            "interactiveBrowser.canReturnToOpeningPage",
            'Label("元のページ", systemImage: "arrowshape.turn.up.backward")',
            "interactiveBrowser.returnToOpeningPage()",
        ])

        lifecycle = self.navigation_delegate.split(
            "func webView(_ webView: WKWebView, didStartProvisionalNavigation", 1
        )[1]
        self.assertGreaterEqual(
            lifecycle.count("webView === self.webView"),
            7,
        )

    def test_user_activated_new_window_links_keep_native_navigation_history(self):
        ui_delegate = self.browser.split(
            "extension IPadInteractiveMediaBrowser: WKUIDelegate", 1
        )[1]
        same_window_link = ui_delegate.split("let opensNewWindow", 1)[1].split(
            "if Self.relaxedWebCompatibilityEnabled", 1
        )[0]

        self.assert_contracts(
            same_window_link,
            [
                "navigationAction.targetFrame == nil",
                "navigationAction.sourceFrame.isMainFrame",
                "navigationAction.navigationType == .linkActivated",
                "!navigationAction.shouldPerformDownload",
                '(navigationAction.request.httpMethod ?? "GET").uppercased() == "GET"',
                "Self.sanitizedPublicHTTPSURL(",
                "!Self.isHighConfidenceAdvertisementNavigationURL(destinationURL)",
                "prepareForMainNavigation()",
                "isLoading = true",
                "webView.load(navigationAction.request)",
                "activeNavigation = navigation",
                "return nil",
            ],
        )
        self.assertNotIn("showOpenedPage()", same_window_link)

        same_document = self.browser.split(
            "private func acceptSameDocumentRouteInPlace()", 1
        )[1].split("private func resetSameDocumentNavigationState()", 1)[0]
        self.assertIn("updateNavigationState()", same_document)

        attach = self.browser.split("private func attachDelegates", 1)[1].split(
            "private static func isAllowedTransientPopupURL", 1
        )[0]
        self.assert_contracts(
            attach,
            [
                "canGoBackObservation?.invalidate()",
                "canGoForwardObservation?.invalidate()",
                "webView.observe(",
                "\\.canGoBack",
                "\\.canGoForward",
                "self.webView === webView",
                "self.updateNavigationState()",
            ],
        )

    def test_same_document_player_handoff_keeps_inflight_frame_authority(self):
        route_change = self.browser.split(
            "private func processSameDocumentPageLocation", 1
        )[1].split("private func resetSameDocumentNavigationState", 1)[0]
        self.assert_contracts(route_change, [
            "acceptSameDocumentRouteInPlace()",
            "let currentPageURL = webView.url",
            "let safeURL = Self.sanitizedPublicHTTPSURL(currentPageURL)",
            "addressText = safeURL.absoluteString",
            "authorizedRouteURL = currentPageURL.absoluteString",
            "activateInspectionInKnownFrames()",
        ])
        self.assertNotIn(
            "navigationGeneration = nextGeneration(after: navigationGeneration)",
            route_change,
        )
        self.assertNotIn("pendingSubframeAuthorizations.removeAll()", route_change)
        self.assertNotIn("completedSubframeAuthorizations.removeAll()", route_change)
        self.assertNotIn("subframeNavigationChains.removeAll()", route_change)
        self.assertNotIn("clearMediaSourceState()", route_change)

    def test_initial_child_authority_is_revalidated_at_response_and_commit(self):
        self.assert_contracts(self.browser, [
            "private func authorizationInitiatorIsCurrent(",
            "guard authorizationInitiatorIsCurrent(authorization)",
            "mainFrameChallengeResponseGeneration == nativeGeneration",
            "Self.sameAuthority(currentChain.authority, authorization.chain.authority)",
            "targetPriorDocumentToken",
        ])
        response = self.navigation_delegate.split(
            "decidePolicyFor navigationResponse: WKNavigationResponse", 1
        )[1].split("didStartProvisionalNavigation", 1)[0]
        self.assertLess(
            response.index("guard authorizationInitiatorIsCurrent(authorization)"),
            response.index("completedSubframeAuthorizations.append"),
        )
        verification = self.browser.split("private func verifyFrameReady(", 1)[1].split(
            "private func authorizeSubframeNavigationAction(", 1
        )[0]
        self.assertLess(
            verification.index("authorizationInitiatorIsCurrent(authorization)"),
            verification.index("completedSubframeAuthorizations.remove(at:"),
        )

    def test_only_bounded_cloudflare_local_subframes_are_allowed(self):
        action = self.navigation_delegate.split(
            "decidePolicyFor navigationAction: WKNavigationAction", 1
        )[1].split("decidePolicyFor navigationResponse: WKNavigationResponse", 1)[0]
        self.assert_contracts(action, [
            "Self.isChallengeLocalFrameURL(navigationAction.request.url)",
            "navigationAction.targetFrame?.isMainFrame == false",
            'navigationAction.request.httpMethod ?? "GET"',
            "hasCurrentNativeMainChallenge",
            "sourceIsCurrentMain || sourceIsEligibleChild",
            "challengeLocalFrameNavigationCount",
            "Self.maximumChallengeLocalFrameNavigationCount",
            "challengeLocalFrameNavigationCount += 1",
            "sanitizedPublicHTTPSURL(navigationAction.request.url)",
        ])
        local_helper = self.browser.split(
            "private static func isChallengeLocalFrameURL", 1
        )[1].split("private static func isDirectMediaCandidate", 1)[0]
        self.assert_contracts(local_helper, [
            'value == "about:blank"',
            'value == "about:srcdoc"',
        ])
        self.assertNotIn('scheme?.lowercased() == "blob"', action)
        self.assertNotIn('value == "blob:', local_helper)

    def test_native_cloudflare_frames_use_a_bounded_transparent_webkit_path(self):
        action = self.navigation_delegate.split(
            "decidePolicyFor navigationAction: WKNavigationAction", 1
        )[1].split("decidePolicyFor navigationResponse: WKNavigationResponse", 1)[0]
        self.assert_contracts(action, [
            "navigationAction.targetFrame?.isMainFrame == false",
            "let safeChallengeURL = Self.sanitizedPublicHTTPSURL(",
            "isAllowedNativeChallengeFrameURL(safeChallengeURL)",
            "isEligibleNativeChallengeSource(navigationAction.sourceFrame)",
            '["GET", "POST"].contains(',
            "nativeChallengeFrameNavigationCount",
            "Self.maximumNativeChallengeFrameNavigationCount",
            "nativeChallengeFrameNavigationCount += 1",
        ])
        helper = self.browser.split(
            "private func isCurrentMainOrCloudflareChallengeURL", 1
        )[1].split("private static func isDirectMediaCandidate", 1)[0]
        self.assert_contracts(helper, [
            'host == "challenges.cloudflare.com"',
            'host.hasSuffix(".challenges.cloudflare.com")',
            "Self.isSameOrigin(url, $0)",
            "private func isAllowedNativeChallengeFrameURL",
            "hasCurrentNativeMainChallenge",
            "private func isContextualChallengeNavigationURL",
            "private func isEligibleNativeChallengeSource",
            "sourceFrame.isMainFrame",
            "committedMainDocumentURL?.absoluteString",
        ])
        transparent_destination = helper.split(
            "private func isContextualChallengeNavigationURL", 1
        )[0]
        self.assertNotIn('/cdn-cgi/challenge-platform/', transparent_destination)
        response = self.navigation_delegate.split(
            "decidePolicyFor navigationResponse: WKNavigationResponse", 1
        )[1].split("didStartProvisionalNavigation", 1)[0]
        self.assert_contracts(response, [
            "let hasChallengeResponseHeader",
            "if navigationResponse.isForMainFrame",
            "hasChallengeResponseHeader && isCurrentMainOrCloudflareChallengeURL($0)",
            "isContextualChallengeNavigationURL($0)",
        ])
        transparent_response = response.split(
            "isAllowedNativeChallengeFrameURL(safeResponseURL)", 1
        )[1].split("var subframeAuthorization", 1)[0]
        self.assert_contracts(transparent_response, [
            "navigationResponse.canShowMIMEType",
            'disposition?.contains("attachment") != true',
            "decisionHandler(.allow)",
            "return",
        ])
        self.assertNotIn("insertCandidate", transparent_response)

    def test_stalled_challenge_stops_spinning_and_offers_safari_for_viewing(self):
        self.assert_contracts(self.browser, [
            "challengeCompatibilityTimedOut",
            "challengeCompatibilityTask",
            "beginNativeChallengeCompatibilityWindow()",
            "15_000_000_000",
            "self.hasCurrentNativeMainChallenge",
            "このサイトはアプリ内ブラウザ非対応の可能性があります",
            "self.nativeChallengeFrameNavigationCount",
            "self.challengeLocalFrameNavigationCount",
            "private func setChallengeWaitingStatus()",
            "var currentPublicPageURL: URL?",
        ])
        self.assert_contracts(self.view, [
            "@Environment(\\.openURL) private var openURL",
            "interactiveBrowser.challengeCompatibilityTimedOut",
            "interactiveBrowser.currentPublicPageURL",
            'Label("Safariで開く（閲覧）", systemImage: "safari")',
            "openURL(pageURL)",
            "Safariの確認情報はmiohの解析には引き継がれません",
            "if interactiveBrowser.challengeCompatibilityTimedOut",
        ])

    def test_browser_navigation_dismisses_the_address_keyboard(self):
        self.assert_contracts(self.view, [
            "@FocusState private var browserAddressFocused: Bool",
            ".focused($browserAddressFocused)",
            "private func openBrowserAddress()",
            "browserAddressFocused = false",
        ])

    def test_stop_invalidates_pending_frame_proofs_without_forgetting_documents(self):
        stop = self.browser.split("func stop()", 1)[1].split("func closePage()", 1)[0]
        self.assert_contracts(stop, [
            "navigationGeneration = nextGeneration(after: navigationGeneration)",
            "pendingSubframeAuthorizations.removeAll()",
            "completedSubframeAuthorizations.removeAll()",
            "initialChildChallengeDecisionID = nil",
            "initialChildChallengeFallbackGeneration = nil",
            'statusMessage = "確認を停止しました。再読み込みしてください。"',
        ])
        self.assertNotIn("subframeDocumentEpochs.removeAll()", stop)
        self.assertNotIn("knownFrames.removeAll()", stop)

    def test_unverified_frame_ready_does_not_consume_epoch_capacity(self):
        verification = self.browser.split("private func verifyFrameReady(", 1)[1].split(
            "private func authorizeSubframeNavigationAction(", 1
        )[0]
        before_js_proof = verification.split("webView.callAsyncJavaScript(", 1)[0]
        self.assertIn(
            "documentEpoch = subframeDocumentEpochs[documentToken]",
            before_js_proof,
        )
        self.assertNotIn(
            "subframeDocumentEpochs[documentToken] =",
            before_js_proof,
        )
        self.assert_contracts(verification, [
            "guard currentDocumentEpoch == nil else { return }",
            "case .success(let value) = result",
            "value as? String == documentToken",
            "documentEpoch != nil",
            "self.subframeDocumentEpochs.count < Self.maximumKnownFrameCount",
            "if documentEpoch == nil",
            "self.subframeDocumentEpochs[documentToken] = verifiedDocumentEpoch",
        ])

    def test_subframe_destination_authorization_is_unique_until_commit(self):
        authorization = self.browser.split(
            "private func makeSubframeNavigationAuthorization(", 1
        )[1].split("private func consumeSubframeAuthorization(", 1)[0]
        self.assertIn(
            "!hasOutstandingSubframeAuthorization(for: destinationURL)",
            authorization,
        )
        self.assertLess(
            authorization.index("!hasOutstandingSubframeAuthorization"),
            authorization.index("knownFrames.removeValue"),
        )
        outstanding = self.browser.split(
            "private func hasOutstandingSubframeAuthorization(", 1
        )[1].split("private func promoteSubframeAuthorizationToChallenge(", 1)[0]
        self.assert_contracts(outstanding, [
            "pendingSubframeAuthorizations.contains",
            "completedSubframeAuthorizations.contains",
            "$0.destinationURL == destination",
            "$0.chain.mainNavigationGeneration == navigationGeneration",
        ])

    def test_subframe_challenge_has_a_bounded_verified_successor_chain(self):
        self.assert_contracts(self.browser, [
            "case challenge(chainID: String)",
            "maximumSubframeNavigationHopCount = 8",
            "remainingHopCount",
            "mainNavigationGeneration",
            "activeSubframeChallengeChainIDs",
            "promoteSubframeAuthorizationToChallenge(",
            "authorization.retiresSubframeChallengeOnCommit = true",
            "activeSubframeChallengeChainIDs.remove(chainID)",
            "completeChallengeIfPossible()",
            "guard acceptingScriptCandidates,",
            'if messageType == "media-source"',
            'guard messageType == "candidate"',
        ])

    def test_distinct_native_subframe_challenges_are_serialized(self):
        authorization = self.browser.split(
            "private func makeSubframeNavigationAuthorization(", 1
        )[1].split("private func consumeSubframeAuthorization(", 1)[0]
        promotion = self.browser.split(
            "private func promoteSubframeAuthorizationToChallenge(", 1
        )[1].split("private func revokeSubframeAuthorization(", 1)[0]
        for source in (authorization, promotion):
            self.assert_contracts(source, [
                "activeSubframeChallengeChainIDs.count == 1",
                "activeSubframeChallengeChainIDs.contains(chainID)",
                "guard activeSubframeChallengeChainIDs.isEmpty else { return nil }",
            ])
        response = self.navigation_delegate.split(
            "decidePolicyFor navigationResponse: WKNavigationResponse", 1
        )[1].split("didStartProvisionalNavigation", 1)[0]
        self.assert_contracts(response, [
            "let promotedAuthorization = promoteSubframeAuthorizationToChallenge(",
            "revokeSubframeAuthorization(authorization)",
            "decisionHandler(.cancel)",
        ])

    def test_main_script_challenge_requires_positive_completion_evidence(self):
        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        self.assert_contracts(passive, [
            "const acknowledgedChallengeTokens = new Set();",
            "document.querySelectorAll(",
            "const challengeResponseValues = Array.from(",
            "const nonemptyChallengeTokens = challengeResponseValues.filter(Boolean);",
            "const allResponseFieldsComplete = challengeResponseValues.length > 0",
            "challengeResponseValues.every(value => value.length > 0);",
            "const present = hardChallenge || (embeddedChallenge && !allResponseFieldsComplete);",
            "if (present && !challengeVisible)",
            "activeChallengeEpoch = `${documentToken}:${challengeEpochCounter.toString(36)}`;",
            "for (const token of nonemptyChallengeTokens)",
            "if (present && activeChallengeEpoch)",
            "post({type: 'challenge-hint', challengeEpoch: activeChallengeEpoch});",
            "if (activeChallengeEpoch && !present && allResponseFieldsComplete)",
            "const freshChallengeTokens = new Set(",
            "token => !acknowledgedChallengeTokens.has(token)",
            "if (freshChallengeTokens.size > 0)",
            "const completedEpoch = activeChallengeEpoch;",
            "activeChallengeEpoch = null;",
            "challengeVisible = false;",
            "challengeEpoch: completedEpoch",
        ])
        self.assertEqual(passive.count("'challenge-cleared'"), 1)
        self.assertNotIn("post({type: present ?", passive)
        self.assertNotIn("?.value || ''", passive)
        visible_transition = passive.split("if (present && !challengeVisible)", 1)[1].split(
            "if (activeChallengeEpoch && !present && allResponseFieldsComplete)", 1
        )[0]
        self.assertLess(
            visible_transition.index("acknowledgedChallengeTokens.add"),
            visible_transition.index("if (present && activeChallengeEpoch)"),
        )
        epoch_retransmission = visible_transition.split(
            "if (present && activeChallengeEpoch)", 1
        )[1]
        self.assertIn(
            "post({type: 'challenge-hint', challengeEpoch: activeChallengeEpoch});",
            epoch_retransmission,
        )
        epoch_creation = visible_transition.split(
            "if (present && activeChallengeEpoch)", 1
        )[0]
        self.assertNotIn("post({type: 'challenge-hint'", epoch_creation)
        response_values = passive.split(
            "const challengeResponseValues = Array.from(", 1
        )[1].split("const nonemptyChallengeTokens", 1)[0]
        self.assertNotIn("filter(Boolean)", response_values)
        completion = passive.split(
            "if (activeChallengeEpoch && !present && allResponseFieldsComplete)", 1
        )[1].split("if (!present)", 1)[0]
        self.assertIn("challengeVisible = false;", completion)
        authorize_route = passive.split("const authorizeRoute", 1)[1].split(
            "Object.defineProperty(window, '__miohInteractiveAuthorizeRoute'", 1
        )[0]
        self.assertNotIn("challengeVisible = false", authorize_route)
        receive = self.browser.split("private func receiveScriptMessage(", 1)[1].split(
            "private func verifyFrameReady(", 1
        )[0]
        self.assert_contracts(receive, [
            'messageType == "challenge-hint", message.frameInfo.isMainFrame',
            "mainFrameScriptChallengeActive = true",
            'messageType == "challenge-cleared", message.frameInfo.isMainFrame,',
            'body["completed"] as? Bool == true',
            'let challengeEpoch = body["challengeEpoch"] as? String',
            "challengeEpoch == pendingMainScriptChallengeEpoch",
            "pendingMainScriptChallengeEpoch = nil",
            "completeChallengeIfPossible()",
        ])
        self.assertNotIn("activeSubframeChallengeChainIDs.removeAll()", receive)
        verification = self.browser.split("private func verifyFrameReady(", 1)[1].split(
            "private func authorizeSubframeNavigationAction(", 1
        )[0]
        self.assertNotIn("mainFrameScriptChallengeActive = false", verification)
        action_authorization = self.browser.split(
            "private func authorizeSubframeNavigationAction(", 1
        )[1].split("private func makeSubframeNavigationAuthorization(", 1)[0]
        self.assert_contracts(action_authorization, [
            "if case .challenge = authorization.chain.authority",
            "self.challengeActive = true",
        ])
        self.assertNotIn(
            "self.mainFrameScriptChallengeActive = true", action_authorization
        )
        response_delegate = self.navigation_delegate.split(
            "decidePolicyFor navigationResponse: WKNavigationResponse", 1
        )[1].split("didStartProvisionalNavigation", 1)[0]
        self.assert_contracts(response_delegate, [
            "if isChallenge",
            "challengeActive = true",
        ])
        self.assertNotIn(
            "mainFrameScriptChallengeActive = true", response_delegate
        )

    def test_web_content_process_termination_reloads_retained_browser(self):
        termination = self.browser.split(
            "func webViewWebContentProcessDidTerminate(_ webView: WKWebView)", 1
        )[1].split("func webView(\n    _ webView: WKWebView,", 1)[0]
        self.assert_contracts(termination, [
            "guard webView === self.webView, !isClosingPage else { return }",
            "webContentProcessTerminated = true",
            'statusMessage = "ブラウザ処理を再開しています…"',
            "self.resumeAfterBackground()",
        ])
        self.assertNotIn("closePage()", termination)
        resume = self.browser.split("func resumeAfterBackground()", 1)[1].split(
            "/// Closes the visible page", 1
        )[0]
        self.assertIn("reload()", resume)

    def test_additional_candidate_challenge_preserves_visible_browser(self):
        self.assert_contracts(self.store, [
            "@Published private(set) var urlInteractionURL: URL?",
            "catch IPadMediaURLResolverError.interactionRequired",
            "candidate.interactionPageURL",
            "candidate.requestContext.referer",
            "urlInteractionURL = urlInteractionURL ?? targetURL",
            "selectionOwnerID: UUID?",
            "clearResolvedURLInput(ownedBy selectionOwnerID: UUID)",
        ])
        self.assert_contracts(self.view, [
            "store.urlInputRequiresInteraction",
            "store.urlInteractionURL != nil",
            "never reload or replace the visible page from this result",
            "表示中のページはそのまま維持しています",
            "store.clearResolvedURLInput(ownedBy: selectionOwnerID)",
            "store.urlInteractionURL?.absoluteString ?? rawValue",
        ])

    def test_browser_candidates_reenter_the_hardened_resolver(self):
        self.assert_contracts(self.store, [
            "func selectBrowserCandidates(_ candidates: [IPadWebMediaCandidate])",
            "if Self.isInteractionRequired(error) { throw error }",
            "let primaryCandidates = candidates.filter",
            "let supersededFallbackCandidates = candidates.filter",
            "for candidate in pool",
            "guard remainingAttempts > 0 else { break }",
            "catch IPadMediaURLResolverError.unsafeInitialURL",
            "remainingAttempts += 1",
            "let policy = browserResolutionPolicy(for: candidate)",
            "policy: policy",
            "context: candidate.requestContext",
            "acceptResolvedURLSource(",
            "activeURLResolutionID",
            "checkURLResolution(operationID)",
        ])

    def test_vpn_fake_ip_exception_requires_visible_active_playback(self):
        self.assert_contracts(self.store, [
            "private nonisolated static func browserResolutionPolicy(",
            "candidate.selectionState == .activeCurrentSource",
            "let evidence = candidate.mediaEvidence",
            "evidence.isPlaying",
            "evidence.isVisible",
            "evidence.visibilityAttested",
            "evidence.renderedArea >= 4_096",
            "else { return .publicDiscovered }",
            "let approvedOrigin = browserOriginURL(for: candidate.url)",
            "return .visibleBrowserDiscovered(approvedOrigin)",
        ])

    def test_media_visibility_uses_browser_attested_intersection(self):
        self.assert_contracts(self.browser, [
            "const mediaIntersectionStates = new WeakMap();",
            "new IntersectionObserver(",
            "trackVisibility: true",
            "intersectionVisualVisibilityEnabled = true",
            "const rect = entry.intersectionRect;",
            "mediaIntersectionObserver?.observe(element)",
            "mediaIntersectionObserver?.unobserve(element)",
            "style.contentVisibility === 'hidden'",
            "accumulatedOpacity < 0.05",
            "const visualProof = intersectionVisualVisibilityEnabled",
            "intersection?.isVisuallyVisible === true",
            "visibilityAttested = Boolean(intersection && visualProof)",
            "visibilityAttested,",
        ])
        self.assert_contracts(self.view, [
            "await interactiveBrowser.snapshotCandidates()",
            "accepted = await store.selectBrowserCandidates(",
            "selectionOwnerID: selectionOwnerID",
            "if store.urlInputRequiresInteraction",
            "openBrowserForInteraction(",
            "interactiveBrowser.closePage()",
            "selectedTab = .playback",
            "tryAutoStartURLPlayback()",
        ])

    def test_hls_relay_geometry_does_not_relax_attested_network_policy(self):
        self.assert_contracts(self.browser, [
            "let isGeometricallyVisible: Bool",
            'body["isGeometricallyVisible"] as? Bool',
            "let isGeometricallyVisible = false;",
            "isGeometricallyVisible = Boolean(intersection && localTreeVisible",
            "intersection.isIntersecting === true && renderedArea >= 4",
            "visibilityAttested = Boolean(intersection && visualProof)",
            "isVisible = visibilityAttested && isGeometricallyVisible",
            "isGeometricallyVisible,",
        ])
        association = self.browser.split(
            "private func opaquePlaybackAssociations()", 1
        )[1].split("private static func isOpaquePlaybackHLSResponse", 1)[0]
        self.assert_contracts(association, [
            "$0.state.hasOpaqueSource",
            "$0.state.isPlaying",
            "$0.state.isGeometricallyVisible",
            "$0.state.renderedArea >= 4_096",
            "!$0.state.isCompactFloatingOverlay",
            "candidate.documentToken == documentToken",
        ])
        self.assertNotIn("$0.state.visibilityAttested", association)

        native = self.browser.split(
            "private func nativePlaybackHLSRelayEvidence(", 1
        )[1].split("private static func isPageNetworkCredentialObservation", 1)[0]
        self.assert_contracts(native, [
            "isMediaDocumentCurrent(documentToken)",
            "state.isPlaying, !state.isEnded",
            "state.isGeometricallyVisible, state.renderedArea >= 4_096",
            "!state.isCompactFloatingOverlay",
            "now.timeIntervalSince(state.lastObservedAt) <= 5",
            "relayIncludesCredentials: false",
        ])
        # V2 unavailable may authorize only the exact relay transport. Hidden,
        # paused, compact and cross-document states remain ineligible.
        relay_eligible = lambda current, playing, geometric, area, compact: (
            current and playing and geometric and area >= 4096 and not compact
        )
        self.assertTrue(relay_eligible(True, True, True, 4096, False))
        self.assertFalse(relay_eligible(False, True, True, 4096, False))
        self.assertFalse(relay_eligible(True, False, True, 4096, False))
        self.assertFalse(relay_eligible(True, True, False, 4096, False))
        self.assertFalse(relay_eligible(True, True, True, 4096, True))

        credential_authorization = self.browser.split(
            "let credentialAuthorizedURLKeys", 1
        )[1].split("let supersededURLKeys", 1)[0]
        self.assert_contracts(credential_authorization, [
            "state.isPlaying, state.isVisible, state.visibilityAttested",
            "guard state.isVisible, state.visibilityAttested",
        ])
        resolver_policy = self.store.split(
            "private nonisolated static func browserResolutionPolicy(", 1
        )[1].split("private nonisolated static func browserOriginURL", 1)[0]
        self.assert_contracts(resolver_policy, [
            "evidence.isVisible",
            "evidence.visibilityAttested",
            "evidence.renderedArea >= 4_096",
        ])

    def test_video_source_generations_retire_pre_roll_candidates(self):
        self.assert_contracts(self.browser, [
            "struct MediaSlotKey: Hashable",
            "struct MediaSlotState",
            "mediaSourceRevision",
            "candidateRevision",
            "currentSourceHistory",
            "sourceGeneration",
            "slotToken",
            "stateName == \"current\" || stateName == \"cleared\"",
            "generation < existing.generation",
            "existing.currentURL != currentURL",
            "existing?.isEnded != isEnded",
            "activeURLKeys",
            "supersededURLKeys",
            'sourceKind: "active-current-source"',
            "keepNativeProvenance",
            ".activeCurrentSource",
            ".supersededCurrentSource",
            "mediaEvidence: mediaEvidenceByURL",
            "observedDuration: includesObservedDuration ? state.duration : nil",
        ])
        passive = self.browser.split(
            "private static let passiveInstrumentationScript", 1
        )[1].split("private static let pageNetworkObservationScript", 1)[0]
        self.assert_contracts(passive, [
            "const mediaSlotStates = new WeakMap();",
            "const maximumMediaSlots = 32;",
            "const availableMediaSlotTokens = [];",
            "mediaSlotStates.delete(element);",
            "const reportMediaSource =",
            "type: 'media-source'",
            "sourceGeneration: slot.generation",
            "reportMediaSource(element, true, true)",
            "String(element.currentSrc || element.src || '')",
            "normalizedHTTPSURL(rawCurrentSource)",
            "const isEnded = element.ended === true;",
            "isEnded,",
        ])
        self.assertNotIn("inspectValue(element.currentSrc, 'currentSrc')", passive)
        self.assert_contracts(self.view, [
            "interactiveBrowser.likelyPreRollWait",
            "interactiveBrowser.candidateRevision",
            "本編URLの追加発行を待っています",
            "interactiveBrowser.isPreRollWaitCurrent(preRollWait)",
            "短い先行動画の終了と本編への切り替えを待っています",
            "let sourceRevision = interactiveBrowser.mediaSourceRevision",
            "interactiveBrowser.mediaSourceRevision == sourceRevision",
            "本編への切り替えを検出したため、配信を再解析しています",
        ])

    def test_browser_analysis_keeps_watching_and_reranks_late_hls(self):
        analysis = self.view.split(
            "private func analyzeBrowserCandidates()", 1
        )[1].split("private func startRealtimePreview()", 1)[0]
        self.assert_contracts(analysis, [
            "while !Task.isCancelled,",
            "lastAttemptedSourceRevision",
            "lastAttemptedCandidateRevision",
            "let hasNewEvidence =",
            "let periodicRetryDue =",
            "Date().timeIntervalSince(lastAttemptDate) >= 12",
            "interactiveBrowser.activateInspection()",
            "await interactiveBrowser.snapshotCandidates()",
            "await store.selectBrowserCandidates(",
            "try await Task.sleep(nanoseconds: 500_000_000)",
            "本編HLSを監視中です",
            "自動で解析します",
        ])
        self.assertNotIn("for attempt in 0..<5", analysis)
        empty_candidates = analysis.split(
            "guard !candidates.isEmpty else", 1
        )[1].split("let accepted =", 1)[0]
        self.assertNotIn("autoStartURLPlayback = false", empty_candidates)
        self.assertIn("continue", empty_candidates)

        loading_change = self.view.split(
            ".onChange(of: interactiveBrowser.isLoading)", 1
        )[1].split(
            ".onChange(of: interactiveBrowser.successfulPageVisit)", 1
        )[0]
        self.assert_contracts(loading_change, [
            "if isLoading",
            "autoStartURLPlayback || browserAnalysisTask != nil",
            "pendingBrowserAnalysisResume = true",
            "invalidateBrowserAnalysis(preservingResumeIntent: true)",
            "resumePendingBrowserAnalysisIfReady()",
        ])

    def test_compact_lower_right_media_is_exclusion_only(self):
        self.assert_contracts(self.browser, [
            "let isCompactFloatingOverlay: Bool",
            "floatingAdvertisementURLKeys",
            "const isCompactFloatingMediaOverlay = element =>",
            "position === 'fixed' || position === 'sticky'",
            "isCompactLowerRightOverlay(cursor)",
            "isCompactFloatingOverlay: isCompactFloatingOverlay === true",
            '(body["isCompactFloatingOverlay"] as? Bool) == true',
            'body["isCompactFloatingOverlay"] as? Bool',
            "reportedFloatingAdvertisementURLKeys.insert",
            "candidates.removeValue(forKey: key)",
            "excludedURLKeys: floatingAdvertisementURLKeys",
            "!excludedURLKeys.contains($0.url.absoluteString)",
        ])
        receive = self.browser.split(
            "private func receiveMediaSourceMessage", 1
        )[1].split("private func verifyFrameReady", 1)[0]
        self.assertNotIn("allowsCredentialReplay", receive)

    def test_mse_blob_evidence_only_upgrades_recent_same_document_hls(self):
        self.assert_contracts(self.browser, [
            "let documentToken: String?",
            "let observedAt: Date",
            "let hasOpaqueSource: Bool",
            "sourceIdentity: null",
            "const opaqueSourceToken =",
            "rawCurrentSource.startsWith('blob:')",
            "const sourceIdentity = currentURL || opaqueSourceToken;",
            "slot.generation += 1;",
            "opaqueSourceToken ? 'opaque' : 'cleared'",
            'stateName == "opaque"',
            "private func opaquePlaybackAssociations()",
            "state.hasOpaqueSource",
            "now.timeIntervalSince(state.lastObservedAt) <= 5",
            "candidate.documentToken == documentToken",
            "candidate.observedAt >= earliestObservation",
            "candidate.observedAt <= latestObservation",
            "private static func isOpaquePlaybackHLSResponse",
            'case "page-fetch-hls-response", "page-xhr-hls-response",',
            "activeURLKeys.formUnion(",
            ").union(",
            "Dictionary(grouping: eligibleSlots, by: \\.documentToken)",
            ".prefix(8)",
            "slots.count == 1 && matches.count == 1",
            "!compactDocuments.contains(documentToken)",
            "includesObservedDuration: association.includesObservedDuration",
            "mediaEvidence: mediaEvidenceByURL",
            "observedDuration: includesObservedDuration ? state.duration : nil",
        ])
        association = self.browser.split(
            "private func opaquePlaybackAssociations()", 1
        )[1].split("func snapshotCandidates", 1)[0]
        self.assertNotIn("fallbackFrameURL", association)
        self.assertNotIn("isSameOrigin", association)
        self.assertNotIn("navigation-response", association)

    def test_page_hls_observation_survives_multi_stage_players(self):
        self.assert_contracts(self.browser, [
            "const maximumPageNetworkBridgeEvents = 128;",
            "const maximumPageNetworkBridgeLifetimeEvents = 512;",
            "pageNetworkBridgeLifetimeEventCount = 0;",
            "existing.lifetimeEventCount = 0;",
            "const maximumRouteEvents = 128;",
            "const maximumLifetimeEvents = 512;",
            "inspectCurrentPerformanceResources();",
            "new PerformanceObserver",
        ])

    def test_pre_roll_wait_requires_attested_visible_playback(self):
        eligibility = self.browser.split("var likelyPreRollWait", 1)[1].split(
            "func isPreRollWaitCurrent", 1
        )[0]
        self.assert_contracts(eligibility, [
            "state.isPlaying",
            "state.isVisible",
            "state.visibilityAttested",
            "state.renderedArea >= 4_096",
            "duration >= 3, duration <= 90",
        ])
        self.assertNotIn("state.isPlaying || state.currentTime > 0", eligibility)

        current_wait = self.browser.split("func isPreRollWaitCurrent", 1)[1].split(
            "func snapshotCandidates", 1
        )[0]
        self.assert_contracts(current_wait, [
            "state.generation == wait.sourceGeneration",
            "!state.isEnded",
            "state.isVisible",
            "state.visibilityAttested",
            "state.renderedArea >= 4_096",
        ])

    def test_challenge_completion_resumes_pending_analysis_once(self):
        challenge_change = self.view.split(
            ".onChange(of: interactiveBrowser.challengeActive)", 1
        )[1].split(".onChange(of: interactiveBrowser.isLoading)", 1)[0]
        self.assert_contracts(challenge_change, [
            "if challengeActive",
            "pendingBrowserAnalysisResume = true",
            "invalidateBrowserAnalysis(preservingResumeIntent: true)",
            "resumePendingBrowserAnalysisIfReady()",
        ])

        resume = self.view.split(
            "private func resumePendingBrowserAnalysisIfReady()", 1
        )[1].split("private func urlAnalysisIsCurrent", 1)[0]
        self.assert_contracts(resume, [
            "guard pendingBrowserAnalysisResume",
            "browserAnalysisResumeTask == nil",
            "!interactiveBrowser.challengeActive",
            "!interactiveBrowser.challengeCompatibilityTimedOut",
            "!interactiveBrowser.isLoading",
            "expectedAnalysisGeneration == browserAnalysisGeneration",
            "expectedNavigationGeneration == interactiveBrowser.navigationGeneration",
            "pendingBrowserAnalysisResume = false",
            "browserAnalysisResumeTask = nil",
            "analyzeBrowserCandidates()",
        ])
        self.assertEqual(resume.count("analyzeBrowserCandidates()"), 1)
        self.assertLess(
            resume.index("pendingBrowserAnalysisResume = false"),
            resume.index("analyzeBrowserCandidates()"),
        )

    def test_interaction_result_never_navigates_or_reloads_visible_page(self):
        failure = self.view.split(
            "if store.urlInputRequiresInteraction,", 1
        )[1].split("browserFailureMessage =\n          store.urlInputStatus", 1)[0]
        self.assert_contracts(failure, [
            "store.urlInteractionURL != nil",
            "WebKit document, frame and cookie context",
            "same-document player transition",
            "never reload or replace the visible page from this result",
        ])
        self.assertNotIn("interactiveBrowser.navigate", failure)
        self.assertNotIn("interactiveBrowser.reload", failure)

    def test_hls_handoff_is_a_paired_single_owner_webkit_suspension(self):
        self.assert_contracts(self.browser, [
            "final class IPadBrowserMediaHandoffLease",
            "func acquireMediaPlaybackHandoffLease(",
            "while let activeState = mediaHandoffState",
            "endingMediaHandoffID == activeState.id",
            "await waitForMediaPlaybackHandoffEnd(id: activeState.id)",
            "func beginEnding()",
            "beginEndingMediaPlaybackHandoffLease(",
            "finishEndingMediaPlaybackHandoffLease(id: id)",
            "mediaWebView.setAllMediaPlaybackSuspended(suspended)",
            "if let openingPageWebView",
            "if let transientPopupWebView",
            "mediaWebViews.append(contentsOf: retainedPopupOpenerWebViews)",
            "func end() async",
            "let loader = resourceLoader",
            "resourceLoader = nil",
            "await loader?.cancel()",
            "await Self.setMediaPlaybackSuspended(false",
            "state.id == leaseID",
            "navigationGeneration == expectedGeneration",
            "state.visibleWebViewID == ObjectIdentifier(webView)",
        ])
        acquire = self.browser.split(
            "func acquireMediaPlaybackHandoffLease(", 1
        )[1].split("func pauseMediaPlaybackForNativeHandoff", 1)[0]
        self.assertLess(
            acquire.index("setMediaPlaybackSuspended(true"),
            acquire.index("return IPadBrowserMediaHandoffLease"),
        )
        self.assertIn("setMediaPlaybackSuspended(false", acquire)
        self.assertLess(
            acquire.index("await waitForMediaPlaybackHandoffEnd"),
            acquire.index("let generation = navigationGeneration"),
        )
        lease_end = self.browser.split("func end() async", 1)[1].split(
            "fileprivate func download(", 1
        )[0]
        self.assertLess(
            lease_end.index("beginEndingMediaPlaybackHandoffLease("),
            lease_end.index("await loader?.cancel()"),
        )
        release = self.browser.split(
            "fileprivate func endMediaPlaybackHandoffLease", 1
        )[1].split("private static func setMediaPlaybackSuspended", 1)[0]
        self.assertLess(
            release.index("setMediaPlaybackSuspended(false"),
            release.index("mediaHandoffState = nil"),
        )
        lease_end = self.browser.split("func end() async", 1)[1].split(
            "fileprivate func download(", 1
        )[0]
        self.assertLess(
            lease_end.index("await loader?.cancel()"),
            lease_end.index("endMediaPlaybackHandoffLease("),
        )
        loader_cancel = self.browser.split("func cancel() async", 1)[1].split(
            "private func waitForInFlight", 1
        )[0]
        self.assertIn("_ = await task.result", loader_cancel)
        operation_cancel = self.browser.split("private func cancel()", 1)[1].split(
            "private func finish(", 1
        )[0]
        self.assertLess(
            operation_cancel.index("download.cancel"),
            operation_cancel.index("self?.finish"),
        )

    def test_browser_analysis_retry_ends_its_pending_handoff_before_reacquiring(self):
        analysis = self.view.split(
            "private func analyzeBrowserCandidates()", 1
        )[1].split("private func startRealtimePreview()", 1)[0]
        invalidation = self.view.split(
            "private func invalidateBrowserAnalysis(", 1
        )[1].split("private func resumePendingBrowserAnalysisIfReady", 1)[0]
        self.assert_contracts(self.view, [
            "browserAnalysisHandoffLease",
            "browserAnalysisTask != nil || store.isRunning",
        ])
        self.assert_contracts(analysis, [
            "browserAnalysisTask == nil",
            "realtimePlayer.stop()",
            "retirePendingBrowserPlaybackHandoffLease()",
            "acquireMediaPlaybackHandoffLease(replacingActive: true)",
            "browserAnalysisHandoffLease = lease",
            "browserAnalysisHandoffLease = handoffLease",
            "await endBrowserAnalysisHandoffLease(handoffLease)",
        ])
        self.assert_contracts(invalidation, [
            "handoffLease.beginEnding()",
            "Task { @MainActor in await handoffLease.end() }",
        ])

    def test_successful_browser_analysis_does_not_cancel_its_playback_start(self):
        analysis = self.view.split(
            "private func analyzeBrowserCandidates()", 1
        )[1].split("private func startRealtimePreview()", 1)[0]
        tab_change = self.view.split(
            ".onChange(of: selectedTab)", 1
        )[1].split(".onChange(of: scenePhase)", 1)[0]
        self.assert_contracts(analysis, [
            "pendingBrowserAnalysisResume = false",
            "browserPlaybackTransitionInProgress = true",
            "selectedTab = .playback",
            "startAcceptedBrowserPlayback()",
        ])
        self.assert_contracts(tab_change, [
            "tab == .playback, browserPlaybackTransitionInProgress",
            "browserPlaybackTransitionInProgress = false",
            "pendingBrowserAnalysisResume = false",
            "interactiveBrowser.suspendForBackground()",
            "return",
        ])
        transition = tab_change.split(
            "if tab == .playback, browserPlaybackTransitionInProgress", 1
        )[1].split("let shouldResume", 1)[0]
        self.assertNotIn("invalidateBrowserAnalysis", transition)

        direct_start = self.view.split(
            "private func startAcceptedBrowserPlayback()", 1
        )[1].split("private func invalidateURLAnalysis()", 1)[0]
        self.assertNotIn("selectedTab == .playback", direct_start)
        self.assertIn("store.configure(with: prepared)", direct_start)
        self.assertIn("startRealtimePreview()", direct_start)

    def test_browser_waits_for_worker_before_hls_handoff_and_playback_tab(self):
        analysis = self.view.split(
            "private func analyzeBrowserCandidates()", 1
        )[1].split("private func startRealtimePreview()", 1)[0]
        self.assert_contracts(analysis, [
            "guard await prepareWorkerForBrowserPlayback(",
            'browserFailureMessage = "復元モデルの準備完了を待っています…"',
            "while case .validating = worker.preparation",
            "worker.preparedWorker != nil",
            'browserFailureMessage =\n        "復元モデルを準備できませんでした: "',
        ])
        self.assertLess(
            analysis.index("guard await prepareWorkerForBrowserPlayback("),
            analysis.index("acquireMediaPlaybackHandoffLease(replacingActive: true)"),
        )

    def test_navigation_change_retires_stale_hls_handoff_by_lease_identity(self):
        acquire = self.browser.split(
            "func acquireMediaPlaybackHandoffLease(", 1
        )[1].split("func pauseMediaPlaybackForNativeHandoff", 1)[0]
        release = self.browser.split(
            "fileprivate func beginEndingMediaPlaybackHandoffLease", 1
        )[1].split("private static func setMediaPlaybackSuspended", 1)[0]
        self.assert_contracts(acquire, [
            "replacingActive: Bool = false",
            "if replacingActive",
            "activeState.navigationGeneration != navigationGeneration",
            "navigationGeneration: activeState.navigationGeneration",
            "await endMediaPlaybackHandoffLease(",
            "continue",
        ])
        self.assert_contracts(release, [
            "state.id == id",
            "_ = expectedGeneration",
        ])
        self.assertNotIn(
            "state.navigationGeneration == expectedGeneration",
            release,
        )

    def test_hls_relay_evidence_keeps_one_exact_frame_authorization_tuple(self):
        self.assert_contracts(self.browser, [
            "private struct RelayEvidenceKey: Hashable",
            "private static let maximumRelayEvidenceCount = 256",
            "private var relayEvidence: [RelayEvidenceKey: Candidate] = [:]",
            "private func recordHLSRelayEvidence(_ candidate: Candidate)",
            "RelayEvidenceKey(",
            "url: candidate.url.absoluteString",
            "documentToken: documentToken",
            "private func retireHLSRelayEvidence(",
            "retireHLSRelayEvidence(forDocumentToken: documentToken)",
            "let matches = relayEvidence.values.filter",
            "let opaqueRelayEvidenceByURL = Dictionary(",
            "let nativeRelayEvidenceByURL = activeStates.reduce(",
            "let selectedRelayEvidence = preferredHLSRelayEvidence(",
            "let nativeRelayEvidence = nativeRelayEvidenceByURL[urlKey].flatMap",
            "let opaqueRelayEvidence = opaqueRelayEvidenceByURL[urlKey].flatMap",
            "$0.documentToken == candidate.documentToken ? $0 : nil",
            "?? nativeRelayEvidence",
            "?? opaqueRelayEvidence",
            "browserRelayRequiresProbe:",
            "selectedRelayEvidence.map { !$0.relayEligible } ?? false",
            "selectedRelayEvidence?.documentToken ?? candidate.documentToken",
            "browserRelayEligible: selectedRelayEvidence != nil",
            "selectedRelayEvidence?.relayIncludesCredentials ?? false",
        ])
        selected = self.browser.split("let selectedRelayEvidence =", 1)[1].split(
            "let relevantURLs", 1
        )[0]
        self.assertLess(
            selected.index("preferredHLSRelayEvidence("),
            selected.index("?? nativeRelayEvidence"),
        )
        preferred = self.browser.split(
            "private func preferredHLSRelayEvidence(", 1
        )[1].split("func snapshotCandidates()", 1)[0]
        self.assert_contracts(preferred, [
            "preferredDocumentToken == nil",
            "$0.documentToken == preferredDocumentToken",
        ])
        self.assertNotIn("let leftPreferred", preferred)
        # A same-token fulfilled Fetch/XHR tuple wins over provisional native
        # evidence. A different-frame credential tuple cannot fall through.
        def verified_for_candidate(candidate_token, evidence):
            return [
                item for item in evidence
                if candidate_token is None or item[0] == candidate_token
            ]

        evidence = [("frame-b", True), ("frame-a", True)]
        self.assertEqual(
            verified_for_candidate("frame-a", evidence), [("frame-a", True)]
        )
        self.assertEqual(verified_for_candidate("frame-c", evidence), [])
        insertion = self.browser.split(
            "private func insertCandidate(_ candidate: Candidate)", 1
        )[1].split("private static func isReadyCandidate", 1)[0]
        existing_merge = insertion.split("if let existing = candidates[key]", 1)[1].split(
            "nextDiscoveryOrder =", 1
        )[0]
        self.assertIn("let relayEvidenceChanged = recordHLSRelayEvidence(candidate)", existing_merge)
        self.assertLess(
            existing_merge.index("recordHLSRelayEvidence(candidate)"),
            existing_merge.index("existing.provenance != .script"),
        )
        active_merge = self.browser.split(
            "let observed = Candidate(", 1
        )[1].split("} else if snapshot.count", 1)[0]
        self.assertNotIn("relayIncludesCredentials: existing", active_merge)
        self.assertNotIn("relayEligible: existing", active_merge)
        evidence_filter = self.browser.split(
            "private static func shouldRecordHLSRelayEvidence", 1
        )[1].split("@discardableResult", 1)[0]
        self.assertNotIn("candidate.relayIncludesCredentials", evidence_filter)
        evidence_merge = self.browser.split(
            "private func recordHLSRelayEvidence(_ candidate: Candidate)", 1
        )[1].split("private func removeHLSRelayEvidence", 1)[0]
        self.assertIn("let selected = useCandidateAsBase ? candidate : existing", evidence_merge)
        self.assertIn("relayEvidence[key] = selected", evidence_merge)
        self.assertNotIn("existing.relayIncludesCredentials\n          ||", evidence_merge)
        self.assertNotIn("existing.relayEligible ||", evidence_merge)

    def test_passive_reinspection_preserves_current_opaque_generation_evidence(self):
        self.assert_contracts(self.browser, [
            "private static func isPassiveOpaquePlaybackHLSHint",
            'case "performance", "script-text":',
            "private func isInCurrentOpaquePlaybackActivationWindow",
            "isMediaDocumentCurrent(documentToken)",
            "key.documentToken == documentToken",
            "state.hasOpaqueSource && !state.isEnded",
            "state.sourceActivatedAt.addingTimeInterval(-5)",
            "state.sourceActivatedAt.addingTimeInterval(12)",
            "let existingInActivationWindow =",
            "let candidateInActivationWindow =",
            "existingInActivationWindow != candidateInActivationWindow",
            "? candidateInActivationWindow",
            ": candidate.observedAt > existing.observedAt",
        ])
        evidence_merge = self.browser.split(
            "private func recordHLSRelayEvidence(_ candidate: Candidate)", 1
        )[1].split("private func removeHLSRelayEvidence", 1)[0]
        # A repeated inspection after +12 seconds must keep the old in-window
        # exact-frame observation. When a new source generation moves the
        # window, old-out/new-in must select the new observation instead.
        self.assertLess(
            evidence_merge.index("existing.relayEligible != candidate.relayEligible"),
            evidence_merge.index("existingInActivationWindow"),
        )
        self.assertLess(
            evidence_merge.index("existingInActivationWindow != candidateInActivationWindow"),
            evidence_merge.index("? candidateInActivationWindow"),
        )
        select_candidate = lambda existing_in, candidate_in, candidate_is_newer: (
            candidate_in
            if existing_in != candidate_in
            else candidate_is_newer
        )
        self.assertFalse(select_candidate(True, False, True))
        self.assertTrue(select_candidate(False, True, False))

    def test_hls_uses_webkit_download_without_relay_eligibility_or_js_fetch(self):
        self.assert_contracts(self.browser, [
            "private final class IPadBrowserWebKitDownloadOperation",
            "WKDownloadDelegate",
            "webView.startDownload(using: browserRequest)",
            'browserRequest.setValue(nil, forHTTPHeaderField: "Cookie")',
            'browserRequest.setValue(nil, forHTTPHeaderField: "User-Agent")',
            'browserRequest.setValue(nil, forHTTPHeaderField: "Origin")',
            "browserRequest.httpShouldHandleCookies = true",
            "decideDestinationUsing response: URLResponse",
            "downloadDidFinish(_ download: WKDownload)",
            "didFailWithError error: Error",
            "canDownloadHLSWithWebKit(",
            "downloadHLSResourceWithWebKit(",
            "private actor IPadBrowserHLSResourceLoader: IPadHLSResourceLoading",
            "IPadBrowserHLSFetchCacheMode",
            'case live = "default"',
            'case videoOnDemand = "force-cache"',
            "isLive: Bool",
            "cacheMode == .videoOnDemand",
            "private static let maximumCacheBytes = 64 * 1_024 * 1_024",
            "private static let maximumRelayResourceBytes = 32 * 1_024 * 1_024",
            "private var inFlight: [RequestKey: InFlight]",
            "var waiters:",
            "cancelInFlightWaiter(",
            "if entry.waiters.isEmpty",
            "let relayRequest = request",
            "private static func validatedResponse(",
            "contentRangeMatchesRequest(",
            "loaded.response.statusCode == 206",
            "drainingTasks",
            "_ = await task.result",
            "private let relayDispatchGate = IPadBrowserHLSRelayDispatchGate()",
            "try await relayDispatchGate.perform",
            '#"^bytes=([0-9]+-[0-9]*|-[0-9]+)$"#',
        ])
        self.assertNotIn("hlsResourceRelayScript", self.browser)
        self.assertNotIn("relayHLSResource(", self.browser)
        self.assertNotIn("candidate.browserRelayEligible,", self.browser)
        loader = self.browser.split(
            "private actor IPadBrowserHLSResourceLoader", 1
        )[1].split("@MainActor", 1)[0]
        self.assertIn("if cacheMode == .videoOnDemand", loader)
        cache_insert = loader.split("private func insertIntoCache", 1)[1].split(
            "private func purgeExpiredCache", 1
        )[0]
        self.assertIn("guard cacheMode == .videoOnDemand", cache_insert)
        lease_loader = self.browser.split("func resourceLoader(", 1)[1].split(
            "func end() async", 1
        )[0]
        self.assertIn(
            "private var resourceLoader: IPadBrowserHLSResourceLoader?",
            self.browser,
        )
        self.assertIn("isLive: Bool", lease_loader)
        self.assertIn("retainingResolvedResourceURL: URL? = nil", lease_loader)
        self.assertIn("await resourceLoader.updateCacheMode(", lease_loader)
        self.assertIn("cacheMode: isLive ? .live : .videoOnDemand", lease_loader)
        self.assertIn("guard let lease else { throw CancellationError() }", loader)
        relay_task = loader.split(
            "let task = Task<IPadHLSResourceLoadResult?, Error>", 1
        )[1].split("let inFlightID = UUID()", 1)[0]
        self.assertIn("throw CancellationError()", relay_task)
        self.assertNotIn("return nil", relay_task)
        request_key = self.browser.split("private struct RequestKey", 1)[1].split(
            "private struct CacheKey", 1
        )[0]
        self.assertIn("let range: String?", request_key)
        self.assertNotIn("maximumResponseBytes", request_key)
        waiter = self.browser.split("private func waitForInFlight", 1)[1].split(
            "private func cancelInFlightWaiter", 1
        )[0]
        self.assertNotIn("task.cancel()", waiter)
        registration = self.browser.split(
            "private func registerInFlightWaiter", 1
        )[1].split("private func cancelInFlightWaiter", 1)[0]
        self.assertLess(
            registration.index("waiters: [waiterID: waiter]"),
            registration.index("Task { [weak self]"),
        )
        self.assertNotIn("recentlyCompleted", self.browser)
        self.assertNotIn(
            "IPadHLSRelayRangeNormalizer.canonicalRequest(for: request)", loader
        )

    def test_browser_resource_loader_runtime_coalesces_full_and_range(self):
        if sys.platform != "darwin":
            self.skipTest("browser resource loader probe requires macOS")
        swiftc = shutil.which("swiftc")
        if swiftc is None:
            xcrun = shutil.which("xcrun")
            if xcrun is not None:
                swiftc = subprocess.check_output(
                    [xcrun, "--find", "swiftc"], text=True
                ).strip()
        if not swiftc:
            self.skipTest("Swift compiler is required for the loader probe")

        actor_tail = self.browser.split(
            "private actor IPadBrowserHLSResourceLoader", 1
        )[1].split("\n}\n\n/// Gives a user-created", 1)[0]
        actor_source = (
            "import Foundation\n\n"
            "actor IPadBrowserHLSResourceLoader"
            + actor_tail
            + "\n}\n"
        ).replace("IPadBrowserMediaHandoffLease", "RelayLease")

        with tempfile.TemporaryDirectory(prefix="mioh-browser-loader-") as directory:
            directory_path = Path(directory)
            actor_file = directory_path / "IPadBrowserHLSResourceLoader.swift"
            actor_file.write_text(actor_source, encoding="utf-8")
            executable = directory_path / "browser-loader-probe"
            build = subprocess.run(
                [
                    swiftc,
                    "-module-cache-path",
                    str(directory_path / "module-cache"),
                    "-D",
                    "MIOH_TESTING",
                    "-parse-as-library",
                    str(RESOLVER),
                    str(actor_file),
                    str(RESOURCE_LOADER_HARNESS),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                build.returncode,
                0,
                f"browser resource loader probe did not compile:\n{build.stdout}{build.stderr}",
            )
            completed = subprocess.run(
                [str(executable)],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertIn(
            "iPad browser HLS resource loader probe passed", completed.stdout
        )

    def test_webkit_download_runtime_uses_browser_cookie_instead_of_native_snapshot(self):
        if sys.platform != "darwin":
            self.skipTest("WKDownload probe requires macOS")
        xcrun = shutil.which("xcrun")
        if xcrun is None:
            self.skipTest("Xcode is required for the WKDownload probe")

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/page":
                    body = b"<html><body>browser session</body></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Set-Cookie", "session=safari; Path=/")
                elif self.path == "/segment":
                    cookie = self.headers.get("Cookie", "")
                    user_agent = self.headers.get("User-Agent", "")
                    origin = self.headers.get("Origin")
                    accepted = (
                        "session=safari" in cookie
                        and user_agent != "Native-Imitation"
                        and origin != "https://wrong.invalid"
                    )
                    full_body = b"webkit-session-ok"
                    range_header = self.headers.get("Range")
                    range_accepted = accepted and range_header == "bytes=2-5"
                    body = (
                        full_body[2:6]
                        if range_accepted
                        else full_body if accepted else b"native-snapshot-rejected"
                    )
                    self.send_response(206 if range_accepted else 200 if accepted else 429)
                    self.send_header("Content-Type", "video/mp2t")
                    if range_accepted:
                        self.send_header("Accept-Ranges", "bytes")
                        self.send_header("Content-Range", "bytes 2-5/17")
                else:
                    body = b"not found"
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            operation_tail = self.browser.split(
                "private final class IPadBrowserWebKitDownloadOperation", 1
            )[1].split(
                "\n}\n\nprivate enum IPadBrowserHLSFetchCacheMode", 1
            )[0]
            operation_source = (
                "import Foundation\nimport WebKit\n\n"
                "@MainActor\nfinal class IPadBrowserWebKitDownloadOperation"
                + operation_tail
                + "\n}\n"
            )
            with tempfile.TemporaryDirectory(prefix="mioh-wkdownload-") as directory:
                directory_path = Path(directory)
                operation_file = directory_path / "WebKitDownloadOperation.swift"
                operation_file.write_text(operation_source, encoding="utf-8")
                executable = directory_path / "wkdownload-probe"
                build = subprocess.run(
                    [
                        xcrun,
                        "swiftc",
                        "-module-cache-path",
                        str(directory_path / "module-cache"),
                        "-D",
                        "MIOH_TESTING",
                        "-parse-as-library",
                        "-framework",
                        "AppKit",
                        "-framework",
                        "WebKit",
                        str(RESOLVER),
                        str(operation_file),
                        str(WEBKIT_DOWNLOAD_HARNESS),
                        "-o",
                        str(executable),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(
                    build.returncode,
                    0,
                    f"WKDownload probe did not compile:\n{build.stdout}{build.stderr}",
                )
                completed = subprocess.run(
                    [
                        str(executable),
                        f"http://127.0.0.1:{port}/page",
                        f"http://127.0.0.1:{port}/segment",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"WKDownload probe failed:\n{completed.stdout}{completed.stderr}",
                )
                self.assertIn(
                    "WKDownload browser-session probe passed", completed.stdout
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_hls_relay_runtime_streams_bounded_chunks_and_fetch_options(self):
        self.skipTest("JavaScript fetch relay was replaced by WKDownload")
        script = json.dumps(self.hls_relay_runtime_source())
        runtime = """
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
const relay = new AsyncFunction(
  'urlString', 'method', 'includesCredentials', 'cacheMode', 'maximumBytes',
  'timeoutMilliseconds', 'operationID', 'expectedDocumentToken',
  'rangeHeader', %s
);
global.window = {__miohInteractiveDocumentToken: 'document-1'};
let observedOptions = null;
global.fetch = async (url, options) => {
  observedOptions = options;
  let delivered = false;
  return {
    type: 'cors', redirected: false, url, status: 206,
    headers: {forEach(callback) { callback('video/mp2t', 'content-type'); }},
    body: {getReader() { return {
      async read() {
        if (delivered) return {done: true};
        delivered = true;
        return {done: false, value: new Uint8Array(25000).fill(7)};
      },
      async cancel() {}
    }; }}
  };
};
(async () => {
  const result = await relay(
    'https://cdn.example/segment.ts', 'GET', true, 'force-cache', 30000, 5000,
    'operation-1', 'document-1', 'bytes=0-24999'
  );
  process.stdout.write(JSON.stringify({
    result,
    redirect: observedOptions.redirect,
    cache: observedOptions.cache,
    credentials: observedOptions.credentials,
    range: observedOptions.headers.Range
  }));
})().catch(error => { console.error(error); process.exit(1); });
""" % script
        completed = subprocess.run(
            ["node", "-e", runtime],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["result"]["ok"])
        self.assertEqual(payload["result"]["totalBytes"], 25000)
        self.assertEqual(len(payload["result"]["chunks"]), 2)
        self.assertEqual(payload["redirect"], "error")
        self.assertEqual(payload["cache"], "force-cache")
        self.assertEqual(payload["credentials"], "include")
        self.assertEqual(payload["range"], "bytes=0-24999")

    def test_hls_relay_runtime_uses_default_webkit_cache_for_live_hls(self):
        self.skipTest("JavaScript fetch relay was replaced by WKDownload")
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = json.dumps(self.hls_relay_runtime_source())
        runtime = """
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
const relay = new AsyncFunction(
  'urlString', 'method', 'includesCredentials', 'cacheMode', 'maximumBytes',
  'timeoutMilliseconds', 'operationID', 'expectedDocumentToken',
  'rangeHeader', %s
);
global.window = {__miohInteractiveDocumentToken: 'document-live'};
let observedCache = null;
global.fetch = async (url, options) => {
  observedCache = options.cache;
  return {
    type: 'cors', redirected: false, url, status: 200,
    headers: {forEach(callback) {
      callback('application/vnd.apple.mpegurl', 'content-type');
    }},
    body: null
  };
};
(async () => {
  const result = await relay(
    'https://cdn.example/live.m3u8', 'HEAD', false, 'default', 30000, 5000,
    'operation-live', 'document-live', null
  );
  process.stdout.write(JSON.stringify({result, observedCache}));
})().catch(error => { console.error(error); process.exit(1); });
""" % script
        completed = subprocess.run(
            [node, "-e", runtime],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["result"]["ok"])
        self.assertEqual(payload["observedCache"], "default")

    def test_hls_relay_runtime_canonicalizes_equivalent_https_urls(self):
        self.skipTest("JavaScript fetch relay was replaced by WKDownload")
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript behavior probe")
        script = json.dumps(self.hls_relay_runtime_source())
        runtime = """
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
const relay = new AsyncFunction(
  'urlString', 'method', 'includesCredentials', 'cacheMode', 'maximumBytes',
  'timeoutMilliseconds', 'operationID', 'expectedDocumentToken',
  'rangeHeader', %s
);
global.window = {__miohInteractiveDocumentToken: 'document-canonical'};
let observedURL = null;
global.fetch = async (url) => {
  observedURL = url;
  return {
    type: 'cors', redirected: false, url, status: 200,
    headers: {forEach() {}}, body: null
  };
};
(async () => {
  const requestURL = 'https://EXAMPLE.com:443/segment.ts';
  const result = await relay(
    requestURL, 'HEAD', false, 'default', 30000, 5000,
    'operation-canonical', 'document-canonical', null
  );
  process.stdout.write(JSON.stringify({result, observedURL, requestURL}));
})().catch(error => { console.error(error); process.exit(1); });
""" % script
        completed = subprocess.run(
            [node, "-e", runtime],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["result"]["ok"])
        self.assertEqual(payload["observedURL"], "https://example.com/segment.ts")
        self.assertEqual(payload["result"]["url"], payload["requestURL"])

    def test_hls_relay_runtime_marks_post_dispatch_failure_as_attempted(self):
        self.skipTest("JavaScript fetch relay was replaced by WKDownload")
        script = json.dumps(self.hls_relay_runtime_source())
        runtime = """
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
const relay = new AsyncFunction(
  'urlString', 'method', 'includesCredentials', 'cacheMode', 'maximumBytes',
  'timeoutMilliseconds', 'operationID', 'expectedDocumentToken',
  'rangeHeader', %s
);
global.window = {__miohInteractiveDocumentToken: 'document-1'};
global.fetch = async () => { throw new TypeError('CORS'); };
(async () => {
  const result = await relay(
    'https://cdn.example/segment.ts', 'GET', false, 'force-cache', 30000, 5000,
    'operation-2', 'document-1', null
  );
  process.stdout.write(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });
""" % script
        completed = subprocess.run(
            ["node", "-e", runtime],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["kind"], "attempted")

    def test_browser_cleanup_is_cancellable_and_does_not_log_secrets(self):
        self.assert_contracts(self.browser, [
            "IPadWeakInteractiveScriptMessageHandler",
            "weak var delegate",
            "quietTask?.cancel()",
            "webView.stopLoading()",
            "loadHTMLString(Self.blankPageHTML",
        ])
        self.assertIn("cancelURLResolution()", self.store)
        self.assertNotIn("UserDefaults", self.browser)
        self.assertNotIn("print(", self.browser)
        self.assertNotIn("os_log", self.browser)

    def test_webview_validation_does_not_block_main_actor_on_dns(self):
        self.assert_contracts(self.browser, [
            "isPublicHostSyntax",
            '!host.hasSuffix(".local")',
            '!host.hasPrefix("::ffff:")',
        ])
        self.assertNotIn("getaddrinfo", self.browser)
        self.assertNotIn("IPadMediaURLResolver.isPublicHTTPSURL", self.browser)


if __name__ == "__main__":
    unittest.main()
