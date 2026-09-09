from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "MiohRemote" / "MiohRemote"
DISCOVERY = APP / "IPadWebMediaDiscovery.swift"
BROWSER = APP / "IPadInteractiveMediaBrowser.swift"
VIEW = APP / "IPadStandaloneView.swift"
STORE = APP / "IPadStandaloneStore.swift"
RESOLVER = APP / "IPadMediaURLResolver.swift"
PROJECT = ROOT / "apps" / "MiohRemote" / "MiohRemote.xcodeproj" / "project.pbxproj"


class IPadWebMediaDiscoveryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.discovery = DISCOVERY.read_text(encoding="utf-8")
        cls.browser = BROWSER.read_text(encoding="utf-8")
        cls.view = VIEW.read_text(encoding="utf-8")
        cls.store = STORE.read_text(encoding="utf-8")
        cls.resolver = RESOLVER.read_text(encoding="utf-8")
        cls.project = PROJECT.read_text(encoding="utf-8")

    def assert_contracts(self, source, contracts):
        for contract in contracts:
            self.assertIn(contract, source, f"missing source contract: {contract}")

    def test_offscreen_page_execution_is_not_available(self):
        self.assertTrue(DISCOVERY.exists())
        self.assertNotIn("import WebKit", self.discovery)
        self.assertNotIn("WKWebView", self.discovery)
        self.assertNotIn("evaluateJavaScript", self.discovery)
        self.assertNotIn(".load(", self.discovery)
        self.assert_contracts(
            self.discovery,
            [
                "Offscreen page execution is intentionally disabled",
                "final class IPadWebMediaDiscovery",
                "throw IPadWebMediaDiscoveryError.interactionRequired",
            ],
        )
        self.assertIn("IPadWebMediaDiscovery.swift in Sources", self.project)

    def test_discovery_file_only_keeps_shared_types_and_safe_routing_error(self):
        self.assert_contracts(
            self.discovery,
            [
                "struct IPadWebMediaCandidate: Sendable, Equatable",
                "let requestContext: IPadMediaRequestContext",
                "struct IPadWebMediaDiscoveryResult: Sendable, Equatable",
                "func discoverCandidates(from rawValue: String) async throws",
                "Task.checkCancellation()",
                "isValidVisibleBrowserURL",
                'components.scheme?.lowercased() == "https"',
                "components.user == nil",
                "components.password == nil",
            ],
        )

    def test_store_routes_dynamic_pages_to_visible_browser_without_network_fallback(self):
        self.assert_contracts(
            self.store,
            [
                "resolveURLInput(",
                "shouldOpenVisibleBrowser(after: error)",
                'urlInputStatus = "動的ページはブラウザタブで確認してください。"',
                "表示中のブラウザタブへ切り替えます。",
                "throw IPadWebMediaDiscoveryError.interactionRequired",
            ],
        )
        self.assertNotIn(
            "IPadWebMediaDiscovery().discoverCandidates(", self.store
        )

    def test_visible_browser_owns_all_dynamic_page_instrumentation(self):
        self.assert_contracts(
            self.browser,
            [
                "import WebKit",
                "final class IPadInteractiveMediaBrowser",
                "WKWebsiteDataStore.default()",
                "WKWebViewConfiguration()",
                "instrumentationContentWorld",
                "maximumFrameDepth = 8",
                "window.fetch",
                "XMLHttpRequest.prototype.open",
                "MutationObserver",
                "PerformanceObserver",
                "snapshotCandidates()",
            ],
        )

    def test_candidate_keeps_exact_browser_document_for_hls_relay(self):
        self.assert_contracts(
            self.discovery,
            [
                "let browserDocumentToken: String?",
                "browserDocumentToken: String? = nil",
                "self.browserDocumentToken = browserDocumentToken",
                "let browserRelayEligible: Bool",
                "let browserRelayRequiresProbe: Bool",
                "browserRelayRequiresProbe: Bool = false",
                "self.browserRelayRequiresProbe = browserRelayRequiresProbe",
                "let browserRelayIncludesCredentials: Bool",
            ],
        )
        snapshot = self.browser.split("func snapshotCandidates()", 1)[1].split(
            "private static func browserMediaEvidence", 1
        )[0]
        self.assert_contracts(
            snapshot,
            [
                "key: MediaSlotKey, state: MediaSlotState",
                "documentToken: entry.key.documentToken",
                "documentToken: observed.documentToken ?? existing.documentToken",
                "let opaqueRelayEvidenceByURL = Dictionary(",
                "let nativeRelayEvidenceByURL = activeStates.reduce(",
                "let selectedRelayEvidence = preferredHLSRelayEvidence(",
                "let nativeRelayEvidence = nativeRelayEvidenceByURL[urlKey].flatMap",
                "let opaqueRelayEvidence = opaqueRelayEvidenceByURL[urlKey].flatMap",
                "$0.documentToken == candidate.documentToken ? $0 : nil",
                "?? nativeRelayEvidence",
                "?? opaqueRelayEvidence",
                "preferredDocumentToken: candidate.documentToken",
                "selectedRelayEvidence?.documentToken ?? candidate.documentToken",
                "browserRelayEligible: selectedRelayEvidence != nil",
                "browserRelayRequiresProbe:",
                "selectedRelayEvidence.map { !$0.relayEligible } ?? false",
                "selectedRelayEvidence?.relayIncludesCredentials ?? false",
            ],
        )

    def test_visible_browser_candidates_still_use_dns_aware_resolver(self):
        self.assert_contracts(
            self.store,
            [
                "selectBrowserCandidates(",
                "for candidate in pool",
                "guard remainingAttempts > 0 else { break }",
                "catch IPadMediaURLResolverError.unsafeInitialURL",
                "remainingAttempts += 1",
                "maximumPlayableChoices",
                "let policy = browserResolutionPolicy(for: candidate)",
                "policy: policy",
                "candidate.url.absoluteString",
                "context: candidate.requestContext",
            ],
        )

    def test_dynamic_context_is_also_used_for_progressive_downloads(self):
        self.assert_contracts(
            self.store,
            [
                "resolutionPolicy: source.resolutionPolicy",
                "requestContext: source.requestContext",
                "source.requestContext?.applying(to: &request)",
                "IPadMediaURLResolver.isURL(destination, allowedBy: resolutionPolicy)",
                "requestContext?.applying(to: &sanitizedRequest)",
            ],
        )

    def test_interaction_routing_opens_the_visible_browser_tab(self):
        self.assert_contracts(
            self.view,
            [
                "private func openBrowserForInteraction(_ rawValue: String)",
                "selectedTab = .browser",
                "interactiveBrowser.navigate(rawValue)",
                "store.urlInteractionURL?.absoluteString ?? rawValue",
                "invalidateBrowserAnalysis(closePage: true)",
            ],
        )

    def test_challenge_fails_closed_across_static_pages_and_hls_variants(self):
        self.assertGreaterEqual(
            self.resolver.count(
                "catch IPadMediaURLResolverError.interactionRequired"
            ),
            3,
        )
        self.assert_contracts(
            self.resolver,
            [
                "let orderedVariants = Array(",
                "try Self.restorationVariantOrder(variants).prefix(8)",
                "for (variantIndex, variant) in orderedVariants.enumerated()",
                "throw IPadMediaURLResolverError.interactionRequired",
                "validateNoInteractionChallenge(payload.response)",
                "isInteractionChallengeURL",
            ],
        )
        self.assertNotIn("var encounteredInteractionRequired", self.resolver)


if __name__ == "__main__":
    unittest.main()
