import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "MiohRemote" / "MiohRemote"
VIEW = APP / "IPadStandaloneView.swift"
BROWSER = APP / "IPadInteractiveMediaBrowser.swift"
LIBRARY = APP / "IPadBrowserLibraryStore.swift"
STORE = APP / "IPadStandaloneStore.swift"


class IPadBrowserBackgroundRestoreContractTests(unittest.TestCase):
    """Contracts for returning from background without losing browser state.

    History/bookmarks are durable library data. The visible URL, address-field
    draft and the Basic-tab URL draft are session data: they must survive a
    background/foreground transition even when SwiftUI recreates the browser
    representable. A regenerated WKWebView must remain owned by the same
    StateObject controller and use that controller's persistent WebKit store.
    """

    @classmethod
    def setUpClass(cls):
        cls.view = VIEW.read_text(encoding="utf-8")
        cls.browser = BROWSER.read_text(encoding="utf-8")
        cls.library = LIBRARY.read_text(encoding="utf-8")
        cls.store = STORE.read_text(encoding="utf-8")
        cls.scene_phase = cls.view.split(
            ".onChange(of: scenePhase)", 1
        )[1].split(".onReceive(playbackTimer)", 1)[0]

    def assert_contracts(self, source, contracts):
        for contract in contracts:
            with self.subTest(contract=contract):
                self.assertIn(contract, source)

    def test_history_and_bookmarks_remain_durable_across_view_recreation(self):
        self.assert_contracts(
            self.library,
            [
                "struct Entry: Codable, Equatable, Identifiable",
                '"mioh.ipad.browser.history.v1"',
                '"mioh.ipad.browser.bookmarks.v1"',
                "history = Self.loadEntries(",
                "bookmarks = Self.loadEntries(",
                "JSONEncoder().encode(entries)",
                "JSONDecoder().decode([Entry].self, from: data)",
            ],
        )
        self.assertIn(
            "@StateObject private var browserLibrary = IPadBrowserLibraryStore()",
            self.view,
        )

    def test_background_suspends_browser_without_closing_or_blank_replacement(self):
        background = self.scene_phase.split("if phase == .background", 1)[1]
        background = background.split("else if phase == .active", 1)[0]

        # closePage intentionally blanks and replaces WebKit, clears addressText
        # and loses its back/forward list. Backgrounding must use a resumable
        # suspension path instead; explicit Close and tab changes may still use
        # closePage.
        self.assertNotIn("invalidateBrowserAnalysis(closePage: true)", background)
        self.assertNotIn("interactiveBrowser.closePage()", background)
        self.assertIn("interactiveBrowser.suspendForBackground()", background)

    def test_foreground_resumes_same_page_and_browser_analysis_session(self):
        foreground = self.scene_phase.split("else if phase == .active", 1)[1]
        self.assertIn("interactiveBrowser.resumeAfterBackground()", foreground)
        self.assertIn("resumePendingBrowserAnalysisIfReady()", foreground)

    def test_page_address_and_input_url_have_explicit_session_persistence(self):
        combined = self.view + "\n" + self.browser + "\n" + self.store

        # The implementation may use SceneStorage or a Codable defaults-backed
        # browser session, but all three user-visible values need stable keys:
        # the committed page URL, an unsubmitted address draft, and the Basic
        # tab's media URL draft. Resolved signed media URLs/cookies are not part
        # of this persistence contract.
        expected_keys = [
            "mioh.ipad.browser.current-page.v1",
            "mioh.ipad.browser.address-input.v1",
            "mioh.ipad.browser.media-input.v1",
        ]
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertTrue(
                    key in combined,
                    f"missing persistent iPad browser session key: {key}",
                )
        self.assertRegex(
            combined,
            re.compile(r"(?:SceneStorage|UserDefaults|JSONEncoder\(\)\.encode)"),
        )
        self.assertRegex(combined, re.compile(r"(?:restore|load).*Browser", re.I))

    def test_background_does_not_clear_resolved_input_or_url_drafts(self):
        background = self.scene_phase.split("if phase == .background", 1)[1]
        background = background.split("else if phase == .active", 1)[0]
        self.assertNotIn("store.clearInput()", background)
        self.assertNotRegex(background, r"mediaURLText\s*=\s*\"\"")
        self.assertNotRegex(background, r"addressText\s*=\s*\"\"")

    def test_recreated_webview_reuses_stateobject_controller_and_webkit_session(self):
        self.assert_contracts(
            self.view,
            [
                "@StateObject private var interactiveBrowser = IPadInteractiveMediaBrowser()",
                "IPadInteractiveBrowserWebView(browser: interactiveBrowser)",
                ".id(interactiveBrowser.webViewGeneration)",
            ],
        )
        representable = self.browser.split(
            "struct IPadInteractiveBrowserWebView: UIViewRepresentable", 1
        )[1].split("#elseif os(macOS)", 1)[0]
        self.assertIn("browser.webView", representable)
        self.assertNotIn("WKWebView(", representable)

        self.assert_contracts(
            self.browser,
            [
                "private let websiteDataStore: WKWebsiteDataStore",
                "private let messageHandlerProxy:",
                "WKWebsiteDataStore.default()",
                "websiteDataStore: websiteDataStore",
                "messageHandlerProxy: messageHandlerProxy",
                "configuration.websiteDataStore = websiteDataStore",
            ],
        )


if __name__ == "__main__":
    unittest.main()
