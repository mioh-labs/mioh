from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROXY = (
    ROOT
    / "apps"
    / "MiohRemote"
    / "MiohRemote"
    / "IPadAuthenticatedMediaProxy.swift"
)
STORE = ROOT / "apps" / "MiohRemote" / "MiohRemote" / "IPadStandaloneStore.swift"
VIEW = ROOT / "apps" / "MiohRemote" / "MiohRemote" / "IPadStandaloneView.swift"
PROJECT = ROOT / "apps" / "MiohRemote" / "MiohRemote.xcodeproj" / "project.pbxproj"
RESOLVER = ROOT / "apps" / "MiohRemote" / "MiohRemote" / "IPadMediaURLResolver.swift"
HARNESS = ROOT / "tests" / "swift" / "IPadAuthenticatedMediaProxyHarness.swift"


class IPadAuthenticatedMediaProxyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = PROXY.read_text(encoding="utf-8")
        cls.resolver = RESOLVER.read_text(encoding="utf-8")
        cls.store = STORE.read_text(encoding="utf-8")
        cls.view = VIEW.read_text(encoding="utf-8")
        cls.project = PROJECT.read_text(encoding="utf-8")

    def assert_contracts(self, contracts):
        for contract in contracts:
            self.assertIn(contract, self.source, f"missing source contract: {contract}")

    def test_listener_is_loopback_only_and_uses_an_ephemeral_port(self):
        self.assert_contracts([
            "import Network",
            "final class IPadAuthenticatedMediaProxy",
            'parameters.requiredLocalEndpoint = .hostPort(',
            'host: "127.0.0.1"',
            "port: .any",
            "func start() async throws",
            "func stop()",
        ])

    def test_local_urls_are_opaque_and_targets_are_memory_only(self):
        self.assert_contracts([
            "private var targets: [String: TargetEntry] = [:]",
            'UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()',
            'let resourceName = isPlaylist ? "index.m3u8" : "resource"',
            'string: "http://127.0.0.1:\\(listenerPort)/v1/\\(token)/\\(resourceName)"',
            "No origin URL components are exposed",
            "targets.removeAll()",
            "maximumMappedTargets",
        ])
        self.assertNotIn("print(", self.source)
        self.assertNotIn("os_log", self.source)
        self.assertNotIn("Logger(", self.source)

    def test_only_approved_public_https_targets_and_redirects_are_fetched(self):
        self.assert_contracts([
            "sanitizedPublicHTTPSURL",
            "sanitizedHTTPSURL",
            'components.scheme?.lowercased() == "https"',
            "components.user == nil, components.password == nil",
            "IPadMediaURLResolver.isURL(safeURL, allowedBy: resolutionPolicy)",
        ])
        for contract in [
            "willPerformHTTPRedirection response",
            "let safeDestination = IPadMediaURLResolver.sanitizedAbsoluteHTTPURL(",
            "IPadMediaURLResolver.isURL(",
            "entry.options.requestContext?.applying(to: &redirected)",
        ]:
            self.assertIn(contract, self.resolver)

    def test_browser_vpn_policy_is_scoped_to_each_target_and_every_hls_hop(self):
        self.assert_contracts([
            "let resolutionPolicy: IPadMediaURLResolutionPolicy",
            "resolutionPolicy: IPadMediaURLResolutionPolicy = .publicDiscovered",
            "entry.resolutionPolicy == resolutionPolicy",
            "resolutionPolicy: entry.resolutionPolicy",
            "resolutionPolicy: resolutionPolicy",
            "private let resolutionPolicy: IPadMediaURLResolutionPolicy",
        ])
        self.assertIn("resolutionPolicy: source.resolutionPolicy", self.store)
        self.assertGreaterEqual(
            self.source.count("resolutionPolicy: entry.resolutionPolicy"),
            4,
        )
        target_match = self.source.split(
            "if let entry = targets[token], entry.context == context,", 1
        )[1].split("touchTargetLocked(token)", 1)[0]
        self.assertIn("entry.resolutionPolicy == resolutionPolicy", target_match)

    def test_auth_context_range_and_limits_are_applied(self):
        self.assert_contracts([
            "var maximumConcurrentRequests = 4",
            "var maximumMappedTargets = 32_768",
            'request.setValue(range, forHTTPHeaderField: "Range")',
            "entry.context?.applying(to: &request)",
            "await entry.context?.updateCookies(from: payload.response)",
            "maximumConcurrentRequests",
            "maximumRequestHeaderBytes",
            "headerTimeouts",
            "maximumResponseBytes",
            "maximumPlaylistBytes",
            "maximumRedirectCount",
            "requestTimeout",
            "IPadAuthenticatedMediaProxyOriginGate",
            "try await originRequestGate.acquire()",
            "await originRequestGate.release()",
            "maximumPendingConnections",
        ])
        self.assertNotIn("sendError(statusCode: 503", self.source)
        self.assertNotIn("while targets.count >=", self.source)
        self.assertIn("let existingTokens = Set(targets.keys)", self.source)
        for contract in [
            "static let shared = IPadSharedHTTPTransport()",
            "URLSessionConfiguration.ephemeral",
            "configuration.httpCookieStorage = nil",
            "configuration.httpCookieAcceptPolicy = .never",
            "configuration.httpShouldSetCookies = false",
            "configuration.urlCredentialStorage = nil",
            'redirected.setValue(',
            'entry.request.value(forHTTPHeaderField: "Range")',
        ]:
            self.assertIn(contract, self.resolver)

    def test_playlist_accept_is_not_replayed_to_binary_hls_resources(self):
        fetch_body = self.source.split(
            "private func fetch(\n    request localRequest: LocalRequest,", 1
        )[1].split("let operation = IPadAuthenticatedMediaOriginRequest(", 1)[0]
        self.assertIn("if entry.isPlaylist {", fetch_body)
        self.assertIn('forHTTPHeaderField: "Accept"', fetch_body)

        downloader_body = self.resolver.split(
            "final class IPadHLSResourceDownloader", 1
        )[1].split("func materialize(", 1)[0]
        self.assertNotIn('forHTTPHeaderField: "Accept"', downloader_body)

        redirect_body = self.resolver.split(
            "willPerformHTTPRedirection response", 1
        )[1].split("didCompleteWithError", 1)[0]
        self.assertIn(
            'entry.request.value(forHTTPHeaderField: "Accept")',
            redirect_body,
        )
        self.assertLess(
            redirect_body.index('forHTTPHeaderField: "Accept"'),
            redirect_body.index('forHTTPHeaderField: "Range"'),
        )

    def test_https_hls_requests_race_http3_and_preserve_that_choice_on_redirect(self):
        fetch_body = self.source.split(
            "private func fetch(\n    request localRequest: LocalRequest,", 1
        )[1].split("let operation = IPadAuthenticatedMediaOriginRequest(", 1)[0]
        self.assertIn("request.assumesHTTP3Capable = true", fetch_body)

        downloader_body = self.resolver.split(
            "final class IPadHLSResourceDownloader", 1
        )[1].split("func materialize(", 1)[0]
        self.assertIn(
            'if safeURL.scheme?.lowercased() == "https"', downloader_body
        )
        self.assertIn("request.assumesHTTP3Capable = true", downloader_body)

        redirect_body = self.resolver.split(
            "willPerformHTTPRedirection response", 1
        )[1].split("didCompleteWithError", 1)[0]
        self.assertIn(
            "redirected.assumesHTTP3Capable = entry.request.assumesHTTP3Capable",
            redirect_body,
        )
        request_key = self.resolver.split("private struct RequestKey", 1)[1].split(
            "private struct Waiter", 1
        )[0]
        self.assertIn("let assumesHTTP3Capable: Bool", request_key)

    def test_cloudflare_challenge_fails_closed_before_cookie_updates(self):
        self.assert_contracts([
            "onInteractionRequired: @escaping @Sendable (URL?) -> Void",
            "reportInteractionRequired(challengedURL)",
            "didReportInteractionRequired",
        ])
        self.assertIn("IPadMediaURLResolver.interactionChallengeError(", self.resolver)
        self.assertIn("destinationURL: destination", self.resolver)
        response_body = self.resolver.split(
            "didReceive response: URLResponse", 1
        )[1].split("didReceive data: Data", 1)[0]
        self.assertLess(
            response_body.index("interactionChallengeError"),
            response_body.index("completionHandler(.allow)"),
        )
        redirect_body = self.resolver.split(
            "willPerformHTTPRedirection response", 1
        )[1].split("didCompleteWithError", 1)[0]
        self.assertLess(
            redirect_body.index("interactionChallengeError"),
            redirect_body.index("updateCookies(from: response)"),
        )

    def test_hls_lines_and_uri_attributes_are_recursively_rewritten(self):
        self.assert_contracts([
            "let isPlaylist: Bool",
            "rewritePlaylist(",
            "rewriteURIAttributes(",
            '#"URI\\s*=\\s*\\\"([^\\\"]+)\\\""#',
            "mappedLocalURL(",
            'uppercased.hasPrefix("#EXT-X-STREAM-INF:")',
            'uppercased.hasPrefix("#EXT-X-MEDIA:")',
            'pathExtension == "m3u8"',
            "relativeTo: finalURL",
            "application/vnd.apple.mpegurl",
            "Cache-Control", "no-store",
        ])

    def test_master_variant_key_map_and_segment_keep_context_and_url_semantics(self):
        self.assert_contracts([
            "let context: IPadMediaRequestContext?",
            "context: entry.context",
            "entry.context?.applying(to: &request)",
            "await entry.context?.updateCookies(from: payload.response)",
            "relativeTo: finalURL",
            "URL(string: reference, relativeTo: baseURL)?.absoluteURL",
            'uppercased.hasPrefix("#EXT-X-STREAM-INF:")',
            'uppercased.hasPrefix("#EXT-X-MEDIA:")',
            'uppercased.hasPrefix("#EXT-X-I-FRAME-STREAM-INF:")',
            'uppercased.hasPrefix("#EXT-X-RENDITION-REPORT:")',
            '#"URI\\s*=\\s*\\"([^\\"]+)\\""#',
            "let pathExtension = safeURL.pathExtension.lowercased()",
            'pathExtension == "m3u8" || pathExtension == "m3u"',
            "appendingHLSDeliveryDirectives",
            'let separator = safeURL.query == nil ? "?" : "&"',
            "base + separator + directives",
        ])
        redirect = self.resolver.split(
            "willPerformHTTPRedirection response", 1
        )[1].split("didCompleteWithError", 1)[0]
        for header_contract in [
            "redirected.url = destination",
            "redirected.httpMethod = entry.request.httpMethod",
            'entry.request.value(forHTTPHeaderField: "Range")',
            "entry.options.requestContext?.applying(to: &redirected)",
        ]:
            self.assertIn(header_contract, redirect)

    def test_selected_master_preserves_external_audio_group(self):
        self.assert_contracts([
            "forSelectedHLSMaster metadata: IPadHLSMasterMetadata",
            "metadata.syntheticPlaylist(",
            "localAudioURLs[originAudioURL] = localAudioURL",
            "syntheticPlaylist: playlistData",
        ])
        for contract in [
            "struct IPadHLSMasterMetadata",
            "let audioRenditions: [IPadHLSAudioRendition]",
            "#EXT-X-MEDIA:",
            'streamAttributes.append("AUDIO=\\"\\(escaped)\\"")',
        ]:
            self.assertIn(contract, self.resolver)
        self.assertIn("source.hlsPlaylist?.masterMetadata", self.store)
        self.assertIn("masterMetadata.hasSeparateAudio", self.store)
        self.assertIn("forSelectedHLSMaster: masterMetadata", self.store)

    def test_runtime_proxy_relays_authenticated_hls_tree_for_avplayer(self):
        if sys.platform != "darwin":
            self.skipTest("authenticated media proxy probe requires macOS")
        xcrun = shutil.which("xcrun")
        swiftc = shutil.which("swiftc")
        if xcrun:
            compiler_command = [xcrun, "swiftc"]
        elif swiftc:
            compiler_command = [swiftc]
        else:
            compiler_command = []
        if not compiler_command:
            self.skipTest("Swift compiler is required")

        with tempfile.TemporaryDirectory(prefix="mioh-media-proxy-") as directory:
            executable = Path(directory) / "media-proxy-harness"
            build = subprocess.run(
                compiler_command
                + [
                    "-parse-as-library",
                    str(RESOLVER),
                    str(PROXY),
                    str(HARNESS),
                    "-o",
                    str(executable),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            self.assertEqual(
                build.returncode,
                0,
                f"authenticated media proxy harness did not compile:\n{build.stdout}",
            )
            completed = subprocess.run(
                [str(executable)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"authenticated media proxy harness failed:\n{completed.stdout}",
        )
        self.assertIn(
            "iPad authenticated media proxy probe passed",
            completed.stdout,
        )

    def test_playlist_probes_fetch_and_rewrite_the_complete_manifest(self):
        self.assert_contracts([
            'entry.isPlaylist && localRequest.method == "HEAD"',
            '? "GET"',
            'let originRange = entry.isPlaylist ? nil : localRequest.range',
            'bodyCount: transformedData.count',
            'isPlaylist: entry.isPlaylist || isRewrittenPlaylist',
            "if let resourceLoader,",
            "payload = IPadAuthenticatedMediaOriginPayload(",
            "rateLimitRetryUsesSharedCooldown = false",
        ])
        self.assertNotIn("if !entry.isPlaylist, let resourceLoader", self.source)
        self.assertIn("isPlaylist: true", self.store)

    def test_transient_429_is_retried_without_leaking_to_avplayer(self):
        retry = self.source.split(
            "private func fetchWithOriginPermit(", 1
        )[1].split("private func rewritePlaylist(", 1)[0]
        for contract in [
            "private static let maximumRateLimitRetryCount = 2",
            "var rateLimitRetryCount = 0",
            "while true",
            "try Task.checkCancellation()",
            "recordFetchStarted()",
            "response.statusCode == 429",
            'request.method == "GET" || request.method == "HEAD"',
            "rateLimitRetryCount < Self.maximumRateLimitRetryCount",
            "rateLimitRetryCount += 1",
            "catch let error as IPadHLSResourceLoadingError",
            "error == .attemptedUnavailable",
            "if !response.rateLimitRetryUsesSharedCooldown",
            "relayedRateLimitRetryDelays",
            "try await Task.sleep(",
            "Shared native transport records Retry-After/the fallback host",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, self.source if contract.startswith("private static") else retry)
        self.assertEqual(retry.count("try await originRequestGate.acquire()"), 1)
        self.assertEqual(retry.count("await originRequestGate.release()"), 2)

    def test_local_http_surface_is_narrow(self):
        self.assert_contracts([
            'parts[0] == "GET" || parts[0] == "HEAD"',
            'parts[1].hasPrefix("/v1/")',
            'name == "host"',
            'name == "range"',
            "isSafeSingleByteRange",
            "Connection: close",
        ])

    def test_only_bounded_ll_hls_delivery_directives_are_forwarded(self):
        self.assert_contracts([
            "normalizedHLSDeliveryDirectives",
            'case "_HLS_msn", "_HLS_part":',
            'case "_HLS_skip":',
            'value == "YES" || value == "v2"',
            'values["_HLS_part"] == nil || values["_HLS_msn"] != nil',
            "appendingHLSDeliveryDirectives",
            "request.hlsDeliveryDirectives == nil || entry.isPlaylist",
            "hlsDeliveryDirectives: localRequest.hlsDeliveryDirectives",
        ])

    def test_privacy_safe_proxy_diagnostics_are_visible_on_player_failure(self):
        self.assert_contracts([
            "private struct DiagnosticState",
            "func diagnosticSummary() -> String",
            '"接続=\\(state.acceptedConnections)"',
            '"元HTTP=\\(state.lastOriginStatus.map(String.init)',
            "recordPlaylistRewritten()",
            "recordReplyCompleted()",
        ])
        self.assertIn("HLS診断: \\($0.diagnosticSummary())", self.store)
        diagnostics_body = self.source.split("func diagnosticSummary()", 1)[1].split(
            "func start()", 1
        )[0]
        for secret_name in ["cookie", "referer", "userAgent", "absoluteString"]:
            self.assertNotIn(secret_name, diagnostics_body)

    def test_playlist_age_and_unknown_head_length_are_handled(self):
        self.assert_contracts([
            'response.value(forHTTPHeaderField: "Age")',
            'headers.append(("Age", age))',
            'if isHead, !isPlaylist {',
        ])

    def test_proxy_keeps_the_tcp_message_open_until_the_body_is_sent(self):
        send_body = self.source.split(
            "private func send(_ response: LocalResponse", 1
        )[1].split("private func sendError", 1)[0]
        self.assertIn("let streamContext = NWConnection.ContentContext.finalMessage", send_body)
        self.assertIn("connection.batch", send_body)
        self.assertEqual(send_body.count("contentContext: streamContext"), 2)
        self.assertIn("content: response.body", send_body)
        self.assertIn("isComplete: false", send_body)
        self.assertIn("isComplete: true", send_body)
        self.assertIn("contentContext: .finalMessage", send_body)
        self.assertLess(
            send_body.index("isComplete: false"),
            send_body.rindex("content: response.body"),
        )
        self.assertLess(
            send_body.rindex("content: response.body"),
            send_body.rindex("isComplete: true"),
        )
        self.assertNotIn(
            "content: Data(header.utf8),\n      completion:",
            send_body,
        )

    def test_rewritten_plain_uris_materialize_whitespace_as_strings(self):
        self.assertIn(
            'let leading = String(line.prefix { $0 == " " || $0 == "\\t" })',
            self.source,
        )
        self.assertIn(
            'line.reversed().prefix { $0 == " " || $0 == "\\t" }.reversed()',
            self.source,
        )
        self.assertIn(
            "rewrittenLines.append(leading + localURL.absoluteString + trailing)",
            self.source,
        )
        self.assertNotIn('rewrittenLines.append("\\(leading)', self.source)

    def test_proxy_is_retained_by_the_realtime_player_and_in_the_ios_target(self):
        self.assertIn("IPadAuthenticatedMediaProxy.swift in Sources", self.project)
        for contract in [
            "private var authenticatedMediaProxy: IPadAuthenticatedMediaProxy?",
            "try await mediaProxy.start()",
            "let validatedPlaylistURL =",
            "source.hlsPlaylist?.url",
            "?? source.mediaURL",
            "proxiedPlaybackURL = try mediaProxy.localURL(",
            "for: validatedPlaylistURL",
            "appliesOriginContext: false",
            "retiringProxy?.stop()",
            "proxy?.stop()",
            "@Published private(set) var interactionRequiredURL: URL?",
            "requireBrowserInteraction(",
            "case .interactionRequired(let challengedURL):",
        ]:
            self.assertIn(contract, self.store, f"missing store contract: {contract}")
        for contract in [
            ".onChange(of: realtimePlayer.interactionRequiredURL)",
            "realtimePlayer.clearInteractionRequirement()",
            "openBrowserForInteraction(interactionURL.absoluteString)",
        ]:
            self.assertIn(contract, self.view, f"missing view contract: {contract}")


if __name__ == "__main__":
    unittest.main()
