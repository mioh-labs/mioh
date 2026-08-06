import plistlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packaging" / "macOS" / "standalone"
REMOTE_SOURCE = PACKAGE / "RemoteControlServer.swift"
STREAM_SOURCE = PACKAGE / "RemoteStreamingCoordinator.swift"
CLUSTER_SOURCE = PACKAGE / "MiohClusterController.swift"
APP_SOURCE = PACKAGE / "MiohApp.swift"
PLAYER_SOURCE = PACKAGE / "RealtimePlayer.swift"
BUILD_SCRIPT = PACKAGE / "build_app.sh"
INFO_PLIST = PACKAGE / "Info.plist"


class MiohRemoteControlTests(unittest.TestCase):
    def streaming_sources(self):
        """Return the HTTP and media halves as one contract surface.

        Streaming deliberately reuses the authenticated LAN listener, while
        segment production/lifetime lives in a separate coordinator.  Most
        security and HTTP guarantees therefore span both source files.
        """
        self.assertTrue(STREAM_SOURCE.is_file())
        return REMOTE_SOURCE.read_text() + "\n" + STREAM_SOURCE.read_text()

    def test_remote_is_opt_in_lan_only_and_bounded(self):
        source = REMOTE_SOURCE.read_text()

        for contract in [
            "@Published private(set) var enabled = false",
            "@Published var port = 8888",
            "parameters.acceptLocalOnly = true",
            "maximumConnections = 16",
            "maximumHeaderBytes = 32 * 1024",
            "maximumBodyBytes = 64 * 1024",
            'headers["Connection"] = "close"',
            '"http_pipelining_not_supported"',
            "let activeConnections = Array(self.connections.values)",
            "self.engineID == identifier",
            "isUsableIPv6",
        ]:
            self.assertIn(contract, source)
        self.assertNotIn("parameters.allowLocalEndpointReuse = true", source)

    def test_remote_uses_keychain_bearer_auth_for_every_api(self):
        source = REMOTE_SOURCE.read_text()

        for contract in [
            "SecRandomCopyBytes",
            "kSecClassGenericPassword",
            "kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly",
            "kSecUseDataProtectionKeychain",
            ".posixPermissions: 0o600",
            "constantTimeEquals",
            'path.hasPrefix("/api/v1/")',
            'request.headers["authorization"]',
            '"WWW-Authenticate": "Bearer realm=\\"mioh\\""',
        ]:
            self.assertIn(contract, source)
        self.assertLess(source.index('guard authorized(request)'), source.index('switch (request.method, path)'))
        self.assertNotIn('Access-Control-Allow-Origin', source)

    def test_remote_uses_short_human_friendly_access_code(self):
        source = REMOTE_SOURCE.read_text()

        for contract in [
            "static let compactLength = 12",
            'Array("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")',
            'groups.joined(separator: "-")',
            "RemoteControlAccessToken.canonicalize(String(pieces[1]))",
            'placeholder="ABCD-EFGH-JKLM"',
            "大文字小文字・ハイフンの有無は問いません",
        ]:
            self.assertIn(contract, source)
        self.assertNotIn("表示された64桁のトークン", source)

        # HLS capabilities and cluster worker credentials are independent and
        # intentionally retain their existing 64-hex security contract.
        self.assertIn("bytes.count == 64", source)

    def test_remote_exposes_the_full_authenticated_control_surface(self):
        source = REMOTE_SOURCE.read_text()

        for route in [
            "/api/v1/status",
            "/api/v1/playback/play",
            "/api/v1/playback/pause",
            "/api/v1/playback/toggle",
            "/api/v1/playback/stop",
            "/api/v1/playback/seek",
            "/api/v1/playback/volume",
            "/api/v1/playback/mute",
            "/api/v1/export/start",
            "/api/v1/export/stop",
            "/api/v1/settings",
            "/api/v1/assets/list",
            "/api/v1/assets/create-directory",
            "/api/v1/playback/source",
            "/api/v1/playback/original",
            "/api/v1/defaults/save",
            "/api/v1/defaults/load",
            "/api/v1/defaults/reset",
            "/api/v1/cluster/settings",
            "/api/v1/cluster/start",
            "/api/v1/cluster/stop",
        ]:
            self.assertIn(route, source)
        self.assertIn("?.lastPathComponent as Any", source)
        self.assertNotIn("?.path as Any", source)
        self.assertNotIn("/api/v1/log", source)

    def test_remote_settings_reuse_the_native_snapshot_and_validation(self):
        source = REMOTE_SOURCE.read_text()
        app = APP_SOURCE.read_text()

        for contract in [
            "runner.currentDefaultsSnapshot()",
            "JSONDecoder().decode(MiohUserDefaultsSnapshot.self",
            "runner.apply(defaults: snapshot)",
            "configurationRevision",
            'settings_locked_while_running',
            "runner.roiEnhancerModelOptions(for: enhancer)",
            "player.setBufferLimit(runner.previewBufferLimit)",
            '"availabilityByEngine"',
        ]:
            self.assertIn(contract, source)
        self.assertIn("func currentDefaultsSnapshot() -> MiohUserDefaultsSnapshot", app)
        self.assertIn("func apply(defaults snapshot: MiohUserDefaultsSnapshot)", app)
        self.assertIn("selectedPreviewDetectionModel = previewDetectionModel", app)
        self.assertNotIn("private func currentDefaultsSnapshot", app)
        self.assertNotIn("private func apply(defaults", app)

    def test_remote_model_options_report_runtime_availability(self):
        source = REMOTE_SOURCE.read_text()
        app = APP_SOURCE.read_text()

        for contract in [
            '"available": value.available',
            'option["reason"] = reason',
            "runner.restorationModelAvailability",
            "runner.detectionModelAvailability",
            "runner.roiEnhancerModelAvailability",
            "option.disabled=!choice.available&&choice.value!==current",
        ]:
            self.assertIn(contract, source)
        self.assertIn("struct MiohModelAvailability", app)
        self.assertIn("nativeRestorationAsset(resources: resources, model: model)", app)
        self.assertIn("nativeDetectionAsset(resources: resources, model: model)", app)

    def test_remote_model_validation_rechecks_when_engine_changes(self):
        source = REMOTE_SOURCE.read_text()

        self.assertIn("let engineChanged = requestedEngine != runner.restorationEngine", source)
        self.assertGreaterEqual(source.count("engineChanged ||"), 5)

    def test_native_detection_backend_suffix_is_an_execution_contract(self):
        app = APP_SOURCE.read_text()

        for contract in [
            'let requestsCoreAI = model.hasSuffix("-coreai")',
            'let requestsCoreML = model.hasSuffix("-coreml")',
            "if !requestsCoreAI, let coreML = coreMLAsset()",
            "if !requestsCoreML, let coreAI = coreAIAsset()",
        ]:
            self.assertIn(contract, app)

    def test_remote_cluster_node_mutations_are_locked_during_jobs(self):
        source = REMOTE_SOURCE.read_text()

        self.assertEqual(source.count('jsonError(409, "cluster_nodes_locked")'), 2)
        for route in [
            "/api/v1/cluster/node/select",
            "/api/v1/cluster/node/forget",
        ]:
            route_index = source.index(route)
            lock_index = source.index('jsonError(409, "cluster_nodes_locked")', route_index)
            self.assertLess(lock_index - route_index, 500)
            self.assertIn("!cluster.hasActiveWorkerJobs", source[route_index:lock_index])

    def test_remote_global_mutations_are_locked_during_worker_jobs(self):
        source = REMOTE_SOURCE.read_text()
        cluster = CLUSTER_SOURCE.read_text()

        self.assertIn("var hasActiveWorkerJobs: Bool", cluster)
        self.assertIn("workerAttempts.contains { !$0.state.isTerminal }", cluster)
        for route in [
            'case ("PATCH", "/api/v1/settings")',
            'case ("POST", "/api/v1/assets/create-directory")',
            'case ("POST", "/api/v1/defaults/save")',
            'case ("POST", "/api/v1/defaults/load")',
            'case ("POST", "/api/v1/defaults/reset")',
        ]:
            route_index = source.index(route)
            next_case = source.find("\n    case (", route_index + len(route))
            route_body = source[route_index : next_case if next_case >= 0 else None]
            self.assertIn("cluster?.hasActiveWorkerJobs != true", route_body)

    def test_remote_asset_browser_is_opaque_allowlisted_and_bounded(self):
        source = REMOTE_SOURCE.read_text()

        for contract in [
            'id: "asset-" + UUID().uuidString.lowercased()',
            "assetCatalog: [String: RemoteControlAsset]",
            "isAllowedAssetURL",
            "realpath",
            "canonicalAssetURL",
            ".isSymbolicLinkKey",
            ".skipsHiddenFiles",
            "let pageSize = 512",
            'object["offset"]',
            'object["query"]',
            "maximumAssetCount = 65_536",
            "preservesExistingSelection",
            "modelPackageExtensions",
            "videoExtensions",
            "directory_exists_or_outside_root",
            "validHostHeader",
        ]:
            self.assertIn(contract, source)
        # Absolute server paths are translated to opaque IDs before settings
        # leave the authenticated API.
        self.assertIn("assetizeSettings(&settings", source)
        self.assertIn("resolveSettingAssets(&settings)", source)
        self.assertNotIn("assetCatalog.removeAll", source)
        self.assertNotIn('"path": url.path', source)

    def test_remote_file_browser_has_search_and_pagination(self):
        source = REMOTE_SOURCE.read_text()

        for contract in [
            'id="assetSearch"',
            'id="assetPrevious"',
            'id="assetNext"',
            '"previousOffset"',
            '"nextOffset"',
            '"total"',
            "localizedStandardCompare",
        ]:
            self.assertIn(contract, source)

    def test_remote_preserves_existing_external_selections_without_broad_browsing(self):
        source = REMOTE_SOURCE.read_text()

        self.assertIn("preservesExistingSelection: true", source)
        self.assertIn("preservedSettingKeys.contains(key)", source)
        self.assertIn("while a removable volume is offline", source)
        self.assertIn("isAllowedAssetURL(asset.url) || asset.preservesExistingSelection", source)
        self.assertNotIn('roots.append((home, "ホーム"))', source)
        for label in ("ムービー", "デスクトップ", "ダウンロード", "書類"):
            self.assertIn(f'"{label}"', source)

    def test_remote_html_covers_every_persisted_setting_without_inner_html(self):
        source = REMOTE_SOURCE.read_text()
        snapshot_match = re.search(
            r"struct MiohUserDefaultsSnapshot: Codable \{(.*?)\n\}",
            APP_SOURCE.read_text(),
            re.S,
        )
        self.assertIsNotNone(snapshot_match)
        fields = re.findall(r"^\s*var\s+(\w+):", snapshot_match.group(1), re.M)
        self.assertEqual(len(fields), 74)
        for field in fields:
            self.assertIn(f"['{field}'", source, field)
        self.assertIn("replaceChildren", source)
        self.assertIn("textContent", source)
        self.assertNotIn("innerHTML", source)

    def test_remote_html_mirrors_the_native_nine_tab_interface(self):
        source = REMOTE_SOURCE.read_text()
        tabs = [
            ("basic", "基本"),
            ("processing", "分割"),
            ("restoration", "復元"),
            ("detection", "検出"),
            ("output", "出力"),
            ("memory", "メモリ"),
            ("settings", "設定"),
            ("playback", "再生"),
            ("log", "ログ"),
        ]
        indices = []
        for tab, label in tabs:
            contract = (
                f'class="tab-button" role="tab" data-tab="{tab}" '
                f'aria-controls="tab-{tab}"'
            )
            self.assertIn(contract, source)
            self.assertIn(f'id="tab-{tab}" class="tab-panel" role="tabpanel"', source)
            self.assertIn(f">{label}</button>", source)
            indices.append(source.index(f'data-tab="{tab}"'))
        self.assertEqual(indices, sorted(indices))
        self.assertIn('role="tablist"', source)
        self.assertIn("localStorage.setItem('mioh-active-tab',name)", source)
        self.assertIn("overflow-x:auto", source)
        self.assertIn(".tab-button{flex:0 0 auto}", source)

    def test_remote_tab_setting_groups_follow_the_native_sections(self):
        source = REMOTE_SOURCE.read_text()
        anchors = [
            ("basic", "inputPath"),
            ("processing", "noSplit"),
            ("restoration", "restorationModel"),
            ("detection", "detectionModel"),
            ("output", "encodingMode"),
            ("memory", "memoryCleanupInterval"),
            ("playback", "previewBufferLimit"),
        ]
        tab_starts = {
            tab: source.index(f"['{tab}',[")
            for tab, _ in anchors
        }
        ordered_tabs = [tab for tab, _ in anchors]
        for index, (tab, field) in enumerate(anchors):
            end = (
                tab_starts[ordered_tabs[index + 1]]
                if index + 1 < len(ordered_tabs)
                else source.index("function optionsFor")
            )
            self.assertIn(f"['{field}'", source[tab_starts[tab] : end])

        self.assertIn("function fieldVisible(key)", source)
        self.assertIn("function fieldDisabled(key)", source)
        self.assertIn('id="remoteLog"', source)
        self.assertNotIn("/api/v1/log", source)

    def test_app_wires_remote_to_existing_main_actor_controllers(self):
        app = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "@StateObject private var remoteControl = RemoteControlServer()",
            "remoteControl.attach(runner: runner, player: player)",
            'Section("ローカルネットワーク操作")',
            "remoteControl.setEnabled($0)",
            "remoteControl.regenerateToken()",
        ]:
            self.assertIn(contract, app)
        self.assertIn("func remotePlay(runner: RestorationRunner) -> Bool", player)
        self.assertIn("func remotePause() -> Bool", player)
        self.assertIn("func remoteToggle(runner: RestorationRunner) -> Bool", player)

    def test_packaging_links_security_and_declares_local_network_usage(self):
        script = BUILD_SCRIPT.read_text()
        plist = plistlib.loads(INFO_PLIST.read_bytes())

        self.assertIn('-framework Security', script)
        self.assertIn('"$PACKAGE_DIR/RemoteControlServer.swift"', script)
        self.assertIn("NSLocalNetworkUsageDescription", plist)
        self.assertTrue(plist["NSLocalNetworkUsageDescription"])
        for locale in ("en", "zh-Hant"):
            localized = PACKAGE / "Localizations" / f"{locale}.lproj" / "InfoPlist.strings"
            self.assertTrue(localized.is_file())
            self.assertIn("NSLocalNetworkUsageDescription", localized.read_text())

    def test_lan_hls_uses_authenticated_ticket_sessions(self):
        source = self.streaming_sources()

        self.assertIn("/api/v1/stream/session", source)
        self.assertIn("/api/v1/stream/status", source)
        self.assertIn("/api/v1/stream/stop", source)
        self.assertIn("/stream/", source)
        self.assertRegex(source, r"(?i)ticket")
        self.assertRegex(source, r"(?i)(SecRandomCopyBytes|UUID\(\))")
        self.assertIn("/stream/v1/", source)
        self.assertIn("index.m3u8", source)
        self.assertIn('"segment"', source)
        for contract in [
            "issueSession",
            "revokeSession",
            "revokeAllSessions",
            "statusJSON",
            "playlistData",
            "segmentURL",
        ]:
            self.assertIn(contract, source)

        # The long-lived bearer credential must never appear in an HLS URL or
        # playlist.  The opaque, short-lived ticket is the stream capability.
        self.assertNotRegex(source, r"(?i)[?&](token|bearer)=")
        self.assertNotRegex(source, r"(?i)#EXT[^\n]*(token|bearer)")
        self.assertNotIn("Access-Control-Allow-Origin", source)

    def test_lan_hls_manifest_is_live_private_and_not_path_bearing(self):
        source = self.streaming_sources()

        for contract in [
            "application/vnd.apple.mpegurl",
            "#EXTM3U",
            "#EXT-X-TARGETDURATION",
            "#EXT-X-MEDIA-SEQUENCE",
            "#EXTINF",
            "no-store",
        ]:
            self.assertIn(contract, source)
        self.assertRegex(source, r"(?i)(lastPathComponent|segment.*sequence)")
        self.assertNotRegex(source, r"(?i)#EXT[^\n]*file://")

    def test_lan_hls_media_supports_head_and_strict_single_byte_ranges(self):
        source = self.streaming_sources()

        for contract in [
            '"GET"',
            '"HEAD"',
            'headers["range"]',
            '"Accept-Ranges"',
            '"Content-Range"',
            "206",
            "416",
        ]:
            self.assertIn(contract, source)
        self.assertRegex(source, r"(?i)(multiple|comma|contains\(\"[,]\"\))")
        self.assertRegex(source, r"(?i)(suffix|bytes=-|isEmpty)")
        self.assertIn("RemoteHTTPByteRange.parse", source)

    def test_lan_hls_streams_files_in_chunks_without_materializing_segments(self):
        source = self.streaming_sources()

        self.assertIn("FileHandle(forReadingFrom:", source)
        self.assertRegex(source, r"(?:fileChunkBytes|readChunkSize)\s*=\s*256\s*\*\s*1024")
        self.assertRegex(source, r"(?i)(readChunkSize|chunkSize|read\(upToCount:)")
        self.assertRegex(source, r"(?i)(fileRange|byteRange|range)")
        self.assertNotIn("Data(contentsOf:", STREAM_SOURCE.read_text())
        self.assertNotIn("Data(contentsOf:", REMOTE_SOURCE.read_text())

    def test_lan_hls_has_bounded_atomic_rolling_segment_storage(self):
        source = STREAM_SOURCE.read_text()

        self.assertRegex(source, r"(?i)(maximum|max).*segments")
        self.assertRegex(source, r"(?i)(rolling|evict|removeOld|trim)")
        self.assertIn('appendingPathExtension("part")', source)
        self.assertRegex(source, r"(?i)(moveItem|replaceItem|rename)")
        self.assertRegex(source, r"(?i)(reset|generation)")
        self.assertRegex(source, r"(?i)(stop|cleanup|removeItem)")

    def test_lan_hls_muxes_video_copy_with_aac_audio(self):
        source = STREAM_SOURCE.read_text()
        compact = re.sub(r"\s+", "", source)

        self.assertIn('"-c:v","copy"', compact)
        self.assertIn('"-c:a","aac"', compact)
        self.assertRegex(source, r"(?i)(ffmpeg|Process\(\))")

    def test_lan_hls_is_wired_to_realtime_segment_and_generation_hooks(self):
        stream = STREAM_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()
        combined = stream + "\n" + player

        self.assertRegex(combined, r"(?i)(publish|append|register).*segment")
        self.assertIn("case .reset(let source):", stream)
        self.assertIn("reset(to: source)", stream)
        self.assertIn("cancelWorkAndRemoveMedia(revokeTickets: true)", stream)
        self.assertIn("state.source.generation == segment.generation", stream)
        self.assertIn("state.source.generation == pending.generation", stream)
        self.assertRegex(combined, r"(?i)(streaming|remoteStream)")

    def test_lan_hls_does_no_media_work_while_server_is_disabled(self):
        app = APP_SOURCE.read_text()

        self.assertIn(".onChange(of: remoteControl.enabled)", app)
        self.assertIn("setStreamingEventConsumer(remoteStreaming.eventConsumer())", app)
        self.assertIn("setStreamingEventConsumer(nil)", app)
        self.assertIn("remoteStreaming.stop()", app)
        # An unconditional consumer on appearance would remux every preview
        # segment even though LAN control remains opt-in and disabled.
        self.assertNotIn(
            "remoteControl.attachStreaming(remoteStreaming)\n      "
            "player.setStreamingEventConsumer(remoteStreaming.eventConsumer())",
            app,
        )

    def test_lan_hls_page_csp_and_build_include_media_support(self):
        remote = REMOTE_SOURCE.read_text()
        script = BUILD_SCRIPT.read_text()

        self.assertIn("media-src 'self'", remote)
        self.assertIn('"$PACKAGE_DIR/RemoteStreamingCoordinator.swift"', script)


if __name__ == "__main__":
    unittest.main()
