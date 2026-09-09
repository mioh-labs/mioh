import json
import plistlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_APP = ROOT / "apps" / "MiohRemote"
APP_SOURCE = REMOTE_APP / "MiohRemote"
PACKAGE_ROOT = ROOT / "packages" / "MiohRemoteKit"

PACKAGE_MANIFEST = PACKAGE_ROOT / "Package.swift"
PACKAGE_RESOLVED = PACKAGE_ROOT / "Package.resolved"
XCODE_PACKAGE_RESOLVED = (
    REMOTE_APP
    / "MiohRemote.xcodeproj"
    / "project.xcworkspace"
    / "xcshareddata"
    / "swiftpm"
    / "Package.resolved"
)
SFTP_MODELS = PACKAGE_ROOT / "Sources" / "MiohSFTPKit" / "MiohSFTPModels.swift"
SFTP_CREDENTIALS = (
    PACKAGE_ROOT / "Sources" / "MiohSFTPKit" / "MiohSFTPCredentialStore.swift"
)
SFTP_SESSION = PACKAGE_ROOT / "Sources" / "MiohSFTPKit" / "MiohSFTPSession.swift"
REMOTE_CREDENTIALS = (
    PACKAGE_ROOT / "Sources" / "MiohRemoteKit" / "RemoteCredentialStore.swift"
)
VENDORED_SFTP = PACKAGE_ROOT / "Sources" / "NIOSFTP"
VENDORED_SECURITY_TESTS = (
    PACKAGE_ROOT / "Tests" / "NIOSFTPSecurityTests" / "NIOSFTPSecurityTests.swift"
)

SFTP_UI = APP_SOURCE / "IPadSFTPBrowser.swift"
STANDALONE_VIEW = APP_SOURCE / "IPadStandaloneView.swift"
STANDALONE_STORE = APP_SOURCE / "IPadStandaloneStore.swift"
PROJECT = REMOTE_APP / "MiohRemote.xcodeproj" / "project.pbxproj"
INFO_PLIST = APP_SOURCE / "Info.plist"
README = REMOTE_APP / "README.md"
NOTICES = REMOTE_APP / "THIRD_PARTY_NOTICES.md"
LICENSES = REMOTE_APP / "ThirdPartyLicenses"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class MiohRemoteSFTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = _read(PACKAGE_MANIFEST)
        cls.models = _read(SFTP_MODELS)
        cls.credentials = _read(SFTP_CREDENTIALS)
        cls.session = _read(SFTP_SESSION)
        cls.remote_credentials = _read(REMOTE_CREDENTIALS)
        cls.vendored_files = sorted(VENDORED_SFTP.glob("*.swift"))
        cls.vendored = "\n".join(_read(path) for path in cls.vendored_files)
        cls.vendored_security_tests = _read(VENDORED_SECURITY_TESTS)
        cls.sftp_ui = _read(SFTP_UI)
        cls.standalone_view = _read(STANDALONE_VIEW)
        cls.standalone_store = _read(STANDALONE_STORE)
        cls.project = _read(PROJECT)
        cls.info = plistlib.loads(INFO_PLIST.read_bytes())
        cls.readme = _read(README)
        cls.notices = _read(NOTICES)

    def test_package_uses_fixed_swiftnio_and_a_vendored_sftp_target(self):
        self.assertIn(
            '.library(name: "MiohSFTPKit", targets: ["MiohSFTPKit"])',
            self.package,
        )
        for url, version in [
            ("https://github.com/apple/swift-nio.git", "2.101.3"),
            ("https://github.com/apple/swift-nio-ssh.git", "0.15.0"),
        ]:
            self.assertRegex(
                self.package,
                rf'\.package\(\s*url:\s*"{re.escape(url)}",'
                rf'\s*exact:\s*"{re.escape(version)}"\s*\)',
            )

        compact = re.sub(r"\s+", " ", self.package)
        self.assertRegex(
            compact,
            r'\.target\( name: "NIOSFTP", dependencies: \['
            r'.*?\.product\(name: "NIOCore", package: "swift-nio"\)'
            r'.*?\.product\(name: "NIOPosix", package: "swift-nio"\)'
            r'.*?\.product\(name: "NIOSSH", package: "swift-nio-ssh"\)',
        )
        self.assertRegex(
            compact,
            r'\.target\( name: "MiohSFTPKit", dependencies: \[ "NIOSFTP",',
        )
        self.assertIn(
            '.testTarget(name: "NIOSFTPSecurityTests", dependencies: ["NIOSFTP"])',
            self.package,
        )
        self.assertIn(
            '.testTarget(name: "MiohSFTPKitTests", dependencies: ["MiohSFTPKit"])',
            self.package,
        )
        for obsolete in ["SwiftSFTP", "PathWorks", "libssh2", "OpenSSL"]:
            self.assertNotIn(obsolete, self.package)

        for resolved_path in [PACKAGE_RESOLVED, XCODE_PACKAGE_RESOLVED]:
            resolved = json.loads(_read(resolved_path))
            pins = {
                pin["identity"].lower(): pin["state"].get("version")
                for pin in resolved["pins"]
            }
            self.assertEqual(pins["swift-nio"], "2.101.3", resolved_path)
            self.assertEqual(pins["swift-nio-ssh"], "0.15.0", resolved_path)
            self.assertEqual(pins["swift-crypto"], "4.5.1", resolved_path)
            self.assertFalse({"swiftsftp", "pathworks"} & pins.keys(), resolved_path)

    def test_vendored_parser_rejects_frames_and_counts_before_allocation(self):
        self.assertGreaterEqual(len(self.vendored_files), 10)
        for source in self.vendored_files:
            self.assertIn("SPDX-License-Identifier: MIT", _read(source), source.name)

        limits = _read(VENDORED_SFTP / "ByteBuffer+SFTP.swift")
        inbound = _read(VENDORED_SFTP / "SFTPInboundPacketParser.swift")
        attributes = _read(VENDORED_SFTP / "SFTPAttributesCoding.swift")
        client_handler = _read(VENDORED_SFTP / "SFTPClientHandler.swift")
        server_handler = _read(VENDORED_SFTP / "SFTPServerHandler.swift")
        for contract in [
            "maximumFrameBytes: UInt32 = 16 * 1_024 * 1_024",
            "maximumNameEntries: UInt32 = 10_000",
            "maximumExtensions: UInt32 = 4_096",
            "guard length <= SFTPParsingLimits.maximumFrameBytes else",
        ]:
            self.assertIn(contract, limits)

        name_guard = inbound.index(
            "guard count <= SFTPParsingLimits.maximumNameEntries"
        )
        name_reserve = inbound.index("entries.reserveCapacity(Int(count))")
        self.assertLess(name_guard, name_reserve)
        self.assertIn("Int(count) <= payload.readableBytes / 12", inbound)
        self.assertIn(
            "extensions.count < Int(SFTPParsingLimits.maximumExtensions)", inbound
        )

        extended_guard = attributes.index(
            "guard extendedCount <= SFTPParsingLimits.maximumExtensions"
        )
        extended_reserve = attributes.index(
            "extensions.reserveCapacity(Int(extendedCount))"
        )
        self.assertLess(extended_guard, extended_reserve)
        self.assertIn("Int(extendedCount) <= self.readableBytes / 8", attributes)
        self.assertIn(
            "defer { self.inboundBuffer.discardReadBytes() }", client_handler
        )
        self.assertIn(
            "defer { self.inboundBuffer.discardReadBytes() }", server_handler
        )

        for regression in [
            "testOversizedFrameIsRejectedFromHeaderAlone",
            "maximumFrameBytes + 1",
            "testNameCountMustFitPayloadAndConfiguredLimit",
            "testExtendedAttributeCountMustFitPayloadAndConfiguredLimit",
            "UInt32.max",
        ]:
            self.assertIn(regression, self.vendored_security_tests)

    def test_sftp_password_and_host_key_use_a_separate_keychain_namespace(self):
        sftp_service = re.search(
            r'private static let service = "([^"]+)"', self.credentials
        )
        remote_service = re.search(
            r'private static let service = "([^"]+)"', self.remote_credentials
        )
        self.assertIsNotNone(sftp_service)
        self.assertIsNotNone(remote_service)
        self.assertEqual(sftp_service.group(1), "com.mioh-labs.MiohRemote.sftp")
        self.assertNotEqual(sftp_service.group(1), remote_service.group(1))

        for contract in [
            "kSecClassGenericPassword",
            "kSecUseDataProtectionKeychain",
            "kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly",
            "passwordAccount(endpointID:",
            "hostKeyAccount(endpointID:",
            '"password|\\(endpointID.lowercased())|\\(username)"',
            '"host-key|\\(endpointID.lowercased())"',
        ]:
            self.assertIn(contract, self.credentials)

    def test_host_key_is_collected_before_authentication_then_exactly_pinned(self):
        for contract in [
            "MiohSFTPRejectingAuthDelegate",
            "MiohSFTPCapturingHostKeyDelegate",
            "collector.capture(hostKey)",
            "validationCompletePromise.fail(MiohSFTPProbeComplete())",
            "MiohSFTPPinnedHostKeyDelegate",
            "guard hostKey == expected else",
            "serverAuthDelegate: MiohSFTPPinnedHostKeyDelegate(",
            "NIOSSHPublicKey(openSSHPublicKey: identity.key)",
            "withTaskCancellationHandler",
        ]:
            self.assertIn(contract, self.session)
        for forbidden in [
            ".acceptAny",
            ".acceptAll",
            "acceptAnything",
            "acceptUnconditionally",
        ]:
            self.assertNotIn(forbidden, self.session)

        for contract in [
            "SHA256.hash(data:",
            'fingerprint = "SHA256:\\(digest)"',
            "case hostKeyChanged(expected: String, actual: String)",
        ]:
            self.assertIn(contract, self.models)

    def test_directory_and_transfer_operations_are_bounded_and_no_follow(self):
        for contract in [
            "maximumDirectoryEntries = 5_000",
            "transferBufferBytes = 256 * 1_024",
            "operationTimeout: TimeInterval = 30",
            "connection.sftp.openDirectory(path: cleanPath)",
            "connection.sftp.readDirectoryBatch(handle)",
            "let entryLimit = max(0, min(maximumEntries, Self.maximumDirectoryEntries))",
            "entries.reserveCapacity(min(entryLimit, 256))",
            "guard batch.count <= entryLimit - receivedEntryCount else",
            "receivedEntryCount += batch.count",
            "connection.sftp.closeDirectory(handle)",
            "MiohSFTPTransferPolicy.validate(",
            "Task.checkCancellation()",
            "withTaskCancellationHandler",
            "channel.close(promise: nil)",
        ]:
            self.assertIn(contract, self.session)
        self.assertGreaterEqual(
            self.session.count("connection.sftp.closeDirectory(handle)"), 2
        )
        self.assertGreaterEqual(self.session.count("connection.sftp.lstat(path:"), 3)
        self.assertIn("guard kind != .symbolicLink", self.session)
        self.assertIn("case .symbolicLink, .regularFile, .unsupported:", self.session)
        self.assertNotIn("Data(contentsOf:", self.session)
        self.assertNotIn("readToEnd()", self.session)

        for contract in [
            'movieExtensions: Set<String> = ["mp4", "mov", "m4v"]',
            "hardMaximumBytes: Int64 = 20 * 1_024 * 1_024 * 1_024",
            "reserveBytes = max(1 * 1_024 * 1_024 * 1_024, availableBytes / 10)",
        ]:
            self.assertIn(contract, self.models)
        self.assertNotIn("startingPath.trimmingCharacters", self.sftp_ui)
        self.assertRegex(
            self.sftp_ui,
            r"MiohSFTPPath\.normalize\(\s*startingPath,\s*relativeTo: home",
        )

    def test_download_and_upload_publish_only_verified_complete_files(self):
        for contract in [
            'destinationURL.appendingPathExtension("part")',
            "fileManager.createFile(atPath: partialURL.path, contents: nil)",
            "connection.sftp.fstat(file: opened)",
            "buffer.readableBytes <= Int(pending.length)",
            "slots[slotIndex].data.append(",
            "offset: expectedSize",
            "length: 1",
            "try output.synchronize()",
            "fileManager.moveItem(at: partialURL, to: destinationURL)",
            "fileManager.removeItem(at: partialURL)",
            '".mioh-uploading-\\(UUID().uuidString.lowercased()).part"',
            "flags: [.write, .create, .exclusive]",
            "connection.sftp.write(file: opened, offset: offset, data: buffer)",
            "uploaded.size == UInt64(localSize)",
            "connection.sftp.supportsExtension(.hardlink)",
            "connection.sftp.hardlink(from: temporaryPath, to: cleanPath)",
            "connection.sftp.rename(from: temporaryPath, to: cleanPath)",
            "connection.sftp.remove(path: temporaryPath)",
            "awaitCleanup(",
        ]:
            self.assertIn(contract, self.session)

        for pipeline_contract in [
            "readRequestBytes = 64 * 1_024",
            "downloadReadPipelineDepth = 64",
            "readRequestBytes * downloadReadPipelineDepth",
            "maximumSlots: Self.downloadReadPipelineDepth",
            "future: connection.sftp.read(",
            "private static func readPipelinedRange(",
            "while slots.contains(where:",
            "for pending in pendingReads",
        ]:
            self.assertIn(pipeline_contract, self.session)
        self.assertNotIn("private static func completeRead(", self.session)
        self.assertGreaterEqual(
            self.session.count("guard try await !pathExists("), 2
        )
        self.assertGreaterEqual(
            self.session.count("connection.sftp.fstat(file: opened)"), 2
        )

    def test_ipad_ui_routes_sftp_downloads_and_completed_uploads(self):
        combined = self.sftp_ui + "\n" + self.standalone_view
        for contract in [
            "import MiohSFTPKit",
            "IPadSFTPBrowserView",
            "SecureField(",
            "MiohSFTPSession",
            "downloadMovie(",
            "uploadMovie(",
            "MiohSFTPTransferPolicy.maximumResumableDownloadBytes",
        ]:
            self.assertIn(contract, combined)
        for route in [
            "case sftpInput",
            "case sftpUpload",
            "SFTPから動画を選択",
            "完成MP4をSFTPへ送信",
            "mode: .selectInput",
            "mode: .upload(outputURL: outputURL)",
        ]:
            self.assertIn(route, self.standalone_view)
        self.assertIn("selectPersistentDownloadedInput", self.standalone_view)
        self.assertIn("selectPersistentDownloadedInput", self.standalone_store)

    def test_completed_downloads_are_visible_and_persistent_in_files(self):
        self.assertTrue(self.info.get("UIFileSharingEnabled"))
        self.assertTrue(self.info.get("LSSupportsOpeningDocumentsInPlace"))
        for contract in [
            'persistentDownloadDirectoryName = "SFTP Downloads"',
            "for: .documentDirectory",
            "private static func persistentDownloadURL(",
            "fileManager.moveItem(at: result, to: persistentURL)",
            "destinationOfSymbolicLink(atPath: candidate.path)",
            '"\\(baseName) (\\(attempt + 1)).\\(pathExtension)"',
            "このiPad内 > mioh Remote > SFTP Downloads",
        ]:
            self.assertIn(contract, self.sftp_ui)

        download_call = self.sftp_ui.index("self.session.downloadMovie(")
        publish_call = self.sftp_ui.index(
            "fileManager.moveItem(at: result, to: persistentURL)",
            download_call,
        )
        self.assertLess(download_call, publish_call)

        selection_start = self.standalone_store.index(
            "func selectPersistentDownloadedInput("
        )
        selection_end = self.standalone_store.index(
            "@discardableResult", selection_start
        )
        selection = self.standalone_store[selection_start:selection_end]
        self.assertIn("await selectInput(url)", selection)
        self.assertNotIn("downloadedRemoteInputURL = url", selection)
        self.assertNotIn("removeItem", selection)
        self.assertIn(
            "Filesの「このiPad内 > mioh Remote > SFTP Downloads」",
            self.readme,
        )

    def test_transfer_status_is_first_and_shows_bps_and_remaining_time(self):
        browser_start = self.sftp_ui.index("private var browser: some View")
        browser_end = self.sftp_ui.index("private func entryRow", browser_start)
        browser = self.sftp_ui[browser_start:browser_end]
        self.assertLess(
            browser.index("transferSection"),
            browser.index('Label("上のフォルダ"'),
        )
        for contract in [
            "transferBytesPerSecond",
            "estimatedRemainingSeconds",
            "ProcessInfo.processInfo.systemUptime",
            "instantaneousRate",
            "reportsByteProgress",
            'LabeledContent("\u901f\u5ea6"',
            'LabeledContent("\u6b8b\u308a\u6642\u9593"',
            'String(format: "%.1f Mbps"',
            'String(format: "%.0f Kbps"',
            'String(format: "%.0f bps"',
            "ByteCountFormatter.string(",
        ]:
            self.assertIn(contract, self.sftp_ui)

    def test_first_use_prompt_names_endpoint_and_busy_form_fields_are_disabled(self):
        for contract in [
            "MiohSFTPCredentialStore.loadHostKey",
            "MiohSFTPCredentialStore.saveHostKey",
            "pendingHostEndpoint = configuration.endpointID",
            '接続先: \\(store.pendingHostEndpoint ?? "不明")',
            "identity.algorithm",
            "identity.fingerprint",
            "信頼して接続",
            "hostKeyChanged(",
        ]:
            self.assertIn(contract, self.sftp_ui)

        form_start = self.sftp_ui.index("private var connectionForm")
        form_end = self.sftp_ui.index("private var browser", form_start)
        connection_form = self.sftp_ui[form_start:form_end]
        for field in [
            'TextField("ホスト名またはIPアドレス"',
            'TextField("ポート"',
            'TextField("ユーザー名"',
            'SecureField("パスワード"',
            'TextField("開始パス',
            'Toggle("パスワードをKeychainに保存"',
        ]:
            self.assertIn(field, connection_form)
        self.assertRegex(
            connection_form,
            r'Section\("接続先"\)\s*\{[\s\S]+?\}\s*\.disabled\(store\.isBusy\)',
        )
        self.assertGreaterEqual(connection_form.count(".disabled(store.isBusy)"), 3)

        compact_ui = re.sub(r"\s+", " ", self.sftp_ui)
        self.assertNotRegex(
            compact_ui,
            r"UserDefaults\.standard\.set\([^)]*(password|configuration\.password)",
        )

    def test_download_resume_is_identity_bound_and_integrity_checked(self):
        for contract in [
            "MiohSFTPDownloadResumeMetadata",
            "resumeExistingPartial: Bool = false",
            'destinationURL.appendingPathExtension("resume")',
            "saved == metadata",
            "output.seekToEnd()",
            "progress(Int64(offset), Int64(expectedSize))",
            "openedAttributes.size == expectedSize",
            "openedAttributes.modificationTime == pathAttributes.modificationTime",
            "finalAttributes.size == expectedSize",
            "finalAttributes.modificationTime == pathAttributes.modificationTime",
            "let completedURL = URL(fileURLWithPath: partialURL.path)",
            "completedValues.isRegularFile == true",
            "completedValues.isSymbolicLink != true",
            "shouldPreservePartial(after: error)",
        ]:
            self.assertIn(contract, self.session)

        for contract in [
            'resumableDownloadDirectoryName = "SFTP Resume"',
            'resumableDownloadPrefix = "mioh-sftp-resume-"',
            "configuration.credentialID",
            "trustedHostKey",
            "SHA256.hash(data:",
            "resumeExistingPartial: true",
            "maximumResumableDownloadBytes(",
            "resumedBytes",
            "removeExpiredResumeFiles(",
            "resumeRetentionSeconds: TimeInterval = 30 * 24 * 60 * 60",
            "values.isSymbolicLink != true",
        ]:
            self.assertIn(contract, self.sftp_ui)

    def test_movie_metadata_is_loaded_on_demand_without_full_download(self):
        for contract in [
            "import AVFoundation",
            "struct IPadSFTPMovieMetadata",
            "func inspectMetadata(_ entry: MiohSFTPEntry)",
            "IPadSFTPStreamingInput.start(",
            "IPadSFTPMovieMetadata.load(",
            "let asset = AVURLAsset(url: input.localURL)",
            "asset.load(.duration)",
            "asset.loadTracks(withMediaType: .video)",
            "videoTrack.load(.naturalSize)",
            "videoTrack.load(.preferredTransform)",
            "videoTrack.load(.nominalFrameRate)",
            "videoTrack.load(.formatDescriptions)",
            "hasAudio: !audioTracks.isEmpty",
            "maximumCachedMovieMetadataEntries = 256",
            "movieMetadataByPath",
            "IPadSFTPMovieMetadataView",
            'Label("動画情報を確認", systemImage: "info.circle")',
            "metadata.inlineSummary",
            "entry.modifiedAt",
        ]:
            self.assertIn(contract, self.sftp_ui)

        inspect_block = self.sftp_ui.split(
            "func inspectMetadata(_ entry: MiohSFTPEntry)", 1
        )[1].split("func cachedMetadata", 1)[0]
        self.assertIn("input.stop()", inspect_block)
        self.assertNotIn("downloadMovie(", inspect_block)

    def test_background_grace_then_lock_or_expiration_disconnects(self):
        for contract in [
            "@Environment(\\.scenePhase)",
            "if phase == .background",
            "store.enterBackground()",
            "store.enterForeground()",
            "beginBackgroundTask(",
            "idleBackgroundGraceNanoseconds: UInt64 = 20_000_000_000",
            "backgroundGraceExpired()",
            "UIApplication.protectedDataWillBecomeUnavailableNotification",
            ".onDisappear",
            "store.cancelAndDisconnect()",
            ".interactiveDismissDisabled(store.isBusy)",
            "task?.cancel()",
            "await task?.value",
            "await session.disconnect()",
        ]:
            self.assertIn(contract, self.sftp_ui)
        self.assertGreaterEqual(self.sftp_ui.count("store.cancelAndDisconnect()"), 2)
        self.assertIn("if phase == .background", self.standalone_view)
        self.assertIn("sftp.enterBackground()", self.standalone_view)
        self.assertIn("sftp.enterForeground()", self.standalone_view)
        self.assertIn("sftp.cancelAndDisconnect()", self.standalone_view)

    def test_startup_cleanup_only_removes_owned_old_regular_temp_files(self):
        for contract in [
            "processTemporaryFileCutoff = Date()",
            'temporaryInputPrefix = "mioh-sftp-input-"',
            "maximumTemporaryEntriesToInspect = 4_096",
            "maximumTemporaryFilesToRemove = 256",
            "Self.removeOrphanedInputFiles(before: Self.processTemporaryFileCutoff)",
            "options: [.skipsSubdirectoryDescendants]",
            "url.deletingLastPathComponent().standardizedFileURL == temporaryDirectory",
            "isOwnedTemporaryInputName(url.lastPathComponent)",
            "values.isRegularFile == true",
            "values.isSymbolicLink != true",
            "modifiedAt < cutoff",
            'for pathExtension in ["mp4", "mov", "m4v"]',
            'for suffix in [".\\(pathExtension)", ".\\(pathExtension).part"]',
            "UUID(uuidString: identifier) != nil",
        ]:
            self.assertIn(contract, self.sftp_ui)
        self.assertNotIn("fileManager.removeItem(at: temporaryDirectory)", self.sftp_ui)

    def test_xcode_links_browser_and_package_and_bundles_license_resources(self):
        self.assertGreaterEqual(self.project.count("IPadSFTPBrowser.swift"), 3)
        self.assertGreaterEqual(self.project.count("MiohSFTPKit"), 3)
        for contract in [
            "MiohSFTPKit in Frameworks",
            "productName = MiohSFTPKit;",
            "relativePath = ../../packages/MiohRemoteKit;",
            "THIRD_PARTY_NOTICES.md in Resources",
            "ThirdPartyLicenses in Resources",
            "lastKnownFileType = folder; path = ThirdPartyLicenses;",
        ]:
            self.assertIn(contract, self.project)
        self.assertGreaterEqual(
            self.project.count("THIRD_PARTY_NOTICES.md in Resources"), 2
        )
        self.assertGreaterEqual(
            self.project.count("ThirdPartyLicenses in Resources"), 2
        )

    def test_info_plist_and_readme_document_the_operational_safety_contract(self):
        with INFO_PLIST.open("rb") as file:
            info = plistlib.load(file)
        usage = info["NSLocalNetworkUsageDescription"]
        self.assertIn("SFTP", usage.upper())
        self.assertTrue(info["NSAppTransportSecurity"]["NSAllowsLocalNetworking"])

        readme_lower = self.readme.lower()
        for term in ["sftp", "keychain", "lstat", ".part", "vendored", "frame"]:
            self.assertIn(term, readme_lower)
        for term in [
            "ホスト鍵",
            "フィンガープリント",
            "シンボリックリンク",
            "次回起動時",
            "画面ロック",
            "バックグラウンド",
            "一覧件数",
            "拡張属性件数",
        ]:
            self.assertIn(term, self.readme)
        self.assertIn("SwiftNIO/SwiftNIO SSH", self.readme)
        self.assertIn("THIRD_PARTY_NOTICES.md", self.readme)
        self.assertIn("ThirdPartyLicenses", self.readme)
        self.assertNotIn("SwiftSFTP", self.readme)

    def test_notices_and_bundled_license_texts_match_resolved_dependencies(self):
        for component in [
            "swift-nio-sftp",
            "748e88568638463370816dbaaaf9684237258501",
            "SwiftNIO 2.101.3",
            "SwiftNIO SSH 0.15.0",
            "Swift Crypto 4.5.1",
            "Swift ASN.1 1.7.1",
            "Swift Atomics 1.3.1",
            "Swift Collections 1.6.0",
            "Swift System 1.8.1",
            "bounded frame, directory-entry, and extended-attribute parsing",
        ]:
            self.assertIn(component, self.notices)
        for obsolete in ["SwiftSFTP 3.4.2", "PathWorks", "libssh2", "OpenSSL"]:
            self.assertNotIn(obsolete, self.notices)

        expected_files = {
            "swift-nio-sftp-MIT.txt",
            "swift-nio-Apache-2.0.txt",
            "swift-nio-NOTICE.txt",
            "swift-nio-ssh-Apache-2.0.txt",
            "swift-crypto-Apache-2.0.txt",
            "swift-crypto-NOTICE.txt",
            "swift-asn1-Apache-2.0.txt",
            "swift-asn1-NOTICE.txt",
            "swift-atomics-Apache-2.0.txt",
            "swift-collections-Apache-2.0.txt",
            "swift-system-Apache-2.0.txt",
        }
        actual_files = {path.name for path in LICENSES.iterdir() if path.is_file()}
        self.assertTrue(expected_files <= actual_files)
        self.assertIn(
            "Permission is hereby granted", _read(LICENSES / "swift-nio-sftp-MIT.txt")
        )
        for filename in expected_files:
            text = _read(LICENSES / filename)
            self.assertGreater(len(text.strip()), 100, filename)
            if "Apache-2.0" in filename:
                self.assertIn("Apache License", text, filename)


if __name__ == "__main__":
    unittest.main()
