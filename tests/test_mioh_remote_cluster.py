import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "RemoteClusterService.swift"
CONTROLLER = ROOT / "packaging" / "macOS" / "standalone" / "MiohClusterController.swift"
RUNNER = ROOT / "packaging" / "macOS" / "standalone" / "MiohApp.swift"
PIPELINE = ROOT / "packaging" / "macOS" / "standalone" / "NativePreviewPipeline.swift"
HTTP_TRANSFER = (
    ROOT / "packaging" / "macOS" / "standalone" / "RemoteClusterHTTPTransfer.swift"
)
RANGE_ASSET = (
    ROOT
    / "packages"
    / "MiohRemoteKit"
    / "Sources"
    / "MiohRemoteKit"
    / "MiohHTTPRangeAsset.swift"
)
AV_RANGE_HARNESS = ROOT / "tests" / "swift" / "RemoteClusterAVRangeHarness.swift"
BUILD = ROOT / "packaging" / "macOS" / "standalone" / "build_app.sh"


class MiohRemoteClusterFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.controller = CONTROLLER.read_text()
        cls.runner = RUNNER.read_text()
        cls.pipeline = PIPELINE.read_text()
        cls.http_transfer = HTTP_TRANSFER.read_text()
        cls.range_asset = RANGE_ASSET.read_text()
        cls.av_range_harness = AV_RANGE_HARNESS.read_text()
        cls.build = BUILD.read_text()

    def test_cluster_negotiates_http_transfer_with_shared_root_fallback(self):
        for contract in [
            'case sharedRootV1 = "shared-root-v1"',
            'case coordinatorHTTPV1 = "coordinator-http-v1"',
            "struct RemoteClusterHTTPTransferDescriptor: Codable, Hashable, Sendable",
            "let supportedTransferModes: [RemoteClusterTransferMode]?",
            "var effectiveTransferModes: Set<RemoteClusterTransferMode>",
            "let httpTransfer: RemoteClusterHTTPTransferDescriptor?",
            "inputRelativePath: RemoteClusterRelativePath",
            "outputRelativePath: RemoteClusterRelativePath",
            "sharedRootIdentifier",
            "beneath sharedRoot: URL",
        ]:
            self.assertIn(contract, self.source)
        self.assertIn("modes.contains(.coordinatorHTTPV1)", self.controller)
        self.assertIn("modes.contains(.sharedRootV1)", self.controller)

    def test_http_transfer_is_bounded_streamed_and_capability_scoped(self):
        for contract in [
            "SecRandomCopyBytes",
            "count: 32",
            'components.path = "/mioh-cluster/v1/\\(ticket)/\\(leaf)"',
            '("HEAD", let value) where value.hasPrefix("input.")',
            '("GET", let value) where value.hasPrefix("input.")',
            '("PUT", "output.mp4")',
            '"Accept-Ranges": "bytes"',
            'headers["Content-Range"] = "bytes \\(byteRange.lowerBound)-\\(byteRange.upperBound)/\\(size)"',
            "expected <= binding.maximumOutputBytes",
            "O_RDONLY | O_CLOEXEC | O_NOFOLLOW",
            "fstat(fileDescriptor, &value)",
            "return pread(binding.source.fileDescriptor",
            "FileHandle(forWritingTo:",
            "hasher.update(data:",
            "FileManager.default.moveItem(at: part, to: binding.outputURL)",
            "session.upload(for: request, fromFile: file)",
            "URLSessionConfiguration.ephemeral",
            "configuration.timeoutIntervalForResource = 24 * 60 * 60",
            "Set([408, 425, 429])",
            'sendStatus(425, "Too Early"',
            "maximumAcceptedConnections = 128",
            "scheduleUploadIdleTimeout",
            "maximumAggregateOutputBytes",
            'sendStatus(507, "Insufficient Storage"',
        ]:
            self.assertIn(contract, self.http_transfer)
        self.assertNotIn("AVURLAssetHTTPHeaderFieldsKey", self.http_transfer + self.pipeline)
        self.assertNotIn("multipart/form-data", self.http_transfer)

    def test_http_transfer_is_built_and_native_pipeline_accepts_remote_assets(self):
        self.assertIn('"$PACKAGE_DIR/RemoteClusterHTTPTransfer.swift"', self.build)
        self.assertIn(
            '"$ROOT/packages/MiohRemoteKit/Sources/MiohRemoteKit/MiohHTTPRangeAsset.swift"',
            self.build,
        )
        for contract in [
            "parsed.scheme?.lowercased() == \"http\"",
            'parsed.scheme?.lowercased() == "https"',
            "sourceURL = parsed",
            "MiohHTTPRangeAsset(",
            "expectedRemoteByteCount: config.inputByteCount",
            "expectedRemoteSHA256: config.inputSHA256",
            "HTTPクラスタ入力はAVFoundation互換のmp4/mov/m4vのみ対応しています",
        ]:
            self.assertIn(contract, self.pipeline)

    def test_avfoundation_http_bridge_is_paged_strict_and_regression_tested(self):
        for contract in [
            "AVAssetResourceLoaderDelegate",
            "defaultPageBytes = 64 * 1_024",
            "let loadingRequest: AVAssetResourceLoadingRequest",
            'request.setValue("bytes=\\(start)-\\(end)"',
            'http.statusCode == 206',
            'http.value(forHTTPHeaderField: "Content-Range") == expectedContentRange',
            'http.value(forHTTPHeaderField: "Content-Length") == String(count)',
            'http.value(forHTTPHeaderField: "ETag") == self.expectedETag',
            "willPerformHTTPRedirection",
            "completionHandler(nil)",
            "didCancel loadingRequest",
            "session.invalidateAndCancel()",
        ]:
            self.assertIn(contract, self.range_asset)
        self.assertIn("for fastStart in [true, false]", self.av_range_harness)
        self.assertIn("copyNextSampleBuffer()", self.av_range_harness)

    def test_relative_paths_reject_escape_and_existing_symlinks(self):
        for contract in [
            '!value.hasPrefix("/")',
            '!value.hasPrefix("~")',
            '!value.contains("\\\\")',
            '$0 != "." && $0 != ".."',
            ".isSymbolicLinkKey",
            "RemoteClusterPathError.symbolicLink",
            "RemoteClusterPathError.outsideSharedRoot",
        ]:
            self.assertIn(contract, self.source)

    def test_bonjour_discovers_mioh_workers_without_credentials_in_txt(self):
        for contract in [
            'static let serviceType = "_mioh-worker._tcp"',
            ".bonjourWithTXTRecord(",
            "RemoteClusterBonjourMetadata(txtRecord: txtRecord)",
            '"transfer": transferMode.rawValue',
            '"root": sharedRootIdentifier',
            "sensitiveKeys",
            "RemoteClusterBonjourError.sensitiveField",
        ]:
            self.assertIn(contract, self.source)
        txt_record_body = self.source.split("var txtRecord: NWTXTRecord", 1)[1].split(
            "var txtRecordData", 1
        )[0]
        for secret in ['"token"', '"authorization"', '"credential"', '"password"']:
            self.assertNotIn(secret, txt_record_body)

    def test_http_bonjour_accepts_legacy_workers_that_omit_shared_root(self):
        self.assertIn('let root = normalized["root"] ?? ""', self.source)
        self.assertIn(
            "(parsedTransfer == .coordinatorHTTPV1 || !root.isEmpty)",
            self.source,
        )

    def test_cluster_uses_trusted_lan_without_pairing_credentials(self):
        combined = self.source + self.controller + self.runner
        for removed in [
            "RemoteClusterCredentialProviding",
            "credentialProvider",
            "credentialDrafts",
            "normalizedWorkerCredential",
            'Button("認証")',
        ]:
            self.assertNotIn(removed, combined)
        self.assertIn("No pairing code is required", self.source)

    def test_job_contract_is_typed_leased_and_idempotent(self):
        for contract in [
            "struct RemoteClusterJobRequest: Codable, Hashable, Sendable",
            "struct RemoteClusterCapabilities: Codable, Hashable, Sendable",
            "struct RemoteClusterNodeStatus: Codable, Hashable, Sendable",
            "struct RemoteClusterAttemptRecord: Codable, Hashable, Identifiable, Sendable",
            "job_attempt_already_active",
            "request.leaseExpiresAt > now",
            "case duplicate",
            "Exact-attempt retries",
            "func renewLease(",
            "func expireLeases(",
            "decodeStartNanoseconds: Int64",
            "coreEndNanoseconds: Int64",
            "inputByteCount: Int64",
            "inputSHA256: String",
            "restorationAssetSHA256: String",
            "detectorAssetSHA256: String",
        ]:
            self.assertIn(contract, self.source)

    def test_worker_capability_extensions_are_optional_and_preflighted(self):
        for contract in [
            "let maximumRestorationClipLength: Int?",
            "let supportsROIEnhancer: Bool?",
            "let supportsRestorationEffects: Bool?",
            "let supportedInputExtensions: [String]?",
            "let restorationAssetSHA256ByIdentifier: [String: String]?",
            "let detectorAssetSHA256ByIdentifier: [String: String]?",
            "maximumRestorationClipLength: Int? = nil",
            "supportsROIEnhancer: Bool? = nil",
            "supportsRestorationEffects: Bool? = nil",
            "supportedInputExtensions: [String]? = nil",
        ]:
            self.assertIn(contract, self.source)
        for contract in [
            "workerSupports(",
            "options.restorationClipLength > maximum",
            "capabilities.supportsROIEnhancer == false",
            "capabilities.supportsRestorationEffects == false",
            "capabilities.supportedInputExtensions",
            "capabilities.restorationAssetSHA256ByIdentifier?[",
        ]:
            self.assertIn(contract, self.controller)
        self.assertIn(
            "capabilities.detectorAssetSHA256ByIdentifier?[",
            self.controller,
        )
        self.assertIn("clusterCanonicalAssetIdentityMaps()", self.runner)

    def test_manager_exposes_coordinator_and_worker_state(self):
        for contract in [
            "final class RemoteClusterService: ObservableObject",
            "@Published private(set) var discoveredNodes",
            "@Published private(set) var workerSharedRoot",
            "@Published private(set) var workerState",
            "func startCoordinatorDiscovery()",
            "func startWorker(",
            "launcher: @escaping RemoteClusterJobLauncher",
            "transport.accept(connection)",
        ]:
            self.assertIn(contract, self.source)

    def test_runner_is_an_explicit_future_integration_closure(self):
        for contract in [
            "typealias RemoteClusterJobLauncher",
            "launcher: RemoteClusterJobLauncher",
            "try await launcher(request, inputURL, candidateStagingURL)",
            "future runner integration point",
        ]:
            self.assertIn(contract, self.source)
        self.assertNotIn("Process()", self.source)

    def test_rpc_is_bounded_trusted_lan_and_single_request(self):
        for contract in [
            "static let maximumJSONBytes = 64 * 1024",
            "one length-prefixed JSON request and one reply",
            "No pairing code is required",
            "connection.cancel()",
            "case capabilities",
            "case submit(RemoteClusterJobRequest)",
            "case status(RemoteClusterStatusQuery)",
            "case cancel(RemoteClusterCancelRequest)",
            "case renewLease(RemoteClusterRenewLeaseRequest)",
        ]:
            self.assertIn(contract, self.source)

    def test_worker_verifies_input_and_publishes_output_atomically(self):
        for contract in [
            "import CryptoKit",
            "SHA256()",
            "let chunkBytes = 4 * 1024 * 1024",
            "RemoteClusterInputIntegrityCache",
            "inputIntegrity.signature.byteCount == request.inputByteCount",
            "inputIntegrity.sha256 == request.inputSHA256",
            "O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC",
            "withIntermediateDirectories: true",
            "candidateStagingURL",
            "FileManager.default.moveItem(at: candidateStagingURL, to: outputURL)",
            'case .outputExists: "output_exists"',
        ]:
            self.assertIn(contract, self.source)

    def test_lease_renewal_is_bounded_and_coordinator_scoped(self):
        for contract in [
            "struct RemoteClusterRenewLeaseRequest",
            "newExpiration.timeIntervalSince(now) <= maximumLeaseSeconds",
            "existing?.coordinatorNodeID == request.coordinatorNodeID",
            "existing?.leaseID == renewal.leaseID",
            "let renewalAccepted = renewed || alreadyTerminal",
            '"lease_renewal_rejected"',
        ]:
            self.assertIn(contract, self.source)

    def test_planned_job_lease_is_refreshed_only_when_a_lane_takes_it(self):
        for contract in [
            "func withLeaseExpiration(_ expiration: Date) -> Self",
            "jobID: jobID",
            "attemptID: attemptID",
            "leaseID: leaseID",
            "createdAt: createdAt",
            "leaseExpiresAt: expiration",
            "let leasedRequest = job.request.withLeaseExpiration(",
            "Date().addingTimeInterval(14 * 60)",
            "request: leasedRequest",
            "submittedRequest = leasedRequest",
            "localWorkerLedger.submit(\n        leasedRequest,",
        ]:
            self.assertIn(contract, self.source + self.controller)

    def test_leases_expire_without_waiting_for_another_rpc(self):
        for contract in [
            "private var leaseWatchdogs",
            "scheduleLeaseWatchdog(attemptID: request.attemptID)",
            "while let record = self.record(attemptID: attemptID)",
            "self.expireLeases()",
            "jobLedger.cancelAllActive()",
        ]:
            self.assertIn(contract, self.source)

    def test_coordinator_fails_fast_and_rpc_observes_task_cancellation(self):
        for contract in [
            "withThrowingTaskGroup(of: [MiohClusterCompletedShard].self)",
            "while let completed = try await group.next()",
            "await cancelActiveAttempts()",
            "withTaskCancellationHandler",
            "RemoteClusterConnectionCancellation",
        ]:
            self.assertIn(contract, self.controller + self.source)

    def test_workers_are_automatically_verified_after_discovery(self):
        self.assertIn(
            "for node in nodes where !self.verifiedNodeIDs.contains(node.id)",
            self.controller,
        )
        self.assertIn("beginVerification(node, automatically: true)", self.controller)
        self.assertNotIn("node.hasStoredCredential", self.controller)

    def test_client_uses_discovered_endpoint_without_credential(self):
        for contract in [
            "final class RemoteClusterClient",
            "NWConnection(to: endpoint, using: .tcp)",
            "response.requestID == requestID",
            "interface != nil",
            "interface: nil",
        ]:
            self.assertIn(contract, self.source)
        self.assertNotIn("credentialProvider", self.source)

    def test_real_coordinator_and_worker_are_built_into_the_app(self):
        self.assertIn('"$PACKAGE_DIR/RemoteClusterService.swift"', self.build)
        self.assertIn('"$PACKAGE_DIR/MiohClusterController.swift"', self.build)
        for contract in [
            "service.startCoordinatorDiscovery()",
            "service.startWorker(",
            "runner.remoteClusterJobLauncher()",
            "func startExport(using runner: RestorationRunner)",
            "func clusterRestorationOptions() async throws",
            "func runRemoteClusterWorkerJob(",
        ]:
            self.assertIn(contract, self.controller + self.runner)

    def test_coordinator_can_join_as_one_local_worker_lane(self):
        for contract in [
            "@Published var useCoordinatorAsWorker: Bool",
            "lanes.append(.local)",
            "lanes.append(contentsOf: remoteWorkers.map",
            "let jobPool = MiohClusterJobPool(jobs: jobs)",
            "while let job = await jobPool.takeNext()",
            "localWorkerLedger.submit(",
            "sharedRootIdentifier: sharedRootIdentifier",
            "launcher: launcher",
            "localWorkerLedger.renewLease(",
            "localWorkerLedger.cancel(",
            "await localWorkerLedger.cancelAllActiveAndWait()",
            "useCoordinatorAsWorker || !selectedNodeIDs.isEmpty",
            'Toggle("このMacもWorkerとして使用"',
        ]:
            self.assertIn(contract, self.controller + self.runner)

    def test_cancelled_launcher_holds_capacity_until_teardown_finishes(self):
        for contract in [
            "var inFlightExecutionCount: Int { executionTasks.count }",
            "executionTasks[attemptID]?.cancel()",
            "func cancelAllActiveAndWait(now: Date = Date()) async",
            "if let task { await task.value }",
            "ledger.inFlightExecutionCount",
            "previous_attempt_stopping",
        ]:
            self.assertIn(contract, self.source)
        # runAccepted's defer is the only owner allowed to drop the handle.
        self.assertEqual(
            self.source.count("executionTasks.removeValue(forKey:"),
            1,
        )

    def test_remote_attempt_is_tracked_before_submit(self):
        registration = self.controller.index(
            "activeAttempts[job.request.attemptID] = MiohClusterActiveAttempt("
        )
        submit = self.controller.index(
            "let admission = try await client.call(.submit(submittedRequest), node: node)"
        )
        self.assertLess(registration, submit)
        self.assertNotIn("normalizedWorkerCredential", self.controller)
        self.assertNotIn('"64桁hex / ABCD-EFGH-JKLM"', self.runner)

    def test_shards_use_exact_pts_core_ranges_and_temporal_halo(self):
        for contract in [
            "AVAssetReaderTrackOutput(track: track, outputSettings: nil)",
            "timestamps.sort()",
            "let stride = temporalBatchFrames - overlap",
            "coreStart - stride",
            "coreEnd + overlap",
            "outputCoreStartNanoseconds",
            "outputCoreEndNanoseconds",
            "belongsToCoreOutput",
            "expectedCoreFrameCount: coreEnd - coreStart",
            "metrics.processedFrames == job.expectedCoreFrameCount",
        ]:
            self.assertIn(contract, self.controller + self.pipeline)

        # Halo frames must participate in restoration/crossfade first, and
        # only then be removed by the half-open core-range ownership filter.
        crossfade = self.pipeline.index("pixelBuffer: try processor.crossfade(")
        core_filter = self.pipeline.index(
            "let acceptedFrames = framesToEncode.filter", crossfade
        )
        self.assertLess(crossfade, core_filter)
        core_filter_body = self.pipeline[core_filter : core_filter + 900]
        self.assertIn("ptsNanoseconds < start", self.pipeline)
        self.assertIn("ptsNanoseconds >= end", self.pipeline)
        self.assertIn("belongsToCoreOutput(frame.ptsNanoseconds)", core_filter_body)

    def test_cluster_outputs_video_shards_and_muxes_source_audio_once(self):
        self.assertIn("finishWorkerExport(", self.pipeline)
        self.assertIn('"has_audio": false', self.pipeline)
        worker_mux = self.pipeline.split("static func finishWorkerExport(", 1)[1].split(
            "@main", 1
        )[0]
        self.assertIn('"-an", "-c:v", "copy"', worker_mux)
        self.assertIn('"-map", "0:v:0", "-map", "1:a:0?"', self.controller)
        self.assertIn('"-c:v", "copy", "-c:a", "aac"', self.controller)
        merge_body = self.controller.split("private func merge(", 1)[1].split(
            "private func verify(", 1
        )[0]
        self.assertNotIn('"-shortest"', merge_body)

    def test_cluster_integrity_contract_covers_source_and_all_model_assets(self):
        for contract in [
            "inputByteCount",
            "inputSHA256",
            "restorationAssetSHA256",
            "detectorAssetSHA256",
            "roiEnhancerAssetSHA256",
            "input_sha256_mismatch",
        ]:
            self.assertIn(contract, self.source + self.runner)

    def test_ios_control_service_and_worker_discovery_are_separate(self):
        info = (ROOT / "packaging" / "macOS" / "standalone" / "Info.plist").read_text()
        self.assertIn("_mioh._tcp", info)
        self.assertIn("_mioh-worker._tcp", info)
        self.assertNotEqual("_mioh._tcp", "_mioh-worker._tcp")


if __name__ == "__main__":
    unittest.main()
