import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "MiohApp.swift"
BUILD = ROOT / "packaging" / "macOS" / "standalone" / "build_app.sh"


class MiohClusterWorkerLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.build = BUILD.read_text()

    def test_runner_exposes_controller_and_worker_contracts(self):
        for contract in [
            "func clusterRestorationOptions() async throws",
            "func clusterResolvedOutputFile() -> URL?",
            "func remoteClusterJobLauncher() -> RemoteClusterJobLauncher",
            "func runRemoteClusterWorkerJob(",
        ]:
            self.assertIn(contract, self.source)

    def test_worker_uses_padded_decode_and_core_output_ranges(self):
        for contract in [
            "workerMode: true",
            "startNanoseconds: request.mediaRange.decodeStartNanoseconds",
            "decodeEndNanoseconds: request.mediaRange.decodeEndNanoseconds",
            "outputCoreStartNanoseconds: request.mediaRange.coreStartNanoseconds",
            "outputCoreEndNanoseconds: request.mediaRange.coreEndNanoseconds",
            'splitMode: "none"',
            "detectionEmptyLookahead: 1",
            "targetFPS: nil",
            "targetFPSDenominator: nil",
        ]:
            self.assertIn(contract, self.source)

    def test_worker_verifies_assets_and_artifact_identity(self):
        for contract in [
            "computeRemoteClusterAssetSHA256",
            "restorationAssetSHA256.lowercased()",
            "detectorAssetSHA256.lowercased()",
            'object["kind"] as? String == "cluster_artifact"',
            'artifact["job_id"]',
            'artifact["attempt_id"]',
            'artifact["frame_count"]',
            'artifact["core_start_ns"]',
            'artifact["core_end_ns"]',
            'artifact["has_audio"] as? Bool == false',
            "FileManager.default.moveItem(at: artifactURL, to: outputURL)",
        ]:
            self.assertIn(contract, self.source)

    def test_worker_rejects_shard_phase_changing_options(self):
        self.assertIn(
            "request.options.detectionEmptyLookahead == 1", self.source
        )
        self.assertIn("request.options.targetFPSNumerator == nil", self.source)
        self.assertIn("request.options.targetFPSDenominator == nil", self.source)

    def test_model_asset_digest_is_deterministic_and_rejects_symlinks(self):
        for contract in [
            "rootValues.isSymbolicLink != true",
            "values.isSymbolicLink != true",
            ".precomposedStringWithCanonicalMapping",
            "Array($0.relative.utf8).lexicographicallyPrecedes",
            "Set(entries.map(\\.relative)).count == entries.count",
            "hasher.update(data: Data(entry.relative.utf8))",
            "hasher.update(data: Data([0]))",
        ]:
            self.assertIn(contract, self.source)

    def test_cluster_uses_portable_canonical_model_identity_manifest(self):
        for contract in [
            "RemoteClusterCanonicalModelManifest",
            'case formatVersion = "format_version"',
            'case digestAlgorithm = "digest_algorithm"',
            '"models/mioh-cluster-model-identities-v1.json"',
            'manifest.digestAlgorithm == "sha256-tree-v1"',
            "manifest.models[modelIdentifier]",
            "remoteClusterCanonicalAssetDigest(",
            "runtimeAsset: restoration.url",
            "runtimeAsset: detection.url",
        ]:
            self.assertIn(contract, self.source)

        for contract in [
            'CANONICAL_MODEL_MANIFEST="$RESOURCES/models/mioh-cluster-model-identities-v1.json"',
            '"format_version": 1',
            '"digest_algorithm": "sha256-tree-v1"',
            '"basicvsrpp-v1.2-coreai-variable"',
            '"asset_type": "source-collection"',
            '"source_assets": [asset.name for asset in variable_assets]',
            'f"{asset.name}/{candidate.relative_to(asset).as_posix()}"',
        ]:
            self.assertIn(contract, self.build)

        variable_names = self.build.split("variable_names = [", 1)[1].split("]", 1)[0]
        self.assertEqual(variable_names.count('"') // 2, 11)

    def test_manifest_falls_back_only_for_unlisted_or_legacy_assets(self):
        canonical = self.source.split(
            "private func remoteClusterCanonicalAssetDigest(", 1
        )[1].split("/// Asset digest convention", 1)[0]
        self.assertIn("if let identity = manifest.models[modelIdentifier]", canonical)
        self.assertGreaterEqual(
            canonical.count("remoteClusterAssetDigest(runtimeAsset)"),
            2,
        )

    def test_worker_cancellation_reaches_native_process(self):
        for contract in [
            "withTaskCancellationHandler",
            'Data("{\\\"command\\\":\\\"stop\\\"}\\n".utf8)',
            "process.interrupt()",
            "process.terminate()",
        ]:
            self.assertIn(contract, self.source)


if __name__ == "__main__":
    unittest.main()
