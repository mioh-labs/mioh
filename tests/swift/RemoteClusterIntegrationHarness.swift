import CryptoKit
import Foundation

private enum HarnessFailure: Error {
  case discovery
  case response(String)
  case artifact
  case cancellationBarrier
  case leaseRefresh
}

private actor CancellationTeardownProbe {
  private(set) var started = false
  private(set) var finished = false

  func markStarted() { started = true }
  func markFinished() { finished = true }
}

@main
struct RemoteClusterIntegrationHarness {
  @MainActor
  static func main() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-cluster-harness-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let input = root.appendingPathComponent("source.bin")
    let inputData = Data("mioh cluster integration".utf8)
    try inputData.write(to: input)
    let inputDigest = SHA256.hash(data: inputData)
      .map { String(format: "%02x", $0) }.joined()

    let workerID = UUID()
    let coordinatorID = UUID()
    let worker = RemoteClusterService(localNodeID: workerID)
    let coordinator = RemoteClusterService(localNodeID: coordinatorID)
    let capabilities = RemoteClusterCapabilities(
      nodeID: workerID,
      displayName: "mioh harness worker",
      role: .worker,
      sharedRootIdentifier: "harness-root",
      architecture: "arm64",
      operatingSystem: "test",
      maximumConcurrentJobs: 1,
      restorationModelIdentifiers: ["restorer"],
      detectorModelIdentifiers: ["detector"]
    )
    try worker.startWorker(
      sharedRoot: root,
      capabilities: capabilities,
      launcher: { _, _, output in
        let payload = Data("artifact".utf8)
        try payload.write(to: output, options: .atomic)
        return RemoteClusterJobMetrics(
          processedFrames: 18,
          wallSeconds: 0.01,
          outputByteCount: Int64(payload.count)
        )
      }
    )
    coordinator.startCoordinatorDiscovery()
    var node: RemoteClusterDiscoveredNode?
    for _ in 0..<100 {
      if let found = coordinator.discoveredNodes.first(where: { $0.id == workerID }) {
        node = found
        break
      }
      try await Task.sleep(nanoseconds: 50_000_000)
    }
    guard let node else { throw HarnessFailure.discovery }
    let client = RemoteClusterClient(localNodeID: coordinatorID)
    let capabilityResponse = try await client.call(.capabilities, node: node)
    guard capabilityResponse.ok, capabilityResponse.capabilities == capabilities else {
      throw HarnessFailure.response("capabilities")
    }

    let attemptID = UUID()
    let leaseID = UUID()
    let now = Date()
    let request = RemoteClusterJobRequest(
      jobID: UUID(),
      attemptID: attemptID,
      leaseID: leaseID,
      coordinatorNodeID: coordinatorID,
      sharedRootIdentifier: "harness-root",
      inputByteCount: Int64(inputData.count),
      inputSHA256: inputDigest,
      inputRelativePath: try RemoteClusterRelativePath(validating: "source.bin"),
      outputRelativePath: try RemoteClusterRelativePath(validating: "results/shard.mp4"),
      mediaRange: RemoteClusterMediaRange(
        decodeStartNanoseconds: 0,
        decodeEndNanoseconds: 2_000_000_000,
        coreStartNanoseconds: 0,
        coreEndNanoseconds: 1_800_000_000,
        leadingOverlapFrames: 0,
        trailingOverlapFrames: 2
      ),
      options: RemoteClusterRestorationOptions(
        restorationModelIdentifier: "restorer",
        restorationAssetSHA256: String(repeating: "b", count: 64),
        detectorModelIdentifier: "detector",
        detectorAssetSHA256: String(repeating: "c", count: 64),
        restorationClipLength: 18,
        temporalOverlap: 2,
        crossfade: true,
        detectionEmptyLookahead: 1,
        detectFaceMosaics: false,
        blendFeather: 1,
        sharpenStrength: 0,
        detailBoost: 0,
        textureMix: 0,
        smoothStrength: 0,
        effectUpscale: 1,
        roiEnhancerModelIdentifier: nil,
        roiEnhancerAssetSHA256: nil,
        roiEnhancerStrength: 0,
        roiEnhancerScale: 1,
        videoCodec: "h264",
        bitrateMultiplier: 1,
        mp4FastStart: false,
        targetFPSNumerator: nil,
        targetFPSDenominator: nil
      ),
      createdAt: now,
      leaseExpiresAt: now.addingTimeInterval(10 * 60)
    )

    // Jobs are planned as one batch but can wait in a lane queue for hours.
    // Refreshing immediately before submission must preserve idempotency IDs
    // while replacing only the stale deadline.
    let staleRequest = request.withLeaseExpiration(now.addingTimeInterval(-1))
    let leaseLedger = RemoteClusterWorkerJobLedger()
    guard leaseLedger.admit(
      staleRequest,
      sharedRootIdentifier: "harness-root",
      now: now
    ).disposition == .rejected else { throw HarnessFailure.leaseRefresh }
    let refreshedRequest = staleRequest.withLeaseExpiration(
      now.addingTimeInterval(10 * 60)
    )
    guard refreshedRequest.jobID == staleRequest.jobID,
      refreshedRequest.attemptID == staleRequest.attemptID,
      refreshedRequest.leaseID == staleRequest.leaseID,
      refreshedRequest.createdAt == staleRequest.createdAt,
      leaseLedger.admit(
        refreshedRequest,
        sharedRootIdentifier: "harness-root",
        now: now
      ).disposition == .accepted
    else { throw HarnessFailure.leaseRefresh }
    leaseLedger.cancelAllActive()

    let submission = try await client.call(.submit(request), node: node)
    guard submission.ok else {
      throw HarnessFailure.response(submission.errorCode ?? "submit")
    }
    var completed = false
    for _ in 0..<100 {
      let response = try await client.call(
        .status(RemoteClusterStatusQuery(attemptID: attemptID)),
        node: node
      )
      if response.attempt?.state == .completed {
        completed = true
        break
      }
      if response.attempt?.state == .failed {
        throw HarnessFailure.response(response.attempt?.failureCode ?? "failed")
      }
      try await Task.sleep(nanoseconds: 50_000_000)
    }
    let artifact = root.appendingPathComponent("results/shard.mp4")
    guard completed, (try? Data(contentsOf: artifact)) == Data("artifact".utf8) else {
      throw HarnessFailure.artifact
    }

    // A cancelled local launcher remains a capacity occupant until its real
    // teardown returns. This catches the race where the task handle used to be
    // removed at cancellation time and a second native process could overlap.
    let localLedger = RemoteClusterWorkerJobLedger()
    let localAttemptID = UUID()
    let localLeaseID = UUID()
    let localNow = Date()
    let localRequest = RemoteClusterJobRequest(
      jobID: UUID(),
      attemptID: localAttemptID,
      leaseID: localLeaseID,
      coordinatorNodeID: coordinatorID,
      sharedRootIdentifier: "harness-root",
      inputByteCount: Int64(inputData.count),
      inputSHA256: inputDigest,
      inputRelativePath: try RemoteClusterRelativePath(validating: "source.bin"),
      outputRelativePath: try RemoteClusterRelativePath(validating: "results/cancelled.mp4"),
      mediaRange: request.mediaRange,
      options: request.options,
      createdAt: localNow,
      leaseExpiresAt: localNow.addingTimeInterval(10 * 60)
    )
    let teardownProbe = CancellationTeardownProbe()
    let localAdmission = localLedger.submit(
      localRequest,
      sharedRoot: root,
      sharedRootIdentifier: "harness-root",
      launcher: { _, _, _ in
        await teardownProbe.markStarted()
        do {
          try await Task.sleep(nanoseconds: 60_000_000_000)
        } catch {
          let deadline = Date().addingTimeInterval(0.15)
          while Date() < deadline { await Task.yield() }
          await teardownProbe.markFinished()
          throw CancellationError()
        }
        throw HarnessFailure.cancellationBarrier
      }
    )
    guard localAdmission.disposition == .accepted else {
      throw HarnessFailure.cancellationBarrier
    }
    for _ in 0..<100 {
      if await teardownProbe.started { break }
      try await Task.sleep(nanoseconds: 5_000_000)
    }
    guard await teardownProbe.started else { throw HarnessFailure.cancellationBarrier }
    let cancellationStartedAt = Date()
    await localLedger.cancelAllActiveAndWait()
    guard await teardownProbe.finished,
      Date().timeIntervalSince(cancellationStartedAt) >= 0.12,
      localLedger.inFlightExecutionCount == 0
    else {
      throw HarnessFailure.cancellationBarrier
    }
    coordinator.stop()
    worker.stop()
    print("remote-cluster integration: ok")
  }
}
