import Foundation
import XCTest
@testable import MiohRemoteKit

final class MiohRemoteKitTests: XCTestCase {
  func testRestorationOptionsAcceptExactRationalFPS() throws {
    let request = try workerRequest(
      rootID: "test-root",
      inputByteCount: 1,
      inputSHA256: String(repeating: "a", count: 64),
      targetFPSNumerator: 30_000,
      targetFPSDenominator: 1_001
    )
    XCTAssertTrue(request.options.isValid)
    XCTAssertEqual(request.options.targetFPSNumerator, 30_000)
    XCTAssertEqual(request.options.targetFPSDenominator, 1_001)
  }

  func testClusterRelativePathRejectsEscapes() throws {
    XCTAssertThrowsError(try MiohClusterRelativePath(validating: "../movie.mp4"))
    XCTAssertThrowsError(try MiohClusterRelativePath(validating: "/movie.mp4"))
    XCTAssertThrowsError(try MiohClusterRelativePath(validating: "folder//movie.mp4"))
    XCTAssertEqual(
      try MiohClusterRelativePath(validating: "inputs/movie.mp4").rawValue,
      "inputs/movie.mp4"
    )
  }

  func testPortableModelIdentityManifestRejectsEscapingAssets() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-model-manifest-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let manifest = MiohPortableModelIdentityManifest(
      models: [
        "basicvsrpp-v1.2-coreai": .init(
          sha256: String(repeating: "a", count: 64),
          sourceAssets: ["../outside.aimodel"]
        )
      ]
    )
    try JSONEncoder().encode(manifest).write(
      to: root.appendingPathComponent(MiohPortableModelIdentityManifest.fileName)
    )
    XCTAssertThrowsError(try MiohPortableModelIdentityManifest.load(from: root)) { thrown in
      XCTAssertEqual(thrown as? MiohClusterAssetError, .invalidIdentityManifest)
    }
  }

  @MainActor
  func testWorkerLedgerStagesThenAtomicallyPublishesOutput() async throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-worker-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let input = root.appendingPathComponent("source.mp4")
    let source = Data("trusted test input".utf8)
    try source.write(to: input)
    let request = try workerRequest(
      rootID: "test-root",
      inputByteCount: Int64(source.count),
      inputSHA256: MiohClusterAsset.sha256(input)
    )
    let capabilities = workerCapabilities(rootID: "test-root")
    let ledger = MiohClusterWorkerLedger()
    let finalOutput = try request.outputRelativePath.resolve(beneath: root)
    let admission = ledger.submit(
      request,
      sharedRoot: root,
      capabilities: capabilities
    ) { _, _, stagingOutput in
      XCTAssertTrue(stagingOutput.lastPathComponent.hasSuffix(".part"))
      XCTAssertFalse(FileManager.default.fileExists(atPath: finalOutput.path))
      let output = Data("finished output".utf8)
      try output.write(to: stagingOutput)
      return MiohClusterJobMetrics(
        processedFrames: 12,
        wallSeconds: 0.1,
        outputByteCount: Int64(output.count)
      )
    }
    XCTAssertEqual(admission.disposition, .accepted)
    let completed = await waitForTerminal(request.attemptID, ledger: ledger)
    XCTAssertEqual(completed?.state, .completed)
    XCTAssertTrue(FileManager.default.fileExists(atPath: finalOutput.path))
    XCTAssertFalse(
      FileManager.default.fileExists(
        atPath: finalOutput.deletingLastPathComponent()
          .appendingPathComponent(
            ".mioh-cluster-\(request.attemptID.uuidString.lowercased()).part"
          ).path
      )
    )
    XCTAssertFalse(
      FileManager.default.fileExists(
        atPath: finalOutput.appendingPathExtension("mioh-cluster.lock").path
      )
    )
  }

  @MainActor
  func testWorkerRefusesToAdvertiseEmptyModelCapabilities() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-worker-service-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let capabilities = MiohClusterCapabilities(
      nodeID: UUID(),
      displayName: "test iPad",
      sharedRootIdentifier: "test-root",
      architecture: "arm64",
      operatingSystem: "iPadOS",
      restorationModelIdentifiers: [],
      detectorModelIdentifiers: []
    )
    let service = MiohClusterWorkerService()
    XCTAssertThrowsError(
      try service.start(
        sharedRoot: root,
        capabilities: capabilities
      ) { _, _, _ in
        XCTFail("launcher must not run")
        return MiohClusterJobMetrics(processedFrames: 0, wallSeconds: 0, outputByteCount: 0)
      }
    ) { error in
      XCTAssertEqual(error as? MiohClusterWorkerError, .missingModels)
    }
    XCTAssertEqual(service.state, .stopped)
  }

  @MainActor
  func testHTTPWorkerRunsWithoutSharedRootAndCleansStagingFile() async throws {
    let transfer = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/opaque-ticket/input.mp4",
      outputURL: "http://coordinator.local:48991/mioh-cluster/v1/opaque-ticket/output.mp4",
      expiresAt: Date().addingTimeInterval(180),
      maximumOutputBytes: 1_024
    )
    let request = try workerRequest(
      rootID: "",
      inputByteCount: 42,
      inputSHA256: String(repeating: "c", count: 64),
      httpTransfer: transfer
    )
    let capabilities = MiohClusterCapabilities(
      nodeID: UUID(),
      displayName: "test iPad",
      transferMode: .coordinatorHTTPV1,
      sharedRootIdentifier: "",
      architecture: "arm64",
      operatingSystem: "iPadOS",
      restorationModelIdentifiers: ["basicvsrpp-v1.2-coreai-variable"],
      detectorModelIdentifiers: ["v4-fast-coreai"],
      supportedTransferModes: [.coordinatorHTTPV1]
    )
    let stagingURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-ipad-worker", isDirectory: true)
      .appendingPathComponent("\(request.attemptID.uuidString.lowercased()).mp4")
    let expectedOutput = Data("uploaded shard".utf8)
    let ledger = MiohClusterWorkerLedger(
      httpUploader: { receivedTransfer, localFile, byteCount in
        XCTAssertEqual(receivedTransfer, transfer)
        XCTAssertEqual(localFile, stagingURL)
        XCTAssertEqual(byteCount, Int64(expectedOutput.count))
        XCTAssertEqual(try Data(contentsOf: localFile), expectedOutput)
      }
    )
    let admission = ledger.submit(
      request,
      sharedRoot: nil,
      capabilities: capabilities
    ) { _, inputURL, stagingOutput in
      XCTAssertEqual(inputURL.absoluteString, transfer.inputURL)
      XCTAssertEqual(stagingOutput, stagingURL)
      try expectedOutput.write(to: stagingOutput)
      return MiohClusterJobMetrics(
        processedFrames: 12,
        wallSeconds: 0.1,
        outputByteCount: Int64(expectedOutput.count)
      )
    }
    XCTAssertEqual(admission.disposition, .accepted)
    let completed = await waitForTerminal(request.attemptID, ledger: ledger)
    XCTAssertEqual(completed?.state, .completed)
    XCTAssertFalse(FileManager.default.fileExists(atPath: stagingURL.path))
  }

  @MainActor
  func testHTTPWorkerRejectsOversizedOutputBeforeUpload() async throws {
    let transfer = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/opaque-ticket/input.mp4",
      outputURL: "http://coordinator.local:48991/mioh-cluster/v1/opaque-ticket/output.mp4",
      expiresAt: Date().addingTimeInterval(180),
      maximumOutputBytes: 4
    )
    let request = try workerRequest(
      rootID: "",
      inputByteCount: 42,
      inputSHA256: String(repeating: "c", count: 64),
      httpTransfer: transfer
    )
    let capabilities = MiohClusterCapabilities(
      nodeID: UUID(),
      displayName: "test iPad",
      transferMode: .coordinatorHTTPV1,
      sharedRootIdentifier: "",
      architecture: "arm64",
      operatingSystem: "iPadOS",
      restorationModelIdentifiers: ["basicvsrpp-v1.2-coreai-variable"],
      detectorModelIdentifiers: ["v4-fast-coreai"],
      supportedTransferModes: [.coordinatorHTTPV1]
    )
    let stagingURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-ipad-worker", isDirectory: true)
      .appendingPathComponent("\(request.attemptID.uuidString.lowercased()).mp4")
    let output = Data("too large".utf8)
    let ledger = MiohClusterWorkerLedger(
      httpUploader: { _, _, _ in
        XCTFail("oversized output must not be uploaded")
      }
    )
    let admission = ledger.submit(
      request,
      sharedRoot: nil,
      capabilities: capabilities
    ) { _, _, candidate in
      try output.write(to: candidate)
      return MiohClusterJobMetrics(
        processedFrames: 12,
        wallSeconds: 0.1,
        outputByteCount: Int64(output.count)
      )
    }
    XCTAssertEqual(admission.disposition, .accepted)
    let failed = await waitForTerminal(request.attemptID, ledger: ledger)
    XCTAssertEqual(failed?.state, .failed)
    XCTAssertEqual(failed?.failureCode, "output_byte_count_mismatch")
    XCTAssertFalse(FileManager.default.fileExists(atPath: stagingURL.path))
  }

  func testHTTPTransferDescriptorAcceptsOnlyMatchingCapabilityEndpoints() {
    let descriptor = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/opaque_ticket-1/input.mov",
      outputURL: "http://COORDINATOR.local:48991/mioh-cluster/v1/opaque_ticket-1/output.mp4",
      expiresAt: Date().addingTimeInterval(120),
      maximumOutputBytes: 1_024
    )
    XCTAssertTrue(descriptor.isValid())
  }

  @MainActor
  func testHTTPAdmissionUsesLiveJobLeaseNotDescriptorSnapshotExpiry() async throws {
    let initialExpiration = Date().addingTimeInterval(0.5)
    let transfer = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/renewed_ticket/input.mp4",
      outputURL: "http://coordinator.local:48991/mioh-cluster/v1/renewed_ticket/output.mp4",
      expiresAt: initialExpiration,
      maximumOutputBytes: 1_024
    )
    XCTAssertTrue(transfer.isValid(), "new admission starts with a live capability")

    let request = try workerRequest(
      rootID: "",
      inputByteCount: 42,
      inputSHA256: String(repeating: "c", count: 64),
      httpTransfer: transfer,
      leaseExpiresAt: initialExpiration
    )
    let capabilities = MiohClusterCapabilities(
      nodeID: UUID(),
      displayName: "test iPad",
      transferMode: .coordinatorHTTPV1,
      sharedRootIdentifier: "",
      architecture: "arm64",
      operatingSystem: "iPadOS",
      restorationModelIdentifiers: ["basicvsrpp-v1.2-coreai-variable"],
      detectorModelIdentifiers: ["v4-fast-coreai"],
      supportedTransferModes: [.coordinatorHTTPV1]
    )
    let payload = Data("renewed lease output".utf8)
    let ledger = MiohClusterWorkerLedger(httpUploader: { _, file, byteCount in
      XCTAssertTrue(transfer.hasValidStructure())
      XCTAssertFalse(
        transfer.isValid(),
        "completion must use the renewed live lease, not stale descriptor time"
      )
      XCTAssertEqual(try Data(contentsOf: file), payload)
      XCTAssertEqual(byteCount, Int64(payload.count))
    })
    let admission = ledger.submit(
      request,
      sharedRoot: nil,
      capabilities: capabilities
    ) { _, _, candidate in
      try await Task.sleep(nanoseconds: 750_000_000)
      try payload.write(to: candidate)
      return MiohClusterJobMetrics(
        processedFrames: 1,
        wallSeconds: 0.1,
        outputByteCount: Int64(payload.count)
      )
    }
    XCTAssertEqual(admission.disposition, .accepted)
    XCTAssertTrue(
      ledger.renew(
        MiohClusterRenewLeaseRequest(
          attemptID: request.attemptID,
          leaseID: request.leaseID,
          newExpiration: Date().addingTimeInterval(120)
        )
      )
    )
    let completed = await waitForTerminal(request.attemptID, ledger: ledger)
    XCTAssertEqual(completed?.state, .completed)
  }

  @MainActor
  func testHTTPAdmissionRejectsExpiredTransferWithFreshJobLease() throws {
    let transfer = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/expired_transfer/input.mp4",
      outputURL: "http://coordinator.local:48991/mioh-cluster/v1/expired_transfer/output.mp4",
      expiresAt: Date(timeIntervalSince1970: 1),
      maximumOutputBytes: 1_024
    )
    let request = try workerRequest(
      rootID: "",
      inputByteCount: 42,
      inputSHA256: String(repeating: "c", count: 64),
      httpTransfer: transfer,
      leaseExpiresAt: Date().addingTimeInterval(120)
    )
    let capabilities = MiohClusterCapabilities(
      nodeID: UUID(),
      displayName: "test iPad",
      transferMode: .coordinatorHTTPV1,
      sharedRootIdentifier: "",
      architecture: "arm64",
      operatingSystem: "iPadOS",
      restorationModelIdentifiers: ["basicvsrpp-v1.2-coreai-variable"],
      detectorModelIdentifiers: ["v4-fast-coreai"],
      supportedTransferModes: [.coordinatorHTTPV1]
    )
    let ledger = MiohClusterWorkerLedger()
    let admission = ledger.submit(
      request,
      sharedRoot: nil,
      capabilities: capabilities,
      launcher: { _, _, _ in
        XCTFail("expired transfer capability must not launch")
        return MiohClusterJobMetrics(
          processedFrames: 1,
          wallSeconds: 0.1,
          outputByteCount: 1
        )
      }
    )
    XCTAssertEqual(admission.disposition, .rejected)
  }

  @MainActor
  func testHTTPAdmissionStillRejectsExpiredJobLease() throws {
    let transfer = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/expired_job/input.mp4",
      outputURL: "http://coordinator.local:48991/mioh-cluster/v1/expired_job/output.mp4",
      expiresAt: Date(timeIntervalSince1970: 1),
      maximumOutputBytes: 1_024
    )
    let now = Date()
    let request = try workerRequest(
      rootID: "",
      inputByteCount: 42,
      inputSHA256: String(repeating: "c", count: 64),
      httpTransfer: transfer,
      leaseExpiresAt: now.addingTimeInterval(-1)
    )
    let capabilities = MiohClusterCapabilities(
      nodeID: UUID(),
      displayName: "test iPad",
      transferMode: .coordinatorHTTPV1,
      sharedRootIdentifier: "",
      architecture: "arm64",
      operatingSystem: "iPadOS",
      restorationModelIdentifiers: ["basicvsrpp-v1.2-coreai-variable"],
      detectorModelIdentifiers: ["v4-fast-coreai"],
      supportedTransferModes: [.coordinatorHTTPV1]
    )
    let ledger = MiohClusterWorkerLedger()
    let admission = ledger.submit(
      request,
      sharedRoot: nil,
      capabilities: capabilities,
      launcher: { _, _, _ in
        XCTFail("expired lease must not launch")
        return MiohClusterJobMetrics(
          processedFrames: 1, wallSeconds: 0.1, outputByteCount: 1
        )
      },
      now: now
    )
    XCTAssertEqual(admission.disposition, .rejected)
  }

  func testHTTPTransferDescriptorRejectsSSRFDestinationsAndPathConfusion() {
    let input = "http://coordinator.local:48991/mioh-cluster/v1/ticket_123/input.mp4"
    let output = "http://coordinator.local:48991/mioh-cluster/v1/ticket_123/output.mp4"
    let invalidPairs: [(String, String)] = [
      // Output exfiltration / cross-origin SSRF.
      (input, "http://attacker.invalid:48991/mioh-cluster/v1/ticket_123/output.mp4"),
      (input, "http://coordinator.local:48992/mioh-cluster/v1/ticket_123/output.mp4"),
      (input, "https://coordinator.local:48991/mioh-cluster/v1/ticket_123/output.mp4"),
      // Credentials, queries and fragments can alter authority or cache routing.
      ("http://user@coordinator.local:48991/mioh-cluster/v1/ticket_123/input.mp4", output),
      (input, output + "?redirect=http://attacker.invalid"),
      (input + "#ignored", output),
      // Both endpoints must use one exact opaque-ticket namespace.
      (input, "http://coordinator.local:48991/mioh-cluster/v1/other_ticket/output.mp4"),
      ("http://coordinator.local:48991/cluster/v1/ticket_123/input.mp4", output),
      ("http://coordinator.local:48991/mioh-cluster/v1/ticket_123/input.mkv", output),
      (input, "http://coordinator.local:48991/mioh-cluster/v1/ticket_123/result.mp4"),
      // Reject encoded aliases and traversal before Foundation normalizes them.
      ("http://coordinator.local:48991/mioh-cluster/v1/ticket%2Fescape/input.mp4", output),
      ("http://coordinator.local:48991/mioh-cluster/v1/../ticket_123/input.mp4", output),
    ]
    for (candidateInput, candidateOutput) in invalidPairs {
      let descriptor = MiohClusterHTTPTransferDescriptor(
        inputURL: candidateInput,
        outputURL: candidateOutput,
        expiresAt: Date().addingTimeInterval(120),
        maximumOutputBytes: 1_024
      )
      XCTAssertFalse(
        descriptor.isValid(),
        "unexpectedly accepted \(candidateInput) -> \(candidateOutput)"
      )
    }
  }

  func testHTTPUploaderRetriesLostSuccessResponseWithExactFileAndTicket() async throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-http-replay-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let localFile = root.appendingPathComponent("finished.mp4")
    let payload = Data("immutable finished shard".utf8)
    try payload.write(to: localFile)
    let transfer = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/replay_ticket/input.mp4",
      outputURL: "http://coordinator.local:48991/mioh-cluster/v1/replay_ticket/output.mp4",
      expiresAt: Date().addingTimeInterval(120),
      maximumOutputBytes: 1_024
    )
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockURLProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    var requests: [URLRequest] = []
    MockURLProtocol.handler = { request in
      requests.append(request)
      if requests.count == 1 {
        // Models a server that published the exact file but whose 201 response
        // disappeared on the network. The second PUT is an exact replay.
        throw URLError(.networkConnectionLost)
      }
      let response = HTTPURLResponse(
        url: request.url!, statusCode: 201, httpVersion: "HTTP/1.1", headerFields: nil
      )!
      return (response, Data())
    }
    var delays: [UInt64] = []
    try await WorkerHTTPUploader.upload(
      transfer: transfer,
      localFile: localFile,
      byteCount: Int64(payload.count),
      session: session,
      retrySleeper: { delays.append($0) }
    )
    XCTAssertEqual(requests.count, 2)
    XCTAssertEqual(requests.map(\.url?.absoluteString), [transfer.outputURL, transfer.outputURL])
    XCTAssertEqual(requests.map(\.httpMethod), ["PUT", "PUT"])
    XCTAssertEqual(
      requests.map { $0.value(forHTTPHeaderField: "Content-Length") },
      [String(payload.count), String(payload.count)]
    )
    XCTAssertEqual(delays, [250_000_000])
    XCTAssertEqual(try Data(contentsOf: localFile), payload)
  }

  func testHTTPUploaderRetriesOnlyRetryableResponses() async throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-http-status-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let localFile = root.appendingPathComponent("finished.mp4")
    let payload = Data("finished shard".utf8)
    try payload.write(to: localFile)
    let transfer = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/status_ticket/input.mp4",
      outputURL: "http://coordinator.local:48991/mioh-cluster/v1/status_ticket/output.mp4",
      expiresAt: Date().addingTimeInterval(120),
      maximumOutputBytes: 1_024
    )

    let retryConfiguration = URLSessionConfiguration.ephemeral
    retryConfiguration.protocolClasses = [MockURLProtocol.self]
    let retrySession = URLSession(configuration: retryConfiguration)
    defer { retrySession.invalidateAndCancel() }
    var statuses = [425, 500, 201]
    var retryCalls = 0
    MockURLProtocol.handler = { request in
      retryCalls += 1
      let response = HTTPURLResponse(
        url: request.url!,
        statusCode: statuses.removeFirst(),
        httpVersion: "HTTP/1.1",
        headerFields: nil
      )!
      return (response, Data())
    }
    var delays: [UInt64] = []
    try await WorkerHTTPUploader.upload(
      transfer: transfer,
      localFile: localFile,
      byteCount: Int64(payload.count),
      session: retrySession,
      retrySleeper: { delays.append($0) }
    )
    XCTAssertEqual(retryCalls, 3)
    XCTAssertEqual(delays, [250_000_000, 500_000_000])

    let rejectConfiguration = URLSessionConfiguration.ephemeral
    rejectConfiguration.protocolClasses = [MockURLProtocol.self]
    let rejectSession = URLSession(configuration: rejectConfiguration)
    defer { rejectSession.invalidateAndCancel() }
    var rejectCalls = 0
    MockURLProtocol.handler = { request in
      rejectCalls += 1
      let response = HTTPURLResponse(
        url: request.url!, statusCode: 409, httpVersion: "HTTP/1.1", headerFields: nil
      )!
      return (response, Data())
    }
    do {
      try await WorkerHTTPUploader.upload(
        transfer: transfer,
        localFile: localFile,
        byteCount: Int64(payload.count),
        session: rejectSession,
        retrySleeper: { _ in XCTFail("409 digest conflict must not back off or retry") }
      )
      XCTFail("409 digest conflict must be rejected")
    } catch let error as MiohClusterWorkerError {
      XCTAssertEqual(error, .uploadRejected)
    }
    XCTAssertEqual(rejectCalls, 1)
  }

  func testHTTPUploaderCancellationIsNotRetried() async throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-http-cancel-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let localFile = root.appendingPathComponent("finished.mp4")
    let payload = Data("finished shard".utf8)
    try payload.write(to: localFile)
    let transfer = MiohClusterHTTPTransferDescriptor(
      inputURL: "http://coordinator.local:48991/mioh-cluster/v1/cancel_ticket/input.mp4",
      outputURL: "http://coordinator.local:48991/mioh-cluster/v1/cancel_ticket/output.mp4",
      expiresAt: Date().addingTimeInterval(120),
      maximumOutputBytes: 1_024
    )
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockURLProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    var calls = 0
    MockURLProtocol.handler = { _ in
      calls += 1
      throw URLError(.cancelled)
    }
    do {
      try await WorkerHTTPUploader.upload(
        transfer: transfer,
        localFile: localFile,
        byteCount: Int64(payload.count),
        session: session,
        retrySleeper: { _ in XCTFail("cancellation must not back off or retry") }
      )
      XCTFail("cancellation must escape immediately")
    } catch is CancellationError {
      // Expected.
    }
    XCTAssertEqual(calls, 1)
  }

  func testHTTPUploaderUsesBoundedLongTransferTimeouts() {
    let configuration = WorkerHTTPUploader.sessionConfiguration()
    XCTAssertEqual(configuration.timeoutIntervalForRequest, 30 * 60)
    XCTAssertEqual(configuration.timeoutIntervalForResource, 24 * 60 * 60)
    XCTAssertEqual(configuration.httpMaximumConnectionsPerHost, 1)
  }

  func testEndpointBuildsIPv6SafeURL() {
    let endpoint = MiohServerEndpoint(name: "studio", host: "fd00::42", port: 8888)
    XCTAssertEqual(endpoint.baseURL?.absoluteString, "http://[fd00::42]:8888/")
  }

  func testStatusContractDecodesNullableFilenames() throws {
    let data = Data(
      #"{"ok":true,"server":{"enabled":true,"port":8888,"id":"3f4947f4-acde-4fa2-b8cd-b3f4260be421","apiVersion":1},"playback":{"state":"playing","position":12.5,"duration":90,"bufferedSeconds":8,"volume":0.7,"muted":false,"input":null,"hasError":false},"export":{"running":false,"status":"Ready","progress":0,"input":null,"output":null}}"#.utf8
    )
    let status = try JSONDecoder().decode(MiohStatus.self, from: data)
    XCTAssertEqual(status.server.port, 8888)
    XCTAssertEqual(status.server.apiVersion, 1)
    XCTAssertEqual(status.playback.position, 12.5)
    XCTAssertNil(status.export.output)
  }

  func testStreamTicketParsingAcceptsOnlyContractPath() {
    let ticket = String(repeating: "a", count: 64)
    XCTAssertEqual(
      MiohRemoteClient.streamTicket(
        from: URL(string: "http://mioh.local:8888/stream/v1/\(ticket)/index.m3u8")!
      ),
      ticket
    )
    XCTAssertNil(
      MiohRemoteClient.streamTicket(
        from: URL(string: "http://mioh.local:8888/stream/v1/not-a-ticket/index.m3u8")!
      )
    )
  }

  func testClientSendsBearerAndPlaybackBody() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockURLProtocol.self]
    let session = URLSession(configuration: configuration)
    MockURLProtocol.handler = { request in
      XCTAssertEqual(request.url?.path, "/api/v1/playback/seek")
      XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer secret")
      let object = try JSONSerialization.jsonObject(with: MockURLProtocol.bodyData(request)) as? [String: Double]
      XCTAssertEqual(object?["seconds"], 42.25)
      let response = HTTPURLResponse(
        url: request.url!, statusCode: 202, httpVersion: "HTTP/1.1", headerFields: nil
      )!
      return (response, Data(#"{"ok":true,"accepted":true}"#.utf8))
    }
    let client = MiohRemoteClient(
      baseURL: URL(string: "http://mioh.local:8888/")!,
      token: "secret",
      session: session
    )
    try await client.seek(seconds: 42.25)
  }

  private func workerCapabilities(rootID: String) -> MiohClusterCapabilities {
    MiohClusterCapabilities(
      nodeID: UUID(),
      displayName: "test iPad",
      sharedRootIdentifier: rootID,
      architecture: "arm64",
      operatingSystem: "iPadOS",
      restorationModelIdentifiers: ["basicvsrpp-v1.2-coreai-variable"],
      detectorModelIdentifiers: ["v4-fast-coreai"]
    )
  }

  private func workerRequest(
    rootID: String,
    inputByteCount: Int64,
    inputSHA256: String,
    httpTransfer: MiohClusterHTTPTransferDescriptor? = nil,
    leaseExpiresAt: Date = Date().addingTimeInterval(120),
    targetFPSNumerator: Int? = nil,
    targetFPSDenominator: Int? = nil
  ) throws -> MiohClusterJobRequest {
    MiohClusterJobRequest(
      jobID: UUID(),
      attemptID: UUID(),
      leaseID: UUID(),
      coordinatorNodeID: UUID(),
      sharedRootIdentifier: rootID,
      inputByteCount: inputByteCount,
      inputSHA256: inputSHA256,
      inputRelativePath: try MiohClusterRelativePath(validating: "source.mp4"),
      outputRelativePath: try MiohClusterRelativePath(validating: "outputs/result.mp4"),
      mediaRange: MiohClusterMediaRange(
        decodeStartNanoseconds: 0,
        decodeEndNanoseconds: 2_000_000_000,
        coreStartNanoseconds: 0,
        coreEndNanoseconds: 2_000_000_000,
        leadingOverlapFrames: 0,
        trailingOverlapFrames: 0
      ),
      options: MiohClusterRestorationOptions(
        restorationModelIdentifier: "basicvsrpp-v1.2-coreai-variable",
        restorationAssetSHA256: String(repeating: "a", count: 64),
        detectorModelIdentifier: "v4-fast-coreai",
        detectorAssetSHA256: String(repeating: "b", count: 64),
        restorationClipLength: 18,
        temporalOverlap: 4,
        crossfade: true,
        detectionEmptyLookahead: 10,
        detectFaceMosaics: false,
        blendFeather: 1,
        sharpenStrength: 0,
        detailBoost: 0,
        textureMix: 0,
        smoothStrength: 0,
        effectUpscale: 1,
        videoCodec: "h264",
        bitrateMultiplier: 1,
        mp4FastStart: true,
        targetFPSNumerator: targetFPSNumerator,
        targetFPSDenominator: targetFPSDenominator
      ),
      createdAt: Date(),
      leaseExpiresAt: leaseExpiresAt,
      httpTransfer: httpTransfer
    )
  }

  @MainActor
  private func waitForTerminal(
    _ attemptID: UUID,
    ledger: MiohClusterWorkerLedger
  ) async -> MiohClusterAttemptRecord? {
    for _ in 0..<100 {
      if let record = ledger.record(attemptID), record.state.isTerminal { return record }
      try? await Task.sleep(nanoseconds: 10_000_000)
    }
    return ledger.record(attemptID)
  }
}

private final class MockURLProtocol: URLProtocol {
  static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  static func bodyData(_ request: URLRequest) -> Data {
    if let body = request.httpBody { return body }
    guard let stream = request.httpBodyStream else { return Data() }
    stream.open()
    defer { stream.close() }
    var result = Data()
    var bytes = [UInt8](repeating: 0, count: 1024)
    while stream.hasBytesAvailable {
      let count = stream.read(&bytes, maxLength: bytes.count)
      guard count > 0 else { break }
      result.append(bytes, count: count)
    }
    return result
  }

  override func startLoading() {
    do {
      let (response, data) = try Self.handler?(request) ?? {
        throw URLError(.badServerResponse)
      }()
      client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
      client?.urlProtocol(self, didLoad: data)
      client?.urlProtocolDidFinishLoading(self)
    } catch {
      client?.urlProtocol(self, didFailWithError: error)
    }
  }

  override func stopLoading() {}
}
