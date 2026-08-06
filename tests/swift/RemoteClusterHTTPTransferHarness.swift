import CryptoKit
import Foundation

private enum HTTPHarnessFailure: Error {
  case malformedDescriptor
  case head
  case range
  case upload
  case cancellation
}

@main
struct RemoteClusterHTTPTransferHarness {
  static func main() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-cluster-http-harness-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source.mp4")
    let sourceData = Data((0..<65_537).map { UInt8($0 % 251) })
    try sourceData.write(to: source)
    let sourceSHA = SHA256.hash(data: sourceData)
      .map { String(format: "%02x", $0) }.joined()
    let output = root.appendingPathComponent("result/shard.mp4")
    let now = Date()
    let request = RemoteClusterJobRequest(
      jobID: UUID(),
      attemptID: UUID(),
      leaseID: UUID(),
      coordinatorNodeID: UUID(),
      sharedRootIdentifier: "",
      inputByteCount: Int64(sourceData.count),
      inputSHA256: sourceSHA,
      inputRelativePath: try RemoteClusterRelativePath(validating: "source/input.mp4"),
      outputRelativePath: try RemoteClusterRelativePath(validating: "result/shard.mp4"),
      mediaRange: RemoteClusterMediaRange(
        decodeStartNanoseconds: 0,
        decodeEndNanoseconds: 2_000_000_000,
        coreStartNanoseconds: 0,
        coreEndNanoseconds: 1_000_000_000,
        leadingOverlapFrames: 0,
        trailingOverlapFrames: 2
      ),
      options: RemoteClusterRestorationOptions(
        restorationModelIdentifier: "restorer",
        restorationAssetSHA256: String(repeating: "a", count: 64),
        detectorModelIdentifier: "detector",
        detectorAssetSHA256: String(repeating: "b", count: 64),
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
    let server = RemoteClusterHTTPTransferServer(advertisedHost: "127.0.0.1")
    try await server.start()
    defer { server.stop() }
    let pinnedIdentity = try await server.pinSource(source)
    guard pinnedIdentity.byteCount == Int64(sourceData.count),
      pinnedIdentity.sha256 == sourceSHA
    else { throw HTTPHarnessFailure.malformedDescriptor }
    let descriptor = try server.register(
      request: request,
      inputURL: source,
      outputURL: output,
      maximumOutputBytes: 1_048_576
    )
    guard descriptor.isValid,
      URL(string: descriptor.inputURL)?.pathExtension == "mp4",
      URL(string: descriptor.outputURL)?.lastPathComponent == "output.mp4"
    else { throw HTTPHarnessFailure.malformedDescriptor }

    var head = URLRequest(url: URL(string: descriptor.inputURL)!)
    head.httpMethod = "HEAD"
    let (_, headResponse) = try await URLSession.shared.data(for: head)
    guard let headHTTP = headResponse as? HTTPURLResponse,
      headHTTP.statusCode == 200,
      headHTTP.value(forHTTPHeaderField: "Accept-Ranges") == "bytes",
      Int(headHTTP.value(forHTTPHeaderField: "Content-Length") ?? "") == sourceData.count
    else { throw HTTPHarnessFailure.head }

    var range = URLRequest(url: URL(string: descriptor.inputURL)!)
    range.setValue("bytes=1024-2047", forHTTPHeaderField: "Range")
    let (rangeData, rangeResponse) = try await URLSession.shared.data(for: range)
    guard let rangeHTTP = rangeResponse as? HTTPURLResponse,
      rangeHTTP.statusCode == 206,
      rangeData == sourceData.subdata(in: 1024..<2048)
    else { throw HTTPHarnessFailure.range }

    let payload = root.appendingPathComponent("worker-output.mp4")
    let payloadData = Data((0..<32_777).map { UInt8(($0 * 7) % 253) })
    try payloadData.write(to: payload)
    try await RemoteClusterHTTPTransferClient.upload(file: payload, descriptor: descriptor)
    guard (try? Data(contentsOf: output)) == payloadData else {
      throw HTTPHarnessFailure.upload
    }

    // Exact upload retry is idempotent only when byte count and SHA match.
    try await RemoteClusterHTTPTransferClient.upload(file: payload, descriptor: descriptor)
    guard (try? Data(contentsOf: output)) == payloadData else {
      throw HTTPHarnessFailure.upload
    }

    server.unregister(attemptID: request.attemptID)
    try await Task.sleep(nanoseconds: 50_000_000)
    var endpointRejected = false
    do {
      let (_, response) = try await URLSession.shared.data(
        from: URL(string: descriptor.inputURL)!
      )
      endpointRejected = (response as? HTTPURLResponse)?.statusCode == 404
    } catch {
      // A closed listener path may surface as HTTP 404 or a transport error.
      endpointRejected = true
    }
    guard endpointRejected else { throw HTTPHarnessFailure.cancellation }
    print("remote-cluster HTTP transfer: ok")
  }
}
