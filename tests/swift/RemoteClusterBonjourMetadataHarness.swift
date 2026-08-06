import Foundation

private enum HarnessFailure: Error {
  case missingHTTPRootWasRejected
  case missingSharedRootWasAccepted
}

@main
struct RemoteClusterBonjourMetadataHarness {
  static func main() throws {
    let base: [String: String] = [
      "v": String(RemoteClusterCapabilities.protocolVersion),
      "node": UUID().uuidString.lowercased(),
      "name": "mioh iPad Worker",
      "role": RemoteClusterRole.worker.rawValue,
      "jobs": "1",
    ]

    var http = base
    http["transfer"] = RemoteClusterTransferMode.coordinatorHTTPV1.rawValue
    let parsedHTTP = try RemoteClusterBonjourMetadata(dictionary: http)
    guard parsedHTTP.sharedRootIdentifier.isEmpty else {
      throw HarnessFailure.missingHTTPRootWasRejected
    }

    var sharedRoot = base
    sharedRoot["transfer"] = RemoteClusterTransferMode.sharedRootV1.rawValue
    do {
      _ = try RemoteClusterBonjourMetadata(dictionary: sharedRoot)
      throw HarnessFailure.missingSharedRootWasAccepted
    } catch RemoteClusterBonjourError.malformed {
      // Expected: shared-root transport still requires a non-empty root.
    }
  }
}
