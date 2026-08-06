import Foundation
import Network

private enum ProbeFailure: Error {
  case arguments
  case invalidEndpoint
  case discovery(String)
}

@main
struct RemoteClusterPairingProbe {
  @MainActor
  static func main() async throws {
    let arguments = CommandLine.arguments
    guard arguments.count == 4,
      let nodeID = UUID(uuidString: arguments[1]),
      let portValue = UInt16(arguments[3]),
      let port = NWEndpoint.Port(rawValue: portValue)
    else { throw ProbeFailure.arguments }

    let metadata = try RemoteClusterBonjourMetadata(dictionary: [
      "v": String(RemoteClusterCapabilities.protocolVersion),
      "node": nodeID.uuidString.lowercased(),
      "name": "iPad Worker",
      "role": RemoteClusterRole.worker.rawValue,
      "transfer": RemoteClusterTransferMode.coordinatorHTTPV1.rawValue,
      "jobs": "1",
    ])
    let client = RemoteClusterClient(localNodeID: UUID())
    let endpoint: NWEndpoint = arguments[2] == "service"
      ? .service(
        name: metadata.displayName,
        type: RemoteClusterBonjourMetadata.serviceType,
        domain: RemoteClusterBonjourMetadata.serviceDomain,
        interface: nil
      )
      : .hostPort(host: NWEndpoint.Host(arguments[2]), port: port)
    let node = if arguments[2] == "browse" {
      try await browse(nodeID: nodeID)
    } else {
      RemoteClusterDiscoveredNode(
        metadata: metadata,
        endpoint: endpoint,
        serviceName: metadata.displayName,
        serviceDomain: "local.",
        interfaceName: nil,
        lastSeenAt: Date()
      )
    }
    let response = try await client.call(.capabilities, node: node)
    let errorCode = response.errorCode ?? "none"
    print(
      "ok=\(response.ok) error=\(errorCode) "
        + "nodeMatch=\(response.capabilities?.nodeID == nodeID)"
    )
  }

  @MainActor
  private static func browse(nodeID: UUID) async throws -> RemoteClusterDiscoveredNode {
    try await withCheckedThrowingContinuation { continuation in
      let browser = NWBrowser(
        for: .bonjourWithTXTRecord(
          type: RemoteClusterBonjourMetadata.serviceType,
          domain: RemoteClusterBonjourMetadata.serviceDomain
        ),
        using: .tcp
      )
      let lock = NSLock()
      var completed = false
      let finish: @Sendable (Result<RemoteClusterDiscoveredNode, Error>) -> Void = { result in
        lock.lock()
        guard !completed else { lock.unlock(); return }
        completed = true
        lock.unlock()
        browser.cancel()
        continuation.resume(with: result)
      }
      browser.stateUpdateHandler = { state in
        if case .failed(let error) = state {
          finish(.failure(ProbeFailure.discovery(error.localizedDescription)))
        }
      }
      browser.browseResultsChangedHandler = { results, _ in
        for result in results {
          guard case .bonjour(let txtRecord) = result.metadata,
            let metadata = try? RemoteClusterBonjourMetadata(txtRecord: txtRecord),
            metadata.nodeID == nodeID,
            case .service(let name, _, let domain, let interface) = result.endpoint
          else { continue }
          finish(.success(RemoteClusterDiscoveredNode(
            metadata: metadata,
            endpoint: result.endpoint,
            serviceName: name,
            serviceDomain: domain,
            interfaceName: interface?.name,
            lastSeenAt: Date()
          )))
          return
        }
      }
      browser.start(queue: DispatchQueue(label: "mioh.pairing-probe.browser"))
      DispatchQueue.global().asyncAfter(deadline: .now() + 5) {
        finish(.failure(ProbeFailure.discovery("timeout")))
      }
    }
  }
}
