import Combine
import Foundation

/// Finds mioh Mac coordinators advertising `_mioh._tcp` on the local network.
/// The app still supports manual host entry for networks that filter Bonjour.
@MainActor
public final class BonjourDiscovery: NSObject, ObservableObject {
  @Published public private(set) var endpoints: [MiohServerEndpoint] = []
  @Published public private(set) var errorMessage: String?

  private let browser = NetServiceBrowser()
  private var services: [String: NetService] = [:]
  private var resolved: [String: MiohServerEndpoint] = [:]
  private var running = false

  public override init() {
    super.init()
    browser.delegate = self
  }

  public func start() {
    guard !running else { return }
    running = true
    errorMessage = nil
    browser.searchForServices(ofType: "_mioh._tcp.", inDomain: "local.")
  }

  public func stop() {
    guard running else { return }
    running = false
    browser.stop()
    for service in services.values {
      service.stop()
      service.delegate = nil
    }
    services.removeAll()
    resolved.removeAll()
    endpoints = []
  }

  private func key(for service: NetService) -> String {
    "\(service.name)|\(service.type)|\(service.domain)"
  }

  private func publish() {
    endpoints = resolved.values.sorted {
      $0.name.localizedStandardCompare($1.name) == .orderedAscending
    }
  }
}

extension BonjourDiscovery: NetServiceBrowserDelegate, NetServiceDelegate {
  nonisolated public func netServiceBrowser(
    _ browser: NetServiceBrowser,
    didFind service: NetService,
    moreComing: Bool
  ) {
    Task { @MainActor in
      let key = self.key(for: service)
      self.services[key] = service
      service.delegate = self
      service.resolve(withTimeout: 5)
    }
  }

  nonisolated public func netServiceBrowser(
    _ browser: NetServiceBrowser,
    didRemove service: NetService,
    moreComing: Bool
  ) {
    Task { @MainActor in
      let key = self.key(for: service)
      self.services.removeValue(forKey: key)
      self.resolved.removeValue(forKey: key)
      self.publish()
    }
  }

  nonisolated public func netServiceBrowser(
    _ browser: NetServiceBrowser,
    didNotSearch errorDict: [String: NSNumber]
  ) {
    Task { @MainActor in
      self.running = false
      self.errorMessage = "Bonjour discovery failed: \(errorDict)"
    }
  }

  nonisolated public func netServiceDidResolveAddress(_ sender: NetService) {
    Task { @MainActor in
      let key = self.key(for: sender)
      guard let rawHost = sender.hostName?.trimmingCharacters(in: CharacterSet(charactersIn: ".")),
        !rawHost.isEmpty, sender.port > 0
      else { return }
      self.resolved[key] = MiohServerEndpoint(
        name: sender.name,
        host: rawHost,
        port: sender.port
      )
      self.publish()
    }
  }

  nonisolated public func netService(
    _ sender: NetService,
    didNotResolve errorDict: [String: NSNumber]
  ) {
    Task { @MainActor in
      let key = self.key(for: sender)
      self.resolved.removeValue(forKey: key)
      self.publish()
    }
  }
}
