import AVFoundation
import Foundation
import MiohRemoteKit

@MainActor
final class RemoteStore: ObservableObject {
  @Published var serverAddress: String
  @Published var token = ""
  @Published private(set) var connected = false
  @Published private(set) var status: MiohStatus?
  @Published private(set) var errorMessage: String?
  @Published private(set) var busy = false
  @Published private(set) var streamURL: URL?
  @Published private(set) var selectedServerName: String?

  let player = AVPlayer()

  var fullControlURL: URL? {
    Self.normalizedURL(serverAddress)
  }

  var accessTokenIsComplete: Bool {
    Self.isCompleteAccessToken(token)
  }

  var canConnect: Bool {
    Self.normalizedURL(serverAddress) != nil && accessTokenIsComplete && !busy
  }

  private var client: MiohRemoteClient?
  private var streamTicket: String?
  private var pollingTask: Task<Void, Never>?
  private let defaults = UserDefaults.standard

  init() {
    serverAddress = UserDefaults.standard.string(forKey: "miohRemote.serverAddress") ?? ""
    if let url = Self.normalizedURL(serverAddress) {
      token = RemoteCredentialStore.loadToken(serverID: Self.serverID(url)) ?? ""
    }
  }

  deinit {
    pollingTask?.cancel()
  }

  @discardableResult
  func select(_ endpoint: MiohServerEndpoint) -> Bool {
    guard let url = endpoint.baseURL else { return false }
    serverAddress = url.absoluteString
    selectedServerName = endpoint.name
    token = RemoteCredentialStore.loadToken(serverID: Self.serverID(url)) ?? ""
    errorMessage = nil
    return accessTokenIsComplete
  }

  func hasSavedCredentials(for endpoint: MiohServerEndpoint) -> Bool {
    guard let url = endpoint.baseURL,
      let saved = RemoteCredentialStore.loadToken(serverID: Self.serverID(url))
    else { return false }
    return Self.isCompleteAccessToken(saved)
  }

  /// Selects a discovered Mac and reconnects immediately when this device has
  /// already paired with it. First-time pairing stops after selection so the
  /// user only has to enter the code shown on the Mac.
  func connect(to endpoint: MiohServerEndpoint) async {
    guard select(endpoint) else { return }
    await connect()
  }

  func connect() async {
    guard let url = Self.normalizedURL(serverAddress) else {
      errorMessage = "サーバーアドレスを確認してください。"
      return
    }
    guard Self.isCompleteAccessToken(token) else {
      errorMessage = "Macのmiohに表示されるアクセスコードを入力してください。"
      return
    }
    let cleanToken = Self.normalizedAccessToken(token)
    busy = true
    errorMessage = nil
    let candidate = MiohRemoteClient(baseURL: url, token: cleanToken)
    do {
      let latest = try await candidate.status()
      client = candidate
      status = latest
      connected = true
      serverAddress = url.absoluteString
      token = cleanToken
      defaults.set(serverAddress, forKey: "miohRemote.serverAddress")
      try RemoteCredentialStore.saveToken(cleanToken, serverID: Self.serverID(url))
      startPolling()
    } catch {
      connected = false
      client = nil
      errorMessage = error.localizedDescription
    }
    busy = false
  }

  private static func normalizedAccessToken(_ value: String) -> String {
    let compact = compactAccessToken(value)
    let alphabet = Set("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
    guard compact.count == 12, compact.allSatisfy({ alphabet.contains($0) }) else {
      return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    return stride(from: 0, to: 12, by: 4).map { offset in
      let start = compact.index(compact.startIndex, offsetBy: offset)
      let end = compact.index(start, offsetBy: 4)
      return String(compact[start..<end])
    }.joined(separator: "-")
  }

  private static func compactAccessToken(_ value: String) -> String {
    value.uppercased().filter { character in
      character != "-" && !character.isWhitespace
    }
  }

  private static func isCompleteAccessToken(_ value: String) -> Bool {
    let compact = compactAccessToken(value)
    let alphabet = Set("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
    return compact.count == 12 && compact.allSatisfy({ alphabet.contains($0) })
  }

  func disconnect() async {
    pollingTask?.cancel()
    pollingTask = nil
    await stopStream()
    client = nil
    status = nil
    connected = false
    errorMessage = nil
  }

  func refresh() async {
    guard let client else { return }
    do {
      status = try await client.status()
      errorMessage = nil
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func play() async { await command { try await $0.play() } }
  func pause() async { await command { try await $0.pause() } }
  func stopPlayback() async { await command { try await $0.stopPlayback() } }
  func seek(to seconds: Double) async { await command { try await $0.seek(seconds: seconds) } }
  func setVolume(_ volume: Double) async { await command { try await $0.setVolume(volume) } }
  func setMuted(_ muted: Bool) async { await command { try await $0.setMuted(muted) } }
  func startExport() async { await command { try await $0.startExport() } }
  func stopExport() async { await command { try await $0.stopExport() } }

  func startStream() async {
    guard let client else { return }
    if streamTicket != nil { await stopStream() }
    busy = true
    do {
      let issued = try await client.startStream()
      streamTicket = MiohRemoteClient.streamTicket(from: issued.playlistURL)
      streamURL = issued.playlistURL
      player.replaceCurrentItem(with: AVPlayerItem(url: issued.playlistURL))
      player.play()
      errorMessage = nil
    } catch {
      errorMessage = error.localizedDescription
    }
    busy = false
  }

  func stopStream() async {
    player.pause()
    player.replaceCurrentItem(with: nil)
    let ticket = streamTicket
    streamTicket = nil
    streamURL = nil
    if let ticket, let client {
      do {
        try await client.stopStream(ticket: ticket)
      } catch {
        errorMessage = error.localizedDescription
      }
    }
  }

  private func command(_ action: (MiohRemoteClient) async throws -> Void) async {
    guard let client else { return }
    do {
      try await action(client)
      await refresh()
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  private func startPolling() {
    pollingTask?.cancel()
    pollingTask = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        guard !Task.isCancelled, let self else { return }
        await self.refresh()
      }
    }
  }

  private static func normalizedURL(_ raw: String) -> URL? {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    let withScheme = trimmed.contains("://") ? trimmed : "http://\(trimmed)"
    guard var components = URLComponents(string: withScheme),
      components.scheme?.lowercased() == "http",
      components.host?.isEmpty == false
    else { return nil }
    if components.port == nil { components.port = 8888 }
    components.path = "/"
    components.query = nil
    components.fragment = nil
    return components.url
  }

  private static func serverID(_ url: URL) -> String {
    "\(url.host ?? "unknown"):\(url.port ?? 8888)"
  }
}
