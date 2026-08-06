import Foundation

public struct MiohRemoteClient: Sendable {
  public let baseURL: URL
  public let token: String
  private let session: URLSession

  public init(baseURL: URL, token: String, session: URLSession = .shared) {
    self.baseURL = baseURL
    self.token = token
    self.session = session
  }

  public func status() async throws -> MiohStatus {
    try await request(path: "api/v1/status", method: "GET", body: Optional<Bool>.none)
  }

  public func play() async throws { try await accepted(path: "api/v1/playback/play") }
  public func pause() async throws { try await accepted(path: "api/v1/playback/pause") }
  public func toggle() async throws { try await accepted(path: "api/v1/playback/toggle") }
  public func stopPlayback() async throws { try await accepted(path: "api/v1/playback/stop") }

  public func seek(seconds: Double) async throws {
    try await accepted(path: "api/v1/playback/seek", body: ["seconds": seconds])
  }

  public func setVolume(_ volume: Double) async throws {
    try await accepted(path: "api/v1/playback/volume", body: ["volume": volume])
  }

  public func setMuted(_ muted: Bool) async throws {
    try await accepted(path: "api/v1/playback/mute", body: ["muted": muted])
  }

  public func startExport() async throws { try await accepted(path: "api/v1/export/start") }
  public func stopExport() async throws { try await accepted(path: "api/v1/export/stop") }

  public func startStream() async throws -> (session: MiohStreamSession, playlistURL: URL) {
    let response: MiohStreamSession = try await request(
      path: "api/v1/stream/session",
      method: "POST",
      body: Optional<Bool>.none
    )
    guard let playlistURL = URL(string: response.playlist, relativeTo: normalizedBaseURL)?.absoluteURL
    else { throw MiohRemoteError.invalidResponse }
    return (response, playlistURL)
  }

  public func stopStream(ticket: String) async throws {
    try await accepted(path: "api/v1/stream/stop", body: ["ticket": ticket])
  }

  public static func streamTicket(from playlistURL: URL) -> String? {
    let parts = playlistURL.path.split(separator: "/")
    guard parts.count == 4, parts[0] == "stream", parts[1] == "v1",
      parts[3] == "index.m3u8"
    else { return nil }
    let ticket = String(parts[2])
    guard ticket.utf8.count == 64,
      ticket.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
    else { return nil }
    return ticket
  }

  private var normalizedBaseURL: URL {
    baseURL.appendingPathComponent("", isDirectory: true)
  }

  private func accepted(path: String) async throws {
    let _: MiohAcceptedResponse = try await request(
      path: path,
      method: "POST",
      body: Optional<Bool>.none
    )
  }

  private func accepted<Body: Encodable & Sendable>(path: String, body: Body) async throws {
    let _: MiohAcceptedResponse = try await request(path: path, method: "POST", body: body)
  }

  private func request<Response: Decodable, Body: Encodable & Sendable>(
    path: String,
    method: String,
    body: Body?
  ) async throws -> Response {
    guard let url = URL(string: path, relativeTo: normalizedBaseURL)?.absoluteURL
    else { throw MiohRemoteError.invalidBaseURL }

    var request = URLRequest(url: url)
    request.httpMethod = method
    request.timeoutInterval = 15
    request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    if let body {
      request.httpBody = try JSONEncoder().encode(body)
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    } else if method == "POST" {
      request.httpBody = Data()
    }

    let (data, response) = try await session.data(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw MiohRemoteError.invalidResponse
    }
    guard (200..<300).contains(http.statusCode) else {
      let code = (try? JSONDecoder().decode(MiohAPIErrorBody.self, from: data).error)
        ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
      throw MiohRemoteError.http(status: http.statusCode, code: code)
    }
    do {
      return try JSONDecoder().decode(Response.self, from: data)
    } catch {
      throw MiohRemoteError.decoding(error.localizedDescription)
    }
  }
}

private struct MiohAcceptedResponse: Decodable {
  let ok: Bool
  let accepted: Bool
}
