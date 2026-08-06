import Foundation

public struct MiohServerEndpoint: Identifiable, Hashable, Sendable {
  public let name: String
  public let host: String
  public let port: Int

  public init(name: String, host: String, port: Int) {
    self.name = name
    self.host = host
    self.port = port
  }

  public var id: String { "\(name)|\(host)|\(port)" }

  public var baseURL: URL? {
    let escapedHost = host.contains(":") ? "[\(host)]" : host
    return URL(string: "http://\(escapedHost):\(port)/")
  }
}

public struct MiohStatus: Codable, Equatable, Sendable {
  public struct Server: Codable, Equatable, Sendable {
    public let enabled: Bool
    public let port: Int
    public let id: String?
    public let apiVersion: Int?
  }

  public struct Playback: Codable, Equatable, Sendable {
    public let state: String
    public let position: Double
    public let duration: Double
    public let bufferedSeconds: Double
    public let volume: Double
    public let muted: Bool
    public let input: String?
    public let hasError: Bool
  }

  public struct Export: Codable, Equatable, Sendable {
    public let running: Bool
    public let status: String
    public let progress: Double
    public let input: String?
    public let output: String?
  }

  public let ok: Bool
  public let server: Server
  public let playback: Playback
  public let export: Export
}

public struct MiohStreamSession: Codable, Equatable, Sendable {
  public let ok: Bool
  public let playlist: String
  public let expiresAt: String
}

struct MiohAPIErrorBody: Decodable {
  let error: String?
}

public enum MiohRemoteError: LocalizedError, Equatable, Sendable {
  case invalidBaseURL
  case invalidResponse
  case http(status: Int, code: String)
  case decoding(String)

  public var errorDescription: String? {
    switch self {
    case .invalidBaseURL:
      return "The mioh server address is invalid."
    case .invalidResponse:
      return "The mioh server returned an invalid response."
    case .http(let status, let code):
      return "mioh returned HTTP \(status): \(code)"
    case .decoding(let message):
      return "Could not read the mioh response: \(message)"
    }
  }
}
