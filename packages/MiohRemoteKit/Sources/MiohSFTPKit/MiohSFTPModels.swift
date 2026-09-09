import CryptoKit
import Foundation

public struct MiohSFTPConfiguration: Equatable, Sendable {
  public let host: String
  public let port: Int
  public let username: String
  public let password: String

  public init(host: String, port: Int = 22, username: String, password: String) {
    self.host = host
    self.port = port
    self.username = username
    self.password = password
  }

  public func validated() throws -> MiohSFTPConfiguration {
    var cleanHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
    if cleanHost.hasPrefix("["), cleanHost.hasSuffix("]") {
      cleanHost.removeFirst()
      cleanHost.removeLast()
    }
    guard !cleanHost.isEmpty,
      cleanHost.utf8.count <= 255,
      !cleanHost.contains("://"),
      !cleanHost.contains("/"),
      !cleanHost.contains("@"),
      !cleanHost.contains("["),
      !cleanHost.contains("]"),
      !cleanHost.unicodeScalars.contains(where: {
        CharacterSet.whitespacesAndNewlines.contains($0) || $0.value == 0
      })
    else { throw MiohSFTPError.invalidHost }
    guard (1...65_535).contains(port) else {
      throw MiohSFTPError.invalidPort
    }
    let cleanUsername = username.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !cleanUsername.isEmpty,
      cleanUsername.utf8.count <= 256,
      !cleanUsername.unicodeScalars.contains(where: {
        $0.value == 0 || $0.value < 32 || $0.value == 127
      })
    else { throw MiohSFTPError.missingUsername }
    guard !password.isEmpty, password.utf8.count <= 4_096,
      !password.unicodeScalars.contains(where: { $0.value == 0 })
    else { throw MiohSFTPError.missingPassword }
    return MiohSFTPConfiguration(
      host: cleanHost,
      port: port,
      username: cleanUsername,
      password: password
    )
  }

  public var endpointID: String {
    let cleanHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
      .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
      .lowercased()
    let displayHost = cleanHost.contains(":") ? "[\(cleanHost)]" : cleanHost
    return "\(displayHost):\(port)"
  }

  public var credentialID: String {
    "\(endpointID)|\(username.trimmingCharacters(in: .whitespacesAndNewlines))"
  }
}

public struct MiohSFTPHostIdentity: Equatable, Sendable {
  public let algorithm: String
  public let key: String
  public let fingerprint: String

  public init(shortHandKey: String) throws {
    guard shortHandKey.utf8.count <= 65_536 else {
      throw MiohSFTPError.invalidHostKey
    }
    let fields = shortHandKey.split(whereSeparator: { $0.isWhitespace })
    guard fields.count >= 2 else { throw MiohSFTPError.invalidHostKey }
    let algorithm = String(fields[0])
    let encodedKey = String(fields[1])
    guard (algorithm.hasPrefix("ssh-") || algorithm.hasPrefix("ecdsa-")),
      let keyBytes = Data(base64Encoded: encodedKey), !keyBytes.isEmpty
    else { throw MiohSFTPError.invalidHostKey }

    let digest = Data(SHA256.hash(data: keyBytes))
      .base64EncodedString()
      .replacingOccurrences(of: "=", with: "")
    self.algorithm = algorithm
    self.key = "\(algorithm) \(encodedKey)"
    fingerprint = "SHA256:\(digest)"
  }
}

public struct MiohSFTPEntry: Identifiable, Equatable, Sendable {
  public enum Kind: Equatable, Sendable {
    case directory
    case movie
  }

  public let name: String
  public let path: String
  public let kind: Kind
  public let byteCount: Int64?
  public let modifiedAt: Date?

  public init(
    name: String,
    path: String,
    kind: Kind,
    byteCount: Int64? = nil,
    modifiedAt: Date? = nil
  ) {
    self.name = name
    self.path = path
    self.kind = kind
    self.byteCount = byteCount
    self.modifiedAt = modifiedAt
  }

  public var id: String { path }
  public var isDirectory: Bool { kind == .directory }
}

/// Opaque handle for one regular movie opened on a dedicated SFTP session.
/// The remote path and credentials intentionally stay inside that session.
public struct MiohSFTPStreamingFile: Equatable, Sendable {
  public let id: UUID
  public let byteCount: Int64
  public let pathExtension: String

  init(id: UUID, byteCount: Int64, pathExtension: String) {
    self.id = id
    self.byteCount = byteCount
    self.pathExtension = pathExtension
  }
}

public enum MiohSFTPPath {
  public static let movieExtensions: Set<String> = ["mp4", "mov", "m4v"]

  public static func normalize(
    _ path: String,
    relativeTo base: String = "/"
  ) throws -> String {
    let value = path
    guard value.utf8.count <= 4_096,
      !value.unicodeScalars.contains(where: { $0.value == 0 })
    else {
      throw MiohSFTPError.invalidRemotePath
    }
    let baseValue = base.hasPrefix("/") ? base : "/\(base)"
    let combined: String
    if value.isEmpty || value == "." {
      combined = baseValue
    } else if value.hasPrefix("/") {
      combined = value
    } else {
      combined = baseValue == "/" ? "/\(value)" : "\(baseValue)/\(value)"
    }

    var components: [Substring] = []
    for component in combined.split(separator: "/", omittingEmptySubsequences: true) {
      switch component {
      case ".":
        continue
      case "..":
        if !components.isEmpty { components.removeLast() }
      default:
        guard !component.unicodeScalars.contains(where: {
          $0.value == 0 || $0.value < 32 || $0.value == 127
        }) else { throw MiohSFTPError.invalidRemotePath }
        components.append(component)
      }
    }
    return components.isEmpty ? "/" : "/" + components.joined(separator: "/")
  }

  public static func appending(_ name: String, to directory: String) throws -> String {
    guard !name.isEmpty, name.utf8.count <= 255,
      name != ".", name != "..", !name.contains("/"),
      !name.unicodeScalars.contains(where: {
        $0.value == 0 || $0.value < 32 || $0.value == 127
      })
    else { throw MiohSFTPError.invalidRemotePath }
    let cleanDirectory = try normalize(directory)
    return cleanDirectory == "/" ? "/\(name)" : "\(cleanDirectory)/\(name)"
  }

  public static func parent(of path: String) throws -> String {
    let cleanPath = try normalize(path)
    guard cleanPath != "/" else { return "/" }
    let components = cleanPath.split(separator: "/")
    guard components.count > 1 else { return "/" }
    return "/" + components.dropLast().joined(separator: "/")
  }

  public static func isSupportedMovie(_ path: String) -> Bool {
    guard let name = path.split(separator: "/").last,
      let dot = name.lastIndex(of: "."), dot != name.startIndex
    else { return false }
    let ext = name[name.index(after: dot)...].lowercased()
    return movieExtensions.contains(ext)
  }
}

public enum MiohSFTPTransferPolicy {
  public static let hardMaximumBytes: Int64 = 20 * 1_024 * 1_024 * 1_024
  public static let minimumFreeBytes: Int64 = 16 * 1_024 * 1_024

  public static func maximumDownloadBytes(availableBytes: Int64) throws -> Int64 {
    let reserveBytes = max(1 * 1_024 * 1_024 * 1_024, availableBytes / 10)
    guard availableBytes > reserveBytes + minimumFreeBytes else {
      throw MiohSFTPError.insufficientStorage
    }
    return min(hardMaximumBytes, availableBytes - reserveBytes)
  }

  /// Returns the safe final-file limit when part of the file is already held
  /// in an app-owned resume file. `availableBytes` excludes those retained
  /// bytes, so add them back without ever exceeding the global hard limit.
  public static func maximumResumableDownloadBytes(
    availableBytes: Int64,
    retainedPartialBytes: Int64
  ) throws -> Int64 {
    let freshCapacity = try maximumDownloadBytes(availableBytes: availableBytes)
    let retained = min(hardMaximumBytes, max(0, retainedPartialBytes))
    guard freshCapacity <= hardMaximumBytes - retained else {
      return hardMaximumBytes
    }
    return min(hardMaximumBytes, freshCapacity + retained)
  }

  public static func validate(byteCount: UInt64, maximumBytes: Int64) throws {
    guard byteCount <= UInt64(max(0, maximumBytes)) else {
      throw MiohSFTPError.fileTooLarge(maximumBytes: maximumBytes)
    }
  }
}

public enum MiohSFTPError: LocalizedError, Equatable, Sendable {
  case invalidHost
  case invalidPort
  case missingUsername
  case missingPassword
  case invalidHostKey
  case hostKeyChanged(expected: String, actual: String)
  case notConnected
  case invalidRemotePath
  case unsupportedFile
  case tooManyEntries(maximum: Int)
  case fileTooLarge(maximumBytes: Int64)
  case insufficientStorage
  case remoteFileAlreadyExists
  case incompleteTransfer
  case localFile(String)

  public var errorDescription: String? {
    switch self {
    case .invalidHost:
      return "SFTPサーバーのホスト名を確認してください。"
    case .invalidPort:
      return "SFTPポートは1〜65535で入力してください。"
    case .missingUsername:
      return "SFTPユーザー名を入力してください。"
    case .missingPassword:
      return "SFTPパスワードを入力してください。"
    case .invalidHostKey:
      return "SFTPサーバーのホスト鍵を確認できませんでした。"
    case .hostKeyChanged(let expected, let actual):
      return "SFTPサーバーのホスト鍵が変わりました（保存済み: \(expected)、現在: \(actual)）。接続を中止しました。"
    case .notConnected:
      return "SFTPサーバーへ接続していません。"
    case .invalidRemotePath:
      return "SFTP上のパスが不正です。"
    case .unsupportedFile:
      return "MP4、MOV、M4Vの通常ファイルだけを転送できます。"
    case .tooManyEntries(let maximum):
      return "フォルダ内の項目が多すぎます（上限\(maximum)件）。"
    case .fileTooLarge(let maximumBytes):
      let maximum = ByteCountFormatter.string(fromByteCount: maximumBytes, countStyle: .file)
      return "ファイルが転送上限（\(maximum)）を超えています。"
    case .insufficientStorage:
      return "iPadの空き容量が不足しています。"
    case .remoteFileAlreadyExists:
      return "同名のファイルがSFTPサーバーに既にあります。"
    case .incompleteTransfer:
      return "SFTP転送のサイズが一致しませんでした。"
    case .localFile(let message):
      return "ローカルファイルを処理できませんでした: \(message)"
    }
  }
}
