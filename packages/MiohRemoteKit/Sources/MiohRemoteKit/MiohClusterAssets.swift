import CryptoKit
import Foundation
#if canImport(CoreAI)
  import CoreAI
#endif

public enum MiohClusterAssetError: LocalizedError, Equatable, Sendable {
  case missing
  case unsupportedType
  case symbolicLink
  case normalizedPathCollision
  case missingCoreAI
  case missingMainFunction
  case invalidIdentityManifest
  case missingModelIdentity
  case identityDigestMismatch

  public var errorDescription: String? {
    switch self {
    case .missing: "モデル資産が見つかりません。"
    case .unsupportedType: "モデル資産は通常ファイルまたはディレクトリである必要があります。"
    case .symbolicLink: "クラスタモデル資産にシンボリックリンクは使用できません。"
    case .normalizedPathCollision: "モデル資産に正規化後の重複パスがあります。"
    case .missingCoreAI: "Core AIを利用できるiPadOS 27以降が必要です。"
    case .missingMainFunction: "Core AIモデルにmain関数がありません。"
    case .invalidIdentityManifest: "モデル識別マニフェストが不正です。"
    case .missingModelIdentity: "このモデルの共通識別情報がありません。"
    case .identityDigestMismatch: "モデル資産が共通識別情報と一致しません。"
    }
  }
}

public enum MiohClusterAsset {
  /// Matches the directory digest convention used by the Mac coordinator.
  public static func sha256(_ url: URL) throws -> String {
    let keys: Set<URLResourceKey> = [
      .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
    ]
    let rootValues = try url.resourceValues(forKeys: keys)
    guard rootValues.isSymbolicLink != true else {
      throw MiohClusterAssetError.symbolicLink
    }

    func updateFile(_ file: URL, hasher: inout SHA256) throws {
      let handle = try FileHandle(forReadingFrom: file)
      defer { try? handle.close() }
      while let chunk = try handle.read(upToCount: 1_024 * 1_024), !chunk.isEmpty {
        hasher.update(data: chunk)
      }
    }

    var hasher = SHA256()
    if rootValues.isRegularFile == true {
      try updateFile(url, hasher: &hasher)
    } else if rootValues.isDirectory == true {
      var enumerationError: Error?
      guard let enumerator = FileManager.default.enumerator(
        at: url,
        includingPropertiesForKeys: Array(keys),
        options: [],
        errorHandler: { _, error in
          enumerationError = error
          return false
        }
      ) else { throw MiohClusterAssetError.missing }
      let rootPath = url.standardizedFileURL.path
      var entries: [(relative: String, url: URL)] = []
      for case let candidate as URL in enumerator {
        let values = try candidate.resourceValues(forKeys: keys)
        guard values.isSymbolicLink != true else {
          throw MiohClusterAssetError.symbolicLink
        }
        guard values.isRegularFile == true else { continue }
        let path = candidate.standardizedFileURL.path
        guard path.hasPrefix(rootPath + "/") else {
          throw MiohClusterAssetError.unsupportedType
        }
        entries.append((
          String(path.dropFirst(rootPath.count + 1))
            .precomposedStringWithCanonicalMapping,
          candidate
        ))
      }
      if let enumerationError { throw enumerationError }
      entries.sort {
        Array($0.relative.utf8).lexicographicallyPrecedes(Array($1.relative.utf8))
      }
      guard Set(entries.map(\.relative)).count == entries.count else {
        throw MiohClusterAssetError.normalizedPathCollision
      }
      for entry in entries {
        hasher.update(data: Data(entry.relative.utf8))
        hasher.update(data: Data([0]))
        try updateFile(entry.url, hasher: &hasher)
        hasher.update(data: Data([0]))
      }
    } else {
      throw MiohClusterAssetError.unsupportedType
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  /// Hashes several source assets as one portable logical model. Each file is
  /// prefixed by its source-asset directory name, matching the Mac manifest.
  public static func sha256Collection(_ roots: [(String, URL)]) throws -> String {
    let keys: Set<URLResourceKey> = [
      .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
    ]
    var entries: [(relative: String, url: URL)] = []
    for (name, root) in roots {
      let values = try root.resourceValues(forKeys: keys)
      guard values.isDirectory == true else {
        throw MiohClusterAssetError.unsupportedType
      }
      guard values.isSymbolicLink != true else {
        throw MiohClusterAssetError.symbolicLink
      }
      guard let enumerator = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: Array(keys),
        options: []
      ) else { throw MiohClusterAssetError.missing }
      let rootPath = root.standardizedFileURL.path
      for case let candidate as URL in enumerator {
        let candidateValues = try candidate.resourceValues(forKeys: keys)
        guard candidateValues.isSymbolicLink != true else {
          throw MiohClusterAssetError.symbolicLink
        }
        guard candidateValues.isRegularFile == true else { continue }
        let path = candidate.standardizedFileURL.path
        guard path.hasPrefix(rootPath + "/") else {
          throw MiohClusterAssetError.unsupportedType
        }
        entries.append((
          "\(name)/\(path.dropFirst(rootPath.count + 1))"
            .precomposedStringWithCanonicalMapping,
          candidate
        ))
      }
    }
    entries.sort {
      Array($0.relative.utf8).lexicographicallyPrecedes(Array($1.relative.utf8))
    }
    guard Set(entries.map(\.relative)).count == entries.count else {
      throw MiohClusterAssetError.normalizedPathCollision
    }
    var hasher = SHA256()
    for entry in entries {
      hasher.update(data: Data(entry.relative.utf8))
      hasher.update(data: Data([0]))
      let handle = try FileHandle(forReadingFrom: entry.url)
      defer { try? handle.close() }
      while let chunk = try handle.read(upToCount: 1_024 * 1_024), !chunk.isEmpty {
        hasher.update(data: chunk)
      }
      hasher.update(data: Data([0]))
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

public struct MiohCoreAIAssetValidation: Equatable, Sendable {
  public let url: URL
  public let sha256: String
  public let functionName: String

  public init(url: URL, sha256: String, functionName: String = "main") {
    self.url = url
    self.sha256 = sha256
    self.functionName = functionName
  }
}

public struct MiohPortableModelIdentityManifest: Codable, Equatable, Sendable {
  public static let fileName = "mioh-cluster-model-identities-v1.json"
  public static let schemaVersion = 1
  public static let digestAlgorithm = "sha256-tree-v1"

  public struct Entry: Codable, Equatable, Sendable {
    public let sha256: String
    public let assetType: String
    public let sourceAssets: [String]

    public init(
      sha256: String,
      assetType: String = "source-tree",
      sourceAssets: [String]
    ) {
      self.sha256 = sha256
      self.assetType = assetType
      self.sourceAssets = sourceAssets
    }

    enum CodingKeys: String, CodingKey {
      case sha256
      case assetType = "asset_type"
      case sourceAssets = "source_assets"
    }
  }

  public let schemaVersion: Int
  public let digestAlgorithm: String
  public let models: [String: Entry]

  public init(
    schemaVersion: Int = Self.schemaVersion,
    digestAlgorithm: String = Self.digestAlgorithm,
    models: [String: Entry]
  ) {
    self.schemaVersion = schemaVersion
    self.digestAlgorithm = digestAlgorithm
    self.models = models
  }

  enum CodingKeys: String, CodingKey {
    case schemaVersion = "format_version"
    case digestAlgorithm = "digest_algorithm"
    case models
  }

  public static func load(from modelRoot: URL) throws -> Self {
    let data = try Data(contentsOf: modelRoot.appendingPathComponent(fileName))
    let manifest = try JSONDecoder().decode(Self.self, from: data)
    guard manifest.schemaVersion == schemaVersion,
      manifest.digestAlgorithm == digestAlgorithm,
      !manifest.models.isEmpty,
      manifest.models.allSatisfy({ modelID, entry in
        !modelID.isEmpty && isSHA256(entry.sha256)
          && ((entry.assetType == "source-tree" && entry.sourceAssets.count == 1)
            || (entry.assetType == "source-collection" && !entry.sourceAssets.isEmpty))
          && entry.sourceAssets.allSatisfy {
            MiohClusterRelativePath(rawValue: $0) != nil
          }
      })
    else { throw MiohClusterAssetError.invalidIdentityManifest }
    return manifest
  }

  public func validateCoreAIModel(
    identifier: String,
    beneath modelRoot: URL
  ) async throws -> MiohValidatedPortableModelIdentity {
    guard let entry = models[identifier], entry.sourceAssets.count == 1,
      let relativePath = MiohClusterRelativePath(rawValue: entry.sourceAssets[0])
    else { throw MiohClusterAssetError.missingModelIdentity }
    let assetURL = try relativePath.resolve(beneath: modelRoot)
    let validation = try await MiohCoreAIAssetValidator.validateMainFunction(at: assetURL)
    guard validation.sha256 == entry.sha256 else {
      throw MiohClusterAssetError.identityDigestMismatch
    }
    return MiohValidatedPortableModelIdentity(
      identifier: identifier,
      assetURL: assetURL,
      canonicalSHA256: entry.sha256,
      functionName: validation.functionName
    )
  }

  /// Verifies the portable source asset without instantiating a Core AI
  /// runtime. Useful during app startup because some very large fixed-shape
  /// graphs can abort inside Metal before Swift can catch an error.
  public func validateCoreAIModelDigest(
    identifier: String,
    beneath modelRoot: URL
  ) throws -> MiohValidatedPortableModelIdentity {
    guard let entry = models[identifier], entry.assetType == "source-tree",
      entry.sourceAssets.count == 1,
      let relativePath = MiohClusterRelativePath(rawValue: entry.sourceAssets[0])
    else { throw MiohClusterAssetError.missingModelIdentity }
    let assetURL = try relativePath.resolve(beneath: modelRoot)
    guard try MiohClusterAsset.sha256(assetURL) == entry.sha256 else {
      throw MiohClusterAssetError.identityDigestMismatch
    }
    return MiohValidatedPortableModelIdentity(
      identifier: identifier,
      assetURL: assetURL,
      canonicalSHA256: entry.sha256,
      functionName: "main"
    )
  }

  public func validateCoreAIModelCollection(
    identifier: String,
    beneath modelRoot: URL
  ) async throws -> MiohValidatedPortableModelCollection {
    guard let entry = models[identifier],
      entry.assetType == "source-collection",
      !entry.sourceAssets.isEmpty
    else { throw MiohClusterAssetError.missingModelIdentity }
    var assets: [String: URL] = [:]
    var digestRoots: [(String, URL)] = []
    for source in entry.sourceAssets {
      guard let relativePath = MiohClusterRelativePath(rawValue: source) else {
        throw MiohClusterAssetError.invalidIdentityManifest
      }
      let url = try relativePath.resolve(beneath: modelRoot)
      _ = try await MiohCoreAIAssetValidator.validateMainFunction(at: url)
      assets[source] = url
      digestRoots.append((source, url))
    }
    guard try MiohClusterAsset.sha256Collection(digestRoots) == entry.sha256 else {
      throw MiohClusterAssetError.identityDigestMismatch
    }
    return MiohValidatedPortableModelCollection(
      identifier: identifier,
      assetURLsBySourceName: assets,
      canonicalSHA256: entry.sha256
    )
  }

  public func validateCoreAIModelCollectionDigest(
    identifier: String,
    beneath modelRoot: URL
  ) throws -> MiohValidatedPortableModelCollection {
    guard let entry = models[identifier],
      entry.assetType == "source-collection",
      !entry.sourceAssets.isEmpty
    else { throw MiohClusterAssetError.missingModelIdentity }
    var assets: [String: URL] = [:]
    var digestRoots: [(String, URL)] = []
    for source in entry.sourceAssets {
      guard let relativePath = MiohClusterRelativePath(rawValue: source) else {
        throw MiohClusterAssetError.invalidIdentityManifest
      }
      let url = try relativePath.resolve(beneath: modelRoot)
      assets[source] = url
      digestRoots.append((source, url))
    }
    guard try MiohClusterAsset.sha256Collection(digestRoots) == entry.sha256 else {
      throw MiohClusterAssetError.identityDigestMismatch
    }
    return MiohValidatedPortableModelCollection(
      identifier: identifier,
      assetURLsBySourceName: assets,
      canonicalSHA256: entry.sha256
    )
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy {
      (48...57).contains($0) || (97...102).contains($0)
    }
  }
}

public struct MiohValidatedPortableModelIdentity: Equatable, Sendable {
  public let identifier: String
  public let assetURL: URL
  public let canonicalSHA256: String
  public let functionName: String

  public init(
    identifier: String,
    assetURL: URL,
    canonicalSHA256: String,
    functionName: String
  ) {
    self.identifier = identifier
    self.assetURL = assetURL
    self.canonicalSHA256 = canonicalSHA256
    self.functionName = functionName
  }
}

public struct MiohValidatedPortableModelCollection: Equatable, Sendable {
  public let identifier: String
  public let assetURLsBySourceName: [String: URL]
  public let canonicalSHA256: String

  public init(
    identifier: String,
    assetURLsBySourceName: [String: URL],
    canonicalSHA256: String
  ) {
    self.identifier = identifier
    self.assetURLsBySourceName = assetURLsBySourceName
    self.canonicalSHA256 = canonicalSHA256
  }
}

public enum MiohCoreAIAssetValidator {
  /// Loads the source/compiled asset and verifies the function before a worker
  /// advertises its model identifier. Loading is intentionally required:
  /// filename-only checks would create a worker that accepts jobs it cannot run.
  public static func validateMainFunction(at url: URL) async throws
    -> MiohCoreAIAssetValidation
  {
    #if canImport(CoreAI)
      if #available(iOS 27.0, macOS 27.0, *) {
        let model = try await AIModel(contentsOf: url)
        guard let _ = try model.loadFunction(named: "main") else {
          throw MiohClusterAssetError.missingMainFunction
        }
        return try MiohCoreAIAssetValidation(
          url: url,
          sha256: MiohClusterAsset.sha256(url)
        )
      }
    #endif
    throw MiohClusterAssetError.missingCoreAI
  }
}
