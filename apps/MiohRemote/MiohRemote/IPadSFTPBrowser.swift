import AVFoundation
import CryptoKit
import Foundation
import MiohSFTPKit
import SwiftUI
import UIKit

private enum IPadSFTPMovieMetadataError: LocalizedError {
  case invalidVideo

  var errorDescription: String? {
    "SFTP上の動画情報を読み取れませんでした。"
  }
}

struct IPadSFTPMovieMetadata: Identifiable, Equatable, Sendable {
  let path: String
  let name: String
  let container: String
  let byteCount: Int64
  let modifiedAt: Date?
  let duration: TimeInterval
  let width: Int?
  let height: Int?
  let framesPerSecond: Double?
  let codec: String
  let hasAudio: Bool

  var id: String { path }

  var inlineSummary: String {
    var components = [Self.durationText(duration)]
    if let width, let height { components.append("\(width)×\(height)") }
    if let framesPerSecond {
      components.append(String(format: "%.2f fps", framesPerSecond))
    }
    components.append(codec)
    return components.joined(separator: " • ")
  }

  var averageBitRateText: String {
    guard duration > 0 else { return "不明" }
    let megabitsPerSecond = Double(byteCount) * 8 / duration / 1_000_000
    return String(format: "%.2f Mbps", megabitsPerSecond)
  }

  static func load(
    from input: IPadSFTPStreamingInput,
    entry: MiohSFTPEntry
  ) async throws -> IPadSFTPMovieMetadata {
    let asset = AVURLAsset(url: input.localURL)
    async let loadedDuration = asset.load(.duration)
    async let loadedVideoTracks = asset.loadTracks(withMediaType: .video)
    async let loadedAudioTracks = asset.loadTracks(withMediaType: .audio)
    let (duration, videoTracks, audioTracks) = try await (
      loadedDuration,
      loadedVideoTracks,
      loadedAudioTracks
    )
    guard duration.isNumeric, duration.seconds.isFinite, duration.seconds > 0,
      let videoTrack = videoTracks.first
    else { throw IPadSFTPMovieMetadataError.invalidVideo }

    async let loadedNaturalSize = videoTrack.load(.naturalSize)
    async let loadedTransform = videoTrack.load(.preferredTransform)
    async let loadedFrameRate = videoTrack.load(.nominalFrameRate)
    async let loadedDescriptions = videoTrack.load(.formatDescriptions)
    let (naturalSize, transform, nominalFrameRate, formatDescriptions) = try await (
      loadedNaturalSize,
      loadedTransform,
      loadedFrameRate,
      loadedDescriptions
    )
    let transformedSize = naturalSize.applying(transform)
    let width = Int(abs(transformedSize.width).rounded())
    let height = Int(abs(transformedSize.height).rounded())
    let frameRate = Double(nominalFrameRate)
    let mediaSubtype = formatDescriptions.first.map {
      CMFormatDescriptionGetMediaSubType($0)
    }

    return IPadSFTPMovieMetadata(
      path: entry.path,
      name: entry.name,
      container: URL(fileURLWithPath: entry.name).pathExtension.uppercased(),
      byteCount: input.byteCount,
      modifiedAt: entry.modifiedAt,
      duration: duration.seconds,
      width: width > 0 ? width : nil,
      height: height > 0 ? height : nil,
      framesPerSecond: frameRate.isFinite && frameRate > 0 ? frameRate : nil,
      codec: codecName(mediaSubtype),
      hasAudio: !audioTracks.isEmpty
    )
  }

  static func durationText(_ duration: TimeInterval) -> String {
    let total = max(0, Int(duration.rounded()))
    let hours = total / 3_600
    let minutes = (total % 3_600) / 60
    let seconds = total % 60
    return hours > 0
      ? String(format: "%d:%02d:%02d", hours, minutes, seconds)
      : String(format: "%02d:%02d", minutes, seconds)
  }

  private static func codecName(_ subtype: FourCharCode?) -> String {
    guard let subtype else { return "コーデック不明" }
    switch fourCC(subtype) {
    case "hvc1", "hev1": return "HEVC"
    case "avc1": return "H.264"
    case "av01": return "AV1"
    case "vp09": return "VP9"
    case "mp4v": return "MPEG-4 Video"
    case "jpeg": return "Motion JPEG"
    case "ap4h": return "ProRes 4444"
    case "apch": return "ProRes 422 HQ"
    case "apcn": return "ProRes 422"
    case "apcs": return "ProRes 422 LT"
    case "apco": return "ProRes 422 Proxy"
    default: return fourCC(subtype)
    }
  }

  private static func fourCC(_ value: FourCharCode) -> String {
    let bytes: [UInt8] = [
      UInt8((value >> 24) & 0xff),
      UInt8((value >> 16) & 0xff),
      UInt8((value >> 8) & 0xff),
      UInt8(value & 0xff),
    ]
    return String(bytes: bytes, encoding: .ascii) ?? String(format: "0x%08X", value)
  }
}

enum IPadSFTPBrowserMode {
  case selectInput
  case upload(outputURL: URL)

  var title: String {
    switch self {
    case .selectInput: "SFTPから動画を選択"
    case .upload: "SFTPへアップロード"
    }
  }
}

@MainActor
final class IPadSFTPStore: ObservableObject {
  private static let processTemporaryFileCutoff = Date()
  private static let temporaryInputPrefix = "mioh-sftp-input-"
  private static let persistentDownloadDirectoryName = "SFTP Downloads"
  private static let resumableDownloadDirectoryName = "SFTP Resume"
  private static let resumableDownloadPrefix = "mioh-sftp-resume-"
  private static let maximumPersistentNameAttempts = 1_000
  private static let maximumTemporaryEntriesToInspect = 4_096
  private static let maximumTemporaryFilesToRemove = 256
  private static let maximumResumeEntriesToInspect = 4_096
  private static let maximumExpiredResumeFilesToRemove = 256
  private static let maximumCachedMovieMetadataEntries = 256
  private static let resumeRetentionSeconds: TimeInterval = 30 * 24 * 60 * 60
  private static let idleBackgroundGraceNanoseconds: UInt64 = 20_000_000_000
  private static var activeResumeDestinations: Set<String> = []

  enum Activity: Equatable {
    case idle
    case checkingHost
    case connecting
    case listing
    case inspecting(String)
    case disconnecting
    case downloading(String)
    case preparingStream(String)
    case uploading(String)

    var label: String {
      switch self {
      case .idle: ""
      case .checkingHost: "ホスト鍵を確認中…"
      case .connecting: "接続中…"
      case .listing: "フォルダを読み込み中…"
      case .inspecting(let name): "\(name)の動画情報を確認中…"
      case .disconnecting: "切断中…"
      case .downloading(let name): "\(name)を取得中…"
      case .preparingStream(let name): "\(name)のストリーミングを準備中…"
      case .uploading(let name): "\(name)を送信中…"
      }
    }

    var isTransfer: Bool {
      switch self {
      case .downloading, .preparingStream, .uploading: true
      default: false
      }
    }

    var reportsByteProgress: Bool {
      switch self {
      case .downloading, .uploading: true
      default: false
      }
    }
  }

  private enum DefaultsKey {
    static let host = "mioh.ipad.sftp.host.v1"
    static let port = "mioh.ipad.sftp.port.v1"
    static let username = "mioh.ipad.sftp.username.v1"
    static let path = "mioh.ipad.sftp.path.v1"
    static let rememberPassword = "mioh.ipad.sftp.remember-password.v1"
  }

  @Published var host: String
  @Published var portText: String
  @Published var username: String
  @Published var password: String
  @Published var startingPath: String
  @Published var rememberPassword: Bool
  @Published private(set) var isConnected = false
  @Published private(set) var currentPath = "/"
  @Published private(set) var entries: [MiohSFTPEntry] = []
  @Published private(set) var activity: Activity = .idle
  @Published private(set) var transferredBytes: Int64 = 0
  @Published private(set) var totalBytes: Int64 = 0
  @Published private(set) var resumedBytes: Int64 = 0
  @Published private(set) var transferBytesPerSecond: Double = 0
  @Published private(set) var estimatedRemainingSeconds: TimeInterval?
  @Published private(set) var movieMetadataByPath: [String: IPadSFTPMovieMetadata] = [:]
  @Published private(set) var inspectingMetadataPath: String?
  @Published var presentedMovieMetadata: IPadSFTPMovieMetadata?
  @Published private(set) var errorMessage: String?
  @Published private(set) var successMessage: String?
  @Published private(set) var pendingHostIdentity: MiohSFTPHostIdentity?
  @Published private(set) var pendingHostEndpoint: String?
  @Published private(set) var hasSavedHostKey = false

  private let session = MiohSFTPSession()
  private var pendingConfiguration: MiohSFTPConfiguration?
  private var activeConfiguration: MiohSFTPConfiguration?
  private var activeTrustedHostKey: String?
  private var operationTask: Task<Void, Never>?
  private var operationID: UUID?
  private var disconnectID: UUID?
  private var previousIdleTimerDisabled: Bool?
  private var isInBackground = false
  private var backgroundDisconnectTask: Task<Void, Never>?
  private var backgroundTaskIdentifier: UIBackgroundTaskIdentifier = .invalid
  private var lastProgressSampleTime: TimeInterval?
  private var lastProgressSampleBytes: Int64 = 0
  private var movieMetadataCacheOrder: [String] = []
  private var metadataInspectionInput: IPadSFTPStreamingInput?

  init(defaults: UserDefaults = .standard) {
    host = defaults.string(forKey: DefaultsKey.host) ?? ""
    let savedPort = defaults.integer(forKey: DefaultsKey.port)
    portText = String(savedPort == 0 ? 22 : savedPort)
    username = defaults.string(forKey: DefaultsKey.username) ?? ""
    startingPath = defaults.string(forKey: DefaultsKey.path) ?? "."
    rememberPassword = defaults.object(forKey: DefaultsKey.rememberPassword) == nil
      ? true : defaults.bool(forKey: DefaultsKey.rememberPassword)
    password = ""
    restoreStoredSecurityState()
    Self.removeOrphanedInputFiles(before: Self.processTemporaryFileCutoff)
    Self.removeExpiredResumeFiles(
      before: Date().addingTimeInterval(-Self.resumeRetentionSeconds)
    )
  }

  var isBusy: Bool { activity != .idle }

  var progress: Double {
    guard totalBytes > 0 else { return 0 }
    return min(1, max(0, Double(transferredBytes) / Double(totalBytes)))
  }

  var canNavigateToParent: Bool { isConnected && currentPath != "/" }

  func reloadStoredCredentials() {
    guard !isConnected, !isBusy else { return }
    restoreStoredSecurityState()
  }

  func passwordPersistenceChanged() {
    UserDefaults.standard.set(
      rememberPassword,
      forKey: DefaultsKey.rememberPassword
    )
    guard !rememberPassword, let endpointID = currentEndpointID(),
      !username.isEmpty
    else { return }
    MiohSFTPCredentialStore.removePassword(
      endpointID: endpointID,
      username: username
    )
  }

  func connect() {
    guard !isBusy, !isConnected else { return }
    do {
      let configuration = try makeConfiguration()
      clearMessages()
      clearMovieMetadataCache()
      pendingHostIdentity = nil
      pendingHostEndpoint = nil
      pendingConfiguration = nil
      let operationID = begin(.checkingHost)
      operationTask = Task { [weak self] in
        guard let self else { return }
        await self.checkHostAndConnect(configuration, operationID: operationID)
      }
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func trustPendingHostAndConnect() {
    guard !isBusy, let identity = pendingHostIdentity,
      let configuration = pendingConfiguration
    else { return }
    do {
      try MiohSFTPCredentialStore.saveHostKey(
        identity.key,
        endpointID: configuration.endpointID
      )
      hasSavedHostKey = true
      pendingHostIdentity = nil
      pendingHostEndpoint = nil
      pendingConfiguration = nil
      clearMessages()
      let operationID = begin(.connecting)
      operationTask = Task { [weak self] in
        guard let self else { return }
        await self.establishConnection(
          configuration,
          trustedIdentity: identity,
          operationID: operationID
        )
      }
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func rejectPendingHost() {
    pendingHostIdentity = nil
    pendingHostEndpoint = nil
    pendingConfiguration = nil
    activeConfiguration = nil
    activeTrustedHostKey = nil
    errorMessage = "SFTPサーバーを信頼しなかったため、接続を中止しました。"
  }

  func removeSavedHostKey() {
    guard !isConnected, !isBusy, let endpointID = currentEndpointID() else {
      return
    }
    MiohSFTPCredentialStore.removeHostKey(endpointID: endpointID)
    hasSavedHostKey = false
    errorMessage = nil
    successMessage = "保存済みのホスト鍵を削除しました。次回接続時に再確認します。"
  }

  func disconnect() {
    guard activity != .disconnecting else { return }
    cancelBackgroundGrace(endSystemTask: true)
    activity = .disconnecting
    cancelCurrentOperationAndThenDisconnect()
    isConnected = false
    currentPath = "/"
    entries = []
    restoreIdleTimer()
    transferredBytes = 0
    totalBytes = 0
    resumedBytes = 0
    clearTransferMetrics()
    clearMovieMetadataCache()
    pendingHostIdentity = nil
    pendingHostEndpoint = nil
    pendingConfiguration = nil
    activeConfiguration = nil
    activeTrustedHostKey = nil
    if !rememberPassword { password = "" }
  }

  func cancelAndDisconnect() {
    disconnect()
  }

  /// iOS does not offer background URLSession semantics for SSH. Keep the
  /// socket and an active transfer alive for the finite background-task grace
  /// period instead of tearing them down at the scene transition itself.
  func enterBackground() {
    guard !isInBackground else { return }
    isInBackground = true
    guard isConnected || isBusy else { return }
    beginSystemBackgroundTaskIfNeeded()
    if !activity.isTransfer {
      scheduleBackgroundDisconnect()
    }
  }

  func enterForeground() {
    guard isInBackground || backgroundTaskIdentifier != .invalid else { return }
    isInBackground = false
    cancelBackgroundGrace(endSystemTask: true)
  }

  func cancelTransfer() {
    guard activity.isTransfer else { return }
    cancelBackgroundGrace(endSystemTask: true)
    activity = .disconnecting
    cancelCurrentOperationAndThenDisconnect()
    restoreIdleTimer()
    transferredBytes = 0
    totalBytes = 0
    resumedBytes = 0
    clearTransferMetrics()
    clearMovieMetadataCache()
    successMessage = nil
    isConnected = false
    entries = []
    activeConfiguration = nil
    activeTrustedHostKey = nil
    if !rememberPassword { password = "" }
    errorMessage = "SFTP転送を中止しました。ダウンロードは再接続後に同じファイルを選ぶと続きから再開できます。"
  }

  func refresh() {
    loadDirectory(currentPath)
  }

  func open(_ entry: MiohSFTPEntry) {
    guard entry.isDirectory else { return }
    loadDirectory(entry.path)
  }

  func openParent() {
    guard canNavigateToParent else { return }
    do {
      loadDirectory(try MiohSFTPPath.parent(of: currentPath))
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func inspectMetadata(_ entry: MiohSFTPEntry) {
    guard isConnected, !isBusy, entry.kind == .movie else { return }
    if let cached = cachedMetadata(for: entry) {
      presentedMovieMetadata = cached
      return
    }
    guard let configuration = activeConfiguration,
      let trustedHostKey = activeTrustedHostKey
    else { return }
    clearMessages()
    inspectingMetadataPath = entry.path
    let operationID = begin(.inspecting(entry.name))
    operationTask = Task { [weak self] in
      guard let self else { return }
      do {
        let input = try await IPadSFTPStreamingInput.start(
          configuration: configuration,
          trustedHostKey: trustedHostKey,
          entry: entry
        )
        guard self.operationID == operationID, !Task.isCancelled else {
          input.stop()
          return
        }
        self.metadataInspectionInput = input
        defer {
          input.stop()
          if self.metadataInspectionInput === input {
            self.metadataInspectionInput = nil
          }
        }
        let metadata = try await IPadSFTPMovieMetadata.load(
          from: input,
          entry: entry
        )
        try Task.checkCancellation()
        guard self.operationID == operationID else { return }
        self.cacheMovieMetadata(metadata)
        self.presentedMovieMetadata = metadata
        self.finishOperation(operationID)
      } catch {
        self.finishOperation(operationID, error: error)
      }
    }
  }

  func cachedMetadata(for entry: MiohSFTPEntry) -> IPadSFTPMovieMetadata? {
    guard let metadata = movieMetadataByPath[entry.path],
      entry.byteCount == nil || entry.byteCount == metadata.byteCount,
      entry.modifiedAt == metadata.modifiedAt
    else { return nil }
    return metadata
  }

  func isInspectingMetadata(_ entry: MiohSFTPEntry) -> Bool {
    inspectingMetadataPath == entry.path
  }

  func download(
    _ entry: MiohSFTPEntry,
    completion: @escaping @MainActor (URL) -> Void
  ) {
    guard isConnected, !isBusy, entry.kind == .movie,
      let configuration = activeConfiguration,
      let trustedHostKey = activeTrustedHostKey
    else { return }
    do {
      let fileManager = FileManager.default
      let resumableDestination = try Self.resumableDownloadURL(
        entry: entry,
        configuration: configuration,
        trustedHostKey: trustedHostKey,
        fileManager: fileManager
      )
      let destinationKey = resumableDestination.standardizedFileURL.path
      let documentsDirectory = try fileManager.url(
        for: .documentDirectory,
        in: .userDomainMask,
        appropriateFor: nil,
        create: true
      )
      let values = try documentsDirectory.resourceValues(forKeys: [
        .volumeAvailableCapacityForImportantUsageKey,
        .volumeAvailableCapacityKey,
      ])
      let availableBytes = values.volumeAvailableCapacityForImportantUsage
        ?? values.volumeAvailableCapacity.map(Int64.init) ?? 0
      let retainedBytes = Self.safePartialByteCount(
        for: resumableDestination,
        fileManager: fileManager
      )
      let maximumBytes = try MiohSFTPTransferPolicy.maximumResumableDownloadBytes(
        availableBytes: availableBytes,
        retainedPartialBytes: retainedBytes
      )
      guard Self.activeResumeDestinations.insert(destinationKey).inserted else {
        throw MiohSFTPError.localFile("同じファイルのダウンロードが既に進行中です。")
      }

      clearMessages()
      transferredBytes = retainedBytes
      totalBytes = entry.byteCount ?? 0
      resumedBytes = retainedBytes
      startTransferMetrics(initialBytes: retainedBytes)
      let operationID = begin(.downloading(entry.name))
      preventIdleTimer()
      operationTask = Task { [weak self] in
        defer { Self.activeResumeDestinations.remove(destinationKey) }
        guard let self else { return }
        var publishedURL: URL?
        do {
          let result = try await self.session.downloadMovie(
            remotePath: entry.path,
            to: resumableDestination,
            maximumBytes: maximumBytes,
            resumeExistingPartial: true
          ) { [weak self] completed, total in
            Task { @MainActor [weak self] in
              self?.updateProgress(
                completed: completed,
                total: total,
                operationID: operationID
              )
            }
          }
          try Task.checkCancellation()
          guard self.operationID == operationID else {
            try? fileManager.removeItem(at: result)
            return
          }
          let persistentURL = try Self.persistentDownloadURL(
            fileName: entry.name,
            fileManager: fileManager
          )
          try fileManager.moveItem(at: result, to: persistentURL)
          publishedURL = persistentURL
          try Task.checkCancellation()
          guard self.operationID == operationID else {
            try? fileManager.removeItem(at: persistentURL)
            return
          }
          self.finishOperation(operationID)
          self.successMessage =
            "Filesの「このiPad内 > mioh Remote > SFTP Downloads」に保存しました。"
          completion(persistentURL)
          publishedURL = nil
        } catch {
          // The session retains its metadata-bound .part file on cancellation
          // and transport failure. Only a fully staged result is discarded
          // here if publishing to Files did not complete.
          try? fileManager.removeItem(at: resumableDestination)
          if let publishedURL { try? fileManager.removeItem(at: publishedURL) }
          self.finishOperation(operationID, error: error)
        }
      }
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func startStreaming(
    _ entry: MiohSFTPEntry,
    completion: @escaping @MainActor (IPadSFTPStreamingInput) -> Void
  ) {
    guard isConnected, !isBusy, entry.kind == .movie,
      let configuration = activeConfiguration,
      let trustedHostKey = activeTrustedHostKey
    else { return }
    clearMessages()
    transferredBytes = 0
    totalBytes = 0
    let operationID = begin(.preparingStream(entry.name))
    preventIdleTimer()
    operationTask = Task { [weak self] in
      guard let self else { return }
      do {
        let input = try await IPadSFTPStreamingInput.start(
          configuration: configuration,
          trustedHostKey: trustedHostKey,
          entry: entry
        )
        try Task.checkCancellation()
        guard self.operationID == operationID else {
          input.stop()
          return
        }
        self.finishOperation(operationID)
        self.successMessage = "SFTPストリーミングの準備が完了しました。"
        completion(input)
      } catch {
        self.finishOperation(operationID, error: error)
      }
    }
  }

  func reportStreamingSelectionFailure() {
    successMessage = nil
    errorMessage = "選択したファイルを復元再生用の動画として開けませんでした。"
  }

  func reportStreamingPlaybackUnavailable(_ reason: String) {
    successMessage = nil
    errorMessage = reason
  }

  func upload(
    localURL: URL,
    fileName: String,
    completion: @escaping @MainActor (String) -> Void
  ) {
    guard isConnected, !isBusy else { return }
    do {
      let destination = try MiohSFTPPath.appending(fileName, to: currentPath)
      guard MiohSFTPPath.isSupportedMovie(destination) else {
        throw MiohSFTPError.unsupportedFile
      }
      clearMessages()
      transferredBytes = 0
      let size = try localURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
      totalBytes = Int64(size)
      startTransferMetrics(initialBytes: 0)
      let operationID = begin(.uploading(fileName))
      preventIdleTimer()
      operationTask = Task { [weak self] in
        guard let self else { return }
        do {
          let remotePath = try await self.session.uploadMovie(
            localURL: localURL,
            to: destination
          ) { [weak self] completed, total in
            Task { @MainActor [weak self] in
              self?.updateProgress(
                completed: completed,
                total: total,
                operationID: operationID
              )
            }
          }
          try Task.checkCancellation()
          guard self.operationID == operationID else { return }
          self.finishOperation(operationID)
          self.successMessage = "SFTPへのアップロードが完了しました。"
          completion(remotePath)
        } catch {
          self.finishOperation(operationID, error: error)
        }
      }
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  private func checkHostAndConnect(
    _ configuration: MiohSFTPConfiguration,
    operationID: UUID
  ) async {
    do {
      let actual = try await MiohSFTPSession.fetchHostIdentity(
        configuration: configuration
      )
      try Task.checkCancellation()
      guard self.operationID == operationID else { return }
      if let savedKey = MiohSFTPCredentialStore.loadHostKey(
        endpointID: configuration.endpointID
      ) {
        let expected = try MiohSFTPHostIdentity(shortHandKey: savedKey)
        guard expected.key == actual.key else {
          throw MiohSFTPError.hostKeyChanged(
            expected: expected.fingerprint,
            actual: actual.fingerprint
          )
        }
        activity = .connecting
        await establishConnection(
          configuration,
          trustedIdentity: expected,
          operationID: operationID
        )
      } else {
        finishOperation(operationID)
        pendingConfiguration = configuration
        pendingHostEndpoint = configuration.endpointID
        pendingHostIdentity = actual
      }
    } catch {
      finishOperation(operationID, error: error)
    }
  }

  private func establishConnection(
    _ configuration: MiohSFTPConfiguration,
    trustedIdentity: MiohSFTPHostIdentity,
    operationID: UUID
  ) async {
    do {
      let home = try await session.connect(
        configuration: configuration,
        trustedHostKey: trustedIdentity.key
      )
      try Task.checkCancellation()
      guard self.operationID == operationID else {
        await session.disconnect()
        return
      }
      activity = .listing
      let requestedPath = try MiohSFTPPath.normalize(
        startingPath,
        relativeTo: home
      )
      let listing: (String, [MiohSFTPEntry])
      do {
        listing = (requestedPath, try await session.listDirectory(requestedPath))
      } catch where requestedPath != home {
        listing = (home, try await session.listDirectory(home))
      }
      try Task.checkCancellation()
      guard self.operationID == operationID else {
        await session.disconnect()
        return
      }
      currentPath = listing.0
      entries = listing.1
      startingPath = listing.0
      isConnected = true
      activeConfiguration = configuration
      activeTrustedHostKey = trustedIdentity.key
      hasSavedHostKey = true
      persistProfile(configuration: configuration, path: listing.0)
      try updateStoredPassword(configuration)
      finishOperation(operationID)
      successMessage = "SFTPサーバーへ安全に接続しました。"
    } catch {
      await session.disconnect()
      isConnected = false
      activeConfiguration = nil
      activeTrustedHostKey = nil
      finishOperation(operationID, error: error)
    }
  }

  private func loadDirectory(_ path: String) {
    guard isConnected, !isBusy else { return }
    clearMessages()
    let operationID = begin(.listing)
    operationTask = Task { [weak self] in
      guard let self else { return }
      do {
        let cleanPath = try MiohSFTPPath.normalize(path)
        let entries = try await self.session.listDirectory(cleanPath)
        try Task.checkCancellation()
        guard self.operationID == operationID else { return }
        self.currentPath = cleanPath
        self.startingPath = cleanPath
        self.entries = entries
        UserDefaults.standard.set(cleanPath, forKey: DefaultsKey.path)
        self.finishOperation(operationID)
      } catch {
        self.finishOperation(operationID, error: error)
      }
    }
  }

  private func makeConfiguration() throws -> MiohSFTPConfiguration {
    guard let port = Int(portText.trimmingCharacters(in: .whitespacesAndNewlines))
    else { throw MiohSFTPError.invalidPort }
    return try MiohSFTPConfiguration(
      host: host,
      port: port,
      username: username,
      password: password
    ).validated()
  }

  private func currentEndpointID() -> String? {
    guard let port = Int(portText.trimmingCharacters(in: .whitespacesAndNewlines))
    else { return nil }
    return try? MiohSFTPConfiguration(
      host: host,
      port: port,
      username: username.isEmpty ? "_" : username,
      password: "_"
    ).validated().endpointID
  }

  private func restoreStoredSecurityState() {
    guard let port = Int(portText),
      let configuration = try? MiohSFTPConfiguration(
        host: host,
        port: port,
        username: username.isEmpty ? "_" : username,
        password: "_"
      ).validated()
    else {
      hasSavedHostKey = false
      password = ""
      return
    }
    hasSavedHostKey = MiohSFTPCredentialStore.loadHostKey(
      endpointID: configuration.endpointID
    ) != nil
    if rememberPassword, !username.isEmpty {
      password = MiohSFTPCredentialStore.loadPassword(
        endpointID: configuration.endpointID,
        username: username
      ) ?? ""
    } else {
      password = ""
    }
  }

  private func persistProfile(
    configuration: MiohSFTPConfiguration,
    path: String
  ) {
    let defaults = UserDefaults.standard
    defaults.set(configuration.host, forKey: DefaultsKey.host)
    defaults.set(configuration.port, forKey: DefaultsKey.port)
    defaults.set(configuration.username, forKey: DefaultsKey.username)
    defaults.set(path, forKey: DefaultsKey.path)
    defaults.set(rememberPassword, forKey: DefaultsKey.rememberPassword)
  }

  private func updateStoredPassword(
    _ configuration: MiohSFTPConfiguration
  ) throws {
    if rememberPassword {
      try MiohSFTPCredentialStore.savePassword(
        configuration.password,
        endpointID: configuration.endpointID,
        username: configuration.username
      )
    } else {
      MiohSFTPCredentialStore.removePassword(
        endpointID: configuration.endpointID,
        username: configuration.username
      )
    }
  }

  private func begin(_ activity: Activity) -> UUID {
    let operationID = UUID()
    self.operationID = operationID
    self.activity = activity
    return operationID
  }

  private func finishOperation(_ operationID: UUID, error: Error? = nil) {
    guard self.operationID == operationID else { return }
    let wasTransfer = activity.isTransfer
    self.operationID = nil
    operationTask = nil
    activity = .idle
    inspectingMetadataPath = nil
    if wasTransfer { restoreIdleTimer() }
    if isInBackground, wasTransfer {
      endSystemBackgroundTask()
      scheduleBackgroundDisconnect()
    }
    if let error, !(error is CancellationError) {
      errorMessage = error.localizedDescription
    }
  }

  private func updateProgress(
    completed: Int64,
    total: Int64,
    operationID: UUID
  ) {
    guard self.operationID == operationID else { return }
    let reportedBytes = max(0, completed)
    // The session may deliberately restart at zero after rejecting stale
    // resume metadata. Accept that first reset, but ignore a genuinely stale
    // callback once payload-rate sampling has started.
    if reportedBytes < transferredBytes, transferBytesPerSecond > 0 { return }
    let completedBytes = reportedBytes
    let totalByteCount = max(0, total)
    if completedBytes == 0 { resumedBytes = 0 }

    let now = ProcessInfo.processInfo.systemUptime
    if let lastProgressSampleTime {
      let elapsed = now - lastProgressSampleTime
      let byteDelta = completedBytes - lastProgressSampleBytes
      if byteDelta > 0, elapsed >= 0.05 {
        let instantaneousRate = Double(byteDelta) / elapsed
        if instantaneousRate.isFinite, instantaneousRate > 0 {
          transferBytesPerSecond = transferBytesPerSecond > 0
            ? (transferBytesPerSecond * 0.65) + (instantaneousRate * 0.35)
            : instantaneousRate
        }
        self.lastProgressSampleTime = now
        lastProgressSampleBytes = completedBytes
      } else if byteDelta == 0, transferBytesPerSecond == 0 {
        // The first callback reports the verified resume offset. Start timing
        // after connection/open validation so ETA measures payload transfer.
        self.lastProgressSampleTime = now
        lastProgressSampleBytes = completedBytes
      } else if byteDelta < 0 {
        transferBytesPerSecond = 0
        estimatedRemainingSeconds = nil
        self.lastProgressSampleTime = now
        lastProgressSampleBytes = completedBytes
      }
    } else {
      lastProgressSampleTime = now
      lastProgressSampleBytes = completedBytes
    }

    transferredBytes = completedBytes
    totalBytes = totalByteCount
    let remainingBytes = max(0, totalByteCount - completedBytes)
    if remainingBytes > 0, transferBytesPerSecond.isFinite,
      transferBytesPerSecond > 0
    {
      let estimate = Double(remainingBytes) / transferBytesPerSecond
      estimatedRemainingSeconds = estimate.isFinite ? max(0, estimate) : nil
    } else {
      estimatedRemainingSeconds = nil
    }
  }

  private func startTransferMetrics(initialBytes: Int64) {
    transferBytesPerSecond = 0
    estimatedRemainingSeconds = nil
    lastProgressSampleTime = ProcessInfo.processInfo.systemUptime
    lastProgressSampleBytes = max(0, initialBytes)
  }

  private func clearTransferMetrics() {
    transferBytesPerSecond = 0
    estimatedRemainingSeconds = nil
    lastProgressSampleTime = nil
    lastProgressSampleBytes = 0
  }

  private func cancelCurrentOperationAndThenDisconnect() {
    let task = operationTask
    operationID = nil
    operationTask = nil
    task?.cancel()
    metadataInspectionInput?.stop()
    metadataInspectionInput = nil
    inspectingMetadataPath = nil
    let disconnectID = UUID()
    self.disconnectID = disconnectID
    Task { [session] in
      await task?.value
      await session.disconnect()
      guard self.disconnectID == disconnectID else { return }
      self.disconnectID = nil
      self.activity = .idle
    }
  }

  private func clearMessages() {
    errorMessage = nil
    successMessage = nil
  }

  private func cacheMovieMetadata(_ metadata: IPadSFTPMovieMetadata) {
    movieMetadataCacheOrder.removeAll { $0 == metadata.path }
    movieMetadataCacheOrder.append(metadata.path)
    movieMetadataByPath[metadata.path] = metadata
    while movieMetadataCacheOrder.count > Self.maximumCachedMovieMetadataEntries {
      let oldest = movieMetadataCacheOrder.removeFirst()
      movieMetadataByPath.removeValue(forKey: oldest)
    }
  }

  private func clearMovieMetadataCache() {
    movieMetadataByPath.removeAll()
    movieMetadataCacheOrder.removeAll()
    presentedMovieMetadata = nil
    inspectingMetadataPath = nil
  }

  private func preventIdleTimer() {
    guard previousIdleTimerDisabled == nil else { return }
    previousIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
    UIApplication.shared.isIdleTimerDisabled = true
  }

  private func restoreIdleTimer() {
    guard let previousIdleTimerDisabled else { return }
    self.previousIdleTimerDisabled = nil
    UIApplication.shared.isIdleTimerDisabled = previousIdleTimerDisabled
  }

  private func beginSystemBackgroundTaskIfNeeded() {
    guard backgroundTaskIdentifier == .invalid else { return }
    backgroundTaskIdentifier = UIApplication.shared.beginBackgroundTask(
      withName: "mioh SFTP transfer"
    ) { [weak self] in
      Task { @MainActor [weak self] in
        self?.backgroundGraceExpired()
      }
    }
  }

  private func endSystemBackgroundTask() {
    guard backgroundTaskIdentifier != .invalid else { return }
    let identifier = backgroundTaskIdentifier
    backgroundTaskIdentifier = .invalid
    UIApplication.shared.endBackgroundTask(identifier)
  }

  private func scheduleBackgroundDisconnect() {
    backgroundDisconnectTask?.cancel()
    backgroundDisconnectTask = Task { @MainActor [weak self] in
      do {
        try await Task.sleep(nanoseconds: Self.idleBackgroundGraceNanoseconds)
      } catch {
        return
      }
      guard let self, self.isInBackground else { return }
      self.backgroundGraceExpired()
    }
  }

  private func cancelBackgroundGrace(endSystemTask: Bool) {
    backgroundDisconnectTask?.cancel()
    backgroundDisconnectTask = nil
    if endSystemTask { endSystemBackgroundTask() }
  }

  private func backgroundGraceExpired() {
    guard isInBackground else {
      cancelBackgroundGrace(endSystemTask: true)
      return
    }
    let hadDownload: Bool
    if case .downloading = activity {
      hadDownload = true
    } else {
      hadDownload = false
    }
    disconnect()
    errorMessage = hadDownload
      ? "バックグラウンド実行の猶予が終了しました。同じファイルを選ぶと続きから再開します。"
      : "バックグラウンド実行の猶予が終了したため、SFTP接続を閉じました。"
  }

  private static func resumableDownloadURL(
    entry: MiohSFTPEntry,
    configuration: MiohSFTPConfiguration,
    trustedHostKey: String,
    fileManager: FileManager
  ) throws -> URL {
    guard entry.kind == .movie, MiohSFTPPath.isSupportedMovie(entry.name) else {
      throw MiohSFTPError.unsupportedFile
    }
    let directory = try resumableDownloadDirectory(
      fileManager: fileManager,
      create: true
    )
    let identity = [
      configuration.credentialID,
      trustedHostKey,
      entry.path,
    ].joined(separator: "\0")
    let digest = SHA256.hash(data: Data(identity.utf8))
      .map { String(format: "%02x", $0) }
      .joined()
    let pathExtension = URL(fileURLWithPath: entry.name).pathExtension.lowercased()
    guard digest.count == 64, MiohSFTPPath.movieExtensions.contains(pathExtension)
    else { throw MiohSFTPError.unsupportedFile }
    return directory
      .appendingPathComponent("\(resumableDownloadPrefix)\(digest)")
      .appendingPathExtension(pathExtension)
  }

  private static func resumableDownloadDirectory(
    fileManager: FileManager,
    create: Bool
  ) throws -> URL {
    let applicationSupport = try fileManager.url(
      for: .applicationSupportDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: create
    ).standardizedFileURL
    var directory = applicationSupport.appendingPathComponent(
      resumableDownloadDirectoryName,
      isDirectory: true
    )
    let exists = fileManager.fileExists(atPath: directory.path)
      || (try? fileManager.destinationOfSymbolicLink(atPath: directory.path)) != nil
    if exists {
      let values = try directory.resourceValues(forKeys: [
        .isDirectoryKey,
        .isSymbolicLinkKey,
      ])
      guard values.isDirectory == true, values.isSymbolicLink != true else {
        throw MiohSFTPError.localFile("再開用フォルダを安全に使用できません。")
      }
    } else if create {
      try fileManager.createDirectory(
        at: directory,
        withIntermediateDirectories: false
      )
    } else {
      throw MiohSFTPError.localFile("再開用フォルダがありません。")
    }
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try? directory.setResourceValues(values)
    return directory
  }

  private static func safePartialByteCount(
    for destinationURL: URL,
    fileManager: FileManager
  ) -> Int64 {
    let partialURL = destinationURL.appendingPathExtension("part")
    guard let values = try? partialURL.resourceValues(forKeys: [
      .isRegularFileKey,
      .isSymbolicLinkKey,
      .fileSizeKey,
    ]), values.isRegularFile == true, values.isSymbolicLink != true,
      let byteCount = values.fileSize, byteCount > 0
    else { return 0 }
    return Int64(byteCount)
  }

  private static func removeExpiredResumeFiles(before cutoff: Date) {
    let fileManager = FileManager.default
    guard let directory = try? resumableDownloadDirectory(
      fileManager: fileManager,
      create: false
    ) else { return }
    let keys: [URLResourceKey] = [
      .isRegularFileKey,
      .isSymbolicLinkKey,
      .contentModificationDateKey,
    ]
    guard let enumerator = fileManager.enumerator(
      at: directory,
      includingPropertiesForKeys: keys,
      options: [.skipsSubdirectoryDescendants],
      errorHandler: { _, _ in false }
    ) else { return }

    var inspected = 0
    var removed = 0
    for case let url as URL in enumerator {
      guard inspected < maximumResumeEntriesToInspect,
        removed < maximumExpiredResumeFilesToRemove
      else { break }
      inspected += 1
      let destinationPath = resumeDestinationPath(forArtifactURL: url)
      guard url.deletingLastPathComponent().standardizedFileURL == directory,
        isOwnedResumeFileName(url.lastPathComponent),
        !activeResumeDestinations.contains(destinationPath),
        let values = try? url.resourceValues(forKeys: Set(keys)),
        values.isRegularFile == true,
        values.isSymbolicLink != true,
        let modifiedAt = values.contentModificationDate,
        modifiedAt < cutoff
      else { continue }
      if (try? fileManager.removeItem(at: url)) != nil { removed += 1 }
    }
  }

  private static func resumeDestinationPath(forArtifactURL url: URL) -> String {
    if url.pathExtension == "part" || url.pathExtension == "resume" {
      return url.deletingPathExtension().standardizedFileURL.path
    }
    return url.standardizedFileURL.path
  }

  private static func isOwnedResumeFileName(_ name: String) -> Bool {
    guard name.hasPrefix(resumableDownloadPrefix) else { return false }
    let remainder = String(name.dropFirst(resumableDownloadPrefix.count))
    guard remainder.count > 64 else { return false }
    let digest = remainder.prefix(64)
    guard digest.utf8.count == 64,
      digest.utf8.allSatisfy({
        (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
      })
    else {
      return false
    }
    let suffix = String(remainder.dropFirst(64))
    return ["mp4", "mov", "m4v"].contains { pathExtension in
      suffix == ".\(pathExtension)"
        || suffix == ".\(pathExtension).part"
        || suffix == ".\(pathExtension).resume"
    }
  }

  /// Publishes completed SFTP downloads inside the app's Documents container,
  /// which is exposed to Files by UIFileSharingEnabled. The transfer itself is
  /// completed in private Application Support first, so Files never presents
  /// an incomplete movie.
  private static func persistentDownloadURL(
    fileName: String,
    fileManager: FileManager
  ) throws -> URL {
    guard !fileName.isEmpty,
      !fileName.contains("/"),
      !fileName.contains("\0"),
      MiohSFTPPath.isSupportedMovie(fileName)
    else { throw MiohSFTPError.unsupportedFile }

    let documentsDirectory = try fileManager.url(
      for: .documentDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true
    ).standardizedFileURL
    let downloadDirectory = documentsDirectory.appendingPathComponent(
      persistentDownloadDirectoryName,
      isDirectory: true
    )
    if fileManager.fileExists(atPath: downloadDirectory.path) {
      let values = try downloadDirectory.resourceValues(forKeys: [
        .isDirectoryKey,
        .isSymbolicLinkKey,
      ])
      guard values.isDirectory == true, values.isSymbolicLink != true else {
        throw MiohSFTPError.localFile(
          "FilesのSFTP Downloadsフォルダを安全に使用できません。"
        )
      }
    } else {
      try fileManager.createDirectory(
        at: downloadDirectory,
        withIntermediateDirectories: false
      )
    }

    let sourceName = URL(fileURLWithPath: fileName)
    let pathExtension = sourceName.pathExtension
    let baseName = sourceName.deletingPathExtension().lastPathComponent
    guard !baseName.isEmpty, !pathExtension.isEmpty else {
      throw MiohSFTPError.unsupportedFile
    }
    for attempt in 0..<maximumPersistentNameAttempts {
      let candidateName = attempt == 0
        ? fileName
        : "\(baseName) (\(attempt + 1)).\(pathExtension)"
      let candidate = downloadDirectory.appendingPathComponent(candidateName)
      guard candidate.deletingLastPathComponent().standardizedFileURL
        == downloadDirectory.standardizedFileURL
      else { throw MiohSFTPError.invalidRemotePath }
      if !fileManager.fileExists(atPath: candidate.path),
        (try? fileManager.destinationOfSymbolicLink(atPath: candidate.path)) == nil
      {
        return candidate
      }
    }
    throw MiohSFTPError.localFile(
      "FilesのSFTP Downloadsに保存できるファイル名を作成できませんでした。"
    )
  }

  private static func removeOrphanedInputFiles(before cutoff: Date) {
    let fileManager = FileManager.default
    let temporaryDirectory = fileManager.temporaryDirectory.standardizedFileURL
    let keys: [URLResourceKey] = [
      .isRegularFileKey,
      .isSymbolicLinkKey,
      .contentModificationDateKey,
    ]
    guard let enumerator = fileManager.enumerator(
      at: temporaryDirectory,
      includingPropertiesForKeys: keys,
      options: [.skipsSubdirectoryDescendants],
      errorHandler: { _, _ in false }
    ) else { return }

    var inspected = 0
    var removed = 0
    for case let url as URL in enumerator {
      guard inspected < maximumTemporaryEntriesToInspect,
        removed < maximumTemporaryFilesToRemove
      else { break }
      inspected += 1
      guard url.deletingLastPathComponent().standardizedFileURL == temporaryDirectory,
        isOwnedTemporaryInputName(url.lastPathComponent),
        let values = try? url.resourceValues(forKeys: Set(keys)),
        values.isRegularFile == true,
        values.isSymbolicLink != true,
        let modifiedAt = values.contentModificationDate,
        modifiedAt < cutoff
      else { continue }
      if (try? fileManager.removeItem(at: url)) != nil {
        removed += 1
      }
    }
  }

  private static func isOwnedTemporaryInputName(_ name: String) -> Bool {
    guard name.hasPrefix(temporaryInputPrefix) else { return false }
    let remainder = String(name.dropFirst(temporaryInputPrefix.count))
    for pathExtension in ["mp4", "mov", "m4v"] {
      for suffix in [".\(pathExtension)", ".\(pathExtension).part"]
      where remainder.hasSuffix(suffix) {
        let identifier = String(remainder.dropLast(suffix.count))
        return UUID(uuidString: identifier) != nil
      }
    }
    return false
  }
}

struct IPadSFTPBrowserView: View {
  @Environment(\.dismiss) private var dismiss
  @Environment(\.scenePhase) private var scenePhase
  @ObservedObject var store: IPadSFTPStore
  let mode: IPadSFTPBrowserMode
  let onDownloaded: (URL) -> Void
  let onStreamingRequested: (MiohSFTPEntry) -> Void
  let onUploaded: (String) -> Void
  let streamingUnavailableReason: String?

  @State private var uploadFileName: String
  @State private var confirmingHostKeyRemoval = false

  init(
    store: IPadSFTPStore,
    mode: IPadSFTPBrowserMode,
    onDownloaded: @escaping (URL) -> Void = { _ in },
    onStreamingRequested: @escaping (MiohSFTPEntry) -> Void = { _ in },
    onUploaded: @escaping (String) -> Void = { _ in },
    streamingUnavailableReason: String? = nil
  ) {
    self.store = store
    self.mode = mode
    self.onDownloaded = onDownloaded
    self.onStreamingRequested = onStreamingRequested
    self.onUploaded = onUploaded
    self.streamingUnavailableReason = streamingUnavailableReason
    switch mode {
    case .selectInput:
      _uploadFileName = State(initialValue: "")
    case .upload(let outputURL):
      _uploadFileName = State(initialValue: outputURL.lastPathComponent)
    }
  }

  var body: some View {
    NavigationStack {
      Group {
        if store.isConnected {
          browser
        } else {
          connectionForm
        }
      }
      .navigationTitle(mode.title)
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .cancellationAction) {
          Button("閉じる") {
            store.cancelAndDisconnect()
            dismiss()
          }
        }
        if store.isConnected {
          ToolbarItem(placement: .confirmationAction) {
            Button("切断") { store.disconnect() }
              .disabled(store.isBusy)
          }
        }
      }
    }
    .interactiveDismissDisabled(store.isBusy)
    .alert(
      "SFTPサーバーを信頼しますか？",
      isPresented: Binding(
        get: { store.pendingHostIdentity != nil },
        set: { _ in }
      ),
      presenting: store.pendingHostIdentity
    ) { _ in
      Button("キャンセル", role: .cancel) { store.rejectPendingHost() }
      Button("信頼して接続") { store.trustPendingHostAndConnect() }
    } message: { identity in
      Text(
        "接続先が正しいことを管理者に確認してください。\n\n接続先: \(store.pendingHostEndpoint ?? "不明")\n方式: \(identity.algorithm)\nフィンガープリント: \(identity.fingerprint)"
      )
    }
    .confirmationDialog(
      "保存済みのホスト鍵を削除しますか？",
      isPresented: $confirmingHostKeyRemoval,
      titleVisibility: .visible
    ) {
      Button("ホスト鍵を削除", role: .destructive) {
        store.removeSavedHostKey()
      }
      Button("キャンセル", role: .cancel) {}
    } message: {
      Text("次回接続時に新しいフィンガープリントの確認が必要になります。")
    }
    .sheet(item: $store.presentedMovieMetadata) { metadata in
      IPadSFTPMovieMetadataView(metadata: metadata)
    }
    .onChange(of: scenePhase) { phase in
      if phase == .background {
        store.enterBackground()
      } else if phase == .active {
        store.enterForeground()
      }
    }
    .onReceive(
      NotificationCenter.default.publisher(
        for: UIApplication.protectedDataWillBecomeUnavailableNotification
      )
    ) { _ in
      store.cancelAndDisconnect()
    }
    .onDisappear {
      if scenePhase == .background {
        store.enterBackground()
      } else {
        store.cancelAndDisconnect()
      }
    }
  }

  private var connectionForm: some View {
    Form {
      Section("接続先") {
        TextField("ホスト名またはIPアドレス", text: $store.host)
          .textInputAutocapitalization(.never)
          .autocorrectionDisabled()
          .onChange(of: store.host) { _ in store.reloadStoredCredentials() }
        TextField("ポート", text: $store.portText)
          .keyboardType(.numberPad)
          .onChange(of: store.portText) { _ in store.reloadStoredCredentials() }
        TextField("ユーザー名", text: $store.username)
          .textInputAutocapitalization(.never)
          .autocorrectionDisabled()
          .onChange(of: store.username) { _ in store.reloadStoredCredentials() }
        SecureField("パスワード", text: $store.password)
          .textContentType(.password)
        TextField("開始パス（例: /home/user/videos）", text: $store.startingPath)
          .textInputAutocapitalization(.never)
          .autocorrectionDisabled()
        Toggle("パスワードをKeychainに保存", isOn: $store.rememberPassword)
          .onChange(of: store.rememberPassword) { _ in
            store.passwordPersistenceChanged()
          }
      }
      .disabled(store.isBusy)

      Section {
        Button {
          store.connect()
        } label: {
          if store.isBusy {
            HStack {
              ProgressView()
              Text(store.activity.label)
            }
          } else {
            Label("接続", systemImage: "network")
          }
        }
        .disabled(store.isBusy)
      } footer: {
        Text("初回はSSHホスト鍵のフィンガープリントを確認します。パスワードと信頼済みホスト鍵はUserDefaultsではなくKeychainに保存します。")
      }

      if store.hasSavedHostKey {
        Section("信頼情報") {
          Button("保存済みホスト鍵を削除", role: .destructive) {
            confirmingHostKeyRemoval = true
          }
          .disabled(store.isBusy)
        }
      }

      messageSections
    }
  }

  private var browser: some View {
    List {
      transferSection

      Section {
        HStack(spacing: 12) {
          Button {
            store.openParent()
          } label: {
            Label("上のフォルダ", systemImage: "arrow.up")
          }
          .disabled(!store.canNavigateToParent || store.isBusy)
          Spacer()
          Button {
            store.refresh()
          } label: {
            Image(systemName: "arrow.clockwise")
          }
          .disabled(store.isBusy)
          .accessibilityLabel("再読み込み")
        }
        Text(store.currentPath)
          .font(.caption.monospaced())
          .foregroundStyle(.secondary)
          .textSelection(.enabled)
      }

      Section {
        if store.entries.isEmpty, !store.isBusy {
          Text("表示できるフォルダまたはMP4・MOV・M4Vがありません。")
            .foregroundStyle(.secondary)
        }
        ForEach(store.entries) { entry in
          entryRow(entry)
        }
      } header: {
        Text("フォルダ / 動画")
      } footer: {
        if isInputSelectionMode {
          VStack(alignment: .leading, spacing: 4) {
            Text("動画をタップするとFilesの「このiPad内 > mioh Remote > SFTP Downloads」へ保存します。右の情報ボタンで長さ・解像度・fps・コーデックを確認できます。長押しすると、ダウンロードまたはストリーミングで復元再生を選べます。")
            if let streamingUnavailableReason {
              Text(streamingUnavailableReason)
            }
          }
        }
      }

      if case .upload(let outputURL) = mode {
        Section("アップロード") {
          TextField("ファイル名", text: $uploadFileName)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .disabled(store.isBusy)
          Button {
            store.upload(
              localURL: outputURL,
              fileName: uploadFileName
            ) { path in
              onUploaded(path)
            }
          } label: {
            Label("このフォルダへ送信", systemImage: "arrow.up.doc")
          }
          .disabled(
            store.isBusy
              || uploadFileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          )
          Text("同名ファイルは上書きしません。送信完了後に一時名から確定名へ変更します。")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }

      messageSections
    }
  }

  @ViewBuilder
  private func entryRow(_ entry: MiohSFTPEntry) -> some View {
    if entry.isDirectory {
      Button {
        store.open(entry)
      } label: {
        HStack(spacing: 12) {
          Image(systemName: "folder.fill").foregroundStyle(.blue)
          Text(entry.name).foregroundStyle(.primary)
          Spacer()
          Image(systemName: "chevron.forward")
            .font(.caption)
            .foregroundStyle(.tertiary)
        }
      }
      .disabled(store.isBusy)
    } else {
      let metadata = store.cachedMetadata(for: entry)
      HStack(spacing: 10) {
        Button {
          downloadEntry(entry)
        } label: {
          HStack(spacing: 12) {
            Image(systemName: "film").foregroundStyle(.purple)
            VStack(alignment: .leading, spacing: 3) {
              Text(entry.name).foregroundStyle(.primary)
              let summary = fileSummary(entry)
              if !summary.isEmpty {
                Text(summary)
                  .font(.caption)
                  .foregroundStyle(.secondary)
              }
              if let metadata {
                Text(metadata.inlineSummary)
                  .font(.caption.monospacedDigit())
                  .foregroundStyle(Color.accentColor)
              }
            }
            Spacer()
            if case .selectInput = mode {
              Image(systemName: "arrow.down.circle")
                .foregroundStyle(Color.accentColor)
            }
          }
          .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, alignment: .leading)
        .disabled(store.isBusy || !isInputSelectionMode)

        Button {
          store.inspectMetadata(entry)
        } label: {
          if store.isInspectingMetadata(entry) {
            ProgressView()
          } else {
            Image(systemName: metadata == nil ? "info.circle" : "info.circle.fill")
          }
        }
        .buttonStyle(.borderless)
        .disabled(store.isBusy)
        .accessibilityLabel("動画情報を確認")
      }
      .contextMenu {
        Button {
          store.inspectMetadata(entry)
        } label: {
          Label("動画情報を確認", systemImage: "info.circle")
        }
        if case .selectInput = mode {
          Button {
            downloadEntry(entry)
          } label: {
            Label("ダウンロード", systemImage: "arrow.down.circle")
          }
          Button {
            onStreamingRequested(entry)
          } label: {
            Label(
              "ストリーミングで復元再生",
              systemImage: "play.rectangle.on.rectangle"
            )
          }
          .disabled(streamingUnavailableReason != nil)
        }
      }
      .accessibilityAction(named: "動画情報を確認") {
        store.inspectMetadata(entry)
      }
      .accessibilityAction(named: "ダウンロード") {
        downloadEntry(entry)
      }
      .accessibilityAction(named: "ストリーミングで復元再生") {
        guard isInputSelectionMode, streamingUnavailableReason == nil else { return }
        onStreamingRequested(entry)
      }
    }
  }

  private func downloadEntry(_ entry: MiohSFTPEntry) {
    guard isInputSelectionMode, !store.isBusy else { return }
    store.download(entry) { url in
      onDownloaded(url)
      dismiss()
    }
  }

  private func fileSummary(_ entry: MiohSFTPEntry) -> String {
    var parts: [String] = []
    if let byteCount = entry.byteCount {
      parts.append(
        ByteCountFormatter.string(fromByteCount: byteCount, countStyle: .file)
      )
    }
    if let modifiedAt = entry.modifiedAt {
      parts.append(
        modifiedAt.formatted(date: .abbreviated, time: .shortened)
      )
    }
    return parts.joined(separator: " • ")
  }

  @ViewBuilder
  private var transferSection: some View {
    if store.activity.isTransfer {
      Section("転送") {
        Text(store.activity.label)
        if store.activity.reportsByteProgress {
          ProgressView(value: store.progress)
        } else {
          ProgressView()
        }
        if store.totalBytes > 0 {
          Text(
            "\(ByteCountFormatter.string(fromByteCount: store.transferredBytes, countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: store.totalBytes, countStyle: .file))"
          )
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
        }
        if store.activity.reportsByteProgress,
          store.transferredBytes < store.totalBytes
        {
          if store.transferBytesPerSecond > 0 {
            LabeledContent(
              "速度",
              value: transferRate(store.transferBytesPerSecond)
            )
            .monospacedDigit()
            if let remaining = store.estimatedRemainingSeconds {
              LabeledContent("残り時間", value: remainingTime(remaining))
                .monospacedDigit()
            }
          } else {
            LabeledContent("速度", value: "計測中…")
            LabeledContent("残り時間", value: "計測中…")
          }
        }
        if store.resumedBytes > 0 {
          Label(
            "\(ByteCountFormatter.string(fromByteCount: store.resumedBytes, countStyle: .file))から再開",
            systemImage: "arrow.clockwise"
          )
          .font(.caption)
          .foregroundStyle(.secondary)
        }
        Button("転送を中止", role: .destructive) {
          store.cancelTransfer()
        }
      }
    } else if store.isBusy {
      Section {
        HStack {
          ProgressView()
          Text(store.activity.label)
        }
      }
    }
  }

  @ViewBuilder
  private var messageSections: some View {
    if let successMessage = store.successMessage {
      Section {
        Label(successMessage, systemImage: "checkmark.circle.fill")
          .foregroundStyle(.green)
      }
    }
    if let errorMessage = store.errorMessage {
      Section {
        Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
      }
    }
  }

  private var isInputSelectionMode: Bool {
    if case .selectInput = mode { return true }
    return false
  }

  private func transferRate(_ bytesPerSecond: Double) -> String {
    let bitsPerSecond = max(0, bytesPerSecond) * 8
    let bitRate: String
    if bitsPerSecond >= 1_000_000_000 {
      bitRate = String(format: "%.2f Gbps", bitsPerSecond / 1_000_000_000)
    } else if bitsPerSecond >= 1_000_000 {
      bitRate = String(format: "%.1f Mbps", bitsPerSecond / 1_000_000)
    } else if bitsPerSecond >= 1_000 {
      bitRate = String(format: "%.0f Kbps", bitsPerSecond / 1_000)
    } else {
      bitRate = String(format: "%.0f bps", bitsPerSecond)
    }
    let byteRate = ByteCountFormatter.string(
      fromByteCount: Int64(min(1_000_000_000_000, max(0, bytesPerSecond)).rounded()),
      countStyle: .decimal
    )
    return "\(bitRate)（\(byteRate)/s）"
  }

  private func remainingTime(_ seconds: TimeInterval) -> String {
    let total = max(0, Int(seconds.rounded(.up)))
    let days = total / 86_400
    let hours = (total % 86_400) / 3_600
    let minutes = (total % 3_600) / 60
    let remainder = total % 60
    if days > 0 {
      return String(format: "%d日 %02d:%02d:%02d", days, hours, minutes, remainder)
    }
    if hours > 0 {
      return String(format: "%02d:%02d:%02d", hours, minutes, remainder)
    }
    return String(format: "%02d:%02d", minutes, remainder)
  }
}

private struct IPadSFTPMovieMetadataView: View {
  @Environment(\.dismiss) private var dismiss
  let metadata: IPadSFTPMovieMetadata

  var body: some View {
    NavigationStack {
      List {
        Section("SFTP上の動画") {
          Text(metadata.name)
            .textSelection(.enabled)
          LabeledContent("容量", value: ByteCountFormatter.string(
            fromByteCount: metadata.byteCount,
            countStyle: .file
          ))
          if let modifiedAt = metadata.modifiedAt {
            LabeledContent(
              "更新日時",
              value: modifiedAt.formatted(date: .long, time: .standard)
            )
          }
          LabeledContent("コンテナ", value: metadata.container.isEmpty ? "不明" : metadata.container)
        }

        Section("動画情報") {
          LabeledContent("長さ", value: IPadSFTPMovieMetadata.durationText(metadata.duration))
          LabeledContent("解像度", value: resolutionText)
          LabeledContent("fps", value: frameRateText)
          LabeledContent("コーデック", value: metadata.codec)
          LabeledContent("音声", value: metadata.hasAudio ? "あり" : "なし")
          LabeledContent("平均データレート", value: metadata.averageBitRateText)
        }

        Section {
          Text("動画全体を保存せず、メタデータの解析に必要な範囲だけをSFTPから読み込みます。ファイルの構造によっては確認に時間がかかることがあります。")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }
      .navigationTitle("動画情報")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("閉じる") { dismiss() }
        }
      }
    }
  }

  private var resolutionText: String {
    guard let width = metadata.width, let height = metadata.height else {
      return "不明"
    }
    return "\(width)×\(height)"
  }

  private var frameRateText: String {
    guard let framesPerSecond = metadata.framesPerSecond else { return "不明" }
    return String(format: "%.2f fps", framesPerSecond)
  }
}
