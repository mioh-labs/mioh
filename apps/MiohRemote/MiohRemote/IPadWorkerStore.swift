import Combine
import Foundation
import MiohRemoteKit
import UIKit

enum IPadWorkerPreparationState: Equatable {
  case idle
  case validating
  case ready(restorations: [String], detectors: [String], maximumFrames: Int)
  case failed(String)
}

@MainActor
final class IPadWorkerStore: ObservableObject {
  @Published var displayName: String {
    didSet { defaults.set(displayName, forKey: Keys.displayName) }
  }
  @Published var sharedRootIdentifier: String {
    didSet { defaults.set(sharedRootIdentifier, forKey: Keys.sharedRootIdentifier) }
  }
  @Published private(set) var sharedRootURL: URL?
  @Published private(set) var preparation: IPadWorkerPreparationState = .idle
  @Published private(set) var serviceState: MiohClusterWorkerState = .stopped
  @Published private(set) var attempts: [MiohClusterAttemptRecord] = []
  @Published private(set) var errorMessage: String?
  @Published private(set) var lifecycleMessage: String?

  let nodeID: UUID

  var isRunning: Bool {
    switch serviceState {
    case .starting, .ready, .waiting: true
    case .stopped, .failed: false
    }
  }

  var startBlocker: String? {
    guard UIDevice.current.userInterfaceIdiom == .pad else {
      return "Worker実行にはiPadが必要です。iPhoneではMacの操作と映像視聴だけを利用できます。"
    }
    guard #available(iOS 27.0, *) else {
      return "Core AI WorkerにはiPadOS 27以降が必要です。"
    }
    guard case .ready = preparation, prepared != nil else {
      switch preparation {
      case .validating: return "同梱モデルを検証中です。"
      case .failed(let reason): return reason
      default: return "同梱Core AIモデルの検証が必要です。"
      }
    }
    return nil
  }

  var stateLabel: String {
    switch serviceState {
    case .stopped: "停止中"
    case .starting: "開始中"
    case .ready(let port): "受付中（port \(port)）"
    case .waiting(let reason): "待機中: \(reason)"
    case .failed(let reason): "失敗: \(reason)"
    }
  }

  private enum Keys {
    static let nodeID = "mioh.cluster.ipad.node-id.v1"
    static let displayName = "mioh.cluster.ipad.display-name.v1"
    static let sharedRootIdentifier = "mioh.cluster.ipad.shared-root-id.v1"
    static let sharedRootBookmark = "mioh.cluster.ipad.shared-root-bookmark.v1"
  }

  private let defaults = UserDefaults.standard
  private let service = MiohClusterWorkerService()
  private var prepared: MiohIPadPreparedWorker?
  private var cancellables: Set<AnyCancellable> = []
  private var hasSecurityScopedRootAccess = false

  var preparedWorker: MiohIPadPreparedWorker? { prepared }

  init() {
    if let stored = defaults.string(forKey: Keys.nodeID),
      let parsed = UUID(uuidString: stored)
    {
      nodeID = parsed
    } else {
      let created = UUID()
      nodeID = created
      defaults.set(created.uuidString.lowercased(), forKey: Keys.nodeID)
    }
    displayName = defaults.string(forKey: Keys.displayName)
      ?? "\(UIDevice.current.name) Worker"
    sharedRootIdentifier = defaults.string(forKey: Keys.sharedRootIdentifier) ?? ""
    restoreSharedRootBookmark()
    service.$state
      .receive(on: DispatchQueue.main)
      .sink { [weak self] state in
        guard let self else { return }
        self.serviceState = state
        if case .failed(let reason) = state {
          self.errorMessage = reason
          self.releaseSharedRootAccess()
          UIApplication.shared.isIdleTimerDisabled = false
        }
      }
      .store(in: &cancellables)
    service.ledger.$attempts
      .receive(on: DispatchQueue.main)
      .sink { [weak self] in self?.attempts = $0 }
      .store(in: &cancellables)
  }

  func prepareModels() async {
    guard !isRunning else { return }
    preparation = .validating
    prepared = nil
    errorMessage = nil
    do {
      let value = try await MiohIPadWorkerEngine.prepare()
      prepared = value
      preparation = .ready(
        restorations: value.restorationModelIdentifiers,
        detectors: value.detectorModelIdentifiers,
        maximumFrames: value.maximumRestorationClipLength
      )
    } catch {
      let reason = error.localizedDescription
      preparation = .failed(reason)
      errorMessage = reason
    }
  }

  func selectSharedRoot(_ url: URL) {
    guard !isRunning else {
      errorMessage = "Workerを停止してから共有フォルダを変更してください。"
      return
    }
    let gainedAccess = url.startAccessingSecurityScopedResource()
    defer { if gainedAccess { url.stopAccessingSecurityScopedResource() } }
    do {
      var isDirectory: ObjCBool = false
      guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
        isDirectory.boolValue
      else { throw MiohClusterWorkerError.missingSharedRoot }
      let bookmark = try url.bookmarkData(
        options: .minimalBookmark,
        includingResourceValuesForKeys: nil,
        relativeTo: nil
      )
      defaults.set(bookmark, forKey: Keys.sharedRootBookmark)
      sharedRootURL = url
      errorMessage = nil
    } catch {
      errorMessage = "共有フォルダを保存できません: \(error.localizedDescription)"
    }
  }

  func clearSharedRoot() {
    guard !isRunning else { return }
    sharedRootURL = nil
    defaults.removeObject(forKey: Keys.sharedRootBookmark)
  }

  func start() {
    guard !isRunning, startBlocker == nil, let prepared else { return }
    errorMessage = nil
    lifecycleMessage = nil
    let rootID = sharedRootIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
    let fallbackRoot = rootID.isEmpty ? nil : sharedRootURL
    if let fallbackRoot {
      let gainedAccess = fallbackRoot.startAccessingSecurityScopedResource()
      guard gainedAccess
        || FileManager.default.isReadableFile(atPath: fallbackRoot.path)
      else {
        errorMessage = "共有フォルダのセキュリティアクセスを開始できません。Filesで選び直してください。"
        return
      }
      hasSecurityScopedRootAccess = gainedAccess
    }
    do {
      let name = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
      let capabilities = prepared.makeCapabilities(
        nodeID: nodeID,
        displayName: name.isEmpty ? "mioh iPad Worker" : name,
        sharedRootIdentifier: fallbackRoot == nil ? nil : rootID
      )
      try service.start(
        sharedRoot: fallbackRoot,
        capabilities: capabilities,
        launcher: prepared.launcher
      )
      UIApplication.shared.isIdleTimerDisabled = true
    } catch {
      releaseSharedRootAccess()
      errorMessage = error.localizedDescription
    }
  }

  func stop(reason: String? = nil) async {
    await service.stopAndWait()
    releaseSharedRootAccess()
    UIApplication.shared.isIdleTimerDisabled = false
    lifecycleMessage = reason
  }

  private func restoreSharedRootBookmark() {
    guard let bookmark = defaults.data(forKey: Keys.sharedRootBookmark) else { return }
    do {
      var stale = false
      let url = try URL(
        resolvingBookmarkData: bookmark,
        options: .withoutUI,
        relativeTo: nil,
        bookmarkDataIsStale: &stale
      )
      sharedRootURL = url
      if stale {
        let gainedAccess = url.startAccessingSecurityScopedResource()
        defer { if gainedAccess { url.stopAccessingSecurityScopedResource() } }
        let refreshed = try url.bookmarkData(
          options: .minimalBookmark,
          includingResourceValuesForKeys: nil,
          relativeTo: nil
        )
        defaults.set(refreshed, forKey: Keys.sharedRootBookmark)
      }
    } catch {
      defaults.removeObject(forKey: Keys.sharedRootBookmark)
      sharedRootURL = nil
      errorMessage = "共有フォルダを復元できません。Filesで選び直してください。"
    }
  }

  private func releaseSharedRootAccess() {
    if hasSecurityScopedRootAccess, let root = sharedRootURL {
      root.stopAccessingSecurityScopedResource()
    }
    hasSecurityScopedRootAccess = false
  }
}
