import AppKit
import AVFoundation
import Combine
import CryptoKit
import Foundation

enum MiohClusterRoleSelection: String, CaseIterable, Identifiable {
  case off
  case coordinator
  case worker

  var id: String { rawValue }

  var label: String {
    switch self {
    case .off: "無効"
    case .coordinator: "Coordinator（配布・結合）"
    case .worker: "Worker（復元）"
    }
  }
}

enum MiohClusterControllerError: LocalizedError {
  case invalidSharedRoot
  case inputOutsideSharedRoot
  case outputOutsideSharedRoot
  case sourceMetadataUnavailable
  case noAuthorizedWorkers
  case incompatibleWorker(String)
  case connectionFailed(String)
  case jobRejected(String)
  case jobFailed(String)
  case missingArtifact(String)
  case cancelled
  case processFailed(String)

  var errorDescription: String? {
    switch self {
    case .invalidSharedRoot:
      return "共有ルートを選択し、全Macで同じ共有ルートIDを設定してください"
    case .inputOutsideSharedRoot:
      return "入力動画は共有ルート以下に置いてください"
    case .outputOutsideSharedRoot:
      return "出力先は共有ルート以下に置いてください"
    case .sourceMetadataUnavailable:
      return "動画の長さまたはフレームレートを取得できません"
    case .noAuthorizedWorkers:
      return "接続可能なWorkerを1台以上選択してください"
    case .incompatibleWorker(let name):
      return "Workerのモデルまたは共有ルートが一致しません: \(name)"
    case .connectionFailed(let name):
      return "Workerへ接続できませんでした: \(name)"
    case .jobRejected(let reason):
      return "Workerがジョブを拒否しました: \(reason)"
    case .jobFailed(let reason):
      return "Workerジョブが失敗しました: \(reason)"
    case .missingArtifact(let path):
      return "Worker成果物が共有フォルダにありません: \(path)"
    case .cancelled:
      return "クラスタ書き出しを停止しました"
    case .processFailed(let detail):
      return "映像片または音声の結合に失敗しました: \(detail)"
    }
  }
}

private struct MiohClusterPlannedJob: Sendable {
  let index: Int
  /// Number of presentation-timestamp entries in the half-open core range.
  /// The coordinator checks this against the worker metric before accepting
  /// the shard, so an AVAssetReader boundary quirk cannot silently drop or
  /// duplicate a VFR/NTSC frame.
  let expectedCoreFrameCount: Int
  let request: RemoteClusterJobRequest
  let localOutputURL: URL
}

/// A lane asks for another shard only after finishing its current shard.
/// Faster Macs can therefore do more work without assigning a fixed backlog
/// to a slower Mac or iPad. Actor isolation keeps every shard single-owner.
private actor MiohClusterJobPool {
  private let jobs: [MiohClusterPlannedJob]
  private var nextIndex = 0

  init(jobs: [MiohClusterPlannedJob]) {
    self.jobs = jobs
  }

  func takeNext() -> MiohClusterPlannedJob? {
    guard nextIndex < jobs.count else { return nil }
    defer { nextIndex += 1 }
    return jobs[nextIndex]
  }
}

private struct MiohClusterCompletedShard: Sendable {
  let index: Int
  let outputRelativePath: RemoteClusterRelativePath
  let outputURL: URL
  let metrics: RemoteClusterJobMetrics
}

private struct MiohClusterMediaIndex: Sendable {
  let presentationTimestamps: [Int64]
  let endNanoseconds: Int64
  let frameRate: Double
}

private struct MiohClusterRemoteWorker: Sendable {
  let node: RemoteClusterDiscoveredNode
  let capabilities: RemoteClusterCapabilities
  let transferMode: RemoteClusterTransferMode
}

private enum MiohClusterExecutionLane {
  case local
  case remote(
    RemoteClusterDiscoveredNode,
    RemoteClusterCapabilities,
    RemoteClusterTransferMode
  )

  var displayName: String {
    switch self {
    case .local:
      return Host.current().localizedName ?? "このMac"
    case .remote(let node, _, _):
      return node.metadata.displayName
    }
  }
}

private struct MiohClusterActiveAttempt {
  let lane: MiohClusterExecutionLane
  let attemptID: UUID
  let leaseID: UUID
}

@MainActor
final class MiohClusterController: ObservableObject {
  @Published var role: MiohClusterRoleSelection = .off
  @Published var sharedRootPath: String
  @Published var sharedRootIdentifier: String
  @Published var shardMinutes: Int
  @Published var useForExport = false
  @Published var useCoordinatorAsWorker: Bool {
    didSet {
      UserDefaults.standard.set(
        useCoordinatorAsWorker,
        forKey: Self.coordinatorWorkerDefaultsKey
      )
    }
  }
  @Published private(set) var serviceActive = false
  @Published private(set) var status = "無効"
  @Published private(set) var progress = 0.0
  @Published private(set) var isRunning = false
  @Published private(set) var discoveredNodes: [RemoteClusterDiscoveredNode] = []
  @Published private(set) var verifiedNodeIDs: Set<UUID> = []
  @Published var selectedNodeIDs: Set<UUID> = []
  @Published private(set) var workerAttempts: [RemoteClusterAttemptRecord] = []
  @Published private(set) var log = ""

  /// Worker jobs run independently of the Coordinator export flag. Global
  /// runner/default mutations must therefore also treat an accepted or
  /// running inbound job as active work.
  var hasActiveWorkerJobs: Bool {
    workerAttempts.contains { !$0.state.isTerminal }
  }

  private static let nodeIDDefaultsKey = "mioh.cluster.node-id.v1"
  private static let rootPathDefaultsKey = "mioh.cluster.shared-root-path.v1"
  private static let rootIDDefaultsKey = "mioh.cluster.shared-root-id.v1"
  private static let shardMinutesDefaultsKey = "mioh.cluster.shard-minutes.v1"
  private static let coordinatorWorkerDefaultsKey = "mioh.cluster.coordinator-worker.v1"
  private static let httpRangeInputExtensions: Set<String> = ["mp4", "mov", "m4v"]

  let localNodeID: UUID
  private let service: RemoteClusterService
  private let client: RemoteClusterClient
  private let transferServer = RemoteClusterHTTPTransferServer()
  private let localWorkerLedger = RemoteClusterWorkerJobLedger()
  private weak var runner: RestorationRunner?
  private var observations: Set<AnyCancellable> = []
  private var operationTask: Task<Void, Never>?
  private var activeAttempts: [UUID: MiohClusterActiveAttempt] = [:]
  private var verificationTasks: [UUID: Task<Void, Never>] = [:]
  private var failedAutomaticVerificationNodeIDs: Set<UUID> = []
  private var completedJobCount = 0
  private var totalJobCount = 0

  init() {
    let defaults = UserDefaults.standard
    if let stored = defaults.string(forKey: Self.nodeIDDefaultsKey),
      let nodeID = UUID(uuidString: stored)
    {
      localNodeID = nodeID
    } else {
      let nodeID = UUID()
      localNodeID = nodeID
      defaults.set(nodeID.uuidString.lowercased(), forKey: Self.nodeIDDefaultsKey)
    }
    sharedRootPath = defaults.string(forKey: Self.rootPathDefaultsKey) ?? ""
    sharedRootIdentifier = defaults.string(forKey: Self.rootIDDefaultsKey) ?? "mioh-shared"
    let storedShardMinutes = defaults.integer(forKey: Self.shardMinutesDefaultsKey)
    shardMinutes = storedShardMinutes == 0 ? 5 : min(30, max(1, storedShardMinutes))
    useCoordinatorAsWorker = defaults.object(forKey: Self.coordinatorWorkerDefaultsKey) == nil
      ? true
      : defaults.bool(forKey: Self.coordinatorWorkerDefaultsKey)

    let service = RemoteClusterService(localNodeID: localNodeID)
    self.service = service
    client = RemoteClusterClient(localNodeID: localNodeID)

    service.$discoveredNodes
      .receive(on: RunLoop.main)
      .sink { [weak self] nodes in
        guard let self else { return }
        self.discoveredNodes = nodes
        let liveIDs = Set(nodes.map(\.id))
        self.verifiedNodeIDs.formIntersection(liveIDs)
        self.selectedNodeIDs.formIntersection(liveIDs)
        self.failedAutomaticVerificationNodeIDs.formIntersection(liveIDs)
        let departedIDs = Set(self.verificationTasks.keys).subtracting(liveIDs)
        for nodeID in departedIDs {
          self.verificationTasks.removeValue(forKey: nodeID)?.cancel()
        }
        // Every discovered trusted-LAN Worker is verified for protocol and
        // capability compatibility without a pairing-code step.
        for node in nodes where !self.verifiedNodeIDs.contains(node.id)
          && !self.failedAutomaticVerificationNodeIDs.contains(node.id)
        {
          self.beginVerification(node, automatically: true)
        }
      }
      .store(in: &observations)
    service.$discoveryState
      .receive(on: RunLoop.main)
      .sink { [weak self] state in self?.consume(discoveryState: state) }
      .store(in: &observations)
    service.$workerState
      .receive(on: RunLoop.main)
      .sink { [weak self] state in self?.consume(workerState: state) }
      .store(in: &observations)
    service.jobLedger.$attemptRecords
      .receive(on: RunLoop.main)
      .sink { [weak self] records in self?.workerAttempts = records }
      .store(in: &observations)
  }

  deinit {
    operationTask?.cancel()
    verificationTasks.values.forEach { $0.cancel() }
    transferServer.stop()
  }

  func attach(runner: RestorationRunner) {
    self.runner = runner
  }

  func chooseSharedRoot() {
    let panel = NSOpenPanel()
    panel.title = "クラスタ共有ルートを選択"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    sharedRootPath = url.path
    persistSettings()
  }

  func updateSettings() {
    shardMinutes = min(30, max(1, shardMinutes))
    sharedRootIdentifier = sharedRootIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
    persistSettings()
  }

  func activate(using runner: RestorationRunner) {
    deactivate(preserveRole: true)
    self.runner = runner
    do {
      guard role != .off else {
        status = "無効"
        return
      }
      updateSettings()
      switch role {
      case .off:
        break
      case .coordinator:
        service.startCoordinatorDiscovery()
        serviceActive = true
        status = "Workerを探索中"
      case .worker:
        let root = try? validatedSharedRoot()
        let identityMaps = try runner.clusterCanonicalAssetIdentityMaps()
        let capabilities = RemoteClusterCapabilities(
          nodeID: localNodeID,
          displayName: Host.current().localizedName ?? "mioh worker",
          role: .worker,
          sharedRootIdentifier: root == nil ? "" : sharedRootIdentifier,
          architecture: Self.machineArchitecture(),
          operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
          maximumConcurrentJobs: 1,
          restorationModelIdentifiers: runner.restorationModels.filter {
            $0.contains("coreai") && $0 != "カスタム"
          },
          detectorModelIdentifiers: runner.detectionModels.filter { $0 != "カスタム" },
          maximumRestorationClipLength: 180,
          supportsROIEnhancer: true,
          supportsRestorationEffects: true,
          supportedInputExtensions: nil,
          restorationAssetSHA256ByIdentifier: identityMaps.restoration,
          detectorAssetSHA256ByIdentifier: identityMaps.detector,
          transferMode: .coordinatorHTTPV1,
          supportedTransferModes: root == nil
            ? [.coordinatorHTTPV1]
            : [.coordinatorHTTPV1, .sharedRootV1]
        )
        try service.startWorker(
          sharedRoot: root,
          capabilities: capabilities,
          launcher: runner.remoteClusterJobLauncher()
        )
        serviceActive = true
        status = "Workerを開始中"
      }
    } catch {
      service.stop()
      serviceActive = false
      status = "クラスタ起動失敗"
      appendLog(error.localizedDescription + "\n")
    }
  }

  func deactivate(preserveRole: Bool = false) {
    stopExport()
    localWorkerLedger.cancelAllActive()
    service.stop()
    transferServer.stop()
    serviceActive = false
    verifiedNodeIDs.removeAll()
    selectedNodeIDs.removeAll()
    if !preserveRole { role = .off }
    status = "無効"
  }

  func forget(_ node: RemoteClusterDiscoveredNode) {
    verificationTasks.removeValue(forKey: node.id)?.cancel()
    failedAutomaticVerificationNodeIDs.remove(node.id)
    verifiedNodeIDs.remove(node.id)
    selectedNodeIDs.remove(node.id)
  }

  func toggleSelection(_ nodeID: UUID, selected: Bool) {
    guard verifiedNodeIDs.contains(nodeID) else { return }
    if selected { selectedNodeIDs.insert(nodeID) } else { selectedNodeIDs.remove(nodeID) }
  }

  func startExport(using runner: RestorationRunner) {
    guard !isRunning else { return }
    self.runner = runner
    isRunning = true
    progress = 0
    status = "クラスタ書き出しを準備中"
    log = ""
    operationTask = Task { [weak self, weak runner] in
      guard let self, let runner else { return }
      do {
        try await self.performExport(using: runner)
        guard !Task.isCancelled else { throw MiohClusterControllerError.cancelled }
        self.progress = 1
        self.status = "完了"
        self.appendLog("クラスタ書き出しが完了しました。\n")
      } catch is CancellationError {
        self.status = "停止"
        self.appendLog("クラスタ書き出しを停止しました。\n")
      } catch {
        self.status = "エラー"
        self.appendLog(error.localizedDescription + "\n")
      }
      self.activeAttempts.removeAll()
      self.isRunning = false
      self.operationTask = nil
    }
  }

  func stopExport() {
    guard isRunning else { return }
    operationTask?.cancel()
    operationTask = nil
    let attempts = Array(activeAttempts.values)
    activeAttempts.removeAll()
    for active in attempts {
      switch active.lane {
      case .local:
        _ = localWorkerLedger.cancel(
          attemptID: active.attemptID,
          leaseID: active.leaseID
        )
      case .remote(let node, _, _):
        Task { [client] in
          _ = try? await client.call(
            .cancel(
              RemoteClusterCancelRequest(
                attemptID: active.attemptID,
                leaseID: active.leaseID
              )
            ),
            node: node
          )
        }
      }
    }
    status = "停止中"
  }

  func canStartExport(runner: RestorationRunner) -> Bool {
    role == .coordinator && serviceActive && useForExport && runner.canStart
      && !isRunning && (useCoordinatorAsWorker || !selectedNodeIDs.isEmpty)
  }

  private func performExport(using runner: RestorationRunner) async throws {
    guard role == .coordinator, serviceActive else {
      throw MiohClusterControllerError.noAuthorizedWorkers
    }
    guard let inputURL = runner.inputURL,
      let outputURL = runner.clusterResolvedOutputFile()
    else {
      throw MiohClusterControllerError.sourceMetadataUnavailable
    }
    let configuredSharedRoot = try? validatedSharedRoot()
    let sharedInputRelative: RemoteClusterRelativePath?
    if let configuredSharedRoot {
      sharedInputRelative = try relativePath(for: inputURL, beneath: configuredSharedRoot)
    } else {
      sharedInputRelative = nil
    }
    let logicalExtension = inputURL.pathExtension.lowercased()
    let virtualInputRelative = try RemoteClusterRelativePath(
      validating: "source/input.\(logicalExtension.isEmpty ? "bin" : logicalExtension)"
    )
    let inputRelative = sharedInputRelative ?? virtualInputRelative

    let selectedWorkers = discoveredNodes.filter {
      selectedNodeIDs.contains($0.id) && verifiedNodeIDs.contains($0.id)
    }
    guard useCoordinatorAsWorker || !selectedWorkers.isEmpty else {
      throw MiohClusterControllerError.noAuthorizedWorkers
    }
    let options = try await runner.clusterRestorationOptions()
    var remoteWorkers: [MiohClusterRemoteWorker] = []
    for worker in selectedWorkers {
      let response = try await client.call(.capabilities, node: worker)
      guard response.ok, let capabilities = response.capabilities,
        capabilities.restorationModelIdentifiers.contains(options.restorationModelIdentifier),
        capabilities.detectorModelIdentifiers.contains(options.detectorModelIdentifier),
        workerSupports(
          capabilities,
          options: options,
          inputExtension: inputURL.pathExtension
        )
      else {
        throw MiohClusterControllerError.incompatibleWorker(worker.metadata.displayName)
      }
      let modes = capabilities.effectiveTransferModes
      let extensionValue = inputURL.pathExtension.lowercased()
      let transferMode: RemoteClusterTransferMode?
      if modes.contains(.coordinatorHTTPV1),
        Self.httpRangeInputExtensions.contains(extensionValue)
      {
        transferMode = .coordinatorHTTPV1
      } else if modes.contains(.sharedRootV1),
        capabilities.sharedRootIdentifier == sharedRootIdentifier,
        sharedInputRelative != nil
      {
        transferMode = .sharedRootV1
      } else {
        transferMode = nil
      }
      guard let transferMode else {
        throw MiohClusterControllerError.incompatibleWorker(worker.metadata.displayName)
      }
      remoteWorkers.append(
        MiohClusterRemoteWorker(
          node: worker,
          capabilities: capabilities,
          transferMode: transferMode
        )
      )
    }
    let requiresSharedRoot = remoteWorkers.contains {
      $0.transferMode == .sharedRootV1
    }
    let root = try coordinatorWorkingRoot(requiresSharedRoot: requiresSharedRoot)
    if remoteWorkers.contains(where: { $0.transferMode == .coordinatorHTTPV1 }) {
      try await transferServer.start()
    }
    defer { transferServer.stop() }
    var lanes: [MiohClusterExecutionLane] = []
    if useCoordinatorAsWorker {
      // The local lane bypasses only the TCP/auth boundary because it is in
      // process. It still enters the same ledger with the same validated job
      // contract, source hash, path resolution, lease, staging and atomic
      // publication rules as every remote Worker.
      await localWorkerLedger.cancelAllActiveAndWait()
      lanes.append(.local)
    }
    lanes.append(contentsOf: remoteWorkers.map {
      MiohClusterExecutionLane.remote($0.node, $0.capabilities, $0.transferMode)
    })

    status = "フレーム境界を索引中"
    let media = try await Self.mediaIndex(inputURL)
    let inputByteCount: Int64
    let inputSHA256: String
    status = "入力のSHA-256を確認中"
    if remoteWorkers.contains(where: { $0.transferMode == .coordinatorHTTPV1 }) {
      // This single open descriptor is the same object later used by every
      // HTTP Range request, closing the hash-to-serve TOCTOU window without
      // re-hashing a large source once per shard.
      let identity = try await transferServer.pinSource(inputURL)
      inputByteCount = identity.byteCount
      inputSHA256 = identity.sha256
    } else {
      let attributes = try FileManager.default.attributesOfItem(atPath: inputURL.path)
      guard let number = attributes[.size] as? NSNumber, number.int64Value > 0 else {
        throw MiohClusterControllerError.sourceMetadataUnavailable
      }
      inputByteCount = number.int64Value
      inputSHA256 = try await Self.sha256(inputURL)
    }
    try Task.checkCancellation()
    if remoteWorkers.contains(where: { $0.transferMode == .coordinatorHTTPV1 }) {
      let aggregateLimit = try Self.maximumAggregateUploadedBytes(
        sourceByteCount: inputByteCount,
        bitrateMultiplier: options.bitrateMultiplier,
        stagingRoot: root
      )
      try transferServer.configureMaximumAggregateOutputBytes(aggregateLimit)
      appendLog(
        "HTTP成果物の総受信上限: \(ByteCountFormatter.string(fromByteCount: aggregateLimit, countStyle: .file))\n"
      )
    }

    let sessionID = UUID()
    let sessionPath = try RemoteClusterRelativePath(
      validating: ".mioh-cluster/\(sessionID.uuidString.lowercased())"
    )
    let sessionURL = try sessionPath.resolve(beneath: root)
    try FileManager.default.createDirectory(
      at: sessionURL,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    let jobs = try makeJobs(
      sessionID: sessionID,
      inputRelativePath: inputRelative,
      inputByteCount: inputByteCount,
      inputSHA256: inputSHA256,
      media: media,
      options: options,
      sessionURL: sessionURL
    )
    guard !jobs.isEmpty else { throw MiohClusterControllerError.sourceMetadataUnavailable }
    completedJobCount = 0
    totalJobCount = jobs.count
    status = "\(lanes.count)レーンへ\(jobs.count)ジョブを配布中"
    appendLog(
      "クラスタ開始: \(lanes.count)レーン（ローカル "
        + "\(useCoordinatorAsWorker ? 1 : 0) / リモート \(remoteWorkers.count)） / "
        + "\(jobs.count)ジョブ / "
        + "HTTP転送 \(remoteWorkers.filter { $0.transferMode == .coordinatorHTTPV1 }.count) / "
        + "共有ルートfallback \(remoteWorkers.filter { $0.transferMode == .sharedRootV1 }.count)\n"
    )

    let jobPool = MiohClusterJobPool(jobs: jobs)
    var shards: [MiohClusterCompletedShard] = []
    do {
      try await withThrowingTaskGroup(of: [MiohClusterCompletedShard].self) { group in
        for lane in lanes {
          group.addTask { @MainActor [weak self] in
            guard let self else { throw CancellationError() }
            var laneShards: [MiohClusterCompletedShard] = []
            while let job = await jobPool.takeNext() {
              try Task.checkCancellation()
              switch lane {
              case .local:
                laneShards.append(contentsOf: try await self.runLocal(
                  queue: [job],
                  sharedRoot: root,
                  inputURL: inputURL,
                  launcher: runner.remoteClusterJobLauncher()
                ))
              case .remote(let worker, let capabilities, let transferMode):
                laneShards.append(contentsOf: try await self.runRemote(
                  queue: [job],
                  on: worker,
                  capabilities: capabilities,
                  transferMode: transferMode,
                  inputURL: inputURL,
                  sourceDurationNanoseconds: media.endNanoseconds
                ))
              }
            }
            return laneShards
          }
        }
        // Observe workers in completion order. A failed lane therefore
        // cancels its peers immediately instead of waiting for an earlier,
        // still-running lane in array order.
        while let completed = try await group.next() {
          shards.append(contentsOf: completed)
        }
      }
    } catch {
      await cancelActiveAttempts()
      throw error
    }
    try Task.checkCancellation()
    let ordered = shards.sorted { $0.index < $1.index }
    guard ordered.count == jobs.count else {
      throw MiohClusterControllerError.jobFailed("成果物数が一致しません")
    }
    let shardURLs = try ordered.map { shard -> URL in
      guard FileManager.default.fileExists(atPath: shard.outputURL.path) else {
        throw MiohClusterControllerError.missingArtifact(shard.outputRelativePath.rawValue)
      }
      return shard.outputURL
    }
    status = "映像片と音声を結合中"
    try await merge(
      shards: shardURLs,
      source: inputURL,
      output: outputURL,
      fastStart: options.mp4FastStart,
      runner: runner
    )
    if !runner.keepTemp {
      try? FileManager.default.removeItem(at: sessionURL)
    }
  }

  private func workerSupports(
    _ capabilities: RemoteClusterCapabilities,
    options: RemoteClusterRestorationOptions,
    inputExtension: String
  ) -> Bool {
    if let maximum = capabilities.maximumRestorationClipLength,
      options.restorationClipLength > maximum
    {
      return false
    }
    let usesROIEnhancer = options.roiEnhancerModelIdentifier != nil
      && options.roiEnhancerStrength > 0
    if capabilities.supportsROIEnhancer == false, usesROIEnhancer { return false }

    let usesEffects = options.sharpenStrength != 0
      || options.detailBoost != 0
      || options.textureMix != 0
      || options.smoothStrength != 0
      || options.effectUpscale != 1
    if capabilities.supportsRestorationEffects == false, usesEffects { return false }

    if let supported = capabilities.supportedInputExtensions {
      let normalized = inputExtension.lowercased()
        .trimmingCharacters(in: CharacterSet(charactersIn: "."))
      if !supported.contains(normalized) { return false }
    }
    if let advertised = capabilities.restorationAssetSHA256ByIdentifier?[
      options.restorationModelIdentifier
    ], advertised.lowercased() != options.restorationAssetSHA256.lowercased() {
      return false
    }
    if let advertised = capabilities.detectorAssetSHA256ByIdentifier?[
      options.detectorModelIdentifier
    ], advertised.lowercased() != options.detectorAssetSHA256.lowercased() {
      return false
    }
    return true
  }

  private func runRemote(
    queue: [MiohClusterPlannedJob],
    on node: RemoteClusterDiscoveredNode,
    capabilities: RemoteClusterCapabilities,
    transferMode: RemoteClusterTransferMode,
    inputURL: URL,
    sourceDurationNanoseconds: Int64
  ) async throws -> [MiohClusterCompletedShard] {
    var completed: [MiohClusterCompletedShard] = []
    for job in queue {
      try Task.checkCancellation()
      // Lease time starts when this lane actually takes the job, not when the
      // full export plan was built. Long queues must not submit stale work.
      let leasedRequest = job.request.withLeaseExpiration(
        Date().addingTimeInterval(14 * 60)
      )
      let submittedRequest: RemoteClusterJobRequest
      if transferMode == .coordinatorHTTPV1 {
        let descriptor = try transferServer.register(
          request: leasedRequest,
          inputURL: inputURL,
          outputURL: job.localOutputURL,
          maximumOutputBytes: Self.maximumUploadedShardBytes(
            sourceByteCount: leasedRequest.inputByteCount,
            sourceDurationNanoseconds: sourceDurationNanoseconds,
            mediaRange: leasedRequest.mediaRange,
            bitrateMultiplier: leasedRequest.options.bitrateMultiplier
          )
        )
        submittedRequest = leasedRequest.withHTTPTransfer(descriptor)
      } else {
        submittedRequest = leasedRequest
      }
      defer {
        if transferMode == .coordinatorHTTPV1 {
          transferServer.unregister(attemptID: job.request.attemptID)
        }
      }
      // Register before the submit RPC. If the connection is cancelled after
      // the Worker accepted the request but before its response reaches us,
      // fail-fast cleanup still knows which lease-scoped attempt to cancel.
      activeAttempts[job.request.attemptID] = MiohClusterActiveAttempt(
        lane: .remote(node, capabilities, transferMode),
        attemptID: job.request.attemptID,
        leaseID: job.request.leaseID
      )
      let admission = try await client.call(.submit(submittedRequest), node: node)
      guard admission.ok,
        let admitted = admission.admission,
        admitted.disposition == .accepted || admitted.disposition == .duplicate
      else {
        throw MiohClusterControllerError.jobRejected(
          admission.errorCode ?? admission.admission?.reason ?? "unknown"
        )
      }
      appendLog("[\(node.metadata.displayName)] job \(job.index + 1) 開始\n")
      var nextRenewal = Date().addingTimeInterval(5 * 60)
      while true {
        try Task.checkCancellation()
        if Date() >= nextRenewal {
          let renewal = RemoteClusterRenewLeaseRequest(
            attemptID: job.request.attemptID,
            leaseID: job.request.leaseID,
            newExpiration: Date().addingTimeInterval(14 * 60)
          )
          let response = try await client.call(.renewLease(renewal), node: node)
          guard response.ok else {
            throw MiohClusterControllerError.jobFailed(
              response.errorCode ?? "lease_renewal_failed"
            )
          }
          if transferMode == .coordinatorHTTPV1,
            !transferServer.renew(
              attemptID: job.request.attemptID,
              leaseID: job.request.leaseID,
              until: renewal.newExpiration
            )
          {
            throw MiohClusterControllerError.jobFailed("transfer_lease_renewal_failed")
          }
          nextRenewal = Date().addingTimeInterval(5 * 60)
        }
        let response = try await client.call(
          .status(RemoteClusterStatusQuery(attemptID: job.request.attemptID)),
          node: node
        )
        guard response.ok, let attempt = response.attempt else {
          throw MiohClusterControllerError.jobFailed(
            response.errorCode ?? "status_unavailable"
          )
        }
        switch attempt.state {
        case .accepted, .running:
          try await Task.sleep(nanoseconds: 1_000_000_000)
        case .completed:
          guard let metrics = attempt.metrics else {
            throw MiohClusterControllerError.jobFailed("metrics_missing")
          }
          completed.append(
            try completedShard(
              for: job,
              metrics: metrics,
              laneName: node.metadata.displayName
            )
          )
          break
        case .failed:
          throw MiohClusterControllerError.jobFailed(
            attempt.failureCode ?? "launcher_failed"
          )
        case .cancelled, .expired:
          throw MiohClusterControllerError.jobFailed(attempt.state.rawValue)
        }
        if attempt.state == .completed { break }
      }
    }
    return completed
  }

  private func runLocal(
    queue: [MiohClusterPlannedJob],
    sharedRoot: URL,
    inputURL: URL,
    launcher: @escaping RemoteClusterJobLauncher
  ) async throws -> [MiohClusterCompletedShard] {
    let lane = MiohClusterExecutionLane.local
    var completed: [MiohClusterCompletedShard] = []
    for job in queue {
      try Task.checkCancellation()
      let leasedRequest = job.request.withLeaseExpiration(
        Date().addingTimeInterval(14 * 60)
      )
      let admission = localWorkerLedger.submit(
        leasedRequest,
        sharedRoot: sharedRoot,
        sharedRootIdentifier: sharedRootIdentifier,
        launcher: launcher,
        localInputURL: inputURL,
        localOutputURL: job.localOutputURL
      )
      guard admission.disposition == .accepted || admission.disposition == .duplicate else {
        throw MiohClusterControllerError.jobRejected(admission.reason ?? "local_job_rejected")
      }
      activeAttempts[job.request.attemptID] = MiohClusterActiveAttempt(
        lane: lane,
        attemptID: job.request.attemptID,
        leaseID: job.request.leaseID
      )
      appendLog("[\(lane.displayName)] job \(job.index + 1) 開始\n")
      var nextRenewal = Date().addingTimeInterval(5 * 60)
      while true {
        try Task.checkCancellation()
        guard let attempt = localWorkerLedger.record(attemptID: job.request.attemptID) else {
          throw MiohClusterControllerError.jobFailed("local_status_unavailable")
        }
        switch attempt.state {
        case .accepted, .running:
          if Date() >= nextRenewal {
            guard localWorkerLedger.renewLease(
              attemptID: job.request.attemptID,
              leaseID: job.request.leaseID,
              until: Date().addingTimeInterval(14 * 60)
            ) else {
              throw MiohClusterControllerError.jobFailed("local_lease_renewal_failed")
            }
            nextRenewal = Date().addingTimeInterval(5 * 60)
          }
          try await Task.sleep(nanoseconds: 1_000_000_000)
        case .completed:
          guard let metrics = attempt.metrics else {
            throw MiohClusterControllerError.jobFailed("metrics_missing")
          }
          completed.append(
            try completedShard(
              for: job,
              metrics: metrics,
              laneName: lane.displayName
            )
          )
          break
        case .failed:
          throw MiohClusterControllerError.jobFailed(
            attempt.failureCode ?? "local_launcher_failed"
          )
        case .cancelled, .expired:
          throw MiohClusterControllerError.jobFailed(attempt.state.rawValue)
        }
        if attempt.state == .completed { break }
      }
    }
    return completed
  }

  private func completedShard(
    for job: MiohClusterPlannedJob,
    metrics: RemoteClusterJobMetrics,
    laneName: String
  ) throws -> MiohClusterCompletedShard {
    guard metrics.processedFrames == job.expectedCoreFrameCount else {
      throw MiohClusterControllerError.jobFailed(
        "core_frame_count_mismatch: expected "
          + "\(job.expectedCoreFrameCount), got \(metrics.processedFrames)"
      )
    }
    let attributes = try FileManager.default.attributesOfItem(
      atPath: job.localOutputURL.path
    )
    guard let size = attributes[.size] as? NSNumber,
      size.int64Value > 0,
      size.int64Value == metrics.outputByteCount
    else {
      throw MiohClusterControllerError.jobFailed("output_byte_count_mismatch")
    }
    activeAttempts.removeValue(forKey: job.request.attemptID)
    completedJobCount += 1
    progress = Double(completedJobCount) / Double(max(1, totalJobCount)) * 0.95
    status = "クラスタ処理中 \(completedJobCount)/\(totalJobCount)"
    appendLog(
      "[\(laneName)] job \(job.index + 1) 完了 "
        + "\(metrics.processedFrames)フレーム\n"
    )
    return MiohClusterCompletedShard(
      index: job.index,
      outputRelativePath: job.request.outputRelativePath,
      outputURL: job.localOutputURL,
      metrics: metrics
    )
  }

  private func cancelActiveAttempts() async {
    let attempts = Array(activeAttempts.values)
    activeAttempts.removeAll()
    for active in attempts {
      guard case .local = active.lane else { continue }
      _ = localWorkerLedger.cancel(
        attemptID: active.attemptID,
        leaseID: active.leaseID
      )
    }
    // Unstructured cancellation RPCs do not inherit the already-cancelled
    // export task. Await all of them together with the local teardown barrier.
    var cancellationTasks: [Task<Void, Never>] = [
      Task { @MainActor [localWorkerLedger] in
        await localWorkerLedger.cancelAllActiveAndWait()
      }
    ]
    for active in attempts {
      guard case .remote(let node, _, _) = active.lane else { continue }
      cancellationTasks.append(Task { @MainActor [client] in
        _ = try? await client.call(
          .cancel(
            RemoteClusterCancelRequest(
              attemptID: active.attemptID,
              leaseID: active.leaseID
            )
          ),
          node: node
        )
      })
    }
    for task in cancellationTasks {
      await task.value
    }
  }

  private func makeJobs(
    sessionID: UUID,
    inputRelativePath: RemoteClusterRelativePath,
    inputByteCount: Int64,
    inputSHA256: String,
    media: MiohClusterMediaIndex,
    options: RemoteClusterRestorationOptions,
    sessionURL: URL
  ) throws -> [MiohClusterPlannedJob] {
    let temporalBatchFrames = options.restorationClipLength
    let overlap = options.temporalOverlap
    let stride = temporalBatchFrames - overlap
    guard stride > 0, media.frameRate > 0 else {
      throw MiohClusterControllerError.sourceMetadataUnavailable
    }
    let totalFrames = media.presentationTimestamps.count
    guard totalFrames > 0 else {
      throw MiohClusterControllerError.sourceMetadataUnavailable
    }
    let requestedCoreFrames = max(
      stride,
      Int(Double(shardMinutes * 60) * media.frameRate)
    )
    let coreStrideCount = max(1, requestedCoreFrames / stride)
    let coreFramesPerJob = coreStrideCount * stride
    func timestamp(_ frame: Int) -> Int64 {
      if frame >= totalFrames { return media.endNanoseconds }
      return media.presentationTimestamps[max(0, frame)]
    }

    var jobs: [MiohClusterPlannedJob] = []
    var coreStart = 0
    while coreStart < totalFrames {
      let index = jobs.count
      let coreEnd = min(totalFrames, coreStart + coreFramesPerJob)
      // One complete preceding stride primes recurrent state. The right halo
      // supplies the last owned batch's overlap without emitting duplicates.
      let decodeStart = coreStart == 0 ? 0 : max(0, coreStart - stride)
      let decodeEnd = min(totalFrames, coreEnd + overlap)
      let outputPath = try RemoteClusterRelativePath(
        validating: ".mioh-cluster/\(sessionID.uuidString.lowercased())/"
          + String(format: "job-%06d.mp4", index)
      )
      let localOutputURL = sessionURL.appendingPathComponent(
        String(format: "job-%06d.mp4", index),
        isDirectory: false
      )
      let now = Date()
      let request = RemoteClusterJobRequest(
        jobID: UUID(),
        attemptID: UUID(),
        leaseID: UUID(),
        coordinatorNodeID: localNodeID,
        sharedRootIdentifier: sharedRootIdentifier,
        inputByteCount: inputByteCount,
        inputSHA256: inputSHA256,
        inputRelativePath: inputRelativePath,
        outputRelativePath: outputPath,
        mediaRange: RemoteClusterMediaRange(
          decodeStartNanoseconds: timestamp(decodeStart),
          decodeEndNanoseconds: timestamp(decodeEnd),
          coreStartNanoseconds: timestamp(coreStart),
          coreEndNanoseconds: timestamp(coreEnd),
          leadingOverlapFrames: coreStart - decodeStart,
          trailingOverlapFrames: decodeEnd - coreEnd
        ),
        options: options,
        createdAt: now,
        leaseExpiresAt: now.addingTimeInterval(14 * 60)
      )
      jobs.append(
        MiohClusterPlannedJob(
          index: index,
          expectedCoreFrameCount: coreEnd - coreStart,
          request: request,
          localOutputURL: localOutputURL
        )
      )
      coreStart = coreEnd
    }
    return jobs
  }

  private func merge(
    shards: [URL],
    source: URL,
    output: URL,
    fastStart: Bool,
    runner: RestorationRunner
  ) async throws {
    guard !shards.isEmpty else { throw MiohClusterControllerError.jobFailed("empty") }
    if FileManager.default.fileExists(atPath: output.path), !runner.overwrite {
      throw RunnerError.unsupportedFeature(
        "出力ファイルが存在します: \(output.path)\n"
          + "置き換える場合は「基本」タブの「既存結果を上書き」を有効にしてください"
      )
    }
    let resources = try runner.resourceDirectory()
    let ffmpeg = resources.appendingPathComponent("bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw RunnerError.missingResource("FFmpeg")
    }
    let temporaryRoot = runner.ffmpegTempDirectory.isEmpty
      ? FileManager.default.temporaryDirectory
      : URL(fileURLWithPath: runner.ffmpegTempDirectory, isDirectory: true)
    let working = temporaryRoot.appendingPathComponent(
      "mioh-cluster-merge-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: working, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: working) }
    let manifest = working.appendingPathComponent("shards.txt")
    let body = shards.map { url in
      "file '\(url.path.replacingOccurrences(of: "'", with: "'\\''"))'"
    }.joined(separator: "\n") + "\n"
    try Data(body.utf8).write(to: manifest, options: .atomic)
    try FileManager.default.createDirectory(
      at: output.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let outputExtension = output.pathExtension.isEmpty ? "mp4" : output.pathExtension
    let part = output.deletingPathExtension()
      .appendingPathExtension("part")
      .appendingPathExtension(outputExtension)
    try? FileManager.default.removeItem(at: part)
    var arguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      "-f", "concat", "-safe", "0", "-i", manifest.path,
      "-i", source.path,
      "-map", "0:v:0", "-map", "1:a:0?",
      "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
      // Workers emit video-only shards. Add the original audio exactly once.
      // Do not use -shortest: a source whose audio ends a fraction early must
      // not lose an otherwise valid final restored video frame.
      "-map_metadata", "1",
    ]
    if fastStart && outputExtension.lowercased() == "mp4" {
      arguments.append(contentsOf: ["-movflags", "+faststart"])
    }
    arguments.append(part.path)
    try await Self.runProcess(ffmpeg, arguments: arguments, workingDirectory: working)
    try? FileManager.default.removeItem(at: output)
    try FileManager.default.moveItem(at: part, to: output)
  }

  private func beginVerification(
    _ node: RemoteClusterDiscoveredNode,
    automatically: Bool
  ) {
    guard verificationTasks[node.id] == nil else { return }
    verificationTasks[node.id] = Task { [weak self] in
      guard let self else { return }
      await self.verify(node, automatically: automatically)
    }
  }

  private func verify(
    _ node: RemoteClusterDiscoveredNode,
    automatically: Bool
  ) async {
    defer { verificationTasks.removeValue(forKey: node.id) }
    do {
      let response = try await client.call(.capabilities, node: node)
      try Task.checkCancellation()
      guard response.ok, let capabilities = response.capabilities,
        capabilities.nodeID == node.id,
        (capabilities.effectiveTransferModes.contains(.coordinatorHTTPV1)
          || capabilities.sharedRootIdentifier == sharedRootIdentifier)
      else {
        throw MiohClusterControllerError.connectionFailed(node.metadata.displayName)
      }
      verifiedNodeIDs.insert(node.id)
      selectedNodeIDs.insert(node.id)
      failedAutomaticVerificationNodeIDs.remove(node.id)
      appendLog("Worker接続成功: \(node.metadata.displayName)\n")
    } catch is CancellationError {
      return
    } catch {
      verifiedNodeIDs.remove(node.id)
      selectedNodeIDs.remove(node.id)
      if automatically { failedAutomaticVerificationNodeIDs.insert(node.id) }
      appendLog("Worker接続失敗: \(node.metadata.displayName) — \(error.localizedDescription)\n")
    }
  }

  private func validatedSharedRoot() throws -> URL {
    let identifier = sharedRootIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !identifier.isEmpty, identifier.utf8.count <= 128, !sharedRootPath.isEmpty else {
      throw MiohClusterControllerError.invalidSharedRoot
    }
    let root = URL(fileURLWithPath: sharedRootPath, isDirectory: true)
      .standardizedFileURL.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else { throw MiohClusterControllerError.invalidSharedRoot }
    return root
  }

  /// The Coordinator needs a local staging root, not a network mount. A user-
  /// selected root is retained for `shared-root-v1` fallback; otherwise a
  /// private Application Support directory is sufficient for HTTP Workers.
  private func coordinatorWorkingRoot(requiresSharedRoot: Bool) throws -> URL {
    if requiresSharedRoot { return try validatedSharedRoot() }
    guard let applicationSupport = FileManager.default.urls(
      for: .applicationSupportDirectory,
      in: .userDomainMask
    ).first else { throw MiohClusterControllerError.invalidSharedRoot }
    let root = applicationSupport.appendingPathComponent(
      "mioh/cluster-coordinator-v1",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: root,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    return root.standardizedFileURL.resolvingSymlinksInPath()
  }

  private static func maximumUploadedShardBytes(
    sourceByteCount: Int64,
    sourceDurationNanoseconds: Int64,
    mediaRange: RemoteClusterMediaRange,
    bitrateMultiplier: Double
  ) -> Int64 {
    let oneMiB = 1_048_576.0
    let minimum = 256 * oneMiB
    let maximum = 128 * 1_073_741_824.0
    let rangeDuration = max(
      1,
      mediaRange.decodeEndNanoseconds - mediaRange.decodeStartNanoseconds
    )
    let durationFraction = min(
      1,
      Double(rangeDuration) / Double(max(1, sourceDurationNanoseconds))
    )
    let sourceShare = Double(max(1, sourceByteCount)) * durationFraction
    // Three times the configured bitrate plus container headroom tolerates
    // unusually complex 4K/8K scenes without granting every shard the whole
    // source-file budget.
    let estimated = sourceShare * max(1, bitrateMultiplier) * 3 + 128 * oneMiB
    return Int64(min(maximum, max(minimum, estimated.rounded(.up))))
  }

  private static func maximumAggregateUploadedBytes(
    sourceByteCount: Int64,
    bitrateMultiplier: Double,
    stagingRoot: URL
  ) throws -> Int64 {
    let desired = Double(max(1, sourceByteCount)) * max(3, bitrateMultiplier * 3)
      + 1_073_741_824.0
    let values = try stagingRoot.resourceValues(forKeys: [
      .volumeAvailableCapacityForImportantUsageKey
    ])
    let available = Double(max(1, values.volumeAvailableCapacityForImportantUsage ?? 0))
    let diskBound = available > 1 ? available * 0.75 : desired
    let bounded = min(Double(Int64.max), max(1, min(desired, diskBound)))
    return Int64(bounded.rounded(.down))
  }

  private func relativePath(
    for candidate: URL,
    beneath sharedRoot: URL
  ) throws -> RemoteClusterRelativePath? {
    let root = sharedRoot.standardizedFileURL.resolvingSymlinksInPath()
    let standardized = candidate.standardizedFileURL
    let rootComponents = root.pathComponents
    let candidateComponents = standardized.pathComponents
    guard candidateComponents.count > rootComponents.count,
      Array(candidateComponents.prefix(rootComponents.count)) == rootComponents
    else { return nil }
    let relative = candidateComponents.dropFirst(rootComponents.count).joined(separator: "/")
    let path = try RemoteClusterRelativePath(validating: relative)
    // Resolving here rejects an existing symlink component before work is sent.
    _ = try path.resolve(beneath: root)
    return path
  }

  private func persistSettings() {
    let defaults = UserDefaults.standard
    defaults.set(sharedRootPath, forKey: Self.rootPathDefaultsKey)
    defaults.set(sharedRootIdentifier, forKey: Self.rootIDDefaultsKey)
    defaults.set(shardMinutes, forKey: Self.shardMinutesDefaultsKey)
    defaults.set(useCoordinatorAsWorker, forKey: Self.coordinatorWorkerDefaultsKey)
  }

  private func appendLog(_ text: String) {
    log += text
    if log.count > 12_000 { log = String(log.suffix(8_000)) }
    runner?.appendExternalLog(text)
  }

  private func consume(discoveryState: RemoteClusterDiscoveryState) {
    guard role == .coordinator else { return }
    switch discoveryState {
    case .stopped: if serviceActive { status = "探索停止" }
    case .starting: status = "Workerを探索中"
    case .ready: status = "Worker探索中（\(discoveredNodes.count)台）"
    case .waiting(let reason): status = "探索待機: \(reason)"
    case .failed(let reason): status = "探索失敗: \(reason)"
    }
  }

  private func consume(workerState: RemoteClusterWorkerState) {
    guard role == .worker else { return }
    switch workerState {
    case .stopped: if serviceActive { status = "Worker停止" }
    case .starting: status = "Workerを開始中"
    case .ready(let port): status = "Worker待受中（port \(port)）"
    case .waiting(let reason): status = "Worker待機: \(reason)"
    case .failed(let reason): status = "Worker起動失敗: \(reason)"
    }
  }

  /// Build the job boundary table from real presentation timestamps instead
  /// of multiplying an average FPS. This keeps VFR/NTSC sources on exact
  /// frame boundaries and prevents adjacent workers from dropping or
  /// duplicating a frame at a shard edge. Samples remain compressed here.
  private static func mediaIndex(_ url: URL) async throws
    -> MiohClusterMediaIndex
  {
    try await Task.detached(priority: .utility) {
      let asset = AVURLAsset(url: url)
      guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        throw MiohClusterControllerError.sourceMetadataUnavailable
      }
      let reader = try AVAssetReader(asset: asset)
      let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
      output.alwaysCopiesSampleData = false
      guard reader.canAdd(output) else {
        throw MiohClusterControllerError.sourceMetadataUnavailable
      }
      reader.add(output)
      guard reader.startReading() else {
        throw reader.error ?? MiohClusterControllerError.sourceMetadataUnavailable
      }
      var timestamps: [Int64] = []
      timestamps.reserveCapacity(32_768)
      while let sample = output.copyNextSampleBuffer() {
        try Task.checkCancellation()
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        guard pts.isValid, !pts.isIndefinite else { continue }
        let seconds = CMTimeGetSeconds(pts)
        if seconds.isFinite, seconds >= 0 {
          timestamps.append(Int64((seconds * 1_000_000_000).rounded()))
        }
      }
      guard reader.status == .completed, !timestamps.isEmpty else {
        throw reader.error ?? MiohClusterControllerError.sourceMetadataUnavailable
      }
      timestamps.sort()
      var unique: [Int64] = []
      unique.reserveCapacity(timestamps.count)
      for value in timestamps where unique.last != value { unique.append(value) }
      guard !unique.isEmpty else {
        throw MiohClusterControllerError.sourceMetadataUnavailable
      }
      let assetDuration = try await asset.load(.duration)
      let assetEnd = Int64(
        max(0, CMTimeGetSeconds(assetDuration) * 1_000_000_000).rounded()
      )
      let fallbackDelta: Int64
      if unique.count > 1 {
        let positive = zip(unique.dropFirst(), unique)
          .map { $0.0 - $0.1 }
          .filter { $0 > 0 }
          .sorted()
        fallbackDelta = positive.isEmpty ? 33_333_333 : positive[positive.count / 2]
      } else {
        let nominal = Double(try await track.load(.nominalFrameRate))
        fallbackDelta = nominal > 0
          ? Int64((1_000_000_000 / nominal).rounded()) : 33_333_333
      }
      let end = max(unique.last! + fallbackDelta, assetEnd)
      let span = Double(max(1, end - unique[0])) / 1_000_000_000
      let rate = Double(unique.count) / span
      guard rate.isFinite, rate > 0 else {
        throw MiohClusterControllerError.sourceMetadataUnavailable
      }
      return MiohClusterMediaIndex(
        presentationTimestamps: unique,
        endNanoseconds: end,
        frameRate: rate
      )
    }.value
  }

  private static func sha256(_ url: URL) async throws -> String {
    try await Task.detached(priority: .utility) {
      let handle = try FileHandle(forReadingFrom: url)
      defer { try? handle.close() }
      var hasher = SHA256()
      while let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty {
        try Task.checkCancellation()
        hasher.update(data: chunk)
      }
      return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }.value
  }

  private static func runProcess(
    _ executable: URL,
    arguments: [String],
    workingDirectory: URL
  ) async throws {
    let process = Process()
    let errorPipe = Pipe()
    process.executableURL = executable
    process.arguments = arguments
    process.currentDirectoryURL = workingDirectory
    process.standardOutput = FileHandle.nullDevice
    process.standardError = errorPipe
    try process.run()
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        process.terminationHandler = { completed in
          let detail = String(
            data: errorPipe.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
          )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
          if completed.terminationStatus == 0 {
            continuation.resume()
          } else {
            continuation.resume(
              throwing: MiohClusterControllerError.processFailed(
                detail.isEmpty ? "exit \(completed.terminationStatus)" : detail
              )
            )
          }
        }
      }
    } onCancel: {
      process.terminate()
    }
  }

  private static func machineArchitecture() -> String {
#if arch(arm64)
    return "arm64"
#elseif arch(x86_64)
    return "x86_64"
#else
    return "unknown"
#endif
  }

  private static func hex(_ data: Data) -> String {
    data.map { String(format: "%02x", $0) }.joined()
  }
}
