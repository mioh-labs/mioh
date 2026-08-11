import Combine
import Darwin
import Foundation
import Network
import Security

// MARK: - Streaming integration contract

struct RemoteStreamSession: Sendable {
  let ticket: String
  let expiresAt: Date
}

enum RemoteStreamSessionIssue: Sendable {
  case issued(RemoteStreamSession)
  case unavailable
  case capacityReached
}

/// Narrow adapter between the LAN HTTP surface and the media packager. The
/// coordinator owns all paths and only returns already-authorized assets, so
/// an HTTP path can never be translated directly into an arbitrary file URL.
@MainActor
protocol RemoteStreamingCoordinating: AnyObject {
  func eventConsumer() -> RealtimeStreamingEventConsumer
  func setServerEnabled(_ enabled: Bool)
  func issueSession() -> RemoteStreamSessionIssue
  func revokeSession(ticket: String)
  func revokeAllSessions()
  func statusJSON() -> [String: Any]
  func playlistData(ticket: String) -> Data?
  func segmentURL(ticket: String, sequence: Int) -> URL?
}

private struct RemoteControlAsset {
  let id: String
  let url: URL
  let label: String
  let preservesExistingSelection: Bool
  let preservedSettingKeys: Set<String>
}

private enum RemoteControlAssetPurpose: String {
  case input
  case previewInput = "preview-input"
  case directory
  case model
}

private enum RemoteControlConfigurationError: Error {
  case invalidJSON
  case invalidAsset(String)
}

// MARK: - Public service

/// A deliberately small LAN-only HTTP control surface for mioh.
///
/// The listener accepts one HTTP/1.1 request per TCP connection.  The root
/// page is public so a new device can display the token entry screen; every
/// `/api/v1/*` route requires the Keychain-backed bearer token.
@MainActor
final class RemoteControlServer: ObservableObject {
  @Published private(set) var enabled = false
  @Published var port = 8888
  @Published private(set) var status = L("無効")
  @Published private(set) var token: String
  @Published private(set) var urls: [URL] = []

  /// Stable, non-secret identity advertised through Bonjour. Authentication
  /// still uses the independently stored bearer token; the identifier only
  /// lets an iPhone remember which discovered Mac it paired with.
  let serverID = RemoteControlServerIdentity.loadOrCreate()

  private weak var runner: RestorationRunner?
  private weak var player: RealtimePlayerController?
  private weak var cluster: MiohClusterController?
  private weak var streaming: (any RemoteStreamingCoordinating)?
  private var engine: RemoteHTTPServerEngine?
  private var engineID = UUID()
  private var keychainWarning: String?
  private var configurationRevision = 1
  private var configurationSnapshotData: Data?
  private var assetCatalog: [String: RemoteControlAsset] = [:]
  private static let maximumAssetCount = 65_536

  init() {
    let loaded = RemoteControlTokenStore.loadOrCreate()
    token = loaded.token
    keychainWarning = loaded.warning
  }

  /// Installs the app objects controlled by the API without making the HTTP
  /// layer an owner of either long-lived processing object.
  func attach(runner: RestorationRunner, player: RealtimePlayerController) {
    self.runner = runner
    self.player = player
  }

  func attachCluster(_ cluster: MiohClusterController) {
    self.cluster = cluster
  }

  /// Installs the independently-owned live-stream coordinator. Keeping this
  /// separate from `attach(runner:player:)` preserves source compatibility for
  /// existing windows and makes the media lifetime explicit.
  func attachStreaming(_ streaming: any RemoteStreamingCoordinating) {
    self.streaming = streaming
    streaming.setServerEnabled(enabled)
  }

  func setEnabled(_ newValue: Bool) {
    newValue ? start() : stop()
  }

  /// Restarts the listener after a port edit while preserving the token.
  func restart() {
    guard enabled else { return }
    stop()
    start()
  }

  func regenerateToken() {
    streaming?.revokeAllSessions()
    let generated = RemoteControlTokenStore.regenerate()
    token = generated.token
    keychainWarning = generated.warning
    if enabled {
      status =
        generated.warning == nil
        ? L("待受中")
        : L("待受中（トークンをKeychainへ保存できませんでした）")
    }
  }

  private func start() {
    guard !enabled else { return }
    guard (1...65_535).contains(port), let listenerPort = NWEndpoint.Port(rawValue: UInt16(port))
    else {
      status = L("ポート番号が正しくありません")
      return
    }

    let identifier = UUID()
    engineID = identifier
    let newEngine = RemoteHTTPServerEngine()
    engine = newEngine
    enabled = true
    urls = []
    status = L("開始中")

    do {
      try newEngine.start(
        port: listenerPort,
        service: RemoteHTTPBonjourService(
          name: Host.current().localizedName ?? "mioh",
          type: "_mioh._tcp",
          txtRecord: NWTXTRecord([
            "api": "1",
            "id": serverID.uuidString.lowercased(),
            "features": "control,hls",
          ])
        ),
        requestHandler: { [weak self] request, reply in
          Task { @MainActor [weak self] in
            guard let self, self.enabled, self.engineID == identifier else {
              reply(.json(status: 503, object: ["ok": false, "error": "server_unavailable"]))
              return
            }
            reply(self.route(request))
          }
        },
        stateHandler: { [weak self] state in
          Task { @MainActor [weak self] in
            guard let self, self.engineID == identifier else { return }
            switch state {
            case .ready(let boundPort):
              self.port = Int(boundPort.rawValue)
              self.urls = Self.availableURLs(port: Int(boundPort.rawValue))
              self.status =
                self.keychainWarning == nil
                ? L("待受中")
                : L("待受中（トークンをKeychainへ保存できませんでした）")
            case .failed(let message):
              self.streaming?.setServerEnabled(false)
              self.engine = nil
              self.enabled = false
              self.urls = []
              self.status = L("起動失敗") + ": " + message
            case .cancelled:
              if self.enabled {
                self.engine = nil
                self.enabled = false
                self.urls = []
                self.status = L("無効")
              }
            }
          }
        }
      )
      streaming?.setServerEnabled(true)
    } catch {
      streaming?.setServerEnabled(false)
      engine = nil
      enabled = false
      urls = []
      status = L("起動失敗") + ": " + error.localizedDescription
    }
  }

  private func stop() {
    engineID = UUID()
    streaming?.setServerEnabled(false)
    streaming?.revokeAllSessions()
    engine?.stop()
    engine = nil
    enabled = false
    urls = []
    status = L("無効")
  }

  private func route(_ request: RemoteHTTPRequest) -> RemoteHTTPResponse {
    let path = request.path
    if path == "/" {
      guard request.method == "GET" else { return .methodNotAllowed(["GET"]) }
      return .html(RemoteControlHTML.page)
    }

    if path.hasPrefix("/stream/v1/") {
      return routeMedia(request)
    }

    guard path.hasPrefix("/api/v1/") else { return .jsonError(404, "not_found") }
    guard validHostHeader(request) else { return .jsonError(421, "invalid_host") }
    guard authorized(request) else {
      return .jsonError(
        401,
        "unauthorized",
        headers: ["WWW-Authenticate": "Bearer realm=\"mioh\""]
      )
    }
    guard let runner, let player else {
      return .jsonError(503, "control_adapter_unavailable")
    }

    switch (request.method, path) {
    case ("GET", "/api/v1/status"):
      return statusResponse(runner: runner, player: player)

    case ("GET", "/api/v1/settings"):
      return settingsResponse(runner: runner, player: player)

    case ("PATCH", "/api/v1/settings"), ("POST", "/api/v1/settings"):
      guard !runner.isRunning, cluster?.isRunning != true,
        cluster?.hasActiveWorkerJobs != true
      else {
        return .jsonError(409, "settings_locked_while_running")
      }
      return applySettings(request, runner: runner, player: player)

    case ("POST", "/api/v1/assets/list"):
      return listAssets(request, runner: runner, player: player)

    case ("POST", "/api/v1/assets/create-directory"):
      guard !runner.isRunning, cluster?.isRunning != true,
        cluster?.hasActiveWorkerJobs != true
      else {
        return .jsonError(409, "settings_locked_while_running")
      }
      return createAssetDirectory(request)

    case ("POST", "/api/v1/playback/source"):
      guard !runner.isRunning, cluster?.isRunning != true else {
        return .jsonError(409, "settings_locked_while_running")
      }
      guard let object = requestJSONObject(request),
        let assetID = object["assetID"] as? String,
        let url = assetURL(assetID),
        assetIsSelectable(url, purpose: .previewInput)
      else { return .jsonError(400, "invalid_preview_asset") }
      player.selectPreviewInput(url, runner: runner)
      return .json(status: 200, object: ["ok": true, "revision": configurationRevision])

    case ("POST", "/api/v1/playback/original"):
      guard let object = requestJSONObject(request), let enabled = object["enabled"] as? Bool
      else { return .jsonError(400, "invalid_original_setting") }
      player.showOriginal = enabled
      return .json(status: 200, object: ["ok": true])

    case ("POST", "/api/v1/defaults/save"):
      guard !runner.isRunning, cluster?.isRunning != true,
        cluster?.hasActiveWorkerJobs != true
      else {
        return .jsonError(409, "settings_locked_while_running")
      }
      runner.saveCurrentDefaults()
      return .json(status: 200, object: ["ok": true, "status": runner.defaultsStatus])

    case ("POST", "/api/v1/defaults/load"):
      guard !runner.isRunning, cluster?.isRunning != true,
        cluster?.hasActiveWorkerJobs != true
      else {
        return .jsonError(409, "settings_locked_while_running")
      }
      player.stop()
      runner.loadSavedDefaults()
      synchronizeConfigurationRevision(runner)
      return .json(status: 200, object: ["ok": true, "status": runner.defaultsStatus])

    case ("POST", "/api/v1/defaults/reset"):
      guard !runner.isRunning, cluster?.isRunning != true,
        cluster?.hasActiveWorkerJobs != true
      else {
        return .jsonError(409, "settings_locked_while_running")
      }
      player.stop()
      runner.resetDefaultsToFactory()
      synchronizeConfigurationRevision(runner)
      return .json(status: 200, object: ["ok": true, "status": runner.defaultsStatus])

    case ("PATCH", "/api/v1/cluster/settings"), ("POST", "/api/v1/cluster/settings"):
      guard let cluster else { return .jsonError(503, "cluster_unavailable") }
      guard !runner.isRunning, !cluster.isRunning, !cluster.serviceActive else {
        return .jsonError(409, "cluster_settings_locked")
      }
      return applyClusterSettings(request, cluster: cluster)

    case ("POST", "/api/v1/cluster/start"):
      guard let cluster else { return .jsonError(503, "cluster_unavailable") }
      guard !runner.isRunning, !cluster.isRunning, !cluster.serviceActive else {
        return .jsonError(409, "cluster_already_active")
      }
      cluster.activate(using: runner)
      return cluster.serviceActive ? .accepted() : .jsonError(409, "cluster_did_not_start")

    case ("POST", "/api/v1/cluster/stop"):
      cluster?.deactivate(preserveRole: true)
      return .accepted()

    case ("POST", "/api/v1/cluster/node/select"):
      guard let cluster else { return .jsonError(503, "cluster_unavailable") }
      guard !cluster.isRunning, !cluster.hasActiveWorkerJobs else {
        return .jsonError(409, "cluster_nodes_locked")
      }
      guard let object = requestJSONObject(request),
        let rawID = object["id"] as? String, let id = UUID(uuidString: rawID),
        let selected = object["selected"] as? Bool,
        cluster.discoveredNodes.contains(where: { $0.id == id })
      else { return .jsonError(400, "invalid_cluster_node") }
      cluster.toggleSelection(id, selected: selected)
      return .accepted()

    case ("POST", "/api/v1/cluster/node/forget"):
      guard let cluster else { return .jsonError(503, "cluster_unavailable") }
      guard !cluster.isRunning, !cluster.hasActiveWorkerJobs else {
        return .jsonError(409, "cluster_nodes_locked")
      }
      guard let object = requestJSONObject(request),
        let rawID = object["id"] as? String, let id = UUID(uuidString: rawID),
        let node = cluster.discoveredNodes.first(where: { $0.id == id })
      else { return .jsonError(400, "invalid_cluster_node") }
      cluster.forget(node)
      return .accepted()

    case ("GET", "/api/v1/stream/status"):
      guard let streaming else { return .jsonError(503, "stream_unavailable") }
      return .json(status: 200, object: streaming.statusJSON())

    case ("POST", "/api/v1/stream/session"):
      guard let streaming else { return .jsonError(503, "stream_unavailable") }
      switch streaming.issueSession() {
      case .issued(let session):
        // Re-publish the player's currently retained queue so a viewer can
        // start immediately even when playback is paused or the preview
        // worker has already reached its buffer limit.
        player.setStreamingEventConsumer(streaming.eventConsumer())
        let formatter = ISO8601DateFormatter()
        return .json(
          status: 201,
          object: [
            "ok": true,
            "playlist": "/stream/v1/\(session.ticket)/index.m3u8",
            "expiresAt": formatter.string(from: session.expiresAt),
          ]
        )
      case .unavailable:
        return .jsonError(409, "stream_not_ready")
      case .capacityReached:
        return .jsonError(429, "stream_session_limit")
      }

    case ("POST", "/api/v1/stream/stop"):
      guard let streaming else { return .jsonError(503, "stream_unavailable") }
      guard let object = requestJSONObject(request),
        let ticket = object["ticket"] as? String,
        Self.isMediaTicket(ticket)
      else {
        return .jsonError(400, "invalid_stream_ticket")
      }
      streaming.revokeSession(ticket: ticket)
      return .accepted()

    case ("POST", "/api/v1/playback/play"):
      guard player.remotePlay(runner: runner) else {
        return .jsonError(409, "no_playback_input")
      }
      return .accepted()

    case ("POST", "/api/v1/playback/pause"):
      guard player.remotePause() else {
        return .jsonError(409, "playback_not_active")
      }
      return .accepted()

    case ("POST", "/api/v1/playback/toggle"):
      guard player.previewInputURL != nil else {
        return .jsonError(409, "no_playback_input")
      }
      guard player.remoteToggle(runner: runner) else {
        return .jsonError(409, "playback_not_active")
      }
      return .accepted()

    case ("POST", "/api/v1/playback/stop"):
      player.stop()
      return .accepted()

    case ("POST", "/api/v1/playback/seek"):
      guard player.previewInputURL != nil else {
        return .jsonError(409, "no_playback_input")
      }
      guard let object = requestJSONObject(request),
        let seconds = number(object["seconds"]), seconds.isFinite, seconds >= 0
      else {
        return .jsonError(400, "invalid_seconds")
      }
      player.seek(to: seconds)
      return .accepted()

    case ("POST", "/api/v1/playback/volume"):
      guard let object = requestJSONObject(request),
        let value = number(object["volume"]), value.isFinite, (0...1).contains(value)
      else {
        return .jsonError(400, "invalid_volume")
      }
      player.setVolume(value)
      return .accepted()

    case ("POST", "/api/v1/playback/mute"):
      guard let object = requestJSONObject(request), let muted = object["muted"] as? Bool
      else {
        return .jsonError(400, "invalid_muted")
      }
      player.setMuted(muted)
      return .accepted()

    case ("POST", "/api/v1/export/start"):
      guard !runner.isRunning, cluster?.isRunning != true else {
        return .jsonError(409, "export_already_running")
      }
      guard runner.canStart else { return .jsonError(409, "export_not_configured") }
      if let cluster,
        cluster.useForExport,
        cluster.role == .coordinator
      {
        guard cluster.canStartExport(runner: runner) else {
          return .jsonError(409, "cluster_not_ready")
        }
        cluster.startExport(using: runner)
        return cluster.isRunning ? .accepted() : .jsonError(409, "export_did_not_start")
      }
      runner.start()
      return runner.isRunning ? .accepted() : .jsonError(409, "export_did_not_start")

    case ("POST", "/api/v1/export/stop"):
      if cluster?.isRunning == true { cluster?.stopExport() }
      if runner.isRunning { runner.stop() }
      return .accepted()

    default:
      let allowed = ["/api/v1/status", "/api/v1/settings", "/api/v1/stream/status"].contains(path)
        ? ["GET"] : ["POST"]
      return .methodNotAllowed(allowed)
    }
  }

  private func routeMedia(_ request: RemoteHTTPRequest) -> RemoteHTTPResponse {
    guard request.method == "GET" || request.method == "HEAD" else {
      return .methodNotAllowed(["GET", "HEAD"])
    }
    guard let streaming else { return .jsonError(503, "stream_unavailable") }
    let pieces = request.path.split(separator: "/", omittingEmptySubsequences: true)
    guard pieces.count >= 4, pieces[0] == "stream", pieces[1] == "v1" else {
      return .jsonError(404, "not_found")
    }
    let ticket = String(pieces[2])
    guard Self.isMediaTicket(ticket) else { return .jsonError(404, "not_found") }

    if pieces.count == 4, pieces[3] == "index.m3u8" {
      guard let data = streaming.playlistData(ticket: ticket) else {
        return .jsonError(404, "stream_not_found")
      }
      return .data(
        status: 200,
        contentType: "application/vnd.apple.mpegurl",
        data: request.method == "HEAD" ? Data() : data,
        representedLength: data.count,
        headers: ["Cache-Control": "no-store"]
      )
    }

    guard pieces.count == 5, pieces[3] == "segment",
      pieces[4].hasSuffix(".ts")
    else {
      return .jsonError(404, "not_found")
    }
    let rawSequence = pieces[4].dropLast(3)
    guard !rawSequence.isEmpty, rawSequence.allSatisfy(\.isNumber),
      let sequence = Int(rawSequence), sequence >= 0,
      let url = streaming.segmentURL(ticket: ticket, sequence: sequence)
    else {
      return .jsonError(404, "stream_not_found")
    }
    return Self.fileResponse(
      url: url,
      request: request,
      contentType: "video/mp2t"
    )
  }

  private static func isMediaTicket(_ value: String) -> Bool {
    let bytes = value.utf8
    return bytes.count == 64
      && bytes.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
  }

  private static func fileResponse(
    url: URL,
    request: RemoteHTTPRequest,
    contentType: String
  ) -> RemoteHTTPResponse {
    guard url.isFileURL,
      let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
      values.isRegularFile == true,
      let fileSize = values.fileSize,
      fileSize >= 0
    else {
      return .jsonError(404, "stream_not_found")
    }

    switch RemoteHTTPByteRange.parse(request.headers["range"], fileSize: UInt64(fileSize)) {
    case .invalid:
      return .data(
        status: 416,
        contentType: contentType,
        data: Data(),
        representedLength: 0,
        headers: [
          "Accept-Ranges": "bytes",
          "Cache-Control": "private, max-age=30",
          "Content-Range": "bytes */\(fileSize)",
        ]
      )
    case .full:
      return .file(
        status: 200,
        url: url,
        offset: 0,
        length: UInt64(fileSize),
        sendBody: request.method == "GET",
        contentType: contentType,
        headers: [
          "Accept-Ranges": "bytes",
          "Cache-Control": "private, max-age=30",
        ]
      )
    case .range(let lower, let upper):
      let length = upper - lower + 1
      return .file(
        status: 206,
        url: url,
        offset: lower,
        length: length,
        sendBody: request.method == "GET",
        contentType: contentType,
        headers: [
          "Accept-Ranges": "bytes",
          "Cache-Control": "private, max-age=30",
          "Content-Range": "bytes \(lower)-\(upper)/\(fileSize)",
        ]
      )
    }
  }

  private func authorized(_ request: RemoteHTTPRequest) -> Bool {
    guard let authorization = request.headers["authorization"] else { return false }
    let pieces = authorization.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
    guard pieces.count == 2, pieces[0].lowercased() == "bearer" else { return false }
    guard
      let supplied = RemoteControlAccessToken.canonicalize(String(pieces[1])),
      let expected = RemoteControlAccessToken.canonicalize(token)
    else { return false }
    return Self.constantTimeEquals(supplied, expected)
  }

  // MARK: - Full remote configuration

  private func settingsResponse(
    runner: RestorationRunner,
    player: RealtimePlayerController
  ) -> RemoteHTTPResponse {
    do {
      synchronizeConfigurationRevision(runner)
      var settings = try jsonObject(runner.currentDefaultsSnapshot())
      var assetLabels: [String: Any] = [:]
      assetizeSettings(&settings, labels: &assetLabels)

      var previewSource: [String: Any] = ["assetID": NSNull(), "label": NSNull()]
      if let url = player.previewInputURL,
        let asset = registerAsset(
          url,
          preservesExistingSelection: true,
          preservedSettingKey: "__previewSource"
        )
      {
        previewSource = ["assetID": asset.id, "label": asset.label]
      }
      return .json(
        status: 200,
        object: [
          "ok": true,
          "schemaVersion": 1,
          "revision": configurationRevision,
          "settings": settings,
          "assetLabels": assetLabels,
          "preview": [
            "source": previewSource,
            "showOriginal": player.showOriginal,
          ],
          "options": configurationOptions(runner),
          "cluster": clusterJSON(),
        ]
      )
    } catch {
      return .jsonError(500, "settings_serialization_failed")
    }
  }

  private func applySettings(
    _ request: RemoteHTTPRequest,
    runner: RestorationRunner,
    player: RealtimePlayerController
  ) -> RemoteHTTPResponse {
    guard let object = requestJSONObject(request),
      let suppliedRevision = integer(object["revision"]),
      var settings = object["settings"] as? [String: Any]
    else { return .jsonError(400, "invalid_settings_payload") }
    synchronizeConfigurationRevision(runner)
    guard suppliedRevision == configurationRevision else {
      return .jsonError(409, "stale_settings_revision")
    }

    do {
      try resolveSettingAssets(&settings)
    } catch RemoteControlConfigurationError.invalidAsset(let field) {
      return .jsonError(400, "invalid_or_unknown_asset_\(field)")
    } catch {
      return .jsonError(400, "invalid_or_unknown_asset")
    }
    do {
      let data = try JSONSerialization.data(withJSONObject: settings, options: [.sortedKeys])
      let snapshot = try JSONDecoder().decode(MiohUserDefaultsSnapshot.self, from: data)
      guard validate(snapshot, runner: runner) else {
        return .jsonError(400, "invalid_settings")
      }

      let previousState = player.state
      let previewWasActive = [
        RealtimePlayerState.loading, .buffering, .playing, .paused, .seeking,
      ].contains(previousState)
      let previousPosition = player.position
      runner.apply(defaults: snapshot)
      player.setBufferLimit(runner.previewBufferLimit)
      if let original = object["showOriginal"] as? Bool {
        player.showOriginal = original
      }
      synchronizeConfigurationRevision(runner)
      if previewWasActive, player.previewInputURL != nil {
        player.start(
          runner: runner,
          at: previousPosition,
          autoPlay: previousState != .paused,
          preserveCurrentSource: true
        )
      }
      return settingsResponse(runner: runner, player: player)
    } catch {
      return .jsonError(400, "invalid_settings_payload")
    }
  }

  private func configurationOptions(_ runner: RestorationRunner) -> [String: Any] {
    let engines = runner.supportsPythonEngine ? ["native", "python"] : ["native"]
    func availabilityJSON(_ value: MiohModelAvailability) -> [String: Any] {
      var result: [String: Any] = ["available": value.available]
      if let reason = value.reason { result["reason"] = reason }
      return result
    }
    func modelOption(
      _ id: String,
      label: String? = nil,
      availabilityForEngine: (String) -> MiohModelAvailability
    ) -> [String: Any] {
      let current = availabilityForEngine(runner.restorationEngine)
      var option: [String: Any] = [
        "id": id,
        "label": label ?? id,
        "available": current.available,
        "availabilityByEngine": Dictionary(uniqueKeysWithValues: engines.map {
          ($0, availabilityJSON(availabilityForEngine($0)))
        }),
      ]
      if let reason = current.reason { option["reason"] = reason }
      return option
    }
    let restorationModels = runner.restorationModels.map {
      let model = $0
      return modelOption(model) {
        runner.restorationModelAvailability(model, engine: $0)
      }
    }
    let detectionModels = runner.detectionModels.map {
      let model = $0
      return modelOption(model) {
        runner.detectionModelAvailability(model, engine: $0)
      }
    }
    let previewDetectionModels = runner.previewDetectionModels.map {
      let model = $0
      return modelOption(model) {
        runner.detectionModelAvailability(model, engine: $0)
      }
    }
    let roiModels = Dictionary(uniqueKeysWithValues: runner.enhancerModels.map { enhancer in
      (
        enhancer,
        runner.roiEnhancerModelOptions(for: enhancer).map { model in
          var option = modelOption(
            model.name,
            label: model.label,
            availabilityForEngine: { engine in
              runner.roiEnhancerModelAvailability(model.name, engine: engine)
            })
          option["scale"] = model.scale
          return option
        }
      )
    })
    return [
      "restorationEngines": engines,
      "executors": ["process", "thread"],
      "devices": runner.supportsPythonEngine ? ["mps", "cpu", "cuda:0"] : ["mps"],
      "encodingModes": ["auto", "preset", "custom"],
      "encodingPresets": runner.encodingPresets,
      "restorationModels": restorationModels,
      "detectionModels": detectionModels,
      "previewRestorationModels": restorationModels,
      "previewDetectionModels": previewDetectionModels,
      "roiEnhancers": runner.enhancerModels,
      "roiEnhancerModels": roiModels,
      "frameRates": runner.frameRateOptions.map {
        [
          "numerator": $0.numerator,
          "denominator": $0.denominator,
          "label": $0.label,
        ] as [String: Any]
      },
      "projectionModes": ["通常", "VR180", "360"],
      "videoLayouts": ["Mono", "SBS 左右", "上下"],
      "eyes": ["左目", "右目"],
      "hlsQualities": PreviewHLSQuality.allCases.map {
        ["value": $0.rawValue, "label": $0.label]
      },
      "mergeEncoders": ["copy", "h264", "hevc"],
    ]
  }

  private func clusterJSON() -> [String: Any] {
    guard let cluster else { return ["available": false] }
    var sharedRoot: [String: Any] = ["assetID": NSNull(), "label": NSNull()]
    if !cluster.sharedRootPath.isEmpty,
      let asset = registerAsset(
        URL(fileURLWithPath: cluster.sharedRootPath, isDirectory: true),
        preservesExistingSelection: true,
        preservedSettingKey: "__clusterRoot"
      )
    {
      sharedRoot = ["assetID": asset.id, "label": asset.label]
    }
    return [
      "available": true,
      "role": cluster.role.rawValue,
      "sharedRoot": sharedRoot,
      "sharedRootIdentifier": cluster.sharedRootIdentifier,
      "shardMinutes": cluster.shardMinutes,
      "useForExport": cluster.useForExport,
      "useCoordinatorAsWorker": cluster.useCoordinatorAsWorker,
      "serviceActive": cluster.serviceActive,
      "status": cluster.status,
      "nodes": cluster.discoveredNodes.map { node in
        [
          "id": node.id.uuidString.lowercased(),
          "name": node.metadata.displayName,
          "verified": cluster.verifiedNodeIDs.contains(node.id),
          "selected": cluster.selectedNodeIDs.contains(node.id),
        ] as [String: Any]
      },
    ]
  }

  private func applyClusterSettings(
    _ request: RemoteHTTPRequest,
    cluster: MiohClusterController
  ) -> RemoteHTTPResponse {
    guard let object = requestJSONObject(request),
      let roleText = object["role"] as? String,
      let role = MiohClusterRoleSelection(rawValue: roleText),
      let identifier = object["sharedRootIdentifier"] as? String,
      !identifier.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      identifier.utf8.count <= 128,
      let shardMinutes = integer(object["shardMinutes"]), (1...30).contains(shardMinutes),
      let useForExport = object["useForExport"] as? Bool,
      let useCoordinatorAsWorker = object["useCoordinatorAsWorker"] as? Bool
    else { return .jsonError(400, "invalid_cluster_settings") }

    var rootPath = ""
    if let assetID = object["sharedRootAssetID"] as? String, !assetID.isEmpty {
      guard let asset = assetRecord(assetID), let url = assetURL(assetID) else {
        return .jsonError(400, "invalid_cluster_root")
      }
      if FileManager.default.fileExists(atPath: url.path) {
        guard assetIsSelectable(url, purpose: .directory) else {
          return .jsonError(400, "invalid_cluster_root")
        }
      } else if !asset.preservedSettingKeys.contains("__clusterRoot") {
        return .jsonError(400, "invalid_cluster_root")
      }
      rootPath = url.path
    }
    cluster.role = role
    cluster.sharedRootPath = rootPath
    cluster.sharedRootIdentifier = identifier
    cluster.shardMinutes = shardMinutes
    cluster.useForExport = useForExport
    cluster.useCoordinatorAsWorker = useCoordinatorAsWorker
    cluster.updateSettings()
    return .json(status: 200, object: ["ok": true, "cluster": clusterJSON()])
  }

  private func jsonObject<T: Encodable>(_ value: T) throws -> [String: Any] {
    let data = try JSONEncoder().encode(value)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw RemoteControlConfigurationError.invalidJSON
    }
    return object
  }

  /// Keeps optimistic concurrency tied to the native settings themselves,
  /// not only to mutations that happened to arrive through this HTTP server.
  /// A change made in the Mac UI therefore invalidates an older Web form
  /// instead of being silently overwritten by its next 74-field PATCH.
  private func synchronizeConfigurationRevision(_ runner: RestorationRunner) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let data = try? encoder.encode(runner.currentDefaultsSnapshot()) else { return }
    if let previous = configurationSnapshotData, previous != data {
      configurationRevision += 1
    }
    configurationSnapshotData = data
  }

  private func assetizeSettings(
    _ settings: inout [String: Any],
    labels: inout [String: Any]
  ) {
    for key in Self.settingPathKeys {
      guard let path = settings[key] as? String, path.hasPrefix("/") else { continue }
      guard let asset = registerAsset(
        URL(fileURLWithPath: path),
        preservesExistingSelection: true,
        preservedSettingKey: key
      ) else {
        settings[key] = NSNull()
        labels[key] = "選択をMac側で確認してください"
        continue
      }
      settings[key] = asset.id
      labels[key] = asset.label
    }
  }

  private func resolveSettingAssets(_ settings: inout [String: Any]) throws {
    for (key, purpose) in Self.settingPathPurposes {
      guard let value = settings[key] else { continue }
      if value is NSNull { continue }
      guard let string = value as? String else {
        throw RemoteControlConfigurationError.invalidAsset(key)
      }
      if string.isEmpty { continue }
      // roiEnhancerModel normally contains a built-in model identifier. Only
      // the opaque asset form is resolved as a filesystem selection.
      if key == "roiEnhancerModel", !string.hasPrefix("asset-") { continue }
      // Python-compatible custom selectors may be a registered model name as
      // well as a file. Absolute paths are never accepted from the browser;
      // files still have to arrive as opaque asset capabilities.
      if Self.customModelSettingKeys.contains(key), !string.hasPrefix("asset-") {
        guard Self.validModelIdentifier(string) else {
          throw RemoteControlConfigurationError.invalidAsset(key)
        }
        continue
      }
      guard let asset = assetRecord(string), let url = assetURL(string) else {
        throw RemoteControlConfigurationError.invalidAsset(key + "_unknown")
      }
      if !FileManager.default.fileExists(atPath: url.path) {
        guard asset.preservedSettingKeys.contains(key) else {
          throw RemoteControlConfigurationError.invalidAsset(key + "_missing")
        }
        // Preserve an existing selection while a removable volume is offline.
        // A new unavailable path can never get
        // such a field-bound capability from the Web browser.
        settings[key] = url.path
        continue
      }
      if let failure = assetSelectionFailure(url, purpose: purpose) {
        throw RemoteControlConfigurationError.invalidAsset(
          key + "_not_selectable_" + failure
        )
      }
      settings[key] = url.path
    }
  }

  private func validate(
    _ value: MiohUserDefaultsSnapshot,
    runner: RestorationRunner
  ) -> Bool {
    let requestedEngine = value.restorationEngine ?? "native"
    let allowedDevices = requestedEngine == "python"
      ? Set(["mps", "cpu", "cuda:0"])
      : Set(["mps"])
    let allowedMergeEncoders = Set([
      "copy", "h264", "hevc", "h264_videotoolbox", "hevc_videotoolbox",
      "libx264", "libx265",
    ])
    guard (value.restorationEngine == nil
        || (runner.supportsPythonEngine
          ? ["native", "python"].contains(value.restorationEngine!)
          : value.restorationEngine == "native")),
      (1...16).contains(value.parallelWorkers),
      (1...3).contains(value.nativeParallelWorkers ?? 1),
      ["process", "thread"].contains(value.executor),
      allowedDevices.contains(value.device),
      (1...128).contains(value.segmentCount),
      (10...3_600).contains(value.segmentDuration),
      allowedMergeEncoders.contains(value.mergeEncoder),
      ["auto", "preset", "custom"].contains(value.encodingMode),
      runner.encodingPresets.contains(value.encodingPreset),
      value.bitrateMultiplier.isFinite, (0.1...100).contains(value.bitrateMultiplier),
      (0...100).contains(value.quality), (0...51).contains(value.qmin),
      (0...51).contains(value.qmax), (1...120_000).contains(value.fps),
      (1...1_001).contains(value.fpsDenominator ?? 1),
      runner.restorationModels.contains(value.restorationModel),
      (1...180).contains(value.maxClipLength),
      (value.restoreMaxFrames == -1 || (1...180).contains(value.restoreMaxFrames)),
      (0...120).contains(value.restoreTemporalOverlap ?? 8),
      value.sharpenStrength.isFinite, (0...2).contains(value.sharpenStrength),
      value.detailBoost.isFinite, (0...1).contains(value.detailBoost),
      value.blendFeather.isFinite, (0...3).contains(value.blendFeather),
      value.textureMix.isFinite, (0...1).contains(value.textureMix),
      value.smoothStrength.isFinite, (0...1).contains(value.smoothStrength),
      (1...4).contains(value.effectUpscale),
      runner.enhancerModels.contains(value.roiEnhancer),
      (1...8).contains(value.roiEnhancerScale),
      value.roiEnhancerStrength.isFinite, (0...1).contains(value.roiEnhancerStrength),
      (0...1_024).contains(value.roiEnhancerTile),
      runner.detectionModels.contains(value.detectionModel),
      (0...300).contains(value.detectionEmptyLookahead),
      value.previewBufferLimit.isFinite, (1...60).contains(value.previewBufferLimit),
      runner.restorationModels.contains(value.previewRestorationModel ?? ""),
      runner.previewDetectionModels.contains(value.previewDetectionModel ?? ""),
      PreviewHLSQuality(rawValue: value.previewHLSQuality ?? "") != nil,
      ["通常", "VR180", "360"].contains(value.previewProjectionMode ?? ""),
      ["Mono", "SBS 左右", "上下"].contains(value.previewVideoLayout ?? ""),
      ["左目", "右目"].contains(value.previewEye ?? ""),
      (value.previewCameraFOV ?? 60).isFinite,
      (45...105).contains(value.previewCameraFOV ?? 60),
      (1...100).contains(value.memoryCleanupInterval),
      value.cleanupTriggerGB.isFinite, (0...1_024).contains(value.cleanupTriggerGB),
      value.mpsMemoryFraction.isFinite, (0.01...1).contains(value.mpsMemoryFraction),
      value.encoderOptions.utf8.count <= 4_096
    else { return false }

    // Keep an already-selected unavailable model visible so unrelated settings
    // can still be changed, but never let the Web API switch to an asset that
    // this installation cannot execute.  Use the requested engine because a
    // single transaction may change both engine and model.
    let engineChanged = requestedEngine != runner.restorationEngine
    if (engineChanged || value.restorationModel != runner.restorationModel),
      !runner.restorationModelAvailability(
        value.restorationModel,
        engine: requestedEngine
      ).available
    { return false }
    if (engineChanged || value.detectionModel != runner.detectionModel),
      !runner.detectionModelAvailability(
        value.detectionModel,
        engine: requestedEngine
      ).available
    { return false }
    if (engineChanged || value.previewRestorationModel != runner.previewRestorationModel),
      !runner.restorationModelAvailability(
        value.previewRestorationModel ?? "",
        engine: requestedEngine
      ).available
    { return false }
    if (engineChanged || value.previewDetectionModel != runner.previewDetectionModel),
      !runner.detectionModelAvailability(
        value.previewDetectionModel ?? "",
        engine: requestedEngine
      ).available
    { return false }
    if value.roiEnhancer != "none", !value.roiEnhancerModel.isEmpty,
      (engineChanged || value.roiEnhancerModel != runner.roiEnhancerModel),
      !runner.roiEnhancerModelAvailability(
        value.roiEnhancerModel,
        engine: requestedEngine
      ).available
    { return false }

    if value.restorationModel == "カスタム" && value.customRestorationModel.isEmpty {
      return false
    }
    if value.detectionModel == "カスタム" && value.customDetectionModel.isEmpty {
      return false
    }
    if value.previewRestorationModel == "カスタム"
      && (value.previewCustomRestorationModel ?? "").isEmpty
    { return false }
    if value.previewDetectionModel == "カスタム"
      && (value.previewCustomDetectionModel ?? "").isEmpty
    { return false }
    let roiBuiltIns = runner.roiEnhancerModelOptions(for: value.roiEnhancer).map(\.name)
    if value.roiEnhancer != "none", !value.roiEnhancerModel.isEmpty,
      !roiBuiltIns.contains(value.roiEnhancerModel),
      !value.roiEnhancerModel.hasPrefix("/")
    { return false }
    return true
  }

  private func integer(_ value: Any?) -> Int? {
    guard let number = value as? NSNumber, String(cString: number.objCType) != "c" else {
      return nil
    }
    let double = number.doubleValue
    guard double.isFinite, double.rounded() == double else { return nil }
    return Int(exactly: number.int64Value)
  }

  private func validHostHeader(_ request: RemoteHTTPRequest) -> Bool {
    guard let raw = request.headers["host"]?.trimmingCharacters(in: .whitespacesAndNewlines),
      !raw.isEmpty
    else { return false }
    let host: String
    if raw.hasPrefix("["), let closing = raw.firstIndex(of: "]") {
      host = String(raw[raw.index(after: raw.startIndex)..<closing]).lowercased()
    } else {
      host = raw.split(separator: ":", maxSplits: 1).first.map(String.init)?.lowercased() ?? ""
    }
    var allowed: Set<String> = ["localhost", "127.0.0.1", "::1"]
    let localName = (Host.current().localizedName ?? ProcessInfo.processInfo.hostName).lowercased()
    if !localName.isEmpty {
      allowed.insert(localName)
      allowed.insert(localName.contains(".") ? localName : localName + ".local")
    }
    allowed.formUnion(Host.current().addresses.map { $0.lowercased() })
    return allowed.contains(host)
  }

  // MARK: - Opaque, allow-listed asset catalogue

  private func listAssets(
    _ request: RemoteHTTPRequest,
    runner: RestorationRunner,
    player: RealtimePlayerController
  ) -> RemoteHTTPResponse {
    guard let object = requestJSONObject(request),
      let rawPurpose = object["purpose"] as? String,
      let purpose = RemoteControlAssetPurpose(rawValue: rawPurpose)
    else { return .jsonError(400, "invalid_asset_purpose") }
    let offset = integer(object["offset"]) ?? 0
    let query = (object["query"] as? String ?? "")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard offset >= 0, query.utf8.count <= 256 else {
      return .jsonError(400, "invalid_asset_page")
    }

    if object["directoryID"] == nil || object["directoryID"] is NSNull {
      let roots = allowedAssetRoots().compactMap { root -> [String: Any]? in
        guard let asset = registerAsset(root.url) else { return nil }
        return [
          "assetID": asset.id,
          "name": root.label,
          "kind": "directory",
          "selectable": assetIsSelectable(root.url, purpose: purpose),
          "browseable": true,
        ]
      }
      return .json(status: 200, object: [
        "ok": true, "current": NSNull(), "parentID": NSNull(),
        "entries": roots, "offset": 0, "total": roots.count,
        "previousOffset": NSNull(), "nextOffset": NSNull(), "truncated": false,
      ])
    }

    guard let directoryID = object["directoryID"] as? String,
      let directory = assetURL(directoryID),
      (try? directory.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    else { return .jsonError(400, "invalid_asset_directory") }

    let keys: Set<URLResourceKey> = [
      .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
    ]
    guard let urls = try? FileManager.default.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: Array(keys),
      options: [.skipsHiddenFiles]
    ) else { return .jsonError(403, "asset_directory_unreadable") }

    let normalizedQuery = query.folding(
      options: [.caseInsensitive, .diacriticInsensitive], locale: .current
    )
    let candidates = urls.compactMap { url -> (URL, URLResourceValues)? in
      guard let values = try? url.resourceValues(forKeys: keys),
        values.isSymbolicLink != true,
        (values.isDirectory == true || values.isRegularFile == true)
      else { return nil }
      if !normalizedQuery.isEmpty {
        let name = url.lastPathComponent.folding(
          options: [.caseInsensitive, .diacriticInsensitive], locale: .current
        )
        guard name.contains(normalizedQuery) else { return nil }
      }
      return (url, values)
    }.sorted { left, right in
      let leftDirectory = left.1.isDirectory == true
      let rightDirectory = right.1.isDirectory == true
      if leftDirectory != rightDirectory { return leftDirectory }
      return left.0.lastPathComponent.localizedStandardCompare(
        right.0.lastPathComponent
      ) == .orderedAscending
    }
    let pageSize = 512
    let start = min(offset, candidates.count)
    let end = min(start + pageSize, candidates.count)
    let entries: [[String: Any]] = candidates[start..<end].compactMap { url, values in
      guard let asset = registerAsset(url) else { return nil }
      let package = Self.modelPackageExtensions.contains(url.pathExtension.lowercased())
      return [
        "assetID": asset.id,
        "name": url.lastPathComponent,
        "kind": values.isDirectory == true ? "directory" : "file",
        "selectable": assetIsSelectable(url, purpose: purpose),
        "browseable": values.isDirectory == true && !package,
        "size": values.fileSize as Any,
      ]
    }
    let parent = parentAsset(of: directory)
    return .json(status: 200, object: [
      "ok": true,
      "current": displayLabel(for: directory),
      "currentID": directoryID,
      "parentID": parent?.id as Any,
      "entries": entries,
      "offset": start,
      "total": candidates.count,
      "previousOffset": start > 0 ? max(0, start - pageSize) : NSNull(),
      "nextOffset": end < candidates.count ? end : NSNull(),
      "truncated": candidates.count > pageSize,
    ])
  }

  private func createAssetDirectory(_ request: RemoteHTTPRequest) -> RemoteHTTPResponse {
    guard let object = requestJSONObject(request),
      let parentID = object["parentID"] as? String,
      let parent = assetURL(parentID),
      assetIsSelectable(parent, purpose: .directory),
      let rawName = object["name"] as? String
    else { return .jsonError(400, "invalid_directory_request") }
    let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !name.isEmpty, name != ".", name != "..", !name.contains("/"),
      !name.contains("\\"), !name.contains("\0"), name.utf8.count <= 255
    else { return .jsonError(400, "invalid_directory_name") }
    let url = parent.appendingPathComponent(name, isDirectory: true).standardizedFileURL
    guard isAllowedAssetURL(url), !FileManager.default.fileExists(atPath: url.path) else {
      return .jsonError(409, "directory_exists_or_outside_root")
    }
    do {
      try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
      guard let asset = registerAsset(url) else {
        throw RemoteControlConfigurationError.invalidAsset("directory")
      }
      return .json(status: 201, object: [
        "ok": true, "assetID": asset.id, "name": url.lastPathComponent,
      ])
    } catch {
      return .jsonError(403, "directory_creation_failed")
    }
  }

  private func registerAsset(
    _ input: URL,
    preservesExistingSelection: Bool = false,
    preservedSettingKey: String? = nil
  ) -> RemoteControlAsset? {
    let url = canonicalAssetURL(input)
    let preserves = preservesExistingSelection || preservedSettingKey != nil
    guard isAllowedAssetURL(url) || preserves else {
      return nil
    }
    if let existing = assetCatalog.values.first(where: { $0.url == url }) {
      var preservedKeys = existing.preservedSettingKeys
      if let preservedSettingKey { preservedKeys.insert(preservedSettingKey) }
      if preserves || preservedKeys != existing.preservedSettingKeys {
        let upgraded = RemoteControlAsset(
          id: existing.id,
          url: existing.url,
          label: existing.label,
          preservesExistingSelection: existing.preservesExistingSelection || preserves,
          preservedSettingKeys: preservedKeys
        )
        assetCatalog[existing.id] = upgraded
        return upgraded
      }
      return existing
    }
    if assetCatalog.count >= Self.maximumAssetCount {
      guard preserves,
        let victim = assetCatalog.first(where: { !$0.value.preservesExistingSelection })?.key
      else { return nil }
      assetCatalog.removeValue(forKey: victim)
    }
    var preservedKeys = Set<String>()
    if let preservedSettingKey { preservedKeys.insert(preservedSettingKey) }
    let asset = RemoteControlAsset(
      id: "asset-" + UUID().uuidString.lowercased(),
      url: url,
      label: displayLabel(for: url),
      preservesExistingSelection: preserves,
      preservedSettingKeys: preservedKeys
    )
    assetCatalog[asset.id] = asset
    return asset
  }

  private func assetURL(_ id: String) -> URL? {
    guard id.hasPrefix("asset-"), let asset = assetCatalog[id],
      (isAllowedAssetURL(asset.url) || asset.preservesExistingSelection),
      (FileManager.default.fileExists(atPath: asset.url.path)
        || asset.preservesExistingSelection)
    else { return nil }
    return asset.url
  }

  private func assetRecord(_ id: String) -> RemoteControlAsset? {
    guard id.hasPrefix("asset-") else { return nil }
    return assetCatalog[id]
  }

  private func parentAsset(of url: URL) -> RemoteControlAsset? {
    let parent = canonicalAssetURL(url.deletingLastPathComponent())
    guard parent != url, isAllowedAssetURL(parent) else { return nil }
    return registerAsset(parent)
  }

  private func allowedAssetRoots() -> [(url: URL, label: String)] {
    var roots: [(URL, String)] = []
    let home = canonicalAssetURL(FileManager.default.homeDirectoryForCurrentUser)
    for (component, label) in [
      ("Movies", "ムービー"), ("Desktop", "デスクトップ"),
      ("Downloads", "ダウンロード"), ("Documents", "書類"),
    ] {
      let candidate = canonicalAssetURL(
        home.appendingPathComponent(component, isDirectory: true)
      )
      var isDirectory: ObjCBool = false
      if FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
        isDirectory.boolValue
      {
        roots.append((candidate, label))
      }
    }
    let temporary = canonicalAssetURL(FileManager.default.temporaryDirectory)
    if temporary != home { roots.append((temporary, "一時フォルダ")) }
    let systemTemporary = canonicalAssetURL(
      URL(fileURLWithPath: "/private/tmp", isDirectory: true)
    )
    if systemTemporary != home && systemTemporary != temporary {
      roots.append((systemTemporary, "システム一時フォルダ"))
    }
    let volumes = URL(fileURLWithPath: "/Volumes", isDirectory: true)
    if let mounted = try? FileManager.default.contentsOfDirectory(
      at: volumes,
      includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
      options: [.skipsHiddenFiles]
    ) {
      for volume in mounted {
        let values = try? volume.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values?.isDirectory == true, values?.isSymbolicLink != true else { continue }
        roots.append((canonicalAssetURL(volume), volume.lastPathComponent))
      }
    }
    var seen = Set<String>()
    return roots.filter { seen.insert($0.0.path).inserted }
  }

  private func isAllowedAssetURL(_ input: URL) -> Bool {
    let url = canonicalAssetURL(input)
    return allowedAssetRoots().contains { root in
      Self.isDescendant(url, of: root.url)
    }
  }

  private static func isDescendant(_ candidate: URL, of root: URL) -> Bool {
    let rootParts = root.pathComponents
    let candidateParts = candidate.pathComponents
    return candidateParts.count >= rootParts.count
      && Array(candidateParts.prefix(rootParts.count)) == rootParts
  }

  private func displayLabel(for input: URL) -> String {
    let url = canonicalAssetURL(input)
    for root in allowedAssetRoots() where Self.isDescendant(url, of: root.url) {
      let relative = url.pathComponents.dropFirst(root.url.pathComponents.count)
      return ([root.label] + relative).joined(separator: "/")
    }
    return url.lastPathComponent
  }

  /// Foundation's URL implementation can leave a symlink in the final path
  /// component (`/tmp` on macOS). NSString resolves that final component too,
  /// which is required both for selecting `/tmp` and for the root-containment
  /// check to remain symlink-safe.
  private func canonicalAssetURL(_ input: URL) -> URL {
    var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
    if input.path.withCString({ realpath($0, &buffer) }) != nil {
      return URL(
        fileURLWithPath: String(cString: buffer),
        isDirectory: input.hasDirectoryPath
      )
    }
    // A newly-created child does not exist yet. Resolve its existing parent,
    // then append the single validated component used by create-directory.
    let parent = input.deletingLastPathComponent()
    var parentBuffer = [CChar](repeating: 0, count: Int(PATH_MAX))
    if parent.path.withCString({ realpath($0, &parentBuffer) }) != nil {
      return URL(
        fileURLWithPath: String(cString: parentBuffer),
        isDirectory: true
      ).appendingPathComponent(
        input.lastPathComponent,
        isDirectory: input.hasDirectoryPath
      )
    }
    return URL(
      fileURLWithPath: (input.path as NSString).standardizingPath,
      isDirectory: input.hasDirectoryPath
    )
  }

  private func assetIsSelectable(_ url: URL, purpose: RemoteControlAssetPurpose) -> Bool {
    assetSelectionFailure(url, purpose: purpose) == nil
  }

  private func assetSelectionFailure(
    _ url: URL,
    purpose: RemoteControlAssetPurpose
  ) -> String? {
    let canonical = canonicalAssetURL(url)
    let pinned = assetCatalog.values.contains {
      $0.url == canonical && $0.preservesExistingSelection
    }
    guard isAllowedAssetURL(canonical) || pinned else { return "outside_root" }
    guard let values = try? url.resourceValues(forKeys: [
      .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
    ]) else { return "unreadable" }
    guard values.isSymbolicLink != true else { return "symlink" }
    let ext = url.pathExtension.lowercased()
    switch purpose {
    case .directory:
      guard values.isDirectory == true else { return "not_directory" }
      return FileManager.default.isWritableFile(atPath: url.path) ? nil : "not_writable"
    case .input:
      return values.isDirectory == true
        || (values.isRegularFile == true && Self.videoExtensions.contains(ext))
        ? nil : "not_video_or_directory"
    case .previewInput:
      return values.isRegularFile == true && Self.videoExtensions.contains(ext)
        ? nil : "not_video"
    case .model:
      return (values.isRegularFile == true || values.isDirectory == true)
        && Self.modelExtensions.contains(ext) ? nil : "not_model"
    }
  }

  private static let videoExtensions: Set<String> = [
    "mp4", "m4v", "mov", "mkv", "avi", "webm", "ts", "mts", "m2ts",
    "mpg", "mpeg", "wmv", "asf",
  ]
  private static let modelPackageExtensions: Set<String> = [
    "aimodel", "aimodelc", "mlpackage", "mlmodelc",
  ]
  private static let modelExtensions = modelPackageExtensions.union([
    "pth", "pt", "safetensors", "ckpt", "mlmodel",
  ])
  private static let settingPathPurposes: [String: RemoteControlAssetPurpose] = [
    "inputPath": .input,
    "outputPath": .directory,
    "tempDirectory": .directory,
    "ffmpegTempDirectory": .directory,
    "ladaTempDirectory": .directory,
    "customRestorationModel": .model,
    "roiEnhancerModel": .model,
    "customDetectionModel": .model,
    "previewCustomRestorationModel": .model,
    "previewCustomDetectionModel": .model,
  ]
  private static let customModelSettingKeys: Set<String> = [
    "customRestorationModel", "customDetectionModel",
    "previewCustomRestorationModel", "previewCustomDetectionModel",
  ]
  private static let settingPathKeys = Array(settingPathPurposes.keys)

  private static func validModelIdentifier(_ value: String) -> Bool {
    guard !value.isEmpty, value.utf8.count <= 255, !value.hasPrefix("/"),
      !value.contains("\\"), !value.contains("\0"), value != ".", value != ".."
    else { return false }
    return value.unicodeScalars.allSatisfy {
      CharacterSet.alphanumerics.contains($0)
        || "-_.:+".unicodeScalars.contains($0)
    }
  }

  private func statusResponse(
    runner: RestorationRunner,
    player: RealtimePlayerController
  ) -> RemoteHTTPResponse {
    let safePosition = player.position.isFinite ? player.position : 0
    let safeDuration = player.duration.isFinite ? player.duration : 0
    let safeBuffered = player.bufferedSeconds.isFinite ? player.bufferedSeconds : 0
    let clusterRunning = cluster?.isRunning == true
    let activeProgress = clusterRunning ? (cluster?.progress ?? 0) : runner.progress
    let safeProgress = activeProgress.isFinite ? activeProgress : 0
    return .json(
      status: 200,
      object: [
        "ok": true,
        "server": [
          "enabled": enabled,
          "port": port,
          "id": serverID.uuidString.lowercased(),
          "apiVersion": 1,
        ],
        "playback": [
          "state": player.state.rawValue,
          "position": safePosition,
          "duration": safeDuration,
          "bufferedSeconds": safeBuffered,
          "volume": player.volume,
          "muted": player.muted,
          "input": player.previewInputURL?.lastPathComponent as Any,
          "hasError": !player.errorMessage.isEmpty,
        ],
        "export": [
          "running": runner.isRunning || clusterRunning,
          "status": clusterRunning ? (cluster?.status ?? runner.status) : runner.status,
          "progress": safeProgress,
          "input": runner.inputURL?.lastPathComponent as Any,
          "output": runner.outputURL?.lastPathComponent as Any,
        ],
      ])
  }

  private func requestJSONObject(_ request: RemoteHTTPRequest) -> [String: Any]? {
    guard request.headers["content-type"]?.lowercased().hasPrefix("application/json") == true,
      !request.body.isEmpty,
      let value = try? JSONSerialization.jsonObject(with: request.body),
      let object = value as? [String: Any]
    else { return nil }
    return object
  }

  private func number(_ value: Any?) -> Double? {
    guard let number = value as? NSNumber else { return nil }
    // JSON booleans bridge to NSNumber too; do not accept them as numeric API values.
    guard String(cString: number.objCType) != "c" else { return nil }
    return number.doubleValue
  }

  private static func constantTimeEquals(_ lhs: String, _ rhs: String) -> Bool {
    let left = Array(lhs.utf8)
    let right = Array(rhs.utf8)
    var difference = UInt8(truncatingIfNeeded: left.count ^ right.count)
    let count = max(left.count, right.count)
    for index in 0..<count {
      let a = index < left.count ? left[index] : 0
      let b = index < right.count ? right[index] : 0
      difference |= a ^ b
    }
    return difference == 0
  }

  private static func availableURLs(port: Int) -> [URL] {
    var hosts = ["127.0.0.1"]
    let localName = Host.current().localizedName ?? ProcessInfo.processInfo.hostName
    if !localName.isEmpty {
      hosts.append(localName.contains(".") ? localName : localName + ".local")
    }
    for address in Host.current().addresses {
      if isUsableIPv4(address) || isUsableIPv6(address) {
        hosts.append(address)
      }
    }
    var seen = Set<String>()
    return hosts.compactMap { host in
      guard seen.insert(host).inserted else { return nil }
      let formattedHost = host.contains(":") ? "[\(host)]" : host
      return URL(string: "http://\(formattedHost):\(port)/")
    }
  }

  private static func isUsableIPv4(_ value: String) -> Bool {
    let pieces = value.split(separator: ".")
    guard pieces.count == 4,
      pieces.allSatisfy({ piece in Int(piece).map { (0...255).contains($0) } ?? false })
    else { return false }
    return !value.hasPrefix("127.") && !value.hasPrefix("169.254.") && value != "0.0.0.0"
  }

  private static func isUsableIPv6(_ value: String) -> Bool {
    let normalized = value.lowercased()
    return normalized.contains(":")
      && normalized != "::"
      && normalized != "::1"
      && !normalized.hasPrefix("fe80:")
      && !normalized.contains("%")
  }
}

// MARK: - Keychain token

/// Human-entered LAN credential. Twelve symbols from a 32-character alphabet
/// provide 60 bits of entropy while avoiding visually ambiguous I/O/0/1.
/// Hyphens, whitespace and letter case are ignored when accepting input.
private enum RemoteControlAccessToken {
  static let compactLength = 12
  private static let groupLength = 4
  private static let alphabet = Array("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
  private static let allowed = Set(alphabet)

  static func generate() -> String {
    var bytes = [UInt8](repeating: 0, count: compactLength)
    if SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) != errSecSuccess {
      var generator = SystemRandomNumberGenerator()
      for index in bytes.indices {
        bytes[index] = UInt8.random(in: UInt8.min...UInt8.max, using: &generator)
      }
    }
    let compact = String(bytes.map { alphabet[Int($0) & 31] })
    return format(compact)
  }

  static func canonicalize(_ value: String) -> String? {
    let compact = value.uppercased().filter { character in
      character != "-" && !character.isWhitespace
    }
    guard compact.count == compactLength, compact.allSatisfy({ allowed.contains($0) })
    else { return nil }
    return format(compact)
  }

  private static func format(_ compact: String) -> String {
    var groups: [String] = []
    var start = compact.startIndex
    while start < compact.endIndex {
      let end = compact.index(start, offsetBy: groupLength, limitedBy: compact.endIndex)
        ?? compact.endIndex
      groups.append(String(compact[start..<end]))
      start = end
    }
    return groups.joined(separator: "-")
  }
}

private enum RemoteControlTokenStore {
  struct Result {
    let token: String
    let warning: String?
  }

  private static let account = "bearer-token-v1"
  private static var service: String {
    (Bundle.main.bundleIdentifier ?? "com.okatti.lada.coreai") + ".remote-control"
  }

  static func loadOrCreate() -> Result {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
      kSecUseDataProtectionKeychain as String: true,
      kSecMatchLimit as String: kSecMatchLimitOne,
      kSecReturnData as String: true,
    ]
    var item: CFTypeRef?
    let readStatus = SecItemCopyMatching(query as CFDictionary, &item)
    if readStatus == errSecSuccess, let data = item as? Data,
      let value = String(data: data, encoding: .utf8),
      let canonical = RemoteControlAccessToken.canonicalize(value)
    {
      removeFallback()
      if canonical == value {
        return Result(token: canonical, warning: nil)
      }
      return Result(token: canonical, warning: store(canonical))
    }
    if let token = loadFallback() {
      let warning = store(token)
      if warning == nil { removeFallback() }
      return Result(token: token, warning: warning)
    }
    let token = generate()
    let warning = store(token)
    if warning != nil { saveFallback(token) } else { removeFallback() }
    return Result(token: token, warning: warning)
  }

  static func regenerate() -> Result {
    let token = generate()
    let warning = store(token)
    if warning != nil { saveFallback(token) } else { removeFallback() }
    return Result(token: token, warning: warning)
  }

  private static func store(_ token: String) -> String? {
    let identity: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
      kSecUseDataProtectionKeychain as String: true,
    ]
    let attributes: [String: Any] = [
      kSecValueData as String: Data(token.utf8),
      kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
    ]
    let updateStatus = SecItemUpdate(identity as CFDictionary, attributes as CFDictionary)
    if updateStatus == errSecSuccess { return nil }
    guard updateStatus == errSecItemNotFound else {
      return "Keychain update failed (\(updateStatus))"
    }
    var addition: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
      kSecUseDataProtectionKeychain as String: true,
    ]
    addition.merge(attributes) { _, new in new }
    let addStatus = SecItemAdd(addition as CFDictionary, nil)
    return addStatus == errSecSuccess ? nil : "Keychain add failed (\(addStatus))"
  }

  /// Development signatures may not have Data Protection Keychain access.
  /// Keep a mode-0600 fallback so that this never blocks app startup or rotates
  /// the remote token on every launch. Eligible distribution builds continue
  /// to use Keychain as the primary store.
  private static var fallbackURL: URL? {
    guard
      let root = FileManager.default.urls(
        for: .applicationSupportDirectory,
        in: .userDomainMask
      ).first
    else { return nil }
    return
      root
      .appendingPathComponent("mioh", isDirectory: true)
      .appendingPathComponent("remote-control-token-v1", isDirectory: false)
  }

  private static func loadFallback() -> String? {
    guard let url = fallbackURL,
      let value = try? String(contentsOf: url, encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines),
      let canonical = RemoteControlAccessToken.canonicalize(value)
    else { return nil }
    return canonical
  }

  private static func saveFallback(_ token: String) {
    guard let url = fallbackURL else { return }
    let directory = url.deletingLastPathComponent()
    try? FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    try? Data((token + "\n").utf8).write(to: url, options: .atomic)
    try? FileManager.default.setAttributes(
      [.posixPermissions: 0o600],
      ofItemAtPath: url.path
    )
  }

  private static func removeFallback() {
    guard let url = fallbackURL else { return }
    try? FileManager.default.removeItem(at: url)
  }

  private static func generate() -> String {
    RemoteControlAccessToken.generate()
  }
}

// MARK: - HTTP transport

private struct RemoteHTTPRequest: Sendable {
  let method: String
  let path: String
  let headers: [String: String]
  let body: Data
}

private struct RemoteHTTPFileBody: Sendable {
  let url: URL
  let offset: UInt64
  let length: UInt64
  let sendBody: Bool
}

private enum RemoteHTTPResponseBody: Sendable {
  case data(Data, representedLength: Int)
  case file(RemoteHTTPFileBody)
}

private struct RemoteHTTPResponse: Sendable {
  let status: Int
  let headers: [String: String]
  let body: RemoteHTTPResponseBody

  static func html(_ text: String) -> Self {
    Self(
      status: 200,
      headers: [
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
        "Content-Security-Policy":
          "default-src 'self'; connect-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
      ],
      body: .data(Data(text.utf8), representedLength: Data(text.utf8).count)
    )
  }

  static func data(
    status: Int,
    contentType: String,
    data: Data,
    representedLength: Int? = nil,
    headers: [String: String] = [:]
  ) -> Self {
    var resultHeaders = headers
    resultHeaders["Content-Type"] = contentType
    return Self(
      status: status,
      headers: resultHeaders,
      body: .data(data, representedLength: representedLength ?? data.count)
    )
  }

  static func file(
    status: Int,
    url: URL,
    offset: UInt64,
    length: UInt64,
    sendBody: Bool,
    contentType: String,
    headers: [String: String] = [:]
  ) -> Self {
    var resultHeaders = headers
    resultHeaders["Content-Type"] = contentType
    return Self(
      status: status,
      headers: resultHeaders,
      body: .file(
        RemoteHTTPFileBody(
          url: url,
          offset: offset,
          length: length,
          sendBody: sendBody
        )
      )
    )
  }

  static func accepted() -> Self {
    json(status: 202, object: ["ok": true, "accepted": true])
  }

  static func json(status: Int, object: [String: Any], headers: [String: String] = [:]) -> Self {
    let sanitized = sanitizeJSON(object)
    let data =
      (try? JSONSerialization.data(withJSONObject: sanitized, options: [.sortedKeys]))
      ?? Data("{\"ok\":false,\"error\":\"serialization_failed\"}".utf8)
    var resultHeaders = headers
    resultHeaders["Content-Type"] = "application/json; charset=utf-8"
    resultHeaders["Cache-Control"] = "no-store"
    return Self(
      status: status,
      headers: resultHeaders,
      body: .data(data, representedLength: data.count)
    )
  }

  static func jsonError(
    _ status: Int,
    _ error: String,
    headers: [String: String] = [:]
  ) -> Self {
    json(status: status, object: ["ok": false, "error": error], headers: headers)
  }

  static func methodNotAllowed(_ methods: [String]) -> Self {
    jsonError(405, "method_not_allowed", headers: ["Allow": methods.joined(separator: ", ")])
  }

  private static func sanitizeJSON(_ value: Any) -> Any {
    if value is NSNull { return value }
    if let dictionary = value as? [String: Any] {
      return dictionary.mapValues(sanitizeJSON)
    }
    if let array = value as? [Any] { return array.map(sanitizeJSON) }
    if let optional = value as? OptionalProtocol {
      return optional.remoteJSONValue.map(sanitizeJSON) ?? NSNull()
    }
    return value
  }
}

private enum RemoteHTTPByteRange {
  enum Result: Equatable {
    case full
    case range(UInt64, UInt64)
    case invalid
  }

  /// Accepts exactly one RFC 7233 byte range. Multiple ranges would require a
  /// multipart body and are deliberately rejected to keep memory bounded.
  static func parse(_ value: String?, fileSize: UInt64) -> Result {
    guard let value else { return .full }
    guard fileSize > 0, value.hasPrefix("bytes="), !value.contains(",") else {
      return .invalid
    }
    let spec = value.dropFirst("bytes=".count)
    guard !spec.isEmpty, spec.allSatisfy({ $0.isNumber || $0 == "-" }),
      spec.filter({ $0 == "-" }).count == 1,
      let dash = spec.firstIndex(of: "-")
    else {
      return .invalid
    }
    let left = spec[..<dash]
    let right = spec[spec.index(after: dash)...]
    if left.isEmpty {
      guard let suffix = UInt64(right), suffix > 0 else { return .invalid }
      let length = min(suffix, fileSize)
      return .range(fileSize - length, fileSize - 1)
    }
    guard let lower = UInt64(left), lower < fileSize else { return .invalid }
    if right.isEmpty { return .range(lower, fileSize - 1) }
    guard let requestedUpper = UInt64(right), requestedUpper >= lower else {
      return .invalid
    }
    return .range(lower, min(requestedUpper, fileSize - 1))
  }
}

/// Lets the JSON sanitizer turn an optional filename into JSON null without
/// force-unwrapping application state.
private protocol OptionalProtocol {
  var remoteJSONValue: Any? { get }
}

extension Optional: OptionalProtocol {
  fileprivate var remoteJSONValue: Any? { self }
}

private enum RemoteHTTPServerState: Sendable {
  case ready(NWEndpoint.Port)
  case failed(String)
  case cancelled
}

private struct RemoteHTTPBonjourService: Sendable {
  let name: String
  let type: String
  let txtRecord: NWTXTRecord
}

private enum RemoteControlServerIdentity {
  private static let defaultsKey = "mioh.remote-control.server-id.v1"

  static func loadOrCreate(defaults: UserDefaults = .standard) -> UUID {
    if let value = defaults.string(forKey: defaultsKey),
      let identifier = UUID(uuidString: value)
    {
      return identifier
    }
    let identifier = UUID()
    defaults.set(identifier.uuidString.lowercased(), forKey: defaultsKey)
    return identifier
  }
}

private final class RemoteHTTPServerEngine: @unchecked Sendable {
  typealias RequestHandler =
    @Sendable (
      RemoteHTTPRequest,
      @escaping @Sendable (RemoteHTTPResponse) -> Void
    ) -> Void
  typealias StateHandler = @Sendable (RemoteHTTPServerState) -> Void

  private let queue = DispatchQueue(label: "com.okatti.mioh.remote-http")
  private var listener: NWListener?
  private var connections: [UUID: RemoteHTTPConnection] = [:]
  private var requestHandler: RequestHandler?
  private var stateHandler: StateHandler?
  private let maximumConnections = 16

  func start(
    port: NWEndpoint.Port,
    service: RemoteHTTPBonjourService? = nil,
    requestHandler: @escaping RequestHandler,
    stateHandler: @escaping StateHandler
  ) throws {
    let parameters = NWParameters.tcp
    parameters.acceptLocalOnly = true
    let listener = try NWListener(using: parameters, on: port)
    if let service {
      listener.service = NWListener.Service(
        name: service.name,
        type: service.type,
        domain: "local.",
        txtRecord: service.txtRecord
      )
    }
    self.listener = listener
    self.requestHandler = requestHandler
    self.stateHandler = stateHandler

    listener.stateUpdateHandler = { [weak self, weak listener] state in
      guard let self else { return }
      switch state {
      case .ready:
        self.stateHandler?(.ready(listener?.port ?? port))
      case .failed(let error):
        self.stateHandler?(.failed(error.localizedDescription))
      case .cancelled:
        self.stateHandler?(.cancelled)
      default:
        break
      }
    }
    listener.newConnectionHandler = { [weak self] connection in
      self?.accept(connection)
    }
    listener.start(queue: queue)
  }

  func stop() {
    listener?.cancel()
    listener = nil
    queue.async { [weak self] in
      guard let self else { return }
      let activeConnections = Array(self.connections.values)
      self.connections.removeAll()
      for connection in activeConnections { connection.cancel() }
      self.requestHandler = nil
    }
  }

  private func accept(_ connection: NWConnection) {
    dispatchPrecondition(condition: .onQueue(queue))
    guard connections.count < maximumConnections, let requestHandler else {
      rejectBusy(connection)
      return
    }
    let id = UUID()
    let client = RemoteHTTPConnection(
      id: id,
      connection: connection,
      queue: queue,
      requestHandler: requestHandler,
      completion: { [weak self] completedID in
        self?.connections.removeValue(forKey: completedID)
      }
    )
    connections[id] = client
    client.start()
  }

  private func rejectBusy(_ connection: NWConnection) {
    connection.stateUpdateHandler = { state in
      guard case .ready = state else { return }
      let response = RemoteHTTPWire.encode(.jsonError(503, "too_many_connections"))
      connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
    }
    connection.start(queue: queue)
  }
}

private final class RemoteHTTPConnection: @unchecked Sendable {
  private static let maximumHeaderBytes = 32 * 1024
  private static let maximumBodyBytes = 64 * 1024
  private static let fileChunkBytes = 256 * 1024

  private let id: UUID
  private let connection: NWConnection
  private let queue: DispatchQueue
  private let requestHandler: RemoteHTTPServerEngine.RequestHandler
  private let completion: @Sendable (UUID) -> Void
  private var buffer = Data()
  private var finished = false
  private var timeout: DispatchWorkItem?

  init(
    id: UUID,
    connection: NWConnection,
    queue: DispatchQueue,
    requestHandler: @escaping RemoteHTTPServerEngine.RequestHandler,
    completion: @escaping @Sendable (UUID) -> Void
  ) {
    self.id = id
    self.connection = connection
    self.queue = queue
    self.requestHandler = requestHandler
    self.completion = completion
  }

  func start() {
    let timeout = DispatchWorkItem { [weak self] in
      self?.respond(.jsonError(408, "request_timeout"))
    }
    self.timeout = timeout
    queue.asyncAfter(deadline: .now() + 15, execute: timeout)
    connection.stateUpdateHandler = { [weak self] state in
      switch state {
      case .ready:
        self?.receive()
      case .failed, .cancelled:
        self?.finish()
      default:
        break
      }
    }
    connection.start(queue: queue)
  }

  func cancel() {
    connection.cancel()
    finish()
  }

  private func receive() {
    guard !finished else { return }
    connection.receive(minimumIncompleteLength: 1, maximumLength: 16 * 1024) {
      [weak self] data, _, complete, error in
      guard let self, !self.finished else { return }
      if let data { self.buffer.append(data) }
      if self.buffer.count > Self.maximumHeaderBytes + Self.maximumBodyBytes {
        self.respond(.jsonError(413, "request_too_large"))
        return
      }
      switch RemoteHTTPRequestParser.inspect(
        self.buffer,
        maximumHeaderBytes: Self.maximumHeaderBytes,
        maximumBodyBytes: Self.maximumBodyBytes
      ) {
      case .needMore:
        if complete || error != nil {
          self.respond(.jsonError(400, "incomplete_request"))
        } else {
          self.receive()
        }
      case .failure(let status, let code):
        self.respond(.jsonError(status, code))
      case .request(let request):
        self.requestHandler(request) { [weak self] response in
          self?.queue.async { self?.respond(response) }
        }
      }
    }
  }

  private func respond(_ response: RemoteHTTPResponse) {
    guard !finished else { return }
    finished = true
    timeout?.cancel()
    switch response.body {
    case .data(let body, let representedLength):
      let data = RemoteHTTPWire.encode(
        response,
        data: body,
        representedLength: representedLength
      )
      connection.send(
        content: data,
        completion: .contentProcessed { [weak self] _ in
          self?.connection.cancel()
          self?.finish()
        })
    case .file(let file):
      sendFile(response: response, file: file)
    }
  }

  private func sendFile(response: RemoteHTTPResponse, file: RemoteHTTPFileBody) {
    let head = RemoteHTTPWire.encodeHead(response, contentLength: file.length)
    guard file.sendBody, file.length > 0 else {
      connection.send(
        content: head,
        completion: .contentProcessed { [weak self] _ in
          self?.connection.cancel()
          self?.finish()
        })
      return
    }
    guard let handle = try? FileHandle(forReadingFrom: file.url) else {
      sendUnavailableFileResponse()
      return
    }
    do {
      try handle.seek(toOffset: file.offset)
    } catch {
      try? handle.close()
      // The coordinator may retire a sliding-window segment between routing
      // and opening. It is safer to fail the request than serve a new path.
      sendUnavailableFileResponse()
      return
    }
    connection.send(
      content: head,
      completion: .contentProcessed { [weak self] error in
        guard let self, error == nil else {
          try? handle.close()
          self?.connection.cancel()
          self?.finish()
          return
        }
        self.sendFileChunk(handle: handle, remaining: file.length)
      })
  }

  private func sendUnavailableFileResponse() {
    let unavailable = RemoteHTTPResponse.jsonError(404, "stream_not_found")
    let data = RemoteHTTPWire.encode(unavailable)
    connection.send(
      content: data,
      completion: .contentProcessed { [weak self] _ in
        self?.connection.cancel()
        self?.finish()
      })
  }

  private func sendFileChunk(handle: FileHandle, remaining: UInt64) {
    guard remaining > 0 else {
      try? handle.close()
      connection.cancel()
      finish()
      return
    }
    let length = Int(min(UInt64(Self.fileChunkBytes), remaining))
    let data: Data
    do {
      data = try handle.read(upToCount: length) ?? Data()
    } catch {
      try? handle.close()
      connection.cancel()
      finish()
      return
    }
    guard !data.isEmpty else {
      try? handle.close()
      connection.cancel()
      finish()
      return
    }
    connection.send(
      content: data,
      completion: .contentProcessed { [weak self] error in
        guard let self, error == nil else {
          try? handle.close()
          self?.connection.cancel()
          self?.finish()
          return
        }
        self.sendFileChunk(
          handle: handle,
          remaining: remaining - UInt64(data.count)
        )
      })
  }

  private func finish() {
    guard finished == false || timeout != nil else { return }
    finished = true
    timeout?.cancel()
    timeout = nil
    completion(id)
  }
}

private enum RemoteHTTPRequestParseResult {
  case needMore(expectedBytes: Int?)
  case failure(status: Int, code: String)
  case request(RemoteHTTPRequest)
}

private enum RemoteHTTPRequestParser {
  private static let tokenCharacters = CharacterSet(
    charactersIn: "!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  )

  static func inspect(
    _ data: Data,
    maximumHeaderBytes: Int,
    maximumBodyBytes: Int
  ) -> RemoteHTTPRequestParseResult {
    let separator = Data([13, 10, 13, 10])
    guard let separatorRange = data.range(of: separator) else {
      return data.count > maximumHeaderBytes
        ? .failure(status: 431, code: "headers_too_large")
        : .needMore(expectedBytes: nil)
    }
    let headerBytes = separatorRange.upperBound
    guard headerBytes <= maximumHeaderBytes else {
      return .failure(status: 431, code: "headers_too_large")
    }
    guard let headerText = String(data: data[..<separatorRange.lowerBound], encoding: .utf8),
      !headerText.unicodeScalars.contains(where: { $0.value == 0 })
    else {
      return .failure(status: 400, code: "invalid_headers")
    }

    let lines = headerText.components(separatedBy: "\r\n")
    guard let requestLine = lines.first, !requestLine.isEmpty else {
      return .failure(status: 400, code: "invalid_request_line")
    }
    let requestPieces = requestLine.split(separator: " ", omittingEmptySubsequences: false)
    guard requestPieces.count == 3, !requestPieces.contains(where: { $0.isEmpty }),
      requestPieces[2] == "HTTP/1.1"
    else {
      return .failure(status: 400, code: "http_1_1_required")
    }
    let method = String(requestPieces[0])
    guard method.allSatisfy({ $0.isASCII && ($0.isUppercase || !$0.isLetter) }) else {
      return .failure(status: 400, code: "invalid_method")
    }
    let target = String(requestPieces[1])
    guard target.hasPrefix("/"), !target.contains("#"),
      let components = URLComponents(string: target), components.host == nil
    else {
      return .failure(status: 400, code: "invalid_target")
    }

    var headers: [String: String] = [:]
    for line in lines.dropFirst() {
      guard !line.isEmpty, line.first != " ", line.first != "\t",
        let colon = line.firstIndex(of: ":")
      else {
        return .failure(status: 400, code: "invalid_header")
      }
      let rawName = String(line[..<colon])
      guard !rawName.isEmpty,
        rawName.unicodeScalars.allSatisfy({ tokenCharacters.contains($0) })
      else {
        return .failure(status: 400, code: "invalid_header_name")
      }
      let name = rawName.lowercased()
      guard headers[name] == nil else {
        return .failure(status: 400, code: "duplicate_header")
      }
      let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
      guard
        !value.unicodeScalars.contains(where: { scalar in
          scalar.value < 32 && scalar.value != 9
        })
      else {
        return .failure(status: 400, code: "invalid_header_value")
      }
      headers[name] = value
    }
    guard headers["host"]?.isEmpty == false else {
      return .failure(status: 400, code: "host_required")
    }
    guard headers["transfer-encoding"] == nil else {
      return .failure(status: 400, code: "transfer_encoding_not_supported")
    }
    let contentLength: Int
    if let rawLength = headers["content-length"] {
      guard !rawLength.isEmpty, rawLength.allSatisfy(\.isNumber),
        let parsed = Int(rawLength), parsed <= maximumBodyBytes
      else {
        return .failure(
          status: Int(rawLength) == nil ? 400 : 413,
          code: Int(rawLength) == nil ? "invalid_content_length" : "body_too_large"
        )
      }
      contentLength = parsed
    } else {
      contentLength = 0
    }
    let total = headerBytes + contentLength
    if data.count < total { return .needMore(expectedBytes: total) }
    guard data.count == total else {
      return .failure(status: 400, code: "http_pipelining_not_supported")
    }
    let body = data.subdata(in: headerBytes..<total)
    return .request(
      RemoteHTTPRequest(
        method: method,
        path: components.path.isEmpty ? "/" : components.path,
        headers: headers,
        body: body
      )
    )
  }
}

private enum RemoteHTTPWire {
  static func encode(_ response: RemoteHTTPResponse) -> Data {
    switch response.body {
    case .data(let data, let representedLength):
      return encode(response, data: data, representedLength: representedLength)
    case .file(let file):
      return encodeHead(response, contentLength: file.length)
    }
  }

  static func encode(
    _ response: RemoteHTTPResponse,
    data: Data,
    representedLength: Int
  ) -> Data {
    var result = encodeHead(response, contentLength: UInt64(representedLength))
    result.append(data)
    return result
  }

  static func encodeHead(_ response: RemoteHTTPResponse, contentLength: UInt64) -> Data {
    var headers = response.headers
    headers["Connection"] = "close"
    headers["Content-Length"] = String(contentLength)
    headers["X-Content-Type-Options"] = "nosniff"
    headers["Referrer-Policy"] = "no-referrer"
    let reason = reasonPhrase(response.status)
    var head = "HTTP/1.1 \(response.status) \(reason)\r\n"
    for (name, value) in headers.sorted(by: { $0.key < $1.key }) {
      head += "\(name): \(value)\r\n"
    }
    head += "\r\n"
    return Data(head.utf8)
  }

  private static func reasonPhrase(_ status: Int) -> String {
    switch status {
    case 200: return "OK"
    case 201: return "Created"
    case 206: return "Partial Content"
    case 202: return "Accepted"
    case 400: return "Bad Request"
    case 401: return "Unauthorized"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 408: return "Request Timeout"
    case 409: return "Conflict"
    case 413: return "Payload Too Large"
    case 431: return "Request Header Fields Too Large"
    case 416: return "Range Not Satisfiable"
    case 429: return "Too Many Requests"
    case 503: return "Service Unavailable"
    default: return "Error"
    }
  }
}

// MARK: - Embedded remote UI

private enum RemoteControlHTML {
  static let page = #"""
    <!doctype html><html lang="ja"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
    <meta name="color-scheme" content="light dark"><title>mioh remote</title>
    <style>
    :root{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;color-scheme:light dark;--bg:#f5f5f7;--surface:#fff;--group:#f2f2f7;--text:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--accent:#087df1;--danger:#d83a3a;--shadow:#00000018}*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--text)}button,input,select,textarea{font:inherit}button{border:0;border-radius:9px;background:var(--accent);color:#fff;min-height:34px;padding:7px 13px;font-weight:600;cursor:pointer}button.secondary{background:#76767b}button.danger{background:var(--danger)}button.ghost{background:transparent;color:var(--text);border:1px solid var(--line)}button:disabled{opacity:.42;cursor:default}input,select,textarea{min-height:34px;border:1px solid var(--line);border-radius:7px;padding:6px 9px;background:var(--surface);color:var(--text)}input[type=password]{width:100%}input[type=checkbox]{min-height:auto}input[type=range]{flex:1;min-width:180px}.grow{flex:1}.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.meta{font-variant-numeric:tabular-nums;font-size:.82rem;color:var(--muted);overflow-wrap:anywhere}.error{color:#ff453a;min-height:1.2em}.auth-gate{min-height:100vh;display:grid;place-items:center;padding:20px}.auth-card{width:min(520px,100%);background:var(--surface);padding:24px;border-radius:18px;box-shadow:0 12px 44px var(--shadow)}.auth-brand{display:flex;align-items:center;gap:12px;margin-bottom:20px}.appicon{width:38px;height:38px;border-radius:9px;background:linear-gradient(145deg,#fff,#d9d9dd);border:1px solid var(--line);display:grid;place-items:center;color:#666;font-size:.72rem;font-weight:700}.app-shell{min-height:100vh;display:grid;grid-template-rows:auto auto 1fr auto}.app-header{height:68px;padding:0 22px;display:flex;align-items:center;gap:12px;background:var(--surface);border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:11px}.brand h1{font-size:1.08rem;margin:0}.brand-subtitle{font-size:.76rem;color:var(--muted)}.model-name{font-size:.72rem;color:var(--muted);margin-top:2px}.badge{font-size:.78rem;padding:5px 9px;border-radius:999px;background:var(--group)}.tabbar{display:flex;justify-content:center;gap:2px;overflow-x:auto;padding:8px 12px;background:var(--surface);border-bottom:1px solid var(--line);scrollbar-width:none}.tabbar::-webkit-scrollbar{display:none}.tab-button{flex:0 0 auto;min-height:30px;padding:5px 12px;border-radius:9px;background:transparent;color:var(--text);font-size:.82rem;font-weight:600;white-space:nowrap}.tab-button[aria-selected=true]{background:#e7e7ea}.workspace{width:min(980px,100%);margin:auto;padding:16px 16px 96px}.tab-panel{min-height:520px}.settings-group,.card{background:var(--surface);border:1px solid color-mix(in srgb,var(--line) 72%,transparent);border-radius:13px;margin:0 0 14px;padding:16px;box-shadow:0 1px 3px var(--shadow)}.settings-group h2,.card h2{font-size:.95rem;margin:0 0 13px}.formgrid{display:grid;grid-template-columns:minmax(190px,1fr) minmax(250px,1.65fr);gap:10px 16px;align-items:center}.formgrid>label{font-size:.86rem}.formgrid input:not([type=checkbox]),.formgrid select,.formgrid textarea{width:100%}.formgrid textarea{min-height:86px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.pathbox{display:flex;gap:7px;align-items:center}.pathlabel{flex:1;min-width:0;overflow-wrap:anywhere;font-size:.82rem}.field-note{grid-column:1/-1;margin:-3px 0 4px}.settings-actions{position:sticky;bottom:57px;z-index:4;background:color-mix(in srgb,var(--surface) 92%,transparent);backdrop-filter:blur(14px);padding:10px;border:1px solid var(--line);border-radius:12px;box-shadow:0 4px 18px var(--shadow)}video{display:block;width:100%;aspect-ratio:16/9;background:#000;border-radius:9px;margin-bottom:10px}.seek{width:100%}.node{border:1px solid var(--line);border-radius:10px;padding:10px;margin-top:8px}.log-view{min-height:500px;margin:0;white-space:pre-wrap;overflow-wrap:anywhere;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}.app-footer{position:fixed;left:0;right:0;bottom:0;z-index:8;background:color-mix(in srgb,var(--surface) 94%,transparent);backdrop-filter:blur(16px);border-top:1px solid var(--line);padding:max(8px,env(safe-area-inset-bottom)) 18px}.progress-track{height:3px;background:var(--group);border-radius:2px;overflow:hidden;margin-bottom:8px}.progress-bar{height:100%;width:0;background:var(--accent);transition:width .2s}.footer-content{display:flex;gap:10px;align-items:center}.footer-status{flex:1;text-align:center}.ok{color:#28a745}dialog{width:min(720px,calc(100% - 24px));max-height:80vh;border:1px solid var(--line);border-radius:16px;background:var(--surface);color:var(--text);padding:0}.dialoghead{position:sticky;top:0;background:var(--surface);padding:14px;border-bottom:1px solid var(--line)}.filelist{padding:10px}.fileitem{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:center;padding:7px;border-bottom:1px solid var(--line)}
    @media(prefers-color-scheme:dark){:root{--bg:#1c1c1e;--surface:#2c2c2e;--group:#3a3a3c;--text:#f5f5f7;--muted:#aeaeb2;--line:#48484a;--shadow:#0008}.tab-button[aria-selected=true]{background:#48484a}.appicon{background:linear-gradient(145deg,#555,#222);color:#ddd}}
    @media(max-width:620px){.app-header{height:62px;padding:0 12px}.brand-subtitle{display:none}.workspace{padding:12px 10px 100px}.tabbar{justify-content:flex-start}.formgrid{grid-template-columns:1fr}.formgrid>label{margin-top:7px}.formgrid>label:first-child{margin-top:0}.field-note{grid-column:auto}.row:not(.footer-content) button{flex:1}.tab-button{flex:0 0 auto}.fileitem{grid-template-columns:1fr auto}.fileitem .meta{display:none}.settings-actions{position:static;display:grid;grid-template-columns:1fr 1fr;margin-top:12px}.settings-actions .meta{grid-column:1/-1;text-align:center}.footer-status{font-size:.72rem}.app-footer{padding-left:10px;padding-right:10px}}
    </style></head><body>
    <section id="authGate" class="auth-gate"><div class="auth-card"><div class="auth-brand"><div class="appicon">mio</div><div><strong>mioh</strong><div class="brand-subtitle">Motion-Informed Optical Healing</div></div></div><h2>Webリモコンへ接続</h2><div class="row"><div class="grow"><input id="token" type="password" autocomplete="current-password" autocapitalize="characters" spellcheck="false" maxlength="14" placeholder="ABCD-EFGH-JKLM"></div><button id="connect">接続</button></div><p class="meta">Macのmiohに表示された12文字のアクセスコードを入力してください。大文字小文字・ハイフンの有無は問いません。信頼できる家庭・社内LAN専用です。</p><div id="authError" class="error"></div></div></section>
    <main id="appShell" class="app-shell" hidden>
      <header class="app-header"><div class="brand"><div class="appicon">mio</div><div><div><strong>mioh</strong> <span class="brand-subtitle">Motion-Informed Optical Healing</span></div><div id="modelName" class="model-name">設定を読み込み中…</div></div></div><div class="grow"></div><span id="error" class="error"></span><span id="connection" class="badge">未接続</span></header>
      <nav id="tabBar" class="tabbar" role="tablist" aria-label="mioh設定"><button class="tab-button" role="tab" data-tab="basic" aria-controls="tab-basic" aria-selected="true">基本</button><button class="tab-button" role="tab" data-tab="processing" aria-controls="tab-processing" aria-selected="false">分割</button><button class="tab-button" role="tab" data-tab="restoration" aria-controls="tab-restoration" aria-selected="false">復元</button><button class="tab-button" role="tab" data-tab="detection" aria-controls="tab-detection" aria-selected="false">検出</button><button class="tab-button" role="tab" data-tab="output" aria-controls="tab-output" aria-selected="false">出力</button><button class="tab-button" role="tab" data-tab="memory" aria-controls="tab-memory" aria-selected="false">メモリ</button><button class="tab-button" role="tab" data-tab="settings" aria-controls="tab-settings" aria-selected="false">設定</button><button class="tab-button" role="tab" data-tab="playback" aria-controls="tab-playback" aria-selected="false">再生</button><button class="tab-button" role="tab" data-tab="log" aria-controls="tab-log" aria-selected="false">ログ</button></nav>
      <div class="workspace">
        <section id="tab-basic" class="tab-panel" role="tabpanel"><div id="settings-basic"></div></section>
        <section id="tab-processing" class="tab-panel" role="tabpanel" hidden><div id="settings-processing"></div></section>
        <section id="tab-restoration" class="tab-panel" role="tabpanel" hidden><div id="settings-restoration"></div></section>
        <section id="tab-detection" class="tab-panel" role="tabpanel" hidden><div id="settings-detection"></div></section>
        <section id="tab-output" class="tab-panel" role="tabpanel" hidden><div id="settings-output"></div></section>
        <section id="tab-memory" class="tab-panel" role="tabpanel" hidden><div id="settings-memory"></div></section>
        <section id="tab-settings" class="tab-panel" role="tabpanel" hidden><section class="settings-group"><h2>ローカルネットワーク操作</h2><div class="formgrid"><label>状態</label><span id="connectionSettings" class="meta">接続中</span><label>認証</label><button id="changeToken" class="ghost">アクセスコードを変更</button></div></section><section class="settings-group"><h2>ローカル復元クラスタ</h2><div id="clusterForm"></div></section><section class="settings-group"><h2>ユーザーデフォルト</h2><p class="meta">現在の各タブの値を、このMacユーザーのmiohデフォルトとして保存します。</p><div class="row"><button id="saveDefaults">現在の設定をデフォルトに保存</button><button id="loadDefaults" class="secondary">保存済みデフォルトを読み込み</button><button id="resetDefaults" class="danger">初期値に戻す</button></div><p id="settingsState" class="meta"></p></section><section class="settings-group"><h2>保存対象</h2><p class="meta">入力・出力、一時フォルダ、分割、復元、検出、出力、メモリ、再生バッファを保存します。ログ、進捗、実行中状態は保存しません。</p></section></section>
        <section id="tab-playback" class="tab-panel" role="tabpanel" hidden><section class="settings-group"><h2>再生動画</h2><div id="previewSource" class="pathbox"><span class="pathlabel">素材未選択</span><button id="choosePreview" class="ghost">素材を選択</button></div></section><div id="settings-playback"></div><section class="card"><video id="streamVideo" playsinline controls></video><div id="playMeta" class="meta">状態を取得中…</div><input id="seek" class="seek" type="range" min="0" max="1" step="0.1" value="0"><div class="row"><button data-api="playback/play">再生</button><button data-api="playback/pause" class="secondary">一時停止</button><button data-api="playback/toggle" class="secondary">切替</button><button data-api="playback/stop" class="danger">停止</button><label><input id="showOriginal" type="checkbox"> 処理前</label></div><div class="row" style="margin-top:10px"><label>音量</label><input id="volume" type="range" min="0" max="1" step="0.01" value="1"><label><input id="mute" type="checkbox"> ミュート</label></div><div class="row" style="margin-top:12px"><button id="streamStart">映像配信を開始</button><button id="streamReconnect" class="secondary">再接続</button><button id="streamStop" class="danger">配信停止</button><span id="streamMeta" class="meta">停止中</span></div></section></section>
        <section id="tab-log" class="tab-panel" role="tabpanel" hidden><pre id="remoteLog" class="log-view">Webリモコンを接続しています…</pre></section>
        <div class="settings-actions row"><button id="applySettings">設定を反映</button><button id="reloadSettings" class="secondary">再読込</button><span class="meta">処理中は設定を変更できません。</span></div>
      </div>
      <footer class="app-footer"><div class="progress-track"><div id="footerProgress" class="progress-bar"></div></div><div class="footer-content"><button data-api="export/stop" class="danger">停止</button><span id="exportMeta" class="footer-status meta">状態を取得中…</span><button data-api="export/start">開始</button></div></footer>
    </main>
    <dialog id="assetDialog"><div class="dialoghead"><div class="row"><button id="assetParent" class="secondary">上へ</button><strong id="assetLocation" class="grow">場所</strong><button id="assetClose" class="danger">閉じる</button></div><div class="row" style="margin-top:9px"><input id="assetSearch" class="grow" type="search" placeholder="このフォルダを検索"><button id="assetSearchButton" class="ghost">検索</button></div><div id="createDirectoryRow" class="row" style="margin-top:9px"><input id="directoryName" class="grow" placeholder="新しいフォルダ名"><button id="createDirectory" class="ghost">作成</button></div><div class="row" style="margin-top:9px"><button id="assetPrevious" class="secondary">前へ</button><span id="assetPage" class="meta grow"></span><button id="assetNext" class="secondary">次へ</button></div></div><div id="assetFiles" class="filelist"></div></dialog>
    <script>
    const $=id=>document.getElementById(id),token=$('token'),err=$('error'),authErr=$('authError'),badge=$('connection'),streamVideo=$('streamVideo'),appShell=$('appShell'),authGate=$('authGate');let mediaTicket='',config=null,browserTarget=null,browserPurpose='input',browserCurrent=null,browserQuery='',browserOffset=0,lastStatus='';
    token.value=localStorage.getItem('mioh-token')||'';
    function normalizeToken(v){const c=v.toUpperCase().replace(/[\s-]/g,'');return /^[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{12}$/.test(c)?c.match(/.{4}/g).join('-'):v.trim()}
    async function request(path,method='POST',body){const h={Authorization:'Bearer '+token.value},o={method,headers:h};if(body!==undefined){h['Content-Type']='application/json';o.body=JSON.stringify(body)}const r=await fetch('/api/v1/'+path,o);let j={};try{j=await r.json()}catch{}if(!r.ok){const message=j.error||('HTTP '+r.status);if(path!=='status')appendRemoteLog(`${method} ${path}: ${message}`);throw new Error(message)}if(path!=='status'&&path!=='settings')appendRemoteLog(`${method} ${path}: 完了`);return j}
    function format(s){s=Math.max(0,Math.floor(s||0));return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
    function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
    function appendRemoteLog(message){const view=$('remoteLog');if(!view)return;const stamp=new Date().toLocaleTimeString();const previous=view.textContent==='Webリモコンを接続しています…'?'':view.textContent;view.textContent=(previous?previous+'\n':'')+`[${stamp}] ${message}`;view.scrollTop=view.scrollHeight}
    function setConnected(connected){authGate.hidden=connected;appShell.hidden=!connected;if(connected){badge.textContent='接続中';$('connectionSettings').textContent='接続中 / '+location.host;authErr.textContent=''}else{badge.textContent='未接続';setTimeout(()=>token.focus(),0)}}
    const tabOrder=['basic','processing','restoration','detection','output','memory','settings','playback','log'];
    function selectTab(name){if(!tabOrder.includes(name))name='basic';let selected=null;for(const button of document.querySelectorAll('.tab-button')){const active=button.dataset.tab===name;button.setAttribute('aria-selected',String(active));if(active)selected=button}for(const tab of tabOrder)$('tab-'+tab).hidden=tab!==name;localStorage.setItem('mioh-active-tab',name);if(selected&&!appShell.hidden)selected.scrollIntoView({block:'nearest',inline:'center'})}
    const settingTabs=[
      ['basic',[
        ['ファイル',[['inputPath','入力','path','input'],['outputPath','出力','path','directory'],['tempDirectory','一時フォルダ','path','directory'],['ffmpegTempDirectory','FFmpeg一時フォルダ','path','directory'],['ladaTempDirectory','mioh一時フォルダ','path','directory']]],
        ['実行',[['restorationEngine','実行エンジン','select','restorationEngines'],['device','デバイス','select','devices'],['fp16','FP16','bool'],['autoOptimize','自動最適化','bool'],['overwrite','既存結果を上書き','bool']]]
      ]],
      ['processing',[
        ['並列処理',[['parallelWorkers','Python並列数','number',1,16,1],['nativeParallelWorkers','ネイティブ並列数','number',1,3,1],['executor','実行方式','select','executors']]],
        ['セグメント',[['noSplit','分割しない','bool'],['useSegmentCount','分割方法','boolSelect'],['segmentCount','分割数','number',1,128,1],['segmentDuration','長さ（秒）','number',10,3600,10],['mergeEncoder','結合エンコーダー','text'],['deleteSegments','処理済みセグメントを削除','bool'],['keepTemp','一時ファイルを保持','bool'],['forceSplit','強制的に再分割','bool']]]
      ]],
      ['restoration',[
        ['モデル',[['restorationModel','復元モデル','select','restorationModels'],['customRestorationModel','モデルパス','modelValue'],['useMaxClipLength','最大クリップ長を指定','bool'],['maxClipLength','最大クリップ長','number',1,180,1],['useRestoreMaxFrames','復元チャンク数を指定','bool'],['restoreMaxFrames','復元チャンク数','number',-1,180,1],['restoreTemporalOverlap','Temporal overlap','number',0,120,1],['restoreCrossfade','クロスフェードを有効化','bool']]],
        ['合成',[['sharpenStrength','シャープ','range',0,2,0.05],['detailBoost','ディテール','range',0,1,0.05],['blendFeather','境界フェザー','range',0,3,0.05],['textureMix','テクスチャ','range',0,1,0.01],['smoothStrength','スムージング','range',0,1,0.05],['effectUpscale','エフェクト倍率','number',1,4,1]]],
        ['ROIエンハンサー',[['roiEnhancer','方式','select','roiEnhancers'],['roiEnhancerModel','モデル','roiModel'],['roiEnhancerScale','倍率','number',1,8,1],['roiEnhancerStrength','強度','range',0,1,0.05],['roiEnhancerTile','タイル','number',0,1024,32]]]
      ]],
      ['detection',[
        ['検出モデル',[['detectionModel','モデル','select','detectionModels'],['customDetectionModel','モデルパス','modelValue'],['detectionEmptyLookahead','無検出時の判定間隔','number',0,300,1],['detectFaceMosaics','顔モザイクを検出','bool']]]
      ]],
      ['output',[
        ['エンコーダー',[['encodingMode','設定方法','select','encodingModes'],['encodingPreset','プリセット','select','encodingPresets'],['encoder','エンコーダー','text'],['bitrateMultiplier','ビットレート倍率','number',0.1,100,0.1],['mp4FastStart','MP4 Fast Start','bool']]],
        ['FFmpeg詳細設定',[['encoderOptions','追加FFmpegオプション','textarea']]],
        ['品質',[['useQuality','Qualityを指定','bool'],['quality','Quality','number',0,100,1],['useQMin','Qminを指定','bool'],['qmin','Qmin','number',0,51,1],['useQMax','Qmaxを指定','bool'],['qmax','Qmax','number',0,51,1]]],
        ['フレームレート',[['useFPS','FPS変換','bool'],['fps','FPS分子','number',1,120000,1],['fpsDenominator','FPS分母','number',1,1001,1],['preFPSConversion','復元前にFPS変換','bool']]]
      ]],
      ['memory',[
        ['メモリ管理',[['memoryCleanupInterval','掃除間隔','number',1,100,1],['cleanupTriggerGB','掃除開始空き容量（GB）','number',0,1024,0.1],['useMPSMemoryFraction','MPSメモリ比率を指定','bool'],['mpsMemoryFraction','MPSメモリ比率','number',0.01,1,0.01],['logMPSMemory','MPSメモリ統計を記録','bool']]]
      ]],
      ['settings',[
        ['HLS再生',[['previewUseSafariCompatibleHLS','Safari互換通信','bool'],['previewHLSQuality','HLS画質','select','hlsQualities']]]
      ]],
      ['playback',[
        ['再生設定',[['previewRestorationModel','復元モデル','select','previewRestorationModels'],['previewCustomRestorationModel','復元モデルパス','modelValue'],['previewDetectionModel','再生用検出モデル','select','previewDetectionModels'],['previewCustomDetectionModel','検出モデルパス','modelValue'],['previewRealtimeOptimization','リアルタイム最適化','bool'],['previewBufferLimit','バッファ上限（秒）','range',1,60,1]]],
        ['VR表示',[['previewProjectionMode','表示','select','projectionModes'],['previewVideoLayout','形式','select','videoLayouts'],['previewEye','目','select','eyes'],['previewCameraFOV','視野角','range',45,105,1]]]
      ]]
    ];
    function optionsFor(spec){const values=spec[2]==='roiModel'?(config.options.roiEnhancerModels[config.settings.roiEnhancer]||[]):(config.options[spec[3]]||[]),engine=config.settings.restorationEngine||'native';return values.map(x=>{if(typeof x==='string')return{value:x,label:x,available:true,reason:''};const state=x.availabilityByEngine?.[engine]||x;return{value:x.id||x.value,label:x.label||x.id,available:state.available!==false,reason:state.reason||''}})}
    function isCustom(value){return value==='カスタム'||value==='custom'}
    function fieldVisible(key){const s=config.settings,python=s.restorationEngine==='python';switch(key){case'device':case'fp16':case'autoOptimize':case'parallelWorkers':case'executor':case'mergeEncoder':return python;case'nativeParallelWorkers':return !python;case'customRestorationModel':return isCustom(s.restorationModel);case'maxClipLength':return Boolean(s.useMaxClipLength);case'restoreMaxFrames':return Boolean(s.useRestoreMaxFrames);case'roiEnhancerModel':return s.roiEnhancer!=='none';case'customDetectionModel':return isCustom(s.detectionModel);case'encodingPreset':return s.encodingMode==='preset';case'encoder':return s.encodingMode==='custom';case'segmentCount':return Boolean(s.useSegmentCount);case'segmentDuration':return !s.useSegmentCount;case'mpsMemoryFraction':return Boolean(s.useMPSMemoryFraction);case'previewRestorationModel':case'previewDetectionModel':return python;case'previewCustomRestorationModel':return python&&isCustom(s.previewRestorationModel);case'previewCustomDetectionModel':return python&&isCustom(s.previewDetectionModel);default:return true}}
    function fieldDisabled(key){const s=config.settings,python=s.restorationEngine==='python';if(['parallelWorkers','executor','useSegmentCount','segmentCount','segmentDuration','forceSplit'].includes(key)&&s.noSplit)return true;if(['roiEnhancerScale','roiEnhancerStrength','roiEnhancerTile'].includes(key)&&s.roiEnhancer==='none')return true;if(key==='quality'&&!s.useQuality)return true;if(key==='qmin'&&!s.useQMin)return true;if(key==='qmax'&&!s.useQMax)return true;if(['fps','fpsDenominator'].includes(key)&&!s.useFPS)return true;if(key==='preFPSConversion'&&(!s.useFPS||(python&&s.noSplit)))return true;return false}
    const dependencyKeys=new Set(['restorationEngine','restorationModel','detectionModel','useMaxClipLength','useRestoreMaxFrames','roiEnhancer','encodingMode','useSegmentCount','noSplit','useQuality','useQMin','useQMax','useFPS','useMPSMemoryFraction','previewRestorationModel','previewDetectionModel','previewRealtimeOptimization']);
    function changed(key,value){config.settings[key]=value;if(key==='roiEnhancer')config.settings.roiEnhancerModel='';if(dependencyKeys.has(key))renderSettings();$('modelName').textContent=config.settings.restorationModel||'復元モデル未選択'}
    function sectionNote(tab,title){const s=config.settings;if(tab==='basic'&&title==='実行'&&s.restorationEngine!=='python')return'FP16 / Apple Silicon自動最適化で、デコードから書き出しまでを1つのSwiftプロセスで実行します。';if(tab==='processing'&&title==='並列処理'&&s.restorationEngine!=='python')return'Swiftネイティブの自動段階並列を使用します。';if(tab==='processing'&&title==='セグメント'&&s.noSplit)return'元動画をsegmentsへコピーせず、そのまま1本で処理します。';if(tab==='output'&&title==='FFmpeg詳細設定')return'例: -pix_fmt yuv420p10le -profile:v main10 -b:v 20M';if(tab==='playback'&&title==='再生設定'&&s.previewRealtimeOptimization)return'復元は維持し、再生中は合成パラメータとROIエンハンサーを完全バイパスします。';return''}
    function renderSettings(){if(!config)return;for(const [tab,sections] of settingTabs){const root=$('settings-'+tab);root.replaceChildren();for(const [title,fields] of sections){const section=el('section',undefined,'settings-group'),heading=el('h2',title),grid=el('div',undefined,'formgrid');section.append(heading);for(const spec of fields){const [key,label,type]=spec;if(!fieldVisible(key))continue;const labelNode=el('label',label);grid.append(labelNode);let control;if(type==='bool'){control=el('input');control.type='checkbox';control.checked=Boolean(config.settings[key]);control.onchange=()=>changed(key,control.checked)}else if(type==='boolSelect'){control=el('select');for(const pair of [[true,'個数'],[false,'秒数']]){const option=el('option',pair[1]);option.value=String(pair[0]);option.selected=Boolean(config.settings[key])===pair[0];control.append(option)}control.onchange=()=>changed(key,control.value==='true')}else if(type==='path'){control=pathControl(key,spec[3])}else if(type==='modelValue'){control=modelValueControl(key)}else if(type==='range'){const wrap=el('div',undefined,'row'),range=el('input'),value=el('span',String(config.settings[key]??''),'meta');range.type='range';range.min=spec[3];range.max=spec[4];range.step=spec[5];range.value=config.settings[key]??spec[3];range.oninput=()=>{const number=Number(range.value);config.settings[key]=number;value.textContent=String(number)};wrap.append(range,value);control=wrap}else if(type==='select'||type==='roiModel'){control=el('select');let choices=optionsFor(spec);const current=config.settings[key];if(current!==null&&current!==undefined&&!choices.some(x=>x.value===current))choices.push({value:current,label:(config.assetLabels[key]||current),available:true,reason:''});for(const choice of choices){const suffix=choice.available?'':`（${choice.reason||'未導入'}）`,option=el('option',choice.label+suffix);option.value=choice.value;option.selected=choice.value===current;option.disabled=!choice.available&&choice.value!==current;if(choice.reason)option.title=choice.reason;control.append(option)}control.onchange=()=>changed(key,control.value);if(type==='roiModel'){const wrap=el('div',undefined,'pathbox'),browse=el('button','参照','ghost');browse.type='button';browse.onclick=()=>openBrowser(key,'model');wrap.append(control,browse);control=wrap}}else{control=el(type==='textarea'?'textarea':'input');if(type==='number'){control.type='number';control.min=spec[3];control.max=spec[4];control.step=spec[5];control.value=config.settings[key]??'';control.onchange=()=>changed(key,Number(control.value))}else{control.value=config.settings[key]??'';control.oninput=()=>changed(key,control.value)}}if(fieldDisabled(key)){for(const item of control.matches?.('input,select,textarea,button')?[control]:control.querySelectorAll?.('input,select,textarea,button')||[])item.disabled=true}grid.append(control)}const note=sectionNote(tab,title);if(note){const noteNode=el('p',note,'meta field-note');grid.append(noteNode)}section.append(grid);root.append(section)}}}
    function pathControl(key,purpose){const w=el('div',undefined,'pathbox'),label=el('span',config.assetLabels[key]||'未選択','pathlabel'),choose=el('button','参照','ghost'),clear=el('button','解除','secondary');choose.type='button';choose.onclick=()=>openBrowser(key,purpose);clear.type='button';clear.onclick=()=>{config.settings[key]='';config.assetLabels[key]='';renderSettings()};w.append(label,choose,clear);return w}
    function modelValueControl(key){const w=el('div',undefined,'pathbox'),value=config.settings[key]||'',usingAsset=String(value).startsWith('asset-'),input=el('input'),choose=el('button','参照','ghost'),clear=el('button','解除','secondary');input.type='text';input.value=usingAsset?'':value;input.placeholder=usingAsset?(config.assetLabels[key]||'選択済みモデル'):'モデルIDまたはファイルを選択';input.oninput=()=>{config.settings[key]=input.value;config.assetLabels[key]=''};choose.type='button';choose.onclick=()=>openBrowser(key,'model');clear.type='button';clear.onclick=()=>{config.settings[key]='';config.assetLabels[key]='';renderSettings()};w.append(input,choose,clear);return w}
    async function loadSettings(){config=await request('settings','GET');renderSettings();renderPreview();renderCluster();$('showOriginal').checked=Boolean(config.preview.showOriginal);$('settingsState').textContent='読込済み r'+config.revision;$('modelName').textContent=config.settings.restorationModel||'復元モデル未選択';setConnected(true);selectTab(localStorage.getItem('mioh-active-tab')||'basic');appendRemoteLog('Macの設定を読み込みました')}
    async function applySettings(){const next=await request('settings','PATCH',{revision:config.revision,settings:config.settings,showOriginal:$('showOriginal').checked});config=next;renderSettings();renderPreview();renderCluster();$('settingsState').textContent='反映済み r'+config.revision;$('modelName').textContent=config.settings.restorationModel||'復元モデル未選択';appendRemoteLog('全タブの設定をMacへ反映しました')}
    function renderPreview(){const box=$('previewSource'),label=box.querySelector('.pathlabel');label.textContent=config?.preview?.source?.label||'素材未選択'}
    function renderCluster(){
      const root=$('clusterForm');root.replaceChildren();
      if(!config||!config.cluster.available){root.append(el('p','クラスタ機能は利用できません','meta'));return}
      const c=config.cluster,g=el('div',undefined,'formgrid');
      function add(label,node){g.append(el('label',label),node)}
      const role=el('select');
      for(const v of [['off','無効'],['coordinator','Coordinator'],['worker','Worker']]){const o=el('option',v[1]);o.value=v[0];o.selected=c.role===v[0];role.append(o)}
      role.onchange=()=>c.role=role.value;add('役割',role);
      const shared=el('div',undefined,'pathbox');shared.append(el('span',c.sharedRoot?.label||'未選択','pathlabel'));
      const browse=el('button','参照','ghost'),clearRoot=el('button','解除','secondary');browse.type='button';browse.onclick=()=>openBrowser('__clusterRoot','directory');clearRoot.type='button';clearRoot.onclick=()=>{c.sharedRoot={assetID:null,label:null};renderCluster()};shared.append(browse,clearRoot);add('共有ルート',shared);
      for(const spec of [['sharedRootIdentifier','共有ルートID','text'],['shardMinutes','ジョブ長（分）','number'],['useCoordinatorAsWorker','このMacもWorker','bool'],['useForExport','クラスタで書き出す','bool']]){const i=el('input');if(spec[2]==='bool'){i.type='checkbox';i.checked=Boolean(c[spec[0]]);i.onchange=()=>c[spec[0]]=i.checked}else{i.type=spec[2];i.value=c[spec[0]];i.oninput=()=>c[spec[0]]=spec[2]==='number'?Number(i.value):i.value}add(spec[1],i)}
      root.append(g);
      const actions=el('div',undefined,'row'),save=el('button','クラスタ設定を反映');
      save.onclick=async()=>{config.cluster=(await request('cluster/settings','PATCH',{role:c.role,sharedRootAssetID:c.sharedRoot?.assetID||'',sharedRootIdentifier:c.sharedRootIdentifier,shardMinutes:c.shardMinutes,useForExport:c.useForExport,useCoordinatorAsWorker:c.useCoordinatorAsWorker})).cluster;renderCluster()};
      const service=el('button',c.serviceActive?'サービス停止':'サービス開始',c.serviceActive?'danger':'secondary');service.onclick=async()=>{await request(c.serviceActive?'cluster/stop':'cluster/start');await loadSettings()};actions.append(save,service,el('span',c.status,'meta'));root.append(actions);
      for(const n of c.nodes||[]){const card=el('div',undefined,'node'),head=el('div',undefined,'row'),check=el('input');check.type='checkbox';check.checked=n.selected;check.disabled=!n.verified;check.onchange=async()=>{await request('cluster/node/select','POST',{id:n.id,selected:check.checked});await loadSettings()};head.append(check,el('strong',n.name),el('span',n.verified?'接続済み':'確認中','meta'));card.append(head);root.append(card)}
    }
    async function openBrowser(target,purpose){browserTarget=target;browserPurpose=purpose;browserCurrent=null;browserQuery='';browserOffset=0;$('assetSearch').value='';$('createDirectoryRow').style.display=purpose==='directory'?'flex':'none';$('assetDialog').showModal();await listAssets(null,0)}
    async function listAssets(directoryID,offset=0){const data=await request('assets/list','POST',{purpose:browserPurpose,directoryID,offset,query:browserQuery});browserCurrent=data.currentID||null;browserOffset=Number(data.offset||0);$('assetLocation').textContent=data.current||'場所を選択';$('assetParent').disabled=!data.parentID;$('assetParent').onclick=()=>{browserQuery='';browserOffset=0;$('assetSearch').value='';listAssets(data.parentID,0)};$('assetPrevious').disabled=data.previousOffset===null||data.previousOffset===undefined;$('assetPrevious').onclick=()=>listAssets(browserCurrent,Number(data.previousOffset||0));$('assetNext').disabled=data.nextOffset===null||data.nextOffset===undefined;$('assetNext').onclick=()=>listAssets(browserCurrent,Number(data.nextOffset||0));$('assetPage').textContent=data.total?`${browserOffset+1}–${Math.min(browserOffset+data.entries.length,data.total)} / ${data.total}件`:'0件';const list=$('assetFiles');list.replaceChildren();for(const item of data.entries){const row=el('div',undefined,'fileitem'),name=el(item.browseable?'button':'span',item.name,item.browseable?'ghost':'');if(item.browseable)name.onclick=()=>{browserQuery='';browserOffset=0;$('assetSearch').value='';listAssets(item.assetID,0)};row.append(name,el('span',item.kind==='directory'?'フォルダ':'ファイル','meta'));const pick=el('button','選択');pick.disabled=!item.selectable;pick.onclick=()=>selectAsset(item);row.append(pick);list.append(row)}}
    async function selectAsset(item){if(browserTarget==='__preview'){await request('playback/source','POST',{assetID:item.assetID});await loadSettings()}else if(browserTarget==='__clusterRoot'){config.cluster.sharedRoot={assetID:item.assetID,label:item.name};renderCluster()}else{config.settings[browserTarget]=item.assetID;config.assetLabels[browserTarget]=item.name;renderSettings()}$('assetDialog').close()}
    async function refresh(){if(!token.value){setConnected(false);authErr.textContent='アクセスコードを入力してください';return}try{const j=await request('status','GET');setConnected(true);err.textContent='';authErr.textContent='';const p=j.playback,e=j.export,percent=Math.max(0,Math.min(100,Number(e.progress||0)*100));$('playMeta').textContent=`${p.state}  ${format(p.position)} / ${format(p.duration)}  先読み ${Number(p.bufferedSeconds).toFixed(1)}秒`;$('seek').max=Math.max(1,p.duration);if(document.activeElement!==$('seek'))$('seek').value=p.position;$('volume').value=p.volume;$('mute').checked=p.muted;$('exportMeta').textContent=`${e.running?'処理中':'待機中'}  ${percent.toFixed(1)}%  ${e.status}`;$('footerProgress').style.width=percent+'%';const status=`${e.running?'処理中':'待機中'} ${e.status}`;badge.textContent=e.running?`${Math.round(percent)}%`:'待機中';$('connectionSettings').textContent='接続中 / '+location.host;if(status!==lastStatus){lastStatus=status;appendRemoteLog(status)}}catch(e){setConnected(false);authErr.textContent=e.message}}
    async function stopStream(){const old=mediaTicket;mediaTicket='';streamVideo.pause();streamVideo.removeAttribute('src');streamVideo.load();$('streamMeta').textContent='停止中';if(old)try{await request('stream/stop','POST',{ticket:old})}catch(e){err.textContent=e.message}}
    async function startStream(){if(mediaTicket)await stopStream();const s=await request('stream/session');mediaTicket=s.playlist.split('/')[3]||'';streamVideo.src=s.playlist;streamVideo.load();try{await streamVideo.play()}catch{}$('streamMeta').textContent='接続中（音声付きHLS）'}
    $('connect').onclick=async()=>{token.value=normalizeToken(token.value);localStorage.setItem('mioh-token',token.value);authErr.textContent='';try{await refresh();await loadSettings()}catch(e){setConnected(false);authErr.textContent=e.message}};
    $('changeToken').onclick=()=>{setConnected(false);token.select()};
    document.querySelectorAll('.tab-button').forEach(button=>button.onclick=()=>selectTab(button.dataset.tab));
    document.querySelectorAll('[data-api]').forEach(button=>button.onclick=async()=>{try{await request(button.dataset.api);await refresh()}catch(e){err.textContent=e.message}});
    $('applySettings').onclick=()=>applySettings().catch(e=>err.textContent=e.message);
    $('reloadSettings').onclick=()=>loadSettings().catch(e=>err.textContent=e.message);
    $('saveDefaults').onclick=()=>request('defaults/save').then(result=>{$('settingsState').textContent=result.status;appendRemoteLog('現在の設定をデフォルトに保存しました')}).catch(e=>err.textContent=e.message);
    $('loadDefaults').onclick=()=>request('defaults/load').then(()=>loadSettings()).catch(e=>err.textContent=e.message);
    $('resetDefaults').onclick=()=>{if(confirm('全設定を初期値に戻しますか？'))request('defaults/reset').then(()=>loadSettings()).catch(e=>err.textContent=e.message)};
    $('choosePreview').onclick=()=>openBrowser('__preview','preview-input');$('assetClose').onclick=()=>$('assetDialog').close();$('assetSearchButton').onclick=()=>{browserQuery=$('assetSearch').value.trim();listAssets(browserCurrent,0).catch(e=>err.textContent=e.message)};$('assetSearch').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();$('assetSearchButton').click()}};$('createDirectory').onclick=async()=>{if(!browserCurrent)return;await request('assets/create-directory','POST',{parentID:browserCurrent,name:$('directoryName').value});$('directoryName').value='';await listAssets(browserCurrent,browserOffset)};
    $('streamStart').onclick=()=>startStream().catch(e=>err.textContent=e.message);$('streamReconnect').onclick=()=>startStream().catch(e=>err.textContent=e.message);$('streamStop').onclick=stopStream;$('seek').onchange=e=>request('playback/seek','POST',{seconds:Number(e.target.value)}).catch(x=>err.textContent=x.message);$('volume').onchange=e=>request('playback/volume','POST',{volume:Number(e.target.value)}).catch(x=>err.textContent=x.message);$('mute').onchange=e=>request('playback/mute','POST',{muted:e.target.checked}).catch(x=>err.textContent=x.message);$('showOriginal').onchange=e=>request('playback/original','POST',{enabled:e.target.checked}).catch(x=>err.textContent=x.message);
    selectTab(localStorage.getItem('mioh-active-tab')||'basic');setInterval(refresh,1000);if(token.value){refresh().then(loadSettings).catch(e=>{setConnected(false);authErr.textContent=e.message})}else setConnected(false)
    </script></body></html>
    """#
}
