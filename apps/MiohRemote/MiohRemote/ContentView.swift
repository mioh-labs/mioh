import AVKit
import MiohRemoteKit
import SwiftUI
import WebKit

struct ContentView: View {
  var body: some View {
    Group {
      if UIDevice.current.userInterfaceIdiom == .pad {
        IPadStandaloneView()
      } else {
        RemoteControlRoot()
      }
    }
  }
}

struct FullRemoteControlView: View {
  @EnvironmentObject private var store: RemoteStore

  var body: some View {
    NavigationStack {
      Group {
        if store.connected, let url = store.fullControlURL {
          MiohWebControlView(url: url, accessToken: store.token)
        } else {
          VStack(spacing: 12) {
            Image(systemName: "desktopcomputer")
              .font(.largeTitle)
            Text("Macへ接続してください").font(.headline)
            Text(
              "iPad版の「設定」からMac版miohへ接続すると、Mac側の設定・ログ・再生画面を利用できます。"
            )
              .font(.caption)
              .foregroundStyle(.secondary)
              .multilineTextAlignment(.center)
          }
          .padding()
        }
      }
      .navigationTitle("mioh フル機能")
      .navigationBarTitleDisplayMode(.inline)
    }
  }
}

private struct MiohWebControlView: UIViewRepresentable {
  let url: URL
  let accessToken: String

  final class Coordinator {
    var loadedKey: String?
  }

  func makeCoordinator() -> Coordinator { Coordinator() }

  func makeUIView(context: Context) -> WKWebView {
    let configuration = WKWebViewConfiguration()
    configuration.allowsInlineMediaPlayback = true
    configuration.mediaTypesRequiringUserActionForPlayback = []
    let webView = WKWebView(frame: .zero, configuration: configuration)
    webView.allowsBackForwardNavigationGestures = true
    load(webView, context: context)
    return webView
  }

  func updateUIView(_ webView: WKWebView, context: Context) {
    load(webView, context: context)
  }

  private func load(_ webView: WKWebView, context: Context) {
    let key = "\(url.absoluteString)|\(accessToken)"
    guard context.coordinator.loadedKey != key else { return }
    context.coordinator.loadedKey = key
    webView.configuration.userContentController.removeAllUserScripts()
    let tokenLiteral = Self.javaScriptString(accessToken)
    webView.configuration.userContentController.addUserScript(
      WKUserScript(
        source: "localStorage.setItem('mioh-token', \(tokenLiteral));",
        injectionTime: .atDocumentStart,
        forMainFrameOnly: true
      )
    )
    webView.load(URLRequest(url: url))
  }

  private static func javaScriptString(_ value: String) -> String {
    guard let data = try? JSONEncoder().encode(value),
      let literal = String(data: data, encoding: .utf8)
    else { return "\"\"" }
    return literal
  }
}

struct RemoteControlRoot: View {
  @EnvironmentObject private var store: RemoteStore
  @StateObject private var discovery = BonjourDiscovery()

  var body: some View {
    NavigationStack {
      Group {
        if store.connected { RemoteDashboard() }
        else { ConnectionView(discovery: discovery) }
      }
      .navigationTitle("mioh Remote")
      .navigationBarTitleDisplayMode(.inline)
    }
    .onAppear { discovery.start() }
    .onDisappear { discovery.stop() }
  }
}

struct IPadWorkerView: View {
  @EnvironmentObject private var worker: IPadWorkerStore
  @State private var choosingRoot = false

  var body: some View {
    NavigationStack {
      Form {
        Section("状態") {
          LabeledContent("Worker", value: worker.stateLabel)
          preparationRow
          Text("iPad Workerはアプリが前景にある間だけ動作します。画面ロックやアプリ切替で安全に停止します。")
            .font(.caption)
            .foregroundStyle(.secondary)
          if let message = worker.lifecycleMessage {
            Text(message).font(.caption).foregroundStyle(.orange)
          }
        }

        Section {
          Label(
            "信頼できるローカルLAN専用です。TLSがないため、インターネットへ公開しないでください。",
            systemImage: "exclamationmark.triangle.fill"
          )
          .font(.caption)
          .foregroundStyle(.orange)
        }

        Section("このiPad") {
          TextField("表示名", text: $worker.displayName)
            .disabled(worker.isRunning)
          LabeledContent("Node ID", value: worker.nodeID.uuidString.lowercased())
            .font(.caption)
            .textSelection(.enabled)
        }

        Section {
          TextField("Macと同じ共有ルートID（予備経路）", text: $worker.sharedRootIdentifier)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .disabled(worker.isRunning)
          LabeledContent(
            "フォルダ",
            value: worker.sharedRootURL?.lastPathComponent ?? "未選択"
          )
          HStack {
            Button("Filesで選択") { choosingRoot = true }
              .disabled(worker.isRunning)
            Spacer()
            Button("解除", role: .destructive) { worker.clearSharedRoot() }
              .disabled(worker.isRunning || worker.sharedRootURL == nil)
          }
        } header: {
          Text("共有メディア")
        } footer: {
          Text("通常はMacが専用HTTP Rangeで必要な範囲だけ配信するため設定不要です。SMB共有を予備経路として使う場合だけ、同じフォルダと共有ルートIDを設定します。モデルは常にアプリ同梱の検証済み資産を使用します。")
        }

        Section("対応範囲") {
          Label("MP4 / MOV / M4V", systemImage: "film")
          Label("BasicVSR++ T18 / T36 / T90 / 可変長", systemImage: "cpu")
          Label("検出モデル v2 / v3.1 / v4 / VR-v2", systemImage: "viewfinder")
          Label("同時ジョブ 1件・最大90フレーム", systemImage: "rectangle.stack")
          Text("ROIエンハンサー、復元後エフェクト、MKV入力には対応しません。音声はMac側が最終出力へ結合します。")
            .font(.caption)
            .foregroundStyle(.secondary)
        }

        if let blocker = worker.startBlocker, !worker.isRunning {
          Section { Text(blocker).foregroundStyle(.orange) }
        }
        if let error = worker.errorMessage {
          Section { Text(error).foregroundStyle(.red) }
        }

        Section {
          Button {
            if worker.isRunning {
              Task { await worker.stop() }
            } else {
              worker.start()
            }
          } label: {
            Label(
              worker.isRunning ? "Workerを停止" : "Workerを開始",
              systemImage: worker.isRunning ? "stop.fill" : "play.fill"
            )
            .frame(maxWidth: .infinity)
          }
          .buttonStyle(.borderedProminent)
          .tint(worker.isRunning ? .red : .blue)
          .disabled(!worker.isRunning && worker.startBlocker != nil)

          Button("同梱モデルを再検証") {
            Task { await worker.prepareModels() }
          }
          .disabled(worker.isRunning || worker.preparation == .validating)
        }

        if !worker.attempts.isEmpty {
          Section("最近のジョブ") {
            ForEach(worker.attempts.reversed()) { attempt in
              VStack(alignment: .leading, spacing: 3) {
                HStack {
                  Text(attempt.state.rawValue)
                  Spacer()
                  Text(attempt.updatedAt, style: .time)
                }
                Text(attempt.outputRelativePath.rawValue)
                  .font(.caption)
                  .foregroundStyle(.secondary)
                  .lineLimit(2)
                if let reason = attempt.failureCode {
                  Text(reason).font(.caption).foregroundStyle(.red)
                }
              }
            }
          }
        }
      }
      .navigationTitle("iPad Worker")
      .navigationBarTitleDisplayMode(.inline)
      .fileImporter(
        isPresented: $choosingRoot,
        allowedContentTypes: [.folder],
        allowsMultipleSelection: false
      ) { result in
        switch result {
        case .success(let urls):
          if let url = urls.first { worker.selectSharedRoot(url) }
        case .failure:
          break
        }
      }
      .task {
        if worker.preparation == .idle { await worker.prepareModels() }
      }
    }
  }

  @ViewBuilder
  private var preparationRow: some View {
    switch worker.preparation {
    case .idle:
      LabeledContent("モデル", value: "未検証")
    case .validating:
      HStack { ProgressView(); Text("同梱モデルを検証中…") }
    case .ready(let restorations, let detectors, let maximumFrames):
      VStack(alignment: .leading, spacing: 2) {
        Label("モデル検証済み", systemImage: "checkmark.seal.fill")
          .foregroundStyle(.green)
        Text("復元 \(restorations.count)種・検出 \(detectors.count)種・最大\(maximumFrames)フレーム")
          .font(.caption)
          .foregroundStyle(.secondary)
        DisclosureGroup("モデル一覧") {
          ForEach(restorations, id: \.self) { Text($0).font(.caption) }
          ForEach(detectors, id: \.self) { Text($0).font(.caption) }
        }
      }
    case .failed(let reason):
      VStack(alignment: .leading, spacing: 2) {
        Label("モデル検証失敗", systemImage: "xmark.octagon.fill")
          .foregroundStyle(.red)
        Text(reason).font(.caption)
      }
    }
  }
}

private struct ConnectionView: View {
  @EnvironmentObject private var store: RemoteStore
  @ObservedObject var discovery: BonjourDiscovery
  @FocusState private var accessCodeFocused: Bool
  @State private var automaticConnectionAttempts: Set<String> = []

  var body: some View {
    Form {
      Section("ローカルネットワーク") {
        if discovery.endpoints.isEmpty {
          HStack {
            ProgressView()
            Text("miohを検索中…")
          }
        } else {
          ForEach(discovery.endpoints) { endpoint in
            Button {
              select(endpoint)
            } label: {
              HStack {
                Image(systemName: "desktopcomputer")
                VStack(alignment: .leading) {
                  Text(endpoint.name)
                  Text(
                    store.hasSavedCredentials(for: endpoint)
                      ? "保存済み・タップして接続"
                      : "タップして初回コードを入力"
                  )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right")
                  .font(.caption)
                  .foregroundStyle(.tertiary)
              }
            }
            .disabled(store.busy)
          }
        }
        if let message = discovery.errorMessage {
          Text(message).foregroundStyle(.red).font(.caption)
        }
        Label(
          "信頼できるローカルLAN専用です。TLSがないため、インターネットへ公開しないでください。",
          systemImage: "exclamationmark.triangle.fill"
        )
        .font(.caption)
        .foregroundStyle(.orange)
      }

      Section {
        if let name = store.selectedServerName {
          Label(name, systemImage: "desktopcomputer")
        }
        SecureField("アクセスコード", text: $store.token)
          .textInputAutocapitalization(.characters)
          .autocorrectionDisabled()
          .focused($accessCodeFocused)
        Button {
          Task { await store.connect() }
        } label: {
          HStack {
            if store.busy { ProgressView() }
            Text("接続")
          }
          .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .disabled(!store.canConnect)

        DisclosureGroup("手動接続") {
          TextField("mioh.local:8888", text: $store.serverAddress)
            .textInputAutocapitalization(.never)
            .keyboardType(.URL)
            .autocorrectionDisabled()
        }
      } header: {
        Text("接続")
      } footer: {
        Text("初回だけMacに表示されたコードを入力します。次回からはMacを自動検出して接続します。コードはこの端末のKeychainに保存されます。")
      }

      if let error = store.errorMessage {
        Section { Text(error).foregroundStyle(.red) }
      }
    }
    .onAppear {
      attemptAutomaticConnection(discovery.endpoints)
    }
    .onChange(of: discovery.endpoints) { endpoints in
      attemptAutomaticConnection(endpoints)
    }
    .onChange(of: store.token) { _ in
      guard store.accessTokenIsComplete, store.fullControlURL != nil,
        !store.busy, !store.connected
      else { return }
      Task { await store.connect() }
    }
  }

  private func select(_ endpoint: MiohServerEndpoint) {
    if store.select(endpoint) {
      Task { await store.connect() }
    } else {
      accessCodeFocused = true
    }
  }

  private func attemptAutomaticConnection(_ endpoints: [MiohServerEndpoint]) {
    guard endpoints.count == 1, let endpoint = endpoints.first,
      !automaticConnectionAttempts.contains(endpoint.id),
      !store.connected, !store.busy,
      store.hasSavedCredentials(for: endpoint)
    else { return }
    automaticConnectionAttempts.insert(endpoint.id)
    Task { await store.connect(to: endpoint) }
  }
}

private struct RemoteDashboard: View {
  @EnvironmentObject private var store: RemoteStore
  @State private var seekValue = 0.0
  @State private var editingSeek = false
  @State private var volumeValue = 1.0

  var body: some View {
    ScrollView {
      VStack(spacing: 16) {
        VideoPlayer(player: store.player)
          .aspectRatio(16 / 9, contentMode: .fit)
          .background(.black)
          .clipShape(RoundedRectangle(cornerRadius: 12))
          .overlay {
            if store.streamURL == nil {
              VStack(spacing: 8) {
                Image(systemName: "play.rectangle")
                  .font(.largeTitle)
                Text("ストリーム未接続").font(.headline)
                Text("Macで復元プレビューを開始してから接続します。")
                  .font(.caption)
              }
              .foregroundColor(.white)
            }
          }

        if let status = store.status {
          playbackControls(status)
          exportControls(status)
        }

        if let error = store.errorMessage {
          Text(error)
            .frame(maxWidth: .infinity, alignment: .leading)
            .foregroundStyle(.red)
            .font(.callout)
        }
      }
      .padding()
    }
    .toolbar {
      ToolbarItem(placement: .topBarLeading) {
        Button("切断") { Task { await store.disconnect() } }
      }
      ToolbarItem(placement: .topBarTrailing) {
        if store.busy { ProgressView() }
        else { Button { Task { await store.refresh() } } label: { Image(systemName: "arrow.clockwise") } }
      }
    }
    .onChange(of: store.status?.playback.position) { position in
      guard !editingSeek, let position else { return }
      seekValue = position
    }
    .onChange(of: store.status?.playback.volume) { volume in
      guard let volume else { return }
      volumeValue = volume
    }
  }

  @ViewBuilder
  private func playbackControls(_ status: MiohStatus) -> some View {
    GroupBox("再生") {
      VStack(spacing: 12) {
        HStack {
          Label(status.playback.state, systemImage: "waveform")
          Spacer()
          Text(time(seekValue) + " / " + time(status.playback.duration))
            .monospacedDigit()
        }

        Slider(
          value: $seekValue,
          in: 0...max(status.playback.duration, 1),
          onEditingChanged: { editing in
            editingSeek = editing
            if !editing { Task { await store.seek(to: seekValue) } }
          }
        )

        HStack(spacing: 18) {
          Button { Task { await store.play() } } label: { Label("再生", systemImage: "play.fill") }
          Button { Task { await store.pause() } } label: { Label("一時停止", systemImage: "pause.fill") }
          Button { Task { await store.stopPlayback() } } label: { Label("停止", systemImage: "stop.fill") }
        }
        .buttonStyle(.bordered)

        HStack {
          Image(systemName: status.playback.muted ? "speaker.slash.fill" : "speaker.wave.2.fill")
          Slider(value: $volumeValue, in: 0...1) { editing in
            if !editing { Task { await store.setVolume(volumeValue) } }
          }
          Toggle("消音", isOn: Binding(
            get: { status.playback.muted },
            set: { value in Task { await store.setMuted(value) } }
          ))
          .labelsHidden()
        }

        HStack {
          Button {
            Task { await store.startStream() }
          } label: {
            Label(store.streamURL == nil ? "映像へ接続" : "再接続", systemImage: "dot.radiowaves.left.and.right")
          }
          .buttonStyle(.borderedProminent)
          Button("映像を切断", role: .destructive) { Task { await store.stopStream() } }
            .disabled(store.streamURL == nil)
        }
      }
    }
  }

  @ViewBuilder
  private func exportControls(_ status: MiohStatus) -> some View {
    GroupBox("書き出し") {
      VStack(alignment: .leading, spacing: 10) {
        Text(status.export.status)
        ProgressView(value: min(max(status.export.progress, 0), 1))
        if let input = status.export.input { LabeledContent("入力", value: input) }
        if let output = status.export.output { LabeledContent("出力", value: output) }
        HStack {
          Button { Task { await store.startExport() } } label: {
            Label("開始", systemImage: "play.circle")
          }
          .disabled(status.export.running)
          Button(role: .destructive) { Task { await store.stopExport() } } label: {
            Label("停止", systemImage: "stop.circle")
          }
          .disabled(!status.export.running)
        }
        .buttonStyle(.bordered)
        Text("入力・出力と復元設定はMac版miohで選択します。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      .frame(maxWidth: .infinity, alignment: .leading)
    }
  }

  private func time(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds >= 0 else { return "00:00" }
    let total = Int(seconds.rounded(.down))
    let hours = total / 3600
    let minutes = (total % 3600) / 60
    let remainder = total % 60
    return hours > 0
      ? String(format: "%d:%02d:%02d", hours, minutes, remainder)
      : String(format: "%02d:%02d", minutes, remainder)
  }
}
