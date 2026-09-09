import AVKit
import Combine
import MiohSFTPKit
import SwiftUI
import UniformTypeIdentifiers

struct IPadStandaloneView: View {
  private static let windowControlsLeadingInset: CGFloat = 72
  private static let selectedTabDefaultsKey =
    "mioh.ipad.browser.selected-tab.v1"
  private static let currentPageDefaultsKey =
    "mioh.ipad.browser.current-page.v1"
  private static let addressInputDefaultsKey =
    "mioh.ipad.browser.address-input.v1"
  private static let mediaInputDefaultsKey =
    "mioh.ipad.browser.media-input.v1"
  private static let resumeAnalysisDefaultsKey =
    "mioh.ipad.browser.resume-analysis.v1"

  private enum WorkspaceTab: String, CaseIterable, Identifiable {
    case basic
    case browser
    case split
    case restoration
    case detection
    case output
    case memory
    case settings
    case playback
    case log

    var id: String { rawValue }

    var title: String {
      switch self {
      case .basic: "基本"
      case .browser: "ブラウザ"
      case .split: "分割"
      case .restoration: "復元"
      case .detection: "検出"
      case .output: "出力"
      case .memory: "メモリ"
      case .settings: "設定"
      case .playback: "再生"
      case .log: "ログ"
      }
    }

    var icon: String {
      switch self {
      case .basic: "slider.horizontal.3"
      case .browser: "globe"
      case .split: "square.split.2x1"
      case .restoration: "wand.and.stars"
      case .detection: "viewfinder"
      case .output: "video"
      case .memory: "memorychip"
      case .settings: "gearshape"
      case .playback: "play.rectangle"
      case .log: "terminal"
      }
    }
  }

  private enum ActiveSheet: String, Identifiable {
    case remote
    case fullRemote
    case worker
    case share
    case sftpInput
    case sftpUpload

    var id: String { rawValue }
  }

  private enum BrowserLibrarySection: String, CaseIterable, Identifiable {
    case history
    case bookmarks

    var id: String { rawValue }

    var title: String {
      switch self {
      case .history: "履歴"
      case .bookmarks: "ブックマーク"
      }
    }

    var icon: String {
      switch self {
      case .history: "clock"
      case .bookmarks: "star"
      }
    }
  }

  private struct BrowserLibrarySheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var store: IPadBrowserLibraryStore
    @State private var selectedSection: BrowserLibrarySection
    @State private var confirmingClear = false
    let openEntry: (IPadBrowserLibraryStore.Entry) -> Void

    init(
      store: IPadBrowserLibraryStore,
      initialSection: BrowserLibrarySection,
      openEntry: @escaping (IPadBrowserLibraryStore.Entry) -> Void
    ) {
      self.store = store
      _selectedSection = State(initialValue: initialSection)
      self.openEntry = openEntry
    }

    private var entries: [IPadBrowserLibraryStore.Entry] {
      switch selectedSection {
      case .history: store.history
      case .bookmarks: store.bookmarks
      }
    }

    var body: some View {
      NavigationStack {
        VStack(spacing: 0) {
          Picker("表示", selection: $selectedSection) {
            ForEach(BrowserLibrarySection.allCases) { section in
              Label(section.title, systemImage: section.icon).tag(section)
            }
          }
          .pickerStyle(.segmented)
          .padding(.horizontal, 16)
          .padding(.vertical, 10)

          Divider()

          if entries.isEmpty {
            VStack(spacing: 12) {
              Spacer()
              Image(systemName: selectedSection.icon)
                .font(.system(size: 36))
                .foregroundStyle(.secondary)
              Text(selectedSection.title + "はまだありません")
                .font(.headline)
              Text(
                selectedSection == .history
                  ? "表示を完了したページがここに追加されます。"
                  : "ブラウザ上部の星を押すと追加できます。"
              )
              .font(.subheadline)
              .foregroundStyle(.secondary)
              Spacer()
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
          } else {
            List {
              ForEach(entries) { entry in
                Button {
                  dismiss()
                  openEntry(entry)
                } label: {
                  HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 5) {
                      Text(entry.title)
                        .font(.body.weight(.medium))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                      Text(entry.url)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                      Text(
                        entry.date.formatted(
                          date: .abbreviated,
                          time: .shortened
                        )
                      )
                      .font(.caption2)
                      .foregroundStyle(.tertiary)
                    }
                    Spacer(minLength: 8)
                    Image(systemName: "chevron.forward")
                      .font(.caption.weight(.semibold))
                      .foregroundStyle(.tertiary)
                  }
                  .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .swipeActions {
                  Button(role: .destructive) {
                    remove(entry)
                  } label: {
                    Label("削除", systemImage: "trash")
                  }
                }
              }
            }
            .listStyle(.plain)
          }
        }
        .navigationTitle(selectedSection.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .cancellationAction) {
            Button("すべて削除", role: .destructive) {
              confirmingClear = true
            }
            .disabled(entries.isEmpty)
          }
          ToolbarItem(placement: .confirmationAction) {
            Button("閉じる") { dismiss() }
          }
        }
        .confirmationDialog(
          selectedSection.title + "をすべて削除しますか？",
          isPresented: $confirmingClear,
          titleVisibility: .visible
        ) {
          Button("すべて削除", role: .destructive) { clearSelectedSection() }
          Button("キャンセル", role: .cancel) {}
        }
      }
    }

    private func remove(_ entry: IPadBrowserLibraryStore.Entry) {
      switch selectedSection {
      case .history: store.removeHistory(id: entry.id)
      case .bookmarks: store.removeBookmark(id: entry.id)
      }
    }

    private func clearSelectedSection() {
      switch selectedSection {
      case .history: store.clearHistory()
      case .bookmarks: store.clearBookmarks()
      }
    }
  }

  @EnvironmentObject private var worker: IPadWorkerStore
  @EnvironmentObject private var remote: RemoteStore
  @Environment(\.scenePhase) private var scenePhase
  @Environment(\.openURL) private var openURL
  @StateObject private var store = IPadStandaloneStore()
  @StateObject private var realtimePlayer = IPadRealtimePreviewController()
  @StateObject private var interactiveBrowser = IPadInteractiveMediaBrowser()
  @StateObject private var browserLibrary = IPadBrowserLibraryStore()
  @StateObject private var sftp = IPadSFTPStore()
  @State private var selectedTab: WorkspaceTab
  @State private var choosingInput = false
  @State private var mediaURLText: String
  @State private var autoStartURLPlayback = false
  @State private var urlAnalysisTask: Task<Void, Never>?
  @State private var urlAnalysisGeneration = 0
  @State private var sftpStreamingSelectionTask: Task<Void, Never>?
  @State private var sftpStreamingSelectionGeneration = 0
  @State private var browserAnalysisTask: Task<Void, Never>?
  @State private var browserAnalysisGeneration = 0
  @State private var browserAnalysisResumeTask: Task<Void, Never>?
  @State private var browserPlaybackTransitionInProgress = false
  // The analysis task owns this lease only until it is transferred to the
  // realtime player. Keeping it outside the task lets cancellation mark the
  // lease as ending synchronously, so an immediate retry waits instead of
  // being rejected as a second active HLS handoff.
  @State private var browserAnalysisHandoffLease:
    IPadBrowserMediaHandoffLease?
  @State private var browserHandoffLease: IPadBrowserMediaHandoffLease?
  @State private var browserHLSResourceLoader:
    (any IPadHLSResourceLoading)?
  @State private var pendingBrowserAnalysisResume: Bool
  @State private var browserFailureMessage: String?
  @State private var showingBrowserCandidates = false
  @State private var showingBrowserLibrary: BrowserLibrarySection?
  @State private var activeSheet: ActiveSheet?
  @State private var player: AVPlayer?
  @State private var playbackVolume = 1.0
  @State private var playbackMuted = false
  @State private var playbackPosition = 0.0
  @State private var playbackDuration = 0.0
  @State private var editingPlaybackPosition = false
  @State private var realtimeSeekPosition = 0.0
  @State private var editingRealtimePosition = false
  @State private var showingFullscreenPlayback = false
  @State private var fullscreenControlsVisible = true
  @State private var fullscreenControlsHideTask: Task<Void, Never>?
  @FocusState private var browserAddressFocused: Bool

  private let playbackTimer = Timer.publish(
    every: 0.25,
    on: .main,
    in: .common
  ).autoconnect()

  private let restorationModels = [
    "basicvsrpp-v1.2-coreai-variable",
    "basicvsrpp-v1.2-coreai",
    "basicvsrpp-v1.2-coreai-t36",
    "basicvsrpp-v1.2-coreai-t90",
  ]
  private let detectorModels = [
    "v4-fast-coreml",
    "v4-accurate-coreml",
    "v3.1-fast-coreml",
    "v3.1-accurate-coreml",
    "v2-coreml",
    "vr-v2-accurate-coreml",
  ]

  init() {
    let arguments = ProcessInfo.processInfo.arguments
    let defaults = UserDefaults.standard
    let requestedTab: WorkspaceTab? =
      arguments.contains("-miohBrowser")
      ? .browser
      : (arguments.contains("-miohPlayback")
        ? .playback
        : (arguments.contains("-miohStandalone") ? .restoration : nil))
    let restoredTab = defaults.string(
      forKey: Self.selectedTabDefaultsKey
    ).flatMap(WorkspaceTab.init(rawValue:)) ?? .basic
    _selectedTab = State(
      initialValue: requestedTab ?? restoredTab
    )
    _mediaURLText = State(
      initialValue: defaults.string(forKey: Self.mediaInputDefaultsKey) ?? ""
    )
    _pendingBrowserAnalysisResume = State(
      initialValue: defaults.bool(forKey: Self.resumeAnalysisDefaultsKey)
    )
    #if DEBUG
      _showingFullscreenPlayback = State(
        initialValue: arguments.contains("-miohFullscreenPlayback")
      )
    #endif
  }

  var body: some View {
    workspace
    .fileImporter(
      isPresented: $choosingInput,
      allowedContentTypes: [.movie],
      allowsMultipleSelection: false
    ) { result in
      guard case .success(let urls) = result, let url = urls.first else {
        return
      }
      autoStartURLPlayback = false
      invalidateURLAnalysis()
      invalidateBrowserAnalysis(closePage: true)
      realtimePlayer.stop()
      player?.pause()
      Task { await store.selectInput(url) }
    }
    .sheet(
      item: $activeSheet,
      onDismiss: { cancelPendingSFTPStreamingSelection() }
    ) { sheet in
      switch sheet {
      case .remote:
        RemoteControlRoot()
      case .fullRemote:
        FullRemoteControlView()
      case .worker:
        IPadWorkerView()
      case .share:
        if let outputURL = store.outputURL {
          ActivityView(items: [outputURL])
        }
      case .sftpInput:
        IPadSFTPBrowserView(
          store: sftp,
          mode: .selectInput,
          onDownloaded: { url in
            autoStartURLPlayback = false
            invalidateURLAnalysis()
            invalidateBrowserAnalysis(closePage: true)
            realtimePlayer.stop()
            player?.pause()
            Task { await store.selectPersistentDownloadedInput(url) }
          },
          onStreamingRequested: { entry in
            startSFTPStreamingPlayback(entry)
          },
          streamingUnavailableReason: sftpStreamingUnavailableReason
        )
      case .sftpUpload:
        if let outputURL = store.outputURL {
          IPadSFTPBrowserView(
            store: sftp,
            mode: .upload(outputURL: outputURL)
          )
        }
      }
    }
    .sheet(isPresented: $showingBrowserCandidates) {
      browserCandidateList
    }
    .sheet(item: $showingBrowserLibrary) { section in
      browserLibrarySheet(section)
    }
    .fullScreenCover(isPresented: $showingFullscreenPlayback) {
      fullscreenPlaybackView
    }
    .onChange(of: store.restorationModelIdentifier) { _ in
      store.normalizeClipSettings()
    }
    .onChange(of: store.clipLength) { _ in
      store.normalizeClipSettings()
    }
    .onChange(of: worker.preparation) { _ in
      tryAutoStartURLPlayback()
    }
    .onChange(of: store.outputURL) { outputURL in
      player?.pause()
      player = outputURL.map { AVPlayer(url: $0) }
      player?.volume = Float(playbackVolume)
      player?.isMuted = playbackMuted
      playbackPosition = 0
      playbackDuration = 0
    }
    .onChange(of: playbackVolume) { value in
      player?.volume = Float(value)
    }
    .onChange(of: playbackMuted) { value in
      player?.isMuted = value
    }
    .onChange(of: realtimePlayer.position) { value in
      if !editingRealtimePosition { realtimeSeekPosition = value }
    }
    .onChange(of: realtimePlayer.interactionRequiredURL) { interactionURL in
      guard let interactionURL, scenePhase == .active else { return }
      realtimePlayer.clearInteractionRequirement()
      openBrowserForInteraction(interactionURL.absoluteString)
      browserFailureMessage =
        "配信側の確認が再び必要です。チェックボックスは必要な場合だけ表示されます。完了後に配信を解析してください。"
    }
    .onChange(of: interactiveBrowser.challengeActive) { challengeActive in
      guard selectedTab == .browser else { return }
      if challengeActive {
        if autoStartURLPlayback || browserAnalysisTask != nil
          || pendingBrowserAnalysisResume
        {
          pendingBrowserAnalysisResume = true
        }
        invalidateBrowserAnalysis(preservingResumeIntent: true)
        browserFailureMessage =
          "Cloudflareの確認待ちです。チェックボックスは必要な場合だけ表示されます。"
      } else {
        resumePendingBrowserAnalysisIfReady()
      }
    }
    .onChange(of: interactiveBrowser.isLoading) { isLoading in
      guard selectedTab == .browser else { return }
      if isLoading {
        let shouldResume =
          pendingBrowserAnalysisResume
          || autoStartURLPlayback || browserAnalysisTask != nil
        guard shouldResume else { return }
        pendingBrowserAnalysisResume = true
        invalidateBrowserAnalysis(preservingResumeIntent: true)
        browserFailureMessage =
          "ページ内プレイヤーの読み込み後も本編HLSの監視を続けます…"
      } else {
        resumePendingBrowserAnalysisIfReady()
      }
    }
    .onChange(of: interactiveBrowser.successfulPageVisit) { visit in
      guard let visit else { return }
      browserLibrary.recordVisit(url: visit.url, title: visit.title)
      persistBrowserPageVisit(visit.url)
    }
    .onChange(of: interactiveBrowser.navigationGeneration) { _ in
      guard selectedTab == .browser else { return }
      let shouldResumeAnalysis =
        pendingBrowserAnalysisResume
        || autoStartURLPlayback || browserAnalysisTask != nil
      browserFailureMessage = nil
      pendingBrowserAnalysisResume = shouldResumeAnalysis
      invalidateBrowserAnalysis(
        preservingResumeIntent: shouldResumeAnalysis
      )
      if shouldResumeAnalysis {
        browserFailureMessage = "ページ内の切り替え後も本編URLを追跡しています…"
        resumePendingBrowserAnalysisIfReady()
      }
    }
    .onChange(of: mediaURLText) { value in
      persistMediaInputDraft(value)
    }
    .onChange(of: pendingBrowserAnalysisResume) { shouldResume in
      UserDefaults.standard.set(
        shouldResume,
        forKey: Self.resumeAnalysisDefaultsKey
      )
    }
    .onChange(of: selectedTab) { tab in
      UserDefaults.standard.set(
        tab.rawValue,
        forKey: Self.selectedTabDefaultsKey
      )
      if tab != .browser {
        if tab == .playback, browserPlaybackTransitionInProgress {
          // analyzeBrowserCandidates deliberately selected Playback. Do not
          // interpret that programmatic transition as the user abandoning an
          // unfinished browser analysis: doing so clears autoStartURLPlayback
          // and leaves Playback idle, then re-runs analysis when Browser is
          // opened again.
          browserPlaybackTransitionInProgress = false
          pendingBrowserAnalysisResume = false
          browserAnalysisResumeTask?.cancel()
          browserAnalysisResumeTask = nil
          interactiveBrowser.suspendForBackground()
          return
        }
        let shouldResume =
          pendingBrowserAnalysisResume || autoStartURLPlayback
          || browserAnalysisTask != nil
        pendingBrowserAnalysisResume = shouldResume
        invalidateBrowserAnalysis(
          preservingResumeIntent: shouldResume
        )
        interactiveBrowser.suspendForBackground()
      } else {
        interactiveBrowser.resumeAfterBackground()
        restorePersistedBrowserSessionIfNeeded()
        resumePendingBrowserAnalysisIfReady()
      }
    }
    .onChange(of: scenePhase) { phase in
      if phase == .active {
        sftp.enterForeground()
      }
      if phase == .background {
        let shouldResume =
          selectedTab == .browser
          && (pendingBrowserAnalysisResume || autoStartURLPlayback
            || browserAnalysisTask != nil)
        pendingBrowserAnalysisResume = shouldResume
        persistWorkspaceSession()
        interactiveBrowser.suspendForBackground()
        autoStartURLPlayback = false
        invalidateURLAnalysis()
        invalidateBrowserAnalysis(
          preservingResumeIntent: shouldResume
        )
        if store.isRunning {
          store.cancel()
        }
        realtimePlayer.stop()
        cancelPendingSFTPStreamingSelection()
        store.clearSFTPStreamingInput()
        sftp.enterBackground()
      } else if phase == .active, selectedTab == .browser {
        interactiveBrowser.resumeAfterBackground()
        restorePersistedBrowserSessionIfNeeded()
        resumePendingBrowserAnalysisIfReady()
      }
    }
    .onReceive(playbackTimer) { _ in
      updatePlaybackProgress()
    }
    .onReceive(
      NotificationCenter.default.publisher(
        for: UIApplication.protectedDataWillBecomeUnavailableNotification
      )
    ) { _ in
      realtimePlayer.stop()
      cancelPendingSFTPStreamingSelection()
      store.clearSFTPStreamingInput()
      sftp.cancelAndDisconnect()
    }
    .onDisappear {
      let shouldResume =
        selectedTab == .browser
        && (pendingBrowserAnalysisResume || autoStartURLPlayback
          || browserAnalysisTask != nil)
      pendingBrowserAnalysisResume = shouldResume
      persistWorkspaceSession()
      interactiveBrowser.suspendForBackground()
      autoStartURLPlayback = false
      invalidateURLAnalysis()
      invalidateBrowserAnalysis(
        preservingResumeIntent: shouldResume
      )
      player?.pause()
      realtimePlayer.stop()
      cancelPendingSFTPStreamingSelection()
      store.clearSFTPStreamingInput()
      if scenePhase == .background {
        sftp.enterBackground()
      } else {
        sftp.cancelAndDisconnect()
      }
    }
    .task {
      if selectedTab == .browser {
        interactiveBrowser.resumeAfterBackground()
        restorePersistedBrowserSessionIfNeeded()
      }
      if worker.preparation == .idle {
        await worker.prepareModels()
      }
      if let prepared = worker.preparedWorker {
        store.configure(with: prepared)
      }
    }
  }

  private var workspace: some View {
    VStack(spacing: 0) {
      brandHeader
      Divider()
      tabStrip
      Divider()
      workspaceContent(for: selectedTab)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .disabled(realtimePlayer.isActive && selectedTab != .playback)
      if selectedTab != .browser {
        runFooter
      }
    }
    .background(Color(uiColor: .systemGroupedBackground))
  }

  private var brandHeader: some View {
    HStack(spacing: 12) {
      Image("MiohLogo")
        .resizable()
        .interpolation(.high)
        .accessibilityHidden(true)
        .frame(width: 34, height: 34)

      VStack(alignment: .leading, spacing: 2) {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
          Text("mioh")
            .font(.title2.weight(.semibold))
          Text("Motion-Informed Optical Healing")
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .minimumScaleFactor(0.8)
        }
        Text(store.restorationModelIdentifier)
          .font(.caption)
          .foregroundStyle(.secondary)
          .lineLimit(1)
      }
      .layoutPriority(1)

      Spacer(minLength: 12)

      Text(headerStatusText)
        .font(.callout.monospacedDigit())
        .foregroundStyle(headerStatusColor)
        .lineLimit(1)
    }
    .padding(.leading, Self.windowControlsLeadingInset)
    .padding(.trailing, 20)
    .frame(height: 66)
    .background(Color(uiColor: .systemBackground))
    .accessibilityElement(children: .combine)
  }

  private var tabStrip: some View {
    HStack(spacing: 0) {
      ForEach(WorkspaceTab.allCases) { tab in
        Button {
          withAnimation(.easeInOut(duration: 0.16)) {
            selectedTab = tab
          }
        } label: {
          VStack(spacing: 3) {
            Image(systemName: tab.icon)
              .font(.system(size: 14, weight: .semibold))
            Text(tab.title)
              .font(.caption2.weight(.semibold))
              .lineLimit(1)
              .minimumScaleFactor(0.7)
          }
          .foregroundStyle(selectedTab == tab ? Color.accentColor : .secondary)
          .frame(maxWidth: .infinity)
          .frame(minHeight: 44)
          .padding(.horizontal, 2)
          .padding(.vertical, 5)
          .contentShape(Rectangle())
          .background {
            if selectedTab == tab {
              Capsule().fill(Color.accentColor.opacity(0.12))
            }
          }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(tab.title)
        .frame(maxWidth: .infinity)
      }
    }
    .padding(.leading, Self.windowControlsLeadingInset)
    .padding(.trailing, 10)
    .padding(.vertical, 6)
    .frame(maxWidth: .infinity)
    .background(Color(uiColor: .secondarySystemBackground))
  }

  @ViewBuilder
  private func workspaceContent(for tab: WorkspaceTab) -> some View {
    switch tab {
    case .basic:
      basicView
    case .browser:
      browserView
    case .split:
      splitView
    case .restoration:
      restorationView
    case .detection:
      detectionView
    case .output:
      outputView
    case .memory:
      memoryView
    case .settings:
      settingsView
    case .playback:
      playbackView
    case .log:
      logView
    }
  }

  private var basicView: some View {
    Form {
      Section("入力 / 出力") {
        Button {
          choosingInput = true
        } label: {
          Label(
            store.inputURL == nil ? "入力動画を選択" : "入力動画を変更",
            systemImage: "film.stack"
          )
        }
        .disabled(store.isRunning)
        Button {
          activeSheet = .sftpInput
        } label: {
          Label("SFTPから動画を選択", systemImage: "externaldrive.connected.to.line.below")
        }
        .disabled(store.isRunning)
        TextField("動画または配信ページのURL", text: $mediaURLText)
          .textInputAutocapitalization(.never)
          .autocorrectionDisabled()
          .keyboardType(.URL)
          .disabled(store.isRunning || store.isResolvingURL)
          .onSubmit { analyzeURLAndStartPlayback() }
        Button {
          analyzeURLAndStartPlayback()
        } label: {
          if store.isResolvingURL {
            HStack {
              ProgressView()
              Text("URLを解析中…")
            }
          } else {
            Label("URLを解析", systemImage: "link.badge.plus")
          }
        }
        .disabled(
          store.isRunning || store.isResolvingURL
            || mediaURLText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        )
        if let urlInputStatus = store.urlInputStatus {
          Text(urlInputStatus)
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        if let inputURL = store.inputURL {
          LabeledContent(
            "入力",
            value: store.inputDisplayName ?? inputURL.lastPathComponent
          )
          if store.isHLSInput {
            LabeledContent("形式", value: "HLSストリーミング")
          } else if store.isSFTPStreamingInput {
            LabeledContent("形式", value: "SFTPストリーミング")
          }
          if let duration = store.inputDuration {
            LabeledContent(
              store.isLiveHLSInput ? "現在の配信範囲" : "長さ",
              value: time(duration)
            )
          }
          Button("入力を解除", role: .destructive) {
            autoStartURLPlayback = false
            invalidateURLAnalysis()
            invalidateBrowserAnalysis(closePage: true)
            realtimePlayer.stop()
            store.clearInput()
          }
          .disabled(store.isRunning)
        } else {
          Text("Filesの動画、HLS URL、または動画を含むWebページを指定できます。")
            .foregroundStyle(.secondary)
        }
        LabeledContent("保存先", value: store.outputLocationLabel)
        Text("完成したMP4はFilesアプリの「このiPad内 ＞ mioh Remote」に残ります。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("実行") {
        LabeledContent("エンジン", value: "Swiftネイティブ / Core AI")
        LabeledContent("デバイス", value: "このiPad")
        LabeledContent("精度", value: "FP16")
        LabeledContent("同時ジョブ", value: "1")
        Toggle("処理中は画面をスリープしない", isOn: $store.keepScreenAwake)
          .disabled(store.isRunning)
      }

      Section("状態") {
        LabeledContent("処理", value: store.state.label)
        modelPreparationStatus
        if worker.isRunning {
          Label(
            "Worker動作中はiPad単体復元を開始できません。",
            systemImage: "exclamationmark.triangle.fill"
          )
          .foregroundStyle(.orange)
        }
        if case .failed(let message) = store.state {
          Text(message).foregroundStyle(.red)
        }
      }
    }
  }

  private var browserView: some View {
    VStack(spacing: 0) {
      HStack(spacing: 8) {
        Button {
          invalidateBrowserAnalysis()
          interactiveBrowser.goBack()
        } label: {
          Image(systemName: "chevron.backward")
        }
        .disabled(!interactiveBrowser.canGoBack)
        .accessibilityLabel("戻る")

        Button {
          invalidateBrowserAnalysis()
          interactiveBrowser.goForward()
        } label: {
          Image(systemName: "chevron.forward")
        }
        .disabled(!interactiveBrowser.canGoForward)
        .accessibilityLabel("進む")

        Button {
          invalidateBrowserAnalysis()
          if interactiveBrowser.isLoading {
            interactiveBrowser.stop()
          } else {
            interactiveBrowser.reload()
          }
        } label: {
          Image(
            systemName: interactiveBrowser.isLoading
              ? "xmark" : "arrow.clockwise"
          )
        }
        .accessibilityLabel(
          interactiveBrowser.isLoading ? "読み込みを停止" : "再読み込み"
        )

        TextField(
          "https://…",
          text: $interactiveBrowser.addressText
        )
        .textFieldStyle(.roundedBorder)
        .textInputAutocapitalization(.never)
        .autocorrectionDisabled()
        .keyboardType(.URL)
        .submitLabel(.go)
        .focused($browserAddressFocused)
        .onSubmit { openBrowserAddress() }

        Button {
          toggleCurrentBrowserBookmark()
        } label: {
          Image(
            systemName: currentBrowserPageIsBookmarked ? "star.fill" : "star"
          )
        }
        .buttonStyle(.bordered)
        .disabled(
          interactiveBrowser.currentPublicPageURL == nil
            || interactiveBrowser.isLoading || interactiveBrowser.challengeActive
        )
        .accessibilityLabel(
          currentBrowserPageIsBookmarked
            ? "ブックマークを解除" : "ブックマークに追加"
        )

        Menu {
          Button {
            showingBrowserLibrary = .history
          } label: {
            Label("履歴", systemImage: "clock")
          }
          Button {
            showingBrowserLibrary = .bookmarks
          } label: {
            Label("ブックマーク", systemImage: "star")
          }
        } label: {
          Image(systemName: "book.closed")
        }
        .buttonStyle(.bordered)
        .accessibilityLabel("履歴とブックマーク")

        Button("移動") {
          openBrowserAddress()
        }
        .buttonStyle(.borderedProminent)
        .disabled(
          interactiveBrowser.addressText.trimmingCharacters(
            in: .whitespacesAndNewlines
          ).isEmpty
        )

        Button("閉じる") {
          closeBrowserTab()
        }
        .buttonStyle(.bordered)
      }
      .padding(.horizontal, 14)
      .padding(.vertical, 9)
      .background(Color(uiColor: .secondarySystemBackground))

      Divider()

      HStack(spacing: 10) {
        if interactiveBrowser.isLoading || store.isResolvingURL {
          ProgressView()
            .controlSize(.small)
        } else {
          Image(
            systemName: interactiveBrowser.challengeActive
              ? "hand.raised.fill" : "network"
          )
          .foregroundStyle(
            interactiveBrowser.challengeActive ? .orange : .secondary
          )
        }

        VStack(alignment: .leading, spacing: 2) {
          Text(browserStatusText)
            .font(.subheadline.weight(.medium))
            .lineLimit(2)
          if !interactiveBrowser.pageTitle.isEmpty {
            Text(interactiveBrowser.pageTitle)
              .font(.caption)
              .foregroundStyle(.secondary)
              .lineLimit(1)
          }
        }

        Spacer(minLength: 12)

        if interactiveBrowser.candidateCount > 0 {
          Button {
            showingBrowserCandidates = true
          } label: {
            Label(
              "候補 \(interactiveBrowser.candidateCount)件",
              systemImage: "list.bullet.rectangle"
            )
            .font(.caption.monospacedDigit())
          }
          .buttonStyle(.bordered)
        }

        if interactiveBrowser.hasOpenedPage {
          Button {
            invalidateBrowserAnalysis()
            browserAddressFocused = false
            interactiveBrowser.showOpenedPage()
          } label: {
            Label("開いたページ", systemImage: "rectangle.on.rectangle")
          }
          .buttonStyle(.borderedProminent)
        }

        if interactiveBrowser.canReturnToOpeningPage {
          Button {
            invalidateBrowserAnalysis()
            browserAddressFocused = false
            interactiveBrowser.returnToOpeningPage()
          } label: {
            Label("元のページ", systemImage: "arrowshape.turn.up.backward")
          }
          .buttonStyle(.bordered)
        }

        if interactiveBrowser.challengeCompatibilityTimedOut,
          let pageURL = interactiveBrowser.currentPublicPageURL
        {
          Button {
            browserAddressFocused = false
            openURL(pageURL)
          } label: {
            Label("Safariで開く（閲覧）", systemImage: "safari")
          }
          .buttonStyle(.bordered)
          .accessibilityHint("Safariの確認情報はmiohの解析には引き継がれません")
        }

        Button {
          analyzeBrowserCandidates()
        } label: {
          Label("配信を解析", systemImage: "magnifyingglass")
        }
        .buttonStyle(.borderedProminent)
        .disabled(
          browserAnalysisTask != nil || store.isRunning || store.isResolvingURL
            || interactiveBrowser.challengeActive || interactiveBrowser.isLoading
            || interactiveBrowser.addressText.trimmingCharacters(
              in: .whitespacesAndNewlines
            ).isEmpty
        )
      }
      .padding(.horizontal, 14)
      .padding(.vertical, 8)
      .background(.ultraThinMaterial)

      Divider()

      IPadInteractiveBrowserWebView(browser: interactiveBrowser)
        .id(interactiveBrowser.webViewGeneration)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(uiColor: .systemBackground))
    }
  }

  private var browserCandidateList: some View {
    NavigationStack {
      List {
        Section {
          ForEach(interactiveBrowser.candidateSummaries) { candidate in
            VStack(alignment: .leading, spacing: 7) {
              HStack(spacing: 8) {
                Text(
                  "#\((interactiveBrowser.candidateSummaries.firstIndex(of: candidate) ?? 0) + 1)"
                )
                .font(.caption.monospacedDigit().weight(.semibold))
                Text(candidate.verificationLabel)
                  .font(.caption.weight(.semibold))
                  .foregroundStyle(candidate.isVerified ? .green : .orange)
                Spacer()
                Text(candidate.sourceLabel)
                  .font(.caption)
                  .foregroundStyle(.secondary)
                if candidate.frameDepth > 0 {
                  Text("frame \(candidate.frameDepth)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
                }
              }
              Text(candidate.url)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, 4)
          }
        } header: {
          Text("検出順・優先順")
        } footer: {
          Text(
            "未確認候補には、動画そのものではないページ・iframe・通信履歴も含まれます。再生ソース／応答確認済みが本編候補です。"
          )
        }
      }
      .navigationTitle("配信候補 \(interactiveBrowser.candidateCount)件")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("閉じる") { showingBrowserCandidates = false }
        }
      }
    }
    .presentationDetents([.medium, .large])
  }

  private func browserLibrarySheet(
    _ initialSection: BrowserLibrarySection
  ) -> some View {
    BrowserLibrarySheet(
      store: browserLibrary,
      initialSection: initialSection,
      openEntry: openBrowserLibraryEntry
    )
    .presentationDetents([.medium, .large])
  }

  private var splitView: some View {
    Form {
      Section("並列処理") {
        Picker("復元runner数", selection: $store.parallelRestorationLanes) {
          Text("1 — 標準").tag(1)
          Text("2 — 高負荷").tag(2)
          Text("3 — 最大").tag(3)
        }
        .pickerStyle(.segmented)
        .disabled(store.isRunning)
        Text(
          store.parallelRestorationLanes == 1
            ? "全編復元とローカル動画のリアルタイム復元で、Core AI復元runnerを1つ使用します。"
            : "全編復元とローカル動画のリアルタイム復元で、独立したCore AI復元runnerを\(store.parallelRestorationLanes)つ使用します。再生順を保ったまま連続区間を同時処理します。"
        )
        .font(.caption)
        .foregroundStyle(.secondary)
        if store.parallelRestorationLanes > 1 {
          Label(
            "メモリ使用量が増えます。不安定な場合は1または2に下げてください。",
            systemImage: "exclamationmark.triangle"
          )
          .font(.caption)
          .foregroundStyle(.orange)
        }
      }

      Section("分割") {
        LabeledContent("方式", value: "ネイティブ自動分割")
        LabeledContent("ファイル分割", value: "なし（完成MP4は1本）")
        Text("動画はBasicVSR++のクリップ境界を保って処理し、並列laneの完了後に元の順序で1本のMP4へ結合します。一時データは完成または中止時に整理します。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("処理クリップ") {
        Stepper(
          "クリップ長 \(store.clipLength)フレーム",
          value: $store.clipLength,
          in: 1...store.maximumClipLength
        )
        .disabled(store.isRunning)
        Stepper(
          "時間オーバーラップ \(store.temporalOverlap)フレーム",
          value: $store.temporalOverlap,
          in: 0...max(0, store.clipLength - 1)
        )
        .disabled(store.isRunning)
        Toggle("境界をクロスフェード", isOn: $store.crossfade)
          .disabled(store.isRunning)
        Text("クリップ処理に合わせて「再生」タブのライブ画像が順次更新されます。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }

  private var restorationView: some View {
    Form {
      Section("復元モデル") {
        Picker("BasicVSR++", selection: $store.restorationModelIdentifier) {
          ForEach(restorationModels, id: \.self) { identifier in
            Text(restorationLabel(identifier)).tag(identifier)
          }
        }
        .disabled(store.isRunning)
        LabeledContent("選択中", value: store.restorationModelIdentifier)
          .font(.caption)
          .textSelection(.enabled)
        Text("可変長、固定T18、T36、T90をすべて同梱しています。T36/T90はiPadのMetalメモリに合わせて同等の可変長ランナーで実行します。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("時間方向") {
        Stepper(
          "最大クリップ \(store.clipLength)フレーム",
          value: $store.clipLength,
          in: 1...store.maximumClipLength
        )
        .disabled(store.isRunning)
        Stepper(
          "オーバーラップ \(store.temporalOverlap)フレーム",
          value: $store.temporalOverlap,
          in: 0...max(0, store.clipLength - 1)
        )
        .disabled(store.isRunning)
        Toggle("クロスフェード", isOn: $store.crossfade)
          .disabled(store.isRunning)
      }

      Section("合成") {
        VStack(alignment: .leading, spacing: 8) {
          HStack {
            Text("境界フェザー")
            Spacer()
            Text(store.blendFeather, format: .number.precision(.fractionLength(2)))
              .monospacedDigit()
          }
          Slider(value: $store.blendFeather, in: 0...0.5)
        }
        .disabled(store.isRunning)
        LabeledContent("復元後エフェクト", value: "1×・無加工")
        Text("iPadネイティブ経路は復元結果をそのまま合成し、追加の後処理エフェクトは使用しません。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }

  private var detectionView: some View {
    Form {
      Section("モザイク検出") {
        Picker("検出モデル", selection: $store.detectorModelIdentifier) {
          ForEach(detectorModels, id: \.self) { identifier in
            Text(detectorLabel(identifier)).tag(identifier)
          }
        }
        .disabled(store.isRunning)
        LabeledContent("選択中", value: store.detectorModelIdentifier)
          .font(.caption)
          .textSelection(.enabled)
        Toggle("顔だけを復元", isOn: $store.detectFaceMosaics)
          .disabled(store.isRunning)
        Stepper(
          "空フレーム先読み \(store.detectionEmptyLookahead)",
          value: $store.detectionEmptyLookahead,
          in: 1...120
        )
        .disabled(store.isRunning)
        Text("指定フレーム窓の先頭と末尾がともに未検出なら、中間フレームの検出と復元を省略します。通常復元とリアルタイム復元の両方に適用されます。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("対応モデル") {
        ForEach(detectorModels, id: \.self) { identifier in
          HStack {
            Image(
              systemName: identifier == store.detectorModelIdentifier
                ? "checkmark.circle.fill" : "circle"
            )
            .foregroundStyle(
              identifier == store.detectorModelIdentifier
                ? Color.green : .secondary)
            Text(detectorLabel(identifier))
          }
        }
      }
    }
  }

  private var outputView: some View {
    Form {
      Section("映像") {
        Picker("映像コーデック", selection: $store.videoCodec) {
          Text("HEVC").tag("hevc")
          Text("H.264").tag("h264")
        }
        .pickerStyle(.segmented)
        .disabled(store.isRunning)
        VStack(alignment: .leading, spacing: 8) {
          HStack {
            Text("ビットレート倍率")
            Spacer()
            Text(store.bitrateMultiplier, format: .number.precision(.fractionLength(1)))
              .monospacedDigit()
            Text("×")
          }
          Slider(value: $store.bitrateMultiplier, in: 0.5...3, step: 0.1)
        }
        .disabled(store.isRunning)
        Toggle("高速開始（Fast Start）", isOn: $store.mp4FastStart)
          .disabled(store.isRunning)
        Toggle("FPS変換", isOn: $store.useFPS)
          .disabled(store.isRunning)
        Picker("フレームレート", selection: $store.selectedFrameRate) {
          ForEach(IPadStandaloneStore.frameRateOptions) { option in
            Text("\(option.label)fps").tag(option.id)
          }
        }
        .disabled(store.isRunning || !store.useFPS)
        Text(
          store.useFPS
            ? "復元前に\(store.targetFPSLabel)fpsへ変換し、検出・復元するフレーム数を減らします。元動画より高いレートは指定できません。"
            : "入力動画のフレームレートを維持します。"
        )
        .font(.caption)
        .foregroundStyle(.secondary)
        LabeledContent("音声", value: "元動画から自動結合")
      }

      Section("保存先") {
        LabeledContent("Files", value: store.outputLocationLabel)
        if let outputURL = store.outputURL {
          LabeledContent("ファイル", value: outputURL.lastPathComponent)
          if let metrics = store.metrics {
            LabeledContent("処理フレーム", value: "\(metrics.processedFrames)")
            LabeledContent(
              "処理時間",
              value: String(format: "%.1f秒", metrics.wallSeconds)
            )
            LabeledContent(
              "ファイルサイズ",
              value: byteCount(metrics.outputByteCount)
            )
          }
          Button {
            activeSheet = .share
          } label: {
            Label("Files・写真へ保存／共有", systemImage: "square.and.arrow.up")
          }
          Button {
            activeSheet = .sftpUpload
          } label: {
            Label("完成MP4をSFTPへ送信", systemImage: "arrow.up.doc")
          }
          Button("結果表示を閉じる", role: .destructive) {
            player?.pause()
            player = nil
            store.clearOutput()
          }
        } else {
          Text("復元完了後、ここからFiles・写真・AirDropへ共有できます。")
            .foregroundStyle(.secondary)
        }
      }
    }
  }

  private var memoryView: some View {
    Form {
      Section("メモリ管理") {
        LabeledContent("モデル読込", value: "実行時に選択モデルだけ")
        LabeledContent("同時推論", value: "1ジョブ")
        LabeledContent("最大クリップ", value: "\(store.maximumClipLength)フレーム")
        LabeledContent("プレビュー解像度", value: "最大960px")
        Text("Core AIとiPadOSのMetalメモリ管理を使用し、巨大な固定グラフを同時に読み込みません。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("再生バッファ") {
        Toggle("復元中ライブプレビュー", isOn: $store.livePreviewEnabled)
          .disabled(store.isRunning)
        HStack {
          Text("通信入力の先読み上限")
          Slider(value: $store.previewBufferLimit, in: 1...60, step: 1)
          Text("\(Int(store.previewBufferLimit))秒")
            .font(.caption.monospacedDigit())
            .frame(width: 48, alignment: .trailing)
        }
        .disabled(realtimePlayer.isActive)
        Text("ローカルファイルは秒数制限なしで動画末尾までディスクへ先行復元します。SFTPストリーミングとHLSはこの上限まで先読みします。シーク時は指定位置から復元を再開します。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }

  private var settingsView: some View {
    Form {
      Section("同梱モデル") {
        modelPreparationStatus
        if case .ready(let restorations, let detectors, let maximumFrames) =
          worker.preparation
        {
          LabeledContent("復元", value: "\(restorations.count)種")
          LabeledContent("検出", value: "\(detectors.count)種")
          LabeledContent("最大", value: "\(maximumFrames)フレーム")
        }
        Button("同梱モデルを再検証") {
          Task {
            await worker.prepareModels()
            if let prepared = worker.preparedWorker {
              store.configure(with: prepared)
            }
          }
        }
        .disabled(store.isRunning || worker.isRunning || worker.preparation == .validating)
      }

      Section("HLSストリーミング") {
        Picker("通信方式", selection: $store.hlsStreamingMode) {
          ForEach(IPadHLSStreamingMode.allCases) { mode in
            Text(mode.label).tag(mode)
          }
        }
        .disabled(store.isRunning || store.isResolvingURL)
        Picker("画質", selection: $store.hlsQualityPreference) {
          ForEach(IPadHLSQualityPreference.allCases) { quality in
            Text(quality.label).tag(quality)
          }
        }
        .disabled(store.isRunning || store.isResolvingURL)
        Text(
          store.hlsStreamingMode == .safariCompatible
            ? "VODは再生クロックで制限せず、通信・復元が処理できる速度で取り込みます。AES-128暗号化HLSにも対応します。ライブ配信は実時間で処理します。"
            : "区間を並列先読みする高速方式です。指定画質を超えない最も高いvariantを選びます。"
        )
        .font(.caption)
        .foregroundStyle(.secondary)
      }

      Section("リアルタイム復元") {
        Toggle(
          "再生を最大29.97/30fpsにする",
          isOn: $store.limitHighFrameRateBeforeRestoration
        )
        .disabled(store.isRunning)
        Text("59.94fpsは29.97fps、60fpsは30fpsへ、検出に入る前に間引きます。30fps以下は変更しません。")
          .font(.caption)
          .foregroundStyle(.secondary)

        Stepper(
          "検出後にスキップ \(store.detectionMaskReuseSkipFrames)フレーム",
          value: $store.detectionMaskReuseSkipFrames,
          in: 0...12
        )
        .disabled(store.isRunning)
        Text("0は毎フレーム検出です。1以上ではCore ML検出後の指定フレーム数だけ、直前の検出領域とマスクを再利用します。CoreAI復元は対象フレームすべてに適用します。")
          .font(.caption)
          .foregroundStyle(.secondary)

        Picker("復元フレームレート", selection: $store.realtimeFrameRateMode) {
          ForEach(IPadRealtimeFrameRateMode.allCases) { mode in
            Text(mode.label).tag(mode)
          }
        }
        .disabled(store.isRunning)
        Text("元フレームレート維持では上の最大30fps処理後のPTSを使います。24fps固定はさらに24fpsへ間引き、自動は復元が再生に追いつかない場合だけ24fpsへ切り替えます。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("ユーザーデフォルト") {
        Button("現在の設定をデフォルトに保存") {
          store.saveSettingsAsDefaults()
        }
        .disabled(store.isRunning)
        Button("保存済みデフォルトを読み込み") {
          store.loadSavedSettings()
        }
        .disabled(store.isRunning || !store.hasSavedSettings)
        Button("初期値に戻す", role: .destructive) {
          store.resetSettings()
        }
        .disabled(store.isRunning)
        Text("復元・検出・出力・メモリ・再生設定を保存し、次回起動時に自動適用します。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("Mac連携（任意）") {
        LabeledContent("接続", value: remote.connected ? "接続中" : "未接続")
        Button {
          activeSheet = .remote
        } label: {
          Label("Macリモート操作", systemImage: "desktopcomputer")
        }
        .disabled(store.isRunning)
        Button {
          activeSheet = .fullRemote
        } label: {
          Label("Mac版の全設定画面", systemImage: "macwindow")
        }
        .disabled(store.isRunning || !remote.connected)
        Text("iPad単体復元にはMac接続も認証コードも必要ありません。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("ローカル復元クラスタ") {
        LabeledContent("iPad Worker", value: worker.stateLabel)
        Button {
          activeSheet = .worker
        } label: {
          Label("Worker設定を開く", systemImage: "ipad.and.arrow.forward")
        }
        .disabled(store.isRunning)
        Text("Workerのクラスタ接続に認証コードはありません。信頼できるLAN内だけで使用します。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }

  private var playbackView: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        GroupBox {
          ZStack {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
              .fill(.black)

            if showingFullscreenPlayback {
              Color.black
            } else {
              playbackVideoSurface
            }

            if realtimeSessionVisible || store.isRunning {
              VStack {
                HStack {
                  Label(
                    realtimeSessionVisible ? "リアルタイム復元" : "復元ライブ",
                    systemImage: "dot.radiowaves.left.and.right"
                  )
                  .font(.caption.weight(.semibold))
                  .padding(.horizontal, 10)
                  .padding(.vertical, 6)
                  .background(.ultraThinMaterial, in: Capsule())
                  Spacer()
                }
                Spacer()
              }
              .padding(12)
            }

            if canEnterFullscreenPlayback {
              VStack {
                HStack {
                  Spacer()
                  Button {
                    showingFullscreenPlayback = true
                  } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                      .font(.headline)
                      .frame(width: 44, height: 44)
                      .background(.ultraThinMaterial, in: Circle())
                  }
                  .buttonStyle(.plain)
                  .foregroundStyle(.white)
                  .accessibilityLabel("フルスクリーン")
                }
                Spacer()
              }
              .padding(12)
            }
          }
          .aspectRatio(16 / 9, contentMode: .fit)
        } label: {
          HStack {
            Text(playbackPanelTitle)
            Spacer()
            if realtimeSessionVisible {
              Text(time(realtimePlayer.position))
                .font(.caption.monospacedDigit())
            } else if let position = store.livePreviewPosition,
              store.outputURL == nil
            {
              Text(time(position))
                .font(.caption.monospacedDigit())
            }
          }
        }

        if realtimeSessionVisible {
          GroupBox("再生操作") {
            VStack(alignment: .leading, spacing: 12) {
              HStack(spacing: 14) {
                if store.isSFTPStreamingInput {
                  Label(
                    sftpStreamingBitRate(realtimePlayer.sftpBitsPerSecond),
                    systemImage: "arrow.down.circle"
                  )
                  Text("Range \(realtimePlayer.sftpActiveRangeReads)並列")
                  Text(
                    String(
                      format: "応答 %.2f秒",
                      realtimePlayer.sftpRangeLatencySeconds
                    )
                  )
                }
                if realtimePlayer.restorationRealtimeFactor > 0 {
                  Text(
                    String(
                      format: "全体処理 %.2f倍速 / RTF %.2f",
                      1 / realtimePlayer.restorationRealtimeFactor,
                      realtimePlayer.restorationRealtimeFactor
                    )
                  )
                } else {
                  Text("復元能力を計測中")
                }
                if realtimePlayer.processingFramesPerSecond > 0 {
                  Text(
                    String(
                      format: "全体 %.2f fps",
                      realtimePlayer.processingFramesPerSecond
                    )
                  )
                }
                if realtimePlayer.restorationFramesPerSecond > 0 {
                  Text(
                    String(
                      format: "実復元 %.2f fps",
                      realtimePlayer.restorationFramesPerSecond
                    )
                  )
                } else if realtimePlayer.restorationRealtimeFactor > 0 {
                  Text("実復元なし（素通し）")
                }
                Text("\(realtimePlayer.restorationParallelLanes) runner")
                Spacer()
                Text(realtimePlayer.pipelineStageLabel)
              }
              .font(.caption.monospacedDigit())
              .foregroundStyle(.secondary)

              HStack {
                Text(time(editingRealtimePosition ? realtimeSeekPosition : realtimePlayer.position))
                  .font(.caption.monospacedDigit())
                  .frame(width: 58, alignment: .leading)
                Slider(
                  value: Binding(
                    get: {
                      editingRealtimePosition
                        ? realtimeSeekPosition : realtimePlayer.position
                    },
                    set: { realtimeSeekPosition = $0 }
                  ),
                  in: 0...max(realtimePlayer.duration, 0.01),
                  onEditingChanged: { editing in
                    if editing {
                      realtimeSeekPosition = realtimePlayer.position
                      editingRealtimePosition = true
                    } else {
                      let target = realtimeSeekPosition
                      editingRealtimePosition = false
                      realtimePlayer.seek(to: target)
                    }
                  }
                )
                .disabled(!realtimePlayer.isSeekable)
                Text(
                  realtimePlayer.isSeekable
                    ? time(realtimePlayer.duration) : "ライブ"
                )
                .font(.caption.monospacedDigit())
                .frame(width: 58, alignment: .trailing)
              }

              HStack(spacing: 12) {
                Button {
                  if realtimeNeedsRestart {
                    startRealtimePreview()
                  } else {
                    realtimePlayer.togglePlayback()
                  }
                } label: {
                  Label(
                    realtimePlayer.isPlaybackRequested ? "一時停止" : "再生",
                    systemImage: realtimePlayer.isPlaybackRequested
                      ? "pause.fill" : "play.fill"
                  )
                }
                Button(role: .destructive) {
                  realtimePlayer.stop()
                } label: {
                  Label("停止", systemImage: "stop.fill")
                }
                Toggle("処理前", isOn: $realtimePlayer.showOriginal)
                  .toggleStyle(.switch)
                  .fixedSize(horizontal: true, vertical: false)
                Spacer()
                if !realtimePlayer.cachesLocalInputToEnd {
                  Text("先読み \(realtimePlayer.bufferedSeconds, specifier: "%.1f")秒")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                } else {
                  Text("ディスクキャッシュ \(realtimePlayer.bufferedSeconds, specifier: "%.1f")秒（末尾まで）")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                }
              }
              .buttonStyle(.bordered)

              HStack {
                Image(
                  systemName: realtimePlayer.muted
                    ? "speaker.slash.fill" : "speaker.wave.2.fill"
                )
                Slider(value: $realtimePlayer.volume, in: 0...1)
                  .frame(maxWidth: 240)
                Toggle("消音", isOn: $realtimePlayer.muted)
                  .toggleStyle(.switch)
              }

              HStack {
                Text(realtimePlayer.state.label)
                if let transport = realtimePlayer.hlsTransportLabel {
                  Text("・\(transport)")
                }
                Spacer()
                Text("復元位置 \(time(realtimePlayer.processingPosition))")
                  .monospacedDigit()
              }
              .font(.caption)
              .foregroundStyle(.secondary)

              if let error = realtimePreviewError {
                Text(error)
                  .font(.caption)
                  .foregroundStyle(.red)
              }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
          }
        } else if store.isRunning {
          GroupBox("処理速度でのライブ表示") {
            VStack(alignment: .leading, spacing: 9) {
              ProgressView(value: store.progress)
              HStack {
                Text(store.state.label)
                Spacer()
                Text("\(Int((store.progress * 100).rounded()))%")
                  .monospacedDigit()
              }
              Text("推論は動画の実時間より重いため、復元済みフレームを処理速度で順次表示します。完成後は通常速度で再生できます。")
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
          }
        } else if store.outputURL == nil {
          GroupBox("リアルタイム再生") {
            HStack(spacing: 12) {
              Button {
                startRealtimePreview()
              } label: {
                Label("再生", systemImage: "play.fill")
              }
              .buttonStyle(.borderedProminent)
              .disabled(realtimeStartBlocker != nil)
              Text("復元済みの2秒セグメントを先読みして流します。再生開始後は時刻スライダーでシークできます。")
                .font(.caption)
                .foregroundStyle(.secondary)
              Spacer()
            }
          }
        }

        if store.outputURL != nil, !realtimeSessionVisible {
          GroupBox("再生操作") {
            VStack(spacing: 12) {
              HStack {
                Text(time(playbackPosition))
                  .font(.caption.monospacedDigit())
                  .frame(width: 58, alignment: .leading)
                Slider(
                  value: $playbackPosition,
                  in: 0...max(playbackDuration, 0.01),
                  onEditingChanged: { editing in
                    editingPlaybackPosition = editing
                    if !editing { seekPlayback(to: playbackPosition) }
                  }
                )
                .disabled(playbackDuration <= 0)
                Text(time(playbackDuration))
                  .font(.caption.monospacedDigit())
                  .frame(width: 58, alignment: .trailing)
              }

              HStack(spacing: 12) {
                Button {
                  player?.play()
                } label: {
                  Label("再生", systemImage: "play.fill")
                }
                Button {
                  player?.pause()
                } label: {
                  Label("一時停止", systemImage: "pause.fill")
                }
                Button {
                  playbackPosition = 0
                  seekPlayback(to: 0)
                  player?.play()
                } label: {
                  Label("先頭から", systemImage: "backward.end.fill")
                }
              }
              .buttonStyle(.bordered)

              HStack {
                Image(systemName: playbackMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                Slider(value: $playbackVolume, in: 0...1)
                Toggle("消音", isOn: $playbackMuted)
                  .labelsHidden()
              }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
          }
        }
      }
      .padding(18)
    }
  }

  @ViewBuilder
  private var playbackVideoSurface: some View {
    if realtimeSessionVisible {
      VideoPlayer(player: realtimePlayer.sourcePlayer)
        .allowsHitTesting(false)
      IPadPersistentPlayerSurface(
        player: realtimePlayer.restoredPlayer,
        videoOutputProvider: realtimePlayer.restoredVideoOutput(for:),
        initialPixelBuffer: realtimePlayer.latestRestoredPixelBuffer,
        onFrameDisplayed: realtimePlayer.didDisplayRestoredFrame(
          from:pixelBuffer:)
      )
      .opacity(realtimePlayer.showsSourceFrame ? 0 : 1)
      .allowsHitTesting(false)
    } else if let outputURL = store.outputURL, let player {
      VideoPlayer(player: player)
        .accessibilityLabel(outputURL.lastPathComponent)
    } else if let image = store.livePreviewImage {
      Image(decorative: image, scale: 1, orientation: .up)
        .resizable()
        .scaledToFit()
    } else {
      VStack(spacing: 10) {
        Image(systemName: "play.rectangle")
          .font(.system(size: 42))
        Text(store.isRunning ? "最初の復元フレームを処理中…" : "再生する映像はまだありません")
          .font(.headline)
        Text("Filesで動画を選び、下の「再生」を押すと復元済み映像を順次再生します。")
          .font(.caption)
      }
      .multilineTextAlignment(.center)
      .foregroundStyle(.white.opacity(0.85))
      .padding()
    }
  }

  private var fullscreenPlaybackView: some View {
    ZStack {
      Color.black.ignoresSafeArea()

      playbackVideoSurface
        .frame(maxWidth: .infinity, maxHeight: .infinity)

      Color.clear
        .contentShape(Rectangle())
        .onTapGesture {
          toggleFullscreenControls()
        }
        .accessibilityElement()
        .accessibilityLabel(
          fullscreenControlsVisible
            ? "再生コントロールを隠す" : "再生コントロールを表示"
        )
        .accessibilityAddTraits(.isButton)

      if fullscreenControlsVisible {
        VStack(spacing: 16) {
          HStack {
            Spacer()
            Button {
              dismissFullscreenPlayback()
            } label: {
              Image(systemName: "xmark")
                .font(.headline.weight(.semibold))
                .frame(width: 44, height: 44)
                .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .accessibilityLabel("フルスクリーンを閉じる")
          }

          Spacer()

          fullscreenPlaybackControls
            .padding(14)
            .background(
              .ultraThinMaterial,
              in: RoundedRectangle(cornerRadius: 16, style: .continuous)
            )
        }
        .padding(20)
        .transition(.opacity)
      }
    }
    .animation(.easeInOut(duration: 0.2), value: fullscreenControlsVisible)
    .onAppear {
      revealFullscreenControls()
    }
    .onDisappear {
      resetFullscreenControls()
    }
    .preferredColorScheme(.dark)
    .statusBarHidden(true)
  }

  @ViewBuilder
  private var fullscreenPlaybackControls: some View {
    if realtimeSessionVisible {
      VStack(spacing: 12) {
        HStack {
          Text(time(editingRealtimePosition ? realtimeSeekPosition : realtimePlayer.position))
            .font(.caption.monospacedDigit())
          Slider(
            value: Binding(
              get: {
                editingRealtimePosition
                  ? realtimeSeekPosition : realtimePlayer.position
              },
              set: { realtimeSeekPosition = $0 }
            ),
            in: 0...max(realtimePlayer.duration, 0.01),
            onEditingChanged: { editing in
              if editing {
                beginFullscreenControlInteraction()
                realtimeSeekPosition = realtimePlayer.position
                editingRealtimePosition = true
              } else {
                let target = realtimeSeekPosition
                editingRealtimePosition = false
                realtimePlayer.seek(to: target)
                finishFullscreenControlInteraction()
              }
            }
          )
          .disabled(!realtimePlayer.isSeekable)
          Text(
            realtimePlayer.isSeekable
              ? time(realtimePlayer.duration) : "ライブ"
          )
          .font(.caption.monospacedDigit())
        }

        HStack(spacing: 16) {
          Button {
            if realtimeNeedsRestart {
              startRealtimePreview()
            } else {
              realtimePlayer.togglePlayback()
            }
            revealFullscreenControls()
          } label: {
            Label(
              realtimePlayer.isPlaybackRequested ? "一時停止" : "再生",
              systemImage: realtimePlayer.isPlaybackRequested
                ? "pause.fill" : "play.fill"
            )
          }
          .buttonStyle(.borderedProminent)

          Toggle(
            "処理前",
            isOn: Binding(
              get: { realtimePlayer.showOriginal },
              set: { value in
                realtimePlayer.showOriginal = value
                revealFullscreenControls()
              }
            )
          )
          .toggleStyle(.switch)
          .fixedSize(horizontal: true, vertical: false)

          Spacer()

          if realtimePlayer.cachesLocalInputToEnd {
            Text("キャッシュ \(realtimePlayer.bufferedSeconds, specifier: "%.1f")秒 / 末尾まで")
              .font(.caption.monospacedDigit())
          } else {
            Text("先読み \(realtimePlayer.bufferedSeconds, specifier: "%.1f")秒")
              .font(.caption.monospacedDigit())
          }
        }
      }
    } else if store.outputURL != nil {
      VStack(spacing: 12) {
        HStack {
          Text(time(playbackPosition))
            .font(.caption.monospacedDigit())
          Slider(
            value: $playbackPosition,
            in: 0...max(playbackDuration, 0.01),
            onEditingChanged: { editing in
              editingPlaybackPosition = editing
              if editing {
                beginFullscreenControlInteraction()
              } else {
                seekPlayback(to: playbackPosition)
                finishFullscreenControlInteraction()
              }
            }
          )
          .disabled(playbackDuration <= 0)
          Text(time(playbackDuration))
            .font(.caption.monospacedDigit())
        }

        HStack(spacing: 16) {
          Button {
            player?.play()
            revealFullscreenControls()
          } label: {
            Label("再生", systemImage: "play.fill")
          }
          Button {
            player?.pause()
            revealFullscreenControls()
          } label: {
            Label("一時停止", systemImage: "pause.fill")
          }
          Spacer()
        }
        .buttonStyle(.bordered)
      }
    } else if store.isRunning {
      VStack(alignment: .leading, spacing: 8) {
        ProgressView(value: store.progress)
        HStack {
          Text(store.state.label)
          Spacer()
          Text("\(Int((store.progress * 100).rounded()))%")
            .monospacedDigit()
        }
        .font(.caption)
      }
    } else {
      Text(playbackPanelTitle)
        .font(.caption)
    }
  }

  private func revealFullscreenControls() {
    fullscreenControlsHideTask?.cancel()
    fullscreenControlsHideTask = nil
    fullscreenControlsVisible = true
    guard showingFullscreenPlayback else { return }
    fullscreenControlsHideTask = Task { @MainActor in
      do {
        try await Task.sleep(nanoseconds: 3_000_000_000)
      } catch {
        return
      }
      guard !Task.isCancelled, showingFullscreenPlayback,
        !editingRealtimePosition, !editingPlaybackPosition
      else { return }
      fullscreenControlsVisible = false
      fullscreenControlsHideTask = nil
    }
  }

  private func hideFullscreenControls() {
    fullscreenControlsHideTask?.cancel()
    fullscreenControlsHideTask = nil
    fullscreenControlsVisible = false
  }

  private func toggleFullscreenControls() {
    if fullscreenControlsVisible {
      hideFullscreenControls()
    } else {
      revealFullscreenControls()
    }
  }

  private func beginFullscreenControlInteraction() {
    fullscreenControlsHideTask?.cancel()
    fullscreenControlsHideTask = nil
    fullscreenControlsVisible = true
  }

  private func finishFullscreenControlInteraction() {
    revealFullscreenControls()
  }

  private func dismissFullscreenPlayback() {
    fullscreenControlsHideTask?.cancel()
    fullscreenControlsHideTask = nil
    showingFullscreenPlayback = false
  }

  private func resetFullscreenControls() {
    fullscreenControlsHideTask?.cancel()
    fullscreenControlsHideTask = nil
    fullscreenControlsVisible = true
  }

  private var logView: some View {
    VStack(spacing: 0) {
      HStack {
        Text("処理ログ")
          .font(.headline)
        Text("\(store.logs.count)件")
          .font(.caption)
          .foregroundStyle(.secondary)
        Spacer()
        Button("消去", role: .destructive) {
          store.clearLogs()
        }
        .disabled(store.isRunning)
      }
      .padding(.horizontal, 18)
      .padding(.vertical, 12)

      Divider()

      ScrollView {
        LazyVStack(alignment: .leading, spacing: 8) {
          ForEach(store.logs) { entry in
            HStack(alignment: .firstTextBaseline, spacing: 10) {
              Text(entry.timestamp, style: .time)
                .foregroundStyle(.secondary)
              Text(entry.level.rawValue)
                .foregroundStyle(logColor(entry.level))
                .frame(width: 48, alignment: .leading)
              Text(entry.message)
                .textSelection(.enabled)
              Spacer(minLength: 0)
            }
            .font(.system(.caption, design: .monospaced))
            .frame(maxWidth: .infinity, alignment: .leading)
          }
        }
        .padding(18)
      }
      .background(Color(uiColor: .systemBackground))
    }
  }

  private var runFooter: some View {
    VStack(spacing: 8) {
      if store.isRunning || store.progress > 0 {
        ProgressView(value: store.progress)
          .tint(store.isRunning ? .blue : .green)
      }
      HStack(spacing: 12) {
        Button {
          choosingInput = true
        } label: {
          Label("Files", systemImage: "folder")
        }
        .buttonStyle(.bordered)
        .disabled(store.isRunning || realtimePlayer.isActive)

        Button {
          if store.isResolvingURL {
            invalidateURLAnalysis()
            invalidateBrowserAnalysis()
          } else if store.isRunning {
            store.cancel()
          } else if let prepared = worker.preparedWorker {
            realtimePlayer.stop()
            store.start(prepared: prepared)
            selectedTab = .playback
          }
        } label: {
          Label(
            footerIsStopping ? "停止" : "開始",
            systemImage: footerIsStopping ? "stop.fill" : "play.fill"
          )
          .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .tint(footerIsStopping ? .red : .blue)
        .disabled(
          !footerIsStopping
            && (!store.canRunFullRestoration
              || worker.preparedWorker == nil
              || worker.isRunning
              || realtimePlayer.isActive)
        )
      }
      if let blocker = startBlocker {
        Text(blocker)
          .font(.caption)
          .foregroundStyle(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
    }
    .padding(.horizontal, 16)
    .padding(.vertical, 10)
    .background(.ultraThinMaterial)
  }

  @ViewBuilder
  private var modelPreparationStatus: some View {
    switch worker.preparation {
    case .idle:
      LabeledContent("モデル", value: "未検証")
    case .validating:
      HStack {
        ProgressView()
        Text("同梱モデルの整合性を確認中…")
      }
    case .ready(let restorations, let detectors, let maximumFrames):
      VStack(alignment: .leading, spacing: 3) {
        Label("モデル準備完了", systemImage: "checkmark.seal.fill")
          .foregroundStyle(.green)
        Text("復元 \(restorations.count)種・検出 \(detectors.count)種・最大\(maximumFrames)フレーム")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    case .failed(let reason):
      VStack(alignment: .leading, spacing: 3) {
        Label("モデルを準備できません", systemImage: "xmark.octagon.fill")
          .foregroundStyle(.red)
        Text(reason)
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }

  private var startBlocker: String? {
    if store.isSFTPStreamingInput {
      return "SFTPストリーミング入力は「再生」タブで復元しながら再生します。全編書き出しには長押しメニューの「ダウンロード」を使用してください。"
    }
    if store.isHLSInput {
      return "HLS入力は「再生」タブで復元しながら再生します。完成MP4の全編書き出しは利用できません。"
    }
    return realtimeStartBlocker
  }

  private var realtimeStartBlocker: String? {
    guard !store.isRunning else { return nil }
    if realtimePlayer.isActive {
      return "リアルタイム再生を停止すると全編復元を開始できます。"
    }
    if store.inputURL == nil { return "FilesまたはURLから入力動画を選択してください。" }
    if worker.isRunning { return "iPad Workerを停止すると単体復元を開始できます。" }
    if let blocker = worker.startBlocker { return blocker }
    return nil
  }

  private var footerIsStopping: Bool {
    store.isRunning
  }

  private var realtimeSessionVisible: Bool {
    realtimePlayer.state != .idle
  }

  private var canEnterFullscreenPlayback: Bool {
    realtimeSessionVisible || store.outputURL != nil
      || store.livePreviewImage != nil || store.isRunning
  }

  private var playbackPanelTitle: String {
    if realtimeSessionVisible { return "リアルタイム復元" }
    if store.outputURL != nil { return "復元結果" }
    return store.isRunning ? "復元中プレビュー" : "リアルタイム復元"
  }

  private var realtimePreviewError: String? {
    guard case .failed(let message) = realtimePlayer.state else { return nil }
    return message
  }

  private var realtimeNeedsRestart: Bool {
    switch realtimePlayer.state {
    case .idle, .ended, .failed: true
    case .loading, .buffering, .followingLiveEdge, .playing, .paused: false
    }
  }

  private var browserStatusText: String {
    if interactiveBrowser.challengeCompatibilityTimedOut,
      let status = interactiveBrowser.statusMessage
    {
      return status
    }
    if store.isResolvingURL, let status = store.urlInputStatus {
      return status
    }
    if let browserFailureMessage { return browserFailureMessage }
    return interactiveBrowser.statusMessage
      ?? "URLを入力してページを開いてください。"
  }

  private var currentBrowserPageIsBookmarked: Bool {
    browserLibrary.isBookmarked(interactiveBrowser.currentPublicPageURL)
  }

  private func persistWorkspaceSession() {
    let defaults = UserDefaults.standard
    defaults.set(selectedTab.rawValue, forKey: Self.selectedTabDefaultsKey)
    defaults.set(
      pendingBrowserAnalysisResume,
      forKey: Self.resumeAnalysisDefaultsKey
    )
    persistMediaInputDraft(mediaURLText)
    if !interactiveBrowser.challengeActive,
      let pageURL = interactiveBrowser.currentPublicPageURL,
      let stablePage = IPadInteractiveMediaBrowser.persistableSessionAddress(
        pageURL.absoluteString
      )
    {
      defaults.set(stablePage, forKey: Self.currentPageDefaultsKey)
    }
    if !interactiveBrowser.challengeActive,
      let stableAddress = IPadInteractiveMediaBrowser.persistableSessionAddress(
        interactiveBrowser.addressText
      )
    {
      defaults.set(stableAddress, forKey: Self.addressInputDefaultsKey)
    }
  }

  private func persistBrowserPageVisit(_ pageURL: URL) {
    guard let stablePage = IPadInteractiveMediaBrowser.persistableSessionAddress(
      pageURL.absoluteString
    ) else { return }
    let defaults = UserDefaults.standard
    defaults.set(stablePage, forKey: Self.currentPageDefaultsKey)
    if let stableAddress = IPadInteractiveMediaBrowser.persistableSessionAddress(
      interactiveBrowser.addressText
    ) {
      defaults.set(stableAddress, forKey: Self.addressInputDefaultsKey)
    }
  }

  private func restorePersistedBrowserSessionIfNeeded() {
    guard interactiveBrowser.currentPublicPageURL == nil,
      interactiveBrowser.webView.url == nil
    else { return }
    let defaults = UserDefaults.standard
    guard let storedPage = defaults.string(
      forKey: Self.currentPageDefaultsKey
    ), let stablePage = IPadInteractiveMediaBrowser.persistableSessionAddress(
      storedPage
    ) else { return }

    if let storedAddress = defaults.string(
      forKey: Self.addressInputDefaultsKey
    ), let stableAddress = IPadInteractiveMediaBrowser.persistableSessionAddress(
      storedAddress
    ) {
      interactiveBrowser.addressText = stableAddress
    } else {
      interactiveBrowser.addressText = stablePage
    }
    interactiveBrowser.navigate(stablePage)
  }

  private func clearPersistedBrowserSession() {
    let defaults = UserDefaults.standard
    defaults.removeObject(forKey: Self.currentPageDefaultsKey)
    defaults.removeObject(forKey: Self.addressInputDefaultsKey)
    defaults.set(false, forKey: Self.resumeAnalysisDefaultsKey)
  }

  private func persistMediaInputDraft(_ rawValue: String) {
    let defaults = UserDefaults.standard
    let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
      let stableValue = IPadInteractiveMediaBrowser.persistableSessionAddress(
        trimmed
      )
    else {
      defaults.removeObject(forKey: Self.mediaInputDefaultsKey)
      return
    }
    defaults.set(stableValue, forKey: Self.mediaInputDefaultsKey)
  }

  private func toggleCurrentBrowserBookmark() {
    guard !interactiveBrowser.isLoading, !interactiveBrowser.challengeActive,
      let pageURL = interactiveBrowser.currentPublicPageURL
    else { return }
    browserLibrary.toggleBookmark(
      url: pageURL,
      title: interactiveBrowser.pageTitle
    )
  }

  private func openBrowserLibraryEntry(
    _ entry: IPadBrowserLibraryStore.Entry
  ) {
    showingBrowserLibrary = nil
    guard let pageURL = entry.resolvedURL else { return }
    selectedTab = .browser
    interactiveBrowser.addressText = pageURL.absoluteString
    openBrowserAddress()
  }

  private func openBrowserAddress() {
    let rawValue = interactiveBrowser.addressText.trimmingCharacters(
      in: .whitespacesAndNewlines
    )
    guard !rawValue.isEmpty else { return }
    browserAddressFocused = false
    autoStartURLPlayback = false
    browserFailureMessage = nil
    invalidateURLAnalysis()
    invalidateBrowserAnalysis()
    realtimePlayer.stop()
    player?.pause()
    mediaURLText = rawValue
    interactiveBrowser.navigate(rawValue)
  }

  private func openBrowserForInteraction(_ rawValue: String) {
    browserAddressFocused = false
    invalidateBrowserAnalysis()
    browserFailureMessage = nil
    mediaURLText = rawValue
    selectedTab = .browser
    interactiveBrowser.addressText = rawValue
    interactiveBrowser.navigate(rawValue)
  }

  private func closeBrowserTab() {
    browserAddressFocused = false
    autoStartURLPlayback = false
    browserFailureMessage = nil
    invalidateURLAnalysis()
    invalidateBrowserAnalysis(closePage: true)
    selectedTab = .basic
  }

  private func analyzeBrowserCandidates() {
    guard scenePhase == .active, selectedTab == .browser,
      browserAnalysisTask == nil,
      !store.isRunning, !store.isResolvingURL,
      !interactiveBrowser.challengeActive, !interactiveBrowser.isLoading
    else { return }
    browserFailureMessage = nil
    // Starting a new browser analysis supersedes any previous realtime HLS
    // consumer. Stop it first so its lease enters the paired ending path.
    realtimePlayer.stop()
    invalidateBrowserAnalysis()
    retirePendingBrowserPlaybackHandoffLease()
    autoStartURLPlayback = true
    interactiveBrowser.activateInspection()
    let generation = browserAnalysisGeneration
    let navigationGeneration = interactiveBrowser.navigationGeneration
    let selectionOwnerID = UUID()
    browserAnalysisTask = Task { @MainActor in
      defer {
        if browserAnalysisGeneration == generation {
          browserAnalysisTask = nil
        }
      }
      guard await prepareWorkerForBrowserPlayback(
        generation: generation,
        navigationGeneration: navigationGeneration
      ) else { return }
      var lastAttemptedSourceRevision: Int?
      var lastAttemptedCandidateRevision: Int?
      var lastAttemptDate = Date()

      // The first click commonly starts only a poster or pre-roll. Keep the
      // visible document and its network observers alive until the real
      // player issues HLS, and re-rank whenever media/candidate revisions
      // change. Navigation, tab changes and backgrounding still cancel this
      // task through browserAnalysisGeneration.
      while !Task.isCancelled,
        browserAnalysisIsCurrent(generation, navigationGeneration)
      {
        if let preRollWait = interactiveBrowser.likelyPreRollWait,
          interactiveBrowser.isPreRollWaitCurrent(preRollWait)
        {
          browserFailureMessage =
            "短い先行動画の終了と本編への切り替えを待っています（監視中）…"
          do {
            try await Task.sleep(nanoseconds: 500_000_000)
          } catch {
            return
          }
          continue
        }

        let sourceRevision = interactiveBrowser.mediaSourceRevision
        let candidateRevision = interactiveBrowser.candidateRevision
        let hasNewEvidence =
          sourceRevision != lastAttemptedSourceRevision
          || candidateRevision != lastAttemptedCandidateRevision
        let periodicRetryDue = Date().timeIntervalSince(lastAttemptDate) >= 12
        guard hasNewEvidence || periodicRetryDue else {
          browserFailureMessage =
            "本編HLSを監視中です。ページ内の再生ボタンを押すと自動で解析します…"
          do {
            try await Task.sleep(nanoseconds: 500_000_000)
          } catch {
            return
          }
          continue
        }

        if periodicRetryDue {
          // Refresh the route observation epoch so long multi-stage players
          // cannot exhaust an old page-world observation window.
          interactiveBrowser.activateInspection()
        }
        lastAttemptedSourceRevision = sourceRevision
        lastAttemptedCandidateRevision = candidateRevision
        lastAttemptDate = Date()
        let candidates = await interactiveBrowser.snapshotCandidates()
        guard !Task.isCancelled,
          browserAnalysisIsCurrent(generation, navigationGeneration)
        else { return }
        guard !candidates.isEmpty else {
          browserFailureMessage =
            "配信URLを監視中です。ページ内の動画を再生すると自動で解析します…"
          do {
            try await Task.sleep(nanoseconds: 500_000_000)
          } catch {
            return
          }
          continue
        }
        var resolvingHandoffLease: IPadBrowserMediaHandoffLease?
        var resolvingResourceLoader: (any IPadHLSResourceLoading)?
        if store.hlsStreamingMode == .safariCompatible {
          do {
            let lease = try await interactiveBrowser
              .acquireMediaPlaybackHandoffLease(replacingActive: true)
            browserAnalysisHandoffLease = lease
            guard !Task.isCancelled,
              browserAnalysisIsCurrent(generation, navigationGeneration)
            else {
              await endBrowserAnalysisHandoffLease(lease)
              return
            }
            guard let loader = await lease.resourceLoader(
              for: candidates[0],
              isLive: false
            ) else {
              await endBrowserAnalysisHandoffLease(lease)
              browserFailureMessage =
                "Safari/WebKitのHLS通信を開始できませんでした。"
              return
            }
            resolvingHandoffLease = lease
            resolvingResourceLoader = loader
          } catch {
            browserFailureMessage =
              "Safari/WebKitのHLS通信を引き継げませんでした: \(error.localizedDescription)"
            return
          }
        }
        let accepted = await store.selectBrowserCandidates(
          candidates,
          selectionOwnerID: selectionOwnerID,
          hlsResourceLoader: resolvingResourceLoader
        )
        guard !Task.isCancelled,
          browserAnalysisIsCurrent(generation, navigationGeneration)
        else {
          if let resolvingHandoffLease {
            await endBrowserAnalysisHandoffLease(resolvingHandoffLease)
          }
          if accepted {
            store.clearResolvedURLInput(ownedBy: selectionOwnerID)
          }
          return
        }
        guard accepted else {
          if let resolvingHandoffLease {
            await endBrowserAnalysisHandoffLease(resolvingHandoffLease)
          }
          if store.urlInputRequiresInteraction, store.urlInteractionURL != nil {
            // A media/CDN challenge URL is meaningful only inside the current
            // WebKit document, frame and cookie context. Even its owning page
            // can become stale after a same-document player transition, so
            // never reload or replace the visible page from this result.
            browserFailureMessage =
              "配信側の確認応答を検出しました。表示中のページはそのまま維持しています。本編HLSを監視中です…"
          } else {
            browserFailureMessage =
              "本編URLの追加発行を待っています…ページ内動画が始まると自動で再解析します。"
          }
          do {
            try await Task.sleep(nanoseconds: 500_000_000)
          } catch {
            return
          }
          continue
        }

        if resolvingHandoffLease == nil {
          // Fast mode can keep observing the page while resolution runs. Give
          // a just-discovered pre-roll a short quiet window before handoff.
          try? await Task.sleep(nanoseconds: 750_000_000)
          guard !Task.isCancelled,
            browserAnalysisIsCurrent(generation, navigationGeneration)
          else {
            store.clearResolvedURLInput(ownedBy: selectionOwnerID)
            return
          }
          guard interactiveBrowser.mediaSourceRevision == sourceRevision,
            interactiveBrowser.candidateRevision == candidateRevision
          else {
            store.clearResolvedURLInput(ownedBy: selectionOwnerID)
            lastAttemptedSourceRevision = nil
            lastAttemptedCandidateRevision = nil
            browserFailureMessage =
              "本編への切り替えを検出したため、配信を再解析しています…"
            continue
          }
        }

        let handoffLease: IPadBrowserMediaHandoffLease
        if let resolvingHandoffLease {
          handoffLease = resolvingHandoffLease
        } else {
          do {
            handoffLease = try await interactiveBrowser
              .acquireMediaPlaybackHandoffLease(replacingActive: true)
            browserAnalysisHandoffLease = handoffLease
          } catch {
            store.clearResolvedURLInput(ownedBy: selectionOwnerID)
            browserFailureMessage =
              "Safari/WebKitのHLS通信を引き継げませんでした: \(error.localizedDescription)"
            return
          }
        }
        guard !Task.isCancelled,
          browserAnalysisIsCurrent(generation, navigationGeneration)
        else {
          await endBrowserAnalysisHandoffLease(handoffLease)
          store.clearResolvedURLInput(ownedBy: selectionOwnerID)
          return
        }
        let resourceLoader = await handoffLease.resourceLoader(
          for: candidates[0],
          isLive: store.resolvedMediaSource?.hlsPlaylist?.isLive == true,
          retainingResolvedResourceURL:
            store.resolvedMediaSource?.hlsPlaylist?.url
        )
        guard resourceLoader != nil else {
          await endBrowserAnalysisHandoffLease(handoffLease)
          store.clearResolvedURLInput(ownedBy: selectionOwnerID)
          browserFailureMessage =
            "Safari/WebKitのHLS通信を開始できませんでした。"
          return
        }
        guard !Task.isCancelled,
          browserAnalysisIsCurrent(generation, navigationGeneration)
        else {
          await endBrowserAnalysisHandoffLease(handoffLease)
          store.clearResolvedURLInput(ownedBy: selectionOwnerID)
          return
        }
        if browserAnalysisHandoffLease === handoffLease {
          browserAnalysisHandoffLease = nil
        }
        browserHandoffLease = handoffLease
        browserHLSResourceLoader = resourceLoader
        browserFailureMessage = nil
        pendingBrowserAnalysisResume = false
        browserAnalysisResumeTask?.cancel()
        browserAnalysisResumeTask = nil
        browserPlaybackTransitionInProgress = true
        selectedTab = .playback
        startAcceptedBrowserPlayback()
        return
      }
    }
  }

  private func prepareWorkerForBrowserPlayback(
    generation: Int,
    navigationGeneration: Int
  ) async -> Bool {
    browserFailureMessage = "復元モデルの準備完了を待っています…"
    switch worker.preparation {
    case .idle, .failed:
      await worker.prepareModels()
    case .validating:
      while case .validating = worker.preparation {
        do {
          try await Task.sleep(nanoseconds: 100_000_000)
        } catch {
          return false
        }
        guard !Task.isCancelled,
          browserAnalysisIsCurrent(generation, navigationGeneration)
        else { return false }
      }
    case .ready:
      break
    }
    guard !Task.isCancelled,
      browserAnalysisIsCurrent(generation, navigationGeneration)
    else { return false }
    guard worker.preparedWorker != nil else {
      browserFailureMessage =
        "復元モデルを準備できませんでした: "
        + (worker.startBlocker ?? "モデル検証に失敗しました。")
      return false
    }
    browserFailureMessage = "本編HLSを解析しています…"
    return true
  }

  private func startRealtimePreview() {
    guard !store.isRunning, let prepared = worker.preparedWorker else { return }
    do {
      player?.pause()
      let configuration = try store.realtimePreviewConfiguration(
        prepared: prepared,
        hlsResourceLoader: browserHLSResourceLoader
      )
      let handoffLease = browserHandoffLease
      browserHandoffLease = nil
      browserHLSResourceLoader = nil
      realtimePlayer.start(
        prepared: prepared,
        configuration: configuration,
        browserHandoffLease: handoffLease
      )
    } catch {
      if let handoffLease = browserHandoffLease {
        browserHandoffLease = nil
        browserHLSResourceLoader = nil
        handoffLease.beginEnding()
        Task { @MainActor in await handoffLease.end() }
      }
      realtimePlayer.reportFailure(error.localizedDescription)
    }
  }

  private func startSFTPStreamingPlayback(_ entry: MiohSFTPEntry) {
    guard activeSheet == .sftpInput, !store.isRunning else { return }
    if let unavailableReason = sftpStreamingUnavailableReason {
      sftp.reportStreamingPlaybackUnavailable(unavailableReason)
      return
    }
    cancelPendingSFTPStreamingSelection()
    sftpStreamingSelectionGeneration &+= 1
    let selectionGeneration = sftpStreamingSelectionGeneration
    autoStartURLPlayback = false
    invalidateURLAnalysis()
    invalidateBrowserAnalysis(closePage: true)
    realtimePlayer.stop()
    player?.pause()
    sftp.startStreaming(entry) { input in
      sftpStreamingSelectionTask = Task { @MainActor in
        guard !Task.isCancelled,
          selectionGeneration == sftpStreamingSelectionGeneration,
          scenePhase == .active,
          activeSheet == .sftpInput
        else {
          input.stop()
          return
        }
        guard await store.selectSFTPStreamingInput(input) else {
          if !Task.isCancelled,
            selectionGeneration == sftpStreamingSelectionGeneration,
            activeSheet == .sftpInput
          {
            sftp.reportStreamingSelectionFailure()
            sftpStreamingSelectionTask = nil
          }
          return
        }
        guard !Task.isCancelled,
          selectionGeneration == sftpStreamingSelectionGeneration,
          scenePhase == .active,
          activeSheet == .sftpInput
        else {
          store.clearSFTPStreamingInput()
          return
        }
        switch worker.preparation {
        case .idle, .failed:
          await worker.prepareModels()
        case .validating:
          while case .validating = worker.preparation {
            do {
              try await Task.sleep(nanoseconds: 100_000_000)
            } catch {
              break
            }
          }
        case .ready:
          break
        }
        guard !Task.isCancelled, worker.preparedWorker != nil else {
          autoStartURLPlayback = false
          store.clearSFTPStreamingInput()
          if !Task.isCancelled,
            selectionGeneration == sftpStreamingSelectionGeneration,
            activeSheet == .sftpInput
          {
            sftp.reportStreamingPlaybackUnavailable(
              worker.startBlocker ?? "復元モデルを準備できませんでした。"
            )
            sftpStreamingSelectionTask = nil
          }
          return
        }
        guard !Task.isCancelled,
          selectionGeneration == sftpStreamingSelectionGeneration,
          scenePhase == .active,
          activeSheet == .sftpInput,
          store.isSFTPStreamingInput
        else {
          store.clearSFTPStreamingInput()
          return
        }
        sftpStreamingSelectionTask = nil
        activeSheet = nil
        selectedTab = .playback
        autoStartURLPlayback = true
        tryAutoStartURLPlayback()
      }
    }
  }

  private var sftpStreamingUnavailableReason: String? {
    guard UIDevice.current.userInterfaceIdiom == .pad,
      !ProcessInfo.processInfo.isiOSAppOnMac
    else {
      return "ストリーミング復元再生には実機のiPadが必要です。"
    }
    #if targetEnvironment(simulator)
      return "ストリーミング復元再生はiPad実機で利用できます。"
    #else
      guard #available(iOS 27.0, *) else {
        return "ストリーミング復元再生にはiPadOS 27以降が必要です。"
      }
      return nil
    #endif
  }

  private func cancelPendingSFTPStreamingSelection() {
    guard let task = sftpStreamingSelectionTask else { return }
    sftpStreamingSelectionGeneration &+= 1
    sftpStreamingSelectionTask = nil
    task.cancel()
    store.clearSFTPStreamingInput()
  }

  private func analyzeURLAndStartPlayback() {
    let rawValue = mediaURLText.trimmingCharacters(
      in: .whitespacesAndNewlines
    )
    guard !rawValue.isEmpty else { return }
    realtimePlayer.stop()
    player?.pause()
    autoStartURLPlayback = true
    invalidateBrowserAnalysis(closePage: true)
    invalidateURLAnalysis()
    autoStartURLPlayback = true
    let generation = urlAnalysisGeneration
    let selectionOwnerID = UUID()
    urlAnalysisTask = Task {
      defer {
        if urlAnalysisGeneration == generation {
          urlAnalysisTask = nil
        }
      }
      let accepted = await store.selectURLInput(
        rawValue,
        selectionOwnerID: selectionOwnerID
      )
      guard !Task.isCancelled, urlAnalysisIsCurrent(generation) else {
        if accepted {
          store.clearResolvedURLInput(ownedBy: selectionOwnerID)
        }
        return
      }
      guard accepted else {
        autoStartURLPlayback = false
        if store.urlInputRequiresInteraction, scenePhase == .active {
          openBrowserForInteraction(
            store.urlInteractionURL?.absoluteString ?? rawValue
          )
        }
        return
      }
      guard !Task.isCancelled, urlAnalysisIsCurrent(generation) else {
        store.clearResolvedURLInput(ownedBy: selectionOwnerID)
        return
      }
      selectedTab = .playback
      tryAutoStartURLPlayback()
      if !autoStartURLPlayback { return }
      switch worker.preparation {
      case .idle, .failed:
        await worker.prepareModels()
      case .validating, .ready:
        break
      }
      guard !Task.isCancelled, urlAnalysisIsCurrent(generation) else {
        store.clearResolvedURLInput(ownedBy: selectionOwnerID)
        return
      }
      tryAutoStartURLPlayback()
    }
  }

  private func tryAutoStartURLPlayback() {
    guard scenePhase == .active, selectedTab == .playback,
      autoStartURLPlayback,
      !store.isResolvingURL, !store.isRunning,
      store.inputURL != nil,
      let prepared = worker.preparedWorker
    else { return }
    store.configure(with: prepared)
    autoStartURLPlayback = false
    startRealtimePreview()
  }

  /// Browser analysis has already validated the source, prepared the worker
  /// and transferred WebKit's HLS session by the time it reaches this path.
  /// Do not route the one-shot handoff through `tryAutoStartURLPlayback`: a
  /// SwiftUI tab write can become visible on the following update cycle, so
  /// reading `selectedTab` immediately used to leave Playback idle on sites
  /// whose player performs a final page transition (for example Supjav).
  private func startAcceptedBrowserPlayback() {
    guard scenePhase == .active, autoStartURLPlayback,
      !store.isResolvingURL, !store.isRunning,
      store.inputURL != nil, let prepared = worker.preparedWorker
    else { return }
    store.configure(with: prepared)
    autoStartURLPlayback = false
    startRealtimePreview()
  }

  private func invalidateURLAnalysis() {
    urlAnalysisGeneration &+= 1
    urlAnalysisTask?.cancel()
    urlAnalysisTask = nil
    store.cancelURLResolution()
  }

  private func invalidateBrowserAnalysis(
    closePage: Bool = false,
    preservingResumeIntent: Bool = false
  ) {
    autoStartURLPlayback = false
    if !preservingResumeIntent {
      pendingBrowserAnalysisResume = false
    }
    browserAnalysisResumeTask?.cancel()
    browserAnalysisResumeTask = nil
    browserAnalysisGeneration &+= 1
    browserAnalysisTask?.cancel()
    browserAnalysisTask = nil
    if let handoffLease = browserAnalysisHandoffLease {
      browserAnalysisHandoffLease = nil
      handoffLease.beginEnding()
      Task { @MainActor in await handoffLease.end() }
    }
    store.cancelURLResolution()
    if closePage {
      clearPersistedBrowserSession()
      interactiveBrowser.closePage()
    }
  }

  private func endBrowserAnalysisHandoffLease(
    _ handoffLease: IPadBrowserMediaHandoffLease
  ) async {
    if browserAnalysisHandoffLease === handoffLease {
      browserAnalysisHandoffLease = nil
    }
    handoffLease.beginEnding()
    await handoffLease.end()
  }

  private func retirePendingBrowserPlaybackHandoffLease() {
    guard let handoffLease = browserHandoffLease else { return }
    browserHandoffLease = nil
    browserHLSResourceLoader = nil
    handoffLease.beginEnding()
    Task { @MainActor in await handoffLease.end() }
  }

  private func resumePendingBrowserAnalysisIfReady() {
    guard pendingBrowserAnalysisResume,
      browserAnalysisResumeTask == nil,
      scenePhase == .active,
      selectedTab == .browser,
      !interactiveBrowser.challengeActive,
      !interactiveBrowser.challengeCompatibilityTimedOut,
      !interactiveBrowser.isLoading
    else { return }

    let expectedAnalysisGeneration = browserAnalysisGeneration
    let expectedNavigationGeneration = interactiveBrowser.navigationGeneration
    browserAnalysisResumeTask = Task { @MainActor in
      do {
        try await Task.sleep(nanoseconds: 300_000_000)
      } catch {
        return
      }
      guard !Task.isCancelled,
        pendingBrowserAnalysisResume,
        expectedAnalysisGeneration == browserAnalysisGeneration,
        expectedNavigationGeneration == interactiveBrowser.navigationGeneration,
        scenePhase == .active,
        selectedTab == .browser,
        !interactiveBrowser.challengeActive,
        !interactiveBrowser.challengeCompatibilityTimedOut,
        !interactiveBrowser.isLoading
      else {
        if expectedAnalysisGeneration == browserAnalysisGeneration {
          browserAnalysisResumeTask = nil
        }
        return
      }
      pendingBrowserAnalysisResume = false
      browserAnalysisResumeTask = nil
      browserFailureMessage = "確認完了後の配信を再解析しています…"
      analyzeBrowserCandidates()
    }
  }

  private func urlAnalysisIsCurrent(_ generation: Int) -> Bool {
    generation == urlAnalysisGeneration && scenePhase == .active
  }

  private func browserAnalysisIsCurrent(
    _ generation: Int,
    _ navigationGeneration: Int
  ) -> Bool {
    generation == browserAnalysisGeneration && scenePhase == .active
      && selectedTab == .browser
      && navigationGeneration == interactiveBrowser.navigationGeneration
      && !interactiveBrowser.challengeActive
      && !interactiveBrowser.isLoading
  }

  private func updatePlaybackProgress() {
    guard let player else { return }
    if !editingPlaybackPosition {
      let current = player.currentTime().seconds
      if current.isFinite, current >= 0 { playbackPosition = current }
    }
    let itemDuration = player.currentItem?.duration.seconds ?? 0
    if itemDuration.isFinite, itemDuration > 0 {
      playbackDuration = itemDuration
      if playbackPosition > itemDuration { playbackPosition = itemDuration }
    }
  }

  private func seekPlayback(to seconds: Double) {
    guard let player, seconds.isFinite else { return }
    let target = max(0, min(seconds, playbackDuration))
    player.seek(
      to: CMTime(seconds: target, preferredTimescale: 600),
      toleranceBefore: .zero,
      toleranceAfter: .zero
    )
  }

  private var stateColor: Color {
    switch store.state {
    case .idle: .secondary
    case .preparing, .restoring, .muxingAudio: .blue
    case .completed: .green
    case .failed: .red
    }
  }

  private var headerStatusColor: Color {
    if case .failed = store.state { return .red }
    return .secondary
  }

  private var headerStatusText: String {
    guard store.isRunning else { return store.state.label }
    return "\(store.state.label) \(Int((store.progress * 100).rounded()))%"
  }

  private func restorationLabel(_ identifier: String) -> String {
    switch identifier {
    case "basicvsrpp-v1.2-coreai-variable": "可変長（推奨）"
    case "basicvsrpp-v1.2-coreai": "固定 T18"
    case "basicvsrpp-v1.2-coreai-t36": "T36"
    case "basicvsrpp-v1.2-coreai-t90": "T90"
    default: identifier
    }
  }

  private func detectorLabel(_ identifier: String) -> String {
    identifier
      .replacingOccurrences(of: "-coreml", with: "")
      .replacingOccurrences(of: "-coreai", with: "")
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

  private func byteCount(_ value: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
  }

  private func sftpStreamingBitRate(_ bitsPerSecond: Double) -> String {
    let value = max(0, bitsPerSecond)
    if value >= 1_000_000_000 {
      return String(format: "SFTP %.2f Gbps", value / 1_000_000_000)
    }
    if value >= 1_000_000 {
      return String(format: "SFTP %.2f Mbps", value / 1_000_000)
    }
    if value >= 1_000 {
      return String(format: "SFTP %.0f Kbps", value / 1_000)
    }
    return String(format: "SFTP %.0f bps", value)
  }

  private func logColor(_ level: IPadStandaloneLogEntry.Level) -> Color {
    switch level {
    case .info: .secondary
    case .success: .green
    case .warning: .orange
    case .error: .red
    }
  }
}

private struct ActivityView: UIViewControllerRepresentable {
  let items: [Any]

  func makeUIViewController(context: Context) -> UIActivityViewController {
    UIActivityViewController(activityItems: items, applicationActivities: nil)
  }

  func updateUIViewController(
    _ uiViewController: UIActivityViewController,
    context: Context
  ) {}
}
