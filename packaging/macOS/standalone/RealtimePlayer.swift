import AppKit
import AVFoundation
import AVKit
import Foundation
import SwiftUI

enum RealtimePlayerState: String {
  case idle
  case loading
  case buffering
  case playing
  case paused
  case seeking
  case ended
  case failed

  var label: String {
    switch self {
    case .idle: return "待機中"
    case .loading: return "モデルを読み込み中"
    case .buffering: return "バッファ中"
    case .playing: return "再生中"
    case .paused: return "一時停止"
    case .seeking: return "シーク中"
    case .ended: return "再生終了"
    case .failed: return "エラー"
    }
  }
}

struct PreviewWorkerEvent: Decodable {
  let kind: String
  let generation: Int
  let sequence: Int?
  let startNs: Int64?
  let endNs: Int64?
  let path: String?
  let duration: Double?
  let fps: Double?
  let width: Int?
  let height: Int?
  let message: String?
  let detail: String?
  let positionNs: Int64?
  let seconds: Double?
}

private struct PreviewSegment {
  let sequence: Int
  let startSeconds: Double
  let endSeconds: Double
  let url: URL
}

@MainActor
final class RealtimePlayerController: ObservableObject {
  @Published var state: RealtimePlayerState = .idle
  @Published var previewInputURL: URL?
  @Published var position = 0.0
  @Published var duration = 0.0
  @Published var bufferedSeconds = 0.0
  @Published var showOriginal = false
  @Published var volume = 1.0
  @Published var muted = false
  @Published var errorMessage = ""

  let sourcePlayer = AVPlayer()
  let restoredPlayer = AVQueuePlayer()
  let driftToleranceSeconds = 0.080
  let sourceSeekToleranceSeconds = 0.25
  private let sourceSeekWatchdogNanoseconds: UInt64 = 2_000_000_000
  private let sourceSeekWatchdogEpsilonSeconds = 0.05

  private var sourceSeekTolerance: CMTime {
    CMTime(seconds: sourceSeekToleranceSeconds, preferredTimescale: 600)
  }

  private var worker: Process?
  private var workerInput: Pipe?
  private var stdoutPipe: Pipe?
  private var stderrPipe: Pipe?
  private var stdoutBuffer = ""
  private var activeWorkerSessionToken: UUID?
  private var activeSourceSeekRequestToken: UUID?
  private var generation = 0
  private var nextSequence = 0
  private var queuedSegments: [PreviewSegment] = []
  private var itemSegments: [ObjectIdentifier: PreviewSegment] = [:]
  private var notificationTokens: [ObjectIdentifier: NSObjectProtocol] = [:]
  private var timeObserver: Any?
  private var sessionDirectory: URL?
  private var requestedStartSeconds = 0.0
  private var shouldPlay = true
  private var generationHasStarted = false
  private var generationStartPending = false
  private var generationSourceSeekCompleted = false
  private let maximumGenerationSourceSeekRetries = 2
  private var generationSourceSeekRetryCount = 0
  private var workerGenerationEnded = false
  private weak var runner: RestorationRunner?

  init() {
    restoredPlayer.isMuted = true
    sourcePlayer.volume = 1
  }

  deinit {
    if let timeObserver {
      sourcePlayer.removeTimeObserver(timeObserver)
    }
    for token in notificationTokens.values {
      NotificationCenter.default.removeObserver(token)
    }
  }

  func start(runner: RestorationRunner, at startSeconds: Double = 0) {
    stop()
    guard let input = previewInputURL else {
      fail("再生タブで入力動画を選択してください")
      return
    }
    do {
      let resources = try runner.resourceDirectory()
      let python = resources.appendingPathComponent("runtime/bin/python3.12")
      let script = resources.appendingPathComponent(
        "runtime/lib/python3.12/site-packages/mioh_preview_worker.py"
      )
      guard FileManager.default.isExecutableFile(atPath: python.path) else {
        throw RunnerError.missingResource("Python runtime")
      }
      guard FileManager.default.fileExists(atPath: script.path) else {
        throw RunnerError.missingResource("Realtime preview worker")
      }

      let tempRoot: URL
      if runner.ladaTempDirectory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        tempRoot = FileManager.default.temporaryDirectory
      } else {
        tempRoot = URL(fileURLWithPath: runner.ladaTempDirectory, isDirectory: true)
      }
      let session = tempRoot.appendingPathComponent(
        "mioh-preview-\(UUID().uuidString)", isDirectory: true
      )
      try FileManager.default.createDirectory(at: session, withIntermediateDirectories: true)

      let process = Process()
      let inputPipe = Pipe()
      let outputPipe = Pipe()
      let errorPipe = Pipe()
      let sessionToken = UUID()
      process.executableURL = python
      process.arguments = [script.path] + (try runner.previewArguments(
        resources: resources,
        outputDirectory: session,
        input: input
      )) + ["--start-ns", String(Int64(startSeconds * 1_000_000_000))]
      process.environment = runner.environment(resources: resources, python: python)
      process.standardInput = inputPipe
      process.standardOutput = outputPipe
      process.standardError = errorPipe

      self.runner = runner
      worker = process
      workerInput = inputPipe
      stdoutPipe = outputPipe
      stderrPipe = errorPipe
      activeWorkerSessionToken = sessionToken
      stdoutBuffer = ""
      sessionDirectory = session
      generation = 0
      nextSequence = 0
      requestedStartSeconds = startSeconds
      shouldPlay = true
      generationHasStarted = false
      generationStartPending = false
      generationSourceSeekCompleted = false
      generationSourceSeekRetryCount = 0
      activeSourceSeekRequestToken = nil
      workerGenerationEnded = false
      state = .loading
      errorMessage = ""
      sourcePlayer.replaceCurrentItem(with: AVPlayerItem(url: input))
      sourcePlayer.volume = muted ? 0 : Float(volume)

      outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
        let data = handle.availableData
        guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
        Task { @MainActor in
          guard let self, self.activeWorkerSessionToken == sessionToken else { return }
          self.consumeWorkerOutput(text)
        }
      }
      errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
        let data = handle.availableData
        guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
        Task { @MainActor in
          guard let self, self.activeWorkerSessionToken == sessionToken else { return }
          self.runner?.appendExternalLog(text)
        }
      }
      process.terminationHandler = { [weak self] completed in
        Task { @MainActor in
          guard let self,
            self.activeWorkerSessionToken == sessionToken,
            self.worker === completed
          else { return }
          self.stdoutPipe?.fileHandleForReading.readabilityHandler = nil
          self.stderrPipe?.fileHandleForReading.readabilityHandler = nil
          self.worker = nil
          if completed.terminationStatus != 0 && self.state != .idle {
            self.fail("プレビューワーカーが終了しました")
          }
        }
      }
      installTimeObserver(sessionToken: sessionToken)
      try process.run()
    } catch {
      fail(error.localizedDescription)
      cleanupSession()
    }
  }

  func togglePlayback() {
    if state == .playing {
      shouldPlay = false
      sourcePlayer.pause()
      restoredPlayer.pause()
      state = .paused
    } else if state == .paused || state == .buffering {
      shouldPlay = true
      resumeIfBuffered()
    } else if state == .idle || state == .ended || state == .failed, let runner {
      start(runner: runner, at: position)
    }
  }

  func choosePreviewInput() {
    let panel = NSOpenPanel()
    panel.title = "再生動画を選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    stop()
    previewInputURL = url
    position = 0
    duration = 0
    errorMessage = ""
    sourcePlayer.replaceCurrentItem(with: AVPlayerItem(url: url))
  }

  func seek(to seconds: Double) {
    guard worker != nil else {
      position = seconds
      return
    }
    guard let seekSessionToken = activeWorkerSessionToken else { return }
    shouldPlay = state != .paused
    sourcePlayer.pause()
    restoredPlayer.pause()
    position = min(max(seconds, 0), duration)
    requestedStartSeconds = position
    generation += 1
    let seekGeneration = generation
    nextSequence = 0
    generationHasStarted = false
    generationStartPending = true
    generationSourceSeekCompleted = false
    generationSourceSeekRetryCount = 0
    let seekRequestToken = UUID()
    let seekTargetSeconds = position
    let seekWatchdogNanoseconds = sourceSeekWatchdogNanoseconds
    activeSourceSeekRequestToken = seekRequestToken
    workerGenerationEnded = false
    clearRestoredQueue(deleteFiles: true)
    state = .seeking
    sourcePlayer.seek(
      to: CMTime(seconds: position, preferredTimescale: 600),
      toleranceBefore: sourceSeekTolerance,
      toleranceAfter: sourceSeekTolerance
    ) { [weak self] finished in
      Task { @MainActor in
        guard let self,
          self.activeWorkerSessionToken == seekSessionToken,
          self.generation == seekGeneration,
          self.activeSourceSeekRequestToken == seekRequestToken
        else { return }
        self.activeSourceSeekRequestToken = nil
        self.generationStartPending = false
        guard self.shouldPlay, self.state != .failed, self.state != .ended else { return }
        guard finished else {
          guard self.generationSourceSeekRetryCount < self.maximumGenerationSourceSeekRetries else {
            self.fail("プレビューのシークに繰り返し失敗しました")
            return
          }
          self.generationSourceSeekRetryCount += 1
          self.state = .buffering
          self.resumeIfBuffered()
          return
        }
        self.generationSourceSeekRetryCount = 0
        self.generationSourceSeekCompleted = true
        self.resumeIfBuffered()
      }
    }
    Task { @MainActor [weak self] in
      try? await Task.sleep(nanoseconds: seekWatchdogNanoseconds)
      self?.resolveStalledSourceSeek(
        targetSeconds: seekTargetSeconds,
        sessionToken: seekSessionToken,
        seekGeneration: seekGeneration,
        requestToken: seekRequestToken
      )
    }
    sendCommand(["command": "seek", "position_ns": Int64(position * 1_000_000_000)])
  }

  func restartWithCurrentSettings(runner: RestorationRunner) {
    start(runner: runner, at: position)
  }

  func setVolume(_ value: Double) {
    volume = min(max(value, 0), 1)
    sourcePlayer.volume = muted ? 0 : Float(volume)
  }

  func setBufferLimit(_ seconds: Double) {
    guard worker != nil else { return }
    sendCommand(["command": "set_buffer_limit", "seconds": seconds])
    bufferPolicyDidChange()
  }

  func bufferPolicyDidChange() {
    guard worker != nil else { return }
    resumeIfBuffered()
  }

  func setMuted(_ value: Bool) {
    muted = value
    sourcePlayer.volume = value ? 0 : Float(volume)
  }

  func stop() {
    shouldPlay = false
    generationHasStarted = false
    generationStartPending = false
    generationSourceSeekCompleted = false
    generationSourceSeekRetryCount = 0
    activeSourceSeekRequestToken = nil
    workerGenerationEnded = false
    activeWorkerSessionToken = nil
    stdoutBuffer = ""
    sourcePlayer.pause()
    restoredPlayer.pause()
    if worker != nil {
      sendCommand(["command": "stop"])
      worker?.terminate()
    }
    stdoutPipe?.fileHandleForReading.readabilityHandler = nil
    stderrPipe?.fileHandleForReading.readabilityHandler = nil
    worker = nil
    workerInput = nil
    stdoutPipe = nil
    stderrPipe = nil
    clearRestoredQueue(deleteFiles: true)
    cleanupSession()
    state = .idle
    bufferedSeconds = 0
  }

  private func consumeWorkerOutput(_ text: String) {
    stdoutBuffer += text
    while let newline = stdoutBuffer.firstIndex(of: "\n") {
      let line = String(stdoutBuffer[..<newline])
      stdoutBuffer.removeSubrange(...newline)
      guard let data = line.data(using: .utf8) else { continue }
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      do {
        let event = try decoder.decode(PreviewWorkerEvent.self, from: data)
        handle(event)
      } catch {
        runner?.appendExternalLog("Invalid preview event: \(line)\n")
      }
    }
  }

  private func handle(_ event: PreviewWorkerEvent) {
    guard event.generation == generation else { return }
    switch event.kind {
    case "ready":
      duration = event.duration ?? 0
      state = .buffering
    case "segment":
      guard let sequence = event.sequence,
        let startNs = event.startNs,
        let endNs = event.endNs,
        let path = event.path
      else { return }
      let segment = PreviewSegment(
        sequence: sequence,
        startSeconds: Double(startNs) / 1_000_000_000,
        endSeconds: Double(endNs) / 1_000_000_000,
        url: URL(fileURLWithPath: path)
      )
      enqueue(segment)
      resumeIfBuffered()
    case "ended":
      workerGenerationEnded = true
      if queuedSegments.isEmpty {
        shouldPlay = false
        sourcePlayer.pause()
        restoredPlayer.pause()
        state = .ended
      } else {
        resumeIfBuffered()
      }
    case "error":
      fail([event.message, event.detail].compactMap { $0 }.joined(separator: ": "))
    case "buffer_limit":
      guard let seconds = event.seconds else { return }
      runner?.appendExternalLog("プレビューバッファ上限を適用: \(Int(seconds))秒\n")
    default:
      break
    }
  }

  private func enqueue(_ segment: PreviewSegment) {
    guard segment.sequence == nextSequence else { return }
    guard let itemSessionToken = activeWorkerSessionToken else { return }
    let itemGeneration = generation
    nextSequence += 1
    let item = AVPlayerItem(url: segment.url)
    let identifier = ObjectIdentifier(item)
    itemSegments[identifier] = segment
    queuedSegments.append(segment)
    restoredPlayer.insert(item, after: nil)
    let token = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemDidPlayToEndTime,
      object: item,
      queue: .main
    ) { [weak self] _ in
      Task { @MainActor in
        guard let self,
          self.activeWorkerSessionToken == itemSessionToken,
          self.generation == itemGeneration
        else { return }
        self.finished(item: item)
      }
    }
    notificationTokens[identifier] = token
    updateBufferedDuration()
  }

  private func finished(item: AVPlayerItem) {
    let identifier = ObjectIdentifier(item)
    if let token = notificationTokens.removeValue(forKey: identifier) {
      NotificationCenter.default.removeObserver(token)
    }
    guard let segment = itemSegments.removeValue(forKey: identifier) else { return }
    try? FileManager.default.removeItem(at: segment.url)
    queuedSegments.removeAll { $0.sequence == segment.sequence }
    updateBufferedDuration()
    if queuedSegments.isEmpty {
      if workerGenerationEnded {
        sourcePlayer.pause()
        restoredPlayer.pause()
        shouldPlay = false
        state = .ended
      } else if state == .playing {
        sourcePlayer.pause()
        restoredPlayer.pause()
        state = .buffering
      }
    }
  }

  private func resumeIfBuffered() {
    guard shouldPlay else { return }
    guard state != .failed, state != .ended else { return }
    guard state != .playing, !generationStartPending else { return }
    if state == .paused {
      startPlayersFromCurrentPosition()
      return
    }
    guard let runner else { return }
    let ready = PreviewBufferPolicy.canStart(
      bufferedSeconds: bufferedSeconds,
      selectedBufferLimit: runner.previewBufferLimit,
      generationHasStarted: generationHasStarted,
      shortenRebuffer: runner.previewShortenedRebuffer,
      endOfFile: workerGenerationEnded,
      hasQueuedSegments: !queuedSegments.isEmpty
    )
    guard ready else {
      if state != .loading && state != .seeking { state = .buffering }
      return
    }
    if generationHasStarted {
      startPlayersFromCurrentPosition()
      return
    }
    if generationSourceSeekCompleted {
      generationHasStarted = true
      startPlayersFromCurrentPosition()
      return
    }

    guard let startingSessionToken = activeWorkerSessionToken else { return }
    let startingGeneration = generation
    let startingSeekRequestToken = UUID()
    let startingTargetSeconds = requestedStartSeconds
    let seekWatchdogNanoseconds = sourceSeekWatchdogNanoseconds
    generationStartPending = true
    activeSourceSeekRequestToken = startingSeekRequestToken
    sourcePlayer.seek(
      to: CMTime(seconds: requestedStartSeconds, preferredTimescale: 600),
      toleranceBefore: sourceSeekTolerance,
      toleranceAfter: sourceSeekTolerance
    ) { [weak self] finished in
      Task { @MainActor in
        guard let self,
          self.activeWorkerSessionToken == startingSessionToken,
          self.generation == startingGeneration,
          self.activeSourceSeekRequestToken == startingSeekRequestToken
        else { return }
        self.activeSourceSeekRequestToken = nil
        self.generationStartPending = false
        guard self.shouldPlay, self.state != .failed, self.state != .ended else { return }
        guard finished else {
          guard self.generationSourceSeekRetryCount < self.maximumGenerationSourceSeekRetries else {
            self.fail("プレビューのシークに繰り返し失敗しました")
            return
          }
          self.generationSourceSeekRetryCount += 1
          self.state = .buffering
          self.resumeIfBuffered()
          return
        }
        self.generationSourceSeekRetryCount = 0
        self.generationSourceSeekCompleted = true
        guard self.shouldPlay else { return }
        guard let runner = self.runner else { return }
        let latestPolicyAllowsPlayback = PreviewBufferPolicy.canStart(
          bufferedSeconds: self.bufferedSeconds,
          selectedBufferLimit: runner.previewBufferLimit,
          generationHasStarted: self.generationHasStarted,
          shortenRebuffer: runner.previewShortenedRebuffer,
          endOfFile: self.workerGenerationEnded,
          hasQueuedSegments: !self.queuedSegments.isEmpty
        )
        guard latestPolicyAllowsPlayback else {
          self.state = .buffering
          return
        }
        self.generationHasStarted = true
        self.startPlayersFromCurrentPosition()
      }
    }
    Task { @MainActor [weak self] in
      try? await Task.sleep(nanoseconds: seekWatchdogNanoseconds)
      self?.resolveStalledSourceSeek(
        targetSeconds: startingTargetSeconds,
        sessionToken: startingSessionToken,
        seekGeneration: startingGeneration,
        requestToken: startingSeekRequestToken
      )
    }
  }

  private func resolveStalledSourceSeek(
    targetSeconds: Double,
    sessionToken: UUID,
    seekGeneration: Int,
    requestToken: UUID
  ) {
    guard activeSourceSeekRequestToken == requestToken,
      activeWorkerSessionToken == sessionToken,
      generation == seekGeneration,
      generationStartPending
    else { return }
    activeSourceSeekRequestToken = nil
    generationStartPending = false
    guard shouldPlay, state != .failed, state != .ended else { return }

    let currentSeconds = sourcePlayer.currentTime().seconds
    guard currentSeconds.isFinite,
      abs(currentSeconds - targetSeconds) <=
        sourceSeekToleranceSeconds + sourceSeekWatchdogEpsilonSeconds
    else {
      guard generationSourceSeekRetryCount < maximumGenerationSourceSeekRetries else {
        fail("プレビューのシークに繰り返し失敗しました")
        return
      }
      generationSourceSeekRetryCount += 1
      state = .buffering
      resumeIfBuffered()
      return
    }

    generationSourceSeekRetryCount = 0
    generationSourceSeekCompleted = true
    resumeIfBuffered()
  }

  private func startPlayersFromCurrentPosition() {
    sourcePlayer.play()
    restoredPlayer.play()
    state = .playing
  }

  private func installTimeObserver(sessionToken: UUID) {
    if let timeObserver {
      sourcePlayer.removeTimeObserver(timeObserver)
    }
    timeObserver = sourcePlayer.addPeriodicTimeObserver(
      forInterval: CMTime(seconds: 0.2, preferredTimescale: 600),
      queue: .main
    ) { [weak self] time in
      Task { @MainActor in
        guard let self, self.activeWorkerSessionToken == sessionToken else { return }
        self.tick(sourceSeconds: time.seconds)
      }
    }
  }

  private func tick(sourceSeconds: Double) {
    guard sourceSeconds.isFinite else { return }
    position = sourceSeconds
    updateBufferedDuration()
    guard state == .playing,
      let active = queuedSegments.first,
      restoredPlayer.currentTime().seconds.isFinite
    else { return }
    let restoredAbsolute = active.startSeconds + restoredPlayer.currentTime().seconds
    if abs(restoredAbsolute - sourceSeconds) > driftToleranceSeconds {
      let local = max(0, sourceSeconds - active.startSeconds)
      restoredPlayer.seek(
        to: CMTime(seconds: local, preferredTimescale: 600),
        toleranceBefore: .zero,
        toleranceAfter: .zero
      )
    }
  }

  private func updateBufferedDuration() {
    guard let last = queuedSegments.last else {
      bufferedSeconds = 0
      return
    }
    bufferedSeconds = max(0, last.endSeconds - position)
  }

  private func sendCommand(_ payload: [String: Any]) {
    guard let handle = workerInput?.fileHandleForWriting,
      let data = try? JSONSerialization.data(withJSONObject: payload),
      var line = String(data: data, encoding: .utf8)?.data(using: .utf8)
    else { return }
    line.append(0x0A)
    try? handle.write(contentsOf: line)
  }

  private func clearRestoredQueue(deleteFiles: Bool) {
    restoredPlayer.removeAllItems()
    for token in notificationTokens.values { NotificationCenter.default.removeObserver(token) }
    notificationTokens.removeAll()
    if deleteFiles {
      for segment in queuedSegments { try? FileManager.default.removeItem(at: segment.url) }
    }
    queuedSegments.removeAll()
    itemSegments.removeAll()
    bufferedSeconds = 0
  }

  private func cleanupSession() {
    guard let sessionDirectory else { return }
    try? FileManager.default.removeItem(at: sessionDirectory)
    self.sessionDirectory = nil
  }

  private func fail(_ message: String) {
    shouldPlay = false
    sourcePlayer.pause()
    restoredPlayer.pause()
    errorMessage = message
    state = .failed
    runner?.appendExternalLog("Realtime preview: \(message)\n")
  }
}

struct RealtimePlayerView: View {
  @ObservedObject var controller: RealtimePlayerController
  @ObservedObject var runner: RestorationRunner
  @State private var seekPosition = 0.0

  var body: some View {
    VStack(spacing: 12) {
      PathRow(title: "再生動画", icon: "film", url: controller.previewInputURL, action: controller.choosePreviewInput)

      ZStack {
        Color.black
        VideoPlayer(player: controller.sourcePlayer)
          .opacity(controller.showOriginal ? 1 : 0.001)
        VideoPlayer(player: controller.restoredPlayer)
          .opacity(controller.showOriginal ? 0.001 : 1)
        if controller.state == .loading || controller.state == .buffering || controller.state == .seeking {
          VStack(spacing: 8) {
            ProgressView()
            Text("バッファ中")
          }
          .padding(18)
          .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 10))
          .foregroundStyle(.white)
        }
      }
      .aspectRatio(16 / 9, contentMode: .fit)
      .clipShape(RoundedRectangle(cornerRadius: 8))

      HStack {
        Text(time(controller.position))
          .font(.caption.monospacedDigit()).frame(width: 68)
        Slider(
          value: Binding(
            get: { controller.position },
            set: { seekPosition = $0 }
          ),
          in: 0...max(controller.duration, 0.01),
          onEditingChanged: { editing in
            if !editing { controller.seek(to: seekPosition) }
          }
        )
        Text(time(controller.duration))
          .font(.caption.monospacedDigit()).frame(width: 68)
      }

      HStack(spacing: 12) {
        Text("バッファ上限")
        Slider(
          value: Binding(
            get: { runner.previewBufferLimit },
            set: { value in
              runner.previewBufferLimit = value
              controller.setBufferLimit(value)
            }
          ),
          in: 1...60,
          step: 1
        )
        .frame(maxWidth: 320)
        Text("\(Int(runner.previewBufferLimit))秒")
          .font(.caption.monospacedDigit())
          .frame(width: 48, alignment: .trailing)
        Toggle(
          "再バッファを短縮",
          isOn: Binding(
            get: { runner.previewShortenedRebuffer },
            set: { value in
              runner.previewShortenedRebuffer = value
              controller.bufferPolicyDidChange()
            }
          )
        )
        .toggleStyle(.checkbox)
        .help("再生途中のバッファ切れだけ最大4秒で復帰します")
        Spacer()
      }

      HStack(spacing: 12) {
        if controller.state == .idle || controller.state == .ended || controller.state == .failed {
          Button { controller.start(runner: runner) } label: { Label("再生", systemImage: "play.fill") }
            .buttonStyle(.borderedProminent).disabled(controller.previewInputURL == nil)
        } else if controller.state == .playing {
          Button(action: controller.togglePlayback) { Label("一時停止", systemImage: "pause.fill") }
        } else {
          Button(action: controller.togglePlayback) { Label("再生", systemImage: "play.fill") }
        }
        Button(role: .destructive, action: controller.stop) { Label("停止", systemImage: "stop.fill") }
          .disabled(controller.state == .idle)
        Toggle("処理前", isOn: $controller.showOriginal).toggleStyle(.switch)
        Spacer()
        Text("先読み \(controller.bufferedSeconds, specifier: "%.1f")秒")
          .font(.caption).foregroundStyle(.secondary)
        Button {
          controller.setMuted(!controller.muted)
        } label: {
          Image(systemName: controller.muted ? "speaker.slash.fill" : "speaker.wave.2.fill")
        }.buttonStyle(.borderless)
        Slider(
          value: Binding(get: { controller.volume }, set: controller.setVolume),
          in: 0...1
        ).frame(width: 100)
        Button("設定を反映して再開") { controller.restartWithCurrentSettings(runner: runner) }
          .disabled(controller.state == .idle)
      }
      if !controller.errorMessage.isEmpty {
        Text(controller.errorMessage).foregroundStyle(.red).font(.caption)
      } else {
        Text(controller.state.label).font(.caption).foregroundStyle(.secondary)
      }
    }
    .padding(.vertical, 12)
  }

  private func time(_ seconds: Double) -> String {
    guard seconds.isFinite else { return "00:00" }
    let value = max(0, Int(seconds))
    return String(format: "%02d:%02d", value / 60, value % 60)
  }
}
