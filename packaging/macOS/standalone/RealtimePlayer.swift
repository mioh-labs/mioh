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
  let startupSegmentCount = 3
  let rebufferSegmentCount = 2
  let driftToleranceSeconds = 0.080

  private var worker: Process?
  private var workerInput: Pipe?
  private var stdoutPipe: Pipe?
  private var stderrPipe: Pipe?
  private var stdoutBuffer = ""
  private var generation = 0
  private var nextSequence = 0
  private var queuedSegments: [PreviewSegment] = []
  private var itemSegments: [ObjectIdentifier: PreviewSegment] = [:]
  private var notificationTokens: [NSObjectProtocol] = []
  private var timeObserver: Any?
  private var sessionDirectory: URL?
  private var requestedStartSeconds = 0.0
  private var shouldPlay = true
  private var generationHasStarted = false
  private var generationStartPending = false
  private weak var runner: RestorationRunner?

  init() {
    restoredPlayer.isMuted = true
    sourcePlayer.volume = 1
  }

  deinit {
    if let timeObserver {
      sourcePlayer.removeTimeObserver(timeObserver)
    }
    for token in notificationTokens {
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
      sessionDirectory = session
      generation = 0
      nextSequence = 0
      requestedStartSeconds = startSeconds
      shouldPlay = true
      generationHasStarted = false
      generationStartPending = false
      state = .loading
      errorMessage = ""
      sourcePlayer.replaceCurrentItem(with: AVPlayerItem(url: input))
      sourcePlayer.volume = muted ? 0 : Float(volume)

      outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
        let data = handle.availableData
        guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
        Task { @MainActor in self?.consumeWorkerOutput(text) }
      }
      errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
        let data = handle.availableData
        guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
        Task { @MainActor in self?.runner?.appendExternalLog(text) }
      }
      process.terminationHandler = { [weak self] completed in
        Task { @MainActor in
          guard let self, self.worker === completed else { return }
          self.stdoutPipe?.fileHandleForReading.readabilityHandler = nil
          self.stderrPipe?.fileHandleForReading.readabilityHandler = nil
          self.worker = nil
          if completed.terminationStatus != 0 && self.state != .idle {
            self.fail("プレビューワーカーが終了しました")
          }
        }
      }
      installTimeObserver()
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
    shouldPlay = state != .paused
    sourcePlayer.pause()
    restoredPlayer.pause()
    position = min(max(seconds, 0), duration)
    requestedStartSeconds = position
    generation += 1
    nextSequence = 0
    generationHasStarted = false
    generationStartPending = false
    clearRestoredQueue(deleteFiles: true)
    state = .seeking
    sourcePlayer.seek(
      to: CMTime(seconds: position, preferredTimescale: 600),
      toleranceBefore: .zero,
      toleranceAfter: .zero
    )
    sendCommand(["command": "seek", "position_ns": Int64(position * 1_000_000_000)])
  }

  func restartWithCurrentSettings(runner: RestorationRunner) {
    start(runner: runner, at: position)
  }

  func setVolume(_ value: Double) {
    volume = min(max(value, 0), 1)
    sourcePlayer.volume = muted ? 0 : Float(volume)
  }

  func setMuted(_ value: Bool) {
    muted = value
    sourcePlayer.volume = value ? 0 : Float(volume)
  }

  func stop() {
    shouldPlay = false
    generationHasStarted = false
    generationStartPending = false
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
      if queuedSegments.isEmpty {
        state = .ended
      } else {
        resumeIfBuffered(endOfFile: true)
      }
    case "error":
      fail([event.message, event.detail].compactMap { $0 }.joined(separator: ": "))
    default:
      break
    }
  }

  private func enqueue(_ segment: PreviewSegment) {
    guard segment.sequence == nextSequence else { return }
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
      Task { @MainActor in self?.finished(item: item) }
    }
    notificationTokens.append(token)
    updateBufferedDuration()
  }

  private func finished(item: AVPlayerItem) {
    guard let segment = itemSegments.removeValue(forKey: ObjectIdentifier(item)) else { return }
    try? FileManager.default.removeItem(at: segment.url)
    queuedSegments.removeAll { $0.sequence == segment.sequence }
    updateBufferedDuration()
    if queuedSegments.isEmpty && state == .playing {
      sourcePlayer.pause()
      restoredPlayer.pause()
      state = .buffering
    }
  }

  private func resumeIfBuffered(endOfFile: Bool = false) {
    guard shouldPlay else { return }
    guard state != .playing, !generationStartPending else { return }
    if state == .paused {
      startPlayersFromCurrentPosition()
      return
    }
    let required = generationHasStarted ? rebufferSegmentCount : startupSegmentCount
    guard queuedSegments.count >= required || (endOfFile && !queuedSegments.isEmpty) else {
      if state != .loading && state != .seeking { state = .buffering }
      return
    }
    if generationHasStarted {
      startPlayersFromCurrentPosition()
      return
    }

    let startingGeneration = generation
    generationStartPending = true
    sourcePlayer.seek(
      to: CMTime(seconds: requestedStartSeconds, preferredTimescale: 600),
      toleranceBefore: .zero,
      toleranceAfter: .zero
    ) { [weak self] _ in
      Task { @MainActor in
        guard let self else { return }
        guard self.generation == startingGeneration else { return }
        self.generationStartPending = false
        guard self.shouldPlay else { return }
        self.generationHasStarted = true
        self.startPlayersFromCurrentPosition()
      }
    }
  }

  private func startPlayersFromCurrentPosition() {
    sourcePlayer.play()
    restoredPlayer.play()
    state = .playing
  }

  private func installTimeObserver() {
    if let timeObserver {
      sourcePlayer.removeTimeObserver(timeObserver)
    }
    timeObserver = sourcePlayer.addPeriodicTimeObserver(
      forInterval: CMTime(seconds: 0.2, preferredTimescale: 600),
      queue: .main
    ) { [weak self] time in
      Task { @MainActor in self?.tick(sourceSeconds: time.seconds) }
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
    for token in notificationTokens { NotificationCenter.default.removeObserver(token) }
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
