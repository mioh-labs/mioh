import AVFoundation
import AVKit
import SwiftUI

@MainActor
final class MiohVideoPreviewController: ObservableObject {
  let player = AVPlayer()

  @Published private(set) var currentSeconds = 0.0
  @Published private(set) var isPlaying = false

  private var loadedURL: URL?
  private var timeObserver: Any?
  private var isScrubbing = false
  private var resumeAfterScrubbing = false

  init() {
    timeObserver = player.addPeriodicTimeObserver(
      forInterval: CMTime(seconds: 1.0 / 30.0, preferredTimescale: 60_000),
      queue: .main
    ) { [weak self] time in
      Task { @MainActor [weak self] in
        guard let self, !self.isScrubbing else { return }
        if time.seconds.isFinite { self.currentSeconds = max(0, time.seconds) }
        self.isPlaying = self.player.timeControlStatus == .playing
      }
    }
  }

  deinit {
    if let timeObserver { player.removeTimeObserver(timeObserver) }
  }

  func load(_ url: URL) {
    guard loadedURL != url else { return }
    player.pause()
    player.replaceCurrentItem(with: AVPlayerItem(url: url))
    player.actionAtItemEnd = .pause
    loadedURL = url
    currentSeconds = 0
    isPlaying = false
    isScrubbing = false
    resumeAfterScrubbing = false
  }

  func seek(to seconds: Double, duration: Double) {
    let clamped = min(max(0, seconds), max(0, duration))
    currentSeconds = clamped
    player.seek(
      to: CMTime(seconds: clamped, preferredTimescale: 60_000),
      toleranceBefore: .zero,
      toleranceAfter: .zero
    )
  }

  func setScrubbing(_ editing: Bool) {
    if editing, !isScrubbing {
      resumeAfterScrubbing = player.timeControlStatus == .playing
      isScrubbing = true
      player.pause()
      isPlaying = false
    } else if !editing, isScrubbing {
      isScrubbing = false
      if resumeAfterScrubbing {
        player.play()
        isPlaying = true
      }
      resumeAfterScrubbing = false
    }
  }

  func togglePlayback(duration: Double) {
    guard duration > 0 else { return }
    if player.timeControlStatus == .playing {
      player.pause()
      isPlaying = false
    } else {
      if currentSeconds >= duration - 0.01 { seek(to: 0, duration: duration) }
      player.play()
      isPlaying = true
    }
  }

  func pause() {
    player.pause()
    isPlaying = false
  }
}

struct MiohVideoRangeSelector: View {
  private enum TimeInputField: Hashable { case start, duration, end }

  let url: URL
  let duration: Double
  let start: Double
  let end: Double
  let isDisabled: Bool
  let setStart: (Double) -> Void
  let setDuration: (Double) -> Void
  let setEnd: (Double) -> Void
  let selectFull: () -> Void

  @StateObject private var preview = MiohVideoPreviewController()
  @FocusState private var focusedField: TimeInputField?

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      VideoPlayer(player: preview.player)
        .frame(maxWidth: .infinity)
        .frame(height: 300)
        .background(Color.black)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

      Slider(
        value: Binding(
          get: { min(max(0, preview.currentSeconds), safeDuration) },
          set: { preview.seek(to: $0, duration: duration) }
        ),
        in: 0...safeDuration,
        step: 0.01,
        onEditingChanged: preview.setScrubbing
      )

      HStack(spacing: 10) {
        Button {
          preview.togglePlayback(duration: duration)
        } label: {
          Label(
            preview.isPlaying ? "一時停止" : "再生",
            systemImage: preview.isPlaying ? "pause.fill" : "play.fill"
          )
        }
        Text("\(Self.timecode(preview.currentSeconds)) / \(Self.timecode(duration))")
          .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        Spacer()
        Button("開始位置へ") { preview.seek(to: start, duration: duration) }
        Button("ここを開始に") { setStart(preview.currentSeconds) }
          .buttonStyle(.borderedProminent)
        Button("ここを終了に") { setEnd(preview.currentSeconds) }
          .buttonStyle(.borderedProminent)
        Button("終了位置へ") { preview.seek(to: end, duration: duration) }
      }
      .controlSize(.small)

      MiohRangeTimeline(duration: duration, start: start, end: end)
      HStack {
        Text(Self.timecode(start))
        Spacer()
        Text("選択: \(Self.timecode(max(0, end - start)))").foregroundStyle(.secondary)
        Spacer()
        Text(Self.timecode(end))
      }
      .font(.caption.monospacedDigit())

      HStack {
        Text("共通目盛（動画全体）")
          .font(.caption.weight(.semibold))
        Text("先頭  \(Self.timecode(0))")
        Spacer()
        Text("末尾  \(Self.timecode(duration))")
      }
      .font(.caption.monospacedDigit())
      .foregroundStyle(.secondary)

      timeField("開始時刻", field: .start, value: start, minimum: 0,
                maximum: max(0.01, end - 0.05), setter: setStart)
      timeField("選択時間（長さ）", field: .duration, value: max(0, end - start),
                minimum: 0.05, maximum: max(0.05, duration - start), setter: setDuration)
      timeField("終了時刻", field: .end, value: end,
                minimum: min(duration, start + 0.05), maximum: max(0.01, duration), setter: setEnd)

      HStack {
        Button("動画全体を選択", action: selectFull)
        Spacer()
        Text("各スライダーの左端は動画の先頭、右端は末尾です。0.01秒単位で選択し、音声も同じ範囲へ切り詰めます。")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
    .disabled(isDisabled || duration <= 0)
    .onAppear { preview.load(url) }
    .onChange(of: url) { _, value in preview.load(value) }
    .onChange(of: isDisabled) { _, disabled in if disabled { preview.pause() } }
    .onDisappear { preview.pause() }
  }

  private func timeField(
    _ title: String,
    field: TimeInputField,
    value: Double,
    minimum: Double,
    maximum: Double,
    setter: @escaping (Double) -> Void
  ) -> some View {
    let safeMaximum = max(minimum + 0.01, maximum)
    let binding = Binding<Double>(
      get: { min(max(minimum, value), safeMaximum) },
      set: { setter(min(max(minimum, $0), safeMaximum)) }
    )
    return LabeledContent(title) {
      HStack(spacing: 12) {
        Slider(value: binding, in: 0...safeDuration, step: 0.01)
          .frame(minWidth: 240)
        Text(Self.timecode(value))
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
          .frame(width: 92, alignment: .trailing)
        TextField("秒数", value: binding,
                  format: .number.precision(.fractionLength(0...3)))
          .textFieldStyle(.roundedBorder)
          .font(.body.monospacedDigit())
          .multilineTextAlignment(.trailing)
          .frame(width: 96)
          .focused($focusedField, equals: field)
        Text("秒").foregroundStyle(.secondary)
      }
    }
  }

  private var safeDuration: Double { max(0.01, duration.isFinite ? duration : 0.01) }

  static func timecode(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds >= 0 else { return "00:00.00" }
    let centiseconds = Int((seconds * 100).rounded())
    let hours = centiseconds / 360_000
    let minutes = (centiseconds / 6_000) % 60
    let remainder = centiseconds % 6_000
    return hours > 0
      ? String(format: "%d:%02d:%02d.%02d", hours, minutes, remainder / 100, remainder % 100)
      : String(format: "%02d:%02d.%02d", minutes, remainder / 100, remainder % 100)
  }
}

private struct MiohRangeTimeline: View {
  let duration: Double
  let start: Double
  let end: Double

  var body: some View {
    GeometryReader { geometry in
      let safeDuration = max(0.001, duration)
      let startFraction = min(1, max(0, start / safeDuration))
      let endFraction = min(1, max(startFraction, end / safeDuration))
      ZStack(alignment: .leading) {
        Capsule().fill(Color.secondary.opacity(0.18))
        Capsule().fill(Color.accentColor)
          .frame(width: max(2, geometry.size.width * (endFraction - startFraction)))
          .offset(x: geometry.size.width * startFraction)
      }
    }
    .frame(height: 10)
  }
}
