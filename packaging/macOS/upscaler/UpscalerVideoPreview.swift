// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AVFoundation
import AVKit
import Combine
import SwiftUI

struct UpscalerVideoSeekRequest: Equatable {
  let id = UUID()
  let seconds: Double
}

@MainActor
final class UpscalerVideoPreviewController: ObservableObject {
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
        let seconds = time.seconds
        if seconds.isFinite {
          self.currentSeconds = max(0, seconds)
        }
        self.isPlaying = self.player.timeControlStatus == .playing
      }
    }
  }

  deinit {
    if let timeObserver {
      player.removeTimeObserver(timeObserver)
    }
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
    let upperBound = max(0, duration)
    let clamped = min(max(0, seconds), upperBound)
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
      if currentSeconds >= duration - 0.01 {
        seek(to: 0, duration: duration)
      }
      player.play()
      isPlaying = true
    }
  }

  func pause() {
    player.pause()
    isPlaying = false
  }
}

struct UpscalerVideoPreview: View {
  let url: URL
  let duration: Double
  let start: Double
  let end: Double
  let isDisabled: Bool
  let seekRequest: UpscalerVideoSeekRequest?
  let setStart: (Double) -> Void
  let setEnd: (Double) -> Void

  @StateObject private var preview = UpscalerVideoPreviewController()

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      VideoPlayer(player: preview.player)
        .frame(maxWidth: .infinity)
        .frame(height: 320)
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
      .disabled(duration <= 0 || isDisabled)

      HStack(spacing: 10) {
        Button {
          preview.togglePlayback(duration: duration)
        } label: {
          Label(
            preview.isPlaying ? "一時停止" : "再生",
            systemImage: preview.isPlaying ? "pause.fill" : "play.fill"
          )
        }
        .disabled(duration <= 0 || isDisabled)

        Text(
          "\(VideoUpscaleController.timecode(preview.currentSeconds))"
            + " / \(VideoUpscaleController.timecode(duration))"
        )
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)

        Spacer()

        Button("開始位置へ") {
          preview.seek(to: start, duration: duration)
        }
        Button("ここを開始に") { setStart(preview.currentSeconds) }
          .buttonStyle(.borderedProminent)
        Button("ここを終了に") { setEnd(preview.currentSeconds) }
          .buttonStyle(.borderedProminent)
        Button("終了位置へ") {
          preview.seek(to: end, duration: duration)
        }
      }
      .controlSize(.small)

      Text("入力動画を直接再生しています。スライダーでシークし、現在位置を開始または終了に設定できます。")
        .font(.caption)
        .foregroundStyle(.secondary)
    }
    .onAppear { preview.load(url) }
    .onChange(of: url) { _, newURL in preview.load(newURL) }
    .onChange(of: seekRequest) { _, request in
      guard let request else { return }
      preview.seek(to: request.seconds, duration: duration)
    }
    .onChange(of: isDisabled) { _, disabled in
      if disabled { preview.pause() }
    }
    .onDisappear { preview.pause() }
  }

  private var safeDuration: Double {
    max(0.01, duration.isFinite ? duration : 0.01)
  }
}
