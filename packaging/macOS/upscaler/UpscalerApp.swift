// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AppKit
import AVKit
import SwiftUI

@main
struct MiohUpscalerApp: App {
  var body: some Scene {
    WindowGroup("mioh upscaler") {
      UpscalerContentView()
    }
    .defaultSize(width: 940, height: 780)
    .windowResizability(.contentSize)
  }
}

private struct UpscalerContentView: View {
  private enum TimeInputField: Hashable {
    case start
    case duration
    case end
  }

  @StateObject private var upscaler = VideoUpscaleController()
  @StateObject private var h3Generation = MiniMaxH3Controller()
  @StateObject private var modelSetup = UpscalerModelSetupController()
  @State private var showLog = false
  @State private var selectedTab = UpscalerWorkspaceTab.upscale
  @State private var previewSeekRequest: UpscalerVideoSeekRequest?
  @State private var showingModelSetup = false
  @State private var checkedInitialModelSetup = false
  @FocusState private var focusedTimeInput: TimeInputField?

  var body: some View {
    TabView(selection: $selectedTab) {
      upscaleWorkspace
        .tabItem {
          Label("アップスケール", systemImage: "arrow.up.left.and.arrow.down.right")
        }
        .tag(UpscalerWorkspaceTab.upscale)
      videoGenerationWorkspace
        .tabItem {
          Label("動画生成", systemImage: "sparkles.rectangle.stack")
        }
        .tag(UpscalerWorkspaceTab.videoGeneration)
    }
    .frame(minWidth: 820, minHeight: 680)
    .onAppear {
      guard !checkedInitialModelSetup else { return }
      checkedInitialModelSetup = true
      if !upscaler.selectedModelReady {
        presentModelSetup()
      }
    }
    .onChange(of: upscaler.upscalerModel) { _, _ in
      if !upscaler.selectedModelReady {
        presentModelSetup()
      }
    }
    .sheet(isPresented: $showingModelSetup) {
      UpscalerModelSetupView(controller: modelSetup) {
        upscaler.applyModelSetupDestination(modelSetup.destinationPath)
      }
    }
  }

  private var upscaleWorkspace: some View {
    VStack(spacing: 0) {
      header
      Divider()
      ScrollView {
        Form {
          videoSection
          rangeSection
          modelSection
          outputSection
          if showLog { logSection }
        }
        .formStyle(.grouped)
        .padding(.horizontal, 10)
      }
      Divider()
      progressArea
      Divider()
      footer
    }
  }

  private var videoGenerationWorkspace: some View {
    VStack(spacing: 0) {
      videoGenerationHeader
      Divider()
      MiniMaxH3GenerationView(
        controller: h3Generation,
        upscalerInputURL: upscaler.inputURL
      )
      Divider()
      videoGenerationFooter
    }
  }

  private var videoGenerationHeader: some View {
    HStack(spacing: 12) {
      applicationIcon
      VStack(alignment: .leading, spacing: 2) {
        Text("動画生成").font(.title2.weight(.semibold))
        Text("MiniMax H3 / 10Eros-Max H3")
          .font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
      Text(h3Generation.status)
        .font(.callout.monospacedDigit())
        .foregroundStyle(
          h3Generation.status.contains("失敗") ? .red : .secondary
        )
    }
    .padding(.horizontal, 20)
    .frame(height: 66)
  }

  private var videoGenerationFooter: some View {
    HStack(spacing: 10) {
      Button(action: h3Generation.revealOutput) {
        Image(systemName: "folder.badge.gearshape")
      }
      .help("出力をFinderで表示")
      .disabled(h3Generation.outputPath.isEmpty)
      Spacer()
      if h3Generation.isRunning {
        Button(role: .destructive, action: h3Generation.stop) {
          Label("停止", systemImage: "stop.fill")
        }
      } else {
        Button(action: h3Generation.start) {
          Label("動画生成を開始", systemImage: "sparkles")
        }
        .buttonStyle(.borderedProminent)
        .disabled(!h3Generation.canStart() || upscaler.isRunning)
      }
    }
    .padding(.horizontal, 20)
    .frame(height: 58)
  }

  private var header: some View {
    HStack(spacing: 12) {
      applicationIcon
      VStack(alignment: .leading, spacing: 2) {
        Text("mioh upscaler").font(.title2.weight(.semibold))
        Text("FlashVSR Tiny / AdcSR Core AI")
          .font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
      Text(upscaler.status)
        .font(.callout.monospacedDigit())
        .foregroundStyle(upscaler.status == "エラー" ? .red : .secondary)
    }
    .padding(.horizontal, 20)
    .frame(height: 66)
  }

  private var videoSection: some View {
    Section("動画") {
      UpscalerPathRow(
        title: "入力動画", icon: "film.stack", url: upscaler.inputURL,
        action: upscaler.chooseInput, actionLabel: "動画を選択…"
      )
      if upscaler.inputURL != nil {
        UpscalerSourceInfoRow(
          info: upscaler.sourceInfo, failure: upscaler.sourceInfoFailure
        )
      }
    }
  }

  private var rangeSection: some View {
    Section("アップスケール範囲") {
      if let inputURL = upscaler.inputURL {
        UpscalerVideoPreview(
          url: inputURL,
          duration: upscaler.durationSeconds,
          start: upscaler.normalizedStartSeconds,
          end: upscaler.normalizedEndSeconds,
          isDisabled: upscaler.isRunning,
          seekRequest: previewSeekRequest,
          setStart: upscaler.setStartSeconds,
          setEnd: upscaler.setEndSeconds
        )
      }
      UpscalerRangeTimeline(
        duration: upscaler.durationSeconds,
        start: upscaler.normalizedStartSeconds,
        end: upscaler.normalizedEndSeconds
      )
      HStack {
        Text(VideoUpscaleController.timecode(upscaler.normalizedStartSeconds))
        Spacer()
        Text("選択: \(VideoUpscaleController.timecode(upscaler.selectedDurationSeconds))")
          .foregroundStyle(.secondary)
        Spacer()
        Text(VideoUpscaleController.timecode(upscaler.normalizedEndSeconds))
      }
      .font(.caption.monospacedDigit())
      timeField(
        "開始",
        field: .start,
        value: Binding(
          get: { upscaler.normalizedStartSeconds },
          set: { upscaler.setStartSeconds($0) }
        ),
        onCommit: {
          requestPreviewSeek(to: upscaler.normalizedStartSeconds)
        }
      )
      timeField(
        "範囲時間",
        field: .duration,
        value: Binding(
          get: { upscaler.selectedDurationSeconds },
          set: { upscaler.setSelectedDurationSeconds($0) }
        ),
        minimum: 0.05,
        maximum: max(0.05, upscaler.durationSeconds - upscaler.normalizedStartSeconds),
        onCommit: {
          requestPreviewSeek(to: upscaler.normalizedEndSeconds)
        }
      )
      timeField(
        "終了",
        field: .end,
        value: Binding(
          get: { upscaler.normalizedEndSeconds },
          set: { upscaler.setEndSeconds($0) }
        ),
        onCommit: {
          requestPreviewSeek(to: upscaler.normalizedEndSeconds)
        }
      )
      Button("動画全体を選択") { upscaler.selectFullRange() }
        .disabled(upscaler.sourceInfo == nil || upscaler.isRunning)
      Text("開始・終了・範囲時間は、スライダーまたは数値入力で0.01秒単位に指定できます。選択範囲だけをデコードします。")
        .font(.caption).foregroundStyle(.secondary)
    }
    .disabled(upscaler.sourceInfo == nil || upscaler.isRunning)
  }

  private var modelSection: some View {
    Section("アップスケーラー") {
      Picker("モデル", selection: $upscaler.upscalerModel) {
        Text("FlashVSR Tiny（動画・時間整合）").tag("flashvsr")
        Text("AdcSR（軽量な1-step拡散）").tag("adcsr")
      }
      .pickerStyle(.segmented)
      HStack {
        Label(
          upscaler.selectedModelReady ? "モデル設定済み" : "モデルがありません",
          systemImage: upscaler.selectedModelReady
            ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
        )
        .foregroundStyle(upscaler.selectedModelReady ? .green : .orange)
        Spacer()
        Button("モデルを自動設定…", action: presentModelSetup)
      }
      Picker("出力指定", selection: $upscaler.sizingMode) {
        Text("倍率").tag("multiple")
        Text("解像度").tag("custom")
      }
      .pickerStyle(.segmented)
      if upscaler.sizingMode == "multiple" {
        Picker("倍率", selection: $upscaler.scale) {
          Text("2倍").tag(2)
          Text("4倍").tag(4)
        }
        .pickerStyle(.segmented)
        if upscaler.selectedUpscaler == .adcSR, upscaler.scale == 2 {
          Text("AdcSRは内部では常に4倍推論し、最後に2倍へ縮小します。処理時間は4倍出力とほぼ同じです。")
            .font(.caption).foregroundStyle(.orange)
        }
      } else {
        customSizeFields
      }
      Picker("計算デバイス", selection: $upscaler.computeMode) {
        Text(upscaler.selectedUpscaler == .adcSR ? "GPU優先（推奨）" : "Hybrid（推奨）")
          .tag("hybrid")
        Text("自動").tag("automatic")
        Text("GPU").tag("gpu")
      }
      if upscaler.selectedUpscaler == .adcSR {
        Toggle(
          "optical flowで時間方向を安定化",
          isOn: $upscaler.adcSRTemporalStabilization
        )
        LabeledContent("高周波残差の混合") {
          HStack(spacing: 10) {
            Slider(
              value: $upscaler.adcSRTemporalStrength, in: 0...0.25, step: 0.01
            )
            .frame(width: 190)
            Text("\(Int((upscaler.adcSRTemporalStrength * 100).rounded()))%")
              .monospacedDigit().frame(width: 42, alignment: .trailing)
          }
        }
        .disabled(!upscaler.adcSRTemporalStabilization)
      }
      modelDetails
      if upscaler.selectedUpscaler == .adcSR {
        UpscalerPathSettingRow(
          title: "AdcSR Core AIモデル（外部）",
          value: $upscaler.adcSRRootPath,
          action: upscaler.chooseAdcSRRoot
        )
        Text("128px・16px重複タイルを4倍化してmmapへfeather blendし、前フレームの高周波残差だけをoptical flowで混合します。")
          .font(.caption).foregroundStyle(.secondary)
      } else {
        UpscalerPathSettingRow(
          title: "FlashVSR Core AIモデル（外部）",
          value: $upscaler.flashVSRRootPath,
          action: upscaler.chooseFlashVSRRoot
        )
        Text("85フレーム単位で共有デコードし、タイル処理・合成・書き込みの進捗を表示します。")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
    .disabled(upscaler.isRunning)
  }

  private var customSizeFields: some View {
    Group {
      LabeledContent("指定解像度") {
        HStack(spacing: 8) {
          TextField(
            "幅", value: Binding(
              get: { upscaler.targetWidth },
              set: { upscaler.setTargetWidth($0) }
            ), format: .number
          )
          .frame(width: 88)
          Text("×").foregroundStyle(.secondary)
          TextField(
            "高さ", value: Binding(
              get: { upscaler.targetHeight },
              set: { upscaler.setTargetHeight($0) }
            ), format: .number
          )
          .frame(width: 88)
          Text("px").foregroundStyle(.secondary)
        }
      }
      Toggle("入力の縦横比を固定", isOn: $upscaler.preserveAspectRatio)
      if upscaler.selectedUpscaler == .flashVSR {
        Picker("生成品質", selection: $upscaler.qualityMode) {
          Text("高速（2x優先）").tag("fast")
          Text("高品質（4x許可）").tag("quality")
        }
        .pickerStyle(.segmented)
      }
      HStack {
        Text("プリセット").foregroundStyle(.secondary)
        Button("1920×1080") { upscaler.setTargetHeight(1080) }
        Button("3840×2160") { upscaler.setTargetHeight(2160) }
        Spacer()
      }
      if let error = upscaler.customSizeError {
        Label(error, systemImage: "exclamationmark.triangle.fill")
          .font(.caption).foregroundStyle(.orange)
      }
    }
  }

  private var modelDetails: some View {
    Group {
      LabeledContent("入力解像度") {
        Text(upscaler.sourceInfo?.resolutionText ?? "—").monospacedDigit()
      }
      LabeledContent("出力解像度") {
        Text(upscaler.outputResolutionText).monospacedDigit()
      }
      LabeledContent("空間タイル数") {
        Text(upscaler.tileCountText).monospacedDigit()
      }
      LabeledContent(
        upscaler.selectedUpscaler == .adcSR
          ? "一時ディスク領域／フレーム" : "一時ディスク領域／セグメント"
      ) { Text(upscaler.scratchSpaceText).monospacedDigit() }
      LabeledContent("実行方式") {
        Text(upscaler.runtimeText).foregroundStyle(.secondary)
      }
      LabeledContent("モデル") {
        Text(upscaler.modelAvailabilityText)
          .foregroundStyle(
            upscaler.modelAvailabilityText.contains("見つかりません")
              ? .orange : .secondary
          )
      }
    }
  }

  private var outputSection: some View {
    Section("書き出し") {
      UpscalerPathRow(
        title: "出力動画", icon: "square.and.arrow.down",
        url: upscaler.outputURL, action: upscaler.chooseOutput,
        actionLabel: "保存先…"
      )
      Toggle("元動画の音声を保持", isOn: $upscaler.preserveAudio)
      LabeledContent("映像コーデック") { Text("H.264") }
      Text("既存ファイルは上書きしません。中間動画を作らず、最終映像を1回だけ圧縮します。")
        .font(.caption).foregroundStyle(.secondary)
    }
    .disabled(upscaler.isRunning)
  }

  private var logSection: some View {
    Section("ログ") {
      ScrollView {
        Text(upscaler.log.isEmpty ? " " : upscaler.log)
          .font(.system(.caption, design: .monospaced))
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .topLeading)
      }
      .frame(minHeight: 180)
    }
  }

  private var progressArea: some View {
    VStack(spacing: 7) {
      ProgressView(value: upscaler.progress).progressViewStyle(.linear)
      HStack {
        Text(upscaler.status).lineLimit(1)
        Spacer()
        Text("経過 \(upscaler.elapsedText) · 残り目安 \(upscaler.estimatedRemainingText)")
          .monospacedDigit()
      }
      .font(.caption).foregroundStyle(.secondary)
    }
    .padding(.horizontal, 20).padding(.vertical, 10)
  }

  private var footer: some View {
    HStack(spacing: 10) {
      Button(action: upscaler.revealOutput) {
        Image(systemName: "folder.badge.gearshape")
      }
      .help("出力をFinderで表示")
      .disabled(upscaler.outputURL == nil)
      Toggle("ログ", isOn: $showLog).toggleStyle(.button)
      Spacer()
      if upscaler.isRunning {
        Button(role: .destructive, action: upscaler.stop) {
          Label("停止", systemImage: "stop.fill")
        }
      } else {
        Button(action: upscaler.start) {
          Label("アップスケール開始", systemImage: "play.fill")
        }
        .buttonStyle(.borderedProminent)
        .disabled(!upscaler.canStart || h3Generation.isRunning)
      }
    }
    .padding(.horizontal, 20).frame(height: 58)
  }

  private func timeField(
    _ title: String,
    field: TimeInputField,
    value: Binding<Double>,
    minimum: Double = 0,
    maximum: Double? = nil,
    onCommit: @escaping () -> Void = {}
  ) -> some View {
    let requestedMaximum = maximum ?? upscaler.durationSeconds
    // SwiftUI traps when Slider is initialized with a zero-width range.  The
    // media duration is intentionally zero while metadata is still loading,
    // so keep a harmless one-step range even though the containing section is
    // disabled during that interval.
    let safeMaximum = max(minimum + 0.01, requestedMaximum)
    let safeValue = Binding<Double>(
      get: { min(max(minimum, value.wrappedValue), safeMaximum) },
      set: { value.wrappedValue = min(max(minimum, $0), safeMaximum) }
    )
    return LabeledContent(title) {
      HStack(spacing: 12) {
        Slider(
          value: safeValue,
          in: minimum...safeMaximum,
          step: 0.01
        )
        .frame(minWidth: 240)
        Text("数値入力")
          .font(.caption)
          .foregroundStyle(.secondary)
        TextField(
          "秒数", value: safeValue,
          format: .number.precision(.fractionLength(0...3))
        )
        .textFieldStyle(.roundedBorder)
        .font(.body.monospacedDigit())
        .multilineTextAlignment(.trailing)
        .frame(width: 96)
        .accessibilityLabel("\(title)を秒で直接入力")
        .focused($focusedTimeInput, equals: field)
        .onSubmit {
          focusedTimeInput = nil
        }
        .onChange(of: focusedTimeInput) { oldValue, newValue in
          if oldValue == field, newValue != field {
            onCommit()
          }
        }
        Text("秒").foregroundStyle(.secondary)
      }
    }
  }

  private var applicationIcon: some View {
    Image(nsImage: NSApp.applicationIconImage)
      .resizable()
      .scaledToFit()
      .frame(width: 40, height: 40)
      .accessibilityLabel("mioh upscaler")
  }

  private func requestPreviewSeek(to seconds: Double) {
    previewSeekRequest = UpscalerVideoSeekRequest(seconds: seconds)
  }

  private func presentModelSetup() {
    modelSetup.prepare(
      for: upscaler.selectedUpscaler,
      preferredPath: upscaler.selectedModelRootPath
    )
    showingModelSetup = true
  }
}

private enum UpscalerWorkspaceTab: Hashable {
  case upscale
  case videoGeneration
}

private struct UpscalerPathRow: View {
  let title: String
  let icon: String
  let url: URL?
  let action: () -> Void
  let actionLabel: String

  var body: some View {
    HStack(spacing: 12) {
      Image(systemName: icon).frame(width: 20).foregroundStyle(.secondary)
      VStack(alignment: .leading, spacing: 3) {
        Text(title).font(.caption).foregroundStyle(.secondary)
        Text(url?.path ?? "未選択")
          .lineLimit(1).truncationMode(.middle)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      Button(actionLabel, action: action).buttonStyle(.bordered)
    }
    .frame(minHeight: 42)
  }
}

private struct UpscalerPathSettingRow: View {
  let title: String
  @Binding var value: String
  let action: () -> Void

  var body: some View {
    LabeledContent(title) {
      HStack {
        TextField("", text: $value, prompt: Text("未指定"))
          .textFieldStyle(.roundedBorder).frame(width: 380)
        Button(action: action) { Image(systemName: "folder") }
          .buttonStyle(.borderless).help("選択")
      }
    }
  }
}

private struct UpscalerSourceInfoRow: View {
  let info: UpscalerMediaInfo?
  let failure: String?

  var body: some View {
    HStack(alignment: .top, spacing: 12) {
      Image(systemName: "info.circle").foregroundStyle(.secondary)
      if let info {
        VStack(alignment: .leading, spacing: 4) {
          HStack(spacing: 18) {
            field("解像度", info.resolutionText)
            field("フレームレート", "\(info.frameRateText)fps")
            field("長さ", info.durationText)
          }
          HStack(spacing: 18) {
            field("コーデック", info.videoCodec)
            if let bitRate = info.bitRateText { field("ビットレート", bitRate) }
            if let size = info.fileSizeText { field("サイズ", size) }
            field("音声", info.audio ?? "なし")
          }
        }
      } else if let failure {
        Text("入力情報を読み取れません: \(failure)")
          .font(.caption).foregroundStyle(.secondary)
      } else {
        ProgressView().controlSize(.small)
        Text("入力情報を読み取り中…").font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
    }
    .frame(minHeight: 42)
  }

  private func field(_ title: String, _ value: String) -> some View {
    VStack(alignment: .leading, spacing: 1) {
      Text(title).font(.caption2).foregroundStyle(.secondary)
      Text(value).font(.caption).monospacedDigit()
    }
  }
}

private struct UpscalerRangeTimeline: View {
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
