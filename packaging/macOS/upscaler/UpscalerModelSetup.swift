// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AppKit
import Foundation
import SwiftUI

private struct ModelSetupProgressEvent {
  let fraction: Double
  let message: String
}

@MainActor
final class UpscalerModelSetupController: ObservableObject {
  private static let destinationDefaultsKey =
    "com.okatti.mioh.upscaler.modelSetupDestination"

  @Published var destinationPath: String {
    didSet {
      UserDefaults.standard.set(
        destinationPath,
        forKey: Self.destinationDefaultsKey
      )
      refreshInstallationState()
    }
  }
  @Published var installFlashVSR = true
  @Published var installAdcSR = false
  @Published private(set) var flashVSRInstalled = false
  @Published private(set) var adcSRInstalled = false
  @Published private(set) var isRunning = false
  @Published private(set) var progress = 0.0
  @Published private(set) var status = "未確認"
  @Published private(set) var log = ""
  @Published private(set) var completedSuccessfully = false

  private var process: Process?
  private var outputBuffer = ""
  private var completion: (() -> Void)?
  private var cancellationRequested = false

  init() {
    destinationPath = UserDefaults.standard.string(
      forKey: Self.destinationDefaultsKey
    ) ?? FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent("Documents/lada/model_weights", isDirectory: true)
      .path
    refreshInstallationState()
  }

  var canStart: Bool {
    !isRunning
      && (installFlashVSR || installAdcSR)
      && !destinationPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  func prepare(for kind: VideoUpscaleController.UpscalerKind, preferredPath: String) {
    if !preferredPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      var preferred = URL(fileURLWithPath: preferredPath).standardizedFileURL
      if preferred.lastPathComponent == VideoUpscaleController.nativeDirectoryName
        || preferred.pathExtension == "aimodel"
      {
        preferred.deleteLastPathComponent()
      }
      destinationPath = preferred.path
    }
    installFlashVSR = kind == .flashVSR
    installAdcSR = kind == .adcSR
    completedSuccessfully = false
    refreshInstallationState()
  }

  func chooseDestination() {
    let panel = NSOpenPanel()
    panel.title = "モデルの配置フォルダを選択"
    panel.prompt = "ここに配置"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    let current = URL(fileURLWithPath: destinationPath).standardizedFileURL
    panel.directoryURL = FileManager.default.fileExists(atPath: current.path)
      ? current : current.deletingLastPathComponent()
    guard panel.runModal() == .OK, let url = panel.url else { return }
    destinationPath = url.standardizedFileURL.path
  }

  func refreshInstallationState() {
    let root = URL(fileURLWithPath: destinationPath).standardizedFileURL
    let flash = root.appendingPathComponent(
      VideoUpscaleController.nativeDirectoryName,
      isDirectory: true
    )
    flashVSRInstalled = (0..<30).allSatisfy { index in
      FileManager.default.fileExists(
        atPath: flash.appendingPathComponent(
          String(format: "dit_block_%02d.aimodel", index),
          isDirectory: true
        ).path
      )
    } && ["patch_head", "lq_projection", "tcdecoder"].allSatisfy { stem in
      FileManager.default.fileExists(
        atPath: flash.appendingPathComponent("\(stem).aimodel", isDirectory: true).path
      ) || FileManager.default.fileExists(
        atPath: flash.appendingPathComponent("\(stem).mlmodelc", isDirectory: true).path
      )
    }
    let adcsr = root.appendingPathComponent(
      "adcsr_x4_float32.aimodel",
      isDirectory: true
    )
    adcSRInstalled = FileManager.default.fileExists(
      atPath: adcsr.appendingPathComponent("main.mlirb").path
    ) && FileManager.default.fileExists(
      atPath: adcsr.appendingPathComponent("metadata.json").path
    )
    if !isRunning {
      status = flashVSRInstalled || adcSRInstalled ? "配置状態を確認済み" : "モデルがありません"
    }
  }

  func start(completion: @escaping () -> Void) {
    guard canStart,
      let script = Bundle.main.resourceURL?.appendingPathComponent(
        "model-tools/setup-upscaler-models.zsh"
      ),
      FileManager.default.isExecutableFile(atPath: script.path)
    else {
      status = "自動設定ツールが見つかりません"
      return
    }
    let destination = URL(fileURLWithPath: destinationPath).standardizedFileURL
    do {
      try FileManager.default.createDirectory(
        at: destination,
        withIntermediateDirectories: true
      )
    } catch {
      status = "配置フォルダを作成できません: \(error.localizedDescription)"
      return
    }

    let task = Process()
    task.executableURL = script
    var arguments = ["--destination", destination.path]
    if installFlashVSR { arguments.append("--flashvsr") }
    if installAdcSR { arguments.append("--adcsr") }
    task.arguments = arguments
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else {
        return
      }
      Task { @MainActor [weak self] in self?.consume(text) }
    }
    task.terminationHandler = { [weak self, weak pipe] completed in
      pipe?.fileHandleForReading.readabilityHandler = nil
      Task { @MainActor [weak self] in
        self?.finish(exitCode: completed.terminationStatus)
      }
    }

    isRunning = true
    cancellationRequested = false
    completedSuccessfully = false
    progress = 0.01
    status = "モデル自動設定を開始しています"
    log = "配置先: \(destination.path)\n"
    self.completion = completion
    process = task
    do {
      try task.run()
    } catch {
      process = nil
      isRunning = false
      status = "自動設定を開始できません: \(error.localizedDescription)"
    }
  }

  func stop() {
    guard isRunning else { return }
    cancellationRequested = true
    status = "自動設定を停止しています"
    process?.terminate()
  }

  private func consume(_ text: String) {
    outputBuffer += text
    while let newline = outputBuffer.firstIndex(of: "\n") {
      let line = String(outputBuffer[..<newline])
      outputBuffer.removeSubrange(...newline)
      if let event = parseProgress(line) {
        progress = min(1, max(progress, event.fraction))
        status = event.message
      } else if !line.isEmpty {
        log += line + "\n"
      }
    }
  }

  private func parseProgress(_ line: String) -> ModelSetupProgressEvent? {
    let fields = line.split(separator: "|", maxSplits: 2).map(String.init)
    guard fields.count == 3, fields[0] == "MIOH_SETUP",
      let fraction = Double(fields[1])
    else { return nil }
    return ModelSetupProgressEvent(fraction: fraction, message: fields[2])
  }

  private func finish(exitCode: Int32) {
    if !outputBuffer.isEmpty {
      consume("\n")
    }
    process = nil
    isRunning = false
    refreshInstallationState()
    if cancellationRequested {
      status = "モデル自動設定を停止しました。途中ファイルは次回再開に使われます。"
    } else if exitCode == 0 {
      progress = 1
      completedSuccessfully = true
      status = "モデル自動設定が完了しました"
      completion?()
    } else {
      status = "モデル自動設定に失敗しました（終了コード \(exitCode)）"
    }
    completion = nil
    cancellationRequested = false
  }
}

struct UpscalerModelSetupView: View {
  @ObservedObject var controller: UpscalerModelSetupController
  let applyConfiguration: () -> Void
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        Image(systemName: "square.and.arrow.down.on.square.fill")
          .font(.system(size: 34))
          .foregroundStyle(.tint)
        VStack(alignment: .leading) {
          Text("初回モデル自動設定").font(.title2.weight(.semibold))
          Text("公式重みの取得、Mac Core AI変換、配置、パス設定を自動で行います。")
            .foregroundStyle(.secondary)
        }
      }

      GroupBox("配置先") {
        HStack {
          TextField("モデルフォルダ", text: $controller.destinationPath)
            .textFieldStyle(.roundedBorder)
          Button("選択…", action: controller.chooseDestination)
        }
        .padding(8)
      }

      GroupBox("セットアップするモデル") {
        VStack(alignment: .leading, spacing: 10) {
          Toggle(isOn: $controller.installFlashVSR) {
            VStack(alignment: .leading) {
              Text("FlashVSR-v1.1")
              Text("公式重み約6.5 GBを取得し、このMacでCore AIへ変換します。作業用に18 GB以上の空きが必要です。")
                .font(.caption).foregroundStyle(.secondary)
            }
          }
          Text(controller.flashVSRInstalled ? "配置済み" : "未配置")
            .font(.caption).foregroundStyle(controller.flashVSRInstalled ? .green : .orange)
          Divider()
          Toggle(isOn: $controller.installAdcSR) {
            VStack(alignment: .leading) {
              Text("AdcSR ×4")
              Text("配布元のMac Core AIモデル約1.7 GBを取得し、チェックサム検証します。")
                .font(.caption).foregroundStyle(.secondary)
            }
          }
          Text(controller.adcSRInstalled ? "配置済み" : "未配置")
            .font(.caption).foregroundStyle(controller.adcSRInstalled ? .green : .orange)
        }
        .padding(8)
      }

      ProgressView(value: controller.progress)
      Text(controller.status).font(.callout)
      if !controller.log.isEmpty {
        ScrollView {
          Text(controller.log)
            .font(.caption.monospaced())
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(height: 120)
        .padding(6)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
      }

      HStack {
        if controller.isRunning {
          Button("停止", role: .destructive, action: controller.stop)
        } else {
          Button("閉じる") { dismiss() }
        }
        Spacer()
        Button("自動設定を開始") {
          controller.start {
            applyConfiguration()
          }
        }
        .buttonStyle(.borderedProminent)
        .disabled(!controller.canStart)
        if controller.completedSuccessfully {
          Button("完了") {
            applyConfiguration()
            dismiss()
          }
        }
      }
    }
    .padding(22)
    .frame(minWidth: 680, minHeight: 600)
  }
}
