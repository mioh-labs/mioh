import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

private struct MiniMaxH3UIProgressEvent: Decodable {
  let stage: String
  let state: String
  let progress: Double
  let message: String
}

@MainActor
final class MiniMaxH3Controller: ObservableObject {
  private static let maximumIdentityImages = 10
  private static let manifestPathDefaultsKey =
    "com.okatti.mioh.upscaler.10erosMaxH3ManifestPath"
  private static let legacyManifestPathDefaultsKey =
    "com.okatti.lada.coreai.10erosMaxH3ManifestPath"

  @Published var prompt = "モザイクを除去して最高品質の動画を生成する。"
  @Published var backend = "coreai"
  @Published var width = 864
  @Published var height = 480
  @Published var duration = 10.0
  @Published var seed = "261662374822964"
  @Published var manifestPath: String {
    didSet {
      UserDefaults.standard.set(
        manifestPath,
        forKey: Self.manifestPathDefaultsKey
      )
    }
  }
  @Published private(set) var inputURLs: [URL] = []
  @Published private(set) var usesUpscalerInput = true
  @Published var outputPath = ""
  @Published var progress = 0.0
  @Published var status = "待機中"
  @Published var log = ""
  @Published var isRunning = false

  private var process: Process?
  private var standardOutputBuffer = Data()
  private var currentUpscalerInput: URL?
  private var automaticOutputPath = ""

  init() {
    let savedManifestPath = UserDefaults.standard.string(
      forKey: Self.manifestPathDefaultsKey
    ) ?? UserDefaults.standard.string(
      forKey: Self.legacyManifestPathDefaultsKey
    ) ?? ""
    manifestPath = Self.resolvePipelineManifestPath(savedManifestPath)
  }

  var supportsRuntime: Bool {
    ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 27
  }

  var runnerURL: URL? {
    Bundle.main.resourceURL?.appendingPathComponent(
      "bin/mioh-minimax-h3-native"
    )
  }

  var modelReady: Bool {
    Self.isPipelineManifest(
      URL(fileURLWithPath: effectiveManifestPath)
    )
  }

  private var effectiveManifestPath: String {
    Self.resolvePipelineManifestPath(manifestPath)
  }

  var isImageSequence: Bool {
    !inputURLs.isEmpty && inputURLs.allSatisfy(Self.isImage)
  }

  var inputSummary: String {
    if inputURLs.isEmpty { return "未指定" }
    if isImageSequence {
      if inputURLs.count == 1 { return inputURLs[0].path }
      return "人物参照画像 \(inputURLs.count)枚（先頭画像が基準）"
    }
    return inputURLs[0].path
  }

  func canStart() -> Bool {
    guard !isRunning, supportsRuntime, modelReady, validInputSelection,
      !outputPath.isEmpty,
      !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      UInt64(seed) != nil,
      let runnerURL,
      FileManager.default.isExecutableFile(atPath: runnerURL.path)
    else { return false }
    return true
  }

  func prepare(upscalerInput: URL?) {
    guard usesUpscalerInput, upscalerInput != currentUpscalerInput else { return }
    currentUpscalerInput = upscalerInput
    setInputURLs(
      upscalerInput.map { [$0.standardizedFileURL] } ?? [],
      upscalerInput: true
    )
  }

  func useUpscalerInput(_ input: URL?) {
    currentUpscalerInput = input
    setInputURLs(
      input.map { [$0.standardizedFileURL] } ?? [],
      upscalerInput: true
    )
  }

  func chooseInput() {
    let panel = NSOpenPanel()
    panel.title = "MiniMax H3の入力動画または画像を選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = true
    panel.allowedContentTypes = [.movie, .image]
    guard panel.runModal() == .OK else { return }
    var urls = panel.urls.map(\.standardizedFileURL)
    let allImages = !urls.isEmpty && urls.allSatisfy(Self.isImage)
    if urls.count > 1, !allImages {
      status = "複数選択できるのは画像だけです"
      return
    }
    if allImages, urls.count > Self.maximumIdentityImages {
      status = "人物参照画像は最大\(Self.maximumIdentityImages)枚です"
      return
    }
    if allImages {
      urls.sort {
        $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent)
          == .orderedAscending
      }
    }
    setInputURLs(urls, upscalerInput: false)
  }

  private func setInputURLs(_ urls: [URL], upscalerInput: Bool) {
    inputURLs = urls
    usesUpscalerInput = upscalerInput
    guard let first = urls.first else { return }
    let suffix = urls.count > 1 ? "-\(urls.count)-images" : ""
    let proposed = first.deletingPathExtension().path
      + suffix + "-minimax-h3.mp4"
    if outputPath.isEmpty || outputPath == automaticOutputPath {
      outputPath = proposed
      automaticOutputPath = proposed
    }
  }

  private var validInputSelection: Bool {
    guard !inputURLs.isEmpty else { return false }
    if inputURLs.count == 1 {
      return Self.isImage(inputURLs[0]) || Self.isMovie(inputURLs[0])
    }
    return inputURLs.count <= Self.maximumIdentityImages
      && inputURLs.allSatisfy(Self.isImage)
  }

  private static func isImage(_ url: URL) -> Bool {
    UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) == true
  }

  private static func isMovie(_ url: URL) -> Bool {
    UTType(filenameExtension: url.pathExtension)?.conforms(to: .movie) == true
  }

  private static func resolvePipelineManifestPath(_ path: String) -> String {
    guard !path.isEmpty else { return path }
    let selected = URL(fileURLWithPath: path).standardizedFileURL
    if isPipelineManifest(selected) { return selected.path }
    let sibling = selected.deletingLastPathComponent()
      .appendingPathComponent("manifest.json")
      .standardizedFileURL
    return isPipelineManifest(sibling) ? sibling.path : selected.path
  }

  private static func isPipelineManifest(_ url: URL) -> Bool {
    guard let data = try? Data(contentsOf: url),
      let object = try? JSONSerialization.jsonObject(with: data),
      let dictionary = object as? [String: Any]
    else { return false }
    return dictionary["schemaVersion"] != nil
      && dictionary["modelIdentifier"] != nil
      && dictionary["stages"] != nil
      && dictionary["sigmas"] != nil
  }

  func chooseManifest() {
    let panel = NSOpenPanel()
    panel.title = "MiniMax H3 Swiftマニフェストを選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    panel.allowedContentTypes = [.json]
    if panel.runModal() == .OK, let url = panel.url {
      let selected = url.standardizedFileURL.path
      let resolved = Self.resolvePipelineManifestPath(selected)
      guard Self.isPipelineManifest(URL(fileURLWithPath: resolved)) else {
        status = "MiniMax H3のパイプラインmanifest.jsonを選択してください"
        return
      }
      manifestPath = resolved
      if selected != resolved {
        status = "同じフォルダのパイプラインmanifest.jsonへ補正しました"
      }
    }
  }

  func chooseOutput() {
    let panel = NSSavePanel()
    panel.title = "H3生成動画の保存先"
    panel.nameFieldStringValue = URL(fileURLWithPath: outputPath).lastPathComponent
    panel.allowedContentTypes = [.mpeg4Movie]
    if panel.runModal() == .OK, let url = panel.url {
      outputPath = url.standardizedFileURL.path
      automaticOutputPath = ""
    }
  }

  func start() {
    guard canStart(), let runnerURL else {
      if !supportsRuntime {
        status = "macOS 27が必要です"
      } else if !modelReady {
        status = "変換済みMiniMax H3モデルが未インストールです"
      } else {
        status = "H3設定を確認してください"
      }
      return
    }
    let cachesRoot = FileManager.default.urls(
      for: .cachesDirectory,
      in: .userDomainMask
    ).first!
    let preferredCache = cachesRoot.appendingPathComponent(
      "com.okatti.mioh.upscaler/10eros-max-h3",
      isDirectory: true
    )
    let legacyCache = cachesRoot.appendingPathComponent(
      "com.okatti.lada.coreai/10eros-max-h3",
      isDirectory: true
    )
    // Preserve completed stage caches created by earlier Mioh builds. New
    // installations use the upscaler-owned path.
    let cache = FileManager.default.fileExists(atPath: legacyCache.path)
      && !FileManager.default.fileExists(atPath: preferredCache.path)
      ? legacyCache : preferredCache
    do {
      try FileManager.default.createDirectory(
        at: cache,
        withIntermediateDirectories: true
      )
    } catch {
      status = "キャッシュ作成失敗"
      log += "\(error.localizedDescription)\n"
      return
    }
    let task = Process()
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    task.executableURL = runnerURL
    let resolvedManifestPath = effectiveManifestPath
    if manifestPath != resolvedManifestPath {
      manifestPath = resolvedManifestPath
    }
    var arguments = [
      "run",
      "--manifest", resolvedManifestPath,
      "--output", outputPath,
      "--prompt", prompt,
      "--cache", cache.path,
      "--backend", backend,
      "--width", String(width),
      "--height", String(height),
      "--duration", String(duration),
      "--seed", seed,
    ]
    if isImageSequence {
      do {
        let data = try JSONEncoder().encode(inputURLs.map(\.path))
        arguments += [
          "--input-images-json",
          String(decoding: data, as: UTF8.self),
        ]
      } catch {
        status = "画像入力の準備に失敗しました"
        log += "\(error.localizedDescription)\n"
        return
      }
    } else if let video = inputURLs.first {
      arguments += ["--input", video.path]
    }
    task.arguments = arguments
    task.standardOutput = outputPipe
    task.standardError = errorPipe
    outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      guard !data.isEmpty else { return }
      Task { @MainActor in self?.consumeStandardOutput(data) }
    }
    errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      guard !data.isEmpty else { return }
      Task { @MainActor in
        self?.log += String(decoding: data, as: UTF8.self)
      }
    }
    task.terminationHandler = { [weak self] completed in
      Task { @MainActor in
        outputPipe.fileHandleForReading.readabilityHandler = nil
        errorPipe.fileHandleForReading.readabilityHandler = nil
        guard let self else { return }
        self.process = nil
        self.isRunning = false
        if completed.terminationStatus == 0 {
          self.progress = 1
          self.status = "MiniMax H3生成完了"
        } else if self.status == "停止中" {
          self.status = "停止"
        } else {
          self.status = "MiniMax H3生成失敗"
        }
      }
    }
    do {
      progress = 0
      status = "MiniMax H3準備中"
      log = "Swift / \(backend == "coreai" ? "Core AI" : "Core ML")\n"
      standardOutputBuffer.removeAll(keepingCapacity: true)
      try task.run()
      process = task
      isRunning = true
    } catch {
      status = "MiniMax H3起動失敗"
      log += "\(error.localizedDescription)\n"
    }
  }

  func stop() {
    guard let process, process.isRunning else { return }
    status = "停止中"
    process.interrupt()
    DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2) {
      if process.isRunning { process.terminate() }
    }
  }

  func revealOutput() {
    guard !outputPath.isEmpty else { return }
    NSWorkspace.shared.activateFileViewerSelecting([
      URL(fileURLWithPath: outputPath)
    ])
  }

  private func consumeStandardOutput(_ data: Data) {
    standardOutputBuffer.append(data)
    while let newline = standardOutputBuffer.firstIndex(of: 0x0A) {
      let lineData = standardOutputBuffer.prefix(upTo: newline)
      standardOutputBuffer.removeSubrange(...newline)
      guard !lineData.isEmpty else { continue }
      if let event = try? JSONDecoder().decode(
        MiniMaxH3UIProgressEvent.self,
        from: Data(lineData)
      ) {
        progress = event.progress
        status = event.message
        log += "[\(event.stage)] \(event.state): \(event.message)\n"
      } else {
        log += String(decoding: lineData, as: UTF8.self) + "\n"
      }
    }
  }
}

struct MiniMaxH3GenerationView: View {
  @ObservedObject var controller: MiniMaxH3Controller
  let upscalerInputURL: URL?

  var body: some View {
    Form {
      Section("動画生成（MiniMax H3）") {
        LabeledContent("入力") {
          HStack {
            Text(controller.inputSummary)
              .lineLimit(1)
              .truncationMode(.middle)
              .textSelection(.enabled)
            Button("選択", action: controller.chooseInput)
            if !controller.usesUpscalerInput, upscalerInputURL != nil {
              Button("アップスケール入力を使用") {
                controller.useUpscalerInput(upscalerInputURL)
              }
            }
          }
        }
        if controller.isImageSequence, controller.inputURLs.count > 1 {
          DisclosureGroup("選択した画像（\(controller.inputURLs.count)枚）") {
            ForEach(controller.inputURLs, id: \.path) { url in
              Text(url.path)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
          }
        }
        Text("動画は1本、画像は最大10枚を選択できます。先頭画像を生成の基準にし、追加画像は同一人物の外見参照として使います。時間変化はプロンプトで指定してください。")
          .font(.caption)
          .foregroundStyle(.secondary)
        LabeledContent("実行方式") {
          Text("Core AI（BF16 DiT はGPU、対応部分はANE）")
        }
        LabeledContent("モデルマニフェスト") {
          HStack {
            TextField("", text: $controller.manifestPath)
              .textFieldStyle(.roundedBorder)
            Button(action: controller.chooseManifest) {
              Image(systemName: "folder")
            }
            .buttonStyle(.borderless)
          }
        }
        if !controller.supportsRuntime {
          Text("Core AI版MiniMax H3にはmacOS 27以降が必要です")
            .foregroundStyle(.red)
        } else if !controller.modelReady {
          Text("モデルは内蔵されません。外部のMiniMax H3 manifest.jsonを選択してください")
            .foregroundStyle(.orange)
        }
      }
      Section("プロンプト") {
        TextEditor(text: $controller.prompt)
          .font(.body)
          .frame(minHeight: 64, maxHeight: 120)
        Text("自由入力（現在のCore AIプロファイルは最大16 Qwenトークン。超過分は末尾を省略）")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Section("生成設定") {
        LabeledContent("解像度") {
          Text("864×480（MiniMax H3 固定プロファイル）")
            .monospacedDigit()
        }
        LabeledContent("長さ") {
          Text("10.0秒").monospacedDigit()
        }
        LabeledContent("Seed") {
          TextField("", text: $controller.seed)
            .multilineTextAlignment(.trailing)
            .frame(width: 220)
        }
        LabeledContent("出力") {
          HStack {
            TextField("", text: $controller.outputPath)
              .textFieldStyle(.roundedBorder)
            Button(action: controller.chooseOutput) {
              Image(systemName: "folder")
            }
            .buttonStyle(.borderless)
          }
        }
      }
      Section("進捗") {
        ProgressView(value: controller.progress)
        Text(controller.status)
          .font(.callout.monospacedDigit())
        ScrollView {
          Text(controller.log.isEmpty ? " " : controller.log)
            .font(.system(.caption, design: .monospaced))
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .frame(minHeight: 110, maxHeight: 180)
        .background(Color(nsColor: .textBackgroundColor))
      }
    }
    .formStyle(.grouped)
    .onAppear { controller.prepare(upscalerInput: upscalerInputURL) }
    .onChange(of: upscalerInputURL) { _, value in
      controller.prepare(upscalerInput: value)
    }
  }
}
