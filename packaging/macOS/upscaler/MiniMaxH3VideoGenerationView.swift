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

private struct MiniMaxH3ResolutionProfile: Identifiable, Hashable {
  let width: Int
  let height: Int
  let note: String

  var id: String { "\(width)x\(height)" }
  var label: String { "\(width)×\(height)（\(note)）" }

  static let supported: [Self] = [
    .init(width: 640, height: 352, note: "低負荷・横"),
    .init(width: 864, height: 480, note: "標準・横"),
    .init(width: 960, height: 544, note: "高精細・横"),
    .init(width: 1024, height: 576, note: "最高精細・横"),
    .init(width: 640, height: 640, note: "正方形"),
    .init(width: 768, height: 768, note: "最高精細・正方形"),
    .init(width: 480, height: 864, note: "標準・縦"),
    .init(width: 576, height: 1024, note: "最高精細・縦"),
  ]
}

@MainActor
final class MiniMaxH3Controller: ObservableObject {
  private static let maximumIdentityImages = 8
  private static let maximumVisibleLogCharacters = 24_000
  private static let retainedLogCharacters = 16_000
  private static let manifestPathDefaultsKey =
    "com.okatti.mioh.upscaler.10erosMaxH3ManifestPath"
  private static let legacyManifestPathDefaultsKey =
    "com.okatti.lada.coreai.10erosMaxH3ManifestPath"

  @Published var prompt = "モザイクを除去して最高品質の動画を生成する。"
  @Published var backend = "coreai"
  @Published var resolutionProfileID = "864x480" {
    didSet {
      guard let profile = Self.resolutionProfiles.first(where: {
        $0.id == resolutionProfileID
      }) else { return }
      width = profile.width
      height = profile.height
    }
  }
  @Published private(set) var width = 864
  @Published private(set) var height = 480
  @Published var duration = 10.0
  @Published var seed = "261662374822964"
  @Published var manifestPath: String {
    didSet {
      UserDefaults.standard.set(
        manifestPath,
        forKey: Self.manifestPathDefaultsKey
      )
      refreshConditioningMode()
    }
  }
  @Published private(set) var supportsPromptOnly = false
  @Published private(set) var inputURLs: [URL] = []
  @Published private(set) var usesUpscalerInput = true
  @Published var imageReferenceScope = MiniMaxH3ImageReferenceScope.wholeImage
  @Published private(set) var faceReferences: [MiniMaxH3FaceReference] = []
  @Published private(set) var isDetectingFaces = false
  @Published var outputPath = ""
  @Published var progress = 0.0
  @Published var status = "待機中"
  @Published var log = ""
  @Published var isRunning = false

  private var process: Process?
  private var standardOutputBuffer = Data()
  private var standardErrorBuffer = Data()
  private var currentUpscalerInput: URL?
  private var automaticOutputPath = ""
  private var faceDetectionTask: Task<Void, Never>?
  private var faceReferenceDirectory: URL?

  fileprivate static let resolutionProfiles = MiniMaxH3ResolutionProfile.supported

  init() {
    let savedManifestPath = UserDefaults.standard.string(
      forKey: Self.manifestPathDefaultsKey
    ) ?? UserDefaults.standard.string(
      forKey: Self.legacyManifestPathDefaultsKey
    ) ?? ""
    manifestPath = Self.resolvePipelineManifestPath(savedManifestPath)
    refreshConditioningMode()
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
    if supportsPromptOnly { return "なし（プロンプトのみ）" }
    if inputURLs.isEmpty { return "未指定" }
    if isImageSequence {
      if imageReferenceScope == .faceOnly {
        if isDetectingFaces { return "人物の顔を検出中" }
        return "顔参照 \(selectedFaceReferences.count)件（検出\(faceReferences.count)件）"
      }
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
    guard upscalerInput != currentUpscalerInput else { return }
    currentUpscalerInput = upscalerInput
    guard usesUpscalerInput, !supportsPromptOnly else { return }
    setInputURLs(
      upscalerInput.map { [$0.standardizedFileURL] } ?? [],
      upscalerInput: true
    )
  }

  func useUpscalerInput(_ input: URL?) {
    currentUpscalerInput = input
    guard !supportsPromptOnly else { return }
    setInputURLs(
      input.map { [$0.standardizedFileURL] } ?? [],
      upscalerInput: true
    )
  }

  func chooseInput() {
    guard !supportsPromptOnly else {
      status = "プロンプトのみでは参照入力を使用しません"
      return
    }
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
    resetFaceReferences()
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
    if imageReferenceScope == .faceOnly,
      !urls.isEmpty,
      urls.allSatisfy(Self.isImage)
    {
      detectFaces()
    }
  }

  private var validInputSelection: Bool {
    if supportsPromptOnly { return inputURLs.isEmpty }
    guard !inputURLs.isEmpty else { return false }
    if isImageSequence, imageReferenceScope == .faceOnly {
      let count = selectedFaceReferences.count
      return !isDetectingFaces && count > 0
        && count <= Self.maximumIdentityImages
    }
    if inputURLs.count == 1 {
      return Self.isImage(inputURLs[0]) || Self.isMovie(inputURLs[0])
    }
    return inputURLs.count <= Self.maximumIdentityImages
      && inputURLs.allSatisfy(Self.isImage)
  }

  var selectedFaceReferenceCount: Int {
    selectedFaceReferences.count
  }

  private var selectedFaceReferences: [MiniMaxH3FaceReference] {
    faceReferences.filter(\.isSelected).sorted {
      if $0.sourceIndex != $1.sourceIndex {
        return $0.sourceIndex < $1.sourceIndex
      }
      return $0.faceIndex < $1.faceIndex
    }
  }

  func selectImageReferenceScope(_ scope: MiniMaxH3ImageReferenceScope) {
    guard !isRunning, imageReferenceScope != scope else { return }
    imageReferenceScope = scope
    if scope == .faceOnly, isImageSequence {
      detectFaces()
    } else if scope == .wholeImage {
      resetFaceReferences()
    }
  }

  func detectFaces() {
    guard !isRunning, isImageSequence else { return }
    resetFaceReferences()
    let sourceURLs = inputURLs
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "mioh-h3-face-references-\(UUID().uuidString)",
        isDirectory: true
      )
    faceReferenceDirectory = directory
    isDetectingFaces = true
    status = "参照画像から顔を検出中"
    faceDetectionTask = Task { [weak self] in
      do {
        let faces = try await MiniMaxH3FaceReferenceProcessor.detectFaces(
          in: sourceURLs,
          destinationDirectory: directory
        )
        guard !Task.isCancelled, let self else { return }
        self.faceReferences = faces
        self.isDetectingFaces = false
        self.faceDetectionTask = nil
        if faces.isEmpty {
          self.status = "顔を検出できませんでした。画像全体を使用してください"
        } else if faces.count > Self.maximumIdentityImages {
          self.status = "\(faces.count)件検出。使用する顔を最大\(Self.maximumIdentityImages)件選択してください"
        } else {
          self.status = "顔を\(faces.count)件検出しました"
        }
      } catch is CancellationError {
        guard let self else { return }
        self.isDetectingFaces = false
        self.faceDetectionTask = nil
      } catch {
        guard let self else { return }
        self.isDetectingFaces = false
        self.faceDetectionTask = nil
        self.status = "顔検出に失敗しました"
        self.appendLog("顔検出: \(error.localizedDescription)\n")
      }
    }
  }

  func setFaceReferenceSelected(_ id: String, selected: Bool) {
    guard !isRunning,
      let index = faceReferences.firstIndex(where: { $0.id == id })
    else { return }
    if selected, selectedFaceReferences.count >= Self.maximumIdentityImages {
      status = "使用できる顔参照は最大\(Self.maximumIdentityImages)件です"
      return
    }
    faceReferences[index].isSelected = selected
  }

  func setFaceReferenceSubject(_ id: String, subjectIndex: Int) {
    guard !isRunning,
      (1...Self.maximumIdentityImages).contains(subjectIndex),
      let index = faceReferences.firstIndex(where: { $0.id == id })
    else { return }
    faceReferences[index].subjectIndex = subjectIndex
  }

  func groupSelectedFacesAsOneSubject() {
    guard !isRunning else { return }
    for index in faceReferences.indices where faceReferences[index].isSelected {
      faceReferences[index].subjectIndex = 1
    }
    status = "選択した顔を人物1へまとめました"
  }

  private func resetFaceReferences() {
    faceDetectionTask?.cancel()
    faceDetectionTask = nil
    isDetectingFaces = false
    faceReferences = []
    if let faceReferenceDirectory {
      try? FileManager.default.removeItem(at: faceReferenceDirectory)
    }
    faceReferenceDirectory = nil
  }

  private func faceReferencePrompt(_ originalPrompt: String) -> String {
    MiniMaxH3FaceReferenceProcessor.faceOnlyPrompt(
      originalPrompt,
      references: selectedFaceReferences
    )
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

  private static func isPromptOnlyManifest(_ path: String) -> Bool {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
      let object = try? JSONSerialization.jsonObject(with: data),
      let dictionary = object as? [String: Any]
    else { return false }
    if let mode = dictionary["conditioningMode"] as? String {
      return mode == "fl2va"
    }
    let identifier = dictionary["modelIdentifier"] as? String ?? ""
    return identifier.localizedCaseInsensitiveContains("fl2va")
  }

  private func refreshConditioningMode() {
    let promptOnly = Self.isPromptOnlyManifest(manifestPath)
    guard supportsPromptOnly != promptOnly else { return }
    supportsPromptOnly = promptOnly
    if promptOnly {
      resetFaceReferences()
      inputURLs = []
      usesUpscalerInput = false
      ensurePromptOnlyOutputPath()
    }
  }

  func selectGenerationMode(promptOnly: Bool) {
    guard !isRunning else { return }
    let current = URL(fileURLWithPath: effectiveManifestPath)
      .standardizedFileURL
    let filename = promptOnly ? "manifest-fl2va.json" : "manifest.json"
    let candidate = current.deletingLastPathComponent()
      .appendingPathComponent(filename)
      .standardizedFileURL
    guard Self.isPipelineManifest(candidate),
      Self.isPromptOnlyManifest(candidate.path) == promptOnly
    else {
      status = promptOnly
        ? "FL2VA prompt-onlyモデルが未変換です"
        : "Ref2VA参照モデルが見つかりません"
      return
    }
    manifestPath = candidate.path
    if promptOnly {
      resetFaceReferences()
      inputURLs = []
      usesUpscalerInput = false
      ensurePromptOnlyOutputPath()
      status = "プロンプトのみ（FL2VA）"
    } else {
      usesUpscalerInput = true
      setInputURLs(
        currentUpscalerInput.map { [$0.standardizedFileURL] } ?? [],
        upscalerInput: true
      )
      status = "参照画像／動画（Ref2VA）"
    }
  }

  private func ensurePromptOnlyOutputPath() {
    guard outputPath.isEmpty || outputPath == automaticOutputPath else { return }
    let desktop = FileManager.default.urls(
      for: .desktopDirectory,
      in: .userDomainMask
    ).first!
    let proposed = desktop.appendingPathComponent(
      "10eros-max-h3-prompt-(seed).mp4"
    ).path
    outputPath = proposed
    automaticOutputPath = proposed
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
      appendLog("\(error.localizedDescription)\n")
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
    let runtimeImageURLs = isImageSequence
      && imageReferenceScope == .faceOnly
      ? selectedFaceReferences.map(\.cropURL)
      : inputURLs
    let runtimePrompt = isImageSequence
      && imageReferenceScope == .faceOnly
      ? faceReferencePrompt(prompt)
      : prompt
    var arguments = [
      "run",
      "--manifest", resolvedManifestPath,
      "--output", outputPath,
      "--prompt", runtimePrompt,
      "--cache", cache.path,
      "--backend", backend,
      "--width", String(width),
      "--height", String(height),
      "--duration", String(duration),
      "--seed", seed,
    ]
    if isImageSequence {
      do {
        let data = try JSONEncoder().encode(runtimeImageURLs.map(\.path))
        arguments += [
          "--input-images-json",
          String(decoding: data, as: UTF8.self),
        ]
      } catch {
        status = "画像入力の準備に失敗しました"
        appendLog("\(error.localizedDescription)\n")
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
        self?.consumeStandardError(data)
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
      standardErrorBuffer.removeAll(keepingCapacity: true)
      try task.run()
      process = task
      isRunning = true
    } catch {
      status = "MiniMax H3起動失敗"
      appendLog("\(error.localizedDescription)\n")
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
        appendLog("[\(event.stage)] \(event.state): \(event.message)\n")
      } else {
        let line = String(decoding: lineData, as: UTF8.self)
        // Core AI/MPSGraph can emit compiler diagnostics on stdout as well as
        // stderr. Keep the user-facing log limited to pipeline progress and
        // actionable runner errors.
        guard !isInternalCoreAIWarning(line) else { continue }
        appendLog(line + "\n")
      }
    }
  }

  private func consumeStandardError(_ data: Data) {
    standardErrorBuffer.append(data)
    while let newline = standardErrorBuffer.firstIndex(of: 0x0A) {
      let lineData = standardErrorBuffer.prefix(upTo: newline)
      standardErrorBuffer.removeSubrange(...newline)
      guard !lineData.isEmpty else { continue }
      let line = String(decoding: lineData, as: UTF8.self)
      if isInternalCoreAIWarning(line) { continue }
      appendLog(line + "\n")
    }
  }

  private func isInternalCoreAIWarning(_ line: String) -> Bool {
    let normalized = line.lowercased()
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    let fragments = [
      "#aicode.",
      "aicode.serialization",
      "ane_validation_message",
      "anecompiler",
      "aneccompile(",
      "mlir mps to anec",
      "failed: ane i/o op",
      "incompatible element type for ane",
      "ane compilation failed",
      "full compile with ane as preferred device failed",
      "no ane hash for architecture",
      "gpu-only model or wrong target",
    ]
    if fragments.contains(where: { normalized.contains($0) }) {
      return true
    }

    // On macOS 27, the serialized ANE diagnostic is sometimes split across
    // writes. Once its aicode-bearing body is removed above, Core AI leaves
    // these otherwise meaningless continuation lines behind:
    //   2026-... mioh-minimax-h3-native[pid:thread]
    //   Error:
    //   )}}}
    // Real runner failures use `mioh-minimax-h3-native: <message>` on one
    // line, so these signatures can be discarded without hiding them.
    if normalized.contains("mioh-minimax-h3-native[") {
      return true
    }
    if normalized == "error:" || normalized == "warning:" {
      return true
    }
    let diagnosticPunctuation = CharacterSet(charactersIn: "(){}[]<>,:#=)")
    if !trimmed.isEmpty,
      trimmed.unicodeScalars.allSatisfy({ diagnosticPunctuation.contains($0) })
    {
      return true
    }
    return false
  }

  private func appendLog(_ text: String) {
    log += text
    guard log.count > Self.maximumVisibleLogCharacters else { return }
    log = "…以前のログを省略…\n" + log.suffix(Self.retainedLogCharacters)
  }
}

struct MiniMaxH3GenerationView: View {
  @ObservedObject var controller: MiniMaxH3Controller
  let upscalerInputURL: URL?

  var body: some View {
    Form {
      Section("動画生成（MiniMax H3）") {
        LabeledContent("生成モード") {
          Picker(
            "",
            selection: Binding(
              get: { controller.supportsPromptOnly },
              set: { controller.selectGenerationMode(promptOnly: $0) }
            )
          ) {
            Text("参照画像／動画").tag(false)
            Text("プロンプトのみ").tag(true)
          }
          .labelsHidden()
          .pickerStyle(.segmented)
          .frame(width: 260)
        }
        LabeledContent("入力") {
          if controller.supportsPromptOnly {
            Text(controller.inputSummary)
              .foregroundStyle(.secondary)
          } else {
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
        }
        if controller.isImageSequence {
          LabeledContent("画像の参照範囲") {
            Picker(
              "",
              selection: Binding(
                get: { controller.imageReferenceScope },
                set: { controller.selectImageReferenceScope($0) }
              )
            ) {
              ForEach(MiniMaxH3ImageReferenceScope.allCases) { scope in
                Text(scope.label).tag(scope)
              }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .frame(width: 220)
            .disabled(controller.isRunning)
          }
          if controller.imageReferenceScope == .faceOnly {
            LabeledContent("顔参照") {
              HStack(spacing: 10) {
                if controller.isDetectingFaces {
                  ProgressView().controlSize(.small)
                  Text("検出中")
                } else {
                  Text(
                    "使用 \(controller.selectedFaceReferenceCount) / "
                      + "検出 \(controller.faceReferences.count)"
                  )
                    .monospacedDigit()
                  Button("再検出", action: controller.detectFaces)
                    .disabled(controller.isRunning)
                  Button(
                    "選択顔を人物1へまとめる",
                    action: controller.groupSelectedFacesAsOneSubject
                  )
                    .disabled(
                      controller.isRunning
                        || controller.selectedFaceReferenceCount == 0
                    )
                }
              }
            }
            if !controller.faceReferences.isEmpty {
              DisclosureGroup(
                "検出した顔（使用する顔と人物番号を指定）"
              ) {
                VStack(alignment: .leading, spacing: 8) {
                  ForEach(controller.faceReferences) { reference in
                    MiniMaxH3FaceReferenceRow(
                      controller: controller,
                      reference: reference
                    )
                    if reference.id != controller.faceReferences.last?.id {
                      Divider()
                    }
                  }
                }
                .padding(.vertical, 4)
              }
            }
            Text(
              "同一人物の別画像には同じ人物番号を指定してください。H3には選択した顔クロップだけを渡し、服装・姿勢・背景・構図はプロンプトから生成します。"
            )
              .font(.caption)
              .foregroundStyle(.secondary)
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
        Text(
          controller.supportsPromptOnly
            ? "FL2VAモデルが文字条件だけから映像と音声を生成します。参照画像・動画は使用しません。"
            : "動画は1本、画像は最大8枚を選択できます。参照素材は縦横比を保って中央に収め、生成動画の縦横比は下の解像度で指定します。"
        )
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
        Text(
          controller.supportsPromptOnly
            ? "自由入力（最大4152 Qwenトークン。超過分は末尾を省略）"
            : "自由入力（参照画像・動画の視覚トークンを除く範囲を使用。超過分は末尾を省略）"
        )
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Section("生成設定") {
        LabeledContent("解像度") {
          HStack(spacing: 10) {
            Picker("", selection: $controller.resolutionProfileID) {
              ForEach(MiniMaxH3Controller.resolutionProfiles) { profile in
                Text(profile.label).tag(profile.id)
              }
            }
            .labelsHidden()
            .frame(width: 230)
            Text("24fps固定")
              .font(.caption)
              .foregroundStyle(.secondary)
              .monospacedDigit()
          }
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

private struct MiniMaxH3FaceReferenceRow: View {
  @ObservedObject var controller: MiniMaxH3Controller
  let reference: MiniMaxH3FaceReference

  var body: some View {
    HStack(spacing: 10) {
      Toggle(
        "",
        isOn: Binding(
          get: { reference.isSelected },
          set: {
            controller.setFaceReferenceSelected(reference.id, selected: $0)
          }
        )
      )
        .labelsHidden()
        .disabled(controller.isRunning)
      if let preview = NSImage(contentsOf: reference.cropURL) {
        Image(nsImage: preview)
          .resizable()
          .scaledToFill()
          .frame(width: 58, height: 58)
          .clipShape(RoundedRectangle(cornerRadius: 6))
      } else {
        Image(systemName: "person.crop.square")
          .frame(width: 58, height: 58)
          .background(.quaternary)
          .clipShape(RoundedRectangle(cornerRadius: 6))
      }
      VStack(alignment: .leading, spacing: 3) {
        Text(reference.sourceLabel)
          .lineLimit(1)
          .truncationMode(.middle)
        Text("検出信頼度 \(Int((reference.confidence * 100).rounded()))%")
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
      }
      Spacer()
      Picker(
        "人物",
        selection: Binding(
          get: { reference.subjectIndex },
          set: {
            controller.setFaceReferenceSubject(
              reference.id,
              subjectIndex: $0
            )
          }
        )
      ) {
        ForEach(1...8, id: \.self) { index in
          Text("人物\(index)").tag(index)
        }
      }
        .frame(width: 100)
        .disabled(controller.isRunning || !reference.isSelected)
    }
  }
}
