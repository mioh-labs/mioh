// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AppKit
import AVFoundation
import Foundation
import UniformTypeIdentifiers

/// Drives the standalone FlashVSR and AdcSR Core AI upscalers. No Python or
/// PyTorch process participates in this controller.
@MainActor
final class VideoUpscaleController: ObservableObject {
  static let nativeDirectoryName = "FlashVSR-v1.1-coreai-grid16"
  static let adcSRDirectoryName = "AdcSR-CoreAI"

  enum UpscalerKind: String {
    case flashVSR = "flashvsr"
    case adcSR = "adcsr"
  }

  @Published var inputURL: URL? {
    didSet {
      guard inputURL != oldValue else { return }
      refreshSourceInfo()
    }
  }
  @Published var outputURL: URL?
  @Published var sourceInfo: UpscalerMediaInfo?
  @Published var sourceInfoFailure: String?
  @Published var startSeconds = 0.0
  @Published var endSeconds = 0.0
  @Published var sizingMode = "multiple" {
    didSet {
      guard sizingMode != oldValue else { return }
      if outputIsAutomatic { updateAutomaticOutputURL() }
    }
  }
  @Published var scale = 2 {
    didSet {
      guard scale != oldValue else { return }
      if outputIsAutomatic { updateAutomaticOutputURL() }
    }
  }
  @Published var qualityMode = "fast"
  @Published var computeMode = "hybrid"
  @Published var upscalerModel = UpscalerKind.flashVSR.rawValue {
    didSet {
      guard upscalerModel != oldValue else { return }
      UserDefaults.standard.set(upscalerModel, forKey: Self.modelDefaultsKey)
      if outputIsAutomatic { updateAutomaticOutputURL() }
    }
  }
  @Published var adcSRTemporalStabilization = true
  @Published var adcSRTemporalStrength = 0.12
  @Published var targetWidth = 1920
  @Published var targetHeight = 1080
  @Published var preserveAspectRatio = true
  @Published var preserveAudio = true
  @Published var flashVSRRootPath: String {
    didSet {
      UserDefaults.standard.set(flashVSRRootPath, forKey: Self.rootDefaultsKey)
    }
  }
  @Published var adcSRRootPath: String {
    didSet {
      UserDefaults.standard.set(adcSRRootPath, forKey: Self.adcSRRootDefaultsKey)
    }
  }
  @Published var progress = 0.0
  @Published var status = "待機中"
  @Published var log = ""
  @Published var elapsedText = "—"
  @Published var estimatedRemainingText = "計算中"
  @Published var isRunning = false

  private static let rootDefaultsKey = "mioh.flashvsr.root"
  private static let adcSRRootDefaultsKey = "mioh.adcsr.root"
  private static let modelDefaultsKey = "mioh.upscaler.model"

  private var process: Process?
  private var lineBuffer = ""
  private var temporaryDirectory: URL?
  private var trimmedInputURL: URL?
  private var resultsDirectory: URL?
  private var finalOutputURL: URL?
  private var resizedOutputURL: URL?
  private var activeInstallation: Installation?
  private var cancelRequested = false
  private var outputIsAutomatic = true
  private var sourceProbeGeneration = UUID()
  private var upscaleStartedAt: Date?

  init() {
    flashVSRRootPath = UserDefaults.standard.string(
      forKey: Self.rootDefaultsKey
    ) ?? ""
    adcSRRootPath = UserDefaults.standard.string(
      forKey: Self.adcSRRootDefaultsKey
    ) ?? ""
    if let saved = UserDefaults.standard.string(forKey: Self.modelDefaultsKey),
      UpscalerKind(rawValue: saved) != nil
    {
      upscalerModel = saved
    }
  }

  var selectedUpscaler: UpscalerKind {
    UpscalerKind(rawValue: upscalerModel) ?? .flashVSR
  }

  var selectedModelReady: Bool {
    resolvedInstallation != nil
  }

  var selectedModelRootPath: String {
    selectedUpscaler == .adcSR ? adcSRRootPath : flashVSRRootPath
  }

  func applyModelSetupDestination(_ path: String) {
    let standardized = URL(fileURLWithPath: path).standardizedFileURL.path
    flashVSRRootPath = standardized
    adcSRRootPath = standardized
    objectWillChange.send()
  }

  var modelTitle: String {
    selectedUpscaler == .adcSR ? "AdcSR ×4" : "FlashVSR-v1.1 Tiny/Compact"
  }

  var durationSeconds: Double {
    guard let duration = sourceInfo?.durationSeconds,
      duration.isFinite, duration > 0
    else { return 0 }
    return duration
  }

  var selectedDurationSeconds: Double {
    max(0, normalizedEndSeconds - normalizedStartSeconds)
  }

  var normalizedStartSeconds: Double {
    min(max(0, startSeconds), max(0, durationSeconds - 0.05))
  }

  var normalizedEndSeconds: Double {
    min(max(normalizedStartSeconds + 0.05, endSeconds), durationSeconds)
  }

  var canStart: Bool {
    inputURL != nil && outputURL != nil && sourceInfo != nil
      && selectedDurationSeconds > 0 && resolvedInstallation != nil
      && customSizeError == nil
      && !isRunning
  }

  var outputResolutionText: String {
    guard sourceInfo != nil else { return "—" }
    return "\(requestedOutputWidth)×\(requestedOutputHeight)"
  }

  var customSizeError: String? {
    guard sizingMode == "custom", let sourceInfo else { return nil }
    guard targetWidth > 0, targetHeight > 0 else {
      return "出力サイズを指定してください"
    }
    guard targetWidth >= sourceInfo.width, targetHeight >= sourceInfo.height else {
      return "アップスケール先は入力解像度以上にしてください"
    }
    let ratio = max(
      Double(targetWidth) / Double(sourceInfo.width),
      Double(targetHeight) / Double(sourceInfo.height)
    )
    if ratio > 4.0 { return "指定できる最大倍率は4倍です" }
    return nil
  }

  var inferenceScale: Int {
    if selectedUpscaler == .adcSR { return 4 }
    guard sizingMode == "custom", let sourceInfo else { return scale }
    let ratio = max(
      Double(targetWidth) / Double(sourceInfo.width),
      Double(targetHeight) / Double(sourceInfo.height)
    )
    if ratio <= 2.0 { return 2 }
    return qualityMode == "quality" ? 4 : 2
  }

  var requestedOutputWidth: Int {
    guard sizingMode == "custom", sourceInfo != nil else {
      return (sourceInfo?.width ?? 0) * scale
    }
    return targetWidth
  }

  var requestedOutputHeight: Int {
    guard sizingMode == "custom", sourceInfo != nil else {
      return (sourceInfo?.height ?? 0) * scale
    }
    return targetHeight
  }

  var modelAvailabilityText: String {
    guard let installation = resolvedInstallation else {
      return "\(modelTitle) Core AIモデルが見つかりません"
    }
    return "外部 · \(nativeBundleSize(at: installation.modelDirectory))"
  }

  var runtimeText: String {
    if selectedUpscaler == .adcSR {
      let compute = computeMode == "hybrid" ? "GPU優先" : computeMode
      let temporal = adcSRTemporalStabilization
        ? "optical-flow高周波残差 \(Int((adcSRTemporalStrength * 100).rounded()))%"
        : "時間安定化なし"
      return "AdcSR one-step diffusion-GAN · Swift / \(compute) / FP32 · \(temporal)"
    }
    let compute = computeMode == "hybrid" ? "GPU + ANE hybrid" : computeMode
    return "FlashVSR-v1.1 Tiny/Compact · Swift / \(compute) / FP16"
  }

  var tileCountText: String {
    guard let sourceInfo else { return "—" }
    let side = selectedUpscaler == .adcSR ? 128 : 256 / inferenceScale
    let overlap = selectedUpscaler == .adcSR ? 16 : max(4, side / 8)
    let step = side - overlap
    func count(_ length: Int) -> Int {
      guard length > side else { return 1 }
      let regular = (length - side) / step + 1
      let lastRegular = (regular - 1) * step
      return lastRegular == length - side ? regular : regular + 1
    }
    return "\(count(sourceInfo.width) * count(sourceInfo.height))"
  }

  var scratchSpaceText: String {
    guard let sourceInfo else { return "—" }
    if selectedUpscaler == .adcSR {
      // One mmap-backed RGBA32F frame. AdcSR is frame-independent, so there
      // is no reason to retain an 85-frame temporal segment.
      let width = Int64(sourceInfo.width * 4)
      let height = Int64(sourceInfo.height * 4)
      return ByteCountFormatter.string(
        fromByteCount: width * height * 16, countStyle: .file
      )
    }
    let frames = min(
      85,
      max(1, Int(ceil(selectedDurationSeconds * sourceInfo.frameRate)))
    )
    let width = Int64(sourceInfo.width * inferenceScale)
    let height = Int64(sourceInfo.height * inferenceScale)
    let canvasBytes = width * height * 4
    let rowBytes = width * 256 * 4
    let bytes = (canvasBytes + rowBytes) * Int64(frames)
    return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
  }

  func chooseInput() {
    let panel = NSOpenPanel()
    panel.title = "アップスケールする動画を選択"
    panel.allowedContentTypes = [.movie]
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    if let inputURL {
      panel.directoryURL = inputURL.deletingLastPathComponent()
    }
    guard panel.runModal() == .OK, let url = panel.url else { return }
    outputIsAutomatic = true
    inputURL = url.standardizedFileURL
    updateAutomaticOutputURL()
  }

  func chooseOutput() {
    let panel = NSSavePanel()
    panel.title = "アップスケール動画の保存先"
    panel.allowedContentTypes = [.mpeg4Movie]
    panel.canCreateDirectories = true
    panel.nameFieldStringValue = outputURL?.lastPathComponent
      ?? automaticOutputFilename()
    if let outputURL {
      panel.directoryURL = outputURL.deletingLastPathComponent()
    } else if let inputURL {
      panel.directoryURL = inputURL.deletingLastPathComponent()
    }
    guard panel.runModal() == .OK, let url = panel.url else { return }
    outputURL = url.standardizedFileURL
    outputIsAutomatic = false
  }

  func chooseFlashVSRRoot() {
    let panel = NSOpenPanel()
    panel.title = "FlashVSR Core AIモデルのフォルダを選択"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    if !flashVSRRootPath.isEmpty {
      panel.directoryURL = URL(fileURLWithPath: flashVSRRootPath)
    }
    guard panel.runModal() == .OK, let url = panel.url else { return }
    flashVSRRootPath = url.standardizedFileURL.path
  }

  func chooseAdcSRRoot() {
    let panel = NSOpenPanel()
    panel.title = "AdcSR Core AIモデルまたは格納フォルダを選択"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    if !adcSRRootPath.isEmpty {
      panel.directoryURL = URL(fileURLWithPath: adcSRRootPath)
    }
    guard panel.runModal() == .OK, let url = panel.url else { return }
    adcSRRootPath = url.standardizedFileURL.path
  }

  func selectFullRange() {
    startSeconds = 0
    endSeconds = durationSeconds
  }

  func setTargetWidth(_ value: Int) {
    targetWidth = Self.evenDimension(value)
    if preserveAspectRatio, let sourceInfo {
      targetHeight = Self.evenDimension(
        Int((Double(targetWidth) * Double(sourceInfo.height)
          / Double(sourceInfo.width)).rounded())
      )
    }
    if outputIsAutomatic { updateAutomaticOutputURL() }
  }

  func setTargetHeight(_ value: Int) {
    targetHeight = Self.evenDimension(value)
    if preserveAspectRatio, let sourceInfo {
      targetWidth = Self.evenDimension(
        Int((Double(targetHeight) * Double(sourceInfo.width)
          / Double(sourceInfo.height)).rounded())
      )
    }
    if outputIsAutomatic { updateAutomaticOutputURL() }
  }

  func setStartSeconds(_ value: Double) {
    startSeconds = min(max(0, value), max(0, normalizedEndSeconds - 0.05))
  }

  func setSelectedDurationSeconds(_ value: Double) {
    guard value.isFinite else { return }
    let available = durationSeconds - normalizedStartSeconds
    guard available > 0 else { return }
    let minimum = min(0.05, available)
    let selected = min(max(minimum, value), available)
    endSeconds = normalizedStartSeconds + selected
  }

  func setEndSeconds(_ value: Double) {
    endSeconds = min(max(normalizedStartSeconds + 0.05, value), durationSeconds)
  }

  func start() {
    guard let inputURL, let outputURL, let resources = Bundle.main.resourceURL
    else { return }
    let start = normalizedStartSeconds
    let end = normalizedEndSeconds
    guard end > start else {
      fail("アップスケールする時間範囲が空です")
      return
    }
    guard inputURL.standardizedFileURL != outputURL.standardizedFileURL else {
      fail("入力動画と出力動画には別のファイルを指定してください")
      return
    }
    guard !FileManager.default.fileExists(atPath: outputURL.path) else {
      fail("出力ファイルはすでに存在します。別の保存先を指定してください")
      return
    }
    guard let installation = resolvedInstallation else {
      fail("\(modelTitle) Core AIモデルまたはネイティブ実行ファイルが見つかりません")
      return
    }
    let ffmpeg = resources.appendingPathComponent("bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      fail("範囲切り出し用のFFmpegが見つかりません")
      return
    }

    do {
      let work = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mioh-upscale-\(UUID().uuidString)",
        isDirectory: true
      )
      let results = work.appendingPathComponent("results", isDirectory: true)
      try FileManager.default.createDirectory(
        at: results,
        withIntermediateDirectories: true
      )
      let trimmed = work.appendingPathComponent("input-range.mov")

      temporaryDirectory = work
      trimmedInputURL = trimmed
      resultsDirectory = results
      finalOutputURL = outputURL
      activeInstallation = installation
      progress = 0
      status = "選択範囲を準備中"
      log = ""
      elapsedText = "—"
      estimatedRemainingText = "計算中"
      upscaleStartedAt = nil
      lineBuffer = ""
      cancelRequested = false
      isRunning = true

      appendLog("入力: \(inputURL.path)\n")
      appendLog(
        "範囲: \(Self.timecode(start)) 〜 \(Self.timecode(end)) "
          + "（\(Self.timecode(end - start))）\n"
      )
      appendLog(
        "モデル: \(modelTitle) / \(computeMode) / \(inferenceScale)x推論\n"
      )
      appendLog("出力サイズ: \(requestedOutputWidth)×\(requestedOutputHeight)\n")
      appendLog("実行: \(runtimeText)\n")
      appendLog("出力: \(outputURL.path)\n\n")

      let task = Process()
      task.executableURL = ffmpeg
      var arguments = [
        "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
        "-i", inputURL.path,
        "-ss", Self.number(start),
        "-t", Self.number(end - start),
        "-map", "0:v:0",
        "-vf", "setpts=PTS-STARTPTS",
        "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv444p10le",
      ]
      if preserveAudio {
        arguments += [
          "-map", "0:a:0?", "-af", "asetpts=PTS-STARTPTS",
          "-c:a", "aac", "-b:a", "192k",
        ]
      } else {
        arguments.append("-an")
      }
      arguments.append(trimmed.path)
      task.arguments = arguments
      try launch(task, phase: .trim)
    } catch {
      cleanupTemporaryFiles()
      fail(error.localizedDescription)
    }
  }

  func stop() {
    guard isRunning else { return }
    cancelRequested = true
    status = "停止中…"
    process?.terminate()
  }

  func revealOutput() {
    guard let outputURL else { return }
    NSWorkspace.shared.activateFileViewerSelecting([outputURL])
  }

  private enum ProcessPhase {
    case trim
    case upscale
    case mux
  }

  private struct Installation {
    let kind: UpscalerKind
    let runner: URL
    let modelDirectory: URL
    let resources: URL
  }

  private var resolvedInstallation: Installation? {
    guard #available(macOS 27.0, *) else { return nil }
    guard let resources = Bundle.main.resourceURL else { return nil }
    switch selectedUpscaler {
    case .flashVSR:
      return resolvedFlashVSRInstallation(resources: resources)
    case .adcSR:
      return resolvedAdcSRInstallation(resources: resources)
    }
  }

  private func resolvedFlashVSRInstallation(resources: URL) -> Installation? {
    let runner = resources.appendingPathComponent("bin/flashvsr-coreai-video")
    guard FileManager.default.isExecutableFile(atPath: runner.path) else {
      return nil
    }

    var externalRoots: [URL] = []
    if !flashVSRRootPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      externalRoots.append(URL(fileURLWithPath: flashVSRRootPath))
    }
    if let environmentRoot = ProcessInfo.processInfo.environment[
      "MIOH_FLASHVSR_ROOT"
    ], !environmentRoot.isEmpty {
      externalRoots.append(URL(fileURLWithPath: environmentRoot))
    }
    externalRoots.append(
      FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Documents/lada/model_weights", isDirectory: true)
    )
    externalRoots.append(
      FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Documents/FlashVSR_plus", isDirectory: true)
    )
    for root in externalRoots {
      for model in flashVSRModelCandidates(under: root) {
        if let installation = validateFlashVSRInstallation(
          runner: runner,
          model: model,
          resources: resources
        ) {
          return installation
        }
      }
    }
    return nil
  }

  private func flashVSRModelCandidates(under root: URL) -> [URL] {
    [
      root,
      root.appendingPathComponent(Self.nativeDirectoryName, isDirectory: true),
      root.appendingPathComponent(
        "model_weights/\(Self.nativeDirectoryName)", isDirectory: true
      ),
      root.appendingPathComponent("build/coreai-native/grid16", isDirectory: true),
    ]
  }

  private func validateFlashVSRInstallation(
    runner: URL,
    model: URL,
    resources: URL
  ) -> Installation? {
    let hybridComponents = ["patch_head", "lq_projection", "tcdecoder"]
    let hybridPresent = hybridComponents.allSatisfy { stem in
      ["\(stem).mlmodelc", "\(stem).aimodel"].contains { name in
        FileManager.default.fileExists(
          atPath: model.appendingPathComponent(name, isDirectory: true).path
        )
      }
    }
    let blocksPresent = (0..<30).allSatisfy { index in
      FileManager.default.fileExists(
        atPath: model.appendingPathComponent(
          String(format: "dit_block_%02d.aimodel", index), isDirectory: true
        ).path
      )
    }
    guard FileManager.default.isExecutableFile(atPath: runner.path),
      hybridPresent, blocksPresent
    else { return nil }
    return Installation(
      kind: .flashVSR,
      runner: runner,
      modelDirectory: model,
      resources: resources
    )
  }

  private func resolvedAdcSRInstallation(resources: URL) -> Installation? {
    let runner = resources.appendingPathComponent("bin/adcsr-coreai-video")
    guard FileManager.default.isExecutableFile(atPath: runner.path) else { return nil }
    var roots: [URL] = []
    if !adcSRRootPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      roots.append(URL(fileURLWithPath: adcSRRootPath))
    }
    if let environmentRoot = ProcessInfo.processInfo.environment["MIOH_ADCSR_ROOT"],
      !environmentRoot.isEmpty
    {
      roots.append(URL(fileURLWithPath: environmentRoot))
    }
    let documents = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent("Documents", isDirectory: true)
    roots += [
      documents.appendingPathComponent("AdcSR-CoreAI", isDirectory: true),
      documents.appendingPathComponent("lada/model_weights", isDirectory: true),
    ]
    for root in roots {
      if let model = findAdcSRModel(under: root) {
        return Installation(
          kind: .adcSR,
          runner: runner,
          modelDirectory: model,
          resources: resources
        )
      }
    }
    return nil
  }

  private func findAdcSRModel(under root: URL) -> URL? {
    let candidates: [URL]
    if root.pathExtension == "aimodel" {
      candidates = [root]
    } else {
      candidates = [
        root.appendingPathComponent("adcsr_x4_float32.aimodel", isDirectory: true),
        root.appendingPathComponent(
          "model_weights/adcsr_x4_float32.aimodel", isDirectory: true
        ),
        root.appendingPathComponent(
          "AdcSR-CoreAI/adcsr_x4_float32.aimodel", isDirectory: true
        ),
      ]
    }
    return candidates.first { candidate in
      FileManager.default.fileExists(
        atPath: candidate.appendingPathComponent("main.mlirb").path
      ) && FileManager.default.fileExists(
        atPath: candidate.appendingPathComponent("metadata.json").path
      )
    }
  }

  private func launch(_ task: Process, phase: ProcessPhase) throws {
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      guard !data.isEmpty, let text = String(data: data, encoding: .utf8)
      else { return }
      Task { @MainActor in self?.consume(text, phase: phase) }
    }
    task.terminationHandler = { [weak self, weak pipe] completed in
      pipe?.fileHandleForReading.readabilityHandler = nil
      Task { @MainActor in self?.finish(completed, phase: phase) }
    }
    process = task
    try task.run()
  }

  private func finish(_ completed: Process, phase: ProcessPhase) {
    flushLineBuffer(phase: phase)
    process = nil
    if cancelRequested {
      status = "停止"
      isRunning = false
      cleanupTemporaryFiles()
      return
    }
    guard completed.terminationStatus == 0 else {
      switch phase {
      case .trim: fail("選択範囲の切り出しに失敗しました")
      case .upscale: fail("\(modelTitle)アップスケールに失敗しました")
      case .mux: fail("音声と映像の結合に失敗しました")
      }
      cleanupTemporaryFiles()
      return
    }
    switch phase {
    case .trim:
      do { try startUpscaler() }
      catch {
        fail(error.localizedDescription)
        cleanupTemporaryFiles()
      }
    case .upscale:
      do { try finishUpscalePhase() }
      catch {
        fail(error.localizedDescription)
        cleanupTemporaryFiles()
      }
    case .mux:
      guard let resizedOutputURL else {
        fail("指定解像度の出力が見つかりません")
        cleanupTemporaryFiles()
        return
      }
      finishOutput(from: resizedOutputURL)
    }
  }

  private func startUpscaler() throws {
    guard let installation = activeInstallation,
      let trimmedInputURL, let resultsDirectory
    else { throw UpscaleControllerError.incompleteState }
    status = "\(modelTitle)でアップスケール中"
    progress = max(progress, 0.03)
    lineBuffer = ""
    upscaleStartedAt = Date()

    let task = Process()
    task.executableURL = installation.runner
    let filename = installation.kind == .adcSR
      ? "adcsr-native.mp4" : "flashvsr-native.mp4"
    var arguments = [
      "--input", trimmedInputURL.path,
      "--output", resultsDirectory.appendingPathComponent(filename).path,
      "--models", installation.modelDirectory.path,
      "--output-width", String(requestedOutputWidth),
      "--output-height", String(requestedOutputHeight),
      "--compute", computeMode,
    ]
    if installation.kind == .flashVSR {
      arguments += ["--scale", String(inferenceScale), "--seed", "0"]
    } else {
      let strength = adcSRTemporalStabilization ? adcSRTemporalStrength : 0
      arguments += ["--temporal-strength", String(format: "%.3f", strength)]
    }
    task.arguments = arguments
    try launch(task, phase: .upscale)
  }

  private func finishUpscalePhase() throws {
    guard let resultsDirectory, let installation = activeInstallation else {
      throw UpscaleControllerError.incompleteState
    }
    let filename = installation.kind == .adcSR
      ? "adcsr-native.mp4" : "flashvsr-native.mp4"
    let expected = resultsDirectory.appendingPathComponent(filename)
    guard FileManager.default.fileExists(atPath: expected.path) else {
      throw UpscaleControllerError.missingUpscalerOutput
    }
    try startFinalMux(from: expected)
  }

  private func startFinalMux(from source: URL) throws {
    guard let installation = activeInstallation,
      let temporaryDirectory, let trimmedInputURL
    else { throw UpscaleControllerError.incompleteState }
    let ffmpeg = installation.resources.appendingPathComponent("bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw UpscaleControllerError.missingFFmpeg
    }
    let resized = temporaryDirectory.appendingPathComponent("final-sized.mp4")
    resizedOutputURL = resized
    status = "音声と映像を結合中"
    progress = max(progress, 0.98)
    lineBuffer = ""
    let task = Process()
    task.executableURL = ffmpeg
    var arguments = [
      "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
      "-i", source.path,
      "-i", trimmedInputURL.path,
      "-map", "0:v:0", "-map", "1:a:0?",
      "-c:v", "copy",
    ]
    arguments += ["-c:a", "copy", "-movflags", "+faststart", resized.path]
    task.arguments = arguments
    try launch(task, phase: .mux)
  }

  private func finishOutput(from source: URL) {
    guard let outputURL = finalOutputURL else {
      fail("出力ファイルの確定に失敗しました")
      cleanupTemporaryFiles()
      return
    }
    do {
      guard !FileManager.default.fileExists(atPath: outputURL.path) else {
        throw UpscaleControllerError.outputAlreadyExists
      }
      try FileManager.default.moveItem(at: source, to: outputURL)
      progress = 1
      status = "完了"
      isRunning = false
      appendLog("\n完了: \(outputURL.path)\n")
      cleanupTemporaryFiles()
    } catch {
      fail(error.localizedDescription)
      cleanupTemporaryFiles()
    }
  }

  private func consume(_ text: String, phase: ProcessPhase) {
    let normalized = text.replacingOccurrences(of: "\r", with: "\n")
    lineBuffer += normalized
    while let newline = lineBuffer.firstIndex(of: "\n") {
      let line = String(lineBuffer[..<newline])
      lineBuffer.removeSubrange(...newline)
      consumeLine(line, phase: phase)
    }
  }

  private func flushLineBuffer(phase: ProcessPhase) {
    guard !lineBuffer.isEmpty else { return }
    let line = lineBuffer
    lineBuffer = ""
    consumeLine(line, phase: phase)
  }

  private func consumeLine(_ line: String, phase: ProcessPhase) {
    guard !line.isEmpty else { return }
    let clean = line.replacingOccurrences(
      of: "\u{001B}\\[[0-9;]*[A-Za-z]",
      with: "",
      options: .regularExpression
    )
    if phase == .upscale,
      clean.hasPrefix("PROGRESS "),
      let percent = Double(clean.dropFirst("PROGRESS ".count))
    {
      let bounded = min(100, max(0, percent))
      progress = max(progress, min(0.98, 0.03 + bounded / 100 * 0.95))
      if let started = upscaleStartedAt {
        let elapsed = max(0, Date().timeIntervalSince(started))
        elapsedText = Self.durationLabel(elapsed)
        if bounded >= 0.1, bounded < 100 {
          let remaining = elapsed * (100 - bounded) / bounded
          estimatedRemainingText = Self.durationLabel(remaining)
        } else if bounded >= 100 {
          estimatedRemainingText = "0秒"
        }
      }
    }
    if phase == .upscale, clean.hasPrefix("STAGE ") {
      status = String(clean.dropFirst("STAGE ".count))
    }
    appendLog(clean + "\n")
  }

  private func refreshSourceInfo() {
    sourceInfo = nil
    sourceInfoFailure = nil
    startSeconds = 0
    endSeconds = 0
    let generation = UUID()
    sourceProbeGeneration = generation
    guard let inputURL else { return }
    status = "動画情報を読み取り中"
    Task { [weak self] in
      let outcome = await UpscalerMediaProbe.read(url: inputURL)
      guard let self, self.sourceProbeGeneration == generation,
        self.inputURL == inputURL
      else { return }
      switch outcome {
      case .success(let info):
        self.sourceInfo = info
        self.startSeconds = 0
        self.endSeconds = max(0, info.durationSeconds)
        if self.sizingMode == "custom", self.preserveAspectRatio {
          self.setTargetHeight(self.targetHeight)
        }
        self.status = "待機中"
      case .failure(let message):
        self.sourceInfoFailure = message
        self.status = "動画を読み取れません"
      }
    }
  }

  private func updateAutomaticOutputURL() {
    guard outputIsAutomatic, let inputURL else { return }
    outputURL = inputURL.deletingLastPathComponent().appendingPathComponent(
      automaticOutputFilename()
    )
  }

  private func automaticOutputFilename() -> String {
    let stem = inputURL?.deletingPathExtension().lastPathComponent ?? "upscaled"
    let size = sizingMode == "custom"
      ? "\(targetWidth)x\(targetHeight)"
      : "\(scale)x"
    return "\(stem)-\(selectedUpscaler.rawValue)-\(size).mp4"
  }

  private func nativeBundleSize(at directory: URL) -> String {
    let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
    let enumerator = FileManager.default.enumerator(
      at: directory, includingPropertiesForKeys: Array(keys),
      options: [.skipsHiddenFiles]
    )
    var total: Int64 = 0
    while let file = enumerator?.nextObject() as? URL {
      if let values = try? file.resourceValues(forKeys: keys),
        values.isRegularFile == true
      {
        total += Int64(values.fileSize ?? 0)
      }
    }
    guard total > 0 else { return "モデル確認済み" }
    return ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
  }

  private func appendLog(_ text: String) {
    log += text
    if log.count > 200_000 {
      log.removeFirst(log.count - 160_000)
    }
  }

  private func fail(_ message: String) {
    process = nil
    status = "エラー"
    isRunning = false
    appendLog("エラー: \(message)\n")
  }

  private func cleanupTemporaryFiles() {
    guard let temporaryDirectory else { return }
    let expectedPrefix = "mioh-upscale-"
    if temporaryDirectory.deletingLastPathComponent().standardizedFileURL
      == FileManager.default.temporaryDirectory.standardizedFileURL,
      temporaryDirectory.lastPathComponent.hasPrefix(expectedPrefix)
    {
      try? FileManager.default.removeItem(at: temporaryDirectory)
    }
    self.temporaryDirectory = nil
    trimmedInputURL = nil
    resultsDirectory = nil
    finalOutputURL = nil
    resizedOutputURL = nil
    activeInstallation = nil
  }

  static func timecode(_ seconds: Double) -> String {
    guard seconds.isFinite else { return "00:00.000" }
    let value = max(0, seconds)
    let hours = Int(value) / 3600
    let minutes = (Int(value) % 3600) / 60
    let remainder = value - Double(hours * 3600 + minutes * 60)
    return hours > 0
      ? String(format: "%d:%02d:%06.3f", hours, minutes, remainder)
      : String(format: "%02d:%06.3f", minutes, remainder)
  }

  private static func durationLabel(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds >= 0 else { return "—" }
    let total = Int(seconds.rounded())
    let days = total / 86_400
    let hours = (total % 86_400) / 3_600
    let minutes = (total % 3_600) / 60
    if days > 0 { return "\(days)日 \(hours)時間" }
    if hours > 0 { return "\(hours)時間 \(minutes)分" }
    if minutes > 0 { return "\(minutes)分" }
    return "\(total)秒"
  }

  private static func number(_ value: Double) -> String {
    String(format: "%.6f", value)
  }

  private static func evenDimension(_ value: Int) -> Int {
    let positive = max(2, value)
    return positive.isMultiple(of: 2) ? positive : positive + 1
  }
}

private enum UpscaleControllerError: LocalizedError {
  case incompleteState
  case missingUpscalerOutput
  case missingFFmpeg
  case outputAlreadyExists

  var errorDescription: String? {
    switch self {
    case .incompleteState:
      return "アップスケール処理の一時状態が失われました"
    case .missingUpscalerOutput:
      return "アップスケーラーの出力動画が見つかりません"
    case .missingFFmpeg:
      return "指定解像度への変換に必要なFFmpegが見つかりません"
    case .outputAlreadyExists:
      return "出力ファイルはすでに存在します"
    }
  }
}
