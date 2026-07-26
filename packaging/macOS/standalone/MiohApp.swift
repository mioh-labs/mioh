import AppKit
import Foundation
import SwiftUI

private let appProgressPrefix = "@@LADA_PROGRESS@@"

func L(_ key: String) -> String {
  NSLocalizedString(key, comment: "")
}

struct AppProgressEvent: Decodable {
  let kind: String
  let lane: String
  let segment: Int?
  let text: String
  let percent: Double?
}

struct PlatformCapabilities {
  let supportsCoreAI: Bool

  init(
    operatingSystemVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion
  ) {
    supportsCoreAI = operatingSystemVersion.majorVersion >= 27
  }

  var defaultRestorationModel: String {
    supportsCoreAI ? "basicvsrpp-v1.2-coreai-t90" : "basicvsrpp-v1.2"
  }

  var previewRestorationModel: String {
    supportsCoreAI ? "basicvsrpp-v1.2-coreai" : "basicvsrpp-v1.2"
  }

  let baseRestorationModels = ["basicvsrpp-v1.2", "カスタム"]
  let coreAIRestorationModels = [
    "basicvsrpp-v1.2-coreai-t90", "basicvsrpp-v1.2-coreai-t36",
    "basicvsrpp-v1.2-coreai", "basicvsrpp-v1.2-coreai-variable",
  ]

  var restorationModels: [String] {
    supportsCoreAI ? coreAIRestorationModels + baseRestorationModels : baseRestorationModels
  }

  let baseDetectionModels = [
    "v2-coreml", "v3.1-fast-coreml", "v3.1-accurate-coreml",
    "v4-fast-coreml", "v4-accurate-coreml", "vr-v2-accurate-coreml", "カスタム",
  ]

  var detectionModels: [String] {
    supportsCoreAI ? baseDetectionModels + ["v4-fast-coreai"] : baseDetectionModels
  }
}

struct ROIEnhancerModelOption: Identifiable {
  let name: String
  let label: String
  let scale: Int

  var id: String { name }
}

struct MiohUserDefaultsSnapshot: Codable {
  var inputPath: String?
  var outputPath: String?
  var tempDirectory: String
  var ffmpegTempDirectory: String
  var ladaTempDirectory: String
  var overwrite: Bool

  var parallelWorkers: Int
  var executor: String
  var useSegmentCount: Bool
  var segmentCount: Int
  var segmentDuration: Int
  var mergeEncoder: String
  var deleteSegments: Bool
  var keepTemp: Bool
  var forceSplit: Bool

  var device: String
  var fp16: Bool
  var autoOptimize: Bool

  var encodingMode: String
  var encodingPreset: String
  var encoder: String
  var encoderOptions: String
  var bitrateMultiplier: Double
  var useQuality: Bool
  var quality: Int
  var useQMin: Bool
  var qmin: Int
  var useQMax: Bool
  var qmax: Int
  var useFPS: Bool
  var fps: Int
  var preFPSConversion: Bool
  var mp4FastStart: Bool

  var restorationModel: String
  var customRestorationModel: String
  var useMaxClipLength: Bool
  var maxClipLength: Int
  var useRestoreMaxFrames: Bool
  var restoreMaxFrames: Int
  var restoreTemporalOverlap: Int?
  var restoreCrossfade: Bool?
  var sharpenStrength: Double
  var detailBoost: Double
  var blendFeather: Double
  var textureMix: Double
  var smoothStrength: Double
  var effectUpscale: Int
  var roiEnhancer: String
  var roiEnhancerModel: String
  var roiEnhancerScale: Int
  var roiEnhancerStrength: Double
  var roiEnhancerTile: Int

  var detectionModel: String
  var customDetectionModel: String
  var detectionEmptyLookahead: Int
  var detectFaceMosaics: Bool

  var previewBufferLimit: Double
  var previewRestorationModel: String?
  var previewCustomRestorationModel: String?
  var previewProjectionMode: String?
  var previewVideoLayout: String?
  var previewEye: String?
  var previewCameraFOV: Double?

  var memoryCleanupInterval: Int
  var cleanupTriggerGB: Double
  var useMPSMemoryFraction: Bool
  var mpsMemoryFraction: Double
  var logMPSMemory: Bool

  static func factory(capabilities: PlatformCapabilities) -> MiohUserDefaultsSnapshot {
    MiohUserDefaultsSnapshot(
      inputPath: nil,
      outputPath: nil,
      tempDirectory: "/tmp",
      ffmpegTempDirectory: "",
      ladaTempDirectory: "",
      overwrite: false,
      parallelWorkers: 1,
      executor: "process",
      useSegmentCount: true,
      segmentCount: 4,
      segmentDuration: 60,
      mergeEncoder: "copy",
      deleteSegments: false,
      keepTemp: true,
      forceSplit: false,
      device: "mps",
      fp16: true,
      autoOptimize: true,
      encodingMode: "preset",
      encodingPreset: "hevc-apple-gpu-balanced",
      encoder: "hevc_videotoolbox",
      encoderOptions: "",
      bitrateMultiplier: 3.0,
      useQuality: false,
      quality: 70,
      useQMin: false,
      qmin: 10,
      useQMax: false,
      qmax: 30,
      useFPS: false,
      fps: 30,
      preFPSConversion: false,
      mp4FastStart: false,
      restorationModel: capabilities.defaultRestorationModel,
      customRestorationModel: "",
      useMaxClipLength: false,
      maxClipLength: 178,
      useRestoreMaxFrames: false,
      restoreMaxFrames: -1,
      restoreTemporalOverlap: 8,
      restoreCrossfade: true,
      sharpenStrength: 0.0,
      detailBoost: 0.0,
      blendFeather: 1.0,
      textureMix: 0.0,
      smoothStrength: 0.0,
      effectUpscale: 1,
      roiEnhancer: "none",
      roiEnhancerModel: "",
      roiEnhancerScale: 4,
      roiEnhancerStrength: 0.0,
      roiEnhancerTile: 0,
      detectionModel: "v2-coreml",
      customDetectionModel: "",
      detectionEmptyLookahead: 10,
      detectFaceMosaics: false,
      previewBufferLimit: 8.0,
      previewRestorationModel: capabilities.previewRestorationModel,
      previewCustomRestorationModel: "",
      previewProjectionMode: "通常",
      previewVideoLayout: "SBS 左右",
      previewEye: "左目",
      previewCameraFOV: 60,
      memoryCleanupInterval: 1,
      cleanupTriggerGB: 4.0,
      useMPSMemoryFraction: true,
      mpsMemoryFraction: 0.46,
      logMPSMemory: false
    )
  }
}

@MainActor
final class RestorationRunner: ObservableObject {
  @Published var inputURL: URL?
  @Published var outputURL: URL?
  @Published var progress = 0.0
  @Published var status = "待機中"
  @Published var log = ""
  @Published var isRunning = false
  @Published var defaultsStatus = "未保存"

  @Published var tempDirectory = "/tmp"
  @Published var ffmpegTempDirectory = ""
  @Published var ladaTempDirectory = ""
  @Published var overwrite = false

  @Published var parallelWorkers = 1
  @Published var executor = "process"
  @Published var useSegmentCount = true
  @Published var segmentCount = 4
  @Published var segmentDuration = 60
  @Published var mergeEncoder = "copy"
  @Published var deleteSegments = false
  @Published var keepTemp = true
  @Published var forceSplit = false

  @Published var device = "mps"
  @Published var fp16 = true
  @Published var autoOptimize = true

  @Published var encodingMode = "preset"
  @Published var encodingPreset = "hevc-apple-gpu-balanced"
  @Published var encoder = "hevc_videotoolbox"
  @Published var encoderOptions = ""
  @Published var bitrateMultiplier = 3.0
  @Published var useQuality = false
  @Published var quality = 70
  @Published var useQMin = false
  @Published var qmin = 10
  @Published var useQMax = false
  @Published var qmax = 30
  @Published var useFPS = false
  @Published var fps = 30
  @Published var preFPSConversion = false
  @Published var mp4FastStart = false

  @Published var restorationModel: String
  @Published var customRestorationModel = ""
  @Published var useMaxClipLength = false
  @Published var maxClipLength = 178
  @Published var useRestoreMaxFrames = false
  @Published var restoreMaxFrames = -1
  @Published var restoreTemporalOverlap = 8
  @Published var restoreCrossfade = true
  @Published var sharpenStrength = 0.0
  @Published var detailBoost = 0.0
  @Published var blendFeather = 1.0
  @Published var textureMix = 0.0
  @Published var smoothStrength = 0.0
  @Published var effectUpscale = 1
  @Published var roiEnhancer = "none"
  @Published var roiEnhancerModel = ""
  @Published var roiEnhancerScale = 4
  @Published var roiEnhancerStrength = 0.0
  @Published var roiEnhancerTile = 0

  @Published var detectionModel: String
  @Published var customDetectionModel = ""
  @Published var detectionEmptyLookahead = 10
  @Published var detectFaceMosaics = false

  @Published var previewBufferLimit = 8.0
  @Published var previewRestorationModel: String
  @Published var previewCustomRestorationModel = ""
  @Published var previewProjectionMode = "通常"
  @Published var previewVideoLayout = "SBS 左右"
  @Published var previewEye = "左目"
  @Published var previewCameraFOV = 60.0

  @Published var memoryCleanupInterval = 1
  @Published var cleanupTriggerGB = 4.0
  @Published var useMPSMemoryFraction = true
  @Published var mpsMemoryFraction = 0.46
  @Published var logMPSMemory = false

  private var process: Process?
  private var lineBuffer = ""
  private var logHistory = ""
  private var activeProgress: [String: AppProgressEvent] = [:]
  private var activeProgressOrder: [String] = []
  private let capabilities: PlatformCapabilities
  private let defaultsKey = "mioh.userProcessingDefaults.v1"

  init(capabilities: PlatformCapabilities = PlatformCapabilities()) {
    self.capabilities = capabilities
    restorationModel = capabilities.defaultRestorationModel
    previewRestorationModel = capabilities.previewRestorationModel
    detectionModel = "v2-coreml"
    loadSavedDefaultsOnLaunch()
  }

  let encodingPresets = [
    "h264-cpu-uhq", "h264-cpu-fast", "h264-apple-gpu-balanced",
    "hevc-apple-gpu-balanced", "av1-cpu-uhq",
  ]
  var restorationModels: [String] { capabilities.restorationModels }
  var detectionModels: [String] { capabilities.detectionModels }
  let enhancerModels = ["none", "realesrgan", "mewzoom", "swinir", "spandrel"]
  private let knownROIEnhancerModelNames: Set<String> = [
    "realesrgan-x2", "realesrgan-x2-coreai",
    "realesrgan-x4", "realesrgan-x4-coreml", "realesrgan-x4-coreai",
    "realesr-general-x4v3-coreml", "realesr-general-x4v3-coreai",
    "mewzoom-x4-coreml", "mewzoom-x4-coreml-512",
    "swinir-x4-coreml", "swinir-real-x4-coreml",
    "nomos-webphoto-realplksr-x4",
    "nomos-webphoto-realplksr-x4-coreml",
    "nomos-webphoto-realplksr-x4-coreai",
    "nomos-uni-span-x4", "nomos-uni-compact-x2",
  ]

  var roiEnhancerModelOptions: [ROIEnhancerModelOption] {
    var options: [ROIEnhancerModelOption]
    switch roiEnhancer {
    case "realesrgan":
      options = [
        ROIEnhancerModelOption(
          name: "realesrgan-x2-coreai",
          label: "Real-ESRGAN x2plus — Core AI (2x)",
          scale: 2
        ),
        ROIEnhancerModelOption(
          name: "realesrgan-x4-coreai",
          label: "Real-ESRGAN x4plus — Core AI (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "realesrgan-x4-coreml",
          label: "Real-ESRGAN x4plus — Core ML (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "realesr-general-x4v3-coreai",
          label: "Real-ESRGAN Compact — Core AI (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "realesr-general-x4v3-coreml",
          label: "Real-ESRGAN Compact — Core ML (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "realesrgan-x2",
          label: "Real-ESRGAN x2plus — PyTorch (2x)",
          scale: 2
        ),
        ROIEnhancerModelOption(
          name: "realesrgan-x4",
          label: "Real-ESRGAN x4plus — PyTorch (4x)",
          scale: 4
        ),
      ]
    case "mewzoom":
      options = [
        ROIEnhancerModelOption(
          name: "mewzoom-x4-coreml",
          label: "MewZoom 256 — Core ML (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "mewzoom-x4-coreml-512",
          label: "MewZoom 512 — Core ML (4x)",
          scale: 4
        ),
      ]
    case "swinir":
      options = [
        ROIEnhancerModelOption(
          name: "swinir-real-x4-coreml",
          label: "SwinIR Real — Core ML (4x)",
          scale: 4
        ),
      ]
    case "spandrel":
      options = [
        ROIEnhancerModelOption(
          name: "nomos-webphoto-realplksr-x4-coreai",
          label: "Nomos WebPhoto RealPLKSR — Core AI (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "nomos-webphoto-realplksr-x4-coreml",
          label: "Nomos WebPhoto RealPLKSR — Core ML (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "nomos-webphoto-realplksr-x4",
          label: "Nomos WebPhoto RealPLKSR — Spandrel (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "nomos-uni-span-x4",
          label: "Nomos Uni SPAN — Spandrel (4x)",
          scale: 4
        ),
        ROIEnhancerModelOption(
          name: "nomos-uni-compact-x2",
          label: "Nomos Uni Compact — Spandrel (2x)",
          scale: 2
        ),
      ]
    default:
      options = []
    }
    if !capabilities.supportsCoreAI {
      options.removeAll { $0.name.contains("coreai") }
    }
    let selected = roiEnhancerModel.trimmingCharacters(in: .whitespacesAndNewlines)
    if !selected.isEmpty && !options.contains(where: { $0.name == selected })
      && !knownROIEnhancerModelNames.contains(selected)
    {
      let displayName = URL(fileURLWithPath: selected).lastPathComponent
      options.append(
        ROIEnhancerModelOption(
          name: selected,
          label: "カスタム — \(displayName)",
          scale: roiEnhancerScale
        )
      )
    }
    return options
  }

  var canStart: Bool {
    inputURL != nil && outputURL != nil && !isRunning
  }

  func chooseInput() {
    let panel = NSOpenPanel()
    panel.title = "入力を選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    if panel.runModal() == .OK { inputURL = panel.url }
  }

  func chooseOutput() {
    let panel = NSOpenPanel()
    panel.title = "出力フォルダを選択"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    if panel.runModal() == .OK { outputURL = panel.url }
  }

  func choosePath(_ keyPath: ReferenceWritableKeyPath<RestorationRunner, String>) {
    let panel = NSOpenPanel()
    panel.canChooseFiles = true
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    if panel.runModal() == .OK { self[keyPath: keyPath] = panel.url?.path ?? "" }
  }

  func chooseROIEnhancerModel() {
    let panel = NSOpenPanel()
    panel.title = "ROIエンハンサーモデルを選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let path = panel.url?.path else { return }
    roiEnhancerModel = path
    let normalized = panel.url?.lastPathComponent.lowercased() ?? ""
    if normalized.contains("x2") || normalized.hasPrefix("2x") {
      roiEnhancerScale = 2
    } else if normalized.contains("x4") || normalized.hasPrefix("4x") {
      roiEnhancerScale = 4
    }
  }

  func selectROIEnhancer(_ enhancer: String) {
    roiEnhancer = enhancer
    synchronizeROIEnhancerModel(forceDefault: true)
  }

  func selectROIEnhancerModel(_ model: String) {
    roiEnhancerModel = model
    if let option = roiEnhancerModelOptions.first(where: { $0.name == model }) {
      roiEnhancerScale = option.scale
    }
  }

  func start() {
    guard let inputURL, let outputURL else { return }
    do {
      normalizeModelSelections()
      let resources = try resourceDirectory()
      let python = resources.appendingPathComponent("runtime/bin/python3.12")
      let processor = resources.appendingPathComponent(
        "runtime/lib/python3.12/site-packages/process_video_parallel.py"
      )
      guard FileManager.default.isExecutableFile(atPath: python.path) else {
        throw RunnerError.missingResource("Python runtime")
      }
      guard FileManager.default.fileExists(atPath: processor.path) else {
        throw RunnerError.missingResource("Parallel processor")
      }

      let task = Process()
      task.executableURL = python
      task.currentDirectoryURL = processor.deletingLastPathComponent()
      task.arguments = [processor.path] + (try processingArguments(
        resources: resources,
        input: inputURL,
        output: outputURL
      ))
      task.environment = environment(resources: resources, python: python)

      let pipe = Pipe()
      task.standardOutput = pipe
      task.standardError = pipe
      pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
        let data = handle.availableData
        guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
        Task { @MainActor in self?.consume(text) }
      }
      task.terminationHandler = { [weak self] completed in
        pipe.fileHandleForReading.readabilityHandler = nil
        Task { @MainActor in
          guard let self else { return }
          if !self.lineBuffer.isEmpty {
            let finalLine = self.lineBuffer
            self.lineBuffer = ""
            self.consumeLine(finalLine)
          }
          self.activeProgress.removeAll()
          self.activeProgressOrder.removeAll()
          self.rebuildVisibleLog()
          self.isRunning = false
          self.process = nil
          if completed.terminationStatus == 0 {
            self.progress = 1
            self.status = "完了"
          } else if completed.terminationReason == .uncaughtSignal {
            self.status = "停止"
          } else {
            self.status = "エラー"
          }
        }
      }

      lineBuffer = ""
      logHistory = ""
      activeProgress.removeAll()
      activeProgressOrder.removeAll()
      log = ""
      progress = 0
      status = "準備中"
      isRunning = true
      process = task
      try task.run()
    } catch {
      isRunning = false
      status = "エラー"
      appendLog(error.localizedDescription + "\n")
    }
  }

  func stop() {
    process?.interrupt()
    status = "停止中"
  }

  func revealOutput() {
    guard let outputURL else { return }
    NSWorkspace.shared.activateFileViewerSelecting([outputURL])
  }

  func saveCurrentDefaults() {
    do {
      let snapshot = currentDefaultsSnapshot()
      let data = try JSONEncoder().encode(snapshot)
      UserDefaults.standard.set(data, forKey: defaultsKey)
      defaultsStatus = "保存しました"
    } catch {
      defaultsStatus = "保存に失敗: \(error.localizedDescription)"
    }
  }

  func loadSavedDefaults() {
    guard let data = UserDefaults.standard.data(forKey: defaultsKey) else {
      defaultsStatus = "保存済みデフォルトはありません"
      return
    }
    do {
      let snapshot = try JSONDecoder().decode(MiohUserDefaultsSnapshot.self, from: data)
      apply(defaults: snapshot)
      defaultsStatus = "保存済みデフォルトを読み込みました"
    } catch {
      defaultsStatus = "読み込みに失敗: \(error.localizedDescription)"
    }
  }

  func resetDefaultsToFactory() {
    UserDefaults.standard.removeObject(forKey: defaultsKey)
    apply(defaults: .factory(capabilities: capabilities))
    defaultsStatus = "初期値に戻しました"
  }

  private func loadSavedDefaultsOnLaunch() {
    guard let data = UserDefaults.standard.data(forKey: defaultsKey),
      let snapshot = try? JSONDecoder().decode(MiohUserDefaultsSnapshot.self, from: data)
    else { return }
    apply(defaults: snapshot)
    defaultsStatus = "保存済みデフォルトを適用済み"
  }

  private func currentDefaultsSnapshot() -> MiohUserDefaultsSnapshot {
    MiohUserDefaultsSnapshot(
      inputPath: inputURL?.path,
      outputPath: outputURL?.path,
      tempDirectory: tempDirectory,
      ffmpegTempDirectory: ffmpegTempDirectory,
      ladaTempDirectory: ladaTempDirectory,
      overwrite: overwrite,
      parallelWorkers: parallelWorkers,
      executor: executor,
      useSegmentCount: useSegmentCount,
      segmentCount: segmentCount,
      segmentDuration: segmentDuration,
      mergeEncoder: mergeEncoder,
      deleteSegments: deleteSegments,
      keepTemp: keepTemp,
      forceSplit: forceSplit,
      device: device,
      fp16: fp16,
      autoOptimize: autoOptimize,
      encodingMode: encodingMode,
      encodingPreset: encodingPreset,
      encoder: encoder,
      encoderOptions: encoderOptions,
      bitrateMultiplier: bitrateMultiplier,
      useQuality: useQuality,
      quality: quality,
      useQMin: useQMin,
      qmin: qmin,
      useQMax: useQMax,
      qmax: qmax,
      useFPS: useFPS,
      fps: fps,
      preFPSConversion: preFPSConversion,
      mp4FastStart: mp4FastStart,
      restorationModel: restorationModel,
      customRestorationModel: customRestorationModel,
      useMaxClipLength: useMaxClipLength,
      maxClipLength: maxClipLength,
      useRestoreMaxFrames: useRestoreMaxFrames,
      restoreMaxFrames: restoreMaxFrames,
      restoreTemporalOverlap: restoreTemporalOverlap,
      restoreCrossfade: restoreCrossfade,
      sharpenStrength: sharpenStrength,
      detailBoost: detailBoost,
      blendFeather: blendFeather,
      textureMix: textureMix,
      smoothStrength: smoothStrength,
      effectUpscale: effectUpscale,
      roiEnhancer: roiEnhancer,
      roiEnhancerModel: roiEnhancerModel,
      roiEnhancerScale: roiEnhancerScale,
      roiEnhancerStrength: roiEnhancerStrength,
      roiEnhancerTile: roiEnhancerTile,
      detectionModel: detectionModel,
      customDetectionModel: customDetectionModel,
      detectionEmptyLookahead: detectionEmptyLookahead,
      detectFaceMosaics: detectFaceMosaics,
      previewBufferLimit: previewBufferLimit,
      previewRestorationModel: previewRestorationModel,
      previewCustomRestorationModel: previewCustomRestorationModel,
      previewProjectionMode: previewProjectionMode,
      previewVideoLayout: previewVideoLayout,
      previewEye: previewEye,
      previewCameraFOV: previewCameraFOV,
      memoryCleanupInterval: memoryCleanupInterval,
      cleanupTriggerGB: cleanupTriggerGB,
      useMPSMemoryFraction: useMPSMemoryFraction,
      mpsMemoryFraction: mpsMemoryFraction,
      logMPSMemory: logMPSMemory
    )
  }

  private func apply(defaults snapshot: MiohUserDefaultsSnapshot) {
    inputURL = snapshot.inputPath.flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
    outputURL = snapshot.outputPath.flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
    tempDirectory = snapshot.tempDirectory
    ffmpegTempDirectory = snapshot.ffmpegTempDirectory
    ladaTempDirectory = snapshot.ladaTempDirectory
    overwrite = snapshot.overwrite

    parallelWorkers = min(max(snapshot.parallelWorkers, 1), 16)
    executor = ["process", "thread"].contains(snapshot.executor) ? snapshot.executor : "process"
    useSegmentCount = snapshot.useSegmentCount
    segmentCount = min(max(snapshot.segmentCount, 1), 128)
    segmentDuration = min(max(snapshot.segmentDuration, 10), 3600)
    mergeEncoder = snapshot.mergeEncoder
    deleteSegments = snapshot.deleteSegments
    keepTemp = snapshot.keepTemp
    forceSplit = snapshot.forceSplit

    device = snapshot.device
    fp16 = snapshot.fp16
    autoOptimize = snapshot.autoOptimize

    encodingMode = ["auto", "preset", "custom"].contains(snapshot.encodingMode) ? snapshot.encodingMode : "preset"
    encodingPreset = encodingPresets.contains(snapshot.encodingPreset) ? snapshot.encodingPreset : "hevc-apple-gpu-balanced"
    encoder = snapshot.encoder
    encoderOptions = snapshot.encoderOptions
    bitrateMultiplier = snapshot.bitrateMultiplier
    useQuality = snapshot.useQuality
    quality = min(max(snapshot.quality, 0), 100)
    useQMin = snapshot.useQMin
    qmin = min(max(snapshot.qmin, 0), 51)
    useQMax = snapshot.useQMax
    qmax = min(max(snapshot.qmax, 0), 51)
    useFPS = snapshot.useFPS
    fps = min(max(snapshot.fps, 1), 240)
    preFPSConversion = snapshot.preFPSConversion
    mp4FastStart = snapshot.mp4FastStart

    restorationModel = restorationModels.contains(snapshot.restorationModel) ? snapshot.restorationModel : capabilities.defaultRestorationModel
    customRestorationModel = snapshot.customRestorationModel
    useMaxClipLength = snapshot.useMaxClipLength
    maxClipLength = min(max(snapshot.maxClipLength, 1), 3600)
    useRestoreMaxFrames = snapshot.useRestoreMaxFrames
    restoreMaxFrames = snapshot.restoreMaxFrames
    restoreTemporalOverlap = min(max(snapshot.restoreTemporalOverlap ?? 8, 0), 120)
    restoreCrossfade = snapshot.restoreCrossfade ?? true
    sharpenStrength = min(max(snapshot.sharpenStrength, 0), 1)
    detailBoost = min(max(snapshot.detailBoost, 0), 1)
    blendFeather = min(max(snapshot.blendFeather, 0), 3)
    textureMix = min(max(snapshot.textureMix, 0), 1)
    smoothStrength = min(max(snapshot.smoothStrength, 0), 1)
    effectUpscale = min(max(snapshot.effectUpscale, 1), 4)
    roiEnhancer = enhancerModels.contains(snapshot.roiEnhancer) ? snapshot.roiEnhancer : "none"
    roiEnhancerModel = snapshot.roiEnhancerModel
    roiEnhancerScale = min(max(snapshot.roiEnhancerScale, 1), 8)
    roiEnhancerStrength = min(max(snapshot.roiEnhancerStrength, 0), 1)
    roiEnhancerTile = min(max(snapshot.roiEnhancerTile, 0), 1024)

    detectionModel = detectionModels.contains(snapshot.detectionModel) ? snapshot.detectionModel : "v2-coreml"
    customDetectionModel = snapshot.customDetectionModel
    detectionEmptyLookahead = min(max(snapshot.detectionEmptyLookahead, 0), 300)
    detectFaceMosaics = snapshot.detectFaceMosaics

    previewBufferLimit = min(max(snapshot.previewBufferLimit, 1), 60)
    previewRestorationModel = restorationModels.contains(
      snapshot.previewRestorationModel ?? ""
    ) ? snapshot.previewRestorationModel! : capabilities.previewRestorationModel
    previewCustomRestorationModel = snapshot.previewCustomRestorationModel ?? ""
    previewProjectionMode = ["通常", "VR180", "360"].contains(snapshot.previewProjectionMode ?? "")
      ? snapshot.previewProjectionMode!
      : "通常"
    previewVideoLayout = ["Mono", "SBS 左右", "上下"].contains(snapshot.previewVideoLayout ?? "")
      ? snapshot.previewVideoLayout!
      : "SBS 左右"
    previewEye = ["左目", "右目"].contains(snapshot.previewEye ?? "")
      ? snapshot.previewEye!
      : "左目"
    previewCameraFOV = min(max(snapshot.previewCameraFOV ?? 60, 45), 105)

    memoryCleanupInterval = min(max(snapshot.memoryCleanupInterval, 1), 100)
    cleanupTriggerGB = snapshot.cleanupTriggerGB
    useMPSMemoryFraction = snapshot.useMPSMemoryFraction
    mpsMemoryFraction = snapshot.mpsMemoryFraction
    logMPSMemory = snapshot.logMPSMemory
    normalizeModelSelections()
  }

  func previewArguments(resources: URL, outputDirectory: URL, input: URL) throws -> [String] {
    normalizeModelSelections()
    let previewModel = try resolvedPreviewRestorationModel(in: resources)
    try rejectUnsupportedCoreAIModel(previewModel)
    let detection = try resolvedDetectionModel(in: resources)
    if roiEnhancer != "none" { try rejectUnsupportedCoreAIModel(roiEnhancerModel) }
    var args = ["--input", input.path, "--output-dir", outputDirectory.path]
    add(&args, "--device", device)
    args.append(fp16 ? "--fp16" : "--no-fp16")
    add(&args, "--restoration-model", previewModel)
    add(&args, "--detection-model", detection)
    let automaticClipLength: Int
    switch previewModel {
    case "basicvsrpp-v1.2-coreai": automaticClipLength = 98
    case "basicvsrpp-v1.2-coreai-t36": automaticClipLength = 104
    case "basicvsrpp-v1.2-coreai-t90": automaticClipLength = 178
    case "basicvsrpp-v1.2-coreai-variable": automaticClipLength = 180
    default: automaticClipLength = 180
    }
    add(&args, "--max-clip-length", useMaxClipLength ? maxClipLength : automaticClipLength)
    if useRestoreMaxFrames { add(&args, "--restore-max-frames", restoreMaxFrames) }
    add(&args, "--restore-temporal-overlap", restoreTemporalOverlap)
    args.append(restoreCrossfade ? "--enable-crossfade" : "--disable-crossfade")
    add(&args, "--sharpen-strength", sharpenStrength)
    add(&args, "--detail-boost", detailBoost)
    add(&args, "--blend-feather", blendFeather)
    add(&args, "--texture-mix", textureMix)
    add(&args, "--smooth-strength", smoothStrength)
    add(&args, "--roi-enhancer", roiEnhancer)
    addOptional(&args, "--roi-enhancer-model", roiEnhancerModel)
    add(&args, "--roi-enhancer-scale", roiEnhancerScale)
    add(&args, "--roi-enhancer-strength", roiEnhancerStrength)
    add(&args, "--roi-enhancer-tile", roiEnhancerTile)
    add(&args, "--effect-upscale", effectUpscale)
    add(&args, "--detection-empty-lookahead", detectionEmptyLookahead)
    addFlag(&args, "--detect-face-mosaics", detectFaceMosaics)
    add(&args, "--buffer-limit", previewBufferLimit)
    return args
  }

  private func processingArguments(resources: URL, input: URL, output: URL) throws -> [String] {
    var args = ["--input", input.path, "--output", output.path]
    add(&args, "--temp-dir", tempDirectory)
    addOptional(&args, "--ffmpeg-temp-dir", ffmpegTempDirectory)
    addOptional(&args, "--lada-temp-dir", ladaTempDirectory)
    add(&args, "--parallel-workers", parallelWorkers)
    add(&args, "--executor", executor)
    if useSegmentCount {
      add(&args, "--segment-count", segmentCount)
    } else {
      add(&args, "--segment-duration", segmentDuration)
    }
    add(&args, "--merge-encoder", mergeEncoder)
    addFlag(&args, "--delete-segments", deleteSegments)
    addFlag(&args, "--keep-temp", keepTemp)
    addFlag(&args, "--force-split", forceSplit)
    add(&args, "--device", device)
    args.append(fp16 ? "--fp16" : "--no-fp16")

    if encodingMode == "preset" {
      add(&args, "--encoding-preset", encodingPreset)
    } else if encodingMode == "custom" {
      add(&args, "--encoder", encoder)
    }
    addOptional(&args, "--encoder-options", encoderOptions)
    add(&args, "--bitrate-multiplier", bitrateMultiplier)
    if useQuality { add(&args, "--quality", quality) }
    if useQMin { add(&args, "--qmin", qmin) }
    if useQMax { add(&args, "--qmax", qmax) }
    if useFPS { add(&args, "--fps", fps) }
    addFlag(&args, "--pre-fps-conversion", useFPS && preFPSConversion)
    addFlag(&args, "--mp4-fast-start", mp4FastStart)
    args.append(autoOptimize ? "--auto-optimize" : "--no-auto-optimize")

    let restoration = try resolvedRestorationModel(in: resources)
    add(&args, "--mosaic-restoration-model", restoration)
    if useMaxClipLength { add(&args, "--max-clip-length", maxClipLength) }
    if useRestoreMaxFrames { add(&args, "--restore-max-frames", restoreMaxFrames) }
    add(&args, "--restore-temporal-overlap", restoreTemporalOverlap)
    args.append(restoreCrossfade ? "--enable-crossfade" : "--disable-crossfade")
    add(&args, "--restore-sharpen-strength", sharpenStrength)
    add(&args, "--restore-detail-boost", detailBoost)
    add(&args, "--restore-blend-feather", blendFeather)
    add(&args, "--restore-texture-mix", textureMix)
    add(&args, "--restore-smooth-strength", smoothStrength)
    add(&args, "--restore-effect-upscale", effectUpscale)
    add(&args, "--restore-roi-enhancer", roiEnhancer)
    if roiEnhancer != "none" {
      try rejectUnsupportedCoreAIModel(roiEnhancerModel)
    }
    addOptional(&args, "--restore-roi-enhancer-model-path", roiEnhancerModel)
    add(&args, "--restore-roi-enhancer-scale", roiEnhancerScale)
    add(&args, "--restore-roi-enhancer-strength", roiEnhancerStrength)
    add(&args, "--restore-roi-enhancer-tile", roiEnhancerTile)

    add(&args, "--mosaic-detection-model", try resolvedDetectionModel(in: resources))
    add(&args, "--mosaic-detection-empty-lookahead", detectionEmptyLookahead)
    args.append(detectFaceMosaics ? "--detect-face-mosaics" : "--no-detect-face-mosaics")
    add(&args, "--memory-cleanup-interval", memoryCleanupInterval)
    add(&args, "--cleanup-trigger-gb", cleanupTriggerGB)
    if useMPSMemoryFraction { add(&args, "--mps-memory-fraction", mpsMemoryFraction) }
    addFlag(&args, "--log-mps-memory", logMPSMemory)
    addFlag(&args, "--overwrite", overwrite)
    return args
  }

  private func resolvedRestorationModel(in resources: URL) throws -> String {
    if restorationModel == "カスタム" {
      guard !customRestorationModel.isEmpty else { throw RunnerError.missingValue("復元モデル") }
      try rejectUnsupportedCoreAIModel(customRestorationModel)
      return customRestorationModel
    }
    return restorationModel
  }

  private func resolvedPreviewRestorationModel(in resources: URL) throws -> String {
    if previewRestorationModel == "カスタム" {
      guard !previewCustomRestorationModel.isEmpty else {
        throw RunnerError.missingValue("再生用復元モデル")
      }
      try rejectUnsupportedCoreAIModel(previewCustomRestorationModel)
      return previewCustomRestorationModel
    }
    try rejectUnsupportedCoreAIModel(previewRestorationModel)
    return previewRestorationModel
  }

  private func resolvedDetectionModel(in resources: URL) throws -> String {
    if detectionModel == "カスタム" {
      guard !customDetectionModel.isEmpty else { throw RunnerError.missingValue("検出モデル") }
      try rejectUnsupportedCoreAIModel(customDetectionModel)
      return customDetectionModel
    }
    return detectionModel
  }

  private func normalizeModelSelections() {
    if !restorationModels.contains(restorationModel) {
      restorationModel = capabilities.defaultRestorationModel
    }
    if !restorationModels.contains(previewRestorationModel) {
      previewRestorationModel = capabilities.previewRestorationModel
    }
    if !detectionModels.contains(detectionModel) {
      detectionModel = "v2-coreml"
    }
    synchronizeROIEnhancerModel()
  }

  private func synchronizeROIEnhancerModel(forceDefault: Bool = false) {
    if roiEnhancer == "none" {
      roiEnhancerModel = ""
      return
    }
    let options = roiEnhancerModelOptions
    if !forceDefault,
      let selected = options.first(where: { $0.name == roiEnhancerModel })
    {
      roiEnhancerScale = selected.scale
      return
    }
    if !forceDefault,
      !roiEnhancerModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      !knownROIEnhancerModelNames.contains(roiEnhancerModel)
    {
      return
    }
    if let preferred = options.first {
      roiEnhancerModel = preferred.name
      roiEnhancerScale = preferred.scale
    } else {
      roiEnhancerModel = ""
    }
  }

  private func rejectUnsupportedCoreAIModel(_ model: String) throws {
    guard !capabilities.supportsCoreAI else { return }
    let normalized = model.lowercased()
    if normalized.contains("coreai")
      || normalized.hasSuffix(".aimodel")
      || normalized.hasSuffix(".aimodelc")
    {
      throw RunnerError.unsupportedFeature("CoreAIモデルにはmacOS 27以降が必要です")
    }
  }

  func environment(resources: URL, python: URL) -> [String: String] {
    var result = ProcessInfo.processInfo.environment
    let sitePackages = resources.appendingPathComponent("runtime/lib/python3.12/site-packages")
    result["PYTHONHOME"] = resources.appendingPathComponent("runtime").path
    result["PYTHONPATH"] = sitePackages.path
    result["LADA_MODEL_WEIGHTS_DIR"] = resources.appendingPathComponent("models").path
    if capabilities.supportsCoreAI {
      result["LADA_COREAI_SWIFT_RUNNER"] = resources.appendingPathComponent("bin/lada-coreai-runner").path
      result["LADA_VARIABLE_COREAI_SWIFT_RUNNER"] = resources.appendingPathComponent("bin/lada-basicvsrpp-variable-runner").path
#if MIOH_PORTABLE_COREAI
      result.removeValue(forKey: "LADA_COREAI_ARCHITECTURE")
#else
      result["LADA_COREAI_ARCHITECTURE"] = "h17s"
#endif
    } else {
      result.removeValue(forKey: "LADA_COREAI_PYTHON")
      result.removeValue(forKey: "LADA_COREAI_SWIFT_RUNNER")
      result.removeValue(forKey: "LADA_VARIABLE_COREAI_SWIFT_RUNNER")
      result.removeValue(forKey: "LADA_COREAI_ARCHITECTURE")
    }
    result["PATH"] = [resources.appendingPathComponent("bin").path, "/usr/bin", "/bin", "/usr/sbin", "/sbin"].joined(separator: ":")
    result["PYTHONUNBUFFERED"] = "1"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result["PYTHONWARNINGS"] = "ignore::SyntaxWarning"
    result["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    result["LADA_DEFORM_CONV_BACKEND"] = "mps_deform_conv"
    result["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    result["LADA_APP_PROGRESS"] = "1"
    return result
  }

  private func add(_ args: inout [String], _ option: String, _ value: String) {
    args.append(contentsOf: [option, value])
  }

  private func add<T>(_ args: inout [String], _ option: String, _ value: T) {
    args.append(contentsOf: [option, String(describing: value)])
  }

  private func addOptional(_ args: inout [String], _ option: String, _ value: String) {
    if !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { add(&args, option, value) }
  }

  private func addFlag(_ args: inout [String], _ option: String, _ enabled: Bool) {
    if enabled { args.append(option) }
  }

  private func consume(_ text: String) {
    let normalized = text
      .replacingOccurrences(of: "\r\n", with: "\n")
      .replacingOccurrences(of: "\r", with: "\n")
    lineBuffer += normalized
    while let newline = lineBuffer.firstIndex(of: "\n") {
      let line = String(lineBuffer[..<newline])
      lineBuffer.removeSubrange(...newline)
      consumeLine(line)
    }
  }

  private func consumeLine(_ line: String) {
    if line.hasPrefix(appProgressPrefix) {
      let payload = String(line.dropFirst(appProgressPrefix.count))
      if let data = payload.data(using: .utf8),
        let event = try? JSONDecoder().decode(AppProgressEvent.self, from: data),
        event.kind == "progress" || event.kind == "complete"
      {
        consumeProgressEvent(event)
        return
      }
    }
    appendLog(line + "\n")
    updateProgress(from: line)
  }

  private func consumeProgressEvent(_ event: AppProgressEvent) {
    if event.kind == "progress" {
      if activeProgress[event.lane] == nil {
        activeProgressOrder.append(event.lane)
      }
      activeProgress[event.lane] = event
      if let percent = event.percent {
        setProgress(percent)
      }
    } else if event.kind == "complete" {
      activeProgress.removeValue(forKey: event.lane)
      activeProgressOrder.removeAll { $0 == event.lane }
      if !event.text.isEmpty {
        logHistory += event.text + "\n"
        trimLogHistory()
      }
    }
    rebuildVisibleLog()
  }

  private func updateProgress(from text: String) {
    let range = NSRange(text.startIndex..., in: text)
    let patterns = [#"(?:Processing video|ビデオの処理中):\s+(\d+)%"#, #"進捗:\s+(\d+(?:\.\d+)?)%"#]
    for pattern in patterns {
      let regex = try? NSRegularExpression(pattern: pattern)
      if let match = regex?.matches(in: text, range: range).last,
        let percentRange = Range(match.range(at: 1), in: text),
        let percent = Double(text[percentRange])
      {
        setProgress(percent)
      }
    }
  }

  private func setProgress(_ percent: Double) {
    progress = min(max(percent / 100, 0), 1)
    status = "処理中 \(Int(percent))%"
  }

  func appendExternalLog(_ text: String) {
    appendLog(text)
  }

  private func appendLog(_ text: String) {
    logHistory += text
    trimLogHistory()
    rebuildVisibleLog()
  }

  private func trimLogHistory() {
    if logHistory.count > 30_000 {
      logHistory = String(logHistory.suffix(20_000))
    }
  }

  private func rebuildVisibleLog() {
    let activeLines = activeProgressOrder.compactMap { lane -> String? in
      guard let event = activeProgress[lane] else { return nil }
      if let segment = event.segment {
        return "[segment \(segment)] \(event.text)"
      }
      return event.text
    }
    if activeLines.isEmpty {
      log = logHistory
    } else {
      let separator = logHistory.isEmpty || logHistory.hasSuffix("\n") ? "" : "\n"
      log = logHistory + separator + activeLines.joined(separator: "\n")
    }
  }

  func resourceDirectory() throws -> URL {
    guard let resources = Bundle.main.resourceURL else {
      throw RunnerError.missingResource("App resources")
    }
    return resources
  }
}

enum RunnerError: LocalizedError {
  case missingResource(String)
  case missingValue(String)
  case unsupportedFeature(String)

  var errorDescription: String? {
    switch self {
    case .missingResource(let name): return String(format: L("必要なリソースが見つかりません: %@"), name)
    case .missingValue(let name): return String(format: L("値を指定してください: %@"), name)
    case .unsupportedFeature(let message): return message
    }
  }
}

struct PathRow: View {
  let title: String
  let icon: String
  let url: URL?
  let action: () -> Void

  var body: some View {
    HStack(spacing: 12) {
      Image(systemName: icon).frame(width: 20).foregroundStyle(.secondary)
      VStack(alignment: .leading, spacing: 3) {
        Text(L(title)).font(.caption).foregroundStyle(.secondary)
        Text(url?.path ?? L("未選択"))
          .lineLimit(1).truncationMode(.middle)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      Button(action: action) { Image(systemName: "folder") }
        .buttonStyle(.borderless).help(L("選択"))
    }
    .frame(minHeight: 42)
  }
}

struct PathSettingRow: View {
  let title: String
  @Binding var value: String
  let action: () -> Void

  var body: some View {
    LabeledContent(L(title)) {
      HStack {
        TextField("", text: $value, prompt: Text(L("未指定")))
          .textFieldStyle(.roundedBorder).frame(width: 380)
        Button(action: action) { Image(systemName: "folder") }
          .buttonStyle(.borderless).help(L("選択"))
      }
    }
  }
}

struct ContentView: View {
  @StateObject private var runner = RestorationRunner()
  @StateObject private var player = RealtimePlayerController()

  var body: some View {
    VStack(spacing: 0) {
      header
      Divider()
      TabView {
        basicTab.tabItem { Label("基本", systemImage: "slider.horizontal.3") }
        processingTab.tabItem { Label("分割", systemImage: "square.split.2x1") }
        restorationTab.tabItem { Label("復元", systemImage: "wand.and.stars") }
        detectionTab.tabItem { Label("検出", systemImage: "viewfinder") }
        encodingTab.tabItem { Label("出力", systemImage: "video") }
        memoryTab.tabItem { Label("メモリ", systemImage: "memorychip") }
        settingsTab.tabItem { Label("設定", systemImage: "gearshape") }
        RealtimePlayerView(controller: player, runner: runner)
          .tabItem { Label("再生", systemImage: "play.rectangle") }
        logTab.tabItem { Label("ログ", systemImage: "terminal") }
      }
      .padding(.horizontal, 14)
      ProgressView(value: runner.progress).progressViewStyle(.linear).padding(.horizontal, 20)
      Divider()
      footer
    }
    .frame(minWidth: 820, minHeight: 680)
  }

  private var header: some View {
    HStack(spacing: 12) {
      Image(nsImage: NSImage(named: "AppIcon") ?? NSImage()).resizable().frame(width: 34, height: 34)
      VStack(alignment: .leading, spacing: 2) {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
          Text("mioh").font(.title2.weight(.semibold))
          Text("Motion-Informed Optical Healing")
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        Text(runner.restorationModel).font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
      Text(L(runner.status)).font(.callout.monospacedDigit())
        .foregroundStyle(runner.status == "エラー" ? .red : .secondary)
    }
    .padding(.horizontal, 20).frame(height: 66)
  }

  private var basicTab: some View {
    Form {
      Section("ファイル") {
        PathRow(title: "入力", icon: "film", url: runner.inputURL, action: runner.chooseInput)
        PathRow(title: "出力", icon: "externaldrive", url: runner.outputURL, action: runner.chooseOutput)
        PathSettingRow(title: "一時フォルダ", value: $runner.tempDirectory) { runner.choosePath(\.tempDirectory) }
        PathSettingRow(title: "FFmpeg一時フォルダ", value: $runner.ffmpegTempDirectory) { runner.choosePath(\.ffmpegTempDirectory) }
        PathSettingRow(title: "mioh一時フォルダ", value: $runner.ladaTempDirectory) { runner.choosePath(\.ladaTempDirectory) }
      }
      Section("実行") {
        Picker("デバイス", selection: $runner.device) {
          Text("MPS").tag("mps"); Text("CPU").tag("cpu"); Text("CUDA 0").tag("cuda:0")
        }
        Toggle("FP16", isOn: $runner.fp16)
        Toggle("自動最適化", isOn: $runner.autoOptimize)
        Toggle("既存結果を上書き", isOn: $runner.overwrite)
      }
    }.formStyle(.grouped)
  }

  private var processingTab: some View {
    Form {
      Section("並列処理") {
        LabeledContent("並列数") { Stepper(value: $runner.parallelWorkers, in: 1...16) { Text("\(runner.parallelWorkers)") } }
        Picker("実行方式", selection: $runner.executor) { Text("プロセス").tag("process"); Text("スレッド").tag("thread") }
      }
      Section("セグメント") {
        Picker("分割方法", selection: $runner.useSegmentCount) {
          Text("個数").tag(true); Text("秒数").tag(false)
        }.pickerStyle(.segmented)
        if runner.useSegmentCount {
          LabeledContent("分割数") { Stepper(value: $runner.segmentCount, in: 1...128) { Text("\(runner.segmentCount)") } }
        } else {
          LabeledContent("長さ（秒）") { Stepper(value: $runner.segmentDuration, in: 10...3600, step: 10) { Text("\(runner.segmentDuration)") } }
        }
        LabeledContent("結合エンコーダー") { TextField("", text: $runner.mergeEncoder).frame(width: 220) }
        Toggle("処理済みセグメントを削除", isOn: $runner.deleteSegments)
        Toggle("一時ファイルを保持", isOn: $runner.keepTemp)
        Toggle("強制的に再分割", isOn: $runner.forceSplit)
      }
    }.formStyle(.grouped)
  }

  private var restorationTab: some View {
    Form {
      Section("モデル") {
        Picker("復元モデル", selection: $runner.restorationModel) {
          ForEach(runner.restorationModels, id: \.self) { Text(L($0)).tag($0) }
        }
        if runner.restorationModel == "カスタム" {
          PathSettingRow(title: "モデルパス", value: $runner.customRestorationModel) { runner.choosePath(\.customRestorationModel) }
        }
        Toggle("最大クリップ長を指定", isOn: $runner.useMaxClipLength)
        if runner.useMaxClipLength {
          LabeledContent("最大クリップ長") { Stepper(value: $runner.maxClipLength, in: 1...3600) { Text("\(runner.maxClipLength)") } }
        }
        Toggle("復元チャンク数を指定", isOn: $runner.useRestoreMaxFrames)
        if runner.useRestoreMaxFrames {
          LabeledContent("復元チャンク数") { TextField("", value: $runner.restoreMaxFrames, format: .number).frame(width: 110) }
        }
        LabeledContent("Temporal overlap") {
          Stepper(value: $runner.restoreTemporalOverlap, in: 0...120) {
            Text("\(runner.restoreTemporalOverlap)")
          }
        }
        Toggle("クロスフェードを有効化", isOn: $runner.restoreCrossfade)
      }
      Section("合成") {
        doubleSliderField("シャープ", value: $runner.sharpenStrength, range: 0...2, step: 0.05)
        doubleSliderField("ディテール", value: $runner.detailBoost, range: 0...1, step: 0.05)
        doubleSliderField("境界フェザー", value: $runner.blendFeather, range: 0...3, step: 0.05)
        doubleSliderField("テクスチャ", value: $runner.textureMix, range: 0...1, step: 0.01)
        doubleSliderField("スムージング", value: $runner.smoothStrength, range: 0...1, step: 0.05)
        LabeledContent("エフェクト倍率") { Stepper(value: $runner.effectUpscale, in: 1...4) { Text("\(runner.effectUpscale)x") } }
      }
      Section("ROIエンハンサー") {
        Picker("方式", selection: Binding(
          get: { runner.roiEnhancer },
          set: { runner.selectROIEnhancer($0) }
        )) {
          ForEach(runner.enhancerModels, id: \.self) { Text($0).tag($0) }
        }
        if runner.roiEnhancer != "none" {
          LabeledContent("モデル") {
            HStack {
              Picker("", selection: Binding(
                get: { runner.roiEnhancerModel },
                set: { runner.selectROIEnhancerModel($0) }
              )) {
                ForEach(runner.roiEnhancerModelOptions) { option in
                  Text(L(option.label)).tag(option.name)
                }
              }
              .labelsHidden()
              Button { runner.chooseROIEnhancerModel() } label: {
                Image(systemName: "folder")
              }
              .help("一覧にない対応モデルを選択")
            }
          }
        }
        LabeledContent("倍率") { Stepper(value: $runner.roiEnhancerScale, in: 1...8) { Text("\(runner.roiEnhancerScale)x") } }
          .disabled(runner.roiEnhancer == "none")
        doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05).disabled(runner.roiEnhancer == "none")
        integerSliderField("タイル", value: $runner.roiEnhancerTile, range: 0...1024, step: 32).disabled(runner.roiEnhancer == "none")
      }
    }.formStyle(.grouped)
  }

  private var detectionTab: some View {
    Form {
      Section("検出モデル") {
        Picker("モデル", selection: $runner.detectionModel) {
          ForEach(runner.detectionModels, id: \.self) { Text(L($0)).tag($0) }
        }
        if runner.detectionModel == "カスタム" {
          PathSettingRow(title: "モデルパス", value: $runner.customDetectionModel) { runner.choosePath(\.customDetectionModel) }
        }
        LabeledContent("空検出先読み") {
          Stepper(value: $runner.detectionEmptyLookahead, in: 0...300) { Text("\(runner.detectionEmptyLookahead)") }
        }
        Toggle("顔モザイクを検出", isOn: $runner.detectFaceMosaics)
      }
    }.formStyle(.grouped)
  }

  private var encodingTab: some View {
    Form {
      Section("エンコーダー") {
        Picker("設定方法", selection: $runner.encodingMode) {
          Text("自動").tag("auto"); Text("プリセット").tag("preset"); Text("カスタム").tag("custom")
        }.pickerStyle(.segmented)
        if runner.encodingMode == "preset" {
          Picker("プリセット", selection: $runner.encodingPreset) {
            ForEach(runner.encodingPresets, id: \.self) { Text($0).tag($0) }
          }
        } else if runner.encodingMode == "custom" {
          LabeledContent("エンコーダー") { TextField("", text: $runner.encoder).frame(width: 260) }
        }
        doubleField("ビットレート倍率", value: $runner.bitrateMultiplier)
        Toggle("MP4 Fast Start", isOn: $runner.mp4FastStart)
      }
      Section("FFmpeg詳細設定") {
        Text("追加FFmpegオプション").font(.headline)
        TextEditor(text: $runner.encoderOptions)
          .font(.system(.body, design: .monospaced))
          .frame(minHeight: 72)
          .overlay(
            RoundedRectangle(cornerRadius: 6)
              .stroke(Color.secondary.opacity(0.25))
          )
        Text("例: -pix_fmt yuv420p10le -profile:v main10 -b:v 20M")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Section("品質") {
        optionalInt("Quality", enabled: $runner.useQuality, value: $runner.quality, range: 0...100)
        optionalInt("Qmin", enabled: $runner.useQMin, value: $runner.qmin, range: 0...51)
        optionalInt("Qmax", enabled: $runner.useQMax, value: $runner.qmax, range: 0...51)
      }
      Section("フレームレート") {
        optionalInt("FPS", enabled: $runner.useFPS, value: $runner.fps, range: 1...240)
        Toggle("復元前にFPS変換", isOn: $runner.preFPSConversion).disabled(!runner.useFPS)
      }
    }.formStyle(.grouped)
  }

  private var memoryTab: some View {
    Form {
      Section("メモリ管理") {
        LabeledContent("掃除間隔") { Stepper(value: $runner.memoryCleanupInterval, in: 1...100) { Text("\(runner.memoryCleanupInterval)") } }
        doubleField("掃除開始空き容量（GB）", value: $runner.cleanupTriggerGB)
        Toggle("MPSメモリ比率を指定", isOn: $runner.useMPSMemoryFraction)
        if runner.useMPSMemoryFraction { doubleField("MPSメモリ比率", value: $runner.mpsMemoryFraction) }
        Toggle("MPSメモリ統計を記録", isOn: $runner.logMPSMemory)
      }
    }.formStyle(.grouped)
  }

  private var settingsTab: some View {
    Form {
      Section("ユーザーデフォルト") {
        Text("現在の各タブの値を、このMacユーザーのmiohデフォルトとして保存します。次回起動時に自動で適用されます。")
          .font(.callout)
          .foregroundStyle(.secondary)
        HStack(spacing: 12) {
          Button(action: runner.saveCurrentDefaults) {
            Label("現在の設定をデフォルトに保存", systemImage: "square.and.arrow.down")
          }
          .buttonStyle(.borderedProminent)
          Button(action: runner.loadSavedDefaults) {
            Label("保存済みデフォルトを読み込み", systemImage: "arrow.clockwise")
          }
          Button(role: .destructive, action: runner.resetDefaultsToFactory) {
            Label("初期値に戻す", systemImage: "trash")
          }
        }
        Text(L(runner.defaultsStatus))
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Section("保存対象") {
        Text("入力/出力、一時フォルダ、分割、復元、検出、出力、メモリ、再生バッファまで保存します。ログ、進捗、実行中状態は保存しません。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }.formStyle(.grouped)
  }

  private var logTab: some View {
    ScrollView {
      Text(runner.log.isEmpty ? " " : runner.log)
        .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .topLeading).padding(12)
    }.background(Color(nsColor: .textBackgroundColor)).padding(.vertical, 10)
  }

  private var footer: some View {
    HStack(spacing: 10) {
      Button(action: runner.revealOutput) { Image(systemName: "folder.badge.gearshape") }
        .help(L("出力をFinderで表示")).disabled(runner.outputURL == nil)
      Spacer()
      if runner.isRunning {
        Button(role: .destructive, action: runner.stop) { Label("停止", systemImage: "stop.fill") }
      } else {
        Button(action: runner.start) { Label("開始", systemImage: "play.fill") }
          .buttonStyle(.borderedProminent).disabled(!runner.canStart)
      }
    }.padding(.horizontal, 20).frame(height: 58)
  }

  private func doubleField(_ title: String, value: Binding<Double>) -> some View {
    LabeledContent(title) {
      TextField("", value: value, format: .number.precision(.fractionLength(0...3)))
        .multilineTextAlignment(.trailing).frame(width: 110)
    }
  }

  private func doubleSliderField(
    _ title: String,
    value: Binding<Double>,
    range: ClosedRange<Double>,
    step: Double
  ) -> some View {
    let clampedValue = Binding<Double>(
      get: { min(max(value.wrappedValue, range.lowerBound), range.upperBound) },
      set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }
    )
    return LabeledContent(title) {
      HStack(spacing: 12) {
        Slider(value: clampedValue, in: range, step: step)
          .frame(minWidth: 220)
        TextField("", value: clampedValue, format: .number.precision(.fractionLength(0...3)))
          .multilineTextAlignment(.trailing)
          .frame(width: 72)
      }
    }
  }

  private func integerSliderField(
    _ title: String,
    value: Binding<Int>,
    range: ClosedRange<Int>,
    step: Int
  ) -> some View {
    let clampedValue = Binding<Int>(
      get: { min(max(value.wrappedValue, range.lowerBound), range.upperBound) },
      set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }
    )
    let sliderValue = Binding<Double>(
      get: { Double(clampedValue.wrappedValue) },
      set: { clampedValue.wrappedValue = Int($0.rounded()) }
    )
    return LabeledContent(title) {
      HStack(spacing: 12) {
        Slider(
          value: sliderValue,
          in: Double(range.lowerBound)...Double(range.upperBound),
          step: Double(step)
        )
        .frame(minWidth: 220)
        TextField("", value: clampedValue, format: .number)
          .multilineTextAlignment(.trailing)
          .frame(width: 72)
      }
    }
  }

  private func optionalInt(
    _ title: String,
    enabled: Binding<Bool>,
    value: Binding<Int>,
    range: ClosedRange<Int>
  ) -> some View {
    HStack {
      Toggle(title, isOn: enabled)
      Spacer()
      Stepper(value: value, in: range) { Text("\(value.wrappedValue)").frame(width: 42, alignment: .trailing) }
        .disabled(!enabled.wrappedValue)
    }
  }
}

@main
struct MiohStandaloneApp: App {
  var body: some Scene {
    WindowGroup { ContentView() }
      .windowResizability(.contentMinSize)
      .defaultSize(width: 920, height: 760)
  }
}
