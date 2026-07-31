import AppKit
import AVFoundation
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

private struct NativeExportConfiguration: Encodable {
  let mode = "export"
  let input: String
  let outputDirectory: String
  let ffmpegTemporaryDirectory: String
  let miohTemporaryDirectory: String
  let outputFile: String
  let ffmpeg: String
  let detectionModel: String
  let detectionCandidateChannels: Int
  let detectionComputeUnits: String?
  let restorationModels: String
  let restorationRunner: String
  let restorationFrameCount: Int?
  let startNanoseconds: Int64 = 0
  let generation: Int = 0
  let splitMode: String
  let segmentCount: Int
  let segmentSeconds: Double
  let bufferLimitSeconds: Double = 1
  let temporalBatchFrames: Int
  let temporalOverlap: Int
  let ringCapacity: Int
  let confidenceThreshold: Float = 0.25
  let iouThreshold: Float = 0.7
  let contextFraction: Float = 0.30
  let blendFeather: Float
  let sharpenStrength: Float
  let detailBoost: Float
  let textureMix: Float
  let smoothStrength: Float
  let effectUpscale: Int
  let detectionEmptyLookahead: Int
  let detectFaceMosaics: Bool
  let crossfade: Bool
  let targetFPS: Int?
  let targetFPSDenominator: Int?
  let preFPSConversion: Bool
  let videoCodec: String
  let averageBitRate: Int?
  let bitrateMultiplier: Double
  let mp4FastStart: Bool
}

private struct NativePreviewLaunchConfiguration: Encodable {
  let input: String
  let outputDirectory: String
  let ffmpegTemporaryDirectory: String?
  let miohTemporaryDirectory: String?
  let ffmpeg: String
  let detectionModel: String
  let detectionCandidateChannels: Int
  let detectionComputeUnits: String?
  let restorationModels: String
  let restorationRunner: String
  let restorationFrameCount: Int?
  let startNanoseconds: Int64
  let generation: Int
  let segmentSeconds: Double = 2
  let bufferLimitSeconds: Double
  let temporalBatchFrames: Int
  let ringCapacity: Int
  let confidenceThreshold: Float = 0.25
  let iouThreshold: Float = 0.7
  let contextFraction: Float = 0.30
  let blendFeather: Float
  let sharpenStrength: Float
  let detailBoost: Float
  let textureMix: Float
  let smoothStrength: Float
  let effectUpscale: Int
  let detectionEmptyLookahead: Int
  let detectFaceMosaics: Bool
  let videoCodec: String = "h264"
}

struct NativePreviewInvocation {
  let executable: URL
  let configuration: Data
  let environment: [String: String]
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
    supportsCoreAI ? "basicvsrpp-v1.2-coreai-variable" : "basicvsrpp-v1.2"
  }

  var previewDetectionModel: String {
    supportsCoreAI ? "v4-accurate-coreai" : "v4-accurate-coreml"
  }

  let baseRestorationModels = ["basicvsrpp-v1.2", "カスタム"]
  var coreAIRestorationModels: [String] {
    var models = [
      "basicvsrpp-v1.2-coreai-t90", "basicvsrpp-v1.2-coreai-t36",
      "basicvsrpp-v1.2-coreai", "basicvsrpp-v1.2-coreai-variable",
    ]
#if MIOH_DEDICATED_VARIABLE_HQ
    models.append("basicvsrpp-v1.2-coreai-variable-hq")
#endif
    return models
  }

  var restorationModels: [String] {
    supportsCoreAI ? coreAIRestorationModels + baseRestorationModels : baseRestorationModels
  }

  let baseDetectionModels = [
    "v2-coreml", "v3.1-fast-coreml", "v3.1-accurate-coreml",
    "v4-fast-coreml", "v4-accurate-coreml", "vr-v2-accurate-coreml", "カスタム",
  ]

  var detectionModels: [String] {
    guard supportsCoreAI else { return baseDetectionModels }
    var models = baseDetectionModels + [
        "v2-coreai", "v3.1-fast-coreai", "v3.1-accurate-coreai",
        "v4-fast-coreai", "v4-accurate-coreai", "vr-v2-accurate-coreai",
      ]
#if !MIOH_PORTABLE_COREAI
    models.append("jasna-v6-coreai")
    models.append("jasna-v6-large-coreai")
#endif
    return models
  }

  var previewDetectionModels: [String] {
    detectionModels.filter { !$0.hasPrefix("jasna-v6") }
  }

  /// Only the portable/universal package ships `Resources/runtime`, so only it
  /// can fall back to the Python restoration and preview path.
  var bundlesPythonRuntime: Bool {
#if MIOH_PORTABLE_COREAI
    return true
#else
    return false
#endif
  }
}

/// A selectable frame rate held as its exact rational, so the value the user
/// picks is the value the writer stamps — 29.970fps is 30000/1001 throughout.
struct FrameRateOption: Identifiable, Hashable {
  let numerator: Int
  let denominator: Int

  var key: String { "\(numerator)/\(denominator)" }
  var id: String { key }
  var value: Double { Double(numerator) / Double(max(1, denominator)) }
  /// 29.970, not 30. Whole rates stay whole so 25 does not read as 25.000,
  /// including a source reported as 30000/1000 rather than 30/1.
  var label: String {
    if abs(value - value.rounded()) < 0.0005 {
      return String(Int(value.rounded()))
    }
    return String(format: "%.3f", value)
  }

  static func isNTSC(numerator: Int, denominator: Int) -> Bool {
    guard numerator > 0, denominator > 0 else { return false }
    let pulled = Double(numerator) * 1.001 / Double(denominator)
    return abs(pulled - pulled.rounded()) < 0.001 && pulled.rounded() >= 1
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

  // Optional so preferences written before the engine selector still decode.
  var restorationEngine: String?

  var parallelWorkers: Int
  var executor: String
  var useSegmentCount: Bool
  var segmentCount: Int
  var segmentDuration: Int
  var noSplit: Bool?
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
  var fpsDenominator: Int?
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
  var previewDetectionModel: String?
  var previewCustomDetectionModel: String?
  var previewRealtimeOptimization: Bool?
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
      restorationEngine: "native",
      parallelWorkers: 1,
      executor: "process",
      useSegmentCount: true,
      segmentCount: 4,
      segmentDuration: 60,
      noSplit: false,
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
      fpsDenominator: 1,
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
      previewDetectionModel: capabilities.previewDetectionModel,
      previewCustomDetectionModel: "",
      previewRealtimeOptimization: true,
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
  @Published var inputURL: URL? {
    didSet {
      guard inputURL != oldValue else { return }
      refreshSourceFrameRate(for: inputURL)
    }
  }
  @Published var sourceFrameRate: (numerator: Int, denominator: Int)?
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

  // "native" runs the Swift Core AI pipeline. "python" runs the bundled
  // interpreter against process_video_parallel.py / mioh_preview_worker.py and
  // only exists in the portable/universal package.
  @Published var restorationEngine = "native"

  @Published var parallelWorkers = 1
  @Published var executor = "process"
  @Published var useSegmentCount = true
  @Published var segmentCount = 4
  @Published var segmentDuration = 60
  @Published var noSplit = false
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
  // The target rate is a rational: 29.970fps is 30000/1001, never 30.
  @Published var fps = 30
  @Published var fpsDenominator = 1
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
  @Published var previewDetectionModel: String
  @Published var previewCustomDetectionModel = ""
  @Published var previewRealtimeOptimization = true
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
  private var processInput: Pipe?
  private var nativeExportConfigurationURL: URL?
  private var nativeExportDirectoryURL: URL?
  private var nativeExportFFmpegDirectoryURL: URL?
  private var nativeExportMiohDirectoryURL: URL?
  private var nativeExportPreservesTemporaryFiles = false
  private var runningNativeExport = false
  private var nativeExportLastProgressBucket = -1
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
    previewDetectionModel = capabilities.previewDetectionModel
    detectionModel = "v2-coreml"
    loadSavedDefaultsOnLaunch()
  }

  let encodingPresets = [
    "h264-cpu-uhq", "h264-cpu-fast", "h264-apple-gpu-balanced",
    "hevc-apple-gpu-balanced", "av1-cpu-uhq",
  ]
  var restorationModels: [String] { capabilities.restorationModels }
  var detectionModels: [String] { capabilities.detectionModels }
  var previewDetectionModels: [String] { capabilities.previewDetectionModels }
  var supportsPythonEngine: Bool { capabilities.bundlesPythonRuntime }
  var usesPythonEngine: Bool { supportsPythonEngine && restorationEngine == "python" }

  /// The rates a user can actually pick, each as its exact rational. NTSC
  /// entries are the whole rate over 1.001 — 29.970 is 30000/1001, never 30.
  static let standardFrameRates: [FrameRateOption] = [
    FrameRateOption(numerator: 24000, denominator: 1001),
    FrameRateOption(numerator: 24, denominator: 1),
    FrameRateOption(numerator: 25, denominator: 1),
    FrameRateOption(numerator: 30000, denominator: 1001),
    FrameRateOption(numerator: 30, denominator: 1),
    FrameRateOption(numerator: 48, denominator: 1),
    FrameRateOption(numerator: 50, denominator: 1),
    FrameRateOption(numerator: 60000, denominator: 1001),
    FrameRateOption(numerator: 60, denominator: 1),
    FrameRateOption(numerator: 100, denominator: 1),
    FrameRateOption(numerator: 120000, denominator: 1001),
    FrameRateOption(numerator: 120, denominator: 1),
  ]

  var targetFrameRate: FrameRateOption {
    FrameRateOption(numerator: fps, denominator: max(1, fpsDenominator))
  }
  var targetFPSValue: Double { targetFrameRate.value }
  var targetFPSDescription: String { targetFrameRate.label }
  /// "（30000/1001）" for NTSC rates, empty for whole ones.
  var targetFPSDetail: String {
    fpsDenominator == 1 ? "" : "（\(fps)/\(fpsDenominator)）"
  }

  var selectedFrameRate: String {
    get { FrameRateOption(numerator: fps, denominator: max(1, fpsDenominator)).key }
    set {
      guard let option = frameRateOptions.first(where: { $0.key == newValue })
      else { return }
      fps = option.numerator
      fpsDenominator = option.denominator
    }
  }

  /// Conversion is down-only, and a rate never crosses between the NTSC and
  /// whole-number families: 59.940 halves to 29.970, 60 halves to 30. Once
  /// the source is known the list is narrowed to what it can actually reach.
  var frameRateOptions: [FrameRateOption] {
    var options = Self.standardFrameRates
    if let source = sourceFrameRate {
      let sourceValue = Double(source.numerator) / Double(source.denominator)
      let sourceNTSC = FrameRateOption.isNTSC(
        numerator: source.numerator,
        denominator: source.denominator
      )
      options = options.filter { option in
        guard option.value <= sourceValue + 0.001 else { return false }
        return FrameRateOption.isNTSC(
          numerator: option.numerator,
          denominator: option.denominator
        ) == sourceNTSC
      }
    }
    // Never drop the current selection out from under the picker.
    let current = FrameRateOption(
      numerator: fps,
      denominator: max(1, fpsDenominator)
    )
    if !options.contains(current) {
      options.append(current)
    }
    return options.sorted { $0.value < $1.value }
  }

  /// process_video_parallel.py takes an integer `--fps`, so the Python engine
  /// gets the nearest whole rate.
  var pythonTargetFPS: Int { max(1, Int(targetFPSValue.rounded())) }

  var sourceFPSDescription: String? {
    sourceFrameRate.map {
      FrameRateOption(numerator: $0.numerator, denominator: $0.denominator).label
    }
  }

  /// Reads the container's own timebase. `minFrameDuration` keeps 1001/60000
  /// intact where `nominalFrameRate` would already have rounded it to 59.94.
  private func refreshSourceFrameRate(for url: URL?) {
    sourceFrameRate = nil
    guard let url else { return }
    Task { [weak self] in
      let asset = AVURLAsset(url: url)
      guard
        let track = try? await asset.loadTracks(withMediaType: .video).first,
        let duration = try? await track.load(.minFrameDuration),
        duration.isValid, duration.value > 0, duration.timescale > 0
      else {
        return
      }
      let rate = (
        numerator: Int(duration.timescale),
        denominator: Int(duration.value)
      )
      await MainActor.run { [weak self] in
        guard let self, self.inputURL == url else { return }
        self.sourceFrameRate = rate
      }
    }
  }
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
      if usesPythonEngine {
        let pythonTask = try makePythonExportTask(
          resources: resources,
          input: inputURL,
          output: outputURL
        )
        try launch(pythonTask.process, pipe: pythonTask.output)
        appendPythonExportStartLog(input: inputURL, output: outputURL)
        return
      }
      let nativeOutputURL = resolvedOutputFile(
        input: inputURL,
        selectedOutput: outputURL
      )
      let nativeTask = try makeNativeExportTask(
        resources: resources,
        input: inputURL,
        output: nativeOutputURL
      )
      try launch(nativeTask.process, pipe: nativeTask.output)
      appendNativeExportStartLog(
        input: inputURL,
        output: nativeOutputURL,
        configuration: nativeTask.configuration
      )
    } catch {
      isRunning = false
      status = "エラー"
      cleanupNativeExportArtifacts()
      appendLog(error.localizedDescription + "\n")
    }
  }

  private func resolvedOutputFile(
    input: URL,
    selectedOutput: URL
  ) -> URL {
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(
      atPath: selectedOutput.path,
      isDirectory: &isDirectory
    )
    let selectedDirectory = (exists && isDirectory.boolValue)
      || selectedOutput.hasDirectoryPath
      || selectedOutput.pathExtension.isEmpty
    guard selectedDirectory else { return selectedOutput }

    let stem = input.deletingPathExtension().lastPathComponent
    let ext = input.pathExtension.isEmpty ? "mp4" : input.pathExtension
    return selectedOutput
      .appendingPathComponent("\(stem)-UC")
      .appendingPathExtension(ext)
  }

  private func launch(_ task: Process, pipe: Pipe) throws {
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
        self.processInput = nil
        self.cleanupNativeExportArtifacts()
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
    nativeExportLastProgressBucket = -1
    log = ""
    progress = 0
    status = "準備中"
    isRunning = true
    process = task
    try task.run()
  }

  func stop() {
    if runningNativeExport, let processInput {
      let command = Data("{\"command\":\"stop\"}\n".utf8)
      try? processInput.fileHandleForWriting.write(contentsOf: command)
      try? processInput.fileHandleForWriting.close()
    }
    process?.interrupt()
    status = "停止中"
  }

  private func makeNativeExportTask(
    resources: URL,
    input: URL,
    output: URL
  ) throws -> (
    process: Process,
    output: Pipe,
    configuration: NativeExportConfiguration
  ) {
    guard capabilities.supportsCoreAI else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ書き出しにはmacOS 27以降が必要です"
      )
    }
    guard [
      "basicvsrpp-v1.2-coreai",
      "basicvsrpp-v1.2-coreai-t36",
      "basicvsrpp-v1.2-coreai-t90",
      "basicvsrpp-v1.2-coreai-variable",
      "basicvsrpp-v1.2-coreai-variable-hq",
    ].contains(restorationModel)
    else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ書き出しに未対応の復元モデルです: \(restorationModel)"
      )
    }
    guard restorationModel != "カスタム",
      detectionModel != "カスタム",
      !detectionModel.hasPrefix("jasna-")
    else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ書き出しに未対応のモデル指定です"
      )
    }
    // Swift owns the decoder/detector/restorer/encoder pipeline and schedules
    // those stages concurrently inside one process. Device selection, tensor
    // precision, worker count, executor and merge strategy belonged to the
    // removed Python runtime, so they are normalized while loading defaults
    // instead of becoming runtime failure conditions.
    guard roiEnhancer == "none",
      abs(roiEnhancerStrength) < 1e-9
    else {
      throw RunnerError.unsupportedFeature(
        "ROIエンハンサーはSwiftネイティブ書き出しでは無効です。"
          + "「none・強度0」に設定してください"
      )
    }
    guard encodingMode == "preset",
      encoderOptions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      !useQuality, !useQMin, !useQMax,
      encodingPreset == "hevc-apple-gpu-balanced"
        || encodingPreset == "h264-apple-gpu-balanced"
    else {
      throw RunnerError.unsupportedFeature(
        "選択したエンコード詳細設定はSwiftネイティブ書き出しへ移行中です。"
          + "設定を無視せず停止しました"
      )
    }
    if FileManager.default.fileExists(atPath: output.path), !overwrite {
      throw RunnerError.unsupportedFeature(
        "出力ファイルが存在します: \(output.path)\n"
          + "置き換える場合は「基本」タブの「既存結果を上書き」を有効にしてください"
      )
    }

    let runner = resources.appendingPathComponent(
      "bin/mioh-native-coreai-preview"
    )
    let ffmpeg = resources.appendingPathComponent("bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: runner.path) else {
      throw RunnerError.missingResource("Swift native export runner")
    }
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw RunnerError.missingResource("FFmpeg")
    }
    guard let detection = nativeDetectionAsset(
      resources: resources,
      model: detectionModel
    ) else {
      throw RunnerError.missingResource(
        "Swift native detection model: \(detectionModel)"
      )
    }
    guard let restoration = nativeRestorationAsset(
      resources: resources,
      model: restorationModel
    ) else {
      throw RunnerError.missingResource(
        "Swift native restoration model: \(restorationModel)"
      )
    }
    let restorationRunner = resources.appendingPathComponent(
      "bin/\(restoration.runnerName)"
    )
    guard FileManager.default.isExecutableFile(
      atPath: restorationRunner.path
    ) else {
      throw RunnerError.missingResource(
        "Swift native restoration runner: \(restoration.runnerName)"
      )
    }

    let generalBase = tempDirectory.isEmpty
      ? FileManager.default.temporaryDirectory
      : URL(fileURLWithPath: tempDirectory, isDirectory: true)
    try FileManager.default.createDirectory(
      at: generalBase,
      withIntermediateDirectories: true
    )
    let exportDirectory = generalBase.appendingPathComponent(
      "mioh-swift-export-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: exportDirectory,
      withIntermediateDirectories: true
    )
    let ffmpegBase = ffmpegTempDirectory.isEmpty
      ? exportDirectory
      : URL(fileURLWithPath: ffmpegTempDirectory, isDirectory: true)
    let ffmpegWorkDirectory = ffmpegBase.appendingPathComponent(
      "mioh-ffmpeg-export-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: ffmpegWorkDirectory,
      withIntermediateDirectories: true
    )
    let miohBase = ladaTempDirectory.isEmpty
      ? FileManager.default.temporaryDirectory
      : URL(fileURLWithPath: ladaTempDirectory, isDirectory: true)
    let miohWorkDirectory = miohBase.appendingPathComponent(
      "mioh-native-runtime-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: miohWorkDirectory,
      withIntermediateDirectories: true
    )
    // Record every directory as soon as it exists so a later configuration
    // or process-launch failure is cleaned up by the common error path.
    nativeExportDirectoryURL = exportDirectory
    nativeExportFFmpegDirectoryURL = ffmpegWorkDirectory
    nativeExportMiohDirectoryURL = miohWorkDirectory
    var requestedClipFrames = useMaxClipLength
      ? max(1, maxClipLength)
      : (restoration.fixedFrameCount ?? 180)
    if useRestoreMaxFrames, restoreMaxFrames > 0 {
      requestedClipFrames = min(requestedClipFrames, restoreMaxFrames)
    }
    let clipFrames = min(
      restoration.fixedFrameCount ?? requestedClipFrames,
      requestedClipFrames
    )
    let overlap = min(
      max(0, restoreTemporalOverlap),
      max(0, clipFrames - 1)
    )
    let splitMode = noSplit ? "none" : (useSegmentCount ? "count" : "duration")
    let configuration = NativeExportConfiguration(
      input: input.path,
      outputDirectory: exportDirectory.path,
      ffmpegTemporaryDirectory: ffmpegWorkDirectory.path,
      miohTemporaryDirectory: miohWorkDirectory.path,
      outputFile: output.path,
      ffmpeg: ffmpeg.path,
      detectionModel: detection.url.path,
      detectionCandidateChannels: detection.candidateChannels,
      detectionComputeUnits: detection.computeUnits,
      restorationModels: restoration.url.path,
      restorationRunner: restorationRunner.path,
      restorationFrameCount: restoration.fixedFrameCount,
      splitMode: splitMode,
      segmentCount: max(1, segmentCount),
      segmentSeconds: Double(max(10, segmentDuration)),
      temporalBatchFrames: clipFrames,
      temporalOverlap: overlap,
      ringCapacity: max(clipFrames + overlap + 8, 32),
      blendFeather: Float(blendFeather),
      sharpenStrength: Float(sharpenStrength),
      detailBoost: Float(detailBoost),
      textureMix: Float(textureMix),
      smoothStrength: Float(smoothStrength),
      effectUpscale: effectUpscale,
      detectionEmptyLookahead: max(0, detectionEmptyLookahead),
      detectFaceMosaics: detectFaceMosaics,
      crossfade: restoreCrossfade,
      targetFPS: useFPS ? max(1, fps) : nil,
      targetFPSDenominator: useFPS ? max(1, fpsDenominator) : nil,
      preFPSConversion: preFPSConversion,
      videoCodec: encodingPreset.hasPrefix("h264") ? "h264" : "hevc",
      averageBitRate: nil,
      bitrateMultiplier: bitrateMultiplier,
      mp4FastStart: mp4FastStart
    )
    let configurationURL = exportDirectory.appendingPathComponent(
      "configuration.json"
    )
    try JSONEncoder().encode(configuration).write(
      to: configurationURL,
      options: .atomic
    )
    let task = Process()
    let inputPipe = Pipe()
    let outputPipe = Pipe()
    task.executableURL = runner
    task.arguments = [configurationURL.path]
    task.standardInput = inputPipe
    task.standardOutput = outputPipe
    task.standardError = outputPipe
    var nativeEnvironment = ProcessInfo.processInfo.environment
    nativeEnvironment["TMPDIR"] = miohWorkDirectory.path
    nativeEnvironment["TEMP"] = miohWorkDirectory.path
    nativeEnvironment["TMP"] = miohWorkDirectory.path
    task.environment = nativeEnvironment
    processInput = inputPipe
    nativeExportConfigurationURL = configurationURL
    nativeExportPreservesTemporaryFiles = keepTemp && !deleteSegments
    runningNativeExport = true
    return (task, outputPipe, configuration)
  }

  // MARK: - Bundled Python engine
  //
  // Only the portable/universal package carries Resources/runtime. These
  // entry points drive process_video_parallel.py and mioh_preview_worker.py
  // through that interpreter, which is the fallback for machines where the
  // Swift Core AI pipeline is unavailable or produces worse results.

  private func bundledPythonExecutable(resources: URL) throws -> URL {
    let python = resources.appendingPathComponent("runtime/bin/python3.12")
    guard capabilities.bundlesPythonRuntime,
      FileManager.default.isExecutableFile(atPath: python.path)
    else {
      throw RunnerError.missingResource("Python runtime")
    }
    return python
  }

  private func makePythonExportTask(
    resources: URL,
    input: URL,
    output: URL
  ) throws -> (process: Process, output: Pipe) {
    let python = try bundledPythonExecutable(resources: resources)
    let processor = resources.appendingPathComponent(
      "runtime/lib/python3.12/site-packages/process_video_parallel.py"
    )
    guard FileManager.default.fileExists(atPath: processor.path) else {
      throw RunnerError.missingResource("Parallel processor")
    }

    let task = Process()
    task.executableURL = python
    task.currentDirectoryURL = processor.deletingLastPathComponent()
    task.arguments = [processor.path] + (try processingArguments(
      resources: resources,
      input: input,
      output: output
    ))
    task.environment = environment(resources: resources, python: python)

    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    // The Python runner reports progress as plain text, not the native
    // pipeline's JSON events, and owns its own temporary directories.
    runningNativeExport = false
    processInput = nil
    return (task, pipe)
  }

  func previewArguments(
    resources: URL,
    outputDirectory: URL,
    input: URL
  ) throws -> [String] {
    normalizeModelSelections()
    let previewModel = try resolvedPreviewRestorationModel(in: resources)
    try rejectUnsupportedCoreAIModel(previewModel)
    let detection = try resolvedPreviewDetectionModel(in: resources)
    if !previewRealtimeOptimization && roiEnhancer != "none" {
      try rejectUnsupportedCoreAIModel(roiEnhancerModel)
    }
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
    case "basicvsrpp-v1.2-coreai-variable":
      automaticClipLength = previewRealtimeOptimization ? 90 : 180
    case "basicvsrpp-v1.2-coreai-variable-hq":
      automaticClipLength = 180
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
    add(&args, "--roi-enhancer", previewRealtimeOptimization ? "none" : roiEnhancer)
    if !previewRealtimeOptimization {
      addOptional(&args, "--roi-enhancer-model", roiEnhancerModel)
    }
    add(&args, "--roi-enhancer-scale", roiEnhancerScale)
    add(&args, "--roi-enhancer-strength", previewRealtimeOptimization ? 0 : roiEnhancerStrength)
    add(&args, "--roi-enhancer-tile", roiEnhancerTile)
    add(&args, "--effect-upscale", previewRealtimeOptimization ? 1 : effectUpscale)
    add(&args, "--detection-empty-lookahead", detectionEmptyLookahead)
    addFlag(&args, "--detect-face-mosaics", detectFaceMosaics)
    add(&args, "--buffer-limit", previewBufferLimit)
    if previewRealtimeOptimization { args.append("--realtime-optimize") }
    return args
  }

  private func processingArguments(resources: URL, input: URL, output: URL) throws -> [String] {
    var args = ["--input", input.path, "--output", output.path]
    add(&args, "--temp-dir", tempDirectory)
    addOptional(&args, "--ffmpeg-temp-dir", ffmpegTempDirectory)
    addOptional(&args, "--lada-temp-dir", ladaTempDirectory)
    add(&args, "--parallel-workers", parallelWorkers)
    add(&args, "--executor", executor)
    if noSplit {
      args.append("--no-split")
    } else if useSegmentCount {
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
    if useFPS { add(&args, "--fps", pythonTargetFPS) }
    addFlag(&args, "--pre-fps-conversion", useFPS && preFPSConversion && !noSplit)
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

  private func resolvedPreviewDetectionModel(in resources: URL) throws -> String {
    if previewDetectionModel == "カスタム" {
      guard !previewCustomDetectionModel.isEmpty else {
        throw RunnerError.missingValue("再生用検出モデル")
      }
      try rejectUnsupportedCoreAIModel(previewCustomDetectionModel)
      return previewCustomDetectionModel
    }
    try rejectUnsupportedCoreAIModel(previewDetectionModel)
    return previewDetectionModel
  }

  func environment(resources: URL, python: URL) -> [String: String] {
    var result = ProcessInfo.processInfo.environment
    let sitePackages = resources.appendingPathComponent("runtime/lib/python3.12/site-packages")
    result["PYTHONHOME"] = resources.appendingPathComponent("runtime").path
    result["PYTHONPATH"] = sitePackages.path
    result["LADA_MODEL_WEIGHTS_DIR"] = resources.appendingPathComponent("models").path
    result["LADA_PREVIEW_VIDEOTOOLBOX_RUNNER"] = resources
      .appendingPathComponent("bin/mioh-preview-videotoolbox-encoder").path
    if capabilities.supportsCoreAI {
      result["LADA_NATIVE_COREAI_PREVIEW_RUNNER"] = resources
        .appendingPathComponent("bin/mioh-native-coreai-preview").path
      result["LADA_NATIVE_SWIFT_PREVIEW"] = "1"
      result["LADA_COREAI_SWIFT_RUNNER"] = resources.appendingPathComponent("bin/lada-coreai-runner").path
      result["LADA_VARIABLE_COREAI_SWIFT_RUNNER"] = resources.appendingPathComponent("bin/lada-basicvsrpp-variable-runner").path
#if MIOH_DEDICATED_VARIABLE_HQ
      result["LADA_VARIABLE_COREAI_HQ_SWIFT_RUNNER"] = resources.appendingPathComponent("bin/lada-basicvsrpp-variable-hq-runner").path
#endif
#if MIOH_PORTABLE_COREAI
      result.removeValue(forKey: "LADA_COREAI_ARCHITECTURE")
#else
      result["LADA_COREAI_ARCHITECTURE"] = "h17s"
#endif
    } else {
      result.removeValue(forKey: "LADA_COREAI_PYTHON")
      result.removeValue(forKey: "LADA_NATIVE_COREAI_PREVIEW_RUNNER")
      result.removeValue(forKey: "LADA_NATIVE_SWIFT_PREVIEW")
      result.removeValue(forKey: "LADA_COREAI_SWIFT_RUNNER")
      result.removeValue(forKey: "LADA_VARIABLE_COREAI_SWIFT_RUNNER")
      result.removeValue(forKey: "LADA_VARIABLE_COREAI_HQ_SWIFT_RUNNER")
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

  private func appendPythonExportStartLog(input: URL, output: URL) {
    appendLog(
      """
      ======================================================================
      Python書き出し（バンドルランタイム）
      ======================================================================
      入力: \(input.path)
      出力: \(output.path)
      復元モデル: \(restorationModel)
      検出モデル: \(detectionModel)
      デバイス: \(device) / \(fp16 ? "FP16" : "FP32")
      並列数: \(parallelWorkers)（\(executor)）
      ======================================================================

      """
    )
  }

  private func appendNativeExportStartLog(
    input: URL,
    output: URL,
    configuration: NativeExportConfiguration
  ) {
    let codec = configuration.videoCodec.uppercased()
    let splitDescription: String
    switch configuration.splitMode {
    case "none":
      splitDescription = "分割なし"
    case "count":
      splitDescription = "\(configuration.segmentCount)分割"
    default:
      splitDescription = "\(Int(configuration.segmentSeconds))秒ごと"
    }
    appendLog(
      """
      ======================================================================
      Swiftネイティブ書き出し
      ======================================================================
      入力: \(input.path)
      出力: \(output.path)
      復元モデル: \(restorationModel)
      検出モデル: \(detectionModel)
      分割: \(splitDescription)
      復元Clip長: \(configuration.temporalBatchFrames)フレーム
      Temporal overlap: \(configuration.temporalOverlap)フレーム
      Crossfade: \(configuration.crossfade ? "有効" : "無効")
      シャープ: \(String(format: "%.2f", configuration.sharpenStrength))
      ディテール: \(String(format: "%.2f", configuration.detailBoost))
      テクスチャ: \(String(format: "%.2f", configuration.textureMix))
      スムージング: \(String(format: "%.2f", configuration.smoothStrength))
      エフェクト倍率: \(configuration.effectUpscale)x
      ROIエンハンサー: 無効
      空検出先読み: \(configuration.detectionEmptyLookahead)フレーム
      顔モザイク検出: \(configuration.detectFaceMosaics ? "有効" : "無効")
      FPS変換: \(configuration.targetFPS == nil ? "なし" : "\(targetFPSDescription)fps（\(configuration.preFPSConversion ? "復元前" : "復元後")）")
      エンコーダー: \(codec) VideoToolbox
      ビットレート倍率: \(String(format: "%.1f", configuration.bitrateMultiplier))倍
      一時フォルダ: \(configuration.outputDirectory)
      FFmpeg一時フォルダ: \(configuration.ffmpegTemporaryDirectory)
      mioh一時フォルダ: \(configuration.miohTemporaryDirectory)
      ======================================================================
      動画情報を読み込み中...

      """
    )
  }

  private func nativeLogDuration(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds >= 0 else { return "--:--" }
    let total = Int(seconds.rounded())
    let hours = total / 3600
    let minutes = (total % 3600) / 60
    let remainingSeconds = total % 60
    if hours > 0 {
      return String(
        format: "%02d:%02d:%02d",
        hours,
        minutes,
        remainingSeconds
      )
    }
    return String(format: "%02d:%02d", minutes, remainingSeconds)
  }

  private func nativeRestorationAsset(
    resources: URL,
    model: String
  ) -> (url: URL, fixedFrameCount: Int?, runnerName: String)? {
    let models = resources.appendingPathComponent("models", isDirectory: true)
    let prefix: String
    let fixedFrameCount: Int?
    let runnerName: String
    switch model {
    case "basicvsrpp-v1.2-coreai-t90":
      prefix = "basicvsrpp-v1.2-t90-fp16"
      fixedFrameCount = 90
      runnerName = "lada-coreai-runner"
    case "basicvsrpp-v1.2-coreai-t36":
      prefix = "basicvsrpp-v1.2-t36-fp16"
      fixedFrameCount = 36
      runnerName = "lada-coreai-runner"
    case "basicvsrpp-v1.2-coreai":
      prefix = "basicvsrpp-v1.2-t18-fp16"
      fixedFrameCount = 18
      runnerName = "lada-coreai-runner"
    case "basicvsrpp-v1.2-coreai-variable-hq":
      prefix = "basicvsrpp-v1.2-variable-hq-coreai"
      fixedFrameCount = nil
      runnerName = "lada-basicvsrpp-variable-hq-runner"
    case "basicvsrpp-v1.2-coreai-variable":
      prefix = "basicvsrpp-v1.2-variable-coreai"
      fixedFrameCount = nil
      runnerName = "lada-basicvsrpp-variable-runner"
    default:
      return nil
    }
    guard let url = firstModelAsset(
      in: models,
      prefixes: [prefix],
      suffix: ".aimodelc"
    ) else {
      return nil
    }
    return (url, fixedFrameCount, runnerName)
  }

  private func nativeDetectionAsset(
    resources: URL,
    model: String
  ) -> (url: URL, candidateChannels: Int, computeUnits: String?)? {
    let models = resources.appendingPathComponent("models", isDirectory: true)
    let base = model
      .replacingOccurrences(of: "-coreai", with: "")
      .replacingOccurrences(of: "-coreml", with: "")
    let stem: String
    switch base {
    case "v2": stem = "lada_mosaic_detection_model_v2"
    case "v3.1-fast": stem = "lada_mosaic_detection_model_v3.1_fast"
    case "v3.1-accurate": stem = "lada_mosaic_detection_model_v3.1_accurate"
    case "v4-fast": stem = "lada_mosaic_detection_model_v4_fast"
    case "v4-accurate": stem = "lada_mosaic_detection_model_v4_accurate"
    case "vr-v2-accurate":
      stem = "lada_mosaic_detection_model_vr_v2_accurate"
    default:
      return nil
    }
    let candidateChannels = base == "v2" ? 37 : 38
    // Detection on Core ML/ANE and restoration on Core AI avoids resource
    // contention and was faster in the measured native preview pipeline.
    if let coreML = firstModelAsset(
      in: models,
      prefixes: [stem],
      suffix: ".mlmodelc"
    ) {
      return (coreML, candidateChannels, "cpuAndNeuralEngine")
    }
    if let coreAI = firstModelAsset(
      in: models,
      prefixes: ["\(stem)-fp16", stem],
      suffix: ".aimodelc"
    ) {
      return (coreAI, candidateChannels, nil)
    }
    return nil
  }

  private func firstModelAsset(
    in directory: URL,
    prefixes: [String],
    suffix: String
  ) -> URL? {
    guard let entries = try? FileManager.default.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    ) else {
      return nil
    }
    return entries.first {
      let name = $0.lastPathComponent
      return prefixes.contains(where: { name.hasPrefix($0) })
        && name.hasSuffix(suffix)
    }
  }

  private func cleanupNativeExportArtifacts() {
    processInput = nil
    nativeExportConfigurationURL = nil
    if nativeExportPreservesTemporaryFiles {
      nativeExportFFmpegDirectoryURL = nil
      nativeExportMiohDirectoryURL = nil
      nativeExportDirectoryURL = nil
      nativeExportPreservesTemporaryFiles = false
      runningNativeExport = false
      return
    }
    if let nativeExportFFmpegDirectoryURL {
      try? FileManager.default.removeItem(at: nativeExportFFmpegDirectoryURL)
    }
    nativeExportFFmpegDirectoryURL = nil
    if let nativeExportMiohDirectoryURL {
      try? FileManager.default.removeItem(at: nativeExportMiohDirectoryURL)
    }
    nativeExportMiohDirectoryURL = nil
    if let nativeExportDirectoryURL {
      try? FileManager.default.removeItem(at: nativeExportDirectoryURL)
    }
    nativeExportDirectoryURL = nil
    nativeExportPreservesTemporaryFiles = false
    runningNativeExport = false
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
      restorationEngine: restorationEngine,
      parallelWorkers: parallelWorkers,
      executor: executor,
      useSegmentCount: useSegmentCount,
      segmentCount: segmentCount,
      segmentDuration: segmentDuration,
      noSplit: noSplit,
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
      fpsDenominator: fpsDenominator,
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
      previewDetectionModel: previewDetectionModel,
      previewCustomDetectionModel: previewCustomDetectionModel,
      previewRealtimeOptimization: previewRealtimeOptimization,
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
    restorationEngine = supportsPythonEngine
      && snapshot.restorationEngine == "python" ? "python" : "native"

    if supportsPythonEngine {
      parallelWorkers = min(max(snapshot.parallelWorkers, 1), 16)
      executor = ["process", "thread"].contains(snapshot.executor)
        ? snapshot.executor : "process"
    } else {
      // Kept in the persisted schema for backward-compatible decoding only.
      // Native export uses one process with an internally pipelined scheduler.
      parallelWorkers = 1
      executor = "process"
    }
    useSegmentCount = snapshot.useSegmentCount
    segmentCount = min(max(snapshot.segmentCount, 1), 128)
    segmentDuration = min(max(snapshot.segmentDuration, 10), 3600)
    noSplit = snapshot.noSplit ?? false
    mergeEncoder = supportsPythonEngine ? snapshot.mergeEncoder : "copy"
    deleteSegments = snapshot.deleteSegments
    keepTemp = snapshot.keepTemp
    forceSplit = snapshot.forceSplit

    if supportsPythonEngine {
      device = snapshot.device
      fp16 = snapshot.fp16
      autoOptimize = snapshot.autoOptimize
    } else {
      // Native Core AI export has one supported execution contract. Preserve
      // these fields in the on-disk schema so older preferences still decode,
      // but never let stale Python-era values make a native export fail.
      device = "mps"
      fp16 = true
      autoOptimize = true
    }

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
    fps = max(1, snapshot.fps)
    fpsDenominator = max(1, snapshot.fpsDenominator ?? 1)
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
    sharpenStrength = min(max(snapshot.sharpenStrength, 0), 2)
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
    previewDetectionModel = previewDetectionModels.contains(
      snapshot.previewDetectionModel ?? ""
    ) ? snapshot.previewDetectionModel! : capabilities.previewDetectionModel
    previewCustomDetectionModel = snapshot.previewCustomDetectionModel ?? ""
    previewRealtimeOptimization = snapshot.previewRealtimeOptimization ?? true
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

  func nativePreviewInvocation(
    resources: URL,
    outputDirectory: URL,
    input: URL,
    startNanoseconds: Int64,
    generation: Int
  ) throws -> NativePreviewInvocation {
    normalizeModelSelections()
    guard capabilities.supportsCoreAI else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ再生にはmacOS 27以降が必要です"
      )
    }
    let previewModel = capabilities.previewRestorationModel
    try rejectUnsupportedCoreAIModel(previewModel)
    let previewDetectionModel = capabilities.previewDetectionModel
    try rejectUnsupportedCoreAIModel(previewDetectionModel)
    let effectiveEnhancerStrength = previewRealtimeOptimization
      ? 0
      : roiEnhancerStrength
    let effectiveUpscale = previewRealtimeOptimization ? 1 : effectUpscale
    guard abs(effectiveEnhancerStrength) < 1e-9
    else {
      throw RunnerError.unsupportedFeature(
        "リアルタイム再生のROIエンハンサーは無効です。"
          + "強度を0に設定してください"
      )
    }
    let executable = resources.appendingPathComponent(
      "bin/mioh-native-coreai-preview"
    )
    guard FileManager.default.isExecutableFile(atPath: executable.path) else {
      throw RunnerError.missingResource("Swift native preview runner")
    }
    guard let restoration = nativeRestorationAsset(
      resources: resources,
      model: previewModel
    ) else {
      throw RunnerError.missingResource(
        "Swift native restoration model: \(previewModel)"
      )
    }
    let restorationRunner = resources.appendingPathComponent(
      "bin/\(restoration.runnerName)"
    )
    guard FileManager.default.isExecutableFile(atPath: restorationRunner.path)
    else {
      throw RunnerError.missingResource(
        "Swift native restoration runner: \(restoration.runnerName)"
      )
    }
    guard let detection = nativeDetectionAsset(
      resources: resources,
      model: previewDetectionModel
    ) else {
      throw RunnerError.missingResource(
        "Swift native detection model: \(previewDetectionModel)"
      )
    }
    let temporalLimit = detection.computeUnits == nil ? 18 : 36
    var requestedFrames = useMaxClipLength ? maxClipLength : temporalLimit
    if useRestoreMaxFrames, restoreMaxFrames > 0 {
      requestedFrames = restoreMaxFrames
    }
    let temporalFrames = min(temporalLimit, max(2, requestedFrames))
    let ffmpeg = resources.appendingPathComponent("bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw RunnerError.missingResource("FFmpeg")
    }
    let ffmpegTemporary = ffmpegTempDirectory.isEmpty
      ? outputDirectory.path
      : ffmpegTempDirectory
    let miohTemporary = ladaTempDirectory.isEmpty
      ? outputDirectory.path
      : ladaTempDirectory
    let configuration = NativePreviewLaunchConfiguration(
      input: input.path,
      outputDirectory: outputDirectory.path,
      ffmpegTemporaryDirectory: ffmpegTemporary,
      miohTemporaryDirectory: miohTemporary,
      ffmpeg: ffmpeg.path,
      detectionModel: detection.url.path,
      detectionCandidateChannels: detection.candidateChannels,
      detectionComputeUnits: detection.computeUnits,
      restorationModels: restoration.url.path,
      restorationRunner: restorationRunner.path,
      restorationFrameCount: restoration.fixedFrameCount,
      startNanoseconds: startNanoseconds,
      generation: generation,
      bufferLimitSeconds: previewBufferLimit,
      temporalBatchFrames: temporalFrames,
      ringCapacity: max(temporalFrames * 2, 24),
      blendFeather: Float(blendFeather),
      sharpenStrength: Float(sharpenStrength),
      detailBoost: Float(detailBoost),
      textureMix: Float(textureMix),
      smoothStrength: Float(smoothStrength),
      effectUpscale: effectiveUpscale,
      detectionEmptyLookahead: max(0, detectionEmptyLookahead),
      detectFaceMosaics: detectFaceMosaics
    )
    var environment = ProcessInfo.processInfo.environment
    environment["PATH"] = [
      resources.appendingPathComponent("bin").path,
      "/usr/bin", "/bin", "/usr/sbin", "/sbin",
    ].joined(separator: ":")
    environment["TMPDIR"] = miohTemporary
    environment["TEMP"] = miohTemporary
    environment["TMP"] = miohTemporary
    return NativePreviewInvocation(
      executable: executable,
      configuration: try JSONEncoder().encode(configuration),
      environment: environment
    )
  }

  private func normalizeModelSelections() {
    if restorationModel == "basicvsrpp-v1.2-coreai-variable-chunk6" {
      restorationModel = "basicvsrpp-v1.2-coreai-variable"
    }
    if previewRestorationModel == "basicvsrpp-v1.2-coreai-variable-chunk6" {
      previewRestorationModel = "basicvsrpp-v1.2-coreai-variable"
    }
    if !restorationModels.contains(restorationModel) {
      restorationModel = capabilities.defaultRestorationModel
    }
    if !restorationModels.contains(previewRestorationModel) {
      previewRestorationModel = capabilities.previewRestorationModel
    }
    if !previewDetectionModels.contains(previewDetectionModel) {
      previewDetectionModel = capabilities.previewDetectionModel
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
    if runningNativeExport,
      let data = line.data(using: .utf8),
      let payload = try? JSONSerialization.jsonObject(with: data)
        as? [String: Any],
      let kind = payload["kind"] as? String
    {
      switch kind {
      case "ready":
        let width = payload["width"] as? Int ?? 0
        let height = payload["height"] as? Int ?? 0
        let fps = payload["fps"] as? Double ?? 0
        let duration = payload["duration"] as? Double ?? 0
        appendLog(
          String(
            format:
              "動画情報: %dx%d / %.3ffps / %@\nSwift処理を開始しました\n\n",
            width,
            height,
            fps,
            nativeLogDuration(duration)
          )
        )
        return
      case "export_progress":
        if let percent = payload["percent"] as? Double {
          setProgress(percent)
          let bucket = Int(percent) / 5
          if bucket > nativeExportLastProgressBucket, bucket > 0 {
            nativeExportLastProgressBucket = bucket
            let position = payload["position_seconds"] as? Double ?? 0
            let duration = payload["duration_seconds"] as? Double ?? 0
            let frames = payload["encoded_frames"] as? Int ?? 0
            let fps = payload["throughput_fps"] as? Double ?? 0
            let eta = payload["eta_seconds"] as? Double ?? 0
            appendLog(
              String(
                format:
                  "進捗: %3d%% | %@ / %@ | %dフレーム | %.1ffps | 残り %@\n",
                min(100, Int(percent.rounded(.down))),
                nativeLogDuration(position),
                nativeLogDuration(duration),
                frames,
                fps,
                nativeLogDuration(eta)
              )
            )
          }
        }
        return
      case "export_finalizing":
        status = "音声を結合中"
        appendLog("\n映像処理完了。音声を結合して出力を確定中...\n")
        return
      case "segment":
        return
      case "progress":
        return
      case "native_stats":
        let fps = payload["throughput_fps"] as? Double ?? 0
        let frames = payload["decoded_frames"] as? Int ?? 0
        let elapsed = payload["elapsed_seconds"] as? Double ?? 0
        let detected = payload["detected_frames"] as? Int ?? 0
        let batches = payload["restored_batches"] as? Int ?? 0
        appendLog(
          String(
            format:
              "処理統計: %dフレーム / 検出対象%dフレーム / 復元%dクリップ / %.1ffps / 経過 %@\n",
            frames,
            detected,
            batches,
            fps,
            nativeLogDuration(elapsed)
          )
        )
        return
      case "ended":
        let output = payload["output"] as? String ?? outputURL?.path ?? ""
        appendLog(
          """

          ======================================================================
          書き出し完了
          出力: \(output)
          ======================================================================

          """
        )
        return
      case "error":
        let message = payload["message"] as? String ?? "Swift書き出しエラー"
        let detail = payload["detail"] as? String ?? ""
        appendLog("\(message): \(detail)\n")
        return
      default:
        break
      }
    }
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
        if runner.supportsPythonEngine {
          Picker("実行エンジン", selection: $runner.restorationEngine) {
            Text("Swiftネイティブ / Core AI").tag("native")
            Text("Python（バンドルランタイム）").tag("python")
          }
          Text(runner.usesPythonEngine
            ? "バンドルされたPython 3.12でprocess_video_parallel.pyを実行します"
            : "デコードから書き出しまでを1つのSwiftプロセスで実行します")
            .font(.caption)
            .foregroundStyle(.secondary)
        } else {
          LabeledContent("実行エンジン") {
            Text("Swiftネイティブ / Core AI")
          }
        }
        if runner.usesPythonEngine {
          Picker("デバイス", selection: $runner.device) {
            Text("MPS").tag("mps"); Text("CPU").tag("cpu"); Text("CUDA 0").tag("cuda:0")
          }
          Toggle("FP16", isOn: $runner.fp16)
          Toggle("自動最適化", isOn: $runner.autoOptimize)
        } else {
          LabeledContent("精度・最適化") {
            Text("FP16 / Apple Silicon自動最適化")
          }
        }
        Toggle("既存結果を上書き", isOn: $runner.overwrite)
      }
    }.formStyle(.grouped)
  }

  private var processingTab: some View {
    Form {
      Section("並列処理") {
        if runner.usesPythonEngine {
          LabeledContent("並列数") { Stepper(value: $runner.parallelWorkers, in: 1...16) { Text("\(runner.parallelWorkers)") } }
            .disabled(runner.noSplit)
          Picker("実行方式", selection: $runner.executor) { Text("プロセス").tag("process"); Text("スレッド").tag("thread") }
            .disabled(runner.noSplit)
        } else {
          LabeledContent("実行方式") {
            Text("Swiftネイティブ（自動段階並列）")
          }
          Text("デコード・検出・復元・エンコードを1プロセス内で並行実行します")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }
      Section("セグメント") {
        Toggle("分割しない", isOn: $runner.noSplit)
        if runner.noSplit {
          Text("元動画をsegmentsへコピーせず、そのまま1本で処理します")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        Picker("分割方法", selection: $runner.useSegmentCount) {
          Text("個数").tag(true); Text("秒数").tag(false)
        }.pickerStyle(.segmented).disabled(runner.noSplit)
        if runner.useSegmentCount {
          LabeledContent("分割数") { Stepper(value: $runner.segmentCount, in: 1...128) { Text("\(runner.segmentCount)") } }
            .disabled(runner.noSplit)
        } else {
          LabeledContent("長さ（秒）") { Stepper(value: $runner.segmentDuration, in: 10...3600, step: 10) { Text("\(runner.segmentDuration)") } }
            .disabled(runner.noSplit)
        }
        if runner.usesPythonEngine {
          LabeledContent("結合エンコーダー") { TextField("", text: $runner.mergeEncoder).frame(width: 220) }
        }
        Toggle("処理済みセグメントを削除", isOn: $runner.deleteSegments)
        Toggle("一時ファイルを保持", isOn: $runner.keepTemp)
        Toggle("強制的に再分割", isOn: $runner.forceSplit)
          .disabled(runner.noSplit)
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
        HStack {
          Toggle("FPS", isOn: $runner.useFPS)
          Spacer()
          Picker("", selection: $runner.selectedFrameRate) {
            ForEach(runner.frameRateOptions, id: \.key) { option in
              Text("\(option.label)fps").tag(option.key)
            }
          }
          .labelsHidden()
          .frame(width: 130)
          .disabled(!runner.useFPS)
        }
        if let source = runner.sourceFPSDescription {
          Text("元動画 \(source)fps → \(runner.targetFPSDescription)fps\(runner.targetFPSDetail)で書き出します")
            .font(.caption)
            .foregroundStyle(.secondary)
        } else {
          Text("入力を選ぶと、その素材で選べるフレームレートだけに絞り込まれます")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        if runner.usesPythonEngine {
          Text("Pythonエンジンは整数FPSのみ対応です（\(runner.pythonTargetFPS)fpsで実行します）")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        Toggle("復元前にFPS変換", isOn: $runner.preFPSConversion)
          .disabled(!runner.useFPS || (runner.usesPythonEngine && runner.noSplit))
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
