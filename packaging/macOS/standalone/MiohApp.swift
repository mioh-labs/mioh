import AppKit
import AVFoundation
import Combine
import CryptoKit
import Foundation
import SwiftUI

private let appProgressPrefix = "@@LADA_PROGRESS@@"
private let miohMaximumClipFrames = 180
func L(_ key: String) -> String {
  NSLocalizedString(key, comment: "")
}

func restorationModelLabel(_ identifier: String) -> String {
  return L(identifier)
}

struct AppProgressEvent: Decodable {
  let kind: String
  let lane: String
  let segment: Int?
  let text: String
  let percent: Double?
}

/// Portable model identity shared by dedicated Mac, Universal Mac and iPad.
/// Runtime compiler output is deliberately absent because `.aimodelc` and
/// `.mlmodelc` are device-specific representations of the same source model.
private struct RemoteClusterCanonicalModelManifest: Decodable {
  let formatVersion: Int
  let digestAlgorithm: String
  let models: [String: RemoteClusterCanonicalModelIdentity]

  enum CodingKeys: String, CodingKey {
    case formatVersion = "format_version"
    case digestAlgorithm = "digest_algorithm"
    case models
  }
}

private struct RemoteClusterCanonicalModelIdentity: Decodable {
  let sha256: String
  let assetType: String
  let sourceAssets: [String]

  enum CodingKeys: String, CodingKey {
    case sha256
    case assetType = "asset_type"
    case sourceAssets = "source_assets"
  }
}

struct NativeExportConfiguration: Codable, Sendable {
  let mode = "export"
  let input: String
  let inputByteCount: Int64?
  let inputSHA256: String?
  var outputDirectory: String
  var ffmpegTemporaryDirectory: String
  var miohTemporaryDirectory: String
  let outputFile: String
  let ffmpeg: String
  let detectionModel: String
  let detectionBackend: String
  let detectionInputSize: Int
  let detectionCandidateChannels: Int
  let detectionQueries: Int
  let detectionLogitClasses: Int
  let detectionMaxDet: Int
  let detectionComputeUnits: String?
  let restorationModels: String
  let restorationRunner: String
  let restorationFrameCount: Int?
  let startNanoseconds: Int64
  let decodeEndNanoseconds: Int64?
  let outputCoreStartNanoseconds: Int64?
  let outputCoreEndNanoseconds: Int64?
  let workerMode: Bool
  let jobID: String?
  let attemptID: String?
  let generation: Int
  let splitMode: String
  let segmentCount: Int
  let segmentSeconds: Double
  let bufferLimitSeconds: Double = 1
  let temporalBatchFrames: Int
  let temporalOverlap: Int
  let ringCapacity: Int
  let nativeParallelWorkers: Int
  var decoderBackend: String? = nil
  var detailedDiagnostics: Bool? = nil
  var exportDetectionMasks: Bool? = nil
  var overwriteDetectionMasks: Bool? = nil
  let confidenceThreshold: Float
  let iouThreshold: Float = 0.7
  let contextFraction: Float = 0.30
  let blendFeather: Float
  /// Paste the restoration only over the detected mosaic, not the whole crop.
  var restoreMaskOnly: Bool? = nil
  /// "labelField" gives every mosaic pixel one owner (CELF); nil keeps the
  /// per-detection restoration.
  var multiMosaicMode: String? = nil
  let sharpenStrength: Float
  let detailBoost: Float
  let textureMix: Float
  let smoothStrength: Float
  let effectUpscale: Int
  let roiEnhancerModel: String?
  let roiEnhancerStrength: Float
  let roiEnhancerScale: Int
  let roiExpertMode: Bool
  /// SwiftVR and ROI Distill: Apple temporal noise filter (0 turns it off).
  var swiftVRTemporalFilter: Float? = nil
  /// SwiftVR only: frames on each side blended by the stabilization.
  var swiftVRStabilizationRadius: Int? = nil
  /// SwiftVR and ROI Distill: frames over which the crop view is averaged.
  var swiftVRFrameSmoothing: Int? = nil
  /// ROI Refine: ROI Distill layered on scenes whose crop exceeds roiRefineDistillMinimumCrop source pixels.
  var roiRefineDistillModel: String? = nil
  var roiRefineDistillMinimumCrop: Int? = nil
  /// ROI Refine: largest internal LTX canvas. Defaults to 512; 1024 is an explicit quality/cost opt-in.
  var roiRefineMaximumWindowSide: Int? = nil
  /// ROI Refine: temporal continuation frames between 33-frame LTX windows (1 fast, 9 quality).
  var roiRefineConditionFrames: Int? = nil
  /// ROI Refine: denoising steps per window, 1...8 (default 2).
  var roiRefineSteps: Int? = nil
  let detectionEmptyLookahead: Int
  let detectionMaskReuseSkipFrames: Int
  let detectFaceMosaics: Bool
  let crossfade: Bool
  let targetFPS: Int?
  let targetFPSDenominator: Int?
  let preFPSConversion: Bool
  let videoCodec: String
  let averageBitRate: Int?
  let bitrateMultiplier: Double
  let mp4FastStart: Bool
  /// Reuse the contiguous completed export segments in `outputDirectory` and continue after them.
  var resumeExistingSegments: Bool? = nil

  enum CodingKeys: String, CodingKey {
    case mode
    case input
    case inputByteCount
    case inputSHA256
    case outputDirectory
    case ffmpegTemporaryDirectory
    case miohTemporaryDirectory
    case outputFile
    case ffmpeg
    case detectionModel
    case detectionBackend
    case detectionInputSize
    case detectionCandidateChannels
    case detectionQueries
    case detectionLogitClasses
    case detectionMaxDet
    case detectionComputeUnits
    case restorationModels
    case restorationRunner
    case restorationFrameCount
    case startNanoseconds
    case decodeEndNanoseconds
    case outputCoreStartNanoseconds
    case outputCoreEndNanoseconds
    case workerMode
    case jobID
    case attemptID
    case generation
    case splitMode
    case segmentCount
    case segmentSeconds
    case bufferLimitSeconds
    case temporalBatchFrames
    case temporalOverlap
    case ringCapacity
    case nativeParallelWorkers
    case decoderBackend
    case detailedDiagnostics
    case exportDetectionMasks
    case overwriteDetectionMasks
    case confidenceThreshold
    case iouThreshold
    case contextFraction
    case blendFeather
    case restoreMaskOnly
    case multiMosaicMode
    case sharpenStrength
    case detailBoost
    case textureMix
    case smoothStrength
    case effectUpscale
    case roiEnhancerModel
    case roiEnhancerStrength
    case roiEnhancerScale
    case roiExpertMode
    case swiftVRTemporalFilter
    case swiftVRStabilizationRadius
    case swiftVRFrameSmoothing
    case roiRefineDistillModel
    case roiRefineDistillMinimumCrop
    case roiRefineMaximumWindowSide
    case roiRefineConditionFrames
    case roiRefineSteps
    case detectionEmptyLookahead
    case detectionMaskReuseSkipFrames
    case detectFaceMosaics
    case crossfade
    case targetFPS
    case targetFPSDenominator
    case preFPSConversion
    case videoCodec
    case averageBitRate
    case bitrateMultiplier
    case mp4FastStart
    case resumeExistingSegments
  }
}

private struct NativePreviewLaunchConfiguration: Encodable {
  var decoderBackend: String? = nil
  var detailedDiagnostics: Bool? = nil
  let input: String
  let outputDirectory: String
  let ffmpegTemporaryDirectory: String?
  let miohTemporaryDirectory: String?
  let ffmpeg: String
  let detectionModel: String
  let detectionBackend: String
  let detectionInputSize: Int
  let detectionCandidateChannels: Int
  let detectionQueries: Int
  let detectionLogitClasses: Int
  let detectionMaxDet: Int
  let detectionComputeUnits: String?
  let restorationModels: String
  let restorationRunner: String
  let restorationFrameCount: Int?
  let startNanoseconds: Int64
  let decodeEndNanoseconds: Int64?
  let outputCoreStartNanoseconds: Int64?
  let outputCoreEndNanoseconds: Int64?
  let generation: Int
  let segmentSeconds: Double = 2
  let bufferLimitSeconds: Double
  let temporalBatchFrames: Int
  let temporalOverlap: Int
  let ringCapacity: Int
  let confidenceThreshold: Float
  let iouThreshold: Float = 0.7
  let contextFraction: Float = 0.30
  let blendFeather: Float
  /// Paste the restoration only over the detected mosaic, not the whole crop.
  var restoreMaskOnly: Bool? = nil
  /// "labelField" gives every mosaic pixel one owner (CELF); nil keeps the
  /// per-detection restoration.
  var multiMosaicMode: String? = nil
  let sharpenStrength: Float
  let detailBoost: Float
  let textureMix: Float
  let smoothStrength: Float
  let effectUpscale: Int
  let roiEnhancerModel: String?
  let roiEnhancerStrength: Float
  let roiEnhancerScale: Int
  let roiExpertMode: Bool
  let detectionEmptyLookahead: Int
  let detectionMaskReuseSkipFrames: Int
  let detectFaceMosaics: Bool
  let crossfade: Bool
  let maximumFPS: Int?
  let preFPSConversion: Bool
  let videoCodec: String = "h264"
}

struct NativePreviewInvocation {
  let executable: URL
  let configuration: Data
  let environment: [String: String]
}

private final class RemoteClusterOutputCapture: @unchecked Sendable {
  private let lock = NSLock()
  private var data = Data()
  private let maximumBytes = 4 * 1_024 * 1_024

  func append(_ chunk: Data) {
    guard !chunk.isEmpty else { return }
    lock.withLock {
      data.append(chunk)
      if data.count > maximumBytes {
        data.removeFirst(data.count - maximumBytes)
      }
    }
  }

  func snapshot() -> Data { lock.withLock { data } }
}

private final class RemoteClusterProcessResult: @unchecked Sendable {
  private let lock = NSLock()
  private var status: Int32?
  private var continuation: CheckedContinuation<Int32, Never>?

  func finish(_ status: Int32) {
    let pending: CheckedContinuation<Int32, Never>? = lock.withLock {
      guard self.status == nil else { return nil }
      self.status = status
      let pending = continuation
      continuation = nil
      return pending
    }
    pending?.resume(returning: status)
  }

  func value() async -> Int32 {
    await withCheckedContinuation { continuation in
      let completed: Int32? = lock.withLock {
        if let status { return status }
        self.continuation = continuation
        return nil
      }
      if let completed {
        continuation.resume(returning: completed)
      }
    }
  }
}

private final class RemoteClusterProcessControl: @unchecked Sendable {
  private let lock = NSLock()
  private let process: Process
  private let input: Pipe
  private var launched = false
  private var cancellationRequested = false
  private var stopSent = false

  init(process: Process, input: Pipe) {
    self.process = process
    self.input = input
  }

  func didLaunch() {
    let cancelNow = lock.withLock {
      launched = true
      guard cancellationRequested, !stopSent else { return false }
      stopSent = true
      return true
    }
    if cancelNow { stopProcess() }
  }

  func cancel() {
    let stopNow = lock.withLock {
      cancellationRequested = true
      guard launched, !stopSent else { return false }
      stopSent = true
      return true
    }
    if stopNow { stopProcess() }
  }

  private func stopProcess() {
    let handle = input.fileHandleForWriting
    if process.isRunning {
      _ = MacChildProcessPipe.write(
        Data("{\"command\":\"stop\"}\n".utf8),
        to: handle
      )
      process.interrupt()
      DispatchQueue.global(qos: .utility).asyncAfter(
        deadline: .now() + 2
      ) { [process] in
        if process.isRunning { process.terminate() }
      }
    }
    try? handle.close()
  }
}

struct PlatformCapabilities {
  let supportsCoreAI: Bool

  init(
    operatingSystemVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion
  ) {
    supportsCoreAI = operatingSystemVersion.majorVersion >= 27
  }

  var defaultRestorationModel: String {
    "basicvsrpp-v1.2-coreai-t90"
  }

  var previewRestorationModel: String {
    "basicvsrpp-v1.2-coreai-variable"
  }

  var previewDetectionModel: String {
    supportsCoreAI ? "v4-fast-coreai" : "v4-fast-coreml"
  }

  var coreAIRestorationModels: [String] {
    [
      "basicvsrpp-v1.2-coreai-t90", "basicvsrpp-v1.2-coreai-t36",
      "basicvsrpp-v1.2-coreai", "basicvsrpp-v1.2-coreai-variable",
    ]
  }

  var restorationModels: [String] {
    coreAIRestorationModels
  }

  let baseDetectionModels = [
    "v2-coreml", "v3.1-fast-coreml", "v3.1-accurate-coreml",
    "v4-fast-coreml", "v4-accurate-coreml", "vr-v2-accurate-coreml",
    "celfnet-o2-coreml",
  ]

  var detectionModels: [String] {
    guard supportsCoreAI else { return baseDetectionModels }
    var models = baseDetectionModels + [
        "v2-coreai", "v3.1-fast-coreai", "v3.1-accurate-coreai",
        "v4-fast-coreai", "v4-accurate-coreai", "vr-v2-accurate-coreai",
      ]
#if !MIOH_PORTABLE_COREAI
    models.append("jasna-v6-coreml")
    models.append("jasna-v6-large-coreml")
    models.append("jasna-v6-coreai")
    models.append("jasna-v6-large-coreai")
#endif
    return models
  }

  var previewDetectionModels: [String] {
    detectionModels.filter {
      !$0.hasPrefix("jasna-v6") || $0.hasSuffix("-coreml")
    }
  }

}

/// Installation/runtime status exposed to the authenticated Web remote.
/// Keep the reason deliberately generic: it is user-facing and must not leak
/// bundle paths or other host filesystem details.
struct MiohModelAvailability {
  let available: Bool
  let reason: String?

  static let available = MiohModelAvailability(available: true, reason: nil)
  static let notInstalled = MiohModelAvailability(
    available: false,
    reason: "モデル未導入"
  )
  static let nativeUnsupported = MiohModelAvailability(
    available: false,
    reason: "Swiftネイティブ非対応"
  )
}

/// What the input file actually is, read once when it is selected. This is
/// reporting only — nothing in the pipeline branches on it, so a container
/// that misreports a field cannot change how the export runs.
struct SourceMediaInfo: Equatable, Sendable {
  var width: Int
  var height: Int
  var frameRate: Double
  var durationSeconds: Double
  var videoCodec: String
  var dataRateBitsPerSecond: Double
  var audio: String?
  var fileSizeBytes: Int64?

  var resolutionText: String { "\(width)×\(height)" }

  /// Matches the frame rate picker: 29.970 stays 29.970, 30 stays 30.
  var frameRateText: String {
    FrameRateOption(
      numerator: max(1, Int((frameRate * 1000).rounded())),
      denominator: 1000
    ).label
  }

  var durationText: String {
    guard durationSeconds.isFinite, durationSeconds > 0 else { return "—" }
    let total = Int(durationSeconds.rounded())
    let hours = total / 3600
    let minutes = (total % 3600) / 60
    let seconds = total % 60
    return hours > 0
      ? String(format: "%d:%02d:%02d", hours, minutes, seconds)
      : String(format: "%d:%02d", minutes, seconds)
  }

  var bitRateText: String? {
    guard dataRateBitsPerSecond > 0 else { return nil }
    let mbps = dataRateBitsPerSecond / 1_000_000
    return mbps >= 10
      ? String(format: "%.0f Mbps", mbps)
      : String(format: "%.1f Mbps", mbps)
  }

  var fileSizeText: String? {
    fileSizeBytes.map {
      ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)
    }
  }

  /// The one-line summary shown under the input path.
  var summary: String {
    var parts = [
      resolutionText,
      "\(frameRateText)fps",
      videoCodec,
      durationText,
    ]
    if let bitRateText { parts.append(bitRateText) }
    if let audio { parts.append(audio) }
    if let fileSizeText { parts.append(fileSizeText) }
    return parts.joined(separator: " · ")
  }
}

enum SourceMediaProbe {
  enum Outcome: Sendable {
    case success(SourceMediaInfo)
    case failure(String)
  }

  static func read(url: URL) async -> Outcome {
    let asset = AVURLAsset(url: url)
    do {
      guard let track = try await asset.loadTracks(withMediaType: .video).first
      else {
        return .failure("映像トラックが見つかりません")
      }
      let size = try await track.load(.naturalSize)
      let transform = try await track.load(.preferredTransform)
      let oriented = size.applying(transform)
      let frameRate = try await track.load(.nominalFrameRate)
      let dataRate = try await track.load(.estimatedDataRate)
      let duration = try await asset.load(.duration)
      let formats = try await track.load(.formatDescriptions)
      let fileSize = (
        try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize
      ).flatMap { $0 }

      return .success(
        SourceMediaInfo(
          width: max(1, Int(abs(oriented.width).rounded())),
          height: max(1, Int(abs(oriented.height).rounded())),
          frameRate: max(0, Double(frameRate)),
          durationSeconds: duration.seconds,
          videoCodec: formats.first.map { name(for: $0) } ?? "不明",
          dataRateBitsPerSecond: max(0, Double(dataRate)),
          audio: await audioSummary(of: asset),
          fileSizeBytes: fileSize.map(Int64.init)
        )
      )
    } catch {
      return .failure(error.localizedDescription)
    }
  }

  private static func audioSummary(of asset: AVAsset) async -> String? {
    guard
      let track = try? await asset.loadTracks(withMediaType: .audio).first,
      let formats = try? await track.load(.formatDescriptions),
      let format = formats.first
    else {
      return nil
    }
    var parts = [name(for: format)]
    if let basic = CMAudioFormatDescriptionGetStreamBasicDescription(format)?
      .pointee
    {
      if basic.mChannelsPerFrame > 0 {
        parts.append("\(basic.mChannelsPerFrame)ch")
      }
      if basic.mSampleRate > 0 {
        parts.append(String(format: "%.1fkHz", basic.mSampleRate / 1000))
      }
    }
    return parts.joined(separator: " ")
  }

  private static func name(for format: CMFormatDescription) -> String {
    let subtype = CMFormatDescriptionGetMediaSubType(format)
    switch subtype {
    case kCMVideoCodecType_H264: return "H.264"
    case kCMVideoCodecType_HEVC: return "HEVC"
    case kCMVideoCodecType_HEVCWithAlpha: return "HEVC+α"
    case kCMVideoCodecType_AV1: return "AV1"
    case kCMVideoCodecType_VP9: return "VP9"
    case kCMVideoCodecType_MPEG4Video: return "MPEG-4"
    case kCMVideoCodecType_MPEG2Video: return "MPEG-2"
    case kCMVideoCodecType_AppleProRes422: return "ProRes 422"
    case kCMVideoCodecType_AppleProRes422HQ: return "ProRes 422 HQ"
    case kCMVideoCodecType_AppleProRes4444: return "ProRes 4444"
    case kAudioFormatMPEG4AAC: return "AAC"
    case kAudioFormatLinearPCM: return "PCM"
    case kAudioFormatOpus: return "Opus"
    case kAudioFormatFLAC: return "FLAC"
    case kAudioFormatAC3: return "AC-3"
    default:
      return fourCharCode(subtype)
    }
  }

  private static func fourCharCode(_ value: FourCharCode) -> String {
    let bytes = [
      UInt8((value >> 24) & 0xFF),
      UInt8((value >> 16) & 0xFF),
      UInt8((value >> 8) & 0xFF),
      UInt8(value & 0xFF),
    ]
    let text = String(bytes: bytes, encoding: .ascii) ?? ""
    let trimmed = text.trimmingCharacters(in: .whitespaces)
    return trimmed.isEmpty ? "不明" : trimmed
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

enum PreviewHLSQuality: String, CaseIterable, Identifiable {
  case automatic = "auto"
  case p1080 = "1080p"
  case p720 = "720p"
  case p480 = "480p"

  var id: String { rawValue }

  var label: String {
    switch self {
    case .automatic: return "自動"
    case .p1080: return "1080p固定"
    case .p720: return "720p固定"
    case .p480: return "480p固定"
    }
  }

  var targetHeight: Int? {
    switch self {
    case .automatic: return nil
    case .p1080: return 1_080
    case .p720: return 720
    case .p480: return 480
    }
  }
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
  var nativeParallelWorkers: Int?
  var decoderBackend: String? = nil
  var detailedDiagnostics: Bool? = nil
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
  var restoreMaskOnly: Bool?
  var multiMosaicMode: String?
  var textureMix: Double
  var smoothStrength: Double
  var effectUpscale: Int
  var roiEnhancer: String
  var roiEnhancerModel: String
  /// The model each ROI enhancer method used last (built-in name or local path).
  var roiEnhancerModelsByMethod: [String: String]?
  var roiEnhancerScale: Int
  var roiEnhancerStrength: Double
  var roiEnhancerTile: Int
  var roiExpertMode: Bool?
  var swiftVRTemporalFilter: Double?
  var swiftVRStabilizationRadius: Int?
  var swiftVRFrameSmoothing: Int?
  var roiRefineDistill: Bool?
  /// Explicit ROI Refine canvas used when the fast preset is off.
  var roiRefineWindowSide: Int?
  /// Explicit LTX continuation length used when the fast preset is off.
  var roiRefineContinuationFrames: Int?
  /// ROI Refine denoising steps (1...8).
  var roiRefineSteps: Int?
  /// Legacy 1024 toggle, retained so older saved settings still decode.
  var roiRefineUse1024Window: Bool?
  var roiRefineFastMode: Bool?

  var detectionModel: String
  var customDetectionModel: String
  var detectionEmptyLookahead: Int
  var detectionMaskReuseSkipFrames: Int?
  var exportDetectionMasks: Bool?
  var detectFaceMosaics: Bool

  var previewBufferLimit: Double
  var previewRestorationModel: String?
  var previewCustomRestorationModel: String?
  var previewDetectionModel: String?
  var previewCustomDetectionModel: String?
  var previewRealtimeOptimization: Bool?
  var previewLimitHighFrameRate: Bool?
  var previewUseSafariCompatibleHLS: Bool?
  var previewHLSQuality: String?
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
      nativeParallelWorkers: 1,
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
      maxClipLength: 180,
      useRestoreMaxFrames: false,
      restoreMaxFrames: -1,
      restoreTemporalOverlap: 8,
      restoreCrossfade: true,
      sharpenStrength: 0.0,
      detailBoost: 0.0,
      blendFeather: 1.0,
      restoreMaskOnly: true,
      multiMosaicMode: "perDetection",
      textureMix: 0.0,
      smoothStrength: 0.0,
      effectUpscale: 1,
      roiEnhancer: "none",
      roiEnhancerModel: "",
      roiEnhancerModelsByMethod: nil,
      roiEnhancerScale: 4,
      roiEnhancerStrength: 0.0,
      roiEnhancerTile: 0,
      roiExpertMode: false,
      swiftVRTemporalFilter: 1.0,
      swiftVRStabilizationRadius: 1,
      swiftVRFrameSmoothing: 15,
      roiRefineDistill: true,
      roiRefineWindowSide: 512,
      roiRefineContinuationFrames: 9,
      roiRefineSteps: 2,
      roiRefineUse1024Window: false,
      roiRefineFastMode: false,
      detectionModel: "v2-coreml",
      customDetectionModel: "",
      detectionEmptyLookahead: 10,
      detectionMaskReuseSkipFrames: 0,
      exportDetectionMasks: false,
      detectFaceMosaics: false,
      previewBufferLimit: 8.0,
      previewRestorationModel: capabilities.previewRestorationModel,
      previewCustomRestorationModel: "",
      previewDetectionModel: capabilities.previewDetectionModel,
      previewCustomDetectionModel: "",
      previewRealtimeOptimization: true,
      previewLimitHighFrameRate: false,
      previewUseSafariCompatibleHLS: false,
      previewHLSQuality: PreviewHLSQuality.automatic.rawValue,
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
      outputFileName = ""
      refreshOutputPathText()
      refreshSourceInfo(for: inputURL)
    }
  }
  @Published var sourceInfo: SourceMediaInfo?
  @Published var sourceInfoFailure: String?
  @Published var sourceBatchSummary: String?
  @Published var rangeStartSeconds = 0.0
  @Published var rangeEndSeconds = 0.0
  @Published var outputURL: URL?
  /// Output file name typed in the 基本 tab; "" = the default "<input name>-UC.<ext>".
  @Published var outputFileName = ""

  /// The 出力 row: the full output path (folder + file name) as typed. For a single input file the last
  /// path component is the file name and the rest the folder, unless the path is a folder: then the
  /// default file name is used. A folder input (batch) takes a folder.
  /// Rewritten only when the input or the chosen folder changes, so typing is never overridden.
  @Published var outputPathText = "" {
    didSet {
      guard outputPathText != oldValue else { return }
      let path = (outputPathText.trimmingCharacters(in: .whitespacesAndNewlines) as NSString).expandingTildeInPath
      guard !path.isEmpty else {
        outputURL = nil
        outputFileName = ""
        return
      }
      // A folder (an existing one, or a path ending in "/") saves under the default file name.
      var isFolder: ObjCBool = false
      if inputURL == nil || inputIsDirectory || path.hasSuffix("/")
        || (FileManager.default.fileExists(atPath: path, isDirectory: &isFolder) && isFolder.boolValue) {
        outputURL = URL(fileURLWithPath: path, isDirectory: true)
        outputFileName = ""
        return
      }
      let url = URL(fileURLWithPath: path)
      outputURL = url.deletingLastPathComponent()
      let name = url.lastPathComponent
      outputFileName = inputURL.map { name == MacNativeExportBatchPlanner.defaultFileName(input: $0) } == true ? "" : name
    }
  }

  private var inputIsDirectory: Bool {
    guard let inputURL else { return false }
    var isDirectory: ObjCBool = false
    return FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDirectory) && isDirectory.boolValue
  }

  var sourceDurationSeconds: Double {
    guard let value = sourceInfo?.durationSeconds,
      value.isFinite, value > 0, !inputIsDirectory
    else { return 0 }
    return value
  }

  var normalizedRangeStartSeconds: Double {
    min(max(0, rangeStartSeconds), max(0, sourceDurationSeconds - 0.05))
  }

  var normalizedRangeEndSeconds: Double {
    min(max(normalizedRangeStartSeconds + 0.05, rangeEndSeconds), sourceDurationSeconds)
  }

  var selectedRangeDurationSeconds: Double {
    max(0, normalizedRangeEndSeconds - normalizedRangeStartSeconds)
  }

  var usesPartialVideoRange: Bool {
    sourceDurationSeconds > 0
      && (normalizedRangeStartSeconds > 0.000_5
        || normalizedRangeEndSeconds < sourceDurationSeconds - 0.000_5)
  }

  func selectFullVideoRange() {
    rangeStartSeconds = 0
    rangeEndSeconds = sourceDurationSeconds
  }

  func setRangeStartSeconds(_ value: Double) {
    guard value.isFinite else { return }
    rangeStartSeconds = min(max(0, value), max(0, normalizedRangeEndSeconds - 0.05))
  }

  func setSelectedRangeDurationSeconds(_ value: Double) {
    guard value.isFinite else { return }
    let available = sourceDurationSeconds - normalizedRangeStartSeconds
    guard available > 0 else { return }
    rangeEndSeconds = normalizedRangeStartSeconds
      + min(max(min(0.05, available), value), available)
  }

  func setRangeEndSeconds(_ value: Double) {
    guard value.isFinite else { return }
    rangeEndSeconds = min(
      max(normalizedRangeStartSeconds + 0.05, value),
      sourceDurationSeconds
    )
  }

  /// The output file that already exists for a single-file input (nil when none, or for folder batches).
  var existingOutputFile: URL? {
    guard let inputURL, let outputURL, !inputIsDirectory else { return nil }
    let output = resolvedOutputFile(input: inputURL, selectedOutput: outputURL)
    var isDirectory: ObjCBool = false
    return FileManager.default.fileExists(atPath: output.path, isDirectory: &isDirectory) && !isDirectory.boolValue
      ? output : nil
  }

  /// Start after the user confirmed replacing this existing output file (this run only).
  func startReplacing(_ output: URL) {
    confirmedReplaceOutput = output.standardizedFileURL
    start()
  }
  private var confirmedReplaceOutput: URL?

  func refreshOutputPathText() {
    guard let outputURL else {
      outputPathText = ""
      return
    }
    guard let inputURL, !inputIsDirectory else {
      outputPathText = outputURL.path
      return
    }
    outputPathText = MacNativeExportBatchPlanner.resolvedOutputFile(
      input: inputURL, selectedOutput: outputURL, fileName: outputFileName
    ).path
  }
  @Published var progress = 0.0
  @Published var status = "待機中"
  @Published var log = ""
  @Published var isRunning = false {
    didSet { updateExportActivity() }
  }
  /// While work runs, tell macOS it is user-initiated. Without this, App Nap
  /// throttled mioh and the export processes it launches once the window left
  /// the foreground: the same SwiftVR export took about 2.3x as long from the
  /// app as from a shell (209-232 s vs 8-9 min).
  private var exportActivity: NSObjectProtocol?

  private func updateExportActivity() {
    if isRunning, exportActivity == nil {
      exportActivity = ProcessInfo.processInfo.beginActivity(
        options: [.userInitiated, .idleSystemSleepDisabled],
        reason: "mioh restoration or export")
    } else if !isRunning, let activity = exportActivity {
      ProcessInfo.processInfo.endActivity(activity)
      exportActivity = nil
    }
  }
  @Published var defaultsStatus = "未保存"

  @Published var tempDirectory = "/tmp"
  @Published var ffmpegTempDirectory = ""
  @Published var ladaTempDirectory = ""
  @Published var overwrite = false

  // Kept in the persisted/remote schema for backward compatibility. Runtime
  // execution is always the Swift Core AI pipeline.
  @Published var restorationEngine = "native"

  @Published var parallelWorkers = 1
  @Published var nativeParallelWorkers = 1
  @Published var decoderBackend = "avfoundationAsync"
  @Published var detailedDiagnostics = false
  @Published var executor = "process"
  @Published var useSegmentCount = true
  @Published var segmentCount = 4
  @Published var segmentDuration = 60
  @Published var noSplit = false
  @Published var mergeEncoder = "copy"
  @Published var deleteSegments = false
  @Published var keepTemp = true
  @Published var resumeInterruptedExport = true
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
  @Published var maxClipLength = 180
  @Published var useRestoreMaxFrames = false
  @Published var restoreMaxFrames = -1
  @Published var restoreTemporalOverlap = 8
  @Published var restoreCrossfade = true
  @Published var sharpenStrength = 0.0
  @Published var detailBoost = 0.0
  @Published var blendFeather = 1.0
  /// Paste the restoration only over the detected mosaic (not faces or other
  /// unmosaicked parts of the crop).
  @Published var restoreMaskOnly = true
  /// How overlapping mosaics are restored: "perDetection" (each detection,
  /// the established behaviour) or "labelField" (CELF: one owner per mosaic
  /// pixel, see MosaicLabelFieldTracker.swift).
  @Published var multiMosaicMode = "perDetection"
  @Published var textureMix = 0.0
  @Published var smoothStrength = 0.0
  @Published var effectUpscale = 1
  @Published var roiEnhancer = "none"
  @Published var roiEnhancerModel = ""
  /// Last model (built-in name or local file/folder) per ROI enhancer method, so
  /// switching the method brings back where that method's model was.
  @Published var roiEnhancerModelsByMethod: [String: String] = [:]
  @Published var roiEnhancerScale = 4
  @Published var roiEnhancerStrength = 0.0
  @Published var roiEnhancerTile = 0
  @Published var roiExpertMode = false
  /// Strength of the temporal noise filter on SwiftVR's frames (0 = off).
  @Published var swiftVRTemporalFilter = 1.0
  /// Frames on each side that SwiftVR's stabilization blends (1 = ±1).
  @Published var swiftVRStabilizationRadius = 1
  /// Frames over which SwiftVR's view of the crop is averaged (0 = off).
  @Published var swiftVRFrameSmoothing = 15
  /// ROI Refine: layer ROI Distill on crops over 512 px (refine draws at 512).
  @Published var roiRefineDistill = true
  /// ROI Refine canvas used by the custom settings (384, 512 or 1024; a saved 256 reads as 384).
  @Published var roiRefineWindowSide = 512
  /// Clean frames carried into the next 33-frame LTX window (1 or 9).
  @Published var roiRefineContinuationFrames = 9
  /// ROI Refine denoising steps per 33-frame window (1...8; 2 is the measured default).
  @Published var roiRefineSteps = 2
  /// Cap ROI Refine at 384px and keep one continuation frame between LTX windows.
  @Published var roiRefineFastMode = false

  @Published var detectionModel: String
  @Published var customDetectionModel = ""
  @Published var detectionEmptyLookahead = 10
  @Published var detectionMaskReuseSkipFrames = 0
  @Published var exportDetectionMasks = false
  @Published var detectFaceMosaics = false

  @Published var previewBufferLimit = 8.0
  @Published var previewRestorationModel: String
  @Published var previewCustomRestorationModel = ""
  @Published var previewDetectionModel: String
  @Published var previewCustomDetectionModel = ""
  @Published var previewRealtimeOptimization = true
  @Published var previewLimitHighFrameRate = false
  @Published var previewUseSafariCompatibleHLS = false
  @Published var previewHLSQuality = PreviewHLSQuality.automatic.rawValue
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
  private var nativeExportBatchPending: [MacNativeExportBatchItem] = []
  private var nativeExportBatchCurrent: MacNativeExportBatchItem?
  private var nativeExportBatchResources: URL?
  private var nativeExportBatchTotal = 0
  private var nativeExportBatchCompleted = 0
  private var nativeExportBatchActive = false
  private var nativeExportBatchStopRequested = false
  private var lineBuffer = ""
  private var logHistory = ""
  private var activeProgress: [String: AppProgressEvent] = [:]
  private var activeProgressOrder: [String] = []
  private var remoteClusterAssetDigests: [String: String] = [:]
  private let capabilities: PlatformCapabilities
  private let defaultsKey = "mioh.userProcessingDefaults.v1"
  /// The settings as last used, remembered automatically and restored at
  /// launch. The saved defaults above stay a separate, explicit preset.
  private let lastSettingsKey = "mioh.lastProcessingSettings.v1"
  private var lastSettingsData: Data?
  private var settingsAutosave: AnyCancellable?
  private var terminationObserver: NSObjectProtocol?

  init(capabilities: PlatformCapabilities = PlatformCapabilities()) {
    self.capabilities = capabilities
    restorationModel = capabilities.defaultRestorationModel
    previewRestorationModel = capabilities.previewRestorationModel
    previewDetectionModel = capabilities.previewDetectionModel
    detectionModel = "v2-coreml"
    loadSavedDefaultsOnLaunch()
    startRememberingSettings()
  }

  /// Records the settings whenever they change (at most every 2 seconds,
  /// only when they differ from the last record) and once more at quit.
  private func startRememberingSettings() {
    lastSettingsData = UserDefaults.standard.data(forKey: lastSettingsKey)
    settingsAutosave = objectWillChange
      .throttle(for: .seconds(2), scheduler: RunLoop.main, latest: true)
      .sink { [weak self] _ in self?.rememberSettings() }
    terminationObserver = NotificationCenter.default.addObserver(
      forName: NSApplication.willTerminateNotification, object: nil, queue: .main
    ) { [weak self] _ in
      MainActor.assumeIsolated { self?.rememberSettings() }
    }
  }

  private func rememberSettings() {
    guard let data = try? JSONEncoder().encode(currentDefaultsSnapshot()),
      data != lastSettingsData
    else { return }
    UserDefaults.standard.set(data, forKey: lastSettingsKey)
    lastSettingsData = data
  }

  let encodingPresets = [
    "h264-cpu-uhq", "h264-cpu-fast", "h264-apple-gpu-balanced",
    "hevc-apple-gpu-balanced", "av1-cpu-uhq",
  ]
  var restorationModels: [String] { capabilities.restorationModels }
  var nativePreviewRestorationModels: [String] {
    capabilities.supportsCoreAI ? capabilities.coreAIRestorationModels : []
  }
  var previewRestorationModels: [String] {
    nativePreviewRestorationModels
  }
  var detectionModels: [String] { capabilities.detectionModels }
  var previewDetectionModels: [String] { capabilities.previewDetectionModels }
  var preservesRealtimeCompositeParameters: Bool {
#if MIOH_PORTABLE_COREAI
    return false
#else
    return true
#endif
  }

  /// Resolve availability through the same asset lookup used at execution
  /// time.  The Web remote can therefore show every supported identifier but
  /// prevents selecting a model that is absent from this installation.
  func restorationModelAvailability(
    _ model: String,
    engine _: String? = nil
  ) -> MiohModelAvailability {
    if !capabilities.supportsCoreAI {
      return .nativeUnsupported
    }
    guard let resources = try? resourceDirectory() else { return .notInstalled }
    guard model.contains("coreai") else { return .nativeUnsupported }
    guard let asset = nativeRestorationAsset(resources: resources, model: model)
    else { return .notInstalled }
    let executable = resources.appendingPathComponent("bin/\(asset.runnerName)")
    return FileManager.default.isExecutableFile(atPath: executable.path)
      ? .available : .notInstalled
  }

  func detectionModelAvailability(
    _ model: String,
    engine _: String? = nil
  ) -> MiohModelAvailability {
    if !capabilities.supportsCoreAI {
      return .nativeUnsupported
    }
    guard let resources = try? resourceDirectory() else { return .notInstalled }
    return nativeDetectionAsset(resources: resources, model: model) == nil
      ? .notInstalled : .available
  }

  func roiEnhancerModelAvailability(
    _ model: String,
    engine _: String? = nil
  ) -> MiohModelAvailability {
    if model.isEmpty { return .available }
    guard let resources = try? resourceDirectory() else { return .notInstalled }
    if nativeROIEnhancerAsset(
      resources: resources,
      model: model,
      requestedScale: roiEnhancerScale
    ) != nil {
      return .available
    }
    return .nativeUnsupported
  }

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
    get { targetFrameRate.key }
    set {
      guard let option = frameRateOptions.first(where: { $0.key == newValue })
      else { return }
      fps = option.numerator
      fpsDenominator = option.denominator
    }
  }

  /// The full table, always. Narrowing it to what a probed source can reach
  /// only hid the rate the user wanted whenever the probe was wrong, and the
  /// pipeline already rejects an up-conversion with a clear message.
  var frameRateOptions: [FrameRateOption] {
    var options = Self.standardFrameRates
    // Never drop the current selection out from under the picker.
    if !options.contains(targetFrameRate) {
      options.append(targetFrameRate)
    }
    return options.sorted { $0.value < $1.value }
  }

  private func refreshSourceInfo(for url: URL?) {
    sourceInfo = nil
    sourceInfoFailure = nil
    sourceBatchSummary = nil
    rangeStartSeconds = 0
    rangeEndSeconds = 0
    guard let url else { return }
    if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
      let candidates = (try? FileManager.default.contentsOfDirectory(
        at: url,
        includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
        options: [.skipsHiddenFiles]
      )) ?? []
      let count = candidates.filter { candidate in
        guard MacNativeExportBatchPlanner.videoExtensions.contains(
          candidate.pathExtension.lowercased()
        ), let values = try? candidate.resourceValues(forKeys: [
          .isRegularFileKey, .isSymbolicLinkKey,
        ]) else { return false }
        return values.isRegularFile == true && values.isSymbolicLink != true
      }.count
      sourceBatchSummary = "バッチ入力: 直下の対応動画 \(count)本"
      return
    }
    Task { [weak self] in
      let probed = await SourceMediaProbe.read(url: url)
      await MainActor.run { [weak self] in
        // A slow probe must not overwrite a newer selection.
        guard let self, self.inputURL == url else { return }
        switch probed {
        case .success(let info):
          self.sourceInfo = info
          self.rangeStartSeconds = 0
          self.rangeEndSeconds = max(0, info.durationSeconds)
        case .failure(let message):
          self.sourceInfoFailure = message
        }
      }
    }
  }
  let enhancerModels = ["none", "swiftvr", "roi-distill", "roi-refine"]
  static func enhancerDisplayName(_ method: String) -> String {
    ["none": "なし", "swiftvr": "SwiftVR", "roi-distill": "ROI Distill", "roi-refine": "ROI Refine"][method] ?? method
  }
  private let knownROIEnhancerModelNames: Set<String> = [
    "roi-distill-x2-coreml",
  ]

  var roiEnhancerModelOptions: [ROIEnhancerModelOption] {
    roiEnhancerModelOptions(for: roiEnhancer, includingSelectedModel: true)
  }

  var roiEnhancerSupportsFrameSmoothing: Bool {
    roiEnhancer == "swiftvr" || roiEnhancer == "roi-distill" || roiEnhancer == "roi-refine"
  }

  /// The same model catalogue is used by SwiftUI and the authenticated Web
  /// remote. Keeping it here prevents the browser UI from drifting from the
  /// actual build/OS capabilities.
  func roiEnhancerModelOptions(
    for enhancer: String,
    includingSelectedModel: Bool = false
  ) -> [ROIEnhancerModelOption] {
    var options: [ROIEnhancerModelOption]
    switch enhancer {
    case "roi-distill":
      options = [
        ROIEnhancerModelOption(
          name: "roi-distill-x2-coreml",
          label: "ROI Distill 12ブロック（step 19000）— Core ML / ANE（2x）",
          scale: 2
        ),
      ]
    case "swiftvr", "roi-refine":
      options = [] // The large local model pack is selected as a directory.
    default:
      options = []
    }
    if !capabilities.supportsCoreAI {
      options.removeAll { $0.name.contains("coreai") }
    }
    let selected = includingSelectedModel
      ? roiEnhancerModel.trimmingCharacters(in: .whitespacesAndNewlines)
      : ""
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

  func chooseOutput() {
    let panel = NSOpenPanel()
    panel.title = "出力フォルダを選択"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    panel.directoryURL = outputURL
    if panel.runModal() == .OK {
      outputURL = panel.url
      outputFileName = ""       // a chosen folder saves under the default file name
      refreshOutputPathText()
    }
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
    if let remembered = roiEnhancerModelsByMethod[roiEnhancer], remembered.hasPrefix("/") {
      // Open the chooser where this method's model was last time.
      panel.directoryURL = URL(fileURLWithPath: remembered).deletingLastPathComponent()
    }
    guard panel.runModal() == .OK, let path = panel.url?.path else { return }
    roiEnhancerModel = path
    applyROIEnhancerScale(forCustomPath: path)
    rememberROIEnhancerModel()
  }

  private func applyROIEnhancerScale(forCustomPath path: String) {
    let normalized = URL(fileURLWithPath: path).lastPathComponent.lowercased()
    if roiEnhancer == "roi-distill" {
      roiEnhancerScale = 2
      return
    }
    if roiEnhancer == "swiftvr" {
      roiEnhancerScale = roiEnhancerScale == 2 ? 2 : 4
      return
    }
    if roiEnhancer == "roi-refine" {
      roiEnhancerScale = 2
      return
    }
    if normalized.contains("x2") || normalized.hasPrefix("2x") {
      roiEnhancerScale = 2
    } else if normalized.contains("x4") || normalized.hasPrefix("4x") {
      roiEnhancerScale = 4
    }
  }

  private func rememberROIEnhancerModel() {
    let model = roiEnhancerModel.trimmingCharacters(in: .whitespacesAndNewlines)
    guard roiEnhancer != "none", !model.isEmpty else { return }
    roiEnhancerModelsByMethod[roiEnhancer] = model
  }

  /// The model this method used last, if it is still available: a built-in
  /// option of this method, or a local file/folder that still exists.
  private func rememberedROIEnhancerModel(for enhancer: String) -> String? {
    guard let model = roiEnhancerModelsByMethod[enhancer], !model.isEmpty else { return nil }
    if roiEnhancerModelOptions(for: enhancer).contains(where: { $0.name == model }) { return model }
    if model.hasPrefix("/"), FileManager.default.fileExists(atPath: model) { return model }
    return nil
  }

  func selectROIEnhancer(_ enhancer: String) {
    rememberROIEnhancerModel()
    roiEnhancer = enhancer
    if let remembered = rememberedROIEnhancerModel(for: enhancer) {
      roiEnhancerModel = remembered
      if let option = roiEnhancerModelOptions(for: enhancer).first(where: { $0.name == remembered }) {
        roiEnhancerScale = option.scale
      } else {
        applyROIEnhancerScale(forCustomPath: remembered)
      }
      return
    }
    synchronizeROIEnhancerModel(forceDefault: true)
    if enhancer == "roi-distill" || enhancer == "roi-refine" { roiEnhancerScale = 2 }
    if enhancer == "swiftvr" {
      roiEnhancerScale = roiEnhancerScale == 2 ? 2 : 4
    }
  }

  func selectROIEnhancerModel(_ model: String) {
    roiEnhancerModel = model
    if let option = roiEnhancerModelOptions.first(where: { $0.name == model }) {
      roiEnhancerScale = option.scale
    }
    rememberROIEnhancerModel()
  }

  func start() {
    defer { confirmedReplaceOutput = nil }
    guard let inputURL, let outputURL else { return }
    do {
      normalizeModelSelections()
      let resources = try resourceDirectory()
      resetNativeExportBatchState()
      let plan = try MacNativeExportBatchPlanner.plan(
        input: inputURL,
        selectedOutput: outputURL,
        overwrite: overwrite,
        fileName: outputFileName
      )
      try launchNativeExportPlan(plan, resources: resources)
    } catch {
      isRunning = false
      status = "エラー"
      cleanupNativeExportArtifacts()
      resetNativeExportBatchState()
      appendLog(error.localizedDescription + "\n")
    }
  }

  private func resolvedOutputFile(
    input: URL,
    selectedOutput: URL
  ) -> URL {
    MacNativeExportBatchPlanner.resolvedOutputFile(
      input: input,
      selectedOutput: selectedOutput,
      fileName: outputFileName
    )
  }

  private func launchNativeExportPlan(
    _ plan: MacNativeExportBatchPlan,
    resources: URL
  ) throws {
    lineBuffer = ""
    logHistory = ""
    activeProgress.removeAll()
    activeProgressOrder.removeAll()
    nativeExportLastProgressBucket = -1
    log = ""
    progress = 0
    nativeExportBatchPending = plan.items
    nativeExportBatchCurrent = nil
    nativeExportBatchResources = resources
    nativeExportBatchTotal = plan.items.count
    nativeExportBatchCompleted = 0
    nativeExportBatchActive = true
    nativeExportBatchStopRequested = false

    if plan.isDirectoryBatch {
      appendLog(
        "======================================================================\n"
          + "Swiftネイティブ・バッチ書き出し\n"
          + "======================================================================\n"
          + "検出: \(plan.discoveredCount)本 / 未処理: \(plan.items.count)本"
          + " / 出力済みスキップ: \(plan.skippedOutputs.count)本\n"
          + "======================================================================\n\n"
      )
      for skipped in plan.skippedOutputs.prefix(20) {
        appendLog("スキップ（出力済み）: \(skipped.lastPathComponent)\n")
      }
      if plan.skippedOutputs.count > 20 {
        appendLog("ほか \(plan.skippedOutputs.count - 20)本をスキップ\n")
      }
      if !plan.skippedOutputs.isEmpty { appendLog("\n") }
    }

    guard !plan.items.isEmpty else {
      progress = 1
      status = "完了（未処理なし）"
      isRunning = false
      appendLog("未処理の動画はありません。\n")
      resetNativeExportBatchState()
      return
    }
    isRunning = true
    try launchNextNativeExportBatchItem()
  }

  private func launchNextNativeExportBatchItem() throws {
    guard nativeExportBatchActive,
      !nativeExportBatchStopRequested,
      let resources = nativeExportBatchResources,
      let item = nativeExportBatchPending.first
    else { return }
    let itemIndex = nativeExportBatchCompleted + 1
    let nativeTask = try makeNativeExportTask(
      resources: resources,
      input: item.input,
      output: item.output
    )
    nativeExportBatchCurrent = item
    activeProgress.removeAll()
    activeProgressOrder.removeAll()
    nativeExportLastProgressBucket = -1
    lineBuffer = ""
    status = nativeExportBatchTotal > 1
      ? "バッチ \(itemIndex)/\(nativeExportBatchTotal) 準備中"
      : "準備中"

    let pipe = nativeTask.output
    pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      guard !data.isEmpty, let text = String(data: data, encoding: .utf8)
      else { return }
      Task { @MainActor in self?.consume(text) }
    }
    nativeTask.process.terminationHandler = { [weak self, weak pipe] completed in
      pipe?.fileHandleForReading.readabilityHandler = nil
      Task { @MainActor in
        self?.finishNativeExportBatchItem(completed)
      }
    }
    process = nativeTask.process
    try nativeTask.process.run()
    nativeExportBatchPending.removeFirst()
    if nativeExportBatchTotal > 1 {
      appendLog(
        "----------------------------------------------------------------------\n"
          + "バッチ \(itemIndex)/\(nativeExportBatchTotal): "
          + item.input.lastPathComponent + "\n"
          + "----------------------------------------------------------------------\n"
      )
    }
    appendNativeExportStartLog(
      input: item.input,
      output: item.output,
      configuration: nativeTask.configuration
    )
  }

  private func finishNativeExportBatchItem(_ completedProcess: Process) {
    if !lineBuffer.isEmpty {
      let finalLine = lineBuffer
      lineBuffer = ""
      consumeLine(finalLine)
    }
    activeProgress.removeAll()
    activeProgressOrder.removeAll()
    rebuildVisibleLog()
    process = nil
    processInput = nil
    cleanupNativeExportArtifacts()

    guard completedProcess.terminationStatus == 0 else {
      nativeExportBatchPending.removeAll()
      isRunning = false
      status = completedProcess.terminationReason == .uncaughtSignal
        || nativeExportBatchStopRequested ? "停止" : "エラー"
      resetNativeExportBatchState()
      return
    }

    nativeExportBatchCompleted += 1
    progress = nativeExportBatchTotal > 0
      ? Double(nativeExportBatchCompleted) / Double(nativeExportBatchTotal)
      : 1
    if nativeExportBatchTotal > 1 {
      appendLog(
        "バッチ完了: \(nativeExportBatchCompleted)/\(nativeExportBatchTotal)\n\n"
      )
    }
    nativeExportBatchCurrent = nil

    guard !nativeExportBatchStopRequested,
      !nativeExportBatchPending.isEmpty
    else {
      isRunning = false
      status = nativeExportBatchStopRequested ? "停止" : "完了"
      progress = nativeExportBatchStopRequested ? progress : 1
      if !nativeExportBatchStopRequested, nativeExportBatchTotal > 1 {
        appendLog(
          "======================================================================\n"
            + "バッチ処理完了: \(nativeExportBatchCompleted)本\n"
            + "======================================================================\n"
        )
      }
      resetNativeExportBatchState()
      return
    }

    do {
      try launchNextNativeExportBatchItem()
    } catch {
      isRunning = false
      status = "エラー"
      cleanupNativeExportArtifacts()
      appendLog(error.localizedDescription + "\n")
      resetNativeExportBatchState()
    }
  }

  private func resetNativeExportBatchState() {
    nativeExportBatchPending.removeAll()
    nativeExportBatchCurrent = nil
    nativeExportBatchResources = nil
    nativeExportBatchTotal = 0
    nativeExportBatchCompleted = 0
    nativeExportBatchActive = false
    nativeExportBatchStopRequested = false
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
    if nativeExportBatchActive {
      nativeExportBatchStopRequested = true
      nativeExportBatchPending.removeAll()
    }
    if runningNativeExport, let processInput {
      let command = Data("{\"command\":\"stop\"}\n".utf8)
      _ = MacChildProcessPipe.write(command, to: processInput.fileHandleForWriting)
      self.processInput = nil
      try? processInput.fileHandleForWriting.close()
    }
    // A SwiftVR export stops between DiT groups once its control reader sees
    // the stop command, finishing the current scene without SwiftVR; an
    // interrupt would end it in the middle of Core ML inference instead.
    if roiEnhancer != "swiftvr" || !runningNativeExport {
      process?.interrupt()
    }
    status = "停止中"
  }

  private func resumableNativeExport(
    in base: URL,
    input: URL,
    output: URL
  ) -> (directory: URL, configuration: NativeExportConfiguration)? {
    guard resumeInterruptedExport,
      let directories = try? FileManager.default.contentsOfDirectory(
        at: base,
        includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
        options: [.skipsHiddenFiles]
      )
    else { return nil }
    let expectedInput = input.standardizedFileURL.path
    let expectedOutput = output.standardizedFileURL.path
    let candidates = directories.filter {
      $0.lastPathComponent.hasPrefix("mioh-swift-export-")
    }.sorted {
      let left = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
      let right = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
      return left > right
    }
    for directory in candidates {
      let configurationURL = directory.appendingPathComponent("configuration.json")
      guard let data = try? Data(contentsOf: configurationURL),
        let configuration = try? JSONDecoder().decode(NativeExportConfiguration.self, from: data),
        URL(fileURLWithPath: configuration.input).standardizedFileURL.path == expectedInput,
        URL(fileURLWithPath: configuration.outputFile).standardizedFileURL.path == expectedOutput,
        FileManager.default.fileExists(
          atPath: directory.appendingPathComponent("export-g\(configuration.generation)-000000.mp4").path
        )
      else { continue }
      return (directory, configuration)
    }
    return nil
  }

  private func resumeConfigurationsMatch(
    _ previous: NativeExportConfiguration,
    _ current: NativeExportConfiguration
  ) -> Bool {
    func normalized(_ configuration: NativeExportConfiguration) -> NSDictionary? {
      guard let data = try? JSONEncoder().encode(configuration),
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      else { return nil }
      var value = object
      for key in [
        "outputDirectory", "ffmpegTemporaryDirectory", "miohTemporaryDirectory",
        "resumeExistingSegments",
      ] { value.removeValue(forKey: key) }
      return value as NSDictionary
    }
    guard let left = normalized(previous), let right = normalized(current) else { return false }
    return left == right
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
    ].contains(restorationModel)
    else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ書き出しに未対応の復元モデルです: \(restorationModel)"
      )
    }
    guard restorationModel != "カスタム",
      detectionModel != "カスタム"
    else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ書き出しに未対応のモデル指定です"
      )
    }
    // Swift owns the decoder/detector/restorer/encoder pipeline. The native
    // lane count controls its independent Core AI restoration runners.
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
    if FileManager.default.fileExists(atPath: output.path), !overwrite,
      output.standardizedFileURL != confirmedReplaceOutput {
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
    let nativeEnhancer: (url: URL, scale: Int)?
    if roiEnhancer != "none", roiEnhancerStrength > 0 {
      guard let resolved = nativeROIEnhancerAsset(
        resources: resources,
        model: roiEnhancerModel,
        requestedScale: roiEnhancerScale
      ) else {
        throw RunnerError.missingResource(
          "Swift native ROI enhancer: \(roiEnhancerModel)"
        )
      }
      nativeEnhancer = resolved
    } else {
      nativeEnhancer = nil
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
    let resumeCandidate = resumableNativeExport(in: generalBase, input: input, output: output)
    var exportDirectory = resumeCandidate?.directory ?? generalBase.appendingPathComponent(
      "mioh-swift-export-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: exportDirectory, withIntermediateDirectories: true)
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
    var requestedClipFrames = min(
      miohMaximumClipFrames,
      useMaxClipLength
        ? max(1, maxClipLength)
        : (restoration.fixedFrameCount ?? miohMaximumClipFrames)
    )
    if useRestoreMaxFrames, restoreMaxFrames > 0 {
      requestedClipFrames = min(
        requestedClipFrames,
        min(miohMaximumClipFrames, restoreMaxFrames)
      )
    }
    let clipFrames = min(
      min(
        restoration.fixedFrameCount ?? requestedClipFrames,
        miohMaximumClipFrames
      ),
      requestedClipFrames
    )
    let overlap = min(
      max(0, restoreTemporalOverlap),
      max(0, clipFrames - 1)
    )
    let splitMode = noSplit ? "none" : (useSegmentCount ? "count" : "duration")
    let selectedCoreStartNanoseconds: Int64? = usesPartialVideoRange
      ? Int64((normalizedRangeStartSeconds * 1_000_000_000).rounded()) : nil
    let selectedCoreEndNanoseconds: Int64? = usesPartialVideoRange
      ? Int64((normalizedRangeEndSeconds * 1_000_000_000).rounded()) : nil
    // Decode a small lead-in so temporal restoration reaches the selected
    // first frame with real preceding context. Only the exact half-open core
    // range is emitted and muxed.
    let selectedDecodeStartNanoseconds = selectedCoreStartNanoseconds.map {
      max(0, $0 - 500_000_000)
    } ?? 0
    // ROI Refine owns one resident LTX server and serializes every scene on it.
    // Extra restoration lanes cannot run a second LTX request, but retain another
    // full batch and compete for Core AI/GPU memory.
    let effectiveNativeParallelWorkers = roiEnhancer == "roi-refine"
      ? 1 : min(max(nativeParallelWorkers, 1), 10)
    var configuration = NativeExportConfiguration(
      input: input.path,
      inputByteCount: nil,
      inputSHA256: nil,
      outputDirectory: exportDirectory.path,
      ffmpegTemporaryDirectory: ffmpegWorkDirectory.path,
      miohTemporaryDirectory: miohWorkDirectory.path,
      outputFile: output.path,
      ffmpeg: ffmpeg.path,
      detectionModel: detection.url.path,
      detectionBackend: detection.backend,
      detectionInputSize: detection.inputSize,
      detectionCandidateChannels: detection.candidateChannels,
      detectionQueries: detection.queries,
      detectionLogitClasses: detection.logitClasses,
      detectionMaxDet: detection.maxDetections,
      detectionComputeUnits: detection.computeUnits,
      restorationModels: restoration.url.path,
      restorationRunner: restorationRunner.path,
      restorationFrameCount: restoration.fixedFrameCount,
      startNanoseconds: selectedDecodeStartNanoseconds,
      decodeEndNanoseconds: selectedCoreEndNanoseconds,
      outputCoreStartNanoseconds: selectedCoreStartNanoseconds,
      outputCoreEndNanoseconds: selectedCoreEndNanoseconds,
      workerMode: false,
      jobID: nil,
      attemptID: nil,
      generation: 0,
      splitMode: splitMode,
      segmentCount: max(1, segmentCount),
      segmentSeconds: Double(max(10, segmentDuration)),
      temporalBatchFrames: clipFrames,
      temporalOverlap: overlap,
      ringCapacity: max(clipFrames + overlap + 8, 32),
      nativeParallelWorkers: effectiveNativeParallelWorkers,
      decoderBackend: decoderBackend,
      detailedDiagnostics: detailedDiagnostics,
      exportDetectionMasks: exportDetectionMasks,
      overwriteDetectionMasks: overwrite,
      confidenceThreshold: detection.confidenceThreshold,
      blendFeather: Float(blendFeather),
      restoreMaskOnly: restoreMaskOnly,
      multiMosaicMode: multiMosaicMode == "labelField" ? "labelField" : nil,
      sharpenStrength: Float(sharpenStrength),
      detailBoost: Float(detailBoost),
      textureMix: Float(textureMix),
      smoothStrength: Float(smoothStrength),
      effectUpscale: effectUpscale,
      roiEnhancerModel: nativeEnhancer?.url.path,
      roiEnhancerStrength: nativeEnhancer == nil ? 0 : Float(roiEnhancerStrength),
      roiEnhancerScale: nativeEnhancer?.scale ?? max(1, roiEnhancerScale),
      roiExpertMode: roiExpertMode,
      swiftVRTemporalFilter: roiEnhancer != "none"
        ? Float(min(max(swiftVRTemporalFilter, 0), 1)) : nil,
      swiftVRStabilizationRadius: roiEnhancer == "swiftvr" || roiEnhancer == "roi-refine"
        ? min(max(swiftVRStabilizationRadius, 1), 8) : nil,
      swiftVRFrameSmoothing: roiEnhancerSupportsFrameSmoothing
        ? min(max(swiftVRFrameSmoothing, 0), 30) : nil,
      roiRefineDistillModel: roiEnhancer == "roi-refine" && roiRefineDistill
        ? existingPath(resources.appendingPathComponent("models/roi-distill-x2.mlmodelc")) : nil,
      roiRefineDistillMinimumCrop: roiEnhancer == "roi-refine" && roiRefineDistill ? 512 : nil,
      roiRefineMaximumWindowSide: roiEnhancer == "roi-refine"
        ? (roiRefineFastMode ? 384 : roiRefineWindowSide) : nil,
      roiRefineConditionFrames: roiEnhancer == "roi-refine"
        ? (roiRefineFastMode ? 1 : roiRefineContinuationFrames) : nil,
      roiRefineSteps: roiEnhancer == "roi-refine" ? roiRefineSteps : nil,
      detectionEmptyLookahead: max(0, detectionEmptyLookahead),
      detectionMaskReuseSkipFrames: min(
        max(detectionMaskReuseSkipFrames, 0),
        8
      ),
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
    if let resumeCandidate,
      resumeConfigurationsMatch(resumeCandidate.configuration, configuration)
    {
      configuration.resumeExistingSegments = true
    } else if resumeCandidate != nil {
      // A prior run for the same paths used different processing settings. Never mix its segments with this run.
      exportDirectory = generalBase.appendingPathComponent(
        "mioh-swift-export-\(UUID().uuidString)", isDirectory: true)
      try FileManager.default.createDirectory(at: exportDirectory, withIntermediateDirectories: true)
      configuration.outputDirectory = exportDirectory.path
      nativeExportDirectoryURL = exportDirectory
      if ffmpegTempDirectory.isEmpty {
        // The provisional FFmpeg directory was placed under the rejected
        // resume candidate. Keep every new-run artifact under the new export.
        try? FileManager.default.removeItem(at: ffmpegWorkDirectory)
        let freshFFmpegWorkDirectory = exportDirectory.appendingPathComponent(
          "mioh-ffmpeg-export-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
          at: freshFFmpegWorkDirectory, withIntermediateDirectories: true)
        configuration.ffmpegTemporaryDirectory = freshFFmpegWorkDirectory.path
        nativeExportFFmpegDirectoryURL = freshFFmpegWorkDirectory
      }
    }
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
    guard MacChildProcessPipe.prepare(inputPipe.fileHandleForWriting) else {
      throw CocoaError(.fileWriteUnknown)
    }
    processInput = inputPipe
    nativeExportConfigurationURL = configurationURL
    nativeExportPreservesTemporaryFiles = keepTemp && !deleteSegments
    runningNativeExport = true
    return (task, outputPipe, configuration)
  }

  /// Snapshot the current native settings into the immutable contract sent
  /// to cluster workers. The same resolved assets and tree-digest helper are
  /// used by the worker launcher, so an identifier cannot silently select a
  /// different set of weights on another Mac.
  func clusterRestorationOptions() async throws
    -> RemoteClusterRestorationOptions
  {
    normalizeModelSelections()
    guard capabilities.supportsCoreAI else {
      throw RunnerError.unsupportedFeature(
        "クラスタ復元にはmacOS 27以降が必要です"
      )
    }
    guard restorationModel != "カスタム", detectionModel != "カスタム"
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタ復元では同梱モデルだけを使用できます"
      )
    }
    let resources = try resourceDirectory()
    guard let restoration = nativeRestorationAsset(
      resources: resources,
      model: restorationModel
    ) else {
      throw RunnerError.missingResource(
        "Swift native restoration model: \(restorationModel)"
      )
    }
    guard let detection = nativeDetectionAsset(
      resources: resources,
      model: detectionModel
    ) else {
      throw RunnerError.missingResource(
        "Swift native detection model: \(detectionModel)"
      )
    }
    let restorationDigest = try await remoteClusterCanonicalAssetDigest(
      resources: resources,
      modelIdentifier: restorationModel,
      runtimeAsset: restoration.url
    )
    let detectionDigest = try await remoteClusterCanonicalAssetDigest(
      resources: resources,
      modelIdentifier: detectionModel,
      runtimeAsset: detection.url
    )
    var requestedFrames = min(
      miohMaximumClipFrames,
      useMaxClipLength
        ? max(1, maxClipLength)
        : (restoration.fixedFrameCount ?? miohMaximumClipFrames)
    )
    if useRestoreMaxFrames, restoreMaxFrames > 0 {
      requestedFrames = min(
        requestedFrames,
        min(miohMaximumClipFrames, restoreMaxFrames)
      )
    }
    let clipFrames = min(
      restoration.fixedFrameCount ?? requestedFrames,
      requestedFrames
    )
    let overlap = min(max(0, restoreTemporalOverlap), clipFrames - 1)
    let enhancerIdentifier: String? =
      roiEnhancer != "none" && roiEnhancerStrength > 0
      ? roiEnhancerModel : nil
    let enhancerAsset: (url: URL, scale: Int)?
    if let enhancerIdentifier {
      guard !enhancerIdentifier.hasPrefix("/"),
        let resolved = nativeROIEnhancerAsset(
          resources: resources,
          model: enhancerIdentifier,
          requestedScale: roiEnhancerScale
        ) else {
        throw RunnerError.unsupportedFeature(
          "クラスタ復元では同梱ROIエンハンサーだけを使用できます"
        )
      }
      enhancerAsset = resolved
    } else {
      enhancerAsset = nil
    }
    let enhancerDigest: String?
    if let enhancerAsset, let enhancerIdentifier {
      enhancerDigest = try await remoteClusterCanonicalAssetDigest(
        resources: resources,
        modelIdentifier: enhancerIdentifier,
        runtimeAsset: enhancerAsset.url
      )
    } else {
      enhancerDigest = nil
    }
    // Hoisted with explicit types: inline ternaries in an initializer this wide
    // push the expression past the type checker's budget.
    let enhancerStrength: Float = enhancerIdentifier == nil
      ? 0 : Float(roiEnhancerStrength)
    let codec: String = encodingPreset.hasPrefix("h264") ? "h264" : "hevc"
    let clampedEffectUpscale: Int = max(1, min(4, effectUpscale))
    let enhancerScale: Int = max(1, min(8, roiEnhancerScale))
    return RemoteClusterRestorationOptions(
      restorationModelIdentifier: restorationModel,
      restorationAssetSHA256: restorationDigest,
      detectorModelIdentifier: detectionModel,
      detectorAssetSHA256: detectionDigest,
      restorationClipLength: clipFrames,
      temporalOverlap: overlap,
      crossfade: restoreCrossfade,
      detectionEmptyLookahead: min(max(detectionEmptyLookahead, 1), 300),
      detectFaceMosaics: detectFaceMosaics,
      detectionMaskReuseSkipFrames: min(
        max(detectionMaskReuseSkipFrames, 0),
        8
      ),
      blendFeather: Float(blendFeather),
      sharpenStrength: Float(sharpenStrength),
      detailBoost: Float(detailBoost),
      textureMix: Float(textureMix),
      smoothStrength: Float(smoothStrength),
      effectUpscale: clampedEffectUpscale,
      roiEnhancerModelIdentifier: enhancerIdentifier,
      roiEnhancerAssetSHA256: enhancerDigest,
      roiEnhancerStrength: enhancerStrength,
      roiEnhancerScale: enhancerScale,
      videoCodec: codec,
      bitrateMultiplier: bitrateMultiplier,
      mp4FastStart: mp4FastStart,
      targetFPSNumerator: useFPS ? max(1, fps) : nil,
      targetFPSDenominator: useFPS ? max(1, fpsDenominator) : nil
    )
  }

  /// Advertises the same portable identities used by job contracts. Optional
  /// maps keep old Workers wire-compatible while allowing new coordinators to
  /// reject same-name/different-weight nodes before dispatch.
  func clusterCanonicalAssetIdentityMaps() throws -> (
    restoration: [String: String]?, detector: [String: String]?
  ) {
    let resources = try resourceDirectory()
    guard let manifest = try remoteClusterCanonicalManifest(resources: resources) else {
      return (nil, nil)
    }
    let restorationIDs = Set(restorationModels.filter { $0 != "カスタム" })
    let detectorIDs = Set(detectionModels.filter { $0 != "カスタム" })
    var restoration: [String: String] = [:]
    var detector: [String: String] = [:]
    for modelID in restorationIDs {
      if let identity = manifest.models[modelID] {
        restoration[modelID] = try Self.validatedCanonicalSHA256(
          identity,
          modelIdentifier: modelID
        )
      }
    }
    for modelID in detectorIDs {
      if let identity = manifest.models[modelID] {
        detector[modelID] = try Self.validatedCanonicalSHA256(
          identity,
          modelIdentifier: modelID
        )
      }
    }
    return (restoration, detector)
  }

  func clusterResolvedOutputFile() -> URL? {
    guard let inputURL, let outputURL else { return nil }
    return resolvedOutputFile(input: inputURL, selectedOutput: outputURL)
  }

  func remoteClusterJobLauncher() -> RemoteClusterJobLauncher {
    { [weak self] request, inputURL, outputURL in
      guard let self else {
        throw RunnerError.unsupportedFeature(
          "クラスタワーカーが終了しました"
        )
      }
      return try await self.runRemoteClusterWorkerJob(
        request,
        inputURL: inputURL,
        outputURL: outputURL
      )
    }
  }

  func runRemoteClusterWorkerJob(
    _ request: RemoteClusterJobRequest,
    inputURL: URL,
    outputURL: URL
  ) async throws -> RemoteClusterJobMetrics {
    guard capabilities.supportsCoreAI else {
      throw RunnerError.unsupportedFeature(
        "クラスタ復元にはmacOS 27以降が必要です"
      )
    }
    guard request.protocolVersion == RemoteClusterJobRequest.protocolVersion,
      request.mediaRange.isValid,
      request.options.isValid
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタジョブの設定がworker v1契約と一致しません"
      )
    }
    if request.httpTransfer == nil {
      let inputValues = try inputURL.resourceValues(forKeys: [
        .isRegularFileKey, .fileSizeKey,
      ])
      guard inputValues.isRegularFile == true,
        Int64(inputValues.fileSize ?? -1) == request.inputByteCount
      else {
        throw RunnerError.unsupportedFeature(
          "クラスタ入力ファイルのサイズがジョブ契約と一致しません"
        )
      }
    }
    guard !inputURL.isFileURL
      || inputURL.standardizedFileURL != outputURL.standardizedFileURL
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタ入力と出力に同じファイルは指定できません"
      )
    }
    guard !FileManager.default.fileExists(atPath: outputURL.path) else {
      throw RunnerError.unsupportedFeature(
        "クラスタ出力ファイルが既に存在します: \(outputURL.path)"
      )
    }

    let resources = try resourceDirectory()
    guard let restoration = nativeRestorationAsset(
      resources: resources,
      model: request.options.restorationModelIdentifier
    ) else {
      throw RunnerError.missingResource(
        "Swift native restoration model: "
          + request.options.restorationModelIdentifier
      )
    }
    guard let detection = nativeDetectionAsset(
      resources: resources,
      model: request.options.detectorModelIdentifier
    ) else {
      throw RunnerError.missingResource(
        "Swift native detection model: "
          + request.options.detectorModelIdentifier
      )
    }
    let restorationDigest = try await remoteClusterCanonicalAssetDigest(
      resources: resources,
      modelIdentifier: request.options.restorationModelIdentifier,
      runtimeAsset: restoration.url
    )
    let detectionDigest = try await remoteClusterCanonicalAssetDigest(
      resources: resources,
      modelIdentifier: request.options.detectorModelIdentifier,
      runtimeAsset: detection.url
    )
    guard restorationDigest
      == request.options.restorationAssetSHA256.lowercased()
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタ復元モデルのSHA-256が一致しません"
      )
    }
    guard detectionDigest == request.options.detectorAssetSHA256.lowercased()
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタ検出モデルのSHA-256が一致しません"
      )
    }

    let runner = resources.appendingPathComponent(
      "bin/mioh-native-coreai-preview"
    )
    let ffmpeg = resources.appendingPathComponent("bin/ffmpeg")
    let restorationRunner = resources.appendingPathComponent(
      "bin/\(restoration.runnerName)"
    )
    for executable in [runner, ffmpeg, restorationRunner]
    where !FileManager.default.isExecutableFile(atPath: executable.path) {
      throw RunnerError.missingResource(executable.lastPathComponent)
    }

    let enhancer: (url: URL, scale: Int)?
    if let identifier = request.options.roiEnhancerModelIdentifier,
      request.options.roiEnhancerStrength > 0
    {
      // Cluster identifiers are names, never arbitrary absolute paths.
      guard !identifier.hasPrefix("/"),
        let resolved = nativeROIEnhancerAsset(
          resources: resources,
          model: identifier,
          requestedScale: request.options.roiEnhancerScale
        )
      else {
        throw RunnerError.missingResource(
          "Swift native ROI enhancer: \(identifier)"
        )
      }
      enhancer = resolved
    } else {
      enhancer = nil
    }
    if let enhancer {
      guard let identifier = request.options.roiEnhancerModelIdentifier,
        let expected = request.options.roiEnhancerAssetSHA256
      else {
        throw RunnerError.unsupportedFeature(
          "クラスタROIエンハンサーのSHA-256がありません"
        )
      }
      let digest = try await remoteClusterCanonicalAssetDigest(
        resources: resources,
        modelIdentifier: identifier,
        runtimeAsset: enhancer.url
      )
      guard digest == expected.lowercased() else {
        throw RunnerError.unsupportedFeature(
          "クラスタROIエンハンサーのSHA-256が一致しません"
        )
      }
    }

    let base = tempDirectory.isEmpty
      ? FileManager.default.temporaryDirectory
      : URL(fileURLWithPath: tempDirectory, isDirectory: true)
    let workerRoot = base.appendingPathComponent(
      "mioh-cluster-\(request.attemptID.uuidString.lowercased())",
      isDirectory: true
    )
    let segmentDirectory = workerRoot.appendingPathComponent(
      "segments",
      isDirectory: true
    )
    let ffmpegDirectory = workerRoot.appendingPathComponent(
      "ffmpeg",
      isDirectory: true
    )
    let runtimeDirectory = workerRoot.appendingPathComponent(
      "runtime",
      isDirectory: true
    )
    // The ledger reserves an extensionless `.part` staging path. ffmpeg needs
    // a real media extension while muxing, so finalize inside the private
    // worker directory and atomically move that artifact to the reservation.
    let artifactURL = workerRoot.appendingPathComponent("artifact.mp4")
    try FileManager.default.createDirectory(
      at: workerRoot,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o700],
      ofItemAtPath: workerRoot.path
    )
    try FileManager.default.createDirectory(
      at: segmentDirectory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    try FileManager.default.createDirectory(
      at: ffmpegDirectory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    try FileManager.default.createDirectory(
      at: runtimeDirectory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    try FileManager.default.createDirectory(
      at: outputURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: workerRoot) }

    let requestedFrames = min(
      miohMaximumClipFrames,
      max(1, request.options.restorationClipLength)
    )
    let clipFrames = min(
      restoration.fixedFrameCount ?? requestedFrames,
      requestedFrames
    )
    let overlap = min(
      max(0, request.options.temporalOverlap),
      clipFrames - 1
    )
    let configuration = NativeExportConfiguration(
      input: inputURL.isFileURL ? inputURL.path : inputURL.absoluteString,
      inputByteCount: request.httpTransfer == nil ? nil : request.inputByteCount,
      inputSHA256: request.httpTransfer == nil ? nil : request.inputSHA256,
      outputDirectory: segmentDirectory.path,
      ffmpegTemporaryDirectory: ffmpegDirectory.path,
      miohTemporaryDirectory: runtimeDirectory.path,
      outputFile: artifactURL.path,
      ffmpeg: ffmpeg.path,
      detectionModel: detection.url.path,
      detectionBackend: detection.backend,
      detectionInputSize: detection.inputSize,
      detectionCandidateChannels: detection.candidateChannels,
      detectionQueries: detection.queries,
      detectionLogitClasses: detection.logitClasses,
      detectionMaxDet: detection.maxDetections,
      detectionComputeUnits: detection.computeUnits,
      restorationModels: restoration.url.path,
      restorationRunner: restorationRunner.path,
      restorationFrameCount: restoration.fixedFrameCount,
      startNanoseconds: request.mediaRange.decodeStartNanoseconds,
      decodeEndNanoseconds: request.mediaRange.decodeEndNanoseconds,
      outputCoreStartNanoseconds: request.mediaRange.coreStartNanoseconds,
      outputCoreEndNanoseconds: request.mediaRange.coreEndNanoseconds,
      workerMode: true,
      jobID: request.jobID.uuidString.lowercased(),
      attemptID: request.attemptID.uuidString.lowercased(),
      generation: 0,
      splitMode: "none",
      segmentCount: 1,
      segmentSeconds: 60,
      temporalBatchFrames: clipFrames,
      temporalOverlap: overlap,
      ringCapacity: max(clipFrames + overlap + 8, 32),
      nativeParallelWorkers: 1,
      // Remote workers retain their existing reader contract; local diagnostics
      // do not silently change a cluster's remote input transport.
      confidenceThreshold: detection.confidenceThreshold,
      blendFeather: request.options.blendFeather,
      restoreMaskOnly: true,
      sharpenStrength: request.options.sharpenStrength,
      detailBoost: request.options.detailBoost,
      textureMix: request.options.textureMix,
      smoothStrength: request.options.smoothStrength,
      effectUpscale: max(1, min(4, request.options.effectUpscale)),
      roiEnhancerModel: enhancer?.url.path,
      roiEnhancerStrength: enhancer == nil
        ? 0 : request.options.roiEnhancerStrength,
      roiEnhancerScale: enhancer?.scale
        ?? max(1, request.options.roiEnhancerScale),
      roiExpertMode: false,
      detectionEmptyLookahead: request.options.detectionEmptyLookahead,
      detectionMaskReuseSkipFrames:
        request.options.detectionMaskReuseSkipFrames ?? 0,
      detectFaceMosaics: request.options.detectFaceMosaics,
      crossfade: request.options.crossfade,
      targetFPS: request.options.targetFPSNumerator,
      targetFPSDenominator: request.options.targetFPSDenominator,
      preFPSConversion: false,
      videoCodec: request.options.videoCodec,
      averageBitRate: nil,
      bitrateMultiplier: request.options.bitrateMultiplier,
      mp4FastStart: request.options.mp4FastStart
    )
    let configurationURL = workerRoot.appendingPathComponent(
      "configuration.json"
    )
    try JSONEncoder().encode(configuration).write(
      to: configurationURL,
      options: .atomic
    )
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o600],
      ofItemAtPath: configurationURL.path
    )

    let task = Process()
    let inputPipe = Pipe()
    let outputPipe = Pipe()
    task.executableURL = runner
    task.arguments = [configurationURL.path]
    task.standardInput = inputPipe
    task.standardOutput = outputPipe
    task.standardError = outputPipe
    var environment = ProcessInfo.processInfo.environment
    environment["TMPDIR"] = runtimeDirectory.path
    environment["TEMP"] = runtimeDirectory.path
    environment["TMP"] = runtimeDirectory.path
    task.environment = environment
    guard MacChildProcessPipe.prepare(inputPipe.fileHandleForWriting) else {
      throw CocoaError(.fileWriteUnknown)
    }

    let capture = RemoteClusterOutputCapture()
    let result = RemoteClusterProcessResult()
    let control = RemoteClusterProcessControl(
      process: task,
      input: inputPipe
    )
    task.terminationHandler = { completed in
      result.finish(completed.terminationStatus)
    }
    let startedAt = Date()
    try Task.checkCancellation()
    try task.run()
    control.didLaunch()
    let drainTask = Task.detached(priority: .utility) {
      let handle = outputPipe.fileHandleForReading
      while let chunk = try? handle.read(upToCount: 64 * 1_024),
        !chunk.isEmpty
      {
        capture.append(chunk)
      }
    }
    let status = await withTaskCancellationHandler {
      await result.value()
    } onCancel: {
      control.cancel()
    }
    await drainTask.value
    try Task.checkCancellation()
    let processOutput = capture.snapshot()
    guard status == 0 else {
      let detail = String(data: processOutput, encoding: .utf8) ?? ""
      throw RunnerError.unsupportedFeature(
        "クラスタワーカーが終了コード\(status)で失敗しました: "
          + String(detail.suffix(2_000))
      )
    }

    var artifact: [String: Any]?
    for line in processOutput.split(separator: 0x0A) {
      guard let object = try? JSONSerialization.jsonObject(with: Data(line))
        as? [String: Any],
        object["kind"] as? String == "cluster_artifact"
      else { continue }
      artifact = object
    }
    guard let artifact,
      artifact["job_id"] as? String
        == request.jobID.uuidString.lowercased(),
      artifact["attempt_id"] as? String
        == request.attemptID.uuidString.lowercased(),
      (artifact["path"] as? String).map({
        URL(fileURLWithPath: $0).standardizedFileURL
          == artifactURL.standardizedFileURL
      }) == true,
      let processedFrames = (artifact["frame_count"] as? NSNumber)?.intValue,
      processedFrames > 0,
      (artifact["core_start_ns"] as? NSNumber)?.int64Value
        == request.mediaRange.coreStartNanoseconds,
      (artifact["core_end_ns"] as? NSNumber)?.int64Value
        == request.mediaRange.coreEndNanoseconds,
      artifact["has_audio"] as? Bool == false,
      FileManager.default.fileExists(atPath: artifactURL.path)
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタワーカーの成果物イベントが不正です"
      )
    }
    try FileManager.default.moveItem(at: artifactURL, to: outputURL)
    let outputValues = try outputURL.resourceValues(forKeys: [
      .isRegularFileKey, .fileSizeKey,
    ])
    guard outputValues.isRegularFile == true,
      let outputSize = outputValues.fileSize, outputSize > 0
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタワーカーの出力ファイルが空です"
      )
    }
    return RemoteClusterJobMetrics(
      processedFrames: processedFrames,
      wallSeconds: max(0.001, Date().timeIntervalSince(startedAt)),
      outputByteCount: Int64(outputSize)
    )
  }

  private func remoteClusterAssetDigest(_ url: URL) async throws -> String {
    let key = url.standardizedFileURL.path
    if let cached = remoteClusterAssetDigests[key] { return cached }
    let digest = try await Task.detached(priority: .utility) {
      try Self.computeRemoteClusterAssetSHA256(url)
    }.value
    remoteClusterAssetDigests[key] = digest
    return digest
  }

  /// Returns the digest of the portable source model represented by a model
  /// ID. Both coordinator and Worker use this path, so a Mac `.aimodelc` and
  /// an iPad `.aimodel` compare equal when they came from the same source.
  /// Models absent from the v1 manifest retain the previous runtime-tree
  /// digest as a compatibility fallback.
  private func remoteClusterCanonicalAssetDigest(
    resources: URL,
    modelIdentifier: String,
    runtimeAsset: URL
  ) async throws -> String {
    let cacheKey = "canonical-v1:\(modelIdentifier)"
    if let cached = remoteClusterAssetDigests[cacheKey] { return cached }

    let manifestURL = resources.appendingPathComponent(
      "models/mioh-cluster-model-identities-v1.json"
    )
    let digest: String
    if FileManager.default.fileExists(atPath: manifestURL.path),
      let manifest = try remoteClusterCanonicalManifest(resources: resources)
    {
      if let identity = manifest.models[modelIdentifier] {
        digest = try Self.validatedCanonicalSHA256(
          identity,
          modelIdentifier: modelIdentifier
        )
      } else {
        digest = try await remoteClusterAssetDigest(runtimeAsset)
      }
    } else {
      digest = try await remoteClusterAssetDigest(runtimeAsset)
    }
    remoteClusterAssetDigests[cacheKey] = digest
    return digest
  }

  private func remoteClusterCanonicalManifest(
    resources: URL
  ) throws -> RemoteClusterCanonicalModelManifest? {
    let manifestURL = resources.appendingPathComponent(
      "models/mioh-cluster-model-identities-v1.json"
    )
    guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
    let manifest = try JSONDecoder().decode(
      RemoteClusterCanonicalModelManifest.self,
      from: Data(contentsOf: manifestURL)
    )
    guard manifest.formatVersion == 1,
      manifest.digestAlgorithm == "sha256-tree-v1"
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタモデルidentity manifestの形式に対応していません"
      )
    }
    return manifest
  }

  private nonisolated static func validatedCanonicalSHA256(
    _ identity: RemoteClusterCanonicalModelIdentity,
    modelIdentifier: String
  ) throws -> String {
    let normalized = identity.sha256.lowercased()
    guard normalized.utf8.count == 64,
      normalized.utf8.allSatisfy({ byte in
        (48...57).contains(byte) || (97...102).contains(byte)
      }),
      !identity.assetType.isEmpty,
      !identity.sourceAssets.isEmpty
    else {
      throw RunnerError.unsupportedFeature(
        "クラスタモデルidentity manifestのエントリが不正です: \(modelIdentifier)"
      )
    }
    return normalized
  }

  /// Asset digest convention shared by coordinator and worker:
  /// * a regular file hashes its raw bytes;
  /// * a directory rejects symlinks, recursively includes regular files only,
  ///   sorts NFC-normalized POSIX relative paths by UTF-8 bytes, then hashes
  ///   `relativePath + NUL + fileBytes + NUL` for every entry.
  /// The result is lowercase hexadecimal SHA-256.
  private nonisolated static func computeRemoteClusterAssetSHA256(
    _ url: URL
  ) throws -> String {
    let keys: Set<URLResourceKey> = [
      .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
    ]
    let rootValues = try url.resourceValues(forKeys: keys)
    guard rootValues.isSymbolicLink != true else {
      throw RunnerError.unsupportedFeature(
        "クラスタモデル資産にシンボリックリンクは使用できません"
      )
    }

    func updateFile(_ file: URL, hasher: inout SHA256) throws {
      let handle = try FileHandle(forReadingFrom: file)
      defer { try? handle.close() }
      while let chunk = try handle.read(upToCount: 1_024 * 1_024),
        !chunk.isEmpty
      {
        hasher.update(data: chunk)
      }
    }

    var hasher = SHA256()
    if rootValues.isRegularFile == true {
      try updateFile(url, hasher: &hasher)
    } else if rootValues.isDirectory == true {
      var enumerationError: Error?
      guard let enumerator = FileManager.default.enumerator(
        at: url,
        includingPropertiesForKeys: Array(keys),
        options: [],
        errorHandler: { _, error in
          enumerationError = error
          return false
        }
      ) else {
        throw RunnerError.missingResource(url.lastPathComponent)
      }
      let rootPath = url.standardizedFileURL.path
      var entries: [(relative: String, url: URL)] = []
      for case let candidate as URL in enumerator {
        let values = try candidate.resourceValues(forKeys: keys)
        guard values.isSymbolicLink != true else {
          throw RunnerError.unsupportedFeature(
            "クラスタモデル資産にシンボリックリンクは使用できません"
          )
        }
        guard values.isRegularFile == true else { continue }
        let candidatePath = candidate.standardizedFileURL.path
        guard candidatePath.hasPrefix(rootPath + "/") else {
          throw RunnerError.unsupportedFeature(
            "クラスタモデル資産のパスが不正です"
          )
        }
        let relative = String(candidatePath.dropFirst(rootPath.count + 1))
          .precomposedStringWithCanonicalMapping
        entries.append((relative, candidate))
      }
      if let enumerationError { throw enumerationError }
      entries.sort {
        Array($0.relative.utf8).lexicographicallyPrecedes(
          Array($1.relative.utf8)
        )
      }
      guard Set(entries.map(\.relative)).count == entries.count else {
        throw RunnerError.unsupportedFeature(
          "クラスタモデル資産に正規化後の重複パスがあります"
        )
      }
      for entry in entries {
        hasher.update(data: Data(entry.relative.utf8))
        hasher.update(data: Data([0]))
        try updateFile(entry.url, hasher: &hasher)
        hasher.update(data: Data([0]))
      }
    } else {
      throw RunnerError.missingResource(url.lastPathComponent)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
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
      入力: \(configuration.detailedDiagnostics == true ? "診断ログでは非表示" : input.path)
      出力: \(configuration.detailedDiagnostics == true ? "診断ログでは非表示" : output.path)
      処理範囲: \(configuration.outputCoreStartNanoseconds.flatMap { start in configuration.outputCoreEndNanoseconds.map { end in "\(nativeLogDuration(Double(start) / 1_000_000_000)) 〜 \(nativeLogDuration(Double(end) / 1_000_000_000))（\(nativeLogDuration(Double(end - start) / 1_000_000_000))）" } } ?? "動画全体")
      読み込み方式: \(configuration.decoderBackend ?? "avfoundationAsync")
      詳細診断: \(configuration.detailedDiagnostics == true ? "有効（5秒間隔）" : "無効")
      復元モデル: \(restorationModel)
      検出モデル: \(detectionModel)
      分割: \(splitDescription)
      中断再開: \(configuration.resumeExistingSegments == true ? "完成済みセグメントを検査して再利用" : "新規処理")
      復元Clip長: \(configuration.temporalBatchFrames)フレーム
      ネイティブ並列数: \(configuration.nativeParallelWorkers)レーン
      Temporal overlap: \(configuration.temporalOverlap)フレーム
      Crossfade: \(configuration.crossfade ? "有効" : "無効")
      シャープ: \(String(format: "%.2f", configuration.sharpenStrength))
      ディテール: \(String(format: "%.2f", configuration.detailBoost))
      テクスチャ: \(String(format: "%.2f", configuration.textureMix))
      スムージング: \(String(format: "%.2f", configuration.smoothStrength))
      エフェクト倍率: \(configuration.effectUpscale)x
      ROIエンハンサー: \(configuration.roiEnhancerModel == nil ? "無効" : "有効（\(roiEnhancerModel)、強度 \(String(format: "%.2f", configuration.roiEnhancerStrength))）")
      ROI Refineモード: \(configuration.roiRefineMaximumWindowSide.map { "最大\($0)px・引き継ぎ\(configuration.roiRefineConditionFrames ?? 9)フレーム・\(configuration.roiRefineSteps ?? 2)ステップ" } ?? "対象外")
      枠のなめらかさ: \(configuration.swiftVRFrameSmoothing ?? 0)フレーム
      大きいROIのDistill重ね: \(configuration.roiRefineDistillModel == nil ? "なし" : "512px超に適用")
      揺らぎ低減: \(String(format: "%.2f", configuration.swiftVRTemporalFilter ?? 0))（Apple時間方向ノイズ補正）
      空検出先読み: \(configuration.detectionEmptyLookahead)フレーム
      検出後スキップ: \(configuration.detectionMaskReuseSkipFrames)フレーム
      検出マスクJSON保存: \(configuration.exportDetectionMasks == true ? "有効（出力動画の隣に .masks.json）" : "無効")
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
    case "basicvsrpp-v1.2-coreai-variable":
      prefix = "basicvsrpp-v1.2-variable-coreai"
      fixedFrameCount = nil
      runnerName = "lada-basicvsrpp-variable-runner"
    default:
      return nil
    }
    for suffix in [".aimodelc", ".aimodel"] {
      if let url = firstModelAsset(
        in: models,
        prefixes: [prefix],
        suffix: suffix
      ) {
        return (url, fixedFrameCount, runnerName)
      }
    }
    return nil
  }

  private func nativeDetectionAsset(
    resources: URL,
    model: String
  ) -> (
    url: URL,
    backend: String,
    inputSize: Int,
    candidateChannels: Int,
    queries: Int,
    logitClasses: Int,
    maxDetections: Int,
    confidenceThreshold: Float,
    computeUnits: String?
  )? {
    let models = resources.appendingPathComponent("models", isDirectory: true)
    let requestsCoreAI = model.hasSuffix("-coreai")
    let requestsCoreML = model.hasSuffix("-coreml")
    let base = model
      .replacingOccurrences(of: "-coreai", with: "")
      .replacingOccurrences(of: "-coreml", with: "")
    if model == "celfnet-o2-coreml" {
      for suffix in [".mlmodelc", ".mlpackage"] {
        if let asset = firstModelAsset(
          in: models, prefixes: ["celfnet-o2-epoch09"], suffix: suffix
        ) {
          return (asset, "celfnet", 960, 0, 0, 0, 0, 0.5, "cpuAndGPU")
        }
      }
      return nil
    }
    if base == "jasna-v6" || base == "jasna-v6-large" {
      let large = base == "jasna-v6-large"
      let stem = large
        ? "rfdetr-v6-large-768-fp32"
        : "rfdetr-v6-576-fp32"
      if requestsCoreML {
        for suffix in [".mlmodelc", ".mlpackage"] {
          if let coreML = firstModelAsset(
            in: models,
            prefixes: [stem],
            suffix: suffix
          ) {
            return (
              coreML,
              "rfdetr",
              large ? 768 : 576,
              0,
              200,
              3,
              16,
              large ? 0.40 : 0.35,
              "cpuAndGPU"
            )
          }
        }
        return nil
      }
      guard requestsCoreAI else { return nil }
      for suffix in [".aimodelc", ".aimodel"] {
        if let coreAI = firstModelAsset(
          in: models,
          prefixes: [stem],
          suffix: suffix
        ) {
          return (
            coreAI,
            "rfdetr",
            large ? 768 : 576,
            0,
            200,
            3,
            16,
            large ? 0.40 : 0.35,
            nil
          )
        }
      }
      return nil
    }
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
    func coreMLAsset() -> URL? {
      for suffix in [".mlmodelc", ".mlpackage"] {
        if let url = firstModelAsset(in: models, prefixes: [stem], suffix: suffix) {
          return url
        }
      }
      return nil
    }
    func coreAIAsset() -> URL? {
      for suffix in [".aimodelc", ".aimodel"] {
        if let url = firstModelAsset(
          in: models,
          prefixes: ["\(stem)-fp16", stem],
          suffix: suffix
        ) {
          return url
        }
      }
      return nil
    }
    // Explicit backend suffixes are contracts, not preferences. Unsuffixed
    // legacy identifiers keep the measured Core ML-first fallback behavior.
    if !requestsCoreAI, let coreML = coreMLAsset() {
      return (
        coreML, "yolo", 640, candidateChannels,
        0, 0, 0, 0.25, "cpuAndNeuralEngine"
      )
    }
    if !requestsCoreML, let coreAI = coreAIAsset() {
      return (
        coreAI, "yolo", 640, candidateChannels,
        0, 0, 0, 0.25, nil
      )
    }
    return nil
  }

  /// Resolve the selected UI model to a backend the Swift pipeline can load.
  /// Core AI selections stay on Core AI when a compiled asset is available;
  /// Core ML and legacy PyTorch labels resolve to the equivalent Core ML
  /// export. This keeps the user's chosen model semantics in the native
  /// pipeline.
  private func existingPath(_ url: URL) -> String? {
    FileManager.default.fileExists(atPath: url.path) ? url.path : nil
  }

  private func nativeROIEnhancerAsset(
    resources: URL,
    model: String,
    requestedScale: Int
  ) -> (url: URL, scale: Int)? {
    let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    let custom = URL(fileURLWithPath: trimmed)
    if FileManager.default.fileExists(atPath: custom.appendingPathComponent(
      "roi-refine-pack.json").path) {
      // A ROI Refine pack: 256px BasicVSR++ ROI redrawn at 512px.
      return (custom, 2)
    }
    if FileManager.default.fileExists(atPath: custom.appendingPathComponent(
      "reae-stateful-encoder-28f-1024-fp32.mlpackage").path),
      FileManager.default.fileExists(atPath: custom.appendingPathComponent(
        "native-4x-t7-fp16/components/patch.mlpackage").path) {
      // A SwiftVR pack: 4x, or 2x when chosen (512px output, about a quarter
      // of the DiT work). The worker checks that the 2x assets exist.
      return (custom, requestedScale == 2 ? 2 : 4)
    }
    if custom.isFileURL,
      FileManager.default.fileExists(atPath: custom.path),
      ["aimodel", "aimodelc", "mlpackage", "mlmodelc"].contains(
        custom.pathExtension.lowercased()
      )
    {
      return (custom, max(1, min(8, requestedScale)))
    }

    let models = resources.appendingPathComponent("models", isDirectory: true)
    let scale: Int
    let coreAIPrefixes: [String]
    let coreMLPrefixes: [String]
    switch trimmed {
    case "roi-distill-x2-coreml":
      scale = 2
      coreAIPrefixes = []
      coreMLPrefixes = ["roi-distill-x2"]
    default:
      return nil
    }

    for suffix in [".aimodelc", ".aimodel"] where !coreAIPrefixes.isEmpty {
      if let asset = firstModelAsset(
        in: models,
        prefixes: coreAIPrefixes,
        suffix: suffix
      ) {
        return (asset, scale)
      }
    }
    for suffix in [".mlmodelc", ".mlpackage"] {
      if let asset = firstModelAsset(
        in: models,
        prefixes: coreMLPrefixes,
        suffix: suffix
      ) {
        return (asset, scale)
      }
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

  /// Starts from the settings as last used; before any were remembered,
  /// from the saved defaults.
  private func loadSavedDefaultsOnLaunch() {
    if let data = UserDefaults.standard.data(forKey: lastSettingsKey),
      let snapshot = try? JSONDecoder().decode(MiohUserDefaultsSnapshot.self, from: data)
    {
      apply(defaults: snapshot)
      defaultsStatus = "前回の設定で起動しました"
      return
    }
    guard let data = UserDefaults.standard.data(forKey: defaultsKey),
      let snapshot = try? JSONDecoder().decode(MiohUserDefaultsSnapshot.self, from: data)
    else { return }
    apply(defaults: snapshot)
    defaultsStatus = "保存済みデフォルトを適用済み"
  }

  /// Single source of truth for both local defaults and remote configuration.
  /// Callers must still validate untrusted input before applying it.
  func currentDefaultsSnapshot() -> MiohUserDefaultsSnapshot {
    MiohUserDefaultsSnapshot(
      inputPath: inputURL?.path,
      outputPath: outputURL?.path,
      tempDirectory: tempDirectory,
      ffmpegTempDirectory: ffmpegTempDirectory,
      ladaTempDirectory: ladaTempDirectory,
      overwrite: overwrite,
      restorationEngine: restorationEngine,
      parallelWorkers: parallelWorkers,
      nativeParallelWorkers: nativeParallelWorkers,
      decoderBackend: decoderBackend,
      detailedDiagnostics: detailedDiagnostics,
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
      restoreMaskOnly: restoreMaskOnly,
      multiMosaicMode: multiMosaicMode,
      textureMix: textureMix,
      smoothStrength: smoothStrength,
      effectUpscale: effectUpscale,
      roiEnhancer: roiEnhancer,
      roiEnhancerModel: roiEnhancerModel,
      roiEnhancerModelsByMethod: roiEnhancerModelsByMethod,
      roiEnhancerScale: roiEnhancerScale,
      roiEnhancerStrength: roiEnhancerStrength,
      roiEnhancerTile: roiEnhancerTile,
      roiExpertMode: roiExpertMode,
      swiftVRTemporalFilter: swiftVRTemporalFilter,
      swiftVRStabilizationRadius: swiftVRStabilizationRadius,
      swiftVRFrameSmoothing: swiftVRFrameSmoothing,
      roiRefineDistill: roiRefineDistill,
      roiRefineWindowSide: roiRefineWindowSide,
      roiRefineContinuationFrames: roiRefineContinuationFrames,
      roiRefineSteps: roiRefineSteps,
      roiRefineUse1024Window: roiRefineWindowSide == 1024,
      roiRefineFastMode: roiRefineFastMode,
      detectionModel: detectionModel,
      customDetectionModel: customDetectionModel,
      detectionEmptyLookahead: detectionEmptyLookahead,
      detectionMaskReuseSkipFrames: detectionMaskReuseSkipFrames,
      exportDetectionMasks: exportDetectionMasks,
      detectFaceMosaics: detectFaceMosaics,
      previewBufferLimit: previewBufferLimit,
      previewRestorationModel: previewRestorationModel,
      previewCustomRestorationModel: previewCustomRestorationModel,
      previewDetectionModel: previewDetectionModel,
      previewCustomDetectionModel: previewCustomDetectionModel,
      previewRealtimeOptimization: previewRealtimeOptimization,
      previewLimitHighFrameRate: previewLimitHighFrameRate,
      previewUseSafariCompatibleHLS: previewUseSafariCompatibleHLS,
      previewHLSQuality: previewHLSQuality,
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

  /// Applies a previously validated snapshot using exactly the same
  /// normalization rules as the local settings UI/defaults loader.
  func apply(defaults snapshot: MiohUserDefaultsSnapshot) {
    inputURL = snapshot.inputPath.flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
    outputURL = snapshot.outputPath.flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
    refreshOutputPathText()
    tempDirectory = snapshot.tempDirectory
    ffmpegTempDirectory = snapshot.ffmpegTempDirectory
    ladaTempDirectory = snapshot.ladaTempDirectory
    overwrite = snapshot.overwrite
    restorationEngine = "native"

    // Kept in the persisted schema for backward-compatible decoding only.
    // Native export uses nativeParallelWorkers instead of these legacy fields.
    parallelWorkers = 1
    executor = "process"
    nativeParallelWorkers = min(
      max(snapshot.nativeParallelWorkers ?? 1, 1),
      10
    )
    decoderBackend = ["avfoundationAsync", "avfoundationLegacy", "ffmpegSoftware"]
      .contains(snapshot.decoderBackend ?? "")
      ? snapshot.decoderBackend! : "avfoundationAsync"
    detailedDiagnostics = snapshot.detailedDiagnostics ?? false
    useSegmentCount = snapshot.useSegmentCount
    segmentCount = min(max(snapshot.segmentCount, 1), 128)
    segmentDuration = min(max(snapshot.segmentDuration, 10), 3600)
    noSplit = snapshot.noSplit ?? false
    mergeEncoder = "copy"
    deleteSegments = snapshot.deleteSegments
    keepTemp = snapshot.keepTemp
    forceSplit = snapshot.forceSplit

    // Native Core AI export has one supported execution contract. Preserve
    // these fields in the on-disk schema so older preferences still decode,
    // but never let stale values make a native export fail.
    device = "mps"
    fp16 = true
    autoOptimize = true

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
    maxClipLength = min(max(snapshot.maxClipLength, 1), miohMaximumClipFrames)
    useRestoreMaxFrames = snapshot.useRestoreMaxFrames
    restoreMaxFrames = snapshot.restoreMaxFrames > 0
      ? min(snapshot.restoreMaxFrames, miohMaximumClipFrames)
      : snapshot.restoreMaxFrames
    restoreTemporalOverlap = min(max(snapshot.restoreTemporalOverlap ?? 8, 0), 120)
    restoreCrossfade = snapshot.restoreCrossfade ?? true
    sharpenStrength = min(max(snapshot.sharpenStrength, 0), 5)
    detailBoost = min(max(snapshot.detailBoost, 0), 1)
    blendFeather = min(max(snapshot.blendFeather, 0), 3)
    restoreMaskOnly = snapshot.restoreMaskOnly ?? true
    multiMosaicMode = snapshot.multiMosaicMode == "labelField" ? "labelField" : "perDetection"
    textureMix = min(max(snapshot.textureMix, 0), 1)
    smoothStrength = min(max(snapshot.smoothStrength, 0), 1)
    effectUpscale = min(max(snapshot.effectUpscale, 1), 4)
    roiEnhancer = enhancerModels.contains(snapshot.roiEnhancer) ? snapshot.roiEnhancer : "none"
    roiEnhancerModel = snapshot.roiEnhancerModel
    roiEnhancerModelsByMethod = snapshot.roiEnhancerModelsByMethod ?? [:]
    roiEnhancerScale = min(max(snapshot.roiEnhancerScale, 1), 8)
    if roiEnhancer == "roi-distill" { roiEnhancerScale = 2 }
    roiEnhancerStrength = min(max(snapshot.roiEnhancerStrength, 0), 1)
    roiEnhancerTile = min(max(snapshot.roiEnhancerTile, 0), 1024)
    // Expert ROI is hidden from the GUI (its code stays), so a previously
    // saved "on" must not keep applying invisibly.
    roiExpertMode = false
    swiftVRTemporalFilter = min(max(snapshot.swiftVRTemporalFilter ?? 1, 0), 1)
    swiftVRStabilizationRadius = min(max(snapshot.swiftVRStabilizationRadius ?? 1, 1), 8)
    swiftVRFrameSmoothing = min(max(snapshot.swiftVRFrameSmoothing ?? 15, 0), 30)
    roiRefineDistill = snapshot.roiRefineDistill ?? true
    let savedWindowSide = snapshot.roiRefineWindowSide
      ?? ((snapshot.roiRefineUse1024Window ?? false) ? 1024 : 512)
    roiRefineWindowSide = savedWindowSide == 256 ? 384
      : ([384, 512, 1024].contains(savedWindowSide) ? savedWindowSide : 512)
    let savedContinuation = snapshot.roiRefineContinuationFrames ?? 9
    roiRefineContinuationFrames = savedContinuation <= 1 ? 1 : 9
    roiRefineSteps = min(max(snapshot.roiRefineSteps ?? 2, 1), 8)
    roiRefineFastMode = snapshot.roiRefineFastMode ?? false

    detectionModel = detectionModels.contains(snapshot.detectionModel) ? snapshot.detectionModel : "v2-coreml"
    customDetectionModel = snapshot.customDetectionModel
    detectionEmptyLookahead = min(max(snapshot.detectionEmptyLookahead, 0), 300)
    detectionMaskReuseSkipFrames = min(
      max(snapshot.detectionMaskReuseSkipFrames ?? 0, 0),
      8
    )
    detectFaceMosaics = snapshot.detectFaceMosaics
    exportDetectionMasks = snapshot.exportDetectionMasks ?? false

    previewBufferLimit = min(max(snapshot.previewBufferLimit, 1), 60)
    previewRestorationModel = previewRestorationModels.contains(
      snapshot.previewRestorationModel ?? ""
    ) ? snapshot.previewRestorationModel! : capabilities.previewRestorationModel
    previewCustomRestorationModel = snapshot.previewCustomRestorationModel ?? ""
    previewDetectionModel = previewDetectionModels.contains(
      snapshot.previewDetectionModel ?? ""
    ) ? snapshot.previewDetectionModel! : capabilities.previewDetectionModel
    previewCustomDetectionModel = snapshot.previewCustomDetectionModel ?? ""
    previewRealtimeOptimization = snapshot.previewRealtimeOptimization ?? true
    previewLimitHighFrameRate = snapshot.previewLimitHighFrameRate ?? false
    previewUseSafariCompatibleHLS =
      snapshot.previewUseSafariCompatibleHLS ?? false
    previewHLSQuality = PreviewHLSQuality(
      rawValue: snapshot.previewHLSQuality ?? ""
    )?.rawValue ?? PreviewHLSQuality.automatic.rawValue
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
    generation: Int,
    decodeEndNanoseconds: Int64? = nil,
    outputCoreStartNanoseconds: Int64? = nil,
    outputCoreEndNanoseconds: Int64? = nil
  ) throws -> NativePreviewInvocation {
    normalizeModelSelections()
    guard capabilities.supportsCoreAI else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ再生にはmacOS 27以降が必要です"
      )
    }
    guard previewRestorationModel != "カスタム" else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ再生はカスタム復元モデルに未対応です"
      )
    }
    let selectedPreviewModel = previewRestorationModel
    try rejectUnsupportedCoreAIModel(selectedPreviewModel)
    let previewModel = selectedPreviewModel
    guard previewDetectionModel != "カスタム" else {
      throw RunnerError.unsupportedFeature(
        "Swiftネイティブ再生はカスタム検出モデルに未対応です"
      )
    }
    let selectedPreviewDetectionModel = previewDetectionModel
    try rejectUnsupportedCoreAIModel(selectedPreviewDetectionModel)
    let skipsCompositeParameters = previewRealtimeOptimization
    let effectiveEnhancerStrength = skipsCompositeParameters || roiEnhancer == "swiftvr" || roiEnhancer == "roi-refine"
      ? 0
      : roiEnhancerStrength
    let effectiveUpscale = skipsCompositeParameters ? 1 : effectUpscale
    let effectiveBlendFeather = blendFeather
    let effectiveSharpenStrength = skipsCompositeParameters ? 0 : sharpenStrength
    let effectiveDetailBoost = skipsCompositeParameters ? 0 : detailBoost
    let effectiveTextureMix = skipsCompositeParameters ? 0 : textureMix
    let effectiveSmoothStrength = skipsCompositeParameters ? 0 : smoothStrength
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
      model: selectedPreviewDetectionModel
    ) else {
      throw RunnerError.missingResource(
        "Swift native detection model: \(selectedPreviewDetectionModel)"
      )
    }
    let nativeEnhancer: (url: URL, scale: Int)?
    if roiEnhancer != "none", effectiveEnhancerStrength > 0 {
      guard let resolved = nativeROIEnhancerAsset(
        resources: resources,
        model: roiEnhancerModel,
        requestedScale: roiEnhancerScale
      ) else {
        throw RunnerError.missingResource(
          "Swift native ROI enhancer: \(roiEnhancerModel)"
        )
      }
      nativeEnhancer = resolved
    } else {
      nativeEnhancer = nil
    }
    let usesVariableTemporalModel = restoration.fixedFrameCount == nil
    let temporalLimit = usesVariableTemporalModel
      ? 48
      : (detection.computeUnits == nil ? 30 : 36)
    var requestedFrames = min(
      miohMaximumClipFrames,
      useMaxClipLength ? maxClipLength : temporalLimit
    )
    if useRestoreMaxFrames, restoreMaxFrames > 0 {
      requestedFrames = min(miohMaximumClipFrames, restoreMaxFrames)
    }
    let temporalFrames = min(
      restoration.fixedFrameCount ?? temporalLimit,
      min(temporalLimit, max(2, requestedFrames))
    )
    let previewOverlap = min(
      max(0, usesVariableTemporalModel ? 6 : restoreTemporalOverlap),
      max(0, temporalFrames - 1)
    )
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
      decoderBackend: decoderBackend,
      detailedDiagnostics: detailedDiagnostics,
      input: input.isFileURL ? input.path : input.absoluteString,
      outputDirectory: outputDirectory.path,
      ffmpegTemporaryDirectory: ffmpegTemporary,
      miohTemporaryDirectory: miohTemporary,
      ffmpeg: ffmpeg.path,
      detectionModel: detection.url.path,
      detectionBackend: detection.backend,
      detectionInputSize: detection.inputSize,
      detectionCandidateChannels: detection.candidateChannels,
      detectionQueries: detection.queries,
      detectionLogitClasses: detection.logitClasses,
      detectionMaxDet: detection.maxDetections,
      detectionComputeUnits: detection.computeUnits,
      restorationModels: restoration.url.path,
      restorationRunner: restorationRunner.path,
      restorationFrameCount: restoration.fixedFrameCount,
      startNanoseconds: startNanoseconds,
      decodeEndNanoseconds: decodeEndNanoseconds,
      outputCoreStartNanoseconds: outputCoreStartNanoseconds,
      outputCoreEndNanoseconds: outputCoreEndNanoseconds,
      generation: generation,
      bufferLimitSeconds: previewBufferLimit,
      temporalBatchFrames: temporalFrames,
      temporalOverlap: previewOverlap,
      ringCapacity: max(temporalFrames * 2, 24),
      confidenceThreshold: detection.confidenceThreshold,
      blendFeather: Float(effectiveBlendFeather),
      restoreMaskOnly: restoreMaskOnly,
      multiMosaicMode: multiMosaicMode == "labelField" ? "labelField" : nil,
      sharpenStrength: Float(effectiveSharpenStrength),
      detailBoost: Float(effectiveDetailBoost),
      textureMix: Float(effectiveTextureMix),
      smoothStrength: Float(effectiveSmoothStrength),
      effectUpscale: effectiveUpscale,
      roiEnhancerModel: nativeEnhancer?.url.path,
      roiEnhancerStrength: nativeEnhancer == nil ? 0 : Float(effectiveEnhancerStrength),
      roiEnhancerScale: nativeEnhancer?.scale ?? max(1, roiEnhancerScale),
      roiExpertMode: roiExpertMode,
      detectionEmptyLookahead: max(0, detectionEmptyLookahead),
      detectionMaskReuseSkipFrames: min(
        max(detectionMaskReuseSkipFrames, 0),
        8
      ),
      detectFaceMosaics: detectFaceMosaics,
      crossfade: restoreCrossfade,
      maximumFPS: previewLimitHighFrameRate ? 30 : nil,
      preFPSConversion: true
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
    if !previewRestorationModels.contains(previewRestorationModel) {
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
      case "diagnostic":
        appendLog("診断: \(line)\n")
        return
      case "export_finalizing":
        let message = payload["message"] as? String ?? "音声を結合しています"
        status = message
        appendLog("\n\(message)...\n")
        return
      case "mask_export_saved":
        let count = payload["frame_count"] as? Int ?? 0
        let path = payload["path"] as? String ?? ""
        appendLog("検出マスクJSON保存完了: \(count)フレーム\n\(detailedDiagnostics ? "保存先は出力動画の隣です。" : path)\n")
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
        let detectionSeconds = payload["detection_seconds"] as? Double ?? 0
        let preparationSeconds = payload["preparation_seconds"] as? Double ?? 0
        let restorationSeconds = payload["restoration_seconds"] as? Double ?? 0
        let compositionSeconds = payload["composition_seconds"] as? Double ?? 0
        let batchLanes = payload["native_parallel_workers"] as? Int ?? 1
        func elapsedPercent(_ seconds: Double) -> Double {
          elapsed > 0 ? seconds / elapsed * 100 : 0
        }
        appendLog(
          String(
            format:
              "処理統計: %dフレーム / 検出対象%dフレーム / 復元%dクリップ / %.1ffps / 経過 %@\n並列構成: 時間バッチ%dレーン\n処理内訳（工程は並行するため割合の合計は100%%になりません）: 検出 %.2f秒 (%.1f%%) / 準備 %.2f秒 (%.1f%%) / 復元 %.2f秒 (%.1f%%) / 合成 %.2f秒 (%.1f%%)\n",
            frames,
            detected,
            batches,
            fps,
            nativeLogDuration(elapsed),
            batchLanes,
            detectionSeconds,
            elapsedPercent(detectionSeconds),
            preparationSeconds,
            elapsedPercent(preparationSeconds),
            restorationSeconds,
            elapsedPercent(restorationSeconds),
            compositionSeconds,
            elapsedPercent(compositionSeconds)
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
    let unitProgress = min(max(percent / 100, 0), 1)
    if nativeExportBatchActive, nativeExportBatchTotal > 1 {
      progress = min(
        1,
        (Double(nativeExportBatchCompleted) + unitProgress)
          / Double(nativeExportBatchTotal)
      )
      status = "処理中 \(nativeExportBatchCompleted + 1)/\(nativeExportBatchTotal) \(Int(percent))%"
    } else {
      progress = unitProgress
      status = "処理中 \(Int(percent))%"
    }
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
  var actionLabel: String?

  var body: some View {
    HStack(spacing: 12) {
      Image(systemName: icon).frame(width: 20).foregroundStyle(.secondary)
      VStack(alignment: .leading, spacing: 3) {
        Text(L(title)).font(.caption).foregroundStyle(.secondary)
        Text(url.map { $0.isFileURL ? $0.path : $0.absoluteString } ?? L("未選択"))
          .lineLimit(1).truncationMode(.middle)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      if let actionLabel {
        Button(L(actionLabel), action: action)
          .buttonStyle(.bordered)
      } else {
        Button(action: action) { Image(systemName: "folder") }
          .buttonStyle(.borderless).help(L("選択"))
      }
    }
    .frame(minHeight: 42)
  }
}

/// The input file's own numbers, shown next to the path that produced them.
struct SourceInfoRow: View {
  let info: SourceMediaInfo?
  let failure: String?
  let batchSummary: String?

  var body: some View {
    HStack(alignment: .top, spacing: 12) {
      Image(systemName: "info.circle")
        .frame(width: 20)
        .foregroundStyle(.secondary)
      if let batchSummary {
        Text(batchSummary)
          .font(.caption)
          .foregroundStyle(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
      } else if let info {
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
        .frame(maxWidth: .infinity, alignment: .leading)
      } else if let failure {
        Text("入力情報を読み取れません: \(failure)")
          .font(.caption)
          .foregroundStyle(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
      } else {
        HStack(spacing: 8) {
          ProgressView().controlSize(.small)
          Text("入力情報を読み取り中…").font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
      }
    }
    .frame(minHeight: 42)
  }

  private func field(_ title: String, _ value: String) -> some View {
    VStack(alignment: .leading, spacing: 1) {
      Text(L(title)).font(.caption2).foregroundStyle(.secondary)
      Text(value).font(.caption).monospacedDigit()
    }
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


private enum WorkspaceTab: Hashable {
  case basic
  case browser
  case processing
  case restoration
  case detection
  case encoding
  case memory
  case settings
  case playback
  case log
}

private struct MovableWindowConfigurator: NSViewRepresentable {
  func makeNSView(context: Context) -> NSView {
    let view = MovableWindowAttachmentView()
    view.isHidden = true
    return view
  }

  func updateNSView(_ nsView: NSView, context: Context) {
    configure(nsView.window)
  }

  private func configure(_ window: NSWindow?) {
    window?.isMovable = true
    window?.isMovableByWindowBackground = true
  }

  private final class MovableWindowAttachmentView: NSView {
    override func viewDidMoveToWindow() {
      super.viewDidMoveToWindow()
      window?.isMovable = true
      window?.isMovableByWindowBackground = true
    }
  }
}

private struct WindowDragRegion: NSViewRepresentable {
  func makeNSView(context: Context) -> NSView {
    DragView()
  }

  func updateNSView(_ nsView: NSView, context: Context) {}

  private final class DragView: NSView {
    override var mouseDownCanMoveWindow: Bool { true }

    override func mouseDown(with event: NSEvent) {
      guard let window else {
        super.mouseDown(with: event)
        return
      }
      window.isMovable = true
      window.performDrag(with: event)
    }
  }
}

/// App-wide state. The window used to own these objects, so closing it
/// discarded the settings, the player, a running export and the remote
/// server, and reopening started from scratch. Owned here, they outlive the
/// window, and a reopened window shows them as they were.
@MainActor
private final class MiohAppState: ObservableObject {
  static let shared = MiohAppState()

  let runner = RestorationRunner()
  let player = RealtimePlayerController()
  let mediaBrowser = MacMediaBrowserController()
  let remoteControl = RemoteControlServer()
  let remoteStreaming = RemoteStreamingCoordinator()
  let cluster = MiohClusterController()
  @Published var selectedTab: WorkspaceTab = .basic
  private var subscriptions: Set<AnyCancellable> = []

  private init() {
    // Wired once here; the window's onAppear ran this on every open.
    cluster.attach(runner: runner)
    remoteControl.attach(runner: runner, player: player)
    remoteControl.attachCluster(cluster)
    remoteControl.attachStreaming(remoteStreaming)
    remoteControl.activateIfRemembered()
    if remoteControl.enabled {
      player.setStreamingEventConsumer(remoteStreaming.eventConsumer())
    }
    // @Published publishes before the new value is stored; hop to the next
    // main-loop turn so the handlers see it, as onChange did.
    cluster.$role.dropFirst().removeDuplicates()
      .receive(on: RunLoop.main)
      .sink { [weak self] _ in self?.cluster.deactivate(preserveRole: true) }
      .store(in: &subscriptions)
    remoteControl.$enabled.dropFirst().removeDuplicates()
      .receive(on: RunLoop.main)
      .sink { [weak self] enabled in
        guard let self else { return }
        if enabled {
          player.setStreamingEventConsumer(remoteStreaming.eventConsumer())
        } else {
          player.setStreamingEventConsumer(nil)
          remoteStreaming.stop()
        }
      }
      .store(in: &subscriptions)
  }
}

struct ContentView: View {
  @ObservedObject private var appState = MiohAppState.shared
  @ObservedObject private var runner = MiohAppState.shared.runner
  @ObservedObject private var player = MiohAppState.shared.player
  @ObservedObject private var mediaBrowser = MiohAppState.shared.mediaBrowser
  @ObservedObject private var remoteControl = MiohAppState.shared.remoteControl
  @ObservedObject private var remoteStreaming = MiohAppState.shared.remoteStreaming
  @ObservedObject private var cluster = MiohAppState.shared.cluster
  @State private var replaceOutputCandidate: URL?

  var body: some View {
    VStack(spacing: 0) {
      header
      Divider()
      TabView(selection: $appState.selectedTab) {
        basicTab
          .tabItem { Label("基本", systemImage: "slider.horizontal.3") }
          .tag(WorkspaceTab.basic)
        MacMediaBrowserView(
          controller: mediaBrowser,
          player: player,
          runner: runner,
          selectPlayback: { appState.selectedTab = .playback }
        )
          .tabItem { Label("ブラウザ", systemImage: "globe") }
          .tag(WorkspaceTab.browser)
        processingTab
          .tabItem { Label("分割", systemImage: "square.split.2x1") }
          .tag(WorkspaceTab.processing)
        restorationTab
          .tabItem { Label("復元", systemImage: "wand.and.stars") }
          .tag(WorkspaceTab.restoration)
        detectionTab
          .tabItem { Label("検出", systemImage: "viewfinder") }
          .tag(WorkspaceTab.detection)
        encodingTab
          .tabItem { Label("出力", systemImage: "video") }
          .tag(WorkspaceTab.encoding)
        memoryTab
          .tabItem { Label("メモリ", systemImage: "memorychip") }
          .tag(WorkspaceTab.memory)
        settingsTab.tabItem { Label("設定", systemImage: "gearshape") }
          .tag(WorkspaceTab.settings)
        RealtimePlayerView(controller: player, runner: runner)
          .tabItem { Label("再生", systemImage: "play.rectangle") }
          .tag(WorkspaceTab.playback)
        logTab
          .tabItem { Label("ログ", systemImage: "terminal") }
          .tag(WorkspaceTab.log)
      }
      .padding(.horizontal, 14)
      ProgressView(value: cluster.isRunning ? cluster.progress : runner.progress)
        .progressViewStyle(.linear).padding(.horizontal, 20)
      Divider()
      footer
    }
    .frame(minWidth: 820, minHeight: 680)
    .background(MovableWindowConfigurator().frame(width: 0, height: 0))
    .onAppear {
      InputPanelThumbnailCache.shared.prepare(initialURL: runner.inputURL)
    }
  }

  private func chooseInput() {
    let panel = NSOpenPanel()
    let thumbnailCache = InputPanelThumbnailCache.shared
    // Never compete with the standard panel's own Quick Look work.
    thumbnailCache.pause()
    panel.title = "入力ファイルまたはフォルダを選択"
    panel.allowedContentTypes = []
    panel.canChooseFiles = true
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    if let inputURL = runner.inputURL {
      panel.directoryURL = inputURL.hasDirectoryPath
        ? inputURL : inputURL.deletingLastPathComponent()
    }
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      Task { @MainActor in
        runner.inputURL = url.standardizedFileURL
        // Warm the selected folder for the next dialog, after this one closes.
        thumbnailCache.prepare(initialURL: url)
      }
    }
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
      let visibleStatus = cluster.isRunning || cluster.serviceActive
        ? cluster.status : runner.status
      Text(L(visibleStatus)).font(.callout.monospacedDigit())
        .foregroundStyle(visibleStatus.contains("失敗") || visibleStatus == "エラー" ? .red : .secondary)
    }
    .padding(.horizontal, 20).frame(height: 66)
    .background(WindowDragRegion())
  }

  private var basicTab: some View {
    Form {
      Section("ファイル") {
        PathRow(
          title: "入力",
          icon: "film",
          url: runner.inputURL,
          action: chooseInput,
          actionLabel: "入力を選択…"
        )
        if runner.inputURL != nil {
          SourceInfoRow(
            info: runner.sourceInfo,
            failure: runner.sourceInfoFailure,
            batchSummary: runner.sourceBatchSummary
          )
        }
        if let input = runner.inputURL, runner.sourceInfo != nil {
          DisclosureGroup("処理範囲", isExpanded: .constant(true)) {
            MiohVideoRangeSelector(
              url: input,
              duration: runner.sourceDurationSeconds,
              start: runner.normalizedRangeStartSeconds,
              end: runner.normalizedRangeEndSeconds,
              isDisabled: runner.isRunning || cluster.isRunning,
              setStart: runner.setRangeStartSeconds,
              setDuration: runner.setSelectedRangeDurationSeconds,
              setEnd: runner.setRangeEndSeconds,
              selectFull: runner.selectFullVideoRange
            )
            .padding(.top, 8)
          }
        }
        PathSettingRow(title: "出力", value: $runner.outputPathText, action: runner.chooseOutput)
        if runner.existingOutputFile != nil {
          Label(
            runner.overwrite
              ? L("同じ名前のファイルがあります。上書きされます")
              : L("同じ名前のファイルがあります。開始時に置き換えるか確認します"),
            systemImage: "exclamationmark.triangle.fill"
          )
          .font(.caption)
          .foregroundStyle(.orange)
        }
        PathSettingRow(title: "一時フォルダ", value: $runner.tempDirectory) { runner.choosePath(\.tempDirectory) }
        PathSettingRow(title: "FFmpeg一時フォルダ", value: $runner.ffmpegTempDirectory) { runner.choosePath(\.ffmpegTempDirectory) }
        PathSettingRow(title: "mioh一時フォルダ", value: $runner.ladaTempDirectory) { runner.choosePath(\.ladaTempDirectory) }
      }
      Section("実行") {
        LabeledContent("実行エンジン") {
          Text("Swiftネイティブ / Core AI")
        }
        Text("デコードから書き出しまでをSwiftネイティブ経路で実行します")
          .font(.caption)
          .foregroundStyle(.secondary)
        LabeledContent("精度・最適化") {
          Text("FP16 / Apple Silicon自動最適化")
        }
        Toggle("既存結果を上書き", isOn: $runner.overwrite)
      }
    }.formStyle(.grouped)
  }

  private var processingTab: some View {
    Form {
      Section("読み込み・診断") {
        Picker("動画の読み込み方式", selection: $runner.decoderBackend) {
          Text("AVFoundation 非同期（既定）").tag("avfoundationAsync")
          Text("AVFoundation 同期（比較用）").tag("avfoundationLegacy")
          Text("FFmpeg ソフトウェア（比較用）").tag("ffmpegSoftware")
        }.pickerStyle(.menu).disabled(runner.isRunning)
        Toggle("詳細な診断ログ（5秒間隔）", isOn: $runner.detailedDiagnostics)
          .disabled(runner.isRunning)
        Text("macOS 27以降のSwiftネイティブ書き出し・プレビューに適用。復元モデルは変えず、読み込みだけを比較します。FFmpeg方式はCPU負荷が増える場合があります。")
          .font(.caption).foregroundStyle(.secondary)
        Text("診断には区間fps、各工程の時間、バッファ待ち、メモリ・スワップ量を記録します。素材名・元動画パス・FFmpegの生ログは記録しません。")
          .font(.caption).foregroundStyle(.secondary)
      }
      Section("並列処理") {
        LabeledContent("実行方式") {
          Text("Swiftネイティブ（段階並列）")
        }
        Picker("ネイティブ並列数", selection: $runner.nativeParallelWorkers) {
          ForEach(1...10, id: \.self) { lane in
            Text("\(lane)レーン").tag(lane)
          }
        }
        .pickerStyle(.menu)
        Text(runner.roiEnhancer == "roi-refine"
          ? "ROI Refineは1つのLTXサーバーを使うため、書き出し時は自動的に1レーンになります"
          : "連続する時間バッチを\(runner.nativeParallelWorkers)レーンで処理します")
          .font(.caption)
          .foregroundStyle(.secondary)
        if runner.nativeParallelWorkers >= 4 {
          Text("4〜10レーンでは復元モデルと中間フレームのメモリ使用量が増えます。スワップが増える場合は並列数を下げてください")
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
        Toggle("処理済みセグメントを削除", isOn: $runner.deleteSegments)
        Toggle("一時ファイルを保持", isOn: $runner.keepTemp)
        Toggle("中断済みセグメントから再開", isOn: $runner.resumeInterruptedExport)
        Text("同じ入力・出力・設定の完成済みセグメントだけを再利用します。不完全な .part は無視します。")
          .font(.caption)
          .foregroundStyle(.secondary)
        Toggle("強制的に再分割", isOn: $runner.forceSplit)
          .disabled(runner.noSplit)
      }
    }.formStyle(.grouped)
  }

  private var restorationTab: some View {
    Form {
      Section("モデル") {
        Picker("復元モデル", selection: $runner.restorationModel) {
          ForEach(runner.restorationModels, id: \.self) {
            Text(restorationModelLabel($0)).tag($0)
          }
        }
        Toggle("最大クリップ長を指定", isOn: $runner.useMaxClipLength)
        if runner.useMaxClipLength {
          LabeledContent("最大クリップ長") { Stepper(value: $runner.maxClipLength, in: 1...miohMaximumClipFrames) { Text("\(runner.maxClipLength)") } }
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
        doubleSliderField("シャープ", value: $runner.sharpenStrength, range: 0...5, step: 0.05)
        doubleSliderField("ディテール", value: $runner.detailBoost, range: 0...1, step: 0.05)
        doubleSliderField("境界フェザー", value: $runner.blendFeather, range: 0...3, step: 0.05)
        Toggle("モザイク部分だけ貼り戻す", isOn: $runner.restoreMaskOnly)
        Text("オンにすると、復元結果を検出したモザイクの範囲（と少しの余白）だけに貼り戻し、切り出し枠に入った顔などモザイクのない部分は元の映像のままにします。オフは従来どおり切り出し枠全体を貼り戻します。")
          .font(.caption)
          .foregroundStyle(.secondary)
        doubleSliderField("テクスチャ", value: $runner.textureMix, range: 0...1, step: 0.01)
        doubleSliderField("スムージング", value: $runner.smoothStrength, range: 0...1, step: 0.05)
        LabeledContent("エフェクト倍率") { Stepper(value: $runner.effectUpscale, in: 1...4) { Text("\(runner.effectUpscale)x") } }
      }
      Section("ROIエンハンサー") {
        Picker("方式", selection: Binding(
          get: { runner.roiEnhancer },
          set: { runner.selectROIEnhancer($0) }
        )) {
          ForEach(runner.enhancerModels, id: \.self) { Text(RestorationRunner.enhancerDisplayName($0)).tag($0) }
        }
        if runner.roiEnhancer != "none" {
          if runner.roiEnhancer == "roi-distill" {
            Text("実験用のSwiftVR蒸留モデル。復元済みの同じROIの前後2フレームを使い、256pxから512pxへ補強します。5フレーム入力のCore MLモデルを選択してください。既存SwiftVRとは別処理で、画質改善は未承認です。")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
          if runner.roiEnhancer == "swiftvr" {
            Text("書き出し専用。BasicVSR++で復元した各シーンをその場でSwiftVRにかけて合成します（1シーン数秒〜十数秒）。モデルフォルダを選択してください。")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
          if runner.roiEnhancer == "roi-refine" {
            Text("書き出し専用。BasicVSR++で復元した各シーンのROIを、LTX-2.3（Sulphur 2）のIC-LoRAで512pxに描き直して合成します。33フレームずつ、モザイクの範囲だけを描き直します（33フレームで約17秒、メモリ約15GB）。描かれるのはそれらしい細部で、元の細部の再現ではありません。描き直しパックのフォルダを選択してください。")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
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
          if runner.roiEnhancerModel.hasPrefix("/") {
            Text("モデルの場所: \(runner.roiEnhancerModel)")
              .font(.caption)
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
              .lineLimit(2)
              .truncationMode(.middle)
          }
        }
        if runner.roiEnhancer == "swiftvr" {
          Picker("倍率", selection: $runner.roiEnhancerScale) {
            Text("2x").tag(2)
            Text("4x").tag(4)
          }
          .pickerStyle(.segmented)
          Text("ROIの大きさに関係なく、すべてのシーンをこの倍率で処理します。2xは出力512pxで、4x（1024px）より大幅に速くなります。")
            .font(.caption)
            .foregroundStyle(.secondary)
          integerSliderField("なじませ範囲", value: $runner.swiftVRStabilizationRadius, range: 1...8, step: 1)
          Text("SwiftVRの結果を前後何フレームまで混ぜるか。BasicVSR++の結果が似ている所だけを混ぜるので、広げても動いた所は混ざらず、止まった場面ほどチラつきが減ります。1が従来どおり。")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        if runner.roiEnhancer != "none" {
          doubleSliderField("揺らぎ低減", value: $runner.swiftVRTemporalFilter, range: 0...1, step: 0.05)
          Text("エンハンサーの合成結果に、前後のフレームを使うAppleの時間方向ノイズ除去をかけて、質感のチラつきを抑えます。検出マスクの内側だけに適用します。0でオフ。強くすると細かいざらつきも減ります。")
            .font(.caption)
            .foregroundStyle(.secondary)
          if runner.roiEnhancer == "roi-refine" {
            Toggle("高速モード（384px・引き継ぎ1）", isOn: $runner.roiRefineFastMode)
            Text("内部計算窓を最大384pxにし、33フレーム窓の引き継ぎを9から1フレームへ減らします。ROI Refineは1レーンで実行します。512px高品質モードより約2倍の高速化を狙う設定です。")
              .font(.caption)
              .foregroundStyle(.secondary)
            DisclosureGroup("詳細設定") {
              Picker("内部計算の最大サイズ", selection: $runner.roiRefineWindowSide) {
                Text("384px").tag(384)
                Text("512px").tag(512)
                Text("1024px").tag(1024)
              }
              .pickerStyle(.segmented)
              Picker("窓間の引き継ぎ", selection: $runner.roiRefineContinuationFrames) {
                Text("1フレーム").tag(1)
                Text("9フレーム").tag(9)
              }
              .pickerStyle(.segmented)
            }
            .disabled(runner.roiRefineFastMode)
            Text(runner.roiRefineFastMode
              ? "高速プリセットが詳細設定より優先されます。個別調整する場合は高速モードをオフにしてください。"
              : "内部サイズと窓間の引き継ぎを個別に指定します。1024pxは実測で約6倍遅く、現在のLoRAでは細部差が小さい設定です。")
              .font(.caption)
              .foregroundStyle(.secondary)
            Picker("ステップ数", selection: $runner.roiRefineSteps) {
              ForEach(1...8, id: \.self) { Text("\($0)").tag($0) }
            }
            .pickerStyle(.segmented)
            Text("33フレーム窓ごとのノイズ除去の回数。処理時間はほぼ回数に比例します。2が標準（3と同等の見た目で約3割速い）。4以上は強いノイズから描き始めるので、元の形から離れやすくなります。")
              .font(.caption)
              .foregroundStyle(.secondary)
            Toggle("大きいROIにROI Distillを重ねる", isOn: $runner.roiRefineDistill)
            Text("ROI Refineは512pxで描き直すので、それより大きいROI（元の切り出しが512px超）は解像度が足りません。ROI Refineの結果にROI Distillの細部を上乗せします（処理時間は約1割増え、細部は約3割増えます）。")
              .font(.caption)
              .foregroundStyle(.secondary)
            integerSliderField("なじませ範囲", value: $runner.swiftVRStabilizationRadius, range: 1...8, step: 1)
            Text("ROI Refineの結果を前後何フレームまで混ぜるか。BasicVSR++の結果が似ている所だけを混ぜるので、動いた所は混ざらず、止まった場面ほど細部の揺らぎが減ります。1でオフ。")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
          if runner.roiEnhancerSupportsFrameSmoothing {
            integerSliderField("枠のなめらかさ", value: $runner.swiftVRFrameSmoothing, range: 0...30, step: 1)
            Text("エンハンサーに渡す切り出し枠の位置と大きさを、前後のフレームで平均してなめらかにします。結果は元の枠へ戻して貼り付けます。BasicVSR++の復元と検出マスクは変わりません。0でオフ。")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }
        doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05).disabled(runner.roiEnhancer == "none")
      }
    }.formStyle(.grouped)
  }


  private var detectionTab: some View {
    Form {
      Section("検出モデル") {
        Picker("モデル", selection: $runner.detectionModel) {
          ForEach(runner.detectionModels, id: \.self) { Text(L($0)).tag($0) }
        }
        LabeledContent("無検出時の判定間隔") {
          Stepper(value: $runner.detectionEmptyLookahead, in: 0...300) { Text("\(runner.detectionEmptyLookahead)") }
        }
        LabeledContent("検出後のマスク再利用") {
          Stepper(
            value: $runner.detectionMaskReuseSkipFrames,
            in: 0...8
          ) {
            Text("\(runner.detectionMaskReuseSkipFrames)フレーム")
          }
        }
        Text("無検出時の先頭・末尾判定は常に先に実行します。検出がある範囲だけ、1〜8では指定した後続フレームへ直前の領域とマスクを再利用し、保存とリアルタイム再生の両方を高速化します。0は従来どおりです。")
          .font(.caption)
          .foregroundStyle(.secondary)
        Toggle("顔モザイクを検出", isOn: $runner.detectFaceMosaics)
        Toggle("検出マスクをJSONに保存（学習用）", isOn: $runner.exportDetectionMasks)
          .disabled(runner.isRunning)
        Text("書き出し動画の隣に .masks.json を保存します。各フレームの時刻・検出枠・画素単位の検出マスクと、CELF有効時は補正後の持ち主・範囲・重みも記録します。保存とマスク展開に追加の時間・容量が必要です。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Section("複数モザイク") {
        Picker("複数モザイクの扱い", selection: $runner.multiMosaicMode) {
          Text("検出ごと（従来）").tag("perDetection")
          Text("画素ごとの持ち主（CELF）").tag("labelField")
        }
        Text("CELFは、重なったモザイクの画素ひとつひとつに持ち主を決めて1つずつ復元します。検出の枠がまとまったり分かれたりしても、モザイクごとの範囲と縮尺が変わらず、同じ場所を二重に貼りません。")
          .font(.caption)
          .foregroundStyle(.secondary)
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
        Text("\(runner.targetFPSDescription)fps\(runner.targetFPSDetail)で書き出します。元動画より高いレートは指定できません")
          .font(.caption)
          .foregroundStyle(.secondary)
        Toggle("復元前にFPS変換", isOn: $runner.preFPSConversion)
          .disabled(!runner.useFPS)
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
      Section("HLS再生") {
        Picker(
          "HLS通信",
          selection: $runner.previewUseSafariCompatibleHLS
        ) {
          Text("高速（区間先読み）").tag(false)
          Text("Safari互換（429回避）").tag(true)
        }
        .pickerStyle(.segmented)
        Text(
          runner.previewUseSafariCompatibleHLS
            ? "AVFoundationの映像を取り込むため低速ですが、HTTP 429を回避しやすい方式です。"
            : "HLS区間を並行して先読みする高速方式です。配信元によってはHTTP 429になる場合があります。"
        )
        .font(.caption)
        .foregroundStyle(.secondary)
        Picker("HLS画質", selection: $runner.previewHLSQuality) {
          ForEach(PreviewHLSQuality.allCases) { quality in
            Text(quality.label).tag(quality.rawValue)
          }
        }
        .pickerStyle(.menu)
        Text(
          runner.previewHLSQuality == PreviewHLSQuality.automatic.rawValue
            ? "自動では配信状況に応じて画質を選びます。固定時は高速・Safari互換の両方で指定variantを使用します。"
            : "指定画質以下で最も高いvariantを固定使用し、自動的な低画質への切り替えを行いません。"
        )
        .font(.caption)
        .foregroundStyle(.secondary)
        Text("変更は次回のHLS復元再生から適用されます。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Section("ローカルネットワーク操作") {
        Toggle(
          "同じLANからmiohを操作する",
          isOn: Binding(
            get: { remoteControl.enabled },
            set: { remoteControl.setEnabled($0) }
          )
        )
        LabeledContent("状態") {
          Text(remoteControl.status)
            .foregroundStyle(
              remoteControl.enabled && !remoteControl.urls.isEmpty
                ? .green : .secondary
            )
        }
        if remoteControl.enabled {
          if remoteControl.urls.isEmpty {
            Text("同じLANから接続できるアドレスを確認中です。")
              .font(.caption)
              .foregroundStyle(.secondary)
          } else if let url = remoteControl.urls.first {
            LabeledContent("接続") {
              HStack(spacing: 8) {
                Button("Webリモコンを開く") { NSWorkspace.shared.open(url) }
                  .buttonStyle(.borderedProminent)
                Button("接続情報をコピー") {
                  NSPasteboard.general.clearContents()
                  NSPasteboard.general.setString(
                    "\(url.absoluteString)\n\(remoteControl.token)",
                    forType: .string
                  )
                }
              }
            }
          }
          LabeledContent("初回アクセスコード") {
            HStack(spacing: 8) {
              Text(remoteControl.token)
                .font(.system(.caption, design: .monospaced))
                .lineLimit(1)
                .textSelection(.enabled)
              Button("コピー") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(remoteControl.token, forType: .string)
              }
            }
          }
          Text("mioh Remoteでは表示されたMacをタップし、このコードを初回だけ入力します。次回からは自動接続します。")
            .font(.caption)
            .foregroundStyle(.secondary)
          DisclosureGroup("詳細設定") {
            LabeledContent("ポート") {
              TextField("", value: $remoteControl.port, format: .number)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
                .disabled(remoteControl.enabled)
            }
            ForEach(remoteControl.urls, id: \.absoluteString) { url in
              LabeledContent("接続URL") {
                Text(url.absoluteString)
                  .font(.system(.caption, design: .monospaced))
                  .textSelection(.enabled)
              }
            }
            Button("アクセスコードを再生成", role: .destructive) {
              remoteControl.regenerateToken()
            }
          }
          Text("信頼できる家庭・社内LAN専用です。インターネットには公開しないでください。")
            .font(.caption)
            .foregroundStyle(.orange)
        } else {
          Text("一度有効にすると、このMacでは次回起動時から自動で待ち受けます。")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }
      Section("ローカル復元クラスタ") {
        Picker("役割", selection: $cluster.role) {
          ForEach(MiohClusterRoleSelection.allCases) { role in
            Text(role.label).tag(role)
          }
        }
        LabeledContent("共有ルート") {
          HStack(spacing: 8) {
            TextField("", text: $cluster.sharedRootPath)
              .textFieldStyle(.roundedBorder)
              .frame(width: 360)
            Button { cluster.chooseSharedRoot() } label: {
              Image(systemName: "folder")
            }
          }
        }
        LabeledContent("共有ルートID") {
          TextField("", text: $cluster.sharedRootIdentifier)
            .textFieldStyle(.roundedBorder)
            .frame(width: 220)
        }
        if cluster.role == .coordinator {
          LabeledContent("ジョブ長") {
            Stepper(value: $cluster.shardMinutes, in: 1...30) {
              Text("\(cluster.shardMinutes)分")
            }
          }
          Toggle("このMacもWorkerとして使用", isOn: $cluster.useCoordinatorAsWorker)
            .disabled(cluster.isRunning)
          Text("有効時はCoordinatorでも同じジョブ契約を1並列で処理します。")
            .font(.caption)
            .foregroundStyle(.secondary)
          Toggle("クラスタで書き出す", isOn: $cluster.useForExport)
        }
        HStack(spacing: 10) {
          if cluster.serviceActive {
            Button("クラスタサービスを停止", role: .destructive) {
              cluster.deactivate(preserveRole: true)
            }
          } else if cluster.role != .off {
            Button("クラスタサービスを開始") {
              cluster.activate(using: runner)
            }
            .buttonStyle(.borderedProminent)
          }
          Text(cluster.status)
            .font(.callout.monospacedDigit())
            .foregroundStyle(cluster.status.contains("失敗") ? .red : .secondary)
        }

        if cluster.role == .worker {
          Text("同じローカルLAN上のCoordinatorから、認証コードなしで利用できます。")
            .font(.caption)
            .foregroundStyle(.secondary)
          if !cluster.workerAttempts.isEmpty {
            ForEach(cluster.workerAttempts.suffix(4)) { attempt in
              LabeledContent(String(attempt.jobID.uuidString.prefix(8))) {
                Text(attempt.state.rawValue)
                  .font(.system(.caption, design: .monospaced))
              }
            }
          }
        }

        if cluster.role == .coordinator {
          if cluster.discoveredNodes.isEmpty {
            Text(cluster.serviceActive
              ? "同じLAN上のWorkerを探索しています。"
              : "サービスを開始するとWorkerを自動検出します。")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
          ForEach(cluster.discoveredNodes) { node in
            VStack(alignment: .leading, spacing: 7) {
              HStack {
                Toggle(
                  isOn: Binding(
                    get: { cluster.selectedNodeIDs.contains(node.id) },
                    set: { cluster.toggleSelection(node.id, selected: $0) }
                  )
                ) {
                  Text(node.metadata.displayName)
                }
                .disabled(!cluster.verifiedNodeIDs.contains(node.id))
                Spacer()
                Text(cluster.verifiedNodeIDs.contains(node.id) ? "接続済み" : "確認中")
                  .font(.caption)
                  .foregroundStyle(cluster.verifiedNodeIDs.contains(node.id) ? .green : .secondary)
              }
            }
          }
        }
        Text("標準HTTP転送では共有ルートは不要です。旧Workerとの共有フォルダ方式も併用できます。TLSはないため信頼できる家庭・社内LANだけで使い、インターネットへ公開しないでください。FPS変換は復元後に全Worker共通の時間軸で適用します。")
          .font(.caption)
          .foregroundStyle(.orange)
      }
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
        Text("入力/出力、一時フォルダ、分割、復元、検出、出力、メモリ、再生バッファ、HLS通信方式・画質まで保存します。ログ、進捗、実行中状態は保存しません。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }.formStyle(.grouped)
  }

  private var logTab: some View {
    ScrollView {
      Text(activeLogText.isEmpty ? " " : activeLogText)
        .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .topLeading).padding(12)
    }.background(Color(nsColor: .textBackgroundColor)).padding(.vertical, 10)
  }

  private var activeLogText: String {
    if cluster.isRunning { return cluster.log }
    if runner.isRunning { return runner.log }
    if runner.log.isEmpty, !cluster.log.isEmpty { return cluster.log }
    return runner.log
  }

  private var footer: some View {
    restorationFooter
  }


  private var restorationFooter: some View {
    HStack(spacing: 10) {
      Button(action: runner.revealOutput) { Image(systemName: "folder.badge.gearshape") }
        .help(L("出力をFinderで表示")).disabled(runner.outputURL == nil)
      Spacer()
      if cluster.isRunning {
        Button(role: .destructive, action: cluster.stopExport) {
          Label("停止", systemImage: "stop.fill")
        }
      } else if runner.isRunning {
        Button(role: .destructive, action: runner.stop) { Label("停止", systemImage: "stop.fill") }
      } else {
        Button(
          action: {
            if cluster.useForExport && cluster.role == .coordinator {
              cluster.startExport(using: runner)
            } else if !runner.overwrite, let existing = runner.existingOutputFile {
              replaceOutputCandidate = existing
            } else {
              runner.start()
            }
          }
        ) { Label("開始", systemImage: "play.fill") }
          .buttonStyle(.borderedProminent)
          .alert(
            L("同じ名前のファイルがあります"),
            isPresented: Binding(
              get: { replaceOutputCandidate != nil },
              set: { if !$0 { replaceOutputCandidate = nil } }
            ),
            presenting: replaceOutputCandidate
          ) { output in
            Button(L("置き換える"), role: .destructive) { runner.startReplacing(output) }
            Button(L("キャンセル"), role: .cancel) {}
          } message: { output in
            Text(String(format: L("「%@」は既に存在します。置き換えますか？"), output.lastPathComponent))
          }
          .disabled(
            cluster.useForExport && cluster.role == .coordinator
              ? !cluster.canStartExport(runner: runner)
              : (!runner.canStart
                || (cluster.role == .worker && cluster.serviceActive))
          )
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

#if !MIOH_REMOTE_SETTINGS_HARNESS
@main
#endif
struct MiohStandaloneApp: App {
  var body: some Scene {
    WindowGroup(id: MiohWindowCommands.mainWindowID) { ContentView() }
      .windowResizability(.contentMinSize)
      .defaultSize(width: 920, height: 760)
      // One window over the shared state; closing it keeps everything, and
      // a Dock click or ウインドウ > miohウインドウを表示 (⌘0) shows it again.
      .commands { MiohWindowCommands() }
  }
}

private struct MiohWindowCommands: Commands {
  static let mainWindowID = "main"
  @Environment(\.openWindow) private var openWindow

  var body: some Commands {
    CommandGroup(replacing: .newItem) {}
    CommandGroup(before: .windowArrangement) {
      Button("miohウインドウを表示") { showMainWindow() }
        .keyboardShortcut("0", modifiers: .command)
    }
  }

  /// Brings the existing window forward (restoring it from the Dock if
  /// minimized); opens it again only when it was closed, so there is never
  /// more than one.
  private func showMainWindow() {
    let existing = NSApp.windows.first {
      $0.identifier?.rawValue.hasPrefix(Self.mainWindowID) == true
        && ($0.isVisible || $0.isMiniaturized)
    }
    if let existing {
      if existing.isMiniaturized { existing.deminiaturize(nil) }
      existing.makeKeyAndOrderFront(nil)
    } else {
      openWindow(id: Self.mainWindowID)
    }
    NSApp.activate()
  }
}
