// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AVFoundation
import Accelerate
import CoreAI
import CoreImage
import CoreML
import CoreVideo
import Darwin
import Foundation

private let detectorSize = 640
private let prototypeSize = 160
private let restorationSize = 256
private let candidateCount = 8400
private let maximumTemporalBatchFrames = 180
private let maximumInternalExportSegmentSeconds = 60.0

private struct NativePreviewConfiguration: Decodable {
  let mode: String?
  let input: String
  let outputDirectory: String
  let ffmpegTemporaryDirectory: String?
  let miohTemporaryDirectory: String?
  let outputFile: String?
  let ffmpeg: String?
  let detectionModel: String
  let detectionBackend: String?
  let detectionInputSize: Int?
  let detectionCandidateChannels: Int
  let detectionQueries: Int?
  let detectionLogitClasses: Int?
  let detectionMaxDet: Int?
  let detectionComputeUnits: String?
  let restorationModels: String
  let restorationRunner: String
  let restorationFrameCount: Int?
  let startNanoseconds: Int64
  let generation: Int
  let splitMode: String?
  let segmentCount: Int?
  let segmentSeconds: Double
  let bufferLimitSeconds: Double
  let temporalBatchFrames: Int
  let temporalOverlap: Int?
  let ringCapacity: Int
  let confidenceThreshold: Float
  let iouThreshold: Float
  let contextFraction: Float
  let blendFeather: Float?
  let sharpenStrength: Float?
  let detailBoost: Float?
  let textureMix: Float?
  let smoothStrength: Float?
  let effectUpscale: Int?
  let roiEnhancerModel: String?
  let roiEnhancerStrength: Float?
  let roiEnhancerScale: Int?
  let detectionEmptyLookahead: Int?
  let detectFaceMosaics: Bool?
  let crossfade: Bool?
  // The exact target rate: 29.970fps arrives as 30000/1001, never as 30.
  // A configuration written before rational rates omits the denominator, and
  // the whole-number request is then resolved against the source timebase.
  let targetFPS: Int?
  let targetFPSDenominator: Int?
  let preFPSConversion: Bool?
  let videoCodec: String?
  let averageBitRate: Int?
  let bitrateMultiplier: Double?
  let mp4FastStart: Bool?

  var isExport: Bool { mode == "export" }
}

private enum NativePreviewError: LocalizedError {
  case invalidArguments
  case invalidConfiguration(String)
  case missingVideoTrack
  case reader(String)
  case pixelBuffer(String)
  case detector(String)
  case restorer(String)
  case export(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: mioh-native-coreai-preview <configuration.json>"
    case .invalidConfiguration(let message):
      return "invalid native preview configuration: \(message)"
    case .missingVideoTrack:
      return "input has no video track"
    case .reader(let message):
      return "native preview decoder failed: \(message)"
    case .pixelBuffer(let message):
      return "native preview pixel buffer failed: \(message)"
    case .detector(let message):
      return "native preview detector failed: \(message)"
    case .restorer(let message):
      return "native preview restorer failed: \(message)"
    case .export(let message):
      return "native export failed: \(message)"
    }
  }
}

private struct VideoDescription {
  let width: Int
  let height: Int
  let fpsNumerator: Int
  let fpsDenominator: Int
  let durationSeconds: Double
  let estimatedDataRate: Double
}

private struct DecodedFrame: @unchecked Sendable {
  let pixelBuffer: CVPixelBuffer
  let ptsNanoseconds: Int64
}

private struct ProcessedFrame: @unchecked Sendable {
  let pixelBuffer: CVPixelBuffer
  let ptsNanoseconds: Int64
}

/// Frame rates are never converted between the integer and NTSC families.
/// A 59.94fps source halved is 29.97fps, a 60fps source halved is 30fps, and
/// the request itself only ever names the whole number.
enum NTSCFrameRate {
  /// True when `numerator / denominator` is a whole rate divided by 1.001.
  static func isNTSC(numerator: Int, denominator: Int) -> Bool {
    guard numerator > 0, denominator > 0 else { return false }
    let pulled = Double(numerator) * 1.001 / Double(denominator)
    return abs(pulled - pulled.rounded()) < 0.001 && pulled.rounded() >= 1
  }

  static func target(
    wholeFPS: Int,
    sourceNumerator: Int,
    sourceDenominator: Int
  ) -> (numerator: Int, denominator: Int) {
    let whole = max(1, wholeFPS)
    guard isNTSC(numerator: sourceNumerator, denominator: sourceDenominator)
    else {
      return (whole, 1)
    }
    return (whole * 1000, 1001)
  }
}

/// Selects frames from a VFR or CFR stream using presentation timestamps.
/// Keeping the original PTS for accepted frames preserves audio sync, while
/// SegmentWriter emits them at the requested constant output frame rate.
/// The step is carried as an exact rational (whole nanoseconds plus a
/// remainder over the numerator) so an NTSC rate such as 30000/1001 does not
/// accumulate the ~0.33ns per frame that rounding 33.3667ms would.
private struct PTSFrameRateGate: Sendable {
  private let numerator: Int64
  private let intervalWhole: Int64
  private let intervalRemainder: Int64
  private var nextPTS: Int64?
  private var carry: Int64 = 0

  init(numerator: Int, denominator: Int) {
    let frames = Int64(max(1, numerator))
    let seconds = Int64(max(1, denominator))
    let step = seconds * 1_000_000_000
    self.numerator = frames
    intervalWhole = max(1, step / frames)
    intervalRemainder = step % frames
  }

  private mutating func advance(_ pts: Int64) -> Int64 {
    var next = pts + intervalWhole
    carry += intervalRemainder
    if carry >= numerator {
      carry -= numerator
      next += 1
    }
    return next
  }

  mutating func accepts(_ ptsNanoseconds: Int64) -> Bool {
    guard let nextPTS else {
      self.nextPTS = advance(ptsNanoseconds)
      return true
    }
    guard ptsNanoseconds >= nextPTS else { return false }
    var following = nextPTS
    repeat {
      following = advance(following)
    } while following <= ptsNanoseconds
    self.nextPTS = following
    return true
  }
}

/// A bounded producer/consumer ring. CMSampleBuffer-backed CVPixelBuffers are
/// retained by each slot and released as soon as the corresponding output
/// frame has been handed to VideoToolbox.
private final class PixelBufferRing: @unchecked Sendable {
  private let condition = NSCondition()
  private var slots: [DecodedFrame?]
  private var readIndex = 0
  private var writeIndex = 0
  private var count = 0
  private var finished = false
  private var stopped = false
  private var failure: Error?

  init(capacity: Int) {
    slots = Array(repeating: nil, count: max(2, capacity))
  }

  func push(_ frame: DecodedFrame) -> Bool {
    condition.lock()
    defer { condition.unlock() }
    while count == slots.count && !stopped {
      condition.wait()
    }
    guard !stopped else { return false }
    slots[writeIndex] = frame
    writeIndex = (writeIndex + 1) % slots.count
    count += 1
    condition.broadcast()
    return true
  }

  func pop() throws -> DecodedFrame? {
    condition.lock()
    defer { condition.unlock() }
    while count == 0 && !finished && !stopped {
      condition.wait()
    }
    if let failure {
      throw failure
    }
    guard count > 0 else { return nil }
    let frame = slots[readIndex]
    slots[readIndex] = nil
    readIndex = (readIndex + 1) % slots.count
    count -= 1
    condition.broadcast()
    return frame
  }

  func complete(_ error: Error? = nil) {
    condition.lock()
    failure = error
    finished = true
    condition.broadcast()
    condition.unlock()
  }

  func stop() {
    condition.lock()
    stopped = true
    slots = Array(repeating: nil, count: slots.count)
    count = 0
    condition.broadcast()
    condition.unlock()
  }
}

private final class ContinuousVideoDecoder: @unchecked Sendable {
  private let asset: AVURLAsset
  private let track: AVAssetTrack
  private let ring: PixelBufferRing
  private let startTime: CMTime
  private var reader: AVAssetReader?
  private var task: Task<Void, Never>?

  init(
    input: URL,
    startNanoseconds: Int64,
    ring: PixelBufferRing
  ) async throws {
    asset = AVURLAsset(url: input)
    let tracks = try await asset.loadTracks(withMediaType: .video)
    guard let track = tracks.first else {
      throw NativePreviewError.missingVideoTrack
    }
    self.track = track
    self.ring = ring
    startTime = CMTime(
      value: max(0, startNanoseconds),
      timescale: 1_000_000_000
    )
  }

  func description() async throws -> VideoDescription {
    let transform = try await track.load(.preferredTransform)
    let natural = try await track.load(.naturalSize).applying(transform)
    let frameRate = try await track.load(.nominalFrameRate)
    let duration = try await asset.load(.duration)
    let estimatedDataRate = try await track.load(.estimatedDataRate)
    // nominalFrameRate is the track average. minFrameDuration looks more
    // precise but reports the shortest observed gap, so a VFR clip comes back
    // well above its real rate; it is not usable here.
    let fps = max(1.0, Double(frameRate))
    let scale = 1000
    return VideoDescription(
      width: max(1, Int(abs(natural.width).rounded())),
      height: max(1, Int(abs(natural.height).rounded())),
      fpsNumerator: max(1, Int((fps * Double(scale)).rounded())),
      fpsDenominator: scale,
      durationSeconds: duration.seconds,
      estimatedDataRate: Double(estimatedDataRate)
    )
  }

  func start() throws {
    let reader = try AVAssetReader(asset: asset)
    if startTime > .zero {
      let duration = asset.duration - startTime
      reader.timeRange = CMTimeRange(
        start: startTime,
        duration: max(duration, .zero)
      )
    }
    let settings: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferMetalCompatibilityKey as String: true,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: settings
    )
    // The decoder-owned IOSurface remains valid while the ring slot retains
    // its CMSampleBuffer/CVPixelBuffer. Avoiding this copy is the main reason
    // for the native path.
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
      throw NativePreviewError.reader("cannot add video output")
    }
    reader.add(output)
    guard reader.startReading() else {
      throw NativePreviewError.reader(
        reader.error?.localizedDescription ?? "startReading failed"
      )
    }
    self.reader = reader
    task = Task.detached(priority: .userInitiated) { [ring] in
      while !Task.isCancelled {
        guard let sample = output.copyNextSampleBuffer() else { break }
        guard let image = CMSampleBufferGetImageBuffer(sample) else {
          continue
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        let ptsNanoseconds = Int64(
          (pts.seconds * 1_000_000_000).rounded()
        )
        if !ring.push(
          DecodedFrame(pixelBuffer: image, ptsNanoseconds: ptsNanoseconds)
        ) {
          break
        }
      }
      if reader.status == .failed {
        ring.complete(
          NativePreviewError.reader(
            reader.error?.localizedDescription ?? "decode failed"
          )
        )
      } else {
        ring.complete()
      }
    }
  }

  func stop() async {
    task?.cancel()
    reader?.cancelReading()
    ring.stop()
    await task?.value
    task = nil
    reader = nil
  }
}

private enum DetectionMaskProjection {
  case yoloLetterbox
  case directResize
}

private struct Detection {
  let left: Int
  let top: Int
  let right: Int
  let bottom: Int
  let confidence: Float
  let classIndex: Int
  /// YOLO stores a 640px letterboxed binary mask. RF-DETR stores the selected
  /// query's model-space mask logits and resizes them directly to the source.
  let detectorMask: [Float]
  let maskWidth: Int
  let maskHeight: Int
  let maskProjection: DetectionMaskProjection
  let maskThreshold: Float

  var area: Int {
    max(0, right - left + 1) * max(0, bottom - top + 1)
  }
}

private struct DetectionCandidate {
  let x: Float
  let y: Float
  let width: Float
  let height: Float
  let confidence: Float
  let classIndex: Int
  let coefficients: [Float]
}

private protocol NativeDetecting: AnyObject {
  func detect(
    _ source: CVPixelBuffer,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) async throws -> [Detection]
}

private final class CoreAIDetector: NativeDetecting {
  private let function: InferenceFunction?
  private let coreMLModel: MLModel?
  private var temporaryCompiledModelURL: URL?
  private let candidateChannels: Int
  private let classCount: Int
  private let context = CIContext(options: [.cacheIntermediates: false])
  private let detectorPool: CVPixelBufferPool
  private var normalizationScratch = [Float](
    repeating: 0,
    count: detectorSize * detectorSize * 4
  )

  init(
    modelURL: URL,
    candidateChannels: Int,
    computeUnits: String? = nil
  ) async throws {
    guard candidateChannels == 37 || candidateChannels == 38 else {
      throw NativePreviewError.invalidConfiguration(
        "detector candidate channels must be 37 or 38"
      )
    }
    self.candidateChannels = candidateChannels
    classCount = candidateChannels - 4 - 32
    let modelExtension = modelURL.pathExtension.lowercased()
    if modelExtension == "mlmodelc" || modelExtension == "mlpackage" {
      let configuration = MLModelConfiguration()
      switch (computeUnits ?? "cpuAndGPU").lowercased() {
      case "all":
        configuration.computeUnits = .all
      case "cpuonly":
        configuration.computeUnits = .cpuOnly
      case "cpuandneuralengine", "cpuandane":
        configuration.computeUnits = .cpuAndNeuralEngine
      case "cpuandgpu":
        configuration.computeUnits = .cpuAndGPU
      default:
        throw NativePreviewError.invalidConfiguration(
          "unsupported Core ML detector compute units: \(computeUnits ?? "")"
        )
      }
      let loadURL: URL
      if modelExtension == "mlpackage" {
        loadURL = try await MLModel.compileModel(at: modelURL)
        temporaryCompiledModelURL = loadURL
      } else {
        loadURL = modelURL
      }
      coreMLModel = try MLModel(
        contentsOf: loadURL,
        configuration: configuration
      )
      function = nil
    } else {
      let model = try await AIModel(contentsOf: modelURL)
      guard let loadedFunction = try model.loadFunction(named: "main") else {
        throw NativePreviewError.detector("main function is missing")
      }
      function = loadedFunction
      coreMLModel = nil
    }
    detectorPool = try Self.makePool(width: detectorSize, height: detectorSize)
  }

  deinit {
    if let temporaryCompiledModelURL {
      try? FileManager.default.removeItem(at: temporaryCompiledModelURL)
    }
    CVPixelBufferPoolFlush(detectorPool, .excessBuffers)
  }

  private static func makePool(width: Int, height: Int) throws
    -> CVPixelBufferPool
  {
    let attributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferMetalCompatibilityKey as String: true,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    var pool: CVPixelBufferPool?
    let result = CVPixelBufferPoolCreate(
      kCFAllocatorDefault,
      nil,
      attributes as CFDictionary,
      &pool
    )
    guard result == kCVReturnSuccess, let pool else {
      throw NativePreviewError.pixelBuffer("pool creation returned \(result)")
    }
    return pool
  }

  private func letterbox(_ source: CVPixelBuffer)
    throws -> (CVPixelBuffer, Float, Float, Float)
  {
    let sourceWidth = Float(CVPixelBufferGetWidth(source))
    let sourceHeight = Float(CVPixelBufferGetHeight(source))
    let scale = min(
      Float(detectorSize) / sourceWidth,
      Float(detectorSize) / sourceHeight
    )
    let renderedWidth = sourceWidth * scale
    let renderedHeight = sourceHeight * scale
    let padX = (Float(detectorSize) - renderedWidth) * 0.5
    let padY = (Float(detectorSize) - renderedHeight) * 0.5
    var output: CVPixelBuffer?
    let result = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      detectorPool,
      &output
    )
    guard result == kCVReturnSuccess, let output else {
      throw NativePreviewError.pixelBuffer(
        "detector pool allocation returned \(result)"
      )
    }
    let background = CIImage(
      color: CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255)
    ).cropped(to: CGRect(x: 0, y: 0, width: detectorSize, height: detectorSize))
    let input = CIImage(cvPixelBuffer: source)
      .transformed(by: CGAffineTransform(scaleX: CGFloat(scale), y: CGFloat(scale)))
      .transformed(
        by: CGAffineTransform(translationX: CGFloat(padX), y: CGFloat(padY))
      )
    context.render(input.composited(over: background), to: output)
    return (output, scale, padX, padY)
  }

  private func normalizedNCHW(_ pixelBuffer: CVPixelBuffer) throws
    -> NDArray
  {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw NativePreviewError.pixelBuffer("detector base address unavailable")
    }
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    var array = NDArray(
      shape: [1, 3, detectorSize, detectorSize],
      scalarType: .float16
    )
    var view = array.mutableView(as: Float16.self)
    try view.withUnsafeMutablePointer { destination, _, _ in
      try normalizationScratch.withUnsafeMutableBytes { scratchBytes in
        guard let scratchBase = scratchBytes.baseAddress else {
          throw NativePreviewError.detector(
            "normalization scratch buffer is unavailable"
          )
        }
        let plane = detectorSize * detectorSize
        let planeBytes = plane * MemoryLayout<Float>.stride
        var source = vImage_Buffer(
          data: base,
          height: vImagePixelCount(detectorSize),
          width: vImagePixelCount(detectorSize),
          rowBytes: rowBytes
        )
        // vImage names the byte order ARGB. BGRA input is mapped by routing
        // its four output planes as B, G, R, A.
        var blue = vImage_Buffer(
          data: scratchBase,
          height: vImagePixelCount(detectorSize),
          width: vImagePixelCount(detectorSize),
          rowBytes: detectorSize * MemoryLayout<Float>.stride
        )
        var green = vImage_Buffer(
          data: scratchBase.advanced(by: planeBytes),
          height: vImagePixelCount(detectorSize),
          width: vImagePixelCount(detectorSize),
          rowBytes: detectorSize * MemoryLayout<Float>.stride
        )
        var red = vImage_Buffer(
          data: scratchBase.advanced(by: 2 * planeBytes),
          height: vImagePixelCount(detectorSize),
          width: vImagePixelCount(detectorSize),
          rowBytes: detectorSize * MemoryLayout<Float>.stride
        )
        var alpha = vImage_Buffer(
          data: scratchBase.advanced(by: 3 * planeBytes),
          height: vImagePixelCount(detectorSize),
          width: vImagePixelCount(detectorSize),
          rowBytes: detectorSize * MemoryLayout<Float>.stride
        )
        var maximum: [Float] = [1, 1, 1, 1]
        var minimum: [Float] = [0, 0, 0, 0]
        let conversion = vImageConvert_ARGB8888toPlanarF(
          &source,
          &blue,
          &green,
          &red,
          &alpha,
          &maximum,
          &minimum,
          vImage_Flags(kvImageNoFlags)
        )
        guard conversion == kvImageNoError else {
          throw NativePreviewError.detector(
            "BGRA normalization returned \(conversion)"
          )
        }
        for channel in 0..<3 {
          var sourcePlane = [red, green, blue][channel]
          var destinationPlane = vImage_Buffer(
            data: destination.advanced(by: channel * plane),
            height: vImagePixelCount(detectorSize),
            width: vImagePixelCount(detectorSize),
            rowBytes: detectorSize * MemoryLayout<Float16>.stride
          )
          let halfConversion = vImageConvert_PlanarFtoPlanar16F(
            &sourcePlane,
            &destinationPlane,
            vImage_Flags(kvImageNoFlags)
          )
          guard halfConversion == kvImageNoError else {
            throw NativePreviewError.detector(
              "Float16 normalization returned \(halfConversion)"
            )
          }
        }
      }
    }
    return array
  }

  func detect(
    _ source: CVPixelBuffer,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) async throws -> [Detection] {
    let (letterboxed, scale, padX, padY) = try letterbox(source)
    let candidates: [Float]
    let prototypes: [Float]
    if let coreMLModel {
      let input = try MLDictionaryFeatureProvider(dictionary: [
        "image": MLFeatureValue(pixelBuffer: letterboxed)
      ])
      let prediction = try await coreMLModel.prediction(from: input)
      let arrays = prediction.featureNames.compactMap {
        prediction.featureValue(for: $0)?.multiArrayValue
      }
      guard let candidateArray = arrays.first(where: {
        Self.shape($0) == [1, candidateChannels, candidateCount]
      }), let prototypeArray = arrays.first(where: {
        Self.shape($0) == [1, 32, prototypeSize, prototypeSize]
      }) else {
        throw NativePreviewError.detector(
          "Core ML detector output shape is unexpected"
        )
      }
      candidates = try readFloat32(
        candidateArray,
        expectedShape: [1, candidateChannels, candidateCount]
      )
      prototypes = try readFloat32(
        prototypeArray,
        expectedShape: [1, 32, prototypeSize, prototypeSize]
      )
    } else {
      guard let function else {
        throw NativePreviewError.detector("detector backend is unavailable")
      }
      let input = try normalizedNCHW(letterboxed)
      var outputs = try await function.run(inputs: ["image": input])
      guard let candidateArray = outputs.remove("candidates")?.ndArray,
        let prototypeArray = outputs.remove("prototypes")?.ndArray
      else {
        throw NativePreviewError.detector("raw output is missing")
      }
      candidates = try readFloat16(
        candidateArray,
        expectedShape: [1, candidateChannels, candidateCount]
      )
      prototypes = try readFloat16(
        prototypeArray,
        expectedShape: [1, 32, prototypeSize, prototypeSize]
      )
    }
    var decoded: [DetectionCandidate] = []
    decoded.reserveCapacity(64)
    for index in 0..<candidateCount {
      var bestClass = 0
      var bestConfidence = -Float.infinity
      for classIndex in 0..<classCount {
        let value = candidates[(4 + classIndex) * candidateCount + index]
        if value > bestConfidence {
          bestConfidence = value
          bestClass = classIndex
        }
      }
      guard bestConfidence >= confidenceThreshold else { continue }
      var coefficients: [Float] = []
      coefficients.reserveCapacity(32)
      for channel in 0..<32 {
        coefficients.append(
          candidates[(4 + classCount + channel) * candidateCount + index]
        )
      }
      decoded.append(
        DetectionCandidate(
          x: candidates[index],
          y: candidates[candidateCount + index],
          width: candidates[2 * candidateCount + index],
          height: candidates[3 * candidateCount + index],
          confidence: bestConfidence,
          classIndex: bestClass,
          coefficients: coefficients
        )
      )
    }
    let retained = nonMaximumSuppression(decoded, threshold: iouThreshold)
    let sourceWidth = Float(CVPixelBufferGetWidth(source))
    let sourceHeight = Float(CVPixelBufferGetHeight(source))
    let sourceWidthInt = Int(sourceWidth)
    let sourceHeightInt = Int(sourceHeight)
    return retained.compactMap { chosen in
      let detectorLeft = chosen.x - chosen.width * 0.5
      let detectorTop = chosen.y - chosen.height * 0.5
      let detectorRight = chosen.x + chosen.width * 0.5
      let detectorBottom = chosen.y + chosen.height * 0.5
      // Python's int(torch.clip(...)) truncates toward zero.
      let left = Int(max(0, min(sourceWidth, (detectorLeft - padX) / scale)))
      let top = Int(max(0, min(sourceHeight, (detectorTop - padY) / scale)))
      let right = Int(max(0, min(sourceWidth, (detectorRight - padX) / scale)))
      let bottom = Int(max(0, min(sourceHeight, (detectorBottom - padY) / scale)))
      guard right > left, bottom > top else { return nil }
      return Detection(
        left: min(sourceWidthInt - 1, left),
        top: min(sourceHeightInt - 1, top),
        right: min(sourceWidthInt - 1, right),
        bottom: min(sourceHeightInt - 1, bottom),
        confidence: chosen.confidence,
        classIndex: chosen.classIndex,
        detectorMask: Self.makeDetectorMask(
          coefficients: chosen.coefficients,
          prototypes: prototypes,
          box: (
            detectorLeft,
            detectorTop,
            detectorRight,
            detectorBottom
          )
        ),
        maskWidth: detectorSize,
        maskHeight: detectorSize,
        maskProjection: .yoloLetterbox,
        maskThreshold: 0.5
      )
    }
  }

  private static func shape(_ array: MLMultiArray) -> [Int] {
    array.shape.map(\.intValue)
  }

  private func readFloat32(
    _ array: MLMultiArray,
    expectedShape: [Int]
  ) throws -> [Float] {
    let actualShape = Self.shape(array)
    guard actualShape == expectedShape else {
      throw NativePreviewError.detector(
        "unexpected Core ML output shape \(actualShape), expected \(expectedShape)"
      )
    }
    guard array.dataType == .float32 else {
      throw NativePreviewError.detector(
        "unexpected Core ML output type \(array.dataType.rawValue)"
      )
    }
    let expectedStrides = expectedShape.indices.map { index in
      expectedShape[(index + 1)...].reduce(1, *)
    }
    let actualStrides = array.strides.map(\.intValue)
    guard actualStrides == expectedStrides else {
      throw NativePreviewError.detector(
        "Core ML detector output is not contiguous"
      )
    }
    let count = expectedShape.reduce(1, *)
    let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
  }

  private func readFloat16(_ array: NDArray, expectedShape: [Int]) throws
    -> [Float]
  {
    let view = array.view(as: Float16.self)
    guard view.isContiguous else {
      throw NativePreviewError.detector("output is not contiguous")
    }
    return try view.withUnsafePointer { pointer, shape, _ in
      let matchesShape = shape.count == expectedShape.count
        && (0..<shape.count).allSatisfy {
          shape[$0] == expectedShape[$0]
        }
      guard matchesShape else {
        let actualShape = (0..<shape.count).map { shape[$0] }
        throw NativePreviewError.detector(
          "unexpected output shape \(actualShape), expected \(expectedShape)"
        )
      }
      let count = expectedShape.reduce(1, *)
      var result = [Float](repeating: 0, count: count)
      let conversion = result.withUnsafeMutableBytes { destinationBytes in
        var source = vImage_Buffer(
          data: UnsafeMutableRawPointer(mutating: pointer),
          height: 1,
          width: vImagePixelCount(count),
          rowBytes: count * MemoryLayout<Float16>.stride
        )
        var destination = vImage_Buffer(
          data: destinationBytes.baseAddress!,
          height: 1,
          width: vImagePixelCount(count),
          rowBytes: count * MemoryLayout<Float>.stride
        )
        return vImageConvert_Planar16FtoPlanarF(
          &source,
          &destination,
          vImage_Flags(kvImageNoFlags)
        )
      }
      guard conversion == kvImageNoError else {
        throw NativePreviewError.detector(
          "detector output conversion returned \(conversion)"
        )
      }
      return result
    }
  }

  private func nonMaximumSuppression(
    _ candidates: [DetectionCandidate],
    threshold: Float
  ) -> [DetectionCandidate] {
    let sorted = candidates.sorted { $0.confidence > $1.confidence }
    var kept: [DetectionCandidate] = []
    for candidate in sorted {
      if kept.contains(where: {
        $0.classIndex == candidate.classIndex
          && Self.iou($0, candidate) > threshold
      }) {
        continue
      }
      kept.append(candidate)
      if kept.count >= 100 { break }
    }
    return kept
  }

  private static func iou(
    _ first: DetectionCandidate,
    _ second: DetectionCandidate
  ) -> Float {
    let firstLeft = first.x - first.width * 0.5
    let firstTop = first.y - first.height * 0.5
    let secondLeft = second.x - second.width * 0.5
    let secondTop = second.y - second.height * 0.5
    let intersectionWidth = max(
      0,
      min(firstLeft + first.width, secondLeft + second.width)
        - max(firstLeft, secondLeft)
    )
    let intersectionHeight = max(
      0,
      min(firstTop + first.height, secondTop + second.height)
        - max(firstTop, secondTop)
    )
    let intersection = intersectionWidth * intersectionHeight
    return intersection / max(
      first.width * first.height + second.width * second.height - intersection,
      1e-6
    )
  }

  private static func makeDetectorMask(
    coefficients: [Float],
    prototypes: [Float],
    box: (Float, Float, Float, Float)
  ) -> [Float] {
    var logits = [Float](repeating: 0, count: prototypeSize * prototypeSize)
    let plane = prototypeSize * prototypeSize
    guard coefficients.count == 32, prototypes.count == 32 * plane else {
      return [Float](repeating: 0, count: detectorSize * detectorSize)
    }
    coefficients.withUnsafeBufferPointer { coefficientBuffer in
      prototypes.withUnsafeBufferPointer { prototypeBuffer in
        logits.withUnsafeMutableBufferPointer { logitBuffer in
          vDSP_mmul(
            coefficientBuffer.baseAddress!, 1,
            prototypeBuffer.baseAddress!, 1,
            logitBuffer.baseAddress!, 1,
            1, vDSP_Length(plane), 32
          )
        }
      }
    }
    var mask = [Float](repeating: 0, count: detectorSize * detectorSize)
    let left = max(0, Int(floor(box.0)))
    let top = max(0, Int(floor(box.1)))
    let right = min(detectorSize, Int(ceil(box.2)))
    let bottom = min(detectorSize, Int(ceil(box.3)))
    guard left < right, top < bottom else { return mask }
    for y in top..<bottom {
      let sourceY = (Float(y) + 0.5) / 4 - 0.5
      let y0 = max(0, min(prototypeSize - 1, Int(floor(sourceY))))
      let y1 = min(prototypeSize - 1, y0 + 1)
      let fy = max(0, min(1, sourceY - Float(y0)))
      for x in left..<right {
        let sourceX = (Float(x) + 0.5) / 4 - 0.5
        let x0 = max(0, min(prototypeSize - 1, Int(floor(sourceX))))
        let x1 = min(prototypeSize - 1, x0 + 1)
        let fx = max(0, min(1, sourceX - Float(x0)))
        let upper = logits[y0 * prototypeSize + x0] * (1 - fx)
          + logits[y0 * prototypeSize + x1] * fx
        let lower = logits[y1 * prototypeSize + x0] * (1 - fx)
          + logits[y1 * prototypeSize + x1] * fx
        if upper * (1 - fy) + lower * fy > 0 {
          mask[y * detectorSize + x] = 1
        }
      }
    }
    return mask
  }
}

/// Native RF-DETR Seg adapter for the dedicated mioh build. The model uses a
/// direct square resize, ImageNet normalization, normalized cx/cy/w/h boxes,
/// and one low-resolution logit mask per query. This intentionally does not
/// share YOLO's letterbox or prototype-mask postprocessing.
#if !MIOH_PORTABLE_COREAI
private final class RFDETRCoreAIDetector: NativeDetecting {
  private let function: InferenceFunction
  private let resolution: Int
  private let queries: Int
  private let logitClasses: Int
  private let maxDetections: Int

  init(
    modelURL: URL,
    resolution: Int,
    queries: Int,
    logitClasses: Int,
    maxDetections: Int
  ) async throws {
    guard resolution > 0, resolution % 4 == 0,
      queries > 0, logitClasses > 0, maxDetections > 0
    else {
      throw NativePreviewError.invalidConfiguration(
        "invalid RF-DETR deployment shape"
      )
    }
    let model = try await AIModel(contentsOf: modelURL)
    guard let loadedFunction = try model.loadFunction(named: "main") else {
      throw NativePreviewError.detector(
        "RF-DETR main function is missing"
      )
    }
    function = loadedFunction
    self.resolution = resolution
    self.queries = queries
    self.logitClasses = logitClasses
    self.maxDetections = maxDetections
  }

  private func normalizedNCHW(_ source: CVPixelBuffer) throws -> NDArray {
    CVPixelBufferLockBaseAddress(source, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(source) else {
      throw NativePreviewError.pixelBuffer(
        "RF-DETR source base address unavailable"
      )
    }
    let sourceWidth = CVPixelBufferGetWidth(source)
    let sourceHeight = CVPixelBufferGetHeight(source)
    let rowBytes = CVPixelBufferGetBytesPerRow(source)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    var array = NDArray(
      shape: [1, 3, resolution, resolution],
      scalarType: .float32
    )
    let view = array.mutableView(as: Float.self)
    view.withUnsafeMutablePointer { destination, _, _ in
      let plane = resolution * resolution
      let mean: [Float] = [0.485, 0.456, 0.406]
      let inverseStd: [Float] = [
        1 / 0.229, 1 / 0.224, 1 / 0.225,
      ]
      var x0 = [Int](repeating: 0, count: resolution)
      var x1 = [Int](repeating: 0, count: resolution)
      var xFraction = [Float](repeating: 0, count: resolution)
      for x in 0..<resolution {
        let sourceX = (Float(x) + 0.5)
          * Float(sourceWidth) / Float(resolution) - 0.5
        let clamped = max(0, min(Float(sourceWidth - 1), sourceX))
        x0[x] = Int(floor(clamped))
        x1[x] = min(sourceWidth - 1, x0[x] + 1)
        xFraction[x] = clamped - Float(x0[x])
      }
      for y in 0..<resolution {
        let sourceY = (Float(y) + 0.5)
          * Float(sourceHeight) / Float(resolution) - 0.5
        let clampedY = max(0, min(Float(sourceHeight - 1), sourceY))
        let y0 = Int(floor(clampedY))
        let y1 = min(sourceHeight - 1, y0 + 1)
        let fy = clampedY - Float(y0)
        let upperRow = y0 * rowBytes
        let lowerRow = y1 * rowBytes
        for x in 0..<resolution {
          let fx = xFraction[x]
          let upperLeft = upperRow + x0[x] * 4
          let upperRight = upperRow + x1[x] * 4
          let lowerLeft = lowerRow + x0[x] * 4
          let lowerRight = lowerRow + x1[x] * 4
          let index = y * resolution + x
          for channel in 0..<3 {
            let bgraChannel = 2 - channel
            let upper = Float(pixels[upperLeft + bgraChannel]) * (1 - fx)
              + Float(pixels[upperRight + bgraChannel]) * fx
            let lower = Float(pixels[lowerLeft + bgraChannel]) * (1 - fx)
              + Float(pixels[lowerRight + bgraChannel]) * fx
            let unit = (upper * (1 - fy) + lower * fy) / 255
            destination[channel * plane + index] =
              (unit - mean[channel]) * inverseStd[channel]
          }
        }
      }
    }
    return array
  }

  private func readFloat32(_ array: NDArray, expectedShape: [Int]) throws
    -> [Float]
  {
    let view = array.view(as: Float.self)
    guard view.isContiguous else {
      throw NativePreviewError.detector(
        "RF-DETR output is not contiguous"
      )
    }
    return try view.withUnsafePointer { pointer, shape, _ in
      let actualShape = (0..<shape.count).map { shape[$0] }
      guard actualShape == expectedShape else {
        throw NativePreviewError.detector(
          "unexpected RF-DETR output shape \(actualShape), expected \(expectedShape)"
        )
      }
      return Array(
        UnsafeBufferPointer(
          start: pointer,
          count: expectedShape.reduce(1, *)
        )
      )
    }
  }

  @inline(__always)
  private static func sigmoid(_ value: Float) -> Float {
    1 / (1 + exp(-max(-80, min(80, value))))
  }

  func detect(
    _ source: CVPixelBuffer,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) async throws -> [Detection] {
    _ = iouThreshold
    let input = try normalizedNCHW(source)
    var outputs = try await function.run(inputs: ["image": input])
    guard let boxesArray = outputs.remove("boxes")?.ndArray,
      let logitsArray = outputs.remove("logits")?.ndArray,
      let masksArray = outputs.remove("masks")?.ndArray
    else {
      throw NativePreviewError.detector(
        "RF-DETR output is missing boxes, logits, or masks"
      )
    }
    let maskSize = resolution / 4
    let maskPlane = maskSize * maskSize
    let boxes = try readFloat32(
      boxesArray,
      expectedShape: [1, queries, 4]
    )
    let logits = try readFloat32(
      logitsArray,
      expectedShape: [1, queries, logitClasses]
    )
    let masks = try readFloat32(
      masksArray,
      expectedShape: [1, queries, maskSize, maskSize]
    )
    let ranked = (0..<queries).map { query -> (Int, Float) in
      var best = -Float.infinity
      for classIndex in 0..<logitClasses {
        best = max(best, logits[query * logitClasses + classIndex])
      }
      return (query, best)
    }.sorted { $0.1 > $1.1 }

    let sourceWidth = CVPixelBufferGetWidth(source)
    let sourceHeight = CVPixelBufferGetHeight(source)
    var detections: [Detection] = []
    detections.reserveCapacity(maxDetections)
    for (query, bestLogit) in ranked.prefix(maxDetections) {
      let confidence = Self.sigmoid(bestLogit)
      guard confidence > confidenceThreshold else { continue }
      let boxOffset = query * 4
      let centerX = boxes[boxOffset]
      let centerY = boxes[boxOffset + 1]
      let boxWidth = boxes[boxOffset + 2]
      let boxHeight = boxes[boxOffset + 3]
      let left = Int(
        max(0, min(1, centerX - boxWidth * 0.5)
          * Float(sourceWidth))
      )
      let top = Int(
        max(0, min(1, centerY - boxHeight * 0.5)
          * Float(sourceHeight))
      )
      let right = Int(
        max(0, min(1, centerX + boxWidth * 0.5) * Float(sourceWidth))
      )
      let bottom = Int(
        max(0, min(1, centerY + boxHeight * 0.5) * Float(sourceHeight))
      )
      guard right > left, bottom > top else { continue }
      let maskOffset = query * maskPlane
      detections.append(
        Detection(
          left: min(sourceWidth - 1, left),
          top: min(sourceHeight - 1, top),
          right: min(sourceWidth - 1, right),
          bottom: min(sourceHeight - 1, bottom),
          confidence: confidence,
          classIndex: 0,
          detectorMask: Array(
            masks[maskOffset..<(maskOffset + maskPlane)]
          ),
          maskWidth: maskSize,
          maskHeight: maskSize,
          maskProjection: .directResize,
          maskThreshold: 0
        )
      )
    }
    return detections
  }
}
#endif

private struct DetectedFrame {
  let frame: DecodedFrame
  let detections: [Detection]
}

private struct DetectedBatch {
  let frames: [DetectedFrame]
  let skipPrefix: Int
}

private enum DetectionPipelineEvent {
  case batch(DetectedBatch)
  case finished(
    decodedFrames: Int,
    detectedFrames: Int,
    detectionSeconds: Double
  )
}

private struct IntBox {
  let left: Int
  let top: Int
  let right: Int
  let bottom: Int

  var width: Int { right - left + 1 }
  var height: Int { bottom - top + 1 }

  func overlaps(_ other: IntBox) -> Bool {
    left < other.right && other.left < right
      && top < other.bottom && other.top < bottom
  }

  func union(_ other: IntBox) -> IntBox {
    IntBox(
      left: min(left, other.left),
      top: min(top, other.top),
      right: max(right, other.right),
      bottom: max(bottom, other.bottom)
    )
  }
}

/// Swift-to-Swift transport for the validated variable BasicVSR++ runner.
/// Python is not involved; the mmap exists only because the already-validated
/// 15-asset runner is a separate Swift executable.
private protocol NativeRestoring: AnyObject {
  func restore(_ frames: [Float16], frameCount: Int) throws -> [Float16]
}

private final class VariableRestorerBridge: NativeRestoring {
  private let maximumFrames: Int
  private let sequenceBytes: Int
  private let descriptorURL: URL
  private let sharedURL: URL
  private let descriptorHandle: FileHandle
  private let sharedHandle: FileHandle
  private let mapping: UnsafeMutableRawPointer
  private let mappingBytes: Int
  private let process: Process
  private let input: FileHandle
  private let output: FileHandle

  init(
    runner: URL,
    models: URL,
    maximumFrames: Int
  ) throws {
    self.maximumFrames = maximumFrames
    let frameBytes = 3 * restorationSize * restorationSize
      * MemoryLayout<Float16>.stride
    sequenceBytes = maximumFrames * frameBytes
    mappingBytes = sequenceBytes * 2
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-native-restorer-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root,
      withIntermediateDirectories: true
    )
    descriptorURL = root.appendingPathComponent("descriptor.json")
    sharedURL = root.appendingPathComponent("frames.bin")
    let descriptor: [String: Any] = [
      "maximumFrames": maximumFrames,
      "inputOffset": 0,
      "outputOffset": sequenceBytes,
      "byteCount": mappingBytes,
    ]
    try JSONSerialization.data(withJSONObject: descriptor)
      .write(to: descriptorURL)
    FileManager.default.createFile(atPath: sharedURL.path, contents: nil)
    descriptorHandle = try FileHandle(forReadingFrom: descriptorURL)
    sharedHandle = try FileHandle(forUpdating: sharedURL)
    try sharedHandle.truncate(atOffset: UInt64(mappingBytes))
    mapping = mmap(
      nil,
      mappingBytes,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      sharedHandle.fileDescriptor,
      0
    )
    guard mapping != MAP_FAILED else {
      throw NativePreviewError.restorer("unable to map shared buffer")
    }
    process = Process()
    let stdinPipe = Pipe()
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.executableURL = runner
    process.arguments = [
      models.path,
      descriptorURL.path,
      sharedURL.path,
    ]
    process.standardInput = stdinPipe
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    try process.run()
    input = stdinPipe.fileHandleForWriting
    output = stdoutPipe.fileHandleForReading
  }

  deinit {
    var stop = UInt16.max.littleEndian
    try? withUnsafeBytes(of: &stop) { data in
      try input.write(contentsOf: data)
    }
    input.closeFile()
    if process.isRunning {
      process.terminate()
    }
    process.waitUntilExit()
    munmap(mapping, mappingBytes)
    try? descriptorHandle.close()
    try? sharedHandle.close()
    try? FileManager.default.removeItem(
      at: descriptorURL.deletingLastPathComponent()
    )
  }

  func restore(_ frames: [Float16], frameCount: Int) throws -> [Float16] {
    guard frameCount > 0, frameCount <= maximumFrames else {
      throw NativePreviewError.restorer("invalid frame count \(frameCount)")
    }
    let expected = frameCount * 3 * restorationSize * restorationSize
    guard frames.count == expected else {
      throw NativePreviewError.restorer(
        "invalid input count \(frames.count), expected \(expected)"
      )
    }
    frames.withUnsafeBytes {
      memcpy(mapping, $0.baseAddress!, $0.count)
    }
    var command = UInt16(frameCount).littleEndian
    try withUnsafeBytes(of: &command) { data in
      try input.write(contentsOf: data)
    }
    guard let response = try output.read(upToCount: 1),
      response == Data([0])
    else {
      throw NativePreviewError.restorer("runner returned an invalid response")
    }
    let pointer = mapping.advanced(by: sequenceBytes)
      .assumingMemoryBound(to: Float16.self)
    return Array(UnsafeBufferPointer(start: pointer, count: expected))
  }
}

private final class FixedRestorerBridge: NativeRestoring {
  private let frameCount: Int
  private let frameElements = 3 * restorationSize * restorationSize
  private let descriptorURL: URL
  private let sharedURL: URL
  private let sharedHandle: FileHandle
  private let mapping: UnsafeMutableRawPointer
  private let mappingBytes: Int
  private let process: Process
  private let input: FileHandle
  private let output: FileHandle

  init(runner: URL, model: URL, frameCount: Int) throws {
    guard frameCount > 0 else {
      throw NativePreviewError.restorer("fixed frame count must be positive")
    }
    self.frameCount = frameCount
    let tensorBytes = frameCount * frameElements * MemoryLayout<Float16>.stride
    mappingBytes = tensorBytes * 2
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-native-fixed-restorer-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root,
      withIntermediateDirectories: true
    )
    descriptorURL = root.appendingPathComponent("descriptor.json")
    sharedURL = root.appendingPathComponent("frames.bin")
    let shape = [1, frameCount, 3, restorationSize, restorationSize]
    let descriptor: [String: Any] = [
      "function": "main",
      "slotCount": 1,
      "slotStride": mappingBytes,
      "inputs": [[
        "name": "frames",
        "shape": shape,
        "offset": 0,
        "byteCount": tensorBytes,
      ]],
      "outputs": [[
        "name": "restored",
        "shape": shape,
        "offset": tensorBytes,
        "byteCount": tensorBytes,
      ]],
    ]
    try JSONSerialization.data(withJSONObject: descriptor)
      .write(to: descriptorURL, options: .atomic)
    FileManager.default.createFile(atPath: sharedURL.path, contents: nil)
    sharedHandle = try FileHandle(forUpdating: sharedURL)
    try sharedHandle.truncate(atOffset: UInt64(mappingBytes))
    mapping = mmap(
      nil,
      mappingBytes,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      sharedHandle.fileDescriptor,
      0
    )
    guard mapping != MAP_FAILED else {
      throw NativePreviewError.restorer("unable to map fixed shared buffer")
    }
    process = Process()
    let stdinPipe = Pipe()
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.executableURL = runner
    process.arguments = [model.path, descriptorURL.path, sharedURL.path]
    process.standardInput = stdinPipe
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    try process.run()
    input = stdinPipe.fileHandleForWriting
    output = stdoutPipe.fileHandleForReading
  }

  deinit {
    try? input.write(contentsOf: Data([255]))
    try? input.close()
    if process.isRunning {
      process.terminate()
    }
    process.waitUntilExit()
    munmap(mapping, mappingBytes)
    try? sharedHandle.close()
    try? FileManager.default.removeItem(
      at: descriptorURL.deletingLastPathComponent()
    )
  }

  func restore(_ frames: [Float16], frameCount actualCount: Int) throws
    -> [Float16]
  {
    guard actualCount > 0, actualCount <= frameCount,
      frames.count == actualCount * frameElements
    else {
      throw NativePreviewError.restorer(
        "invalid fixed restoration input \(actualCount)"
      )
    }
    frames.withUnsafeBytes {
      memcpy(mapping, $0.baseAddress!, $0.count)
    }
    if actualCount < frameCount {
      let last = mapping.advanced(by: (actualCount - 1) * frameElements * 2)
      for index in actualCount..<frameCount {
        memcpy(
          mapping.advanced(by: index * frameElements * 2),
          last,
          frameElements * 2
        )
      }
    }
    try input.write(contentsOf: Data([0]))
    guard let response = try output.read(upToCount: 1),
      response == Data([0])
    else {
      throw NativePreviewError.restorer(
        "fixed runner returned an invalid response"
      )
    }
    let restored = mapping.advanced(by: frameCount * frameElements * 2)
      .assumingMemoryBound(to: Float16.self)
    return Array(
      UnsafeBufferPointer(
        start: restored,
        count: actualCount * frameElements
      )
    )
  }
}

private struct NativeSceneFrame {
  let batchIndex: Int
  let source: CVPixelBuffer
  var box: IntBox
  var detections: [Detection]
}

private struct NativeScene {
  var frames: [NativeSceneFrame] = []

  var lastBox: IntBox? { frames.last?.box }
  var lastFrameIndex: Int? { frames.last?.batchIndex }

  mutating func add(
    batchIndex: Int,
    source: CVPixelBuffer,
    detection: Detection
  ) {
    let box = IntBox(
      left: detection.left,
      top: detection.top,
      right: detection.right,
      bottom: detection.bottom
    )
    if frames.last?.batchIndex == batchIndex {
      frames[frames.count - 1].box = frames[frames.count - 1].box.union(box)
      frames[frames.count - 1].detections.append(detection)
    } else {
      frames.append(
        NativeSceneFrame(
          batchIndex: batchIndex,
          source: source,
          box: box,
          detections: [detection]
        )
      )
    }
  }
}

private struct NativeClipGeometry {
  let cropBox: IntBox
  let resizedWidth: Int
  let resizedHeight: Int
  let padTop: Int
  let padLeft: Int
}

/// Fixed-shape ROI enhancer used by the all-Swift pipeline. Core AI models
/// consume/produce planar FP16 directly. Core ML image models use a pair of
/// small, bounded CVPixelBuffer pools and are immediately reduced back to the
/// 256px restoration grid, so no upscaled frame is retained by the pipeline.
private final class NativeROIEnhancer {
  private let function: InferenceFunction?
  private var coreMLModel: MLModel?
  private var compiledModelURL: URL?
  private var inputName: String?
  private var outputName: String?
  private var inputWidth: Int
  private var inputHeight: Int
  private let scale: Int
  private let strength: Float
  private let imageContext = CIContext(options: [
    .workingColorSpace: NSNull(),
    .outputColorSpace: NSNull(),
    .cacheIntermediates: false,
  ])
  private var sourcePool: CVPixelBufferPool?
  private var modelInputPool: CVPixelBufferPool?
  private var reducedPool: CVPixelBufferPool?

  init(modelURL: URL, scale: Int, strength: Float) async throws {
    self.scale = max(1, min(8, scale))
    self.strength = max(0, min(1, strength))
    inputWidth = restorationSize
    inputHeight = restorationSize
    inputName = nil
    outputName = nil
    sourcePool = nil
    modelInputPool = nil
    reducedPool = nil
    coreMLModel = nil
    compiledModelURL = nil

    if ["aimodel", "aimodelc"].contains(
      modelURL.pathExtension.lowercased()
    ) {
      let model = try await AIModel(contentsOf: modelURL)
      guard let loadedFunction = try model.loadFunction(named: "main") else {
        throw NativePreviewError.restorer(
          "ROI enhancer main function is missing"
        )
      }
      function = loadedFunction
      return
    }

    function = nil
    let loadURL: URL
    if modelURL.pathExtension.lowercased() == "mlpackage" {
      loadURL = try await Task.detached(priority: .userInitiated) {
        try MLModel.compileModel(at: modelURL)
      }.value
      compiledModelURL = loadURL
    } else {
      loadURL = modelURL
    }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    let loaded = try MLModel(contentsOf: loadURL, configuration: configuration)
    let imageInputs = loaded.modelDescription.inputDescriptionsByName.filter {
      $0.value.type == .image
    }
    let imageOutputs = loaded.modelDescription.outputDescriptionsByName.filter {
      $0.value.type == .image
    }
    guard let input = imageInputs.first,
      let output = imageOutputs.first,
      let constraint = input.value.imageConstraint
    else {
      throw NativePreviewError.invalidConfiguration(
        "ROI enhancer must expose one image input and one image output"
      )
    }
    coreMLModel = loaded
    inputName = input.key
    outputName = output.key
    inputWidth = constraint.pixelsWide
    inputHeight = constraint.pixelsHigh
    guard inputWidth > 0, inputHeight > 0 else {
      throw NativePreviewError.invalidConfiguration(
        "ROI enhancer input dimensions are invalid"
      )
    }
    sourcePool = try Self.makePool(
      width: restorationSize,
      height: restorationSize
    )
    modelInputPool = (inputWidth == restorationSize
      && inputHeight == restorationSize)
      ? nil
      : try Self.makePool(width: inputWidth, height: inputHeight)
    reducedPool = try Self.makePool(
      width: restorationSize,
      height: restorationSize
    )
  }

  deinit {
    if let sourcePool { CVPixelBufferPoolFlush(sourcePool, .excessBuffers) }
    if let modelInputPool {
      CVPixelBufferPoolFlush(modelInputPool, .excessBuffers)
    }
    if let reducedPool { CVPixelBufferPoolFlush(reducedPool, .excessBuffers) }
    coreMLModel = nil
    if let compiledModelURL {
      try? FileManager.default.removeItem(at: compiledModelURL)
    }
  }

  func enhance(_ restored: [Float16], masks: [[Float]]) async throws
    -> [Float16]
  {
    let plane = restorationSize * restorationSize
    let frameElements = plane * 3
    guard strength > 0,
      restored.count == masks.count * frameElements
    else { return restored }
    var result = restored
    for frameIndex in masks.indices {
      guard masks[frameIndex].contains(where: { $0 > 0.5 }) else { continue }
      let offset = frameIndex * frameElements
      let enhanced: [Float16]
      if let function {
        enhanced = try await enhanceCoreAI(
          function: function,
          restored: restored,
          offset: offset
        )
      } else {
        enhanced = try enhanceCoreML(restored: restored, offset: offset)
      }
      guard enhanced.count == frameElements else {
        throw NativePreviewError.restorer(
          "ROI enhancer returned \(enhanced.count) elements; expected \(frameElements)"
        )
      }
      let mask = masks[frameIndex]
      for channel in 0..<3 {
        let channelOffset = channel * plane
        for pixel in 0..<plane where mask[pixel] > 0.5 {
          let destination = offset + channelOffset + pixel
          let base = Float(restored[destination])
          let detail = Float(enhanced[channelOffset + pixel])
          result[destination] = Float16(
            max(0, min(1, base + (detail - base) * strength))
          )
        }
      }
    }
    return result
  }

  private func enhanceCoreAI(
    function: InferenceFunction,
    restored: [Float16],
    offset: Int
  ) async throws -> [Float16] {
    let plane = restorationSize * restorationSize
    let frameElements = plane * 3
    var input = NDArray(
      shape: [1, 3, restorationSize, restorationSize],
      scalarType: .float16
    )
    var inputView = input.mutableView(as: Float16.self)
    _ = inputView.withUnsafeMutablePointer { destination, _, _ in
      restored.withUnsafeBytes { source in
        memcpy(
          destination,
          source.baseAddress!.advanced(
            by: offset * MemoryLayout<Float16>.stride
          ),
          frameElements * MemoryLayout<Float16>.stride
        )
      }
    }
    var outputs = try await function.run(inputs: ["image": input])
    guard let output = outputs.remove("enhanced")?.ndArray else {
      throw NativePreviewError.restorer(
        "Core AI ROI enhancer output is missing"
      )
    }
    let outputSize = restorationSize * scale
    let expectedShape = [1, 3, outputSize, outputSize]
    let view = output.view(as: Float16.self)
    guard view.isContiguous else {
      throw NativePreviewError.restorer(
        "Core AI ROI enhancer output is not contiguous"
      )
    }
    return try view.withUnsafePointer { pointer, shape, _ in
      let actualShape = (0..<shape.count).map { shape[$0] }
      guard actualShape == expectedShape else {
        throw NativePreviewError.restorer(
          "Core AI ROI enhancer output shape \(actualShape), expected \(expectedShape)"
        )
      }
      var reduced = [Float16](repeating: 0, count: frameElements)
      let highPlane = outputSize * outputSize
      let divisor = Float(scale * scale)
      for channel in 0..<3 {
        for y in 0..<restorationSize {
          for x in 0..<restorationSize {
            var sum: Float = 0
            for subY in 0..<scale {
              let sourceY = y * scale + subY
              let row = channel * highPlane + sourceY * outputSize
              for subX in 0..<scale {
                let value = max(
                  0,
                  min(1, Float(pointer[row + x * scale + subX]))
                )
                // The Python backend converts the upscaled result to uint8
                // before INTER_AREA reduction. Preserve that contract.
                sum += Float(Int((value * 255).rounded())) / 255
              }
            }
            reduced[channel * plane + y * restorationSize + x] =
              Float16(sum / divisor)
          }
        }
      }
      return reduced
    }
  }

  private func enhanceCoreML(restored: [Float16], offset: Int) throws
    -> [Float16]
  {
    guard let model = coreMLModel,
      let inputName,
      let outputName,
      let sourcePool,
      let reducedPool
    else {
      throw NativePreviewError.restorer(
        "Core ML ROI enhancer backend is unavailable"
      )
    }
    return try autoreleasepool {
      let source = try Self.allocate(from: sourcePool, label: "enhancer input")
      try Self.writePlanarRGB(
        restored,
        offset: offset,
        to: source
      )
      let modelInput: CVPixelBuffer
      if let modelInputPool {
        modelInput = try Self.allocate(
          from: modelInputPool,
          label: "enhancer resized input"
        )
        let image = CIImage(cvPixelBuffer: source).transformed(
          by: CGAffineTransform(
            scaleX: CGFloat(inputWidth) / CGFloat(restorationSize),
            y: CGFloat(inputHeight) / CGFloat(restorationSize)
          )
        )
        imageContext.render(
          image,
          to: modelInput,
          bounds: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight),
          colorSpace: nil
        )
      } else {
        modelInput = source
      }
      let provider = try MLDictionaryFeatureProvider(dictionary: [
        inputName: MLFeatureValue(pixelBuffer: modelInput)
      ])
      let prediction = try model.prediction(from: provider)
      guard let enhanced = prediction.featureValue(
        for: outputName
      )?.imageBufferValue else {
        throw NativePreviewError.restorer(
          "Core ML ROI enhancer image output is missing"
        )
      }
      let reduced = try Self.allocate(
        from: reducedPool,
        label: "enhancer reduced output"
      )
      let outputWidth = CVPixelBufferGetWidth(enhanced)
      let outputHeight = CVPixelBufferGetHeight(enhanced)
      let image = CIImage(cvPixelBuffer: enhanced).transformed(
        by: CGAffineTransform(
          scaleX: CGFloat(restorationSize) / CGFloat(outputWidth),
          y: CGFloat(restorationSize) / CGFloat(outputHeight)
        )
      )
      imageContext.render(
        image,
        to: reduced,
        bounds: CGRect(
          x: 0,
          y: 0,
          width: restorationSize,
          height: restorationSize
        ),
        colorSpace: nil
      )
      return try Self.readPlanarRGB(from: reduced)
    }
  }

  private static func makePool(width: Int, height: Int) throws
    -> CVPixelBufferPool
  {
    let poolAttributes: [String: Any] = [
      kCVPixelBufferPoolMinimumBufferCountKey as String: 1,
      kCVPixelBufferPoolMaximumBufferAgeKey as String: 1,
    ]
    let pixelAttributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferMetalCompatibilityKey as String: true,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    var pool: CVPixelBufferPool?
    let status = CVPixelBufferPoolCreate(
      kCFAllocatorDefault,
      poolAttributes as CFDictionary,
      pixelAttributes as CFDictionary,
      &pool
    )
    guard status == kCVReturnSuccess, let pool else {
      throw NativePreviewError.pixelBuffer(
        "ROI enhancer pool creation returned \(status)"
      )
    }
    return pool
  }

  private static func allocate(
    from pool: CVPixelBufferPool,
    label: String
  ) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pool,
      &buffer
    )
    guard status == kCVReturnSuccess, let buffer else {
      throw NativePreviewError.pixelBuffer(
        "\(label) allocation returned \(status)"
      )
    }
    return buffer
  }

  private static func writePlanarRGB(
    _ source: [Float16],
    offset: Int,
    to pixelBuffer: CVPixelBuffer
  ) throws {
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw NativePreviewError.pixelBuffer(
        "ROI enhancer input base address unavailable"
      )
    }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    guard width == restorationSize, height == restorationSize else {
      throw NativePreviewError.pixelBuffer(
        "ROI enhancer source buffer must be 256x256"
      )
    }
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    let plane = restorationSize * restorationSize
    for y in 0..<restorationSize {
      let row = pixels.advanced(by: y * rowBytes)
      for x in 0..<restorationSize {
        let pixel = y * restorationSize + x
        row[x * 4] = UInt8(
          max(0, min(255, Int((Float(source[offset + 2 * plane + pixel]) * 255).rounded())))
        )
        row[x * 4 + 1] = UInt8(
          max(0, min(255, Int((Float(source[offset + plane + pixel]) * 255).rounded())))
        )
        row[x * 4 + 2] = UInt8(
          max(0, min(255, Int((Float(source[offset + pixel]) * 255).rounded())))
        )
        row[x * 4 + 3] = 255
      }
    }
  }

  private static func readPlanarRGB(from pixelBuffer: CVPixelBuffer) throws
    -> [Float16]
  {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw NativePreviewError.pixelBuffer(
        "ROI enhancer output base address unavailable"
      )
    }
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    let plane = restorationSize * restorationSize
    var result = [Float16](repeating: 0, count: plane * 3)
    for y in 0..<restorationSize {
      let row = pixels.advanced(by: y * rowBytes)
      for x in 0..<restorationSize {
        let pixel = y * restorationSize + x
        result[pixel] = Float16(Float(row[x * 4 + 2]) / 255)
        result[plane + pixel] = Float16(Float(row[x * 4 + 1]) / 255)
        result[2 * plane + pixel] = Float16(Float(row[x * 4]) / 255)
      }
    }
    return result
  }
}

private struct NativeRestoreEffects {
  let sharpen: Float
  let detail: Float
  let texture: Float
  let smoothing: Float
  let upscale: Int

  var isEnabled: Bool {
    sharpen > 0 || detail > 0 || texture > 0 || smoothing > 0
  }
}

/// Native counterpart of Python's Scene -> Clip -> restore -> unpad ->
/// per-frame resize -> create_blend_mask -> full-frame composite path.
///
/// There is deliberately no one-square-per-window shortcut here. Each
/// tracked frame keeps its own crop while the whole scene shares only the two
/// resize scale factors, exactly like ``MosaicDetector.Clip``.
private final class NativeFrameProcessor {
  private let outputPool: CVPixelBufferPool
  private let outputAllocationAttributes: CFDictionary
  private let outputBufferLimit: Int
  private let restorer: any NativeRestoring
  private let roiEnhancer: NativeROIEnhancer?
  private let blendFeather: Float
  private let effects: NativeRestoreEffects
  private let detectionEmptyLookahead: Int
  // Crossfade frames use the same un-managed BGRA values as the rest of the
  // native pipeline. Keep this GPU context, but bound every CIImage/filter
  // lifetime with an autorelease pool in crossfade().
  private let crossfadeContext = CIContext(options: [
    .workingColorSpace: NSNull(),
    .outputColorSpace: NSNull(),
    .cacheIntermediates: false,
  ])
  private(set) var lastRestoredSceneCount = 0
  private(set) var preparationSeconds = 0.0
  private(set) var restorationSeconds = 0.0
  private(set) var compositionSeconds = 0.0

  init(
    width: Int,
    height: Int,
    restorer: any NativeRestoring,
    roiEnhancer: NativeROIEnhancer?,
    blendFeather: Float,
    effects: NativeRestoreEffects,
    outputBufferLimit: Int,
    detectionEmptyLookahead: Int
  ) throws {
    self.restorer = restorer
    self.roiEnhancer = roiEnhancer
    self.blendFeather = max(0, blendFeather)
    self.effects = effects
    self.outputBufferLimit = max(16, outputBufferLimit)
    outputAllocationAttributes = [
      kCVPixelBufferPoolAllocationThresholdKey as String:
        max(16, outputBufferLimit)
    ] as CFDictionary
    self.detectionEmptyLookahead = max(0, detectionEmptyLookahead)
    outputPool = try Self.makePool(width: width, height: height)
  }

  deinit {
    CVPixelBufferPoolFlush(outputPool, .excessBuffers)
  }

  /// The normal path is a single CoreVideo allocation call. Waiting only
  /// begins after the pool reaches the pipeline-derived live-buffer ceiling,
  /// replacing swap-producing growth with backpressure without taxing normal
  /// throughput.
  private func allocateOutputBuffer(context: String) throws -> CVPixelBuffer {
    let deadline = Date().addingTimeInterval(60)
    while true {
      var output: CVPixelBuffer?
      let status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
        kCFAllocatorDefault,
        outputPool,
        outputAllocationAttributes,
        &output
      )
      if status == kCVReturnSuccess, let output {
        return output
      }
      guard status == kCVReturnWouldExceedAllocationThreshold else {
        throw NativePreviewError.pixelBuffer(
          "\(context) allocation returned \(status)"
        )
      }
      // Only unused buffers are released; buffers owned by the encoder,
      // current batch or crossfade tail remain valid.
      CVPixelBufferPoolFlush(outputPool, .excessBuffers)
      guard Date() < deadline else {
        throw NativePreviewError.pixelBuffer(
          "\(context) waited 60 seconds for the \(outputBufferLimit)-buffer pool"
        )
      }
      Thread.sleep(forTimeInterval: 0.001)
    }
  }

  private static func makePool(width: Int, height: Int) throws
    -> CVPixelBufferPool
  {
    let attributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferMetalCompatibilityKey as String: true,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    var pool: CVPixelBufferPool?
    let status = CVPixelBufferPoolCreate(
      kCFAllocatorDefault,
      nil,
      attributes as CFDictionary,
      &pool
    )
    guard status == kCVReturnSuccess, let pool else {
      throw NativePreviewError.pixelBuffer("pool creation returned \(status)")
    }
    return pool
  }

  func process(_ detected: [DetectedFrame]) async throws -> [CVPixelBuffer] {
    guard !detected.isEmpty else { return [] }
    var outputs = detected.map { $0.frame.pixelBuffer }
    let scenes = trackScenes(detected)
    lastRestoredSceneCount = scenes.count
    for scene in scenes.sorted(by: {
      ($0.frames.first?.batchIndex ?? 0) < ($1.frames.first?.batchIndex ?? 0)
    }) {
      let (restored, geometries, masks) = try await restore(scene)
      let compositionStart = Date()
      let restoredFrameElements =
        3 * restorationSize * restorationSize
      let composed = UnsafeMutablePointer<CVPixelBuffer?>.allocate(
        capacity: scene.frames.count
      )
      composed.initialize(repeating: nil, count: scene.frames.count)
      defer {
        composed.deinitialize(count: scene.frames.count)
        composed.deallocate()
      }
      let errorLock = NSLock()
      var compositionError: Error?
      DispatchQueue.concurrentPerform(iterations: scene.frames.count) {
        index in
        do {
          composed[index] = try composite(
            source: outputs[scene.frames[index].batchIndex],
            restored: restored,
            restoredOffset: index * restoredFrameElements,
            geometry: geometries[index],
            hardMask: masks[index]
          )
        } catch {
          errorLock.lock()
          if compositionError == nil {
            compositionError = error
          }
          errorLock.unlock()
        }
      }
      if let compositionError {
        throw compositionError
      }
      for index in scene.frames.indices {
        let frameIndex = scene.frames[index].batchIndex
        guard let frame = composed[index] else {
          throw NativePreviewError.pixelBuffer(
            "parallel composition produced no frame at index \(index)"
          )
        }
        outputs[frameIndex] = frame
      }
      compositionSeconds += Date().timeIntervalSince(compositionStart)
    }
    return outputs
  }

  func crossfade(
    earlier: CVPixelBuffer,
    later: CVPixelBuffer,
    laterWeight: Float
  ) throws -> CVPixelBuffer {
    guard CVPixelBufferGetWidth(earlier) == CVPixelBufferGetWidth(later),
      CVPixelBufferGetHeight(earlier) == CVPixelBufferGetHeight(later)
    else {
      throw NativePreviewError.pixelBuffer(
        "crossfade input dimensions do not match"
      )
    }
    // Overlap frames that no scene touched in either batch are literally the
    // same decoded buffer. Blending one with itself can only lose fidelity,
    // so hand it straight back.
    if earlier === later {
      return earlier
    }
    let output = try allocateOutputBuffer(context: "crossfade")
    try autoreleasepool {
      guard let filter = CIFilter(name: "CIDissolveTransition") else {
        throw NativePreviewError.pixelBuffer(
          "CIDissolveTransition is unavailable"
        )
      }
      let extent = CGRect(
        x: 0,
        y: 0,
        width: CVPixelBufferGetWidth(earlier),
        height: CVPixelBufferGetHeight(earlier)
      )
      let earlierImage = CIImage(cvPixelBuffer: earlier)
      let laterImage = CIImage(cvPixelBuffer: later)
      filter.setValue(earlierImage, forKey: kCIInputImageKey)
      filter.setValue(laterImage, forKey: kCIInputTargetImageKey)
      filter.setValue(
        max(0, min(1, laterWeight)),
        forKey: kCIInputTimeKey
      )
      guard let image = filter.outputImage?.cropped(to: extent) else {
        throw NativePreviewError.pixelBuffer(
          "crossfade filter produced no image"
        )
      }
      crossfadeContext.render(image, to: output)
    }
    CVBufferPropagateAttachments(earlier, output)
    return output
  }

  private func trackScenes(_ detected: [DetectedFrame]) -> [NativeScene] {
    var scenes: [NativeScene] = []
    for (frameIndex, item) in detected.enumerated() {
      for detection in item.detections {
        let box = IntBox(
          left: detection.left,
          top: detection.top,
          right: detection.right,
          bottom: detection.bottom
        )
        var matchingIndex: Int?
        for index in scenes.indices {
          guard let lastBox = scenes[index].lastBox,
            let lastFrame = scenes[index].lastFrameIndex,
            frameIndex - lastFrame <= detectionEmptyLookahead + 1,
            lastBox.overlaps(box)
          else {
            continue
          }
          matchingIndex = index
          break
        }
        if let matchingIndex {
          scenes[matchingIndex].add(
            batchIndex: frameIndex,
            source: item.frame.pixelBuffer,
            detection: detection
          )
        } else {
          var scene = NativeScene()
          scene.add(
            batchIndex: frameIndex,
            source: item.frame.pixelBuffer,
            detection: detection
          )
          scenes.append(scene)
        }
      }
    }
    return scenes.filter { !$0.frames.isEmpty }
  }

  private func restore(
    _ scene: NativeScene
  ) async throws -> ([Float16], [NativeClipGeometry], [[Float]]) {
    let preparationStart = Date()
    let width = CVPixelBufferGetWidth(scene.frames[0].source)
    let height = CVPixelBufferGetHeight(scene.frames[0].source)
    let cropBoxes = scene.frames.map {
      Self.cropToBox(
        $0.box,
        imageWidth: width,
        imageHeight: height
      )
    }
    let maxWidth = cropBoxes.map(\.width).max() ?? 1
    let maxHeight = cropBoxes.map(\.height).max() ?? 1
    let scaleWidth = Float(restorationSize) / Float(maxWidth)
    let scaleHeight = Float(restorationSize) / Float(maxHeight)
    var geometries: [NativeClipGeometry] = []
    geometries.reserveCapacity(scene.frames.count)
    for index in scene.frames.indices {
      let cropBox = cropBoxes[index]
      let resizedWidth = max(1, Int(Float(cropBox.width) * scaleWidth))
      let resizedHeight = max(1, Int(Float(cropBox.height) * scaleHeight))
      let padTop = Int(ceil(Double(restorationSize - resizedHeight) / 2))
      let padLeft = Int(ceil(Double(restorationSize - resizedWidth) / 2))
      let geometry = NativeClipGeometry(
        cropBox: cropBox,
        resizedWidth: resizedWidth,
        resizedHeight: resizedHeight,
        padTop: padTop,
        padLeft: padLeft
      )
      geometries.append(geometry)
    }
    let frameInputs = UnsafeMutablePointer<[Float16]?>.allocate(
      capacity: scene.frames.count
    )
    let frameMasks = UnsafeMutablePointer<[Float]?>.allocate(
      capacity: scene.frames.count
    )
    frameInputs.initialize(repeating: nil, count: scene.frames.count)
    frameMasks.initialize(repeating: nil, count: scene.frames.count)
    defer {
      frameMasks.deinitialize(count: scene.frames.count)
      frameMasks.deallocate()
      frameInputs.deinitialize(count: scene.frames.count)
      frameInputs.deallocate()
    }
    let errorLock = NSLock()
    var preparationError: Error?
    DispatchQueue.concurrentPerform(iterations: scene.frames.count) {
      index in
      do {
        frameMasks[index] = Self.makeCropMask(
          scene.frames[index],
          cropBox: cropBoxes[index],
          imageWidth: width,
          imageHeight: height
        )
        frameInputs[index] = try Self.makeModelInput(
          source: scene.frames[index].source,
          geometry: geometries[index]
        )
      } catch {
        errorLock.lock()
        if preparationError == nil {
          preparationError = error
        }
        errorLock.unlock()
      }
    }
    if let preparationError {
      throw preparationError
    }
    var hardMasks: [[Float]] = []
    hardMasks.reserveCapacity(scene.frames.count)
    var modelInput: [Float16] = []
    modelInput.reserveCapacity(
      scene.frames.count * 3 * restorationSize * restorationSize
    )
    for index in scene.frames.indices {
      guard let mask = frameMasks[index], let input = frameInputs[index] else {
        throw NativePreviewError.pixelBuffer(
          "parallel preparation produced no frame at index \(index)"
        )
      }
      hardMasks.append(mask)
      modelInput.append(contentsOf: input)
      frameMasks[index] = nil
      frameInputs[index] = nil
    }
    preparationSeconds += Date().timeIntervalSince(preparationStart)
    let restorationStart = Date()
    var restored = try restorer.restore(
      modelInput,
      frameCount: scene.frames.count
    )
    if let roiEnhancer {
      let modelMasks = geometries.indices.map {
        Self.modelMask(
          cropMask: hardMasks[$0],
          geometry: geometries[$0]
        )
      }
      restored = try await roiEnhancer.enhance(restored, masks: modelMasks)
    }
    restorationSeconds += Date().timeIntervalSince(restorationStart)
    if effects.isEnabled {
      restored = Self.applyRestoreEffects(
        restored: restored,
        original: modelInput,
        geometries: geometries,
        cropMasks: hardMasks,
        effects: effects
      )
    }
    return (restored, geometries, hardMasks)
  }

  private static func cropToBox(
    _ input: IntBox,
    imageWidth: Int,
    imageHeight: Int
  ) -> IntBox {
    var top = input.top
    var left = input.left
    var bottom = input.bottom
    var right = input.right
    var width = right - left + 1
    var height = bottom - top + 1
    let border = max(20, Int(Float(max(width, height)) * 0.06))
    top = max(0, top - border)
    left = max(0, left - border)
    bottom = min(imageHeight - 1, bottom + border)
    right = min(imageWidth - 1, right + border)
    width = right - left + 1
    height = bottom - top + 1
    let downScale = min(
      1,
      min(
        Float(restorationSize) / Float(width),
        Float(restorationSize) / Float(height)
      )
    )
    let missingWidth = Int(
      (Float(restorationSize) - Float(width) * downScale) / downScale
    )
    let missingHeight = Int(
      (Float(restorationSize) - Float(height) * downScale) / downScale
    )
    let availableLeft = left
    let availableRight = imageWidth - 1 - right
    let availableTop = top
    let availableBottom = imageHeight - 1 - bottom
    let budgetWidth = width
    let budgetHeight = height

    let expandWidthLR = min(
      availableLeft,
      availableRight,
      missingWidth / 2,
      budgetWidth
    )
    let expandWidthLeft = min(
      availableLeft - expandWidthLR,
      missingWidth - expandWidthLR * 2,
      budgetWidth - expandWidthLR
    )
    let expandWidthRight = min(
      availableRight - expandWidthLR,
      missingWidth - expandWidthLR * 2 - expandWidthLeft,
      budgetWidth - expandWidthLR - expandWidthLeft
    )
    let expandHeightTB = min(
      availableTop,
      availableBottom,
      missingHeight / 2,
      budgetHeight
    )
    let expandHeightTop = min(
      availableTop - expandHeightTB,
      missingHeight - expandHeightTB * 2,
      budgetHeight - expandHeightTB
    )
    let expandHeightBottom = min(
      availableBottom - expandHeightTB,
      missingHeight - expandHeightTB * 2 - expandHeightTop,
      budgetHeight - expandHeightTB - expandHeightTop
    )
    left -= Int(floor(Double(expandWidthLR) / 2)) + expandWidthLeft
    right += Int(ceil(Double(expandWidthLR) / 2)) + expandWidthRight
    top -= Int(floor(Double(expandHeightTB) / 2)) + expandHeightTop
    bottom += Int(ceil(Double(expandHeightTB) / 2)) + expandHeightBottom
    return IntBox(
      left: max(0, left),
      top: max(0, top),
      right: min(imageWidth - 1, right),
      bottom: min(imageHeight - 1, bottom)
    )
  }

  @inline(__always)
  private static func reflected(_ value: Int, count: Int) -> Int {
    guard count > 1 else { return 0 }
    var result = value
    while result < 0 || result >= count {
      if result < 0 {
        result = -result
      }
      if result >= count {
        result = 2 * count - 2 - result
      }
    }
    return result
  }

  private static func makeModelInput(
    source: CVPixelBuffer,
    geometry: NativeClipGeometry
  ) throws -> [Float16] {
    CVPixelBufferLockBaseAddress(source, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(source) else {
      throw NativePreviewError.pixelBuffer("crop base address unavailable")
    }
    let rowBytes = CVPixelBufferGetBytesPerRow(source)
    let sourceWidth = CVPixelBufferGetWidth(source)
    let sourceHeight = CVPixelBufferGetHeight(source)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    let plane = restorationSize * restorationSize
    var output = [Float16](repeating: 0, count: plane * 3)
    var x0 = [Int](repeating: 0, count: restorationSize)
    var x1 = [Int](repeating: 0, count: restorationSize)
    var xFraction = [Float](repeating: 0, count: restorationSize)
    for x in 0..<restorationSize {
      let resizedX = reflected(
        x - geometry.padLeft,
        count: geometry.resizedWidth
      )
      let sourceX = Float(geometry.cropBox.left)
        + (Float(resizedX) + 0.5)
          * Float(geometry.cropBox.width) / Float(geometry.resizedWidth)
        - 0.5
      let clampedX = max(0, min(Float(sourceWidth - 1), sourceX))
      let lowerX = Int(floor(clampedX))
      x0[x] = lowerX
      x1[x] = min(sourceWidth - 1, lowerX + 1)
      xFraction[x] = clampedX - Float(lowerX)
    }
    var y0 = [Int](repeating: 0, count: restorationSize)
    var y1 = [Int](repeating: 0, count: restorationSize)
    var yFraction = [Float](repeating: 0, count: restorationSize)
    for y in 0..<restorationSize {
      let resizedY = reflected(
        y - geometry.padTop,
        count: geometry.resizedHeight
      )
      let sourceY = Float(geometry.cropBox.top)
        + (Float(resizedY) + 0.5)
          * Float(geometry.cropBox.height) / Float(geometry.resizedHeight)
        - 0.5
      let clampedY = max(0, min(Float(sourceHeight - 1), sourceY))
      let lowerY = Int(floor(clampedY))
      y0[y] = lowerY
      y1[y] = min(sourceHeight - 1, lowerY + 1)
      yFraction[y] = clampedY - Float(lowerY)
    }
    for y in 0..<restorationSize {
      let upperRow = y0[y] * rowBytes
      let lowerRow = y1[y] * rowBytes
      let fy = yFraction[y]
      for x in 0..<restorationSize {
        let leftOffset = x0[x] * 4
        let rightOffset = x1[x] * 4
        let fx = xFraction[x]
        @inline(__always)
        func channel(_ channel: Int) -> Float {
          let p00 = Float(pixels[upperRow + leftOffset + channel])
          let p01 = Float(pixels[upperRow + rightOffset + channel])
          let p10 = Float(pixels[lowerRow + leftOffset + channel])
          let p11 = Float(pixels[lowerRow + rightOffset + channel])
          return (p00 * (1 - fx) + p01 * fx) * (1 - fy)
            + (p10 * (1 - fx) + p11 * fx) * fy
        }
        let index = y * restorationSize + x
        output[index] = Float16(channel(2) / 255)
        output[plane + index] = Float16(channel(1) / 255)
        output[2 * plane + index] = Float16(channel(0) / 255)
      }
    }
    return output
  }

  private static func makeCropMask(
    _ frame: NativeSceneFrame,
    cropBox: IntBox,
    imageWidth: Int,
    imageHeight: Int
  ) -> [Float] {
    var mask = [Float](repeating: 0, count: cropBox.width * cropBox.height)
    for y in 0..<cropBox.height {
      let sourceY = cropBox.top + y
      for x in 0..<cropBox.width {
        let sourceX = cropBox.left + x
        if frame.detections.contains(where: {
          hardMask(
            $0,
            sourceX: sourceX,
            sourceY: sourceY,
            imageWidth: imageWidth,
            imageHeight: imageHeight
          )
        }) {
          mask[y * cropBox.width + x] = 1
        }
      }
    }
    return mask
  }

  /// Swift-native counterpart of ``apply_restore_effect_upscale``. Effects
  /// operate on the restored 256px clip before it is resized and composited,
  /// and every stage is gated by the model-space ROI mask so clean context is
  /// bit-for-bit preserved.
  private static func applyRestoreEffects(
    restored: [Float16],
    original: [Float16],
    geometries: [NativeClipGeometry],
    cropMasks: [[Float]],
    effects: NativeRestoreEffects
  ) -> [Float16] {
    let frameElements = 3 * restorationSize * restorationSize
    guard effects.isEnabled,
      restored.count == geometries.count * frameElements,
      original.count == restored.count,
      cropMasks.count == geometries.count
    else {
      return restored
    }
    var output = restored
    for frameIndex in geometries.indices {
      let offset = frameIndex * frameElements
      let restoredFrame = restored[offset..<(offset + frameElements)].map {
        Float($0)
      }
      let originalFrame = original[offset..<(offset + frameElements)].map {
        Float($0)
      }
      let mask = modelMask(
        cropMask: cropMasks[frameIndex],
        geometry: geometries[frameIndex]
      )
      let processed = processEffects(
        restored: restoredFrame,
        original: originalFrame,
        mask: mask,
        width: restorationSize,
        height: restorationSize,
        effects: effects
      )
      for index in 0..<frameElements {
        output[offset + index] = Float16(processed[index])
      }
    }
    return output
  }

  private static func modelMask(
    cropMask: [Float],
    geometry: NativeClipGeometry
  ) -> [Float] {
    let cropWidth = geometry.cropBox.width
    let cropHeight = geometry.cropBox.height
    guard cropMask.count == cropWidth * cropHeight else {
      return [Float](repeating: 0, count: restorationSize * restorationSize)
    }
    var result = [Float](
      repeating: 0,
      count: restorationSize * restorationSize
    )
    for modelY in 0..<restorationSize {
      let resizedY = modelY - geometry.padTop
      guard resizedY >= 0, resizedY < geometry.resizedHeight else { continue }
      let cropY = min(
        cropHeight - 1,
        max(
          0,
          Int(
            (Float(resizedY) + 0.5) * Float(cropHeight)
              / Float(geometry.resizedHeight)
          )
        )
      )
      for modelX in 0..<restorationSize {
        let resizedX = modelX - geometry.padLeft
        guard resizedX >= 0, resizedX < geometry.resizedWidth else { continue }
        let cropX = min(
          cropWidth - 1,
          max(
            0,
            Int(
              (Float(resizedX) + 0.5) * Float(cropWidth)
                / Float(geometry.resizedWidth)
            )
          )
        )
        result[modelY * restorationSize + modelX] =
          cropMask[cropY * cropWidth + cropX] > 0.5 ? 1 : 0
      }
    }
    return result
  }

  private static func processEffects(
    restored: [Float],
    original: [Float],
    mask: [Float],
    width: Int,
    height: Int,
    effects: NativeRestoreEffects
  ) -> [Float] {
    guard mask.contains(where: { $0 > 0.5 }) else { return restored }
    let scale = max(1, min(4, effects.upscale))
    let workWidth = width * scale
    let workHeight = height * scale
    let base: [Float]
    let source: [Float]
    let workMask: [Float]
    if scale == 1 {
      base = restored
      source = original
      workMask = mask
    } else {
      base = resizePlanarBilinear(
        restored,
        sourceWidth: width,
        sourceHeight: height,
        destinationWidth: workWidth,
        destinationHeight: workHeight
      )
      source = resizePlanarBilinear(
        original,
        sourceWidth: width,
        sourceHeight: height,
        destinationWidth: workWidth,
        destinationHeight: workHeight
      )
      workMask = resizeMaskNearest(
        mask,
        sourceWidth: width,
        sourceHeight: height,
        destinationWidth: workWidth,
        destinationHeight: workHeight
      )
    }
    var processed = base
    if effects.texture > 0 {
      let small = maskedGaussianPlanar(
        source,
        mask: workMask,
        width: workWidth,
        height: workHeight,
        sigma: 0.7
      )
      let large = maskedGaussianPlanar(
        source,
        mask: workMask,
        width: workWidth,
        height: workHeight,
        sigma: 2
      )
      var candidate = processed
      for index in candidate.indices {
        candidate[index] = clamp01(
          processed[index] + (small[index] - large[index]) * effects.texture
        )
      }
      processed = maskedMix(
        base: processed,
        processed: candidate,
        mask: workMask
      )
    }
    if effects.detail > 0 {
      let candidate = adaptiveLumaContrast(
        processed,
        width: workWidth,
        height: workHeight,
        strength: effects.detail
      )
      processed = maskedMix(
        base: processed,
        processed: candidate,
        mask: workMask
      )
    }
    if effects.sharpen > 0 {
      let blurred = gaussianPlanar(
        processed,
        width: workWidth,
        height: workHeight,
        sigma: 1
      )
      var candidate = processed
      for index in candidate.indices {
        candidate[index] = clamp01(
          processed[index] * (1 + effects.sharpen)
            - blurred[index] * effects.sharpen
        )
      }
      processed = maskedMix(
        base: processed,
        processed: candidate,
        mask: workMask
      )
    }
    if effects.smoothing > 0 {
      let amount = min(1, effects.smoothing)
      let blurred = gaussianPlanar(
        processed,
        width: workWidth,
        height: workHeight,
        sigma: 1
      )
      var candidate = processed
      for index in candidate.indices {
        candidate[index] =
          processed[index] * (1 - amount) + blurred[index] * amount
      }
      processed = maskedMix(
        base: processed,
        processed: candidate,
        mask: workMask
      )
    }
    guard scale > 1 else { return processed }
    let reduced = downsamplePlanarArea(
      processed,
      sourceWidth: workWidth,
      sourceHeight: workHeight,
      scale: scale
    )
    return maskedMix(base: restored, processed: reduced, mask: mask)
  }

  private static func maskedMix(
    base: [Float],
    processed: [Float],
    mask: [Float]
  ) -> [Float] {
    let plane = mask.count
    guard base.count == processed.count, base.count == plane * 3 else {
      return base
    }
    var result = base
    for channel in 0..<3 {
      let offset = channel * plane
      for index in 0..<plane {
        let amount = max(0, min(1, mask[index]))
        result[offset + index] =
          base[offset + index] * (1 - amount)
          + processed[offset + index] * amount
      }
    }
    return result
  }

  private static func gaussianKernel(sigma: Float) -> [Float] {
    let radius = max(1, Int(ceil(Double(sigma * 3))))
    let denominator = 2 * sigma * sigma
    var kernel = (-radius...radius).map {
      exp(-Float($0 * $0) / denominator)
    }
    let total = kernel.reduce(0, +)
    for index in kernel.indices {
      kernel[index] /= total
    }
    return kernel
  }

  private static func gaussianSingle(
    _ input: [Float],
    width: Int,
    height: Int,
    sigma: Float
  ) -> [Float] {
    let kernel = gaussianKernel(sigma: sigma)
    let radius = kernel.count / 2
    var horizontal = [Float](repeating: 0, count: input.count)
    var output = [Float](repeating: 0, count: input.count)
    for y in 0..<height {
      for x in 0..<width {
        var value: Float = 0
        for tap in kernel.indices {
          let sourceX = reflected(x + tap - radius, count: width)
          value += input[y * width + sourceX] * kernel[tap]
        }
        horizontal[y * width + x] = value
      }
    }
    for y in 0..<height {
      for x in 0..<width {
        var value: Float = 0
        for tap in kernel.indices {
          let sourceY = reflected(y + tap - radius, count: height)
          value += horizontal[sourceY * width + x] * kernel[tap]
        }
        output[y * width + x] = value
      }
    }
    return output
  }

  private static func gaussianPlanar(
    _ input: [Float],
    width: Int,
    height: Int,
    sigma: Float
  ) -> [Float] {
    let plane = width * height
    var result = [Float](repeating: 0, count: plane * 3)
    for channel in 0..<3 {
      let offset = channel * plane
      let blurred = gaussianSingle(
        Array(input[offset..<(offset + plane)]),
        width: width,
        height: height,
        sigma: sigma
      )
      result.replaceSubrange(offset..<(offset + plane), with: blurred)
    }
    return result
  }

  private static func maskedGaussianPlanar(
    _ input: [Float],
    mask: [Float],
    width: Int,
    height: Int,
    sigma: Float
  ) -> [Float] {
    let plane = width * height
    let blurredMask = gaussianSingle(
      mask,
      width: width,
      height: height,
      sigma: sigma
    )
    var result = [Float](repeating: 0, count: plane * 3)
    for channel in 0..<3 {
      let offset = channel * plane
      var weighted = [Float](repeating: 0, count: plane)
      for index in 0..<plane {
        weighted[index] = input[offset + index] * mask[index]
      }
      let blurred = gaussianSingle(
        weighted,
        width: width,
        height: height,
        sigma: sigma
      )
      for index in 0..<plane {
        result[offset + index] =
          blurred[index] / max(blurredMask[index], 1e-6)
      }
    }
    return result
  }

  /// CLAHE-style 8x8 luminance equalization. Keeping the operation on luma
  /// preserves chroma while matching the Python detail control's local
  /// contrast semantics.
  private static func adaptiveLumaContrast(
    _ input: [Float],
    width: Int,
    height: Int,
    strength: Float
  ) -> [Float] {
    let plane = width * height
    guard input.count == plane * 3 else { return input }
    var luma = [Float](repeating: 0, count: plane)
    for index in 0..<plane {
      luma[index] =
        input[index] * 0.2126
        + input[plane + index] * 0.7152
        + input[2 * plane + index] * 0.0722
    }
    let columns = 8
    let rows = 8
    var tables = [Float](
      repeating: 0,
      count: columns * rows * 256
    )
    let clipLimit = 1 + min(1, max(0, strength)) * 2
    for tileY in 0..<rows {
      let top = tileY * height / rows
      let bottom = (tileY + 1) * height / rows
      for tileX in 0..<columns {
        let left = tileX * width / columns
        let right = (tileX + 1) * width / columns
        let pixelCount = max(1, (right - left) * (bottom - top))
        let limit = max(1, Int(clipLimit * Float(pixelCount) / 256))
        var histogram = [Int](repeating: 0, count: 256)
        for y in top..<bottom {
          for x in left..<right {
            let bin = min(255, max(0, Int(luma[y * width + x] * 255)))
            histogram[bin] += 1
          }
        }
        var excess = 0
        for bin in histogram.indices where histogram[bin] > limit {
          excess += histogram[bin] - limit
          histogram[bin] = limit
        }
        let uniform = excess / 256
        let remainder = excess % 256
        for bin in histogram.indices {
          histogram[bin] += uniform + (bin < remainder ? 1 : 0)
        }
        let tableOffset = (tileY * columns + tileX) * 256
        var cumulative = 0
        for bin in histogram.indices {
          cumulative += histogram[bin]
          tables[tableOffset + bin] =
            min(1, Float(cumulative) / Float(pixelCount))
        }
      }
    }
    var result = input
    let tileWidth = Float(width) / Float(columns)
    let tileHeight = Float(height) / Float(rows)
    for y in 0..<height {
      let tileY = (Float(y) + 0.5) / tileHeight - 0.5
      let y0 = max(0, min(rows - 1, Int(floor(tileY))))
      let y1 = min(rows - 1, y0 + 1)
      let fy = max(0, min(1, tileY - Float(y0)))
      for x in 0..<width {
        let tileX = (Float(x) + 0.5) / tileWidth - 0.5
        let x0 = max(0, min(columns - 1, Int(floor(tileX))))
        let x1 = min(columns - 1, x0 + 1)
        let fx = max(0, min(1, tileX - Float(x0)))
        let index = y * width + x
        let bin = min(255, max(0, Int(luma[index] * 255)))
        func table(_ tx: Int, _ ty: Int) -> Float {
          tables[(ty * columns + tx) * 256 + bin]
        }
        let upper = table(x0, y0) * (1 - fx) + table(x1, y0) * fx
        let lower = table(x0, y1) * (1 - fx) + table(x1, y1) * fx
        let equalized = upper * (1 - fy) + lower * fy
        let mixed = luma[index] * (1 - strength) + equalized * strength
        let delta = mixed - luma[index]
        result[index] = clamp01(input[index] + delta)
        result[plane + index] = clamp01(input[plane + index] + delta)
        result[2 * plane + index] =
          clamp01(input[2 * plane + index] + delta)
      }
    }
    return result
  }

  private static func resizePlanarBilinear(
    _ input: [Float],
    sourceWidth: Int,
    sourceHeight: Int,
    destinationWidth: Int,
    destinationHeight: Int
  ) -> [Float] {
    let sourcePlane = sourceWidth * sourceHeight
    let destinationPlane = destinationWidth * destinationHeight
    var output = [Float](repeating: 0, count: destinationPlane * 3)
    for y in 0..<destinationHeight {
      let sourceY =
        (Float(y) + 0.5) * Float(sourceHeight) / Float(destinationHeight)
        - 0.5
      let y0 = max(0, min(sourceHeight - 1, Int(floor(sourceY))))
      let y1 = min(sourceHeight - 1, y0 + 1)
      let fy = max(0, min(1, sourceY - Float(y0)))
      for x in 0..<destinationWidth {
        let sourceX =
          (Float(x) + 0.5) * Float(sourceWidth) / Float(destinationWidth)
          - 0.5
        let x0 = max(0, min(sourceWidth - 1, Int(floor(sourceX))))
        let x1 = min(sourceWidth - 1, x0 + 1)
        let fx = max(0, min(1, sourceX - Float(x0)))
        let destinationIndex = y * destinationWidth + x
        for channel in 0..<3 {
          let offset = channel * sourcePlane
          let upper =
            input[offset + y0 * sourceWidth + x0] * (1 - fx)
            + input[offset + y0 * sourceWidth + x1] * fx
          let lower =
            input[offset + y1 * sourceWidth + x0] * (1 - fx)
            + input[offset + y1 * sourceWidth + x1] * fx
          output[channel * destinationPlane + destinationIndex] =
            upper * (1 - fy) + lower * fy
        }
      }
    }
    return output
  }

  private static func resizeMaskNearest(
    _ input: [Float],
    sourceWidth: Int,
    sourceHeight: Int,
    destinationWidth: Int,
    destinationHeight: Int
  ) -> [Float] {
    var output = [Float](
      repeating: 0,
      count: destinationWidth * destinationHeight
    )
    for y in 0..<destinationHeight {
      let sourceY = min(
        sourceHeight - 1,
        y * sourceHeight / destinationHeight
      )
      for x in 0..<destinationWidth {
        let sourceX = min(
          sourceWidth - 1,
          x * sourceWidth / destinationWidth
        )
        output[y * destinationWidth + x] =
          input[sourceY * sourceWidth + sourceX]
      }
    }
    return output
  }

  private static func downsamplePlanarArea(
    _ input: [Float],
    sourceWidth: Int,
    sourceHeight: Int,
    scale: Int
  ) -> [Float] {
    let width = sourceWidth / scale
    let height = sourceHeight / scale
    let sourcePlane = sourceWidth * sourceHeight
    let destinationPlane = width * height
    var output = [Float](repeating: 0, count: destinationPlane * 3)
    let divisor = Float(scale * scale)
    for channel in 0..<3 {
      let sourceOffset = channel * sourcePlane
      let destinationOffset = channel * destinationPlane
      for y in 0..<height {
        for x in 0..<width {
          var sum: Float = 0
          for innerY in 0..<scale {
            let row = (y * scale + innerY) * sourceWidth
            for innerX in 0..<scale {
              sum += input[
                sourceOffset + row + x * scale + innerX
              ]
            }
          }
          output[destinationOffset + y * width + x] = sum / divisor
        }
      }
    }
    return output
  }

  @inline(__always)
  private static func clamp01(_ value: Float) -> Float {
    max(0, min(1, value))
  }

  private static func hardMask(
    _ detection: Detection,
    sourceX: Int,
    sourceY: Int,
    imageWidth: Int,
    imageHeight: Int
  ) -> Bool {
    let detectorX: Float
    let detectorY: Float
    switch detection.maskProjection {
    case .yoloLetterbox:
      let scale = min(
        Float(detectorSize) / Float(imageHeight),
        Float(detectorSize) / Float(imageWidth)
      )
      let padX = (Float(detectorSize) - Float(imageWidth) * scale) / 2
      let padY = (Float(detectorSize) - Float(imageHeight) * scale) / 2
      let left = Int(round(padX - 0.1))
      let top = Int(round(padY - 0.1))
      let right = detectorSize - Int(round(padX + 0.1))
      let bottom = detectorSize - Int(round(padY + 0.1))
      detectorX = Float(left)
        + (Float(sourceX) + 0.5) * Float(right - left) / Float(imageWidth) - 0.5
      detectorY = Float(top)
        + (Float(sourceY) + 0.5) * Float(bottom - top) / Float(imageHeight) - 0.5
    case .directResize:
      detectorX = (Float(sourceX) + 0.5)
        * Float(detection.maskWidth) / Float(imageWidth) - 0.5
      detectorY = (Float(sourceY) + 0.5)
        * Float(detection.maskHeight) / Float(imageHeight) - 0.5
    }
    let value = bilinearScalar(
      detection.detectorMask,
      width: detection.maskWidth,
      height: detection.maskHeight,
      x: detectorX,
      y: detectorY
    )
    return value > detection.maskThreshold
  }

  private func composite(
    source: CVPixelBuffer,
    restored: [Float16],
    restoredOffset: Int,
    geometry: NativeClipGeometry,
    hardMask: [Float]
  ) throws -> CVPixelBuffer {
    let output = try allocateOutputBuffer(context: "composite")
    CVPixelBufferLockBaseAddress(source, .readOnly)
    CVPixelBufferLockBaseAddress(output, [])
    defer {
      CVPixelBufferUnlockBaseAddress(output, [])
      CVPixelBufferUnlockBaseAddress(source, .readOnly)
    }
    guard let sourceBase = CVPixelBufferGetBaseAddress(source),
      let outputBase = CVPixelBufferGetBaseAddress(output)
    else {
      throw NativePreviewError.pixelBuffer("composite base address unavailable")
    }
    let width = CVPixelBufferGetWidth(source)
    let height = CVPixelBufferGetHeight(source)
    let sourceRowBytes = CVPixelBufferGetBytesPerRow(source)
    let outputRowBytes = CVPixelBufferGetBytesPerRow(output)
    for y in 0..<height {
      memcpy(
        outputBase.advanced(by: y * outputRowBytes),
        sourceBase.advanced(by: y * sourceRowBytes),
        min(sourceRowBytes, outputRowBytes)
      )
    }
    let plane = restorationSize * restorationSize
    let blendMask = Self.createBlendMask(
      hardMask,
      width: geometry.cropBox.width,
      height: geometry.cropBox.height,
      feather: blendFeather
    )
    let destination = outputBase.assumingMemoryBound(to: UInt8.self)
    let cropWidth = geometry.cropBox.width
    let cropHeight = geometry.cropBox.height
    var x0 = [Int](repeating: 0, count: cropWidth)
    var x1 = [Int](repeating: 0, count: cropWidth)
    var xFraction = [Float](repeating: 0, count: cropWidth)
    for cropX in 0..<cropWidth {
      let restoredX = Float(geometry.padLeft)
        + (Float(cropX) + 0.5)
          * Float(geometry.resizedWidth) / Float(cropWidth)
        - 0.5
      let clampedX = max(0, min(Float(restorationSize - 1), restoredX))
      let lowerX = Int(floor(clampedX))
      x0[cropX] = lowerX
      x1[cropX] = min(restorationSize - 1, lowerX + 1)
      xFraction[cropX] = clampedX - Float(lowerX)
    }
    var y0 = [Int](repeating: 0, count: cropHeight)
    var y1 = [Int](repeating: 0, count: cropHeight)
    var yFraction = [Float](repeating: 0, count: cropHeight)
    for cropY in 0..<cropHeight {
      let restoredY = Float(geometry.padTop)
        + (Float(cropY) + 0.5)
          * Float(geometry.resizedHeight) / Float(cropHeight)
        - 0.5
      let clampedY = max(0, min(Float(restorationSize - 1), restoredY))
      let lowerY = Int(floor(clampedY))
      y0[cropY] = lowerY
      y1[cropY] = min(restorationSize - 1, lowerY + 1)
      yFraction[cropY] = clampedY - Float(lowerY)
    }
    for cropY in 0..<geometry.cropBox.height {
      let sourceY = geometry.cropBox.top + cropY
      let upperRow = y0[cropY] * restorationSize
      let lowerRow = y1[cropY] * restorationSize
      let fy = yFraction[cropY]
      for cropX in 0..<geometry.cropBox.width {
        let alpha = blendMask[
          cropY * geometry.cropBox.width + cropX
        ]
        guard alpha > 1e-4 else { continue }
        let sourceX = geometry.cropBox.left + cropX
        guard sourceX >= 0, sourceX < width, sourceY >= 0, sourceY < height
        else { continue }
        let pixel = destination.advanced(
          by: sourceY * outputRowBytes + sourceX * 4
        )
        let leftX = x0[cropX]
        let rightX = x1[cropX]
        let fx = xFraction[cropX]
        @inline(__always)
        func sample(_ offset: Int) -> Float {
          let upper =
            Float(restored[restoredOffset + offset + upperRow + leftX])
              * (1 - fx)
            + Float(restored[
              restoredOffset + offset + upperRow + rightX
            ]) * fx
          let lower =
            Float(restored[restoredOffset + offset + lowerRow + leftX])
              * (1 - fx)
            + Float(restored[
              restoredOffset + offset + lowerRow + rightX
            ]) * fx
          return (upper * (1 - fy) + lower * fy) * 255
        }
        let restoredBlue = sample(2 * plane)
        let restoredGreen = sample(plane)
        let restoredRed = sample(0)
        pixel[0] = UInt8(max(0, min(255, Int(Float(pixel[0]) * (1 - alpha) + restoredBlue * alpha))))
        pixel[1] = UInt8(max(0, min(255, Int(Float(pixel[1]) * (1 - alpha) + restoredGreen * alpha))))
        pixel[2] = UInt8(max(0, min(255, Int(Float(pixel[2]) * (1 - alpha) + restoredRed * alpha))))
        pixel[3] = 255
      }
    }
    // Pool buffers start untagged. Without the decoder's colour attachments
    // the writer would treat composited frames differently from the
    // untouched frames around them and shift their levels.
    CVBufferPropagateAttachments(source, output)
    return output
  }

  private static func createBlendMask(
    _ mask: [Float],
    width: Int,
    height: Int,
    feather: Float
  ) -> [Float] {
    guard feather > 0 else { return mask }
    let innerHeight = Int(Float(height) * 0.95)
    let innerWidth = Int(Float(width) * 0.95)
    let outerHeight = height - innerHeight
    let outerWidth = width - innerWidth
    let borderSize = Int(round(Float(min(outerHeight, outerWidth)) * feather))
    if borderSize < 5 {
      return [Float](repeating: 1, count: width * height)
    }
    let kernel = borderSize.isMultiple(of: 2) ? borderSize + 1 : borderSize
    let padTop = outerHeight / 2
    let padLeft = outerWidth / 2
    var blend = mask
    for y in padTop..<min(height, padTop + innerHeight) {
      for x in padLeft..<min(width, padLeft + innerWidth) {
        blend[y * width + x] = 1
      }
    }
    let radius = kernel / 2
    var horizontal = [Float](repeating: 0, count: blend.count)
    var prefix = [Float](repeating: 0, count: width + 2 * radius + 1)
    for y in 0..<height {
      prefix[0] = 0
      for extendedX in 0..<(width + 2 * radius) {
        let sourceX = reflected(extendedX - radius, count: width)
        prefix[extendedX + 1] =
          prefix[extendedX] + blend[y * width + sourceX]
      }
      for x in 0..<width {
        horizontal[y * width + x] =
          (prefix[x + kernel] - prefix[x]) / Float(kernel)
      }
    }
    var result = [Float](repeating: 0, count: blend.count)
    prefix = [Float](repeating: 0, count: height + 2 * radius + 1)
    for x in 0..<width {
      prefix[0] = 0
      for extendedY in 0..<(height + 2 * radius) {
        let sourceY = reflected(extendedY - radius, count: height)
        prefix[extendedY + 1] =
          prefix[extendedY] + horizontal[sourceY * width + x]
      }
      for y in 0..<height {
        result[y * width + x] =
          (prefix[y + kernel] - prefix[y]) / Float(kernel)
      }
    }
    return result
  }

  @inline(__always)
  private static func bilinearScalar(
    _ values: [Float],
    width: Int,
    height: Int,
    x: Float,
    y: Float
  ) -> Float {
    let clampedX = max(0, min(Float(width - 1), x))
    let clampedY = max(0, min(Float(height - 1), y))
    let x0 = Int(floor(clampedX))
    let y0 = Int(floor(clampedY))
    let x1 = min(width - 1, x0 + 1)
    let y1 = min(height - 1, y0 + 1)
    let fx = clampedX - Float(x0)
    let fy = clampedY - Float(y0)
    let upper = values[y0 * width + x0] * (1 - fx)
      + values[y0 * width + x1] * fx
    let lower = values[y1 * width + x0] * (1 - fx)
      + values[y1 * width + x1] * fx
    return upper * (1 - fy) + lower * fy
  }

  @inline(__always)
  private static func bilinearFloat16(
    _ values: [Float16],
    width: Int,
    height: Int,
    x: Float,
    y: Float,
    offset: Int
  ) -> Float {
    let clampedX = max(0, min(Float(width - 1), x))
    let clampedY = max(0, min(Float(height - 1), y))
    let x0 = Int(floor(clampedX))
    let y0 = Int(floor(clampedY))
    let x1 = min(width - 1, x0 + 1)
    let y1 = min(height - 1, y0 + 1)
    let fx = clampedX - Float(x0)
    let fy = clampedY - Float(y0)
    let upper = Float(values[offset + y0 * width + x0]) * (1 - fx)
      + Float(values[offset + y0 * width + x1]) * fx
    let lower = Float(values[offset + y1 * width + x0]) * (1 - fx)
      + Float(values[offset + y1 * width + x1]) * fx
    return upper * (1 - fy) + lower * fy
  }

  @inline(__always)
  private static func bilinearBGRA(
    _ pixels: UnsafePointer<UInt8>,
    rowBytes: Int,
    width: Int,
    height: Int,
    x: Float,
    y: Float
  ) -> (Float, Float, Float) {
    let clampedX = max(0, min(Float(width - 1), x))
    let clampedY = max(0, min(Float(height - 1), y))
    let x0 = Int(floor(clampedX))
    let y0 = Int(floor(clampedY))
    let x1 = min(width - 1, x0 + 1)
    let y1 = min(height - 1, y0 + 1)
    let fx = clampedX - Float(x0)
    let fy = clampedY - Float(y0)
    @inline(__always)
    func channel(_ channel: Int) -> Float {
      let p00 = Float(pixels[y0 * rowBytes + x0 * 4 + channel])
      let p01 = Float(pixels[y0 * rowBytes + x1 * 4 + channel])
      let p10 = Float(pixels[y1 * rowBytes + x0 * 4 + channel])
      let p11 = Float(pixels[y1 * rowBytes + x1 * 4 + channel])
      return (p00 * (1 - fx) + p01 * fx) * (1 - fy)
        + (p10 * (1 - fx) + p11 * fx) * fy
    }
    return (channel(2), channel(1), channel(0))
  }
}

private final class PreviewControl: @unchecked Sendable {
  private let condition = NSCondition()
  private var stopped = false
  private var bufferLimitSeconds: Double
  private var releasedThrough = -1

  init(bufferLimitSeconds: Double) {
    self.bufferLimitSeconds = bufferLimitSeconds
  }

  func runReader() {
    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
      while let line = readLine() {
        guard let data = line.data(using: .utf8),
          let payload = try? JSONSerialization.jsonObject(with: data)
            as? [String: Any],
          let command = payload["command"] as? String
        else {
          continue
        }
        self?.condition.lock()
        if command == "stop" {
          self?.stopped = true
        } else if command == "release_through",
          let sequence = payload["sequence"] as? Int
        {
          self?.releasedThrough = max(self?.releasedThrough ?? -1, sequence)
        } else if command == "set_buffer_limit",
          let seconds = payload["seconds"] as? Double,
          seconds > 0
        {
          self?.bufferLimitSeconds = seconds
        }
        self?.condition.broadcast()
        self?.condition.unlock()
      }
      self?.condition.lock()
      self?.stopped = true
      self?.condition.broadcast()
      self?.condition.unlock()
    }
  }

  func shouldStop() -> Bool {
    condition.withLock { stopped }
  }

  func waitForCapacity(nextSequence: Int, segmentSeconds: Double) -> Bool {
    condition.lock()
    defer { condition.unlock() }
    while !stopped {
      let retained = nextSequence - releasedThrough - 1
      let limit = max(1, Int(ceil(bufferLimitSeconds / segmentSeconds)))
      if retained < limit { return true }
      condition.wait(until: Date(timeIntervalSinceNow: 0.1))
    }
    return false
  }
}

private func emit(_ payload: [String: Any]) {
  guard let data = try? JSONSerialization.data(withJSONObject: payload),
    let line = String(data: data, encoding: .utf8)
  else {
    return
  }
  print(line)
  fflush(stdout)
}

private enum NativeExportSupport {
  static func runProcess(
    executable: URL,
    arguments: [String],
    temporaryDirectory: URL?,
    failureMessage: String
  ) throws -> String {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = executable
    process.arguments = arguments
    if let temporaryDirectory {
      var environment = ProcessInfo.processInfo.environment
      environment["TMPDIR"] = temporaryDirectory.path
      environment["TEMP"] = temporaryDirectory.path
      environment["TMP"] = temporaryDirectory.path
      process.environment = environment
    }
    process.standardOutput = pipe
    process.standardError = pipe
    try process.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let output = String(data: data, encoding: .utf8) ?? ""
    guard process.terminationStatus == 0 else {
      throw NativePreviewError.export(
        "\(failureMessage): \(output.trimmingCharacters(in: .whitespacesAndNewlines))"
      )
    }
    return output
  }

  static func prepareInput(
    source: URL,
    ffmpeg: URL?,
    directory: URL
  ) async throws -> (url: URL, temporary: URL?) {
    let directlySupportedExtensions = Set(["mp4", "mov", "m4v"])
    let asset = AVURLAsset(url: source)
    let playable = (try? await asset.load(.isPlayable)) ?? false
    if directlySupportedExtensions.contains(source.pathExtension.lowercased()),
      playable
    {
      return (source, nil)
    }
    guard let ffmpeg else {
      throw NativePreviewError.export(
        "入力をAVFoundation互換にするffmpegがありません"
      )
    }
    let root = directory.appendingPathComponent(
      "native-input-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: root,
      withIntermediateDirectories: true
    )
    let compatible = root.appendingPathComponent("input.mp4")
    var arguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      "-i", source.path,
      "-map", "0:v:0", "-map", "0:a?",
      "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
      "-movflags", "+faststart",
    ]
    if try isHEVC(
      source: source,
      ffmpeg: ffmpeg,
      temporaryDirectory: directory
    ) {
      arguments.append(contentsOf: ["-tag:v", "hvc1"])
    }
    arguments.append(compatible.path)
    do {
      _ = try runProcess(
        executable: ffmpeg,
        arguments: arguments,
        temporaryDirectory: directory,
        failureMessage: "入力のremuxに失敗しました"
      )
    } catch {
      _ = try runProcess(
        executable: ffmpeg,
        arguments: [
          "-hide_banner", "-loglevel", "error", "-y",
          "-i", source.path,
          "-map", "0:v:0", "-map", "0:a?",
          "-c:v", "hevc_videotoolbox", "-q:v", "65",
          "-tag:v", "hvc1",
          "-c:a", "aac", "-b:a", "192k",
          "-movflags", "+faststart",
          compatible.path,
        ],
        temporaryDirectory: directory,
        failureMessage: "入力のVideoToolbox変換に失敗しました"
      )
    }
    return (compatible, root)
  }

  private static func isHEVC(
    source: URL,
    ffmpeg: URL,
    temporaryDirectory: URL
  ) throws -> Bool {
    let ffprobe = ffmpeg.deletingLastPathComponent()
      .appendingPathComponent("ffprobe")
    guard FileManager.default.isExecutableFile(atPath: ffprobe.path) else {
      return false
    }
    let output = try runProcess(
      executable: ffprobe,
      arguments: [
        "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        source.path,
      ],
      temporaryDirectory: temporaryDirectory,
      failureMessage: "映像コーデックの確認に失敗しました"
    )
    return output.trimmingCharacters(in: .whitespacesAndNewlines) == "hevc"
  }

  static func finishExport(
    segments: [SegmentEvent],
    source: URL,
    output: URL,
    ffmpeg: URL,
    workingDirectory: URL,
    fastStart: Bool
  ) throws {
    guard !segments.isEmpty else {
      throw NativePreviewError.export("書き出された映像セグメントがありません")
    }
    try FileManager.default.createDirectory(
      at: workingDirectory,
      withIntermediateDirectories: true
    )
    let manifest = workingDirectory.appendingPathComponent(
      "concat-\(UUID().uuidString).txt"
    )
    defer { try? FileManager.default.removeItem(at: manifest) }
    let contents = segments.sorted { $0.sequence < $1.sequence }.map {
      let escaped = $0.path.replacingOccurrences(of: "'", with: "'\\''")
      return "file '\(escaped)'"
    }.joined(separator: "\n") + "\n"
    try Data(contents.utf8).write(to: manifest, options: .atomic)

    try FileManager.default.createDirectory(
      at: output.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let ext = output.pathExtension.isEmpty ? "mp4" : output.pathExtension
    let part = output.deletingPathExtension()
      .appendingPathExtension("part")
      .appendingPathExtension(ext)
    try? FileManager.default.removeItem(at: part)
    var arguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      "-f", "concat", "-safe", "0", "-i", manifest.path,
      "-i", source.path,
      "-map", "0:v:0", "-map", "1:a:0?",
      "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
      "-map_metadata", "1",
    ]
    if fastStart && ext.lowercased() == "mp4" {
      arguments.append(contentsOf: ["-movflags", "+faststart"])
    }
    arguments.append(part.path)
    _ = try runProcess(
      executable: ffmpeg,
      arguments: arguments,
      temporaryDirectory: workingDirectory,
      failureMessage: "映像と音声の結合に失敗しました"
    )
    try? FileManager.default.removeItem(at: output)
    try FileManager.default.moveItem(at: part, to: output)
  }
}

@main
private struct NativePreviewPipeline {
  static func main() async {
    var generation = 0
    do {
      guard CommandLine.arguments.count == 2 else {
        throw NativePreviewError.invalidArguments
      }
      let configURL = URL(fileURLWithPath: CommandLine.arguments[1])
      let config = try JSONDecoder().decode(
        NativePreviewConfiguration.self,
        from: Data(contentsOf: configURL)
      )
      generation = config.generation
      try await run(config: config)
    } catch {
      emit([
        "kind": "error",
        "generation": generation,
        "message": "Swiftネイティブプレビューを開始できませんでした",
        "detail": error.localizedDescription,
      ])
      let message = "mioh-native-coreai-preview: \(error.localizedDescription)\n"
      FileHandle.standardError.write(Data(message.utf8))
      exit(EXIT_FAILURE)
    }
  }

  private static func run(config: NativePreviewConfiguration) async throws {
    guard config.temporalBatchFrames > 0,
      config.temporalBatchFrames <= maximumTemporalBatchFrames,
      config.ringCapacity >= config.temporalBatchFrames
    else {
      throw NativePreviewError.invalidConfiguration(
        "temporal batch must be 1...\(maximumTemporalBatchFrames) frames and fit in the ring"
      )
    }
    let overlap = config.isExport
      ? min(max(0, config.temporalOverlap ?? 0), config.temporalBatchFrames - 1)
      : 0
    let outputDirectory = URL(
      fileURLWithPath: config.outputDirectory,
      isDirectory: true
    )
    let ffmpegTemporaryDirectory = config.ffmpegTemporaryDirectory.map {
      URL(fileURLWithPath: $0, isDirectory: true)
    } ?? outputDirectory
    let ffmpegURL = config.ffmpeg.map { URL(fileURLWithPath: $0) }
    let preparedInput = try await NativeExportSupport.prepareInput(
      source: URL(fileURLWithPath: config.input),
      ffmpeg: ffmpegURL,
      directory: ffmpegTemporaryDirectory
    )
    defer {
      if let temporary = preparedInput.temporary {
        try? FileManager.default.removeItem(at: temporary)
      }
    }

    let ring = PixelBufferRing(capacity: config.ringCapacity)
    let decoder = try await ContinuousVideoDecoder(
      input: preparedInput.url,
      startNanoseconds: config.startNanoseconds,
      ring: ring
    )
    let video = try await decoder.description()
    let sourceFPS = Double(video.fpsNumerator) / Double(video.fpsDenominator)
    let targetRate: (numerator: Int, denominator: Int)? = config.targetFPS.map {
      requested in
      if let denominator = config.targetFPSDenominator {
        return (max(1, requested), max(1, denominator))
      }
      return NTSCFrameRate.target(
        wholeFPS: max(1, requested),
        sourceNumerator: video.fpsNumerator,
        sourceDenominator: video.fpsDenominator
      )
    }
    let targetFPSValue = targetRate.map {
      Double($0.numerator) / Double($0.denominator)
    }
    if let targetFPSValue, targetFPSValue > sourceFPS + 0.01 {
      throw NativePreviewError.invalidConfiguration(
        String(
          format:
            "Swift native FPS conversion currently supports down-conversion only (source %.3ffps, requested %.3ffps)",
          sourceFPS,
          targetFPSValue
        )
      )
    }
    let outputFPSNumerator = targetRate?.numerator ?? video.fpsNumerator
    let outputFPSDenominator = targetRate?.denominator ?? video.fpsDenominator
    let requestedSegmentSeconds: Double
    if config.isExport {
      switch config.splitMode ?? "duration" {
      case "none":
        requestedSegmentSeconds = max(1, video.durationSeconds + 1)
      case "count":
        requestedSegmentSeconds = max(
          1,
          video.durationSeconds / Double(max(1, config.segmentCount ?? 1))
        )
      default:
        requestedSegmentSeconds = max(1, config.segmentSeconds)
      }
    } else {
      requestedSegmentSeconds = config.segmentSeconds
    }
    // A user-visible "no split" export still closes the internal
    // AVAssetWriter regularly. The completed files are concatenated into one
    // final output by finishExport(), while VideoToolbox releases its retained
    // IOSurfaces at each internal boundary.
    let writerSegmentSeconds = config.isExport
      ? min(maximumInternalExportSegmentSeconds, requestedSegmentSeconds)
      : requestedSegmentSeconds
    let detector: any NativeDetecting
#if MIOH_PORTABLE_COREAI
    detector = try await CoreAIDetector(
      modelURL: URL(fileURLWithPath: config.detectionModel),
      candidateChannels: config.detectionCandidateChannels,
      computeUnits: config.detectionComputeUnits
    )
#else
    if config.detectionBackend == "rfdetr" {
      detector = try await RFDETRCoreAIDetector(
        modelURL: URL(fileURLWithPath: config.detectionModel),
        resolution: config.detectionInputSize ?? 576,
        queries: config.detectionQueries ?? 200,
        logitClasses: config.detectionLogitClasses ?? 3,
        maxDetections: config.detectionMaxDet ?? 16
      )
    } else {
      detector = try await CoreAIDetector(
        modelURL: URL(fileURLWithPath: config.detectionModel),
        candidateChannels: config.detectionCandidateChannels,
        computeUnits: config.detectionComputeUnits
      )
    }
#endif
    let restorer: any NativeRestoring
    if let fixedFrames = config.restorationFrameCount {
      restorer = try FixedRestorerBridge(
        runner: URL(fileURLWithPath: config.restorationRunner),
        model: URL(fileURLWithPath: config.restorationModels),
        frameCount: fixedFrames
      )
    } else {
      restorer = try VariableRestorerBridge(
        runner: URL(fileURLWithPath: config.restorationRunner),
        models: URL(fileURLWithPath: config.restorationModels),
        maximumFrames: config.temporalBatchFrames
      )
    }
    let roiEnhancer: NativeROIEnhancer?
    if let modelPath = config.roiEnhancerModel,
      (config.roiEnhancerStrength ?? 0) > 0
    {
      roiEnhancer = try await NativeROIEnhancer(
        modelURL: URL(fileURLWithPath: modelPath),
        scale: config.roiEnhancerScale ?? 4,
        strength: config.roiEnhancerStrength ?? 0
      )
    } else {
      roiEnhancer = nil
    }
    let processor = try NativeFrameProcessor(
      width: video.width,
      height: video.height,
      restorer: restorer,
      roiEnhancer: roiEnhancer,
      blendFeather: config.blendFeather ?? 1,
      effects: NativeRestoreEffects(
        sharpen: max(0, min(2, config.sharpenStrength ?? 0)),
        detail: max(0, min(1, config.detailBoost ?? 0)),
        texture: max(0, min(1, config.textureMix ?? 0)),
        smoothing: max(0, min(1, config.smoothStrength ?? 0)),
        upscale: max(1, min(4, config.effectUpscale ?? 1))
      ),
      // One batch may be in VideoToolbox, one remains the current output,
      // and one temporary generation can exist while an overlapping scene
      // replaces that output. Crossfade owns at most two overlap tails.
      outputBufferLimit: max(
        64,
        config.temporalBatchFrames * 3 + overlap * 2 + 16
      ),
      detectionEmptyLookahead: config.detectionEmptyLookahead ?? 0
    )
    let codec: AVVideoCodecType
    switch config.videoCodec?.lowercased() {
    case "h264":
      codec = .h264
    case "hevc":
      codec = .hevc
    default:
      // Preview configurations predate the export-only codec field and omit
      // it. Preserve their established low-latency H.264 contract; only a
      // full-file export defaults to HEVC.
      codec = config.isExport ? .hevc : .h264
    }
    let sourceBasedBitRate: Int? = video.estimatedDataRate > 0
      ? Int(
        min(
          120_000_000,
          max(
            2_000_000,
            video.estimatedDataRate * (config.bitrateMultiplier ?? 3)
          )
        )
      )
      : nil
    let writer = try SegmentWriter(
      outputDirectory: outputDirectory,
      width: video.width,
      height: video.height,
      fpsNumerator: outputFPSNumerator,
      fpsDenominator: outputFPSDenominator,
      generation: config.generation,
      segmentSeconds: writerSegmentSeconds,
      codec: codec,
      averageBitRate: config.averageBitRate ?? sourceBasedBitRate,
      realTime: !config.isExport,
      filePrefix: config.isExport ? "export" : "preview"
    )
    let control = PreviewControl(
      bufferLimitSeconds: config.isExport
        ? Double.greatestFiniteMagnitude
        : config.bufferLimitSeconds
    )
    let wallStart = Date()
    control.runReader()
    emit([
      "kind": "ready",
      "generation": config.generation,
      "duration": video.durationSeconds,
      "fps": Double(outputFPSNumerator) / Double(outputFPSDenominator),
      "source_fps": sourceFPS,
      "fps_conversion_stage": targetRate == nil
        ? "none"
        : ((config.preFPSConversion ?? false) ? "before_restore" : "after_restore"),
      "width": video.width,
      "height": video.height,
      "segment_seconds": writerSegmentSeconds,
      "requested_segment_seconds": requestedSegmentSeconds,
      "internal_segment_seconds": writerSegmentSeconds,
      "output_buffer_limit": max(
        64,
        config.temporalBatchFrames * 3 + overlap * 2 + 16
      ),
      "pipeline": config.isExport
        ? "swift-native-export"
        : "swift-cvpixelbuffer-ring-coreai",
    ])
    try decoder.start()
    var pendingEncoding: Task<([SegmentEvent], Int), Error>?
    do {
      var nextSequence = 0
      var decodedFrames = 0
      var detectedFrames = 0
      var restoredBatches = 0
      var detectionSeconds = 0.0
      var completedSegments: [SegmentEvent] = []
      var encodedFrames = 0
      var deferredCrossfadeTail: [ProcessedFrame] = []
      var postRestorationFrameRateGate: PTSFrameRateGate? =
        (config.preFPSConversion ?? false)
        ? nil
        : targetRate.map {
          PTSFrameRateGate(numerator: $0.numerator, denominator: $0.denominator)
        }

      // Keep at most two detected batches resident: one being restored and
      // one being prepared. This overlaps detection CPU/Core AI work with the
      // current restoration without allowing memory to grow with clip length.
      let availableBatchSlots = DispatchSemaphore(value: 1)
      let detectionEvents = AsyncThrowingStream<
        DetectionPipelineEvent, Error
      > { continuation in
        let detectionTask = Task.detached(priority: .userInitiated) {
          do {
            var pending: [DetectedFrame] = []
            pending.reserveCapacity(config.temporalBatchFrames)
            var localDecodedFrames = 0
            var localDetectedFrames = 0
            var localDetectionSeconds = 0.0
            var batchIndex = 0
            var newFramesSinceYield = 0
            let emptyLookaheadFrames = max(
              1,
              config.detectionEmptyLookahead ?? 0
            )
            var lookaheadWindow: [DecodedFrame] = []
            lookaheadWindow.reserveCapacity(emptyLookaheadFrames)
            var preRestorationFrameRateGate: PTSFrameRateGate? =
              (config.preFPSConversion ?? false)
              ? targetRate.map {
                PTSFrameRateGate(
                  numerator: $0.numerator,
                  denominator: $0.denominator
                )
              }
              : nil

            func yieldPending() {
              guard !pending.isEmpty, newFramesSinceYield > 0 else { return }
              availableBatchSlots.wait()
              if control.shouldStop() {
                availableBatchSlots.signal()
                return
              }
              let skipPrefix = batchIndex == 0 ? 0 : overlap
              continuation.yield(
                .batch(
                  DetectedBatch(frames: pending, skipPrefix: skipPrefix)
                )
              )
              batchIndex += 1
              if overlap > 0 {
                pending = Array(pending.suffix(overlap))
              } else {
                pending.removeAll(keepingCapacity: true)
              }
              newFramesSinceYield = 0
            }

            func inferDetections(
              _ frame: DecodedFrame
            ) async throws -> [Detection] {
              let detectionStart = Date()
              let allDetections = try await detector.detect(
                frame.pixelBuffer,
                confidenceThreshold: config.confidenceThreshold,
                iouThreshold: config.iouThreshold
              )
              localDetectionSeconds += Date().timeIntervalSince(
                detectionStart
              )
              return (config.detectFaceMosaics ?? false)
                ? allDetections.filter { $0.classIndex == 0 }
                : allDetections
            }

            func appendDetectedFrame(
              _ frame: DecodedFrame,
              detections: [Detection]
            ) {
              if !detections.isEmpty {
                localDetectedFrames += 1
              }
              pending.append(
                DetectedFrame(frame: frame, detections: detections)
              )
              newFramesSinceYield += 1
              if pending.count == config.temporalBatchFrames {
                yieldPending()
              }
            }

            func processLookaheadWindow() async throws {
              guard !lookaheadWindow.isEmpty else { return }
              if lookaheadWindow.count == 1 {
                let detections = try await inferDetections(
                  lookaheadWindow[0]
                )
                appendDetectedFrame(
                  lookaheadWindow[0],
                  detections: detections
                )
                lookaheadWindow.removeAll(keepingCapacity: true)
                return
              }

              // Match Python's empty-lookahead contract: sample the first and
              // last frame. Only when both are empty may the whole range skip
              // inference. If either is positive, infer every middle frame;
              // reuse the endpoint results instead of evaluating them twice.
              let firstDetections = try await inferDetections(
                lookaheadWindow[0]
              )
              let lastIndex = lookaheadWindow.count - 1
              if !firstDetections.isEmpty {
                // A positive first frame already commits this range to full
                // inference. Do not preflight the last frame and then infer
                // it a second time as part of the full range.
                appendDetectedFrame(
                  lookaheadWindow[0],
                  detections: firstDetections
                )
                for index in 1...lastIndex {
                  let detections = try await inferDetections(
                    lookaheadWindow[index]
                  )
                  appendDetectedFrame(
                    lookaheadWindow[index],
                    detections: detections
                  )
                }
                lookaheadWindow.removeAll(keepingCapacity: true)
                return
              }

              let lastDetections = try await inferDetections(
                lookaheadWindow[lastIndex]
              )
              if lastDetections.isEmpty {
                for frame in lookaheadWindow {
                  appendDetectedFrame(frame, detections: [])
                }
              } else {
                for index in lookaheadWindow.indices {
                  let detections: [Detection]
                  if index == 0 {
                    detections = firstDetections
                  } else if index == lastIndex {
                    detections = lastDetections
                  } else {
                    detections = try await inferDetections(
                      lookaheadWindow[index]
                    )
                  }
                  appendDetectedFrame(
                    lookaheadWindow[index],
                    detections: detections
                  )
                }
              }
              lookaheadWindow.removeAll(keepingCapacity: true)
            }

            while !control.shouldStop(), let frame = try ring.pop() {
              localDecodedFrames += 1
              if var gate = preRestorationFrameRateGate {
                let accepted = gate.accepts(frame.ptsNanoseconds)
                preRestorationFrameRateGate = gate
                if !accepted { continue }
              }
              lookaheadWindow.append(frame)
              if lookaheadWindow.count == emptyLookaheadFrames {
                try await processLookaheadWindow()
              }
            }
            if !control.shouldStop() {
              // Process the final partial range using the same endpoint rule.
              try await processLookaheadWindow()
              yieldPending()
              continuation.yield(
                .finished(
                  decodedFrames: localDecodedFrames,
                  detectedFrames: localDetectedFrames,
                  detectionSeconds: localDetectionSeconds
                )
              )
            }
            continuation.finish()
          } catch {
            continuation.finish(throwing: error)
          }
        }
        continuation.onTermination = { _ in
          detectionTask.cancel()
          availableBatchSlots.signal()
          availableBatchSlots.signal()
        }
      }

      for try await event in detectionEvents {
        switch event {
        case .batch(let detectedBatch):
          do {
            let batch = detectedBatch.frames
            let outputs = try await processor.process(batch)
            restoredBatches += processor.lastRestoredSceneCount
            if let encodingTask = pendingEncoding {
              let (segments, completedNextSequence) =
                try await encodingTask.value
              for segment in segments {
                emitSegment(segment, generation: config.generation)
                completedSegments.append(segment)
              }
              nextSequence = completedNextSequence
            }
            // A completed detached Task retains its captured frame array until
            // the handle is released. Drop it before crossfade allocates the
            // next generation of pixel buffers.
            pendingEncoding = nil
            let startingSequence = nextSequence
            var framesToEncode: [ProcessedFrame] = []
            if (config.crossfade ?? false), overlap > 0 {
              let prefixCount = min(detectedBatch.skipPrefix, outputs.count)
              let matchedCount = min(prefixCount, deferredCrossfadeTail.count)
              if matchedCount > 0 {
                for index in 0..<matchedCount {
                  framesToEncode.append(
                    ProcessedFrame(
                      pixelBuffer: try processor.crossfade(
                        earlier: deferredCrossfadeTail[index].pixelBuffer,
                        later: outputs[index],
                        laterWeight: Float(index + 1)
                          / Float(matchedCount + 1)
                      ),
                      ptsNanoseconds:
                        deferredCrossfadeTail[index].ptsNanoseconds
                    )
                  )
                }
              }
              if deferredCrossfadeTail.count > matchedCount {
                framesToEncode.append(
                  contentsOf: deferredCrossfadeTail[matchedCount...]
                )
              }
              if prefixCount > matchedCount {
                for index in matchedCount..<prefixCount {
                  framesToEncode.append(
                    ProcessedFrame(
                      pixelBuffer: outputs[index],
                      ptsNanoseconds: batch[index].frame.ptsNanoseconds
                    )
                  )
                }
              }
              let uniqueCount = max(0, outputs.count - prefixCount)
              let tailCount = min(overlap, uniqueCount)
              let bodyEnd = outputs.count - tailCount
              if prefixCount < bodyEnd {
                for index in prefixCount..<bodyEnd {
                  framesToEncode.append(
                    ProcessedFrame(
                      pixelBuffer: outputs[index],
                      ptsNanoseconds: batch[index].frame.ptsNanoseconds
                    )
                  )
                }
              }
              deferredCrossfadeTail = []
              if tailCount > 0 {
                for index in bodyEnd..<outputs.count {
                  deferredCrossfadeTail.append(
                    ProcessedFrame(
                      pixelBuffer: outputs[index],
                      ptsNanoseconds: batch[index].frame.ptsNanoseconds
                    )
                  )
                }
              }
            } else {
              for index in detectedBatch.skipPrefix..<outputs.count {
                framesToEncode.append(
                  ProcessedFrame(
                    pixelBuffer: outputs[index],
                    ptsNanoseconds: batch[index].frame.ptsNanoseconds
                  )
                )
              }
            }
            let acceptedFrames = framesToEncode.filter { frame in
              guard var gate = postRestorationFrameRateGate else { return true }
              let accepted = gate.accepts(frame.ptsNanoseconds)
              postRestorationFrameRateGate = gate
              return accepted
            }
            pendingEncoding = Task.detached(priority: .userInitiated) {
              var completedSegments: [SegmentEvent] = []
              var encodingNextSequence = startingSequence
              for frame in acceptedFrames {
                guard (
                  config.isExport
                    ? !control.shouldStop()
                    : control.waitForCapacity(
                      nextSequence: encodingNextSequence,
                      segmentSeconds: config.segmentSeconds
                    )
                ) else {
                  break
                }
                if let segment = try await writer.append(
                  pixelBuffer: frame.pixelBuffer,
                  ptsNanoseconds: frame.ptsNanoseconds
                ) {
                  completedSegments.append(segment)
                  encodingNextSequence = segment.sequence + 1
                }
              }
              return (
                completedSegments,
                encodingNextSequence
              )
            }
            encodedFrames += acceptedFrames.count
            if config.isExport,
              let last = batch.last
            {
              let position = Double(last.frame.ptsNanoseconds) / 1_000_000_000
              let percent = min(
                100,
                max(0, position / max(0.001, video.durationSeconds) * 100)
              )
              let elapsed = max(0.001, Date().timeIntervalSince(wallStart))
              let throughput = Double(encodedFrames) / elapsed
              let eta = percent > 0
                ? elapsed * max(0, 100 - percent) / percent
                : 0
              emit([
                "kind": "export_progress",
                "generation": config.generation,
                "percent": percent,
                "position_seconds": position,
                "duration_seconds": video.durationSeconds,
                "encoded_frames": encodedFrames,
                "elapsed_seconds": elapsed,
                "throughput_fps": throughput,
                "eta_seconds": eta,
              ])
            }
            availableBatchSlots.signal()
          } catch {
            availableBatchSlots.signal()
            throw error
          }
        case .finished(
          let completedDecodedFrames,
          let completedDetectedFrames,
          let completedDetectionSeconds
        ):
          decodedFrames = completedDecodedFrames
          detectedFrames = completedDetectedFrames
          detectionSeconds = completedDetectionSeconds
        }
      }

      if control.shouldStop() {
        pendingEncoding?.cancel()
        _ = try? await pendingEncoding?.value
        pendingEncoding = nil
      }
      if !control.shouldStop() {
        if let encodingTask = pendingEncoding {
          let (segments, completedNextSequence) =
            try await encodingTask.value
          for segment in segments {
            emitSegment(segment, generation: config.generation)
            completedSegments.append(segment)
          }
          nextSequence = completedNextSequence
        }
        pendingEncoding = nil
        if !deferredCrossfadeTail.isEmpty {
          for frame in deferredCrossfadeTail {
            if var gate = postRestorationFrameRateGate {
              let accepted = gate.accepts(frame.ptsNanoseconds)
              postRestorationFrameRateGate = gate
              if !accepted { continue }
            }
            if let segment = try await writer.append(
              pixelBuffer: frame.pixelBuffer,
              ptsNanoseconds: frame.ptsNanoseconds
            ) {
              emitSegment(segment, generation: config.generation)
              completedSegments.append(segment)
              nextSequence = segment.sequence + 1
            }
            encodedFrames += 1
          }
          deferredCrossfadeTail.removeAll(keepingCapacity: false)
        }
        if let segment = try await writer.finish() {
          emitSegment(segment, generation: config.generation)
          completedSegments.append(segment)
          nextSequence = segment.sequence + 1
        }
        if config.isExport {
          guard let outputFile = config.outputFile,
            let ffmpegURL
          else {
            throw NativePreviewError.invalidConfiguration(
              "export mode requires outputFile and ffmpeg"
            )
          }
          emit([
            "kind": "export_finalizing",
            "generation": config.generation,
            "message": "音声を結合しています",
          ])
          try NativeExportSupport.finishExport(
            segments: completedSegments,
            source: URL(fileURLWithPath: config.input),
            output: URL(fileURLWithPath: outputFile),
            ffmpeg: ffmpegURL,
            workingDirectory: ffmpegTemporaryDirectory,
            fastStart: config.mp4FastStart ?? false
          )
        }
        let elapsed = max(0.001, Date().timeIntervalSince(wallStart))
        emit([
          "kind": "native_stats",
          "generation": config.generation,
          "decoded_frames": decodedFrames,
          "detected_frames": detectedFrames,
          "restored_batches": restoredBatches,
          "encoded_frames": encodedFrames,
          "source_fps": sourceFPS,
          "output_fps": Double(outputFPSNumerator)
            / Double(outputFPSDenominator),
          "detection_seconds": detectionSeconds,
          "preparation_seconds": processor.preparationSeconds,
          "restoration_seconds": processor.restorationSeconds,
          "composition_seconds": processor.compositionSeconds,
          "elapsed_seconds": elapsed,
          "throughput_fps": Double(decodedFrames) / elapsed,
        ])
        var ended: [String: Any] = [
          "kind": "ended",
          "generation": config.generation,
        ]
        if let outputFile = config.outputFile {
          ended["output"] = outputFile
        }
        emit(ended)
      }
    } catch {
      pendingEncoding?.cancel()
      _ = try? await pendingEncoding?.value
      await decoder.stop()
      writer.discard()
      throw error
    }
    await decoder.stop()
    writer.discard()
  }

  private static func emitSegment(
    _ segment: SegmentEvent,
    generation: Int
  ) {
    emit([
      "kind": "segment",
      "generation": generation,
      "sequence": segment.sequence,
      "start_ns": segment.startNs,
      "end_ns": segment.endNs,
      "path": segment.path,
      "codec": segment.codec,
    ])
    emit([
      "kind": "progress",
      "generation": generation,
      "position_ns": segment.endNs,
    ])
  }
}
