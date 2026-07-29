// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AVFoundation
import Accelerate
import CoreAI
import CoreImage
import CoreVideo
import Darwin
import Foundation

private let detectorSize = 640
private let prototypeSize = 160
private let restorationSize = 256
private let candidateCount = 8400

private struct NativePreviewConfiguration: Decodable {
  let input: String
  let outputDirectory: String
  let detectionModel: String
  let detectionCandidateChannels: Int
  let restorationModels: String
  let restorationRunner: String
  let startNanoseconds: Int64
  let generation: Int
  let segmentSeconds: Double
  let bufferLimitSeconds: Double
  let temporalBatchFrames: Int
  let ringCapacity: Int
  let confidenceThreshold: Float
  let iouThreshold: Float
  let contextFraction: Float
  let blendFeather: Float?
}

private enum NativePreviewError: LocalizedError {
  case invalidArguments
  case invalidConfiguration(String)
  case missingVideoTrack
  case reader(String)
  case pixelBuffer(String)
  case detector(String)
  case restorer(String)

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
    }
  }
}

private struct VideoDescription {
  let width: Int
  let height: Int
  let fpsNumerator: Int
  let fpsDenominator: Int
  let durationSeconds: Double
}

private struct DecodedFrame: @unchecked Sendable {
  let pixelBuffer: CVPixelBuffer
  let ptsNanoseconds: Int64
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
    let fps = max(1.0, Double(frameRate))
    let scale = 1000
    return VideoDescription(
      width: max(1, Int(abs(natural.width).rounded())),
      height: max(1, Int(abs(natural.height).rounded())),
      fpsNumerator: max(1, Int((fps * Double(scale)).rounded())),
      fpsDenominator: scale,
      durationSeconds: duration.seconds
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

private struct Detection {
  let left: Int
  let top: Int
  let right: Int
  let bottom: Int
  let confidence: Float
  let classIndex: Int
  /// This is the same 640x640 hard mask produced by Ultralytics
  /// ``process_mask(..., upsample=True)`` before letterbox removal.
  let detectorMask: [UInt8]

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

private final class CoreAIDetector {
  private let function: InferenceFunction
  private let candidateChannels: Int
  private let classCount: Int
  private let context = CIContext(options: [.cacheIntermediates: false])
  private let detectorPool: CVPixelBufferPool
  private var normalizationScratch = [Float](
    repeating: 0,
    count: detectorSize * detectorSize * 4
  )

  init(modelURL: URL, candidateChannels: Int) async throws {
    guard candidateChannels == 37 || candidateChannels == 38 else {
      throw NativePreviewError.invalidConfiguration(
        "detector candidate channels must be 37 or 38"
      )
    }
    self.candidateChannels = candidateChannels
    classCount = candidateChannels - 4 - 32
    let model = try await AIModel(contentsOf: modelURL)
    guard let function = try model.loadFunction(named: "main") else {
      throw NativePreviewError.detector("main function is missing")
    }
    self.function = function
    detectorPool = try Self.makePool(width: detectorSize, height: detectorSize)
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
    let input = try normalizedNCHW(letterboxed)
    var outputs = try await function.run(inputs: ["image": input])
    guard let candidateArray = outputs.remove("candidates")?.ndArray,
      let prototypeArray = outputs.remove("prototypes")?.ndArray
    else {
      throw NativePreviewError.detector("raw output is missing")
    }
    let candidates = try readFloat16(
      candidateArray,
      expectedShape: [1, candidateChannels, candidateCount]
    )
    let prototypes = try readFloat16(
      prototypeArray,
      expectedShape: [1, 32, prototypeSize, prototypeSize]
    )
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
        )
      )
    }
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
  ) -> [UInt8] {
    var logits = [Float](repeating: 0, count: prototypeSize * prototypeSize)
    let plane = prototypeSize * prototypeSize
    guard coefficients.count == 32, prototypes.count == 32 * plane else {
      return [UInt8](repeating: 0, count: detectorSize * detectorSize)
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
    var mask = [UInt8](repeating: 0, count: detectorSize * detectorSize)
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

private struct DetectedFrame {
  let frame: DecodedFrame
  let detections: [Detection]
}

private enum DetectionPipelineEvent {
  case batch([DetectedFrame])
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
private final class VariableRestorerBridge {
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

/// Native counterpart of Python's Scene -> Clip -> restore -> unpad ->
/// per-frame resize -> create_blend_mask -> full-frame composite path.
///
/// There is deliberately no one-square-per-window shortcut here. Each
/// tracked frame keeps its own crop while the whole scene shares only the two
/// resize scale factors, exactly like ``MosaicDetector.Clip``.
private final class NativeFrameProcessor {
  private let outputPool: CVPixelBufferPool
  private let restorer: VariableRestorerBridge
  private let blendFeather: Float
  private(set) var lastRestoredSceneCount = 0
  private(set) var preparationSeconds = 0.0
  private(set) var restorationSeconds = 0.0
  private(set) var compositionSeconds = 0.0

  init(
    width: Int,
    height: Int,
    restorer: VariableRestorerBridge,
    blendFeather: Float
  ) throws {
    self.restorer = restorer
    self.blendFeather = max(0, blendFeather)
    outputPool = try Self.makePool(width: width, height: height)
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

  func process(_ detected: [DetectedFrame]) throws -> [CVPixelBuffer] {
    guard !detected.isEmpty else { return [] }
    var outputs = detected.map { $0.frame.pixelBuffer }
    let scenes = trackScenes(detected)
    lastRestoredSceneCount = scenes.count
    for scene in scenes.sorted(by: {
      ($0.frames.first?.batchIndex ?? 0) < ($1.frames.first?.batchIndex ?? 0)
    }) {
      let (restored, geometries, masks) = try restore(scene)
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
            frameIndex - lastFrame <= 1,
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
  ) throws -> ([Float16], [NativeClipGeometry], [[Float]]) {
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
    let restored = try restorer.restore(
      modelInput,
      frameCount: scene.frames.count
    )
    restorationSeconds += Date().timeIntervalSince(restorationStart)
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

  private static func hardMask(
    _ detection: Detection,
    sourceX: Int,
    sourceY: Int,
    imageWidth: Int,
    imageHeight: Int
  ) -> Bool {
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
    let detectorX = Float(left)
      + (Float(sourceX) + 0.5) * Float(right - left) / Float(imageWidth) - 0.5
    let detectorY = Float(top)
      + (Float(sourceY) + 0.5) * Float(bottom - top) / Float(imageHeight) - 0.5
    let value = bilinearScalar(
      detection.detectorMask,
      width: detectorSize,
      height: detectorSize,
      x: detectorX,
      y: detectorY
    )
    return value > 0.5
  }

  private func composite(
    source: CVPixelBuffer,
    restored: [Float16],
    restoredOffset: Int,
    geometry: NativeClipGeometry,
    hardMask: [Float]
  ) throws -> CVPixelBuffer {
    var output: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      outputPool,
      &output
    )
    guard status == kCVReturnSuccess, let output else {
      throw NativePreviewError.pixelBuffer(
        "output allocation returned \(status)"
      )
    }
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
    _ values: [UInt8],
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
    let upper = Float(values[y0 * width + x0]) * (1 - fx)
      + Float(values[y0 * width + x1]) * fx
    let lower = Float(values[y1 * width + x0]) * (1 - fx)
      + Float(values[y1 * width + x1]) * fx
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
      config.ringCapacity >= config.temporalBatchFrames
    else {
      throw NativePreviewError.invalidConfiguration(
        "ring capacity must cover the temporal batch"
      )
    }

    let ring = PixelBufferRing(capacity: config.ringCapacity)
    let decoder = try await ContinuousVideoDecoder(
      input: URL(fileURLWithPath: config.input),
      startNanoseconds: config.startNanoseconds,
      ring: ring
    )
    let video = try await decoder.description()
    let detector = try await CoreAIDetector(
      modelURL: URL(fileURLWithPath: config.detectionModel),
      candidateChannels: config.detectionCandidateChannels
    )
    let restorer = try VariableRestorerBridge(
      runner: URL(fileURLWithPath: config.restorationRunner),
      models: URL(fileURLWithPath: config.restorationModels),
      maximumFrames: config.temporalBatchFrames
    )
    let processor = try NativeFrameProcessor(
      width: video.width,
      height: video.height,
      restorer: restorer,
      blendFeather: config.blendFeather ?? 1
    )
    let writer = try SegmentWriter(
      outputDirectory: URL(
        fileURLWithPath: config.outputDirectory,
        isDirectory: true
      ),
      width: video.width,
      height: video.height,
      fpsNumerator: video.fpsNumerator,
      fpsDenominator: video.fpsDenominator,
      generation: config.generation,
      segmentSeconds: config.segmentSeconds
    )
    let control = PreviewControl(
      bufferLimitSeconds: config.bufferLimitSeconds
    )
    let wallStart = Date()
    control.runReader()
    emit([
      "kind": "ready",
      "generation": config.generation,
      "duration": video.durationSeconds,
      "fps": Double(video.fpsNumerator) / Double(video.fpsDenominator),
      "width": video.width,
      "height": video.height,
      "segment_seconds": config.segmentSeconds,
      "pipeline": "swift-cvpixelbuffer-ring-coreai",
    ])
    try decoder.start()
    var pendingEncoding: Task<([SegmentEvent], Int), Error>?
    do {
      var nextSequence = 0
      var decodedFrames = 0
      var detectedFrames = 0
      var restoredBatches = 0
      var detectionSeconds = 0.0

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

            func yieldPending() {
              guard !pending.isEmpty else { return }
              availableBatchSlots.wait()
              if control.shouldStop() {
                availableBatchSlots.signal()
                return
              }
              continuation.yield(.batch(pending))
              pending.removeAll(keepingCapacity: true)
            }

            while !control.shouldStop(), let frame = try ring.pop() {
              localDecodedFrames += 1
              let detectionStart = Date()
              let detections = try await detector.detect(
                frame.pixelBuffer,
                confidenceThreshold: config.confidenceThreshold,
                iouThreshold: config.iouThreshold
              )
              localDetectionSeconds += Date().timeIntervalSince(
                detectionStart
              )
              if !detections.isEmpty {
                localDetectedFrames += 1
              }
              pending.append(
                DetectedFrame(frame: frame, detections: detections)
              )
              if pending.count == config.temporalBatchFrames {
                yieldPending()
              }
            }
            if !control.shouldStop() {
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
        case .batch(let batch):
          do {
            let outputs = try processor.process(batch)
            restoredBatches += processor.lastRestoredSceneCount
            if let pendingEncoding {
              let (segments, completedNextSequence) =
                try await pendingEncoding.value
              for segment in segments {
                emitSegment(segment, generation: config.generation)
              }
              nextSequence = completedNextSequence
            }
            let startingSequence = nextSequence
            pendingEncoding = Task.detached(priority: .userInitiated) {
              var completedSegments: [SegmentEvent] = []
              var encodingNextSequence = startingSequence
              for (index, output) in outputs.enumerated() {
                guard control.waitForCapacity(
                  nextSequence: encodingNextSequence,
                  segmentSeconds: config.segmentSeconds
                ) else {
                  break
                }
                if let segment = try await writer.append(
                  pixelBuffer: output,
                  ptsNanoseconds: batch[index].frame.ptsNanoseconds
                ) {
                  completedSegments.append(segment)
                  encodingNextSequence = segment.sequence + 1
                }
              }
              return (completedSegments, encodingNextSequence)
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
        if let pendingEncoding {
          let (segments, completedNextSequence) =
            try await pendingEncoding.value
          for segment in segments {
            emitSegment(segment, generation: config.generation)
          }
          nextSequence = completedNextSequence
        }
        if let segment = try await writer.finish() {
          emitSegment(segment, generation: config.generation)
          nextSequence = segment.sequence + 1
        }
        let elapsed = max(0.001, Date().timeIntervalSince(wallStart))
        emit([
          "kind": "native_stats",
          "generation": config.generation,
          "decoded_frames": decodedFrames,
          "detected_frames": detectedFrames,
          "restored_batches": restoredBatches,
          "detection_seconds": detectionSeconds,
          "preparation_seconds": processor.preparationSeconds,
          "restoration_seconds": processor.restorationSeconds,
          "composition_seconds": processor.compositionSeconds,
          "elapsed_seconds": elapsed,
          "throughput_fps": Double(decodedFrames) / elapsed,
        ])
        emit(["kind": "ended", "generation": config.generation])
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
