import CoreVideo
import Foundation
import Metal

private let miohIPadDetectorSize = 640
private let miohIPadRestorationSize = 256

struct MiohIPadDecodedFrame: @unchecked Sendable {
  let pixelBuffer: CVPixelBuffer
  let ptsNanoseconds: Int64
}

/// Owns restored FP16 pixels without requiring the variable Core AI path to
/// duplicate its shared Metal output into a second Swift allocation.
struct MiohIPadRestoredFrames: @unchecked Sendable {
  private enum Storage {
    case values([Float16])
    case metalBuffer(MTLBuffer)
  }

  private let storage: Storage
  let count: Int

  init(_ values: [Float16]) {
    storage = .values(values)
    count = values.count
  }

  init(metalBuffer: MTLBuffer, count: Int) {
    precondition(count >= 0)
    precondition(
      metalBuffer.length >= count * MemoryLayout<Float16>.stride
    )
    storage = .metalBuffer(metalBuffer)
    self.count = count
  }

  func withUnsafeBufferPointer<Result>(
    _ body: (UnsafeBufferPointer<Float16>) throws -> Result
  ) rethrows -> Result {
    switch storage {
    case .values(let values):
      return try values.withUnsafeBufferPointer(body)
    case .metalBuffer(let buffer):
      return try body(
        UnsafeBufferPointer(
          start: buffer.contents().assumingMemoryBound(to: Float16.self),
          count: count
        )
      )
    }
  }

  /// Detaches a shared Core AI output before the single restoration workspace
  /// is handed to the next realtime lane. The copy is only the 256px ROI
  /// tensor; full-resolution IOSurfaces remain shared and are never copied.
  func copiedValues() -> [Float16] {
    withUnsafeBufferPointer { Array($0) }
  }
}

/// The detector produces a hard 640x640 mask. Storing one Float per pixel
/// costs 1.56 MiB for every detection and exhausted the app memory budget as
/// soon as three realtime runners reached a mosaic together. A packed hard
/// mask keeps exactly the same binary information in 50 KiB.
struct MiohIPadBinaryMask: Sendable, Equatable {
  let pixelCount: Int
  private(set) var words: [UInt64]

  init(pixelCount: Int) {
    self.pixelCount = max(0, pixelCount)
    words = [UInt64](
      repeating: 0,
      count: (self.pixelCount + UInt64.bitWidth - 1) / UInt64.bitWidth
    )
  }

  var containsSetPixel: Bool { words.contains { $0 != 0 } }

  mutating func set(_ index: Int) {
    guard index >= 0, index < pixelCount else { return }
    words[index / UInt64.bitWidth] |= UInt64(1) << (index % UInt64.bitWidth)
  }

  func isSet(_ index: Int) -> Bool {
    guard index >= 0, index < pixelCount else { return false }
    return words[index / UInt64.bitWidth]
      & (UInt64(1) << (index % UInt64.bitWidth)) != 0
  }

  mutating func formUnion(_ other: Self) {
    guard pixelCount == other.pixelCount, words.count == other.words.count else {
      return
    }
    for index in words.indices { words[index] |= other.words[index] }
  }
}

struct MiohIPadDetection: Sendable {
  let left: Int
  let top: Int
  let right: Int
  let bottom: Int
  let confidence: Float
  let classIndex: Int
  /// Binary mask in the detector's 640x640 letterboxed coordinate space.
  let detectorMask: MiohIPadBinaryMask
}

struct MiohIPadDetectedFrame: @unchecked Sendable {
  let frame: MiohIPadDecodedFrame
  let detections: [MiohIPadDetection]
}

@available(iOS 27.0, *)
protocol MiohIPadRestoring: AnyObject, Sendable {
  func restore(_ frames: [Float16], frameCount: Int) async throws
    -> MiohIPadRestoredFrames
}

/// Three local-file lanes remain active, but running three BasicVSR++ graphs
/// simultaneously exceeds the iOS process memory budget in mosaic regions.
/// The detector, empty-window bypass, decoding, and encoding stay three-way;
/// only the high-watermark model invocation is serialized. Device evidence
/// showed that even two simultaneous graphs can exceed the process budget.
@available(iOS 27.0, *)
private actor MiohIPadRestorationMemoryGate {
  static let shared = MiohIPadRestorationMemoryGate(limit: 1)

  private var available: Int
  private var waiters: [CheckedContinuation<Void, Never>] = []

  init(limit: Int) {
    available = max(1, limit)
  }

  func acquire() async {
    if available > 0 {
      available -= 1
      return
    }
    await withCheckedContinuation { continuation in
      waiters.append(continuation)
    }
  }

  func release() {
    if waiters.isEmpty {
      available += 1
    } else {
      waiters.removeFirst().resume()
    }
  }
}

private struct MiohIPadIntBox: Sendable, Hashable {
  let left: Int
  let top: Int
  let right: Int
  let bottom: Int

  var width: Int { right - left + 1 }
  var height: Int { bottom - top + 1 }

  func overlaps(_ other: Self) -> Bool {
    left < other.right && other.left < right
      && top < other.bottom && other.top < bottom
  }

  func union(_ other: Self) -> Self {
    Self(
      left: min(left, other.left),
      top: min(top, other.top),
      right: max(right, other.right),
      bottom: max(bottom, other.bottom)
    )
  }
}

private struct MiohIPadSceneFrame: @unchecked Sendable {
  let batchIndex: Int
  let source: CVPixelBuffer
  var box: MiohIPadIntBox
  var detections: [MiohIPadDetection]
}

private struct MiohIPadScene: @unchecked Sendable {
  var frames: [MiohIPadSceneFrame] = []

  var lastBox: MiohIPadIntBox? { frames.last?.box }
  var lastFrameIndex: Int? { frames.last?.batchIndex }

  mutating func add(
    batchIndex: Int,
    source: CVPixelBuffer,
    detection: MiohIPadDetection
  ) {
    let box = MiohIPadIntBox(
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
        MiohIPadSceneFrame(
          batchIndex: batchIndex,
          source: source,
          box: box,
          detections: [detection]
        )
      )
    }
  }
}

private struct MiohIPadClipGeometry: Sendable, Hashable {
  let cropBox: MiohIPadIntBox
  let resizedWidth: Int
  let resizedHeight: Int
  let padTop: Int
  let padLeft: Int
}

private struct MiohIPadSamplingAxis: Sendable {
  let lower: [Int]
  let upper: [Int]
  let fraction: [Float]
}

private struct MiohIPadCompositePlan: Sendable {
  let blendMask: [Float]
  let x: MiohIPadSamplingAxis
  let y: MiohIPadSamplingAxis
}

private struct MiohIPadPreparedScene: @unchecked Sendable {
  let scene: MiohIPadScene
  let geometries: [MiohIPadClipGeometry]
  let compositePlans: [MiohIPadCompositePlan]
  let modelInput: [Float16]
}

/// iPad counterpart of the macOS native Scene -> Clip -> BasicVSR++ -> mask
/// blend path. Each detection track owns per-frame crops while the track uses
/// shared resize factors, preserving motion instead of freezing one square
/// ROI for the whole temporal window.
@available(iOS 27.0, *)
final class MiohIPadFrameProcessor {
  enum ProcessingError: LocalizedError {
    case pixelBuffer(String)
    case restoration(String)

    var errorDescription: String? {
      switch self {
      case .pixelBuffer(let detail): "Pixel buffer処理に失敗しました: \(detail)"
      case .restoration(let detail): "BasicVSR++復元に失敗しました: \(detail)"
      }
    }
  }

  private let outputPool: CVPixelBufferPool
  private let restorer: any MiohIPadRestoring
  private let blendFeather: Float
  private let maximumTrackingGap: Int
  private let detectorX: MiohIPadSamplingAxis
  private let detectorY: MiohIPadSamplingAxis
  private(set) var lastRestoredFrameCount = 0
  private(set) var lastRestorationSeconds = 0.0
  private(set) var lastPreparationSeconds = 0.0
  private(set) var lastCompositingSeconds = 0.0

  init(
    width: Int,
    height: Int,
    restorer: any MiohIPadRestoring,
    blendFeather: Float,
    detectionEmptyLookahead: Int
  ) throws {
    self.restorer = restorer
    self.blendFeather = max(0, blendFeather)
    maximumTrackingGap = max(1, detectionEmptyLookahead + 1)
    (detectorX, detectorY) = Self.makeDetectorSamplingAxes(
      imageWidth: width,
      imageHeight: height
    )
    outputPool = try Self.makePool(width: width, height: height)
  }

  deinit {
    CVPixelBufferPoolFlush(outputPool, .excessBuffers)
  }

  func process(_ detected: [MiohIPadDetectedFrame]) async throws
    -> [CVPixelBuffer]
  {
    lastRestoredFrameCount = 0
    lastRestorationSeconds = 0
    lastPreparationSeconds = 0
    lastCompositingSeconds = 0
    guard !detected.isEmpty else { return [] }
    var outputs = detected.map { $0.frame.pixelBuffer }
    for scene in trackScenes(detected) {
      try Task.checkCancellation()
      let preparationStarted = ContinuousClock.now
      let prepared = try prepare(scene)
      lastPreparationSeconds += Self.seconds(
        preparationStarted.duration(to: .now)
      )

      // Only the shared BasicVSR++ workspace must be serialized. Crop and mask
      // preparation can overlap a different lane's inference. Detach the small
      // 256px ROI result before releasing the permit so the next invocation
      // cannot overwrite it while this lane composites into its IOSurfaces.
      let restored: [Float16]
      await MiohIPadRestorationMemoryGate.shared.acquire()
      do {
        try Task.checkCancellation()
        let restorationStarted = ContinuousClock.now
        let shared = try await restorer.restore(
          prepared.modelInput,
          frameCount: prepared.scene.frames.count
        )
        restored = shared.copiedValues()
        lastRestoredFrameCount += prepared.scene.frames.count
        lastRestorationSeconds += Self.seconds(
          restorationStarted.duration(to: .now)
        )
        await MiohIPadRestorationMemoryGate.shared.release()
      } catch {
        await MiohIPadRestorationMemoryGate.shared.release()
        throw error
      }

      let compositingStarted = ContinuousClock.now
      let elements = 3 * miohIPadRestorationSize * miohIPadRestorationSize
      for index in prepared.scene.frames.indices {
        let frameIndex = prepared.scene.frames[index].batchIndex
        outputs[frameIndex] = try restored.withUnsafeBufferPointer {
          restoredBuffer in
          try composite(
            source: outputs[frameIndex],
            restoredBuffer: restoredBuffer,
            restoredOffset: index * elements,
            geometry: prepared.geometries[index],
            plan: prepared.compositePlans[index]
          )
        }
      }
      lastCompositingSeconds += Self.seconds(
        compositingStarted.duration(to: .now)
      )
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
    else { throw ProcessingError.pixelBuffer("crossfade dimensions") }
    if earlier === later { return earlier }
    let output = try allocateOutput()
    CVPixelBufferLockBaseAddress(earlier, .readOnly)
    CVPixelBufferLockBaseAddress(later, .readOnly)
    CVPixelBufferLockBaseAddress(output, [])
    defer {
      CVPixelBufferUnlockBaseAddress(output, [])
      CVPixelBufferUnlockBaseAddress(later, .readOnly)
      CVPixelBufferUnlockBaseAddress(earlier, .readOnly)
    }
    guard let first = CVPixelBufferGetBaseAddress(earlier),
      let second = CVPixelBufferGetBaseAddress(later),
      let destination = CVPixelBufferGetBaseAddress(output)
    else { throw ProcessingError.pixelBuffer("crossfade base address") }
    let width = CVPixelBufferGetWidth(output)
    let height = CVPixelBufferGetHeight(output)
    let firstStride = CVPixelBufferGetBytesPerRow(earlier)
    let secondStride = CVPixelBufferGetBytesPerRow(later)
    let destinationStride = CVPixelBufferGetBytesPerRow(output)
    let amount = max(0, min(1, laterWeight))
    let firstBytes = first.assumingMemoryBound(to: UInt8.self)
    let secondBytes = second.assumingMemoryBound(to: UInt8.self)
    let outputBytes = destination.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
      for x in 0..<width {
        let firstOffset = y * firstStride + x * 4
        let secondOffset = y * secondStride + x * 4
        let outputOffset = y * destinationStride + x * 4
        for channel in 0..<3 {
          outputBytes[outputOffset + channel] = UInt8(
            max(
              0,
              min(
                255,
                Int(
                  Float(firstBytes[firstOffset + channel]) * (1 - amount)
                    + Float(secondBytes[secondOffset + channel]) * amount
                )
              )
            )
          )
        }
        outputBytes[outputOffset + 3] = 255
      }
    }
    CVBufferPropagateAttachments(earlier, output)
    return output
  }

  private func trackScenes(_ detected: [MiohIPadDetectedFrame])
    -> [MiohIPadScene]
  {
    var scenes: [MiohIPadScene] = []
    for (frameIndex, item) in detected.enumerated() {
      for detection in item.detections {
        let box = MiohIPadIntBox(
          left: detection.left,
          top: detection.top,
          right: detection.right,
          bottom: detection.bottom
        )
        let matchingIndex = scenes.indices.first { index in
          guard let lastBox = scenes[index].lastBox,
            let lastFrame = scenes[index].lastFrameIndex
          else { return false }
          return frameIndex - lastFrame <= maximumTrackingGap
            && lastBox.overlaps(box)
        }
        if let matchingIndex {
          scenes[matchingIndex].add(
            batchIndex: frameIndex,
            source: item.frame.pixelBuffer,
            detection: detection
          )
        } else {
          var scene = MiohIPadScene()
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

  private func prepare(_ scene: MiohIPadScene) throws
    -> MiohIPadPreparedScene
  {
    guard let first = scene.frames.first else {
      throw ProcessingError.restoration("empty track")
    }
    let width = CVPixelBufferGetWidth(first.source)
    let height = CVPixelBufferGetHeight(first.source)
    let cropBoxes = scene.frames.map {
      Self.cropToBox($0.box, imageWidth: width, imageHeight: height)
    }
    let maxWidth = cropBoxes.map(\.width).max() ?? 1
    let maxHeight = cropBoxes.map(\.height).max() ?? 1
    let scaleWidth = Float(miohIPadRestorationSize) / Float(maxWidth)
    let scaleHeight = Float(miohIPadRestorationSize) / Float(maxHeight)
    var geometries: [MiohIPadClipGeometry] = []
    geometries.reserveCapacity(scene.frames.count)
    var compositePlans: [MiohIPadCompositePlan] = []
    compositePlans.reserveCapacity(scene.frames.count)
    let frameElements =
      3 * miohIPadRestorationSize
      * miohIPadRestorationSize
    var modelInput = [Float16](
      repeating: 0,
      count: scene.frames.count * frameElements
    )
    var inputAxes:
      [MiohIPadClipGeometry: (
        x: MiohIPadSamplingAxis, y: MiohIPadSamplingAxis
      )] = [:]
    var compositeAxes:
      [MiohIPadClipGeometry: (
        x: MiohIPadSamplingAxis, y: MiohIPadSamplingAxis
      )] = [:]
    var previousMaskFrame: MiohIPadSceneFrame?
    var previousMaskCrop: MiohIPadIntBox?
    var previousBlendMask: [Float]?
    for index in scene.frames.indices {
      try Task.checkCancellation()
      let cropBox = cropBoxes[index]
      let resizedWidth = max(1, Int(Float(cropBox.width) * scaleWidth))
      let resizedHeight = max(1, Int(Float(cropBox.height) * scaleHeight))
      let geometry = MiohIPadClipGeometry(
        cropBox: cropBox,
        resizedWidth: resizedWidth,
        resizedHeight: resizedHeight,
        padTop: Int(ceil(Double(miohIPadRestorationSize - resizedHeight) / 2)),
        padLeft: Int(ceil(Double(miohIPadRestorationSize - resizedWidth) / 2))
      )
      geometries.append(geometry)
      let axes =
        inputAxes[geometry]
        ?? Self.makeModelInputAxes(
          geometry: geometry,
          sourceWidth: width,
          sourceHeight: height
        )
      inputAxes[geometry] = axes
      try modelInput.withUnsafeMutableBufferPointer { destination in
        try Self.writeModelInput(
          source: scene.frames[index].source,
          axes: axes,
          destination: destination,
          offset: index * frameElements
        )
      }

      let blendMask: [Float]
      if let previousMaskFrame, previousMaskCrop == cropBox,
        Self.hasSameDetectorMask(previousMaskFrame, scene.frames[index]),
        let previousBlendMask
      {
        blendMask = previousBlendMask
      } else {
        blendMask = Self.createBlendMask(
          makeCropMask(scene.frames[index], cropBox: cropBox),
          width: cropBox.width,
          height: cropBox.height,
          feather: blendFeather
        )
        previousMaskFrame = scene.frames[index]
        previousMaskCrop = cropBox
        previousBlendMask = blendMask
      }
      let outputAxes =
        compositeAxes[geometry]
        ?? Self.makeCompositeAxes(
          geometry: geometry
        )
      compositeAxes[geometry] = outputAxes
      compositePlans.append(
        MiohIPadCompositePlan(
          blendMask: blendMask,
          x: outputAxes.x,
          y: outputAxes.y
        )
      )
    }
    return MiohIPadPreparedScene(
      scene: scene,
      geometries: geometries,
      compositePlans: compositePlans,
      modelInput: modelInput
    )
  }

  private static func seconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds)
      + Double(components.attoseconds) / 1_000_000_000_000_000_000
  }

  private static func cropToBox(
    _ input: MiohIPadIntBox,
    imageWidth: Int,
    imageHeight: Int
  ) -> MiohIPadIntBox {
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
        Float(miohIPadRestorationSize) / Float(width),
        Float(miohIPadRestorationSize) / Float(height)
      )
    )
    let missingWidth = max(
      0,
      Int(
        (Float(miohIPadRestorationSize) - Float(width) * downScale)
          / downScale
      )
    )
    let missingHeight = max(
      0,
      Int(
        (Float(miohIPadRestorationSize) - Float(height) * downScale)
          / downScale
      )
    )
    let availableLeft = left
    let availableRight = imageWidth - 1 - right
    let availableTop = top
    let availableBottom = imageHeight - 1 - bottom
    let expandWidthLR = min(
      availableLeft, availableRight, missingWidth / 2, width
    )
    let expandWidthLeft = min(
      availableLeft - expandWidthLR,
      missingWidth - expandWidthLR * 2,
      width - expandWidthLR
    )
    let expandWidthRight = min(
      availableRight - expandWidthLR,
      missingWidth - expandWidthLR * 2 - expandWidthLeft,
      width - expandWidthLR - expandWidthLeft
    )
    let expandHeightTB = min(
      availableTop, availableBottom, missingHeight / 2, height
    )
    let expandHeightTop = min(
      availableTop - expandHeightTB,
      missingHeight - expandHeightTB * 2,
      height - expandHeightTB
    )
    let expandHeightBottom = min(
      availableBottom - expandHeightTB,
      missingHeight - expandHeightTB * 2 - expandHeightTop,
      height - expandHeightTB - expandHeightTop
    )
    left -= Int(floor(Double(expandWidthLR) / 2)) + expandWidthLeft
    right += Int(ceil(Double(expandWidthLR) / 2)) + expandWidthRight
    top -= Int(floor(Double(expandHeightTB) / 2)) + expandHeightTop
    bottom += Int(ceil(Double(expandHeightTB) / 2)) + expandHeightBottom
    return MiohIPadIntBox(
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
      if result < 0 { result = -result }
      if result >= count { result = 2 * count - 2 - result }
    }
    return result
  }

  private static func makeModelInputAxes(
    geometry: MiohIPadClipGeometry,
    sourceWidth: Int,
    sourceHeight: Int
  ) -> (x: MiohIPadSamplingAxis, y: MiohIPadSamplingAxis) {
    func axis(
      count: Int,
      pad: Int,
      resizedCount: Int,
      cropStart: Int,
      cropCount: Int,
      sourceCount: Int
    ) -> MiohIPadSamplingAxis {
      var lower = [Int](repeating: 0, count: count)
      var upper = lower
      var fraction = [Float](repeating: 0, count: count)
      for index in 0..<count {
        let resized = reflected(index - pad, count: resizedCount)
        let source =
          Float(cropStart)
          + (Float(resized) + 0.5) * Float(cropCount)
          / Float(resizedCount) - 0.5
        let clamped = max(0, min(Float(sourceCount - 1), source))
        lower[index] = Int(floor(clamped))
        upper[index] = min(sourceCount - 1, lower[index] + 1)
        fraction[index] = clamped - Float(lower[index])
      }
      return MiohIPadSamplingAxis(
        lower: lower,
        upper: upper,
        fraction: fraction
      )
    }
    return (
      x: axis(
        count: miohIPadRestorationSize,
        pad: geometry.padLeft,
        resizedCount: geometry.resizedWidth,
        cropStart: geometry.cropBox.left,
        cropCount: geometry.cropBox.width,
        sourceCount: sourceWidth
      ),
      y: axis(
        count: miohIPadRestorationSize,
        pad: geometry.padTop,
        resizedCount: geometry.resizedHeight,
        cropStart: geometry.cropBox.top,
        cropCount: geometry.cropBox.height,
        sourceCount: sourceHeight
      )
    )
  }

  private static func writeModelInput(
    source: CVPixelBuffer,
    axes: (x: MiohIPadSamplingAxis, y: MiohIPadSamplingAxis),
    destination output: UnsafeMutableBufferPointer<Float16>,
    offset: Int
  ) throws {
    CVPixelBufferLockBaseAddress(source, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(source) else {
      throw ProcessingError.pixelBuffer("crop base address")
    }
    let rowBytes = CVPixelBufferGetBytesPerRow(source)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    let plane = miohIPadRestorationSize * miohIPadRestorationSize
    guard offset >= 0, output.count >= offset + plane * 3 else {
      throw ProcessingError.restoration("input tensor is too small")
    }
    for y in 0..<miohIPadRestorationSize {
      let upperRow = axes.y.lower[y] * rowBytes
      let lowerRow = axes.y.upper[y] * rowBytes
      let fy = axes.y.fraction[y]
      for x in 0..<miohIPadRestorationSize {
        let leftOffset = axes.x.lower[x] * 4
        let rightOffset = axes.x.upper[x] * 4
        let fx = axes.x.fraction[x]
        @inline(__always)
        func channel(_ channel: Int) -> Float {
          let p00 = Float(pixels[upperRow + leftOffset + channel])
          let p01 = Float(pixels[upperRow + rightOffset + channel])
          let p10 = Float(pixels[lowerRow + leftOffset + channel])
          let p11 = Float(pixels[lowerRow + rightOffset + channel])
          return (p00 * (1 - fx) + p01 * fx) * (1 - fy)
            + (p10 * (1 - fx) + p11 * fx) * fy
        }
        let index = y * miohIPadRestorationSize + x
        output[offset + index] = Float16(channel(2) / 255)
        output[offset + plane + index] = Float16(channel(1) / 255)
        output[offset + 2 * plane + index] = Float16(channel(0) / 255)
      }
    }
  }

  private static func makeDetectorSamplingAxes(
    imageWidth: Int,
    imageHeight: Int
  ) -> (MiohIPadSamplingAxis, MiohIPadSamplingAxis) {
    let scale = min(
      Float(miohIPadDetectorSize) / Float(imageHeight),
      Float(miohIPadDetectorSize) / Float(imageWidth)
    )
    let padX = (Float(miohIPadDetectorSize) - Float(imageWidth) * scale) / 2
    let padY = (Float(miohIPadDetectorSize) - Float(imageHeight) * scale) / 2
    let left = Int(round(padX - 0.1))
    let top = Int(round(padY - 0.1))
    let right = miohIPadDetectorSize - Int(round(padX + 0.1))
    let bottom = miohIPadDetectorSize - Int(round(padY + 0.1))

    func axis(
      sourceCount: Int,
      detectorStart: Int,
      detectorCount: Int
    ) -> MiohIPadSamplingAxis {
      var lower = [Int](repeating: 0, count: sourceCount)
      var upper = lower
      var fraction = [Float](repeating: 0, count: sourceCount)
      for source in 0..<sourceCount {
        let detector =
          Float(detectorStart)
          + (Float(source) + 0.5) * Float(detectorCount)
          / Float(sourceCount) - 0.5
        let clamped = max(
          0,
          min(Float(miohIPadDetectorSize - 1), detector)
        )
        lower[source] = Int(floor(clamped))
        upper[source] = min(miohIPadDetectorSize - 1, lower[source] + 1)
        fraction[source] = clamped - Float(lower[source])
      }
      return MiohIPadSamplingAxis(
        lower: lower,
        upper: upper,
        fraction: fraction
      )
    }
    return (
      axis(
        sourceCount: imageWidth,
        detectorStart: left,
        detectorCount: right - left
      ),
      axis(
        sourceCount: imageHeight,
        detectorStart: top,
        detectorCount: bottom - top
      )
    )
  }

  private func makeCropMask(
    _ frame: MiohIPadSceneFrame,
    cropBox: MiohIPadIntBox
  ) -> [Float] {
    var mask = [Float](repeating: 0, count: cropBox.width * cropBox.height)
    var combinedDetectorMask = MiohIPadBinaryMask(
      pixelCount: miohIPadDetectorSize * miohIPadDetectorSize
    )
    for detection in frame.detections {
      combinedDetectorMask.formUnion(detection.detectorMask)
    }
    for y in 0..<cropBox.height {
      let sourceY = cropBox.top + y
      let upperRow = detectorY.lower[sourceY] * miohIPadDetectorSize
      let lowerRow = detectorY.upper[sourceY] * miohIPadDetectorSize
      let fy = detectorY.fraction[sourceY]
      for x in 0..<cropBox.width {
        let sourceX = cropBox.left + x
        let leftX = detectorX.lower[sourceX]
        let rightX = detectorX.upper[sourceX]
        let fx = detectorX.fraction[sourceX]
        let upper =
          (combinedDetectorMask.isSet(upperRow + leftX) ? 1 - fx : 0)
          + (combinedDetectorMask.isSet(upperRow + rightX) ? fx : 0)
        let lower =
          (combinedDetectorMask.isSet(lowerRow + leftX) ? 1 - fx : 0)
          + (combinedDetectorMask.isSet(lowerRow + rightX) ? fx : 0)
        if upper * (1 - fy) + lower * fy > 0.5 {
          mask[y * cropBox.width + x] = 1
        }
      }
    }
    return mask
  }

  private static func hasSameDetectorMask(
    _ lhs: MiohIPadSceneFrame,
    _ rhs: MiohIPadSceneFrame
  ) -> Bool {
    guard lhs.detections.count == rhs.detections.count else { return false }
    return zip(lhs.detections, rhs.detections).allSatisfy { first, second in
      first.left == second.left && first.top == second.top
        && first.right == second.right && first.bottom == second.bottom
        && first.classIndex == second.classIndex
        && first.detectorMask == second.detectorMask
    }
  }

  private func composite(
    source: CVPixelBuffer,
    restoredBuffer restored: UnsafeBufferPointer<Float16>,
    restoredOffset: Int,
    geometry: MiohIPadClipGeometry,
    plan: MiohIPadCompositePlan
  ) throws -> CVPixelBuffer {
    // Detection and model-input extraction are complete before compositing.
    // The decoded frame is exclusively owned by this batch from this point,
    // so update it in place. Allocating a full-resolution output for every
    // mosaic frame retained an extra segment of buffers per runner; three
    // realtime runners could then exhaust IOSurface/CVPixelBufferPool memory.
    let output = source
    CVPixelBufferLockBaseAddress(output, [])
    defer { CVPixelBufferUnlockBaseAddress(output, []) }
    guard let outputBase = CVPixelBufferGetBaseAddress(output) else {
      throw ProcessingError.pixelBuffer("composite base address")
    }
    let width = CVPixelBufferGetWidth(output)
    let height = CVPixelBufferGetHeight(output)
    let outputRowBytes = CVPixelBufferGetBytesPerRow(output)
    let plane = miohIPadRestorationSize * miohIPadRestorationSize
    guard restoredOffset >= 0, restored.count >= restoredOffset + plane * 3
    else { throw ProcessingError.restoration("output tensor is too small") }
    let destination = outputBase.assumingMemoryBound(to: UInt8.self)
    let cropWidth = geometry.cropBox.width
    let cropHeight = geometry.cropBox.height
    for cropY in 0..<cropHeight {
      let destinationY = geometry.cropBox.top + cropY
      let upperRow = plan.y.lower[cropY] * miohIPadRestorationSize
      let lowerRow = plan.y.upper[cropY] * miohIPadRestorationSize
      let fy = plan.y.fraction[cropY]
      for cropX in 0..<cropWidth {
        let alpha = plan.blendMask[cropY * cropWidth + cropX]
        guard alpha > 0.0001 else { continue }
        let destinationX = geometry.cropBox.left + cropX
        guard destinationX >= 0, destinationX < width,
          destinationY >= 0, destinationY < height
        else { continue }
        let pixel = destination.advanced(
          by: destinationY * outputRowBytes + destinationX * 4
        )
        let leftX = plan.x.lower[cropX]
        let rightX = plan.x.upper[cropX]
        let fx = plan.x.fraction[cropX]
        @inline(__always)
        func sample(_ channel: Int) -> Float {
          let channelOffset = restoredOffset + channel * plane
          let upper =
            Float(restored[channelOffset + upperRow + leftX])
            * (1 - fx)
            + Float(restored[channelOffset + upperRow + rightX]) * fx
          let lower =
            Float(restored[channelOffset + lowerRow + leftX])
            * (1 - fx)
            + Float(restored[channelOffset + lowerRow + rightX]) * fx
          return upper * (1 - fy) + lower * fy
        }
        let blue = sample(2) * 255
        let green = sample(1) * 255
        let red = sample(0) * 255
        pixel[0] = UInt8(max(0, min(255, Int(Float(pixel[0]) * (1 - alpha) + blue * alpha))))
        pixel[1] = UInt8(max(0, min(255, Int(Float(pixel[1]) * (1 - alpha) + green * alpha))))
        pixel[2] = UInt8(max(0, min(255, Int(Float(pixel[2]) * (1 - alpha) + red * alpha))))
        pixel[3] = 255
      }
    }
    return output
  }

  private static func makeCompositeAxes(
    geometry: MiohIPadClipGeometry
  ) -> (x: MiohIPadSamplingAxis, y: MiohIPadSamplingAxis) {
    func axis(
      count: Int,
      pad: Int,
      resizedCount: Int
    ) -> MiohIPadSamplingAxis {
      var lower = [Int](repeating: 0, count: count)
      var upper = lower
      var fraction = [Float](repeating: 0, count: count)
      for index in 0..<count {
        let restored =
          Float(pad)
          + (Float(index) + 0.5) * Float(resizedCount) / Float(count)
          - 0.5
        let clamped = max(
          0,
          min(Float(miohIPadRestorationSize - 1), restored)
        )
        lower[index] = Int(floor(clamped))
        upper[index] = min(miohIPadRestorationSize - 1, lower[index] + 1)
        fraction[index] = clamped - Float(lower[index])
      }
      return MiohIPadSamplingAxis(
        lower: lower,
        upper: upper,
        fraction: fraction
      )
    }
    return (
      x: axis(
        count: geometry.cropBox.width,
        pad: geometry.padLeft,
        resizedCount: geometry.resizedWidth
      ),
      y: axis(
        count: geometry.cropBox.height,
        pad: geometry.padTop,
        resizedCount: geometry.resizedHeight
      )
    )
  }

  private static func createBlendMask(
    _ mask: [Float],
    width: Int,
    height: Int,
    feather: Float
  ) -> [Float] {
    guard width > 0, height > 0, mask.count == width * height else {
      return [Float](repeating: 0, count: max(0, width * height))
    }
    guard feather > 0 else { return mask }
    let innerHeight = Int(Float(height) * 0.95)
    let innerWidth = Int(Float(width) * 0.95)
    let outerHeight = height - innerHeight
    let outerWidth = width - innerWidth
    let borderSize = Int(round(Float(min(outerHeight, outerWidth)) * feather))
    if borderSize < 5 { return [Float](repeating: 1, count: width * height) }
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
        prefix[extendedX + 1] = prefix[extendedX] + blend[y * width + sourceX]
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
  private static func bilinearBinaryScalar(
    _ values: MiohIPadBinaryMask,
    width: Int,
    height: Int,
    x: Float,
    y: Float
  ) -> Float {
    guard values.pixelCount == width * height, width > 0, height > 0 else {
      return 0
    }
    let clampedX = max(0, min(Float(width - 1), x))
    let clampedY = max(0, min(Float(height - 1), y))
    let x0 = Int(floor(clampedX))
    let y0 = Int(floor(clampedY))
    let x1 = min(width - 1, x0 + 1)
    let y1 = min(height - 1, y0 + 1)
    let fx = clampedX - Float(x0)
    let fy = clampedY - Float(y0)
    @inline(__always)
    func value(_ x: Int, _ y: Int) -> Float {
      values.isSet(y * width + x) ? 1 : 0
    }
    let upper = value(x0, y0) * (1 - fx) + value(x1, y0) * fx
    let lower = value(x0, y1) * (1 - fx) + value(x1, y1) * fx
    return upper * (1 - fy) + lower * fy
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
      throw ProcessingError.pixelBuffer("pool creation returned \(status)")
    }
    return pool
  }

  private func allocateOutput() throws -> CVPixelBuffer {
    var output: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      outputPool,
      &output
    )
    guard status == kCVReturnSuccess, let output else {
      throw ProcessingError.pixelBuffer("pool allocation returned \(status)")
    }
    return output
  }
}
