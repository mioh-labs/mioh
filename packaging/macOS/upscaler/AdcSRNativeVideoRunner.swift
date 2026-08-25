// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AVFoundation
import CoreAI
import CoreImage
import CoreMedia
import CoreVideo
import Darwin
import Foundation
import Metal
import Vision
import VideoToolbox

// Kept in the standalone upscaler target; Mioh itself does not link this runner.
private let adcSRTileOverlap = 16
private let adcSRLowFrequencyAnchorStrength: Float = 1

private func adcSRRationalFrameRate(_ fps: Double) -> (numerator: Int, denominator: Int) {
  let candidates = [
    (24_000, 1_001), (30_000, 1_001), (60_000, 1_001),
    (24, 1), (25, 1), (30, 1), (50, 1), (60, 1),
  ]
  return candidates.min {
    abs(Double($0.0) / Double($0.1) - fps)
      < abs(Double($1.0) / Double($1.1) - fps)
  } ?? (max(1, Int(fps.rounded())), 1)
}

private func adcSRFrameTime(_ frame: Int, frameRate: Double) -> CMTime {
  let rate = adcSRRationalFrameRate(frameRate)
  return CMTime(value: Int64(frame * rate.denominator), timescale: Int32(rate.numerator))
}

@available(macOS 27.0, *)
private enum AdcSRVideoError: LocalizedError {
  case argument(String)
  case media(String)
  case tensor(String)
  case writer(String)

  var errorDescription: String? {
    switch self {
    case .argument(let message): return message
    case .media(let message): return message
    case .tensor(let message): return message
    case .writer(let message): return message
    }
  }
}

@available(macOS 27.0, *)
private struct AdcSRVideoArguments {
  let input: URL
  let output: URL
  let models: URL
  let outputWidth: Int?
  let outputHeight: Int?
  let computePolicy: AdcSRComputePolicy
  let temporalStrength: Float

  static func parse() throws -> AdcSRVideoArguments {
    let raw = Array(CommandLine.arguments.dropFirst())
    var values: [String: String] = [:]
    var index = 0
    while index < raw.count {
      let key = raw[index]
      guard key.hasPrefix("--"), index + 1 < raw.count else {
        throw AdcSRVideoError.argument("不正な引数です: \(key)")
      }
      values[key] = raw[index + 1]
      index += 2
    }
    guard let input = values["--input"], let output = values["--output"],
          let models = values["--models"] else {
      throw AdcSRVideoError.argument(
        "usage: adcsr-coreai-video --input in.mov --output out.mp4 "
          + "--models adcsr_x4_float32.aimodel [--output-width W --output-height H] "
          + "[--compute gpu|hybrid|automatic|neuralEngine] [--temporal-strength 0...0.5]"
      )
    }
    let width = try parseDimension(values["--output-width"], name: "output-width")
    let height = try parseDimension(values["--output-height"], name: "output-height")
    guard (width == nil) == (height == nil) else {
      throw AdcSRVideoError.argument(
        "--output-width と --output-height は両方指定してください"
      )
    }
    let computeValue = values["--compute"] ?? "gpu"
    guard let compute = AdcSRComputePolicy(rawValue: computeValue) else {
      throw AdcSRVideoError.argument("不正なcompute指定です: \(computeValue)")
    }
    let temporalStrength = Float(values["--temporal-strength"] ?? "0.12") ?? -1
    guard temporalStrength >= 0, temporalStrength <= 0.5 else {
      throw AdcSRVideoError.argument("--temporal-strength は0〜0.5にしてください")
    }
    return AdcSRVideoArguments(
      input: URL(fileURLWithPath: input),
      output: URL(fileURLWithPath: output),
      models: URL(fileURLWithPath: models),
      outputWidth: width,
      outputHeight: height,
      computePolicy: compute,
      temporalStrength: temporalStrength
    )
  }

  private static func parseDimension(_ value: String?, name: String) throws -> Int? {
    guard let value else { return nil }
    guard let number = Int(value), number > 0, number % 2 == 0 else {
      throw AdcSRVideoError.argument("--\(name) は正の偶数にしてください")
    }
    return number
  }
}

@available(macOS 27.0, *)
private struct AdcSRVideoMetadata {
  let asset: AVURLAsset
  let track: AVAssetTrack
  let transform: CGAffineTransform
  let width: Int
  let height: Int
  let frameRate: Double
  let frameTimes: [CMTime]

  var frameCount: Int { frameTimes.count }

  static func load(url: URL) async throws -> AdcSRVideoMetadata {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw AdcSRVideoError.media("入力に映像トラックがありません")
    }
    let natural = try await track.load(.naturalSize)
    let transform = try await track.load(.preferredTransform)
    let oriented = natural.applying(transform)
    let frameRate = max(1, Double(try await track.load(.nominalFrameRate)))
    let frameTimes = try await readFrameTimes(asset: asset, track: track)
    guard !frameTimes.isEmpty else {
      throw AdcSRVideoError.media("デコードできるフレームがありません")
    }
    return AdcSRVideoMetadata(
      asset: asset,
      track: track,
      transform: transform,
      width: max(1, Int(abs(oriented.width).rounded())),
      height: max(1, Int(abs(oriented.height).rounded())),
      frameRate: frameRate,
      frameTimes: frameTimes
    )
  }

  func presentationTime(for frame: Int) -> CMTime {
    guard frameTimes.indices.contains(frame) else {
      return adcSRFrameTime(frame, frameRate: frameRate)
    }
    return frameTimes[frame] - frameTimes[0]
  }

  private static func readFrameTimes(
    asset: AVAsset, track: AVAssetTrack
  ) async throws -> [CMTime] {
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
      ]
    )
    let provider = reader.outputProvider(for: output)
    do {
      try reader.start()
    } catch {
      throw AdcSRVideoError.media(
        "入力を読み込めません: \(error.localizedDescription)"
      )
    }
    var times: [CMTime] = []
    while let sample = try await provider.next() {
      times.append(sample.presentationTimeStamp)
    }
    guard reader.status == .completed else {
      throw AdcSRVideoError.media(
        reader.error?.localizedDescription ?? "フレーム数の取得に失敗しました"
      )
    }
    return times
  }
}

@available(macOS 27.0, *)
private struct AdcSRTile: Hashable {
  let x: Int
  let y: Int
  let validWidth: Int
  let validHeight: Int
  let blendLeft: Int
  let blendRight: Int
  let blendTop: Int
  let blendBottom: Int
}

@available(macOS 27.0, *)
private func adcSRTiles(width: Int, height: Int) -> [AdcSRTile] {
  let side = AdcSRNativePipeline.inputSide
  let step = side - adcSRTileOverlap
  func starts(_ extent: Int) -> [Int] {
    guard extent > side else { return [0] }
    let span = extent - side
    let intervalCount = max(1, Int(ceil(Double(span) / Double(step))))
    return (0...intervalCount).map { index in
      Int((Double(index) * Double(span) / Double(intervalCount)).rounded())
    }
  }
  let horizontal = starts(width)
  let vertical = starts(height)
  return vertical.enumerated().flatMap { yIndex, y in
    horizontal.enumerated().map { xIndex, x in
      AdcSRTile(
        x: x,
        y: y,
        validWidth: min(side, width - x),
        validHeight: min(side, height - y),
        blendLeft: xIndex > 0 ? max(0, horizontal[xIndex - 1] + side - x) : 0,
        blendRight: xIndex + 1 < horizontal.count
          ? max(0, x + side - horizontal[xIndex + 1]) : 0,
        blendTop: yIndex > 0 ? max(0, vertical[yIndex - 1] + side - y) : 0,
        blendBottom: yIndex + 1 < vertical.count
          ? max(0, y + side - vertical[yIndex + 1]) : 0
      )
    }
  }
}

@available(macOS 27.0, *)
private final class AdcSRFrameReader {
  private typealias DynamicSample =
    CMReadySampleBuffer<CMSampleBuffer.DynamicContent>

  private let reader: AVAssetReader
  private let provider: AVAssetReaderOutput.Provider<DynamicSample>

  init(metadata: AdcSRVideoMetadata) throws {
    reader = try AVAssetReader(asset: metadata.asset)
    let output = AVAssetReaderTrackOutput(
      track: metadata.track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
        kCVPixelBufferMetalCompatibilityKey as String: true,
      ]
    )
    provider = reader.outputProvider(for: output)
    do {
      try reader.start()
    } catch {
      throw AdcSRVideoError.media(
        "共有デコードを開始できません: \(error.localizedDescription)"
      )
    }
  }

  func next() async throws -> CVPixelBuffer? {
    guard let sample = try await provider.next() else {
      if reader.status == .failed {
        throw AdcSRVideoError.media(
          reader.error?.localizedDescription ?? "共有デコードに失敗しました"
        )
      }
      return nil
    }
    guard let pixelSample = CMReadySampleBuffer<CVReadOnlyPixelBuffer>(sample)
    else {
      throw AdcSRVideoError.media("デコード済みフレームに画像がありません")
    }
    return pixelSample.content.withUnsafeBuffer { $0 }
  }

  func validateCompleted() throws {
    guard reader.status == .completed else {
      throw AdcSRVideoError.media(
        reader.error?.localizedDescription ?? "入力の末尾までデコードできませんでした"
      )
    }
  }
}

@available(macOS 27.0, *)
private final class AdcSRFramePreparer {
  private let context: CIContext

  init(device: MTLDevice) {
    context = CIContext(mtlDevice: device, options: [
      .cacheIntermediates: false,
      .highQualityDownsample: true,
    ])
  }

  func orient(
    _ source: CVPixelBuffer, metadata: AdcSRVideoMetadata
  ) throws -> CVPixelBuffer {
    var destination: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      metadata.width,
      metadata.height,
      kCVPixelFormatType_32BGRA,
      [
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ] as CFDictionary,
      &destination
    )
    guard status == kCVReturnSuccess, let destination else {
      throw AdcSRVideoError.media("向き補正フレームを確保できません: \(status)")
    }
    var image = CIImage(cvPixelBuffer: source).transformed(by: metadata.transform)
    let extent = image.extent
    image = image.transformed(
      by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
    )
    context.render(
      image,
      to: destination,
      bounds: CGRect(x: 0, y: 0, width: metadata.width, height: metadata.height),
      colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
    )
    return destination
  }

  func tile(_ source: CVPixelBuffer, at tile: AdcSRTile) throws -> [Float] {
    let side = AdcSRNativePipeline.inputSide
    let pixels = side * side
    var result = [Float](repeating: 0, count: 3 * pixels)
    CVPixelBufferLockBaseAddress(source, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(source) else {
      throw AdcSRVideoError.media("共有フレームにCPUアドレスがありません")
    }
    let width = CVPixelBufferGetWidth(source)
    let height = CVPixelBufferGetHeight(source)
    let rowBytes = CVPixelBufferGetBytesPerRow(source)
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    for ty in 0..<side {
      let sourceY = min(tile.y + ty, height - 1)
      for tx in 0..<side {
        let sourceX = min(tile.x + tx, width - 1)
        let offset = sourceY * rowBytes + sourceX * 4
        let pixel = ty * side + tx
        result[pixel] = Float(bytes[offset + 2]) / 127.5 - 1
        result[pixels + pixel] = Float(bytes[offset + 1]) / 127.5 - 1
        result[2 * pixels + pixel] = Float(bytes[offset]) / 127.5 - 1
      }
    }
    return result
  }

  func statistics(_ source: CVPixelBuffer) throws -> (SIMD4<Float>, SIMD4<Float>) {
    CVPixelBufferLockBaseAddress(source, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(source) else {
      throw AdcSRVideoError.media("共有フレームにCPUアドレスがありません")
    }
    let width = CVPixelBufferGetWidth(source)
    let height = CVPixelBufferGetHeight(source)
    let rowBytes = CVPixelBufferGetBytesPerRow(source)
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    var mean = SIMD4<Float>(repeating: 0)
    var squared = SIMD4<Float>(repeating: 0)
    for y in 0..<height {
      for x in 0..<width {
        let offset = y * rowBytes + x * 4
        let values = SIMD4<Float>(
          Float(bytes[offset + 2]) / 127.5 - 1,
          Float(bytes[offset + 1]) / 127.5 - 1,
          Float(bytes[offset]) / 127.5 - 1,
          0
        )
        mean += values
        squared += values * values
      }
    }
    let count = Float(width * height)
    mean /= count
    squared /= count
    var standardDeviation = SIMD4<Float>(repeating: 1)
    for channel in 0..<3 {
      standardDeviation[channel] = sqrt(max(
        1e-6, squared[channel] - mean[channel] * mean[channel]
      ))
    }
    return (mean, standardDeviation)
  }
}

@available(macOS 27.0, *)
private struct AdcSRTemporalContext {
  let flowCurrentToPrevious: CVPixelBuffer
  let currentSource: CVPixelBuffer
  let previousSource: CVPixelBuffer
  let previousOutput: CVPixelBuffer
  let strength: Float
}

@available(macOS 27.0, *)
private final class AdcSROpticalFlow {
  func currentToPrevious(
    current: CVPixelBuffer, previous: CVPixelBuffer
  ) throws -> CVPixelBuffer {
    guard CVPixelBufferGetWidth(current) == CVPixelBufferGetWidth(previous),
          CVPixelBufferGetHeight(current) == CVPixelBufferGetHeight(previous) else {
      throw AdcSRVideoError.media("optical flowの前後フレームサイズが一致しません")
    }
    // Handler image is the current frame and the targeted image is the
    // previous frame. The resulting backward flow can therefore be sampled
    // directly at each current-frame destination pixel.
    let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: previous)
    request.computationAccuracy = .low
    request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
    request.keepNetworkOutput = false
    let handler = VNImageRequestHandler(
      cvPixelBuffer: current, orientation: .up, options: [:]
    )
    try handler.perform([request])
    guard let flow = request.results?.first?.pixelBuffer else {
      throw AdcSRVideoError.media("optical flowを生成できません")
    }
    return flow
  }
}

@available(macOS 27.0, *)
private final class AdcSRMappedCanvas {
  let width: Int
  let height: Int
  let rowBytes: Int
  let byteCount: Int
  let pointer: UnsafeMutableRawPointer

  private let url: URL
  private let descriptor: Int32
  private let metalBuffer: MTLBuffer

  init(width: Int, height: Int, directory: URL, device: MTLDevice) throws {
    let pixels = width.multipliedReportingOverflow(by: height)
    guard width > 0, height > 0, !pixels.overflow else {
      throw AdcSRVideoError.media("合成キャンバスのサイズが不正です")
    }
    let bytes = pixels.partialValue.multipliedReportingOverflow(
      by: 4 * MemoryLayout<Float>.stride
    )
    guard !bytes.overflow else {
      throw AdcSRVideoError.media("合成キャンバスが大きすぎます")
    }
    self.width = width
    self.height = height
    rowBytes = width * 4 * MemoryLayout<Float>.stride
    byteCount = bytes.partialValue
    url = directory.appendingPathComponent("adcsr-canvas-\(UUID().uuidString).rgba32f")
    descriptor = Darwin.open(url.path, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR)
    guard descriptor >= 0 else {
      throw AdcSRVideoError.media(
        "合成ストアを作成できません: \(String(cString: strerror(errno)))"
      )
    }
    guard ftruncate(descriptor, off_t(byteCount)) == 0 else {
      Darwin.close(descriptor)
      throw AdcSRVideoError.media(
        "合成ストアを確保できません: \(String(cString: strerror(errno)))"
      )
    }
    let mapped = mmap(nil, byteCount, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0)
    guard mapped != MAP_FAILED, let mapped else {
      Darwin.close(descriptor)
      throw AdcSRVideoError.media(
        "合成ストアをマップできません: \(String(cString: strerror(errno)))"
      )
    }
    pointer = mapped
    guard let buffer = device.makeBuffer(
      bytesNoCopy: mapped,
      length: byteCount,
      options: .storageModeShared,
      deallocator: nil
    ) else {
      munmap(mapped, byteCount)
      Darwin.close(descriptor)
      throw AdcSRVideoError.media("Metalから合成ストアへアクセスできません")
    }
    metalBuffer = buffer
  }

  func buffer() -> MTLBuffer { metalBuffer }

  func statistics() -> (SIMD4<Float>, SIMD4<Float>) {
    let values = pointer.assumingMemoryBound(to: Float.self)
    let pixels = width * height
    var mean = SIMD4<Float>(repeating: 0)
    var squared = SIMD4<Float>(repeating: 0)
    for pixel in 0..<pixels {
      let offset = pixel * 4
      let weight = max(values[offset + 3], 1e-6)
      let rgb = SIMD4<Float>(
        values[offset] / weight,
        values[offset + 1] / weight,
        values[offset + 2] / weight,
        0
      )
      mean += rgb
      squared += rgb * rgb
    }
    let count = Float(pixels)
    mean /= count
    squared /= count
    var standardDeviation = SIMD4<Float>(repeating: 1)
    for channel in 0..<3 {
      standardDeviation[channel] = sqrt(max(
        1e-6, squared[channel] - mean[channel] * mean[channel]
      ))
    }
    return (mean, standardDeviation)
  }

  func render(
    to destination: CVPixelBuffer,
    sourceMean: SIMD4<Float>,
    sourceStandardDeviation: SIMD4<Float>,
    outputMean: SIMD4<Float>,
    outputStandardDeviation: SIMD4<Float>,
    temporal: AdcSRTemporalContext?
  ) throws {
    guard CVPixelBufferGetWidth(destination) == width,
          CVPixelBufferGetHeight(destination) == height else {
      throw AdcSRVideoError.writer("色合わせ先のサイズが一致しません")
    }
    CVPixelBufferLockBaseAddress(destination, [])
    defer { CVPixelBufferUnlockBaseAddress(destination, []) }
    guard let destinationBase = CVPixelBufferGetBaseAddress(destination) else {
      throw AdcSRVideoError.writer("色合わせ先にCPUアドレスがありません")
    }
    let destinationBytes = destinationBase.assumingMemoryBound(to: UInt8.self)
    let destinationRowBytes = CVPixelBufferGetBytesPerRow(destination)
    let values = pointer.assumingMemoryBound(to: Float.self)
    var temporalBuffers: (
      current: UnsafeMutableRawPointer,
      previous: UnsafeMutableRawPointer,
      output: UnsafeMutableRawPointer,
      flow: UnsafeMutableRawPointer
    )?
    if let temporal {
      CVPixelBufferLockBaseAddress(temporal.currentSource, .readOnly)
      CVPixelBufferLockBaseAddress(temporal.previousSource, .readOnly)
      CVPixelBufferLockBaseAddress(temporal.previousOutput, .readOnly)
      CVPixelBufferLockBaseAddress(temporal.flowCurrentToPrevious, .readOnly)
      guard let current = CVPixelBufferGetBaseAddress(temporal.currentSource),
            let previous = CVPixelBufferGetBaseAddress(temporal.previousSource),
            let output = CVPixelBufferGetBaseAddress(temporal.previousOutput),
            let flow = CVPixelBufferGetBaseAddress(temporal.flowCurrentToPrevious) else {
        CVPixelBufferUnlockBaseAddress(temporal.flowCurrentToPrevious, .readOnly)
        CVPixelBufferUnlockBaseAddress(temporal.previousOutput, .readOnly)
        CVPixelBufferUnlockBaseAddress(temporal.previousSource, .readOnly)
        CVPixelBufferUnlockBaseAddress(temporal.currentSource, .readOnly)
        throw AdcSRVideoError.media("時間方向安定化バッファにアクセスできません")
      }
      temporalBuffers = (current, previous, output, flow)
    }
    defer {
      if let temporal, temporalBuffers != nil {
        CVPixelBufferUnlockBaseAddress(temporal.flowCurrentToPrevious, .readOnly)
        CVPixelBufferUnlockBaseAddress(temporal.previousOutput, .readOnly)
        CVPixelBufferUnlockBaseAddress(temporal.previousSource, .readOnly)
        CVPixelBufferUnlockBaseAddress(temporal.currentSource, .readOnly)
      }
    }
    var gain = SIMD4<Float>(repeating: 1)
    for channel in 0..<3 {
      gain[channel] = sourceStandardDeviation[channel]
        / max(outputStandardDeviation[channel], 1e-6)
    }
    for y in 0..<height {
      for x in 0..<width {
        let pixel = y * width + x
        let source = pixel * 4
        let weight = max(values[source + 3], 1e-6)
        var rgb = SIMD4<Float>(repeating: 0)
        for channel in 0..<3 {
          let raw = values[source + channel] / weight
          rgb[channel] = (raw - outputMean[channel]) * gain[channel]
            + sourceMean[channel]
        }
        if let temporal, let buffers = temporalBuffers {
          let lowX = (Float(x) + 0.5) / 4 - 0.5
          let lowY = (Float(y) + 0.5) / 4 - 0.5
          let flow = Self.flow(
            buffers.flow,
            rowBytes: CVPixelBufferGetBytesPerRow(temporal.flowCurrentToPrevious),
            width: CVPixelBufferGetWidth(temporal.flowCurrentToPrevious),
            height: CVPixelBufferGetHeight(temporal.flowCurrentToPrevious),
            x: lowX,
            y: lowY
          )
          let previousX = lowX + flow.x
          let previousY = lowY + flow.y
          let sourceWidth = CVPixelBufferGetWidth(temporal.previousSource)
          let sourceHeight = CVPixelBufferGetHeight(temporal.previousSource)
          if previousX >= 0, previousY >= 0,
             previousX <= Float(sourceWidth - 1),
             previousY <= Float(sourceHeight - 1) {
            let currentLow = Self.sampleRGB(
              buffers.current,
              rowBytes: CVPixelBufferGetBytesPerRow(temporal.currentSource),
              width: sourceWidth,
              height: sourceHeight,
              x: lowX,
              y: lowY
            )
            let previousLow = Self.sampleRGB(
              buffers.previous,
              rowBytes: CVPixelBufferGetBytesPerRow(temporal.previousSource),
              width: sourceWidth,
              height: sourceHeight,
              x: previousX,
              y: previousY
            )
            let previousHigh = Self.sampleRGB(
              buffers.output,
              rowBytes: CVPixelBufferGetBytesPerRow(temporal.previousOutput),
              width: CVPixelBufferGetWidth(temporal.previousOutput),
              height: CVPixelBufferGetHeight(temporal.previousOutput),
              x: (previousX + 0.5) * 4 - 0.5,
              y: (previousY + 0.5) * 4 - 0.5
            )
            let difference = (
              abs(currentLow.x - previousLow.x)
                + abs(currentLow.y - previousLow.y)
                + abs(currentLow.z - previousLow.z)
            ) / 3
            let confidence = max(0, min(1, 1 - difference * 1.5))
            for channel in 0..<3 {
              let previousResidual = max(
                -0.35,
                min(0.35, previousHigh[channel] - previousLow[channel])
              )
              let currentResidual = max(
                -0.35,
                min(0.35, rgb[channel] - currentLow[channel])
              )
              rgb[channel] += temporal.strength * confidence
                * (previousResidual - currentResidual)
            }
          }
        }
        let destination = y * destinationRowBytes + x * 4
        destinationBytes[destination] = Self.byte(rgb[2])
        destinationBytes[destination + 1] = Self.byte(rgb[1])
        destinationBytes[destination + 2] = Self.byte(rgb[0])
        destinationBytes[destination + 3] = 255
      }
    }
  }

  private static func byte(_ normalized: Float) -> UInt8 {
    UInt8(max(0, min(255, ((normalized * 0.5 + 0.5) * 255).rounded())))
  }

  private static func flow(
    _ base: UnsafeMutableRawPointer,
    rowBytes: Int,
    width: Int,
    height: Int,
    x: Float,
    y: Float
  ) -> SIMD2<Float> {
    let ix = max(0, min(width - 1, Int(x.rounded())))
    let iy = max(0, min(height - 1, Int(y.rounded())))
    let row = base.advanced(by: iy * rowBytes).assumingMemoryBound(to: Float.self)
    return SIMD2<Float>(row[ix * 2], row[ix * 2 + 1])
  }

  private static func sampleRGB(
    _ base: UnsafeMutableRawPointer,
    rowBytes: Int,
    width: Int,
    height: Int,
    x: Float,
    y: Float
  ) -> SIMD4<Float> {
    let clampedX = max(0, min(Float(width - 1), x))
    let clampedY = max(0, min(Float(height - 1), y))
    let x0 = Int(floor(clampedX))
    let y0 = Int(floor(clampedY))
    let x1 = min(width - 1, x0 + 1)
    let y1 = min(height - 1, y0 + 1)
    let tx = clampedX - Float(x0)
    let ty = clampedY - Float(y0)
    func pixel(_ px: Int, _ py: Int) -> SIMD4<Float> {
      let bytes = base.advanced(by: py * rowBytes + px * 4)
        .assumingMemoryBound(to: UInt8.self)
      return SIMD4<Float>(
        Float(bytes[2]) / 127.5 - 1,
        Float(bytes[1]) / 127.5 - 1,
        Float(bytes[0]) / 127.5 - 1,
        0
      )
    }
    let top = pixel(x0, y0) * (1 - tx) + pixel(x1, y0) * tx
    let bottom = pixel(x0, y1) * (1 - tx) + pixel(x1, y1) * tx
    return top * (1 - ty) + bottom * ty
  }

  deinit {
    munmap(pointer, byteCount)
    Darwin.close(descriptor)
    try? FileManager.default.removeItem(at: url)
  }
}

@available(macOS 27.0, *)
private final class AdcSRMetalCompositor {
  private static let source = #"""
  #include <metal_stdlib>
  using namespace metal;

  inline float sample_planar_128(
      device const float *values,
      uint channel,
      float2 location) {
    constexpr uint side = 128;
    constexpr uint plane = side * side;
    const float2 point = clamp(location, float2(0.0f), float2(127.0f));
    const uint2 lower = uint2(floor(point));
    const uint2 upper = min(lower + uint2(1), uint2(127));
    const float2 fraction = point - float2(lower);
    const uint offset = channel * plane;
    const float top = mix(
      values[offset + lower.y * side + lower.x],
      values[offset + lower.y * side + upper.x],
      fraction.x
    );
    const float bottom = mix(
      values[offset + upper.y * side + lower.x],
      values[offset + upper.y * side + upper.x],
      fraction.x
    );
    return mix(top, bottom, fraction.y);
  }

  inline float raised_cosine(float position, uint extent) {
    if (extent == 0) return 1.0f;
    const float phase = clamp(position / float(extent), 0.0f, 1.0f);
    return 0.5f - 0.5f * cos(3.14159265358979323846f * phase);
  }

  kernel void downsample_adcsr(
      device const float *sr [[buffer(0)]],
      device float *low [[buffer(1)]],
      uint2 gid [[thread_position_in_grid]]) {
    constexpr uint lowSide = 128;
    constexpr uint highSide = 512;
    if (gid.x >= lowSide || gid.y >= lowSide) return;
    const uint lowPlane = lowSide * lowSide;
    const uint highPlane = highSide * highSide;
    for (uint channel = 0; channel < 3; ++channel) {
      float sum = 0.0f;
      const uint highX = gid.x * 4;
      const uint highY = gid.y * 4;
      for (uint dy = 0; dy < 4; ++dy) {
        for (uint dx = 0; dx < 4; ++dx) {
          sum += sr[channel * highPlane + (highY + dy) * highSide + highX + dx];
        }
      }
      low[channel * lowPlane + gid.y * lowSide + gid.x] = sum * (1.0f / 16.0f);
    }
  }

  kernel void accumulate_adcsr(
      device const float *sr [[buffer(0)]],
      device const float *srLow [[buffer(1)]],
      device const float *lr [[buffer(2)]],
      device float4 *canvas [[buffer(3)]],
      constant uint4 &layout [[buffer(4)]],
      constant uint2 &validSize [[buffer(5)]],
      constant uint4 &blendMargins [[buffer(6)]],
      constant float &lowFrequencyAnchorStrength [[buffer(7)]],
      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= validSize.x || gid.y >= validSize.y) return;
    const uint tileSide = 512;
    const uint plane = tileSide * tileSide;
    const uint tilePixel = gid.y * tileSide + gid.x;
    float wx = 1.0f;
    float wy = 1.0f;
    if (gid.x < blendMargins.x) {
      wx = raised_cosine(float(gid.x) + 0.5f, blendMargins.x);
    }
    if (gid.x >= tileSide - blendMargins.y) {
      wx = min(
        wx,
        raised_cosine(float(tileSide - gid.x) - 0.5f, blendMargins.y)
      );
    }
    if (gid.y < blendMargins.z) {
      wy = raised_cosine(float(gid.y) + 0.5f, blendMargins.z);
    }
    if (gid.y >= tileSide - blendMargins.w) {
      wy = min(
        wy,
        raised_cosine(float(tileSide - gid.y) - 0.5f, blendMargins.w)
      );
    }
    const float weight = wx * wy;
    const uint destination = (layout.y + gid.y) * layout.z + layout.x + gid.x;
    const float2 lowLocation = (float2(gid) + 0.5f) * 0.25f - 0.5f;
    float3 rgb = float3(sr[tilePixel], sr[plane + tilePixel], sr[2 * plane + tilePixel]);
    const float3 modelLow = float3(
      sample_planar_128(srLow, 0, lowLocation),
      sample_planar_128(srLow, 1, lowLocation),
      sample_planar_128(srLow, 2, lowLocation)
    );
    const float3 sourceLow = float3(
      sample_planar_128(lr, 0, lowLocation),
      sample_planar_128(lr, 1, lowLocation),
      sample_planar_128(lr, 2, lowLocation)
    );
    rgb += lowFrequencyAnchorStrength * (sourceLow - modelLow);
    canvas[destination] += float4(rgb * weight, weight);
  }
  """#

  let device: MTLDevice
  let imageContext: CIContext
  private let queue: MTLCommandQueue
  private let downsample: MTLComputePipelineState
  private let accumulation: MTLComputePipelineState

  init() throws {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
      throw AdcSRVideoError.media("Metalを利用できません")
    }
    let library = try device.makeLibrary(source: Self.source, options: nil)
    guard let downsampleFunction = library.makeFunction(name: "downsample_adcsr"),
          let accumulationFunction = library.makeFunction(name: "accumulate_adcsr") else {
      throw AdcSRVideoError.media("AdcSR Metal合成関数がありません")
    }
    self.device = device
    self.queue = queue
    downsample = try device.makeComputePipelineState(function: downsampleFunction)
    accumulation = try device.makeComputePipelineState(function: accumulationFunction)
    imageContext = CIContext(mtlDevice: device, options: [
      .cacheIntermediates: false,
      .highQualityDownsample: true,
    ])
  }

  func add(
    output: NDArray, input: [Float], tile: AdcSRTile, canvas: AdcSRMappedCanvas
  ) throws {
    guard output.shape == [1, 3, 512, 512], output.scalarType == .float32 else {
      throw AdcSRVideoError.tensor("AdcSRタイル出力の形状または型が不正です")
    }
    guard input.count == 3 * 128 * 128 else {
      throw AdcSRVideoError.tensor("AdcSRタイル入力の形状が不正です")
    }
    let outputBytes = 3 * 512 * 512 * MemoryLayout<Float>.stride
    let lowBytes = 3 * 128 * 128 * MemoryLayout<Float>.stride
    let view = output.view(as: Float.self)
    try view.withUnsafePointer { pointer, _, _ in
      guard let outputBuffer = device.makeBuffer(
        bytes: pointer, length: outputBytes, options: .storageModeShared
      ), let inputBuffer = device.makeBuffer(
        bytes: input, length: lowBytes, options: .storageModeShared
      ), let lowBuffer = device.makeBuffer(
        length: lowBytes, options: .storageModePrivate
      ), let command = queue.makeCommandBuffer(),
        let lowEncoder = command.makeComputeCommandEncoder() else {
        throw AdcSRVideoError.media("AdcSR Metal合成バッファを作成できません")
      }
      lowEncoder.setComputePipelineState(downsample)
      lowEncoder.setBuffer(outputBuffer, offset: 0, index: 0)
      lowEncoder.setBuffer(lowBuffer, offset: 0, index: 1)
      lowEncoder.dispatchThreads(
        MTLSize(width: 128, height: 128, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
      )
      lowEncoder.endEncoding()
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw AdcSRVideoError.media("AdcSR Metal合成エンコーダーを作成できません")
      }
      encoder.setComputePipelineState(accumulation)
      encoder.setBuffer(outputBuffer, offset: 0, index: 0)
      encoder.setBuffer(lowBuffer, offset: 0, index: 1)
      encoder.setBuffer(inputBuffer, offset: 0, index: 2)
      encoder.setBuffer(canvas.buffer(), offset: 0, index: 3)
      var layout = SIMD4<UInt32>(
        UInt32(tile.x * 4), UInt32(tile.y * 4), UInt32(canvas.width), UInt32(canvas.height)
      )
      var validSize = SIMD2<UInt32>(
        UInt32(tile.validWidth * 4), UInt32(tile.validHeight * 4)
      )
      var blendMargins = SIMD4<UInt32>(
        UInt32(tile.blendLeft * 4), UInt32(tile.blendRight * 4),
        UInt32(tile.blendTop * 4), UInt32(tile.blendBottom * 4)
      )
      var anchorStrength = adcSRLowFrequencyAnchorStrength
      encoder.setBytes(&layout, length: MemoryLayout.size(ofValue: layout), index: 4)
      encoder.setBytes(&validSize, length: MemoryLayout.size(ofValue: validSize), index: 5)
      encoder.setBytes(
        &blendMargins, length: MemoryLayout.size(ofValue: blendMargins), index: 6
      )
      encoder.setBytes(
        &anchorStrength, length: MemoryLayout.size(ofValue: anchorStrength), index: 7
      )
      encoder.dispatchThreads(
        MTLSize(width: Int(validSize.x), height: Int(validSize.y), depth: 1),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
      )
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      withExtendedLifetime((outputBuffer, inputBuffer, lowBuffer)) {}
      if command.status == .error {
        throw AdcSRVideoError.media(
          command.error?.localizedDescription ?? "AdcSR Metal合成に失敗しました"
        )
      }
    }
  }
}

@available(macOS 27.0, *)
private final class AdcSRVideoWriter {
  private let writer: AVAssetWriter
  private let receiver: AVAssetWriterInput.PixelBufferReceiver
  private let imageContext: CIContext
  private let fpsNumerator: Int
  private let fpsDenominator: Int
  private var lastPresentationTime: CMTime?
  private(set) var frameCount = 0

  let width: Int
  let height: Int

  init(
    url: URL, width: Int, height: Int, frameRate: Double,
    bitRate: Int, imageContext: CIContext
  ) throws {
    self.width = width
    self.height = height
    self.imageContext = imageContext
    let rate = adcSRRationalFrameRate(frameRate)
    fpsNumerator = rate.numerator
    fpsDenominator = rate.denominator
    writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let settings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoEncoderSpecificationKey: [
        kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder as String: true,
        kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String: false,
      ],
      AVVideoColorPropertiesKey: [
        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
      ],
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: bitRate,
        AVVideoExpectedSourceFrameRateKey: frameRate,
        AVVideoAllowFrameReorderingKey: false,
        AVVideoMaxKeyFrameIntervalKey: max(1, Int(frameRate * 2)),
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      ],
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.mediaTimeScale = 120_000
    writer.movieTimeScale = 120_000
    var attributes = CVPixelBufferCreationAttributes(
      pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA),
      size: CVImageSize(width: width, height: height)
    )
    attributes.backing = .ioSurface
    receiver = writer.inputPixelBufferReceiver(
      for: input,
      pixelBufferAttributes: attributes
    )
    do {
      try writer.start()
    } catch {
      throw AdcSRVideoError.writer(
        "動画の書き出しを開始できません: \(error.localizedDescription)"
      )
    }
    writer.startSession(atSourceTime: .zero)
  }

  func makeNativePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
    var value: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
      [
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ] as CFDictionary,
      &value
    )
    guard status == kCVReturnSuccess, let value else {
      throw AdcSRVideoError.writer("ネイティブ出力フレームを確保できません: \(status)")
    }
    return value
  }

  private func makeOutputPixelBuffer() throws -> CVPixelBuffer {
    guard let pool = receiver.pixelBufferPool else {
      throw AdcSRVideoError.writer("動画ライターのピクセルバッファプールがありません")
    }
    return try pool.makeMutablePixelBuffer().withUnsafeBuffer { $0 }
  }

  func append(native: CVPixelBuffer, presentationTime requested: CMTime) async throws {
    let destination = try makeOutputPixelBuffer()
    let nativeWidth = CVPixelBufferGetWidth(native)
    let nativeHeight = CVPixelBufferGetHeight(native)
    if nativeWidth == width, nativeHeight == height {
      CVPixelBufferLockBaseAddress(native, .readOnly)
      CVPixelBufferLockBaseAddress(destination, [])
      guard let sourceBase = CVPixelBufferGetBaseAddress(native),
            let destinationBase = CVPixelBufferGetBaseAddress(destination) else {
        CVPixelBufferUnlockBaseAddress(destination, [])
        CVPixelBufferUnlockBaseAddress(native, .readOnly)
        throw AdcSRVideoError.writer("出力フレームにCPUアドレスがありません")
      }
      let sourceRowBytes = CVPixelBufferGetBytesPerRow(native)
      let destinationRowBytes = CVPixelBufferGetBytesPerRow(destination)
      for y in 0..<height {
        memcpy(
          destinationBase.advanced(by: y * destinationRowBytes),
          sourceBase.advanced(by: y * sourceRowBytes),
          width * 4
        )
      }
      CVPixelBufferUnlockBaseAddress(destination, [])
      CVPixelBufferUnlockBaseAddress(native, .readOnly)
    } else {
      let filter = CIFilter(name: "CILanczosScaleTransform")
      filter?.setValue(CIImage(cvPixelBuffer: native), forKey: kCIInputImageKey)
      let verticalScale = Double(height) / Double(nativeHeight)
      let horizontalScale = Double(width) / Double(nativeWidth)
      filter?.setValue(verticalScale, forKey: kCIInputScaleKey)
      filter?.setValue(horizontalScale / verticalScale, forKey: kCIInputAspectRatioKey)
      guard let image = filter?.outputImage?.cropped(
        to: CGRect(x: 0, y: 0, width: width, height: height)
      ) else {
        throw AdcSRVideoError.writer("Lanczosリサイズを作成できません")
      }
      imageContext.render(
        image,
        to: destination,
        bounds: CGRect(x: 0, y: 0, width: width, height: height),
        colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
      )
    }

    var time = requested
    if let lastPresentationTime, time <= lastPresentationTime {
      time = lastPresentationTime + CMTime(
        value: Int64(fpsDenominator), timescale: Int32(fpsNumerator)
      )
    }
    do {
      try await receiver.append(
        CVReadOnlyPixelBuffer(unsafeBuffer: destination),
        with: time
      )
    } catch {
      throw AdcSRVideoError.writer(
        "動画フレームを書き出せません: \(error.localizedDescription)"
      )
    }
    lastPresentationTime = time
    frameCount += 1
  }

  func finish() async throws {
    receiver.finish()
    await writer.finishWriting()
    guard writer.status == .completed else {
      throw AdcSRVideoError.writer(
        writer.error?.localizedDescription ?? "動画の書き出しを完了できません"
      )
    }
  }
}

@available(macOS 27.0, *)
private final class AdcSRProgressReporter {
  private var lastPercent = -1.0

  func emit(_ value: Double) {
    let percent = min(100, max(0, value))
    guard percent >= 100 || percent - lastPercent >= 0.01 else { return }
    lastPercent = percent
    print(String(format: "PROGRESS %.2f", percent))
    fflush(stdout)
  }
}

@available(macOS 27.0, *)
@main
private struct AdcSRNativeVideoRunner {
  static func main() async {
    do {
      let arguments = try AdcSRVideoArguments.parse()
      guard !FileManager.default.fileExists(atPath: arguments.output.path) else {
        throw AdcSRVideoError.argument(
          "出力ファイルはすでに存在します: \(arguments.output.path)"
        )
      }
      print("STAGE 正確なフレーム数とPTSを確認中")
      fflush(stdout)
      let metadata = try await AdcSRVideoMetadata.load(url: arguments.input)
      let tiles = adcSRTiles(width: metadata.width, height: metadata.height)
      let nativeWidth = metadata.width * AdcSRNativePipeline.scale
      let nativeHeight = metadata.height * AdcSRNativePipeline.scale
      let outputWidth = arguments.outputWidth ?? nativeWidth
      let outputHeight = arguments.outputHeight ?? nativeHeight
      let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
        "adcsr-coreai-\(UUID().uuidString)", isDirectory: true
      )
      try FileManager.default.createDirectory(
        at: temporary, withIntermediateDirectories: true
      )
      defer { try? FileManager.default.removeItem(at: temporary) }

      print(
        "AdcSR Core AI: \(metadata.width)x\(metadata.height), "
          + "\(metadata.frameCount) frames, \(tiles.count) tiles/frame, "
          + "native \(nativeWidth)x\(nativeHeight)"
      )
      print("STAGE Metal合成器を準備中")
      fflush(stdout)
      let metal = try AdcSRMetalCompositor()
      let preparer = AdcSRFramePreparer(device: metal.device)
      print("STAGE AdcSRモデルを読み込み中")
      fflush(stdout)
      let pipeline = try await AdcSRNativePipeline(
        modelLocation: arguments.models,
        computePolicy: arguments.computePolicy
      )
      print("Compute policy: \(pipeline.computeSummary)")
      fflush(stdout)
      let bitRate = Int(min(160_000_000, max(
        12_000_000,
        Double(outputWidth * outputHeight) * metadata.frameRate * 0.12
      )))
      let writer = try AdcSRVideoWriter(
        url: arguments.output,
        width: outputWidth,
        height: outputHeight,
        frameRate: metadata.frameRate,
        bitRate: bitRate,
        imageContext: metal.imageContext
      )
      let reader = try AdcSRFrameReader(metadata: metadata)
      let progress = AdcSRProgressReporter()
      let opticalFlow = AdcSROpticalFlow()
      var previousSource: CVPixelBuffer?
      var previousOutput: CVPixelBuffer?
      var frameIndex = 0
      while let decoded = try await reader.next() {
        guard frameIndex < metadata.frameCount else {
          throw AdcSRVideoError.media(
            "実デコード数が事前確認したフレーム数を上回りました"
          )
        }
        print("FRAME \(frameIndex + 1)/\(metadata.frameCount)")
        print("STAGE フレーム \(frameIndex + 1)/\(metadata.frameCount)を共有デコード中")
        fflush(stdout)
        let source = try preparer.orient(decoded, metadata: metadata)
        let sourceStatistics = try preparer.statistics(source)
        var temporal: AdcSRTemporalContext?
        if arguments.temporalStrength > 0,
           let previousSource,
           let previousOutput {
          print(
            "STAGE フレーム \(frameIndex + 1)/\(metadata.frameCount)のoptical flowを計算中"
          )
          fflush(stdout)
          let flow = try opticalFlow.currentToPrevious(
            current: source, previous: previousSource
          )
          temporal = AdcSRTemporalContext(
            flowCurrentToPrevious: flow,
            currentSource: source,
            previousSource: previousSource,
            previousOutput: previousOutput,
            strength: arguments.temporalStrength
          )
        }
        let requiredBytes = Int64(nativeWidth) * Int64(nativeHeight) * 16
        let capacity = try? temporary.resourceValues(
          forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage
        if let capacity, requiredBytes > capacity {
          throw AdcSRVideoError.media(
            "一時ディスク容量が不足しています。必要: "
              + ByteCountFormatter.string(fromByteCount: requiredBytes, countStyle: .file)
          )
        }
        let canvas = try AdcSRMappedCanvas(
          width: nativeWidth,
          height: nativeHeight,
          directory: temporary,
          device: metal.device
        )
        print("STAGE フレーム \(frameIndex + 1)/\(metadata.frameCount)をタイル推論中")
        fflush(stdout)
        for (tileIndex, tile) in tiles.enumerated() {
          print("TILE \(tileIndex + 1)/\(tiles.count)")
          let input = try preparer.tile(source, at: tile)
          let output = try await pipeline.upscale(tile: input)
          try metal.add(output: output, input: input, tile: tile, canvas: canvas)
          let local = 0.02 + 0.90 * Double(tileIndex + 1) / Double(tiles.count)
          progress.emit(
            100 * (Double(frameIndex) + local) / Double(metadata.frameCount)
          )
        }
        print("STAGE フレーム \(frameIndex + 1)/\(metadata.frameCount)を全体色補正・書き込み中")
        fflush(stdout)
        let outputStatistics = canvas.statistics()
        let native = try writer.makeNativePixelBuffer(
          width: nativeWidth, height: nativeHeight
        )
        try canvas.render(
          to: native,
          sourceMean: sourceStatistics.0,
          sourceStandardDeviation: sourceStatistics.1,
          outputMean: outputStatistics.0,
          outputStandardDeviation: outputStatistics.1,
          temporal: temporal
        )
        try await writer.append(
          native: native,
          presentationTime: metadata.presentationTime(for: frameIndex)
        )
        previousSource = source
        previousOutput = native
        frameIndex += 1
        progress.emit(100 * Double(frameIndex) / Double(metadata.frameCount))
      }
      try reader.validateCompleted()
      guard frameIndex == metadata.frameCount else {
        throw AdcSRVideoError.media(
          "フレーム数が一致しません: 入力 \(metadata.frameCount)、出力 \(frameIndex)"
        )
      }
      try await writer.finish()
      guard writer.frameCount == metadata.frameCount else {
        throw AdcSRVideoError.writer(
          "書き出しフレーム数が一致しません: 入力 \(metadata.frameCount)、出力 \(writer.frameCount)"
        )
      }
      progress.emit(100)
      print("STAGE 完了（\(writer.frameCount)/\(metadata.frameCount)フレーム）")
      print("AdcSR Core AI completed: \(arguments.output.path)")
    } catch {
      FileHandle.standardError.write(
        Data("adcsr-coreai-video: \(error.localizedDescription)\n".utf8)
      )
      exit(EXIT_FAILURE)
    }
  }
}
