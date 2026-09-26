// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

// Full-frame video adapter for the existing 256px SwiftVR temporal model.
// Models remain external; this process tiles each short temporal scene,
// feathers the tile boundaries, and streams encoded frames to disk.

import AVFoundation
import CoreVideo
import Foundation
import VideoToolbox

private enum VideoError: LocalizedError {
  case invalid(String)
  var errorDescription: String? {
    if case .invalid(let message) = self { return message }
    return nil
  }
}

private struct Arguments {
  let input: URL
  let output: URL
  let models: URL
  let scale: Int

  static func parse() throws -> Self {
    let raw = Array(CommandLine.arguments.dropFirst())
    guard raw.count.isMultiple(of: 2) else {
      throw VideoError.invalid("usage: swiftvr-coreml-video --input FILE --output FILE --models DIR --scale 2|4")
    }
    var options: [String: String] = [:]
    for index in stride(from: 0, to: raw.count, by: 2) {
      options[raw[index]] = raw[index + 1]
    }
    guard let input = options["--input"], let output = options["--output"],
      let models = options["--models"], let scale = Int(options["--scale"] ?? "4"),
      [2, 4].contains(scale) else {
      throw VideoError.invalid("SwiftVR requires --input, --output, --models and --scale 2|4")
    }
    return Self(input: URL(fileURLWithPath: input), output: URL(fileURLWithPath: output),
      models: URL(fileURLWithPath: models, isDirectory: true), scale: scale)
  }
}

private struct VideoFrame {
  let buffer: CVPixelBuffer
  let time: CMTime
}

private final class VideoReader {
  private typealias Sample = CMReadySampleBuffer<CMSampleBuffer.DynamicContent>
  let width: Int
  let height: Int
  let frameRate: Double
  let expectedFrames: Int
  private let reader: AVAssetReader
  private let provider: AVAssetReaderOutput.Provider<Sample>

  init(url: URL) async throws {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw VideoError.invalid("SwiftVR: input has no video track")
    }
    let size = try await track.load(.naturalSize)
    width = Int(size.width.rounded())
    height = Int(size.height.rounded())
    guard width > 0, height > 0 else {
      throw VideoError.invalid("SwiftVR: invalid source dimensions")
    }
    frameRate = max(1, Double(try await track.load(.nominalFrameRate)))
    let duration = try await asset.load(.duration)
    expectedFrames = max(1, Int(ceil(CMTimeGetSeconds(duration) * frameRate)))
    reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
    ])
    provider = reader.outputProvider(for: output)
    try reader.start()
  }

  func next() async throws -> VideoFrame? {
    guard let sample = try await provider.next() else {
      if reader.status == .failed {
        throw VideoError.invalid(reader.error?.localizedDescription ?? "SwiftVR: decode failed")
      }
      return nil
    }
    guard let image = CMReadySampleBuffer<CVReadOnlyPixelBuffer>(sample) else {
      throw VideoError.invalid("SwiftVR: decoded frame has no pixel buffer")
    }
    return VideoFrame(buffer: image.content.withUnsafeBuffer { $0 },
      time: sample.presentationTimeStamp)
  }
}

private func positions(_ length: Int) -> [Int] {
  guard length > 256 else { return [0] }
  var values = Array(stride(from: 0, through: length - 256, by: 192))
  if values.last != length - 256 { values.append(length - 256) }
  return values
}

private func tileInput(_ frames: [VideoFrame], x: Int, y: Int) -> [Float16] {
  let plane = 256 * 256
  var values = [Float16](repeating: 0, count: frames.count * 3 * plane)
  for (frameIndex, frame) in frames.enumerated() {
    let buffer = frame.buffer
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { continue }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    let source = base.assumingMemoryBound(to: UInt8.self)
    let offset = frameIndex * 3 * plane
    for row in 0..<256 {
      let sourceY = min(height - 1, y + row)
      for column in 0..<256 {
        let sourceX = min(width - 1, x + column)
        let pixel = source.advanced(by: sourceY * stride + sourceX * 4)
        let index = row * 256 + column
        values[offset + index] = Float16(Float(pixel[2]) / 255)
        values[offset + plane + index] = Float16(Float(pixel[1]) / 255)
        values[offset + 2 * plane + index] = Float16(Float(pixel[0]) / 255)
      }
    }
  }
  return values
}

private final class TileCanvas {
  let width: Int
  let height: Int
  let frameCount: Int
  private(set) var pixels: [UInt8]
  private var weights: [UInt16]

  init(width: Int, height: Int, frameCount: Int) throws {
    let (area, overflow) = width.multipliedReportingOverflow(by: height)
    guard !overflow, area > 0, frameCount > 0,
      area <= Int.max / (frameCount * 4) else {
      throw VideoError.invalid("SwiftVR: output canvas is too large")
    }
    self.width = width
    self.height = height
    self.frameCount = frameCount
    pixels = [UInt8](repeating: 0, count: area * frameCount * 4)
    weights = [UInt16](repeating: 0, count: area * frameCount)
  }

  func blend(_ output: [Float16], frame: Int, tileX: Int, tileY: Int,
    xPositions: [Int], yPositions: [Int], column: Int, row: Int, scale: Int) {
    let side = 256 * scale
    let plane = side * side
    let left = column == 0 ? 0 : (xPositions[column - 1] + 256 - tileX) * scale
    let right = column + 1 == xPositions.count ? 0
      : (tileX + 256 - xPositions[column + 1]) * scale
    let top = row == 0 ? 0 : (yPositions[row - 1] + 256 - tileY) * scale
    let bottom = row + 1 == yPositions.count ? 0
      : (tileY + 256 - yPositions[row + 1]) * scale
    pixels.withUnsafeMutableBufferPointer { destination in
      weights.withUnsafeMutableBufferPointer { accumulated in
        for localY in 0..<min(side, height - tileY * scale) {
          let yWeight = min(
            top > 0 ? Float(localY + 1) / Float(top + 1) : 1,
            bottom > 0 ? Float(side - localY) / Float(bottom + 1) : 1)
          for localX in 0..<min(side, width - tileX * scale) {
            let xWeight = min(
              left > 0 ? Float(localX + 1) / Float(left + 1) : 1,
              right > 0 ? Float(side - localX) / Float(right + 1) : 1)
            let addition = max(1, Int((xWeight * yWeight * 256).rounded()))
            let target = frame * width * height
              + (tileY * scale + localY) * width + tileX * scale + localX
            let old = Int(accumulated[target])
            let sum = old + addition
            let source = localY * side + localX
            let byte = target * 4
            for (channel, planeIndex) in [(0, 2), (1, 1), (2, 0)] {
              let sample = Float(output[planeIndex * plane + source])
              let value = max(0, min(255, Int((sample * 255).rounded())))
              destination[byte + channel] = UInt8((Int(destination[byte + channel]) * old
                + value * addition + sum / 2) / sum)
            }
            destination[byte + 3] = 255
            accumulated[target] = UInt16(min(Int(UInt16.max), sum))
          }
        }
      }
    }
  }
}

private final class VideoWriter {
  private let writer: AVAssetWriter
  private let receiver: AVAssetWriterInput.PixelBufferReceiver
  private let width: Int
  private let height: Int
  private var lastTime: CMTime?
  private let frameStep: CMTime
  private var firstTime: CMTime?

  init(url: URL, width: Int, height: Int, frameRate: Double) throws {
    self.width = width
    self.height = height
    frameStep = CMTime(seconds: 1 / frameRate, preferredTimescale: 120_000)
    writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let bitrate = Int(min(160_000_000, max(10_000_000,
      Double(width * height) * frameRate * 0.15)))
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: bitrate,
        AVVideoExpectedSourceFrameRateKey: frameRate,
        AVVideoAllowFrameReorderingKey: false,
      ],
    ])
    var attributes = CVPixelBufferCreationAttributes(
      pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA),
      size: CVImageSize(width: width, height: height))
    attributes.backing = .ioSurface
    receiver = writer.inputPixelBufferReceiver(for: input,
      pixelBufferAttributes: attributes)
    try writer.start()
    writer.startSession(atSourceTime: .zero)
  }

  func append(_ canvas: TileCanvas, frame: Int, time: CMTime,
    previous: TileCanvas? = nil, previousFrame: Int = 0, blend: Float = 1) async throws {
    guard let pool = receiver.pixelBufferPool else {
      throw VideoError.invalid("SwiftVR: video pixel buffer pool is unavailable")
    }
    let buffer = try pool.makeMutablePixelBuffer().withUnsafeBuffer { $0 }
    CVPixelBufferLockBaseAddress(buffer, [])
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
      CVPixelBufferUnlockBaseAddress(buffer, [])
      throw VideoError.invalid("SwiftVR: output frame has no address")
    }
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    let target = base.assumingMemoryBound(to: UInt8.self)
    canvas.pixels.withUnsafeBufferPointer { source in
      previous?.pixels.withUnsafeBufferPointer { prior in
        for y in 0..<height {
          let current = source.baseAddress!.advanced(by: (frame * height + y) * width * 4)
          let older = prior.baseAddress!.advanced(by: (previousFrame * height + y) * width * 4)
          let destination = target.advanced(by: y * stride)
          for index in 0..<(width * 4) {
            destination[index] = UInt8((Float(older[index]) * (1 - blend)
              + Float(current[index]) * blend).rounded())
          }
        }
      }
      if previous == nil {
        for y in 0..<height {
          memcpy(target.advanced(by: y * stride),
            source.baseAddress!.advanced(by: (frame * height + y) * width * 4),
            width * 4)
        }
      }
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    if firstTime == nil { firstTime = time }
    var adjusted = time - (firstTime ?? .zero)
    if let lastTime, adjusted <= lastTime { adjusted = lastTime + frameStep }
    try await receiver.append(CVReadOnlyPixelBuffer(unsafeBuffer: buffer),
      with: adjusted)
    lastTime = adjusted
  }

  func finish() async throws {
    receiver.finish()
    await writer.finishWriting()
    guard writer.status == .completed else {
      throw VideoError.invalid(writer.error?.localizedDescription ?? "SwiftVR: final encode failed")
    }
  }
}

@main
private enum SwiftVRNativeVideoRunner {
  static func main() async {
    do {
      try await run()
    } catch {
      FileHandle.standardError.write(Data("swiftvr-coreml-video: \(error.localizedDescription)\n".utf8))
      exit(1)
    }
  }

  private static func run() async throws {
    let args = try Arguments.parse()
    guard !FileManager.default.fileExists(atPath: args.output.path) else {
      throw VideoError.invalid("SwiftVR: output file already exists")
    }
    try SwiftVRROIAssets.validate(args.models, scale: args.scale)
    let cache = try swiftVRPersistentCompiledCache()
    let reader = try await VideoReader(url: args.input)
    let width = reader.width * args.scale
    let height = reader.height * args.scale
    let writer = try VideoWriter(url: args.output, width: width,
      height: height, frameRate: reader.frameRate)
    let xs = positions(reader.width)
    let ys = positions(reader.height)
    let tileCount = xs.count * ys.count
    print("STAGE SwiftVR \(args.scale)x · \(tileCount) tiles · temporal scenes")
    fflush(stdout)

    var carryFrames: [VideoFrame] = []
    var carryOutput: TileCanvas?
    var written = 0
    while true {
      var frames = carryFrames
      var ended = false
      while frames.count < 33 {
        guard let frame = try await reader.next() else { ended = true; break }
        frames.append(frame)
      }
      if frames.isEmpty { break }
      if ended && frames.count == carryFrames.count, let carryOutput {
        for index in 0..<carryFrames.count {
          try await writer.append(carryOutput, frame: 25 + index,
            time: carryFrames[index].time)
          written += 1
        }
        break
      }
      let canvas = try TileCanvas(width: width, height: height,
        frameCount: frames.count)
      for (row, y) in ys.enumerated() {
        for (column, x) in xs.enumerated() {
          try autoreleasepool {
            let input = tileInput(frames, x: x, y: y)
            try input.withUnsafeBufferPointer { values in
              try runSwiftVRScene(root: args.models, compiledCache: cache,
                input: values, frames: frames.count, scale: args.scale,
                deliver: { frame, output in
                  canvas.blend(output, frame: frame, tileX: x, tileY: y,
                    xPositions: xs, yPositions: ys, column: column,
                    row: row, scale: args.scale)
                }, shouldStop: { false })
            }
          }
          let tileIndex = row * xs.count + column + 1
          let progress = min(99, 100 * (Double(written)
            + Double(tileIndex) / Double(tileCount) * Double(frames.count))
            / Double(reader.expectedFrames))
          print(String(format: "PROGRESS %.2f", progress))
          fflush(stdout)
        }
      }
      let writeCount = ended ? frames.count : frames.count - 8
      for index in 0..<writeCount {
        if let carryOutput, index < carryFrames.count {
          try await writer.append(canvas, frame: index, time: frames[index].time,
            previous: carryOutput, previousFrame: 25 + index,
            blend: Float(index + 1) / Float(carryFrames.count + 1))
        } else {
          try await writer.append(canvas, frame: index, time: frames[index].time)
        }
        written += 1
      }
      if ended { break }
      carryFrames = Array(frames.suffix(8))
      carryOutput = canvas
      releaseSwiftVRModels()
    }
    try await writer.finish()
    print("PROGRESS 100")
    print("STAGE SwiftVR completed: \(written) frames")
    fflush(stdout)
  }
}
