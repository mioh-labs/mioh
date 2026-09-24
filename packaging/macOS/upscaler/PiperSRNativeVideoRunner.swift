// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AVFoundation
import CoreImage
import CoreML
import CoreVideo
import Darwin
import Foundation
import Metal
import VideoToolbox

private enum PiperSRError: LocalizedError {
  case invalid(String)
  case media(String)
  case model(String)
  case writer(String)

  var errorDescription: String? {
    switch self {
    case .invalid(let value), .media(let value), .model(let value),
         .writer(let value): return value
    }
  }
}

private struct PiperSRArguments {
  let input: URL
  let output: URL
  let models: URL
  let outputWidth: Int
  let outputHeight: Int
  let sharpness: Float

  static func parse() throws -> Self {
    let raw = Array(CommandLine.arguments.dropFirst())
    guard raw.count.isMultiple(of: 2) else {
      throw PiperSRError.invalid("PiperSRの引数は名前と値の組で指定してください")
    }
    var values: [String: String] = [:]
    for index in stride(from: 0, to: raw.count, by: 2) {
      guard raw[index].hasPrefix("--") else {
        throw PiperSRError.invalid("不正な引数です: \(raw[index])")
      }
      values[raw[index]] = raw[index + 1]
    }
    guard let input = values["--input"], let output = values["--output"],
          let models = values["--models"],
          let width = Int(values["--output-width"] ?? ""),
          let height = Int(values["--output-height"] ?? ""),
          let sharpness = Float(values["--sharpness"] ?? "0"),
          sharpness.isFinite, (0...1).contains(sharpness),
          width > 0, height > 0, width.isMultiple(of: 2), height.isMultiple(of: 2)
    else {
      throw PiperSRError.invalid(
        "usage: pipersr-coreml-video --input in.mov --output out.mp4 "
          + "--models directory --output-width W --output-height H "
          + "[--sharpness 0...1]"
      )
    }
    return Self(
      input: URL(fileURLWithPath: input), output: URL(fileURLWithPath: output),
      models: URL(fileURLWithPath: models), outputWidth: width,
      outputHeight: height, sharpness: sharpness
    )
  }
}

private struct PiperSRMedia {
  let asset: AVURLAsset
  let track: AVAssetTrack
  let transform: CGAffineTransform
  let width: Int
  let height: Int
  let frameRate: Double
  let duration: Double

  static func load(_ url: URL) async throws -> Self {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw PiperSRError.media("映像トラックがありません")
    }
    let natural = try await track.load(.naturalSize)
    let transform = try await track.load(.preferredTransform)
    let oriented = natural.applying(transform)
    let duration = try await asset.load(.duration)
    return Self(
      asset: asset, track: track, transform: transform,
      width: max(1, Int(abs(oriented.width).rounded())),
      height: max(1, Int(abs(oriented.height).rounded())),
      frameRate: max(1, Double(try await track.load(.nominalFrameRate))),
      duration: max(0, CMTimeGetSeconds(duration))
    )
  }
}

private final class PiperSRReader {
  private typealias Sample = CMReadySampleBuffer<CMSampleBuffer.DynamicContent>
  private let reader: AVAssetReader
  private let provider: AVAssetReaderOutput.Provider<Sample>

  init(media: PiperSRMedia) throws {
    reader = try AVAssetReader(asset: media.asset)
    let output = AVAssetReaderTrackOutput(
      track: media.track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
      ]
    )
    provider = reader.outputProvider(for: output)
    try reader.start()
  }

  func next() async throws -> (CVPixelBuffer, CMTime)? {
    guard let sample = try await provider.next() else {
      if reader.status == .failed {
        throw PiperSRError.media(
          reader.error?.localizedDescription ?? "フレームのデコードに失敗しました"
        )
      }
      return nil
    }
    guard let image = CMReadySampleBuffer<CVReadOnlyPixelBuffer>(sample) else {
      throw PiperSRError.media("映像サンプルに画像がありません")
    }
    return (image.content.withUnsafeBuffer { $0 }, sample.presentationTimeStamp)
  }

  func validateCompleted() throws {
    guard reader.status == .completed else {
      throw PiperSRError.media(
        reader.error?.localizedDescription ?? "入力の末尾までデコードできませんでした"
      )
    }
  }
}

private func piperBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
  var value: CVPixelBuffer?
  let result = CVPixelBufferCreate(
    kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
    [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary,
    &value
  )
  guard result == kCVReturnSuccess, let value else {
    throw PiperSRError.media("ピクセルバッファを確保できません: \(result)")
  }
  return value
}

/// Single-frame luminance unsharp mask, gated by a Sobel edge measure. The
/// same 3x3 samples serve the blur and the edge detector, so flat-area grain
/// receives little or no sharpening without an additional image pass.
private final class PiperSRSharpen {
  private let cache: CVMetalTextureCache
  private let queue: MTLCommandQueue
  private let pipeline: MTLComputePipelineState
  private let strength: Float

  init(strength: Float) throws {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
      throw PiperSRError.model("PiperSRのシャープ処理を初期化できません")
    }
    var textureCache: CVMetalTextureCache?
    let status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    guard status == kCVReturnSuccess, let textureCache else {
      throw PiperSRError.model("PiperSRのMetalテクスチャを初期化できません: \(status)")
    }
    let source = """
      #include <metal_stdlib>
      using namespace metal;
      kernel void piperLumaUnsharp(
        texture2d<float, access::read> input [[texture(0)]],
        texture2d<float, access::write> output [[texture(1)]],
        constant float &strength [[buffer(0)]],
        uint2 id [[thread_position_in_grid]]) {
        uint width = input.get_width();
        uint height = input.get_height();
        if (id.x >= width || id.y >= height) return;
        constexpr float weights[3] = {0.259f, 0.482f, 0.259f};
        constexpr float3 lumaWeights = float3(0.299f, 0.587f, 0.114f);
        float4 center = input.read(id);
        float blurred = 0.0f;
        float luminance[3][3];
        for (int dy = -1; dy <= 1; ++dy) {
          uint y = uint(clamp(int(id.y) + dy, 0, int(height) - 1));
          for (int dx = -1; dx <= 1; ++dx) {
            uint x = uint(clamp(int(id.x) + dx, 0, int(width) - 1));
            float value = dot(input.read(uint2(x, y)).rgb, lumaWeights);
            luminance[dy + 1][dx + 1] = value;
            blurred += value * weights[dx + 1] * weights[dy + 1];
          }
        }
        float gx = (luminance[0][2] + 2.0f * luminance[1][2] + luminance[2][2]
                  - luminance[0][0] - 2.0f * luminance[1][0] - luminance[2][0]) * 0.125f;
        float gy = (luminance[2][0] + 2.0f * luminance[2][1] + luminance[2][2]
                  - luminance[0][0] - 2.0f * luminance[0][1] - luminance[0][2]) * 0.125f;
        float edge = smoothstep(0.025f, 0.080f, length(float2(gx, gy)));
        float detail = (luminance[1][1] - blurred) * strength * edge;
        output.write(float4(clamp(center.rgb + detail, 0.0f, 1.0f), center.a), id);
      }
      """
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "piperLumaUnsharp") else {
      throw PiperSRError.model("PiperSRのシャープ処理関数が見つかりません")
    }
    self.cache = textureCache
    self.queue = queue
    self.pipeline = try device.makeComputePipelineState(function: function)
    self.strength = strength
  }

  func apply(to input: CVPixelBuffer) throws -> CVPixelBuffer {
    let width = CVPixelBufferGetWidth(input)
    let height = CVPixelBufferGetHeight(input)
    let output = try piperBuffer(width: width, height: height)
    func texture(_ buffer: CVPixelBuffer) throws -> (CVMetalTexture, MTLTexture) {
      var reference: CVMetalTexture?
      let status = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, cache, buffer, nil, .bgra8Unorm,
        width, height, 0, &reference
      )
      guard status == kCVReturnSuccess, let reference,
            let texture = CVMetalTextureGetTexture(reference) else {
        throw PiperSRError.model("PiperSRのシャープ処理用テクスチャを作成できません: \(status)")
      }
      return (reference, texture)
    }
    let (inputReference, inputTexture) = try texture(input)
    let (outputReference, outputTexture) = try texture(output)
    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
      throw PiperSRError.model("PiperSRのシャープ処理を開始できません")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(inputTexture, index: 0)
    encoder.setTexture(outputTexture, index: 1)
    var amount = strength
    encoder.setBytes(&amount, length: MemoryLayout<Float>.size, index: 0)
    encoder.dispatchThreads(
      MTLSize(width: width, height: height, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    withExtendedLifetime((inputReference, outputReference)) {}
    guard command.status == .completed else {
      throw PiperSRError.model(
        "PiperSRのシャープ処理に失敗しました: \(command.error?.localizedDescription ?? "不明なエラー")"
      )
    }
    return output
  }
}

private final class PiperSRMetalOutput {
  let device: MTLDevice
  private let queue: MTLCommandQueue
  private let pipeline: MTLComputePipelineState

  init() throws {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
      throw PiperSRError.model("PiperSRのMetal出力変換を初期化できません")
    }
    self.device = device
    self.queue = queue
    let source = """
      #include <metal_stdlib>
      using namespace metal;
      kernel void piperSRToBGRA(
        device const half *src [[buffer(0)]],
        device uchar4 *dst [[buffer(1)]],
        constant uint2 &size [[buffer(2)]],
        uint2 id [[thread_position_in_grid]]) {
        if (id.x >= size.x || id.y >= size.y) return;
        uint pixel = id.y * size.x + id.x;
        uint plane = size.x * size.y;
        half3 rgb = clamp(half3(src[pixel], src[pixel + plane],
                                src[pixel + 2 * plane]), 0.0h, 1.0h);
        dst[pixel] = uchar4(uchar(rgb.z * 255.0h), uchar(rgb.y * 255.0h),
                           uchar(rgb.x * 255.0h), 255);
      }
      """
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "piperSRToBGRA") else {
      throw PiperSRError.model("PiperSRのMetal変換関数が見つかりません")
    }
    pipeline = try device.makeComputePipelineState(function: function)
  }

  func submit(
    _ result: MLMultiArray, session: PiperSRFullFrameSession
  ) throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
      throw PiperSRError.model("PiperSRのMetal出力変換を開始できません")
    }
    memcpy(session.metalSource.contents(), result.dataPointer, session.tensorByteCount)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(session.metalSource, offset: 0, index: 0)
    encoder.setBuffer(session.metalDestination, offset: 0, index: 1)
    var size = SIMD2<UInt32>(UInt32(session.outputWidth), UInt32(session.outputHeight))
    encoder.setBytes(&size, length: MemoryLayout.size(ofValue: size), index: 2)
    encoder.dispatchThreads(
      MTLSize(width: session.outputWidth, height: session.outputHeight, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
    )
    encoder.endEncoding()
    command.commit()
    return command
  }
}

private final class PiperSRFullFrameSession {
  let input: MLMultiArray
  let provider: MLDictionaryFeatureProvider
  let orientedInput: CVPixelBuffer
  let output: CVPixelBuffer
  let metalSource: MTLBuffer
  let metalDestination: MTLBuffer
  let outputWidth: Int
  let outputHeight: Int
  let tensorByteCount: Int

  init(width: Int, height: Int, device: MTLDevice) throws {
    input = try MLMultiArray(
      shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
      dataType: .float16
    )
    provider = try MLDictionaryFeatureProvider(dictionary: [
      "input": MLFeatureValue(multiArray: input)
    ])
    orientedInput = try piperBuffer(width: width, height: height)
    outputWidth = width * 2
    outputHeight = height * 2
    output = try piperBuffer(width: outputWidth, height: outputHeight)
    tensorByteCount = outputWidth * outputHeight * 3 * MemoryLayout<Float16>.size
    guard let source = device.makeBuffer(
      length: tensorByteCount, options: .storageModeShared
    ), let destination = device.makeBuffer(
      length: outputWidth * outputHeight * 4, options: .storageModeShared
    ) else {
      throw PiperSRError.model("PiperSRのMetalバッファを確保できません")
    }
    metalSource = source
    metalDestination = destination
  }
}

private struct PiperSRPendingFrame {
  let slot: Int
  let command: MTLCommandBuffer
  let time: CMTime
}

private final class PiperSRProcessor {
  static let tileSide = 256
  static let overlap = 32

  private let videoModel: MLModel?
  private let tileModel: MLModel?
  private let tileProvider: MLDictionaryFeatureProvider?
  private let metalOutput: PiperSRMetalOutput?
  private let fullFrameSessions: [PiperSRFullFrameSession]
  private let fullFrameOutputName: String?
  private let context = CIContext(options: [.cacheIntermediates: false])
  private let sourceWidth: Int
  private let sourceHeight: Int
  private let fullFrame: Bool
  private let tiles: [(x: Int, y: Int)]
  private let tileBuffer: CVPixelBuffer
  private var sums: [Float]
  private var weights: [Float]

  var tileCount: Int { fullFrame ? 1 : tiles.count }
  var isFullFrame: Bool { fullFrame }
  var modeName: String { fullFrame ? "動画固定サイズ・全画面" : "256pxタイル" }

  init(models: URL, width: Int, height: Int) throws {
    sourceWidth = width
    sourceHeight = height
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    let videoURL = models.appendingPathComponent("PiperSR_2x_video_720p_fp16.mlmodelc")
    let tileURL = models.appendingPathComponent("PiperSR_2x_256.mlmodelc")
    guard FileManager.default.fileExists(atPath: videoURL.path),
          FileManager.default.fileExists(atPath: tileURL.path) else {
      throw PiperSRError.model("PiperSRの同梱モデルが見つかりません")
    }
    fullFrame = [
      (640, 360), (854, 480), (1280, 720),
      (360, 640), (480, 854), (720, 1280),
    ].contains { $0.0 == width && $0.1 == height }
    videoModel = fullFrame
      ? try MLModel(contentsOf: videoURL, configuration: configuration) : nil
    fullFrameOutputName = videoModel?.modelDescription.outputDescriptionsByName.keys.first
    if fullFrame {
      let converter = try PiperSRMetalOutput()
      metalOutput = converter
      fullFrameSessions = try (0..<2).map { _ in
        try PiperSRFullFrameSession(width: width, height: height, device: converter.device)
      }
    } else {
      metalOutput = nil
      fullFrameSessions = []
    }
    tileModel = fullFrame
      ? nil : try MLModel(contentsOf: tileURL, configuration: configuration)
    func starts(_ extent: Int) -> [Int] {
      guard extent > Self.tileSide else { return [0] }
      let span = extent - Self.tileSide
      let count = max(1, Int(ceil(Double(span) / Double(Self.tileSide - Self.overlap))))
      return (0...count).map { Int((Double($0) * Double(span) / Double(count)).rounded()) }
    }
    tiles = starts(height).flatMap { y in starts(width).map { (x: $0, y: y) } }
    tileBuffer = try piperBuffer(width: Self.tileSide, height: Self.tileSide)
    tileProvider = tileModel == nil ? nil : try MLDictionaryFeatureProvider(dictionary: [
      "input_image": MLFeatureValue(pixelBuffer: tileBuffer)
    ])
    let pixels = width * height * 4
    sums = fullFrame ? [] : [Float](repeating: 0, count: pixels * 3)
    weights = fullFrame ? [] : [Float](repeating: 0, count: pixels)
  }

  func orient(
    _ decoded: CVPixelBuffer, transform: CGAffineTransform,
    into destination: CVPixelBuffer? = nil
  ) throws -> CVPixelBuffer {
    let destination = try destination ?? piperBuffer(width: sourceWidth, height: sourceHeight)
    var image = CIImage(cvPixelBuffer: decoded).transformed(by: transform)
    let extent = image.extent
    image = image.transformed(
      by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
    )
    context.render(
      image, to: destination,
      bounds: CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight),
      colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
    )
    return destination
  }

  func upscale(_ source: CVPixelBuffer, progress: (Int, Int) -> Void) throws -> CVPixelBuffer {
    return try upscaleTiled(source, progress: progress)
  }

  func prepareFullFrame(
    _ decoded: CVPixelBuffer, transform: CGAffineTransform, slot: Int
  ) throws {
    let session = fullFrameSessions[slot]
    let source = try orient(decoded, transform: transform, into: session.orientedInput)
    let plane = sourceWidth * sourceHeight
    let values = session.input.dataPointer.assumingMemoryBound(to: Float16.self)
    CVPixelBufferLockBaseAddress(source, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(source) else {
      throw PiperSRError.media("入力フレームにCPUアドレスがありません")
    }
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    let row = CVPixelBufferGetBytesPerRow(source)
    for y in 0..<sourceHeight {
      for x in 0..<sourceWidth {
        let sourceIndex = y * row + x * 4
        let index = y * sourceWidth + x
        values[index] = Float16(Float(bytes[sourceIndex + 2]) / 255)
        values[plane + index] = Float16(Float(bytes[sourceIndex + 1]) / 255)
        values[2 * plane + index] = Float16(Float(bytes[sourceIndex]) / 255)
      }
    }
  }

  func predictFullFrame(slot: Int, time: CMTime) throws -> PiperSRPendingFrame {
    guard let videoModel, let metalOutput, let fullFrameOutputName else {
      throw PiperSRError.model("PiperSR動画モデルを読み込めません")
    }
    let session = fullFrameSessions[slot]
    let prediction = try videoModel.prediction(from: session.provider)
    guard let result = prediction.featureValue(for: fullFrameOutputName)?.multiArrayValue,
          result.shape.map(\.intValue) == [1, 3, session.outputHeight, session.outputWidth],
          result.dataType == .float16,
          result.strides.map(\.intValue) == [
            3 * session.outputHeight * session.outputWidth,
            session.outputHeight * session.outputWidth,
            session.outputWidth, 1,
          ] else {
      throw PiperSRError.model("PiperSR動画モデルのFP16出力形状が一致しません")
    }
    let command = try metalOutput.submit(result, session: session)
    return PiperSRPendingFrame(slot: slot, command: command, time: time)
  }

  func finishFullFrame(_ pending: PiperSRPendingFrame) throws -> CVPixelBuffer {
    let command = pending.command
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw PiperSRError.model(
        "PiperSRのMetal変換に失敗しました: \(command.error?.localizedDescription ?? "不明なエラー")"
      )
    }
    let session = fullFrameSessions[pending.slot]
    let destination = session.output
    CVPixelBufferLockBaseAddress(destination, [])
    defer { CVPixelBufferUnlockBaseAddress(destination, []) }
    guard let base = CVPixelBufferGetBaseAddress(destination) else {
      throw PiperSRError.media("出力フレームにCPUアドレスがありません")
    }
    let source = session.metalDestination.contents()
    let rowBytes = session.outputWidth * 4
    let stride = CVPixelBufferGetBytesPerRow(destination)
    for y in 0..<session.outputHeight {
      memcpy(base.advanced(by: y * stride), source.advanced(by: y * rowBytes), rowBytes)
    }
    return destination
  }

  private func upscaleTiled(
    _ source: CVPixelBuffer, progress: (Int, Int) -> Void
  ) throws -> CVPixelBuffer {
    guard let tileModel, let tileProvider else {
      throw PiperSRError.model("PiperSRタイルモデルを読み込めません")
    }
    for index in sums.indices { sums[index] = 0 }
    for index in weights.indices { weights[index] = 0 }
    let outputWidth = sourceWidth * 2
    let outputHeight = sourceHeight * 2
    let sourceRow = CVPixelBufferGetBytesPerRow(source)
    let tileRow = CVPixelBufferGetBytesPerRow(tileBuffer)
    CVPixelBufferLockBaseAddress(source, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
    guard let sourceBase = CVPixelBufferGetBaseAddress(source) else {
      throw PiperSRError.media("入力フレームにCPUアドレスがありません")
    }
    let sourceBytes = sourceBase.assumingMemoryBound(to: UInt8.self)
    for (tileIndex, tile) in tiles.enumerated() {
      CVPixelBufferLockBaseAddress(tileBuffer, [])
      guard let tileBase = CVPixelBufferGetBaseAddress(tileBuffer) else {
        CVPixelBufferUnlockBaseAddress(tileBuffer, [])
        throw PiperSRError.media("PiperSRタイルにCPUアドレスがありません")
      }
      let tileBytes = tileBase.assumingMemoryBound(to: UInt8.self)
      for y in 0..<Self.tileSide {
        let sy = min(tile.y + y, sourceHeight - 1)
        for x in 0..<Self.tileSide {
          let sx = min(tile.x + x, sourceWidth - 1)
          let sourceIndex = sy * sourceRow + sx * 4
          let tileIndex = y * tileRow + x * 4
          for channel in 0..<4 {
            tileBytes[tileIndex + channel] = sourceBytes[sourceIndex + channel]
          }
        }
      }
      CVPixelBufferUnlockBaseAddress(tileBuffer, [])
      let prediction = try tileModel.prediction(from: tileProvider)
      guard let enhanced = prediction.featureValue(for: "output_image")?.imageBufferValue,
            CVPixelBufferGetWidth(enhanced) == 512,
            CVPixelBufferGetHeight(enhanced) == 512,
            CVPixelBufferGetPixelFormatType(enhanced) == kCVPixelFormatType_32BGRA else {
        throw PiperSRError.model("PiperSRタイルモデルの出力が512px BGRAではありません")
      }
      CVPixelBufferLockBaseAddress(enhanced, .readOnly)
      guard let enhancedBase = CVPixelBufferGetBaseAddress(enhanced) else {
        CVPixelBufferUnlockBaseAddress(enhanced, .readOnly)
        throw PiperSRError.model("PiperSR出力にCPUアドレスがありません")
      }
      let enhancedBytes = enhancedBase.assumingMemoryBound(to: UInt8.self)
      let enhancedRow = CVPixelBufferGetBytesPerRow(enhanced)
      let validWidth = min(Self.tileSide, sourceWidth - tile.x) * 2
      let validHeight = min(Self.tileSide, sourceHeight - tile.y) * 2
      let feather = Self.overlap * 2
      for y in 0..<validHeight {
        let oy = tile.y * 2 + y
        let wy = Self.edgeWeight(
          y, length: validHeight, feather: feather,
          leading: tile.y > 0,
          trailing: tile.y + Self.tileSide < sourceHeight
        )
        for x in 0..<validWidth {
          let ox = tile.x * 2 + x
          let wx = Self.edgeWeight(
            x, length: validWidth, feather: feather,
            leading: tile.x > 0,
            trailing: tile.x + Self.tileSide < sourceWidth
          )
          let weight = wx * wy
          let pixel = oy * outputWidth + ox
          let sourceIndex = y * enhancedRow + x * 4
          let base = pixel * 3
          sums[base] += Float(enhancedBytes[sourceIndex]) * weight
          sums[base + 1] += Float(enhancedBytes[sourceIndex + 1]) * weight
          sums[base + 2] += Float(enhancedBytes[sourceIndex + 2]) * weight
          weights[pixel] += weight
        }
      }
      CVPixelBufferUnlockBaseAddress(enhanced, .readOnly)
      progress(tileIndex + 1, tiles.count)
    }
    let destination = try piperBuffer(width: outputWidth, height: outputHeight)
    CVPixelBufferLockBaseAddress(destination, [])
    defer { CVPixelBufferUnlockBaseAddress(destination, []) }
    guard let outputBase = CVPixelBufferGetBaseAddress(destination) else {
      throw PiperSRError.media("出力フレームにCPUアドレスがありません")
    }
    let output = outputBase.assumingMemoryBound(to: UInt8.self)
    let row = CVPixelBufferGetBytesPerRow(destination)
    for y in 0..<outputHeight {
      for x in 0..<outputWidth {
        let pixel = y * outputWidth + x
        let weight = max(weights[pixel], 1e-6)
        let sourceIndex = pixel * 3
        let destinationIndex = y * row + x * 4
        for channel in 0..<3 {
          output[destinationIndex + channel] = UInt8(
            min(255, max(0, (sums[sourceIndex + channel] / weight).rounded()))
          )
        }
        output[destinationIndex + 3] = 255
      }
    }
    return destination
  }

  private static func edgeWeight(
    _ position: Int, length: Int, feather: Int,
    leading: Bool, trailing: Bool
  ) -> Float {
    var weight: Float = 1
    if leading {
      weight *= min(1, Float(position) / Float(feather))
    }
    if trailing {
      weight *= min(1, Float(length - 1 - position) / Float(feather))
    }
    return max(weight, 1e-4)
  }
}

private final class PiperSRWriter {
  private let writer: AVAssetWriter
  private let receiver: AVAssetWriterInput.PixelBufferReceiver
  private let context = CIContext(options: [.cacheIntermediates: false])
  private let width: Int
  private let height: Int
  private let frameStep: CMTime
  private var lastTime: CMTime?
  private(set) var frameCount = 0

  init(url: URL, width: Int, height: Int, frameRate: Double) throws {
    self.width = width
    self.height = height
    frameStep = CMTime(seconds: 1 / frameRate, preferredTimescale: 120_000)
    writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let bitRate = Int(min(160_000_000, max(
      8_000_000, Double(width * height) * frameRate * 0.12
    )))
    let settings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoEncoderSpecificationKey: [
        kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder as String: true
      ],
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: bitRate,
        AVVideoExpectedSourceFrameRateKey: frameRate,
        AVVideoAllowFrameReorderingKey: false,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      ],
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.mediaTimeScale = 120_000
    var attributes = CVPixelBufferCreationAttributes(
      pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA),
      size: CVImageSize(width: width, height: height)
    )
    attributes.backing = .ioSurface
    receiver = writer.inputPixelBufferReceiver(
      for: input, pixelBufferAttributes: attributes
    )
    try writer.start()
    writer.startSession(atSourceTime: .zero)
  }

  func append(_ source: CVPixelBuffer, at requestedTime: CMTime) async throws {
    guard let pool = receiver.pixelBufferPool else {
      throw PiperSRError.writer("動画ライターのバッファプールがありません")
    }
    let destination = try pool.makeMutablePixelBuffer().withUnsafeBuffer { $0 }
    let sourceWidth = CVPixelBufferGetWidth(source)
    let sourceHeight = CVPixelBufferGetHeight(source)
    let image = CIImage(cvPixelBuffer: source)
    let scaled = image.transformed(
      by: CGAffineTransform(
        scaleX: CGFloat(width) / CGFloat(sourceWidth),
        y: CGFloat(height) / CGFloat(sourceHeight)
      )
    )
    context.render(
      scaled, to: destination,
      bounds: CGRect(x: 0, y: 0, width: width, height: height),
      colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
    )
    var time = requestedTime
    if let lastTime, time <= lastTime { time = lastTime + frameStep }
    do {
      try await receiver.append(
        CVReadOnlyPixelBuffer(unsafeBuffer: destination), with: time
      )
    } catch {
      throw PiperSRError.writer("フレームを書き出せません: \(error.localizedDescription)")
    }
    lastTime = time
    frameCount += 1
  }

  func finish() async throws {
    receiver.finish()
    await writer.finishWriting()
    guard writer.status == .completed else {
      throw PiperSRError.writer(
        writer.error?.localizedDescription ?? "書き出しを完了できません"
      )
    }
  }
}

@main
private struct PiperSRNativeVideoRunner {
  static func main() async {
    do {
      let args = try PiperSRArguments.parse()
      guard !FileManager.default.fileExists(atPath: args.output.path) else {
        throw PiperSRError.invalid("出力ファイルはすでに存在します")
      }
      print("STAGE 入力動画を確認中")
      let media = try await PiperSRMedia.load(args.input)
      guard args.outputWidth <= media.width * 2,
            args.outputHeight <= media.height * 2 else {
        throw PiperSRError.invalid("PiperSRの最大出力は入力の2倍です")
      }
      let processor = try PiperSRProcessor(
        models: args.models, width: media.width, height: media.height
      )
      let sharpener = args.sharpness > 0 ? try PiperSRSharpen(strength: args.sharpness) : nil
      print("PiperSR: \(media.width)x\(media.height), \(processor.modeName), \(processor.tileCount) tiles/frame")
      print(String(format: "PiperSR sharpness: %.2f", args.sharpness))
      print("STAGE Core ML / ANEでアップスケール中")
      fflush(stdout)
      let reader = try PiperSRReader(media: media)
      let writer = try PiperSRWriter(
        url: args.output, width: args.outputWidth,
        height: args.outputHeight, frameRate: media.frameRate
      )
      var firstTime: CMTime?
      var frame = 0
      if processor.isFullFrame {
        var slot = 0
        var pending: PiperSRPendingFrame?
        while let (decoded, time) = try await reader.next() {
          firstTime = firstTime ?? time
          try autoreleasepool {
            try processor.prepareFullFrame(decoded, transform: media.transform, slot: slot)
          }
          let current = try processor.predictFullFrame(
            slot: slot, time: time - firstTime!
          )
          if let pending {
            let result = try processor.finishFullFrame(pending)
            let adjusted = try sharpener?.apply(to: result) ?? result
            try await writer.append(adjusted, at: pending.time)
          }
          pending = current
          slot = 1 - slot
          frame += 1
          print("FRAME \(frame)")
          if media.duration > 0 {
            let elapsed = max(0, CMTimeGetSeconds(time - firstTime!))
            print(String(format: "PROGRESS %.2f", min(99, elapsed / media.duration * 100)))
          }
          fflush(stdout)
        }
        if let pending {
          let result = try processor.finishFullFrame(pending)
          let adjusted = try sharpener?.apply(to: result) ?? result
          try await writer.append(adjusted, at: pending.time)
        }
      } else {
        while let (decoded, time) = try await reader.next() {
          let result = try autoreleasepool { () throws -> CVPixelBuffer in
            let oriented = try processor.orient(decoded, transform: media.transform)
            return try processor.upscale(oriented) { index, count in
              print("TILE \(index)/\(count)")
            }
          }
          firstTime = firstTime ?? time
          frame += 1
          print("FRAME \(frame)")
          fflush(stdout)
          let adjusted = try sharpener?.apply(to: result) ?? result
          try await writer.append(adjusted, at: time - firstTime!)
          if media.duration > 0 {
            let elapsed = max(0, CMTimeGetSeconds(time - firstTime!))
            print(String(format: "PROGRESS %.2f", min(99, elapsed / media.duration * 100)))
            fflush(stdout)
          }
        }
      }
      try reader.validateCompleted()
      guard frame > 0 else { throw PiperSRError.media("入力動画が空です") }
      try await writer.finish()
      print("PROGRESS 100")
      print("STAGE 完了（\(writer.frameCount)フレーム）")
      fflush(stdout)
    } catch {
      FileHandle.standardError.write(
        Data("pipersr-coreml-video: \(error.localizedDescription)\n".utf8)
      )
      exit(EXIT_FAILURE)
    }
  }
}
