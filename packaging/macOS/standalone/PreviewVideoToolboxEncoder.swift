// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import Accelerate
import AVFoundation
import CoreVideo
import Darwin
import Foundation
import VideoToolbox

enum EncoderError: LocalizedError {
  case invalidArguments
  case invalidValue(String)
  case mappingFailed(String)
  case unexpectedEndOfInput
  case pixelBuffer(String)
  case writer(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return """
        usage: mioh-preview-videotoolbox-encoder \
        <shared-file> <output-dir> <width> <height> \
        <fps-numerator> <fps-denominator> <generation> <segment-seconds>
        """
    case .invalidValue(let value):
      return "invalid preview encoder value: \(value)"
    case .mappingFailed(let path):
      return "unable to map preview frame file: \(path)"
    case .unexpectedEndOfInput:
      return "preview encoder command stream ended unexpectedly"
    case .pixelBuffer(let message):
      return "preview pixel buffer failure: \(message)"
    case .writer(let message):
      return "preview VideoToolbox writer failure: \(message)"
    }
  }
}

struct SegmentEvent: Codable {
  let sequence: Int
  let startNs: Int64
  let endNs: Int64
  let path: String
  let codec: String

  enum CodingKeys: String, CodingKey {
    case sequence
    case startNs = "start_ns"
    case endNs = "end_ns"
    case path
    case codec
  }
}

struct CommandResponse: Codable {
  let status: String
  let activePath: String?
  let segment: SegmentEvent?

  enum CodingKeys: String, CodingKey {
    case status
    case activePath = "active_path"
    case segment
  }
}

final class SegmentWriter {
  private let outputDirectory: URL
  private let width: Int
  private let height: Int
  private let fpsNumerator: Int
  private let fpsDenominator: Int
  private let generation: Int
  private let segmentNanoseconds: Int64
  private let frameDurationNanoseconds: Int64
  private let codec: AVVideoCodecType
  private let requestedAverageBitRate: Int?
  private let realTime: Bool
  private let filePrefix: String

  private var sequence = 0
  private var segmentStartNanoseconds: Int64?
  private var lastPTS: Int64?
  private var framesInSegment = 0
  private var writer: AVAssetWriter?
  private var writerInput: AVAssetWriterInput?
  private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
  private var workingURL: URL?
  private var finalURL: URL?

  init(
    outputDirectory: URL,
    width: Int,
    height: Int,
    fpsNumerator: Int,
    fpsDenominator: Int,
    generation: Int,
    segmentSeconds: Double,
    codec: AVVideoCodecType = .h264,
    averageBitRate: Int? = nil,
    realTime: Bool = true,
    filePrefix: String = "preview"
  ) throws {
    guard width > 0, height > 0 else {
      throw EncoderError.invalidValue("frame dimensions")
    }
    guard fpsNumerator > 0, fpsDenominator > 0, fpsNumerator <= Int(Int32.max) else {
      throw EncoderError.invalidValue("frame rate")
    }
    guard segmentSeconds > 0 else {
      throw EncoderError.invalidValue("segment duration")
    }
    self.outputDirectory = outputDirectory
    self.width = width
    self.height = height
    self.fpsNumerator = fpsNumerator
    self.fpsDenominator = fpsDenominator
    self.generation = generation
    self.codec = codec
    requestedAverageBitRate = averageBitRate
    self.realTime = realTime
    self.filePrefix = filePrefix
    segmentNanoseconds = Int64(segmentSeconds * 1_000_000_000)
    frameDurationNanoseconds = Int64(
      Double(1_000_000_000 * fpsDenominator) / Double(fpsNumerator)
    )
    try FileManager.default.createDirectory(
      at: outputDirectory,
      withIntermediateDirectories: true
    )
  }

  var activePath: String? {
    workingURL?.path
  }

  private func paths(for sequence: Int) -> (working: URL, final: URL) {
    let name = String(
      format: "\(filePrefix)-g%d-%06d.mp4",
      generation,
      sequence
    )
    let final = outputDirectory.appendingPathComponent(name)
    return (
      outputDirectory.appendingPathComponent("\(name).part"),
      final
    )
  }

  private func openSegment(startNanoseconds: Int64) throws {
    let paths = paths(for: sequence)
    try? FileManager.default.removeItem(at: paths.working)
    try? FileManager.default.removeItem(at: paths.final)

    let writer = try AVAssetWriter(outputURL: paths.working, fileType: .mp4)
    let frameRate = Double(fpsNumerator) / Double(fpsDenominator)
    let pixelRate = Double(width * height) * frameRate
    let averageBitRate = requestedAverageBitRate ?? Int(
      min(35_000_000, max(4_000_000, pixelRate * 0.10))
    )
    let profile: String = codec == .hevc
      ? (kVTProfileLevel_HEVC_Main_AutoLevel as String)
      : AVVideoProfileLevelH264HighAutoLevel
    let settings: [String: Any] = [
      AVVideoCodecKey: codec,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      // Prefer VideoToolbox hardware, but do not make preview availability
      // depend on a free hardware encoder session. macOS may temporarily
      // exhaust those sessions while another mioh/export process is active.
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
        AVVideoAverageBitRateKey: averageBitRate,
        AVVideoExpectedSourceFrameRateKey: frameRate,
        AVVideoAllowFrameReorderingKey: false,
        AVVideoMaxKeyFrameIntervalKey: max(1, Int(frameRate * 2)),
        AVVideoProfileLevelKey: profile,
      ],
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.expectsMediaDataInRealTime = realTime
    let attributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferMetalCompatibilityKey as String: true,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: attributes
    )
    guard writer.canAdd(input) else {
      throw EncoderError.writer("cannot add \(codec.rawValue) input")
    }
    writer.add(input)
    guard writer.startWriting() else {
      throw EncoderError.writer(
        writer.error?.localizedDescription ?? "startWriting failed"
      )
    }
    writer.startSession(atSourceTime: .zero)

    self.writer = writer
    writerInput = input
    self.adaptor = adaptor
    workingURL = paths.working
    finalURL = paths.final
    segmentStartNanoseconds = startNanoseconds
    framesInSegment = 0
  }

  private func makePixelBuffer(fromBGR source: UnsafeMutableRawPointer) throws
    -> CVPixelBuffer
  {
    guard let pool = adaptor?.pixelBufferPool else {
      throw EncoderError.pixelBuffer("AVAssetWriter did not create a pool")
    }
    var optionalBuffer: CVPixelBuffer?
    let result = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pool,
      &optionalBuffer
    )
    guard result == kCVReturnSuccess, let pixelBuffer = optionalBuffer else {
      throw EncoderError.pixelBuffer("pool allocation returned \(result)")
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    guard let destination = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw EncoderError.pixelBuffer("base address unavailable")
    }

    var sourceBuffer = vImage_Buffer(
      data: source,
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: width * 3
    )
    var destinationBuffer = vImage_Buffer(
      data: destination,
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer)
    )
    // The three source bytes are B,G,R. The RGB->RGBA routine preserves
    // channel order and appends alpha, producing the BGRA layout requested by
    // the pixel-buffer pool without an intermediate ndarray or VideoFrame.
    let conversion = vImageConvert_RGB888toRGBA8888(
      &sourceBuffer,
      nil,
      255,
      &destinationBuffer,
      false,
      vImage_Flags(kvImageNoFlags)
    )
    guard conversion == kvImageNoError else {
      throw EncoderError.pixelBuffer("vImage conversion returned \(conversion)")
    }
    CVBufferSetAttachment(
      pixelBuffer,
      kCVImageBufferColorPrimariesKey,
      kCVImageBufferColorPrimaries_ITU_R_709_2,
      .shouldPropagate
    )
    CVBufferSetAttachment(
      pixelBuffer,
      kCVImageBufferTransferFunctionKey,
      kCVImageBufferTransferFunction_ITU_R_709_2,
      .shouldPropagate
    )
    CVBufferSetAttachment(
      pixelBuffer,
      kCVImageBufferYCbCrMatrixKey,
      kCVImageBufferYCbCrMatrix_ITU_R_709_2,
      .shouldPropagate
    )
    return pixelBuffer
  }

  func append(source: UnsafeMutableRawPointer, ptsNanoseconds: Int64) async throws
    -> SegmentEvent?
  {
    let pixelBuffer = try makePixelBuffer(fromBGR: source)
    return try await append(
      pixelBuffer: pixelBuffer,
      ptsNanoseconds: ptsNanoseconds
    )
  }

  /// Append an existing CVPixelBuffer without staging it through Python or a
  /// BGR mmap. The native preview pipeline uses this entry point so decoded
  /// IOSurfaces remain inside Swift from AVAssetReader through VideoToolbox.
  func append(pixelBuffer: CVPixelBuffer, ptsNanoseconds: Int64) async throws
    -> SegmentEvent?
  {
    var completed: SegmentEvent?
    if let start = segmentStartNanoseconds,
      ptsNanoseconds >= start + segmentNanoseconds
    {
      completed = try await closeSegment(endNanoseconds: start + segmentNanoseconds)
    }
    if writer == nil {
      try openSegment(startNanoseconds: ptsNanoseconds)
    }
    guard let writer, let input = writerInput, let adaptor else {
      throw EncoderError.writer("segment is not open")
    }
    while !input.isReadyForMoreMediaData {
      if writer.status == .failed || writer.status == .cancelled {
        throw EncoderError.writer(
          writer.error?.localizedDescription ?? "writer stopped accepting frames"
        )
      }
      try await Task.sleep(nanoseconds: 250_000)
    }
    let presentationTime = CMTime(
      value: Int64(framesInSegment * fpsDenominator),
      timescale: Int32(fpsNumerator)
    )
    guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
      throw EncoderError.writer(
        writer.error?.localizedDescription ?? "pixel-buffer append failed"
      )
    }
    framesInSegment += 1
    lastPTS = ptsNanoseconds
    return completed
  }

  func finish() async throws -> SegmentEvent? {
    guard let lastPTS else { return nil }
    return try await closeSegment(
      endNanoseconds: lastPTS + frameDurationNanoseconds
    )
  }

  private func closeSegment(endNanoseconds: Int64) async throws -> SegmentEvent {
    guard let writer, let input = writerInput, let workingURL, let finalURL,
      let start = segmentStartNanoseconds
    else {
      throw EncoderError.writer("cannot close a segment that is not open")
    }
    input.markAsFinished()
    await withCheckedContinuation { continuation in
      writer.finishWriting {
        continuation.resume()
      }
    }
    guard writer.status == .completed else {
      throw EncoderError.writer(
        writer.error?.localizedDescription ?? "finishWriting failed"
      )
    }
    try FileManager.default.moveItem(at: workingURL, to: finalURL)
    let event = SegmentEvent(
      sequence: sequence,
      startNs: start,
      endNs: endNanoseconds,
      path: finalURL.path,
      codec: codec == .hevc ? "hevc_videotoolbox" : "h264_videotoolbox"
    )
    sequence += 1
    self.writer = nil
    writerInput = nil
    adaptor = nil
    self.workingURL = nil
    self.finalURL = nil
    segmentStartNanoseconds = nil
    framesInSegment = 0
    return event
  }

  func discard() {
    writer?.cancelWriting()
    if let workingURL {
      try? FileManager.default.removeItem(at: workingURL)
    }
    writer = nil
    writerInput = nil
    adaptor = nil
    workingURL = nil
    finalURL = nil
    segmentStartNanoseconds = nil
  }
}

#if !MIOH_NATIVE_PREVIEW_PIPELINE
@main
private struct PreviewVideoToolboxEncoder {
  static func main() async {
    do {
      try await run()
    } catch {
      let message = "mioh-preview-videotoolbox-encoder: \(error.localizedDescription)\n"
      FileHandle.standardError.write(Data(message.utf8))
      exit(EXIT_FAILURE)
    }
  }

  private static func run() async throws {
    guard CommandLine.arguments.count == 9 else {
      throw EncoderError.invalidArguments
    }
    let sharedPath = CommandLine.arguments[1]
    let outputDirectory = URL(
      fileURLWithPath: CommandLine.arguments[2],
      isDirectory: true
    )
    guard let width = Int(CommandLine.arguments[3]),
      let height = Int(CommandLine.arguments[4]),
      let fpsNumerator = Int(CommandLine.arguments[5]),
      let fpsDenominator = Int(CommandLine.arguments[6]),
      let generation = Int(CommandLine.arguments[7]),
      let segmentSeconds = Double(CommandLine.arguments[8])
    else {
      throw EncoderError.invalidArguments
    }
    let frameBytes = width * height * 3
    let mappingBytes = frameBytes + MemoryLayout<Int64>.stride
    let descriptor = try FileManager.default.attributesOfItem(atPath: sharedPath)
    guard let fileSize = descriptor[.size] as? NSNumber,
      fileSize.intValue >= mappingBytes
    else {
      throw EncoderError.invalidValue("shared frame file size")
    }
    let fileDescriptor = open(sharedPath, O_RDWR)
    guard fileDescriptor >= 0 else {
      throw EncoderError.mappingFailed(sharedPath)
    }
    defer { close(fileDescriptor) }
    let mapping = mmap(
      nil,
      mappingBytes,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      fileDescriptor,
      0
    )
    guard mapping != MAP_FAILED, let mapping else {
      throw EncoderError.mappingFailed(sharedPath)
    }
    defer { munmap(mapping, mappingBytes) }

    let encoder = try SegmentWriter(
      outputDirectory: outputDirectory,
      width: width,
      height: height,
      fpsNumerator: fpsNumerator,
      fpsDenominator: fpsDenominator,
      generation: generation,
      segmentSeconds: segmentSeconds
    )
    let input = FileHandle.standardInput
    let output = FileHandle.standardOutput
    while true {
      guard let command = try input.read(upToCount: 1), !command.isEmpty else {
        encoder.discard()
        break
      }
      let response: CommandResponse
      switch command[0] {
      case 0:
        let pts = mapping.advanced(by: frameBytes).loadUnaligned(as: Int64.self)
        let segment = try await encoder.append(
          source: mapping,
          ptsNanoseconds: Int64(littleEndian: pts)
        )
        response = CommandResponse(
          status: "ok",
          activePath: encoder.activePath,
          segment: segment
        )
      case 1:
        let segment = try await encoder.finish()
        response = CommandResponse(status: "ok", activePath: nil, segment: segment)
      case 2:
        encoder.discard()
        response = CommandResponse(status: "ok", activePath: nil, segment: nil)
      default:
        throw EncoderError.invalidValue("command \(command[0])")
      }
      var payload = try JSONEncoder().encode(response)
      payload.append(0x0A)
      try output.write(contentsOf: payload)
      if command[0] == 1 || command[0] == 2 {
        break
      }
    }
  }
}
#endif
