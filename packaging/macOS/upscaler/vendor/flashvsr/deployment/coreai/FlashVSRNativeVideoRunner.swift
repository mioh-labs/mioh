import AVFoundation
import CoreAI
import CoreImage
import CoreMedia
import CoreVideo
import Darwin
import Foundation
import Metal
import VideoToolbox

// Process a bounded temporal window across every spatial tile before moving
// forward.  The previous tile-major implementation rendered the entire video
// once per tile, so a 720p 2x job produced 84 full-length temporary movies and
// could not expose a usable output until all of them had completed.
private let nativeTemporalSegmentFrames = 85
// Replaying 21 frames is intentional: retaining every tile's 30 DiT KV
// caches across segment boundaries would require hundreds of MB per tile.
// The replay keeps memory bounded while progress still advances per segment.
private let nativeTemporalWarmupFrames = 21
private let nativeTemporalLookaheadFrames = 4

private func nativeRationalFrameRate(_ fps: Double) -> (numerator: Int, denominator: Int) {
    for candidate in [(24_000, 1001), (30_000, 1001), (60_000, 1001)] {
        if abs(fps - Double(candidate.0) / Double(candidate.1)) < 0.02 {
            return candidate
        }
    }
    return (max(1, Int(fps.rounded())), 1)
}

private func nativeFrameTime(_ frame: Int, frameRate: Double) -> CMTime {
    let rate = nativeRationalFrameRate(frameRate)
    return CMTime(
        value: Int64(frame * rate.denominator),
        timescale: CMTimeScale(rate.numerator)
    )
}

@available(macOS 27.0, *)
private enum NativeVideoError: LocalizedError {
    case argument(String)
    case media(String)
    case tensor(String)
    case writer(String)

    var errorDescription: String? {
        switch self {
        case .argument(let value): return value
        case .media(let value): return value
        case .tensor(let value): return value
        case .writer(let value): return value
        }
    }
}

@available(macOS 27.0, *)
private struct NativeVideoArguments {
    let input: URL
    let output: URL
    let models: URL
    let scale: Int
    let outputWidth: Int?
    let outputHeight: Int?
    let seed: UInt64
    let computePolicy: FlashVSRComputePolicy

    static func parse() throws -> NativeVideoArguments {
        var values: [String: String] = [:]
        var index = 1
        while index < CommandLine.arguments.count {
            let key = CommandLine.arguments[index]
            guard key.hasPrefix("--"), index + 1 < CommandLine.arguments.count else {
                throw NativeVideoError.argument(
                    "Usage: --input VIDEO --output MP4 --models DIR --scale 2|4 "
                        + "[--output-width W --output-height H] [--seed N]"
                )
            }
            values[key] = CommandLine.arguments[index + 1]
            index += 2
        }
        guard let input = values["--input"], let output = values["--output"],
              let models = values["--models"], let scale = Int(values["--scale"] ?? ""),
              scale == 2 || scale == 4 else {
            throw NativeVideoError.argument("--input, --output, --models and --scale 2|4 are required")
        }
        let outputWidth = values["--output-width"].flatMap(Int.init)
        let outputHeight = values["--output-height"].flatMap(Int.init)
        guard (outputWidth == nil) == (outputHeight == nil),
              outputWidth.map({ $0 > 0 && $0.isMultiple(of: 2) }) ?? true,
              outputHeight.map({ $0 > 0 && $0.isMultiple(of: 2) }) ?? true else {
            throw NativeVideoError.argument(
                "--output-width and --output-height must be supplied together as positive even values"
            )
        }
        return NativeVideoArguments(
            input: URL(fileURLWithPath: input),
            output: URL(fileURLWithPath: output),
            models: URL(fileURLWithPath: models, isDirectory: true),
            scale: scale,
            outputWidth: outputWidth,
            outputHeight: outputHeight,
            seed: UInt64(values["--seed"] ?? "0") ?? 0,
            computePolicy: FlashVSRComputePolicy(
                rawValue: values["--compute"] ?? "hybrid"
            ) ?? .hybrid
        )
    }
}

@available(macOS 27.0, *)
private struct NativeVideoMetadata {
    let asset: AVURLAsset
    let track: AVAssetTrack
    let transform: CGAffineTransform
    let width: Int
    let height: Int
    let frameRate: Double
    let frameCount: Int
    let duration: CMTime
    let frameTimes: [CMTime]

    static func load(url: URL) async throws -> NativeVideoMetadata {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NativeVideoError.media("The input has no video track")
        }
        let natural = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let oriented = natural.applying(transform)
        let nominal = max(1, Double(try await track.load(.nominalFrameRate)))
        let duration = try await asset.load(.duration)
        let frameTimes = try readFrameTimes(asset: asset, track: track)
        guard !frameTimes.isEmpty else {
            throw NativeVideoError.media("The input has no decodable frames")
        }
        return NativeVideoMetadata(
            asset: asset,
            track: track,
            transform: transform,
            width: max(1, Int(abs(oriented.width).rounded())),
            height: max(1, Int(abs(oriented.height).rounded())),
            frameRate: nominal,
            frameCount: frameTimes.count,
            duration: duration,
            frameTimes: frameTimes
        )
    }

    func presentationTime(for frame: Int) -> CMTime {
        guard frameTimes.indices.contains(frame) else {
            return nativeFrameTime(frame, frameRate: frameRate)
        }
        return frameTimes[frame] - frameTimes[0]
    }

    private static func readFrameTimes(
        asset: AVAsset, track: AVAssetTrack
    ) throws -> [CMTime] {
        let reader = try AVAssetReader(asset: asset)
        // Count decoded images rather than compressed samples. On macOS 27 an
        // H.264 access unit may be vended as several compressed sample buffers,
        // while the BGRA output is exactly one buffer per display frame.
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
            ]
        )
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw NativeVideoError.media("Cannot count input frames") }
        reader.add(output)
        guard reader.startReading() else {
            throw NativeVideoError.media(reader.error?.localizedDescription ?? "Cannot read input")
        }
        var times: [CMTime] = []
        while let sample = output.copyNextSampleBuffer() {
            times.append(CMSampleBufferGetPresentationTimeStamp(sample))
        }
        guard reader.status == .completed else {
            throw NativeVideoError.media(reader.error?.localizedDescription ?? "Frame count failed")
        }
        return times
    }
}

@available(macOS 27.0, *)
private struct NativeTile: Hashable {
    let x: Int
    let y: Int
    let validWidth: Int
    let validHeight: Int
}

@available(macOS 27.0, *)
private struct NativeTileRow {
    let y: Int
    let tiles: [NativeTile]
    let validHeight: Int
}

@available(macOS 27.0, *)
private func nativeTiles(width: Int, height: Int, scale: Int) -> [NativeTile] {
    let side = 256 / scale
    let overlap = max(4, side / 8)
    func positions(_ length: Int) -> [Int] {
        guard length > side else { return [0] }
        let step = side - overlap
        let span = length - side
        let intervalCount = max(1, Int(ceil(Double(span) / Double(step))))
        return (0...intervalCount).map { index in
            Int((Double(index) * Double(span) / Double(intervalCount)).rounded())
        }
    }
    return positions(height).flatMap { y in
        positions(width).map { x in
            NativeTile(
                x: x, y: y,
                validWidth: min(side, width - x),
                validHeight: min(side, height - y)
            )
        }
    }
}

@available(macOS 27.0, *)
private func nativeTileRows(width: Int, height: Int, scale: Int) -> [NativeTileRow] {
    let tiles = nativeTiles(width: width, height: height, scale: scale)
    return Dictionary(grouping: tiles, by: \.y).keys.sorted().map { y in
        let rowTiles = tiles.filter { $0.y == y }.sorted { $0.x < $1.x }
        return NativeTileRow(
            y: y,
            tiles: rowTiles,
            validHeight: rowTiles.map(\.validHeight).max() ?? 0
        )
    }
}

@available(macOS 27.0, *)
private struct NativePreparedFrame {
    // Channel-first RGB in the FlashVSR [-1, 1] input range.
    let values: [Float16]
}

@available(macOS 27.0, *)
private struct NativePreparedGroup {
    let frames: [NativePreparedFrame]
    let tensor: NDArray
}

@available(macOS 27.0, *)
private final class NativeDecodedSegment {
    let frames: [CVPixelBuffer]

    init(
        metadata: NativeVideoMetadata,
        startFrame: Int,
        processingFrameCount: Int,
        reportFrames: (Int, Int) -> Void
    ) throws {
        let requested = min(
            max(0, metadata.frameCount - startFrame),
            processingFrameCount + nativeTemporalLookaheadFrames
        )
        guard requested > 0 else {
            throw NativeVideoError.media("The temporal segment has no source frames")
        }
        let reader = try AVAssetReader(asset: metadata.asset)
        let startTime = metadata.frameTimes[startFrame]
        let endFrame = startFrame + requested
        let endTime = endFrame < metadata.frameCount
            ? metadata.frameTimes[endFrame] : metadata.duration
        let rangeDuration = endTime > startTime
            ? endTime - startTime
            : nativeFrameTime(requested, frameRate: metadata.frameRate)
        reader.timeRange = CMTimeRange(
            start: startTime,
            duration: rangeDuration
        )
        let output = AVAssetReaderTrackOutput(
            track: metadata.track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String:
                    Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
        )
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw NativeVideoError.media("Cannot create the shared segment decoder")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw NativeVideoError.media(
                reader.error?.localizedDescription ?? "Cannot start the shared segment decoder"
            )
        }
        var decoded: [CVPixelBuffer] = []
        decoded.reserveCapacity(requested)
        while decoded.count < requested,
              let sample = output.copyNextSampleBuffer(),
              let buffer = CMSampleBufferGetImageBuffer(sample) {
            decoded.append(buffer)
            reportFrames(decoded.count, requested)
        }
        guard !decoded.isEmpty else {
            throw NativeVideoError.media(
                reader.error?.localizedDescription ?? "The shared segment decoder returned no frames"
            )
        }
        frames = decoded
    }

    func frame(at index: Int) -> CVPixelBuffer {
        frames[min(max(0, index), frames.count - 1)]
    }
}

@available(macOS 27.0, *)
private final class NativeFramePreparer {
    private let context = CIContext(options: [
        .cacheIntermediates: false,
        .highQualityDownsample: true,
    ])

    func render(
        _ buffer: CVPixelBuffer,
        metadata: NativeVideoMetadata,
        tile: NativeTile,
        tileSide: Int
    ) throws -> NativePreparedFrame {
        var image = CIImage(cvPixelBuffer: buffer).transformed(by: metadata.transform)
        let transformedExtent = image.extent
        image = image.transformed(
            by: CGAffineTransform(
                translationX: -transformedExtent.minX,
                y: -transformedExtent.minY
            )
        )
        let sourceY = metadata.height - tile.y - tileSide
        image = image.clampedToExtent().cropped(
            to: CGRect(x: tile.x, y: sourceY, width: tileSide, height: tileSide)
        )
        let factor = 256.0 / Double(tileSide)
        image = image.transformed(by: CGAffineTransform(scaleX: factor, y: factor))
        image = image.transformed(
            by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY)
        ).cropped(to: CGRect(x: 0, y: 0, width: 256, height: 256))

        var rgba = [UInt8](repeating: 0, count: 256 * 256 * 4)
        context.render(
            image,
            toBitmap: &rgba,
            rowBytes: 256 * 4,
            bounds: CGRect(x: 0, y: 0, width: 256, height: 256),
            format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )
        let pixels = 256 * 256
        var chw = [Float16](repeating: 0, count: 3 * pixels)
        for pixel in 0..<pixels {
            let source = pixel * 4
            chw[pixel] = Float16(Float(rgba[source]) / 127.5 - 1.0)
            chw[pixels + pixel] = Float16(Float(rgba[source + 1]) / 127.5 - 1.0)
            chw[2 * pixels + pixel] = Float16(Float(rgba[source + 2]) / 127.5 - 1.0)
        }
        return NativePreparedFrame(values: chw)
    }
}

@available(macOS 27.0, *)
private final class NativeTileFrameReader {
    private let metadata: NativeVideoMetadata
    private let tile: NativeTile
    private let tileSide: Int
    private let segment: NativeDecodedSegment
    private let preparer: NativeFramePreparer
    private var readCount = 0

    init(
        metadata: NativeVideoMetadata,
        tile: NativeTile,
        scale: Int,
        segment: NativeDecodedSegment,
        preparer: NativeFramePreparer
    ) {
        self.metadata = metadata
        self.tile = tile
        tileSide = 256 / scale
        self.segment = segment
        self.preparer = preparer
    }

    func next() throws -> NativePreparedFrame {
        let buffer = segment.frame(at: readCount)
        readCount += 1
        return try preparer.render(
            buffer, metadata: metadata, tile: tile, tileSide: tileSide
        )
    }
}

@available(macOS 27.0, *)
private func makeGroup(_ frames: [NativePreparedFrame]) throws -> NativePreparedGroup {
    guard frames.count == 4, frames.allSatisfy({ $0.values.count == 3 * 256 * 256 }) else {
        throw NativeVideoError.tensor("A decoder/LQ group must contain four 256x256 RGB frames")
    }
    let pixels = 256 * 256
    var tensor = NDArray(shape: [1, 3, 4, 256, 256], scalarType: .float16)
    let view = tensor.mutableView(as: Float16.self)
    view.withUnsafeMutablePointer { target, _, _ in
        for channel in 0..<3 {
            for frame in 0..<4 {
                frames[frame].values.withUnsafeBufferPointer { source in
                    target.advanced(by: (channel * 4 + frame) * pixels).update(
                        from: source.baseAddress!.advanced(by: channel * pixels),
                        count: pixels
                    )
                }
            }
        }
    }
    return NativePreparedGroup(frames: frames, tensor: tensor)
}

@available(macOS 27.0, *)
private func makeWarmup(_ frame: NativePreparedFrame) throws -> NDArray {
    guard frame.values.count == 3 * 256 * 256 else {
        throw NativeVideoError.tensor("Invalid first LQ frame")
    }
    var tensor = NDArray(shape: [1, 3, 1, 256, 256], scalarType: .float16)
    let view = tensor.mutableView(as: Float16.self)
    view.withUnsafeMutablePointer { target, _, _ in
        frame.values.withUnsafeBufferPointer { source in
            target.update(from: source.baseAddress!, count: source.count)
        }
    }
    return tensor
}

@available(macOS 27.0, *)
private struct NativeCoordinateNormalGenerator {
    let seed: UInt64
    let outputOriginX: Int
    let outputOriginY: Int

    private func mixed(_ input: UInt64) -> UInt64 {
        var value = input &+ 0x9E3779B97F4A7C15
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        value ^= value >> 31
        return value
    }

    private func uniform(_ input: UInt64) -> Double {
        (Double(mixed(input) >> 11) + 0.5) / Double(1 << 53)
    }

    func latent(frames: Int, temporalFrameStart: Int) -> NDArray {
        var tensor = NDArray(shape: [1, 16, frames, 32, 32], scalarType: .float16)
        let view = tensor.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { target, _, _ in
            for channel in 0..<16 {
                for frame in 0..<frames {
                    let globalTime = temporalFrameStart + frame * 4
                    for y in 0..<32 {
                        let globalY = outputOriginY + y * 8
                        for x in 0..<32 {
                            let globalX = outputOriginX + x * 8
                            var coordinate = seed
                            coordinate ^= UInt64(bitPattern: Int64(globalTime))
                                &* 0xD6E8FEB86659FD93
                            coordinate ^= UInt64(bitPattern: Int64(globalY))
                                &* 0xA5A3564E27F8862F
                            coordinate ^= UInt64(bitPattern: Int64(globalX))
                                &* 0x9E3779B185EBCA87
                            coordinate ^= UInt64(channel) &* 0xC2B2AE3D27D4EB4F
                            let u1 = max(
                                uniform(coordinate), Double.leastNonzeroMagnitude
                            )
                            let u2 = uniform(coordinate ^ 0x94D049BB133111EB)
                            let normal = sqrt(-2.0 * log(u1))
                                * cos(2.0 * Double.pi * u2)
                            let index = ((channel * frames + frame) * 32 + y) * 32 + x
                            target[index] = Float16(normal)
                        }
                    }
                }
            }
        }
        return tensor
    }
}

@available(macOS 27.0, *)
private final class NativeVideoWriter {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let fpsNumerator: Int
    private let fpsDenominator: Int
    private(set) var frameCount = 0
    private var lastPresentationTime: CMTime?

    init(
        url: URL,
        width: Int,
        height: Int,
        frameRate: Double,
        bitRate: Int,
        fragmentIntervalFrames: Int? = nil,
        expectedFrameCount: Int? = nil
    ) throws {
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let rate = nativeRationalFrameRate(frameRate)
        fpsNumerator = rate.numerator
        fpsDenominator = rate.denominator
        if let fragmentIntervalFrames, fragmentIntervalFrames > 0 {
            let interval = nativeFrameTime(fragmentIntervalFrames, frameRate: frameRate)
            writer.movieFragmentInterval = interval
            writer.initialMovieFragmentInterval = interval
            writer.shouldOptimizeForNetworkUse = true
            if let expectedFrameCount, expectedFrameCount > 0 {
                writer.overallDurationHint = nativeFrameTime(
                    expectedFrameCount,
                    frameRate: frameRate
                )
            }
        }
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
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        // Keep enough precision for CFR and VFR sources. 120000 is divisible
        // by the common 24000/30000/60000 video time scales, unlike a nominal
        // integer frame rate such as 23 or 29.
        writer.movieTimeScale = 120_000
        input.mediaTimeScale = 120_000
        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
        )
        guard writer.canAdd(input) else { throw NativeVideoError.writer("Cannot add H.264 writer input") }
        writer.add(input)
        guard writer.startWriting() else {
            throw NativeVideoError.writer(writer.error?.localizedDescription ?? "Cannot start video writer")
        }
        writer.startSession(atSourceTime: .zero)
    }

    func makePixelBuffer() throws -> CVPixelBuffer {
        guard let pool = adaptor.pixelBufferPool else {
            throw NativeVideoError.writer("The video writer has no pixel-buffer pool")
        }
        var value: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &value)
        guard status == kCVReturnSuccess, let value else {
            throw NativeVideoError.writer("Pixel-buffer allocation failed: \(status)")
        }
        return value
    }

    func append(
        _ pixelBuffer: CVPixelBuffer,
        presentationTime requestedTime: CMTime? = nil
    ) async throws {
        while !input.isReadyForMoreMediaData {
            if writer.status == .failed || writer.status == .cancelled {
                throw NativeVideoError.writer(writer.error?.localizedDescription ?? "Video writer stopped")
            }
            try await Task.sleep(nanoseconds: 250_000)
        }
        var time = requestedTime ?? CMTime(
            value: Int64(frameCount * fpsDenominator), timescale: Int32(fpsNumerator)
        )
        if let lastPresentationTime, time <= lastPresentationTime {
            time = lastPresentationTime + CMTime(
                value: Int64(fpsDenominator), timescale: Int32(fpsNumerator)
            )
        }
        guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
            throw NativeVideoError.writer(writer.error?.localizedDescription ?? "Cannot append video frame")
        }
        lastPresentationTime = time
        frameCount += 1
    }

    func finish() async throws {
        input.markAsFinished()
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw NativeVideoError.writer(writer.error?.localizedDescription ?? "Cannot finish video writer")
        }
    }
}

@available(macOS 27.0, *)
private final class NativeMappedFrames {
    let width: Int
    let height: Int
    let frameCount: Int
    let rowBytes: Int
    let frameBytes: Int
    let frameStride: Int

    private let url: URL
    private let descriptor: Int32
    private let mapping: UnsafeMutableRawPointer
    private let mappedBytes: Int
    private let device: MTLDevice

    init(
        width: Int,
        height: Int,
        frameCount: Int,
        directory: URL,
        stem: String,
        device: MTLDevice
    ) throws {
        let pixelResult = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, frameCount > 0,
              !pixelResult.overflow,
              !pixelResult.partialValue.multipliedReportingOverflow(by: 4).overflow else {
            throw NativeVideoError.media("Invalid mapped-frame dimensions")
        }
        let pixels = pixelResult.partialValue
        self.width = width
        self.height = height
        self.frameCount = frameCount
        rowBytes = width * 4
        frameBytes = pixels * 4
        let page = Int(getpagesize())
        frameStride = ((frameBytes + page - 1) / page) * page
        let totalResult = frameStride.multipliedReportingOverflow(by: frameCount)
        guard !totalResult.overflow else {
            throw NativeVideoError.media("Mapped-frame store is too large")
        }
        mappedBytes = totalResult.partialValue
        self.device = device
        url = directory.appendingPathComponent("\(stem)-\(UUID().uuidString).bgra")
        descriptor = Darwin.open(url.path, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw NativeVideoError.media("Cannot create mapped-frame store: \(String(cString: strerror(errno)))")
        }
        guard ftruncate(descriptor, off_t(mappedBytes)) == 0 else {
            Darwin.close(descriptor)
            throw NativeVideoError.media("Cannot size mapped-frame store: \(String(cString: strerror(errno)))")
        }
        let value = mmap(nil, mappedBytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0)
        guard value != MAP_FAILED, let value else {
            Darwin.close(descriptor)
            throw NativeVideoError.media("Cannot map frame store: \(String(cString: strerror(errno)))")
        }
        mapping = value
    }

    func buffer(for frame: Int) throws -> MTLBuffer {
        guard frame >= 0, frame < frameCount else {
            throw NativeVideoError.tensor("Mapped frame index is out of range")
        }
        let pointer = mapping.advanced(by: frame * frameStride)
        guard let buffer = device.makeBuffer(
            bytesNoCopy: pointer,
            length: frameStride,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            throw NativeVideoError.media("Metal cannot access a mapped frame")
        }
        return buffer
    }

    func pointer(for frame: Int) throws -> UnsafeMutableRawPointer {
        guard frame >= 0, frame < frameCount else {
            throw NativeVideoError.tensor("Mapped frame index is out of range")
        }
        return mapping.advanced(by: frame * frameStride)
    }

    func discard(frame: Int) {
        guard frame >= 0, frame < frameCount else { return }
        _ = madvise(mapping.advanced(by: frame * frameStride), frameStride, MADV_DONTNEED)
    }

    func discardAll() {
        _ = madvise(mapping, mappedBytes, MADV_DONTNEED)
    }

    deinit {
        munmap(mapping, mappedBytes)
        Darwin.close(descriptor)
        try? FileManager.default.removeItem(at: url)
    }
}

@available(macOS 27.0, *)
private final class NativeMetalCompositor {
    private static let source = #"""
    #include <metal_stdlib>
    using namespace metal;

    inline float raised_cosine(float position, uint extent) {
      if (extent == 0) return 1.0f;
      const float phase = clamp(position / float(extent), 0.0f, 1.0f);
      return 0.5f - 0.5f * cos(3.14159265358979323846f * phase);
    }

    kernel void correct_horizontal(
        device const half *content [[buffer(0)]],
        device const half *style [[buffer(1)]],
        device uchar4 *row [[buffer(2)]],
        constant float4 &contentMean [[buffer(3)]],
        constant float4 &contentStd [[buffer(4)]],
        constant float4 &styleMean [[buffer(5)]],
        constant float4 &styleStd [[buffer(6)]],
        constant uint4 &layout [[buffer(7)]],
        constant uint &overlapLeft [[buffer(8)]],
        uint2 gid [[thread_position_in_grid]]) {
      const uint width = layout.x;
      const uint originX = layout.y;
      const uint validWidth = layout.z;
      const uint validHeight = layout.w;
      if (gid.x >= validWidth || gid.y >= validHeight) return;
      const uint pixel = gid.y * 256 + gid.x;
      const uint pixels = 256 * 256;
      float3 rgb;
      for (uint channel = 0; channel < 3; ++channel) {
        const float value = float(content[channel * pixels + pixel]);
        rgb[channel] = clamp(
          (value - contentMean[channel]) / max(contentStd[channel], 1e-5f)
            * styleStd[channel] + styleMean[channel],
          0.0f, 1.0f
        );
      }
      float4 generated = float4(rgb.b, rgb.g, rgb.r, 1.0f) * 255.0f;
      const uint destination = gid.y * width + originX + gid.x;
      float alpha = 1.0f;
      if (originX > 0 && gid.x < overlapLeft) {
        alpha = raised_cosine(float(gid.x) + 0.5f, overlapLeft);
      }
      const float4 blended = mix(float4(row[destination]), generated, alpha);
      row[destination] = uchar4(clamp(blended + 0.5f, 0.0f, 255.0f));
    }

    kernel void vertical_blend(
        device const uchar4 *row [[buffer(0)]],
        device uchar4 *canvas [[buffer(1)]],
        constant uint4 &layout [[buffer(2)]],
        constant uint &overlapTop [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
      const uint width = layout.x;
      const uint originY = layout.y;
      const uint validWidth = layout.z;
      const uint validHeight = layout.w;
      if (gid.x >= validWidth || gid.y >= validHeight) return;
      const uint source = gid.y * width + gid.x;
      const uint destination = (originY + gid.y) * width + gid.x;
      float alpha = 1.0f;
      if (originY > 0 && gid.y < overlapTop) {
        alpha = raised_cosine(float(gid.y) + 0.5f, overlapTop);
      }
      const float4 blended = mix(float4(canvas[destination]), float4(row[source]), alpha);
      canvas[destination] = uchar4(clamp(blended + 0.5f, 0.0f, 255.0f));
    }
    """#

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let correction: MTLComputePipelineState
    private let vertical: MTLComputePipelineState
    let imageContext: CIContext

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw NativeVideoError.media("Metal is unavailable")
        }
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.source, options: nil)
        } catch {
            throw NativeVideoError.media("Cannot compile Metal compositor: \(error.localizedDescription)")
        }
        guard let correctionFunction = library.makeFunction(name: "correct_horizontal"),
              let verticalFunction = library.makeFunction(name: "vertical_blend") else {
            throw NativeVideoError.media("Metal compositor functions are missing")
        }
        self.device = device
        self.queue = queue
        correction = try device.makeComputePipelineState(function: correctionFunction)
        vertical = try device.makeComputePipelineState(function: verticalFunction)
        imageContext = CIContext(mtlDevice: device, options: [
            .cacheIntermediates: false,
            .highQualityDownsample: true,
        ])
    }

    private func statistics(
        content: UnsafePointer<Float16>,
        style: UnsafePointer<Float16>
    ) -> (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) {
        let pixels = 256 * 256
        var contentMean = SIMD4<Float>(repeating: 0)
        var contentSquared = SIMD4<Float>(repeating: 0)
        var styleMean = SIMD4<Float>(repeating: 0)
        var styleSquared = SIMD4<Float>(repeating: 0)
        for channel in 0..<3 {
            let offset = channel * pixels
            var contentSum: Float = 0
            var contentSquareSum: Float = 0
            var styleSum: Float = 0
            var styleSquareSum: Float = 0
            for pixel in 0..<pixels {
                let contentValue = Float(content[offset + pixel])
                let styleValue = (Float(style[offset + pixel]) + 1) * 0.5
                contentSum += contentValue
                contentSquareSum += contentValue * contentValue
                styleSum += styleValue
                styleSquareSum += styleValue * styleValue
            }
            contentMean[channel] = contentSum / Float(pixels)
            contentSquared[channel] = contentSquareSum / Float(pixels)
            styleMean[channel] = styleSum / Float(pixels)
            styleSquared[channel] = styleSquareSum / Float(pixels)
        }
        var contentStd = SIMD4<Float>(repeating: 1)
        var styleStd = SIMD4<Float>(repeating: 1)
        for channel in 0..<3 {
            contentStd[channel] = sqrt(max(
                1e-5, contentSquared[channel] - contentMean[channel] * contentMean[channel]
            ))
            styleStd[channel] = sqrt(max(
                1e-5, styleSquared[channel] - styleMean[channel] * styleMean[channel]
            ))
        }
        return (contentMean, contentStd, styleMean, styleStd)
    }

    func addChunk(
        video: NDArray,
        references: [NativePreparedFrame],
        internalStart: Int,
        acceptedRange: Range<Int>,
        rowStore: NativeMappedFrames,
        tile: NativeTile,
        previousTile: NativeTile?,
        scale: Int
    ) throws {
        guard video.shape.count == 5, video.shape[0] == 1, video.shape[2] == 3,
              video.shape[3] == 256, video.shape[4] == 256,
              references.count >= video.shape[1] else {
            throw NativeVideoError.tensor("Invalid decoded video tensor")
        }
        guard let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw NativeVideoError.media("Cannot create Metal correction command")
        }
        encoder.setComputePipelineState(correction)
        let pixels = 256 * 256
        let frameScalars = 3 * pixels
        var retained: [MTLBuffer] = []
        let sourceView = video.view(as: Float16.self)
        try sourceView.withUnsafePointer { source, _, _ in
            for localFrame in 0..<video.shape[1] {
                let internalFrame = internalStart + localFrame
                guard acceptedRange.contains(internalFrame) else { continue }
                let outputFrame = internalFrame - acceptedRange.lowerBound
                let content = source.advanced(by: localFrame * frameScalars)
                let style = references[localFrame].values
                let stats = style.withUnsafeBufferPointer { stylePointer in
                    statistics(content: content, style: stylePointer.baseAddress!)
                }
                guard let contentBuffer = device.makeBuffer(
                    bytes: content,
                    length: frameScalars * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                ), let styleBuffer = style.withUnsafeBytes({ bytes in
                    device.makeBuffer(
                        bytes: bytes.baseAddress!, length: bytes.count,
                        options: .storageModeShared
                    )
                }) else {
                    throw NativeVideoError.media("Cannot allocate Metal correction buffers")
                }
                retained += [contentBuffer, styleBuffer]
                encoder.setBuffer(contentBuffer, offset: 0, index: 0)
                encoder.setBuffer(styleBuffer, offset: 0, index: 1)
                encoder.setBuffer(try rowStore.buffer(for: outputFrame), offset: 0, index: 2)
                var contentMean = stats.0
                var contentStd = stats.1
                var styleMean = stats.2
                var styleStd = stats.3
                var layout = SIMD4<UInt32>(
                    UInt32(rowStore.width), UInt32(tile.x * scale),
                    UInt32(tile.validWidth * scale), UInt32(tile.validHeight * scale)
                )
                let overlap = previousTile.map {
                    max(0, ($0.x + $0.validWidth - tile.x) * scale)
                } ?? 0
                var overlapLeft = UInt32(overlap)
                encoder.setBytes(&contentMean, length: MemoryLayout.size(ofValue: contentMean), index: 3)
                encoder.setBytes(&contentStd, length: MemoryLayout.size(ofValue: contentStd), index: 4)
                encoder.setBytes(&styleMean, length: MemoryLayout.size(ofValue: styleMean), index: 5)
                encoder.setBytes(&styleStd, length: MemoryLayout.size(ofValue: styleStd), index: 6)
                encoder.setBytes(&layout, length: MemoryLayout.size(ofValue: layout), index: 7)
                encoder.setBytes(&overlapLeft, length: MemoryLayout.size(ofValue: overlapLeft), index: 8)
                encoder.dispatchThreads(
                    MTLSize(width: tile.validWidth * scale, height: tile.validHeight * scale, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
                )
            }
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        withExtendedLifetime(retained) {}
        if command.status == .error {
            throw NativeVideoError.media(
                command.error?.localizedDescription ?? "Metal color correction failed"
            )
        }
    }

    func finishRow(
        rowStore: NativeMappedFrames,
        canvas: NativeMappedFrames,
        row: NativeTileRow,
        previousRow: NativeTileRow?,
        scale: Int
    ) throws {
        guard let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw NativeVideoError.media("Cannot create Metal stitching command")
        }
        encoder.setComputePipelineState(vertical)
        let originY = row.y * scale
        let validHeight = row.validHeight * scale
        let overlap = previousRow.map {
            max(0, ($0.y + $0.validHeight - row.y) * scale)
        } ?? 0
        for frame in 0..<canvas.frameCount {
            encoder.setBuffer(try rowStore.buffer(for: frame), offset: 0, index: 0)
            encoder.setBuffer(try canvas.buffer(for: frame), offset: 0, index: 1)
            var layout = SIMD4<UInt32>(
                UInt32(canvas.width), UInt32(originY),
                UInt32(canvas.width), UInt32(validHeight)
            )
            var overlapTop = UInt32(overlap)
            encoder.setBytes(&layout, length: MemoryLayout.size(ofValue: layout), index: 2)
            encoder.setBytes(&overlapTop, length: MemoryLayout.size(ofValue: overlapTop), index: 3)
            encoder.dispatchThreads(
                MTLSize(width: canvas.width, height: validHeight, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
            )
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if command.status == .error {
            throw NativeVideoError.media(
                command.error?.localizedDescription ?? "Metal tile stitching failed"
            )
        }
    }
}

@available(macOS 27.0, *)
private final class NativeSegmentCompositor {
    private let canvas: NativeMappedFrames
    private let rowStore: NativeMappedFrames
    private let metal: NativeMetalCompositor
    private let scale: Int
    private let frameCount: Int

    init(
        sourceWidth: Int,
        sourceHeight: Int,
        scale: Int,
        frameCount: Int,
        directory: URL,
        metal: NativeMetalCompositor
    ) throws {
        self.scale = scale
        self.frameCount = frameCount
        self.metal = metal
        let canvasBytes = Int64(sourceWidth) * Int64(scale)
            * Int64(sourceHeight) * Int64(scale) * 4 * Int64(frameCount)
        let rowBytes = Int64(sourceWidth) * Int64(scale) * 256 * 4 * Int64(frameCount)
        let requiredBytes = canvasBytes + rowBytes
        let capacity = try? directory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage
        if let capacity, requiredBytes > capacity {
            throw NativeVideoError.media(
                "Insufficient temporary storage: needs "
                    + ByteCountFormatter.string(
                        fromByteCount: requiredBytes, countStyle: .file
                    )
            )
        }
        print(
            "Mapped composition store: "
                + ByteCountFormatter.string(fromByteCount: requiredBytes, countStyle: .file)
        )
        fflush(stdout)
        canvas = try NativeMappedFrames(
            width: sourceWidth * scale, height: sourceHeight * scale,
            frameCount: frameCount, directory: directory, stem: "canvas",
            device: metal.device
        )
        rowStore = try NativeMappedFrames(
            width: sourceWidth * scale, height: 256,
            frameCount: frameCount, directory: directory, stem: "row",
            device: metal.device
        )
    }

    func addChunk(
        video: NDArray,
        references: [NativePreparedFrame],
        internalStart: Int,
        warmupFrameCount: Int,
        tile: NativeTile,
        previousTile: NativeTile?
    ) throws {
        try metal.addChunk(
            video: video, references: references, internalStart: internalStart,
            acceptedRange: warmupFrameCount..<(warmupFrameCount + frameCount),
            rowStore: rowStore, tile: tile, previousTile: previousTile, scale: scale
        )
    }

    func finishRow(_ row: NativeTileRow, previousRow: NativeTileRow?) throws {
        try metal.finishRow(
            rowStore: rowStore, canvas: canvas, row: row,
            previousRow: previousRow, scale: scale
        )
        rowStore.discardAll()
    }

    func appendFrames(
        to writer: NativeVideoWriter,
        outputWidth: Int,
        outputHeight: Int,
        presentationTimes: [CMTime],
        reportFrames: (Int) -> Void
    ) async throws {
        guard presentationTimes.count == frameCount else {
            throw NativeVideoError.writer("Presentation-time count does not match segment frames")
        }
        for frame in 0..<frameCount {
            let destination = try writer.makePixelBuffer()
            let sourcePointer = try canvas.pointer(for: frame)
            if outputWidth == canvas.width, outputHeight == canvas.height {
                CVPixelBufferLockBaseAddress(destination, [])
                guard let destinationBase = CVPixelBufferGetBaseAddress(destination) else {
                    CVPixelBufferUnlockBaseAddress(destination, [])
                    throw NativeVideoError.writer("Output pixel buffer has no address")
                }
                let destinationRowBytes = CVPixelBufferGetBytesPerRow(destination)
                for y in 0..<canvas.height {
                    memcpy(
                        destinationBase.advanced(by: y * destinationRowBytes),
                        sourcePointer.advanced(by: y * canvas.rowBytes),
                        canvas.rowBytes
                    )
                }
                CVPixelBufferUnlockBaseAddress(destination, [])
            } else {
                var sourceBuffer: CVPixelBuffer?
                let status = CVPixelBufferCreateWithBytes(
                    kCFAllocatorDefault,
                    canvas.width,
                    canvas.height,
                    kCVPixelFormatType_32BGRA,
                    sourcePointer,
                    canvas.rowBytes,
                    nil,
                    nil,
                    nil,
                    &sourceBuffer
                )
                guard status == kCVReturnSuccess, let sourceBuffer else {
                    throw NativeVideoError.writer("Cannot expose the composed frame: \(status)")
                }
                let filter = CIFilter(name: "CILanczosScaleTransform")
                filter?.setValue(CIImage(cvPixelBuffer: sourceBuffer), forKey: kCIInputImageKey)
                let verticalScale = Double(outputHeight) / Double(canvas.height)
                let horizontalScale = Double(outputWidth) / Double(canvas.width)
                filter?.setValue(verticalScale, forKey: kCIInputScaleKey)
                filter?.setValue(horizontalScale / verticalScale, forKey: kCIInputAspectRatioKey)
                guard let image = filter?.outputImage?.cropped(
                    to: CGRect(x: 0, y: 0, width: outputWidth, height: outputHeight)
                ) else {
                    throw NativeVideoError.writer("Lanczos resize could not be created")
                }
                metal.imageContext.render(
                    image,
                    to: destination,
                    bounds: CGRect(x: 0, y: 0, width: outputWidth, height: outputHeight),
                    colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
                )
            }
            try await writer.append(
                destination, presentationTime: presentationTimes[frame]
            )
            canvas.discard(frame: frame)
            reportFrames(frame + 1)
        }
    }
}

@available(macOS 27.0, *)
private func processTileSegment(
    tile: NativeTile,
    previousTile: NativeTile?,
    metadata: NativeVideoMetadata,
    scale: Int,
    seed: UInt64,
    pipeline: FlashVSRNativePipeline,
    decodedSegment: NativeDecodedSegment,
    preparer: NativeFramePreparer,
    compositor: NativeSegmentCompositor,
    segmentStart: Int,
    segmentFrameCount: Int,
    warmupFrameCount: Int,
    reportInternalFrames: (Int, Int) -> Void
) async throws {
    let processingStart = segmentStart - warmupFrameCount
    let processingFrameCount = warmupFrameCount + segmentFrameCount
    let reader = NativeTileFrameReader(
        metadata: metadata,
        tile: tile,
        scale: scale,
        segment: decodedSegment,
        preparer: preparer
    )
    let firstFrame = try reader.next()
    var groups: [Int: NativePreparedGroup] = [:]
    groups[0] = try makeGroup([firstFrame, firstFrame, firstFrame, firstFrame])
    for groupIndex in 1...6 {
        groups[groupIndex] = try makeGroup((0..<4).map { _ in try reader.next() })
    }
    let normal = NativeCoordinateNormalGenerator(
        seed: seed,
        outputOriginX: tile.x * scale,
        outputOriginY: tile.y * scale
    )
    let lqFirst = (1...6).map { groups[$0]!.tensor }
    let decoderFirst = (0...5).map { groups[$0]!.tensor }
    var latentFrameStart = processingStart - 3
    var chunk = try await pipeline.firstChunk(
        latent: normal.latent(frames: 6, temporalFrameStart: latentFrameStart),
        firstLQFrame: makeWarmup(firstFrame),
        lqGroups: lqFirst,
        decoderGroups: decoderFirst
    )
    var references = [firstFrame]
    for groupIndex in 1...5 { references.append(contentsOf: groups[groupIndex]!.frames) }
    try compositor.addChunk(
        video: chunk.video,
        references: references,
        internalStart: 0,
        warmupFrameCount: warmupFrameCount,
        tile: tile,
        previousTile: previousTile
    )
    var written = 0
    written += min(chunk.video.shape[1], processingFrameCount)
    reportInternalFrames(written, processingFrameCount)
    latentFrameStart += 6 * 4
    let paddedOutput = processingFrameCount < 21
        ? 21 : ((processingFrameCount - 5 + 7) / 8) * 8 + 5
    let chunkCount = (paddedOutput + 4 - 1) / 8 - 2
    if chunkCount > 1 {
        for chunkIndex in 1..<chunkCount {
            let firstNew = 2 * chunkIndex + 5
            let secondNew = firstNew + 1
            groups[firstNew] = try makeGroup((0..<4).map { _ in try reader.next() })
            groups[secondNew] = try makeGroup((0..<4).map { _ in try reader.next() })
            let decoderStart = 2 * chunkIndex + 4
            chunk = try await pipeline.nextChunk(
                latent: normal.latent(frames: 2, temporalFrameStart: latentFrameStart),
                lqGroups: [groups[firstNew]!.tensor, groups[secondNew]!.tensor],
                decoderGroups: [groups[decoderStart]!.tensor, groups[decoderStart + 1]!.tensor],
                state: chunk.state
            )
            references = groups[decoderStart]!.frames + groups[decoderStart + 1]!.frames
            try compositor.addChunk(
                video: chunk.video,
                references: references,
                internalStart: written,
                warmupFrameCount: warmupFrameCount,
                tile: tile,
                previousTile: previousTile
            )
            written += min(chunk.video.shape[1], processingFrameCount - written)
            latentFrameStart += 2 * 4
            for old in Array(groups.keys) where old < decoderStart { groups.removeValue(forKey: old) }
            reportInternalFrames(written, processingFrameCount)
        }
    }
    guard written == processingFrameCount else {
        throw NativeVideoError.tensor(
            "Tile emitted \(written)/\(processingFrameCount) segment frames"
        )
    }
    reportInternalFrames(processingFrameCount, processingFrameCount)
}

@available(macOS 27.0, *)
private final class NativeProgressReporter {
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
private struct FlashVSRNativeVideoRunner {
    static func main() async {
        do {
            let arguments = try NativeVideoArguments.parse()
            guard !FileManager.default.fileExists(atPath: arguments.output.path) else {
                throw NativeVideoError.argument("Output already exists: \(arguments.output.path)")
            }
            let metadata = try await NativeVideoMetadata.load(url: arguments.input)
            let rows = nativeTileRows(
                width: metadata.width, height: metadata.height, scale: arguments.scale
            )
            let tiles = rows.flatMap(\.tiles)
            let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
                "flashvsr-coreai-\(UUID().uuidString)", isDirectory: true
            )
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: temporary) }
            let segmentCount = (metadata.frameCount + nativeTemporalSegmentFrames - 1)
                / nativeTemporalSegmentFrames
            print(
                "FlashVSR-v1.1 Core AI: \(metadata.width)x\(metadata.height), "
                    + "\(metadata.frameCount) frames, \(tiles.count) tiles, "
                    + "\(segmentCount) temporal segments"
            )
            fflush(stdout)
            print("STAGE Metal合成器を準備中")
            fflush(stdout)
            let metal = try NativeMetalCompositor()
            print("STAGE FlashVSRモデルを読み込み中")
            fflush(stdout)
            let pipeline = try await FlashVSRNativePipeline(
                modelsDirectory: arguments.models,
                computePolicy: arguments.computePolicy
            )
            print("Compute policy: \(pipeline.computeSummary)")
            fflush(stdout)
            let nativeOutputWidth = metadata.width * arguments.scale
            let nativeOutputHeight = metadata.height * arguments.scale
            let outputWidth = arguments.outputWidth ?? nativeOutputWidth
            let outputHeight = arguments.outputHeight ?? nativeOutputHeight
            let bitRate = Int(min(160_000_000, max(12_000_000,
                Double(outputWidth * outputHeight) * metadata.frameRate * 0.12)))
            let writer = try NativeVideoWriter(
                url: arguments.output,
                width: outputWidth,
                height: outputHeight,
                frameRate: metadata.frameRate,
                bitRate: bitRate,
                fragmentIntervalFrames: nativeTemporalSegmentFrames,
                expectedFrameCount: metadata.frameCount
            )
            let progress = NativeProgressReporter()
            let preparer = NativeFramePreparer()
            var segmentStart = 0
            while segmentStart < metadata.frameCount {
                let frameCount = min(
                    nativeTemporalSegmentFrames,
                    metadata.frameCount - segmentStart
                )
                let warmup = min(nativeTemporalWarmupFrames, segmentStart)
                let segmentIndex = segmentStart / nativeTemporalSegmentFrames
                let segmentDirectory = temporary.appendingPathComponent(
                    String(format: "segment-%06d", segmentIndex),
                    isDirectory: true
                )
                try FileManager.default.createDirectory(
                    at: segmentDirectory,
                    withIntermediateDirectories: true
                )
                print(
                    "SEGMENT \(segmentIndex + 1)/\(segmentCount) "
                        + "frames \(segmentStart)...\(segmentStart + frameCount - 1) "
                        + "warmup=\(warmup)"
                )
                fflush(stdout)
                let compositor = try NativeSegmentCompositor(
                    sourceWidth: metadata.width,
                    sourceHeight: metadata.height,
                    scale: arguments.scale,
                    frameCount: frameCount,
                    directory: segmentDirectory,
                    metal: metal
                )
                let processingStart = segmentStart - warmup
                let processingFrameCount = warmup + frameCount
                do {
                    print(
                        "STAGE セグメント \(segmentIndex + 1)/\(segmentCount)を共有デコード中"
                    )
                    fflush(stdout)
                    let decoded = try NativeDecodedSegment(
                        metadata: metadata,
                        startFrame: processingStart,
                        processingFrameCount: processingFrameCount
                    ) { decodedFrames, requestedFrames in
                        let local = 0.05 * Double(decodedFrames) / Double(requestedFrames)
                        progress.emit(
                            100.0 * (Double(segmentStart) + Double(frameCount) * local)
                                / Double(metadata.frameCount)
                        )
                    }
                    var flatTileIndex = 0
                    var previousRow: NativeTileRow?
                    for (rowIndex, row) in rows.enumerated() {
                        print(
                            "STAGE セグメント \(segmentIndex + 1)/\(segmentCount)・"
                                + "タイル行 \(rowIndex + 1)/\(rows.count)を推論中"
                        )
                        fflush(stdout)
                        var previousTile: NativeTile?
                        for tile in row.tiles {
                            let currentTileIndex = flatTileIndex
                            try await processTileSegment(
                                tile: tile,
                                previousTile: previousTile,
                                metadata: metadata,
                                scale: arguments.scale,
                                seed: arguments.seed,
                                pipeline: pipeline,
                                decodedSegment: decoded,
                                preparer: preparer,
                                compositor: compositor,
                                segmentStart: segmentStart,
                                segmentFrameCount: frameCount,
                                warmupFrameCount: warmup
                            ) { completedInternalFrames, totalInternalFrames in
                                let tileFraction = (
                                    Double(currentTileIndex)
                                        + Double(completedInternalFrames)
                                            / Double(totalInternalFrames)
                                ) / Double(tiles.count)
                                let local = 0.05 + 0.85 * tileFraction
                                progress.emit(
                                    100.0 * (Double(segmentStart) + Double(frameCount) * local)
                                        / Double(metadata.frameCount)
                                )
                            }
                            previousTile = tile
                            flatTileIndex += 1
                        }
                        try compositor.finishRow(row, previousRow: previousRow)
                        previousRow = row
                    }
                }
                print(
                    "STAGE セグメント \(segmentIndex + 1)/\(segmentCount)を最終動画へ書き込み中"
                )
                fflush(stdout)
                try await compositor.appendFrames(
                    to: writer,
                    outputWidth: outputWidth,
                    outputHeight: outputHeight,
                    presentationTimes: (0..<frameCount).map {
                        metadata.presentationTime(for: segmentStart + $0)
                    }
                ) { writtenFrames in
                    let local = 0.90 + 0.10 * Double(writtenFrames) / Double(frameCount)
                    progress.emit(
                        100.0 * (Double(segmentStart) + Double(frameCount) * local)
                            / Double(metadata.frameCount)
                    )
                }
                try FileManager.default.removeItem(at: segmentDirectory)
                segmentStart += frameCount
            }
            try await writer.finish()
            progress.emit(100)
            print("STAGE 完了")
            print("FlashVSR-v1.1 Core AI completed: \(arguments.output.path)")
        } catch {
            FileHandle.standardError.write(
                Data("flashvsr-coreai-video: \(error.localizedDescription)\n".utf8)
            )
            exit(EXIT_FAILURE)
        }
    }
}
