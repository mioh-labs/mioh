import AVFoundation
import CoreGraphics
import CoreImage
import CoreMedia
import CoreVideo
import Foundation

enum H3NativeMedia {
  private static let imageContext = CIContext(options: [
    .workingColorSpace: NSNull(),
    .outputColorSpace: NSNull(),
    .cacheIntermediates: false,
  ])

  static func probe(_ url: URL) async throws
    -> (duration: Double, width: Int, height: Int, hasAudio: Bool)
  {
    let asset = AVURLAsset(url: url)
    let duration = try await asset.load(.duration).seconds
    guard duration.isFinite, duration > 0 else {
      throw H3NativeError.media("input duration is unavailable")
    }
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw H3NativeError.media("input has no video track")
    }
    let size = try await track.load(.naturalSize)
    let transform = try await track.load(.preferredTransform)
    let transformed = CGRect(origin: .zero, size: size).applying(transform)
    let audio = try await asset.loadTracks(withMediaType: .audio)
    return (
      duration,
      max(1, Int(abs(transformed.width).rounded())),
      max(1, Int(abs(transformed.height).rounded())),
      !audio.isEmpty
    )
  }

  static func probeDuration(_ url: URL) async throws -> Double {
    let duration = try await AVURLAsset(url: url).load(.duration).seconds
    guard duration.isFinite, duration > 0 else {
      throw H3NativeError.media("media duration is unavailable")
    }
    return duration
  }

  static func probeImages(_ urls: [URL]) throws -> (width: Int, height: Int) {
    guard !urls.isEmpty else {
      throw H3NativeError.media("select at least one input image")
    }
    var firstSize: (width: Int, height: Int)?
    for url in urls {
      guard let image = CIImage(
        contentsOf: url,
        options: [.applyOrientationProperty: true]
      ) else {
        throw H3NativeError.media("cannot decode input image: \(url.path)")
      }
      let extent = image.extent.integral
      guard extent.width >= 1, extent.height >= 1 else {
        throw H3NativeError.media("input image has no pixels: \(url.path)")
      }
      if firstSize == nil {
        firstSize = (Int(extent.width), Int(extent.height))
      }
    }
    return firstSize!
  }

  static func decodeReferenceImages(
    urls: [URL],
    width: Int,
    height: Int,
    frameCount: Int
  ) throws -> H3Tensor {
    _ = try probeImages(urls)
    guard frameCount > 0 else {
      throw H3NativeError.media("reference image frame count must be positive")
    }
    return try decodeImageFrames(
      urls: urls,
      imageIndices: [Int](repeating: 0, count: frameCount),
      width: width,
      height: height
    )
  }

  static func decodeReferenceImage(
    url: URL,
    width: Int,
    height: Int
  ) throws -> H3Tensor {
    try decodeImageFrames(
      urls: [url],
      imageIndices: [0],
      width: width,
      height: height
    )
  }

  static func decodeIdentityReferenceImages(
    urls: [URL],
    width: Int,
    height: Int
  ) throws -> H3Tensor {
    _ = try probeImages(urls)
    // Qwen consumes one two-frame vision block per still image. The second
    // frame is an identical temporal-patch mate, not a request to turn the
    // image into a ten-second reference video.
    let indices = urls.indices.flatMap { [$0, $0] }
    return try decodeImageFrames(
      urls: urls,
      imageIndices: indices,
      width: width,
      height: height
    )
  }

  static func decodeReferenceImageSequence(
    urls: [URL],
    width: Int,
    height: Int
  ) throws -> H3Tensor {
    _ = try probeImages(urls)
    return try decodeImageFrames(
      urls: urls,
      imageIndices: Array(urls.indices),
      width: width,
      height: height
    )
  }

  private static func decodeImageFrames(
    urls: [URL],
    imageIndices: [Int],
    width: Int,
    height: Int
  ) throws -> H3Tensor {
    guard !urls.isEmpty, !imageIndices.isEmpty,
      imageIndices.allSatisfy({ urls.indices.contains($0) })
    else {
      throw H3NativeError.media("invalid identity reference image sequence")
    }
    let pool = try makePixelBufferPool(width: width, height: height)
    let plane = width * height
    let frameCount = imageIndices.count
    var bytes = Data(count: 3 * plane * frameCount * MemoryLayout<Float16>.stride)
    var activeImageIndex = -1
    var activePixelBuffer: CVPixelBuffer?
    for frame in 0..<frameCount {
      let imageIndex = imageIndices[frame]
      if activePixelBuffer == nil || activeImageIndex != imageIndex {
        guard let image = CIImage(
          contentsOf: urls[imageIndex],
          options: [.applyOrientationProperty: true]
        ) else {
          throw H3NativeError.media(
            "cannot decode input image: \(urls[imageIndex].path)"
          )
        }
        activePixelBuffer = try render(
          image,
          width: width,
          height: height,
          pool: pool
        )
        activeImageIndex = imageIndex
      }
      try bytes.withUnsafeMutableBytes { raw in
        try appendNCTHW(
          activePixelBuffer!, frame: frame, frameCount: frameCount,
          destination: raw.bindMemory(to: Float16.self)
        )
      }
    }
    return try H3Tensor(
      shape: [1, 3, frameCount, height, width],
      scalarType: .float16,
      bytes: bytes
    )
  }

  static func silentAudio(
    durationSeconds: Double,
    sampleRate: Int = 32_000
  ) throws -> H3Tensor {
    let sampleCount = max(
      1,
      Int((durationSeconds * Double(sampleRate)).rounded())
    )
    return try silentAudio(sampleFrames: sampleCount)
  }

  static func silentAudio(sampleFrames: Int) throws -> H3Tensor {
    let sampleCount = max(1, sampleFrames)
    return try H3Tensor(
      float32: [Float](repeating: 0, count: 2 * sampleCount),
      shape: [1, 2, sampleCount]
    )
  }

  static func exactAudioGridSampleFrames(
    latentFrames: Int,
    sampleRate: Int = 32_000
  ) throws -> Int {
    guard latentFrames > 0 else {
      throw H3NativeError.invalidJob(
        "H3 audio grid needs at least one latent frame"
      )
    }
    guard sampleRate > 0,
      sampleRate % H3Geometry.audioLatentFramesPerSecond == 0
    else {
      throw H3NativeError.invalidJob(
        "H3 audio grid requires an integral \(H3Geometry.audioLatentFramesPerSecond)Hz latent rate at \(sampleRate)Hz"
      )
    }
    return latentFrames
      * (sampleRate / H3Geometry.audioLatentFramesPerSecond)
  }

  static func decodeReferenceVideo(
    url: URL,
    width: Int,
    height: Int,
    frameCount: Int
  ) async throws -> H3Tensor {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw H3NativeError.media("input has no video track")
    }
    let transform = try await track.load(.preferredTransform)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_32BGRA)
      ]
    )
    let provider = reader.outputProvider(for: output)
    do {
      try reader.start()
    } catch {
      throw H3NativeError.media(
        "AVAssetReader did not start: \(error.localizedDescription)"
      )
    }
    let pool = try makePixelBufferPool(width: width, height: height)
    let plane = width * height
    let frameElements = 3 * plane
    var bytes = Data(
      count: frameElements * frameCount * MemoryLayout<Float16>.stride
    )
    var firstTimestamp: Double?
    var selected = 0
    while selected < frameCount, let sample = try await provider.next() {
      let timestamp = sample.presentationTimeStamp.seconds
      if firstTimestamp == nil { firstTimestamp = timestamp }
      let relative = timestamp - (firstTimestamp ?? timestamp)
      let wanted = Double(selected) / Double(H3Geometry.framesPerSecond)
      guard relative + 1e-7 >= wanted,
        let pixelSample = CMReadySampleBuffer<CVReadOnlyPixelBuffer>(sample)
      else { continue }
      let source = pixelSample.content.withUnsafeBuffer { $0 }
      let rendered = try autoreleasepool {
        try render(
          source, transform: transform, width: width, height: height, pool: pool
        )
      }
      try bytes.withUnsafeMutableBytes { raw in
        try appendNCTHW(
          rendered, frame: selected, frameCount: frameCount,
          destination: raw.bindMemory(to: Float16.self)
        )
      }
      selected += 1
    }
    guard selected == frameCount, reader.status != .failed else {
      throw H3NativeError.media(
        "decoded \(selected)/\(frameCount) reference frames: "
          + (reader.error?.localizedDescription ?? "input ended early")
      )
    }
    return try H3Tensor(
      shape: [1, 3, frameCount, height, width],
      scalarType: .float16,
      bytes: bytes
    )
  }

  static func decodeReferenceAudio(
    url: URL,
    durationSeconds: Double,
    startSeconds: Double = 0,
    sampleRate: Int = 32_000,
    exactSampleFrames: Int? = nil
  ) async throws -> H3Tensor {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
      if let exactSampleFrames {
        return try silentAudio(sampleFrames: exactSampleFrames)
      }
      return try silentAudio(durationSeconds: durationSeconds, sampleRate: sampleRate)
    }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: sampleRate,
        AVNumberOfChannelsKey: 2,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
      ]
    )
    let provider = reader.outputProvider(for: output)
    do {
      try reader.start()
    } catch {
      throw H3NativeError.media(
        "audio reader did not start: \(error.localizedDescription)"
      )
    }
    let maximumFrames = max(
      1,
      exactSampleFrames
        ?? Int((durationSeconds * Double(sampleRate)).rounded())
    )
    var discardFrames = max(
      0,
      Int((startSeconds * Double(sampleRate)).rounded())
    )
    var interleaved: [Float] = []
    interleaved.reserveCapacity(maximumFrames * 2)
    while interleaved.count < maximumFrames * 2,
      let sample = try await provider.next()
    {
      let payload: Data? = try sample.withUnsafeSampleBuffer { sampleBuffer in
        guard let block = CMSampleBufferGetDataBuffer(sampleBuffer) else {
          return nil
        }
        let byteCount = CMBlockBufferGetDataLength(block)
        guard byteCount > 0,
          byteCount % MemoryLayout<Float>.stride == 0
        else { return nil }
        var data = Data(count: byteCount)
        let status = data.withUnsafeMutableBytes { raw in
          CMBlockBufferCopyDataBytes(
            block,
            atOffset: 0,
            dataLength: byteCount,
            destination: raw.baseAddress!
          )
        }
        guard status == noErr else {
          throw H3NativeError.media("audio block copy returned \(status)")
        }
        return data
      }
      guard let payload else { continue }
      let values = payload.withUnsafeBytes {
        Array($0.bindMemory(to: Float.self))
      }
      guard values.count % 2 == 0 else {
        throw H3NativeError.media("decoded audio block is not interleaved stereo")
      }
      let payloadFrames = values.count / 2
      // AVFoundation may label decoder pre-roll with the requested time-range
      // timestamp even though the PCM still starts at an earlier FLAC seek
      // point. Decode continuously from the beginning and discard an exact
      // sample count instead of trusting compressed-audio seek timestamps.
      let skipFrames = min(discardFrames, payloadFrames)
      discardFrames -= skipFrames
      guard discardFrames == 0, skipFrames < payloadFrames else { continue }
      let writtenFrames = interleaved.count / 2
      let appendFrames = min(
        payloadFrames - skipFrames,
        maximumFrames - writtenFrames
      )
      let lower = skipFrames * 2
      let upper = lower + appendFrames * 2
      interleaved.append(contentsOf: values[lower..<upper])
    }
    guard reader.status != .failed else {
      throw H3NativeError.media(
        reader.error?.localizedDescription ?? "audio decoding failed"
      )
    }
    if interleaved.count > maximumFrames * 2 {
      interleaved.removeSubrange((maximumFrames * 2)..<interleaved.count)
    }
    if interleaved.count < maximumFrames * 2 {
      interleaved += [Float](
        repeating: 0,
        count: maximumFrames * 2 - interleaved.count
      )
    }
    var planar = [Float](repeating: 0, count: maximumFrames * 2)
    for frame in 0..<maximumFrames {
      planar[frame] = interleaved[frame * 2]
      planar[maximumFrames + frame] = interleaved[frame * 2 + 1]
    }
    return try H3Tensor(
      float32: planar,
      shape: [1, 2, maximumFrames]
    )
  }

  static func fitAudioLatent(
    _ source: H3Tensor,
    to targetShape: [Int]
  ) throws -> H3Tensor {
    guard source.shape.count == 4, targetShape.count == 4,
      source.shape[0] == targetShape[0],
      source.shape[1] == targetShape[1],
      source.shape[2] == targetShape[2]
    else {
      throw H3NativeError.invalidTensor(
        "audio conditioning latent cannot fit \(source.shape) to \(targetShape)"
      )
    }
    let sourceValues = try source.floatValues()
    let sourceTime = source.shape[3]
    let targetTime = targetShape[3]
    let rows = targetShape[0] * targetShape[1] * targetShape[2]
    let copyTime = min(sourceTime, targetTime)
    var target = [Float](repeating: 0, count: rows * targetTime)
    for row in 0..<rows {
      let sourceStart = row * sourceTime
      let targetStart = row * targetTime
      target.replaceSubrange(
        targetStart..<(targetStart + copyTime),
        with: sourceValues[sourceStart..<(sourceStart + copyTime)]
      )
    }
    return try H3Tensor(float32: target, shape: targetShape)
  }

  static func writeReferenceImage(
    video: H3Tensor,
    frame: Int,
    outputURL: URL
  ) throws {
    guard video.shape.count == 5, video.shape[0] == 1, video.shape[1] == 3,
      video.shape.indices.contains(2), video.shape[2] > frame, frame >= 0
    else {
      throw H3NativeError.invalidTensor(
        "continuation image frame is outside the decoded video"
      )
    }
    let width = video.shape[4]
    let height = video.shape[3]
    let pool = try makePixelBufferPool(width: width, height: height)
    var allocatedPixelBuffer: CVPixelBuffer?
    let allocationResult = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pool,
      &allocatedPixelBuffer
    )
    guard allocationResult == kCVReturnSuccess,
      let pixelBuffer = allocatedPixelBuffer
    else {
      throw H3NativeError.media(
        "continuation image pixel allocation returned \(allocationResult)"
      )
    }
    let pixels = try video.floatValues()
    try writeBGRA(pixels, shape: video.shape, frame: frame, to: pixelBuffer)
    try FileManager.default.createDirectory(
      at: outputURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    if FileManager.default.fileExists(atPath: outputURL.path) {
      try FileManager.default.removeItem(at: outputURL)
    }
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
      ?? CGColorSpaceCreateDeviceRGB()
    try imageContext.writePNGRepresentation(
      of: CIImage(cvPixelBuffer: pixelBuffer),
      to: outputURL,
      format: .RGBA8,
      colorSpace: colorSpace,
      options: [:]
    )
  }

  @available(macOS 27.0, *)
  static func writeMovie(
    video: H3Tensor,
    audio: H3Tensor?,
    outputURL: URL,
    durationSeconds: Double,
    trimStartSeconds: Double = 0,
    outputWidth: Int? = nil,
    outputHeight: Int? = nil,
    frameRate: Int = 24,
    audioSampleRate: Int = 32_000
  ) async throws {
    guard video.shape.count == 5, video.shape[0] == 1, video.shape[1] == 3 else {
      throw H3NativeError.invalidTensor("decoded video must be NCTHW RGB")
    }
    guard trimStartSeconds.isFinite, trimStartSeconds >= 0 else {
      throw H3NativeError.media("movie trim start must be non-negative")
    }
    let sourceStartFrame = Int(
      (trimStartSeconds * Double(frameRate)).rounded()
    )
    let requestedFrameCount = max(
      1,
      Int((durationSeconds * Double(frameRate)).rounded())
    )
    guard sourceStartFrame + requestedFrameCount <= video.shape[2] else {
      throw H3NativeError.media(
        "movie trim requires \(sourceStartFrame + requestedFrameCount) frames, got \(video.shape[2])"
      )
    }
    let frameCount = requestedFrameCount
    let sourceHeight = video.shape[3]
    let sourceWidth = video.shape[4]
    let height = outputHeight ?? sourceHeight
    let width = outputWidth ?? sourceWidth
    guard width > 0, height > 0, width <= sourceWidth, height <= sourceHeight,
      (sourceWidth - width) % 2 == 0,
      (sourceHeight - height) % 2 == 0
    else {
      throw H3NativeError.media(
        "movie output must be a centered crop of the decoded video"
      )
    }
    let pixels = try video.floatValues()
    let fileManager = FileManager.default
    try fileManager.createDirectory(
      at: outputURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    if fileManager.fileExists(atPath: outputURL.path) {
      try fileManager.removeItem(at: outputURL)
    }
    let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
    let videoInput = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.hevc,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: max(6_000_000, width * height * 8),
          AVVideoExpectedSourceFrameRateKey: frameRate,
          AVVideoMaxKeyFrameIntervalKey: frameRate * 2,
        ],
      ]
    )
    var pixelBufferAttributes = CVPixelBufferCreationAttributes(
      pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA),
      size: CVImageSize(width: width, height: height)
    )
    pixelBufferAttributes.backing = .ioSurface
    let videoReceiver = writer.inputPixelBufferReceiver(
      for: videoInput,
      pixelBufferAttributes: pixelBufferAttributes
    )

    var audioReceiver: AVAssetWriterInput.SampleBufferReceiver?
    var audioSamples: [Float]?
    if let audio {
      let decodedSamples = try interleavedAudio(audio)
      let trimSampleValues = max(
        0,
        Int((trimStartSeconds * Double(audioSampleRate)).rounded()) * 2
      )
      let requestedSampleValues = max(
        2,
        Int((durationSeconds * Double(audioSampleRate)).rounded()) * 2
      )
      let upper = min(
        decodedSamples.count,
        trimSampleValues + requestedSampleValues
      )
      var selected = trimSampleValues < upper
        ? Array(decodedSamples[trimSampleValues..<upper])
        : []
      if selected.count < requestedSampleValues {
        selected += [Float](
          repeating: 0,
          count: requestedSampleValues - selected.count
        )
      }
      audioSamples = selected
      let input = AVAssetWriterInput(
        mediaType: .audio,
        outputSettings: [
          AVFormatIDKey: kAudioFormatMPEG4AAC,
          AVSampleRateKey: audioSampleRate,
          AVNumberOfChannelsKey: 2,
          AVEncoderBitRateKey: 192_000,
        ]
      )
      audioReceiver = writer.inputReceiver(for: input)
    }
    try writer.start()
    writer.startSession(atSourceTime: .zero)
    guard let pool = videoReceiver.pixelBufferPool else {
      throw H3NativeError.media("writer pixel buffer pool is unavailable")
    }
    let audioTask: Task<Void, Error>?
    if let audioReceiver, let audioSamples {
      audioTask = Task {
        defer { audioReceiver.finish() }
        try await appendAudio(
          audioSamples,
          sampleRate: audioSampleRate,
          receiver: audioReceiver
        )
      }
    } else {
      audioTask = nil
    }
    do {
      for frame in 0..<frameCount {
        let pixelBuffer = try pool.makeMutablePixelBuffer()
        try pixelBuffer.withUnsafeBuffer { unsafeBuffer in
          try writeBGRA(
            pixels,
            shape: video.shape,
            frame: sourceStartFrame + frame,
            to: unsafeBuffer
          )
        }
        let timestamp = CMTime(
          value: CMTimeValue(frame),
          timescale: CMTimeScale(frameRate)
        )
        try await videoReceiver.append(
          CVReadOnlyPixelBuffer(pixelBuffer),
          with: timestamp
        )
      }
      videoReceiver.finish()
      try await audioTask?.value
    } catch {
      audioTask?.cancel()
      videoReceiver.finish()
      audioReceiver?.finish()
      writer.cancelWriting()
      throw error
    }
    await writer.finishWriting()
    guard writer.status == .completed else {
      throw H3NativeError.media(
        writer.error?.localizedDescription ?? "movie finalization failed"
      )
    }
  }

  private static func makePixelBufferPool(width: Int, height: Int) throws
    -> CVPixelBufferPool
  {
    let attributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
    ]
    var pool: CVPixelBufferPool?
    let result = CVPixelBufferPoolCreate(
      kCFAllocatorDefault,
      nil,
      attributes as CFDictionary,
      &pool
    )
    guard result == kCVReturnSuccess, let pool else {
      throw H3NativeError.media("pixel buffer pool returned \(result)")
    }
    return pool
  }

  private static func render(
    _ source: CVPixelBuffer,
    transform: CGAffineTransform,
    width: Int,
    height: Int,
    pool: CVPixelBufferPool
  ) throws -> CVPixelBuffer {
    var output: CVPixelBuffer?
    let result = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pool,
      &output
    )
    guard result == kCVReturnSuccess, let output else {
      throw H3NativeError.media("render pixel allocation returned \(result)")
    }
    let image = CIImage(cvPixelBuffer: source).transformed(by: transform)
    return try render(image, width: width, height: height, output: output)
  }

  private static func render(
    _ source: CIImage,
    width: Int,
    height: Int,
    pool: CVPixelBufferPool
  ) throws -> CVPixelBuffer {
    var output: CVPixelBuffer?
    let result = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pool,
      &output
    )
    guard result == kCVReturnSuccess, let output else {
      throw H3NativeError.media("render pixel allocation returned \(result)")
    }
    return try render(source, width: width, height: height, output: output)
  }

  private static func render(
    _ source: CIImage,
    width: Int,
    height: Int,
    output: CVPixelBuffer
  ) throws -> CVPixelBuffer {
    var normalized = source
    let extent = normalized.extent
    guard extent.width > 0, extent.height > 0 else {
      throw H3NativeError.media("input image extent is empty")
    }
    normalized = normalized.transformed(
      by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
    )
    // Reference media is conditioning material, not the output canvas. Use a
    // single scale factor so portrait-to-landscape and landscape-to-portrait
    // generation never stretches the subject in either direction.
    let uniformScale = min(
      CGFloat(width) / extent.width,
      CGFloat(height) / extent.height
    )
    var foreground = normalized.transformed(
      by: CGAffineTransform(scaleX: uniformScale, y: uniformScale)
    )
    let fittedExtent = foreground.extent
    foreground = foreground.transformed(
      by: CGAffineTransform(
        translationX: (CGFloat(width) - fittedExtent.width) / 2
          - fittedExtent.minX,
        y: (CGFloat(height) - fittedExtent.height) / 2
          - fittedExtent.minY
      )
    )
    let targetBounds = CGRect(x: 0, y: 0, width: width, height: height)
    // Black letterbox pillars become a spatial conditioning mask in Ref2VA:
    // portrait subjects are generated inside a darker vertical slab while the
    // expanded sides follow a different exposure. Preserve aspect ratio, then
    // extend the fitted image's boundary pixels to the canvas edge. The value
    // at the join is exactly continuous, unlike a black or aspect-fill
    // backdrop, so the reference cannot stamp a rectangular exposure mask into
    // the generated latent.
    let background = foreground.clampedToExtent().cropped(to: targetBounds)
    let image = foreground.composited(over: background).cropped(to: targetBounds)
    imageContext.render(
      image,
      to: output,
      bounds: targetBounds,
      colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
    )
    return output
  }

  private static func appendNCTHW(
    _ pixelBuffer: CVPixelBuffer,
    frame: Int,
    frameCount: Int,
    destination: UnsafeMutableBufferPointer<Float16>
  ) throws {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw H3NativeError.media("pixel base address is unavailable")
    }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let plane = width * height
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
      let row = bytes.advanced(by: y * rowBytes)
      for x in 0..<width {
        let pixel = row.advanced(by: x * 4)
        let spatial = y * width + x
        destination[(0 * frameCount + frame) * plane + spatial] =
          Float16(Float(pixel[2]) / 255)
        destination[(1 * frameCount + frame) * plane + spatial] =
          Float16(Float(pixel[1]) / 255)
        destination[(2 * frameCount + frame) * plane + spatial] =
          Float16(Float(pixel[0]) / 255)
      }
    }
  }

  private static func writeBGRA(
    _ pixels: [Float],
    shape: [Int],
    frame: Int,
    to pixelBuffer: CVPixelBuffer
  ) throws {
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw H3NativeError.media("writer pixel base address is unavailable")
    }
    let frameCount = shape[2]
    let sourceHeight = shape[3]
    let sourceWidth = shape[4]
    let outputHeight = CVPixelBufferGetHeight(pixelBuffer)
    let outputWidth = CVPixelBufferGetWidth(pixelBuffer)
    guard outputWidth <= sourceWidth, outputHeight <= sourceHeight else {
      throw H3NativeError.media("writer crop exceeds decoded video bounds")
    }
    let cropX = (sourceWidth - outputWidth) / 2
    let cropY = (sourceHeight - outputHeight) / 2
    let plane = sourceHeight * sourceWidth
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<outputHeight {
      let row = bytes.advanced(by: y * rowBytes)
      for x in 0..<outputWidth {
        let spatial = (y + cropY) * sourceWidth + (x + cropX)
        func channel(_ c: Int) -> UInt8 {
          let value = pixels[(c * frameCount + frame) * plane + spatial]
          return UInt8(clamping: Int((min(1, max(0, value)) * 255).rounded()))
        }
        let pixel = row.advanced(by: x * 4)
        pixel[0] = channel(2)
        pixel[1] = channel(1)
        pixel[2] = channel(0)
        pixel[3] = 255
      }
    }
  }

  private static func interleavedAudio(_ tensor: H3Tensor) throws -> [Float] {
    let values = try tensor.floatValues()
    guard tensor.shape.count == 3, tensor.shape[0] == 1 else {
      throw H3NativeError.invalidTensor("decoded audio must have rank three")
    }
    if tensor.shape[1] == 2 {
      let samples = tensor.shape[2]
      var result = [Float](repeating: 0, count: samples * 2)
      for index in 0..<samples {
        result[index * 2] = values[index]
        result[index * 2 + 1] = values[samples + index]
      }
      return result
    }
    if tensor.shape[2] == 2 { return values }
    throw H3NativeError.invalidTensor("decoded audio needs two channels")
  }

  @available(macOS 27.0, *)
  private static func appendAudio(
    _ interleaved: [Float],
    sampleRate: Int,
    receiver: AVAssetWriterInput.SampleBufferReceiver
  ) async throws {
    var description: CMAudioFormatDescription?
    var format = AudioStreamBasicDescription(
      mSampleRate: Double(sampleRate),
      mFormatID: kAudioFormatLinearPCM,
      mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
      mBytesPerPacket: 8,
      mFramesPerPacket: 1,
      mBytesPerFrame: 8,
      mChannelsPerFrame: 2,
      mBitsPerChannel: 32,
      mReserved: 0
    )
    let formatStatus = CMAudioFormatDescriptionCreate(
      allocator: kCFAllocatorDefault,
      asbd: &format,
      layoutSize: 0,
      layout: nil,
      magicCookieSize: 0,
      magicCookie: nil,
      extensions: nil,
      formatDescriptionOut: &description
    )
    guard formatStatus == noErr, let description else {
      throw H3NativeError.media("audio format creation returned \(formatStatus)")
    }
    let totalFrames = interleaved.count / 2
    let chunkFrames = 1024
    var start = 0
    while start < totalFrames {
      let frames = min(chunkFrames, totalFrames - start)
      let byteCount = frames * 2 * MemoryLayout<Float>.stride
      var block: CMBlockBuffer?
      let blockStatus = CMBlockBufferCreateWithMemoryBlock(
        allocator: kCFAllocatorDefault,
        memoryBlock: nil,
        blockLength: byteCount,
        blockAllocator: kCFAllocatorDefault,
        customBlockSource: nil,
        offsetToData: 0,
        dataLength: byteCount,
        flags: 0,
        blockBufferOut: &block
      )
      guard blockStatus == kCMBlockBufferNoErr, let block else {
        throw H3NativeError.media("audio block creation returned \(blockStatus)")
      }
      let copyStatus = interleaved.withUnsafeBytes { raw in
        CMBlockBufferReplaceDataBytes(
          with: raw.baseAddress!.advanced(by: start * 2 * MemoryLayout<Float>.stride),
          blockBuffer: block,
          offsetIntoDestination: 0,
          dataLength: byteCount
        )
      }
      guard copyStatus == noErr else {
        throw H3NativeError.media("audio block fill returned \(copyStatus)")
      }
      var timing = CMSampleTimingInfo(
        duration: CMTime(value: 1, timescale: CMTimeScale(sampleRate)),
        presentationTimeStamp: CMTime(
          value: CMTimeValue(start),
          timescale: CMTimeScale(sampleRate)
        ),
        decodeTimeStamp: .invalid
      )
      var sample: CMSampleBuffer?
      let sampleStatus = CMSampleBufferCreateReady(
        allocator: kCFAllocatorDefault,
        dataBuffer: block,
        formatDescription: description,
        sampleCount: frames,
        sampleTimingEntryCount: 1,
        sampleTimingArray: &timing,
        sampleSizeEntryCount: 0,
        sampleSizeArray: nil,
        sampleBufferOut: &sample
      )
      guard sampleStatus == noErr, let sample else {
        throw H3NativeError.media(
          "audio sample creation returned \(sampleStatus)"
        )
      }
      try await receiver.append(
        CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(
          unsafeBuffer: sample
        )
      )
      start += frames
    }
  }
}
