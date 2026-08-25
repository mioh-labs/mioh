import AVFoundation
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
    var values = [Float16](
      repeating: 0,
      count: 3 * plane * frameCount
    )
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
      try appendNCTHW(
        activePixelBuffer!,
        frame: frame,
        frameCount: frameCount,
        destination: &values
      )
    }
    return try H3Tensor(
      float16: values,
      shape: [1, 3, frameCount, height, width]
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
    return try H3Tensor(
      float32: [Float](repeating: 0, count: 2 * sampleCount),
      shape: [1, 2, sampleCount]
    )
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
    var values = [Float16](repeating: 0, count: frameElements * frameCount)
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
      let rendered = try render(
        source,
        transform: transform,
        width: width,
        height: height,
        pool: pool
      )
      try appendNCTHW(
        rendered,
        frame: selected,
        frameCount: frameCount,
        destination: &values
      )
      selected += 1
    }
    guard selected == frameCount, reader.status != .failed else {
      throw H3NativeError.media(
        "decoded \(selected)/\(frameCount) reference frames: "
          + (reader.error?.localizedDescription ?? "input ended early")
      )
    }
    return try H3Tensor(
      float16: values,
      shape: [1, 3, frameCount, height, width]
    )
  }

  static func decodeReferenceAudio(
    url: URL,
    durationSeconds: Double,
    sampleRate: Int = 32_000
  ) async throws -> H3Tensor {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
      return try silentAudio(
        durationSeconds: durationSeconds,
        sampleRate: sampleRate
      )
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
    let maximumFrames = max(1, Int((durationSeconds * Double(sampleRate)).rounded()))
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
      guard let data = payload else { continue }
      data.withUnsafeBytes { raw in
        interleaved.append(contentsOf: raw.bindMemory(to: Float.self))
      }
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

  @available(macOS 27.0, *)
  static func writeMovie(
    video: H3Tensor,
    audio: H3Tensor?,
    outputURL: URL,
    frameRate: Int = 24,
    audioSampleRate: Int = 32_000
  ) async throws {
    guard video.shape.count == 5, video.shape[0] == 1, video.shape[1] == 3 else {
      throw H3NativeError.invalidTensor("decoded video must be NCTHW RGB")
    }
    let frameCount = video.shape[2]
    let height = video.shape[3]
    let width = video.shape[4]
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
      audioSamples = try interleavedAudio(audio)
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
            frame: frame,
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
    destination: inout [Float16]
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
    let height = shape[3]
    let width = shape[4]
    let plane = height * width
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
      let row = bytes.advanced(by: y * rowBytes)
      for x in 0..<width {
        let spatial = y * width + x
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
