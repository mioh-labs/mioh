import AVFoundation
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import Vision

struct MiniMaxH3VideoMaskCandidate: Identifiable {
  let id: String
  let index: Int
  let previewURL: URL
  let boundingBox: CGRect
  let confidence: Float

  var label: String { "人物\(index + 1)" }
}

enum MiniMaxH3ReferenceVideoMaskError: LocalizedError {
  case noVideoTrack
  case writerUnavailable
  case readerFailed(String)
  case writerFailed(String)

  var errorDescription: String? {
    switch self {
    case .noVideoTrack:
      return "マスク対象の動画トラックが見つかりません"
    case .writerUnavailable:
      return "マスク動画の書き出し準備に失敗しました"
    case .readerFailed(let message):
      return "マスク対象動画の読み込みに失敗しました: \(message)"
    case .writerFailed(let message):
      return "マスク動画の書き出しに失敗しました: \(message)"
    }
  }
}

enum MiniMaxH3ReferenceVideoMaskProcessor {
  private static let context = CIContext(options: [
    .workingColorSpace: NSNull(),
    .outputColorSpace: NSNull(),
    .cacheIntermediates: false,
  ])

  static func createMaskedReferenceVideo(
    sourceURL: URL,
    targetDescription: String,
    targetIndex: Int?,
    durationSeconds: Double,
    outputURL: URL
  ) async throws -> URL {
    let asset = AVURLAsset(url: sourceURL)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw MiniMaxH3ReferenceVideoMaskError.noVideoTrack
    }
    let naturalSize = try await track.load(.naturalSize)
    let transform = try await track.load(.preferredTransform)
    let transformed = CGRect(origin: .zero, size: naturalSize).applying(transform)
    let width = max(2, Int(abs(transformed.width).rounded()))
    let height = max(2, Int(abs(transformed.height).rounded()))
    let frameRate = 24
    let frameDuration = CMTime(value: 1, timescale: CMTimeScale(frameRate))
    let maximumFrames = max(1, Int((durationSeconds * Double(frameRate)).rounded()))

    let fileManager = FileManager.default
    try fileManager.createDirectory(
      at: outputURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    if fileManager.fileExists(atPath: outputURL.path) {
      try fileManager.removeItem(at: outputURL)
    }

    let reader = try AVAssetReader(asset: asset)
    let readerOutput = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_32BGRA)
      ]
    )
    let provider = reader.outputProvider(for: readerOutput)

    let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
    let writerInput = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.hevc,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: max(4_000_000, width * height * 6),
          AVVideoExpectedSourceFrameRateKey: frameRate,
          AVVideoMaxKeyFrameIntervalKey: frameRate * 2,
        ],
      ]
    )
    writerInput.expectsMediaDataInRealTime = false
    guard writer.canAdd(writerInput) else {
      throw MiniMaxH3ReferenceVideoMaskError.writerUnavailable
    }
    writer.add(writerInput)
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: writerInput,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
    )

    try reader.start()
    try writer.start()
    writer.startSession(atSourceTime: .zero)
    guard let pool = adaptor.pixelBufferPool else {
      throw MiniMaxH3ReferenceVideoMaskError.writerUnavailable
    }

    let selector = TargetSelector(
      description: targetDescription,
      targetIndex: targetIndex
    )
    var firstTimestamp: Double?
    var written = 0
    while written < maximumFrames, let sample = try await provider.next() {
      let timestamp = sample.presentationTimeStamp.seconds
      if firstTimestamp == nil { firstTimestamp = timestamp }
      let relative = timestamp - (firstTimestamp ?? timestamp)
      let wanted = Double(written) / Double(frameRate)
      guard relative + 1e-7 >= wanted,
        let pixelSample = CMReadySampleBuffer<CVReadOnlyPixelBuffer>(sample)
      else { continue }
      let source = pixelSample.content.withUnsafeBuffer { $0 }
      let rendered = try renderOriented(
        source,
        transform: transform,
        width: width,
        height: height,
        pool: pool
      )
      let masked = try maskedFrame(
        rendered,
        width: width,
        height: height,
        pool: pool,
        selector: selector
      )
      while !writerInput.isReadyForMoreMediaData {
        try await Task.sleep(for: .milliseconds(2))
      }
      let presentationTime = CMTimeMultiply(
        frameDuration,
        multiplier: Int32(written)
      )
      guard adaptor.append(masked, withPresentationTime: presentationTime) else {
        throw MiniMaxH3ReferenceVideoMaskError.writerFailed(
          writer.error?.localizedDescription ?? "append failed"
        )
      }
      written += 1
    }
    writerInput.markAsFinished()
    await writer.finishWriting()
    guard reader.status != .failed else {
      throw MiniMaxH3ReferenceVideoMaskError.readerFailed(
        reader.error?.localizedDescription ?? "unknown error"
      )
    }
    guard writer.status == .completed else {
      throw MiniMaxH3ReferenceVideoMaskError.writerFailed(
        writer.error?.localizedDescription ?? "unknown error"
      )
    }
    return outputURL
  }

  static func detectMaskCandidates(
    sourceURL: URL,
    destinationDirectory: URL,
    maximumCandidates: Int = 8
  ) async throws -> [MiniMaxH3VideoMaskCandidate] {
    let asset = AVURLAsset(url: sourceURL)
    let duration = try await asset.load(.duration).seconds
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = CMTime(seconds: 0.2, preferredTimescale: 600)
    generator.requestedTimeToleranceAfter = CMTime(seconds: 0.2, preferredTimescale: 600)

    struct DetectionFrame {
      let image: CGImage
      let people: [(box: CGRect, confidence: Float)]
      let faces: [(box: CGRect, confidence: Float)]
      let faceMatches: Int
    }

    func image(at seconds: Double) async throws -> CGImage {
      let requested = CMTime(seconds: seconds, preferredTimescale: 600)
      if #available(macOS 13.0, *) {
        return try await generator.image(at: requested).image
      } else {
        var actual = CMTime.zero
        return try generator.copyCGImage(at: requested, actualTime: &actual)
      }
    }

    func detectionFrame(at seconds: Double) async throws -> DetectionFrame {
      let cgImage = try await image(at: seconds)
      let personRequest = VNDetectHumanRectanglesRequest()
      personRequest.upperBodyOnly = false
      let faceRequest = VNDetectFaceRectanglesRequest()
      let handler = VNImageRequestHandler(cgImage: cgImage)
      try handler.perform([personRequest, faceRequest])
      let people = (personRequest.results ?? []).map {
        (box: $0.boundingBox, confidence: $0.confidence)
      }
      let visionFaces = (faceRequest.results ?? []).map {
        (box: $0.boundingBox, confidence: $0.confidence)
      }
      let ciFaces = coreImageFaceBoxes(in: cgImage).map {
        (box: $0, confidence: Float(0.5))
      }
      let faces = mergeFaceBoxes(visionFaces + ciFaces)
      let faceMatches = people.reduce(0) { count, person in
        count + faces.filter { face in
          let intersection = person.box.intersection(face.box)
          guard !intersection.isNull else { return false }
          let faceArea = max(0.0001, face.box.width * face.box.height)
          return intersection.width * intersection.height / faceArea > 0.25
        }.count
      }
      return DetectionFrame(
        image: cgImage,
        people: people,
        faces: faces,
        faceMatches: faceMatches
      )
    }

    let finiteDuration = duration.isFinite && duration > 0 ? duration : 0
    let sampleSeconds = [
      0,
      min(0.5, finiteDuration),
      min(1.0, finiteDuration),
      min(2.0, finiteDuration),
      min(3.0, finiteDuration),
      min(5.0, finiteDuration),
      finiteDuration * 0.35,
      finiteDuration * 0.65,
      max(0, finiteDuration - 0.5),
    ]
    var uniqueSampleSeconds: [Double] = []
    for seconds in sampleSeconds where seconds.isFinite && seconds >= 0 {
      let rounded = (seconds * 10).rounded() / 10
      if !uniqueSampleSeconds.contains(where: { abs($0 - rounded) < 0.05 }) {
        uniqueSampleSeconds.append(rounded)
      }
    }

    var bestFrame: DetectionFrame?
    for seconds in uniqueSampleSeconds {
      let frame = try await detectionFrame(at: seconds)
      if frame.people.isEmpty && frame.faces.isEmpty { continue }
      if let current = bestFrame {
        if frame.faceMatches != current.faceMatches {
          if frame.faceMatches > current.faceMatches { bestFrame = frame }
        } else if frame.faces.count != current.faces.count {
          if frame.faces.count > current.faces.count { bestFrame = frame }
        } else if frame.people.count > current.people.count {
          bestFrame = frame
        }
      } else {
        bestFrame = frame
      }
    }

    guard let bestFrame else { return [] }
    let cgImage = bestFrame.image
    let observations = (!bestFrame.faces.isEmpty
      ? bestFrame.faces
      : bestFrame.people)
      .sorted {
        if abs($0.box.minX - $1.box.minX) > 0.02 {
          return $0.box.minX < $1.box.minX
        }
        return $0.box.width * $0.box.height
          > $1.box.width * $1.box.height
      }
      .prefix(maximumCandidates)
    try FileManager.default.createDirectory(
      at: destinationDirectory,
      withIntermediateDirectories: true
    )
    let sourceImage = CIImage(cgImage: cgImage)
    let extent = sourceImage.extent
    let faceBoxes = bestFrame.faces.map {
      imageRect(fromNormalizedVisionBox: $0.box, extent: extent)
    }
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
      ?? CGColorSpaceCreateDeviceRGB()
    return try observations.enumerated().map { index, observation in
      let personRect = imageRect(
        fromNormalizedVisionBox: observation.box,
        extent: extent
      )
      let rect = previewRect(
        forPersonRect: personRect,
        faceBoxes: faceBoxes,
        extent: extent
      ).insetBy(dx: -extent.width * 0.01, dy: -extent.height * 0.01)
        .intersection(extent)
      let preview = sourceImage.cropped(to: rect)
      let output = destinationDirectory.appendingPathComponent(
        String(format: "person-%02d.png", index + 1)
      )
      try context.writePNGRepresentation(
        of: preview,
        to: output,
        format: .RGBA8,
        colorSpace: colorSpace
      )
      return MiniMaxH3VideoMaskCandidate(
        id: "\(sourceURL.path)-person-\(index)",
        index: index,
        previewURL: output,
        boundingBox: observation.box,
        confidence: observation.confidence
      )
    }
  }

  private static func renderOriented(
    _ source: CVPixelBuffer,
    transform: CGAffineTransform,
    width: Int,
    height: Int,
    pool: CVPixelBufferPool
  ) throws -> CVPixelBuffer {
    var output: CVPixelBuffer?
    let result = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
    guard result == kCVReturnSuccess, let output else {
      throw MiniMaxH3ReferenceVideoMaskError.writerUnavailable
    }
    let image = CIImage(cvPixelBuffer: source)
    let transformed = image.transformed(by: transform)
    let extent = transformed.extent
    let normalized = transformed
      .transformed(by: CGAffineTransform(
        translationX: -extent.minX,
        y: -extent.minY
      ))
    context.render(
      normalized,
      to: output,
      bounds: CGRect(x: 0, y: 0, width: width, height: height),
      colorSpace: nil
    )
    return output
  }

  private static func maskedFrame(
    _ input: CVPixelBuffer,
    width: Int,
    height: Int,
    pool: CVPixelBufferPool,
    selector: TargetSelector
  ) throws -> CVPixelBuffer {
    let extent = CGRect(x: 0, y: 0, width: width, height: height)
    let image = CIImage(cvPixelBuffer: input)
    let observations = try detectHumans(input)
    guard let selected = selector.select(from: observations, extent: extent) else {
      return input
    }
    let personMask = try segmentationMask(input, extent: extent)
    let selectedMask = personMask
      .applyingFilter(
        "CICrop",
        parameters: ["inputRectangle": CIVector(cgRect: selected.insetBy(dx: -selected.width * 0.08, dy: -selected.height * 0.08))]
      )
      .cropped(to: extent)
    let tint = CIImage(color: CIColor(red: 1, green: 0, blue: 0.85, alpha: 0.42))
      .cropped(to: extent)
    let tinted = tint.composited(over: image)
    let outputImage = tinted.applyingFilter(
      "CIBlendWithMask",
      parameters: [
        kCIInputBackgroundImageKey: image,
        kCIInputMaskImageKey: selectedMask
          .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 3])
          .cropped(to: extent),
      ]
    )
    var output: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
    guard let output else {
      throw MiniMaxH3ReferenceVideoMaskError.writerUnavailable
    }
    context.render(outputImage, to: output, bounds: extent, colorSpace: nil)
    return output
  }

  private static func detectHumans(_ pixelBuffer: CVPixelBuffer) throws
    -> [CGRect]
  {
    let request = VNDetectHumanRectanglesRequest()
    request.upperBodyOnly = false
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
    try handler.perform([request])
    return (request.results ?? []).map(\.boundingBox)
  }

  private static func previewRect(
    forPersonRect personRect: CGRect,
    faceBoxes: [CGRect],
    extent: CGRect
  ) -> CGRect {
    let matchingFace = faceBoxes
      .filter { face in
        let intersection = face.intersection(personRect)
        guard !intersection.isNull else { return false }
        let faceArea = max(1, face.width * face.height)
        return intersection.width * intersection.height / faceArea > 0.25
      }
      .max { lhs, rhs in
        lhs.width * lhs.height < rhs.width * rhs.height
      }
    guard let face = matchingFace else {
      return personRect.insetBy(
        dx: -extent.width * 0.03,
        dy: -extent.height * 0.03
      )
    }
    let side = max(face.width, face.height) * 2.25
    let center = CGPoint(x: face.midX, y: face.midY)
    let unclamped = CGRect(
      x: center.x - side / 2,
      y: center.y - side / 2,
      width: side,
      height: side
    )
    if extent.contains(unclamped) {
      return unclamped
    }
    let adjustedX = min(max(unclamped.minX, extent.minX), extent.maxX - side)
    let adjustedY = min(max(unclamped.minY, extent.minY), extent.maxY - side)
    let adjusted = CGRect(
      x: adjustedX,
      y: adjustedY,
      width: side,
      height: side
    )
    return adjusted.intersection(extent)
  }

  private static func coreImageFaceBoxes(in cgImage: CGImage) -> [CGRect] {
    let image = CIImage(cgImage: cgImage)
    let detector = CIDetector(
      ofType: CIDetectorTypeFace,
      context: context,
      options: [
        CIDetectorAccuracy: CIDetectorAccuracyLow,
        CIDetectorTracking: false,
      ]
    )
    let features = detector?.features(
      in: image,
      options: [
        CIDetectorImageOrientation: 1,
      ]
    ) ?? []
    let extent = image.extent
    return features.compactMap { feature in
      guard feature.bounds.width > 4, feature.bounds.height > 4 else {
        return nil
      }
      return CGRect(
        x: feature.bounds.minX / extent.width,
        y: feature.bounds.minY / extent.height,
        width: feature.bounds.width / extent.width,
        height: feature.bounds.height / extent.height
      )
    }
  }

  private static func mergeFaceBoxes(
    _ faces: [(box: CGRect, confidence: Float)]
  ) -> [(box: CGRect, confidence: Float)] {
    var merged: [(box: CGRect, confidence: Float)] = []
    for face in faces.sorted(by: {
      $0.box.width * $0.box.height > $1.box.width * $1.box.height
    }) {
      let duplicatesExisting = merged.contains { existing in
        let intersection = existing.box.intersection(face.box)
        guard !intersection.isNull else { return false }
        let smallerArea = max(
          0.0001,
          min(
            existing.box.width * existing.box.height,
            face.box.width * face.box.height
          )
        )
        return intersection.width * intersection.height / smallerArea > 0.35
      }
      if !duplicatesExisting {
        merged.append(face)
      }
    }
    return merged
  }

  private static func segmentationMask(
    _ pixelBuffer: CVPixelBuffer,
    extent: CGRect
  ) throws -> CIImage {
    let request = VNGeneratePersonSegmentationRequest()
    request.qualityLevel = .balanced
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
    try handler.perform([request])
    guard let result = request.results?.first?.pixelBuffer else {
      return CIImage(color: .black).cropped(to: extent)
    }
    return CIImage(cvPixelBuffer: result)
      .transformed(by: CGAffineTransform(
        scaleX: extent.width / CGFloat(CVPixelBufferGetWidth(result)),
        y: extent.height / CGFloat(CVPixelBufferGetHeight(result))
      ))
      .cropped(to: extent)
  }

  private struct TargetSelector {
    let description: String
    let targetIndex: Int?

    func select(from normalizedBoxes: [CGRect], extent: CGRect) -> CGRect? {
      guard !normalizedBoxes.isEmpty else { return nil }
      let boxes = normalizedBoxes.map {
        imageRect(fromNormalizedVisionBox: $0, extent: extent)
      }
      let ordered = boxes.sorted { $0.midX < $1.midX }
      if let targetIndex, ordered.indices.contains(targetIndex) {
        return ordered[targetIndex]
      }
      let text = description.lowercased()
      if text.contains("左") || text.contains("left") {
        return boxes.min { $0.midX < $1.midX }
      }
      if text.contains("右") || text.contains("right") {
        return boxes.max { $0.midX < $1.midX }
      }
      if text.contains("中央") || text.contains("真ん中")
        || text.contains("center") || text.contains("centre")
      {
        let centerX = extent.midX
        return boxes.min { abs($0.midX - centerX) < abs($1.midX - centerX) }
      }
      if text.contains("奥") || text.contains("back") {
        return boxes.min { $0.width * $0.height < $1.width * $1.height }
      }
      if text.contains("手前") || text.contains("front") {
        return boxes.max { $0.width * $0.height < $1.width * $1.height }
      }
      return boxes.max { $0.width * $0.height < $1.width * $1.height }
    }
  }

  private static func imageRect(
    fromNormalizedVisionBox box: CGRect,
    extent: CGRect
  ) -> CGRect {
    CGRect(
      x: extent.minX + box.minX * extent.width,
      y: extent.minY + (1 - box.maxY) * extent.height,
      width: box.width * extent.width,
      height: box.height * extent.height
    )
  }

}
