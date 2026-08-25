import AppKit
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

enum MiniMaxH3ImageReferenceScope: String, CaseIterable, Identifiable {
  case wholeImage
  case faceOnly

  var id: String { rawValue }

  var label: String {
    switch self {
    case .wholeImage: "画像全体"
    case .faceOnly: "顔のみ"
    }
  }
}

struct MiniMaxH3FaceReference: Identifiable {
  let id: String
  let sourceURL: URL
  let cropURL: URL
  let sourceIndex: Int
  let faceIndex: Int
  let confidence: Float
  var isSelected: Bool
  var subjectIndex: Int

  var sourceLabel: String {
    "\(sourceURL.lastPathComponent)・顔\(faceIndex + 1)"
  }
}

private struct MiniMaxH3DetectedFace: Sendable {
  let id: String
  let sourceURL: URL
  let cropURL: URL
  let sourceIndex: Int
  let faceIndex: Int
  let confidence: Float
}

enum MiniMaxH3FaceReferenceProcessor {
  static let maximumReferences = 8

  static func faceOnlyPrompt(
    _ originalPrompt: String,
    references: [MiniMaxH3FaceReference]
  ) -> String {
    let selected = references.filter(\.isSelected).sorted {
      if $0.sourceIndex != $1.sourceIndex {
        return $0.sourceIndex < $1.sourceIndex
      }
      return $0.faceIndex < $1.faceIndex
    }
    guard !selected.isEmpty else { return originalPrompt }
    var pictureLabelsBySubject: [Int: [String]] = [:]
    for (index, face) in selected.enumerated() {
      pictureLabelsBySubject[face.subjectIndex, default: []]
        .append("<Picture \(index + 1)>")
    }
    let definitions = pictureLabelsBySubject.keys.sorted().map { subject in
      let pictures = pictureLabelsBySubject[subject, default: []]
        .joined(separator: ", ")
      return "<Subject \(subject)> is the person whose facial identity comes from \(pictures)."
    }.joined(separator: "\n")
    let retention = pictureLabelsBySubject.keys.sorted().map { subject in
      "<Subject \(subject)>: partially_preserved - preserve facial identity only."
    }.joined(separator: "\n")
    return """
      subject_definitions:
      \(definitions)

      summary:
      [reference generation] Use only each subject's facial structure, eyes, nose, mouth, hairline, and recognizable identity from the reference pictures. Regenerate clothing, body pose, background, framing, lighting, and camera angle from the user request.

      retention_analysis:
      \(retention)

      user_request:
      \(originalPrompt)
      """
  }

  static func detectFaces(
    in sourceURLs: [URL],
    destinationDirectory: URL
  ) async throws -> [MiniMaxH3FaceReference] {
    let detected = try await Task.detached(priority: .userInitiated) {
      try detectFacesSynchronously(
        in: sourceURLs,
        destinationDirectory: destinationDirectory
      )
    }.value
    return detected.enumerated().map { index, face in
      MiniMaxH3FaceReference(
        id: face.id,
        sourceURL: face.sourceURL,
        cropURL: face.cropURL,
        sourceIndex: face.sourceIndex,
        faceIndex: face.faceIndex,
        confidence: face.confidence,
        isSelected: index < maximumReferences,
        subjectIndex: min(index + 1, maximumReferences)
      )
    }
  }

  private static func detectFacesSynchronously(
    in sourceURLs: [URL],
    destinationDirectory: URL
  ) throws -> [MiniMaxH3DetectedFace] {
    let fileManager = FileManager.default
    try fileManager.createDirectory(
      at: destinationDirectory,
      withIntermediateDirectories: true
    )
    let context = CIContext(options: [
      .cacheIntermediates: false,
    ])
    var detected: [MiniMaxH3DetectedFace] = []
    for (sourceIndex, sourceURL) in sourceURLs.enumerated() {
      try Task.checkCancellation()
      let image = try loadOrientedImage(sourceURL, context: context)
      let request = VNDetectFaceRectanglesRequest()
      let handler = VNImageRequestHandler(
        cgImage: image,
        orientation: .up,
        options: [:]
      )
      try handler.perform([request])
      let observations = (request.results ?? [])
        .filter { $0.confidence >= 0.45 }
        .sorted { left, right in
          let verticalDistance = abs(
            left.boundingBox.midY - right.boundingBox.midY
          )
          if verticalDistance > 0.12 {
            return left.boundingBox.midY > right.boundingBox.midY
          }
          return left.boundingBox.midX < right.boundingBox.midX
        }
      for (faceIndex, observation) in observations.enumerated() {
        try Task.checkCancellation()
        let cropRect = expandedFaceCrop(
          observation.boundingBox,
          imageWidth: image.width,
          imageHeight: image.height
        )
        guard let crop = image.cropping(to: cropRect) else { continue }
        let filename = String(
          format: "source-%02d-face-%02d.png", sourceIndex + 1, faceIndex + 1
        )
        let cropURL = destinationDirectory.appendingPathComponent(filename)
        try writePNG(crop, to: cropURL)
        detected.append(
          MiniMaxH3DetectedFace(
            id: "\(sourceIndex)-\(faceIndex)-\(cropRect.debugDescription)",
            sourceURL: sourceURL,
            cropURL: cropURL,
            sourceIndex: sourceIndex,
            faceIndex: faceIndex,
            confidence: observation.confidence
          )
        )
      }
    }
    return detected
  }

  private static func loadOrientedImage(
    _ url: URL,
    context: CIContext
  ) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let rawImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      throw CocoaError(.fileReadCorruptFile)
    }
    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
      as? [CFString: Any]
    let orientationNumber = properties?[kCGImagePropertyOrientation] as? NSNumber
    let orientation = Int32(orientationNumber?.intValue ?? 1)
    let oriented = CIImage(cgImage: rawImage)
      .oriented(forExifOrientation: orientation)
    let extent = oriented.extent.integral
    let normalized = oriented.transformed(
      by: CGAffineTransform(
        translationX: -extent.minX,
        y: -extent.minY
      )
    )
    guard let image = context.createCGImage(
      normalized,
      from: CGRect(origin: .zero, size: extent.size)
    ) else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return image
  }

  private static func expandedFaceCrop(
    _ normalizedFace: CGRect,
    imageWidth: Int,
    imageHeight: Int
  ) -> CGRect {
    let width = CGFloat(imageWidth)
    let height = CGFloat(imageHeight)
    let face = CGRect(
      x: normalizedFace.minX * width,
      y: (1 - normalizedFace.maxY) * height,
      width: normalizedFace.width * width,
      height: normalizedFace.height * height
    )
    // Vision's rectangle covers the central face. Include hair, chin and a
    // small shoulder cue without retaining the original full composition.
    let side = max(face.width * 1.75, face.height * 1.95)
    let center = CGPoint(
      x: face.midX,
      y: face.midY - face.height * 0.08
    )
    let proposed = CGRect(
      x: center.x - side / 2,
      y: center.y - side / 2,
      width: side,
      height: side
    ).integral
    let imageBounds = CGRect(x: 0, y: 0, width: width, height: height)
    return proposed.intersection(imageBounds).integral
  }

  private static func writePNG(_ image: CGImage, to url: URL) throws {
    guard let destination = CGImageDestinationCreateWithURL(
      url as CFURL,
      UTType.png.identifier as CFString,
      1,
      nil
    ) else {
      throw CocoaError(.fileWriteUnknown)
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw CocoaError(.fileWriteUnknown)
    }
  }
}
