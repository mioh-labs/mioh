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

private enum MiniMaxH3FaceReferenceError: LocalizedError {
  case personMaskUnavailable(String)

  var errorDescription: String? {
    switch self {
    case .personMaskUnavailable(let filename):
      return "顔のみ参照の人物マスクを作成できませんでした: \(filename)"
    }
  }
}

enum MiniMaxH3FaceReferenceProcessor {
  static let maximumReferences = 8

  /// Runs the same Vision crop used by the app's 「顔のみ」 control for an
  /// external caller such as the MCP server. Automated multi-image requests
  /// describe one subject by default, so every selected crop shares Subject 1.
  static func prepareAutomationReferences(
    in sourceURLs: [URL],
    destinationDirectory: URL
  ) throws -> [MiniMaxH3FaceReference] {
    let detected = try detectFacesSynchronously(
      in: sourceURLs,
      destinationDirectory: destinationDirectory
    )
    return detected.prefix(maximumReferences).map { face in
      MiniMaxH3FaceReference(
        id: face.id,
        sourceURL: face.sourceURL,
        cropURL: face.cropURL,
        sourceIndex: face.sourceIndex,
        faceIndex: face.faceIndex,
        confidence: face.confidence,
        isSelected: true,
        subjectIndex: 1
      )
    }
  }

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
    let subjectNumbers = pictureLabelsBySubject.keys.sorted()
    let definitions = subjectNumbers.map { subject in
      let pictures = pictureLabelsBySubject[subject, default: []]
        .joined(separator: ", ")
      return "<Subject \(subject)> is the person whose facial identity comes from \(pictures)."
    }.joined(separator: "\n")
    let retention = subjectNumbers.map { subject in
      "<Subject \(subject)>: partially_preserved - preserve facial identity only."
    }.joined(separator: "\n")
    let subjects = subjectNumbers.map { "<Subject \($0)>" }
      .joined(separator: ", ")
    if isStructuredH3Prompt(originalPrompt) {
      return augmentStructuredH3Prompt(
        originalPrompt,
        pictureLabelsBySubject: pictureLabelsBySubject,
        subjectNumbers: subjectNumbers
      )
    }
    return """
      subject_definitions:
      \(definitions)

      summary:
      [reference generation] Generate the requested video with \(subjects) as visible facial-identity references. Use only each subject's facial structure, eyes, nose, mouth, hairline, and recognizable identity from the reference pictures. Regenerate clothing, body pose, background, framing, lighting, and camera angle from the requested scene.

      retention_analysis:
      \(retention)

      detailed_description:
      [Shot 1] Follow this user direction: \(originalPrompt)
      Use \(subjects) only as visible facial-identity references. Reference labels such as <Subject 1> and <Picture 1> are silent control metadata. Never speak, narrate, sing, subtitle, or render a reference label as visible text.

      overall_soundscape:
      Generate only sounds explicitly requested in detailed_description. Do not add narration, voice-over, dialogue, singing, or spoken reference labels unless the user explicitly requests speech.

      non_diegetic_music:
      N/A unless explicitly requested in detailed_description.
      """
  }

  private static let structuredSectionHeaders = [
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
  ]

  private static func isStructuredH3Prompt(_ prompt: String) -> Bool {
    let lines = prompt.components(separatedBy: .newlines)
    var searchStart = 0
    for header in structuredSectionHeaders {
      guard let index = lines[searchStart...].firstIndex(where: {
        $0.trimmingCharacters(in: .whitespacesAndNewlines)
          .lowercased() == header
      }) else { return false }
      searchStart = index + 1
    }
    return true
  }

  private static func augmentStructuredH3Prompt(
    _ prompt: String,
    pictureLabelsBySubject: [Int: [String]],
    subjectNumbers: [Int]
  ) -> String {
    var lines = prompt.components(separatedBy: .newlines)
    guard let definitionsIndex = lines.firstIndex(where: {
      $0.trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased() == "subject_definitions:"
    }),
      let summaryIndex = lines[(definitionsIndex + 1)...].firstIndex(where: {
        $0.trimmingCharacters(in: .whitespacesAndNewlines)
          .lowercased() == "summary:"
      })
    else { return prompt }

    var missingBindings: [String] = []
    for subject in subjectNumbers {
      let pictures = pictureLabelsBySubject[subject, default: []]
        .joined(separator: ", ")
      let prefix = "<Subject \(subject)>"
      if let lineIndex = lines[(definitionsIndex + 1)..<summaryIndex]
        .firstIndex(where: {
          $0.trimmingCharacters(in: .whitespacesAndNewlines)
            .hasPrefix(prefix)
        })
      {
        let binding = " Its facial identity comes from \(pictures); use those pictures only for facial structure, eyes, nose, mouth, hairline, and recognizable identity."
        if !lines[lineIndex].contains("facial identity comes from") {
          lines[lineIndex] += binding
        }
      } else {
        missingBindings.append(
          "\(prefix) is the person whose facial identity comes from \(pictures); use those pictures only for facial structure, eyes, nose, mouth, hairline, and recognizable identity."
        )
      }
    }
    if !missingBindings.isEmpty {
      lines.insert(contentsOf: missingBindings, at: definitionsIndex + 1)
    }

    let metadataNotice =
      "Reference labels such as <Subject 1> and <Picture 1> are silent control metadata. Never speak, narrate, sing, subtitle, or render a reference label as visible text."
    if !lines.contains(where: { $0.contains("silent control metadata") }),
      let detailsIndex = lines.firstIndex(where: {
        $0.trimmingCharacters(in: .whitespacesAndNewlines)
          .lowercased() == "detailed_description:"
      })
    {
      lines.insert(metadataNotice, at: detailsIndex + 1)
    }
    return lines.joined(separator: "\n")
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
      guard observations.isEmpty || !Task.isCancelled else {
        throw CancellationError()
      }
      let identityImage = try backgroundSoftenedIdentityImage(
        image,
        sourceName: sourceURL.lastPathComponent,
        context: context
      )
      for (faceIndex, observation) in observations.enumerated() {
        try Task.checkCancellation()
        let cropRect = expandedFaceCrop(
          observation.boundingBox,
          imageWidth: image.width,
          imageHeight: image.height
        )
        // A source that is already a close-up must not silently turn
        // `face_only` into a nearly full-frame scene reference. Keep the face
        // and feathered person boundary, then pad any out-of-bounds part with
        // neutral pixels instead of clipping the square back to the source.
        guard let crop = paddedSquareCrop(
          identityImage,
          rect: cropRect,
          context: context
        ) else { continue }
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

  private static func backgroundSoftenedIdentityImage(
    _ image: CGImage,
    sourceName: String,
    context: CIContext
  ) throws -> CGImage {
    let request = VNGeneratePersonSegmentationRequest()
    request.qualityLevel = .accurate
    let handler = VNImageRequestHandler(
      cgImage: image,
      orientation: .up,
      options: [:]
    )
    try handler.perform([request])
    guard let pixelBuffer = request.results?.first?.pixelBuffer else {
      throw MiniMaxH3FaceReferenceError.personMaskUnavailable(sourceName)
    }
    let foreground = CIImage(cgImage: image)
    let sourceMask = CIImage(cvPixelBuffer: pixelBuffer)
    guard sourceMask.extent.width > 0, sourceMask.extent.height > 0 else {
      throw MiniMaxH3FaceReferenceError.personMaskUnavailable(sourceName)
    }
    let mask = sourceMask.transformed(
      by: CGAffineTransform(
        scaleX: foreground.extent.width / sourceMask.extent.width,
        y: foreground.extent.height / sourceMask.extent.height
      )
    ).cropped(to: foreground.extent)
    let minimumEdge = min(foreground.extent.width, foreground.extent.height)
    let featherRadius = max(12, minimumEdge * 0.018)
    let backgroundBlurRadius = max(28, minimumEdge * 0.045)
    let removalExpansion = max(8, minimumEdge * 0.012)
    let featheredMask = mask
      // Vision's person mask already extends slightly beyond fine hair. A
      // straight blur expands it farther into the source background, causing
      // signs, colors, and a dark silhouette ring to be blended back into the
      // supposedly neutral identity reference. Contract by twice the feather
      // radius first so the soft transition lands inside the subject edge.
      .applyingFilter(
        "CIMorphologyMinimum",
        parameters: [kCIInputRadiusKey: featherRadius * 2]
      )
      .applyingFilter(
        "CIGaussianBlur",
        parameters: [kCIInputRadiusKey: featherRadius]
      )
      .cropped(to: foreground.extent)
    let removalMask = mask
      .applyingFilter(
        "CIMorphologyMaximum",
        parameters: [kCIInputRadiusKey: removalExpansion]
      )
      .cropped(to: foreground.extent)
    let neutralBackground = CIImage(
      color: CIColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1)
    ).cropped(to: foreground.extent)
    // Remove the subject silhouette before blurring. Blurring the original
    // image spreads dark hair and clothing into the background and creates a
    // second ghost contour just outside the feathered foreground mask.
    let subjectRemovedBackground = neutralBackground.applyingFilter(
      "CIBlendWithMask",
      parameters: [
        kCIInputBackgroundImageKey: foreground,
        kCIInputMaskImageKey: removalMask,
      ]
    )
    let flattenedBackground = subjectRemovedBackground.applyingFilter(
      "CIDissolveTransition",
      parameters: [
        "inputTargetImage": neutralBackground,
        "inputTime": 0.7,
      ]
    )
    let softenedBackground = flattenedBackground
      .clampedToExtent()
      .applyingFilter(
        "CIGaussianBlur",
        parameters: [kCIInputRadiusKey: backgroundBlurRadius]
      )
      .applyingFilter(
        "CIColorControls",
        parameters: [
          kCIInputSaturationKey: 0.04,
          kCIInputContrastKey: 0.55,
        ]
      )
      .cropped(to: foreground.extent)
    let isolated = foreground.applyingFilter(
      "CIBlendWithMask",
      parameters: [
        kCIInputBackgroundImageKey: softenedBackground,
        kCIInputMaskImageKey: featheredMask,
      ]
    )
    guard let result = context.createCGImage(isolated, from: foreground.extent)
    else {
      throw MiniMaxH3FaceReferenceError.personMaskUnavailable(sourceName)
    }
    return result
  }

  private static func paddedSquareCrop(
    _ image: CGImage,
    rect: CGRect,
    context: CIContext
  ) -> CGImage? {
    guard rect.width > 0, rect.height > 0 else { return nil }
    let source = CIImage(cgImage: image)
    // `expandedFaceCrop` is expressed in CGImage's top-left coordinates;
    // Core Image uses a bottom-left origin.
    let crop = CGRect(
      x: rect.minX,
      y: CGFloat(image.height) - rect.maxY,
      width: rect.width,
      height: rect.height
    ).integral
    // `expandedFaceCrop` normally shifts the square inside the source. If a
    // very large face makes that impossible, extend the already-softened edge
    // pixels instead of inserting a flat band with a hard boundary.
    let padded = source
      .clampedToExtent()
      .cropped(to: crop)
      .transformed(
        by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY)
      )
    let outputBounds = CGRect(
      x: 0, y: 0, width: crop.width, height: crop.height
    )
    return context.createCGImage(padded, from: outputBounds)
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
    // Vision's rectangle covers the central face. Include hair and chin, but
    // cap the reference to at most half of the source area. Close-up inputs
    // otherwise expand to essentially the original image and retain its
    // background, clothing, and composition.
    let requestedSide = max(face.width * 1.75, face.height * 1.95)
    let maximumSceneSafeSide = sqrt(width * height * 0.5)
    let minimumFaceSafeSide = max(face.width * 1.08, face.height * 1.18)
    let side = max(
      minimumFaceSafeSide,
      min(requestedSide, maximumSceneSafeSide)
    )
    let center = CGPoint(
      x: face.midX,
      y: face.midY - face.height * 0.08
    )
    var proposed = CGRect(
      x: center.x - side / 2,
      y: center.y - side / 2,
      width: side,
      height: side
    ).integral
    // Preserve the requested square by translating it into the source bounds
    // whenever it fits. Intersecting it with the bounds changes the aspect
    // ratio; padding it unnecessarily creates a straight conditioning edge.
    if proposed.width <= width, proposed.height <= height {
      proposed.origin.x = min(max(0, proposed.origin.x), width - proposed.width)
      proposed.origin.y = min(max(0, proposed.origin.y), height - proposed.height)
    }
    return proposed
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
