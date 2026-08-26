import Foundation
import ImageIO

@main
struct MiniMaxH3FaceReferenceProbe {
  static func main() async throws {
    guard CommandLine.arguments.count >= 3 else {
      throw NSError(
        domain: "MiniMaxH3FaceReferenceProbe",
        code: 64,
        userInfo: [
          NSLocalizedDescriptionKey:
            "usage: MiniMaxH3FaceReferenceProbe <output-directory> <image>..."
        ]
      )
    }
    let output = URL(fileURLWithPath: CommandLine.arguments[1])
    let images = CommandLine.arguments.dropFirst(2).map {
      URL(fileURLWithPath: $0).standardizedFileURL
    }
    let faces = try await MiniMaxH3FaceReferenceProcessor.detectFaces(
      in: images,
      destinationDirectory: output
    )
    for face in faces {
      guard let source = CGImageSourceCreateWithURL(face.cropURL as CFURL, nil),
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
          as? [CFString: Any],
        let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
        let height = properties[kCGImagePropertyPixelHeight] as? NSNumber,
        width.intValue > 0,
        height.intValue > 0
      else {
        throw NSError(
          domain: "MiniMaxH3FaceReferenceProbe",
          code: 65,
          userInfo: [NSLocalizedDescriptionKey: "invalid face crop"]
        )
      }
      print(
        "source=\(face.sourceIndex) face=\(face.faceIndex) "
          + "subject=\(face.subjectIndex) selected=\(face.isSelected) "
          + "confidence=\(face.confidence) crop=\(width)x\(height) "
          + "path=\(face.cropURL.path)"
      )
    }
    if faces.count >= 2 {
      var grouped = faces
      grouped[0].subjectIndex = 1
      grouped[1].subjectIndex = 1
      let prompt = MiniMaxH3FaceReferenceProcessor.faceOnlyPrompt(
        "Two people walk through a new environment.",
        references: grouped
      )
      guard prompt.contains(
        "<Subject 1> is the person whose facial identity comes from <Picture 1>, <Picture 2>."
      ),
        !prompt.contains("<Subject 2> is the person"),
        prompt.contains("Regenerate clothing, body pose, background"),
        prompt.contains("detailed_description:"),
        prompt.contains("overall_soundscape:"),
        prompt.contains("non_diegetic_music:"),
        prompt.contains("Reference labels such as <Subject 1> and <Picture 1> are silent control metadata."),
        prompt.contains("Do not add narration, voice-over, dialogue, singing, or spoken reference labels unless the user explicitly requests speech."),
        !prompt.contains("user_request:"),
        prompt.contains("Two people walk through a new environment.")
      else {
        throw NSError(
          domain: "MiniMaxH3FaceReferenceProbe",
          code: 66,
          userInfo: [NSLocalizedDescriptionKey: "invalid subject/picture mapping"]
        )
      }
      print("grouped-prompt=PASS")
    }
    print("faces=\(faces.count)")
  }
}
