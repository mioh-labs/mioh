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
        "<Subject 1> is the person whose facial identity comes from <Picture 1>, <Picture 2>. The pictures are identity sources only, not frame, pose, outfit, body, lighting, background, or composition references."
      ),
        !prompt.contains("<Subject 2> is the person"),
        prompt.contains("Regenerate clothing, body pose, background"),
        prompt.contains("detailed_description:"),
        prompt.contains("overall_soundscape:"),
        prompt.contains("non_diegetic_music:"),
        prompt.contains("Reference labels such as <Subject 1> and <Picture 1> are silent control metadata."),
        prompt.contains("FACE IDENTITY ISOLATION:"),
        prompt.contains("Do not reproduce the reference pictures themselves."),
        prompt.contains("Do not copy their original clothing, body pose, body proportions, hand pose, camera angle, crop, background, room, lighting, color mood, photo texture, or composition"),
        prompt.contains("Do not transfer, blend, copy, clone, or echo their facial identity onto any other person."),
        prompt.contains("Unreferenced performers, friends, crowds, dancers, reflections, posters, and background people must have clearly different faces"),
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

      let structuredInput = """
        subject_definitions:
        <Subject 1> is an adult singer in a red floral dress.
        <Audio 1> is the complete song.

        summary:
        [reference generation + audio reuse] A coastal music video.

        retention_analysis:
        <Subject 1>: fully_preserved - preserve identity.
        <Audio 1>: fully_copy - reuse the song.

        detailed_description:
        [Shot 1] <Subject 1> sings while walking beside the sea.

        overall_soundscape:
        No added ambience.

        non_diegetic_music:
        <Audio 1> is reused directly.
        """
      let structuredPrompt = MiniMaxH3FaceReferenceProcessor.faceOnlyPrompt(
        structuredInput,
        references: grouped
      )
      guard structuredPrompt.components(separatedBy: "subject_definitions:").count == 2,
        structuredPrompt.components(separatedBy: "summary:").count == 2,
        structuredPrompt.components(separatedBy: "retention_analysis:").count == 2,
        structuredPrompt.components(separatedBy: "detailed_description:").count == 2,
        structuredPrompt.components(separatedBy: "overall_soundscape:").count == 2,
        structuredPrompt.components(separatedBy: "non_diegetic_music:").count == 2,
        structuredPrompt.contains(
          "Its facial identity comes from <Picture 1>, <Picture 2>"
        ),
        structuredPrompt.contains("Do not use those pictures as frame, pose, outfit, body, lighting, background, mood, or composition references."),
        structuredPrompt.contains("FACE IDENTITY ISOLATION:"),
        structuredPrompt.contains("<Audio 1> is reused directly."),
        !structuredPrompt.contains("Follow this user direction")
      else {
        throw NSError(
          domain: "MiniMaxH3FaceReferenceProbe",
          code: 67,
          userInfo: [NSLocalizedDescriptionKey: "structured prompt was nested"]
        )
      }
      print("structured-prompt=PASS")
    }
    print("faces=\(faces.count)")
  }
}
