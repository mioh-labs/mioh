import Foundation

enum H3ReferenceEditMode: String, Codable, Sendable, CaseIterable {
  case none
  case faceSwap = "face-swap"
  case bodySwap = "body-swap"

  func promptPrefix(
    hasVideo: Bool,
    hasImages: Bool,
    hasAudio: Bool,
    targetDescription: String = ""
  ) -> String {
    guard self != .none, hasVideo, hasImages else { return "" }
    let target = targetDescription
      .trimmingCharacters(in: .whitespacesAndNewlines)
    let targetSubject = target.isEmpty
      ? "the main on-screen person"
      : "the target person identified by this mask description: \(target)"
    let targetSentence = target.isEmpty
      ? "Use the most visually central or narratively primary person in <Video 1> as the edit target."
      : "Use only the person matching this mask description as the edit target: \(target). Do not edit any other person in <Video 1>."
    let physicalMaskSentence =
      "If <Video 1> contains a translucent magenta overlay on a person, treat that overlay only as an invisible edit mask identifying the replacement target; never render the magenta overlay in the final video."
    let audioDefinition = hasAudio
      ? "\n<Audio 1> is an optional external audio reference for lip synchronization or final soundtrack timing."
      : ""
    let audioRetention = hasAudio
      ? "\n<Audio 1>: reference - use only the timing, voice, beat, or soundtrack relationship explicitly requested by the user."
      : ""
    let soundscape = hasAudio
      ? "<Audio 1> may guide timing only where the user prompt asks for it. Do not add unrelated dialogue or sound effects."
      : "Reuse the audible timing and ambience of <Video 1> when the source audio is present. Do not add unrelated dialogue or sound effects."
    switch self {
    case .none:
      return ""
    case .faceSwap:
      return """
        subject_definitions:
        <Subject 1> is the replacement face identity defined by <Picture 1> and any other images explicitly bound to <Subject 1>. Preserve the facial structure, eyes, nose, mouth, skin tone, age impression, and hairstyle cues from the image references.
        <Video 1> is the source video to be edited. Preserve its body performance, head pose, camera movement, framing, lighting, background, clothing, timing, and scene continuity unless the user prompt explicitly changes them.\(audioDefinition)

        summary:
        [video editing + reference generation] The target video is an edited version of <Video 1>. Replace only the visible face identity of \(targetSubject) with <Subject 1>, while keeping the original body, clothes, motion, camera, lighting, background, other people, and shot timing from <Video 1>.

        retention_analysis:
        <Subject 1> (face identity): attribute_transfer - transfer the referenced face identity onto \(targetSubject) in <Video 1> without copying the reference photo's background, clothing, pose, or framing.
        <Video 1> (source video edit): fully_preserved - preserve the original performance, body motion, camera path, scene layout, lighting, wardrobe, all unselected people, and temporal structure; only the selected face identity changes.\(audioRetention)

        detailed_description:
        The edit must look like the original footage was filmed with the replacement face already present. \(targetSentence) \(physicalMaskSentence) In [Shot 1], follow the source motion and camera timing of <Video 1> exactly. Replace only the selected person's face with <Subject 1> while preserving the original head rotation, gaze direction, expression intensity, mouth motion, body pose, clothing, hands, background, nearby people, and lighting integration. Do not introduce the reference image background, do not change the shot location, and do not redraw the body as a new character unless the user explicitly requests it.

        overall_soundscape:
        \(soundscape)

        non_diegetic_music:
        Preserve any music relationship requested by the user. If no music instruction is given, do not invent new music.
        """
    case .bodySwap:
      return """
        subject_definitions:
        <Subject 1> is the replacement character identity defined by <Picture 1> and any other images explicitly bound to <Subject 1>. Preserve the face, hairstyle, body proportions, clothing style, and overall person identity from the image references.
        <Video 1> is the source video to be edited. Preserve its camera movement, action timing, scene layout, lighting, background, and motion rhythm while replacing the main on-screen person.\(audioDefinition)

        summary:
        [video editing + reference generation] The target video is an edited version of <Video 1>. Replace \(targetSubject) with <Subject 1> as a full character replacement, while keeping the original camera, timing, scene, lighting, other people, and overall motion structure from <Video 1>.

        retention_analysis:
        <Subject 1> (replacement character): fully_preserved - use the reference images for the replacement person's face identity, body appearance, hairstyle, and wardrobe cues; do not copy the reference image background.
        <Video 1> (source video edit): partially_preserved - preserve the original camera, environment, lighting, physical timing, motion path, and all unselected people, but replace the selected visible person with <Subject 1>.\(audioRetention)

        detailed_description:
        The edit must look like <Subject 1> naturally performed inside the original source footage. \(targetSentence) \(physicalMaskSentence) In [Shot 1], follow the camera movement, pacing, perspective, scene geography, and action timing of <Video 1>. Replace only the selected person as a whole character with <Subject 1>, matching the original pose, motion path, scale, contact with the ground, and interaction with the environment. Keep the background, unselected people, and camera timing stable. Do not import the reference photo background, do not create extra people, and do not change the source scene unless the user explicitly requests it.

        overall_soundscape:
        \(soundscape)

        non_diegetic_music:
        Preserve any music relationship requested by the user. If no music instruction is given, do not invent new music.
        """
    }
  }
}
