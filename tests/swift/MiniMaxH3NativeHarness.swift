import Foundation

@main
struct MiniMaxH3NativeHarness {
  static func main() throws {
    if CommandLine.arguments.count > 1 {
      let prompt = try String(
        contentsOfFile: CommandLine.arguments[1],
        encoding: .utf8
      )
      let flat = try H3FlatTimelinePrompt.parse(prompt)
      try expect(flat != nil, "flat absolute timeline is detected")
      try expect(flat?.entries.count == 27, "full MV has 27 flat intervals")
      try expect(
        flat?.entries.first?.transition == .cut
          && flat?.entries.last?.transition == .continue,
        "flat timeline exposes only cut and continue transitions"
      )
      try expect(
        abs((flat?.entries.last?.endSeconds ?? 0) - 210.5) < 0.000_1,
        "flat timeline covers the complete source audio"
      )
      let compiled = try flat!.compiledPrompt(
        entryIndex: 1,
        directive: "DIRECTIVE"
      )
      try expect(
        compiled.contains("DIRECTIVE")
          && compiled.contains("Continue from her unfinished smile")
          && !compiled.contains("In a small sunlit apartment")
          && !compiled.contains("[8.000-16.250 continue]"),
        "compiled model prompt contains one concrete time range and no marker"
      )
    }
    try expect(H3Geometry.alignFrameCount(240) == 243, "240 frames align to 243")
    let referenceFrames = try H3Geometry.referenceFrameCount(available: 240, output: 243)
    try expect(
      referenceFrames == 226,
      "reference is trimmed down to the nearest 17n+5 count"
    )
    try expect(H3Geometry.videoLatentFrames(pixelFrames: 243) == 72, "video latent T")
    try expect(H3Geometry.audioLatentFrames(pixelFrames: 243) == 405, "audio latent T")
    let qwenIndices = H3Geometry.qwenVideoSampleIndices(frameCount: 226)
    try expect(qwenIndices.count == 20, "Qwen packs paired vision frames")
    try expect(qwenIndices.first == 0 && qwenIndices.last == 216, "Qwen sample bounds")
    try expect(qwenIndices[18] == 216 && qwenIndices[19] == 216, "odd sample padding")
    try expect(
      H3Geometry.alignedGenerationFrameCount(durationSeconds: 15) == 362,
      "15 seconds align to the H3 temporal geometry"
    )
    let longReferenceFrames = try H3Geometry.referenceFrameCount(
      available: 360,
      output: 362
    )
    try expect(longReferenceFrames == 345, "15-second reference frame count")
    let longQwenIndices = H3Geometry.qwenVideoSampleIndices(frameCount: 345)
    try expect(longQwenIndices.count == 20, "long Qwen reference fits ten blocks")
    try expect(
      longQwenIndices.first == 0 && longQwenIndices.last == 336,
      "long Qwen sampling preserves the complete time range"
    )
    let canvas = H3Geometry.adaptCanvas(width: 1920, height: 1080)
    try expect(canvas.width == 1344 && canvas.height == 768, "H3 canvas adaptation")

    let longFormPrompt = """
      subject_definitions:
      <Subject 1> is the same singer.

      detailed_description:
      Keep the same wardrobe in every Shot.
      [Shot 1] The singer walks on a beach.
      [Shot 2] At 00:10.000, the singer walks through a city.
      [Shot 3] At 00:20.000, the singer stands on a rooftop.

      overall_soundscape:
      No added ambience.

      non_diegetic_music:
      Reuse <Audio 1>.
      """
    let cityPrompt = try H3ShotPromptSelector.select(
      longFormPrompt,
      shotIndex: 1
    )
    try expect(
      cityPrompt.contains("subject_definitions:")
        && cityPrompt.contains("[Shot 2]")
        && cityPrompt.contains("overall_soundscape:")
        && !cityPrompt.contains("[Shot 1]")
        && !cityPrompt.contains("[Shot 3]")
        && cityPrompt.range(of: "[Shot 2]")!.lowerBound
          < cityPrompt.range(of: "Keep the same wardrobe")!.lowerBound,
      "logical Shot receives only its own scene and the global sections"
    )
    let unnamedPrompt = try H3ShotPromptSelector.select(
      longFormPrompt,
      shotIndex: 8
    )
    try expect(
      unnamedPrompt.contains("subject_definitions:")
        && unnamedPrompt.contains("overall_soundscape:")
        && !unnamedPrompt.contains("[Shot"),
      "an unnamed logical Shot never receives all named scenes"
    )
    let continuationPrompt = H3ShotPromptSelector.continuationContext(
      longFormPrompt
    )
    try expect(
      continuationPrompt.contains("subject_definitions:")
        && continuationPrompt.contains("overall_soundscape:")
        && continuationPrompt.contains("non_diegetic_music:")
        && !continuationPrompt.contains("detailed_description:")
        && !continuationPrompt.contains("walks on a beach")
        && !continuationPrompt.contains("walks through a city")
        && !continuationPrompt.contains("stands on a rooftop"),
      "continuation context keeps invariants without replaying Shot actions"
    )
    try expect(
      H3MusicVideoSeed.value(base: 42, intervalIndex: 0)
        != H3MusicVideoSeed.value(base: 42, intervalIndex: 1),
      "flat intervals use distinct future-noise trajectories"
    )
    try expect(
      H3MusicVideoSeed.value(base: 42, intervalIndex: 2)
        != H3MusicVideoSeed.value(base: 42, intervalIndex: 0),
      "nonadjacent flat intervals use distinct noise trajectories"
    )

    let audioAtNoise = H3AudioConditioning.samplerState(
      clean: [2], noise: [1], videoSigma: 1,
      videoShift: 12, audioShift: 3
    )
    try expect(close(audioAtNoise[0], 1), "audio condition starts at fixed noise")
    let audioNearClean = H3AudioConditioning.samplerState(
      clean: [2], noise: [1], videoSigma: 0.000_001,
      videoShift: 12, audioShift: 3
    )
    try expect(
      abs(audioNearClean[0] - 8) < 0.001,
      "audio condition ends in the sampler's 4x clean coordinate"
    )

    let continuationValues = (0..<(24 * 9)).map(Float.init)
    let continuationTensor = try H3Tensor(
      float32: continuationValues,
      shape: [1, 24, 9, 1, 1]
    )
    let continuationTail = try H3VideoConditioning.tail(
      continuationTensor
    )
    try expect(
      continuationTail.shape == [1, 24, 7, 1, 1],
      "Part continuation keeps one complete seven-token VAE decoder tile"
    )
    let continuationTailValues = try continuationTail.floatValues()
    try expect(
      continuationTailValues[0] == 2 && continuationTailValues[6] == 8
        && continuationTailValues[7] == 11,
      "Part continuation preserves the exact chronological seven-token suffix"
    )
    try expect(
      H3Geometry.alignedGenerationFrameCount(durationSeconds: 6) == 158
        && H3Geometry.alignedFrameCount(notAfter: 144) == 141
        && H3Geometry.isAlignedFrameCount(141)
        && !H3Geometry.isAlignedFrameCount(144),
      "visible continuation recognizes only exact 5+17n H3 boundaries"
    )
    let visibleValues = (0..<(24 * 17)).map(Float.init)
    let visibleTensor = try H3Tensor(
      float32: visibleValues,
      shape: [1, 24, 17, 1, 1]
    )
    let visibleTail = try H3VideoConditioning.tail(
      visibleTensor,
      endingAtPixelFrameCount: 39
    )
    let visibleTailValues = try visibleTail.floatValues()
    try expect(
      visibleTailValues[0] == 5 && visibleTailValues[6] == 11
        && visibleTailValues[7] == 22,
      "Part continuation excludes latent frames beyond the written movie end"
    )
    var rejectedUnalignedVisibleEnd = false
    do {
      _ = try H3VideoConditioning.tail(
        visibleTensor,
        endingAtPixelFrameCount: 40
      )
    } catch {
      rejectedUnalignedVisibleEnd = true
    }
    try expect(
      rejectedUnalignedVisibleEnd,
      "Part continuation rejects an unaligned written-movie boundary"
    )

    let hybridVideoTail = try H3VideoConditioning.tail(
      continuationTensor,
      maximumTokens: H3VideoConditioning.hybridStoredTokens
    )
    try expect(
      hybridVideoTail.shape == [1, 24, 9, 1, 1]
        && H3VideoConditioning.hybridHistoryTokens == 2,
      "hybrid continuation retains two older tokens plus the seven-token prefix"
    )
    let audioShape = [1, 32, 2, 80]
    let audioTensor = try H3Tensor(
      float32: (0..<audioShape.reduce(1, *)).map(Float.init),
      shape: audioShape
    )
    let hybridAudioTail = try H3AudioConditioning.tail(
      audioTensor,
      maximumFrames: H3AudioConditioning.hybridStoredLatentFrames,
      endingAtPixelFrameCount: 30
    )
    let hybridAudioValues = try hybridAudioTail.floatValues()
    try expect(
      hybridAudioTail.shape == [1, 32, 2, 50]
        && hybridAudioValues[0] == 0
        && hybridAudioValues[49] == 49
        && hybridAudioValues[50] == 80,
      "hybrid continuation retains the matching fifty-tick AV window"
    )
    let hybridPrefix = try H3AudioConditioning.slice(
      hybridAudioTail,
      frames: H3AudioConditioning.hybridHistoryLatentFrames..<H3AudioConditioning.hybridStoredLatentFrames
    )
    let targetAudio = try H3Tensor(
      float32: [Float](repeating: -1, count: 32 * 2 * 60),
      shape: [1, 32, 2, 60]
    )
    let replacedAudio = try H3AudioConditioning.replacingPrefix(
      in: targetAudio,
      with: hybridPrefix
    )
    let replacedAudioValues = try replacedAudio.floatValues()
    try expect(
      replacedAudioValues[0] == 13
        && replacedAudioValues[36] == 49
        && replacedAudioValues[37] == -1,
      "Continuum audio prefix replacement leaves future ticks untouched"
    )
    let stateURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-h3-hybrid-\(UUID().uuidString).plist")
    defer { try? FileManager.default.removeItem(at: stateURL) }
    try H3TemporalLatentStore.write(
      H3TemporalContinuationState(
        video: hybridVideoTail,
        audio: hybridAudioTail
      ),
      to: stateURL
    )
    let restoredState = try H3TemporalLatentStore.load(from: stateURL)
    try expect(
      restoredState.video.shape == hybridVideoTail.shape
        && restoredState.video.bytes == hybridVideoTail.bytes
        && restoredState.audio?.shape == hybridAudioTail.shape
        && restoredState.audio?.bytes == hybridAudioTail.bytes,
      "hybrid AV continuation state survives the resumable store round trip"
    )

    let tensor = try H3Tensor(float32: [1, 2, 3, 4], shape: [1, 2, 2])
    let half = try tensor.converted(to: .float16)
    try expect(half.scalarType == .float16 && half.bytes.count == 8, "Float16 boundary")
    let roundTrip = try half.converted(to: .float32).floatValues()
    try expect(roundTrip == [1, 2, 3, 4], "tensor conversion round trip")

    let euler = H3ResMultistep.euler(
      x: [1, 3], denoised: [0, 1], sigma: 1, sigmaDown: 0.5
    )
    try expect(close(euler[0], 0.5) && close(euler[1], 2), "Euler update")
    let coefficients = H3ResMultistep.secondOrderCoefficients(
      sigma: 0.8,
      oldSigmaDown: 0.8,
      sigmaDown: 0.5,
      previousSigma: 1
    )
    try expect(
      coefficients.expNegativeH.isFinite && coefficients.b1.isFinite
        && coefficients.b2.isFinite,
      "second-order coefficients are finite"
    )

    var rngA = H3SplitMix64(seed: 42)
    var rngB = H3SplitMix64(seed: 42)
    try expect(rngA.normal(count: 32) == rngB.normal(count: 32), "seed reproducibility")

    let keyA = H3StageCache.key(parts: [Data("prompt-a".utf8)])
    let keyB = H3StageCache.key(parts: [Data("prompt-b".utf8)])
    try expect(keyA != keyB, "cache key separates prompt conditions")
    print("MiniMaxH3NativeHarness: PASS")
  }

  private static func close(_ lhs: Float, _ rhs: Float) -> Bool {
    abs(lhs - rhs) < 0.000_01
  }

  private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    guard condition() else {
      throw H3NativeError.invalidArguments("test failed: \(message)")
    }
  }
}
