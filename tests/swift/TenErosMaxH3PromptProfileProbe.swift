import Foundation

@main
struct TenErosMaxH3PromptProfileProbe {
  static func main() throws {
    guard CommandLine.arguments.count == 2 else {
      throw H3NativeError.invalidArguments(
        "usage: TenErosMaxH3PromptProfileProbe <qwen25-tokenizer-directory>"
      )
    }
    let tokenizer = try H3QwenBPETokenizer(
      directory: URL(fileURLWithPath: CommandLine.arguments[1])
    )
    var tokens = try tokenizer.encode("<Video 1>: ")
    for block in stride(from: 0, to: 20, by: 2) {
      let timestamp0 = Double(block) / 2
      let timestamp1 = Double(block + 1) / 2
      tokens += try tokenizer.encode(
        String(format: "<%.1f seconds>", (timestamp0 + timestamp1) / 2)
      )
      tokens.append(H3QwenPresentation.visionStart)
      tokens.append(
        contentsOf: repeatElement(H3QwenPresentation.imagePad, count: 405)
      )
      tokens.append(H3QwenPresentation.visionEnd)
    }
    let prefixCount = tokens.count
    let capacity = 4152 - prefixCount
    let defaultPrompt = try tokenizer.encode(
      "モザイクを除去して最高品質の動画を生成する。"
    )
    let englishPrompt = try tokenizer.encode(
      "Create an ultra-high-resolution, clean, cinematic video, masterwork."
    )
    print(
      "prefix=\(prefixCount) capacity=\(capacity) "
        + "default=\(defaultPrompt.count) english=\(englishPrompt.count)"
    )
    guard prefixCount == 4136, capacity == 16,
      defaultPrompt.count <= capacity, englishPrompt.count <= capacity
    else {
      throw H3NativeError.invalidTensor(
        "10Eros/Qwen free-prompt budget changed unexpectedly"
      )
    }

    let identityVideo = try H3Tensor(
      float16: [Float16](
        repeating: 0,
        count: 3 * 3 * 2 * 480 * 864
      ),
      shape: [1, 3, 3 * 2, 480, 864]
    )
    let identityPrompt = "A woman walks naturally through a quiet city at night. "
      + "overall_soundscape: soft footsteps and distant traffic. "
      + "non_diegetic_music: gentle electric piano with a clear melody."
    let identityPresentation = try H3QwenPresentation.makeReferenceVideo(
      prompt: identityPrompt,
      video: identityVideo,
      tokenizer: tokenizer,
      fixedSequenceLength: 4152,
      identityReferenceCount: 3
    )
    guard identityPresentation.inputIDs.shape == [1, 4152],
      identityPresentation.effectiveSequenceLength < 4152,
      identityPresentation.imageGridTHW.shape == [3, 3],
      identityPresentation.pixelValues.shape == [4_860, 1_536],
      identityPresentation.usedPromptTokenCount
        == identityPresentation.promptTokenCount,
      identityPresentation.usedPromptTokenCount > 16
    else {
      throw H3NativeError.invalidTensor(
        "MiniMax/Qwen identity-reference profile is incompatible"
      )
    }
  }
}
