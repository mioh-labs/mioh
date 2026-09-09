import Foundation

enum MiniMaxH3PromptAssistantProvider: String, CaseIterable, Identifiable {
  case llamaCpp
  case openAICompatible
  case ollama

  var id: String { rawValue }

  var label: String {
    switch self {
    case .llamaCpp: return "llama.cpp"
    case .openAICompatible: return "OpenAI互換"
    case .ollama: return "Ollama"
    }
  }
}

struct MiniMaxH3PromptAssistantContext {
  let promptOnly: Bool
  let hasVideo: Bool
  let imageCount: Int
  let hasAudio: Bool
  let faceOnly: Bool
  let referenceEditMode: H3ReferenceEditMode
  let durationSeconds: Double
  let qwenSequenceLength: Int
}

enum MiniMaxH3PromptAssistantError: LocalizedError {
  case invalidEndpoint(String)
  case emptyInstruction
  case emptyResponse
  case serverError(String)

  var errorDescription: String? {
    switch self {
    case .invalidEndpoint(let value):
      return "ローカルAIのURLが不正です: \(value)"
    case .emptyInstruction:
      return "プロンプト生成の指示が空です"
    case .emptyResponse:
      return "ローカルAIからプロンプト本文が返りませんでした"
    case .serverError(let message):
      return "ローカルAIの応答エラー: \(message)"
    }
  }
}

enum MiniMaxH3PromptAssistant {
  static func generatePrompt(
    instruction: String,
    provider: MiniMaxH3PromptAssistantProvider,
    endpoint: String,
    model: String,
    context: MiniMaxH3PromptAssistantContext
  ) async throws -> String {
    let trimmedInstruction = instruction.trimmingCharacters(
      in: .whitespacesAndNewlines
    )
    guard !trimmedInstruction.isEmpty else {
      throw MiniMaxH3PromptAssistantError.emptyInstruction
    }
    let trimmedModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
    let system = systemPrompt(for: context)
    let user = """
      User request written in Japanese:
      \(trimmedInstruction)

      Rewrite it into one complete MiniMax H3 prompt. Return only the prompt text.
      """

    switch provider {
    case .llamaCpp:
      return try await generateWithOpenAICompatible(
        endpoint: endpoint,
        model: trimmedModel.isEmpty ? "gemma-4" : trimmedModel,
        system: system,
        user: user,
        defaultEndpoint: "http://127.0.0.1:18080"
      )
    case .ollama:
      return try await generateWithOllama(
        endpoint: endpoint,
        model: trimmedModel.isEmpty ? "qwen2.5:14b" : trimmedModel,
        system: system,
        user: user
      )
    case .openAICompatible:
      return try await generateWithOpenAICompatible(
        endpoint: endpoint,
        model: trimmedModel.isEmpty ? "local-model" : trimmedModel,
        system: system,
        user: user,
        defaultEndpoint: "http://127.0.0.1:1234"
      )
    }
  }

  private static func systemPrompt(
    for context: MiniMaxH3PromptAssistantContext
  ) -> String {
    let mode: String
    if context.referenceEditMode == .faceSwap {
      mode = "Ref2VA video editing / face swap"
    } else if context.referenceEditMode == .bodySwap {
      mode = "Ref2VA video editing / body swap"
    } else if context.hasVideo || context.imageCount > 0 || context.hasAudio {
      mode = "full-reference Ref2VA"
    } else if context.promptOnly {
      mode = "text-to-video / prompt-only FL2VA"
    } else {
      mode = "MiniMax H3 video generation"
    }

    let referenceSummary = """
      Current upscaler context:
      - mode: \(mode)
      - duration: \(String(format: "%.2f", context.durationSeconds)) seconds
      - source video supplied: \(context.hasVideo ? "yes, label it <Video 1>" : "no")
      - reference image count: \(context.imageCount) \(context.imageCount > 0 ? "(label them <Picture 1>, <Picture 2>, ... and use <Subject 1> when a person identity is intended)" : "")
      - external audio supplied: \(context.hasAudio ? "yes, label it <Audio 1>" : "no")
      - face-only reference mode: \(context.faceOnly ? "yes" : "no")
      - Qwen prompt budget: about \(context.qwenSequenceLength) tokens before visual references reduce available text rows
      """

    let structuredFormat: String
    if context.hasVideo || context.imageCount > 0 || context.hasAudio
      || context.referenceEditMode != .none
    {
      structuredFormat = """
        Use this exact section order:
        subject_definitions:
        summary:
        retention_analysis:
        detailed_description:
        overall_soundscape:
        non_diegetic_music:

        Write in English except for literal dialogue, lyrics, or visible text that the user explicitly wants preserved.
        Keep reference labels consistent: <Subject 1>, <Picture 1>, <Video 1>, <Audio 1>.
        Explain exactly what must be preserved, what may change, and what each reference contributes.
        """
    } else {
      structuredFormat = """
        Use this exact section order:
        integrated_multimodal_description:
        overall_soundscape:
        non_diegetic_music:

        Write in English except for literal dialogue, lyrics, or visible text that the user explicitly wants preserved.
        """
    }

    let swapRules: String
    switch context.referenceEditMode {
    case .none:
      swapRules = ""
    case .faceSwap:
      swapRules = """

        Face swap rules:
        - The target video is an edited version of <Video 1>.
        - Transfer only <Subject 1>'s face identity to the selected target person.
        - Preserve the source video's body, clothing, pose, camera, scene, lighting, timing, and all unselected people.
        - If a colored/physical mask is present in <Video 1>, treat it only as an invisible target selector; never render the mask color.
        """
    case .bodySwap:
      swapRules = """

        Body swap rules:
        - The target video is an edited version of <Video 1>.
        - Replace the selected target person as a whole character with <Subject 1>.
        - Preserve camera timing, scene layout, lighting, motion path, and all unselected people.
        - If a colored/physical mask is present in <Video 1>, treat it only as an invisible target selector; never render the mask color.
        """
    }

    return """
      You are a MiniMax H3 prompt writer embedded in mioh upscaler.
      Convert the user's Japanese creative instruction into a complete prompt that can be passed directly to MiniMax H3.

      \(referenceSummary)

      \(structuredFormat)

      Quality rules:
      - Be concrete and visual: subject, composition, motion, camera, lighting, environment, timing, and continuity.
      - Do not mention implementation details, UI, buttons, token limits, or that you are an AI.
      - Do not include markdown fences.
      - Avoid unresolved labels or vague summaries.
      - Keep the prompt compact enough to survive the model's prompt budget, but detailed enough for H3 to follow.
      \(swapRules)
      """
  }

  private static func generateWithOllama(
    endpoint: String,
    model: String,
    system: String,
    user: String
  ) async throws -> String {
    let url = try endpointURL(endpoint, defaultValue: "http://127.0.0.1:11434")
      .appendingPathComponent("api/chat")
    let request = OllamaChatRequest(
      model: model,
      messages: [
        .init(role: "system", content: system),
        .init(role: "user", content: user),
      ],
      stream: false
    )
    let response: OllamaChatResponse = try await postJSON(request, to: url)
    let content = response.message?.content.trimmingCharacters(
      in: .whitespacesAndNewlines
    ) ?? ""
    guard !content.isEmpty else {
      throw MiniMaxH3PromptAssistantError.emptyResponse
    }
    return content
  }

  private static func generateWithOpenAICompatible(
    endpoint: String,
    model: String,
    system: String,
    user: String,
    defaultEndpoint: String
  ) async throws -> String {
    let url = try endpointURL(endpoint, defaultValue: defaultEndpoint)
      .appendingPathComponent("v1/chat/completions")
    let request = OpenAICompatibleChatRequest(
      model: model,
      messages: [
        .init(role: "system", content: system),
        .init(role: "user", content: user),
      ],
      temperature: 0.4
    )
    let response: OpenAICompatibleChatResponse = try await postJSON(
      request,
      to: url
    )
    let content = response.choices.first?.message.content.trimmingCharacters(
      in: .whitespacesAndNewlines
    ) ?? ""
    guard !content.isEmpty else {
      throw MiniMaxH3PromptAssistantError.emptyResponse
    }
    return content
  }

  private static func endpointURL(
    _ endpoint: String,
    defaultValue: String
  ) throws -> URL {
    let value = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
    let resolved = value.isEmpty ? defaultValue : value
    guard let url = URL(string: resolved), url.scheme != nil else {
      throw MiniMaxH3PromptAssistantError.invalidEndpoint(resolved)
    }
    return url
  }

  private static func postJSON<RequestBody: Encodable, ResponseBody: Decodable>(
    _ body: RequestBody,
    to url: URL
  ) async throws -> ResponseBody {
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.timeoutInterval = 240
    request.httpBody = try JSONEncoder().encode(body)
    let (data, response) = try await URLSession.shared.data(for: request)
    if let http = response as? HTTPURLResponse,
      !(200..<300).contains(http.statusCode)
    {
      let message = String(data: data, encoding: .utf8)
        ?? "HTTP \(http.statusCode)"
      throw MiniMaxH3PromptAssistantError.serverError(message)
    }
    return try JSONDecoder().decode(ResponseBody.self, from: data)
  }
}

private struct OllamaChatRequest: Encodable {
  struct Message: Encodable {
    let role: String
    let content: String
  }

  let model: String
  let messages: [Message]
  let stream: Bool
}

private struct OllamaChatResponse: Decodable {
  struct Message: Decodable {
    let content: String
  }

  let message: Message?
}

private struct OpenAICompatibleChatRequest: Encodable {
  struct Message: Encodable {
    let role: String
    let content: String
  }

  let model: String
  let messages: [Message]
  let temperature: Double
}

private struct OpenAICompatibleChatResponse: Decodable {
  struct Choice: Decodable {
    struct Message: Decodable {
      let content: String
    }

    let message: Message
  }

  let choices: [Choice]
}
