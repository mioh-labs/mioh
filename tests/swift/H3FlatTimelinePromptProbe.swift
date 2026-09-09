import Foundation

@main
struct H3FlatTimelinePromptProbe {
  static func main() throws {
    guard (2...4).contains(CommandLine.arguments.count) else {
      throw H3NativeError.invalidArguments(
        "expected a prompt path, optional entry count, and optional end time"
      )
    }
    let expectedEntryCount = CommandLine.arguments.count >= 3
      ? Int(CommandLine.arguments[2]) ?? 35 : 35
    let expectedEndSeconds = CommandLine.arguments.count >= 4
      ? Double(CommandLine.arguments[3]) ?? 210.5 : 210.5
    let prompt = try String(
      contentsOfFile: CommandLine.arguments[1],
      encoding: .utf8
    )
    guard let plan = try H3FlatTimelinePrompt.parse(prompt) else {
      throw H3NativeError.invalidJob("flat timeline was not detected")
    }
    guard plan.entries.count == expectedEntryCount else {
      throw H3NativeError.invalidJob(
        "expected \(expectedEntryCount) explicit entries, got \(plan.entries.count)"
      )
    }
    guard abs((plan.entries.last?.endSeconds ?? 0) - expectedEndSeconds)
      < 0.000_1
    else {
      throw H3NativeError.invalidJob(
        "timeline does not end at \(expectedEndSeconds) seconds"
      )
    }
    let normalizedBodies = plan.entries.map {
      $0.body.lowercased().filter { !$0.isWhitespace }
    }
    guard Set(normalizedBodies).count == plan.entries.count else {
      throw H3NativeError.invalidJob("timeline contains duplicate interval bodies")
    }
    for index in plan.entries.indices.dropFirst() {
      let previous = plan.entries[index - 1]
      let current = plan.entries[index]
      guard abs(previous.endSeconds - current.startSeconds) <= 1.0 / 24.0 else {
        throw H3NativeError.invalidJob("timeline is not contiguous at entry \(index)")
      }
    }
    for (index, entry) in plan.entries.enumerated() {
      let compiled = try plan.compiledPrompt(
        entryIndex: index,
        directive: "INTERVAL-\(index)"
      )
      guard compiled.contains("INTERVAL-\(index)"),
        compiled.contains(entry.body),
        !compiled.contains("[\(entry.startSeconds)-")
      else {
        throw H3NativeError.invalidJob(
          "compiled prompt \(index) does not isolate its interval body"
        )
      }
    }
    for entry in plan.entries where entry.transition == .cut {
      let singleTake = H3FlatTimelinePrompt.singleTakeBody(entry.body)
      let normalized = singleTake.trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
      guard !normalized.hasPrefix("hard cut to"),
        !normalized.hasPrefix("final hard cut to"),
        !normalized.hasPrefix("cut to")
      else {
        throw H3NativeError.invalidJob(
          "cut interval still asks H3 to perform an in-clip edit"
        )
      }
    }
    print(
      "valid: \(expectedEntryCount) unique entries, 0.000-\(expectedEndSeconds) seconds"
    )
  }
}
