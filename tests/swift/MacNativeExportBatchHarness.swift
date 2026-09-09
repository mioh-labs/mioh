import Foundation

@main
struct MacNativeExportBatchHarness {
  static func main() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-native-batch-\(UUID().uuidString)",
      isDirectory: true
    )
    let input = root.appendingPathComponent("input", isDirectory: true)
    let output = root.appendingPathComponent("output", isDirectory: true)
    try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let a = input.appendingPathComponent("A movie.mp4")
    let b = input.appendingPathComponent("b.MKV")
    let ignored = input.appendingPathComponent("notes.txt")
    try Data("a".utf8).write(to: a)
    try Data("b".utf8).write(to: b)
    try Data("ignored".utf8).write(to: ignored)
    let nested = input.appendingPathComponent("nested", isDirectory: true)
    try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
    try Data("nested".utf8).write(
      to: nested.appendingPathComponent("nested.mov")
    )

    let completed = output.appendingPathComponent("b-UC.MKV")
    try Data("done".utf8).write(to: completed)
    let plan = try MacNativeExportBatchPlanner.plan(
      input: input,
      selectedOutput: output,
      overwrite: false
    )
    guard plan.isDirectoryBatch,
      plan.discoveredCount == 2,
      plan.items.map(\.input.lastPathComponent) == ["A movie.mp4"],
      plan.items.map(\.output.lastPathComponent) == ["A movie-UC.mp4"],
      plan.skippedOutputs.map(\.lastPathComponent) == ["b-UC.MKV"]
    else { throw HarnessError.directoryPlan }

    let overwritePlan = try MacNativeExportBatchPlanner.plan(
      input: input,
      selectedOutput: output,
      overwrite: true
    )
    guard overwritePlan.items.map(\.input.lastPathComponent)
        == ["A movie.mp4", "b.MKV"],
      overwritePlan.skippedOutputs.isEmpty
    else { throw HarnessError.overwritePlan }

    let explicitOutput = root.appendingPathComponent("single-result.mp4")
    try Data("old".utf8).write(to: explicitOutput)
    let filePlan = try MacNativeExportBatchPlanner.plan(
      input: a,
      selectedOutput: explicitOutput,
      overwrite: false
    )
    guard !filePlan.isDirectoryBatch,
      filePlan.items.map(\.input.lastPathComponent) == ["A movie.mp4"],
      filePlan.items.map(\.output.lastPathComponent) == ["single-result.mp4"],
      filePlan.skippedOutputs.isEmpty
    else { throw HarnessError.filePlan }

    print("Mac native export batch harness passed")
  }

  enum HarnessError: Error {
    case directoryPlan
    case overwritePlan
    case filePlan
  }
}
