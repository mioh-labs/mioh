import Foundation

struct MacNativeExportBatchItem: Equatable {
  let input: URL
  let output: URL
}

struct MacNativeExportBatchPlan {
  let items: [MacNativeExportBatchItem]
  let discoveredCount: Int
  let skippedOutputs: [URL]
  let isDirectoryBatch: Bool
}

enum MacNativeExportBatchPlanner {
  static let videoExtensions: Set<String> = [
    "mp4", "m4v", "mov", "mkv", "avi", "webm", "ts", "mts", "m2ts",
    "mpg", "mpeg", "wmv", "asf",
  ]

  static func plan(
    input: URL,
    selectedOutput: URL,
    overwrite: Bool,
    fileManager: FileManager = .default
  ) throws -> MacNativeExportBatchPlan {
    let inputValues = try input.resourceValues(forKeys: [
      .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
    ])
    guard inputValues.isSymbolicLink != true else {
      throw PlannerError.symbolicLink(input.path)
    }
    if inputValues.isRegularFile == true {
      return MacNativeExportBatchPlan(
        items: [MacNativeExportBatchItem(
          input: input,
          output: resolvedOutputFile(input: input, selectedOutput: selectedOutput)
        )],
        discoveredCount: 1,
        skippedOutputs: [],
        isDirectoryBatch: false
      )
    }
    guard inputValues.isDirectory == true else {
      throw PlannerError.invalidInput(input.path)
    }

    let outputDirectory = try resolvedOutputDirectory(
      selectedOutput,
      fileManager: fileManager
    )
    let keys: Set<URLResourceKey> = [
      .isRegularFileKey, .isSymbolicLinkKey,
    ]
    let candidates = try fileManager.contentsOfDirectory(
      at: input,
      includingPropertiesForKeys: Array(keys),
      options: [.skipsHiddenFiles]
    ).filter { candidate in
      guard videoExtensions.contains(candidate.pathExtension.lowercased()),
        let values = try? candidate.resourceValues(forKeys: keys)
      else { return false }
      return values.isRegularFile == true && values.isSymbolicLink != true
    }.sorted {
      $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent)
        == .orderedAscending
    }
    guard !candidates.isEmpty else {
      throw PlannerError.noVideos(input.path)
    }

    var items: [MacNativeExportBatchItem] = []
    var skipped: [URL] = []
    for candidate in candidates {
      let output = resolvedOutputFile(
        input: candidate,
        selectedOutput: outputDirectory
      )
      if !overwrite, fileManager.fileExists(atPath: output.path) {
        skipped.append(output)
      } else {
        items.append(MacNativeExportBatchItem(input: candidate, output: output))
      }
    }
    return MacNativeExportBatchPlan(
      items: items,
      discoveredCount: candidates.count,
      skippedOutputs: skipped,
      isDirectoryBatch: true
    )
  }

  static func resolvedOutputFile(input: URL, selectedOutput: URL) -> URL {
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(
      atPath: selectedOutput.path,
      isDirectory: &isDirectory
    )
    let selectedDirectory = (exists && isDirectory.boolValue)
      || selectedOutput.hasDirectoryPath
      || selectedOutput.pathExtension.isEmpty
    guard selectedDirectory else { return selectedOutput }

    let stem = input.deletingPathExtension().lastPathComponent
    let ext = input.pathExtension.isEmpty ? "mp4" : input.pathExtension
    return selectedOutput
      .appendingPathComponent("\(stem)-UC")
      .appendingPathExtension(ext)
  }

  private static func resolvedOutputDirectory(
    _ selectedOutput: URL,
    fileManager: FileManager
  ) throws -> URL {
    var isDirectory: ObjCBool = false
    if fileManager.fileExists(
      atPath: selectedOutput.path,
      isDirectory: &isDirectory
    ) {
      guard isDirectory.boolValue else {
        throw PlannerError.batchOutputMustBeDirectory(selectedOutput.path)
      }
      return selectedOutput
    }
    guard selectedOutput.pathExtension.isEmpty || selectedOutput.hasDirectoryPath
    else {
      throw PlannerError.batchOutputMustBeDirectory(selectedOutput.path)
    }
    try fileManager.createDirectory(
      at: selectedOutput,
      withIntermediateDirectories: true
    )
    return selectedOutput
  }

  enum PlannerError: LocalizedError {
    case invalidInput(String)
    case symbolicLink(String)
    case noVideos(String)
    case batchOutputMustBeDirectory(String)

    var errorDescription: String? {
      switch self {
      case .invalidInput(let path):
        return "入力は動画ファイルまたはディレクトリではありません: \(path)"
      case .symbolicLink(let path):
        return "シンボリックリンクは入力に指定できません: \(path)"
      case .noVideos(let path):
        return "ディレクトリ直下に対応動画がありません: \(path)"
      case .batchOutputMustBeDirectory(let path):
        return "ディレクトリ入力では出力フォルダを指定してください: \(path)"
      }
    }
  }
}
