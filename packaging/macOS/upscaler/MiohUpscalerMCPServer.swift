// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AppKit
import Foundation

private let mcpServerVersion = "0.14.3"

private struct MCPProgressEvent: Decodable {
  let stage: String?
  let state: String?
  let progress: Double?
  let message: String?
}

private struct MCPJobSnapshot: Encodable {
  let id: String
  let kind: String
  let state: String
  let progress: Double
  let message: String
  let output: String
  let pid: Int32?
  let startedAt: String
  let finishedAt: String?
  let logTail: String
}

private final class MCPManagedJob: @unchecked Sendable {
  let id: String
  let kind: String
  let output: String
  let startedAt = ISO8601DateFormatter().string(from: Date())
  var state = "starting"
  var progress = 0.0
  var message = "起動中"
  var finishedAt: String?
  var log = ""
  var stdoutBuffer = Data()
  var process: Process?
  let cleanupDirectories: [URL]

  init(
    id: String,
    kind: String,
    output: String,
    cleanupDirectories: [URL]
  ) {
    self.id = id
    self.kind = kind
    self.output = output
    self.cleanupDirectories = cleanupDirectories
  }
}

private final class MCPJobManager: @unchecked Sendable {
  static let shared = MCPJobManager()

  private let lock = NSLock()
  private var jobs: [String: MCPManagedJob] = [:]

  func launch(
    kind: String,
    executable: URL,
    arguments: [String],
    output: String,
    environment: [String: String]? = nil,
    cleanupDirectories: [URL] = []
  ) throws -> MCPJobSnapshot {
    let id = UUID().uuidString.lowercased()
    let job = MCPManagedJob(
      id: id,
      kind: kind,
      output: output,
      cleanupDirectories: cleanupDirectories
    )
    let process = Process()
    let stdout = Pipe()
    let stderr = Pipe()
    process.executableURL = executable
    process.arguments = arguments
    process.environment = environment
    process.standardInput = FileHandle.nullDevice
    process.standardOutput = stdout
    process.standardError = stderr
    job.process = process

    stdout.fileHandleForReading.readabilityHandler = { [weak self, weak job] handle in
      let data = handle.availableData
      guard !data.isEmpty, let self, let job else { return }
      self.consumeStandardOutput(data, for: job)
    }
    stderr.fileHandleForReading.readabilityHandler = { [weak self, weak job] handle in
      let data = handle.availableData
      guard !data.isEmpty, let self, let job else { return }
      self.appendLog(String(decoding: data, as: UTF8.self), to: job)
    }
    process.terminationHandler = { [weak self, weak job, weak stdout, weak stderr] task in
      stdout?.fileHandleForReading.readabilityHandler = nil
      stderr?.fileHandleForReading.readabilityHandler = nil
      guard let self, let job else { return }
      self.lock.withLock {
        if job.state == "stopping" {
          job.state = "stopped"
          job.message = "停止しました"
        } else if task.terminationStatus == 0 {
          job.state = "completed"
          job.progress = 1
          job.message = "完了"
        } else {
          job.state = "failed"
          job.message = "終了コード \(task.terminationStatus)"
        }
        job.finishedAt = ISO8601DateFormatter().string(from: Date())
      }
      for directory in job.cleanupDirectories {
        try? FileManager.default.removeItem(at: directory)
      }
    }

    lock.withLock { jobs[id] = job }
    do {
      try process.run()
      lock.withLock {
        job.state = "running"
        job.message = "実行中"
      }
    } catch {
      lock.withLock {
        job.state = "failed"
        job.message = error.localizedDescription
        job.finishedAt = ISO8601DateFormatter().string(from: Date())
      }
      for directory in cleanupDirectories {
        try? FileManager.default.removeItem(at: directory)
      }
      throw error
    }
    return snapshot(job)
  }

  func status(id: String) -> MCPJobSnapshot? {
    lock.withLock {
      guard let job = jobs[id] else { return nil }
      return snapshotUnlocked(job)
    }
  }

  func all() -> [MCPJobSnapshot] {
    lock.withLock {
      jobs.values.sorted { $0.startedAt > $1.startedAt }.map(snapshotUnlocked)
    }
  }

  func stop(id: String) -> MCPJobSnapshot? {
    let process: Process? = lock.withLock {
      guard let job = jobs[id] else { return nil }
      guard job.state == "running" || job.state == "starting" else {
        return job.process
      }
      job.state = "stopping"
      job.message = "停止中"
      return job.process
    }
    guard let process else { return status(id: id) }
    if process.isRunning { process.interrupt() }
    DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2) {
      if process.isRunning { process.terminate() }
    }
    return status(id: id)
  }

  private func consumeStandardOutput(_ data: Data, for job: MCPManagedJob) {
    lock.withLock {
      job.stdoutBuffer.append(data)
      while let newline = job.stdoutBuffer.firstIndex(of: 0x0A) {
        let line = Data(job.stdoutBuffer.prefix(upTo: newline))
        job.stdoutBuffer.removeSubrange(...newline)
        guard !line.isEmpty else { continue }
        if let event = try? JSONDecoder().decode(MCPProgressEvent.self, from: line) {
          if let progress = event.progress, progress.isFinite {
            job.progress = min(1, max(job.progress, progress))
          }
          job.message = event.message ?? event.state ?? event.stage ?? job.message
          appendLogUnlocked(String(decoding: line, as: UTF8.self) + "\n", to: job)
        } else {
          appendLogUnlocked(String(decoding: line, as: UTF8.self) + "\n", to: job)
        }
      }
    }
  }

  private func appendLog(_ text: String, to job: MCPManagedJob) {
    lock.withLock { appendLogUnlocked(text, to: job) }
  }

  private func appendLogUnlocked(_ text: String, to job: MCPManagedJob) {
    job.log += text
    if job.log.count > 32_000 {
      job.log = String(job.log.suffix(24_000))
    }
  }

  private func snapshot(_ job: MCPManagedJob) -> MCPJobSnapshot {
    lock.withLock { snapshotUnlocked(job) }
  }

  private func snapshotUnlocked(_ job: MCPManagedJob) -> MCPJobSnapshot {
    MCPJobSnapshot(
      id: job.id,
      kind: job.kind,
      state: job.state,
      progress: job.progress,
      message: job.message,
      output: job.output,
      pid: job.process?.isRunning == true ? job.process?.processIdentifier : nil,
      startedAt: job.startedAt,
      finishedAt: job.finishedAt,
      logTail: String(job.log.suffix(8_000))
    )
  }
}

private extension NSLock {
  func withLock<T>(_ body: () throws -> T) rethrows -> T {
    lock()
    defer { unlock() }
    return try body()
  }
}

private struct MCPUpscaleRequest: Codable {
  let input: String
  let output: String
  let model: String
  let modelRoot: String?
  let startSeconds: Double
  let endSeconds: Double?
  let durationSeconds: Double?
  let scale: Int
  let outputWidth: Int?
  let outputHeight: Int?
  let preserveAspectRatio: Bool
  let preserveAudio: Bool
  let computeMode: String
  let qualityMode: String
  let adcSRTemporalStrength: Double
}

private enum MCPServerError: LocalizedError {
  case invalidArguments(String)
  case missingFile(String)
  case unavailable(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments(let message), .missingFile(let message),
      .unavailable(let message):
      return message
    }
  }
}

private final class MiohMCPServer {
  private let manager = MCPJobManager.shared
  private let executableURL: URL
  private let resourcesURL: URL

  init() {
    executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
      .standardizedFileURL
    resourcesURL = executableURL.deletingLastPathComponent()
      .deletingLastPathComponent()
  }

  func run() {
    while let line = readLine(strippingNewline: true) {
      guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        let data = line.data(using: .utf8),
        let request = try? JSONSerialization.jsonObject(with: data)
          as? [String: Any]
      else { continue }
      handle(request)
    }
  }

  private func handle(_ request: [String: Any]) {
    let method = request["method"] as? String ?? ""
    let id = request["id"]
    guard id != nil else { return }
    do {
      switch method {
      case "initialize":
        let requested = (request["params"] as? [String: Any])?["protocolVersion"]
          as? String
        respond(
          id: id,
          result: [
            "protocolVersion": requested ?? "2025-06-18",
            "capabilities": ["tools": ["listChanged": false]],
            "serverInfo": ["name": "mioh-upscaler", "version": mcpServerVersion],
          ]
        )
      case "ping":
        respond(id: id, result: [:])
      case "tools/list":
        respond(id: id, result: ["tools": toolDefinitions()])
      case "tools/call":
        let params = request["params"] as? [String: Any] ?? [:]
        let name = params["name"] as? String ?? ""
        let arguments = params["arguments"] as? [String: Any] ?? [:]
        let value = try callTool(name, arguments: arguments)
        respondTool(id: id, value: value)
      default:
        respondError(id: id, code: -32601, message: "method not found: \(method)")
      }
    } catch {
      respondToolError(id: id, message: error.localizedDescription)
    }
  }

  private func callTool(_ name: String, arguments: [String: Any]) throws -> Any {
    switch name {
    case "mioh_capabilities":
      return try capabilities()
    case "mioh_start_video_generation":
      return try startVideoGeneration(arguments)
    case "mioh_start_upscale":
      return try startUpscale(arguments)
    case "mioh_get_job_status":
      let id = try requiredString("job_id", in: arguments)
      guard let snapshot = manager.status(id: id) else {
        throw MCPServerError.invalidArguments("job not found: \(id)")
      }
      return try jsonObject(snapshot)
    case "mioh_list_jobs":
      return try jsonObject(manager.all())
    case "mioh_stop_job":
      let id = try requiredString("job_id", in: arguments)
      guard let snapshot = manager.stop(id: id) else {
        throw MCPServerError.invalidArguments("job not found: \(id)")
      }
      return try jsonObject(snapshot)
    case "mioh_open_app":
      try openApplication()
      return ["opened": true, "app": applicationURL().path]
    default:
      throw MCPServerError.invalidArguments("unknown tool: \(name)")
    }
  }

  private func capabilities() throws -> [String: Any] {
    let bin = resourcesURL.appendingPathComponent("bin", isDirectory: true)
    let h3 = bin.appendingPathComponent("mioh-minimax-h3-native")
    let flash = bin.appendingPathComponent("flashvsr-coreai-video")
    let adcsr = bin.appendingPathComponent("adcsr-coreai-video")
    return [
      "app": applicationURL().path,
      "native_swift": true,
      "prompt_passthrough": "exact",
      "video_generation": FileManager.default.isExecutableFile(atPath: h3.path),
      "flashvsr": FileManager.default.isExecutableFile(atPath: flash.path),
      "adcsr": FileManager.default.isExecutableFile(atPath: adcsr.path),
      "default_manifest": defaultManifestPath(),
      "music_video_continuation_modes": [
        "hybrid-av", "latent-prefix", "first", "first-last-provided",
        "first-last-generated",
      ],
      "reference_edit_modes": ["none", "face_swap", "body_swap"],
      "tools": toolDefinitions().compactMap { $0["name"] },
    ]
  }

  private func startVideoGeneration(_ values: [String: Any]) throws -> Any {
    let prompt = try requiredString("prompt", in: values)
    let output = try requiredPath("output", in: values, mustExist: false)
    try validateNewOutput(output)
    let inputPaths = (values["input_files"] as? [String] ?? []).map {
      URL(fileURLWithPath: $0).standardizedFileURL.path
    }
    for path in inputPaths where !FileManager.default.fileExists(atPath: path) {
      throw MCPServerError.missingFile("input file not found: \(path)")
    }
    let audioPath = values["audio_input"] as? String
    if let audioPath, !FileManager.default.fileExists(atPath: audioPath) {
      throw MCPServerError.missingFile("audio input not found: \(audioPath)")
    }
    let referenceScope = values["reference_scope"] as? String ?? "whole_image"
    guard referenceScope == "whole_image" || referenceScope == "face_only" else {
      throw MCPServerError.invalidArguments(
        "reference_scope must be whole_image or face_only"
      )
    }
    let videoInputPaths = inputPaths.filter { !isImage($0) }
    let imageInputPaths = inputPaths.filter(isImage)
    guard videoInputPaths.count <= 1,
      videoInputPaths.count + imageInputPaths.count == inputPaths.count,
      imageInputPaths.count <= 8
    else {
      throw MCPServerError.invalidArguments(
        "input_files may contain one video plus up to eight images"
      )
    }
    let runtimeVideoPath = videoInputPaths.first
    var runtimeImagePaths = imageInputPaths
    var runtimeImageSubjects = (values["input_image_subjects"] as? [Int]) ?? []
    if !runtimeImageSubjects.isEmpty {
      guard runtimeImageSubjects.count == imageInputPaths.count,
        runtimeImageSubjects.allSatisfy({ (1...8).contains($0) })
      else {
        throw MCPServerError.invalidArguments(
          "input_image_subjects must match image input_files and use Subject numbers 1...8"
        )
      }
    }
    var runtimePrompt = prompt
    var cleanupDirectories: [URL] = []
    if referenceScope == "face_only", !imageInputPaths.isEmpty {
      guard imageInputPaths.count <= 8 else {
        throw MCPServerError.invalidArguments(
          "face_only requires one to eight reference images"
        )
      }
      let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(
          "mioh-h3-mcp-face-references-\(UUID().uuidString)",
          isDirectory: true
        )
      do {
        let references = try MiniMaxH3FaceReferenceProcessor
          .prepareAutomationReferences(
            in: imageInputPaths.map { URL(fileURLWithPath: $0) },
            destinationDirectory: directory
          )
        guard !references.isEmpty else {
          throw MCPServerError.invalidArguments(
            "face_only could not detect a face in the selected images"
          )
        }
        runtimeImagePaths = references.map(\.cropURL.path)
        runtimeImageSubjects = references.map(\.subjectIndex)
        runtimePrompt = MiniMaxH3FaceReferenceProcessor.faceOnlyPrompt(
          prompt,
          references: references
        )
        cleanupDirectories = [directory]
      } catch {
        try? FileManager.default.removeItem(at: directory)
        throw error
      }
    }
    let swapModeValue = values["swap_mode"] as? String ?? "none"
    let referenceEditMode: H3ReferenceEditMode
    switch swapModeValue {
    case "none":
      referenceEditMode = .none
    case "face_swap", "face-swap":
      referenceEditMode = .faceSwap
    case "body_swap", "body-swap":
      referenceEditMode = .bodySwap
    default:
      throw MCPServerError.invalidArguments(
        "swap_mode must be none, face_swap, or body_swap"
      )
    }
    if referenceEditMode != .none {
      guard runtimeVideoPath != nil, !runtimeImagePaths.isEmpty else {
        throw MCPServerError.invalidArguments(
          "swap_mode requires one source video and at least one reference image"
        )
      }
      runtimePrompt =
        referenceEditMode.promptPrefix(
          hasVideo: true,
          hasImages: true,
          hasAudio: audioPath != nil,
          targetDescription: values["swap_target"] as? String ?? ""
        ) + "\n\nuser_prompt:\n" + runtimePrompt
    }
    let musicVideo = values["music_video"] as? Bool ?? false
    let continuationMode = values["continuation_mode"] as? String ?? "hybrid-av"
    guard [
      "hybrid-av", "latent-prefix", "first", "first-last-provided",
      "first-last-generated",
    ].contains(continuationMode) else {
      throw MCPServerError.invalidArguments(
        "continuation_mode must be hybrid-av, latent-prefix, first, first-last-provided, or first-last-generated"
      )
    }
    let lastFrameDirectory = values["last_frame_directory"] as? String
    if musicVideo, continuationMode == "first-last-provided" {
      guard let lastFrameDirectory else {
        throw MCPServerError.invalidArguments(
          "first-last-provided requires last_frame_directory"
        )
      }
      var isDirectory: ObjCBool = false
      guard FileManager.default.fileExists(
        atPath: lastFrameDirectory,
        isDirectory: &isDirectory
      ), isDirectory.boolValue else {
        throw MCPServerError.missingFile(
          "last frame directory not found: \(lastFrameDirectory)"
        )
      }
    }
    let storyboardDirectory = values["storyboard_directory"] as? String
    if musicVideo, let storyboardDirectory {
      var isDirectory: ObjCBool = false
      guard FileManager.default.fileExists(
        atPath: storyboardDirectory,
        isDirectory: &isDirectory
      ), isDirectory.boolValue else {
        throw MCPServerError.missingFile(
          "storyboard directory not found: \(storyboardDirectory)"
        )
      }
    }
    let manifest = try resolvedManifest(
      explicit: values["manifest"] as? String,
      promptOnly: runtimeVideoPath == nil && runtimeImagePaths.isEmpty
    )
    let runner = resourcesURL.appendingPathComponent(
      "bin/mioh-minimax-h3-native"
    )
    guard FileManager.default.isExecutableFile(atPath: runner.path) else {
      throw MCPServerError.unavailable("MiniMax H3 runner is missing")
    }
    let width = integer("width", in: values, default: 864)
    let height = integer("height", in: values, default: 480)
    let outputWidth = integer("output_width", in: values, default: width)
    let outputHeight = integer("output_height", in: values, default: height)
    let duration = number("duration_seconds", in: values, default: 10)
    let audioStart = number("audio_start_seconds", in: values, default: 0)
    let seed = (values["seed"] as? String) ?? "261662374822964"
    guard width > 0, height > 0, outputWidth > 0, outputHeight > 0,
      duration >= 2, duration <= 15, UInt64(seed) != nil
    else {
      throw MCPServerError.invalidArguments("invalid dimensions, duration, or seed")
    }
    try FileManager.default.createDirectory(
      at: URL(fileURLWithPath: output).deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
      .first!.appendingPathComponent(
        "com.okatti.mioh.upscaler/10eros-max-h3", isDirectory: true
      )
    try FileManager.default.createDirectory(
      at: caches,
      withIntermediateDirectories: true
    )
    var arguments = [
      musicVideo ? "music-video" : "run",
      "--manifest", manifest,
      "--output", output,
      // Preserve the caller's wording, adding only the explicit reference-edit
      // contract requested through swap_mode.
      "--prompt", runtimePrompt,
      "--cache", caches.path,
      "--backend", "coreai",
      "--width", String(width),
      "--height", String(height),
      "--output-width", String(outputWidth),
      "--output-height", String(outputHeight),
      "--duration", posix(duration),
      "--seed", seed,
    ]
    if musicVideo {
      arguments += ["--music-video-continuation", continuationMode]
      if let lastFrameDirectory,
        continuationMode == "first-last-provided"
      {
        arguments += [
          "--music-video-last-frame-directory",
          URL(fileURLWithPath: lastFrameDirectory).standardizedFileURL.path,
        ]
      }
      if let storyboardDirectory {
        arguments += [
          "--music-video-storyboard-directory",
          URL(fileURLWithPath: storyboardDirectory).standardizedFileURL.path,
        ]
      }
    }
    if let audioPath {
      arguments += [
        "--audio-input", URL(fileURLWithPath: audioPath).standardizedFileURL.path,
        "--audio-start", posix(audioStart),
      ]
    }
    if let runtimeVideoPath {
      arguments += ["--input", runtimeVideoPath]
    }
    if referenceEditMode != .none {
      arguments += [
        "--reference-edit-mode", referenceEditMode.rawValue,
        "--physical-reference-mask", "1",
      ]
      let target = (values["swap_target"] as? String ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
      if !target.isEmpty {
        arguments += ["--reference-edit-target", target]
      }
      if let targetIndex = values["swap_target_index"] as? Int {
        guard targetIndex >= 0 else {
          throw MCPServerError.invalidArguments(
            "swap_target_index must be non-negative"
          )
        }
        arguments += ["--reference-edit-target-index", String(targetIndex)]
      }
    }
    if !runtimeImagePaths.isEmpty {
      let data = try JSONEncoder().encode(runtimeImagePaths)
      arguments += ["--input-images-json", String(decoding: data, as: UTF8.self)]
      if !runtimeImageSubjects.isEmpty {
        let subjectsData = try JSONEncoder().encode(runtimeImageSubjects)
        arguments += [
          "--input-image-subjects-json",
          String(decoding: subjectsData, as: UTF8.self),
        ]
      }
    }
    let snapshot = try manager.launch(
      kind: musicVideo ? "music_video" : "video_generation",
      executable: runner,
      arguments: arguments,
      output: output,
      cleanupDirectories: cleanupDirectories
    )
    return try jsonObject(snapshot)
  }

  private func startUpscale(_ values: [String: Any]) throws -> Any {
    let input = try requiredPath("input", in: values, mustExist: true)
    let output = try requiredPath("output", in: values, mustExist: false)
    try validateNewOutput(output)
    let model = (values["model"] as? String ?? "flashvsr").lowercased()
    guard model == "flashvsr" || model == "adcsr" else {
      throw MCPServerError.invalidArguments("model must be flashvsr or adcsr")
    }
    let request = MCPUpscaleRequest(
      input: input,
      output: output,
      model: model,
      modelRoot: values["model_root"] as? String,
      startSeconds: number("start_seconds", in: values, default: 0),
      endSeconds: optionalNumber("end_seconds", in: values),
      durationSeconds: optionalNumber("duration_seconds", in: values),
      scale: integer("scale", in: values, default: 2),
      outputWidth: optionalInteger("output_width", in: values),
      outputHeight: optionalInteger("output_height", in: values),
      preserveAspectRatio: values["preserve_aspect_ratio"] as? Bool ?? true,
      preserveAudio: values["preserve_audio"] as? Bool ?? true,
      computeMode: values["compute_mode"] as? String ?? "hybrid",
      qualityMode: values["quality_mode"] as? String ?? "fast",
      adcSRTemporalStrength: number(
        "adcsr_temporal_strength", in: values, default: 0.12
      )
    )
    guard request.startSeconds >= 0, request.scale == 2 || request.scale == 4 else {
      throw MCPServerError.invalidArguments("invalid start_seconds or scale")
    }
    let requestDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-upscaler-mcp", isDirectory: true)
    try FileManager.default.createDirectory(
      at: requestDirectory,
      withIntermediateDirectories: true
    )
    let requestURL = requestDirectory.appendingPathComponent(
      "upscale-\(UUID().uuidString).json"
    )
    try JSONEncoder().encode(request).write(to: requestURL, options: .atomic)
    let snapshot = try manager.launch(
      kind: "upscale_\(model)",
      executable: executableURL,
      arguments: ["upscale-worker", requestURL.path],
      output: output
    )
    return try jsonObject(snapshot)
  }

  private func resolvedManifest(explicit: String?, promptOnly: Bool) throws -> String {
    let selected = explicit?.trimmingCharacters(in: .whitespacesAndNewlines)
    let base = (selected?.isEmpty == false ? selected! : defaultManifestPath())
    var url = URL(fileURLWithPath: base).standardizedFileURL
    if promptOnly, url.lastPathComponent == "manifest.json" {
      let promptOnlyURL = url.deletingLastPathComponent()
        .appendingPathComponent("manifest-fl2va.json")
      if FileManager.default.fileExists(atPath: promptOnlyURL.path) {
        url = promptOnlyURL
      }
    }
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw MCPServerError.missingFile("manifest not found: \(url.path)")
    }
    return url.path
  }

  private func defaultManifestPath() -> String {
    let external =
      "/Volumes/Project_HD/model_weights/minimax-h3-native/manifest.json"
    if FileManager.default.fileExists(atPath: external) { return external }
    return FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(
        "Documents/lada/model_weights/minimax-h3-native/manifest.json"
      ).path
  }

  private func validateNewOutput(_ path: String) throws {
    if FileManager.default.fileExists(atPath: path) {
      throw MCPServerError.invalidArguments("output already exists: \(path)")
    }
  }

  private func openApplication() throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
    process.arguments = [applicationURL().path]
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
      throw MCPServerError.unavailable("could not open mioh upscaler")
    }
  }

  private func applicationURL() -> URL {
    resourcesURL.deletingLastPathComponent().deletingLastPathComponent()
  }

  private func requiredString(_ name: String, in values: [String: Any]) throws -> String {
    guard let string = values[name] as? String,
      !string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { throw MCPServerError.invalidArguments("missing \(name)") }
    return string
  }

  private func requiredPath(
    _ name: String,
    in values: [String: Any],
    mustExist: Bool
  ) throws -> String {
    let raw = try requiredString(name, in: values)
    let path = URL(fileURLWithPath: raw).standardizedFileURL.path
    if mustExist, !FileManager.default.fileExists(atPath: path) {
      throw MCPServerError.missingFile("file not found: \(path)")
    }
    return path
  }

  private func integer(_ name: String, in values: [String: Any], default fallback: Int) -> Int {
    optionalInteger(name, in: values) ?? fallback
  }

  private func optionalInteger(_ name: String, in values: [String: Any]) -> Int? {
    if let number = values[name] as? NSNumber { return number.intValue }
    if let string = values[name] as? String { return Int(string) }
    return nil
  }

  private func number(
    _ name: String,
    in values: [String: Any],
    default fallback: Double
  ) -> Double {
    optionalNumber(name, in: values) ?? fallback
  }

  private func optionalNumber(_ name: String, in values: [String: Any]) -> Double? {
    if let number = values[name] as? NSNumber { return number.doubleValue }
    if let string = values[name] as? String { return Double(string) }
    return nil
  }

  private func isImage(_ path: String) -> Bool {
    ["png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "webp"]
      .contains(URL(fileURLWithPath: path).pathExtension.lowercased())
  }

  private func posix(_ value: Double) -> String {
    String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), value)
  }

  private func jsonObject<T: Encodable>(_ value: T) throws -> Any {
    try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
  }

  private func respondTool(id: Any?, value: Any) {
    let data = (try? JSONSerialization.data(
      withJSONObject: value,
      options: [.prettyPrinted, .sortedKeys]
    )) ?? Data("null".utf8)
    let text = String(decoding: data, as: UTF8.self)
    respond(id: id, result: ["content": [["type": "text", "text": text]]])
  }

  private func respondToolError(id: Any?, message: String) {
    respond(
      id: id,
      result: [
        "content": [["type": "text", "text": message]],
        "isError": true,
      ]
    )
  }

  private func respond(id: Any?, result: Any) {
    write(["jsonrpc": "2.0", "id": id ?? NSNull(), "result": result])
  }

  private func respondError(id: Any?, code: Int, message: String) {
    write([
      "jsonrpc": "2.0", "id": id ?? NSNull(),
      "error": ["code": code, "message": message],
    ])
  }

  private func write(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
  }

  private func toolDefinitions() -> [[String: Any]] {
    [
      tool("mioh_capabilities", "利用可能なmioh機能とモデル状態を返します。", [:]),
      tool(
        "mioh_start_video_generation",
        "MiniMax H3動画生成を開始します。swap_mode指定時だけRef2VA用プロンプトを自動補強します。",
        [
          "prompt": property("string", "H3へそのまま渡す完全なプロンプト"),
          "output": property("string", "新規MP4の絶対パス"),
          "input_files": arrayProperty("string", "参照動画1本と参照画像最大8枚を同時指定できます"),
          "input_image_subjects": arrayProperty(
            "integer",
            "画像input_filesに対応するSubject番号。例: [1,1,2]"
          ),
          "reference_scope": enumProperty(
            ["whole_image", "face_only"],
            "画像全体、またはUIと同じVision顔クロップを使用"
          ),
          "swap_mode": enumProperty(
            ["none", "face_swap", "body_swap"],
            "動画+画像入力時にFace SwapまたはBody Swap用Ref2VA指示を自動追加"
          ),
          "swap_target": property(
            "string",
            "複数人動画で置換対象を指定するマスク説明。例: 左の男性、赤い服の人物"
          ),
          "swap_target_index": property(
            "integer",
            "検出人物候補の0始まり番号。指定時はswap_targetの左右指定より優先"
          ),
          "audio_input": property("string", "音源の絶対パス"),
          "audio_start_seconds": property("number", "音源の開始位置"),
          "music_video": property("boolean", "音源全体を解析する長尺MVモード"),
          "continuation_mode": enumProperty(
            [
              "hybrid-av", "latent-prefix", "first", "first-last-provided",
              "first-last-generated",
            ],
            "Part継続方式。Continuum＋H3-Extend融合、従来latent-prefix、First、Codex指定Last、mioh自動Lastを選択"
          ),
          "last_frame_directory": property(
            "string",
            "Codex作成Last画像のフォルダ（shot-0000-part-01-last.png形式）"
          ),
          "storyboard_directory": property(
            "string",
            "flat timeline各cutの実写構図アンカー（entry-0000.png形式）"
          ),
          "manifest": property("string", "H3 manifestの絶対パス"),
          "width": property("integer", "内部生成幅"),
          "height": property("integer", "内部生成高"),
          "output_width": property("integer", "出力幅"),
          "output_height": property("integer", "出力高"),
          "duration_seconds": property("number", "2〜15秒。MVではショット上限"),
          "seed": property("string", "UInt64 seed。精度維持のため文字列"),
        ],
        required: ["prompt", "output"]
      ),
      tool(
        "mioh_start_upscale",
        "FlashVSRまたはAdcSRでアップスケールを開始します。",
        [
          "input": property("string", "入力動画の絶対パス"),
          "output": property("string", "新規MP4の絶対パス"),
          "model": enumProperty(["flashvsr", "adcsr"], "使用モデル"),
          "model_root": property("string", "モデル格納場所"),
          "start_seconds": property("number", "開始秒"),
          "end_seconds": property("number", "終了秒"),
          "duration_seconds": property("number", "範囲秒数"),
          "scale": enumIntegerProperty([2, 4], "倍率"),
          "output_width": property("integer", "任意の出力幅"),
          "output_height": property("integer", "任意の出力高"),
          "preserve_aspect_ratio": property("boolean", "縦横比を維持"),
          "preserve_audio": property("boolean", "元音声を維持"),
          "compute_mode": enumProperty(["hybrid", "gpu"], "計算方式"),
          "quality_mode": enumProperty(["fast", "quality"], "FlashVSR品質"),
          "adcsr_temporal_strength": property("number", "AdcSR時間安定化0〜0.25"),
        ],
        required: ["input", "output"]
      ),
      tool(
        "mioh_get_job_status", "ジョブの進捗、PID、ログ末尾を返します。",
        ["job_id": property("string", "開始時に返されたjob id")],
        required: ["job_id"]
      ),
      tool("mioh_list_jobs", "このMCPセッションのジョブ一覧を返します。", [:]),
      tool(
        "mioh_stop_job", "実行中のジョブを安全に停止します。",
        ["job_id": property("string", "停止対象job id")],
        required: ["job_id"]
      ),
      tool("mioh_open_app", "mioh upscalerの画面を開きます。", [:]),
    ]
  }

  private func tool(
    _ name: String,
    _ description: String,
    _ properties: [String: Any],
    required: [String] = []
  ) -> [String: Any] {
    var schema: [String: Any] = [
      "type": "object",
      "properties": properties,
      "additionalProperties": false,
    ]
    if !required.isEmpty { schema["required"] = required }
    return ["name": name, "description": description, "inputSchema": schema]
  }

  private func property(_ type: String, _ description: String) -> [String: Any] {
    ["type": type, "description": description]
  }

  private func arrayProperty(_ itemType: String, _ description: String) -> [String: Any] {
    ["type": "array", "items": ["type": itemType], "description": description]
  }

  private func enumProperty(_ values: [String], _ description: String) -> [String: Any] {
    ["type": "string", "enum": values, "description": description]
  }

  private func enumIntegerProperty(_ values: [Int], _ description: String) -> [String: Any] {
    ["type": "integer", "enum": values, "description": description]
  }
}

@main
private struct MiohUpscalerMCPMain {
  static func main() async {
    if CommandLine.arguments.count == 3,
      CommandLine.arguments[1] == "upscale-worker"
    {
      do {
        try await runUpscaleWorker(
          requestURL: URL(fileURLWithPath: CommandLine.arguments[2])
        )
      } catch {
        emitWorker(progress: 0, state: "failed", message: error.localizedDescription)
        exit(1)
      }
      return
    }
    MiohMCPServer().run()
  }

  @MainActor
  private static func runUpscaleWorker(requestURL: URL) async throws {
    let request = try JSONDecoder().decode(
      MCPUpscaleRequest.self,
      from: Data(contentsOf: requestURL)
    )
    let executable = URL(fileURLWithPath: CommandLine.arguments[0])
      .standardizedFileURL
    let resources = executable.deletingLastPathComponent()
      .deletingLastPathComponent()
    let controller = VideoUpscaleController(resourceURL: resources)
    controller.upscalerModel = request.model
    controller.computeMode = request.computeMode
    controller.qualityMode = request.qualityMode
    controller.preserveAspectRatio = request.preserveAspectRatio
    controller.preserveAudio = request.preserveAudio
    controller.adcSRTemporalStrength = min(
      0.25, max(0, request.adcSRTemporalStrength)
    )
    if let modelRoot = request.modelRoot {
      if request.model == "adcsr" {
        controller.adcSRRootPath = modelRoot
      } else {
        controller.flashVSRRootPath = modelRoot
      }
    }
    controller.inputURL = URL(fileURLWithPath: request.input).standardizedFileURL
    let probeDeadline = Date().addingTimeInterval(30)
    while controller.sourceInfo == nil && controller.sourceInfoFailure == nil {
      guard Date() < probeDeadline else {
        throw MCPServerError.unavailable("timed out while reading input video")
      }
      try await Task.sleep(for: .milliseconds(100))
    }
    if let failure = controller.sourceInfoFailure {
      throw MCPServerError.unavailable(failure)
    }
    controller.startSeconds = request.startSeconds
    if let end = request.endSeconds {
      controller.endSeconds = end
    } else if let duration = request.durationSeconds {
      controller.endSeconds = request.startSeconds + duration
    } else {
      controller.endSeconds = controller.durationSeconds
    }
    if let width = request.outputWidth, let height = request.outputHeight {
      controller.sizingMode = "custom"
      controller.preserveAspectRatio = false
      controller.targetWidth = width
      controller.targetHeight = height
    } else {
      controller.sizingMode = "multiple"
      controller.scale = request.scale
    }
    controller.outputURL = URL(fileURLWithPath: request.output).standardizedFileURL
    guard controller.canStart else {
      throw MCPServerError.unavailable(
        controller.customSizeError ?? controller.modelAvailabilityText
      )
    }
    controller.start()
    guard controller.isRunning else {
      throw MCPServerError.unavailable(controller.status)
    }
    var lastStatus = ""
    var lastProgress = -1.0
    while controller.isRunning {
      if controller.status != lastStatus || controller.progress - lastProgress >= 0.005 {
        emitWorker(
          progress: controller.progress,
          state: "running",
          message: controller.status
        )
        lastStatus = controller.status
        lastProgress = controller.progress
      }
      try await Task.sleep(for: .milliseconds(250))
    }
    guard controller.status != "エラー" && controller.status != "停止" else {
      throw MCPServerError.unavailable(
        controller.log.isEmpty ? controller.status : String(controller.log.suffix(4_000))
      )
    }
    emitWorker(progress: 1, state: "completed", message: controller.status)
  }

  private static func emitWorker(progress: Double, state: String, message: String) {
    let object: [String: Any] = [
      "stage": "upscale", "state": state,
      "progress": progress, "message": message,
    ]
    guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
  }
}
