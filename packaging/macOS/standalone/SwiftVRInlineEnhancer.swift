// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

// One-step SwiftVR ROI enhancement for mioh exports. Each BasicVSR++ scene is
// handed to a long-lived SwiftVR worker right after restoration, and its
// 1024px result is composited before the frame is encoded. There is no
// sidecar, no second pass and no re-encode.

import Foundation

enum SwiftVRROIAssets {
  static func isRoot(_ url: URL) -> Bool {
    FileManager.default.fileExists(atPath: url.appendingPathComponent(
      "reae-stateful-encoder-28f-1024-fp32.mlpackage"
    ).path) && FileManager.default.fileExists(atPath: url.appendingPathComponent(
      "native-4x-t7-fp16/components/patch.mlpackage"
    ).path)
  }

  static func validate(_ root: URL) throws {
    let fixed = [
      "reae-stateful-encoder-24f-1024-fp32.mlpackage",
      "reae-stateful-encoder-28f-1024-fp32.mlpackage",
      "reae-stateful-decoder-6latent-1024-fp32.mlpackage",
      "reae-stateful-decoder-7latent-1024-fp32.mlpackage",
    ]
    for relative in fixed where !FileManager.default.fileExists(
      atPath: root.appendingPathComponent(relative).path
    ) {
      throw NSError(domain: "SwiftVRROIAssets", code: 1, userInfo: [
        NSLocalizedDescriptionKey: "SwiftVR model is missing \(relative)"
      ])
    }
    for variant in ["t6", "t7"] {
      let prefix = "native-4x-\(variant)-fp16"
      let components = ["patch.mlpackage", "head.mlpackage", "context.f32",
        "modulation.f32", "rope-cosine.f32", "rope-sine.f32"]
      for name in components {
        let relative = "\(prefix)/components/\(name)"
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent(
          relative).path) else {
          throw NSError(domain: "SwiftVRROIAssets", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "SwiftVR model is missing \(relative)"
          ])
        }
      }
      for layer in 0..<30 {
        let name = String(format: "dit-block-%02d-%@-4x-float16.mlpackage",
          layer, variant)
        let relative = "\(prefix)/\(name)"
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent(
          relative).path) else {
          throw NSError(domain: "SwiftVRROIAssets", code: 3, userInfo: [
            NSLocalizedDescriptionKey: "SwiftVR model is missing \(relative)"
          ])
        }
      }
    }
  }
}

/// Compiled Core ML models persist across exports: compiling the model pack
/// takes minutes and leaves the GPU idle. Entries unused for 30 days are
/// removed; the worker refreshes the date of every entry it uses.
func swiftVRPersistentCompiledCache() throws -> URL {
  let fileManager = FileManager.default
  let cache: URL
  if let configured = ProcessInfo.processInfo.environment[
    "MIOH_SWIFTVR_COMPILED_CACHE"], !configured.isEmpty
  {
    cache = URL(fileURLWithPath: configured, isDirectory: true)
  } else {
    guard let caches = fileManager.urls(for: .cachesDirectory,
      in: .userDomainMask).first else {
      throw NSError(domain: "SwiftVR", code: 10, userInfo: [
        NSLocalizedDescriptionKey: "User cache directory is unavailable"
      ])
    }
    cache = caches
      .appendingPathComponent("com.okatti.lada.coreai", isDirectory: true)
      .appendingPathComponent("mioh", isDirectory: true)
      .appendingPathComponent("swiftvr-compiled", isDirectory: true)
  }
  try fileManager.createDirectory(at: cache, withIntermediateDirectories: true)
  let expiry = Date().addingTimeInterval(-30 * 24 * 60 * 60)
  for entry in (try? fileManager.contentsOfDirectory(at: cache,
    includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
  where entry.pathExtension == "mlmodelc" {
    let modified = try? entry.resourceValues(
      forKeys: [.contentModificationDateKey]).contentModificationDate
    if let modified, modified < expiry {
      try? fileManager.removeItem(at: entry)
    }
  }
  return cache
}

/// A long-lived `mioh-native-swiftvr-clip --serve` process. Scenes are sent
/// one at a time as JSON lines; the worker answers `SWIFTVR-DONE <output>`.
/// Its progress lines go to stderr, because this process's stdout carries
/// the JSON event stream read by mioh.
final class SwiftVRWorkerSession: @unchecked Sendable {
  private static let replyPrefix = "SWIFTVR-DONE "
  private let process = Process()
  private let requests = Pipe()
  private let responses = Pipe()
  private let lock = NSLock()
  private var pending = Data()
  private var completed: [String] = []

  init(worker: URL, model: URL, environment: [String: String]) throws {
    // A worker that exits between scenes must surface as a write error, not
    // terminate this process with SIGPIPE.
    signal(SIGPIPE, SIG_IGN)
    process.executableURL = worker
    process.arguments = [model.path, "--serve"]
    process.environment = environment
    process.standardInput = requests
    process.standardOutput = responses
    process.standardError = FileHandle.standardError
    responses.fileHandleForReading.readabilityHandler = { [weak self] handle in
      self?.receive(handle.availableData)
    }
    try process.run()
  }

  var isRunning: Bool { process.isRunning }

  private func receive(_ data: Data) {
    lock.lock()
    defer { lock.unlock() }
    pending.append(data)
    while let newline = pending.firstIndex(of: 0x0A) {
      let line = String(decoding: pending[pending.startIndex..<newline], as: UTF8.self)
      pending.removeSubrange(pending.startIndex...newline)
      if line.hasPrefix(Self.replyPrefix) {
        completed.append(String(line.dropFirst(Self.replyPrefix.count)))
      } else {
        FileHandle.standardError.write(Data(("SwiftVR: " + line + "\n").utf8))
      }
    }
  }

  private func takeCompleted() -> String? {
    lock.lock()
    defer { lock.unlock() }
    return completed.isEmpty ? nil : completed.removeFirst()
  }

  func infer(
    input: URL, output: URL, frames: Int, shouldStop: () -> Bool
  ) async throws {
    let request = try JSONSerialization.data(withJSONObject: [
      "input": input.path, "output": output.path, "frames": frames,
    ]) + Data([0x0A])
    do {
      try requests.fileHandleForWriting.write(contentsOf: request)
    } catch {
      throw NSError(domain: "SwiftVR", code: 11, userInfo: [
        NSLocalizedDescriptionKey: "SwiftVR worker is not accepting scenes"
      ])
    }
    while true {
      if let finished = takeCompleted() {
        guard finished == output.path else {
          throw NSError(domain: "SwiftVR", code: 12, userInfo: [
            NSLocalizedDescriptionKey: "SwiftVR worker answered for another scene"
          ])
        }
        return
      }
      if shouldStop() {
        terminate()
        throw CancellationError()
      }
      guard process.isRunning else {
        throw NSError(domain: "SwiftVR", code: 13, userInfo: [
          NSLocalizedDescriptionKey:
            "SwiftVR worker failed (\(process.terminationStatus))"
        ])
      }
      try await Task.sleep(for: .milliseconds(50))
    }
  }

  /// Closing the input lets an idle worker exit normally; one still
  /// inferring is terminated after a short grace period.
  func terminate() {
    try? requests.fileHandleForWriting.close()
    if process.isRunning {
      for _ in 0..<20 where process.isRunning {
        Thread.sleep(forTimeInterval: 0.05)
      }
      if process.isRunning { process.terminate() }
      process.waitUntilExit()
    }
    responses.fileHandleForReading.readabilityHandler = nil
  }
}

/// Runs SwiftVR over one restored scene at a time for the export pipeline.
/// Scene frames travel to the worker as temporary files that are removed as
/// soon as the scene is composited, so disk use is bounded by one scene
/// (about 0.4 MB in and 6 MB out per frame).
final class SwiftVRSceneEnhancer: @unchecked Sendable {
  static let outputSide = 1024

  let strength: Float
  /// Set once the export's stop control exists.
  var shouldStop: () -> Bool = { false }
  private let model: URL
  private let worker: URL
  private let workDirectory: URL
  private let environment: [String: String]
  private var session: SwiftVRWorkerSession?

  init(model: URL, strength: Float, workDirectory: URL) throws {
    try SwiftVRROIAssets.validate(model)
    self.model = model
    self.strength = max(0, min(1, strength))
    self.workDirectory = workDirectory
    worker = URL(fileURLWithPath: CommandLine.arguments[0])
      .deletingLastPathComponent()
      .appendingPathComponent("mioh-native-swiftvr-clip")
    guard FileManager.default.isExecutableFile(atPath: worker.path) else {
      throw NSError(domain: "SwiftVR", code: 14, userInfo: [
        NSLocalizedDescriptionKey: "SwiftVR worker is missing: \(worker.path)"
      ])
    }
    var environment = ProcessInfo.processInfo.environment
    environment["MIOH_SWIFTVR_COMPILED_CACHE"] = try swiftVRPersistentCompiledCache().path
    self.environment = environment
  }

  deinit { session?.terminate() }

  /// `restored` holds the scene's frames as consecutive planar RGB 256px
  /// FP16 images, exactly the BasicVSR++ result that is composited. Returns
  /// nil when the export is being stopped, so the pipeline finishes the scene
  /// without SwiftVR and stops the same way as an export without it.
  func enhance(restored: [Float16], frameCount: Int) async throws -> SwiftVRSceneOutput? {
    let plane = 3 * 256 * 256
    guard frameCount > 0, restored.count >= frameCount * plane else {
      throw NSError(domain: "SwiftVR", code: 15, userInfo: [
        NSLocalizedDescriptionKey: "SwiftVR scene input is incomplete"
      ])
    }
    let root = workDirectory.appendingPathComponent(
      "swiftvr-scene-\(UUID().uuidString)", isDirectory: true)
    let input = root.appendingPathComponent("input", isDirectory: true)
    let output = root.appendingPathComponent("output", isDirectory: true)
    try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
    do {
      try restored.withUnsafeBytes { bytes in
        for frame in 0..<frameCount {
          let start = frame * plane * MemoryLayout<Float16>.size
          try Data(bytes[start..<(start + plane * MemoryLayout<Float16>.size)])
            .write(to: input.appendingPathComponent(String(format: "%04d.f16", frame)))
        }
      }
      // Core ML occasionally aborts the worker with an uncatchable exception
      // (seen once as an MPSGraph "unexpected rank" error that did not
      // reproduce). A dead worker is restarted once per scene.
      for attempt in 1...2 {
        if session == nil {
          session = try SwiftVRWorkerSession(
            worker: worker, model: model, environment: environment)
        }
        do {
          try await session?.infer(
            input: input, output: output, frames: frameCount,
            shouldStop: shouldStop)
          break
        } catch where attempt == 1 && session?.isRunning == false && !shouldStop() {
          FileHandle.standardError.write(Data(
            "SwiftVR worker stopped unexpectedly; restarting it\n".utf8))
          session?.terminate()
          session = nil
        }
      }
    } catch {
      try? FileManager.default.removeItem(at: root)
      if shouldStop() || error is CancellationError { return nil }
      throw error
    }
    try? FileManager.default.removeItem(at: input)
    return SwiftVRSceneOutput(root: root, output: output)
  }

  func close() {
    session?.terminate()
    session = nil
  }
}

/// The worker's 1024px planar RGB FP16 frames for one scene.
struct SwiftVRSceneOutput {
  let root: URL
  let output: URL

  func frame(_ index: Int) throws -> [Float16] {
    let side = SwiftVRSceneEnhancer.outputSide
    let url = output.appendingPathComponent(String(format: "%04d.f16", index))
    let data = try Data(contentsOf: url)
    guard data.count == 3 * side * side * MemoryLayout<Float16>.size else {
      throw NSError(domain: "SwiftVR", code: 16, userInfo: [
        NSLocalizedDescriptionKey: "Unexpected SwiftVR output size: \(url.path)"
      ])
    }
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
  }

  func remove() {
    try? FileManager.default.removeItem(at: root)
  }
}
