// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AVFoundation
import CoreVideo
import Foundation

private enum PostprocessFailure: LocalizedError {
  case invalid(String)
  var errorDescription: String? {
    if case .invalid(let message) = self { return message }
    return nil
  }
}

private struct ROIFrameJob {
  let sceneIndex: Int
  let frame: SwiftVRROIFrameRecord
  let low: URL
  let high: URL
}

/// A separate, retryable pass. It never modifies the completed BasicVSR++
/// movie or the saved 256px ROI inputs.
@main
private enum SwiftVRROIPostprocess {
  static func main() async throws {
    guard CommandLine.arguments.count == 7 else {
      throw PostprocessFailure.invalid(
        "usage: mioh-native-swiftvr-roi-postprocess <roi-sidecar> <model-root> <output.mp4> <strength> <ffmpeg> <swiftvr-helper>"
      )
    }
    let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let model = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
    let output = URL(fileURLWithPath: CommandLine.arguments[3])
    guard let strength = Float(CommandLine.arguments[4]),
      (0...1).contains(strength) else {
      throw PostprocessFailure.invalid("SwiftVR strength must be between 0 and 1")
    }
    let ffmpeg = URL(fileURLWithPath: CommandLine.arguments[5])
    let helper = URL(fileURLWithPath: CommandLine.arguments[6])
    let fileManager = FileManager.default
    let stopFile = root.appendingPathComponent("stop-requested")
    let progressFile = root.appendingPathComponent("postprocess-status.json")
    if fileManager.fileExists(atPath: stopFile.path) {
      try fileManager.removeItem(at: stopFile)
    }
    guard !fileManager.fileExists(atPath: output.path) else {
      throw PostprocessFailure.invalid("Output already exists: \(output.path)")
    }
    try SwiftVRROIAssets.validate(model)
    let manifest = try JSONDecoder().decode(SwiftVRROISidecarManifest.self,
      from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
    let totalFrames = manifest.outputFramePTS.count
    let totalInferenceFrames = manifest.scenes.reduce(0) {
      $0 + $1.frames.count
    }
    var completedInferenceFrames = 0
    func report(_ phase: String, completed: Int, sceneIndex: Int? = nil,
      sceneFrames: Int = 0, sceneOutput: URL? = nil) throws {
      var payload: [String: Any] = [
        "phase": phase,
        "completed_frames": completed,
        "total_frames": totalFrames,
        "completed_inference_frames": completedInferenceFrames,
        "total_inference_frames": totalInferenceFrames,
      ]
      if let sceneIndex {
        payload["scene_index"] = sceneIndex + 1
        payload["scene_count"] = manifest.scenes.count
        payload["scene_frames"] = sceneFrames
      }
      if let sceneOutput { payload["scene_output"] = sceneOutput.path }
      try JSONSerialization.data(withJSONObject: payload)
        .write(to: progressFile, options: .atomic)
    }
    try report("preparing", completed: 0)
    let restored = URL(fileURLWithPath: manifest.restoredVideo)
    guard manifest.version == 1, manifest.complete,
      fileManager.fileExists(atPath: restored.path),
      manifest.frameWidth > 0, manifest.frameHeight > 0,
      manifest.fpsNumerator > 0, manifest.fpsDenominator > 0 else {
      throw PostprocessFailure.invalid("Incomplete BasicVSR++ movie or ROI sidecar")
    }
    if manifest.scenes.isEmpty || strength == 0 {
      try fileManager.copyItem(at: restored, to: output)
      return
    }
    // Compiling the model pack takes minutes and leaves the GPU idle, so the
    // compiled models persist across exports instead of per-run temporaries.
    let compiledCache = try persistentCompiledCache()
    var helperEnvironment = ProcessInfo.processInfo.environment
    helperEnvironment["MIOH_SWIFTVR_COMPILED_CACHE"] = compiledCache.path
    var helperSession: SwiftVRHelperSession?
    defer { helperSession?.terminate() }

    let enhancedRoot = root.appendingPathComponent("swiftvr-output", isDirectory: true)
    let modelKey = model.standardizedFileURL.path
    let modelMarker = enhancedRoot.appendingPathComponent("model-path.txt")
    if fileManager.fileExists(atPath: modelMarker.path) {
      let previous = try String(contentsOf: modelMarker, encoding: .utf8)
      if previous != modelKey {
        let cachedScenes = (try? fileManager.contentsOfDirectory(
          at: enhancedRoot, includingPropertiesForKeys: nil)) ?? []
        let hasHighFrames = cachedScenes.contains { scene in
          ((try? fileManager.contentsOfDirectory(
            at: scene, includingPropertiesForKeys: nil)) ?? [])
            .contains { $0.pathExtension == "f16" }
        }
        guard !hasHighFrames else {
          throw PostprocessFailure.invalid(
            "Existing SwiftVR ROI output uses a different model; choose a new sidecar"
          )
        }
        try modelKey.write(to: modelMarker, atomically: true,
          encoding: .utf8)
      }
    } else {
      try fileManager.createDirectory(at: enhancedRoot, withIntermediateDirectories: true)
      try modelKey.write(to: modelMarker, atomically: true, encoding: .utf8)
    }
    let keepHighCache = ProcessInfo.processInfo.environment[
      "MIOH_SWIFTVR_KEEP_HIGH_CACHE"] == "1"
    var recordsByPTS: [Int64: [ROIFrameJob]] = [:]
    var sceneOutputFolders: [URL] = []
    for (sceneIndex, scene) in manifest.scenes.enumerated() {
      guard scene.directory.range(of: #"^scene-[0-9]{6}$"#,
        options: .regularExpression) != nil else {
        throw PostprocessFailure.invalid("Unsafe ROI scene directory")
      }
      let sceneRoot = root.appendingPathComponent(scene.directory, isDirectory: true)
      let sceneOutput = enhancedRoot.appendingPathComponent(scene.directory,
        isDirectory: true)
      sceneOutputFolders.append(sceneOutput)
      for (index, frame) in scene.frames.enumerated() {
        if frame.outputEligible == false { continue }
        let high = sceneOutput.appendingPathComponent(
          String(format: "%04d.f16", index))
        let low = sceneRoot.appendingPathComponent(frame.inputFile)
        guard fileManager.fileExists(atPath: low.path) else {
          throw PostprocessFailure.invalid("Saved ROI input is missing: \(low.path)")
        }
        let existing = recordsByPTS[frame.ptsNanoseconds] ?? []
        // Adjacent temporal batches save their overlap twice, but with
        // crossfade disabled only the earlier batch is encoded. Retain other
        // spatially distinct ROIs in the same frame.
        let duplicate = frame.outputEligible == nil && existing.contains { prior in
          let a = prior.frame
          let left = max(a.cropLeft, frame.cropLeft)
          let top = max(a.cropTop, frame.cropTop)
          let right = min(a.cropLeft + a.cropWidth,
            frame.cropLeft + frame.cropWidth)
          let bottom = min(a.cropTop + a.cropHeight,
            frame.cropTop + frame.cropHeight)
          let intersection = max(0, right - left) * max(0, bottom - top)
          let union = a.cropWidth * a.cropHeight
            + frame.cropWidth * frame.cropHeight - intersection
          return union > 0 && Float(intersection) / Float(union) > 0.5
        }
        if !duplicate {
          recordsByPTS[frame.ptsNanoseconds, default: []].append(
            ROIFrameJob(sceneIndex: sceneIndex,
              frame: frame, low: low, high: high))
        }
      }
    }
    var outputCountByPTS: [Int64: Int] = [:]
    for pts in manifest.outputFramePTS {
      outputCountByPTS[pts, default: 0] += 1
    }
    var remainingFramesByScene: [Int: Int] = [:]
    for (pts, jobs) in recordsByPTS {
      for sceneIndex in Set(jobs.map(\.sceneIndex)) {
        remainingFramesByScene[sceneIndex, default: 0]
          += outputCountByPTS[pts] ?? 0
      }
    }
    var completedScenes: Set<Int> = []

    let asset = AVURLAsset(url: restored)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw PostprocessFailure.invalid("Completed restoration has no video track")
    }
    let sourceCodec = try await track.load(.formatDescriptions).first.map {
      CMFormatDescriptionGetMediaSubType($0)
    }
    let codec: AVVideoCodecType = sourceCodec == kCMVideoCodecType_HEVC
      ? .hevc : .h264
    let reader = try AVAssetReader(asset: asset)
    let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ])
    let provider = reader.outputProvider(for: trackOutput)
    try reader.start()
    let segmentDirectory = root.appendingPathComponent("enhanced-segments",
      isDirectory: true)
    let writer = try SegmentWriter(
      outputDirectory: segmentDirectory,
      width: manifest.frameWidth, height: manifest.frameHeight,
      fpsNumerator: manifest.fpsNumerator,
      fpsDenominator: manifest.fpsDenominator,
      generation: 0, segmentSeconds: 60,
      codec: codec,
      averageBitRate: Int(min(80_000_000, max(12_000_000,
        Double(manifest.frameWidth * manifest.frameHeight)
          * Double(manifest.fpsNumerator)
          / Double(manifest.fpsDenominator) * 0.22))),
      realTime: false, filePrefix: "swiftvr"
    )
    defer { writer.discard() }
    var segments: [SegmentEvent] = []
    for index in manifest.outputFramePTS.indices {
      if fileManager.fileExists(atPath: stopFile.path) {
        throw PostprocessFailure.invalid("SwiftVR processing was stopped")
      }
      let pts = manifest.outputFramePTS[index]
      let jobs = recordsByPTS[pts] ?? []
      for sceneIndex in Set(jobs.map(\.sceneIndex)).sorted()
        where !completedScenes.contains(sceneIndex) {
        let scene = manifest.scenes[sceneIndex]
        let sceneRoot = root.appendingPathComponent(scene.directory,
          isDirectory: true)
        let sceneOutput = sceneOutputFolders[sceneIndex]
        let allOutputsPresent = scene.frames.indices.allSatisfy { frameIndex in
          fileManager.fileExists(atPath: sceneOutput.appendingPathComponent(
            String(format: "%04d.f16", frameIndex)).path)
        }
        if !allOutputsPresent {
          try fileManager.createDirectory(at: sceneOutput,
            withIntermediateDirectories: true)
          try report("inference", completed: index,
            sceneIndex: sceneIndex, sceneFrames: scene.frames.count,
            sceneOutput: sceneOutput)
          // Core ML occasionally aborts the worker with an uncatchable
          // exception (seen once as an MPSGraph "unexpected rank" error that
          // did not reproduce). A dead worker is restarted once per scene.
          for attempt in 1...2 {
            if helperSession == nil {
              helperSession = try SwiftVRHelperSession(helper: helper,
                model: model, environment: helperEnvironment)
            }
            do {
              try helperSession?.infer(
                input: sceneRoot.appendingPathComponent("input"),
                output: sceneOutput, frames: scene.frames.count,
                stopFile: stopFile)
              break
            } catch where attempt == 1
              && helperSession?.isRunning == false
              && !fileManager.fileExists(atPath: stopFile.path)
            {
              print("SwiftVR worker stopped unexpectedly; restarting it for scene \(sceneIndex)")
              helperSession?.terminate()
              helperSession = nil
            }
          }
        }
        completedInferenceFrames += scene.frames.count
        try report("compositing", completed: index,
          sceneIndex: sceneIndex, sceneFrames: scene.frames.count)
        completedScenes.insert(sceneIndex)
      }
      guard let sample = try await provider.next(),
        let pixelBuffer = sample.withUnsafeSampleBuffer({
          CMSampleBufferGetImageBuffer($0)
        }) else {
        throw PostprocessFailure.invalid("Restored video frame count differs from ROI sidecar")
      }
      let writable = try copyFrame(pixelBuffer)
      for job in jobs {
        try composite(job.frame, low: job.low, high: job.high,
          into: writable, strength: strength)
      }
      if let segment = try await writer.append(pixelBuffer: writable,
        ptsNanoseconds: pts) {
        segments.append(segment)
      }
      for sceneIndex in Set(jobs.map(\.sceneIndex)) {
        remainingFramesByScene[sceneIndex, default: 0] -= 1
        if !keepHighCache,
          remainingFramesByScene[sceneIndex] == 0,
          fileManager.fileExists(atPath: sceneOutputFolders[sceneIndex].path) {
          try fileManager.removeItem(at: sceneOutputFolders[sceneIndex])
        }
      }
      if (index + 1) % 5 == 0 || index + 1 == totalFrames {
        try report("compositing", completed: index + 1)
      }
    }
    let hasExtraFrame = try await provider.next() != nil
    guard reader.status == .completed,
      !hasExtraFrame else {
      throw PostprocessFailure.invalid(
        "Restored movie/ROI frame count mismatch"
      )
    }
    if let segment = try await writer.finish() { segments.append(segment) }
    try report("muxing", completed: totalFrames)
    try mux(segments: segments, audio: restored, output: output,
      ffmpeg: ffmpeg, directory: root, stopFile: stopFile)
    try report("complete", completed: totalFrames)
  }

  private static func readHalf(_ url: URL, count: Int) throws -> [Float16] {
    let data = try Data(contentsOf: url)
    guard data.count == count * MemoryLayout<Float16>.size else {
      throw PostprocessFailure.invalid("Unexpected ROI buffer size: \(url.path)")
    }
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
  }

  private static func copyFrame(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
    let width = CVPixelBufferGetWidth(source)
    let height = CVPixelBufferGetHeight(source)
    guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA else {
      throw PostprocessFailure.invalid("Restored movie must decode as BGRA")
    }
    var destination: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
      kCVPixelFormatType_32BGRA,
      [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary,
      &destination)
    guard status == kCVReturnSuccess, let destination else {
      throw PostprocessFailure.invalid("Could not allocate SwiftVR output frame")
    }
    CVPixelBufferLockBaseAddress(source, .readOnly)
    CVPixelBufferLockBaseAddress(destination, [])
    defer {
      CVPixelBufferUnlockBaseAddress(destination, [])
      CVPixelBufferUnlockBaseAddress(source, .readOnly)
    }
    guard let src = CVPixelBufferGetBaseAddress(source),
      let dst = CVPixelBufferGetBaseAddress(destination) else {
      throw PostprocessFailure.invalid("Could not copy restored frame")
    }
    let sourceStride = CVPixelBufferGetBytesPerRow(source)
    let destStride = CVPixelBufferGetBytesPerRow(destination)
    for y in 0..<height {
      memcpy(dst.advanced(by: y * destStride),
        src.advanced(by: y * sourceStride), width * 4)
    }
    CVBufferPropagateAttachments(source, destination)
    return destination
  }

  private static func sample(_ values: [Float16], size: Int,
    channel: Int, x: Float, y: Float) -> Float {
    let xx = max(0, min(Float(size - 1), x))
    let yy = max(0, min(Float(size - 1), y))
    let x0 = Int(xx), y0 = Int(yy)
    let x1 = min(size - 1, x0 + 1), y1 = min(size - 1, y0 + 1)
    let dx = xx - Float(x0), dy = yy - Float(y0)
    let plane = channel * size * size
    let a = Float(values[plane + y0 * size + x0])
    let b = Float(values[plane + y0 * size + x1])
    let c = Float(values[plane + y1 * size + x0])
    let d = Float(values[plane + y1 * size + x1])
    return (a * (1 - dx) + b * dx) * (1 - dy)
      + (c * (1 - dx) + d * dx) * dy
  }

  private static func composite(_ frame: SwiftVRROIFrameRecord,
    low: URL, high: URL, into buffer: CVPixelBuffer,
    strength: Float) throws {
    let lowPixels = try readHalf(low, count: 3 * 256 * 256)
    let highPixels = try readHalf(high, count: 3 * 1024 * 1024)
    let sceneDirectory = low.deletingLastPathComponent().deletingLastPathComponent()
    let mask = try readHalf(sceneDirectory.appendingPathComponent(frame.maskFile),
      count: frame.cropWidth * frame.cropHeight)
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    guard frame.cropLeft >= 0, frame.cropTop >= 0,
      frame.cropWidth > 0, frame.cropHeight > 0,
      frame.cropLeft + frame.cropWidth <= width,
      frame.cropTop + frame.cropHeight <= height else {
      throw PostprocessFailure.invalid("ROI extends outside restored frame")
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
      throw PostprocessFailure.invalid("Could not access restored frame pixels")
    }
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<frame.cropHeight {
      let modelY = max(0, min(255, Float(frame.padTop)
        + (Float(y) + 0.5) * Float(frame.resizedHeight)
          / Float(frame.cropHeight) - 0.5))
      let highY = (modelY + 0.5) * 4 - 0.5
      for x in 0..<frame.cropWidth {
        let weight = strength * Float(mask[y * frame.cropWidth + x])
        if weight <= 0 { continue }
        let modelX = max(0, min(255, Float(frame.padLeft)
          + (Float(x) + 0.5) * Float(frame.resizedWidth)
            / Float(frame.cropWidth) - 0.5))
        let highX = (modelX + 0.5) * 4 - 0.5
        let pixel = (frame.cropTop + y) * stride + (frame.cropLeft + x) * 4
        for channel in 0..<3 {
          let modelChannel = 2 - channel // BGRA -> planar RGB
          let delta = sample(highPixels, size: 1024,
            channel: modelChannel, x: highX, y: highY)
            - sample(lowPixels, size: 256,
              channel: modelChannel, x: modelX, y: modelY)
          let value = Float(bytes[pixel + channel]) + weight * 255 * delta
          bytes[pixel + channel] = UInt8(max(0, min(255, value)).rounded())
        }
      }
    }
  }

  /// `~/Library/Caches/<bundle>/mioh/swiftvr-compiled`. Entries that no
  /// export has used for 30 days are removed; the helper refreshes the date
  /// of every entry it uses.
  private static func persistentCompiledCache() throws -> URL {
    let fileManager = FileManager.default
    let cache: URL
    if let configured = ProcessInfo.processInfo.environment[
      "MIOH_SWIFTVR_COMPILED_CACHE"], !configured.isEmpty
    {
      cache = URL(fileURLWithPath: configured, isDirectory: true)
    } else {
      guard let caches = fileManager.urls(for: .cachesDirectory,
        in: .userDomainMask).first else {
        throw PostprocessFailure.invalid("User cache directory is unavailable")
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

  private static func run(_ executable: URL, _ arguments: [String],
    stopFile: URL? = nil, environment: [String: String]? = nil) throws {
    let process = Process()
    process.executableURL = executable
    process.arguments = arguments
    process.environment = environment
    try process.run()
    while process.isRunning {
      if let stopFile, FileManager.default.fileExists(atPath: stopFile.path) {
        process.terminate()
        process.waitUntilExit()
        throw PostprocessFailure.invalid("SwiftVR processing was stopped")
      }
      Thread.sleep(forTimeInterval: 0.1)
    }
    guard process.terminationStatus == 0 else {
      throw PostprocessFailure.invalid(
        "Process failed (\(process.terminationStatus)): \(executable.lastPathComponent)"
      )
    }
  }

  private static func mux(segments: [SegmentEvent], audio: URL,
    output: URL, ffmpeg: URL, directory: URL,
    stopFile: URL) throws {
    guard !segments.isEmpty else {
      throw PostprocessFailure.invalid("No SwiftVR-enhanced video segments")
    }
    let list = directory.appendingPathComponent("enhanced-concat.txt")
    let lines = segments.sorted { $0.sequence < $1.sequence }.map { segment in
      "file '" + segment.path.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }.joined(separator: "\n") + "\n"
    try lines.write(to: list, atomically: true, encoding: .utf8)
    let part = output.deletingPathExtension().appendingPathExtension("part.mp4")
    if FileManager.default.fileExists(atPath: part.path) {
      // This exact sidecar-owned path can remain after an interrupted mux.
      try FileManager.default.removeItem(at: part)
    }
    func arguments(_ audioCodec: String) -> [String] {
      ["-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-f", "concat", "-safe", "0", "-i", list.path,
        "-i", audio.path, "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", audioCodec,
        "-map_metadata", "1",
        "-movflags", "+faststart", part.path]
    }
    do {
      try run(ffmpeg, arguments("copy"), stopFile: stopFile)
    } catch {
      if FileManager.default.fileExists(atPath: stopFile.path) {
        throw error
      }
      if FileManager.default.fileExists(atPath: part.path) {
        try FileManager.default.removeItem(at: part)
      }
      try run(ffmpeg, arguments("aac"), stopFile: stopFile)
    }
    try FileManager.default.moveItem(at: part, to: output)
  }
}

/// One SwiftVR worker serves every scene of an export instead of a new
/// process per scene. Scenes are still requested one at a time, when their
/// first output frame is reached, so the high-resolution cache stays bounded.
private final class SwiftVRHelperSession: @unchecked Sendable {
  private static let replyPrefix = "SWIFTVR-DONE "
  private let process = Process()
  private let requests = Pipe()
  private let responses = Pipe()
  private let lock = NSLock()
  private var pending = Data()
  private var completed: [String] = []

  init(helper: URL, model: URL, environment: [String: String]) throws {
    // A worker that exits between scenes must surface as an error on write,
    // not terminate this process with SIGPIPE.
    signal(SIGPIPE, SIG_IGN)
    process.executableURL = helper
    process.arguments = [model.path, "--serve"]
    process.environment = environment
    process.standardInput = requests
    process.standardOutput = responses
    responses.fileHandleForReading.readabilityHandler = { [weak self] handle in
      self?.receive(handle.availableData)
    }
    try process.run()
  }

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
        // Keep the worker's progress lines in postprocess.log as before.
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
      }
    }
  }

  private func takeCompleted() -> String? {
    lock.lock()
    defer { lock.unlock() }
    return completed.isEmpty ? nil : completed.removeFirst()
  }

  func infer(input: URL, output: URL, frames: Int, stopFile: URL) throws {
    let request = try JSONSerialization.data(withJSONObject: [
      "input": input.path, "output": output.path, "frames": frames,
    ]) + Data([0x0A])
    do {
      try requests.fileHandleForWriting.write(contentsOf: request)
    } catch {
      throw PostprocessFailure.invalid(
        "SwiftVR worker is not accepting scenes (\(process.terminationStatusIfExited))"
      )
    }
    while true {
      if let finished = takeCompleted() {
        guard finished == output.path else {
          throw PostprocessFailure.invalid("SwiftVR worker answered for another scene")
        }
        return
      }
      if FileManager.default.fileExists(atPath: stopFile.path) {
        terminate()
        throw PostprocessFailure.invalid("SwiftVR processing was stopped")
      }
      guard process.isRunning else {
        throw PostprocessFailure.invalid(
          "SwiftVR worker failed (\(process.terminationStatus))"
        )
      }
      Thread.sleep(forTimeInterval: 0.1)
    }
  }

  var isRunning: Bool { process.isRunning }

  /// Ends the worker. Closing its input lets an idle worker exit normally;
  /// one still inferring is terminated.
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

private extension Process {
  var terminationStatusIfExited: String {
    isRunning ? "running" : "exit \(terminationStatus)"
  }
}
