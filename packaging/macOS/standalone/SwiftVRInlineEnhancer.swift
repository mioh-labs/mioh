// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

// One-step SwiftVR ROI enhancement for mioh exports. Each BasicVSR++ scene is
// handed to a long-lived SwiftVR worker right after restoration, and its
// 512px (2x) or 1024px (4x) result is composited before the frame is encoded. There is no
// sidecar, no second pass and no re-encode.

import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

enum SwiftVRROIAssets {
  static func isRoot(_ url: URL) -> Bool {
    FileManager.default.fileExists(atPath: url.appendingPathComponent(
      "reae-stateful-encoder-28f-1024-fp32.mlpackage"
    ).path) && FileManager.default.fileExists(atPath: url.appendingPathComponent(
      "native-4x-t7-fp16/components/patch.mlpackage"
    ).path)
  }

  /// Checks the assets one enlargement factor needs: the ReAE graphs at its
  /// output size, the DiT components for both chunk shapes, and the DiT
  /// stack either as grouped packages or as 30 per-block packages per shape.
  static func validate(_ root: URL, scale: Int = 4) throws {
    let fileManager = FileManager.default
    func require(_ relative: String, code: Int) throws {
      guard fileManager.fileExists(atPath: root.appendingPathComponent(relative).path)
      else {
        throw NSError(domain: "SwiftVRROIAssets", code: code, userInfo: [
          NSLocalizedDescriptionKey: "SwiftVR \(scale)x model is missing \(relative)"
        ])
      }
    }
    guard [2, 4].contains(scale) else {
      throw NSError(domain: "SwiftVRROIAssets", code: 4, userInfo: [
        NSLocalizedDescriptionKey: "SwiftVR supports 2x and 4x, not \(scale)x"
      ])
    }
    let size = 256 * scale
    for relative in [
      "reae-stateful-encoder-24f-\(size)-fp32.mlpackage",
      "reae-stateful-encoder-28f-\(size)-fp32.mlpackage",
      "reae-stateful-decoder-6latent-\(size)-fp32.mlpackage",
      "reae-stateful-decoder-7latent-\(size)-fp32.mlpackage",
    ] {
      try require(relative, code: 1)
    }
    let grouped = ((try? fileManager.contentsOfDirectory(atPath: root
      .appendingPathComponent("native-\(scale)x-fp16-grouped").path)) ?? [])
      .filter { $0.hasPrefix("dit-group-") && $0.hasSuffix(".mlpackage") }
    for variant in ["t6", "t7"] {
      let prefix = "native-\(scale)x-\(variant)-fp16"
      for name in ["patch.mlpackage", "head.mlpackage", "context.f32",
        "modulation.f32", "rope-cosine.f32", "rope-sine.f32"]
      {
        try require("\(prefix)/components/\(name)", code: 2)
      }
      guard grouped.isEmpty else { continue }
      for layer in 0..<30 {
        try require(String(format: "%@/dit-block-%02d-%@-%dx-float16.mlpackage",
          prefix, layer, variant, scale), code: 3)
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

/// Admits one caller at a time without blocking a thread while it waits.
private final class SwiftVRTurn: @unchecked Sendable {
  private let lock = NSLock()
  private var busy = false
  private var waiters: [CheckedContinuation<Void, Never>] = []

  func acquire() async {
    await withCheckedContinuation { (waiter: CheckedContinuation<Void, Never>) in
      lock.lock()
      if busy {
        waiters.append(waiter)
        lock.unlock()
      } else {
        busy = true
        lock.unlock()
        waiter.resume()
      }
    }
  }

  func release() {
    lock.lock()
    if waiters.isEmpty {
      busy = false
      lock.unlock()
    } else {
      let next = waiters.removeFirst()
      lock.unlock()
      next.resume()
    }
  }
}

/// Runs SwiftVR over one restored scene at a time inside the export process.
/// Scene frames go from memory straight into Core ML, and each result frame
/// is handed to the compositor as soon as it is decoded, so only the frames
/// not yet composited are held (about 1.5 MB (2x) or 6 MB (4x) each).
final class SwiftVRSceneEnhancer: @unchecked Sendable {
  let strength: Float
  /// 2 (512px output, about a quarter of the DiT work) or 4 (1024px).
  let scale: Int
  var outputSide: Int { 256 * scale }
  /// Strength of Apple's temporal noise filter over the composited SwiftVR
  /// frames; 0 turns it off.
  let temporalFilter: Float
  /// Frames on each side that the stabilization blends (1...8).
  let stabilizationRadius: Int
  /// Set once the export's stop control exists.
  var shouldStop: () -> Bool = { false }
  private let model: URL
  private let compiledCache: URL
  /// Export lanes take turns: a lane holds the turn while SwiftVR produces
  /// its scene, and the other lanes keep detecting, restoring, compositing
  /// and encoding meanwhile. Running two scenes at once would double the
  /// resident DiT memory without more GPU to run on.
  private let turn = SwiftVRTurn()
  /// Core ML runs off the Swift concurrency pool; SwiftVR's model state is
  /// process-wide, so one queue serves every scene.
  private static let queue = DispatchQueue(
    label: "com.okatti.mioh.swiftvr", qos: .userInitiated,
    autoreleaseFrequency: .workItem)

  init(
    model: URL, strength: Float, scale: Int, temporalFilter: Float, stabilizationRadius: Int
  ) throws {
    try SwiftVRROIAssets.validate(model, scale: scale)
    self.scale = scale
    self.stabilizationRadius = max(1, min(8, stabilizationRadius))
    self.model = model
    self.strength = max(0, min(1, strength))
    let temporalFilter = max(0, min(1, temporalFilter))
    if temporalFilter > 0, !VTTemporalNoiseFilterConfiguration.isSupported {
      FileHandle.standardError.write(Data(
        "SwiftVR: the temporal noise filter is unavailable on this Mac; it stays off\n".utf8))
      self.temporalFilter = 0
    } else {
      self.temporalFilter = temporalFilter
    }
    compiledCache = try swiftVRPersistentCompiledCache()
  }

  /// Starts SwiftVR on a scene once this lane has the turn and returns
  /// without waiting for it; frames arrive through the returned output.
  /// `restored` holds the scene's frames as consecutive planar RGB 256px
  /// FP16 images, exactly the BasicVSR++ result that is composited. Returns
  /// nil when the export is being stopped.
  func enhance(restored: [Float16], frameCount: Int) async throws -> SwiftVRSceneOutput? {
    let plane = 3 * 256 * 256
    guard frameCount > 0, restored.count >= frameCount * plane else {
      throw NSError(domain: "SwiftVR", code: 15, userInfo: [
        NSLocalizedDescriptionKey: "SwiftVR scene input is incomplete"
      ])
    }
    await turn.acquire()
    if shouldStop() {
      turn.release()
      return nil
    }
    let output = SwiftVRSceneOutput(side: outputSide)
    let (model, compiledCache, scale, shouldStop, turn) =
      (model, compiledCache, scale, shouldStop, turn)
    Self.queue.async {
      defer { turn.release() }
      do {
        try restored.withUnsafeBufferPointer { input in
          try runSwiftVRScene(
            root: model, compiledCache: compiledCache, input: input,
            frames: frameCount, scale: scale,
            deliver: { output.deliver($0, $1) },
            shouldStop: { shouldStop() || output.isCancelled })
        }
        output.finish(nil)
      } catch {
        output.finish(error)
      }
    }
    return output
  }

  func close() {
    Self.queue.sync { releaseSwiftVRModels() }
  }
}

/// SwiftVR's 512px or 1024px planar RGB FP16 frames for one scene, handed
/// from the SwiftVR queue to one compositing task as they are produced.
final class SwiftVRSceneOutput: @unchecked Sendable {
  let side: Int
  private let lock = NSLock()
  private var frames: [Int: [Float16]] = [:]
  private var finished = false
  private var failure: Error?
  private var cancelled = false
  private var waiter: CheckedContinuation<Void, Never>?

  fileprivate init(side: Int) {
    self.side = side
  }

  fileprivate var isCancelled: Bool {
    lock.withLock { cancelled }
  }

  fileprivate func deliver(_ index: Int, _ pixels: [Float16]) {
    let waiting = lock.withLock { () -> CheckedContinuation<Void, Never>? in
      frames[index] = pixels
      defer { waiter = nil }
      return waiter
    }
    waiting?.resume()
  }

  fileprivate func finish(_ error: Error?) {
    let waiting = lock.withLock { () -> CheckedContinuation<Void, Never>? in
      finished = true
      failure = error is CancellationError ? nil : error
      defer { waiter = nil }
      return waiter
    }
    waiting?.resume()
  }

  /// Waits for frame `index`. Returns nil when SwiftVR stopped before
  /// producing it, and throws when SwiftVR failed.
  func frame(_ index: Int) async throws -> [Float16]? {
    while true {
      let ready = try lock.withLock { () throws -> [Float16]?? in
        if let pixels = frames[index] { return .some(pixels) }
        if let failure { throw failure }
        return finished ? .some(nil) : nil
      }
      if let ready { return ready }
      await withCheckedContinuation { (waiting: CheckedContinuation<Void, Never>) in
        let resumeNow = lock.withLock { () -> Bool in
          if frames[index] != nil || finished { return true }
          waiter = waiting
          return false
        }
        if resumeNow { waiting.resume() }
      }
    }
  }

  /// Releases the frames before `index`, which the compositor no longer needs.
  func discard(before index: Int) {
    lock.withLock { frames = frames.filter { $0.key >= index } }
  }

  /// Stops SwiftVR at its next check when the compositor gives up early.
  func cancel() {
    lock.withLock { cancelled = true }
  }
}

/// Apple's temporal noise filter (VideoToolbox, macOS 26) over one scene's
/// composited SwiftVR frames. SwiftVR re-synthesizes texture every frame and
/// amplifies small changes in its input into flicker; the filter compares
/// each frame with its motion-compensated neighbours (one before, two after).
/// The export blends the result back only where SwiftVR was blended in.
///
/// The filter accepts only compressed 4:2:0 formats, so each rectangle passes
/// from the 8-bit BGRA frame through 8-bit full-range 4:2:0 and back.
final class SwiftVRTemporalNoiseFilter {
  let previousCount: Int
  let nextCount: Int
  private let width: Int
  private let height: Int
  private let strength: Float
  private let processor = VTFrameProcessor()
  private let transfer: VTPixelTransferSession
  private let sourcePool: CVPixelBufferPool
  private let destinationPool: CVPixelBufferPool
  private let packedPool: CVPixelBufferPool
  private var frames: [Int: VTFrameProcessorFrame] = [:]

  init(width: Int, height: Int, strength: Float) throws {
    self.width = width
    self.height = height
    self.strength = strength
    guard let configuration = VTTemporalNoiseFilterConfiguration(
      frameWidth: width, frameHeight: height,
      sourcePixelFormat: kCVPixelFormatType_Lossless_420YpCbCr8BiPlanarFullRange)
    else { throw Self.failure("is not available for \(width)x\(height) frames") }
    previousCount = configuration.previousFrameCount ?? 1
    nextCount = configuration.nextFrameCount ?? 2
    try processor.startSession(configuration: configuration)
    var session: VTPixelTransferSession?
    guard VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &session)
      == noErr, let session
    else { throw Self.failure("could not create a pixel transfer session") }
    transfer = session
    sourcePool = try Self.pool(configuration.sourcePixelBufferAttributes)
    destinationPool = try Self.pool(configuration.destinationPixelBufferAttributes)
    packedPool = try Self.pool([
      kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ])
  }

  deinit { processor.endSession() }

  /// Hands over the rectangle at (`left`, `top`) of composited BGRA frame
  /// `index`.
  func add(_ index: Int, from frame: CVPixelBuffer, left: Int, top: Int) throws {
    let packed = try buffer(packedPool)
    CVPixelBufferLockBaseAddress(frame, .readOnly)
    CVPixelBufferLockBaseAddress(packed, [])
    let sourceRow = CVPixelBufferGetBytesPerRow(frame)
    let targetRow = CVPixelBufferGetBytesPerRow(packed)
    let source = CVPixelBufferGetBaseAddress(frame)!
    let target = CVPixelBufferGetBaseAddress(packed)!
    for y in 0..<height {
      memcpy(target + y * targetRow, source + (top + y) * sourceRow + left * 4, width * 4)
    }
    CVPixelBufferUnlockBaseAddress(packed, [])
    CVPixelBufferUnlockBaseAddress(frame, .readOnly)
    let yuv = try buffer(sourcePool)
    guard VTPixelTransferSessionTransferImage(transfer, from: packed, to: yuv) == noErr
    else { throw Self.failure("could not convert a frame to 4:2:0") }
    guard let processorFrame = VTFrameProcessorFrame(
      buffer: yuv, presentationTimeStamp: time(index))
    else { throw Self.failure("rejected a frame") }
    frames[index] = processorFrame
  }

  /// Filters frame `index` against the frames already added around it,
  /// releases the frames it no longer needs, and returns the rectangle as
  /// BGRA. A frame without a predecessor starts a new sequence.
  func filtered(_ index: Int) async throws -> CVPixelBuffer {
    guard let source = frames[index] else { throw Self.failure("is missing frame \(index)") }
    var previous: [VTFrameProcessorFrame] = []
    for offset in stride(from: previousCount, through: 1, by: -1) {
      if let frame = frames[index - offset] { previous.append(frame) }
    }
    var next: [VTFrameProcessorFrame] = []
    for offset in 1...max(1, nextCount) {
      guard let frame = frames[index + offset] else { break }
      next.append(frame)
    }
    let destination = try buffer(destinationPool)
    guard let destinationFrame = VTFrameProcessorFrame(
      buffer: destination, presentationTimeStamp: time(index)),
      let parameters = VTTemporalNoiseFilterParameters(
        sourceFrame: source, nextFrames: next, previousFrames: previous,
        destinationFrame: destinationFrame, filterStrength: strength,
        hasDiscontinuity: previous.isEmpty)
    else { throw Self.failure("rejected the parameters for frame \(index)") }
    try await processor.process(parameters: parameters)
    frames = frames.filter { $0.key > index - previousCount }
    let packed = try buffer(packedPool)
    guard VTPixelTransferSessionTransferImage(transfer, from: destination, to: packed) == noErr
    else { throw Self.failure("could not convert a filtered frame back to BGRA") }
    return packed
  }

  private func time(_ index: Int) -> CMTime {
    CMTime(value: CMTimeValue(index) * 1001, timescale: 30000)
  }

  private func buffer(_ pool: CVPixelBufferPool) throws -> CVPixelBuffer {
    var pixels: CVPixelBuffer?
    guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixels) == kCVReturnSuccess,
      let pixels
    else { throw Self.failure("could not allocate a frame") }
    return pixels
  }

  private static func pool(_ attributes: [String: Any]) throws -> CVPixelBufferPool {
    var pool: CVPixelBufferPool?
    guard CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
      == kCVReturnSuccess, let pool
    else { throw failure("could not create a frame pool") }
    return pool
  }

  private static func failure(_ message: String) -> NSError {
    NSError(domain: "SwiftVR", code: 17, userInfo: [
      NSLocalizedDescriptionKey: "SwiftVR temporal noise filter \(message)"
    ])
  }
}
