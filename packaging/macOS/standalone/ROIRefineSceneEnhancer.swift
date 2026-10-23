import CoreML
import Foundation
import VideoToolbox

/// A ROI refine pack: LTX-2.3 (Sulphur 2) with the ROI IC-LoRA, its 48 DiT blocks as int8 Core AI assets and its
/// video VAE in MLX, served by the pack's own `bin/ltx-video roi-refine-server`.
enum ROIRefinePack {
  static func isRoot(_ url: URL) -> Bool {
    FileManager.default.fileExists(atPath: url.appendingPathComponent("roi-refine-pack.json").path)
  }

  static func validate(_ root: URL) throws {
    for relative in ["bin/ltx-video", "bin/mlx-swift_Cmlx.bundle", "vae.safetensors", "blocks", "constants"] {
      guard FileManager.default.fileExists(atPath: root.appendingPathComponent(relative).path) else {
        throw NSError(domain: "ROIRefine", code: 1, userInfo: [
          NSLocalizedDescriptionKey: "ROI refine pack is missing \(relative): \(root.path)"
        ])
      }
    }
  }
}

/// Redraws whole ROI scenes (the 256px BasicVSR++ crops) at 512 or 1024 px, in 33-frame windows, in a resident
/// child process. Scenes run one at a time: the child holds about 15 GB, and a second copy would not run faster
/// on the same GPU. Takes about 17 s per 33 frames in a 512 window, about 70 s in a 1024 window.
final class ROIRefineSceneEnhancer: @unchecked Sendable {
  /// Refined frames come back at 512. Only with the 1024 canvas opted in does a crop over 512 source pixels come
  /// back at 1024 (drawn in the 1024 window); on a ~1000 px ROI that measured no more detail than 512 upscaled
  /// (the LoRA learned 256 -> 512) at ~6x the time, so by default large crops take the ROI Distill layer.
  func outputSide(cropSide: Int) -> Int { maximumWindowSide >= 1024 && cropSide > 512 ? 1024 : 512 }
  let strength: Float
  /// Apple temporal noise filter strength on the composited frames (0 = off), as for ROI Distill.
  let temporalFilter: Float
  /// Neighbouring refined frames blended in where BasicVSR++ matches (1 = off), as SwiftVR's なじませ範囲.
  let stabilizationRadius: Int
  /// Crop-view smoothing window in frames (0/1 = off), as SwiftVR's 枠のなめらかさ.
  let frameSmoothing: Int
  let framesPerSecond: Double
  /// ROI Distill layered on scenes whose largest crop side exceeds `distillMinimumCrop` source pixels.
  private let distill: DistilledROIEnhancer?
  let distillMinimumCrop: Int
  /// Largest internal LTX canvas requested from the server (512 by default, 1024 only by explicit opt-in).
  let maximumWindowSide: Int
  private let pack: URL
  private var process = Process()
  private var input: FileHandle
  private var lines: AsyncLineSequence<FileHandle.AsyncBytes>.AsyncIterator
  private var restarts = 0
  private let scratch: URL
  private let turn = ROIRefineTurn()
  private var sceneNumber = 0

  init(
    pack: URL, strength: Float, temporalFilter: Float = 0, stabilizationRadius: Int = 1, frameSmoothing: Int = 0,
    framesPerSecond: Double, distillModel: URL? = nil, distillMinimumCrop: Int = 512,
    maximumWindowSide: Int = 512
  ) async throws {
    try ROIRefinePack.validate(pack)
    self.distillMinimumCrop = distillMinimumCrop
    self.maximumWindowSide = maximumWindowSide >= 1024 ? 1024 : 512
    if let distillModel {
      let configuration = MLModelConfiguration()
      configuration.computeUnits = DistilledROIEnhancer.computeUnits
      distill = try DistilledROIEnhancer(model: MLModel(contentsOf: distillModel, configuration: configuration), scale: 2)
    } else {
      distill = nil
    }
    self.strength = max(0, min(1, strength))
    self.stabilizationRadius = max(1, min(8, stabilizationRadius))
    self.frameSmoothing = max(0, min(30, frameSmoothing))
    let requestedFilter = max(0, min(1, temporalFilter))
    if requestedFilter > 0, !VTTemporalNoiseFilterConfiguration.isSupported {
      FileHandle.standardError.write(Data(
        "ROI Refine: Apple temporal noise filter unavailable on this Mac; kept off\n".utf8))
      self.temporalFilter = 0
    } else {
      self.temporalFilter = requestedFilter
    }
    self.framesPerSecond = framesPerSecond
    scratch = FileManager.default.temporaryDirectory
      .appendingPathComponent("mioh-roi-refine-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    self.pack = pack
    (process, input, lines) = try Self.launch(pack)
    guard let ready = try await lines.next(), ready.hasPrefix("ready") else {
      process.terminate()
      throw NSError(domain: "ROIRefine", code: 2, userInfo: [
        NSLocalizedDescriptionKey: "ROI refine server did not start"
      ])
    }
    FileHandle.standardError.write(Data(
      "ROI Refine: server ready (\(ready)); Core AI DiT + MLX VAE, 33-frame windows, \(framesPerSecond) fps; internal canvas up to \(self.maximumWindowSide) px, Apple temporal noise filter \(self.temporalFilter), stabilization radius \(self.stabilizationRadius), frame smoothing \(self.frameSmoothing), ROI Distill layer \(distill == nil ? "off" : "on crops over \(distillMinimumCrop) px")\n".utf8))
  }

  private static func launch(_ pack: URL) throws
    -> (Process, FileHandle, AsyncLineSequence<FileHandle.AsyncBytes>.AsyncIterator)
  {
    let process = Process(), stdin = Pipe(), stdout = Pipe()
    process.executableURL = pack.appendingPathComponent("bin/ltx-video")
    process.arguments = ["roi-refine-server", "--pack", pack.path]
    process.standardInput = stdin
    process.standardOutput = stdout
    process.standardError = FileHandle.standardError
    // a dead server must surface as a write error, not SIGPIPE killing the export
    guard MacChildProcessPipe.prepare(stdin.fileHandleForWriting) else {
      throw NSError(domain: "ROIRefine", code: 7, userInfo: [
        NSLocalizedDescriptionKey: "ROI refine server pipe setup failed"
      ])
    }
    try process.run()
    ROIRefineChildMemory.register(process.processIdentifier)
    return (process, stdin.fileHandleForWriting, stdout.fileHandleForReading.bytes.lines.makeAsyncIterator())
  }

  /// Ends a server and waits for it to be gone, so a restart never holds two ~15 GB servers at once: closing
  /// stdin ends the request loop, SIGTERM follows after 5 s and SIGKILL after 10 s.
  private static func stop(_ process: Process, input: FileHandle) {
    try? input.close()
    let start = Date()
    var terminated = false
    while process.isRunning {
      let waited = Date().timeIntervalSince(start)
      if waited > 10 {
        kill(process.processIdentifier, SIGKILL)
      } else if waited > 5, !terminated {
        process.terminate()
        terminated = true
      }
      usleep(50_000)
    }
    ROIRefineChildMemory.unregister(process.processIdentifier)
  }

  /// A long export must not end because the server died: start a new one, and the current scene keeps
  /// BasicVSR++. Gives up (throws) after 5 restarts.
  private func restart(after reason: String) async throws {
    restarts += 1
    FileHandle.standardError.write(Data(
      "ROI Refine: server lost (\(reason)); restart \(restarts), this scene keeps BasicVSR++\n".utf8))
    guard restarts <= 5 else {
      throw NSError(domain: "ROIRefine", code: 6, userInfo: [
        NSLocalizedDescriptionKey: "ROI refine server failed \(restarts) times"
      ])
    }
    Self.stop(process, input: input)
    (process, input, lines) = try Self.launch(pack)
    guard let ready = try await lines.next(), ready.hasPrefix("ready") else {
      throw NSError(domain: "ROIRefine", code: 2, userInfo: [
        NSLocalizedDescriptionKey: "ROI refine server did not restart"
      ])
    }
  }

  /// `restored`: the scene's frames as planar RGB 256px FP16 in 0...1; `masks`: per frame 256x256 model masks.
  /// `aspect`: the true width / height the square crops show (crop width and height are scaled to 256
  /// separately); the server redraws them at that shape. Returns the scene's frames as planar RGB FP16 in 0...1
  /// at `outputSide(cropSide:)`, or nil when the server failed on it. `cropSide`: the scene's longest crop side in
  /// source pixels; it picks the server's window side.
  func refine(restored: [Float16], masks: [[Float]], frameCount: Int, aspect: Double, cropSide: Int)
    async throws -> [Float16]?
  {
    let outputSide = outputSide(cropSide: cropSide)
    let plane = 256 * 256
    guard frameCount > 0, restored.count >= frameCount * 3 * plane, masks.count == frameCount,
      masks.allSatisfy({ $0.count == plane })
    else {
      throw NSError(domain: "ROIRefine", code: 3, userInfo: [
        NSLocalizedDescriptionKey: "ROI refine scene input is incomplete"
      ])
    }
    await turn.acquire()
    defer { turn.release() }
    sceneNumber += 1
    let base = scratch.appendingPathComponent("scene-\(sceneNumber)")
    let restoredURL = base.appendingPathExtension("f16"), maskURL = base.appendingPathExtension("f32")
    let outputURL = base.appendingPathExtension("out.f16")
    defer {
      for url in [restoredURL, maskURL, outputURL] { try? FileManager.default.removeItem(at: url) }
    }
    try restored[0 ..< frameCount * 3 * plane].withUnsafeBytes { try Data($0).write(to: restoredURL) }
    try masks.flatMap { $0 }.withUnsafeBytes { try Data($0).write(to: maskURL) }
    let request: [String: Any] = [
      "restored": restoredURL.path, "masks": maskURL.path, "frames": frameCount,
      "fps": framesPerSecond, "output": outputURL.path, "aspect": aspect,
      "cropSide": cropSide, "outputSide": outputSide, "maximumWindowSide": maximumWindowSide,
    ]
    var line = try JSONSerialization.data(withJSONObject: request)
    line.append(0x0A)
    let reply: String
    do {
      try input.write(contentsOf: line)
      guard let next = try await lines.next() else {
        try await restart(after: "exited")
        return nil
      }
      reply = next
    } catch let error as NSError where error.domain != "ROIRefine" {
      try await restart(after: error.localizedDescription)
      return nil
    }
    guard reply.hasPrefix("ok") else {
      FileHandle.standardError.write(Data("ROI Refine: scene kept BasicVSR++ (\(reply))\n".utf8))
      return nil
    }
    let data = try Data(contentsOf: outputURL)
    let pixels: [Float16] = data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    let expected = frameCount * 3 * outputSide * outputSide
    guard pixels.count == expected else {
      throw NSError(domain: "ROIRefine", code: 5, userInfo: [
        NSLocalizedDescriptionKey: "ROI refine output has \(pixels.count) values, expected \(expected)"
      ])
    }
    FileHandle.standardError.write(Data("ROI Refine: \(frameCount)-frame scene \(reply)\n".utf8))
    return pixels
  }

  /// ROI Distill on top of refined 512px frames: each frame reduced to 256 (2x2 mean) is Distill's input; the
  /// frame gains Distill's output minus that reduction brought back up (bilinear), so the refined detail stays
  /// and Distill adds its own. Frames are planar RGB 512 in 0...1; `frameIndices` are the scene's batch indices.
  func layerDistill(_ frames: [[Float16]], frameIndices: [Int]) throws -> [[Float16]] {
    guard let distill, frames.allSatisfy({ $0.count == 3 * DistilledROIEnhancer.outputSide * DistilledROIEnhancer.outputSide })
    else { return frames }
    let side = DistilledROIEnhancer.outputSide, small = DistilledROIEnhancer.side, plane = side * side, smallPlane = small * small
    var reduced = [Float16](repeating: 0, count: frames.count * 3 * smallPlane)
    for (frame, pixels) in frames.enumerated() {
      for channel in 0..<3 {
        for y in 0..<small {
          for x in 0..<small {
            let top = channel * plane + 2 * y * side + 2 * x
            let mean = (Float(pixels[top]) + Float(pixels[top + 1]) + Float(pixels[top + side])
              + Float(pixels[top + side + 1])) / 4
            reduced[(frame * 3 + channel) * smallPlane + y * small + x] = Float16(mean)
          }
        }
      }
    }
    return try frames.indices.map { frame in
      let enhanced = try distill.enhance(restored: reduced, centre: frame, frameIndices: frameIndices)
      var output = frames[frame]
      for channel in 0..<3 {
        for y in 0..<side {
          for x in 0..<side {
            let index = channel * plane + y * side + x
            // bilinear, pixel centres aligned (a nearest step would leave a 2 px grid in the difference)
            let sy = max(0, min(Float(small - 1), (Float(y) + 0.5) / 2 - 0.5))
            let sx = max(0, min(Float(small - 1), (Float(x) + 0.5) / 2 - 0.5))
            let y0 = Int(sy), x0 = Int(sx), y1 = min(small - 1, y0 + 1), x1 = min(small - 1, x0 + 1)
            let wy = sy - Float(y0), wx = sx - Float(x0)
            let row = (frame * 3 + channel) * smallPlane
            func at(_ yy: Int, _ xx: Int) -> Float { Float(reduced[row + yy * small + xx]) }
            let base = (at(y0, x0) * (1 - wx) + at(y0, x1) * wx) * (1 - wy) + (at(y1, x0) * (1 - wx) + at(y1, x1) * wx) * wy
            output[index] = Float16(min(1, max(0, Float(output[index]) + Float(enhanced[index]) - base)))
          }
        }
      }
      return output
    }
  }

  func close() {
    Self.stop(process, input: input)
    try? FileManager.default.removeItem(at: scratch)
  }
}

/// The ROI Refine server is a separate process holding 10-19 GB (DiT weights, VAE, MLX arrays and GPU memory),
/// which the export's own memory figures do not see. Its pids are kept here so the diagnostics can add its
/// physical footprint.
enum ROIRefineChildMemory {
  private static let lock = NSLock()
  nonisolated(unsafe) private static var pids = Set<Int32>()

  static func register(_ pid: Int32) { lock.withLock { _ = pids.insert(pid) } }
  static func unregister(_ pid: Int32) { lock.withLock { _ = pids.remove(pid) } }

  /// Summed phys_footprint of the registered children in MiB, nil when there are none.
  static func footprintMiB() -> Double? {
    let current = lock.withLock { pids }
    guard !current.isEmpty else { return nil }
    var total: UInt64 = 0
    for pid in current {
      var info = rusage_info_v4()
      let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
          proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
        }
      }
      if status == 0 { total += info.ri_phys_footprint }
    }
    return Double(total) / 1_048_576
  }
}

/// Export lanes take turns on the one server.
private final class ROIRefineTurn: @unchecked Sendable {
  private let lock = NSLock()
  private var busy = false
  private var waiters: [CheckedContinuation<Void, Never>] = []

  func acquire() async {
    await withCheckedContinuation { (waiting: CheckedContinuation<Void, Never>) in
      let resumeNow = lock.withLock { () -> Bool in
        if busy { waiters.append(waiting); return false }
        busy = true
        return true
      }
      if resumeNow { waiting.resume() }
    }
  }

  func release() {
    let next = lock.withLock { () -> CheckedContinuation<Void, Never>? in
      if waiters.isEmpty { busy = false; return nil }
      return waiters.removeFirst()
    }
    next?.resume()
  }
}
