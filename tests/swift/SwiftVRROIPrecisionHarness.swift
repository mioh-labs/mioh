// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0
// Opt-in integration benchmark: real ROI worker, synthetic input only.
import CryptoKit
import Foundation

@main
struct SwiftVRROIPrecisionHarness {
  static func main() async throws {
    let arguments = CommandLine.arguments
    guard (5...6).contains(arguments.count), let scale = Int(arguments[2]),
      let count = Int(arguments[4]), count > 0,
      ["fp16", "fp32", "alternate", "singleton"].contains(arguments[3])
    else {
      throw NSError(
        domain: "ROI benchmark", code: 1,
        userInfo: [
          NSLocalizedDescriptionKey:
            "usage: benchmark MODEL_ROOT SCALE fp16|fp32|alternate FRAMES [SCENES]"
        ])
    }
    let scenes = arguments.count == 6 ? Int(arguments[5]) ?? 0 : 1
    guard scenes > 0 else {
      throw NSError(
        domain: "ROI benchmark", code: 4,
        userInfo: [
          NSLocalizedDescriptionKey: "SCENES must be positive"
        ])
    }
    let enhancer: SwiftVRSceneEnhancer
    #if MIOH_ROI_PRECISION_BASELINE
      enhancer = try SwiftVRSceneEnhancer(
        model: URL(fileURLWithPath: arguments[1]), strength: 1, scale: scale,
        temporalFilter: 0, stabilizationRadius: 1, frameSmoothing: 0)
    #else
      if arguments[3] == "fp16" {
        // Exercise the production default, not just an explicitly enabled flag.
        enhancer = try SwiftVRSceneEnhancer(
          model: URL(fileURLWithPath: arguments[1]), strength: 1, scale: scale,
          temporalFilter: 0, stabilizationRadius: 1, frameSmoothing: 0)
      } else {
        enhancer = try SwiftVRSceneEnhancer(
          model: URL(fileURLWithPath: arguments[1]), strength: 1, scale: scale,
          temporalFilter: 0, stabilizationRadius: 1, frameSmoothing: 0,
          preserveDiTPrecision: false)
      }
    #endif
    if arguments[3] == "singleton" {
      precondition(count == 1)
      let input = [Float16](repeating: 0, count: 3 * 256 * 256)
      let output = try await enhancer.enhance(restored: input, frameCount: 1)
      precondition(output == nil, "singleton unexpectedly started SwiftVR")
      enhancer.close()
      FileHandle.standardOutput.write(Data("{\"skipped\":true,\"frames\":1}\n".utf8))
      return
    }
    let alternate =
      arguments[3] == "alternate"
      ? try SwiftVRSceneEnhancer(
        model: URL(fileURLWithPath: arguments[1]), strength: 1, scale: scale,
        temporalFilter: 0, stabilizationRadius: 1, frameSmoothing: 0)
      : nil
    defer { enhancer.close() }
    let plane = 256 * 256
    var input = [Float16](repeating: 0, count: count * 3 * plane)
    for frame in 0..<count {
      for channel in 0..<3 {
        for y in 0..<256 {
          for x in 0..<256 {
            let gradient = Float((x + y + channel * 47 + frame * 3) % 256) / 255
            let checker = ((x + frame * 2) / 16 + y / 16) % 2 == 0
            input[frame * 3 * plane + channel * plane + y * 256 + x] =
              Float16(gradient * 0.7 + (checker ? 0.25 : 0.05))
          }
        }
      }
    }
    var reference: [String]?
    var secondsPerScene: [Double] = []
    for scene in 0..<scenes {
      let start = ContinuousClock.now
      let current = scene % 2 == 1 ? alternate ?? enhancer : enhancer
      guard let output = try await current.enhance(restored: input, frameCount: count)
      else { throw CancellationError() }
      defer { output.cancel() }
      var digests: [String] = []
      for frame in 0..<count {
        guard let pixels = try await output.frame(frame),
          pixels.count == 3 * output.side * output.side,
          pixels.allSatisfy({ $0.isFinite })
        else {
          throw NSError(
            domain: "ROI benchmark", code: 2,
            userInfo: [
              NSLocalizedDescriptionKey: "missing, malformed, or non-finite frame \(frame)"
            ])
        }
        let digest = pixels.withUnsafeBytes { SHA256.hash(data: Data($0)) }
        digests.append(digest.map { String(format: "%02x", $0) }.joined())
        output.discard(before: frame + 1)
      }
      guard try await output.frame(count) == nil else {
        throw NSError(
          domain: "ROI benchmark", code: 3,
          userInfo: [
            NSLocalizedDescriptionKey: "unexpected extra output frame"
          ])
      }
      let duration = start.duration(to: .now).components
      let seconds = Double(duration.seconds) + Double(duration.attoseconds) / 1e18
      secondsPerScene.append(seconds)
      if let reference, reference != digests {
        throw NSError(
          domain: "ROI benchmark", code: 5,
          userInfo: [
            NSLocalizedDescriptionKey: "output changed between identical scenes"
          ])
      }
      reference = digests
    }
    let result: [String: Any] = [
      "scale": scale, "mode": arguments[3], "frames": count,
      "seconds": secondsPerScene.reduce(0, +), "scene_seconds": secondsPerScene,
      "frame_sha256": reference!,
    ]
    let json = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    FileHandle.standardOutput.write(json)
    FileHandle.standardOutput.write(Data("\n".utf8))
  }
}
