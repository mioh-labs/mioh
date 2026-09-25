// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

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

/// Persisted between the complete BasicVSR++ export and the optional SwiftVR
/// pass. The 256px planar FP16 pixels are not read back from the encoded MP4.
struct SwiftVRROIFrameRecord: Codable {
  let ptsNanoseconds: Int64
  let outputEligible: Bool?
  let inputFile: String
  let maskFile: String
  let cropLeft: Int
  let cropTop: Int
  let cropWidth: Int
  let cropHeight: Int
  let resizedWidth: Int
  let resizedHeight: Int
  let padLeft: Int
  let padTop: Int
}

struct SwiftVRROISceneRecord: Codable {
  let directory: String
  let frames: [SwiftVRROIFrameRecord]
}

struct SwiftVRROISidecarManifest: Codable {
  let version: Int
  let complete: Bool
  let restoredVideo: String
  let frameWidth: Int
  let frameHeight: Int
  let fpsNumerator: Int
  let fpsDenominator: Int
  let outputFramePTS: [Int64]
  let scenes: [SwiftVRROISceneRecord]
}

struct SwiftVRROIFrameInput {
  let ptsNanoseconds: Int64
  let outputEligible: Bool
  let pixels: [Float16]
  let mask: [Float]
  let cropLeft: Int
  let cropTop: Int
  let cropWidth: Int
  let cropHeight: Int
  let resizedWidth: Int
  let resizedHeight: Int
  let padLeft: Int
  let padTop: Int
}

final class SwiftVRROISidecarRecorder: @unchecked Sendable {
  let root: URL
  private let lock = NSLock()
  private var scenes: [SwiftVRROISceneRecord] = []
  private var outputFramePTS: [Int64] = []

  init(root: URL) throws {
    self.root = root.standardizedFileURL
    guard !FileManager.default.fileExists(atPath: self.root.path) else {
      throw NSError(domain: "SwiftVRROISidecar", code: 1, userInfo: [
        NSLocalizedDescriptionKey: "ROI sidecar already exists: \(self.root.path)"
      ])
    }
    try FileManager.default.createDirectory(
      at: self.root, withIntermediateDirectories: true
    )
  }

  func recordScene(_ frames: [SwiftVRROIFrameInput]) throws {
    guard !frames.isEmpty else { return }
    lock.lock()
    defer { lock.unlock() }
    let directoryName = String(format: "scene-%06d", scenes.count)
    let sceneDirectory = root.appendingPathComponent(directoryName,
      isDirectory: true)
    let input = sceneDirectory.appendingPathComponent("input", isDirectory: true)
    let masks = sceneDirectory.appendingPathComponent("masks", isDirectory: true)
    try FileManager.default.createDirectory(at: input,
      withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: masks,
      withIntermediateDirectories: true)
    var records: [SwiftVRROIFrameRecord] = []
    records.reserveCapacity(frames.count)
    for (index, frame) in frames.enumerated() {
      guard frame.pixels.count == 3 * 256 * 256,
        frame.cropWidth > 0, frame.cropHeight > 0,
        frame.mask.count == frame.cropWidth * frame.cropHeight else {
        throw NSError(domain: "SwiftVRROISidecar", code: 2, userInfo: [
          NSLocalizedDescriptionKey: "Invalid saved ROI frame geometry"
        ])
      }
      let name = String(format: "%04d.f16", index)
      let maskName = String(format: "%04d.f16", index)
      try frame.pixels.withUnsafeBytes { raw in
        try Data(raw).write(to: input.appendingPathComponent(name),
          options: .atomic)
      }
      let halfMask = frame.mask.map { Float16(max(0, min(1, $0))) }
      try halfMask.withUnsafeBytes { raw in
        try Data(raw).write(to: masks.appendingPathComponent(maskName),
          options: .atomic)
      }
      records.append(SwiftVRROIFrameRecord(
        ptsNanoseconds: frame.ptsNanoseconds,
        outputEligible: frame.outputEligible,
        inputFile: "input/\(name)",
        maskFile: "masks/\(maskName)",
        cropLeft: frame.cropLeft,
        cropTop: frame.cropTop,
        cropWidth: frame.cropWidth,
        cropHeight: frame.cropHeight,
        resizedWidth: frame.resizedWidth,
        resizedHeight: frame.resizedHeight,
        padLeft: frame.padLeft,
        padTop: frame.padTop
      ))
    }
    scenes.append(SwiftVRROISceneRecord(
      directory: directoryName, frames: records
    ))
  }

  func appendOutputFrames(_ timestamps: [Int64]) {
    lock.lock()
    outputFramePTS.append(contentsOf: timestamps)
    lock.unlock()
  }

  func finish(
    restoredVideo: URL, width: Int, height: Int,
    fpsNumerator: Int, fpsDenominator: Int
  ) throws -> SwiftVRROISidecarManifest {
    lock.lock()
    defer { lock.unlock() }
    guard FileManager.default.fileExists(atPath: restoredVideo.path),
      !outputFramePTS.isEmpty else {
      throw NSError(domain: "SwiftVRROISidecar", code: 3, userInfo: [
        NSLocalizedDescriptionKey: "Restored video is not complete"
      ])
    }
    let manifest = SwiftVRROISidecarManifest(
      version: 1, complete: true, restoredVideo: restoredVideo.path,
      frameWidth: width, frameHeight: height,
      fpsNumerator: fpsNumerator, fpsDenominator: fpsDenominator,
      outputFramePTS: outputFramePTS, scenes: scenes
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(manifest).write(
      to: root.appendingPathComponent("manifest.json"), options: .atomic
    )
    return manifest
  }
}
