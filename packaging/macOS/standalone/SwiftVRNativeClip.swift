// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

// Native Swift/Core ML SwiftVR ROI clip worker. mioh launches this only for
// export-time scene-level enhancement; it does not replace BasicVSR++.
import CoreGraphics
import CoreML
import Foundation
import ImageIO
import UniformTypeIdentifiers

private enum ClipError: Error, CustomStringConvertible {
  case invalid(String)

  var description: String {
    switch self {
    case .invalid(let message): message
    }
  }
}

private let outputSize = 1024
private let inputSize = 256
private let latentSide = 64
// A scene runs many 24-frame chunks through the same DiT blocks. Compiling
// each package again for every chunk leaves the GPU idle for most of the job.
// Keep compiled URLs only for this worker's lifetime; the next scene gets a
// fresh worker, so this does not accumulate loaded model weights in memory.
private var compiledModels: [URL: URL] = [:]
// The t6 DiT stack is revisited for every 24-frame continuation. Retaining
// all 30 large models can exhaust unified memory, so pin only the first 12;
// the remaining layers still load on demand.
private var residentModels: [URL: MLModel] = [:]
private let sharedCompiledRoot = ProcessInfo.processInfo.environment[
  "MIOH_SWIFTVR_COMPILED_CACHE"].map {
    URL(fileURLWithPath: $0, isDirectory: true)
  }

private func compiledCacheName(for package: URL) -> String {
  // Swift's Hasher is process-randomized, so it cannot name a cache shared by
  // separate scene workers. The full standardized path is stable for this job.
  var hash: UInt64 = 14_695_981_039_346_656_037
  for byte in package.standardizedFileURL.path.utf8 {
    hash = (hash ^ UInt64(byte)) &* 1_099_511_628_211
  }
  return String(format: "%016llx.mlmodelc", hash)
}

private func shouldKeepResident(_ package: URL) -> Bool {
  guard package.deletingLastPathComponent().lastPathComponent
    == "native-4x-t6-fp16",
    package.lastPathComponent.hasPrefix("dit-block-") else { return false }
  let name = package.lastPathComponent
  let layer = Int(name.dropFirst("dit-block-".count).prefix(2)) ?? 30
  return layer < 12
}

private func array(_ shape: [Int]) throws -> MLMultiArray {
  try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
}

private func fromFile(_ url: URL, shape: [Int]) throws -> MLMultiArray {
  let data = try Data(contentsOf: url)
  let count = shape.reduce(1, *)
  guard data.count == count * MemoryLayout<Float>.size else {
    throw ClipError.invalid("Unexpected byte count: \(url.path)")
  }
  let result = try array(shape)
  data.withUnsafeBytes { source in
    if let base = source.baseAddress {
      result.dataPointer.copyMemory(from: base, byteCount: data.count)
    }
  }
  return result
}

private func checkedDataPointer(_ value: MLMultiArray) throws
  -> (UnsafeMutableRawPointer, MLMultiArrayDataType) {
  let dataType = value.dataType
  guard dataType == .float16 || dataType == .float32 || dataType == .double else {
    throw ClipError.invalid("Unsupported Core ML scalar type")
  }
  return (value.dataPointer, dataType)
}

@inline(__always)
private func readValue(_ pointer: UnsafeMutableRawPointer,
  type: MLMultiArrayDataType, at offset: Int) -> Float {
  switch type {
  case .float16:
    return Float(pointer.assumingMemoryBound(to: Float16.self)[offset])
  case .float32:
    return pointer.assumingMemoryBound(to: Float.self)[offset]
  case .double:
    return Float(pointer.assumingMemoryBound(to: Double.self)[offset])
  default:
    preconditionFailure("Scalar type was checked before entering the pixel loop")
  }
}

private func offset(_ linear: Int, shape: [Int], strides: [Int]) -> Int {
  var remaining = linear
  var result = 0
  for axis in stride(from: shape.count - 1, through: 0, by: -1) {
    let index = remaining % shape[axis]
    remaining /= shape[axis]
    result += index * strides[axis]
  }
  return result
}

private func contiguous(_ value: MLMultiArray, shape: [Int]) throws -> MLMultiArray {
  guard value.shape.map(\.intValue) == shape else {
    throw ClipError.invalid("Unexpected Core ML shape \(value.shape)")
  }
  let result = try array(shape)
  let target = result.dataPointer.assumingMemoryBound(to: Float.self)
  let strides = value.strides.map(\.intValue)
  let (source, dataType) = try checkedDataPointer(value)
  let count = shape.reduce(1, *)
  var expectedStride = 1
  var isContiguous = true
  for axis in shape.indices.reversed() {
    if strides[axis] != expectedStride { isContiguous = false }
    expectedStride *= shape[axis]
  }
  if isContiguous, dataType == .float32 {
    target.update(from: source.assumingMemoryBound(to: Float.self), count: count)
  } else if isContiguous {
    for linear in 0..<count {
      target[linear] = readValue(source, type: dataType, at: linear)
    }
  } else {
    for linear in 0..<count {
      target[linear] = readValue(source, type: dataType,
        at: offset(linear, shape: shape, strides: strides))
    }
  }
  return result
}

private func predictFeatures(
  _ package: URL, values: [String: Any]
) throws -> MLFeatureProvider {
  let compiled: URL
  if let cached = compiledModels[package] {
    compiled = cached
  } else if let sharedCompiledRoot {
    try FileManager.default.createDirectory(at: sharedCompiledRoot,
      withIntermediateDirectories: true)
    let cached = sharedCompiledRoot.appendingPathComponent(
      compiledCacheName(for: package))
    if !FileManager.default.fileExists(atPath: cached.path) {
      let temporary = try MLModel.compileModel(at: package)
      try FileManager.default.moveItem(at: temporary, to: cached)
    }
    compiledModels[package] = cached
    compiled = cached
  } else {
    compiled = try MLModel.compileModel(at: package)
    compiledModels[package] = compiled
  }
  let model: MLModel
  if let resident = residentModels[package] {
    model = resident
  } else {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    model = try MLModel(contentsOf: compiled, configuration: configuration)
    if shouldKeepResident(package) { residentModels[package] = model }
  }
  return try model.prediction(
    from: MLDictionaryFeatureProvider(dictionary: values)
  )
}

private func feature(
  _ features: MLFeatureProvider, name: String, source: URL
) throws -> MLMultiArray {
  guard let result = features.featureValue(for: name)?.multiArrayValue else {
    throw ClipError.invalid("Missing \(name) from \(source.lastPathComponent)")
  }
  return result
}

private func predict(
  _ package: URL, values: [String: Any], output: String
) throws -> MLMultiArray {
  try feature(predictFeatures(package, values: values), name: output, source: package)
}

private func followingStates(
  _ features: MLFeatureProvider, source: URL
) throws -> [String: Any] {
  var result: [String: Any] = [:]
  for index in 0..<9 {
    result["state_\(index)"] = try feature(
      features, name: "next_state_\(index)", source: source
    )
  }
  return result
}

private func imagePixels(_ url: URL) throws -> [UInt8] {
  guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
    let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
    image.width == inputSize, image.height == inputSize
  else { throw ClipError.invalid("Expected a 256x256 PNG: \(url.path)") }
  var rgba = [UInt8](repeating: 0, count: inputSize * inputSize * 4)
  let drawn = rgba.withUnsafeMutableBytes { raw -> Bool in
    guard let base = raw.baseAddress,
      let context = CGContext(
        data: base, width: inputSize, height: inputSize,
        bitsPerComponent: 8, bytesPerRow: inputSize * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      )
    else { return false }
    // For a CGImage drawn directly into this bitmap context, row zero already
    // matches the top PNG row. Flipping it here inverted the test clip.
    context.draw(image, in: CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
    return true
  }
  guard drawn else { throw ClipError.invalid("Could not decode \(url.path)") }
  return rgba
}

private func inputFrames(
  _ folder: URL, start: Int, validCount: Int, paddedCount: Int
) throws -> MLMultiArray {
  let result = try array([1, paddedCount, 3, outputSize, outputSize])
  let destination = result.dataPointer.assumingMemoryBound(to: Float.self)
  let allFiles = try FileManager.default.contentsOfDirectory(
    at: folder, includingPropertiesForKeys: nil
  )
  let rawFiles = allFiles.filter { $0.pathExtension.lowercased() == "f16" }
  let usesRawInput = !rawFiles.isEmpty
  let files = (usesRawInput ? rawFiles : allFiles.filter {
    $0.pathExtension.lowercased() == "png"
  })
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
  guard validCount > 0, paddedCount >= validCount,
    !files.isEmpty, start <= files.count + 3 else {
    throw ClipError.invalid("Input PNG range is unavailable")
  }
  let plane = outputSize * outputSize
  for frame in 0..<paddedCount {
    let frameURL = files[min(start + min(frame, validCount - 1), files.count - 1)]
    let pixels = usesRawInput ? nil : try imagePixels(frameURL)
    let rawPixels: [Float16]?
    if usesRawInput {
      let data = try Data(contentsOf: frameURL)
      guard data.count == 3 * inputSize * inputSize * MemoryLayout<Float16>.size else {
        throw ClipError.invalid("Invalid planar FP16 input: \(frameURL.path)")
      }
      rawPixels = data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    } else {
      rawPixels = nil
    }
    for y in 0..<outputSize {
      let sourceY = max(0, min(Float(inputSize - 1),
        (Float(y) + 0.5) * Float(inputSize) / Float(outputSize) - 0.5))
      let y0 = Int(sourceY)
      let y1 = min(inputSize - 1, y0 + 1)
      let wy = sourceY - Float(y0)
      for x in 0..<outputSize {
        let sourceX = max(0, min(Float(inputSize - 1),
          (Float(x) + 0.5) * Float(inputSize) / Float(outputSize) - 0.5))
        let x0 = Int(sourceX)
        let x1 = min(inputSize - 1, x0 + 1)
        let wx = sourceX - Float(x0)
        for channel in 0..<3 {
          let a: Float
          let b: Float
          let c: Float
          let d: Float
          if let rawPixels {
            let base = channel * inputSize * inputSize
            a = Float(rawPixels[base + y0 * inputSize + x0])
            b = Float(rawPixels[base + y0 * inputSize + x1])
            c = Float(rawPixels[base + y1 * inputSize + x0])
            d = Float(rawPixels[base + y1 * inputSize + x1])
          } else if let pixels {
            a = Float(pixels[(y0 * inputSize + x0) * 4 + channel]) / 255
            b = Float(pixels[(y0 * inputSize + x1) * 4 + channel]) / 255
            c = Float(pixels[(y1 * inputSize + x0) * 4 + channel]) / 255
            d = Float(pixels[(y1 * inputSize + x1) * 4 + channel]) / 255
          } else {
            throw ClipError.invalid("Missing input frame pixels")
          }
          let upper = a * (1 - wx) + b * wx
          let lower = c * (1 - wx) + d * wx
          destination[(frame * 3 + channel) * plane + y * outputSize + x] =
            upper * (1 - wy) + lower * wy
        }
      }
    }
    print("Prepared frame \(frame)")
  }
  return result
}

private func zeroStates(_ sizes: [[Int]]) throws -> [String: Any] {
  var result: [String: Any] = [:]
  for (index, shape) in sizes.enumerated() {
    let value = try array([1] + shape)
    memset(
      value.dataPointer, 0,
      shape.reduce(1, *) * MemoryLayout<Float>.size
    )
    result["state_\(index)"] = value
  }
  return result
}

private func rotary(
  _ components: URL, axis: String, temporalOffset: Int,
  latentCount: Int
) throws -> MLMultiArray {
  guard temporalOffset >= 0, temporalOffset + latentCount <= 1024 else {
    throw ClipError.invalid("Temporal RoPE table is too short")
  }
  let path = components.appendingPathComponent("rope-\(axis).f32")
  let table = try fromFile(path, shape: [1024, 128])
  let source = table.dataPointer.assumingMemoryBound(to: Float.self)
  let result = try array([1, latentCount * 32 * 32, 1, 128])
  let target = result.dataPointer.assumingMemoryBound(to: Float.self)
  for time in 0..<latentCount {
    for y in 0..<32 {
      for x in 0..<32 {
        let start = ((time * 32 + y) * 32 + x) * 128
        for dimension in 0..<44 { target[start + dimension] = source[(time + temporalOffset) * 128 + dimension] }
        for dimension in 0..<42 { target[start + 44 + dimension] = source[y * 128 + 44 + dimension] }
        for dimension in 0..<42 { target[start + 86 + dimension] = source[x * 128 + 86 + dimension] }
      }
    }
  }
  return result
}

private func savePNG(
  _ frames: MLMultiArray, decoderFrame: Int,
  outputFrame: Int, latentCount: Int, folder: URL
) throws {
  let shape = [1, latentCount * 4, 3, outputSize, outputSize]
  guard frames.shape.map(\.intValue) == shape else {
    throw ClipError.invalid("Unexpected decoded frame shape")
  }
  let strides = frames.strides.map(\.intValue)
  let (source, dataType) = try checkedDataPointer(frames)
  let plane = outputSize * outputSize
  var rgba = [UInt8](repeating: 255, count: plane * 4)
  for pixel in 0..<plane {
    for channel in 0..<3 {
      let sourceOffset = decoderFrame * strides[1] + channel * strides[2]
        + (pixel / outputSize) * strides[3] + (pixel % outputSize) * strides[4]
      let value = readValue(source, type: dataType, at: sourceOffset)
      rgba[pixel * 4 + channel] = UInt8(
        max(0, min(255, Int((max(0, min(1, value)) * 255).rounded())))
      )
    }
  }
  let data = Data(rgba) as CFData
  guard let provider = CGDataProvider(data: data),
    let image = CGImage(
      width: outputSize, height: outputSize, bitsPerComponent: 8,
      bitsPerPixel: 32, bytesPerRow: outputSize * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
      provider: provider, decode: nil,
      shouldInterpolate: false, intent: .defaultIntent
    )
  else { throw ClipError.invalid("Could not construct PNG frame") }
  let url = folder.appendingPathComponent(String(format: "%04d.png", outputFrame))
  guard let destination = CGImageDestinationCreateWithURL(
    url as CFURL, UTType.png.identifier as CFString, 1, nil
  ) else { throw ClipError.invalid("Could not create \(url.path)") }
  CGImageDestinationAddImage(destination, image, nil)
  guard CGImageDestinationFinalize(destination) else {
    throw ClipError.invalid("Could not write \(url.path)")
  }
}

private func saveRaw(
  _ frames: MLMultiArray, decoderFrame: Int,
  outputFrame: Int, latentCount: Int, folder: URL
) throws {
  let shape = [1, latentCount * 4, 3, outputSize, outputSize]
  guard frames.shape.map(\.intValue) == shape else {
    throw ClipError.invalid("Unexpected decoded frame shape")
  }
  let strides = frames.strides.map(\.intValue)
  let (source, dataType) = try checkedDataPointer(frames)
  let plane = outputSize * outputSize
  var planar = [Float16](repeating: 0, count: plane * 3)
  for channel in 0..<3 {
    for pixel in 0..<plane {
      let sourceOffset = decoderFrame * strides[1] + channel * strides[2]
        + (pixel / outputSize) * strides[3] + (pixel % outputSize) * strides[4]
      planar[channel * plane + pixel] = Float16(
        max(0, min(1, readValue(source, type: dataType, at: sourceOffset)))
      )
    }
  }
  let url = folder.appendingPathComponent(String(format: "%04d.f16", outputFrame))
  try planar.withUnsafeBytes { try Data($0).write(to: url, options: .atomic) }
}

private func trailingLatents(
  _ latents: MLMultiArray, count: Int, trailing: Int
) throws -> MLMultiArray {
  guard (1...count).contains(trailing) else {
    throw ClipError.invalid("Invalid preceding latent count")
  }
  let source = try contiguous(latents, shape: [count, 48, 64, 64])
  let result = try array([trailing, 48, 64, 64])
  let elements = 48 * 64 * 64
  memcpy(
    result.dataPointer,
    source.dataPointer.advanced(by: (count - trailing) * elements * MemoryLayout<Float>.size),
    trailing * elements * MemoryLayout<Float>.size
  )
  return result
}

private func denoise(
  latents: MLMultiArray, encodedCount: Int, validCount: Int,
  previous: MLMultiArray?, temporalOffset: Int,
  assetRoot: URL, started: Date
) throws -> MLMultiArray {
  guard [6, 7].contains(encodedCount), (1...encodedCount).contains(validCount) else {
    throw ClipError.invalid("Invalid encoded or valid latent count")
  }
  let ditCount = encodedCount == 7 || previous != nil ? 7 : 6
  let padding = ditCount - validCount
  let raw = try contiguous(latents, shape: [encodedCount, 48, 64, 64])
  let rawValues = raw.dataPointer.assumingMemoryBound(to: Float.self)
  let previousValues = previous?.dataPointer.assumingMemoryBound(to: Float.self)
  let ditInput = try array([1, 48, ditCount, 64, 64])
  let ditValues = ditInput.dataPointer.assumingMemoryBound(to: Float.self)
  let spatial = latentSide * latentSide
  for time in 0..<ditCount {
    for channel in 0..<48 {
      for position in 0..<spatial {
        let target = (channel * ditCount + time) * spatial + position
        if time < padding {
          ditValues[target] = previousValues?[
            (time * 48 + channel) * spatial + position
          ] ?? 0
        } else {
          ditValues[target] = rawValues[
            ((time - padding) * 48 + channel) * spatial + position
          ]
        }
      }
    }
  }
  let shapeName = ditCount == 6 ? "t6" : "t7"
  let variant = assetRoot.appendingPathComponent("native-4x-\(shapeName)-fp16")
  let components = variant.appendingPathComponent("components")
  let tokens = ditCount * 32 * 32
  var hidden = try contiguous(
    predict(components.appendingPathComponent("patch.mlpackage"),
      values: ["latents": ditInput], output: "output"),
    shape: [1, tokens, 3072]
  )
  let conditions: [String: Any] = [
    "context": try fromFile(components.appendingPathComponent("context.f32"), shape: [1, 512, 3072]),
    "modulation": try fromFile(components.appendingPathComponent("modulation.f32"), shape: [1, 6, 3072]),
    "cosine": try rotary(components, axis: "cosine", temporalOffset: temporalOffset,
      latentCount: ditCount),
    "sine": try rotary(components, axis: "sine", temporalOffset: temporalOffset,
      latentCount: ditCount),
  ]
  for layer in 0..<30 {
    let name = String(format: "dit-block-%02d-%@-4x-float16.mlpackage", layer, shapeName)
    var inputs = conditions
    inputs["hidden"] = hidden
    hidden = try contiguous(
      predict(variant.appendingPathComponent(name), values: inputs, output: "output"),
      shape: [1, tokens, 3072]
    )
    print("DiT \(layer + 1)/30 elapsed=\(Date().timeIntervalSince(started))s")
  }
  let velocity = try contiguous(
    predict(components.appendingPathComponent("head.mlpackage"),
      values: ["hidden": hidden], output: "output"),
    shape: [1, 48, ditCount, 64, 64]
  )
  let velocityValues = velocity.dataPointer.assumingMemoryBound(to: Float.self)
  let decodedInput = try array([1, encodedCount, 48, 64, 64])
  let decodedValues = decodedInput.dataPointer.assumingMemoryBound(to: Float.self)
  for time in 0..<encodedCount {
    for channel in 0..<48 {
      for position in 0..<spatial {
        let index = (time * 48 + channel) * spatial + position
        let validTime = min(time, validCount - 1)
        let rawIndex = (validTime * 48 + channel) * spatial + position
        let ditIndex = (channel * ditCount + validTime + padding) * spatial + position
        decodedValues[index] = rawValues[rawIndex] - velocityValues[ditIndex]
      }
    }
  }
  return decodedInput
}

@main
private enum SwiftVRNativeClipRunner {
  static func main() throws {
    defer {
      residentModels.removeAll()
      if sharedCompiledRoot == nil {
        for compiled in compiledModels.values {
          try? FileManager.default.removeItem(at: compiled)
        }
      }
      compiledModels.removeAll()
    }
    guard [4, 5].contains(CommandLine.arguments.count) else {
      throw ClipError.invalid("usage: swiftvr-native-clip <asset-root> <input-png-directory> <output-png-directory> [frame-count]")
    }
    let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let input = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
    let output = URL(fileURLWithPath: CommandLine.arguments[3], isDirectory: true)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let inputs = try FileManager.default.contentsOfDirectory(
      at: input, includingPropertiesForKeys: nil
    )
    let rawOutput = inputs.contains { $0.pathExtension.lowercased() == "f16" }
    let availableFrames = inputs.filter {
      $0.pathExtension.lowercased() == (rawOutput ? "f16" : "png")
    }.count
    let totalFrames: Int
    if CommandLine.arguments.count == 5 {
      guard let requested = Int(CommandLine.arguments[4]),
        requested > 0, requested <= availableFrames else {
        throw ClipError.invalid("frame-count must be between 1 and \(availableFrames)")
      }
      totalFrames = requested
    } else {
      totalFrames = availableFrames
    }
    guard totalFrames > 0 else {
      throw ClipError.invalid("Input directory has no PNG frames")
    }
    let started = Date()
    let firstCount = min(totalFrames, 28)
    let frames = try inputFrames(
      input, start: 0, validCount: firstCount, paddedCount: 28
    )
    var encoderInputs = try zeroStates(
      Array(repeating: [64, 256, 256], count: 3)
        + Array(repeating: [64, 128, 128], count: 3)
        + Array(repeating: [64, 64, 64], count: 3)
    )
    encoderInputs["frames"] = frames
    let firstEncoderURL = root.appendingPathComponent(
      "reae-stateful-encoder-28f-1024-fp32.mlpackage"
    )
    let firstEncoded = try predictFeatures(firstEncoderURL, values: encoderInputs)
    let firstLatents = try feature(
      firstEncoded, name: "latents", source: firstEncoderURL
    )
    print("Encoded 28 frames in \(Date().timeIntervalSince(started))s")
    let firstDenoised = try denoise(
      latents: firstLatents, encodedCount: 7,
      validCount: min(7, (max(totalFrames, 1) - 1 + 3) / 4 + 1),
      previous: nil, temporalOffset: 0,
      assetRoot: root, started: started
    )
    var decoderInputs = try zeroStates(
      Array(repeating: [512, 64, 64], count: 3)
        + Array(repeating: [256, 128, 128], count: 3)
        + Array(repeating: [128, 256, 256], count: 3)
    )
    decoderInputs["latents"] = firstDenoised
    let firstDecoderURL = root.appendingPathComponent(
      "reae-stateful-decoder-7latent-1024-fp32.mlpackage"
    )
    let firstDecoded = try predictFeatures(firstDecoderURL, values: decoderInputs)
    let firstOutput = try feature(firstDecoded, name: "frames", source: firstDecoderURL)
    let firstOutputCount = min(totalFrames, 25)
    for index in 0..<firstOutputCount {
      if rawOutput {
        try saveRaw(firstOutput, decoderFrame: index + 3,
          outputFrame: index, latentCount: 7, folder: output)
      } else {
        try savePNG(firstOutput, decoderFrame: index + 3,
          outputFrame: index, latentCount: 7, folder: output)
      }
    }
    let paddedTotal = ((totalFrames - 1 + 3) / 4) * 4 + 1
    var nextInput = 28
    var nextOutput = 25
    var encoded = firstEncoded
    var encodedURL = firstEncoderURL
    var decoded = firstDecoded
    var decodedURL = firstDecoderURL
    var rawLatents = firstLatents
    var rawLatentCount = 7
    var latentOffset = 7
    while nextOutput < totalFrames {
      let remaining = paddedTotal - nextInput
      let isLast = remaining <= 24
      let chunkFrames = min(remaining, 24)
      let availableFrames = max(1, min(totalFrames - nextInput, chunkFrames))
      let continuationFrames = try inputFrames(
        input, start: nextInput, validCount: availableFrames, paddedCount: 24
      )
      var encoderInputs = try followingStates(encoded, source: encodedURL)
      encoderInputs["frames"] = continuationFrames
      let encoderURL = root.appendingPathComponent(
        "reae-stateful-encoder-24f-1024-fp32.mlpackage"
      )
      let nextEncoded = try predictFeatures(encoderURL, values: encoderInputs)
      let nextLatents = try feature(
        nextEncoded, name: "latents", source: encoderURL
      )
      let validLatents = isLast ? (chunkFrames - 1) / 4 + 1 : 6
      let padLatents = isLast ? 7 - validLatents : 0
      let precedingLatents = padLatents > 0
        ? try trailingLatents(rawLatents, count: rawLatentCount,
          trailing: padLatents)
        : nil
      // MIDDLE uses the six-latent graph without overlap. LAST prepends the
      // previous raw latents (not denoised latents) to the seven-latent graph.
      let nextDenoised = try denoise(
        latents: nextLatents, encodedCount: 6, validCount: validLatents,
        previous: precedingLatents,
        temporalOffset: max(0, latentOffset - padLatents),
        assetRoot: root, started: started
      )
      var decoderInputs = try followingStates(decoded, source: decodedURL)
      decoderInputs["latents"] = nextDenoised
      let decoderURL = root.appendingPathComponent(
        "reae-stateful-decoder-6latent-1024-fp32.mlpackage"
      )
      let nextDecoded = try predictFeatures(decoderURL, values: decoderInputs)
      let decodedFrames = try feature(nextDecoded, name: "frames", source: decoderURL)
      let outputCount = min(totalFrames - nextOutput, 24)
      for index in 0..<outputCount {
        if rawOutput {
          try saveRaw(decodedFrames, decoderFrame: index,
            outputFrame: nextOutput + index, latentCount: 6, folder: output)
        } else {
          try savePNG(decodedFrames, decoderFrame: index,
            outputFrame: nextOutput + index, latentCount: 6, folder: output)
        }
      }
      nextInput += chunkFrames
      nextOutput += outputCount
      latentOffset += validLatents
      encoded = nextEncoded
      encodedURL = encoderURL
      decoded = nextDecoded
      decodedURL = decoderURL
      rawLatents = nextLatents
      rawLatentCount = 6
    }
    print("Swift/Core ML clip complete in \(Date().timeIntervalSince(started))s: \(output.path)")
  }
}
