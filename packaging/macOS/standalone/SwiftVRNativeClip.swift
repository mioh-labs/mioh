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

private let inputSize = 256

/// Sizes and asset names for one enlargement factor. 4x turns the 256px
/// restoration into 1024px; 2x into 512px with a quarter of the DiT tokens.
private struct Geometry {
  let scale: Int

  init(scale: Int) throws {
    guard [2, 4].contains(scale) else {
      throw ClipError.invalid("scale must be 2 or 4")
    }
    self.scale = scale
  }

  var outputSize: Int { inputSize * scale }
  var latentSide: Int { outputSize / 16 }
  /// DiT tokens per latent frame along each axis (2x2 patches).
  var tokenSide: Int { latentSide / 2 }
  var packName: String { "\(scale)x" }
  func pack(_ shapeName: String) -> String { "native-\(packName)-\(shapeName)-fp16" }
  func encoder(frames: Int) -> String {
    "reae-stateful-encoder-\(frames)f-\(outputSize)-fp32.mlpackage"
  }
  func decoder(latents: Int) -> String {
    "reae-stateful-decoder-\(latents)latent-\(outputSize)-fp32.mlpackage"
  }
}
// Compiling a package takes about ten times longer than running it, and
// nothing uses the GPU meanwhile. Compiled models therefore persist across
// exports in the shared cache the postprocess supplies.
private var compiledModels: [URL: URL] = [:]
// Models are loaded for each use and released afterwards. A loaded DiT block
// holds far more than its 312 MB of weights (GPU buffers for 7,168 tokens):
// keeping the stack resident reached 42–63 GB and swapped, while loading on
// demand ran a warm 49-frame scene in 16.7 s, as fast as keeping 12 resident.
private let sharedCompiledRoot = ProcessInfo.processInfo.environment[
  "MIOH_SWIFTVR_COMPILED_CACHE"].map {
    URL(fileURLWithPath: $0, isDirectory: true)
  }

private func packageFiles(_ package: URL) -> [URL] {
  let data = package.appendingPathComponent("Data/com.apple.CoreML")
  return [
    data.appendingPathComponent("model.mlmodel"),
    data.appendingPathComponent("weights/weight.bin"),
  ]
}

private func compiledCacheName(for package: URL) -> String {
  // Swift's Hasher is process-randomized, so it cannot name a cache shared by
  // separate workers. The key covers the path and the size and modification
  // time of the package contents, so a replaced model pack is recompiled.
  var identity = package.standardizedFileURL.path
  for file in packageFiles(package) {
    let values = try? file.resourceValues(
      forKeys: [.fileSizeKey, .contentModificationDateKey])
    identity += "|\(values?.fileSize ?? -1)"
      + "|\(values?.contentModificationDate?.timeIntervalSince1970 ?? -1)"
  }
  var hash: UInt64 = 14_695_981_039_346_656_037
  for byte in identity.utf8 {
    hash = (hash ^ UInt64(byte)) &* 1_099_511_628_211
  }
  return String(format: "%016llx.mlmodelc", hash)
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

private func compiledURL(for package: URL) throws -> URL {
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
      do {
        try FileManager.default.moveItem(at: temporary, to: cached)
      } catch where FileManager.default.fileExists(atPath: cached.path) {
        // Another export compiled the same package first.
        try? FileManager.default.removeItem(at: temporary)
      }
    }
    // The postprocess prunes entries that no export has used for a while.
    try? FileManager.default.setAttributes(
      [.modificationDate: Date()], ofItemAtPath: cached.path)
    compiledModels[package] = cached
    compiled = cached
  } else {
    compiled = try MLModel.compileModel(at: package)
    compiledModels[package] = compiled
  }
  return compiled
}

private func loadModel(compiled: URL, functionName: String? = nil) throws -> MLModel {
  let configuration = MLModelConfiguration()
  configuration.computeUnits = .all
  configuration.functionName = functionName
  return try MLModel(contentsOf: compiled, configuration: configuration)
}

/// ReAE graphs are loaded for every chunk on purpose. Their 1024px GPU working
/// buffers are released with the model; keeping the four encoders/decoders
/// resident pushed a 180-frame scene to a 49.8 GB footprint and 3x the time.
private func predictFeatures(
  _ package: URL, values: [String: Any]
) throws -> MLFeatureProvider {
  try predictFeatures(loadModel(compiled: compiledURL(for: package)), values: values)
}

private func predictFeatures(
  _ model: MLModel, values: [String: Any]
) throws -> MLFeatureProvider {
  try model.prediction(from: MLDictionaryFeatureProvider(dictionary: values))
}

/// When the model pack provides the DiT stack as a few multi-layer groups,
/// the groups load once per worker and stay loaded. A group shares one set of
/// GPU working buffers across its layers, so a whole stack fits in memory
/// where 30 separate resident blocks swapped.
///
/// Preferred layout: `native-<scale>x-fp16-grouped/dit-group-AA-BB-<scale>x-float16.mlpackage`,
/// each a multifunction package with a `t7` and a `t6` function. The shapes
/// share their weights on disk and in memory (a t6 function added 0.48 GB to
/// a group, a separate package about 1.5 GB). Export clips reach 180 frames,
/// and their middle chunks need t6. Older packs carry a t7-only stack in
/// `native-4x-t7-fp16-grouped`.
private var residentGroups: [String: [MLModel]] = [:]

private func groupedStack(
  _ assetRoot: URL, shapeName: String, geometry: Geometry
) throws -> [MLModel]? {
  let shared = assetRoot.appendingPathComponent(
    "native-\(geometry.packName)-fp16-grouped", isDirectory: true)
  let single = assetRoot.appendingPathComponent(
    "\(geometry.pack(shapeName))-grouped", isDirectory: true)
  for (directory, functionName) in [(shared, shapeName as String?), (single, nil)] {
    let key = "\(directory.path)#\(shapeName)"
    if let resident = residentGroups[key] { return resident }
    guard let names = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
      .filter({ $0.hasPrefix("dit-group-") && $0.hasSuffix(".mlpackage") })
      .sorted(),
      !names.isEmpty
    else { continue }
    var nextLayer = 0
    for name in names {
      let parts = name.split(separator: "-")
      guard parts.count > 3, let first = Int(parts[2]), let last = Int(parts[3]),
        first == nextLayer, last >= first
      else { throw ClipError.invalid("DiT groups must cover layers contiguously: \(name)") }
      nextLayer = last + 1
    }
    guard nextLayer == 30 else {
      throw ClipError.invalid("DiT groups cover \(nextLayer) of 30 layers")
    }
    let models = try names.map { name in
      try pooled {
        try loadModel(
          compiled: compiledURL(for: directory.appendingPathComponent(name)),
          functionName: functionName)
      }
    }
    residentGroups[key] = models
    return models
  }
  return nil
}

/// Loads a model on another thread. Loading a DiT block takes longer than
/// running it and uses the CPU, so the next block loads while the current one
/// runs on the GPU. At most two blocks are alive at once.
private final class PendingModel: @unchecked Sendable {
  private let finished = DispatchSemaphore(value: 0)
  private var result: Result<MLModel, Error>?

  init(compiled: URL) {
    DispatchQueue.global(qos: .userInitiated).async {
      self.result = Result { try autoreleasepool { try loadModel(compiled: compiled) } }
      self.finished.signal()
    }
  }

  func wait() throws -> MLModel {
    finished.wait()
    return try result!.get()
  }
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

/// Core ML returns its inputs, outputs and loaded models autoreleased.
/// This worker has no run loop, so without explicit pools nothing is freed
/// until exit: one 30-block chunk alone leaves several GB behind, and a
/// worker serving many scenes grows until the system swaps.
private func pooled<Result>(_ body: () throws -> Result) rethrows -> Result {
  try autoreleasepool(invoking: body)
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
  _ folder: URL, start: Int, validCount: Int, paddedCount: Int, geometry: Geometry
) throws -> MLMultiArray {
  let outputSize = geometry.outputSize
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
  let inputPlane = inputSize * inputSize
  // Decode every frame to planar Float, then upscale 2x or 4x. Frames are
  // independent and write disjoint output planes, so both steps run in parallel.
  var sources = [[Float]](repeating: [], count: paddedCount)
  var readFailures = [Error?](repeating: nil, count: paddedCount)
  sources.withUnsafeMutableBufferPointer { output in
    readFailures.withUnsafeMutableBufferPointer { failure in
      DispatchQueue.concurrentPerform(iterations: paddedCount) { frame in
        do {
          let frameURL = files[min(start + min(frame, validCount - 1), files.count - 1)]
          output[frame] = try planarSource(frameURL, raw: usesRawInput)
        } catch {
          failure[frame] = error
        }
      }
    }
  }
  if let failure = readFailures.lazy.compactMap({ $0 }).first { throw failure }
  // Half-pixel-centred bilinear taps are identical for every row, column,
  // channel and frame, so compute them once. A Metal version was measured
  // slower (0.06-0.08 s vs 0.03 s per chunk) because of buffer setup.
  let result = try array([1, paddedCount, 3, outputSize, outputSize])
  let destination = result.dataPointer.assumingMemoryBound(to: Float.self)
  let taps = (0..<outputSize).map { index -> (Int, Int, Float) in
    let position = max(0, min(Float(inputSize - 1),
      (Float(index) + 0.5) * Float(inputSize) / Float(outputSize) - 0.5))
    let lower = Int(position)
    return (lower, min(inputSize - 1, lower + 1), position - Float(lower))
  }
  DispatchQueue.concurrentPerform(iterations: paddedCount) { frame in
    let source = sources[frame]
    for channel in 0..<3 {
      let base = channel * inputPlane
      let target = destination + (frame * 3 + channel) * plane
      for y in 0..<outputSize {
        let (y0, y1, wy) = taps[y]
        let upperRow = base + y0 * inputSize
        let lowerRow = base + y1 * inputSize
        for x in 0..<outputSize {
          let (x0, x1, wx) = taps[x]
          let upper = source[upperRow + x0] * (1 - wx) + source[upperRow + x1] * wx
          let lower = source[lowerRow + x0] * (1 - wx) + source[lowerRow + x1] * wx
          target[y * outputSize + x] = upper * (1 - wy) + lower * wy
        }
      }
    }
  }
  print("Prepared \(paddedCount) frames")
  return result
}

/// One input frame as planar RGB Float in [0, 1].
private func planarSource(_ url: URL, raw: Bool) throws -> [Float] {
  let inputPlane = inputSize * inputSize
  var source = [Float](repeating: 0, count: 3 * inputPlane)
  if raw {
    let data = try Data(contentsOf: url)
    guard data.count == 3 * inputPlane * MemoryLayout<Float16>.size else {
      throw ClipError.invalid("Invalid planar FP16 input: \(url.path)")
    }
    data.withUnsafeBytes { bytes in
      let values = bytes.bindMemory(to: Float16.self)
      for index in 0..<(3 * inputPlane) { source[index] = Float(values[index]) }
    }
  } else {
    let pixels = try imagePixels(url)
    for channel in 0..<3 {
      for index in 0..<inputPlane {
        source[channel * inputPlane + index] = Float(pixels[index * 4 + channel]) / 255
      }
    }
  }
  return source
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
  latentCount: Int, side: Int
) throws -> MLMultiArray {
  guard temporalOffset >= 0, temporalOffset + latentCount <= 1024 else {
    throw ClipError.invalid("Temporal RoPE table is too short")
  }
  let path = components.appendingPathComponent("rope-\(axis).f32")
  let table = try fromFile(path, shape: [1024, 128])
  let source = table.dataPointer.assumingMemoryBound(to: Float.self)
  let result = try array([1, latentCount * side * side, 1, 128])
  let target = result.dataPointer.assumingMemoryBound(to: Float.self)
  for time in 0..<latentCount {
    for y in 0..<side {
      for x in 0..<side {
        let start = ((time * side + y) * side + x) * 128
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
  outputFrame: Int, latentCount: Int, folder: URL, outputSize: Int
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
  outputFrame: Int, latentCount: Int, folder: URL, outputSize: Int
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
  _ latents: MLMultiArray, count: Int, trailing: Int, side: Int
) throws -> MLMultiArray {
  guard (1...count).contains(trailing) else {
    throw ClipError.invalid("Invalid preceding latent count")
  }
  let source = try contiguous(latents, shape: [count, 48, side, side])
  let result = try array([trailing, 48, side, side])
  let elements = 48 * side * side
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
  assetRoot: URL, geometry: Geometry, started: Date
) throws -> MLMultiArray {
  let latentSide = geometry.latentSide
  let tokenSide = geometry.tokenSide
  guard [6, 7].contains(encodedCount), (1...encodedCount).contains(validCount) else {
    throw ClipError.invalid("Invalid encoded or valid latent count")
  }
  let ditCount = encodedCount == 7 || previous != nil ? 7 : 6
  let padding = ditCount - validCount
  let raw = try contiguous(latents, shape: [encodedCount, 48, latentSide, latentSide])
  let rawValues = raw.dataPointer.assumingMemoryBound(to: Float.self)
  let previousValues = previous?.dataPointer.assumingMemoryBound(to: Float.self)
  let ditInput = try array([1, 48, ditCount, latentSide, latentSide])
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
  let variant = assetRoot.appendingPathComponent(geometry.pack(shapeName))
  let components = variant.appendingPathComponent("components")
  let tokens = ditCount * tokenSide * tokenSide
  var hidden = try pooled {
    try contiguous(
      predict(components.appendingPathComponent("patch.mlpackage"),
        values: ["latents": ditInput], output: "output"),
      shape: [1, tokens, 3072]
    )
  }
  let conditions: [String: Any] = [
    "context": try fromFile(components.appendingPathComponent("context.f32"), shape: [1, 512, 3072]),
    "modulation": try fromFile(components.appendingPathComponent("modulation.f32"), shape: [1, 6, 3072]),
    "cosine": try rotary(components, axis: "cosine", temporalOffset: temporalOffset,
      latentCount: ditCount, side: tokenSide),
    "sine": try rotary(components, axis: "sine", temporalOffset: temporalOffset,
      latentCount: ditCount, side: tokenSide),
  ]
  if let groups = try groupedStack(assetRoot, shapeName: shapeName, geometry: geometry) {
    for (index, model) in groups.enumerated() {
      var inputs = conditions
      inputs["hidden"] = hidden
      hidden = try pooled {
        try contiguous(
          feature(predictFeatures(model, values: inputs), name: "output",
            source: variant),
          shape: [1, tokens, 3072]
        )
      }
      print("DiT group \(index + 1)/\(groups.count) elapsed=\(Date().timeIntervalSince(started))s")
    }
  } else {
    let blocks = try (0..<30).map { layer in
      try compiledURL(for: variant.appendingPathComponent(String(
        format: "dit-block-%02d-%@-%@-float16.mlpackage", layer, shapeName,
        geometry.packName)))
    }
    var pending = PendingModel(compiled: blocks[0])
    for layer in blocks.indices {
      let model = try pending.wait()
      if layer + 1 < blocks.count {
        pending = PendingModel(compiled: blocks[layer + 1])
      }
      var inputs = conditions
      inputs["hidden"] = hidden
      hidden = try pooled {
        try contiguous(
          feature(predictFeatures(model, values: inputs), name: "output",
            source: blocks[layer]),
          shape: [1, tokens, 3072]
        )
      }
      print("DiT \(layer + 1)/30 elapsed=\(Date().timeIntervalSince(started))s")
    }
  }
  let velocity = try pooled {
    try contiguous(
      predict(components.appendingPathComponent("head.mlpackage"),
        values: ["hidden": hidden], output: "output"),
      shape: [1, 48, ditCount, latentSide, latentSide]
    )
  }
  let velocityValues = velocity.dataPointer.assumingMemoryBound(to: Float.self)
  let decodedInput = try array([1, encodedCount, 48, latentSide, latentSide])
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
      if sharedCompiledRoot == nil {
        for compiled in compiledModels.values {
          try? FileManager.default.removeItem(at: compiled)
        }
      }
      compiledModels.removeAll()
    }
    let arguments = CommandLine.arguments
    if arguments.count == 3, arguments[2] == "--serve" {
      try serve(root: URL(fileURLWithPath: arguments[1], isDirectory: true))
      return
    }
    guard (4...6).contains(arguments.count) else {
      throw ClipError.invalid("usage: swiftvr-native-clip <asset-root> (--serve | <input-png-directory> <output-png-directory> [frame-count [scale]])")
    }
    let requested: Int?
    if arguments.count >= 5 {
      guard let value = Int(arguments[4]) else {
        throw ClipError.invalid("frame-count must be an integer")
      }
      requested = value
    } else {
      requested = nil
    }
    try runScene(
      root: URL(fileURLWithPath: arguments[1], isDirectory: true),
      input: URL(fileURLWithPath: arguments[2], isDirectory: true),
      output: URL(fileURLWithPath: arguments[3], isDirectory: true),
      requestedFrames: requested,
      geometry: Geometry(scale: arguments.count == 6 ? Int(arguments[5]) ?? 0 : 4)
    )
  }

  /// Serves every scene of one export from this process instead of starting
  /// a worker per scene. Each stdin line is a JSON request
  /// `{"input": path, "output": path, "frames": count, "scale": 2|4}` (scale
  /// defaults to 4); each completed scene
  /// is acknowledged by one stdout line `SWIFTVR-DONE <output path>`.
  static func serve(root: URL) throws {
    struct Request: Decodable {
      let input: String
      let output: String
      let frames: Int
      let scale: Int?
    }
    while let line = readLine() {
      guard !line.isEmpty else { continue }
      let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
      try pooled {
        try runScene(
          root: root,
          input: URL(fileURLWithPath: request.input, isDirectory: true),
          output: URL(fileURLWithPath: request.output, isDirectory: true),
          requestedFrames: request.frames,
          geometry: Geometry(scale: request.scale ?? 4)
        )
      }
      print("SWIFTVR-DONE \(request.output)")
      fflush(stdout)
    }
  }

  static func runScene(
    root: URL, input: URL, output: URL, requestedFrames: Int?, geometry: Geometry
  ) throws {
    let size = geometry.outputSize
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let inputs = try FileManager.default.contentsOfDirectory(
      at: input, includingPropertiesForKeys: nil
    )
    let rawOutput = inputs.contains { $0.pathExtension.lowercased() == "f16" }
    let availableFrames = inputs.filter {
      $0.pathExtension.lowercased() == (rawOutput ? "f16" : "png")
    }.count
    let totalFrames: Int
    if let requested = requestedFrames {
      guard requested > 0, requested <= availableFrames else {
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
      input, start: 0, validCount: firstCount, paddedCount: 28, geometry: geometry
    )
    var encoderInputs = try zeroStates(
      Array(repeating: [64, size / 4, size / 4], count: 3)
        + Array(repeating: [64, size / 8, size / 8], count: 3)
        + Array(repeating: [64, size / 16, size / 16], count: 3)
    )
    encoderInputs["frames"] = frames
    let firstEncoderURL = root.appendingPathComponent(geometry.encoder(frames: 28))
    let firstEncoded = try predictFeatures(firstEncoderURL, values: encoderInputs)
    let firstLatents = try feature(
      firstEncoded, name: "latents", source: firstEncoderURL
    )
    print("Encoded 28 frames in \(Date().timeIntervalSince(started))s")
    let firstDenoised = try denoise(
      latents: firstLatents, encodedCount: 7,
      validCount: min(7, (max(totalFrames, 1) - 1 + 3) / 4 + 1),
      previous: nil, temporalOffset: 0,
      assetRoot: root, geometry: geometry, started: started
    )
    var decoderInputs = try zeroStates(
      Array(repeating: [512, size / 16, size / 16], count: 3)
        + Array(repeating: [256, size / 8, size / 8], count: 3)
        + Array(repeating: [128, size / 4, size / 4], count: 3)
    )
    decoderInputs["latents"] = firstDenoised
    let firstDecoderURL = root.appendingPathComponent(geometry.decoder(latents: 7))
    let firstDecoded = try predictFeatures(firstDecoderURL, values: decoderInputs)
    let firstOutput = try feature(firstDecoded, name: "frames", source: firstDecoderURL)
    let firstOutputCount = min(totalFrames, 25)
    for index in 0..<firstOutputCount {
      if rawOutput {
        try saveRaw(firstOutput, decoderFrame: index + 3,
          outputFrame: index, latentCount: 7, folder: output,
          outputSize: size)
      } else {
        try savePNG(firstOutput, decoderFrame: index + 3,
          outputFrame: index, latentCount: 7, folder: output,
          outputSize: size)
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
      try pooled {
        let remaining = paddedTotal - nextInput
        let isLast = remaining <= 24
        let chunkFrames = min(remaining, 24)
        let availableFrames = max(1, min(totalFrames - nextInput, chunkFrames))
        let continuationFrames = try inputFrames(
          input, start: nextInput, validCount: availableFrames, paddedCount: 24,
          geometry: geometry
        )
        var encoderInputs = try followingStates(encoded, source: encodedURL)
        encoderInputs["frames"] = continuationFrames
        let encoderURL = root.appendingPathComponent(geometry.encoder(frames: 24))
        let nextEncoded = try predictFeatures(encoderURL, values: encoderInputs)
        let nextLatents = try feature(
          nextEncoded, name: "latents", source: encoderURL
        )
        let validLatents = isLast ? (chunkFrames - 1) / 4 + 1 : 6
        let padLatents = isLast ? 7 - validLatents : 0
        let precedingLatents = padLatents > 0
          ? try trailingLatents(rawLatents, count: rawLatentCount,
            trailing: padLatents, side: geometry.latentSide)
          : nil
        // MIDDLE uses the six-latent graph without overlap. LAST prepends the
        // previous raw latents (not denoised latents) to the seven-latent graph.
        let nextDenoised = try denoise(
          latents: nextLatents, encodedCount: 6, validCount: validLatents,
          previous: precedingLatents,
          temporalOffset: max(0, latentOffset - padLatents),
          assetRoot: root, geometry: geometry, started: started
        )
        var decoderInputs = try followingStates(decoded, source: decodedURL)
        decoderInputs["latents"] = nextDenoised
        let decoderURL = root.appendingPathComponent(geometry.decoder(latents: 6))
        let nextDecoded = try predictFeatures(decoderURL, values: decoderInputs)
        let decodedFrames = try feature(nextDecoded, name: "frames", source: decoderURL)
        let outputCount = min(totalFrames - nextOutput, 24)
        for index in 0..<outputCount {
          if rawOutput {
            try saveRaw(decodedFrames, decoderFrame: index,
              outputFrame: nextOutput + index, latentCount: 6, folder: output,
              outputSize: size)
          } else {
            try savePNG(decodedFrames, decoderFrame: index,
              outputFrame: nextOutput + index, latentCount: 6, folder: output,
              outputSize: size)
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
    }
    print("Swift/Core ML clip complete in \(Date().timeIntervalSince(started))s: \(output.path)")
  }
}
