import Foundation

enum H3VideoVAETiling {
  static let clipFrames = 17
  static let tokensPerClip = 5
  static let tokenDrop = 3
  static let tileSize = 256
  static let minimumOverlap = 64
  static let spatialRatio = 16

  struct AxisPlan: Sendable {
    let starts: [Int]
    let lengths: [Int]
    let overlaps: [Int]
  }

  static func axisPlan(length: Int) throws -> AxisPlan {
    guard length >= tileSize else {
      throw H3NativeError.unsupported(
        "MiniMax H3 native video VAE currently needs both canvas edges >= \(tileSize)"
      )
    }
    if length == tileSize {
      return AxisPlan(starts: [0], lengths: [tileSize], overlaps: [])
    }
    var count = Int(ceil(Double(length) / Double(tileSize)))
    var overlaps: [Int] = []
    var remaining = 0
    while true {
      overlaps = Array(repeating: minimumOverlap, count: count - 1)
      remaining = tileSize * count - overlaps.reduce(0, +) - length
      if remaining < 0 {
        count += 1
      } else {
        break
      }
    }
    let units = remaining / spatialRatio
    if !overlaps.isEmpty {
      for index in 0..<units {
        overlaps[index % overlaps.count] += spatialRatio
      }
    }
    var starts = [0]
    for index in overlaps.indices {
      starts.append(starts[index] + tileSize - overlaps[index])
    }
    guard starts.last! + tileSize == length else {
      throw H3NativeError.invalidTensor(
        "video VAE tile plan covers \(starts.last! + tileSize), expected \(length)"
      )
    }
    return AxisPlan(
      starts: starts,
      lengths: Array(repeating: tileSize, count: count),
      overlaps: overlaps
    )
  }
}

@available(macOS 27.0, *)
enum H3VideoVAEEncoder {
  static func encode(
    video: H3Tensor,
    runner: H3StageRunner,
    progress: ((Int, Int) -> Void)? = nil
  ) async throws -> H3Tensor {
    guard video.shape.count == 5, video.shape[0] == 1, video.shape[1] == 3 else {
      throw H3NativeError.invalidTensor(
        "video VAE input must be NCTHW [1,3,T,H,W], got \(video.shape)"
      )
    }
    let frames = video.shape[2]
    let height = video.shape[3]
    let width = video.shape[4]
    let vertical = try H3VideoVAETiling.axisPlan(length: height)
    let horizontal = try H3VideoVAETiling.axisPlan(length: width)
    let clips = Int(ceil(Double(frames) / Double(H3VideoVAETiling.clipFrames)))
    let totalTiles = clips * vertical.starts.count * horizontal.starts.count
    let inputType = runner.manifest.inputConstraints?["videoTile"]?.scalarType
      ?? .float16
    guard inputType == .float16 || inputType == .float32 else {
      throw H3NativeError.invalidManifest("videoTile must use float16 or float32")
    }
    let source = try video.float16Values()
    var clipLatents = [[Float]]()
    clipLatents.reserveCapacity(clips)
    var completed = 0
    for clip in 0..<clips {
      var rawRows = [[[Float]]]()
      rawRows.reserveCapacity(vertical.starts.count)
      for y in vertical.starts {
        var row = [[Float]]()
        row.reserveCapacity(horizontal.starts.count)
        for x in horizontal.starts {
          let tileValues = extractTile(
            source: source,
            sourceFrames: frames,
            sourceHeight: height,
            sourceWidth: width,
            clip: clip,
            y: y,
            x: x
          )
          let tile: H3Tensor
          if inputType == .float16 {
            tile = try H3Tensor(
              float16: tileValues,
              shape: [1, 3, 17, 256, 256]
            )
          } else {
            tile = try H3Tensor(
              float32: tileValues.map(Float.init),
              shape: [1, 3, 17, 256, 256]
            )
          }
          let output = try await runner.predict(["videoTile": tile])
          guard let latent = output["videoLatentTile"] else {
            throw H3NativeError.missingTensor("videoEncoder.videoLatentTile")
          }
          guard latent.shape == [1, 24, 5, 16, 16] else {
            throw H3NativeError.invalidTensor(
              "video latent tile is \(latent.shape), expected [1,24,5,16,16]"
            )
          }
          row.append(try latent.floatValues())
          completed += 1
          progress?(completed, totalTiles)
        }
        rawRows.append(row)
      }
      clipLatents.append(
        stitch(
          rows: rawRows,
          vertical: vertical,
          horizontal: horizontal,
          outputHeight: height / H3VideoVAETiling.spatialRatio,
          outputWidth: width / H3VideoVAETiling.spatialRatio
        )
      )
    }
    let latentHeight = height / H3VideoVAETiling.spatialRatio
    let latentWidth = width / H3VideoVAETiling.spatialRatio
    let outputFrames = clips * H3VideoVAETiling.tokensPerClip
      - H3VideoVAETiling.tokenDrop
    var output = [Float16](
      repeating: 0,
      count: 24 * outputFrames * latentHeight * latentWidth
    )
    let plane = latentHeight * latentWidth
    for channel in 0..<24 {
      for clip in 0..<clips {
        let sourceClip = clipLatents[clip]
        for token in 0..<H3VideoVAETiling.tokensPerClip {
          let destinationToken = clip * H3VideoVAETiling.tokensPerClip + token
          guard destinationToken < outputFrames else { continue }
          let sourceOffset = (channel * 5 + token) * plane
          let destinationOffset = (channel * outputFrames + destinationToken) * plane
          for index in 0..<plane {
            output[destinationOffset + index] = Float16(sourceClip[sourceOffset + index])
          }
        }
      }
    }
    return try H3Tensor(
      float16: output,
      shape: [1, 24, outputFrames, latentHeight, latentWidth]
    )
  }

  private static func extractTile(
    source: [Float16],
    sourceFrames: Int,
    sourceHeight: Int,
    sourceWidth: Int,
    clip: Int,
    y: Int,
    x: Int
  ) -> [Float16] {
    let tilePlane = H3VideoVAETiling.tileSize * H3VideoVAETiling.tileSize
    var tile = [Float16](
      repeating: 0,
      count: 3 * H3VideoVAETiling.clipFrames * tilePlane
    )
    for channel in 0..<3 {
      for localFrame in 0..<H3VideoVAETiling.clipFrames {
        let frame = min(
          sourceFrames - 1,
          clip * H3VideoVAETiling.clipFrames + localFrame
        )
        let sourceBase = ((channel * sourceFrames + frame) * sourceHeight + y)
          * sourceWidth + x
        let destinationBase = (channel * H3VideoVAETiling.clipFrames + localFrame)
          * tilePlane
        for row in 0..<H3VideoVAETiling.tileSize {
          let sourceStart = sourceBase + row * sourceWidth
          let destinationStart = destinationBase + row * H3VideoVAETiling.tileSize
          tile.replaceSubrange(
            destinationStart..<(destinationStart + H3VideoVAETiling.tileSize),
            with: source[sourceStart..<(sourceStart + H3VideoVAETiling.tileSize)]
          )
        }
      }
    }
    return tile
  }

  private static func stitch(
    rows: [[[Float]]],
    vertical: H3VideoVAETiling.AxisPlan,
    horizontal: H3VideoVAETiling.AxisPlan,
    outputHeight: Int,
    outputWidth: Int
  ) -> [Float] {
    let channels = 24
    let frames = 5
    let tileHeight = 16
    let tileWidth = 16
    var output = [Float](
      repeating: 0,
      count: channels * frames * outputHeight * outputWidth
    )
    var outputY = 0
    for rowIndex in rows.indices {
      var outputX = 0
      let bottomTrim = rowIndex < rows.count - 1
        ? vertical.overlaps[rowIndex] / H3VideoVAETiling.spatialRatio : 0
      let keptHeight = tileHeight - bottomTrim
      for columnIndex in rows[rowIndex].indices {
        var tile = rows[rowIndex][columnIndex]
        if rowIndex > 0 {
          blend(
            previous: rows[rowIndex - 1][columnIndex],
            current: &tile,
            extent: vertical.overlaps[rowIndex - 1] / H3VideoVAETiling.spatialRatio,
            axis: 0
          )
        }
        if columnIndex > 0 {
          blend(
            previous: rows[rowIndex][columnIndex - 1],
            current: &tile,
            extent: horizontal.overlaps[columnIndex - 1] / H3VideoVAETiling.spatialRatio,
            axis: 1
          )
        }
        let rightTrim = columnIndex < rows[rowIndex].count - 1
          ? horizontal.overlaps[columnIndex] / H3VideoVAETiling.spatialRatio : 0
        let keptWidth = tileWidth - rightTrim
        for channel in 0..<channels {
          for frame in 0..<frames {
            for localY in 0..<keptHeight {
              for localX in 0..<keptWidth {
                let sourceIndex = (((channel * frames + frame) * tileHeight + localY)
                  * tileWidth + localX)
                let destinationIndex = (((channel * frames + frame) * outputHeight
                  + outputY + localY) * outputWidth + outputX + localX)
                output[destinationIndex] = tile[sourceIndex]
              }
            }
          }
        }
        outputX += keptWidth
      }
      outputY += keptHeight
    }
    return output
  }

  private static func blend(
    previous: [Float],
    current: inout [Float],
    extent: Int,
    axis: Int
  ) {
    guard extent > 0 else { return }
    let channels = 24
    let frames = 5
    let tileHeight = 16
    let tileWidth = 16
    for channel in 0..<channels {
      for frame in 0..<frames {
        for y in 0..<tileHeight {
          for x in 0..<tileWidth {
            let position = axis == 0 ? y : x
            guard position < extent else { continue }
            let previousY = axis == 0 ? tileHeight - extent + y : y
            let previousX = axis == 1 ? tileWidth - extent + x : x
            let previousIndex = (((channel * frames + frame) * tileHeight + previousY)
              * tileWidth + previousX)
            let currentIndex = (((channel * frames + frame) * tileHeight + y)
              * tileWidth + x)
            let currentWeight = Float(position) / Float(extent)
            current[currentIndex] = previous[previousIndex] * (1 - currentWeight)
              + current[currentIndex] * currentWeight
          }
        }
      }
    }
  }
}

@available(macOS 27.0, *)
enum H3VideoVAEDecoder {
  private static let channels = 3
  private static let decodedFramesPerTile = 28
  private static let tokensPerChunk = 5
  private static let tokenOverlap = 2
  private static let tokenDrop = 3
  private static let framePrePadding = 3
  private static let frameOverlap = 5
  private static let framesPerToken = 4
  private static let chunkFrames = 20
  private static let pixelMean: [Float] = [0.485, 0.456, 0.406]
  private static let pixelStandardDeviation: [Float] = [0.229, 0.224, 0.225]

  private struct TemporalPlan {
    let chunks: Int
    let paddedTokens: Int
    let outputFrames: Int
  }

  static func decode(
    latent: H3Tensor,
    runner: H3StageRunner,
    progress: ((Int, Int) -> Void)? = nil
  ) async throws -> H3Tensor {
    guard latent.shape.count == 5, latent.shape[0] == 1, latent.shape[1] == 24 else {
      throw H3NativeError.invalidTensor(
        "video VAE latent must be NCTHW [1,24,T,H,W], got \(latent.shape)"
      )
    }
    let sourceFrames = latent.shape[2]
    let latentHeight = latent.shape[3]
    let latentWidth = latent.shape[4]
    let height = latentHeight * H3VideoVAETiling.spatialRatio
    let width = latentWidth * H3VideoVAETiling.spatialRatio
    let vertical = try H3VideoVAETiling.axisPlan(length: height)
    let horizontal = try H3VideoVAETiling.axisPlan(length: width)
    let temporal = temporalPlan(tokens: sourceFrames)
    let tilesPerChunk = vertical.starts.count * horizontal.starts.count
    let totalTiles = temporal.chunks * tilesPerChunk
    let inputType = runner.manifest.inputConstraints?["videoLatentTile"]?.scalarType
      ?? .float16
    guard inputType == .float16 || inputType == .float32 else {
      throw H3NativeError.invalidManifest(
        "videoLatentTile must use float16 or float32"
      )
    }
    let source = try latent.float16Values()
    var output = [Float16](
      repeating: 0,
      count: channels * temporal.outputFrames * height * width
    )
    var writePosition = 0
    var previousOverlap: [Float]?
    var completedTiles = 0

    for chunk in 0..<temporal.chunks {
      let raw = try await decodeSpatialChunk(
        source: source,
        sourceFrames: sourceFrames,
        sourceHeight: latentHeight,
        sourceWidth: latentWidth,
        startToken: chunk * tokensPerChunk,
        height: height,
        width: width,
        vertical: vertical,
        horizontal: horizontal,
        inputType: inputType,
        runner: runner,
        progress: { localCompleted in
          progress?(completedTiles + localCompleted, totalTiles)
        }
      )
      completedTiles += tilesPerChunk
      var current = raw
      if let previousOverlap {
        blendTemporal(
          previous: previousOverlap,
          current: &current,
          currentStartFrame: framePrePadding,
          height: height,
          width: width
        )
      }
      writeFinalized(
        source: current,
        sourceFrameRange: framePrePadding..<chunkFrames,
        height: height,
        width: width,
        destination: &output,
        destinationFrames: temporal.outputFrames,
        writePosition: &writePosition
      )
      previousOverlap = extractFrames(
        source: current,
        frameRange: (chunkFrames + framePrePadding)..<decodedFramesPerTile,
        height: height,
        width: width
      )
    }
    if let previousOverlap {
      writeFinalized(
        source: previousOverlap,
        sourceFrameRange: 0..<frameOverlap,
        height: height,
        width: width,
        destination: &output,
        destinationFrames: temporal.outputFrames,
        writePosition: &writePosition
      )
    }
    guard writePosition == temporal.outputFrames else {
      throw H3NativeError.invalidTensor(
        "video decoder wrote \(writePosition) frames, expected \(temporal.outputFrames)"
      )
    }
    return try H3Tensor(
      float16: output,
      shape: [1, channels, temporal.outputFrames, height, width]
    )
  }

  private static func temporalPlan(tokens: Int) -> TemporalPlan {
    var pseudoTotal = tokens + tokenDrop
    var paddedTokens = (tokensPerChunk - pseudoTotal % tokensPerChunk)
      % tokensPerChunk
    pseudoTotal += paddedTokens
    var chunks = pseudoTotal / tokensPerChunk - 1
    if chunks < 1 {
      paddedTokens += tokensPerChunk
      chunks += 1
      pseudoTotal += tokensPerChunk
    }
    let paddedLength = tokens + paddedTokens
    var outputFrames = 0
    var finalOverlapFrames = 0
    for chunk in 0..<chunks {
      let start = chunk * tokensPerChunk
      let end = start + tokensPerChunk + tokenOverlap
      let clipTokens = max(0, min(end, paddedLength) - min(start, paddedLength))
      let clipFrames = clipTokens * framesPerToken
      for split in 0..<2 {
        let frameStart = split * chunkFrames
        let frameEnd = min(frameStart + chunkFrames, clipFrames)
        let count = max(0, frameEnd - frameStart - framePrePadding)
        if split == 0 { outputFrames += count } else { finalOverlapFrames = count }
      }
    }
    outputFrames += finalOverlapFrames
    if paddedTokens > 0 {
      let originalTokens = paddedLength - paddedTokens
      for token in 0..<paddedTokens {
        outputFrames -= (originalTokens + token) % tokensPerChunk == 0 ? 1 : 4
      }
    }
    return TemporalPlan(
      chunks: chunks,
      paddedTokens: paddedTokens,
      outputFrames: outputFrames
    )
  }

  private static func decodeSpatialChunk(
    source: [Float16],
    sourceFrames: Int,
    sourceHeight: Int,
    sourceWidth: Int,
    startToken: Int,
    height: Int,
    width: Int,
    vertical: H3VideoVAETiling.AxisPlan,
    horizontal: H3VideoVAETiling.AxisPlan,
    inputType: H3ScalarType,
    runner: H3StageRunner,
    progress: ((Int) -> Void)?
  ) async throws -> [Float] {
    var canvas = [Float](
      repeating: 0,
      count: channels * decodedFramesPerTile * height * width
    )
    var rowTails = [[Float]]()
    var outputY = 0
    var completed = 0
    for rowIndex in vertical.starts.indices {
      let y = vertical.starts[rowIndex]
      var newTails = [[Float]]()
      var leftTail: [Float]?
      var outputX = 0
      for columnIndex in horizontal.starts.indices {
        let x = horizontal.starts[columnIndex]
        let tileValues = extractLatentTile(
          source: source,
          sourceFrames: sourceFrames,
          sourceHeight: sourceHeight,
          sourceWidth: sourceWidth,
          startToken: startToken,
          y: y / H3VideoVAETiling.spatialRatio,
          x: x / H3VideoVAETiling.spatialRatio
        )
        let input: H3Tensor
        if inputType == .float16 {
          input = try H3Tensor(
            float16: tileValues,
            shape: [1, 24, 7, 16, 16]
          )
        } else {
          input = try H3Tensor(
            float32: tileValues.map(Float.init),
            shape: [1, 24, 7, 16, 16]
          )
        }
        let prediction = try await runner.predict(["videoLatentTile": input])
        guard let tensor = prediction["videoRawTile"] else {
          throw H3NativeError.missingTensor("videoDecoder.videoRawTile")
        }
        guard tensor.shape == [1, 3, 28, 256, 256] else {
          throw H3NativeError.invalidTensor(
            "video raw tile is \(tensor.shape), expected [1,3,28,256,256]"
          )
        }
        var tile = try tensor.floatValues()
        guard tile.allSatisfy(\.isFinite) else {
          throw H3NativeError.inference(
            "video decoder produced NaN or infinity; refusing a corrupted frame"
          )
        }
        if rowIndex < vertical.starts.count - 1 {
          newTails.append(
            extractSpatialTail(
              source: tile,
              extent: vertical.overlaps[rowIndex],
              axis: 0
            )
          )
        }
        let nextLeftTail = columnIndex < horizontal.starts.count - 1
          ? extractSpatialTail(
            source: tile,
            extent: horizontal.overlaps[columnIndex],
            axis: 1
          ) : nil
        if rowIndex > 0 {
          blendSpatialTail(
            previous: rowTails[columnIndex],
            current: &tile,
            extent: vertical.overlaps[rowIndex - 1],
            axis: 0
          )
        }
        if columnIndex > 0, let leftTail {
          blendSpatialTail(
            previous: leftTail,
            current: &tile,
            extent: horizontal.overlaps[columnIndex - 1],
            axis: 1
          )
        }
        leftTail = nextLeftTail
        let keptHeight = H3VideoVAETiling.tileSize
          - (rowIndex < vertical.starts.count - 1 ? vertical.overlaps[rowIndex] : 0)
        let keptWidth = H3VideoVAETiling.tileSize
          - (columnIndex < horizontal.starts.count - 1
            ? horizontal.overlaps[columnIndex] : 0)
        copySpatialTile(
          source: tile,
          keptHeight: keptHeight,
          keptWidth: keptWidth,
          destination: &canvas,
          destinationHeight: height,
          destinationWidth: width,
          destinationY: outputY,
          destinationX: outputX
        )
        outputX += keptWidth
        completed += 1
        progress?(completed)
      }
      rowTails = newTails
      outputY += H3VideoVAETiling.tileSize
        - (rowIndex < vertical.starts.count - 1 ? vertical.overlaps[rowIndex] : 0)
    }
    return canvas
  }

  private static func extractLatentTile(
    source: [Float16],
    sourceFrames: Int,
    sourceHeight: Int,
    sourceWidth: Int,
    startToken: Int,
    y: Int,
    x: Int
  ) -> [Float16] {
    let tilePlane = 16 * 16
    var tile = [Float16](repeating: 0, count: 24 * 7 * tilePlane)
    for channel in 0..<24 {
      for localToken in 0..<7 {
        let token = min(sourceFrames - 1, startToken + localToken)
        let sourceBase = ((channel * sourceFrames + token) * sourceHeight + y)
          * sourceWidth + x
        let destinationBase = (channel * 7 + localToken) * tilePlane
        for row in 0..<16 {
          let sourceStart = sourceBase + row * sourceWidth
          let destinationStart = destinationBase + row * 16
          tile.replaceSubrange(
            destinationStart..<(destinationStart + 16),
            with: source[sourceStart..<(sourceStart + 16)]
          )
        }
      }
    }
    return tile
  }

  private static func extractSpatialTail(
    source: [Float],
    extent: Int,
    axis: Int
  ) -> [Float] {
    if axis == 0 {
      var result = [Float](
        repeating: 0,
        count: channels * decodedFramesPerTile * extent * H3VideoVAETiling.tileSize
      )
      for channel in 0..<channels {
        for frame in 0..<decodedFramesPerTile {
          for y in 0..<extent {
            let sourceStart = (((channel * decodedFramesPerTile + frame)
              * H3VideoVAETiling.tileSize
              + H3VideoVAETiling.tileSize - extent + y)
              * H3VideoVAETiling.tileSize)
            let destinationStart = ((channel * decodedFramesPerTile + frame) * extent + y)
              * H3VideoVAETiling.tileSize
            result.replaceSubrange(
              destinationStart..<(destinationStart + H3VideoVAETiling.tileSize),
              with: source[sourceStart..<(sourceStart + H3VideoVAETiling.tileSize)]
            )
          }
        }
      }
      return result
    }
    var result = [Float](
      repeating: 0,
      count: channels * decodedFramesPerTile * H3VideoVAETiling.tileSize * extent
    )
    for channel in 0..<channels {
      for frame in 0..<decodedFramesPerTile {
        for y in 0..<H3VideoVAETiling.tileSize {
          for x in 0..<extent {
            let sourceIndex = (((channel * decodedFramesPerTile + frame)
              * H3VideoVAETiling.tileSize + y) * H3VideoVAETiling.tileSize
              + H3VideoVAETiling.tileSize - extent + x)
            let destinationIndex = (((channel * decodedFramesPerTile + frame)
              * H3VideoVAETiling.tileSize + y) * extent + x)
            result[destinationIndex] = source[sourceIndex]
          }
        }
      }
    }
    return result
  }

  private static func blendSpatialTail(
    previous: [Float],
    current: inout [Float],
    extent: Int,
    axis: Int
  ) {
    guard extent > 0 else { return }
    for channel in 0..<channels {
      for frame in 0..<decodedFramesPerTile {
        for y in 0..<H3VideoVAETiling.tileSize {
          for x in 0..<H3VideoVAETiling.tileSize {
            let position = axis == 0 ? y : x
            guard position < extent else { continue }
            let previousIndex = axis == 0
              ? (((channel * decodedFramesPerTile + frame) * extent + y)
                * H3VideoVAETiling.tileSize + x)
              : (((channel * decodedFramesPerTile + frame)
                * H3VideoVAETiling.tileSize + y) * extent + x)
            let currentIndex = (((channel * decodedFramesPerTile + frame)
              * H3VideoVAETiling.tileSize + y) * H3VideoVAETiling.tileSize + x)
            let currentWeight = Float(position) / Float(extent)
            current[currentIndex] = previous[previousIndex] * (1 - currentWeight)
              + current[currentIndex] * currentWeight
          }
        }
      }
    }
  }

  private static func copySpatialTile(
    source: [Float],
    keptHeight: Int,
    keptWidth: Int,
    destination: inout [Float],
    destinationHeight: Int,
    destinationWidth: Int,
    destinationY: Int,
    destinationX: Int
  ) {
    for channel in 0..<channels {
      for frame in 0..<decodedFramesPerTile {
        for y in 0..<keptHeight {
          for x in 0..<keptWidth {
            let sourceIndex = (((channel * decodedFramesPerTile + frame)
              * H3VideoVAETiling.tileSize + y) * H3VideoVAETiling.tileSize + x)
            let destinationIndex = (((channel * decodedFramesPerTile + frame)
              * destinationHeight + destinationY + y) * destinationWidth
              + destinationX + x)
            destination[destinationIndex] = source[sourceIndex]
          }
        }
      }
    }
  }

  private static func blendTemporal(
    previous: [Float],
    current: inout [Float],
    currentStartFrame: Int,
    height: Int,
    width: Int
  ) {
    let plane = height * width
    for channel in 0..<channels {
      for frame in 0..<frameOverlap {
        let weight = Float(frame) / Float(frameOverlap)
        let previousStart = (channel * frameOverlap + frame) * plane
        let currentStart = (channel * decodedFramesPerTile + currentStartFrame + frame)
          * plane
        for index in 0..<plane {
          current[currentStart + index] = previous[previousStart + index] * (1 - weight)
            + current[currentStart + index] * weight
        }
      }
    }
  }

  private static func extractFrames(
    source: [Float],
    frameRange: Range<Int>,
    height: Int,
    width: Int
  ) -> [Float] {
    let plane = height * width
    var result = [Float](
      repeating: 0,
      count: channels * frameRange.count * plane
    )
    for channel in 0..<channels {
      for (destinationFrame, sourceFrame) in frameRange.enumerated() {
        let sourceStart = (channel * decodedFramesPerTile + sourceFrame) * plane
        let destinationStart = (channel * frameRange.count + destinationFrame) * plane
        result.replaceSubrange(
          destinationStart..<(destinationStart + plane),
          with: source[sourceStart..<(sourceStart + plane)]
        )
      }
    }
    return result
  }

  private static func writeFinalized(
    source: [Float],
    sourceFrameRange: Range<Int>,
    height: Int,
    width: Int,
    destination: inout [Float16],
    destinationFrames: Int,
    writePosition: inout Int
  ) {
    let framesToWrite = min(sourceFrameRange.count, destinationFrames - writePosition)
    guard framesToWrite > 0 else { return }
    let sourceFrames = source.count / (channels * height * width)
    let plane = height * width
    for channel in 0..<channels {
      for localFrame in 0..<framesToWrite {
        let sourceFrame = sourceFrameRange.lowerBound + localFrame
        let sourceStart = (channel * sourceFrames + sourceFrame) * plane
        let destinationStart = (channel * destinationFrames + writePosition + localFrame)
          * plane
        for index in 0..<plane {
          let value = source[sourceStart + index] * pixelStandardDeviation[channel]
            + pixelMean[channel]
          destination[destinationStart + index] = Float16(min(1, max(0, value)))
        }
      }
    }
    writePosition += framesToWrite
  }
}

extension H3Tensor {
  func float16Values() throws -> [Float16] {
    switch scalarType {
    case .float16:
      return bytes.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    case .float32:
      return try floatValues().map(Float16.init)
    default:
      throw H3NativeError.invalidTensor(
        "\(scalarType.rawValue) cannot be read as Float16"
      )
    }
  }
}
