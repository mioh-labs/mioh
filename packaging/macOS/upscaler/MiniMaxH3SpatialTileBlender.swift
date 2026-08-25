import Foundation

struct H3SpatialTileCorrection: Sendable, Equatable {
  let gains: [Float]
  let offsets: [Float]
  let sampleCount: Int
}

/// Stitches independently decoded VAE tiles without turning their low-frequency
/// exposure differences into visible bands.
///
/// Every tile after the first is matched to pixels already present in the
/// overlap. The match is deliberately limited to a small affine correction per
/// channel, shared by every frame in the temporal chunk, so it cannot introduce
/// frame-to-frame exposure pumping. Tiles are then accumulated with separable
/// raised-cosine windows and normalized once at the end.
struct H3SpatialTileBlender {
  private static let maximumGainDelta: Float = 0.15
  private static let maximumOffset: Float = 0.12
  private static let sampleStride = 4
  private static let frameSampleStride = 2

  let channels: Int
  let frames: Int
  let height: Int
  let width: Int
  let tileSize: Int

  private var sums: [Float]
  private var spatialWeights: [Float]

  init(channels: Int, frames: Int, height: Int, width: Int, tileSize: Int) {
    precondition(channels > 0 && frames > 0 && height > 0 && width > 0)
    precondition(tileSize > 0 && tileSize <= height && tileSize <= width)
    self.channels = channels
    self.frames = frames
    self.height = height
    self.width = width
    self.tileSize = tileSize
    sums = [Float](repeating: 0, count: channels * frames * height * width)
    spatialWeights = [Float](repeating: 0, count: height * width)
  }

  @discardableResult
  mutating func add(
    tile: [Float],
    originY: Int,
    originX: Int,
    topOverlap: Int,
    bottomOverlap: Int,
    leftOverlap: Int,
    rightOverlap: Int
  ) -> H3SpatialTileCorrection {
    precondition(tile.count == channels * frames * tileSize * tileSize)
    precondition(originY >= 0 && originY + tileSize <= height)
    precondition(originX >= 0 && originX + tileSize <= width)

    let correction = estimateCorrection(
      tile: tile,
      originY: originY,
      originX: originX
    )
    let horizontal = Self.axisWeights(
      size: tileSize,
      leadingOverlap: leftOverlap,
      trailingOverlap: rightOverlap
    )
    let vertical = Self.axisWeights(
      size: tileSize,
      leadingOverlap: topOverlap,
      trailingOverlap: bottomOverlap
    )
    var tileWeights = [Float](repeating: 0, count: tileSize * tileSize)
    for y in 0..<tileSize {
      for x in 0..<tileSize {
        tileWeights[y * tileSize + x] = vertical[y] * horizontal[x]
      }
    }

    for y in 0..<tileSize {
      let destinationY = originY + y
      for x in 0..<tileSize {
        let destinationX = originX + x
        spatialWeights[destinationY * width + destinationX]
          += tileWeights[y * tileSize + x]
      }
    }

    for channel in 0..<channels {
      let gain = correction.gains[channel]
      let offset = correction.offsets[channel]
      for frame in 0..<frames {
        for y in 0..<tileSize {
          let destinationY = originY + y
          for x in 0..<tileSize {
            let sourceIndex = (((channel * frames + frame) * tileSize + y)
              * tileSize + x)
            let destinationIndex = (((channel * frames + frame) * height
              + destinationY) * width + originX + x)
            let weight = tileWeights[y * tileSize + x]
            sums[destinationIndex] += (tile[sourceIndex] * gain + offset) * weight
          }
        }
      }
    }
    return correction
  }

  mutating func finalized() -> [Float] {
    for channel in 0..<channels {
      for frame in 0..<frames {
        for y in 0..<height {
          for x in 0..<width {
            let spatialIndex = y * width + x
            let destinationIndex = (((channel * frames + frame) * height + y)
              * width + x)
            let weight = spatialWeights[spatialIndex]
            precondition(weight > 0, "spatial VAE blend left an uncovered pixel")
            sums[destinationIndex] /= weight
          }
        }
      }
    }
    return sums
  }

  private func estimateCorrection(
    tile: [Float],
    originY: Int,
    originX: Int
  ) -> H3SpatialTileCorrection {
    var counts = [Int](repeating: 0, count: channels)
    var sumX = [Double](repeating: 0, count: channels)
    var sumY = [Double](repeating: 0, count: channels)
    var sumXX = [Double](repeating: 0, count: channels)
    var sumXY = [Double](repeating: 0, count: channels)

    for y in stride(from: 0, to: tileSize, by: Self.sampleStride) {
      let destinationY = originY + y
      for x in stride(from: 0, to: tileSize, by: Self.sampleStride) {
        let destinationX = originX + x
        let spatialIndex = destinationY * width + destinationX
        let accumulatedWeight = spatialWeights[spatialIndex]
        guard accumulatedWeight > 1e-6 else { continue }
        for frame in stride(from: 0, to: frames, by: Self.frameSampleStride) {
          for channel in 0..<channels {
            let sourceIndex = (((channel * frames + frame) * tileSize + y)
              * tileSize + x)
            let destinationIndex = (((channel * frames + frame) * height
              + destinationY) * width + destinationX)
            let sourceValue = Double(tile[sourceIndex])
            let destinationValue = Double(sums[destinationIndex] / accumulatedWeight)
            counts[channel] += 1
            sumX[channel] += sourceValue
            sumY[channel] += destinationValue
            sumXX[channel] += sourceValue * sourceValue
            sumXY[channel] += sourceValue * destinationValue
          }
        }
      }
    }

    var gains = [Float](repeating: 1, count: channels)
    var offsets = [Float](repeating: 0, count: channels)
    for channel in 0..<channels where counts[channel] >= 256 {
      let count = Double(counts[channel])
      let meanX = sumX[channel] / count
      let meanY = sumY[channel] / count
      let variance = sumXX[channel] - sumX[channel] * sumX[channel] / count
      let covariance = sumXY[channel] - sumX[channel] * sumY[channel] / count
      var gain = variance > count * 1e-8 ? Float(covariance / variance) : 1
      gain = min(1 + Self.maximumGainDelta, max(1 - Self.maximumGainDelta, gain))
      var offset = Float(meanY) - gain * Float(meanX)
      offset = min(Self.maximumOffset, max(-Self.maximumOffset, offset))
      gains[channel] = gain
      offsets[channel] = offset
    }
    return H3SpatialTileCorrection(
      gains: gains,
      offsets: offsets,
      sampleCount: counts.min() ?? 0
    )
  }

  private static func axisWeights(
    size: Int,
    leadingOverlap: Int,
    trailingOverlap: Int
  ) -> [Float] {
    precondition(leadingOverlap >= 0 && trailingOverlap >= 0)
    precondition(leadingOverlap < size && trailingOverlap < size)
    var result = [Float](repeating: 1, count: size)
    if leadingOverlap > 0 {
      for index in 0..<leadingOverlap {
        let phase = Double(index) + 0.5
        let sine = sin(.pi * 0.5 * phase / Double(leadingOverlap))
        result[index] *= Float(sine * sine)
      }
    }
    if trailingOverlap > 0 {
      for index in 0..<trailingOverlap {
        let phase = Double(index) + 0.5
        let cosine = cos(.pi * 0.5 * phase / Double(trailingOverlap))
        result[size - trailingOverlap + index] *= Float(cosine * cosine)
      }
    }
    return result
  }
}
