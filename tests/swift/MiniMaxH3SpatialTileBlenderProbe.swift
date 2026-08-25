import Foundation

@main
struct MiniMaxH3SpatialTileBlenderProbe {
  static func main() {
    let channels = 3
    let frames = 3
    let height = 480
    let width = 864
    let tileSize = 256
    let verticalStarts = [0, 112, 224]
    let verticalOverlaps = [144, 144]
    let horizontalStarts = [0, 144, 288, 448, 608]
    let horizontalOverlaps = [112, 112, 96, 96]

    let truth = makeTruth(
      channels: channels,
      frames: frames,
      height: height,
      width: width
    )
    var blender = H3SpatialTileBlender(
      channels: channels,
      frames: frames,
      height: height,
      width: width,
      tileSize: tileSize
    )
    var corrections = [H3SpatialTileCorrection]()
    for row in verticalStarts.indices {
      for column in horizontalStarts.indices {
        let tileNumber = row * horizontalStarts.count + column
        let distortionGain: Float = tileNumber == 0
          ? 1 : 1 + Float((tileNumber % 5) - 2) * 0.03
        let distortionOffset: Float = tileNumber == 0
          ? 0 : Float((tileNumber % 4) - 2) * 0.018
        let tile = extractDistortedTile(
          truth: truth,
          channels: channels,
          frames: frames,
          height: height,
          width: width,
          tileSize: tileSize,
          originY: verticalStarts[row],
          originX: horizontalStarts[column],
          gain: distortionGain,
          offset: distortionOffset
        )
        corrections.append(
          blender.add(
            tile: tile,
            originY: verticalStarts[row],
            originX: horizontalStarts[column],
            topOverlap: row > 0 ? verticalOverlaps[row - 1] : 0,
            bottomOverlap: row < verticalStarts.count - 1 ? verticalOverlaps[row] : 0,
            leftOverlap: column > 0 ? horizontalOverlaps[column - 1] : 0,
            rightOverlap: column < horizontalStarts.count - 1
              ? horizontalOverlaps[column] : 0
          )
        )
      }
    }
    let stitched = blender.finalized()
    var absoluteError: Double = 0
    var maximumError: Float = 0
    for index in truth.indices {
      let error = abs(stitched[index] - truth[index])
      absoluteError += Double(error)
      maximumError = max(maximumError, error)
    }
    let meanError = absoluteError / Double(truth.count)
    guard meanError < 0.001, maximumError < 0.01 else {
      fatalError("spatial blend regression: mean=\(meanError), max=\(maximumError)")
    }
    guard corrections.dropFirst().allSatisfy({ $0.sampleCount >= 256 }) else {
      fatalError("spatial overlap did not provide enough matching samples")
    }
    print(
      "shape=[1,\(channels),\(frames),\(height),\(width)] "
        + "mean_error=\(meanError) max_error=\(maximumError) "
        + "matched_tiles=\(corrections.dropFirst().count)"
    )
  }

  private static func makeTruth(
    channels: Int,
    frames: Int,
    height: Int,
    width: Int
  ) -> [Float] {
    var result = [Float](repeating: 0, count: channels * frames * height * width)
    for channel in 0..<channels {
      for frame in 0..<frames {
        for y in 0..<height {
          for x in 0..<width {
            let index = (((channel * frames + frame) * height + y) * width + x)
            result[index] = Float(channel) * 0.08 + Float(frame) * 0.015
              + Float(x) * 0.00025 + Float(y) * 0.00018
              + sin(Float(x + y * 2) * 0.037) * 0.04
          }
        }
      }
    }
    return result
  }

  private static func extractDistortedTile(
    truth: [Float],
    channels: Int,
    frames: Int,
    height: Int,
    width: Int,
    tileSize: Int,
    originY: Int,
    originX: Int,
    gain: Float,
    offset: Float
  ) -> [Float] {
    var tile = [Float](repeating: 0, count: channels * frames * tileSize * tileSize)
    for channel in 0..<channels {
      for frame in 0..<frames {
        for y in 0..<tileSize {
          for x in 0..<tileSize {
            let sourceIndex = (((channel * frames + frame) * height + originY + y)
              * width + originX + x)
            let destinationIndex = (((channel * frames + frame) * tileSize + y)
              * tileSize + x)
            tile[destinationIndex] = truth[sourceIndex] * gain + offset
          }
        }
      }
    }
    return tile
  }
}
