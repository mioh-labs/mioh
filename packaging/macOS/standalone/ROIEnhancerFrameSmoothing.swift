// SPDX-License-Identifier: AGPL-3.0
import Foundation

/// Frame pixel centres -> restoration-grid pixel centres.
struct ROIEnhancerGridPlacement {
  let scaleX: Float
  let offsetX: Float
  let scaleY: Float
  let offsetY: Float
}

struct ROIEnhancerSmoothedView {
  let scaleX: Float
  let offsetX: Float
  let scaleY: Float
  let offsetY: Float
  let smoothScaleX: Float
  let smoothOffsetX: Float
  let smoothScaleY: Float
  let smoothOffsetY: Float
}

/// Shared geometry only: never changes masks, object ownership or restoration.
enum ROIEnhancerFrameSmoothing {
  static let gridSide = 256

  static func views(
    placements: [ROIEnhancerGridPlacement], window: Int, frameIndices: [Int]? = nil
  ) -> [ROIEnhancerSmoothedView] {
    if let frameIndices {
      precondition(frameIndices.count == placements.count)
      precondition(zip(frameIndices, frameIndices.dropFirst()).allSatisfy { $0 < $1 })
    }
    let half = max(0, min(30, window) / 2)
    return placements.indices.map { index in
      var first = max(0, index - half)
      var last = min(placements.count - 1, index + half)
      if let frameIndices {
        // A detection gap is not a continuous temporal neighbourhood.
        var lower = index
        var upper = index
        while lower > first, frameIndices[lower] - frameIndices[lower - 1] == 1 {
          lower -= 1
        }
        while upper < last, frameIndices[upper + 1] - frameIndices[upper] == 1 {
          upper += 1
        }
        first = lower
        last = upper
      }
      let range = first...last
      let count = Float(range.count)
      let mean = range.reduce((Float(0), Float(0), Float(0), Float(0))) {
        ($0.0 + placements[$1].scaleX, $0.1 + placements[$1].offsetX,
         $0.2 + placements[$1].scaleY, $0.3 + placements[$1].offsetY)
      }
      let own = placements[index]
      return ROIEnhancerSmoothedView(
        scaleX: own.scaleX, offsetX: own.offsetX, scaleY: own.scaleY, offsetY: own.offsetY,
        smoothScaleX: mean.0 / count, smoothOffsetX: mean.1 / count,
        smoothScaleY: mean.2 / count, smoothOffsetY: mean.3 / count)
    }
  }

  static func mapBack(
    _ pixels: [Float16], side: Int, view: ROIEnhancerSmoothedView
  ) -> [Float16] {
    let factor = Float(side) / Float(gridSide)
    let xs = (0..<side).map { u -> Float in
      let grid = (Float(u) + 0.5) / factor - 0.5
      let smooth = (grid - view.offsetX) / view.scaleX * view.smoothScaleX
        + view.smoothOffsetX
      return (smooth + 0.5) * factor - 0.5
    }
    let ys = (0..<side).map { v -> Float in
      let grid = (Float(v) + 0.5) / factor - 0.5
      let smooth = (grid - view.offsetY) / view.scaleY * view.smoothScaleY
        + view.smoothOffsetY
      return (smooth + 0.5) * factor - 0.5
    }
    var output = pixels
    resample(pixels, sourceOffset: 0, sourceSide: side, xs: xs, ys: ys,
      into: &output, targetOffset: 0, targetSide: side)
    return output
  }

  /// Map ONLY the learned detail back. A full-image inverse warp would also
  /// blur/change BasicVSR++ through two extra interpolation steps.
  static func mapResidualBack(
    _ enhanced: [Float16], side: Int, original: [Float16], smoothed: [Float16],
    offset: Int, view: ROIEnhancerSmoothedView, mask: [Float]? = nil
  ) -> [Float16] {
    if view.scaleX == view.smoothScaleX, view.offsetX == view.smoothOffsetX,
      view.scaleY == view.smoothScaleY, view.offsetY == view.smoothOffsetY {
      return enhanced
    }
    let coordinates = (0..<side).map { (Float($0) + 0.5) * Float(gridSide) / Float(side) - 0.5 }
    var ownBase = [Float16](repeating: 0, count: 3 * side * side)
    var smoothBase = ownBase
    // DistilledROIEnhancer clamps the 256px tensor before inference. Match
    // that exact base here, including any BasicVSR++ overshoot, so clipping
    // is not mistaken for learned detail and shifted back as a bright seam.
    let smoothModelFrame = smoothed[offset..<(offset + 3 * gridSide * gridSide)]
      .map { Float16(max(0, min(1, Float($0)))) }
    resample(original, sourceOffset: offset, sourceSide: gridSide,
      xs: coordinates, ys: coordinates, into: &ownBase, targetOffset: 0, targetSide: side)
    resample(smoothModelFrame, sourceOffset: 0, sourceSide: gridSide,
      xs: coordinates, ys: coordinates, into: &smoothBase, targetOffset: 0, targetSide: side)
    precondition(enhanced.count == ownBase.count)
    let support = mask.map { resampledMask($0, xs: coordinates, ys: coordinates) }
    let plane = side * side
    let residual = enhanced.indices.map {
      // The texture backend preserves raw restoration outside its mask.
      // That region must not turn into an input-clipping residual.
      support?[$0 % plane] == 0 ? Float16(0) : Float16(Float(enhanced[$0]) - Float(smoothBase[$0]))
    }
    let mapped = mapBack(residual, side: side, view: view)
    return mapped.indices.map {
      if mask != nil, mapped[$0] == 0 { return ownBase[$0] }
      return Float16(max(0, min(1, Float(ownBase[$0]) + Float(mapped[$0]))))
    }
  }

  /// Temporary model masks through the same pixel-centre transform as both
  /// RGB inputs. Detector masks and ownership are NOT changed. Zero outside
  /// the grid prevents extending border masks into unseen pixels.
  static func reframeMasks(_ masks: [[Float]], views: [ROIEnhancerSmoothedView]) -> [[Float]] {
    precondition(masks.count == views.count)
    return masks.indices.map { frame in
      let view = views[frame]
      precondition(view.scaleX > 0 && view.scaleY > 0 && view.smoothScaleX > 0 && view.smoothScaleY > 0)
      if view.scaleX == view.smoothScaleX, view.offsetX == view.smoothOffsetX,
        view.scaleY == view.smoothScaleY, view.offsetY == view.smoothOffsetY { return masks[frame] }
      let xs = (0..<gridSide).map {
        (Float($0) - view.smoothOffsetX) / view.smoothScaleX * view.scaleX + view.offsetX
      }
      let ys = (0..<gridSide).map {
        (Float($0) - view.smoothOffsetY) / view.smoothScaleY * view.scaleY + view.offsetY
      }
      return resampledMask(masks[frame], xs: xs, ys: ys)
    }
  }

  private static func resampledMask(_ mask: [Float], xs: [Float], ys: [Float]) -> [Float] {
    precondition(mask.count == gridSide * gridSide)
    precondition(mask.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
    func sample(_ x: Int, _ y: Int) -> Float {
      guard x >= 0, x < gridSide, y >= 0, y < gridSide else { return 0 }
      return mask[y * gridSide + x]
    }
    return ys.flatMap { y -> [Float] in
      let iy = Int(floor(y)), ay = y - Float(iy)
      return xs.map { x in
        let ix = Int(floor(x)), ax = x - Float(ix)
        let top = sample(ix, iy) * (1 - ax) + sample(ix + 1, iy) * ax
        let bottom = sample(ix, iy + 1) * (1 - ax) + sample(ix + 1, iy + 1) * ax
        return max(0, min(1, top * (1 - ay) + bottom * ay))
      }
    }
  }

  static func resample(
    _ source: [Float16], sourceOffset: Int, sourceSide: Int, xs: [Float], ys: [Float],
    into output: inout [Float16], targetOffset: Int, targetSide: Int
  ) {
    let sourcePlane = sourceSide * sourceSide
    let targetPlane = targetSide * targetSide
    let taps = { (position: Float) -> (Int, Int, Float) in
      let clamped = max(0, min(Float(sourceSide - 1), position))
      let lower = Int(clamped)
      return (lower, min(sourceSide - 1, lower + 1), clamped - Float(lower))
    }
    let xTaps = xs.map(taps)
    let yTaps = ys.map(taps)
    source.withUnsafeBufferPointer { input in
      output.withUnsafeMutableBufferPointer { target in
        DispatchQueue.concurrentPerform(iterations: targetSide) { y in
          let (y0, y1, wy) = yTaps[y]
          for channel in 0..<3 {
            let plane = sourceOffset + channel * sourcePlane
            let row = targetOffset + channel * targetPlane + y * targetSide
            for x in 0..<targetSide {
              let (x0, x1, wx) = xTaps[x]
              let upper = Float(input[plane + y0 * sourceSide + x0]) * (1 - wx)
                + Float(input[plane + y0 * sourceSide + x1]) * wx
              let lower = Float(input[plane + y1 * sourceSide + x0]) * (1 - wx)
                + Float(input[plane + y1 * sourceSide + x1]) * wx
              target[row + x] = Float16(upper * (1 - wy) + lower * wy)
            }
          }
        }
      }
    }
  }
}
