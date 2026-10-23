// SPDX-License-Identifier: AGPL-3.0
import Foundation

@main
struct ROIEnhancerFrameSmoothingHarness {
  static func main() {
    func placement(_ offset: Float, scale: Float = 1) -> ROIEnhancerGridPlacement {
      ROIEnhancerGridPlacement(scaleX: scale, offsetX: offset, scaleY: scale, offsetY: offset)
    }
    let placements = [placement(0), placement(2), placement(-2)]
    for window in [0, 1] {
      let views = ROIEnhancerFrameSmoothing.views(placements: placements, window: window)
      precondition(views.map(\.smoothOffsetX) == [0, 2, -2])
    }
    let views = ROIEnhancerFrameSmoothing.views(placements: placements, window: 3)
    precondition(views.map(\.smoothOffsetX) == [1, 0, 0])
    let gapped = ROIEnhancerFrameSmoothing.views(
      placements: placements, window: 30, frameIndices: [0, 1, 10])
    precondition(gapped.map(\.smoothOffsetX) == [1, 1, -2])
    let singleton = ROIEnhancerFrameSmoothing.views(
      placements: [placement(9, scale: 0.6)], window: 15, frameIndices: [20])[0]
    precondition(singleton.smoothOffsetX == singleton.offsetX)
    precondition(singleton.smoothScaleX == singleton.scaleX)
    precondition(ROIEnhancerFrameSmoothing.views(placements: [], window: 15).isEmpty)

    let side = 256
    let elements = 3 * side * side
    var original = [Float16](repeating: 0, count: elements)
    for channel in 0..<3 {
      for y in 0..<side {
        for x in 0..<side {
          original[channel * side * side + y * side + x] = Float16(
            0.15 + Float((x * 7 + y * 13 + channel * 19) % 31) / 60)
        }
      }
    }
    let moved = ROIEnhancerSmoothedView(
      scaleX: 1, offsetX: 0, scaleY: 1, offsetY: 0,
      smoothScaleX: 0.98, smoothOffsetX: 2.4,
      smoothScaleY: 1.02, smoothOffsetY: -1.7)
    let xs = (0..<side).map { (Float($0) - moved.smoothOffsetX) / moved.smoothScaleX }
    let ys = (0..<side).map { (Float($0) - moved.smoothOffsetY) / moved.smoothScaleY }
    var smoothed = original
    ROIEnhancerFrameSmoothing.resample(original, sourceOffset: 0, sourceSide: side,
      xs: xs, ys: ys, into: &smoothed, targetOffset: 0, targetSide: side)
    let highSide = 512
    let highCoordinates = (0..<highSide).map { (Float($0) + 0.5) / 2 - 0.5 }
    var ownBase = [Float16](repeating: 0, count: 3 * highSide * highSide)
    var smoothBase = ownBase
    ROIEnhancerFrameSmoothing.resample(original, sourceOffset: 0, sourceSide: side,
      xs: highCoordinates, ys: highCoordinates, into: &ownBase, targetOffset: 0, targetSide: highSide)
    ROIEnhancerFrameSmoothing.resample(smoothed, sourceOffset: 0, sourceSide: side,
      xs: highCoordinates, ys: highCoordinates, into: &smoothBase, targetOffset: 0, targetSide: highSide)
    let noDetail = ROIEnhancerFrameSmoothing.mapResidualBack(
      smoothBase, side: highSide, original: original, smoothed: smoothed, offset: 0, view: moved)
    precondition(noDetail == ownBase, "smoothing must not blur the restoration base")
    let clippedBase = ROIEnhancerFrameSmoothing.mapResidualBack(
      [Float16](repeating: 0, count: ownBase.count), side: highSide, original: original,
      smoothed: [Float16](repeating: -0.1, count: elements), offset: 0, view: moved)
    precondition(clippedBase == ownBase, "input clipping must not become a bright residual")
    let detailed = smoothBase.map { Float16(Float($0) + 0.02) }
    let mapped = ROIEnhancerFrameSmoothing.mapResidualBack(
      detailed, side: highSide, original: original, smoothed: smoothed, offset: 0, view: moved)
    for index in mapped.indices {
      precondition(abs(Float(mapped[index]) - Float(ownBase[index]) - 0.02) < 0.001)
    }
    let unchanged = ROIEnhancerSmoothedView(
      scaleX: 1, offsetX: 3, scaleY: 1, offsetY: 4,
      smoothScaleX: 1, smoothOffsetX: 3, smoothScaleY: 1, smoothOffsetY: 4)
    precondition(ROIEnhancerFrameSmoothing.mapBack(detailed, side: highSide, view: unchanged) == detailed)
    precondition(ROIEnhancerFrameSmoothing.mapResidualBack(
      detailed, side: highSide, original: original, smoothed: smoothed, offset: 0, view: unchanged) == detailed)
    // Inverse mapping uses pixel centres and output scale, not integer ROI origins.
    let shift = ROIEnhancerSmoothedView(
      scaleX: 1, offsetX: 0, scaleY: 1, offsetY: 0,
      smoothScaleX: 1, smoothOffsetX: 2, smoothScaleY: 1, smoothOffsetY: 0)
    var ramp = [Float16](repeating: 0, count: 3 * highSide * highSide)
    for i in ramp.indices { ramp[i] = Float16(Float(i % highSide) / Float(highSide)) }
    let shifted = ROIEnhancerFrameSmoothing.mapBack(ramp, side: highSide, view: shift)
    precondition(shifted[100 * highSide + 100] == ramp[100 * highSide + 104])
    // Clip offsets must not mix another frame's grid into the residual.
    let second = ROIEnhancerFrameSmoothing.mapResidualBack(
      smoothBase, side: highSide, original: [Float16](repeating: 0, count: elements) + original,
      smoothed: [Float16](repeating: 0, count: elements) + smoothed,
      offset: elements, view: moved)
    precondition(second == ownBase)
    // v6: masks share both RGB inputs' forward transform, not the inverse
    // writeback transform. Identity is exact and no border is replicated.
    let mask = (0..<(side * side)).map { Float($0 % side + $0 / side) / 512 }
    precondition(ROIEnhancerFrameSmoothing.reframeMasks([mask], views: [unchanged]) == [mask])
    let shiftedMask = ROIEnhancerFrameSmoothing.reframeMasks([mask], views: [shift])[0]
    precondition(shiftedMask[100 * side + 100] == mask[100 * side + 98])
    let full = [Float](repeating: 1, count: side * side)
    let movedFull = ROIEnhancerFrameSmoothing.reframeMasks([full], views: [shift])[0]
    precondition(movedFull[100 * side] == 0 && movedFull[100 * side + 1] == 0)
    precondition(movedFull[100 * side + 2] == 1)
    let sourceRGB = mask.map(Float16.init) + mask.map(Float16.init) + mask.map(Float16.init)
    let originalRGB = sourceRGB.map { Float16(0.25 + Float($0) * 0.5) }
    var alignedRGB = sourceRGB, alignedOriginal = originalRGB
    ROIEnhancerFrameSmoothing.resample(sourceRGB, sourceOffset: 0, sourceSide: side,
      xs: xs, ys: ys, into: &alignedRGB, targetOffset: 0, targetSide: side)
    ROIEnhancerFrameSmoothing.resample(originalRGB, sourceOffset: 0, sourceSide: side,
      xs: xs, ys: ys, into: &alignedOriginal, targetOffset: 0, targetSide: side)
    let alignedMask = ROIEnhancerFrameSmoothing.reframeMasks([mask], views: [moved])[0]
    for y in 16..<240 {
      for x in 16..<240 {
        let pixel = y * side + x
        precondition(abs(Float(alignedRGB[pixel]) - alignedMask[pixel]) < 0.001)
        precondition(abs(Float(alignedOriginal[pixel]) - 0.25 - alignedMask[pixel] * 0.5) < 0.001)
      }
    }
    // A mask-free v6 output is raw bilinear restoration, NOT a clipped
    // model prediction. It must never become artificial detail after mapping.
    let raw = [Float16](repeating: -0.125, count: elements)
    let rawHigh = [Float16](repeating: -0.125, count: ownBase.count)
    let rawBack = ROIEnhancerFrameSmoothing.mapResidualBack(
      rawHigh, side: highSide, original: raw, smoothed: raw, offset: 0,
      view: moved, mask: [Float](repeating: 0, count: side * side))
    precondition(rawBack == rawHigh, "unmasked overshoot must not become a clipping residual")
    // With a full mask, learned detail is inverse-mapped without blurring
    // the original restoration. Legacy Distill behaviour remains unchanged.
    let maskedMapped = ROIEnhancerFrameSmoothing.mapResidualBack(
      detailed, side: highSide, original: original, smoothed: smoothed, offset: 0,
      view: moved, mask: full)
    precondition(maskedMapped == mapped)
    print("ROI frame smoothing tests passed")
  }
}
