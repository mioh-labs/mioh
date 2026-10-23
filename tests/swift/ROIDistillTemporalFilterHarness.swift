// SPDX-License-Identifier: AGPL-3.0
import CoreMedia
import CoreVideo
import Foundation

@main
struct ROIDistillTemporalFilterHarness {
  static func buffer(width: Int, height: Int, phase: Int) throws -> CVPixelBuffer {
    var result: CVPixelBuffer?
    let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
      [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary, &result)
    guard status == kCVReturnSuccess, let result else { throw NSError(domain: "test", code: 1) }
    CVPixelBufferLockBaseAddress(result, [])
    let pointer = CVPixelBufferGetBaseAddress(result)!.assumingMemoryBound(to: UInt8.self)
    let row = CVPixelBufferGetBytesPerRow(result)
    for y in 0..<height {
      for x in 0..<width {
        let noise = ((x * 17 + y * 31 + phase * 13) % 17) - 8
        for channel in 0..<3 { pointer[y * row + x * 4 + channel] = UInt8(100 + channel * 20 + noise) }
        pointer[y * row + x * 4 + 3] = 255
      }
    }
    CVPixelBufferUnlockBaseAddress(result, [])
    return result
  }

  static func pixels(_ buffer: CVPixelBuffer) -> [UInt8] {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let pointer = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let row = CVPixelBufferGetBytesPerRow(buffer)
    return (0..<CVPixelBufferGetHeight(buffer)).flatMap {
      Array(UnsafeBufferPointer(start: pointer + $0 * row, count: CVPixelBufferGetWidth(buffer) * 4))
    }
  }

  static func main() async throws {
    // Mask and ownership both gate writes. Test offset coordinates, partial
    // weights and alpha preservation with the production compositor.
    let destination = try buffer(width: 12, height: 10, phase: 0)
    let filtered = try buffer(width: 8, height: 6, phase: 8)
    let before = pixels(destination), source = pixels(filtered)
    let enhancer: [Float] = (0..<48).map { $0 % 3 == 0 ? 0 : 1 }
    let ownership: [Float] = (0..<48).map { $0 % 5 == 0 ? 0 : ($0 % 2 == 0 ? 0.5 : 1) }
    let weights = SwiftVRTemporalNoiseFilter.blendWeights(enhancer: enhancer, ownership: ownership)
    SwiftVRTemporalNoiseFilter.blend(filtered, left: 2, top: 3, into: destination,
      cropLeft: 2, cropTop: 3, cropWidth: 8, cropHeight: 6, weights: weights)
    let after = pixels(destination)
    for y in 0..<10 {
      for x in 0..<12 {
        let inside = (2..<10).contains(x) && (3..<9).contains(y)
        let local = inside ? (y - 3) * 8 + x - 2 : 0
        let weight = inside ? weights[local] : 0
        for channel in 0..<4 {
          let index = (y * 12 + x) * 4 + channel
          let expected = channel == 3 || weight == 0 ? before[index] : UInt8(
            (Float(before[index]) + weight * (Float(source[local * 4 + channel]) - Float(before[index]))).rounded())
          precondition(after[index] == expected)
        }
      }
    }
    print("ROI temporal masked writeback tests passed")
    let width = 640, height = 192
    guard SwiftVRTemporalNoiseFilter.supports(width: width, height: height) else {
      print("Apple temporal filter unsupported")
      return
    }
    precondition(!SwiftVRTemporalNoiseFilter.supports(width: 1, height: 1))
    let start = Date()
    var completed: [Int] = []
    // Separate sessions simulate a detection gap. A new run never borrows
    // earlier frames. Exercise runs shorter than the required lookahead.
    for run in [0..<2, 2..<7] {
      let filter = try SwiftVRTemporalNoiseFilter(width: width, height: height, strength: 1)
      var originals: [Int: [UInt8]] = [:]
      for index in run {
        let frame = try buffer(width: width, height: height, phase: index)
        originals[index] = pixels(frame)
        // Non-29.97 and irregular intervals use original frame timestamps.
        let pts = CMTime(value: 2_000_000_000 + Int64(index) * 41_700_000 + Int64(index % 2) * 300_000,
          timescale: 1_000_000_000)
        try filter.add(index, from: frame, left: 0, top: 0, presentationTimeStamp: pts)
        precondition(pixels(frame) == originals[index])
        let ready = index - filter.nextCount
        if ready >= run.lowerBound {
          let output = try await filter.filtered(ready)
          precondition(CVPixelBufferGetWidth(output) == width && CVPixelBufferGetHeight(output) == height)
          completed.append(ready)
        }
      }
      for index in max(run.lowerBound, run.upperBound - filter.nextCount)..<run.upperBound {
        _ = try await filter.filtered(index)
        completed.append(index)
      }
    }
    precondition(completed == Array(0..<7))
    print("Apple temporal filter processed 7 synthetic frames in \(Date().timeIntervalSince(start))s")
  }
}
