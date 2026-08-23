import Foundation

@main
struct MiniMaxH3NativeHarness {
  static func main() throws {
    try expect(H3Geometry.alignFrameCount(240) == 243, "240 frames align to 243")
    let referenceFrames = try H3Geometry.referenceFrameCount(available: 240, output: 243)
    try expect(
      referenceFrames == 226,
      "reference is trimmed down to the nearest 17n+5 count"
    )
    try expect(H3Geometry.videoLatentFrames(pixelFrames: 243) == 72, "video latent T")
    try expect(H3Geometry.audioLatentFrames(pixelFrames: 243) == 405, "audio latent T")
    let qwenIndices = H3Geometry.qwenVideoSampleIndices(frameCount: 226)
    try expect(qwenIndices.count == 19, "Qwen samples the reference at 2 fps")
    try expect(qwenIndices.first == 0 && qwenIndices.last == 216, "Qwen sample bounds")
    let canvas = H3Geometry.adaptCanvas(width: 1920, height: 1080)
    try expect(canvas.width == 1344 && canvas.height == 768, "H3 canvas adaptation")

    let tensor = try H3Tensor(float32: [1, 2, 3, 4], shape: [1, 2, 2])
    let half = try tensor.converted(to: .float16)
    try expect(half.scalarType == .float16 && half.bytes.count == 8, "Float16 boundary")
    let roundTrip = try half.converted(to: .float32).floatValues()
    try expect(roundTrip == [1, 2, 3, 4], "tensor conversion round trip")

    let euler = H3ResMultistep.euler(
      x: [1, 3], denoised: [0, 1], sigma: 1, sigmaDown: 0.5
    )
    try expect(close(euler[0], 0.5) && close(euler[1], 2), "Euler update")
    let coefficients = H3ResMultistep.secondOrderCoefficients(
      sigma: 0.8,
      oldSigmaDown: 0.8,
      sigmaDown: 0.5,
      previousSigma: 1
    )
    try expect(
      coefficients.expNegativeH.isFinite && coefficients.b1.isFinite
        && coefficients.b2.isFinite,
      "second-order coefficients are finite"
    )

    var rngA = H3SplitMix64(seed: 42)
    var rngB = H3SplitMix64(seed: 42)
    try expect(rngA.normal(count: 32) == rngB.normal(count: 32), "seed reproducibility")

    let keyA = H3StageCache.key(parts: [Data("prompt-a".utf8)])
    let keyB = H3StageCache.key(parts: [Data("prompt-b".utf8)])
    try expect(keyA != keyB, "cache key separates prompt conditions")
    print("MiniMaxH3NativeHarness: PASS")
  }

  private static func close(_ lhs: Float, _ rhs: Float) -> Bool {
    abs(lhs - rhs) < 0.000_01
  }

  private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    guard condition() else {
      throw H3NativeError.invalidArguments("test failed: \(message)")
    }
  }
}
