import Foundation

@main
struct MiniMaxH3TemporalContinuationProbe {
  private static let qwenContinuationFrames = 4

  static func main() throws {
    try validateFixedTargetPrefix()
    let urls = CommandLine.arguments.dropFirst().map {
      URL(fileURLWithPath: $0).standardizedFileURL
    }
    if urls.isEmpty {
      print("fixed target-prefix clamping ok")
      return
    }
    guard urls.count == qwenContinuationFrames else {
      throw H3NativeError.invalidJob(
        "expected \(qwenContinuationFrames) continuation frames"
      )
    }
    let tensor = try H3NativeMedia.decodeReferenceImageSequence(
      urls: urls,
      width: 256,
      height: 256
    )
    guard tensor.shape == [1, 3, qwenContinuationFrames, 256, 256]
    else {
      throw H3NativeError.invalidTensor(
        "unexpected continuation tensor shape: \(tensor.shape)"
      )
    }
    let values = try tensor.floatValues()
    let frameElements = 3 * 256 * 256
    let finalOffset = (qwenContinuationFrames - 1) * 256 * 256
    var totalAbsoluteChange = 0.0
    for channel in 0..<3 {
      let first = channel * qwenContinuationFrames * 256 * 256
      let final = first + finalOffset
      for pixel in 0..<(256 * 256) {
        totalAbsoluteChange += abs(
          Double(values[first + pixel]) - Double(values[final + pixel])
        )
      }
    }
    let meanAbsoluteChange = totalAbsoluteChange / Double(frameElements)
    guard meanAbsoluteChange > 0.001 else {
      throw H3NativeError.invalidTensor(
        "continuation decoder lost chronological frame variation"
      )
    }
    print("shape=\(tensor.shape) meanAbsoluteChange=\(meanAbsoluteChange)")
  }

  private static func validateFixedTargetPrefix() throws {
    let targetShape = [1, 24, 12, 1, 1]
    let prefixShape = [1, 24, H3VideoConditioning.partContinuationTokens, 1, 1]
    let target = (0..<targetShape.reduce(1, *)).map(Float.init)
    let clean = (0..<prefixShape.reduce(1, *)).map {
      Float($0) + 1_000
    }
    let noise = try H3VideoConditioning.prefixValues(
      from: target,
      targetShape: targetShape,
      prefixShape: prefixShape
    )
    let atNoise = try H3VideoConditioning.clampTargetPrefix(
      target: target,
      targetShape: targetShape,
      cleanPrefix: clean,
      noisePrefix: noise,
      prefixShape: prefixShape,
      sigma: 1
    )
    guard atNoise == target else {
      throw H3NativeError.invalidTensor(
        "sigma-one continuation prefix did not preserve its fixed noise"
      )
    }
    let atClean = try H3VideoConditioning.clampTargetPrefix(
      target: target,
      targetShape: targetShape,
      cleanPrefix: clean,
      noisePrefix: noise,
      prefixShape: prefixShape,
      sigma: 0
    )
    let targetTime = targetShape[2]
    let prefixTime = prefixShape[2]
    for channel in 0..<targetShape[1] {
      for token in 0..<targetTime {
        let targetIndex = channel * targetTime + token
        if token < prefixTime {
          let prefixIndex = channel * prefixTime + token
          guard atClean[targetIndex] == clean[prefixIndex] else {
            throw H3NativeError.invalidTensor(
              "clean continuation prefix was not clamped at channel \(channel), token \(token)"
            )
          }
        } else if atClean[targetIndex] != target[targetIndex] {
          throw H3NativeError.invalidTensor(
            "continuation clamp changed a future target token"
          )
        }
      }
    }
  }
}
