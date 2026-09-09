import Foundation

@available(macOS 27.0, *)
@main
struct MiniMaxH3AudioSeekProbe {
  static func main() async throws {
    guard CommandLine.arguments.count == 2 else {
      throw NSError(
        domain: "MiniMaxH3AudioSeekProbe",
        code: 64,
        userInfo: [
          NSLocalizedDescriptionKey:
            "usage: MiniMaxH3AudioSeekProbe <audio-file>"
        ]
      )
    }
    let url = URL(fileURLWithPath: CommandLine.arguments[1])
      .standardizedFileURL
    let continuous = try await H3NativeMedia.decodeReferenceAudio(
      url: url,
      durationSeconds: 20,
      startSeconds: 0
    )
    let seeked = try await H3NativeMedia.decodeReferenceAudio(
      url: url,
      durationSeconds: 10,
      startSeconds: 10
    )
    let continuousValues = try continuous.floatValues()
    let seekedValues = try seeked.floatValues()
    let fullFrames = continuous.shape[2]
    let seekFrames = seeked.shape[2]
    guard fullFrames == seekFrames * 2 else {
      throw NSError(
        domain: "MiniMaxH3AudioSeekProbe",
        code: 65,
        userInfo: [NSLocalizedDescriptionKey: "unexpected audio shapes"]
      )
    }
    var squaredError = 0.0
    var referenceEnergy = 0.0
    var dot = 0.0
    var seekEnergy = 0.0
    for channel in 0..<2 {
      let fullBase = channel * fullFrames + seekFrames
      let seekBase = channel * seekFrames
      for frame in 0..<seekFrames {
        let expected = Double(continuousValues[fullBase + frame])
        let actual = Double(seekedValues[seekBase + frame])
        let difference = actual - expected
        squaredError += difference * difference
        referenceEnergy += expected * expected
        seekEnergy += actual * actual
        dot += expected * actual
      }
    }
    let count = Double(seekFrames * 2)
    let rmse = sqrt(squaredError / count)
    let correlation = dot / sqrt(referenceEnergy * seekEnergy)
    FileHandle.standardError.write(
      Data("correlation=\(correlation) rmse=\(rmse)\n".utf8)
    )
    guard correlation >= 0.9999, rmse <= 1e-5 else {
      throw NSError(
        domain: "MiniMaxH3AudioSeekProbe",
        code: 66,
        userInfo: [
          NSLocalizedDescriptionKey:
            "seeked audio does not match the continuous timeline"
        ]
      )
    }
  }
}
