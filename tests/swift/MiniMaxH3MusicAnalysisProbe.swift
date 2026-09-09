import Foundation

@main
struct MiniMaxH3MusicAnalysisProbe {
  static func main() throws {
    let sampleRate = 1_000
    let duration = 30
    var mono = [Float](repeating: 0, count: sampleRate * duration)
    for index in mono.indices {
      let time = Double(index) / Double(sampleRate)
      let frequency: Double
      let amplitude: Double
      if time < 10 {
        frequency = 5
        amplitude = 0.08
      } else if time < 20 {
        frequency = 27
        amplitude = 0.30
      } else {
        frequency = 11
        amplitude = 0.14
      }
      mono[index] = Float(amplitude * sin(2 * .pi * frequency * time))
    }
    let stereo = mono + mono
    let tensor = try H3Tensor(
      float32: stereo,
      shape: [1, 2, mono.count]
    )
    let timeline = try H3MusicVideoAnalyzer.timeline(
      audio: tensor,
      sampleRate: sampleRate,
      totalDuration: Double(duration),
      maximumDuration: 10
    )
    let cuts = timeline.dropLast().map { $0.startSeconds + $0.durationSeconds }
    guard cuts.contains(where: { abs($0 - 10) <= 0.6 }),
      cuts.contains(where: { abs($0 - 20) <= 0.6 })
    else {
      fatalError("musical cuts were not recovered: \(cuts)")
    }
    guard timeline.allSatisfy({ $0.durationSeconds >= 2 && $0.durationSeconds <= 10 })
    else {
      fatalError("shot duration escaped the supported range: \(timeline)")
    }
    let total = timeline.map(\.durationSeconds).reduce(0, +)
    guard abs(total - Double(duration)) < 0.001 else {
      fatalError("timeline duration mismatch: \(total)")
    }
    let manual = try H3MusicVideoAnalyzer.timeline(
      audio: tensor,
      sampleRate: sampleRate,
      totalDuration: Double(duration),
      cutPoints: [7.23, 19.77]
    )
    let manualCuts = manual.dropLast().map {
      $0.startSeconds + $0.durationSeconds
    }
    guard manual.count == 3,
      abs(manualCuts[0] - 7.25) < 0.001,
      abs(manualCuts[1] - 19.75) < 0.001,
      manual.allSatisfy({ $0.durationSeconds >= 2 })
    else {
      fatalError("manual composition points were not preserved: \(manual)")
    }
    let designs = (0..<500).map {
      H3MusicVideoShotDesigner.design(
        index: $0,
        count: 500,
        normalizedEnergy: Double($0 % 3) / 2
      )
    }
    guard Set(designs).count == designs.count else {
      fatalError("shot designs repeated")
    }
    let longShot = H3MusicVideoSegment(
      startSeconds: 4,
      durationSeconds: 23,
      boundaryStrength: 1,
      normalizedEnergy: 0.5
    )
    let chunks = try H3MusicVideoAnalyzer.generationChunks([longShot])
    guard chunks.count == 3,
      chunks.allSatisfy({ $0.durationSeconds + $0.preRollSeconds <= 10 }),
      chunks.allSatisfy({ $0.logicalShotIndex == 0 }),
      chunks.allSatisfy({ $0.preRollSeconds == 0 }),
      chunks.dropLast().allSatisfy({
        Int(($0.durationSeconds * 24).rounded()) % 17 == 5
      }),
      abs(chunks.map(\.durationSeconds).reduce(0, +) - 23) < 0.001
    else {
      fatalError("23-second logical shot was not continued correctly: \(chunks)")
    }
    let prefixPreRollSeconds = Double(
      H3VideoConditioning.partContinuationPixelFrames
    ) / Double(H3Geometry.framesPerSecond)
    let prefixedChunks = try H3MusicVideoAnalyzer.generationChunks(
      [longShot],
      continuationPreRollSeconds: prefixPreRollSeconds
    )
    guard prefixedChunks.count >= 3,
      prefixedChunks.first?.preRollSeconds == 0,
      prefixedChunks.dropFirst().allSatisfy({
        Int(($0.preRollSeconds * 24).rounded())
          == H3VideoConditioning.partContinuationPixelFrames
      }),
      prefixedChunks.allSatisfy({
        $0.durationSeconds + $0.preRollSeconds <= 10
      }),
      prefixedChunks.dropLast().allSatisfy({
        Int((($0.durationSeconds + $0.preRollSeconds) * 24).rounded())
          % 17 == 5
      }),
      abs(prefixedChunks.map(\.durationSeconds).reduce(0, +) - 23) < 0.001,
      H3Geometry.videoLatentFrames(
        pixelFrames: H3VideoConditioning.partContinuationPixelFrames
      ) == H3VideoConditioning.partContinuationTokens
    else {
      fatalError(
        "fixed-prefix Part allocation regressed: \(prefixedChunks)"
      )
    }
    let cutPreRollSeconds = Double(H3MusicVideoBoundary.cutPreRollFrames) / 24
    let cutIntervals = try H3MusicVideoAnalyzer.generationIntervals(
      [
        H3MusicVideoSegment(
          startSeconds: 0,
          durationSeconds: 8,
          boundaryStrength: 0,
          normalizedEnergy: 0.5
        ),
        H3MusicVideoSegment(
          startSeconds: 8,
          durationSeconds: 15,
          boundaryStrength: 1,
          normalizedEnergy: 0.5
        ),
      ],
      continuationPreRollSeconds: prefixPreRollSeconds,
      cutPreRollSeconds: cutPreRollSeconds
    )
    guard cutIntervals.first?.preRollSeconds == 0,
      Int((cutIntervals[1].preRollSeconds * 24).rounded())
        == H3MusicVideoBoundary.cutPreRollFrames,
      cutIntervals[1].transition == .cut,
      cutIntervals[2].transition == .continue,
      cutIntervals.dropLast().allSatisfy({
        Int((($0.durationSeconds + $0.preRollSeconds) * 24).rounded())
          % 17 == 5
      }),
      H3MusicVideoBoundary.blendFrames(
        transition: .continue,
        preRollFrames: H3VideoConditioning.partContinuationPixelFrames
      ) == H3MusicVideoBoundary.continuationBlendFrames,
      H3MusicVideoBoundary.blendFrames(
        transition: .cut,
        preRollFrames: H3MusicVideoBoundary.cutPreRollFrames
      ) == 0
    else {
      fatalError("Cut pre-roll or Continue overlap regressed: \(cutIntervals)")
    }
    func payloadFrames(
      _ segmentFrames: Int,
      maximumFrames: Int
    ) throws -> [Int] {
      let chunks = try H3MusicVideoAnalyzer.generationChunks(
        [
          H3MusicVideoSegment(
            startSeconds: 0,
            durationSeconds: Double(segmentFrames) / 24,
            boundaryStrength: 0,
            normalizedEnergy: 0.5
          )
        ],
        maximumDuration: Double(maximumFrames) / 24
      )
      return chunks.map { Int(($0.durationSeconds * 24).rounded()) }
    }
    let balancedAuto274 = try payloadFrames(274, maximumFrames: 240)
    let balancedAuto288 = try payloadFrames(288, maximumFrames: 240)
    let balancedAuto720 = try payloadFrames(720, maximumFrames: 240)
    let balanced1080p480 = try payloadFrames(480, maximumFrames: 144)
    guard balancedAuto274 == [124, 150],
      balancedAuto288 == [141, 147],
      balancedAuto720 == [175, 175, 175, 195],
      balanced1080p480 == [107, 124, 124, 125]
    else {
      fatalError(
        "balanced Part allocation regressed: "
          + "\(balancedAuto274), \(balancedAuto288), "
          + "\(balancedAuto720), \(balanced1080p480)"
      )
    }
    for maximumFrames in [240, 144] {
      for segmentFrames in 48...4_319 {
        let frames = try payloadFrames(
          segmentFrames,
          maximumFrames: maximumFrames
        )
        let maximumContinuationFrames = stride(
          from: maximumFrames,
          through: 48,
          by: -1
        ).first(where: { $0 % 17 == 5 })!
        let minimumPartCount = segmentFrames <= maximumFrames
          ? 1
          : Int(
            ceil(
              Double(segmentFrames - maximumFrames)
                / Double(maximumContinuationFrames)
            )
          ) + 1
        guard frames.count == minimumPartCount,
          frames.reduce(0, +) == segmentFrames,
          frames.allSatisfy({ $0 >= 48 && $0 <= maximumFrames }),
          frames.dropLast().allSatisfy({ $0 % 17 == 5 })
        else {
          fatalError(
            "balanced Part invariants failed for segment=\(segmentFrames), "
              + "maximum=\(maximumFrames): \(frames)"
          )
        }
      }
    }
    for maximumFrames in [240, 144] {
      for segmentFrames in 48...4_319 {
        let prefixed = try H3MusicVideoAnalyzer.generationChunks(
          [
            H3MusicVideoSegment(
              startSeconds: 0,
              durationSeconds: Double(segmentFrames) / 24,
              boundaryStrength: 0,
              normalizedEnergy: 0.5
            )
          ],
          maximumDuration: Double(maximumFrames) / 24,
          continuationPreRollSeconds: prefixPreRollSeconds
        )
        let payloads = prefixed.map {
          Int(($0.durationSeconds * 24).rounded())
        }
        let generated = prefixed.map {
          Int((($0.durationSeconds + $0.preRollSeconds) * 24).rounded())
        }
        guard payloads.reduce(0, +) == segmentFrames,
          prefixed.first?.preRollSeconds == 0,
          prefixed.dropFirst().allSatisfy({
            Int(($0.preRollSeconds * 24).rounded())
              == H3VideoConditioning.partContinuationPixelFrames
          }),
          payloads.allSatisfy({ $0 >= 48 }),
          generated.allSatisfy({ $0 <= maximumFrames }),
          generated.dropLast().allSatisfy({ $0 % 17 == 5 })
        else {
          fatalError(
            "fixed-prefix Part invariants failed for segment=\(segmentFrames), "
              + "maximum=\(maximumFrames): \(prefixed)"
          )
        }
      }
    }
    let fixed = H3MusicVideoAnalyzer.fixedTimeline(
      totalDuration: 18.04,
      duration: 6
    )
    guard fixed.count == 3,
      fixed.allSatisfy({ $0.durationSeconds >= 2 }),
      abs(fixed.map(\.durationSeconds).reduce(0, +) - 18.041_666_666_7) < 0.001
    else {
      fatalError("fixed timeline left an unusable tail shot: \(fixed)")
    }
    let officialChunks = try H3MusicVideoAnalyzer.generationChunks(
      [
        H3MusicVideoSegment(
          startSeconds: 0,
          durationSeconds: 20,
          boundaryStrength: 0,
          normalizedEnergy: 0.5
        )
      ],
      maximumDuration: 6
    )
    guard officialChunks.count == 4,
      officialChunks.allSatisfy({ $0.durationSeconds <= 6 }),
      officialChunks.dropLast().allSatisfy({
        Int(($0.durationSeconds * 24).rounded()) % 17 == 5
      }),
      abs(officialChunks.map(\.durationSeconds).reduce(0, +) - 20) < 0.001
    else {
      fatalError("1080p chunks escaped visible H3 boundaries: \(officialChunks)")
    }
    print("ok: cuts=\(cuts), uniqueDesigns=\(Set(designs).count)")
  }
}
