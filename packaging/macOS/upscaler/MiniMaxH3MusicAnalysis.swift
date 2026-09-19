import Foundation

struct H3MusicVideoSegment: Sendable, Equatable {
  let startSeconds: Double
  let durationSeconds: Double
  let boundaryStrength: Double
  let normalizedEnergy: Double
}

struct H3MusicVideoChunk: Sendable, Equatable {
  let logicalShotIndex: Int
  let partIndex: Int
  let partCount: Int
  let startSeconds: Double
  let durationSeconds: Double
  let preRollSeconds: Double
  let boundaryStrength: Double
  let normalizedEnergy: Double
}

enum H3MusicVideoTransition: String, Sendable, Equatable {
  case cut
  case `continue`
}

/// One flat generation interval. Scene/group nesting is intentionally absent:
/// the transition alone decides whether native AV state is inherited.
struct H3MusicVideoInterval: Sendable, Equatable {
  let index: Int
  let transition: H3MusicVideoTransition
  let continuesToNext: Bool
  let startSeconds: Double
  let durationSeconds: Double
  let preRollSeconds: Double
  let boundaryStrength: Double
  let normalizedEnergy: Double
}

enum H3MusicVideoBoundary {
  /// A Cut needs enough hidden lead-in for Ref2VA/Qwen identity context to
  /// settle into the requested location before frame zero becomes visible.
  /// Two complete temporal periods cover the measured 0.08–0.80 second
  /// identity-to-scene settling window while preserving H3's 5+17n phase.
  static let cutPreRollFrames = 34

  /// Continue movies retain the last six frames of their protected prefix.
  /// Final assembly overlaps those frames with the preceding movie, so the
  /// total timeline length and master-audio synchronization stay unchanged.
  static let continuationBlendFrames = 6

  static func cutPreRollFrames(
    intervalIndex: Int,
    payloadFrames: Int,
    maximumFrames: Int,
    suppliesContinuation: Bool
  ) -> Int {
    guard intervalIndex > 0, payloadFrames < maximumFrames else { return 0 }
    let capacity = maximumFrames - payloadFrames
    if suppliesContinuation {
      // The payload is already 5+17n aligned. Only whole 17-frame periods may
      // be added without shifting the temporal-tail phase.
      return capacity >= cutPreRollFrames ? cutPreRollFrames : 0
    }
    return min(cutPreRollFrames, capacity)
  }

  static func blendFrames(
    transition: H3MusicVideoTransition,
    preRollFrames: Int,
    overrideFrames: Int? = nil
  ) -> Int {
    guard transition == .continue else { return 0 }
    let requested = overrideFrames ?? continuationBlendFrames
    return min(max(0, requested), max(0, preRollFrames))
  }
}

enum H3MusicVideoAnalyzer {
  private struct FeatureFrame {
    let time: Double
    let energy: Double
    let differenceEnergy: Double
    let zeroCrossingRate: Double
  }

  /// Builds the smallest practical number of shots while moving each cut to a
  /// nearby musical event. Audio is expected in `[1, channels, samples]` form.
  static func timeline(
    audio: H3Tensor,
    sampleRate: Int,
    totalDuration: Double,
    maximumDuration: Double,
    minimumDuration: Double = 2,
    preferredDuration: Double? = nil,
    frameRate: Double = 24
  ) throws -> [H3MusicVideoSegment] {
    guard sampleRate > 0, totalDuration > 0, maximumDuration >= minimumDuration,
      frameRate > 0
    else {
      throw H3NativeError.invalidJob("invalid music-analysis timing")
    }
    guard audio.shape.count == 3, audio.shape[0] == 1,
      audio.shape[1] > 0, audio.shape[2] > 0
    else {
      throw H3NativeError.invalidTensor(
        "music analysis expects [1, channels, samples], got \(audio.shape)"
      )
    }
    let samples = try audio.floatValues()
    let channelCount = audio.shape[1]
    let sampleCount = audio.shape[2]
    let analyzedDuration = min(
      totalDuration,
      Double(sampleCount) / Double(sampleRate)
    )
    let duration = quantize(analyzedDuration, frameRate: frameRate)
    guard duration >= minimumDuration else {
      return [
        H3MusicVideoSegment(
          startSeconds: 0,
          durationSeconds: max(0, duration),
          boundaryStrength: 0,
          normalizedEnergy: 0
        )
      ]
    }

    let hopSamples = max(1, Int((0.05 * Double(sampleRate)).rounded()))
    var mono = [Float](repeating: 0, count: sampleCount)
    for channel in 0..<channelCount {
      let offset = channel * sampleCount
      for sampleIndex in 0..<sampleCount {
        mono[sampleIndex] += samples[offset + sampleIndex]
      }
    }
    let inverseChannels = Float(1) / Float(channelCount)
    for index in mono.indices { mono[index] *= inverseChannels }

    let features = featureFrames(
      mono,
      sampleRate: sampleRate,
      hopSamples: hopSamples,
      duration: duration
    )
    let novelty = noveltyScores(features)
    if duration <= maximumDuration {
      return [
        H3MusicVideoSegment(
          startSeconds: 0,
          durationSeconds: duration,
          boundaryStrength: 0,
          normalizedEnergy: 0.5
        )
      ]
    }

    let optimized = try optimizedBoundaries(
      novelty: novelty,
      features: features,
      duration: duration,
      minimumDuration: minimumDuration,
      maximumDuration: maximumDuration,
      preferredDuration: min(
        maximumDuration,
        max(minimumDuration, preferredDuration ?? min(8.5, maximumDuration))
      ),
      frameRate: frameRate
    )
    let boundaries = optimized.times
    let strengths = optimized.strengths

    var result: [H3MusicVideoSegment] = []
    let shotCount = boundaries.count - 1
    let segmentEnergies = robustUnitScale((0..<shotCount).map { index in
      meanEnergy(features, from: boundaries[index], to: boundaries[index + 1])
    })
    result.reserveCapacity(shotCount)
    for index in 0..<shotCount {
      let start = boundaries[index]
      let end = boundaries[index + 1]
      result.append(
        H3MusicVideoSegment(
          startSeconds: start,
          durationSeconds: end - start,
          boundaryStrength: strengths[index + 1],
          normalizedEnergy: segmentEnergies[index]
        )
      )
    }
    return result
  }

  /// Builds a timeline from composition-change points chosen by the user.
  /// Points are relative to the selected audio start and are snapped to video
  /// frame boundaries. The supplied audio is still analyzed so each shot keeps
  /// the energy value used by the shot designer.
  static func timeline(
    audio: H3Tensor,
    sampleRate: Int,
    totalDuration: Double,
    cutPoints: [Double],
    minimumDuration: Double = 2,
    frameRate: Double = 24
  ) throws -> [H3MusicVideoSegment] {
    guard sampleRate > 0, totalDuration > 0, minimumDuration > 0,
      frameRate > 0
    else {
      throw H3NativeError.invalidJob("invalid manual music timeline")
    }
    guard audio.shape.count == 3, audio.shape[0] == 1,
      audio.shape[1] > 0, audio.shape[2] > 0
    else {
      throw H3NativeError.invalidTensor(
        "music analysis expects [1, channels, samples], got \(audio.shape)"
      )
    }
    let samples = try audio.floatValues()
    let channelCount = audio.shape[1]
    let sampleCount = audio.shape[2]
    let duration = quantize(
      min(totalDuration, Double(sampleCount) / Double(sampleRate)),
      frameRate: frameRate
    )
    guard duration > 0 else {
      throw H3NativeError.invalidJob("manual music timeline is empty")
    }

    let quantizedCuts = Array(
      Set(cutPoints.map { quantize($0, frameRate: frameRate) })
    ).sorted()
    guard quantizedCuts.allSatisfy({ $0 > 0 && $0 < duration }) else {
      throw H3NativeError.invalidJob(
        "manual composition points must be inside the selected audio range"
      )
    }
    let boundaries = [0.0] + quantizedCuts + [duration]
    for index in 0..<(boundaries.count - 1) {
      guard boundaries[index + 1] - boundaries[index]
        >= minimumDuration - 0.5 / frameRate
      else {
        throw H3NativeError.invalidJob(
          "manual composition points and audio edges must be at least \(minimumDuration) seconds apart"
        )
      }
    }

    let hopSamples = max(1, Int((0.05 * Double(sampleRate)).rounded()))
    var mono = [Float](repeating: 0, count: sampleCount)
    for channel in 0..<channelCount {
      let offset = channel * sampleCount
      for sampleIndex in 0..<sampleCount {
        mono[sampleIndex] += samples[offset + sampleIndex]
      }
    }
    let inverseChannels = Float(1) / Float(channelCount)
    for index in mono.indices { mono[index] *= inverseChannels }
    let features = featureFrames(
      mono,
      sampleRate: sampleRate,
      hopSamples: hopSamples,
      duration: duration
    )
    let novelty = noveltyScores(features)
    let segmentEnergies = robustUnitScale((0..<(boundaries.count - 1)).map {
      meanEnergy(features, from: boundaries[$0], to: boundaries[$0 + 1])
    })

    return (0..<(boundaries.count - 1)).map { index in
      H3MusicVideoSegment(
        startSeconds: boundaries[index],
        durationSeconds: boundaries[index + 1] - boundaries[index],
        boundaryStrength: index == 0
          ? 0
          : noveltyStrength(
            at: boundaries[index],
            novelty: novelty,
            features: features
          ),
        normalizedEnergy: segmentEnergies[index]
      )
    }
  }

  static func fixedTimeline(
    totalDuration: Double,
    duration: Double,
    frameRate: Double = 24
  ) -> [H3MusicVideoSegment] {
    let total = quantize(totalDuration, frameRate: frameRate)
    guard total > 0, duration > 0 else { return [] }
    let nominalFrames = max(1, Int((duration * frameRate).rounded()))
    let minimumFrames = max(1, Int((2 * frameRate).rounded()))
    let totalFrames = max(1, Int((total * frameRate).rounded()))
    var frameCounts = Array(
      repeating: nominalFrames,
      count: totalFrames / nominalFrames
    )
    let remainder = totalFrames % nominalFrames
    if remainder > 0 {
      if remainder < minimumFrames, !frameCounts.isEmpty {
        frameCounts[frameCounts.count - 1] += remainder
      } else {
        frameCounts.append(remainder)
      }
    }
    if frameCounts.isEmpty { frameCounts = [totalFrames] }
    var startFrame = 0
    return frameCounts.map { frameCount in
      defer { startFrame += frameCount }
      return H3MusicVideoSegment(
        startSeconds: Double(startFrame) / frameRate,
        durationSeconds: Double(frameCount) / frameRate,
        boundaryStrength: 0,
        normalizedEnergy: 0.5
      )
    }
  }

  static func generationChunks(
    _ segments: [H3MusicVideoSegment],
    maximumDuration: Double = 10,
    continuationPreRollSeconds: Double = 0,
    cutPreRollSeconds: Double = 0,
    frameRate: Double = 24
  ) throws -> [H3MusicVideoChunk] {
    guard maximumDuration > 0, continuationPreRollSeconds >= 0,
      cutPreRollSeconds >= 0,
      continuationPreRollSeconds < maximumDuration, frameRate > 0
    else {
      throw H3NativeError.invalidJob("invalid music-video chunk timing")
    }
    let maximumFrames = max(
      1,
      Int((maximumDuration * frameRate).rounded(.down))
    )
    let preRollFrames = Int(
      (continuationPreRollSeconds * frameRate).rounded()
    )
    let requestedCutPreRollFrames = Int(
      (cutPreRollSeconds * frameRate).rounded()
    )
    let continuationPayloadFrames = maximumFrames - preRollFrames
    guard continuationPayloadFrames > 0 else {
      throw H3NativeError.invalidJob("continuation pre-roll consumes the chunk")
    }
    var chunks: [H3MusicVideoChunk] = []
    for (logicalShotIndex, segment) in segments.enumerated() {
      let segmentFrames = max(
        1,
        Int((segment.durationSeconds * frameRate).rounded())
      )
      let startFrame = Int((segment.startSeconds * frameRate).rounded())
      let minimumPartFrames = max(1, Int((2 * frameRate).rounded()))
      var partFrameCounts: [Int] = []
      var remainingFrames = segmentFrames
      func partPreRollFrames(_ partIndex: Int, payloadFrames: Int? = nil,
                             suppliesContinuation: Bool = true) -> Int {
        guard partIndex == 0 else { return preRollFrames }
        guard logicalShotIndex > 0, requestedCutPreRollFrames > 0 else {
          return 0
        }
        if let payloadFrames {
          return H3MusicVideoBoundary.cutPreRollFrames(
            intervalIndex: logicalShotIndex,
            payloadFrames: payloadFrames,
            maximumFrames: maximumFrames,
            suppliesContinuation: suppliesContinuation
          )
        }
        return min(requestedCutPreRollFrames, H3MusicVideoBoundary.cutPreRollFrames)
      }
      // First determine the minimum feasible Part count. These provisional
      // values are replaced by a balanced allocation below; only the count is
      // retained so balancing can never add another full H3 invocation.
      while true {
        let partIndex = partFrameCounts.count
        let partPreRollFrames = partPreRollFrames(partIndex)
        let capacity = maximumFrames - partPreRollFrames
        guard capacity >= minimumPartFrames else {
          throw H3NativeError.media(
            "music-video continuation capacity is below two seconds"
          )
        }
        if remainingFrames <= capacity {
          partFrameCounts.append(remainingFrames)
          break
        }

        // Every Part that supplies a temporal tail must end exactly on H3's
        // 5+17n decoded-frame boundary. Otherwise the generated latent extends
        // beyond the movie frames and continuation jumps into unseen future.
        var payload = capacity
        while (payload + partPreRollFrames) % 17 != 5 { payload -= 1 }
        while remainingFrames - payload < minimumPartFrames {
          payload -= 17
        }
        guard payload >= minimumPartFrames else {
          throw H3NativeError.media(
            "logical shot cannot be split on an H3 temporal boundary"
          )
        }
        partFrameCounts.append(payload)
        remainingFrames -= payload
      }
      let partCount = partFrameCounts.count
      if partCount > 1 {
        let minimumContinuationPayload = max(
          minimumPartFrames,
          {
            var value = minimumPartFrames
            while (value + preRollFrames) % 17 != 5 { value += 1 }
            return value
          }()
        )
        var maximumContinuationPayload = continuationPayloadFrames
        while (maximumContinuationPayload + preRollFrames) % 17 != 5 {
          maximumContinuationPayload -= 1
        }
        guard minimumContinuationPayload <= maximumContinuationPayload else {
          throw H3NativeError.media(
            "music-video continuation has no supported H3 temporal boundary"
          )
        }

        partFrameCounts.removeAll(keepingCapacity: true)
        remainingFrames = segmentFrames
        for partIndex in 0..<(partCount - 1) {
          let partPreRollFrames = partPreRollFrames(partIndex)
          let capacity = maximumFrames - partPreRollFrames
          let remainingPartCount = partCount - partIndex
          let ideal = remainingFrames / remainingPartCount
          var payload = min(capacity, ideal)
          while payload >= minimumPartFrames,
            (payload + partPreRollFrames) % 17 != 5
          {
            payload -= 1
          }

          let futureNonFinalCount = partCount - partIndex - 2
          let maximumFutureFrames = futureNonFinalCount
            * maximumContinuationPayload + continuationPayloadFrames
          let minimumFutureFrames = futureNonFinalCount
            * minimumContinuationPayload + minimumPartFrames
          while remainingFrames - payload > maximumFutureFrames,
            payload + 17 <= capacity
          {
            payload += 17
          }
          while remainingFrames - payload < minimumFutureFrames,
            payload - 17 >= minimumPartFrames
          {
            payload -= 17
          }
          guard payload >= minimumPartFrames,
            payload <= capacity,
            (payload + partPreRollFrames) % 17 == 5,
            remainingFrames - payload >= minimumFutureFrames,
            remainingFrames - payload <= maximumFutureFrames
          else {
            throw H3NativeError.media(
              "logical shot cannot be balanced on H3 temporal boundaries"
            )
          }
          partFrameCounts.append(payload)
          remainingFrames -= payload
        }
        partFrameCounts.append(remainingFrames)
      }
      var relativeStart = 0
      for partIndex in 0..<partCount {
        let partFrames = partFrameCounts[partIndex]
        let partPreRollFrames = partPreRollFrames(
          partIndex,
          payloadFrames: partFrames,
          suppliesContinuation: partIndex + 1 < partCount
        )
        guard partFrames > 0,
          partFrames + partPreRollFrames <= maximumFrames
        else {
          throw H3NativeError.media(
            "logical shot could not be divided into supported H3 chunks"
          )
        }
        chunks.append(
          H3MusicVideoChunk(
            logicalShotIndex: logicalShotIndex,
            partIndex: partIndex,
            partCount: partCount,
            startSeconds: Double(startFrame + relativeStart) / frameRate,
            durationSeconds: Double(partFrames) / frameRate,
            preRollSeconds: Double(partPreRollFrames) / frameRate,
            boundaryStrength: segment.boundaryStrength,
            normalizedEnergy: segment.normalizedEnergy
          )
        )
        relativeStart += partFrames
      }
    }
    return chunks
  }

  /// Flattens the legacy duration balancer into an execution timeline. The
  /// old local indices are used only to calculate legal H3 frame lengths and
  /// never reach prompts, progress text, filenames, or model conditioning.
  static func generationIntervals(
    _ segments: [H3MusicVideoSegment],
    maximumDuration: Double = 10,
    continuationPreRollSeconds: Double = 0,
    cutPreRollSeconds: Double = 0,
    frameRate: Double = 24
  ) throws -> [H3MusicVideoInterval] {
    let balanced = try generationChunks(
      segments,
      maximumDuration: maximumDuration,
      continuationPreRollSeconds: continuationPreRollSeconds,
      cutPreRollSeconds: cutPreRollSeconds,
      frameRate: frameRate
    )
    return balanced.enumerated().map { index, chunk in
      H3MusicVideoInterval(
        index: index,
        transition: chunk.partIndex == 0 ? .cut : .continue,
        continuesToNext: chunk.partIndex + 1 < chunk.partCount,
        startSeconds: chunk.startSeconds,
        durationSeconds: chunk.durationSeconds,
        preRollSeconds: chunk.preRollSeconds,
        boundaryStrength: chunk.boundaryStrength,
        normalizedEnergy: chunk.normalizedEnergy
      )
    }
  }

  private static func featureFrames(
    _ samples: [Float],
    sampleRate: Int,
    hopSamples: Int,
    duration: Double
  ) -> [FeatureFrame] {
    let usableSamples = min(
      samples.count,
      Int((duration * Double(sampleRate)).rounded(.up))
    )
    guard usableSamples > 0 else { return [] }
    var result: [FeatureFrame] = []
    result.reserveCapacity((usableSamples + hopSamples - 1) / hopSamples)
    var lower = 0
    while lower < usableSamples {
      let upper = min(usableSamples, lower + hopSamples)
      var squareSum = 0.0
      var differenceSquareSum = 0.0
      var crossings = 0
      var previous = Double(samples[lower])
      for index in lower..<upper {
        let value = Double(samples[index])
        squareSum += value * value
        if index > lower {
          let difference = value - previous
          differenceSquareSum += difference * difference
          if (value >= 0) != (previous >= 0) { crossings += 1 }
        }
        previous = value
      }
      let count = max(1, upper - lower)
      result.append(
        FeatureFrame(
          time: Double(lower + count / 2) / Double(sampleRate),
          energy: log10(1e-8 + sqrt(squareSum / Double(count))),
          differenceEnergy: log10(
            1e-8 + sqrt(differenceSquareSum / Double(max(1, count - 1)))
          ),
          zeroCrossingRate: Double(crossings) / Double(max(1, count - 1))
        )
      )
      lower = upper
    }
    return result
  }

  private static func noveltyScores(_ features: [FeatureFrame]) -> [Double] {
    guard features.count > 2 else { return [Double](repeating: 0, count: features.count) }
    var raw = [Double](repeating: 0, count: features.count)
    for index in 1..<features.count {
      let current = features[index]
      let previous = features[index - 1]
      let onset = max(0, current.energy - previous.energy)
      let energyChange = abs(current.energy - previous.energy)
      let timbreChange = abs(current.differenceEnergy - previous.differenceEnergy)
        + 0.6 * abs(current.zeroCrossingRate - previous.zeroCrossingRate)
      let radius = 4
      let beforeLower = max(0, index - radius)
      let afterUpper = min(features.count, index + radius + 1)
      let surrounding = features[beforeLower..<afterUpper]
        .map(\.energy).reduce(0, +) / Double(afterUpper - beforeLower)
      let quietBreak = max(0, surrounding - current.energy)
      raw[index] = 0.35 * onset + 0.25 * energyChange
        + 0.25 * timbreChange + 0.15 * quietBreak
    }
    // A broad local maximum represents one musical event. Suppress adjacent
    // 50 ms peaks so a drum transient is not counted as several boundaries.
    var peaked = raw
    for index in raw.indices {
      let lower = max(0, index - 3)
      let upper = min(raw.count, index + 4)
      peaked[index] = raw[lower..<upper].max() ?? raw[index]
    }
    return robustUnitScale(peaked)
  }

  private static func robustUnitScale(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }
    let sorted = values.sorted()
    let low = sorted[Int(Double(sorted.count - 1) * 0.20)]
    let high = sorted[Int(Double(sorted.count - 1) * 0.95)]
    let scale = max(1e-9, high - low)
    return values.map { min(1, max(0, ($0 - low) / scale)) }
  }

  private static func optimizedBoundaries(
    novelty: [Double],
    features: [FeatureFrame],
    duration: Double,
    minimumDuration: Double,
    maximumDuration: Double,
    preferredDuration: Double,
    frameRate: Double
  ) throws -> (times: [Double], strengths: [Double]) {
    let totalFrames = max(1, Int((duration * frameRate).rounded()))
    let minimumFrames = max(1, Int((minimumDuration * frameRate).rounded(.up)))
    let maximumFrames = max(
      minimumFrames,
      Int((maximumDuration * frameRate).rounded(.down))
    )
    let firstFeatureTime = features.first?.time ?? 0
    let featureHop = features.count > 1
      ? max(1e-6, features[1].time - features[0].time)
      : 0.05
    func strength(atFrame frame: Int) -> Double {
      guard !novelty.isEmpty else { return 0 }
      let time = Double(frame) / frameRate
      let index = min(
        novelty.count - 1,
        max(0, Int(((time - firstFeatureTime) / featureHop).rounded()))
      )
      return novelty[index]
    }

    var scores = [Double](repeating: -Double.infinity, count: totalFrames + 1)
    var predecessors = [Int](repeating: -1, count: totalFrames + 1)
    scores[0] = 0
    if totalFrames >= minimumFrames {
      for end in minimumFrames...totalFrames {
        let shortest = max(0, end - maximumFrames)
        let latest = end - minimumFrames
        guard shortest <= latest else { continue }
        for start in shortest...latest where scores[start].isFinite {
          let segmentDuration = Double(end - start) / frameRate
          // An additional shot costs one point. Only a genuinely strong
          // structural event can justify increasing the minimum shot count.
          let shotCost = 1.15
          let durationPenalty = 0.025
            * abs(segmentDuration - preferredDuration)
          let boundaryReward = end == totalFrames
            ? 0
            : 1.25 * strength(atFrame: end)
          let score = scores[start] + boundaryReward - shotCost
            - durationPenalty
          if score > scores[end] {
            scores[end] = score
            predecessors[end] = start
          }
        }
      }
    }
    guard predecessors[totalFrames] >= 0 else {
      throw H3NativeError.media(
        "music duration cannot be divided into supported shot lengths"
      )
    }
    var reversedFrames = [totalFrames]
    var cursor = totalFrames
    while cursor > 0 {
      cursor = predecessors[cursor]
      guard cursor >= 0 else {
        throw H3NativeError.media("music timeline reconstruction failed")
      }
      reversedFrames.append(cursor)
    }
    let frames = reversedFrames.reversed()
    let times = frames.map { Double($0) / frameRate }
    let strengths = frames.map { frame in
      frame == 0 || frame == totalFrames ? 0 : strength(atFrame: frame)
    }
    return (times, strengths)
  }

  private static func meanEnergy(
    _ features: [FeatureFrame],
    from start: Double,
    to end: Double
  ) -> Double {
    let selected = features.filter { $0.time >= start && $0.time < end }
    guard !selected.isEmpty else { return 0 }
    return selected.map(\.energy).reduce(0, +) / Double(selected.count)
  }

  private static func noveltyStrength(
    at time: Double,
    novelty: [Double],
    features: [FeatureFrame]
  ) -> Double {
    guard !novelty.isEmpty, let firstTime = features.first?.time else {
      return 0
    }
    let hop = features.count > 1
      ? max(1e-6, features[1].time - features[0].time)
      : 0.05
    let index = min(
      novelty.count - 1,
      max(0, Int(((time - firstTime) / hop).rounded()))
    )
    return novelty[index]
  }

  private static func quantize(_ time: Double, frameRate: Double) -> Double {
    (time * frameRate).rounded() / frameRate
  }
}

enum H3MusicVideoShotDesigner {
  static func design(
    index: Int,
    count: Int,
    normalizedEnergy: Double
  ) -> String {
    let framings = [
      "an expansive environmental wide shot with the subject small against the landscape",
      "a full-body composition with generous headroom and a clearly readable path",
      "a medium-full composition that preserves footsteps, clothing movement, and surroundings",
      "a waist-up composition with the face and natural singing articulation clearly readable",
      "a stable chest-up portrait composition with layered background depth",
      "an intimate facial close-up with both eyes and mouth kept sharply readable",
      "a profile medium shot with strong negative space in the direction of travel",
      "an over-the-shoulder medium-wide view that reveals the destination ahead",
      "a symmetrical centered full-body tableau with architectural depth",
      "an asymmetrical rule-of-thirds medium-full composition with foreground layering",
      "a distant silhouette-like wide composition with a long visible horizon",
    ]
    let angles = [
      "at eye level from the front",
      "from a restrained low three-quarter angle",
      "from a gentle high three-quarter angle",
      "in a clean side profile",
      "from behind at a three-quarter angle",
      "from the front-left diagonal",
      "from the front-right diagonal",
      "along the depth axis of the location",
      "from a slightly canted but controlled editorial angle",
    ]
    let movements = [
      "with a slow lateral dolly",
      "with a smooth parallel tracking move",
      "with a subtle backward tracking move that never becomes a zoom",
      "with a restrained forward dolly that ends before changing shot size",
      "with a slow clockwise arc",
      "with a slow counter-clockwise arc",
      "with a locked-off camera and movement only inside the frame",
      "with a stabilized walking-camera feel",
      "with a gentle crane rise",
      "with a gentle crane descent",
      "with a compressed telephoto pan",
      "with a quiet handheld drift and no sudden reframing",
      "with a static start followed by one deliberate lateral reveal",
    ]
    let blocking = [
      "The subject travels left-to-right without reversing direction.",
      "The subject travels right-to-left without reversing direction.",
      "The subject approaches diagonally while remaining in the same shot size.",
      "The subject moves away diagonally and briefly looks back toward camera.",
      "The subject remains near center while the background supplies motion.",
      "The subject enters from foreground and settles in the middle distance.",
      "The subject begins in profile and naturally turns toward camera.",
    ]
    // These mutually prime cycle lengths make the full composition unique for
    // 9,009 consecutive shots while varying every visual dimension early.
    let framing = framings[index % framings.count]
    let angle = angles[(index * 5) % angles.count]
    let movement = movements[(index * 7) % movements.count]
    let staging = blocking[(index * 3) % blocking.count]
    let energy = normalizedEnergy >= 0.67
      ? "Match the high musical energy with confident subject motion, but keep the camera controlled."
      : normalizedEnergy <= 0.33
        ? "Match the quiet musical energy with restrained motion and a longer visual hold."
        : "Match the moderate musical energy with fluid, unhurried motion."
    return "Shot \(index + 1) of \(count): use \(framing), \(angle), \(movement). \(staging) \(energy)"
  }
}
