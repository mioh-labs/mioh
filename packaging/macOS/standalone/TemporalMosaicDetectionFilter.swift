// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

/// Require evidence from two distinct model observations before a detection
/// becomes a restoration ROI. Carry state across lookahead windows and batches;
/// reused masks cannot count as another observation. Every input frame is emitted
/// in order, even if all its detections are rejected. Only a bounded future window
/// of frames and a bounded history of detection metadata are held.
struct TemporalMosaicDetectionFilter<Frame, Detection> {
  struct Output {
    let frame: Frame
    let detections: [Detection]
    let rejectedCount: Int
    let isObservation: Bool
  }

  private struct Pending {
    let index: Int
    let frame: Frame
    let detections: [Detection]
    let isObservation: Bool
  }

  private struct Observation {
    let index: Int
    let detections: [Detection]
  }

  private let maximumGap: Int
  private let supports: (Detection, Detection) -> Bool
  private var nextIndex = 0
  private var pending: [Pending] = []
  private var history: [Observation] = []

  init(maximumGap: Int, supports: @escaping (Detection, Detection) -> Bool) {
    self.maximumGap = max(1, maximumGap)
    self.supports = supports
  }

  mutating func append(
    _ frame: Frame, detections: [Detection], isObservation: Bool = true
  ) -> Output? {
    pending.append(Pending(index: nextIndex, frame: frame, detections: detections,
      isObservation: isObservation))
    nextIndex += 1
    return pending.count > maximumGap ? emitFirst() : nil
  }

  mutating func finish() -> [Output] {
    var output: [Output] = []
    while !pending.isEmpty { output.append(emitFirst()) }
    return output
  }

  private mutating func emitFirst() -> Output {
    let current = pending.removeFirst()
    history.removeAll { $0.index < current.index - maximumGap }
    if current.isObservation, !current.detections.isEmpty {
      history.append(Observation(index: current.index, detections: current.detections))
    }
    let future = pending.filter { $0.isObservation && !$0.detections.isEmpty }
    let kept = current.detections.filter { candidate in
      var observations = 0
      for observation in history {
        if observation.detections.contains(where: { supports(candidate, $0) }) {
          observations += 1
          if observations >= 2 { return true }
        }
      }
      for observation in future {
        if observation.detections.contains(where: { supports(candidate, $0) }) {
          observations += 1
          if observations >= 2 { return true }
        }
      }
      return false
    }
    return Output(frame: current.frame, detections: kept,
      rejectedCount: current.detections.count - kept.count,
      isObservation: current.isObservation)
  }
}
