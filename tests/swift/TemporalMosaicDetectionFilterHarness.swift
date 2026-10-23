import Foundation

private struct Box {
  let kind: Int
  let left: Int
  let right: Int
}

@main
private enum TemporalMosaicDetectionFilterHarness {
  static func main() throws {
    let a = Box(kind: 0, left: 0, right: 10)
    let b = Box(kind: 0, left: 20, right: 30)
    let otherClass = Box(kind: 1, left: 0, right: 10)
    func run(_ inputs: [[Box]], gap: Int = 1, observations: [Bool]? = nil) -> [[Int]] {
      var filter = TemporalMosaicDetectionFilter<Int, Box>(maximumGap: gap) {
        $0.kind == $1.kind && $0.left <= $1.right && $1.left <= $0.right
      }
      var outputs: [TemporalMosaicDetectionFilter<Int, Box>.Output] = []
      for (index, boxes) in inputs.enumerated() {
        if let output = filter.append(index, detections: boxes,
          isObservation: observations?[index] ?? true)
        {
          outputs.append(output)
        }
        precondition(outputs.count == max(0, index + 1 - max(1, gap)))
      }
      outputs += filter.finish()
      precondition(outputs.map(\.frame) == Array(inputs.indices), "frame order/count changed")
      precondition(filter.finish().isEmpty, "EOF emitted twice")
      return outputs.map { $0.detections.map(\.left) }
    }
    var cases: [String: [[Int]]] = [:]
    cases["empty"] = run([[], [], []])
    cases["only_frame"] = run([[a]])
    cases["first_singleton"] = run([[a], [], []])
    cases["middle_singleton"] = run([[], [a], []])
    cases["last_singleton"] = run([[], [], [a]])
    cases["two_frames"] = run([[a], [a]])
    cases["intermittent_track"] = run([[a], [], [], [a]], gap: 3)
    cases["outside_horizon"] = run([[a], [], [], [], [a]], gap: 3)
    cases["two_boxes_one_observation"] = run([[], [a, a], []])
    cases["different_objects"] = run([[a], [b]])
    cases["different_classes"] = run([[a], [otherClass]])
    cases["one_extra_object"] = run([[a], [a, b], [a]])
    cases["reused_singleton"] = run([[a], [a], [a], []], gap: 3,
      observations: [true, false, false, true])
    cases["confirmed_reuse"] = run([[a], [a], [a], [a]], gap: 3,
      observations: [true, false, false, true])
    var boundary = [[Box]](repeating: [], count: 202)
    boundary[179] = [a]
    boundary[180] = [a]
    let atBoundary = run(boundary, gap: 11)
    cases["batch_boundary"] = Array(atBoundary[178...181])
    let json = try JSONSerialization.data(withJSONObject: cases, options: [.sortedKeys])
    FileHandle.standardOutput.write(json)
  }
}
