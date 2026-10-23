import Foundation

@main
enum DetectionMaskJSONHarness {
  static func main() throws {
    func project(_ values: [Float], mw: Int = 4, mh: Int = 2,
                 sw: Int = 4, sh: Int = 2, rect: [Int] = [0, 0, 4, 2],
                 box: [Int] = [0, 0, 4, 2], threshold: Float = 0.5) throws -> DetectionMaskJSONRegion {
      try DetectionMaskJSONRegion.project(
        values: values, maskWidth: mw, maskHeight: mh, sourceWidth: sw, sourceHeight: sh,
        samplingRect: rect, bbox: box, threshold: threshold, confidence: 0.9, classIndex: 0)
    }
    func unpack(_ region: DetectionMaskJSONRegion) -> [Int] {
      region.mask.counts.enumerated().flatMap { index, length in
        Array(repeating: index % 2, count: length)
      }
    }
    let pattern: [Float] = [1, 0, 1, 1, 0, 0, 0, 1]
    let region = try project(pattern)
    assert(region.mask.counts == [0, 1, 1, 2, 3, 1])
    assert(unpack(region) == pattern.map(Int.init))
    let empty = try project(Array(repeating: 0.5, count: 8))
    assert(empty.mask.counts == [8]) // strict >, not >=
    let full = try project(Array(repeating: 1, count: 8))
    assert(full.mask.counts == [0, 8])
    let cropped = try project(pattern, box: [2, 0, 2, 2])
    assert(cropped.bbox == [2, 0, 2, 2])
    assert(unpack(cropped) == [1, 1, 0, 1])
    let padded = [Float](repeating: 1, count: 4) + pattern + [Float](repeating: 1, count: 4)
    let letterbox = try project(padded, mh: 4, rect: [0, 1, 4, 2])
    assert(unpack(letterbox) == unpack(region))
    let resized = try project([1, 0, 0, 0], mw: 2, mh: 2, sw: 4, sh: 4,
                              rect: [0, 0, 2, 2], box: [0, 0, 4, 4])
    assert(unpack(resized) == [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    do { _ = try project([]); assertionFailure("bad mask accepted") }
    catch DetectionMaskJSONError.invalidMask {}

    let field = LabelFieldFrame(gridWidth: 3, gridHeight: 2, cellSize: 4,
      paste: [0, 7, 7, 0, 9, 0], coverage: [0, 1, 0.5, 0, 0.25, 0])
    let filled = field.filling([9: [(cell: 5, weight: 0.75)]])
    let celf = DetectionMaskJSONCELF.capture(filled, width: 10, height: 7)!
    assert(celf.grid_size == [2, 3] && celf.cell_size == 4)
    assert(celf.stored_rect == [1, 0, 2, 2])
    func expand(_ runs: DetectionMaskJSONCELF.Runs) -> [Int] {
      zip(runs.values, runs.lengths).flatMap { value, length in
        Array(repeating: value, count: length)
      }
    }
    assert(expand(celf.owners) == [7, 7, 9, 9])
    assert(expand(celf.coverage_u8) == [255, 128, 64, 191])
    assert(celf.regions.map(\.track_id) == [7, 9])
    assert(celf.regions[0].bbox == [4, 0, 6, 4])
    assert(celf.regions[1].bbox == [4, 4, 6, 3])
    assert(field.owner(cellX: 2, cellY: 1) == 0) // original is not mutated
    assert(DetectionMaskJSONCELF.capture(nil, width: 10, height: 7) == nil)
    let emptyField = LabelFieldFrame(gridWidth: 3, gridHeight: 2, cellSize: 4,
      paste: [Int32](repeating: 0, count: 6), coverage: [Float](repeating: 0, count: 6))
    let emptyCELF = DetectionMaskJSONCELF.capture(emptyField, width: 10, height: 7)!
    assert(emptyCELF.owners.values.isEmpty && emptyCELF.regions.isEmpty)
    do { _ = try project(Array(repeating: .nan, count: 8)); assertionFailure("NaN accepted") }
    catch DetectionMaskJSONError.invalidMask {}

    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("mask-json-test-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let video = directory.appendingPathComponent("output.mp4")
    func writer(overwrite: Bool = false) throws -> DetectionMaskJSONWriter {
      try DetectionMaskJSONWriter(
        outputVideo: video, width: 4, height: 2,
        sourceFPSNumerator: 30000, sourceFPSDenominator: 1001,
        outputFPSNumerator: 30000, outputFPSDenominator: 1001,
        emptyLookahead: 10, maskReuseSkipFrames: 0,
        multiMosaicMode: "perDetection", overwrite: overwrite)
    }
    func read(_ url: URL) throws -> [String: Any] {
      try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }
    let stream = try writer()
    try stream.append(ptsNanoseconds: 50, detectionStatus: "model", observationPTS: 50, regions: [region, cropped])
    try stream.append(ptsNanoseconds: 100, detectionStatus: "empty_lookahead", observationPTS: nil, regions: [])
    try stream.append(ptsNanoseconds: 150, detectionStatus: "reused", observationPTS: 50, regions: [region],
      celf: celf, celfCrossfadePrevious: DetectionMaskJSONCELF.capture(field, width: 10, height: 7),
      crossfadeLaterWeight: 0.5)
    do {
      try stream.append(ptsNanoseconds: 150, detectionStatus: "model", observationPTS: 150, regions: [])
      assertionFailure("unordered frame accepted")
    } catch DetectionMaskJSONError.frameOrder {}
    try stream.finish(complete: true)
    let json = try read(stream.outputURL)
    assert(json["complete"] as? Bool == true && json["frame_count"] as? Int == 3)
    let frames = json["frames"] as! [[String: Any]]
    assert(frames.map { $0["frame_index"] as! Int } == [0, 1, 2])
    assert((frames[0]["detections"] as! [Any]).count == 2)
    assert((frames[1]["detections"] as! [Any]).isEmpty)
    assert(frames[2]["observation_pts_ns"] as? Int == 50)
    let savedCELF = frames[2]["celf"] as! [String: Any]
    assert(savedCELF["stage"] as? String == "after_tracker_and_gap_fill")
    assert(savedCELF["stored_rect"] as? [Int] == [1, 0, 2, 2])
    assert(frames[2]["celf_crossfade_previous"] is [String: Any])
    assert(frames[2]["crossfade_later_weight"] as? Double == 0.5)
    assert(frames[0]["celf"] == nil)
    assert(json["input"] == nil && json["source_path"] == nil)
    assert(!FileManager.default.fileExists(atPath: stream.partialURL.path))
    do { _ = try writer(); assertionFailure("unexpected overwrite") }
    catch DetectionMaskJSONError.existingFile {}
    let interrupted = try writer(overwrite: true)
    try interrupted.finish(complete: false)
    let partial = try read(interrupted.partialURL)
    assert(partial["complete"] as? Bool == false && partial["frame_count"] as? Int == 0)
    let preserved = try read(stream.outputURL)
    assert(preserved["frame_count"] as? Int == 3)
    let replacement = try writer(overwrite: true)
    try replacement.append(ptsNanoseconds: 300, detectionStatus: "model", observationPTS: 300, regions: [])
    try replacement.finish(complete: true)
    let replaced = try read(stream.outputURL)
    assert(replaced["complete"] as? Bool == true && replaced["frame_count"] as? Int == 1)
    print("Detection mask JSON tests passed")
  }
}
