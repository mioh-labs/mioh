// SPDX-License-Identifier: AGPL-3.0
import Foundation

/// Detection masks, not the feathered/composited restoration alpha. Coordinates
/// are in the oriented source frame, with a top-left origin and exclusive ends.
struct DetectionMaskJSONRegion: Encodable {
  let bbox: [Int] // x, y, width, height
  let confidence: Float
  let class_index: Int
  let mask: Mask

  struct Mask: Encodable {
    let size: [Int] // height, width of bbox-local mask
    let counts: [Int] // alternating zero/one runs, starting with zeros
    let order = "row-major"
  }

  /// Match the native compositor's bilinear sampling and strict threshold.
  /// samplingRect describes the unpadded model-space image (YOLO letterbox),
  /// or the whole mask tensor (direct-resize detectors).
  static func project(
    values: [Float], maskWidth: Int, maskHeight: Int,
    sourceWidth: Int, sourceHeight: Int, samplingRect: [Int],
    bbox: [Int], threshold: Float, confidence: Float, classIndex: Int
  ) throws -> DetectionMaskJSONRegion {
    guard maskWidth > 0, maskHeight > 0,
      values.count == maskWidth * maskHeight,
      sourceWidth > 0, sourceHeight > 0, samplingRect.count == 4,
      samplingRect[2] > 0, samplingRect[3] > 0,
      bbox.count == 4, bbox[2] > 0, bbox[3] > 0,
      bbox[0] >= 0, bbox[1] >= 0,
      bbox[0] + bbox[2] <= sourceWidth, bbox[1] + bbox[3] <= sourceHeight,
      confidence.isFinite, threshold.isFinite
    else { throw DetectionMaskJSONError.invalidMask }

    func axis(_ origin: Int, _ count: Int, _ source: Int, _ start: Int,
              _ extent: Int, _ mask: Int) -> [(Int, Int, Float)] {
      (origin..<(origin + count)).map { pixel in
        let p = Float(start) + (Float(pixel) + 0.5) * Float(extent)
          / Float(source) - 0.5
        let clamped = max(0, min(Float(mask - 1), p))
        let lower = Int(floor(clamped))
        return (lower, min(mask - 1, lower + 1), clamped - Float(lower))
      }
    }
    let xs = axis(bbox[0], bbox[2], sourceWidth, samplingRect[0], samplingRect[2], maskWidth)
    let ys = axis(bbox[1], bbox[3], sourceHeight, samplingRect[1], samplingRect[3], maskHeight)
    var counts: [Int] = []
    var current = false
    var length = 0
    for (y0, y1, fy) in ys {
      for (x0, x1, fx) in xs {
        let upper = values[y0 * maskWidth + x0] * (1 - fx) + values[y0 * maskWidth + x1] * fx
        let lower = values[y1 * maskWidth + x0] * (1 - fx) + values[y1 * maskWidth + x1] * fx
        let value = upper * (1 - fy) + lower * fy
        guard value.isFinite else { throw DetectionMaskJSONError.invalidMask }
        let bit = value > threshold
        if bit != current {
          counts.append(length)
          length = 0
          current = bit
        }
        length += 1
      }
    }
    counts.append(length)
    return DetectionMaskJSONRegion(
      bbox: bbox, confidence: confidence, class_index: classIndex,
      mask: Mask(size: [bbox[3], bbox[2]], counts: counts))
  }
}

enum DetectionMaskJSONError: LocalizedError {
  case invalidMask, existingFile, closed, frameOrder
  var errorDescription: String? {
    switch self {
    case .invalidMask: return "検出マスクJSON: マスクの寸法または値が不正です。"
    case .existingFile: return "検出マスクJSONが既に存在します。別の出力名にするか上書きを有効にしてください。"
    case .closed: return "検出マスクJSONは既に閉じられています。"
    case .frameOrder: return "検出マスクJSON: フレーム時刻が順序どおりではありません。"
    }
  }
}

/// Lossless snapshot of CELF's native cell grid, including ownership, feather
/// coverage and cells filled across detection gaps. Multi-value RLE is not
/// binary RLE: each value is repeated for its matching length.
struct DetectionMaskJSONCELF: Encodable {
  let stage = "after_tracker_and_gap_fill"
  let grid_size: [Int] // height, width
  let cell_size: Int
  let stored_rect: [Int] // x, y, width, height, in cells
  let owners: Runs
  let coverage_u8: Runs
  let regions: [Region]

  static func capture(_ field: LabelFieldFrame?, width: Int, height: Int) -> DetectionMaskJSONCELF? {
    guard let field else { return nil }
    var owners: [Int] = []
    var coverage: [Int] = []
    owners.reserveCapacity(field.width * field.height)
    coverage.reserveCapacity(field.width * field.height)
    for y in field.originY..<(field.originY + field.height) {
      for x in field.originX..<(field.originX + field.width) {
        let (owner, weight) = field.ownerAndWeight(cellX: x, cellY: y)
        owners.append(Int(owner))
        coverage.append(Int((weight * 255).rounded()))
      }
    }
    let regions = field.boxes.keys.sorted().compactMap { id -> Region? in
      guard let box = field.boxes[id] else { return nil }
      let left = max(0, box.left), top = max(0, box.top)
      let right = min(width, box.right + 1), bottom = min(height, box.bottom + 1)
      guard right > left, bottom > top else { return nil }
      return Region(track_id: id, bbox: [left, top, right - left, bottom - top])
    }
    return DetectionMaskJSONCELF(
      grid_size: [field.gridHeight, field.gridWidth], cell_size: field.cellSize,
      stored_rect: [field.originX, field.originY, field.width, field.height],
      owners: .init(owners), coverage_u8: .init(coverage), regions: regions)
  }

  struct Region: Encodable {
    let track_id: Int
    let bbox: [Int] // source-pixel x, y, width, height
  }

  struct Runs: Encodable {
    let values: [Int]
    let lengths: [Int]
    let order = "row-major"

    init(_ items: [Int]) {
      var values: [Int] = []
      var lengths: [Int] = []
      for item in items {
        if values.last == item {
          lengths[lengths.count - 1] += 1
        } else {
          values.append(item)
          lengths.append(1)
        }
      }
      self.values = values
      self.lengths = lengths
    }
  }
}

/// Only the ordered batch consumer writes here. Stream records to disk instead
/// of retaining a video's masks in RAM. Interrupted runs remain valid JSON with
/// complete=false in a unique partial file, never replacing the last good file.
final class DetectionMaskJSONWriter {
  let outputURL: URL
  let partialURL: URL
  private var handle: FileHandle?
  private let overwrite: Bool
  private let encoder = JSONEncoder()
  private(set) var frameCount = 0
  private var lastPTS: Int64?

  init(outputVideo: URL, width: Int, height: Int,
       sourceFPSNumerator: Int, sourceFPSDenominator: Int,
       outputFPSNumerator: Int, outputFPSDenominator: Int,
       emptyLookahead: Int, maskReuseSkipFrames: Int,
       multiMosaicMode: String, overwrite: Bool = false) throws {
    outputURL = outputVideo.deletingPathExtension().appendingPathExtension("masks.json")
    partialURL = outputURL.deletingLastPathComponent().appendingPathComponent(
      outputURL.deletingPathExtension().lastPathComponent + ".partial-" + UUID().uuidString + ".json")
    self.overwrite = overwrite
    if FileManager.default.fileExists(atPath: outputURL.path), !overwrite {
      throw DetectionMaskJSONError.existingFile
    }
    guard FileManager.default.createFile(atPath: partialURL.path, contents: nil) else {
      throw CocoaError(.fileWriteUnknown)
    }
    handle = try FileHandle(forWritingTo: partialURL)
    let metadata: [String: Any] = [
      "schema": "mioh-detection-masks-v1", "width": width, "height": height,
      "coordinate_system": "oriented-source-pixels-top-left",
      "mask_kind": "binary-detector-mask-before-tracking-and-feather",
      "bbox_format": "xywh-exclusive", "frame_index_base": 0,
      "source_fps": ["numerator": sourceFPSNumerator, "denominator": sourceFPSDenominator],
      "output_fps": ["numerator": outputFPSNumerator, "denominator": outputFPSDenominator],
      "empty_lookahead": emptyLookahead, "mask_reuse_skip_frames": maskReuseSkipFrames,
      "multi_mosaic_mode": multiMosaicMode,
    ]
    var header = try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys])
    header.removeLast() // continue the top-level object with a streamed array
    header.append(Data(",\"frames\":[\n".utf8))
    try handle?.write(contentsOf: header)
  }

  func append(ptsNanoseconds: Int64, detectionStatus: String,
              observationPTS: Int64?, regions: [DetectionMaskJSONRegion],
              celf: DetectionMaskJSONCELF? = nil,
              celfCrossfadePrevious: DetectionMaskJSONCELF? = nil,
              crossfadeLaterWeight: Float? = nil) throws {
    guard let handle else { throw DetectionMaskJSONError.closed }
    if let lastPTS, ptsNanoseconds <= lastPTS { throw DetectionMaskJSONError.frameOrder }
    struct Frame: Encodable {
      let frame_index: Int
      let source_pts_ns: Int64
      let detection_status: String
      let observation_pts_ns: Int64?
      let detections: [DetectionMaskJSONRegion]
      let celf: DetectionMaskJSONCELF?
      let celf_crossfade_previous: DetectionMaskJSONCELF?
      let crossfade_later_weight: Float?
    }
    let data = try encoder.encode(Frame(
      frame_index: frameCount, source_pts_ns: ptsNanoseconds,
      detection_status: detectionStatus, observation_pts_ns: observationPTS, detections: regions,
      celf: celf, celf_crossfade_previous: celfCrossfadePrevious,
      crossfade_later_weight: crossfadeLaterWeight))
    if frameCount > 0 { try handle.write(contentsOf: Data(",\n".utf8)) }
    try handle.write(contentsOf: data)
    frameCount += 1
    lastPTS = ptsNanoseconds
  }

  func finish(complete: Bool) throws {
    guard let handle else { return }
    try handle.write(contentsOf: Data(
      "\n],\"frame_count\":\(frameCount),\"complete\":\(complete)}\n".utf8))
    try handle.synchronize()
    try handle.close()
    self.handle = nil
    if complete {
      if FileManager.default.fileExists(atPath: outputURL.path) {
        guard overwrite else { throw DetectionMaskJSONError.existingFile }
        _ = try FileManager.default.replaceItemAt(outputURL, withItemAt: partialURL)
      } else {
        try FileManager.default.moveItem(at: partialURL, to: outputURL)
      }
    }
  }

  deinit { try? finish(complete: false) }
}
