import Foundation

final class H3QwenBPETokenizer: @unchecked Sendable {
  private let vocabulary: [String: Int32]
  private let mergeRanks: [String: Int]
  private let byteEncoder: [UInt8: String]
  private var cache: [String: [Int32]] = [:]
  private let lock = NSLock()
  private let expression: NSRegularExpression

  init(directory: URL) throws {
    let vocabularyURL = directory.appendingPathComponent("vocab.json")
    let mergesURL = directory.appendingPathComponent("merges.txt")
    guard FileManager.default.fileExists(atPath: vocabularyURL.path),
      FileManager.default.fileExists(atPath: mergesURL.path)
    else {
      throw H3NativeError.missingAsset(
        "Qwen tokenizer needs vocab.json and merges.txt in \(directory.path)"
      )
    }
    let rawVocabulary = try JSONDecoder().decode(
      [String: Int].self,
      from: Data(contentsOf: vocabularyURL)
    )
    vocabulary = rawVocabulary.mapValues(Int32.init)
    let lines = try String(contentsOf: mergesURL, encoding: .utf8)
      .split(whereSeparator: \.isNewline)
    var ranks: [String: Int] = [:]
    var rank = 0
    for line in lines {
      if line.hasPrefix("#") { continue }
      let pieces = line.split(separator: " ", maxSplits: 1).map(String.init)
      guard pieces.count == 2 else { continue }
      ranks[Self.pairKey(pieces[0], pieces[1])] = rank
      rank += 1
    }
    mergeRanks = ranks
    byteEncoder = Self.makeByteEncoder()
    expression = try NSRegularExpression(
      pattern: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
    )
  }

  func encode(_ text: String) throws -> [Int32] {
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    var result: [Int32] = []
    var coveredUTF16 = 0
    for match in expression.matches(in: text, range: range) {
      guard let swiftRange = Range(match.range, in: text) else { continue }
      if match.range.location > coveredUTF16 {
        let missing = NSRange(
          location: coveredUTF16,
          length: match.range.location - coveredUTF16
        )
        if let missingRange = Range(missing, in: text) {
          result += try encodePiece(String(text[missingRange]))
        }
      }
      result += try encodePiece(String(text[swiftRange]))
      coveredUTF16 = match.range.location + match.range.length
    }
    if coveredUTF16 < range.length {
      let trailing = NSRange(
        location: coveredUTF16,
        length: range.length - coveredUTF16
      )
      if let trailingRange = Range(trailing, in: text) {
        result += try encodePiece(String(text[trailingRange]))
      }
    }
    return result
  }

  private func encodePiece(_ piece: String) throws -> [Int32] {
    if let cached = lock.withLock({ cache[piece] }) { return cached }
    var symbols = Array(piece.utf8).map { byteEncoder[$0]! }
    while symbols.count > 1 {
      var bestIndex: Int?
      var bestRank = Int.max
      for index in 0..<(symbols.count - 1) {
        if let candidate = mergeRanks[
          Self.pairKey(symbols[index], symbols[index + 1])
        ], candidate < bestRank {
          bestRank = candidate
          bestIndex = index
        }
      }
      guard bestIndex != nil else { break }
      var merged: [String] = []
      merged.reserveCapacity(symbols.count)
      var index = 0
      while index < symbols.count {
        if index + 1 < symbols.count,
          mergeRanks[Self.pairKey(symbols[index], symbols[index + 1])] == bestRank
        {
          merged.append(symbols[index] + symbols[index + 1])
          index += 2
        } else {
          merged.append(symbols[index])
          index += 1
        }
      }
      symbols = merged
    }
    let ids = try symbols.map { symbol -> Int32 in
      guard let id = vocabulary[symbol] else {
        throw H3NativeError.invalidManifest(
          "Qwen vocabulary has no token for byte sequence \(symbol.debugDescription)"
        )
      }
      return id
    }
    lock.withLock { cache[piece] = ids }
    return ids
  }

  private static func pairKey(_ left: String, _ right: String) -> String {
    left + "\u{0}" + right
  }

  private static func makeByteEncoder() -> [UInt8: String] {
    var bytes = Array(33...126) + Array(161...172) + Array(174...255)
    var unicode = bytes
    var extra = 0
    for byte in 0...255 where !bytes.contains(byte) {
      bytes.append(byte)
      unicode.append(256 + extra)
      extra += 1
    }
    return Dictionary(uniqueKeysWithValues: zip(bytes, unicode).map { byte, scalar in
      (UInt8(byte), String(UnicodeScalar(scalar)!))
    })
  }
}

struct H3QwenPresentation: Sendable {
  static let visionStart: Int32 = 151652
  static let visionEnd: Int32 = 151653
  static let imagePad: Int32 = 151655
  static let textPad: Int32 = 151643

  let inputIDs: H3Tensor
  let attentionMask: H3Tensor
  let pixelValues: H3Tensor
  let imageGridTHW: H3Tensor
  let positionIDs: H3Tensor
  let tokenTags: H3Tensor
  let effectiveSequenceLength: Int
  let promptTokenCount: Int
  let usedPromptTokenCount: Int

  var stageInputs: [String: H3Tensor] {
    [
      "inputIDs": inputIDs,
      "attentionMask": attentionMask,
      "pixelValues": pixelValues,
      "imageGridTHW": imageGridTHW,
      "positionIDs": positionIDs,
      "tokenTags": tokenTags,
    ]
  }

  private struct VisionSpan {
    let start: Int
    let size: Int
    let gridHeight: Int
    let gridWidth: Int
  }

  static func makeTextOnly(
    prompt: String,
    tokenizer: H3QwenBPETokenizer,
    fixedSequenceLength: Int? = nil
  ) throws -> H3QwenPresentation {
    let promptTokens = try tokenizer.encode(prompt)
    let usedPromptTokens: [Int32]
    if let fixedSequenceLength {
      guard fixedSequenceLength > 0 else {
        throw H3NativeError.invalidTensor(
          "Qwen fixed sequence length must be positive"
        )
      }
      usedPromptTokens = Array(promptTokens.prefix(fixedSequenceLength))
    } else {
      usedPromptTokens = promptTokens
    }
    var tokenIDs = usedPromptTokens.isEmpty ? [textPad] : usedPromptTokens
    let contentSequenceLength = tokenIDs.count
    if let fixedSequenceLength, tokenIDs.count < fixedSequenceLength {
      tokenIDs.append(
        contentsOf: repeatElement(
          textPad,
          count: fixedSequenceLength - tokenIDs.count
        )
      )
    }
    let positions = makePositionIDs(sequenceLength: tokenIDs.count, spans: [])
    return H3QwenPresentation(
      inputIDs: try H3Tensor(int32: tokenIDs, shape: [1, tokenIDs.count]),
      attentionMask: try H3Tensor(
        int32: (0..<tokenIDs.count).map {
          $0 < contentSequenceLength ? 1 : 0
        },
        shape: [1, tokenIDs.count]
      ),
      pixelValues: try H3Tensor(float32: [], shape: [0, 1536]),
      imageGridTHW: try H3Tensor(int32: [], shape: [0, 3]),
      positionIDs: try H3Tensor(
        int32: positions,
        shape: [3, tokenIDs.count]
      ),
      tokenTags: try H3Tensor(
        int32: [Int32](repeating: 1, count: tokenIDs.count),
        shape: [1, tokenIDs.count]
      ),
      // The Core AI Qwen graph is padded to its compiled sequence length, but
      // native T2VA sends only real prompt rows into the DiT/text refiner.
      effectiveSequenceLength: contentSequenceLength,
      promptTokenCount: promptTokens.count,
      usedPromptTokenCount: usedPromptTokens.count
    )
  }

  static func makeReferenceVideo(
    prompt: String,
    video: H3Tensor,
    tokenizer: H3QwenBPETokenizer,
    fixedSequenceLength: Int? = nil,
    identityReferenceCount: Int? = nil
  ) throws -> H3QwenPresentation {
    guard video.shape.count == 5, video.shape[0] == 1, video.shape[1] == 3 else {
      throw H3NativeError.invalidTensor(
        "Qwen reference video must be NCTHW with three channels"
      )
    }
    let frameCount = video.shape[2]
    let height = video.shape[3]
    let width = video.shape[4]
    guard height % 32 == 0, width % 32 == 0 else {
      throw H3NativeError.invalidTensor(
        "Qwen reference dimensions must be multiples of 32"
      )
    }
    let source = try video.floatValues()
    var sampleIndices: [Int]
    if let identityReferenceCount {
      guard identityReferenceCount > 0,
        identityReferenceCount <= H3Geometry.identityVisionBlocks,
        frameCount == identityReferenceCount * 2
      else {
        throw H3NativeError.invalidTensor(
          "identity references need one paired visual block per image"
        )
      }
      sampleIndices = Array(0..<frameCount)
    } else {
      sampleIndices = H3Geometry.qwenVideoSampleIndices(frameCount: frameCount)
      if sampleIndices.count % 2 == 1, let last = sampleIndices.last {
        sampleIndices.append(last)
      }
    }

    var tokenIDs: [Int32] = []
    var patches: [Float] = []
    var grids: [Int32] = []
    var spans: [VisionSpan] = []

    if identityReferenceCount == nil {
      tokenIDs += try tokenizer.encode("<Video 1>: ")
    }
    for block in stride(from: 0, to: sampleIndices.count, by: 2) {
      if let identityReferenceCount {
        let imageIndex = min(identityReferenceCount - 1, block / 2)
        tokenIDs += try tokenizer.encode("<Picture \(imageIndex + 1)>: ")
      } else {
        let timestamp0 = Double(block) / 2.0
        let timestamp1 = Double(block + 1) / 2.0
        let timestamp = (timestamp0 + timestamp1) / 2.0
        tokenIDs += try tokenizer.encode(String(format: "<%.1f seconds>", timestamp))
      }
      tokenIDs.append(visionStart)
      let start = tokenIDs.count
      let gridHeight = height / 16
      let gridWidth = width / 16
      let visualTokenCount = (gridHeight / 2) * (gridWidth / 2)
      tokenIDs.append(contentsOf: repeatElement(imagePad, count: visualTokenCount))
      tokenIDs.append(visionEnd)
      spans.append(
        VisionSpan(
          start: start,
          size: visualTokenCount,
          gridHeight: gridHeight,
          gridWidth: gridWidth
        )
      )
      grids += [1, Int32(gridHeight), Int32(gridWidth)]
      appendVideoBlockPatches(
        source: source,
        shape: video.shape,
        firstFrame: sampleIndices[block],
        secondFrame: sampleIndices[block + 1],
        destination: &patches
      )
    }
    let promptTokens = try tokenizer.encode(prompt)
    let usedPromptTokens: [Int32]
    if let fixedSequenceLength {
      let available = fixedSequenceLength - tokenIDs.count
      guard available > 0 else {
        throw H3NativeError.invalidTensor(
          "Qwen visual prefix needs \(tokenIDs.count) tokens, compiled model has \(fixedSequenceLength)"
        )
      }
      usedPromptTokens = Array(promptTokens.prefix(available))
    } else {
      usedPromptTokens = promptTokens
    }
    tokenIDs += usedPromptTokens
    if tokenIDs.isEmpty { tokenIDs = [textPad] }
    let contentSequenceLength = tokenIDs.count
    if let fixedSequenceLength, tokenIDs.count < fixedSequenceLength {
      tokenIDs.append(
        contentsOf: repeatElement(
          textPad,
          count: fixedSequenceLength - tokenIDs.count
        )
      )
    }
    // Qwen itself is compiled at a fixed sequence length, but the dynamic DiT
    // must receive only real image/text rows. Passing padded rows changes the
    // conditioning and previously left just 16 usable prompt tokens.
    let effectiveSequenceLength = contentSequenceLength

    var tags = [Int32](repeating: 1, count: tokenIDs.count)
    for span in spans {
      let lower = max(0, span.start - 1)
      let upper = min(tags.count, span.start + span.size + 1)
      for index in lower..<upper { tags[index] = 0 }
    }
    let positions = makePositionIDs(sequenceLength: tokenIDs.count, spans: spans)
    let gridHeight = height / 16
    let gridWidth = width / 16
    let patchRows = spans.count * gridHeight * gridWidth
    guard patches.count == patchRows * 1536 else {
      throw H3NativeError.invalidTensor(
        "Qwen patch packing produced \(patches.count), expected \(patchRows * 1536)"
      )
    }
    return H3QwenPresentation(
      inputIDs: try H3Tensor(int32: tokenIDs, shape: [1, tokenIDs.count]),
      attentionMask: try H3Tensor(
        int32: (0..<tokenIDs.count).map {
          $0 < contentSequenceLength ? 1 : 0
        },
        shape: [1, tokenIDs.count]
      ),
      pixelValues: try H3Tensor(
        float32: patches,
        shape: [patchRows, 1536]
      ),
      imageGridTHW: try H3Tensor(
        int32: grids,
        shape: [spans.count, 3]
      ),
      positionIDs: try H3Tensor(
        int32: positions,
        shape: [3, tokenIDs.count]
      ),
      tokenTags: try H3Tensor(int32: tags, shape: [1, tokenIDs.count]),
      effectiveSequenceLength: effectiveSequenceLength,
      promptTokenCount: promptTokens.count,
      usedPromptTokenCount: usedPromptTokens.count
    )
  }

  private static func appendVideoBlockPatches(
    source: [Float],
    shape: [Int],
    firstFrame: Int,
    secondFrame: Int,
    destination: inout [Float]
  ) {
    let frames = shape[2]
    let height = shape[3]
    let width = shape[4]
    let gridHeight = height / 16
    let gridWidth = width / 16
    func value(frame: Int, channel: Int, y: Int, x: Int) -> Float {
      let safeFrame = min(max(0, frame), frames - 1)
      let index = ((channel * frames + safeFrame) * height + y) * width + x
      return source[index] * 2 - 1
    }
    destination.reserveCapacity(
      destination.count + gridHeight * gridWidth * 1536
    )
    for mergedY in 0..<(gridHeight / 2) {
      for mergedX in 0..<(gridWidth / 2) {
        for patchY in 0..<2 {
          for patchX in 0..<2 {
            let gridY = mergedY * 2 + patchY
            let gridX = mergedX * 2 + patchX
            for channel in 0..<3 {
              for frame in [firstFrame, secondFrame] {
                for y in 0..<16 {
                  for x in 0..<16 {
                    destination.append(
                      value(
                        frame: frame,
                        channel: channel,
                        y: gridY * 16 + y,
                        x: gridX * 16 + x
                      )
                    )
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  private static func makePositionIDs(
    sequenceLength: Int,
    spans: [VisionSpan]
  ) -> [Int32] {
    var result = [Int32](repeating: 0, count: 3 * sequenceLength)
    var offset = 0
    var initialized = false
    func set(_ axis: Int, _ index: Int, _ value: Int) {
      result[axis * sequenceLength + index] = Int32(value)
    }
    for span in spans {
      if !initialized {
        for axis in 0..<3 {
          for index in 0..<span.start { set(axis, index, index) }
        }
        initialized = true
      }
      let end = span.start + span.size
      let maximumDimension = max(span.gridHeight, span.gridWidth) / 2
      let nextStart = span.start + maximumDimension
      if end < sequenceLength {
        for axis in 0..<3 {
          for index in end..<sequenceLength {
            set(axis, index, nextStart + offset + index - end)
          }
        }
      }
      for index in span.start..<end {
        set(0, index, span.start + offset)
        let local = index - span.start
        set(1, index, span.start + offset + local / (span.gridWidth / 2))
        set(2, index, span.start + offset + local % (span.gridWidth / 2))
      }
      offset += maximumDimension - span.size
    }
    if !initialized {
      for axis in 0..<3 {
        for index in 0..<sequenceLength { set(axis, index, index) }
      }
    }
    return result
  }
}
