import AVFoundation
import CoreMedia
import Foundation

/// A deliberately small MPEG-TS demuxer used for the provider's H.264
/// segments when AVFoundation cannot open a standalone `.ts` file directly.
/// It copies compressed H.264 access units into MP4; it does not decode or
/// re-encode video and intentionally ignores audio and every non-H.264 PID.
enum IPadMPEGTSRemuxer {
  struct ConcatenatedResult: Sendable {
    let sourceOffsets: [TimeInterval]
    let sourceDurations: [TimeInterval]
    let duration: TimeInterval
  }

  enum RemuxError: LocalizedError {
    case invalidInput(String)
    case unsupported(String)
    case malformed(String)
    case writer(String)

    var errorDescription: String? {
      switch self {
      case .invalidInput(let detail):
        return "MPEG-TS入力が不正です: \(detail)"
      case .unsupported(let detail):
        return "未対応のMPEG-TSです: \(detail)"
      case .malformed(let detail):
        return "MPEG-TSの解析に失敗しました: \(detail)"
      case .writer(let detail):
        return "MPEG-TSからMP4を作成できません: \(detail)"
      }
    }
  }

  private static let packetSize = 188
  private static let clockRate: CMTimeScale = 90_000
  private static let maximumInputBytes = 64 * 1_024 * 1_024
  private static let maximumPESBytes = 64 * 1_024 * 1_024
  private static let maximumConcatenatedInputs = 8
  private static let timestampWrap: Int64 = 1 << 33

  private struct PESAccessUnit {
    let nals: [Data]
    let pts: Int64
    let dts: Int64
    let isSync: Bool
  }

  private struct ParsedStream {
    let sps: Data
    let pps: Data
    let units: [PESAccessUnit]
  }

  private struct PSIAssembler {
    var bytes = Data()
    var expectedLength: Int?

    mutating func consume(_ payload: Data, startsSection: Bool) throws -> [Data] {
      var offset = 0
      if startsSection {
        guard let pointer = payload.first else { return [] }
        offset = 1 + Int(pointer)
        guard offset <= payload.count else {
          throw RemuxError.malformed("PSI pointer_fieldが範囲外です")
        }
        bytes.removeAll(keepingCapacity: true)
        expectedLength = nil
      }
      guard offset < payload.count else { return [] }
      bytes.append(payload[offset...])

      var sections: [Data] = []
      while true {
        if expectedLength == nil, bytes.count >= 3 {
          if bytes[0] == 0xFF {
            bytes.removeAll(keepingCapacity: true)
            break
          }
          let sectionLength = (Int(bytes[1] & 0x0F) << 8) | Int(bytes[2])
          guard sectionLength >= 4, sectionLength <= 1_021 else {
            throw RemuxError.malformed("PSI section_lengthが不正です")
          }
          expectedLength = 3 + sectionLength
        }
        guard let expectedLength, bytes.count >= expectedLength else { break }
        sections.append(bytes.prefix(expectedLength))
        bytes = Data(bytes.dropFirst(expectedLength))
        self.expectedLength = nil
      }
      return sections
    }
  }

  static func remux(inputURL: URL, outputURL: URL) async throws {
    try Task.checkCancellation()
    guard outputURL.isFileURL else {
      throw RemuxError.invalidInput("ローカルファイルだけを処理できます")
    }
    let stream = try loadStream(from: inputURL)
    try await write(stream, to: outputURL)
  }

  /// Uses the payload rather than the URL suffix to recognize a materialized
  /// HLS transport stream. This also protects sessions created by an older
  /// build that may have saved TS bytes with an `.mp4` extension.
  static func appearsToBeTransportStream(_ inputURL: URL) -> Bool {
    guard inputURL.isFileURL,
      let handle = try? FileHandle(forReadingFrom: inputURL)
    else { return false }
    defer { try? handle.close() }
    guard let prefix = try? handle.read(upToCount: packetSize * 3),
      prefix.count >= packetSize * 3
    else { return false }
    return prefix[0] == 0x47
      && prefix[packetSize] == 0x47
      && prefix[packetSize * 2] == 0x47
  }

  /// Remuxes independently timestamped HLS transport streams into one writer
  /// session. This avoids AVMutableComposition, which rejects some valid
  /// provider edit lists on iOS with an unhelpful GenericObjCError.
  static func remux(
    inputURLs: [URL],
    outputURL: URL
  ) async throws -> ConcatenatedResult {
    try Task.checkCancellation()
    guard (1...maximumConcatenatedInputs).contains(inputURLs.count),
      outputURL.isFileURL
    else {
      throw RemuxError.invalidInput(
        "1〜\(maximumConcatenatedInputs)個のローカルTS区間を指定してください"
      )
    }
    let streams = try inputURLs.map(loadStream)
    let concatenated = try concatenateStreams(streams)
    try await write(concatenated.stream, to: outputURL)
    return concatenated.result
  }

  private static func loadStream(from inputURL: URL) throws -> ParsedStream {
    guard inputURL.isFileURL else {
      throw RemuxError.invalidInput("ローカルファイルだけを処理できます")
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: inputURL.path)
    guard let byteCount = (attributes[.size] as? NSNumber)?.intValue,
      byteCount >= packetSize,
      byteCount <= maximumInputBytes
    else {
      throw RemuxError.invalidInput("ファイルサイズが処理範囲外です")
    }
    let data = try Data(contentsOf: inputURL, options: .mappedIfSafe)
    return try parseTransportStream(data)
  }

  private static func concatenateStreams(
    _ streams: [ParsedStream]
  ) throws -> (stream: ParsedStream, result: ConcatenatedResult) {
    guard let first = streams.first else {
      throw RemuxError.invalidInput("TS区間がありません")
    }
    var mergedUnits: [PESAccessUnit] = []
    var sourceOffsets: [TimeInterval] = []
    var sourceDurations: [TimeInterval] = []
    var decodeCursor: Int64 = 0
    var presentationCursor: Int64 = 0

    for (index, stream) in streams.enumerated() {
      guard stream.sps == first.sps, stream.pps == first.pps else {
        throw RemuxError.unsupported(
          "連続HLS区間でH.264 SPS/PPSが変更されました"
        )
      }
      let durations = accessUnitDurations(stream.units)
      guard let firstUnit = stream.units.first,
        let lastDuration = durations.last
      else {
        throw RemuxError.malformed("H.264 access unitを検出できません")
      }
      let localPresentationStart = zip(stream.units, durations)
        .map { $0.0.pts }
        .min() ?? firstUnit.pts
      let localPresentationEnd = zip(stream.units, durations)
        .map { unit, duration in unit.pts + duration }
        .max() ?? (firstUnit.pts + lastDuration)
      let decodeShift = decodeCursor - firstUnit.dts
      let presentationShift = presentationCursor - localPresentationStart
      let shift = index == 0 ? decodeShift : max(decodeShift, presentationShift)
      let shiftedUnits = stream.units.map { unit in
        PESAccessUnit(
          nals: unit.nals,
          pts: unit.pts + shift,
          dts: unit.dts + shift,
          isSync: unit.isSync
        )
      }
      guard let lastUnit = shiftedUnits.last else {
        throw RemuxError.malformed("H.264 access unitを検出できません")
      }
      let shiftedPresentationStart = localPresentationStart + shift
      let shiftedPresentationEnd = localPresentationEnd + shift
      sourceOffsets.append(
        Double(max(0, shiftedPresentationStart)) / Double(clockRate)
      )
      sourceDurations.append(
        Double(max(1, shiftedPresentationEnd - shiftedPresentationStart))
          / Double(clockRate)
      )
      mergedUnits.append(contentsOf: shiftedUnits)
      decodeCursor = lastUnit.dts + lastDuration
      presentationCursor = max(presentationCursor, shiftedPresentationEnd)
    }

    for index in 1..<mergedUnits.count
    where mergedUnits[index].dts <= mergedUnits[index - 1].dts
    {
      throw RemuxError.unsupported("連結後のPES DTSが単調増加ではありません")
    }
    let totalTicks = max(decodeCursor, presentationCursor)
    return (
      ParsedStream(sps: first.sps, pps: first.pps, units: mergedUnits),
      ConcatenatedResult(
        sourceOffsets: sourceOffsets,
        sourceDurations: sourceDurations,
        duration: Double(max(1, totalTicks)) / Double(clockRate)
      )
    )
  }

  private static func accessUnitDurations(_ units: [PESAccessUnit]) -> [Int64] {
    guard !units.isEmpty else { return [] }
    let deltas = zip(units.dropFirst(), units).map {
      max(1, $0.dts - $1.dts)
    }
    let fallbackDuration = deltas.last ?? 3_003
    return units.indices.map { index in
      index + 1 < units.count
        ? max(1, units[index + 1].dts - units[index].dts)
        : fallbackDuration
    }
  }

  private static func parseTransportStream(_ data: Data) throws -> ParsedStream {
    guard let syncOffset = findSyncOffset(in: data) else {
      throw RemuxError.invalidInput("188バイトMPEG-TS同期語を検出できません")
    }

    var pat = PSIAssembler()
    var pmt = PSIAssembler()
    var pmtPID: Int?
    var videoPID: Int?
    var pes = Data()
    var pesUnits: [Data] = []

    var packetStart = syncOffset
    while packetStart + packetSize <= data.count {
      guard data[packetStart] == 0x47 else {
        throw RemuxError.malformed("TS同期が途中で失われました")
      }
      let second = data[packetStart + 1]
      let third = data[packetStart + 2]
      if second & 0x80 != 0 {
        throw RemuxError.malformed("transport_error_indicatorを検出しました")
      }
      let startsPayload = second & 0x40 != 0
      let pid = (Int(second & 0x1F) << 8) | Int(third)
      let adaptationControl = (data[packetStart + 3] >> 4) & 0x03
      guard adaptationControl != 0 else {
        throw RemuxError.malformed("adaptation_field_controlが予約値です")
      }

      var payloadOffset = packetStart + 4
      if adaptationControl == 2 || adaptationControl == 3 {
        let adaptationLength = Int(data[payloadOffset])
        payloadOffset += 1 + adaptationLength
        guard payloadOffset <= packetStart + packetSize else {
          throw RemuxError.malformed("adaptation fieldがTS packetを超えています")
        }
      }
      let hasPayload = adaptationControl == 1 || adaptationControl == 3
      if hasPayload, payloadOffset < packetStart + packetSize {
        let payload = data[payloadOffset..<(packetStart + packetSize)]
        if pid == 0 {
          for section in try pat.consume(Data(payload), startsSection: startsPayload) {
            pmtPID = try parsePAT(section) ?? pmtPID
          }
        } else if pid == pmtPID {
          for section in try pmt.consume(Data(payload), startsSection: startsPayload) {
            videoPID = try parsePMT(section) ?? videoPID
          }
        } else if pid == videoPID {
          if startsPayload, !pes.isEmpty {
            pesUnits.append(pes)
            pes.removeAll(keepingCapacity: true)
          }
          if !pes.isEmpty || startsPayload {
            guard pes.count + payload.count <= maximumPESBytes else {
              throw RemuxError.malformed("PES packetが上限を超えています")
            }
            pes.append(payload)
          }
        }
      }
      packetStart += packetSize
    }
    if !pes.isEmpty { pesUnits.append(pes) }

    guard pmtPID != nil else { throw RemuxError.unsupported("PATにPMTがありません") }
    guard videoPID != nil else {
      throw RemuxError.unsupported("PMTにH.264/AVC映像がありません")
    }
    guard !pesUnits.isEmpty else { throw RemuxError.malformed("H.264 PESが空です") }

    var sps: Data?
    var pps: Data?
    var accessUnits: [PESAccessUnit] = []
    var previousDTS: Int64?
    for bytes in pesUnits {
      let parsed = try parsePES(bytes, previousDTS: previousDTS)
      previousDTS = parsed.dts
      let nals = annexBNALUnits(parsed.payload)
      for nal in nals where !nal.isEmpty {
        switch nal[0] & 0x1F {
        case 7 where sps == nil: sps = nal
        case 8 where pps == nil: pps = nal
        default: break
        }
      }
      let sampleNALs = nals.filter {
        guard let first = $0.first else { return false }
        return ![7, 8, 9].contains(first & 0x1F)
      }
      let containsPicture = sampleNALs.contains {
        guard let first = $0.first else { return false }
        return (1...5).contains(first & 0x1F)
      }
      guard containsPicture else { continue }
      accessUnits.append(
        PESAccessUnit(
          nals: sampleNALs,
          pts: parsed.pts,
          dts: parsed.dts,
          isSync: sampleNALs.contains { ($0.first ?? 0) & 0x1F == 5 }
        )
      )
    }
    guard let sps, let pps else {
      throw RemuxError.unsupported("H.264 SPS/PPSを検出できません")
    }
    guard !accessUnits.isEmpty else {
      throw RemuxError.malformed("H.264 access unitを検出できません")
    }
    for index in 1..<accessUnits.count where accessUnits[index].dts <= accessUnits[index - 1].dts {
      throw RemuxError.unsupported("PESのDTSが単調増加ではありません")
    }
    return ParsedStream(sps: sps, pps: pps, units: accessUnits)
  }

  private static func findSyncOffset(in data: Data) -> Int? {
    let limit = min(packetSize, data.count)
    for offset in 0..<limit where data[offset] == 0x47 {
      var valid = true
      for multiplier in 1...2 {
        let position = offset + multiplier * packetSize
        if position < data.count, data[position] != 0x47 {
          valid = false
          break
        }
      }
      if valid { return offset }
    }
    return nil
  }

  private static func parsePAT(_ section: Data) throws -> Int? {
    guard section.count >= 12, section[0] == 0 else { return nil }
    var offset = 8
    let end = section.count - 4
    while offset + 4 <= end {
      let program = (Int(section[offset]) << 8) | Int(section[offset + 1])
      if program != 0 {
        return (Int(section[offset + 2] & 0x1F) << 8) | Int(section[offset + 3])
      }
      offset += 4
    }
    return nil
  }

  private static func parsePMT(_ section: Data) throws -> Int? {
    guard section.count >= 16, section[0] == 2 else { return nil }
    let programInfoLength = (Int(section[10] & 0x0F) << 8) | Int(section[11])
    var offset = 12 + programInfoLength
    let end = section.count - 4
    guard offset <= end else { throw RemuxError.malformed("PMT program_infoが範囲外です") }
    while offset + 5 <= end {
      let streamType = section[offset]
      let pid = (Int(section[offset + 1] & 0x1F) << 8) | Int(section[offset + 2])
      let infoLength = (Int(section[offset + 3] & 0x0F) << 8) | Int(section[offset + 4])
      if streamType == 0x1B { return pid }
      offset += 5 + infoLength
      guard offset <= end else { throw RemuxError.malformed("PMT ES_infoが範囲外です") }
    }
    return nil
  }

  private static func parsePES(
    _ bytes: Data,
    previousDTS: Int64?
  ) throws -> (payload: Data, pts: Int64, dts: Int64) {
    guard bytes.count >= 14, bytes[0] == 0, bytes[1] == 0, bytes[2] == 1 else {
      throw RemuxError.malformed("PES start codeがありません")
    }
    let flags = bytes[7]
    let headerLength = Int(bytes[8])
    let payloadStart = 9 + headerLength
    guard payloadStart <= bytes.count else {
      throw RemuxError.malformed("PES headerが範囲外です")
    }
    let timestampFlags = (flags >> 6) & 0x03
    guard timestampFlags == 2 || timestampFlags == 3, bytes.count >= 14 else {
      throw RemuxError.unsupported("PTSのないH.264 PESです")
    }
    let rawPTS = try parseTimestamp(bytes, at: 9)
    let rawDTS = timestampFlags == 3 ? try parseTimestamp(bytes, at: 14) : rawPTS
    let dts = unwrap(rawDTS, near: previousDTS)
    let pts = unwrap(rawPTS, near: dts)

    var payloadEnd = bytes.count
    let pesPacketLength = (Int(bytes[4]) << 8) | Int(bytes[5])
    if pesPacketLength > 0 { payloadEnd = min(payloadEnd, 6 + pesPacketLength) }
    guard payloadStart < payloadEnd else { throw RemuxError.malformed("PES payloadが空です") }
    return (Data(bytes[payloadStart..<payloadEnd]), pts, dts)
  }

  private static func parseTimestamp(_ bytes: Data, at offset: Int) throws -> Int64 {
    guard offset + 5 <= bytes.count else {
      throw RemuxError.malformed("PES timestampが途切れています")
    }
    guard bytes[offset] & 1 == 1, bytes[offset + 2] & 1 == 1, bytes[offset + 4] & 1 == 1 else {
      throw RemuxError.malformed("PES timestamp marker bitが不正です")
    }
    return (Int64(bytes[offset] & 0x0E) << 29)
      | (Int64(bytes[offset + 1]) << 22)
      | (Int64(bytes[offset + 2] & 0xFE) << 14)
      | (Int64(bytes[offset + 3]) << 7)
      | Int64(bytes[offset + 4] >> 1)
  }

  private static func unwrap(_ raw: Int64, near reference: Int64?) -> Int64 {
    guard let reference else { return raw }
    let cycle = Int64((Double(reference - raw) / Double(timestampWrap)).rounded())
    return raw + cycle * timestampWrap
  }

  private static func annexBNALUnits(_ data: Data) -> [Data] {
    struct Start {
      let prefix: Int
      let nal: Int
    }
    var starts: [Start] = []
    var index = 0
    while index + 3 <= data.count {
      if data[index] == 0, data[index + 1] == 0 {
        if data[index + 2] == 1 {
          starts.append(Start(prefix: index, nal: index + 3))
          index += 3
          continue
        }
        if index + 4 <= data.count, data[index + 2] == 0, data[index + 3] == 1 {
          starts.append(Start(prefix: index, nal: index + 4))
          index += 4
          continue
        }
      }
      index += 1
    }
    guard !starts.isEmpty else { return [] }
    return starts.indices.compactMap { position in
      let start = starts[position].nal
      var end = position + 1 < starts.count ? starts[position + 1].prefix : data.count
      while end > start, data[end - 1] == 0 { end -= 1 }
      return start < end ? Data(data[start..<end]) : nil
    }
  }

  private static func makeFormatDescription(sps: Data, pps: Data) throws -> CMVideoFormatDescription
  {
    var result: CMFormatDescription?
    let status = sps.withUnsafeBytes { spsBytes in
      pps.withUnsafeBytes { ppsBytes in
        let pointers: [UnsafePointer<UInt8>] = [
          spsBytes.bindMemory(to: UInt8.self).baseAddress!,
          ppsBytes.bindMemory(to: UInt8.self).baseAddress!,
        ]
        let sizes = [sps.count, pps.count]
        return pointers.withUnsafeBufferPointer { pointerBuffer in
          sizes.withUnsafeBufferPointer { sizeBuffer in
            CMVideoFormatDescriptionCreateFromH264ParameterSets(
              allocator: kCFAllocatorDefault,
              parameterSetCount: 2,
              parameterSetPointers: pointerBuffer.baseAddress!,
              parameterSetSizes: sizeBuffer.baseAddress!,
              nalUnitHeaderLength: 4,
              formatDescriptionOut: &result
            )
          }
        }
      }
    }
    guard status == noErr, let result else {
      throw RemuxError.unsupported("H.264 SPS/PPSの形式が不正です (\(status))")
    }
    return result
  }

  private static func makeSampleBuffer(
    unit: PESAccessUnit,
    format: CMVideoFormatDescription,
    baseDTS: Int64,
    duration: Int64
  ) throws -> CMSampleBuffer {
    var avcc = Data()
    for nal in unit.nals {
      guard nal.count <= Int(UInt32.max) else {
        throw RemuxError.malformed("H.264 NAL unitが大きすぎます")
      }
      var length = UInt32(nal.count).bigEndian
      withUnsafeBytes(of: &length) { avcc.append(contentsOf: $0) }
      avcc.append(nal)
    }
    var blockBuffer: CMBlockBuffer?
    var status = CMBlockBufferCreateWithMemoryBlock(
      allocator: kCFAllocatorDefault,
      memoryBlock: nil,
      blockLength: avcc.count,
      blockAllocator: kCFAllocatorDefault,
      customBlockSource: nil,
      offsetToData: 0,
      dataLength: avcc.count,
      flags: 0,
      blockBufferOut: &blockBuffer
    )
    guard status == kCMBlockBufferNoErr, let blockBuffer else {
      throw RemuxError.writer("圧縮映像bufferを確保できません (\(status))")
    }
    status = avcc.withUnsafeBytes { bytes in
      CMBlockBufferReplaceDataBytes(
        with: bytes.baseAddress!,
        blockBuffer: blockBuffer,
        offsetIntoDestination: 0,
        dataLength: avcc.count
      )
    }
    guard status == kCMBlockBufferNoErr else {
      throw RemuxError.writer("圧縮映像bufferへ書き込めません (\(status))")
    }
    var timing = CMSampleTimingInfo(
      duration: CMTime(value: duration, timescale: clockRate),
      presentationTimeStamp: CMTime(value: unit.pts - baseDTS, timescale: clockRate),
      decodeTimeStamp: CMTime(value: unit.dts - baseDTS, timescale: clockRate)
    )
    var sampleSize = avcc.count
    var sampleBuffer: CMSampleBuffer?
    status = CMSampleBufferCreateReady(
      allocator: kCFAllocatorDefault,
      dataBuffer: blockBuffer,
      formatDescription: format,
      sampleCount: 1,
      sampleTimingEntryCount: 1,
      sampleTimingArray: &timing,
      sampleSizeEntryCount: 1,
      sampleSizeArray: &sampleSize,
      sampleBufferOut: &sampleBuffer
    )
    guard status == noErr, let sampleBuffer else {
      throw RemuxError.writer("CMSampleBufferを作成できません (\(status))")
    }
    // AVAssetWriter otherwise omits the MP4 sync-sample table and treats every
    // P/B frame as a random-access point. HLS restoration often begins inside
    // a segment, so AVAssetReader must be able to seek back to the real IDR.
    if !unit.isSync,
      let attachments = CMSampleBufferGetSampleAttachmentsArray(
        sampleBuffer,
        createIfNecessary: true
      )
    {
      let dictionary = unsafeBitCast(
        CFArrayGetValueAtIndex(attachments, 0),
        to: CFMutableDictionary.self
      )
      CFDictionarySetValue(
        dictionary,
        Unmanaged.passUnretained(kCMSampleAttachmentKey_NotSync).toOpaque(),
        Unmanaged.passUnretained(kCFBooleanTrue).toOpaque()
      )
    }
    return sampleBuffer
  }

  private static func diagnostic(_ error: Error) -> String {
    let value = error as NSError
    return "\(error.localizedDescription) [\(value.domain):\(value.code)]"
  }

  private static func write(_ stream: ParsedStream, to outputURL: URL) async throws {
    try? FileManager.default.removeItem(at: outputURL)
    let format = try makeFormatDescription(sps: stream.sps, pps: stream.pps)
    let writer: AVAssetWriter
    do {
      writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
    } catch {
      throw RemuxError.writer("writer作成: \(diagnostic(error))")
    }
    writer.shouldOptimizeForNetworkUse = true
    writer.movieTimeScale = clockRate
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: nil,
      sourceFormatHint: format
    )
    input.expectsMediaDataInRealTime = false
    input.mediaTimeScale = clockRate
    guard writer.canAdd(input) else { throw RemuxError.writer("video inputを追加できません") }
    writer.add(input)
    guard writer.startWriting() else {
      throw RemuxError.writer(writer.error?.localizedDescription ?? "startWriting failed")
    }
    writer.startSession(atSourceTime: .zero)

    let baseDTS = stream.units[0].dts
    let deltas = zip(stream.units.dropFirst(), stream.units).map { max(1, $0.dts - $1.dts) }
    let fallbackDuration = deltas.last ?? 3_003
    do {
      for index in stream.units.indices {
        try Task.checkCancellation()
        while !input.isReadyForMoreMediaData {
          if writer.status == .failed || writer.status == .cancelled {
            throw RemuxError.writer(writer.error?.localizedDescription ?? "writer stopped")
          }
          try await Task.sleep(nanoseconds: 250_000)
        }
        let duration =
          index + 1 < stream.units.count
          ? max(1, stream.units[index + 1].dts - stream.units[index].dts)
          : fallbackDuration
        let sample = try makeSampleBuffer(
          unit: stream.units[index],
          format: format,
          baseDTS: baseDTS,
          duration: duration
        )
        guard input.append(sample) else {
          throw RemuxError.writer(writer.error?.localizedDescription ?? "append failed")
        }
      }
      input.markAsFinished()
      await withCheckedContinuation { continuation in
        writer.finishWriting { continuation.resume() }
      }
      guard writer.status == .completed else {
        throw RemuxError.writer(writer.error?.localizedDescription ?? "finishWriting failed")
      }
    } catch {
      writer.cancelWriting()
      try? FileManager.default.removeItem(at: outputURL)
      throw error
    }
  }
}

/// Builds one timestamp-normalized movie from adjacent HLS media resources.
/// MPEG-TS resources are first remuxed independently because many providers
/// restart PTS/DTS at every segment. AVMutableComposition then places every
/// normalized track on one continuous timeline without decoding or re-encoding.
enum IPadHLSIntervalAssembler {
  static let maximumInputCount = 8

  struct Result: Sendable {
    let sourceOffsets: [TimeInterval]
    let sourceDurations: [TimeInterval]
    let duration: TimeInterval
  }

  enum AssemblyError: LocalizedError {
    case invalidInput(String)
    case unsupported(String)
    case decode(String)
    case export(String)

    var errorDescription: String? {
      switch self {
      case .invalidInput(let detail):
        return "HLS連結入力が不正です: \(detail)"
      case .unsupported(let detail):
        return "HLS区間を連結できません: \(detail)"
      case .decode(let detail):
        return "HLS連結動画をデコードできません: \(detail)"
      case .export(let detail):
        return "HLS連結動画を作成できません: \(detail)"
      }
    }
  }

  static func concatenate(
    inputURLs: [URL],
    outputURL: URL,
    temporaryDirectory: URL
  ) async throws -> Result {
    try Task.checkCancellation()
    guard !inputURLs.isEmpty, inputURLs.count <= maximumInputCount else {
      throw AssemblyError.invalidInput("1〜\(maximumInputCount)区間を指定してください")
    }
    guard outputURL.isFileURL, temporaryDirectory.isFileURL,
      inputURLs.allSatisfy(\.isFileURL)
    else {
      throw AssemblyError.invalidInput("ローカルファイルだけを処理できます")
    }

    try FileManager.default.createDirectory(
      at: temporaryDirectory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )

    if inputURLs.allSatisfy({
      $0.pathExtension.lowercased() == "ts"
        || IPadMPEGTSRemuxer.appearsToBeTransportStream($0)
    }) {
      let result = try await IPadMPEGTSRemuxer.remux(
        inputURLs: inputURLs,
        outputURL: outputURL
      )
      return Result(
        sourceOffsets: result.sourceOffsets,
        sourceDurations: result.sourceDurations,
        duration: result.duration
      )
    }

    if inputURLs.count == 1, let inputURL = inputURLs.first {
      let asset = AVURLAsset(url: inputURL)
      let track = try await loadVideoTrack(
        from: asset,
        detail: "単一区間"
      )
      let timeRange = try await loadTimeRange(
        from: track,
        detail: "単一区間"
      )
      let start = timeRange.start.seconds
      let duration = timeRange.duration.seconds
      guard timeRange.start.isNumeric, timeRange.duration.isNumeric,
        start.isFinite, start >= 0, duration.isFinite, duration > 0
      else {
        throw AssemblyError.invalidInput("区間の時間情報が不正です")
      }
      try? FileManager.default.removeItem(at: outputURL)
      do {
        try FileManager.default.copyItem(at: inputURL, to: outputURL)
      } catch {
        try? FileManager.default.removeItem(at: outputURL)
        throw AssemblyError.export(
          "単一区間のコピー: \(diagnostic(error))"
        )
      }
      return Result(
        sourceOffsets: [start],
        sourceDurations: [duration],
        duration: start + duration
      )
    }

    var normalizedURLs: [URL] = []
    var temporaryURLs: [URL] = []
    var completed = false
    defer {
      for url in temporaryURLs {
        try? FileManager.default.removeItem(at: url)
      }
      if !completed {
        try? FileManager.default.removeItem(at: outputURL)
      }
    }

    for (index, inputURL) in inputURLs.enumerated() {
      try Task.checkCancellation()
      let inputExtension = inputURL.pathExtension.lowercased()
      switch inputExtension {
      case "ts":
        let normalizedURL = temporaryDirectory.appendingPathComponent(
          "mioh-hls-normalized-\(index)-\(UUID().uuidString.lowercased()).mp4",
          isDirectory: false
        )
        try await IPadMPEGTSRemuxer.remux(
          inputURL: inputURL,
          outputURL: normalizedURL
        )
        normalizedURLs.append(normalizedURL)
        temporaryURLs.append(normalizedURL)
      case "mp4", "mov", "m4v", "m4s":
        normalizedURLs.append(inputURL)
      default:
        throw AssemblyError.unsupported(
          "\(inputExtension.isEmpty ? "拡張子なし" : inputExtension)形式"
        )
      }
    }

    let composition = AVMutableComposition()
    guard
      let compositionVideo = composition.addMutableTrack(
        withMediaType: .video,
        preferredTrackID: kCMPersistentTrackID_Invalid
      )
    else {
      throw AssemblyError.export("映像トラックを作成できません")
    }

    var cursor = CMTime.zero
    var sourceOffsets: [TimeInterval] = []
    var sourceDurations: [TimeInterval] = []
    for (index, inputURL) in normalizedURLs.enumerated() {
      try Task.checkCancellation()
      let asset = AVURLAsset(url: inputURL)
      let detail = "区間\(index + 1)"
      let videoTrack = try await loadVideoTrack(from: asset, detail: detail)
      let timeRange = try await loadTimeRange(from: videoTrack, detail: detail)
      let durationSeconds = timeRange.duration.seconds
      let offsetSeconds = cursor.seconds
      guard timeRange.start.isNumeric, timeRange.duration.isNumeric,
        durationSeconds.isFinite, durationSeconds > 0,
        offsetSeconds.isFinite, offsetSeconds >= 0
      else {
        throw AssemblyError.invalidInput("区間\(index + 1)の時間情報が不正です")
      }
      if index == 0 {
        do {
          compositionVideo.preferredTransform = try await videoTrack.load(
            .preferredTransform
          )
        } catch is CancellationError {
          throw CancellationError()
        } catch {
          throw AssemblyError.decode(
            "\(detail)の表示変換: \(diagnostic(error))"
          )
        }
      }
      sourceOffsets.append(offsetSeconds)
      sourceDurations.append(durationSeconds)
      do {
        try compositionVideo.insertTimeRange(
          timeRange,
          of: videoTrack,
          at: cursor
        )
      } catch {
        throw AssemblyError.export(
          "\(detail)の時間範囲挿入: \(diagnostic(error))"
        )
      }
      cursor = CMTimeAdd(cursor, timeRange.duration)
    }

    let durationSeconds = cursor.seconds
    guard durationSeconds.isFinite, durationSeconds > 0 else {
      throw AssemblyError.invalidInput("連結後の長さが不正です")
    }
    try? FileManager.default.removeItem(at: outputURL)
    guard
      let exporter = AVAssetExportSession(
        asset: composition,
        presetName: AVAssetExportPresetPassthrough
      )
    else {
      throw AssemblyError.export("passthrough exportを開始できません")
    }
    guard exporter.supportedFileTypes.contains(.mp4) else {
      throw AssemblyError.unsupported("MP4 passthrough出力")
    }
    exporter.outputURL = outputURL
    exporter.outputFileType = .mp4
    exporter.shouldOptimizeForNetworkUse = true
    let sendableExporter = IPadHLSIntervalExportSession(exporter)
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, Error>) in
        exporter.exportAsynchronously {
          let finished = sendableExporter.value
          switch finished.status {
          case .completed:
            continuation.resume()
          case .cancelled:
            continuation.resume(throwing: CancellationError())
          default:
            let failure = finished.error.map {
              AssemblyError.export(
                "passthrough export: \(diagnostic($0))"
              )
            } ?? AssemblyError.export("passthrough exportに失敗しました")
            continuation.resume(throwing: failure)
          }
        }
      }
    } onCancel: {
      sendableExporter.value.cancelExport()
    }
    try Task.checkCancellation()
    guard
      FileManager.default.fileExists(atPath: outputURL.path),
      let byteCount = (
        try FileManager.default.attributesOfItem(atPath: outputURL.path)[.size]
          as? NSNumber
      )?.intValue,
      byteCount > 0
    else {
      throw AssemblyError.export("出力ファイルが空です")
    }
    completed = true
    return Result(
      sourceOffsets: sourceOffsets,
      sourceDurations: sourceDurations,
      duration: durationSeconds
    )
  }

  /// Verifies the exact operation realtime restoration needs: a fresh
  /// AVAssetReader must be able to seek near the selected HLS interval and
  /// produce a decoded frame. A successful remux alone does not guarantee
  /// this on device, especially around independent HLS timestamp epochs.
  static func validateDecodableVideo(
    at inputURL: URL,
    near startSeconds: TimeInterval
  ) async throws {
    let asset = AVURLAsset(url: inputURL)
    let track: AVAssetTrack
    do {
      guard let first = try await asset.loadTracks(withMediaType: .video).first else {
        throw AssemblyError.decode("映像トラックがありません")
      }
      track = first
    } catch is CancellationError {
      throw CancellationError()
    } catch let error as AssemblyError {
      throw error
    } catch {
      throw AssemblyError.decode("映像情報: \(diagnostic(error))")
    }

    let timeRange: CMTimeRange
    do {
      timeRange = try await track.load(.timeRange)
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      throw AssemblyError.decode("時間情報: \(diagnostic(error))")
    }
    let trackStart = timeRange.start.seconds
    let trackEnd = CMTimeRangeGetEnd(timeRange).seconds
    guard timeRange.start.isNumeric, timeRange.duration.isNumeric,
      trackStart.isFinite, trackEnd.isFinite, trackEnd > trackStart,
      startSeconds.isFinite
    else {
      throw AssemblyError.decode("時間範囲が不正です")
    }
    let decodeStart = min(
      max(trackStart, startSeconds),
      max(trackStart, trackEnd - 0.001)
    )
    let decodeDuration = min(1, trackEnd - decodeStart)
    guard decodeDuration > 0 else {
      throw AssemblyError.decode("検証できる映像範囲がありません")
    }

    let reader: AVAssetReader
    do {
      reader = try AVAssetReader(asset: asset)
    } catch {
      throw AssemblyError.decode("reader作成: \(diagnostic(error))")
    }
    reader.timeRange = CMTimeRange(
      start: CMTime(seconds: decodeStart, preferredTimescale: 90_000),
      duration: CMTime(seconds: decodeDuration, preferredTimescale: 90_000)
    )
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String:
          Int(kCVPixelFormatType_32BGRA)
      ]
    )
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
      throw AssemblyError.decode("readerへ映像出力を追加できません")
    }
    reader.add(output)
    guard reader.startReading() else {
      throw AssemblyError.decode(
        "reader開始: \(reader.error.map(diagnostic) ?? "不明なエラー")"
      )
    }
    defer { reader.cancelReading() }
    while let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      if CMSampleBufferGetImageBuffer(sample) != nil { return }
    }
    if reader.status == .failed {
      throw AssemblyError.decode(
        "frame取得: \(reader.error.map(diagnostic) ?? "不明なエラー")"
      )
    }
    throw AssemblyError.decode("復号できるフレームがありません")
  }

  private static func loadVideoTrack(
    from asset: AVAsset,
    detail: String
  ) async throws -> AVAssetTrack {
    do {
      guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        throw AssemblyError.unsupported("\(detail)に映像トラックがありません")
      }
      return track
    } catch is CancellationError {
      throw CancellationError()
    } catch let error as AssemblyError {
      throw error
    } catch {
      throw AssemblyError.decode(
        "\(detail)の映像情報: \(diagnostic(error))"
      )
    }
  }

  private static func loadTimeRange(
    from track: AVAssetTrack,
    detail: String
  ) async throws -> CMTimeRange {
    do {
      return try await track.load(.timeRange)
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      throw AssemblyError.decode(
        "\(detail)の時間情報: \(diagnostic(error))"
      )
    }
  }

  private static func diagnostic(_ error: Error) -> String {
    let value = error as NSError
    return "\(error.localizedDescription) [\(value.domain):\(value.code)]"
  }
}

private final class IPadHLSIntervalExportSession: @unchecked Sendable {
  let value: AVAssetExportSession

  init(_ value: AVAssetExportSession) {
    self.value = value
  }
}
