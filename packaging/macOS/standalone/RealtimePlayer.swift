import AppKit
import AVFoundation
import AVKit
import CoreVideo
import Darwin
import Foundation
import Metal
import Network
import QuartzCore
import SceneKit
import SwiftUI

enum PreviewProjectionMode: String, CaseIterable, Identifiable {
  case normal = "通常"
  case vr180 = "VR180"
  case sphere360 = "360"

  var id: String { rawValue }
  var displayName: String { L(rawValue) }
}

enum PreviewVideoLayout: String, CaseIterable, Identifiable {
  case mono = "Mono"
  case sbs = "SBS 左右"
  case topBottom = "上下"

  var id: String { rawValue }
  var displayName: String { L(rawValue) }
}

enum PreviewEye: String, CaseIterable, Identifiable {
  case left = "左目"
  case right = "右目"

  var id: String { rawValue }
  var displayName: String { L(rawValue) }
}

private struct PreviewVRDetection {
  let projection: PreviewProjectionMode
  let layout: PreviewVideoLayout
  let reason: String

  var isVR: Bool { projection != .normal }
}

private enum PreviewVRDetector {
  static func detect(url: URL) async -> PreviewVRDetection {
    let path = url.path.lowercased()
    let filename = url.deletingPathExtension().lastPathComponent.lowercased()
    let asset = AVURLAsset(url: url)
    var metadataText = ""
    if let metadata = try? await asset.load(.metadata) {
      for item in metadata {
        if let identifier = item.identifier?.rawValue {
          metadataText += " " + identifier.lowercased()
        }
        if let key = item.commonKey?.rawValue {
          metadataText += " " + key.lowercased()
        }
        if let value = try? await item.load(.stringValue) {
          metadataText += " " + value.lowercased()
        }
      }
    }
    let containerText = await Task.detached(priority: .utility) {
      readContainerSignature(url: url)
    }.value
    let declaredEvidence = path + " " + metadataText
    let sphericalEvidence = metadataText + " " + containerText

    var width = 0.0
    var height = 0.0
    if let track = try? await asset.loadTracks(withMediaType: .video).first,
       let naturalSize = try? await track.load(.naturalSize) {
      let transform = (try? await track.load(.preferredTransform)) ?? .identity
      let transformed = naturalSize.applying(transform)
      width = abs(transformed.width)
      height = abs(transformed.height)
    }
    let aspect = height > 0 ? width / height : 0
    // 16:9 normal video is 1.778. A stereo VR180 frame is normally 2:1,
    // so do not let a generic wide frame become evidence for VR by itself.
    let isStereoVRShape = aspect >= 1.9 && aspect <= 2.1

    let has360Hint = containsAny(declaredEvidence, [
      "vr360", "360vr", "360_vr", "360-vr", "spherical=true",
      "sphericalvideo", "projectiontype=equirectangular", "projection_type=equirectangular",
    ])
    let hasExplicit180Hint = containsAny(declaredEvidence, [
      "vr180", "vr_180", "vr-180", "180vr", "180_vr", "180-vr",
    ])
    // MDVR is a useful product-code hint, but is too short to trust as an
    // unrestricted substring. Require the code in the filename and a 2:1
    // stereo frame before it can select VR180.
    let hasMDVRProductHint = filename.range(
      of: #"(^|[^a-z0-9])mdvr[-_ ]?[0-9]+"#,
      options: .regularExpression
    ) != nil
    let hasSphericalMetadata = containsAny(sphericalEvidence, [
      "gspherical", "spherical=true", "sv3d", "equirectangular",
    ])
    let hasStereoMetadata = containsAny(sphericalEvidence, [
      "st3d", "stereo_mode=sbs", "stereo_mode=top-bottom",
    ])
    let topBottom = containsAny(declaredEvidence, [
      "top-bottom", "top_bottom", "topbottom", "over-under", "over_under",
    ])
    let explicitSBS = containsAny(declaredEvidence, [
      "left-right", "left_right", "side-by-side", "side_by_side", "stereo_mode=sbs", " sbs ",
    ])
    let layout: PreviewVideoLayout = topBottom
      ? .topBottom
      : ((explicitSBS || isStereoVRShape) ? .sbs : .mono)

    if hasExplicit180Hint
      || (hasMDVRProductHint && isStereoVRShape)
      || (hasSphericalMetadata && hasStereoMetadata && isStereoVRShape) {
      return PreviewVRDetection(projection: .vr180, layout: layout, reason: L("VR180情報を検出"))
    }
    if has360Hint || hasSphericalMetadata {
      return PreviewVRDetection(projection: .sphere360, layout: layout, reason: L("全天球メタデータを検出"))
    }
    return PreviewVRDetection(projection: .normal, layout: .mono, reason: L("通常動画"))
  }

  private static func containsAny(_ value: String, _ candidates: [String]) -> Bool {
    candidates.contains { value.contains($0) }
  }

  private static func readContainerSignature(url: URL) -> String {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return "" }
    defer { try? handle.close() }
    let sampleSize = 8 * 1024 * 1024
    let head = (try? handle.read(upToCount: sampleSize)) ?? Data()
    var samples = [head]
    if let size = try? handle.seekToEnd(), size > UInt64(sampleSize) {
      let tailSize = UInt64(min(sampleSize, Int(size)))
      try? handle.seek(toOffset: size - tailSize)
      if let tail = try? handle.read(upToCount: Int(tailSize)) {
        samples.append(tail)
      }
    }
    var signatures: [String] = []
    for sample in samples {
      // Short four-character strings can occur by chance in compressed video.
      // Accept sv3d/st3d only when they are the type of a plausible MP4 box.
      if containsMP4Box(sample, type: "sv3d") { signatures.append("sv3d") }
      if containsMP4Box(sample, type: "st3d") { signatures.append("st3d") }
      let text = String(decoding: sample, as: UTF8.self).lowercased()
      for marker in [
        "gspherical", "spherical=true", "equirectangular",
        "projectiontype=equirectangular", "projection_type=equirectangular",
        "stereo_mode=sbs", "stereo_mode=top-bottom",
      ] where text.contains(marker) {
        signatures.append(marker)
      }
    }
    return signatures.joined(separator: " ")
  }

  private static func containsMP4Box(_ data: Data, type: String) -> Bool {
    let marker = Array(type.utf8)
    guard marker.count == 4, data.count >= 8 else { return false }
    return data.withUnsafeBytes { rawBuffer in
      let bytes = rawBuffer.bindMemory(to: UInt8.self)
      for index in 4...(bytes.count - 4) {
        guard bytes[index] == marker[0], bytes[index + 1] == marker[1],
          bytes[index + 2] == marker[2], bytes[index + 3] == marker[3]
        else { continue }
        let size = UInt32(bytes[index - 4]) << 24
          | UInt32(bytes[index - 3]) << 16
          | UInt32(bytes[index - 2]) << 8
          | UInt32(bytes[index - 1])
        if size == 0 || size == 1 || (size >= 8 && Int(size) <= data.count - index + 4) {
          return true
        }
      }
      return false
    }
  }
}

enum RealtimePlayerState: String {
  case idle
  case loading
  case buffering
  case playing
  case paused
  case seeking
  case ended
  case failed

  var label: String {
    switch self {
    case .idle: return L("待機中")
    case .loading: return L("読み込み中")
    case .buffering: return L("バッファ中")
    case .playing: return L("再生中")
    case .paused: return L("一時停止")
    case .seeking: return L("シーク中")
    case .ended: return L("再生終了")
    case .failed: return L("エラー")
    }
  }
}

struct PreviewWorkerEvent: Decodable {
  let kind: String
  let generation: Int
  let sequence: Int?
  let startNs: Int64?
  let endNs: Int64?
  let path: String?
  let duration: Double?
  let fps: Double?
  let width: Int?
  let height: Int?
  let message: String?
  let detail: String?
  let positionNs: Int64?
  let seconds: Double?
  let segmentSeconds: Double?
  let codec: String?
}

private struct PreviewSegment {
  let sequence: Int
  let startSeconds: Double
  let endSeconds: Double
  let url: URL
}

/// Stable, file-oriented events exposed to optional consumers such as the LAN
/// streaming adapter. The restoration worker remains the sole producer; a
/// consumer must retain a segment immediately because normal local playback
/// deletes consumed files from the rolling queue.
struct RealtimeStreamingSource: Sendable {
  let generation: Int
  let inputURL: URL
  let ffmpegURL: URL
  let segmentSeconds: Double
}

struct RealtimeStreamingSegment: Sendable {
  let generation: Int
  let sequence: Int
  let startSeconds: Double
  let endSeconds: Double
  let url: URL
  let codec: String
}

enum RealtimeStreamingEvent: Sendable {
  case reset(RealtimeStreamingSource)
  case segment(RealtimeStreamingSegment)
  case ended(generation: Int)
  case stopped(generation: Int)
}

typealias RealtimeStreamingEventConsumer = @MainActor (RealtimeStreamingEvent) -> Void

private struct PreparedSourcePlayerItem {
  let item: AVPlayerItem
  let duration: Double
  let resourceLoader: HEV1LoopbackServer?
  let processingInputURL: URL
  let compatibilityMode: SourceCompatibilityMode

  var usesVirtualContainer: Bool { resourceLoader != nil }
}

private enum SourceCompatibilityMode: Equatable {
  case direct
  case virtualHEV1
  case remuxed
  case transcoded
}

private enum SourcePlaybackError: LocalizedError {
  case missingHEV1SampleEntry
  case incompatibleVirtualContainer
  case invalidFileSize
  case loopbackServerFailed
  case missingFFmpeg
  case compatibilityConversionFailed

  var errorDescription: String? {
    switch self {
    case .missingHEV1SampleEntry:
      return L("AVFoundation互換にできるHEV1トラックが見つかりません")
    case .incompatibleVirtualContainer:
      return L("この動画はAVFoundationで再生できません")
    case .invalidFileSize:
      return L("動画ファイルの大きさを取得できません")
    case .loopbackServerFailed:
      return L("AVFoundation互換ストリーミングを開始できません")
    case .missingFFmpeg:
      return L("互換動画の作成に必要なFFmpegが見つかりません")
    case .compatibilityConversionFailed:
      return L("この動画をAVFoundation互換形式に変換できません")
    }
  }
}

/// A cancellable FFmpeg invocation used only when AVFoundation cannot open the
/// original container.  stderr is streamed to the app log, so a long-running
/// compatibility conversion never fills a pipe and deadlocks.
private final class SourceCompatibilityJob {
  let process = Process()
  private let errorPipe = Pipe()
  private let errorHandler: @Sendable (String) -> Void

  init(
    executable: URL,
    arguments: [String],
    errorHandler: @escaping @Sendable (String) -> Void
  ) {
    self.errorHandler = errorHandler
    process.executableURL = executable
    process.arguments = arguments
    process.standardInput = FileHandle.nullDevice
    process.standardOutput = FileHandle.nullDevice
    process.standardError = errorPipe
  }

  func run() async throws -> Int32 {
    errorPipe.fileHandleForReading.readabilityHandler = { [errorHandler] handle in
      let data = handle.availableData
      guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
      errorHandler(text)
    }
    return try await withCheckedThrowingContinuation { continuation in
      process.terminationHandler = { [errorPipe] completed in
        errorPipe.fileHandleForReading.readabilityHandler = nil
        continuation.resume(returning: completed.terminationStatus)
      }
      do {
        try process.run()
      } catch {
        errorPipe.fileHandleForReading.readabilityHandler = nil
        continuation.resume(throwing: error)
      }
    }
  }

  func cancel() {
    if process.isRunning {
      process.terminate()
    }
  }
}

/// Exposes a local HEV1 MP4 through an HTTP byte-range endpoint while replacing
/// only the sample-entry identifier in the moov atom.  AVFoundation treats a
/// custom URL resource loader as one all-data request for some large MP4 files;
/// that makes it wait for multi-gigabyte files and time out before inspection.
/// HTTP range semantics let AVFoundation request only the moov atom and samples
/// required for the current playback position.
private final class HEV1LoopbackServer {
  private static let replacement: [UInt8] = [0x68, 0x76, 0x63, 0x31] // hvc1
  private static let readChunkSize = 2 * 1024 * 1024
  private static let maximumRequestHeaderSize = 64 * 1024

  let sourceURL: URL
  let fileSize: UInt64
  let patchOffsets: [UInt64]
  private let listener: NWListener
  private let queue = DispatchQueue(label: "com.okatti.mioh.hev1-loopback", qos: .userInitiated)
  private var port: NWEndpoint.Port?

  init(sourceURL: URL) throws {
    self.sourceURL = sourceURL
    let attributes = try FileManager.default.attributesOfItem(atPath: sourceURL.path)
    guard let size = attributes[.size] as? NSNumber else {
      throw SourcePlaybackError.invalidFileSize
    }
    fileSize = size.uint64Value
    patchOffsets = try Self.findHEV1Offsets(in: sourceURL, fileSize: fileSize)
    guard !patchOffsets.isEmpty else { throw SourcePlaybackError.missingHEV1SampleEntry }
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
    listener = try NWListener(using: parameters)

    let ready = DispatchSemaphore(value: 0)
    let startupLock = NSLock()
    var startupError: Error?
    var startupFinished = false
    listener.stateUpdateHandler = { [weak listener] state in
      switch state {
      case .ready:
        startupLock.lock()
        if !startupFinished {
          startupFinished = true
          ready.signal()
        }
        startupLock.unlock()
      case .failed(let error):
        startupLock.lock()
        startupError = error
        if !startupFinished {
          startupFinished = true
          ready.signal()
        }
        startupLock.unlock()
      case .cancelled:
        _ = listener
      default:
        break
      }
    }
    listener.newConnectionHandler = { [weak self] connection in
      self?.accept(connection)
    }
    listener.start(queue: queue)
    guard ready.wait(timeout: .now() + 5) == .success else {
      listener.cancel()
      throw SourcePlaybackError.loopbackServerFailed
    }
    if startupError != nil {
      listener.cancel()
      throw SourcePlaybackError.loopbackServerFailed
    }
    guard let selectedPort = listener.port else {
      listener.cancel()
      throw SourcePlaybackError.loopbackServerFailed
    }
    port = selectedPort
  }

  deinit {
    listener.cancel()
  }

  func makeAsset() -> AVURLAsset {
    let virtualURL = URL(string: "http://127.0.0.1:\(port!.rawValue)/video.mp4")!
    return AVURLAsset(url: virtualURL)
  }

  private func accept(_ connection: NWConnection) {
    connection.stateUpdateHandler = { state in
      if case .failed = state { connection.cancel() }
    }
    connection.start(queue: queue)
    receiveRequest(on: connection, accumulated: Data())
  }

  private func receiveRequest(on connection: NWConnection, accumulated: Data) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 16 * 1024) {
      [weak self] data, _, isComplete, error in
      guard let self else {
        connection.cancel()
        return
      }
      var request = accumulated
      if let data { request.append(data) }
      if request.range(of: Data("\r\n\r\n".utf8)) != nil {
        self.respond(to: request, on: connection)
        return
      }
      if error != nil || isComplete || request.count > Self.maximumRequestHeaderSize {
        connection.cancel()
        return
      }
      self.receiveRequest(on: connection, accumulated: request)
    }
  }

  private func respond(to requestData: Data, on connection: NWConnection) {
    guard let request = String(data: requestData, encoding: .utf8) else {
      sendError(status: 400, reason: "Bad Request", on: connection)
      return
    }
    let lines = request.components(separatedBy: "\r\n")
    let requestParts = lines.first?.split(separator: " ") ?? []
    guard requestParts.count >= 2 else {
      sendError(status: 400, reason: "Bad Request", on: connection)
      return
    }
    let method = requestParts[0].uppercased()
    guard method == "GET" || method == "HEAD" else {
      sendError(status: 405, reason: "Method Not Allowed", on: connection)
      return
    }
    let rangeLine = lines.first { $0.lowercased().hasPrefix("range:") }
    let requestedRange = rangeLine.flatMap(parseRangeHeader)
    if rangeLine != nil && requestedRange == nil {
      sendError(status: 416, reason: "Range Not Satisfiable", on: connection)
      return
    }
    let start = requestedRange?.lowerBound ?? 0
    let end = requestedRange?.upperBound ?? (fileSize - 1)
    let partial = requestedRange != nil
    let length = end - start + 1
    var header = partial
      ? "HTTP/1.1 206 Partial Content\r\n"
      : "HTTP/1.1 200 OK\r\n"
    header += "Content-Type: video/mp4\r\n"
    header += "Accept-Ranges: bytes\r\n"
    header += "Content-Length: \(length)\r\n"
    if partial {
      header += "Content-Range: bytes \(start)-\(end)/\(fileSize)\r\n"
    }
    header += "Cache-Control: no-store\r\n"
    header += "Connection: close\r\n\r\n"
    connection.send(
      content: Data(header.utf8),
      completion: .contentProcessed { [weak self] error in
        guard error == nil, method == "GET", let self else {
          connection.cancel()
          return
        }
        do {
          let handle = try FileHandle(forReadingFrom: self.sourceURL)
          self.stream(
            connection: connection,
            handle: handle,
            cursor: start,
            endInclusive: end
          )
        } catch {
          connection.cancel()
        }
      }
    )
  }

  private func parseRangeHeader(_ line: String) -> ClosedRange<UInt64>? {
    guard let separator = line.firstIndex(of: ":") else { return nil }
    let value = line[line.index(after: separator)...]
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard value.lowercased().hasPrefix("bytes=") else { return nil }
    let spec = value.dropFirst(6).split(separator: ",", maxSplits: 1)[0]
    let bounds = spec.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
    guard bounds.count == 2, let start = UInt64(bounds[0]), start < fileSize else {
      return nil
    }
    let requestedEnd = bounds[1].isEmpty ? fileSize - 1 : UInt64(bounds[1])
    guard let requestedEnd, requestedEnd >= start else { return nil }
    return start...min(requestedEnd, fileSize - 1)
  }

  private func sendError(status: Int, reason: String, on connection: NWConnection) {
    let response =
      "HTTP/1.1 \(status) \(reason)\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    connection.send(content: Data(response.utf8), isComplete: true, completion: .idempotent)
  }

  private func stream(
    connection: NWConnection,
    handle: FileHandle,
    cursor: UInt64,
    endInclusive: UInt64
  ) {
    guard cursor <= endInclusive else {
      try? handle.close()
      connection.cancel()
      return
    }
    do {
      let remaining = endInclusive - cursor + 1
      let count = Int(min(UInt64(Self.readChunkSize), remaining))
      try handle.seek(toOffset: cursor)
      guard let sourceData = try handle.read(upToCount: count), !sourceData.isEmpty else {
        throw CocoaError(.fileReadUnknown)
      }
      let next = cursor + UInt64(sourceData.count)
      let complete = next > endInclusive
      connection.send(
        content: patch(sourceData, startingAt: cursor),
        isComplete: complete,
        completion: .contentProcessed { [weak self] error in
          guard error == nil, !complete, let self else {
            try? handle.close()
            if error != nil { connection.cancel() }
            return
          }
          self.stream(
            connection: connection,
            handle: handle,
            cursor: next,
            endInclusive: endInclusive
          )
        }
      )
    } catch {
      try? handle.close()
      connection.cancel()
    }
  }

  private func patch(_ data: Data, startingAt start: UInt64) -> Data {
    var bytes = [UInt8](data)
    let end = start + UInt64(bytes.count)
    for patchOffset in patchOffsets where patchOffset < end && patchOffset + 4 > start {
      for byteIndex in 0..<4 {
        let absoluteOffset = patchOffset + UInt64(byteIndex)
        guard absoluteOffset >= start, absoluteOffset < end else { continue }
        bytes[Int(absoluteOffset - start)] = Self.replacement[byteIndex]
      }
    }
    return Data(bytes)
  }

  private static func findHEV1Offsets(in url: URL, fileSize: UInt64) throws -> [UInt64] {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var offset: UInt64 = 0
    while offset + 8 <= fileSize {
      try handle.seek(toOffset: offset)
      guard let header = try handle.read(upToCount: 16), header.count >= 8 else { break }
      var atomSize = readUInt32BE(header, at: 0)
      var headerSize: UInt64 = 8
      if atomSize == 1 {
        guard header.count >= 16 else { break }
        atomSize = readUInt64BE(header, at: 8)
        headerSize = 16
      } else if atomSize == 0 {
        atomSize = fileSize - offset
      }
      guard atomSize >= headerSize, atomSize <= fileSize - offset else { break }
      let type = String(data: header[4..<8], encoding: .ascii) ?? ""
      if type == "moov" {
        guard atomSize <= 256 * 1024 * 1024 else {
          throw SourcePlaybackError.incompatibleVirtualContainer
        }
        try handle.seek(toOffset: offset)
        guard let moov = try handle.read(upToCount: Int(atomSize)) else { return [] }
        let bytes = [UInt8](moov)
        guard bytes.count >= 4 else { return [] }
        var result: [UInt64] = []
        for index in 0...(bytes.count - 4)
        where bytes[index] == 0x68 && bytes[index + 1] == 0x65
          && bytes[index + 2] == 0x76 && bytes[index + 3] == 0x31
        {
          result.append(offset + UInt64(index))
        }
        return result
      }
      offset += atomSize
    }
    return []
  }

  private static func readUInt32BE(_ data: Data, at offset: Int) -> UInt64 {
    data[offset..<(offset + 4)].reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
  }

  private static func readUInt64BE(_ data: Data, at offset: Int) -> UInt64 {
    data[offset..<(offset + 8)].reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
  }
}

private enum PreviewProjectionGeometry {
  static func makeSphere(
    projection: PreviewProjectionMode,
    layout: PreviewVideoLayout,
    eye: PreviewEye
  ) -> SCNGeometry {
    let horizontalSegments = projection == .sphere360 ? 128 : 72
    let verticalSegments = 72
    let phiStart = projection == .sphere360 ? -Double.pi : Double.pi / 2
    let phiLength = projection == .sphere360 ? Double.pi * 2 : Double.pi
    let uv = uvWindow(layout: layout, eye: eye)
    let radius = 20.0

    var vertices: [SCNVector3] = []
    var texcoords: [CGPoint] = []
    var indices: [Int32] = []

    for y in 0...verticalSegments {
      let v = Double(y) / Double(verticalSegments)
      let theta = v * Double.pi
      for x in 0...horizontalSegments {
        let u = Double(x) / Double(horizontalSegments)
        let phi = phiStart + u * phiLength
        vertices.append(
          SCNVector3(
            radius * sin(theta) * sin(phi),
            radius * cos(theta),
            radius * sin(theta) * cos(phi)
          )
        )
        // The geometry is viewed from inside the sphere. Its increasing
        // longitude runs right-to-left from the camera, so using increasing
        // texture U mirrors the video. Reverse U inside the selected eye's
        // window while preserving the SBS/top-bottom eye assignment.
        texcoords.append(CGPoint(x: uv.maxX - uv.width * u, y: uv.minY + uv.height * v))
      }
    }

    let stride = horizontalSegments + 1
    for y in 0..<verticalSegments {
      for x in 0..<horizontalSegments {
        let a = Int32(y * stride + x)
        let b = Int32(y * stride + x + 1)
        let c = Int32((y + 1) * stride + x)
        let d = Int32((y + 1) * stride + x + 1)
        indices.append(contentsOf: [a, c, b, b, c, d])
      }
    }

    return SCNGeometry(
      sources: [
        SCNGeometrySource(vertices: vertices),
        SCNGeometrySource(textureCoordinates: texcoords),
      ],
      elements: [SCNGeometryElement(indices: indices, primitiveType: .triangles)]
    )
  }

  static func uvWindow(layout: PreviewVideoLayout, eye: PreviewEye) -> CGRect {
    switch layout {
    case .mono:
      return CGRect(x: 0, y: 0, width: 1, height: 1)
    case .sbs:
      return CGRect(x: eye == .right ? 0.5 : 0, y: 0, width: 0.5, height: 1)
    case .topBottom:
      return CGRect(x: 0, y: eye == .right ? 0 : 0.5, width: 1, height: 0.5)
    }
  }
}

struct VRPreviewSceneView: NSViewRepresentable {
  let playerItem: AVPlayerItem?
  let projection: PreviewProjectionMode
  let layout: PreviewVideoLayout
  let eye: PreviewEye
  let cameraFOV: Double

  func makeCoordinator() -> Coordinator {
    Coordinator()
  }

  func makeNSView(context: Context) -> SCNView {
    context.coordinator.makeView()
  }

  func updateNSView(_ nsView: SCNView, context: Context) {
    context.coordinator.update(
      playerItem: playerItem,
      projection: projection,
      layout: layout,
      eye: eye,
      cameraFOV: cameraFOV
    )
  }

  final class Coordinator: NSObject, SCNSceneRendererDelegate {
    private let scene = SCNScene()
    private let cameraNode = SCNNode()
    private let videoNode = SCNNode()
    private weak var displayedItem: AVPlayerItem?
    private var videoOutput: AVPlayerItemVideoOutput?
    private var textureCache: CVMetalTextureCache?
    private var currentPixelBuffer: CVPixelBuffer?
    private var currentVideoTexture: CVMetalTexture?
    private let textureLock = NSLock()
    private var currentProjection: PreviewProjectionMode = .vr180
    private var currentLayout: PreviewVideoLayout = .sbs
    private var currentEye: PreviewEye = .left
    private var yaw: CGFloat = 0
    private var pitch: CGFloat = 0

    deinit {
      if let displayedItem, let videoOutput {
        displayedItem.remove(videoOutput)
      }
      textureLock.withLock {
        currentPixelBuffer = nil
        currentVideoTexture = nil
        if let textureCache {
          CVMetalTextureCacheFlush(textureCache, 0)
        }
      }
    }

    func makeView() -> SCNView {
      let view = SCNView(
        frame: .zero,
        options: [SCNView.Option.preferredRenderingAPI.rawValue: SCNRenderingAPI.metal.rawValue]
      )
      view.scene = scene
      view.backgroundColor = NSColor.black
      view.allowsCameraControl = false
      view.rendersContinuously = true
      view.isPlaying = true
      view.preferredFramesPerSecond = 60
      view.antialiasingMode = .none
      view.delegate = self
      if let device = view.device ?? MTLCreateSystemDefaultDevice() {
        var cache: CVMetalTextureCache?
        if CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
          == kCVReturnSuccess
        {
          textureCache = cache
        }
      }

      cameraNode.camera = SCNCamera()
      cameraNode.camera?.fieldOfView = 60
      cameraNode.position = SCNVector3(0, 0, 0)
      scene.rootNode.addChildNode(cameraNode)
      scene.rootNode.addChildNode(videoNode)

      let pan = NSPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
      view.addGestureRecognizer(pan)
      let magnify = NSMagnificationGestureRecognizer(target: self, action: #selector(handleMagnify(_:)))
      view.addGestureRecognizer(magnify)
      rebuildGeometry()
      return view
    }

    func update(
      playerItem: AVPlayerItem?,
      projection: PreviewProjectionMode,
      layout: PreviewVideoLayout,
      eye: PreviewEye,
      cameraFOV: Double
    ) {
      let geometryChanged = projection != currentProjection || layout != currentLayout || eye != currentEye
      currentProjection = projection
      currentLayout = layout
      currentEye = eye
      cameraNode.camera?.fieldOfView = min(max(cameraFOV, 45), 105)
      if geometryChanged {
        rebuildGeometry()
      }
      attachVideoOutputIfNeeded(to: playerItem)
    }

    private func rebuildGeometry() {
      videoNode.geometry = PreviewProjectionGeometry.makeSphere(
        projection: currentProjection,
        layout: currentLayout,
        eye: currentEye
      )
      let material = SCNMaterial()
      material.isDoubleSided = true
      material.lightingModel = .constant
      material.diffuse.contents = NSColor.black
      material.diffuse.minificationFilter = .linear
      material.diffuse.magnificationFilter = .linear
      material.diffuse.mipFilter = .none
      videoNode.geometry?.materials = [material]
      if let currentVideoTexture, let metalTexture = CVMetalTextureGetTexture(currentVideoTexture) {
        material.diffuse.contents = metalTexture
      }
    }

    private func attachVideoOutputIfNeeded(to item: AVPlayerItem?) {
      guard displayedItem !== item else { return }
      if let displayedItem, let videoOutput {
        displayedItem.remove(videoOutput)
      }
      displayedItem = item
      textureLock.withLock {
        videoOutput = nil
        videoNode.geometry?.firstMaterial?.diffuse.contents = NSColor.black
        currentPixelBuffer = nil
        currentVideoTexture = nil
        if let textureCache {
          // AVPlayerItem replacement is a generation boundary. Releasing the
          // old texture before flushing prevents 8K seek generations from
          // accumulating stale IOSurface-backed cache entries.
          CVMetalTextureCacheFlush(textureCache, 0)
        }
      }
      guard let item else { return }
      let attributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferMetalCompatibilityKey as String: true,
      ]
      let output = AVPlayerItemVideoOutput(pixelBufferAttributes: attributes)
      item.add(output)
      output.requestNotificationOfMediaDataChange(withAdvanceInterval: 1.0 / 60.0)
      textureLock.withLock {
        videoOutput = output
      }
    }

    func renderer(_ renderer: any SCNSceneRenderer, updateAtTime time: TimeInterval) {
      textureLock.lock()
      defer { textureLock.unlock() }
      guard let videoOutput, let textureCache else { return }
      let itemTime = videoOutput.itemTime(forHostTime: CACurrentMediaTime())
      guard videoOutput.hasNewPixelBuffer(forItemTime: itemTime),
        let pixelBuffer = videoOutput.copyPixelBuffer(
          forItemTime: itemTime,
          itemTimeForDisplay: nil
        )
      else {
        return
      }

      let width = CVPixelBufferGetWidth(pixelBuffer)
      let height = CVPixelBufferGetHeight(pixelBuffer)
      var videoTexture: CVMetalTexture?
      let status = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        textureCache,
        pixelBuffer,
        nil,
        .bgra8Unorm_srgb,
        width,
        height,
        0,
        &videoTexture
      )
      guard status == kCVReturnSuccess, let videoTexture,
        let metalTexture = CVMetalTextureGetTexture(videoTexture)
      else {
        return
      }

      currentPixelBuffer = pixelBuffer
      currentVideoTexture = videoTexture
      videoNode.geometry?.firstMaterial?.diffuse.contents = metalTexture
    }

    @objc private func handlePan(_ gesture: NSPanGestureRecognizer) {
      let translation = gesture.translation(in: gesture.view)
      gesture.setTranslation(.zero, in: gesture.view)
      yaw -= translation.x * 0.006
      pitch -= translation.y * 0.006
      pitch = max(-CGFloat.pi / 2 + 0.01, min(CGFloat.pi / 2 - 0.01, pitch))
      cameraNode.eulerAngles = SCNVector3(pitch, yaw, 0)
    }

    @objc private func handleMagnify(_ gesture: NSMagnificationGestureRecognizer) {
      guard let camera = cameraNode.camera else { return }
      camera.fieldOfView = max(45, min(105, camera.fieldOfView - Double(gesture.magnification) * 18))
      gesture.magnification = 0
    }
  }
}

@MainActor
final class RealtimePlayerController: ObservableObject {
  private static weak var activeRestorationController:
    RealtimePlayerController?

  @Published var state: RealtimePlayerState = .idle
  @Published var previewInputURL: URL?
  @Published var position = 0.0
  @Published var duration = 0.0
  @Published var bufferedSeconds = 0.0
  @Published var showOriginal = false
  @Published var volume = 1.0
  @Published var muted = false
  @Published var errorMessage = ""
  @Published var playbackDetail = ""
  @Published var sourceOnlyPlayback = false
  @Published var isVRVideo = false
  @Published var isDetectingVR = false
  @Published var vrDetectionDetail = ""

  let sourcePlayer = AVPlayer()
  let restoredPlayer = AVQueuePlayer()
  let startupSegmentCount = 3
  let rebufferSegmentCount = 2
  let hlsVODStartupSegmentCount = 3
  let hlsVODRebufferSegmentCount = 2
  let driftToleranceSeconds = 0.080
  let driftCorrectionGraceSeconds = 0.350
  let hlsDriftCorrectionGraceSeconds = 0.120
  let hlsDriftToleranceSeconds = 0.120
  let hlsDriftResumeToleranceSeconds = 0.050
  let hlsDriftSeekToleranceSeconds = 0.100
  let hlsClockObservationIntervalSeconds = 0.080
  let hlsOperationWatchdogSeconds = 2.0
  let hlsMaximumInitialSeekAttempts = 3
  let hlsMaximumSynchronizedStartAttempts = 3

  private var worker: Process?
  private var workerRetirementTask: Task<Void, Never>?
  private var workerInput: Pipe?
  private var stdoutPipe: Pipe?
  private var stderrPipe: Pipe?
  private var stdoutBuffer = ""
  private var generation = 0
  private var nextSequence = 0
  private var releasedThroughSequence = -1
  private var queuedSegments: [PreviewSegment] = []
  private var sourceBufferedSeconds = 0.0
  private var itemSegments: [ObjectIdentifier: PreviewSegment] = [:]
  private var notificationTokens: [NSObjectProtocol] = []
  private var itemEndNotificationTokens: [ObjectIdentifier: NSObjectProtocol] = [:]
  private var timeObserver: Any?
  private var restoredTimeObserver: Any?
  private var sourceItemStatusObservation: NSKeyValueObservation?
  private var sourceTimeControlObservation: NSKeyValueObservation?
  private var sourceLoadedTimeRangesObservation: NSKeyValueObservation?
  private var hlsSourceSeekableTimeRangesObservation: NSKeyValueObservation?
  private var hlsNotificationTokens: [NSObjectProtocol] = []
  private var sourceResourceLoader: HEV1LoopbackServer?
  private var sourceProcessingInputURL: URL?
  private var sourceCompatibilityDirectory: URL?
  private var sourceCompatibilityJob: SourceCompatibilityJob?
  private var sessionDirectory: URL?
  private var hlsSource: IPadResolvedMediaSource?
  private var hlsProductionTask: Task<Void, Never>?
  private var hlsProducer: MacHLSRealtimeProducer?
  private var hlsProducerRetirementTask: Task<Void, Never>?
  private var hlsMediaProxy: IPadAuthenticatedMediaProxy?
  private var hlsSourceIsReady = false
  private var hlsInitialSeekCompleted = false
  private var hlsSourceReady = false
  private var hlsSourceSeekCompleted = false
  private var hlsSourceTimeOffset = 0.0
  private var hlsSeekInFlight = false
  private var hlsSeekRevision = 0
  private var hlsSeekToLiveWindowStart = false
  private var hlsSourceReachedEnd = false
  private var hlsRestoredClockFallbackActive = false
  private var requestedStartSeconds = 0.0
  private var shouldPlay = true
  private var generationHasStarted = false
  private var generationStartPending = false
  private var generationReachedEOF = false
  private var sourceSeekNeedsBuffer = false
  private var previewSegmentSeconds = 2.0
  private var currentRestoredItemIdentifier: ObjectIdentifier?
  private var currentRestoredItemStartedAt = 0.0
  private var hlsRestoredHeldForSourceCatchup = false
  private var hlsDriftCorrectionInFlight = false
  private var hlsInitialSeekAttempt = 0
  private var hlsInitialSeekWatchdogTask: Task<Void, Never>?
  private var hlsSynchronizedStartRevision = 0
  private var hlsSynchronizedStartInFlight = false
  private var hlsSynchronizedStartAttempt = 0
  private var hlsSynchronizedStartWatchdogTask: Task<Void, Never>?
  private weak var runner: RestorationRunner?
  private var activePreviewSettingsSignature: String?
  private var streamingSource: RealtimeStreamingSource?
  private var streamingSegmentCodecs: [Int: String] = [:]
  private var streamingEventConsumer: RealtimeStreamingEventConsumer?

  var showsSourceFrameWhilePreparingRestoration: Bool {
    guard !sourceOnlyPlayback, !generationHasStarted else { return false }
    return state == .loading || state == .seeking || state == .buffering
  }

  var isHLSInput: Bool { hlsSource?.kind == .hls }

  var isLiveHLSInput: Bool {
    hlsSource?.hlsPlaylist?.isLive == true
  }

  var isSeekable: Bool { !isLiveHLSInput }

  var canShowOriginal: Bool {
    !hlsRestoredClockFallbackActive && !hlsSourceReachedEnd
  }

  init() {
    restoredPlayer.isMuted = true
    restoredPlayer.actionAtItemEnd = .advance
    sourcePlayer.volume = 1
    restoredTimeObserver = restoredPlayer.addPeriodicTimeObserver(
      forInterval: CMTime(
        seconds: hlsClockObservationIntervalSeconds,
        preferredTimescale: 600
      ),
      queue: .main
    ) { [weak self] time in
      Task { @MainActor in self?.tickRestored(seconds: time.seconds) }
    }
  }

  deinit {
    hlsProductionTask?.cancel()
    hlsInitialSeekWatchdogTask?.cancel()
    hlsSynchronizedStartWatchdogTask?.cancel()
    hlsProducer?.cancel()
    hlsMediaProxy?.stop()
    try? workerInput?.fileHandleForWriting.close()
    if let worker, worker.isRunning {
      let processIdentifier = worker.processIdentifier
      if kill(-processIdentifier, SIGTERM) != 0 {
        worker.terminate()
      }
      // A controller can disappear when SwiftUI replaces a window/view
      // without calling the user-facing stop action. Do not leave its Core AI
      // descendants resident indefinitely if graceful termination stalls.
      DispatchQueue.global(qos: .utility).asyncAfter(
        deadline: .now() + 1
      ) {
        if kill(processIdentifier, 0) == 0 {
          _ = kill(-processIdentifier, SIGKILL)
        }
      }
    }
    sourceCompatibilityJob?.cancel()
    sourceItemStatusObservation?.invalidate()
    sourceTimeControlObservation?.invalidate()
    sourceLoadedTimeRangesObservation?.invalidate()
    hlsSourceSeekableTimeRangesObservation?.invalidate()
    if let timeObserver {
      sourcePlayer.removeTimeObserver(timeObserver)
    }
    if let restoredTimeObserver {
      restoredPlayer.removeTimeObserver(restoredTimeObserver)
    }
    for token in notificationTokens {
      NotificationCenter.default.removeObserver(token)
    }
    for token in itemEndNotificationTokens.values {
      NotificationCenter.default.removeObserver(token)
    }
    for token in hlsNotificationTokens {
      NotificationCenter.default.removeObserver(token)
    }
  }

  func start(
    runner: RestorationRunner,
    at startSeconds: Double = 0,
    autoPlay: Bool = true,
    preserveCurrentSource: Bool = false
  ) {
    let previousController = Self.activeRestorationController
    var previousControllerRetirement: Task<Void, Never>?
    var previousControllerHLSRetirement: Task<Void, Never>?
    if let previousController, previousController !== self {
      previousController.stop()
      previousControllerRetirement =
        previousController.workerRetirementTask
      previousControllerHLSRetirement =
        previousController.hlsProducerRetirementTask
    }
    Self.activeRestorationController = self
    let canReuseCurrentSource = preserveCurrentSource && sourcePlayer.currentItem != nil
    stop(
      preserveSourceItem: canReuseCurrentSource,
      preserveHLSSelection: false
    )
    Self.activeRestorationController = self
    let retirement = workerRetirementTask
    let hlsRetirement = hlsProducerRetirementTask
    guard let input = previewInputURL else {
      fail("再生タブで入力動画を選択してください")
      return
    }
    do {
      let resources = try runner.resourceDirectory()
      let tempRoot: URL
      if runner.ladaTempDirectory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        tempRoot = FileManager.default.temporaryDirectory
      } else {
        tempRoot = URL(fileURLWithPath: runner.ladaTempDirectory, isDirectory: true)
      }
      let session = tempRoot.appendingPathComponent(
        "mioh-preview-\(UUID().uuidString)", isDirectory: true
      )
      try FileManager.default.createDirectory(at: session, withIntermediateDirectories: true)

      self.runner = runner
      activePreviewSettingsSignature = previewSettingsSignature(for: runner)
      sessionDirectory = session
      generation += 1
      let startingGeneration = generation
      nextSequence = 0
      releasedThroughSequence = -1
      requestedStartSeconds = startSeconds
      position = startSeconds
      shouldPlay = autoPlay
      generationHasStarted = false
      generationStartPending = false
      generationReachedEOF = false
      sourceSeekNeedsBuffer = false
      sourceOnlyPlayback = runner.previewProjectionMode != "通常"
      state = .loading
      errorMessage = ""
      playbackDetail = sourceOnlyPlayback ? "VR動画のコンテナを確認中" : ""

      if !sourceOnlyPlayback {
        let source = RealtimeStreamingSource(
          generation: startingGeneration,
          inputURL: input,
          ffmpegURL: resources.appendingPathComponent("bin/ffmpeg"),
          segmentSeconds: previewSegmentSeconds
        )
        streamingSource = source
        streamingSegmentCodecs.removeAll(keepingCapacity: true)
        streamingEventConsumer?(.reset(source))
      }

      if canReuseCurrentSource && !sourceOnlyPlayback {
        // Keep the already-decoded source item visible while the restoration
        // worker retires and refills. This gives immediate visual feedback at
        // the requested time without weakening the process-per-seek boundary.
        sourcePlayer.pause()
        sourcePlayer.seek(
          to: CMTime(seconds: startSeconds, preferredTimescale: 600),
          toleranceBefore: .zero,
          toleranceAfter: .zero
        ) { [weak self] _ in
          Task { @MainActor in
            guard let self, self.generation == startingGeneration else { return }
            self.position = startSeconds
          }
        }
      }

      if sourceOnlyPlayback {
        Task { @MainActor [self] in
          if let previousControllerRetirement {
            await previousControllerRetirement.value
          }
          if let previousControllerHLSRetirement {
            await previousControllerHLSRetirement.value
          }
          if let retirement {
            await retirement.value
          }
          if let hlsRetirement {
            await hlsRetirement.value
          }
          guard self.generation == startingGeneration, self.sourceOnlyPlayback else { return }
          self.startSourceOnlyPlayback(
            input: input,
            resources: resources,
            tempRoot: tempRoot,
            generation: startingGeneration,
            startSeconds: startSeconds
          )
        }
        return
      }

      Task { @MainActor [self] in
        if let previousControllerRetirement {
          await previousControllerRetirement.value
        }
        if let previousControllerHLSRetirement {
          await previousControllerHLSRetirement.value
        }
        if let retirement {
          await retirement.value
        }
        if let hlsRetirement {
          await hlsRetirement.value
        }
        guard self.generation == startingGeneration else { return }
        self.workerRetirementTask = nil
        let prepared: PreparedSourcePlayerItem?
        if canReuseCurrentSource {
          prepared = nil
        } else {
          do {
            prepared = try await self.prepareSourcePlayerItem(
              input: input,
              resources: resources,
              tempRoot: tempRoot
            )
          } catch {
            guard self.generation == startingGeneration else { return }
            self.fail("元動画を開けません: \(error.localizedDescription)")
            self.cleanupSession()
            self.cleanupSourceCompatibility()
            return
          }
        }
        guard self.generation == startingGeneration, self.state == .loading || self.state == .buffering else {
          return
        }
        let processingInput = prepared?.processingInputURL ?? self.sourceProcessingInputURL ?? input
        let process = Process()
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        do {
          if runner.usesPythonEngine {
            // The bundled interpreter speaks the same stdout event protocol as
            // the Swift pipeline, so only the launch differs.
            let python = resources.appendingPathComponent(
              "runtime/bin/python3.12"
            )
            let script = resources.appendingPathComponent(
              "runtime/lib/python3.12/site-packages/mioh_preview_worker.py"
            )
            guard FileManager.default.isExecutableFile(atPath: python.path) else {
              throw RunnerError.missingResource("Python runtime")
            }
            guard FileManager.default.fileExists(atPath: script.path) else {
              throw RunnerError.missingResource("Realtime preview worker")
            }
            process.executableURL = python
            process.arguments = [script.path] + (try runner.previewArguments(
              resources: resources,
              outputDirectory: session,
              input: processingInput
            )) + [
              "--start-ns", String(Int64(startSeconds * 1_000_000_000)),
              "--generation", String(startingGeneration),
            ]
            process.environment = runner.environment(
              resources: resources,
              python: python
            )
          } else {
            let invocation = try runner.nativePreviewInvocation(
              resources: resources,
              outputDirectory: session,
              input: processingInput,
              startNanoseconds: Int64(startSeconds * 1_000_000_000),
              generation: startingGeneration
            )
            let configurationURL = session.appendingPathComponent(
              "native-preview-configuration.json"
            )
            try invocation.configuration.write(
              to: configurationURL,
              options: .atomic
            )
            process.executableURL = invocation.executable
            process.arguments = [configurationURL.path]
            process.environment = invocation.environment
          }
        } catch {
          self.fail(error.localizedDescription)
          self.cleanupSession()
          return
        }
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        guard MacChildProcessPipe.prepare(inputPipe.fileHandleForWriting) else {
          self.fail("プレビューワーカーのstdinを安全に準備できません")
          self.cleanupSession()
          return
        }
        self.worker = process
        self.workerInput = inputPipe
        self.stdoutPipe = outputPipe
        self.stderrPipe = errorPipe
        if let prepared {
          self.sourceResourceLoader = prepared.resourceLoader
          self.sourceProcessingInputURL = prepared.processingInputURL
          self.sourcePlayer.replaceCurrentItem(with: prepared.item)
        }
        self.sourcePlayer.volume = self.muted ? 0 : Float(self.volume)
        if prepared?.usesVirtualContainer == true {
          self.runner?.appendExternalLog(
            "再生: AVFoundation互換の仮想コンテナを使用します（ファイル変換なし）\n"
          )
        } else if prepared?.compatibilityMode == .remuxed {
          self.runner?.appendExternalLog(
            "再生: AVFoundation互換MP4へremuxした動画を、表示・検出・復元で共有します\n"
          )
        } else if prepared?.compatibilityMode == .transcoded {
          self.runner?.appendExternalLog(
            "再生: 非対応コーデックをVideoToolboxで互換変換し、表示・検出・復元で共有します\n"
          )
        }

        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
          let data = handle.availableData
          guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
          Task { @MainActor in self?.consumeWorkerOutput(text) }
        }
        errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
          let data = handle.availableData
          guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
          Task { @MainActor in self?.runner?.appendExternalLog(text) }
        }
        process.terminationHandler = { [weak self, inputPipe, outputPipe, errorPipe] completed in
          Task { @MainActor in
            guard let self, self.worker === completed else { return }
            self.worker = nil
            if self.workerInput === inputPipe {
              self.workerInput = nil
              try? inputPipe.fileHandleForWriting.close()
            }
            // Let readabilityHandler deliver any final segment/error event
            // already buffered in the pipe before classifying the exit.
            try? await Task.sleep(nanoseconds: 100_000_000)
            outputPipe.fileHandleForReading.readabilityHandler = nil
            errorPipe.fileHandleForReading.readabilityHandler = nil
            if self.stdoutPipe === outputPipe { self.stdoutPipe = nil }
            if self.stderrPipe === errorPipe { self.stderrPipe = nil }
            guard self.sessionDirectory == session, self.worker == nil else { return }
            if self.state != .idle && self.state != .ended && self.state != .failed
              && !self.generationReachedEOF
            {
              self.fail(
                completed.terminationStatus == 0
                  ? "プレビューワーカーとの接続が終了しました"
                  : "プレビューワーカーが終了しました"
              )
            }
          }
        }
        self.installTimeObserver()
        do {
          try process.run()
        } catch {
          self.fail(error.localizedDescription)
          self.cleanupSession()
        }
      }
    } catch {
      fail(error.localizedDescription)
      cleanupSession()
    }
  }

  /// Starts inbound HLS restoration. AVPlayer receives an authenticated
  /// loopback playlist for the original audio/clock, while the producer
  /// materializes adjacent media resources into local MP4 windows before
  /// passing them to the existing Core AI preview worker.
  func startHLS(
    source: IPadResolvedMediaSource,
    runner: RestorationRunner,
    at startSeconds: Double = 0,
    autoPlay: Bool = true
  ) {
    guard source.kind == .hls,
      let playlist = source.hlsPlaylist,
      !playlist.segments.isEmpty,
      playlist.duration > 0
    else {
      fail("再生可能なHLSメディア区間がありません")
      return
    }

    let previousController = Self.activeRestorationController
    var previousControllerRetirement: Task<Void, Never>?
    var previousControllerHLSRetirement: Task<Void, Never>?
    if let previousController, previousController !== self {
      previousController.stop()
      previousControllerRetirement = previousController.workerRetirementTask
      previousControllerHLSRetirement =
        previousController.hlsProducerRetirementTask
    }
    stop(preserveHLSSelection: false)
    Self.activeRestorationController = self
    let localWorkerRetirement = workerRetirementTask
    let localHLSRetirement = hlsProducerRetirementTask

    do {
      let resources = try runner.resourceDirectory()
      let tempRoot: URL
      if runner.ladaTempDirectory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        tempRoot = FileManager.default.temporaryDirectory
      } else {
        tempRoot = URL(fileURLWithPath: runner.ladaTempDirectory, isDirectory: true)
      }
      let session = tempRoot.appendingPathComponent(
        "mioh-hls-preview-\(UUID().uuidString.lowercased())",
        isDirectory: true
      )
      try FileManager.default.createDirectory(
        at: session,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
      )

      let availableDuration = max(0.001, playlist.duration)
      let target = playlist.isLive
        ? max(0, playlist.segments.suffix(startupSegmentCount).first?.startSeconds ?? 0)
        : min(max(0, startSeconds), max(0, availableDuration - 0.001))
      self.runner = runner
      activePreviewSettingsSignature = previewSettingsSignature(for: runner)
      sessionDirectory = session
      hlsSource = source
      previewInputURL = source.playbackURL
      generation += 1
      let startingGeneration = generation
      nextSequence = 0
      releasedThroughSequence = -1
      requestedStartSeconds = target
      position = target
      duration = availableDuration
      bufferedSeconds = 0
      sourceBufferedSeconds = 0
      shouldPlay = autoPlay
      generationHasStarted = false
      generationStartPending = false
      generationReachedEOF = false
      sourceSeekNeedsBuffer = false
      hlsSourceIsReady = false
      hlsInitialSeekCompleted = false
      hlsSourceReady = false
      hlsSourceSeekCompleted = false
      hlsSourceTimeOffset = 0
      invalidateHLSInitialSeek()
      hlsSeekToLiveWindowStart = false
      hlsSourceReachedEnd = false
      hlsRestoredClockFallbackActive = false
      currentRestoredItemIdentifier = nil
      currentRestoredItemStartedAt = 0
      hlsRestoredHeldForSourceCatchup = false
      hlsDriftCorrectionInFlight = false
      hlsInitialSeekAttempt = 0
      cancelSynchronizedHLSStart()
      sourceOnlyPlayback = false
      isVRVideo = false
      isDetectingVR = false
      vrDetectionDetail = ""
      errorMessage = ""
      playbackDetail = playlist.isLive
        ? "ライブHLSの区間を取得中"
        : "HLS区間を取得・連結中"
      state = autoPlay ? .loading : .paused

      let streaming = RealtimeStreamingSource(
        generation: startingGeneration,
        inputURL: source.mediaURL,
        ffmpegURL: resources.appendingPathComponent("bin/ffmpeg"),
        segmentSeconds: previewSegmentSeconds
      )
      streamingSource = streaming
      streamingSegmentCodecs.removeAll(keepingCapacity: true)
      streamingEventConsumer?(.reset(streaming))

      let proxy = IPadAuthenticatedMediaProxy { [weak self] _ in
        Task { @MainActor [weak self] in
          guard let self, self.generation == startingGeneration else { return }
          self.degradeHLSSourcePlayback(
            generation: startingGeneration,
            reason: "HLSの認証情報を更新できませんでした"
          )
        }
      }
      hlsMediaProxy = proxy
      let producer = MacHLSRealtimeProducer(
        source: source,
        runner: runner,
        resources: resources,
        sessionDirectory: session,
        startSeconds: target,
        generation: startingGeneration,
        log: { [weak runner] text in
          Task { @MainActor in runner?.appendExternalLog(text) }
        }
      )
      producer.updateOutputBufferLimits(
        hlsOutputBufferLimits(for: runner.previewBufferLimit)
      )
      hlsProducer = producer

      hlsProductionTask = Task { @MainActor [weak self] in
        guard let self else { return }
        do {
          if let previousControllerRetirement {
            await previousControllerRetirement.value
          }
          if let previousControllerHLSRetirement {
            await previousControllerHLSRetirement.value
          }
          if let localWorkerRetirement {
            await localWorkerRetirement.value
          }
          if let localHLSRetirement {
            await localHLSRetirement.value
          }
          try Task.checkCancellation()
          guard self.generation == startingGeneration else { return }

          try await proxy.start()
          // Match MiohRemote: the audible source clock and restoration
          // producer must consume the same selected media playlist. Opening
          // the parent master here can choose another rendition with a
          // different timestamp origin and makes A/V drift impossible to
          // correct reliably.
          let localPlaybackURL: URL
          if let masterMetadata = playlist.masterMetadata,
            masterMetadata.hasSeparateAudio
          {
            localPlaybackURL = try proxy.localURL(
              forSelectedHLSMaster: masterMetadata,
              context: source.requestContext,
              resolutionPolicy: source.resolutionPolicy
            )
          } else {
            localPlaybackURL = try proxy.localURL(
              for: playlist.url,
              context: source.requestContext,
              isPlaylist: true,
              resolutionPolicy: source.resolutionPolicy
            )
          }
          guard self.generation == startingGeneration else { return }

          let item = AVPlayerItem(asset: AVURLAsset(url: localPlaybackURL))
          item.preferredMaximumResolution = CGSize(width: 1_920, height: 1_080)
          item.preferredForwardBufferDuration = playlist.isLive
            ? max(2, runner.previewBufferLimit)
            : min(6, max(2, runner.previewBufferLimit))
          var sourceItemInstalled = false

          // A paused AVPlayerItem still opens its playlist and commonly probes
          // the first media range while becoming ready. For VOD, let the
          // producer deliver one restored output first so the source clock and
          // restoration downloader cannot race the same signed startup URL.
          // A live playlist must be attached eagerly or its sliding seekable
          // window can move past the restoration start while the worker warms.
          if playlist.isLive {
            sourceItemInstalled = self.installPreparedHLSSourceItem(
              item,
              generation: startingGeneration
            )
            if sourceItemInstalled {
              self.runner?.appendExternalLog(
                "HLS再生: 認証情報を保持したローカルプレイリストを準備しました\n"
              )
            }
          }

          try await producer.run { [weak self] event in
            guard let self, self.generation == startingGeneration,
              !Task.isCancelled
            else { return }
            if !sourceItemInstalled,
              case .segment = event
            {
              sourceItemInstalled = self.installPreparedHLSSourceItem(
                item,
                generation: startingGeneration
              )
              if sourceItemInstalled {
                self.runner?.appendExternalLog(
                  "HLS再生: 復元先頭区間の準備後、認証情報を保持した"
                    + "ローカルプレイリストを開きました\n"
                )
              }
            }
            self.handleHLSProductionEvent(event, generation: startingGeneration)
          }
        } catch is CancellationError {
          return
        } catch {
          guard self.generation == startingGeneration else { return }
          if let fallback = producer.takePendingVariantFallbackSource() {
            producer.cancel()
            let fallbackPosition = self.position.isFinite
              ? max(0, self.position)
              : target
            let fallbackAutoPlay = self.shouldPlay
            self.hlsProductionTask = nil
            self.scheduleHLSVariantFallbackRestart(
              from: source,
              to: fallback,
              runner: runner,
              position: fallbackPosition,
              autoPlay: fallbackAutoPlay,
              generation: startingGeneration,
              reason: error.localizedDescription
            )
            return
          }
          producer.cancel()
          proxy.stop()
          self.fail("HLSリアルタイム復元に失敗しました: \(error.localizedDescription)")
        }
        guard self.generation == startingGeneration else { return }
        self.hlsProductionTask = nil
      }
    } catch {
      fail(error.localizedDescription)
      cleanupSession()
    }
  }

  @discardableResult
  private func installPreparedHLSSourceItem(
    _ item: AVPlayerItem,
    generation expectedGeneration: Int
  ) -> Bool {
    guard generation == expectedGeneration, hlsSource != nil,
      sourcePlayer.currentItem !== item
    else { return false }
    sourcePlayer.automaticallyWaitsToMinimizeStalling = true
    sourcePlayer.replaceCurrentItem(with: item)
    sourcePlayer.volume = muted ? 0 : Float(volume)
    installTimeObserver()
    installHLSPlaybackObservers(
      item: item,
      generation: expectedGeneration
    )
    return true
  }

  private func scheduleHLSVariantFallbackRestart(
    from currentSource: IPadResolvedMediaSource,
    to fallbackSource: IPadResolvedMediaSource,
    runner: RestorationRunner,
    position: Double,
    autoPlay: Bool,
    generation expectedGeneration: Int,
    reason: String
  ) {
    let currentHost = Self.hlsSourceHost(currentSource)
    let fallbackHost = Self.hlsSourceHost(fallbackSource)
    let currentQuality = Self.hlsSourceQuality(currentSource)
    let fallbackQuality = Self.hlsSourceQuality(fallbackSource)
    runner.appendExternalLog(
      "HLS再生: \(reason)\n"
        + "HLS再生: 配信元 \(currentHost) → \(fallbackHost)、"
        + "品質 \(currentQuality) → \(fallbackQuality) へ切り替え、"
        + "同じ位置から再開します\n"
    )
    Task { @MainActor [weak self] in
      // This task is created from the retiring producer's catch path. Yield so
      // its owner can clear hlsProductionTask before startHLS retires the old
      // proxy/session and installs the fallback generation.
      await Task.yield()
      guard let self, self.generation == expectedGeneration else { return }
      self.startHLS(
        source: fallbackSource,
        runner: runner,
        at: position,
        autoPlay: autoPlay
      )
    }
  }

  private static func hlsSourceHost(
    _ source: IPadResolvedMediaSource
  ) -> String {
    source.hlsPlaylist?.masterMetadata?.selectedVideoPlaylistURL.host
      ?? source.hlsPlaylist?.url.host
      ?? source.mediaURL.host
      ?? "不明"
  }

  private static func hlsSourceQuality(
    _ source: IPadResolvedMediaSource
  ) -> String {
    guard let metadata = source.hlsPlaylist?.masterMetadata else {
      return "不明"
    }
    var components: [String] = []
    if let width = metadata.width, let height = metadata.height,
      width > 0, height > 0
    {
      components.append("\(width)x\(height)")
    }
    if metadata.bandwidth > 0 {
      components.append(
        String(format: "%.1f Mbps", Double(metadata.bandwidth) / 1_000_000)
      )
    }
    return components.isEmpty ? "不明" : components.joined(separator: " / ")
  }

  private func handleHLSProductionEvent(
    _ event: MacHLSProductionEvent,
    generation expectedGeneration: Int
  ) {
    guard generation == expectedGeneration else { return }
    switch event {
    case .ready(let mediaDuration, let isLive):
      duration = max(duration, mediaDuration)
      if !hlsRestoredClockFallbackActive {
        playbackDetail = isLive
          ? "ライブ端から復元バッファを準備中"
          : "連続HLS区間から復元バッファを準備中"
      }
      if shouldPlay { state = .buffering }
    case .discontinuity(let newPosition):
      // A live media playlist may slide past the sequence the producer was
      // waiting for. Old restored items and their synthetic clock can no
      // longer be compared with AVPlayer's refreshed presentation timeline.
      let sourcePlaybackUnavailable = hlsRestoredClockFallbackActive
      let refreshedSourceItem = sourcePlaybackUnavailable
        ? nil
        : sourcePlayer.currentItem
      let sourceRemainsReady = refreshedSourceItem?.status == .readyToPlay
      restoredPlayer.pause()
      if let lastOutputSequence = queuedSegments.map(\.sequence).max() {
        hlsProducer?.acknowledgeOutputConsumed(through: lastOutputSequence)
      }
      clearRestoredQueue(deleteFiles: true)
      streamingSegmentCodecs.removeAll(keepingCapacity: true)
      if let streamingSource {
        streamingEventConsumer?(.reset(streamingSource))
      }
      requestedStartSeconds = max(0, newPosition)
      position = requestedStartSeconds
      duration = max(duration, requestedStartSeconds)
      generationHasStarted = false
      generationStartPending = false
      hlsInitialSeekCompleted = false
      hlsSourceIsReady = sourceRemainsReady
      hlsSourceReady = sourceRemainsReady
      hlsSourceSeekCompleted = false
      hlsSourceTimeOffset = 0
      invalidateHLSInitialSeek()
      hlsSeekToLiveWindowStart = true
      hlsSourceReachedEnd = false
      hlsRestoredClockFallbackActive = sourcePlaybackUnavailable
      currentRestoredItemIdentifier = nil
      currentRestoredItemStartedAt = 0
      hlsRestoredHeldForSourceCatchup = false
      hlsDriftCorrectionInFlight = false
      hlsInitialSeekAttempt = 0
      cancelSynchronizedHLSStart()
      if shouldPlay {
        state = .buffering
        playbackDetail = "ライブHLSの更新位置へ追従中"
      }
      if let item = refreshedSourceItem {
        seekHLSClockWhenReady(item: item, generation: expectedGeneration)
      }
    case .segment(
      let sequence,
      let startSeconds,
      let endSeconds,
      let url,
      let codec
    ):
      let segment = PreviewSegment(
        sequence: sequence,
        startSeconds: startSeconds,
        endSeconds: endSeconds,
        url: url
      )
      enqueue(segment)
      streamingSegmentCodecs[sequence] = codec
      streamingEventConsumer?(
        .segment(
          RealtimeStreamingSegment(
            generation: expectedGeneration,
            sequence: sequence,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            url: url,
            codec: codec
          )
        )
      )
      if !hlsRestoredClockFallbackActive {
        playbackDetail = ""
      }
      resumeIfBuffered()
    case .progress(let processingPosition, let mediaDuration):
      duration = max(duration, mediaDuration)
      if !hlsRestoredClockFallbackActive,
        (state == .loading || state == .buffering)
      {
        playbackDetail = String(
          format: "HLS復元中 %.1f / %.1f秒",
          processingPosition,
          mediaDuration
        )
      }
    case .ended(let finalDuration):
      duration = max(duration, finalDuration)
      generationReachedEOF = true
      streamingEventConsumer?(.ended(generation: expectedGeneration))
      if queuedSegments.isEmpty {
        shouldPlay = false
        state = .ended
      } else {
        if !hlsRestoredClockFallbackActive {
          playbackDetail = ""
        }
        resumeIfBuffered(endOfFile: true)
      }
    }
  }

  private func startSourceOnlyPlayback(
    input: URL,
    resources: URL,
    tempRoot: URL,
    generation startingGeneration: Int,
    startSeconds: Double
  ) {
    Task { @MainActor [self] in
      let prepared: PreparedSourcePlayerItem
      do {
        prepared = try await self.prepareSourcePlayerItem(
          input: input,
          resources: resources,
          tempRoot: tempRoot
        )
      } catch {
        guard self.generation == startingGeneration, self.sourceOnlyPlayback else { return }
        self.fail("VR動画を開けません: \(error.localizedDescription)")
        self.cleanupSession()
        self.cleanupSourceCompatibility()
        return
      }
      guard self.generation == startingGeneration, self.sourceOnlyPlayback else { return }
      let item = prepared.item
      item.preferredForwardBufferDuration = max(1, self.runner?.previewBufferLimit ?? 8)
      self.sourceResourceLoader = prepared.resourceLoader
      self.sourceProcessingInputURL = prepared.processingInputURL
      self.sourcePlayer.replaceCurrentItem(with: item)
      self.sourcePlayer.volume = self.muted ? 0 : Float(self.volume)
      self.bufferedSeconds = 0
      self.sourceBufferedSeconds = 0
      self.duration = prepared.duration
      self.installTimeObserver()
      self.installSourcePlaybackObservers(item: item, generation: startingGeneration)
      self.runner?.appendExternalLog("VR再生: 復元モデルを読み込まず、元動画を直接再生します\n")
      if prepared.usesVirtualContainer {
        self.runner?.appendExternalLog(
          "VR再生: 全編remuxを行わず、AVFoundation互換の仮想コンテナを使用します\n"
        )
      } else if prepared.compatibilityMode == .remuxed {
        self.runner?.appendExternalLog("VR再生: MKV/非互換コンテナをMP4へremuxしました\n")
      } else if prepared.compatibilityMode == .transcoded {
        self.runner?.appendExternalLog("VR再生: 非対応コーデックをVideoToolboxで互換変換しました\n")
      }
      self.state = .buffering
      if startSeconds > 0 {
        self.sourcePlayer.seek(
          to: CMTime(seconds: startSeconds, preferredTimescale: 600),
          toleranceBefore: .zero,
          toleranceAfter: .zero
        ) { [weak self] _ in
          Task { @MainActor in
            guard let self, self.generation == startingGeneration, self.sourceOnlyPlayback else { return }
            self.sourcePlayer.play()
          }
        }
      } else {
        self.sourcePlayer.play()
      }
    }
  }

  private func shouldRestartPreviewForCurrentSettings() -> Bool {
    guard let runner, let activePreviewSettingsSignature else {
      return false
    }
    return previewSettingsSignature(for: runner) != activePreviewSettingsSignature
  }

  private func previewSettingsSignature(for runner: RestorationRunner) -> String {
    [
      "restoration=\(runner.previewRestorationModel)",
      "customRestoration=\(runner.previewCustomRestorationModel)",
      "detection=\(runner.previewDetectionModel)",
      "customDetection=\(runner.previewCustomDetectionModel)",
      "realtimeOptimization=\(runner.previewRealtimeOptimization)",
      "preserveComposite=\(runner.preservesRealtimeCompositeParameters)",
      "useMaxClip=\(runner.useMaxClipLength)",
      "maxClip=\(runner.maxClipLength)",
      "useRestoreMax=\(runner.useRestoreMaxFrames)",
      "restoreMax=\(runner.restoreMaxFrames)",
      "overlap=\(runner.restoreTemporalOverlap)",
      "crossfade=\(runner.restoreCrossfade)",
      "sharpen=\(runner.previewRealtimeOptimization ? 0 : runner.sharpenStrength)",
      "detail=\(runner.previewRealtimeOptimization ? 0 : runner.detailBoost)",
      "feather=\(runner.blendFeather)",
      "texture=\(runner.previewRealtimeOptimization ? 0 : runner.textureMix)",
      "smooth=\(runner.previewRealtimeOptimization ? 0 : runner.smoothStrength)",
      "upscale=\(runner.previewRealtimeOptimization ? 1 : runner.effectUpscale)",
      "roiEnhancer=\(runner.previewRealtimeOptimization ? "none" : runner.roiEnhancer)",
      "roiModel=\(runner.previewRealtimeOptimization ? "" : runner.roiEnhancerModel)",
      "roiScale=\(runner.previewRealtimeOptimization ? 1 : runner.roiEnhancerScale)",
      "roiStrength=\(runner.previewRealtimeOptimization ? 0 : runner.roiEnhancerStrength)",
    ].joined(separator: "|")
  }

  func togglePlayback() {
    if state == .playing {
      shouldPlay = false
      cancelSynchronizedHLSStart()
      sourcePlayer.pause()
      restoredPlayer.pause()
      state = .paused
    } else if state == .paused || state == .buffering {
      shouldPlay = true
      if shouldRestartPreviewForCurrentSettings(),
        let runner
      {
        shouldPlay = true
        restartWithCurrentSettings(runner: runner)
        return
      }
      if sourceOnlyPlayback {
        if sourceSeekNeedsBuffer {
          resumeSourceAfterSeekIfBuffered()
        } else {
          sourcePlayer.play()
          state = .playing
        }
      } else {
        resumeIfBuffered()
      }
    } else if state == .idle || state == .ended || state == .failed, let runner {
      if let hlsSource {
        startHLS(
          source: hlsSource,
          runner: runner,
          at: state == .ended ? 0 : position
        )
      } else {
        start(runner: runner, at: position)
      }
    }
  }

  func startSelectedInput(runner: RestorationRunner) {
    if let hlsSource {
      startHLS(
        source: hlsSource,
        runner: runner,
        at: state == .ended ? 0 : position
      )
    } else {
      start(runner: runner, at: state == .ended ? 0 : position)
    }
  }

  /// Idempotent playback controls used by the LAN remote. Keeping these
  /// decisions in the controller avoids exposing AVPlayer or worker details
  /// to the HTTP layer and prevents a repeated play request from pausing an
  /// already-playing video.
  @discardableResult
  func remotePlay(runner: RestorationRunner) -> Bool {
    switch state {
    case .playing:
      return true
    case .paused, .buffering:
      togglePlayback()
      return true
    case .idle, .ended, .failed:
      guard previewInputURL != nil else { return false }
      if let hlsSource {
        startHLS(
          source: hlsSource,
          runner: runner,
          at: state == .ended ? 0 : position
        )
      } else {
        start(runner: runner, at: position)
      }
      return true
    case .loading, .seeking:
      shouldPlay = true
      return true
    }
  }

  @discardableResult
  func remotePause() -> Bool {
    switch state {
    case .paused:
      return true
    case .playing:
      togglePlayback()
      return true
    case .buffering:
      shouldPlay = false
      cancelSynchronizedHLSStart()
      sourcePlayer.pause()
      restoredPlayer.pause()
      state = .paused
      return true
    case .loading, .seeking:
      shouldPlay = false
      return true
    case .idle, .ended, .failed:
      return false
    }
  }

  @discardableResult
  func remoteToggle(runner: RestorationRunner) -> Bool {
    switch state {
    case .playing:
      return remotePause()
    case .paused, .buffering, .idle, .ended, .failed:
      return remotePlay(runner: runner)
    case .loading, .seeking:
      shouldPlay.toggle()
      return true
    }
  }

  func choosePreviewInput(runner: RestorationRunner) {
    let panel = NSOpenPanel()
    panel.title = "再生動画を選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    selectPreviewInput(url, runner: runner)
  }

  /// Selects a source through the same reset/VR-detection transaction used by
  /// the native file picker. The Web remote calls this rather than assigning
  /// `previewInputURL` directly, which would leave stale player/model state.
  func selectPreviewInput(_ url: URL, runner: RestorationRunner) {
    stop(preserveHLSSelection: false)
    previewInputURL = url
    position = 0
    duration = 0
    errorMessage = ""
    playbackDetail = ""
    sourcePlayer.replaceCurrentItem(with: nil)
    isVRVideo = false
    isDetectingVR = true
    vrDetectionDetail = "VR形式を確認中"
    runner.previewProjectionMode = PreviewProjectionMode.normal.rawValue
    Task { [weak self] in
      let detection = await PreviewVRDetector.detect(url: url)
      guard let self, self.previewInputURL == url else { return }
      self.isVRVideo = detection.isVR
      self.isDetectingVR = false
      self.vrDetectionDetail = detection.reason
      self.runner = runner
      runner.previewProjectionMode = detection.projection.rawValue
      runner.previewVideoLayout = detection.layout.rawValue
      if detection.isVR {
        runner.previewEye = PreviewEye.left.rawValue
      }
    }
  }

  func seek(to seconds: Double) {
    if let hlsSource {
      guard hlsSource.hlsPlaylist?.isLive != true else { return }
      position = min(max(seconds, 0), max(duration, 0.01))
      guard let runner else {
        fail("HLS復元を再開できませんでした")
        return
      }
      let resumeAfterSeek = state != .paused
      startHLS(
        source: hlsSource,
        runner: runner,
        at: position,
        autoPlay: resumeAfterSeek
      )
      return
    }
    if sourceOnlyPlayback {
      position = min(max(seconds, 0), max(duration, 0.01))
      let startingGeneration = generation
      let resumeAfterSeek = state != .paused
      shouldPlay = resumeAfterSeek
      sourceSeekNeedsBuffer = resumeAfterSeek
      sourcePlayer.pause()
      state = .seeking
      sourcePlayer.seek(
        to: CMTime(seconds: position, preferredTimescale: 600),
        toleranceBefore: .zero,
        toleranceAfter: .zero
      ) { [weak self] _ in
        Task { @MainActor in
          guard let self, self.generation == startingGeneration, self.sourceOnlyPlayback else { return }
          if resumeAfterSeek {
            self.state = .buffering
            self.playbackDetail = "シーク後のバッファを準備中"
            self.sourcePlayer.preroll(atRate: 1.0) { [weak self] _ in
              Task { @MainActor in
                self?.resumeSourceAfterSeekIfBuffered()
              }
            }
          } else {
            self.sourceSeekNeedsBuffer = false
            self.state = .paused
          }
        }
      }
      return
    }
    let resumeAfterSeek = state != .paused
    position = min(max(seconds, 0), max(duration, 0.01))
    guard let runner else {
      fail("プレビューワーカーを再開できませんでした")
      return
    }
    // A seek is a generation boundary. Start a fresh worker process instead
    // of trying to mutate a running restoration pipeline through stdin. This
    // makes source position, model state, segment files and event generation
    // a single transaction and prevents stale generations from filling the
    // buffer while the player waits for the new one.
    start(
      runner: runner,
      at: position,
      autoPlay: resumeAfterSeek,
      preserveCurrentSource: true
    )
  }

  func restartWithCurrentSettings(runner: RestorationRunner) {
    if let hlsSource {
      startHLS(source: hlsSource, runner: runner, at: position)
    } else {
      start(runner: runner, at: position)
    }
  }

  func setVolume(_ value: Double) {
    volume = min(max(value, 0), 1)
    sourcePlayer.volume = muted || hlsRestoredClockFallbackActive
      ? 0
      : Float(volume)
  }

  func setBufferLimit(_ seconds: Double) {
    if sourceOnlyPlayback {
      sourcePlayer.currentItem?.preferredForwardBufferDuration = max(1, seconds)
      return
    }
    if hlsSource != nil {
      let sourceBufferSeconds = isLiveHLSInput
        ? max(2, seconds)
        : min(6, max(2, seconds))
      sourcePlayer.currentItem?.preferredForwardBufferDuration = sourceBufferSeconds
      hlsProducer?.updateOutputBufferLimits(hlsOutputBufferLimits(for: seconds))
      return
    }
    guard worker != nil else { return }
    sendCommand(["command": "set_buffer_limit", "seconds": seconds])
  }

  private func hlsOutputBufferLimits(
    for requestedSeconds: Double
  ) -> MacHLSRealtimeProducer.OutputBufferLimits {
    // Never configure less credit than the queue needs for its initial start;
    // otherwise a very small UI buffer limit can stop the producer before the
    // third startup segment is emitted and neither side can make progress.
    // Encoded 29.97/59.94 fps files are commonly a few milliseconds longer
    // than the nominal two-second worker interval. One additional nominal
    // segment of credit guarantees that all three startup items can be
    // admitted before playback begins and starts returning credits.
    let minimumStartupSeconds =
      Double(hlsVODStartupSegmentCount + 1) * previewSegmentSeconds
    let seconds = max(
      minimumStartupSeconds,
      requestedSeconds.isFinite ? requestedSeconds : 60
    )
    // Seconds is the user-facing capacity. Keep a separate generous item cap
    // so short tail/discontinuity outputs cannot exhaust item credit before
    // their accumulated duration reaches the startup threshold.
    let defaultLimits = MacHLSRealtimeProducer.OutputBufferLimits.playbackDefault
    let items = max(
      defaultLimits.items,
      Int(ceil(seconds / max(0.5, previewSegmentSeconds)))
    )
    return MacHLSRealtimeProducer.OutputBufferLimits(
      seconds: seconds,
      items: items,
      bytes: defaultLimits.bytes
    )
  }

  func setMuted(_ value: Bool) {
    muted = value
    sourcePlayer.volume = value || hlsRestoredClockFallbackActive
      ? 0
      : Float(volume)
  }

  func stop(
    preserveSourceItem: Bool = false,
    preserveHLSSelection: Bool = true
  ) {
    let stoppedGeneration = generation
    let retiringHLSProducer = hlsProducer
    let retiringHLSSession = hlsSource == nil ? nil : sessionDirectory
    let precedingHLSRetirement = hlsProducerRetirementTask
    if streamingSource?.generation == stoppedGeneration {
      streamingEventConsumer?(.stopped(generation: stoppedGeneration))
    }
    streamingSource = nil
    streamingSegmentCodecs.removeAll(keepingCapacity: false)
    generation += 1
    hlsProductionTask?.cancel()
    hlsProductionTask = nil
    hlsProducer = nil
    hlsMediaProxy?.stop()
    hlsMediaProxy = nil
    if !preserveHLSSelection {
      hlsSource = nil
    }
    sourceCompatibilityJob?.cancel()
    sourceCompatibilityJob = nil
    sourceItemStatusObservation?.invalidate()
    sourceItemStatusObservation = nil
    sourceTimeControlObservation?.invalidate()
    sourceTimeControlObservation = nil
    sourceLoadedTimeRangesObservation?.invalidate()
    sourceLoadedTimeRangesObservation = nil
    hlsSourceSeekableTimeRangesObservation?.invalidate()
    hlsSourceSeekableTimeRangesObservation = nil
    for token in hlsNotificationTokens {
      NotificationCenter.default.removeObserver(token)
    }
    hlsNotificationTokens.removeAll()
    hlsSourceIsReady = false
    hlsInitialSeekCompleted = false
    hlsSourceReady = false
    hlsSourceSeekCompleted = false
    hlsSourceTimeOffset = 0
    invalidateHLSInitialSeek()
    hlsSeekToLiveWindowStart = false
    hlsSourceReachedEnd = false
    hlsRestoredClockFallbackActive = false
    currentRestoredItemIdentifier = nil
    currentRestoredItemStartedAt = 0
    hlsRestoredHeldForSourceCatchup = false
    hlsDriftCorrectionInFlight = false
    hlsInitialSeekAttempt = 0
    cancelSynchronizedHLSStart()
    sourceOnlyPlayback = false
    shouldPlay = false
    generationHasStarted = false
    generationStartPending = false
    generationReachedEOF = false
    sourceSeekNeedsBuffer = false
    activePreviewSettingsSignature = nil
    sourcePlayer.pause()
    if !preserveSourceItem {
      sourcePlayer.replaceCurrentItem(with: nil)
      sourceResourceLoader = nil
      sourceProcessingInputURL = nil
      cleanupSourceCompatibility()
    }
    restoredPlayer.pause()
    let retiringWorker = worker
    if retiringWorker != nil {
      sendCommand(["command": "stop"])
    }
    try? workerInput?.fileHandleForWriting.close()
    stdoutPipe?.fileHandleForReading.readabilityHandler = nil
    stderrPipe?.fileHandleForReading.readabilityHandler = nil
    worker = nil
    workerInput = nil
    stdoutPipe = nil
    stderrPipe = nil
    clearRestoredQueue(deleteFiles: true)
    if let retiringHLSProducer {
      // The producer owns active downloader/process handles and files below
      // the HLS session. Never remove that tree until run() has unwound. Chain
      // repeated stops so a rapidly replaced generation cannot bypass an
      // older producer that is still retiring.
      retiringHLSProducer.cancel()
      sessionDirectory = nil
      hlsProducerRetirementTask = Task { @MainActor in
        if let precedingHLSRetirement {
          await precedingHLSRetirement.value
        }
        await retiringHLSProducer.cancelAndWait()
        if let retiringHLSSession {
          try? FileManager.default.removeItem(at: retiringHLSSession)
        }
      }
    } else {
      cleanupSession()
    }
    state = .idle
    bufferedSeconds = 0
    sourceBufferedSeconds = 0
    playbackDetail = ""
    if let retiringWorker {
      if retiringWorker.isRunning {
        workerRetirementTask = Task { @MainActor in
          let processIdentifier = retiringWorker.processIdentifier
          // Give the worker a bounded grace period to stop its restoration
          // threads and Core AI children after stdin is closed.
          for _ in 0..<40 where retiringWorker.isRunning {
            try? await Task.sleep(nanoseconds: 50_000_000)
          }
          if retiringWorker.isRunning {
            if kill(-processIdentifier, SIGTERM) != 0 {
              retiringWorker.terminate()
            }
          }
          for _ in 0..<20 where retiringWorker.isRunning {
            try? await Task.sleep(nanoseconds: 50_000_000)
          }
          if retiringWorker.isRunning {
            _ = kill(-processIdentifier, SIGKILL)
          }
          // A new generation must not load Core AI assets until the old process
          // tree is gone. This is the serialization barrier for repeated seeks.
          while retiringWorker.isRunning {
            try? await Task.sleep(nanoseconds: 20_000_000)
          }
        }
      } else {
        workerRetirementTask = nil
      }
    }
    if !preserveSourceItem, Self.activeRestorationController === self {
      Self.activeRestorationController = nil
    }
  }

  private func prepareSourcePlayerItem(
    input: URL,
    resources: URL,
    tempRoot: URL
  ) async throws -> PreparedSourcePlayerItem {
    let source = AVURLAsset(url: input)
    if (try? await source.load(.isPlayable)) == true {
      return PreparedSourcePlayerItem(
        item: AVPlayerItem(asset: source),
        duration: await sourceDuration(source),
        resourceLoader: nil,
        processingInputURL: input,
        compatibilityMode: .direct
      )
    }

    // HEV1 8K files may contain perfectly decodable samples while AVFoundation
    // rejects the MP4 sample-entry identifier.  A loopback byte-range server
    // exposes the original file byte-for-byte except for hev1 -> hvc1 inside the
    // moov atom. This preserves random access without modifying or copying it.
    if ["mp4", "m4v", "mov"].contains(input.pathExtension.lowercased()),
      let resourceLoader = try? HEV1LoopbackServer(sourceURL: input)
    {
      let compatibleAsset = resourceLoader.makeAsset()
      if (try? await compatibleAsset.load(.isPlayable)) == true {
        return PreparedSourcePlayerItem(
          item: AVPlayerItem(asset: compatibleAsset),
          duration: await sourceDuration(compatibleAsset),
          resourceLoader: resourceLoader,
          // Native AVAssetReader can generally open the original HEV1 file;
          // only AVPlayer needs the hvc1 sample-entry view.
          processingInputURL: input,
          compatibilityMode: .virtualHEV1
        )
      }
    }

    let ffmpeg = resources.appendingPathComponent("bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath: ffmpeg.path) else {
      throw SourcePlaybackError.missingFFmpeg
    }
    cleanupSourceCompatibility()
    let compatibilityDirectory = tempRoot.appendingPathComponent(
      "mioh-source-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: compatibilityDirectory,
      withIntermediateDirectories: true
    )
    sourceCompatibilityDirectory = compatibilityDirectory
    let compatibleURL = compatibilityDirectory.appendingPathComponent("source-compatible.mp4")

    playbackDetail = "AVFoundation互換MP4へremux中"
    runner?.appendExternalLog(
      "再生: \(input.lastPathComponent) を映像再エンコードなしで互換MP4へremuxします\n"
    )
    var arguments = [
      "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
      "-i", input.path,
      "-map", "0:v:0", "-map", "0:a:0?",
      "-sn", "-dn",
      "-c:v", "copy",
      "-c:a", "aac", "-b:a", "192k",
      "-movflags", "+faststart",
      "-avoid_negative_ts", "make_zero",
      compatibleURL.path,
    ]
    var status = try await runCompatibilityJob(
      ffmpeg: ffmpeg,
      arguments: arguments
    )
    if status == 0 {
      let remuxedAsset = AVURLAsset(url: compatibleURL)
      if (try? await remuxedAsset.load(.isPlayable)) == true {
        playbackDetail = ""
        return PreparedSourcePlayerItem(
          item: AVPlayerItem(asset: remuxedAsset),
          duration: await sourceDuration(remuxedAsset),
          resourceLoader: nil,
          processingInputURL: compatibleURL,
          compatibilityMode: .remuxed
        )
      }
    }

    // Some HEVC streams keep a hev1 sample entry after a plain MKV/MP4 remux.
    // Re-run stream-copy with hvc1 before paying the cost of re-encoding.
    try? FileManager.default.removeItem(at: compatibleURL)
    arguments.insert(contentsOf: ["-tag:v", "hvc1"], at: arguments.count - 1)
    status = try await runCompatibilityJob(ffmpeg: ffmpeg, arguments: arguments)
    if status == 0 {
      let taggedAsset = AVURLAsset(url: compatibleURL)
      if (try? await taggedAsset.load(.isPlayable)) == true {
        playbackDetail = ""
        return PreparedSourcePlayerItem(
          item: AVPlayerItem(asset: taggedAsset),
          duration: await sourceDuration(taggedAsset),
          resourceLoader: nil,
          processingInputURL: compatibleURL,
          compatibilityMode: .remuxed
        )
      }
    }

    // A genuinely unsupported codec cannot be fixed at the container layer.
    // This is the last-resort path: VideoToolbox performs the conversion once,
    // and every downstream consumer shares the resulting local MP4.
    try? FileManager.default.removeItem(at: compatibleURL)
    playbackDetail = "非対応コーデックをVideoToolboxで互換変換中"
    runner?.appendExternalLog(
      "再生: remuxだけでは開けないため、映像をVideoToolbox HEVCへ互換変換します\n"
    )
    arguments = [
      "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
      "-i", input.path,
      "-map", "0:v:0", "-map", "0:a:0?",
      "-sn", "-dn",
      "-c:v", "hevc_videotoolbox", "-q:v", "65", "-tag:v", "hvc1",
      "-c:a", "aac", "-b:a", "192k",
      "-movflags", "+faststart",
      "-avoid_negative_ts", "make_zero",
      compatibleURL.path,
    ]
    status = try await runCompatibilityJob(ffmpeg: ffmpeg, arguments: arguments)
    guard status == 0 else {
      throw SourcePlaybackError.compatibilityConversionFailed
    }
    let transcodedAsset = AVURLAsset(url: compatibleURL)
    guard (try? await transcodedAsset.load(.isPlayable)) == true else {
      throw SourcePlaybackError.compatibilityConversionFailed
    }
    playbackDetail = ""
    return PreparedSourcePlayerItem(
      item: AVPlayerItem(asset: transcodedAsset),
      duration: await sourceDuration(transcodedAsset),
      resourceLoader: nil,
      processingInputURL: compatibleURL,
      compatibilityMode: .transcoded
    )
  }

  private func sourceDuration(_ asset: AVURLAsset) async -> Double {
    guard let durationTime = try? await asset.load(.duration) else { return 0 }
    let seconds = CMTimeGetSeconds(durationTime)
    return seconds.isFinite ? max(0, seconds) : 0
  }

  private func runCompatibilityJob(ffmpeg: URL, arguments: [String]) async throws -> Int32 {
    let job = SourceCompatibilityJob(
      executable: ffmpeg,
      arguments: arguments
    ) { [weak self] text in
      Task { @MainActor in
        self?.runner?.appendExternalLog(text)
      }
    }
    sourceCompatibilityJob = job
    defer {
      if sourceCompatibilityJob === job {
        sourceCompatibilityJob = nil
      }
    }
    return try await job.run()
  }

  private func consumeWorkerOutput(_ text: String) {
    stdoutBuffer += text
    while let newline = stdoutBuffer.firstIndex(of: "\n") {
      let line = String(stdoutBuffer[..<newline])
      stdoutBuffer.removeSubrange(...newline)
      guard let data = line.data(using: .utf8) else { continue }
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      do {
        let event = try decoder.decode(PreviewWorkerEvent.self, from: data)
        handle(event)
      } catch {
        runner?.appendExternalLog("Invalid preview event: \(line)\n")
      }
    }
  }

  private func handle(_ event: PreviewWorkerEvent) {
    guard event.generation == generation else { return }
    switch event.kind {
    case "ready":
      duration = event.duration ?? 0
      previewSegmentSeconds = max(0.1, event.segmentSeconds ?? 2.0)
      if let existing = streamingSource, existing.generation == generation,
        abs(existing.segmentSeconds - previewSegmentSeconds) > 0.000_001
      {
        let updated = RealtimeStreamingSource(
          generation: existing.generation,
          inputURL: existing.inputURL,
          ffmpegURL: existing.ffmpegURL,
          segmentSeconds: previewSegmentSeconds
        )
        streamingSource = updated
        streamingEventConsumer?(.reset(updated))
      }
      state = shouldPlay ? .buffering : .paused
    case "segment":
      guard let sequence = event.sequence,
        let startNs = event.startNs,
        let endNs = event.endNs,
        let path = event.path
      else { return }
      let segment = PreviewSegment(
        sequence: sequence,
        startSeconds: Double(startNs) / 1_000_000_000,
        endSeconds: Double(endNs) / 1_000_000_000,
        url: URL(fileURLWithPath: path)
      )
      enqueue(segment)
      let codec = event.codec ?? "h264_videotoolbox"
      streamingSegmentCodecs[sequence] = codec
      streamingEventConsumer?(
        .segment(
          RealtimeStreamingSegment(
            generation: generation,
            sequence: sequence,
            startSeconds: segment.startSeconds,
            endSeconds: segment.endSeconds,
            url: segment.url,
            codec: codec
          )
        )
      )
      resumeIfBuffered()
    case "ended":
      generationReachedEOF = true
      streamingEventConsumer?(.ended(generation: generation))
      if queuedSegments.isEmpty {
        shouldPlay = false
        sourcePlayer.pause()
        restoredPlayer.pause()
        state = .ended
      } else {
        resumeIfBuffered(endOfFile: true)
      }
    case "error":
      fail([event.message, event.detail].compactMap { $0 }.joined(separator: ": "))
    case "buffer_limit":
      guard let seconds = event.seconds else { return }
      runner?.appendExternalLog("プレビューバッファ上限を適用: \(Int(seconds))秒\n")
    case "buffer_full":
      // The worker limits storage by finalized segment count, while the UI
      // measures the actual timestamp range. Timestamp discontinuities can
      // therefore report 24.8 s for a physically full 27 s buffer. Once the
      // worker confirms the current limit is full, waiting for an impossible
      // extra segment would deadlock playback.
      let requestedLimit = max(1, runner?.previewBufferLimit ?? 8)
      guard event.seconds == nil || abs((event.seconds ?? requestedLimit) - requestedLimit) < 0.1
      else { return }
      resumeIfBuffered(bufferIsFull: true)
    default:
      break
    }
  }

  private func enqueue(_ segment: PreviewSegment) {
    guard segment.sequence == nextSequence else { return }
    nextSequence += 1
    let item = AVPlayerItem(url: segment.url)
    let identifier = ObjectIdentifier(item)
    itemSegments[identifier] = segment
    queuedSegments.append(segment)
    restoredPlayer.insert(item, after: nil)
    let token = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemDidPlayToEndTime,
      object: item,
      queue: .main
    ) { [weak self] _ in
      Task { @MainActor in self?.finished(item: item) }
    }
    itemEndNotificationTokens[identifier] = token
    updateBufferedDuration()
  }

  private func finished(item: AVPlayerItem) {
    guard let segment = itemSegments[ObjectIdentifier(item)] else { return }
    releaseConsumedSegments(through: segment.sequence)
    if queuedSegments.isEmpty && state == .playing {
      if hlsSource != nil,
        !hlsRestoredClockFallbackActive,
        !hlsSourceReachedEnd,
        hlsSourceIsAtExpectedEnd()
      {
        // AVPlayerItemDidPlayToEndTime is not guaranteed when we pause the
        // source in the same run-loop turn that the restored queue empties.
        // Preserve the terminal clock state so a late final restored item does
        // not try to restart an AVPlayer already parked at EOF.
        hlsSourceReachedEnd = true
        showOriginal = false
        hlsRestoredHeldForSourceCatchup = false
        hlsDriftCorrectionInFlight = false
        cancelSynchronizedHLSStart()
      }
      sourcePlayer.pause()
      restoredPlayer.pause()
      if generationReachedEOF {
        shouldPlay = false
        state = .ended
      } else {
        state = .buffering
      }
    }
  }

  private func resumeIfBuffered(
    endOfFile: Bool = false,
    bufferIsFull: Bool = false
  ) {
    guard shouldPlay else { return }
    guard state != .playing, !generationStartPending else { return }
    if hlsSource != nil {
      guard hlsSourceClockIsReadyForSynchronizedPlayback
        || canStartHLSWithRestoredClockFallback
      else {
        if let item = sourcePlayer.currentItem {
          seekHLSClockWhenReady(item: item, generation: generation)
        }
        state = .buffering
        if playbackDetail.isEmpty {
          playbackDetail = "HLS元動画の再生位置を準備中"
        }
        return
      }
    }
    if state == .paused, generationHasStarted,
      restoredPlayer.currentItem != nil
    {
      startPlayersFromCurrentPosition()
      return
    }
    let nominalRequired =
      Double(requiredSegmentCountForCurrentGeneration())
      * previewSegmentSeconds
    let required = min(
      nominalRequired,
      max(0, duration - requestedStartSeconds)
    )
    // Segment duration can vary slightly with source timestamps. Decide from
    // the actual completed time range, not from a nominal segment count.
    guard bufferedSeconds + 0.1 >= required
      || (bufferIsFull && !queuedSegments.isEmpty)
      || (endOfFile && !queuedSegments.isEmpty)
    else {
      if state != .loading && state != .seeking { state = .buffering }
      return
    }
    if generationHasStarted {
      startPlayersFromCurrentPosition()
      return
    }

    if hlsSource != nil {
      generationHasStarted = true
      startPlayersFromCurrentPosition()
      return
    }

    let startingGeneration = generation
    generationStartPending = true
    sourcePlayer.seek(
      to: CMTime(seconds: requestedStartSeconds, preferredTimescale: 600),
      toleranceBefore: .zero,
      toleranceAfter: .zero
    ) { [weak self] _ in
      Task { @MainActor in
        guard let self else { return }
        guard self.generation == startingGeneration else { return }
        self.generationStartPending = false
        guard self.shouldPlay else { return }
        self.generationHasStarted = true
        self.startPlayersFromCurrentPosition()
      }
    }
  }

  private func requiredSegmentCountForCurrentGeneration() -> Int {
    if hlsSource != nil, !isLiveHLSInput {
      return generationHasStarted
        ? hlsVODRebufferSegmentCount
        : hlsVODStartupSegmentCount
    }
    return generationHasStarted ? rebufferSegmentCount : startupSegmentCount
  }

  private func startPlayersFromCurrentPosition() {
    if hlsSource != nil {
      if shouldPreferRestoredHLSPlayback {
        restoredPlayer.play()
        state = .playing
        if hlsRestoredClockFallbackActive, playbackDetail.isEmpty {
          playbackDetail = "元動画側の再生を継続できないため、復元映像のみ再生中（音声なし）"
        }
        return
      }
      guard hlsSourceClockIsReadyForSynchronizedPlayback
      else {
        state = .buffering
        playbackDetail = "HLS元動画の再生位置を準備中"
        return
      }
      hlsRestoredClockFallbackActive = false
      beginSynchronizedHLSStart()
      return
    }
    sourcePlayer.play()
    restoredPlayer.play()
    state = .playing
  }

  private var hlsSourceClockIsReadyForSynchronizedPlayback: Bool {
    hlsSourceReady && hlsSourceSeekCompleted
  }

  private var canStartHLSWithRestoredClockFallback: Bool {
    hlsRestoredClockFallbackActive
      && restoredPlayer.currentItem != nil
      && !queuedSegments.isEmpty
  }

  private var shouldPreferRestoredHLSPlayback: Bool {
    canStartHLSWithRestoredClockFallback
      || (hlsSourceReachedEnd
        && restoredPlayer.currentItem != nil
        && !queuedSegments.isEmpty)
  }

  /// Preroll both independent players before starting them in the same main
  /// actor transaction. `sourcePlayer.play()` followed by a later KVO callback
  /// used to let the audible HLS clock run ahead while Core AI kept the main
  /// thread busy. AVPlayer cannot share a timebase across these two assets, but
  /// adjacent starts after both prerolls keep the initial skew bounded and the
  /// periodic one-way correction below handles the remaining few frames.
  private func beginSynchronizedHLSStart() {
    guard hlsSource != nil,
      hlsSourceClockIsReadyForSynchronizedPlayback,
      !hlsSourceReachedEnd,
      !hlsRestoredClockFallbackActive,
      shouldPlay,
      let sourceItem = sourcePlayer.currentItem,
      let restoredItem = restoredPlayer.currentItem
    else { return }
    guard !hlsSynchronizedStartInFlight else { return }

    hlsSynchronizedStartInFlight = true
    hlsSynchronizedStartRevision &+= 1
    let revision = hlsSynchronizedStartRevision
    hlsSynchronizedStartAttempt += 1
    let expectedGeneration = generation
    sourcePlayer.pause()
    restoredPlayer.pause()
    state = .buffering
    playbackDetail = "HLS音声と復元映像を同期中"
    scheduleSynchronizedHLSStartWatchdog(
      sourceItem: sourceItem,
      restoredItem: restoredItem,
      generation: expectedGeneration,
      revision: revision
    )

    sourcePlayer.preroll(atRate: 1.0) { [weak self, weak sourceItem, weak restoredItem] sourceReady in
      Task { @MainActor in
        guard let self, let sourceItem, let restoredItem,
          self.generation == expectedGeneration,
          self.hlsSynchronizedStartRevision == revision,
          self.sourcePlayer.currentItem === sourceItem,
          self.restoredPlayer.currentItem === restoredItem,
          self.shouldPlay
        else { return }
        guard sourceReady else {
          self.finishSynchronizedHLSStartRetry(
            generation: expectedGeneration,
            revision: revision,
            detail: self.hlsSourceWaitingDescription()
          )
          return
        }
        self.restoredPlayer.preroll(atRate: 1.0) {
          [weak self, weak sourceItem, weak restoredItem] restoredReady in
          Task { @MainActor in
            guard let self, let sourceItem, let restoredItem,
              self.generation == expectedGeneration,
              self.hlsSynchronizedStartRevision == revision,
              self.sourcePlayer.currentItem === sourceItem,
              self.restoredPlayer.currentItem === restoredItem,
              self.shouldPlay
            else { return }
            guard restoredReady else {
              self.finishSynchronizedHLSStartRetry(
                generation: expectedGeneration,
                revision: revision,
                detail: "復元映像を再バッファ中"
              )
              return
            }
            self.hlsSynchronizedStartWatchdogTask?.cancel()
            self.hlsSynchronizedStartWatchdogTask = nil
            self.hlsSynchronizedStartInFlight = false
            self.hlsSynchronizedStartAttempt = 0
            // Keep these calls adjacent. Waiting for source timeControlStatus
            // before starting restored video reintroduced a main-thread-sized
            // audio lead on every initial start and queue refill.
            self.sourcePlayer.play()
            self.restoredPlayer.play()
            self.state = .playing
            self.playbackDetail = ""
          }
        }
      }
    }
  }

  private func finishSynchronizedHLSStartRetry(
    generation expectedGeneration: Int,
    revision: Int,
    detail: String
  ) {
    guard generation == expectedGeneration,
      hlsSynchronizedStartRevision == revision
    else { return }
    hlsSynchronizedStartWatchdogTask?.cancel()
    hlsSynchronizedStartWatchdogTask = nil
    // Invalidate the old completion closures before asking AVFoundation to
    // cancel. A cancellation is allowed to complete a pending preroll, so the
    // callbacks must already fail their revision guard if that happens inline.
    hlsSynchronizedStartRevision &+= 1
    let retryRevision = hlsSynchronizedStartRevision
    hlsSynchronizedStartInFlight = false
    sourcePlayer.cancelPendingPrerolls()
    restoredPlayer.cancelPendingPrerolls()
    state = .buffering
    playbackDetail = detail
    if hlsSynchronizedStartAttempt >= hlsMaximumSynchronizedStartAttempts {
      degradeHLSSourcePlayback(
        generation: expectedGeneration,
        reason: "HLS音声と復元映像の同期準備がタイムアウトしました"
      )
      return
    }
    Task { @MainActor [weak self] in
      do {
        try await Task.sleep(nanoseconds: 80_000_000)
      } catch {
        return
      }
      guard let self,
        self.generation == expectedGeneration,
        self.hlsSynchronizedStartRevision == retryRevision,
        self.shouldPlay
      else { return }
      self.resumeIfBuffered()
    }
  }

  private func scheduleSynchronizedHLSStartWatchdog(
    sourceItem: AVPlayerItem,
    restoredItem: AVPlayerItem,
    generation expectedGeneration: Int,
    revision: Int
  ) {
    hlsSynchronizedStartWatchdogTask?.cancel()
    let timeoutNanoseconds = UInt64(hlsOperationWatchdogSeconds * 1_000_000_000)
    hlsSynchronizedStartWatchdogTask = Task { @MainActor [weak self, weak sourceItem, weak restoredItem] in
      do {
        try await Task.sleep(nanoseconds: timeoutNanoseconds)
      } catch {
        return
      }
      guard let self, let sourceItem, let restoredItem,
        self.generation == expectedGeneration,
        self.hlsSynchronizedStartRevision == revision,
        self.hlsSynchronizedStartInFlight,
        self.sourcePlayer.currentItem === sourceItem,
        self.restoredPlayer.currentItem === restoredItem,
        self.shouldPlay
      else { return }
      self.finishSynchronizedHLSStartRetry(
        generation: expectedGeneration,
        revision: revision,
        detail: "HLS音声と復元映像の同期準備を再試行中"
      )
    }
  }

  private func cancelSynchronizedHLSStart() {
    hlsSynchronizedStartRevision &+= 1
    hlsSynchronizedStartInFlight = false
    hlsSynchronizedStartAttempt = 0
    hlsSynchronizedStartWatchdogTask?.cancel()
    hlsSynchronizedStartWatchdogTask = nil
    sourcePlayer.cancelPendingPrerolls()
    restoredPlayer.cancelPendingPrerolls()
  }

  /// Observes the original HLS player independently from restored segment
  /// items. The source is both the audio track and the authoritative clock,
  /// but restored HLS video is allowed to keep playing through short source
  /// stalls when its own buffer is already ahead. Terminal AVFoundation item
  /// errors degrade to the already-restored queue when one is available.
  /// That preserves video playback even though source audio is unavailable.
  private func installHLSPlaybackObservers(
    item: AVPlayerItem,
    generation: Int
  ) {
    sourceItemStatusObservation?.invalidate()
    sourceTimeControlObservation?.invalidate()
    hlsSourceSeekableTimeRangesObservation?.invalidate()
    for token in hlsNotificationTokens {
      NotificationCenter.default.removeObserver(token)
    }
    hlsNotificationTokens.removeAll(keepingCapacity: true)

    sourceItemStatusObservation = item.observe(
      \.status,
      options: [.initial, .new]
    ) { [weak self, weak item] _, _ in
      Task { @MainActor in
        guard let self, let item else { return }
        self.updateHLSPlaybackState(item: item, generation: generation)
      }
    }
    sourceTimeControlObservation = sourcePlayer.observe(
      \.timeControlStatus,
      options: [.initial, .new]
    ) { [weak self, weak item] _, _ in
      Task { @MainActor in
        guard let self, let item else { return }
        self.updateHLSPlaybackState(item: item, generation: generation)
      }
    }
    hlsSourceSeekableTimeRangesObservation = item.observe(
      \.seekableTimeRanges,
      options: [.initial, .new]
    ) { [weak self, weak item] _, _ in
      Task { @MainActor in
        guard let self, let item else { return }
        self.seekHLSClockWhenReady(item: item, generation: generation)
      }
    }

    let stalled = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemPlaybackStalled,
      object: item,
      queue: .main
    ) { [weak self, weak item] _ in
      Task { @MainActor in
        guard let self, let item,
          self.generation == generation,
          self.hlsSource != nil,
          self.sourcePlayer.currentItem === item,
          self.state != .failed,
          !self.hlsRestoredClockFallbackActive
        else { return }
        self.restoredPlayer.pause()
        if self.shouldPlay {
          self.state = .buffering
          self.playbackDetail = "HLS元動画を再バッファ中"
        }
      }
    }
    hlsNotificationTokens.append(stalled)

    let failedToEnd = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemFailedToPlayToEndTime,
      object: item,
      queue: .main
    ) { [weak self, weak item] notification in
      Task { @MainActor in
        guard let self, let item,
          self.generation == generation,
          self.hlsSource != nil,
          self.sourcePlayer.currentItem === item
        else { return }
        let underlying = notification.userInfo?[AVPlayerItemFailedToPlayToEndTimeErrorKey]
          as? Error
        self.degradeHLSSourcePlayback(
          generation: generation,
          reason: underlying?.localizedDescription
            ?? item.error?.localizedDescription
            ?? "Cannot Open"
        )
      }
    }
    hlsNotificationTokens.append(failedToEnd)

    let ended = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemDidPlayToEndTime,
      object: item,
      queue: .main
    ) { [weak self, weak item] _ in
      Task { @MainActor in
        guard let self, let item,
          self.generation == generation,
          self.hlsSource != nil,
          self.sourcePlayer.currentItem === item
        else { return }
        self.handleHLSSourceDidReachEnd(item: item, generation: generation)
      }
    }
    hlsNotificationTokens.append(ended)
  }

  private func handleHLSSourceDidReachEnd(
    item: AVPlayerItem,
    generation expectedGeneration: Int
  ) {
    guard generation == expectedGeneration,
      hlsSource != nil,
      sourcePlayer.currentItem === item,
      !hlsRestoredClockFallbackActive
    else { return }
    let sourceTimeline = hlsSourceTimelineSeconds() ?? 0
    let playlistDuration = hlsSource?.hlsPlaylist?.duration ?? 0
    let expectedEnd = max(duration, playlistDuration)
    let tolerance = hlsExpectedEndToleranceSeconds()
    if isLiveHLSInput || expectedEnd - sourceTimeline > tolerance {
      degradeHLSSourcePlayback(
        generation: expectedGeneration,
        reason: String(
          format: "HLS元動画が予定より早く終了しました（%.2f / %.2f秒）",
          sourceTimeline,
          expectedEnd
        )
      )
      return
    }

    hlsSourceReachedEnd = true
    showOriginal = false
    hlsRestoredHeldForSourceCatchup = false
    hlsDriftCorrectionInFlight = false
    cancelSynchronizedHLSStart()
    if generationReachedEOF && queuedSegments.isEmpty {
      shouldPlay = false
      restoredPlayer.pause()
      state = .ended
    } else if shouldPlay, restoredPlayer.currentItem != nil {
      restoredPlayer.play()
      state = .playing
    }
  }

  private func hlsSourceTimelineSeconds() -> Double? {
    let sourceSeconds = sourcePlayer.currentTime().seconds
    guard sourceSeconds.isFinite else { return nil }
    return max(0, sourceSeconds - hlsSourceTimeOffset)
  }

  private func hlsExpectedEndToleranceSeconds() -> Double {
    let target = hlsSource?.hlsPlaylist?.targetDuration ?? previewSegmentSeconds
    return max(1, min(2, target))
  }

  private func hlsSourceIsAtExpectedEnd() -> Bool {
    guard !isLiveHLSInput, let sourceTimeline = hlsSourceTimelineSeconds() else {
      return false
    }
    let expectedEnd = max(duration, hlsSource?.hlsPlaylist?.duration ?? 0)
    return expectedEnd - sourceTimeline <= hlsExpectedEndToleranceSeconds()
  }

  private func updateHLSPlaybackState(
    item: AVPlayerItem,
    generation: Int
  ) {
    guard self.generation == generation, hlsSource != nil,
      sourcePlayer.currentItem === item, state != .failed,
      !hlsSourceReachedEnd,
      !hlsRestoredClockFallbackActive
    else { return }

    switch item.status {
    case .failed:
      degradeHLSSourcePlayback(
        generation: generation,
        reason: item.error?.localizedDescription ?? "Cannot Open"
      )
      return
    case .unknown:
      hlsSourceIsReady = false
      hlsSourceReady = false
      if shouldPlay {
        state = .loading
        playbackDetail = "HLS元動画を開いています"
      }
      return
    case .readyToPlay:
      hlsSourceIsReady = true
      hlsSourceReady = true
      seekHLSClockWhenReady(item: item, generation: generation)
    @unknown default:
      hlsSourceIsReady = false
      hlsSourceReady = false
      if shouldPlay {
        state = .loading
        playbackDetail = "HLS元動画を開いています"
      }
      return
    }

    guard hlsSourceSeekCompleted || hlsRestoredClockFallbackActive else {
      if shouldPlay {
        state = .buffering
        playbackDetail = "HLS元動画の再生位置を準備中"
      }
      return
    }
    guard shouldPlay else {
      state = .paused
      playbackDetail = ""
      return
    }
    guard generationHasStarted else {
      if state != .loading { state = .buffering }
      return
    }

    // A deliberate source pause holds the audible clock while a forward-only
    // restored-video correction is in flight. Do not reinterpret that pause as
    // an AVFoundation decoder stall and restart the state machine underneath
    // the seek completion.
    if hlsDriftCorrectionInFlight { return }

    switch sourcePlayer.timeControlStatus {
    case .playing:
      if !hlsSourceReachedEnd {
        if !hlsRestoredHeldForSourceCatchup {
          restoredPlayer.play()
        }
        state = .playing
        playbackDetail = hlsRestoredHeldForSourceCatchup
          ? "HLS音声へ同期中"
          : ""
      }
    case .waitingToPlayAtSpecifiedRate:
      restoredPlayer.pause()
      state = .buffering
      playbackDetail = hlsSourceWaitingDescription()
    case .paused:
      if !hlsSourceReachedEnd {
        restoredPlayer.pause()
        state = .buffering
        playbackDetail = "HLS元動画のデコーダ開始待ち"
      }
    @unknown default:
      restoredPlayer.pause()
      state = .buffering
      playbackDetail = "HLS元動画をバッファ中"
    }
  }

  private func seekHLSClockWhenReady(
    item: AVPlayerItem,
    generation: Int
  ) {
    guard self.generation == generation, hlsSource != nil,
      sourcePlayer.currentItem === item,
      item.status == .readyToPlay,
      hlsSourceReady,
      !hlsSourceSeekCompleted,
      !hlsSeekInFlight
    else { return }

    let syntheticTarget = requestedStartSeconds
    let sourceTarget: Double
    if isLiveHLSInput {
      let seekableRanges = item.seekableTimeRanges.compactMap { value -> ClosedRange<Double>? in
        let range = value.timeRangeValue
        let start = CMTimeGetSeconds(range.start)
        let end = CMTimeGetSeconds(CMTimeRangeGetEnd(range))
        guard start.isFinite, end.isFinite, end >= start else { return nil }
        return start...end
      }
      guard let seekable = seekableRanges.last else {
        if shouldPlay {
          state = .buffering
          playbackDetail = "ライブHLSの再生可能範囲を確認中"
        }
        return
      }
      if hlsSeekToLiveWindowStart {
        sourceTarget = seekable.lowerBound
      } else {
        let distanceFromLiveEdge = max(0, duration - syntheticTarget)
        sourceTarget = min(
          seekable.upperBound,
          max(seekable.lowerBound, seekable.upperBound - distanceFromLiveEdge)
        )
      }
    } else {
      sourceTarget = max(0, syntheticTarget)
    }

    hlsSeekInFlight = true
    hlsInitialSeekAttempt += 1
    let attempt = hlsInitialSeekAttempt
    hlsSeekRevision &+= 1
    let revision = hlsSeekRevision
    sourcePlayer.pause()
    scheduleHLSInitialSeekWatchdog(
      item: item,
      generation: generation,
      revision: revision,
      attempt: attempt
    )
    sourcePlayer.seek(
      to: CMTime(seconds: sourceTarget, preferredTimescale: 600),
      toleranceBefore: .zero,
      toleranceAfter: .zero
    ) { [weak self, weak item] finished in
      Task { @MainActor in
        guard let self, let item,
          self.generation == generation,
          self.hlsSeekRevision == revision,
          self.hlsSource != nil,
          self.sourcePlayer.currentItem === item
        else { return }
        self.hlsInitialSeekWatchdogTask?.cancel()
        self.hlsInitialSeekWatchdogTask = nil
        self.hlsSeekInFlight = false
        guard finished else {
          self.retryOrDegradeHLSInitialSeek(
            item: item,
            generation: generation,
            revision: revision,
            attempt: attempt,
            reason: "HLS元動画の開始位置を設定できませんでした"
          )
          return
        }
        let actualSourceTime = self.sourcePlayer.currentTime().seconds
        guard actualSourceTime.isFinite else {
          self.retryOrDegradeHLSInitialSeek(
            item: item,
            generation: generation,
            revision: revision,
            attempt: attempt,
            reason: "HLS元動画の時間情報を取得できませんでした"
          )
          return
        }
        // Anchor once from the actual completed seek for both live and VOD.
        // Unlike the removed running re-anchor, this does not hide drift; it
        // only removes the constant error when AVFoundation lands a VOD seek a
        // few frames away from the requested synthetic timestamp.
        self.hlsSourceTimeOffset = actualSourceTime - syntheticTarget
        self.hlsInitialSeekCompleted = true
        self.hlsSourceSeekCompleted = true
        self.hlsInitialSeekAttempt = 0
        self.hlsSeekToLiveWindowStart = false
        self.position = syntheticTarget
        if self.shouldPlay {
          self.state = .buffering
          self.playbackDetail = ""
          self.resumeIfBuffered()
        } else {
          self.state = .paused
          self.playbackDetail = ""
        }
      }
    }
  }

  private func retryOrDegradeHLSInitialSeek(
    item: AVPlayerItem,
    generation expectedGeneration: Int,
    revision: Int,
    attempt: Int,
    reason: String
  ) {
    guard generation == expectedGeneration,
      hlsSeekRevision == revision,
      hlsSource != nil,
      sourcePlayer.currentItem === item,
      !hlsRestoredClockFallbackActive
    else { return }
    hlsInitialSeekWatchdogTask?.cancel()
    hlsInitialSeekWatchdogTask = nil
    hlsSeekInFlight = false
    hlsSeekRevision &+= 1
    let retryRevision = hlsSeekRevision
    item.cancelPendingSeeks()
    if attempt >= hlsMaximumInitialSeekAttempts {
      degradeHLSSourcePlayback(
        generation: expectedGeneration,
        reason: "\(reason)（\(attempt)回試行）"
      )
      return
    }
    state = shouldPlay ? .buffering : .paused
    playbackDetail = "HLS元動画の再生位置を再確認中"
    Task { @MainActor [weak self, weak item] in
      try? await Task.sleep(nanoseconds: 100_000_000)
      guard let self, let item,
        self.generation == expectedGeneration,
        self.hlsSeekRevision == retryRevision,
        self.sourcePlayer.currentItem === item,
        !self.hlsRestoredClockFallbackActive
      else { return }
      self.seekHLSClockWhenReady(item: item, generation: expectedGeneration)
    }
  }

  private func scheduleHLSInitialSeekWatchdog(
    item: AVPlayerItem,
    generation expectedGeneration: Int,
    revision: Int,
    attempt: Int
  ) {
    hlsInitialSeekWatchdogTask?.cancel()
    let timeoutNanoseconds = UInt64(hlsOperationWatchdogSeconds * 1_000_000_000)
    hlsInitialSeekWatchdogTask = Task { @MainActor [weak self, weak item] in
      do {
        try await Task.sleep(nanoseconds: timeoutNanoseconds)
      } catch {
        return
      }
      guard let self, let item,
        self.generation == expectedGeneration,
        self.hlsSeekRevision == revision,
        self.hlsSeekInFlight,
        self.sourcePlayer.currentItem === item,
        !self.hlsRestoredClockFallbackActive
      else { return }
      self.retryOrDegradeHLSInitialSeek(
        item: item,
        generation: expectedGeneration,
        revision: revision,
        attempt: attempt,
        reason: "HLS元動画の開始位置設定がタイムアウトしました"
      )
    }
  }

  private func invalidateHLSInitialSeek() {
    hlsInitialSeekWatchdogTask?.cancel()
    hlsInitialSeekWatchdogTask = nil
    hlsSeekInFlight = false
    hlsSeekRevision &+= 1
    sourcePlayer.currentItem?.cancelPendingSeeks()
  }

  private func hlsSourceWaitingDescription() -> String {
    switch sourcePlayer.reasonForWaitingToPlay {
    case .evaluatingBufferingRate:
      return "HLS元動画の読込速度を確認中"
    case .toMinimizeStalls:
      return "HLS元動画をバッファ中"
    case .noItemToPlay:
      return "HLS元動画を開いています"
    default:
      return "HLS元動画のデコーダ開始待ち"
    }
  }

  private func installSourcePlaybackObservers(item: AVPlayerItem, generation: Int) {
    sourceItemStatusObservation?.invalidate()
    sourceTimeControlObservation?.invalidate()

    sourceItemStatusObservation = item.observe(\.status, options: [.initial, .new]) {
      [weak self, weak item] _, _ in
      Task { @MainActor in
        guard let self, let item else { return }
        self.updateSourcePlaybackState(item: item, generation: generation)
      }
    }
    sourceTimeControlObservation = sourcePlayer.observe(
      \.timeControlStatus,
      options: [.initial, .new]
    ) { [weak self, weak item] _, _ in
      Task { @MainActor in
        guard let self, let item else { return }
        self.updateSourcePlaybackState(item: item, generation: generation)
      }
    }
    sourceLoadedTimeRangesObservation = item.observe(
      \.loadedTimeRanges,
      options: [.initial, .new]
    ) { [weak self, weak item] _, _ in
      Task { @MainActor in
        guard let self, let item else { return }
        guard self.generation == generation, self.sourcePlayer.currentItem === item else { return }
        self.updateSourceBufferedDuration()
        self.resumeSourceAfterSeekIfBuffered()
      }
    }

    let token = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemDidPlayToEndTime,
      object: item,
      queue: .main
    ) { [weak self] _ in
      Task { @MainActor in
        guard let self, self.generation == generation, self.sourceOnlyPlayback else { return }
        self.shouldPlay = false
        self.state = .ended
      }
    }
    notificationTokens.append(token)
  }

  private func updateSourcePlaybackState(item: AVPlayerItem, generation: Int) {
    guard self.generation == generation, sourceOnlyPlayback, sourcePlayer.currentItem === item else {
      return
    }
    guard state != .seeking && state != .ended && state != .failed else { return }
    switch item.status {
    case .failed:
      fail("VR再生に失敗しました: \(item.error?.localizedDescription ?? "不明なAVPlayerエラー")")
    case .unknown:
      playbackDetail = "VR動画を開いています"
      state = .loading
    case .readyToPlay:
      guard shouldPlay else {
        playbackDetail = ""
        state = .paused
        return
      }
      if sourceSeekNeedsBuffer {
        resumeSourceAfterSeekIfBuffered()
        return
      }
      switch sourcePlayer.timeControlStatus {
      case .playing:
        playbackDetail = ""
        state = .playing
      case .waitingToPlayAtSpecifiedRate:
        playbackDetail = sourceWaitingDescription()
        state = .buffering
      case .paused:
        playbackDetail = "VRデコーダの開始待ち"
        state = .buffering
      @unknown default:
        playbackDetail = "VR動画をバッファ中"
        state = .buffering
      }
    @unknown default:
      playbackDetail = "VR動画を開いています"
      state = .loading
    }
  }

  private func sourceWaitingDescription() -> String {
    switch sourcePlayer.reasonForWaitingToPlay {
    case .evaluatingBufferingRate:
      return "VR動画の読込速度を確認中"
    case .toMinimizeStalls:
      return "VR動画をバッファ中"
    case .noItemToPlay:
      return "VR動画を開いています"
    default:
      return "VRデコーダの開始待ち"
    }
  }

  private func resumeSourceAfterSeekIfBuffered() {
    // Do not let a loaded-time-range notification from the pre-seek position
    // start playback before AVPlayer has completed the exact seek. The seek
    // completion moves the state from .seeking to .buffering; only then may
    // buffered media satisfy this transaction.
    guard sourceOnlyPlayback, sourceSeekNeedsBuffer, shouldPlay, state == .buffering else {
      return
    }
    updateSourceBufferedDuration()
    let remaining = max(0, duration - position)
    // A seek does not need to fill the entire configured rolling buffer.
    // Start as soon as a short playable lead is available; AVFoundation keeps
    // filling toward the configured upper limit while playback continues.
    let requested = max(1, min(runner?.previewBufferLimit ?? 8, previewSegmentSeconds))
    let required = min(requested, remaining)
    guard required <= 0.1 || bufferedSeconds + 0.1 >= required else {
      playbackDetail = "シーク後のバッファを準備中 \(bufferedSeconds.formatted(.number.precision(.fractionLength(1)))) / \(required.formatted(.number.precision(.fractionLength(1))))秒"
      state = .buffering
      return
    }
    sourceSeekNeedsBuffer = false
    playbackDetail = ""
    sourcePlayer.play()
    state = .playing
  }

  private func installTimeObserver() {
    if let timeObserver {
      sourcePlayer.removeTimeObserver(timeObserver)
    }
    let interval = hlsSource == nil ? 0.2 : hlsClockObservationIntervalSeconds
    timeObserver = sourcePlayer.addPeriodicTimeObserver(
      forInterval: CMTime(seconds: interval, preferredTimescale: 600),
      queue: .main
    ) { [weak self] time in
      Task { @MainActor in self?.tick(sourceSeconds: time.seconds) }
    }
  }

  private func tick(sourceSeconds: Double) {
    guard sourceSeconds.isFinite else { return }
    if hlsShouldUseRestoredClock {
      tickRestored(seconds: restoredPlayer.currentTime().seconds)
      return
    }
    let playbackTimelineSeconds: Double
    if hlsSource != nil {
      guard hlsSourceClockIsReadyForSynchronizedPlayback else {
        if generationHasStarted {
          tickRestored(seconds: restoredPlayer.currentTime().seconds)
        } else {
          position = requestedStartSeconds
        }
        return
      }
      playbackTimelineSeconds = max(0, sourceSeconds - hlsSourceTimeOffset)
      position = min(duration, playbackTimelineSeconds)
    } else if !sourceOnlyPlayback && !generationHasStarted
      && (state == .loading || state == .seeking || state == .buffering)
    {
      // A paused AVPlayer may briefly report its pre-seek timestamp while the
      // exact seek is completing. Keep the UI bar pinned to the user's target.
      position = requestedStartSeconds
      playbackTimelineSeconds = requestedStartSeconds
    } else {
      position = sourceSeconds
      playbackTimelineSeconds = sourceSeconds
    }
    if sourceOnlyPlayback {
      updateSourceBufferedDuration()
      // AVPlayer does not guarantee another loadedTimeRanges or preroll
      // callback after the threshold has been crossed. Reconcile the seek on
      // the same periodic clock that updates the displayed lead time so the UI
      // and playback state cannot disagree indefinitely.
      resumeSourceAfterSeekIfBuffered()
      return
    }
    retireSegmentsBeforeCurrentItem()
    updateBufferedDuration()
    guard state == .playing,
      !hlsSourceReachedEnd,
      let currentItem = restoredPlayer.currentItem,
      let active = itemSegments[ObjectIdentifier(currentItem)]
    else { return }
    let restoredLocalSeconds = restoredPlayer.currentTime().seconds
    guard restoredLocalSeconds.isFinite else { return }
    if hlsSource != nil {
      let itemIdentifier = ObjectIdentifier(currentItem)
      let now = ProcessInfo.processInfo.systemUptime
      if itemIdentifier != currentRestoredItemIdentifier {
        currentRestoredItemIdentifier = itemIdentifier
        currentRestoredItemStartedAt = now
        return
      }
      guard now - currentRestoredItemStartedAt >= hlsDriftCorrectionGraceSeconds else {
        return
      }
    }
    let restoredAbsolute = active.startSeconds + restoredLocalSeconds
    let allowedDrift = hlsSource == nil
      ? driftToleranceSeconds
      : hlsDriftToleranceSeconds
    let drift = restoredAbsolute - playbackTimelineSeconds
    if hlsSource != nil, hlsRestoredHeldForSourceCatchup {
      if drift <= hlsDriftResumeToleranceSeconds {
        hlsRestoredHeldForSourceCatchup = false
        if shouldPlay, sourcePlayer.timeControlStatus == .playing {
          restoredPlayer.play()
          playbackDetail = ""
        }
      } else {
        // Hysteresis prevents rapid pause/play oscillation around the trigger
        // threshold while the audible source clock catches the held frame.
        restoredPlayer.pause()
        playbackDetail = "HLS音声へ同期中"
        return
      }
    }
    guard abs(drift) > allowedDrift else { return }

    if hlsSource != nil, drift > allowedDrift {
      // Keep the current restored frame visible while its audible source
      // clock catches up. Moving the video backwards caused the prior rewind
      // loop; rewriting the source offset merely hid a real audio drift.
      hlsRestoredHeldForSourceCatchup = true
      restoredPlayer.pause()
      playbackDetail = "HLS音声へ同期中"
      return
    }

    var correctionItem = currentItem
    var correctionSegment = active
    if hlsSource != nil, drift < -allowedDrift {
      let availableItems = restoredPlayer.items()
      guard let targetItem = availableItems.first(where: { item in
        guard let segment = itemSegments[ObjectIdentifier(item)] else { return false }
        return playbackTimelineSeconds < segment.endSeconds
      }), let targetSegment = itemSegments[ObjectIdentifier(targetItem)] else {
        // Audio has outrun every restored item currently available. Freeze the
        // source clock until a future queue item covers it; seeking the current
        // item to its end one item at a time produced seconds of audible skew.
        sourcePlayer.pause()
        restoredPlayer.pause()
        state = .buffering
        playbackDetail = "復元映像がHLS音声へ追いつくのを待っています"
        return
      }
      if targetItem !== currentItem {
        restoredPlayer.pause()
        while let head = restoredPlayer.currentItem, head !== targetItem {
          restoredPlayer.advanceToNextItem()
        }
        guard restoredPlayer.currentItem === targetItem else {
          sourcePlayer.pause()
          state = .buffering
          playbackDetail = "復元映像の再生位置を準備中"
          return
        }
        releaseConsumedSegments(through: targetSegment.sequence - 1)
        correctionItem = targetItem
        correctionSegment = targetSegment
        currentRestoredItemIdentifier = ObjectIdentifier(targetItem)
        currentRestoredItemStartedAt = ProcessInfo.processInfo.systemUptime
      }
    }

    let maximumLocalTime = max(
      0,
      correctionSegment.endSeconds - correctionSegment.startSeconds - 0.001
    )
    let local = min(
      max(0, playbackTimelineSeconds - correctionSegment.startSeconds),
      maximumLocalTime
    )
    if hlsSource != nil {
      guard !hlsDriftCorrectionInFlight else { return }
      hlsDriftCorrectionInFlight = true
      // Freeze the authoritative audio clock while seeking the restored queue.
      // Otherwise the target is already stale by the time a loaded Core AI
      // frame finishes seeking, causing another correction every clock tick.
      sourcePlayer.pause()
      restoredPlayer.pause()
      state = .buffering
      playbackDetail = "HLS音声と復元映像を同期中"
      let expectedGeneration = generation
      let expectedItemIdentifier = ObjectIdentifier(correctionItem)
      let seekTolerance = CMTime(
        seconds: hlsDriftSeekToleranceSeconds,
        preferredTimescale: 600
      )
      restoredPlayer.seek(
        to: CMTime(seconds: local, preferredTimescale: 600),
        toleranceBefore: seekTolerance,
        toleranceAfter: seekTolerance
      ) { [weak self] finished in
        Task { @MainActor in
          guard let self, self.generation == expectedGeneration else { return }
          self.hlsDriftCorrectionInFlight = false
          guard !self.hlsRestoredClockFallbackActive,
            let currentItem = self.restoredPlayer.currentItem,
            ObjectIdentifier(currentItem) == expectedItemIdentifier
          else { return }
          guard finished else {
            if self.shouldPlay { self.resumeIfBuffered() }
            return
          }
          self.hlsRestoredHeldForSourceCatchup = false
          if self.shouldPlay {
            // Keep these starts adjacent for the same reason as the initial
            // preroll path: neither independent player may gain a main-thread
            // scheduling turn over the other.
            self.sourcePlayer.play()
            self.restoredPlayer.play()
            self.state = .playing
            self.playbackDetail = ""
          } else {
            self.state = .paused
            self.playbackDetail = ""
          }
        }
      }
    } else {
      restoredPlayer.seek(
        to: CMTime(seconds: local, preferredTimescale: 600),
        toleranceBefore: .zero,
        toleranceAfter: .zero
      )
    }
  }

  private var hlsShouldUseRestoredClock: Bool {
    hlsSource != nil
      && (hlsRestoredClockFallbackActive || hlsSourceReachedEnd)
      && generationHasStarted
  }

  private func tickRestored(seconds restoredLocalSeconds: Double) {
    guard hlsShouldUseRestoredClock, restoredLocalSeconds.isFinite else { return }
    retireSegmentsBeforeCurrentItem()
    guard let currentItem = restoredPlayer.currentItem,
      let active = itemSegments[ObjectIdentifier(currentItem)]
    else { return }
    position = max(0, active.startSeconds + restoredLocalSeconds)
    updateBufferedDuration()
  }

  /// The original HLS AVPlayer is only the audio/source-clock companion for
  /// the independently restored local queue. A CDN can invalidate that item
  /// while already-downloaded restoration windows remain perfectly usable.
  /// Keep the producer and restored queue alive instead of turning this
  /// auxiliary-player failure into a terminal restoration failure.
  private func degradeHLSSourcePlayback(
    generation expectedGeneration: Int,
    reason: String
  ) {
    guard generation == expectedGeneration,
      hlsSource != nil,
      state != .idle,
      state != .ended,
      state != .failed,
      !hlsSourceReachedEnd,
      !hlsRestoredClockFallbackActive
    else { return }

    let generationWasPlaying = generationHasStarted
    hlsRestoredClockFallbackActive = true
    hlsRestoredHeldForSourceCatchup = false
    hlsDriftCorrectionInFlight = false
    hlsInitialSeekAttempt = 0
    cancelSynchronizedHLSStart()

    sourceItemStatusObservation?.invalidate()
    sourceItemStatusObservation = nil
    sourceTimeControlObservation?.invalidate()
    sourceTimeControlObservation = nil
    hlsSourceSeekableTimeRangesObservation?.invalidate()
    hlsSourceSeekableTimeRangesObservation = nil
    for token in hlsNotificationTokens {
      NotificationCenter.default.removeObserver(token)
    }
    hlsNotificationTokens.removeAll(keepingCapacity: false)

    if let timeObserver {
      sourcePlayer.removeTimeObserver(timeObserver)
      self.timeObserver = nil
    }

    // Do not synchronously detach AVPlayer's current item or stop its loopback
    // proxy from a status/notification callback. AVFoundation may still be
    // unwinding resource requests for that item. Retain both until the normal
    // stop/generation boundary, while keeping the failed source silent and
    // paused so only the restored queue remains active.
    sourcePlayer.pause()
    sourcePlayer.volume = 0
    hlsSourceIsReady = false
    hlsSourceReady = false
    hlsSourceSeekCompleted = false
    invalidateHLSInitialSeek()
    showOriginal = false
    errorMessage = ""

    let detail = "元動画側の再生を継続できないため、復元映像のみ再生中（音声なし）"
    runner?.appendExternalLog(
      "HLS再生: 元動画側で \(reason)。復元済み映像へ切り替えて継続します（音声なし）\n"
    )
    if shouldPlay, generationWasPlaying, restoredPlayer.currentItem != nil {
      restoredPlayer.play()
      state = .playing
      playbackDetail = detail
    } else if shouldPlay {
      restoredPlayer.pause()
      state = .buffering
      playbackDetail = detail
      resumeIfBuffered()
    } else {
      restoredPlayer.pause()
      state = .paused
      playbackDetail = detail
    }
  }

  private func updateSourceBufferedDuration() {
    guard let item = sourcePlayer.currentItem else {
      sourceBufferedSeconds = 0
      if sourceOnlyPlayback { bufferedSeconds = 0 }
      return
    }
    var furthestEnd = position
    for value in item.loadedTimeRanges {
      let range = value.timeRangeValue
      let start = CMTimeGetSeconds(range.start)
      let end = CMTimeGetSeconds(CMTimeRangeGetEnd(range))
      guard start.isFinite, end.isFinite, start <= position + 0.25 else { continue }
      furthestEnd = max(furthestEnd, end)
    }
    sourceBufferedSeconds = max(0, furthestEnd - position)
    if sourceOnlyPlayback {
      bufferedSeconds = sourceBufferedSeconds
    }
  }

  private func updateBufferedDuration() {
    guard let last = queuedSegments.last else {
      bufferedSeconds = 0
      return
    }
    bufferedSeconds = max(0, last.endSeconds - position)
  }

  private func retireSegmentsBeforeCurrentItem() {
    guard let currentItem = restoredPlayer.currentItem,
      let activeSegment = itemSegments[ObjectIdentifier(currentItem)]
    else { return }
    // AVQueuePlayer can advance even when the per-item end notification is
    // delayed or dropped. The current item is an independent second source of
    // truth: every earlier sequence is definitely consumed and can be released.
    releaseConsumedSegments(through: activeSegment.sequence - 1)
  }

  private func releaseConsumedSegments(through sequence: Int) {
    guard sequence >= 0, sequence > releasedThroughSequence else { return }
    hlsProducer?.acknowledgeOutputConsumed(through: sequence)
    let releasedSegments = queuedSegments.filter { $0.sequence <= sequence }
    guard !releasedSegments.isEmpty else {
      releasedThroughSequence = sequence
      return
    }

    let releasedIdentifiers = itemSegments.compactMap { identifier, segment in
      segment.sequence <= sequence ? identifier : nil
    }
    for identifier in releasedIdentifiers {
      if let token = itemEndNotificationTokens.removeValue(forKey: identifier) {
        NotificationCenter.default.removeObserver(token)
      }
      itemSegments.removeValue(forKey: identifier)
    }
    queuedSegments.removeAll { $0.sequence <= sequence }
    for released in releasedSegments {
      streamingSegmentCodecs.removeValue(forKey: released.sequence)
    }
    releasedThroughSequence = sequence

    // The worker owns finalized segment files and acknowledges consumption by
    // deleting them. This makes its filesystem-based capacity check and the
    // player's queue advance one transaction. If the worker has already ended,
    // remove the files locally as a bounded fallback.
    let accepted = sendCommand([
      "command": "release_through",
      "sequence": sequence,
    ])
    if !accepted {
      for segment in releasedSegments {
        do {
          try FileManager.default.removeItem(at: segment.url)
        } catch where (error as NSError).code != NSFileNoSuchFileError {
          runner?.appendExternalLog(
            "再生済みバッファを解放できませんでした: \(segment.url.lastPathComponent): \(error.localizedDescription)\n"
          )
        } catch {
          // The worker may have already released this segment.
        }
      }
    }
    updateBufferedDuration()
  }

  @discardableResult
  private func sendCommand(_ payload: [String: Any]) -> Bool {
    guard let process = worker,
      process.isRunning,
      let inputPipe = workerInput,
      let data = try? JSONSerialization.data(withJSONObject: payload),
      var line = String(data: data, encoding: .utf8)?.data(using: .utf8)
    else { return false }
    line.append(0x0A)
    let handle = inputPipe.fileHandleForWriting
    guard MacChildProcessPipe.write(line, to: handle) else {
      if workerInput === inputPipe {
        workerInput = nil
      }
      try? handle.close()
      return false
    }
    return true
  }

  private func clearRestoredQueue(deleteFiles: Bool) {
    restoredPlayer.removeAllItems()
    for token in notificationTokens { NotificationCenter.default.removeObserver(token) }
    notificationTokens.removeAll()
    for token in itemEndNotificationTokens.values {
      NotificationCenter.default.removeObserver(token)
    }
    itemEndNotificationTokens.removeAll()
    if deleteFiles {
      for segment in queuedSegments { try? FileManager.default.removeItem(at: segment.url) }
    }
    queuedSegments.removeAll()
    itemSegments.removeAll()
    releasedThroughSequence = -1
    bufferedSeconds = 0
    if hlsSource == nil {
      sourceBufferedSeconds = 0
    }
  }

  /// Installs or removes a non-owning stream consumer. Attaching in the
  /// middle of playback publishes a consistent snapshot of the active
  /// generation and every segment retained by the local queue.
  func setStreamingEventConsumer(_ consumer: RealtimeStreamingEventConsumer?) {
    streamingEventConsumer = consumer
    guard let consumer, let source = streamingSource else { return }
    consumer(.reset(source))
    for segment in queuedSegments {
      consumer(
        .segment(
          RealtimeStreamingSegment(
            generation: source.generation,
            sequence: segment.sequence,
            startSeconds: segment.startSeconds,
            endSeconds: segment.endSeconds,
            url: segment.url,
            codec: streamingSegmentCodecs[segment.sequence] ?? "h264_videotoolbox"
          )
        )
      )
    }
    if generationReachedEOF {
      consumer(.ended(generation: source.generation))
    }
  }

  private func cleanupSession() {
    guard let sessionDirectory else { return }
    try? FileManager.default.removeItem(at: sessionDirectory)
    self.sessionDirectory = nil
  }

  private func cleanupSourceCompatibility() {
    guard let sourceCompatibilityDirectory else { return }
    try? FileManager.default.removeItem(at: sourceCompatibilityDirectory)
    self.sourceCompatibilityDirectory = nil
  }

  private func fail(_ message: String) {
    hlsProducer?.cancel()
    hlsMediaProxy?.stop()
    sourceItemStatusObservation?.invalidate()
    sourceItemStatusObservation = nil
    sourceTimeControlObservation?.invalidate()
    sourceTimeControlObservation = nil
    sourceLoadedTimeRangesObservation?.invalidate()
    sourceLoadedTimeRangesObservation = nil
    hlsSourceSeekableTimeRangesObservation?.invalidate()
    hlsSourceSeekableTimeRangesObservation = nil
    for token in hlsNotificationTokens {
      NotificationCenter.default.removeObserver(token)
    }
    hlsNotificationTokens.removeAll()
    hlsSourceIsReady = false
    hlsInitialSeekCompleted = false
    hlsSourceReady = false
    hlsSourceSeekCompleted = false
    hlsSourceTimeOffset = 0
    invalidateHLSInitialSeek()
    hlsSeekToLiveWindowStart = false
    hlsSourceReachedEnd = false
    hlsRestoredClockFallbackActive = false
    currentRestoredItemIdentifier = nil
    currentRestoredItemStartedAt = 0
    hlsRestoredHeldForSourceCatchup = false
    hlsDriftCorrectionInFlight = false
    hlsInitialSeekAttempt = 0
    cancelSynchronizedHLSStart()
    sourceSeekNeedsBuffer = false
    sourcePlayer.pause()
    restoredPlayer.pause()
    errorMessage = message
    playbackDetail = ""
    state = .failed
    runner?.appendExternalLog("Realtime preview: \(message)\n")
  }
}

struct RealtimePlayerView: View {
  @ObservedObject var controller: RealtimePlayerController
  @ObservedObject var runner: RestorationRunner
  @State private var seekPosition = 0.0
  @State private var isScrubbing = false

  var body: some View {
    VStack(spacing: 12) {
      PathRow(
        title: "再生動画",
        icon: "film",
        url: controller.previewInputURL,
        action: { controller.choosePreviewInput(runner: runner) }
      )

      if !controller.isVRVideo {
        VStack(alignment: .leading, spacing: 8) {
          // The Swift native preview picks its own Core AI assets. Only the
          // bundled Python worker takes these selections.
          if runner.usesPythonEngine {
            HStack(spacing: 12) {
              Picker("復元モデル", selection: $runner.previewRestorationModel) {
                ForEach(runner.restorationModels, id: \.self) { model in
                  Text(L(model)).tag(model)
                }
              }
              .frame(maxWidth: 430)
              if runner.previewRestorationModel == "カスタム" {
                TextField("モデル名またはパス", text: $runner.previewCustomRestorationModel)
                  .textFieldStyle(.roundedBorder)
                  .frame(maxWidth: 360)
                Button {
                  runner.choosePath(\.previewCustomRestorationModel)
                } label: {
                  Image(systemName: "folder")
                }
              }
              Spacer()
            }
            HStack(spacing: 12) {
              Picker("再生用検出モデル", selection: $runner.previewDetectionModel) {
                ForEach(runner.previewDetectionModels, id: \.self) { model in
                  Text(L(model)).tag(model)
                }
              }
              .frame(maxWidth: 430)
              if runner.previewDetectionModel == "カスタム" {
                TextField("モデル名またはパス", text: $runner.previewCustomDetectionModel)
                  .textFieldStyle(.roundedBorder)
                  .frame(maxWidth: 360)
                Button {
                  runner.choosePath(\.previewCustomDetectionModel)
                } label: {
                  Image(systemName: "folder")
                }
              }
              Spacer()
            }
          }
          HStack(spacing: 12) {
            Toggle("リアルタイム最適化", isOn: $runner.previewRealtimeOptimization)
              .toggleStyle(.checkbox)
            Spacer()
          }
          if runner.previewRealtimeOptimization {
            Text("復元は維持し、再生中は合成エフェクトとROIエンハンサーをバイパスします")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }
      }

      ZStack {
        Color.black
        if runner.previewProjectionMode == "通常" {
          if controller.prefersSourceVideoLayer {
            VideoPlayer(player: controller.sourcePlayer)
          } else {
            VideoPlayer(player: controller.restoredPlayer)
          }
        } else {
          VRPreviewSceneView(
            playerItem: controller.sourceOnlyPlayback || controller.showOriginal
              ? controller.sourcePlayer.currentItem
              : controller.restoredPlayer.currentItem,
            projection: PreviewProjectionMode(rawValue: runner.previewProjectionMode) ?? .vr180,
            layout: PreviewVideoLayout(rawValue: runner.previewVideoLayout) ?? .sbs,
            eye: PreviewEye(rawValue: runner.previewEye) ?? .left,
            cameraFOV: runner.previewCameraFOV
          )
        }
        if controller.shouldShowProcessingOverlay {
          VStack(spacing: 8) {
            ProgressView()
            Text(controller.processingOverlayLabel)
          }
          .padding(18)
          .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 10))
          .foregroundStyle(.white)
        }
      }
      .aspectRatio(16 / 9, contentMode: .fit)
      .clipShape(RoundedRectangle(cornerRadius: 8))

      HStack {
        Text(time(controller.position))
          .font(.caption.monospacedDigit()).frame(width: 68)
        Slider(
          value: Binding(
            get: { isScrubbing ? seekPosition : controller.position },
            set: { seekPosition = $0 }
          ),
          in: 0...max(controller.duration, 0.01),
          onEditingChanged: { editing in
            if editing {
              seekPosition = controller.position
              isScrubbing = true
            } else {
              let target = seekPosition
              isScrubbing = false
              controller.seek(to: target)
            }
          }
        )
        .disabled(!controller.isSeekable)
        Text(time(controller.duration))
          .font(.caption.monospacedDigit()).frame(width: 68)
        if controller.isLiveHLSInput {
          Text("ライブ")
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
        }
      }

      HStack(spacing: 12) {
        Text("バッファ上限")
        Slider(
          value: Binding(
            get: { runner.previewBufferLimit },
            set: { value in
              runner.previewBufferLimit = value
              controller.setBufferLimit(value)
            }
          ),
          in: 1...60,
          step: 1
        )
        .frame(maxWidth: 320)
        Text("\(Int(runner.previewBufferLimit))秒")
          .font(.caption.monospacedDigit())
          .frame(width: 48, alignment: .trailing)
        Spacer()
      }

      if controller.isVRVideo {
        HStack(spacing: 12) {
          Picker("表示", selection: $runner.previewProjectionMode) {
            ForEach(PreviewProjectionMode.allCases.filter { $0 != .normal }) { mode in
              Text(mode.displayName).tag(mode.rawValue)
            }
          }
          .frame(width: 150)
          Picker("形式", selection: $runner.previewVideoLayout) {
            ForEach(PreviewVideoLayout.allCases) { layout in
              Text(layout.displayName).tag(layout.rawValue)
            }
          }
          .frame(width: 140)
          Picker("目", selection: $runner.previewEye) {
            ForEach(PreviewEye.allCases) { eye in
              Text(eye.displayName).tag(eye.rawValue)
            }
          }
          .frame(width: 110)
          Text("視野角")
          Slider(value: $runner.previewCameraFOV, in: 45...105, step: 1)
            .frame(maxWidth: 220)
          Text("\(Int(runner.previewCameraFOV))°")
            .font(.caption.monospacedDigit())
            .frame(width: 44, alignment: .trailing)
          Spacer()
        }
      } else if controller.isDetectingVR {
        HStack(spacing: 8) {
          ProgressView().controlSize(.small)
          Text(L(controller.vrDetectionDetail)).font(.caption).foregroundStyle(.secondary)
          Spacer()
        }
      }

      HStack(spacing: 12) {
        if controller.state == .idle || controller.state == .ended || controller.state == .failed {
          Button { controller.startSelectedInput(runner: runner) } label: { Label("再生", systemImage: "play.fill") }
            .buttonStyle(.borderedProminent)
            .disabled(controller.previewInputURL == nil || controller.isDetectingVR)
        } else if controller.state == .playing {
          Button(action: controller.togglePlayback) { Label("一時停止", systemImage: "pause.fill") }
        } else {
          Button(action: controller.togglePlayback) { Label("再生", systemImage: "play.fill") }
        }
        Button(role: .destructive, action: { controller.stop() }) {
          Label("停止", systemImage: "stop.fill")
        }
          .disabled(controller.state == .idle)
        Toggle(
          "処理前",
          isOn: Binding(
            get: { controller.showOriginal },
            set: { controller.showOriginal = $0 && controller.canShowOriginal }
          )
        )
        .toggleStyle(.switch)
        .disabled(!controller.canShowOriginal)
        Spacer()
        Text("先読み \(controller.bufferedSeconds, specifier: "%.1f")秒")
          .font(.caption).foregroundStyle(.secondary)
        Button {
          controller.setMuted(!controller.muted)
        } label: {
          Image(systemName: controller.muted ? "speaker.slash.fill" : "speaker.wave.2.fill")
        }.buttonStyle(.borderless)
        Slider(
          value: Binding(get: { controller.volume }, set: controller.setVolume),
          in: 0...1
        ).frame(width: 100)
        Button("設定を反映して再開") { controller.restartWithCurrentSettings(runner: runner) }
          .disabled(controller.state == .idle)
      }
      if !controller.errorMessage.isEmpty {
        Text(L(controller.errorMessage)).foregroundStyle(.red).font(.caption)
      } else {
        Text(L(controller.statusLabel)).font(.caption).foregroundStyle(.secondary)
      }
    }
    .padding(.vertical, 12)
  }

  private func time(_ seconds: Double) -> String {
    guard seconds.isFinite else { return "00:00" }
    let value = max(0, Int(seconds))
    return String(format: "%02d:%02d", value / 60, value % 60)
  }
}

private extension RealtimePlayerController {
  var showsRestoredFrameWhileHLSBuffers: Bool {
    hlsSource != nil
      && !sourceOnlyPlayback
      && !showOriginal
      && generationHasStarted
      && restoredPlayer.currentItem != nil
      && (state == .loading || state == .buffering || state == .seeking)
  }

  var prefersSourceVideoLayer: Bool {
    showOriginal
      || (showsSourceFrameWhilePreparingRestoration && !showsRestoredFrameWhileHLSBuffers)
  }

  var shouldShowProcessingOverlay: Bool {
    if showsRestoredFrameWhileHLSBuffers { return false }
    if sourceOnlyPlayback {
      return state == .loading || state == .buffering || state == .seeking
    }
    return state == .loading || state == .buffering || state == .seeking
  }

  var processingOverlayLabel: String {
    if !playbackDetail.isEmpty { return L(playbackDetail) }
    if !sourceOnlyPlayback { return L("バッファ中") }
    if state == .seeking { return L("シーク中") }
    return playbackDetail.isEmpty ? L("VR動画を準備中") : L(playbackDetail)
  }

  var statusLabel: String {
    if sourceOnlyPlayback {
      switch state {
      case .loading, .buffering:
        return playbackDetail.isEmpty ? L("VR動画を準備中") : L(playbackDetail)
      default: return state.label
      }
    }
    if !playbackDetail.isEmpty { return L(playbackDetail) }
    return state.label
  }
}
