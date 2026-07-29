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
}

private struct PreviewSegment {
  let sequence: Int
  let startSeconds: Double
  let endSeconds: Double
  let url: URL
}

private struct PreparedSourcePlayerItem {
  let item: AVPlayerItem
  let duration: Double
  let resourceLoader: HEV1LoopbackServer?

  var usesVirtualContainer: Bool { resourceLoader != nil }
}

private enum SourcePlaybackError: LocalizedError {
  case missingHEV1SampleEntry
  case incompatibleVirtualContainer
  case invalidFileSize
  case loopbackServerFailed

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
  let driftToleranceSeconds = 0.080

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
  private var itemSegments: [ObjectIdentifier: PreviewSegment] = [:]
  private var notificationTokens: [NSObjectProtocol] = []
  private var timeObserver: Any?
  private var sourceItemStatusObservation: NSKeyValueObservation?
  private var sourceTimeControlObservation: NSKeyValueObservation?
  private var sourceLoadedTimeRangesObservation: NSKeyValueObservation?
  private var sourceResourceLoader: HEV1LoopbackServer?
  private var sessionDirectory: URL?
  private var requestedStartSeconds = 0.0
  private var shouldPlay = true
  private var generationHasStarted = false
  private var generationStartPending = false
  private var generationReachedEOF = false
  private var sourceSeekNeedsBuffer = false
  private var previewSegmentSeconds = 2.0
  private weak var runner: RestorationRunner?

  var showsSourceFrameWhilePreparingRestoration: Bool {
    guard !sourceOnlyPlayback, !generationHasStarted else { return false }
    return state == .loading || state == .seeking || state == .buffering
  }

  init() {
    restoredPlayer.isMuted = true
    restoredPlayer.actionAtItemEnd = .advance
    sourcePlayer.volume = 1
  }

  deinit {
    sourceItemStatusObservation?.invalidate()
    sourceTimeControlObservation?.invalidate()
    sourceLoadedTimeRangesObservation?.invalidate()
    if let timeObserver {
      sourcePlayer.removeTimeObserver(timeObserver)
    }
    for token in notificationTokens {
      NotificationCenter.default.removeObserver(token)
    }
  }

  func start(
    runner: RestorationRunner,
    at startSeconds: Double = 0,
    autoPlay: Bool = true,
    preserveCurrentSource: Bool = false
  ) {
    let canReuseCurrentSource = preserveCurrentSource && sourcePlayer.currentItem != nil
    stop(preserveSourceItem: canReuseCurrentSource)
    let retirement = workerRetirementTask
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
        startSourceOnlyPlayback(
          input: input,
          generation: startingGeneration,
          startSeconds: startSeconds
        )
        return
      }

      let python = resources.appendingPathComponent("runtime/bin/python3.12")
      let script = resources.appendingPathComponent(
        "runtime/lib/python3.12/site-packages/mioh_preview_worker.py"
      )
      guard FileManager.default.isExecutableFile(atPath: python.path) else {
        throw RunnerError.missingResource("Python runtime")
      }
      guard FileManager.default.fileExists(atPath: script.path) else {
        throw RunnerError.missingResource("Realtime preview worker")
      }

      let process = Process()
      let inputPipe = Pipe()
      let outputPipe = Pipe()
      let errorPipe = Pipe()
      process.executableURL = python
      process.arguments = [script.path] + (try runner.previewArguments(
        resources: resources,
        outputDirectory: session,
        input: input
      )) + [
        "--start-ns", String(Int64(startSeconds * 1_000_000_000)),
        "--generation", String(startingGeneration),
      ]
      process.environment = runner.environment(resources: resources, python: python)
      process.standardInput = inputPipe
      process.standardOutput = outputPipe
      process.standardError = errorPipe

      Task { @MainActor [self] in
        if let retirement {
          await retirement.value
        }
        guard self.generation == startingGeneration else { return }
        self.workerRetirementTask = nil
        let prepared: PreparedSourcePlayerItem?
        if canReuseCurrentSource {
          prepared = nil
        } else {
          do {
            prepared = try await self.prepareSourcePlayerItem(input: input)
          } catch {
            guard self.generation == startingGeneration else { return }
            self.fail("元動画を開けません: \(error.localizedDescription)")
            self.cleanupSession()
            return
          }
        }
        guard self.generation == startingGeneration, self.state == .loading || self.state == .buffering else {
          return
        }
        self.worker = process
        self.workerInput = inputPipe
        self.stdoutPipe = outputPipe
        self.stderrPipe = errorPipe
        if let prepared {
          self.sourceResourceLoader = prepared.resourceLoader
          self.sourcePlayer.replaceCurrentItem(with: prepared.item)
        }
        self.sourcePlayer.volume = self.muted ? 0 : Float(self.volume)
        if prepared?.usesVirtualContainer == true {
          self.runner?.appendExternalLog(
            "再生: AVFoundation互換の仮想コンテナを使用します（ファイル変換なし）\n"
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
        process.terminationHandler = { [weak self, outputPipe, errorPipe] completed in
          Task { @MainActor in
            guard let self, self.worker === completed else { return }
            self.worker = nil
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

  private func startSourceOnlyPlayback(
    input: URL,
    generation startingGeneration: Int,
    startSeconds: Double
  ) {
    Task { @MainActor [self] in
      let prepared: PreparedSourcePlayerItem
      do {
        prepared = try await self.prepareSourcePlayerItem(input: input)
      } catch {
        guard self.generation == startingGeneration, self.sourceOnlyPlayback else { return }
        self.fail("VR動画を開けません: \(error.localizedDescription)")
        self.cleanupSession()
        return
      }
      guard self.generation == startingGeneration, self.sourceOnlyPlayback else { return }
      let item = prepared.item
      item.preferredForwardBufferDuration = max(1, self.runner?.previewBufferLimit ?? 8)
      self.sourceResourceLoader = prepared.resourceLoader
      self.sourcePlayer.replaceCurrentItem(with: item)
      self.sourcePlayer.volume = self.muted ? 0 : Float(self.volume)
      self.bufferedSeconds = 0
      self.duration = prepared.duration
      self.installTimeObserver()
      self.installSourcePlaybackObservers(item: item, generation: startingGeneration)
      self.runner?.appendExternalLog("VR再生: 復元モデルを読み込まず、元動画を直接再生します\n")
      if prepared.usesVirtualContainer {
        self.runner?.appendExternalLog(
          "VR再生: 全編remuxを行わず、AVFoundation互換の仮想コンテナを使用します\n"
        )
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

  func togglePlayback() {
    if state == .playing {
      shouldPlay = false
      sourcePlayer.pause()
      restoredPlayer.pause()
      state = .paused
    } else if state == .paused || state == .buffering {
      shouldPlay = true
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
      start(runner: runner, at: position)
    }
  }

  func choosePreviewInput(runner: RestorationRunner) {
    let panel = NSOpenPanel()
    panel.title = "再生動画を選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    stop()
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
    start(runner: runner, at: position)
  }

  func setVolume(_ value: Double) {
    volume = min(max(value, 0), 1)
    sourcePlayer.volume = muted ? 0 : Float(volume)
  }

  func setBufferLimit(_ seconds: Double) {
    if sourceOnlyPlayback {
      sourcePlayer.currentItem?.preferredForwardBufferDuration = max(1, seconds)
      return
    }
    guard worker != nil else { return }
    sendCommand(["command": "set_buffer_limit", "seconds": seconds])
  }

  func setMuted(_ value: Bool) {
    muted = value
    sourcePlayer.volume = value ? 0 : Float(volume)
  }

  func stop(preserveSourceItem: Bool = false) {
    generation += 1
    sourceItemStatusObservation?.invalidate()
    sourceItemStatusObservation = nil
    sourceTimeControlObservation?.invalidate()
    sourceTimeControlObservation = nil
    sourceLoadedTimeRangesObservation?.invalidate()
    sourceLoadedTimeRangesObservation = nil
    sourceOnlyPlayback = false
    shouldPlay = false
    generationHasStarted = false
    generationStartPending = false
    generationReachedEOF = false
    sourceSeekNeedsBuffer = false
    sourcePlayer.pause()
    if !preserveSourceItem {
      sourcePlayer.replaceCurrentItem(with: nil)
      sourceResourceLoader = nil
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
    cleanupSession()
    state = .idle
    bufferedSeconds = 0
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
  }

  private func prepareSourcePlayerItem(input: URL) async throws -> PreparedSourcePlayerItem {
    let source = AVURLAsset(url: input)
    let durationTime = try await source.load(.duration)
    let durationSeconds = CMTimeGetSeconds(durationTime)
    let duration = durationSeconds.isFinite ? max(0, durationSeconds) : 0
    if try await source.load(.isPlayable) {
      return PreparedSourcePlayerItem(
        item: AVPlayerItem(asset: source),
        duration: duration,
        resourceLoader: nil
      )
    }

    // HEV1 8K files may contain perfectly decodable samples while AVFoundation
    // rejects the MP4 sample-entry identifier.  A loopback byte-range server
    // exposes the original file byte-for-byte except for hev1 -> hvc1 inside the
    // moov atom. This preserves random access without modifying or copying it.
    let resourceLoader = try HEV1LoopbackServer(sourceURL: input)
    let compatibleAsset = resourceLoader.makeAsset()
    return PreparedSourcePlayerItem(
      item: AVPlayerItem(asset: compatibleAsset),
      duration: duration,
      resourceLoader: resourceLoader
    )
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
      resumeIfBuffered()
    case "ended":
      generationReachedEOF = true
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
    notificationTokens.append(token)
    updateBufferedDuration()
  }

  private func finished(item: AVPlayerItem) {
    guard let segment = itemSegments[ObjectIdentifier(item)] else { return }
    releaseConsumedSegments(through: segment.sequence)
    if queuedSegments.isEmpty && state == .playing {
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
    if state == .paused {
      startPlayersFromCurrentPosition()
      return
    }
    let nominalRequired =
      Double(generationHasStarted ? rebufferSegmentCount : startupSegmentCount)
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

  private func startPlayersFromCurrentPosition() {
    sourcePlayer.play()
    restoredPlayer.play()
    state = .playing
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
    guard sourceOnlyPlayback, sourceSeekNeedsBuffer, shouldPlay else { return }
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
    timeObserver = sourcePlayer.addPeriodicTimeObserver(
      forInterval: CMTime(seconds: 0.2, preferredTimescale: 600),
      queue: .main
    ) { [weak self] time in
      Task { @MainActor in self?.tick(sourceSeconds: time.seconds) }
    }
  }

  private func tick(sourceSeconds: Double) {
    guard sourceSeconds.isFinite else { return }
    if !sourceOnlyPlayback && !generationHasStarted
      && (state == .loading || state == .seeking || state == .buffering)
    {
      // A paused AVPlayer may briefly report its pre-seek timestamp while the
      // exact seek is completing. Keep the UI bar pinned to the user's target.
      position = requestedStartSeconds
    } else {
      position = sourceSeconds
    }
    if sourceOnlyPlayback {
      updateSourceBufferedDuration()
      return
    }
    retireSegmentsBeforeCurrentItem()
    updateBufferedDuration()
    guard state == .playing,
      let active = queuedSegments.first,
      restoredPlayer.currentTime().seconds.isFinite
    else { return }
    let restoredAbsolute = active.startSeconds + restoredPlayer.currentTime().seconds
    if abs(restoredAbsolute - sourceSeconds) > driftToleranceSeconds {
      let local = max(0, sourceSeconds - active.startSeconds)
      restoredPlayer.seek(
        to: CMTime(seconds: local, preferredTimescale: 600),
        toleranceBefore: .zero,
        toleranceAfter: .zero
      )
    }
  }

  private func updateSourceBufferedDuration() {
    guard let item = sourcePlayer.currentItem else {
      bufferedSeconds = 0
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
    bufferedSeconds = max(0, furthestEnd - position)
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
    let releasedSegments = queuedSegments.filter { $0.sequence <= sequence }
    guard !releasedSegments.isEmpty else {
      releasedThroughSequence = sequence
      return
    }

    let releasedIdentifiers = itemSegments.compactMap { identifier, segment in
      segment.sequence <= sequence ? identifier : nil
    }
    for identifier in releasedIdentifiers {
      itemSegments.removeValue(forKey: identifier)
    }
    queuedSegments.removeAll { $0.sequence <= sequence }
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
    guard let handle = workerInput?.fileHandleForWriting,
      let data = try? JSONSerialization.data(withJSONObject: payload),
      var line = String(data: data, encoding: .utf8)?.data(using: .utf8)
    else { return false }
    line.append(0x0A)
    do {
      try handle.write(contentsOf: line)
      return true
    } catch {
      return false
    }
  }

  private func clearRestoredQueue(deleteFiles: Bool) {
    restoredPlayer.removeAllItems()
    for token in notificationTokens { NotificationCenter.default.removeObserver(token) }
    notificationTokens.removeAll()
    if deleteFiles {
      for segment in queuedSegments { try? FileManager.default.removeItem(at: segment.url) }
    }
    queuedSegments.removeAll()
    itemSegments.removeAll()
    releasedThroughSequence = -1
    bufferedSeconds = 0
  }

  private func cleanupSession() {
    guard let sessionDirectory else { return }
    try? FileManager.default.removeItem(at: sessionDirectory)
    self.sessionDirectory = nil
  }

  private func fail(_ message: String) {
    sourceItemStatusObservation?.invalidate()
    sourceItemStatusObservation = nil
    sourceTimeControlObservation?.invalidate()
    sourceTimeControlObservation = nil
    sourceLoadedTimeRangesObservation?.invalidate()
    sourceLoadedTimeRangesObservation = nil
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
            Toggle("リアルタイム最適化", isOn: $runner.previewRealtimeOptimization)
              .toggleStyle(.checkbox)
            Spacer()
          }
          if runner.previewRealtimeOptimization {
            Text("復元は維持し、再生中だけROIエンハンサーと拡大後処理を省略します")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }
      }

      ZStack {
        Color.black
        if runner.previewProjectionMode == "通常" {
          VideoPlayer(player: controller.sourcePlayer)
            .opacity(
              controller.showOriginal || controller.showsSourceFrameWhilePreparingRestoration
                ? 1 : 0.001
            )
          VideoPlayer(player: controller.restoredPlayer)
            .opacity(
              controller.showOriginal || controller.showsSourceFrameWhilePreparingRestoration
                ? 0.001 : 1
            )
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
        Text(time(controller.duration))
          .font(.caption.monospacedDigit()).frame(width: 68)
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
          Button { controller.start(runner: runner) } label: { Label("再生", systemImage: "play.fill") }
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
        Toggle("処理前", isOn: $controller.showOriginal).toggleStyle(.switch)
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
  var shouldShowProcessingOverlay: Bool {
    if sourceOnlyPlayback {
      return state == .loading || state == .buffering || state == .seeking
    }
    return state == .loading || state == .buffering || state == .seeking
  }

  var processingOverlayLabel: String {
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
    return state.label
  }
}
