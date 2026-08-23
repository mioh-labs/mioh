// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import AVFoundation
import CoreMedia
import Foundation

struct UpscalerMediaInfo: Equatable, Sendable {
  var width: Int
  var height: Int
  var frameRate: Double
  var durationSeconds: Double
  var videoCodec: String
  var dataRateBitsPerSecond: Double
  var audio: String?
  var fileSizeBytes: Int64?

  var resolutionText: String { "\(width)×\(height)" }

  var frameRateText: String {
    if abs(frameRate - frameRate.rounded()) < 0.0005 {
      return String(Int(frameRate.rounded()))
    }
    return String(format: "%.3f", frameRate)
  }

  var durationText: String {
    guard durationSeconds.isFinite, durationSeconds > 0 else { return "—" }
    let total = Int(durationSeconds.rounded())
    let hours = total / 3600
    let minutes = (total % 3600) / 60
    let seconds = total % 60
    return hours > 0
      ? String(format: "%d:%02d:%02d", hours, minutes, seconds)
      : String(format: "%d:%02d", minutes, seconds)
  }

  var bitRateText: String? {
    guard dataRateBitsPerSecond > 0 else { return nil }
    let mbps = dataRateBitsPerSecond / 1_000_000
    return mbps >= 10
      ? String(format: "%.0f Mbps", mbps)
      : String(format: "%.1f Mbps", mbps)
  }

  var fileSizeText: String? {
    fileSizeBytes.map {
      ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)
    }
  }
}

enum UpscalerMediaProbe {
  enum Outcome: Sendable {
    case success(UpscalerMediaInfo)
    case failure(String)
  }

  static func read(url: URL) async -> Outcome {
    let asset = AVURLAsset(url: url)
    do {
      guard let track = try await asset.loadTracks(withMediaType: .video).first
      else { return .failure("映像トラックが見つかりません") }
      let size = try await track.load(.naturalSize)
      let transform = try await track.load(.preferredTransform)
      let oriented = size.applying(transform)
      let frameRate = try await track.load(.nominalFrameRate)
      let dataRate = try await track.load(.estimatedDataRate)
      let duration = try await asset.load(.duration)
      let formats = try await track.load(.formatDescriptions)
      let fileSize = (
        try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize
      ).flatMap { $0 }
      return .success(
        UpscalerMediaInfo(
          width: max(1, Int(abs(oriented.width).rounded())),
          height: max(1, Int(abs(oriented.height).rounded())),
          frameRate: max(0, Double(frameRate)),
          durationSeconds: duration.seconds,
          videoCodec: formats.first.map { name(for: $0) } ?? "不明",
          dataRateBitsPerSecond: max(0, Double(dataRate)),
          audio: await audioSummary(of: asset),
          fileSizeBytes: fileSize.map(Int64.init)
        )
      )
    } catch {
      return .failure(error.localizedDescription)
    }
  }

  private static func audioSummary(of asset: AVAsset) async -> String? {
    guard
      let track = try? await asset.loadTracks(withMediaType: .audio).first,
      let formats = try? await track.load(.formatDescriptions),
      let format = formats.first
    else { return nil }
    var parts = [name(for: format)]
    if let basic = CMAudioFormatDescriptionGetStreamBasicDescription(format)?
      .pointee
    {
      if basic.mChannelsPerFrame > 0 {
        parts.append("\(basic.mChannelsPerFrame)ch")
      }
      if basic.mSampleRate > 0 {
        parts.append(String(format: "%.1fkHz", basic.mSampleRate / 1000))
      }
    }
    return parts.joined(separator: " ")
  }

  private static func name(for format: CMFormatDescription) -> String {
    let subtype = CMFormatDescriptionGetMediaSubType(format)
    switch subtype {
    case kCMVideoCodecType_H264: return "H.264"
    case kCMVideoCodecType_HEVC: return "HEVC"
    case kCMVideoCodecType_HEVCWithAlpha: return "HEVC+α"
    case kCMVideoCodecType_AV1: return "AV1"
    case kCMVideoCodecType_VP9: return "VP9"
    case kCMVideoCodecType_MPEG4Video: return "MPEG-4"
    case kCMVideoCodecType_MPEG2Video: return "MPEG-2"
    case kCMVideoCodecType_AppleProRes422: return "ProRes 422"
    case kCMVideoCodecType_AppleProRes422HQ: return "ProRes 422 HQ"
    case kCMVideoCodecType_AppleProRes4444: return "ProRes 4444"
    case kAudioFormatMPEG4AAC: return "AAC"
    case kAudioFormatLinearPCM: return "PCM"
    case kAudioFormatOpus: return "Opus"
    case kAudioFormatFLAC: return "FLAC"
    case kAudioFormatAC3: return "AC-3"
    default: return fourCharCode(subtype)
    }
  }

  private static func fourCharCode(_ value: FourCharCode) -> String {
    let bytes = [
      UInt8((value >> 24) & 0xFF), UInt8((value >> 16) & 0xFF),
      UInt8((value >> 8) & 0xFF), UInt8(value & 0xFF),
    ]
    let text = String(bytes: bytes, encoding: .ascii) ?? ""
    let trimmed = text.trimmingCharacters(in: .whitespaces)
    return trimmed.isEmpty ? "不明" : trimmed
  }
}
