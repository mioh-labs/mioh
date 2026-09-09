import AppKit
import Foundation
@preconcurrency import QuickLookThumbnailing

/// Asks Quick Look to prepare small video thumbnails only while the open panel
/// is closed. Quick Look itself owns the cache; mioh pauses all warming before
/// showing `NSOpenPanel` so it cannot compete with a selection click.
@MainActor
final class InputPanelThumbnailCache {
  static let shared = InputPanelThumbnailCache()

  nonisolated private static let videoExtensions: Set<String> = [
    "3g2", "3gp", "avi", "flv", "m2ts", "m4v", "mkv", "mov", "mp4",
    "mpeg", "mpg", "mts", "ts", "webm", "wmv",
  ]

  private let generator = QLThumbnailGenerator.shared
  private var directoryTask: Task<Void, Never>?
  private var currentRequest: QLThumbnailGenerator.Request?
  private var currentRequestID: UUID?
  private var queuedURLs: [URL] = []
  private var warmedPaths: Set<String> = []
  private var directoryGeneration = UUID()

  func prepare(initialURL: URL?) {
    guard let initialURL else { return }
    if Self.isVideo(initialURL) {
      enqueue(initialURL, first: true)
      prepareDirectory(initialURL.deletingLastPathComponent())
    } else {
      prepareDirectory(initialURL)
    }
  }

  func pause() {
    directoryTask?.cancel()
    directoryTask = nil
    queuedURLs.removeAll(keepingCapacity: false)
    if let currentRequest {
      generator.cancel(currentRequest)
      self.currentRequest = nil
      currentRequestID = nil
    }
  }

  private func prepareDirectory(_ directory: URL) {
    directoryTask?.cancel()
    directoryGeneration = UUID()
    let generation = directoryGeneration

    directoryTask = Task { [weak self] in
      let urls = await Task.detached(priority: .utility) {
        let entries = (try? FileManager.default.contentsOfDirectory(
          at: directory,
          includingPropertiesForKeys: nil,
          options: [.skipsHiddenFiles]
        )) ?? []
        return entries
          .filter(Self.isVideo)
          .sorted {
            $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent)
              == .orderedAscending
          }
      }.value

      guard let self, !Task.isCancelled, generation == self.directoryGeneration
      else { return }
      for url in urls where !self.warmedPaths.contains(url.path) {
        if !self.queuedURLs.contains(url) { self.queuedURLs.append(url) }
      }
      self.generateNextIfNeeded()
    }
  }

  private func enqueue(_ url: URL, first: Bool) {
    guard Self.isVideo(url), !warmedPaths.contains(url.path) else { return }
    queuedURLs.removeAll { $0 == url }
    if first {
      queuedURLs.insert(url, at: 0)
    } else {
      queuedURLs.append(url)
    }
    generateNextIfNeeded()
  }

  private func generateNextIfNeeded() {
    guard currentRequest == nil else { return }
    while !queuedURLs.isEmpty {
      let url = queuedURLs.removeFirst()
      guard !warmedPaths.contains(url.path) else { continue }

      let request = QLThumbnailGenerator.Request(
        fileAt: url,
        size: CGSize(width: 320, height: 180),
        scale: 1,
        representationTypes: [.lowQualityThumbnail, .thumbnail]
      )
      let requestID = UUID()
      currentRequest = request
      currentRequestID = requestID
      generator.generateBestRepresentation(for: request) { [weak self] _, _ in
        Task { @MainActor [weak self] in
          guard let self, self.currentRequestID == requestID else { return }
          self.warmedPaths.insert(url.path)
          self.currentRequest = nil
          self.currentRequestID = nil
          self.generateNextIfNeeded()
        }
      }
      return
    }
  }

  nonisolated private static func isVideo(_ url: URL) -> Bool {
    videoExtensions.contains(url.pathExtension.lowercased())
  }
}
