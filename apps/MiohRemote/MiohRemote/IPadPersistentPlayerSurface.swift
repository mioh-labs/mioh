import AVFoundation
import CoreMedia
import QuartzCore
import SwiftUI
import UIKit
import VideoToolbox

/// Keeps one restored-video display layer alive while an AVQueuePlayer advances
/// between independently encoded segment items.
struct IPadPersistentPlayerSurface: UIViewRepresentable {
  typealias VideoOutputProvider =
    (AVPlayerItem) -> AVPlayerItemVideoOutput?
  typealias FrameDisplayedHandler =
    (AVPlayerItem, CVPixelBuffer) -> Void

  let player: AVPlayer
  let videoOutputProvider: VideoOutputProvider
  let initialPixelBuffer: CVPixelBuffer?
  let onFrameDisplayed: FrameDisplayedHandler

  func makeUIView(context: Context) -> IPadPersistentPlayerSurfaceView {
    let view = IPadPersistentPlayerSurfaceView()
    view.setPlayer(
      player,
      videoOutputProvider: videoOutputProvider,
      initialPixelBuffer: initialPixelBuffer,
      onFrameDisplayed: onFrameDisplayed
    )
    return view
  }

  func updateUIView(
    _ uiView: IPadPersistentPlayerSurfaceView,
    context: Context
  ) {
    uiView.setPlayer(
      player,
      videoOutputProvider: videoOutputProvider,
      initialPixelBuffer: initialPixelBuffer,
      onFrameDisplayed: onFrameDisplayed
    )
  }

  static func dismantleUIView(
    _ uiView: IPadPersistentPlayerSurfaceView,
    coordinator: Void
  ) {
    uiView.disconnect()
  }
}

final class IPadPersistentPlayerSurfaceView: UIView {
  private static let boundarySnapshotLeadSeconds = 0.20

  override class var layerClass: AnyClass {
    AVSampleBufferDisplayLayer.self
  }

  private final class DisplayLinkTarget: NSObject {
    weak var surface: IPadPersistentPlayerSurfaceView?

    @objc func displayLinkDidFire(_ displayLink: CADisplayLink) {
      surface?.displayLinkDidFire(displayLink)
    }
  }

  private var sampleBufferDisplayLayer: AVSampleBufferDisplayLayer {
    layer as! AVSampleBufferDisplayLayer
  }

  private let boundarySnapshotLayer = CALayer()
  private let displayLinkTarget = DisplayLinkTarget()
  private var displayLink: CADisplayLink?
  private weak var player: AVPlayer?
  private weak var displayedItem: AVPlayerItem?
  private var currentItemObservation: NSKeyValueObservation?
  private var videoOutput: AVPlayerItemVideoOutput?
  private var videoOutputProvider: IPadPersistentPlayerSurface.VideoOutputProvider?
  private var onFrameDisplayed: IPadPersistentPlayerSurface.FrameDisplayedHandler?
  private var hasSubmittedFrame = false
  private var lastSubmittedPixelBuffer: CVPixelBuffer?
  private var lastSubmittedItemIdentifier: ObjectIdentifier?
  private var boundarySnapshotItemIdentifier: ObjectIdentifier?
  private var boundaryReleaseItemIdentifier: ObjectIdentifier?
  private var boundaryReleaseDisplayTick: UInt64?
  private var displayTick: UInt64 = 0

  override init(frame: CGRect) {
    super.init(frame: frame)
    configureDisplayLayer()
    displayLinkTarget.surface = self
  }

  required init?(coder: NSCoder) {
    super.init(coder: coder)
    configureDisplayLayer()
    displayLinkTarget.surface = self
  }

  deinit {
    currentItemObservation?.invalidate()
    displayLink?.invalidate()
  }

  override func didMoveToWindow() {
    super.didMoveToWindow()
    if window == nil {
      displayLink?.isPaused = true
    } else {
      installDisplayLinkIfNeeded()
      displayLink?.isPaused = false
    }
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    withoutLayerAnimations {
      boundarySnapshotLayer.frame = bounds
    }
  }

  func setPlayer(
    _ player: AVPlayer,
    videoOutputProvider:
      @escaping IPadPersistentPlayerSurface.VideoOutputProvider,
    initialPixelBuffer: CVPixelBuffer?,
    onFrameDisplayed:
      @escaping IPadPersistentPlayerSurface.FrameDisplayedHandler
  ) {
    self.videoOutputProvider = videoOutputProvider
    self.onFrameDisplayed = onFrameDisplayed
    seedDisplayLayerIfNeeded(with: initialPixelBuffer)
    guard self.player !== player else {
      refreshVideoOutputIfNeeded()
      return
    }
    currentItemObservation?.invalidate()
    self.player = player
    currentItemObservation = player.observe(
      \.currentItem,
      options: [.initial, .new]
    ) { [weak self] player, _ in
      DispatchQueue.main.async {
        self?.selectVideoOutput(for: player.currentItem)
      }
    }
  }

  func disconnect() {
    currentItemObservation?.invalidate()
    currentItemObservation = nil
    player = nil
    displayedItem = nil
    videoOutput = nil
    videoOutputProvider = nil
    onFrameDisplayed = nil
    displayLink?.invalidate()
    displayLink = nil
    // Do not flush the display layer. Retaining its image through teardown also
    // prevents a transient black surface during SwiftUI hierarchy changes.
  }

  private func configureDisplayLayer() {
    isOpaque = false
    backgroundColor = .clear
    clipsToBounds = true
    sampleBufferDisplayLayer.isOpaque = false
    sampleBufferDisplayLayer.backgroundColor = UIColor.clear.cgColor
    sampleBufferDisplayLayer.controlTimebase = nil
    sampleBufferDisplayLayer.videoGravity = .resizeAspect
    boundarySnapshotLayer.backgroundColor = UIColor.clear.cgColor
    boundarySnapshotLayer.contentsGravity = .resizeAspect
    boundarySnapshotLayer.contentsScale = UIScreen.main.scale
    boundarySnapshotLayer.isHidden = true
    boundarySnapshotLayer.actions = [
      "bounds": NSNull(),
      "contents": NSNull(),
      "hidden": NSNull(),
      "opacity": NSNull(),
      "position": NSNull(),
    ]
    layer.addSublayer(boundarySnapshotLayer)
  }

  private func installDisplayLinkIfNeeded() {
    guard displayLink == nil else { return }
    let displayLink = CADisplayLink(
      target: displayLinkTarget,
      selector: #selector(DisplayLinkTarget.displayLinkDidFire(_:))
    )
    displayLink.preferredFramesPerSecond = 60
    displayLink.add(to: .main, forMode: .common)
    self.displayLink = displayLink
  }

  private func selectVideoOutput(for item: AVPlayerItem?) {
    guard displayedItem !== item else {
      refreshVideoOutputIfNeeded()
      return
    }
    let previousItemIdentifier = displayedItem.map(ObjectIdentifier.init)
    if let previousItemIdentifier,
      boundarySnapshotItemIdentifier == nil,
      lastSubmittedItemIdentifier == previousItemIdentifier,
      let lastSubmittedPixelBuffer
    {
      showBoundarySnapshot(
        pixelBuffer: lastSubmittedPixelBuffer,
        itemIdentifier: previousItemIdentifier
      )
    }
    boundaryReleaseItemIdentifier = nil
    boundaryReleaseDisplayTick = nil
    displayedItem = item
    videoOutput = item.flatMap { videoOutputProvider?($0) }
    videoOutput?.requestNotificationOfMediaDataChange(
      withAdvanceInterval: 0.03
    )
    // Intentionally do not flush sampleBufferDisplayLayer here. The last
    // decoded image remains visible until the new item supplies its first one.
  }

  private func refreshVideoOutputIfNeeded() {
    guard videoOutput == nil, let displayedItem else { return }
    videoOutput = videoOutputProvider?(displayedItem)
    videoOutput?.requestNotificationOfMediaDataChange(
      withAdvanceInterval: 0.03
    )
  }

  private func displayLinkDidFire(_ displayLink: CADisplayLink) {
    displayTick &+= 1
    releaseBoundarySnapshotIfReady()
    refreshVideoOutputIfNeeded()
    recoverDisplayLayerIfNeeded(at: displayLink.timestamp)
    guard let videoOutput,
      sampleBufferDisplayLayer.isReadyForMoreMediaData
    else { return }
    let hostTime = displayLink.timestamp + displayLink.duration
    let itemTime = videoOutput.itemTime(forHostTime: hostTime)
    guard itemTime.isNumeric,
      videoOutput.hasNewPixelBuffer(forItemTime: itemTime)
    else { return }

    var displayTime = CMTime.invalid
    guard
      let pixelBuffer = videoOutput.copyPixelBuffer(
        forItemTime: itemTime,
        itemTimeForDisplay: &displayTime
      )
    else { return }
    if enqueue(pixelBuffer: pixelBuffer, hostTime: hostTime),
      let displayedItem
    {
      updateBoundaryGuard(
        pixelBuffer: pixelBuffer,
        item: displayedItem,
        itemTime: displayTime.isNumeric ? displayTime : itemTime
      )
      onFrameDisplayed?(displayedItem, pixelBuffer)
    }
  }

  private func updateBoundaryGuard(
    pixelBuffer: CVPixelBuffer,
    item: AVPlayerItem,
    itemTime: CMTime
  ) {
    let itemIdentifier = ObjectIdentifier(item)
    lastSubmittedItemIdentifier = itemIdentifier

    if let snapshotItemIdentifier = boundarySnapshotItemIdentifier,
      snapshotItemIdentifier != itemIdentifier
    {
      if boundaryReleaseItemIdentifier != itemIdentifier {
        boundaryReleaseItemIdentifier = itemIdentifier
        // The sample-buffer layer commits asynchronously. Keep the old image
        // above it for two complete display-link turns after the first frame
        // from the next queue item has been enqueued.
        boundaryReleaseDisplayTick = displayTick &+ 2
      }
      return
    }

    let duration = item.duration
    guard duration.isNumeric, itemTime.isNumeric,
      duration.seconds.isFinite, itemTime.seconds.isFinite
    else { return }
    let remainingSeconds = duration.seconds - itemTime.seconds
    guard remainingSeconds >= -0.05,
      remainingSeconds <= Self.boundarySnapshotLeadSeconds
    else { return }

    // Conversion is deliberately limited to the final fraction of each item.
    // Updating the visible guard with every final frame avoids a freeze while
    // guaranteeing that the display layer can never expose its clear backing
    // store during an AVQueuePlayer item transition.
    showBoundarySnapshot(
      pixelBuffer: pixelBuffer,
      itemIdentifier: itemIdentifier
    )
  }

  private func showBoundarySnapshot(
    pixelBuffer: CVPixelBuffer,
    itemIdentifier: ObjectIdentifier
  ) {
    var image: CGImage?
    guard
      VTCreateCGImageFromCVPixelBuffer(
        pixelBuffer,
        options: nil,
        imageOut: &image
      ) == noErr,
      let image
    else { return }

    withoutLayerAnimations {
      boundarySnapshotLayer.contents = image
      boundarySnapshotLayer.isHidden = false
    }
    boundarySnapshotItemIdentifier = itemIdentifier
    boundaryReleaseItemIdentifier = nil
    boundaryReleaseDisplayTick = nil
  }

  private func releaseBoundarySnapshotIfReady() {
    guard let releaseDisplayTick = boundaryReleaseDisplayTick,
      displayTick >= releaseDisplayTick,
      let releaseItemIdentifier = boundaryReleaseItemIdentifier,
      displayedItem.map(ObjectIdentifier.init) == releaseItemIdentifier
    else { return }

    withoutLayerAnimations {
      boundarySnapshotLayer.isHidden = true
      boundarySnapshotLayer.contents = nil
    }
    boundarySnapshotItemIdentifier = nil
    boundaryReleaseItemIdentifier = nil
    boundaryReleaseDisplayTick = nil
  }

  private func withoutLayerAnimations(_ changes: () -> Void) {
    CATransaction.begin()
    CATransaction.setDisableActions(true)
    changes()
    CATransaction.commit()
  }

  private func enqueue(
    pixelBuffer: CVPixelBuffer,
    hostTime: CFTimeInterval
  ) -> Bool {
    var formatDescription: CMVideoFormatDescription?
    guard
      CMVideoFormatDescriptionCreateForImageBuffer(
        allocator: kCFAllocatorDefault,
        imageBuffer: pixelBuffer,
        formatDescriptionOut: &formatDescription
      ) == noErr,
      let formatDescription
    else { return false }

    var timing = CMSampleTimingInfo(
      duration: .invalid,
      presentationTimeStamp: CMTime(
        seconds: hostTime,
        preferredTimescale: 1_000_000
      ),
      decodeTimeStamp: .invalid
    )
    var sampleBuffer: CMSampleBuffer?
    guard
      CMSampleBufferCreateReadyWithImageBuffer(
        allocator: kCFAllocatorDefault,
        imageBuffer: pixelBuffer,
        formatDescription: formatDescription,
        sampleTiming: &timing,
        sampleBufferOut: &sampleBuffer
      ) == noErr,
      let sampleBuffer
    else { return false }

    markForImmediateDisplay(sampleBuffer)
    sampleBufferDisplayLayer.enqueue(sampleBuffer)
    hasSubmittedFrame = true
    lastSubmittedPixelBuffer = pixelBuffer
    return true
  }

  private func recoverDisplayLayerIfNeeded(at hostTime: CFTimeInterval) {
    guard
      sampleBufferDisplayLayer.status == .failed
        || sampleBufferDisplayLayer.requiresFlushToResumeDecoding
    else { return }
    sampleBufferDisplayLayer.flush()
    hasSubmittedFrame = false
    guard sampleBufferDisplayLayer.isReadyForMoreMediaData,
      let lastSubmittedPixelBuffer
    else { return }
    _ = enqueue(
      pixelBuffer: lastSubmittedPixelBuffer,
      hostTime: hostTime
    )
  }

  private func seedDisplayLayerIfNeeded(with pixelBuffer: CVPixelBuffer?) {
    guard !hasSubmittedFrame, let pixelBuffer else { return }
    _ = enqueue(
      pixelBuffer: pixelBuffer,
      hostTime: CACurrentMediaTime()
    )
  }

  private func markForImmediateDisplay(_ sampleBuffer: CMSampleBuffer) {
    guard
      let attachments = CMSampleBufferGetSampleAttachmentsArray(
        sampleBuffer,
        createIfNecessary: true
      ),
      CFArrayGetCount(attachments) > 0
    else { return }
    let dictionary = unsafeBitCast(
      CFArrayGetValueAtIndex(attachments, 0),
      to: CFMutableDictionary.self
    )
    CFDictionarySetValue(
      dictionary,
      Unmanaged.passUnretained(
        kCMSampleAttachmentKey_DisplayImmediately
      ).toOpaque(),
      Unmanaged.passUnretained(kCFBooleanTrue).toOpaque()
    )
  }
}
