import Foundation

private enum HarnessFailure: Error, CustomStringConvertible {
  case assertion(String)

  var description: String {
    switch self {
    case .assertion(let message): message
    }
  }
}

private func require(
  _ condition: @autoclosure () -> Bool,
  _ message: String
) throws {
  guard condition() else { throw HarnessFailure.assertion(message) }
}

/// A deterministic model of the parts of RealtimePlayerController that are
/// allowed to mutate when the auxiliary HLS audio/source player disappears.
/// The real controller is tied to AVFoundation and the Core AI worker, so this
/// small harness deliberately drives the callback ordering independently of
/// wall-clock media decoding.
@MainActor
private final class HLSAudioFallbackStateMachine {
  enum State: Equatable {
    case loading
    case buffering
    case playing
    case failed
  }

  var generation = 41
  var state: State = .buffering
  var shouldPlay = true
  var generationHasStarted = false
  var hlsSourceAvailable = true
  var currentSourceItem = 7
  var sourceObserversInstalled = true
  var notificationObserversInstalled = true
  var sourceTimeObserverInstalled = true
  var sourcePaused = false
  var sourceMuted = false
  var proxyRunning = true
  var producerRunning = true
  var restoredQueue: [Int] = []
  var acknowledgedThrough = -1
  var fallbackActive = false
  var fallbackTransitions = 0
  var advisoryEmptyAudioResults = 0
  var ignoredCallbacks = 0
  var nextSequence = 0

  func enqueueRestored(_ sequence: Int) {
    guard sequence == nextSequence else { return }
    nextSequence += 1
    restoredQueue.append(sequence)
    resumeIfBuffered()
  }

  func finishRestored(_ sequence: Int) {
    guard restoredQueue.first == sequence else { return }
    restoredQueue.removeFirst()
    acknowledgedThrough = sequence
    if restoredQueue.isEmpty, state == .playing {
      state = .buffering
    }
  }

  func legacyAudioPreflightCompleted(
    item: Int,
    callbackGeneration: Int,
    audioTrackCount: Int
  ) {
    guard callbackGeneration == generation,
      currentSourceItem == item,
      !fallbackActive
    else {
      ignoredCallbacks += 1
      return
    }
    // HLS can expose an empty AVAsset track list while AVFoundation is still
    // resolving a muxed or alternate audio rendition. This observation is
    // advisory; only an actual source playback failure may trigger fallback.
    if audioTrackCount == 0 { advisoryEmptyAudioResults += 1 }
  }

  func sourceKVOCallback(item: Int, callbackGeneration: Int) {
    guard callbackGeneration == generation,
      hlsSourceAvailable,
      currentSourceItem == item,
      state != .failed,
      !fallbackActive
    else {
      ignoredCallbacks += 1
      return
    }
    degrade(callbackGeneration: callbackGeneration)
  }

  func failedToEndNotification(item: Int, callbackGeneration: Int) {
    guard callbackGeneration == generation,
      hlsSourceAvailable,
      currentSourceItem == item
    else {
      ignoredCallbacks += 1
      return
    }
    degrade(callbackGeneration: callbackGeneration)
  }

  func playbackStalledNotification(item: Int, callbackGeneration: Int) {
    guard callbackGeneration == generation,
      hlsSourceAvailable,
      currentSourceItem == item,
      state != .failed,
      !fallbackActive
    else {
      ignoredCallbacks += 1
      return
    }
    state = .buffering
  }

  func proxyCallback(callbackGeneration: Int) {
    guard callbackGeneration == generation else {
      ignoredCallbacks += 1
      return
    }
    degrade(callbackGeneration: callbackGeneration)
  }

  func staleSeekCompletion(item: Int, callbackGeneration: Int) {
    guard callbackGeneration == generation,
      hlsSourceAvailable,
      currentSourceItem == item,
      !fallbackActive
    else {
      ignoredCallbacks += 1
      return
    }
    state = .playing
  }

  func replaceGeneration() {
    generation += 1
    currentSourceItem = 99
    hlsSourceAvailable = true
    fallbackActive = false
    sourceObserversInstalled = true
    notificationObserversInstalled = true
    sourceTimeObserverInstalled = true
    sourcePaused = false
    sourceMuted = false
    proxyRunning = true
    state = .buffering
  }

  private func degrade(callbackGeneration: Int) {
    guard callbackGeneration == generation,
      hlsSourceAvailable,
      state != .failed,
      !fallbackActive
    else {
      ignoredCallbacks += 1
      return
    }

    // This mirrors the production ordering: publish the idempotence barrier
    // first, then detach every source-side callback. The producer and restored
    // queue are intentionally not touched.
    fallbackActive = true
    fallbackTransitions += 1
    sourceObserversInstalled = false
    notificationObserversInstalled = false
    sourceTimeObserverInstalled = false
    sourcePaused = true
    sourceMuted = true

    if shouldPlay, generationHasStarted, !restoredQueue.isEmpty {
      state = .playing
    } else if shouldPlay {
      state = .buffering
      resumeIfBuffered()
    }
  }

  private func resumeIfBuffered() {
    guard shouldPlay, fallbackActive, restoredQueue.count >= 3 else { return }
    generationHasStarted = true
    state = .playing
  }
}

private enum Callback: CaseIterable {
  case legacyAudioPreflightResult
  case sourceKVO
  case playbackStalled
  case failedToEnd
  case proxy
}

private func permutations<T>(_ values: [T]) -> [[T]] {
  guard let first = values.first else { return [[]] }
  return permutations(Array(values.dropFirst())).flatMap { suffix in
    (0...suffix.count).map { index in
      var result = suffix
      result.insert(first, at: index)
      return result
    }
  }
}

@main
private struct HLSAudioFallbackStateHarness {
  @MainActor
  static func main() async throws {
    var permutationCount = 0
    for ordering in permutations(Callback.allCases) {
      let machine = HLSAudioFallbackStateMachine()
      machine.enqueueRestored(0)
      machine.enqueueRestored(1)
      machine.enqueueRestored(2)
      machine.generationHasStarted = true
      machine.state = .playing

      for callback in ordering {
        switch callback {
        case .legacyAudioPreflightResult:
          machine.legacyAudioPreflightCompleted(
            item: 7,
            callbackGeneration: 41,
            audioTrackCount: 0
          )
        case .sourceKVO:
          machine.sourceKVOCallback(item: 7, callbackGeneration: 41)
        case .playbackStalled:
          machine.playbackStalledNotification(item: 7, callbackGeneration: 41)
        case .failedToEnd:
          machine.failedToEndNotification(item: 7, callbackGeneration: 41)
        case .proxy:
          machine.proxyCallback(callbackGeneration: 41)
        }
      }
      machine.staleSeekCompletion(item: 7, callbackGeneration: 41)

      try require(machine.fallbackActive, "fallback was not activated")
      try require(machine.fallbackTransitions == 1, "fallback was not idempotent")
      try require(machine.advisoryEmptyAudioResults <= 1, "audio validation repeated")
      try require(machine.state == .playing, "restored playback did not continue")
      try require(machine.producerRunning, "fallback stopped the producer")
      try require(machine.restoredQueue == [0, 1, 2], "fallback cleared restored items")
      try require(!machine.sourceObserversInstalled, "source KVO remained installed")
      try require(!machine.notificationObserversInstalled, "source notification remained installed")
      try require(!machine.sourceTimeObserverInstalled, "source clock remained installed")
      try require(machine.sourcePaused, "source player was not paused")
      try require(machine.sourceMuted, "source player was not muted")
      try require(machine.currentSourceItem == 7, "fallback destroyed the source item")
      try require(machine.proxyRunning, "fallback stopped the source proxy")

      machine.finishRestored(0)
      try require(machine.acknowledgedThrough == 0, "restored output was not acknowledged")
      try require(machine.restoredQueue == [1, 2], "restored queue did not advance")
      permutationCount += 1
    }

    // An empty AVAsset audio track list is not conclusive for HLS and must not
    // tear down the source item or loopback proxy.
    let advisory = HLSAudioFallbackStateMachine()
    advisory.legacyAudioPreflightCompleted(
      item: 7,
      callbackGeneration: 41,
      audioTrackCount: 0
    )
    try require(!advisory.fallbackActive, "empty track metadata triggered fallback")
    try require(advisory.advisoryEmptyAudioResults == 1, "empty metadata was not logged")
    try require(advisory.currentSourceItem == 7, "advisory result destroyed source item")
    try require(advisory.proxyRunning, "advisory result stopped proxy")

    // A real source failure can arrive before any restored item exists. Later
    // producer output must still fill the queue and start from its own clock.
    let early = HLSAudioFallbackStateMachine()
    early.sourceKVOCallback(item: 7, callbackGeneration: 41)
    early.failedToEndNotification(item: 7, callbackGeneration: 41)
    early.proxyCallback(callbackGeneration: 41)
    try require(early.state == .buffering, "empty restored queue should buffer")
    try require(early.producerRunning, "early fallback stopped the producer")
    try require(early.currentSourceItem == 7, "early fallback destroyed source item")
    try require(early.proxyRunning, "early fallback stopped proxy")
    early.enqueueRestored(0)
    early.enqueueRestored(1)
    early.enqueueRestored(2)
    try require(early.state == .playing, "late restored output did not start")
    try require(early.generationHasStarted, "restored clock was not activated")

    // Replacement/seek must form a hard generation boundary for every delayed
    // callback retained by AVFoundation or the old loopback proxy.
    let replaced = HLSAudioFallbackStateMachine()
    replaced.replaceGeneration()
    replaced.legacyAudioPreflightCompleted(
      item: 7,
      callbackGeneration: 41,
      audioTrackCount: 0
    )
    replaced.sourceKVOCallback(item: 7, callbackGeneration: 41)
    replaced.failedToEndNotification(item: 7, callbackGeneration: 41)
    replaced.proxyCallback(callbackGeneration: 41)
    replaced.staleSeekCompletion(item: 7, callbackGeneration: 41)
    try require(!replaced.fallbackActive, "old generation degraded the replacement")
    try require(replaced.state == .buffering, "old callback changed replacement state")
    try require(replaced.proxyRunning, "old callback stopped the replacement proxy")

    print("ok fallback permutations=\(permutationCount)")
  }
}
