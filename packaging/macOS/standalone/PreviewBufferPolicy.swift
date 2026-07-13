struct PreviewBufferPolicy {
  static let shortenedRebufferSeconds = 4.0

  static func requiredSeconds(
    selectedBufferLimit: Double,
    generationHasStarted: Bool,
    shortenRebuffer: Bool
  ) -> Double {
    let selected = max(0, selectedBufferLimit)
    guard generationHasStarted, shortenRebuffer else { return selected }
    return min(selected, shortenedRebufferSeconds)
  }

  static func canStart(
    bufferedSeconds: Double,
    selectedBufferLimit: Double,
    generationHasStarted: Bool,
    shortenRebuffer: Bool,
    endOfFile: Bool,
    hasQueuedSegments: Bool
  ) -> Bool {
    guard hasQueuedSegments else { return false }
    if endOfFile { return true }
    let required = requiredSeconds(
      selectedBufferLimit: selectedBufferLimit,
      generationHasStarted: generationHasStarted,
      shortenRebuffer: shortenRebuffer
    )
    return bufferedSeconds + 0.001 >= required
  }
}
