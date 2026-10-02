// Compiled alongside the actual file-private production decoder definitions.
private let outputLock = NSLock()
private func emit(_ payload: [String: Any]) {
  outputLock.withLock {
    let data = try! JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
    fflush(stdout)
  }
}

@main
private enum DecoderHarness {
  static func main() async {
    let args = CommandLine.arguments
    let backend = try! NativeDecoderBackend.resolve(args[1])
    let mode = args[5]
    let metrics = NativePipelineDiagnostics(generation: 9, backend: backend.rawValue)
    metrics.start()
    let ring = PixelBufferRing(capacity: 3, diagnostics: metrics)
    var decoder: ContinuousVideoDecoder?
    do {
      let source = try await ContinuousVideoDecoder(
        input: URL(fileURLWithPath: args[2]),
        startNanoseconds: Int64(args[3])!,
        endNanoseconds: args[4] == "none" ? nil : Int64(args[4])!,
        backend: backend, ffmpeg: URL(fileURLWithPath: args[6]),
        diagnostics: metrics, ring: ring)
      decoder = source
      try source.start()
      if mode == "cancel" || mode == "idle" {
        try await Task.sleep(for: .milliseconds(mode == "idle" ? 5300 : 300))
        await source.stop()
        emit(["kind": "result", "cancelled": true])
      } else {
        let result = try await Task.detached {
          var timestamps: [Int64] = []
          var sizes: [[Int]] = []
          while let frame = try ring.pop() {
            timestamps.append(frame.ptsNanoseconds)
            sizes.append([CVPixelBufferGetWidth(frame.pixelBuffer),
                          CVPixelBufferGetHeight(frame.pixelBuffer)])
          }
          return (timestamps, sizes)
        }.value
        await source.stop()
        emit(["kind": "result", "pts": result.0, "sizes": result.1])
      }
    } catch {
      await decoder?.stop()
      emit(["kind": "result", "error": error.localizedDescription])
    }
    metrics.stop()
  }
}
