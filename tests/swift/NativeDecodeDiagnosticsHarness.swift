import Foundation

@main
struct NativeDecodeDiagnosticsHarness {
  static func main() throws {
    let leaf = NSError(domain: NSOSStatusErrorDomain, code: -12909, userInfo: [
      NSLocalizedDescriptionKey: "private-title /Volumes/private/example.mp4",
      NSFilePathErrorKey: "/Volumes/private/example.mp4",
    ])
    let outer = NSError(domain: "AVFoundationErrorDomain", code: -11821, userInfo: [
      NSLocalizedDescriptionKey: "Cannot Decode private-title.mp4",
      NSLocalizedFailureReasonErrorKey: "https://private.invalid/example.mp4?token=secret",
      NSUnderlyingErrorKey: leaf,
    ])
    let readerError = NSError(domain: NSOSStatusErrorDomain, code: -12911)
    let sidecarError = NSError(domain: NSCocoaErrorDomain, code: 256)
    var deep = NSError(domain: NSOSStatusErrorDomain, code: 0)
    for index in 1...30 {
      deep = NSError(domain: NSOSStatusErrorDomain, code: index,
                     userInfo: [NSUnderlyingErrorKey: deep])
    }
    let multiple = NSError(domain: "AVFoundationErrorDomain", code: -11800,
      userInfo: [NSMultipleUnderlyingErrorsKey: [leaf, readerError]])
    let wide = NSError(domain: NSCocoaErrorDomain, code: 1,
      userInfo: [NSMultipleUnderlyingErrorsKey: (0..<30).map {
        NSError(domain: NSOSStatusErrorDomain, code: $0)
      }])
    let cases: [String: NativePreviewError] = [
      "nested": NativeDecodeDiagnostics.failure(outer, stage: "decoded.next",
        decodedFrames: 123, lastPTS: 4_100_000_000, readerStatus: 3,
        readerError: readerError, sidecarStatus: 2, sidecarError: sidecarError),
      "start": NativeDecodeDiagnostics.failure(outer, stage: "reader.start"),
      "deep": NativeDecodeDiagnostics.failure(deep, stage: "decoded.next"),
      "multiple": NativeDecodeDiagnostics.failure(multiple, stage: "decoded.next"),
      "wide": NativeDecodeDiagnostics.failure(wide, stage: "decoded.next"),
      "duplicate": NativeDecodeDiagnostics.failure(outer, stage: "decoded.next",
        readerError: outer, sidecarError: leaf),
      "native": NativeDecodeDiagnostics.failure(
        NativePreviewError.reader("decoded H.264 sample has no image buffer"),
        stage: "decoded.validation"),
      "unsafe_domain": NativeDecodeDiagnostics.failure(
        NSError(domain: "file:///Volumes/private/example.mp4", code: 1), stage: "reader.create"),
    ]
    var result: [String: Any] = [:]
    for (name, error) in cases {
      guard case .reader(let json) = error else { fatalError("expected reader error") }
      result[name] = try JSONSerialization.jsonObject(with: Data(json.utf8))
    }
    let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
  }
}
