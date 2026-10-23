import Foundation

@main
struct CelfNetDecoderHarness {
  static func main() throws {
    let input = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    let output = input.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let objects = CelfNetObjectDecoder.decode(output)
    let labels = CelfNetObjectDecoder.inputLabels(objects)
    let prior = CelfNetObjectDecoder.prior(labels)
    let report: [String: Any] = [
      "objects": objects.map { ["pixels": $0.pixels, "score": $0.score] as [String: Any] },
      "zeroChannel": prior[..<(960 * 544)].allSatisfy { $0 == 0 },
      "paddingEmpty": labels[..<(960 * 2)].allSatisfy { $0 == 0 }
        && labels[(960 * 542)...].allSatisfy { $0 == 0 },
    ]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    if CommandLine.arguments.count > 2 {
      try prior.withUnsafeBytes { try Data($0).write(to: URL(fileURLWithPath: CommandLine.arguments[2])) }
    }
  }
}
