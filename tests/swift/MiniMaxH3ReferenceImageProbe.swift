import Foundation

@main
struct MiniMaxH3ReferenceImageProbe {
  static func main() throws {
    guard (2...3).contains(CommandLine.arguments.count) else {
      throw H3NativeError.invalidArguments(
        "usage: MiniMaxH3ReferenceImageProbe <image> [output.bin]"
      )
    }
    let width = 864
    let height = 480
    let tensor = try H3NativeMedia.decodeReferenceImage(
      url: URL(fileURLWithPath: CommandLine.arguments[1]),
      width: width,
      height: height
    )
    let values = try tensor.floatValues()
    var edgeSum: Double = 0
    var centerSum: Double = 0
    var edgeCount = 0
    var centerCount = 0
    for channel in 0..<3 {
      for y in 0..<height {
        for x in 0..<width {
          let value = Double(values[(channel * height + y) * width + x])
          if x < 64 || x >= width - 64 {
            edgeSum += value
            edgeCount += 1
          }
          if (width / 2 - 32)..<(width / 2 + 32) ~= x {
            centerSum += value
            centerCount += 1
          }
        }
      }
    }
    let edgeMean = edgeSum / Double(edgeCount)
    let centerMean = centerSum / Double(centerCount)
    guard edgeMean > 0.02 else {
      fatalError("portrait reference still has black side pillars: \(edgeMean)")
    }
    if CommandLine.arguments.count == 3 {
      try tensor.bytes.write(
        to: URL(fileURLWithPath: CommandLine.arguments[2]),
        options: .atomic
      )
    }
    print(
      "shape=\(tensor.shape) edge_mean=\(edgeMean) center_mean=\(centerMean)"
    )
  }
}
