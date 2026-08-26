import CryptoKit
import Foundation

@available(macOS 27.0, *)
@main
struct TenErosMaxH3FullShapeProbe {
  static func main() async throws {
    guard (3...8).contains(CommandLine.arguments.count),
      let tokens = Int(CommandLine.arguments[2]), tokens > 0
    else {
      throw H3NativeError.invalidArguments(
        "usage: TenErosMaxH3FullShapeProbe <model.aimodel[c]> <tokens> "
          + "[bfloat16|float16] [gpu|ane|cpu] [function] "
          + "[graph-salt-input] [graph-salt-width]"
      )
    }
    let scalarType = H3ScalarType(
      rawValue: CommandLine.arguments.count >= 4
        ? CommandLine.arguments[3] : "bfloat16"
    )
    guard let scalarType else {
      throw H3NativeError.invalidArguments("unsupported scalar type")
    }
    let preferredCompute = CommandLine.arguments.count >= 5
      ? CommandLine.arguments[4] : "gpu"
    let functionName = CommandLine.arguments.count >= 6
      ? CommandLine.arguments[5] : "main"
    let graphSaltInput = CommandLine.arguments.count >= 7
      ? CommandLine.arguments[6] : nil
    let graphSaltWidth = CommandLine.arguments.count >= 8
      ? Int(CommandLine.arguments[7]) : nil

    func zeros(_ shape: [Int]) throws -> H3Tensor {
      try H3Tensor(
        shape: shape,
        scalarType: scalarType,
        bytes: Data(count: shape.reduce(1, *) * scalarType.byteCount)
      )
    }

    var inputBindings = [
      "hiddenStates": "hidden_states",
      "timestepCoordinates": "timestep_coordinates",
      "modulationWeights": "modulation_weights",
      "ropeCosine": "rope_cosine",
      "ropeSine": "rope_sine",
    ]
    if let graphSaltInput { inputBindings["graphSalt"] = graphSaltInput }
    let manifest = H3StageManifest(
      backend: .coreAI,
      asset: CommandLine.arguments[1],
      function: functionName,
      computeUnits: preferredCompute,
      inputs: inputBindings,
      outputs: ["hiddenStatesOut": "hidden_states_out"],
      inputConstraints: nil,
      outputConstraints: nil
    )
    let runner = try await H3StageRunner(
      name: "10eros.dit.fullShape",
      manifest: manifest,
      baseDirectory: URL(fileURLWithPath: "/")
    )
    var inputs = [
      "hiddenStates": try zeros([tokens, 5_376]),
      "timestepCoordinates": try zeros([4, 8]),
      "modulationWeights": try zeros([tokens, 12]),
      "ropeCosine": try zeros([tokens, 48]),
      "ropeSine": try zeros([tokens, 48]),
    ]
    if let graphSaltInput {
      guard let graphSaltWidth, graphSaltWidth > 0 else {
        throw H3NativeError.invalidArguments(
          "graph-salt-width is required with graph-salt-input"
        )
      }
      _ = graphSaltInput
      inputs["graphSalt"] = try zeros([graphSaltWidth])
    }
    let result = try await runner.predict(inputs)
    guard let output = result["hiddenStatesOut"] else {
      throw H3NativeError.missingTensor("hiddenStatesOut")
    }
    let digest = SHA256.hash(data: output.bytes).map {
      String(format: "%02x", $0)
    }.joined()
    print(
      "shape=\(output.shape) scalarType=\(output.scalarType.rawValue) "
        + "sha256=\(digest)"
    )
  }
}
