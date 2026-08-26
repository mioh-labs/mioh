import Foundation

@available(macOS 27.0, *)
@main
struct TenErosMaxH3DiTBlockProbe {
  static func main() async throws {
    guard CommandLine.arguments.count >= 4 else {
      throw H3NativeError.invalidArguments(
        "usage: TenErosMaxH3DiTBlockProbe <model.aimodelc> [...] "
          + "<reference-directory> <actual.f32>"
      )
    }
    let modelPaths = Array(CommandLine.arguments[1..<(CommandLine.arguments.count - 2)])
    let reference = URL(
      fileURLWithPath: CommandLine.arguments[CommandLine.arguments.count - 2]
    )
    let outputPath = CommandLine.arguments.last!
    let metadata = try JSONSerialization.jsonObject(
      with: Data(contentsOf: reference.appendingPathComponent("metadata.json"))
    ) as! [String: Any]
    let shapes = metadata["inputShapes"] as! [[Int]]
    let outputShape = metadata["outputShape"] as! [Int]
    let scalarType = H3ScalarType(
      rawValue: metadata["scalarType"] as? String ?? "bfloat16"
    ) ?? .bfloat16
    let semantics = [
      "hiddenStates", "timestepCoordinates", "modulationWeights",
      "ropeCosine", "ropeSine",
    ]
    let modelNames = [
      "hidden_states", "timestep_coordinates", "modulation_weights",
      "rope_cosine", "rope_sine",
    ]
    let files = [
      "hidden_states.f32", "timestep_coordinates.f32",
      "modulation_weights.f32", "rope_cosine.f32", "rope_sine.f32",
    ]
    var constraints: [String: H3TensorConstraint] = [:]
    var inputs: [String: H3Tensor] = [:]
    for index in semantics.indices {
      let data = try Data(contentsOf: reference.appendingPathComponent(files[index]))
      let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
      let tensor = try H3Tensor(float32: values, shape: shapes[index])
        .converted(to: scalarType)
      constraints[semantics[index]] = H3TensorConstraint(
        scalarType: scalarType,
        shape: shapes[index]
      )
      inputs[semantics[index]] = tensor
    }
    var output: H3Tensor?
    for (index, modelPath) in modelPaths.enumerated() {
      let manifest = H3StageManifest(
        backend: .coreAI,
        asset: modelPath,
        function: "main",
        computeUnits: ProcessInfo.processInfo.environment[
          "H3_PROBE_COMPUTE"
        ] ?? "gpu",
        inputs: Dictionary(uniqueKeysWithValues: zip(semantics, modelNames)),
        outputs: ["hiddenStatesOut": "hidden_states_out"],
        inputConstraints: constraints,
        outputConstraints: [
          "hiddenStatesOut": H3TensorConstraint(
            scalarType: scalarType,
            shape: outputShape
          )
        ]
      )
      let runner = try await H3StageRunner(
        name: "10eros.dit.block.\(index)",
        manifest: manifest,
        baseDirectory: URL(fileURLWithPath: "/")
      )
      if let output { inputs["hiddenStates"] = output }
      output = try await runner.predict(inputs)["hiddenStatesOut"]
    }
    guard let output else { throw H3NativeError.missingTensor("hiddenStatesOut") }
    let values = try output.floatValues()
    try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(
      to: URL(fileURLWithPath: outputPath),
      options: .atomic
    )
    let rms = sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(values.count))
    print("shape=\(output.shape) mean=\(values.reduce(0, +) / Float(values.count)) rms=\(rms)")
  }
}
