import CoreAI
import CoreML
import Darwin
import Foundation

private enum VerificationError: LocalizedError {
  case invalidArguments
  case missingAsset(String)
  case unexpectedAssets([String])
  case invalidCollection(String, missing: [String], unexpected: [String])
  case missingFunction(String)
  case invalidJSON(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: mioh-dedicated-model-verifier <models-dir> <architecture>"
    case .missingAsset(let name):
      return "missing dedicated model asset: \(name)"
    case .unexpectedAssets(let names):
      return "unexpected dedicated Core AI assets: \(names.joined(separator: ", "))"
    case .invalidCollection(let name, let missing, let unexpected):
      return "invalid \(name): missing=\(missing), unexpected=\(unexpected)"
    case .missingFunction(let name):
      return "Core AI model has no main function: \(name)"
    case .invalidJSON(let name):
      return "invalid dedicated model metadata: \(name)"
    }
  }
}

@main
private struct DedicatedModelVerifier {
  private static let fixedCoreAIStems = [
    "basicvsrpp-v1.2-t18-fp16",
    "basicvsrpp-v1.2-t36-fp16",
    "basicvsrpp-v1.2-t90-fp16",
    "lada_mosaic_detection_model_v2-fp16",
    "lada_mosaic_detection_model_v3.1_fast-fp16",
    "lada_mosaic_detection_model_v3.1_accurate-fp16",
    "lada_mosaic_detection_model_v4_fast-fp16",
    "lada_mosaic_detection_model_v4_accurate-fp16",
    "lada_mosaic_detection_model_vr_v2_accurate-fp16",
    "RealESRGAN_x2plus-256-fp16",
    "RealESRGAN_x4plus-256-fp16",
    "realesr-general-x4v3-256-fp16",
    "4xNomosWebPhoto_RealPLKSR-256-fp16",
  ]

  private static let chunk6Names = [
    "spatial6", "flow6",
    "backward_1_start6", "backward_1_continue6",
    "forward_1_start6", "forward_1_continue6",
    "backward_2_start6", "backward_2_continue6",
    "forward_2_start6", "forward_2_continue6",
    "reconstruction6",
  ]

  private static let coreMLNames = Set([
    "lada_mosaic_detection_model_v2.mlmodelc",
    "lada_mosaic_detection_model_v3.1_fast.mlmodelc",
    "lada_mosaic_detection_model_v3.1_accurate.mlmodelc",
    "lada_mosaic_detection_model_v4_fast.mlmodelc",
    "lada_mosaic_detection_model_v4_accurate.mlmodelc",
    "lada_mosaic_detection_model_vr_v2_accurate.mlmodelc",
    "rfdetr-v6-576-fp32.mlmodelc",
    "rfdetr-v6-large-768-fp32.mlmodelc",
    "RealESRGAN_x4plus_256.mlmodelc",
    "realesr-general-x4v3_256.mlmodelc",
    "MewZoom-V1-4X-Unet_256.mlmodelc",
    "swinir-real-x4_256.mlmodelc",
    "4xNomosWebPhoto_RealPLKSR_256.mlmodelc",
    "PiperSR_2x_256.mlmodelc",
  ])

  static func main() async {
    do {
      try await verify()
    } catch {
      FileHandle.standardError.write(
        Data("mioh-dedicated-model-verifier: \(error.localizedDescription)\n".utf8)
      )
      exit(EXIT_FAILURE)
    }
  }

  private static func verify() async throws {
    guard CommandLine.arguments.count == 3 else {
      throw VerificationError.invalidArguments
    }
    let modelsDirectory = URL(
      fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let architecture = CommandLine.arguments[2]
    let fileManager = FileManager.default

    let fixedNames = Set(
      fixedCoreAIStems.map { "\($0).\(architecture).aimodelc" })
    let collectionNames = Set([
      "basicvsrpp-v1.2-variable-coreai.\(architecture).aimodelc",
    ])
    let expectedTopLevel = fixedNames.union(collectionNames)
    let entries = try fileManager.contentsOfDirectory(
      at: modelsDirectory,
      includingPropertiesForKeys: [.isDirectoryKey],
      options: [.skipsHiddenFiles]
    )
    let actualTopLevel = Set(
      entries.lazy
        .filter { $0.lastPathComponent.hasSuffix(".aimodelc") }
        .map(\.lastPathComponent)
    )
    let unexpected = actualTopLevel.subtracting(expectedTopLevel).sorted()
    guard unexpected.isEmpty else {
      throw VerificationError.unexpectedAssets(unexpected)
    }
    for name in expectedTopLevel.sorted() {
      guard actualTopLevel.contains(name) else {
        throw VerificationError.missingAsset(name)
      }
    }

    for name in fixedNames.sorted() {
      try await loadMainFunction(modelsDirectory.appendingPathComponent(name))
    }
    try await verifyCollection(
      modelsDirectory.appendingPathComponent(
        "basicvsrpp-v1.2-variable-coreai.\(architecture).aimodelc"),
      architecture: architecture,
      names: chunk6Names
    )
    for name in [
      "rfdetr-v6-576-fp32.aimodel",
      "rfdetr-v6-large-768-fp32.aimodel",
    ] {
      let source = modelsDirectory.appendingPathComponent(name)
      guard fileManager.fileExists(atPath: source.path) else {
        throw VerificationError.missingAsset(name)
      }
      try await loadMainFunction(source)
    }

    let actualCoreMLNames = Set(
      entries.lazy
        .filter { $0.lastPathComponent.hasSuffix(".mlmodelc") }
        .map(\.lastPathComponent)
    )
    let missingCoreML = coreMLNames.subtracting(actualCoreMLNames).sorted()
    let unexpectedCoreML = actualCoreMLNames.subtracting(coreMLNames).sorted()
    guard missingCoreML.isEmpty, unexpectedCoreML.isEmpty else {
      throw VerificationError.invalidCollection(
        "Core ML asset set",
        missing: missingCoreML,
        unexpected: unexpectedCoreML
      )
    }
    let coreMLModels = entries.filter {
      coreMLNames.contains($0.lastPathComponent)
    }
    for modelURL in coreMLModels {
      _ = try MLModel(contentsOf: modelURL)
    }

    for name in [
      "basicvsrpp-v1.2-variable-coreai.provenance.json",
      "mioh-cluster-model-identities-v1.json",
    ] {
      let url = modelsDirectory.appendingPathComponent(name)
      guard fileManager.fileExists(atPath: url.path) else {
        throw VerificationError.missingAsset(name)
      }
      let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
      guard object is [String: Any] else {
        throw VerificationError.invalidJSON(name)
      }
    }
    print(
      "Dedicated model verification passed: "
        + "\(expectedTopLevel.count) Core AI assets, "
        + "\(coreMLModels.count) Core ML assets"
    )
  }

  private static func verifyCollection(
    _ collection: URL,
    architecture: String,
    names: [String]
  ) async throws {
    let expected = Set(
      names.map { "basicvsrpp-variable-\($0).\(architecture).aimodelc" })
    let entries = try FileManager.default.contentsOfDirectory(
      at: collection,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    )
    let actual = Set(
      entries.lazy
        .filter { $0.lastPathComponent.hasSuffix(".aimodelc") }
        .map(\.lastPathComponent)
    )
    let missing = expected.subtracting(actual).sorted()
    let unexpected = actual.subtracting(expected).sorted()
    guard missing.isEmpty, unexpected.isEmpty else {
      throw VerificationError.invalidCollection(
        collection.lastPathComponent,
        missing: missing,
        unexpected: unexpected
      )
    }
    for model in entries where expected.contains(model.lastPathComponent) {
      try await loadMainFunction(model)
    }
  }

  private static func loadMainFunction(_ modelURL: URL) async throws {
    let model = try await AIModel(contentsOf: modelURL)
    guard try model.loadFunction(named: "main") != nil else {
      throw VerificationError.missingFunction(modelURL.lastPathComponent)
    }
  }
}
