import CoreAI
import CoreML
import Foundation

@available(macOS 27.0, *)
public struct FlashVSRNativeStreamState {
    fileprivate var keyCaches: [NDArray]
    fileprivate var valueCaches: [NDArray]
    fileprivate var lqCache1: NDArray
    fileprivate var lqCache2: NDArray
    fileprivate var decoderMemories: [NDArray]
    fileprivate var temporalOffset: Int
}

@available(macOS 27.0, *)
public struct FlashVSRNativeChunk {
    public let video: NDArray
    public let state: FlashVSRNativeStreamState
}

@available(macOS 27.0, *)
public enum FlashVSRComputePolicy: String {
    case hybrid
    case automatic
    case gpu
    case neuralEngine
}

private enum NativePipelineError: LocalizedError {
    case missingAsset(String)
    case missingFunction(String, String)
    case missingOutput(String)
    case invalidShape(String)
    case invalidChunkCount(Int)
    case unsupportedScalarType(String)

    var errorDescription: String? {
        switch self {
        case .missingAsset(let name): return "Missing FlashVSR Core AI asset: \(name)"
        case .missingFunction(let asset, let name):
            return "Missing function \(name) in \(asset)"
        case .missingOutput(let name): return "Missing Core AI output: \(name)"
        case .invalidShape(let message): return "Invalid tensor shape: \(message)"
        case .invalidChunkCount(let count):
            return "The first FlashVSR chunk requires six LQ groups, got \(count)"
        case .unsupportedScalarType(let value):
            return "Unsupported native tensor scalar type: \(value)"
        }
    }
}

@available(macOS 27.0, *)
private protocol NativeInferenceFunction {
    func run(inputs: [String: NDArray]) async throws -> [String: NDArray]
}

@available(macOS 27.0, *)
private struct CoreAIFunctionAdapter: NativeInferenceFunction {
    let function: InferenceFunction
    let outputNames: [String]

    func run(inputs: [String: NDArray]) async throws -> [String: NDArray] {
        var native = try await function.run(inputs: inputs)
        var outputs: [String: NDArray] = [:]
        for name in outputNames {
            guard let array = native.remove(name)?.ndArray else {
                throw NativePipelineError.missingOutput(name)
            }
            outputs[name] = array
        }
        return outputs
    }
}

/// Bridges fixed-shape convolution/projection functions to Core ML so the M5
/// execution planner can place supported operations on ANE.  DiT attention
/// stays in Core AI on the GPU; only component boundaries cross this bridge.
@available(macOS 27.0, *)
private final class CoreMLFunctionAdapter: NativeInferenceFunction {
    private let model: MLModel
    private let outputMapping: [String: String]

    init(model: MLModel, outputMapping: [String: String]) {
        self.model = model
        self.outputMapping = outputMapping
    }

    private func copyToNDArray(_ array: MLMultiArray) throws -> NDArray {
        let shape = array.shape.map(\.intValue)
        let sourceStrides = array.strides.map(\.intValue)
        let scalarCount = shape.reduce(1, *)
        guard !shape.isEmpty, shape.count == sourceStrides.count else {
            throw NativePipelineError.invalidShape("Core ML output")
        }
        var result = NDArray(shape: shape, scalarType: .float16)
        var destination = result.mutableView(as: Float16.self)
        try destination.withUnsafeMutablePointer { target, _, _ in
            try array.withUnsafeBufferPointer(ofType: Float16.self) { source in
                guard let sourceBase = source.baseAddress else {
                    throw NativePipelineError.invalidShape("empty Core ML output")
                }
                let lastCount = shape.last!
                let outerCount = scalarCount / lastCount
                for outer in 0..<outerCount {
                    var residual = outer
                    var sourceOffset = 0
                    if shape.count > 1 {
                        for axis in stride(from: shape.count - 2, through: 0, by: -1) {
                            let coordinate = residual % shape[axis]
                            residual /= shape[axis]
                            sourceOffset += coordinate * sourceStrides[axis]
                        }
                    }
                    let targetOffset = outer * lastCount
                    if sourceStrides.last == 1 {
                        guard sourceOffset + lastCount <= source.count else {
                            throw NativePipelineError.invalidShape("Core ML output strides")
                        }
                        target.advanced(by: targetOffset).update(
                            from: sourceBase.advanced(by: sourceOffset), count: lastCount
                        )
                    } else {
                        for index in 0..<lastCount {
                            let offset = sourceOffset + index * sourceStrides.last!
                            guard offset < source.count else {
                                throw NativePipelineError.invalidShape("Core ML output strides")
                            }
                            target[targetOffset + index] = sourceBase[offset]
                        }
                    }
                }
            }
        }
        return result
    }

    func run(inputs: [String: NDArray]) async throws -> [String: NDArray] {
        var features: [String: MLFeatureValue] = [:]
        for (name, value) in inputs {
            guard value.scalarType == .float16 else {
                throw NativePipelineError.unsupportedScalarType("\(value.scalarType)")
            }
            var strides = Array(repeating: 1, count: value.shape.count)
            if value.shape.count > 1 {
                for index in stride(from: value.shape.count - 2, through: 0, by: -1) {
                    strides[index] = strides[index + 1] * value.shape[index + 1]
                }
            }
            let array = MLMultiArray(
                shape: value.shape, dataType: .float16, strides: strides
            )
            let source = value.view(as: Float16.self)
            try array.withUnsafeMutableBufferPointer(ofType: Float16.self) { target, _ in
                try source.withUnsafePointer { pointer, _, _ in
                    target.baseAddress?.update(from: pointer, count: target.count)
                }
            }
            features[name] = MLFeatureValue(multiArray: array)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let prediction = try await model.prediction(from: provider)
        var outputs: [String: NDArray] = [:]
        for (actualName, logicalName) in outputMapping {
            guard let array = prediction.featureValue(for: actualName)?.multiArrayValue else {
                throw NativePipelineError.missingOutput(actualName)
            }
            guard array.dataType == .float16 else {
                throw NativePipelineError.unsupportedScalarType("\(array.dataType)")
            }
            outputs[logicalName] = try copyToNDArray(array)
        }
        return outputs
    }
}

/// Python-free FlashVSR-v1.1 runtime. Every learned operation is loaded from
/// Core AI assets; Swift owns the causal state and exact 3-D RoPE positions.
@available(macOS 27.0, *)
public final class FlashVSRNativePipeline {
    private let gridHeight = 16
    private let gridWidth = 16
    private var retainedModels: [AIModel] = []
    private let patchFirst: any NativeInferenceFunction
    private let patchNext: any NativeInferenceFunction
    private let headFirst: any NativeInferenceFunction
    private let headNext: any NativeInferenceFunction
    private let lqWarmup: any NativeInferenceFunction
    private let lqNext: any NativeInferenceFunction
    private let decoderStep: any NativeInferenceFunction
    private let blockFirst: [any NativeInferenceFunction]
    private let blockNext: [any NativeInferenceFunction]

    public let computeSummary: String

    public init(
        modelsDirectory: URL,
        computePolicy: FlashVSRComputePolicy = .hybrid
    ) async throws {
        let available = ComputeUnitKind.availableKinds
        func preferred(_ kind: ComputeUnitKind) -> SpecializationOptions {
            available.contains(kind)
                ? SpecializationOptions(preferredComputeUnitKind: kind)
                : .default
        }
        let tensorOptions: SpecializationOptions
        let attentionOptions: SpecializationOptions
        let coreMLComputeUnits: MLComputeUnits
        switch computePolicy {
        case .hybrid:
            // Convolutions and fixed-shape projections map well to the ANE;
            // dynamic top-k and masked SDPA remain GPU-preferred. Every option
            // still allows all available units for per-operation fallback.
            tensorOptions = preferred(.neuralEngine)
            attentionOptions = preferred(.gpu)
            coreMLComputeUnits = .all
        case .automatic:
            tensorOptions = .default
            attentionOptions = .default
            coreMLComputeUnits = .all
        case .gpu:
            tensorOptions = preferred(.gpu)
            attentionOptions = preferred(.gpu)
            coreMLComputeUnits = .cpuAndGPU
        case .neuralEngine:
            tensorOptions = preferred(.neuralEngine)
            attentionOptions = preferred(.neuralEngine)
            coreMLComputeUnits = .cpuAndNeuralEngine
        }
        var loadedModels: [AIModel] = []

        func asset(named stem: String) throws -> URL {
            let exact = ["\(stem).aimodel", "\(stem).aimodelc"]
            for name in exact {
                let candidate = modelsDirectory.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
            let entries = try FileManager.default.contentsOfDirectory(
                at: modelsDirectory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            if let candidate = entries.first(where: {
                $0.lastPathComponent.hasPrefix(stem + ".")
                    && ($0.pathExtension == "aimodel" || $0.pathExtension == "aimodelc")
            }) {
                return candidate
            }
            throw NativePipelineError.missingAsset(stem)
        }

        func load(
            _ stem: String,
            options: SpecializationOptions
        ) async throws -> AIModel {
            let model = try await AIModel(contentsOf: asset(named: stem), options: options)
            loadedModels.append(model)
            return model
        }

        func function(
            _ model: AIModel, asset: String, name: String, outputs: [String]
        ) throws -> any NativeInferenceFunction {
            guard let value = try model.loadFunction(named: name) else {
                throw NativePipelineError.missingFunction(asset, name)
            }
            return CoreAIFunctionAdapter(function: value, outputNames: outputs)
        }

        func coreMLAsset(named stem: String) -> URL? {
            let candidate = modelsDirectory.appendingPathComponent("\(stem).mlmodelc")
            return FileManager.default.fileExists(atPath: candidate.path)
                ? candidate : nil
        }

        func coreMLFunction(
            asset: URL, name: String, outputMapping: [String: String]
        ) async throws -> any NativeInferenceFunction {
            let configuration = MLModelConfiguration()
            configuration.functionName = name
            configuration.computeUnits = coreMLComputeUnits
            let model = try await MLModel.load(
                contentsOf: asset, configuration: configuration
            )
            return CoreMLFunctionAdapter(model: model, outputMapping: outputMapping)
        }

        let coreMLFront = coreMLAsset(named: "patch_head")
        let coreMLLQ = coreMLAsset(named: "lq_projection")
        let coreMLDecoder = coreMLAsset(named: "tcdecoder")
        if let coreMLFront {
            patchFirst = try await coreMLFunction(
                asset: coreMLFront, name: "patch_first", outputMapping: ["x": "x"]
            )
            patchNext = try await coreMLFunction(
                asset: coreMLFront, name: "patch_next", outputMapping: ["x": "x"]
            )
            headFirst = try await coreMLFunction(
                asset: coreMLFront, name: "head_first", outputMapping: ["latent": "latent"]
            )
            headNext = try await coreMLFunction(
                asset: coreMLFront, name: "head_next", outputMapping: ["latent": "latent"]
            )
        } else {
            let front = try await load("patch_head", options: tensorOptions)
            patchFirst = try function(
                front, asset: "patch_head", name: "patch_first", outputs: ["x"]
            )
            patchNext = try function(
                front, asset: "patch_head", name: "patch_next", outputs: ["x"]
            )
            headFirst = try function(
                front, asset: "patch_head", name: "head_first", outputs: ["latent"]
            )
            headNext = try function(
                front, asset: "patch_head", name: "head_next", outputs: ["latent"]
            )
        }

        if let coreMLLQ {
            lqWarmup = try await coreMLFunction(
                asset: coreMLLQ, name: "warmup",
                outputMapping: ["cache1": "cache1", "cache2": "cache2"]
            )
            lqNext = try await coreMLFunction(
                asset: coreMLLQ, name: "next",
                outputMapping: [
                    "lq": "lq", "next_cache1": "cache1", "next_cache2": "cache2"
                ]
            )
        } else {
            let lq = try await load("lq_projection", options: tensorOptions)
            lqWarmup = try function(
                lq, asset: "lq_projection", name: "warmup", outputs: ["cache1", "cache2"]
            )
            lqNext = try function(
                lq, asset: "lq_projection", name: "next",
                outputs: ["lq", "cache1", "cache2"]
            )
        }

        if let coreMLDecoder {
            decoderStep = try await coreMLFunction(
                asset: coreMLDecoder, name: "step",
                outputMapping: Dictionary(
                    uniqueKeysWithValues: [("video", "video")]
                        + (0..<9).map { ("next_memory\($0)", "memory\($0)") }
                )
            )
        } else {
            let decoder = try await load("tcdecoder", options: tensorOptions)
            decoderStep = try function(
                decoder, asset: "tcdecoder", name: "step",
                outputs: ["video"] + (0..<9).map { "memory\($0)" }
            )
        }

        var firstFunctions: [any NativeInferenceFunction] = []
        var nextFunctions: [any NativeInferenceFunction] = []
        for index in 0..<30 {
            let stem = String(format: "dit_block_%02d", index)
            let model = try await load(stem, options: attentionOptions)
            let outputs = ["x", "cache_k", "cache_v"]
            firstFunctions.append(try function(
                model, asset: stem, name: "first_chunk", outputs: outputs
            ))
            nextFunctions.append(try function(
                model, asset: stem, name: "next_chunk", outputs: outputs
            ))
        }
        blockFirst = firstFunctions
        blockNext = nextFunctions
        retainedModels = loadedModels
        var planned: [String] = []
        if coreMLFront != nil { planned.append("front/head") }
        if coreMLLQ != nil { planned.append("LQ") }
        if coreMLDecoder != nil { planned.append("decoder") }
        let tensorRuntime = planned.isEmpty
            ? "Core AI tensor path"
            : "Core ML planner [\(planned.joined(separator: ","))]"
        computeSummary = "\(computePolicy.rawValue) · \(tensorRuntime) + Core AI GPU · available=\(available)"
    }

    public func firstChunk(
        latent: NDArray,
        firstLQFrame: NDArray,
        lqGroups: [NDArray],
        decoderGroups: [NDArray]
    ) async throws -> FlashVSRNativeChunk {
        guard lqGroups.count == 6 else {
            throw NativePipelineError.invalidChunkCount(lqGroups.count)
        }
        guard decoderGroups.count == 6 else {
            throw NativePipelineError.invalidChunkCount(decoderGroups.count)
        }
        var warmup = try await lqWarmup.run(inputs: ["first_frame": firstLQFrame])
        var lqCache1 = try take(&warmup, "cache1")
        var lqCache2 = try take(&warmup, "cache2")
        var lqParts: [NDArray] = []
        for group in lqGroups {
            var result = try await lqNext.run(inputs: [
                "frames": group,
                "cache1": lqCache1,
                "cache2": lqCache2,
            ])
            lqParts.append(try take(&result, "lq"))
            lqCache1 = try take(&result, "cache1")
            lqCache2 = try take(&result, "cache2")
        }
        let lq = try concatenateTokens(lqParts)
        var patch = try await patchFirst.run(inputs: ["latent": latent])
        var x = try take(&patch, "x")
        let rope = try makeRoPE(frames: 6, temporalOffset: 0)
        let zeroLQ = NDArray(shape: [1, 1536, 1536], scalarType: .float16)
        var keys: [NDArray] = []
        var values: [NDArray] = []
        for index in 0..<30 {
            var result = try await blockFirst[index].run(inputs: [
                "x": x,
                "lq": index == 0 ? lq : zeroLQ,
                "rope_cos": rope.cos,
                "rope_sin": rope.sin,
            ])
            x = try take(&result, "x")
            keys.append(try take(&result, "cache_k"))
            values.append(try take(&result, "cache_v"))
        }
        var head = try await headFirst.run(inputs: ["x": x])
        let prediction = try take(&head, "latent")
        let denoised = try subtract(latent, prediction)
        var memories = initialDecoderMemories()
        var videoGroups: [NDArray] = []
        for index in 0..<6 {
            let result = try await decodeStep(
                latent: sliceLatentFrame(denoised, index: index, frames: 6),
                condition: decoderGroups[index],
                memories: memories
            )
            videoGroups.append(result.video)
            memories = result.memories
        }
        // The original streaming decoder prepends three repeated frames to
        // the first condition group and removes their decoded outputs.
        let video = try concatenateVideoGroups(videoGroups, droppingFirst: 3)
        return FlashVSRNativeChunk(
            video: video,
            state: FlashVSRNativeStreamState(
                keyCaches: keys,
                valueCaches: values,
                lqCache1: lqCache1,
                lqCache2: lqCache2,
                decoderMemories: memories,
                temporalOffset: 6
            )
        )
    }

    public func nextChunk(
        latent: NDArray,
        lqGroups: [NDArray],
        decoderGroups: [NDArray],
        state: FlashVSRNativeStreamState
    ) async throws -> FlashVSRNativeChunk {
        guard lqGroups.count == 2 else {
            throw NativePipelineError.invalidChunkCount(lqGroups.count)
        }
        guard decoderGroups.count == 2 else {
            throw NativePipelineError.invalidChunkCount(decoderGroups.count)
        }
        var lqCache1 = state.lqCache1
        var lqCache2 = state.lqCache2
        var lqParts: [NDArray] = []
        for group in lqGroups {
            var result = try await lqNext.run(inputs: [
                "frames": group,
                "cache1": lqCache1,
                "cache2": lqCache2,
            ])
            lqParts.append(try take(&result, "lq"))
            lqCache1 = try take(&result, "cache1")
            lqCache2 = try take(&result, "cache2")
        }
        let lq = try concatenateTokens(lqParts)
        var patch = try await patchNext.run(inputs: ["latent": latent])
        var x = try take(&patch, "x")
        let rope = try makeRoPE(frames: 2, temporalOffset: state.temporalOffset)
        let zeroLQ = NDArray(shape: [1, 512, 1536], scalarType: .float16)
        var keys: [NDArray] = []
        var values: [NDArray] = []
        for index in 0..<30 {
            var result = try await blockNext[index].run(inputs: [
                "x": x,
                "lq": index == 0 ? lq : zeroLQ,
                "rope_cos": rope.cos,
                "rope_sin": rope.sin,
                "cache_k": state.keyCaches[index],
                "cache_v": state.valueCaches[index],
            ])
            x = try take(&result, "x")
            keys.append(try take(&result, "cache_k"))
            values.append(try take(&result, "cache_v"))
        }
        var head = try await headNext.run(inputs: ["x": x])
        let prediction = try take(&head, "latent")
        let denoised = try subtract(latent, prediction)
        var memories = state.decoderMemories
        var videoGroups: [NDArray] = []
        for index in 0..<2 {
            let result = try await decodeStep(
                latent: sliceLatentFrame(denoised, index: index, frames: 2),
                condition: decoderGroups[index],
                memories: memories
            )
            videoGroups.append(result.video)
            memories = result.memories
        }
        let video = try concatenateVideoGroups(videoGroups, droppingFirst: 0)
        return FlashVSRNativeChunk(
            video: video,
            state: FlashVSRNativeStreamState(
                keyCaches: keys,
                valueCaches: values,
                lqCache1: lqCache1,
                lqCache2: lqCache2,
                decoderMemories: memories,
                temporalOffset: state.temporalOffset + 2
            )
        )
    }

    private func take(
        _ outputs: inout [String: NDArray],
        _ name: String
    ) throws -> NDArray {
        guard let array = outputs.removeValue(forKey: name) else {
            throw NativePipelineError.missingOutput(name)
        }
        return array
    }

    private func concatenateTokens(_ arrays: [NDArray]) throws -> NDArray {
        guard let first = arrays.first, first.shape.count == 3,
              first.shape[0] == 1, first.shape[2] == 1536 else {
            throw NativePipelineError.invalidShape("LQ tokens")
        }
        let count = arrays.reduce(0) { $0 + $1.shape[1] }
        var destination = NDArray(shape: [1, count, 1536], scalarType: .float16)
        var destinationView = destination.mutableView(as: Float16.self)
        try destinationView.withUnsafeMutablePointer { target, _, _ in
            var scalarOffset = 0
            for array in arrays {
                let sourceView = array.view(as: Float16.self)
                let scalars = array.shape.reduce(1, *)
                guard sourceView.isContiguous else {
                    throw NativePipelineError.invalidShape("non-contiguous LQ output")
                }
                try sourceView.withUnsafePointer { source, _, _ in
                    target.advanced(by: scalarOffset).update(from: source, count: scalars)
                    scalarOffset += scalars
                }
            }
        }
        return destination
    }

    private func subtract(_ lhs: NDArray, _ rhs: NDArray) throws -> NDArray {
        guard lhs.shape == rhs.shape else {
            throw NativePipelineError.invalidShape("latent subtraction")
        }
        var result = NDArray(shape: lhs.shape, scalarType: .float16)
        var outputView = result.mutableView(as: Float16.self)
        let leftView = lhs.view(as: Float16.self)
        let rightView = rhs.view(as: Float16.self)
        let scalarCount = lhs.shape.reduce(1, *)
        try outputView.withUnsafeMutablePointer { output, _, _ in
            try leftView.withUnsafePointer { left, _, _ in
                try rightView.withUnsafePointer { right, _, _ in
                    for index in 0..<scalarCount {
                        output[index] = left[index] - right[index]
                    }
                }
            }
        }
        return result
    }

    private func initialDecoderMemories() -> [NDArray] {
        var result: [NDArray] = []
        for _ in 0..<3 {
            result.append(NDArray(shape: [1, 512, 32, 32], scalarType: .float16))
        }
        for _ in 0..<3 {
            result.append(NDArray(shape: [1, 256, 64, 64], scalarType: .float16))
        }
        for _ in 0..<3 {
            result.append(NDArray(shape: [1, 128, 128, 128], scalarType: .float16))
        }
        return result
    }

    private func decodeStep(
        latent: NDArray,
        condition: NDArray,
        memories: [NDArray]
    ) async throws -> (video: NDArray, memories: [NDArray]) {
        guard memories.count == 9 else {
            throw NativePipelineError.invalidShape("decoder memories")
        }
        var inputs: [String: NDArray] = [
            "latent": latent,
            "condition": condition,
        ]
        for index in 0..<9 {
            inputs["memory\(index)"] = memories[index]
        }
        var outputs = try await decoderStep.run(inputs: inputs)
        let video = try take(&outputs, "video")
        var next: [NDArray] = []
        for index in 0..<9 {
            next.append(try take(&outputs, "memory\(index)"))
        }
        return (video, next)
    }

    private func sliceLatentFrame(
        _ latent: NDArray,
        index: Int,
        frames: Int
    ) throws -> NDArray {
        guard latent.shape == [1, 16, frames, 32, 32], index < frames else {
            throw NativePipelineError.invalidShape("latent frame")
        }
        var result = NDArray(shape: [1, 16, 32, 32], scalarType: .float16)
        let sourceView = latent.view(as: Float16.self)
        var targetView = result.mutableView(as: Float16.self)
        try targetView.withUnsafeMutablePointer { target, _, _ in
            try sourceView.withUnsafePointer { source, _, _ in
                for channel in 0..<16 {
                    let sourceOffset = (channel * frames + index) * 32 * 32
                    let targetOffset = channel * 32 * 32
                    target.advanced(by: targetOffset).update(
                        from: source.advanced(by: sourceOffset),
                        count: 32 * 32
                    )
                }
            }
        }
        return result
    }

    private func concatenateVideoGroups(
        _ groups: [NDArray],
        droppingFirst: Int
    ) throws -> NDArray {
        guard !groups.isEmpty else {
            throw NativePipelineError.invalidShape("empty decoder output")
        }
        let totalFrames = groups.reduce(0) { $0 + $1.shape[1] }
        guard groups.allSatisfy({ $0.shape.count == 5 && $0.shape[0] == 1
            && $0.shape[2] == 3 && $0.shape[3] == 256 && $0.shape[4] == 256 }),
              droppingFirst >= 0, droppingFirst < totalFrames else {
            throw NativePipelineError.invalidShape("decoder video")
        }
        let frameScalars = 3 * 256 * 256
        var result = NDArray(
            shape: [1, totalFrames - droppingFirst, 3, 256, 256],
            scalarType: .float16
        )
        var targetView = result.mutableView(as: Float16.self)
        try targetView.withUnsafeMutablePointer { target, _, _ in
            var destinationFrame = 0
            var globalFrame = 0
            for group in groups {
                let sourceView = group.view(as: Float16.self)
                try sourceView.withUnsafePointer { source, _, _ in
                    for localFrame in 0..<group.shape[1] {
                        defer { globalFrame += 1 }
                        guard globalFrame >= droppingFirst else { continue }
                        target.advanced(by: destinationFrame * frameScalars).update(
                            from: source.advanced(by: localFrame * frameScalars),
                            count: frameScalars
                        )
                        destinationFrame += 1
                    }
                }
            }
        }
        return result
    }

    private func makeRoPE(
        frames: Int,
        temporalOffset: Int
    ) throws -> (cos: NDArray, sin: NDArray) {
        let tokens = frames * gridHeight * gridWidth
        var cosine = NDArray(shape: [tokens, 1, 64], scalarType: .float16)
        var sine = NDArray(shape: [tokens, 1, 64], scalarType: .float16)
        var cosineValues = [Float16](repeating: 0, count: tokens * 64)
        var sineValues = [Float16](repeating: 0, count: tokens * 64)
        var token = 0
        for frame in 0..<frames {
            for row in 0..<gridHeight {
                for column in 0..<gridWidth {
                    var pair = 0
                    func writeAxis(position: Int, dimension: Int) {
                        for index in 0..<(dimension / 2) {
                            let exponent = Double(index * 2) / Double(dimension)
                            let angle = Double(position) * pow(10_000.0, -exponent)
                            cosineValues[token * 64 + pair] = Float16(cos(angle))
                            sineValues[token * 64 + pair] = Float16(sin(angle))
                            pair += 1
                        }
                    }
                    writeAxis(position: temporalOffset + frame, dimension: 44)
                    writeAxis(position: row, dimension: 42)
                    writeAxis(position: column, dimension: 42)
                    token += 1
                }
            }
        }
        cosine.mutableView(as: Float16.self).withUnsafeMutablePointer { target, _, _ in
            cosineValues.withUnsafeBufferPointer {
                target.update(from: $0.baseAddress!, count: $0.count)
            }
        }
        sine.mutableView(as: Float16.self).withUnsafeMutablePointer { target, _, _ in
            sineValues.withUnsafeBufferPointer {
                target.update(from: $0.baseAddress!, count: $0.count)
            }
        }
        return (cosine, sine)
    }
}
