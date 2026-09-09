import Foundation
import Metal
import MiohRemoteKit

#if canImport(CoreAI)
  import CoreAI

  @available(iOS 27.0, *)
  final class MiohIPadVariableRestorer: MiohIPadRestoring, @unchecked Sendable {
    private static let imageSize = 256
    private static let featureSize = 64
    private static let featureChannels = 64
    private static let fusedChannels = 320
    private static let chunkSize = 6
    private static let frameElements = 3 * imageSize * imageSize
    private static let featurePlaneElements = featureSize * featureSize
    private static let featureElements = featureChannels * featurePlaneElements
    private static let flowElements = 2 * featurePlaneElements

    private struct Branch {
      let name: String
      let backward: Bool
    }

    private static let branches = [
      Branch(name: "backward_1", backward: true),
      Branch(name: "forward_1", backward: false),
      Branch(name: "backward_2", backward: true),
      Branch(name: "forward_2", backward: false),
    ]

    private final class Workspace {
      let maximumFrames: Int
      let frameBuffer: MTLBuffer
      let featureBuffer: MTLBuffer
      let backwardFlowBuffer: MTLBuffer
      let forwardFlowBuffer: MTLBuffer
      let restoredBuffer: MTLBuffer
      let chunkFrameBuffer: MTLBuffer
      let flowFrameBuffer: MTLBuffer
      let contextBuffer: MTLBuffer
      let flowChunkBuffer: MTLBuffer
      let featureChunkBuffer: MTLBuffer
      let restoredChunkBuffer: MTLBuffer
      // Native-state continuation graphs mutate these three boundary tensors
      // in place. Explicit-I/O assets remain supported and leave them unused.
      let nativeStateN1Buffer: MTLBuffer
      let nativeStateN2Buffer: MTLBuffer
      let nativePreviousFlowBuffer: MTLBuffer
      let spatialStream = ComputeStream()
      let flowStream = ComputeStream()
      let propagationStream = ComputeStream()
      let reconstructionStream = ComputeStream()

      init(maximumFrames: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
          throw MiohIPadWorkerEngineError.restorer("Metal device unavailable")
        }
        self.maximumFrames = maximumFrames
        let half = MemoryLayout<Float16>.stride
        let frameBytes = MiohIPadVariableRestorer.frameElements * half
        let perFrameFeatureBytes =
          MiohIPadVariableRestorer.fusedChannels
          * MiohIPadVariableRestorer.featurePlaneElements * half
        let flowBytes = MiohIPadVariableRestorer.flowElements * half
        func makeBuffer(_ length: Int) throws -> MTLBuffer {
          guard let value = device.makeBuffer(
            length: length,
            options: .storageModeShared
          ) else {
            throw MiohIPadWorkerEngineError.restorer(
              "variable workspace allocation failed"
            )
          }
          memset(value.contents(), 0, length)
          return value
        }
        frameBuffer = try makeBuffer(maximumFrames * frameBytes)
        featureBuffer = try makeBuffer(maximumFrames * perFrameFeatureBytes)
        backwardFlowBuffer = try makeBuffer(
          (maximumFrames + MiohIPadVariableRestorer.chunkSize) * flowBytes
        )
        forwardFlowBuffer = try makeBuffer(
          (maximumFrames + MiohIPadVariableRestorer.chunkSize) * flowBytes
        )
        restoredBuffer = try makeBuffer(maximumFrames * frameBytes)
        chunkFrameBuffer = try makeBuffer(
          MiohIPadVariableRestorer.chunkSize * frameBytes
        )
        flowFrameBuffer = try makeBuffer(
          (MiohIPadVariableRestorer.chunkSize + 1) * frameBytes
        )
        contextBuffer = try makeBuffer(
          MiohIPadVariableRestorer.chunkSize * perFrameFeatureBytes
        )
        flowChunkBuffer = try makeBuffer(
          MiohIPadVariableRestorer.chunkSize * flowBytes
        )
        featureChunkBuffer = try makeBuffer(
          MiohIPadVariableRestorer.chunkSize
            * MiohIPadVariableRestorer.featureElements * half
        )
        restoredChunkBuffer = try makeBuffer(
          MiohIPadVariableRestorer.chunkSize * frameBytes
        )
        nativeStateN1Buffer = try makeBuffer(
          MiohIPadVariableRestorer.featureElements * half
        )
        nativeStateN2Buffer = try makeBuffer(
          MiohIPadVariableRestorer.featureElements * half
        )
        nativePreviousFlowBuffer = try makeBuffer(
          MiohIPadVariableRestorer.flowElements * half
        )
      }
    }

    private let functions: [String: InferenceFunction]
    private let workspace: Workspace

    /// Prefer an ahead-of-time specialization for the current Core AI device.
    /// Source `.aimodel` assets remain bundled for identity verification and as
    /// a fallback on architectures for which this build has no specialization.
    static func runtimeAssetURL(for sourceURL: URL) -> URL {
      let sourceBaseName = sourceURL.deletingPathExtension().lastPathComponent
      let architecture = AIModel.deviceArchitectureName
      let compiledName = "\(sourceBaseName).\(architecture).aimodelc"
      let compiledURL = sourceURL.deletingLastPathComponent()
        .appendingPathComponent(compiledName, isDirectory: true)
      guard FileManager.default.fileExists(atPath: compiledURL.path) else {
        return sourceURL
      }
      return compiledURL
    }

    init(
      assetURLsBySourceName: [String: URL],
      maximumFrames: Int = 90
    ) async throws {
      var names = ["spatial6", "flow6", "reconstruction6"]
      for branch in Self.branches {
        names.append("\(branch.name)_start6")
        names.append("\(branch.name)_continue6")
      }
      var loaded: [String: InferenceFunction] = [:]
      for name in names {
        let sourceName = "basicvsrpp-variable-\(name).aimodel"
        guard let url = assetURLsBySourceName[sourceName] else {
          throw MiohIPadWorkerEngineError.restorer(
            "variable asset is missing: \(sourceName)"
          )
        }
        let runtimeURL = Self.runtimeAssetURL(for: url)
        let model: AIModel
        do {
          model = try await AIModel(contentsOf: runtimeURL)
        } catch is CancellationError {
          throw CancellationError()
        } catch {
          throw Self.stageFailure(
            "モデル初期化 \(runtimeURL.lastPathComponent)",
            error: error
          )
        }
        do {
          guard let function = try model.loadFunction(named: "main") else {
            throw MiohClusterAssetError.missingMainFunction
          }
          loaded[name] = function
        } catch is CancellationError {
          throw CancellationError()
        } catch {
          throw Self.stageFailure(
            "main関数読込み \(runtimeURL.lastPathComponent)",
            error: error
          )
        }
      }
      functions = loaded
      workspace = try Workspace(maximumFrames: maximumFrames)
    }

    func restore(_ frames: [Float16], frameCount: Int) async throws
      -> MiohIPadRestoredFrames
    {
      guard frameCount > 0, frameCount <= workspace.maximumFrames,
        frames.count == frameCount * Self.frameElements
      else {
        throw MiohIPadWorkerEngineError.restorer("invalid variable-length input")
      }
      let frameBytes = Self.frameElements * MemoryLayout<Float16>.stride
      _ = frames.withUnsafeBytes { source in
        memcpy(workspace.frameBuffer.contents(), source.baseAddress!, frameCount * frameBytes)
      }
      try await infer(frameCount: frameCount)
      return MiohIPadRestoredFrames(
        metalBuffer: workspace.restoredBuffer,
        count: frameCount * Self.frameElements
      )
    }

    private func infer(frameCount: Int) async throws {
      let half = MemoryLayout<Float16>.stride
      let frameBytes = Self.frameElements * half
      let featureBytes = Self.featureElements * half
      let perFrameFeatureBytes =
        Self.fusedChannels * Self.featurePlaneElements * half
      let flowBytes = Self.flowElements * half
      memset(workspace.featureBuffer.contents(), 0, frameCount * perFrameFeatureBytes)
      guard let spatial = functions["spatial6"],
        let flow = functions["flow6"],
        let reconstruction = functions["reconstruction6"]
      else {
        throw MiohIPadWorkerEngineError.restorer("variable functions are missing")
      }

      for start in stride(from: 0, to: frameCount, by: Self.chunkSize) {
        let valid = min(Self.chunkSize, frameCount - start)
        packFrames(
          indices: (0..<Self.chunkSize).map { min(start + $0, frameCount - 1) },
          frameCount: frameCount,
          destination: workspace.chunkFrameBuffer
        )
        let input = InferenceFunction.AsyncValue(
          unsafeBuffer: workspace.chunkFrameBuffer,
          byteOffset: 0,
          scalarType: .float16,
          shape: [Self.chunkSize, 3, Self.imageSize, Self.imageSize]
        )
        var destination = InferenceFunction.AsyncMutableValue(
          unsafeBuffer: workspace.featureChunkBuffer,
          byteOffset: 0,
          scalarType: .float16,
          shape: [Self.chunkSize, Self.featureChannels, Self.featureSize, Self.featureSize]
        )
        var outputs = InferenceFunction.AsyncMutableViews()
        outputs.insert(&destination, for: "features")
        do {
          _ = try spatial.encode(
            inputs: ["frames": input],
            outputViews: outputs,
            to: workspace.spatialStream
          )
        } catch {
          throw Self.stageFailure(
            "spatial encode chunk \(start / Self.chunkSize)",
            error: error
          )
        }
        await workspace.spatialStream.currentWorkCompleted()
        for offset in 0..<valid {
          memcpy(
            workspace.featureBuffer.contents().advanced(
              by: (start + offset) * perFrameFeatureBytes
            ),
            workspace.featureChunkBuffer.contents().advanced(
              by: offset * featureBytes
            ),
            featureBytes
          )
        }
      }

      if frameCount > 1 {
        for start in stride(from: 0, to: frameCount - 1, by: Self.chunkSize) {
          packFrames(
            indices: (0...Self.chunkSize).map {
              min(start + $0, frameCount - 1)
            },
            frameCount: frameCount,
            destination: workspace.flowFrameBuffer
          )
          let input = InferenceFunction.AsyncValue(
            unsafeBuffer: workspace.flowFrameBuffer,
            byteOffset: 0,
            scalarType: .float16,
            shape: [Self.chunkSize + 1, 3, Self.imageSize, Self.imageSize]
          )
          var backward = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: workspace.backwardFlowBuffer,
            byteOffset: start * flowBytes,
            scalarType: .float16,
            shape: [Self.chunkSize, 2, Self.featureSize, Self.featureSize]
          )
          var forward = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: workspace.forwardFlowBuffer,
            byteOffset: start * flowBytes,
            scalarType: .float16,
            shape: [Self.chunkSize, 2, Self.featureSize, Self.featureSize]
          )
          var outputs = InferenceFunction.AsyncMutableViews()
          outputs.insert(&backward, for: "backward")
          outputs.insert(&forward, for: "forward")
          do {
            _ = try flow.encode(
              inputs: ["frames": input],
              outputViews: outputs,
              to: workspace.flowStream
            )
          } catch {
            throw Self.stageFailure(
              "flow encode chunk \(start / Self.chunkSize)",
              error: error
            )
          }
          await workspace.flowStream.currentWorkCompleted()
        }
      }

      for (branchIndex, branch) in Self.branches.enumerated() {
        let indices = branch.backward
          ? Array((0..<frameCount).reversed()) : Array(0..<frameCount)
        let directional = branch.backward
          ? workspace.backwardFlowBuffer : workspace.forwardFlowBuffer
        let contextChannels = Self.featureChannels * (branchIndex + 1)
        let contextBytes = contextChannels * Self.featurePlaneElements * half
        let outputChannel = Self.featureChannels * (branchIndex + 1)
        for chunkStart in stride(from: 0, to: frameCount, by: Self.chunkSize) {
          let valid = min(Self.chunkSize, frameCount - chunkStart)
          let chunkIndices = (0..<Self.chunkSize).map {
            indices[min(chunkStart + $0, frameCount - 1)]
          }
          for offset in 0..<Self.chunkSize {
            memcpy(
              workspace.contextBuffer.contents().advanced(by: offset * contextBytes),
              workspace.featureBuffer.contents().advanced(
                by: chunkIndices[offset] * perFrameFeatureBytes
              ),
              contextBytes
            )
          }
          let context = InferenceFunction.AsyncValue(
            unsafeBuffer: workspace.contextBuffer,
            byteOffset: 0,
            scalarType: .float16,
            shape: [Self.chunkSize, contextChannels, Self.featureSize, Self.featureSize]
          )
          var inputs: [String: InferenceFunction.AsyncValue] = ["contexts": context]
          let functionName: String
          if chunkStart == 0 {
            functionName = "\(branch.name)_start6"
            for position in 1..<Self.chunkSize {
              if position < frameCount {
                let frameIndex = indices[position]
                let flowIndex = branch.backward ? frameIndex : frameIndex - 1
                memcpy(
                  workspace.flowChunkBuffer.contents().advanced(
                    by: (position - 1) * flowBytes
                  ),
                  directional.contents().advanced(by: flowIndex * flowBytes),
                  flowBytes
                )
              } else {
                memset(
                  workspace.flowChunkBuffer.contents().advanced(
                    by: (position - 1) * flowBytes
                  ),
                  0,
                  flowBytes
                )
              }
            }
            inputs["flows"] = InferenceFunction.AsyncValue(
              unsafeBuffer: workspace.flowChunkBuffer,
              byteOffset: 0,
              scalarType: .float16,
              shape: [Self.chunkSize - 1, 2, Self.featureSize, Self.featureSize]
            )
          } else {
            functionName = "\(branch.name)_continue6"
            var lastFlowIndex = 0
            for offset in 0..<Self.chunkSize {
              let position = chunkStart + offset
              if position < frameCount {
                let frameIndex = indices[position]
                lastFlowIndex = branch.backward ? frameIndex : frameIndex - 1
              }
              memcpy(
                workspace.flowChunkBuffer.contents().advanced(by: offset * flowBytes),
                directional.contents().advanced(by: lastFlowIndex * flowBytes),
                flowBytes
              )
            }
            let previousFrame = indices[chunkStart - 1]
            let olderFrame = indices[chunkStart - 2]
            let previousFlowIndex = branch.backward
              ? previousFrame : previousFrame - 1
            inputs["state_n1"] = InferenceFunction.AsyncValue(
              unsafeBuffer: workspace.featureBuffer,
              byteOffset: previousFrame * perFrameFeatureBytes
                + outputChannel * Self.featurePlaneElements * half,
              scalarType: .float16,
              shape: [1, Self.featureChannels, Self.featureSize, Self.featureSize]
            )
            inputs["state_n2"] = InferenceFunction.AsyncValue(
              unsafeBuffer: workspace.featureBuffer,
              byteOffset: olderFrame * perFrameFeatureBytes
                + outputChannel * Self.featurePlaneElements * half,
              scalarType: .float16,
              shape: [1, Self.featureChannels, Self.featureSize, Self.featureSize]
            )
            inputs["flows"] = InferenceFunction.AsyncValue(
              unsafeBuffer: workspace.flowChunkBuffer,
              byteOffset: 0,
              scalarType: .float16,
              shape: [Self.chunkSize, 2, Self.featureSize, Self.featureSize]
            )
            inputs["flow_previous"] = InferenceFunction.AsyncValue(
              unsafeBuffer: directional,
              byteOffset: previousFlowIndex * flowBytes,
              scalarType: .float16,
              shape: [1, 2, Self.featureSize, Self.featureSize]
            )
          }
          guard let function = functions[functionName] else {
            throw MiohIPadWorkerEngineError.restorer(
              "variable function is missing: \(functionName)"
            )
          }
          var destination = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: workspace.featureChunkBuffer,
            byteOffset: 0,
            scalarType: .float16,
            shape: [Self.chunkSize, Self.featureChannels, Self.featureSize, Self.featureSize]
          )
          var outputs = InferenceFunction.AsyncMutableViews()
          outputs.insert(&destination, for: "features")
          do {
            let nativeStateNames = Set(function.descriptor.stateNames)
            if nativeStateNames.isEmpty {
              _ = try function.encode(
                inputs: inputs,
                outputViews: outputs,
                to: workspace.propagationStream
              )
            } else {
              let expectedStateNames = Set([
                "state_n1", "state_n2", "flow_previous",
              ])
              guard chunkStart > 0,
                nativeStateNames == expectedStateNames,
                let flows = inputs["flows"]
              else {
                throw MiohIPadWorkerEngineError.restorer(
                  "unsupported native state contract for \(functionName): "
                    + "\(nativeStateNames.sorted())"
                )
              }
              // Seed the mutable boundary state once from the start chunk.
              // Every later continuation chunk reuses the values Core AI
              // mutated directly, avoiding three boundary tensor copies.
              if chunkStart == Self.chunkSize {
                let previousFrame = indices[chunkStart - 1]
                let olderFrame = indices[chunkStart - 2]
                let previousFlowIndex = branch.backward
                  ? previousFrame : previousFrame - 1
                memcpy(
                  workspace.nativeStateN1Buffer.contents(),
                  workspace.featureBuffer.contents().advanced(
                    by: previousFrame * perFrameFeatureBytes
                      + outputChannel * Self.featurePlaneElements * half
                  ),
                  featureBytes
                )
                memcpy(
                  workspace.nativeStateN2Buffer.contents(),
                  workspace.featureBuffer.contents().advanced(
                    by: olderFrame * perFrameFeatureBytes
                      + outputChannel * Self.featurePlaneElements * half
                  ),
                  featureBytes
                )
                memcpy(
                  workspace.nativePreviousFlowBuffer.contents(),
                  directional.contents().advanced(
                    by: previousFlowIndex * flowBytes
                  ),
                  flowBytes
                )
              }
              var stateN1 = InferenceFunction.AsyncMutableValue(
                unsafeBuffer: workspace.nativeStateN1Buffer,
                byteOffset: 0,
                scalarType: .float16,
                shape: [1, Self.featureChannels, Self.featureSize, Self.featureSize]
              )
              var stateN2 = InferenceFunction.AsyncMutableValue(
                unsafeBuffer: workspace.nativeStateN2Buffer,
                byteOffset: 0,
                scalarType: .float16,
                shape: [1, Self.featureChannels, Self.featureSize, Self.featureSize]
              )
              var previousFlow = InferenceFunction.AsyncMutableValue(
                unsafeBuffer: workspace.nativePreviousFlowBuffer,
                byteOffset: 0,
                scalarType: .float16,
                shape: [1, 2, Self.featureSize, Self.featureSize]
              )
              var states = InferenceFunction.AsyncMutableViews()
              states.insert(&stateN1, for: "state_n1")
              states.insert(&stateN2, for: "state_n2")
              states.insert(&previousFlow, for: "flow_previous")
              _ = try function.encode(
                inputs: ["contexts": context, "flows": flows],
                states: states,
                outputViews: outputs,
                to: workspace.propagationStream
              )
            }
          } catch {
            throw Self.stageFailure(
              "propagation encode \(functionName) chunk "
                + "\(chunkStart / Self.chunkSize)",
              error: error
            )
          }
          await workspace.propagationStream.currentWorkCompleted()
          for offset in 0..<valid {
            memcpy(
              workspace.featureBuffer.contents().advanced(
                by: chunkIndices[offset] * perFrameFeatureBytes
                  + outputChannel * Self.featurePlaneElements * half
              ),
              workspace.featureChunkBuffer.contents().advanced(
                by: offset * featureBytes
              ),
              featureBytes
            )
          }
        }
      }

      for start in stride(from: 0, to: frameCount, by: Self.chunkSize) {
        let valid = min(Self.chunkSize, frameCount - start)
        let indices = (0..<Self.chunkSize).map {
          min(start + $0, frameCount - 1)
        }
        packFrames(
          indices: indices,
          frameCount: frameCount,
          destination: workspace.chunkFrameBuffer
        )
        for offset in 0..<Self.chunkSize {
          memcpy(
            workspace.contextBuffer.contents().advanced(
              by: offset * perFrameFeatureBytes
            ),
            workspace.featureBuffer.contents().advanced(
              by: indices[offset] * perFrameFeatureBytes
            ),
            perFrameFeatureBytes
          )
        }
        let frames = InferenceFunction.AsyncValue(
          unsafeBuffer: workspace.chunkFrameBuffer,
          byteOffset: 0,
          scalarType: .float16,
          shape: [Self.chunkSize, 3, Self.imageSize, Self.imageSize]
        )
        let features = InferenceFunction.AsyncValue(
          unsafeBuffer: workspace.contextBuffer,
          byteOffset: 0,
          scalarType: .float16,
          shape: [Self.chunkSize, Self.fusedChannels, Self.featureSize, Self.featureSize]
        )
        var destination = InferenceFunction.AsyncMutableValue(
          unsafeBuffer: workspace.restoredChunkBuffer,
          byteOffset: 0,
          scalarType: .float16,
          shape: [Self.chunkSize, 3, Self.imageSize, Self.imageSize]
        )
        var outputs = InferenceFunction.AsyncMutableViews()
        outputs.insert(&destination, for: "restored")
        do {
          _ = try reconstruction.encode(
            inputs: ["frames": frames, "features": features],
            outputViews: outputs,
            to: workspace.reconstructionStream
          )
        } catch {
          throw Self.stageFailure(
            "reconstruction encode chunk \(start / Self.chunkSize)",
            error: error
          )
        }
        await workspace.reconstructionStream.currentWorkCompleted()
        memcpy(
          workspace.restoredBuffer.contents().advanced(by: start * frameBytes),
          workspace.restoredChunkBuffer.contents(),
          valid * frameBytes
        )
      }
    }

    private func packFrames(
      indices: [Int],
      frameCount: Int,
      destination: MTLBuffer
    ) {
      let frameBytes = Self.frameElements * MemoryLayout<Float16>.stride
      for (offset, index) in indices.enumerated() {
        precondition(index >= 0 && index < frameCount)
        memcpy(
          destination.contents().advanced(by: offset * frameBytes),
          workspace.frameBuffer.contents().advanced(by: index * frameBytes),
          frameBytes
        )
      }
    }

    private static func stageFailure(
      _ stage: String,
      error: Error
    ) -> MiohIPadWorkerEngineError {
      let value = error as NSError
      return .restorer(
        "可変長BasicVSR++ \(stage): \(error.localizedDescription) "
          + "[\(value.domain):\(value.code)]"
      )
    }
  }
#endif
