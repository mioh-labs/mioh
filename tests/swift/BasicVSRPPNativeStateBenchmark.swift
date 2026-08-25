// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

import CoreAI
import Foundation
import Metal

private let chunkSize = 6
private let featureChannels = 64
private let contextChannels = 256
private let featureSize = 64
private let featurePlane = featureSize * featureSize
private let stateElements = featureChannels * featurePlane
private let contextElements = chunkSize * contextChannels * featurePlane
private let flowElements = chunkSize * 2 * featurePlane
private let previousFlowElements = 2 * featurePlane
private let outputElements = chunkSize * stateElements

enum BenchmarkError: LocalizedError {
  case invalidArguments
  case metalUnavailable
  case allocationFailed
  case missingFunction(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments:
      return "usage: basicvsrpp-native-state-benchmark <shipping.aimodelc> <control.aimodelc> <stateful.aimodelc>"
    case .metalUnavailable: return "Metal device is unavailable"
    case .allocationFailed: return "Metal buffer allocation failed"
    case .missingFunction(let path): return "Core AI main function is missing: \(path)"
    }
  }
}

private struct Buffers {
  let contexts: MTLBuffer
  let flows: MTLBuffer
  let seedN1: MTLBuffer
  let seedN2: MTLBuffer
  let seedPreviousFlow: MTLBuffer
  let shippingOutput: MTLBuffer
  let controlOutput: MTLBuffer
  let statefulN1: MTLBuffer
  let statefulN2: MTLBuffer
  let statefulPreviousFlow: MTLBuffer
  let statefulOutput: MTLBuffer

  init(device: MTLDevice) throws {
    func make(_ elements: Int) throws -> MTLBuffer {
      guard let buffer = device.makeBuffer(
        length: elements * MemoryLayout<Float16>.stride,
        options: .storageModeShared)
      else { throw BenchmarkError.allocationFailed }
      memset(buffer.contents(), 0, buffer.length)
      return buffer
    }
    contexts = try make(contextElements)
    flows = try make(flowElements)
    seedN1 = try make(stateElements)
    seedN2 = try make(stateElements)
    seedPreviousFlow = try make(previousFlowElements)
    shippingOutput = try make(outputElements)
    controlOutput = try make(outputElements)
    statefulN1 = try make(stateElements)
    statefulN2 = try make(stateElements)
    statefulPreviousFlow = try make(previousFlowElements)
    statefulOutput = try make(outputElements)

    fill(contexts, scale: 0.2, phase: 3)
    fill(flows, scale: 0.08, phase: 17)
    fill(seedN1, scale: 0.1, phase: 29)
    fill(seedN2, scale: 0.1, phase: 43)
    fill(seedPreviousFlow, scale: 0.08, phase: 59)
    resetStateful()
  }

  func resetStateful() {
    memcpy(statefulN1.contents(), seedN1.contents(), seedN1.length)
    memcpy(statefulN2.contents(), seedN2.contents(), seedN2.length)
    memcpy(
      statefulPreviousFlow.contents(), seedPreviousFlow.contents(),
      seedPreviousFlow.length)
  }

  private func fill(_ buffer: MTLBuffer, scale: Float, phase: Int) {
    let pointer = buffer.contents().bindMemory(
      to: Float16.self, capacity: buffer.length / MemoryLayout<Float16>.stride)
    for index in 0..<(buffer.length / MemoryLayout<Float16>.stride) {
      let centered = Float((index * 37 + phase) % 257) / 128.0 - 1.0
      pointer[index] = Float16(centered * scale)
    }
  }
}

@main
struct BasicVSRPPNativeStateBenchmark {
  static func main() async throws {
    if CommandLine.arguments.count == 4,
      CommandLine.arguments[1] == "--timing"
    {
      try await runIsolatedTiming(
        mode: CommandLine.arguments[2], path: CommandLine.arguments[3])
      return
    }
    guard CommandLine.arguments.count == 4 else {
      throw BenchmarkError.invalidArguments
    }
    let shippingPath = CommandLine.arguments[1]
    let controlPath = CommandLine.arguments[2]
    let statefulPath = CommandLine.arguments[3]
    let shipping = try await loadFunction(shippingPath)
    let control = try await loadFunction(controlPath)
    let stateful = try await loadFunction(statefulPath)
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw BenchmarkError.metalUnavailable
    }
    let buffers = try Buffers(device: device)
    let shippingStream = ComputeStream()
    let controlStream = ComputeStream()
    let statefulStream = ComputeStream()

    for _ in 0..<5 {
      _ = try await runExplicit(
        shipping, buffers: buffers, output: buffers.shippingOutput,
        stream: shippingStream)
      _ = try await runExplicit(
        control, buffers: buffers, output: buffers.controlOutput,
        stream: controlStream)
      buffers.resetStateful()
      _ = try await runStateful(
        stateful, buffers: buffers, stream: statefulStream)
    }

    buffers.resetStateful()
    _ = try await runExplicit(
      shipping, buffers: buffers, output: buffers.shippingOutput,
      stream: shippingStream)
    _ = try await runExplicit(
      control, buffers: buffers, output: buffers.controlOutput,
      stream: controlStream)
    _ = try await runStateful(stateful, buffers: buffers, stream: statefulStream)
    let shippingControlMetrics = compare(
      buffers.shippingOutput, buffers.controlOutput, count: outputElements)
    let outputMetrics = compare(
      buffers.controlOutput, buffers.statefulOutput, count: outputElements)
    let n1Metrics = compareSlice(
      buffers.controlOutput, offset: 5 * stateElements,
      buffers.statefulN1, count: stateElements)
    let n2Metrics = compareSlice(
      buffers.controlOutput, offset: 4 * stateElements,
      buffers.statefulN2, count: stateElements)
    let flowMetrics = compareSlice(
      buffers.flows, offset: 5 * previousFlowElements,
      buffers.statefulPreviousFlow, count: previousFlowElements)

    var shippingTimes: [Double] = []
    var controlTimes: [Double] = []
    var statefulTimes: [Double] = []
    for index in 0..<40 {
      shippingTimes.append(
        try await runExplicit(
          shipping, buffers: buffers, output: buffers.shippingOutput,
          stream: shippingStream))
      if index.isMultiple(of: 2) {
        controlTimes.append(
          try await runExplicit(
            control, buffers: buffers, output: buffers.controlOutput,
            stream: controlStream))
        buffers.resetStateful()
        statefulTimes.append(
          try await runStateful(
            stateful, buffers: buffers, stream: statefulStream))
      } else {
        buffers.resetStateful()
        statefulTimes.append(
          try await runStateful(
            stateful, buffers: buffers, stream: statefulStream))
        controlTimes.append(
          try await runExplicit(
            control, buffers: buffers, output: buffers.controlOutput,
            stream: controlStream))
      }
    }
    let shippingMedian = median(shippingTimes)
    let controlMedian = median(controlTimes)
    let statefulMedian = median(statefulTimes)
    let document: [String: Any] = [
      "shippingMedianMilliseconds": shippingMedian,
      "controlMedianMilliseconds": controlMedian,
      "statefulMedianMilliseconds": statefulMedian,
      "statefulSpeedup": controlMedian / statefulMedian,
      "timingOrder": "alternating-control-stateful-stateful-control",
      "shippingVsControl": shippingControlMetrics,
      "controlVsStateful": [
        "output": outputMetrics,
        "stateN1": n1Metrics,
        "stateN2": n2Metrics,
        "flowPrevious": flowMetrics,
      ],
    ]
    let data = try JSONSerialization.data(
      withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
  }

  private static func runIsolatedTiming(mode: String, path: String) async throws {
    let function = try await loadFunction(path)
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw BenchmarkError.metalUnavailable
    }
    let buffers = try Buffers(device: device)
    let stream = ComputeStream()
    for _ in 0..<10 {
      if mode == "control" {
        _ = try await runExplicit(
          function, buffers: buffers, output: buffers.controlOutput, stream: stream)
      } else if mode == "stateful" {
        buffers.resetStateful()
        _ = try await runStateful(function, buffers: buffers, stream: stream)
      } else {
        throw BenchmarkError.invalidArguments
      }
    }
    var times: [Double] = []
    for _ in 0..<100 {
      if mode == "control" {
        times.append(
          try await runExplicit(
            function, buffers: buffers, output: buffers.controlOutput, stream: stream))
      } else {
        buffers.resetStateful()
        times.append(
          try await runStateful(function, buffers: buffers, stream: stream))
      }
    }
    let document: [String: Any] = [
      "mode": mode,
      "medianMilliseconds": median(times),
      "p10Milliseconds": percentile(times, fraction: 0.10),
      "p90Milliseconds": percentile(times, fraction: 0.90),
      "runs": times.count,
    ]
    let data = try JSONSerialization.data(
      withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
  }

  private static func loadFunction(_ path: String) async throws -> InferenceFunction {
    let model = try await AIModel(contentsOf: URL(fileURLWithPath: path))
    guard let function = try model.loadFunction(named: "main") else {
      throw BenchmarkError.missingFunction(path)
    }
    return function
  }

  private static func runExplicit(
    _ function: InferenceFunction, buffers: Buffers, output outputBuffer: MTLBuffer,
    stream: ComputeStream
  ) async throws -> Double {
    let contexts = asyncInput(
      buffers.contexts, shape: [chunkSize, contextChannels, featureSize, featureSize])
    let n1 = asyncInput(
      buffers.seedN1, shape: [1, featureChannels, featureSize, featureSize])
    let n2 = asyncInput(
      buffers.seedN2, shape: [1, featureChannels, featureSize, featureSize])
    let flows = asyncInput(
      buffers.flows, shape: [chunkSize, 2, featureSize, featureSize])
    let previousFlow = asyncInput(
      buffers.seedPreviousFlow, shape: [1, 2, featureSize, featureSize])
    var output = InferenceFunction.AsyncMutableValue(
      unsafeBuffer: outputBuffer, byteOffset: 0, scalarType: .float16,
      shape: [chunkSize, featureChannels, featureSize, featureSize])
    var outputs = InferenceFunction.AsyncMutableViews()
    outputs.insert(&output, for: "features")
    let started = DispatchTime.now().uptimeNanoseconds
    _ = try function.encode(
      inputs: [
        "contexts": contexts, "state_n1": n1, "state_n2": n2,
        "flows": flows, "flow_previous": previousFlow,
      ], outputViews: outputs, to: stream)
    await stream.currentWorkCompleted()
    return Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000.0
  }

  private static func runStateful(
    _ function: InferenceFunction, buffers: Buffers, stream: ComputeStream
  ) async throws -> Double {
    let contexts = asyncInput(
      buffers.contexts, shape: [chunkSize, contextChannels, featureSize, featureSize])
    let flows = asyncInput(
      buffers.flows, shape: [chunkSize, 2, featureSize, featureSize])
    var n1 = asyncOutput(
      buffers.statefulN1, shape: [1, featureChannels, featureSize, featureSize])
    var n2 = asyncOutput(
      buffers.statefulN2, shape: [1, featureChannels, featureSize, featureSize])
    var previousFlow = asyncOutput(
      buffers.statefulPreviousFlow, shape: [1, 2, featureSize, featureSize])
    var states = InferenceFunction.AsyncMutableViews()
    states.insert(&n1, for: "state_n1")
    states.insert(&n2, for: "state_n2")
    states.insert(&previousFlow, for: "flow_previous")
    var output = asyncOutput(
      buffers.statefulOutput,
      shape: [chunkSize, featureChannels, featureSize, featureSize])
    var outputs = InferenceFunction.AsyncMutableViews()
    outputs.insert(&output, for: "features")
    let started = DispatchTime.now().uptimeNanoseconds
    _ = try function.encode(
      inputs: ["contexts": contexts, "flows": flows], states: states,
      outputViews: outputs, to: stream)
    await stream.currentWorkCompleted()
    return Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000.0
  }

  private static func asyncInput(
    _ buffer: MTLBuffer, shape: [Int]
  ) -> InferenceFunction.AsyncValue {
    InferenceFunction.AsyncValue(
      unsafeBuffer: buffer, byteOffset: 0, scalarType: .float16, shape: shape)
  }

  private static func asyncOutput(
    _ buffer: MTLBuffer, shape: [Int]
  ) -> InferenceFunction.AsyncMutableValue {
    InferenceFunction.AsyncMutableValue(
      unsafeBuffer: buffer, byteOffset: 0, scalarType: .float16, shape: shape)
  }

  private static func compare(
    _ lhs: MTLBuffer, _ rhs: MTLBuffer, count: Int
  ) -> [String: Double] {
    compareSlice(lhs, offset: 0, rhs, count: count)
  }

  private static func compareSlice(
    _ lhs: MTLBuffer, offset: Int, _ rhs: MTLBuffer, count: Int
  ) -> [String: Double] {
    let a = lhs.contents().bindMemory(
      to: Float16.self, capacity: offset + count).advanced(by: offset)
    let b = rhs.contents().bindMemory(to: Float16.self, capacity: count)
    var maximum = 0.0
    var sum = 0.0
    var squareSum = 0.0
    for index in 0..<count {
      let difference = abs(Double(Float(a[index]) - Float(b[index])))
      maximum = max(maximum, difference)
      sum += difference
      squareSum += difference * difference
    }
    return [
      "maxAbsoluteError": maximum,
      "meanAbsoluteError": sum / Double(count),
      "rmse": sqrt(squareSum / Double(count)),
    ]
  }

  private static func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
      return (sorted[middle - 1] + sorted[middle]) * 0.5
    }
    return sorted[middle]
  }

  private static func percentile(
    _ values: [Double], fraction: Double
  ) -> Double {
    let sorted = values.sorted()
    let index = min(
      sorted.count - 1,
      max(0, Int((Double(sorted.count - 1) * fraction).rounded())))
    return sorted[index]
  }
}
