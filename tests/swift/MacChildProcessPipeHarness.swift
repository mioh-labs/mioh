import Darwin
import Foundation

private enum HarnessError: Error, CustomStringConvertible {
  case assertion(String)

  var description: String {
    switch self {
    case .assertion(let message): message
    }
  }
}

private func require(
  _ condition: @autoclosure () -> Bool,
  _ message: String
) throws {
  guard condition() else { throw HarnessError.assertion(message) }
}

private func command(_ name: String, sequence: Int) -> Data {
  var data = Data("{\"command\":\"\(name)\",\"sequence\":\(sequence)}".utf8)
  data.append(0x0A)
  return data
}

private func consumeOnePipeWrite(_ handle: FileHandle) {
  var buffer = [UInt8](repeating: 0, count: 256)
  _ = Darwin.read(handle.fileDescriptor, &buffer, buffer.count)
}

private final class RaceProbeState: @unchecked Sendable {
  private let lock = NSLock()
  private var output = Data()
  private var successes = 0
  private var failures = 0

  func setOutput(_ data: Data) {
    lock.lock()
    output = data
    lock.unlock()
  }

  func recordWrite(_ succeeded: Bool) {
    lock.lock()
    if succeeded {
      successes += 1
    } else {
      failures += 1
    }
    lock.unlock()
  }

  func snapshot() -> (output: Data, successes: Int, failures: Int) {
    lock.lock()
    defer { lock.unlock() }
    return (output, successes, failures)
  }
}

/// Closing the only read endpoint is the minimal regression for the real
/// failure: Darwin would normally deliver SIGPIPE before FileHandle could
/// surface EPIPE as a Swift Error. Reaching the assertions after `write` is
/// therefore part of the contract, not just its Boolean result.
private func verifyClosedReadEndpoint() throws {
  let pipe = Pipe()
  let writer = pipe.fileHandleForWriting
  try require(MacChildProcessPipe.prepare(writer), "could not prepare pipe writer")
  try pipe.fileHandleForReading.close()

  try require(
    !MacChildProcessPipe.write(command("release_through", sequence: 1), to: writer),
    "write unexpectedly succeeded after the read endpoint closed"
  )
  try require(
    !MacChildProcessPipe.write(command("release_through", sequence: 2), to: writer),
    "a repeated write unexpectedly succeeded after EPIPE"
  )
  try? writer.close()
}

/// Exercises the same Foundation `Process.standardInput` ownership path used by
/// the app, rather than only a manually closed Pipe. Once `/usr/bin/true`
/// exits, its stdin reader is gone and a late controller command must surface
/// as a failed write without delivering SIGPIPE to the harness process.
private func verifyExitedChildProcessStdin() throws {
  let process = Process()
  let input = Pipe()
  process.executableURL = URL(fileURLWithPath: "/usr/bin/true")
  process.standardInput = input
  let writer = input.fileHandleForWriting
  try require(
    MacChildProcessPipe.prepare(writer),
    "could not prepare child-process stdin"
  )
  try process.run()
  process.waitUntilExit()
  try require(process.terminationStatus == 0, "fake child did not exit cleanly")
  try require(
    !MacChildProcessPipe.write(command("release_through", sequence: 9), to: writer),
    "write unexpectedly succeeded after child process exit"
  )
  try? writer.close()
}

/// Models the HLS producer race which caused the app to disappear just after
/// entering `playing`: stdout can still contain finalized-segment events while
/// the worker has already closed stdin. The player drains those events and
/// sends `release_through` commands concurrently.
private func verifyStdoutDrainAndReleaseRace() throws {
  let workerInput = Pipe()
  let workerOutput = Pipe()
  let writer = workerInput.fileHandleForWriting
  try require(MacChildProcessPipe.prepare(writer), "could not prepare fake worker stdin")

  let workerReady = DispatchSemaphore(value: 0)
  let workerClosedInput = DispatchSemaphore(value: 0)
  let workerFinished = DispatchSemaphore(value: 0)
  let drainFinished = DispatchSemaphore(value: 0)
  let writerFinished = DispatchSemaphore(value: 0)
  let probeState = RaceProbeState()

  DispatchQueue.global(qos: .userInitiated).async {
    let data = workerOutput.fileHandleForReading.readDataToEndOfFile()
    probeState.setOutput(data)
    drainFinished.signal()
  }

  DispatchQueue.global(qos: .userInitiated).async {
    // Consume the initial command so at least one parent write is known to
    // have crossed the pipe before the simulated worker teardown.
    consumeOnePipeWrite(workerInput.fileHandleForReading)
    workerReady.signal()
    for sequence in 0..<128 {
      let event = Data("{\"kind\":\"segment\",\"sequence\":\(sequence)}\n".utf8)
      try? workerOutput.fileHandleForWriting.write(contentsOf: event)
      if sequence == 7 {
        try? workerInput.fileHandleForReading.close()
        workerClosedInput.signal()
      }
      usleep(250)
    }
    try? workerOutput.fileHandleForWriting.close()
    workerFinished.signal()
  }

  try require(
    MacChildProcessPipe.write(command("start", sequence: 0), to: writer),
    "initial fake worker command failed"
  )
  try require(
    workerReady.wait(timeout: .now() + 2) == .success,
    "fake worker did not consume the initial command"
  )

  DispatchQueue.global(qos: .userInitiated).async {
    for sequence in 0..<512 {
      let succeeded = MacChildProcessPipe.write(
        command("release_through", sequence: sequence),
        to: writer
      )
      probeState.recordWrite(succeeded)
      usleep(100)
    }
    writerFinished.signal()
  }

  try require(
    workerClosedInput.wait(timeout: .now() + 2) == .success,
    "fake worker did not close stdin"
  )
  try require(
    writerFinished.wait(timeout: .now() + 4) == .success,
    "release writer did not settle after worker stdin closed"
  )
  try require(
    workerFinished.wait(timeout: .now() + 2) == .success,
    "fake worker did not finish stdout"
  )
  try require(
    drainFinished.wait(timeout: .now() + 2) == .success,
    "stdout drain did not reach EOF"
  )

  let result = probeState.snapshot()
  let outputText = String(decoding: result.output, as: UTF8.self)
  try require(outputText.contains("\"sequence\":0"), "stdout lost its first event")
  try require(outputText.contains("\"sequence\":127"), "stdout lost its final event")
  try require(result.successes > 0, "race never exercised a live worker write")
  try require(result.failures > 0, "race never exercised EPIPE after worker exit")
  try? writer.close()
}

/// Models cancellation after one accepted command. Once cancellation closes
/// the worker's read endpoint, a late acknowledgement must become `false`
/// rather than terminating the parent process with SIGPIPE.
private func verifyWriteAfterCancellation() throws {
  let pipe = Pipe()
  let writer = pipe.fileHandleForWriting
  try require(MacChildProcessPipe.prepare(writer), "could not prepare cancellation pipe")

  let consumed = DispatchSemaphore(value: 0)
  DispatchQueue.global(qos: .userInitiated).async {
    consumeOnePipeWrite(pipe.fileHandleForReading)
    try? pipe.fileHandleForReading.close()
    consumed.signal()
  }

  try require(
    MacChildProcessPipe.write(command("release_through", sequence: 3), to: writer),
    "pre-cancellation write failed"
  )
  try require(
    consumed.wait(timeout: .now() + 2) == .success,
    "fake cancelled worker did not close stdin"
  )
  try require(
    !MacChildProcessPipe.write(command("release_through", sequence: 4), to: writer),
    "post-cancellation write unexpectedly succeeded"
  )
  try? writer.close()
}

@main
private struct MacChildProcessPipeHarness {
  static func main() {
    do {
      try verifyClosedReadEndpoint()
      try verifyExitedChildProcessStdin()
      try verifyStdoutDrainAndReleaseRace()
      try verifyWriteAfterCancellation()
      print("ok mac child pipe SIGPIPE regression")
    } catch {
      FileHandle.standardError.write(Data("pipe harness failed: \(error)\n".utf8))
      exit(1)
    }
  }
}
