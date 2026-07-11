import AppKit
import CoreAI
import SwiftUI

@MainActor
final class RestorationRunner: ObservableObject {
  @Published var inputURL: URL?
  @Published var outputURL: URL?
  @Published var progress = 0.0
  @Published var status = "待機中"
  @Published var log = ""
  @Published var isRunning = false

  private var process: Process?
  private var outputBuffer = ""

  var canStart: Bool {
    inputURL != nil && outputURL != nil && !isRunning
  }

  func chooseInput() {
    let panel = NSOpenPanel()
    panel.title = "入力を選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    if panel.runModal() == .OK {
      inputURL = panel.url
    }
  }

  func chooseOutput() {
    let panel = NSOpenPanel()
    panel.title = "出力フォルダを選択"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    if panel.runModal() == .OK {
      outputURL = panel.url
    }
  }

  func start() {
    guard let inputURL, let outputURL else { return }
    do {
      let resources = try resourceDirectory()
      let python = resources.appendingPathComponent("runtime/bin/python3.12")
      let restorationModel = try restorationModel(in: resources)
      let detectionModel = resources.appendingPathComponent(
        "models/lada_mosaic_detection_model_v2.mlpackage"
      )

      guard FileManager.default.isExecutableFile(atPath: python.path) else {
        throw RunnerError.missingResource("Python runtime")
      }
      guard FileManager.default.fileExists(atPath: detectionModel.path) else {
        throw RunnerError.missingResource("Detection model")
      }

      let task = Process()
      task.executableURL = python
      task.currentDirectoryURL = resources
      task.arguments = [
        "-m", "lada.cli.main",
        "--input", inputURL.path,
        "--output", outputURL.path,
        "--device", "mps",
        "--fp16",
        "--encoding-preset", "hevc-apple-gpu-balanced",
        "--mosaic-restoration-model", restorationModel.path,
        "--max-clip-length", "178",
        "--mosaic-detection-model", detectionModel.path,
        "--mosaic-detection-empty-lookahead", "10",
        "--no-detect-face-mosaics",
      ]

      var environment = ProcessInfo.processInfo.environment
      environment["PYTHONHOME"] = resources.appendingPathComponent("runtime").path
      environment["PYTHONPATH"] =
        resources.appendingPathComponent(
          "runtime/lib/python3.12/site-packages"
        ).path
      environment["LADA_MODEL_WEIGHTS_DIR"] = resources.appendingPathComponent("models").path
      environment["LADA_COREAI_SWIFT_RUNNER"] =
        resources.appendingPathComponent(
          "bin/lada-coreai-runner"
        ).path
      environment["PATH"] = [
        resources.appendingPathComponent("bin").path,
        "/usr/bin", "/bin", "/usr/sbin", "/sbin",
      ].joined(separator: ":")
      environment["PYTHONUNBUFFERED"] = "1"
      environment["PYTHONDONTWRITEBYTECODE"] = "1"
      environment["PYTHONWARNINGS"] = "ignore::SyntaxWarning"
      environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
      environment["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
      task.environment = environment

      let pipe = Pipe()
      task.standardOutput = pipe
      task.standardError = pipe
      pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
        let data = handle.availableData
        guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
        Task { @MainActor in self?.consume(text) }
      }

      task.terminationHandler = { [weak self] completed in
        pipe.fileHandleForReading.readabilityHandler = nil
        Task { @MainActor in
          guard let self else { return }
          self.isRunning = false
          self.process = nil
          if completed.terminationStatus == 0 {
            self.progress = 1
            self.status = "完了"
          } else if completed.terminationReason == .uncaughtSignal {
            self.status = "停止"
          } else {
            self.status = "エラー"
          }
        }
      }

      outputBuffer = ""
      log = ""
      progress = 0
      status = "準備中"
      isRunning = true
      process = task
      try task.run()
    } catch {
      isRunning = false
      status = "エラー"
      appendLog(error.localizedDescription + "\n")
    }
  }

  func stop() {
    process?.terminate()
    status = "停止中"
  }

  func revealOutput() {
    guard let outputURL else { return }
    NSWorkspace.shared.activateFileViewerSelecting([outputURL])
  }

  private func consume(_ text: String) {
    outputBuffer += text
    appendLog(text.replacingOccurrences(of: "\r", with: "\n"))
    let range = NSRange(outputBuffer.startIndex..., in: outputBuffer)
    let regex = try? NSRegularExpression(
      pattern: #"(?:Processing video|ビデオの処理中):\s+(\d+)%"#
    )
    if let match = regex?.matches(in: outputBuffer, range: range).last,
      let percentRange = Range(match.range(at: 1), in: outputBuffer),
      let percent = Double(outputBuffer[percentRange])
    {
      progress = min(max(percent / 100, 0), 1)
      status = "処理中 \(Int(percent))%"
    }
    if outputBuffer.count > 40_000 {
      outputBuffer = String(outputBuffer.suffix(20_000))
    }
  }

  private func appendLog(_ text: String) {
    log += text
    if log.count > 30_000 {
      log = String(log.suffix(20_000))
    }
  }

  private func resourceDirectory() throws -> URL {
    guard let resources = Bundle.main.resourceURL else {
      throw RunnerError.missingResource("App resources")
    }
    return resources
  }

  private func restorationModel(in resources: URL) throws -> URL {
    let models = resources.appendingPathComponent("models")
    let architecture = AIModel.deviceArchitectureName
    let candidates = try FileManager.default.contentsOfDirectory(
      at: models,
      includingPropertiesForKeys: nil
    )
    if let compiled = candidates.first(where: {
      $0.pathExtension == "aimodelc"
        && $0.lastPathComponent.contains(".\(architecture).aimodelc")
        && $0.lastPathComponent.contains("t90")
    }) {
      return compiled
    }
    if let source = candidates.first(where: {
      $0.pathExtension == "aimodel" && $0.lastPathComponent.contains("t90")
    }) {
      return source
    }
    throw RunnerError.missingResource("Core AI T90 model for \(architecture)")
  }
}

enum RunnerError: LocalizedError {
  case missingResource(String)

  var errorDescription: String? {
    switch self {
    case .missingResource(let name):
      return "必要なリソースが見つかりません: \(name)"
    }
  }
}

struct PathRow: View {
  let title: String
  let icon: String
  let url: URL?
  let action: () -> Void

  var body: some View {
    HStack(spacing: 12) {
      Image(systemName: icon)
        .frame(width: 20)
        .foregroundStyle(.secondary)
      VStack(alignment: .leading, spacing: 3) {
        Text(title).font(.caption).foregroundStyle(.secondary)
        Text(url?.path ?? "未選択")
          .lineLimit(1)
          .truncationMode(.middle)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      Button(action: action) {
        Image(systemName: "folder")
      }
      .buttonStyle(.borderless)
      .help("選択")
    }
    .frame(minHeight: 48)
  }
}

struct ContentView: View {
  @StateObject private var runner = RestorationRunner()

  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 12) {
        Image(nsImage: NSImage(named: "AppIcon") ?? NSImage())
          .resizable()
          .frame(width: 34, height: 34)
        VStack(alignment: .leading, spacing: 2) {
          Text("Lada").font(.title2.weight(.semibold))
          Text("Core AI T90").font(.caption).foregroundStyle(.secondary)
        }
        Spacer()
        Text(runner.status)
          .font(.callout.monospacedDigit())
          .foregroundStyle(runner.status == "エラー" ? .red : .secondary)
      }
      .padding(.horizontal, 20)
      .frame(height: 70)

      Divider()

      VStack(spacing: 8) {
        PathRow(
          title: "入力",
          icon: "film",
          url: runner.inputURL,
          action: runner.chooseInput
        )
        Divider()
        PathRow(
          title: "出力",
          icon: "externaldrive",
          url: runner.outputURL,
          action: runner.chooseOutput
        )
      }
      .padding(.horizontal, 20)
      .padding(.vertical, 10)

      Divider()

      HStack(spacing: 18) {
        Label("T90", systemImage: "square.stack.3d.up")
        Label("178 frames", systemImage: "timeline.selection")
        Label("HEVC", systemImage: "video")
        Spacer()
      }
      .font(.callout)
      .foregroundStyle(.secondary)
      .padding(.horizontal, 20)
      .frame(height: 48)

      ProgressView(value: runner.progress)
        .progressViewStyle(.linear)
        .padding(.horizontal, 20)

      ScrollView {
        Text(runner.log.isEmpty ? " " : runner.log)
          .font(.system(.caption, design: .monospaced))
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .topLeading)
          .padding(10)
      }
      .background(Color(nsColor: .textBackgroundColor))
      .padding(20)
      .frame(maxHeight: .infinity)

      Divider()

      HStack(spacing: 10) {
        Button(action: runner.revealOutput) {
          Image(systemName: "folder.badge.gearshape")
        }
        .help("出力をFinderで表示")
        .disabled(runner.outputURL == nil)

        Spacer()

        if runner.isRunning {
          Button(role: .destructive, action: runner.stop) {
            Label("停止", systemImage: "stop.fill")
          }
        } else {
          Button(action: runner.start) {
            Label("開始", systemImage: "play.fill")
          }
          .buttonStyle(.borderedProminent)
          .disabled(!runner.canStart)
        }
      }
      .padding(.horizontal, 20)
      .frame(height: 62)
    }
    .frame(minWidth: 720, minHeight: 540)
  }
}

@main
struct LadaStandaloneApp: App {
  var body: some Scene {
    WindowGroup {
      ContentView()
    }
    .windowResizability(.contentMinSize)
    .defaultSize(width: 760, height: 620)
  }
}
