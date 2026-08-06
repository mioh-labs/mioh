import Foundation

@main
struct RemoteControlSettingsHarness {
  @MainActor
  static func main() async {
    let capabilities = PlatformCapabilities(
      operatingSystemVersion: OperatingSystemVersion(
        majorVersion: 27,
        minorVersion: 0,
        patchVersion: 0
      )
    )
    let runner = RestorationRunner(capabilities: capabilities)
    let player = RealtimePlayerController()
    let cluster = MiohClusterController()
    cluster.attach(runner: runner)
    let server = RemoteControlServer()
    server.attach(runner: runner, player: player)
    server.attachCluster(cluster)
    server.port = 18_991
    server.setEnabled(true)

    let deadline = Date().addingTimeInterval(5)
    while server.urls.isEmpty && Date() < deadline {
      try? await Task.sleep(nanoseconds: 50_000_000)
    }
    guard !server.urls.isEmpty else {
      FileHandle.standardError.write(Data("server failed\n".utf8))
      exit(2)
    }
    let line = "READY \(server.port) \(server.token)\n"
    FileHandle.standardOutput.write(Data(line.utf8))
    try? await Task.sleep(nanoseconds: 120_000_000_000)
    server.setEnabled(false)
  }
}
