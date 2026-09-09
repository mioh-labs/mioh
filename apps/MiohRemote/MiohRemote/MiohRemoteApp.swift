import SwiftUI

@main
struct MiohRemoteApp: App {
  @StateObject private var store = RemoteStore()
  @StateObject private var worker = IPadWorkerStore()
  @Environment(\.scenePhase) private var scenePhase

  var body: some Scene {
    WindowGroup {
      ContentView()
        .environmentObject(store)
        .environmentObject(worker)
        .onChange(of: scenePhase) { phase in
          guard phase == .background, worker.isRunning else { return }
          Task {
            await worker.stop(
              reason: "iPad Workerは前景実行専用です。アプリがバックグラウンドへ移ったため停止しました。"
            )
          }
        }
    }
  }
}
