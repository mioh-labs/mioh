import AppKit
import Darwin
import Foundation
import WebKit

@MainActor
private final class BridgeMessageHandler: NSObject, WKScriptMessageHandler {
  private(set) var receivedPayload: String?

  func userContentController(
    _ userContentController: WKUserContentController,
    didReceive message: WKScriptMessage
  ) {
    guard message.name == "miohBridgeProbe",
      message.world.name == "com.mioh-labs.bridge-probe",
      let payload = message.body as? String
    else { return }
    receivedPayload = payload
  }
}

@main
@MainActor
private enum WKContentWorldBridgeHarness {
  static func main() {
    _ = NSApplication.shared
    NSApp.setActivationPolicy(.prohibited)

    let handler = BridgeMessageHandler()
    let contentWorld = WKContentWorld.world(
      name: "com.mioh-labs.bridge-probe"
    )
    let controller = WKUserContentController()
    controller.add(
      handler,
      contentWorld: contentWorld,
      name: "miohBridgeProbe"
    )
    controller.addUserScript(
      WKUserScript(
        source: """
          document.addEventListener('mioh-hls-bridge-probe', event => {
            if (typeof event.detail !== 'string') return;
            window.webkit.messageHandlers.miohBridgeProbe.postMessage(event.detail);
          }, true);
          """,
        injectionTime: .atDocumentStart,
        forMainFrameOnly: true,
        in: contentWorld
      )
    )

    let configuration = WKWebViewConfiguration()
    configuration.userContentController = controller
    let webView = WKWebView(frame: .zero, configuration: configuration)
    webView.loadHTMLString(
      """
      <!doctype html><meta charset="utf-8">
      <script>
        document.dispatchEvent(new CustomEvent('mioh-hls-bridge-probe', {
          detail: JSON.stringify({kind: 'page-fetch-hls-response', ok: true})
        }));
      </script>
      """,
      baseURL: URL(string: "https://bridge-probe.invalid/")
    )

    let deadline = Date().addingTimeInterval(10)
    while handler.receivedPayload == nil, Date() < deadline {
      RunLoop.main.run(until: Date().addingTimeInterval(0.05))
    }

    guard let payload = handler.receivedPayload,
      payload.contains("page-fetch-hls-response"),
      payload.contains("\"ok\":true")
    else {
      FileHandle.standardError.write(
        Data("WKContentWorld bridge probe failed\n".utf8)
      )
      exit(1)
    }
    print("WKContentWorld bridge probe passed")
  }
}
