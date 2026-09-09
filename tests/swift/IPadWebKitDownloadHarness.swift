import AppKit
import Foundation
import WebKit

private enum WebKitDownloadHarnessFailure: Error {
  case invalidArguments
  case navigation(String)
  case response(Int, String)
}

@MainActor
private final class WebKitNavigationWaiter: NSObject, WKNavigationDelegate {
  private var continuation: CheckedContinuation<Void, Error>?

  func load(_ request: URLRequest, in webView: WKWebView) async throws {
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Void, Error>) in
      self.continuation = continuation
      webView.navigationDelegate = self
      webView.load(request)
    }
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    let continuation = continuation
    self.continuation = nil
    continuation?.resume()
  }

  func webView(
    _ webView: WKWebView,
    didFail navigation: WKNavigation!,
    withError error: Error
  ) {
    fail(error)
  }

  func webView(
    _ webView: WKWebView,
    didFailProvisionalNavigation navigation: WKNavigation!,
    withError error: Error
  ) {
    fail(error)
  }

  private func fail(_ error: Error) {
    let continuation = continuation
    self.continuation = nil
    continuation?.resume(
      throwing: WebKitDownloadHarnessFailure.navigation(
        error.localizedDescription
      )
    )
  }
}

@main
private struct IPadWebKitDownloadHarness {
  @MainActor
  static func main() async throws {
    guard CommandLine.arguments.count == 3,
      let pageURL = URL(string: CommandLine.arguments[1]),
      let resourceURL = URL(string: CommandLine.arguments[2])
    else { throw WebKitDownloadHarnessFailure.invalidArguments }

    _ = NSApplication.shared
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    let webView = WKWebView(
      frame: NSRect(x: 0, y: 0, width: 640, height: 480),
      configuration: configuration
    )
    let navigationWaiter = WebKitNavigationWaiter()
    try await navigationWaiter.load(URLRequest(url: pageURL), in: webView)

    var request = URLRequest(url: resourceURL)
    request.httpMethod = "GET"
    // Production deliberately removes these native snapshots and lets the
    // current WebKit page/session supply browser identity and cookies.
    request.setValue("stale=native", forHTTPHeaderField: "Cookie")
    request.setValue("Native-Imitation", forHTTPHeaderField: "User-Agent")
    request.setValue("https://wrong.invalid", forHTTPHeaderField: "Origin")
    request.setValue("bytes=2-5", forHTTPHeaderField: "Range")
    request.httpShouldHandleCookies = false
    let operation = IPadBrowserWebKitDownloadOperation(
      webView: webView,
      request: request,
      maximumResponseBytes: 1_024,
      resolutionPolicy: .userSubmitted
    )
    let loaded = try await operation.run()
    let body = String(decoding: loaded.data, as: UTF8.self)
    guard loaded.response.statusCode == 206, body == "bkit",
      loaded.response.value(forHTTPHeaderField: "Content-Range")
        == "bytes 2-5/17"
    else {
      throw WebKitDownloadHarnessFailure.response(
        loaded.response.statusCode,
        body
      )
    }
    print("WKDownload browser-session probe passed")
  }
}
