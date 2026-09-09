import Combine
import Foundation

/// Persists only user-visible page metadata. WebKit cookies, request headers,
/// page contents, and discovered media URLs remain in their existing stores.
@MainActor
final class IPadBrowserLibraryStore: ObservableObject {
  struct Entry: Codable, Equatable, Identifiable {
    let url: String
    let title: String
    let date: Date

    var id: String { url }
    var resolvedURL: URL? { URL(string: url) }
  }

  @Published private(set) var history: [Entry]
  @Published private(set) var bookmarks: [Entry]

  private static let historyDefaultsKey = "mioh.ipad.browser.history.v1"
  private static let bookmarksDefaultsKey = "mioh.ipad.browser.bookmarks.v1"
  private static let maximumStoredDataSize = 1_024 * 1_024
  private static let maximumURLLength = 4_096
  private static let maximumTitleLength = 160

  private let defaults: UserDefaults
  private let maximumHistoryCount: Int
  private let maximumBookmarkCount: Int

  init(
    defaults: UserDefaults = .standard,
    maximumHistoryCount: Int = 200,
    maximumBookmarkCount: Int = 200
  ) {
    self.defaults = defaults
    self.maximumHistoryCount = max(1, maximumHistoryCount)
    self.maximumBookmarkCount = max(1, maximumBookmarkCount)
    history = Self.loadEntries(
      from: defaults,
      key: Self.historyDefaultsKey,
      maximumCount: max(1, maximumHistoryCount)
    )
    bookmarks = Self.loadEntries(
      from: defaults,
      key: Self.bookmarksDefaultsKey,
      maximumCount: max(1, maximumBookmarkCount)
    )
  }

  func recordVisit(url: URL, title: String, date: Date = Date()) {
    guard let safeURL = Self.sanitizedPublicHTTPSURL(url) else { return }
    let entry = Self.makeEntry(url: safeURL, title: title, date: date)
    history.removeAll { $0.url == entry.url }
    history.insert(entry, at: 0)
    if history.count > maximumHistoryCount {
      history.removeLast(history.count - maximumHistoryCount)
    }
    persist(history, key: Self.historyDefaultsKey)
  }

  func isBookmarked(_ url: URL?) -> Bool {
    guard let url, let safeURL = Self.sanitizedPublicHTTPSURL(url) else {
      return false
    }
    return bookmarks.contains { $0.url == safeURL.absoluteString }
  }

  @discardableResult
  func toggleBookmark(url: URL, title: String, date: Date = Date()) -> Bool {
    guard let safeURL = Self.sanitizedPublicHTTPSURL(url) else { return false }
    let canonicalURL = safeURL.absoluteString
    if let index = bookmarks.firstIndex(where: { $0.url == canonicalURL }) {
      bookmarks.remove(at: index)
      persist(bookmarks, key: Self.bookmarksDefaultsKey)
      return false
    }

    bookmarks.insert(
      Self.makeEntry(url: safeURL, title: title, date: date),
      at: 0
    )
    if bookmarks.count > maximumBookmarkCount {
      bookmarks.removeLast(bookmarks.count - maximumBookmarkCount)
    }
    persist(bookmarks, key: Self.bookmarksDefaultsKey)
    return true
  }

  func removeHistory(id: Entry.ID) {
    history.removeAll { $0.id == id }
    persist(history, key: Self.historyDefaultsKey)
  }

  func removeBookmark(id: Entry.ID) {
    bookmarks.removeAll { $0.id == id }
    persist(bookmarks, key: Self.bookmarksDefaultsKey)
  }

  func clearHistory() {
    history.removeAll()
    defaults.removeObject(forKey: Self.historyDefaultsKey)
  }

  func clearBookmarks() {
    bookmarks.removeAll()
    defaults.removeObject(forKey: Self.bookmarksDefaultsKey)
  }

  private func persist(_ entries: [Entry], key: String) {
    guard let data = try? JSONEncoder().encode(entries),
      data.count <= Self.maximumStoredDataSize
    else { return }
    defaults.set(data, forKey: key)
  }

  private static func loadEntries(
    from defaults: UserDefaults,
    key: String,
    maximumCount: Int
  ) -> [Entry] {
    guard let data = defaults.data(forKey: key),
      data.count <= maximumStoredDataSize,
      let decoded = try? JSONDecoder().decode([Entry].self, from: data)
    else { return [] }

    var seen = Set<String>()
    var entries: [Entry] = []
    for storedEntry in decoded {
      guard entries.count < maximumCount,
        let rawURL = URL(string: storedEntry.url),
        let safeURL = sanitizedPublicHTTPSURL(rawURL),
        seen.insert(safeURL.absoluteString).inserted
      else { continue }
      entries.append(
        makeEntry(
          url: safeURL,
          title: storedEntry.title,
          date: storedEntry.date
        )
      )
    }
    return entries
  }

  private static func makeEntry(url: URL, title: String, date: Date) -> Entry {
    Entry(
      url: url.absoluteString,
      title: sanitizedTitle(title, fallbackURL: url),
      date: date.timeIntervalSinceReferenceDate.isFinite ? date : Date()
    )
  }

  private static func sanitizedTitle(_ rawTitle: String, fallbackURL: URL) -> String {
    let withoutControls = rawTitle.components(separatedBy: .controlCharacters)
      .joined(separator: " ")
    let collapsed = withoutControls.split(whereSeparator: { $0.isWhitespace })
      .joined(separator: " ")
    let trimmed = String(collapsed.prefix(maximumTitleLength))
    if !trimmed.isEmpty { return trimmed }
    return fallbackURL.host ?? fallbackURL.absoluteString
  }

  private static func sanitizedPublicHTTPSURL(_ rawURL: URL) -> URL? {
    guard rawURL.absoluteString.utf8.count <= maximumURLLength,
      var components = URLComponents(url: rawURL, resolvingAgainstBaseURL: true),
      components.scheme?.lowercased() == "https",
      components.user == nil, components.password == nil,
      let rawHost = components.host, isPublicHostSyntax(rawHost)
    else { return nil }

    components.scheme = "https"
    components.host = rawHost.lowercased()
    components.fragment = nil
    if components.port == 443 { components.port = nil }
    if let queryItems = components.queryItems {
      let filteredItems = queryItems.filter { !isSensitiveQueryName($0.name) }
      components.queryItems = filteredItems.isEmpty ? nil : filteredItems
    }
    guard let safeURL = components.url,
      safeURL.absoluteString.utf8.count <= maximumURLLength
    else { return nil }
    return safeURL
  }

  private static func isSensitiveQueryName(_ rawName: String) -> Bool {
    let name = rawName.lowercased()
    if name.hasPrefix("x-amz-") || name.hasPrefix("x-goog-") { return true }
    return [
      "access_token", "auth", "authorization", "code", "credential",
      "credentials", "jwt", "passwd", "password", "session", "sessionid",
      "sig", "signature", "token",
    ].contains(name)
  }

  /// Nonblocking syntax validation suitable for UI persistence paths.
  private static func isPublicHostSyntax(_ rawHost: String) -> Bool {
    let host = rawHost.lowercased().trimmingCharacters(
      in: CharacterSet(charactersIn: ".[]")
    )
    guard !host.isEmpty, host != "localhost",
      !host.hasSuffix(".localhost"), !host.hasSuffix(".local"),
      !host.hasPrefix("0x")
    else { return false }

    let rawIPv4Parts = host.split(separator: ".", omittingEmptySubsequences: false)
    if rawIPv4Parts.count == 4,
      rawIPv4Parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) })
    {
      guard rawIPv4Parts.allSatisfy({ $0.count == 1 || !$0.hasPrefix("0") }) else {
        return false
      }
      let parts = rawIPv4Parts.compactMap { UInt8($0) }
      guard parts.count == 4 else { return false }
      let first = parts[0]
      let second = parts[1]
      let third = parts[2]
      if first == 0 || first == 10 || first == 127 || first >= 224 { return false }
      if first == 100, (64...127).contains(second) { return false }
      if first == 169, second == 254 { return false }
      if first == 172, (16...31).contains(second) { return false }
      if first == 192 {
        if second == 0, third == 0 || third == 2 { return false }
        if second == 88, third == 99 { return false }
        if second == 168 { return false }
      }
      if first == 198, second == 18 || second == 19 { return false }
      if first == 198, second == 51, third == 100 { return false }
      if first == 203, second == 0, third == 113 { return false }
      return true
    }
    if host.allSatisfy({ $0.isNumber || $0 == "." }) { return false }
    if host.contains(":") {
      return host != "::" && host != "::1" && !host.hasPrefix("::ffff:")
        && !host.hasPrefix("fc") && !host.hasPrefix("fd")
        && !host.hasPrefix("fe8") && !host.hasPrefix("fe9")
        && !host.hasPrefix("fea") && !host.hasPrefix("feb")
    }
    return true
  }
}
