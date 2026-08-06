import Foundation

@main
@MainActor
struct IPadBrowserLibraryStoreHarness {
  private static let historyKey = "mioh.ipad.browser.history.v1"
  private static let bookmarksKey = "mioh.ipad.browser.bookmarks.v1"

  static func main() throws {
    let suiteName = "mioh.browser-library-probe.\(UUID().uuidString)"
    guard let defaults = UserDefaults(suiteName: suiteName) else {
      fatalError("could not create isolated defaults suite")
    }
    defer { defaults.removePersistentDomain(forName: suiteName) }

    let store = IPadBrowserLibraryStore(
      defaults: defaults,
      maximumHistoryCount: 3,
      maximumBookmarkCount: 2
    )
    expect(store.history.isEmpty, "history must start empty")
    expect(store.bookmarks.isEmpty, "bookmarks must start empty")

    let firstDate = Date(timeIntervalSinceReferenceDate: 100)
    let privateURL = requiredURL(
      "https://EXAMPLE.com:443/watch?token=secret&quality=hd#player"
    )
    store.recordVisit(
      url: privateURL,
      title: "  First\u{0007}   Page  ",
      date: firstDate
    )
    expect(store.history.count == 1, "valid visit must be recorded")
    expect(
      store.history[0].url == "https://example.com/watch?quality=hd",
      "stored URL must be canonical and strip fragment/sensitive query"
    )
    expect(store.history[0].title == "First Page", "title must be normalized")
    expect(store.history[0].date == firstDate, "visit date must be retained")

    store.recordVisit(url: requiredURL("http://example.com/insecure"), title: "HTTP")
    store.recordVisit(url: requiredURL("https://localhost/private"), title: "Local")
    store.recordVisit(url: requiredURL("https://192.168.1.2/private"), title: "LAN")
    store.recordVisit(
      url: requiredURL("https://user:password@example.com/private"),
      title: "Credentials"
    )
    expect(store.history.count == 1, "unsafe visits must be ignored")

    store.recordVisit(
      url: requiredURL("https://example.org/a"), title: "A",
      date: Date(timeIntervalSinceReferenceDate: 200)
    )
    store.recordVisit(
      url: requiredURL("https://example.org/b"), title: "B",
      date: Date(timeIntervalSinceReferenceDate: 300)
    )
    store.recordVisit(
      url: requiredURL("https://example.org/c"), title: "C",
      date: Date(timeIntervalSinceReferenceDate: 400)
    )
    expect(
      store.history.map(\.title) == ["C", "B", "A"],
      "history must be newest-first and bounded"
    )
    store.recordVisit(
      url: requiredURL("https://EXAMPLE.org:443/b#again"),
      title: "B updated",
      date: Date(timeIntervalSinceReferenceDate: 500)
    )
    expect(store.history.count == 3, "repeat visits must deduplicate")
    expect(store.history[0].title == "B updated", "repeat visit must be promoted")
    expect(
      store.history.filter { $0.url == "https://example.org/b" }.count == 1,
      "canonical duplicate must occur once"
    )

    let reloaded = IPadBrowserLibraryStore(
      defaults: defaults,
      maximumHistoryCount: 3,
      maximumBookmarkCount: 2
    )
    expect(reloaded.history == store.history, "history must persist across stores")

    expect(
      store.toggleBookmark(url: privateURL, title: "Saved", date: firstDate),
      "first bookmark toggle must add"
    )
    expect(
      store.isBookmarked(requiredURL("https://example.com/watch?quality=hd#other")),
      "bookmark lookup must use canonical URL"
    )
    expect(
      !store.toggleBookmark(url: privateURL, title: "Saved", date: firstDate),
      "second bookmark toggle must remove"
    )
    expect(!store.isBookmarked(privateURL), "removed bookmark must not remain")

    store.toggleBookmark(url: requiredURL("https://example.org/a"), title: "A")
    store.toggleBookmark(url: requiredURL("https://example.org/b"), title: "B")
    store.toggleBookmark(url: requiredURL("https://example.org/c"), title: "C")
    expect(
      store.bookmarks.map(\.title) == ["C", "B"],
      "bookmarks must be newest-first and bounded"
    )
    let persistedBookmarks = IPadBrowserLibraryStore(
      defaults: defaults,
      maximumHistoryCount: 3,
      maximumBookmarkCount: 2
    )
    expect(
      persistedBookmarks.bookmarks == store.bookmarks,
      "bookmarks must persist across stores"
    )

    let removedHistoryID = store.history[0].id
    store.removeHistory(id: removedHistoryID)
    expect(
      !store.history.contains { $0.id == removedHistoryID },
      "individual history removal must persist in memory"
    )
    let removedBookmarkID = store.bookmarks[0].id
    store.removeBookmark(id: removedBookmarkID)
    expect(
      !store.bookmarks.contains { $0.id == removedBookmarkID },
      "individual bookmark removal must persist in memory"
    )

    let storedEntries = [
      IPadBrowserLibraryStore.Entry(
        url: "https://EXAMPLE.net:443/path?auth=secret&view=full#fragment",
        title: "  Stored   Page ",
        date: firstDate
      ),
      IPadBrowserLibraryStore.Entry(
        url: "https://example.net/path?view=full",
        title: "Duplicate",
        date: firstDate
      ),
      IPadBrowserLibraryStore.Entry(
        url: "https://127.0.0.1/private",
        title: "Unsafe",
        date: firstDate
      ),
    ]
    defaults.set(try JSONEncoder().encode(storedEntries), forKey: historyKey)
    let sanitizedReload = IPadBrowserLibraryStore(
      defaults: defaults,
      maximumHistoryCount: 3,
      maximumBookmarkCount: 2
    )
    expect(sanitizedReload.history.count == 1, "loaded entries must sanitize/dedupe")
    expect(
      sanitizedReload.history[0].url == "https://example.net/path?view=full",
      "loaded URL must be canonical and private data removed"
    )
    expect(
      sanitizedReload.history[0].title == "Stored Page",
      "loaded title must be normalized"
    )

    defaults.set(Data("not-json".utf8), forKey: historyKey)
    defaults.set(Data(repeating: 0, count: 2 * 1_024 * 1_024), forKey: bookmarksKey)
    let corruptReload = IPadBrowserLibraryStore(defaults: defaults)
    expect(corruptReload.history.isEmpty, "corrupt history must fail closed")
    expect(corruptReload.bookmarks.isEmpty, "oversized bookmarks must fail closed")

    store.clearHistory()
    store.clearBookmarks()
    expect(store.history.isEmpty, "clear history must empty memory")
    expect(store.bookmarks.isEmpty, "clear bookmarks must empty memory")
    expect(defaults.object(forKey: historyKey) == nil, "clear history must remove defaults")
    expect(
      defaults.object(forKey: bookmarksKey) == nil,
      "clear bookmarks must remove defaults"
    )

    print("iPad browser library store probe passed")
  }

  private static func requiredURL(_ value: String) -> URL {
    guard let url = URL(string: value) else { fatalError("invalid test URL") }
    return url
  }

  private static func expect(
    _ condition: @autoclosure () -> Bool,
    _ message: String
  ) {
    guard condition() else { fatalError(message) }
  }
}
