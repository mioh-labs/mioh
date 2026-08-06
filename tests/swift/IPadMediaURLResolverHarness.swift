import Foundation

private enum ResolverHarnessFailure: Error, CustomStringConvertible {
  case assertion(String)
  case usage

  var description: String {
    switch self {
    case .assertion(let message): message
    case .usage: "usage: IPadMediaURLResolverHarness <scenario> <base-url>"
    }
  }
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  guard condition() else { throw ResolverHarnessFailure.assertion(message) }
}

private func endpoint(_ path: String, relativeTo baseURL: URL) -> URL {
  URL(string: path, relativeTo: baseURL)!.absoluteURL
}

@main
private struct IPadMediaURLResolverHarness {
  static func main() async throws {
    guard CommandLine.arguments.count == 3,
      let baseURL = URL(string: CommandLine.arguments[2])
    else { throw ResolverHarnessFailure.usage }

    switch CommandLine.arguments[1] {
    case "master":
      try await verifyMasterPlaylist(relativeTo: baseURL)
    case "discontinuity":
      try await verifyDiscontinuityMetadata(relativeTo: baseURL)
    case "png-prefixed-ts":
      try await verifyPNGPrefixedTransportStream(relativeTo: baseURL)
    case "mapped-container-normalization":
      try await verifyMappedContainerNormalization(relativeTo: baseURL)
    case "html":
      try await verifyHTMLDiscovery(relativeTo: baseURL)
    case "nested-pages":
      try await verifyNestedPageDiscovery(relativeTo: baseURL)
    case "page-depth":
      try await verifyPageDepthLimit(relativeTo: baseURL)
    case "body-budget":
      try await verifyBodyBudget(relativeTo: baseURL)
    case "drm":
      try await verifyDRMRejection(relativeTo: baseURL)
    case "invalid-playlists":
      try await verifyInvalidPlaylists(relativeTo: baseURL)
    case "request-budget":
      try await verifyRequestBudget(relativeTo: baseURL)
    case "redirect":
      try await verifyRedirect(relativeTo: baseURL)
    case "redirect-limit":
      try await verifyRedirectLimit(relativeTo: baseURL)
    case "redirect-private":
      try await verifyPrivateRedirectRejection(relativeTo: baseURL)
    case "cancellation":
      try await verifyCancellation(relativeTo: baseURL)
    case "request-context":
      try await verifyRequestContext(relativeTo: baseURL)
    case "interaction-required":
      try await verifyInteractionRequired(relativeTo: baseURL)
    case "head-fallback":
      try await verifyHEADFallback(relativeTo: baseURL)
    case "challenged-rendition-fallback":
      try await verifyChallengedRenditionFallback(relativeTo: baseURL)
    case "public-url-policy":
      try verifyPublicURLPolicy()
    case "cookie-updates":
      try await verifyCookieUpdates(relativeTo: baseURL)
    case "challenge-priority":
      try await verifyChallengePriority(relativeTo: baseURL)
    case "browser-selection":
      try verifyBrowserMediaSelection()
    case "html-ad-selection":
      try await verifyHTMLAdvertisementSelection(relativeTo: baseURL)
    case "unlabeled-playlist-reference":
      try await verifyUnlabeledPlaylistReference(relativeTo: baseURL)
    case "media-limits":
      try verifyMediaLimits()
    default:
      throw ResolverHarnessFailure.usage
    }
  }

  private static func verifyMediaLimits() throws {
    try require(
      IPadRestorationMediaLimits.accepts(
        width: 1_920,
        height: 1_080,
        clipLength: 18
      ),
      "the normal 1080p T18 budget was rejected"
    )
    try require(
      !IPadRestorationMediaLimits.accepts(
        width: 1_920,
        height: 1_080,
        clipLength: 30
      ),
      "the normal restoration budget unexpectedly changed to T30"
    )
    try require(
      IPadRestorationMediaLimits.accepts(
        width: 1_920,
        height: 1_080,
        clipLength: 30,
        pixelFrameBudget: IPadRestorationMediaLimits.maximumRealtimePixelFrames
      ),
      "the realtime 1080p T30 budget was rejected"
    )
    try require(
      !IPadRestorationMediaLimits.accepts(
        width: 1_920,
        height: 1_080,
        clipLength: 31,
        pixelFrameBudget: IPadRestorationMediaLimits.maximumRealtimePixelFrames
      ),
      "the realtime budget exceeded T30"
    )
  }

  private static func verifyBrowserMediaSelection() throws {
    func hls(
      _ path: String,
      duration: Double,
      isLive: Bool,
      targetDuration: Double = 6
    ) -> IPadResolvedMediaSource {
      let url = URL(string: "https://media.example/\(path)")!
      let resource = IPadHLSResource(
        url: URL(string: "https://media.example/segment.ts")!,
        byteRange: nil
      )
      let playlist = IPadHLSMediaPlaylist(
        url: url,
        segments: [
          IPadHLSMediaSegment(
            sequence: 0,
            duration: duration,
            resource: resource,
            initializationResource: nil,
            startSeconds: 0
          )
        ],
        isLive: isLive,
        duration: duration,
        targetDuration: targetDuration
      )
      return IPadResolvedMediaSource(
        kind: .hls,
        submittedURL: url,
        playbackURL: url,
        mediaURL: url,
        contentType: "application/vnd.apple.mpegurl",
        hlsPlaylist: playlist,
        resolutionPolicy: .publicDiscovered,
        requestContext: nil
      )
    }

    func progressive(_ path: String) -> IPadResolvedMediaSource {
      let url = URL(string: "https://media.example/\(path)")!
      return IPadResolvedMediaSource(
        kind: .progressive,
        submittedURL: url,
        playbackURL: url,
        mediaURL: url,
        contentType: "video/mp4",
        hlsPlaylist: nil,
        resolutionPolicy: .publicDiscovered,
        requestContext: nil
      )
    }

    func evidence(_ duration: Double, generation: Int) -> IPadBrowserMediaEvidence {
      IPadBrowserMediaEvidence(
        observedDuration: duration,
        isPlaying: true,
        isVisible: true,
        visibilityAttested: true,
        renderedArea: 1_280 * 720,
        sourceGeneration: generation,
        activationOrder: generation
      )
    }

    let rollingPreRoll = hls("rolling/index.m3u8", duration: 18, isLive: true)
    let feature = hls("feature/index.m3u8", duration: 5_400, isLive: false)
    try require(
      IPadBrowserMediaSourceSelector.preferredIndex(
        in: [rollingPreRoll, feature]
      ) == 1,
      "a rolling pre-roll displaced the long on-demand programme"
    )
    try require(
      IPadBrowserMediaSourceSelector.shouldAcceptImmediately(
        feature,
        evidence: nil
      ),
      "a long finite programme was not eligible for immediate acceptance"
    )
    try require(
      !IPadBrowserMediaSourceSelector.shouldAcceptImmediately(
        rollingPreRoll,
        evidence: nil
      ),
      "a short rolling pre-roll was accepted immediately"
    )
    let liveProgramme = hls("live/index.m3u8", duration: 36, isLive: true)
    let shortPreview = hls("preview/index.m3u8", duration: 20, isLive: false)
    try require(
      IPadBrowserMediaSourceSelector.preferredIndex(
        in: [liveProgramme, shortPreview]
      ) == 0,
      "a short preview displaced a genuine live stream"
    )

    let shorterFeature = hls("short-feature/index.m3u8", duration: 300, isLive: false)
    try require(
      IPadBrowserMediaSourceSelector.shouldAcceptImmediately(
        shorterFeature,
        evidence: evidence(300, generation: 3)
      ),
      "a visible long-playing source was not eligible for immediate acceptance"
    )
    try require(
      IPadBrowserMediaSourceSelector.preferredIndex(
        in: [shorterFeature, feature]
      ) == 1,
      "the longest on-demand programme was not preferred"
    )
    try require(
      IPadBrowserMediaSourceSelector.preferredIndex(in: [shortPreview]) == 0,
      "a sole short clip must remain usable as a fallback"
    )

    let progressiveLeadIn = progressive("lead-in.mp4")
    let progressiveProgramme = progressive("programme.mp4")
    try require(
      IPadBrowserMediaSourceSelector.preferredIndex(
        in: [progressiveLeadIn, progressiveProgramme],
        evidence: [evidence(30, generation: 1), evidence(60, generation: 2)]
      ) == 1,
      "observed progressive duration did not displace the shorter lead-in"
    )
    try require(
      IPadBrowserMediaSourceSelector.preferredIndex(
        in: [progressiveProgramme, rollingPreRoll],
        evidence: [evidence(600, generation: 2), nil]
      ) == 0,
      "a short parsed playlist displaced the longer observed programme"
    )
  }

  private static func verifyHTMLAdvertisementSelection(
    relativeTo baseURL: URL
  ) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    let pageURL = endpoint("selection/page", relativeTo: baseURL)
    let source = try await resolver.resolve(pageURL.absoluteString)
    try require(source.kind == .hls, "HTML media selection did not return HLS")
    try require(
      source.playbackURL.path == "/selection/programme.m3u8",
      "the first short lead-in displaced the later programme: \(source.playbackURL)"
    )
    try require(
      abs((source.hlsPlaylist?.duration ?? 0) - 600) < 0.001,
      "the selected programme duration is incorrect"
    )
  }

  private static func verifyUnlabeledPlaylistReference(
    relativeTo baseURL: URL
  ) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    let submittedURL = endpoint("relay/outer.m3u8", relativeTo: baseURL)
    let source = try await resolver.resolve(submittedURL.absoluteString)
    try require(source.kind == .hls, "unlabeled reference did not return HLS")
    try require(
      source.playbackURL.path == "/relay/outer.m3u8",
      "outer relay URL was not preserved as playback URL"
    )
    try require(
      source.mediaURL.path == "/relay/inner/high.m3u8",
      "nested master did not select the bounded high rendition: \(source.mediaURL)"
    )
    try require(
      abs((source.hlsPlaylist?.duration ?? 0) - 5) < 0.001,
      "nested media playlist duration is incorrect"
    )
  }

  private static func verifyMasterPlaylist(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    let submittedURL = endpoint("hls/master.m3u8", relativeTo: baseURL)
    let source = try await resolver.resolve(submittedURL.absoluteString)

    try require(source.kind == .hls, "master playlist was not classified as HLS")
    try require(source.submittedURL == submittedURL, "submitted URL was not preserved")
    try require(
      source.mediaURL.path == "/hls/high/index.m3u8",
      "bounded 1080p relative variant was not selected: \(source.mediaURL)"
    )
    let playlist = try source.hlsPlaylist
      .unwrap(or: ResolverHarnessFailure.assertion("resolved HLS playlist is absent"))
    try require(playlist.url == source.mediaURL, "media URL and playlist URL disagree")
    try require(playlist.segments.count == 2, "unexpected media segment count")
    try require(!playlist.isLive, "ENDLIST playlist was reported as live")
    try require(abs(playlist.duration - 4.0) < 0.001, "playlist duration is incorrect")

    let first = playlist.segments[0]
    let second = playlist.segments[1]
    try require(first.sequence == 41 && second.sequence == 42, "media sequence was lost")
    try require(first.resource.url.path == "/hls/assets/media.mp4", "segment URL is wrong")
    try require(second.resource.url == first.resource.url, "relative segment URL is unstable")
    try require(
      first.initializationResource?.url.path == "/hls/assets/media.mp4",
      "relative initialization URL is wrong"
    )
    try require(
      first.initializationResource?.byteRange == IPadHLSByteRange(offset: 0, length: 4),
      "initialization byte range is wrong"
    )
    try require(
      first.resource.byteRange == IPadHLSByteRange(offset: 4, length: 3),
      "explicit segment byte range is wrong"
    )
    try require(
      second.resource.byteRange == IPadHLSByteRange(offset: 7, length: 2),
      "implicit segment byte-range offset was not inherited"
    )

    let outputDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-url-resolver-harness-\(UUID().uuidString)",
      isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: outputDirectory) }
    let downloader = IPadHLSResourceDownloader(
      maximumResourceBytes: 1_024,
      requestTimeout: 3
    )
    let materializedURL = try await downloader.materialize(
      segment: first,
      in: outputDirectory
    )
    try require(materializedURL.pathExtension == "mp4", "fMP4 materialization has wrong suffix")
    let materializedData = try Data(contentsOf: materializedURL)
    try require(
      materializedData == Data("ABCDEFG".utf8),
      "initialization and media byte ranges were not materialized in order"
    )
    let secondSegmentData = try await downloader.data(for: second.resource)
    try require(
      secondSegmentData == Data("HI".utf8),
      "implicit byte range downloaded the wrong bytes"
    )
  }

  private static func verifyDiscontinuityMetadata(
    relativeTo baseURL: URL
  ) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    let submittedURL = endpoint("hls/discontinuity.m3u8", relativeTo: baseURL)
    let source = try await resolver.resolve(submittedURL.absoluteString)
    try require(source.kind == .hls, "discontinuity playlist was not HLS")
    let playlist = try source.hlsPlaylist.unwrap(
      or: ResolverHarnessFailure.assertion("discontinuity playlist is absent")
    )
    try require(playlist.segments.count == 3, "unexpected discontinuity segment count")
    try require(!playlist.isLive, "discontinuity VOD was reported as live")
    try require(abs(playlist.duration - 9) < 0.001, "discontinuity duration is wrong")
    try require(
      playlist.segments.map(\.sequence) == [500, 501, 502],
      "media sequence changed around discontinuities"
    )
    try require(
      playlist.segments.map(\.discontinuitySequence) == [7, 8, 9],
      "DISCONTINUITY-SEQUENCE/EXT-X-DISCONTINUITY metadata was not preserved"
    )
    try require(
      playlist.segments.map(\.startSeconds) == [0, 2, 5],
      "timeline offsets changed around discontinuities"
    )
    try require(
      playlist.segments.map(\.resource.url.lastPathComponent)
        == ["first.ts", "second.ts", "third.ts"],
      "segment URLs changed around discontinuities"
    )
  }

  private static func verifyPNGPrefixedTransportStream(
    relativeTo baseURL: URL
  ) async throws {
    var expectedTransportStream = Data()
    for packetIndex in 0..<5 {
      expectedTransportStream.append(0x47)
      expectedTransportStream.append(
        Data(repeating: UInt8(0x10 + packetIndex), count: 187)
      )
    }

    func segment(_ path: String, sequence: Int64) -> IPadHLSMediaSegment {
      IPadHLSMediaSegment(
        sequence: sequence,
        duration: 1,
        resource: IPadHLSResource(
          url: endpoint(path, relativeTo: baseURL),
          byteRange: nil
        ),
        initializationResource: nil,
        startSeconds: 0
      )
    }

    let outputDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-png-prefixed-ts-harness-\(UUID().uuidString)",
      isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: outputDirectory) }
    let downloader = IPadHLSResourceDownloader(
      maximumResourceBytes: 4_096,
      requestTimeout: 3
    )

    let disguisedURL = try await downloader.materialize(
      segment: segment("materialize/png-prefixed.bin", sequence: 230),
      in: outputDirectory
    )
    let disguisedData = try Data(contentsOf: disguisedURL)
    try require(disguisedURL.pathExtension == "ts", "normalized TS has wrong suffix")
    try require(disguisedData.first == 0x47, "normalized TS does not start at sync byte")
    try require(
      disguisedData == expectedTransportStream,
      "PNG-like 230-byte prefix was not stripped exactly"
    )

    let ordinaryURL = try await downloader.materialize(
      segment: segment("materialize/ordinary.ts", sequence: 231),
      in: outputDirectory
    )
    let ordinaryData = try Data(contentsOf: ordinaryURL)
    try require(ordinaryURL.pathExtension == "ts", "ordinary TS has wrong suffix")
    try require(
      ordinaryData == expectedTransportStream,
      "ordinary MPEG-TS was not preserved byte-for-byte"
    )

    let misleadingURL = try await downloader.materialize(
      segment: segment("materialize/disguised.mp4", sequence: 232),
      in: outputDirectory
    )
    let misleadingData = try Data(contentsOf: misleadingURL)
    try require(
      misleadingURL.pathExtension == "ts",
      "MPEG-TS payload inherited a misleading remote MP4 suffix"
    )
    try require(
      misleadingData == expectedTransportStream,
      "MPEG-TS payload behind an MP4 URL was not preserved byte-for-byte"
    )
  }

  private static func verifyMappedContainerNormalization(
    relativeTo baseURL: URL
  ) async throws {
    func box(_ type: String, payload: String) -> Data {
      let payloadBytes = Array(payload.utf8)
      let size = UInt32(8 + payloadBytes.count)
      var result = Data([
        UInt8((size >> 24) & 0xff),
        UInt8((size >> 16) & 0xff),
        UInt8((size >> 8) & 0xff),
        UInt8(size & 0xff),
      ])
      result.append(contentsOf: type.utf8)
      result.append(contentsOf: payloadBytes)
      return result
    }

    let initializationData = box("ftyp", payload: "isom")
      + box("moov", payload: "init")
    let fragmentData = box("styp", payload: "msdh")
      + box("moof", payload: "fragment")
      + box("mdat", payload: "media")
    let selfContainedData = box("ftyp", payload: "isom")
      + box("moov", payload: "complete")
      + box("mdat", payload: "media")
    var transportData = Data()
    for packetIndex in 0..<5 {
      transportData.append(0x47)
      transportData.append(
        Data(repeating: UInt8(0x10 + packetIndex), count: 187)
      )
    }

    let initializationResource = IPadHLSResource(
      url: endpoint("materialize/init.mp4", relativeTo: baseURL),
      byteRange: nil
    )
    func segment(_ path: String, sequence: Int64) -> IPadHLSMediaSegment {
      IPadHLSMediaSegment(
        sequence: sequence,
        duration: 1,
        resource: IPadHLSResource(
          url: endpoint(path, relativeTo: baseURL),
          byteRange: nil
        ),
        initializationResource: initializationResource,
        startSeconds: 0
      )
    }

    let outputDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-map-normalization-harness-\(UUID().uuidString)",
      isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: outputDirectory) }
    let downloader = IPadHLSResourceDownloader(
      maximumResourceBytes: 4_096,
      requestTimeout: 3
    )

    let fragmentURL = try await downloader.materialize(
      segment: segment("materialize/fragment.m4s", sequence: 240),
      in: outputDirectory
    )
    let materializedFragmentData = try Data(contentsOf: fragmentURL)
    try require(fragmentURL.pathExtension == "mp4", "mapped fMP4 has wrong suffix")
    try require(
      materializedFragmentData == initializationData + fragmentData,
      "mapped fMP4 did not retain exactly one initialization section"
    )

    let selfContainedURL = try await downloader.materialize(
      segment: segment("materialize/self-contained.mp4", sequence: 241),
      in: outputDirectory
    )
    let materializedSelfContainedData = try Data(contentsOf: selfContainedURL)
    try require(
      materializedSelfContainedData == selfContainedData,
      "self-contained MP4 received a duplicate initialization section"
    )

    let mappedTransportURL = try await downloader.materialize(
      segment: segment("materialize/mapped-transport.mp4", sequence: 242),
      in: outputDirectory
    )
    let materializedTransportData = try Data(contentsOf: mappedTransportURL)
    try require(
      mappedTransportURL.pathExtension == "ts",
      "mapped MPEG-TS was misclassified as MP4"
    )
    try require(
      materializedTransportData == transportData,
      "mapped MPEG-TS received ISO initialization bytes"
    )
  }

  private static func verifyHTMLDiscovery(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    let pageURL = endpoint("embed/page", relativeTo: baseURL)
    let source = try await resolver.resolve(pageURL.absoluteString)
    try require(source.kind == .hls, "HTML progressive fallback was preferred over HLS")
    try require(source.submittedURL == pageURL, "HTML submitted URL was not preserved")
    try require(
      source.playbackURL.path == "/embed/stream/live.m3u8",
      "relative HTML HLS candidate was not resolved"
    )
    try require(
      source.playbackURL.query == "token=a&b=c",
      "escaped HTML query was not decoded"
    )
    try require(
      source.hlsPlaylist?.segments.first?.resource.url.path == "/embed/chunks/a.ts",
      "media segment was not resolved relative to the discovered playlist"
    )
  }

  private static func verifyNestedPageDiscovery(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    let pageURL = endpoint("layers/root", relativeTo: baseURL)
    let source = try await resolver.resolve(pageURL.absoluteString)
    try require(source.kind == .hls, "nested HLS did not beat the outer MP4 fallback")
    try require(source.submittedURL == pageURL, "nested submitted URL was not preserved")
    try require(
      source.playbackURL.path == "/layers/final.m3u8",
      "three-layer player URL was not resolved: \(source.playbackURL)"
    )
    try require(
      source.hlsPlaylist?.segments.first?.resource.url.path == "/layers/segment.ts",
      "nested media segment URL is wrong"
    )
  }

  private static func verifyPageDepthLimit(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    let pageURL = endpoint("depth/root", relativeTo: baseURL)
    let source = try await resolver.resolve(pageURL.absoluteString)
    try require(source.kind == .progressive, "over-depth HLS unexpectedly replaced fallback")
    try require(
      source.playbackURL.path == "/depth/fallback.mp4",
      "outer progressive fallback was not retained"
    )
  }

  private static func verifyBodyBudget(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(
      maximumResponseBytes: 1_024,
      requestTimeout: 3,
      maximumCumulativeResponseBytes: 1_024
    )
    do {
      _ = try await resolver.resolve(
        endpoint("body-budget/root", relativeTo: baseURL).absoluteString
      )
      throw ResolverHarnessFailure.assertion("cumulative response budget was not enforced")
    } catch IPadMediaURLResolverError.resolutionLimitExceeded {
      return
    }
  }

  private static func verifyDRMRejection(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    do {
      _ = try await resolver.resolve(
        endpoint("drm/encrypted.m3u8", relativeTo: baseURL).absoluteString
      )
      throw ResolverHarnessFailure.assertion("encrypted HLS was accepted")
    } catch IPadMediaURLResolverError.encryptedPlaylist {
      return
    }
  }

  private static func verifyInvalidPlaylists(relativeTo baseURL: URL) async throws {
    let invalidPaths = [
      "invalid/extinf-infinity.m3u8",
      "invalid/extinf-zero.m3u8",
      "invalid/extinf-overflow.m3u8",
      "invalid/duplicate-media-sequence.m3u8",
      "invalid/late-media-sequence.m3u8",
      "invalid/media-sequence-overflow.m3u8",
    ]
    for path in invalidPaths {
      let resolver = IPadMediaURLResolver(requestTimeout: 3)
      do {
        _ = try await resolver.resolve(endpoint(path, relativeTo: baseURL).absoluteString)
        throw ResolverHarnessFailure.assertion("invalid playlist was accepted: \(path)")
      } catch IPadMediaURLResolverError.invalidPlaylist(_) {
        continue
      }
    }
  }

  private static func verifyRequestBudget(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(
      requestTimeout: 3,
      maximumRequestCount: 2,
      resolutionTimeout: 10
    )
    do {
      _ = try await resolver.resolve(
        endpoint("hls/master.m3u8", relativeTo: baseURL).absoluteString)
      throw ResolverHarnessFailure.assertion("request budget was not enforced")
    } catch IPadMediaURLResolverError.resolutionLimitExceeded {
      return
    }
  }

  private static func verifyRedirect(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(maximumRedirectCount: 2, requestTimeout: 3)
    let source = try await resolver.resolve(endpoint("go", relativeTo: baseURL).absoluteString)
    try require(source.kind == .hls, "redirected playlist was not classified as HLS")
    try require(
      source.playbackURL.path == "/redirected/media.m3u8",
      "final redirect URL was not retained"
    )
    try require(
      source.hlsPlaylist?.segments.first?.resource.url.path == "/redirected/chunk.ts",
      "redirect destination was not used as the relative URL base"
    )
  }

  private static func verifyRedirectLimit(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(maximumRedirectCount: 1, requestTimeout: 3)
    do {
      _ = try await resolver.resolve(
        endpoint("redirect-loop/0", relativeTo: baseURL).absoluteString)
      throw ResolverHarnessFailure.assertion("redirect limit was not enforced")
    } catch IPadMediaURLResolverError.tooManyRedirects {
      return
    }
  }

  private static func verifyPrivateRedirectRejection(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(maximumRedirectCount: 2, requestTimeout: 3)
    do {
      _ = try await resolver.resolve(
        endpoint("redirect-private", relativeTo: baseURL).absoluteString
      )
      throw ResolverHarnessFailure.assertion("cross-origin local redirect was accepted")
    } catch IPadMediaURLResolverError.unsafeURL {
      return
    }
  }

  private static func verifyCancellation(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 5)
    let slowURL = endpoint("slow", relativeTo: baseURL)
    let task = Task { try await resolver.resolve(slowURL.absoluteString) }
    try await Task.sleep(nanoseconds: 100_000_000)
    let cancellationStart = Date()
    task.cancel()
    do {
      _ = try await task.value
      throw ResolverHarnessFailure.assertion("cancelled resolution completed successfully")
    } catch is CancellationError {
      try require(
        Date().timeIntervalSince(cancellationStart) < 1.0,
        "request cancellation did not promptly resume the awaiting task"
      )
    }
  }

  private static func verifyRequestContext(relativeTo baseURL: URL) async throws {
    let host = try baseURL.host.unwrap(
      or: ResolverHarnessFailure.assertion("fixture host is missing")
    )
    let sessionCookie = try IPadMediaRequestCookie(
      name: "mioh_session",
      value: "allowed",
      domain: host,
      path: "/context"
    ).unwrap(or: ResolverHarnessFailure.assertion("test cookie is invalid"))
    let expiredCookie = try IPadMediaRequestCookie(
      name: "expired",
      value: "must-not-leak",
      domain: host,
      path: "/context",
      expiresAt: Date(timeIntervalSince1970: 1)
    ).unwrap(or: ResolverHarnessFailure.assertion("expired test cookie is invalid"))
    let foreignCookie = try IPadMediaRequestCookie(
      name: "foreign",
      value: "must-not-leak",
      domain: "other.invalid",
      path: "/"
    ).unwrap(or: ResolverHarnessFailure.assertion("foreign test cookie is invalid"))
    let secureCookie = try IPadMediaRequestCookie(
      name: "secure_only",
      value: "must-not-use-over-http",
      domain: host,
      path: "/context",
      isSecure: true
    ).unwrap(or: ResolverHarnessFailure.assertion("secure test cookie is invalid"))
    let referer = endpoint("watch/page?token=secret#player", relativeTo: baseURL)
    let context = IPadMediaRequestContext(
      cookies: [sessionCookie, expiredCookie, foreignCookie, secureCookie],
      userAgent: "MiohContextHarness/1.0",
      referer: referer,
      origin: referer
    )
    try require(context.referer?.query == nil, "Referer query was retained")
    try require(context.referer?.fragment == nil, "Referer fragment was retained")
    try require(context.origin?.path.isEmpty == true, "Origin path was retained")
    try require(context.origin?.query == nil, "Origin query was retained")

    var foreignRequest = URLRequest(url: URL(string: "https://unrelated.invalid/context")!)
    context.applying(to: &foreignRequest)
    try require(
      foreignRequest.value(forHTTPHeaderField: "Cookie") == nil,
      "cookie leaked to an unrelated host"
    )
    try require(
      foreignRequest.value(forHTTPHeaderField: "Referer")
        == "\(baseURL.scheme ?? "http")://\(baseURL.host ?? ""):\(baseURL.port ?? 80)/",
      "cross-origin Referer retained an embedding path"
    )
    try require(
      foreignRequest.value(forHTTPHeaderField: "Origin")
        == "\(baseURL.scheme ?? "http")://\(baseURL.host ?? ""):\(baseURL.port ?? 80)",
      "sanitized browser Origin was not applied"
    )
    let domainCookie = try IPadMediaRequestCookie(
      name: "domain_cookie",
      value: "allowed-subdomain",
      domain: "example.test",
      path: "/hls",
      isSecure: true,
      includesSubdomains: true
    ).unwrap(or: ResolverHarnessFailure.assertion("domain test cookie is invalid"))
    let hostCookie = try IPadMediaRequestCookie(
      name: "host_cookie",
      value: "host-only",
      domain: "auth.example.test",
      path: "/hls",
      isSecure: true
    ).unwrap(or: ResolverHarnessFailure.assertion("host test cookie is invalid"))
    let domainContext = IPadMediaRequestContext(
      cookies: [domainCookie, hostCookie],
      cookieSourceURL: URL(string: "https://www.example.test/watch")
    )
    var subdomainRequest = URLRequest(
      url: URL(string: "https://cdn.example.test/hls/master.m3u8")!
    )
    domainContext.applying(to: &subdomainRequest)
    try require(
      subdomainRequest.value(forHTTPHeaderField: "Cookie")
        == "domain_cookie=allowed-subdomain",
      "cookie domain or host-only boundary is incorrect"
    )
    var siblingPathRequest = URLRequest(
      url: URL(string: "https://cdn.example.test/hls-other/master.m3u8")!
    )
    domainContext.applying(to: &siblingPathRequest)
    try require(
      siblingPathRequest.value(forHTTPHeaderField: "Cookie") == nil,
      "cookie path boundary was ignored"
    )
    let crossSiteNoneCookie = try IPadMediaRequestCookie(
      name: "cross_site_none",
      value: "allowed-only-with-native-proof",
      domain: "media.other.test",
      path: "/hls",
      isSecure: true,
      sameSitePolicy: .none
    ).unwrap(or: ResolverHarnessFailure.assertion("SameSite=None cookie is invalid"))
    let untrustedCrossSiteContext = IPadMediaRequestContext(
      cookies: [crossSiteNoneCookie],
      cookieSourceURL: URL(string: "https://www.example.test/watch"),
      allowsCrossSiteCredentialReplay: false
    )
    var untrustedCrossSiteRequest = URLRequest(
      url: URL(string: "https://media.other.test/hls/master.m3u8")!
    )
    untrustedCrossSiteContext.applying(to: &untrustedCrossSiteRequest)
    try require(
      untrustedCrossSiteRequest.value(forHTTPHeaderField: "Cookie") == nil,
      "untrusted script provenance replayed a cross-site cookie"
    )
    let nativeCrossSiteContext = IPadMediaRequestContext(
      cookies: [crossSiteNoneCookie],
      cookieSourceURL: URL(string: "https://www.example.test/watch"),
      allowsCrossSiteCredentialReplay: true
    )
    var nativeCrossSiteRequest = URLRequest(
      url: URL(string: "https://media.other.test/hls/master.m3u8")!
    )
    nativeCrossSiteContext.applying(to: &nativeCrossSiteRequest)
    try require(
      nativeCrossSiteRequest.value(forHTTPHeaderField: "Cookie")
        == "cross_site_none=allowed-only-with-native-proof",
      "native response provenance did not preserve SameSite=None"
    )

    let source = try await IPadMediaURLResolver(requestTimeout: 3).resolve(
      endpoint("context/master.m3u8", relativeTo: baseURL).absoluteString,
      context: context
    )
    try require(source.requestContext == context, "request context was not preserved")
    let playlist = try source.hlsPlaylist.unwrap(
      or: ResolverHarnessFailure.assertion("context playlist is absent")
    )
    let segment = try playlist.segments.first.unwrap(
      or: ResolverHarnessFailure.assertion("context playlist segment is absent")
    )
    let downloader = IPadHLSResourceDownloader(
      maximumResourceBytes: 1_024,
      requestTimeout: 3,
      requestContext: source.requestContext
    )
    let data = try await downloader.data(for: segment.resource)
    try require(data == Data("context-ok".utf8), "context segment download failed")
  }

  private static func verifyInteractionRequired(relativeTo baseURL: URL) async throws {
    do {
      _ = try await IPadMediaURLResolver(requestTimeout: 3).resolve(
        endpoint("challenge", relativeTo: baseURL).absoluteString
      )
      throw ResolverHarnessFailure.assertion("challenge response was accepted")
    } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
      try require(
        challengedURL?.path == "/challenge",
        "challenge response URL was not preserved"
      )
      return
    }
  }

  private static func verifyHEADFallback(relativeTo baseURL: URL) async throws {
    for path in [
      "head-fallback/challenge.m3u8",
      "head-fallback/failure.m3u8",
    ] {
      let source = try await IPadMediaURLResolver(requestTimeout: 3).resolve(
        endpoint(path, relativeTo: baseURL).absoluteString
      )
      try require(source.kind == .hls, "HEAD fallback did not resolve HLS: \(path)")
      try require(
        source.hlsPlaylist?.segments.count == 1,
        "bounded GET did not parse the fallback media playlist: \(path)"
      )
    }
  }

  private static func verifyChallengedRenditionFallback(
    relativeTo baseURL: URL
  ) async throws {
    let source = try await IPadMediaURLResolver(requestTimeout: 3).resolve(
      endpoint("variant-fallback/master.m3u8", relativeTo: baseURL).absoluteString
    )
    try require(source.kind == .hls, "master fallback did not return HLS")
    try require(
      source.mediaURL.path == "/variant-fallback/good.m3u8",
      "a challenged rendition prevented the next playable rendition"
    )
  }

  private static func verifyChallengePriority(relativeTo baseURL: URL) async throws {
    let resolver = IPadMediaURLResolver(requestTimeout: 3)
    for path in ["challenge-priority/page", "challenge-priority/master.m3u8"] {
      do {
        _ = try await resolver.resolve(endpoint(path, relativeTo: baseURL).absoluteString)
        throw ResolverHarnessFailure.assertion(
          "challenge was lost after another candidate failed: \(path)"
        )
      } catch IPadMediaURLResolverError.interactionRequired(let challengedURL) {
        try require(
          challengedURL?.path == "/challenge",
          "nested challenge response URL was not preserved: \(path)"
        )
        continue
      }
    }
  }

  private static func verifyCookieUpdates(relativeTo baseURL: URL) async throws {
    let host = try baseURL.host.unwrap(
      or: ResolverHarnessFailure.assertion("fixture host is missing")
    )
    let staleCookie = try IPadMediaRequestCookie(
      name: "stale_token",
      value: "remove-me",
      domain: host,
      path: "/cookie-rotate"
    ).unwrap(or: ResolverHarnessFailure.assertion("stale test cookie is invalid"))
    let context = IPadMediaRequestContext(
      cookies: [staleCookie],
      userAgent: "MiohCookieUpdateHarness/1.0",
      referer: endpoint("watch/page?secret=removed", relativeTo: baseURL)
    )
    let source = try await IPadMediaURLResolver(requestTimeout: 3).resolve(
      endpoint("cookie-rotate/start", relativeTo: baseURL).absoluteString,
      context: context
    )
    try require(source.kind == .hls, "rotating-cookie HLS was not resolved")
    let playlist = try source.hlsPlaylist.unwrap(
      or: ResolverHarnessFailure.assertion("rotating-cookie playlist is absent")
    )
    let segment = try playlist.segments.first.unwrap(
      or: ResolverHarnessFailure.assertion("rotating-cookie segment is absent")
    )
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mioh-cookie-update-harness-\(UUID().uuidString)",
      isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: directory) }
    let downloader = IPadHLSResourceDownloader(
      maximumResourceBytes: 1_024,
      requestTimeout: 3,
      requestContext: source.requestContext
    )
    let outputURL = try await downloader.materialize(
      segment: segment,
      in: directory
    )
    let outputData = try Data(contentsOf: outputURL)
    try require(
      outputData == Data("INITMEDIA".utf8),
      "updated cookies did not reach init and segment requests"
    )
    let cookieNames = Set(source.requestContext?.cookies.map(\.name) ?? [])
    try require(
      ["redirect_token", "master_token", "variant_token", "init_token"]
        .allSatisfy(cookieNames.contains),
      "response cookies were not retained in the shared context"
    )
    try require(
      !cookieNames.contains("stale_token"),
      "expired response cookie was not removed"
    )
  }

  private static func verifyPublicURLPolicy() throws {
    let rejected = [
      "https://127.1/media.m3u8",
      "https://10.1/media.m3u8",
      "https://2130706433/media.m3u8",
      "https://0x7f000001/media.m3u8",
      "https://localhost/media.m3u8",
    ]
    for rawValue in rejected {
      let url = try URL(string: rawValue).unwrap(
        or: ResolverHarnessFailure.assertion("invalid policy fixture")
      )
      try require(
        !IPadMediaURLResolver.isPublicHTTPSURL(url),
        "private or legacy address passed public policy: \(rawValue)"
      )
    }

    let publicURL = try URL(string: "https://1.1.1.1/media.m3u8").unwrap(
      or: ResolverHarnessFailure.assertion("invalid public policy fixture")
    )
    try require(
      IPadMediaURLResolver.isPublicHTTPSURL(publicURL),
      "public IPv4 address was rejected"
    )
  }
}

extension Optional {
  fileprivate func unwrap(or error: @autoclosure () -> Error) throws -> Wrapped {
    guard let self else { throw error() }
    return self
  }
}
