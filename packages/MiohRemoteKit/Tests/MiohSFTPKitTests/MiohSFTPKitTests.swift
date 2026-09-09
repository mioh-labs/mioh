import XCTest
@testable import MiohSFTPKit

final class MiohSFTPKitTests: XCTestCase {
  func testConfigurationNormalizesIPv6AndBuildsStableEndpointID() throws {
    let configuration = try MiohSFTPConfiguration(
      host: " [2001:db8::1] ",
      port: 2222,
      username: " alice ",
      password: "secret"
    ).validated()

    XCTAssertEqual(configuration.host, "2001:db8::1")
    XCTAssertEqual(configuration.username, "alice")
    XCTAssertEqual(configuration.endpointID, "[2001:db8::1]:2222")
    XCTAssertEqual(configuration.credentialID, "[2001:db8::1]:2222|alice")
  }

  func testConfigurationRejectsURLAndMissingCredentials() {
    XCTAssertThrowsError(
      try MiohSFTPConfiguration(
        host: "sftp://example.com",
        username: "alice",
        password: "secret"
      ).validated()
    )
    XCTAssertThrowsError(
      try MiohSFTPConfiguration(
        host: "example.com",
        port: 0,
        username: "alice",
        password: "secret"
      ).validated()
    )
    XCTAssertThrowsError(
      try MiohSFTPConfiguration(
        host: "[2001:db8::1",
        username: "alice",
        password: "secret"
      ).validated()
    )
    XCTAssertThrowsError(
      try MiohSFTPConfiguration(
        host: "example.com",
        username: "ali\nce",
        password: "secret"
      ).validated()
    )
    XCTAssertThrowsError(
      try MiohSFTPConfiguration(
        host: "example.com",
        username: "alice",
        password: "secret\0suffix"
      ).validated()
    )
    XCTAssertThrowsError(
      try MiohSFTPConfiguration(
        host: "example.com",
        username: "",
        password: "secret"
      ).validated()
    )
    XCTAssertThrowsError(
      try MiohSFTPConfiguration(
        host: "example.com",
        username: "alice",
        password: ""
      ).validated()
    )
  }

  func testHostIdentityUsesOpenSSHSHA256Fingerprint() throws {
    let identity = try MiohSFTPHostIdentity(shortHandKey: "ssh-ed25519 YWJj comment")

    XCTAssertEqual(identity.algorithm, "ssh-ed25519")
    XCTAssertEqual(identity.key, "ssh-ed25519 YWJj")
    XCTAssertEqual(
      identity.fingerprint,
      "SHA256:ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0"
    )
  }

  func testRemotePathNormalizationCannotEscapeRoot() throws {
    XCTAssertEqual(
      try MiohSFTPPath.normalize("../movies/./clip.mp4", relativeTo: "/home/alice"),
      "/home/movies/clip.mp4"
    )
    XCTAssertEqual(try MiohSFTPPath.normalize("../../..", relativeTo: "/home"), "/")
    XCTAssertEqual(try MiohSFTPPath.parent(of: "/home/alice"), "/home")
    XCTAssertEqual(try MiohSFTPPath.parent(of: "/"), "/")
    XCTAssertEqual(
      try MiohSFTPPath.appending(" folder ", to: "/home"),
      "/home/ folder "
    )
    XCTAssertEqual(try MiohSFTPPath.normalize("/home/ folder "), "/home/ folder ")
    XCTAssertThrowsError(try MiohSFTPPath.appending("../clip.mp4", to: "/home"))
    XCTAssertThrowsError(try MiohSFTPPath.appending("bad/name.mp4", to: "/home"))
  }

  func testMovieExtensionFilterIsCaseInsensitive() {
    XCTAssertTrue(MiohSFTPPath.isSupportedMovie("clip.MP4"))
    XCTAssertTrue(MiohSFTPPath.isSupportedMovie("movie.mov"))
    XCTAssertTrue(MiohSFTPPath.isSupportedMovie("movie.m4v"))
    XCTAssertFalse(MiohSFTPPath.isSupportedMovie("movie.mkv"))
    XCTAssertFalse(MiohSFTPPath.isSupportedMovie(".mp4"))
  }

  func testDownloadCapacityReservesSpaceAndAppliesHardLimit() throws {
    let gib: Int64 = 1_024 * 1_024 * 1_024
    XCTAssertEqual(
      try MiohSFTPTransferPolicy.maximumDownloadBytes(availableBytes: 20 * gib),
      18 * gib
    )
    XCTAssertEqual(
      try MiohSFTPTransferPolicy.maximumDownloadBytes(availableBytes: 30 * gib),
      20 * gib
    )
    XCTAssertThrowsError(
      try MiohSFTPTransferPolicy.maximumDownloadBytes(availableBytes: gib)
    )
    XCTAssertEqual(
      try MiohSFTPTransferPolicy.maximumResumableDownloadBytes(
        availableBytes: 10 * gib,
        retainedPartialBytes: 3 * gib
      ),
      12 * gib
    )
    XCTAssertEqual(
      try MiohSFTPTransferPolicy.maximumResumableDownloadBytes(
        availableBytes: 30 * gib,
        retainedPartialBytes: 19 * gib
      ),
      20 * gib
    )
  }

  func testResumeMetadataReusesOnlyMatchingRegularPartial() throws {
    let fileManager = FileManager.default
    let directory = fileManager.temporaryDirectory.appendingPathComponent(
      "mioh-sftp-resume-tests-\(UUID().uuidString)",
      isDirectory: true
    )
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? fileManager.removeItem(at: directory) }

    let destination = directory.appendingPathComponent("movie.mp4")
    let partial = destination.appendingPathExtension("part")
    let resume = destination.appendingPathExtension("resume")
    let metadata = MiohSFTPDownloadResumeMetadata(
      remotePath: "/movies/movie.mp4",
      expectedSize: 100,
      expectedModificationTime: 123
    )

    XCTAssertEqual(
      try MiohSFTPSession.prepareLocalDownload(
        partialURL: partial,
        metadataURL: resume,
        metadata: metadata,
        allowsResume: true,
        fileManager: fileManager
      ),
      0
    )
    try Data("verified-prefix".utf8).write(to: partial)
    XCTAssertEqual(
      try MiohSFTPSession.prepareLocalDownload(
        partialURL: partial,
        metadataURL: resume,
        metadata: metadata,
        allowsResume: true,
        fileManager: fileManager
      ),
      UInt64(Data("verified-prefix".utf8).count)
    )

    let changedRemote = MiohSFTPDownloadResumeMetadata(
      remotePath: metadata.remotePath,
      expectedSize: metadata.expectedSize,
      expectedModificationTime: 124
    )
    XCTAssertEqual(
      try MiohSFTPSession.prepareLocalDownload(
        partialURL: partial,
        metadataURL: resume,
        metadata: changedRemote,
        allowsResume: true,
        fileManager: fileManager
      ),
      0
    )
    let resetAttributes = try fileManager.attributesOfItem(atPath: partial.path)
    XCTAssertEqual((resetAttributes[.size] as? NSNumber)?.intValue, 0)
  }

  func testResumePreparationReplacesSymlinkWithoutTouchingItsTarget() throws {
    let fileManager = FileManager.default
    let directory = fileManager.temporaryDirectory.appendingPathComponent(
      "mioh-sftp-resume-symlink-tests-\(UUID().uuidString)",
      isDirectory: true
    )
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? fileManager.removeItem(at: directory) }

    let destination = directory.appendingPathComponent("movie.mp4")
    let partial = destination.appendingPathExtension("part")
    let resume = destination.appendingPathExtension("resume")
    let target = directory.appendingPathComponent("must-not-change")
    let sentinel = Data("sentinel".utf8)
    try sentinel.write(to: target)
    try fileManager.createSymbolicLink(
      atPath: partial.path,
      withDestinationPath: target.path
    )

    let metadata = MiohSFTPDownloadResumeMetadata(
      remotePath: "/movies/movie.mp4",
      expectedSize: 100,
      expectedModificationTime: 123
    )
    XCTAssertEqual(
      try MiohSFTPSession.prepareLocalDownload(
        partialURL: partial,
        metadataURL: resume,
        metadata: metadata,
        allowsResume: true,
        fileManager: fileManager
      ),
      0
    )
    XCTAssertEqual(try Data(contentsOf: target), sentinel)
    let values = try partial.resourceValues(forKeys: [
      .isRegularFileKey,
      .isSymbolicLinkKey,
    ])
    XCTAssertEqual(values.isRegularFile, true)
    XCTAssertNotEqual(values.isSymbolicLink, true)
  }
}
