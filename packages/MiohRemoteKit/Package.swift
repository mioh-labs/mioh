// swift-tools-version: 5.9

import PackageDescription

let package = Package(
  name: "MiohRemoteKit",
  platforms: [
    .iOS(.v16),
    .macOS(.v13),
  ],
  products: [
    .library(name: "MiohRemoteKit", targets: ["MiohRemoteKit"]),
    .library(name: "MiohSFTPKit", targets: ["MiohSFTPKit"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/apple/swift-nio.git",
      exact: "2.101.3"
    ),
    .package(
      url: "https://github.com/apple/swift-nio-ssh.git",
      exact: "0.15.0"
    ),
  ],
  targets: [
    .target(name: "MiohRemoteKit"),
    .target(
      name: "NIOSFTP",
      dependencies: [
        .product(name: "NIOCore", package: "swift-nio"),
        .product(name: "NIOPosix", package: "swift-nio"),
        .product(name: "NIOSSH", package: "swift-nio-ssh"),
      ]
    ),
    .target(
      name: "MiohSFTPKit",
      dependencies: [
        "NIOSFTP",
        .product(name: "NIOCore", package: "swift-nio"),
        .product(name: "NIOPosix", package: "swift-nio"),
        .product(name: "NIOSSH", package: "swift-nio-ssh"),
      ]
    ),
    .testTarget(name: "MiohRemoteKitTests", dependencies: ["MiohRemoteKit"]),
    .testTarget(name: "MiohSFTPKitTests", dependencies: ["MiohSFTPKit"]),
    .testTarget(name: "NIOSFTPSecurityTests", dependencies: ["NIOSFTP"]),
  ]
)
