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
  ],
  targets: [
    .target(name: "MiohRemoteKit"),
    .testTarget(name: "MiohRemoteKitTests", dependencies: ["MiohRemoteKit"]),
  ]
)
