// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import Foundation
import NIOCore
import NIOSSH

public struct SFTPVersion: Sendable, Equatable {
    public var rawValue: UInt32

    public init(_ rawValue: UInt32) {
        self.rawValue = rawValue
    }

    public static let v3 = SFTPVersion(3)
}

public struct SFTPExtension: Sendable, Equatable {
    public var name: String
    public var data: ByteBuffer

    public init(name: String, data: ByteBuffer) {
        self.name = name
        self.data = data
    }
}

public enum SFTPExtensionName: String, CaseIterable, Sendable {
    case posixRename = "posix-rename@openssh.com"
    case statvfs = "statvfs@openssh.com"
    case fstatvfs = "fstatvfs@openssh.com"
    case hardlink = "hardlink@openssh.com"
    case fsync = "fsync@openssh.com"
    case copyData = "copy-data"

    var supportedVersion: String {
        switch self {
        case .statvfs, .fstatvfs:
            return "2"
        case .posixRename, .hardlink, .fsync, .copyData:
            return "1"
        }
    }
}

public struct SFTPServerCapabilities: Sendable, Equatable {
    public let rawExtensions: [SFTPExtension]

    private let advertisedVersionsByName: [String: [String]]

    public init(rawExtensions: [SFTPExtension]) {
        self.rawExtensions = rawExtensions
        self.advertisedVersionsByName = Dictionary(
            grouping: rawExtensions,
            by: \.name
        ).mapValues { entries in
            entries.map { String(decoding: $0.data.readableBytesView, as: UTF8.self) }
        }
    }

    public func isAdvertised(_ name: String) -> Bool {
        self.advertisedVersionsByName[name] != nil
    }

    public func advertisedVersions(for name: String) -> [String] {
        self.advertisedVersionsByName[name] ?? []
    }

    public func advertisedVersions(for extensionName: SFTPExtensionName) -> [String] {
        self.advertisedVersions(for: extensionName.rawValue)
    }

    public func supports(_ extensionName: SFTPExtensionName) -> Bool {
        self.advertisedVersions(for: extensionName).contains(extensionName.supportedVersion)
    }
}

public struct SFTPFileSystemFlags: OptionSet, Sendable, Hashable {
    public let rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    public static let readOnly = SFTPFileSystemFlags(rawValue: 0x1)
    public static let noSetUID = SFTPFileSystemFlags(rawValue: 0x2)
}

public struct SFTPFileSystemAttributes: Sendable, Equatable {
    public var blockSize: UInt64
    public var fundamentalBlockSize: UInt64
    public var totalBlocks: UInt64
    public var freeBlocks: UInt64
    public var availableBlocks: UInt64
    public var totalFileNodes: UInt64
    public var freeFileNodes: UInt64
    public var availableFileNodes: UInt64
    public var fileSystemID: UInt64
    public var flags: SFTPFileSystemFlags
    public var maximumNameLength: UInt64

    public init(
        blockSize: UInt64,
        fundamentalBlockSize: UInt64,
        totalBlocks: UInt64,
        freeBlocks: UInt64,
        availableBlocks: UInt64,
        totalFileNodes: UInt64,
        freeFileNodes: UInt64,
        availableFileNodes: UInt64,
        fileSystemID: UInt64,
        flags: SFTPFileSystemFlags,
        maximumNameLength: UInt64
    ) {
        self.blockSize = blockSize
        self.fundamentalBlockSize = fundamentalBlockSize
        self.totalBlocks = totalBlocks
        self.freeBlocks = freeBlocks
        self.availableBlocks = availableBlocks
        self.totalFileNodes = totalFileNodes
        self.freeFileNodes = freeFileNodes
        self.availableFileNodes = availableFileNodes
        self.fileSystemID = fileSystemID
        self.flags = flags
        self.maximumNameLength = maximumNameLength
    }
}

public struct SFTPFileHandle: Sendable, Equatable {
    public var bytes: ByteBuffer

    public init(bytes: ByteBuffer) {
        self.bytes = bytes
    }
}

public struct SFTPDirectoryHandle: Sendable, Equatable {
    public var bytes: ByteBuffer

    public init(bytes: ByteBuffer) {
        self.bytes = bytes
    }
}

public struct SFTPNameEntry: Sendable, Equatable {
    public var filename: String
    public var longname: String
    public var attributes: SFTPAttributes

    public init(filename: String, longname: String, attributes: SFTPAttributes) {
        self.filename = filename
        self.longname = longname
        self.attributes = attributes
    }
}

public struct SFTPOpenFlags: OptionSet, Sendable, Hashable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    public static let read = SFTPOpenFlags(rawValue: 0x00000001)
    public static let write = SFTPOpenFlags(rawValue: 0x00000002)
    public static let append = SFTPOpenFlags(rawValue: 0x00000004)
    public static let create = SFTPOpenFlags(rawValue: 0x00000008)
    public static let truncate = SFTPOpenFlags(rawValue: 0x00000010)
    public static let exclusive = SFTPOpenFlags(rawValue: 0x00000020)
}

public struct SFTPAttributeFlags: OptionSet, Sendable, Hashable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    public static let size = SFTPAttributeFlags(rawValue: 0x00000001)
    public static let uidgid = SFTPAttributeFlags(rawValue: 0x00000002)
    public static let permissions = SFTPAttributeFlags(rawValue: 0x00000004)
    public static let acmodtime = SFTPAttributeFlags(rawValue: 0x00000008)
    public static let extended = SFTPAttributeFlags(rawValue: 0x80000000)
    static let supported: SFTPAttributeFlags = [.size, .uidgid, .permissions, .acmodtime, .extended]
}

public struct SFTPAttributes: Sendable, Equatable {
    public var size: UInt64?
    public var uid: UInt32?
    public var gid: UInt32?
    public var permissions: UInt32?
    public var accessTime: UInt32?
    public var modificationTime: UInt32?
    public var extended: [SFTPExtension]

    public init(
        size: UInt64? = nil,
        uid: UInt32? = nil,
        gid: UInt32? = nil,
        permissions: UInt32? = nil,
        accessTime: UInt32? = nil,
        modificationTime: UInt32? = nil,
        extended: [SFTPExtension] = []
    ) {
        self.size = size
        self.uid = uid
        self.gid = gid
        self.permissions = permissions
        self.accessTime = accessTime
        self.modificationTime = modificationTime
        self.extended = extended
    }

    var flags: SFTPAttributeFlags {
        var flags: SFTPAttributeFlags = []
        if self.size != nil {
            flags.insert(.size)
        }
        if self.uid != nil || self.gid != nil {
            flags.insert(.uidgid)
        }
        if self.permissions != nil {
            flags.insert(.permissions)
        }
        if self.accessTime != nil || self.modificationTime != nil {
            flags.insert(.acmodtime)
        }
        if !self.extended.isEmpty {
            flags.insert(.extended)
        }
        return flags
    }
}

public enum SFTPStatusCode: Sendable, Equatable {
    case ok
    case eof
    case noSuchFile
    case permissionDenied
    case failure
    case badMessage
    case noConnection
    case connectionLost
    case operationUnsupported
    case unknown(UInt32)

    init(rawValue: UInt32) {
        switch rawValue {
        case 0: self = .ok
        case 1: self = .eof
        case 2: self = .noSuchFile
        case 3: self = .permissionDenied
        case 4: self = .failure
        case 5: self = .badMessage
        case 6: self = .noConnection
        case 7: self = .connectionLost
        case 8: self = .operationUnsupported
        default: self = .unknown(rawValue)
        }
    }

    var rawValue: UInt32 {
        switch self {
        case .ok: return 0
        case .eof: return 1
        case .noSuchFile: return 2
        case .permissionDenied: return 3
        case .failure: return 4
        case .badMessage: return 5
        case .noConnection: return 6
        case .connectionLost: return 7
        case .operationUnsupported: return 8
        case .unknown(let value): return value
        }
    }
}

public struct SFTPStatus: Sendable, Equatable {
    public var code: SFTPStatusCode
    public var message: String
    public var languageTag: String

    public init(code: SFTPStatusCode, message: String = "", languageTag: String = "") {
        self.code = code
        self.message = message
        self.languageTag = languageTag
    }
}

public enum SFTPRequestMessage: Sendable, Equatable {
    case open(path: String, pflags: SFTPOpenFlags, attributes: SFTPAttributes)
    case close(handle: ByteBuffer)
    case read(handle: ByteBuffer, offset: UInt64, length: UInt32)
    case write(handle: ByteBuffer, offset: UInt64, data: ByteBuffer)
    case lstat(path: String)
    case fstat(handle: ByteBuffer)
    case setstat(path: String, attributes: SFTPAttributes)
    case fsetstat(handle: ByteBuffer, attributes: SFTPAttributes)
    case opendir(path: String)
    case readdir(handle: ByteBuffer)
    case remove(path: String)
    case mkdir(path: String, attributes: SFTPAttributes)
    case rmdir(path: String)
    case realpath(path: String)
    case stat(path: String)
    case rename(oldPath: String, newPath: String)
    case readlink(path: String)
    case symlink(linkPath: String, targetPath: String)
    case extended(name: String, data: ByteBuffer)
}

public enum SFTPResponseMessage: Sendable, Equatable {
    case status(SFTPStatus)
    case handle(ByteBuffer)
    case data(ByteBuffer)
    case name([SFTPNameEntry])
    case attributes(SFTPAttributes)
    case extendedReply(ByteBuffer)
}

public enum SFTPError: Error, Sendable, Equatable {
    case invalidChannelType
    case sessionNotReady
    case subsystemRejected
    case unsupportedVersion(UInt32)
    case unsupportedExtension(SFTPExtensionName)
    case channelClosed
    case protocolViolation(String)
    case unexpectedResponse(String)
    case status(SFTPStatus)
    case invalidPath(String)
}

public struct SFTPClientEvent: Sendable, Equatable {
    public var standardError: ByteBuffer

    public init(standardError: ByteBuffer) {
        self.standardError = standardError
    }
}

public struct SFTPServerFileHandle: Sendable, Equatable {
    public var bytes: ByteBuffer

    public init(bytes: ByteBuffer) {
        self.bytes = bytes
    }
}

public struct SFTPServerDirectoryHandle: Sendable, Equatable {
    public var bytes: ByteBuffer

    public init(bytes: ByteBuffer) {
        self.bytes = bytes
    }
}

public struct SFTPServerContext: Sendable {
    public let channel: Channel

    public var eventLoop: EventLoop {
        self.channel.eventLoop
    }

    public var allocator: ByteBufferAllocator {
        self.channel.allocator
    }

    public init(channel: Channel) {
        self.channel = channel
    }
}

public protocol SFTPServerBackend: Sendable {
    var advertisedExtensions: [SFTPExtension] { get }

    func open(
        path: String,
        flags: SFTPOpenFlags,
        attributes: SFTPAttributes,
        context: SFTPServerContext
    ) -> EventLoopFuture<SFTPServerFileHandle>
    func close(fileHandle: SFTPServerFileHandle, context: SFTPServerContext) -> EventLoopFuture<Void>
    func close(directoryHandle: SFTPServerDirectoryHandle, context: SFTPServerContext) -> EventLoopFuture<Void>
    func read(
        fileHandle: SFTPServerFileHandle,
        offset: UInt64,
        length: UInt32,
        context: SFTPServerContext
    ) -> EventLoopFuture<ByteBuffer?>
    func write(
        fileHandle: SFTPServerFileHandle,
        offset: UInt64,
        data: ByteBuffer,
        context: SFTPServerContext
    ) -> EventLoopFuture<Void>
    func stat(path: String, context: SFTPServerContext) -> EventLoopFuture<SFTPAttributes>
    func lstat(path: String, context: SFTPServerContext) -> EventLoopFuture<SFTPAttributes>
    func fstat(fileHandle: SFTPServerFileHandle, context: SFTPServerContext) -> EventLoopFuture<SFTPAttributes>
    func setstat(path: String, attributes: SFTPAttributes, context: SFTPServerContext) -> EventLoopFuture<Void>
    func fsetstat(
        fileHandle: SFTPServerFileHandle,
        attributes: SFTPAttributes,
        context: SFTPServerContext
    ) -> EventLoopFuture<Void>
    func opendir(path: String, context: SFTPServerContext) -> EventLoopFuture<SFTPServerDirectoryHandle>
    func readdir(
        directoryHandle: SFTPServerDirectoryHandle,
        context: SFTPServerContext
    ) -> EventLoopFuture<[SFTPNameEntry]?>
    func remove(path: String, context: SFTPServerContext) -> EventLoopFuture<Void>
    func mkdir(path: String, attributes: SFTPAttributes, context: SFTPServerContext) -> EventLoopFuture<Void>
    func rmdir(path: String, context: SFTPServerContext) -> EventLoopFuture<Void>
    func realpath(path: String, context: SFTPServerContext) -> EventLoopFuture<String>
    func rename(oldPath: String, newPath: String, context: SFTPServerContext) -> EventLoopFuture<Void>
    func readlink(path: String, context: SFTPServerContext) -> EventLoopFuture<String>
    func symlink(linkPath: String, targetPath: String, context: SFTPServerContext) -> EventLoopFuture<Void>
    func posixRename(oldPath: String, newPath: String, context: SFTPServerContext) -> EventLoopFuture<Void>
    func fsync(fileHandle: SFTPServerFileHandle, context: SFTPServerContext) -> EventLoopFuture<Void>
    func statvfs(path: String, context: SFTPServerContext) -> EventLoopFuture<SFTPFileSystemAttributes>
    func fstatvfs(
        fileHandle: SFTPServerFileHandle,
        context: SFTPServerContext
    ) -> EventLoopFuture<SFTPFileSystemAttributes>
    func hardlink(oldPath: String, newPath: String, context: SFTPServerContext) -> EventLoopFuture<Void>
    func copyData(
        from source: SFTPServerFileHandle,
        readOffset: UInt64,
        length: UInt64,
        to destination: SFTPServerFileHandle,
        writeOffset: UInt64,
        context: SFTPServerContext
    ) -> EventLoopFuture<Void>
}

public final class SFTPServer: @unchecked Sendable {
    public let channel: Channel
    public let backend: any SFTPServerBackend

    private let handler: SFTPServerHandler

    private init(channel: Channel, backend: any SFTPServerBackend, handler: SFTPServerHandler) {
        self.channel = channel
        self.backend = backend
        self.handler = handler
    }

    public static func start(on channel: Channel, backend: any SFTPServerBackend) -> EventLoopFuture<SFTPServer> {
        let handler = SFTPServerHandler(loop: channel.eventLoop, allocator: channel.allocator, backend: backend)
        return channel.pipeline.addHandler(handler).map {
            SFTPServer(channel: channel, backend: backend, handler: handler)
        }
    }
}

