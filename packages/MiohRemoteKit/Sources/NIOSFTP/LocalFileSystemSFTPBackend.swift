// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import Foundation
import NIOCore

#if canImport(Darwin)
import Darwin
private typealias PlatformStat = Darwin.stat
private typealias PlatformStatFS = Darwin.statfs
#elseif canImport(Glibc)
import Glibc
private typealias PlatformStat = Glibc.stat
private typealias PlatformStatFS = Glibc.statfs
#endif

public final class LocalFileSystemSFTPBackend: SFTPServerBackend, @unchecked Sendable {
    public let advertisedExtensions: [SFTPExtension]

    private struct OpenFile {
        var hostPath: String
        var descriptor: Int32
    }

    private struct OpenDirectory {
        var hostPath: String
        var entries: [SFTPNameEntry]
        var offset: Int
    }

    private let rootURL: URL
    private let rootPath: String
    private let fileManager = FileManager.default
    private var fileHandles: [String: OpenFile] = [:]
    private var directoryHandles: [String: OpenDirectory] = [:]
    private var nextHandleID: UInt64 = 0

    public init(rootPath: String, advertisedExtensions: [SFTPExtension]? = nil) throws {
        let rootURL = URL(fileURLWithPath: rootPath, isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: rootURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw SFTPError.invalidPath(rootPath)
        }

        self.rootURL = rootURL
        self.rootPath = rootURL.path
        self.advertisedExtensions = advertisedExtensions ?? [
            .init(name: SFTPExtensionName.posixRename.rawValue, data: ByteBuffer(string: "1")),
            .init(name: SFTPExtensionName.statvfs.rawValue, data: ByteBuffer(string: "2")),
            .init(name: SFTPExtensionName.fstatvfs.rawValue, data: ByteBuffer(string: "2")),
            .init(name: SFTPExtensionName.hardlink.rawValue, data: ByteBuffer(string: "1")),
            .init(name: SFTPExtensionName.fsync.rawValue, data: ByteBuffer(string: "1")),
            .init(name: SFTPExtensionName.copyData.rawValue, data: ByteBuffer(string: "1")),
        ]
    }

    public func open(
        path: String,
        flags: SFTPOpenFlags,
        attributes: SFTPAttributes,
        context: SFTPServerContext
    ) -> EventLoopFuture<SFTPServerFileHandle> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: true)
            let descriptor = Self.posixOpen(hostPath, self.openMode(flags), mode_t(attributes.permissions ?? 0o644))
            guard descriptor >= 0 else {
                throw Self.errorForErrno(path: path)
            }
            let handle = self.makeFileHandle(prefix: "file")
            let key = String(buffer: handle.bytes)
            self.fileHandles[key] = .init(hostPath: hostPath, descriptor: descriptor)
            return handle
        }
    }

    public func close(fileHandle: SFTPServerFileHandle, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let key = String(buffer: fileHandle.bytes)
            guard let openFile = self.fileHandles.removeValue(forKey: key) else {
                throw SFTPError.status(.init(code: .failure, message: "Invalid file handle"))
            }
            guard Self.posixClose(openFile.descriptor) == 0 else {
                throw Self.errorForErrno(path: openFile.hostPath)
            }
        }
    }

    public func close(directoryHandle: SFTPServerDirectoryHandle, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let key = String(buffer: directoryHandle.bytes)
            guard self.directoryHandles.removeValue(forKey: key) != nil else {
                throw SFTPError.status(.init(code: .failure, message: "Invalid directory handle"))
            }
        }
    }

    public func read(
        fileHandle: SFTPServerFileHandle,
        offset: UInt64,
        length: UInt32,
        context: SFTPServerContext
    ) -> EventLoopFuture<ByteBuffer?> {
        context.eventLoop.makeCompletedFuture {
            let openFile = try self.openFile(for: fileHandle)
            // A peer controls `length`. Keep one request from forcing a multi-GB
            // allocation even when this optional server backend is embedded.
            let boundedLength = min(length, SFTPParsingLimits.maximumServerReadBytes)
            var bytes = [UInt8](repeating: 0, count: Int(boundedLength))
            let readCount = Self.posixPRead(openFile.descriptor, &bytes, bytes.count, off_t(offset))
            guard readCount >= 0 else {
                throw Self.errorForErrno(path: openFile.hostPath)
            }
            guard readCount > 0 else {
                return nil
            }
            return ByteBuffer(bytes: bytes.prefix(Int(readCount)))
        }
    }

    public func write(
        fileHandle: SFTPServerFileHandle,
        offset: UInt64,
        data: ByteBuffer,
        context: SFTPServerContext
    ) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let openFile = try self.openFile(for: fileHandle)
            let bytes = Array(data.readableBytesView)
            var written = 0
            while written < bytes.count {
                let chunk = bytes.withUnsafeBytes { rawBuffer in
                    Self.posixPWrite(
                        openFile.descriptor,
                        rawBuffer.baseAddress!.advanced(by: written),
                        bytes.count - written,
                        off_t(offset) + off_t(written)
                    )
                }
                guard chunk >= 0 else {
                    throw Self.errorForErrno(path: openFile.hostPath)
                }
                written += chunk
            }
        }
    }

    public func stat(path: String, context: SFTPServerContext) -> EventLoopFuture<SFTPAttributes> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: true)
            return try Self.attributes(for: hostPath, followSymlink: true)
        }
    }

    public func lstat(path: String, context: SFTPServerContext) -> EventLoopFuture<SFTPAttributes> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: false)
            return try Self.attributes(for: hostPath, followSymlink: false)
        }
    }

    public func fstat(fileHandle: SFTPServerFileHandle, context: SFTPServerContext) -> EventLoopFuture<SFTPAttributes> {
        context.eventLoop.makeCompletedFuture {
            let openFile = try self.openFile(for: fileHandle)
            var statInfo = PlatformStat()
            guard Self.posixFStat(openFile.descriptor, &statInfo) == 0 else {
                throw Self.errorForErrno(path: openFile.hostPath)
            }
            return Self.attributes(from: statInfo)
        }
    }

    public func setstat(path: String, attributes: SFTPAttributes, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: true)
            try self.apply(attributes: attributes, toHostPath: hostPath, descriptor: nil)
        }
    }

    public func fsetstat(
        fileHandle: SFTPServerFileHandle,
        attributes: SFTPAttributes,
        context: SFTPServerContext
    ) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let openFile = try self.openFile(for: fileHandle)
            try self.apply(attributes: attributes, toHostPath: openFile.hostPath, descriptor: openFile.descriptor)
        }
    }

    public func opendir(path: String, context: SFTPServerContext) -> EventLoopFuture<SFTPServerDirectoryHandle> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: true)
            let entries = try self.readDirectoryEntries(at: hostPath)
            let handle = self.makeDirectoryHandle(prefix: "dir")
            self.directoryHandles[String(buffer: handle.bytes)] = .init(hostPath: hostPath, entries: entries, offset: 0)
            return handle
        }
    }

    public func readdir(
        directoryHandle: SFTPServerDirectoryHandle,
        context: SFTPServerContext
    ) -> EventLoopFuture<[SFTPNameEntry]?> {
        context.eventLoop.makeCompletedFuture {
            let key = String(buffer: directoryHandle.bytes)
            guard var directory = self.directoryHandles[key] else {
                throw SFTPError.status(.init(code: .failure, message: "Invalid directory handle"))
            }
            guard directory.offset < directory.entries.count else {
                return nil
            }
            let end = min(
                directory.entries.count,
                directory.offset + SFTPParsingLimits.maximumServerDirectoryBatchEntries
            )
            let batch = Array(directory.entries[directory.offset..<end])
            directory.offset = end
            self.directoryHandles[key] = directory
            return batch
        }
    }

    public func remove(path: String, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: false)
            guard Self.posixUnlink(hostPath) == 0 else {
                throw Self.errorForErrno(path: path)
            }
        }
    }

    public func mkdir(path: String, attributes: SFTPAttributes, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPathLexically(for: path)
            guard Self.posixMkdir(hostPath, mode_t(attributes.permissions ?? 0o755)) == 0 else {
                throw Self.errorForErrno(path: path)
            }
        }
    }

    public func rmdir(path: String, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: false)
            guard Self.posixRmdir(hostPath) == 0 else {
                throw Self.errorForErrno(path: path)
            }
        }
    }

    public func realpath(path: String, context: SFTPServerContext) -> EventLoopFuture<String> {
        context.eventLoop.makeCompletedFuture {
            if path.isEmpty || path == "." {
                return "/"
            }
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: true)
            return try self.clientPath(forHostPath: hostPath)
        }
    }

    public func rename(oldPath: String, newPath: String, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let oldHostPath = try self.resolveHostPath(for: oldPath, followFinalSymlink: true)
            let newHostPath = try self.resolveHostPathLexically(for: newPath)
            guard !self.fileManager.fileExists(atPath: newHostPath) else {
                throw SFTPError.status(.init(code: .failure, message: "Failure"))
            }
            guard Self.posixRename(oldHostPath, newHostPath) == 0 else {
                throw Self.errorForErrno(path: oldPath)
            }
        }
    }

    public func readlink(path: String, context: SFTPServerContext) -> EventLoopFuture<String> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: false)
            let target = try self.readlinkTarget(for: hostPath)
            return try self.clientVisibleLinkTarget(storedTarget: target, parentHostPath: (hostPath as NSString).deletingLastPathComponent)
        }
    }

    public func symlink(linkPath: String, targetPath: String, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let linkHostPath = try self.resolveHostPathLexically(for: linkPath)
            let storedTarget = try self.storedSymlinkTarget(fromClientTarget: targetPath, parentHostPath: (linkHostPath as NSString).deletingLastPathComponent)
            guard Self.posixSymlink(storedTarget, linkHostPath) == 0 else {
                throw Self.errorForErrno(path: linkPath)
            }
        }
    }

    public func posixRename(oldPath: String, newPath: String, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let oldHostPath = try self.resolveHostPath(for: oldPath, followFinalSymlink: true)
            let newHostPath = try self.resolveHostPathLexically(for: newPath)
            guard Self.posixRename(oldHostPath, newHostPath) == 0 else {
                throw Self.errorForErrno(path: oldPath)
            }
        }
    }

    public func fsync(fileHandle: SFTPServerFileHandle, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let openFile = try self.openFile(for: fileHandle)
            guard Self.posixFSync(openFile.descriptor) == 0 else {
                throw Self.errorForErrno(path: openFile.hostPath)
            }
        }
    }

    public func statvfs(path: String, context: SFTPServerContext) -> EventLoopFuture<SFTPFileSystemAttributes> {
        context.eventLoop.makeCompletedFuture {
            let hostPath = try self.resolveHostPath(for: path, followFinalSymlink: true)
            let descriptor = Self.posixOpen(hostPath, O_RDONLY, 0)
            guard descriptor >= 0 else {
                throw Self.errorForErrno(path: path)
            }
            defer { _ = Self.posixClose(descriptor) }
            return try Self.fileSystemAttributes(forDescriptor: descriptor)
        }
    }

    public func fstatvfs(
        fileHandle: SFTPServerFileHandle,
        context: SFTPServerContext
    ) -> EventLoopFuture<SFTPFileSystemAttributes> {
        context.eventLoop.makeCompletedFuture {
            let openFile = try self.openFile(for: fileHandle)
            return try Self.fileSystemAttributes(forDescriptor: openFile.descriptor)
        }
    }

    public func hardlink(oldPath: String, newPath: String, context: SFTPServerContext) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let source = try self.resolveHostPath(for: oldPath, followFinalSymlink: true)
            let target = try self.resolveHostPathLexically(for: newPath)
            guard Self.posixLink(source, target) == 0 else {
                throw Self.errorForErrno(path: oldPath)
            }
        }
    }

    public func copyData(
        from source: SFTPServerFileHandle,
        readOffset: UInt64,
        length: UInt64,
        to destination: SFTPServerFileHandle,
        writeOffset: UInt64,
        context: SFTPServerContext
    ) -> EventLoopFuture<Void> {
        context.eventLoop.makeCompletedFuture {
            let sourceFile = try self.openFile(for: source)
            let destinationFile = try self.openFile(for: destination)
            let totalBytes = try self.fileSize(descriptor: sourceFile.descriptor)
            if readOffset > totalBytes {
                throw SFTPError.status(.init(code: .eof))
            }
            let finalLength = length == 0 ? totalBytes - readOffset : min(length, totalBytes - readOffset)
            let chunkSize = 32 * 1024
            var copied: UInt64 = 0
            while copied < finalLength {
                let chunk = Int(min(UInt64(chunkSize), finalLength - copied))
                var bytes = [UInt8](repeating: 0, count: chunk)
                let readCount = Self.posixPRead(sourceFile.descriptor, &bytes, chunk, off_t(readOffset + copied))
                guard readCount >= 0 else {
                    throw Self.errorForErrno(path: sourceFile.hostPath)
                }
                guard readCount > 0 else {
                    break
                }
                var written = 0
                while written < readCount {
                    let writeCount = bytes.withUnsafeBytes { rawBuffer in
                        Self.posixPWrite(
                            destinationFile.descriptor,
                            rawBuffer.baseAddress!.advanced(by: written),
                            readCount - written,
                            off_t(writeOffset + copied + UInt64(written))
                        )
                    }
                    guard writeCount >= 0 else {
                        throw Self.errorForErrno(path: destinationFile.hostPath)
                    }
                    written += writeCount
                }
                copied += UInt64(readCount)
            }
        }
    }

    private func openFile(for handle: SFTPServerFileHandle) throws -> OpenFile {
        let key = String(buffer: handle.bytes)
        guard let openFile = self.fileHandles[key] else {
            throw SFTPError.status(.init(code: .failure, message: "Invalid file handle"))
        }
        return openFile
    }

    private func makeFileHandle(prefix: String) -> SFTPServerFileHandle {
        defer { self.nextHandleID &+= 1 }
        return .init(bytes: ByteBuffer(string: "\(prefix)-\(self.nextHandleID)"))
    }

    private func makeDirectoryHandle(prefix: String) -> SFTPServerDirectoryHandle {
        defer { self.nextHandleID &+= 1 }
        return .init(bytes: ByteBuffer(string: "\(prefix)-\(self.nextHandleID)"))
    }

    private func readDirectoryEntries(at hostPath: String) throws -> [SFTPNameEntry] {
        let contents = try self.fileManager.contentsOfDirectory(atPath: hostPath).sorted()
        var entries: [SFTPNameEntry] = [
            .init(filename: ".", longname: ".", attributes: try Self.attributes(for: hostPath, followSymlink: false)),
            .init(filename: "..", longname: "..", attributes: try Self.attributes(for: (hostPath as NSString).deletingLastPathComponent, followSymlink: false)),
        ]
        for name in contents {
            let child = (hostPath as NSString).appendingPathComponent(name)
            entries.append(.init(filename: name, longname: name, attributes: try Self.attributes(for: child, followSymlink: false)))
        }
        return entries
    }

    private func apply(attributes: SFTPAttributes, toHostPath hostPath: String, descriptor: Int32?) throws {
        if let size = attributes.size {
            if let descriptor {
                guard Self.posixFTruncate(descriptor, off_t(size)) == 0 else {
                    throw Self.errorForErrno(path: hostPath)
                }
            } else {
                guard Self.posixTruncate(hostPath, off_t(size)) == 0 else {
                    throw Self.errorForErrno(path: hostPath)
                }
            }
        }
        if let permissions = attributes.permissions {
            if let descriptor {
                guard Self.posixFChmod(descriptor, mode_t(permissions & 0o7777)) == 0 else {
                    throw Self.errorForErrno(path: hostPath)
                }
            } else {
                guard Self.posixChmod(hostPath, mode_t(permissions & 0o7777)) == 0 else {
                    throw Self.errorForErrno(path: hostPath)
                }
            }
        }
        if attributes.accessTime != nil || attributes.modificationTime != nil {
            let statInfo = try Self.statInfo(for: hostPath, followSymlink: true)
            let access = timeval(tv_sec: attributes.accessTime.map { Int($0) } ?? Int(statInfo.st_atimespec.tv_sec), tv_usec: 0)
            let modified = timeval(tv_sec: attributes.modificationTime.map { Int($0) } ?? Int(statInfo.st_mtimespec.tv_sec), tv_usec: 0)
            var times = [access, modified]
            let result: Int32
            if let descriptor {
                result = Self.posixFUTimes(descriptor, &times)
            } else {
                result = Self.posixUTimes(hostPath, &times)
            }
            guard result == 0 else {
                throw Self.errorForErrno(path: hostPath)
            }
        }
    }

    private func resolveHostPathLexically(for clientPath: String) throws -> String {
        let components = try self.normalizedComponents(for: clientPath)
        var current = self.rootPath
        for component in components {
            current = (current as NSString).appendingPathComponent(component)
        }
        guard self.isInsideRoot(current) else {
            throw SFTPError.invalidPath(clientPath)
        }
        return current
    }

    private func resolveHostPath(for clientPath: String, followFinalSymlink: Bool) throws -> String {
        let components = try self.normalizedComponents(for: clientPath)
        var current = self.rootPath
        for (index, component) in components.enumerated() {
            current = (current as NSString).appendingPathComponent(component)
            let shouldResolve = index < components.count - 1 || followFinalSymlink
            if shouldResolve {
                current = try self.resolveSymlinksIfNeeded(at: current)
            }
            guard self.isInsideRoot(current) else {
                throw SFTPError.invalidPath(clientPath)
            }
        }
        return current
    }

    private func normalizedComponents(for clientPath: String) throws -> [String] {
        let rawComponents = clientPath.split(separator: "/", omittingEmptySubsequences: true)
        var normalized: [String] = []
        for component in rawComponents {
            switch component {
            case ".":
                continue
            case "..":
                guard !normalized.isEmpty else {
                    throw SFTPError.invalidPath(clientPath)
                }
                normalized.removeLast()
            default:
                normalized.append(String(component))
            }
        }
        return normalized
    }

    private func resolveSymlinksIfNeeded(at hostPath: String) throws -> String {
        var currentPath = hostPath
        while true {
            var statInfo = PlatformStat()
            guard Self.posixLStat(currentPath, &statInfo) == 0 else {
                if errno == ENOENT {
                    return currentPath
                }
                throw Self.errorForErrno(path: currentPath)
            }
            guard (statInfo.st_mode & S_IFMT) == S_IFLNK else {
                return currentPath
            }
            let target = try self.readlinkTarget(for: currentPath)
            if target.hasPrefix("/") {
                currentPath = target
            } else {
                currentPath = URL(fileURLWithPath: (currentPath as NSString).deletingLastPathComponent)
                    .appendingPathComponent(target).standardizedFileURL.path
            }
            guard self.isInsideRoot(currentPath) else {
                throw SFTPError.invalidPath(hostPath)
            }
        }
    }

    private func readlinkTarget(for hostPath: String) throws -> String {
        let size = Int(PATH_MAX)
        var buffer = [CChar](repeating: 0, count: size)
        let readCount = Self.posixReadlink(hostPath, &buffer, size)
        guard readCount >= 0 else {
            throw Self.errorForErrno(path: hostPath)
        }
        return String(decoding: buffer.prefix(Int(readCount)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    private func storedSymlinkTarget(fromClientTarget targetPath: String, parentHostPath: String) throws -> String {
        if targetPath.hasPrefix("/") {
            return try self.resolveHostPathLexically(for: targetPath)
        }
        let combined = URL(fileURLWithPath: parentHostPath).appendingPathComponent(targetPath).standardizedFileURL.path
        guard self.isInsideRoot(combined) else {
            throw SFTPError.invalidPath(targetPath)
        }
        return targetPath
    }

    private func clientVisibleLinkTarget(storedTarget: String, parentHostPath: String) throws -> String {
        if storedTarget.hasPrefix("/") {
            return try self.clientPath(forHostPath: storedTarget)
        }
        let combined = URL(fileURLWithPath: parentHostPath).appendingPathComponent(storedTarget).standardizedFileURL.path
        guard self.isInsideRoot(combined) else {
            throw SFTPError.invalidPath(storedTarget)
        }
        return storedTarget
    }

    private func clientPath(forHostPath hostPath: String) throws -> String {
        let resolved = URL(fileURLWithPath: hostPath).standardizedFileURL.path
        guard self.isInsideRoot(resolved) else {
            throw SFTPError.invalidPath(hostPath)
        }
        if resolved == self.rootPath {
            return "/"
        }
        let suffix = String(resolved.dropFirst(self.rootPath.count))
        return suffix.hasPrefix("/") ? suffix : "/\(suffix)"
    }

    private func isInsideRoot(_ hostPath: String) -> Bool {
        let standardized = URL(fileURLWithPath: hostPath).standardizedFileURL.path
        return standardized == self.rootPath || standardized.hasPrefix(self.rootPath + "/")
    }

    private func fileSize(descriptor: Int32) throws -> UInt64 {
        var statInfo = PlatformStat()
        guard Self.posixFStat(descriptor, &statInfo) == 0 else {
            throw SFTPError.status(.init(code: .failure, message: "Failed to stat file"))
        }
        return UInt64(statInfo.st_size)
    }

    private static func statInfo(for hostPath: String, followSymlink _: Bool) throws -> PlatformStat {
        var statInfo = PlatformStat()
        let result = Self.posixLStat(hostPath, &statInfo)
        guard result == 0 else {
            throw Self.errorForErrno(path: hostPath)
        }
        return statInfo
    }

    private static func attributes(for hostPath: String, followSymlink: Bool) throws -> SFTPAttributes {
        try self.attributes(from: self.statInfo(for: hostPath, followSymlink: followSymlink))
    }

    private static func attributes(from statInfo: PlatformStat) -> SFTPAttributes {
        .init(
            size: UInt64(statInfo.st_size),
            uid: statInfo.st_uid,
            gid: statInfo.st_gid,
            permissions: UInt32(statInfo.st_mode),
            accessTime: UInt32(statInfo.st_atimespec.tv_sec),
            modificationTime: UInt32(statInfo.st_mtimespec.tv_sec)
        )
    }

    private static func errorForErrno(path: String) -> SFTPError {
        switch errno {
        case ENOENT:
            return .status(.init(code: .noSuchFile, message: "No such file"))
        case EACCES, EPERM:
            return .status(.init(code: .permissionDenied, message: "Permission denied"))
        case EEXIST:
            return .status(.init(code: .failure, message: "Failure"))
        case ENOTEMPTY:
            return .status(.init(code: .failure, message: "Failure"))
        default:
            return .status(.init(code: .failure, message: String(cString: strerror(errno))))
        }
    }

    private func openMode(_ flags: SFTPOpenFlags) -> Int32 {
        var mode: Int32
        switch (flags.contains(.read), flags.contains(.write)) {
        case (true, true):
            mode = O_RDWR
        case (false, true):
            mode = O_WRONLY
        default:
            mode = O_RDONLY
        }
        if flags.contains(.append) {
            mode |= O_APPEND
        }
        if flags.contains(.create) {
            mode |= O_CREAT
        }
        if flags.contains(.truncate) {
            mode |= O_TRUNC
        }
        if flags.contains(.exclusive) {
            mode |= O_EXCL
        }
        return mode
    }

    private static func fileSystemAttributes(forDescriptor descriptor: Int32) throws -> SFTPFileSystemAttributes {
        var stats = PlatformStatFS()
        guard Self.posixFStatFS(descriptor, &stats) == 0 else {
            throw SFTPError.status(.init(code: .failure, message: "Failed to stat filesystem"))
        }
        return self.fileSystemAttributes(from: stats)
    }

    private static func fileSystemAttributes(from stats: PlatformStatFS) -> SFTPFileSystemAttributes {
        #if canImport(Darwin)
        let blockSize = UInt64(stats.f_bsize)
        let fundamentalBlockSize = UInt64(stats.f_iosize)
        let totalFileNodes = UInt64(stats.f_files)
        let freeFileNodes = UInt64(stats.f_ffree)
        let availableFileNodes = UInt64(stats.f_ffree)
        let fileSystemID = UInt64(bitPattern: Int64(stats.f_fsid.val.0)) ^ (UInt64(bitPattern: Int64(stats.f_fsid.val.1)) << 32)
        let maximumNameLength = UInt64(NAME_MAX)
        let flags = SFTPFileSystemFlags(rawValue: (stats.f_flags & UInt32(MNT_RDONLY)) != 0 ? SFTPFileSystemFlags.readOnly.rawValue : 0)
        #else
        let blockSize = UInt64(stats.f_bsize)
        let fundamentalBlockSize = UInt64(stats.f_bsize)
        let totalFileNodes = UInt64(stats.f_files)
        let freeFileNodes = UInt64(stats.f_ffree)
        let availableFileNodes = UInt64(stats.f_favail)
        let fileSystemID = UInt64(bitPattern: Int64(stats.f_fsid.__val.0)) ^ (UInt64(bitPattern: Int64(stats.f_fsid.__val.1)) << 32)
        let maximumNameLength = UInt64(stats.f_namelen)
        let flags = SFTPFileSystemFlags(rawValue: (stats.f_flags & UInt64(ST_RDONLY)) != 0 ? SFTPFileSystemFlags.readOnly.rawValue : 0)
        #endif

        return .init(
            blockSize: blockSize,
            fundamentalBlockSize: fundamentalBlockSize,
            totalBlocks: UInt64(stats.f_blocks),
            freeBlocks: UInt64(stats.f_bfree),
            availableBlocks: UInt64(stats.f_bavail),
            totalFileNodes: totalFileNodes,
            freeFileNodes: freeFileNodes,
            availableFileNodes: availableFileNodes,
            fileSystemID: fileSystemID,
            flags: flags,
            maximumNameLength: maximumNameLength
        )
    }

    #if canImport(Darwin)
    private static func posixOpen(_ path: String, _ flags: Int32, _ mode: mode_t) -> Int32 { Darwin.open(path, flags, mode) }
    private static func posixClose(_ descriptor: Int32) -> Int32 { Darwin.close(descriptor) }
    private static func posixPRead(_ descriptor: Int32, _ buffer: inout [UInt8], _ count: Int, _ offset: off_t) -> Int {
        buffer.withUnsafeMutableBytes { Darwin.pread(descriptor, $0.baseAddress!, count, offset) }
    }
    private static func posixPWrite(_ descriptor: Int32, _ buffer: UnsafeRawPointer, _ count: Int, _ offset: off_t) -> Int {
        Darwin.pwrite(descriptor, buffer, count, offset)
    }
    private static func posixFStat(_ descriptor: Int32, _ statInfo: UnsafeMutablePointer<PlatformStat>) -> Int32 { Darwin.fstat(descriptor, statInfo) }
    private static func posixUnlink(_ path: String) -> Int32 { Darwin.unlink(path) }
    private static func posixMkdir(_ path: String, _ mode: mode_t) -> Int32 { Darwin.mkdir(path, mode) }
    private static func posixRmdir(_ path: String) -> Int32 { Darwin.rmdir(path) }
    private static func posixRename(_ oldPath: String, _ newPath: String) -> Int32 { Darwin.rename(oldPath, newPath) }
    private static func posixSymlink(_ target: String, _ path: String) -> Int32 { Darwin.symlink(target, path) }
    private static func posixFSync(_ descriptor: Int32) -> Int32 { Darwin.fsync(descriptor) }
    private static func posixLink(_ existing: String, _ newPath: String) -> Int32 { Darwin.link(existing, newPath) }
    private static func posixFTruncate(_ descriptor: Int32, _ size: off_t) -> Int32 { Darwin.ftruncate(descriptor, size) }
    private static func posixTruncate(_ path: String, _ size: off_t) -> Int32 { Darwin.truncate(path, size) }
    private static func posixFChmod(_ descriptor: Int32, _ mode: mode_t) -> Int32 { Darwin.fchmod(descriptor, mode) }
    private static func posixChmod(_ path: String, _ mode: mode_t) -> Int32 { Darwin.chmod(path, mode) }
    private static func posixFUTimes(_ descriptor: Int32, _ times: UnsafeMutablePointer<timeval>) -> Int32 { Darwin.futimes(descriptor, times) }
    private static func posixUTimes(_ path: String, _ times: UnsafeMutablePointer<timeval>) -> Int32 { Darwin.utimes(path, times) }
    private static func posixLStat(_ path: String, _ statInfo: UnsafeMutablePointer<PlatformStat>) -> Int32 { Darwin.lstat(path, statInfo) }
    private static func posixReadlink(_ path: String, _ buffer: UnsafeMutablePointer<CChar>, _ size: Int) -> Int { Darwin.readlink(path, buffer, size) }
    private static func posixFStatFS(_ descriptor: Int32, _ stats: UnsafeMutablePointer<PlatformStatFS>) -> Int32 { Darwin.fstatfs(descriptor, stats) }
    #else
    private static func posixOpen(_ path: String, _ flags: Int32, _ mode: mode_t) -> Int32 { Glibc.open(path, flags, mode) }
    private static func posixClose(_ descriptor: Int32) -> Int32 { Glibc.close(descriptor) }
    private static func posixPRead(_ descriptor: Int32, _ buffer: inout [UInt8], _ count: Int, _ offset: off_t) -> Int {
        buffer.withUnsafeMutableBytes { Glibc.pread(descriptor, $0.baseAddress!, count, offset) }
    }
    private static func posixPWrite(_ descriptor: Int32, _ buffer: UnsafeRawPointer, _ count: Int, _ offset: off_t) -> Int {
        Glibc.pwrite(descriptor, buffer, count, offset)
    }
    private static func posixFStat(_ descriptor: Int32, _ statInfo: UnsafeMutablePointer<PlatformStat>) -> Int32 { Glibc.fstat(descriptor, statInfo) }
    private static func posixUnlink(_ path: String) -> Int32 { Glibc.unlink(path) }
    private static func posixMkdir(_ path: String, _ mode: mode_t) -> Int32 { Glibc.mkdir(path, mode) }
    private static func posixRmdir(_ path: String) -> Int32 { Glibc.rmdir(path) }
    private static func posixRename(_ oldPath: String, _ newPath: String) -> Int32 { Glibc.rename(oldPath, newPath) }
    private static func posixSymlink(_ target: String, _ path: String) -> Int32 { Glibc.symlink(target, path) }
    private static func posixFSync(_ descriptor: Int32) -> Int32 { Glibc.fsync(descriptor) }
    private static func posixLink(_ existing: String, _ newPath: String) -> Int32 { Glibc.link(existing, newPath) }
    private static func posixFTruncate(_ descriptor: Int32, _ size: off_t) -> Int32 { Glibc.ftruncate(descriptor, size) }
    private static func posixTruncate(_ path: String, _ size: off_t) -> Int32 { Glibc.truncate(path, size) }
    private static func posixFChmod(_ descriptor: Int32, _ mode: mode_t) -> Int32 { Glibc.fchmod(descriptor, mode) }
    private static func posixChmod(_ path: String, _ mode: mode_t) -> Int32 { Glibc.chmod(path, mode) }
    private static func posixFUTimes(_ descriptor: Int32, _ times: UnsafeMutablePointer<timeval>) -> Int32 { futimes(descriptor, times) }
    private static func posixUTimes(_ path: String, _ times: UnsafeMutablePointer<timeval>) -> Int32 { Glibc.utimes(path, times) }
    private static func posixLStat(_ path: String, _ statInfo: UnsafeMutablePointer<PlatformStat>) -> Int32 { Glibc.lstat(path, statInfo) }
    private static func posixReadlink(_ path: String, _ buffer: UnsafeMutablePointer<CChar>, _ size: Int) -> Int { Glibc.readlink(path, buffer, size) }
    private static func posixFStatFS(_ descriptor: Int32, _ stats: UnsafeMutablePointer<PlatformStatFS>) -> Int32 { Glibc.fstatfs(descriptor, stats) }
    #endif
}
