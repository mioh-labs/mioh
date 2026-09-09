// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore
import NIOSSH

public final class SFTPClient: @unchecked Sendable {
    public let channel: Channel
    public let serverCapabilities: SFTPServerCapabilities

    private let handler: SFTPClientHandler

    private init(channel: Channel, handler: SFTPClientHandler, serverCapabilities: SFTPServerCapabilities) {
        self.channel = channel
        self.handler = handler
        self.serverCapabilities = serverCapabilities
    }

    public static func start(on channel: Channel) -> EventLoopFuture<SFTPClient> {
        let handler = SFTPClientHandler(loop: channel.eventLoop, allocator: channel.allocator)
        return channel.pipeline.addHandler(handler).flatMap {
            handler.startupFuture.map { capabilities in
                SFTPClient(channel: channel, handler: handler, serverCapabilities: capabilities)
            }
        }
    }

    public static func openChannel(with sshHandler: NIOSSHHandler, on channel: Channel) -> EventLoopFuture<SFTPClient> {
        let sftpPromise = channel.eventLoop.makePromise(of: SFTPClient.self)
        let childChannelPromise = channel.eventLoop.makePromise(of: Channel.self)

        // Authentication failure or a parent-channel close can prevent the
        // child initializer from ever running. Cascade that failure so callers
        // do not wait forever on the independent SFTP promise.
        childChannelPromise.futureResult.whenFailure { error in
            sftpPromise.fail(error)
        }

        channel.eventLoop.execute {
            sshHandler.createChannel(childChannelPromise, channelType: .session) { childChannel, channelType in
                guard channelType == .session else {
                    return childChannel.eventLoop.makeFailedFuture(SFTPError.invalidChannelType)
                }

                let handler = SFTPClientHandler(loop: childChannel.eventLoop, allocator: childChannel.allocator)
                sftpPromise.completeWith(
                    handler.startupFuture.map { capabilities in
                        SFTPClient(channel: childChannel, handler: handler, serverCapabilities: capabilities)
                    }
                )

                return childChannel.pipeline.addHandler(handler).flatMapError { error in
                    sftpPromise.fail(error)
                    return childChannel.eventLoop.makeFailedFuture(error)
                }
            }
        }

        return sftpPromise.futureResult
    }

    public func send(_ message: SFTPRequestMessage) -> EventLoopFuture<SFTPResponseMessage> {
        if self.channel.eventLoop.inEventLoop {
            return self.handler.send(message)
        }

        let promise = self.channel.eventLoop.makePromise(of: SFTPResponseMessage.self)
        self.channel.eventLoop.execute {
            promise.completeWith(self.handler.send(message))
        }
        return promise.futureResult
    }

    public func supportsExtension(_ extensionName: SFTPExtensionName) -> Bool {
        self.serverCapabilities.supports(extensionName)
    }

    public func openFile(
        path: String,
        flags: SFTPOpenFlags,
        attributes: SFTPAttributes = .init()
    ) -> EventLoopFuture<SFTPFileHandle> {
        self.send(.open(path: path, pflags: flags, attributes: attributes)).flatMapThrowing { response in
            switch response {
            case .handle(let handle):
                return .init(bytes: handle)
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("OPEN expected HANDLE or STATUS")
            }
        }
    }

    public func closeFile(_ handle: SFTPFileHandle) -> EventLoopFuture<Void> {
        self.closeHandle(handle.bytes)
    }

    public func closeDirectory(_ handle: SFTPDirectoryHandle) -> EventLoopFuture<Void> {
        self.closeHandle(handle.bytes)
    }

    public func read(file: SFTPFileHandle, offset: UInt64, length: UInt32) -> EventLoopFuture<ByteBuffer?> {
        self.send(.read(handle: file.bytes, offset: offset, length: length)).flatMapThrowing { response in
            switch response {
            case .data(let buffer):
                return buffer
            case .status(let status) where status.code == .eof:
                return nil
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("READ expected DATA or STATUS")
            }
        }
    }

    public func write(file: SFTPFileHandle, offset: UInt64, data: ByteBuffer) -> EventLoopFuture<Void> {
        self.writeHandle(file.bytes, offset: offset, data: data)
    }

    public func openDirectory(path: String) -> EventLoopFuture<SFTPDirectoryHandle> {
        self.send(.opendir(path: path)).flatMapThrowing { response in
            switch response {
            case .handle(let handle):
                return .init(bytes: handle)
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("OPENDIR expected HANDLE or STATUS")
            }
        }
    }

    public func readDirectoryBatch(_ handle: SFTPDirectoryHandle) -> EventLoopFuture<[SFTPNameEntry]?> {
        self.send(.readdir(handle: handle.bytes)).flatMapThrowing { response in
            switch response {
            case .name(let entries):
                return entries
            case .status(let status) where status.code == .eof:
                return nil
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("READDIR expected NAME or STATUS")
            }
        }
    }

    public func stat(path: String) -> EventLoopFuture<SFTPAttributes> {
        self.attributes(for: .stat(path: path), operation: "STAT")
    }

    public func lstat(path: String) -> EventLoopFuture<SFTPAttributes> {
        self.attributes(for: .lstat(path: path), operation: "LSTAT")
    }

    public func fstat(file: SFTPFileHandle) -> EventLoopFuture<SFTPAttributes> {
        self.attributes(for: .fstat(handle: file.bytes), operation: "FSTAT")
    }

    public func setstat(path: String, attributes: SFTPAttributes) -> EventLoopFuture<Void> {
        self.statusOnly(.setstat(path: path, attributes: attributes), operation: "SETSTAT")
    }

    public func fsetstat(file: SFTPFileHandle, attributes: SFTPAttributes) -> EventLoopFuture<Void> {
        self.statusOnly(.fsetstat(handle: file.bytes, attributes: attributes), operation: "FSETSTAT")
    }

    public func remove(path: String) -> EventLoopFuture<Void> {
        self.statusOnly(.remove(path: path), operation: "REMOVE")
    }

    public func mkdir(path: String, attributes: SFTPAttributes = .init()) -> EventLoopFuture<Void> {
        self.statusOnly(.mkdir(path: path, attributes: attributes), operation: "MKDIR")
    }

    public func rmdir(path: String) -> EventLoopFuture<Void> {
        self.statusOnly(.rmdir(path: path), operation: "RMDIR")
    }

    public func realpath(_ path: String) -> EventLoopFuture<String> {
        self.send(.realpath(path: path)).flatMapThrowing { response in
            switch response {
            case .name(let entries):
                guard let first = entries.first else {
                    throw SFTPError.unexpectedResponse("REALPATH returned no entries")
                }
                return first.filename
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("REALPATH expected NAME or STATUS")
            }
        }
    }

    public func rename(from oldPath: String, to newPath: String) -> EventLoopFuture<Void> {
        self.statusOnly(.rename(oldPath: oldPath, newPath: newPath), operation: "RENAME")
    }

    public func readlink(path: String) -> EventLoopFuture<String> {
        self.send(.readlink(path: path)).flatMapThrowing { response in
            switch response {
            case .name(let entries):
                guard let first = entries.first else {
                    throw SFTPError.unexpectedResponse("READLINK returned no entries")
                }
                return first.filename
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("READLINK expected NAME or STATUS")
            }
        }
    }

    public func symlink(linkPath: String, targetPath: String) -> EventLoopFuture<Void> {
        self.statusOnly(.symlink(linkPath: linkPath, targetPath: targetPath), operation: "SYMLINK")
    }

    public func posixRename(from oldPath: String, to newPath: String) -> EventLoopFuture<Void> {
        self.extendedStatusOnly(.posixRename, operation: "POSIX_RENAME") { body in
            body.writeSFTPString(oldPath)
            body.writeSFTPString(newPath)
        }
    }

    public func fsync(file: SFTPFileHandle) -> EventLoopFuture<Void> {
        self.extendedStatusOnly(.fsync, operation: "FSYNC") { body in
            body.writeSFTPImmutableStringBuffer(file.bytes)
        }
    }

    public func statvfs(path: String) -> EventLoopFuture<SFTPFileSystemAttributes> {
        self.extendedReply(.statvfs, operation: "STATVFS") { body in
            body.writeSFTPString(path)
        } decode: { payload in
            try payload.readSFTPFileSystemAttributes()
        }
    }

    public func fstatvfs(file: SFTPFileHandle) -> EventLoopFuture<SFTPFileSystemAttributes> {
        self.extendedReply(.fstatvfs, operation: "FSTATVFS") { body in
            body.writeSFTPImmutableStringBuffer(file.bytes)
        } decode: { payload in
            try payload.readSFTPFileSystemAttributes()
        }
    }

    public func hardlink(from oldPath: String, to newPath: String) -> EventLoopFuture<Void> {
        self.extendedStatusOnly(.hardlink, operation: "HARDLINK") { body in
            body.writeSFTPString(oldPath)
            body.writeSFTPString(newPath)
        }
    }

    public func copyData(
        from source: SFTPFileHandle,
        readOffset: UInt64,
        length: UInt64,
        to destination: SFTPFileHandle,
        writeOffset: UInt64
    ) -> EventLoopFuture<Void> {
        self.extendedStatusOnly(.copyData, operation: "COPY_DATA") { body in
            body.writeSFTPImmutableStringBuffer(source.bytes)
            body.writeInteger(readOffset)
            body.writeInteger(length)
            body.writeSFTPImmutableStringBuffer(destination.bytes)
            body.writeInteger(writeOffset)
        }
    }

    private func closeHandle(_ handle: ByteBuffer) -> EventLoopFuture<Void> {
        self.statusOnly(.close(handle: handle), operation: "CLOSE")
    }

    private func writeHandle(_ handle: ByteBuffer, offset: UInt64, data: ByteBuffer) -> EventLoopFuture<Void> {
        self.statusOnly(.write(handle: handle, offset: offset, data: data), operation: "WRITE")
    }

    private func attributes(for request: SFTPRequestMessage, operation: String) -> EventLoopFuture<SFTPAttributes> {
        self.send(request).flatMapThrowing { response in
            switch response {
            case .attributes(let attributes):
                return attributes
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("\(operation) expected ATTRS or STATUS")
            }
        }
    }

    private func statusOnly(_ request: SFTPRequestMessage, operation: String) -> EventLoopFuture<Void> {
        self.send(request).flatMapThrowing { response in
            switch response {
            case .status(let status) where status.code == .ok:
                return ()
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("\(operation) expected STATUS")
            }
        }
    }

    private func extendedStatusOnly(
        _ extensionName: SFTPExtensionName,
        operation: String,
        payloadWriter: (inout ByteBuffer) -> Void
    ) -> EventLoopFuture<Void> {
        self.extendedRequest(extensionName, operation: operation, payloadWriter: payloadWriter).flatMapThrowing { response in
            switch response {
            case .status(let status) where status.code == .ok:
                return ()
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("\(operation) expected STATUS")
            }
        }
    }

    private func extendedReply<T>(
        _ extensionName: SFTPExtensionName,
        operation: String,
        payloadWriter: (inout ByteBuffer) -> Void,
        decode: @escaping @Sendable (inout ByteBuffer) throws -> T
    ) -> EventLoopFuture<T> {
        self.extendedRequest(extensionName, operation: operation, payloadWriter: payloadWriter).flatMapThrowing { response in
            switch response {
            case .extendedReply(var payload):
                return try decode(&payload)
            case .status(let status):
                throw SFTPError.status(status)
            default:
                throw SFTPError.unexpectedResponse("\(operation) expected EXTENDED_REPLY or STATUS")
            }
        }
    }

    private func extendedRequest(
        _ extensionName: SFTPExtensionName,
        operation: String,
        payloadWriter: (inout ByteBuffer) -> Void
    ) -> EventLoopFuture<SFTPResponseMessage> {
        guard self.supportsExtension(extensionName) else {
            return self.channel.eventLoop.makeFailedFuture(SFTPError.unsupportedExtension(extensionName))
        }

        var payload = self.channel.allocator.buffer(capacity: 128)
        payloadWriter(&payload)
        return self.send(.extended(name: extensionName.rawValue, data: payload))
            .flatMapThrowing { response in
                if case .status(let status) = response, status.code == .operationUnsupported {
                    throw SFTPError.unsupportedExtension(extensionName)
                }
                return response
            }
    }
}
