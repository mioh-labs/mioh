// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore
import NIOSSH

public final class SFTPServerHandler: ChannelDuplexHandler, @unchecked Sendable {
    public typealias InboundIn = SSHChannelData
    public typealias OutboundIn = Never
    public typealias OutboundOut = SSHChannelData

    private let backend: any SFTPServerBackend
    private var stateMachine = SFTPServerStateMachine()
    private var inboundBuffer: ByteBuffer

    public init(loop: EventLoop, allocator: ByteBufferAllocator, backend: any SFTPServerBackend) {
        self.backend = backend
        self.inboundBuffer = allocator.buffer(capacity: 0)
    }

    public func handlerAdded(context: ChannelHandlerContext) {
        context.channel.setOption(ChannelOptions.allowRemoteHalfClosure, value: true).whenFailure { _ in }
    }

    public func channelInactive(context: ChannelHandlerContext) {
        _ = self.stateMachine.failSession(error: SFTPError.channelClosed)
        context.fireChannelInactive()
    }

    public func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        switch event {
        case let subsystem as SSHChannelRequestEvent.SubsystemRequest:
            self.execute(
                self.stateMachine.receiveSubsystemRequest(
                    subsystem: subsystem.subsystem,
                    wantReply: subsystem.wantReply
                ),
                context: context
            )
        default:
            context.fireUserInboundEventTriggered(event)
        }
    }

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let data = self.unwrapInboundIn(data)
        guard case .byteBuffer(let bytes) = data.data else {
            self.fail(context: context, error: SFTPError.protocolViolation("Unsupported SSH IOData payload"))
            return
        }

        switch data.type {
        case .channel:
            self.execute(self.stateMachine.validateDataPacketBeforeParse(), context: context)
            guard self.stateMachine.phase != .closed else {
                return
            }
            var bytes = bytes
            self.inboundBuffer.writeBuffer(&bytes)
            do {
                try self.processInbound(context: context)
            } catch {
                self.fail(context: context, error: error)
            }
        case .stdErr:
            self.fail(context: context, error: SFTPError.protocolViolation("Unsupported SSH extended data stream"))
        default:
            self.fail(context: context, error: SFTPError.protocolViolation("Unsupported SSH extended data stream"))
        }
    }

    public func errorCaught(context: ChannelHandlerContext, error: Error) {
        self.fail(context: context, error: error)
        context.fireErrorCaught(error)
    }

    private func processInbound(context: ChannelHandlerContext) throws {
        // Keep a partial packet, but release bytes belonging to frames that
        // were already dispatched so long sessions remain memory-bounded.
        defer { self.inboundBuffer.discardReadBytes() }
        while let packet = try self.inboundBuffer.readSFTPServerFrame() {
            let action = self.stateMachine.receivePacket(packet, extensions: self.backend.advertisedExtensions)
            self.execute(action, context: context)

            guard self.stateMachine.phase != .closed else {
                return
            }

            switch packet {
            case .`init`:
                continue
            case .request(let requestID, let message):
                self.handleRequest(message, requestID: requestID, context: context)
            }
        }
    }

    private func handleRequest(_ message: SFTPRequestMessage, requestID: UInt32, context: ChannelHandlerContext) {
        let serverContext = SFTPServerContext(channel: context.channel)
        let future = self.dispatch(message, context: serverContext)
        future.whenComplete { result in
            switch result {
            case .success(let response):
                self.writeAndFlush(
                    SFTPResponseEncoder.encode(response, requestID: requestID, allocator: context.channel.allocator),
                    context: context
                )
            case .failure(let error):
                self.writeAndFlush(
                    SFTPResponseEncoder.encode(.status(Self.status(from: error)), requestID: requestID, allocator: context.channel.allocator),
                    context: context
                )
            }
        }
    }

    private func dispatch(_ message: SFTPRequestMessage, context: SFTPServerContext) -> EventLoopFuture<SFTPResponseMessage> {
        switch message {
        case .open(let path, let flags, let attributes):
            return self.backend.open(path: path, flags: flags, attributes: attributes, context: context)
                .map { .handle($0.bytes) }
        case .close(let handle):
            return self.closeHandle(handle: handle, context: context)
        case .read(let handle, let offset, let length):
            return self.backend.read(fileHandle: .init(bytes: handle), offset: offset, length: length, context: context)
                .map { payload in
                    payload.map(SFTPResponseMessage.data) ?? .status(.init(code: .eof))
                }
        case .write(let handle, let offset, let data):
            return self.backend.write(fileHandle: .init(bytes: handle), offset: offset, data: data, context: context)
                .map { .status(.init(code: .ok)) }
        case .lstat(let path):
            return self.backend.lstat(path: path, context: context).map(SFTPResponseMessage.attributes)
        case .fstat(let handle):
            return self.backend.fstat(fileHandle: .init(bytes: handle), context: context).map(SFTPResponseMessage.attributes)
        case .setstat(let path, let attributes):
            return self.backend.setstat(path: path, attributes: attributes, context: context).map { .status(.init(code: .ok)) }
        case .fsetstat(let handle, let attributes):
            return self.backend.fsetstat(fileHandle: .init(bytes: handle), attributes: attributes, context: context)
                .map { .status(.init(code: .ok)) }
        case .opendir(let path):
            return self.backend.opendir(path: path, context: context).map { .handle($0.bytes) }
        case .readdir(let handle):
            return self.backend.readdir(directoryHandle: .init(bytes: handle), context: context)
                .map { entries in
                    entries.map(SFTPResponseMessage.name) ?? .status(.init(code: .eof))
                }
        case .remove(let path):
            return self.backend.remove(path: path, context: context).map { .status(.init(code: .ok)) }
        case .mkdir(let path, let attributes):
            return self.backend.mkdir(path: path, attributes: attributes, context: context).map { .status(.init(code: .ok)) }
        case .rmdir(let path):
            return self.backend.rmdir(path: path, context: context).map { .status(.init(code: .ok)) }
        case .realpath(let path):
            return self.backend.realpath(path: path, context: context)
                .map { .name([.init(filename: $0, longname: $0, attributes: .init())]) }
        case .stat(let path):
            return self.backend.stat(path: path, context: context).map(SFTPResponseMessage.attributes)
        case .rename(let oldPath, let newPath):
            return self.backend.rename(oldPath: oldPath, newPath: newPath, context: context).map { .status(.init(code: .ok)) }
        case .readlink(let path):
            return self.backend.readlink(path: path, context: context)
                .map { .name([.init(filename: $0, longname: $0, attributes: .init())]) }
        case .symlink(let linkPath, let targetPath):
            return self.backend.symlink(linkPath: linkPath, targetPath: targetPath, context: context).map { .status(.init(code: .ok)) }
        case .extended(let name, var data):
            return self.dispatchExtended(name: name, payload: &data, context: context)
        }
    }

    private func dispatchExtended(
        name: String,
        payload: inout ByteBuffer,
        context: SFTPServerContext
    ) -> EventLoopFuture<SFTPResponseMessage> {
        switch name {
        case SFTPExtensionName.posixRename.rawValue:
            guard let oldPath = payload.readSFTPString(), let newPath = payload.readSFTPString() else {
                return context.eventLoop.makeFailedFuture(SFTPError.protocolViolation("Invalid posix-rename payload"))
            }
            return self.backend.posixRename(oldPath: oldPath, newPath: newPath, context: context).map { .status(.init(code: .ok)) }
        case SFTPExtensionName.fsync.rawValue:
            guard let handle = payload.readSFTPStringBuffer() else {
                return context.eventLoop.makeFailedFuture(SFTPError.protocolViolation("Invalid fsync payload"))
            }
            return self.backend.fsync(fileHandle: .init(bytes: handle), context: context).map { .status(.init(code: .ok)) }
        case SFTPExtensionName.statvfs.rawValue:
            guard let path = payload.readSFTPString() else {
                return context.eventLoop.makeFailedFuture(SFTPError.protocolViolation("Invalid statvfs payload"))
            }
            return self.backend.statvfs(path: path, context: context).map { attributes in
                var payload = context.allocator.buffer(capacity: 128)
                payload.writeSFTPFileSystemAttributes(attributes)
                return .extendedReply(payload)
            }
        case SFTPExtensionName.fstatvfs.rawValue:
            guard let handle = payload.readSFTPStringBuffer() else {
                return context.eventLoop.makeFailedFuture(SFTPError.protocolViolation("Invalid fstatvfs payload"))
            }
            return self.backend.fstatvfs(fileHandle: .init(bytes: handle), context: context).map { attributes in
                var payload = context.allocator.buffer(capacity: 128)
                payload.writeSFTPFileSystemAttributes(attributes)
                return .extendedReply(payload)
            }
        case SFTPExtensionName.hardlink.rawValue:
            guard let oldPath = payload.readSFTPString(), let newPath = payload.readSFTPString() else {
                return context.eventLoop.makeFailedFuture(SFTPError.protocolViolation("Invalid hardlink payload"))
            }
            return self.backend.hardlink(oldPath: oldPath, newPath: newPath, context: context).map { .status(.init(code: .ok)) }
        case SFTPExtensionName.copyData.rawValue:
            guard let readHandle = payload.readSFTPStringBuffer(),
                let readOffset = payload.readInteger(as: UInt64.self),
                let length = payload.readInteger(as: UInt64.self),
                let writeHandle = payload.readSFTPStringBuffer(),
                let writeOffset = payload.readInteger(as: UInt64.self)
            else {
                return context.eventLoop.makeFailedFuture(SFTPError.protocolViolation("Invalid copy-data payload"))
            }
            return self.backend.copyData(
                from: .init(bytes: readHandle),
                readOffset: readOffset,
                length: length,
                to: .init(bytes: writeHandle),
                writeOffset: writeOffset,
                context: context
            ).map { .status(.init(code: .ok)) }
        default:
            return context.eventLoop.makeFailedFuture(SFTPError.status(.init(code: .operationUnsupported)))
        }
    }

    private func closeHandle(handle: ByteBuffer, context: SFTPServerContext) -> EventLoopFuture<SFTPResponseMessage> {
        let file = self.backend.close(fileHandle: .init(bytes: handle), context: context)
            .map { SFTPResponseMessage.status(.init(code: .ok)) }
        return file.flatMapError { _ in
            self.backend.close(directoryHandle: .init(bytes: handle), context: context)
                .map { .status(.init(code: .ok)) }
        }
    }

    private func writeAndFlush(_ buffer: ByteBuffer, context: ChannelHandlerContext) {
        let wrapped = self.wrapOutboundOut(.init(type: .channel, data: .byteBuffer(buffer)))
        context.writeAndFlush(wrapped, promise: nil)
    }

    private func execute(_ action: SFTPServerStateMachine.Action, context: ChannelHandlerContext) {
        switch action {
        case .none:
            return
        case .sendSubsystemSuccess:
            context.triggerUserOutboundEvent(ChannelSuccessEvent(), promise: nil)
        case .sendSubsystemFailure:
            context.triggerUserOutboundEvent(ChannelFailureEvent(), promise: nil)
        case .sendVersion(let version, let extensions):
            self.writeAndFlush(SFTPResponseEncoder.encodeVersion(version, extensions: extensions, allocator: context.channel.allocator), context: context)
        case .failSession(let error):
            self.fail(context: context, error: error)
        }
    }

    private func fail(context: ChannelHandlerContext, error: Error) {
        guard case .none = self.stateMachine.failSession(error: error) else {
            context.close(promise: nil)
            return
        }
    }

    private static func status(from error: Error) -> SFTPStatus {
        if let sftpError = error as? SFTPError {
            switch sftpError {
            case .status(let status):
                return status
            case .unsupportedVersion, .protocolViolation:
                return .init(code: .badMessage, message: String(describing: sftpError))
            case .invalidPath(let path):
                return .init(code: .permissionDenied, message: "Invalid path \(path)")
            default:
                return .init(code: .failure, message: String(describing: sftpError))
            }
        }
        return .init(code: .failure, message: String(describing: error))
    }
}
