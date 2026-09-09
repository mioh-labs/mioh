// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore
import NIOSSH

final class SFTPClientHandler: ChannelDuplexHandler, @unchecked Sendable {
    typealias InboundIn = SSHChannelData
    typealias OutboundIn = Never
    typealias OutboundOut = SSHChannelData

    private(set) var startupFuture: EventLoopFuture<SFTPServerCapabilities>

    private let startupPromise: EventLoopPromise<SFTPServerCapabilities>
    private var stateMachine = SFTPClientStateMachine()
    private var context: ChannelHandlerContext?
    private var inboundBuffer: ByteBuffer

    init(loop: EventLoop, allocator: ByteBufferAllocator) {
        self.startupPromise = loop.makePromise(of: SFTPServerCapabilities.self)
        self.startupFuture = self.startupPromise.futureResult
        self.inboundBuffer = allocator.buffer(capacity: 0)
    }

    func handlerAdded(context: ChannelHandlerContext) {
        self.context = context
        context.channel.setOption(ChannelOptions.allowRemoteHalfClosure, value: true).whenFailure { _ in }
        self.execute(self.stateMachine.beginStartupIfNeeded(channelIsActive: context.channel.isActive), context: context)
    }

    func channelActive(context: ChannelHandlerContext) {
        self.execute(self.stateMachine.beginStartupIfNeeded(channelIsActive: context.channel.isActive), context: context)
        context.fireChannelActive()
    }

    func channelInactive(context: ChannelHandlerContext) {
        self.execute(self.stateMachine.failSession(error: SFTPError.channelClosed), context: context)
        context.fireChannelInactive()
    }

    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        switch event {
        case is ChannelSuccessEvent:
            let action = self.stateMachine.receiveSubsystemSuccess()
            if case .none = action {
                context.fireUserInboundEventTriggered(event)
            } else {
                self.execute(action, context: context)
            }
        case is ChannelFailureEvent:
            let action = self.stateMachine.receiveSubsystemFailure()
            if case .none = action {
                context.fireUserInboundEventTriggered(event)
            } else {
                self.execute(action, context: context)
            }
        default:
            context.fireUserInboundEventTriggered(event)
        }
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let data = self.unwrapInboundIn(data)
        guard case .byteBuffer(let bytes) = data.data else {
            self.execute(
                self.stateMachine.failSession(error: SFTPError.protocolViolation("Unsupported SSH IOData payload")),
                context: context
            )
            return
        }

        switch data.type {
        case .channel:
            var bytes = bytes
            self.inboundBuffer.writeBuffer(&bytes)
            do {
                try self.processInbound(context: context)
            } catch {
                self.execute(self.stateMachine.failSession(error: error), context: context)
            }
        case .stdErr:
            context.fireUserInboundEventTriggered(SFTPClientEvent(standardError: bytes))
        default:
            self.execute(
                self.stateMachine.failSession(error: SFTPError.protocolViolation("Unsupported SSH extended data stream")),
                context: context
            )
        }
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        self.execute(self.stateMachine.failSession(error: error), context: context)
        context.fireErrorCaught(error)
    }

    func send(_ message: SFTPRequestMessage) -> EventLoopFuture<SFTPResponseMessage> {
        guard let context = self.context else {
            return self.startupPromise.futureResult.eventLoop.makeFailedFuture(SFTPError.sessionNotReady)
        }

        let promise = context.eventLoop.makePromise(of: SFTPResponseMessage.self)
        let requestID: UInt32
        do {
            requestID = try self.stateMachine.enqueueRequest(message, promise: promise)
        } catch {
            promise.fail(error)
            return promise.futureResult
        }
        let buffer = SFTPRequestEncoder.encode(message, requestID: requestID, allocator: context.channel.allocator)
        self.writeAndFlush(buffer: buffer, context: context)
        return promise.futureResult
    }

    private func writeAndFlush(buffer: ByteBuffer, context: ChannelHandlerContext) {
        let loopBoundContext = NIOLoopBound(context, eventLoop: context.eventLoop)
        let eventLoop = context.eventLoop
        let wrapped = self.wrapOutboundOut(.init(type: .channel, data: .byteBuffer(buffer)))
        context.writeAndFlush(wrapped).whenFailure { error in
            eventLoop.execute {
                self.execute(self.stateMachine.failSession(error: error), context: loopBoundContext.value)
            }
        }
    }

    private func processInbound(context: ChannelHandlerContext) throws {
        // Retain only an incomplete trailing frame. Without compaction the
        // writer index grows with total transfer volume even after packets
        // have been consumed.
        defer { self.inboundBuffer.discardReadBytes() }
        while let packet = try self.inboundBuffer.readSFTPFrame() {
            self.execute(self.stateMachine.receivePacket(packet), context: context)
        }
    }

    private func execute(_ action: SFTPClientStateMachine.Action, context: ChannelHandlerContext) {
        switch action {
        case .none:
            return
        case .sendSubsystemRequest:
            let loopBoundContext = NIOLoopBound(context, eventLoop: context.eventLoop)
            let eventLoop = context.eventLoop
            context.triggerUserOutboundEvent(
                SSHChannelRequestEvent.SubsystemRequest(subsystem: "sftp", wantReply: true)
            ).whenFailure { error in
                eventLoop.execute {
                    self.execute(self.stateMachine.failSession(error: error), context: loopBoundContext.value)
                }
            }
        case .sendInit:
            self.writeAndFlush(
                buffer: SFTPRequestEncoder.encodeInit(version: .v3, allocator: context.channel.allocator),
                context: context
            )
        case .startupSucceeded(let extensions):
            self.startupPromise.succeed(.init(rawExtensions: extensions))
        case .requestSucceeded(let promise, let response):
            promise.succeed(response)
        case .sessionFailed(let error, let failStartup, let pendingPromises):
            if failStartup {
                self.startupPromise.fail(error)
            }
            pendingPromises.forEach { $0.fail(error) }
            context.close(promise: nil)
        }
    }
}
