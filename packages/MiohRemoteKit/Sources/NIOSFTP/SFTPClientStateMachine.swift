// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore

struct SFTPClientStateMachine {
    enum Phase {
        case idle
        case waitingForSubsystemReply
        case waitingForVersion
        case ready
        case closed
    }

    enum ResponseKind: Hashable {
        case status
        case handle
        case data
        case name
        case attributes
        case extendedReply
    }

    struct PendingRequest {
        var message: SFTPRequestMessage
        var allowedResponses: Set<ResponseKind>
        var promise: EventLoopPromise<SFTPResponseMessage>
    }

    enum Action {
        case none
        case sendSubsystemRequest
        case sendInit
        case startupSucceeded([SFTPExtension])
        case requestSucceeded(EventLoopPromise<SFTPResponseMessage>, SFTPResponseMessage)
        case sessionFailed(Error, failStartup: Bool, pendingPromises: [EventLoopPromise<SFTPResponseMessage>])
    }

    private(set) var phase: Phase = .idle
    private var nextRequestID: UInt32 = 0
    private var pendingRequests: [UInt32: PendingRequest] = [:]
    private var startupResolved = false
    private var isFailing = false

    mutating func beginStartupIfNeeded(channelIsActive: Bool) -> Action {
        guard self.phase == .idle, channelIsActive else {
            return .none
        }
        self.phase = .waitingForSubsystemReply
        return .sendSubsystemRequest
    }

    mutating func receiveSubsystemSuccess() -> Action {
        guard self.phase == .waitingForSubsystemReply else {
            return .none
        }
        self.phase = .waitingForVersion
        return .sendInit
    }

    mutating func receiveSubsystemFailure() -> Action {
        guard self.phase == .waitingForSubsystemReply else {
            return .none
        }
        return self.failSession(error: SFTPError.subsystemRejected)
    }

    mutating func enqueueRequest(
        _ message: SFTPRequestMessage,
        promise: EventLoopPromise<SFTPResponseMessage>
    ) throws -> UInt32 {
        guard self.phase == .ready else {
            throw SFTPError.sessionNotReady
        }

        let requestID = self.allocateRequestID()
        self.pendingRequests[requestID] = .init(
            message: message,
            allowedResponses: Self.allowedResponses(for: message),
            promise: promise
        )
        return requestID
    }

    mutating func receivePacket(_ packet: SFTPInboundPacket) -> Action {
        switch packet {
        case .version(let version, let extensions):
            guard self.phase == .waitingForVersion else {
                return self.failSession(error: SFTPError.unexpectedResponse("Received VERSION outside startup"))
            }
            guard version == .v3 else {
                return self.failSession(error: SFTPError.unsupportedVersion(version.rawValue))
            }
            self.phase = .ready
            self.startupResolved = true
            return .startupSucceeded(extensions)
        case .response(let requestID, let response):
            guard self.phase == .ready else {
                return self.failSession(error: SFTPError.unexpectedResponse("Received response before startup completed"))
            }
            guard let pending = self.pendingRequests.removeValue(forKey: requestID) else {
                return self.failSession(error: SFTPError.unexpectedResponse("Received response for unknown request id \(requestID)"))
            }
            guard pending.allowedResponses.contains(Self.responseKind(for: response)) else {
                return self.failSession(
                    error: SFTPError.unexpectedResponse("Received invalid response for \(Self.operationName(for: pending.message))"),
                    additionalPromises: [pending.promise]
                )
            }
            return .requestSucceeded(pending.promise, response)
        }
    }

    mutating func failSession(error: Error) -> Action {
        self.failSession(error: error, additionalPromises: [])
    }

    private mutating func failSession(error: Error, additionalPromises: [EventLoopPromise<SFTPResponseMessage>]) -> Action {
        guard !self.isFailing else {
            return .none
        }
        self.isFailing = true
        self.phase = .closed
        let failStartup = !self.startupResolved
        self.startupResolved = true
        let pendingPromises = additionalPromises + self.pendingRequests.values.map(\.promise)
        self.pendingRequests.removeAll()
        return .sessionFailed(error, failStartup: failStartup, pendingPromises: pendingPromises)
    }

    private mutating func allocateRequestID() -> UInt32 {
        while self.pendingRequests[self.nextRequestID] != nil {
            self.nextRequestID &+= 1
        }
        defer {
            self.nextRequestID &+= 1
        }
        return self.nextRequestID
    }

    private static func responseKind(for response: SFTPResponseMessage) -> ResponseKind {
        switch response {
        case .status:
            return .status
        case .handle:
            return .handle
        case .data:
            return .data
        case .name:
            return .name
        case .attributes:
            return .attributes
        case .extendedReply:
            return .extendedReply
        }
    }

    private static func allowedResponses(for message: SFTPRequestMessage) -> Set<ResponseKind> {
        switch message {
        case .open, .opendir:
            return [.status, .handle]
        case .read:
            return [.status, .data]
        case .lstat, .fstat, .stat:
            return [.status, .attributes]
        case .readdir, .realpath, .readlink:
            return [.status, .name]
        case .extended:
            return [.status, .extendedReply]
        case .close, .write, .setstat, .fsetstat, .remove, .mkdir, .rmdir, .rename, .symlink:
            return [.status]
        }
    }

    private static func operationName(for message: SFTPRequestMessage) -> String {
        switch message {
        case .open: return "OPEN"
        case .close: return "CLOSE"
        case .read: return "READ"
        case .write: return "WRITE"
        case .lstat: return "LSTAT"
        case .fstat: return "FSTAT"
        case .setstat: return "SETSTAT"
        case .fsetstat: return "FSETSTAT"
        case .opendir: return "OPENDIR"
        case .readdir: return "READDIR"
        case .remove: return "REMOVE"
        case .mkdir: return "MKDIR"
        case .rmdir: return "RMDIR"
        case .realpath: return "REALPATH"
        case .stat: return "STAT"
        case .rename: return "RENAME"
        case .readlink: return "READLINK"
        case .symlink: return "SYMLINK"
        case .extended: return "EXTENDED"
        }
    }
}

