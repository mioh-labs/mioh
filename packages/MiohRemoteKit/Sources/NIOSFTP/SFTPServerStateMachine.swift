// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore

struct SFTPServerStateMachine {
    enum Phase {
        case idle
        case waitingForInit
        case ready
        case closed
    }

    enum Action {
        case none
        case sendSubsystemSuccess
        case sendSubsystemFailure
        case sendVersion(SFTPVersion, [SFTPExtension])
        case failSession(Error)
    }

    private(set) var phase: Phase = .idle
    private var subsystemAccepted = false
    private var isFailing = false

    mutating func receiveSubsystemRequest(subsystem: String, wantReply: Bool) -> Action {
        guard self.phase == .idle else {
            return .failSession(SFTPError.protocolViolation("Received subsystem request after startup"))
        }

        guard subsystem == "sftp" else {
            self.phase = .closed
            return wantReply ? .sendSubsystemFailure : .none
        }

        self.phase = .waitingForInit
        self.subsystemAccepted = true
        return wantReply ? .sendSubsystemSuccess : .none
    }

    mutating func receivePacket(_ packet: SFTPServerInboundPacket, extensions: [SFTPExtension]) -> Action {
        switch packet {
        case .`init`(let version):
            guard self.phase == .waitingForInit else {
                return self.failSession(error: SFTPError.unexpectedResponse("Received INIT outside startup"))
            }
            guard version.rawValue >= SFTPVersion.v3.rawValue else {
                return self.failSession(error: SFTPError.unsupportedVersion(version.rawValue))
            }
            self.phase = .ready
            return .sendVersion(.v3, extensions)
        case .request:
            guard self.phase == .ready else {
                return self.failSession(error: SFTPError.unexpectedResponse("Received request before startup completed"))
            }
            return .none
        }
    }

    mutating func validateDataPacketBeforeParse() -> Action {
        guard self.subsystemAccepted else {
            return self.failSession(error: SFTPError.protocolViolation("Received SFTP data before subsystem accepted"))
        }
        return .none
    }

    mutating func failSession(error: Error) -> Action {
        guard !self.isFailing else {
            return .none
        }
        self.isFailing = true
        self.phase = .closed
        return .failSession(error)
    }
}

