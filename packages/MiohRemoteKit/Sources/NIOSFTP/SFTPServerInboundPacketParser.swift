// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore

enum SFTPServerInboundPacket: Equatable {
    case `init`(SFTPVersion)
    case request(id: UInt32, SFTPRequestMessage)
}

enum SFTPServerInboundPacketParser {
    static func parse(type: UInt8, payload: ByteBuffer) throws -> SFTPServerInboundPacket {
        var payload = payload

        switch type {
        case SFTPPacketType.`init`:
            guard let version = payload.readInteger(as: UInt32.self) else {
                throw SFTPError.protocolViolation("Missing SFTP init version field")
            }
            return .`init`(SFTPVersion(version))
        case SFTPPacketType.open,
            SFTPPacketType.close,
            SFTPPacketType.read,
            SFTPPacketType.write,
            SFTPPacketType.lstat,
            SFTPPacketType.fstat,
            SFTPPacketType.setstat,
            SFTPPacketType.fsetstat,
            SFTPPacketType.opendir,
            SFTPPacketType.readdir,
            SFTPPacketType.remove,
            SFTPPacketType.mkdir,
            SFTPPacketType.rmdir,
            SFTPPacketType.realpath,
            SFTPPacketType.stat,
            SFTPPacketType.rename,
            SFTPPacketType.readlink,
            SFTPPacketType.symlink,
            SFTPPacketType.extended:
            let (requestID, message) = try SFTPRequestDecoder.decode(type: type, payload: payload)
            return .request(id: requestID, message)
        default:
            throw SFTPError.protocolViolation("Unsupported SFTP server packet type \(type)")
        }
    }
}

