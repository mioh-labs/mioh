// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore

enum SFTPRequestDecoder {
    static func decode(type: UInt8, payload: ByteBuffer) throws -> (UInt32, SFTPRequestMessage) {
        var payload = payload
        guard let requestID = payload.readInteger(as: UInt32.self) else {
            throw SFTPError.protocolViolation("Missing SFTP request id")
        }
        switch type {
        case SFTPPacketType.open:
            guard let path = payload.readSFTPString(),
                let flags = payload.readInteger(as: UInt32.self)
            else {
                throw SFTPError.protocolViolation("Invalid OPEN request")
            }
            let attributes = try payload.readSFTPAttributes()
            return (requestID, .open(path: path, pflags: .init(rawValue: flags), attributes: attributes))
        case SFTPPacketType.close:
            guard let handle = payload.readSFTPStringBuffer() else {
                throw SFTPError.protocolViolation("Invalid CLOSE request")
            }
            return (requestID, .close(handle: handle))
        case SFTPPacketType.read:
            guard let handle = payload.readSFTPStringBuffer(),
                let offset = payload.readInteger(as: UInt64.self),
                let length = payload.readInteger(as: UInt32.self)
            else {
                throw SFTPError.protocolViolation("Invalid READ request")
            }
            return (requestID, .read(handle: handle, offset: offset, length: length))
        case SFTPPacketType.write:
            guard let handle = payload.readSFTPStringBuffer(),
                let offset = payload.readInteger(as: UInt64.self),
                let data = payload.readSFTPStringBuffer()
            else {
                throw SFTPError.protocolViolation("Invalid WRITE request")
            }
            return (requestID, .write(handle: handle, offset: offset, data: data))
        case SFTPPacketType.lstat:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid LSTAT request")
            }
            return (requestID, .lstat(path: path))
        case SFTPPacketType.fstat:
            guard let handle = payload.readSFTPStringBuffer() else {
                throw SFTPError.protocolViolation("Invalid FSTAT request")
            }
            return (requestID, .fstat(handle: handle))
        case SFTPPacketType.setstat:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid SETSTAT request")
            }
            return (requestID, .setstat(path: path, attributes: try payload.readSFTPAttributes()))
        case SFTPPacketType.fsetstat:
            guard let handle = payload.readSFTPStringBuffer() else {
                throw SFTPError.protocolViolation("Invalid FSETSTAT request")
            }
            return (requestID, .fsetstat(handle: handle, attributes: try payload.readSFTPAttributes()))
        case SFTPPacketType.opendir:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid OPENDIR request")
            }
            return (requestID, .opendir(path: path))
        case SFTPPacketType.readdir:
            guard let handle = payload.readSFTPStringBuffer() else {
                throw SFTPError.protocolViolation("Invalid READDIR request")
            }
            return (requestID, .readdir(handle: handle))
        case SFTPPacketType.remove:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid REMOVE request")
            }
            return (requestID, .remove(path: path))
        case SFTPPacketType.mkdir:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid MKDIR request")
            }
            return (requestID, .mkdir(path: path, attributes: try payload.readSFTPAttributes()))
        case SFTPPacketType.rmdir:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid RMDIR request")
            }
            return (requestID, .rmdir(path: path))
        case SFTPPacketType.realpath:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid REALPATH request")
            }
            return (requestID, .realpath(path: path))
        case SFTPPacketType.stat:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid STAT request")
            }
            return (requestID, .stat(path: path))
        case SFTPPacketType.rename:
            guard let oldPath = payload.readSFTPString(),
                let newPath = payload.readSFTPString()
            else {
                throw SFTPError.protocolViolation("Invalid RENAME request")
            }
            return (requestID, .rename(oldPath: oldPath, newPath: newPath))
        case SFTPPacketType.readlink:
            guard let path = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid READLINK request")
            }
            return (requestID, .readlink(path: path))
        case SFTPPacketType.symlink:
            guard let targetPath = payload.readSFTPString(),
                let linkPath = payload.readSFTPString()
            else {
                throw SFTPError.protocolViolation("Invalid SYMLINK request")
            }
            return (requestID, .symlink(linkPath: linkPath, targetPath: targetPath))
        case SFTPPacketType.extended:
            guard let name = payload.readSFTPString() else {
                throw SFTPError.protocolViolation("Invalid EXTENDED request")
            }
            return (requestID, .extended(name: name, data: payload))
        default:
            throw SFTPError.protocolViolation("Unsupported SFTP request type \(type)")
        }
    }
}

enum SFTPResponseEncoder {
    static func encodeVersion(_ version: SFTPVersion, extensions: [SFTPExtension], allocator: ByteBufferAllocator) -> ByteBuffer {
        var buffer = allocator.buffer(capacity: 64)
        buffer.writeSFTPFrame(type: SFTPPacketType.version) { body in
            body.writeInteger(version.rawValue)
            for extensionData in extensions {
                body.writeSFTPString(extensionData.name)
                body.writeSFTPImmutableStringBuffer(extensionData.data)
            }
        }
        return buffer
    }

    static func encode(_ response: SFTPResponseMessage, requestID: UInt32, allocator: ByteBufferAllocator) -> ByteBuffer {
        var buffer = allocator.buffer(capacity: 256)
        buffer.writeSFTPFrame(type: type(for: response)) { body in
            body.writeInteger(requestID)
            switch response {
            case .status(let status):
                body.writeInteger(status.code.rawValue)
                body.writeSFTPString(status.message)
                body.writeSFTPString(status.languageTag)
            case .handle(let handle):
                body.writeSFTPImmutableStringBuffer(handle)
            case .data(let data):
                body.writeSFTPImmutableStringBuffer(data)
            case .name(let entries):
                body.writeInteger(UInt32(entries.count))
                for entry in entries {
                    body.writeSFTPString(entry.filename)
                    body.writeSFTPString(entry.longname)
                    body.writeSFTPAttributes(entry.attributes)
                }
            case .attributes(let attributes):
                body.writeSFTPAttributes(attributes)
            case .extendedReply(let data):
                var data = data
                body.writeBuffer(&data)
            }
        }
        return buffer
    }

    private static func type(for response: SFTPResponseMessage) -> UInt8 {
        switch response {
        case .status: return SFTPPacketType.status
        case .handle: return SFTPPacketType.handle
        case .data: return SFTPPacketType.data
        case .name: return SFTPPacketType.name
        case .attributes: return SFTPPacketType.attrs
        case .extendedReply: return SFTPPacketType.extendedReply
        }
    }
}

