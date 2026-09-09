// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore

enum SFTPRequestEncoder {
    static func encode(_ message: SFTPRequestMessage, requestID: UInt32, allocator: ByteBufferAllocator) -> ByteBuffer {
        var buffer = allocator.buffer(capacity: 256)
        buffer.writeSFTPFrame(type: type(for: message)) { body in
            body.writeInteger(requestID)
            switch message {
            case .open(let path, let pflags, let attributes):
                body.writeSFTPString(path)
                body.writeInteger(pflags.rawValue)
                body.writeSFTPAttributes(attributes)
            case .close(let handle):
                body.writeSFTPImmutableStringBuffer(handle)
            case .read(let handle, let offset, let length):
                body.writeSFTPImmutableStringBuffer(handle)
                body.writeInteger(offset)
                body.writeInteger(length)
            case .write(let handle, let offset, let data):
                body.writeSFTPImmutableStringBuffer(handle)
                body.writeInteger(offset)
                body.writeSFTPImmutableStringBuffer(data)
            case .lstat(let path), .opendir(let path), .remove(let path), .rmdir(let path), .realpath(let path),
                 .stat(let path), .readlink(let path):
                body.writeSFTPString(path)
            case .fstat(let handle), .readdir(let handle):
                body.writeSFTPImmutableStringBuffer(handle)
            case .setstat(let path, let attributes):
                body.writeSFTPString(path)
                body.writeSFTPAttributes(attributes)
            case .fsetstat(let handle, let attributes):
                body.writeSFTPImmutableStringBuffer(handle)
                body.writeSFTPAttributes(attributes)
            case .mkdir(let path, let attributes):
                body.writeSFTPString(path)
                body.writeSFTPAttributes(attributes)
            case .rename(let oldPath, let newPath):
                body.writeSFTPString(oldPath)
                body.writeSFTPString(newPath)
            case .symlink(let linkPath, let targetPath):
                body.writeSFTPString(targetPath)
                body.writeSFTPString(linkPath)
            case .extended(let name, let data):
                body.writeSFTPString(name)
                var data = data
                body.writeBuffer(&data)
            }
        }
        return buffer
    }

    static func encodeInit(version: SFTPVersion, allocator: ByteBufferAllocator) -> ByteBuffer {
        var buffer = allocator.buffer(capacity: 16)
        buffer.writeSFTPFrame(type: SFTPPacketType.`init`) { body in
            body.writeInteger(version.rawValue)
        }
        return buffer
    }

    private static func type(for message: SFTPRequestMessage) -> UInt8 {
        switch message {
        case .open: return SFTPPacketType.open
        case .close: return SFTPPacketType.close
        case .read: return SFTPPacketType.read
        case .write: return SFTPPacketType.write
        case .lstat: return SFTPPacketType.lstat
        case .fstat: return SFTPPacketType.fstat
        case .setstat: return SFTPPacketType.setstat
        case .fsetstat: return SFTPPacketType.fsetstat
        case .opendir: return SFTPPacketType.opendir
        case .readdir: return SFTPPacketType.readdir
        case .remove: return SFTPPacketType.remove
        case .mkdir: return SFTPPacketType.mkdir
        case .rmdir: return SFTPPacketType.rmdir
        case .realpath: return SFTPPacketType.realpath
        case .stat: return SFTPPacketType.stat
        case .rename: return SFTPPacketType.rename
        case .readlink: return SFTPPacketType.readlink
        case .symlink: return SFTPPacketType.symlink
        case .extended: return SFTPPacketType.extended
        }
    }
}

