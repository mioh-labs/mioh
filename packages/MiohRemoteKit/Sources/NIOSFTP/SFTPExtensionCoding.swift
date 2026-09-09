// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore

extension ByteBuffer {
    mutating func readSFTPFileSystemAttributes() throws -> SFTPFileSystemAttributes {
        guard
            let blockSize = self.readInteger(as: UInt64.self),
            let fundamentalBlockSize = self.readInteger(as: UInt64.self),
            let totalBlocks = self.readInteger(as: UInt64.self),
            let freeBlocks = self.readInteger(as: UInt64.self),
            let availableBlocks = self.readInteger(as: UInt64.self),
            let totalFileNodes = self.readInteger(as: UInt64.self),
            let freeFileNodes = self.readInteger(as: UInt64.self),
            let availableFileNodes = self.readInteger(as: UInt64.self),
            let fileSystemID = self.readInteger(as: UInt64.self),
            let rawFlags = self.readInteger(as: UInt64.self),
            let maximumNameLength = self.readInteger(as: UInt64.self)
        else {
            throw SFTPError.protocolViolation("Invalid STATVFS extended reply")
        }

        return .init(
            blockSize: blockSize,
            fundamentalBlockSize: fundamentalBlockSize,
            totalBlocks: totalBlocks,
            freeBlocks: freeBlocks,
            availableBlocks: availableBlocks,
            totalFileNodes: totalFileNodes,
            freeFileNodes: freeFileNodes,
            availableFileNodes: availableFileNodes,
            fileSystemID: fileSystemID,
            flags: .init(rawValue: rawFlags),
            maximumNameLength: maximumNameLength
        )
    }

    @discardableResult
    mutating func writeSFTPFileSystemAttributes(_ attributes: SFTPFileSystemAttributes) -> Int {
        var written = 0
        written += self.writeInteger(attributes.blockSize)
        written += self.writeInteger(attributes.fundamentalBlockSize)
        written += self.writeInteger(attributes.totalBlocks)
        written += self.writeInteger(attributes.freeBlocks)
        written += self.writeInteger(attributes.availableBlocks)
        written += self.writeInteger(attributes.totalFileNodes)
        written += self.writeInteger(attributes.freeFileNodes)
        written += self.writeInteger(attributes.availableFileNodes)
        written += self.writeInteger(attributes.fileSystemID)
        written += self.writeInteger(attributes.flags.rawValue)
        written += self.writeInteger(attributes.maximumNameLength)
        return written
    }
}

