// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore

extension ByteBuffer {
    mutating func readSFTPAttributes() throws -> SFTPAttributes {
        guard let rawFlags = self.readInteger(as: UInt32.self) else {
            throw SFTPError.protocolViolation("Missing SFTP attribute flags")
        }
        let flags = SFTPAttributeFlags(rawValue: rawFlags)
        let unsupportedBits = rawFlags & ~SFTPAttributeFlags.supported.rawValue
        guard unsupportedBits == 0 else {
            throw SFTPError.protocolViolation("Unsupported SFTP attribute flags \(unsupportedBits)")
        }

        var attributes = SFTPAttributes()
        if flags.contains(.size) {
            guard let size = self.readInteger(as: UInt64.self) else {
                throw SFTPError.protocolViolation("Missing SFTP attribute size")
            }
            attributes.size = size
        }
        if flags.contains(.uidgid) {
            guard let uid = self.readInteger(as: UInt32.self),
                let gid = self.readInteger(as: UInt32.self)
            else {
                throw SFTPError.protocolViolation("Missing SFTP uid/gid")
            }
            attributes.uid = uid
            attributes.gid = gid
        }
        if flags.contains(.permissions) {
            guard let permissions = self.readInteger(as: UInt32.self) else {
                throw SFTPError.protocolViolation("Missing SFTP permissions")
            }
            attributes.permissions = permissions
        }
        if flags.contains(.acmodtime) {
            guard let accessTime = self.readInteger(as: UInt32.self),
                let modificationTime = self.readInteger(as: UInt32.self)
            else {
                throw SFTPError.protocolViolation("Missing SFTP time attributes")
            }
            attributes.accessTime = accessTime
            attributes.modificationTime = modificationTime
        }
        if flags.contains(.extended) {
            guard let extendedCount = self.readInteger(as: UInt32.self) else {
                throw SFTPError.protocolViolation("Missing SFTP extended attribute count")
            }
            guard extendedCount <= SFTPParsingLimits.maximumExtensions,
                Int(extendedCount) <= self.readableBytes / 8
            else {
                throw SFTPError.protocolViolation(
                    "Invalid or excessive SFTP extended attribute count"
                )
            }
            var extensions: [SFTPExtension] = []
            extensions.reserveCapacity(Int(extendedCount))
            for _ in 0..<extendedCount {
                guard let name = self.readSFTPString(
                    maximumBytes: SFTPParsingLimits.maximumExtensionNameBytes
                ), let data = self.readSFTPStringBuffer(
                    maximumBytes: SFTPParsingLimits.maximumExtensionDataBytes
                ) else {
                    throw SFTPError.protocolViolation("Invalid SFTP extended attribute")
                }
                extensions.append(.init(name: name, data: data))
            }
            attributes.extended = extensions
        }
        return attributes
    }

    @discardableResult
    mutating func writeSFTPAttributes(_ attributes: SFTPAttributes) -> Int {
        var written = self.writeInteger(attributes.flags.rawValue)
        if let size = attributes.size {
            written += self.writeInteger(size)
        }
        if attributes.flags.contains(.uidgid) {
            written += self.writeInteger(attributes.uid ?? 0)
            written += self.writeInteger(attributes.gid ?? 0)
        }
        if let permissions = attributes.permissions {
            written += self.writeInteger(permissions)
        }
        if attributes.flags.contains(.acmodtime) {
            written += self.writeInteger(attributes.accessTime ?? 0)
            written += self.writeInteger(attributes.modificationTime ?? 0)
        }
        if attributes.flags.contains(.extended) {
            written += self.writeInteger(UInt32(attributes.extended.count))
            for extensionData in attributes.extended {
                written += self.writeSFTPString(extensionData.name)
                written += self.writeSFTPImmutableStringBuffer(extensionData.data)
            }
        }
        return written
    }
}
