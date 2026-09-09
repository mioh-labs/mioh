// Copyright (c) 2026 EkkoG.
// SPDX-License-Identifier: MIT

import NIOCore

enum SFTPParsingLimits {
    static let maximumFrameBytes: UInt32 = 16 * 1_024 * 1_024
    static let maximumNameEntries: UInt32 = 10_000
    static let maximumExtensions: UInt32 = 4_096
    static let maximumServerReadBytes: UInt32 = 1 * 1_024 * 1_024
    static let maximumServerDirectoryBatchEntries = 256
    static let maximumFilenameBytes = 255
    static let maximumLongnameBytes = 4_096
    static let maximumStatusMessageBytes = 4_096
    static let maximumLanguageTagBytes = 64
    static let maximumHandleBytes = 64 * 1_024
    static let maximumExtensionNameBytes = 256
    static let maximumExtensionDataBytes = 1 * 1_024 * 1_024
}

extension ByteBuffer {
    mutating func readSFTPStringBuffer() -> ByteBuffer? {
        guard let length = self.readInteger(as: UInt32.self) else {
            return nil
        }
        return self.readSlice(length: Int(length))
    }

    mutating func readSFTPString() -> String? {
        guard let bytes = self.readSFTPStringBuffer() else {
            return nil
        }
        return String(buffer: bytes)
    }

    mutating func readSFTPString(maximumBytes: Int) -> String? {
        guard let length = self.getInteger(at: self.readerIndex, as: UInt32.self),
            length <= UInt32(maximumBytes)
        else {
            return nil
        }
        return self.readSFTPString()
    }

    mutating func readSFTPStringBuffer(maximumBytes: Int) -> ByteBuffer? {
        guard let length = self.getInteger(at: self.readerIndex, as: UInt32.self),
            length <= UInt32(maximumBytes)
        else {
            return nil
        }
        return self.readSFTPStringBuffer()
    }

    @discardableResult
    mutating func writeSFTPString<Buffer: Collection>(_ bytes: Buffer) -> Int where Buffer.Element == UInt8 {
        let written = self.writeInteger(UInt32(bytes.count))
        return written + self.writeBytes(bytes)
    }

    @discardableResult
    mutating func writeSFTPString(_ value: String) -> Int {
        self.writeSFTPString(value.utf8)
    }

    @discardableResult
    mutating func writeSFTPStringBuffer(_ value: inout ByteBuffer) -> Int {
        let written = self.writeInteger(UInt32(value.readableBytes))
        let bufferWritten = self.writeBuffer(&value)
        return written + bufferWritten
    }

    @discardableResult
    mutating func writeSFTPImmutableStringBuffer(_ value: ByteBuffer) -> Int {
        var copy = value
        return self.writeSFTPStringBuffer(&copy)
    }

    mutating func readSFTPFrame() throws -> SFTPInboundPacket? {
        try self.readSFTPFrame(using: SFTPInboundPacketParser.parse(type:payload:))
    }

    mutating func readSFTPServerFrame() throws -> SFTPServerInboundPacket? {
        try self.readSFTPFrame(using: SFTPServerInboundPacketParser.parse(type:payload:))
    }

    private mutating func readSFTPFrame<T>(
        using parser: (UInt8, ByteBuffer) throws -> T
    ) throws -> T? {
        guard let length = self.getInteger(at: self.readerIndex, as: UInt32.self) else {
            return nil
        }

        guard length >= 1 else {
            throw SFTPError.protocolViolation("SFTP packet length must include a type byte")
        }
        guard length <= SFTPParsingLimits.maximumFrameBytes else {
            throw SFTPError.protocolViolation("SFTP packet exceeds the configured frame limit")
        }

        let frameLength = Int(length)
        guard self.readableBytes >= frameLength + 4 else {
            return nil
        }

        _ = self.readInteger(as: UInt32.self)
        guard let typeRaw = self.readInteger(as: UInt8.self) else {
            return nil
        }
        guard let payload = self.readSlice(length: frameLength - 1) else {
            return nil
        }
        return try parser(typeRaw, payload)
    }

    @discardableResult
    mutating func writeSFTPFrame(type: UInt8, bodyWriter: (inout ByteBuffer) throws -> Void) rethrows -> Int {
        let lengthIndex = self.writerIndex
        self.writeInteger(UInt32(0))
        self.writeInteger(type)
        let bodyStart = self.writerIndex
        try bodyWriter(&self)
        let totalLength = self.writerIndex - bodyStart + 1
        self.setInteger(UInt32(totalLength), at: lengthIndex)
        return totalLength + 4
    }
}
