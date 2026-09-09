import NIOCore
import XCTest
@testable import NIOSFTP

final class NIOSFTPSecurityTests: XCTestCase {
  func testOversizedFrameIsRejectedFromHeaderAlone() {
    var buffer = ByteBuffer()
    buffer.writeInteger(SFTPParsingLimits.maximumFrameBytes + 1)

    XCTAssertThrowsError(try buffer.readSFTPFrame())
    XCTAssertEqual(buffer.readerIndex, 0)
  }

  func testNameCountMustFitPayloadAndConfiguredLimit() {
    var payload = ByteBuffer()
    payload.writeInteger(UInt32(1))
    payload.writeInteger(UInt32.max)

    XCTAssertThrowsError(
      try SFTPInboundPacketParser.parse(
        type: SFTPPacketType.name,
        payload: payload
      )
    )
  }

  func testExtendedAttributeCountMustFitPayloadAndConfiguredLimit() {
    var attributes = ByteBuffer()
    attributes.writeInteger(SFTPAttributeFlags.extended.rawValue)
    attributes.writeInteger(UInt32.max)

    XCTAssertThrowsError(try attributes.readSFTPAttributes())
  }

  func testNameMetadataHasIndependentSizeLimits() {
    var payload = ByteBuffer()
    payload.writeInteger(UInt32(7))
    payload.writeInteger(UInt32(1))
    payload.writeInteger(UInt32(SFTPParsingLimits.maximumFilenameBytes + 1))
    payload.writeRepeatingByte(
      UInt8(ascii: "a"),
      count: SFTPParsingLimits.maximumFilenameBytes + 1
    )
    payload.writeSFTPString("")
    payload.writeInteger(UInt32(0))

    XCTAssertThrowsError(
      try SFTPInboundPacketParser.parse(
        type: SFTPPacketType.name,
        payload: payload
      )
    )
  }

  func testKnownResponseRejectsTrailingPayload() {
    var payload = ByteBuffer()
    payload.writeInteger(UInt32(8))
    payload.writeInteger(UInt32(SFTPStatusCode.ok.rawValue))
    payload.writeSFTPString("ok")
    payload.writeSFTPString("")
    payload.writeInteger(UInt8(0xff))

    XCTAssertThrowsError(
      try SFTPInboundPacketParser.parse(
        type: SFTPPacketType.status,
        payload: payload
      )
    )
  }
}
