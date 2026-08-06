import Foundation
import Security

public enum RemoteCredentialStore {
  private static let service = "com.mioh-labs.MiohRemote.credentials"

  public static func loadToken(serverID: String) -> String? {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: serverID,
      kSecUseDataProtectionKeychain as String: true,
      kSecMatchLimit as String: kSecMatchLimitOne,
      kSecReturnData as String: true,
    ]
    var result: CFTypeRef?
    guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
      let data = result as? Data
    else { return nil }
    return String(data: data, encoding: .utf8)
  }

  public static func saveToken(_ token: String, serverID: String) throws {
    let identity: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: serverID,
      kSecUseDataProtectionKeychain as String: true,
    ]
    let attributes: [String: Any] = [
      kSecValueData as String: Data(token.utf8),
      kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
    ]
    let update = SecItemUpdate(identity as CFDictionary, attributes as CFDictionary)
    if update == errSecSuccess { return }
    guard update == errSecItemNotFound else { throw CredentialError.status(update) }
    var addition = identity
    addition.merge(attributes) { _, replacement in replacement }
    let add = SecItemAdd(addition as CFDictionary, nil)
    guard add == errSecSuccess else { throw CredentialError.status(add) }
  }

  public static func removeToken(serverID: String) {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: serverID,
      kSecUseDataProtectionKeychain as String: true,
    ]
    SecItemDelete(query as CFDictionary)
  }

  public enum CredentialError: LocalizedError {
    case status(OSStatus)

    public var errorDescription: String? {
      switch self {
      case .status(let status):
        return SecCopyErrorMessageString(status, nil) as String? ?? "Keychain error \(status)"
      }
    }
  }
}
