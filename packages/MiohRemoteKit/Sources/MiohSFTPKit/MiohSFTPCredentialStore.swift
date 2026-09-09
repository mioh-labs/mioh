import Foundation
import Security

public enum MiohSFTPCredentialStore {
  private static let service = "com.mioh-labs.MiohRemote.sftp"

  public static func loadPassword(endpointID: String, username: String) -> String? {
    load(account: passwordAccount(endpointID: endpointID, username: username))
  }

  public static func savePassword(
    _ password: String,
    endpointID: String,
    username: String
  ) throws {
    try save(
      password,
      account: passwordAccount(endpointID: endpointID, username: username)
    )
  }

  public static func removePassword(endpointID: String, username: String) {
    remove(account: passwordAccount(endpointID: endpointID, username: username))
  }

  public static func loadHostKey(endpointID: String) -> String? {
    load(account: hostKeyAccount(endpointID: endpointID))
  }

  public static func saveHostKey(_ key: String, endpointID: String) throws {
    try save(key, account: hostKeyAccount(endpointID: endpointID))
  }

  public static func removeHostKey(endpointID: String) {
    remove(account: hostKeyAccount(endpointID: endpointID))
  }

  private static func passwordAccount(endpointID: String, username: String) -> String {
    "password|\(endpointID.lowercased())|\(username)"
  }

  private static func hostKeyAccount(endpointID: String) -> String {
    "host-key|\(endpointID.lowercased())"
  }

  private static func load(account: String) -> String? {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
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

  private static func save(_ value: String, account: String) throws {
    let identity: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
      kSecUseDataProtectionKeychain as String: true,
    ]
    let attributes: [String: Any] = [
      kSecValueData as String: Data(value.utf8),
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

  private static func remove(account: String) {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
      kSecUseDataProtectionKeychain as String: true,
    ]
    SecItemDelete(query as CFDictionary)
  }

  public enum CredentialError: LocalizedError {
    case status(OSStatus)

    public var errorDescription: String? {
      switch self {
      case .status(let status):
        return SecCopyErrorMessageString(status, nil) as String?
          ?? "Keychain error \(status)"
      }
    }
  }
}
