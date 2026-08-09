import Darwin
import Foundation

/// Writes commands to child-process stdin without allowing a closed pipe to
/// terminate the hosting app with SIGPIPE.
///
/// `FileHandle.write` can throw EPIPE, but Darwin delivers SIGPIPE before Swift
/// gets a chance to turn that condition into an Error. `F_SETNOSIGPIPE` scopes
/// the protection to the parent-side descriptor, so child processes retain
/// their normal signal disposition after exec.
enum MacChildProcessPipe {
  @discardableResult
  static func prepare(_ handle: FileHandle) -> Bool {
    let descriptor = handle.fileDescriptor
    guard descriptor >= 0 else { return false }
    return fcntl(descriptor, F_SETNOSIGPIPE, 1) == 0
  }

  @discardableResult
  static func write(_ data: Data, to handle: FileHandle) -> Bool {
    do {
      try handle.write(contentsOf: data)
      return true
    } catch {
      return false
    }
  }
}
