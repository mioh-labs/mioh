# mioh App Rename Design

## Goal

Rename the standalone macOS application from `Lada` to lowercase `mioh` without renaming or changing the underlying Lada processing engine.

## User-facing name

- The app bundle is `mioh.app`.
- Finder, the menu bar, window identity, and the in-app header show `mioh`.
- The main executable is named `mioh`.
- Distribution output is `mioh-0.11.0-unsigned.dmg`, with volume name `mioh` and `mioh.app` at its root.
- The visible temporary-directory label becomes `mioh一時フォルダ` while it continues to pass the internal `--lada-temp-dir` option.

## Compatibility boundary

- Keep `CFBundleIdentifier` as `com.okatti.lada.coreai` so macOS treats the renamed build as the existing application identity.
- Keep the Python package, CLI module, environment variables, Core AI helper name, model filenames, and internal `lada` paths unchanged.
- Keep the existing icon asset unchanged.

## Source and build layout

- Rename the Swift app entry source to `MiohApp.swift` and its `@main` type to `MiohStandaloneApp`.
- Update the standalone build script and tests to use the new source, executable, app, and DMG names.
- Remove obsolete generated `Lada.app` and `Lada-0.11.0-unsigned.dmg` artifacts during a successful mioh build to avoid duplicate copies with the same Bundle ID.

## Verification

- Test the expected product names and the preserved Bundle ID before implementation.
- Run the standalone app tests.
- Build and code-sign `mioh.app`.
- Verify `Info.plist`, executable name, DMG name, bundled processing script, and ROI model assets.
- Launch the built app and confirm its accessibility-visible title is `mioh`.
