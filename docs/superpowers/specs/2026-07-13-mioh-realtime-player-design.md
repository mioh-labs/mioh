# mioh Buffered Real-Time Player Design

## Goal

Add a native macOS player to mioh that begins playing restored video after a short buffer, keeps processing ahead of playback, supports seeking, preserves synchronized source audio, and can switch between original and restored video.

## Scope

The first release includes:

- a new `再生` tab in the standalone SwiftUI application;
- opening the input selected in the existing `基本` tab;
- play, pause, stop, seek, volume, and mute controls;
- original/restored display switching;
- a visible loading and rebuffering state;
- approximately six seconds of restored video buffered ahead;
- restoration and detection settings captured from the existing tabs when preview starts;
- temporary preview cleanup when preview stops or the application exits.

Export behavior and output files remain unchanged. Subtitles, frame stepping, playlists, and AirPlay are outside this first release.

## Approaches Considered

### Persistent restoration worker plus compressed segments (selected)

A persistent Python worker loads the selected detection and restoration models once, restores frames from the requested source timestamp, and uses VideoToolbox H.264 to write two-second video-only MP4 segments. Swift receives segment-ready events and plays the ordered segments through `AVQueuePlayer`. A separate source `AVPlayer` provides audio and acts as the playback clock.

This preserves a native application and a persistent model while keeping a six-second 1080p buffer compressed. The bundled PyAV and ffmpeg installations both expose `h264_videotoolbox`.

### Raw frames from Python to Swift

Python could stream BGRA frames directly to a Swift display layer. This avoids preview encoding but five to eight seconds of 1080p BGRA frames consumes roughly 1.2 to 2 GB, which is unsuitable for the minimum M1 8 GB target.

### Bundle the legacy GStreamer player

The original GTK application already has a working real-time GStreamer pipeline. Bundling GTK, GStreamer, and PyGObject would substantially increase application size and add a second UI/runtime stack, so it is not selected.

## Architecture

### Swift player controller

`RealtimePlayerController` is an `@MainActor` observable object owned by the new player tab. It is responsible for:

- validating the selected input;
- snapshotting the current model and restoration settings;
- starting and stopping the preview worker;
- reading newline-delimited JSON events from worker stdout;
- maintaining ordered ready-segment metadata;
- feeding restored segments to an `AVQueuePlayer`;
- controlling the source `AVPlayer` used for audio and original display;
- applying seek and play/pause operations to both players;
- publishing player state, position, duration, buffered duration, and errors to SwiftUI.

The player state is one of `idle`, `loading`, `buffering`, `playing`, `paused`, `seeking`, `ended`, or `failed`.

### Python preview worker

`mioh_preview_worker.py` runs inside the bundled Python runtime. It receives all preview configuration as command-line arguments and control commands as newline-delimited JSON on stdin.

At startup it:

1. resolves and loads the detection and restoration models once;
2. reads video metadata;
3. emits a `ready` event containing duration, frame rate, width, and height;
4. creates a `FrameRestorer` using the loaded models;
5. starts restoration at the requested nanosecond timestamp.

Restored BGR frames are encoded by PyAV with `h264_videotoolbox` into independent two-second MP4 files. Segments use a stable sequence number and carry their absolute source start/end timestamps in the event protocol.

Seeking stops only the current `FrameRestorer`, empties the pending frame queue, removes unconsumed segments, and constructs a new `FrameRestorer` with the same loaded model objects at the requested timestamp. Models are not reloaded.

### Event protocol

Worker stdout contains only JSON event lines. Diagnostics go to stderr so they cannot corrupt the protocol.

Events are:

- `ready`: video metadata and worker generation;
- `segment`: generation, sequence, absolute start/end nanoseconds, and path;
- `progress`: generation, processed position, and buffered-ahead estimate;
- `buffer_full`: the configured look-ahead limit has been reached;
- `buffer_limit`: generation and the applied buffer-limit seconds, emitted after
  a live `set_buffer_limit` command has updated the worker configuration;
- `ended`: no more restored frames;
- `error`: a user-readable message plus a diagnostic detail string.

Every event includes a monotonically increasing generation number. Swift ignores events from an older generation after a seek.

Control commands are:

- `seek` with an absolute nanosecond position;
- `set_buffer_limit` with seconds;
- `stop`.

Pause does not tear down the model. Swift pauses both players while the worker continues until its compressed look-ahead limit is reached, then the worker waits.

## Buffering and Storage

Each restored segment is two seconds long. Playback starts when at least three ordered segments, approximately six seconds, are ready, or when end-of-file is reached with a shorter remainder.

The worker stops producing when eight seconds are ready ahead of the last consumed position. Swift sends consumption acknowledgements by deleting played files; the worker also observes the configured limit and blocks production. Swift deletes a segment after `AVQueuePlayer` has advanced beyond it.

Preview files live under the configured mioh temporary directory when provided, otherwise under the system temporary directory. Each session uses a unique directory. Cleanup occurs on stop, new preview, application termination, and the next launch for abandoned directories older than one day.

## Audio and Synchronization

The source `AVPlayer` is the master clock and plays the original audio. Its video layer is visible only in original mode. The restored `AVQueuePlayer` is muted and visible only in restored mode.

When restored playback starts, both players seek to the absolute start timestamp and start together. At each restored segment boundary Swift compares the source clock with the restored absolute segment position. A drift greater than 80 milliseconds triggers a precise seek on the restored queue before playback resumes.

When restored data underruns, both players pause. Playback resumes only after at least two ordered segments are available, preventing audio from running ahead of restored video.

Switching to original mode does not stop restoration. It shows the source video at the current source clock immediately. Switching back to restored mode waits for the corresponding restored generation and displays buffering if necessary.

## Settings and Platform Compatibility

Preview captures the current settings when the user presses the preview start button. Changing model, detection, composition, ROI-enhancer, or memory settings while preview is active does not mutate the running worker; the UI offers `設定を反映して再開` to restart preview at the current timestamp.

The existing platform capability rules remain authoritative:

- macOS 26 exposes and accepts MPS restoration and Core ML/PyTorch detection;
- macOS 27 and newer may use CoreAI restoration and detection;
- CoreAI custom paths are rejected on macOS 26;
- the selected parallel-worker value is not restricted, although the preview worker itself is one persistent restoration pipeline rather than the export segment executor.

## Failure Handling

- Worker startup failure moves the controller to `failed` and shows stderr detail in the log tab.
- A malformed or stale event is ignored and logged.
- A missing or invalid segment triggers one rebuffer attempt; a repeated failure stops preview with an error.
- VideoToolbox encoder initialization failure falls back to `libx264` with the ultrafast preset and emits a warning.
- Seeking cancels queued items and older-generation events before accepting new segments.
- Stop first requests orderly shutdown, then terminates the worker if it has not exited within three seconds.

## Testing

Python tests cover protocol serialization, two-second segment boundaries, VideoToolbox-to-libx264 fallback, buffer backpressure, model reuse across seek, stale-generation cleanup, and orderly stop.

Swift source contract tests cover the new player tab, AVFoundation linkage, player state, controls, original/restored toggle, JSON protocol handling, generation filtering, rebuffer thresholds, audio clock synchronization, and worker packaging.

Integration verification uses a synthetic video to confirm that the worker produces playable ordered segments, emits valid events, seeks without reloading models, and removes its temporary session. The standalone app is then compiled for macOS 26 and the complete Python test suite is run.
