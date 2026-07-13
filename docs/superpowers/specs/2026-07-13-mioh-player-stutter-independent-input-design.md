# mioh Player Stutter and Independent Input Design

## Goal

Stop the restored player from jumping back to the generation start whenever a
new preview segment arrives, and let the playback tab select its own input
video independently from the basic tab. Restoration, detection, composition,
ROI enhancer, device, and memory settings remain shared with the existing
tabs.

## Observed failure

The built application was exercised through its playback tab. Playback reached
about six seconds with 3.8 seconds reported ahead, then the source position
jumped to zero while the reported look-ahead jumped to 11.4 seconds. The state
remained `playing` throughout.

Every segment event currently calls `resumeIfBuffered`. Once three or more
segments are queued, that method seeks the source player to
`requestedStartSeconds` and starts both players, even when they are already
playing. Pause resume and underrun recovery use the same method, so they can
also seek back to the generation start.

## Playback state design

Add an explicit `generationHasStarted` flag to distinguish the one-time start
of a generation from normal segment delivery and rebuffering.

- A new worker generation starts with `generationHasStarted = false`.
- Initial playback waits for three segments, seeks the source player to the
  requested generation start once, starts both players, and sets the flag.
- Segment events received while already playing only append to the queue.
  They never seek or restart either player.
- Pause resume starts both players from their current positions without a
  seek.
- Underrun recovery waits for two segments and starts both players from their
  current boundary positions without a seek.
- An explicit user seek increments the worker generation, clears the restored
  queue, records the new generation start, and resets the flag.
- Stop clears the flag and both player queues.

The existing 80 ms drift correction is not changed in this pass. The verified
generation-reset defect is fixed first. If a smaller hitch remains after live
verification, drift correction and MP4 item boundaries will be measured as a
separate issue rather than changed speculatively.

## Independent playback input

`RealtimePlayerController` owns a published `previewInputURL` that is separate
from `RestorationRunner.inputURL`.

The playback tab adds a file row with the selected path and a folder button.
The button opens a single-file movie picker. Selecting a new movie stops any
active preview session, assigns the new preview URL, resets position and
duration, and prepares the source player. The play button is disabled until a
preview movie has been selected. There is no fallback to the basic-tab input.

`RestorationRunner.previewArguments` receives the preview input URL explicitly
and places that path in `--input`. It continues to read processing settings
from the runner. Normal export continues to read `RestorationRunner.inputURL`
and is otherwise unchanged.

## Error handling

- Cancelling the movie picker preserves the current preview input.
- A missing or unreadable preview file produces the existing player error
  state and does not affect the basic-tab input.
- Choosing another movie during playback stops the worker and removes its
  temporary segments before preparing the new source.
- Stale worker events remain rejected by generation number.

## Alternatives rejected

A local HLS playlist could provide a more specialized segmented playback
pipeline, but it would add playlist lifecycle and discontinuity handling
without addressing the verified restart bug more directly. Appending to a
single growing MP4 is unsuitable because finalized MP4 metadata is not safely
available as an indefinitely growing playback asset.

## Testing and verification

Automated source-contract tests will cover the independent preview input,
explicit preview argument input, generation-start flag, one-time initial seek,
no-seek pause resume, and no-seek underrun recovery. Existing worker and app
tests must continue to pass, and the Swift application sources must compile for
arm64 macOS 26.

The standalone app will then be rebuilt and exercised with an actual video.
After restored playback begins, sampled positions must progress for at least
12 seconds without returning to the generation start. The dedicated playback
input will also be selected through the playback tab and verified to leave the
basic-tab input unchanged.
