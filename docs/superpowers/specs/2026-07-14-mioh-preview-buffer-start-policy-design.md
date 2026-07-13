# mioh Preview Buffer Start Policy Design

## Goal

Make the buffer-limit slider define how much restored video must be ready before
playback starts or resumes. Keep a user-selectable fast recovery mode for brief
rebuffering during playback.

## Root Cause

The worker already limits generated preview data using the selected buffer
seconds, but the Swift player starts from fixed segment counts:

- three segments for initial playback;
- two segments after playback has already started.

Because each segment is normally two seconds, a large buffer selection such as
60 seconds still starts after roughly six seconds. The selected limit currently
controls production capacity, not the playback start threshold.

## User Interface

- Keep the existing `バッファ上限` slider at 1 through 60 seconds, in
  one-second steps.
- Add a `再バッファを短縮` checkbox beside the buffer controls.
- The checkbox defaults to off.
- When off, both initial playback and recovery from rebuffering wait for the
  selected number of buffered seconds.
- When on, initial playback still waits for the selected number of seconds, but
  recovery from rebuffering resumes after at least the smaller of 4.0 seconds
  and the selected buffer limit.

## Playback Policy

Swift uses the actual queued media duration rather than assuming every segment
has an exact fixed duration.

- Initial playback target: the current buffer-limit value.
- Rebuffer target with `再バッファを短縮` off: the current buffer-limit value.
- Rebuffer target with `再バッファを短縮` on: the smaller of 4.0 seconds and
  the selected buffer limit. This prevents a 1-to-3-second worker limit from
  making the four-second recovery target impossible to reach.
- A changed slider value becomes the active threshold immediately while the
  player is buffering.
- Changing the slider during active playback does not pause, seek, restart the
  worker, change generation, or clear queued segments. The new value applies to
  the next buffering decision and continues to update the worker production
  limit through the existing command.
- At end of file, if the remaining duration is shorter than the selected target,
  playback begins once all remaining restored segments are ready.

The target comparison uses the queued range from the current requested start
position through the end of the last queued segment. This avoids depending on
the source player's periodically updated playhead while playback is stopped for
buffering.

## Data Ownership

`RestorationRunner` remains the source of truth for both preview preferences:

- selected buffer-limit seconds;
- whether shortened rebuffer recovery is enabled.

`RealtimePlayerController` reads these values when deciding whether enough
restored media is queued. The Python worker continues to own only generation
capacity and accepts the existing live `set_buffer_limit` command.

## Errors and Edge Cases

- The UI keeps the buffer limit positive and bounded to 60 seconds.
- If no complete restored segment exists, playback remains buffering even at
  end of file.
- A short source or a seek near the end cannot deadlock waiting for an
  impossible target; the end-of-file path permits playback of all available
  remaining segments.
- The fast-recovery option does not affect initial startup.

## Verification

- Add a failing source-contract test for the new preference and removal of the
  fixed startup-segment condition.
- Verify the controller's required-buffer calculation covers initial startup,
  normal rebuffering, fast rebuffering, slider changes while buffering, and the
  short end-of-file case.
- Compile the macOS 26 Swift target and run the complete Python test suite.
- Rebuild and sign the standalone app and DMG.
- In the built app, confirm playback stays in `バッファ中` until the selected
  threshold is reached, then confirm `再バッファを短縮` only changes recovery
  behavior and never the initial startup target.
