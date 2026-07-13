# mioh Preview Buffer Slider Design

## Goal

Let the user choose the realtime preview buffer limit from the playback tab,
up to one minute, without interrupting active playback.

## User Interface

- Add a `バッファ上限` slider to the playback tab.
- Allow values from 1 through 60 seconds in one-second steps.
- Show the selected value beside the slider as an integer number of seconds.
- Keep 8 seconds as the default.

## Data Flow

- `RestorationRunner` owns the selected buffer limit so the playback UI and
  preview worker arguments use one source of truth.
- Starting a preview passes the selected value through `--buffer-limit`.
- Moving the slider during an active preview sends the existing
  `set_buffer_limit` command to the worker immediately.
- Moving it while idle only changes the value used by the next preview.
- Changing the limit does not seek, restart, pause, or clear queued segments.

## Validation and Errors

- The UI constrains values to 1...60 seconds.
- The worker keeps its existing positive-value validation as a defensive
  boundary for non-UI callers.
- A failed live command follows the existing worker failure and log handling;
  no new error surface is introduced.

## Tests

- Verify the Swift UI exposes a 1...60 second slider with an 8-second default.
- Verify preview arguments use the selected value instead of the former fixed
  `8.0` value.
- Verify an active controller sends `set_buffer_limit` without restarting the
  preview generation.
- Run the complete Python suite, compile both Swift targets, rebuild and sign
  the app, and exercise the slider in the playback tab.
