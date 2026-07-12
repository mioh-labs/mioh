# Progress Output Design

## Goal

Keep Lada progress readable without changing video processing behavior. The CLI and macOS app must update progress in place, and concurrent workers must not overwrite one another.

## Chosen approach

The parent `process_video_parallel.py` process owns presentation. Workers no longer write progress bars directly to shared stdout. They send structured progress events to the parent through a queue. The parent assigns one stable display lane per active worker or segment.

This is preferred over letting workers emit ANSI cursor controls because independent processes can interleave terminal writes. It is also preferred over simple rate limiting because rate limiting does not prevent concurrent progress lines from colliding.

## CLI behavior

- On an interactive terminal, the parent renders one in-place row per active worker. Only the parent moves the cursor or redraws rows.
- A completed segment leaves one ordinary completion line in history, then its lane becomes available to another segment.
- Warnings, errors, phase changes, and summary output remain ordinary durable log lines.
- When stdout is redirected or is not interactive, progress uses newline-delimited structured output at a restrained rate instead of terminal cursor controls.

## macOS app behavior

- The app sets an environment marker so the parallel runner emits machine-readable progress events.
- Swift parses those events and keeps one current row per worker or segment.
- Progress updates replace the matching current row rather than being appended to history.
- Ordinary output remains in the scrollable log. Completion removes the active row and adds a final completion line.
- The existing overall progress bar and status label continue to update from progress percentages.

## Boundaries and failure handling

- Video processing commands, model selection, encoder settings, and worker counts are unchanged.
- Malformed or unknown progress events are preserved as ordinary log text rather than discarded.
- Queue or renderer failure must not terminate restoration; at worst, progress falls back to ordinary output.
- Log history retains its existing size cap.

## Testing

- Unit-test progress-event parsing and stable lane assignment.
- Unit-test that worker progress is reported to the parent rather than printed directly.
- Test non-interactive output and completion cleanup.
- Add a source-level macOS app test for structured progress handling, then build the standalone app to catch Swift compiler errors.
- Run the focused tests first, followed by the relevant parallel-processing and standalone-app test suites.
