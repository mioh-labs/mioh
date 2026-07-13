# mioh Restoration Slider Controls Design

## Goal

Make restoration composition and ROI enhancer tuning faster in the standalone mioh GUI by combining sliders with editable numeric fields.

## Scope

Replace the current numeric-only controls with slider-plus-number controls for:

| Section | Setting | Range | Step |
| --- | --- | ---: | ---: |
| Composition | Sharpen | 0.0–1.0 | 0.05 |
| Composition | Detail | 0.0–1.0 | 0.05 |
| Composition | Blend feather | 0.0–3.0 | 0.05 |
| Composition | Texture | 0.0–1.0 | 0.01 |
| Composition | Smoothing | 0.0–1.0 | 0.05 |
| ROI enhancer | Strength | 0.0–1.0 | 0.05 |
| ROI enhancer | Tile | 0–1024 | 32 |

Keep both multiplier controls unchanged:

- Composition effect scale remains a `Stepper` over 1–4.
- ROI enhancer scale remains a `Stepper` over 1–8.

Keep the model path and all other restoration controls unchanged.

## UI Components

Add two private reusable SwiftUI helpers in `MiohApp.swift`:

- A double slider row that accepts a title, `Binding<Double>`, closed range, and step. It displays a flexible-width `Slider` followed by a trailing editable number field formatted to at most three fractional digits.
- An integer slider row that accepts a title, `Binding<Int>`, closed range, and step. It displays a `Slider` backed by a `Double` bridge followed by an editable integer field.

Both helpers clamp values to their declared ranges when a numeric edit is committed. Existing `@Published` properties remain the source of truth, so processing argument generation and defaults do not change.

## Layout and State

Each row keeps its Japanese label on the left. The slider expands into the available form width and the numeric field stays narrow on the right. ROI strength and tile retain the existing disabled state when the enhancer method is `none`.

No setting persistence format or command-line argument changes are needed. Starting a job passes the same seven property values through the existing options.

## Testing

Extend `tests/test_standalone_app_options.py` to verify:

- all seven requested settings use the new slider helpers with their exact ranges and steps;
- the two multiplier settings still use their existing `Stepper` controls;
- ROI strength and tile remain disabled when the enhancer is `none`;
- existing command-line options remain exposed.

Run the standalone GUI option tests and the complete Python test suite. Build the standalone Swift source through the existing macOS app build path if its dependencies are already available; otherwise the static option tests remain the non-packaging verification boundary.
