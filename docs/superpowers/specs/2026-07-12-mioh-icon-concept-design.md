# mioh Icon Concept Design

## Goal

Create and integrate a new `mioh` app icon based on a custom lowercase wordmark where `h` encloses `i`.

## Mark construction

- The readable sequence is lowercase `m`, a large lowercase `h` containing `i`, then a separate lowercase `o`.
- The `m`, enclosed `i`, and final `o` use the same nominal lowercase font size, stroke weight, and baseline.
- The `i` stem reaches the same x-height as `m` and `o`; its dot sits above that x-height, so the complete dotted `i` is taller than `m` and `o`.
- The enclosing `h` uses the same stroke thickness as `i`, but is extended upward and outward so it surrounds the complete dotted `i`.
- Draw the complete lowercase `h` with a left stem, arch, and right downstroke.
- Place the complete lowercase `i`, including its dot, entirely inside the open space between the two legs of `h`.
- The `i` does not share a stem with `h` and must not sit outside the `h` arch.
- Keep `m` and the final `o` unchanged and separate from the `h/i` construction.

## Visual direction

- Minimal, geometric, vector-friendly logo treatment.
- Dark charcoal mark on a soft light-gray macOS icon field.
- Centered, generous padding, strong silhouette at small sizes.
- No gradients, shadows, 3D effects, decorative objects, watermark, or additional text.

## Deliverable

- Preserve the approved square PNG as `lada/gui/icons/mioh-icon.png`.
- Generate the macOS `AppIcon.icns` from that PNG during standalone app builds.
- Keep the previous `lada-logo-gray.png` source file in place for compatibility, but do not use it for the `mioh` standalone build.
