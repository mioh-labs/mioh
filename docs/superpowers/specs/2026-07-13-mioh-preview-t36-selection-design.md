# mioh Preview T36 Model Selection Design

## Goal

Use the fixed-T36 Core AI BasicVSR++ model automatically for the real-time
player on macOS 27 or newer, while leaving normal export model selection and
its T90 default unchanged. On macOS 26, preview continues to use the standard
`basicvsrpp-v1.2` MPS model.

## Selected approach

The Swift runner will choose a preview-only restoration model while building
the preview worker arguments. It will not mutate the restoration model shown
in the restoration tab and will not change `processingArguments`, so normal
exports continue to use the user's selected model. The existing macOS 27
capability check remains the authority:

- macOS 27 or newer: preview uses `basicvsrpp-v1.2-coreai-t36`.
- macOS 26: preview uses `basicvsrpp-v1.2`.
- Normal export: unchanged; its default remains T90 on macOS 27 and standard
  v1.2 on macOS 26.

This is preferred over adding a second picker because the requested behavior
is automatic. It is also preferred over remapping in the Python worker because
model policy belongs with the GUI capability and argument selection logic.

## Clip length

Automatic preview clip length must be derived from the preview-only model, not
from the export selection. T36 therefore uses 104 frames, which is the existing
padding-free streaming length. macOS 26 standard v1.2 retains 180 frames. If
the user explicitly enables the maximum clip length field, that explicit value
continues to take precedence.

## User-visible behavior

The restoration tab remains the export setting. Starting or restarting the
real-time player does not change that picker. The player uses T36 internally on
macOS 27, including when export remains set to T90.

## Testing

Source contract tests will prove that preview selects T36 on macOS 27, falls
back to standard v1.2 on macOS 26, calculates the automatic clip length from
the preview model, and leaves the T90 export default intact. The standalone
Swift sources will then be compiled for the macOS 26 application target, and
the focused and complete Python test suites will be run before completion.
