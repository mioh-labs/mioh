# mioh Icon Concept Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one square preview of the approved `mio` plus overlaid `h` ligature concept.

**Architecture:** Use the built-in image generation tool for a preview-only raster concept. Keep the current application icon untouched until the user approves the visual direction.

**Tech Stack:** OpenAI built-in image generation.

## Global Constraints

- Base text must read lowercase `mio`.
- The `i` and lowercase `h` share one vertical stem, with the `i` dot retained.
- Use dark charcoal on a soft light-gray square macOS icon field.
- No gradients, shadows, 3D effects, objects, watermark, or extra text.
- Do not replace the current project icon during this task.

---

### Task 1: Generate and review the concept

**Files:**
- Preview only: generated image remains in the built-in image output location.
- Preserve: `lada/gui/icons/lada-logo-gray.png`

**Interfaces:**
- Produces one inline square logo preview for user review.

- [ ] **Step 1: Generate the approved composition**

Use this prompt:

```text
Use case: logo-brand
Asset type: square macOS application icon concept preview
Primary request: Create a minimal custom lowercase wordmark based on the exact readable letters "mio". Overlay a lowercase "h" onto the "i" so the i and h share exactly one vertical stem. Keep the i dot clearly visible. Add only the h shoulder and right downstroke, flowing rightward from the shared i stem. The result should suggest "mioh" as one compact ligature while the base "mio" remains readable.
Style/medium: clean geometric vector-style logo, crisp flat shapes, professional optical spacing
Composition/framing: centered on a square icon field with generous padding and a strong small-size silhouette
Color palette: dark charcoal mark on a soft light-gray background
Text (verbatim): "mio" with the overlaid lowercase h ligature on i
Constraints: one shared i/h stem; retain i dot; no separate h after the o; no other letters
Avoid: gradients, shadows, bevels, 3D, decorative objects, watermark, mockup framing, extra text
```

- [ ] **Step 2: Inspect the output**

Verify that `mio` is legible, the `i` dot remains, the `h` shares the `i` stem, and no separate trailing `h` or extra text appears.

- [ ] **Step 3: Present the preview**

Render the generated image inline without replacing the current app icon or modifying consuming source files.
