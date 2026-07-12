# mioh Icon Concept Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one square preview of the approved `m` + enclosing `h/i` + `o` concept.

**Architecture:** Use the built-in image generation tool for a preview-only raster concept. Keep the current application icon untouched until the user approves the visual direction.

**Tech Stack:** OpenAI built-in image generation.

## Global Constraints

- The sequence must be lowercase `m`, a large `h` containing `i`, then a separate `o`.
- The full point-topped `i` must sit entirely between the left and right legs of `h`.
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
Primary request: Edit the reference logo into a minimal custom lowercase wordmark arranged as "m", then a large lowercase "h", then a separate lowercase "o". Place a smaller complete lowercase "i", including its round dot and vertical stem, entirely inside the open space under the h arch, centered between the h's left stem and right downstroke. The h visibly surrounds the i with both legs outside it.
Style/medium: clean geometric vector-style logo, crisp flat shapes, professional optical spacing
Composition/framing: centered on a square icon field with generous padding and a strong small-size silhouette
Color palette: dark charcoal mark on a soft light-gray background
Text construction: lowercase m + enclosing h/i glyph + lowercase o
Constraints: i has its own stem; i is fully inside h; h has two visible outer legs; retain i dot; o stays separate; no extra letters
Avoid: gradients, shadows, bevels, 3D, decorative objects, watermark, mockup framing, extra text
```

- [ ] **Step 2: Inspect the output**

Verify that `mio` is legible, the `i` dot remains, the `h` shares the `i` stem, and no separate trailing `h` or extra text appears.

- [ ] **Step 3: Present the preview**

Render the generated image inline without replacing the current app icon or modifying consuming source files.
