# Core AI specialization aliases weight-distinct programs

> Submission status: **submitted to Apple on 2026-08-25 as `FB24503701`**.
>
> Attached files: `apple-feedback-coreai-weight-cache-20260825.zip` and
> `CoreAIWeightCacheCanary.swift`.

## Environment

- macOS 27.0 Developer Beta 7, build `26A5421a`
- Xcode 27.0 Developer Beta 6, build `27A5252f`
- `coreai-build 3600.83.1`
- PyTorch 2.11.0
- Core AI Torch 0.4.2
- Core AI Core 1.0.0b2
- M5 Pro specialization `h17s`

## Reproduction

```zsh
.venv-coreai/bin/python scripts/apple/canary_coreai_weight_cache_identity.py
```

The canary exports two structurally identical Float32 functions that differ
only in a learned constant:

```text
model A: y = x * 2
model B: y = x * 5
input:   x = 3
```

It executes the source and compiled assets in `ABAB`, `BABA`, `AABB`, and
`BBAA` orders.

## Result

Source execution is correct in every order: A returns `6`, B returns `15`.
The two source `main.hash` files are distinct.

After compiling A and then B, the two compiled `main.hash` files and compiled
`main-h17s.mlirb` files are byte-identical. Every compiled invocation returns
the A result `6`, including B in a fresh Swift process:

```text
expected ABAB: 6, 15, 6, 15
actual ABAB:   6,  6, 6,  6
expected BABA: 15, 6, 15, 6
actual BABA:    6, 6,  6, 6
```

This is not merely a stale runtime output cache: the compiler emitted the
wrong identical program for B. The durable machine report is
`/tmp/mioh-coreai-weight-cache/report.json`.

## Current workaround

Repeated H3 blocks retain their weight SHA and layer range in the entry-point
identity and carry a structurally consumed salt input/output. Removing that
identity would make the current application vulnerable to this still-present
compiler-cache alias.
