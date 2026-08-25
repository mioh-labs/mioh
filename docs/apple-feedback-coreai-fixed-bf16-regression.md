# coreai-build fixed-shape BF16 input regression

> Submission status: **submitted to Apple on 2026-08-25 as `FB24503497`**.
>
> Attached reproduction bundle: `apple-feedback-coreai-fixed-bf16-20260825.zip`.

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
.venv-coreai/bin/python scripts/apple/canary_coreai_fixed_bf16.py
```

The canary exports a fixed-shape BF16 function `y = 2x + 1`, verifies its
declared descriptors, runs the source asset, and then asks `coreai-build` for
an ahead-of-time GPU specialization.

## Result

The source asset declares BF16 input and output and executes correctly:

```text
inputScalarType:  bfloat16
outputScalarType: bfloat16
input:             [1, -2, 0.5, 4]
output:            [3, -3, 2, 9]
```

`coreai-build compile` aborts with return code `-6`:

```text
Incompatible element type for parameter at index 0,
mlir module expected element type bf16 but received f16
```

The exact Beta 6 failure therefore remains in Beta 7. The durable machine
report is `/tmp/mioh-coreai-fixed-bf16/report.json`.

## Current workaround

Do not ahead-of-time specialize exact fixed-shape BF16 source assets. Mioh's
H3 path uses dynamic BF16 DiT assets, which compile and execute correctly, and
keeps the fixed-BF16 path out of `coreai-build`.
