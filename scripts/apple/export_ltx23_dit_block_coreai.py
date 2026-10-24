#!/usr/bin/env python3
"""Export LTX-2.3 video DiT blocks (with an IC-LoRA baked in) as a Core AI asset, verify it and time it.

Video-only path of ltx-core's BasicAVTransformerBlock for the mioh ROI-refine window: fixed token count (target
plus appended reference tokens), fixed prompt context, gated self- and cross-attention with split RoPE, cross-
attention AdaLN. The per-token AdaLN rows come in compact: `modulation` holds one projected timestep row per
distinct sigma and `row_select` (one-hot, tokens x rows) picks each token's row, so the asset never carries a
per-token 9 x 4096 table.

usage: export_ltx23_dit_block_coreai.py --checkpoint CKPT.safetensors --layer L [--layer-count N]
           [--lora LORA.safetensors] --output ASSET [--tokens 2560] [--context 1024] [--benchmark 5]
           [--max-tokens 2560 --bench-tokens 640,1440,2560]   (variable token count)
           [--int8-group 64 --weight-bits 4]                  (packed int4 weights)"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

DIM = 4096
HEADS = 32
HEAD_DIM = 128
FFN = 16384
ADA = 9
EPS = 1e-6
PREFIX = "model.diffusion_model.transformer_blocks"
MIOH_PREFIX = "mioh_q.transformer_blocks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--tokens", type=int, default=2560)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--context-kv", action="store_true",
                        help="take precomputed text cross-attention keys/values (context_key/context_value, "
                             "blocks x S x 4096 fp16) instead of context + prompt_modulation")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="export a variable token count up to this (0 = fixed --tokens)")
    parser.add_argument("--bench-tokens", default="", help="comma list of token counts to benchmark (variable export)")
    parser.add_argument("--rows", type=int, default=2, help="distinct timestep rows (sigma, 0)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=int, default=5)
    parser.add_argument("--int8-group", type=int, default=0,
                        help="quantized weights with this input group size (0 = bf16; option name kept for compatibility)")
    parser.add_argument("--weight-bits", type=int, choices=(4, 8), default=8,
                        help="packed weight precision when --int8-group is nonzero (default: 8)")
    parser.add_argument("--skip-reference-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _unpack_mlx_affine_int4(
    packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, group_size: int = 64
) -> torch.Tensor:
    """Dequantize MLX affine int4 weights without materialising the full checkpoint.

    MLX stores eight unsigned four-bit values in each uint32, least-significant
    nibble first, then applies one scale and bias to each input group.
    """
    if packed.dtype != torch.uint32 or packed.ndim != 2:
        raise ValueError("mioh int4 weights must be a 2D uint32 tensor")
    rows, words = packed.shape
    columns = words * 8
    if scales.shape != biases.shape or scales.shape != (rows, columns // group_size):
        raise ValueError("mioh int4 scales/biases do not match weight shape")
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    values = ((packed.to(torch.int64)[..., None] >> shifts) & 0xF).reshape(rows, columns).float()
    return (values.reshape(rows, -1, group_size)
            * scales.float()[..., None] + biases.float()[..., None]).reshape(rows, columns).to(torch.bfloat16)


def _mioh_name(name: str) -> str:
    return (name.replace("attn1.to_out.0", "attn1.to_out")
            .replace("attn2.to_out.0", "attn2.to_out")
            .replace("ff.net.0.proj", "ff.project_in.proj")
            .replace("ff.net.2", "ff.project_out"))


def load_weights(checkpoint: Path, layer: int, lora: Path | None, strength: float) -> dict[str, torch.Tensor]:
    """bf16 block tensors with `W += strength * B @ A` for every LoRA pair (alpha = rank, as ltx-trainer saves)."""
    with safe_open(str(checkpoint), framework="pt") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("mioh_pack_quantization") == "int4":
            if lora is not None:
                raise ValueError("a mioh pack already has its LoRA baked in; do not pass --lora")
            prefix = f"{MIOH_PREFIX}.{layer}."
            keys = set(handle.keys())
            weights: dict[str, torch.Tensor] = {}
            expected = [key[len(prefix):] for key in keys if key.startswith(prefix)]
            for packed_name in expected:
                if packed_name.endswith((".scales", ".biases")):
                    continue
                source = prefix + packed_name
                target = packed_name
                target = (target.replace("attn1.to_out", "attn1.to_out.0")
                          .replace("attn2.to_out", "attn2.to_out.0")
                          .replace("ff.project_in.proj", "ff.net.0.proj")
                          .replace("ff.project_out", "ff.net.2"))
                value = handle.get_tensor(source)
                if packed_name.endswith(".weight") and value.dtype == torch.uint32:
                    base = source[:-len(".weight")]
                    value = _unpack_mlx_affine_int4(
                        value, handle.get_tensor(base + ".scales"), handle.get_tensor(base + ".biases"))
                weights[target] = value
        else:
            prefix = f"{PREFIX}.{layer}."
            weights = {key[len(prefix):]: handle.get_tensor(key) for key in handle.keys()
                       if key.startswith(prefix) and not any(s in key for s in ("audio", "a2v", "v2a", "av_ca"))}
    if lora is not None:
        with safe_open(str(lora), framework="pt") as handle:
            keys = set(handle.keys())
            for name in list(weights):
                if not name.endswith(".weight"):
                    continue
                base = f"diffusion_model.transformer_blocks.{layer}.{name[:-len('.weight')]}"
                if f"{base}.lora_A.weight" in keys:
                    down = handle.get_tensor(f"{base}.lora_A.weight").float()
                    up = handle.get_tensor(f"{base}.lora_B.weight").float()
                    weights[name] = (weights[name].float() + strength * (up @ down)).to(torch.bfloat16)
    return weights


def rms(value: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + EPS)
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized.to(value.dtype)


def rope(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    """split RoPE on (heads, tokens, head_dim) with cos/sin (heads, tokens, head_dim / 2)."""
    first, second = value[..., : HEAD_DIM // 2], value[..., HEAD_DIM // 2:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


INT8_GROUP = 0  # set from --int8-group; 0 keeps bf16 weights
WEIGHT_BITS = 8


class Linear(torch.nn.Module):
    """bf16 linear, or grouped symmetric int8/int4 weights expanded by Core AI at load time.

    Int4 uses Core AI's packed ``IntxTensor`` representation, so two weights occupy one byte in the asset. The
    scale remains one bf16 value per output row and input group, matching the established int8 export.
    """

    def __init__(self, weights: dict[str, torch.Tensor], name: str) -> None:
        super().__init__()
        weight = weights[f"{name}.weight"]
        self.register_buffer("bias", weights[f"{name}.bias"].contiguous())
        self.quantized = None
        if INT8_GROUP and weight.shape[1] % INT8_GROUP == 0 and weight.shape[0] >= 64:
            from coreai_torch._compression.custom_layers import WeightDequantizeModule

            rows, cols = weight.shape
            grouped = weight.float().reshape(rows, cols // INT8_GROUP, INT8_GROUP)
            maximum = (1 << (WEIGHT_BITS - 1)) - 1
            scale = (grouped.abs().amax(-1, keepdim=True) / maximum).clamp(min=1e-8)
            signed = (grouped / scale).round().clamp(-maximum, maximum).to(torch.int8).reshape(rows, cols)
            # Packed signed int4 works with Core AI's fp16 scale path. Its uint4 + zero-point equivalent does not:
            # on macOS 27.2 RC, a zero-point with more than one input group is expanded incorrectly.
            if WEIGHT_BITS == 4:
                data, zero_point, input_dtype = signed, None, torch.int4
            else:
                data, zero_point, input_dtype = signed, None, None
            scale_dtype = torch.float16 if WEIGHT_BITS == 4 else torch.bfloat16
            self.quantized = WeightDequantizeModule(
                quantized_data=data, scale=scale.reshape(rows, cols // INT8_GROUP).to(scale_dtype),
                zero_point=zero_point, input_dtype=input_dtype, output_dtype=scale_dtype)
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.contiguous())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.quantized is None else self.quantized()
        return torch.nn.functional.linear(value, weight.to(value.dtype), self.bias.to(value.dtype))


class Attention(torch.nn.Module):
    def __init__(self, weights: dict[str, torch.Tensor], name: str, kv_inputs: bool = False) -> None:
        super().__init__()
        self.q = Linear(weights, f"{name}.to_q")
        if not kv_inputs:   # with precomputed keys/values the asset carries no key/value projection
            self.k, self.v = Linear(weights, f"{name}.to_k"), Linear(weights, f"{name}.to_v")
            self.register_buffer("k_norm", weights[f"{name}.k_norm.weight"])
        self.out = Linear(weights, f"{name}.to_out.0")
        self.gate = Linear(weights, f"{name}.to_gate_logits")
        self.register_buffer("q_norm", weights[f"{name}.q_norm.weight"])

    def forward(self, value, context, cosine=None, sine=None, key=None, values=None):
        """`key` (normalised) and `values` may come precomputed, (S, HEADS * HEAD_DIM) each; then `context` is
        unused."""
        tokens = value.shape[0]
        query = rms(self.q(value), self.q_norm).reshape(tokens, HEADS, HEAD_DIM).transpose(0, 1)
        if key is None:
            key, values = rms(self.k(context), self.k_norm), self.v(context)
        key = key.reshape(-1, HEADS, HEAD_DIM).transpose(0, 1)
        values = values.reshape(-1, HEADS, HEAD_DIM).transpose(0, 1)
        if cosine is not None:
            query, key = rope(query, cosine, sine), rope(key, cosine, sine)
        attended = torch.nn.functional.scaled_dot_product_attention(query[None], key[None], values[None])[0]
        gates = 2.0 * torch.sigmoid(self.gate(value))                       # (tokens, heads)
        attended = attended.transpose(0, 1) * gates[..., None]
        return self.out(attended.reshape(tokens, HEADS * HEAD_DIM))


class Block(torch.nn.Module):
    def __init__(self, weights: dict[str, torch.Tensor], context_kv: bool = False) -> None:
        super().__init__()
        self.attn1 = Attention(weights, "attn1")
        self.attn2 = Attention(weights, "attn2", kv_inputs=context_kv)
        self.ff_in = Linear(weights, "ff.net.0.proj")
        self.ff_out = Linear(weights, "ff.net.2")
        self.register_buffer("table", weights["scale_shift_table"])
        self.register_buffer("prompt_table", weights["prompt_scale_shift_table"])

    def forward(self, x, modulation, row_select, prompt_modulation, context, cosine, sine):
        ada = (self.table.float()[None] + modulation.float()).to(x.dtype)      # (rows, 9, dim)
        per_token = (row_select @ ada.reshape(ada.shape[0], -1)).reshape(-1, ADA, DIM)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_q, scale_q, gate_q = per_token.unbind(1)
        normalized = rms(x) * (1 + scale_msa) + shift_msa
        x = x + self.attn1(normalized, normalized, cosine, sine) * gate_msa
        shift_kv, scale_kv = (self.prompt_table.to(prompt_modulation.dtype) + prompt_modulation).unbind(0)
        x = x + self.attn2(rms(x) * (1 + scale_q) + shift_q, context * (1 + scale_kv) + shift_kv) * gate_q
        hidden = torch.nn.functional.gelu(self.ff_in(rms(x) * (1 + scale_mlp) + shift_mlp), approximate="tanh")
        return x + self.ff_out(hidden) * gate_mlp

    def forward_kv(self, x, modulation, row_select, context_key, context_value, cosine, sine):
        """As `forward`, with the text cross-attention keys/values precomputed for this step (the prompt and the
        sigma schedule are fixed, so the modulated context projection is the same for every window)."""
        ada = (self.table.float()[None] + modulation.float()).to(x.dtype)
        per_token = (row_select @ ada.reshape(ada.shape[0], -1)).reshape(-1, ADA, DIM)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_q, scale_q, gate_q = per_token.unbind(1)
        normalized = rms(x) * (1 + scale_msa) + shift_msa
        x = x + self.attn1(normalized, normalized, cosine, sine) * gate_msa
        x = x + self.attn2(rms(x) * (1 + scale_q) + shift_q, None, key=context_key, values=context_value) * gate_q
        hidden = torch.nn.functional.gelu(self.ff_in(rms(x) * (1 + scale_mlp) + shift_mlp), approximate="tanh")
        return x + self.ff_out(hidden) * gate_mlp

    def context_kv(self, context, prompt_modulation):
        """The text cross-attention key (normalised) and value for one step's prompt modulation (float32)."""
        shift_kv, scale_kv = (self.prompt_table.float() + prompt_modulation.float()).unbind(0)
        modulated = context.float() * (1 + scale_kv) + shift_kv
        return rms(self.attn2.k(modulated), self.attn2.k_norm), self.attn2.v(modulated)


class Group(torch.nn.Module):
    """Adjacent blocks in one asset, plus a zero-valued weight-bound salt in the primary output so assets with the
    same graph but different weights are never deduplicated by Core AI (see export_10eros_max_h3_dit_block.py)."""

    def __init__(self, blocks: list[Block]) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x, modulation, row_select, prompt_modulation, context, cosine, sine, salt):
        # float32 at the asset boundary and bfloat16 internally. Packed int4 weights use fp16 scales because the
        # Core AI matrix kernel requires them, then Linear casts the expanded weights back to the activation dtype.
        # Keeping activations in bf16 is essential: four LTX blocks overflow when evaluated entirely in fp16.
        values = [v.to(torch.bfloat16) for v in (x, modulation, row_select, prompt_modulation, context, cosine, sine)]
        x = values[0]
        for block in self.blocks:
            x = block(x, *values[1:])
        return x.float() + salt.sum()


class KVGroup(Group):
    """Group whose blocks take precomputed text cross-attention keys/values: context_key / context_value are
    (blocks, S, DIM) float16 for the current step, in place of context and prompt_modulation."""

    def forward(self, x, modulation, row_select, context_key, context_value, cosine, sine, salt):
        x, modulation, row_select, cosine, sine = (
            v.to(torch.bfloat16) for v in (x, modulation, row_select, cosine, sine))
        keys, values = context_key.to(torch.bfloat16), context_value.to(torch.bfloat16)
        for index, block in enumerate(self.blocks):
            x = block.forward_kv(x, modulation, row_select, keys[index], values[index], cosine, sine)
        return x.float() + salt.sum()


def kv_example_inputs(tokens: int, context: int, rows: int, blocks: int, salt: int, seed: int = 0):
    x, modulation, row_select, _, _, cosine, sine, salt_value = example_inputs(
        tokens, context, rows, torch.float32, salt, seed)
    generator = torch.Generator().manual_seed(seed + 7)
    key = torch.randn(blocks, context, DIM, generator=generator).to(torch.float16)
    value = torch.randn(blocks, context, DIM, generator=generator).to(torch.float16)
    return (x, modulation, row_select, key, value, cosine, sine, salt_value)


def example_inputs(tokens: int, context: int, rows: int, dtype: torch.dtype, salt: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, DIM, generator=generator)
    modulation = torch.randn(rows, ADA, DIM, generator=generator) * 0.1
    row_select = torch.nn.functional.one_hot(torch.arange(tokens) % rows, rows).float()
    prompt_modulation = torch.randn(2, DIM, generator=generator) * 0.1
    ctx = torch.randn(context, DIM, generator=generator)
    angle = torch.rand(HEADS, tokens, HEAD_DIM // 2, generator=generator) * 6.28
    values = (x, modulation, row_select, prompt_modulation, ctx, angle.cos(), angle.sin())
    return tuple(v.to(dtype) for v in values) + (torch.zeros(salt, dtype=dtype),)


def check_against_ltx_core(weights: dict[str, torch.Tensor]) -> float:
    """Max relative error of Block vs ltx-core's BasicAVTransformerBlock on a small float32 problem."""
    from ltx_core.model.transformer.attention import AttentionOps
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
    from ltx_core.model.transformer.transformer_args import AdaLNTimestepEmbedding, TransformerArgs

    reference = BasicAVTransformerBlock(
        video=TransformerConfig(DIM, HEADS, HEAD_DIM, DIM, apply_gated_attention=True, cross_attention_adaln=True),
        rope_type=LTXRopeType.SPLIT, norm_eps=EPS, attention_ops=AttentionOps())
    reference.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
    reference = reference.eval()
    mine = Block({k: v.float() for k, v in weights.items()}).eval()
    x, modulation, row_select, prompt_modulation, context, cosine, sine, _ = example_inputs(64, 16, 2, torch.float32, 1)
    indices = row_select.argmax(-1)
    args = TransformerArgs(
        x=x[None], context=context[None], context_mask=None,
        timestep_embedding=AdaLNTimestepEmbedding(
            modulation=modulation.reshape(1, 2, ADA * DIM), embedding=torch.zeros(1, 2, DIM), token_indices=indices),
        positional_embeddings=(cosine[None], sine[None]), cross_positional_embeddings=None,
        cross_scale_shift_timestep=None, cross_gate_timestep=None, enabled=True,
        prompt_timestep=prompt_modulation.reshape(1, 1, 2 * DIM),
        self_attention_mask=None)
    with torch.no_grad():
        expected = reference(args, None)[0].x[0]
        got = mine(x, modulation, row_select, prompt_modulation, context, cosine, sine)
    return float((got - expected).abs().max() / expected.abs().max())


def export(group: torch.nn.Module, inputs, destination: Path, identity: str, max_tokens: int = 0) -> None:
    import coreai_torch

    context_kv = isinstance(group, KVGroup)
    dynamic_shapes = None
    if max_tokens:
        tokens = torch.export.Dim("tokens", min=2, max=max_tokens)
        # x, modulation, row_select, prompt_modulation|context_key, context|context_value, rope_cosine, rope_sine, salt
        dynamic_shapes = ({0: tokens}, None, {0: tokens}, None, None, {1: tokens}, {1: tokens}, None)
    exported = torch.export.export(group, inputs, dynamic_shapes=dynamic_shapes)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["x", "modulation", "row_select",
                     *(["context_key", "context_value"] if context_kv else ["prompt_modulation", "context"]),
                     "rope_cosine", "rope_sine", f"salt_{identity}"],
        output_names=["x_out"],
        entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)


def benchmark(asset: Path, inputs, reference: torch.Tensor, repeats: int, identity: str) -> dict:
    import asyncio

    from coreai.runtime import NDArray
    from lada.coreai.source_runtime import load_source_model

    runner = asyncio.Runner()
    start = time.perf_counter()
    model = load_source_model(runner, asset, purpose="ltx23 dit block")
    function = model.load_function("main")
    load_seconds = time.perf_counter() - start
    names = ["x", "modulation", "row_select", "prompt_modulation", "context", "rope_cosine", "rope_sine",
             f"salt_{identity}"]
    feeds = {name: NDArray(np.ascontiguousarray(value.numpy().astype(np.float32))) for name, value in zip(names, inputs)}

    async def call():
        result = await function(feeds)
        return result[next(iter(result))].numpy()

    times, output = [], None
    for _ in range(repeats):
        tic = time.perf_counter()
        output = runner.run(call())
        times.append(time.perf_counter() - tic)
    runner.close()
    out = torch.from_numpy(np.asarray(output, np.float32))
    error = float((out - reference).abs().max() / reference.abs().max())
    return {"loadSeconds": round(load_seconds, 2), "callSeconds": [round(t, 3) for t in times],
            "maxRelativeError": error}


def main() -> int:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output) if args.output.is_dir() else args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    global INT8_GROUP, WEIGHT_BITS
    INT8_GROUP = args.int8_group
    WEIGHT_BITS = args.weight_bits
    layers = range(args.layer, args.layer + args.layer_count)
    blocks, report = [], {"layers": list(layers), "tokens": args.tokens, "context": args.context,
                          "int8Group": args.int8_group, "weightBits": args.weight_bits,
                          "maxTokens": args.max_tokens, "contextKV": args.context_kv}
    for layer in layers:
        weights = load_weights(args.checkpoint, layer, args.lora, args.lora_strength)
        if not args.skip_reference_check and layer == args.layer:
            report["ltxCoreRelativeError"] = check_against_ltx_core(weights)
            print("ltx-core check", report["ltxCoreRelativeError"], flush=True)
        blocks.append(Block(weights, context_kv=args.context_kv).eval())
    identity = hashlib.sha256(f"{args.checkpoint.name}:{args.lora}:{args.layer}:{args.layer_count}".encode()).hexdigest()[:16]
    salt = args.layer + 1
    report["saltInput"], report["saltWidth"] = f"salt_{identity}", salt
    group = (KVGroup(blocks) if args.context_kv else Group(blocks)).eval()
    inputs = (kv_example_inputs(args.tokens, args.context, args.rows, len(blocks), salt, seed=1) if args.context_kv
              else example_inputs(args.tokens, args.context, args.rows, torch.float32, salt, seed=1))
    tic = time.perf_counter()
    export(group, inputs, args.output, identity, args.max_tokens)
    report["exportSeconds"] = round(time.perf_counter() - tic, 1)
    print("exported", args.output, report["exportSeconds"], "s", flush=True)
    if args.benchmark and not args.context_kv:
        with torch.no_grad():
            reference = group(*inputs).float()
        report["coreai"] = benchmark(args.output, inputs, reference, args.benchmark, identity)
        for count in [int(t) for t in args.bench_tokens.split(",") if t]:
            sized = example_inputs(count, args.context, args.rows, torch.float32, salt, seed=2)
            with torch.no_grad():
                expected = group(*sized).float()
            report[f"coreai_{count}"] = benchmark(args.output, sized, expected, args.benchmark, identity)
            print(count, report[f"coreai_{count}"], flush=True)
    report["assetBytes"] = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file()) if args.output.is_dir() \
        else args.output.stat().st_size
    print(json.dumps(report, indent=1))
    (args.output.parent / f"{args.output.name}.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
