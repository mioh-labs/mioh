"""Export BasicVSR++ deform alignment weights for MLX experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


MODULES = ("backward_1", "forward_1", "backward_2", "forward_2")
TENSOR_SUFFIXES = (
    "weight",
    "bias",
    "conv_offset.0.weight",
    "conv_offset.0.bias",
    "conv_offset.2.weight",
    "conv_offset.2.bias",
    "conv_offset.4.weight",
    "conv_offset.4.bias",
    "conv_offset.6.weight",
    "conv_offset.6.bias",
)
NUM_BACKBONE_BLOCKS = 15
NUM_FEATURE_EXTRACT_BLOCKS = 5
BACKBONE_TENSOR_SUFFIXES = (
    "main.0.weight",
    "main.0.bias",
    *(
        item
        for block_index in range(NUM_BACKBONE_BLOCKS)
        for item in (
            f"main.2.{block_index}.conv1.weight",
            f"main.2.{block_index}.conv1.bias",
            f"main.2.{block_index}.conv2.weight",
            f"main.2.{block_index}.conv2.bias",
        )
    ),
)
FEATURE_EXTRACT_TENSOR_SUFFIXES = (
    "0.weight",
    "0.bias",
    "2.weight",
    "2.bias",
    "4.main.0.weight",
    "4.main.0.bias",
    *(
        item
        for block_index in range(NUM_FEATURE_EXTRACT_BLOCKS)
        for item in (
            f"4.main.2.{block_index}.conv1.weight",
            f"4.main.2.{block_index}.conv1.bias",
            f"4.main.2.{block_index}.conv2.weight",
            f"4.main.2.{block_index}.conv2.bias",
        )
    ),
)
RECONSTRUCTION_TENSOR_SUFFIXES = (
    "reconstruction.main.0.weight",
    "reconstruction.main.0.bias",
    *(
        item
        for block_index in range(NUM_FEATURE_EXTRACT_BLOCKS)
        for item in (
            f"reconstruction.main.2.{block_index}.conv1.weight",
            f"reconstruction.main.2.{block_index}.conv1.bias",
            f"reconstruction.main.2.{block_index}.conv2.weight",
            f"reconstruction.main.2.{block_index}.conv2.bias",
        )
    ),
    "upsample1.upsample_conv.weight",
    "upsample1.upsample_conv.bias",
    "upsample2.upsample_conv.weight",
    "upsample2.upsample_conv.bias",
    "conv_hr.weight",
    "conv_hr.bias",
    "conv_last.weight",
    "conv_last.bias",
)
SPYNET_TENSOR_SUFFIXES = (
    "mean",
    "std",
    *(
        item
        for pyramid_level in range(6)
        for layer in range(5)
        for item in (
            f"basic_module.{pyramid_level}.basic_module.{layer}.conv.weight",
            f"basic_module.{pyramid_level}.basic_module.{layer}.conv.bias",
        )
    ),
)


def export_deform_alignment_weights(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    prefix: str = "generator_ema",
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state_dict(checkpoint_path)
    manifest: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "prefix": prefix,
        "modules": {},
        "backbones": {},
        "feature_extract": {},
        "reconstruction": {},
        "spynet": {},
    }
    modules: dict[str, object] = {}
    backbones: dict[str, object] = {}
    feature_tensors = _feature_extract_tensors(state, prefix=prefix)
    if feature_tensors:
        npz_name = f"{prefix}.feat_extract.npz"
        np.savez(output_dir / npz_name, **feature_tensors)
        first_weight = feature_tensors["0.weight"]
        manifest["feature_extract"] = {
            "npz": npz_name,
            "input_channels": int(first_weight.shape[1]),
            "mid_channels": int(first_weight.shape[0]),
            "num_blocks": NUM_FEATURE_EXTRACT_BLOCKS,
            "tensors": {name: list(value.shape) for name, value in feature_tensors.items()},
        }
    reconstruction_tensors = _reconstruction_tensors(state, prefix=prefix)
    if reconstruction_tensors:
        npz_name = f"{prefix}.reconstruction.npz"
        np.savez(output_dir / npz_name, **reconstruction_tensors)
        first_weight = reconstruction_tensors["reconstruction.main.0.weight"]
        manifest["reconstruction"] = {
            "npz": npz_name,
            "input_channels": int(first_weight.shape[1]),
            "mid_channels": int(first_weight.shape[0]),
            "num_blocks": NUM_FEATURE_EXTRACT_BLOCKS,
            "tensors": {name: list(value.shape) for name, value in reconstruction_tensors.items()},
        }
    spynet_tensors = _spynet_tensors(state, prefix=prefix)
    if spynet_tensors:
        npz_name = f"{prefix}.spynet.npz"
        np.savez(output_dir / npz_name, **spynet_tensors)
        manifest["spynet"] = {
            "npz": npz_name,
            "num_modules": 6,
            "tensors": {name: list(value.shape) for name, value in spynet_tensors.items()},
        }
    for module in MODULES:
        tensors = _module_tensors(state, prefix=prefix, module=module)
        if not tensors:
            pass
        else:
            npz_name = f"{prefix}.deform_align.{module}.npz"
            np.savez(output_dir / npz_name, **tensors)
            modules[module] = {
                "npz": npz_name,
                "weight_shape": list(tensors["weight"].shape),
                "bias_shape": list(tensors["bias"].shape),
                "offset_out_channels": int(tensors["conv_offset.6.weight"].shape[0]),
                "deform_groups": int(tensors["conv_offset.6.weight"].shape[0] // 27),
                "tensors": {name: list(value.shape) for name, value in tensors.items()},
            }

        backbone_tensors = _backbone_tensors(state, prefix=prefix, module=module)
        if backbone_tensors:
            npz_name = f"{prefix}.backbone.{module}.npz"
            np.savez(output_dir / npz_name, **backbone_tensors)
            input_weight = backbone_tensors["main.0.weight"]
            backbones[module] = {
                "npz": npz_name,
                "input_channels": int(input_weight.shape[1]),
                "mid_channels": int(input_weight.shape[0]),
                "num_blocks": NUM_BACKBONE_BLOCKS,
                "tensors": {name: list(value.shape) for name, value in backbone_tensors.items()},
            }

    manifest["modules"] = modules
    manifest["backbones"] = backbones
    manifest_path = output_dir / "deform_alignment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "params_ema", "params"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _module_tensors(
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
    module: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    base = f"{prefix}.deform_align.{module}"
    for suffix in TENSOR_SUFFIXES:
        key = f"{base}.{suffix}"
        tensor = state.get(key)
        if tensor is None:
            return {}
        output[suffix] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return output


def _backbone_tensors(
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
    module: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    base = f"{prefix}.backbone.{module}"
    for suffix in BACKBONE_TENSOR_SUFFIXES:
        key = f"{base}.{suffix}"
        tensor = state.get(key)
        if tensor is None:
            return {}
        output[suffix] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return output


def _feature_extract_tensors(
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    base = f"{prefix}.feat_extract"
    for suffix in FEATURE_EXTRACT_TENSOR_SUFFIXES:
        key = f"{base}.{suffix}"
        tensor = state.get(key)
        if tensor is None:
            return {}
        output[suffix] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return output


def _reconstruction_tensors(
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for suffix in RECONSTRUCTION_TENSOR_SUFFIXES:
        key = f"{prefix}.{suffix}"
        tensor = state.get(key)
        if tensor is None:
            return {}
        output[suffix] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return output


def _spynet_tensors(
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    base = f"{prefix}.spynet"
    for suffix in SPYNET_TENSOR_SUFFIXES:
        key = f"{base}.{suffix}"
        tensor = state.get(key)
        if tensor is None:
            return {}
        output[suffix] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="generator_ema")
    args = parser.parse_args()

    manifest_path = export_deform_alignment_weights(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
