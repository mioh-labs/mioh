# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a SwinIR real-world x4 model to Core ML for ROI enhancement.

SwinIR is a Swin Transformer image restoration model. This exporter targets
the official real-world x4 checkpoints and wraps them in the same fixed-size
image contract used by Lada's other Core ML ROI enhancers.

The official SwinIR architecture is loaded from a local checkout of
https://github.com/JingyunLiang/SwinIR instead of vendoring the full model
into Lada. Its only timm dependencies are three small layer helpers; a local
compatibility module is installed when timm is unavailable so the standalone
mioh runtime remains self-contained.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

DEFAULT_MODEL = Path("model_weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth")
DEFAULT_OUTPUT_DIR = Path("model_weights")
DEFAULT_OUTPUT_NAME = "swinir-real-x4"
DEFAULT_MODEL_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
)


def _install_timm_layers_compat(torch) -> None:
    """Provide the three timm helpers used by the pinned SwinIR source.

    The Universal app intentionally does not ship the full timm package. The
    pinned official ``network_swinir.py`` imports only ``DropPath``,
    ``to_2tuple`` and ``trunc_normal_`` from it, so compatible local
    definitions avoid adding an unrelated training dependency to the app.
    Existing timm installations are left untouched.
    """

    try:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != "timm":
            raise

    def to_2tuple(value):
        if isinstance(value, tuple):
            return value
        return (value, value)

    def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
            super().__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = DropPath
    layers.to_2tuple = to_2tuple
    layers.trunc_normal_ = torch.nn.init.trunc_normal_

    models = types.ModuleType("timm.models")
    models.layers = layers
    timm = types.ModuleType("timm")
    timm.models = models
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.layers"] = layers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SwinIR real-world x4 to Core ML")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--swinir-repo-dir", type=Path, required=True, help="Local checkout of JingyunLiang/SwinIR")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--arch", choices=("medium", "large"), default="medium")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def build_swinir(repo_dir: Path, imgsz: int, scale: int, arch: str):
    import torch

    repo_dir = repo_dir.resolve()
    if not (repo_dir / "models" / "network_swinir.py").exists():
        raise FileNotFoundError(repo_dir / "models" / "network_swinir.py")
    _install_timm_layers_compat(torch)
    sys.path.insert(0, str(repo_dir))
    try:
        from models.network_swinir import SwinIR
    finally:
        try:
            sys.path.remove(str(repo_dir))
        except ValueError:
            pass

    if arch == "large":
        embed_dim = 240
        depths = [6, 6, 6, 6, 6, 6, 6, 6, 6]
        num_heads = [8, 8, 8, 8, 8, 8, 8, 8, 8]
        resi_connection = "3conv"
    else:
        embed_dim = 180
        depths = [6, 6, 6, 6, 6, 6]
        num_heads = [6, 6, 6, 6, 6, 6]
        resi_connection = "1conv"

    return SwinIR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection=resi_connection,
    )


def _load_state_dict(torch, model_path: Path) -> dict:
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in state:
                state = state[key]
                break
    return {key.removeprefix("module."): value for key, value in state.items()}


def export_model(
    model_path: Path,
    swinir_repo_dir: Path,
    output_dir: Path,
    output_name: str,
    imgsz: int,
    scale: int,
    arch: str,
    allow_overwrite: bool = False,
) -> Path:
    import coremltools as ct
    import torch

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Download the default checkpoint with:\n"
            f"  curl -L -o {DEFAULT_MODEL} {DEFAULT_MODEL_URL}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}_{imgsz}.mlpackage"
    if output_path.exists() and not allow_overwrite:
        return output_path

    net = build_swinir(swinir_repo_dir, imgsz, scale, arch)
    net.load_state_dict(_load_state_dict(torch, model_path), strict=True)
    net.eval()

    class ImageWrapper(torch.nn.Module):
        """Maps 0..1 float RGB input to a 0..255 RGB Core ML image output."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x).clamp(0.0, 1.0) * 255.0

    wrapper = ImageWrapper(net).eval()
    example = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        out = wrapper(example)
    assert out.shape[-2:] == (imgsz * scale, imgsz * scale), f"unexpected output shape: {out.shape}"
    traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="enhanced", color_layout=ct.colorlayout.RGB)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.user_defined_metadata["lada.enhancer"] = "swinir"
    mlmodel.user_defined_metadata["lada.scale"] = str(scale)
    mlmodel.user_defined_metadata["lada.imgsz"] = str(imgsz)
    mlmodel.user_defined_metadata["lada.prefer_pre_resize"] = "1"
    mlmodel.user_defined_metadata["lada.source"] = model_path.name
    mlmodel.user_defined_metadata["lada.swinir_arch"] = arch
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exported = export_model(
        args.model,
        args.swinir_repo_dir,
        args.output_dir,
        args.output_name,
        args.imgsz,
        args.scale,
        args.arch,
        args.allow_overwrite,
    )
    print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
