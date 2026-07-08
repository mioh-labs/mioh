# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a SRVGGNetCompact (Real-ESRGAN "compact") model to Core ML.

realesr-general-x4v3 is the tiny (1.2M param) compact Real-ESRGAN variant
trained for real-world degradation removal — deblur, denoise and
de-compression — which is exactly the ROI enhancer role (the same job as
jasna's proprietary unet-4x secondary restoration), but an order of
magnitude lighter than the RRDBNet x4plus model. Being small and free of
attention/pixel-unshuffle it schedules cleanly on the Neural Engine.

The architecture is vendored here (a self-contained ~40 line net) so the
export does not need the basicsr package, which pulls a heavy dependency
tree and needs a torchvision compatibility patch. The export takes a
fixed imgsz RGB image (LADA's restoration ROI size) and emits the x4
image in FP16.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = Path("model_weights/realesr-general-x4v3.pth")
DEFAULT_OUTPUT_DIR = Path("model_weights")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SRVGGNetCompact to Core ML")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num-conv", type=int, default=32, help="32 for realesr-general-x4v3, 16 for animevideov3")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def build_srvgg(scale: int, num_conv: int):
    """Vendored SRVGGNetCompact (bit-identical to basicsr.archs.srvgg_arch)."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
            super().__init__()
            self.upscale = upscale
            self.body = nn.ModuleList()
            self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(nn.PReLU(num_parameters=num_feat))
            self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            # residual over a nearest-neighbour upscale of the input
            base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
            return out + base

    class ImageWrapper(nn.Module):
        """Maps the 0..1 float output to 0..255 for a Core ML image output."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x).clamp(0.0, 1.0) * 255.0

    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=scale)
    return net, ImageWrapper(net)


def export_model(model_path: Path, output_dir: Path, imgsz: int, scale: int, num_conv: int, allow_overwrite: bool = False) -> Path:
    import coremltools as ct
    import torch

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_path.stem}_{imgsz}.mlpackage"
    if output_path.exists() and not allow_overwrite:
        return output_path

    net, wrapper = build_srvgg(scale, num_conv)
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("params_ema") or state.get("params") or state, strict=True)
    wrapper.eval()

    example = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        out = wrapper(example)
    assert out.shape[-1] == imgsz * scale, f"unexpected output scale: {out.shape} for imgsz {imgsz}"
    traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="enhanced", color_layout=ct.colorlayout.RGB)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    # lada.enhancer stays "realesrgan" so it flows through the existing
    # realesrgan validation/backend path (Core ML .mlpackage). The
    # pre-resize (enhance the 256px crop before compositing) is the
    # higher-quality path MewZoom already uses; opt this export into it.
    mlmodel.user_defined_metadata["lada.enhancer"] = "realesrgan"
    mlmodel.user_defined_metadata["lada.scale"] = str(scale)
    mlmodel.user_defined_metadata["lada.imgsz"] = str(imgsz)
    mlmodel.user_defined_metadata["lada.prefer_pre_resize"] = "1"
    mlmodel.user_defined_metadata["lada.source"] = model_path.name
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exported = export_model(args.model, args.output_dir, args.imgsz, args.scale, args.num_conv, args.allow_overwrite)
    print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
