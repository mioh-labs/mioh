## Developer Installation (macOS)
This section provides instructions for installing Lada from source on macOS: the **CLI** (lada-cli) and optionally the **GUI** (GTK 4 + Libadwaita).

> [!NOTE]
> This is for macOS. If you're on Linux, follow the [Linux Installation](linux_install.md). If you're on Windows, follow the [Windows Installation](windows_install.md).
>
> For a standalone CLI and GUI build with PyInstaller, see [packaging/macOS/README.md](../packaging/macOS/README.md).

### Install CLI

1) Install system dependencies

   * [uv](https://docs.astral.sh/uv/getting-started/installation/) (e.g. `brew install uv`)
   * FFmpeg (e.g. `brew install ffmpeg`)
   * git (e.g. `brew install git`)

   ```bash
   brew install ffmpeg git uv
   ```

2) Get the source code

   ```bash
   git clone https://codeberg.org/ladaapp/lada.git
   cd lada
   ```

3) Create a virtual environment and install Python dependencies

   On macOS, use the **cpu** extra. PyTorch's default CPU build includes **MPS (Metal)** support, so you can use Apple Silicon or Intel Mac GPUs when available.

   ```bash
   uv venv
   source .venv/bin/activate
   uv sync --extra cpu
   ```

   Check that PyTorch sees your GPU (optional; skip if you only want CPU):

   ```bash
   # MPS (Metal) - Apple Silicon or Intel Mac with supported GPU
   uv run --no-project python -c "import torch; print(torch.backends.mps.is_available())"
   ```

   If this prints *True*, you can pass `--device mps` to the CLI for GPU-accelerated inference.

4) Apply patches

   ```bash
   patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/increase_mms_time_limit.patch
   patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/remove_ultralytics_telemetry.patch
   patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
   ```

5) Download model weights

   Download the necessary model weights from HuggingFace:

   ```bash
   curl -L -o model_weights/lada_mosaic_detection_model_v2.pt 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true'
   curl -L -o model_weights/lada_mosaic_detection_model_v4_accurate.pt 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true'
   curl -L -o model_weights/lada_mosaic_detection_model_v4_fast.pt 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true'
   curl -L -o model_weights/lada_mosaic_restoration_model_generic_v1.2.pth 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true'
   ```

   For DeepMosaics restoration you can also download their pretrained model:

   ```bash
   curl -L -o model_weights/3rd_party/clean_youknow_video.pth 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t'
   ```

You can now run the CLI with `lada-cli`.

> [!TIP]
> Remember: To start Lada ensure you:
> * `cd` into the project root directory
> * Activate the virtual environment with `source .venv/bin/activate`
> * Run the CLI with `lada-cli` (use `--device mps` for Metal GPU if available)

### Install GUI

The GUI uses **GTK 4** and **Libadwaita**. On macOS you need to install these and GStreamer via Homebrew, then install the Python GUI dependencies.

1) Complete the [CLI install](#install-cli) above (venv, patches, model weights).

2) Install GTK 4, Libadwaita, and GStreamer (required for video playback in the GUI):

   ```bash
   brew install gtk4 libadwaita adwaita-icon-theme gstreamer
   ```

   Without `adwaita-icon-theme`, some icons may not appear.

3) Install Python GUI dependencies (in the same venv):

   ```bash
   uv sync --extra gui
   ```

   If you already ran `uv sync --extra cpu`, this adds the `gui` extra (pycairo, PyGObject). To have both CPU/MPS and GUI: `uv sync --extra cpu --extra gui`.

4) Run the GUI:

   ```bash
   lada
   ```

   Or from the project root with venv active:

   ```bash
   uv run lada
   ```

### Install Translations (optional)

If you prefer the app in a language other than English:

1) Install gettext

   ```bash
   brew install gettext
   ```

2) Compile translations

   ```bash
   bash translations/compile_po.sh
   ```

The CLI will use translations based on your locale (e.g. `LANG` or `LANGUAGE`).

### Core ML Detection on Apple Neural Engine (optional)

On Apple Silicon you can run mosaic detection through Core ML instead of
PyTorch/MPS. Detection then executes on the Neural Engine / GPU as chosen
by Core ML, stays off the MPS command queue used by restoration, and is
significantly faster per frame.

1) Export the detection model (one-time). Install the Core ML extra
   first — it covers all Core ML export and runtime dependencies
   (`coremltools`, `huggingface_hub`, `safetensors`):

   ```bash
   pip install -e ".[apple-coreml]"   # or: uv sync --extra cpu --extra apple-coreml
   python scripts/apple/export_v4_fast_coreml.py --output-dir model_weights
   ```

   For the PyTorch (.pth) ROI enhancer backend, add the
   `roi-enhancer-torch` extra and run `python apply_lada_patches.py`
   afterwards.

2) Run with the Core ML detector:

   ```bash
   lada-cli --input <video> --mosaic-detection-model v4-fast-coreml
   ```

   The name `v4-fast-coreml` resolves to
   `model_weights/lada_mosaic_detection_model_v4_fast.mlpackage`. You can
   also pass a path to any exported `.mlpackage` directly.

Parity against the PyTorch detector can be checked with
`scripts/apple/validate_v4_fast_coreml.py`; see
`docs/apple/v4-fast-coreml-postprocess.md` for the postprocess contract.

Detection runs pinned to CPU+Neural Engine (`LADA_COREML_COMPUTE_UNITS`
overrides this, e.g. `ALL`). Note that the ANE duty cycle is small — a few
milliseconds per frame at restoration-bound throughput — so monitoring
tools like asitop will show ANE utilization near zero even though
detection is running there.

### Real-ESRGAN ROI Enhancer on the Neural Engine (optional)

The optional ROI enhancer (`--restore-roi-enhancer realesrgan`) can also
run through Core ML. Export once:

```bash
python scripts/apple/export_realesrgan_coreml.py \
  --model model_weights/RealESRGAN_x4plus.pth --scale 4
```

Then pass the `.mlpackage` as the enhancer model:

```bash
lada-cli --input <video> --mosaic-detection-model v4-fast-coreml \
  --restore-roi-enhancer realesrgan \
  --restore-roi-enhancer-model-path model_weights/RealESRGAN_x4plus_256.mlpackage \
  --restore-roi-enhancer-scale 4 --restore-roi-enhancer-strength 0.25
```

Measured on this machine: ~300 ms per enhanced frame on the Neural
Engine vs ~1.8 s (fp16 ~1.6 s) through PyTorch/MPS, with PSNR 57 dB
against the fp32 output. The PyTorch path is still used when the model
path ends in .pth.

The compact `realesr-general-x4v3` model can also be exported to the new
Core AI format on macOS 27:

```bash
.venv-coreai/bin/python scripts/apple/export_srvgg_coreai.py \
  --allow-overwrite
```

Select it explicitly with `realesr-general-x4v3-coreai`:

```bash
python process_video_parallel.py --input <video> --output <output> \
  --restore-roi-enhancer realesrgan \
  --restore-roi-enhancer-model-path realesr-general-x4v3-coreai \
  --restore-roi-enhancer-scale 4 --restore-roi-enhancer-strength 0.25
```

On this M5 Pro, replacing PReLU with an equivalent Core AI-friendly expression
gave about 48 ms per 256px ROI on the local Core AI GPU runtime, while the
existing Core ML/Neural Engine export measured about 24 ms. Core AI with an
explicit Neural Engine specialization reached about 17 ms, but enabling the OS
Core AI runtime also slowed the custom-Metal BasicVSR++ model. Therefore the
Core AI SRVGG backend remains an explicit comparison option; Core ML SRVGG is
recommended when using Core AI BasicVSR++.

The larger RRDB-based `RealESRGAN_x4plus` model has the same experimental Core
AI path:

```bash
.venv-coreai/bin/python scripts/apple/export_realesrgan_coreai.py \
  --allow-overwrite
```

Its registered name is `realesrgan-x4-coreai`. On this M5 Pro it measured about
427 ms per 256px ROI on the local Core AI GPU runtime and 146 ms with an explicit
Neural Engine specialization. The existing Core ML/Neural Engine export took
about 126 ms, so `realesrgan-x4-coreml` remains the recommended x4plus backend.
Core AI and Core ML outputs measured 52.83 dB PSNR on the deterministic test
image.

An alternative enhancer is MewZoom (Apache-2.0, UNet-based, trained for
blur/noise/compression artifact removal — the open counterpart of
jasna's unet-4x):

```bash
python scripts/apple/export_mewzoom_coreml.py
# Optional higher-resolution input variant. Slower, but can preserve more
# detail before LADA resizes the restored ROI back onto the source frame.
python scripts/apple/export_mewzoom_coreml.py --imgsz 512
```

The MewZoom architecture is vendored in `lada/models/mewzoom` — do not
`pip install mewzoom`, its `torch~=2.9` pin would downgrade PyTorch.

Then pass `model_weights/MewZoom-V1-4X-Unet_256.mlpackage` as
`--restore-roi-enhancer-model-path`, or use the registered
`mewzoom-x4-coreml-512` name after exporting the 512px variant.
Measured ~100 ms/frame on the Neural Engine, faster than Real-ESRGAN (~300 ms). The ANE compiler
crashes on this network's decoder as-is (conv emitting 1536 channels
into pixel_shuffle); the export script works around it by splitting the
shuffle into channel chunks, which is bit-identical. Visually MewZoom
denoises more while Real-ESRGAN sharpens more — pick per source.

The enhancer can also be selected by name alone; the model path defaults
to the `<enhancer>-x4-coreml` export in `model_weights/`:

```bash
lada-cli --input <video> --mosaic-detection-model v4-fast-coreml \
  --restore-roi-enhancer mewzoom \
  --restore-roi-enhancer-scale 4 --restore-roi-enhancer-strength 0.25
```

MewZoom Core ML enhancer output is applied before the restored crop is resized
back to the original ROI, so the 4x output has a chance to survive the final
downscale/composite step. Try `mewzoom-x4-coreml-512` when the default 256px
export looks too subtle on larger ROIs.

### SwinIR ROI Enhancer on the Neural Engine (optional)

SwinIR can also be exported as a Core ML ROI enhancer. It is a heavier
Transformer-based restoration model, so treat it as an experimental quality
comparison against MewZoom and Real-ESRGAN/SRVGG rather than a default fast
path.

Install the Core ML extra and the `timm` dependency used by the official
SwinIR architecture:

```bash
uv sync --extra cpu --extra apple-coreml
uv pip install timm
```

Clone the official SwinIR repository once, then export the real-world x4
checkpoint:

```bash
git clone https://github.com/JingyunLiang/SwinIR.git vendor/SwinIR
curl -L -o model_weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
  https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
python scripts/apple/export_swinir_coreml.py \
  --swinir-repo-dir vendor/SwinIR \
  --model model_weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
```

The exported model is registered as both `swinir-x4-coreml` and
`swinir-real-x4-coreml`:

```bash
lada-cli --input <video> --mosaic-detection-model v4-fast-coreml \
  --restore-roi-enhancer swinir \
  --restore-roi-enhancer-model-path swinir-x4-coreml \
  --restore-roi-enhancer-scale 4 --restore-roi-enhancer-strength 0.25
```

Like MewZoom and SRVGG, the export opts into pre-resize enhancement using
`lada.prefer_pre_resize=1`, so the enhancer runs on the fixed restored crop
before LADA resizes it back onto the original ROI.
