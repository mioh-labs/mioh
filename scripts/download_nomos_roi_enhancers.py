# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Download and verify the three optional Nomos CLI ROI enhancer models."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


MODELS = {
    "4xNomosWebPhoto_RealPLKSR.safetensors": "9be0228f98156a100d6636d99b373ed2785b999723f9adc4cca504329ab157f2",
    "4xNomosUni_span_multijpg.safetensors": "3bedff643a1ba51b12e0174ebca62649a930ae3e7b0868be9706d8659d4d32a2",
    "2xNomosUni_compact_multijpg.safetensors": "ea51bc7aa05c801e42c85aa84007b8bb1dcd3f11d85d5979b26b29e4c8610401",
}
RELEASE_BASE = "https://github.com/Phhofm/models/releases/download"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output_dir: Path, force: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename, expected_hash in MODELS.items():
        path = output_dir / filename
        if path.exists() and not force and sha256(path) == expected_hash:
            downloaded.append(path)
            continue
        tag = filename.removesuffix(".safetensors")
        url = f"{RELEASE_BASE}/{tag}/{filename}"
        temporary = path.with_suffix(path.suffix + ".part")
        with urlopen(url) as response, temporary.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        actual_hash = sha256(temporary)
        if actual_hash != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {filename}: {actual_hash}")
        temporary.replace(path)
        downloaded.append(path)
    return downloaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("model_weights"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for path in download(args.output_dir, args.force):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
