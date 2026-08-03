#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Prepare a fresh BasicVSR++ fine-tuning checkpoint from EMA weights.

MMEngine GAN checkpoints contain both ``generator`` (the weights updated by the
optimizer) and ``generator_ema`` (the weights used for inference).  Loading such
a checkpoint normally and starting a new run trains ``generator``, even when the
deployed reference model came from ``generator_ema``.  This tool makes that
choice explicit by copying the EMA state into both generator branches while
discarding all resume-only state.

The source checkpoint is deserialized with ``weights_only=False`` because an
MMEngine checkpoint may contain trusted Python metadata.  Consequently the CLI
requires an explicit ``--trust-checkpoint`` acknowledgement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = 1
RAW_PREFIX = "generator."
EMA_PREFIX = "generator_ema."
PROVENANCE_KEY = "mioh_finetune_initialization"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prefixed_state(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


def _validate_generator_pairs(state_dict: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _prefixed_state(state_dict, RAW_PREFIX)
    ema = _prefixed_state(state_dict, EMA_PREFIX)
    if not raw:
        raise ValueError(f"source checkpoint has no {RAW_PREFIX!r} state")
    if not ema:
        raise ValueError(f"source checkpoint has no {EMA_PREFIX!r} state")

    raw_suffixes = set(raw)
    ema_suffixes = set(ema)
    if raw_suffixes != ema_suffixes:
        raw_only = sorted(raw_suffixes - ema_suffixes)
        ema_only = sorted(ema_suffixes - raw_suffixes)
        raise ValueError(
            "generator and generator_ema suffix sets differ: "
            f"raw_only={raw_only[:8]!r}, ema_only={ema_only[:8]!r}"
        )

    for suffix in sorted(raw_suffixes):
        raw_shape = _shape(raw[suffix])
        ema_shape = _shape(ema[suffix])
        if raw_shape != ema_shape:
            raise ValueError(
                f"shape mismatch for {suffix!r}: "
                f"generator={raw_shape}, generator_ema={ema_shape}"
            )
        if not isinstance(ema[suffix], torch.Tensor):
            raise TypeError(
                f"EMA state {EMA_PREFIX + suffix!r} is not a tensor: "
                f"{type(ema[suffix]).__name__}"
            )

    return tuple(sorted(raw_suffixes))


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _reset_step_counters(state_dict: Mapping[str, Any]) -> list[str]:
    reset = []
    for key, value in state_dict.items():
        if key == "step_counter" or key.endswith(".step_counter"):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"step counter {key!r} is not a tensor: "
                    f"{type(value).__name__}"
                )
            value.zero_()
            reset.append(key)
    return reset


def _source_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    meta = checkpoint.get("meta")
    if not isinstance(meta, Mapping):
        return {}

    result = {}
    for key in ("iter", "epoch", "experiment_name", "time", "mmengine_version"):
        value = meta.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _atomic_torch_save(payload: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The file replacement is already atomic. Directory fsync is an
            # additional durability measure that is unavailable on some FSes.
            pass
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def prepare_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    trust_checkpoint: bool,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a model-only initialization checkpoint from a trusted source."""

    if not trust_checkpoint:
        raise PermissionError(
            "refusing to deserialize with weights_only=False without "
            "trust_checkpoint=True (CLI: --trust-checkpoint)"
        )

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("source and output checkpoint paths must be different")
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output checkpoint already exists: {output_path}; use --overwrite"
        )

    source_sha256 = _sha256(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"checkpoint root must be a mapping, got {type(checkpoint).__name__}"
        )
    source_state = checkpoint.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise TypeError("checkpoint must contain a mapping named 'state_dict'")

    suffixes = _validate_generator_pairs(source_state)
    output_state = copy.deepcopy(source_state)
    for suffix in suffixes:
        ema_value = source_state[EMA_PREFIX + suffix]
        output_state[RAW_PREFIX + suffix] = _clone_tensor(ema_value)
        output_state[EMA_PREFIX + suffix] = _clone_tensor(ema_value)

    reset_step_counters = _reset_step_counters(output_state)
    provenance = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_path),
        "source_sha256": source_sha256,
        "source_metadata": _source_metadata(checkpoint),
        "source_prefix": EMA_PREFIX,
        "destination_prefixes": [RAW_PREFIX, EMA_PREFIX],
        "generator_suffix_count": len(suffixes),
        "state_entry_count": len(output_state),
        "reset_step_counters": reset_step_counters,
        "optimizer_state_preserved": False,
        "message_hub_preserved": False,
    }
    output_checkpoint = {
        "meta": {PROVENANCE_KEY: provenance},
        "state_dict": output_state,
    }
    _atomic_torch_save(output_checkpoint, output_path)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a fresh BasicVSR++ fine-tuning checkpoint whose raw and "
            "EMA generators both start from the source EMA weights."
        )
    )
    parser.add_argument("source", type=Path, help="trusted MMEngine checkpoint")
    parser.add_argument("output", type=Path, help="new initialization checkpoint")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="allow trusted local checkpoint deserialization with weights_only=False",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output file"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provenance = prepare_checkpoint(
        args.source,
        args.output,
        trust_checkpoint=args.trust_checkpoint,
        overwrite=args.overwrite,
    )
    print(f"Prepared: {args.output.expanduser().resolve()}")
    print(f"Source SHA-256: {provenance['source_sha256']}")
    print(f"Generator entries copied: {provenance['generator_suffix_count']}")
    print("Fresh start: optimizer/message-hub/runner iteration state omitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
