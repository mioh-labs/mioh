# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Shared loading behavior for portable Core AI source models."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import time
from pathlib import Path
from typing import Any


def _model_identity(model_path: Path) -> str:
    hasher = hashlib.sha256()
    hash_file = model_path / "main.hash"
    if hash_file.is_file():
        hasher.update(hash_file.read_bytes())
    else:
        hasher.update(str(model_path.resolve()).encode("utf-8"))
    return hasher.hexdigest()


def _model_load_lock_path(model_path: Path) -> Path:
    configured = os.environ.get("LADA_COREAI_MODEL_LOCK_DIR")
    directory = (
        Path(configured).expanduser()
        if configured
        else Path.home() / "Library" / "Caches" / "mioh" / "coreai-model-locks"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_model_identity(model_path)}.lock"


def _is_concurrent_install_collision(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        isinstance(exc, FileExistsError)
        or "already exists" in message
        or "same name already exists" in message
        or "file exists" in message
    )


def load_source_model(
    runner: asyncio.Runner,
    model_path: Path,
    *,
    purpose: str,
) -> Any:
    try:
        from coreai.runtime import AIModel
    except ImportError as exc:
        raise RuntimeError(
            f"Core AI {purpose} requires the isolated coreai-torch environment"
        ) from exc

    print(
        "Core AIモデルを準備中（初回はこのMac向けに最適化します）: "
        f"{model_path.name}",
        flush=True,
    )
    lock_path = _model_load_lock_path(model_path)
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"Core AIモデルの準備待ち（別ワーカーが処理中）: {model_path.name}",
                flush=True,
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        for attempt in range(3):
            try:
                model = runner.run(AIModel.load(model_path))
                break
            except Exception as exc:
                if not _is_concurrent_install_collision(exc) or attempt == 2:
                    raise RuntimeError(
                        f"Core AI {purpose}モデルの読み込みに失敗しました "
                        f"({model_path.name}): {exc}"
                    ) from exc
                time.sleep(0.25 * (attempt + 1))
    print(f"Core AIモデルの準備完了: {model_path.name}", flush=True)
    return model
