# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Shared loading behavior for portable Core AI source models."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


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
    try:
        model = runner.run(AIModel.load(model_path))
    except Exception as exc:
        raise RuntimeError(
            f"Core AI {purpose}モデルの読み込みに失敗しました "
            f"({model_path.name}): {exc}"
        ) from exc
    print(f"Core AIモデルの準備完了: {model_path.name}", flush=True)
    return model
