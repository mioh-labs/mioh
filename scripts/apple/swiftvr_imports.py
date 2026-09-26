"""Load SwiftVR models without importing its video I/O package.

Upstream's ``swiftvr.__init__`` eagerly imports ``decord``, which has no
published macOS arm64 wheel. Model conversion needs only the transformer.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def load_transformer(source: Path):
    source = Path(source)
    for name, path in (
        ("swiftvr", source / "swiftvr"),
        ("swiftvr.models", source / "swiftvr" / "models"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    name = "swiftvr.models.transformer"
    path = source / "swiftvr" / "models" / "transformer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
