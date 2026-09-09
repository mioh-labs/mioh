# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Lazy registration of Lada's native MPS ``grid_sample`` backward kernel."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import threading
from pathlib import Path

import torch


logger = logging.getLogger(__name__)

_LOAD_LOCK = threading.Lock()
_LOAD_ATTEMPTED = False
_LOAD_ERROR: BaseException | None = None


def has_native_mps_grid_sample_backward() -> bool:
    """Return whether ``aten::grid_sampler_2d_backward`` has an MPS kernel."""

    try:
        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::grid_sampler_2d_backward", "MPS"
            )
        )
    except (AttributeError, RuntimeError):
        return False


def _extension_name() -> str:
    version = re.match(r"\d+\.\d+", torch.__version__)
    suffix = version.group(0).replace(".", "_") if version else "unknown"
    return f"lada_mps_grid_sample_backward_{suffix}"


def _is_disabled() -> bool:
    value = os.environ.get("LADA_MPS_GRID_SAMPLE_BACKWARD", "1")
    return value.strip().lower() in {"0", "false", "no", "off"}


def _load_without_ninja(source: Path) -> None:
    """Build the registration library with setuptools when Ninja is absent."""

    import fcntl

    from setuptools import Distribution
    from setuptools.command.build_ext import build_ext
    from torch.utils.cpp_extension import CppExtension, get_default_build_root

    class ObjCppBuildExt(build_ext):
        def build_extension(self, extension):
            if ".mm" not in self.compiler.src_extensions:
                self.compiler.src_extensions.append(".mm")
            original_compile = self.compiler._compile

            def compile_source(
                obj, source_path, extension_suffix, compiler_args, extra_args, options
            ):
                if source_path.endswith(".mm"):
                    self.compiler.compiler_so = [
                        "clang++",
                        *self.compiler.compiler_so[1:],
                    ]
                return original_compile(
                    obj,
                    source_path,
                    extension_suffix,
                    compiler_args,
                    extra_args,
                    options,
                )

            self.compiler._compile = compile_source
            super().build_extension(extension)

    fingerprint = hashlib.sha256()
    fingerprint.update(source.read_bytes())
    fingerprint.update(torch.__version__.encode())
    fingerprint.update(sys.implementation.cache_tag.encode())
    digest = fingerprint.hexdigest()[:12]
    name = f"{_extension_name()}_{digest}"
    build_root = Path(
        os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root())
    ) / name
    build_root.mkdir(parents=True, exist_ok=True)

    extension = CppExtension(
        name=name,
        sources=[str(source)],
        extra_compile_args=["-std=c++20", "-O3"],
        extra_link_args=[
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
        ],
    )
    distribution = Distribution(
        {
            "name": name,
            "ext_modules": [extension],
            "cmdclass": {"build_ext": ObjCppBuildExt},
        }
    )
    distribution.script_name = "setup.py"
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(build_root)
    command.build_temp = str(build_root / "temp")
    command.inplace = False

    lock_path = build_root / "build.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        library_path = Path(command.get_ext_fullpath(name))
        if not library_path.exists():
            distribution.run_command("build_ext")
            library_path = Path(command.get_ext_fullpath(name))
        torch.ops.load_library(str(library_path.resolve()))


def enable_native_mps_grid_sample_backward(*, raise_on_error: bool = False) -> bool:
    """Build and load the missing MPS dispatch once, returning its availability.

    PyTorch caches the compiled extension, so only the first training process
    pays the compilation cost.  Inference never calls this function because no
    grid-sample input requires a gradient.
    """

    global _LOAD_ATTEMPTED, _LOAD_ERROR

    if has_native_mps_grid_sample_backward():
        return True
    if _is_disabled() or sys.platform != "darwin" or not torch.backends.mps.is_built():
        return False

    with _LOAD_LOCK:
        if has_native_mps_grid_sample_backward():
            return True
        if _LOAD_ATTEMPTED:
            if raise_on_error and _LOAD_ERROR is not None:
                raise RuntimeError(
                    "The native MPS grid_sample backward extension failed to load"
                ) from _LOAD_ERROR
            return False

        _LOAD_ATTEMPTED = True
        source = Path(__file__).with_name("csrc") / "mps_grid_sample_backward.mm"
        try:
            from torch.utils.cpp_extension import is_ninja_available, load

            if is_ninja_available():
                load(
                    name=_extension_name(),
                    sources=[str(source)],
                    extra_cflags=["-std=c++20", "-O3"],
                    extra_ldflags=[
                        "-framework",
                        "Metal",
                        "-framework",
                        "Foundation",
                    ],
                    is_python_module=False,
                    verbose=os.environ.get("LADA_MPS_EXTENSION_VERBOSE", "0")
                    == "1",
                )
            else:
                logger.info(
                    "Ninja is unavailable; building MPS grid_sample backward "
                    "with setuptools"
                )
                _load_without_ninja(source)
            if not has_native_mps_grid_sample_backward():
                raise RuntimeError(
                    "extension loaded without registering the MPS dispatch"
                )
            logger.info("Enabled native MPS grid_sample backward")
            return True
        except Exception as error:
            _LOAD_ERROR = error
            if raise_on_error:
                raise RuntimeError(
                    "Could not build the native MPS grid_sample backward extension"
                ) from error
            logger.warning(
                "Native MPS grid_sample backward is unavailable: %s", error
            )
            return False


def native_mps_grid_sample_backward_error() -> BaseException | None:
    """Return the cached extension-load error, if loading was attempted."""

    return _LOAD_ERROR
