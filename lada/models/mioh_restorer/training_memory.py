# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Memory monitoring and cleanup helpers for long MiohRestorer training runs."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Literal

import psutil
import torch


GIB = 1024**3
MemoryStatus = Literal["normal", "warning", "critical"]


@dataclass(frozen=True)
class MemoryThresholds:
    """Pressure limits checked at safe boundaries between training steps."""

    warning_mps_ratio: float = 0.80
    critical_mps_ratio: float = 0.92
    warning_system_available_gib: float = 8.0
    critical_system_available_gib: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 < self.warning_mps_ratio < self.critical_mps_ratio <= 1.0:
            raise ValueError(
                "MPS memory ratios must satisfy 0 < warning < critical <= 1"
            )
        if not (
            0.0
            < self.critical_system_available_gib
            < self.warning_system_available_gib
        ):
            raise ValueError(
                "system memory limits must satisfy 0 < critical < warning"
            )


@dataclass(frozen=True)
class MemorySnapshot:
    """One process-local MPS and system-memory sample."""

    stage: str
    mps_current_bytes: int | None
    mps_driver_bytes: int | None
    mps_recommended_bytes: int | None
    system_available_bytes: int
    system_total_bytes: int
    swap_used_bytes: int

    @property
    def mps_pressure_ratio(self) -> float | None:
        if not self.mps_recommended_bytes:
            return None
        used = self.mps_driver_bytes
        if used is None:
            used = self.mps_current_bytes
        if used is None:
            return None
        return max(0.0, used / self.mps_recommended_bytes)

    @property
    def system_available_gib(self) -> float:
        return self.system_available_bytes / GIB

    @property
    def mps_reclaimable_bytes(self) -> int:
        if self.mps_driver_bytes is None or self.mps_current_bytes is None:
            return 0
        return max(0, self.mps_driver_bytes - self.mps_current_bytes)

    @property
    def effective_system_available_gib(self) -> float:
        """Available memory after discounting the process's reusable MPS cache."""

        return (self.system_available_bytes + self.mps_reclaimable_bytes) / GIB

    def as_record(self) -> dict[str, float | str | None]:
        def to_gib(value: int | None) -> float | None:
            return None if value is None else round(value / GIB, 3)

        return {
            "stage": self.stage,
            "mps_current_gib": to_gib(self.mps_current_bytes),
            "mps_driver_gib": to_gib(self.mps_driver_bytes),
            "mps_recommended_gib": to_gib(self.mps_recommended_bytes),
            "mps_pressure_ratio": (
                None
                if self.mps_pressure_ratio is None
                else round(self.mps_pressure_ratio, 4)
            ),
            "mps_reclaimable_gib": round(self.mps_reclaimable_bytes / GIB, 3),
            "system_available_gib": round(self.system_available_gib, 3),
            "effective_system_available_gib": round(
                self.effective_system_available_gib, 3
            ),
            "system_total_gib": round(self.system_total_bytes / GIB, 3),
            "swap_used_gib": round(self.swap_used_bytes / GIB, 3),
        }


class TrainingMemoryGuard:
    """Observe memory without flushing useful caches during normal training."""

    def __init__(
        self,
        device: torch.device,
        thresholds: MemoryThresholds,
    ) -> None:
        self.device = device
        self.thresholds = thresholds

    def capture(
        self,
        stage: str,
        *,
        synchronize: bool = False,
    ) -> MemorySnapshot:
        current: int | None = None
        driver: int | None = None
        recommended: int | None = None
        if self.device.type == "mps" and torch.backends.mps.is_available():
            try:
                if synchronize and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                if hasattr(torch.mps, "current_allocated_memory"):
                    current = int(torch.mps.current_allocated_memory())
                if hasattr(torch.mps, "driver_allocated_memory"):
                    driver = int(torch.mps.driver_allocated_memory())
                if hasattr(torch.mps, "recommended_max_memory"):
                    recommended = int(torch.mps.recommended_max_memory())
            except RuntimeError:
                # System memory still gives the guard a safe fallback signal.
                current = None
                driver = None
                recommended = None

        virtual_memory = psutil.virtual_memory()
        try:
            swap_used_bytes = int(psutil.swap_memory().used)
        except OSError:
            # macOS may deny the underlying sysctl inside a sandbox.  Swap is
            # supplementary telemetry; MPS and available RAM still provide
            # the pressure guard's actual stop signals.
            swap_used_bytes = 0
        return MemorySnapshot(
            stage=stage,
            mps_current_bytes=current,
            mps_driver_bytes=driver,
            mps_recommended_bytes=recommended,
            system_available_bytes=int(virtual_memory.available),
            system_total_bytes=int(virtual_memory.total),
            swap_used_bytes=swap_used_bytes,
        )

    def status(self, snapshot: MemorySnapshot) -> MemoryStatus:
        mps_ratio = snapshot.mps_pressure_ratio
        if (
            mps_ratio is not None
            and mps_ratio >= self.thresholds.critical_mps_ratio
        ) or (
            snapshot.system_available_gib
            <= self.thresholds.critical_system_available_gib
        ):
            return "critical"
        if (
            mps_ratio is not None
            and mps_ratio >= self.thresholds.warning_mps_ratio
        ) or (
            snapshot.effective_system_available_gib
            <= self.thresholds.warning_system_available_gib
        ):
            return "warning"
        return "normal"


def release_device_memory(device: torch.device, *, synchronize: bool = True) -> None:
    """Release unreachable tensors and cached accelerator allocations."""

    gc.collect()
    if device.type == "mps" and torch.backends.mps.is_available():
        try:
            if synchronize and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except RuntimeError:
            pass
        try:
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except RuntimeError:
            pass
    elif device.type == "cuda" and torch.cuda.is_available():
        try:
            if synchronize:
                torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            if synchronize:
                torch.xpu.synchronize(device)
            torch.xpu.empty_cache()
        except RuntimeError:
            pass


def tree_to_cpu(value: Any) -> Any:
    """Copy a nested checkpoint value to CPU without retaining autograd graphs."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    return value
