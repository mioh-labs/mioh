# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
MPS (Metal Performance Shaders) Utilities for Apple Silicon

This module provides utility functions for safe MPS operations,
including fallback mechanisms for unsupported operations.
"""

import torch
import torch.nn.functional as F
import logging
from functools import wraps
import psutil

logger = logging.getLogger(__name__)


def get_mps_memory_stats():
    """
    Return a unified snapshot of MPS / system memory state.
    """
    stats = {
        "available": torch.backends.mps.is_available(),
        "recommended_max_bytes": None,
        "current_allocated_bytes": None,
        "driver_allocated_bytes": None,
        "system_available_bytes": None,
        "system_total_bytes": None,
        "pressure_ratio": None,
    }
    vm = psutil.virtual_memory()
    stats["system_available_bytes"] = vm.available
    stats["system_total_bytes"] = vm.total

    if not stats["available"]:
        return stats

    try:
        if hasattr(torch.mps, "recommended_max_memory"):
            stats["recommended_max_bytes"] = int(torch.mps.recommended_max_memory())
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    except Exception as e:
        logger.debug(f"Could not query MPS memory stats: {e}")

    recommended = stats["recommended_max_bytes"]
    driver_allocated = stats["driver_allocated_bytes"]
    current_allocated = stats["current_allocated_bytes"]
    if recommended and driver_allocated is not None:
        stats["pressure_ratio"] = min(1.0, max(0.0, driver_allocated / max(recommended, 1)))
    elif recommended and current_allocated is not None:
        stats["pressure_ratio"] = min(1.0, max(0.0, current_allocated / max(recommended, 1)))

    return stats


def get_mps_available_memory_gb():
    """
    Estimate usable MPS headroom in GiB.
    """
    stats = get_mps_memory_stats()
    recommended = stats["recommended_max_bytes"]
    driver_allocated = stats["driver_allocated_bytes"]
    system_available = stats["system_available_bytes"]
    if recommended and driver_allocated is not None:
        return max(0.0, (recommended - driver_allocated) / (1024 ** 3))
    return max(0.0, system_available / (1024 ** 3))


def configure_mps_runtime(memory_fraction=None, verbose=True):
    """
    Apply process-level MPS memory limit when supported.
    """
    if not torch.backends.mps.is_available():
        return False
    if memory_fraction is None:
        return False
    try:
        fraction = float(memory_fraction)
    except (TypeError, ValueError):
        logger.warning("Invalid MPS memory fraction: %s", memory_fraction)
        return False
    if not (0.0 < fraction <= 1.0):
        logger.warning("MPS memory fraction must be within (0, 1], got %s", fraction)
        return False
    if not hasattr(torch.mps, "set_per_process_memory_fraction"):
        return False
    try:
        torch.mps.set_per_process_memory_fraction(fraction)
        if verbose:
            logger.info("Configured torch.mps per-process memory fraction to %.3f", fraction)
        return True
    except Exception as e:
        logger.warning("Could not set MPS memory fraction %.3f: %s", fraction, e)
        return False

def check_mps_tensor_validity(tensor, name="tensor", silent=False):
    """
    Check if MPS tensor is valid for operations
    
    Args:
        tensor: PyTorch tensor
        name: Name for logging
        silent: If True, don't log warnings
    
    Returns:
        bool: True if tensor is valid
    """
    if not isinstance(tensor, torch.Tensor):
        if not silent:
            logger.warning(f"{name} is not a tensor")
        return False
    
    if tensor.device.type != 'mps':
        return True  # Non-MPS tensors are always valid
    
    # Check for empty tensor
    if tensor.numel() == 0:
        if not silent:
            logger.warning(f"{name} is empty (numel=0)")
        return False
    
    # Check for NaN or Inf
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        if not silent:
            logger.warning(f"{name} contains NaN or Inf values")
        return False
    
    return True

def ensure_mps_tensor_contiguous(tensor):
    """
    Ensure MPS tensor is contiguous in memory
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        Contiguous tensor
    """
    if isinstance(tensor, torch.Tensor) and tensor.device.type == 'mps':
        if not tensor.is_contiguous():
            return tensor.contiguous()
    return tensor

def safe_mps_grid_sample(input, grid, mode='bilinear', padding_mode='zeros', 
                         align_corners=None):
    """
    Safe grid_sample operation for MPS with automatic fallback
    
    MPS does not support 'border' padding mode, so we automatically
    fallback to 'zeros' or CPU computation if needed.
    
    Args:
        input: Input tensor (N, C, H, W)
        grid: Grid tensor (N, H_out, W_out, 2)
        mode: Interpolation mode ('bilinear' or 'nearest')
        padding_mode: Padding mode ('zeros', 'border', 'reflection')
        align_corners: Whether to align corners
    
    Returns:
        Output tensor
    """
    device = input.device
    
    # Check if using MPS
    if device.type == 'mps':
        # MPS does not support 'border' padding mode
        if padding_mode == 'border':
            logger.debug("MPS: 'border' padding not supported, using 'zeros' instead")
            padding_mode = 'zeros'
        
        # Try MPS operation
        try:
            output = F.grid_sample(
                input, grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners
            )
            return output
        except RuntimeError as e:
            # If MPS fails, fallback to CPU
            logger.warning(f"MPS grid_sample failed: {e}, falling back to CPU")
            input_cpu = input.cpu()
            grid_cpu = grid.cpu()
            
            output_cpu = F.grid_sample(
                input_cpu, grid_cpu,
                mode=mode,
                padding_mode=padding_mode if padding_mode != 'border' else 'zeros',
                align_corners=align_corners
            )
            
            return output_cpu.to(device)
    else:
        # Non-MPS device, use standard grid_sample
        return F.grid_sample(
            input, grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners
        )

def mps_safe_operation(fallback_to_cpu=True):
    """
    Decorator for MPS-safe operations with automatic CPU fallback
    
    Args:
        fallback_to_cpu: If True, fallback to CPU on MPS errors
    
    Usage:
        @mps_safe_operation()
        def my_function(tensor):
            # Your operation here
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if fallback_to_cpu and any(isinstance(arg, torch.Tensor) and arg.device.type == 'mps' 
                                          for arg in args):
                    logger.warning(f"MPS operation failed in {func.__name__}: {e}, falling back to CPU")
                    
                    # Move tensors to CPU
                    args_cpu = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor) and arg.device.type == 'mps':
                            args_cpu.append(arg.cpu())
                        else:
                            args_cpu.append(arg)
                    
                    kwargs_cpu = {}
                    for key, value in kwargs.items():
                        if isinstance(value, torch.Tensor) and value.device.type == 'mps':
                            kwargs_cpu[key] = value.cpu()
                        else:
                            kwargs_cpu[key] = value
                    
                    # Execute on CPU
                    result = func(*args_cpu, **kwargs_cpu)
                    
                    # Move result back to MPS
                    if isinstance(result, torch.Tensor):
                        return result.to('mps')
                    elif isinstance(result, (list, tuple)):
                        return type(result)(
                            r.to('mps') if isinstance(r, torch.Tensor) else r 
                            for r in result
                        )
                    return result
                else:
                    raise
        return wrapper
    return decorator

def optimize_mps_memory():
    """
    Optimize MPS memory usage by clearing cache
    
    This should be called periodically during long-running operations
    to prevent memory accumulation.
    """
    if torch.backends.mps.is_available():
        try:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.debug("MPS cache cleared")
        except Exception as e:
            logger.debug(f"Could not clear MPS cache: {e}")

def get_mps_info():
    """
    Get MPS device information
    
    Returns:
        dict: MPS device information
    """
    info = {
        'available': torch.backends.mps.is_available(),
        'built': torch.backends.mps.is_built()
    }
    
    if info['available']:
        try:
            # Get device properties if available
            device = torch.device('mps')
            info['device'] = str(device)
        except Exception as e:
            logger.debug(f"Could not get MPS device info: {e}")
    
    return info

def tensor_to_numpy(tensor):
    """
    Safely convert PyTorch Tensor to NumPy array
    Supports MPS/CUDA/CPU tensors
    
    Args:
        tensor: PyTorch Tensor, NumPy array, or other types
    
    Returns:
        NumPy array
    """
    if isinstance(tensor, torch.Tensor):
        # MPS or CUDA tensors must be moved to CPU first
        if tensor.device.type != 'cpu':
            return tensor.cpu().numpy()
        else:
            return tensor.numpy()
    # Already a numpy array
    elif hasattr(tensor, '__array__'):
        return tensor.__array__()
    else:
        # Other types, return as-is
        return tensor
