"""
Smart Optimization Controller
Decides whether to enable optimization based on actual load,
avoiding performance loss under light workloads.
"""

import torch
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque


class LoadLevel(Enum):
    """Load levels"""
    LIGHT = "light"      # Light: simple preview, single-frame processing
    MEDIUM = "medium"    # Medium: small batch, low-resolution restoration
    HEAVY = "heavy"      # Heavy: large batch, high-resolution restoration
    STRESS = "stress"    # Stress: extremely large batch, ultra-high resolution


@dataclass
class LoadMetrics:
    """Load metrics"""
    frame_size: Tuple[int, int]  # Frame size
    batch_size: int              # Batch size
    processing_complexity: float # Processing complexity (0–1)
    concurrent_tasks: int        # Concurrent tasks
    memory_usage_mb: float       # Memory usage (MB)
    estimated_compute_time: float # Estimated compute time (seconds)


@dataclass
class OptimizationThresholds:
    """Thresholds for enabling optimization"""
    # Thresholds derived from benchmark results
    min_frame_pixels: int = 1920 * 1080      # Minimum pixels (1080p)
    min_batch_size: int = 4                  # Minimum batch size
    min_processing_complexity: float = 0.6   # Minimum processing complexity
    min_concurrent_tasks: int = 2            # Minimum concurrent tasks
    min_memory_usage_mb: float = 500         # Minimum memory usage
    min_compute_time_sec: float = 0.5        # Minimum compute time


class SmartOptimizationController:
    """Smart optimization controller"""
    
    def __init__(self, thresholds: Optional[OptimizationThresholds] = None):
        self.thresholds = thresholds or OptimizationThresholds()
        self.load_history = deque(maxlen=10)  # Keep the last 10 load records
        self.optimization_enabled = False
        self.current_load_level = LoadLevel.LIGHT
        self.performance_stats = {
            'optimization_decisions': 0,
            'optimization_enabled_count': 0,
            'optimization_disabled_count': 0,
            'load_level_distribution': {level.value: 0 for level in LoadLevel}
        }
        self._lock = threading.Lock()
    
    def analyze_load(self, metrics: LoadMetrics) -> LoadLevel:
        """Analyze current load level"""
        frame_pixels = metrics.frame_size[0] * metrics.frame_size[1]
        
        # Compute load score
        load_score = 0
        
        # Frame size weight (40%)
        if frame_pixels >= 3840 * 2160:  # 4K+
            load_score += 0.4
        elif frame_pixels >= 2560 * 1440:  # 1440p+
            load_score += 0.35
        elif frame_pixels >= 1920 * 1080:  # 1080p+
            load_score += 0.25  # Increase 1080p weight
        else:  # 720p and below
            load_score += 0.1
        
        # Batch size weight (25%)
        if metrics.batch_size >= 8:
            load_score += 0.25
        elif metrics.batch_size >= 4:
            load_score += 0.2
        elif metrics.batch_size >= 2:
            load_score += 0.15
        else:
            load_score += 0.05
        
        # Processing complexity weight (20%)
        load_score += metrics.processing_complexity * 0.2
        
        # Concurrent tasks weight (10%)
        if metrics.concurrent_tasks >= 4:
            load_score += 0.1
        elif metrics.concurrent_tasks >= 2:
            load_score += 0.07
        else:
            load_score += 0.03
        
        # Memory usage weight (5%)
        if metrics.memory_usage_mb >= 1000:
            load_score += 0.05
        elif metrics.memory_usage_mb >= 500:
            load_score += 0.03
        else:
            load_score += 0.01
        
        # Determine level based on load score — adjusted thresholds
        if load_score >= 0.75:  # Lower STRESS threshold
            return LoadLevel.STRESS
        elif load_score >= 0.55:  # Lower HEAVY threshold
            return LoadLevel.HEAVY
        elif load_score >= 0.35:  # Lower MEDIUM threshold
            return LoadLevel.MEDIUM
        else:
            return LoadLevel.LIGHT
    
    def should_enable_optimization(self, metrics: LoadMetrics) -> bool:
        """Decide whether optimization should be enabled"""
        with self._lock:
            load_level = self.analyze_load(metrics)
            self.current_load_level = load_level
            self.load_history.append((time.time(), load_level, metrics))
            
            # Update statistics
            self.performance_stats['optimization_decisions'] += 1
            self.performance_stats['load_level_distribution'][load_level.value] += 1
            
            # Decision logic from benchmark results:
            # - LIGHT/MEDIUM: negative optimization (-23%), disable
            # - HEAVY/STRESS: positive optimization (+63%/+51%), enable
            should_optimize = load_level in [LoadLevel.HEAVY, LoadLevel.STRESS]
            
            # Additional threshold checks to ensure truly heavy load
            if should_optimize:
                frame_pixels = metrics.frame_size[0] * metrics.frame_size[1]
                should_optimize = (
                    frame_pixels >= self.thresholds.min_frame_pixels or
                    metrics.batch_size >= self.thresholds.min_batch_size or
                    metrics.processing_complexity >= self.thresholds.min_processing_complexity or
                    metrics.concurrent_tasks >= self.thresholds.min_concurrent_tasks or
                    metrics.memory_usage_mb >= self.thresholds.min_memory_usage_mb or
                    metrics.estimated_compute_time >= self.thresholds.min_compute_time_sec
                )
            
            # Update stats
            if should_optimize:
                self.performance_stats['optimization_enabled_count'] += 1
            else:
                self.performance_stats['optimization_disabled_count'] += 1
            
            self.optimization_enabled = should_optimize
            return should_optimize
    
    def get_optimization_recommendation(self, metrics: LoadMetrics) -> Dict[str, Any]:
        """Get optimization recommendation"""
        load_level = self.analyze_load(metrics)
        should_optimize = self.should_enable_optimization(metrics)
        
        recommendation = {
            'load_level': load_level.value,
            'should_optimize': should_optimize,
            'confidence': self._calculate_confidence(metrics),
            'reasons': self._get_optimization_reasons(metrics, load_level),
            'expected_performance_gain': self._get_expected_gain(load_level),
            'optimization_config': self._get_optimization_config(load_level) if should_optimize else None
        }
        
        return recommendation
    
    def _calculate_confidence(self, metrics: LoadMetrics) -> float:
        """Calculate decision confidence"""
        # Compute confidence based on history and current metrics
        if len(self.load_history) < 3:
            return 0.7  # Initial confidence
        
        # Check load stability
        recent_levels = [entry[1] for entry in list(self.load_history)[-3:]]
        stability = len(set(recent_levels)) / len(recent_levels)  # Smaller is more stable
        
        # Metric clarity
        frame_pixels = metrics.frame_size[0] * metrics.frame_size[1]
        clarity_score = 0
        
        if frame_pixels >= 1920 * 1080:  # Clearly high resolution
            clarity_score += 0.3
        if metrics.batch_size >= 4:  # Clearly batched processing
            clarity_score += 0.3
        if metrics.processing_complexity >= 0.7:  # Clearly high complexity
            clarity_score += 0.4
        
        confidence = (1 - stability) * 0.5 + clarity_score * 0.5
        return min(max(confidence, 0.3), 0.95)  # Clamp to 0.3–0.95
    
    def _get_optimization_reasons(self, metrics: LoadMetrics, load_level: LoadLevel) -> List[str]:
        """Get reasons for the optimization decision"""
        reasons = []
        frame_pixels = metrics.frame_size[0] * metrics.frame_size[1]
        
        if load_level in [LoadLevel.LIGHT, LoadLevel.MEDIUM]:
            reasons.append(f"Load level is {load_level.value}; optimization overhead outweighs benefits")
            if frame_pixels < 1920 * 1080:
                reasons.append("Lower resolution; GPU optimization effectiveness is limited")
            if metrics.batch_size < 4:
                reasons.append("Small batch size; parallel optimization gains are modest")
        else:
            reasons.append(f"Load level is {load_level.value}; optimization benefits are significant")
            if frame_pixels >= 1920 * 1080:
                reasons.append("High-resolution processing; GPU acceleration is effective")
            if metrics.batch_size >= 4:
                reasons.append("Large batch processing; parallel optimization gains are substantial")
            if metrics.processing_complexity >= 0.7:
                reasons.append("High complexity processing; optimization algorithms perform well")
        
        return reasons
    
    def _get_expected_gain(self, load_level: LoadLevel) -> str:
        """Get expected performance gain"""
        # Based on benchmark results
        gains = {
            LoadLevel.LIGHT: "-23.0%",
            LoadLevel.MEDIUM: "-23.2%", 
            LoadLevel.HEAVY: "+63.4%",
            LoadLevel.STRESS: "+51.3%"
        }
        return gains.get(load_level, "unknown")
    
    def _get_optimization_config(self, load_level: LoadLevel) -> Dict[str, Any]:
        """Get optimization configuration for the given load level"""
        if load_level == LoadLevel.HEAVY:
            return {
                'enable_gpu_acceleration': True,
                'enable_memory_optimization': True,
                'enable_batch_processing': True,
                'enable_predictive_scheduling': False,  # Keep stable under heavy load
                'max_concurrent_tasks': 4,
                'memory_pool_size': '1GB'
            }
        elif load_level == LoadLevel.STRESS:
            return {
                'enable_gpu_acceleration': True,
                'enable_memory_optimization': True,
                'enable_batch_processing': True,
                'enable_predictive_scheduling': True,   # Enable predictive scheduling under stress load
                'max_concurrent_tasks': 6,
                'memory_pool_size': '2GB'
            }
        else:
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            total_decisions = self.performance_stats['optimization_decisions']
            if total_decisions == 0:
                return self.performance_stats
            
            stats = self.performance_stats.copy()
            stats['optimization_rate'] = self.performance_stats['optimization_enabled_count'] / total_decisions
            stats['current_load_level'] = self.current_load_level.value
            stats['optimization_currently_enabled'] = self.optimization_enabled
            
            return stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self.performance_stats = {
                'optimization_decisions': 0,
                'optimization_enabled_count': 0,
                'optimization_disabled_count': 0,
                'load_level_distribution': {level.value: 0 for level in LoadLevel}
            }
            self.load_history.clear()


def create_load_metrics_from_frame_data(
    frame_data: torch.Tensor,
    batch_size: int = 1,
    processing_mode: str = "preview",
    concurrent_tasks: int = 1
) -> LoadMetrics:
    """Create load metrics from frame data"""
    
    # Get frame size
    if len(frame_data.shape) == 4:  # [B, C, H, W]
        height, width = frame_data.shape[2], frame_data.shape[3]
        actual_batch_size = frame_data.shape[0]
    elif len(frame_data.shape) == 3:  # [C, H, W]
        height, width = frame_data.shape[1], frame_data.shape[2]
        actual_batch_size = 1
    else:
        height, width = 720, 1280  # Defaults
        actual_batch_size = 1
    
    # Determine complexity based on processing mode
    complexity_map = {
        "preview": 0.2,
        "mosaic_detection": 0.4,
        "mosaic_restoration": 0.8,
        "batch_export": 0.9,
        "quality_enhancement": 0.7
    }
    processing_complexity = complexity_map.get(processing_mode, 0.5)
    
    # Estimate memory usage
    frame_pixels = height * width
    memory_per_pixel = 4 * 3  # Assume RGB float32
    memory_usage_mb = (frame_pixels * memory_per_pixel * max(batch_size, actual_batch_size)) / (1024 * 1024)
    
    # Estimate compute time
    base_time_per_megapixel = 0.1  # Base time
    megapixels = frame_pixels / (1024 * 1024)
    estimated_compute_time = megapixels * base_time_per_megapixel * processing_complexity
    
    return LoadMetrics(
        frame_size=(width, height),
        batch_size=max(batch_size, actual_batch_size),
        processing_complexity=processing_complexity,
        concurrent_tasks=concurrent_tasks,
        memory_usage_mb=memory_usage_mb,
        estimated_compute_time=estimated_compute_time
    )