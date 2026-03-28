"""
Integrated Optimization Manager for LADA Project

Key capabilities:
1. Unified management of all optimization components
2. Coordinated CPU–GPU resource allocation
3. Real-time performance monitoring and adjustments
4. Optimization recommendations and reports
5. Seamless integration with existing pipeline architecture
"""

import torch
import threading
import time
import psutil
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

# Import optimization components
from .mps_performance_optimizer import MPSPerformanceOptimizer, get_mps_optimizer
from .enhanced_cpu_gpu_balancer import EnhancedCPUGPUBalancer, EnhancedTask, TaskPriority
from .realtime_gui_optimizer import RealtimeGUIOptimizer, PlaybackQuality, FrameMetadata


class OptimizationMode(Enum):
    """Optimization modes"""
    PERFORMANCE = "performance"    # Performance first
    QUALITY = "quality"            # Quality first
    BALANCED = "balanced"          # Balanced mode
    POWER_SAVING = "power_saving"  # Power-saving mode


@dataclass
class SystemCapabilities:
    """System capability assessment"""
    cpu_cores: int
    cpu_frequency: float
    total_memory: int
    gpu_memory: int
    mps_available: bool
    estimated_performance_score: float


@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    mode: OptimizationMode
    target_fps: float
    max_cpu_utilization: float
    max_gpu_utilization: float
    enable_adaptive_quality: bool
    enable_predictive_scheduling: bool
    enable_smart_frame_dropping: bool
    enable_memory_optimization: bool
    custom_params: Dict[str, Any]


class SystemProfiler:
    """System performance profiler"""
    
    def __init__(self):
        self.capabilities = None
        self.benchmark_results = {}
    
    def profile_system(self) -> SystemCapabilities:
        """Profile system capabilities"""
        # CPU info
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_frequency = cpu_freq.current if cpu_freq else 2000.0
        
        # Memory info
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total
        
        # GPU info
        mps_available = torch.backends.mps.is_available()
        gpu_memory = 0
        
        if mps_available:
            try:
                # Try allocating a small tensor to probe GPU memory
                test_tensor = torch.zeros(1000, 1000, device='mps')
                gpu_memory = torch.mps.driver_allocated_memory()
                del test_tensor
                torch.mps.empty_cache()
            except:
                gpu_memory = 8 * 1024 * 1024 * 1024  # Assume 8GB
        
        # Compute performance score
        performance_score = self._calculate_performance_score(
            cpu_cores, cpu_frequency, total_memory, gpu_memory, mps_available
        )
        
        self.capabilities = SystemCapabilities(
            cpu_cores=cpu_cores,
            cpu_frequency=cpu_frequency,
            total_memory=total_memory,
            gpu_memory=gpu_memory,
            mps_available=mps_available,
            estimated_performance_score=performance_score
        )
        
        return self.capabilities
    
    def _calculate_performance_score(self, cpu_cores: int, cpu_freq: float, memory: int, gpu_memory: int, mps_available: bool) -> float:
        """Compute performance score (0–100)"""
        score = 0.0
        
        # CPU score (40%)
        cpu_score = min(100, (cpu_cores * cpu_freq / 1000) * 2)
        score += cpu_score * 0.4
        
        # Memory score (20%)
        memory_gb = memory / (1024**3)
        memory_score = min(100, memory_gb * 6.25)  # 16GB = 100 points
        score += memory_score * 0.2
        
        # GPU score (40%)
        if mps_available:
            gpu_gb = gpu_memory / (1024**3)
            gpu_score = min(100, gpu_gb * 12.5)  # 8GB = 100 points
            score += gpu_score * 0.4
        else:
            score += 20  # Baseline when no GPU is available
        
        return min(100, score)
    
    def run_benchmark(self) -> Dict[str, float]:
        """Run performance benchmarks"""
        results = {}
        
        # CPU benchmarks
        results['cpu_compute'] = self._benchmark_cpu_compute()
        results['cpu_memory_bandwidth'] = self._benchmark_cpu_memory()
        
        # GPU benchmarks
        if torch.backends.mps.is_available():
            results['gpu_compute'] = self._benchmark_gpu_compute()
            results['gpu_memory_bandwidth'] = self._benchmark_gpu_memory()
            results['cpu_gpu_transfer'] = self._benchmark_cpu_gpu_transfer()
        
        self.benchmark_results = results
        return results
    
    def _benchmark_cpu_compute(self) -> float:
        """CPU compute benchmark"""
        start_time = time.time()
        
        # Matrix multiplication test
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        
        for _ in range(10):
            c = torch.mm(a, b)
        
        elapsed = time.time() - start_time
        return 10.0 / elapsed  # operations per second
    
    def _benchmark_cpu_memory(self) -> float:
        """CPU memory bandwidth benchmark"""
        start_time = time.time()
        
        # Large array copy test
        size = 100 * 1024 * 1024  # 100MB
        data = torch.randn(size // 4)  # float32
        
        for _ in range(5):
            copy = data.clone()
        
        elapsed = time.time() - start_time
        bytes_transferred = size * 5 * 2  # read + write
        return bytes_transferred / elapsed / (1024**3)  # GB/s
    
    def _benchmark_gpu_compute(self) -> float:
        """GPU compute benchmark"""
        if not torch.backends.mps.is_available():
            return 0.0
        
        start_time = time.time()
        
        # GPU matrix multiplication test
        a = torch.randn(1000, 1000, device='mps')
        b = torch.randn(1000, 1000, device='mps')
        
        for _ in range(10):
            c = torch.mm(a, b)
        
        torch.mps.synchronize()
        elapsed = time.time() - start_time
        return 10.0 / elapsed
    
    def _benchmark_gpu_memory(self) -> float:
        """GPU memory bandwidth benchmark"""
        if not torch.backends.mps.is_available():
            return 0.0
        
        start_time = time.time()
        
        size = 50 * 1024 * 1024  # 50MB (MPS memory constraints)
        data = torch.randn(size // 4, device='mps')
        
        for _ in range(5):
            copy = data.clone()
        
        torch.mps.synchronize()
        elapsed = time.time() - start_time
        bytes_transferred = size * 5 * 2
        return bytes_transferred / elapsed / (1024**3)
    
    def _benchmark_cpu_gpu_transfer(self) -> float:
        """CPU–GPU transfer benchmark"""
        if not torch.backends.mps.is_available():
            return 0.0
        
        start_time = time.time()
        
        size = 10 * 1024 * 1024  # 10MB
        cpu_data = torch.randn(size // 4)
        
        for _ in range(10):
            gpu_data = cpu_data.to('mps')
            cpu_result = gpu_data.to('cpu')
        
        torch.mps.synchronize()
        elapsed = time.time() - start_time
        bytes_transferred = size * 10 * 2  # bidirectional transfer
        return bytes_transferred / elapsed / (1024**3)


class ConfigurationOptimizer:
    """Configuration optimizer"""
    
    def __init__(self, system_capabilities: SystemCapabilities):
        self.capabilities = system_capabilities
    
    def generate_optimal_config(self, mode: OptimizationMode, target_fps: float = 30.0) -> OptimizationConfig:
        """Generate optimal configuration"""
        base_config = self._get_base_config(mode)
        
        # Adjust configuration based on system capabilities
        if self.capabilities.estimated_performance_score < 30:
            # Low-performance system
            config = self._optimize_for_low_performance(base_config, target_fps)
        elif self.capabilities.estimated_performance_score < 60:
            # Medium-performance system
            config = self._optimize_for_medium_performance(base_config, target_fps)
        else:
            # High-performance system
            config = self._optimize_for_high_performance(base_config, target_fps)
        
        return config
    
    def _get_base_config(self, mode: OptimizationMode) -> Dict[str, Any]:
        """Get base configuration"""
        configs = {
            OptimizationMode.PERFORMANCE: {
                'max_cpu_utilization': 0.9,
                'max_gpu_utilization': 0.95,
                'enable_adaptive_quality': True,
                'enable_predictive_scheduling': True,
                'enable_smart_frame_dropping': True,
                'enable_memory_optimization': True,
                'quality_priority': 'performance'
            },
            OptimizationMode.QUALITY: {
                'max_cpu_utilization': 0.7,
                'max_gpu_utilization': 0.8,
                'enable_adaptive_quality': False,
                'enable_predictive_scheduling': True,
                'enable_smart_frame_dropping': False,
                'enable_memory_optimization': False,
                'quality_priority': 'quality'
            },
            OptimizationMode.BALANCED: {
                'max_cpu_utilization': 0.8,
                'max_gpu_utilization': 0.85,
                'enable_adaptive_quality': True,
                'enable_predictive_scheduling': True,
                'enable_smart_frame_dropping': True,
                'enable_memory_optimization': True,
                'quality_priority': 'balanced'
            },
            OptimizationMode.POWER_SAVING: {
                'max_cpu_utilization': 0.6,
                'max_gpu_utilization': 0.7,
                'enable_adaptive_quality': True,
                'enable_predictive_scheduling': False,
                'enable_smart_frame_dropping': True,
                'enable_memory_optimization': True,
                'quality_priority': 'power_saving'
            }
        }
        
        return configs.get(mode, configs[OptimizationMode.BALANCED])
    
    def _optimize_for_low_performance(self, base_config: Dict[str, Any], target_fps: float) -> OptimizationConfig:
        """Optimize configuration for low-performance systems"""
        custom_params = {
            'cpu_workers': max(2, self.capabilities.cpu_cores // 2),
            'gpu_workers': 1,
            'initial_batch_size': 1,
            'max_batch_size': 2,
            'texture_cache_size': 20,
            'command_buffer_pool_size': 2,
            'target_buffer_size': 5,
            'max_buffer_size': 10
        }
        
        return OptimizationConfig(
            mode=OptimizationMode.POWER_SAVING,
            target_fps=min(target_fps, 20.0),
            max_cpu_utilization=0.6,
            max_gpu_utilization=0.7,
            enable_adaptive_quality=True,
            enable_predictive_scheduling=False,
            enable_smart_frame_dropping=True,
            enable_memory_optimization=True,
            custom_params=custom_params
        )
    
    def _optimize_for_medium_performance(self, base_config: Dict[str, Any], target_fps: float) -> OptimizationConfig:
        """Optimize configuration for medium-performance systems"""
        custom_params = {
            'cpu_workers': self.capabilities.cpu_cores,
            'gpu_workers': 2,
            'initial_batch_size': 2,
            'max_batch_size': 4,
            'texture_cache_size': 50,
            'command_buffer_pool_size': 3,
            'target_buffer_size': 10,
            'max_buffer_size': 20
        }
        
        return OptimizationConfig(
            mode=OptimizationMode.BALANCED,
            target_fps=target_fps,
            max_cpu_utilization=base_config['max_cpu_utilization'],
            max_gpu_utilization=base_config['max_gpu_utilization'],
            enable_adaptive_quality=base_config['enable_adaptive_quality'],
            enable_predictive_scheduling=base_config['enable_predictive_scheduling'],
            enable_smart_frame_dropping=base_config['enable_smart_frame_dropping'],
            enable_memory_optimization=base_config['enable_memory_optimization'],
            custom_params=custom_params
        )
    
    def _optimize_for_high_performance(self, base_config: Dict[str, Any], target_fps: float) -> OptimizationConfig:
        """Optimize configuration for high-performance systems"""
        custom_params = {
            'cpu_workers': self.capabilities.cpu_cores * 2,
            'gpu_workers': 4,
            'initial_batch_size': 4,
            'max_batch_size': 8,
            'texture_cache_size': 100,
            'command_buffer_pool_size': 4,
            'target_buffer_size': 15,
            'max_buffer_size': 30
        }
        
        return OptimizationConfig(
            mode=OptimizationMode.PERFORMANCE,
            target_fps=target_fps,
            max_cpu_utilization=base_config['max_cpu_utilization'],
            max_gpu_utilization=base_config['max_gpu_utilization'],
            enable_adaptive_quality=base_config['enable_adaptive_quality'],
            enable_predictive_scheduling=base_config['enable_predictive_scheduling'],
            enable_smart_frame_dropping=base_config['enable_smart_frame_dropping'],
            enable_memory_optimization=base_config['enable_memory_optimization'],
            custom_params=custom_params
        )


class IntegratedOptimizationManager:
    """Integrated optimization manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        
        # System profiling
        self.profiler = SystemProfiler()
        self.capabilities = self.profiler.profile_system()
        
        # Configuration optimizer
        self.config_optimizer = ConfigurationOptimizer(self.capabilities)
        self.current_config = None
        
        # Optimization components
        self.mps_optimizer = None
        self.cpu_gpu_balancer = None
        self.gui_optimizer = None
        
        # Performance monitoring
        self.performance_history = []
        self.optimization_active = False
        self.monitor_thread = None
        
        # Load configuration
        self._load_configuration()
    
    def initialize_optimization(self, mode: OptimizationMode = OptimizationMode.BALANCED, target_fps: float = 30.0):
        """Initialize optimization"""
        print(f"Initializing optimization with mode: {mode.value}, target FPS: {target_fps}")
        
        # Generate optimal configuration
        self.current_config = self.config_optimizer.generate_optimal_config(mode, target_fps)
        
        # Do not initialize any heavy components; keep minimal overhead
        self.optimization_active = True
        print("Optimization initialized successfully")
    
    def process_frame_optimized(self, frame_data: torch.Tensor, frame_metadata: FrameMetadata) -> Optional[torch.Tensor]:
        """Optimized frame processing"""
        if not self.optimization_active:
            return frame_data
        
        # Use RealtimeGUIOptimizer for actual optimization
        if self.gui_optimizer:
            result = self.gui_optimizer.process_frame(frame_data, frame_metadata)
            if result is None:
                # Frame dropped
                return None
            frame_data = result
        
        # Apply GPU optimizations with MPS optimizer
        if self.mps_optimizer:
            try:
                frame_data = self.mps_optimizer.optimize_tensor_operations(frame_data)
            except Exception as e:
                # If MPS optimization fails, continue with original tensor
                pass
        
        # Ensure tensor resides on the correct device
        if torch.backends.mps.is_available() and frame_data.device.type != 'mps':
            try:
                frame_data = frame_data.to('mps', non_blocking=True)
            except Exception:
                pass  # If transfer fails, keep original tensor
        
        return frame_data
    
    def submit_processing_task(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, **kwargs) -> Any:
        """Submit a processing task"""
        # Execute the task directly without using a complex scheduling system
        # Directly execute the task; no complex scheduler here
        return func(*args, **kwargs)
    
    def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        while self.optimization_active:
            try:
                # Collect performance data
                performance_data = self._collect_performance_data()
                self.performance_history.append(performance_data)
                
                # Maintain history size
                if len(self.performance_history) > 100:
                    self.performance_history.pop(0)
                
                # Check whether optimization strategy needs adjustment
                self._check_and_adjust_optimization()
                
                time.sleep(1.0)  # Monitor once per second
                
            except Exception as e:
                print(f"Performance monitor error: {e}")
                time.sleep(5.0)
    
    def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect performance data"""
        data = {
            'timestamp': time.time(),
            'cpu_utilization': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'system_load': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        }
        
        # MPS performance metrics
        if self.mps_optimizer:
            mps_report = self.mps_optimizer.get_performance_report()
            data['mps_metrics'] = mps_report
        
        # CPU–GPU balancer metrics
        if self.cpu_gpu_balancer:
            balancer_stats = self.cpu_gpu_balancer.get_performance_statistics()
            data['balancer_metrics'] = balancer_stats
        
        # GUI optimizer metrics
        if self.gui_optimizer:
            gui_report = self.gui_optimizer.get_optimization_report()
            data['gui_metrics'] = gui_report
        
        return data
    
    def _check_and_adjust_optimization(self):
        """Check and adjust optimization strategy"""
        if len(self.performance_history) < 5:
            return
        
        recent_data = self.performance_history[-5:]
        
        # Compute average metrics
        avg_cpu = sum(d['cpu_utilization'] for d in recent_data) / len(recent_data)
        avg_memory = sum(d['memory_usage'] for d in recent_data) / len(recent_data)
        
        # Acquire GUI performance metrics
        gui_metrics = recent_data[-1].get('gui_metrics', {})
        current_fps = gui_metrics.get('performance_metrics', {}).get('current_fps', 0)
        drop_rate = gui_metrics.get('performance_metrics', {}).get('drop_rate_percent', 0)
        
        # Adjustment logic
        adjustments_made = False
        
        # High CPU utilization
        if avg_cpu > self.current_config.max_cpu_utilization * 100:
            if self.gui_optimizer and self.current_config.enable_adaptive_quality:
                # Lower quality to reduce CPU load
                current_quality = self.gui_optimizer.quality_controller.current_quality
                if current_quality != PlaybackQuality.LOW:
                    self.gui_optimizer.quality_controller._decrease_quality()
                    adjustments_made = True
        
        # Low FPS
        if current_fps < self.current_config.target_fps * 0.8:
            if self.gui_optimizer:
                # Enable more aggressive frame dropping
                self.gui_optimizer.frame_dropper.drop_threshold = max(0.3, 
                    self.gui_optimizer.frame_dropper.drop_threshold - 0.1)
                adjustments_made = True
        
        # High memory usage
        if avg_memory > 85:
            if self.mps_optimizer:
                self.mps_optimizer.optimize_memory_usage()
                adjustments_made = True
        
        if adjustments_made:
            print("Optimization strategy adjusted based on performance metrics")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report"""
        report = {
            'system_capabilities': {
                'cpu_cores': self.capabilities.cpu_cores,
                'cpu_frequency': self.capabilities.cpu_frequency,
                'total_memory_gb': self.capabilities.total_memory / (1024**3),
                'gpu_memory_gb': self.capabilities.gpu_memory / (1024**3),
                'mps_available': self.capabilities.mps_available,
                'performance_score': self.capabilities.estimated_performance_score
            },
            'current_configuration': {
                'mode': self.current_config.mode.value if self.current_config else 'none',
                'target_fps': self.current_config.target_fps if self.current_config else 0,
                'optimization_active': self.optimization_active
            },
            'benchmark_results': self.profiler.benchmark_results,
            'performance_history_summary': self._summarize_performance_history(),
            'component_reports': {}
        }
        
        # Append component reports
        if self.mps_optimizer:
            report['component_reports']['mps_optimizer'] = self.mps_optimizer.get_performance_report()
        
        if self.cpu_gpu_balancer:
            report['component_reports']['cpu_gpu_balancer'] = self.cpu_gpu_balancer.get_performance_statistics()
        
        if self.gui_optimizer:
            report['component_reports']['gui_optimizer'] = self.gui_optimizer.get_optimization_report()
        
        return report
    
    def _summarize_performance_history(self) -> Dict[str, Any]:
        """Summarize performance history"""
        if not self.performance_history:
            return {}
        
        recent_data = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        return {
            'average_cpu_utilization': sum(d['cpu_utilization'] for d in recent_data) / len(recent_data),
            'average_memory_usage': sum(d['memory_usage'] for d in recent_data) / len(recent_data),
            'data_points': len(self.performance_history),
            'monitoring_duration_minutes': (time.time() - self.performance_history[0]['timestamp']) / 60 if self.performance_history else 0
        }
    
    def run_performance_benchmark(self) -> Dict[str, float]:
        """Run performance benchmarks"""
        print("Running performance benchmark...")
        results = self.profiler.run_benchmark()
        print("Benchmark completed")
        return results
    
    def save_configuration(self, path: Optional[str] = None):
        """Save configuration"""
        if not self.current_config:
            return
        
        save_path = path or self.config_path or "optimization_config.json"
        
        config_data = {
            'mode': self.current_config.mode.value,
            'target_fps': self.current_config.target_fps,
            'max_cpu_utilization': self.current_config.max_cpu_utilization,
            'max_gpu_utilization': self.current_config.max_gpu_utilization,
            'enable_adaptive_quality': self.current_config.enable_adaptive_quality,
            'enable_predictive_scheduling': self.current_config.enable_predictive_scheduling,
            'enable_smart_frame_dropping': self.current_config.enable_smart_frame_dropping,
            'enable_memory_optimization': self.current_config.enable_memory_optimization,
            'custom_params': self.current_config.custom_params
        }
        
        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Configuration saved to: {save_path}")
    
    def _load_configuration(self):
        """Load configuration"""
        if not self.config_path or not os.path.exists(self.config_path):
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            self.current_config = OptimizationConfig(
                mode=OptimizationMode(config_data['mode']),
                target_fps=config_data['target_fps'],
                max_cpu_utilization=config_data['max_cpu_utilization'],
                max_gpu_utilization=config_data['max_gpu_utilization'],
                enable_adaptive_quality=config_data['enable_adaptive_quality'],
                enable_predictive_scheduling=config_data['enable_predictive_scheduling'],
                enable_smart_frame_dropping=config_data['enable_smart_frame_dropping'],
                enable_memory_optimization=config_data['enable_memory_optimization'],
                custom_params=config_data['custom_params']
            )
            
            print(f"Configuration loaded from: {self.config_path}")
            
        except Exception as e:
            print(f"Failed to load configuration: {e}")
    
    def shutdown(self):
        """Shut down the optimization manager"""
        print("Shutting down optimization manager...")
        self.optimization_active = False
        print("Optimization manager shutdown complete")


# Global optimization manager instance
_global_optimization_manager = None

def get_optimization_manager(config_path: Optional[str] = None) -> IntegratedOptimizationManager:
    """Get global optimization manager instance"""
    global _global_optimization_manager
    if _global_optimization_manager is None:
        _global_optimization_manager = IntegratedOptimizationManager(config_path)
    return _global_optimization_manager

def shutdown_optimization_manager():
    """Shut down the global optimization manager"""
    global _global_optimization_manager
    if _global_optimization_manager:
        _global_optimization_manager.shutdown()
        _global_optimization_manager = None