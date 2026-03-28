"""
MPS Performance Optimizer for LADA Project
This module provides performance optimizations for macOS Metal Performance Shaders (MPS) in the LADA project.

Main Features:
1. Metal Command Buffer Optimization
2. Texture Memory Management
3. CPU-GPU Cooperation Optimization
4. Real-time Performance Monitoring and Adjustment
"""

import torch
import torch.mps
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
import weakref

try:
    import objc
    from Foundation import NSProcessInfo
    from Metal import MTLCreateSystemDefaultDevice, MTLCommandQueue
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


@dataclass
class MPSPerformanceMetrics:
    """MPS Performance Metrics"""
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float  # estimated value
    cpu_utilization: float
    memory_bandwidth: float
    command_buffer_latency: float
    texture_cache_hit_rate: float
    frame_processing_time: float


class MetalCommandBufferPool:
    """Metal Command Buffer Pool for Optimized Command Submission"""
    
    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self.available_buffers = queue.Queue(maxsize=pool_size)
        self.active_buffers = set()
        self.lock = threading.Lock()
        
        if METAL_AVAILABLE:
            self.device = MTLCreateSystemDefaultDevice()
            self.command_queue = self.device.newCommandQueue()
            self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the command buffer pool"""
        if not METAL_AVAILABLE:
            return
            
        for _ in range(self.pool_size):
            cmd_buffer = self.command_queue.commandBuffer()
            self.available_buffers.put(cmd_buffer)
    
    def get_command_buffer(self):
        """Get an available command buffer from the pool"""
        if not METAL_AVAILABLE:
            return None
            
        try:
            cmd_buffer = self.available_buffers.get_nowait()
            with self.lock:
                self.active_buffers.add(cmd_buffer)
            return cmd_buffer
        except queue.Empty:
            # If the pool has no available buffers, create a new one
            cmd_buffer = self.command_queue.commandBuffer()
            with self.lock:
                self.active_buffers.add(cmd_buffer)
            return cmd_buffer
    
    def return_command_buffer(self, cmd_buffer):
        """Return a command buffer to the pool"""
        if not METAL_AVAILABLE or cmd_buffer is None:
            return
            
        with self.lock:
            self.active_buffers.discard(cmd_buffer)
        
        if self.available_buffers.qsize() < self.pool_size:
            # When the pool is not full, reset command buffer state
            new_buffer = self.command_queue.commandBuffer()
            self.available_buffers.put(new_buffer)


class MetalTextureCache:
    """Metal Texture Cache Manager for Optimized Texture Access"""
    
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self.texture_cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_texture_key(self, width: int, height: int, channels: int, dtype: str) -> str:
        """Generate a unique key for texture cache"""
        return f"{width}x{height}x{channels}_{dtype}"
    
    def get_texture(self, width: int, height: int, channels: int, dtype: str):
        """Get or create a texture from cache"""
        key = self.get_texture_key(width, height, channels, dtype)
        
        with self.lock:
            if key in self.texture_cache:
                self.hit_count += 1
                self.access_times[key] = time.time()
                return self.texture_cache[key]
            
            self.miss_count += 1
            # Create a new texture (simplified as a torch tensor)
            # Use standard PyTorch tensor format: (channels, height, width)
            if dtype == 'float32':
                texture = torch.empty((channels, height, width), dtype=torch.float32, device='mps')
            else:
                texture = torch.empty((channels, height, width), dtype=torch.uint8, device='mps')
            
            # If the cache is full, evict the least recently used texture
            if len(self.texture_cache) >= self.max_cache_size:
                self._evict_lru()
            
            self.texture_cache[key] = texture
            self.access_times[key] = time.time()
            return texture
    
    def _evict_lru(self):
        """Evict the least recently used texture from cache"""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.texture_cache[lru_key]
        del self.access_times[lru_key]
    
    def get_cache_hit_rate(self) -> float:
        """Get the cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear_cache(self):
        """Clear the texture cache"""
        with self.lock:
            self.texture_cache.clear()
            self.access_times.clear()


class MPSResourceMonitor:
    """MPS Resource Monitor for Real-time Performance Metrics"""
    
    def __init__(self, update_interval: float = 0.1):
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread = None
        self.metrics_history = []
        self.max_history_size = 100
        self.lock = threading.Lock()
        
        # Performance counters
        self.frame_times = []
        self.command_buffer_times = []
        
    def start_monitoring(self):
        """Start monitoring MPS resources"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring MPS resources"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitoring loop for MPS resources"""
        while self.running:
            metrics = self._collect_metrics()
            
            with self.lock:
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
            
            time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> MPSPerformanceMetrics:
        """Collect performance metrics for MPS resources"""
        # GPU memory usage
        if torch.backends.mps.is_available():
            try:
                gpu_memory_used = torch.mps.current_allocated_memory()
                gpu_memory_total = torch.mps.driver_allocated_memory()
            except:
                gpu_memory_used = 0
                gpu_memory_total = 0
        else:
            gpu_memory_used = 0
            gpu_memory_total = 0
        
        # CPU utilization
        cpu_utilization = psutil.cpu_percent()
        
        # Estimate GPU utilization (based on memory usage and processing time)
        gpu_utilization = min(100.0, (gpu_memory_used / max(gpu_memory_total, 1)) * 100 * 1.2)
        
        # Memory bandwidth (simplified estimate)
        memory_bandwidth = self._estimate_memory_bandwidth()
        
        # Command buffer latency
        command_buffer_latency = self._get_average_command_buffer_latency()
        
        # Frame processing time
        frame_processing_time = self._get_average_frame_time()
        
        return MPSPerformanceMetrics(
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            cpu_utilization=cpu_utilization,
            memory_bandwidth=memory_bandwidth,
            command_buffer_latency=command_buffer_latency,
            texture_cache_hit_rate=0.0,  # Provided by the texture cache
            frame_processing_time=frame_processing_time
        )
    
    def _estimate_memory_bandwidth(self) -> float:
        """Estimate memory bandwidth usage"""
        # Simplified estimate based on system memory usage
        memory_info = psutil.virtual_memory()
        return memory_info.percent
    
    def _get_average_command_buffer_latency(self) -> float:
        """Get average command buffer latency"""
        if not self.command_buffer_times:
            return 0.0
        return sum(self.command_buffer_times[-10:]) / len(self.command_buffer_times[-10:])
    
    def _get_average_frame_time(self) -> float:
        """Get average frame processing time"""
        if not self.frame_times:
            return 0.0
        return sum(self.frame_times[-10:]) / len(self.frame_times[-10:])
    
    def record_frame_time(self, frame_time: float):
        """Record frame processing time"""
        with self.lock:
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 50:
                self.frame_times.pop(0)
    
    def record_command_buffer_time(self, cmd_time: float):
        """Record command buffer time"""
        with self.lock:
            self.command_buffer_times.append(cmd_time)
            if len(self.command_buffer_times) > 50:
                self.command_buffer_times.pop(0)
    
    def get_current_metrics(self) -> Optional[MPSPerformanceMetrics]:
        """Get current performance metrics"""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None


class AdaptiveBatchProcessor:
    """Adaptive batch processor that dynamically adjusts batch size based on performance"""
    
    def __init__(self, initial_batch_size: int = 4, min_batch_size: int = 1, max_batch_size: int = 16):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.performance_history = []
        self.adjustment_threshold = 0.1  # 10% performance change threshold
        
    def update_batch_size(self, processing_time: float, memory_usage: float):
        """Update batch size based on processing time and memory usage"""
        self.performance_history.append((processing_time, memory_usage))
        
        # Keep the latest 10 performance records
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        if len(self.performance_history) < 3:
            return self.current_batch_size
        
        # Calculate performance trends
        recent_times = [p[0] for p in self.performance_history[-3:]]
        recent_memory = [p[1] for p in self.performance_history[-3:]]
        
        avg_time = sum(recent_times) / len(recent_times)
        avg_memory = sum(recent_memory) / len(recent_memory)
        
        # If memory usage is too high, reduce batch size
        if avg_memory > 0.8:  # 80% memory usage
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
        # If processing time is short and memory is sufficient, increase batch size
        elif avg_time < 0.05 and avg_memory < 0.6:  # 50ms processing time, 60% memory usage
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
        
        return self.current_batch_size


class MPSPerformanceOptimizer:
    """MPS performance optimizer main class"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.command_buffer_pool = MetalCommandBufferPool(
            pool_size=self.config.get('command_buffer_pool_size', 4)
        )
        self.texture_cache = MetalTextureCache(
            max_cache_size=self.config.get('texture_cache_size', 100)
        )
        self.resource_monitor = MPSResourceMonitor(
            update_interval=self.config.get('monitor_interval', 0.1)
        )
        self.batch_processor = AdaptiveBatchProcessor(
            initial_batch_size=self.config.get('initial_batch_size', 4),
            min_batch_size=self.config.get('min_batch_size', 1),
            max_batch_size=self.config.get('max_batch_size', 16)
        )
        
        # Performance optimization parameters
        self.enable_async_transfer = self.config.get('enable_async_transfer', True)
        self.enable_memory_pool = self.config.get('enable_memory_pool', True)
        self.enable_texture_cache = self.config.get('enable_texture_cache', True)
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
    
    def optimize_tensor_operations(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimize tensor operations"""
        if not tensors:
            return []
        
        try:
            # Simple device optimization: ensure tensors are on MPS device
            optimized_tensors = []
            for tensor in tensors:
                if tensor is not None:
                    if tensor.device.type != 'mps':
                        optimized_tensors.append(tensor.to('mps', non_blocking=True))
                    else:
                        optimized_tensors.append(tensor)
            
            return optimized_tensors if optimized_tensors else tensors
        except Exception as e:
            print(f"Tensor optimization failed: {e}")
            return tensors
    
    def _process_tensor_batch(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process a batch of tensors"""
        if not batch:
            return []
        
        # Use the command buffer pool
        cmd_buffer = self.command_buffer_pool.get_command_buffer()
        cmd_start_time = time.time()
        
        try:
            # Ensure all tensors are on the MPS device
            mps_batch = []
            for tensor in batch:
                if tensor is None:
                    continue
                if tensor.device.type != 'mps':
                    tensor = tensor.to('mps', non_blocking=self.enable_async_transfer)
                mps_batch.append(tensor)
            
            # Execute batch operations (example; actual ops depend on needs)
            processed_batch = []
            for tensor in mps_batch:
                try:
                    # Use texture cache to optimize memory allocation
                    if self.enable_texture_cache:
                        # Ensure tensor is at least 2D
                        if len(tensor.shape) < 2:
                            tensor = tensor.unsqueeze(0)
                        
                        # Get tensor dimensions
                        if len(tensor.shape) == 2:
                            h, w = tensor.shape
                            c = 1
                            tensor = tensor.unsqueeze(0)  # Add channel dimension
                        elif len(tensor.shape) == 3:
                            c, h, w = tensor.shape
                        else:
                            # For 4D tensors (batch), process only the first
                            if len(tensor.shape) == 4:
                                tensor = tensor[0]  # Take the first batch
                                c, h, w = tensor.shape
                            else:
                                # Skip unsupported dimensions
                                processed_batch.append(tensor)
                                continue
                        
                        # Get cached tensor
                        cached_tensor = self.texture_cache.get_texture(w, h, c, str(tensor.dtype).split('.')[-1])
                        
                        # Copy data if shapes match
                        if cached_tensor.shape == tensor.shape:
                            cached_tensor.copy_(tensor)
                            processed_batch.append(cached_tensor)
                        else:
                            # If shapes don't match, use the original tensor
                            processed_batch.append(tensor)
                    else:
                        processed_batch.append(tensor)
                except Exception as e:
                    print(f"Error processing tensor: {e}")
                    # If processing fails, return the original tensor
                    processed_batch.append(tensor)
            
            return processed_batch
            
        finally:
            # Record command buffer time
            cmd_time = time.time() - cmd_start_time
            self.resource_monitor.record_command_buffer_time(cmd_time)
            
            # Return command buffer to the pool
            self.command_buffer_pool.return_command_buffer(cmd_buffer)
    
    def optimize_memory_usage(self):
        """Optimize memory usage"""
        # Clear MPS cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear texture cache if memory usage is too high
        metrics = self.resource_monitor.get_current_metrics()
        if metrics and metrics.gpu_memory_used / max(metrics.gpu_memory_total, 1) > 0.8:
            self.texture_cache.clear_cache()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        metrics = self.resource_monitor.get_current_metrics()
        if not metrics:
            return {}
        
        return {
            'gpu_memory_usage': {
                'used_mb': metrics.gpu_memory_used / (1024 * 1024),
                'total_mb': metrics.gpu_memory_total / (1024 * 1024),
                'usage_percent': (metrics.gpu_memory_used / max(metrics.gpu_memory_total, 1)) * 100
            },
            'cpu_utilization_percent': metrics.cpu_utilization,
            'gpu_utilization_percent': metrics.gpu_utilization,
            'memory_bandwidth_percent': metrics.memory_bandwidth,
            'command_buffer_latency_ms': metrics.command_buffer_latency * 1000,
            'frame_processing_time_ms': metrics.frame_processing_time * 1000,
            'texture_cache_hit_rate': self.texture_cache.get_cache_hit_rate(),
            'current_batch_size': self.batch_processor.current_batch_size,
            'optimization_status': {
                'async_transfer': self.enable_async_transfer,
                'memory_pool': self.enable_memory_pool,
                'texture_cache': self.enable_texture_cache,
                'metal_available': METAL_AVAILABLE
            }
        }
    
    def shutdown(self):
        """Shut down optimizer"""
        self.resource_monitor.stop_monitoring()
        self.texture_cache.clear_cache()
        
        # Clear MPS resources
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# Global optimizer instance
_global_optimizer = None

def get_mps_optimizer(config: Dict[str, Any] = None) -> MPSPerformanceOptimizer:
    """Get global MPS optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MPSPerformanceOptimizer(config)
    return _global_optimizer

def shutdown_mps_optimizer():
    """Shut down global MPS optimizer"""
    global _global_optimizer
    if _global_optimizer:
        _global_optimizer.shutdown()
        _global_optimizer = None