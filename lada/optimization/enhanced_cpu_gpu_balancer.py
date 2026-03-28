"""
Enhanced CPU-GPU Balancer for LADA Project

Key improvements:
1. Intelligent task scheduling
2. Dynamic load balancing
3. Predictive resource allocation
4. Optimized asynchronous data transfer
5. Memory bandwidth optimization
"""

import torch
import threading
import time
import queue
import psutil
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import numpy as np
from collections import deque
import weakref


class TaskPriority(Enum):
    """Task priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ProcessingMode(Enum):
    """Processing mode"""
    CPU_ONLY = "cpu_only"
    GPU_ONLY = "gpu_only"
    CPU_GPU_PARALLEL = "cpu_gpu_parallel"
    ADAPTIVE = "adaptive"


@dataclass
class TaskMetrics:
    """Task performance metrics"""
    task_id: str
    processing_time: float
    memory_usage: float
    cpu_utilization: float
    gpu_utilization: float
    data_transfer_time: float
    queue_wait_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class EnhancedTask:
    """Enhanced task definition"""
    task_id: str
    func: Callable
    args: Tuple
    kwargs: Dict
    priority: TaskPriority
    preferred_device: str
    estimated_processing_time: float
    memory_requirement: float
    can_split: bool = False
    dependencies: List[str] = None
    callback: Optional[Callable] = None


class PredictiveScheduler:
    """Predictive task scheduler"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.task_history = deque(maxlen=history_size)
        self.performance_models = {}
        self.lock = threading.Lock()
    
    def record_task_completion(self, task: EnhancedTask, metrics: TaskMetrics):
        """Record task completion"""
        with self.lock:
            self.task_history.append((task, metrics))
            self._update_performance_model(task, metrics)
    
    def _update_performance_model(self, task: EnhancedTask, metrics: TaskMetrics):
        """Update performance model"""
        task_type = type(task.func).__name__
        
        if task_type not in self.performance_models:
            self.performance_models[task_type] = {
                'cpu_times': deque(maxlen=20),
                'gpu_times': deque(maxlen=20),
                'memory_usage': deque(maxlen=20),
                'success_rate': deque(maxlen=20)
            }
        
        model = self.performance_models[task_type]
        
        if 'cpu' in task.preferred_device.lower():
            model['cpu_times'].append(metrics.processing_time)
        else:
            model['gpu_times'].append(metrics.processing_time)
        
        model['memory_usage'].append(metrics.memory_usage)
        model['success_rate'].append(1.0 if metrics.success else 0.0)
    
    def predict_optimal_device(self, task: EnhancedTask, current_cpu_load: float, current_gpu_load: float) -> str:
        """Predict optimal device"""
        task_type = type(task.func).__name__
        
        if task_type not in self.performance_models:
            # No historical data; use default strategy
            return self._default_device_selection(task, current_cpu_load, current_gpu_load)
        
        model = self.performance_models[task_type]
        
        # Compute expected processing time
        cpu_time = np.mean(model['cpu_times']) if model['cpu_times'] else float('inf')
        gpu_time = np.mean(model['gpu_times']) if model['gpu_times'] else float('inf')
        
        # Adjust by current load
        cpu_adjusted_time = cpu_time * (1 + current_cpu_load)
        gpu_adjusted_time = gpu_time * (1 + current_gpu_load)
        
        # Choose the device with shorter adjusted time
        if cpu_adjusted_time < gpu_adjusted_time:
            return 'cpu'
        else:
            return task.preferred_device if 'gpu' in task.preferred_device.lower() else 'mps'
    
    def _default_device_selection(self, task: EnhancedTask, cpu_load: float, gpu_load: float) -> str:
        """Default device selection strategy"""
        # Heuristic selection based on task traits and current load
        if task.memory_requirement > 1000:  # Large-memory task
            return 'cpu' if cpu_load < 0.7 else task.preferred_device
        elif task.estimated_processing_time > 0.1:  # Long-running task
            return task.preferred_device if gpu_load < 0.8 else 'cpu'
        else:
            return 'cpu' if cpu_load < gpu_load else task.preferred_device


class AsyncDataTransferManager:
    """Asynchronous data transfer manager"""
    
    def __init__(self, max_concurrent_transfers: int = 4):
        self.max_concurrent_transfers = max_concurrent_transfers
        self.transfer_executor = ThreadPoolExecutor(max_workers=max_concurrent_transfers)
        self.active_transfers = {}
        self.transfer_queue = queue.PriorityQueue()
        self.lock = threading.Lock()
    
    def schedule_transfer(self, data: torch.Tensor, target_device: str, priority: int = 0) -> Future:
        """Schedule a data transfer"""
        transfer_id = f"transfer_{time.time()}_{id(data)}"
        
        future = self.transfer_executor.submit(
            self._perform_transfer, data, target_device, transfer_id
        )
        
        with self.lock:
            self.active_transfers[transfer_id] = future
        
        return future
    
    def _perform_transfer(self, data: torch.Tensor, target_device: str, transfer_id: str) -> torch.Tensor:
        """Perform data transfer"""
        start_time = time.time()
        
        try:
            if data.device.type == target_device:
                return data
            
            # Use non-blocking transfer
            result = data.to(target_device, non_blocking=True)
            
            # Ensure transfer completion
            if target_device == 'mps':
                torch.mps.synchronize()
            elif target_device == 'cuda':
                torch.cuda.synchronize()
            
            transfer_time = time.time() - start_time
            
            return result
            
        finally:
            with self.lock:
                self.active_transfers.pop(transfer_id, None)
    
    def wait_for_transfers(self, futures: List[Future], timeout: Optional[float] = None) -> List[torch.Tensor]:
        """Wait for transfers to complete"""
        results = []
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                print(f"Data transfer failed: {e}")
                results.append(None)
        return results


class MemoryBandwidthOptimizer:
    """Memory bandwidth optimizer"""
    
    def __init__(self):
        self.bandwidth_history = deque(maxlen=50)
        self.optimal_chunk_size = 1024 * 1024  # 1MB
        self.lock = threading.Lock()
    
    def optimize_data_layout(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimize data layout to improve memory bandwidth utilization"""
        if not tensors:
            return tensors

        # Sort by size; process large tensors first
        sorted_tensors = sorted(tensors, key=lambda t: t.numel(), reverse=True)

        # Merge small tensors to reduce transfer count
        optimized_tensors = []
        small_tensors = []
        
        for tensor in sorted_tensors:
            if tensor.numel() * tensor.element_size() < self.optimal_chunk_size:
                small_tensors.append(tensor)
            else:
                optimized_tensors.append(tensor)
        
        # Try merging small tensors
        if small_tensors:
            merged_tensor = self._merge_small_tensors(small_tensors)
            if merged_tensor is not None:
                optimized_tensors.append(merged_tensor)
            else:
                optimized_tensors.extend(small_tensors)
        
        return optimized_tensors
    
    def _merge_small_tensors(self, tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Merge small tensors"""
        try:
            # Check whether tensors can be merged (same dtype and device)
            if not tensors:
                return None
            
            first_tensor = tensors[0]
            if not all(t.dtype == first_tensor.dtype and t.device == first_tensor.device for t in tensors):
                return None
            
            # Flatten and concatenate
            flattened = [t.flatten() for t in tensors]
            merged = torch.cat(flattened, dim=0)
            
            return merged
            
        except Exception:
            return None
    
    def record_bandwidth_usage(self, data_size: int, transfer_time: float):
        """Record bandwidth usage"""
        bandwidth = data_size / transfer_time if transfer_time > 0 else 0
        
        with self.lock:
            self.bandwidth_history.append(bandwidth)
            
            # Dynamically adjust optimal chunk size
            if len(self.bandwidth_history) >= 10:
                avg_bandwidth = np.mean(list(self.bandwidth_history)[-10:])
                if avg_bandwidth > 0:
                    # Adjust chunk size based on average bandwidth
                    self.optimal_chunk_size = min(
                        max(int(avg_bandwidth * 0.01), 512 * 1024),  # Minimum 512KB
                        4 * 1024 * 1024  # Maximum 4MB
                    )


class EnhancedCPUGPUBalancer:
    """Enhanced CPU–GPU collaborative optimizer"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Thread pool configuration
        cpu_workers = self.config.get('cpu_workers', psutil.cpu_count())
        gpu_workers = self.config.get('gpu_workers', 2)
        
        self.cpu_executor = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="CPU-Worker")
        self.gpu_executor = ThreadPoolExecutor(max_workers=gpu_workers, thread_name_prefix="GPU-Worker")
        
        # Task queues
        self.high_priority_queue = queue.PriorityQueue()
        self.normal_priority_queue = queue.PriorityQueue()
        self.low_priority_queue = queue.PriorityQueue()
        
        # Component initialization
        self.scheduler = PredictiveScheduler(
            history_size=self.config.get('scheduler_history_size', 100)
        )
        self.data_transfer_manager = AsyncDataTransferManager(
            max_concurrent_transfers=self.config.get('max_concurrent_transfers', 4)
        )
        self.bandwidth_optimizer = MemoryBandwidthOptimizer()
        
        # Performance monitoring
        self.task_metrics = {}
        self.resource_usage_history = deque(maxlen=100)
        self.lock = threading.Lock()
        
        # Control parameters
        self.processing_mode = ProcessingMode.ADAPTIVE
        self.enable_predictive_scheduling = self.config.get('enable_predictive_scheduling', True)
        self.enable_bandwidth_optimization = self.config.get('enable_bandwidth_optimization', True)
        
        # Start scheduler
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
    
    def submit_task(self, task: EnhancedTask) -> Future:
        """Submit a task"""
        # Choose the appropriate queue
        if task.priority == TaskPriority.CRITICAL:
            priority_value = 0
            task_queue = self.high_priority_queue
        elif task.priority == TaskPriority.HIGH:
            priority_value = 1
            task_queue = self.high_priority_queue
        elif task.priority == TaskPriority.NORMAL:
            priority_value = 2
            task_queue = self.normal_priority_queue
        else:
            priority_value = 3
            task_queue = self.low_priority_queue

        # Create Future
        future = Future()

        # Enqueue task
        task_queue.put((priority_value, time.time(), task, future))

        return future
    
    def _scheduler_loop(self):
        """Scheduler main loop"""
        while self.running:
            try:
                # Process tasks by priority
                task_item = None
                
                # Check high-priority queue first
                try:
                    task_item = self.high_priority_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Then check normal-priority queue
                if task_item is None:
                    try:
                        task_item = self.normal_priority_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Finally check low-priority queue
                if task_item is None:
                    try:
                        task_item = self.low_priority_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                
                if task_item:
                    priority_value, submit_time, task, future = task_item
                    self._process_task(task, future, submit_time)
                    
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(0.1)
    
    def _process_task(self, task: EnhancedTask, future: Future, submit_time: float):
        """Process a single task"""
        start_time = time.time()
        queue_wait_time = start_time - submit_time
        
        try:
            # Get current resource usage
            cpu_load, gpu_load = self._get_current_resource_usage()
            
            # Select optimal device
            if self.enable_predictive_scheduling:
                optimal_device = self.scheduler.predict_optimal_device(task, cpu_load, gpu_load)
            else:
                optimal_device = task.preferred_device
            
            # Prepare data transfer
            transfer_futures = []
            if self.enable_bandwidth_optimization and task.args:
                optimized_args = self.bandwidth_optimizer.optimize_data_layout(
                    [arg for arg in task.args if isinstance(arg, torch.Tensor)]
                )
                
                # Asynchronously transfer data to target device
                for tensor in optimized_args:
                    if isinstance(tensor, torch.Tensor) and tensor.device.type != optimal_device:
                        transfer_future = self.data_transfer_manager.schedule_transfer(
                            tensor, optimal_device
                        )
                        transfer_futures.append(transfer_future)
            
            # Wait for data transfer completion
            if transfer_futures:
                transferred_data = self.data_transfer_manager.wait_for_transfers(transfer_futures)
                # Update task args
                # Simplified here; real use may require complex mapping
            
            # Select executor
            if 'cpu' in optimal_device.lower():
                executor = self.cpu_executor
            else:
                executor = self.gpu_executor
            
            # Execute task
            execution_future = executor.submit(self._execute_task, task, optimal_device)
            result = execution_future.result()
            
            # Record performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            metrics = TaskMetrics(
                task_id=task.task_id,
                processing_time=processing_time,
                memory_usage=self._estimate_memory_usage(task),
                cpu_utilization=cpu_load,
                gpu_utilization=gpu_load,
                data_transfer_time=sum(f.result() if hasattr(f, 'result') else 0 for f in transfer_futures),
                queue_wait_time=queue_wait_time,
                success=True
            )
            
            self.scheduler.record_task_completion(task, metrics)
            
            # Set result
            future.set_result(result)
            
            # Execute callback
            if task.callback:
                try:
                    task.callback(result, metrics)
                except Exception as e:
                    print(f"Callback error: {e}")
                    
        except Exception as e:
            # Record error
            error_metrics = TaskMetrics(
                task_id=task.task_id,
                processing_time=time.time() - start_time,
                memory_usage=0,
                cpu_utilization=cpu_load if 'cpu_load' in locals() else 0,
                gpu_utilization=gpu_load if 'gpu_load' in locals() else 0,
                data_transfer_time=0,
                queue_wait_time=queue_wait_time,
                success=False,
                error_message=str(e)
            )
            
            self.scheduler.record_task_completion(task, error_metrics)
            future.set_exception(e)
    
    def _execute_task(self, task: EnhancedTask, device: str) -> Any:
        """Execute task"""
        # Set device context
        if device == 'mps' and torch.backends.mps.is_available():
            with torch.device('mps'):
                return task.func(*task.args, **task.kwargs)
        elif device == 'cuda' and torch.cuda.is_available():
            with torch.device('cuda'):
                return task.func(*task.args, **task.kwargs)
        else:
            # CPU execution
            return task.func(*task.args, **task.kwargs)
    
    def _get_current_resource_usage(self) -> Tuple[float, float]:
        """Get current resource usage"""
        cpu_load = psutil.cpu_percent() / 100.0
        
        # GPU utilization estimate (MPS has no direct API)
        if torch.backends.mps.is_available():
            try:
                gpu_memory_used = torch.mps.current_allocated_memory()
                gpu_memory_total = torch.mps.driver_allocated_memory()
                gpu_load = gpu_memory_used / max(gpu_memory_total, 1)
            except:
                gpu_load = 0.5  # Default estimate
        else:
            gpu_load = 0.0
        
        return cpu_load, gpu_load
    
    def _estimate_memory_usage(self, task: EnhancedTask) -> float:
        """Estimate task memory usage"""
        memory_usage = 0
        
        for arg in task.args:
            if isinstance(arg, torch.Tensor):
                memory_usage += arg.numel() * arg.element_size()
        
        return memory_usage
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            total_tasks = len(self.task_metrics)
            successful_tasks = sum(1 for m in self.task_metrics.values() if m.success)
            
            if total_tasks == 0:
                return {
                    'total_tasks': 0,
                    'success_rate': 0.0,
                    'average_processing_time': 0.0,
                    'average_queue_wait_time': 0.0
                }
            
            avg_processing_time = np.mean([m.processing_time for m in self.task_metrics.values()])
            avg_queue_wait_time = np.mean([m.queue_wait_time for m in self.task_metrics.values()])
            
            return {
                'total_tasks': total_tasks,
                'success_rate': successful_tasks / total_tasks,
                'average_processing_time': avg_processing_time,
                'average_queue_wait_time': avg_queue_wait_time,
                'queue_sizes': {
                    'high_priority': self.high_priority_queue.qsize(),
                    'normal_priority': self.normal_priority_queue.qsize(),
                    'low_priority': self.low_priority_queue.qsize()
                },
                'resource_usage': self._get_current_resource_usage()
            }
    
    def set_processing_mode(self, mode: ProcessingMode):
        """Set processing mode"""
        self.processing_mode = mode
    
    def shutdown(self):
        """Shut down the balancer"""
        self.running = False
        
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        self.cpu_executor.shutdown(wait=True)
        self.gpu_executor.shutdown(wait=True)


# Helper function
def create_enhanced_task(
    task_id: str,
    func: Callable,
    args: Tuple = (),
    kwargs: Dict = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    preferred_device: str = 'mps',
    estimated_time: float = 0.1,
    memory_requirement: float = 0.0,
    callback: Optional[Callable] = None
) -> EnhancedTask:
    """Create an enhanced task"""
    return EnhancedTask(
        task_id=task_id,
        func=func,
        args=args,
        kwargs=kwargs or {},
        priority=priority,
        preferred_device=preferred_device,
        estimated_processing_time=estimated_time,
        memory_requirement=memory_requirement,
        callback=callback
    )