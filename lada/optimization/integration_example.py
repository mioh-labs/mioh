"""
Integration Example for LADA Optimization Components

Demonstrates integrating and using optimization components in LADA:
1. Integration with FrameRestorer
2. Integration with GStreamer pipeline
3. Integration with GUI components
4. Performance monitoring and tuning
"""

import torch
import time
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import LADA core components
try:
    from src.frame_restorer import FrameRestorer
    from src.gstreamer_pipeline_manager import GStreamerPipelineManager
    from src.cpu_gpu_balancer import CPUGPUBalancer, Task, TaskType
except ImportError as e:
    print(f"Warning: Could not import LADA core components: {e}")
    print("This example will run in simulation mode")

# Import optimization components
from .integrated_optimization_manager import (
    IntegratedOptimizationManager, 
    OptimizationMode,
    get_optimization_manager
)
from .performance_benchmark import PerformanceBenchmark
from .realtime_gui_optimizer import FrameMetadata


class OptimizedFrameRestorer:
    """Optimized frame restorer"""
    
    def __init__(self, original_restorer=None, optimization_config: Optional[Dict[str, Any]] = None):
        self.original_restorer = original_restorer
        self.optimization_manager = get_optimization_manager()
        
        # Initialize optimization
        mode = OptimizationMode.PERFORMANCE
        if optimization_config:
            mode_str = optimization_config.get('mode', 'performance')
            mode = OptimizationMode(mode_str)
        
        target_fps = optimization_config.get('target_fps', 30.0) if optimization_config else 30.0
        self.optimization_manager.initialize_optimization(mode, target_fps)
        
        print(f"OptimizedFrameRestorer initialized with mode: {mode.value}")
    
    def restore_frame(self, frame: torch.Tensor, frame_id: int = 0) -> Optional[torch.Tensor]:
        """Optimized frame restoration"""
        # Create frame metadata
        metadata = FrameMetadata(
            frame_id=frame_id,
            timestamp=time.time(),
            frame_type='video',
            priority=1.0,
            processing_deadline=time.time() + 0.033  # 30fps deadline
        )
        
        # Process frame via optimization manager
        optimized_frame = self.optimization_manager.process_frame_optimized(frame, metadata)
        
        if optimized_frame is None:
            return None  # Frame intelligently dropped
        
        # If original restorer is present, submit as optimized task
        if self.original_restorer:
            def restore_task():
                return self.original_restorer.restore_frame(optimized_frame)
            
            # Submit task through optimization manager
            result = self.optimization_manager.submit_processing_task(restore_task)
            return result
        else:
            # Simulate frame restoration processing
            return self._simulate_frame_restoration(optimized_frame)
    
    def _simulate_frame_restoration(self, frame: torch.Tensor) -> torch.Tensor:
        """Simulate frame restoration processing"""
        # Simple image enhancement simulation
        enhanced = frame * 1.1
        enhanced = torch.clamp(enhanced, 0, 1)
        
        # Simulate some processing time
        time.sleep(0.01)
        
        return enhanced
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return self.optimization_manager.get_comprehensive_report()
    
    def shutdown(self):
        """Shut down optimizer"""
        self.optimization_manager.shutdown()


class OptimizedGStreamerManager:
    """Optimized GStreamer pipeline manager"""
    
    def __init__(self, original_manager=None):
        self.original_manager = original_manager
        self.optimization_manager = get_optimization_manager()
        
        # Initialize if optimization manager is not active
        if not self.optimization_manager.optimization_active:
            self.optimization_manager.initialize_optimization(
                mode=OptimizationMode.BALANCED,
                target_fps=30.0
            )
        
        self.frame_queue = []
        self.processing_active = False
        
        print("OptimizedGStreamerManager initialized")
    
    def start_pipeline(self, pipeline_config: Dict[str, Any]):
        """Start optimized pipeline"""
        print("Starting optimized GStreamer pipeline...")
        
        # Start processing thread
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start original manager if present
        if self.original_manager:
            self.original_manager.start_pipeline(pipeline_config)
    
    def push_frame(self, frame: torch.Tensor, frame_id: int):
        """Push frame to pipeline"""
        metadata = FrameMetadata(
            frame_id=frame_id,
            timestamp=time.time(),
            frame_type='video',
            priority=1.0,
            processing_deadline=time.time() + 0.033
        )
        
        # Process frame through optimization manager
        optimized_frame = self.optimization_manager.process_frame_optimized(frame, metadata)
        
        if optimized_frame is not None:
            self.frame_queue.append((optimized_frame, frame_id))
    
    def _processing_loop(self):
        """Processing loop"""
        while self.processing_active:
            if self.frame_queue:
                frame, frame_id = self.frame_queue.pop(0)
                
                # Process frame
                def process_frame():
                    # Simulate GStreamer processing
                    time.sleep(0.005)  # 5ms processing time
                    return frame
                
                # Submit processing task via optimization manager
                result = self.optimization_manager.submit_processing_task(process_frame)
                
                # Output processed result (connect to actual GStreamer sink here)
                if result is not None:
                    self._output_frame(result, frame_id)
            else:
                time.sleep(0.001)  # Short wait
    
    def _output_frame(self, frame: torch.Tensor, frame_id: int):
        """Output frame"""
        # Connect to actual GStreamer output here
        print(f"Frame {frame_id} processed and output")
    
    def stop_pipeline(self):
        """Stop pipeline"""
        print("Stopping optimized GStreamer pipeline...")
        self.processing_active = False
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
        
        if self.original_manager:
            self.original_manager.stop_pipeline()


class OptimizedVideoProcessor:
    """Optimized video processor — complete example"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize optimization components
        self.frame_restorer = OptimizedFrameRestorer(
            optimization_config=self.config.get('optimization', {})
        )
        self.gstreamer_manager = OptimizedGStreamerManager()
        
        # Performance monitoring
        self.performance_monitor = PerformanceBenchmark("video_processing_results")
        self.processing_stats = {
            'frames_processed': 0,
            'frames_dropped': 0,
            'total_processing_time': 0.0,
            'start_time': None
        }
        
        print("OptimizedVideoProcessor initialized")
    
    def process_video_stream(self, video_frames: List[torch.Tensor], show_progress: bool = True):
        """Process video stream"""
        print(f"Processing video stream with {len(video_frames)} frames...")
        
        self.processing_stats['start_time'] = time.time()
        
        # Start GStreamer pipeline
        pipeline_config = {
            'format': 'RGB',
            'width': video_frames[0].shape[-1] if video_frames else 640,
            'height': video_frames[0].shape[-2] if video_frames else 480,
            'framerate': 30
        }
        self.gstreamer_manager.start_pipeline(pipeline_config)
        
        try:
            for i, frame in enumerate(video_frames):
                frame_start_time = time.time()
                
                # Frame restoration
                restored_frame = self.frame_restorer.restore_frame(frame, frame_id=i)
                
                if restored_frame is not None:
                    # Push to GStreamer pipeline
                    self.gstreamer_manager.push_frame(restored_frame, i)
                    self.processing_stats['frames_processed'] += 1
                else:
                    self.processing_stats['frames_dropped'] += 1
                
                frame_end_time = time.time()
                self.processing_stats['total_processing_time'] += (frame_end_time - frame_start_time)
                
                # Show progress
                if show_progress and (i + 1) % 10 == 0:
                    progress = (i + 1) / len(video_frames) * 100
                    fps = self.processing_stats['frames_processed'] / (time.time() - self.processing_stats['start_time'])
                    print(f"Progress: {progress:.1f}% | FPS: {fps:.1f} | Dropped: {self.processing_stats['frames_dropped']}")
                
                # Simulate real-time processing interval
                time.sleep(0.001)
        
        finally:
            # Stop pipeline
            self.gstreamer_manager.stop_pipeline()
            
            # Generate processing report
            self._generate_processing_report()
    
    def _generate_processing_report(self):
        """Generate processing report"""
        total_time = time.time() - self.processing_stats['start_time']
        avg_fps = self.processing_stats['frames_processed'] / total_time if total_time > 0 else 0
        drop_rate = (self.processing_stats['frames_dropped'] / 
                    (self.processing_stats['frames_processed'] + self.processing_stats['frames_dropped'])) * 100
        
        print("\nVideo Processing Report:")
        print("=" * 40)
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Frames Processed: {self.processing_stats['frames_processed']}")
        print(f"Frames Dropped: {self.processing_stats['frames_dropped']}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Drop Rate: {drop_rate:.2f}%")
        print(f"Avg Processing Time per Frame: {self.processing_stats['total_processing_time'] / self.processing_stats['frames_processed'] * 1000:.2f}ms")
        
        # Get optimization report
        optimization_report = self.frame_restorer.get_performance_report()
        print(f"\nOptimization Efficiency Score: {optimization_report.get('efficiency_score', 'N/A')}")
    
    def run_performance_comparison(self, test_frames: List[torch.Tensor]):
        """Run performance comparison tests"""
        print("Running performance comparison...")
        
        # Create benchmark runner
        benchmark = PerformanceBenchmark("comparison_results")
        
        # Run comparison tests
        results = benchmark.run_comprehensive_benchmark()
        
        print("Performance comparison completed!")
        return results
    
    def shutdown(self):
        """Shut down processor"""
        print("Shutting down OptimizedVideoProcessor...")
        self.frame_restorer.shutdown()
        self.gstreamer_manager.stop_pipeline()


def create_test_video_frames(count: int = 100, size: tuple = (720, 480)) -> List[torch.Tensor]:
    """Create test video frames"""
    frames = []
    height, width = size
    
    for i in range(count):
        # Create gradient test pattern
        frame = torch.zeros(3, height, width)
        
        # Red channel gradient
        frame[0] = torch.linspace(0, 1, width).unsqueeze(0).repeat(height, 1)
        
        # Green channel gradient
        frame[1] = torch.linspace(0, 1, height).unsqueeze(1).repeat(1, width)
        
        # Blue channel varies over time
        frame[2] = (i / count) * torch.ones(height, width)
        
        # Add some noise
        noise = torch.randn_like(frame) * 0.05
        frame = torch.clamp(frame + noise, 0, 1)
        
        frames.append(frame)
    
    return frames


def main():
    """Main function — complete integration example"""
    print("LADA Optimization Integration Example")
    print("=" * 50)
    
    # Create test data
    print("Creating test video frames...")
    test_frames = create_test_video_frames(count=50, size=(720, 480))
    
    # Configure optimization parameters
    optimization_config = {
        'optimization': {
            'mode': 'performance',
            'target_fps': 30.0
        }
    }
    
    # Create optimized video processor
    processor = OptimizedVideoProcessor(optimization_config)
    
    try:
        # Process video stream
        print("\nProcessing video stream with optimization...")
        processor.process_video_stream(test_frames, show_progress=True)
        
        # Run performance comparison (optional)
        print("\nRunning performance comparison...")
        comparison_results = processor.run_performance_comparison(test_frames[:20])  # Use fewer frames for quick test
        
        # Show comparison summary
        if comparison_results:
            avg_improvement = sum(r.performance_gain_percent for r in comparison_results.values()) / len(comparison_results)
            print(f"\nAverage Performance Improvement: {avg_improvement:.1f}%")
            
            for test_name, result in comparison_results.items():
                print(f"{test_name}: {result.performance_gain_percent:.1f}% improvement")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up resources
        processor.shutdown()
        print("\nExample completed!")


def demo_individual_components():
    """Demonstrate standalone use of each optimization component"""
    print("\nDemonstrating Individual Optimization Components")
    print("=" * 60)
    
    # 1. MPS optimizer demo
    print("\n1. MPS Performance Optimizer Demo:")
    from .mps_performance_optimizer import get_mps_optimizer
    
    mps_optimizer = get_mps_optimizer()
    test_tensors = [torch.randn(100, 100) for _ in range(5)]
    
    start_time = time.time()
    optimized_tensors = mps_optimizer.optimize_tensor_operations(test_tensors)
    end_time = time.time()
    
    print(f"   Processed {len(test_tensors)} tensors in {(end_time - start_time)*1000:.2f}ms")
    print(f"   MPS optimization report: {mps_optimizer.get_performance_report()}")
    
    # 2. Enhanced CPU–GPU balancer demo
    print("\n2. Enhanced CPU-GPU Balancer Demo:")
    from .enhanced_cpu_gpu_balancer import EnhancedCPUGPUBalancer, EnhancedTask, TaskPriority
    
    balancer_config = {
        'cpu_workers': 4,
        'gpu_workers': 2,
        'enable_predictive_scheduling': True
    }
    balancer = EnhancedCPUGPUBalancer(balancer_config)
    
    # Submit a test task
    def test_task(x):
        return x * 2
    
    task = EnhancedTask(
        task_id="demo_task",
        func=test_task,
        args=(42,),
        priority=TaskPriority.HIGH,
        preferred_device='cpu'
    )
    
    future = balancer.submit_task(task)
    result = future.result()
    print(f"   Task result: {result}")
    print(f"   Balancer stats: {balancer.get_performance_statistics()}")
    
    balancer.shutdown()
    
    # 3. Realtime GUI optimizer demo
    print("\n3. Realtime GUI Optimizer Demo:")
    from .realtime_gui_optimizer import RealtimeGUIOptimizer, FrameMetadata
    
    gui_config = {
        'target_fps': 30.0,
        'enable_adaptive_quality': True,
        'enable_smart_dropping': True
    }
    gui_optimizer = RealtimeGUIOptimizer(gui_config)
    
    # Process a test frame
    test_frame = torch.randn(3, 480, 640)
    metadata = FrameMetadata(
        frame_id=1,
        timestamp=time.time(),
        frame_type='video',
        priority=1.0
    )
    
    processed_frame = gui_optimizer.process_frame(test_frame, metadata)
    if processed_frame is not None:
        print(f"   Frame processed successfully, shape: {processed_frame.shape}")
    else:
        print("   Frame was dropped by smart dropping algorithm")
    
    print(f"   GUI optimization report: {gui_optimizer.get_optimization_report()}")
    
    gui_optimizer.shutdown()
    
    print("\nIndividual component demonstration completed!")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Demonstrate each component
    demo_individual_components()