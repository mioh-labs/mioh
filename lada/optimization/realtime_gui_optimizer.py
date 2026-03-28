"""
Real-time GUI Optimizer for LADA Project

Features:
- Smart frame dropping and buffer management
- GStreamer pipeline optimization
- Render priority management
- Adaptive quality adjustment
- Latency compensation
"""

import torch
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import weakref


class PlaybackQuality(Enum):
    """Playback quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class FrameDropStrategy(Enum):
    """Frame drop strategies"""
    NONE = "none"
    SIMPLE = "simple"          # Simple dropping
    SMART = "smart"            # Smart dropping
    ADAPTIVE = "adaptive"      # Adaptive dropping


@dataclass
class FrameMetadata:
    """Frame metadata"""
    frame_id: int
    timestamp: float
    processing_time: float
    quality_level: PlaybackQuality
    is_keyframe: bool
    size_bytes: int
    resolution: Tuple[int, int]


@dataclass
class PlaybackMetrics:
    """Playback performance metrics"""
    fps: float
    dropped_frames: int
    average_latency: float
    buffer_utilization: float
    gpu_utilization: float
    cpu_utilization: float
    memory_usage: float
    quality_score: float


class AdaptiveQualityController:
    """Adaptive quality controller"""
    
    def __init__(self, target_fps: float = 30.0, quality_adjustment_interval: float = 1.0):
        self.target_fps = target_fps
        self.quality_adjustment_interval = quality_adjustment_interval
        self.current_quality = PlaybackQuality.HIGH
        
        # Performance history
        self.fps_history = deque(maxlen=30)
        self.latency_history = deque(maxlen=30)
        self.last_adjustment_time = time.time()
        
        # Quality parameter mapping
        self.quality_params = {
            PlaybackQuality.LOW: {
                'scale_factor': 0.5,
                'compression_quality': 0.6,
                'processing_threads': 1,
                'enable_gpu_acceleration': False
            },
            PlaybackQuality.MEDIUM: {
                'scale_factor': 0.75,
                'compression_quality': 0.75,
                'processing_threads': 2,
                'enable_gpu_acceleration': True
            },
            PlaybackQuality.HIGH: {
                'scale_factor': 1.0,
                'compression_quality': 0.85,
                'processing_threads': 3,
                'enable_gpu_acceleration': True
            },
            PlaybackQuality.ULTRA: {
                'scale_factor': 1.0,
                'compression_quality': 0.95,
                'processing_threads': 4,
                'enable_gpu_acceleration': True
            }
        }
    
    def update_performance_metrics(self, fps: float, latency: float):
        """Update performance metrics"""
        self.fps_history.append(fps)
        self.latency_history.append(latency)
        
        # Check whether quality adjustment is needed
        current_time = time.time()
        if current_time - self.last_adjustment_time >= self.quality_adjustment_interval:
            self._adjust_quality()
            self.last_adjustment_time = current_time
    
    def _adjust_quality(self):
        """Adjust playback quality"""
        if len(self.fps_history) < 5:
            return
        
        avg_fps = np.mean(list(self.fps_history)[-5:])
        avg_latency = np.mean(list(self.latency_history)[-5:])
        
        # Quality adjustment logic
        if avg_fps < self.target_fps * 0.8:  # FPS below 80% of target
            self._decrease_quality()
        elif avg_fps > self.target_fps * 0.95 and avg_latency < 0.1:  # FPS near target and low latency
            self._increase_quality()
    
    def _decrease_quality(self):
        """Decrease quality"""
        quality_levels = list(PlaybackQuality)
        current_index = quality_levels.index(self.current_quality)
        
        if current_index > 0:
            self.current_quality = quality_levels[current_index - 1]
            print(f"Quality decreased to: {self.current_quality.value}")
    
    def _increase_quality(self):
        """Increase quality"""
        quality_levels = list(PlaybackQuality)
        current_index = quality_levels.index(self.current_quality)
        
        if current_index < len(quality_levels) - 1:
            self.current_quality = quality_levels[current_index + 1]
            print(f"Quality increased to: {self.current_quality.value}")
    
    def get_current_params(self) -> Dict[str, Any]:
        """Get current quality parameters"""
        return self.quality_params[self.current_quality].copy()


class SmartFrameDropper:
    """Smart frame dropper"""
    
    def __init__(self, strategy: FrameDropStrategy = FrameDropStrategy.ADAPTIVE):
        self.strategy = strategy
        self.frame_importance_cache = {}
        self.drop_threshold = 0.5
        self.consecutive_drops = 0
        self.max_consecutive_drops = 3
    
    def should_drop_frame(self, frame_metadata: FrameMetadata, buffer_size: int, target_buffer_size: int) -> bool:
        """Determine whether a frame should be dropped"""
        if self.strategy == FrameDropStrategy.NONE:
            return False
        
        # Compute buffer pressure
        buffer_pressure = buffer_size / max(target_buffer_size, 1)
        
        if self.strategy == FrameDropStrategy.SIMPLE:
            return self._simple_drop_decision(buffer_pressure)
        elif self.strategy == FrameDropStrategy.SMART:
            return self._smart_drop_decision(frame_metadata, buffer_pressure)
        elif self.strategy == FrameDropStrategy.ADAPTIVE:
            return self._adaptive_drop_decision(frame_metadata, buffer_pressure)
        
        return False
    
    def _simple_drop_decision(self, buffer_pressure: float) -> bool:
        """Simple drop decision"""
        return buffer_pressure > 0.8
    
    def _smart_drop_decision(self, frame_metadata: FrameMetadata, buffer_pressure: float) -> bool:
        """Smart drop decision"""
        # Do not drop keyframes
        if frame_metadata.is_keyframe:
            return False
        
        # Decide based on frame importance
        importance = self._calculate_frame_importance(frame_metadata)
        drop_probability = buffer_pressure * (1 - importance)
        
        return drop_probability > self.drop_threshold
    
    def _adaptive_drop_decision(self, frame_metadata: FrameMetadata, buffer_pressure: float) -> bool:
        """Adaptive drop decision"""
        # Do not drop keyframes
        if frame_metadata.is_keyframe:
            self.consecutive_drops = 0
            return False
        
        # Avoid dropping too many frames consecutively
        if self.consecutive_drops >= self.max_consecutive_drops:
            self.consecutive_drops = 0
            return False
        
        # Composite decision based on multiple factors
        importance = self._calculate_frame_importance(frame_metadata)
        time_pressure = min(1.0, frame_metadata.processing_time / 0.033)  # 33ms frame time for 30fps
        
        drop_score = (buffer_pressure * 0.4 + time_pressure * 0.3 + (1 - importance) * 0.3)
        
        should_drop = drop_score > self.drop_threshold
        
        if should_drop:
            self.consecutive_drops += 1
        else:
            self.consecutive_drops = 0
        
        return should_drop
    
    def _calculate_frame_importance(self, frame_metadata: FrameMetadata) -> float:
        """Calculate frame importance"""
        importance = 0.5  # Base importance
        
        # Keyframes are more important
        if frame_metadata.is_keyframe:
            importance += 0.3
        
        # High-quality frames are more important
        if frame_metadata.quality_level in [PlaybackQuality.HIGH, PlaybackQuality.ULTRA]:
            importance += 0.2
        
        # Short processing time frames are more important (easier to process)
        if frame_metadata.processing_time < 0.02:  # 20ms
            importance += 0.1
        
        return min(1.0, importance)


class GStreamerPipelineOptimizer:
    """GStreamer pipeline optimizer"""
    
    def __init__(self):
        self.optimal_buffer_sizes = {
            'video_queue': 10,
            'audio_queue': 20,
            'decode_queue': 5
        }
        self.pipeline_params = {}
    
    def optimize_pipeline_parameters(self, video_info: Dict[str, Any], performance_metrics: PlaybackMetrics) -> Dict[str, Any]:
        """Optimize pipeline parameters"""
        optimized_params = {}
        
        # Adjust buffer sizes based on video information
        resolution = video_info.get('resolution', (1920, 1080))
        fps = video_info.get('fps', 30)
        
        # Calculate data rate
        data_rate = resolution[0] * resolution[1] * fps * 3  # Assume RGB
        
        # Adjust buffer sizes
        if data_rate > 200_000_000:  # High data rate
            optimized_params['max-size-buffers'] = 5
            optimized_params['max-size-bytes'] = 10 * 1024 * 1024
        else:
            optimized_params['max-size-buffers'] = 10
            optimized_params['max-size-bytes'] = 20 * 1024 * 1024
        
        # Adjust based on performance
        if performance_metrics.fps < 25:
            optimized_params['leaky'] = 'downstream'  # Enable leaky mode
        
        if performance_metrics.average_latency > 0.1:
            optimized_params['max-size-time'] = 100_000_000  # 100ms
        
        return optimized_params
    
    def get_optimal_appsrc_config(self, quality: PlaybackQuality) -> Dict[str, Any]:
        """Get optimal appsrc configuration"""
        base_config = {
            'is-live': True,
            'do-timestamp': True,
            'format': 'time'
        }
        
        if quality == PlaybackQuality.LOW:
            base_config.update({
                'max-bytes': 5 * 1024 * 1024,  # 5MB
                'block': False
            })
        elif quality == PlaybackQuality.MEDIUM:
            base_config.update({
                'max-bytes': 10 * 1024 * 1024,  # 10MB
                'block': True
            })
        else:  # HIGH or ULTRA
            base_config.update({
                'max-bytes': 20 * 1024 * 1024,  # 20MB
                'block': True
            })
        
        return base_config


class LatencyCompensator:
    """Latency compensator"""
    
    def __init__(self, target_latency: float = 0.05):  # 50ms target latency
        self.target_latency = target_latency
        self.latency_history = deque(maxlen=50)
        self.compensation_offset = 0.0
        self.last_update_time = time.time()
    
    def update_latency(self, measured_latency: float):
        """Update latency measurements"""
        self.latency_history.append(measured_latency)
        
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0:  # Update every second
            self._calculate_compensation()
            self.last_update_time = current_time
    
    def _calculate_compensation(self):
        """Calculate latency compensation"""
        if len(self.latency_history) < 10:
            return
        
        avg_latency = np.mean(list(self.latency_history)[-10:])
        latency_variance = np.var(list(self.latency_history)[-10:])
        
        # Compute compensation offset
        if avg_latency > self.target_latency:
            # High latency — increase compensation
            self.compensation_offset = min(0.02, avg_latency - self.target_latency)
        else:
            # Normal latency — decrease compensation
            self.compensation_offset = max(0.0, self.compensation_offset - 0.005)
    
    def get_compensated_timestamp(self, original_timestamp: float) -> float:
        """Get compensated timestamp"""
        return original_timestamp - self.compensation_offset


class RealtimeGUIOptimizer:
    """Realtime GUI optimizer main class"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.quality_controller = AdaptiveQualityController(
            target_fps=self.config.get('target_fps', 30.0),
            quality_adjustment_interval=self.config.get('quality_adjustment_interval', 1.0)
        )
        
        self.frame_dropper = SmartFrameDropper(
            strategy=FrameDropStrategy(self.config.get('frame_drop_strategy', 'adaptive'))
        )
        
        self.pipeline_optimizer = GStreamerPipelineOptimizer()
        
        self.latency_compensator = LatencyCompensator(
            target_latency=self.config.get('target_latency', 0.05)
        )
        
        # Performance monitoring
        self.performance_metrics = PlaybackMetrics(
            fps=0.0,
            dropped_frames=0,
            average_latency=0.0,
            buffer_utilization=0.0,
            gpu_utilization=0.0,
            cpu_utilization=0.0,
            memory_usage=0.0,
            quality_score=0.0
        )
        
        # Frame buffer
        self.frame_buffer = queue.Queue(maxsize=self.config.get('max_buffer_size', 30))
        self.processed_frames = 0
        self.dropped_frames = 0
        
        # Control parameters
        self.enable_adaptive_quality = self.config.get('enable_adaptive_quality', True)
        self.enable_smart_dropping = self.config.get('enable_smart_dropping', True)
        self.enable_latency_compensation = self.config.get('enable_latency_compensation', True)
        
        # Thread control
        self.running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
    
    def process_frame(self, frame_data: torch.Tensor, frame_metadata: FrameMetadata) -> Optional[torch.Tensor]:
        """Process a single frame"""
        start_time = time.time()
        
        # Check whether the frame should be dropped
        if self.enable_smart_dropping:
            buffer_size = self.frame_buffer.qsize()
            target_buffer_size = self.config.get('target_buffer_size', 10)
            
            if self.frame_dropper.should_drop_frame(frame_metadata, buffer_size, target_buffer_size):
                self.dropped_frames += 1
                return None
        
        # Apply quality control
        if self.enable_adaptive_quality:
            quality_params = self.quality_controller.get_current_params()
            frame_data = self._apply_quality_settings(frame_data, quality_params)
        
        # Latency compensation
        if self.enable_latency_compensation:
            compensated_timestamp = self.latency_compensator.get_compensated_timestamp(
                frame_metadata.timestamp
            )
            frame_metadata.timestamp = compensated_timestamp
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self._update_performance_metrics(processing_time)
        
        self.processed_frames += 1
        return frame_data
    
    def _apply_quality_settings(self, frame_data: torch.Tensor, quality_params: Dict[str, Any]) -> torch.Tensor:
        """Apply quality settings"""
        scale_factor = quality_params.get('scale_factor', 1.0)
        
        # Only scale when change is significant to avoid minimal overhead
        if abs(scale_factor - 1.0) > 0.1:
            # Scale frame
            h, w = frame_data.shape[-2:]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Ensure the new size is reasonable
            if new_h > 0 and new_w > 0 and new_h < h * 2 and new_w < w * 2:
                try:
                    # Use faster nearest-neighbor interpolation rather than bilinear
                    frame_data = torch.nn.functional.interpolate(
                        frame_data.unsqueeze(0),
                        size=(new_h, new_w),
                        mode='nearest',
                        align_corners=None
                    ).squeeze(0)
                except Exception as e:
                    # If scaling fails, return the original frame
                    print(f"Frame scaling failed: {e}")
                    pass
        
        return frame_data
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        # Calculate FPS
        if hasattr(self, '_last_frame_time'):
            frame_interval = time.time() - self._last_frame_time
            current_fps = 1.0 / max(frame_interval, 0.001)
            
            # Update quality controller
            if self.enable_adaptive_quality:
                self.quality_controller.update_performance_metrics(current_fps, processing_time)
        
        self._last_frame_time = time.time()
        
        # Update latency compensator
        if self.enable_latency_compensation:
            self.latency_compensator.update_latency(processing_time)
    
    def _optimization_loop(self):
        """Optimization loop"""
        while self.running:
            try:
                # Periodically clean and optimize
                self._cleanup_resources()
                self._optimize_buffer_sizes()
                
                time.sleep(0.5)  # Run every 500ms
                
            except Exception as e:
                print(f"Optimization loop error: {e}")
                time.sleep(1.0)
    
    def _cleanup_resources(self):
        """Clean up resources"""
        # Clear GPU memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Clean expired frames in the buffer
        current_time = time.time()
        while not self.frame_buffer.empty():
            try:
                frame_item = self.frame_buffer.get_nowait()
                frame_timestamp = frame_item.get('timestamp', 0)
                
                # If a frame is too old (over 1 second), discard it
                if current_time - frame_timestamp > 1.0:
                    continue
                else:
                    # Put the frame back into the queue
                    self.frame_buffer.put_nowait(frame_item)
                    break
            except queue.Empty:
                break
    
    def _optimize_buffer_sizes(self):
        """Optimize buffer sizes"""
        # Adjust buffer sizes based on current performance
        current_fps = getattr(self, '_current_fps', 30.0)
        target_fps = self.quality_controller.target_fps
        
        if current_fps < target_fps * 0.8:
            # Performance insufficient — reduce buffer size
            new_max_size = max(5, self.frame_buffer.maxsize - 2)
        elif current_fps > target_fps * 0.95:
            # Good performance — increase buffer size
            new_max_size = min(50, self.frame_buffer.maxsize + 2)
        else:
            return
        
        # Recreate buffer (simplified implementation)
        if new_max_size != self.frame_buffer.maxsize:
            old_items = []
            while not self.frame_buffer.empty():
                try:
                    old_items.append(self.frame_buffer.get_nowait())
                except queue.Empty:
                    break
            
            self.frame_buffer = queue.Queue(maxsize=new_max_size)
            
            # Restore some buffered items
            for item in old_items[:new_max_size]:
                try:
                    self.frame_buffer.put_nowait(item)
                except queue.Full:
                    break
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report"""
        total_frames = self.processed_frames + self.dropped_frames
        drop_rate = self.dropped_frames / max(total_frames, 1)
        
        return {
            'performance_metrics': {
                'processed_frames': self.processed_frames,
                'dropped_frames': self.dropped_frames,
                'drop_rate_percent': drop_rate * 100,
                'current_fps': getattr(self, '_current_fps', 0.0),
                'target_fps': self.quality_controller.target_fps
            },
            'quality_settings': {
                'current_quality': self.quality_controller.current_quality.value,
                'quality_params': self.quality_controller.get_current_params()
            },
            'buffer_status': {
                'buffer_size': self.frame_buffer.qsize(),
                'max_buffer_size': self.frame_buffer.maxsize,
                'utilization_percent': (self.frame_buffer.qsize() / max(self.frame_buffer.maxsize, 1)) * 100
            },
            'optimization_status': {
                'adaptive_quality_enabled': self.enable_adaptive_quality,
                'smart_dropping_enabled': self.enable_smart_dropping,
                'latency_compensation_enabled': self.enable_latency_compensation
            },
            'latency_info': {
                'compensation_offset_ms': self.latency_compensator.compensation_offset * 1000,
                'target_latency_ms': self.latency_compensator.target_latency * 1000
            }
        }
    
    def set_target_fps(self, fps: float):
        """Set target FPS"""
        self.quality_controller.target_fps = fps
    
    def set_quality_level(self, quality: PlaybackQuality):
        """Manually set quality level"""
        self.quality_controller.current_quality = quality
    
    def shutdown(self):
        """Shut down optimizer"""
        self.running = False
        
        if self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5.0)


# Convenience function
def create_frame_metadata(
    frame_id: int,
    timestamp: float = None,
    processing_time: float = 0.0,
    quality_level: PlaybackQuality = PlaybackQuality.HIGH,
    is_keyframe: bool = False,
    size_bytes: int = 0,
    resolution: Tuple[int, int] = (1920, 1080)
) -> FrameMetadata:
    """Create frame metadata"""
    if timestamp is None:
        timestamp = time.time()
    
    return FrameMetadata(
        frame_id=frame_id,
        timestamp=timestamp,
        processing_time=processing_time,
        quality_level=quality_level,
        is_keyframe=is_keyframe,
        size_bytes=size_bytes,
        resolution=resolution
    )