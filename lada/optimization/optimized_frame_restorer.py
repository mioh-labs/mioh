"""
Optimized Frame Restorer
Integrates smart optimization controller; enables optimization only under heavy load
"""

import logging
import torch
import numpy as np
from typing import Optional, Dict, Any, List
import time
import threading

# Import original LADA components
from lada.lib.frame_restorer import FrameRestorer
from lada.lib.mps_optimizer import create_mps_optimizer

# Import optimization components
from .smart_optimization_controller import (
    SmartOptimizationController, 
    LoadMetrics, 
    LoadLevel,
    OptimizationThresholds,
    create_load_metrics_from_frame_data
)
from .integrated_optimization_manager import IntegratedOptimizationManager

logger = logging.getLogger(__name__)


class OptimizedFrameRestorer(FrameRestorer):
    """
    Frame restorer with integrated smart optimization
    Extends FrameRestorer with intelligent optimization features
    """
    
    def __init__(self, device, video_file, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode,
                 mosaic_detection=False, batch_size=4, enable_load_balancing=True, 
                 high_cpu_utilization=True, enable_smart_optimization=True,
                 optimization_thresholds: Optional[OptimizationThresholds] = None):
        
        # Initialize base class
        super().__init__(
            device, video_file, max_clip_length, mosaic_restoration_model_name,
            mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode,
            mosaic_detection, batch_size, enable_load_balancing, high_cpu_utilization
        )
        
        # Smart optimization components
        self.enable_smart_optimization = enable_smart_optimization
        self.optimization_controller = None
        self.optimization_manager = None
        self.optimization_stats = {
            'total_frames_processed': 0,
            'optimization_enabled_frames': 0,
            'optimization_disabled_frames': 0,
            'performance_gains': [],
            'load_level_distribution': {}
        }
        self._stats_lock = threading.Lock()
        
        if self.enable_smart_optimization:
            # Initialize smart optimization controller
            self.optimization_controller = SmartOptimizationController(optimization_thresholds)
            
            # Initialize integrated optimization manager (on demand)
            self.optimization_manager = None
            
            logger.info("Smart optimization enabled for FrameRestorer")
        else:
            logger.info("Smart optimization disabled for FrameRestorer")
    
    def _should_enable_optimization_for_processing(self, images: List[np.ndarray], 
                                                 processing_mode: str = "mosaic_restoration") -> bool:
        """Determine whether to enable optimization for current processing"""
        if not self.enable_smart_optimization or not self.optimization_controller:
            return False
        
        # Create load metrics
        if images and len(images) > 0:
            # Convert to torch tensor for analysis
            sample_image = images[0]
            if isinstance(sample_image, np.ndarray):
                # Convert to torch tensor format [C, H, W]
                if len(sample_image.shape) == 3:  # [H, W, C]
                    tensor_data = torch.from_numpy(sample_image).permute(2, 0, 1).float()
                else:  # [H, W]
                    tensor_data = torch.from_numpy(sample_image).unsqueeze(0).float()
            else:
                tensor_data = sample_image
            
            # If batching, create batched tensor
            if len(images) > 1:
                tensor_data = tensor_data.unsqueeze(0).repeat(len(images), 1, 1, 1)
            
            # Create load metrics
            load_metrics = create_load_metrics_from_frame_data(
                frame_data=tensor_data,
                batch_size=len(images),
                processing_mode=processing_mode,
                concurrent_tasks=1
            )
            
            # Decide whether to enable optimization
            should_optimize = self.optimization_controller.should_enable_optimization(load_metrics)
            
            # Update statistics
            with self._stats_lock:
                self.optimization_stats['total_frames_processed'] += len(images)
                if should_optimize:
                    self.optimization_stats['optimization_enabled_frames'] += len(images)
                else:
                    self.optimization_stats['optimization_disabled_frames'] += len(images)
                
                # Record load level distribution
                load_level = self.optimization_controller.current_load_level.value
                if load_level not in self.optimization_stats['load_level_distribution']:
                    self.optimization_stats['load_level_distribution'][load_level] = 0
                self.optimization_stats['load_level_distribution'][load_level] += len(images)
            
            return should_optimize
        
        return False
    
    def _get_optimization_manager(self):
        """Lazily initialize optimization manager"""
        if self.optimization_manager is None:
            self.optimization_manager = IntegratedOptimizationManager()
            self.optimization_manager.initialize_optimization()
        return self.optimization_manager
    
    def _restore_clip_frames(self, images):
        """Override clip restoration with smart optimization"""
        start_time = time.time()
        
        # Decide whether to enable optimization
        should_optimize = self._should_enable_optimization_for_processing(images, "mosaic_restoration")
        
        if should_optimize:
            logger.debug(f"Enabling optimization for {len(images)} frames (load level: {self.optimization_controller.current_load_level.value})")
            
            # Enable optimized processing
            try:
                optimization_manager = self._get_optimization_manager()
                
                # Preprocess images using optimized parallel handling
                if self.high_cpu_utilization and len(images) > 1:
                    # Use optimization manager for frame processing
                    optimized_images = []
                    for img in images:
                        if isinstance(img, np.ndarray):
                            # Convert to tensor
                            if len(img.shape) == 3:  # [H, W, C]
                                tensor_img = torch.from_numpy(img).permute(2, 0, 1).float()
                            else:  # [H, W]
                                tensor_img = torch.from_numpy(img).unsqueeze(0).float()
                        else:
                            tensor_img = img
                        
                        # Process via optimization manager
                        from .realtime_gui_optimizer import FrameMetadata
                        metadata = FrameMetadata(
                            frame_number=0,
                            timestamp=time.time(),
                            frame_size=tensor_img.shape[-2:],
                            processing_complexity=0.8  # Mosaic restoration is high complexity
                        )
                        
                        optimized_tensor = optimization_manager.process_frame_optimized(tensor_img, metadata)
                        if optimized_tensor is not None:
                            # Convert back to numpy
                            if optimized_tensor.dim() == 3:  # [C, H, W]
                                optimized_img = optimized_tensor.permute(1, 2, 0).cpu().numpy()
                            else:  # [H, W]
                                optimized_img = optimized_tensor.cpu().numpy()
                            optimized_images.append(optimized_img)
                        else:
                            optimized_images.append(img)  # Use original image
                    
                    images = optimized_images
                else:
                    # Single-frame optimized processing
                    if len(images) == 1:
                        img = images[0]
                        if isinstance(img, np.ndarray):
                            if len(img.shape) == 3:  # [H, W, C]
                                tensor_img = torch.from_numpy(img).permute(2, 0, 1).float()
                            else:  # [H, W]
                                tensor_img = torch.from_numpy(img).unsqueeze(0).float()
                        else:
                            tensor_img = img
                        
                        from .realtime_gui_optimizer import FrameMetadata
                        metadata = FrameMetadata(
                            frame_number=0,
                            timestamp=time.time(),
                            frame_size=tensor_img.shape[-2:],
                            processing_complexity=0.8
                        )
                        
                        optimized_tensor = optimization_manager.process_frame_optimized(tensor_img, metadata)
                        if optimized_tensor is not None:
                            if optimized_tensor.dim() == 3:  # [C, H, W]
                                optimized_img = optimized_tensor.permute(1, 2, 0).cpu().numpy()
                            else:  # [H, W]
                                optimized_img = optimized_tensor.cpu().numpy()
                            images = [optimized_img]
                
            except Exception as e:
                logger.warning(f"Optimization failed, falling back to standard processing: {e}")
                # Continue with standard processing
        else:
            logger.debug(f"Optimization disabled for {len(images)} frames (load level: {self.optimization_controller.current_load_level.value if self.optimization_controller else 'unknown'})")
        
        # Call base class standard processing method
        try:
            if should_optimize:
                # For optimized cases, some parameters may need tuning
                restored_images = super()._restore_clip_frames(images)
            else:
                # Standard processing
                restored_images = super()._restore_clip_frames(images)
            
            # Record performance
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.optimization_stats['performance_gains'].append({
                    'processing_time': processing_time,
                    'frame_count': len(images),
                    'optimization_enabled': should_optimize,
                    'load_level': self.optimization_controller.current_load_level.value if self.optimization_controller else 'unknown'
                })
            
            return restored_images
            
        except Exception as e:
            logger.error(f"Frame restoration failed: {e}")
            raise
    
    def _restore_frame(self, frame, frame_num, restored_clips):
        """Override single-frame restoration with smart optimization"""
        # Single-frame processing is usually light load; do not enable optimization
        # Still record statistics
        if self.optimization_controller:
            # Create simple load metrics
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() if isinstance(frame, np.ndarray) else frame
            load_metrics = create_load_metrics_from_frame_data(
                frame_data=frame_tensor,
                batch_size=1,
                processing_mode="frame_blending",
                concurrent_tasks=len(restored_clips) if restored_clips else 1
            )
            
            should_optimize = self.optimization_controller.should_enable_optimization(load_metrics)
            
            with self._stats_lock:
                self.optimization_stats['total_frames_processed'] += 1
                if should_optimize:
                    self.optimization_stats['optimization_enabled_frames'] += 1
                else:
                    self.optimization_stats['optimization_disabled_frames'] += 1
        
        # Call base class method
        return super()._restore_frame(frame, frame_num, restored_clips)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        with self._stats_lock:
            stats = self.optimization_stats.copy()
            
            # Compute optimization rate
            total_frames = stats['total_frames_processed']
            if total_frames > 0:
                stats['optimization_rate'] = stats['optimization_enabled_frames'] / total_frames
            else:
                stats['optimization_rate'] = 0.0
            
            # Compute average performance
            if stats['performance_gains']:
                optimized_times = [g['processing_time'] for g in stats['performance_gains'] if g['optimization_enabled']]
                standard_times = [g['processing_time'] for g in stats['performance_gains'] if not g['optimization_enabled']]
                
                if optimized_times and standard_times:
                    avg_optimized = sum(optimized_times) / len(optimized_times)
                    avg_standard = sum(standard_times) / len(standard_times)
                    stats['average_performance_gain'] = (avg_standard - avg_optimized) / avg_standard * 100
                else:
                    stats['average_performance_gain'] = 0.0
            else:
                stats['average_performance_gain'] = 0.0
            
            # Append controller statistics
            if self.optimization_controller:
                controller_stats = self.optimization_controller.get_performance_stats()
                stats['controller_stats'] = controller_stats
            
            return stats
    
    def get_optimization_recommendation(self, images: List[np.ndarray], 
                                      processing_mode: str = "mosaic_restoration") -> Dict[str, Any]:
        """Get optimization recommendation for current processing"""
        if not self.optimization_controller or not images:
            return {'recommendation': 'optimization_disabled', 'reason': 'Controller not available or no images'}
        
        # Create load metrics
        sample_image = images[0]
        if isinstance(sample_image, np.ndarray):
            if len(sample_image.shape) == 3:  # [H, W, C]
                tensor_data = torch.from_numpy(sample_image).permute(2, 0, 1).float()
            else:  # [H, W]
                tensor_data = torch.from_numpy(sample_image).unsqueeze(0).float()
        else:
            tensor_data = sample_image
        
        if len(images) > 1:
            tensor_data = tensor_data.unsqueeze(0).repeat(len(images), 1, 1, 1)
        
        load_metrics = create_load_metrics_from_frame_data(
            frame_data=tensor_data,
            batch_size=len(images),
            processing_mode=processing_mode,
            concurrent_tasks=1
        )
        
        return self.optimization_controller.get_optimization_recommendation(load_metrics)
    
    def reset_optimization_stats(self):
        """Reset optimization statistics"""
        with self._stats_lock:
            self.optimization_stats = {
                'total_frames_processed': 0,
                'optimization_enabled_frames': 0,
                'optimization_disabled_frames': 0,
                'performance_gains': [],
                'load_level_distribution': {}
            }
        
        if self.optimization_controller:
            self.optimization_controller.reset_stats()
    
    def shutdown_optimization(self):
        """Shut down optimization system"""
        if self.optimization_manager:
            self.optimization_manager.shutdown()
            self.optimization_manager = None
        
        logger.info("Optimization system shutdown completed")


def create_optimized_frame_restorer(device, video_file, max_clip_length, 
                                  mosaic_restoration_model_name,
                                  mosaic_detection_model, mosaic_restoration_model, 
                                  preferred_pad_mode, mosaic_detection=False, 
                                  batch_size=4, enable_load_balancing=True, 
                                  high_cpu_utilization=True, 
                                  enable_smart_optimization=True,
                                  optimization_thresholds: Optional[OptimizationThresholds] = None) -> OptimizedFrameRestorer:
    """Create an optimized frame restorer instance"""
    
    return OptimizedFrameRestorer(
        device=device,
        video_file=video_file,
        max_clip_length=max_clip_length,
        mosaic_restoration_model_name=mosaic_restoration_model_name,
        mosaic_detection_model=mosaic_detection_model,
        mosaic_restoration_model=mosaic_restoration_model,
        preferred_pad_mode=preferred_pad_mode,
        mosaic_detection=mosaic_detection,
        batch_size=batch_size,
        enable_load_balancing=enable_load_balancing,
        high_cpu_utilization=high_cpu_utilization,
        enable_smart_optimization=enable_smart_optimization,
        optimization_thresholds=optimization_thresholds
    )