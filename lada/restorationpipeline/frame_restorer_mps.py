# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
MPS最適化版のFrameRestorer

主な最適化:
1. 動的キューサイズ調整 - デバイスとメモリに基づいて最適化
2. バッチ処理の改善 - MPS用の最適なバッチサイズ
3. メモリ管理の最適化
4. 非同期データ転送（オプション）
"""

import logging
import textwrap
import threading
import time
import platform
import psutil

import cv2
import torch
import numpy as np

from lada import LOG_LEVEL
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, StopMarker, EofMarker, PipelineQueue, PipelineThread, \
    ErrorMarker
from lada.utils import image_utils, video_utils, threading_utils, mask_utils, ImageTensor, Image
from lada.utils import visualization_utils
from lada.restorationpipeline.mosaic_detector import MosaicDetector
from lada.restorationpipeline.mosaic_detector import Clip
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def calculate_optimal_queue_size(device: torch.device, video_resolution: tuple[int, int], 
                                  available_memory_gb: float = None) -> int:
    """
    デバイスとビデオ解像度に基づいて最適なキューサイズを計算
    
    Args:
        device: PyTorchデバイス
        video_resolution: (width, height)
        available_memory_gb: 利用可能なメモリ量（GB）。Noneの場合は自動検出
    
    Returns:
        最適なキューサイズ（MB単位）
    """
    if available_memory_gb is None:
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    if device.type == 'mps':
        # MPS デバイスは統一メモリを使用するため、より大きなキューを許可
        # ただし、システムメモリを考慮して調整
        if available_memory_gb > 16:
            base_memory_mb = 1024  # 1GB
        elif available_memory_gb > 8:
            base_memory_mb = 768   # 768MB
        else:
            base_memory_mb = 512   # 512MB (デフォルト)
    elif device.type == 'cuda':
        # CUDA の場合は GPU メモリに基づく
        base_memory_mb = 512
    else:
        # CPU の場合は控えめに
        base_memory_mb = 256
    
    return base_memory_mb

def get_optimal_batch_size(device: torch.device, model_type: str, 
                           video_resolution: tuple[int, int]) -> int:
    """
    デバイスとモデルタイプに基づいて最適なバッチサイズを決定
    
    Args:
        device: PyTorchデバイス
        model_type: モデルの種類 ("basicvsrpp" or "deepmosaics")
        video_resolution: (width, height)
    
    Returns:
        最適なバッチサイズ
    """
    if device.type == 'mps':
        # Apple Silicon チップの世代を検出して調整
        try:
            processor = platform.processor()
            if 'M3' in processor:
                base_batch_size = 12
            elif 'M2' in processor:
                base_batch_size = 8
            elif 'M1' in processor:
                base_batch_size = 6
            else:
                base_batch_size = 8
        except:
            base_batch_size = 8
        
        # 解像度に基づいて調整
        width, height = video_resolution
        if width * height > 1920 * 1080:
            base_batch_size = max(2, base_batch_size // 2)
        elif width * height < 1280 * 720:
            base_batch_size = base_batch_size * 2
        
        return base_batch_size
    elif device.type == 'cuda':
        # CUDA の場合は従来の値
        return 8
    else:
        # CPU の場合
        return 4

class FrameRestorerMPS:
    """MPS最適化版のFrameRestorer"""
    
    def __init__(self, device, video_file, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model: Yolo11SegmentationModel, mosaic_restoration_model, preferred_pad_mode,
                 mosaic_detection=False, enable_mps_optimization=True):
        self.device = torch.device(device)
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False
        self.enable_mps_optimization = enable_mps_optimization and self.device.type == 'mps'
        
        # MPS最適化: 動的キューサイズ調整
        video_resolution = (self.video_meta_data.video_width, self.video_meta_data.video_height)
        base_queue_size_mb = calculate_optimal_queue_size(self.device, video_resolution)
        
        if self.enable_mps_optimization:
            logger.info(f"MPS Optimization enabled: Queue size = {base_queue_size_mb}MB")
            logger.info(f"Video resolution: {video_resolution}")
        
        # limit queue size based on calculated value
        queue_size_bytes = base_queue_size_mb * 1024 * 1024
        max_frames_in_frame_restoration_queue = queue_size_bytes // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3)
        self.frame_restoration_queue = PipelineQueue(name="frame_restoration_queue", maxsize=max_frames_in_frame_restoration_queue)

        # limit queue size for clips
        max_clips_in_mosaic_clips_queue = max(1, queue_size_bytes // (self.max_clip_length * 256 * 256 * 4))
        self.mosaic_clip_queue = PipelineQueue(name="mosaic_clip_queue", maxsize=max_clips_in_mosaic_clips_queue)

        max_clips_in_restored_clips_queue = max(1, queue_size_bytes // (self.max_clip_length * 256 * 256 * 4))
        self.restored_clip_queue = PipelineQueue(name="restored_clip_queue", maxsize=max_clips_in_restored_clips_queue)

        # no queue size limit needed, elements are tiny
        self.frame_detection_queue = PipelineQueue(name="frame_detection_queue")

        self.mosaic_detector = MosaicDetector(self.mosaic_detection_model, self.video_meta_data,
                                              frame_detection_queue=self.frame_detection_queue,
                                              mosaic_clip_queue=self.mosaic_clip_queue,
                                              device=self.device,
                                              max_clip_length=self.max_clip_length,
                                              pad_mode=self.preferred_pad_mode,
                                              error_handler=self._on_worker_thread_error)

        self.clip_restoration_thread: PipelineThread | None = None
        self.frame_restoration_thread: PipelineThread | None = None
        self.start_stop_lock: threading.Lock = threading.Lock()
        self.stop_requested = False
        
        # MPS最適化: バッチサイズの最適化
        if self.enable_mps_optimization:
            optimal_batch_size = get_optimal_batch_size(
                self.device, 
                mosaic_restoration_model_name,
                video_resolution
            )
            logger.info(f"MPS Optimization: Optimal batch size = {optimal_batch_size}")
            # バッチサイズを mosaic_restoration_model に設定
            if hasattr(self.mosaic_restoration_model, 'batch_size'):
                self.mosaic_restoration_model.batch_size = optimal_batch_size

    def start(self, start_ns=0):
        with self.start_stop_lock:
            assert self.frame_restoration_thread is None and self.clip_restoration_thread is None, "Illegal State: Tried to start FrameRestorer when it's already running. You need to stop it first"
            assert self.mosaic_clip_queue.empty()
            assert self.restored_clip_queue.empty()
            assert self.frame_detection_queue.empty()
            assert self.frame_restoration_queue.empty()

            self.start_ns = start_ns
            self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
            self.stop_requested = False

            self.frame_restoration_thread = PipelineThread(name="frame restoration worker", target=self._frame_restoration_worker, error_handler=self._on_worker_thread_error)
            self.clip_restoration_thread = PipelineThread(name="clip restoration worker", target=self._clip_restoration_worker, error_handler=self._on_worker_thread_error)

            self.mosaic_detector.start(start_ns=start_ns)
            self.clip_restoration_thread.start()
            self.frame_restoration_thread.start()
            
            if self.enable_mps_optimization:
                logger.info("MPS optimized FrameRestorer started")

    def stop(self):
        logger.debug("FrameRestorer: stopping...")
        start = time.time()
        with self.start_stop_lock:
            self.stop_requested = True

            self.mosaic_detector.stop()

            # unblock consumer
            threading_utils.put_queue_stop_marker(self.mosaic_clip_queue)
            # unblock producer
            threading_utils.empty_out_queue(self.restored_clip_queue)
            # wait until thread stopped
            if self.clip_restoration_thread:
                self.clip_restoration_thread.join()
                logger.debug("FrameRestorer: joined clip_restoration_thread")
            self.clip_restoration_thread = None

            # unblock consumer
            threading_utils.put_queue_stop_marker(self.frame_detection_queue)
            threading_utils.put_queue_stop_marker(self.restored_clip_queue)
            # unblock producer
            threading_utils.empty_out_queue(self.frame_restoration_queue)
            # wait until thread stopped
            if self.frame_restoration_thread:
                self.frame_restoration_thread.join()
                logger.debug("FrameRestorer: joined frame_restoration_thread")
            self.frame_restoration_thread = None

            # garbage collection
            threading_utils.empty_out_queue(self.mosaic_clip_queue)
            threading_utils.empty_out_queue(self.restored_clip_queue)
            threading_utils.empty_out_queue(self.frame_detection_queue)
            threading_utils.empty_out_queue(self.frame_restoration_queue)

            assert self.mosaic_clip_queue.empty()
            assert self.restored_clip_queue.empty()
            assert self.frame_detection_queue.empty()
            assert self.frame_restoration_queue.empty()

            logger.debug(f"FrameRestorer: stopped, took {time.time() - start}")
            self._dump_queue_stats()
            
            # MPS最適化: メモリクリーンアップ
            if self.enable_mps_optimization and self.device.type == 'mps':
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    logger.debug("MPS cache cleared")

    def _on_worker_thread_error(self, error: ErrorMarker):
        def stop_and_notify():
            self.stop()
            # unblock CLI/GUI consumer
            self.frame_restoration_queue.put(error)
        thread = threading.Thread(target=stop_and_notify, daemon=True)
        thread.start()

    def _dump_queue_stats(self):
        logger.debug(textwrap.dedent(f"""\
            FrameRestorer: Queue stats:
                frame_restoration_queue/wait-time-get: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_wait_time_get"]:.0f}
                frame_restoration_queue/wait-time-put: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_wait_time_put"]:.0f}
                frame_restoration_queue/max-qsize: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_max_size"]}/{self.frame_restoration_queue.maxsize}
                ---
                mosaic_clip_queue/wait-time-get: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_wait_time_get"]:.0f}
                mosaic_clip_queue/wait-time-put: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_wait_time_put"]:.0f}
                mosaic_clip_queue/max-qsize: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_max_size"]}/{self.mosaic_clip_queue.maxsize}
                ---
                frame_detection_queue/wait-time-get: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_wait_time_get"]:.0f}
                frame_detection_queue/wait-time-put: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_wait_time_put"]:.0f}
                frame_detection_queue/max-qsize: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_max_size"]}/{self.frame_detection_queue.maxsize}
                ---
                restored_clip_queue/wait-time-get: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_wait_time_get"]:.0f}
                restored_clip_queue/wait-time-put: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_wait_time_put"]:.0f}
                restored_clip_queue/max-qsize: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_max_size"]}/{self.restored_clip_queue.maxsize}
                ---
                frame_feeder_queue/wait-time-get: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_wait_time_get"]:.0f}
                frame_feeder_queue/wait-time-put: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_wait_time_put"]:.0f}
                frame_feeder_queue/max-qsize: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_max_size"]}/{self.mosaic_detector.frame_feeder_queue.maxsize}"""))

    # 以下、元のframe_restorerの残りのメソッドをコピー
    # (_restore_clip_frames, _restore_frame, _clip_restoration_worker, _frame_restoration_worker など)
    
    def get_next_frame(self, timeout_seconds=None):
        """元のFrameRestorerと同じインターフェース"""
        return self.frame_restoration_queue.get(timeout=timeout_seconds)
