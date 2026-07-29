# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import ctypes
import gc
import sys
import textwrap
import threading
import time
import platform
import psutil
import types

import cv2
import torch
import numpy as np

from lada import LOG_LEVEL
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, StopMarker, EofMarker, PipelineQueue, PipelineThread, \
    ErrorMarker
from lada.utils import image_utils, video_utils, threading_utils, mask_utils, ImageTensor, Image
from lada.utils import visualization_utils
from lada.utils.mps_utils import (
    get_mps_available_memory_gb,
    get_mps_memory_stats,
    release_mps_memory_if_needed,
    serialized_mps_execution,
)
from lada.restorationpipeline.mosaic_detector import MosaicDetector
from lada.restorationpipeline.mosaic_detector import Clip
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


def _release_darwin_malloc_cache() -> int:
    """Ask macOS malloc zones to return unused pages without touching live data."""
    if platform.system() != "Darwin":
        return 0
    try:
        libc = ctypes.CDLL(None)
        pressure_relief = libc.malloc_zone_pressure_relief
        pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        pressure_relief.restype = ctypes.c_size_t
        return int(pressure_relief(None, 0))
    except (AttributeError, OSError):
        return 0


def apply_restore_sharpening(image: np.ndarray, strength: float, sigma: float = 1.0) -> np.ndarray:
    if strength <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def apply_restore_detail_boost(image: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0 + strength * 2.0, tileGridSize=(8, 8))
    boosted_l = clahe.apply(l_channel)
    mixed_l = cv2.addWeighted(l_channel, 1.0 - strength, boosted_l, strength, 0)
    boosted_lab = cv2.merge((mixed_l, a_channel, b_channel))
    return cv2.cvtColor(boosted_lab, cv2.COLOR_LAB2RGB)


def apply_restore_smoothing(image: np.ndarray, strength: float, sigma: float = 1.0) -> np.ndarray:
    if strength <= 0:
        return image
    strength = min(strength, 1.0)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 - strength, blurred, strength, 0)


def _normalize_roi_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 2:
        mask = mask[:, :, None]
    mask_f = (mask.astype(np.float32) > 0).astype(np.float32)
    if mask_f.shape[:2] != shape:
        mask_f = cv2.resize(mask_f, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask_f.ndim == 2:
            mask_f = mask_f[:, :, None]
    return mask_f


def _masked_gaussian_blur(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    weighted = image * mask
    blurred_weighted = cv2.GaussianBlur(weighted, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if blurred_mask.ndim == 2:
        blurred_mask = blurred_mask[:, :, None]
    return blurred_weighted / np.maximum(blurred_mask, 1e-6)


def _apply_mask_to_processed(base: np.ndarray, processed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_f = _normalize_roi_mask(mask, base.shape[:2])
    mixed = base.astype(np.float32) * (1.0 - mask_f) + processed.astype(np.float32) * mask_f
    return np.clip(mixed, 0, 255).astype(np.uint8)


def apply_restore_texture_mix(restored: np.ndarray, original: np.ndarray, strength: float, mask: np.ndarray | None = None) -> np.ndarray:
    if strength <= 0:
        return restored
    original_f = original.astype(np.float32)
    restored_f = restored.astype(np.float32)
    mask_f = _normalize_roi_mask(mask, restored.shape[:2]) if mask is not None else None
    if mask_f is not None and np.count_nonzero(mask_f) == 0:
        return restored
    if mask_f is None:
        blur_small = cv2.GaussianBlur(original_f, (0, 0), sigmaX=0.7, sigmaY=0.7)
        blur_large = cv2.GaussianBlur(original_f, (0, 0), sigmaX=2.0, sigmaY=2.0)
    else:
        blur_small = _masked_gaussian_blur(original_f, mask_f, sigma=0.7)
        blur_large = _masked_gaussian_blur(original_f, mask_f, sigma=2.0)
    mid_frequency = blur_small - blur_large
    if mask_f is not None:
        mid_frequency *= mask_f
    mixed = restored_f + mid_frequency * strength
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    return _apply_mask_to_processed(restored, mixed, mask_f) if mask_f is not None else mixed


def _install_torchvision_functional_tensor_compat():
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional as functional
    except ImportError:
        return
    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = module


def create_realesrgan_enhancer(model_path: str, scale: int = 2, tile: int = 0, fp16: bool = False, device=None):
    if not model_path:
        raise ValueError("--restore-roi-enhancer-model-path is required when --restore-roi-enhancer realesrgan is used")
    if str(model_path).endswith((".aimodel", ".aimodelc")):
        from lada.restorationpipeline.coreai_roi_enhancer import CoreAIROIEnhancer
        return CoreAIROIEnhancer(model_path, scale=scale)
    if str(model_path).endswith((".mlpackage", ".mlmodelc")):
        from lada.restorationpipeline.coreml_roi_enhancer import CoreMLROIEnhancer
        return CoreMLROIEnhancer(model_path)
    _install_torchvision_functional_tensor_compat()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise RuntimeError("Real-ESRGAN support requires installing realesrgan and basicsr") from exc

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    gpu_id = 0 if torch.cuda.is_available() else None
    device_type = torch.device(device).type if device is not None else None
    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=fp16 and (gpu_id is not None or device_type == 'mps'),
        device=device,
        gpu_id=gpu_id,
    )


def create_spandrel_enhancer(model_path: str, scale: int = 4, tile: int = 0, fp16: bool = False, device=None):
    if not model_path:
        raise ValueError("--restore-roi-enhancer-model-path is required when --restore-roi-enhancer spandrel is used")
    if str(model_path).endswith((".aimodel", ".aimodelc")):
        from lada.restorationpipeline.coreai_roi_enhancer import CoreAIROIEnhancer
        return CoreAIROIEnhancer(model_path, scale=scale)
    if str(model_path).endswith((".mlpackage", ".mlmodelc")):
        from lada.restorationpipeline.coreml_roi_enhancer import CoreMLROIEnhancer
        return CoreMLROIEnhancer(model_path)
    from lada.restorationpipeline.spandrel_roi_enhancer import SpandrelROIEnhancer
    return SpandrelROIEnhancer(model_path, device=device, fp16=fp16, tile=tile)


def apply_restore_roi_enhancer(
    restored: np.ndarray,
    enhancer,
    strength: float,
    scale: int = 2,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if strength <= 0 or enhancer is None:
        return restored
    enhanced_bgr, _ = enhancer.enhance(cv2.cvtColor(restored, cv2.COLOR_RGB2BGR), outscale=scale)
    enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    if enhanced.shape[:2] != restored.shape[:2]:
        enhanced = cv2.resize(enhanced, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
    blended = cv2.addWeighted(restored, 1.0 - strength, enhanced, strength, 0)
    return _apply_mask_to_processed(restored, blended, mask) if mask is not None else blended


def apply_restore_effect_upscale(
    restored: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    scale: int = 1,
    texture_mix: float = 0.0,
    detail_boost: float = 0.0,
    sharpen_strength: float = 0.0,
    smooth_strength: float = 0.0,
) -> np.ndarray:
    if texture_mix <= 0 and detail_boost <= 0 and sharpen_strength <= 0 and smooth_strength <= 0:
        return restored
    mask_f = _normalize_roi_mask(mask, restored.shape[:2])
    if np.count_nonzero(mask_f) == 0:
        return restored
    if scale <= 1:
        processed = apply_restore_texture_mix(restored, original, texture_mix, mask=mask_f)
        processed = apply_restore_detail_boost(processed, detail_boost)
        processed = _apply_mask_to_processed(restored, processed, mask_f)
        processed = apply_restore_sharpening(processed, sharpen_strength)
        processed = _apply_mask_to_processed(restored, processed, mask_f)
        processed = apply_restore_smoothing(processed, smooth_strength)
        return _apply_mask_to_processed(restored, processed, mask_f)

    target_size = (restored.shape[1] * scale, restored.shape[0] * scale)
    restored_hr = cv2.resize(restored, target_size, interpolation=cv2.INTER_CUBIC)
    original_hr = cv2.resize(original, target_size, interpolation=cv2.INTER_CUBIC)
    mask_hr = cv2.resize(mask_f, target_size, interpolation=cv2.INTER_NEAREST)
    if mask_hr.ndim == 2:
        mask_hr = mask_hr[:, :, None]

    processed_hr = apply_restore_texture_mix(restored_hr, original_hr, texture_mix, mask=mask_hr)
    processed_hr = apply_restore_detail_boost(processed_hr, detail_boost)
    processed_hr = _apply_mask_to_processed(restored_hr, processed_hr, mask_hr)
    processed_hr = apply_restore_sharpening(processed_hr, sharpen_strength)
    processed_hr = _apply_mask_to_processed(restored_hr, processed_hr, mask_hr)
    processed_hr = apply_restore_smoothing(processed_hr, smooth_strength)
    processed_hr = _apply_mask_to_processed(restored_hr, processed_hr, mask_hr)

    processed = cv2.resize(processed_hr, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
    return _apply_mask_to_processed(restored, processed, mask_f)


def _get_mps_adaptive_profile(max_clip_length: int) -> dict:
    """
    Adaptive restore profile tuned for Apple Silicon unified memory.
    Defaults are tuned for M1 16GB and remain safe for other Apple Silicon variants.
    """
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    profile = {
        "init_chunk": 16,
        "min_chunk": 6,
        "detect_batch_init": 4,
        "low_mem_gb": 3.5,
        "mid_mem_gb": 5.0,
        "high_mem_gb": 8.0,
    }

    if is_apple_silicon:
        if total_memory_gb >= 30:
            profile.update({
                "init_chunk": 24,
                "min_chunk": 8,
                "detect_batch_init": 6,
                "low_mem_gb": 5.0,
                "mid_mem_gb": 7.0,
                "high_mem_gb": 12.0,
            })
        elif total_memory_gb >= 22:
            profile.update({
                "init_chunk": 20,
                "min_chunk": 7,
                "detect_batch_init": 5,
                "low_mem_gb": 4.0,
                "mid_mem_gb": 6.0,
                "high_mem_gb": 10.0,
            })

    profile["init_chunk"] = max(1, min(profile["init_chunk"], max_clip_length))
    profile["min_chunk"] = max(1, min(profile["min_chunk"], profile["init_chunk"]))
    profile["detect_batch_init"] = max(2, min(profile["detect_batch_init"], 8))
    return profile


def calculate_optimal_queue_size_mb(device: torch.device, video_resolution: tuple[int, int], 
                                     available_memory_gb: float = None) -> int:
    """
    デバイスとビデオ解像度に基づいて最適なキューサイズを計算（MPS最適化）
    
    Args:
        device: PyTorchデバイス
        video_resolution: (width, height)
        available_memory_gb: 利用可能なメモリ量（GB）。Noneの場合は自動検出
    
    Returns:
        最適なキューサイズ（MB単位）
    """
    if available_memory_gb is None:
        if device.type == 'mps':
            available_memory_gb = get_mps_available_memory_gb()
        else:
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    if device.type == 'mps':
        # MPS デバイスは統一メモリを使用するため、より大きなキューを許可
        if available_memory_gb > 16:
            base_memory_mb = 1024  # 1GB
        elif available_memory_gb > 8:
            base_memory_mb = 768   # 768MB
        else:
            base_memory_mb = 512   # 512MB (デフォルト)
        logger.info(f"MPS Optimization: Queue size = {base_memory_mb}MB (Available memory: {available_memory_gb:.1f}GB)")
    elif device.type == 'cuda':
        base_memory_mb = 512
    else:
        # CPU の場合は控えめに
        base_memory_mb = 256

    return base_memory_mb


def _restoration_clip_size(model_name: str, restoration_model) -> int:
    """Return the square ROI size required by the selected restorer."""
    if model_name.startswith("mioh-restorer"):
        runtime = getattr(restoration_model, "runtime", None)
        image_size = getattr(runtime, "image_size", None)
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("MiohRestorer runtime must declare a positive image_size")
        return image_size
    return 256

class FrameRestorer:
    def __init__(self, device, video_file, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model: Yolo11SegmentationModel, mosaic_restoration_model, preferred_pad_mode,
                 mosaic_detection=False, restore_sharpen_strength: float = 0.0,
                 restore_detail_boost: float = 0.0, restore_blend_feather: float = 1.0,
                 restore_texture_mix: float = 0.0, restore_smooth_strength: float = 0.0,
                 restore_roi_enhancer: str = "none",
                 restore_roi_enhancer_model_path: str | None = None,
                 restore_roi_enhancer_scale: int = 2,
                 restore_roi_enhancer_strength: float = 0.0,
                 restore_roi_enhancer_tile: int = 0,
                 restore_effect_upscale: int = 1,
                 fp16_enabled: bool = False,
                 mosaic_detection_empty_lookahead: int = 0,
                 restore_max_frames: int | None = None,
                 restore_temporal_overlap: int = 8,
                 restore_crossfade: bool = True):
        self.device = torch.device(device)
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.restoration_clip_size = _restoration_clip_size(
            mosaic_restoration_model_name,
            mosaic_restoration_model,
        )
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.restore_sharpen_strength = restore_sharpen_strength
        self.restore_detail_boost = restore_detail_boost
        self.restore_blend_feather = restore_blend_feather
        self.restore_texture_mix = restore_texture_mix
        self.restore_smooth_strength = restore_smooth_strength
        self.restore_roi_enhancer_name = restore_roi_enhancer
        self.restore_roi_enhancer_scale = restore_roi_enhancer_scale
        self.restore_roi_enhancer_strength = restore_roi_enhancer_strength
        self.restore_effect_upscale = restore_effect_upscale
        self.restore_max_frames = restore_max_frames
        self.restore_temporal_overlap = restore_temporal_overlap
        self.restore_crossfade = restore_crossfade
        self.restore_roi_enhancer = None
        supported_roi_enhancers = ("realesrgan", "mewzoom", "swinir", "spandrel")
        if restore_roi_enhancer in supported_roi_enhancers:
            # A selected but zero-strength enhancer has no effect on pixels.
            # Do not load its Torch/Core ML/Core AI weights until it can
            # actually be used; some enhancer assets exceed 200 MiB.
            if restore_roi_enhancer_strength > 0:
                if restore_roi_enhancer in ("mewzoom", "swinir") and not str(restore_roi_enhancer_model_path).endswith(".mlpackage"):
                    raise ValueError(f"{restore_roi_enhancer} enhancer requires a Core ML .mlpackage model")
                if restore_roi_enhancer == "spandrel":
                    self.restore_roi_enhancer = create_spandrel_enhancer(
                        restore_roi_enhancer_model_path,
                        scale=restore_roi_enhancer_scale,
                        tile=restore_roi_enhancer_tile,
                        fp16=fp16_enabled,
                        device=self.device,
                    )
                else:
                    self.restore_roi_enhancer = create_realesrgan_enhancer(
                        restore_roi_enhancer_model_path,
                        scale=restore_roi_enhancer_scale,
                        tile=restore_roi_enhancer_tile,
                        fp16=fp16_enabled,
                        device=self.device,
                    )
        elif restore_roi_enhancer != "none":
            raise ValueError(f"Unsupported restore ROI enhancer: {restore_roi_enhancer}")
        self.mosaic_detection_empty_lookahead = mosaic_detection_empty_lookahead
        self.eof = False
        self.stop_requested = False
        self._mps_adaptive_profile = _get_mps_adaptive_profile(self.max_clip_length)
        self._adaptive_restore_chunk_frames = self._mps_adaptive_profile["init_chunk"]

        # MPS最適化: 動的キューサイズ調整
        video_resolution = (self.video_meta_data.video_width, self.video_meta_data.video_height)
        base_queue_size_mb = calculate_optimal_queue_size_mb(self.device, video_resolution)
        queue_size_bytes = base_queue_size_mb * 1024 * 1024

        # limit queue size based on calculated value
        max_frames_in_frame_restoration_queue = queue_size_bytes // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3)
        self.frame_restoration_queue = PipelineQueue(name="frame_restoration_queue", maxsize=max_frames_in_frame_restoration_queue)

        # limit queue size for clips
        clip_bytes = (
            self.max_clip_length
            * self.restoration_clip_size
            * self.restoration_clip_size
            * 4
        )  # 4 = 3 color channels + mask
        max_clips_in_mosaic_clips_queue = max(1, queue_size_bytes // clip_bytes)
        self.mosaic_clip_queue = PipelineQueue(name="mosaic_clip_queue", maxsize=max_clips_in_mosaic_clips_queue)

        # limit queue size for restored clips
        max_clips_in_restored_clips_queue = max(1, queue_size_bytes // clip_bytes)
        self.restored_clip_queue = PipelineQueue(name="restored_clip_queue", maxsize=max_clips_in_restored_clips_queue)

        # no queue size limit needed, elements are tiny
        self.frame_detection_queue = PipelineQueue(name="frame_detection_queue")

        self.mosaic_detector = MosaicDetector(self.mosaic_detection_model, self.video_meta_data,
                                              frame_detection_queue=self.frame_detection_queue,
                                              mosaic_clip_queue=self.mosaic_clip_queue,
                                              device=self.device,
                                              max_clip_length=self.max_clip_length,
                                              clip_size=self.restoration_clip_size,
                                              pad_mode=self.preferred_pad_mode,
                                              batch_size=self._mps_adaptive_profile.get("detect_batch_init", 4),
                                              empty_lookahead_frames=self.mosaic_detection_empty_lookahead,
                                              error_handler=self._on_worker_thread_error)

        self.clip_restoration_thread: PipelineThread | None = None
        self.frame_restoration_thread: PipelineThread | None = None
        self.start_stop_lock: threading.Lock = threading.Lock()
        self.stop_requested = False
        if self.device.type == 'mps':
            logger.info(
                "MPS adaptive profile: restore_chunk(init=%s,min=%s) detect_batch(init=%s) low/mid/high=%.1f/%.1f/%.1f GB",
                self._mps_adaptive_profile["init_chunk"],
                self._mps_adaptive_profile["min_chunk"],
                self._mps_adaptive_profile["detect_batch_init"],
                self._mps_adaptive_profile["low_mem_gb"],
                self._mps_adaptive_profile["mid_mem_gb"],
                self._mps_adaptive_profile["high_mem_gb"],
            )

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
            
            # All worker threads are joined here, so a forced synchronized
            # release is safe and returns unused driver allocations promptly.
            if self.device.type == 'mps':
                try:
                    release_mps_memory_if_needed(force=True, cooldown_seconds=0)
                except Exception as e:
                    logger.debug(f"Could not clear MPS cache: {e}")

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

    def _restore_clip_frames(
        self,
        images: list[ImageTensor],
        masks: list[ImageTensor] | None = None,
    ):
        if self.mosaic_restoration_model_name.startswith("deepmosaics"):
            from lada.restorationpipeline.deepmosaics_mosaic_restorer import DeepmosaicsMosaicRestorer
            assert isinstance(self.mosaic_restoration_model, DeepmosaicsMosaicRestorer)
            restored_clip_images = self.mosaic_restoration_model.restore(images)
        elif self.mosaic_restoration_model_name.startswith("basicvsrpp"):
            from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
            assert isinstance(self.mosaic_restoration_model, BasicvsrppMosaicRestorer)
            restore_max_frames = -1
            if self.restore_max_frames is not None:
                restore_max_frames = self.restore_max_frames
            elif self.device.type == 'mps':
                available_gb = get_mps_available_memory_gb()
                mps_stats = get_mps_memory_stats()
                pressure_ratio = mps_stats.get("pressure_ratio")
                target = self._adaptive_restore_chunk_frames
                min_chunk = self._mps_adaptive_profile["min_chunk"]
                low_mem_gb = self._mps_adaptive_profile["low_mem_gb"]
                mid_mem_gb = self._mps_adaptive_profile["mid_mem_gb"]
                high_mem_gb = self._mps_adaptive_profile["high_mem_gb"]

                if pressure_ratio is not None and pressure_ratio >= 0.92:
                    target = max(min_chunk, target // 2)
                elif pressure_ratio is not None and pressure_ratio >= 0.82:
                    target = max(min_chunk, target - 3)
                elif available_gb < low_mem_gb:
                    target = max(min_chunk, target // 2)
                elif available_gb < mid_mem_gb:
                    target = max(8, target - 2)
                elif available_gb > high_mem_gb and (pressure_ratio is None or pressure_ratio < 0.65):
                    target = min(self.max_clip_length, target + 2)

                self._adaptive_restore_chunk_frames = max(1, min(self.max_clip_length, target))
                restore_max_frames = self._adaptive_restore_chunk_frames
            restored_clip_images = self.mosaic_restoration_model.restore(
                images,
                max_frames=restore_max_frames,
                temporal_overlap=self.restore_temporal_overlap,
                enable_crossfade=self.restore_crossfade,
            )
        elif self.mosaic_restoration_model_name.startswith("mioh-restorer"):
            from lada.restorationpipeline.mioh_restorer import MiohMosaicRestorer

            assert isinstance(self.mosaic_restoration_model, MiohMosaicRestorer)
            restored_clip_images = self.mosaic_restoration_model.restore(
                images,
                masks=masks,
                bidirectional=False,
            )
        else:
            raise NotImplementedError()
        return restored_clip_images

    def _restore_frame(self, frame: ImageTensor, frame_num: int, restored_clips: list[Clip]):
        """
        Takes mosaic frame and restored clips and replaces mosaic regions in frame with restored content from the clips starting at the same frame number as mosaic frame.
        Pops starting frame from each restored clip in the process if they actually start at the same frame number as frame.
        """
        is_cpu_input = frame.device.type == 'cpu'
        target_dtype = torch.float32 if is_cpu_input else self.mosaic_restoration_model.dtype

        def _blend_gpu(blend_mask: torch.Tensor, clip_img: torch.Tensor, orig_clip_box: tuple[int, int, int, int]):
            t, l, b, r = orig_clip_box
            frame_roi = frame[t:b + 1, l:r + 1, :]
            roi_f = frame_roi.to(dtype=self.mosaic_restoration_model.dtype)
            temp = clip_img.to(dtype=self.mosaic_restoration_model.dtype, device=frame_roi.device)
            temp.sub_(roi_f)
            temp.mul_(blend_mask.unsqueeze(-1))
            temp.add_(roi_f)
            temp.round_().clamp_(0, 255)
            frame_roi[:] = temp

        def _blend_cpu(blend_mask: torch.Tensor, clip_img: torch.Tensor, orig_clip_box: tuple[int, int, int, int]):
            blend_mask = blend_mask.cpu().numpy()
            clip_img = clip_img.cpu().numpy()
            t, l, b, r = orig_clip_box
            frame_roi = frame[t:b + 1, l:r + 1, :].numpy()
            temp_buffer = np.empty_like(frame_roi, dtype=np.float32)
            np.subtract(clip_img, frame_roi, out=temp_buffer, dtype=np.float32)
            np.multiply(temp_buffer, blend_mask[..., None], out=temp_buffer)
            np.add(temp_buffer, frame_roi, out=temp_buffer)
            frame_roi[:] = temp_buffer.astype(np.uint8)
            
        blend = _blend_cpu if is_cpu_input else _blend_gpu

        for buffered_clip in [c for c in restored_clips if c.frame_start == frame_num]:
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = buffered_clip.pop()
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
            enhancer_before_resize = (
                self.restore_roi_enhancer_strength > 0
                and self.restore_roi_enhancer is not None
                and getattr(self.restore_roi_enhancer, "prefer_pre_resize", False)
            )
            if enhancer_before_resize:
                if isinstance(clip_img, torch.Tensor):
                    clip_img_np = clip_img.cpu().numpy()
                    clip_mask_np = clip_mask.cpu().numpy() if isinstance(clip_mask, torch.Tensor) else clip_mask
                    clip_img_np = apply_restore_roi_enhancer(
                        clip_img_np,
                        enhancer=self.restore_roi_enhancer,
                        strength=self.restore_roi_enhancer_strength,
                        scale=self.restore_roi_enhancer_scale,
                        mask=clip_mask_np,
                    )
                    clip_img = torch.from_numpy(clip_img_np).to(device=clip_img.device)
                else:
                    clip_img = apply_restore_roi_enhancer(
                        clip_img,
                        enhancer=self.restore_roi_enhancer,
                        strength=self.restore_roi_enhancer_strength,
                        scale=self.restore_roi_enhancer_scale,
                        mask=clip_mask,
                    )
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_mask = image_utils.resize(clip_mask, orig_crop_shape[:2],interpolation=cv2.INTER_NEAREST)
            if (
                self.restore_roi_enhancer_strength > 0
                or self.restore_texture_mix > 0
                or self.restore_detail_boost > 0
                or self.restore_sharpen_strength > 0
                or self.restore_smooth_strength > 0
            ):
                t, l, b, r = orig_clip_box
                original_roi = frame[t:b + 1, l:r + 1, :]
                if isinstance(clip_img, torch.Tensor):
                    clip_img_np = clip_img.cpu().numpy()
                    base_clip_img_np = clip_img_np.copy()
                    clip_mask_np = clip_mask.cpu().numpy() if isinstance(clip_mask, torch.Tensor) else clip_mask
                    original_roi_np = original_roi.cpu().numpy() if isinstance(original_roi, torch.Tensor) else original_roi
                    if not enhancer_before_resize:
                        clip_img_np = apply_restore_roi_enhancer(
                            clip_img_np,
                            enhancer=self.restore_roi_enhancer,
                            strength=self.restore_roi_enhancer_strength,
                            scale=self.restore_roi_enhancer_scale,
                            mask=clip_mask_np,
                        )
                    clip_img_np = apply_restore_effect_upscale(
                        clip_img_np,
                        original_roi_np,
                        clip_mask_np,
                        scale=self.restore_effect_upscale,
                        texture_mix=self.restore_texture_mix,
                        detail_boost=self.restore_detail_boost,
                        sharpen_strength=self.restore_sharpen_strength,
                        smooth_strength=self.restore_smooth_strength,
                    )
                    clip_img_np = _apply_mask_to_processed(base_clip_img_np, clip_img_np, clip_mask_np)
                    clip_img = torch.from_numpy(clip_img_np).to(device=clip_img.device)
                else:
                    base_clip_img = clip_img.copy()
                    if not enhancer_before_resize:
                        clip_img = apply_restore_roi_enhancer(
                            clip_img,
                            enhancer=self.restore_roi_enhancer,
                            strength=self.restore_roi_enhancer_strength,
                            scale=self.restore_roi_enhancer_scale,
                            mask=clip_mask,
                        )
                    clip_img = apply_restore_effect_upscale(
                        clip_img,
                        original_roi,
                        clip_mask,
                        scale=self.restore_effect_upscale,
                        texture_mix=self.restore_texture_mix,
                        detail_boost=self.restore_detail_boost,
                        sharpen_strength=self.restore_sharpen_strength,
                        smooth_strength=self.restore_smooth_strength,
                    )
                    clip_img = _apply_mask_to_processed(base_clip_img, clip_img, clip_mask)
            blend_mask = mask_utils.create_blend_mask(
                clip_mask.float(),
                feather_multiplier=self.restore_blend_feather,
            ).to(device=clip_img.device, dtype=target_dtype)
            blend(blend_mask, clip_img, orig_clip_box)

    def _restore_clip(self, clip: Clip):
        """
        Restores each contained from of the mosaic clip. If self.mosaic_detection is True will instead draw mosaic detection
        boundaries on each frame.
        """
        if self.mosaic_detection:
            restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
        else:
            if (
                self.mosaic_restoration_model_name.startswith("mioh-restorer")
                and clip.masks
            ):
                # Use one temporally stable mask for both model gating and the
                # later full-frame compositor. Otherwise small segmentation
                # contractions reveal a strip of the original mosaic for one
                # frame and produce the characteristic flashing ROI edge.
                clip.masks = mask_utils.stabilize_temporal_masks(clip.masks)
            restored_clip_images = self._restore_clip_frames(clip.frames, clip.masks)
        assert len(restored_clip_images) == len(clip.frames)

        for i in range(len(restored_clip_images)):
            assert clip.frames[i].shape == restored_clip_images[i].shape
            clip.frames[i] = restored_clip_images[i]
        self._move_clip_to_cpu(clip)

    def _move_clip_to_cpu(self, clip: Clip):
        """
        Hand restored clips to the composition thread as CPU tensors. One
        batched transfer per clip is much cheaper than the per-frame
        transfers and device syncs composition triggers otherwise, and it
        lets composition of CPU frames run without the MPS execution lock.
        """
        if self.device.type != 'mps':
            return

        def to_cpu_batched(items):
            if items and all(isinstance(x, torch.Tensor) and x.device.type == 'mps' for x in items):
                for i, item in enumerate(torch.stack(items).cpu().unbind(0)):
                    items[i] = item

        with serialized_mps_execution():
            to_cpu_batched(clip.frames)
            to_cpu_batched(clip.masks)

    def _collect_garbage(self, clip_buffer):
        processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
        has_processed_clips = len(processed_clips) > 0
        for processed_clip in processed_clips:
            clip_buffer.remove(processed_clip)

        if has_processed_clips:
            processed_clips.clear()
            del processed_clip
            if self.device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                released_mps_cache = release_mps_memory_if_needed()
                if released_mps_cache:
                    released_bytes = _release_darwin_malloc_cache()
                    if released_bytes:
                        logger.debug(
                            "Released %.1f MiB of unused malloc pages under memory pressure",
                            released_bytes / (1024 ** 2),
                        )
            else:
                gc.collect()

    def _clip_buffer_contains_all_cips_needed_for_current_restoration(self, current_frame_num, num_mosaic_detections, clip_buffer):
        num_clips_starting_at_frame = len([clip for clip in clip_buffer if clip.frame_start == current_frame_num])
        assert num_clips_starting_at_frame <= num_mosaic_detections
        return num_clips_starting_at_frame == num_mosaic_detections

    def _clip_restoration_worker(self):
        logger.debug("clip restoration worker: started")
        eof = False
        while not (eof or self.stop_requested):
            clip = self.mosaic_clip_queue.get()
            if self.stop_requested or clip is STOP_MARKER:
                logger.debug("clip restoration worker: mosaic_clip_queue consumer unblocked")
                break
            if clip is EOF_MARKER:
                eof = True
                self.restored_clip_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
                    break
            else:
                self._restore_clip(clip)
                if self.device.type == 'mps':
                    release_mps_memory_if_needed()
                self.restored_clip_queue.put(clip)
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
                    break
        if eof:
            logger.debug("clip restoration worker: stopped itself, EOF")
        else:
            logger.debug("clip restoration worker: stopped by request")

    def _read_next_frame(self, video_frames_generator, expected_frame_num) -> tuple[int, np.ndarray, int] | StopMarker | EofMarker:
        try:
            frame, frame_pts = next(video_frames_generator)
        except StopIteration:
            elem = self.frame_detection_queue.get()
            if self.stop_requested or elem is STOP_MARKER:
                logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
                return STOP_MARKER
            assert elem is EOF_MARKER, f"Illegal state: Expected to read EOF_MARKER from detection queue but received f{elem}"
            return EOF_MARKER
        elem = self.frame_detection_queue.get()
        if self.stop_requested or elem is STOP_MARKER:
            logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
            return STOP_MARKER
        assert elem is not EOF_MARKER and elem is not STOP_MARKER, f"Illegal state: Expected to read detection result from detection queue but received {elem}"
        detection_frame_num, num_mosaics_detected = elem
        assert detection_frame_num == expected_frame_num, f"frame detection queue out of sync: received {detection_frame_num} expected {expected_frame_num}"
        return num_mosaics_detected, frame, frame_pts

    def _read_next_clip(self, current_frame_num, clip_buffer) -> StopMarker | EofMarker | None:
        clip = self.restored_clip_queue.get()
        if self.stop_requested or clip is STOP_MARKER:
            logger.debug("frame restoration worker: restored_clip_queue consumer unblocked")
            return STOP_MARKER
        if clip is EOF_MARKER:
            return EOF_MARKER
        assert clip.frame_start >= current_frame_num, "clip queue out of sync!"
        clip_buffer.append(clip)
        return None

    def _frame_restoration_worker(self):
        logger.debug("frame restoration worker: started")
        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()

            frame_num = self.start_frame
            queue_marker = None
            clip_buffer = []

            while not (self.eof or self.stop_requested):
                _frame_result = self._read_next_frame(video_frames_generator, frame_num)
                if self.stop_requested or _frame_result is STOP_MARKER:
                    break
                if _frame_result is EOF_MARKER:
                    self.eof = True
                    self.frame_restoration_queue.put(EOF_MARKER)
                    break
                num_mosaics_detected, frame, frame_pts = _frame_result
                if num_mosaics_detected > 0:
                    while queue_marker is None and not self._clip_buffer_contains_all_cips_needed_for_current_restoration(frame_num, num_mosaics_detected, clip_buffer):
                        queue_marker = self._read_next_clip(frame_num, clip_buffer)
                    if queue_marker is STOP_MARKER:
                        break

                    # Restored clips arrive as CPU tensors, so composing CPU
                    # frames touches no MPS state; only take the MPS lock when
                    # the frame or the optional ROI enhancer runs on the GPU.
                    composition_needs_mps = (
                        frame.device.type != 'cpu'
                        or (self.restore_roi_enhancer is not None and self.restore_roi_enhancer_strength > 0
                            and getattr(self.restore_roi_enhancer, 'uses_torch_device', True))
                    )
                    if self.device.type == 'mps' and composition_needs_mps:
                        with serialized_mps_execution():
                            self._restore_frame(frame, frame_num, clip_buffer)
                    else:
                        self._restore_frame(frame, frame_num, clip_buffer)
                    self.frame_restoration_queue.put((frame, frame_pts))
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                        break
                    self._collect_garbage(clip_buffer)
                else:
                    self.frame_restoration_queue.put((frame, frame_pts))
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                        break
                frame_num += 1
        if self.eof:
            logger.debug("frame restoration worker: stopped itself, EOF")
        else:
            logger.debug("frame restoration worker: stopped by request")

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, int] | ErrorMarker | StopMarker:
        if self.eof and self.frame_restoration_queue.empty():
            raise StopIteration
        else:
            while True:
                elem = self.frame_restoration_queue.get()
                if self.stop_requested or elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                    logger.debug("frame_restoration_queue consumer unblocked")
                    return elem
                if elem is EOF_MARKER:
                    raise StopIteration
                return elem

    def get_frame_restoration_queue(self) -> PipelineQueue:
        return self.frame_restoration_queue
