# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import math
import os.path
import pathlib
import queue
import concurrent.futures as concurrent_futures
from dataclasses import dataclass
from typing import Generator, Optional, Dict

import cv2
import numpy as np
import torch
import ultralytics.models

from lada import LOG_LEVEL
from lada.utils import Mask, Image, Box, VideoMetadata, threading_utils, video_utils
from lada.utils import mask_utils
from lada.utils.scene_utils import crop_to_box_v3
from lada.utils.threading_utils import wait_until_completed
from lada.utils.ultralytics_utils import choose_biggest_detection, convert_yolo_mask, convert_yolo_box
from lada.models.yolo.yolo import Yolo

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@dataclass
class FileProcessingOptions:
    input_dir: str | tuple[pathlib.Path, ...]
    output_dir: pathlib.Path
    start_index: int
    stride_length: int
    scene_min_length: float
    scene_max_length: float
    scene_max_memory: int
    random_extend_masks: bool
    skip4k: bool
    scene_min_frames: int = 24
    scene_continuity_iou: float = 0.2
    scene_gap_frames: int = 3
    probe_min_crop_size: int = 192
    probe_crop_target_size: int = 256
    detection_start_confidence: float = 0.6
    detection_continue_confidence: float = 0.25

@dataclass
class NsfwFrame:
    video_metadata: VideoMetadata
    frame_number: int
    last_frame: bool
    frame: Image
    _box: ultralytics.engine.results.Boxes
    _mask: ultralytics.engine.results.Masks
    object_detected: bool = False
    object_id: int = None
    confidence: float | None = None
    interpolated_mask: Optional[Mask] = None
    interpolated_box: Optional[Box] = None

    @property
    def mask(self) -> Mask:
        if self.interpolated_mask is not None:
            return self.interpolated_mask
        mask = convert_yolo_mask(self._mask, self.frame.shape)
        # TODO: use mask_utils.clean_mask()
        return mask

    @property
    def box(self) -> Box:
        if self.interpolated_box is not None:
            return self.interpolated_box
        return convert_yolo_box(self._box, self.frame.shape)


def box_iou(box_a: Box, box_b: Box) -> float:
    """Return IoU for boxes represented as top, left, bottom, right."""
    top = max(box_a[0], box_b[0])
    left = max(box_a[1], box_b[1])
    bottom = min(box_a[2], box_b[2])
    right = min(box_a[3], box_b[3])
    intersection_height = max(0, bottom - top + 1)
    intersection_width = max(0, right - left + 1)
    intersection = intersection_height * intersection_width
    area_a = max(0, box_a[2] - box_a[0] + 1) * max(0, box_a[3] - box_a[1] + 1)
    area_b = max(0, box_b[2] - box_b[0] + 1) * max(0, box_b[3] - box_b[1] + 1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def interpolate_box(box_a: Box, box_b: Box, alpha: float) -> Box:
    """Linearly interpolate a top/left/bottom/right box."""
    return tuple(
        int(round(start + (end - start) * alpha))
        for start, end in zip(box_a, box_b)
    )


def move_mask_to_box(mask: Mask, source_box: Box, target_box: Box) -> Mask:
    """Move and resize a detected mask into an interpolated box."""
    source_top, source_left, source_bottom, source_right = source_box
    target_top, target_left, target_bottom, target_right = target_box
    height, width = mask.shape[:2]
    source_top = min(max(source_top, 0), height - 1)
    source_bottom = min(max(source_bottom, source_top), height - 1)
    source_left = min(max(source_left, 0), width - 1)
    source_right = min(max(source_right, source_left), width - 1)
    target_top = min(max(target_top, 0), height - 1)
    target_bottom = min(max(target_bottom, target_top), height - 1)
    target_left = min(max(target_left, 0), width - 1)
    target_right = min(max(target_right, target_left), width - 1)
    source_crop = mask[source_top:source_bottom + 1, source_left:source_right + 1]
    target_width = target_right - target_left + 1
    target_height = target_bottom - target_top + 1
    if source_crop.size == 0 or target_width <= 0 or target_height <= 0:
        return np.zeros_like(mask)
    resized = cv2.resize(source_crop, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    result = np.zeros_like(mask)
    result[target_top:target_bottom + 1, target_left:target_right + 1] = resized
    return result


def build_probe_windows(frame_count: int, stride_frames: int) -> list[tuple[int, int, int]]:
    """Return inclusive start/end windows and the probe frame at each window end."""
    if frame_count <= 0 or stride_frames <= 0:
        return []
    windows = []
    window_start = 0
    while window_start < frame_count:
        window_end = min(window_start + stride_frames - 1, frame_count - 1)
        windows.append((window_start, window_end, window_end))
        window_start = window_end + 1
    return windows


def merge_adjacent_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge selected probe windows so continuous regions keep one tracker session."""
    merged = []
    for window_start, window_end in windows:
        if merged and window_start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], window_end))
        else:
            merged.append((window_start, window_end))
    return merged


def crop_meets_minimum_size(
    frame: Image,
    mask: Mask,
    box: Box,
    target_size: int,
    minimum_size: int,
) -> tuple[bool, tuple[int, int]]:
    """Evaluate a detection using the same crop expansion used by dataset output."""
    cropped_frame, _, _, _ = crop_to_box_v3(
        box,
        frame,
        mask,
        (target_size, target_size),
        border_size=0.08,
    )
    height, width = cropped_frame.shape[:2]
    return height >= minimum_size or width >= minimum_size, (width, height)


def detection_meets_confidence(nsfw_frame: NsfwFrame, minimum_confidence: float) -> bool:
    return (
        nsfw_frame.object_detected
        and nsfw_frame.confidence is not None
        and nsfw_frame.confidence >= minimum_confidence
    )


class Scene:

    def __init__(self, video_meta_data, id, scene_min_length, scene_max_length):
        self.video_meta_data: VideoMetadata = video_meta_data
        self.id: int = id
        self.data: Optional[list] = None # will be set when complete() is called
        self._tmp_data: list[NsfwFrame] = []
        self.realized = False
        self.frame_start: int | None = None
        self.frame_end: int | None = None
        self._index: int = 0
        self.scene_max_length: int = scene_max_length
        self.scene_min_length: int = scene_min_length

    def __len__(self):
        return len(self.data) if self.data else len(self._tmp_data)

    def min_length_reached(self):
        return len(self) >= self.scene_min_length

    def max_length_reached(self):
        return len(self) >= self.scene_max_length

    def continues_with(
        self,
        nsfw_frame: NsfwFrame,
        min_iou: float,
        max_gap_frames: int = 0,
    ) -> bool:
        if not self._tmp_data:
            return False
        frame_gap = nsfw_frame.frame_number - self.frame_end - 1
        if frame_gap < 0 or frame_gap > max_gap_frames:
            return False
        same_tracking_id = (
            self.id is not None
            and nsfw_frame.object_id is not None
            and self.id == nsfw_frame.object_id
        )
        if same_tracking_id:
            return True
        return box_iou(self._tmp_data[-1].box, nsfw_frame.box) >= min_iou

    def add_interpolated_gap(self, gap_frames: list[NsfwFrame], next_frame: NsfwFrame):
        """Bridge missed detections using the surrounding masks and boxes."""
        if not gap_frames:
            return
        if len(self) + len(gap_frames) >= self.scene_max_length:
            raise ValueError("interpolated gap would exceed complete scene")
        previous_frame = self._tmp_data[-1]
        previous_box = previous_frame.box
        next_box = next_frame.box
        previous_mask = previous_frame.mask
        next_mask = next_frame.mask
        denominator = len(gap_frames) + 1
        for index, gap_frame in enumerate(gap_frames, start=1):
            alpha = index / denominator
            target_box = interpolate_box(previous_box, next_box, alpha)
            if alpha <= 0.5:
                source_mask, source_box = previous_mask, previous_box
            else:
                source_mask, source_box = next_mask, next_box
            gap_frame.interpolated_box = target_box
            gap_frame.interpolated_mask = move_mask_to_box(source_mask, source_box, target_box)
            self.add_frame(gap_frame)

    def add_frame(self, nsfw_frame: NsfwFrame):
        if self.max_length_reached():
            raise ValueError("cannot add a frame to a complete scene")
        if self.frame_start is None:
            self.frame_start = nsfw_frame.frame_number
            self.frame_end = nsfw_frame.frame_number
            self._tmp_data.append(nsfw_frame)
        else:
            assert nsfw_frame.frame_number == self.frame_end + 1
            self.frame_end = nsfw_frame.frame_number
            self._tmp_data.append(nsfw_frame)

    def complete(self):
        worker_count = 6
        def _convert_data_from_yolo(chunk, chunk_idx_start, chunk_idx_exclusive_end):
            for i, nsfw_frame in enumerate(self._tmp_data[chunk_idx_start:chunk_idx_exclusive_end], start=chunk_idx_start):
                chunk.append((nsfw_frame.frame, nsfw_frame.mask, nsfw_frame.box))

        with concurrent_futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            chunk_indices = list(np.linspace(0, len(self), num=worker_count, dtype=int, endpoint=False))
            futures = []
            chunks = []
            for j, chunk_idx_start in enumerate(chunk_indices):
                chunk_idx_exclusive_end = chunk_indices[j+1] if chunk_idx_start != chunk_indices[-1] else len(self)
                chunk = []
                chunks.append(chunk)
                futures.append(executor.submit(_convert_data_from_yolo, chunk, chunk_idx_start, chunk_idx_exclusive_end))
            wait_until_completed(futures)
            self.data = []
            for chunk in chunks:
                self.data.extend(chunk)
            assert len(self.data) == len(self._tmp_data)
            self._tmp_data = None

    def get_images(self) -> list[Image]:
        return [img for img, _, _ in self.data]

    def get_masks(self) -> list[Mask]:
        return [mask for _, mask, _ in self.data]

    def get_boxes(self) -> list[Box]:
        return [box for _, _, box in self.data]

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, Mask, Box]:
        if self._index < len(self):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item) -> tuple[Image, Mask, Box]:
        return self.data[item]


class CroppedScene:

    def __init__(self, scene: Scene, window_in_seconds=1.0, target_size=(400,400), smoothing=True, border_size=0):
        self.video_meta_data: VideoMetadata = scene.video_meta_data
        self.id: int = scene.id
        self.data: list = []
        self._index: int = 0

        if smoothing:
            smoothed_boxes = SmoothSceneBoxes.smooth_boxes(scene, window_in_seconds, smooth_function='median')
        else:
            smoothed_boxes = scene.get_boxes()
        scene_images = scene.get_images()
        scene_mask_images = scene.get_masks()

        for i, smoothed_box in enumerate(smoothed_boxes):
            cropped_image, cropped_mask_image, cropped_box, scale_factor = crop_to_box_v3(smoothed_box, scene_images[i],
                                                                         scene_mask_images[i], target_size, border_size=border_size)
            self.data.append((cropped_image, cropped_mask_image, cropped_box))

    def __len__(self):
        return len(self.data)

    def get_images(self) -> list[Image]:
        return [img for img, _, _ in self.data]

    def get_masks(self) -> list[Mask]:
        return [mask for _, mask, _ in self.data]

    def get_boxes(self) -> list[Box]:
        """
        Location of cropped area in original image
        """
        return [box for _, _, box in self.data]

    def get_max_width_height(self):
        max_width = 0
        max_height = 0
        for _, _, box in self.data:
            t, l, b, r = box
            width, height = r - l + 1, b - t + 1
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width
        return max_width, max_height

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, Mask, Box]:
        if self._index < len(self):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item) -> tuple[Image, Mask, Box]:
        return self.data[item]


class SmoothSceneBoxes:

    @staticmethod
    def median_filter(data, window=11):
        assert window % 2 != 0
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        data_type = data[0]

        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros(data_size, dtype=data_type)
        for i in range(data_size):
            filtered_data[i] = np.median(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def mean_filter(data, window=11):
        assert window % 2 != 0
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        data_type = data[0]
        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros(data_size, dtype=data_type)
        for i in range(data_size):
            filtered_data[i] = np.mean(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def min_max_filter(data, window, mode):
        assert window % 2 != 0
        func = np.max if mode == 'max' else np.min
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros_like(data)
        for i in range(data_size):
            filtered_data[i] = func(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def smooth_boxes(scene: Scene, window_in_seconds: float, smooth_function='median'):
        _scene_boxes = np.array(scene.get_boxes())
        window_in_frames = min(math.ceil(window_in_seconds * scene.video_meta_data.video_fps), len(scene))
        if window_in_frames % 2 == 0:
            window_in_frames -= 1
        if window_in_frames < 1:
            return _scene_boxes.tolist()

        for i, position in zip(range(4),('t','l','b','r')):
            if smooth_function == 'median':
                _scene_boxes[:, i] = SmoothSceneBoxes.median_filter(_scene_boxes[:, i], window_in_frames)
            elif smooth_function == 'min_max':
                _scene_boxes[:, i] = SmoothSceneBoxes.min_max_filter(_scene_boxes[:, i], window_in_frames, 'max' if position in ('b', 'r') else 'min')
            elif smooth_function == 'mean':
                _scene_boxes[:, i] = SmoothSceneBoxes.mean_filter(_scene_boxes[:, i], window_in_frames)
            else:
                raise NotImplementedError()

        return _scene_boxes.tolist()

    @staticmethod
    def smooth_boxes_center_point(scene: Scene, window_in_seconds: float, smooth_function='median'):
        _scene_boxes = np.array(scene.get_boxes())

        window_in_frames = min(math.ceil(window_in_seconds * scene.video_meta_data.video_fps), len(scene))
        if window_in_frames % 2 == 0:
            window_in_frames -= 1

        heights = _scene_boxes[:, 2] - _scene_boxes[:, 0]
        widths = _scene_boxes[:, 3] - _scene_boxes[:, 1]

        half_widths = widths / 2
        half_heights = heights / 2

        if smooth_function == 'median':
            smoothed_heights = SmoothSceneBoxes.median_filter(heights, window_in_frames)
            smoothed_widths = SmoothSceneBoxes.median_filter(widths, window_in_frames)
            center_x = SmoothSceneBoxes.median_filter(_scene_boxes[:, 1] + half_widths, window_in_frames)
            center_y = SmoothSceneBoxes.median_filter(_scene_boxes[:, 0] + half_heights, window_in_frames)
        else:
            raise NotImplementedError()

        half_smoothed_widths = smoothed_widths / 2
        half_smoothed_heights = smoothed_heights / 2

        new_t = np.clip((center_y - half_smoothed_heights).round().astype(np.int64), 0, scene.video_meta_data.video_height)
        new_b = np.clip((center_y + half_smoothed_heights).round().astype(np.int64), 0, scene.video_meta_data.video_height)
        new_l = np.clip((center_x - half_smoothed_widths).round().astype(np.int64), 0, scene.video_meta_data.video_width)
        new_r = np.clip((center_x + half_smoothed_widths).round().astype(np.int64), 0, scene.video_meta_data.video_width)

        smoothed_boxes = np.stack((new_t, new_l, new_b, new_r), axis=-1)
        return smoothed_boxes.tolist()

def determine_max_scene_length(video_metadata: VideoMetadata, limit_seconds: int | None, limit_memory: int | None):
    scene_max_length = None
    if limit_seconds:
        scene_max_length = limit_seconds
    if limit_memory:
        scene_max_length_memory = video_utils.approx_max_length_by_memory_limit(video_metadata, limit_memory)
        scene_max_length = min(scene_max_length, scene_max_length_memory) if scene_max_length else scene_max_length_memory
    return scene_max_length


def determine_min_scene_frames(
    video_metadata: VideoMetadata,
    minimum_frames: int,
    minimum_seconds: float | None,
) -> int:
    seconds_frames = math.ceil((minimum_seconds or 0) * video_metadata.video_fps)
    return max(minimum_frames, seconds_frames)

def apply_random_mask_extensions(scene: Scene):
    value = np.random.choice([0, 0, 1, 1, 2])
    worker_count = 6
    def _apply_random_mask_extensions(chunk_idx_start, chunk_idx_exclusive_end):
        for i, (img, mask, _) in enumerate(scene.data[chunk_idx_start:chunk_idx_exclusive_end], start=chunk_idx_start):
            mask_extended = mask_utils.extend_mask(mask, value)
            box_extended = mask_utils.get_box(mask_extended)
            scene.data[i] = img, mask_extended, box_extended

    with concurrent_futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        chunk_indices = list(np.linspace(0, len(scene), num=worker_count, dtype=int, endpoint=False))
        futures = []
        for j, chunk_idx_start in enumerate(chunk_indices):
            chunk_idx_exclusive_end = chunk_indices[j+1] if chunk_idx_start != chunk_indices[-1] else len(scene)
            futures.append(executor.submit(_apply_random_mask_extensions, chunk_idx_start, chunk_idx_exclusive_end))
        wait_until_completed(futures)


class NsfwDetector:
    def __init__(self, nsfw_detection_model: Yolo, device: str, file_queue: queue.Queue, frame_queue: queue.Queue, scene_queue: queue.Queue, file_processing_options: FileProcessingOptions, random_extend_masks=True):
        self.nsfw_detection_model: Yolo = nsfw_detection_model
        self.device = torch.device(device) if device is not None else device
        self.file_queue: queue.Queue = file_queue
        self.frame_queue: queue.Queue = frame_queue
        self.scene_queue: queue.Queue = scene_queue
        self.file_processing_options = file_processing_options

        self.metadata: Dict[str, VideoMetadata] = {}
        self.previous_completed_scene_frame_end: Dict[str, Optional[int]] = {}
        self.scenes_counter: Dict[str, int] = {}
        self.random_extend_masks = random_extend_masks

        self.stop_requested = False
        self.thread_pool = concurrent_futures.ThreadPoolExecutor()
        self.frame_detector_thread_futures: list[concurrent_futures.Future] = []
        self.scene_detector_thread_futures: list[concurrent_futures.Future] = []
        # todo: frame thread is faster than scene thread so ideally we could scale it up to multiple consumers. Needs some refactoring first to preserve order of frames
        #  also, more frame threads (processing more than a single file) could be an improvement when NSFW detection becomes a bottleneck when running dataset creation script.
        self.frame_detector_thread_count = 1
        self.scene_detector_thread_count = 1
        self.frame_detector_thread_should_be_running = False
        self.scene_detector_thread_should_be_running = False

        self.no_nsfw_scenes_found_file: pathlib.Path = file_processing_options.output_dir.joinpath("no_nsfw_scenes.txt")
        self.done_processing_file: pathlib.Path = file_processing_options.output_dir.joinpath("done_processing.txt")
        self.files_already_processed: set[str] = set([])
        for file in (self.no_nsfw_scenes_found_file, self.done_processing_file):
            if file.exists():
                with open(file, 'r', encoding='utf-8') as f:
                    for file_path in f:
                        self.files_already_processed.add(file_path.strip())

    def _mark_file_as_processed(self, text_file: pathlib.Path, path_to_save: str):
        if not text_file.exists():
            text_file.parent.mkdir(parents=True, exist_ok=True)
            text_file.touch()
        with open(text_file, 'a', encoding='utf-8') as f:
            f.write(f"{path_to_save}\n")

    def _file_already_processed(self, path_to_check: str):
        return str(path_to_check) in self.files_already_processed and os.path.exists(path_to_check)

    def _process_completed_scene(self, completed_scene: Scene) -> Optional[Scene]:
        """returns Scene if it fits the criteria for a valid completed scene like min/max length"""
        video_file = completed_scene.video_meta_data.video_file
        skip_scene = not completed_scene.min_length_reached()
        if skip_scene:
            return None
        completed_scene.complete()
        self.scenes_counter[video_file] += 1
        completed_scene.id = self.scenes_counter[video_file]
        self.previous_completed_scene_frame_end[video_file] = completed_scene.frame_end
        if self.random_extend_masks:
            apply_random_mask_extensions(completed_scene)
        return completed_scene

    def _init_new_file(self, metadata: VideoMetadata):
        file_path = metadata.video_file
        self.metadata[file_path] = metadata
        self.scene_min_length = determine_min_scene_frames(
            metadata,
            self.file_processing_options.scene_min_frames,
            self.file_processing_options.scene_min_length,
        )
        scene_max_length = determine_max_scene_length (metadata, self.file_processing_options.scene_max_length, self.file_processing_options.scene_max_memory)
        self.scene_max_length = math.ceil(scene_max_length * metadata.video_fps)
        self.stride_length_frames = math.ceil(self.file_processing_options.stride_length * metadata.video_fps)
        self.previous_completed_scene_frame_end[file_path] = None
        self.scenes_counter[file_path] = 0

    def _check_file(self, file_index: 0, file_path: str) -> Optional[VideoMetadata]:
        file_name = pathlib.Path(file_path)
        if file_index < self.file_processing_options.start_index or self._file_already_processed(file_path):
            print(f"{file_index}, Skipping {file_name}: Already processed")
            return None
        if not video_utils.is_video_file(file_path):
            print(f"{file_index}, Skipping {file_name}: Unsupported file format")
            return None
        video_metadata = video_utils.get_video_meta_data(file_path)
        if self.file_processing_options.skip4k and max(video_metadata.video_width, video_metadata.video_height) > 2_000:
            print(f"{file_index}, Skipping {file_name}: 4K")
            return None
        scene_max_length = determine_max_scene_length (video_metadata, self.file_processing_options.scene_max_length, self.file_processing_options.scene_max_memory)
        scene_max_frames = math.ceil(scene_max_length * video_metadata.video_fps)
        scene_min_frames = determine_min_scene_frames(
            video_metadata,
            self.file_processing_options.scene_min_frames,
            self.file_processing_options.scene_min_length,
        )
        if scene_max_frames < scene_min_frames:
            print(f"{file_index}, Skipping {file_name}: Scene maximum length is less than minimum length")
            return None
        return video_metadata

    def add_files(self, video_files):
        for file_index, file_path in enumerate(video_files):
            self.file_queue.put((file_index, file_path))
        self.file_queue.put(None)

    def _frame_detector_worker(self):
        logger.debug("NsfwDetector: frame detector worker: started")
        while self.frame_detector_thread_should_be_running:
            item: tuple[int, str] | None = self.file_queue.get()
            if self.stop_requested:
                logger.debug("NsfwDetector: frame detector worker: file_queue consumer unblocked")
            if self.stop_requested:
                break
            if not item:
                self.frame_queue.put(None)
                break
            video_file_index, video_file_path = item
            video_metadata = self._check_file(video_file_index, video_file_path)
            if not video_metadata:
                continue
            if video_file_path not in self.metadata:
                self._init_new_file(video_metadata)
            print(f"{video_file_index}, Processing {pathlib.Path(video_file_path).name}")
            if self.stride_length_frames > 0:
                frame_iterator = self._adaptive_frame_iterator(video_metadata)
            else:
                frame_iterator = self._full_frame_iterator(video_metadata)

            nsfw_frame = None
            for nsfw_frame in frame_iterator:
                if nsfw_frame:
                    self.frame_queue.put(nsfw_frame)
                if self.stop_requested:
                    logger.debug("NsfwDetector: frame detector worker: frame_queue producer unblocked")
                    break
            if nsfw_frame is None and not self.stop_requested:
                # Even a file without a selected full-scan window must reach the
                # scene worker so it can be marked as processed.
                frame = np.zeros((1, 1, 3), dtype=np.uint8)
                nsfw_frame = NsfwFrame(
                    video_metadata,
                    max(video_metadata.frames_count - 1, 0),
                    True,
                    frame,
                    None,
                    None,
                    False,
                    None,
                )
                self.frame_queue.put(nsfw_frame)

    @staticmethod
    def _result_to_nsfw_frame(video_metadata: VideoMetadata, frame_number: int, result) -> NsfwFrame:
        # Keep detections without a ByteTrack ID as candidates. Scene
        # continuity is decided using both tracking ID and box IoU.
        yolo_box, yolo_mask = choose_biggest_detection(result, tracking_mode=False)
        object_detected = yolo_box is not None
        tracking_id = (
            int(yolo_box.id.item())
            if object_detected and yolo_box.id is not None
            else None
        )
        confidence = float(yolo_box.conf[0].item()) if object_detected else None
        return NsfwFrame(
            video_metadata,
            frame_number,
            False,
            result.orig_img,
            yolo_box,
            yolo_mask,
            object_detected,
            tracking_id,
            confidence,
        )

    def _full_frame_iterator(self, video_metadata: VideoMetadata) -> Generator[NsfwFrame, None, None]:
        previous_frame = None
        for frame_number, result in enumerate(self.nsfw_detection_model.track(
            source=video_metadata.video_file,
            stream=True,
            verbose=False,
            tracker="bytetrack.yaml",
            device=self.device,
            conf=self.file_processing_options.detection_continue_confidence,
        )):
            current_frame = self._result_to_nsfw_frame(video_metadata, frame_number, result)
            if previous_frame is not None:
                yield previous_frame
            previous_frame = current_frame
        if previous_frame is not None:
            previous_frame.last_frame = True
            yield previous_frame

    def _reset_yolo_trackers(self):
        predictor = getattr(self.nsfw_detection_model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    def _probe_window(self, capture: cv2.VideoCapture, video_metadata: VideoMetadata, window: tuple[int, int, int]):
        window_start, window_end, probe_frame_number = window
        capture.set(cv2.CAP_PROP_POS_FRAMES, probe_frame_number)
        success, frame = capture.read()
        if not success:
            print(f"  Probe {probe_frame_number}: decode failed, scanning window as a precaution")
            return window_start, window_end

        results = self.nsfw_detection_model.predict(
            source=frame,
            verbose=False,
            device=self.device,
            conf=self.file_processing_options.detection_start_confidence,
        )
        probe = self._result_to_nsfw_frame(video_metadata, probe_frame_number, results[0])
        if not probe.object_detected:
            print(f"  Probe {probe_frame_number}: no detection, skipping window")
            return None

        meets_minimum, (crop_width, crop_height) = crop_meets_minimum_size(
            probe.frame,
            probe.mask,
            probe.box,
            self.file_processing_options.probe_crop_target_size,
            self.file_processing_options.probe_min_crop_size,
        )
        if not meets_minimum:
            print(f"  Probe {probe_frame_number}: crop {crop_width}x{crop_height}, skipping window")
            return None
        print(
            f"  Probe {probe_frame_number}: crop {crop_width}x{crop_height}, "
            f"full scan {window_start}-{window_end}"
        )
        return window_start, window_end

    def _adaptive_frame_iterator(self, video_metadata: VideoMetadata) -> Generator[NsfwFrame, None, None]:
        windows = build_probe_windows(video_metadata.frames_count, self.stride_length_frames)
        print(
            f"  Adaptive scan: one probe every {self.file_processing_options.stride_length:g}s; "
            f"full scan when crop has a dimension >= {self.file_processing_options.probe_min_crop_size}px"
        )
        with video_utils.VideoReaderOpenCV(video_metadata.video_file) as capture:
            selected_windows = []
            for window in windows:
                if self.stop_requested:
                    return
                selected_window = self._probe_window(capture, video_metadata, window)
                if selected_window is not None:
                    selected_windows.append(selected_window)

            previous_frame = None
            for window_start, window_end in merge_adjacent_windows(selected_windows):
                if self.stop_requested:
                    return
                self._reset_yolo_trackers()
                capture.set(cv2.CAP_PROP_POS_FRAMES, window_start)
                for frame_number in range(window_start, window_end + 1):
                    success, frame = capture.read()
                    if not success:
                        logger.warning("Unable to decode frame %d from %s", frame_number, video_metadata.video_file)
                        break
                    results = self.nsfw_detection_model.track(
                        source=frame,
                        persist=True,
                        verbose=False,
                        tracker="bytetrack.yaml",
                        device=self.device,
                        conf=self.file_processing_options.detection_continue_confidence,
                    )
                    current_frame = self._result_to_nsfw_frame(video_metadata, frame_number, results[0])
                    if previous_frame is not None:
                        yield previous_frame
                    previous_frame = current_frame

            if previous_frame is not None:
                previous_frame.last_frame = True
                yield previous_frame
                if self.stop_requested:
                    logger.debug("NsfwDetector: frame detector worker: frame_queue producer unblocked")

    def _scene_detector_worker(self):
        logger.debug("NsfwDetector: scene detector worker: started")

        scene: Scene | None = None
        nsfw_frame: NsfwFrame
        previous_file: str = None
        previous_file_no_completed_scenes = True
        pending_gap_frames: list[NsfwFrame] = []

        while self.scene_detector_thread_should_be_running:
            nsfw_frame: NsfwFrame | None = self.frame_queue.get()
            if self.stop_requested:
                logger.debug("NsfwDetector: scene detector worker: frame_queue consumer unblocked")
            if self.stop_requested:
                break
            if not nsfw_frame:
                self.scene_queue.put(None)
                if self.stop_requested:
                    logger.debug("NsfwDetector: frame detector worker: scene_queue producer unblocked")
                break

            if not previous_file:
                previous_file = nsfw_frame.video_metadata.video_file
            new_file = previous_file != nsfw_frame.video_metadata.video_file
            if new_file:
                if previous_file_no_completed_scenes:
                    self._mark_file_as_processed(self.no_nsfw_scenes_found_file, previous_file)
                previous_file = nsfw_frame.video_metadata.video_file
                previous_file_no_completed_scenes = True
                pending_gap_frames = []

            can_start_scene = detection_meets_confidence(
                nsfw_frame,
                self.file_processing_options.detection_start_confidence,
            )
            can_continue_scene = detection_meets_confidence(
                nsfw_frame,
                self.file_processing_options.detection_continue_confidence,
            )

            if scene is None:
                if can_start_scene:
                    scene = Scene(nsfw_frame.video_metadata, nsfw_frame.object_id, self.scene_min_length, self.scene_max_length)
                    scene.add_frame(nsfw_frame)
                pending_gap_frames = []
            elif can_continue_scene:
                required_frames = len(pending_gap_frames) + 1
                can_fit = len(scene) + required_frames <= scene.scene_max_length
                continues_same_scene = scene.continues_with(
                    nsfw_frame,
                    self.file_processing_options.scene_continuity_iou,
                    max_gap_frames=self.file_processing_options.scene_gap_frames,
                )
                if continues_same_scene and can_fit:
                    scene.add_interpolated_gap(pending_gap_frames, nsfw_frame)
                    scene.add_frame(nsfw_frame)
                else:
                    completed_scene = self._process_completed_scene(scene)
                    if completed_scene:
                        previous_file_no_completed_scenes = False
                        self.scene_queue.put(completed_scene)
                        if self.stop_requested:
                            logger.debug("NsfwDetector: frame detector worker: scene_queue producer unblocked")
                    scene = None
                    if can_start_scene:
                        scene = Scene(nsfw_frame.video_metadata, nsfw_frame.object_id, self.scene_min_length, self.scene_max_length)
                        scene.add_frame(nsfw_frame)
                pending_gap_frames = []
            elif not nsfw_frame.last_frame:
                pending_gap_frames.append(nsfw_frame)
                if (
                    len(pending_gap_frames) > self.file_processing_options.scene_gap_frames
                    or scene.max_length_reached()
                ):
                    completed_scene = self._process_completed_scene(scene)
                    if completed_scene:
                        previous_file_no_completed_scenes = False
                        self.scene_queue.put(completed_scene)
                        if self.stop_requested:
                            logger.debug("NsfwDetector: frame detector worker: scene_queue producer unblocked")
                    scene = None
                    pending_gap_frames = []

            if scene is not None and nsfw_frame.last_frame:
                completed_scene = self._process_completed_scene(scene)
                if completed_scene and not self.stop_requested:
                    previous_file_no_completed_scenes = False
                    self.scene_queue.put(completed_scene)
                    if self.stop_requested:
                        logger.debug("NsfwDetector: frame detector worker: scene_queue producer unblocked")
                scene = None
                pending_gap_frames = []

            if nsfw_frame.last_frame:
                self._mark_file_as_processed(self.done_processing_file, nsfw_frame.video_metadata.video_file)

    def __call__(self) -> Generator[Scene, None, None]:
        while not self.stop_requested:
            elem = self.scene_queue.get()
            if self.stop_requested:
                logger.debug("scene_queue consumer unblocked")
            if elem is None and not self.stop_requested:
                self.stop()
                break
            yield elem

    def start(self):
        self.stop_requested = False
        self.frame_detector_thread_should_be_running = True
        self.scene_detector_thread_should_be_running = True

        for i in range(self.frame_detector_thread_count):
            self.frame_detector_thread_futures.append(self.thread_pool.submit(self._frame_detector_worker))
        for i in range(self.scene_detector_thread_count):
            self.scene_detector_thread_futures.append(self.thread_pool.submit(self._scene_detector_worker))

    def stop(self):
        logger.debug("NsfwDetector: stopping...")
        self.stop_requested = True
        self.frame_detector_thread_should_be_running = False
        self.scene_detector_thread_should_be_running = False

        # unblock consumer
        for i in range(self.frame_detector_thread_count): threading_utils.put_queue_stop_marker(self.frame_queue, "file_queue", stop_marker=None)
        # unblock producer
        threading_utils.empty_out_queue_until_futures_are_done(self.scene_queue, "frame_queue", self.frame_detector_thread_futures)
        concurrent_futures.wait(self.frame_detector_thread_futures, return_when=concurrent_futures.ALL_COMPLETED)
        logger.debug("NsfwDetector: frame detector worker: stopped")
        self.frame_detector_thread_futures = []

        # unblock consumer
        threading_utils.put_queue_stop_marker(self.scene_queue, "scene_queue", stop_marker=None)
        for i in range(self.scene_detector_thread_count): threading_utils.put_queue_stop_marker(self.frame_queue, "frame_queue", stop_marker=None)
        # unblock producer
        threading_utils.empty_out_queue_until_futures_are_done(self.scene_queue, "scene_queue", self.scene_detector_thread_futures)
        wait_until_completed(self.scene_detector_thread_futures)
        concurrent_futures.wait(self.scene_detector_thread_futures, return_when=concurrent_futures.ALL_COMPLETED)
        logger.debug("NsfwDetector: scene detector worker: stopped")
        self.scene_detector_thread_futures = []

        # garbage collection
        threading_utils.empty_out_queue(self.file_queue, "file_queue")
        threading_utils.empty_out_queue(self.file_queue, "frame_queue")
        threading_utils.empty_out_queue(self.scene_queue, "scene_queue")

        logger.debug(f"NsfwDetector: stopped")
