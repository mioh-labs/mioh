# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import concurrent.futures as concurrent_futures
import queue
from pathlib import Path
from time import sleep

from lada.datasetcreation.detectors.mosaic_detector import MosaicDetector
from lada.datasetcreation.nsfw_scene_detector import NsfwDetector, FileProcessingOptions
from lada.datasetcreation.nsfw_scene_processor import SceneProcessingOptions, SceneProcessor
from lada.datasetcreation.detectors.nudenet_nsfw_detector import NudeNetNsfwDetector
from lada.utils.threading_utils import wait_until_completed, clean_up_completed_futures
from lada.datasetcreation.detectors.watermark_detector import WatermarkDetector
from lada.models.yolo.yolo import Yolo
from lada.utils import video_utils

def parse_args(argv=None):
    parser = argparse.ArgumentParser("Create mosaic restoration dataset")
    parser.add_argument('--workers', type=int, default=2, help="Set number of multiprocessing workers")

    input = parser.add_argument_group('Input')
    input.add_argument('--input', type=Path, nargs='+', required=True,
                       help="one or more video files/directories containing uncensored source videos")
    input.add_argument('--start-index', type=int, default=0, help="Can be used to continue a previous run. Note the index number next to last processed file name")
    input.add_argument('--stride-length', default=0, type=int,
                       help="adaptive scan interval in seconds: probe the end of each interval and fully scan the preceding interval only when the crop meets --min-crop-size; 0 scans every frame")
    input.add_argument('--skip-4k', default=True, action=argparse.BooleanOptionalAction, help="skip videos of 4K resolution or higher. Processing those will use a lot of RAM")


    output = parser.add_argument_group('Output')
    output.add_argument('--output-root', type=Path, default='video_dataset', help="path to directory where dataset should be stored")
    output.add_argument('--out-size', type=int, default=256, help="size (in pixel) of output images")
    output.add_argument('--min-crop-size', type=int, default=192,
                        help="Skip cropped scenes when both source crop dimensions are smaller than this value (default: 192)")
    output.add_argument('--save-uncropped', default=False, action=argparse.BooleanOptionalAction,
                        help="Save uncropped, full-size images and masks")
    output.add_argument('--save-cropped', default=True, action=argparse.BooleanOptionalAction,
                        help="Save cropped images and masks")
    output.add_argument('--resize-crops', default=False, action=argparse.BooleanOptionalAction,
                        help="Resize crops to out-size (zooms in/out to match out-size). adds padding if necessary")
    output.add_argument('--preserve-crops', default=True, action=argparse.BooleanOptionalAction,
                        help="Keeps scale/resolution of cropped scenes. adds padding if necessary")
    output.add_argument('--flat', default=True, action=argparse.BooleanOptionalAction,
                        help="Store frames of all videos in output root directory instead of using sub directories per clip")
    output.add_argument('--save-as-images', default=False, action=argparse.BooleanOptionalAction,
                        help="Save as images instead of videos")

    nsfw_detection = parser.add_argument_group('NSFW detection')
    nsfw_detection.add_argument('--model', type=str, default="model_weights/lada_nsfw_detection_model_v1.3.pt",
                        help="path to NSFW detection model")
    nsfw_detection.add_argument('--model-device', type=str, default="cuda", help="device to run the YOLO model on. E.g. 'cuda' or 'cuda:0'")
    nsfw_detection.add_argument('--detection-start-confidence', type=float, default=0.6,
                        help="minimum YOLO confidence for a probe or a new scene to start (default: 0.6)")
    nsfw_detection.add_argument('--detection-continue-confidence', type=float, default=0.25,
                        help="minimum YOLO confidence for an already tracked scene to continue (default: 0.25)")

    scene_duration_filter = parser.add_argument_group('Scene duration filter')
    scene_duration_filter.add_argument('--scene-min-frames', type=int, default=24,
                        help="minimum number of frames required for a scene (default: 24, matching the training window)")
    scene_duration_filter.add_argument('--scene-min-length', type=float, default=0,
                        help="optional additional minimum scene duration in seconds (default: disabled)")
    scene_duration_filter.add_argument('--scene-max-length', type=float, default=8,
                        help="maximum length of a scene in number of frames. Scenes longer than that will be cut (in seconds)")
    scene_duration_filter.add_argument('--scene-max-memory', default=6144, type=int, help="limits maximum scene length based on approximate memory consumption of the scene. Value should be given in Megabytes (MB)")
    scene_duration_filter.add_argument('--scene-continuity-iou', type=float, default=0.2,
                        help="continue a scene across tracker-ID changes when adjacent boxes have at least this IoU (default: 0.2)")
    scene_duration_filter.add_argument('--scene-gap-frames', type=int, default=3,
                        help="bridge this many consecutive missed detections by interpolating boxes and masks (default: 3)")

    video_quality_evaluation = parser.add_argument_group('Scene video quality evaluation')
    video_quality_evaluation.add_argument('--add-video-quality-metadata', default=True, action=argparse.BooleanOptionalAction, help="If enabled will evaluate video quality and add its results to metadata")
    video_quality_evaluation.add_argument('--enable-video-quality-filter', default=False, action=argparse.BooleanOptionalAction, help="If enabled and scene quality is below scene-min-quality it will be skipped and not land in the dataset.")
    video_quality_evaluation.add_argument('--video-quality-model-device', type=str, default="cuda", help="device to run the video quality model on. E.g. 'cuda' or 'cuda:0'")
    video_quality_evaluation.add_argument('--min-video-quality', type=float, default=0.1,
                        help="minimum quality of a scene as determined by quality estimation model DOVER. Range between 0 and 1 were 1 is highest quality. If scene quality is below this threshold it will be skipped and not land in the dataset.")

    mosaic_creation = parser.add_argument_group('Mosaic creation')
    mosaic_creation.add_argument('--save-mosaic', default=False, action=argparse.BooleanOptionalAction,
                        help="Create and save mosaic images and masks")
    mosaic_creation.add_argument('--degrade-mosaic', default=False, action=argparse.BooleanOptionalAction,
                        help="degrades mosaic and NSFW video clips to better match real world video sources (e.g. video compression artifacts)")

    watermark_detection = parser.add_argument_group('Watermark detection')
    watermark_detection.add_argument('--add-watermark-metadata', default=True, action=argparse.BooleanOptionalAction, help="If enabled will run watermark detection and add its results to metadata")
    watermark_detection.add_argument('--enable-watermark-filter', default=False, action=argparse.BooleanOptionalAction, help="If enabled, scenes obstructed by watermarks (arbitrary text or logos) will be skipped")
    watermark_detection.add_argument('--watermark-model-path', type=str, default="model_weights/lada_watermark_detection_model_v1.3.pt",
                        help="path to watermark detection model")

    nsfw_detection = parser.add_argument_group('NudeNet NSFW detection')
    nsfw_detection.add_argument('--add-nudenet-nsfw-metadata', default=True, action=argparse.BooleanOptionalAction, help="If enabled will run NudeNet NSFW detection and add its results to metadata")
    nsfw_detection.add_argument('--enable-nudenet-nsfw-filter', default=False, action=argparse.BooleanOptionalAction, help="If enabled, scenes which aren't also classified by NudeNet as NSFW will be skipped")
    nsfw_detection.add_argument('--nudenet-nsfw-model-path', type=str, default="model_weights/3rd_party/640m.pt",
                        help="path to NudeNet NSFW detection model")

    censor_detection = parser.add_argument_group('Censor detection (Currently, we just reuse the mosaic detection model so no other censoring methods like blur or black bars will be detected)')
    censor_detection.add_argument('--add-censor-metadata', default=True, action=argparse.BooleanOptionalAction, help="If enabled will run Censor detection and add its results to metadata")
    censor_detection.add_argument('--enable-censor-filter', default=False, action=argparse.BooleanOptionalAction, help="If enabled, scenes which are classified as censored will be skipped")
    censor_detection.add_argument('--censor-model-path', type=str, default="model_weights/lada_mosaic_detection_model_v2.pt",
                        help="path to censor detection model")

    args = parser.parse_args(argv)
    if not 0 <= args.scene_continuity_iou <= 1:
        parser.error("--scene-continuity-iou must be between 0 and 1")
    if not 0 <= args.detection_continue_confidence <= args.detection_start_confidence <= 1:
        parser.error("detection confidence values must satisfy 0 <= continue <= start <= 1")
    if args.scene_min_frames <= 0:
        parser.error("--scene-min-frames must be greater than 0")
    if args.scene_min_length < 0:
        parser.error("--scene-min-length must be zero or greater")
    if args.scene_max_length <= 0:
        parser.error("--scene-max-length must be greater than 0")
    if args.scene_gap_frames < 0:
        parser.error("--scene-gap-frames must be zero or greater")
    if args.stride_length < 0:
        parser.error("--stride-length must be zero or greater")
    if args.min_crop_size < 0:
        parser.error("--min-crop-size must be zero or greater")
    if args.save_cropped and args.resize_crops and args.out_size < args.min_crop_size:
        parser.error("--out-size must be at least --min-crop-size when --resize-crops is enabled")
    return args


def collect_video_files(input_paths: list[Path]) -> list[Path]:
    """Collect video files recursively from one or more inputs without duplicates."""
    files_by_path: dict[str, Path] = {}
    for input_path in input_paths:
        candidates = input_path.rglob("*") if input_path.is_dir() else [input_path]
        for candidate in candidates:
            if not candidate.is_file() or not video_utils.is_video_file(candidate):
                continue
            absolute_path = candidate.absolute()
            files_by_path[str(absolute_path)] = absolute_path
    return [files_by_path[key] for key in sorted(files_by_path)]

def main():
    args = parse_args()

    if not (args.save_cropped or args.save_uncropped):
        print("No save option given. Specify at least one!")
        return

    scenes_executor = concurrent_futures.ThreadPoolExecutor(max_workers=args.workers)

    nsfw_detection_model = Yolo(args.model)

    video_quality_evaluator = None
    if args.add_video_quality_metadata or args.enable_video_quality_filter:
        from lada.models.dover.evaluate import VideoQualityEvaluator
        video_quality_evaluator = VideoQualityEvaluator(device=args.video_quality_model_device)
    watermark_detector = WatermarkDetector(Yolo(args.watermark_model_path), device=args.model_device) if args.add_watermark_metadata or args.enable_watermark_filter else None
    nudenet_nsfw_detector = NudeNetNsfwDetector(Yolo(args.nudenet_nsfw_model_path), device=args.model_device) if args.add_nudenet_nsfw_metadata or args.enable_nudenet_nsfw_filter else None
    censor_detector = MosaicDetector(Yolo(args.censor_model_path), device=args.model_device) if args.add_censor_metadata or args.enable_censor_filter else None

    output_dir = args.output_root
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    input_paths = args.input
    video_files = collect_video_files(input_paths)
    if not video_files:
        raise FileNotFoundError(f"No supported video files found in: {', '.join(map(str, input_paths))}")
    print(f"Found {len(video_files)} input video files")

    file_queue = queue.Queue()
    file_processing_options = FileProcessingOptions(input_dir=tuple(input_paths),
                                                    output_dir=output_dir,
                                                    start_index=args.start_index,
                                                    stride_length=args.stride_length,
                                                    scene_max_length=args.scene_max_length,
                                                    scene_max_memory=args.scene_max_memory,
                                                    scene_min_length=args.scene_min_length,
                                                    scene_min_frames=args.scene_min_frames,
                                                    random_extend_masks=True,
                                                    skip4k=args.skip_4k,
                                                    scene_continuity_iou=args.scene_continuity_iou,
                                                    scene_gap_frames=args.scene_gap_frames,
                                                    probe_min_crop_size=args.min_crop_size,
                                                    probe_crop_target_size=args.out_size,
                                                    detection_start_confidence=args.detection_start_confidence,
                                                    detection_continue_confidence=args.detection_continue_confidence)

    scene_processing_options = SceneProcessingOptions(output_dir=output_dir,
                                                  save_flat=args.flat,
                                                  out_size=args.out_size,
                                                  min_crop_size=args.min_crop_size,
                                                  save_cropped=args.save_cropped,
                                                  save_uncropped=args.save_uncropped,
                                                  resize_crops=args.resize_crops,
                                                  preserve_crops=args.preserve_crops,
                                                  save_mosaic=args.save_mosaic,
                                                  degrade_mosaic=args.degrade_mosaic,
                                                  save_as_images=args.save_as_images,
                                                  quality_evaluation=SceneProcessingOptions.VideoQualityProcessingOptions(args.enable_video_quality_filter, args.add_video_quality_metadata, args.min_video_quality),
                                                  watermark_detection=SceneProcessingOptions.WatermarkDetectionProcessingOptions(args.enable_watermark_filter, args.add_watermark_metadata),
                                                  nudenet_nsfw_detection=SceneProcessingOptions.NudeNetNsfwDetectionProcessingOptions(args.enable_nudenet_nsfw_filter, args.add_nudenet_nsfw_metadata),
                                                  censor_detection = SceneProcessingOptions.CensorDetectionProcessingOptions(args.enable_censor_filter, args.add_censor_metadata))

    nsfw_detector = NsfwDetector(nsfw_detection_model=nsfw_detection_model, device=args.model_device,
                                 file_queue=file_queue,
                                 frame_queue=queue.Queue(50),
                                 scene_queue=queue.Queue(2),
                                 file_processing_options=file_processing_options)
    scene_processor = SceneProcessor(video_quality_evaluator, watermark_detector, nudenet_nsfw_detector, censor_detector)

    try:
        nsfw_detector.start()
        nsfw_detector.add_files(video_files)
        scene_futures = []
        for scene in nsfw_detector():
            print(f"Found scene {scene.id} (frames {scene.frame_start:06d}-{scene.frame_end:06d}, lengths {scene.frame_end-scene.frame_start+1}/{len(scene)}), queuing up for processing")
            scene_futures.append(scenes_executor.submit(scene_processor.process_scene, scene, output_dir, scene_processing_options))
            while len([future for future in scene_futures if not future.done()]) >= args.workers + 1:
                # print(f"workers busy, block until they are available: running {len([future for future in scene_futures if future.running()])}, lets get to work: {len([future for future in scene_futures if not future.done()])}")
                sleep(1)
                pass  # do nothing until workers become available. Otherwise, we could queue up a lot of futures which use a lot of memory as we pass Scene objects
            # we don't care about done futures, lets clean them up to potentially free memory
            clean_up_completed_futures(scene_futures)
            # print(f"deleted done future. futures now {len(scene_futures)}")
        clean_up_completed_futures(scene_futures)
        wait_until_completed(scene_futures)
    finally:
        nsfw_detector.stop()

    scenes_executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
