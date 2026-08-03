import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lada.datasetcreation.nsfw_scene_processor import cropped_scene_exceeds_minimum_size


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "dataset_creation"
    / "create-mosaic-restoration-dataset.py"
)


def load_dataset_script():
    spec = importlib.util.spec_from_file_location("create_mosaic_restoration_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeCroppedScene:
    def __init__(self, shapes):
        self._images = [np.zeros(shape, dtype=np.uint8) for shape in shapes]

    def get_images(self):
        return self._images


def test_every_crop_needs_at_least_one_dimension_of_384():
    assert cropped_scene_exceeds_minimum_size(
        FakeCroppedScene([(384, 384, 3), (512, 640, 3)]), 384
    )
    assert cropped_scene_exceeds_minimum_size(
        FakeCroppedScene([(384, 512, 3)]), 384
    )
    assert cropped_scene_exceeds_minimum_size(
        FakeCroppedScene([(512, 384, 3)]), 384
    )
    assert cropped_scene_exceeds_minimum_size(
        FakeCroppedScene([(200, 384, 3)]), 384
    )
    assert cropped_scene_exceeds_minimum_size(
        FakeCroppedScene([(384, 200, 3)]), 384
    )
    assert not cropped_scene_exceeds_minimum_size(
        FakeCroppedScene([(383, 383, 3)]), 384
    )


def test_any_small_frame_rejects_the_whole_scene():
    assert not cropped_scene_exceeds_minimum_size(
        FakeCroppedScene([(640, 640, 3), (383, 300, 3)]), 384
    )


def test_zero_disables_the_minimum_crop_filter():
    assert cropped_scene_exceeds_minimum_size(FakeCroppedScene([(64, 64, 3)]), 0)


def test_dataset_script_defaults_to_192_and_hysteresis_confidence():
    script = load_dataset_script()
    args = script.parse_args(["--input", "input.mp4"])
    assert args.min_crop_size == 192
    assert args.detection_start_confidence == 0.6
    assert args.detection_continue_confidence == 0.25
    assert args.scene_min_frames == 24
    assert args.scene_min_length == 0
    assert args.scene_max_length == 8
    assert args.scene_gap_frames == 3


def test_resized_output_must_also_exceed_the_crop_minimum():
    script = load_dataset_script()
    args = script.parse_args(["--input", "input.mp4", "--resize-crops"])
    assert args.out_size == 256

    with pytest.raises(SystemExit):
        script.parse_args(
            ["--input", "input.mp4", "--resize-crops", "--min-crop-size", "384"]
        )

    args = script.parse_args(
        ["--input", "input.mp4", "--resize-crops", "--out-size", "384"]
    )
    assert args.out_size == 384


def test_continue_confidence_cannot_exceed_start_confidence():
    script = load_dataset_script()
    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--input",
                "input.mp4",
                "--detection-start-confidence",
                "0.5",
                "--detection-continue-confidence",
                "0.6",
            ]
        )


def test_video_collection_ignores_appledouble_and_partial_files(tmp_path):
    script = load_dataset_script()
    complete = tmp_path / "clip.mkv"
    appledouble = tmp_path / "._clip.mkv"
    partial = tmp_path / ".clip.part.mkv"
    for path in (complete, appledouble, partial):
        path.touch()

    assert script.collect_video_files([tmp_path]) == [complete.resolve()]
